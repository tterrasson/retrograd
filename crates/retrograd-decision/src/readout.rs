//! Where a decision model reads the score of an option.
//!
//! The readout is chosen by the `<arch>.decision.type` metadata and needs the
//! vocabulary for the token ids it reads at: the mask and separator of a
//! `laya`, one letter per option for an `openjev`, the end token of an option
//! for a `kev`. The vocabulary is a trait, so the crate carries no tokenizer.

use crate::request::{Question, QuestionKind};
use crate::{DecisionError, Token};

/// What `lev` reads a `noul` question from: a rating scale of 9 labels, 0 for
/// certainly no, 8 for certainly yes.
const LEV_N_RATINGS: usize = 9;

/// Options a question may have at most, whatever the readout.
pub(crate) const N_OPTIONS_MAX: usize = 255;

/// The letters `openjev` labels its options with, one token each.
const OPENJEV_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The text of the token `kev` reads an option at.
const KEV_OPTION_END: &str = "<|box_end|>";

/// What a readout needs to know about the tokenizer of the model.
pub trait Vocabulary {
    /// Tokenizes `text` without adding the special tokens of the model. With
    /// `parse_special`, the text of a special token in `text` is that token.
    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<Token>;

    fn mask_token(&self) -> Option<Token>;

    fn sep_token(&self) -> Option<Token>;

    /// The text of a token, special tokens included.
    fn token_text(&self, token: Token) -> String;
}

/// How the labels of a [`Readout::LabelLogits`] are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelFamily {
    /// `openjev`: one letter per option, not shown to the template. A `noul`
    /// question lists `true` first. Images are part of the prompt.
    Letters,
    /// `lev`: codes `A..Z` then `AA..ZZ`, shown to the template. The keys of
    /// the context are sorted, the temperature buckets are by size.
    Codes,
}

/// Where the score of each option is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readout {
    /// `laya`: the embeddings output at the mask token that opens each option,
    /// at the column of the question type.
    MarkerColumn {
        marker: Token,
        sep: Token,
        /// The text of `marker`, replaced in the inputs before rendering so an
        /// input cannot forge an option.
        marker_text: String,
    },
    /// `openjev`, `lev`: the logits of one label token per option, at the last
    /// token of the prompt.
    LabelLogits {
        labels: Vec<Token>,
        /// The text of each label, given to the template; empty when the
        /// template does not show labels.
        label_texts: Vec<String>,
        /// Prompts evaluated for a `choice` question, each with the options in
        /// another order; the answer averages them.
        n_variants: usize,
        /// A `noul` question is read from this many labels, the answer is the
        /// expected rating; `None` reads the `true` option.
        noul_scale: Option<usize>,
        family: LabelFamily,
    },
    /// `kev`: the hidden state `[q|k]` at the last token and at the end token of
    /// each option, scored by [`pointer_score`].
    Pointer { marker: Token },
}

impl Readout {
    /// Builds the readout of a `<arch>.decision.type` value; `None` means the
    /// model has no such key.
    pub fn from_decision_type(
        decision_type: Option<&str>,
        vocab: &impl Vocabulary,
    ) -> Result<Self, DecisionError> {
        match decision_type {
            None => Err(DecisionError::NotDecisionModel),
            Some("laya") => {
                let (Some(marker), Some(sep)) = (vocab.mask_token(), vocab.sep_token()) else {
                    return Err(DecisionError::InvalidModel(
                        "decision model has no mask or sep token".into(),
                    ));
                };
                Ok(Self::MarkerColumn {
                    marker,
                    sep,
                    marker_text: vocab.token_text(marker),
                })
            }
            Some("openjev") => {
                let mut labels = Vec::new();
                for letter in OPENJEV_LETTERS.chars() {
                    let tokens = vocab.tokenize(&letter.to_string(), false);
                    let [label] = tokens[..] else {
                        return Err(DecisionError::InvalidModel(format!(
                            "decision label '{letter}' is not a single token"
                        )));
                    };
                    labels.push(label);
                }
                Ok(Self::LabelLogits {
                    labels,
                    label_texts: Vec::new(),
                    n_variants: 1,
                    noul_scale: None,
                    family: LabelFamily::Letters,
                })
            }
            Some("lev") => {
                let mut labels = Vec::new();
                let mut label_texts = Vec::new();
                // Only the codes that are a single token are used.
                for code in lev_codes() {
                    let tokens = vocab.tokenize(&code, false);
                    if let [label] = tokens[..]
                        && labels.len() < N_OPTIONS_MAX
                    {
                        labels.push(label);
                        label_texts.push(code);
                    }
                }
                if labels.len() < LEV_N_RATINGS {
                    return Err(DecisionError::InvalidModel(format!(
                        "decision model has {} single-token labels, the rating scale needs {LEV_N_RATINGS}",
                        labels.len()
                    )));
                }
                Ok(Self::LabelLogits {
                    labels,
                    label_texts,
                    n_variants: 2,
                    noul_scale: Some(LEV_N_RATINGS),
                    family: LabelFamily::Codes,
                })
            }
            Some("kev") => {
                let tokens = vocab.tokenize(KEV_OPTION_END, true);
                let [marker] = tokens[..] else {
                    return Err(DecisionError::InvalidModel(format!(
                        "decision model has no {KEV_OPTION_END} token"
                    )));
                };
                Ok(Self::Pointer { marker })
            }
            Some(name) => Err(DecisionError::UnknownType { name: name.into() }),
        }
    }

    /// The most options a question may have.
    pub fn n_options_max(&self) -> usize {
        match self {
            Self::LabelLogits { labels, .. } => labels.len(),
            Self::MarkerColumn { .. } | Self::Pointer { .. } => N_OPTIONS_MAX,
        }
    }

    /// Whether a `noul` question lists `true` before `false`.
    pub fn noul_true_first(&self) -> bool {
        matches!(
            self,
            Self::LabelLogits {
                family: LabelFamily::Letters,
                ..
            }
        )
    }

    /// Whether the objects of the context are written with sorted keys.
    pub fn sorts_keys(&self) -> bool {
        matches!(
            self,
            Self::LabelLogits {
                family: LabelFamily::Codes,
                ..
            }
        )
    }

    /// Whether the prompt of the model has a place for images.
    pub fn accepts_images(&self) -> bool {
        matches!(
            self,
            Self::LabelLogits {
                family: LabelFamily::Letters,
                ..
            }
        )
    }

    /// The text that must not appear in an input, if the layout depends on it.
    pub fn marker_text(&self) -> Option<&str> {
        match self {
            Self::MarkerColumn { marker_text, .. } if !marker_text.is_empty() => Some(marker_text),
            _ => None,
        }
    }

    /// Prompts evaluated to answer `question`, the options in another order in
    /// each.
    pub fn n_variants(&self, question: &Question) -> usize {
        match self {
            Self::LabelLogits { n_variants, .. }
                if question.kind == QuestionKind::Choice && question.options.len() > 1 =>
            {
                *n_variants
            }
            _ => 1,
        }
    }

    /// Raw scores each variant of `question` returns.
    pub fn n_outputs(&self, question: &Question) -> usize {
        match self {
            Self::LabelLogits {
                noul_scale: Some(scale),
                ..
            } if question.kind == QuestionKind::Noul => *scale,
            _ => question.options.len(),
        }
    }

    /// The tokens whose logits are the scores of `question`, for a label
    /// readout.
    pub fn label_tokens(&self, question: &Question) -> Option<&[Token]> {
        match self {
            Self::LabelLogits { labels, .. } => labels.get(..self.n_outputs(question)),
            _ => None,
        }
    }

    /// The bucket of the temperature that depends on the number of options.
    pub fn temperature_bucket(&self, n_options: usize) -> &'static str {
        if self.sorts_keys() {
            match n_options {
                0..=8 => "small",
                9..=26 => "mid",
                _ => "large",
            }
        } else {
            match n_options {
                0..=2 => "2",
                3..=5 => "3_5",
                6..=10 => "6_10",
                _ => "11",
            }
        }
    }
}

/// `A..Z` then `AA..ZZ`.
fn lev_codes() -> impl Iterator<Item = String> {
    let single = ('A'..='Z').map(String::from);
    let double = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    single.chain(double)
}

/// The score of an option for a [`Readout::Pointer`]: the dot product of the
/// first half of `query` (the last token) and the second half of `key` (the end
/// token of the option), over the square root of the half width. The sums are
/// in `f32`, in order, like the server's.
pub fn pointer_score(query: &[f32], key: &[f32]) -> Result<f32, DecisionError> {
    if query.len() != key.len() || query.is_empty() || !query.len().is_multiple_of(2) {
        return Err(DecisionError::Evaluation(format!(
            "pointer embeddings must have the same even width, got {} and {}",
            query.len(),
            key.len()
        )));
    }
    let half = query.len() / 2;
    let dot = query[..half]
        .iter()
        .zip(&key[half..])
        .fold(0.0_f32, |sum, (q, k)| sum + q * k);
    // Widening: `half` is a model width, far below 2^24 where an f32 stops
    // holding integers exactly.
    Ok(dot / (half as f32).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary where every one-letter text is one token and the
    /// two-letter codes are one token only when their first letter is before
    /// `last_pair_letter`.
    struct Fake {
        pair_below: char,
    }

    impl Vocabulary for Fake {
        fn tokenize(&self, text: &str, parse_special: bool) -> Vec<Token> {
            if text == KEV_OPTION_END {
                return if parse_special {
                    vec![999]
                } else {
                    vec![1, 2, 3]
                };
            }
            let chars: Vec<char> = text.chars().collect();
            match chars[..] {
                [c] => vec![i32::try_from(u32::from(c)).expect("ascii")],
                [a, b] if a < self.pair_below => {
                    vec![1000 + i32::try_from(u32::from(a) * 100 + u32::from(b)).expect("small")]
                }
                _ => vec![1, 2],
            }
        }

        fn mask_token(&self) -> Option<Token> {
            Some(4)
        }

        fn sep_token(&self) -> Option<Token> {
            Some(1)
        }

        fn token_text(&self, token: Token) -> String {
            if token == 4 {
                "<mask>".into()
            } else {
                "<eos>".into()
            }
        }
    }

    #[test]
    fn missing_key_and_unknown_type_are_distinct_errors() {
        let vocab = Fake { pair_below: 'A' };
        assert!(matches!(
            Readout::from_decision_type(None, &vocab),
            Err(DecisionError::NotDecisionModel)
        ));
        match Readout::from_decision_type(Some("nimble"), &vocab) {
            Err(DecisionError::UnknownType { name }) => assert_eq!(name, "nimble"),
            other => panic!("expected UnknownType, got {other:?}"),
        }
    }

    #[test]
    fn laya_keeps_the_marker_text_for_sanitising() {
        let readout =
            Readout::from_decision_type(Some("laya"), &Fake { pair_below: 'A' }).expect("laya");
        assert_eq!(readout.marker_text(), Some("<mask>"));
        assert_eq!(readout.n_options_max(), 255);
        assert!(!readout.noul_true_first());
    }

    #[test]
    fn lev_keeps_only_single_token_codes() {
        // Pairs `AA..CZ` are one token: 26 + 3 * 26 labels.
        let readout =
            Readout::from_decision_type(Some("lev"), &Fake { pair_below: 'D' }).expect("lev");
        let Readout::LabelLogits {
            labels,
            label_texts,
            n_variants,
            noul_scale,
            ..
        } = &readout
        else {
            panic!("lev is a label readout");
        };
        assert_eq!(labels.len(), 26 + 3 * 26);
        assert_eq!(label_texts[26], "AA");
        assert_eq!(label_texts.last().map(String::as_str), Some("CZ"));
        assert_eq!((*n_variants, *noul_scale), (2, Some(9)));
        assert_eq!(readout.n_options_max(), 104);
        assert_eq!(readout.temperature_bucket(8), "small");
        assert_eq!(readout.temperature_bucket(9), "mid");
        assert_eq!(readout.temperature_bucket(27), "large");
    }

    #[test]
    fn lev_labels_are_capped_at_255() {
        let readout =
            Readout::from_decision_type(Some("lev"), &Fake { pair_below: '[' }).expect("lev");
        assert_eq!(readout.n_options_max(), 255);
    }

    #[test]
    fn lev_refuses_a_vocabulary_without_a_rating_scale() {
        struct Poor;
        impl Vocabulary for Poor {
            fn tokenize(&self, text: &str, _: bool) -> Vec<Token> {
                if text.len() == 1 && text < "H" {
                    vec![1]
                } else {
                    vec![1, 2]
                }
            }
            fn mask_token(&self) -> Option<Token> {
                None
            }
            fn sep_token(&self) -> Option<Token> {
                None
            }
            fn token_text(&self, _: Token) -> String {
                String::new()
            }
        }
        // 'A'..='G' is 7 labels, under the 9 ratings.
        assert!(matches!(
            Readout::from_decision_type(Some("lev"), &Poor),
            Err(DecisionError::InvalidModel(_))
        ));
        assert!(matches!(
            Readout::from_decision_type(Some("laya"), &Poor),
            Err(DecisionError::InvalidModel(_))
        ));
    }

    #[test]
    fn openjev_labels_are_the_52_letters() {
        let readout = Readout::from_decision_type(Some("openjev"), &Fake { pair_below: 'A' })
            .expect("openjev");
        assert_eq!(readout.n_options_max(), 52);
        assert!(readout.noul_true_first());
        assert!(readout.accepts_images());
        assert!(!readout.sorts_keys());
    }

    #[test]
    fn kev_needs_a_single_special_end_token() {
        let readout =
            Readout::from_decision_type(Some("kev"), &Fake { pair_below: 'A' }).expect("kev");
        assert_eq!(readout, Readout::Pointer { marker: 999 });
    }

    #[test]
    fn numeric_buckets_match_the_fitting_ranges() {
        let readout =
            Readout::from_decision_type(Some("laya"), &Fake { pair_below: 'A' }).expect("laya");
        let buckets: Vec<_> = [2, 3, 5, 6, 10, 11, 255]
            .into_iter()
            .map(|n| readout.temperature_bucket(n))
            .collect();
        assert_eq!(buckets, ["2", "3_5", "3_5", "6_10", "6_10", "11", "11"]);
    }

    #[test]
    fn pointer_score_reads_query_head_and_key_tail() {
        // half = 2: q = [1, 2], k = [30, 40] -> (1*30 + 2*40) / sqrt(2)
        let score = pointer_score(&[1.0, 2.0, 9.0, 9.0], &[9.0, 9.0, 3.0, 4.0]).expect("score");
        assert!((score - 11.0 / 2.0_f32.sqrt()).abs() < 1e-6);
        assert!(pointer_score(&[1.0, 2.0], &[1.0]).is_err());
        assert!(pointer_score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).is_err());
    }
}
