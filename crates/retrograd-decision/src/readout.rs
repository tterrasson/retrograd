//! Where a decision model reads the score of an option, and everything else
//! its `<arch>.decision.type` decides.
//!
//! Each type llama.cpp knows is one row of [`DECISION_TYPES`]: its labels,
//! the order of a `noul` question, the checks of a request, how the inputs
//! are written for the template, the confidence and the temperature buckets.
//! A behaviour that needs code is written once, as a variant of one of the
//! enums below, and a row names it; nothing branches on a type name.
//!
//! The readout needs the vocabulary for the token ids it reads at: the mask
//! and separator of a `laya`, one letter per option for an `openjev`, the end
//! token of an option for a `kev`. The vocabulary is a trait, so the crate
//! carries no tokenizer.

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

/// The classes of a pooled classifier: inconsistent, consistent.
const POOLED_N_CLASSES: usize = 2;

/// A token of the vocabulary whose text the inputs may spell: a control or a
/// user-defined token.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct SpecialToken {
    /// The text of the token as the vocabulary stores it.
    pub text: String,
    pub control: bool,
    pub user_defined: bool,
}

/// What a readout needs to know about the tokenizer of the model.
pub trait Vocabulary {
    /// Tokenizes `text` without adding the special tokens of the model. With
    /// `parse_special`, the text of a special token in `text` is that token.
    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<Token>;

    fn mask_token(&self) -> Option<Token>;

    fn sep_token(&self) -> Option<Token>;

    /// The text of a token, special tokens included.
    fn token_text(&self, token: Token) -> String;

    /// The bytes of a token, special tokens included: a piece of a multi-byte
    /// character stays a piece.
    fn token_bytes(&self, token: Token) -> Vec<u8> {
        self.token_text(token).into_bytes()
    }

    /// The control and user-defined tokens, in id order.
    fn special_tokens(&self) -> Vec<SpecialToken>;
}

/// What a readout reads from the model file besides its vocabulary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeadMetadata {
    /// `<arch>.decision.labels`, in order.
    pub labels: Vec<String>,
    /// The classifier labels of the model.
    pub n_classes: usize,
    /// The width of the embeddings output.
    pub n_embd_out: i64,
}

/// Where the scores of a type are read, and with which labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// The embeddings output at the mask token that opens each option, in
    /// the column of the question type; `[cls] question [sep] ([mask]
    /// option)* [sep] state [sep]`, cut to the head budget.
    Marker,
    /// The same column read, the prompt cut piece by piece to the budgets
    /// of the d1-omni layout ([`crate::layout::d1omni`]).
    D1Omni,
    /// The logits of one letter per option, the letters not shown.
    Letters,
    /// The logits of one code per option, `A..Z` then `AA..ZZ`, shown.
    Codes,
    /// The logits of the labels of `<arch>.decision.labels`, shown.
    MetadataLabels,
    /// The largest logit of a group of label tokens per option, the labels
    /// chosen per question ([`Readout::question_labels`]).
    LabelGroups(D1Codes),
    /// The hidden state `[q|k]` at the last token and at the end token of
    /// each option, scored by [`pointer_score`].
    Pointer,
    /// A head that reads every question of its prompt from spans marked in
    /// the batch; the score of option `i` is row `i` of the output.
    Joint,
    /// One score per class, the pooled embeddings output.
    Pooled,
}

/// The codes of a choice of more than 26 options whose keys are not letters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D1Codes {
    /// `00`, `01`, ...
    Digits,
    /// `A..Z`, then `AA`, `AB`, ...
    Letters,
}

/// The order of the two options of a `noul` question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoulOrder {
    FalseFirst,
    TrueFirst,
    /// `[false, true]`, unless the criteria describe exactly both and name
    /// `true` first.
    CriteriaOrder,
}

/// What a request may say about the instructions of a question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Instructions {
    Required,
    /// May be null: the prompt has a default question.
    Optional,
    /// May be left out, read as empty text: the template has a default.
    DefaultEmpty,
    /// Must be left out: the task is part of the prompt.
    Absent,
}

/// The questions a type answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionShape {
    /// A `type` is required, a `score` has 2 to 10 levels.
    Standard,
    /// The `type` defaults to `choice`, a `score` has from one level to as
    /// many as the model has labels.
    Lenient,
    /// One fixed binary question: `choice` over `{"0": null, "1": null}` or a
    /// `noul` without criteria, about a state that is the OCR text of two
    /// images.
    OcrConsistency,
}

/// What spoken instructions (an `input_audio` part in place of the text)
/// mean to a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spoken {
    /// Plain JSON, the prompt is not built by the question template.
    Plain,
    /// An audio clip heard where the text would be: media.
    Media,
    /// Refused: the model has no audio input.
    Refused,
}

/// How the inputs are written before the template sees them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputText {
    /// Raw JSON values.
    Raw,
    /// The state, the instructions, the keys and the descriptions flattened to
    /// text, special tokens escaped.
    Kev,
    /// The texts of the control tokens broken by a zero-width space, in keys
    /// too.
    D1Break,
    /// Special tokens escaped in keys and values, and the d1-omni markers.
    D1OmniEscape,
    /// Texts rendered as the reference Python shim renders them, special
    /// token texts quoted with JSON escapes ([`crate::template`]).
    H2o,
    /// The state is the OCR text, its special tokens escaped.
    OcrEscape,
}

/// Which questions one prompt holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptScope {
    /// One question.
    Question,
    /// One question, and the list of all the questions of the request.
    AllQuestions,
    /// Every question of the request, one score per option.
    Joint,
    /// One question, through the joint template.
    JointQuestion,
}

/// The confidence of a `choice` or `score` answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    /// The formulas published by TypeSafe.
    Published,
    /// One minus the normalized entropy, as the reference of the vllm heads.
    Entropy,
    /// The probability of the answer (d1 `api.py`).
    MaxProbability,
}

/// The `score` of a `score` answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreReading {
    /// The expected level.
    Expectation,
    /// The first level whose cumulative probability reaches one half (d1
    /// `api.py`).
    Median,
}

/// The buckets of the temperatures fitted by number of options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buckets {
    /// `2`, `3_5`, `6_10`, `11`.
    Count,
    /// `small`, `mid`, `large`.
    Size,
}

/// One `<arch>.decision.type`: everything that type decides.
#[derive(Debug, PartialEq, Eq)]
pub struct DecisionType {
    pub name: &'static str,
    pub head: HeadKind,
    pub noul_order: NoulOrder,
    /// The options of a `choice` are in the order of their keys.
    pub choice_sorted: bool,
    /// The objects of the context are written with sorted keys.
    pub sorts_keys: bool,
    /// The state, the instructions and the descriptions must be text, an
    /// object or an array.
    pub strict_payload: bool,
    /// A legend entry that is not text is returned as canonical JSON text.
    pub canonical_legend: bool,
    /// A null state is accepted, to ask about media only.
    pub null_state: bool,
    pub instructions: Instructions,
    pub questions: QuestionShape,
    /// A `noul` question also reads its descriptions under `no` and `yes`.
    pub noul_aliases: bool,
    pub spoken: Spoken,
    /// The `video_url` parts of chat messages are media.
    pub video_parts: bool,
    /// The prompt has a place for images.
    pub media: bool,
    pub input: InputText,
    pub prompt: PromptScope,
    pub confidence: Confidence,
    pub score: ScoreReading,
    pub buckets: Buckets,
    /// Prompts evaluated for a `choice` question of several options, each
    /// with the options in another order.
    pub choice_variants: usize,
    /// A `noul` question is read from this many labels, the answer is the
    /// expected rating.
    pub noul_scale: Option<usize>,
    /// The answers of a request with media are not calibrated.
    pub media_uncalibrated: bool,
    /// The prompt must be the only sequence of its forward pass.
    pub alone_in_pass: bool,
    /// A training run may fit this readout.
    pub trainable: bool,
}

/// The row every type starts from: one question per prompt, standard checks,
/// raw inputs.
const BASE: DecisionType = DecisionType {
    name: "",
    head: HeadKind::Codes,
    noul_order: NoulOrder::FalseFirst,
    choice_sorted: false,
    sorts_keys: false,
    strict_payload: false,
    canonical_legend: false,
    null_state: false,
    instructions: Instructions::Required,
    questions: QuestionShape::Standard,
    noul_aliases: false,
    spoken: Spoken::Media,
    video_parts: false,
    media: false,
    input: InputText::Raw,
    prompt: PromptScope::Question,
    confidence: Confidence::Published,
    score: ScoreReading::Expectation,
    buckets: Buckets::Count,
    choice_variants: 1,
    noul_scale: None,
    media_uncalibrated: false,
    alone_in_pass: false,
    trainable: false,
};

/// The rows of the lfm3 d1 models, which differ in their audio input only.
const LFM3_D1: DecisionType = DecisionType {
    name: "lfm3-d1",
    head: HeadKind::LabelGroups(D1Codes::Letters),
    noul_order: NoulOrder::TrueFirst,
    null_state: true,
    spoken: Spoken::Refused,
    video_parts: true,
    media: true,
    input: InputText::D1Break,
    confidence: Confidence::MaxProbability,
    score: ScoreReading::Median,
    // The short convolutions have no memory to keep sequences apart.
    alone_in_pass: true,
    ..BASE
};

/// Every decision type of llama.cpp's decision server.
pub static DECISION_TYPES: [DecisionType; 16] = [
    DecisionType {
        name: "laya",
        head: HeadKind::Marker,
        trainable: true,
        ..BASE
    },
    DecisionType {
        name: "openjev",
        head: HeadKind::Letters,
        noul_order: NoulOrder::TrueFirst,
        media: true,
        trainable: true,
        ..BASE
    },
    DecisionType {
        name: "lev",
        head: HeadKind::Codes,
        // Trained with sorted keys.
        sorts_keys: true,
        buckets: Buckets::Size,
        // Two orders cancel the preference for the first label.
        choice_variants: 2,
        noul_scale: Some(LEV_N_RATINGS),
        trainable: true,
        ..BASE
    },
    DecisionType {
        name: "kev",
        head: HeadKind::Pointer,
        input: InputText::Kev,
        trainable: true,
        ..BASE
    },
    DecisionType {
        name: "nimble",
        head: HeadKind::Codes,
        prompt: PromptScope::AllQuestions,
        ..BASE
    },
    DecisionType {
        name: "clef",
        head: HeadKind::Joint,
        noul_order: NoulOrder::TrueFirst,
        choice_sorted: true,
        spoken: Spoken::Plain,
        media: true,
        prompt: PromptScope::Joint,
        alone_in_pass: true,
        ..BASE
    },
    DecisionType {
        name: "pplx-decider",
        head: HeadKind::Codes,
        media: true,
        ..BASE
    },
    DecisionType {
        name: "vllm-decision2",
        head: HeadKind::Joint,
        noul_order: NoulOrder::CriteriaOrder,
        strict_payload: true,
        canonical_legend: true,
        spoken: Spoken::Plain,
        prompt: PromptScope::JointQuestion,
        confidence: Confidence::Entropy,
        alone_in_pass: true,
        ..BASE
    },
    DecisionType {
        name: "vllm-d3",
        head: HeadKind::Codes,
        canonical_legend: true,
        null_state: true,
        instructions: Instructions::Optional,
        media: true,
        confidence: Confidence::Entropy,
        ..BASE
    },
    DecisionType {
        name: "lfm2-d1",
        head: HeadKind::LabelGroups(D1Codes::Digits),
        noul_order: NoulOrder::TrueFirst,
        null_state: true,
        media: true,
        ..BASE
    },
    DecisionType {
        name: "lfm2-d1-omni",
        head: HeadKind::D1Omni,
        null_state: true,
        noul_aliases: true,
        media: true,
        input: InputText::D1OmniEscape,
        media_uncalibrated: true,
        ..BASE
    },
    DecisionType {
        name: "granite-switch",
        head: HeadKind::Codes,
        noul_order: NoulOrder::TrueFirst,
        ..BASE
    },
    DecisionType {
        name: "h2o-lightning",
        head: HeadKind::MetadataLabels,
        noul_order: NoulOrder::TrueFirst,
        instructions: Instructions::DefaultEmpty,
        questions: QuestionShape::Lenient,
        noul_aliases: true,
        media: true,
        input: InputText::H2o,
        ..BASE
    },
    DecisionType {
        name: "openbmb-minicpm-ocr-jev",
        head: HeadKind::Pooled,
        choice_sorted: true,
        instructions: Instructions::Absent,
        questions: QuestionShape::OcrConsistency,
        media: true,
        input: InputText::OcrEscape,
        ..BASE
    },
    LFM3_D1,
    DecisionType {
        name: "lfm3-d1-omni",
        spoken: Spoken::Media,
        ..LFM3_D1
    },
];

impl DecisionType {
    /// The row of a `<arch>.decision.type` value.
    pub fn named(name: &str) -> Option<&'static Self> {
        DECISION_TYPES.iter().find(|row| row.name == name)
    }

    /// The control and user-defined token texts the inputs of this type keep
    /// as text, and the shortest text that counts: the lfm3 d1 models break
    /// control tokens of two bytes or more, h2o-lightning quotes both kinds.
    fn quoted(&self) -> Option<(bool, usize)> {
        match self.input {
            InputText::D1Break => Some((false, 2)),
            InputText::H2o => Some((true, 1)),
            _ => None,
        }
    }
}

/// Where the scores of a model are read, with the tokens that depend on its
/// vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Head {
    MarkerColumn {
        marker: Token,
        sep: Token,
        /// The text of `marker`, replaced in the inputs before rendering so an
        /// input cannot forge an option.
        marker_text: String,
    },
    D1Omni {
        marker: Token,
    },
    LabelLogits {
        labels: Vec<Token>,
        /// The text of each label, given to the template; empty when the
        /// template does not show labels.
        label_texts: Vec<String>,
    },
    LabelGroups,
    Pointer {
        marker: Token,
    },
    Joint,
    Pooled {
        n_classes: usize,
    },
}

/// The readout of a model: its type and what its vocabulary gives that type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readout {
    pub kind: &'static DecisionType,
    pub head: Head,
    /// The texts of the special tokens that are broken or quoted in the
    /// inputs, longest first.
    pub quoted: Vec<String>,
}

/// The labels of one question: what the template shows and what is read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuestionLabels {
    /// The label of each option, by position; empty when the template shows
    /// none.
    pub texts: Vec<String>,
    /// The tokens whose logits are read, a group of consecutive ones per
    /// score; empty for a readout that reads no logits.
    pub tokens: Vec<Token>,
    /// The tokens of each score, empty when each score is one token.
    pub groups: Vec<usize>,
}

impl Readout {
    /// A readout of `kind` with its head given: what a caller that installs
    /// a readout itself builds.
    pub fn new(kind: &'static DecisionType, head: Head) -> Self {
        Self {
            kind,
            head,
            quoted: Vec::new(),
        }
    }

    /// Builds the readout of a `<arch>.decision.type` value, with the checks
    /// llama-server makes when it loads the model; `None` means the model has
    /// no such key.
    pub fn from_decision_type(
        decision_type: Option<&str>,
        metadata: &HeadMetadata,
        vocab: &impl Vocabulary,
    ) -> Result<Self, DecisionError> {
        let Some(name) = decision_type else {
            return Err(DecisionError::NotDecisionModel);
        };
        let Some(kind) = DecisionType::named(name) else {
            return Err(DecisionError::UnknownType { name: name.into() });
        };
        let head = match kind.head {
            HeadKind::Marker => {
                let (Some(marker), Some(sep)) = (vocab.mask_token(), vocab.sep_token()) else {
                    return Err(DecisionError::InvalidModel(
                        "decision model has no mask or sep token".into(),
                    ));
                };
                Head::MarkerColumn {
                    marker,
                    sep,
                    marker_text: vocab.token_text(marker),
                }
            }
            HeadKind::D1Omni => {
                let Some(marker) = vocab.mask_token() else {
                    return Err(DecisionError::InvalidModel(
                        "decision model has no mask token".into(),
                    ));
                };
                Head::D1Omni { marker }
            }
            HeadKind::Letters => {
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
                Head::LabelLogits {
                    labels,
                    label_texts: Vec::new(),
                }
            }
            HeadKind::Codes => {
                let mut labels = Vec::new();
                let mut label_texts = Vec::new();
                // Only the codes that are a single token are used.
                for code in codes() {
                    let tokens = vocab.tokenize(&code, false);
                    if let [label] = tokens[..]
                        && labels.len() < N_OPTIONS_MAX
                    {
                        labels.push(label);
                        label_texts.push(code);
                    }
                }
                if let Some(scale) = kind.noul_scale
                    && labels.len() < scale
                {
                    return Err(DecisionError::InvalidModel(format!(
                        "decision model has {} single-token labels, the rating scale needs {scale}",
                        labels.len()
                    )));
                }
                Head::LabelLogits {
                    labels,
                    label_texts,
                }
            }
            HeadKind::MetadataLabels => {
                let mut labels = Vec::new();
                for text in &metadata.labels {
                    let tokens = vocab.tokenize(text, false);
                    match tokens[..] {
                        [label] if !labels.contains(&label) => labels.push(label),
                        _ => {
                            return Err(DecisionError::InvalidModel(format!(
                                "decision label \"{text}\" is not a single distinct token"
                            )));
                        }
                    }
                }
                if labels.is_empty() {
                    return Err(DecisionError::InvalidModel(
                        "decision model has no labels".into(),
                    ));
                }
                Head::LabelLogits {
                    labels,
                    label_texts: metadata.labels.clone(),
                }
            }
            HeadKind::LabelGroups(_) => Head::LabelGroups,
            HeadKind::Pointer => {
                let tokens = vocab.tokenize(KEV_OPTION_END, true);
                let [marker] = tokens[..] else {
                    return Err(DecisionError::InvalidModel(format!(
                        "decision model has no {KEV_OPTION_END} token"
                    )));
                };
                Head::Pointer { marker }
            }
            HeadKind::Joint => Head::Joint,
            HeadKind::Pooled => {
                // The classes come from the classifier labels, the request API
                // is binary: [inconsistent, consistent].
                let width = i64::try_from(metadata.n_classes).ok();
                if metadata.n_classes != POOLED_N_CLASSES || width != Some(metadata.n_embd_out) {
                    return Err(DecisionError::InvalidModel(format!(
                        "decision model must have 2 classifier labels and output one score per class, \
                         got {} labels and {} scores",
                        metadata.n_classes, metadata.n_embd_out
                    )));
                }
                Head::Pooled {
                    n_classes: metadata.n_classes,
                }
            }
        };
        let quoted = match kind.quoted() {
            Some((user_defined, min_len)) => quoted_spellings(vocab, user_defined, min_len),
            None => Vec::new(),
        };
        Ok(Self { kind, head, quoted })
    }

    /// The `<arch>.decision.type` of this readout.
    pub fn name(&self) -> &'static str {
        self.kind.name
    }

    /// The most options a question may have.
    pub fn n_options_max(&self) -> usize {
        match &self.head {
            Head::LabelLogits { labels, .. } => labels.len(),
            Head::Pooled { n_classes } => *n_classes,
            _ => N_OPTIONS_MAX,
        }
    }

    /// Whether a `noul` question lists `true` before `false` whatever its
    /// criteria say.
    pub fn noul_true_first(&self) -> bool {
        self.kind.noul_order == NoulOrder::TrueFirst
    }

    /// Whether the objects of the context are written with sorted keys.
    pub fn sorts_keys(&self) -> bool {
        self.kind.sorts_keys
    }

    /// Whether the prompt of the model has a place for images.
    pub fn accepts_images(&self) -> bool {
        self.kind.media
    }

    /// The text that must not appear in an input, if the layout depends on it.
    pub fn marker_text(&self) -> Option<&str> {
        match &self.head {
            Head::MarkerColumn { marker_text, .. } if !marker_text.is_empty() => Some(marker_text),
            _ => None,
        }
    }

    /// Prompts evaluated to answer `question`, the options in another order in
    /// each.
    pub fn n_variants(&self, question: &Question) -> usize {
        if question.kind == QuestionKind::Choice && question.options.len() > 1 {
            self.kind.choice_variants
        } else {
            1
        }
    }

    /// Raw scores each variant of `question` returns.
    pub fn n_outputs(&self, question: &Question) -> usize {
        match self.kind.noul_scale {
            Some(scale) if question.kind == QuestionKind::Noul => scale,
            _ => question.options.len(),
        }
    }

    /// The labels of `question`: the texts the template shows, by position,
    /// and the tokens whose logits are read. Empty for a readout that reads
    /// no logits.
    pub fn question_labels(
        &self,
        question: &Question,
        vocab: &impl Vocabulary,
    ) -> Result<QuestionLabels, DecisionError> {
        match &self.head {
            Head::LabelLogits {
                labels,
                label_texts,
            } => {
                let n_options = question.options.len();
                let n_outputs = self.n_outputs(question);
                let (Some(tokens), Some(texts)) = (
                    labels.get(..n_outputs),
                    if label_texts.is_empty() {
                        Some(&[][..])
                    } else {
                        label_texts.get(..n_options)
                    },
                ) else {
                    return Err(DecisionError::Evaluation(
                        "the question has more outputs than the model has labels".into(),
                    ));
                };
                Ok(QuestionLabels {
                    texts: texts.to_vec(),
                    tokens: tokens.to_vec(),
                    groups: Vec::new(),
                })
            }
            Head::LabelGroups => {
                let HeadKind::LabelGroups(codes) = self.kind.head else {
                    return Err(DecisionError::Evaluation(
                        "a label group readout of a type without label groups".into(),
                    ));
                };
                d1_labels(question, codes, vocab)
            }
            _ => Ok(QuestionLabels::default()),
        }
    }

    /// The bucket of the temperature that depends on the number of options.
    pub fn temperature_bucket(&self, n_options: usize) -> &'static str {
        match self.kind.buckets {
            Buckets::Size => match n_options {
                0..=8 => "small",
                9..=26 => "mid",
                _ => "large",
            },
            Buckets::Count => match n_options {
                0..=2 => "2",
                3..=5 => "3_5",
                6..=10 => "6_10",
                _ => "11",
            },
        }
    }
}

/// `A..Z` then `AA..ZZ`.
fn codes() -> impl Iterator<Item = String> {
    let single = ('A'..='Z').map(String::from);
    let double = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    single.chain(double)
}

/// The texts of the special tokens of the vocabulary that are at least
/// `min_len` bytes long, control tokens and, with `user_defined`, the
/// user-defined ones too; longest first, id order among equals.
fn quoted_spellings(vocab: &impl Vocabulary, user_defined: bool, min_len: usize) -> Vec<String> {
    let mut spellings: Vec<String> = vocab
        .special_tokens()
        .into_iter()
        .filter(|token| token.control || (user_defined && token.user_defined))
        .map(|token| token.text)
        .filter(|text| text.len() >= min_len)
        .collect();
    // Stable: equal lengths keep the order of their ids.
    spellings.sort_by_key(|text| std::cmp::Reverse(text.len()));
    spellings
}

/// The tokens among the tokenizations of `forms` that are a single token,
/// each once, in order.
fn single_tokens(vocab: &impl Vocabulary, forms: &[&str]) -> Vec<Token> {
    let mut out = Vec::new();
    for form in forms {
        if let [token] = vocab.tokenize(form, false)[..]
            && !out.contains(&token)
        {
            out.push(token);
        }
    }
    out
}

/// The labels of the d1 models, as `prompt.py` of their repository chooses
/// them: the level of a score, yes or no for a noul, and for a choice its
/// keys when they are single letters, codes otherwise, a code that is not a
/// distinct single token replaced by the first free one of a pool.
fn d1_labels(
    question: &Question,
    long_codes: D1Codes,
    vocab: &impl Vocabulary,
) -> Result<QuestionLabels, DecisionError> {
    let mut labels = QuestionLabels::default();
    if question.kind != QuestionKind::Choice {
        for option in &question.options {
            let group = match (question.kind, option.key.as_str()) {
                (QuestionKind::Score, key) => single_tokens(vocab, &[key]),
                (_, "true") => single_tokens(vocab, &["yes", "Yes", "YES"]),
                _ => single_tokens(vocab, &["no", "No", "NO"]),
            };
            if group.is_empty() {
                return Err(DecisionError::InvalidModel(format!(
                    "decision label is not a single token: {}",
                    option.key
                )));
            }
            labels.texts.push(option.key.clone());
            labels.groups.push(group.len());
            labels.tokens.extend(group);
        }
        return Ok(labels);
    }

    let n_options = question.options.len();
    let is_letters = question
        .options
        .iter()
        .all(|option| option.key.len() == 1 && option.key.as_bytes()[0].is_ascii_alphabetic());
    let letter = |i: usize| -> String {
        // Narrowing: `i` is below 26 here, a letter offset.
        char::from(b'A' + i as u8).to_string()
    };
    let codes: Vec<String> = (0..n_options)
        .map(|i| {
            if is_letters {
                question.options[i].key.clone()
            } else if n_options <= 26 || i < 26 && long_codes == D1Codes::Letters {
                letter(i)
            } else if long_codes == D1Codes::Letters {
                format!("{}{}", letter((i - 26) / 26), letter((i - 26) % 26))
            } else {
                format!("{i:02}")
            }
        })
        .collect();

    let pool: Vec<String> = ('A'..='Z')
        .map(String::from)
        .chain((0..100).map(|i| format!("{i:02}")))
        .chain(('a'..='z').map(String::from))
        .chain((0..200).map(|i| format!("#{i}")))
        .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
        .collect();

    let mut used: Vec<Token> = Vec::new();
    let mut take = |code: &str, labels: &mut QuestionLabels| -> bool {
        let [token] = vocab.tokenize(code, false)[..] else {
            return false;
        };
        if used.contains(&token) {
            return false;
        }
        used.push(token);
        let mut group = vec![token];
        group.extend(
            single_tokens(vocab, &[&format!(" {code}")])
                .into_iter()
                .filter(|&other| other != token),
        );
        labels.texts.push(code.to_owned());
        labels.groups.push(group.len());
        labels.tokens.extend(group);
        true
    };
    for code in &codes {
        let taken = take(code, &mut labels) || pool.iter().any(|code| take(code, &mut labels));
        if !taken {
            return Err(DecisionError::invalid(format!(
                "no single-token label left for {n_options} options"
            )));
        }
    }
    Ok(labels)
}

/// The score of an option for a pointer readout: the dot product of the
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
pub(crate) mod tests {
    use super::*;
    use crate::json::Json;
    use crate::request::QuestionOption;

    /// The row of a type, for the tests.
    pub(crate) fn kind(name: &str) -> &'static DecisionType {
        DecisionType::named(name).expect("a known type")
    }

    /// A vocabulary where every one-letter text is one token and the
    /// two-letter codes are one token only when their first letter is before
    /// `pair_below`.
    pub(crate) struct Fake {
        pub(crate) pair_below: char,
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
                [' ', c] if c.is_ascii_uppercase() => {
                    vec![2000 + i32::try_from(u32::from(c)).expect("ascii")]
                }
                [a, b] if a < self.pair_below && a != ' ' => {
                    vec![1000 + i32::try_from(u32::from(a) * 100 + u32::from(b)).expect("small")]
                }
                _ if text == "yes" => vec![3000],
                _ if text == "Yes" => vec![3001],
                _ if text == "no" => vec![3002],
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

        fn special_tokens(&self) -> Vec<SpecialToken> {
            ["<s>", "<|im_end|>", "<x>", "a"]
                .iter()
                .enumerate()
                .map(|(i, text)| SpecialToken {
                    text: (*text).into(),
                    control: i != 2,
                    user_defined: i == 2,
                })
                .collect()
        }
    }

    pub(crate) fn laya() -> Readout {
        Readout::new(
            kind("laya"),
            Head::MarkerColumn {
                marker: 4,
                sep: 1,
                marker_text: "<mask>".into(),
            },
        )
    }

    pub(crate) fn lev() -> Readout {
        Readout::new(
            kind("lev"),
            Head::LabelLogits {
                labels: (0..26).collect(),
                label_texts: ('A'..='Z').map(String::from).collect(),
            },
        )
    }

    fn from_type(name: &str, vocab: &impl Vocabulary) -> Result<Readout, DecisionError> {
        Readout::from_decision_type(Some(name), &HeadMetadata::default(), vocab)
    }

    fn question(kind: QuestionKind, keys: &[&str]) -> Question {
        Question {
            kind,
            instructions: Json::from("i"),
            options: keys
                .iter()
                .map(|key| QuestionOption {
                    key: (*key).into(),
                    description: Json::Null,
                })
                .collect(),
        }
    }

    #[test]
    fn every_type_of_the_server_has_one_row() {
        let names: Vec<_> = DECISION_TYPES.iter().map(|row| row.name).collect();
        for name in [
            "laya",
            "openjev",
            "lev",
            "kev",
            "nimble",
            "clef",
            "pplx-decider",
            "vllm-decision2",
            "vllm-d3",
            "lfm2-d1",
            "lfm2-d1-omni",
            "granite-switch",
            "h2o-lightning",
            "openbmb-minicpm-ocr-jev",
            "lfm3-d1",
            "lfm3-d1-omni",
        ] {
            assert_eq!(names.iter().filter(|n| **n == name).count(), 1, "{name}");
        }
        assert_eq!(names.len(), 16);
    }

    #[test]
    fn missing_key_and_unknown_type_are_distinct_errors() {
        let vocab = Fake { pair_below: 'A' };
        assert!(matches!(
            Readout::from_decision_type(None, &HeadMetadata::default(), &vocab),
            Err(DecisionError::NotDecisionModel)
        ));
        match from_type("acme-decider", &vocab) {
            Err(DecisionError::UnknownType { name }) => assert_eq!(name, "acme-decider"),
            other => panic!("expected UnknownType, got {other:?}"),
        }
    }

    #[test]
    fn laya_keeps_the_marker_text_for_sanitising() {
        let readout = from_type("laya", &Fake { pair_below: 'A' }).expect("laya");
        assert_eq!(readout.marker_text(), Some("<mask>"));
        assert_eq!(readout.n_options_max(), 255);
        assert!(!readout.noul_true_first());
        assert!(readout.kind.trainable);
    }

    #[test]
    fn lev_keeps_only_single_token_codes() {
        // Pairs `AA..CZ` are one token: 26 + 3 * 26 labels.
        let readout = from_type("lev", &Fake { pair_below: 'D' }).expect("lev");
        let Head::LabelLogits {
            labels,
            label_texts,
        } = &readout.head
        else {
            panic!("lev is a label readout");
        };
        assert_eq!(labels.len(), 26 + 3 * 26);
        assert_eq!(label_texts[26], "AA");
        assert_eq!(label_texts.last().map(String::as_str), Some("CZ"));
        let choice = question(QuestionKind::Choice, &["a", "b"]);
        let noul = question(QuestionKind::Noul, &["false", "true"]);
        assert_eq!(readout.n_variants(&choice), 2);
        assert_eq!(readout.n_outputs(&noul), 9);
        assert_eq!(readout.n_options_max(), 104);
        assert_eq!(readout.temperature_bucket(8), "small");
        assert_eq!(readout.temperature_bucket(9), "mid");
        assert_eq!(readout.temperature_bucket(27), "large");
        // The noul of a rating scale reads nine labels, shows two.
        let labels = readout
            .question_labels(&noul, &Fake { pair_below: 'D' })
            .expect("labels");
        assert_eq!(labels.tokens.len(), 9);
        assert_eq!(labels.texts, ["A", "B"]);
    }

    #[test]
    fn code_labels_are_capped_at_255() {
        for name in ["lev", "nimble", "pplx-decider", "vllm-d3", "granite-switch"] {
            let readout = from_type(name, &Fake { pair_below: '[' }).expect(name);
            assert_eq!(readout.n_options_max(), 255, "{name}");
        }
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
            fn special_tokens(&self) -> Vec<SpecialToken> {
                Vec::new()
            }
        }
        // 'A'..='G' is 7 labels, under the 9 ratings.
        assert!(matches!(
            from_type("lev", &Poor),
            Err(DecisionError::InvalidModel(_))
        ));
        // Other code readouts have no rating scale.
        assert!(from_type("nimble", &Poor).is_ok());
        for name in ["laya", "lfm2-d1-omni"] {
            assert!(matches!(
                from_type(name, &Poor),
                Err(DecisionError::InvalidModel(_))
            ));
        }
    }

    #[test]
    fn openjev_labels_are_the_52_letters() {
        let readout = from_type("openjev", &Fake { pair_below: 'A' }).expect("openjev");
        assert_eq!(readout.n_options_max(), 52);
        assert!(readout.noul_true_first());
        assert!(readout.accepts_images());
        assert!(!readout.sorts_keys());
        let labels = readout
            .question_labels(
                &question(QuestionKind::Choice, &["x", "y"]),
                &Fake { pair_below: 'A' },
            )
            .expect("labels");
        assert!(labels.texts.is_empty(), "openjev shows no label");
        assert_eq!(labels.tokens, [65, 66]);
    }

    #[test]
    fn kev_needs_a_single_special_end_token() {
        let readout = from_type("kev", &Fake { pair_below: 'A' }).expect("kev");
        assert_eq!(readout.head, Head::Pointer { marker: 999 });
    }

    #[test]
    fn h2o_reads_the_labels_of_the_metadata_and_quotes_special_tokens() {
        let metadata = HeadMetadata {
            labels: vec!["A".into(), "B".into(), "C".into()],
            ..HeadMetadata::default()
        };
        let vocab = Fake { pair_below: 'A' };
        let readout =
            Readout::from_decision_type(Some("h2o-lightning"), &metadata, &vocab).expect("h2o");
        assert_eq!(readout.n_options_max(), 3);
        assert_eq!(readout.quoted, ["<|im_end|>", "<s>", "<x>", "a"]);

        let twice = HeadMetadata {
            labels: vec!["A".into(), "A".into()],
            ..HeadMetadata::default()
        };
        match Readout::from_decision_type(Some("h2o-lightning"), &twice, &vocab) {
            Err(DecisionError::InvalidModel(message)) => {
                assert_eq!(
                    message,
                    "decision label \"A\" is not a single distinct token"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(matches!(
            from_type("h2o-lightning", &vocab),
            Err(DecisionError::InvalidModel(message)) if message == "decision model has no labels"
        ));
    }

    #[test]
    fn lfm3_breaks_control_tokens_of_two_bytes_or_more() {
        let readout = from_type("lfm3-d1", &Fake { pair_below: 'A' }).expect("lfm3");
        assert_eq!(readout.quoted, ["<|im_end|>", "<s>"]);
        assert!(
            from_type("lfm2-d1", &Fake { pair_below: 'A' })
                .expect("lfm2")
                .quoted
                .is_empty()
        );
    }

    #[test]
    fn the_pooled_classifier_needs_two_classes_of_one_score_each() {
        let vocab = Fake { pair_below: 'A' };
        let two = HeadMetadata {
            n_classes: 2,
            n_embd_out: 2,
            ..HeadMetadata::default()
        };
        let readout = Readout::from_decision_type(Some("openbmb-minicpm-ocr-jev"), &two, &vocab)
            .expect("ocr");
        assert_eq!(readout.n_options_max(), 2);
        let wide = HeadMetadata {
            n_classes: 2,
            n_embd_out: 64,
            ..HeadMetadata::default()
        };
        match Readout::from_decision_type(Some("openbmb-minicpm-ocr-jev"), &wide, &vocab) {
            Err(DecisionError::InvalidModel(message)) => assert_eq!(
                message,
                "decision model must have 2 classifier labels and output one score per class, got 2 labels and 64 scores"
            ),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn d1_labels_follow_the_question_type() {
        let vocab = Fake { pair_below: 'A' };
        let readout = from_type("lfm2-d1", &vocab).expect("lfm2-d1");
        // Letters as keys are their own labels, with the spaced form grouped.
        let labels = readout
            .question_labels(&question(QuestionKind::Choice, &["b", "Q"]), &vocab)
            .expect("labels");
        assert_eq!(labels.texts, ["b", "Q"]);
        assert_eq!(labels.tokens, [98, 81, 2081]);
        assert_eq!(labels.groups, [1, 2]);
        // Other keys get codes.
        let labels = readout
            .question_labels(&question(QuestionKind::Choice, &["x1", "x2"]), &vocab)
            .expect("labels");
        assert_eq!(labels.texts, ["A", "B"]);
        // A noul reads its yes and no forms.
        let labels = readout
            .question_labels(&question(QuestionKind::Noul, &["true", "false"]), &vocab)
            .expect("labels");
        assert_eq!(labels.texts, ["true", "false"]);
        assert_eq!(labels.tokens, [3000, 3001, 3002]);
        assert_eq!(labels.groups, [2, 1]);
        // A score reads its levels.
        let labels = readout
            .question_labels(&question(QuestionKind::Score, &["0", "1"]), &vocab)
            .expect("labels");
        assert_eq!(labels.tokens, [48, 49]);
    }

    #[test]
    fn d1_long_choices_take_codes_of_their_scheme() {
        let vocab = Fake { pair_below: '[' };
        let keys: Vec<String> = (0..30).map(|i| format!("k{i}")).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let long = question(QuestionKind::Choice, &keys);
        let lfm3 = from_type("lfm3-d1", &vocab)
            .expect("lfm3")
            .question_labels(&long, &vocab)
            .expect("labels");
        assert_eq!(&lfm3.texts[24..28], ["Y", "Z", "AA", "AB"]);
        // lfm2 numbers them instead.
        let lfm2 = from_type("lfm2-d1", &vocab)
            .expect("lfm2")
            .question_labels(&long, &vocab)
            .expect("labels");
        assert_eq!(&lfm2.texts[..2], ["00", "01"]);
    }

    #[test]
    fn numeric_buckets_match_the_fitting_ranges() {
        let readout = from_type("laya", &Fake { pair_below: 'A' }).expect("laya");
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
