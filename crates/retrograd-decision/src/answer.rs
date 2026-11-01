//! From the raw scores of a model to the answer of a question.
//!
//! The arithmetic is the one of llama-server, operation for operation, so the
//! same scores give the same answer to the last bit: a softmax in `f64` over
//! the scores divided by a temperature, the average of the variants of a
//! question, and the confidence formulas published by TypeSafe.

use std::collections::BTreeMap;

use crate::json::Json;
use indexmap::IndexMap;
use serde::Serialize;

use crate::DecisionError;
use crate::readout::{N_OPTIONS_MAX, Readout};
use crate::request::{Question, QuestionKind};

/// The answer of one question, serialized as llama-server writes it:
/// `{"type": "choice", "choice": …, "probabilities": {…}, "confidence": …}`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Score {
        /// The expectation of the level.
        score: f64,
        /// The description of each level, keyed by its index.
        legend: IndexMap<String, Json>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Noul {
        /// The probability of `true`; for a rating scale, the expected rating
        /// over its range.
        noul: f64,
    },
}

/// The numbers a model carries besides its weights: the temperatures of its
/// softmax and the biases of its score questions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    /// `"<type>"` or `"<type>.<bucket>"`, plus `"image"`.
    temperatures: BTreeMap<String, f32>,
    /// Added to the scores of a `score` question, by number of levels.
    score_bias: BTreeMap<usize, Vec<f32>>,
}

impl Calibration {
    /// Reads the entries of the metadata under `<arch>.decision.`, with that
    /// prefix removed: `temperature.<name>` and `score_bias.<n_levels>.<level>`.
    /// Other entries are ignored.
    ///
    /// The server reads its values from the *text* of the metadata, which
    /// rounds a float to six decimals; pass the same text to get the same
    /// temperatures.
    pub fn from_metadata<'a>(
        entries: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, DecisionError> {
        let mut calibration = Self::default();
        for (key, value) in entries {
            if let Some(name) = key.strip_prefix("temperature.") {
                match value.trim().parse::<f32>() {
                    Ok(temperature) if temperature.is_finite() && temperature > 0.0 => {
                        calibration
                            .temperatures
                            .insert(name.to_owned(), temperature);
                    }
                    _ => {
                        return Err(DecisionError::InvalidModel(format!(
                            "invalid decision temperature: {key} = {value}"
                        )));
                    }
                }
            } else if let Some(rest) = key.strip_prefix("score_bias.") {
                let invalid =
                    || DecisionError::InvalidModel(format!("invalid decision score bias: {key}"));
                let (n_levels, level) = rest.split_once('.').ok_or_else(invalid)?;
                let n_levels: usize = n_levels.parse().map_err(|_| invalid())?;
                let level: usize = level.parse().map_err(|_| invalid())?;
                if !(2..=N_OPTIONS_MAX).contains(&n_levels) || level >= n_levels {
                    return Err(invalid());
                }
                let bias: f32 = value.trim().parse().map_err(|_| invalid())?;
                if !bias.is_finite() {
                    return Err(invalid());
                }
                calibration
                    .score_bias
                    .entry(n_levels)
                    .or_insert_with(|| vec![0.0; n_levels])[level] = bias;
            }
        }
        Ok(calibration)
    }

    /// The temperatures by name: `<type>`, `<type>.<bucket>`, `image`.
    pub fn temperatures(&self) -> &BTreeMap<String, f32> {
        &self.temperatures
    }

    /// This calibration with the temperatures of `entries` in place of its
    /// own, its score biases kept. `entries` are metadata entries with their
    /// `<arch>.decision.` or `adapter.decision.` prefix removed; only the
    /// `temperature.*` ones are read. A set fitted together replaces a set
    /// fitted on other weights instead of mixing with it: a bucket the new set
    /// lacks falls back to its type, not to the old weights' bucket.
    pub fn with_temperatures_of<'a>(
        &self,
        entries: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, DecisionError> {
        let fitted = Self::from_metadata(
            entries
                .into_iter()
                .filter(|(key, _)| key.starts_with("temperature.")),
        )?;
        Ok(Self {
            temperatures: fitted.temperatures,
            score_bias: self.score_bias.clone(),
        })
    }

    /// The temperature of `kind` with `n_options` options: `<type>.<bucket>`,
    /// then `<type>`, then 1.
    pub fn temperature(&self, readout: &Readout, kind: QuestionKind, n_options: usize) -> f32 {
        let name = kind.name();
        let bucketed = temperature_key(readout, kind, n_options);
        [bucketed.as_str(), name]
            .into_iter()
            .find_map(|key| self.temperatures.get(key).copied())
            .unwrap_or(1.0)
    }
}

/// The key of the temperature that is fitted for a question: `<type>.<bucket>`.
pub fn temperature_key(readout: &Readout, kind: QuestionKind, n_options: usize) -> String {
    format!("{}.{}", kind.name(), readout.temperature_bucket(n_options))
}

/// `softmax(logits / temperature)`, stable: the maximum is subtracted first.
pub fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|logit| ((logit - max) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|exp| exp / sum).collect()
}

/// An index or a count as `f64`. Exact: options are at most 255.
fn index_f64(index: usize) -> f64 {
    index as f64
}

/// The position of the first largest value, as `std::max_element` finds it.
pub(crate) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = i;
        }
    }
    best
}

/// `(p_max - 1/n) / (1 - 1/n)`, 0 for a uniform distribution, 1 when one option
/// takes everything.
pub(crate) fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / index_f64(probs.len());
    let p_max = probs[argmax(probs)];
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// One minus the mean distance to the mode, relative to the one of a uniform
/// distribution around its center.
pub(crate) fn confidence_score(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let n = index_f64(probs.len());
    let mode = index_f64(argmax(probs));
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, p) in probs.iter().enumerate() {
        let i = index_f64(i);
        dist += p * (i - mode).abs();
        dist_uniform += (i - (n - 1.0) / 2.0).abs() / n;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

/// Turns the raw scores of `question` into its answer. `scores` has one vector
/// per variant ([`Readout::n_variants`]), each [`Readout::n_outputs`] long, in
/// the order the variant showed the options. `has_media` selects the `image`
/// temperature when the model carries one.
pub fn format_answer(
    readout: &Readout,
    calibration: &Calibration,
    question: &Question,
    scores: &[Vec<f32>],
    has_media: bool,
) -> Result<Answer, DecisionError> {
    if question.options.is_empty() || question.options.len() > readout.n_options_max() {
        return Err(DecisionError::Evaluation(
            "decision question has an invalid number of options".into(),
        ));
    }
    let n = readout.n_outputs(question);
    if scores.len() != readout.n_variants(question) {
        return Err(DecisionError::Evaluation(
            "decision result does not match the number of variants".into(),
        ));
    }

    let mut temperature = calibration.temperature(readout, question.kind, question.options.len());
    if has_media && let Some(&image) = calibration.temperatures.get("image") {
        temperature = image;
    }
    let bias = if question.kind == QuestionKind::Score {
        calibration.score_bias.get(&n)
    } else {
        None
    };

    // The softmax of each variant, then their average.
    let mut probs = vec![0.0; n];
    for (v, variant) in scores.iter().enumerate() {
        if variant.len() != n {
            return Err(DecisionError::Evaluation(
                "decision result does not match the number of options".into(),
            ));
        }
        if variant
            .iter()
            .any(|score| score.is_nan() || *score == f32::INFINITY)
        {
            return Err(DecisionError::Evaluation(
                "the scores are not finite".into(),
            ));
        }
        let mut logits: Vec<f64> = variant.iter().map(|&score| f64::from(score)).collect();
        if let Some(bias) = bias {
            for (logit, bias) in logits.iter_mut().zip(bias) {
                *logit += f64::from(*bias);
            }
        }
        if !logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .is_finite()
        {
            return Err(DecisionError::Evaluation(
                "the scores are not finite".into(),
            ));
        }
        let p = softmax(&logits, f64::from(temperature));
        for (i, p) in p.into_iter().enumerate() {
            // The second variant is in the reverse order.
            probs[if v == 0 { i } else { n - 1 - i }] += p / index_f64(scores.len());
        }
    }

    match question.kind {
        QuestionKind::Noul => {
            if matches!(
                readout,
                Readout::LabelLogits {
                    noul_scale: Some(_),
                    ..
                }
            ) {
                let expected: f64 = probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * index_f64(i) / index_f64(n - 1))
                    .sum();
                return Ok(Answer::Noul { noul: expected });
            }
            let noul = question
                .options
                .iter()
                .zip(&probs)
                .find_map(|(option, p)| (option.key == "true").then_some(*p))
                .ok_or_else(|| {
                    DecisionError::Evaluation("a noul question has no true option".into())
                })?;
            Ok(Answer::Noul { noul })
        }
        QuestionKind::Choice => {
            let confidence = confidence_choice(&probs);
            let choice = question.options[argmax(&probs)].key.clone();
            Ok(Answer::Choice {
                choice,
                probabilities: probabilities(question, &probs),
                confidence,
            })
        }
        QuestionKind::Score => {
            let score: f64 = probs
                .iter()
                .enumerate()
                .map(|(i, p)| index_f64(i) * p)
                .sum();
            let legend = question
                .options
                .iter()
                .map(|option| (option.key.clone(), option.description.clone()))
                .collect();
            Ok(Answer::Score {
                score,
                legend,
                confidence: confidence_score(&probs),
                probabilities: probabilities(question, &probs),
            })
        }
    }
}

fn probabilities(question: &Question, probs: &[f64]) -> IndexMap<String, f64> {
    question
        .options
        .iter()
        .zip(probs)
        .map(|(option, p)| (option.key.clone(), *p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readout::LabelFamily;
    use crate::request::QuestionOption;

    fn laya() -> Readout {
        Readout::MarkerColumn {
            marker: 4,
            sep: 1,
            marker_text: "<mask>".into(),
        }
    }

    fn lev() -> Readout {
        Readout::LabelLogits {
            labels: (0..26).collect(),
            label_texts: ('A'..='Z').map(String::from).collect(),
            n_variants: 2,
            noul_scale: Some(9),
            family: LabelFamily::Codes,
        }
    }

    fn question(kind: QuestionKind, keys: &[&str]) -> Question {
        Question {
            kind,
            instructions: json!("i"),
            options: keys
                .iter()
                .map(|key| QuestionOption {
                    key: (*key).into(),
                    description: json!(format!("d-{key}")),
                })
                .collect(),
        }
    }

    /// The temperatures of `Laya-Q8_0.gguf`, as the text of its metadata.
    fn laya_calibration() -> Calibration {
        Calibration::from_metadata([
            ("type", "laya"),
            ("max_head_tokens", "192"),
            ("temperature.choice", "1.636903"),
            ("temperature.score", "1.251430"),
            ("temperature.noul", "1.983400"),
            ("temperature.choice.3_5", "1.760152"),
            ("temperature.choice.6_10", "1.000016"),
            ("temperature.score.3_5", "1.251430"),
            ("temperature.noul.2", "1.983400"),
            ("temperature.choice.11", "0.100583"),
            ("temperature.choice.2", "1.906356"),
        ])
        .expect("calibration")
    }

    fn approx(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "{left} != {right}");
    }

    #[test]
    fn temperature_falls_back_from_bucket_to_type_to_one() {
        let calibration = laya_calibration();
        let t = |kind, n| calibration.temperature(&laya(), kind, n);
        // Bucket entries.
        assert_eq!(t(QuestionKind::Choice, 2), 1.906356_f32);
        assert_eq!(t(QuestionKind::Choice, 3), 1.760152_f32);
        assert_eq!(t(QuestionKind::Choice, 5), 1.760152_f32);
        assert_eq!(t(QuestionKind::Choice, 6), 1.000016_f32);
        assert_eq!(t(QuestionKind::Choice, 10), 1.000016_f32);
        assert_eq!(t(QuestionKind::Choice, 11), 0.100583_f32);
        assert_eq!(t(QuestionKind::Noul, 2), 1.9834_f32);
        // `score` has only the 3_5 bucket: the others fall back to `score`.
        assert_eq!(t(QuestionKind::Score, 4), 1.25143_f32);
        assert_eq!(t(QuestionKind::Score, 10), 1.25143_f32);
        // No entry at all.
        let none = Calibration::default();
        assert_eq!(none.temperature(&laya(), QuestionKind::Choice, 3), 1.0);
        // A type without bucket entry reads the type.
        let only_type =
            Calibration::from_metadata([("temperature.choice", "1.636903")]).expect("calibration");
        assert_eq!(
            only_type.temperature(&laya(), QuestionKind::Choice, 7),
            1.636903_f32
        );
    }

    #[test]
    fn lev_buckets_are_by_size() {
        let calibration = Calibration::from_metadata([
            ("temperature.choice.small", "2"),
            ("temperature.choice.mid", "3"),
            ("temperature.choice.large", "4"),
        ])
        .expect("calibration");
        let t = |n| calibration.temperature(&lev(), QuestionKind::Choice, n);
        assert_eq!((t(8), t(9), t(26), t(27)), (2.0, 3.0, 3.0, 4.0));
        // The same keys mean nothing to a numeric readout.
        assert_eq!(
            calibration.temperature(&laya(), QuestionKind::Choice, 4),
            1.0
        );
    }

    #[test]
    fn invalid_metadata_is_refused() {
        for (key, value) in [
            ("temperature.choice", "0"),
            ("temperature.choice", "-1"),
            ("temperature.choice", "nan"),
            ("temperature.choice", "warm"),
            ("score_bias.4.4", "1"),
            ("score_bias.1.0", "1"),
            ("score_bias.256.0", "1"),
            ("score_bias.4", "1"),
            ("score_bias.x.0", "1"),
            ("score_bias.4.0", "n/a"),
            ("score_bias.4.0", "NaN"),
            ("score_bias.4.0", "inf"),
            ("score_bias.4.0", "-inf"),
        ] {
            assert!(
                matches!(
                    Calibration::from_metadata([(key, value)]),
                    Err(DecisionError::InvalidModel(_))
                ),
                "{key} = {value}"
            );
        }
    }

    #[test]
    fn choice_with_ties_picks_the_first_and_confidence_is_zero_when_uniform() {
        let q = question(QuestionKind::Choice, &["a", "b", "c"]);
        let answer = format_answer(
            &laya(),
            &Calibration::default(),
            &q,
            &[vec![1.0, 1.0, 1.0]],
            false,
        )
        .expect("answer");
        let Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = answer
        else {
            panic!("a choice answer");
        };
        assert_eq!(choice, "a");
        approx(probabilities["b"], 1.0 / 3.0);
        assert_eq!(confidence, 0.0);
        let keys: Vec<_> = probabilities.keys().map(String::as_str).collect();
        assert_eq!(keys, ["a", "b", "c"]);
    }

    #[test]
    fn choice_confidence_reaches_one_for_a_certain_answer() {
        let q = question(QuestionKind::Choice, &["a", "b"]);
        let answer = format_answer(
            &laya(),
            &Calibration::default(),
            &q,
            &[vec![0.0, 800.0]],
            false,
        )
        .expect("answer");
        let Answer::Choice {
            choice, confidence, ..
        } = answer
        else {
            panic!("a choice answer");
        };
        assert_eq!(choice, "b");
        approx(confidence, 1.0);
        // A single option has nothing to be unsure about.
        approx(confidence_choice(&[1.0]), 1.0);
    }

    #[test]
    fn score_confidence_and_expectation() {
        // All mass on one level: confidence 1, the score is that level.
        let probs = [0.0, 1.0, 0.0, 0.0];
        approx(confidence_score(&probs), 1.0);
        // Uniform: the mean distance to the mode equals the uniform reference.
        approx(confidence_score(&[0.25; 4]), 0.0);

        let q = question(QuestionKind::Score, &["0", "1", "2"]);
        let answer = format_answer(
            &laya(),
            &Calibration::default(),
            &q,
            &[vec![0.0, 0.0, 0.0]],
            false,
        )
        .expect("answer");
        let Answer::Score {
            score, confidence, ..
        } = answer
        else {
            panic!("a score answer");
        };
        approx(score, 1.0);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn noul_is_the_probability_of_true_wherever_it_is_listed() {
        let scores = vec![vec![0.0_f32, 3.0_f32.ln()]];
        let false_first = question(QuestionKind::Noul, &["false", "true"]);
        let Answer::Noul { noul } = format_answer(
            &laya(),
            &Calibration::default(),
            &false_first,
            &scores,
            false,
        )
        .expect("answer") else {
            panic!("a noul answer");
        };
        assert!((noul - 0.75).abs() < 1e-7);

        let true_first = question(QuestionKind::Noul, &["true", "false"]);
        let Answer::Noul { noul } = format_answer(
            &laya(),
            &Calibration::default(),
            &true_first,
            &scores,
            false,
        )
        .expect("answer") else {
            panic!("a noul answer");
        };
        assert!((noul - 0.25).abs() < 1e-7);
    }

    #[test]
    fn lev_averages_the_two_orders_of_a_choice() {
        let q = question(QuestionKind::Choice, &["a", "b", "c"]);
        // Variant 0 sees a, b, c; variant 1 sees c, b, a. Both prefer the first
        // label shown (a bias for "A"), so the average cancels it:
        // variant 0 -> [0.8, 0.1, 0.1], variant 1 -> [0.8, 0.1, 0.1] over c, b, a.
        let logits = |p: [f32; 3]| p.iter().map(|p| p.ln()).collect::<Vec<_>>();
        let scores = vec![logits([0.8, 0.1, 0.1]), logits([0.8, 0.1, 0.1])];
        let Answer::Choice { probabilities, .. } =
            format_answer(&lev(), &Calibration::default(), &q, &scores, false).expect("answer")
        else {
            panic!("a choice answer");
        };
        assert!((probabilities["a"] - 0.45).abs() < 1e-6);
        assert!((probabilities["b"] - 0.10).abs() < 1e-6);
        assert!((probabilities["c"] - 0.45).abs() < 1e-6);
    }

    #[test]
    fn lev_reads_noul_from_its_rating_scale() {
        let q = question(QuestionKind::Noul, &["false", "true"]);
        // All the mass on rating 6 of 0..=8.
        let mut scores = vec![0.0_f32; 9];
        scores[6] = 1000.0;
        let Answer::Noul { noul } =
            format_answer(&lev(), &Calibration::default(), &q, &[scores], false).expect("answer")
        else {
            panic!("a noul answer");
        };
        approx(noul, 6.0 / 8.0);
    }

    #[test]
    fn score_bias_shifts_the_logits_of_its_level_count() {
        let calibration =
            Calibration::from_metadata([("score_bias.3.2", "-1")]).expect("calibration");
        let three = question(QuestionKind::Score, &["0", "1", "2"]);
        let Answer::Score { probabilities, .. } =
            format_answer(&laya(), &calibration, &three, &[vec![0.0, 0.0, 0.0]], false)
                .expect("answer")
        else {
            panic!("a score answer");
        };
        let expected = softmax(&[0.0, 0.0, -1.0], 1.0);
        approx(probabilities["2"], expected[2]);
        // Another level count is not shifted.
        let two = question(QuestionKind::Score, &["0", "1"]);
        let Answer::Score { probabilities, .. } =
            format_answer(&laya(), &calibration, &two, &[vec![0.0, 0.0]], false).expect("answer")
        else {
            panic!("a score answer");
        };
        approx(probabilities["0"], 0.5);
    }

    #[test]
    fn the_image_temperature_applies_to_requests_with_media_only() {
        let calibration =
            Calibration::from_metadata([("temperature.choice", "1"), ("temperature.image", "4")])
                .expect("calibration");
        let q = question(QuestionKind::Choice, &["a", "b"]);
        let run = |has_media| {
            let Answer::Choice { probabilities, .. } =
                format_answer(&laya(), &calibration, &q, &[vec![0.0, 4.0]], has_media)
                    .expect("answer")
            else {
                panic!("a choice answer");
            };
            probabilities["b"]
        };
        approx(run(false), 1.0 / (1.0 + (-4.0_f64).exp()));
        approx(run(true), 1.0 / (1.0 + (-1.0_f64).exp()));
    }

    #[test]
    fn unusable_scores_are_refused() {
        let q = question(QuestionKind::Choice, &["a", "b"]);
        let run = |scores: Vec<Vec<f32>>| {
            format_answer(&laya(), &Calibration::default(), &q, &scores, false)
        };
        assert!(run(vec![vec![f32::NAN, 0.0]]).is_err());
        assert!(run(vec![vec![f32::INFINITY, 0.0]]).is_err());
        assert!(run(vec![vec![f32::NEG_INFINITY, f32::NEG_INFINITY]]).is_err());
        // Wrong number of variants or of outputs.
        assert!(run(vec![]).is_err());
        assert!(run(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).is_err());
        assert!(run(vec![vec![0.0]]).is_err());
        // A negative infinity next to a finite score is a zero probability.
        let Answer::Choice { probabilities, .. } =
            run(vec![vec![f32::NEG_INFINITY, 0.0]]).expect("answer")
        else {
            panic!("a choice answer");
        };
        assert_eq!(probabilities["a"], 0.0);
    }

    #[test]
    fn an_empty_question_is_an_evaluation_error() {
        let q = question(QuestionKind::Choice, &[]);
        assert!(matches!(
            format_answer(&laya(), &Calibration::default(), &q, &[vec![]], false),
            Err(DecisionError::Evaluation(_))
        ));
    }

    #[test]
    fn serialization_has_the_shape_of_llama_server() {
        let q = question(QuestionKind::Choice, &["b", "a"]);
        let answer = format_answer(
            &laya(),
            &Calibration::default(),
            &q,
            &[vec![0.0, 0.0]],
            false,
        )
        .expect("answer");
        let text = serde_json::to_string(&answer).expect("json");
        assert_eq!(
            text,
            r#"{"type":"choice","choice":"b","probabilities":{"b":0.5,"a":0.5},"confidence":0.0}"#
        );
        let noul = serde_json::to_value(Answer::Noul { noul: 0.25 }).expect("json");
        assert_eq!(noul, serde_json::json!({"type": "noul", "noul": 0.25}));
    }
}
