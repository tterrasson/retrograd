//! How good a decision model is on a labelled set.
//!
//! One [`Outcome`] per question: the distribution the model gave over the
//! options and the target distribution. [`Metrics`] sums them into what a
//! caller of the service cares about - is the top option right, how much
//! probability does the right answer get, and can the confidence the service
//! reports be thresholded:
//!
//! - accuracy: the top option is the target's top option;
//! - NLL: `-sum(target * ln p)`, the training objective at the same
//!   temperature;
//! - Brier: `sum((p - target)^2)`, over all options;
//! - ECE: the expected calibration error of the top probability, over
//!   [`ECE_BINS`] equal-width bins;
//! - coverage: for each threshold of [`CONFIDENCE_THRESHOLDS`], the share of
//!   questions whose confidence reaches it and the accuracy on those. The
//!   confidence is the one the service publishes (`confidence` of a choice or
//!   a score), and for a `noul` the same formula over its two outcomes.
//!
//! Everything is reported overall and per question type.

use std::collections::BTreeMap;

use crate::DecisionError;
use crate::answer::{argmax, confidence_choice, confidence_score};
use crate::request::QuestionKind;

/// The bins of the expected calibration error.
pub const ECE_BINS: usize = 15;

/// The confidence thresholds the coverage is reported at.
pub const CONFIDENCE_THRESHOLDS: [f64; 5] = [0.5, 0.7, 0.8, 0.9, 0.95];

/// The smallest probability a log is taken of, so a confident wrong answer
/// costs a large finite loss rather than an infinite one. A probability is
/// also capped at 1, which a rounded softmax can exceed by an ulp.
const MIN_PROBABILITY: f64 = 1e-12;

/// What the model said about one question, and what was right.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub kind: QuestionKind,
    /// The model's distribution over the options, in the question's order.
    pub probabilities: Vec<f64>,
    /// The target distribution, same order, summing to 1.
    pub target: Vec<f64>,
}

impl Outcome {
    pub fn new(
        kind: QuestionKind,
        probabilities: Vec<f64>,
        target: Vec<f64>,
    ) -> Result<Self, DecisionError> {
        let fail = |message: &str| DecisionError::Evaluation(message.to_owned());
        if probabilities.len() < 2 || probabilities.len() != target.len() {
            return Err(fail(
                "an outcome needs two or more probabilities and as many targets",
            ));
        }
        if probabilities
            .iter()
            .chain(&target)
            .any(|p| !p.is_finite() || *p < 0.0)
        {
            return Err(fail(
                "probabilities and targets must be finite and not negative",
            ));
        }
        Ok(Self {
            kind,
            probabilities,
            target,
        })
    }

    fn correct(&self) -> bool {
        argmax(&self.probabilities) == argmax(&self.target)
    }

    fn top_probability(&self) -> f64 {
        self.probabilities[argmax(&self.probabilities)]
    }

    /// The confidence the service publishes for this question.
    fn confidence(&self) -> f64 {
        match self.kind {
            QuestionKind::Score => confidence_score(&self.probabilities),
            QuestionKind::Choice | QuestionKind::Noul => confidence_choice(&self.probabilities),
        }
    }

    fn nll(&self) -> f64 {
        self.target
            .iter()
            .zip(&self.probabilities)
            .filter(|(t, _)| **t > 0.0)
            .map(|(t, p)| -t * p.clamp(MIN_PROBABILITY, 1.0).ln())
            .sum()
    }

    fn brier(&self) -> f64 {
        self.target
            .iter()
            .zip(&self.probabilities)
            .map(|(t, p)| (p - t).powi(2))
            .sum()
    }
}

/// Accuracy on the questions whose confidence reaches a threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coverage {
    pub threshold: f64,
    /// The share of questions at or above the threshold.
    pub coverage: f64,
    /// The accuracy on those; NaN when there is none.
    pub accuracy: f64,
}

/// The metrics of a set of outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub coverage: Vec<Coverage>,
}

impl Summary {
    fn of(outcomes: &[&Outcome]) -> Self {
        let n = outcomes.len();
        let mean = |f: &dyn Fn(&Outcome) -> f64| {
            if n == 0 {
                f64::NAN
            } else {
                outcomes.iter().map(|outcome| f(outcome)).sum::<f64>() / count(n)
            }
        };
        let coverage = CONFIDENCE_THRESHOLDS
            .iter()
            .map(|&threshold| {
                let covered: Vec<&&Outcome> = outcomes
                    .iter()
                    .filter(|outcome| outcome.confidence() >= threshold)
                    .collect();
                Coverage {
                    threshold,
                    coverage: if n == 0 {
                        f64::NAN
                    } else {
                        count(covered.len()) / count(n)
                    },
                    accuracy: if covered.is_empty() {
                        f64::NAN
                    } else {
                        count(covered.iter().filter(|outcome| outcome.correct()).count())
                            / count(covered.len())
                    },
                }
            })
            .collect();
        Self {
            n,
            accuracy: mean(&|outcome| f64::from(u8::from(outcome.correct()))),
            nll: mean(&Outcome::nll),
            brier: mean(&Outcome::brier),
            ece: expected_calibration_error(outcomes),
            coverage,
        }
    }
}

/// The metrics of a set, overall and per question type.
#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    pub overall: Summary,
    /// By [`QuestionKind::name`], only the types present.
    pub per_type: BTreeMap<&'static str, Summary>,
}

impl Metrics {
    pub fn of(outcomes: &[Outcome]) -> Self {
        let all: Vec<&Outcome> = outcomes.iter().collect();
        let mut per_type: BTreeMap<&'static str, Vec<&Outcome>> = BTreeMap::new();
        for outcome in outcomes {
            per_type
                .entry(outcome.kind.name())
                .or_default()
                .push(outcome);
        }
        Self {
            overall: Summary::of(&all),
            per_type: per_type
                .into_iter()
                .map(|(name, outcomes)| (name, Summary::of(&outcomes)))
                .collect(),
        }
    }
}

/// `sum over bins of |accuracy - mean top probability| * share of the bin`.
fn expected_calibration_error(outcomes: &[&Outcome]) -> f64 {
    if outcomes.is_empty() {
        return f64::NAN;
    }
    let mut bins = [(0_usize, 0.0_f64, 0_usize); ECE_BINS];
    for outcome in outcomes {
        let top = outcome.top_probability();
        // Truncation is the binning: `top` is in [0, 1], and 1 goes in the
        // last bin.
        let bin = ((top * count(ECE_BINS)) as usize).min(ECE_BINS - 1);
        bins[bin].0 += 1;
        bins[bin].1 += top;
        bins[bin].2 += usize::from(outcome.correct());
    }
    bins.iter()
        .filter(|(n, _, _)| *n > 0)
        .map(|&(n, confidence, correct)| {
            let n_f = count(n);
            (count(correct) / n_f - confidence / n_f).abs() * n_f / count(outcomes.len())
        })
        .sum()
}

/// A count as `f64`. Exact below 2^53, far above any evaluation set.
fn count(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(kind: QuestionKind, probabilities: &[f64], target: usize) -> Outcome {
        let mut one_hot = vec![0.0; probabilities.len()];
        one_hot[target] = 1.0;
        Outcome::new(kind, probabilities.to_vec(), one_hot).unwrap()
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn accuracy_nll_and_brier_follow_their_definitions() {
        let outcomes = [
            outcome(QuestionKind::Choice, &[0.7, 0.2, 0.1], 0),
            outcome(QuestionKind::Choice, &[0.6, 0.4], 1),
        ];
        let metrics = Metrics::of(&outcomes);
        close(metrics.overall.accuracy, 0.5);
        close(
            metrics.overall.nll,
            (-(0.7_f64).ln() - (0.4_f64).ln()) / 2.0,
        );
        let brier_first = 0.3_f64.powi(2) + 0.2_f64.powi(2) + 0.1_f64.powi(2);
        let brier_second = 0.6_f64.powi(2) + 0.6_f64.powi(2);
        close(metrics.overall.brier, (brier_first + brier_second) / 2.0);
    }

    #[test]
    fn a_calibrated_set_has_no_calibration_error() {
        // Four answers at 0.75, three of them right: the bin's accuracy is
        // its confidence.
        let outcomes = [
            outcome(QuestionKind::Noul, &[0.25, 0.75], 1),
            outcome(QuestionKind::Noul, &[0.25, 0.75], 1),
            outcome(QuestionKind::Noul, &[0.25, 0.75], 1),
            outcome(QuestionKind::Noul, &[0.25, 0.75], 0),
        ];
        close(Metrics::of(&outcomes).overall.ece, 0.0);
        let overconfident = [
            outcome(QuestionKind::Noul, &[0.0, 1.0], 1),
            outcome(QuestionKind::Noul, &[0.0, 1.0], 0),
        ];
        close(Metrics::of(&overconfident).overall.ece, 0.5);
    }

    #[test]
    fn coverage_uses_the_published_confidence() {
        // Confidence of a two-option question: 2 * p_max - 1.
        let outcomes = [
            outcome(QuestionKind::Choice, &[0.96, 0.04], 0),
            outcome(QuestionKind::Choice, &[0.6, 0.4], 1),
        ];
        let summary = Metrics::of(&outcomes).overall;
        let at = |threshold: f64| {
            *summary
                .coverage
                .iter()
                .find(|coverage| coverage.threshold == threshold)
                .unwrap()
        };
        close(at(0.9).coverage, 0.5);
        close(at(0.9).accuracy, 1.0);
        close(at(0.95).coverage, 0.0);
        assert!(at(0.95).accuracy.is_nan());
    }

    #[test]
    fn types_are_reported_apart() {
        let outcomes = [
            outcome(QuestionKind::Choice, &[0.9, 0.1], 0),
            outcome(QuestionKind::Score, &[0.1, 0.2, 0.7], 0),
        ];
        let metrics = Metrics::of(&outcomes);
        assert_eq!(
            metrics.per_type.keys().copied().collect::<Vec<_>>(),
            ["choice", "score"]
        );
        close(metrics.per_type["choice"].accuracy, 1.0);
        close(metrics.per_type["score"].accuracy, 0.0);
    }

    #[test]
    fn a_confident_wrong_answer_costs_a_finite_loss() {
        let metrics = Metrics::of(&[outcome(QuestionKind::Choice, &[1.0, 0.0], 1)]);
        assert!(metrics.overall.nll.is_finite());
        assert!(Outcome::new(QuestionKind::Choice, vec![1.0], vec![1.0]).is_err());
    }
}
