//! Fitting the temperatures of a model on a validation set.
//!
//! One temperature per `(type, bucket)` key, the same keys the answer reads
//! ([`crate::answer::temperature_key`]). Each is the minimizer of the mean
//! negative log-likelihood of the targets under `softmax(scores / T)`. The
//! NLL is convex in `1 / T`, so it is unimodal in `ln T`: a golden-section
//! search on `ln T` finds the minimum with no dependency.

use std::collections::BTreeMap;

use crate::DecisionError;
use crate::answer::temperature_key;
use crate::readout::Readout;
use crate::request::QuestionKind;

/// The temperatures the search considers.
const MIN_TEMPERATURE: f64 = 0.01;
const MAX_TEMPERATURE: f64 = 100.0;

/// The search stops when the interval on `ln T` is this narrow.
const LOG_TOLERANCE: f64 = 1e-10;

const MAX_ITERATIONS: usize = 200;

/// `1 / golden ratio`.
const INVERSE_PHI: f64 = 0.618_033_988_749_894_9;

/// What a model said about one question, and what was right.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    key: String,
    scores: Vec<f64>,
    target: Vec<f64>,
}

impl Sample {
    /// `scores` are the raw outputs of the model for the options of a question
    /// of `kind` with `n_options` options, `target` a probability for each (one
    /// hot, or soft); it is normalized to sum to 1.
    pub fn new(
        readout: &Readout,
        kind: QuestionKind,
        n_options: usize,
        scores: Vec<f64>,
        mut target: Vec<f64>,
    ) -> Result<Self, DecisionError> {
        let fail = |message: &str| DecisionError::Calibration(message.to_owned());
        let n_outputs = match readout {
            Readout::LabelLogits {
                noul_scale: Some(scale),
                ..
            } if kind == QuestionKind::Noul => *scale,
            _ => n_options,
        };
        if n_options == 0 || n_options > readout.n_options_max() || scores.len() != n_outputs {
            return Err(fail("scores must match the question's readout"));
        }
        if scores.len() < 2 || scores.len() != target.len() {
            return Err(fail("a sample needs 2 or more scores and as many targets"));
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(fail("scores must be finite"));
        }
        if target.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(fail("targets must be finite and not negative"));
        }
        let sum: f64 = target.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(fail(
                "a target must have a finite positive total probability",
            ));
        }
        target.iter_mut().for_each(|p| *p /= sum);
        Ok(Self {
            key: temperature_key(readout, kind, n_options),
            scores,
            target,
        })
    }

    /// The temperature this sample informs, `<type>.<bucket>`.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// `-sum(target * ln softmax(scores / T))`.
    fn nll(&self, temperature: f64) -> f64 {
        let max = self
            .scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = self
            .scores
            .iter()
            .map(|score| (score - max) / temperature)
            .collect();
        let log_sum = scaled.iter().map(|x| x.exp()).sum::<f64>().ln();
        self.target
            .iter()
            .zip(&scaled)
            .map(|(p, x)| -p * (x - log_sum))
            .sum()
    }
}

/// The result of one fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub temperature: f64,
    /// Mean NLL at T = 1.
    pub nll_before: f64,
    /// Mean NLL at `temperature`.
    pub nll_after: f64,
    pub n_samples: usize,
}

/// Fits the temperature of samples that share one key. The result stays within
/// [0.01, 100]: scores that separate the targets perfectly want a temperature
/// of 0 and get the lower bound.
pub fn fit_temperature(samples: &[Sample]) -> Result<Fit, DecisionError> {
    if samples.is_empty() {
        return Err(DecisionError::Calibration("no sample to fit".into()));
    }
    if samples.iter().any(|sample| sample.key != samples[0].key) {
        return Err(DecisionError::Calibration(
            "samples must share one temperature key".into(),
        ));
    }
    let mean_nll = |temperature: f64| {
        samples
            .iter()
            .map(|sample| sample.nll(temperature))
            .sum::<f64>()
            / count(samples.len())
    };
    let objective = |log_t: f64| mean_nll(log_t.exp());

    let (mut low, mut high) = (MIN_TEMPERATURE.ln(), MAX_TEMPERATURE.ln());
    let mut c = high - (high - low) * INVERSE_PHI;
    let mut d = low + (high - low) * INVERSE_PHI;
    let (mut f_c, mut f_d) = (objective(c), objective(d));
    for _ in 0..MAX_ITERATIONS {
        if high - low < LOG_TOLERANCE {
            break;
        }
        // A tie is a plateau of zero loss (perfectly separated scores): keep
        // the lower side, the temperature that needs the least softening.
        if f_c <= f_d {
            high = d;
            d = c;
            f_d = f_c;
            c = high - (high - low) * INVERSE_PHI;
            f_c = objective(c);
        } else {
            low = c;
            c = d;
            f_c = f_d;
            d = low + (high - low) * INVERSE_PHI;
            f_d = objective(d);
        }
    }
    let temperature = ((low + high) / 2.0).exp();
    Ok(Fit {
        temperature,
        nll_before: mean_nll(1.0),
        nll_after: mean_nll(temperature),
        n_samples: samples.len(),
    })
}

/// Fits one temperature per key of `samples`.
pub fn fit_all(samples: &[Sample]) -> Result<BTreeMap<String, Fit>, DecisionError> {
    let mut groups: BTreeMap<&str, Vec<Sample>> = BTreeMap::new();
    for sample in samples {
        groups.entry(sample.key()).or_default().push(sample.clone());
    }
    groups
        .into_iter()
        .map(|(key, group)| Ok((key.to_owned(), fit_temperature(&group)?)))
        .collect()
}

/// A sample count as `f64`. Exact below 2^53, far above any validation set.
fn count(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::softmax;

    fn laya() -> Readout {
        Readout::MarkerColumn {
            marker: 4,
            sep: 1,
            marker_text: "<mask>".into(),
        }
    }

    /// Scores that vary from sample to sample, deterministically. The value
    /// cast is below 11, exact.
    fn scores(seed: usize, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (((seed * 7 + i * 13) % 11) as f64 - 5.0) * 0.8)
            .collect()
    }

    #[test]
    fn a_model_that_matches_its_targets_at_t_recovers_t() {
        // Soft targets are exactly softmax(scores / 2.5): cross-entropy is
        // minimal where the model's distribution equals the target.
        let samples: Vec<Sample> = (0..40)
            .map(|seed| {
                let s = scores(seed, 4);
                let target = softmax(&s, 2.5);
                Sample::new(&laya(), QuestionKind::Choice, 4, s, target).expect("sample")
            })
            .collect();
        let fit = fit_temperature(&samples).expect("fit");
        assert!((fit.temperature - 2.5).abs() < 1e-4, "{}", fit.temperature);
        assert!(fit.nll_after < fit.nll_before);
        assert_eq!(fit.n_samples, 40);
    }

    #[test]
    fn overconfident_scores_that_are_often_wrong_get_a_higher_temperature() {
        // Large margins, but the labelled option is the best one only 3 times
        // in 5: softmax(scores) is too sure, the fit must cool it down.
        let samples: Vec<Sample> = (0..50)
            .map(|i| {
                let s = vec![6.0, 0.0, 0.0];
                let right = if i % 5 < 3 { 0 } else { 1 };
                let mut target = vec![0.0; 3];
                target[right] = 1.0;
                Sample::new(&laya(), QuestionKind::Choice, 3, s, target).expect("sample")
            })
            .collect();
        let fit = fit_temperature(&samples).expect("fit");
        assert!(fit.temperature > 3.0, "{}", fit.temperature);
        assert!(fit.nll_after < fit.nll_before);
        // At the optimum the model predicts the observed frequency, 60 %.
        let p = softmax(&[6.0, 0.0, 0.0], fit.temperature)[0];
        assert!((p - 0.6).abs() < 1e-3, "{p}");
    }

    #[test]
    fn perfectly_separated_scores_stop_at_the_lower_bound() {
        let sample = Sample::new(
            &laya(),
            QuestionKind::Choice,
            2,
            vec![5.0, 0.0],
            vec![1.0, 0.0],
        )
        .expect("sample");
        let fit = fit_temperature(&[sample]).expect("fit");
        assert!((fit.temperature - MIN_TEMPERATURE).abs() < 1e-6);
    }

    #[test]
    fn samples_are_fitted_per_type_and_bucket() {
        let mut samples = Vec::new();
        // choice with 2 options (bucket 2) is calibrated at 0.5,
        // choice with 4 options (bucket 3_5) at 3, noul at 1.5.
        for (kind, n, t) in [
            (QuestionKind::Choice, 2, 0.5),
            (QuestionKind::Choice, 4, 3.0),
            (QuestionKind::Noul, 2, 1.5),
        ] {
            for seed in 0..20 {
                let s = scores(seed, n);
                samples.push(
                    Sample::new(&laya(), kind, n, s.clone(), softmax(&s, t)).expect("sample"),
                );
            }
        }
        let fits = fit_all(&samples).expect("fits");
        let keys: Vec<_> = fits.keys().map(String::as_str).collect();
        assert_eq!(keys, ["choice.2", "choice.3_5", "noul.2"]);
        assert!((fits["choice.2"].temperature - 0.5).abs() < 1e-4);
        assert!((fits["choice.3_5"].temperature - 3.0).abs() < 1e-4);
        assert!((fits["noul.2"].temperature - 1.5).abs() < 1e-4);
    }

    #[test]
    fn targets_are_normalized_and_bad_samples_refused() {
        let ok = |target: Vec<f64>| {
            Sample::new(&laya(), QuestionKind::Choice, 2, vec![1.0, 0.0], target)
        };
        let scaled = ok(vec![2.0, 2.0]).expect("sample");
        assert_eq!(scaled.target, [0.5, 0.5]);
        assert!(ok(vec![0.0, 0.0]).is_err());
        assert!(ok(vec![-1.0, 2.0]).is_err());
        assert!(ok(vec![1.0]).is_err());
        assert!(ok(vec![f64::MAX, f64::MAX]).is_err());
        assert!(
            Sample::new(
                &laya(),
                QuestionKind::Choice,
                3,
                vec![1.0, 0.0],
                vec![1.0, 0.0],
            )
            .is_err()
        );
        assert!(
            Sample::new(
                &laya(),
                QuestionKind::Choice,
                2,
                vec![f64::NAN, 0.0],
                vec![1.0, 0.0]
            )
            .is_err()
        );
        assert!(matches!(
            fit_temperature(&[]),
            Err(DecisionError::Calibration(_))
        ));
    }

    #[test]
    fn one_fit_cannot_mix_temperature_keys() {
        let samples: Vec<_> = [QuestionKind::Choice, QuestionKind::Noul]
            .into_iter()
            .map(|kind| {
                Sample::new(&laya(), kind, 2, vec![1.0, 0.0], vec![1.0, 0.0]).expect("sample")
            })
            .collect();
        assert!(matches!(
            fit_temperature(&samples),
            Err(DecisionError::Calibration(_))
        ));
    }
}
