//! The held-out pass: the same terms the step computes, over the evaluation
//! pairs, with nothing trained.

use retrograd_config::PreferenceLoss;
use retrograd_core::{Error, Result};
use retrograd_engine::Trainer;
use retrograd_metrics::MetricValue;

use super::loss::{SideScores, pair_terms};
use super::{PreparedPreference, ReferenceTable, score_pair};

/// What an evaluation measured, averaged over the evaluation pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreferenceEval {
    /// The loss the run minimizes, which is what early stopping compares.
    pub loss: f64,
    /// Share of pairs whose chosen response earns the larger implicit reward.
    pub accuracy: f64,
    pub margin: f64,
    pub chosen_logps: f64,
    pub rejected_logps: f64,
    pub examples: usize,
}

impl PreferenceEval {
    pub fn values(&self) -> Vec<MetricValue> {
        [
            ("eval/preference_loss", self.loss),
            ("eval/accuracy", self.accuracy),
            ("eval/margin", self.margin),
            ("eval/chosen_logps", self.chosen_logps),
            ("eval/rejected_logps", self.rejected_logps),
        ]
        .into_iter()
        .map(|(name, value)| MetricValue {
            name: name.into(),
            value: value as f32,
        })
        .collect()
    }
}

/// Scores every evaluation pair under the current policy.
pub fn evaluate(
    trainer: &mut Trainer,
    prepared: &PreparedPreference,
    reference: Option<&ReferenceTable>,
    loss: &PreferenceLoss,
) -> Result<PreferenceEval> {
    if prepared.eval.is_empty() {
        return Err(Error::invalid(
            "this run has no evaluation pairs, so there is nothing to evaluate",
        ));
    }
    let mut result = PreferenceEval {
        examples: prepared.eval.len(),
        ..PreferenceEval::default()
    };
    for (index, pair) in prepared.eval.iter().enumerate() {
        let current = score_pair(trainer, pair)?;
        let reference = reference.map(|table| table.eval[index]);
        let [chosen, rejected] = [0, 1].map(|side| SideScores {
            sum: current[side],
            tokens: pair.sides()[side].completion_len(),
            reference: reference.map(|reference| reference[side]),
        });
        let terms = pair_terms(loss, chosen, rejected)?;
        let margin = terms.chosen_reward - terms.rejected_reward;
        result.loss += terms.loss;
        result.margin += margin;
        result.accuracy += f64::from(u8::from(margin > 0.0));
        result.chosen_logps += chosen.sum / chosen.tokens as f64;
        result.rejected_logps += rejected.sum / rejected.tokens as f64;
    }
    let count = result.examples as f64;
    result.loss /= count;
    result.accuracy /= count;
    result.margin /= count;
    result.chosen_logps /= count;
    result.rejected_logps /= count;
    Ok(result)
}
