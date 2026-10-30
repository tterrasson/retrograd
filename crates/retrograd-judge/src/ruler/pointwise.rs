//! `pointwise_logprob`: no generation, one score per completion read off the
//! judge's own distribution.
//!
//! Each trajectory is presented alone with a fixed question ("Is this answer
//! correct? Answer yes or no."), and the verdict is the probability the judge
//! puts on `yes` against `no` at the next position. There is no sampling, so
//! the result is deterministic on a fixed model - which is also why this
//! strategy is served by the in-process transport only: an OpenAI-compatible
//! endpoint does not return arbitrary target logprobs reliably.
//!
//! The verdict is `P(yes) / (P(yes) + P(no))`, computed in log-space so two
//! very unlikely labels do not underflow to `0/0`. GRPO normalizes inside the
//! group, so the absolute value matters little and the ranking is what counts.

use retrograd_agent_core::Result;

use crate::Score;
use crate::render::prompt::{self, POINTWISE_LABELS, RenderedGroup};
use crate::transport::{JudgeTransport, ScoreRequest};

use super::{RulerCore, cache_key};

/// `P(yes) / (P(yes) + P(no))` from the two label log-probabilities, via
/// `logsumexp`: `exp` of the difference against the max, so the softmax is
/// stable however negative both logprobs are.
pub fn verdict(logprob_yes: f32, logprob_no: f32) -> f32 {
    let max = logprob_yes.max(logprob_no);
    let yes = (logprob_yes - max).exp();
    let no = (logprob_no - max).exp();
    yes / (yes + no)
}

impl RulerCore {
    /// Scores every trajectory of the group through the transport's scoring
    /// entry point: one request per trajectory, one verdict per request.
    ///
    /// The cache works as for generation-based strategies - the key is the
    /// prompt, which here carries the whole trajectory - but a scoring
    /// request never degrades its wire form on retry: there is no response
    /// format to fall back to, so a failed scoring is simply retried as is.
    pub(super) fn score_pointwise_via(
        &self,
        transport: &mut dyn JudgeTransport,
        rendered: &RenderedGroup,
    ) -> Result<Vec<Score>> {
        let rubric = rendered
            .rubric
            .as_deref()
            .or(self.config.rubric.as_deref())
            .unwrap_or("Evaluate whether the answer achieves the task's goal.");
        let mut requests = Vec::with_capacity(rendered.trajectories.len());
        let mut keys = Vec::with_capacity(rendered.trajectories.len());
        let mut cached = vec![None; rendered.trajectories.len()];
        for (index, trajectory) in rendered.trajectories.iter().enumerate() {
            let prompt = prompt::pointwise_prompt(rubric, rendered, trajectory.id)?;
            let key = cache_key(&self.cache_identity(), &prompt);
            if let Some(scores) = self.cached(&key).filter(|scores| scores.len() == 1) {
                cached[index] = scores.into_iter().next();
            }
            requests.push(ScoreRequest {
                prompt,
                targets: POINTWISE_LABELS
                    .iter()
                    .map(|label| label.to_string())
                    .collect(),
            });
            keys.push(key);
        }
        let pending: Vec<usize> = (0..requests.len())
            .filter(|&index| cached[index].is_none())
            .collect();
        if !pending.is_empty() {
            let wave: Vec<ScoreRequest> = pending
                .iter()
                .map(|&index| requests[index].clone())
                .collect();
            let results = self.run_wave(
                transport,
                wave,
                |request, _attempt| request.clone(),
                |request| request.prompt.as_str(),
                |transport, wave| transport.score(wave),
                |_, request, logprobs| parse_pointwise(request, logprobs),
            );
            for (&index, result) in pending.iter().zip(results) {
                // One failed request fails the group, exactly as a failed
                // generation does: the failure policy decides what that
                // costs, and a group is judged whole or not at all.
                let score = result?;
                self.store(keys[index].clone(), std::slice::from_ref(&score))?;
                cached[index] = Some(score);
            }
        }
        Ok(cached
            .into_iter()
            .map(|entry| entry.expect("every trajectory got a verdict"))
            .collect())
    }
}

/// Turns one request's logprobs into its verdict: the first two targets are
/// the labels, and a transport that returned another count failed the request.
fn parse_pointwise(request: &ScoreRequest, logprobs: Vec<f32>) -> Result<Score> {
    if logprobs.len() != request.targets.len() {
        return Err(retrograd_agent_core::Error::Reward(format!(
            "the transport returned {} logprobs for {} targets",
            logprobs.len(),
            request.targets.len()
        )));
    }
    if logprobs.len() < 2 {
        return Err(retrograd_agent_core::Error::Reward(
            "pointwise scoring needs at least the two labels 'yes' and 'no'".to_owned(),
        ));
    }
    let value = verdict(logprobs[0], logprobs[1]);
    if !value.is_finite() {
        return Err(retrograd_agent_core::Error::Reward(
            "the pointwise verdict is not finite: the judge returned degenerate logprobs"
                .to_owned(),
        ));
    }
    Ok(Score {
        value,
        valid: true,
        explanation: Some(format!(
            "log P({})={:.4}, log P({})={:.4}",
            request.targets[0], logprobs[0], request.targets[1], logprobs[1]
        )),
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_verdict_is_a_log_space_softmax_over_the_two_labels() {
        assert!((verdict(0.0, 0.0) - 0.5).abs() < 1e-6);
        // log 4 vs log 1 -> 0.8
        assert!((verdict(4.0_f32.ln(), 0.0) - 0.8).abs() < 1e-6);
        // Deeply negative logprobs must not underflow to NaN.
        let value = verdict(-1000.0, -1001.0);
        assert!(value.is_finite());
        assert!(value > 0.7 && value < 0.75, "{value}");
    }
}
