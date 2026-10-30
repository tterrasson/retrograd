//! Teacher-forced scoring on the engine, in pure Rust over the step-wise API.
//!
//! The batched generators report logprobs of *sampled* tokens only, so the
//! engine's scoring primitives are built here from [`Engine::step`]: one row
//! per position, feeding that position's token with `sample` set (the only
//! rows whose logits the runtime exposes) and the full vocabulary as
//! `n_top_logprobs`, then reading the target token's logprob out of the
//! returned top-k. The sampled token of each row is discarded and never fed
//! back, so the slot advances by exactly the input tokens and the sequence is
//! decoded once - the same forward pass the runtime's own scorer runs.
//!
//! The scored slot is reclaimed before and after every call and no other slot
//! is touched, so a caller that keeps dormant prefixes elsewhere in the pool
//! (the rollout scheduler) only has to lend a slot it does not hold.

use std::borrow::Cow;

use super::*;
use crate::step::{StepOutput, StepRow};
use crate::{first_masked_target, validate_suffix};

/// The request id the scoring rows carry. The scheduler hands out small
/// positive ids from 1, so this one cannot collide with a generation request's
/// sampler/budget state.
const SCORING_REQUEST: i32 = i32::MAX;

/// The sampling params of a scoring row: the sampler runs (the runtime only
/// exposes logits on sampling rows) but its token is discarded, so the values
/// beyond validity do not matter. Neutral extended fields, so the top-k the
/// row reads back is the raw temperature-1 distribution.
const SCORING_SAMPLING: SamplingParams = SamplingParams {
    temperature: 1.0,
    top_p: 1.0,
    max_new_tokens: 1,
    seed: 0,
};

/// The target map of [`Engine::score_token_suffix`]: target `k` is read from
/// the distribution after `tokens[k-1]`, so entry `position` names
/// `tokens[position + 1]` when that index is a completion target.
fn suffix_targets(tokens: &[i32], n_prompt: usize) -> Vec<Option<i32>> {
    (0..tokens.len())
        .map(|position| {
            let target = position + 1;
            (target >= n_prompt && target < tokens.len()).then(|| tokens[target])
        })
        .collect()
}

impl Engine {
    /// Feeds `tokens` into `slot` one position at a time and returns the
    /// temperature-1 logprob of each target: entry `i` of `targets` is scored
    /// against the distribution after `tokens[..=i]`, and `None` entries are
    /// skipped. Returned values follow increasing position order.
    fn score_positions(
        &mut self,
        slot: i32,
        tokens: &[i32],
        targets: &[Option<i32>],
    ) -> std::result::Result<Vec<f32>, EngineError> {
        if tokens.len() != targets.len() {
            return Err(EngineError::invalid(
                "scoring targets must align with the tokens",
            ));
        }
        if tokens.is_empty() {
            return Err(EngineError::invalid("scoring requires at least one token"));
        }
        let n_ctx = self.context_size()? as usize;
        if tokens.len() > n_ctx {
            return Err(EngineError::invalid(format!(
                "sequence of {} tokens exceeds the context of {n_ctx}",
                tokens.len()
            )));
        }
        let vocab = self.vocab_size()?;
        if vocab == 0 {
            return Err(EngineError::Runtime(
                "the model reports an empty vocabulary".into(),
            ));
        }
        // One slot, reclaimed on the way out whatever happens.
        self.seq_rm(slot, -1, -1)?;
        let scored = self.score_slot(slot, tokens, targets, vocab);
        let freed = self
            .request_free(SCORING_REQUEST)
            .and_then(|()| self.seq_rm(slot, -1, -1));
        let scores = scored?;
        freed?;
        Ok(scores)
    }

    /// The body of [`Engine::score_positions`], over a slot the caller has
    /// cleared and reclaims afterwards.
    fn score_slot(
        &mut self,
        slot: i32,
        tokens: &[i32],
        targets: &[Option<i32>],
        vocab: u32,
    ) -> std::result::Result<Vec<f32>, EngineError> {
        // u32 -> usize widening.
        let mut out = StepOutput::with_capacity(1, vocab as usize);
        let mut scores = Vec::new();
        for (position, target) in targets.iter().enumerate() {
            let row = StepRow {
                request: SCORING_REQUEST,
                slot,
                tokens: Cow::Borrowed(&tokens[position..position + 1]),
                pos0: i32::try_from(position).map_err(|_| {
                    EngineError::Overflow("scoring position does not fit i32".into())
                })?,
                sample: true,
                sampling: SamplingParamsV2::from(SCORING_SAMPLING),
                n_top_logprobs: vocab,
                want_logprob: false,
            };
            self.step(std::slice::from_ref(&row), &mut out)?;
            if let &Some(target) = target {
                let logprob = out
                    .top(0)
                    .find(|&(id, _)| id == target)
                    .map(|(_, logprob)| logprob)
                    .ok_or_else(|| {
                        EngineError::Runtime(format!(
                            "the scored distribution does not contain target token {target}"
                        ))
                    })?;
                scores.push(logprob);
            }
        }
        Ok(scores)
    }

    /// Teacher-forced scores for completion targets only: `log p(tokens[k] |
    /// tokens[..k])` for `k` in `[n_prompt, tokens.len())`, in increasing
    /// target order. The engine counterpart of
    /// [`Trainer::score_token_suffix`](crate::Trainer::score_token_suffix).
    pub fn score_token_suffix(
        &mut self,
        tokens: &[i32],
        n_prompt: usize,
    ) -> std::result::Result<Vec<f32>, EngineError> {
        validate_suffix(tokens, n_prompt)?;
        self.score_positions(0, tokens, &suffix_targets(tokens, n_prompt))
    }

    /// Teacher-forced scores selected by an explicit target-token mask, the
    /// engine counterpart of
    /// [`Trainer::score_masked_tokens`](crate::Trainer::score_masked_tokens):
    /// scores from the first masked target on and returns only the masked
    /// positions, in increasing token-index order.
    pub fn score_masked_tokens(
        &mut self,
        tokens: &[i32],
        train_mask: &[bool],
    ) -> std::result::Result<Vec<f32>, EngineError> {
        self.score_masked_tokens_in(0, tokens, train_mask)
    }

    /// [`Engine::score_masked_tokens`] in a chosen pool slot, which is
    /// cleared before and after the pass; the other slots are left
    /// untouched, so a caller can keep dormant prefixes resident beside it.
    pub fn score_masked_tokens_in(
        &mut self,
        slot: i32,
        tokens: &[i32],
        train_mask: &[bool],
    ) -> std::result::Result<Vec<f32>, EngineError> {
        let first = first_masked_target(tokens, train_mask)?;
        validate_suffix(tokens, first)?;
        let suffix = self.score_positions(slot, tokens, &suffix_targets(tokens, first))?;
        Ok((first..tokens.len())
            .zip(suffix)
            .filter_map(|(target, score)| train_mask[target].then_some(score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::suffix_targets;

    #[test]
    fn suffix_targets_cover_every_completion_position() {
        assert_eq!(
            suffix_targets(&[10, 11, 12, 13, 14], 2),
            vec![None, Some(12), Some(13), Some(14), None],
            "targets 2..len are scored, read from positions 1..len-1"
        );
    }
}

// The SFT evaluation and the dataset-backend impl need the dataset crate, so
// they sit behind the `train` feature like their `Trainer` counterparts; the
// inference path does not activate it.
#[cfg(feature = "train")]
mod train_surface {
    use super::super::{Engine, EngineError};
    use crate::parse_assistant_output;
    use retrograd_core::EvalMetrics;
    use retrograd_dataset::{DatasetBackend, IGNORE_LABEL, PreparedDataset};

    impl Engine {
        /// Forward-only SFT evaluation over a prepared dataset: the mean
        /// negative log-likelihood of every non-ignored label, scored
        /// teacher-forced through the step-wise API. The engine counterpart of
        /// [`Trainer::eval_sft`](crate::Trainer::eval_sft).
        pub fn eval_sft(
            &mut self,
            data: &PreparedDataset,
        ) -> std::result::Result<EvalMetrics, EngineError> {
            if data.is_empty() || data.n_ctx == 0 {
                return Err(EngineError::invalid("evaluation dataset must contain rows"));
            }
            let expected = data.rows().checked_mul(data.n_ctx).ok_or_else(|| {
                EngineError::Overflow("evaluation dataset shape overflows usize".into())
            })?;
            if data.tokens.len() != expected || data.labels.len() != expected {
                return Err(EngineError::invalid(
                    "invalid prepared evaluation dataset shape",
                ));
            }
            let n_ctx = self.context_size()? as usize;
            if data.n_ctx > n_ctx {
                return Err(EngineError::invalid(format!(
                    "evaluation row context {} exceeds the model context {n_ctx}",
                    data.n_ctx
                )));
            }
            let mut negative_log_likelihood = 0.0_f64;
            let mut supervised_tokens = 0_u64;
            for row in 0..data.rows() {
                let start = row * data.n_ctx;
                let end = start + data.n_ctx;
                let tokens = &data.tokens[start..end];
                // Entry i is the label scored against the distribution after
                // tokens[..=i]; IGNORE_LABEL positions carry no loss.
                let targets = data.labels[start..end]
                    .iter()
                    .map(|&label| (label != IGNORE_LABEL).then_some(label))
                    .collect::<Vec<_>>();
                for logprob in self.score_positions(0, tokens, &targets)? {
                    negative_log_likelihood -= f64::from(logprob);
                    supervised_tokens += 1;
                }
            }
            if supervised_tokens == 0 {
                return Err(EngineError::invalid(
                    "evaluation dataset contains no supervised tokens",
                ));
            }
            Ok(EvalMetrics {
                negative_log_likelihood,
                supervised_tokens,
            })
        }
    }

    impl DatasetBackend for Engine {
        fn tokenize_text(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
            Ok(Engine::tokenize_text(self, text)?)
        }

        fn eos_token(&self) -> retrograd_core::Result<i32> {
            Ok(Engine::eos_token(self)?)
        }

        fn format_chat(
            &self,
            messages: &[(&str, &str)],
            add_assistant: bool,
        ) -> retrograd_core::Result<String> {
            Ok(Engine::format_chat(self, messages, add_assistant)?)
        }

        fn tokenize_fragment(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
            Ok(Engine::tokenize_fragment(self, text)?)
        }

        fn is_eog_token(&self, token: i32) -> retrograd_core::Result<bool> {
            Ok(Engine::is_eog_token(self, token)?)
        }

        fn format_chat_messages(
            &self,
            messages_json: &str,
            tools_json: Option<&str>,
            add_assistant: bool,
        ) -> retrograd_core::Result<String> {
            Ok(Engine::format_chat_messages(
                self,
                messages_json,
                tools_json,
                add_assistant,
            )?)
        }

        fn chat_template_supports_tools(&self) -> retrograd_core::Result<bool> {
            Ok(Engine::chat_template_supports_tools(self)?)
        }

        fn tool_call_parser(&self, tools_json: &str) -> retrograd_core::Result<Option<String>> {
            Ok(Engine::tool_call_parser(self, Some(tools_json))?)
        }

        fn parse_assistant(&self, parser: &str, text: &str) -> retrograd_core::Result<String> {
            parse_assistant_output(parser, text)
        }
    }
}
