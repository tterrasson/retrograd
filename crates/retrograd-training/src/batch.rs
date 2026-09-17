//! Training entry point for trajectories collected outside the built-in
//! single-turn GRPO sampler.

use std::sync::Arc;
use std::time::Instant;

use super::Progress;
use super::grpo::group_advantages;
use super::rollout::{
    EpochBatch, EpochState, GrpoObjective, GrpoStepParams, Rollout, RowLayout, TokenStats,
    WeightedStepScratch, check_policy_divergence, is_truncated, mean_std, run_grpo_epoch,
    score_train_mask,
};
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_metrics::MetricValue;
use retrograd_observe::{ObserveBatch, SelectionEntry, SkipReason, TrajectoryObserver};

/// A pre-generated on-policy sequence ready for a GRPO update.
#[derive(Clone, Debug)]
pub struct TrainSequence {
    /// Complete token sequence, including context, policy actions and tool
    /// observations.
    pub tokens: Vec<i32>,
    /// Behavior-policy logprobs in increasing token-index order for exactly
    /// the positions where `train_mask` is true.
    pub old_logprobs: Vec<f32>,
    /// Token-level policy mask. The first token must always be false.
    pub train_mask: Vec<bool>,
    /// Scalar final reward for this trajectory.
    pub reward: f32,
    /// Explicit group identity. Equal ids share a group even when rows are not
    /// contiguous or group sizes differ.
    pub group_id: u64,
    /// Optional undiscounted intermediate return per trainable token. Empty
    /// means terminal-reward-only GRPO; otherwise values align with the true
    /// positions of `train_mask`.
    pub intermediate_returns: Vec<f32>,
}

impl crate::TokenSpan for TrainSequence {
    fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    fn train_mask(&self) -> &[bool] {
        &self.train_mask
    }
}

impl TrainSequence {
    /// Validates collection-time invariants that do not depend on a model.
    pub fn validate(&self) -> Result<()> {
        if self.tokens.len() != self.train_mask.len() {
            return Err(Error::invalid(
                "TrainSequence tokens and train_mask lengths do not match",
            ));
        }
        if self.tokens.len() < 2 {
            return Err(Error::tokenize(
                "TrainSequence requires at least two tokens",
            ));
        }
        if self.train_mask[0] {
            return Err(Error::invalid(
                "the first token of a TrainSequence cannot be trainable",
            ));
        }
        let trained_tokens = self.train_mask.iter().filter(|&&train| train).count();
        if trained_tokens == 0 {
            return Err(Error::invalid(
                "TrainSequence requires at least one trainable token",
            ));
        }
        if self.old_logprobs.len() != trained_tokens {
            return Err(Error::invalid(format!(
                "TrainSequence has {} old_logprobs for {trained_tokens} trainable tokens",
                self.old_logprobs.len()
            )));
        }
        if self.old_logprobs.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid(
                "TrainSequence old_logprobs must all be finite",
            ));
        }
        if !self.intermediate_returns.is_empty()
            && self.intermediate_returns.len() != trained_tokens
        {
            return Err(Error::invalid(format!(
                "TrainSequence has {} intermediate_returns for {trained_tokens} trainable tokens",
                self.intermediate_returns.len()
            )));
        }
        if self
            .intermediate_returns
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::invalid(
                "TrainSequence intermediate_returns must all be finite",
            ));
        }
        if !self.reward.is_finite() {
            return Err(Error::invalid("TrainSequence reward must be finite"));
        }
        Ok(())
    }
}

/// Optimizer parameters for one pre-generated GRPO batch.
#[derive(Clone, Copy, Debug)]
pub struct GrpoBatchParams {
    pub epochs: u32,
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    pub kl_coefficient: f32,
    /// Constant Dr. GRPO token budget used as the loss denominator.
    pub loss_denominator: usize,
    pub seed: u64,
    /// Number of rollout slots the *whole run* will optimize
    /// (`updates × sequences_per_update`), used to size the learning-rate
    /// horizon exactly like the GRPO and PPO CLI paths do.
    ///
    /// The runtime's scheduler step counter accumulates across calls, so a
    /// horizon derived from a single batch would place the second update at
    /// the end of the decay and drive the learning rate to zero. `None` keeps
    /// the single-batch horizon and is only accepted under
    /// [`LrScheduler::Constant`](retrograd_core::LrScheduler::Constant), where the
    /// horizon is unused.
    pub scheduler_total_rollouts: Option<u64>,
}

impl GrpoBatchParams {
    fn validate(&self) -> Result<()> {
        if self.epochs == 0 {
            return Err(Error::invalid(
                "GRPO batch epochs must be greater than zero",
            ));
        }
        if !self.clip_range_low.is_finite()
            || self.clip_range_low <= 0.0
            || self.clip_range_low >= 1.0
            || !self.clip_range_high.is_finite()
            || self.clip_range_high <= 0.0
            || self.clip_range_high >= 1.0
        {
            return Err(Error::invalid(
                "GRPO batch clip ranges must be finite and strictly between 0 and 1",
            ));
        }
        if !self.kl_coefficient.is_finite() || self.kl_coefficient < 0.0 {
            return Err(Error::invalid(
                "GRPO batch kl_coefficient must be finite and non-negative",
            ));
        }
        if self.loss_denominator == 0 {
            return Err(Error::invalid(
                "GRPO batch loss_denominator must be greater than zero",
            ));
        }
        if self.scheduler_total_rollouts == Some(0) {
            return Err(Error::invalid(
                "GRPO batch scheduler_total_rollouts must be greater than zero",
            ));
        }
        Ok(())
    }
}

fn metric(name: &'static str, value: f32) -> MetricValue {
    MetricValue {
        name: name.into(),
        value,
    }
}

/// Spreads a trajectory advantage over its policy tokens using the
/// undiscounted intermediate returns. Centering on the trajectory's own mean
/// return keeps the average token advantage equal to the group-relative
/// advantage, so the Dr. GRPO scale is untouched and only the *within*-
/// trajectory credit moves: earlier tokens, which carry more remaining return,
/// are rewarded above the trajectory baseline and later ones below it.
fn token_advantages(advantage: f32, intermediate_returns: &[f32]) -> Vec<f32> {
    let mean = intermediate_returns
        .iter()
        .map(|&value| value as f64)
        .sum::<f64>()
        / intermediate_returns.len() as f64;
    intermediate_returns
        .iter()
        .map(|&value| advantage + (value as f64 - mean) as f32)
        .collect()
}

/// Where a batch's selection and outcome are exported, and who each sequence
/// is in the update that collected it.
#[derive(Clone)]
pub struct BatchObservation {
    pub observer: Arc<dyn TrajectoryObserver>,
    /// One-based.
    pub update: u32,
    /// `(group, member)` of each sequence, in the order of the batch.
    pub members: Vec<(usize, usize)>,
}

/// Trains one immutable-policy GRPO batch with explicit token masks and
/// group ids. Collection and reward scoring happen before this call; no
/// policy generation occurs while the batch is being optimized.
pub fn train_grpo_batch(
    trainer: &mut Trainer,
    sequences: &[TrainSequence],
    params: &GrpoBatchParams,
    training: &TrainConfig,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    train_grpo_batch_observed(trainer, sequences, params, training, None, on_progress)
}

/// [`train_grpo_batch`], publishing the advantages and the effective mask
/// before the epochs, and the outcome once they all succeeded.
pub fn train_grpo_batch_observed(
    trainer: &mut Trainer,
    sequences: &[TrainSequence],
    params: &GrpoBatchParams,
    training: &TrainConfig,
    observation: Option<&BatchObservation>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    params.validate()?;
    if let Some(observation) = observation
        && observation.members.len() != sequences.len()
    {
        return Err(Error::invalid(format!(
            "the batch observation names {} sequences for a batch of {}",
            observation.members.len(),
            sequences.len()
        )));
    }
    if sequences.is_empty() {
        return Err(Error::invalid("GRPO batch must not be empty"));
    }

    let layout = RowLayout::resolve(trainer, training)?;
    let vocab_size = trainer.vocab_size()?;
    for (index, sequence) in sequences.iter().enumerate() {
        sequence
            .validate()
            .map_err(|error| Error::invalid(format!("sequence {index}: {error}")))?;
        if sequence.tokens.len() > layout.window {
            return Err(Error::invalid(format!(
                "sequence {index} of {} tokens exceeds the trained window {}",
                sequence.tokens.len(),
                layout.window
            )));
        }
        if let Some((position, token)) = sequence
            .tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token < 0 || *token as usize >= vocab_size)
        {
            return Err(Error::invalid(format!(
                "sequence {index} token {position} ({token}) is outside vocabulary size {vocab_size}"
            )));
        }
    }

    // The runtime's scheduler step accumulates across calls, so the horizon
    // must cover the whole run and not this batch. Without it, a decaying
    // schedule would reach zero on the second update.
    let horizon_rollouts = match params.scheduler_total_rollouts {
        Some(rollouts) => rollouts,
        None if training.lr_scheduler == retrograd_core::LrScheduler::Constant => {
            sequences.len() as u64
        }
        None => {
            return Err(Error::invalid(
                "a decaying learning-rate schedule needs GrpoBatchParams::scheduler_total_rollouts \
                 (updates × sequences per update): the per-batch horizon would collapse the \
                 learning rate to zero after the first update",
            ));
        }
    };
    let scheduler_total_steps = (params.epochs as u64)
        .checked_mul(horizon_rollouts)
        .and_then(|steps| steps.checked_mul(layout.steps_per_row))
        .ok_or_else(|| Error::overflow("GRPO batch optimizer step count overflows u64"))?;
    let step_params = GrpoStepParams {
        objective: GrpoObjective {
            clip_range_low: params.clip_range_low,
            clip_range_high: params.clip_range_high,
            kl_coefficient: params.kl_coefficient,
            loss_denominator: params.loss_denominator,
        },
        scheduler_total_steps,
    };

    let rewards = sequences
        .iter()
        .map(|sequence| sequence.reward)
        .collect::<Vec<_>>();
    let group_ids = sequences
        .iter()
        .map(|sequence| sequence.group_id)
        .collect::<Vec<_>>();
    let mut live = vec![true; sequences.len()];
    let (advantages, group_diagnostics) = group_advantages(
        &rewards,
        &group_ids,
        &mut live,
        retrograd_config::AdvantageBaseline::Mean,
    )?;
    let trainable = (0..sequences.len())
        .filter(|&index| live[index])
        .collect::<Vec<_>>();
    if let Some(observation) = observation {
        observation.observer.observe(ObserveBatch::Selection {
            update: observation.update,
            entries: selection_entries(&observation.members, &advantages, &live),
        });
    }

    let rollouts = sequences
        .iter()
        .map(|sequence| Rollout {
            tokens: sequence.tokens.clone(),
            train_mask: sequence.train_mask.clone(),
            old_logprobs: sequence.old_logprobs.clone(),
        })
        .collect::<Vec<_>>();
    let token_advantages = sequences
        .iter()
        .enumerate()
        .map(|(index, sequence)| {
            if sequence.intermediate_returns.is_empty() {
                return None;
            }
            Some(token_advantages(
                advantages[index],
                &sequence.intermediate_returns,
            ))
        })
        .collect::<Vec<_>>();

    // The reference policy is fixed for every epoch. Only live rows are
    // scored; dead group slots advance the scheduler without an update. With a
    // zero coefficient the reference term is identically zero, so the whole
    // teacher-forced pass is skipped and members carry an empty slice.
    let reference_rows = if params.kl_coefficient != 0.0 {
        trainer.with_lora_disabled(|trainer| {
            trainable
                .iter()
                .map(|&index| {
                    score_train_mask(
                        trainer,
                        &sequences[index].tokens,
                        &sequences[index].train_mask,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        Vec::new()
    };
    let mut reference_positions = vec![None; sequences.len()];
    for (position, &index) in trainable.iter().enumerate() {
        reference_positions[index] = Some(position);
    }

    let (reward_mean, reward_std, _) = mean_std(rewards.iter().copied());
    let reward_min = rewards.iter().copied().fold(f32::INFINITY, f32::min);
    let reward_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let trained_tokens = trainable
        .iter()
        .map(|&index| rollouts[index].completion_len())
        .sum::<usize>();
    let trained_fraction = trainable.len() as f32 / sequences.len() as f32;
    let advantage_abs_mean = if trainable.is_empty() {
        0.0
    } else {
        trainable
            .iter()
            .map(|&index| advantages[index].abs() as f64)
            .sum::<f64>()
            / trainable.len() as f64
    };

    // Mean sampled-token entropy proxy, read straight off the behavior
    // logprobs: the entropy-collapse symptom Clip-Higher is meant to fight.
    let entropy = if trainable.is_empty() {
        0.0
    } else {
        let sum: f64 = trainable
            .iter()
            .map(|&index| {
                let logprobs = &rollouts[index].old_logprobs;
                if logprobs.is_empty() {
                    0.0
                } else {
                    -logprobs.iter().map(|&value| value as f64).sum::<f64>() / logprobs.len() as f64
                }
            })
            .sum();
        (sum / trainable.len() as f64) as f32
    };
    let completion_lengths = rollouts
        .iter()
        .map(|rollout| rollout.completion_len())
        .collect::<Vec<_>>();
    let completion_length_mean =
        completion_lengths.iter().sum::<usize>() as f32 / completion_lengths.len() as f32;
    let completion_length_min = completion_lengths.iter().copied().min().unwrap_or(0) as f32;
    let completion_length_max = completion_lengths.iter().copied().max().unwrap_or(0) as f32;
    let truncation_fraction = completion_lengths
        .iter()
        .filter(|&&length| is_truncated(length, params.loss_denominator))
        .count() as f32
        / completion_lengths.len() as f32;

    let mut final_metrics = TrainMetrics::default();
    let mut scratch = WeightedStepScratch::new(&layout);
    let rollout_evals = rollouts
        .iter()
        .map(|rollout| layout.rollout_evals(rollout))
        .collect::<Result<Vec<_>>>()?;
    let epoch_batch = EpochBatch {
        rollouts: &rollouts,
        group_ids: &group_ids,
        advantages: &advantages,
        token_advantages: &token_advantages,
        reference_positions: &reference_positions,
        reference_rows: &reference_rows,
        rollout_evals: &rollout_evals,
    };
    // Trajectories arrive from an external collector, whose logprobs need not
    // come from this scoring path: every step re-scores under the current
    // policy rather than trusting `old_logprobs` for the first one.
    let mut reuse_behavior_logprobs = false;
    for epoch in 0..params.epochs {
        let mut epoch_stats = TokenStats::default();
        let order =
            super::grpo::shuffled_indices(sequences.len(), params.seed ^ (epoch as u64 + 1));
        let started = Instant::now();
        let optimizer_timing_before = trainer.optimizer_timing()?;
        run_grpo_epoch(
            trainer,
            &epoch_batch,
            &order,
            &layout,
            step_params,
            EpochState {
                reuse_behavior_logprobs: &mut reuse_behavior_logprobs,
                scratch: &mut scratch,
                metrics: &mut final_metrics,
                stats: &mut epoch_stats,
            },
            &mut |_, _| Ok(true),
        )?;

        let denominator = trainable.len().max(1) as f32;
        let surrogate_loss = epoch_stats.surrogate_loss / denominator;
        let kl = epoch_stats.kl / denominator;
        let ratio_mean = epoch_stats.ratio_mean / denominator;
        let ratio_max = epoch_stats.ratio_max;
        let clip_fraction = epoch_stats.clip_fraction / denominator;
        let total_loss = surrogate_loss + params.kl_coefficient * kl;
        // Same guard as the GRPO and PPO CLI loops: an epoch whose KL or
        // clipped fraction says the trust region is gone stops the run instead
        // of destroying the adapter. This is the frontend with a judge in the
        // loop, so it is the one with the noisiest reward.
        check_policy_divergence(1, epoch + 1, kl, clip_fraction, ratio_max)?;
        let optimizer_timing = trainer
            .optimizer_timing()?
            .delta_since(optimizer_timing_before);
        // A running maximum, not an accumulator: no delta_since here.
        let optimizer_memory = trainer.optimizer_memory()?;
        final_metrics.epoch = epoch + 1;
        final_metrics.train_loss = total_loss;
        let elapsed = started.elapsed().as_secs_f32();
        final_metrics.tokens_per_second = if elapsed > 0.0 {
            trained_tokens as f32 / elapsed
        } else {
            0.0
        };
        on_progress(Progress {
            boundary: None,
            notes: std::mem::take(&mut scratch.notes),
            metrics: final_metrics,
            values: vec![
                metric("reward/mean", reward_mean as f32),
                metric("reward/min", reward_min),
                metric("reward/max", reward_max),
                metric("reward/std", reward_std.min(f32::MAX as f64) as f32),
                metric("reward/group_std", group_diagnostics.mean_reward_std),
                metric(
                    "batch/zero_std_group_fraction",
                    group_diagnostics.zero_std_fraction,
                ),
                metric("batch/advantage_abs_mean", advantage_abs_mean as f32),
                metric("batch/trained_fraction", trained_fraction),
                metric("completions/length_mean", completion_length_mean),
                metric("completions/length_min", completion_length_min),
                metric("completions/length_max", completion_length_max),
                metric("completions/truncation_fraction", truncation_fraction),
                metric("policy/surrogate_loss", surrogate_loss),
                metric("policy/kl", kl),
                metric("policy/clip_fraction", clip_fraction),
                metric("policy/total_loss", total_loss),
                metric("policy/entropy", entropy),
                metric("policy/ratio_mean", ratio_mean),
                metric("policy/ratio_max", ratio_max),
                metric(
                    "timing/optimizer_graph_build_seconds",
                    optimizer_timing.graph_build_seconds as f32,
                ),
                metric(
                    "timing/optimizer_allocation_seconds",
                    optimizer_timing.allocation_seconds as f32,
                ),
                metric(
                    "timing/optimizer_execution_seconds",
                    optimizer_timing.execution_seconds as f32,
                ),
                metric("optimizer/learning_rate", final_metrics.learning_rate),
            ]
            .into_iter()
            .chain(super::grpo::optimizer_memory_metrics(optimizer_memory))
            .collect(),
        });
    }
    if let Some(observation) = observation {
        observation.observer.observe(ObserveBatch::Outcome {
            update: observation.update,
            entries: observation
                .members
                .iter()
                .zip(&live)
                .map(
                    |(&(group, member), &trained)| retrograd_observe::OutcomeEntry {
                        group: Some(group),
                        member,
                        trained,
                    },
                )
                .collect(),
        });
    }
    Ok(final_metrics)
}

/// A sequence reaches this function only once it was scored, so the one
/// exclusion decided here is a group without signal.
fn selection_entries(
    members: &[(usize, usize)],
    advantages: &[f32],
    live: &[bool],
) -> Vec<SelectionEntry> {
    members
        .iter()
        .zip(advantages)
        .zip(live)
        .map(
            |((&(group, member), &advantage), &eligible)| SelectionEntry {
                group,
                member,
                advantage,
                eligible,
                skip_reason: (!eligible).then_some(SkipReason::ZeroSignal),
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence() -> TrainSequence {
        TrainSequence {
            tokens: vec![1, 2, 3, 4, 5],
            old_logprobs: vec![-1.0, -2.0],
            train_mask: vec![false, false, true, false, true],
            reward: 1.0,
            group_id: 7,
            intermediate_returns: Vec::new(),
        }
    }

    #[test]
    fn sequence_accepts_disjoint_policy_segments() {
        sequence().validate().unwrap();
    }

    #[test]
    fn sequence_rejects_shape_and_numeric_errors() {
        let mut invalid = sequence();
        invalid.train_mask.pop();
        assert!(invalid.validate().is_err());

        let mut invalid = sequence();
        invalid.old_logprobs[0] = f32::NAN;
        assert!(invalid.validate().is_err());

        let mut invalid = sequence();
        invalid.train_mask.fill(false);
        invalid.old_logprobs.clear();
        assert!(invalid.validate().is_err());

        let mut invalid = sequence();
        invalid.intermediate_returns = vec![f32::NAN, 0.0];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn intermediate_returns_preserve_the_group_advantage_on_average() {
        // The trajectory keeps its Dr. GRPO weight: only the split between its
        // own tokens changes, so the mean token advantage stays the group one.
        let advantages = token_advantages(2.0, &[6.0, 4.0, 0.0, -2.0]);
        let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        assert!((mean - 2.0).abs() < 1e-5, "{advantages:?}");
        // Return-to-go decreases along the trajectory, so credit does too.
        assert!(advantages.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn uniform_returns_leave_every_token_at_the_group_advantage() {
        // No intermediate signal must behave exactly like terminal-only GRPO.
        let advantages = token_advantages(-1.5, &[3.0, 3.0, 3.0]);
        assert_eq!(advantages, [-1.5, -1.5, -1.5]);
    }

    #[test]
    fn a_single_policy_token_keeps_the_whole_advantage() {
        assert_eq!(token_advantages(0.75, &[9.0]), [0.75]);
    }

    #[test]
    fn selection_marks_the_sequences_a_group_without_signal_left_out() {
        let entries = selection_entries(
            &[(0, 0), (0, 3), (2, 1)],
            &[0.5, -0.5, 0.0],
            &[true, true, false],
        );
        assert_eq!(entries[1].member, 3);
        assert_eq!(entries[1].advantage, -0.5);
        assert!(entries[1].eligible);
        assert_eq!(entries[1].skip_reason, None);
        assert_eq!((entries[2].group, entries[2].member), (2, 1));
        assert_eq!(entries[2].skip_reason, Some(SkipReason::ZeroSignal));
    }

    #[test]
    fn params_reject_invalid_ranges_and_denominator() {
        let valid = GrpoBatchParams {
            epochs: 1,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            loss_denominator: 16,
            seed: 42,
            scheduler_total_rollouts: None,
        };
        valid.validate().unwrap();
        assert!(
            GrpoBatchParams {
                loss_denominator: 0,
                ..valid
            }
            .validate()
            .is_err()
        );
        assert!(
            GrpoBatchParams {
                clip_range_high: 1.0,
                ..valid
            }
            .validate()
            .is_err()
        );
        assert!(
            GrpoBatchParams {
                scheduler_total_rollouts: Some(0),
                ..valid
            }
            .validate()
            .is_err()
        );
        GrpoBatchParams {
            scheduler_total_rollouts: Some(4096),
            ..valid
        }
        .validate()
        .unwrap();
    }
}
