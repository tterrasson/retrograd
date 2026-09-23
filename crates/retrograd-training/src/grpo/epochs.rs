//! Phase 3 of an update: the GRPO epochs.

use super::*;

/// Phase 3 of an update - the GRPO epochs.
///
/// Stochastic minibatch-size-one steps. Live rollouts are re-scored under the
/// current policy before every step; dead slots advance only the scheduler,
/// preserving the configured LR timeline. Reference scores remain fixed across
/// every update and epoch.
///
/// Returns `false` when a progress callback asked to stop, which ends the run.
#[expect(clippy::too_many_arguments)]
pub(super) fn run_epochs(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    state: &mut RunState,
    baseline: &UpdateBaseline,
    scratch: &mut WeightedStepScratch,
    final_metrics: &mut TrainMetrics,
    observer: Option<&dyn TrajectoryObserver>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<bool> {
    // Destructured so the forty-series `metric!` block below pairs each series
    // with a local of the same name.
    let UpdateBaseline {
        update,
        effective_kl,
        ref rollouts,
        ref group_ids,
        ref advantages,
        ref trainable,
        ref reference_rows,
        ref reference_positions,
        ref judge_tally,
        metrics:
            BatchMetrics {
                mean_reward,
                verifiable_reward_mean,
                judge_reward_mean,
                reward_std,
                reward_min,
                reward_max,
                group_diagnostics,
                advantage_abs_mean,
                completion_length_mean,
                completion_length_min,
                completion_length_max,
                truncation_fraction,
                trained_fraction,
                groups_sampled_fraction,
                distinct_fraction,
                entropy,
                trained_tokens,
                scoring_stats,
                reference_seconds,
                timing:
                    BatchTiming {
                        generation: generation_seconds,
                        behavior_scoring: behavior_scoring_seconds,
                        reward: reward_seconds,
                        judge: judge_seconds,
                    },
            },
    } = *baseline;
    let layout = state.layout;
    let judge = state.judge.as_ref();
    // `timing/sampling_seconds`: generation plus behavior scoring.
    let sampling_seconds = generation_seconds + behavior_scoring_seconds;
    let step_params = GrpoStepParams {
        objective: GrpoObjective {
            clip_range_low: config.clip_range_low,
            clip_range_high: config.clip_range_high,
            kl_coefficient: effective_kl,
            loss_denominator: state.loss_denominator,
        },
        scheduler_total_steps: state.horizon.steps(),
    };
    let rollout_evals = rollouts
        .iter()
        .map(|rollout| layout.rollout_evals(rollout))
        .collect::<Result<Vec<_>>>()?;
    let mut optimizer_step_taken = false;
    let epoch_batch = EpochBatch {
        rollouts,
        group_ids,
        advantages,
        token_advantages: &[],
        reference_positions,
        reference_rows,
        rollout_evals: &rollout_evals,
    };
    for epoch in 0..config.grpo_epochs {
        let mut epoch_stats = TokenStats::default();
        let shuffle_seed =
            (config.sampling.seed as u64) ^ ((update as u64 + 1) << 32) ^ (epoch as u64 + 1);
        let order = shuffled_indices(rollouts.len(), shuffle_seed);
        let optimizer_timing_before = trainer.optimizer_timing()?;
        let epoch_started = Instant::now();
        let mut reuse_behavior_logprobs = !optimizer_step_taken;
        let keep_training = run_grpo_epoch(
            trainer,
            &epoch_batch,
            &order,
            &layout,
            step_params,
            EpochState {
                reuse_behavior_logprobs: &mut reuse_behavior_logprobs,
                scratch,
                metrics: final_metrics,
                stats: &mut epoch_stats,
            },
            &mut |trainer, mut metrics| {
                metrics.epoch = update + 1;
                metrics.epoch_complete = false;
                on_progress(
                    trainer,
                    Progress {
                        metrics,
                        values: Vec::new(),
                        boundary: None,
                        notes: Vec::new(),
                    },
                )
            },
        )?;
        optimizer_step_taken = !reuse_behavior_logprobs;
        final_metrics.epoch = update + 1;
        final_metrics.epoch_complete = false;
        if !keep_training {
            return Ok(false);
        }
        let n = trainable.len().max(1) as f32;
        let surrogate_loss = epoch_stats.surrogate_loss / n;
        let kl = epoch_stats.kl / n;
        let ratio_mean = epoch_stats.ratio_mean / n;
        let ratio_max = epoch_stats.ratio_max;
        let clip_fraction = epoch_stats.clip_fraction / n;
        let total_loss = surrogate_loss + effective_kl * kl;
        // Refuse to carry a diverged policy into the next epoch: from here on
        // every group collapses to identical rewards and the run would spend
        // its remaining budget sampling without optimizing anything.
        check_policy_divergence(update + 1, epoch + 1, kl, clip_fraction, ratio_max)?;
        final_metrics.epoch = update + 1;
        final_metrics.epoch_complete = true;
        final_metrics.train_loss = total_loss;
        let epoch_elapsed = epoch_started.elapsed().as_secs_f32();
        let optimizer_timing = trainer
            .optimizer_timing()?
            .delta_since(optimizer_timing_before);
        let optimizer_seconds = optimizer_timing.total_seconds() as f32;
        // A running maximum, not an accumulator: no delta_since here.
        let optimizer_memory = trainer.optimizer_memory()?;
        final_metrics.tokens_per_second = if epoch_elapsed > 0.0 {
            trained_tokens as f32 / epoch_elapsed
        } else {
            0.0
        };
        let closes_update = epoch + 1 == config.grpo_epochs;
        if closes_update && let Some(observer) = observer {
            let trained = reference_positions
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>();
            observer.observe(ObserveBatch::Outcome {
                update: update + 1,
                entries: outcome(Some(config.group_size), &trained),
            });
        }
        // Persist the state for the *next* update. The effective
        // coefficient used above was captured before this adjustment.
        if closes_update && let Some(schedule) = &config.kl_schedule {
            state.kl_multiplier = next_kl_multiplier(state.kl_multiplier, schedule.target, kl);
        }
        // The batch phases and the packing selection run before this event and
        // have nowhere to print: their lines ride out on it, ahead of the row
        // the caller renders from the same event.
        let notes = state
            .pending_notes
            .drain(..)
            .chain(scratch.notes.drain(..))
            .collect();
        if !on_progress(
            trainer,
            Progress {
                notes,
                // Only the last policy epoch of an update closes a
                // resumable boundary. `prompt_cursor` is carried rather
                // than derived: dynamic sampling advances it by more than
                // one group per update.
                boundary: closes_update.then(|| Boundary {
                    completed_iterations: update as u64 + 1,
                    cursor: state.prompt_cursor as u64,
                    kl_multiplier: Some(state.kl_multiplier),
                }),
                metrics: *final_metrics,
                values: vec![
                    metric!("reward/mean", mean_reward),
                    metric!("reward/verifiable_mean", verifiable_reward_mean),
                    metric!("reward/judge_mean", judge_reward_mean),
                    metric!("reward/min", reward_min),
                    metric!("reward/max", reward_max),
                    metric!("reward/std", reward_std),
                    metric!("reward/group_std", group_diagnostics.mean_reward_std),
                    metric!(
                        "batch/zero_std_group_fraction",
                        group_diagnostics.zero_std_fraction
                    ),
                    metric!("batch/advantage_abs_mean", advantage_abs_mean as f32),
                    metric!("completions/length_mean", completion_length_mean),
                    metric!("completions/length_min", completion_length_min),
                    metric!("completions/length_max", completion_length_max),
                    metric!("completions/truncation_fraction", truncation_fraction),
                    metric!("batch/trained_fraction", trained_fraction),
                    metric!("batch/groups_sampled_fraction", groups_sampled_fraction),
                    metric!("policy/surrogate_loss", surrogate_loss),
                    metric!("policy/kl", kl),
                    metric!("policy/clip_fraction", clip_fraction),
                    metric!("policy/total_loss", total_loss),
                    metric!("policy/entropy", entropy),
                    metric!("policy/ratio_mean", ratio_mean),
                    metric!("policy/ratio_max", ratio_max),
                    metric!("policy/kl_coefficient", effective_kl),
                    metric!("rollouts/distinct_fraction", distinct_fraction),
                    metric!("timing/sampling_seconds", sampling_seconds),
                    metric!("timing/generation_seconds", generation_seconds),
                    metric!("timing/behavior_scoring_seconds", behavior_scoring_seconds),
                    // Shared-prefix decodes per scored group: 1.0 when the
                    // prefix is decoded once and every branch is taken from
                    // it, group_size when the scorer degraded to one prompt
                    // prefill per completion.
                    metric!(
                        "scoring/prefix_decodes_per_group",
                        ratio_or_zero(scoring_stats.prefix_decodes, scoring_stats.calls)
                    ),
                    // Share of scored positions whose target log-probability
                    // was gathered on the device instead of reduced from a
                    // full n_vocab row on the host.
                    metric!(
                        "scoring/device_logprob_fraction",
                        ratio_or_zero(
                            scoring_stats.device_logprob_positions,
                            scoring_stats.scored_positions
                        )
                    ),
                    metric!("timing/reward_seconds", reward_seconds),
                    metric!("timing/judge_seconds", judge_seconds),
                    metric!("timing/reference_seconds", reference_seconds),
                    metric!("timing/optimizer_seconds", optimizer_seconds),
                    metric!(
                        "packing/shared_prefix_fanout",
                        scratch.packing_fanout as f32
                    ),
                    metric!(
                        "packing/shared_prefix_passes",
                        scratch.packing_passes as f32
                    ),
                    metric!(
                        "packing/shared_prefix_fraction",
                        ratio_or_zero(
                            scratch.packing_shared_tokens as u64,
                            scratch.packing_useful_tokens as u64
                        )
                    ),
                    metric!(
                        "packing/physical_tokens",
                        scratch.packing_physical_tokens as f32
                    ),
                    metric!(
                        "packing/padding_fraction",
                        1.0 - ratio_or_zero(
                            scratch.packing_useful_tokens as u64,
                            scratch.packing_physical_tokens as u64
                        )
                    ),
                    metric!(
                        "timing/optimizer_graph_build_seconds",
                        optimizer_timing.graph_build_seconds as f32
                    ),
                    metric!(
                        "timing/optimizer_allocation_seconds",
                        optimizer_timing.allocation_seconds as f32
                    ),
                    metric!(
                        "timing/optimizer_execution_seconds",
                        optimizer_timing.execution_seconds as f32
                    ),
                    metric!("optimizer/learning_rate", final_metrics.learning_rate),
                ]
                .into_iter()
                .chain(optimizer_memory_metrics(optimizer_memory))
                // The judge's own series - request count, latency, cache hit
                // rate, score spread - under the names the agentic loop
                // exports, so one dashboard reads both loops.
                .chain(match &judge {
                    Some(_) => judge_tally.metrics(),
                    None => Vec::new(),
                })
                .chain(judge.map(GroupJudge::metric_values).unwrap_or_default())
                .collect(),
            },
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn next_kl_multiplier(current: f32, target: Option<f32>, measured_kl: f32) -> f32 {
    let Some(target) = target else {
        return current;
    };
    if measured_kl > 2.0 * target {
        (current * 1.5).min(1.0e3)
    } else if measured_kl < 0.5 * target {
        (current / 1.5).max(1.0e-3)
    } else {
        current
    }
}
