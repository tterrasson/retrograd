//! Optimizer steps over rollouts: chunking rollouts into accumulation periods,
//! the exact clipped-surrogate step, one GRPO epoch, and the learning-rate
//! horizon they advance.

use retrograd_core::{Error, Result, SharedPrefixFanout, TrainMetrics};
use retrograd_engine::Trainer;

use super::packing::{PackOutcome, PackRefusal, WeightedStepScratch, packed_width};
use super::sampling::{Rollout, RowLayout};
use super::weights::{
    TokenStats, grpo_token_weights_into, ppo_token_weights_into, score_train_mask_into,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PpoStepParams {
    pub(crate) clip_range: f32,
    pub(crate) kl_coefficient: f32,
    pub(crate) scheduler_total_steps: u64,
}

pub(crate) struct SurrogateStep<'a> {
    pub(crate) rollout: &'a Rollout,
    pub(crate) advantages: &'a [f32],
    pub(crate) layout: &'a RowLayout,
    pub(crate) params: PpoStepParams,
    pub(crate) reuse_behavior_logprobs: bool,
    pub(crate) scratch: &'a mut WeightedStepScratch,
    pub(crate) on_step: &'a mut dyn FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
}

/// Rollouts sampled with deferred policy scoring must be backfilled (PPO's
/// critic path does so in `batch_advantages`) before an optimizer step; a
/// length mismatch would otherwise silently zip away every token weight.
fn require_scored_rollout(rollout: &Rollout) -> Result<()> {
    rollout.first_train_index()?;
    if rollout.old_logprobs.len() != rollout.completion_len() {
        return Err(Error::invalid(
            "rollout is missing its behavior-policy log-probabilities",
        ));
    }
    Ok(())
}

/// One exact clipped-surrogate optimizer step for one rollout: re-score the
/// sequence under the current policy so the ratio is fresh, derive the
/// detached per-token weights, pack the row, and take one weighted step.
pub(crate) fn surrogate_step(
    trainer: &mut Trainer,
    step: SurrogateStep<'_>,
) -> Result<(TrainMetrics, TokenStats, bool)> {
    let SurrogateStep {
        rollout,
        advantages,
        layout,
        params,
        reuse_behavior_logprobs,
        scratch,
        on_step,
    } = step;
    require_scored_rollout(rollout)?;
    if reuse_behavior_logprobs {
        scratch.current_logprobs.clear();
        scratch
            .current_logprobs
            .extend_from_slice(&rollout.old_logprobs);
    } else {
        score_train_mask_into(
            trainer,
            &rollout.tokens,
            &rollout.train_mask,
            &mut scratch.current_logprobs,
            &mut scratch.suffix_logprobs,
        )?;
    }
    let stats = ppo_token_weights_into(
        &mut scratch.token_weights,
        advantages,
        &rollout.old_logprobs,
        &scratch.current_logprobs,
        params.clip_range,
        params.kl_coefficient,
    );
    scratch.begin(1);
    scratch.pack_row(0, rollout, layout, rollout.completion_len())?;
    let (metrics, keep_training) =
        trainer.train_weighted_controlled(&scratch.batch, params.scheduler_total_steps, on_step)?;
    Ok((metrics, stats, keep_training))
}

/// The four numbers that define the Dr. GRPO loss, as opposed to the schedule
/// it is stepped on.
///
/// Named because they always travel together, from [`GrpoStepParams`] to
/// [`grpo_token_weights_into`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct GrpoObjective {
    /// Clip-Higher band: the surrogate ratio is clipped to
    /// `[1 - clip_range_low, 1 + clip_range_high]`.
    pub(crate) clip_range_low: f32,
    pub(crate) clip_range_high: f32,
    pub(crate) kl_coefficient: f32,
    /// Dr. GRPO's constant loss denominator: the configured generation budget,
    /// identical for every completion regardless of its sampled length.
    pub(crate) loss_denominator: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GrpoStepParams {
    pub(crate) objective: GrpoObjective,
    pub(crate) scheduler_total_steps: u64,
}

/// One member of a Dr. GRPO optimizer chunk, borrowed from the caller's
/// per-update buffers.
pub(crate) struct ChunkMember<'a> {
    pub(crate) group_id: u64,
    pub(crate) rollout: &'a Rollout,
    pub(crate) advantage: f32,
    pub(crate) token_advantages: Option<&'a [f32]>,
    /// Frozen-reference logprobs aligned with the rollout's trainable tokens.
    pub(crate) reference_logprobs: &'a [f32],
}

/// Greedy partition of `indices` (an already-shuffled epoch order) into
/// optimizer chunks: consecutive rollouts join a chunk while their combined
/// ubatch evals fit one accumulation period - one optimizer window of
/// real tokens - so a single optimizer step trains as many short rollouts as
/// the batch budget holds. A rollout larger than the budget forms its own
/// multi-step chunk. Returned ranges index into `indices`; `evals` is indexed
/// by rollout index.
pub(crate) fn chunk_by_period(
    indices: &[usize],
    evals: &[u64],
    period: u64,
) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0_usize;
    let mut budget = 0_u64;
    for (position, &index) in indices.iter().enumerate() {
        let cost = evals[index];
        if position > start && budget + cost > period {
            chunks.push(start..position);
            start = position;
            budget = 0;
        }
        budget += cost;
    }
    if start < indices.len() {
        chunks.push(start..indices.len());
    }
    chunks
}

/// Builds optimizer chunks for the packed multi-sequence graph. Prompt groups
/// stay contiguous so their prefix can remain shared; group segments then join
/// the same physical ubatch while both its token and sequence budgets allow.
/// Unused sequence slots each reserve one padding token to keep the runtime
/// graph shape invariant.
pub(crate) fn grouped_optimizer_chunks(
    live_order: &[usize],
    group_ids: &[u64],
    rollouts: &[Rollout],
    evals: &[u64],
    layout: &RowLayout,
) -> Result<Vec<Vec<usize>>> {
    if group_ids.len() != rollouts.len() || evals.len() != rollouts.len() {
        return Err(Error::invalid("optimizer chunk metadata does not align"));
    }
    if layout.shared_prefix_fanout == SharedPrefixFanout::Off {
        return Ok(
            chunk_by_period(live_order, evals, layout.accumulation_period())
                .into_iter()
                .map(|range| live_order[range].to_vec())
                .collect(),
        );
    }

    let mut by_group = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    let mut group_order = Vec::new();
    for &index in live_order {
        if index >= rollouts.len() {
            return Err(Error::overflow("optimizer order index is out of range"));
        }
        let id = group_ids[index];
        if !by_group.contains_key(&id) {
            group_order.push(id);
        }
        by_group.entry(id).or_default().push(index);
    }

    // A reward group is one logical optimizer update. Physical fanout may
    // split it into several graphs later, but it must never split AdamW.
    Ok(group_order
        .into_iter()
        .map(|id| by_group.remove(&id).expect("group was inserted"))
        .collect())
}

/// Learning-rate horizon, re-sized on the optimizer steps a run actually
/// takes.
///
/// The nominal count - one step per rollout slot per epoch - is an upper
/// bound, and a loose one: chunked rollouts share a step, and a row is trained
/// only up to its last active label, so short completions skip their trailing
/// ubatches. Left at the bound, `progress` plateaus well below 1 and `linear` /
/// `cosine` end the run at a large fraction of the peak rate instead of at
/// zero. Each finished update is measured, the remaining ones are priced at
/// the running mean, and the result is clamped to the original bound - so the
/// horizon only ever moves toward the truth, never past it.
pub(crate) struct SchedulerHorizon {
    horizon: u64,
    upper_bound: u64,
    floor: u64,
    measured_steps: u64,
    measured_updates: u64,
}

impl SchedulerHorizon {
    /// `floor` is the warm-up length: the runtime refuses a horizon shorter
    /// than it.
    pub(crate) fn new(upper_bound: u64, floor: u64) -> Self {
        Self {
            horizon: upper_bound.max(floor),
            upper_bound: upper_bound.max(floor),
            floor,
            measured_steps: 0,
            measured_updates: 0,
        }
    }

    pub(crate) fn steps(&self) -> u64 {
        self.horizon
    }

    /// Records one finished update and prices the remaining ones on the mean
    /// cost observed so far.
    pub(crate) fn observe(&mut self, steps_taken: u64, remaining_updates: u64, global_step: u64) {
        self.measured_steps = self.measured_steps.saturating_add(steps_taken);
        self.measured_updates += 1;
        let per_update = self.measured_steps.div_ceil(self.measured_updates);
        self.horizon = global_step
            .saturating_add(per_update.saturating_mul(remaining_updates))
            .max(global_step.saturating_add(1))
            .max(self.floor)
            .min(self.upper_bound);
    }
}

/// Whether a completion of `length` tokens exhausted its generation budget,
/// and is therefore truncated (or indistinguishable from truncated).
///
/// One predicate for both readers: the `truncation_fraction` series and the
/// masking decision that drops those completions. `>=` rather than `==`
/// because a caller is free to hand in externally collected trajectories that
/// ran past the budget the optimizer normalizes on.
pub(crate) fn is_truncated(length: usize, loss_denominator: usize) -> bool {
    length >= loss_denominator
}

/// Everything one GRPO epoch needs about a batch of rollouts, whichever
/// frontend collected them: the built-in sampler or an external agentic
/// rollout. Indices are shared by every slice; `reference_positions` doubles as
/// the liveness mask, `None` marking a filtered slot.
pub(crate) struct EpochBatch<'a> {
    pub(crate) rollouts: &'a [Rollout],
    pub(crate) group_ids: &'a [u64],
    pub(crate) advantages: &'a [f32],
    /// Per-token credit inside a trajectory, when intermediate returns are
    /// available. Empty means terminal-reward-only GRPO.
    pub(crate) token_advantages: &'a [Option<Vec<f32>>],
    pub(crate) reference_positions: &'a [Option<usize>],
    /// Frozen-reference rows in `reference_positions` order. Empty when the
    /// effective KL coefficient is zero and the reference pass was skipped.
    pub(crate) reference_rows: &'a [Vec<f32>],
    pub(crate) rollout_evals: &'a [u64],
}

/// The epoch's mutable working state: the four things `run_grpo_epoch` writes
/// through, as opposed to the batch and the parameters it only reads.
///
/// Grouped because the split is the useful information: the signature now says
/// in one place what an epoch may change, instead of interleaving four `&mut`
/// with five `&`.
pub(crate) struct EpochState<'a> {
    /// Cleared by the first optimizer step of the epoch: after it, the sampled
    /// behaviour logprobs no longer describe the live policy.
    pub(crate) reuse_behavior_logprobs: &'a mut bool,
    pub(crate) scratch: &'a mut WeightedStepScratch,
    pub(crate) metrics: &'a mut TrainMetrics,
    pub(crate) stats: &'a mut TokenStats,
}

/// One GRPO epoch over an already-shuffled slot order: dead slots advance the
/// scheduler, live ones are grouped into optimizer chunks and stepped.
///
/// Shared by the GRPO CLI loop and the pre-generated batch entry point so the
/// two cannot drift on the pieces that matter - the diagnostics feeding the
/// divergence guard, the chunking, and the fixed-horizon scheduler bookkeeping.
/// `metrics` and `stats` are accumulated in place; the returned flag is `false`
/// when a progress callback asked to stop.
pub(crate) fn run_grpo_epoch(
    trainer: &mut Trainer,
    batch: &EpochBatch<'_>,
    order: &[usize],
    layout: &RowLayout,
    params: GrpoStepParams,
    state: EpochState<'_>,
    on_progress: &mut dyn FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
) -> Result<bool> {
    let EpochState {
        reuse_behavior_logprobs,
        scratch,
        metrics,
        stats,
    } = state;
    // Dead slots advance the scheduler without optimizing; live slots keep
    // their shuffled order and are then grouped so one optimizer step trains a
    // full accumulation period of real rollout tokens.
    let mut live_order = Vec::with_capacity(order.len());
    for &index in order {
        if batch.reference_positions[index].is_some() {
            live_order.push(index);
            continue;
        }
        metrics.global_step = trainer.advance_scheduler_steps(layout.steps_per_row)?;
        if !on_progress(trainer, *metrics)? {
            return Ok(false);
        }
    }
    let chunks = grouped_optimizer_chunks(
        &live_order,
        batch.group_ids,
        batch.rollouts,
        batch.rollout_evals,
        layout,
    )?;
    for chunk_indices in chunks {
        let members = chunk_indices
            .iter()
            .map(|&index| ChunkMember {
                group_id: batch.group_ids[index],
                rollout: &batch.rollouts[index],
                advantage: batch.advantages[index],
                token_advantages: batch
                    .token_advantages
                    .get(index)
                    .and_then(|values| values.as_deref()),
                // An empty reference set means the pass was skipped because
                // the effective KL coefficient is zero; the weight kernel
                // reads an empty slice as "no KL term".
                reference_logprobs: if batch.reference_rows.is_empty() {
                    &[]
                } else {
                    &batch.reference_rows[batch.reference_positions[index]
                        .expect("live rollouts carry a reference row")]
                },
            })
            .collect::<Vec<_>>();
        let (step_metrics, chunk_stats, keep_training) = grpo_chunk_step(
            trainer,
            &members,
            layout,
            params,
            // Before the first optimizer step the policy still is the behavior
            // policy, so re-scoring would reproduce `old_logprobs` exactly.
            *reuse_behavior_logprobs,
            scratch,
            on_progress,
        )?;
        *reuse_behavior_logprobs = false;
        *metrics = step_metrics;
        stats.surrogate_loss += chunk_stats.surrogate_loss;
        stats.kl += chunk_stats.kl;
        stats.clip_fraction += chunk_stats.clip_fraction;
        stats.ratio_mean += chunk_stats.ratio_mean;
        stats.ratio_max = stats.ratio_max.max(chunk_stats.ratio_max);
        if !keep_training {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) enum PackingSelection {
    Packed(Vec<std::ops::Range<usize>>),
    Rows {
        status: &'static str,
        reason: String,
    },
}

fn packing_refusal(refusal: PackRefusal) -> PackingSelection {
    let (status, reason) = match refusal {
        PackRefusal::FanoutDoesNotFit {
            required,
            available,
        } => (
            "width",
            format!(
                "physical micro-batch too small: requires {required} tokens, available={available} \
                (including unused sequence slots); increase training.micro_batch and, if needed, \
                training.ctx to a compatible size"
            ),
        ),
        PackRefusal::DivergentPrefix => (
            "prefix",
            "a rollout has fewer than two prompt tokens; no shareable teacher-forcing prefix"
                .into(),
        ),
        PackRefusal::InvalidLayout => ("layout", "invalid packed sequence layout".into()),
    };
    PackingSelection::Rows { status, reason }
}

/// Plan every pass before starting the optimizer transaction. Auto can keep
/// short siblings packed while a longer member occupies a pass of its own.
/// Never mix row and packed training inside an accumulation transaction.
pub(crate) fn select_packing(
    chunk: &[ChunkMember<'_>],
    layout: &RowLayout,
    capability: bool,
) -> Result<PackingSelection> {
    let requested = layout.shared_prefix_fanout;
    if matches!(requested, SharedPrefixFanout::Exact(0 | 1)) {
        return Err(Error::invalid(
            "shared-prefix fanout must be at least two; use off to disable packing",
        ));
    }
    let locked = matches!(
        requested,
        SharedPrefixFanout::Max | SharedPrefixFanout::Exact(_)
    );
    let disabled = if requested == SharedPrefixFanout::Off {
        Some(("off", "disabled by training.shared_prefix_fanout=off"))
    } else if !capability {
        Some((
            "capability",
            "the model/device does not support shared-prefix packed training; inspect the runtime capability report",
        ))
    } else if chunk.len() < 2 {
        Some((
            "members",
            "fewer than two live completions in this optimizer chunk",
        ))
    } else if layout.n_seq_max < 2 {
        Some((
            "sequences",
            "the optimizer has fewer than two sequence slots (n_seq_max)",
        ))
    } else {
        None
    };
    if let Some((status, reason)) = disabled {
        if locked && (status == "capability" || status == "sequences") {
            return Err(Error::invalid(reason));
        }
        return Ok(PackingSelection::Rows {
            status,
            reason: reason.into(),
        });
    }
    let requested_fanout = match requested {
        SharedPrefixFanout::Exact(value) => value as usize,
        _ => chunk.len(),
    }
    .min(chunk.len());
    // `max` is the widest fanout the sequence slots allow; only an explicit
    // integer names a width the slots must hold.
    if matches!(requested, SharedPrefixFanout::Exact(_)) && requested_fanout > layout.n_seq_max {
        return Err(Error::invalid(format!(
            "training.shared_prefix_fanout={requested:?} requires {requested_fanout} sequence slots, n_seq_max={}",
            layout.n_seq_max
        )));
    }
    let max_fanout = requested_fanout.min(layout.n_seq_max);
    let mut ranges = Vec::new();
    let mut start = 0;
    // Sizes are tried widest first, so the refusal kept is the narrowest one
    // tried: the one that tells the user how far the width is from fitting.
    let mut last_refusal = None;
    while start < chunk.len() {
        let max = max_fanout.min(chunk.len() - start);
        let min = if locked { max } else { 1 };
        let mut chosen = None;
        for count in (min..=max).rev() {
            match packed_width(&chunk[start..start + count], layout)? {
                Ok(_) => {
                    chosen = Some(count);
                    break;
                }
                Err(refusal) => last_refusal = Some(refusal),
            }
        }
        let Some(count) = chosen else {
            let fallback = packing_refusal(last_refusal.expect("at least one candidate"));
            if locked && let PackingSelection::Rows { reason, .. } = &fallback {
                return Err(Error::invalid(format!(
                    "training.shared_prefix_fanout={requested:?}: {reason}"
                )));
            }
            return Ok(fallback);
        };
        ranges.push(start..start + count);
        start += count;
    }
    if ranges.iter().all(|range| range.len() == 1) {
        return Ok(packing_refusal(
            last_refusal.expect("multi-member candidate refused"),
        ));
    }
    Ok(PackingSelection::Packed(ranges))
}

/// One Dr. GRPO optimizer chunk: PPO clipping against the rollout policy and
/// a k3 KL penalty against the frozen base-model reference policy. Every
/// member is re-scored under the same current policy, packed as its own row,
/// and trained in one weighted call; the runtime accumulates gradients across
/// the rows' real (ubatch-rounded) content, so short rollouts share optimizer
/// steps instead of paying one full padded row each.
pub(crate) fn grpo_chunk_step(
    trainer: &mut Trainer,
    chunk: &[ChunkMember<'_>],
    layout: &RowLayout,
    params: GrpoStepParams,
    reuse_behavior_logprobs: bool,
    scratch: &mut WeightedStepScratch,
    on_step: &mut dyn FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
) -> Result<(TrainMetrics, TokenStats, bool)> {
    if chunk.is_empty() {
        return Err(Error::invalid("GRPO optimizer chunk must not be empty"));
    }
    scratch.begin_members(chunk.len());
    let mut chunk_stats = TokenStats::default();
    for (slot, member) in chunk.iter().enumerate() {
        require_scored_rollout(member.rollout)?;
        if member
            .token_advantages
            .is_some_and(|values| values.len() != member.rollout.completion_len())
        {
            return Err(Error::invalid(
                "token advantages do not align with the rollout train_mask",
            ));
        }
        if reuse_behavior_logprobs {
            scratch.current_logprobs.clear();
            scratch
                .current_logprobs
                .extend_from_slice(&member.rollout.old_logprobs);
        } else {
            score_train_mask_into(
                trainer,
                &member.rollout.tokens,
                &member.rollout.train_mask,
                &mut scratch.current_logprobs,
                &mut scratch.suffix_logprobs,
            )?;
        }
        let stats = grpo_token_weights_into(
            &mut scratch.member_weights[slot],
            member.advantage,
            member.token_advantages,
            &member.rollout.old_logprobs,
            &scratch.current_logprobs,
            member.reference_logprobs,
            &params.objective,
        );
        chunk_stats.surrogate_loss += stats.surrogate_loss;
        chunk_stats.kl += stats.kl;
        chunk_stats.clip_fraction += stats.clip_fraction;
        chunk_stats.ratio_mean += stats.ratio_mean;
        chunk_stats.ratio_max = chunk_stats.ratio_max.max(stats.ratio_max);
    }
    let capability = trainer.supports_shared_prefix_packed_training()?;
    let requested = layout.shared_prefix_fanout;
    let selection = select_packing(chunk, layout, capability)?;
    let (metrics, keep_training) = if let PackingSelection::Packed(ranges) = &selection {
        let fanout = ranges.iter().map(|range| range.len()).max().unwrap_or(1);
        let passes = u32::try_from(ranges.len())
            .map_err(|_| Error::overflow("packed pass count exceeds u32"))?;
        scratch.packing_fanout = fanout;
        scratch.packing_passes = ranges.len();
        scratch.packing_shared_tokens = 0;
        scratch.packing_useful_tokens = 0;
        scratch.packing_physical_tokens = ranges.len() * layout.ubatch;
        scratch.note_packing(
            format!("packed:{fanout}:{}", ranges.len()),
            format!(
                "shared-prefix packing: fanout={fanout}, passes={passes}, physical_width={} (requested={requested:?})",
                layout.ubatch
            ),
        );
        let mut latest = TrainMetrics::default();
        let mut keep = true;
        for range in ranges {
            let subgroup = &chunk[range.clone()];
            match scratch.pack_sequences(
                subgroup,
                range.start,
                layout,
                params.objective.loss_denominator,
            )? {
                PackOutcome::Packed {
                    useful_tokens,
                    shared_tokens,
                } => {
                    scratch.packing_useful_tokens += useful_tokens;
                    scratch.packing_shared_tokens += shared_tokens;
                }
                PackOutcome::Refused(refusal) => {
                    return Err(Error::runtime(format!(
                        "shared-prefix geometry changed after selection: {refusal:?}"
                    )));
                }
            }
            // ggml scales every accumulated loss by 1/passes. The coefficients
            // are already normalized by the logical update denominator, so
            // cancel that internal average to preserve the exact gradient sum.
            for weight in &mut scratch.packed_sequence_batch.weights {
                *weight *= passes as f32;
            }
            let result = trainer.train_packed_sequences_controlled(
                &scratch.packed_sequence_batch,
                params.scheduler_total_steps,
                passes,
                &mut *on_step,
            )?;
            latest = result.0;
            keep &= result.1;
            if !keep {
                break;
            }
        }
        (latest, keep)
    } else {
        scratch.packing_fanout = 1;
        scratch.packing_passes = chunk.len();
        scratch.packing_shared_tokens = 0;
        scratch.packing_useful_tokens =
            chunk.iter().map(|member| member.rollout.tokens.len()).sum();
        scratch.packing_physical_tokens = chunk.len() * layout.row_width;
        if let PackingSelection::Rows { status, reason } = selection {
            scratch.note_packing(
                status.into(),
                format!(
                    "shared-prefix packing disabled: {reason} (requested={requested:?}, \
                    capability={capability}, live_completions={}, n_seq_max={}, physical_width={})",
                    chunk.len(),
                    layout.n_seq_max,
                    layout.ubatch
                ),
            );
        }
        // Only the row path uses `scratch.batch`; sizing it is a resize of
        // three context-sized vectors, so the packed path never pays for it.
        scratch.begin(chunk.len());
        for (slot, member) in chunk.iter().enumerate() {
            scratch.token_weights.clear();
            scratch
                .token_weights
                .extend_from_slice(&scratch.member_weights[slot]);
            scratch.pack_row(
                slot,
                member.rollout,
                layout,
                params.objective.loss_denominator,
            )?;
        }
        trainer.train_weighted_controlled(&scratch.batch, params.scheduler_total_steps, on_step)?
    };
    Ok((metrics, chunk_stats, keep_training))
}
