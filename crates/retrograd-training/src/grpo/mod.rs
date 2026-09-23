//! Dr. GRPO over the differentiable runtime objective.
//!
//! GRPO replaces PPO's learned critic with a group-relative baseline: every
//! prompt is sampled `group_size` times and the advantage of a completion is
//! its reward centered *within its own group*, constant over the completion's
//! tokens. Unlike original GRPO, Dr. GRPO removes reward-std and response-length
//! normalization: every response is divided by the configured, constant
//! generation budget. Sampling is strictly on-policy (temperature 1, top-p 1).
//!
//! On top of Dr. GRPO, three DAPO corrections apply: the surrogate clip is
//! decoupled (Clip-Higher, `clip_range_low` / `clip_range_high`), groups
//! without reward spread are dropped from the optimizer epochs instead of
//! taking KL-only steps, and - optionally, via `mask_truncated` - completions
//! truncated at the generation budget are excluded from both the baseline and
//! the epochs, since their reward judges an incomplete response.

use std::time::Instant;

use super::observe::{GrpoSlots, grpo_rollouts, observed_prompts, outcome};
use super::rollout::prompts::Prompt;
use super::rollout::{
    EpochBatch, EpochState, GroupJudge, GrpoObjective, GrpoStepParams, JudgeGroup, JudgeTally,
    RewardRow, Rollout, RowLayout, SchedulerHorizon, TokenStats, WeightedStepScratch,
    check_policy_divergence, is_truncated, mean_std, read_prompts, reward_process, run_grpo_epoch,
    sample_rollout_groups_continuous, score, score_rows, score_train_mask_group, tokenize_prompts,
};
use super::{Boundary, Progress};
use retrograd_config::GrpoConfig;
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_engine::{ScoringStats, Trainer};
use retrograd_judge::RewardProcess;
use retrograd_metrics::MetricValue;
use retrograd_observe::{ObserveBatch, RolloutBatch, TrajectoryObserver};

macro_rules! metric {
    ($name:literal, $value:expr_2021) => {
        MetricValue {
            name: $name.into(),
            value: $value,
        }
    };
}

mod baseline;
mod batch;
mod epochs;

use baseline::*;
use batch::*;
use epochs::*;

/// Counter ratio as a metric value; zero when nothing was counted, which is the
/// honest reading for "this path did not run" on a series that is otherwise a
/// share.
pub(crate) fn ratio_or_zero(numerator: u64, denominator: u64) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f32 / denominator as f32
}

/// Device-memory series for one update, measured by the runtime *inside* its
/// optimizer steps.
///
/// Distinct from the `system/vram_*` series, which the host sampler reads between
/// steps and which therefore cannot see the allocation + backward peak. Empty on a
/// CPU-only run rather than three flat zeroes, so an absent GPU never looks like a
/// measurement of zero.
pub(crate) fn optimizer_memory_metrics(
    memory: retrograd_engine::OptimizerMemory,
) -> Vec<MetricValue> {
    if !memory.is_measured() {
        return Vec::new();
    }
    const MIB: f32 = 1024.0 * 1024.0;
    vec![
        metric!(
            "memory/optimizer_device_peak_mib",
            memory.device_peak_used_bytes as f32 / MIB
        ),
        metric!(
            "memory/optimizer_device_used_mib",
            memory.device_used_bytes as f32 / MIB
        ),
        // The term the byte breakdown cannot see: CUDA pool + Vulkan prealloc_*.
        metric!(
            "memory/optimizer_scratch_peak_mib",
            memory.scratch_peak_bytes as f32 / MIB
        ),
    ]
}

pub fn evaluate(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    data: &std::path::Path,
    max_examples: Option<usize>,
) -> Result<super::RewardEvalMetrics> {
    super::rollout::evaluate_rewards(
        trainer,
        data,
        &mut reward_process(&config.reward_command, config.reward_protocol)?,
        &config.sampling,
        training,
        max_examples,
    )
}

pub fn benchmark_rewards(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    data: &std::path::Path,
    limit: Option<usize>,
) -> Result<Vec<f32>> {
    super::rollout::evaluate_reward_values(
        trainer,
        data,
        &mut reward_process(&config.reward_command, config.reward_protocol)?,
        &config.sampling,
        training,
        limit,
    )
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GroupDiagnostics {
    /// Mean within-group reward std (over live members) - the group-collapse
    /// diagnostic.
    pub(crate) mean_reward_std: f32,
    /// Fraction of groups carrying zero learning signal: all live rewards
    /// identical, or fewer than two live members. Those groups are marked
    /// dead and skipped by the optimizer epochs.
    pub(crate) zero_std_fraction: f32,
    /// How many of those groups died of each cause. The two are not the same
    /// failure - identical rewards say the reward cannot separate the
    /// completions, too few live members says something upstream (truncation
    /// masking, a judge drop) emptied the group before the reward was even
    /// consulted - and a diagnostic that names the wrong one sends the reader
    /// to the wrong knob.
    pub(crate) uniform_groups: usize,
    pub(crate) starved_groups: usize,
}

/// Dr. GRPO group-relative advantages: `r - mean_group`, without the reward
/// standard-deviation normalization that biases the weighting across prompts.
///
/// `live` marks the rollouts eligible for training (all of them, unless
/// truncation masking already cleared some). The baseline of a group is the
/// mean over its *live* members only. DAPO-style filtering happens here too:
/// a group without signal - fewer than two live members, or all live rewards
/// identical - is marked fully dead so the optimizer epochs skip it instead
/// of taking KL-only steps. Dead rollouts get a zero advantage.
pub(crate) fn group_advantages(
    rewards: &[f32],
    group_ids: &[u64],
    live: &mut [bool],
    baseline: retrograd_config::AdvantageBaseline,
) -> Result<(Vec<f32>, GroupDiagnostics)> {
    use retrograd_config::AdvantageBaseline;
    if rewards.is_empty() || group_ids.len() != rewards.len() || live.len() != rewards.len() {
        return Err(Error::invalid(
            "rewards, group_ids, and live flags must have the same non-zero length",
        ));
    }
    let mut advantages = vec![0.0_f32; rewards.len()];
    let mut std_sum = 0.0_f64;
    let mut zero_signal_groups = 0_usize;
    let mut uniform_groups = 0_usize;
    let mut starved_groups = 0_usize;
    let mut groups = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    for (index, &group_id) in group_ids.iter().enumerate() {
        groups.entry(group_id).or_default().push(index);
    }
    let mut live_members = Vec::new();
    for indices in groups.values() {
        live_members.clear();
        live_members.extend(
            indices
                .iter()
                .copied()
                .filter(|&index| live[index])
                .map(|index| rewards[index]),
        );
        let (mean, std, live_count) = mean_std(live_members.iter().copied());
        // Detect zero-signal groups by exact reward equality rather than
        // `std == 0.0`: identical but non-representable means (e.g. three
        // rewards of 0.1) leave a ~1e-17 residual std in f64.
        let uniform = live_members.windows(2).all(|pair| pair[0] == pair[1]);
        if live_count < 2 || uniform {
            zero_signal_groups += 1;
            if live_count < 2 {
                starved_groups += 1;
            } else {
                uniform_groups += 1;
            }
            for &index in indices {
                live[index] = false;
            }
            continue;
        }
        std_sum += std;
        // RLOO baselines each member against the mean of the *other* live
        // members: `r - (sum - r)/(n-1)`, which equals the mean-centered
        // advantage rescaled by `n/(n-1)`. Same gradient direction, corrected
        // scale for small groups.
        let rloo_scale = match baseline {
            AdvantageBaseline::Mean => 1.0,
            AdvantageBaseline::LeaveOneOut => live_count as f64 / (live_count - 1) as f64,
        };
        for &index in indices {
            if !live[index] {
                continue;
            }
            let advantage = (rewards[index] as f64 - mean) * rloo_scale;
            if advantage < f32::MIN as f64 || advantage > f32::MAX as f64 {
                return Err(Error::invalid(
                    "centered GRPO reward exceeds the finite f32 range; rescale rewards",
                ));
            }
            advantages[index] = advantage as f32;
        }
    }
    let group_count = groups.len();
    Ok((
        advantages,
        GroupDiagnostics {
            mean_reward_std: (std_sum / group_count as f64).min(f32::MAX as f64) as f32,
            zero_std_fraction: zero_signal_groups as f32 / group_count as f32,
            uniform_groups,
            starved_groups,
        },
    ))
}

/// `reward/mean` split into the two halves it sums: what the reward command
/// demonstrated on its own, and what the judge added on top.
///
/// `judge_terms` is the per-rollout `verdict * weight` carried alongside the
/// rewards, so only the judge half is measured and the verifiable half is the
/// remainder. Derived from the published mean rather than from a second pass
/// over the rewards, which is what makes `verifiable + judge == mean` an
/// identity on the emitted figures instead of a coincidence between two
/// reductions that could later drift apart.
fn reward_split(mean_reward: f64, judge_terms: &[f32]) -> (f32, f32) {
    let (judge_mean, _, _) = mean_std(judge_terms.iter().copied());
    ((mean_reward - judge_mean) as f32, judge_mean as f32)
}

/// Why an update carried no learning signal, in the words of the cause that
/// dominated it. The two causes call for opposite reactions - a reward that
/// cannot separate completions, versus groups emptied before the reward was
/// consulted - so the phrasing names the one that actually happened rather
/// than assuming the first.
fn zero_signal_cause(diagnostics: &GroupDiagnostics) -> String {
    let GroupDiagnostics {
        uniform_groups,
        starved_groups,
        ..
    } = *diagnostics;
    match (uniform_groups, starved_groups) {
        (0, 0) => "every group was dropped before the advantages were computed".to_string(),
        (_, 0) => "every completion of every group received an identical reward".to_string(),
        (0, _) => "every group was left with fewer than two trainable completions, so the \
                   rewards were never compared"
            .to_string(),
        (uniform, starved) => format!(
            "{uniform} group(s) received an identical reward throughout and {starved} were \
             left with fewer than two trainable completions"
        ),
    }
}

/// SplitMix64 (Steele et al., "Fast Splittable Pseudorandom Number
/// Generators", OOPSLA 2014). Hand-rolled rather than pulling in `rand` so
/// seeded shuffles stay bit-for-bit reproducible across dependency and
/// platform changes. Its strong avalanche also decorrelates the epoch
/// shuffle seeds, which differ from one another by only a few bits.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Deterministic Fisher-Yates permutation of `0..len`, used to reorder the
/// minibatch-size-one GRPO steps each epoch so a fixed rollout order does not
/// systematically bias the optimizer. The modulo bias of `% (upper + 1)` is
/// on the order of `len / 2^64` - irrelevant at rollout-batch sizes.
pub(crate) fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut state = seed;
    for upper in (1..len).rev() {
        let index = (splitmix64(&mut state) % (upper as u64 + 1)) as usize;
        indices.swap(upper, index);
    }
    indices
}

/// Resolves the prompt-file slot for a global draw index. Sequential is a
/// round-robin; shuffled re-draws a seeded permutation each full pass over the
/// dataset, so correlated adjacent prompts do not always land in the same
/// update. The permutation is stable within a pass.
#[cfg(test)]
fn prompt_slot_for(
    order: &retrograd_config::PromptOrder,
    prompt_offset: usize,
    len: usize,
    seed: u32,
) -> usize {
    PromptDraws::new(*order, len, seed).slot(prompt_offset)
}

/// Seed of the permutation covering one full pass over the prompt file.
fn pass_seed(seed: u32, pass: u64) -> u64 {
    (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (pass.wrapping_add(1) << 1)
}

/// Prompt draws with the current pass's permutation held between calls.
///
/// The permutation is a function of the pass alone, so it is shuffled once per
/// pass rather than once per draw.
pub(crate) struct PromptDraws {
    order: retrograd_config::PromptOrder,
    len: usize,
    seed: u32,
    pass: Option<u64>,
    permutation: Vec<usize>,
}

impl PromptDraws {
    pub(crate) fn new(order: retrograd_config::PromptOrder, len: usize, seed: u32) -> Self {
        Self {
            order,
            len,
            seed,
            pass: None,
            permutation: Vec::new(),
        }
    }

    pub(crate) fn slot(&mut self, prompt_offset: usize) -> usize {
        match self.order {
            retrograd_config::PromptOrder::Sequential => prompt_offset % self.len,
            retrograd_config::PromptOrder::Shuffled => {
                let pass = (prompt_offset / self.len) as u64;
                if self.pass != Some(pass) {
                    self.permutation = shuffled_indices(self.len, pass_seed(self.seed, pass));
                    self.pass = Some(pass);
                }
                self.permutation[prompt_offset % self.len]
            }
        }
    }
}

/// Mean over groups of the fraction of distinct completions in each group.
/// 1.0 means every member differs; low values flag degenerate sampling.
fn distinct_completion_fraction(completions: &[String], group_ids: &[u64]) -> f32 {
    if completions.is_empty() {
        return 0.0;
    }
    let mut groups = std::collections::BTreeMap::<u64, Vec<&str>>::new();
    for (id, completion) in group_ids.iter().zip(completions) {
        groups.entry(*id).or_default().push(completion.as_str());
    }
    let mut sum = 0.0_f64;
    for members in groups.values() {
        let distinct = members
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<&str>>()
            .len();
        sum += distinct as f64 / members.len() as f64;
    }
    (sum / groups.len() as f64) as f32
}

/// DAPO soft overlong punishment: the reward reduction for a
/// completion of `length` tokens under a `loss_denominator`-token budget. Zero
/// until the last `buffer_tokens`, then ramps linearly to `max_penalty`.
/// Shared by the batch penalty loop and dynamic sampling's signal test so the
/// two never drift.
fn overlong_shaped_penalty(
    penalty: &retrograd_config::OverlongPenalty,
    length: usize,
    loss_denominator: usize,
) -> f32 {
    let threshold = loss_denominator.saturating_sub(penalty.buffer_tokens);
    if length > threshold {
        let over = (length - threshold) as f32;
        let shaped = (over / penalty.buffer_tokens as f32).min(1.0);
        penalty.max_penalty * shaped
    } else {
        0.0
    }
}

/// Whether a single group carries a learning signal: at least two live members
/// with non-identical rewards, evaluated on the same penalized rewards and
/// truncation mask the optimizer sees. Mirrors the zero-signal test in
/// `group_advantages` so dynamic sampling keeps exactly the groups the
/// epochs would train.
fn group_has_signal(
    rollouts: &[Rollout],
    rewards: &[f32],
    mask_truncated: bool,
    overlong_penalty: Option<&retrograd_config::OverlongPenalty>,
    loss_denominator: usize,
) -> bool {
    let mut live_rewards: Vec<f32> = Vec::with_capacity(rollouts.len());
    for (rollout, &reward) in rollouts.iter().zip(rewards) {
        let length = rollout.completion_len();
        if mask_truncated && is_truncated(length, loss_denominator) {
            continue;
        }
        let mut shaped = reward;
        if let Some(penalty) = overlong_penalty {
            shaped -= overlong_shaped_penalty(penalty, length, loss_denominator);
        }
        live_rewards.push(shaped);
    }
    if live_rewards.len() < 2 {
        return false;
    }
    let first = live_rewards[0];
    !live_rewards.iter().all(|&reward| reward == first)
}

/// Runs GRPO end to end: sample a group of completions per prompt, score them
/// with the reward command, turn each group's rewards into group-relative
/// advantages, then take `grpo_epochs` exact clipped-surrogate steps per
/// rollout batch through the runtime's weighted differentiable objective.
pub fn run(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    run_resumed(trainer, config, training, None, None, &mut |_, progress| {
        if progress.metrics.epoch_complete {
            on_progress(progress);
        }
        Ok(true)
    })
}

/// Everything a GRPO run establishes before its first update, plus the state
/// that survives from one update to the next.
///
/// Split out of [`run_resumed`] so the update loop reads as its three phases
/// and nothing else: what is built once for the run lives here, what is rebuilt
/// every update stays in the loop. The mutable half is exactly what a
/// [`Boundary`] restores - a resumed run enters the loop with the state its
/// first launch left behind, not with a fresh one.
struct RunState {
    /// First update index of this launch; zero unless resumed.
    start_update: u32,
    prompts: Vec<Prompt>,
    /// `prompts`, tokenized once and reused across every update.
    prompt_rows: Vec<Vec<i32>>,
    /// Built once for the run, not once per update: it owns the runtime, the
    /// HTTP client and the verdict cache, and the cache is the whole point,
    /// rebuilding it every update would repay every verdict it already holds.
    judge: Option<GroupJudge>,
    /// The reward command, and - in the persistent mode - the one worker that
    /// answers for it. Also built once for the run, and for the same kind of
    /// reason: an update scores one batch per sampling wave, so a process per
    /// batch would pay the command's startup up to `max_resample_factor` times
    /// per update, several hundred times a run, to reload what it already had.
    reward: RewardProcess,
    layout: RowLayout,
    loss_denominator: usize,
    rollouts_per_update: usize,
    horizon: SchedulerHorizon,
    /// Adaptive-KL state: a running multiplier on the base
    /// coefficient, chased toward `kl_schedule.target` between updates. Stays 1
    /// when no target is configured.
    kl_multiplier: f32,
    /// Global round-robin cursor over prompt draws. Advances by the number of
    /// *candidate* groups sampled each update - exactly `prompts_per_update`
    /// without dynamic sampling, more when zero-signal groups are
    /// resampled. Sampling seeds derive from this cursor, so the extra draws
    /// stay deterministic.
    prompt_cursor: usize,
    /// Draw slots through one cached permutation per pass rather than one per
    /// draw; a resumed run enters at its cursor and rebuilds the pass it lands
    /// in, so the sequence of slots is unchanged.
    prompt_draws: PromptDraws,
    /// Consecutive updates in which every group was dropped as zero-signal.
    stalled_updates: u32,
    /// Lines produced by the batch phases, waiting for the next progress
    /// event to carry them out. See [`Progress::notes`]: printing them here
    /// would tear the caller's progress bar.
    pending_notes: Vec<String>,
}

/// Validates the configuration against the restored boundary and builds the
/// run's fixed context. Every failure here happens before any rollout compute.
fn resume_state(
    trainer: &Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    resume: Option<Boundary>,
) -> Result<RunState> {
    config.validate()?;
    let start_update = resume.map_or(0, |boundary| boundary.completed_iterations);
    if start_update >= u64::from(config.updates) {
        return Err(Error::checkpoint(format!(
            "the checkpoint has {start_update} completed updates, at or past grpo.updates ({})",
            config.updates
        )));
    }
    // Sound by the guard above and nothing else: `config.updates` is a u32, so
    // anything strictly below it fits. The comparison is done in u64 rather
    // than the narrowing done first, otherwise an absurd checkpoint would be
    // refused for the wrong reason.
    let start_update = start_update as u32;
    if (training.n_seq_max as usize) < config.group_size {
        return Err(Error::invalid(format!(
            "training.n_seq_max ({}) must cover grpo.group_size ({})",
            training.n_seq_max, config.group_size
        )));
    }
    let prompts = read_prompts(&config.prompts)?;
    let judge = config.judge.as_ref().map(GroupJudge::new).transpose()?;
    let layout = RowLayout::resolve(trainer, training)?;
    let loss_denominator = config.sampling.max_new_tokens as usize;
    // Every prompt must leave room for the full Dr. GRPO generation budget:
    // validated (and tokenized) once, before any rollout compute.
    let prompt_rows = tokenize_prompts(
        trainer,
        &prompts,
        &layout,
        loss_denominator,
        &config.prompts,
    )?;
    let rollouts_per_update = config
        .prompts_per_update
        .checked_mul(config.group_size)
        .ok_or_else(|| retrograd_core::Error::overflow("GRPO rollout count overflows usize"))?;
    let total_steps = u64::from(config.updates)
        .checked_mul(u64::from(config.grpo_epochs))
        .and_then(|value| value.checked_mul(rollouts_per_update as u64))
        .and_then(|value| value.checked_mul(layout.steps_per_row))
        .ok_or_else(|| {
            retrograd_core::Error::overflow("GRPO optimizer step count overflows u64")
        })?;
    // `total_steps` counts every rollout slot. Filtered slots advance the
    // scheduler without optimizing, keeping warm-up and decay aligned with
    // this horizon while still avoiding zero-gradient compute. It is an upper
    // bound - grouped rollouts share a step - so it is re-sized after every
    // update on the steps actually taken, otherwise a decaying schedule would
    // end the run far above zero.
    let horizon = SchedulerHorizon::new(total_steps, training.warmup_steps);
    let prompt_draws = PromptDraws::new(config.prompt_order, prompts.len(), config.sampling.seed);
    Ok(RunState {
        start_update,
        prompts,
        prompt_rows,
        judge,
        reward: reward_process(&config.reward_command, config.reward_protocol)?,
        layout,
        loss_denominator,
        rollouts_per_update,
        horizon,
        kl_multiplier: resume
            .and_then(|boundary| boundary.kl_multiplier)
            .unwrap_or(1.0_f32),
        // The one cursor that comes back from disk rather than from this
        // process: a u64 on the wire, an index here, so it is checked instead
        // of truncated.
        prompt_cursor: match resume {
            Some(boundary) => usize::try_from(boundary.cursor).map_err(|_| {
                Error::overflow(format!(
                    "the checkpoint's prompt cursor ({}) does not fit an index on this target",
                    boundary.cursor
                ))
            })?,
            None => 0,
        },
        prompt_draws,
        stalled_updates: 0,
        pending_notes: Vec::new(),
    })
}

/// [`run`], restarting at a boundary restored from a checkpoint.
/// `resume` carries the number of completed updates and the prompt cursor;
/// both are needed because dynamic sampling makes the cursor independent of
/// the update index. The learning-rate horizon is unchanged, so a resumed run
/// follows the schedule the first launch started.
///
/// `observer` receives each wanted update's rollouts before its epochs, and
/// its outcome once they all succeeded.
pub fn run_resumed(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    resume: Option<Boundary>,
    observer: Option<&dyn TrajectoryObserver>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::grpo", "training");
    let _entered = span.enter();
    let mut state = resume_state(trainer, config, training, resume)?;
    let mut final_metrics = TrainMetrics::default();
    let mut scratch = WeightedStepScratch::new(&state.layout);
    for update in state.start_update..config.updates {
        // Effective KL coefficient for this update: base * linear-warmup *
        // adaptive multiplier. With no `kl_schedule` this is just
        // `config.kl_coefficient`. When the base coefficient is zero, the
        // fixed-reference term is identically zero and the reference forward
        // pass is skipped entirely.
        let warmup_factor = match &config.kl_schedule {
            Some(schedule) if schedule.warmup_updates > 0 => {
                (update as f32 / schedule.warmup_updates as f32).min(1.0)
            }
            _ => 1.0,
        };
        let effective_kl = config.kl_coefficient * warmup_factor * state.kl_multiplier;
        let update_start_step = final_metrics.global_step;
        let batch = assemble_batch(trainer, config, training, &mut state)?;
        let observer = observer.filter(|observer| observer.wants(update + 1));
        let baseline = build_baseline(
            trainer,
            config,
            &mut state,
            update,
            effective_kl,
            batch,
            observer,
        )?;
        if !run_epochs(
            trainer,
            config,
            &mut state,
            &baseline,
            &mut scratch,
            &mut final_metrics,
            observer,
            on_progress,
        )? {
            return Ok(final_metrics);
        }
        // Price the updates left on what this one cost, so the decay reaches
        // its end instead of stopping at the ratio of grouped to nominal steps.
        state.horizon.observe(
            final_metrics.global_step.saturating_sub(update_start_step),
            (config.updates - update - 1) as u64,
            final_metrics.global_step,
        );
    }
    Ok(final_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_config::AdvantageBaseline;

    #[test]
    fn adaptive_kl_continues_from_the_restored_multiplier() {
        assert_eq!(next_kl_multiplier(3.0, Some(0.1), 0.3), 4.5);
        assert_eq!(next_kl_multiplier(3.0, Some(0.1), 0.01), 2.0);
        assert_eq!(next_kl_multiplier(3.0, Some(0.1), 0.1), 3.0);
        assert_eq!(next_kl_multiplier(3.0, None, 10.0), 3.0);
    }

    fn mean_advantages(
        rewards: &[f32],
        group_ids: &[u64],
        live: &mut [bool],
    ) -> Result<(Vec<f32>, GroupDiagnostics)> {
        group_advantages(rewards, group_ids, live, AdvantageBaseline::Mean)
    }

    #[test]
    fn advantages_are_centered_without_std_normalization() {
        let mut live = vec![true; 4];
        let (advantages, _) =
            mean_advantages(&[1.0, 3.0, 10.0, 30.0], &[7, 7, 9, 9], &mut live).unwrap();
        assert_eq!(advantages, vec![-1.0, 1.0, -10.0, 10.0]);
        assert_eq!(live, vec![true; 4]);
    }

    #[test]
    fn leave_one_out_rescales_the_advantage_by_g_over_g_minus_one() {
        // Two live members: RLOO baselines each against the other, so the
        // advantage is the mean-centered value scaled by 2/1 = 2.
        let mut live = vec![true; 2];
        let (advantages, _) = group_advantages(
            &[0.0, 4.0],
            &[1, 1],
            &mut live,
            AdvantageBaseline::LeaveOneOut,
        )
        .unwrap();
        assert_eq!(advantages, vec![-4.0, 4.0]);

        // Three live members: scale 3/2. Mean-centered [-2, 0, 2] -> [-3, 0, 3].
        let mut live = vec![true; 3];
        let (advantages, _) = group_advantages(
            &[0.0, 2.0, 4.0],
            &[1, 1, 1],
            &mut live,
            AdvantageBaseline::LeaveOneOut,
        )
        .unwrap();
        assert_eq!(advantages, vec![-3.0, 0.0, 3.0]);
    }

    #[test]
    fn uniform_group_is_filtered_out() {
        // The first group has no reward spread: it must contribute exactly
        // zero signal and be marked dead so the epochs skip it entirely;
        // the second group still centers normally.
        let mut live = vec![true; 6];
        let (advantages, _) = mean_advantages(
            &[5.0, 5.0, 5.0, 0.0, 1.0, 2.0],
            &[1, 1, 1, 2, 2, 2],
            &mut live,
        )
        .unwrap();
        assert_eq!(&advantages[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(live, vec![false, false, false, true, true, true]);
        assert!(advantages[3] < 0.0 && advantages[5] > 0.0, "{advantages:?}");
    }

    #[test]
    fn group_reward_std_averages_the_per_group_spread() {
        // Group stds: 1 (mean 2 over [1, 3]) and 0 (uniform); mean 0.5.
        let mut live = vec![true; 4];
        let (_, diagnostics) =
            mean_advantages(&[1.0, 3.0, 7.0, 7.0], &[1, 1, 2, 2], &mut live).unwrap();
        assert!((diagnostics.mean_reward_std - 0.5).abs() < 1e-5);
        assert!((diagnostics.zero_std_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn group_advantages_are_batch_mean_free_per_group() {
        let mut live = vec![true; 6];
        let (advantages, _) = mean_advantages(
            &[0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
            &[1, 1, 1, 2, 2, 2],
            &mut live,
        )
        .unwrap();
        for group in advantages.chunks(3) {
            let mean = group.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "{advantages:?}");
        }
    }

    #[test]
    fn masked_member_is_excluded_from_the_group_baseline() {
        // The truncated third member (reward 10) must not pollute the mean of
        // the two live members, and must keep a zero advantage.
        let mut live = vec![true, true, false];
        let (advantages, _) = mean_advantages(&[0.0, 1.0, 10.0], &[7, 7, 7], &mut live).unwrap();
        assert_eq!(advantages, vec![-0.5, 0.5, 0.0]);
        assert_eq!(live, vec![true, true, false]);
    }

    #[test]
    fn group_without_two_live_members_is_dead() {
        // One live member has no baseline: the whole group is dead and counts
        // as a zero-signal group.
        let mut live = vec![true, false, false, true, true, false];
        let (advantages, diagnostics) = mean_advantages(
            &[1.0, 2.0, 3.0, 0.0, 4.0, 9.0],
            &[1, 1, 1, 2, 2, 2],
            &mut live,
        )
        .unwrap();
        assert_eq!(&advantages[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(live, vec![false, false, false, true, true, false]);
        assert_eq!(advantages[3], -2.0);
        assert_eq!(advantages[4], 2.0);
        assert!((diagnostics.zero_std_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_dead_group_says_which_of_the_two_causes_killed_it() {
        // Group 1 is uniform, group 2 is starved by the liveness mask. The two
        // send the reader to different knobs, so the diagnostic separates them
        // instead of asserting the first.
        let mut live = vec![true, true, true, false];
        let (_, diagnostics) =
            mean_advantages(&[5.0, 5.0, 1.0, 2.0], &[1, 1, 2, 2], &mut live).unwrap();
        assert_eq!(diagnostics.uniform_groups, 1);
        assert_eq!(diagnostics.starved_groups, 1);
        let cause = zero_signal_cause(&diagnostics);
        assert!(
            cause.contains("identical reward") && cause.contains("fewer than two"),
            "{cause}"
        );

        let mut live = vec![true; 4];
        let (_, uniform) =
            mean_advantages(&[5.0, 5.0, 1.0, 1.0], &[1, 1, 2, 2], &mut live).unwrap();
        assert_eq!(uniform.starved_groups, 0);
        assert!(!zero_signal_cause(&uniform).contains("fewer than two"));

        let mut live = vec![true, false, true, false];
        let (_, starved) =
            mean_advantages(&[5.0, 5.0, 1.0, 1.0], &[1, 1, 2, 2], &mut live).unwrap();
        assert_eq!(starved.uniform_groups, 0);
        assert!(!zero_signal_cause(&starved).contains("identical reward"));
    }

    #[test]
    fn the_reward_split_sums_back_to_the_published_mean() {
        // Without a judge the terms are zero, so the whole reward is the
        // verifiable half - which is what keeps a judge-free run comparable
        // with one that has a judge.
        let (verifiable, judge) = reward_split(0.42, &[0.0; 4]);
        assert_eq!(judge, 0.0);
        assert!((verifiable - 0.42).abs() < 1e-6, "{verifiable}");

        // With one, the two halves are the published `reward/mean` - the
        // identity `eval/mean_reward` is compared against.
        let terms = [0.30, 0.0, 0.15, 0.15];
        let mean = 0.42_f64;
        let (verifiable, judge) = reward_split(mean, &terms);
        assert!((judge - 0.15).abs() < 1e-6, "{judge}");
        assert!(
            (f64::from(verifiable) + f64::from(judge) - mean).abs() < 1e-6,
            "{verifiable} + {judge} != {mean}"
        );
    }

    #[test]
    fn unrepresentable_centered_reward_is_rejected() {
        let mut live = vec![true; 3];
        assert!(mean_advantages(&[f32::MAX, f32::MAX, -f32::MAX], &[1, 1, 1], &mut live).is_err());
    }

    #[test]
    fn mismatched_live_mask_is_rejected() {
        let mut live = vec![true; 3];
        assert!(mean_advantages(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2], &mut live).is_err());
    }

    #[test]
    fn non_contiguous_and_uneven_groups_are_supported() {
        let mut live = vec![true; 5];
        let (advantages, _) =
            mean_advantages(&[0.0, 10.0, 2.0, 14.0, 4.0], &[1, 2, 1, 2, 1], &mut live).unwrap();
        assert_eq!(advantages, vec![-2.0, -2.0, 0.0, 2.0, 2.0]);
        assert_eq!(live, vec![true; 5]);
    }

    #[test]
    fn shuffle_is_deterministic_and_seeded() {
        let first = shuffled_indices(16, 7);
        assert_eq!(first, shuffled_indices(16, 7));
        assert_ne!(first, shuffled_indices(16, 8));
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn sequential_prompt_order_is_a_strict_round_robin() {
        use retrograd_config::PromptOrder;
        let slots: Vec<usize> = (0..7)
            .map(|offset| prompt_slot_for(&PromptOrder::Sequential, offset, 3, 42))
            .collect();
        assert_eq!(slots, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn shuffled_prompt_order_permutes_each_pass_and_covers_every_prompt() {
        use retrograd_config::PromptOrder;
        let len = 5;
        let first_pass: Vec<usize> = (0..len)
            .map(|offset| prompt_slot_for(&PromptOrder::Shuffled, offset, len, 9))
            .collect();
        let second_pass: Vec<usize> = (len..2 * len)
            .map(|offset| prompt_slot_for(&PromptOrder::Shuffled, offset, len, 9))
            .collect();
        // Each pass is a full permutation of 0..len.
        let mut sorted = first_pass.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..len).collect::<Vec<_>>());
        // Different passes draw different orders, and the draw is deterministic.
        assert_ne!(first_pass, second_pass);
        assert_eq!(
            first_pass,
            (0..len)
                .map(|offset| prompt_slot_for(&PromptOrder::Shuffled, offset, len, 9))
                .collect::<Vec<_>>()
        );
    }

    fn rollout_of_len(completion_len: usize) -> Rollout {
        // A rollout whose completion spans `completion_len` trainable tokens;
        // only the mask matters for the dynamic-sampling signal test.
        let total = completion_len + 1;
        Rollout {
            tokens: vec![0; total],
            train_mask: (0..total).map(|index| index >= 1).collect(),
            old_logprobs: Vec::new(),
        }
    }

    #[test]
    fn group_signal_matches_the_group_advantages_filter() {
        // Two live members with different rewards: signal.
        let rollouts = [rollout_of_len(4), rollout_of_len(4)];
        assert!(group_has_signal(&rollouts, &[0.0, 1.0], false, None, 8));
        // Identical rewards: no signal, matching the uniform-group filter.
        assert!(!group_has_signal(&rollouts, &[2.0, 2.0], false, None, 8));
        // A single member never has a baseline.
        assert!(!group_has_signal(&rollouts[..1], &[1.0], false, None, 8));
    }

    #[test]
    fn group_signal_respects_truncation_masking() {
        // The second member is truncated at the 8-token budget: with masking it
        // drops out, leaving a single live member, so the group has no signal.
        let rollouts = [rollout_of_len(4), rollout_of_len(8)];
        assert!(!group_has_signal(&rollouts, &[0.0, 1.0], true, None, 8));
        // Without masking both count and the spread is real.
        assert!(group_has_signal(&rollouts, &[0.0, 1.0], false, None, 8));
    }

    #[test]
    fn group_signal_uses_penalized_rewards() {
        // Raw rewards are equal, but the overlong penalty only hits the longer
        // completion, creating a spread the optimizer would see.
        let rollouts = [rollout_of_len(4), rollout_of_len(8)];
        let penalty = retrograd_config::OverlongPenalty {
            buffer_tokens: 4,
            max_penalty: 1.0,
        };
        assert!(group_has_signal(
            &rollouts,
            &[1.0, 1.0],
            false,
            Some(&penalty),
            8
        ));
    }

    #[test]
    fn distinct_fraction_averages_per_group_uniqueness() {
        let completions = vec![
            "a".to_string(), // group 1: all identical -> 1/3
            "a".to_string(),
            "a".to_string(),
            "x".to_string(), // group 2: all distinct -> 3/3
            "y".to_string(),
            "z".to_string(),
        ];
        let group_ids = vec![1, 1, 1, 2, 2, 2];
        let fraction = distinct_completion_fraction(&completions, &group_ids);
        // Mean of 1/3 and 1 is 2/3.
        assert!((fraction - 2.0 / 3.0).abs() < 1e-6, "{fraction}");
    }
}
