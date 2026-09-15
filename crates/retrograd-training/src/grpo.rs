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

macro_rules! metric {
    ($name:literal, $value:expr_2021) => {
        MetricValue {
            name: $name.into(),
            value: $value,
        }
    };
}

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

use std::time::Instant;

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
        // Collected once rather than re-derived from a cloned iterator for each
        // of the three reductions below.
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

/// Resolves the prompt-file slot for a global draw index. Sequential is the
/// unchanged round-robin; shuffled re-draws a seeded permutation each full
/// pass over the dataset (item 10), so correlated adjacent prompts do not
/// always land in the same update. The permutation is stable within a pass.
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
/// The permutation is a function of the pass alone, so recomputing it per draw
/// meant a full Fisher-Yates over the whole prompt file - and one allocation of
/// its size - for a single index. Caching it changes nothing observable: the
/// drawn slots are identical, bit for bit.
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

/// Mean over groups of the fraction of distinct completions in each group
/// (item 7). 1.0 means every member differs; low values flag degenerate
/// sampling and the payoff of deduplication (item 15).
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

/// DAPO soft overlong punishment (item 8): the reward reduction for a
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
/// `group_advantages` so dynamic sampling (item 3) keeps exactly the groups the
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

/// Appends one JSONL record per rollout to the completion journal (item 11).
#[allow(clippy::too_many_arguments)]
fn append_completion_log(
    path: &std::path::Path,
    update: u32,
    prompt_indices: &[usize],
    completions: &[String],
    seeds: &[u32],
    rewards: &[f32],
    advantages: &[f32],
    live: &[bool],
) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| {
            Error::runtime(format!("open completion log {}: {error}", path.display()))
        })?;
    let mut buffer = String::new();
    for index in 0..completions.len() {
        let record = serde_json::json!({
            "update": update,
            "prompt_index": prompt_indices[index],
            "seed": seeds[index],
            "completion": completions[index],
            "reward": rewards[index],
            "advantage": advantages[index],
            "live": live[index],
        });
        buffer.push_str(&record.to_string());
        buffer.push('\n');
    }
    file.write_all(buffer.as_bytes())
        .map_err(|error| Error::runtime(format!("write completion log: {error}")))?;
    Ok(())
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
    run_controlled(trainer, config, training, &mut |_, progress| {
        if progress.metrics.epoch_complete {
            on_progress(progress);
        }
        Ok(true)
    })
}

pub fn run_controlled(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    run_resumed(trainer, config, training, None, on_progress)
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
    /// Adaptive-KL state (item 18): a running multiplier on the base
    /// coefficient, chased toward `kl_schedule.target` between updates. Stays 1
    /// when no target is configured.
    kl_multiplier: f32,
    /// Global round-robin cursor over prompt draws. Advances by the number of
    /// *candidate* groups sampled each update - exactly `prompts_per_update`
    /// without dynamic sampling (item 3), more when zero-signal groups are
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

/// [`run_controlled`], restarting at a boundary restored from a checkpoint.
/// `resume` carries the number of completed updates and the prompt cursor;
/// both are needed because dynamic sampling makes the cursor independent of
/// the update index. The learning-rate horizon is unchanged, so a resumed run
/// follows the schedule the first launch started.
pub fn run_resumed(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    resume: Option<Boundary>,
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
        // pass is skipped entirely (item 1).
        let warmup_factor = match &config.kl_schedule {
            Some(schedule) if schedule.warmup_updates > 0 => {
                (update as f32 / schedule.warmup_updates as f32).min(1.0)
            }
            _ => 1.0,
        };
        let effective_kl = config.kl_coefficient * warmup_factor * state.kl_multiplier;
        let update_start_step = final_metrics.global_step;
        let batch = assemble_batch(trainer, config, training, &mut state)?;
        let baseline = build_baseline(trainer, config, &mut state, update, effective_kl, batch)?;
        if !run_epochs(
            trainer,
            config,
            &mut state,
            &baseline,
            &mut scratch,
            &mut final_metrics,
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

/// Wall-clock split of one update's batch assembly, in seconds.
#[derive(Clone, Copy, Debug, Default)]
struct BatchTiming {
    generation: f32,
    behavior_scoring: f32,
    reward: f32,
    judge: f32,
}

/// One update's assembled batch: `prompts_per_update * group_size` rollout
/// slots in group order, one entry per slot in every vector.
struct TrainingBatch {
    rollouts: Vec<Rollout>,
    prompt_indices: Vec<usize>,
    completions: Vec<String>,
    member_seeds: Vec<u32>,
    rewards: Vec<f32>,
    /// What the judge added to each entry of `rewards`, i.e. `verdict *
    /// weight`; zero where no verdict applied and everywhere without a judge.
    /// Same length and same order as `rewards`, so `rewards[i] -
    /// judge_terms[i]` is what the reward command returned for that rollout.
    ///
    /// Carried rather than recomputed because the verdict is gone by the time
    /// the metrics are assembled, and published because `reward/mean` alone
    /// cannot be compared with `eval/mean_reward`: the evaluation drops the
    /// judge deliberately (there is no group to rank), so the two are different
    /// quantities until this splits them apart.
    judge_terms: Vec<f32>,
    /// Whether each rollout's reward carries the judge term the others carry.
    /// All true without a judge, and all true with one until it fails a group:
    /// a reward missing that term is not on the batch's scale, so the members
    /// it belongs to stay out of the baseline and the epochs.
    judged_groups: Vec<bool>,
    /// Candidate groups drawn to fill the batch - exactly `prompts_per_update`
    /// without dynamic sampling, more when zero-signal groups were resampled.
    groups_sampled: usize,
    judge_tally: JudgeTally,
    /// Behavior-scorer counters attributed to this update alone.
    scoring_stats: ScoringStats,
    timing: BatchTiming,
}

/// The scalar series one update publishes. Computed once, in phase 2, and
/// re-emitted by every epoch of the update.
#[derive(Clone, Copy, Debug, Default)]
struct BatchMetrics {
    mean_reward: f32,
    /// `mean_reward` split in two: the reward command's own verdict, and what
    /// the judge added on top. They sum back to `mean_reward`.
    verifiable_reward_mean: f32,
    judge_reward_mean: f32,
    reward_std: f32,
    reward_min: f32,
    reward_max: f32,
    group_diagnostics: GroupDiagnostics,
    advantage_abs_mean: f64,
    completion_length_mean: f32,
    completion_length_min: f32,
    completion_length_max: f32,
    truncation_fraction: f32,
    trained_fraction: f32,
    groups_sampled_fraction: f32,
    distinct_fraction: f32,
    entropy: f32,
    /// Completion tokens over the live slots - the numerator of
    /// `tokens_per_second`.
    trained_tokens: usize,
    scoring_stats: ScoringStats,
    reference_seconds: f32,
    timing: BatchTiming,
}

/// One update's batch, centered: the group-relative advantages, the live slots
/// and the frozen-reference rows the epochs read.
///
/// Phase 2 consumes the [`TrainingBatch`] rather than borrowing it, which is
/// what frees the per-slot text buffers - completions, seeds, prompt indices,
/// before the epochs allocate anything: they are needed only up to the
/// completion journal.
struct UpdateBaseline {
    update: u32,
    /// KL coefficient in force for this update: base * warm-up * adaptive
    /// multiplier. Zero means the reference pass was skipped and
    /// `reference_rows` is empty.
    effective_kl: f32,
    rollouts: Vec<Rollout>,
    group_ids: Vec<u64>,
    advantages: Vec<f32>,
    /// Ascending indices of the slots that reach the optimizer.
    trainable: Vec<usize>,
    reference_rows: Vec<Vec<f32>>,
    /// Per-slot position into `reference_rows`; `None` marks a dead slot, so
    /// this doubles as the liveness mask the epochs read.
    reference_positions: Vec<Option<usize>>,
    judge_tally: JudgeTally,
    metrics: BatchMetrics,
}

/// Phase 1 of an update - assemble the training batch.
///
/// Sample `group_size` completions per prompt - each member with its own
/// derived seed - and score them. With dynamic sampling (item 3), groups
/// without reward spread are discarded and replaced by the next prompts in the
/// round-robin, up to `max_resample_factor * prompts_per_update` candidates, so
/// the trained batch stays full of informative groups as the policy converges.
/// Without it, exactly `prompts_per_update` groups are sampled and all kept.
///
/// Advances `state.prompt_cursor` past every candidate drawn, informative or
/// not: that cursor is what a boundary carries, and resampling is part of the
/// deterministic draw sequence.
fn assemble_batch(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    training: &TrainConfig,
    state: &mut RunState,
) -> Result<TrainingBatch> {
    let layout = state.layout;
    let loss_denominator = state.loss_denominator;
    let rollouts_per_update = state.rollouts_per_update;
    let judge = state.judge.as_ref();
    let prompts = &state.prompts;
    let prompt_rows = &state.prompt_rows;
    let reward = &mut state.reward;
    struct Candidate {
        rollouts: Vec<Rollout>,
        completions: Vec<String>,
        seeds: Vec<u32>,
        prompt_slot: usize,
    }
    struct SpareGroup {
        rollouts: Vec<Rollout>,
        completions: Vec<String>,
        seeds: Vec<u32>,
        rewards: Vec<f32>,
        /// Rides with `rewards` so a group held aside as padding keeps its
        /// reward decomposition intact.
        judge_terms: Vec<f32>,
        prompt_slot: usize,
        judged: bool,
    }
    let mut rollouts = Vec::with_capacity(rollouts_per_update);
    let mut prompt_indices = Vec::with_capacity(rollouts_per_update);
    let mut completions = Vec::with_capacity(rollouts_per_update);
    let mut member_seeds = Vec::with_capacity(rollouts_per_update);
    let mut rewards: Vec<f32> = Vec::with_capacity(rollouts_per_update);
    let mut judge_terms: Vec<f32> = Vec::with_capacity(rollouts_per_update);
    // Whether each rollout's reward carries the judge term the others carry.
    // All true without a judge, and all true with one until it fails a
    // group: a reward missing that term is not on the batch's scale, so the
    // members it belongs to stay out of the baseline and the epochs.
    let mut judged_groups: Vec<bool> = Vec::with_capacity(rollouts_per_update);
    // Zero-signal candidates held aside to pad the batch to a full
    // `prompts_per_update` groups if the resample cap is hit first. Padding
    // groups are dead: they advance only the scheduler, preserving the
    // fixed-horizon LR timeline.
    let mut spare = Vec::<SpareGroup>::new();
    let mut generation_seconds = 0.0_f32;
    let mut behavior_scoring_seconds = 0.0_f32;
    let mut reward_seconds = 0.0_f32;
    let mut judge_seconds = 0.0_f32;
    // Summed over the update's sampling waves, not kept per wave: the
    // failure threshold and every `judge/*` series are the update's.
    let mut judge_tally = JudgeTally::default();
    let mut kept_groups = 0_usize;
    let mut groups_sampled = 0_usize;
    let max_candidate_groups = match &config.dynamic_sampling {
        Some(dynamic) => config
            .prompts_per_update
            .saturating_mul(dynamic.max_resample_factor),
        None => config.prompts_per_update,
    };
    while kept_groups < config.prompts_per_update && groups_sampled < max_candidate_groups {
        let round =
            (config.prompts_per_update - kept_groups).min(max_candidate_groups - groups_sampled);
        let sampling_started = Instant::now();
        let mut candidate_inputs = Vec::with_capacity(round);
        let mut candidate_slots = Vec::with_capacity(round);
        for _ in 0..round {
            let prompt_offset = state.prompt_cursor;
            state.prompt_cursor += 1;
            groups_sampled += 1;
            let prompt_slot = state.prompt_draws.slot(prompt_offset);
            let prompt_tokens = &prompt_rows[prompt_slot];
            let mut seed_offsets = Vec::with_capacity(config.group_size);
            for member in 0..config.group_size {
                let seed_offset = prompt_offset
                    .checked_mul(config.group_size)
                    .and_then(|value| value.checked_add(member))
                    .ok_or_else(|| Error::overflow("GRPO sampling seed offset overflows usize"))?;
                seed_offsets.push(seed_offset);
            }
            candidate_slots.push(prompt_slot);
            candidate_inputs.push((prompt_tokens.as_slice(), seed_offsets));
        }
        let sampled_groups = sample_rollout_groups_continuous(
            trainer,
            &candidate_inputs,
            &config.sampling,
            &layout,
            if training.generation_concurrency == 0 {
                training.n_seq_max as usize
            } else {
                training.generation_concurrency as usize
            },
        )?;
        let mut candidates: Vec<Candidate> = Vec::with_capacity(round);
        for ((prompt_slot, (_, seed_offsets)), group) in candidate_slots
            .into_iter()
            .zip(candidate_inputs)
            .zip(sampled_groups)
        {
            let mut group_rollouts = Vec::with_capacity(config.group_size);
            let mut group_completions = Vec::with_capacity(config.group_size);
            let mut group_seeds = Vec::with_capacity(config.group_size);
            for (seed_offset, (rollout, completion_text)) in seed_offsets.into_iter().zip(group) {
                // Truncating, and deliberately so: the offset is a bit pattern
                // feeding a seed, not a size. `wrapping_add` already says the
                // sum may go round, and the run that reaches 2^32 draws is the
                // one where the sampler repeats a seed - not one that should
                // refuse to continue. The `checked_mul` above guards the index
                // arithmetic, which is a size.
                group_seeds.push(config.sampling.seed.wrapping_add(seed_offset as u32));
                group_completions.push(completion_text);
                group_rollouts.push(rollout);
            }
            candidates.push(Candidate {
                rollouts: group_rollouts,
                completions: group_completions,
                seeds: group_seeds,
                prompt_slot,
            });
        }
        generation_seconds += sampling_started.elapsed().as_secs_f32();

        // Score the whole round in a single reward call, then keep the
        // informative groups (all of them, when dynamic sampling is off).
        let reward_started = Instant::now();
        let prompt_list = &prompts;
        let pairs = candidates.iter().flat_map(|candidate| {
            let slot = candidate.prompt_slot;
            candidate
                .completions
                .iter()
                .map(move |completion| (prompt_list[slot].reward_text(), completion.as_str()))
        });
        let round_rows = match &judge {
            Some(_) => score_rows(reward, pairs)?,
            // No judge: the scalar path, which refuses a `judge_weight`
            // instead of dropping one silently on the floor.
            None => score(reward, pairs)?
                .into_iter()
                .map(|reward| RewardRow {
                    reward,
                    judge_weight: None,
                })
                .collect(),
        };
        reward_seconds += reward_started.elapsed().as_secs_f32();

        // The verdicts, one request per group, blended into what the reward
        // process returned. A group the judge could not score keeps its
        // verifiable reward and is marked here - it will be excluded from
        // the baseline and the epochs below, rather than trained on a scale
        // no other group shares.
        let judge_started = Instant::now();
        let (round_rewards, round_judge_terms, round_judged) = match &judge {
            Some(judge) => {
                let weights = round_rows
                    .iter()
                    .map(|row| row.judge_weight.unwrap_or(judge.weight()))
                    .collect::<Vec<_>>();
                // A group whose every member weighs the verdict at zero is
                // not sent: there is nothing for an opinion to change, so
                // paying for one - and letting a judge outage cost the group
                // or the run - would be spending the update's failure budget
                // on an answer nobody would have read.
                let asked = weights
                    .chunks(config.group_size)
                    .map(|group| group.iter().any(|weight| *weight > 0.0))
                    .collect::<Vec<_>>();
                let groups = candidates
                    .iter()
                    .zip(&asked)
                    .filter(|(_, asked)| **asked)
                    .map(|(candidate, _)| JudgeGroup {
                        group_id: candidate.prompt_slot as u64,
                        prompt: &prompts[candidate.prompt_slot],
                        completions: &candidate.completions,
                    })
                    .collect::<Vec<_>>();
                let mut scored = if groups.is_empty() {
                    Vec::new()
                } else {
                    judge.score(&groups, &mut judge_tally)?
                }
                .into_iter();
                // Back to one entry per rollout of the round. A group that
                // was not sent counts as judged - its reward is complete
                // without a verdict - and contributes none.
                let mut verdicts = Vec::with_capacity(round_rows.len());
                let mut judged = Vec::with_capacity(asked.len());
                for asked in &asked {
                    let members = (0..config.group_size)
                        .map(|_| {
                            if *asked {
                                scored.next().flatten()
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    judged.push(!*asked || members.iter().all(Option::is_some));
                    verdicts.extend(members);
                }
                // The judge term is kept apart rather than folded in and
                // forgotten: it is the only place the two halves of a reward
                // are still separable, and `reward/judge_mean` is what makes
                // `reward/mean` comparable with the judge-free evaluation.
                let terms = weights
                    .iter()
                    .zip(&verdicts)
                    .map(|(weight, verdict)| verdict.unwrap_or(0.0) * weight)
                    .collect::<Vec<_>>();
                let blended = round_rows
                    .iter()
                    .zip(&terms)
                    .map(|(row, term)| row.reward + term)
                    .collect::<Vec<_>>();
                (blended, terms, judged)
            }
            None => (
                round_rows.iter().map(|row| row.reward).collect::<Vec<_>>(),
                vec![0.0; round_rows.len()],
                vec![true; candidates.len()],
            ),
        };
        judge_seconds += judge_started.elapsed().as_secs_f32();

        let mut offset = 0_usize;
        for (candidate, judged) in candidates.into_iter().zip(round_judged) {
            let group_rewards = &round_rewards[offset..offset + config.group_size];
            let group_judge_terms = &round_judge_terms[offset..offset + config.group_size];
            offset += config.group_size;
            // Without dynamic sampling every group is kept; zero-signal
            // groups are marked dead later by `group_advantages`, exactly as
            // before. With it, only informative groups are kept and the rest
            // become potential padding. A group the judge dropped is never
            // *chosen*: its reward is missing the term every other group
            // carries, so resampling has a real one to prefer.
            let keep = config.dynamic_sampling.is_none()
                || (kept_groups < config.prompts_per_update
                    && judged
                    && group_has_signal(
                        &candidate.rollouts,
                        group_rewards,
                        config.mask_truncated,
                        config.overlong_penalty.as_ref(),
                        loss_denominator,
                    ));
            if keep {
                kept_groups += 1;
                prompt_indices.extend(std::iter::repeat_n(
                    candidate.prompt_slot,
                    config.group_size,
                ));
                completions.extend(candidate.completions);
                member_seeds.extend(candidate.seeds);
                rollouts.extend(candidate.rollouts);
                rewards.extend_from_slice(group_rewards);
                judge_terms.extend_from_slice(group_judge_terms);
                judged_groups.extend(std::iter::repeat_n(judged, config.group_size));
            } else {
                spare.push(SpareGroup {
                    rollouts: candidate.rollouts,
                    completions: candidate.completions,
                    seeds: candidate.seeds,
                    rewards: group_rewards.to_vec(),
                    judge_terms: group_judge_terms.to_vec(),
                    prompt_slot: candidate.prompt_slot,
                    judged,
                });
            }
        }
    }
    // Every wave is judged by now, so this is the update's real loss,
    // which is the scale `max_judge_dropped_fraction` is written on. A judge
    // that answered nothing stops the run here rather than leaving an update
    // trained on whatever the verifiable part alone could separate.
    if let Some(judge) = &config.judge {
        judge_tally.check(judge.max_dropped_fraction)?;
    }
    // Resample cap reached before a full batch: pad with the held-aside
    // zero-signal candidates so exactly `prompts_per_update` groups reach
    // the epochs (the padding is dead, and `group_advantages` confirms it).
    if kept_groups < config.prompts_per_update {
        let pad_groups = (config.prompts_per_update - kept_groups).min(spare.len());
        for group in spare.drain(..pad_groups) {
            prompt_indices.extend(std::iter::repeat_n(group.prompt_slot, group.rollouts.len()));
            judged_groups.extend(std::iter::repeat_n(group.judged, group.rollouts.len()));
            completions.extend(group.completions);
            member_seeds.extend(group.seeds);
            rollouts.extend(group.rollouts);
            rewards.extend(group.rewards);
            judge_terms.extend(group.judge_terms);
        }
    }
    // Behavior-policy logprobs for the assembled batch, teacher-forced under
    // the current policy. Order-independent (no optimizer step happened), so
    // this matches the previous per-group timing; counted as sampling.
    let old_logprobs_started = Instant::now();
    let scoring_stats_before = trainer.scoring_stats()?;
    for group in rollouts.chunks_mut(config.group_size) {
        let scores = {
            let members = group.iter().collect::<Vec<_>>();
            score_train_mask_group(trainer, &members)?
        };
        for (rollout, scores) in group.iter_mut().zip(scores) {
            rollout.old_logprobs = scores;
        }
    }
    behavior_scoring_seconds += old_logprobs_started.elapsed().as_secs_f32();
    let scoring_stats = trainer.scoring_stats()?.delta_since(scoring_stats_before);
    Ok(TrainingBatch {
        rollouts,
        prompt_indices,
        completions,
        member_seeds,
        rewards,
        judge_terms,
        judged_groups,
        groups_sampled,
        judge_tally,
        scoring_stats,
        timing: BatchTiming {
            generation: generation_seconds,
            behavior_scoring: behavior_scoring_seconds,
            reward: reward_seconds,
            judge: judge_seconds,
        },
    })
}

/// Phase 2 of an update - reward stats and the group-relative baseline.
///
/// Each reward is centered against its own group's mean. Reward std remains a
/// collapse diagnostic but is deliberately absent from the advantage. This is
/// also where the DAPO shaping happens (overlong penalty, truncation masking)
/// and where a run that has stalled for too long stops.
fn build_baseline(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    state: &mut RunState,
    update: u32,
    effective_kl: f32,
    batch: TrainingBatch,
) -> Result<UpdateBaseline> {
    let TrainingBatch {
        rollouts,
        prompt_indices,
        completions,
        member_seeds,
        mut rewards,
        judge_terms,
        judged_groups,
        groups_sampled,
        judge_tally,
        scoring_stats,
        timing,
    } = batch;
    let loss_denominator = state.loss_denominator;
    let mut stalled_updates = state.stalled_updates;
    // Candidate groups drawn per full batch, relative to `prompts_per_update`
    // (>= 1). Climbs above 1 as dynamic sampling resamples zero-signal
    // groups near convergence - the counterpart of `batch/trained_fraction`.
    let groups_sampled_fraction = groups_sampled as f32 / config.prompts_per_update as f32;
    let (mean_reward_f64, reward_std_f64, _) = mean_std(rewards.iter().copied());
    let mean_reward = mean_reward_f64 as f32;
    let reward_std = reward_std_f64.min(f32::MAX as f64) as f32;
    let reward_min = rewards.iter().copied().fold(f32::INFINITY, f32::min);
    let reward_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // `reward/mean` split into the two halves it sums: what the reward command
    // demonstrated on its own, and what the judge added. Both on the raw
    // rewards, like the statistics above, so `verifiable + judge == mean` holds
    // exactly. Published in every run, judge or not - a decomposition that
    // disappeared without a judge would make two runs incomparable, and a
    // constant zero is the answer "the judge added nothing", not noise.
    debug_assert_eq!(
        rewards.len(),
        judge_terms.len(),
        "the judge terms must stay aligned with the rewards they were blended into"
    );
    let (verifiable_reward_mean, judge_reward_mean) = reward_split(mean_reward_f64, &judge_terms);
    // DAPO soft overlong punishment (item 8): a progressive penalty over
    // the last `buffer_tokens` of the budget, applied to the reward before
    // the group baseline so the policy learns to conclude rather than run
    // to the truncation limit. Reported reward stats above stay raw.
    if let Some(penalty) = &config.overlong_penalty {
        for (reward, rollout) in rewards.iter_mut().zip(&rollouts) {
            *reward -= overlong_shaped_penalty(penalty, rollout.completion_len(), loss_denominator);
        }
    }
    // DAPO overlong filtering: a completion that used the entire budget
    // was truncated (or indistinguishable from truncated), so its reward
    // judges an incomplete response. When masking is on, it is excluded
    // from the group baseline and from the optimizer epochs.
    let mut live = vec![true; rollouts.len()];
    if config.mask_truncated {
        for (flag, rollout) in live.iter_mut().zip(&rollouts) {
            *flag = !is_truncated(rollout.completion_len(), loss_denominator);
        }
    }
    // A group the judge dropped: its members' rewards are the verifiable
    // part alone, a different scale from every other group in the update.
    // Dead, exactly like a group without spread - the loop already knows how
    // to walk past those, and `batch/trained_fraction` already reports them.
    for (flag, judged) in live.iter_mut().zip(&judged_groups) {
        *flag &= judged;
    }
    let group_ids = (0..config.prompts_per_update)
        .flat_map(|group| std::iter::repeat_n(group as u64, config.group_size))
        .collect::<Vec<_>>();
    let (advantages, group_diagnostics) =
        group_advantages(&rewards, &group_ids, &mut live, config.baseline)?;
    // Fraction of unique completions per group (item 7): a precursor to
    // deduplication. Averaged over groups.
    let distinct_fraction = distinct_completion_fraction(&completions, &group_ids);
    // Journal of sampled completions (item 11): dumped every N updates for
    // offline reward-hacking / collapse inspection, before the per-update
    // buffers are dropped.
    if let Some(log) = &config.log_completions
        && update.is_multiple_of(log.every)
    {
        append_completion_log(
            &log.path,
            update,
            &prompt_indices,
            &completions,
            &member_seeds,
            &rewards,
            &advantages,
            &live,
        )?;
    }
    drop(prompt_indices);
    drop(completions);
    drop(member_seeds);
    // Zero-signal groups (and masked truncations) are dead by now: only
    // live rollouts are stepped, so no compute goes to KL-only updates.
    let trainable: Vec<usize> = (0..rollouts.len()).filter(|&index| live[index]).collect();
    // An update where nothing is trainable is legitimate: a model that has
    // not yet earned a single reward gets identical rewards everywhere, and
    // that is an early state, not a failure. What it is not is visible,
    // every series reads zero, exactly like a healthy idle epoch - so each
    // one is announced. Only a long run of them stops the loop, and
    // `max_stalled_updates == 0` disables even that.
    stalled_updates = if trainable.is_empty() {
        stalled_updates + 1
    } else {
        0
    };
    if trainable.is_empty() {
        let budget = match config.max_stalled_updates {
            0 => "in a row, no limit".to_string(),
            limit => format!("of {limit} tolerated in a row"),
        };
        // `update` is the zero-based loop index; every other display - the
        // table rows, the metric epoch - reads `update + 1`, and a note that
        // numbered itself differently sent the reader to the wrong row.
        state.pending_notes.push(format!(
            "update {}: no group carried a learning signal ({stalled_updates} \
             {budget}) - {}.",
            update + 1,
            zero_signal_cause(&group_diagnostics)
        ));
    }
    if config.max_stalled_updates != 0 && stalled_updates >= config.max_stalled_updates {
        return Err(Error::runtime(format!(
            "no group carried a learning signal for {stalled_updates} consecutive updates: \
             {}. {} If instead the model simply has not started earning reward yet, raise \
             grpo.max_stalled_updates (0 never stops).",
            zero_signal_cause(&group_diagnostics),
            if group_diagnostics.starved_groups >= group_diagnostics.uniform_groups {
                "Nothing reached the reward, so the reward is not what to look at: check \
                 completions/truncation_fraction and completions/length_mean - with \
                 grpo.mask_truncated on, completions that run to grpo.sampling.max_new_tokens \
                 are removed from the batch rather than penalized, so a policy drifting \
                 longer empties every group. judge/dropped_group_fraction is the other way \
                 a group leaves before it is scored."
            } else {
                "Either the policy collapsed - check policy/kl and policy/clip_fraction on \
                 the preceding updates - or it converged and the reward can no longer \
                 separate two samples, in which case the run is done and the best checkpoint \
                 holds the result."
            }
        )));
    }
    let trained_fraction = trainable.len() as f32 / rollouts.len() as f32;
    let advantage_abs_mean = if trainable.is_empty() {
        0.0
    } else {
        trainable
            .iter()
            .map(|&index| advantages[index].abs() as f64)
            .sum::<f64>()
            / trainable.len() as f64
    };
    let completion_tokens: usize = rollouts.iter().map(Rollout::completion_len).sum();
    let trained_tokens: usize = trainable
        .iter()
        .map(|&index| rollouts[index].completion_len())
        .sum();
    let completion_length_mean = completion_tokens as f32 / rollouts.len() as f32;
    let completion_length_min = rollouts
        .iter()
        .map(Rollout::completion_len)
        .min()
        .unwrap_or(0) as f32;
    let completion_length_max = rollouts
        .iter()
        .map(Rollout::completion_len)
        .max()
        .unwrap_or(0) as f32;
    let truncation_fraction = rollouts
        .iter()
        .filter(|rollout| is_truncated(rollout.completion_len(), loss_denominator))
        .count() as f32
        / rollouts.len() as f32;
    // Mean sampled-token entropy proxy (item 7): `-mean(log pi)` over live
    // rollouts, read straight off the behavior logprobs. This is the
    // entropy-collapse symptom Clip-Higher is meant to fight.
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
    // Standard GRPO regularizes against a fixed reference policy. The
    // runtime temporarily disables LoRA for these scores, so the anchor is
    // the immutable base model rather than the freshly sampled policy.
    // Only live rollouts are scored: dead ones never reach the optimizer.
    // Item 1: when the *effective* coefficient is zero the reference term
    // is identically zero, so this whole teacher-forced pass is skipped and
    // members carry an empty reference slice. Testing the effective value
    // rather than the base one also skips the pass during the KL warm-up,
    // where `warmup_factor` is still zero and the term cannot contribute.
    //
    // Scored group by group through the same shared-prefix batch the
    // behavior pass uses: the live members of a group share their prompt, so
    // this costs one prefix decode per group instead of one full prompt
    // prefill per rollout. Groups are contiguous and `trainable` is
    // ascending, so concatenating each group's live rows reproduces
    // `trainable` order exactly - which is what `reference_positions` below
    // indexes into.
    let reference_started = Instant::now();
    let reference_rows = if effective_kl != 0.0 {
        trainer.with_lora_disabled(|trainer| {
            let mut rows = Vec::with_capacity(trainable.len());
            for (group_index, group) in rollouts.chunks(config.group_size).enumerate() {
                let base = group_index * config.group_size;
                let members = group
                    .iter()
                    .enumerate()
                    .filter(|(member, _)| live[base + member])
                    .map(|(_, rollout)| rollout)
                    .collect::<Vec<_>>();
                if members.is_empty() {
                    continue;
                }
                rows.extend(score_train_mask_group(trainer, &members)?);
            }
            Ok(rows)
        })?
    } else {
        Vec::new()
    };
    let reference_seconds = reference_started.elapsed().as_secs_f32();
    let mut reference_positions = vec![None; rollouts.len()];
    for (position, &index) in trainable.iter().enumerate() {
        reference_positions[index] = Some(position);
    }
    state.stalled_updates = stalled_updates;
    Ok(UpdateBaseline {
        update,
        effective_kl,
        rollouts,
        group_ids,
        advantages,
        trainable,
        reference_rows,
        reference_positions,
        judge_tally,
        metrics: BatchMetrics {
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
            timing,
        },
    })
}

/// Phase 3 of an update - the GRPO epochs.
///
/// Stochastic minibatch-size-one steps. Live rollouts are re-scored under the
/// current policy before every step; dead slots advance only the scheduler,
/// preserving the configured LR timeline. Reference scores remain fixed across
/// every update and epoch.
///
/// Returns `false` when a progress callback asked to stop, which ends the run.
fn run_epochs(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    state: &mut RunState,
    baseline: &UpdateBaseline,
    scratch: &mut WeightedStepScratch,
    final_metrics: &mut TrainMetrics,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<bool> {
    // Destructured rather than read through `baseline.metrics.…`, and the field
    // names are the ones the update loop used before the split: the `metric!`
    // block below is forty series long and it is the *names* that pair each one
    // with its value. Renaming them here to save this pattern would move that
    // pairing off the page.
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
    // Historical aggregate retained for dashboards and existing exports.
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
                    // prefill per completion (docs/engineering/optims/SAMPLING.md H4).
                    metric!(
                        "scoring/prefix_decodes_per_group",
                        ratio_or_zero(scoring_stats.prefix_decodes, scoring_stats.calls)
                    ),
                    // Share of scored positions whose target log-probability
                    // was gathered on the device instead of reduced from a
                    // full n_vocab row on the host (SAMPLING.md H5).
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

fn next_kl_multiplier(current: f32, target: Option<f32>, measured_kl: f32) -> f32 {
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
