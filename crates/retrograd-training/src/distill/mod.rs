//! On-policy distillation: the student samples, the teacher notes the same
//! tokens, and the log-probability gap becomes a dense per-token advantage.
//!
//! ```text
//! A_t = clamp( log p_T(y_t | y_<t) - log pi_S^behav(y_t | y_<t), +-weight_clip )
//! ```
//!
//! `A_t` is detached and takes the place of GRPO's group-relative scalar in the
//! existing weighted objective, so no new kernel appears anywhere below this
//! module. With one optimizer epoch per batch the policy ratio is exactly 1 and
//! the token weight *is* `A_t`; beyond that the clipped surrogate of
//! `rollout::weights` applies unchanged.
//!
//! What this estimates: `A_t` is a single-sample estimator of the per-token
//! reverse-KL gradient, not the reverse KL itself - that would need the
//! teacher's full distribution, which teacher-forced scoring does not return.
//! The bias is the price of everything else already being in place.

pub mod evaluate;
pub mod offline;
pub mod teacher;

pub use evaluate::{DistillBench, benchmark, evaluate, top1_agreement};
pub use offline::OfflineBatch;
pub use teacher::{SharedTeacher, Teacher, WITNESS_SENTENCES};

use std::time::Instant;

use retrograd_config::DistillConfig;
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_engine::{ScoringStats, Trainer};
use retrograd_metrics::MetricValue;

use crate::grpo::{PromptDraws, optimizer_memory_metrics, ratio_or_zero, shuffled_indices};
use crate::rollout::{
    EpochBatch, EpochState, GrpoObjective, GrpoStepParams, Rollout, RowLayout, SchedulerHorizon,
    TokenStats, WeightedStepScratch, check_policy_divergence, is_truncated, read_prompts,
    run_grpo_epoch, sample_rollout_groups_continuous, score_train_mask_group, tokenize_prompts,
};
use crate::{Boundary, Progress};

macro_rules! metric {
    ($name:literal, $value:expr_2021) => {
        MetricValue {
            name: $name.into(),
            value: $value,
        }
    };
}

/// Per-token advantages for one span: the teacher's log-probability of the
/// student's own token, minus the behaviour log-probability the sampler
/// recorded for it, clamped to `+-weight_clip`.
///
/// The clamp is not cosmetic. On a token the student was confident about and
/// the teacher was not, the gap runs to -20 nats; global gradient-norm clipping
/// then rescales every other token to nearly nothing and a single token owns
/// the update. This is the reasoning behind `REFERENCE_LOG_RATIO_LIMIT` in
/// `rollout::weights`, applied to the same kind of quantity.
pub fn token_advantages(
    teacher_logprobs: &[f32],
    behaviour_logprobs: &[f32],
    weight_clip: f32,
) -> Result<Vec<f32>> {
    if teacher_logprobs.len() != behaviour_logprobs.len() {
        return Err(Error::invalid(format!(
            "the teacher scored {} tokens for {} sampled ones",
            teacher_logprobs.len(),
            behaviour_logprobs.len()
        )));
    }
    if !weight_clip.is_finite() || weight_clip <= 0.0 {
        return Err(Error::invalid(
            "distillation weight_clip must be finite and greater than zero",
        ));
    }
    teacher_logprobs
        .iter()
        .zip(behaviour_logprobs)
        .map(|(&teacher, &behaviour)| {
            if !teacher.is_finite() || !behaviour.is_finite() {
                return Err(Error::runtime(
                    "distillation scored a non-finite log-probability",
                ));
            }
            Ok((teacher - behaviour).clamp(-weight_clip, weight_clip))
        })
        .collect()
}

/// The value at `quantile` of `values`, which this consumes and sorts in place
/// rather than copying: the caller has just built it for this and the vector is
/// one entry per trained token of an update.
///
/// Zero on an empty input, which is the honest reading of "this update trained
/// nothing" on a series that is otherwise a divergence.
fn percentile(mut values: Vec<f32>, quantile: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    // Every entry is finite - `token_advantages` refuses a non-finite
    // log-probability before it can reach here - so a total order exists and
    // `total_cmp` is it.
    values.sort_unstable_by(f32::total_cmp);
    let index = ((values.len() as f64 - 1.0) * quantile.clamp(0.0, 1.0)).ceil() as usize;
    values[index.min(values.len() - 1)]
}

/// Everything a distillation run establishes before its first update, plus the
/// state that survives from one update to the next - the same split as
/// `grpo::RunState`, minus everything only a reward makes meaningful. No judge
/// (nothing is graded), no reward process (nothing is scored by a command), no
/// adaptive-KL multiplier (the teacher is the anchor), and no stall counter: a
/// group of one carries its signal, so there is no such thing as an update
/// without one.
struct RunState {
    /// First update index of this launch; zero unless resumed.
    start_update: u32,
    /// The teacher, loaded once for the run and shared with the scheduled
    /// evaluation. It never gets an adapter, so it costs its weights plus its KV
    /// cache for the whole run and nothing else - and there is only ever one of
    /// it, which is the point of the handle rather than the value.
    teacher: SharedTeacher,
    prompt_rows: Vec<Vec<i32>>,
    layout: RowLayout,
    loss_denominator: usize,
    rollouts_per_update: usize,
    horizon: SchedulerHorizon,
    /// Global cursor over prompt draws. Advances by exactly
    /// `prompts_per_update` per update - there is no resampling here, so the
    /// cursor and the update index cannot drift apart. It is still carried
    /// through the boundary rather than derived, because that is what makes a
    /// later filtering rule a change of one line instead of a change of format.
    prompt_cursor: usize,
    prompt_draws: PromptDraws,
    /// Lines produced by batch assembly, waiting for the next progress event to
    /// carry them out rather than tearing the caller's progress bar.
    pending_notes: Vec<String>,
}

/// Validates the configuration against the restored boundary, opens the teacher
/// and refuses an incompatible pair. Every failure here happens before any
/// rollout compute - and before the teacher has scored a single token.
fn resume_state(
    trainer: &Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    teacher: &SharedTeacher,
    resume: Option<Boundary>,
) -> Result<RunState> {
    config.validate()?;
    let start_update = resume.map_or(0, |boundary| boundary.completed_iterations);
    if start_update >= u64::from(config.updates) {
        return Err(Error::checkpoint(format!(
            "the checkpoint has {start_update} completed updates, at or past distill.updates ({})",
            config.updates
        )));
    }
    // Sound by the guard above and nothing else: `config.updates` is a u32, so
    // anything strictly below it fits. The comparison is done in u64 rather
    // than the narrowing done first, otherwise an absurd checkpoint would be
    // refused for the wrong reason.
    let start_update = start_update as u32;
    if (training.n_seq_max as usize) < config.samples_per_prompt {
        return Err(Error::invalid(format!(
            "training.n_seq_max ({}) must cover distill.samples_per_prompt ({})",
            training.n_seq_max, config.samples_per_prompt
        )));
    }
    {
        // The one check that cannot be deferred. Scoring the student's ids under
        // a different vocabulary returns the log-probabilities of *other
        // tokens*: the numbers are finite, the run trains, and the objective is
        // nonsense. The borrow is scoped so the loop below can take its own.
        let teacher = teacher.get_or_open(&config.teacher_path, training)?;
        teacher.compatibility(trainer)?;
    }
    let prompts = read_prompts(&config.prompts)?;
    let layout = RowLayout::resolve(trainer, training)?;
    let loss_denominator = config.sampling.max_new_tokens as usize;
    // A prompt with no room left for the generation budget is refused here,
    // once, rather than producing a zero-length completion every update it is
    // drawn. It is a property of the prompt file, so it belongs before the loop.
    let prompt_rows = tokenize_prompts(
        trainer,
        &prompts,
        &layout,
        loss_denominator,
        &config.prompts,
    )?;
    let rollouts_per_update = config
        .prompts_per_update
        .checked_mul(config.samples_per_prompt)
        .ok_or_else(|| Error::overflow("distillation rollout count overflows usize"))?;
    let total_steps = u64::from(config.updates)
        .checked_mul(u64::from(config.distill_epochs))
        .and_then(|value| value.checked_mul(rollouts_per_update as u64))
        .and_then(|value| value.checked_mul(layout.steps_per_row))
        .ok_or_else(|| Error::overflow("distillation optimizer step count overflows u64"))?;
    // An upper bound over every rollout slot, re-priced after each update on
    // the steps actually taken: masked completions advance the scheduler
    // without optimizing, and a decaying schedule would otherwise end the run
    // far above zero.
    let horizon = SchedulerHorizon::new(total_steps, training.warmup_steps);
    let prompt_draws = PromptDraws::new(config.prompt_order, prompts.len(), config.sampling.seed);
    Ok(RunState {
        start_update,
        teacher: teacher.clone(),
        prompt_rows,
        layout,
        loss_denominator,
        rollouts_per_update,
        horizon,
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
        pending_notes: Vec::new(),
    })
}

/// Runs on-policy distillation end to end: sample completions from the student,
/// score the very same tokens with the teacher, turn the log-probability gap
/// into a per-token advantage, and take `distill_epochs` weighted steps through
/// the runtime's existing differentiable objective.
pub fn run(
    trainer: &mut Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    run_resumed(
        trainer,
        config,
        training,
        &SharedTeacher::new(),
        None,
        &mut |_, progress| {
            if progress.metrics.epoch_complete {
                on_progress(progress);
            }
            Ok(true)
        },
    )
}

/// [`run`], restarting at a boundary restored from a checkpoint.
/// The update is the unit of resume and the cursor advances by exactly
/// `prompts_per_update` per update, so the two agree by construction; the
/// cursor is still what the boundary carries, for the reason `RunState` gives.
/// `kl_multiplier` is always `None`: the teacher is the anchor, and there is no
/// adaptive coefficient to carry across a restart.
pub fn run_resumed(
    trainer: &mut Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    teacher: &SharedTeacher,
    resume: Option<Boundary>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::distill", "training");
    let _entered = span.enter();
    let mut state = resume_state(trainer, config, training, teacher, resume)?;
    let mut final_metrics = TrainMetrics::default();
    let mut scratch = WeightedStepScratch::new(&state.layout);
    for update in state.start_update..config.updates {
        let update_start_step = final_metrics.global_step;
        let batch = assemble_batch(trainer, config, training, &mut state, update)?;
        if !run_epochs(
            trainer,
            config,
            &mut state,
            &batch,
            &mut scratch,
            &mut final_metrics,
            update,
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
    teacher_scoring: f32,
}

/// One update's assembled batch: `prompts_per_update * samples_per_prompt`
/// rollout slots in group order, with the per-token advantages the epochs read.
struct UpdateBatch {
    rollouts: Vec<Rollout>,
    group_ids: Vec<u64>,
    /// One entry per slot, `None` on a slot the epochs must skip. Every live
    /// entry is the clamped teacher-minus-behaviour gap of `token_advantages`.
    token_advantages: Vec<Option<Vec<f32>>>,
    /// Ascending indices of the slots that reach the optimizer.
    trainable: Vec<usize>,
    /// Per-slot position into the reference-row table; `None`
    /// marks a dead slot, so this doubles as the liveness mask the epochs read.
    reference_positions: Vec<Option<usize>>,
    metrics: BatchMetrics,
}

/// The scalar series one update publishes. Computed once, in phase 1, and
/// re-emitted by every epoch of the update.
#[derive(Clone, Copy, Debug, Default)]
struct BatchMetrics {
    advantage_abs_mean: f64,
    /// Mean of `-A_t` over every trained token, in nats. This is the quantity
    /// the run minimizes - a single-sample estimator of the per-token reverse KL
    /// to the teacher - and the only series here with an interpretable unit,
    /// which is why the loss alone makes a distillation run unreadable.
    teacher_kl_mean: f32,
    /// 95th percentile of the same per-token quantity. The mean moves slowly
    /// and hides the tail; a run whose mean falls while its p95 does not is
    /// aligning the easy tokens and leaving the disagreements alone.
    teacher_kl_p95: f32,
    /// Share of trained tokens whose raw gap reached `weight_clip`. If it does
    /// not fall, the bound is too tight or the student is diverging - and until
    /// it does fall, `teacher_kl_mean` is measuring a clipped quantity rather
    /// than the divergence.
    advantage_clipped_fraction: f32,
    completion_length_mean: f32,
    completion_length_min: f32,
    completion_length_max: f32,
    truncation_fraction: f32,
    trained_fraction: f32,
    entropy: f32,
    /// Completion tokens over the live slots - the numerator of
    /// `tokens_per_second`.
    trained_tokens: usize,
    scoring_stats: ScoringStats,
    timing: BatchTiming,
}

/// Phase 1 of an update - sample, score twice, and centre nothing.
///
/// It differs from `grpo::assemble_batch` in three ways.
/// There is no resampling loop: `group_has_signal` tests whether a group's
/// rewards can be told apart, and here there are no rewards - one sample
/// suffices and its signal is dense, so every group drawn is kept and the
/// cursor advances by exactly `prompts_per_update`. `max_stalled_updates` and
/// `dynamic_sampling` are absent for the same reason rather than defaulted to
/// something inert. What remains is `mask_truncated`.
fn assemble_batch(
    trainer: &mut Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    state: &mut RunState,
    update: u32,
) -> Result<UpdateBatch> {
    let layout = state.layout;
    let loss_denominator = state.loss_denominator;
    let mut prompt_inputs = Vec::with_capacity(config.prompts_per_update);
    for _ in 0..config.prompts_per_update {
        let prompt_offset = state.prompt_cursor;
        state.prompt_cursor += 1;
        let prompt_slot = state.prompt_draws.slot(prompt_offset);
        let mut seed_offsets = Vec::with_capacity(config.samples_per_prompt);
        for member in 0..config.samples_per_prompt {
            let seed_offset = prompt_offset
                .checked_mul(config.samples_per_prompt)
                .and_then(|value| value.checked_add(member))
                .ok_or_else(|| {
                    Error::overflow("distillation sampling seed offset overflows usize")
                })?;
            seed_offsets.push(seed_offset);
        }
        prompt_inputs.push((state.prompt_rows[prompt_slot].as_slice(), seed_offsets));
    }
    let generation_started = Instant::now();
    let sampled_groups = sample_rollout_groups_continuous(
        trainer,
        &prompt_inputs,
        &config.sampling,
        &layout,
        training.effective_generation_concurrency() as usize,
    )?;
    let mut rollouts: Vec<Rollout> = Vec::with_capacity(state.rollouts_per_update);
    for group in sampled_groups {
        // The completion text is dropped here rather than carried: there is no
        // reward command to show it to and no completion journal in this loop,
        // so keeping it would hold `rollouts_per_update` strings alive across
        // every epoch for nothing.
        rollouts.extend(group.into_iter().map(|(rollout, _)| rollout));
    }
    let generation_seconds = generation_started.elapsed().as_secs_f32();

    // Behaviour log-probabilities of the student, teacher-forced under the
    // policy that sampled them. `pi_old` in the surrogate, and the subtrahend
    // of the advantage.
    let behaviour_started = Instant::now();
    let scoring_stats_before = trainer.scoring_stats()?;
    for group in rollouts.chunks_mut(config.samples_per_prompt) {
        let scores = {
            let members = group.iter().collect::<Vec<_>>();
            score_train_mask_group(trainer, &members)?
        };
        for (rollout, scores) in group.iter_mut().zip(scores) {
            rollout.old_logprobs = scores;
        }
    }
    let behavior_scoring_seconds = behaviour_started.elapsed().as_secs_f32();
    let scoring_stats = trainer.scoring_stats()?.delta_since(scoring_stats_before);

    // The teacher's log-probabilities of the *same* tokens, through the same
    // shared-prefix batched call the student's scoring uses. Same call, because
    // `llama_decode` is not invariant by batch: scoring one side sequence by
    // sequence and the other in a group would compare two tile arrangements
    // rather than two models.
    let teacher_started = Instant::now();
    let mut teacher_rows: Vec<Vec<f32>> = Vec::with_capacity(rollouts.len());
    {
        let mut teacher = state.teacher.get_or_open(&config.teacher_path, training)?;
        for group in rollouts.chunks(config.samples_per_prompt) {
            let members = group.iter().collect::<Vec<_>>();
            teacher_rows.extend(teacher.logprobs_group(&members)?);
        }
    }
    let teacher_scoring_seconds = teacher_started.elapsed().as_secs_f32();

    // The truncation filter, and the only one left. A completion cut at the
    // generation budget was never finished, so the gap on its last tokens
    // measures the teacher's opinion of an interrupted sentence.
    let mut live = vec![true; rollouts.len()];
    if config.mask_truncated {
        for (flag, rollout) in live.iter_mut().zip(&rollouts) {
            *flag = !is_truncated(rollout.completion_len(), loss_denominator);
        }
    }
    let group_ids = (0..config.prompts_per_update)
        .flat_map(|group| std::iter::repeat_n(group as u64, config.samples_per_prompt))
        .collect::<Vec<_>>();
    // The advantage itself: a dense, per-token, detached quantity. Computed for
    // the live slots only - a dead slot's vector is never read, and computing
    // it would only invite a non-finite log-probability from a completion the
    // update already decided to skip.
    let mut advantage_rows: Vec<Option<Vec<f32>>> = Vec::with_capacity(rollouts.len());
    for (index, rollout) in rollouts.iter().enumerate() {
        if !live[index] {
            advantage_rows.push(None);
            continue;
        }
        advantage_rows.push(Some(token_advantages(
            &teacher_rows[index],
            &rollout.old_logprobs,
            config.weight_clip,
        )?));
    }

    let trainable: Vec<usize> = (0..rollouts.len()).filter(|&index| live[index]).collect();
    // Legitimate, and it has exactly one cause here: `mask_truncated` is on and
    // every completion of the update ran to the budget. That is a statement
    // about the generation budget or about a policy drifting longer, not about
    // a signal that failed to appear - so it is announced and the run
    // continues, where GRPO counts stalls toward a stop.
    if trainable.is_empty() {
        state.pending_notes.push(format!(
            "update {}: every completion ran to distill.sampling.max_new_tokens ({}) and \
             distill.mask_truncated dropped all of them - raise the budget, or turn the \
             masking off to train on truncated completions.",
            update + 1,
            loss_denominator
        ));
    }
    let mut reference_positions = vec![None; rollouts.len()];
    for (position, &index) in trainable.iter().enumerate() {
        reference_positions[index] = Some(position);
    }

    // The three divergence series, read off the advantages of the live slots
    // only: a dead slot has no `A_t` to fold in, and counting it as a zero
    // divergence would report agreement with the teacher on a completion nobody
    // scored.
    let divergences = trainable
        .iter()
        .filter_map(|&index| advantage_rows[index].as_ref())
        .flatten()
        // `-A_t`, so the series reads as a divergence: positive where the
        // student is more confident than the teacher.
        .map(|&value| -value)
        .collect::<Vec<f32>>();
    let clipped = divergences
        .iter()
        .filter(|value| value.abs() >= config.weight_clip)
        .count();
    let advantage_clipped_fraction = ratio_or_zero(clipped as u64, divergences.len() as u64);
    let teacher_kl_mean = if divergences.is_empty() {
        0.0
    } else {
        (divergences.iter().map(|&value| value as f64).sum::<f64>() / divergences.len() as f64)
            as f32
    };
    let teacher_kl_p95 = percentile(divergences, 0.95);

    let trained_fraction = trainable.len() as f32 / rollouts.len() as f32;
    let advantage_abs_mean = if trainable.is_empty() {
        0.0
    } else {
        let sum: f64 = trainable
            .iter()
            .filter_map(|&index| advantage_rows[index].as_ref())
            .map(|values| {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().map(|&value| value.abs() as f64).sum::<f64>()
                        / values.len() as f64
                }
            })
            .sum();
        sum / trainable.len() as f64
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
    // `-mean(log pi)` over the live rollouts, read straight off the behaviour
    // logprobs: the entropy-collapse symptom, same definition as GRPO's.
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

    Ok(UpdateBatch {
        rollouts,
        group_ids,
        token_advantages: advantage_rows,
        trainable,
        reference_positions,
        metrics: BatchMetrics {
            advantage_abs_mean,
            teacher_kl_mean,
            teacher_kl_p95,
            advantage_clipped_fraction,
            completion_length_mean,
            completion_length_min,
            completion_length_max,
            truncation_fraction,
            trained_fraction,
            entropy,
            trained_tokens,
            scoring_stats,
            timing: BatchTiming {
                generation: generation_seconds,
                behavior_scoring: behavior_scoring_seconds,
                teacher_scoring: teacher_scoring_seconds,
            },
        },
    })
}

/// Phase 2 of an update - the optimizer epochs, `run_grpo_epoch` unchanged.
/// Returns `false` when a progress callback asked to stop, which ends the run.
#[expect(clippy::too_many_arguments)]
fn run_epochs(
    trainer: &mut Trainer,
    config: &DistillConfig,
    state: &mut RunState,
    batch: &UpdateBatch,
    scratch: &mut WeightedStepScratch,
    final_metrics: &mut TrainMetrics,
    update: u32,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<bool> {
    let UpdateBatch {
        ref rollouts,
        ref group_ids,
        ref token_advantages,
        ref trainable,
        ref reference_positions,
        metrics:
            BatchMetrics {
                advantage_abs_mean,
                teacher_kl_mean,
                teacher_kl_p95,
                advantage_clipped_fraction,
                completion_length_mean,
                completion_length_min,
                completion_length_max,
                truncation_fraction,
                trained_fraction,
                entropy,
                trained_tokens,
                scoring_stats,
                timing:
                    BatchTiming {
                        generation: generation_seconds,
                        behavior_scoring: behavior_scoring_seconds,
                        teacher_scoring: teacher_scoring_seconds,
                    },
            },
    } = *batch;
    let layout = state.layout;
    let step_params = GrpoStepParams {
        objective: GrpoObjective {
            clip_range_low: config.clip_range_low,
            clip_range_high: config.clip_range_high,
            kl_coefficient: config.kl_coefficient,
            loss_denominator: state.loss_denominator,
        },
        scheduler_total_steps: state.horizon.steps(),
    };
    let rollout_evals = rollouts
        .iter()
        .map(|rollout| layout.rollout_evals(rollout))
        .collect::<Result<Vec<_>>>()?;
    // Filled with zeros and never read: `weights.rs` takes the per-token
    // advantage as soon as one exists, and one always does here. The zeros are
    // written explicitly rather than left to a default so that a future reader
    // finding a scalar advantage in a distillation batch knows it is a bug and
    // not a value someone chose.
    let scalar_advantages = vec![0.0_f32; rollouts.len()];
    // The optional KL term is separate from the teacher-derived advantages.
    // Score live rows once, before any optimizer epoch changes the policy.
    let reference_rows = if config.kl_coefficient != 0.0 {
        trainer.with_reference_policy(|reference| {
            let mut rows = Vec::with_capacity(trainable.len());
            for (group_index, group) in rollouts.chunks(config.samples_per_prompt).enumerate() {
                let base = group_index * config.samples_per_prompt;
                let members = group
                    .iter()
                    .enumerate()
                    .filter(|(member, _)| reference_positions[base + member].is_some())
                    .map(|(_, rollout)| rollout)
                    .collect::<Vec<_>>();
                rows.extend(score_train_mask_group(reference, &members)?);
            }
            Ok(rows)
        })?
    } else {
        Vec::new()
    };
    let epoch_batch = EpochBatch {
        rollouts,
        group_ids,
        advantages: &scalar_advantages,
        token_advantages,
        reference_positions,
        reference_rows: &reference_rows,
        rollout_evals: &rollout_evals,
    };
    let mut optimizer_step_taken = false;
    for epoch in 0..config.distill_epochs {
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
        let total_loss = surrogate_loss + config.kl_coefficient * kl;
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
        let closes_update = epoch + 1 == config.distill_epochs;
        let notes = state
            .pending_notes
            .drain(..)
            .chain(scratch.notes.drain(..))
            .collect();
        if !on_progress(
            trainer,
            Progress {
                notes,
                // Only the last epoch of an update closes a resumable boundary.
                boundary: closes_update.then(|| Boundary {
                    completed_iterations: update as u64 + 1,
                    cursor: state.prompt_cursor as u64,
                    kl_multiplier: None,
                }),
                metrics: *final_metrics,
                values: vec![
                    metric!("batch/advantage_abs_mean", advantage_abs_mean as f32),
                    // The objective, in nats per token. Everything else on this
                    // list describes how the update ran; these three say
                    // whether it is working.
                    metric!("distill/teacher_kl_mean", teacher_kl_mean),
                    metric!("distill/teacher_kl_p95", teacher_kl_p95),
                    metric!(
                        "distill/advantage_clipped_fraction",
                        advantage_clipped_fraction
                    ),
                    metric!("batch/trained_fraction", trained_fraction),
                    metric!("completions/length_mean", completion_length_mean),
                    metric!("completions/length_min", completion_length_min),
                    metric!("completions/length_max", completion_length_max),
                    metric!("completions/truncation_fraction", truncation_fraction),
                    metric!("policy/surrogate_loss", surrogate_loss),
                    metric!("policy/kl", kl),
                    metric!("policy/clip_fraction", clip_fraction),
                    metric!("policy/total_loss", total_loss),
                    metric!("policy/entropy", entropy),
                    metric!("policy/ratio_mean", ratio_mean),
                    metric!("policy/ratio_max", ratio_max),
                    metric!("policy/kl_coefficient", config.kl_coefficient),
                    metric!(
                        "timing/sampling_seconds",
                        generation_seconds + behavior_scoring_seconds
                    ),
                    metric!("timing/generation_seconds", generation_seconds),
                    metric!("timing/behavior_scoring_seconds", behavior_scoring_seconds),
                    // The term this loop has and GRPO does not: a second
                    // forward model over every sampled token. It is what a run
                    // trades a reward command for, so it is published beside
                    // the other phases rather than folded into one of them.
                    metric!("timing/teacher_scoring_seconds", teacher_scoring_seconds),
                    metric!("timing/optimizer_seconds", optimizer_seconds),
                    metric!(
                        "scoring/prefix_decodes_per_group",
                        ratio_or_zero(scoring_stats.prefix_decodes, scoring_stats.calls)
                    ),
                    metric!(
                        "scoring/device_logprob_fraction",
                        ratio_or_zero(
                            scoring_stats.device_logprob_positions,
                            scoring_stats.scored_positions
                        )
                    ),
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
                .collect(),
            },
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identical_teacher_leaves_every_advantage_at_zero() {
        let logprobs = [-0.5, -2.25, -7.0];
        let advantages = token_advantages(&logprobs, &logprobs, 5.0).expect("advantages");
        assert_eq!(advantages, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_gap_is_signed_and_clamped_on_both_sides() {
        let teacher = [-1.0, -0.5, -30.0];
        let behaviour = [-3.0, -0.25, -1.0];
        let advantages = token_advantages(&teacher, &behaviour, 1.5).expect("advantages");
        assert_eq!(advantages, vec![1.5, -0.25, -1.5]);
    }

    #[test]
    fn a_length_mismatch_is_a_user_error_rather_than_a_silent_truncation() {
        let error = token_advantages(&[-1.0, -2.0], &[-1.0], 5.0).expect_err("mismatch");
        assert!(error.is_user_error(), "{error}");
    }

    #[test]
    fn an_unusable_clip_is_refused() {
        assert!(token_advantages(&[-1.0], &[-1.0], 0.0).is_err());
        assert!(token_advantages(&[-1.0], &[-1.0], f32::NAN).is_err());
        assert!(token_advantages(&[-1.0], &[-1.0], f32::INFINITY).is_err());
    }
}
