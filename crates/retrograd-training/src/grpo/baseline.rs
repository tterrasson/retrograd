//! Phase 2 of an update: reward stats and the group-relative baseline.

use super::*;

/// One update's batch, centered: the group-relative advantages, the live slots
/// and the frozen-reference rows the epochs read.
///
/// Phase 2 consumes the [`TrainingBatch`] rather than borrowing it, which is
/// what frees the per-slot text buffers - completions, seeds, prompt indices,
/// before the epochs allocate anything: they are needed only up to the
/// observer, which takes the completions.
pub(super) struct UpdateBaseline {
    pub(super) update: u32,
    /// KL coefficient in force for this update: base * warm-up * adaptive
    /// multiplier. Zero means the reference pass was skipped and
    /// `reference_rows` is empty.
    pub(super) effective_kl: f32,
    pub(super) rollouts: Vec<Rollout>,
    pub(super) group_ids: Vec<u64>,
    pub(super) advantages: Vec<f32>,
    /// Ascending indices of the slots that reach the optimizer.
    pub(super) trainable: Vec<usize>,
    pub(super) reference_rows: Vec<Vec<f32>>,
    /// Per-slot position into `reference_rows`; `None` marks a dead slot, so
    /// this doubles as the liveness mask the epochs read.
    pub(super) reference_positions: Vec<Option<usize>>,
    pub(super) judge_tally: JudgeTally,
    pub(super) metrics: BatchMetrics,
}

/// Phase 2 of an update - reward stats and the group-relative baseline.
///
/// Each reward is centered against its own group's mean. Reward std remains a
/// collapse diagnostic but is deliberately absent from the advantage. This is
/// also where the DAPO shaping happens (overlong penalty, truncation masking)
/// and where a run that has stalled for too long stops.
///
/// `observer` is only passed for an update it wants.
pub(super) fn build_baseline(
    trainer: &mut Trainer,
    config: &GrpoConfig,
    state: &mut RunState,
    update: u32,
    effective_kl: f32,
    batch: TrainingBatch,
    observer: Option<&dyn TrajectoryObserver>,
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
    let raw_rewards = observer.map(|_| rewards.clone());
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
    match (observer, &raw_rewards) {
        (Some(observer), Some(raw_rewards)) => {
            let rollouts = grpo_rollouts(GrpoSlots {
                update: update + 1,
                group_size: config.group_size,
                loss_denominator,
                mask_truncated: config.mask_truncated,
                prompt_indices: &prompt_indices,
                completions,
                lengths: rollouts.iter().map(Rollout::completion_len).collect(),
                seeds: &member_seeds,
                rewards: &rewards,
                raw_rewards,
                judge_terms: &judge_terms,
                judged: &judged_groups,
                advantages: &advantages,
                live: &live,
            });
            observer.observe(ObserveBatch::Rollouts(RolloutBatch {
                prompts: observed_prompts(&state.prompts, prompt_indices.iter().copied()),
                rollouts,
            }));
        }
        _ => drop(completions),
    }
    drop(prompt_indices);
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
