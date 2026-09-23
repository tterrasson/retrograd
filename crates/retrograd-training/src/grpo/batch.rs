//! Phase 1 of an update: assembling the training batch.

use super::*;

/// Wall-clock split of one update's batch assembly, in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BatchTiming {
    pub(super) generation: f32,
    pub(super) behavior_scoring: f32,
    pub(super) reward: f32,
    pub(super) judge: f32,
}

/// One update's assembled batch: `prompts_per_update * group_size` rollout
/// slots in group order, one entry per slot in every vector.
pub(super) struct TrainingBatch {
    pub(super) rollouts: Vec<Rollout>,
    pub(super) prompt_indices: Vec<usize>,
    pub(super) completions: Vec<String>,
    pub(super) member_seeds: Vec<u32>,
    pub(super) rewards: Vec<f32>,
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
    pub(super) judge_terms: Vec<f32>,
    /// Whether each rollout's reward carries the judge term the others carry.
    /// All true without a judge, and all true with one until it fails a group:
    /// a reward missing that term is not on the batch's scale, so the members
    /// it belongs to stay out of the baseline and the epochs.
    pub(super) judged_groups: Vec<bool>,
    /// Candidate groups drawn to fill the batch - exactly `prompts_per_update`
    /// without dynamic sampling, more when zero-signal groups were resampled.
    pub(super) groups_sampled: usize,
    pub(super) judge_tally: JudgeTally,
    /// Behavior-scorer counters attributed to this update alone.
    pub(super) scoring_stats: ScoringStats,
    pub(super) timing: BatchTiming,
}

/// The scalar series one update publishes. Computed once, in phase 2, and
/// re-emitted by every epoch of the update.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BatchMetrics {
    pub(super) mean_reward: f32,
    /// `mean_reward` split in two: the reward command's own verdict, and what
    /// the judge added on top. They sum back to `mean_reward`.
    pub(super) verifiable_reward_mean: f32,
    pub(super) judge_reward_mean: f32,
    pub(super) reward_std: f32,
    pub(super) reward_min: f32,
    pub(super) reward_max: f32,
    pub(super) group_diagnostics: GroupDiagnostics,
    pub(super) advantage_abs_mean: f64,
    pub(super) completion_length_mean: f32,
    pub(super) completion_length_min: f32,
    pub(super) completion_length_max: f32,
    pub(super) truncation_fraction: f32,
    pub(super) trained_fraction: f32,
    pub(super) groups_sampled_fraction: f32,
    pub(super) distinct_fraction: f32,
    pub(super) entropy: f32,
    /// Completion tokens over the live slots - the numerator of
    /// `tokens_per_second`.
    pub(super) trained_tokens: usize,
    pub(super) scoring_stats: ScoringStats,
    pub(super) reference_seconds: f32,
    pub(super) timing: BatchTiming,
}

/// Phase 1 of an update - assemble the training batch.
///
/// Sample `group_size` completions per prompt - each member with its own
/// derived seed - and score them. With dynamic sampling, groups
/// without reward spread are discarded and replaced by the next prompts in the
/// round-robin, up to `max_resample_factor * prompts_per_update` candidates, so
/// the trained batch stays full of informative groups as the policy converges.
/// Without it, exactly `prompts_per_update` groups are sampled and all kept.
///
/// Advances `state.prompt_cursor` past every candidate drawn, informative or
/// not: that cursor is what a boundary carries, and resampling is part of the
/// deterministic draw sequence.
pub(super) fn assemble_batch(
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
            training.effective_generation_concurrency() as usize,
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
                // Truncating on purpose: the offset only feeds a seed, and a
                // run past 2^32 draws should repeat a seed, not stop. The
                // `checked_mul` above guards the index arithmetic.
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
            // groups are marked dead later by `group_advantages`. With it,
            // only informative groups are kept and the rest
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
    // the current policy. No optimizer step has happened yet, so the order the
    // groups are scored in does not matter.
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
