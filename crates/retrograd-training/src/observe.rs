//! What PPO and GRPO hand to a [`TrajectoryObserver`](retrograd_observe::TrajectoryObserver).
//!
//! Pure functions over the slices the loops already hold, so the loops only
//! decide *whether* to export and *when* to send.

use std::collections::BTreeSet;

use retrograd_observe::{
    ObservedPrompt, ObservedRollout, OutcomeEntry, RolloutContent, SkipReason,
};

use super::rollout::is_truncated;
use super::rollout::prompts::Prompt;

pub(crate) fn prompt_key(index: usize) -> String {
    format!("p:{index}")
}

/// One `prompt` record per distinct index, in first-use order.
pub(crate) fn observed_prompts(
    prompts: &[Prompt],
    indices: impl IntoIterator<Item = usize>,
) -> Vec<ObservedPrompt> {
    let mut seen = BTreeSet::new();
    indices
        .into_iter()
        .filter(|index| seen.insert(*index))
        .map(|index| prompts[index].observed(prompt_key(index)))
        .collect()
}

/// A GRPO update's slots, in group order, at the point its advantages exist.
pub(crate) struct GrpoSlots<'a> {
    /// One-based.
    pub update: u32,
    pub group_size: usize,
    pub loss_denominator: usize,
    pub mask_truncated: bool,
    pub prompt_indices: &'a [usize],
    pub completions: Vec<String>,
    pub lengths: Vec<usize>,
    pub seeds: &'a [u32],
    /// After the overlong penalty.
    pub rewards: &'a [f32],
    /// Before it.
    pub raw_rewards: &'a [f32],
    pub judge_terms: &'a [f32],
    pub judged: &'a [bool],
    pub advantages: &'a [f32],
    /// The effective mask the epochs use.
    pub live: &'a [bool],
}

/// The individual cause first: a truncated member of a dropped group is
/// reported as truncated.
fn grpo_skip_reason(live: bool, masked_truncation: bool, judged: bool) -> Option<SkipReason> {
    match (live, masked_truncation, judged) {
        (true, _, _) => None,
        (false, true, _) => Some(SkipReason::Truncated),
        (false, false, false) => Some(SkipReason::JudgeDropped),
        (false, false, true) => Some(SkipReason::ZeroSignal),
    }
}

pub(crate) fn grpo_rollouts(slots: GrpoSlots<'_>) -> Vec<ObservedRollout> {
    slots
        .completions
        .into_iter()
        .enumerate()
        .map(|(index, completion)| {
            let length = slots.lengths[index];
            let truncated = is_truncated(length, slots.loss_denominator);
            let live = slots.live[index];
            ObservedRollout {
                update: slots.update,
                group: Some(index / slots.group_size),
                member: index % slots.group_size,
                prompt: prompt_key(slots.prompt_indices[index]),
                seed: u64::from(slots.seeds[index]),
                tokens: length,
                truncated,
                reward: Some(slots.rewards[index]),
                reward_raw: Some(slots.raw_rewards[index]),
                judge_term: Some(slots.judge_terms[index]),
                advantage: Some(slots.advantages[index]),
                advantage_min: None,
                advantage_max: None,
                eligible: Some(live),
                trained: (!live).then_some(false),
                skip_reason: grpo_skip_reason(
                    live,
                    slots.mask_truncated && truncated,
                    slots.judged[index],
                ),
                content: RolloutContent::Completion { completion },
            }
        })
        .collect()
}

/// A PPO update's rollouts. PPO has no groups and trains every rollout.
pub(crate) struct PpoSlots<'a> {
    /// One-based.
    pub update: u32,
    /// Draw offset of the first rollout, which picks its prompt and seed.
    pub first_offset: usize,
    pub prompt_count: usize,
    pub sampling_seed: u32,
    pub max_new_tokens: usize,
    pub completions: Vec<String>,
    pub lengths: Vec<usize>,
    pub rewards: &'a [f32],
    /// Per token, after whitening.
    pub advantages: &'a [Vec<f32>],
    pub critic: bool,
}

pub(crate) fn ppo_rollouts(slots: PpoSlots<'_>) -> Vec<ObservedRollout> {
    slots
        .completions
        .into_iter()
        .enumerate()
        .map(|(index, completion)| {
            let offset = slots.first_offset + index;
            let tokens = &slots.advantages[index];
            let (sum, low, high) = tokens.iter().fold(
                (0.0_f64, f32::INFINITY, f32::NEG_INFINITY),
                |(sum, low, high), &value| {
                    (sum + f64::from(value), low.min(value), high.max(value))
                },
            );
            let spread = slots.critic && !tokens.is_empty();
            ObservedRollout {
                update: slots.update,
                group: None,
                member: index,
                prompt: prompt_key(offset % slots.prompt_count),
                // The sampler's own derivation: a seed is a bit pattern, and the
                // offset wraps into it on purpose.
                seed: u64::from(slots.sampling_seed.wrapping_add(offset as u32)),
                tokens: slots.lengths[index],
                truncated: is_truncated(slots.lengths[index], slots.max_new_tokens),
                reward: Some(slots.rewards[index]),
                reward_raw: Some(slots.rewards[index]),
                judge_term: None,
                advantage: (!tokens.is_empty()).then(|| (sum / tokens.len() as f64) as f32),
                advantage_min: spread.then_some(low),
                advantage_max: spread.then_some(high),
                eligible: Some(true),
                trained: None,
                skip_reason: None,
                content: RolloutContent::Completion { completion },
            }
        })
        .collect()
}

/// `outcome` entries for a finished update: `group_size` is `None` for PPO.
pub(crate) fn outcome(group_size: Option<usize>, trained: &[bool]) -> Vec<OutcomeEntry> {
    trained
        .iter()
        .enumerate()
        .map(|(index, &trained)| OutcomeEntry {
            group: group_size.map(|size| index / size),
            member: group_size.map_or(index, |size| index % size),
            trained,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpo_slots_keep_both_rewards_and_the_cause_of_each_exclusion() {
        let rollouts = grpo_rollouts(GrpoSlots {
            update: 3,
            group_size: 2,
            loss_denominator: 4,
            mask_truncated: true,
            prompt_indices: &[5, 5, 9, 9],
            completions: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            lengths: vec![4, 2, 1, 1],
            seeds: &[10, 11, 12, 13],
            rewards: &[0.5, 1.0, 0.0, 0.0],
            raw_rewards: &[1.0, 1.0, 0.0, 0.0],
            judge_terms: &[0.0, 0.0, 0.0, 0.0],
            judged: &[true, true, false, false],
            advantages: &[0.0, 0.0, 0.0, 0.0],
            live: &[false, false, false, false],
        });
        assert_eq!(rollouts.len(), 4);
        assert_eq!(rollouts[0].reward, Some(0.5));
        assert_eq!(rollouts[0].reward_raw, Some(1.0));
        assert!(rollouts[0].truncated);
        let reasons = rollouts
            .iter()
            .map(|rollout| rollout.skip_reason)
            .collect::<Vec<_>>();
        assert_eq!(
            reasons,
            [
                Some(SkipReason::Truncated),
                Some(SkipReason::ZeroSignal),
                Some(SkipReason::JudgeDropped),
                Some(SkipReason::JudgeDropped),
            ]
        );
        assert!(
            rollouts
                .iter()
                .all(|rollout| rollout.trained == Some(false))
        );
        assert_eq!((rollouts[3].group, rollouts[3].member), (Some(1), 1));
        assert_eq!(rollouts[3].prompt, "p:9");
        assert_eq!(rollouts[3].seed, 13);
        assert_eq!(rollouts[3].update, 3);
    }

    #[test]
    fn a_live_grpo_slot_waits_for_its_outcome() {
        let rollouts = grpo_rollouts(GrpoSlots {
            update: 1,
            group_size: 1,
            loss_denominator: 8,
            mask_truncated: false,
            prompt_indices: &[0],
            completions: vec!["a".into()],
            lengths: vec![8],
            seeds: &[0],
            rewards: &[1.0],
            raw_rewards: &[1.0],
            judge_terms: &[0.25],
            judged: &[true],
            advantages: &[0.5],
            live: &[true],
        });
        let rollout = &rollouts[0];
        assert_eq!(rollout.eligible, Some(true));
        assert_eq!(rollout.trained, None);
        assert_eq!(rollout.skip_reason, None);
        // Truncated but not masked: it trains.
        assert!(rollout.truncated);
        assert_eq!(rollout.judge_term, Some(0.25));
    }

    #[test]
    fn ppo_advantages_are_summarized_per_rollout() {
        let rollouts = ppo_rollouts(PpoSlots {
            update: 2,
            first_offset: 5,
            prompt_count: 3,
            sampling_seed: u32::MAX,
            max_new_tokens: 2,
            completions: vec!["a".into(), "b".into()],
            lengths: vec![2, 1],
            rewards: &[1.0, -1.0],
            advantages: &[vec![1.0, 3.0], vec![-1.0]],
            critic: true,
        });
        assert_eq!(rollouts[0].advantage, Some(2.0));
        assert_eq!(rollouts[0].advantage_min, Some(1.0));
        assert_eq!(rollouts[0].advantage_max, Some(3.0));
        assert!(rollouts[0].truncated);
        assert_eq!(rollouts[0].prompt, "p:2");
        assert_eq!(rollouts[1].prompt, "p:0");
        assert_eq!(rollouts[0].seed, 4, "the seed wraps like the sampler's");
        assert_eq!(rollouts[1].group, None);
        assert_eq!(rollouts[1].member, 1);
        assert_eq!(rollouts[1].judge_term, None);

        let without_critic = ppo_rollouts(PpoSlots {
            update: 2,
            first_offset: 0,
            prompt_count: 1,
            sampling_seed: 0,
            max_new_tokens: 4,
            completions: vec!["a".into()],
            lengths: vec![1],
            rewards: &[0.0],
            advantages: &[vec![0.5]],
            critic: false,
        });
        assert_eq!(without_critic[0].advantage_min, None);
    }

    #[test]
    fn outcome_entries_follow_the_grouping() {
        let grouped = outcome(Some(2), &[true, false, true]);
        assert_eq!(
            grouped[2],
            OutcomeEntry {
                group: Some(1),
                member: 0,
                trained: true
            }
        );
        let flat = outcome(None, &[true, true]);
        assert_eq!(flat[1].group, None);
        assert_eq!(flat[1].member, 1);
    }
}
