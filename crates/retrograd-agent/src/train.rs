//! Trajectories → `TrainSequence`.
//!
//! This is deliberately the *only* place where a trajectory meets the training
//! crate. Keeping it here rather than as a method on `Trajectory` is what lets
//! `retrograd-agent-core` - and therefore every environment, container backend,
//! tool and judge above it - compile without the engine and its C++ bridge.

use retrograd_training::batch::TrainSequence;

use crate::trajectory::{Trajectory, TrajectoryGroup};
use crate::{Error, Result};
use retrograd_agent_core::scenario::TruncationPolicy;

/// `truncation` is the caller stating what it means by handing over a partial
/// trajectory: under [`TruncationPolicy::Drop`] a truncated member reaching this
/// point is a bug - its reward describes a response the judge never saw the end
/// of - so it is refused rather than trained on.
pub fn to_train_sequence(
    trajectory: &Trajectory,
    group_id: u64,
    truncation: TruncationPolicy,
) -> Result<TrainSequence> {
    trajectory.validate()?;
    if trajectory.truncated && truncation == TruncationPolicy::Drop {
        return Err(Error::invalid(
            "truncated trajectories must be filtered before training, unless the truncation \
             policy deliberately keeps them",
        ));
    }
    let final_reward = trajectory
        .reward
        .ok_or_else(|| Error::invalid("trajectory has not been rewarded"))?;
    let reward = final_reward + trajectory.step_reward_sum();
    if !reward.is_finite() {
        return Err(Error::invalid(
            "trajectory total reward exceeds the finite f32 range",
        ));
    }
    let has_intermediate = trajectory.steps.iter().any(|step| step.reward.is_some());
    let intermediate_returns = if has_intermediate {
        trajectory
            .train_mask
            .iter()
            .enumerate()
            .filter(|(_, train)| **train)
            .map(|(position, _)| {
                trajectory
                    .steps
                    .iter()
                    .filter(|step| step.token_range.0 >= position)
                    .filter_map(|step| step.reward)
                    .sum::<f32>()
            })
            .collect()
    } else {
        Vec::new()
    };
    Ok(TrainSequence {
        tokens: trajectory.tokens.clone(),
        old_logprobs: trajectory.old_logprobs.clone(),
        train_mask: trajectory.train_mask.clone(),
        reward,
        group_id,
        intermediate_returns,
    })
}

pub fn to_train_sequences(
    group: &TrajectoryGroup,
    truncation: TruncationPolicy,
) -> Result<Vec<TrainSequence>> {
    group.validate()?;
    group
        .trajectories
        .iter()
        .map(|trajectory| to_train_sequence(trajectory, group.group_id, truncation))
        .collect()
}

pub fn groups_to_train_sequences(
    groups: &[TrajectoryGroup],
    truncation: TruncationPolicy,
) -> Result<Vec<TrainSequence>> {
    groups
        .iter()
        .map(|group| to_train_sequences(group, truncation))
        .collect::<Result<Vec<_>>>()
        .map(|groups| groups.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Message, Role, Step, StepKind};

    fn graded_trajectory() -> Trajectory {
        Trajectory {
            scenario_id: "s1".into(),
            messages: vec![Message::text(Role::Tool, "observation")],
            tokens: vec![1, 2, 3, 4, 5],
            old_logprobs: vec![-0.1, -0.2],
            train_mask: vec![false, false, true, false, true],
            steps: vec![
                Step {
                    kind: StepKind::Context,
                    token_range: (0, 2),
                    reward: None,
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (2, 3),
                    reward: None,
                },
                Step {
                    kind: StepKind::ToolResult,
                    token_range: (3, 4),
                    reward: Some(2.0),
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (4, 5),
                    reward: None,
                },
            ],
            reward: Some(1.0),
            truncated: false,
            metadata: Default::default(),
            provenance: None,
        }
    }

    #[test]
    fn step_rewards_become_intermediate_returns_and_join_the_total() {
        let sequence = to_train_sequence(&graded_trajectory(), 9, TruncationPolicy::Drop).unwrap();
        assert_eq!(sequence.group_id, 9);
        assert_eq!(sequence.reward, 3.0);
        assert_eq!(sequence.intermediate_returns, [2.0, 0.0]);
    }

    #[test]
    fn a_truncated_trajectory_needs_the_policy_to_say_so() {
        let mut truncated = graded_trajectory();
        truncated.truncated = true;
        assert!(to_train_sequence(&truncated, 8, TruncationPolicy::Drop).is_err());
        // …unless the caller means it: under MinReward the partial response is
        // kept on purpose, carrying the reward the agentic loop gave it.
        assert_eq!(
            to_train_sequence(&truncated, 8, TruncationPolicy::MinReward)
                .unwrap()
                .reward,
            3.0
        );
    }

    #[test]
    fn an_unrewarded_or_mismatched_group_never_reaches_the_optimizer() {
        let mut unrewarded = graded_trajectory();
        unrewarded.reward = None;
        assert!(to_train_sequence(&unrewarded, 1, TruncationPolicy::Drop).is_err());

        let mut group = TrajectoryGroup {
            group_id: 1,
            scenario_id: "s1".into(),
            trajectories: vec![graded_trajectory(), graded_trajectory()],
        };
        assert_eq!(
            to_train_sequences(&group, TruncationPolicy::Drop)
                .unwrap()
                .len(),
            2
        );
        group.trajectories[1].scenario_id = "other".into();
        assert!(to_train_sequences(&group, TruncationPolicy::Drop).is_err());
    }
}
