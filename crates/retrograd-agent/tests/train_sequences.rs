//! The seam between a trajectory and the training crate, from outside.
//!
//! `to_train_sequence` is deliberately the only place where the two meet
//! (`src/train.rs`), which makes it the one function of `retrograd-agent` whose
//! contract a `tests/` directory should hold.
//! No model, no environment, no tool: a trajectory in,
//! a `TrainSequence` out.

use retrograd_agent::TruncationPolicy;
use retrograd_agent::trajectory::{Message, Role, Step, StepKind, Trajectory, TrajectoryGroup};
use retrograd_agent::{groups_to_train_sequences, to_train_sequence};

/// Five tokens: one of context, then two policy actions of two tokens each.
fn trajectory(reward: Option<f32>, step_rewards: [Option<f32>; 2]) -> Trajectory {
    Trajectory {
        scenario_id: "s1".into(),
        messages: vec![Message::text(Role::User, "prompt")],
        tokens: vec![1, 2, 3, 4, 5],
        old_logprobs: vec![-0.1, -0.2, -0.3, -0.4],
        train_mask: vec![false, true, true, true, true],
        steps: vec![
            Step {
                kind: StepKind::Context,
                token_range: (0, 1),
                reward: None,
            },
            Step {
                kind: StepKind::PolicyAction,
                token_range: (1, 3),
                reward: step_rewards[0],
            },
            Step {
                kind: StepKind::PolicyAction,
                token_range: (3, 5),
                reward: step_rewards[1],
            },
        ],
        reward,
        truncated: false,
        metadata: serde_json::Map::new(),
    }
}

#[test]
fn the_sequence_carries_the_final_reward_plus_every_step_reward() {
    let sequence = to_train_sequence(
        &trajectory(Some(1.0), [Some(0.25), Some(0.5)]),
        7,
        TruncationPolicy::Drop,
    )
    .expect("a rewarded trajectory");

    assert_eq!(sequence.reward, 1.75);
    assert_eq!(sequence.group_id, 7);
    assert_eq!(sequence.tokens.len(), 5);
    assert_eq!(sequence.old_logprobs.len(), 4);
    assert_eq!(
        sequence.intermediate_returns.len(),
        4,
        "one per trainable token"
    );
}

/// Pins what the intermediate returns *are* today, which is not what
/// "return-to-go" would give.
///
/// A trainable token at position `p` is credited with the rewards of the steps
/// whose range **starts at or after** `p`. So the reward of the action a token
/// belongs to reaches only that action's first token: here the first action is
/// worth 0.25 and the second 0.5, and the four trainable tokens get
/// `[0.75, 0.5, 0.5, 0.0]` - the second token of the first action has already
/// lost its own step's reward, and the last token of the last action carries
/// nothing at all.
///
/// A return-to-go would be `[0.75, 0.75, 0.5, 0.5]`. The inline test in
/// `src/train.rs` pins the same shape (`[2.0, 0.0]`), so this is the intended
/// behaviour as far as the code says; it is recorded here rather than changed,
/// because altering it changes what every agentic run trains on and that is not
/// a `tests/` directory's decision to make.
#[test]
fn intermediate_returns_credit_a_step_only_to_the_token_it_starts_at() {
    let sequence = to_train_sequence(
        &trajectory(Some(1.0), [Some(0.25), Some(0.5)]),
        0,
        TruncationPolicy::Drop,
    )
    .expect("a rewarded trajectory");

    assert_eq!(sequence.intermediate_returns, [0.75, 0.5, 0.5, 0.0]);
}

#[test]
fn a_trajectory_without_step_rewards_carries_no_intermediate_returns_at_all() {
    let sequence = to_train_sequence(
        &trajectory(Some(2.0), [None, None]),
        0,
        TruncationPolicy::Drop,
    )
    .expect("a rewarded trajectory");

    assert_eq!(sequence.reward, 2.0);
    assert!(
        sequence.intermediate_returns.is_empty(),
        "an empty vector says 'no step signal'; zeros would say 'measured, and zero'"
    );
}

#[test]
fn a_truncated_member_is_refused_unless_the_policy_deliberately_keeps_it() {
    let mut truncated = trajectory(Some(1.0), [None, None]);
    truncated.truncated = true;

    let refused = to_train_sequence(&truncated, 0, TruncationPolicy::Drop)
        .expect_err("its reward describes a response the judge never saw the end of");
    assert!(refused.to_string().contains("truncated"), "{refused}");

    // `MinReward` is the policy that says "keep it, and train it as the worst
    // member of its group": the caller has stated what a partial trajectory
    // means, so the missing sequence is intentional.
    assert!(to_train_sequence(&truncated, 0, TruncationPolicy::MinReward).is_ok());
}

#[test]
fn an_unrewarded_trajectory_never_becomes_a_training_sequence() {
    let error = to_train_sequence(&trajectory(None, [None, None]), 0, TruncationPolicy::Drop)
        .expect_err("training on an unrewarded trajectory is training on nothing");
    assert!(error.to_string().contains("rewarded"), "{error}");
}

#[test]
fn a_group_flattens_in_order_and_every_member_keeps_the_group_id() {
    let group = TrajectoryGroup {
        group_id: 3,
        scenario_id: "s1".into(),
        trajectories: vec![
            trajectory(Some(0.0), [None, None]),
            trajectory(Some(1.0), [None, None]),
        ],
    };
    let sequences = groups_to_train_sequences(std::slice::from_ref(&group), TruncationPolicy::Drop)
        .expect("a valid group");

    assert_eq!(sequences.len(), 2);
    assert!(sequences.iter().all(|sequence| sequence.group_id == 3));
    assert_eq!(
        sequences.iter().map(|s| s.reward).collect::<Vec<_>>(),
        [0.0, 1.0]
    );
}
