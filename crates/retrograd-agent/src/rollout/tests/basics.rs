//! Single rollouts: the trainable mask, the seed derivation, the token budget
//! and the prefix-stability check.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;
use crate::rollout::RolloutEngine;
use crate::rollout::lockstep::sampling_seed;
use crate::tools::HermesToolCallParser;
use crate::trajectory::StepKind;
use retrograd_agent_core::scenario::RolloutLimits;

#[tokio::test]
async fn single_turn_rollout_builds_an_exact_policy_mask() {
    let engine = RolloutEngine::new(
        Arc::new(FakePolicy::default()),
        RolloutLimits {
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(trajectory.tokens, [1, 2, 11]);
    assert_eq!(trajectory.train_mask, [false, false, true]);
    assert_eq!(trajectory.old_logprobs, [-0.5]);
    assert!(!trajectory.truncated);
}

#[tokio::test]
async fn group_uses_distinct_seeds_and_a_stable_id() {
    let policy = Arc::new(FakePolicy::default());
    let engine = RolloutEngine::new(policy.clone(), RolloutLimits::default()).unwrap();
    let first = engine.rollout_group(&scenario(), 3, 40).await.unwrap();
    let second = engine.rollout_group(&scenario(), 3, 40).await.unwrap();
    let first = first.group.unwrap();
    let second = second.group.unwrap();
    assert_eq!(first.group_id, second.group_id);
    assert_eq!(first.trajectories.len(), 3);
    let seeds = policy.seeds.lock().unwrap();
    let members = &seeds[..3];
    assert_eq!(
        members
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3,
        "group members must sample with distinct seeds: {members:?}"
    );
    // Same base seed replayed: the derivation stays reproducible.
    assert_eq!(members, &seeds[3..6]);
}

#[test]
fn sampling_seeds_separate_members_turns_and_updates() {
    // Members sit in the low bits and updates in the high bits, the range a
    // truncating cast to u32 would have discarded entirely.
    let update_stride = 1_u64 << 32;
    let seeds = [
        sampling_seed(40, 0),
        sampling_seed(41, 0),
        sampling_seed(40, 1),
        sampling_seed(40 + update_stride, 0),
        sampling_seed(41 + update_stride, 0),
    ];
    assert_eq!(
        seeds.iter().collect::<std::collections::HashSet<_>>().len(),
        seeds.len(),
        "member/turn/update seeds must not collide: {seeds:?}"
    );
}

#[tokio::test]
async fn tool_observation_is_untrained_between_two_policy_actions() {
    let engine = engine_with(
        Arc::new(MultiTurnPolicy),
        RolloutLimits {
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 16,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(trajectory.tokens, [1, 2, 10, 20, 11]);
    assert_eq!(trajectory.train_mask, [false, false, true, false, true]);
    assert_eq!(trajectory.old_logprobs, [-0.25, -0.25]);
    assert_eq!(
        trajectory
            .steps
            .iter()
            .map(|step| step.kind)
            .collect::<Vec<_>>(),
        [
            StepKind::Context,
            StepKind::PolicyAction,
            StepKind::ToolResult,
            StepKind::PolicyAction
        ]
    );
}

#[tokio::test]
async fn the_tool_list_is_fetched_once_for_the_whole_engine() {
    // The tool list is fetched once and reused across rollouts, avoiding an MCP
    // round trip for every group and update.
    let tools = Arc::new(CountingTools::default());
    let engine = RolloutEngine::with_tools(
        Arc::new(FakePolicy::default()),
        tools.clone(),
        Arc::new(HermesToolCallParser),
        RolloutLimits::default(),
    )
    .unwrap();
    engine.rollout_group(&scenario(), 4, 1).await.unwrap();
    engine.rollout(&scenario(), 2).await.unwrap();
    assert_eq!(
        tools.listings.load(Ordering::Relaxed),
        1,
        "the tool list must be built once and reused"
    );
}

#[tokio::test]
async fn a_turn_cut_off_by_its_budget_is_truncated() {
    let engine = RolloutEngine::new(
        Arc::new(BudgetPolicy { eog: false }),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 3,
            max_trajectory_tokens: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(trajectory.truncated);
}

#[tokio::test]
async fn a_stop_token_landing_on_the_budget_is_not_truncated() {
    let engine = RolloutEngine::new(
        Arc::new(BudgetPolicy { eog: true }),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 3,
            max_trajectory_tokens: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(
        !trajectory.truncated,
        "a complete response must not be discarded for using its whole budget"
    );
}

#[tokio::test]
async fn the_total_token_budget_truncates_the_trajectory() {
    let engine = RolloutEngine::new(
        Arc::new(BudgetPolicy { eog: true }),
        RolloutLimits {
            max_turns: 8,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(trajectory.truncated);
    assert!(trajectory.tokens.len() <= 5);
}

#[tokio::test]
async fn a_turn_terminator_the_policy_already_emitted_is_not_written_twice() {
    // The template closes the assistant turn the policy just closed itself, so
    // the observation framing opens on a token already in the stream. Appending
    // it blindly would put a doubled terminator in front of every observation,
    // a sequence no chat model has ever been trained on.
    let engine = engine_with(
        Arc::new(SharedTerminatorPolicy),
        RolloutLimits {
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 16,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory.tokens,
        [1, 2, 10, TERMINATOR, 20, 11, TERMINATOR]
    );
    assert_eq!(
        trajectory.train_mask,
        [false, false, true, true, false, true, true]
    );
}

#[tokio::test]
async fn an_unstable_chat_template_fails_instead_of_training_off_policy() {
    let engine = engine_with(
        Arc::new(UnstablePrefixPolicy),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 32,
            ..Default::default()
        },
    );
    let error = engine.rollout(&scenario(), 7).await.unwrap_err();
    assert!(
        error.to_string().contains("rewrote turn 0"),
        "unexpected error: {error}"
    );
}

/// Both readings of a turn that named no tool, on the same policy - one that
/// only ever answers in prose.
///
/// The default is the question-answering one: the answer is the trajectory, and
/// it ends there. The other is for a world that decides its own terminal state,
/// where ending here would make "answer and stop" the shortest, cheapest and
/// therefore best-scoring member of every group.
#[tokio::test]
async fn a_turn_that_calls_nothing_ends_the_trajectory_by_default() {
    let engine = engine_with(
        Arc::new(FakePolicy::default()),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 32,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .count(),
        1
    );
    assert!(!trajectory.truncated);
}

#[tokio::test]
async fn a_turn_that_calls_nothing_costs_a_turn_when_the_world_ends_the_episode() {
    let engine = engine_with(
        Arc::new(FakePolicy::default()),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 32,
            end_on_no_tool_call: false,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .count(),
        4,
        "the whole turn budget is spent, not one turn"
    );
    let observations = trajectory
        .messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .collect::<Vec<_>>();
    assert_eq!(observations.len(), 3, "one per turn but the last");
    assert!(observations.iter().all(|message| message.is_error));
    assert!(
        observations[0].content.contains("called no tool"),
        "unexpected observation: {}",
        observations[0].content
    );
    assert!(
        trajectory.truncated,
        "running out of turns without the environment saying done is a truncation"
    );
}

/// The flag cannot make a run with no catalog spend its whole turn budget
/// asking for a tool that was never declared.
#[tokio::test]
async fn a_run_without_tools_still_ends_on_its_answer() {
    let engine = RolloutEngine::new(
        Arc::new(FakePolicy::default()),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 32,
            end_on_no_tool_call: false,
            ..Default::default()
        },
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .count(),
        1
    );
}

/// The early exit for a policy that keeps missing the format under
/// `end_on_no_tool_call = false`: after N turns in a row without a valid call
/// the trajectory is cut as a truncation, and the turns it would have decoded
/// to reach the turn budget are never decoded.
#[tokio::test]
async fn consecutive_turns_without_a_call_cut_the_trajectory_as_a_truncation() {
    let engine = engine_with(
        Arc::new(FakePolicy::default()),
        RolloutLimits {
            max_turns: 8,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 64,
            end_on_no_tool_call: false,
            max_failed_turns: 2,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .count(),
        2,
        "the second failed turn is the last one decoded"
    );
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Tool)
            .count(),
        1,
        "only the first failed turn came back as an observation"
    );
    assert!(trajectory.truncated, "the cut takes the truncation path");
}

/// The count is consecutive, not total: a turn that called a tool resets it,
/// so a policy that is playing badly is not confused with one that is not
/// playing at all.
#[tokio::test]
async fn a_valid_call_resets_the_failed_turn_count() {
    // Calls a tool on turn 0, then answers in prose on every later turn.
    let engine = engine_with(
        Arc::new(MultiTurnPolicy),
        RolloutLimits {
            max_turns: 8,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 64,
            end_on_no_tool_call: false,
            max_failed_turns: 2,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(
        trajectory
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .count(),
        3,
        "one call, then two failed turns before the cut; the call did not count"
    );
    assert!(trajectory.truncated);
}
