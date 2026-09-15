//! Partial failures and the rollout deadline.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;

use super::*;
use crate::rollout::RolloutEngine;
use crate::tools::HermesToolCallParser;
use crate::trajectory::StepKind;
use retrograd_agent_core::scenario::RolloutLimits;

#[tokio::test]
async fn a_failing_member_costs_its_trajectory_not_the_group() {
    let engine = RolloutEngine::new(
        Arc::new(FlakyPolicy {
            fails: |seed| seed.is_multiple_of(3),
        }),
        group_limits(),
    )
    .unwrap();
    let outcome = engine.rollout_group(&scenario(), 9, 40).await.unwrap();
    let failures = outcome.failures.total();
    assert!(
        (1..9).contains(&failures),
        "the fake must fail some but not all members, got {failures}"
    );
    assert_eq!(outcome.failures.policy, failures, "wrong failure cause");
    assert_eq!(outcome.attempted, 9);
    assert_eq!(outcome.group.unwrap().trajectories.len(), 9 - failures);
    assert!(outcome.last_error.unwrap().contains("no tokens"));
}

#[tokio::test]
async fn a_group_without_two_survivors_is_dropped_whole() {
    let engine = RolloutEngine::new(
        Arc::new(FlakyPolicy {
            fails: |seed| !seed.is_multiple_of(2) || seed.is_multiple_of(4),
        }),
        group_limits(),
    )
    .unwrap();
    let outcome = engine.rollout_group(&scenario(), 2, 40).await.unwrap();
    assert!(outcome.failures.total() >= 1);
    assert!(
        outcome.group.is_none(),
        "one survivor has no relative baseline left"
    );
}

#[tokio::test]
async fn a_group_wide_setup_failure_is_counted_rather_than_propagated() {
    let engine = RolloutEngine::with_tools(
        Arc::new(FakePolicy::default()),
        Arc::new(UnreachableTools),
        Arc::new(HermesToolCallParser),
        group_limits(),
    )
    .unwrap();
    let outcome = engine.rollout_group(&scenario(), 4, 40).await.unwrap();
    assert!(outcome.group.is_none());
    assert_eq!(outcome.attempted, 4);
    assert_eq!(outcome.failures.tool, 4, "{:?}", outcome.failures);
    assert!(outcome.last_error.unwrap().contains("connection refused"));
}

#[tokio::test(start_paused = true)]
async fn an_overrunning_rollout_is_truncated_not_failed() {
    // Same policy as an unbounded token budget: without the deadline this
    // rollout would run every one of its turns.
    let engine = engine_with(
        Arc::new(SlowPolicy {
            per_turn: Duration::from_secs(30),
        }),
        RolloutLimits {
            max_turns: 64,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 4096,
            max_rollout_secs: 60,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(trajectory.truncated, "the deadline must truncate, not fail");
    let turns = trajectory
        .steps
        .iter()
        .filter(|step| step.kind == StepKind::PolicyAction)
        .count();
    assert!(
        (1..64).contains(&turns),
        "the deadline must stop the rollout early, got {turns} turns"
    );
}

#[tokio::test(start_paused = true)]
async fn a_stalled_decode_cannot_outlive_the_deadline() {
    // The deadline must also bound an in-flight policy call; a call that never
    // returns cannot outlive the rollout limit.
    let engine = engine_with(
        Arc::new(StallingPolicy),
        RolloutLimits {
            max_turns: 8,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 4096,
            max_rollout_secs: 60,
            ..Default::default()
        },
    );
    let started = Instant::now();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(
        started.elapsed() <= Duration::from_secs(60),
        "the deadline must bound the decode itself, not just the turn count"
    );
    // Turn 0's action survives; turn 1 never came back.
    assert_eq!(trajectory.tokens, [1, 2, 10, 20]);
    assert!(trajectory.truncated);
}

#[tokio::test(start_paused = true)]
async fn a_hanging_environment_step_cannot_outlive_the_deadline() {
    let factory = Arc::new(CountingFactory {
        step_delay: Some(Duration::from_secs(3600)),
        ..Default::default()
    });
    let engine = engine_with_environments(
        Arc::new(MultiTurnPolicy),
        factory,
        RolloutLimits {
            max_turns: 8,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 4096,
            max_rollout_secs: 60,
            ..Default::default()
        },
    );
    let started = Instant::now();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert!(
        started.elapsed() <= Duration::from_secs(60),
        "a step that never answers must not spend the whole rollout budget"
    );
    // The action is kept, the observation it was waiting for never arrived.
    assert_eq!(trajectory.tokens, [1, 2, 10]);
    assert!(trajectory.truncated);
}
#[tokio::test]
async fn a_zero_deadline_disables_the_check() {
    let engine = engine_with(
        Arc::new(MultiTurnPolicy),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 16,
            max_rollout_secs: 0,
            ..Default::default()
        },
    );
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    assert_eq!(trajectory.tokens, [1, 2, 10, 20, 11]);
}
