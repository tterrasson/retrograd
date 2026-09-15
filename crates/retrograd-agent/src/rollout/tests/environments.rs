//! One environment instance per trajectory: isolation, teardown, terminal
//! state, and the split between a failed action and a broken world.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;
use crate::rollout::RolloutEngine;
use crate::tools::HermesToolCallParser;
use crate::trajectory::Role;
use retrograd_agent_core::scenario::TruncationPolicy;

#[tokio::test]
async fn every_member_gets_its_own_environment_and_all_of_them_are_closed() {
    let factory = Arc::new(CountingFactory::default());
    let engine =
        engine_with_environments(Arc::new(MultiTurnPolicy), factory.clone(), group_limits());
    engine.rollout_group(&scenario(), 8, 40).await.unwrap();
    assert_eq!(factory.ledger.created.load(Ordering::Relaxed), 8);
    assert_eq!(factory.ledger.closed.load(Ordering::Relaxed), 8);

    // Same guarantee when the rollout dies in the middle of a tool turn:
    // teardown is not on the happy path.
    let factory = Arc::new(CountingFactory::default());
    let engine = engine_with_environments(
        Arc::new(UnstablePrefixPolicy),
        factory.clone(),
        group_limits(),
    );
    let outcome = engine.rollout_group(&scenario(), 4, 40).await.unwrap();
    assert!(outcome.group.is_none(), "the whole group must have failed");
    assert_eq!(factory.ledger.created.load(Ordering::Relaxed), 4);
    assert_eq!(
        factory.ledger.closed.load(Ordering::Relaxed),
        4,
        "a failed rollout must still release its environments"
    );
}

#[tokio::test]
async fn the_terminal_environment_state_travels_with_its_trajectory() {
    // Read before `close` - the sandbox that answers it is released there,
    // and stored on the member's own metadata, which is the only channel to
    // the judge. Without this the diff a code task produced is invisible and
    // the judge grades the dialogue instead of the result.
    let factory = Arc::new(CountingFactory {
        summary: Some("diff".into()),
        ..Default::default()
    });
    let engine =
        engine_with_environments(Arc::new(MultiTurnPolicy), factory.clone(), group_limits());
    let group = engine
        .rollout_group(&scenario(), 3, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    assert_eq!(factory.ledger.state_reads.load(Ordering::Relaxed), 3);
    for trajectory in &group.trajectories {
        assert_eq!(
            trajectory.metadata["env_state"]["summary"],
            serde_json::json!("diff after 1 steps")
        );
    }

    // An environment with nothing to report leaves the metadata alone rather
    // than writing an empty state the judge would have to filter out.
    let factory = Arc::new(CountingFactory::default());
    let engine =
        engine_with_environments(Arc::new(MultiTurnPolicy), factory.clone(), group_limits());
    let group = engine
        .rollout_group(&scenario(), 2, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    assert!(group.trajectories[0].metadata.get("env_state").is_none());
}

#[tokio::test]
async fn a_stateful_environment_does_not_leak_between_members() {
    // The whole point of an instance per trajectory: with the shared
    // provider these observations would read 1, 2, 3, 4 - the members would
    // be reading each other's state and the group baseline would be noise.
    let factory = Arc::new(CountingFactory::default());
    let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
    let group = engine
        .rollout_group(&scenario(), 4, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    for trajectory in &group.trajectories {
        let observation = trajectory
            .messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .expect("each member calls the tool once");
        assert!(
            observation.content.ends_with(": 1"),
            "each member must see its own first step: {}",
            observation.content
        );
    }
}

#[tokio::test]
async fn an_environment_that_declares_done_ends_a_complete_trajectory() {
    let factory = Arc::new(CountingFactory {
        done_after: Some(1),
        ..Default::default()
    });
    let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    // Without `done` this policy answers on a second turn: [1, 2, 10, 20, 11].
    assert_eq!(trajectory.tokens, [1, 2, 10, 20]);
    assert!(
        !trajectory.truncated,
        "a task the environment declares finished is complete, not cut short"
    );
}

#[tokio::test]
async fn an_environment_reward_takes_the_group_out_of_the_judge_s_hands() {
    let factory = Arc::new(CountingFactory {
        reward: Some(0.5),
        ..Default::default()
    });
    let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
    let group = engine
        .rollout_group(&scenario(), 4, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    for trajectory in &group.trajectories {
        // The step carries the reward; the trajectory's own final reward is
        // zero but *set*, which is the signal that skips the judge.
        assert_eq!(trajectory.reward, Some(0.0));
        let step_rewards = trajectory
            .steps
            .iter()
            .filter_map(|step| step.reward)
            .collect::<Vec<_>>();
        assert_eq!(step_rewards, [0.5]);
        assert_eq!(
            crate::train::to_train_sequence(trajectory, 1, TruncationPolicy::Drop)
                .unwrap()
                .reward,
            0.5
        );
    }
}

#[tokio::test]
async fn not_submitting_an_explicit_verifier_receives_its_failure_reward() {
    let factory = Arc::new(CountingFactory::default());
    let engine = engine_with_environments(
        Arc::new(DivergentPolicy::default()),
        factory,
        group_limits(),
    );
    let mut task = scenario();
    task.metadata.insert(
        "env".into(),
        serde_json::json!({
            "verify": {
                "command": ["true"],
                "reward_on_success": 1.0,
                "reward_on_failure": -0.75
            }
        }),
    );
    let group = engine
        .rollout_group(&task, 2, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    assert_eq!(group.trajectories.len(), 2);
    for trajectory in &group.trajectories {
        assert_eq!(trajectory.reward, Some(-0.75));
    }
}

#[tokio::test]
async fn an_environment_opening_costs_the_shared_turn_zero_prompt() {
    // A per-seed opening observation makes the members diverge before their
    // first sampled token, so turn 0 can no longer be one shared decode.
    let policy = Arc::new(DivergentPolicy::default());
    let factory = Arc::new(CountingFactory {
        opening: Some("task".into()),
        ..Default::default()
    });
    let engine = engine_with_environments(policy.clone(), factory, group_limits());
    engine.rollout_group(&scenario(), 4, 40).await.unwrap();
    assert!(
        policy.shared_batches.lock().unwrap().is_empty(),
        "distinct openings cannot share a prompt"
    );
    assert_eq!(
        policy.continuous_batches.lock().unwrap()[0],
        4,
        "turn 0 must still batch all four members, just heterogeneously"
    );
}

#[tokio::test]
async fn a_broken_environment_costs_the_trajectory_instead_of_becoming_an_observation() {
    // The trajectory's tokens are conditioned on a world that stopped
    // answering: reading the failure back as an observation would let the
    // rollout finish, be scored and be trained as if it had really happened.
    let factory = Arc::new(CountingFactory {
        breaks_at: Some(1),
        ..Default::default()
    });
    let engine =
        engine_with_environments(Arc::new(MultiTurnPolicy), factory.clone(), group_limits());
    let outcome = engine.rollout_group(&scenario(), 4, 40).await.unwrap();
    assert!(outcome.group.is_none(), "no member can survive this");
    assert_eq!(outcome.failures.tool, 4, "{:?}", outcome.failures);
    assert!(outcome.last_error.unwrap().contains("lost its session"));
    assert_eq!(
        factory.ledger.closed.load(Ordering::Relaxed),
        4,
        "a broken environment must still be released"
    );
}

#[tokio::test]
async fn a_failing_tool_call_is_still_an_observation_the_policy_reads() {
    // The other side of the split: a stateless provider has no world to lose,
    // so its errors are failed *actions*. This is what the MCP provider has
    // always done with its own timeouts.
    let engine = RolloutEngine::with_tools(
        Arc::new(MultiTurnPolicy),
        Arc::new(FailingCallTools),
        Arc::new(HermesToolCallParser),
        group_limits(),
    )
    .unwrap();
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    let observation = trajectory
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("the failure must come back as an observation");
    assert!(observation.is_error);
    assert!(
        observation.content.contains("connection refused"),
        "the policy has to be able to read what went wrong: {}",
        observation.content
    );
    assert!(!trajectory.truncated);
}
