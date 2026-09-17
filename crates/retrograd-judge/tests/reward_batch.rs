//! The batch contract of `apply_group_scores`, from outside the crate.
//!
//! Before this file existed, the surface a training loop actually calls - a
//! failure policy, a dropped-group budget, and the counts it reports - was
//! only ever exercised from a module that can also reach the private
//! helpers.
//!
//! No transport is involved: `apply_group_scores` takes the judge's answers
//! already collected, which is precisely the seam that lets the policy be
//! tested without an endpoint.

use retrograd_agent_core::config::JudgeFailurePolicy;
use retrograd_agent_core::trajectory::{
    Message, Role, Step, StepKind, Trajectory, TrajectoryGroup,
};
use retrograd_agent_core::{Error, Result};
use retrograd_judge::{Score, apply_group_scores};

fn trajectory(scenario: &str) -> Trajectory {
    Trajectory {
        scenario_id: scenario.to_string(),
        messages: vec![Message::text(Role::User, "prompt")],
        tokens: vec![1, 2, 3],
        old_logprobs: vec![-0.5, -0.25],
        train_mask: vec![false, true, true],
        steps: vec![
            Step {
                kind: StepKind::Context,
                token_range: (0, 1),
                reward: None,
            },
            Step {
                kind: StepKind::PolicyAction,
                token_range: (1, 3),
                reward: None,
            },
        ],
        reward: None,
        truncated: false,
        metadata: serde_json::Map::new(),
        provenance: None,
    }
}

fn group(id: u64, members: usize) -> TrajectoryGroup {
    TrajectoryGroup {
        group_id: id,
        scenario_id: format!("s{id}"),
        trajectories: (0..members)
            .map(|_| trajectory(&format!("s{id}")))
            .collect(),
    }
}

fn score(value: f32) -> Score {
    Score {
        value,
        valid: true,
        explanation: None,
        error: None,
    }
}

#[test]
fn valid_scores_reach_the_trajectories_they_were_produced_for() {
    let mut groups = vec![group(1, 2)];
    let metrics = apply_group_scores(
        &mut groups,
        vec![Ok(vec![score(0.25), score(0.75)])],
        JudgeFailurePolicy::Fail,
        1.0,
        false,
    )
    .expect("a well-formed batch");

    assert_eq!(
        groups[0]
            .trajectories
            .iter()
            .map(|t| t.reward)
            .collect::<Vec<_>>(),
        [Some(0.25), Some(0.75)]
    );
    assert_eq!(metrics.counts.groups, 1);
    assert_eq!(metrics.counts.scored_groups, 1);
    assert_eq!(metrics.counts.degenerate_groups, 0);
}

#[test]
fn a_group_whose_members_all_score_alike_is_counted_as_carrying_no_signal() {
    let mut groups = vec![group(1, 2)];
    let metrics = apply_group_scores(
        &mut groups,
        vec![Ok(vec![score(0.5), score(0.5)])],
        JudgeFailurePolicy::Fail,
        1.0,
        false,
    )
    .expect("a well-formed batch");

    // GRPO centres within the group, so this update contributed no gradient.
    assert_eq!(metrics.counts.degenerate_groups, 1);
    assert_eq!(metrics.degenerate_group_fraction, 1.0);
}

#[test]
fn the_failure_policy_is_the_caller_s_and_is_obeyed_in_both_directions() {
    let failure = || Err(Error::Reward("the endpoint refused".into()));

    let mut fail_fast = vec![group(1, 1)];
    let error = apply_group_scores(
        &mut fail_fast,
        vec![failure()],
        JudgeFailurePolicy::Fail,
        1.0,
        false,
    )
    .expect_err("`fail` must not swallow a judge error");
    assert!(error.to_string().contains("refused"), "{error}");

    let mut drop_group = vec![group(1, 1), group(2, 1)];
    let results: Vec<Result<Vec<Score>>> = vec![failure(), Ok(vec![score(1.0)])];
    let metrics = apply_group_scores(
        &mut drop_group,
        results,
        JudgeFailurePolicy::DropGroup,
        1.0,
        false,
    )
    .expect("`drop_group` absorbs the failure");

    assert_eq!(metrics.counts.dropped_groups, 1);
    assert_eq!(metrics.counts.scored_groups, 1);
    assert!(drop_group[0].trajectories[0].reward.is_none());
    assert_eq!(drop_group[1].trajectories[0].reward, Some(1.0));
    // The last failure survives as prose: a fraction alone does not say whether
    // the transport, the credentials or the answers were the problem.
    assert!(metrics.last_error.is_some());
}

#[test]
fn a_budget_the_batch_blew_through_is_an_error_not_a_metric() {
    let mut groups = vec![group(1, 1), group(2, 1)];
    let results: Vec<Result<Vec<Score>>> = vec![
        Err(Error::Reward("down".into())),
        Err(Error::Reward("down".into())),
    ];
    let error = apply_group_scores(
        &mut groups,
        results,
        JudgeFailurePolicy::DropGroup,
        0.25,
        false,
    )
    .expect_err("dropping every group must not pass as a successful update");
    assert!(error.to_string().contains("drop"), "{error}");
}

#[test]
fn a_batch_that_cannot_be_interpreted_is_refused_before_anything_is_written() {
    let mut none: Vec<TrajectoryGroup> = Vec::new();
    assert!(
        apply_group_scores(&mut none, Vec::new(), JudgeFailurePolicy::Fail, 1.0, false).is_err()
    );

    let mut groups = vec![group(1, 1)];
    // One result for two groups, or a nonsensical budget: both are the caller
    // mis-assembling the batch, and neither may reach a trajectory.
    assert!(
        apply_group_scores(
            &mut groups,
            Vec::new(),
            JudgeFailurePolicy::Fail,
            1.0,
            false
        )
        .is_err()
    );
    assert!(
        apply_group_scores(
            &mut groups,
            vec![Ok(vec![score(1.0)])],
            JudgeFailurePolicy::Fail,
            f32::NAN,
            false,
        )
        .is_err()
    );
    assert!(groups[0].trajectories[0].reward.is_none());
}
