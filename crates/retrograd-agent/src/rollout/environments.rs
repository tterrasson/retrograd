//! The lifecycle of the per-trajectory environments: creating them, stepping
//! one member's calls against its own instance, reading the terminal state the
//! judge will need, and releasing them on every path.

use std::sync::Arc;

use futures::future::join_all;
use tokio::time::Instant;

use super::deadline::before_deadline;
use crate::env::{EnvState, Environment, EnvironmentFactory};
use crate::tools::{ParsedAssistant, ToolResult};
use crate::trajectory::Trajectory;
use crate::{Error, Result};

/// One instance per trajectory. A creation failure closes whatever was
/// already built rather than leaking it, and is a group-wide setup error.
pub(super) async fn create_environments(
    factory: Option<&Arc<dyn EnvironmentFactory>>,
    count: usize,
) -> Result<Vec<Box<dyn Environment>>> {
    let Some(factory) = factory else {
        return Ok(Vec::new());
    };
    let mut environments = Vec::with_capacity(count);
    for _ in 0..count {
        match factory.create().await {
            Ok(environment) => environments.push(environment),
            Err(error) => {
                close_environments(&mut environments).await;
                return Err(error);
            }
        }
    }
    Ok(environments)
}

/// What one member's environment produced over a single turn.
pub(super) struct EnvironmentTurn {
    pub(super) observations: Vec<ToolResult>,
    /// Summed over the turn's calls, `None` when the environment graded none of
    /// them - the difference between "worth zero" and "not my job".
    pub(super) reward: Option<f32>,
    pub(super) done: bool,
    /// The environment broke, as opposed to an action of the policy failing.
    /// Fatal to the member, see [`step_environment`].
    pub(super) failure: Option<Error>,
}

/// Runs a member's calls against its own environment, in order.
///
/// An `Err` out of [`Environment::step`] is *not* an observation: it says the
/// environment is broken - a lost HTTP session, an unreachable server - and the
/// trajectory dies with it. Turning it into text the policy reads would let a
/// rollout continue, be scored and be trained against a world that no longer
/// exists. An action that merely failed comes back as `Ok` with `is_error` set,
/// and that one *is* an observation; see [`Environment::step`] for the split.
pub(super) async fn step_environment(
    environment: &mut dyn Environment,
    parsed: &ParsedAssistant,
) -> EnvironmentTurn {
    let mut turn = EnvironmentTurn {
        observations: parsed.parse_errors.clone(),
        reward: None,
        done: false,
        failure: None,
    };
    for call in &parsed.tool_calls {
        match environment.step(call).await {
            Ok(outcome) => {
                if let Some(reward) = outcome.reward {
                    *turn.reward.get_or_insert(0.0) += reward;
                }
                turn.observations.push(outcome.result);
                if outcome.done {
                    // The task is over; the calls the policy queued behind this
                    // one no longer have an environment to act on.
                    turn.done = true;
                    break;
                }
            }
            // The environment is gone; the calls queued behind this one have
            // nothing left to act on either.
            Err(error) => {
                turn.failure = Some(error);
                break;
            }
        }
    }
    turn
}

/// Reads each environment's terminal state and stores it on its trajectory,
/// under `metadata.env_state`.
///
/// The consumer is the judge: for a code task, what should be graded is the diff
/// the trajectory produced, not the dialogue that produced it, and this is the
/// only moment where both still exist. It runs before `close`, which is what
/// releases the sandbox the state is read from.
///
/// A failure here is a warning and nothing more. Nothing downstream is
/// conditioned on the state - the judge falls back to the transcript - so making
/// a trajectory that ran fine depend on a `git diff` would be trading a complete
/// rollout for a summary.
pub(super) async fn attach_env_state(
    environments: &mut [Box<dyn Environment>],
    trajectories: &mut [Result<Trajectory>],
    deadline: Option<Instant>,
) {
    let states = join_all(
        environments
            .iter_mut()
            .zip(trajectories.iter())
            // A failed member has no trajectory to carry the state, and a
            // failed environment is exactly the one whose state cannot be
            // trusted.
            .filter(|(_, trajectory)| trajectory.is_ok())
            .map(|(environment, _)| environment.state()),
    );
    let Some(states) = before_deadline(deadline, states).await else {
        tracing::warn!("reading the environment state ran out of rollout budget");
        return;
    };
    for (trajectory, state) in trajectories
        .iter_mut()
        .filter(|trajectory| trajectory.is_ok())
        .zip(states)
    {
        let state = match state {
            Ok(state) => state,
            Err(error) => {
                tracing::warn!("reading a rollout environment state failed: {error}");
                continue;
            }
        };
        if state == EnvState::default() {
            continue;
        }
        match serde_json::to_value(&state) {
            Ok(value) => {
                let Ok(trajectory) = trajectory else { continue };
                trajectory.metadata.insert("env_state".into(), value);
            }
            Err(error) => tracing::warn!("serializing a rollout environment state: {error}"),
        }
    }
}

/// Closes every instance, whatever the rollout did. A close that fails is
/// logged and not propagated: the trajectories are already collected, and
/// losing them to a failing teardown would be the worse outcome.
pub(super) async fn close_environments(environments: &mut [Box<dyn Environment>]) {
    for error in join_all(
        environments
            .iter_mut()
            .map(|environment| environment.close()),
    )
    .await
    .into_iter()
    .filter_map(Result::err)
    {
        tracing::warn!("closing a rollout environment failed: {error}");
    }
}
