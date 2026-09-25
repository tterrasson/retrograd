//! One environment per trajectory.
//!
//! [`ToolProvider`](crate::ToolProvider) is a *shared*, stateless brick: one
//! instance answers every member of every group. That is exactly right for
//! read-only tools and exactly wrong as soon as the task has state - a
//! filesystem, a shell, a database. Two members of one group writing to the
//! same state contaminate each other, and the GRPO relative baseline stops
//! measuring the policy.
//!
//! An [`Environment`] is therefore created per trajectory, `reset` before the
//! first turn and `close`d at the end of the rollout - including when the
//! rollout fails or hits its deadline.
//!
//! The shape is the Gymnasium/OpenEnv one - `reset`, `step`, `state`, `close`,
//! with the types staying Rust-typed rather than serialized `Action`/
//! `Observation` blobs. Adapting to a wire protocol is a job for a client
//! implementation, not for the contract.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::scenario::Scenario;
use crate::tools::{ToolCall, ToolResult, ToolSpec};

/// What one environment step produced.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutcome {
    pub result: ToolResult,
    /// Step reward, when the environment can grade the action itself. It lands
    /// on the [`StepKind::ToolResult`](crate::StepKind::ToolResult) step as an
    /// intermediate return, and it takes the whole trajectory out of the
    /// judge's hands: a verifiable task should not be scored by an LLM.
    pub reward: Option<f32>,
    /// The environment declares the task finished. A third stop condition next
    /// to "the policy asked for no tool" and "the budget is spent" - and unlike
    /// those two it is not a truncation: the trajectory is complete.
    pub done: bool,
}

impl StepOutcome {
    /// The stateless shape: a plain tool result with no verdict of its own.
    pub fn tool(result: ToolResult) -> Self {
        Self {
            result,
            reward: None,
            done: false,
        }
    }
}

/// A serializable snapshot of where an episode ended up.
///
/// Its consumer is the judge. For a coding task, what decides the reward is the
/// diff that was produced, not the dialogue that produced it; `summary` is where
/// an environment puts that. Judging the transcript instead is the main source
/// of reward noise on tasks whose outcome is inspectable.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvState {
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cumulative_reward: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// A single trajectory's view of the world. Owned by one rollout, never shared.
#[async_trait]
pub trait Environment: Send {
    /// Prepares the environment for one trajectory. The returned text, when
    /// present, is an opening observation appended to the scenario as a user
    /// turn - a task description the environment generated from `seed`, say.
    ///
    /// Returning `None` is what keeps a group's turn 0 on a single shared
    /// prompt: distinct openings make the members diverge before their first
    /// sampled token, so each one has to be rendered and decoded separately.
    async fn reset(&mut self, scenario: &Scenario, seed: u64) -> Result<Option<String>>;

    /// Runs one action, and draws the line between a failed action and a broken
    /// world.
    ///
    /// `Err` means the environment itself is unusable - a lost session, a server
    /// that went away - and it **costs the trajectory**. Nothing else is
    /// truthful: the tokens already collected are conditioned on a world that no
    /// longer answers, so continuing would score and train a rollout that never
    /// really happened.
    ///
    /// An action that merely failed is `Ok` with
    /// [`ToolResult::is_error`](crate::tools::ToolResult::is_error) set. That one
    /// is an observation: it enters the prompt, the policy reads it and reacts,
    /// and it is counted by `agent/tool_error_fraction` rather than as a failure.
    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome>;

    /// The tools a trajectory of `scenario` is offered. Read before `reset`,
    /// once per group, so it must be a function of the scenario and never of
    /// the instance's state: every member of a group sees the same tools.
    async fn tools(&self, scenario: &Scenario) -> Result<Vec<ToolSpec>>;

    /// Where the episode ended up. The default is empty: an environment with
    /// nothing inspectable to report is the common case, and reporting nothing
    /// is better than reporting a placeholder the judge would read as fact.
    ///
    /// `&mut self` although it reads rather than acts, for two reasons: it
    /// matches the rest of the trait (an environment is owned by exactly one
    /// rollout, so there is never a second reader), and a defaulted `&self`
    /// method would force `Environment` to be `Sync` at every call through
    /// `dyn` - a constraint on implementations that nothing here needs.
    async fn state(&mut self) -> Result<EnvState> {
        Ok(EnvState::default())
    }

    async fn close(&mut self) -> Result<()>;
}

#[async_trait]
pub trait EnvironmentFactory: Send + Sync {
    async fn create(&self) -> Result<Box<dyn Environment>>;

    /// Called once, before the first rollout: validate what every scenario asks
    /// of this environment kind, pull images, warm a pool. Failing here is a
    /// configuration error the operator can fix; failing on the 700th rollout is
    /// a lost run.
    async fn prepare(&self, scenarios: &[Scenario]) -> Result<()> {
        let _ = scenarios;
        Ok(())
    }

    /// Best-effort warm-up of `count` instances for the next round of rollouts.
    /// The natural moment to call it is while the optimizer step runs, which is
    /// dead time on the environment side.
    async fn prewarm(&self, count: usize) -> Result<()> {
        let _ = count;
        Ok(())
    }

    /// Releases whatever `prepare` acquired. Never fails a run: the
    /// trajectories are already collected.
    async fn shutdown(&self) {}

    /// Destroys the world *now*, without waiting for the trajectories that are
    /// still using it, and returns how many instances that took down.
    ///
    /// The difference with [`shutdown`](Self::shutdown) is who it is for.
    /// `shutdown` runs on the way out of a finished run: it drains, so a lease
    /// still out is given time to come back and be recycled properly. This one
    /// runs from a signal handler, where nothing is going to come back - the
    /// process is about to exit - and a drain would only mean the containers
    /// outlive it. See [`crate::interrupt`].
    ///
    /// The count is for the operator, who otherwise has no way to tell a
    /// cleanup that ran from one that silently found nothing.
    async fn force_cleanup(&self) -> usize {
        0
    }

    /// The immutable identity of what the episodes ran in - a container image
    /// digest, typically - once `prepare` has resolved it.
    ///
    /// It belongs to the run's record rather than to a log line: a tag moves,
    /// and a run resumed three weeks later against `python:3.12-slim` is not the
    /// same environment. `None` when the environment has no such identity (a
    /// local sandbox, an HTTP server someone else operates).
    fn pinned_image(&self) -> Option<String> {
        None
    }

    /// What the environment side costs, reported next to the training metrics.
    ///
    /// A pool's reuse fraction and acquisition latency are the numbers that say
    /// whether warming containers during the optimizer step bought anything.
    /// Read after every update, so the values are cumulative or last-window as
    /// each implementation sees fit - not a snapshot anyone must sample at a
    /// particular moment.
    fn metric_values(&self) -> Vec<retrograd_metrics::MetricValue> {
        Vec::new()
    }
}
