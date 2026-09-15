//! Batched agentic rollout.
//!
//! [`RolloutEngine`] collects a whole group in lockstep - one decode batch per
//! turn rather than `group_size` interleaved single-sequence generations,
//! while giving every member its own [`Environment`].
//! The public surface, the construction and the rendering cache live here; the
//! turn loop is in `lockstep`, the per-member bookkeeping in `state`, the
//! tool rendering in `render`, the deadline plumbing in `deadline` and the
//! environment lifecycle in `environments`.

mod deadline;
mod environments;
mod lockstep;
mod render;
mod state;

// `pub(crate)` so the agentic loop's tests reuse the one set of fakes rather
// than growing a second policy and a second environment of their own.
#[cfg(test)]
pub(crate) mod tests;

use std::sync::Arc;
use std::time::Duration;

use retrograd_training::batch::TrainSequence;
// Tokio's clock rather than `std`'s: identical in production, but it lets the
// rollout deadline be tested under a paused clock instead of a real sleep.
use tokio::time::Instant;

use self::deadline::{before_deadline, deadline_expired};
use self::environments::{attach_env_state, close_environments, create_environments};
use self::render::{ToolRendering, prompt_tool_rendering};
use crate::FailureKind;
use crate::env::{Environment, EnvironmentFactory, ToolProviderFactory};
use crate::policy::Policy;
use crate::tools::{HermesToolCallParser, ToolCallParser, ToolProvider};
use crate::trajectory::{Message, Trajectory, TrajectoryGroup};
use crate::{Error, Result};
use retrograd_agent_core::scenario::{RolloutLimits, Scenario, TruncationPolicy};

/// Rollout failures of one group, split by cause. Counted rather than
/// propagated: a group survives losing members, and the split is what
/// distinguishes a flaky environment from a broken policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RolloutFailures {
    pub tool: usize,
    pub policy: usize,
    pub other: usize,
}

impl RolloutFailures {
    pub fn total(&self) -> usize {
        self.tool + self.policy + self.other
    }

    fn record(&mut self, kind: FailureKind) {
        match kind {
            FailureKind::Tool => self.tool += 1,
            FailureKind::Policy => self.policy += 1,
            FailureKind::Other => self.other += 1,
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.tool += other.tool;
        self.policy += other.policy;
        self.other += other.other;
    }
}

/// What a group rollout produced. `group` is `None` when fewer than two members
/// survived: a group centered on a single trajectory has no relative baseline
/// left, the same rule truncation already applies.
pub struct GroupOutcome {
    pub group: Option<TrajectoryGroup>,
    pub attempted: usize,
    pub failures: RolloutFailures,
    /// One failure message, kept for the diagnostic an operator actually needs
    /// when an update dies of accumulated drops.
    pub last_error: Option<String>,
}

pub struct RolloutEngine {
    policy: Arc<dyn Policy>,
    /// Hands out one [`Environment`] per trajectory. `None` is a rollout with
    /// no tools at all, where the policy answers in a single turn.
    environments: Option<Arc<dyn EnvironmentFactory>>,
    /// Read only when the template's own parser is not the one in use: under
    /// [`ToolRendering::Prompt`], where the catalog was written into the system
    /// turn in the `<tool_call>` convention this parser reads. The native path
    /// carries its own parser inside the rendering, which is what keeps the two
    /// from being chosen independently.
    fallback_parser: Arc<dyn ToolCallParser>,
    limits: RolloutLimits,
    /// How tools are rendered, decided once. It depends on the environment kind
    /// and on the model's template, not on the instance, while `rollout` runs
    /// once per trajectory per group per update - listing the tools and
    /// re-serializing every schema on each of those was pure repeated work (and
    /// a round trip when the environment is remote). Hence a cache at the
    /// factory's level, not the instance's.
    tools: tokio::sync::OnceCell<ToolRendering>,
}

impl RolloutEngine {
    pub fn new(policy: Arc<dyn Policy>, limits: RolloutLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            policy,
            environments: None,
            fallback_parser: Arc::new(HermesToolCallParser),
            limits,
            tools: tokio::sync::OnceCell::new(),
        })
    }

    /// Stateless tools shared by every trajectory. Equivalent to
    /// [`RolloutEngine::with_environments`] over a
    /// [`ToolProviderFactory`].
    pub fn with_tools(
        policy: Arc<dyn Policy>,
        tools: Arc<dyn ToolProvider>,
        parser: Arc<dyn ToolCallParser>,
        limits: RolloutLimits,
    ) -> Result<Self> {
        Self::with_environments(
            policy,
            Arc::new(ToolProviderFactory::new(tools)),
            parser,
            limits,
        )
    }

    /// One environment instance per trajectory. Use this over
    /// [`RolloutEngine::with_tools`] as soon as the task has state: sharing it
    /// across a group lets the members contaminate each other, and a
    /// group-relative baseline over contaminated members measures nothing.
    pub fn with_environments(
        policy: Arc<dyn Policy>,
        environments: Arc<dyn EnvironmentFactory>,
        parser: Arc<dyn ToolCallParser>,
        limits: RolloutLimits,
    ) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            policy,
            environments: Some(environments),
            fallback_parser: parser,
            limits,
            tools: tokio::sync::OnceCell::new(),
        })
    }

    /// How many tools the run declared, once a rollout has settled the
    /// rendering. Zero before that, and zero for a run with no tools at all,
    /// which is what the update loop needs to tell "the model called nothing"
    /// from "there was nothing to call".
    pub fn declared_tools(&self) -> usize {
        self.tools.get().map_or(0, ToolRendering::declared_tools)
    }

    /// Lists the tools once and asks the policy whether its template can render
    /// them, which together decide the rendering for the whole run.
    async fn tool_rendering(
        &self,
        environment: Option<&dyn Environment>,
    ) -> Result<&ToolRendering> {
        self.tools
            .get_or_try_init(|| async {
                let Some(environment) = environment else {
                    return Ok(ToolRendering::None);
                };
                let specs = environment.tools().await?;
                if specs.is_empty() {
                    return Ok(ToolRendering::None);
                }
                if !self.policy.supports_native_tools().await? {
                    return prompt_tool_rendering(&specs);
                }
                // Asked here and not at construction, because the generated
                // grammar may name the functions: the parser is derived from
                // the template *and* the catalog, and the catalog is only
                // known once the environment has listed it.
                match self.policy.tool_call_parser(&specs).await? {
                    Some(parser) => Ok(ToolRendering::Native { specs, parser }),
                    // The template renders tools but nothing could be derived
                    // to read them back. Rendering natively anyway would
                    // rebuild the exact asymmetry this is here to prevent, so
                    // both halves fall back together.
                    None => {
                        tracing::warn!(
                            "the model's chat template renders tools but yields no parser for \
                             its own call format; falling back to the prompt-described convention"
                        );
                        prompt_tool_rendering(&specs)
                    }
                }
            })
            .await
    }

    /// One rendering call for the whole engine, so the native and prompt paths
    /// cannot drift apart between the opening prompt and the re-render that
    /// follows a tool turn - a difference there would rewrite framing already
    /// committed to the token stream.
    async fn render_framing(
        &self,
        messages: &[Message],
        rendering: &ToolRendering,
        add_assistant: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let tools = match rendering {
            ToolRendering::Native { specs, .. } => specs.as_slice(),
            // The catalog is in the system turn already, or there is none: the
            // template must not be handed one either way.
            ToolRendering::Prompt { .. } | ToolRendering::None => &[],
        };
        self.policy
            .render_chat_framing(messages, tools, add_assistant)
            .await
    }

    /// Collects a single trajectory. Kept as a one-member lockstep rollout so
    /// there is exactly one code path to test.
    pub async fn rollout(&self, scenario: &Scenario, seed: u64) -> Result<Trajectory> {
        self.rollout_many(scenario, &[seed])
            .await?
            .pop()
            .expect("one seed yields one result")
    }

    /// Collects a whole group in lockstep: one decode batch per turn instead of
    /// `group_size` interleaved single-sequence generations. Turn 0 shares the
    /// rendered scenario, so the prompt is decoded once for the group; later
    /// turns have diverged and go through continuous batching on the live
    /// members only.
    ///
    /// Individual members may die without taking the update with them - see
    /// [`GroupOutcome`].
    pub async fn rollout_group(
        &self,
        scenario: &Scenario,
        group_size: usize,
        base_seed: u64,
    ) -> Result<GroupOutcome> {
        if group_size < 2 {
            return Err(Error::invalid("rollout group_size must be at least 2"));
        }
        let seeds = (0..group_size)
            .map(|member| base_seed.wrapping_add(member as u64))
            .collect::<Vec<_>>();
        let results = match self.rollout_many(scenario, &seeds).await {
            Ok(results) => results,
            // Setup failed for the group as a whole (tool listing, scenario
            // rendering): every member is lost, but that is still a counted
            // failure rather than a dead update.
            Err(error) => {
                let mut failures = RolloutFailures::default();
                for _ in 0..group_size {
                    failures.record(error.failure_kind());
                }
                return Ok(GroupOutcome {
                    group: None,
                    attempted: group_size,
                    failures,
                    last_error: Some(error.to_string()),
                });
            }
        };

        let mut trajectories = Vec::with_capacity(group_size);
        let mut failures = RolloutFailures::default();
        let mut last_error = None;
        for result in results {
            match result {
                Ok(trajectory) => trajectories.push(trajectory),
                Err(error) => {
                    failures.record(error.failure_kind());
                    last_error = Some(error.to_string());
                }
            }
        }
        // A group is scored by its environment or by the judge, never half by
        // each. A member the environment happened not to grade would otherwise
        // be sent to the judge together with its siblings, whose step rewards
        // the judge's own reward would then be added on top of.
        if trajectories
            .iter()
            .any(|trajectory| trajectory.reward.is_some())
        {
            for trajectory in &mut trajectories {
                trajectory.reward.get_or_insert(0.0);
            }
        }
        let group = (trajectories.len() >= 2).then(|| TrajectoryGroup {
            group_id: stable_group_id(&scenario.id, base_seed),
            scenario_id: scenario.id.clone(),
            trajectories,
        });
        Ok(GroupOutcome {
            group,
            attempted: group_size,
            failures,
            last_error,
        })
    }

    /// Creates one environment per seed, runs the lockstep loop over them, and
    /// closes every instance - on the error and deadline paths too, which is
    /// why the loop lives in its own function.
    async fn rollout_many(
        &self,
        scenario: &Scenario,
        seeds: &[u64],
    ) -> Result<Vec<Result<Trajectory>>> {
        scenario.validate()?;
        if seeds.is_empty() {
            return Err(Error::invalid("rollout requires at least one seed"));
        }
        // Anchored here rather than inside the lockstep loop: creating the
        // environments is already the rollout's wall clock - a factory that
        // blocks on a container start would otherwise run outside the budget
        // the deadline is supposed to bound end to end.
        let deadline = (self.limits.max_rollout_secs > 0)
            .then(|| Instant::now() + Duration::from_secs(self.limits.max_rollout_secs));
        let mut environments = match before_deadline(
            deadline,
            create_environments(self.environments.as_ref(), seeds.len()),
        )
        .await
        {
            Some(environments) => environments?,
            None => return Err(Error::Tool(deadline_expired("creating the environments"))),
        };
        let mut outcome = self
            .rollout_lockstep(scenario, seeds, &mut environments, deadline)
            .await;
        if let Ok(trajectories) = &mut outcome {
            attach_env_state(&mut environments, trajectories, deadline).await;
        }
        close_environments(&mut environments).await;
        outcome
    }

    /// Kept as an associated function so call sites read the same as before the
    /// conversion moved to [`crate::train`].
    pub fn groups_to_train_sequences(
        groups: &[TrajectoryGroup],
        truncation: TruncationPolicy,
    ) -> Result<Vec<TrainSequence>> {
        crate::train::groups_to_train_sequences(groups, truncation)
    }
}

fn stable_group_id(scenario_id: &str, update_seed: u64) -> u64 {
    // FNV-1a gives a stable, dependency-free id across processes/platforms.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in scenario_id
        .as_bytes()
        .iter()
        .copied()
        .chain(update_seed.to_le_bytes())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
