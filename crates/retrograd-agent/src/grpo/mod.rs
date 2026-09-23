//! Agentic GRPO: the run builder and the loop around it.
//!
//! One update is in [`updates`]; what leaves an update before it reaches the
//! optimizer - truncated members, groups their environment already scored, the
//! cap on the union - is in [`selection`], and the `agent/*` counters in
//! [`metrics`].

mod evaluate;
mod metrics;
mod observe;
mod selection;
mod updates;

#[cfg(test)]
mod fixtures;

use std::sync::Arc;

use retrograd_core::{TrainConfig, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_observe::TrajectoryObserver;
use retrograd_training::Progress;

pub use self::evaluate::AgentEvalMetrics;

use self::updates::{UpdateLoop, run_updates};
use crate::env::{EnvironmentFactory, ToolProviderFactory};
use crate::judge::RewardBackend;
use crate::policy::{Policy, PolicyActor};
use crate::rollout::RolloutEngine;
use crate::tools::{HermesToolCallParser, ToolCallParser, ToolProvider};
use crate::{Error, Result};
use retrograd_agent_core::config::AgentGrpoConfig;
use retrograd_agent_core::scenario::Scenario;

/// What an update boundary offers the caller: the trainer, exclusively, plus
/// everything needed to decide what to do with it.
///
/// A boundary is the one quiet moment in an agentic run - the update's
/// rollouts, judging and optimizer step have completed, and the next update has
/// not started - which is why it is the only place a checkpoint may be written
/// or a run may be stopped.
pub struct UpdateBoundary<'a> {
    /// One-based index of the update that just completed, which is also the
    /// number of updates completed so far.
    pub update: u32,
    pub updates: u32,
    pub global_step: u64,
    /// Scenarios drawn since the start of the run: the resume cursor.
    pub scenarios_consumed: u64,
    pub trainer: &'a mut Trainer,
    /// Present only when [`UpdateHook::should_evaluate`] asked for a pass at
    /// this boundary and the run has evaluation scenarios.
    pub evaluation: Option<AgentEvalMetrics>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentFlow {
    Continue,
    Stop,
}

/// What the caller does between updates: evaluate on a schedule it owns,
/// checkpoint, and decide whether the run goes on.
///
/// The loop owns *running* an evaluation, because the rollout engine and the
/// environments are its own; the hook owns *when*, because the schedule,
/// patience and checkpointing are the frontend's.
pub trait UpdateHook {
    /// Called at each boundary before the pass would run. Cheap and
    /// side-effect-free: it is only a schedule.
    fn should_evaluate(&mut self, update: u32, updates: u32) -> bool {
        let _ = (update, updates);
        false
    }

    fn update_finished(&mut self, boundary: UpdateBoundary<'_>) -> Result<AgentFlow>;
}

/// The trainer is returned even when collection/scoring/training fails, so an
/// embedding (notably PyO3) can preserve model ownership and report the error.
pub struct AgentRunOutcome {
    pub trainer: Option<Trainer>,
    pub result: Result<TrainMetrics>,
}

pub struct AgenticGrpoServices<'p, 'h> {
    /// Hands out one environment per trajectory. Stateless tools go through
    /// [`ToolProviderFactory`]; `AgenticRun::with_tools` does that for you.
    pub environments: Option<Arc<dyn EnvironmentFactory>>,
    pub parser: Option<Arc<dyn ToolCallParser>>,
    /// `None` runs without a judge: only what the environment scored is
    /// trained on, and a group it left unscored is dropped under
    /// [`AgentGrpoConfig::judge_failure`].
    pub reward: Option<Arc<dyn RewardBackend>>,
    pub on_progress: &'p mut dyn FnMut(Progress),
    /// Held-out scenarios, run at the boundaries the hook asks for. Empty means
    /// no evaluation, whatever the hook says.
    pub evaluation: Vec<Scenario>,
    pub hook: Option<&'h mut dyn UpdateHook>,
    /// Updates already completed by a checkpoint this run resumes from. The
    /// scenario selection and every seed are derived from the update index
    /// alone, so restarting at `start_update` replays exactly the schedule the
    /// interrupted run would have had.
    pub start_update: u32,
    /// Receives the trajectories and the summary of every update.
    pub observer: Option<Arc<dyn TrajectoryObserver>>,
}

/// Builder for an agentic GRPO run.
///
/// The required pieces - a trainer, scenarios and the two configs - are
/// constructor arguments because a run is impossible without them. A judge is
/// not one of them: an environment that grades its own steps is a reward
/// source, and [`AgenticRun::with_judge`] is how the other one is added.
/// Everything optional is a method, so adding tools or swapping the tool-call
/// parser does not mean restating the rest:
///
/// ```ignore
/// let outcome = AgenticRun::new(trainer, scenarios, config, training)
///     .with_judge(judge)
///     .with_tools(tools)
///     .on_progress(&mut |progress| eprintln!("{}", progress.metrics.train_loss))
///     .run()
///     .await;
/// ```
pub struct AgenticRun<'a> {
    trainer: Trainer,
    scenarios: Vec<Scenario>,
    config: AgentGrpoConfig,
    training: TrainConfig,
    reward: Option<Arc<dyn RewardBackend>>,
    environments: Option<Arc<dyn EnvironmentFactory>>,
    parser: Option<Arc<dyn ToolCallParser>>,
    on_progress: Option<&'a mut dyn FnMut(Progress)>,
    evaluation: Vec<Scenario>,
    hook: Option<&'a mut dyn UpdateHook>,
    start_update: u32,
    observer: Option<Arc<dyn TrajectoryObserver>>,
}

impl<'a> AgenticRun<'a> {
    pub fn new(
        trainer: Trainer,
        scenarios: Vec<Scenario>,
        config: AgentGrpoConfig,
        training: TrainConfig,
    ) -> Self {
        Self {
            trainer,
            scenarios,
            config,
            training,
            reward: None,
            environments: None,
            parser: None,
            on_progress: None,
            evaluation: Vec::new(),
            hook: None,
            start_update: 0,
            observer: None,
        }
    }

    /// Attaches the judge that scores what the environment did not.
    ///
    /// Without one, a group no environment scored has no reward at all: it is
    /// dropped, or ends the run, according to
    /// [`AgentGrpoConfig::judge_failure`].
    pub fn with_judge(mut self, reward: Arc<dyn RewardBackend>) -> Self {
        self.reward = Some(reward);
        self
    }

    /// Attaches a stateless tool provider, shared by every trajectory. Use
    /// `tools::CompositeToolProvider` to combine MCP servers with in-process
    /// tools.
    pub fn with_tools(self, tools: Arc<dyn ToolProvider>) -> Self {
        self.with_environments(Arc::new(ToolProviderFactory::new(tools)))
    }

    /// Resolves embedders' stateless registered tools without modifying the
    /// built-in crate. Session-bound registry entries must be installed in the
    /// environment that owns their sandbox.
    pub fn with_tool_registry(
        self,
        registry: &crate::tools::ToolRegistry,
        names: &[String],
    ) -> Result<Self> {
        let mut shared = Vec::new();
        for (name, registered) in registry.resolve(names)? {
            match registered {
                crate::tools::RegisteredTool::Shared(tool) => shared.push(tool),
                crate::tools::RegisteredTool::Session(_) => {
                    return Err(crate::Error::invalid(format!(
                        "registered tool '{name}' is session-bound and must be installed in an environment"
                    )));
                }
            }
        }
        let provider = crate::tools::LocalToolProvider::new(shared)?;
        Ok(self.with_tools(Arc::new(provider)))
    }

    /// Attaches an environment factory: one instance per trajectory, reset
    /// before the first turn and closed at the end. Required as soon as the
    /// task has state - see [`crate::env`].
    pub fn with_environments(mut self, environments: Arc<dyn EnvironmentFactory>) -> Self {
        self.environments = Some(environments);
        self
    }

    /// Overrides the tool-call parser. Defaults to the Hermes/Qwen
    /// `<tool_call>` convention.
    pub fn with_parser(mut self, parser: Arc<dyn ToolCallParser>) -> Self {
        self.parser = Some(parser);
        self
    }

    pub fn on_progress(mut self, on_progress: &'a mut dyn FnMut(Progress)) -> Self {
        self.on_progress = Some(on_progress);
        self
    }

    /// Held-out scenarios for the evaluation passes the hook schedules. They
    /// are graded by their environment, not by the judge - see
    /// [`crate::grpo::AgentEvalMetrics`] for why.
    pub fn with_evaluation(mut self, scenarios: Vec<Scenario>) -> Self {
        self.evaluation = scenarios;
        self
    }

    /// Attaches the caller's between-updates policy: when to evaluate, whether
    /// to checkpoint, whether to stop.
    pub fn with_update_hook(mut self, hook: &'a mut dyn UpdateHook) -> Self {
        self.hook = Some(hook);
        self
    }

    /// Restarts after `completed` updates, as a resumed checkpoint asks. The
    /// trainer must already hold the restored optimizer state.
    pub fn starting_at(mut self, completed: u32) -> Self {
        self.start_update = completed;
        self
    }

    /// Exports every update's trajectories, selection, outcome and summary.
    pub fn with_trajectory_observer(mut self, observer: Arc<dyn TrajectoryObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub async fn run(self) -> AgentRunOutcome {
        let mut discard = |_: Progress| {};
        let Self {
            trainer,
            scenarios,
            config,
            training,
            reward,
            environments,
            parser,
            on_progress,
            evaluation,
            hook,
            start_update,
            observer,
        } = self;
        let on_progress = match on_progress {
            Some(callback) => callback,
            None => &mut discard,
        };
        run_agentic_grpo(
            trainer,
            scenarios,
            config,
            training,
            AgenticGrpoServices {
                environments,
                parser,
                reward,
                on_progress,
                evaluation,
                hook,
                start_update,
                observer,
            },
        )
        .await
    }
}

pub async fn run_agentic_grpo(
    trainer: Trainer,
    scenarios: Vec<Scenario>,
    config: AgentGrpoConfig,
    training: TrainConfig,
    services: AgenticGrpoServices<'_, '_>,
) -> AgentRunOutcome {
    let AgenticGrpoServices {
        environments,
        parser,
        reward,
        on_progress,
        evaluation,
        hook,
        start_update,
        observer,
    } = services;
    let validation = config.validate().and_then(|_| {
        if scenarios.is_empty() {
            return Err(Error::invalid("agent scenario list must not be empty"));
        }
        for scenario in scenarios.iter().chain(&evaluation) {
            scenario.validate()?;
        }
        if start_update > config.updates {
            return Err(Error::invalid(format!(
                "the checkpoint resumes after {start_update} updates but the configuration                  only asks for {}",
                config.updates
            )));
        }
        Ok(())
    });
    if let Err(error) = validation {
        return AgentRunOutcome {
            trainer: Some(trainer),
            result: Err(error),
        };
    }

    let sequence_capacity = training.effective_generation_concurrency().max(1) as usize;
    let actor = match PolicyActor::spawn_local(
        trainer,
        64,
        sequence_capacity,
        training.n_seq_max.max(1) as usize,
    ) {
        Ok(actor) => actor,
        Err(error) => {
            return AgentRunOutcome {
                trainer: None,
                result: Err(error),
            };
        }
    };
    let handle = actor.handle();
    let policy: Arc<dyn Policy> = Arc::new(handle.clone());
    let engine = match &environments {
        Some(environments) => RolloutEngine::with_environments(
            policy,
            environments.clone(),
            parser.unwrap_or_else(|| Arc::new(HermesToolCallParser)),
            config.limits,
        ),
        None => RolloutEngine::new(policy, config.limits),
    };
    let result = match engine {
        Ok(engine) => {
            // Everything the environments need for *these* scenarios - a
            // malformed task declaration, a missing image, an unreachable
            // daemon - is settled here. Discovering it on the 700th rollout is
            // a lost run, and it is the operator's own configuration either way.
            match prepare_environments(environments.as_ref(), &scenarios).await {
                Ok(()) => {
                    run_updates(UpdateLoop {
                        engine: &engine,
                        policy: &handle,
                        lender: actor.lender(),
                        scenarios: &scenarios,
                        evaluation: &evaluation,
                        config: &config,
                        training: &training,
                        reward,
                        environments: environments.as_ref(),
                        on_progress,
                        hook,
                        start_update,
                        observer,
                    })
                    .await
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    };
    if let Some(environments) = &environments {
        environments.shutdown().await;
    }
    let restored = actor.into_trainer().await;
    match restored {
        Ok(trainer) => AgentRunOutcome {
            trainer: Some(trainer),
            result,
        },
        Err(restore_error) => AgentRunOutcome {
            trainer: None,
            result: Err(restore_error),
        },
    }
}

async fn prepare_environments(
    environments: Option<&Arc<dyn EnvironmentFactory>>,
    scenarios: &[Scenario],
) -> Result<()> {
    match environments {
        Some(environments) => environments.prepare(scenarios).await,
        None => Ok(()),
    }
}
