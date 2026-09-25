//! Running the `[agent]` algorithm: multi-turn GRPO against a judge and a
//! world.
//!
//! An agentic run gets the same model loading, adapter handling, metrics bus
//! and final save as every other algorithm - the frontend picks an algorithm,
//! not a binary.
//!
//! The run control plane is polled at **update boundaries only**, from the
//! update hook. That is the one place an agentic run offers what a poll needs:
//! the progress callback is infallible and holds no trainer, and everything
//! before the boundary is a live tokio task fanned out over environments that a
//! blocking pause would stall. The consequence to state plainly is the latency,
//! a pause or a cancel takes effect at the end of the update it arrives in, not
//! inside it, and one update is a full round of rollouts.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use retrograd_agent::{
    AgentFlow, AgenticRun, Environments, JudgeBackend, Result as AgentResult, Scenario,
    UpdateBoundary, UpdateHook,
};
// Only the `not(feature = "mcp")` half of `connect_tools` below calls
// `resolve_local`; the MCP half resolves the whole plan instead. Same `cfg` on
// the import, so the build that has the transport does not warn about it.
#[cfg(not(feature = "mcp"))]
use retrograd_agent::ToolPlanResolve;
use retrograd_config::AgentRunConfig;
use retrograd_core::{Error, Result, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_memory as memory;
use retrograd_metrics::{MetricEvent, MetricValue};
use retrograd_observe::{Algorithm as ObservedAlgorithm, ObserveSink};
use retrograd_training as training;

use crate::algorithm::{Context, LiveControls};
use crate::control::{AdHocEvaluation, ControlPoint};
use crate::controller::EvalDirection;
use crate::observe;
use crate::observer::{EvaluationReport, LoopPlan, RolloutEpoch};
use crate::signature::{checked_total_steps, prompts_dataset};

/// Outcome of an agentic run: the trainer comes back separately because the
/// policy actor owns it for the duration, and a failure inside the actor can
/// lose it - in which case the adapter cannot be saved and the caller has to
/// say so rather than panic.
pub(crate) struct AgentOutcome {
    pub trainer: Option<Trainer>,
    pub result: Result<TrainMetrics>,
}

pub(crate) fn run_agent_grpo(
    mut trainer: Trainer,
    agent: &AgentRunConfig,
    ctx: &mut Context<'_>,
) -> AgentOutcome {
    match prepare(&mut trainer, agent, ctx) {
        Ok(prepared) => drive(trainer, agent, prepared, ctx),
        Err(error) => AgentOutcome {
            trainer: Some(trainer),
            result: Err(error),
        },
    }
}

/// Everything settled before a token is generated: the scenarios, the held-out
/// set, the trajectory budget against the loaded context, and the checkpoint
/// this run resumes from.
struct Prepared {
    scenarios: Vec<Scenario>,
    evaluation: Vec<Scenario>,
    trajectory_limit: usize,
    /// Updates the restored checkpoint consumed, when there is one.
    resumed_from_update: Option<u32>,
}

fn prepare(
    trainer: &mut Trainer,
    agent: &AgentRunConfig,
    ctx: &mut Context<'_>,
) -> Result<Prepared> {
    let mut scenarios = read_scenarios(&agent.scenarios)?;
    append_system_suffix(&mut scenarios, &agent.system_suffix);
    // Configure variables before the cached tool-support probe renders the template.
    trainer.set_chat_template_variables(Some(&agent.template_variables_json()))?;
    let context_size = trainer.context_size()?;
    let trajectory_limit = agent.trajectory_limit(context_size)?;
    let evaluation = match &ctx.config.evaluation {
        Some(evaluation) => {
            let mut held_out = read_scenarios(&evaluation.data)?;
            // The same suffix as the training set, applied in the same place.
            append_system_suffix(&mut held_out, &agent.system_suffix);
            // An agentic evaluation costs one full trajectory per scenario,
            // containers, turns, tool calls - so the cap is the lever that
            // decides whether evaluating costs less than training.
            let held_out = evenly_spaced(held_out, evaluation.max_examples);
            check_gradable(agent.environment.as_ref(), &held_out, &evaluation.data)?;
            held_out
        }
        None => Vec::new(),
    };
    let config = &agent.config;
    let total_steps = checked_total_steps(
        "agent GRPO",
        &[
            config.updates as u64,
            config.epochs as u64,
            config.scenarios_per_update as u64,
            config.group_size as u64,
        ],
    )?;
    // The restore happens here, into a trainer the policy actor has not taken
    // yet: after that the optimizer state lives behind the actor's channel.
    let resume = ctx.controller.begin(
        trainer,
        prompts_dataset(&agent.scenarios, config.limits.max_new_tokens_per_turn)?,
        total_steps,
    )?;
    Ok(Prepared {
        scenarios,
        evaluation,
        trajectory_limit,
        resumed_from_update: resume.map(|boundary| boundary.completed_iterations as u32),
    })
}

/// Refuses a held-out set whose scenarios declare no way of being graded.
///
/// The alternative is discovering it at the first evaluation, which on an
/// agentic run is an hour of rollouts later.
///
/// **What "gradable" means depends on the world.** A sandbox - container or
/// local - grades an attempt by running the `verify` command of its
/// `metadata.env` task declaration, so a scenario without one has nothing to
/// score it and can be caught here, from the document alone. An HTTP
/// environment is the other case: the reward comes back from its own `/step`,
/// the scenario carries no verify command, and nothing readable at startup says
/// whether the server grades. Demanding a verify there would refuse a run that
/// is perfectly well configured, so the check is not made and a server that
/// grades nothing is caught by the evaluation pass itself, which refuses an
/// ungraded set rather than averaging over it.
fn check_gradable(
    environment: Option<&retrograd_agent::EnvironmentConfig>,
    scenarios: &[Scenario],
    path: &Path,
) -> Result<()> {
    match environment {
        Some(retrograd_agent::EnvironmentConfig::Http(_)) => return Ok(()),
        Some(_) => {}
        // Unreachable through the TOML loader, which refuses `[evaluation]`
        // without `[agent.environment]`; said again here because this function
        // is what the other frontends' configurations arrive at.
        None => {
            return Err(Error::config(format!(
                "{}: this run has no [agent.environment], so nothing grades a held-out \
                 trajectory. An agentic evaluation measures the environment's own reward; a \
                 judge cannot stand in, because its scores are relative inside a group and not \
                 comparable across updates",
                path.display()
            )));
        }
    }
    let graded = scenarios
        .iter()
        .filter(|scenario| {
            retrograd_agent::env::EnvTask::from_scenario(scenario)
                .is_ok_and(|task| task.verify.is_some())
        })
        .count();
    if graded == 0 {
        return Err(Error::config(format!(
            "{}: no evaluation scenario declares a metadata.env.verify command, so nothing \
             would grade the trajectories. An agentic evaluation measures the environment's \
             own reward; a judge cannot stand in, because its scores are relative inside a \
             group and not comparable across updates",
            path.display()
        )));
    }
    if graded < scenarios.len() {
        return Err(Error::config(format!(
            "{}: {} of {} evaluation scenarios declare no metadata.env.verify command. A \
             mean over a moving subset of the held-out set is not a measurement: either grade \
             them all or take the ungraded ones out",
            path.display(),
            scenarios.len() - graded,
            scenarios.len()
        )));
    }
    Ok(())
}

/// Keeps at most `limit` scenarios, spread across the file rather than taken
/// from the front: a held-out set is usually ordered by topic, and its first
/// `limit` entries are not a sample of it.
fn evenly_spaced(scenarios: Vec<Scenario>, limit: Option<usize>) -> Vec<Scenario> {
    let Some(limit) = limit.filter(|limit| *limit < scenarios.len()) else {
        return scenarios;
    };
    let total = scenarios.len();
    // One index per slot, strictly increasing because `limit < total`, so the
    // picks are spread and their count is exactly `limit`.
    let picked: BTreeSet<usize> = (0..limit).map(|slot| slot * total / limit).collect();
    scenarios
        .into_iter()
        .enumerate()
        .filter(|(index, _)| picked.contains(index))
        .map(|(_, scenario)| scenario)
        .collect()
}

/// `judge eval` - measures the configured judge against reference labels.
///
/// It reads the same `[agent.judge]` table a training run reads, and it loads
/// no model: measuring a judge must not require a GPU, otherwise nobody
/// measures it. The RULER cache is deliberately ignored - a second pass served
/// from cache would report perfect self-consistency for free.
pub fn evaluate_judge(agent: &AgentRunConfig, fixtures: &Path) -> Result<String> {
    let Some(mut judge) = agent.judge.clone() else {
        return Err(Error::config(
            "this configuration declares no [agent.judge]: there is nothing to measure. A run \
             graded by its environment alone is measured by [evaluation], not by `judge eval`",
        ));
    };
    if let retrograd_agent::JudgeConfig::Ruler { config } = &mut judge {
        config.cache_path = None;
    }
    let backend = judge
        .build(|path| path.to_path_buf())
        .map_err(Error::from)?;
    let fixtures = retrograd_agent::judge::parse_fixtures(&fs::read_to_string(fixtures)?)
        .map_err(Error::from)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::runtime(format!("judge eval needs a tokio runtime: {error}")))?;
    let report = runtime
        .block_on(retrograd_agent::judge::evaluate(backend, &fixtures))
        .map_err(Error::from)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| Error::runtime(format!("judge report is not serializable: {error}")))
}

fn drive(
    trainer: Trainer,
    agent: &AgentRunConfig,
    prepared: Prepared,
    ctx: &mut Context<'_>,
) -> AgentOutcome {
    let mut config = agent.config.clone();
    config.limits.max_trajectory_tokens = prepared.trajectory_limit;
    let updates = config.updates;
    let epochs_per_update = config.epochs;

    // The !Send trainer remains pinned to `LocalSet`, while Send judge HTTP
    // tasks use the worker pool. This lets judging of a completed group make
    // progress during the next group's synchronous GPU decode.
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return AgentOutcome {
                trainer: Some(trainer),
                result: Err(Error::runtime(format!(
                    "the agentic loop needs a tokio runtime: {error}"
                ))),
            };
        }
    };
    let local = tokio::task::LocalSet::new();

    // Paths inside the judge were already rebased by the loader, so identity is
    // the right resolver here. Built before the first token, when there is one:
    // a judge that cannot start is refused where it is declared rather than on
    // the first group it is handed.
    let reward = match agent
        .judge
        .as_ref()
        .map(|judge| judge.build(|path| path.to_path_buf()).map_err(Error::from))
        .transpose()
    {
        Ok(reward) => reward,
        Err(error) => {
            return AgentOutcome {
                trainer: Some(trainer),
                result: Err(error),
            };
        }
    };

    // The tools first, in one resolution: the toolsets the environment will
    // execute and the catalogue that describes them come from the same pass,
    // so the hash logged below is the hash of what the model is offered.
    let mut tools = match connect_tools(&local, &runtime, agent, ctx) {
        Ok(tools) => tools,
        Err(error) => {
            return AgentOutcome {
                trainer: Some(trainer),
                result: Err(error),
            };
        }
    };
    ctx.observer
        .info(&format!("tool catalog sha256 {}", tools.catalog_sha256()));

    // Connecting to a daemon, pulling an image and validating every task
    // declaration happens here - before a single token is generated.
    let mut environment = match &agent.environment {
        Some(config) => match local.block_on(&runtime, config.build(tools.toolsets.take())) {
            Ok(factory) => Some(factory),
            Err(error) => {
                tools.shutdown(&local, &runtime);
                return AgentOutcome {
                    trainer: Some(trainer),
                    result: Err(error.into()),
                };
            }
        },
        None => None,
    };
    // The digest, not the tag: a tag moves between a run and its resume, and the
    // resolved reference is what someone has to paste back into the
    // configuration to get the same environment.
    if let Some(image) = environment
        .as_ref()
        .and_then(|factory| factory.pinned_image())
    {
        ctx.observer.info(&format!("environment image {image}"));
    }
    // Said once, because it decides what every reward in the run is: without a
    // judge, a group its environment left unscored has no fallback and is
    // dropped. Reading it at startup is what turns a later "unscored" count
    // from a mystery into a consequence.
    if reward.is_none() {
        ctx.observer
            .info("no [agent.judge]: trajectories are graded by the environment alone");
    }

    // After the environment exists, so a forced interrupt has something to tear
    // down, and before the first rollout, so there is never a window where
    // containers are live and Ctrl+C still takes the default action.
    crate::interrupt::install(environment.clone());

    // The world is taken inside the branch, not while building a tuple: the
    // operands of an `if let` pattern are evaluated before the pattern is
    // tested, so a `take()` there empties the environment even in the common
    // case where there is no provider to compose it with - and the run would
    // then be built with neither a sandbox nor tools.
    if let Some(shared) = tools.provider()
        && let Some(world) = environment.take()
    {
        match local.block_on(
            &runtime,
            retrograd_agent::env::SharedToolsFactory::new(world, shared),
        ) {
            Ok(composed) => environment = Some(Arc::new(composed)),
            Err(error) => {
                return AgentOutcome {
                    trainer: Some(trainer),
                    result: Err(error.into()),
                };
            }
        }
    }

    let training = ctx.config.training.clone();
    let total_epochs = updates as u64 * epochs_per_update as u64;
    ctx.observer.loop_started(&LoopPlan::Rollout {
        algorithm: "agent_grpo",
        updates,
        epochs_per_update,
        total_epochs,
    });
    let sink = match observe::open(
        ctx.config,
        ObservedAlgorithm::AgentGrpo,
        prepared.resumed_from_update.map(u64::from),
        [
            ("group_size", config.group_size as u64),
            ("scenarios_per_update", config.scenarios_per_update as u64),
            (
                "max_new_tokens",
                u64::from(config.limits.max_new_tokens_per_turn),
            ),
            ("updates", u64::from(updates)),
            ("epochs", u64::from(epochs_per_update)),
        ],
    ) {
        Ok(sink) => sink,
        Err(error) => {
            tools.shutdown(&local, &runtime);
            ctx.observer.loop_finished();
            return AgentOutcome {
                trainer: Some(trainer),
                result: Err(error),
            };
        }
    };

    // The progress callback and the boundary hook both need the observer, the
    // bus and the controller, and both are handed to the run at once. A
    // `RefCell` is what lets them share: they are called from the same thread
    // and never while the other is active - a progress event comes from the
    // optimizer step, a boundary from between updates.
    let trajectory_observer = sink.as_ref().map(ObserveSink::observer);
    let shared = RefCell::new(Reporter {
        ctx,
        sink,
        position: Position::default(),
        bus_error: None,
        updates,
        epochs_per_update,
    });
    let mut report = |progress: retrograd_training::Progress| shared.borrow_mut().report(progress);
    let mut hook = BoundaryHook {
        shared: &shared,
        updates,
    };

    let mut run = AgenticRun::new(trainer, prepared.scenarios, config, training)
        .on_progress(&mut report)
        .with_update_hook(&mut hook)
        .with_evaluation(prepared.evaluation)
        .starting_at(prepared.resumed_from_update.unwrap_or(0));
    if let Some(observer) = trajectory_observer {
        run = run.with_trajectory_observer(observer);
    }
    if let Some(reward) = reward {
        run = run.with_judge(reward);
    }
    if environment.is_none()
        && let Some(tools) = tools.provider()
    {
        run = run.with_tools(tools);
    }
    if let Some(environment) = environment {
        run = run.with_environments(environment);
    }
    let outcome = local.block_on(&runtime, run.run());
    tools.shutdown(&local, &runtime);
    let Reporter {
        ctx,
        sink,
        bus_error,
        ..
    } = shared.into_inner();
    observe::close(sink, ctx.observer);
    ctx.observer.loop_finished();

    let result = outcome.result.map_err(Error::from);
    AgentOutcome {
        trainer: outcome.trainer,
        // A metrics sink that failed mid-run is a failed run: the alternative is
        // a TensorBoard directory that silently stops halfway.
        result: match (result, bus_error) {
            (Ok(metrics), None) => Ok(metrics),
            (Ok(_), Some(error)) => Err(error),
            (Err(error), _) => Err(error),
        },
    }
}

/// The half of the run that reports: one row per optimizer epoch, one metrics
/// event per epoch and per evaluation.
struct Reporter<'a, 'b> {
    ctx: &'a mut Context<'b>,
    /// Held here so the progress callback can report it; closed by `drive`.
    sink: Option<ObserveSink>,
    position: Position,
    bus_error: Option<Error>,
    updates: u32,
    epochs_per_update: u32,
}

impl Reporter<'_, '_> {
    fn report(&mut self, progress: retrograd_training::Progress) {
        let Context {
            controller,
            memory,
            observer,
            ..
        } = &mut *self.ctx;
        let metric = |name: &str| {
            progress
                .values
                .iter()
                .find(|value| value.name == name)
                .map(|value| value.value)
                .unwrap_or(f32::NAN)
        };
        controller.note_step(progress.metrics.global_step);
        let (update, policy_epoch, absolute_epoch) = self
            .position
            .observe(progress.metrics.epoch, self.epochs_per_update);
        observer.rollout_epoch(&RolloutEpoch {
            update,
            updates: self.updates,
            policy_epoch,
            epochs_per_update: self.epochs_per_update,
            absolute_epoch,
            global_step: progress.metrics.global_step,
            train_loss: progress.metrics.train_loss,
            // The reward the update actually trained on, which is the batch's
            // own `reward/mean` - the same name and the same quantity the
            // synchronous rollout driver shows. Not `judge/score_mean`: a group
            // its environment scored never reaches the judge, so on a run graded
            // by verify commands that column would sit at zero.
            reward: metric("reward/mean"),
            kl: metric("policy/kl"),
            clip_fraction: metric("policy/clip_fraction"),
            learning_rate: progress.metrics.learning_rate,
            tokens_per_second: progress.metrics.tokens_per_second,
        });
        let mut values: Vec<MetricValue> = progress.values;
        let (memory_snapshot, memory_note) = memory.observe();
        if let Some(snapshot) = memory_snapshot {
            values.extend(memory::metric_values(snapshot));
        }
        values.extend(memory.device_metric_values());
        if let Some(note) = memory_note {
            observer.memory_note(&note);
        }
        if let Some(sink) = &self.sink {
            observe::report(sink, &mut **observer, &mut values);
        }
        self.emit(progress.metrics.epoch, progress.metrics.global_step, values);
    }

    fn emit(&mut self, epoch: u32, global_step: u64, values: Vec<MetricValue>) {
        if let Err(error) = self.ctx.bus.emit(&MetricEvent::Step {
            epoch,
            global_step,
            values,
        }) {
            self.bus_error.get_or_insert(error);
        }
    }
}

/// The half that steers: evaluate on the controller's schedule, checkpoint at
/// the boundary, stop when patience runs out.
struct BoundaryHook<'a, 'b, 'c> {
    shared: &'a RefCell<Reporter<'b, 'c>>,
    updates: u32,
}

impl UpdateHook for BoundaryHook<'_, '_, '_> {
    fn should_evaluate(&mut self, update: u32, updates: u32) -> bool {
        let mut shared = self.shared.borrow_mut();
        if !shared.ctx.controller.should_evaluate(update, updates) {
            return false;
        }
        // Said here rather than after the fact: an agentic evaluation is a
        // round of full trajectories, and the operator should see why the run
        // went quiet while it runs.
        shared
            .ctx
            .observer
            .evaluation_started(update as u64, updates);
        true
    }

    fn update_finished(&mut self, boundary: UpdateBoundary<'_>) -> AgentResult<AgentFlow> {
        let mut shared = self.shared.borrow_mut();
        let shared = &mut *shared;
        let update = boundary.update;
        // Control is consulted first and acted on last, as in every other
        // algorithm: consulted first so a `request_checkpoint` or a cadence
        // change lands before the `checkpoint_boundary` below acts on it, acted
        // on last so a stop still lets this boundary record its evaluation and
        // write its checkpoint. A pause does sit *after* this update's
        // evaluation - the pass runs inside the async loop, before the hook is
        // reached - but the checkpoint that records it is written on the way out
        // of this same call, so waiting here loses nothing.
        //
        // The one control an agentic run cannot serve is the out-of-schedule
        // evaluation: running one needs the rollout engine and the environments,
        // which live inside the async loop that called this hook and cannot be
        // re-entered from it. It says so instead of returning a number from
        // somewhere else.
        let mut ad_hoc = |_: &mut Trainer| -> Result<AdHocEvaluation> {
            Err(Error::invalid(
                "an agentic run cannot evaluate on demand: a held-out pass is a round of full \
                 trajectories through the rollout engine, which is busy running this update. \
                 Set [evaluation].every_iterations to have it run on a schedule",
            ))
        };
        let flow = shared.ctx.control.poll(
            &mut LiveControls {
                trainer: &mut *boundary.trainer,
                controller: &mut *shared.ctx.controller,
                evaluate: &mut ad_hoc,
            },
            ControlPoint {
                iteration: update,
                global_step: boundary.global_step,
                // An update boundary is the only point this hook is called at,
                // and the only one an agentic run can resume from.
                at_boundary: true,
            },
        )?;
        let state = training::Boundary {
            completed_iterations: update as u64,
            cursor: boundary.scenarios_consumed,
            // No adaptive KL in the agentic loop: its coefficient is the one
            // the configuration states, so there is no state to carry.
            kl_multiplier: None,
        };
        let mut keep_training = true;
        if let Some(measured) = boundary.evaluation {
            let outcome = shared.ctx.controller.record_evaluation(
                boundary.trainer,
                measured.mean_reward as f64,
                EvalDirection::Higher,
                update < self.updates,
                Some(state),
                boundary.global_step,
            )?;
            keep_training = outcome.keep_training;
            shared.ctx.observer.evaluation(&EvaluationReport::Rollout {
                update: update as u64,
                updates: self.updates,
                mean_reward: measured.mean_reward,
                reward_min: measured.reward_min,
                reward_max: measured.reward_max,
                examples: measured.scenarios,
                outcome,
            });
            let values = vec![
                MetricValue {
                    name: "eval/mean_reward".into(),
                    value: measured.mean_reward,
                },
                MetricValue {
                    name: "eval/reward_min".into(),
                    value: measured.reward_min,
                },
                MetricValue {
                    name: "eval/reward_max".into(),
                    value: measured.reward_max,
                },
                MetricValue {
                    name: "eval/examples".into(),
                    value: measured.scenarios as f32,
                },
                MetricValue {
                    name: "timing/eval_seconds".into(),
                    value: measured.elapsed_seconds,
                },
            ];
            shared.emit(update, boundary.global_step, values);
        }
        // After the evaluation, which is what updates best/stale/patience: a
        // scheduled checkpoint written before it would carry the previous
        // state.
        let observer = &mut *shared.ctx.observer;
        shared.ctx.controller.checkpoint_boundary(
            boundary.trainer,
            state,
            boundary.global_step,
            &mut |path| observer.checkpoint_written(path),
        )?;
        // Read last, next to the control plane's own verdict, and for the same
        // reason: an interrupt stops the loop but never costs this update the
        // evaluation and the checkpoint it just earned above.
        let interrupted = retrograd_agent::interrupt::stop_requested();
        Ok(if keep_training && flow.is_continue() && !interrupted {
            AgentFlow::Continue
        } else {
            AgentFlow::Stop
        })
    }
}

/// Appends `[agent].system_suffix` to every scenario's system turn.
fn append_system_suffix(scenarios: &mut [Scenario], suffix: &str) {
    if suffix.is_empty() {
        return;
    }
    for scenario in scenarios {
        scenario.system = Some(match scenario.system.take() {
            Some(system) if !system.is_empty() => format!("{system} {suffix}"),
            _ => suffix.to_string(),
        });
    }
}

fn read_scenarios(path: &Path) -> Result<Vec<Scenario>> {
    retrograd_scenario_gen::verify_manifest(path).map_err(Error::from)?;
    let source = fs::read_to_string(path)?;
    let mut scenarios = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            return Err(Error::dataset(
                path.display().to_string(),
                line_number,
                "empty scenario",
            ));
        }
        let scenario: Scenario = serde_json::from_str(line).map_err(|error| {
            Error::dataset(
                path.display().to_string(),
                line_number,
                format!("invalid scenario JSON: {error}"),
            )
        })?;
        scenario.validate().map_err(|error| {
            Error::dataset(path.display().to_string(), line_number, error.to_string())
        })?;
        scenarios.push(scenario);
    }
    if scenarios.is_empty() {
        return Err(Error::dataset(
            path.display().to_string(),
            0,
            "file contains no scenarios",
        ));
    }
    Ok(scenarios)
}

/// The run's resolved tools: the toolsets its environment executes, the MCP
/// providers it connected to, and how to close them.
///
/// Without the feature there is no provider, so the call sites read the same
/// in both builds and the "not compiled in" sentence is said once.
#[cfg(feature = "mcp")]
struct ConnectedTools {
    toolsets: Option<retrograd_agent::tools::Toolsets>,
    provider: Option<Arc<retrograd_agent::tools::McpToolProvider>>,
    catalog_sha256: String,
}

#[cfg(not(feature = "mcp"))]
struct ConnectedTools {
    toolsets: Option<retrograd_agent::tools::Toolsets>,
    catalog_sha256: String,
}

#[cfg(feature = "mcp")]
impl ConnectedTools {
    fn provider(&self) -> Option<Arc<dyn retrograd_agent::tools::ToolProvider>> {
        self.provider
            .as_ref()
            .map(|provider| provider.clone() as Arc<dyn retrograd_agent::tools::ToolProvider>)
    }

    fn shutdown(self, local: &tokio::task::LocalSet, runtime: &tokio::runtime::Runtime) {
        if let Some(provider) = self.provider {
            local.block_on(runtime, provider.shutdown());
        }
    }

    fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }
}

#[cfg(not(feature = "mcp"))]
impl ConnectedTools {
    fn provider(&self) -> Option<Arc<dyn retrograd_agent::tools::ToolProvider>> {
        None
    }

    fn shutdown(self, _local: &tokio::task::LocalSet, _runtime: &tokio::runtime::Runtime) {}

    fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }
}

#[cfg(feature = "mcp")]
fn connect_tools(
    local: &tokio::task::LocalSet,
    runtime: &tokio::runtime::Runtime,
    agent: &AgentRunConfig,
    ctx: &mut Context<'_>,
) -> Result<ConnectedTools> {
    let resolved = local
        .block_on(
            runtime,
            retrograd_agent::tools::resolve(
                &agent.tool_plan,
                &retrograd_agent::tools::ToolRegistry::builtin(),
            ),
        )
        .map_err(Error::from)?;
    // Here rather than at parse time: this is the first point that has the
    // merged view of the inline declarations and every `mcp_config` file, and
    // reading those files is a property of the machine that runs the document,
    // not of the document.
    if agent.environment.is_some()
        && let Some(server) = resolved.servers.iter().find(|server| !server.stateless)
    {
        return Err(Error::config(format!(
            "MCP server '{}' must set stateless = true before it can be shared with an environment; otherwise group members may contaminate each other",
            server.name
        )));
    }
    for warning in &resolved.catalog.warnings {
        ctx.observer.info(warning);
    }
    Ok(ConnectedTools {
        toolsets: resolved.toolsets,
        provider: resolved.provider,
        catalog_sha256: resolved.catalog.sha256,
    })
}

#[cfg(not(feature = "mcp"))]
fn connect_tools(
    _local: &tokio::task::LocalSet,
    _runtime: &tokio::runtime::Runtime,
    agent: &AgentRunConfig,
    _ctx: &mut Context<'_>,
) -> Result<ConnectedTools> {
    if !agent.tool_plan.mcp_servers.is_empty() || !agent.tool_plan.mcp_config_files.is_empty() {
        return Err(Error::config(
            "agent.mcp_servers needs MCP support, which is not compiled into this binary: \
             rebuild with `--features mcp`",
        ));
    }
    let local = agent
        .tool_plan
        .resolve_local(&retrograd_agent::tools::ToolRegistry::builtin())
        .map_err(Error::from)?;
    Ok(ConnectedTools {
        toolsets: local.toolsets,
        catalog_sha256: local.catalog.sha256,
    })
}

/// One-based (update, epoch-within-update, absolute epoch) from the absolute
/// update index the agentic loop reports. Same shape as the synchronous
/// rollout driver's, kept separate because that one is fed by a different
/// event type.
#[derive(Default)]
struct Position {
    current_update: Option<u64>,
    policy_epoch: u64,
}

impl Position {
    fn observe(&mut self, absolute_update: u32, epochs_per_update: u32) -> (u64, u64, u64) {
        let update = absolute_update as u64;
        if self.current_update == Some(update) {
            self.policy_epoch += 1;
        } else {
            self.current_update = Some(update);
            self.policy_epoch = 1;
        }
        let absolute = update.saturating_sub(1) * epochs_per_update as u64 + self.policy_epoch;
        (update, self.policy_epoch, absolute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenarios(count: usize) -> Vec<Scenario> {
        (0..count)
            .map(|index| Scenario {
                id: index.to_string(),
                system: None,
                user: "hello".into(),
                metadata: Default::default(),
            })
            .collect()
    }

    #[test]
    fn the_system_suffix_joins_every_scenario_and_creates_a_turn_when_there_is_none() {
        let mut set = scenarios(3);
        set[0].system = Some("Play well.".into());
        set[1].system = Some(String::new());
        append_system_suffix(&mut set, "Answer with one tool call.");
        assert_eq!(
            set[0].system.as_deref(),
            Some("Play well. Answer with one tool call.")
        );
        // No system turn, and an empty one, both become the suffix alone rather
        // than a leading separator.
        assert_eq!(set[1].system.as_deref(), Some("Answer with one tool call."));
        assert_eq!(set[2].system.as_deref(), Some("Answer with one tool call."));
        // The task itself is untouched: the suffix is a standing instruction,
        // not part of the question.
        assert_eq!(set[0].user, "hello");
    }

    #[test]
    fn an_empty_system_suffix_leaves_every_scenario_exactly_as_written() {
        let mut set = scenarios(2);
        set[0].system = Some("Play well.".into());
        append_system_suffix(&mut set, "");
        assert_eq!(set[0].system.as_deref(), Some("Play well."));
        // Still `None`, not `Some("")`: appending nothing must not invent a
        // system turn the scenario never had.
        assert_eq!(set[1].system, None);
    }

    fn ids(scenarios: &[Scenario]) -> Vec<&str> {
        scenarios
            .iter()
            .map(|scenario| scenario.id.as_str())
            .collect()
    }

    /// What grades a trajectory depends on the world, so the startup check has
    /// to as well: a sandbox is graded by the scenario's own `verify` command,
    /// an HTTP environment by what its `/step` returns. Demanding a verify of
    /// the second would refuse a run that is configured correctly.
    #[test]
    fn the_gradability_check_follows_the_environment() {
        // None of these declares a `metadata.env.verify`.
        let held_out = scenarios(2);
        let path = Path::new("held-out.jsonl");
        let environment = |value: serde_json::Value| -> retrograd_agent::EnvironmentConfig {
            serde_json::from_value(value).expect("a well-formed environment declaration")
        };

        let http = environment(serde_json::json!({
            "type": "http", "base_url": "http://127.0.0.1:8099",
            "request_timeout_secs": 120, "pool_size": 8, "max_result_bytes": 65536
        }));
        check_gradable(Some(&http), &held_out, path).expect("the server grades its own steps");

        let local = environment(serde_json::json!({
            "type": "local", "tools": {"default": "python"}, "allow_unsandboxed": true,
            "setup_timeout_secs": 300, "verify_timeout_secs": null
        }));
        let error = check_gradable(Some(&local), &held_out, path)
            .expect_err("a sandbox grades by verify, and there is none");
        assert!(matches!(&error, Error::Config(_)));
        let error = error.to_string();
        assert!(error.contains("metadata.env.verify"), "{error}");

        let error = check_gradable(None, &held_out, path).expect_err("no world grades nothing");
        assert!(matches!(&error, Error::Config(_)));
        let error = error.to_string();
        assert!(error.contains("[agent.environment]"), "{error}");
    }

    /// The cap decides what an evaluation costs, so it has to keep exactly the
    /// asked-for count and spread it: a held-out file is usually ordered by
    /// topic, and its first N entries are not a sample of it.
    #[test]
    fn the_evaluation_cap_keeps_a_spread_sample_of_the_right_size() {
        assert_eq!(
            ids(&evenly_spaced(scenarios(10), Some(4))),
            ["0", "2", "5", "7"]
        );
        assert_eq!(ids(&evenly_spaced(scenarios(3), Some(1))), ["0"]);
        // A cap at or above the set keeps the set, and no cap keeps it too.
        assert_eq!(evenly_spaced(scenarios(3), Some(3)).len(), 3);
        assert_eq!(evenly_spaced(scenarios(3), Some(9)).len(), 3);
        assert_eq!(evenly_spaced(scenarios(3), None).len(), 3);
        // Single-element set, cap of 1: the only index in range is 0.
        assert_eq!(evenly_spaced(scenarios(1), Some(1)).len(), 1);
    }
}
