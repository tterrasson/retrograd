//! Running a collection: an `agent_grpo` configuration's model, world and
//! judge, rolled out without a single update, and the traces that succeeded
//! written as an SFT dataset.
//!
//! Only what a rollout reads is taken from the configuration - the model and
//! its `init_adapter`, `[agent]` (scenarios, environment, tools, suffix,
//! template variables, limits), the judge and the seed. The optimizer, the
//! update schedule and `[output]` are not read: nothing is trained and nothing
//! is saved but the dataset.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use std::time::Duration;

use retrograd_agent::collect::api::{ApiGenerator, collect_from_api};
use retrograd_agent::{
    AssistantForm, CollectConfig, CollectSink, CollectStats, EnvironmentConfig, Policy,
    PolicyActor, RolloutEngine, ScenarioReport, collect_trajectories,
};
use retrograd_config::{AgentRunConfig, Algorithm, CollectApiConfig, RunConfig};
use retrograd_core::{Error, Result};
use retrograd_dataset::ChatExample;
use retrograd_engine::Trainer;

use crate::RunObserver;
use crate::agent::{append_system_suffix, build_world, read_scenarios};

/// What `collect` is asked for beyond the configuration.
#[derive(Clone, Debug, Default)]
pub struct CollectOptions {
    /// The dataset to write.
    pub out: PathBuf,
    /// Attempts per scenario; the configuration's `group_size` when absent.
    pub k: Option<usize>,
    /// Traces kept per scenario.
    pub keep: usize,
    pub min_reward: Option<f32>,
    /// Keep only verified traces. Absent, it is on exactly when every scenario
    /// declares a `verify` command.
    pub require_verified: Option<bool>,
    /// Write assistant turns verbatim rather than structured.
    pub raw: bool,
    /// Collect from the first `limit` scenarios only.
    pub limit: Option<usize>,
    /// The configuration's `[agent] seed` when absent.
    pub seed: Option<u64>,
    /// Where to write the per-scenario report, as JSON.
    pub report: Option<PathBuf>,
    /// Overwrite `out` when it exists.
    pub force: bool,
    /// Generate with the `[agent.collect_api]` endpoint instead of a local model.
    pub api: bool,
}

/// What a collection did.
#[derive(Debug)]
pub struct CollectOutcome {
    pub stats: CollectStats,
    /// `None` when no trace was kept: nothing is written then, but the report
    /// still is.
    pub written: Option<PathBuf>,
}

pub fn collect(
    config: &RunConfig,
    options: &CollectOptions,
    observer: &mut dyn RunObserver,
) -> Result<CollectOutcome> {
    let Algorithm::AgentGrpo(agent) = &config.algorithm else {
        return Err(Error::config(
            "collect needs a configuration with run.algorithm = 'agent_grpo': it rolls out the \
             scenarios, environment and judge that section declares",
        ));
    };
    if options.out.exists() && !options.force {
        return Err(Error::invalid(format!(
            "refusing to overwrite {} without --force",
            options.out.display()
        )));
    }
    let api = match options.api {
        true => Some(agent.collect_api.as_ref().ok_or_else(|| {
            Error::config("collect --api needs an [agent.collect_api] section naming the endpoint")
        })?),
        false => None,
    };
    let partial = partial_path(&options.out);

    let mut scenarios = read_scenarios(&agent.scenarios)?;
    append_system_suffix(&mut scenarios, &agent.system_suffix);
    if let Some(limit) = options.limit {
        scenarios.truncate(limit);
    }
    let verified = scenarios.iter().all(|scenario| {
        retrograd_agent::env::EnvTask::from_scenario(scenario)
            .is_ok_and(|task| task.verify.is_some())
    });
    let collect_config = CollectConfig {
        k: options.k.unwrap_or(agent.config.group_size),
        keep: options.keep,
        min_reward: options.min_reward,
        require_verified: options.require_verified.unwrap_or(verified),
        form: match options.raw {
            true => AssistantForm::Raw,
            false => AssistantForm::Structured,
        },
        seed: options.seed.unwrap_or(agent.config.seed),
        judge_failure: agent.config.judge_failure,
        environment_grades: matches!(agent.environment, Some(EnvironmentConfig::Http(_))),
        generator: match api {
            Some(api) => api.model.clone(),
            None => config.model.display().to_string(),
        },
    };
    observer.info(&format!(
        "collecting {} scenarios, {} attempts each, keeping up to {} ({})",
        scenarios.len(),
        collect_config.k,
        collect_config.keep,
        collect_config.form.as_str()
    ));

    // The same runtime shape as a training run: the trainer pinned to a
    // `LocalSet`, HTTP on the worker pool.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|error| Error::runtime(format!("collect needs a tokio runtime: {error}")))?;
    let local = tokio::task::LocalSet::new();
    let collection = Collection {
        scenarios: &scenarios,
        config: &collect_config,
        partial: &partial,
        local: &local,
        runtime: &runtime,
    };
    let (stats, written) = match api {
        Some(api) => collection.with_api(agent, api, observer)?,
        None => collection.with_policy(config, agent, observer)?,
    };

    if let Some(report) = &options.report {
        write_report(report, &stats)?;
    }
    if stats.interrupted {
        return Err(Error::runtime(format!(
            "collection interrupted after {} of {} scenarios; the {written} records kept so far \
             are in {}",
            stats.scenarios.len(),
            scenarios.len(),
            partial.display()
        )));
    }
    if written == 0 {
        fs::remove_file(&partial)?;
        return Ok(CollectOutcome {
            stats,
            written: None,
        });
    }
    // What was written reads back as the dataset it claims to be before it
    // takes the name a training run will be pointed at.
    retrograd_dataset::read_chat_jsonl(&partial)?;
    fs::rename(&partial, &options.out)?;
    Ok(CollectOutcome {
        stats,
        written: Some(options.out.clone()),
    })
}

/// What both generators share: the scenarios, the settings, where the records
/// go, and the runtime they run on.
struct Collection<'a> {
    scenarios: &'a [retrograd_agent::Scenario],
    config: &'a CollectConfig,
    partial: &'a Path,
    local: &'a tokio::task::LocalSet,
    runtime: &'a tokio::runtime::Runtime,
}

impl Collection<'_> {
    /// Rollouts of the configuration's own model - or the `--model` override -
    /// through the rollout engine, graded by the environment and the judge.
    fn with_policy(
        &self,
        config: &RunConfig,
        agent: &AgentRunConfig,
        observer: &mut dyn RunObserver,
    ) -> Result<(CollectStats, usize)> {
        let started = Instant::now();
        observer.model_load_started();
        let mut trainer =
            Trainer::new(&config.model, config.training.clone()).inspect_err(|_| {
                observer.model_load_failed();
            })?;
        observer.model_load_finished(started.elapsed());
        if let Some(adapter) = config
            .lora
            .as_ref()
            .and_then(|lora| lora.init_adapter.as_ref())
        {
            observer.info(&format!("adapter: {}", adapter.display()));
            trainer.load_lora(adapter)?;
        }
        // Before anything renders the template: the tool-support probe is cached.
        trainer.set_chat_template_variables(Some(&agent.template_variables_json()))?;
        let mut limits = agent.config.limits;
        limits.max_trajectory_tokens = agent.trajectory_limit(trainer.context_size()?)?;

        let world = build_world(agent, observer, self.local, self.runtime)?;
        let mut sink = FileSink::create(self.partial, observer)?;
        let training = &config.training;
        let collected = self.local.block_on(self.runtime, async {
            let actor = PolicyActor::spawn_local(
                trainer,
                64,
                training.effective_generation_concurrency().max(1) as usize,
                training.n_seq_max.max(1) as usize,
            )?;
            let policy: Arc<dyn Policy> = Arc::new(actor.handle());
            let result = async {
                let engine = match &world.environments {
                    Some(environments) => RolloutEngine::with_environments(
                        policy,
                        environments.clone(),
                        Arc::new(retrograd_agent::tools::HermesToolCallParser),
                        limits,
                    ),
                    None => RolloutEngine::new(policy, limits),
                }?;
                if let Some(environments) = &world.environments {
                    environments.prepare(self.scenarios).await?;
                }
                collect_trajectories(
                    &engine,
                    world.judge.clone(),
                    self.scenarios,
                    self.config,
                    &mut sink,
                )
                .await
            }
            .await;
            if let Some(environments) = &world.environments {
                environments.shutdown().await;
            }
            // The trainer is only dropped: nothing was trained, so there is
            // nothing to save, and a failure to get it back changes no outcome.
            drop(actor.into_trainer().await);
            result
        });
        world.shutdown(self.local, self.runtime);
        let written = sink.finish()?;
        Ok((collected.map_err(Error::from)?, written))
    }

    /// Episodes of a remote model over the same environments. No model is
    /// loaded, and only the environment grades: a judge scores trajectory
    /// groups the API cannot produce.
    fn with_api(
        &self,
        agent: &AgentRunConfig,
        api: &CollectApiConfig,
        observer: &mut dyn RunObserver,
    ) -> Result<(CollectStats, usize)> {
        let key = std::env::var(&api.api_key_env).map_err(|_| {
            Error::config(format!(
                "collect --api: the API key environment variable '{}' is not set",
                api.api_key_env
            ))
        })?;
        let generator = ApiGenerator::connect(
            &api.base_url,
            key,
            Duration::from_secs(api.timeout_secs),
            api.model.clone(),
            api.temperature,
        )
        .map_err(Error::from)?;
        observer.info(&format!("generator: {} at {}", api.model, api.base_url));
        let world = build_world(agent, observer, self.local, self.runtime)?;
        if world.judge.is_some() {
            observer.info(
                "[agent.judge] is not used by collect --api: only the environment grades an API \
                 trace",
            );
        }
        let Some(environments) = world.environments.clone() else {
            world.shutdown(self.local, self.runtime);
            return Err(Error::config(
                "collect --api needs an [agent.environment] or MCP tools to act on",
            ));
        };
        let mut sink = FileSink::create(self.partial, observer)?;
        let collected = self.local.block_on(self.runtime, async {
            let result = async {
                environments.prepare(self.scenarios).await?;
                collect_from_api(
                    &generator,
                    environments.clone(),
                    self.scenarios,
                    agent.config.limits,
                    self.config,
                    &mut sink,
                )
                .await
            }
            .await;
            environments.shutdown().await;
            result
        });
        world.shutdown(self.local, self.runtime);
        let written = sink.finish()?;
        Ok((collected.map_err(Error::from)?, written))
    }
}

/// `out.jsonl` is written as `out.jsonl.tmp` and renamed once complete, so an
/// interrupted collection never leaves a file that looks finished.
fn partial_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

/// One JSONL line per record, and a progress line per scenario.
struct FileSink<'o> {
    file: BufWriter<fs::File>,
    written: usize,
    observer: &'o mut dyn RunObserver,
}

impl<'o> FileSink<'o> {
    fn create(path: &Path, observer: &'o mut dyn RunObserver) -> Result<Self> {
        Ok(Self {
            file: BufWriter::new(fs::File::create(path)?),
            written: 0,
            observer,
        })
    }

    fn finish(mut self) -> Result<usize> {
        self.file.flush()?;
        Ok(self.written)
    }
}

impl CollectSink for FileSink<'_> {
    fn record(&mut self, example: ChatExample) -> retrograd_agent::Result<()> {
        let line = serde_json::to_string(&example)
            .map_err(|error| Error::runtime(format!("serialize a collected record: {error}")))?;
        // Flushed per record: an interrupted collection keeps what it has.
        writeln!(self.file, "{line}")
            .and_then(|()| self.file.flush())
            .map_err(Error::from)?;
        self.written += 1;
        Ok(())
    }

    fn scenario_finished(&mut self, report: &ScenarioReport, index: usize, total: usize) {
        self.observer.info(&format!(
            "[{}/{total}] {}: {} of {} passed, {} kept",
            index + 1,
            report.id,
            report.passed,
            report.attempted,
            report.kept
        ));
    }
}

fn write_report(path: &Path, stats: &CollectStats) -> Result<()> {
    let report = serde_json::json!({
        "attempted": stats.attempted,
        "kept": stats.kept,
        "rejected": stats.rejected,
        "failures": {
            "tool": stats.failures.tool,
            "policy": stats.failures.policy,
            "other": stats.failures.other,
        },
        "unsolved": stats.unsolved().map(|report| &report.id).collect::<Vec<_>>(),
        "scenarios": stats.scenarios,
    });
    let text = serde_json::to_string_pretty(&report)
        .map_err(|error| Error::runtime(format!("the collection report: {error}")))?;
    fs::write(path, text + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_partial_file_sits_next_to_the_output() {
        assert_eq!(
            partial_path(Path::new("data/traces.jsonl")),
            Path::new("data/traces.jsonl.tmp")
        );
    }
}
