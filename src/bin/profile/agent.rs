//! Run `agent_grpo` directly so the profiler receives raw timing and scoring
//! progress instead of the condensed frontend observer events.
//!
//! This driver supports environment-only runs. Checkpoint resume, evaluation,
//! and explicit `[agent.tool_plan]` tools are intentionally unsupported here.

use std::fs;
use std::path::Path;
use std::time::Instant;

use retrograd::config::{Algorithm, RunConfig};
use retrograd::{Error, Result};
use retrograd_agent::{AgenticRun, Environments, JudgeBackend, Scenario, ToolPlanResolve};

use super::{Options, PhaseTotals, UpdateRow, VramTrack, apply_micro_batch_and_packing};
use crate::report::Workload;

pub(crate) fn run(options: Options, mut run_config: RunConfig, wall: Instant) -> Result<()> {
    let Algorithm::AgentGrpo(agent) = &mut run_config.algorithm else {
        unreachable!("agent::run is only called for an Algorithm::AgentGrpo config");
    };

    if !options.full {
        if let Some(value) = options.updates {
            agent.config.updates = value;
        }
        if let Some(value) = options.group_size {
            agent.config.group_size = value;
        }
        if let Some(value) = options.epochs {
            agent.config.epochs = value;
        }
        if let Some(value) = options.prompts_per_update {
            agent.config.scenarios_per_update = value;
        }
        if let Some(value) = options.max_new_tokens {
            agent.config.limits.max_new_tokens_per_turn = value;
        }
    }
    if !agent.tool_plan.mcp_servers.is_empty() || !agent.tool_plan.mcp_config_files.is_empty() {
        return Err(Error::invalid(
            "the profiler does not yet support agent_grpo's MCP servers; profile a config whose \
             tools come only from [agent.environment]",
        ));
    }
    let group_size = agent.config.group_size;
    let scenarios_per_update = agent.config.scenarios_per_update;
    apply_micro_batch_and_packing(&options, &mut run_config, group_size, scenarios_per_update)?;

    let Algorithm::AgentGrpo(agent) = &run_config.algorithm else {
        unreachable!("agent::run is only called for an Algorithm::AgentGrpo config");
    };
    let agent = agent.clone();

    let mut vram = VramTrack::new();
    let mem_start = super::snapshot_bytes();
    let (trainer, init) = super::init_trainer(&run_config, &mut vram)?;

    let scenarios = read_scenarios(&agent.scenarios)?;
    let context_size = trainer.context_size()?;
    let trajectory_limit = agent.trajectory_limit(context_size)?;
    let mut config = agent.config.clone();
    config.limits.max_trajectory_tokens = trajectory_limit;

    let reward = agent
        .judge
        .as_ref()
        .map(|judge| judge.build(|path| path.to_path_buf()))
        .transpose()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|error| {
            Error::runtime(format!("the agentic loop needs a tokio runtime: {error}"))
        })?;
    let local = tokio::task::LocalSet::new();

    // Resolve the toolsets and build the environment before generating the
    // first token, as the main runner does.
    let toolsets = agent
        .tool_plan
        .resolve_local(&retrograd_tools::ToolRegistry::builtin())?
        .toolsets;
    let environment = match &agent.environment {
        Some(environment_config) => {
            Some(local.block_on(&runtime, environment_config.build(toolsets))?)
        }
        None => None,
    };

    let training = run_config.training.clone();
    let mut phases = PhaseTotals::default();
    let mut updates: Vec<UpdateRow> = Vec::new();
    // The policy actor owns the trainer for the run, so memory is sampled after
    // the actor returns it.
    let mut on_progress = |progress: retrograd_training::Progress| {
        if let Some(row) = super::extract_update_row(&progress) {
            phases.add(&row);
            updates.push(row);
        }
    };

    let mut agentic_run =
        AgenticRun::new(trainer, scenarios, config.clone(), training).on_progress(&mut on_progress);
    if let Some(reward) = reward {
        agentic_run = agentic_run.with_judge(reward);
    }
    if let Some(environment) = environment {
        agentic_run = agentic_run.with_environments(environment);
    }

    let train_start = Instant::now();
    let outcome = local.block_on(&runtime, agentic_run.run());
    let train_time = train_start.elapsed();
    let mem_peak = super::peak_bytes();
    vram.sample();

    // A failed policy actor may not return a trainer from which to read memory.
    let mut trainer = outcome.trainer.ok_or_else(|| {
        Error::runtime("the policy actor failed before returning the trainer: nothing to report")
    })?;
    vram.fold_runtime(trainer.optimizer_memory()?);
    let metrics = outcome.result?;

    let workload = Workload {
        updates: config.updates,
        per_update: config.scenarios_per_update,
        per_update_label: "scenarios/upd",
        group_size: config.group_size,
        epochs_per_update: config.epochs,
        max_new_tokens: config.limits.max_new_tokens_per_turn,
    };
    super::print_report(
        &options,
        &run_config,
        &workload,
        &init,
        &mut trainer,
        &phases,
        &updates,
        &metrics,
        mem_start,
        mem_peak,
        &vram,
        train_time,
        wall,
    )
}

/// Read newline-delimited scenarios and verify their generation manifest.
fn read_scenarios(path: &Path) -> Result<Vec<Scenario>> {
    retrograd_scenario_gen::verify_manifest(path)?;
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
            "no scenarios",
        ));
    }
    Ok(scenarios)
}
