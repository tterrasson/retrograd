use std::path::PathBuf;

use retrograd::config::{self, Algorithm};
use retrograd::{Error, Result};
use retrograd_tools::{ToolCatalog, ToolPlanResolve, ToolRegistry};

pub(crate) const FLAGS: &[&str] = &["--force", "--dry-run"];

pub(crate) fn scenarios(args: Vec<String>) -> Result<()> {
    let mut args = args.into_iter();
    if args.next().as_deref() != Some("generate") {
        return Err(Error::invalid(
            "usage: retrograd scenarios generate CONFIG.toml [--force] [--dry-run]",
        ));
    }
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::invalid("scenarios generate requires CONFIG.toml"))?,
    );
    let mut force = false;
    let mut dry_run = false;
    for argument in args {
        if !FLAGS.contains(&argument.as_str()) {
            return Err(Error::invalid(format!(
                "unknown scenarios generate argument '{argument}'"
            )));
        }
        match argument.as_str() {
            "--force" => force = true,
            "--dry-run" => dry_run = true,
            other => {
                return Err(Error::invalid(format!(
                    "unknown scenarios generate argument '{other}'"
                )));
            }
        }
    }
    let run = config::load(path)?;
    let Algorithm::AgentGrpo(agent) = &run.algorithm else {
        return Err(Error::config(
            "scenarios generate needs run.algorithm = 'agent_grpo'",
        ));
    };
    let generation = agent
        .scenario_generation
        .as_ref()
        .ok_or_else(|| Error::config("scenarios generate needs [agent.scenario_generation]"))?;
    let manifest = retrograd_scenario_gen::manifest_path(&agent.scenarios);
    if !force && !dry_run && (agent.scenarios.exists() || manifest.exists()) {
        return Err(Error::invalid(format!(
            "refusing to overwrite {} or its manifest without --force",
            agent.scenarios.display()
        )));
    }
    let catalog = resolve_catalog(agent)?;
    let rendered = retrograd_scenario_gen::render_prompt(
        &catalog,
        generation,
        generation.batch_size.min(generation.count),
        &[],
    )
    .map_err(Error::from)?;
    if dry_run {
        println!("{}", crate::tools::render_catalog(&catalog, false)?);
        println!(
            "prompt_version: {}\nprompt_sha256: {}\ncatalog_bytes: {}/{}",
            retrograd_scenario_gen::PROMPT_VERSION,
            rendered.sha256,
            rendered.catalog_bytes,
            generation.max_catalog_bytes
        );
        return Ok(());
    }
    let corpus = retrograd_scenario_gen::generate_scenarios_blocking(&catalog, generation)
        .map_err(Error::from)?;
    retrograd_scenario_gen::write_corpus(&corpus, &agent.scenarios, force).map_err(Error::from)?;
    println!(
        "generated {} scenarios in {} (catalog {})",
        corpus.scenarios.len(),
        agent.scenarios.display(),
        catalog.sha256
    );
    Ok(())
}

fn resolve_catalog(agent: &retrograd::config::AgentRunConfig) -> Result<ToolCatalog> {
    let registry = ToolRegistry::builtin();
    // Both forms require MCP discovery: a plan using only `mcp_config` files
    // still exposes remote tools and cannot use the local-only catalogue.
    if agent.tool_plan.mcp_servers.is_empty() && agent.tool_plan.mcp_config_files.is_empty() {
        return agent
            .tool_plan
            .local_catalog(&registry)
            .map_err(Error::from);
    }
    #[cfg(feature = "mcp")]
    {
        let resolved =
            retrograd_tools::resolve_blocking(&agent.tool_plan, &registry).map_err(Error::from)?;
        if let Some(provider) = resolved.provider {
            retrograd_tools::shutdown_blocking(provider).map_err(Error::from)?;
        }
        Ok(resolved.catalog)
    }
    #[cfg(not(feature = "mcp"))]
    {
        Err(Error::config(
            "scenario generation with MCP servers needs MCP support, which is not compiled into this binary: rebuild with `--features mcp`",
        ))
    }
}
