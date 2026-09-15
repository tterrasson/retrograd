use std::path::PathBuf;

use retrograd::config::{self, Algorithm};
use retrograd::{Error, Result};
use retrograd_tools::{ToolCatalog, ToolPlanResolve, ToolRegistry};

pub(crate) const FLAGS: &[&str] = &["--json", "--no-connect"];

pub(crate) fn tools(args: Vec<String>) -> Result<()> {
    let mut args = args.into_iter();
    if args.next().as_deref() != Some("list") {
        return Err(Error::invalid(
            "usage: retrograd tools list CONFIG.toml [--json] [--no-connect]",
        ));
    }
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::invalid("tools list requires CONFIG.toml"))?,
    );
    let mut json = false;
    let mut no_connect = false;
    for argument in args {
        if !FLAGS.contains(&argument.as_str()) {
            return Err(Error::invalid(format!(
                "unknown tools list argument '{argument}'"
            )));
        }
        match argument.as_str() {
            "--json" => json = true,
            "--no-connect" => no_connect = true,
            other => {
                return Err(Error::invalid(format!(
                    "unknown tools list argument '{other}'"
                )));
            }
        }
    }
    let run = config::load(path)?;
    let Algorithm::AgentGrpo(agent) = &run.algorithm else {
        return Err(Error::config(
            "tools list needs run.algorithm = 'agent_grpo'",
        ));
    };
    let registry = ToolRegistry::builtin();
    if no_connect {
        let mut catalog = agent
            .tool_plan
            .local_catalog(&registry)
            .map_err(Error::from)?;
        let (servers, warnings) = agent.tool_plan.merged_servers().map_err(Error::from)?;
        catalog.warnings.extend(warnings);
        catalog.warnings.extend(servers.into_iter().map(|server| {
            format!(
                "declared MCP server '{}' ({})",
                server.name,
                if server.stateless {
                    "stateless"
                } else {
                    "stateful"
                }
            )
        }));
        catalog = catalog.canonicalize().map_err(Error::from)?;
        println!("{}", render_catalog(&catalog, json)?);
        return Ok(());
    }
    #[cfg(feature = "mcp")]
    {
        let resolved =
            retrograd_tools::resolve_blocking(&agent.tool_plan, &registry).map_err(Error::from)?;
        println!("{}", render_catalog(&resolved.catalog, json)?);
        if let Some(provider) = resolved.provider {
            retrograd_tools::shutdown_blocking(provider).map_err(Error::from)?;
        }
        Ok(())
    }
    #[cfg(not(feature = "mcp"))]
    {
        Err(Error::config(
            "tools list with connection needs MCP support, which is not compiled into this binary: rebuild with `--features mcp` (or use --no-connect)",
        ))
    }
}

pub(crate) fn render_catalog(catalog: &ToolCatalog, json: bool) -> Result<String> {
    if json {
        return serde_json::to_string_pretty(catalog)
            .map_err(|error| Error::runtime(format!("serialize tool catalog: {error}")));
    }
    let mut output = format!(
        "tools: {}\nresources: {}\nsha256: {}\n",
        catalog.tools.len(),
        catalog.resources.len(),
        catalog.sha256
    );
    for entry in &catalog.tools {
        let description = entry.spec.description.chars().take(96).collect::<String>();
        output.push_str(&format!(
            "- {} [{}] {}\n",
            entry.spec.name,
            if entry.stateful { "session" } else { "shared" },
            description
        ));
    }
    for resource in &catalog.resources {
        output.push_str(&format!(
            "- resource {} {}\n",
            resource.server, resource.uri
        ));
    }
    for warning in &catalog.warnings {
        output.push_str(&format!("warning: {warning}\n"));
    }
    Ok(output)
}
