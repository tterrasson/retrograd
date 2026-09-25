//! Resolving a declarative tool plan into what a run executes *and* the
//! catalogue that describes it - in one pass, so the two cannot disagree.
//!
//! The plan and the catalogue are declarations and live in `retrograd-spec`;
//! what is here is the resolution - the registry lookup, the definition files
//! and scripts read from disk, the MCP handshake.

use std::collections::BTreeMap;
#[cfg(feature = "mcp")]
use std::sync::Arc;

use retrograd_agent_core::{Error, Result};

use crate::{McpServerConfig, RegisteredTool, ToolRegistry, ToolSet, ToolsConfig, Toolsets};
#[cfg(feature = "mcp")]
use crate::{McpToolProvider, ToolProvider};
#[cfg(feature = "mcp")]
use retrograd_spec::tools::{CatalogEntry, ToolSource};
use retrograd_spec::tools::{ToolCatalog, ToolPlan};

/// The session tools of a sandbox environment, resolved: the toolsets it hands
/// trajectories and the catalogue of everything they contain.
pub struct SessionTools {
    pub toolsets: Toolsets,
    pub catalog: ToolCatalog,
}

/// Resolves an environment's `tools` table: its definition files and inline
/// definitions over `registry`, then every selectable toolset. A tool shared
/// by two toolsets is built once.
pub fn resolve_session(config: &ToolsConfig, registry: &ToolRegistry) -> Result<SessionTools> {
    let registry = registry.with_config(config)?;
    let default = config.default_toolset()?;
    let mut built = BTreeMap::new();
    let mut sets = BTreeMap::new();
    let mut members = BTreeMap::new();
    for name in config.selectable()? {
        let mut tools = Vec::new();
        let mut references = Vec::new();
        for definition in registry.toolset(name)? {
            let key = definition.key();
            if !built.contains_key(&key) {
                built.insert(key.clone(), registry.build(&key)?);
            }
            let tool = &built[&key];
            match &tool.tool {
                RegisteredTool::Session(session) => tools.push(session.clone()),
                RegisteredTool::Shared(_) => {
                    return Err(Error::invalid(format!(
                        "toolset '{name}': tool '{key}' is shared, not bound to a sandbox; \
                         a shared tool is handed to the run, not to an environment"
                    )));
                }
            }
            references.push((tool.entry.spec.name.clone(), key.to_string()));
        }
        references.sort();
        members.insert(
            name.to_owned(),
            references
                .into_iter()
                .map(|(_, reference)| reference)
                .collect(),
        );
        sets.insert(name.to_owned(), ToolSet::new(tools)?);
    }
    let catalog = ToolCatalog {
        tools: built.into_values().map(|tool| tool.entry).collect(),
        toolsets: members,
        default_toolset: Some(default.to_owned()),
        ..Default::default()
    }
    .canonicalize()?;
    Ok(SessionTools {
        toolsets: Toolsets::new(default, sets)?,
        catalog,
    })
}

/// What a tool plan can do once there is a registry and a filesystem.
///
/// The plan itself is a declaration and lives in `retrograd-spec`: resolving it
/// reads definition files and `mcp_config` files on disk, which is this crate's
/// job and not a parser's.
pub trait ToolPlanResolve {
    /// Everything but the MCP servers, which need a connection: the session
    /// toolsets, and a catalogue of them.
    fn resolve_local(&self, registry: &ToolRegistry) -> Result<LocalTools>;
    fn merged_servers(&self) -> Result<(Vec<McpServerConfig>, Vec<String>)>;
}

/// A [`ToolPlan`] resolved without connecting to anything.
pub struct LocalTools {
    pub catalog: ToolCatalog,
    /// `None` when the run has no sandbox environment.
    pub toolsets: Option<Toolsets>,
}

impl ToolPlanResolve for ToolPlan {
    fn resolve_local(&self, registry: &ToolRegistry) -> Result<LocalTools> {
        match &self.session {
            Some(config) => {
                let session = resolve_session(config, registry)?;
                Ok(LocalTools {
                    catalog: session.catalog,
                    toolsets: Some(session.toolsets),
                })
            }
            None => Ok(LocalTools {
                catalog: ToolCatalog::default().canonicalize()?,
                toolsets: None,
            }),
        }
    }

    fn merged_servers(&self) -> Result<(Vec<McpServerConfig>, Vec<String>)> {
        crate::load_mcp_config_files(&self.mcp_config_files, &self.mcp_servers)
    }
}

/// A [`ToolPlan`] fully resolved: toolsets built, MCP servers connected.
#[cfg(feature = "mcp")]
pub struct ResolvedToolPlan {
    pub catalog: ToolCatalog,
    /// `None` when the run has no sandbox environment.
    pub toolsets: Option<Toolsets>,
    /// `None` when the plan configured no MCP servers.
    pub provider: Option<Arc<McpToolProvider>>,
    pub servers: Vec<McpServerConfig>,
}

/// Resolves `plan` into its toolsets, its catalogue and, if it names any, a
/// connected [`McpToolProvider`].
///
/// When every MCP server the plan lists is optional and all of them fail to
/// connect, this falls back to the session tools alone rather than failing
/// outright - the run proceeds without MCP tools, with the fallback recorded as
/// a warning on the catalog.
#[cfg(feature = "mcp")]
pub async fn resolve(plan: &ToolPlan, registry: &ToolRegistry) -> Result<ResolvedToolPlan> {
    let LocalTools {
        mut catalog,
        toolsets,
    } = plan.resolve_local(registry)?;
    let (servers, mut warnings) = plan.merged_servers()?;
    let provider = if servers.is_empty() {
        None
    } else {
        let provider = match McpToolProvider::connect(servers.clone()).await {
            Ok(provider) => Arc::new(provider),
            Err(error)
                if !catalog.tools.is_empty() && servers.iter().all(|server| !server.required) =>
            {
                warnings.push(format!(
                    "all optional MCP servers were unavailable; continuing with session tools: {error}"
                ));
                catalog.warnings.append(&mut warnings);
                catalog = catalog.canonicalize()?;
                return Ok(ResolvedToolPlan {
                    catalog,
                    toolsets,
                    provider: None,
                    servers,
                });
            }
            Err(error) => return Err(error),
        };
        warnings.extend(provider.warnings().iter().cloned());
        for spec in provider.list_tools().await? {
            let server = spec
                .name
                .split_once("__")
                .map(|(server, _)| server)
                .unwrap_or_default()
                .to_owned();
            // A server carries session state unless it declared otherwise:
            // labelling every MCP tool stateless would tell the generator - and
            // the manifest - a property only `stateless = true` establishes.
            let stateful = !servers
                .iter()
                .any(|config| config.name == server && config.stateless);
            catalog.tools.push(CatalogEntry {
                id: spec.name.clone(),
                version: None,
                spec,
                source: ToolSource::Mcp { server },
                stateful,
            });
        }
        catalog.resources.extend_from_slice(provider.resources());
        Some(provider)
    };
    catalog.warnings.append(&mut warnings);
    catalog = catalog.canonicalize()?;
    Ok(ResolvedToolPlan {
        catalog,
        toolsets,
        provider,
        servers,
    })
}

/// [`resolve`] for a caller with no async runtime of its own, such as a CLI
/// command. Spins up a throwaway current-thread runtime for the duration of
/// the call.
#[cfg(feature = "mcp")]
pub fn resolve_blocking(plan: &ToolPlan, registry: &ToolRegistry) -> Result<ResolvedToolPlan> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::invalid(format!("tool discovery needs a runtime: {error}")))?;
    runtime.block_on(resolve(plan, registry))
}

/// [`McpToolProvider::shutdown`] for a caller with no async runtime of its
/// own; see [`resolve_blocking`].
#[cfg(feature = "mcp")]
pub fn shutdown_blocking(provider: Arc<McpToolProvider>) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::invalid(format!("tool shutdown needs a runtime: {error}")))?;
    runtime.block_on(provider.shutdown());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(source: &str) -> ToolsConfig {
        toml::from_str(source).unwrap()
    }

    #[test]
    fn a_builtin_toolset_resolves_into_a_toolset_and_its_catalog() {
        let session =
            resolve_session(&config("default = \"python\""), &ToolRegistry::builtin()).unwrap();
        let names = session
            .toolsets
            .select(None)
            .unwrap()
            .specs()
            .iter()
            .map(|spec| spec.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "bash",
                "edit_file",
                "grep",
                "list_dir",
                "python",
                "read_file",
                "run_tests",
                "submit",
                "write_file"
            ]
        );
        assert_eq!(session.catalog.default_toolset.as_deref(), Some("python"));
        assert!(session.catalog.toolsets["python"].contains(&"pytest@1".to_owned()));
        assert_eq!(session.catalog.tools.len(), names.len());
    }

    /// Two toolsets that differ by one version of one tool: the comparison
    /// versions exist for, in one run.
    #[test]
    fn scenario_toolsets_share_tools_and_differ_by_version() {
        let session = resolve_session(
            &config(
                r#"
                default = "python"
                scenario_toolsets = ["python-v2"]

                [[tool]]
                id = "bash"
                version = 2
                description = "Run one shell command and read its output."
                builtin = { factory = "shell" }

                [toolset.python-v2]
                include = ["python"]
                tools = ["bash@2"]
                "#,
            ),
            &ToolRegistry::builtin(),
        )
        .unwrap();
        let bash = |toolset: Option<&str>| {
            session
                .toolsets
                .select(toolset)
                .unwrap()
                .specs()
                .iter()
                .find(|spec| spec.name == "bash")
                .unwrap()
                .description
                .clone()
        };
        assert_ne!(bash(None), bash(Some("python-v2")));
        assert!(session.catalog.toolsets["python-v2"].contains(&"bash@2".to_owned()));
        // bash@1, bash@2 and the eight tools both toolsets share.
        assert_eq!(session.catalog.tools.len(), 10);
    }

    #[test]
    fn the_catalog_hash_follows_what_the_model_reads() {
        let hash = |description: &str| {
            resolve_session(
                &config(&format!(
                    "default = \"x\"\n[[tool]]\nid = \"t\"\nversion = 1\n\
                     description = {description:?}\nexec = {{ argv = [\"t\"] }}\n\
                     [toolset.x]\ntools = [\"t\"]\n"
                )),
                &ToolRegistry::builtin(),
            )
            .unwrap()
            .catalog
            .sha256
        };
        assert_eq!(hash("one"), hash("one"));
        assert_ne!(hash("one"), hash("two"));
    }

    #[test]
    fn a_plan_without_a_sandbox_has_no_toolsets() {
        let local = ToolPlan::default()
            .resolve_local(&ToolRegistry::builtin())
            .unwrap();
        assert!(local.toolsets.is_none() && local.catalog.tools.is_empty());
    }
}
