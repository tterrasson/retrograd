//! Resolving a declarative tool plan into a canonical, reproducible catalogue.
//!
//! The plan and the catalogue are declarations and live in `retrograd-spec`;
//! what is here is the resolution - the registry lookup, the MCP handshake, the
//! files read from disk.

#[cfg(test)]
use crate::builtin::{Profile, ProfileTools};
#[cfg(feature = "mcp")]
use retrograd_agent_core::Error;
use retrograd_agent_core::Result;
#[cfg(test)]
use retrograd_agent_core::ToolSpec;

use crate::{McpServerConfig, RegisteredTool, ToolRegistry};
#[cfg(feature = "mcp")]
use crate::{McpToolProvider, ToolProvider};
use retrograd_spec::tools::{CatalogEntry, ToolCatalog, ToolPlan, ToolSource};
#[cfg(feature = "mcp")]
use std::sync::Arc;

/// What a tool plan can do once there is a registry and a filesystem.
///
/// The plan itself is a declaration and lives in `retrograd-spec`: resolving it
/// reads the built-in registry and the `mcp_config` files on disk, which is this
/// crate's job and not a parser's.
pub trait ToolPlanResolve {
    fn local_catalog(&self, registry: &ToolRegistry) -> Result<ToolCatalog>;
    fn merged_servers(&self) -> Result<(Vec<McpServerConfig>, Vec<String>)>;
}

impl ToolPlanResolve for ToolPlan {
    fn local_catalog(&self, registry: &ToolRegistry) -> Result<ToolCatalog> {
        let names = self
            .builtin
            .clone()
            .unwrap_or_else(|| self.profile.tool_names());
        let mut tools = Vec::new();
        for (_registry_name, registered) in registry.resolve(&names)? {
            let (spec, stateful) = match registered {
                RegisteredTool::Session(tool) => (tool.spec(), true),
                RegisteredTool::Shared(tool) => (tool.spec(), false),
            };
            if !self.filter.exposes(&spec.name) {
                continue;
            }
            tools.push(CatalogEntry {
                spec,
                source: ToolSource::Builtin,
                stateful,
            });
        }
        ToolCatalog {
            tools,
            ..Default::default()
        }
        .canonicalize()
    }

    fn merged_servers(&self) -> Result<(Vec<McpServerConfig>, Vec<String>)> {
        crate::load_mcp_config_files(&self.mcp_config_files, &self.mcp_servers)
    }
}

/// A [`ToolPlan`] fully resolved: built-in tools looked up, MCP servers
/// connected.
#[cfg(feature = "mcp")]
pub struct ResolvedToolPlan {
    pub catalog: ToolCatalog,
    /// `None` when the plan configured no MCP servers.
    pub provider: Option<Arc<McpToolProvider>>,
    pub servers: Vec<McpServerConfig>,
}

/// Resolves `plan` into its catalog and, if it names any, a connected
/// [`McpToolProvider`].
///
/// When every MCP server the plan lists is optional and all of them fail to
/// connect, this falls back to the registered tools alone rather than
/// failing outright - the run proceeds without MCP tools, with the fallback
/// recorded as a warning on the catalog.
#[cfg(feature = "mcp")]
pub async fn resolve(plan: &ToolPlan, registry: &ToolRegistry) -> Result<ResolvedToolPlan> {
    let mut catalog = plan.local_catalog(registry)?;
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
                    "all optional MCP servers were unavailable; continuing with registered tools: {error}"
                ));
                catalog.warnings.append(&mut warnings);
                catalog = catalog.canonicalize()?;
                return Ok(ResolvedToolPlan {
                    catalog,
                    provider: None,
                    servers,
                });
            }
            Err(error) => return Err(error),
        };
        warnings.extend(provider.warnings().iter().cloned());
        for spec in provider.list_tools().await? {
            if !plan.filter.exposes(&spec.name) {
                continue;
            }
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

    fn entry(name: &str) -> CatalogEntry {
        CatalogEntry {
            spec: ToolSpec {
                name: name.into(),
                description: format!("description for {name}"),
                input_schema: serde_json::json!({"type": "object"}),
            },
            source: ToolSource::Builtin,
            stateful: true,
        }
    }

    #[test]
    fn canonical_hash_ignores_discovery_order() {
        let first = ToolCatalog {
            tools: vec![entry("write_file"), entry("read_file")],
            ..Default::default()
        }
        .canonicalize()
        .unwrap();
        let second = ToolCatalog {
            tools: vec![entry("read_file"), entry("write_file")],
            ..Default::default()
        }
        .canonicalize()
        .unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.tools[0].spec.name, "read_file");
    }

    #[test]
    fn profile_resolution_is_the_same_catalog_as_the_legacy_profile() {
        for profile in [Profile::Python, Profile::Typescript] {
            let plan = ToolPlan {
                profile,
                ..Default::default()
            };
            let catalog = plan.local_catalog(&ToolRegistry::builtin()).unwrap();
            let mut expected = profile
                .tools()
                .into_iter()
                .map(|tool| tool.spec().name)
                .collect::<Vec<_>>();
            expected.sort();
            let actual = catalog
                .tools
                .into_iter()
                .map(|entry| entry.spec.name)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}
