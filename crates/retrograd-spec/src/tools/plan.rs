//! The declarative tool plan, and the canonical catalogue it resolves into.
//!
//! Both are *written*: a plan is a TOML table, a catalogue is a JSON artifact a
//! run publishes. Resolving one against a registry, an MCP server or a file on
//! disk is `retrograd-tools`' business, and stays there.

use std::path::PathBuf;

use retrograd_agent_core::{Error, Result, ToolSpec};
use retrograd_core::hex_lower;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::tools::{McpServerConfig, Profile, ToolFilter};

/// A run's declared tool selection: some built-in/registry names, a language
/// profile that fills in the rest, MCP servers (inline and read from files),
/// and a filter applied over their union. `retrograd-tools` resolves this
/// against a live registry and filesystem into a [`ToolCatalog`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolPlan {
    pub builtin: Option<Vec<String>>,
    pub profile: Profile,
    pub mcp_servers: Vec<McpServerConfig>,
    pub mcp_config_files: Vec<PathBuf>,
    pub filter: ToolFilter,
}

/// Where a catalogued tool was resolved from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSource {
    Builtin,
    Registry { name: String },
    Mcp { server: String },
}

/// One resolved tool, ready to be offered to a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub spec: ToolSpec,
    pub source: ToolSource,
    /// Whether this tool carries state across calls within a session (a shell,
    /// a file editor) as opposed to being stateless and safely shared across
    /// group members (a read-only lookup). An MCP server is stateful unless it
    /// explicitly declared itself `stateless`.
    pub stateful: bool,
}

/// One resource (not a tool) an MCP server advertises: a URI a model can read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSpec {
    pub server: String,
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// The resolved, canonical set of tools and resources a run exposes: what a
/// [`ToolPlan`] became once matched against a real registry, filesystem and set
/// of MCP servers. Published as a JSON artifact.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCatalog {
    pub tools: Vec<CatalogEntry>,
    pub resources: Vec<ResourceSpec>,
    pub warnings: Vec<String>,
    pub sha256: String,
}

impl ToolCatalog {
    /// Sorts tools and resources into a deterministic order and recomputes
    /// [`Self::sha256`] over that order, so two runs that declare the same
    /// tools produce byte-identical catalogs regardless of discovery order
    /// (registry lookup order, MCP handshake timing).
    pub fn canonicalize(mut self) -> Result<Self> {
        self.tools
            .sort_by(|a, b| (&a.source, &a.spec.name).cmp(&(&b.source, &b.spec.name)));
        self.resources
            .sort_by(|a, b| (&a.server, &a.uri).cmp(&(&b.server, &b.uri)));
        self.warnings.sort();
        let view = serde_json::json!({
            "tools": self.tools,
            "resources": self.resources,
        });
        let bytes = serde_json::to_vec(&view)
            .map_err(|error| Error::invalid(format!("serialize tool catalog: {error}")))?;
        self.sha256 = hex_lower(&Sha256::digest(bytes));
        Ok(self)
    }

    /// The specs of every catalogued tool, in catalog order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|entry| entry.spec.clone()).collect()
    }
}
