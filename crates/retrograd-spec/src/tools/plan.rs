//! The declarative tool plan, and the canonical catalogue it resolves into.
//!
//! Both are *written*: a plan is a TOML table, a catalogue is a JSON artifact a
//! run publishes. Resolving one against a registry, an MCP server or a file on
//! disk is `retrograd-tools`' business, and stays there.

use std::collections::BTreeMap;
use std::path::PathBuf;

use retrograd_agent_core::{Error, Result, ToolSpec};
use retrograd_core::hex_lower;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::tools::{McpServerConfig, ToolsConfig};

/// A run's declared tools: the session tools of its sandbox environment, if it
/// has one, and the MCP servers shared by every trajectory (inline and read
/// from files). `retrograd-tools` resolves this against a live registry and
/// filesystem into a [`ToolCatalog`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolPlan {
    /// The `tools` table of a container or local environment.
    pub session: Option<ToolsConfig>,
    pub mcp_servers: Vec<McpServerConfig>,
    pub mcp_config_files: Vec<PathBuf>,
}

/// Where a catalogued tool was resolved from, and what decides its behaviour.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSource {
    /// A Rust implementation and the parameters it was built with.
    Builtin {
        factory: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        params: String,
    },
    /// A program run in the sandbox. `sha256` covers the argv, the script's
    /// contents, the protocol and the timeout: the catalog hash changes when
    /// the tool does, not only when its description does.
    Exec {
        sha256: String,
    },
    Mcp {
        server: String,
    },
}

/// One resolved tool, ready to be offered to a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub id: String,
    /// `None` for an MCP tool, which its server versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub spec: ToolSpec,
    pub source: ToolSource,
    /// Whether this tool carries state across calls within a session (a shell,
    /// a file editor) as opposed to being stateless and safely shared across
    /// group members (a read-only lookup). An MCP server is stateful unless it
    /// explicitly declared itself `stateless`.
    pub stateful: bool,
}

impl CatalogEntry {
    /// `id@version`, or the bare id of an MCP tool.
    pub fn reference(&self) -> String {
        match self.version {
            Some(version) => format!("{}@{version}", self.id),
            None => self.id.clone(),
        }
    }
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
    /// Each selectable toolset, as the references of its tools.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub toolsets: BTreeMap<String, Vec<String>>,
    /// The toolset of a scenario that names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_toolset: Option<String>,
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
        self.tools.sort_by(|a, b| {
            (&a.id, a.version, &a.spec.name).cmp(&(&b.id, b.version, &b.spec.name))
        });
        self.tools
            .dedup_by(|a, b| a.id == b.id && a.version == b.version && a.spec.name == b.spec.name);
        self.resources
            .sort_by(|a, b| (&a.server, &a.uri).cmp(&(&b.server, &b.uri)));
        self.warnings.sort();
        let view = serde_json::json!({
            "tools": self.tools,
            "toolsets": self.toolsets,
            "default_toolset": self.default_toolset,
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
