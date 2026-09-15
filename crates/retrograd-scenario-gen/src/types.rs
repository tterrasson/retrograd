use std::collections::BTreeMap;

use retrograd_agent_core::Scenario;
use serde::{Deserialize, Serialize};

/// One scenario as the model returned it, before it is turned into a
/// [`Scenario`] and given an id.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDraft {
    pub task: String,
    pub difficulty: u8,
    pub expected_tools: Vec<String>,
    pub expected_resources: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// A generated corpus and the manifest describing how it was produced.
#[derive(Clone, Debug)]
pub struct GeneratedCorpus {
    pub scenarios: Vec<Scenario>,
    pub manifest: GenerationManifest,
}

/// Why a draft never made it into the corpus, counted by cause so a run's
/// rejection rate can be diagnosed rather than just noted.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RejectionCounts {
    pub duplicate: usize,
    pub near_duplicate: usize,
    pub unknown_tool: usize,
    pub unknown_resource: usize,
    pub invalid: usize,
}

/// What the tool catalogue looked like when the corpus was generated, recorded
/// so a manifest explains its own scenarios without the catalogue beside it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogSummary {
    pub tools: usize,
    pub session_tools: usize,
    pub mcp_servers: Vec<String>,
    /// MCP servers whose tools declared themselves stateless and may therefore
    /// be shared with an environment. A subset of `mcp_servers`, never all of
    /// it: a server not listed here may still hold state per session.
    pub stateless_shared: Vec<String>,
    pub resources: usize,
    pub warnings: Vec<String>,
}

/// Everything a generated corpus needs to explain and reproduce itself:
/// what asked for it, what it was asked of, and what came back.
///
/// Written beside the corpus by [`crate::write_corpus`] and checked against it
/// by [`crate::verify_manifest`], so a corpus and its manifest can never
/// silently drift apart.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationManifest {
    pub schema_version: u32,
    pub generator_model: String,
    pub generator_base_url: String,
    pub prompt_version: String,
    pub prompt_sha256: String,
    pub catalog_sha256: String,
    pub catalog: CatalogSummary,
    pub config_sha256: String,
    pub corpus_sha256: String,
    pub requested: usize,
    pub accepted: usize,
    pub rejected: RejectionCounts,
    pub difficulty_distribution: BTreeMap<String, usize>,
    /// How many accepted scenarios cited each tool in the catalogue, by name.
    /// A tool absent here, or present at zero, was never exercised by this
    /// corpus.
    pub tool_coverage: BTreeMap<String, usize>,
    pub near_duplicate_threshold: f32,
}
