//! Generates agentic scenario corpora with an external LLM.
//!
//! [`generate_scenarios`] asks an OpenAI-compatible model for scenario drafts
//! against [`render_prompt`] and [`response_schema`], validates them and counts
//! what it rejected. [`write_corpus`] writes the corpus beside a manifest that
//! [`verify_manifest`] checks when the corpus is read back. The crate reaches
//! neither the engine nor an MCP client, which `scripts/test-fast-rust.sh`
//! checks on the dependency graph.

mod client;
mod export;
mod generate;
mod prompt;
mod types;
mod validate;

pub use export::{manifest_path, verify_manifest, write_corpus};
pub use generate::{generate_scenarios, generate_scenarios_blocking};
pub use prompt::{PROMPT_VERSION, render_prompt, response_schema};
pub use types::{GeneratedCorpus, GenerationManifest, RejectionCounts, ScenarioDraft};
