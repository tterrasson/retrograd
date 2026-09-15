//! What a configuration file can say about tools: the preset, the filter, the
//! MCP servers, and the plan that names them.

pub mod filter;
pub mod mcp;
pub mod plan;
pub mod profile;

pub use filter::{ToolFilter, glob_match};
pub use mcp::{McpServerConfig, McpTransport};
pub use plan::{CatalogEntry, ResourceSpec, ToolCatalog, ToolPlan, ToolSource};
pub use profile::Profile;
