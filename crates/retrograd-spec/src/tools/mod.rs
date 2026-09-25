//! What a configuration file can say about tools: their definitions, the
//! toolsets that select them, the MCP servers, and the plan that names them.

pub mod definition;
pub mod filter;
pub mod mcp;
pub mod plan;

pub use definition::{
    BuiltinImpl, ExecImpl, ExecProtocol, ExecReply, ToolDefinition, ToolDefinitions, ToolImpl,
    ToolRef, ToolsConfig, ToolsetDefinition,
};
pub use filter::glob_match;
pub use mcp::{McpServerConfig, McpTransport};
pub use plan::{CatalogEntry, ResourceSpec, ToolCatalog, ToolPlan, ToolSource};
