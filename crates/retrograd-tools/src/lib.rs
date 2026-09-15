//! Tool implementations, and the seam a user plugs their own into.
//!
//! Two shapes, and the difference is whether the tool has a world to act on:
//!
//! - [`ToolProvider`] is **stateless and shared** by every trajectory of every
//!   group. Right for a read-only MCP server, a calculator, a lookup into the
//!   training data.
//! - [`SessionTool`] is **bound to one episode**: it is handed the
//!   [`Sandbox`](retrograd_agent_core::Sandbox) of the trajectory that called
//!   it. Right for anything that writes - a shell, a file edit, a test run.
//!
//! Using the first where the second is needed is not a style question. Two
//! members of one group sharing a filesystem contaminate each other, and a
//! group-relative baseline over contaminated members measures nothing.
//!
//! The vocabulary both speak - [`ToolSpec`], [`ToolCall`], [`ToolResult`] -
//! lives in `retrograd-agent-core` and is re-exported here.

pub mod builtin;
pub mod catalog;
pub mod composite;
pub mod config_files;
pub mod local;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod parser;
pub mod registry;
pub mod session;

pub use builtin::{Profile, ProfileTools};
pub use catalog::ToolPlanResolve;
#[cfg(feature = "mcp")]
pub use catalog::{ResolvedToolPlan, resolve, resolve_blocking, shutdown_blocking};
pub use composite::CompositeToolProvider;
pub use config_files::load_mcp_config_files;
pub use local::{LocalTool, LocalToolProvider};
#[cfg(feature = "mcp")]
pub use mcp::McpToolProvider;
pub use parser::{HermesToolCallParser, ParsedAssistant, ToolCallParseError, ToolCallParser};
pub use registry::{RegisteredTool, ToolFactory, ToolRegistry};
pub use retrograd_agent_core::tools::{ToolCall, ToolProvider, ToolResult, ToolSpec};
pub use retrograd_spec::tools::{CatalogEntry, ResourceSpec, ToolCatalog, ToolPlan, ToolSource};
pub use retrograd_spec::tools::{McpServerConfig, McpTransport};
pub use retrograd_spec::tools::{ToolFilter, glob_match};
pub use session::{SessionTool, ToolOutcome, ToolSet, ToolSetBuilder};
