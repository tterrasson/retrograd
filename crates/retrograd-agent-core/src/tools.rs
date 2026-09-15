//! What the policy may ask for, and what comes back.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// A failed *action*, not a broken world: this is an observation the policy
    /// reads and reacts to. Anything that makes the world itself unusable is an
    /// `Err`, never this.
    pub fn error(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            content: content.into(),
            is_error: true,
        }
    }
}

/// A stateless tool source, shared by every trajectory of every group.
///
/// That sharing is exactly right for read-only tools and exactly wrong as soon
/// as the task has state: two members of one group writing to the same world
/// contaminate each other, and the GRPO relative baseline stops measuring the
/// policy. Stateful work goes through [`crate::Environment`] instead.
#[async_trait]
pub trait ToolProvider: Send + Sync {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>>;
    async fn call(&self, call: &ToolCall) -> Result<ToolResult>;
}
