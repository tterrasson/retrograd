//! In-process tools.
//!
//! MCP is the right answer for anything that already speaks the protocol, but a
//! reward-shaping helper, a lookup into the training data, or a deterministic
//! calculator does not deserve a subprocess and a JSON-RPC handshake. A
//! [`LocalTool`] is a trait with two methods; [`LocalToolProvider`] turns a set
//! of them into a [`ToolProvider`] that composes with MCP through
//! [`super::CompositeToolProvider`].

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use super::{ToolCall, ToolProvider, ToolResult, ToolSpec};
use retrograd_agent_core::{Error, Result};

#[async_trait]
pub trait LocalTool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Runs the tool. Returning `Err` marks the observation as failed and hands
    /// the message to the model; it does not abort the rollout, because a tool
    /// that refuses a bad argument is behaviour the policy must learn from.
    async fn call(&self, arguments: serde_json::Value) -> Result<String>;
}

/// A [`ToolProvider`] over a fixed set of [`LocalTool`]s, validated once at
/// construction: no empty name, no name shared by two tools.
pub struct LocalToolProvider {
    tools: Vec<Arc<dyn LocalTool>>,
    specs: Vec<ToolSpec>,
}

impl LocalToolProvider {
    pub fn new(tools: Vec<Arc<dyn LocalTool>>) -> Result<Self> {
        let specs = tools.iter().map(|tool| tool.spec()).collect::<Vec<_>>();
        let mut seen = HashSet::new();
        for spec in &specs {
            if spec.name.is_empty() {
                return Err(Error::invalid("a local tool name must not be empty"));
            }
            if !seen.insert(spec.name.clone()) {
                return Err(Error::invalid(format!(
                    "duplicate local tool name '{}'",
                    spec.name
                )));
            }
        }
        Ok(Self { tools, specs })
    }
}

#[async_trait]
impl ToolProvider for LocalToolProvider {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(self.specs.clone())
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        let Some(index) = self.specs.iter().position(|spec| spec.name == call.name) else {
            return Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("unknown local tool '{}'", call.name),
                is_error: true,
            });
        };
        Ok(match self.tools[index].call(call.arguments.clone()).await {
            Ok(content) => ToolResult {
                call_id: call.id.clone(),
                content,
                is_error: false,
            },
            Err(error) => ToolResult {
                call_id: call.id.clone(),
                content: error.to_string(),
                is_error: true,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    #[async_trait]
    impl LocalTool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".into(),
                description: "Echo the text argument".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}}
                }),
            }
        }

        async fn call(&self, arguments: serde_json::Value) -> Result<String> {
            arguments
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| Error::Tool("echo needs a string 'text'".into()))
        }
    }

    #[tokio::test]
    async fn a_local_tool_answers_and_reports_its_own_errors_as_observations() {
        let provider = LocalToolProvider::new(vec![Arc::new(Echo)]).unwrap();
        assert_eq!(provider.list_tools().await.unwrap()[0].name, "echo");

        let result = provider
            .call(&ToolCall {
                id: "1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"text": "hi"}),
            })
            .await
            .unwrap();
        assert_eq!((result.content.as_str(), result.is_error), ("hi", false));

        // A tool error becomes a failed observation, never a rollout failure.
        let failed = provider
            .call(&ToolCall {
                id: "2".into(),
                name: "echo".into(),
                arguments: serde_json::json!({}),
            })
            .await
            .unwrap();
        assert!(failed.is_error && failed.content.contains("needs a string"));

        let unknown = provider
            .call(&ToolCall {
                id: "3".into(),
                name: "nope".into(),
                arguments: serde_json::json!({}),
            })
            .await
            .unwrap();
        assert!(unknown.is_error && unknown.content.contains("unknown local tool"));
    }

    #[test]
    fn duplicate_local_tool_names_are_rejected_at_construction() {
        let error = LocalToolProvider::new(vec![Arc::new(Echo), Arc::new(Echo)])
            .err()
            .expect("expected a failure");
        assert!(
            error.to_string().contains("duplicate local tool"),
            "{error}"
        );
    }
}
