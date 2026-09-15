//! Composition of tool providers.
//!
//! `RolloutEngine` takes exactly one [`ToolProvider`].
//! [`CompositeToolProvider`] combines MCP servers with in-process tools: it merges
//! the tool lists at construction, refuses ambiguous names up front rather than
//! resolving them by accident at call time, and routes each call to the provider
//! that declared the tool.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::{ToolCall, ToolProvider, ToolResult, ToolSpec};
use retrograd_agent_core::{Error, Result};

/// A [`ToolProvider`] that routes each call to whichever of its underlying
/// providers declared the tool. See the module docs for why construction
/// resolves ambiguous names instead of the first call to hit one.
pub struct CompositeToolProvider {
    providers: Vec<Arc<dyn ToolProvider>>,
    specs: Vec<ToolSpec>,
    routes: HashMap<String, usize>,
}

impl CompositeToolProvider {
    /// Merges `providers`, listing each one's tools once. Listing happens here
    /// so that a rollout never pays for it: `list_tools` is called for every
    /// trajectory of every group, and an MCP round trip per call adds up.
    pub async fn connect(providers: Vec<Arc<dyn ToolProvider>>) -> Result<Self> {
        if providers.is_empty() {
            return Err(Error::invalid(
                "a composite tool provider needs at least one provider",
            ));
        }
        let mut specs = Vec::new();
        let mut routes = HashMap::new();
        for (index, provider) in providers.iter().enumerate() {
            for spec in provider.list_tools().await? {
                if routes.contains_key(&spec.name) {
                    return Err(Error::invalid(format!(
                        "tool name '{}' is provided more than once; rename it or filter it out",
                        spec.name
                    )));
                }
                routes.insert(spec.name.clone(), index);
                specs.push(spec);
            }
        }
        Ok(Self {
            providers,
            specs,
            routes,
        })
    }
}

#[async_trait]
impl ToolProvider for CompositeToolProvider {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(self.specs.clone())
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        match self.routes.get(&call.name) {
            Some(&index) => self.providers[index].call(call).await,
            None => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("unknown tool '{}'", call.name),
                is_error: true,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed {
        name: &'static str,
        answer: &'static str,
    }

    #[async_trait]
    impl ToolProvider for Fixed {
        async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
            Ok(vec![ToolSpec {
                name: self.name.into(),
                description: String::new(),
                input_schema: serde_json::json!({"type": "object"}),
            }])
        }

        async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
            Ok(ToolResult {
                call_id: call.id.clone(),
                content: self.answer.into(),
                is_error: false,
            })
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn calls_route_to_the_provider_that_declared_the_tool() {
        let composite = CompositeToolProvider::connect(vec![
            Arc::new(Fixed {
                name: "a",
                answer: "from-a",
            }),
            Arc::new(Fixed {
                name: "b",
                answer: "from-b",
            }),
        ])
        .await
        .unwrap();
        assert_eq!(composite.list_tools().await.unwrap().len(), 2);
        assert_eq!(composite.call(&call("a")).await.unwrap().content, "from-a");
        assert_eq!(composite.call(&call("b")).await.unwrap().content, "from-b");
        let unknown = composite.call(&call("c")).await.unwrap();
        assert!(unknown.is_error && unknown.content.contains("unknown tool"));
    }

    #[tokio::test]
    async fn an_ambiguous_tool_name_fails_at_construction_not_at_call_time() {
        let error = CompositeToolProvider::connect(vec![
            Arc::new(Fixed {
                name: "same",
                answer: "first",
            }),
            Arc::new(Fixed {
                name: "same",
                answer: "second",
            }),
        ])
        .await
        .err()
        .expect("expected a failure");
        assert!(error.to_string().contains("more than once"), "{error}");
        assert!(CompositeToolProvider::connect(vec![]).await.is_err());
    }
}
