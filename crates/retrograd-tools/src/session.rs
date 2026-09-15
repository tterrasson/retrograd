//! Tools bound to one episode's sandbox.
//!
//! This is the extension point: implement [`SessionTool`], register it in a
//! [`ToolSet`], and the environment that owns the sandbox does the rest. A tool
//! never sees a container, a pool or a rollout - only the
//! [`Sandbox`] contract - so it is testable
//! against a local sandbox with no daemon in sight.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use retrograd_agent_core::{Error, Result, Sandbox, ToolCall, ToolResult, ToolSpec};

use crate::builtin::ProfileTools;

/// What one session tool produced.
///
/// It carries more than a [`ToolResult`] because a tool that acts on the world
/// is also the thing best placed to grade it: `reward` is how a `run_tests`
/// gives a verifiable signal, and `done` is how a `submit` ends the episode.
/// Both project straight onto `StepOutcome`, so the rollout engine needs no
/// knowledge of any of this.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    pub content: String,
    /// The *action* failed. An observation the policy reads, never a rollout
    /// failure - see [`SessionTool::call`].
    pub is_error: bool,
    pub reward: Option<f32>,
    pub done: bool,
}

impl ToolOutcome {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            reward: None,
            done: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            reward: None,
            done: false,
        }
    }

    pub fn with_reward(mut self, reward: f32) -> Self {
        self.reward = Some(reward);
        self
    }

    pub fn finished(mut self) -> Self {
        self.done = true;
        self
    }

    pub fn into_result(self, call_id: impl Into<String>) -> ToolResult {
        ToolResult {
            call_id: call_id.into(),
            content: self.content,
            is_error: self.is_error,
        }
    }
}

#[async_trait]
pub trait SessionTool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Runs the tool against this trajectory's sandbox.
    ///
    /// Returning `Err` says **the sandbox is unusable**, and it costs the
    /// trajectory. A tool that merely failed - bad arguments, a missing file, a
    /// command that exited non-zero - returns `Ok` with
    /// [`ToolOutcome::error`], because that is behaviour the policy has to read
    /// and learn from. Getting this backwards either trains on a world that
    /// stopped existing, or hides a broken run behind plausible-looking text.
    ///
    /// Whatever ends up in `content` is tokenized into the trajectory, so it
    /// must be a deterministic function of the arguments and the sandbox state
    /// - see the module docs of [`retrograd_agent_core::sandbox`].
    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome>;
}

/// A validated set of session tools, keyed by name.
#[derive(Clone, Default)]
pub struct ToolSet {
    tools: Vec<Arc<dyn SessionTool>>,
    specs: Vec<ToolSpec>,
}

impl ToolSet {
    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Dispatches a call. An unknown name is an observation, not a failure: the
    /// policy invented a tool and has to be told so, exactly like the MCP and
    /// local providers already do.
    pub async fn call(&self, sandbox: &dyn Sandbox, call: &ToolCall) -> Result<ToolOutcome> {
        let Some(index) = self.specs.iter().position(|spec| spec.name == call.name) else {
            return Ok(ToolOutcome::error(format!(
                "unknown tool '{}'; available tools: {}",
                call.name,
                self.specs
                    .iter()
                    .map(|spec| spec.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        self.tools[index]
            .call(sandbox, call.arguments.clone())
            .await
    }
}

/// Builds a [`ToolSet`], failing at construction rather than at the 300th
/// rollout: a name collision or a malformed schema discovered mid-run is a lost
/// run.
#[derive(Default)]
pub struct ToolSetBuilder {
    tools: Vec<Arc<dyn SessionTool>>,
    denied: Vec<String>,
}

impl ToolSetBuilder {
    /// Adds the built-in tools of a language profile. Nothing else about the
    /// profile is decided here - the image and the package cache are read by
    /// `retrograd-env`, from the same enum.
    pub fn with_profile(self, profile: crate::builtin::Profile) -> Self {
        self.extend(profile.tools())
    }

    pub fn with_registry(
        mut self,
        registry: &crate::ToolRegistry,
        names: &[String],
    ) -> Result<Self> {
        for (_, tool) in registry.resolve(names)? {
            match tool {
                crate::RegisteredTool::Session(tool) => self.tools.push(tool),
                crate::RegisteredTool::Shared(_) => {
                    return Err(Error::invalid(
                        "a shared registered tool cannot be installed in a session sandbox",
                    ));
                }
            }
        }
        Ok(self)
    }

    pub fn with(mut self, tool: Arc<dyn SessionTool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn extend<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn SessionTool>>,
    {
        self.tools.extend(tools);
        self
    }

    /// Hides tools by name pattern, `*` allowed as a wildcard - the same
    /// spelling `McpServerConfig` filters with, so an operator writes one rule
    /// shape whatever the tool source is.
    pub fn deny<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.denied.extend(patterns.into_iter().map(Into::into));
        self
    }

    pub fn build(self) -> Result<ToolSet> {
        let mut tools = Vec::new();
        let mut specs = Vec::new();
        let mut seen = HashSet::new();
        for tool in self.tools {
            let spec = tool.spec();
            if spec.name.trim().is_empty() {
                return Err(Error::invalid("a session tool name must not be empty"));
            }
            if !spec.input_schema.is_object() {
                return Err(Error::invalid(format!(
                    "session tool '{}' must declare a JSON object input schema",
                    spec.name
                )));
            }
            if self
                .denied
                .iter()
                .any(|pattern| crate::glob_match(pattern, &spec.name))
            {
                continue;
            }
            if !seen.insert(spec.name.clone()) {
                return Err(Error::invalid(format!(
                    "duplicate session tool name '{}'",
                    spec.name
                )));
            }
            specs.push(spec);
            tools.push(tool);
        }
        Ok(ToolSet { tools, specs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named(&'static str);

    #[async_trait]
    impl SessionTool for Named {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.0.into(),
                description: "test".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn call(
            &self,
            _sandbox: &dyn Sandbox,
            _arguments: serde_json::Value,
        ) -> Result<ToolOutcome> {
            Ok(ToolOutcome::ok(self.0))
        }
    }

    struct Unnamed;

    #[async_trait]
    impl SessionTool for Unnamed {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: " ".into(),
                description: String::new(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn call(
            &self,
            _sandbox: &dyn Sandbox,
            _arguments: serde_json::Value,
        ) -> Result<ToolOutcome> {
            Ok(ToolOutcome::ok(""))
        }
    }

    #[test]
    fn a_broken_set_is_refused_at_construction_not_mid_run() {
        assert!(
            ToolSet::builder()
                .with(Arc::new(Named("bash")))
                .with(Arc::new(Named("bash")))
                .build()
                .is_err()
        );
        assert!(ToolSet::builder().with(Arc::new(Unnamed)).build().is_err());
    }

    #[test]
    fn deny_patterns_use_the_same_wildcards_as_mcp_filters() {
        let set = ToolSet::builder()
            .with(Arc::new(Named("bash")))
            .with(Arc::new(Named("write_file")))
            .with(Arc::new(Named("read_file")))
            .deny(["write_*", "bash"])
            .build()
            .unwrap();
        assert_eq!(
            set.specs()
                .iter()
                .map(|spec| &spec.name)
                .collect::<Vec<_>>(),
            ["read_file"]
        );
    }

    #[test]
    fn an_outcome_carries_a_verdict_the_engine_can_use_as_is() {
        let outcome = ToolOutcome::ok("3 passed").with_reward(1.0).finished();
        assert_eq!(outcome.reward, Some(1.0));
        assert!(outcome.done);
        let result = outcome.into_result("call-1");
        assert_eq!(
            (result.call_id.as_str(), result.is_error),
            ("call-1", false)
        );
    }
}
