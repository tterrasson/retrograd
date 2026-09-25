//! Tools bound to one episode's sandbox.
//!
//! This is the Rust extension point: implement [`SessionTool`], register a
//! factory for it in the [`ToolRegistry`](crate::ToolRegistry), and a toolset
//! that names it hands it to the environment that owns the sandbox. A tool
//! never sees a container, a pool or a rollout - only the
//! [`Sandbox`] contract - so it is testable
//! against a local sandbox with no daemon in sight.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use retrograd_agent_core::{Error, Result, Sandbox, ToolCall, ToolResult, ToolSpec};

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

/// A validated set of session tools, ordered by name.
#[derive(Clone, Default)]
pub struct ToolSet {
    tools: Vec<Arc<dyn SessionTool>>,
    specs: Vec<ToolSpec>,
}

impl ToolSet {
    /// Fails at construction rather than at the 300th rollout: a name
    /// collision or a malformed schema discovered mid-run is a lost run. The
    /// order is the names', so the prompt does not depend on how a toolset was
    /// assembled.
    pub fn new(tools: Vec<Arc<dyn SessionTool>>) -> Result<Self> {
        let mut named = Vec::new();
        let mut seen = HashSet::new();
        for tool in tools {
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
            if !seen.insert(spec.name.clone()) {
                return Err(Error::invalid(format!(
                    "duplicate session tool name '{}'",
                    spec.name
                )));
            }
            named.push((spec, tool));
        }
        named.sort_by(|a, b| a.0.name.cmp(&b.0.name));
        let (specs, tools) = named.into_iter().unzip();
        Ok(Self { tools, specs })
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

/// The toolsets a sandbox environment may hand a trajectory: one default, and
/// the others a scenario may select by name.
///
/// Selection is per scenario, so every member of a group gets the same tools -
/// a group whose members saw different tools would compare different tasks.
#[derive(Clone)]
pub struct Toolsets {
    default: String,
    sets: BTreeMap<String, Arc<ToolSet>>,
}

impl Toolsets {
    /// One toolset under one name, which is also the default.
    pub fn single(name: impl Into<String>, tools: ToolSet) -> Result<Self> {
        let name = name.into();
        Self::new(name.clone(), BTreeMap::from([(name, tools)]))
    }

    pub fn new(default: impl Into<String>, sets: BTreeMap<String, ToolSet>) -> Result<Self> {
        let default = default.into();
        if !sets.contains_key(&default) {
            return Err(Error::invalid(format!(
                "default toolset '{default}' is not among the resolved toolsets"
            )));
        }
        if let Some((name, _)) = sets.iter().find(|(_, set)| set.is_empty()) {
            return Err(Error::invalid(format!(
                "toolset '{name}' gives the policy nothing to do"
            )));
        }
        Ok(Self {
            default,
            sets: sets
                .into_iter()
                .map(|(name, set)| (name, Arc::new(set)))
                .collect(),
        })
    }

    pub fn default_name(&self) -> &str {
        &self.default
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sets.keys().map(String::as_str)
    }

    /// The toolset a scenario asked for, or the default. Asking for one the
    /// environment did not declare selectable is a dataset error, refused
    /// before the first rollout.
    pub fn select(&self, requested: Option<&str>) -> Result<&Arc<ToolSet>> {
        let name = requested.unwrap_or(&self.default);
        self.sets.get(name).ok_or_else(|| {
            Error::invalid(format!(
                "toolset '{name}' is not selectable here; selectable: {}",
                self.sets.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
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
        assert!(ToolSet::new(vec![Arc::new(Named("bash")), Arc::new(Named("bash"))]).is_err());
        assert!(ToolSet::new(vec![Arc::new(Unnamed)]).is_err());
    }

    #[test]
    fn a_set_is_ordered_by_name_and_a_toolset_is_selected_by_name() {
        let set =
            ToolSet::new(vec![Arc::new(Named("write_file")), Arc::new(Named("bash"))]).unwrap();
        assert_eq!(
            set.specs()
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            ["bash", "write_file"]
        );
        let toolsets = Toolsets::new(
            "full",
            BTreeMap::from([
                ("full".to_owned(), set),
                (
                    "lite".to_owned(),
                    ToolSet::new(vec![Arc::new(Named("bash"))]).unwrap(),
                ),
            ]),
        )
        .unwrap();
        assert_eq!(toolsets.select(None).unwrap().specs().len(), 2);
        assert_eq!(toolsets.select(Some("lite")).unwrap().specs().len(), 1);
        let error = toolsets.select(Some("other")).err().unwrap().to_string();
        assert!(error.contains("selectable: full, lite"), "{error}");
        assert!(Toolsets::single("x", ToolSet::default()).is_err());
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
