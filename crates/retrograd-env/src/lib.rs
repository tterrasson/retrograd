//! Environment implementations.
//!
//! The [`Environment`] / [`EnvironmentFactory`] contracts and [`StepOutcome`]
//! live in `retrograd-agent-core` and are re-exported here.
//! [`ToolProviderEnvironment`] wraps the stateless [`ToolProvider`] case so
//! every tool-only configuration keeps working unchanged.
//!
//! Three backings, in increasing order of what they can do and of what they
//! cost:
//!
//! - [`ToolProviderEnvironment`] - no world at all, just a shared tool source;
//! - [`HttpEnvironment`] - the world lives behind `reset`/`step`/`state`/`close`
//!   over HTTP, which is also how an OpenEnv server plugs in;
//! - a sandbox-backed environment, where the world is a container (see
//!   `retrograd-container`) or, deliberately, the host ([`LocalSandbox`]).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use retrograd_agent_core::Result;
use retrograd_agent_core::scenario::Scenario;
use retrograd_agent_core::tools::{ToolCall, ToolProvider, ToolResult, ToolSpec};

pub mod config;
#[cfg(feature = "container")]
pub mod container;
#[cfg(feature = "http-env")]
pub mod http;
#[cfg(feature = "local-sandbox")]
pub mod local;
pub mod sandbox;
pub mod task;

pub use config::{EnvironmentConfig, LocalConfig};
#[cfg(feature = "container")]
pub use container::{ContainerEnvironment, ContainerEnvironmentConfig, ContainerPoolConfig};
#[cfg(feature = "http-env")]
pub use http::{HttpEnvironment, HttpEnvironmentConfig, HttpEnvironmentFactory};
#[cfg(feature = "local-sandbox")]
pub use local::{LocalSandbox, LocalSandboxConfig, LocalSandboxProvider};
pub use retrograd_agent_core::env::{EnvState, Environment, EnvironmentFactory, StepOutcome};
pub use retrograd_agent_core::{Lease, Sandbox, SandboxProvider};
pub use sandbox::{SandboxEnvironment, SandboxEnvironmentConfig, SandboxEnvironmentFactory};
pub use task::{EnvTask, Verify, WORKDIR};

/// Adapts a stateless [`ToolProvider`] to [`Environment`]: `step` is `call`,
/// nothing is ever `done`, and no step carries a reward. This is what
/// `RolloutEngine::with_tools` builds, so every tool configuration written
/// before environments existed behaves exactly as it did.
pub struct ToolProviderEnvironment {
    tools: Arc<dyn ToolProvider>,
}

impl ToolProviderEnvironment {
    pub fn new(tools: Arc<dyn ToolProvider>) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl Environment for ToolProviderEnvironment {
    async fn reset(&mut self, _scenario: &Scenario, _seed: u64) -> Result<Option<String>> {
        Ok(None)
    }

    /// A stateless provider has no world to lose, so its errors are failed
    /// *actions* rather than a broken environment: they come back as `is_error`
    /// observations, which is what a tool call has always done here - the MCP
    /// provider already reports its own timeouts and unknown tools that way.
    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        Ok(StepOutcome::tool(match self.tools.call(call).await {
            Ok(result) => result,
            Err(error) => ToolResult {
                call_id: call.id.clone(),
                content: error.to_string(),
                is_error: true,
            },
        }))
    }

    async fn tools(&self) -> Result<Vec<ToolSpec>> {
        self.tools.list_tools().await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Hands out [`ToolProviderEnvironment`]s over one shared provider. Creating an
/// instance is free - the provider itself is what holds the connections.
pub struct ToolProviderFactory {
    tools: Arc<dyn ToolProvider>,
}

impl ToolProviderFactory {
    pub fn new(tools: Arc<dyn ToolProvider>) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl EnvironmentFactory for ToolProviderFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        Ok(Box::new(ToolProviderEnvironment::new(self.tools.clone())))
    }
}

/// Composes one session-bound world with explicitly stateless shared tools.
pub struct SharedToolsFactory {
    world: Arc<dyn EnvironmentFactory>,
    shared: Arc<dyn ToolProvider>,
    shared_specs: Vec<ToolSpec>,
    shared_names: HashSet<String>,
}

impl SharedToolsFactory {
    pub async fn new(
        world: Arc<dyn EnvironmentFactory>,
        shared: Arc<dyn ToolProvider>,
    ) -> Result<Self> {
        let shared_specs = shared.list_tools().await?;
        let mut shared_names = HashSet::new();
        for spec in &shared_specs {
            if !shared_names.insert(spec.name.clone()) {
                return Err(retrograd_agent_core::Error::invalid(format!(
                    "duplicate shared tool '{}'",
                    spec.name
                )));
            }
        }
        // The world's own catalogue is read from each instance rather than
        // snapshotted here from a disposable one: an `HttpEnvironment` answers
        // `tools()` from the live remote, so a cached list can describe a
        // session the trajectory is not in - and building a world only to close
        // it starts and discards a real container before `prepare` ever runs.
        Ok(Self {
            world,
            shared,
            shared_specs,
            shared_names,
        })
    }
}

struct SharedToolsEnvironment {
    world: tokio::sync::Mutex<Box<dyn Environment>>,
    shared: Arc<dyn ToolProvider>,
    shared_specs: Vec<ToolSpec>,
    shared_names: HashSet<String>,
}

#[async_trait]
impl Environment for SharedToolsEnvironment {
    async fn reset(&mut self, scenario: &Scenario, seed: u64) -> Result<Option<String>> {
        self.world.get_mut().reset(scenario, seed).await
    }

    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        if self.shared_names.contains(&call.name) {
            return Ok(StepOutcome::tool(match self.shared.call(call).await {
                Ok(result) => result,
                Err(error) => ToolResult::error(call.id.clone(), error.to_string()),
            }));
        }
        self.world.get_mut().step(call).await
    }

    async fn tools(&self) -> Result<Vec<ToolSpec>> {
        let world_tools = self.world.lock().await.tools().await?;
        // Checked here, per instance and per call: the collision is between what
        // this world advertises now and the shared names, and a shadowed world
        // tool would otherwise be silently routed to the shared provider.
        if let Some(collision) = world_tools
            .iter()
            .find(|spec| self.shared_names.contains(&spec.name))
        {
            return Err(retrograd_agent_core::Error::invalid(format!(
                "tool name collision while composing environment and shared tools: '{}'",
                collision.name
            )));
        }
        let mut combined = world_tools;
        combined.extend(self.shared_specs.iter().cloned());
        combined.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(combined)
    }

    async fn state(&mut self) -> Result<EnvState> {
        self.world.get_mut().state().await
    }

    async fn close(&mut self) -> Result<()> {
        self.world.get_mut().close().await
    }
}

#[async_trait]
impl EnvironmentFactory for SharedToolsFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        Ok(Box::new(SharedToolsEnvironment {
            world: tokio::sync::Mutex::new(self.world.create().await?),
            shared: self.shared.clone(),
            shared_specs: self.shared_specs.clone(),
            shared_names: self.shared_names.clone(),
        }))
    }

    async fn prepare(&self, scenarios: &[Scenario]) -> Result<()> {
        self.world.prepare(scenarios).await
    }

    async fn prewarm(&self, count: usize) -> Result<()> {
        self.world.prewarm(count).await
    }

    async fn shutdown(&self) {
        self.world.shutdown().await
    }

    async fn force_cleanup(&self) -> usize {
        self.world.force_cleanup().await
    }

    fn pinned_image(&self) -> Option<String> {
        self.world.pinned_image()
    }

    fn metric_values(&self) -> Vec<retrograd_metrics::MetricValue> {
        self.world.metric_values()
    }
}

#[cfg(test)]
mod shared_tools_tests {
    use super::*;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "test".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    struct World;
    #[async_trait]
    impl Environment for World {
        async fn reset(&mut self, _: &Scenario, _: u64) -> Result<Option<String>> {
            Ok(None)
        }
        async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
            Ok(StepOutcome::tool(ToolResult::ok(call.id.clone(), "world")))
        }
        async fn tools(&self) -> Result<Vec<ToolSpec>> {
            Ok(vec![spec("session")])
        }
        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    struct WorldFactory;
    #[async_trait]
    impl EnvironmentFactory for WorldFactory {
        async fn create(&self) -> Result<Box<dyn Environment>> {
            Ok(Box::new(World))
        }
    }

    struct Shared {
        name: &'static str,
    }
    #[async_trait]
    impl ToolProvider for Shared {
        async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
            Ok(vec![spec(self.name)])
        }
        async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::ok(call.id.clone(), "shared"))
        }
    }

    #[tokio::test]
    async fn composition_routes_by_name_and_refuses_collisions() {
        let factory =
            SharedToolsFactory::new(Arc::new(WorldFactory), Arc::new(Shared { name: "lookup" }))
                .await
                .unwrap();
        let mut environment = factory.create().await.unwrap();
        assert_eq!(
            environment
                .tools()
                .await
                .unwrap()
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["lookup", "session"]
        );
        for (name, expected) in [("lookup", "shared"), ("session", "world")] {
            let outcome = environment
                .step(&ToolCall {
                    id: "1".into(),
                    name: name.into(),
                    arguments: serde_json::json!({}),
                })
                .await
                .unwrap();
            assert_eq!(outcome.result.content, expected);
        }
        // The world's catalogue is read from the instance, so a name collision
        // is reported when that catalogue is asked for rather than snapshotted
        // at construction from a world built only to be closed.
        let colliding =
            SharedToolsFactory::new(Arc::new(WorldFactory), Arc::new(Shared { name: "session" }))
                .await
                .unwrap();
        let error = match colliding.create().await.unwrap().tools().await {
            Ok(_) => panic!("collision must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("collision"), "{error}");
    }
}
