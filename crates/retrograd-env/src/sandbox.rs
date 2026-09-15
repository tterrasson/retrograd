//! The environment that runs tools in a sandbox.
//!
//! This is where the three halves meet: a [`SandboxProvider`] says *where* an
//! episode runs, a [`ToolSet`] says *what* it may do, and an [`EnvTask`] says
//! what the scenario asks for. None of the three knows about the other two, and
//! none of them knows about Docker.
//!
//! The split that governs every path below is the one from
//! [`Environment::step`]: a tool that failed is an observation the policy reads,
//! a sandbox that is gone is an `Err` that costs the trajectory. Blurring them
//! would train on a world that stopped existing.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use retrograd_agent_core::{
    EnvState, Environment, EnvironmentFactory, Error, ExecRequest, Lease, Result, Sandbox,
    SandboxProvider, Scenario, StepOutcome, ToolCall, ToolSpec,
};
use retrograd_tools::ToolSet;

use crate::task::{EnvTask, Verify};

#[derive(Clone, Debug)]
pub struct SandboxEnvironmentConfig {
    /// The image reference the configuration named - usually a tag. Only used to
    /// refuse a scenario asking for a different one.
    pub image: Option<String>,
    /// What that reference actually resolved to, `name@sha256:…`, once the
    /// provider had a daemon to ask.
    ///
    /// Kept beside `image` rather than replacing it because both spellings are
    /// legitimate in a scenario: one dataset names the tag it was written
    /// against, another names the digest it was verified on, and refusing either
    /// would be refusing the truth. It is also the only value that makes the run
    /// reproducible, so it is what gets logged and reported upwards.
    pub pinned_image: Option<String>,
    /// Budget for the whole `setup` list. Installing a dependency tree is not a
    /// tool call and has no business being bounded like one.
    pub setup_timeout: Duration,
    /// Default budget for the task's `verify` command, when the task does not
    /// name one of its own. Falls back to the sandbox's own exec timeout.
    pub verify_timeout: Option<Duration>,
    /// Cap on the text `state()` hands the judge. A `git diff` of a runaway
    /// episode would otherwise blow the judge's context on one member.
    pub max_summary_bytes: usize,
}

impl Default for SandboxEnvironmentConfig {
    fn default() -> Self {
        Self {
            image: None,
            pinned_image: None,
            setup_timeout: Duration::from_secs(300),
            verify_timeout: None,
            max_summary_bytes: 32 * 1024,
        }
    }
}

/// One episode: one lease, one workspace, one task.
pub struct SandboxEnvironment {
    provider: Arc<dyn SandboxProvider>,
    tools: Arc<ToolSet>,
    config: Arc<SandboxEnvironmentConfig>,
    lease: Option<Lease>,
    task: EnvTask,
    cumulative_reward: Option<f32>,
    done: bool,
}

impl SandboxEnvironment {
    pub fn new(
        provider: Arc<dyn SandboxProvider>,
        tools: Arc<ToolSet>,
        config: Arc<SandboxEnvironmentConfig>,
    ) -> Self {
        Self {
            provider,
            tools,
            config,
            lease: None,
            task: EnvTask::default(),
            cumulative_reward: None,
            done: false,
        }
    }

    fn lease(&self) -> Result<&Lease> {
        self.lease
            .as_ref()
            .ok_or_else(|| Error::Tool("environment used before reset".into()))
    }

    /// Marks the sandbox unfit for another episode. Called on every path where
    /// something went wrong with the world itself: recycling on doubt is the one
    /// failure that shows up in no metric.
    fn poison(&mut self) {
        if let Some(lease) = self.lease.as_mut() {
            lease.poison();
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        let config = self.config.clone();
        let task = self.task.clone();
        let sandbox = self.lease()?.sandbox();
        for (path, contents) in &task.files {
            sandbox.write_file(path, contents.as_bytes()).await?;
        }
        for command in &task.setup {
            let request = ExecRequest::new(["sh", "-c", command])
                .with_timeout(config.setup_timeout)
                .with_max_output_bytes(sandbox.limits().max_output_bytes);
            let output = sandbox.exec(request).await?;
            if !output.succeeded() {
                // Not an observation: the episode would start from a state the
                // scenario does not describe, and everything after it would be
                // scored as if it had.
                return Err(Error::Tool(format!(
                    "setup command failed with exit code {:?}: {}",
                    output.exit_code,
                    output.stderr.trim()
                )));
            }
        }
        Ok(())
    }

    /// Runs the task's verification and turns it into a step reward.
    async fn verify(&self, verify: &Verify, sandbox: &dyn Sandbox) -> Result<(f32, bool)> {
        let timeout = verify.timeout(self.config.verify_timeout, sandbox.limits().exec_timeout);
        let request = ExecRequest::new(verify.command.clone())
            .with_timeout(timeout)
            .with_max_output_bytes(sandbox.limits().max_output_bytes);
        let output = sandbox.exec(request).await?;
        Ok(if output.succeeded() {
            (verify.reward_on_success, true)
        } else {
            (verify.reward_on_failure, false)
        })
    }
}

#[async_trait]
impl Environment for SandboxEnvironment {
    async fn reset(&mut self, scenario: &Scenario, _seed: u64) -> Result<Option<String>> {
        // Re-parsed rather than cached from `prepare`: an environment instance
        // outlives no scenario, and the parse is a few microseconds against a
        // container acquisition.
        self.task = EnvTask::from_scenario(scenario)?;
        self.task.check_image(
            &scenario.id,
            self.config.image.as_deref(),
            self.config.pinned_image.as_deref(),
        )?;
        self.cumulative_reward = None;
        self.done = false;
        // Release any previous lease before awaiting a replacement. Holding it
        // while acquiring deadlocks a one-slot pool on reset.
        self.lease.take();
        self.lease = Some(self.provider.acquire().await?);
        if let Err(error) = self.materialize().await {
            // The workspace is half-written; nobody else gets this sandbox.
            self.poison();
            return Err(error);
        }
        Ok(self.task.instructions.clone())
    }

    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        let sandbox = self.lease()?.sandbox();
        let outcome = match self.tools.call(sandbox.as_ref(), call).await {
            Ok(outcome) => outcome,
            Err(error) => {
                self.poison();
                return Err(error);
            }
        };
        let mut content = outcome.content;
        let mut reward = outcome.reward;
        let done = outcome.done;

        if done && let Some(verify) = self.task.verify.clone() {
            let (value, passed) = match self.verify(&verify, sandbox.as_ref()).await {
                Ok(verdict) => verdict,
                Err(error) => {
                    self.poison();
                    return Err(error);
                }
            };
            // Fixed wording, like every other observation: it is trained on.
            content.push_str(if passed {
                "\nverification: passed"
            } else {
                "\nverification: failed"
            });
            reward = Some(reward.unwrap_or(0.0) + value);
        }

        // A task with a verifiable reward is graded by its own command, never by
        // the judge - so every step of such an episode carries a reward, `0.0`
        // where nothing happened. That is what `split_environment_scored` reads
        // to take the group out of the judge's hands, and it must not depend on
        // whether this particular member happened to submit.
        if reward.is_none() && self.task.verify.is_some() {
            reward = Some(0.0);
        }
        if let Some(value) = reward {
            *self.cumulative_reward.get_or_insert(0.0) += value;
        }
        self.done |= done;
        Ok(StepOutcome {
            result: retrograd_agent_core::ToolResult {
                call_id: call.id.clone(),
                content,
                is_error: outcome.is_error,
            },
            reward,
            done,
        })
    }

    async fn tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(self.tools.specs().to_vec())
    }

    async fn state(&mut self) -> Result<EnvState> {
        let summary = match (&self.task.summary, self.lease.as_ref()) {
            (Some(argv), Some(lease)) => {
                let sandbox = lease.sandbox();
                let request = ExecRequest::new(argv.clone())
                    .with_timeout(sandbox.limits().exec_timeout)
                    .with_max_output_bytes(self.config.max_summary_bytes);
                // A summary is read by the judge after the rollout; failing to
                // produce one must not retroactively cost a trajectory that has
                // already been collected.
                match sandbox.exec(request).await {
                    Ok(output) => Some(output.stdout),
                    Err(error) => {
                        tracing::warn!(%error, "environment summary command failed");
                        None
                    }
                }
            }
            _ => None,
        };
        Ok(EnvState {
            done: self.done,
            cumulative_reward: self.cumulative_reward,
            summary,
            metadata: serde_json::Value::Null,
        })
    }

    async fn close(&mut self) -> Result<()> {
        // Dropping the lease returns the sandbox; the provider does the slow
        // half - wiping, killing leftovers, destroying - on its own.
        self.lease = None;
        Ok(())
    }
}

/// Hands out one [`SandboxEnvironment`] per trajectory over a shared provider.
pub struct SandboxEnvironmentFactory {
    provider: Arc<dyn SandboxProvider>,
    tools: Arc<ToolSet>,
    config: Arc<SandboxEnvironmentConfig>,
}

impl SandboxEnvironmentFactory {
    pub fn new(
        provider: Arc<dyn SandboxProvider>,
        tools: ToolSet,
        config: SandboxEnvironmentConfig,
    ) -> Result<Self> {
        if tools.is_empty() {
            return Err(Error::invalid(
                "a sandbox environment with no tools gives the policy nothing to do",
            ));
        }
        Ok(Self {
            provider,
            tools: Arc::new(tools),
            config: Arc::new(config),
        })
    }
}

impl SandboxEnvironmentFactory {
    /// The immutable reference every episode of this run executes.
    pub fn pinned_image(&self) -> Option<&str> {
        self.config
            .pinned_image
            .as_deref()
            .or(self.config.image.as_deref())
    }
}

#[async_trait]
impl EnvironmentFactory for SandboxEnvironmentFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        Ok(Box::new(SandboxEnvironment::new(
            self.provider.clone(),
            self.tools.clone(),
            self.config.clone(),
        )))
    }

    async fn prepare(&self, scenarios: &[Scenario]) -> Result<()> {
        for scenario in scenarios {
            let task = EnvTask::from_scenario(scenario)?;
            task.check_image(
                &scenario.id,
                self.config.image.as_deref(),
                self.config.pinned_image.as_deref(),
            )?;
        }
        Ok(())
    }

    async fn prewarm(&self, count: usize) -> Result<()> {
        self.provider.prewarm(count).await
    }

    async fn shutdown(&self) {
        self.provider.shutdown().await;
    }

    async fn force_cleanup(&self) -> usize {
        self.provider.force_cleanup().await
    }

    fn metric_values(&self) -> Vec<retrograd_metrics::MetricValue> {
        self.provider.metric_values()
    }

    fn pinned_image(&self) -> Option<String> {
        SandboxEnvironmentFactory::pinned_image(self).map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use retrograd_tools::Profile;

    use super::*;
    use crate::local::{LocalSandboxConfig, LocalSandboxProvider};

    fn provider() -> Arc<dyn SandboxProvider> {
        LocalSandboxProvider::new(LocalSandboxConfig {
            allow_unsandboxed: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn factory(config: SandboxEnvironmentConfig) -> SandboxEnvironmentFactory {
        let tools = ToolSet::builder()
            .with_profile(Profile::Python)
            .build()
            .unwrap();
        SandboxEnvironmentFactory::new(provider(), tools, config).unwrap()
    }

    fn scenario(env: serde_json::Value) -> Scenario {
        Scenario {
            id: "s".into(),
            system: None,
            user: "do it".into(),
            metadata: serde_json::json!({"env": env})
                .as_object()
                .cloned()
                .unwrap(),
        }
    }

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments,
        }
    }

    #[tokio::test]
    async fn a_task_is_materialized_before_the_first_turn() {
        let factory = factory(SandboxEnvironmentConfig::default());
        let mut environment = factory.create().await.unwrap();
        let opening = environment
            .reset(
                &scenario(serde_json::json!({
                    "files": {"src/a.py": "value = 1\n"},
                    "setup": ["mkdir -p out"],
                    "instructions": "Fix it."
                })),
                0,
            )
            .await
            .unwrap();
        assert_eq!(opening.as_deref(), Some("Fix it."));

        let outcome = environment
            .step(&call("read_file", serde_json::json!({"path": "src/a.py"})))
            .await
            .unwrap();
        assert_eq!(outcome.result.content, "value = 1\n");
        assert!(
            outcome.reward.is_none(),
            "no verify means no reward to give"
        );
        environment.close().await.unwrap();
    }

    #[tokio::test]
    async fn resetting_an_environment_releases_the_previous_lease_first() {
        let factory = factory(SandboxEnvironmentConfig::default());
        let mut environment = factory.create().await.unwrap();
        environment
            .reset(
                &scenario(serde_json::json!({"files": {"old.txt": "old"}})),
                0,
            )
            .await
            .unwrap();
        environment
            .reset(&scenario(serde_json::json!({})), 1)
            .await
            .unwrap();
        let outcome = environment
            .step(&call("read_file", serde_json::json!({"path": "old.txt"})))
            .await
            .unwrap();
        assert!(outcome.result.is_error, "old workspace survived reset");
    }

    /// A setup command that fails leaves the episode in a state the scenario
    /// does not describe, so it costs the trajectory rather than becoming an
    /// observation the policy would be trained to work around.
    #[tokio::test]
    async fn a_broken_setup_is_a_lost_trajectory_not_an_observation() {
        let factory = factory(SandboxEnvironmentConfig::default());
        let mut environment = factory.create().await.unwrap();
        let error = environment
            .reset(&scenario(serde_json::json!({"setup": ["exit 7"]})), 0)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("setup command failed"),
            "{error}"
        );
    }

    /// The whole point of `verify`: the reward comes from a command's exit code,
    /// and every step carries one so the group leaves the judge's hands whatever
    /// each member did.
    #[tokio::test]
    async fn a_verifiable_task_grades_itself_at_submission() {
        let factory = factory(SandboxEnvironmentConfig::default());
        let mut environment = factory.create().await.unwrap();
        environment
            .reset(
                &scenario(serde_json::json!({
                    "files": {"answer.txt": "no\n"},
                    "verify": {"command": ["sh", "-c", "grep -q yes answer.txt"],
                               "reward_on_success": 1.0, "reward_on_failure": -0.5},
                    "summary": ["cat", "answer.txt"]
                })),
                0,
            )
            .await
            .unwrap();

        // An ordinary turn is scored zero, not left unscored.
        let outcome = environment
            .step(&call("list_dir", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(outcome.reward, Some(0.0));

        let failed = environment
            .step(&call("submit", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(failed.reward, Some(-0.5));
        assert!(failed.done && failed.result.content.ends_with("verification: failed"));

        // Same episode, now with the expected content: the verdict follows the
        // world, not the wording of the submission.
        environment
            .step(&call(
                "write_file",
                serde_json::json!({"path": "answer.txt", "content": "yes\n"}),
            ))
            .await
            .unwrap();
        let passed = environment
            .step(&call("submit", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(passed.reward, Some(1.0));
        assert!(passed.result.content.ends_with("verification: passed"));

        let state = environment.state().await.unwrap();
        assert_eq!(state.summary.as_deref(), Some("yes\n"));
        assert_eq!(state.cumulative_reward, Some(0.5));
        assert!(state.done);
    }

    /// The budget declared by the scenario is the timeout for its verification
    /// command, independently of the environment-wide timeout.
    #[tokio::test]
    async fn a_task_may_buy_its_verification_more_time_than_the_environment_gives() {
        let config = || SandboxEnvironmentConfig {
            // Short enough that anything but the task's own budget kills the
            // command below.
            verify_timeout: Some(Duration::from_millis(50)),
            ..Default::default()
        };
        let slow = |timeout: serde_json::Value| {
            scenario(serde_json::json!({
                "verify": {"command": ["sh", "-c", "sleep 0.5"],
                           "reward_on_success": 1.0, "reward_on_failure": -1.0,
                           "timeout_secs": timeout}
            }))
        };

        let mut environment = factory(config()).create().await.unwrap();
        environment
            .reset(&slow(serde_json::json!(5)), 0)
            .await
            .unwrap();
        let outcome = environment
            .step(&call("submit", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(
            outcome.reward,
            Some(1.0),
            "the task's own timeout_secs must be what the verify command runs under"
        );

        // And a task that names no budget still falls back to the environment's
        // - which is what kills this one.
        let mut environment = factory(config()).create().await.unwrap();
        environment
            .reset(&slow(serde_json::Value::Null), 0)
            .await
            .unwrap();
        let outcome = environment
            .step(&call("submit", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(outcome.reward, Some(-1.0));
    }

    #[tokio::test]
    async fn preparation_rejects_a_scenario_the_run_could_not_serve() {
        let factory = factory(SandboxEnvironmentConfig {
            image: Some("python:3.12-slim".into()),
            ..Default::default()
        });
        let good = scenario(serde_json::json!({"files": {"a.py": "x"}}));
        factory.prepare(std::slice::from_ref(&good)).await.unwrap();

        let wrong_image = scenario(serde_json::json!({"image": "node:22-slim"}));
        assert!(factory.prepare(&[good, wrong_image]).await.is_err());
    }

    #[tokio::test]
    async fn an_unknown_tool_is_an_observation_and_an_empty_set_is_refused() {
        let factory = factory(SandboxEnvironmentConfig::default());
        let mut environment = factory.create().await.unwrap();
        environment
            .reset(&scenario(serde_json::json!({})), 0)
            .await
            .unwrap();
        let outcome = environment
            .step(&call("teleport", serde_json::json!({})))
            .await
            .expect("an invented tool must not cost the trajectory");
        assert!(outcome.result.is_error && outcome.result.content.contains("unknown tool"));

        assert!(
            SandboxEnvironmentFactory::new(
                provider(),
                ToolSet::default(),
                SandboxEnvironmentConfig::default()
            )
            .is_err()
        );
    }
}
