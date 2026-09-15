//! The `[agent]` section: agentic GRPO, in the same document as every other
//! algorithm.
//!
//! This module adds the section that describes a multi-turn rollout. It shares
//! `[model]`, `[lora]`, `[training]` and `[metrics]` with the other algorithms.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_agent_core::config::{AgentGrpoConfig, JudgeFailurePolicy};
use retrograd_agent_core::scenario::{RolloutLimits, TruncationPolicy};
use retrograd_core::{Error, Result};
use retrograd_spec::env::EnvironmentConfig;
use retrograd_spec::judge::JudgeConfig;
use retrograd_spec::tools::{McpServerConfig, ToolPlan};

/// What the runner needs to execute an agentic run: the loop's own
/// configuration, plus the three things it acts through - scenarios, a judge,
/// and a world.
#[derive(Clone, Debug)]
pub struct AgentRunConfig {
    pub scenarios: PathBuf,
    pub config: AgentGrpoConfig,
    /// `None` means no judge: the reward is the environment's own, and a group
    /// it did not score is a group this run cannot grade.
    pub judge: Option<JudgeConfig>,
    /// `None` means the trajectory has no world of its own and the tools come
    /// from the plan's MCP servers.
    pub environment: Option<EnvironmentConfig>,
    /// Declared, never merged: the servers named inline plus the `mcp_config`
    /// files to read. Merging them is I/O against the running machine, so it
    /// belongs to the runner (`ToolPlanResolve::merged_servers` in
    /// `retrograd-tools`), not to the reader.
    pub tool_plan: ToolPlan,
    pub scenario_generation: Option<ScenarioGenerationConfig>,
    /// `max_trajectory_tokens` as written, before it is clamped to the model's
    /// context. The clamp needs a loaded model, so it happens in the runner and
    /// `config.limits.max_trajectory_tokens` holds the requested value until
    /// then - [`AgentRunConfig::trajectory_limit`] applies it.
    pub explicit_trajectory_limit: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScenarioGenerationConfig {
    pub model: String,
    pub base_url: String,
    pub api_key_env: String,
    pub count: usize,
    pub batch_size: usize,
    pub timeout_secs: u64,
    pub max_retries: usize,
    pub seed: Option<u64>,
    pub custom_instructions: String,
    pub min_difficulty: u8,
    pub max_difficulty: u8,
    pub max_catalog_bytes: usize,
    pub shuffle: bool,
}

impl Default for ScenarioGenerationConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            base_url: String::new(),
            api_key_env: String::new(),
            count: 24,
            batch_size: 12,
            timeout_secs: 120,
            max_retries: 2,
            seed: None,
            custom_instructions: String::new(),
            min_difficulty: 1,
            max_difficulty: 5,
            max_catalog_bytes: 256 * 1024,
            shuffle: true,
        }
    }
}

impl ScenarioGenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty()
            || self.base_url.trim().is_empty()
            || self.api_key_env.trim().is_empty()
        {
            return Err(Error::config(
                "scenario generation model, base_url and api_key_env are required",
            ));
        }
        if self.count == 0
            || self.batch_size == 0
            || self.batch_size > self.count
            || self.timeout_secs == 0
            || self.max_catalog_bytes == 0
        {
            return Err(Error::config(
                "scenario generation counts, timeout and catalogue budget must be positive, with batch_size <= count",
            ));
        }
        if self.min_difficulty < 1
            || self.max_difficulty > 5
            || self.min_difficulty > self.max_difficulty
        {
            return Err(Error::config(
                "scenario generation difficulty range must be within 1..=5",
            ));
        }
        Ok(())
    }
}

impl AgentRunConfig {
    /// The trajectory budget to run with, given the loaded model's context.
    ///
    /// Unset means "the whole context". Set above it is an error rather than a
    /// silent clamp: a budget the model cannot hold is a configuration someone
    /// has to fix, and the loss denominator is derived from it (see
    /// [`AgentGrpoConfig::loss_denominator`]).
    pub fn trajectory_limit(&self, context_size: usize) -> Result<usize> {
        if !self.explicit_trajectory_limit {
            return Ok(context_size);
        }
        let requested = self.config.limits.max_trajectory_tokens;
        if requested > context_size {
            return Err(Error::config(format!(
                "agent.max_trajectory_tokens ({requested}) exceeds the model context \
                 ({context_size})"
            )));
        }
        Ok(requested)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentToml {
    pub scenarios: PathBuf,
    /// An addition, not a default. Absent, the environment's own reward is the
    /// only one - which is the whole configuration of a verifiable task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeConfig>,
    /// Tools served over MCP. With an environment, every server must be
    /// explicitly declared stateless before it may be shared across members.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default, deserialize_with = "deserialize_paths")]
    pub mcp_config: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenario_generation: Option<ScenarioGenerationConfig>,
    /// One stateful environment per trajectory - HTTP, container or (explicitly)
    /// the host.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentConfig>,
    pub updates: u32,
    pub scenarios_per_update: usize,
    pub group_size: usize,
    /// Optimizer passes over one update's rollouts. Spelled in full because
    /// `training.epochs` is a different quantity - passes over an SFT dataset.
    pub epochs_per_update: u32,
    pub max_turns: usize,
    /// Generation budget for one assistant turn. `[grpo.sampling]`'s
    /// `max_new_tokens` is the single-turn equivalent; the two live in
    /// different sections because a turn is not a completion.
    pub max_new_tokens_per_turn: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_trajectory_tokens: Option<usize>,
    pub max_rollout_secs: u64,
    /// `false` turns a turn that named no tool into an error observation
    /// instead of the end of the trajectory; see
    /// [`RolloutLimits::end_on_no_tool_call`]. Leave it alone for a run whose
    /// policy answers, set it for a world that decides when the episode is
    /// over.
    pub end_on_no_tool_call: bool,
    /// Consecutive turns without a valid tool call before the trajectory is
    /// cut as a truncation; see [`RolloutLimits::max_failed_turns`]. Zero
    /// never cuts.
    pub max_failed_turns: usize,
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    pub kl_coefficient: f32,
    pub judge_failure: JudgeFailurePolicy,
    pub max_dropped_fraction: f32,
    /// Drop groups the judge scored identically across every member; see
    /// [`AgentGrpoConfig::drop_degenerate_groups`].
    pub drop_degenerate_groups: bool,
    /// Skip an update left with fewer than two trainable trajectories rather
    /// than failing the run; see [`AgentGrpoConfig::skip_empty_updates`].
    pub skip_empty_updates: bool,
    /// `"drop"` (default) or `"min_reward"` - see [`TruncationPolicy`].
    pub truncation: TruncationPolicy,
    pub seed: u64,
}

impl Default for AgentToml {
    fn default() -> Self {
        let defaults = AgentGrpoConfig::default();
        Self {
            scenarios: PathBuf::new(),
            judge: None,
            mcp_servers: vec![],
            mcp_config: vec![],
            scenario_generation: None,
            environment: None,
            updates: defaults.updates,
            scenarios_per_update: defaults.scenarios_per_update,
            group_size: defaults.group_size,
            epochs_per_update: defaults.epochs,
            max_turns: defaults.limits.max_turns,
            max_new_tokens_per_turn: defaults.limits.max_new_tokens_per_turn,
            max_trajectory_tokens: None,
            max_rollout_secs: defaults.limits.max_rollout_secs,
            end_on_no_tool_call: defaults.limits.end_on_no_tool_call,
            max_failed_turns: defaults.limits.max_failed_turns,
            clip_range_low: defaults.clip_range_low,
            clip_range_high: defaults.clip_range_high,
            kl_coefficient: defaults.kl_coefficient,
            judge_failure: defaults.judge_failure,
            max_dropped_fraction: defaults.max_dropped_fraction,
            drop_degenerate_groups: defaults.drop_degenerate_groups,
            skip_empty_updates: defaults.skip_empty_updates,
            truncation: defaults.truncation,
            seed: defaults.seed,
        }
    }
}

pub(crate) fn build_agent(
    mut value: AgentToml,
    root: &Path,
    resolve: impl Fn(&Path, PathBuf) -> PathBuf,
) -> Result<AgentRunConfig> {
    // `[agent]` carries defaults for everything a run can be tuned with, so a
    // missing table would otherwise deserialize into a run with no scenarios,
    // the one thing it cannot invent.
    if value.scenarios.as_os_str().is_empty() {
        return Err(Error::config("agent.scenarios is required"));
    }
    // A judge is an addition, not the default. What a rollout cannot do without
    // is *a* reward, and an environment that grades its own steps - an HTTP
    // world, or a sandbox running a `metadata.env.verify` command - already is
    // one. So the requirement is on the pair, and it is the pair the message
    // names: a run with neither would train on nothing, and finding that out at
    // the first update is an hour of rollouts too late.
    match &value.judge {
        Some(JudgeConfig::Command { command, .. }) if command.is_empty() => {
            return Err(Error::config(
                "[agent.judge] declares an empty command: give it one, or drop the table \
                 entirely to grade with the environment alone",
            ));
        }
        None if value.environment.is_none() => {
            return Err(Error::config(
                "[agent] declares neither a judge nor an environment, so nothing would give a \
                 trajectory a reward: add [agent.judge], or an [agent.environment] that scores \
                 its own steps",
            ));
        }
        _ => {}
    }
    if let Some(environment) = &value.environment {
        // Shape only: a document must stay readable by a build that cannot run
        // it. The runner reports "not compiled into this binary" when it goes
        // to instantiate the world.
        environment.validate_declaration().map_err(Error::from)?;
    }
    if let Some(generation) = &value.scenario_generation {
        generation.validate()?;
    }
    if let Some(generation) = &mut value.scenario_generation {
        generation.seed.get_or_insert(value.seed);
    }
    let mcp_files = value
        .mcp_config
        .iter()
        .cloned()
        .map(|path| resolve(root, path))
        .collect::<Vec<_>>();
    let (profile, builtin, filter) = value
        .environment
        .as_ref()
        .map(EnvironmentConfig::tool_selection)
        .unwrap_or((
            retrograd_spec::tools::Profile::Custom,
            Some(Vec::new()),
            Default::default(),
        ));
    // Declared, not merged: resolving `mcp_config` files here would mean
    // reading a document requires the machine that will run it - the same
    // invariant `validate_declaration` keeps for environments. A planner or a
    // server loads configurations for machines that are not theirs, and would
    // fail on an absent file or an unset `${VAR}`. Whether every shared server
    // is stateless is checked where the plan is actually resolved, against the
    // merged view, in `retrograd-run`.
    let tool_plan = ToolPlan {
        builtin,
        profile,
        mcp_servers: value.mcp_servers.clone(),
        mcp_config_files: mcp_files,
        filter,
    };
    // What the document itself declares can still be judged without touching
    // the filesystem, so an inline server that forgot `stateless` is reported at
    // parse time rather than after the images have been pulled.
    if value.environment.is_some()
        && let Some(server) = value.mcp_servers.iter().find(|server| !server.stateless)
    {
        return Err(Error::config(format!(
            "MCP server '{}' must set stateless = true before it can be shared with an environment; otherwise group members may contaminate each other",
            server.name
        )));
    }
    let explicit_trajectory_limit = value.max_trajectory_tokens.is_some();
    let config = AgentGrpoConfig {
        updates: value.updates,
        scenarios_per_update: value.scenarios_per_update,
        group_size: value.group_size,
        epochs: value.epochs_per_update,
        clip_range_low: value.clip_range_low,
        clip_range_high: value.clip_range_high,
        kl_coefficient: value.kl_coefficient,
        seed: value.seed,
        limits: RolloutLimits {
            max_turns: value.max_turns,
            max_new_tokens_per_turn: value.max_new_tokens_per_turn,
            // A requested budget is validated against the model's context by the
            // runner; unset stands for "the whole context" and must not fail the
            // limits check here, so it borrows the default.
            max_trajectory_tokens: value
                .max_trajectory_tokens
                .unwrap_or(RolloutLimits::default().max_trajectory_tokens),
            max_rollout_secs: value.max_rollout_secs,
            end_on_no_tool_call: value.end_on_no_tool_call,
            max_failed_turns: value.max_failed_turns,
        },
        judge_failure: value.judge_failure,
        max_dropped_fraction: value.max_dropped_fraction,
        drop_degenerate_groups: value.drop_degenerate_groups,
        skip_empty_updates: value.skip_empty_updates,
        truncation: value.truncation,
    };
    config.validate().map_err(Error::from)?;
    // The one path inside the judge that is written relative to the config file.
    let mut judge = value.judge;
    if let Some(JudgeConfig::Ruler { config }) = &mut judge {
        config.cache_path = config
            .cache_path
            .take()
            .map(|path| resolve(root, path.to_path_buf()));
    }
    Ok(AgentRunConfig {
        scenarios: resolve(root, value.scenarios),
        config,
        judge,
        environment: value.environment,
        tool_plan,
        scenario_generation: value.scenario_generation,
        explicit_trajectory_limit,
    })
}

fn deserialize_paths<'de, D>(deserializer: D) -> std::result::Result<Vec<PathBuf>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(PathBuf),
        Many(Vec<PathBuf>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(path) => vec![path],
        OneOrMany::Many(paths) => paths,
    })
}
