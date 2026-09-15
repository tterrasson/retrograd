//! What a configuration file says about the judge.
//!
//! The RULER judge's own transport, its rendering, its cache and its rubrics
//! live in `retrograd-judge`; what is here is what the file declares - which
//! judge, at which endpoint, with which budgets.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use retrograd_agent_core::{Error, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeContext {
    /// Hard cap on one judge request's prompt. Groups whose rendering exceeds
    /// this are split into several requests (see `retrograd_judge::render::plan_chunks`).
    pub max_request_chars: usize,
    /// Cap on one rendered trajectory. Beyond it, messages are elided from the
    /// middle of the conversation.
    pub max_trajectory_chars: usize,
    /// Cap on one message inside a trajectory. Long tool observations are the
    /// usual offender.
    pub max_message_chars: usize,
    /// Share of an elided message kept from its head; the remainder is kept
    /// from its tail. Tool output usually carries its conclusion at the end,
    /// and the assistant's reasoning its intent at the start, so neither end
    /// can be dropped outright.
    pub head_ratio: f32,
    /// Whether to show each member's terminal environment state - the diff, for
    /// a code task - alongside its transcript. On by default because it costs
    /// nothing when the environment reports none, and because judging the
    /// dialogue rather than the result is the main source of reward noise on any
    /// task whose outcome is a file and not a sentence.
    pub include_env_state: bool,
}

impl Default for JudgeContext {
    fn default() -> Self {
        Self {
            max_request_chars: 60_000,
            max_trajectory_chars: 8_000,
            max_message_chars: 2_000,
            head_ratio: 0.4,
            include_env_state: true,
        }
    }
}

impl JudgeContext {
    pub fn validate(&self) -> Result<()> {
        if self.max_request_chars == 0
            || self.max_trajectory_chars == 0
            || self.max_message_chars == 0
        {
            return Err(Error::invalid(
                "judge context budgets must all be greater than zero",
            ));
        }
        if self.max_message_chars > self.max_trajectory_chars {
            return Err(Error::invalid(
                "judge max_message_chars must not exceed max_trajectory_chars",
            ));
        }
        if self.max_trajectory_chars > self.max_request_chars {
            return Err(Error::invalid(
                "judge max_trajectory_chars must not exceed max_request_chars",
            ));
        }
        if !self.head_ratio.is_finite() || !(0.0..=1.0).contains(&self.head_ratio) {
            return Err(Error::invalid(
                "judge head_ratio must be finite and in [0, 1]",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompactionConfig {
    /// A group is compacted when its longest member exceeds this many
    /// characters. Below it nothing is sent and nothing is spent.
    pub trigger_chars: usize,
    /// Character budget offered to the summary of the middle. Advisory - the
    /// model is asked to respect it and the result is elided if it does not.
    pub target_chars: usize,
    /// Closing messages kept verbatim. The result of a trajectory is the part a
    /// judge is least able to reconstruct from a summary.
    pub keep_last: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            trigger_chars: 6_000,
            target_chars: 1_000,
            keep_last: 4,
        }
    }
}

impl CompactionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.trigger_chars == 0 || self.target_chars == 0 {
            return Err(Error::invalid(
                "judge compaction trigger_chars and target_chars must be greater than zero",
            ));
        }
        if self.keep_last == 0 {
            return Err(Error::invalid(
                "judge compaction keep_last must be greater than zero: the closing turns carry the outcome",
            ));
        }
        if self.target_chars >= self.trigger_chars {
            return Err(Error::invalid(
                "judge compaction target_chars must be below trigger_chars, otherwise compacting cannot shorten anything",
            ));
        }
        Ok(())
    }

    /// The messages of one trajectory that may be replaced by a summary: those
    /// after the opening and before the last `keep_last`. Empty when there is
    /// nothing in between, which is the common case for short trajectories and
    /// costs no request.
    pub fn middle(&self, len: usize) -> std::ops::Range<usize> {
        let start = 1.min(len);
        let end = len.saturating_sub(self.keep_last).max(start);
        start..end
    }
}

/// How pairwise verdicts become one score per trajectory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    /// Wins over matches, ties counting half. Cheap, bounded, and blind to
    /// *whom* a trajectory beat.
    #[default]
    WinRate,
    /// Bradley-Terry strengths, min-max projected into `[0,1]` within the group.
    /// Worth its cost exactly when the schedule is truncated: beating the
    /// group's best is then not the same evidence as beating its worst, and win
    /// rate cannot tell the two apart.
    BradleyTerry,
}

/// How a group is presented to the judge.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum JudgeStrategy {
    /// One listwise request per group when it fits the context budget, split
    /// into anchored chunks when it does not. The default: correct on small
    /// groups, and it degrades into chunking instead of failing on large ones.
    #[default]
    Auto,
    /// Always one listwise request per group. Cheapest, and the only mode whose
    /// scores come from a single comparison - at the cost of failing outright
    /// when the group overruns the judge's window.
    Listwise,
    /// Always split into chunks under the budget. `anchor` repeats the group's
    /// first trajectory in every chunk so per-chunk scores share a scale.
    Chunked {
        #[serde(default = "default_anchor")]
        anchor: bool,
    },
    /// Compare two trajectories at a time and aggregate win rates. The most
    /// reliable signal per judgement and the most robust to long trajectories
    /// (a request never holds more than two), at `max_pairs` requests per group.
    Pairwise {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_pairs: Option<usize>,
        /// Judge every pair in both presentation orders and keep only the
        /// agreements. **This doubles the request count**, and pairwise is
        /// already the most expensive mode: budget it together with `max_pairs`,
        /// which caps `max_pairs × 2` requests per group. What it buys is
        /// `judge/position_disagreement`, the number that says whether the
        /// chosen judge deserves to be listened to at all.
        #[serde(default = "default_both_orders")]
        both_orders: bool,
        #[serde(default)]
        aggregation: Aggregation,
    },
}

fn default_anchor() -> bool {
    true
}

fn default_both_orders() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulerConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
    /// Rubric for pairwise comparisons. Falls back to a pairwise-specific
    /// default rather than to `rubric`, whose wording asks for per-trajectory
    /// scores and confuses a two-way verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairwise_rubric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<PathBuf>,
    #[serde(default)]
    pub strategy: JudgeStrategy,
    #[serde(default)]
    pub context: JudgeContext,
    /// LLM compaction of the middle of each transcript, off by default. See
    /// [`CompactionConfig`] for why it is a group-wide decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionConfig>,
}

impl Default for RulerConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            api_key_env: default_api_key_env(),
            rubric: None,
            pairwise_rubric: None,
            temperature: None,
            max_concurrency: default_concurrency(),
            timeout_secs: default_timeout(),
            max_retries: default_retries(),
            cache_path: None,
            strategy: JudgeStrategy::default(),
            context: JudgeContext::default(),
            compaction: None,
        }
    }
}

fn default_api_key_env() -> String {
    "OPENAI_API_KEY".into()
}
fn default_concurrency() -> usize {
    4
}
fn default_timeout() -> u64 {
    120
}
fn default_retries() -> u32 {
    2
}

impl RulerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.is_empty() || self.model.is_empty() || self.api_key_env.is_empty() {
            return Err(Error::invalid(
                "RULER base_url, model, and api_key_env must not be empty",
            ));
        }
        if self.max_concurrency == 0 || self.timeout_secs == 0 {
            return Err(Error::invalid(
                "RULER concurrency and timeout must be greater than zero",
            ));
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(Error::invalid(
                "RULER temperature must be finite and non-negative",
            ));
        }
        if let JudgeStrategy::Pairwise {
            max_pairs: Some(0), ..
        } = self.strategy
        {
            return Err(Error::invalid(
                "RULER pairwise max_pairs must be greater than zero",
            ));
        }
        if let Some(compaction) = &self.compaction {
            compaction.validate()?;
        }
        self.context.validate()
    }
}

/// Serialized judge selection, shared by every frontend.
///
/// Two spellings are accepted for the RULER variant so the TOML runner and the
/// Python binding can keep the shapes they already use: fields inline next to
/// `type` (TOML, where a nested table would need its own `[…judge.config]`
/// header) or nested under `config` (Python, which serializes the dataclass as
/// one object). Both deserialize into the same value, which is what lets all
/// frontends share one schema instead of re-declaring `RulerConfig` field by
/// field.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JudgeConfig {
    /// An LLM judge scored through the RULER protocol.
    Ruler {
        #[serde(flatten)]
        config: Box<RulerConfig>,
    },
    /// An external command that reads a group and writes back a verdict.
    Command {
        command: Vec<String>,
        timeout_secs: u64,
    },
}

fn default_command_timeout() -> u64 {
    30
}

impl<'de> Deserialize<'de> for JudgeConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let mut value = serde_json::Value::deserialize(deserializer)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| D::Error::custom("judge configuration must be an object"))?;
        let kind = object
            .remove("type")
            .and_then(|kind| kind.as_str().map(str::to_owned))
            .ok_or_else(|| D::Error::custom("judge configuration needs a 'type' field"))?;
        match kind.as_str() {
            "ruler" => {
                let inner = match object.remove("config") {
                    Some(config) => {
                        if !object.is_empty() {
                            return Err(D::Error::custom(
                                "judge 'config' must not be mixed with inline RULER fields",
                            ));
                        }
                        config
                    }
                    None => serde_json::Value::Object(std::mem::take(object)),
                };
                let config: RulerConfig =
                    serde_json::from_value(inner).map_err(D::Error::custom)?;
                Ok(Self::Ruler {
                    config: Box::new(config),
                })
            }
            "command" => {
                let command = object
                    .remove("command")
                    .ok_or_else(|| D::Error::custom("command judge needs a 'command' field"))?;
                let command: Vec<String> =
                    serde_json::from_value(command).map_err(D::Error::custom)?;
                let timeout_secs = match object.remove("timeout_secs") {
                    Some(value) => serde_json::from_value(value).map_err(D::Error::custom)?,
                    None => default_command_timeout(),
                };
                if !object.is_empty() {
                    return Err(D::Error::custom(format!(
                        "unknown command judge fields: {}",
                        object.keys().cloned().collect::<Vec<_>>().join(", ")
                    )));
                }
                Ok(Self::Command {
                    command,
                    timeout_secs,
                })
            }
            other => Err(D::Error::custom(format!("unknown judge type '{other}'"))),
        }
    }
}
