//! What a training loop hands to a [`TrajectoryObserver`](crate::TrajectoryObserver).
//!
//! Plain data only, so the crate depends on neither the agent stack nor the
//! training crate. The field names are the ones written to `observe.jsonl`.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

/// One unit sent through the channel. The producer sends them in this order
/// for an update; any of them may be dropped independently.
#[derive(Clone, Debug)]
pub enum ObserveBatch {
    Rollouts(RolloutBatch),
    /// Agentic GRPO only: the advantages and the effective mask, computed by
    /// the policy actor before the epochs.
    Selection {
        update: u32,
        entries: Vec<SelectionEntry>,
    },
    /// Sent once every epoch of the update succeeded.
    Outcome {
        update: u32,
        entries: Vec<OutcomeEntry>,
    },
    Update(UpdateSummary),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Ppo,
    Grpo,
    AgentGrpo,
}

/// The `run` record, written when the sink opens.
#[derive(Clone, Debug, Serialize)]
pub struct RunInfo {
    pub algorithm: Algorithm,
    pub model: String,
    /// Updates consumed by the checkpoint this run resumes from.
    pub resumed_from_update: Option<u32>,
    pub params: Map<String, Value>,
}

/// The rollouts of one update, with every prompt they reference. The writer
/// skips the prompts it already wrote in the segment.
#[derive(Clone, Debug, Default)]
pub struct RolloutBatch {
    pub prompts: Vec<ObservedPrompt>,
    pub rollouts: Vec<ObservedRollout>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ObservedPrompt {
    /// `p:<index>` for a prompt file line, `s:<scenario_id>` for a scenario.
    pub key: String,
    pub messages: Vec<ObservedMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ObservedMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ObservedToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ObservedMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            is_error: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ObservedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Why a rollout does not reach the optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    Truncated,
    ZeroSignal,
    JudgeDropped,
    Unscored,
    UpdateSkipped,
}

#[derive(Clone, Debug, Serialize)]
pub struct ObservedRollout {
    /// One-based.
    pub update: u32,
    /// `None` for PPO, which has no groups.
    pub group: Option<usize>,
    pub member: usize,
    /// Key of the `prompt` record.
    pub prompt: String,
    pub seed: u64,
    pub tokens: usize,
    pub truncated: bool,
    /// What the optimizer is handed, after shaping.
    pub reward: Option<f32>,
    pub reward_raw: Option<f32>,
    pub judge_term: Option<f32>,
    pub advantage: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage_max: Option<f32>,
    /// `None` until the policy actor publishes its selection.
    pub eligible: Option<bool>,
    /// `Some(false)` only for a confirmed exclusion; a confirmed training
    /// arrives in an `outcome` record.
    pub trained: Option<bool>,
    pub skip_reason: Option<SkipReason>,
    #[serde(flatten)]
    pub content: RolloutContent,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum RolloutContent {
    Completion {
        completion: String,
    },
    Conversation {
        /// After the scenario prefix when `prefix` is true, whole otherwise.
        messages: Vec<ObservedMessage>,
        prefix: bool,
        step_rewards: Vec<StepReward>,
        terminal_reward_raw: Option<f32>,
        judge_explanation: Option<String>,
        /// Entries that differ from the scenario's.
        metadata: Map<String, Value>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StepReward {
    pub step_index: usize,
    pub kind: String,
    pub reward: f32,
    /// Indices into the exported `messages`; empty when unknown.
    pub message_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SelectionEntry {
    pub group: usize,
    pub member: usize,
    pub advantage: f32,
    pub eligible: bool,
    pub skip_reason: Option<SkipReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutcomeEntry {
    pub group: Option<usize>,
    pub member: usize,
    pub trained: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateStatus {
    Completed,
    Skipped,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateSummary {
    pub update: u32,
    pub status: UpdateStatus,
    /// The scalars the update published, under their metric names.
    pub metrics: BTreeMap<String, f32>,
}

impl UpdateSummary {
    pub fn new<'a>(
        update: u32,
        status: UpdateStatus,
        metrics: impl IntoIterator<Item = (&'a str, f32)>,
    ) -> Self {
        Self {
            update,
            status,
            metrics: metrics
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
        }
    }
}
