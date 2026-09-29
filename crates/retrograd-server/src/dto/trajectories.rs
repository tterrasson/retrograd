//! Trajectories (`GET /v1/runs/{id}/trajectories…`)
//!
//! A contract of its own, converted from what `retrograd-observe` reads: the
//! file format and the HTTP schema move at different paces, like the rest of
//! this module and the engine's types.

use super::*;

/// A measurement as the wire wants it: absent when it is not a number.
fn finite(value: Option<f32>) -> Option<f32> {
    value.filter(|value| value.is_finite())
}

schema! {
/// `GET /v1/runs/{id}/trajectories`
///
/// A run without `[observe]` answers too, with `observed: false` and nothing
/// listed: it exists, and the answer is that it exports no trajectories.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TrajectoryOverview {
    pub observed: bool,
    /// `ppo` | `grpo` | `agent_grpo`, from the export itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// The base model's file name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// One update in `every` exports its texts; every update has a summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub every: Option<u32>,
    /// How many times the run started or resumed in this export.
    pub segments: u32,
    /// Lines and batches of the file that could not be read. Batches the run
    /// dropped because the disk was slow are counted by the run's metric
    /// `observe/dropped_batches`, not here.
    pub skipped: u64,
    /// Every update in view, by update number.
    pub updates: Vec<UpdateSummary>,
}
}

schema! {
/// One update of a trajectory export.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct UpdateSummary {
    /// One-based.
    pub update: u32,
    /// The run segment that wrote it: a resume starts a new one.
    pub segment: u32,
    /// `completed` | `skipped`; absent until the update is over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// RFC 3339.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,
    /// The scalars the update published, by metric name.
    pub metrics: BTreeMap<String, f32>,
    pub groups: usize,
    pub rollouts: usize,
    /// Share of the rollouts an `outcome` confirmed trained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trained_fraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_mean: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_max: Option<f32>,
    /// Whether the texts of this update were exported (`every`).
    pub texts: bool,
}
}

impl From<retrograd_observe::reader::UpdateEntry> for UpdateSummary {
    fn from(entry: retrograd_observe::reader::UpdateEntry) -> Self {
        let texts = entry.texts();
        // Counts of at most a few thousand rollouts: exact in f32.
        let trained_fraction =
            (entry.rollouts > 0).then(|| entry.trained as f32 / entry.rollouts as f32);
        Self {
            update: entry.update,
            segment: entry.segment,
            status: entry.status,
            time: entry.time,
            metrics: entry.metrics,
            groups: entry.groups,
            rollouts: entry.rollouts,
            trained_fraction,
            reward_mean: finite(entry.reward_mean),
            reward_min: finite(entry.reward_min),
            reward_max: finite(entry.reward_max),
            texts,
        }
    }
}

schema! {
/// `GET /v1/runs/{id}/trajectories/updates/{update}`: the groups of one
/// update, members without their texts.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct UpdateDetail {
    pub summary: UpdateSummary,
    /// Empty for an update whose texts were not exported.
    pub groups: Vec<GroupSummary>,
    /// Pass back as `?cursor=` for the next page of groups.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}
}

schema! {
/// One group of an update. PPO has no groups: its rollouts are one group
/// whose `group` is absent, addressed as `-`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GroupSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<usize>,
    pub prompt: PromptView,
    /// Sorted by member index.
    pub members: Vec<MemberSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_mean: Option<f32>,
    /// Population standard deviation of the rewards. Zero means the group
    /// carried no signal to learn from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_std: Option<f32>,
    /// Members an `outcome` confirmed trained.
    pub trained: usize,
}
}

schema! {
/// A prompt, whole or cut to a preview.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PromptView {
    pub key: String,
    pub messages: Vec<MessageView>,
    /// True when a message was cut to `preview_chars`.
    pub cut: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<Object>))]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MessageView {
    /// `system` | `user` | `assistant` | `tool`.
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallView>,
    /// On a `tool` message: the call it answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub is_error: bool,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ToolCallView {
    pub id: String,
    pub name: String,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub arguments: serde_json::Value,
}
}

impl MessageView {
    /// `limit` cuts the content to that many characters; `None` keeps it.
    pub fn from_read(
        message: retrograd_observe::reader::Message,
        limit: Option<usize>,
    ) -> (Self, bool) {
        let (content, cut) = match limit {
            Some(limit) => preview(message.content, limit),
            None => (message.content, false),
        };
        (
            Self {
                role: message.role,
                content,
                tool_calls: message
                    .tool_calls
                    .into_iter()
                    .map(|call| ToolCallView {
                        id: call.id,
                        name: call.name,
                        arguments: call.arguments,
                    })
                    .collect(),
                tool_call_id: message.tool_call_id,
                is_error: message.is_error,
            },
            cut,
        )
    }
}

impl PromptView {
    pub fn from_read(
        prompt: retrograd_observe::reader::PromptRecord,
        limit: Option<usize>,
    ) -> Self {
        let mut cut = false;
        let messages = prompt
            .messages
            .into_iter()
            .map(|message| {
                let (view, was_cut) = MessageView::from_read(message, limit);
                cut |= was_cut;
                view
            })
            .collect();
        Self {
            key: prompt.key,
            messages,
            cut,
            reward_text: prompt.reward_text,
            metadata: prompt.metadata,
        }
    }

    /// A prompt whose record is not in the export: its key, nothing else.
    pub fn missing(key: String) -> Self {
        Self {
            key,
            messages: Vec::new(),
            cut: false,
            reward_text: None,
            metadata: None,
        }
    }
}

/// The first `limit` characters of `text`, and whether anything was cut.
fn preview(text: String, limit: usize) -> (String, bool) {
    match text.char_indices().nth(limit) {
        Some((at, _)) => (text[..at].to_string(), true),
        None => (text, false),
    }
}

schema! {
/// One rollout, without its text.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MemberSummary {
    pub member: usize,
    pub seed: u64,
    pub tokens: usize,
    pub truncated: bool,
    /// What the optimizer was handed, after shaping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    /// Before the overlong penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_raw: Option<f32>,
    /// The judge's share of `reward`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_term: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage_max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eligible: Option<bool>,
    /// True once an `outcome` confirms it, false for an exclusion, absent
    /// when the run stopped before saying.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trained: Option<bool>,
    /// `truncated` | `zero_signal` | `judge_dropped` | `unscored` |
    /// `update_skipped`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
    /// Assistant turns of a conversation; zero for a completion.
    pub turns: usize,
    pub tool_calls: usize,
    /// Tool results that reported an error.
    pub tool_errors: usize,
}
}

impl From<retrograd_observe::reader::MemberEntry> for MemberSummary {
    fn from(entry: retrograd_observe::reader::MemberEntry) -> Self {
        Self {
            member: entry.member,
            seed: entry.seed,
            tokens: entry.tokens,
            truncated: entry.truncated,
            reward: finite(entry.reward),
            reward_raw: finite(entry.reward_raw),
            judge_term: finite(entry.judge_term),
            advantage: finite(entry.advantage),
            advantage_min: finite(entry.advantage_min),
            advantage_max: finite(entry.advantage_max),
            eligible: entry.eligible,
            trained: entry.trained,
            skip_reason: entry.skip_reason,
            turns: entry.turns,
            tool_calls: entry.tool_calls,
            tool_errors: entry.tool_errors,
        }
    }
}

impl GroupSummary {
    pub fn new(group: Option<usize>, prompt: PromptView, members: Vec<MemberSummary>) -> Self {
        let rewards: Vec<f32> = members.iter().filter_map(|member| member.reward).collect();
        // A group is a handful of members: the count is exact in f32.
        let count = rewards.len() as f32;
        let reward_mean = (!rewards.is_empty()).then(|| rewards.iter().sum::<f32>() / count);
        let reward_std = reward_mean.map(|mean| {
            (rewards
                .iter()
                .map(|reward| (reward - mean).powi(2))
                .sum::<f32>()
                / count)
                .sqrt()
        });
        Self {
            group,
            prompt,
            trained: members
                .iter()
                .filter(|member| member.trained == Some(true))
                .count(),
            members,
            reward_mean: finite(reward_mean),
            reward_std: finite(reward_std),
        }
    }
}

schema! {
/// `GET /v1/runs/{id}/trajectories/updates/{update}/groups/{group}`: one group
/// with every text.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GroupDetail {
    pub update: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<usize>,
    /// Whole.
    pub prompt: PromptView,
    pub members: Vec<MemberDetail>,
}
}

schema! {
/// One rollout with its text: a `completion`, or a `conversation`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MemberDetail {
    pub summary: MemberSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
}
}

schema! {
/// A trajectory: the messages the policy and its tools exchanged.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Conversation {
    pub messages: Vec<MessageView>,
    /// The messages start after the scenario's own prefix.
    pub prefix: bool,
    pub step_rewards: Vec<StepRewardView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_reward_raw: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_explanation: Option<String>,
    /// The entries that differ from the scenario's.
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct StepRewardView {
    pub step_index: usize,
    pub kind: String,
    pub reward: f32,
    /// Indices into `messages`; empty when unknown.
    pub message_indices: Vec<usize>,
}
}

impl From<retrograd_observe::reader::RolloutRecord> for MemberDetail {
    fn from(record: retrograd_observe::reader::RolloutRecord) -> Self {
        use retrograd_observe::reader::RolloutText;

        let summary = MemberSummary::from(record.entry);
        match record.text {
            RolloutText::Completion(completion) => Self {
                summary,
                completion: Some(completion),
                conversation: None,
            },
            RolloutText::Conversation {
                messages,
                prefix,
                step_rewards,
                terminal_reward_raw,
                judge_explanation,
                metadata,
            } => Self {
                summary,
                completion: None,
                conversation: Some(Conversation {
                    messages: messages
                        .into_iter()
                        .map(|message| MessageView::from_read(message, None).0)
                        .collect(),
                    prefix,
                    step_rewards: step_rewards
                        .into_iter()
                        .filter(|step| step.reward.is_finite())
                        .map(|step| StepRewardView {
                            step_index: step.step_index,
                            kind: step.kind,
                            reward: step.reward,
                            message_indices: step.message_indices,
                        })
                        .collect(),
                    terminal_reward_raw: finite(terminal_reward_raw),
                    judge_explanation,
                    metadata,
                }),
            },
        }
    }
}
