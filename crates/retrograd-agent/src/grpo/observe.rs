//! What an agentic update hands to a [`TrajectoryObserver`].
//!
//! A draft is taken of every trajectory at collection, before anything is
//! filtered; the scoring and the selection then fill it in by `(group,
//! member)`, which the trajectory carries from the engine.

use std::collections::{BTreeMap, HashMap};

use retrograd_observe::{
    ObserveBatch, ObservedMessage, ObservedPrompt, ObservedRollout, ObservedToolCall, RolloutBatch,
    RolloutContent, SkipReason, StepReward,
};
use serde_json::{Map, Value};

use super::selection::UpdateFate;
use crate::trajectory::{Message, Role, StepKind, Trajectory, TrajectoryGroup};
use retrograd_agent_core::scenario::Scenario;

const JUDGE_EXPLANATION: &str = "judge_explanation";

pub(super) fn scenario_key(scenario: &Scenario) -> String {
    format!("s:{}", scenario.id)
}

/// The messages every trajectory of `scenario` opens with.
fn scenario_prefix(scenario: &Scenario) -> Vec<Message> {
    scenario
        .system
        .iter()
        .map(|system| Message::text(Role::System, system))
        .chain([Message::text(Role::User, &scenario.user)])
        .collect()
}

pub(super) fn observed_scenario(scenario: &Scenario) -> ObservedPrompt {
    ObservedPrompt {
        key: scenario_key(scenario),
        messages: scenario_prefix(scenario)
            .iter()
            .map(observed_message)
            .collect(),
        reward_text: None,
        metadata: Some(scenario.metadata.clone()),
    }
}

fn observed_message(message: &Message) -> ObservedMessage {
    ObservedMessage {
        role: message.role.as_str().to_string(),
        content: message.content.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| ObservedToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect(),
        tool_call_id: message.tool_call_id.clone(),
        is_error: message.is_error,
    }
}

fn step_kind(kind: StepKind) -> &'static str {
    match kind {
        StepKind::PolicyAction => "policy_action",
        StepKind::ToolResult => "tool_result",
        StepKind::UserTurn => "user_turn",
        StepKind::Context => "context",
    }
}

/// Entries the trajectory holds that its scenario does not, or holds with
/// another value. The judge's explanation has a field of its own.
fn metadata_changes(trajectory: &Trajectory, scenario: &Scenario) -> Map<String, Value> {
    trajectory
        .metadata
        .iter()
        .filter(|(key, value)| {
            key.as_str() != JUDGE_EXPLANATION && scenario.metadata.get(key.as_str()) != Some(value)
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn judge_explanation(trajectory: &Trajectory) -> Option<String> {
    trajectory
        .metadata
        .get(JUDGE_EXPLANATION)
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// The draft of one collected trajectory, or `None` when the engine did not
/// record who it is.
pub(super) fn draft(
    update: u32,
    group: usize,
    scenario: &Scenario,
    trajectory: &Trajectory,
) -> Option<ObservedRollout> {
    let provenance = trajectory.provenance.as_ref()?;
    let expected = scenario_prefix(scenario);
    let prefix = trajectory.messages.starts_with(&expected);
    let skipped = if prefix { expected.len() } else { 0 };
    let step_rewards = trajectory
        .steps
        .iter()
        .enumerate()
        .filter_map(|(step_index, step)| {
            let reward = step.reward?;
            let message_indices = provenance
                .step_messages
                .get(step_index)
                .map(|indices| {
                    indices
                        .iter()
                        .filter_map(|index| index.checked_sub(skipped))
                        .collect()
                })
                .unwrap_or_default();
            Some(StepReward {
                step_index,
                kind: step_kind(step.kind).to_string(),
                reward,
                message_indices,
            })
        })
        .collect();
    Some(ObservedRollout {
        update,
        group: Some(group),
        member: provenance.member,
        prompt: scenario_key(scenario),
        seed: provenance.seed,
        tokens: trajectory.train_mask.iter().filter(|&&train| train).count(),
        truncated: trajectory.truncated,
        reward: None,
        reward_raw: trajectory.reward.map(|_| trajectory.total_reward()),
        judge_term: None,
        advantage: None,
        advantage_min: None,
        advantage_max: None,
        eligible: None,
        trained: None,
        skip_reason: None,
        content: RolloutContent::Conversation {
            messages: trajectory.messages[skipped..]
                .iter()
                .map(observed_message)
                .collect(),
            prefix,
            step_rewards,
            terminal_reward_raw: trajectory.reward,
            judge_explanation: judge_explanation(trajectory),
            metadata: metadata_changes(trajectory, scenario),
        },
    })
}

/// One update's export, filled in as the update goes.
pub(super) struct UpdateExport<'a> {
    /// By group index, the scenario the group was collected on.
    scenarios: Vec<&'a Scenario>,
    group_of: HashMap<u64, usize>,
    rollouts: BTreeMap<(usize, usize), ObservedRollout>,
}

impl<'a> UpdateExport<'a> {
    pub(super) fn new(scenarios: Vec<&'a Scenario>) -> Self {
        Self {
            scenarios,
            group_of: HashMap::new(),
            rollouts: BTreeMap::new(),
        }
    }

    pub(super) fn collected(
        &mut self,
        group: usize,
        group_id: Option<u64>,
        drafts: Vec<ObservedRollout>,
    ) {
        if let Some(group_id) = group_id {
            self.group_of.insert(group_id, group);
        }
        for draft in drafts {
            self.rollouts.insert((group, draft.member), draft);
        }
    }

    fn identity(&self, group_id: u64, trajectory: &Trajectory) -> Option<(usize, usize)> {
        Some((
            *self.group_of.get(&group_id)?,
            trajectory.provenance.as_ref()?.member,
        ))
    }

    /// Takes the judge's rewards, explanations and metadata, before the groups
    /// it dropped are filtered out.
    pub(super) fn judged(&mut self, groups: &[TrajectoryGroup]) {
        for group in groups {
            for trajectory in &group.trajectories {
                let Some((index, member)) = self.identity(group.group_id, trajectory) else {
                    continue;
                };
                let scenario = self.scenarios[index];
                let Some(draft) = self.rollouts.get_mut(&(index, member)) else {
                    continue;
                };
                draft.reward_raw = trajectory.reward.map(|_| trajectory.total_reward());
                if let RolloutContent::Conversation {
                    terminal_reward_raw,
                    judge_explanation: explanation,
                    metadata,
                    ..
                } = &mut draft.content
                {
                    *terminal_reward_raw = trajectory.reward;
                    *explanation = judge_explanation(trajectory);
                    *metadata = metadata_changes(trajectory, scenario);
                }
            }
        }
    }

    /// Settles every draft against the groups that survived selection and
    /// the update's fate. `None` is a fate that could not be decided: the
    /// survivors are left unknown.
    pub(super) fn settle(&mut self, survivors: &[TrajectoryGroup], fate: Option<&UpdateFate>) {
        let mut kept = HashMap::new();
        for group in survivors {
            for trajectory in &group.trajectories {
                if let Some(identity) = self.identity(group.group_id, trajectory) {
                    kept.insert(identity, trajectory.total_reward());
                }
            }
        }
        for (identity, draft) in &mut self.rollouts {
            match (kept.get(identity), fate) {
                (Some(&total), fate) => {
                    draft.reward = Some(total);
                    if let Some(UpdateFate::Skip(_)) = fate {
                        draft.eligible = Some(false);
                        draft.trained = Some(false);
                        draft.skip_reason = Some(SkipReason::UpdateSkipped);
                    }
                }
                (None, _) => {
                    draft.eligible = Some(false);
                    draft.trained = Some(false);
                    draft.skip_reason = Some(if draft.truncated {
                        SkipReason::Truncated
                    } else {
                        SkipReason::Unscored
                    });
                }
            }
        }
    }

    /// `(group, member)` of each training sequence, in the order
    /// `groups_to_train_sequences` builds them. `None` when one is unknown.
    pub(super) fn members(&self, groups: &[TrajectoryGroup]) -> Option<Vec<(usize, usize)>> {
        groups
            .iter()
            .flat_map(|group| {
                group
                    .trajectories
                    .iter()
                    .map(|trajectory| self.identity(group.group_id, trajectory))
            })
            .collect()
    }

    pub(super) fn batch(self) -> ObserveBatch {
        let mut groups = self
            .rollouts
            .keys()
            .map(|(group, _)| *group)
            .collect::<Vec<_>>();
        groups.dedup();
        let mut seen = std::collections::BTreeSet::new();
        let prompts = groups
            .into_iter()
            .map(|group| self.scenarios[group])
            .filter(|scenario| seen.insert(scenario.id.as_str()))
            .map(observed_scenario)
            .collect();
        ObserveBatch::Rollouts(RolloutBatch {
            prompts,
            rollouts: self.rollouts.into_values().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Provenance, Step};
    use retrograd_agent_core::tools::ToolCall;

    fn scenario() -> Scenario {
        let mut metadata = Map::new();
        metadata.insert("rubric".into(), "be brief".into());
        Scenario {
            id: "task".into(),
            system: Some("system".into()),
            user: "question".into(),
            metadata,
        }
    }

    /// system, user, assistant with two calls, two tool results, assistant.
    fn trajectory(member: usize, truncated: bool) -> Trajectory {
        let scenario = scenario();
        let call = |id: &str, name: &str| ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({"path": "a"}),
        };
        let result = |id: &str| {
            let mut message = Message::text(Role::Tool, format!("result {id}"));
            message.tool_call_id = Some(id.into());
            message
        };
        let mut metadata = scenario.metadata.clone();
        metadata.insert("env_state".into(), "done".into());
        Trajectory {
            scenario_id: scenario.id.clone(),
            messages: vec![
                Message::text(Role::System, "system"),
                Message::text(Role::User, "question"),
                Message {
                    tool_calls: vec![call("c1", "read"), call("c2", "list")],
                    ..Message::text(Role::Assistant, "calling")
                },
                result("c1"),
                result("c2"),
                Message::text(Role::Assistant, "done"),
            ],
            tokens: vec![1, 2, 3, 4, 5],
            old_logprobs: vec![-1.0, -1.0],
            train_mask: vec![false, true, false, true, false],
            steps: vec![
                Step {
                    kind: StepKind::Context,
                    token_range: (0, 1),
                    reward: None,
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (1, 2),
                    reward: None,
                },
                Step {
                    kind: StepKind::ToolResult,
                    token_range: (2, 3),
                    reward: Some(0.5),
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (3, 4),
                    reward: None,
                },
                Step {
                    kind: StepKind::UserTurn,
                    token_range: (4, 5),
                    reward: Some(-0.25),
                },
            ],
            reward: Some(1.0),
            truncated,
            metadata,
            provenance: Some(Provenance {
                member,
                seed: 40 + member as u64,
                step_messages: vec![vec![0, 1], vec![2], vec![3, 4], vec![5], vec![]],
            }),
        }
    }

    fn group(group_id: u64, members: &[(usize, bool)]) -> TrajectoryGroup {
        TrajectoryGroup {
            group_id,
            scenario_id: "task".into(),
            trajectories: members
                .iter()
                .map(|&(member, truncated)| trajectory(member, truncated))
                .collect(),
        }
    }

    #[test]
    fn a_draft_drops_the_prefix_and_keeps_every_step_on_its_messages() {
        let draft = draft(4, 1, &scenario(), &trajectory(2, false)).unwrap();
        assert_eq!((draft.group, draft.member, draft.seed), (Some(1), 2, 42));
        assert_eq!(draft.prompt, "s:task");
        assert_eq!(draft.tokens, 2);
        assert_eq!(draft.reward, None, "no training total before selection");
        assert_eq!(draft.reward_raw, Some(1.25));
        assert_eq!(draft.eligible, None);
        let RolloutContent::Conversation {
            messages,
            prefix,
            step_rewards,
            terminal_reward_raw,
            metadata,
            ..
        } = &draft.content
        else {
            panic!("an agentic draft is a conversation");
        };
        assert!(prefix);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].tool_calls.len(), 2);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("c2"));
        assert_eq!(*terminal_reward_raw, Some(1.0));
        // Two tool results for one step, indexed after the prefix; a step with
        // no recorded message stays unattributed.
        assert_eq!(step_rewards[0].step_index, 2);
        assert_eq!(step_rewards[0].kind, "tool_result");
        assert_eq!(step_rewards[0].message_indices, [1, 2]);
        assert_eq!(step_rewards[1].step_index, 4);
        assert!(step_rewards[1].message_indices.is_empty());
        assert_eq!(metadata.len(), 1, "{metadata:?}");
        assert_eq!(metadata["env_state"], "done");
    }

    #[test]
    fn a_conversation_that_does_not_open_on_its_scenario_is_kept_whole() {
        let mut trajectory = trajectory(0, false);
        trajectory.messages[0].content = "system with a tool catalog".into();
        let draft = draft(1, 0, &scenario(), &trajectory).unwrap();
        let RolloutContent::Conversation {
            messages,
            prefix,
            step_rewards,
            ..
        } = &draft.content
        else {
            panic!("an agentic draft is a conversation");
        };
        assert!(!prefix);
        assert_eq!(messages.len(), 6);
        assert_eq!(step_rewards[0].message_indices, [3, 4]);
    }

    #[test]
    fn an_unrecorded_trajectory_is_not_guessed() {
        let mut trajectory = trajectory(0, false);
        trajectory.provenance = None;
        assert!(draft(1, 0, &scenario(), &trajectory).is_none());
    }

    fn export<'a>(scenario: &'a Scenario, groups: &[TrajectoryGroup]) -> UpdateExport<'a> {
        let mut export = UpdateExport::new(vec![scenario, scenario]);
        for (index, group) in groups.iter().enumerate() {
            let drafts = group
                .trajectories
                .iter()
                .filter_map(|trajectory| draft(3, index, scenario, trajectory))
                .collect();
            export.collected(index, Some(group.group_id), drafts);
        }
        export
    }

    #[test]
    fn identities_survive_filtering_and_reordering() {
        let scenario = scenario();
        let collected = [
            group(10, &[(0, false), (1, true), (2, false)]),
            group(20, &[(0, false), (1, false)]),
        ];
        let mut export = export(&scenario, &collected);

        // The second group was dropped; the truncated member of the first came
        // back last, as `MinReward` puts it.
        let mut survivors = vec![group(10, &[(0, false), (2, false), (1, true)])];
        survivors[0].trajectories[2].reward = Some(-3.0);
        assert_eq!(
            export.members(&survivors),
            Some(vec![(0, 0), (0, 2), (0, 1)])
        );
        export.settle(&survivors, Some(&UpdateFate::Train));
        let ObserveBatch::Rollouts(batch) = export.batch() else {
            panic!("rollouts");
        };
        assert_eq!(batch.prompts.len(), 1, "one scenario, one prompt record");
        let by_identity = batch
            .rollouts
            .iter()
            .map(|rollout| ((rollout.group.unwrap(), rollout.member), rollout))
            .collect::<HashMap<_, _>>();
        let restored = by_identity[&(0, 1)];
        assert!(restored.truncated);
        assert_eq!(restored.skip_reason, None, "MinReward keeps it");
        assert_eq!(restored.reward, Some(-3.0 + 0.25));
        assert_eq!(restored.reward_raw, Some(1.25));
        assert_eq!(restored.eligible, None);
        let dropped = by_identity[&(1, 0)];
        assert_eq!(dropped.skip_reason, Some(SkipReason::Unscored));
        assert_eq!(dropped.trained, Some(false));
        assert_eq!(dropped.reward, None);
    }

    #[test]
    fn a_truncated_member_keeps_its_cause_and_a_skip_covers_the_rest() {
        let scenario = scenario();
        let collected = [group(10, &[(0, false), (1, true)])];
        let mut export = export(&scenario, &collected);
        let survivors = vec![group(10, &[(0, false)])];
        export.settle(
            &survivors,
            Some(&UpdateFate::Skip("nothing to train".into())),
        );
        let ObserveBatch::Rollouts(batch) = export.batch() else {
            panic!("rollouts");
        };
        assert_eq!(
            batch.rollouts[0].skip_reason,
            Some(SkipReason::UpdateSkipped)
        );
        assert_eq!(batch.rollouts[0].reward, Some(1.25));
        assert_eq!(batch.rollouts[1].skip_reason, Some(SkipReason::Truncated));
    }

    #[test]
    fn an_undecided_fate_leaves_the_survivors_unknown() {
        let scenario = scenario();
        let collected = [group(10, &[(0, false), (1, false)])];
        let mut export = export(&scenario, &collected);
        export.settle(&collected, None);
        let ObserveBatch::Rollouts(batch) = export.batch() else {
            panic!("rollouts");
        };
        assert!(
            batch
                .rollouts
                .iter()
                .all(|rollout| rollout.eligible.is_none()
                    && rollout.trained.is_none()
                    && rollout.skip_reason.is_none())
        );
    }

    #[test]
    fn the_judge_verdict_replaces_the_raw_reward_and_brings_its_explanation() {
        let scenario = scenario();
        let mut collected = [group(10, &[(0, false), (1, false)])];
        for trajectory in &mut collected[0].trajectories {
            trajectory.reward = None;
        }
        let mut export = export(&scenario, &collected);
        let mut judged = collected.clone();
        judged[0].trajectories[1].reward = Some(0.75);
        judged[0].trajectories[1]
            .metadata
            .insert(JUDGE_EXPLANATION.into(), "fine".into());
        export.judged(&judged);
        let ObserveBatch::Rollouts(batch) = export.batch() else {
            panic!("rollouts");
        };
        assert_eq!(batch.rollouts[0].reward_raw, None, "unscored stays null");
        assert_eq!(batch.rollouts[1].reward_raw, Some(1.0));
        let RolloutContent::Conversation {
            judge_explanation,
            metadata,
            terminal_reward_raw,
            ..
        } = &batch.rollouts[1].content
        else {
            panic!("conversation");
        };
        assert_eq!(judge_explanation.as_deref(), Some("fine"));
        assert_eq!(*terminal_reward_raw, Some(0.75));
        assert!(!metadata.contains_key(JUDGE_EXPLANATION));
    }
}
