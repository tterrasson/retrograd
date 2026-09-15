use serde::{Deserialize, Serialize};

use crate::tools::ToolCall;
use crate::{Error, Result};

retrograd_core::wire_enum! {
    /// Who a message comes from. The spelling beside each variant is both what
    /// serde reads and writes and what `as_str()` returns.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum Role: serde {
        System = "system",
        User = "user",
        Assistant = "assistant",
        Tool = "tool",
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For `Role::Tool`: the observation reports a failed call. Carried as a
    /// flag rather than sniffed back out of `content`, which cannot separate a
    /// real failure from a successful result that merely mentions one.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl Message {
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            is_error: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    PolicyAction,
    ToolResult,
    UserTurn,
    Context,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub kind: StepKind,
    pub token_range: (usize, usize),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trajectory {
    pub scenario_id: String,
    pub messages: Vec<Message>,
    pub tokens: Vec<i32>,
    pub old_logprobs: Vec<f32>,
    pub train_mask: Vec<bool>,
    pub steps: Vec<Step>,
    pub reward: Option<f32>,
    pub truncated: bool,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl Trajectory {
    pub fn validate(&self) -> Result<()> {
        if self.scenario_id.trim().is_empty() {
            return Err(Error::invalid("trajectory scenario_id must not be empty"));
        }
        if self.tokens.len() != self.train_mask.len() {
            return Err(Error::invalid(
                "trajectory tokens and train_mask lengths do not match",
            ));
        }
        if self.tokens.is_empty() || self.train_mask[0] {
            return Err(Error::invalid(
                "trajectory must contain an untrained first context token",
            ));
        }
        let trained = self.train_mask.iter().filter(|&&train| train).count();
        if trained == 0 || self.old_logprobs.len() != trained {
            return Err(Error::invalid(
                "trajectory old_logprobs must align with a non-empty train_mask",
            ));
        }
        if self.old_logprobs.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid("trajectory old_logprobs must all be finite"));
        }
        if self.reward.is_some_and(|reward| !reward.is_finite()) {
            return Err(Error::invalid("trajectory reward must be finite"));
        }

        let mut cursor = 0;
        for step in &self.steps {
            let (start, end) = step.token_range;
            if start != cursor || end <= start || end > self.tokens.len() {
                return Err(Error::invalid(
                    "trajectory steps must partition the token sequence",
                ));
            }
            let expected = step.kind == StepKind::PolicyAction;
            if self.train_mask[start..end]
                .iter()
                .any(|&train| train != expected)
            {
                return Err(Error::invalid(
                    "only PolicyAction steps may contain trainable tokens",
                ));
            }
            if step.reward.is_some_and(|reward| !reward.is_finite()) {
                return Err(Error::invalid("trajectory step reward must be finite"));
            }
            cursor = end;
        }
        if cursor != self.tokens.len() {
            return Err(Error::invalid(
                "trajectory steps do not cover the full token sequence",
            ));
        }
        Ok(())
    }

    /// Final reward plus every step reward: the quantity GRPO centers a group
    /// on, and the one a truncation policy has to reason about. Turning it into
    /// a `TrainSequence` lives in `retrograd-agent` - that conversion is the
    /// only part of a trajectory that needs the training crate.
    pub fn total_reward(&self) -> f32 {
        self.reward.unwrap_or(0.0) + self.step_reward_sum()
    }

    pub fn step_reward_sum(&self) -> f32 {
        self.steps
            .iter()
            .filter_map(|step| step.reward)
            .sum::<f32>()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryGroup {
    pub group_id: u64,
    pub scenario_id: String,
    pub trajectories: Vec<Trajectory>,
}

impl TrajectoryGroup {
    /// A group is one scenario judged relative to itself, so a member from
    /// another scenario would silently poison the baseline. Checked here rather
    /// than at the training boundary, because every consumer - judge included -
    /// relies on it.
    pub fn validate(&self) -> Result<()> {
        if self.trajectories.is_empty() {
            return Err(Error::invalid("trajectory group must not be empty"));
        }
        for trajectory in &self.trajectories {
            if trajectory.scenario_id != self.scenario_id {
                return Err(Error::invalid(
                    "trajectory group contains a different scenario_id",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_enforces_step_partition_and_policy_mask() {
        let trajectory = Trajectory {
            scenario_id: "s1".into(),
            messages: vec![],
            tokens: vec![1, 2, 3, 4, 5],
            old_logprobs: vec![-0.1, -0.2],
            train_mask: vec![false, false, true, false, true],
            steps: vec![
                Step {
                    kind: StepKind::Context,
                    token_range: (0, 2),
                    reward: None,
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (2, 3),
                    reward: None,
                },
                Step {
                    kind: StepKind::ToolResult,
                    token_range: (3, 4),
                    reward: Some(2.0),
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (4, 5),
                    reward: None,
                },
            ],
            reward: Some(1.0),
            truncated: false,
            metadata: Default::default(),
        };
        trajectory.validate().unwrap();
        // Final reward plus the graded step: what a group is centered on.
        assert_eq!(trajectory.total_reward(), 3.0);

        let mut invalid = trajectory;
        invalid.train_mask[3] = true;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn tool_observations_stay_untrained_and_trajectories_serialize() {
        let trajectory = Trajectory {
            scenario_id: "s".into(),
            messages: vec![Message::text(Role::Tool, "observation")],
            tokens: vec![1, 2, 3],
            old_logprobs: vec![-0.3],
            train_mask: vec![false, false, true],
            steps: vec![
                Step {
                    kind: StepKind::ToolResult,
                    token_range: (0, 2),
                    reward: None,
                },
                Step {
                    kind: StepKind::PolicyAction,
                    token_range: (2, 3),
                    reward: Some(0.25),
                },
            ],
            reward: Some(0.5),
            truncated: false,
            metadata: Default::default(),
        };
        trajectory.validate().unwrap();
        let decoded: Trajectory =
            serde_json::from_str(&serde_json::to_string(&trajectory).unwrap()).unwrap();
        assert_eq!(decoded.train_mask, [false, false, true]);
        assert_eq!(decoded.total_reward(), 0.75);
    }

    #[test]
    fn a_group_refuses_a_member_from_another_scenario() {
        let member = |scenario_id: &str| Trajectory {
            scenario_id: scenario_id.into(),
            messages: vec![],
            tokens: vec![1, 2],
            old_logprobs: vec![-0.1],
            train_mask: vec![false, true],
            steps: vec![],
            reward: Some(0.5),
            truncated: false,
            metadata: Default::default(),
        };
        let mut group = TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: vec![member("s"), member("s")],
        };
        group.validate().unwrap();

        // A stray member would be centered against a baseline that does not
        // describe its own task.
        group.trajectories[1].scenario_id = "other".into();
        assert!(group.validate().is_err());

        group.trajectories.clear();
        assert!(group.validate().is_err());
    }
}
