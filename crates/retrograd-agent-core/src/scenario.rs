//! What a rollout is asked to do, and what bounds it.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub user: String,
    /// Free-form, and the extension point every environment reads: a task
    /// declaration (files, setup, verification), a per-scenario judge rubric.
    /// It is copied verbatim onto every trajectory the scenario produces, so
    /// whatever is put here reaches the judge too.
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl Scenario {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(Error::invalid("scenario id must not be empty"));
        }
        if self.user.is_empty() {
            return Err(Error::invalid("scenario user message must not be empty"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RolloutLimits {
    pub max_turns: usize,
    pub max_new_tokens_per_turn: u32,
    pub max_trajectory_tokens: usize,
    /// Wall-clock budget for a whole rollout. A tool server that hangs past its
    /// own per-call timeout, or a very long multi-turn trajectory, would
    /// otherwise stall an update indefinitely. Overrunning truncates the
    /// trajectory - the same policy as overrunning the token budget, not an
    /// error. Checked at turn boundaries, so a single turn may overshoot it.
    /// Zero disables the deadline.
    pub max_rollout_secs: u64,
    /// What a turn that named no tool at all means.
    ///
    /// `true` - the default, and the only reading that is right for a
    /// question-answering run: the policy stopped calling tools because it has
    /// its answer, and the generation *is* the trajectory's last turn.
    ///
    /// `false` is for a task the policy does not get to declare finished - a
    /// world with a terminal state, which says `done` itself. There, a turn of
    /// prose is a wasted turn and nothing more, so it comes back as an error
    /// observation and the episode goes on. What that changes is not politeness
    /// but the sign of the gradient: ending on the first unparsed turn makes
    /// "answer in prose and stop" the *cheapest* trajectory of its group,
    /// it banks no step rewards, so it beats every member that engaged the
    /// environment and paid for its moves, and GRPO's intra-group baseline
    /// duly trains the policy out of calling tools at all.
    ///
    /// It is not free: a policy that never calls anything now decodes
    /// `max_turns` turns instead of one, so a run that turns this on should
    /// expect `timing/rollout_seconds` to grow with the fraction of turns that
    /// parse nothing, and should watch `agent/tool_calls_per_turn` rather than
    /// wait for the reward to move.
    pub end_on_no_tool_call: bool,
    /// Consecutive turns without a valid tool call after which the trajectory
    /// is cut, as a truncation. Zero - the default - never cuts.
    ///
    /// Under `end_on_no_tool_call = false` a turn of prose or a malformed call
    /// is an error observation and the episode goes on, which is right for the
    /// gradient and expensive for the clock: a policy that never gets the
    /// format right decodes `max_turns` turns of nothing. This is the early
    /// exit for that case. The trajectory ends the way a turn-budget overrun
    /// does - `truncated`, so `truncation` prices it, and `min_reward` puts it
    /// at the bottom of its group where it already belonged - it just gets
    /// there after N wasted turns instead of thirty. Consecutive, not total: one
    /// valid call, parsed or not stepped, resets the count, because a policy
    /// that is calling tools is playing, however badly.
    pub max_failed_turns: usize,
}

impl Default for RolloutLimits {
    fn default() -> Self {
        Self {
            max_turns: 6,
            max_new_tokens_per_turn: 512,
            max_trajectory_tokens: 4096,
            max_rollout_secs: 300,
            end_on_no_tool_call: true,
            max_failed_turns: 0,
        }
    }
}

impl RolloutLimits {
    pub fn validate(&self) -> Result<()> {
        if self.max_turns == 0
            || self.max_new_tokens_per_turn == 0
            || self.max_trajectory_tokens < 2
        {
            return Err(Error::invalid(
                "rollout limits must allow at least one turn, one generated token, and two total tokens",
            ));
        }
        if self.max_trajectory_tokens > u32::MAX as usize {
            return Err(Error::invalid(
                "max_trajectory_tokens must fit in a 32-bit generation budget",
            ));
        }
        Ok(())
    }
}

/// What becomes of a member that ran out of budget mid-response.
///
/// Neither branch is free. A partial response cannot be judged as a completed
/// trajectory, so `Drop` is the honest reading of the reward - but dropping is a
/// selection on length: the model never receives a negative signal for
/// overrunning, and a task that systematically overruns disappears from the
/// gradient entirely rather than being learned out of. `MinReward` keeps the
/// member at the bottom of its own group, which makes overrunning cost
/// something at the price of a reward nobody measured.
///
/// Watch `agent/truncated_fraction` before switching: past roughly 20% the
/// training distribution is no longer the one the scenarios describe, and that
/// is the point at which the trade above stops being academic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationPolicy {
    /// Truncated members never reach the optimizer. They still count against
    /// `max_dropped_fraction`.
    #[default]
    Drop,
    /// Truncated members train with the smallest total reward of their group.
    MinReward,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_defaults_round_trip_and_rejects_invalid_input() {
        let scenario: Scenario = serde_json::from_str(r#"{"id":"s","user":"hello"}"#).unwrap();
        assert_eq!(scenario.system, None);
        assert!(scenario.metadata.is_empty());
        assert!(scenario.validate().is_ok());

        for source in [r#"{"id":" ","user":"hello"}"#, r#"{"id":"s","user":""}"#] {
            let invalid: Scenario = serde_json::from_str(source).unwrap();
            assert!(invalid.validate().is_err(), "accepted {source}");
        }
        assert!(serde_json::from_str::<Scenario>(r#"{"id":"s","user":"x","extra":1}"#).is_err());
    }

    #[test]
    fn rollout_limits_reject_a_budget_nothing_can_run_in() {
        assert!(RolloutLimits::default().validate().is_ok());
        for limits in [
            RolloutLimits {
                max_turns: 0,
                ..Default::default()
            },
            RolloutLimits {
                max_new_tokens_per_turn: 0,
                ..Default::default()
            },
            RolloutLimits {
                max_trajectory_tokens: 1,
                ..Default::default()
            },
            RolloutLimits {
                max_trajectory_tokens: u32::MAX as usize + 1,
                ..Default::default()
            },
        ] {
            assert!(limits.validate().is_err());
        }
    }
}
