//! The `agent/*` metrics of one update.

use retrograd_metrics::MetricValue;

use crate::rollout::RolloutFailures;
use crate::trajectory::{Role, TrajectoryGroup};

/// Counter ratio as a metric value; zero when nothing was counted, which is the
/// honest reading for "this path did not run" on a series that is otherwise a
/// share.
pub(super) fn ratio_or_zero(numerator: u64, denominator: u64) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f32 / denominator as f32
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AgentMetricTotals {
    trajectories: usize,
    tool_calls: usize,
    tool_results: usize,
    tool_errors: usize,
    turns: usize,
    tokens: usize,
    truncated: usize,
}

impl AgentMetricTotals {
    /// Parsed tool calls of the groups that kept a GRPO baseline. Zero over a
    /// whole update, with a non-empty catalog, is one symptom of a model calling
    /// tools in a format the parser does not read - see
    /// [`tool_call_parse_warning`](super::selection::tool_call_parse_warning).
    pub(super) fn tool_calls(&self) -> usize {
        self.tool_calls
    }

    pub(super) fn observe(&mut self, group: &TrajectoryGroup) {
        for trajectory in &group.trajectories {
            self.trajectories += 1;
            self.tool_calls += trajectory
                .messages
                .iter()
                .map(|message| message.tool_calls.len())
                .sum::<usize>();
            for message in trajectory
                .messages
                .iter()
                .filter(|message| message.role == Role::Tool)
            {
                self.tool_results += 1;
                self.tool_errors += usize::from(message.is_error);
            }
            self.turns += trajectory
                .messages
                .iter()
                .filter(|message| message.role == Role::Assistant)
                .count();
            self.tokens += trajectory.tokens.len();
            self.truncated += usize::from(trajectory.truncated);
        }
    }

    pub(super) fn merge(&mut self, other: Self) {
        self.trajectories += other.trajectories;
        self.tool_calls += other.tool_calls;
        self.tool_results += other.tool_results;
        self.tool_errors += other.tool_errors;
        self.turns += other.turns;
        self.tokens += other.tokens;
        self.truncated += other.truncated;
    }

    pub(super) fn values(self, attempted: usize, failures: RolloutFailures) -> Vec<MetricValue> {
        let attempted_count = attempted.max(1) as f32;
        let count = self.trajectories.max(1) as f32;
        vec![
            MetricValue {
                name: "agent/tool_calls_per_traj".into(),
                value: self.tool_calls as f32 / count,
            },
            // Per *turn* as well as per trajectory: a trajectory that stops on
            // its first turn because nothing was parsed has the same
            // `tool_calls_per_traj` as one that ran ten turns and called a tool
            // on the last. Zero here over a whole update, with tools declared,
            // is the signature of a parser that does not read the model's call
            // format.
            MetricValue {
                name: "agent/tool_calls_per_turn".into(),
                value: self.tool_calls as f32 / self.turns.max(1) as f32,
            },
            MetricValue {
                name: "agent/tool_error_fraction".into(),
                value: self.tool_errors as f32 / self.tool_results.max(1) as f32,
            },
            MetricValue {
                name: "agent/turns_per_traj_mean".into(),
                value: self.turns as f32 / count,
            },
            MetricValue {
                name: "agent/truncated_fraction".into(),
                value: self.truncated as f32 / count,
            },
            MetricValue {
                name: "agent/tokens_per_traj_mean".into(),
                value: self.tokens as f32 / count,
            },
            MetricValue {
                name: "agent/failed_fraction".into(),
                value: failures.total() as f32 / attempted_count,
            },
            MetricValue {
                name: "agent/failed_tool".into(),
                value: failures.tool as f32 / attempted_count,
            },
            MetricValue {
                name: "agent/failed_policy".into(),
                value: failures.policy as f32 / attempted_count,
            },
            MetricValue {
                name: "agent/lost_fraction".into(),
                value: 1.0 - (self.trajectories + failures.total()) as f32 / attempted_count,
            },
        ]
    }
}

/// `attempted` counts every rollout the update asked for, including the ones
/// that never produced a trajectory: the failure fractions are meaningless
/// against a denominator that already excludes the failures. The other
/// fractions and the per-trajectory means keep the collected count as their
/// denominator - a rollout that died never got truncated and never called a
/// tool, so counting it there would dilute the very signal the metric exists
/// to raise.
#[cfg(test)]
pub(super) fn agent_metrics(
    groups: &[TrajectoryGroup],
    attempted: usize,
    failures: RolloutFailures,
) -> Vec<MetricValue> {
    let mut totals = AgentMetricTotals::default();
    for group in groups {
        totals.observe(group);
    }
    totals.values(attempted, failures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpo::fixtures::group;

    #[test]
    fn failure_causes_are_reported_separately() {
        let failures = RolloutFailures {
            tool: 2,
            policy: 1,
            other: 0,
        };
        let values = agent_metrics(&[group(&[false, false, false])], 8, failures);
        let value = |name: &str| {
            values
                .iter()
                .find(|metric| metric.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .value
        };
        assert!((value("agent/failed_fraction") - 3.0 / 8.0).abs() < 1e-6);
        assert!((value("agent/failed_tool") - 2.0 / 8.0).abs() < 1e-6);
        assert!((value("agent/failed_policy") - 1.0 / 8.0).abs() < 1e-6);
        // 8 attempted, 3 failed, 3 collected: two members went down with a
        // group that fell under its baseline.
        assert!((value("agent/lost_fraction") - 2.0 / 8.0).abs() < 1e-6);
    }
}
