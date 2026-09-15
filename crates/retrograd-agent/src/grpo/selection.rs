//! What leaves an update before it reaches the optimizer, and what comes back.

use crate::trajectory::{Trajectory, TrajectoryGroup};
use crate::{Error, Result};

/// Where an update's rollouts went, counted once so the cap can say which
/// cause spent them rather than naming all three.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DropBreakdown {
    /// Rollouts that died mid-way and produced no trajectory at all.
    pub failed: usize,
    /// Trajectories that came back cut short - turn budget, trajectory budget,
    /// max_turns or the rollout deadline.
    pub truncated: usize,
    /// Trajectories that came back whole but never got a reward: a group the
    /// judge dropped, or one too small to keep a baseline.
    pub unscored: usize,
}

/// What an update does with what came back from its rollouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum UpdateFate {
    /// Enough trajectories survived to center a group: train on them.
    Train,
    /// Not a single trainable pair came back, and `skip_empty_updates` is on:
    /// the update produces no optimizer step instead of ending the run. Carries
    /// the same accounting the failure would have reported.
    Skip(String),
}

/// Caps how much of an update may disappear before it stops being an update.
///
/// The three ways a rollout fails to reach the optimizer - truncated, failed
/// mid-rollout, or left unscored by the judge - all cost exactly one trainable
/// trajectory, so they are capped once, together, against the number of
/// rollouts the update asked for. Capping any of them separately would mean
/// several thresholds over different denominators, which is how an update can
/// lose most of its data while every individual check stays under its limit.
/// They are *reported* separately, because the fix for each one is different.
///
/// `skip_empty_updates` only covers the one case the threshold cannot express:
/// fewer than two trainable trajectories, where there is no relative baseline to
/// train on whatever the threshold says. An update that kept a baseline but
/// still blew past the threshold stays a failure - that is the signal that the
/// judge or the environment is broken, and it is not something a skip fixes.
pub(super) fn check_dropped_fraction(
    update: u32,
    attempted: usize,
    trained: usize,
    max_dropped_fraction: f32,
    skip_empty_updates: bool,
    breakdown: DropBreakdown,
    last_error: Option<&str>,
) -> Result<UpdateFate> {
    let dropped = 1.0 - trained as f32 / attempted.max(1) as f32;
    if trained >= 2 && dropped <= max_dropped_fraction {
        return Ok(UpdateFate::Train);
    }
    let DropBreakdown {
        failed,
        truncated,
        unscored,
    } = breakdown;
    let report = format!(
        "update {update} lost {:.1}% of its {attempted} rollouts ({truncated} truncated, \
         {failed} failed, {unscored} unscored), leaving {trained} trainable (threshold \
         {:.1}%){}",
        dropped * 100.0,
        max_dropped_fraction * 100.0,
        match last_error {
            Some(error) => format!("; last error: {error}"),
            None => String::new(),
        }
    );
    if trained < 2 && skip_empty_updates {
        return Ok(UpdateFate::Skip(report));
    }
    Err(Error::Reward(report))
}

/// Longest excerpt of a generation shown in the diagnostic below. Enough to see
/// the call markers a model actually emitted, short enough to stay one glance.
const SAMPLE_LIMIT: usize = 400;

/// Builds the warning for a first update that parsed no tool call while tools
/// were declared.
///
/// The failure this names has no other symptom: the parser returns an empty
/// [`ParsedAssistant`](crate::tools::ParsedAssistant) with no error, the
/// trajectory ends without touching the environment, the group reaches the judge
/// as "unrewarded", and whatever the judge says about it is what kills the run,
/// so the message an operator reads accuses the environment server, which
/// answered perfectly. This is deliberately diagnostic rather than fatal. A
/// model may legitimately answer every scenario in an update without a tool,
/// and the update metrics contain only groups that kept a GRPO baseline: calls
/// from a discarded group are not available here either. Zero is therefore a
/// useful symptom, not proof that the parser is wrong.
pub(super) fn tool_call_parse_warning(
    tools_declared: usize,
    attempted: usize,
    tool_calls: usize,
    sample: Option<&str>,
) -> Option<String> {
    if tools_declared == 0 || tool_calls > 0 {
        return None;
    }
    let excerpt = match sample.map(str::trim).filter(|text| !text.is_empty()) {
        Some(text) if text.chars().count() > SAMPLE_LIMIT => {
            let cut = text
                .char_indices()
                .nth(SAMPLE_LIMIT)
                .map_or(text.len(), |(at, _)| at);
            format!("; first generation: {}…", &text[..cut])
        }
        Some(text) => format!("; first generation: {text}"),
        None => String::new(),
    };
    Some(format!(
        "no tool call was parsed over {attempted} rollouts while {tools_declared} tools are \
         declared; this may mean the model's call format is not the one the parser reads{excerpt}"
    ))
}

#[cfg(test)]
mod tool_call_guard_tests {
    use super::*;

    #[test]
    fn a_run_that_declares_no_tool_is_left_alone() {
        assert!(tool_call_parse_warning(0, 8, 0, Some("hello")).is_none());
    }

    #[test]
    fn one_parsed_call_is_enough_to_prove_the_two_halves_agree() {
        assert!(tool_call_parse_warning(3, 8, 1, None).is_none());
    }

    #[test]
    fn none_at_all_names_the_cause_and_shows_what_the_model_wrote() {
        let warning = tool_call_parse_warning(
            3,
            8,
            0,
            Some("<|tool_call_start|>[move(direction='left')]<|tool_call_end|>"),
        )
        .expect("no call parsed while tools are declared");
        assert!(warning.contains("8 rollouts"), "{warning}");
        assert!(warning.contains("3 tools"), "{warning}");
        assert!(warning.contains("<|tool_call_start|>"), "{warning}");
    }

    #[test]
    fn a_long_generation_is_cut_rather_than_dumped() {
        let sample = "é".repeat(SAMPLE_LIMIT * 2);
        let message = tool_call_parse_warning(1, 2, 0, Some(&sample)).expect("no call parsed");
        // Cut on a character boundary, not a byte one: the excerpt is printed.
        assert!(message.ends_with('…'), "{message}");
        assert!(message.chars().filter(|c| *c == 'é').count() == SAMPLE_LIMIT);
    }
}

/// Splits the groups the judge still has to score from the ones their
/// environment already did.
///
/// An LLM opinion is the fallback for tasks nothing can verify, not the nominal
/// path: when the environment graded the steps itself, sending the group to the
/// judge anyway would stack a judge reward on top of those step rewards. The
/// engine already normalizes a group to all-scored or none-scored, so testing
/// one member per group would do; the check is written over every member so a
/// half-scored group would go to the judge rather than train on a silent zero.
pub(super) fn split_environment_scored(
    groups: Vec<TrajectoryGroup>,
) -> (Vec<TrajectoryGroup>, Vec<TrajectoryGroup>) {
    groups.into_iter().partition(|group| {
        group
            .trajectories
            .iter()
            .any(|trajectory| trajectory.reward.is_none())
    })
}

/// Takes the truncated members out of their groups, whatever happens to them
/// next.
///
/// A partial response cannot be judged as a completed trajectory, so it never
/// reaches the judge - under [`TruncationPolicy::Drop`] it never comes back,
/// and under `MinReward` [`restore_truncated_members`] puts it back once its
/// siblings have a score. Only the truncated members leave: the survivors still
/// carry a valid relative baseline, exactly like `mask_truncated` in the
/// single-turn sampler. A group left with fewer than two scored members has no
/// signal at all, so it goes - taking its truncated members with it, since a
/// group minimum over one trajectory is that trajectory.
pub(super) fn withhold_truncated_members(
    groups: &mut Vec<TrajectoryGroup>,
) -> Vec<(u64, Vec<Trajectory>)> {
    let mut withheld = Vec::new();
    for group in groups.iter_mut() {
        let (truncated, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut group.trajectories)
            .into_iter()
            .partition(|trajectory| trajectory.truncated);
        group.trajectories = kept;
        if !truncated.is_empty() {
            withheld.push((group.group_id, truncated));
        }
    }
    groups.retain(|group| group.trajectories.len() >= 2);
    withheld
}

/// Puts the truncated members back, each carrying the smallest total reward of
/// its scored siblings - the negative signal `Drop` cannot give.
///
/// The minimum is over *totals*: the judge's or the environment's final reward
/// plus the step rewards, which is the quantity `to_train_sequence` hands GRPO
/// and centers the group on. A truncated member that had already banked step
/// rewards therefore gets a negative final reward - what has to land on the
/// group minimum is its total, not the slice the judge would have contributed.
pub(super) fn restore_truncated_members(
    groups: &mut [TrajectoryGroup],
    withheld: Vec<(u64, Vec<Trajectory>)>,
) {
    for (group_id, truncated) in withheld {
        // The group did not survive its own scoring: there is no minimum to
        // hand out, and no baseline left to be the bottom of.
        let Some(group) = groups.iter_mut().find(|group| group.group_id == group_id) else {
            continue;
        };
        let Some(minimum) = group.trajectories.iter().map(total_reward).reduce(f32::min) else {
            continue;
        };
        for mut trajectory in truncated {
            trajectory.reward = Some(minimum - step_reward_sum(&trajectory));
            group.trajectories.push(trajectory);
        }
    }
}

fn step_reward_sum(trajectory: &Trajectory) -> f32 {
    trajectory
        .steps
        .iter()
        .filter_map(|step| step.reward)
        .sum::<f32>()
}

/// What GRPO will actually see for this trajectory, final reward and step
/// rewards together.
fn total_reward(trajectory: &Trajectory) -> f32 {
    trajectory.reward.unwrap_or(0.0) + step_reward_sum(trajectory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpo::fixtures::group;
    use crate::trajectory::{Step, StepKind};

    #[test]
    fn truncation_costs_the_member_not_the_whole_group() {
        let mut groups = vec![group(&[false, true, false])];
        let withheld = withhold_truncated_members(&mut groups);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].trajectories.len(), 2);
        assert_eq!(withheld.len(), 1);
    }

    #[test]
    fn a_group_without_a_baseline_is_dropped() {
        let mut groups = vec![group(&[true, true, false]), group(&[false, false])];
        withhold_truncated_members(&mut groups);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].trajectories.len(), 2);
    }

    #[test]
    fn min_reward_puts_a_truncated_member_at_the_bottom_of_its_group() {
        let mut groups = vec![group(&[false, true, false])];
        let withheld = withhold_truncated_members(&mut groups);
        groups[0].trajectories[0].reward = Some(1.0);
        groups[0].trajectories[1].reward = Some(0.25);

        restore_truncated_members(&mut groups, withheld);
        assert_eq!(groups[0].trajectories.len(), 3);
        let restored = groups[0].trajectories.last().unwrap();
        assert!(
            restored.truncated,
            "the flag survives, only the fate changes"
        );
        assert_eq!(
            restored.reward,
            Some(0.25),
            "overrunning must cost the worst reward of the group, not nothing"
        );
    }

    #[test]
    fn min_reward_counts_the_step_rewards_a_truncated_member_already_banked() {
        // An environment-scored group: the reward GRPO sees is final + steps, so
        // the final reward is whatever brings the *total* down to the minimum.
        let mut groups = vec![group(&[false, false, true])];
        let mut withheld = withhold_truncated_members(&mut groups);
        groups[0].trajectories[0].reward = Some(0.0);
        groups[0].trajectories[0].steps.push(Step {
            kind: StepKind::PolicyAction,
            token_range: (1, 2),
            reward: Some(2.0),
        });
        groups[0].trajectories[1].reward = Some(0.0);
        groups[0].trajectories[1].steps.push(Step {
            kind: StepKind::PolicyAction,
            token_range: (1, 2),
            reward: Some(3.0),
        });
        withheld[0].1[0].steps.push(Step {
            kind: StepKind::PolicyAction,
            token_range: (1, 2),
            reward: Some(1.5),
        });

        restore_truncated_members(&mut groups, withheld);
        let restored = groups[0].trajectories.last().unwrap();
        assert_eq!(restored.reward, Some(2.0 - 1.5));
        assert_eq!(total_reward(restored), 2.0);
    }

    #[test]
    fn a_truncated_member_of_a_dead_group_has_nowhere_to_come_back_to() {
        let mut groups = vec![group(&[true, false])];
        let withheld = withhold_truncated_members(&mut groups);
        assert!(groups.is_empty(), "one survivor is not a group");
        restore_truncated_members(&mut groups, withheld);
        assert!(groups.is_empty());
    }

    #[test]
    fn losing_a_minority_of_rollouts_still_produces_an_update() {
        // One member in three lost - to a tool timeout, a truncation, or a
        // judge that skipped its group; the cap does not care which.
        assert_eq!(
            check_dropped_fraction(
                0,
                24,
                16,
                0.5,
                false,
                DropBreakdown::default(),
                Some("tool timeout")
            )
            .unwrap(),
            UpdateFate::Train
        );
    }

    #[test]
    fn losing_most_of_the_rollouts_fails_the_update() {
        let error = check_dropped_fraction(
            3,
            24,
            8,
            0.5,
            false,
            DropBreakdown {
                failed: 6,
                truncated: 5,
                unscored: 5,
            },
            Some("tool timeout"),
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("66.7%"), "{message}");
        assert!(
            message.contains("tool timeout"),
            "the cause an operator needs must survive into the message: {message}"
        );
        assert!(
            message.contains("5 truncated, 6 failed, 5 unscored"),
            "the three causes have three different fixes, so they are named: {message}"
        );
    }

    #[test]
    fn the_union_is_capped_once_not_cause_by_cause() {
        // 30% truncated, 30% failed and 30% unscored each sit under a 50% cap
        // taken separately, yet only a tenth of the update survives.
        assert!(
            check_dropped_fraction(
                0,
                100,
                10,
                0.5,
                false,
                DropBreakdown {
                    failed: 30,
                    truncated: 30,
                    unscored: 30
                },
                None
            )
            .is_err()
        );
    }

    #[test]
    fn a_single_surviving_trajectory_is_never_an_update() {
        // Under any threshold: one trajectory has no relative baseline.
        assert!(
            check_dropped_fraction(0, 2, 1, 1.0, false, DropBreakdown::default(), None).is_err()
        );
    }

    #[test]
    fn an_update_with_nothing_to_train_on_can_be_skipped_instead_of_fatal() {
        let fate = check_dropped_fraction(
            7,
            8,
            0,
            0.5,
            true,
            DropBreakdown {
                failed: 8,
                truncated: 0,
                unscored: 0,
            },
            Some("container start timed out"),
        )
        .unwrap();
        let UpdateFate::Skip(report) = fate else {
            panic!("zero trainable rollouts must skip once the option is on: {fate:?}");
        };
        assert!(
            report.contains("container start timed out"),
            "a skip still has to say why it skipped: {report}"
        );
    }

    #[test]
    fn a_skip_does_not_excuse_an_update_that_still_had_a_baseline() {
        // Two trainable trajectories out of a hundred is a broken judge or a
        // broken environment, not an empty update: the threshold still fails it.
        assert!(
            check_dropped_fraction(
                0,
                100,
                2,
                0.5,
                true,
                DropBreakdown {
                    failed: 98,
                    truncated: 0,
                    unscored: 0
                },
                None
            )
            .is_err()
        );
    }

    #[test]
    fn an_environment_scored_group_never_reaches_the_judge() {
        let mut scored = group(&[false, false]);
        for trajectory in &mut scored.trajectories {
            trajectory.reward = Some(0.0);
        }
        let mut half_scored = group(&[false, false]);
        half_scored.trajectories[0].reward = Some(0.0);

        let (judged, environment_scored) =
            split_environment_scored(vec![group(&[false, false]), scored, half_scored]);
        assert_eq!(environment_scored.len(), 1);
        assert_eq!(
            judged.len(),
            2,
            "an unscored group and a half-scored one both need the judge"
        );
    }
}
