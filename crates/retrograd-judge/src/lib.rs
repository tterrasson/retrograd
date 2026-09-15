//! Scoring backends for groups of completions and trajectories.
//!
//! A [`RewardBackend`] scores the members of a group: [`CommandReward`] through
//! a local process speaking the JSONL protocol of [`process`], [`RulerJudge`]
//! by ranking them against each other with an OpenAI-compatible model.
//! [`apply_group_scores`] folds a judge's responses into the rewards under the
//! configured failure policy. The HTTP client, and with it `reqwest` and
//! `rustls`, stays behind the `http` feature.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use retrograd_metrics::MetricValue;
use serde::{Deserialize, Serialize};

use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_agent_core::{Error, Result};

/// The OpenAI-shaped HTTP client, and with it `reqwest` and `rustls`: behind
/// the `http` feature so a build whose rewards are all commands links neither
/// dependency.
#[cfg(feature = "http")]
pub mod client;
mod command;
pub mod eval;
/// The JSONL reward protocol and the process that speaks it, shared by this
/// crate's `command` backend and the training loop's `reward_command`.
pub mod process;
pub mod render;
pub mod ruler;

pub use command::CommandReward;
pub use eval::{EvalReport, Fixture, evaluate, parse_fixtures};
pub use process::{REWARD_PROTOCOL_VERSION, RewardProcess, RewardProcessError};
pub use render::JudgeContext;
pub use render::compact::CompactionConfig;
pub use render::prompt::{Aggregation, DEFAULT_PAIRWISE_RUBRIC, DEFAULT_RUBRIC};
#[cfg(feature = "http")]
pub use ruler::RulerJudge;
pub use ruler::{JudgeStrategy, RulerConfig, RulerStats};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Score {
    pub value: f32,
    pub valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[async_trait]
pub trait RewardBackend: Send + Sync {
    async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>>;

    fn metric_values(&self) -> Vec<MetricValue> {
        Vec::new()
    }
}

pub use retrograd_spec::judge::JudgeConfig;

/// Builds the judge backend a `JudgeConfig` declares - the one thing
/// `retrograd-spec` cannot do itself without linking the transport.
pub trait JudgeBackend {
    /// `resolve` rebases relative paths (the RULER cache) against the
    /// configuration file's directory; pass the identity function when paths are
    /// already absolute.
    fn build(
        &self,
        resolve: impl Fn(&std::path::Path) -> std::path::PathBuf,
    ) -> Result<Arc<dyn RewardBackend>>;
}

impl JudgeBackend for JudgeConfig {
    fn build(
        &self,
        #[cfg_attr(not(feature = "http"), allow(unused_variables))]
        resolve: impl Fn(&std::path::Path) -> std::path::PathBuf,
    ) -> Result<Arc<dyn RewardBackend>> {
        Ok(match self {
            #[cfg(feature = "http")]
            Self::Ruler { config } => {
                let mut config = (**config).clone();
                config.cache_path = config.cache_path.as_deref().map(&resolve);
                Arc::new(RulerJudge::new(config)?)
            }
            // The configuration is right and the build is not, which is the one
            // sentence a missing transport has to say (same shape as
            // `retrograd-env`'s container arm).
            #[cfg(not(feature = "http"))]
            Self::Ruler { .. } => {
                return Err(Error::invalid(
                    "a RULER judge needs the `http` feature of retrograd-judge",
                ));
            }
            Self::Command {
                command,
                timeout_secs,
            } => Arc::new(CommandReward::with_timeout(
                command.clone(),
                Duration::from_secs(*timeout_secs),
            )?),
        })
    }
}

/// Re-exported: the policy is declared next to the rest of the agentic run
/// configuration, in `retrograd-agent-core`, so the TOML loader can build it
/// without linking a judge transport.
pub use retrograd_agent_core::config::JudgeFailurePolicy;

/// The integers the fractions of [`JudgeBatchMetrics`] are computed from.
///
/// Reported next to them because a *fraction* cannot be merged: a caller that
/// judges an update in several batches - GRPO's dynamic sampling resamples
/// zero-signal groups, and each wave is a batch - has to sum the counts and
/// divide once, or its series describe the last wave rather than the update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JudgeBatchCounts {
    pub groups: usize,
    /// Groups the judge returned usable scores for.
    pub scored_groups: usize,
    /// Groups left unrewarded: a judge failure, or a degenerate group when
    /// `drop_degenerate_groups` is on.
    pub dropped_groups: usize,
    /// Scored groups whose members all got the same value. A share of
    /// `scored_groups`, not of `groups`.
    pub degenerate_groups: usize,
    pub invalid_scores: usize,
    pub total_scores: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct JudgeBatchMetrics {
    pub invalid_score_fraction: f32,
    /// Share of `groups` left unrewarded, judge failures and opted-in
    /// degenerate drops combined; see [`JudgeBatchCounts::dropped_groups`].
    pub dropped_group_fraction: f32,
    /// Share of scored groups whose members all got the same value. Such a group
    /// has a zero advantage everywhere: it consumed a rollout, a judge request
    /// and an optimizer slot, and contributed to no gradient. A number that
    /// climbs is the signal that the task is too easy, too hard, or that the
    /// judge cannot tell the members apart.
    pub degenerate_group_fraction: f32,
    pub score_mean: f32,
    pub score_std: f32,
    /// Why the last dropped group was dropped. Under `drop_group` a judge that
    /// refuses every request costs the update all of its trajectories without
    /// raising anything of its own, and a fraction alone does not say whether
    /// the transport, the credentials or the answers were the problem.
    pub last_error: Option<String>,
    pub counts: JudgeBatchCounts,
}

/// Whether a judged group carries any signal at all.
///
/// GRPO centres rewards within a group, so a group whose scores are all equal
/// produces an advantage of exactly zero for every member, whatever the value.
fn is_degenerate(scores: &[Score]) -> bool {
    let mut values = scores.iter().map(|score| score.value);
    let Some(first) = values.next() else {
        return true;
    };
    values.all(|value| (value - first).abs() <= f32::EPSILON)
}

/// Scores groups concurrently, applies the explicit failure policy, and writes
/// valid rewards/explanations back into the trajectories.
///
/// `drop_degenerate_groups` discards groups that carry no signal. It defaults to
/// off at every call site on purpose: dropping them changes the update's
/// denominator - fewer groups, so a different effective batch size - which is a
/// choice to state, not a default to inherit.
pub async fn score_groups(
    backend: Arc<dyn RewardBackend>,
    groups: &mut [TrajectoryGroup],
    failure_policy: JudgeFailurePolicy,
    max_dropped_fraction: f32,
    drop_degenerate_groups: bool,
) -> Result<JudgeBatchMetrics> {
    let results =
        futures::future::join_all(groups.iter().map(|group| backend.score_group(group))).await;
    apply_group_scores(
        groups,
        results,
        failure_policy,
        max_dropped_fraction,
        drop_degenerate_groups,
    )
}

/// Applies judge responses which were started by the rollout pipeline as soon
/// as each group became complete. Kept separate from [`score_groups`] so early
/// transport concurrency does not duplicate score validation, failure policy,
/// reward attachment or batch metrics.
pub fn apply_group_scores(
    groups: &mut [TrajectoryGroup],
    results: Vec<Result<Vec<Score>>>,
    failure_policy: JudgeFailurePolicy,
    max_dropped_fraction: f32,
    drop_degenerate_groups: bool,
) -> Result<JudgeBatchMetrics> {
    if groups.is_empty() {
        return Err(Error::invalid(
            "judge requires at least one trajectory group",
        ));
    }
    if !max_dropped_fraction.is_finite() || !(0.0..=1.0).contains(&max_dropped_fraction) {
        return Err(Error::invalid(
            "max_dropped_fraction must be finite and in [0, 1]",
        ));
    }
    if results.len() != groups.len() {
        return Err(Error::invalid(
            "judge result count does not match trajectory group count",
        ));
    }
    let total_scores = groups
        .iter()
        .map(|group| group.trajectories.len())
        .sum::<usize>();
    let mut invalid_scores = 0_usize;
    let mut dropped_groups = 0_usize;
    let mut degenerate_groups = 0_usize;
    let mut degenerate_dropped = 0_usize;
    let mut scored_groups = 0_usize;
    let mut valid_values = Vec::new();
    let mut last_error = None;

    for (group, result) in groups.iter_mut().zip(results) {
        let scores = match result {
            Ok(scores)
                if scores.len() == group.trajectories.len()
                    && scores.iter().all(|score| {
                        score.valid && score.value.is_finite() && (0.0..=1.0).contains(&score.value)
                    }) =>
            {
                scores
            }
            Ok(scores) => {
                invalid_scores += scores.iter().filter(|score| !score.valid).count().max(1);
                dropped_groups += 1;
                last_error = Some(format!(
                    "judge returned {} scores for the {} members of group '{}', or scores \
                     outside [0, 1]",
                    scores.len(),
                    group.trajectories.len(),
                    group.scenario_id
                ));
                if matches!(failure_policy, JudgeFailurePolicy::Fail) {
                    return Err(Error::Reward(format!(
                        "judge returned invalid scores for group '{}'",
                        group.scenario_id
                    )));
                }
                continue;
            }
            Err(error) => {
                invalid_scores += group.trajectories.len();
                dropped_groups += 1;
                last_error = Some(error.to_string());
                if matches!(failure_policy, JudgeFailurePolicy::Fail) {
                    return Err(error);
                }
                continue;
            }
        };
        scored_groups += 1;
        if is_degenerate(&scores) {
            degenerate_groups += 1;
            if drop_degenerate_groups {
                // Left unrewarded, so the same downstream filter that handles a
                // judge failure drops it: one path out of the batch, not two.
                // Counted apart from `dropped_groups` because it is not a judge
                // failure - a task everyone solves would otherwise trip the
                // "the judge is broken" threshold. It still costs the agent's
                // cap on unscored trajectories, which is where an opted-in drop
                // belongs.
                degenerate_dropped += 1;
                continue;
            }
        }
        for (trajectory, score) in group.trajectories.iter_mut().zip(scores) {
            trajectory.reward = Some(score.value);
            if let Some(explanation) = score.explanation {
                trajectory.metadata.insert(
                    "judge_explanation".into(),
                    serde_json::Value::String(explanation),
                );
            }
            valid_values.push(score.value);
        }
    }

    let dropped_fraction = dropped_groups as f32 / groups.len() as f32;
    if dropped_fraction > max_dropped_fraction {
        return Err(Error::Reward(format!(
            "judge dropped {:.1}% of groups, above the configured {:.1}% threshold{}",
            dropped_fraction * 100.0,
            max_dropped_fraction * 100.0,
            match &last_error {
                Some(error) => format!("; last judge error: {error}"),
                None => String::new(),
            }
        )));
    }
    let (mean, std) = mean_std(&valid_values);
    Ok(JudgeBatchMetrics {
        invalid_score_fraction: invalid_scores as f32 / total_scores.max(1) as f32,
        dropped_group_fraction: (dropped_groups + degenerate_dropped) as f32 / groups.len() as f32,
        degenerate_group_fraction: if scored_groups == 0 {
            0.0
        } else {
            degenerate_groups as f32 / scored_groups as f32
        },
        score_mean: mean,
        score_std: std,
        last_error,
        counts: JudgeBatchCounts {
            groups: groups.len(),
            scored_groups,
            dropped_groups: dropped_groups + degenerate_dropped,
            degenerate_groups,
            invalid_scores,
            total_scores,
        },
    })
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean as f32, variance.sqrt() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::Trajectory;

    struct PartialJudge;

    #[async_trait]
    impl RewardBackend for PartialJudge {
        async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
            if group.scenario_id == "bad" {
                return Err(Error::Reward("bad group".into()));
            }
            Ok(group
                .trajectories
                .iter()
                .enumerate()
                .map(|(index, _)| Score {
                    value: index as f32,
                    valid: true,
                    explanation: Some("ok".into()),
                    error: None,
                })
                .collect())
        }
    }

    fn group(id: &str) -> TrajectoryGroup {
        let trajectory = Trajectory {
            scenario_id: id.into(),
            messages: vec![],
            tokens: vec![1, 2],
            old_logprobs: vec![-1.0],
            train_mask: vec![false, true],
            steps: vec![],
            reward: None,
            truncated: false,
            metadata: Default::default(),
        };
        TrajectoryGroup {
            group_id: 1,
            scenario_id: id.into(),
            trajectories: vec![trajectory.clone(), trajectory],
        }
    }

    #[test]
    fn judge_config_accepts_both_the_inline_and_nested_ruler_spellings() {
        // TOML runner shape: fields inline next to `type`.
        let inline: JudgeConfig = serde_json::from_str(
            r#"{"type":"ruler","base_url":"http://x/v1","model":"m","strategy":{"mode":"listwise"}}"#,
        )
        .unwrap();
        // Python binding shape: fields nested under `config`.
        let nested: JudgeConfig = serde_json::from_str(
            r#"{"type":"ruler","config":{"base_url":"http://x/v1","model":"m","strategy":{"mode":"listwise"}}}"#,
        )
        .unwrap();
        for parsed in [inline, nested] {
            let JudgeConfig::Ruler { config } = parsed else {
                panic!("expected a RULER judge");
            };
            assert_eq!(config.model, "m");
            assert_eq!(config.strategy, JudgeStrategy::Listwise);
        }

        let command: JudgeConfig =
            serde_json::from_str(r#"{"type":"command","command":["./reward.sh"]}"#).unwrap();
        let JudgeConfig::Command {
            command,
            timeout_secs,
        } = command
        else {
            panic!("expected a command judge");
        };
        assert_eq!(command, ["./reward.sh"]);
        assert_eq!(timeout_secs, 30);
    }

    #[test]
    fn judge_config_rejects_ambiguous_and_unknown_input() {
        for source in [
            // Mixing both spellings hides which one wins.
            r#"{"type":"ruler","config":{"base_url":"http://x/v1","model":"m"},"model":"other"}"#,
            r#"{"type":"llm-as-a-vibe","model":"m"}"#,
            r#"{"base_url":"http://x/v1","model":"m"}"#,
            r#"{"type":"command"}"#,
            r#"{"type":"command","command":["x"],"extra":1}"#,
            // Unknown RULER field: deny_unknown_fields still applies through the
            // inline spelling.
            r#"{"type":"ruler","base_url":"http://x/v1","model":"m","rubrik":"typo"}"#,
        ] {
            assert!(
                serde_json::from_str::<JudgeConfig>(source).is_err(),
                "accepted {source}"
            );
        }
    }

    #[tokio::test]
    async fn drop_group_keeps_other_group_scores_unchanged() {
        let mut groups = vec![group("good"), group("bad")];
        let metrics = score_groups(
            Arc::new(PartialJudge),
            &mut groups,
            JudgeFailurePolicy::DropGroup,
            0.5,
            false,
        )
        .await
        .unwrap();
        assert_eq!(groups[0].trajectories[1].reward, Some(1.0));
        assert_eq!(groups[1].trajectories[0].reward, None);
        assert_eq!(metrics.dropped_group_fraction, 0.5);
    }

    #[tokio::test]
    async fn exceeding_the_dropped_threshold_is_fatal() {
        // Half the groups fail; a threshold below that means the judge is
        // broken and continuing would train on noise.
        let mut groups = vec![group("good"), group("bad")];
        let error = score_groups(
            Arc::new(PartialJudge),
            &mut groups,
            JudgeFailurePolicy::DropGroup,
            0.25,
            false,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("threshold"), "{error}");
    }

    #[tokio::test]
    async fn fail_policy_stops_on_the_first_bad_group() {
        let mut groups = vec![group("bad"), group("good")];
        let error = score_groups(
            Arc::new(PartialJudge),
            &mut groups,
            JudgeFailurePolicy::Fail,
            1.0,
            false,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("bad group"), "{error}");
    }

    struct FlatJudge(f32);

    #[async_trait]
    impl RewardBackend for FlatJudge {
        async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
            Ok(group
                .trajectories
                .iter()
                .map(|_| Score {
                    value: self.0,
                    valid: true,
                    explanation: None,
                    error: None,
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn a_group_with_no_spread_is_reported_and_only_dropped_on_request() {
        // Every member scored the same: the GRPO advantage is zero throughout,
        // so this group trains nothing whatever its value.
        let mut groups = vec![group("a"), group("b")];
        let metrics = score_groups(
            Arc::new(FlatJudge(0.7)),
            &mut groups,
            JudgeFailurePolicy::DropGroup,
            0.0,
            false,
        )
        .await
        .unwrap();
        assert_eq!(metrics.degenerate_group_fraction, 1.0);
        // Reported, not acted on: the rewards are still written.
        assert_eq!(groups[0].trajectories[0].reward, Some(0.7));
        assert_eq!(metrics.dropped_group_fraction, 0.0);

        // Opted in, the same groups leave the batch unrewarded - and a task
        // everyone solves must not be mistaken for a broken judge, so this does
        // not trip the `max_dropped_fraction` threshold.
        let mut groups = vec![group("a"), group("b")];
        let metrics = score_groups(
            Arc::new(FlatJudge(0.7)),
            &mut groups,
            JudgeFailurePolicy::DropGroup,
            0.0,
            true,
        )
        .await
        .unwrap();
        assert_eq!(metrics.degenerate_group_fraction, 1.0);
        assert_eq!(metrics.dropped_group_fraction, 1.0);
        assert!(
            groups
                .iter()
                .flat_map(|group| &group.trajectories)
                .all(|trajectory| trajectory.reward.is_none())
        );
    }

    #[tokio::test]
    async fn a_scored_group_never_keeps_a_partial_reward_set() {
        // Every trajectory of a kept group must carry a reward: the downstream
        // filter assumes rewards arrive all-or-nothing per group.
        let mut groups = vec![group("good")];
        score_groups(
            Arc::new(PartialJudge),
            &mut groups,
            JudgeFailurePolicy::DropGroup,
            0.0,
            false,
        )
        .await
        .unwrap();
        assert!(
            groups[0]
                .trajectories
                .iter()
                .all(|trajectory| trajectory.reward.is_some())
        );
    }
}
