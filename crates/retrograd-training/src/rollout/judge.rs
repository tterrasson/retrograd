//! The judge of `[grpo.judge]`: one verdict per completion, one request per
//! group.
//!
//! The single-turn loop and the agentic one grade the same shape - the
//! `group_size` completions of one prompt - so they grade it through the same
//! backend. Everything a judge needs is already in `retrograd-judge`: the
//! OpenAI-compatible transport, the on-disk verdict cache, the request
//! concurrency, the retries, the context budget, the failure policy and the
//! `judge/*` metrics. What is missing between there and here is a shape
//! conversion and a runtime, and that is all this file is.
//!
//! Two properties are worth stating because they are what make the blend below
//! defensible. A RULER verdict is *relative inside a group*: it says which of
//! these eight completions is better, not what any of them is worth in the
//! absolute. That is exactly the quantity GRPO consumes, since an advantage is
//! a deviation from the group's own mean - and it is exactly the quantity a
//! held-out evaluation cannot use, which is why `[evaluation]` never calls a
//! judge. And a group is judged whole or not at all: a verdict computed on two
//! different scales inside one group would fabricate an advantage that measures
//! the scales rather than the completions.

use std::sync::Arc;

use retrograd_agent_core::config::JudgeFailurePolicy;
use retrograd_agent_core::trajectory::{Message, Role, Trajectory, TrajectoryGroup};
use retrograd_config::GrpoJudge;
use retrograd_core::{Error, Result};
use retrograd_judge::{JudgeBackend, JudgeBatchCounts, RewardBackend};
use retrograd_metrics::MetricValue;

use super::prompts::Prompt;

/// One prompt and the completions sampled from it, as the judge will see them.
pub(crate) struct JudgeGroup<'a> {
    /// Half the presentation-permutation seed. The *prompt's* index in the
    /// dataset, not the group's position in the update: the permutation is what
    /// the request text depends on, so keying it on the prompt is what lets the
    /// verdict cache hit when the same prompt comes round again.
    pub(crate) group_id: u64,
    pub(crate) prompt: &'a Prompt,
    pub(crate) completions: &'a [String],
}

/// A judge, its runtime, and the policy for a group it could not score.
///
/// `max_dropped_fraction` is deliberately not here: it is a property of an
/// update, and an update judges one batch per sampling wave. It lives with the
/// counts it is applied to, in [`JudgeTally::check`].
pub(crate) struct GroupJudge {
    backend: Arc<dyn RewardBackend>,
    runtime: tokio::runtime::Runtime,
    failure: JudgeFailurePolicy,
    weight: f32,
}

impl GroupJudge {
    pub(crate) fn new(config: &GrpoJudge) -> Result<Self> {
        let backend = config
            .config
            .build(std::path::Path::to_path_buf)
            .map_err(|error| Error::config(format!("build the GRPO judge: {error}")))?;
        // Owned rather than borrowed from the caller: training runs on a plain
        // thread with no ambient runtime, and the judge is the only thing in
        // this crate that awaits anything.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| Error::runtime(format!("start the judge runtime: {error}")))?;
        Ok(Self {
            backend,
            runtime,
            failure: config.failure,
            weight: config.weight,
        })
    }

    /// Default weight of a verdict, for a reward line that states none.
    pub(crate) fn weight(&self) -> f32 {
        self.weight
    }

    /// Scores every group, in one request each when the strategy is listwise.
    ///
    /// Returns one entry per completion, flattened in group order: `Some(score)`
    /// in [0, 1], or `None` for every member of a group the judge failed on and
    /// the policy dropped. What was scored and what was lost is accumulated into
    /// `tally`, which is what carries both the failure threshold and the metrics:
    /// an update judges one batch per sampling wave, and neither a threshold
    /// nor a fraction means anything measured on one wave of several.
    pub(crate) fn score(
        &self,
        groups: &[JudgeGroup<'_>],
        tally: &mut JudgeTally,
    ) -> Result<Vec<Option<f32>>> {
        let mut trajectory_groups = groups
            .iter()
            .map(trajectory_group)
            .collect::<Result<Vec<_>>>()?;
        let metrics = self.runtime.block_on(retrograd_judge::score_groups(
            Arc::clone(&self.backend),
            &mut trajectory_groups,
            self.failure,
            // Never trips here: the threshold is the update's, and this is
            // one of its waves. `tally.check` applies it once the update has
            // stopped judging. `JudgeFailurePolicy::Fail` is unaffected,
            // it stops on the first bad group, wherever that group is.
            1.0,
            // Never dropped here: a group every member of which the judge
            // scored alike is already dropped by `group_advantages`, which
            // sees the *blended* reward and can tell a tie the verifiable
            // part broke from a real one.
            false,
        ))?;
        let scores = trajectory_groups
            .iter()
            .flat_map(|group| group.trajectories.iter().map(|member| member.reward))
            .collect::<Vec<_>>();
        tally.merge(&metrics, scores.iter().filter_map(|score| *score));
        Ok(scores)
    }

    pub(crate) fn metric_values(&self) -> Vec<MetricValue> {
        self.backend.metric_values()
    }
}

/// One sampled group as a trajectory group.
///
/// The token fields stay empty, and that is not a shortcut: a judge reads
/// messages and metadata, never tokens, and these trajectories exist for the
/// length of one request. Filling them would mean cloning every rollout's token
/// vector per update for a reader that does not exist.
pub(super) fn trajectory_group(group: &JudgeGroup<'_>) -> Result<TrajectoryGroup> {
    let scenario_id = format!("prompt-{}", group.group_id);
    let mut context = Vec::new();
    for message in &group.prompt.conversation.messages {
        let role = match message.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(Error::invalid(format!(
                    "a judged prompt cannot carry a '{other}' message"
                )));
            }
        };
        context.push(Message::text(role, message.content.clone()));
    }
    let mut metadata = serde_json::Map::new();
    // Honoured by the judge as the group's own criteria, in place of the
    // run-wide rubric: the dataset line is the only place that knows what this
    // particular prompt should have produced.
    if let Some(rubric) = group.prompt.rubric() {
        metadata.insert(
            "rubric".into(),
            serde_json::Value::String(rubric.to_owned()),
        );
    }
    let trajectories = group
        .completions
        .iter()
        .map(|completion| {
            let mut messages = context.clone();
            messages.push(Message::text(Role::Assistant, completion.clone()));
            Trajectory {
                scenario_id: scenario_id.clone(),
                messages,
                tokens: Vec::new(),
                old_logprobs: Vec::new(),
                train_mask: Vec::new(),
                steps: Vec::new(),
                reward: None,
                truncated: false,
                metadata: metadata.clone(),
            }
        })
        .collect();
    Ok(TrajectoryGroup {
        group_id: group.group_id,
        scenario_id,
        trajectories,
    })
}

/// What one update's judging came to, summed over its sampling waves.
///
/// Every quantity here is a count or a value, never a fraction: dynamic
/// sampling judges one batch per wave, and the waves have different sizes. A
/// mean of fractions would weigh a wave of one group like a wave of four, and
/// the last wave - the smallest, the one that completed the batch - would be the
/// one the series described.
#[derive(Default)]
pub(crate) struct JudgeTally {
    counts: JudgeBatchCounts,
    /// Every verdict the update actually obtained, for the mean and the spread.
    values: Vec<f32>,
    last_error: Option<String>,
}

impl JudgeTally {
    fn merge(
        &mut self,
        metrics: &retrograd_judge::JudgeBatchMetrics,
        values: impl Iterator<Item = f32>,
    ) {
        let counts = &metrics.counts;
        self.counts.groups += counts.groups;
        self.counts.scored_groups += counts.scored_groups;
        self.counts.dropped_groups += counts.dropped_groups;
        self.counts.degenerate_groups += counts.degenerate_groups;
        self.counts.invalid_scores += counts.invalid_scores;
        self.counts.total_scores += counts.total_scores;
        self.values.extend(values);
        // The last one wins, and later waves are the more recent failures.
        if metrics.last_error.is_some() {
            self.last_error = metrics.last_error.clone();
        }
    }

    /// Groups the judge lost, against what the document allows to lose.
    ///
    /// Applied once the update has stopped judging, not per wave: a bad wave
    /// that resampling then made up for is not a broken judge, and stopping the
    /// run on it would be stopping on a threshold nobody configured.
    pub(crate) fn check(&self, max_dropped_fraction: f32) -> Result<()> {
        if self.counts.groups == 0 {
            return Ok(());
        }
        let dropped = self.counts.dropped_groups as f32 / self.counts.groups as f32;
        if dropped > max_dropped_fraction {
            return Err(Error::runtime(format!(
                "the judge dropped {:.1}% of this update's groups, above the configured {:.1}%{}",
                dropped * 100.0,
                max_dropped_fraction * 100.0,
                match &self.last_error {
                    Some(error) => format!("; last judge error: {error}"),
                    None => String::new(),
                }
            )));
        }
        Ok(())
    }

    /// The update's series, under the same names the agentic loop exports so one
    /// dashboard reads both.
    pub(crate) fn metrics(&self) -> Vec<MetricValue> {
        let share = |numerator: usize, denominator: usize| {
            if denominator == 0 {
                0.0
            } else {
                numerator as f32 / denominator as f32
            }
        };
        let (mean, deviation) = mean_std(&self.values);
        vec![
            MetricValue {
                name: "judge/invalid_score_fraction".into(),
                value: share(self.counts.invalid_scores, self.counts.total_scores),
            },
            MetricValue {
                name: "judge/dropped_group_fraction".into(),
                value: share(self.counts.dropped_groups, self.counts.groups),
            },
            MetricValue {
                name: "judge/degenerate_group_fraction".into(),
                value: share(self.counts.degenerate_groups, self.counts.scored_groups),
            },
            MetricValue {
                name: "judge/score_mean".into(),
                value: mean,
            },
            MetricValue {
                name: "judge/score_std".into(),
                value: deviation,
            },
        ]
    }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean as f32, variance.sqrt() as f32)
}
