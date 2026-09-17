//! One GRPO update: collect the groups, decide what gets scored by whom, score
//! it, and hand the survivors to the optimizer while the next update's
//! sandboxes warm up.

use std::sync::Arc;
use std::time::Instant;

use retrograd_core::{TrainConfig, TrainMetrics};
use retrograd_metrics::MetricValue;
use retrograd_observe::{
    ObserveBatch, ObservedRollout, TrajectoryObserver, UpdateStatus, UpdateSummary,
};
use retrograd_training::Progress;
use retrograd_training::batch::{BatchObservation, GrpoBatchParams};

use super::evaluate::evaluate_scenarios;
use super::metrics::{AgentMetricTotals, ratio_or_zero};
use super::observe::{UpdateExport, draft};
use super::selection::{
    DropBreakdown, UpdateFate, check_dropped_fraction, restore_truncated_members,
    split_environment_scored, tool_call_parse_warning, withhold_truncated_members,
};
use super::{AgentFlow, UpdateBoundary, UpdateHook};
use crate::env::EnvironmentFactory;
use crate::judge::{JudgeBatchMetrics, RewardBackend, Score, apply_group_scores};
use crate::policy::TrainerLender;
use crate::rollout::{GroupOutcome, RolloutEngine, RolloutFailures};
use crate::trajectory::{Role, Trajectory, TrajectoryGroup};
use crate::{Error, PolicyHandle, Result};
use retrograd_agent_core::config::AgentGrpoConfig;
use retrograd_agent_core::scenario::{Scenario, TruncationPolicy};

struct CollectedGroup {
    attempted: usize,
    failures: RolloutFailures,
    last_error: Option<String>,
    metrics: AgentMetricTotals,
    /// One assistant generation, kept verbatim for the diagnostic in
    /// [`tool_call_parse_warning`]: the format a model actually emitted is what
    /// turns "no tool call was parsed" from a guess into a reading.
    first_assistant_text: Option<String>,
    truncated: usize,
    withheld: Vec<(u64, Vec<Trajectory>)>,
    judged: Option<(TrajectoryGroup, Result<Vec<Score>>)>,
    environment_scored: Option<TrajectoryGroup>,
    rollout_done_at: f32,
    judge_window: Option<(f32, f32)>,
    group_id: Option<u64>,
    /// Every trajectory as it came back, when the update is exported.
    drafts: Vec<ObservedRollout>,
}

/// Where a collected group sits in an exported update: the one-based update,
/// the group's index in it, and its scenario.
type Drafting<'a> = (u32, usize, &'a Scenario);

async fn collect_and_start_judge(
    outcome: Result<GroupOutcome>,
    reward: Option<Arc<dyn RewardBackend>>,
    update_started: Instant,
    drafting: Option<Drafting<'_>>,
) -> Result<CollectedGroup> {
    let outcome = outcome?;
    let rollout_done_at = update_started.elapsed().as_secs_f32();
    let group_id = outcome.group.as_ref().map(|group| group.group_id);
    let drafts = match (drafting, &outcome.group) {
        (Some((update, index, scenario)), Some(group)) => group
            .trajectories
            .iter()
            .filter_map(|trajectory| draft(update, index, scenario, trajectory))
            .collect(),
        _ => Vec::new(),
    };
    let mut metrics = AgentMetricTotals::default();
    let mut first_assistant_text = None;
    if let Some(group) = &outcome.group {
        metrics.observe(group);
        first_assistant_text = group
            .trajectories
            .iter()
            .flat_map(|trajectory| trajectory.messages.iter())
            .find(|message| message.role == Role::Assistant)
            .map(|message| message.content.clone());
    }
    let truncated = outcome
        .group
        .iter()
        .flat_map(|group| group.trajectories.iter())
        .filter(|trajectory| trajectory.truncated)
        .count();
    let mut groups = outcome.group.into_iter().collect::<Vec<_>>();
    let withheld = withhold_truncated_members(&mut groups);
    let (mut judged, mut environment_scored) = split_environment_scored(groups);
    let environment_scored = environment_scored.pop();

    let (judged, judge_window) = match judged.pop() {
        // No judge and no environment reward: the group cannot be graded at
        // all. Reported as a scoring failure rather than dropped on the spot,
        // because that is the same fact and `judge_failure` is where the run
        // already says what to do with it - drop the group, or stop.
        Some(group) if reward.is_none() => (
            Some((
                group,
                Err(Error::Reward(
                    "this run has no judge and the environment scored none of the group's \
                     trajectories, so nothing grades them"
                        .into(),
                )),
            )),
            None,
        ),
        Some(group) => {
            let reward = reward.expect("the None case is matched above");
            let judge_started = update_started.elapsed().as_secs_f32();
            // The trainer stays on its LocalSet; judge I/O is Send and runs on
            // the runtime worker pool, so a synchronous CUDA decode cannot
            // prevent an already-started HTTP request from progressing.
            let task = tokio::spawn(async move {
                let result = reward.score_group(&group).await;
                (group, result)
            });
            let judged = task
                .await
                .map_err(|error| Error::Reward(format!("judge task failed: {error}")))?;
            let judge_finished = update_started.elapsed().as_secs_f32();
            (Some(judged), Some((judge_started, judge_finished)))
        }
        None => (None, None),
    };

    Ok(CollectedGroup {
        attempted: outcome.attempted,
        failures: outcome.failures,
        last_error: outcome.last_error,
        metrics,
        first_assistant_text,
        truncated,
        withheld,
        judged,
        environment_scored,
        rollout_done_at,
        judge_window,
        group_id,
        drafts,
    })
}

/// The series every update boundary knows, trained or not.
fn boundary_values(
    agent_metrics: &[MetricValue],
    rollout_seconds: f32,
    judge_seconds: f32,
    judge_metrics: &JudgeBatchMetrics,
    reward: Option<&Arc<dyn RewardBackend>>,
    environments: Option<&Arc<dyn EnvironmentFactory>>,
) -> Vec<MetricValue> {
    let mut values = agent_metrics.to_vec();
    values.extend([
        MetricValue {
            name: "timing/rollout_seconds".into(),
            value: rollout_seconds,
        },
        MetricValue {
            name: "timing/judge_seconds".into(),
            value: judge_seconds,
        },
        MetricValue {
            name: "judge/invalid_score_fraction".into(),
            value: judge_metrics.invalid_score_fraction,
        },
        MetricValue {
            name: "judge/dropped_group_fraction".into(),
            value: judge_metrics.dropped_group_fraction,
        },
        MetricValue {
            name: "judge/degenerate_group_fraction".into(),
            value: judge_metrics.degenerate_group_fraction,
        },
        MetricValue {
            name: "judge/score_mean".into(),
            value: judge_metrics.score_mean,
        },
        MetricValue {
            name: "judge/score_std".into(),
            value: judge_metrics.score_std,
        },
    ]);
    if let Some(reward) = reward {
        values.extend(reward.metric_values());
    }
    if let Some(environments) = environments {
        values.extend(environments.metric_values());
    }
    values
}

/// Everything one update loop runs against. A struct rather than a dozen
/// parameters, which is what it had grown into.
pub(super) struct UpdateLoop<'a, 'p, 'h> {
    pub engine: &'a RolloutEngine,
    pub policy: &'a PolicyHandle,
    pub lender: TrainerLender,
    pub scenarios: &'a [Scenario],
    pub evaluation: &'a [Scenario],
    pub config: &'a AgentGrpoConfig,
    pub training: &'a TrainConfig,
    /// `None` runs without a judge: the environment's own reward is the only
    /// one, and a group it did not score is handled by `judge_failure`.
    pub reward: Option<Arc<dyn RewardBackend>>,
    pub environments: Option<&'a Arc<dyn EnvironmentFactory>>,
    pub on_progress: &'p mut dyn FnMut(Progress),
    pub hook: Option<&'h mut dyn UpdateHook>,
    pub start_update: u32,
    pub observer: Option<Arc<dyn TrajectoryObserver>>,
}

pub(super) async fn run_updates(loop_: UpdateLoop<'_, '_, '_>) -> Result<TrainMetrics> {
    let UpdateLoop {
        engine,
        policy,
        lender,
        scenarios,
        evaluation,
        config,
        training,
        reward,
        environments,
        on_progress,
        mut hook,
        start_update,
        observer,
    } = loop_;
    let mut final_metrics = TrainMetrics::default();
    for update in start_update..config.updates {
        let scenario_at = |offset: usize| {
            &scenarios[(update as usize * config.scenarios_per_update + offset) % scenarios.len()]
        };
        let mut export = observer
            .as_ref()
            .filter(|observer| observer.wants(update + 1))
            .map(|_| {
                UpdateExport::new((0..config.scenarios_per_update).map(scenario_at).collect())
            });
        let exporting = export.is_some();
        let rollout_started = Instant::now();
        // Lifetime counters, differenced around this update's rollouts: the
        // reuse a whole run accumulated says nothing about whether *this* update
        // paid for its prefixes.
        let generation_before = policy.generation_stats().await?;
        let selected = (0..config.scenarios_per_update).map(|offset| {
            let scenario = scenario_at(offset);
            let seed = config
                .seed
                .wrapping_add((update as u64) << 32)
                .wrapping_add(offset as u64 * config.group_size as u64);
            let reward = reward.clone();
            async move {
                collect_and_start_judge(
                    engine
                        .rollout_group(scenario, config.group_size, seed)
                        .await,
                    reward,
                    rollout_started,
                    exporting.then_some((update + 1, offset, scenario)),
                )
                .await
            }
        });
        let outcomes = futures::future::join_all(selected)
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let Collected {
            attempted,
            failures,
            last_error,
            metric_totals,
            first_assistant_text,
            mut breakdown,
            withheld,
            mut groups,
            judge_results,
            mut environment_scored,
            rollout_seconds,
            judge_first,
            judge_last,
        } = collect(outcomes, export.as_mut());
        // Only the first update needs the compatibility diagnostic. It is a
        // warning, not a refusal: answering directly is valid, and calls from a
        // group that lost its baseline are no longer present in these metrics.
        if update == start_update
            && let Some(warning) = tool_call_parse_warning(
                engine.declared_tools(),
                attempted,
                metric_totals.tool_calls(),
                first_assistant_text.as_deref(),
            )
        {
            tracing::warn!("{warning}");
        }
        let agent_metrics = metric_totals.values(attempted, failures);
        breakdown.failed = failures.total();
        // The judge's own cap is disabled: truncated, failed and unscored
        // members all cost the same trainable trajectory, so the fraction that
        // matters is measured once, below, over their union. Capping the
        // judge's slice separately would apply a second threshold with a
        // different denominator.
        let judge_metrics = if groups.is_empty() {
            JudgeBatchMetrics::default()
        } else {
            apply_group_scores(
                &mut groups,
                judge_results,
                config.judge_failure,
                1.0,
                config.drop_degenerate_groups,
            )?
        };
        if let Some(export) = &mut export {
            export.judged(&groups);
        }
        let judge_seconds = if judge_first.is_finite() {
            (judge_last - judge_first).max(0.0)
        } else {
            0.0
        };
        groups.append(&mut environment_scored);
        groups.retain(|group| {
            group
                .trajectories
                .iter()
                .all(|trajectory| trajectory.reward.is_some())
        });
        if config.truncation == TruncationPolicy::MinReward {
            restore_truncated_members(&mut groups, withheld);
        }
        let trained = groups
            .iter()
            .map(|group| group.trajectories.len())
            .sum::<usize>();
        // Under `MinReward` a truncated member is put back and trains: it did
        // not cost the update anything, so it is not one of its drops.
        breakdown.truncated -= groups
            .iter()
            .flat_map(|group| group.trajectories.iter())
            .filter(|trajectory| trajectory.truncated)
            .count();
        // Whatever is left over once the failures, the truncations and the
        // survivors are accounted for: a group the judge dropped, or one that
        // fell under two members and lost its baseline.
        breakdown.unscored = attempted
            .saturating_sub(breakdown.failed)
            .saturating_sub(breakdown.truncated)
            .saturating_sub(trained);
        // A rollout error comes first - it happened earlier and explains more,
        // but a judge that refused every group is the whole story when the
        // rollouts themselves went through.
        let reported_error = last_error
            .as_deref()
            .or(judge_metrics.last_error.as_deref());
        let fate = match check_dropped_fraction(
            update,
            attempted,
            trained,
            config.max_dropped_fraction,
            config.skip_empty_updates,
            breakdown,
            reported_error,
        ) {
            Ok(fate) => fate,
            Err(error) => {
                // What was collected is still worth reading; nothing trained.
                if let (Some(observer), Some(mut export)) = (&observer, export.take()) {
                    export.settle(&groups, None);
                    observer.observe(export.batch());
                }
                return Err(error);
            }
        };
        let observation = match (&observer, export.take()) {
            (Some(observer), Some(mut export)) => {
                let members = match fate {
                    UpdateFate::Train => export.members(&groups),
                    UpdateFate::Skip(_) => None,
                };
                export.settle(&groups, Some(&fate));
                observer.observe(export.batch());
                members.map(|members| BatchObservation {
                    observer: observer.clone(),
                    update: update + 1,
                    members,
                })
            }
            _ => None,
        };
        let boundary = || {
            boundary_values(
                &agent_metrics,
                rollout_seconds,
                judge_seconds,
                &judge_metrics,
                reward.as_ref(),
                environments,
            )
        };
        // A skipped update still falls through to the boundary below: it
        // consumed its scenarios, and it is the one place the run can be
        // checkpointed or stopped - a run whose updates all come back empty has
        // to stay interruptible.
        if let UpdateFate::Skip(report) = &fate {
            tracing::warn!("{report}; skipping this update (agent.skip_empty_updates)");
        }
        let summary = if fate == UpdateFate::Train {
            let (metrics, summary) = train_update(
                policy,
                config,
                training,
                environments,
                observer.as_ref(),
                &groups,
                observation,
                generation_before,
                update,
                &boundary,
                on_progress,
            )
            .await?;
            final_metrics = metrics;
            summary
        } else if observer.is_some() {
            Some((UpdateStatus::Skipped, boundary()))
        } else {
            None
        };
        if let (Some(observer), Some((status, values))) = (&observer, summary) {
            observer.observe(ObserveBatch::Update(UpdateSummary::new(
                update + 1,
                status,
                values
                    .iter()
                    .map(|value| (value.name.as_ref(), value.value)),
            )));
        }

        // The boundary: the update's rollouts, judging and optimizer step are
        // done and the next update has not started, so this is the one moment
        // the trainer can be borrowed and the run can be stopped.
        let Some(hook) = hook.as_deref_mut() else {
            continue;
        };
        let flow = cross_boundary(
            hook,
            &lender,
            engine,
            evaluation,
            config,
            update,
            final_metrics.global_step,
        )
        .await?;
        if flow == AgentFlow::Stop {
            break;
        }
    }
    Ok(final_metrics)
}

/// The boundary of one update: evaluate if the hook asks, then hand it the
/// trainer. The only point at which the run can be checkpointed or stopped.
async fn cross_boundary(
    hook: &mut dyn UpdateHook,
    lender: &TrainerLender,
    engine: &RolloutEngine,
    evaluation: &[Scenario],
    config: &AgentGrpoConfig,
    update: u32,
    global_step: u64,
) -> Result<AgentFlow> {
    let completed = update + 1;
    let measured = if !evaluation.is_empty() && hook.should_evaluate(completed, config.updates) {
        // Derived from the update, not from a running counter: two evaluations
        // of the same policy at the same point must sample the same way,
        // including after a resume.
        let seed = config.seed ^ ((completed as u64) << 16);
        Some(evaluate_scenarios(engine, evaluation, seed).await?)
    } else {
        None
    };
    let scenarios_consumed = completed as u64 * config.scenarios_per_update as u64;
    lender
        .with_trainer(|trainer| {
            hook.update_finished(UpdateBoundary {
                update: completed,
                updates: config.updates,
                global_step,
                scenarios_consumed,
                trainer,
                evaluation: measured,
            })
        })
        .await
}

/// What one update's rollouts produced, folded out of the per-group outcomes.
///
/// A struct rather than thirteen accumulators in the loop: every field is a
/// separate reduction over the same sequence, and naming the result is what
/// lets the update's own body read as the five phases it is.
struct Collected {
    attempted: usize,
    failures: RolloutFailures,
    /// The last rollout error, reported ahead of a judge refusal: it happened
    /// earlier and explains more.
    last_error: Option<String>,
    metric_totals: AgentMetricTotals,
    first_assistant_text: Option<String>,
    breakdown: DropBreakdown,
    withheld: Vec<(u64, Vec<Trajectory>)>,
    /// Judged groups, paired index for index with `judge_results`.
    groups: Vec<TrajectoryGroup>,
    judge_results: Vec<Result<Vec<Score>>>,
    environment_scored: Vec<TrajectoryGroup>,
    /// Wall seconds to the last group's rollout, and the window the judging of
    /// all of them spans. `judge_first` stays infinite when nothing was judged.
    rollout_seconds: f32,
    judge_first: f32,
    judge_last: f32,
}

fn collect(outcomes: Vec<CollectedGroup>, mut export: Option<&mut UpdateExport>) -> Collected {
    let mut c = Collected {
        attempted: 0,
        failures: RolloutFailures::default(),
        last_error: None,
        metric_totals: AgentMetricTotals::default(),
        first_assistant_text: None,
        breakdown: DropBreakdown::default(),
        withheld: Vec::new(),
        groups: Vec::with_capacity(outcomes.len()),
        judge_results: Vec::new(),
        environment_scored: Vec::new(),
        rollout_seconds: 0.0,
        judge_first: f32::INFINITY,
        judge_last: 0.0,
    };
    for (index, outcome) in outcomes.into_iter().enumerate() {
        if let Some(export) = &mut export {
            export.collected(index, outcome.group_id, outcome.drafts);
        }
        c.attempted += outcome.attempted;
        c.failures.merge(outcome.failures);
        c.last_error = outcome.last_error.or(c.last_error);
        c.metric_totals.merge(outcome.metrics);
        c.first_assistant_text = c.first_assistant_text.or(outcome.first_assistant_text);
        c.breakdown.truncated += outcome.truncated;
        c.withheld.extend(outcome.withheld);
        c.rollout_seconds = c.rollout_seconds.max(outcome.rollout_done_at);
        if let Some((start, end)) = outcome.judge_window {
            c.judge_first = c.judge_first.min(start);
            c.judge_last = c.judge_last.max(end);
        }
        if let Some((group, result)) = outcome.judged {
            c.groups.push(group);
            c.judge_results.push(result);
        }
        c.environment_scored.extend(outcome.environment_scored);
    }
    c
}

/// The optimizer step of one update, and the progress events it produces.
///
/// Returns the metrics the run carries forward and, when an observer is
/// listening, the summary values of the last event.
#[expect(clippy::too_many_arguments)]
async fn train_update(
    policy: &PolicyHandle,
    config: &AgentGrpoConfig,
    training: &TrainConfig,
    environments: Option<&Arc<dyn EnvironmentFactory>>,
    observer: Option<&Arc<dyn TrajectoryObserver>>,
    groups: &[TrajectoryGroup],
    observation: Option<BatchObservation>,
    generation_before: retrograd_engine::GenerationStats,
    update: u32,
    boundary: &dyn Fn() -> Vec<MetricValue>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(TrainMetrics, Option<(UpdateStatus, Vec<MetricValue>)>)> {
    let mut summary = None;
    let sequences = RolloutEngine::groups_to_train_sequences(groups, config.truncation)?;
    let params = GrpoBatchParams {
        epochs: config.epochs,
        clip_range_low: config.clip_range_low,
        clip_range_high: config.clip_range_high,
        kl_coefficient: config.kl_coefficient,
        loss_denominator: config.loss_denominator()?,
        seed: config.seed ^ ((update as u64 + 1) << 32),
        // The learning-rate horizon covers the *run*, not this batch:
        // the runtime's scheduler step accumulates from one update to
        // the next, so a per-batch horizon would decay the rate to
        // zero after the first one. Counted on the nominal slots, like
        // the GRPO CLI, so the timeline does not move with how many
        // trajectories a given update happened to keep.
        scheduler_total_rollouts: Some(
            config.updates as u64 * config.scenarios_per_update as u64 * config.group_size as u64,
        ),
    };
    // The optimizer step is dead time on the environment side, and it is
    // the only window wide enough to pay for a container start. Warming the
    // *next* update's sandboxes here is what turns `env/acquire_ms_mean`
    // from a container creation into a hash-map lookup. A failure is not
    // one: the pool creates on demand anyway.
    let training_step = async {
        let started = Instant::now();
        let result = policy
            .train_grpo_batch(sequences, params, training.clone(), observation)
            .await;
        (result, started.elapsed().as_secs_f32())
    };
    let ((metrics, progress), optimizer_seconds) =
        match environments.filter(|_| update + 1 < config.updates) {
            Some(environments) => {
                let wanted = config.scenarios_per_update * config.group_size;
                let prewarm = async {
                    if let Err(error) = environments.prewarm(wanted).await {
                        tracing::warn!(%error, "prewarming the next update's environments failed");
                    }
                };
                let (training, ()) = futures::future::join(training_step, prewarm).await;
                let (result, seconds) = training;
                (result?, seconds)
            }
            None => {
                let (result, seconds) = training_step.await;
                (result?, seconds)
            }
        };
    let generation = policy
        .generation_stats()
        .await?
        .delta_since(generation_before);
    let mut final_metrics = metrics;
    final_metrics.epoch = update + 1;
    let last = progress.len().saturating_sub(1);
    for (position, mut event) in progress.into_iter().enumerate() {
        event.metrics.epoch = update + 1;
        event.values.extend(boundary());
        event.values.extend([
            MetricValue {
                name: "timing/optimizer_wall_seconds".into(),
                value: optimizer_seconds,
            },
            // The share of what the rollout asked to be resident that was
            // already there. Zero on a single-turn run, where every
            // prompt is new; it is the multi-turn one where the ratio is
            // the cost model, since a trajectory that re-decodes its
            // prefix every turn spends prefill in the square of its
            // turn count.
            MetricValue {
                name: "generation/prefill_reuse_fraction".into(),
                value: ratio_or_zero(generation.reused_tokens, generation.prompt_tokens),
            },
            MetricValue {
                name: "generation/prefilled_tokens".into(),
                value: generation.prefilled_tokens as f32,
            },
            // Non-zero means the generation context has fewer sequences
            // than the update has live trajectories, so slots are being
            // taken from trajectories that will come back for them.
            // `generation_concurrency` is the only lever for it.
            MetricValue {
                name: "generation/kv_eviction_fraction".into(),
                value: ratio_or_zero(generation.evictions, generation.sequences),
            },
        ]);
        if observer.is_some() && position == last {
            summary = Some((UpdateStatus::Completed, event.values.clone()));
        }
        on_progress(event);
    }
    Ok((final_metrics, summary))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use tokio::sync::Semaphore;

    use super::*;
    use crate::grpo::fixtures::group;

    struct SignallingJudge {
        started: Arc<Semaphore>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl RewardBackend for SignallingJudge {
        async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.add_permits(1);
            Ok(group
                .trajectories
                .iter()
                .map(|_| Score {
                    value: 0.5,
                    valid: true,
                    explanation: None,
                    error: None,
                })
                .collect())
        }
    }

    fn outcome() -> Result<GroupOutcome> {
        Ok(GroupOutcome {
            group: Some(group(&[false, false])),
            attempted: 2,
            failures: RolloutFailures::default(),
            last_error: None,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_completed_group_starts_judging_before_the_next_rollout_finishes() {
        let judge = Arc::new(SignallingJudge {
            started: Arc::new(Semaphore::new(0)),
            calls: AtomicUsize::new(0),
        });
        let started = Instant::now();
        let first = collect_and_start_judge(outcome(), Some(judge.clone()), started, None);
        let second = async {
            let permit = judge.started.acquire().await.unwrap();
            permit.forget();
            collect_and_start_judge(outcome(), Some(judge.clone()), started, None).await
        };
        let (first, second) = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            futures::future::join(first, second),
        )
        .await
        .expect("the second rollout waits for the first judge to start");
        assert!(first.unwrap().judged.is_some());
        assert!(second.unwrap().judged.is_some());
        assert_eq!(judge.calls.load(Ordering::SeqCst), 2);
    }

    /// Without a judge, a group the environment did not score is not silently
    /// dropped here: it comes back as a scoring failure, which is what
    /// `judge_failure` is asked about downstream.
    #[tokio::test]
    async fn a_group_no_environment_scored_is_a_failure_when_there_is_no_judge() {
        let collected = collect_and_start_judge(outcome(), None, Instant::now(), None)
            .await
            .expect("collecting never fails on its own");
        let (_, result) = collected.judged.expect("the group needed a score");
        let error = result
            .expect_err("nothing could have scored it")
            .to_string();
        assert!(error.contains("no judge"), "got {error}");
    }
}
