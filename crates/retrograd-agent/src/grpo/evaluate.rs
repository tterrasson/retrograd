//! Held-out evaluation of an agentic policy.
//!
//! **What is measured, and why it is not the judge.** Every RULER strategy,
//! listwise, chunked, pairwise - scores the members of a group *against each
//! other*. Those scores are renormalized inside every group, so their mean is
//! not comparable from one update to the next: an "improving" curve built on
//! them would be an artefact of the normalization, and early stopping on it
//! would stop on noise. A judge-based number is therefore deliberately not
//! offered here.
//!
//! What is comparable is the reward the **environment** itself puts on a
//! trajectory: a `verify` command, a test suite, a task's own grading. It is an
//! absolute measure of the same quantity at every update, which is exactly what
//! `best_eval` and `patience` need. A run whose environment does not grade gets
//! a sentence saying so, not a number that means nothing.
//!
//! **All of the held-out set, or none of it.** A pass where some rollouts fail
//! or come back ungraded is refused rather than averaged over the survivors:
//! the survivors are chosen by the policy, so a policy that breaks on the hard
//! scenarios would report a *better* mean for having broken on them - and that
//! mean is what `best_eval` and early stopping act on.

use crate::env::EnvTask;
use crate::rollout::RolloutEngine;
use crate::{Error, Result};
use retrograd_agent_core::scenario::Scenario;

/// One held-out pass: one trajectory per scenario, graded by its environment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AgentEvalMetrics {
    pub mean_reward: f32,
    pub reward_min: f32,
    pub reward_max: f32,
    /// Scenarios the pass covered. Always the whole held-out set: a pass that
    /// did not grade every one of them is refused rather than reported, so this
    /// is the denominator of `mean_reward` and not a subset of it.
    pub scenarios: usize,
    /// Wall time of the complete held-out pass, including environments and
    /// generation. Kept here so the boundary reporter can export it beside the
    /// evaluation values rather than on an unrelated optimizer epoch.
    pub elapsed_seconds: f32,
}

/// Runs `scenarios` once each and aggregates what their environments returned.
///
/// One trajectory per scenario, not a group: a group exists to give a judge
/// something to compare, and this pass has no judge. `seed` is derived from the
/// update so two evaluations of the same policy at the same point sample the
/// same way.
pub(super) async fn evaluate_scenarios(
    engine: &RolloutEngine,
    scenarios: &[Scenario],
    seed: u64,
) -> Result<AgentEvalMetrics> {
    let started = std::time::Instant::now();
    if scenarios.is_empty() {
        return Err(Error::invalid("evaluation scenario list must not be empty"));
    }
    let passes = scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| engine.rollout(scenario, seed.wrapping_add(index as u64)));
    let trajectories = futures::future::join_all(passes).await;

    let mut rewards = Vec::with_capacity(trajectories.len());
    let mut ungraded = 0;
    let mut failed = 0;
    let mut last_error = None;
    for (scenario, trajectory) in scenarios.iter().zip(trajectories) {
        match trajectory {
            // The same total the optimizer trains on (`crate::train`): an
            // environment that grades its steps sets the trajectory's own
            // reward to zero and puts the grade on the steps, so reading either
            // one alone would report nothing.
            Ok(trajectory) => match trajectory.reward {
                Some(reward) => rewards.push(reward + trajectory.step_reward_sum()),
                None => {
                    // A sandbox task with an explicit verifier has a known
                    // failure value. Producing no valid tool call (and thus no
                    // submit) is a failed attempt on that scale, not an absent
                    // grade that should abort the whole run.
                    let failure_reward = EnvTask::from_scenario(scenario)
                        .ok()
                        .and_then(|task| task.verify.map(|verify| verify.reward_on_failure));
                    match failure_reward {
                        Some(reward) => rewards.push(reward),
                        None => ungraded += 1,
                    }
                }
            },
            // A failed rollout is not an ungraded one: it produced nothing to
            // grade. Both are refused below, for the same reason.
            Err(error) => {
                failed += 1;
                last_error = Some(error.to_string());
            }
        }
    }
    // The whole held-out set or nothing. Dropping the misses from the
    // denominator would make the mean an average over whichever scenarios the
    // policy managed to finish: a policy that fails or leaves ungraded exactly
    // the hard ones would score *higher* for it, and that number decides
    // `best_eval` and early stopping. There is no honest fill-in either - a
    // fixed penalty would have to be on the environment's own reward scale,
    // which this code does not know.
    if ungraded > 0 || failed > 0 {
        let missed = ungraded + failed;
        let mut detail = format!(
            "{missed} of {} evaluation rollouts produced no grade ({failed} failed, {ungraded} \
             came back ungraded). An evaluation pass covers the whole held-out set or it is \
             refused: a mean over the rest is taken on a subset that moves with the policy, so \
             failing the hard scenarios would improve the score",
            scenarios.len()
        );
        if ungraded > 0 {
            detail.push_str(
                ". An agentic evaluation measures the reward the environment puts on a \
                 trajectory - a verify command, a test suite, a task's own grading. A judge \
                 cannot stand in: its scores are relative inside a group and not comparable \
                 across updates",
            );
        }
        if let Some(error) = last_error {
            detail.push_str(&format!(". Last rollout failure: {error}"));
        }
        return Err(Error::invalid(detail));
    }
    let scenarios = rewards.len();
    let total: f32 = rewards.iter().sum();
    Ok(AgentEvalMetrics {
        mean_reward: total / scenarios as f32,
        reward_min: rewards.iter().copied().fold(f32::INFINITY, f32::min),
        reward_max: rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        scenarios,
        elapsed_seconds: started.elapsed().as_secs_f32(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::env::{Environment, EnvironmentFactory};
    use crate::rollout::tests::{
        CountingFactory, MultiTurnPolicy, engine_with_environments, group_limits, scenario,
    };
    use crate::tools::HermesToolCallParser;

    fn held_out(count: usize) -> Vec<Scenario> {
        (0..count)
            .map(|index| Scenario {
                id: format!("eval-{index}"),
                ..scenario()
            })
            .collect()
    }

    /// The number an agentic evaluation reports is the same total the optimizer
    /// trains on - the environment's grade, which for a step-grading
    /// environment lives on the steps and not on `trajectory.reward`.
    #[tokio::test]
    async fn an_environment_graded_pass_reports_the_reward_the_optimizer_sees() {
        let factory = Arc::new(CountingFactory {
            reward: Some(0.5),
            ..Default::default()
        });
        let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
        let measured = evaluate_scenarios(&engine, &held_out(3), 7)
            .await
            .expect("a grading environment produces a measurement");
        assert_eq!(measured.scenarios, 3);
        assert_eq!(measured.mean_reward, 0.5);
        assert_eq!(measured.reward_min, 0.5);
        assert_eq!(measured.reward_max, 0.5);
        assert!(measured.elapsed_seconds >= 0.0);
    }

    #[tokio::test]
    async fn a_verifiable_attempt_without_a_submit_gets_the_declared_failure_reward() {
        let factory = Arc::new(CountingFactory::default());
        let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
        let mut scenarios = held_out(2);
        for scenario in &mut scenarios {
            scenario.metadata.insert(
                "env".into(),
                serde_json::json!({
                    "verify": {
                        "command": ["python", "-c", "raise SystemExit(1)"],
                        "reward_on_success": 1.0,
                        "reward_on_failure": -0.25
                    }
                }),
            );
        }
        let measured = evaluate_scenarios(&engine, &scenarios, 7)
            .await
            .expect("the verifier declares the missing submit's failure scale");
        assert_eq!(measured.mean_reward, -0.25);
        assert_eq!(measured.reward_min, -0.25);
        assert_eq!(measured.reward_max, -0.25);
    }

    /// The case the whole design turns on: with nothing to grade a trajectory,
    /// the honest answer is a sentence, not a number. A judge cannot stand in,
    /// its scores are relative inside a group.
    #[tokio::test]
    async fn an_ungrading_environment_is_told_it_cannot_be_evaluated() {
        let factory = Arc::new(CountingFactory::default());
        let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
        let error = evaluate_scenarios(&engine, &held_out(2), 7)
            .await
            .expect_err("nothing graded these trajectories")
            .to_string();
        assert!(error.contains("came back ungraded"), "{error}");
        assert!(error.contains("A judge cannot stand in"), "{error}");
    }

    /// The case that decides `best_eval`: half the set graded, half not. A mean
    /// over the graded half would reward a policy for failing the other one, so
    /// the pass is refused and the count says how much of the set it lost.
    #[tokio::test]
    async fn a_partly_graded_pass_is_refused_rather_than_averaged_over_the_survivors() {
        struct HalfGrading {
            graded: CountingFactory,
            ungraded: CountingFactory,
            created: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EnvironmentFactory for HalfGrading {
            async fn create(&self) -> Result<Box<dyn Environment>> {
                if self
                    .created
                    .fetch_add(1, Ordering::Relaxed)
                    .is_multiple_of(2)
                {
                    self.graded.create().await
                } else {
                    self.ungraded.create().await
                }
            }
        }

        let factory = Arc::new(HalfGrading {
            graded: CountingFactory {
                reward: Some(0.5),
                ..Default::default()
            },
            ungraded: CountingFactory::default(),
            created: AtomicUsize::new(0),
        });
        let engine = RolloutEngine::with_environments(
            Arc::new(MultiTurnPolicy),
            factory,
            Arc::new(HermesToolCallParser),
            group_limits(),
        )
        .expect("the engine builds");
        let error = evaluate_scenarios(&engine, &held_out(4), 7)
            .await
            .expect_err("half a held-out set is not a measurement")
            .to_string();
        assert!(error.contains("2 of 4"), "{error}");
        assert!(error.contains("came back ungraded"), "{error}");
    }

    #[tokio::test]
    async fn an_empty_held_out_set_is_refused_rather_than_averaged_over_nothing() {
        let factory = Arc::new(CountingFactory::default());
        let engine = engine_with_environments(Arc::new(MultiTurnPolicy), factory, group_limits());
        assert!(evaluate_scenarios(&engine, &[], 7).await.is_err());
    }
}
