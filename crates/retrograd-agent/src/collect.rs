//! Rejection sampling: roll a group out per scenario, keep what succeeded, and
//! write it as chat records a warm-start can train on.
//!
//! Nothing here is new machinery. The rollouts are the engine's, the grading is
//! the update loop's own ([`score_group`]), and the records are
//! [`to_chat_example`]'s - so a trace is kept on exactly the reward it would
//! have been trained on, and prepares back to the stream it was collected as.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use retrograd_dataset::ChatExample;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::export::{AssistantForm, to_chat_example};
use crate::grpo::score_group;
use crate::judge::{JudgeFailurePolicy, RewardBackend, apply_group_scores};
use crate::rollout::{RolloutEngine, RolloutFailures};
use crate::trajectory::{Role, Trajectory, TrajectoryGroup};
use crate::{Error, Result, interrupt};
use retrograd_agent_core::scenario::Scenario;

/// What to collect and what to keep of it.
#[derive(Clone, Debug)]
pub struct CollectConfig {
    /// Rollouts per scenario. One is a plain rollout and has to be graded by
    /// its environment: a judge scores a trajectory relative to its group.
    pub k: usize,
    /// Traces kept per scenario, best first. A cap, so that the scenarios a
    /// policy already solves every time do not make up the whole dataset.
    pub keep: usize,
    /// A trace below this total reward is not kept.
    pub min_reward: Option<f32>,
    /// Keep only traces whose environment verified them.
    pub require_verified: bool,
    pub form: AssistantForm,
    /// Scenario `i`'s group is rolled out from `seed + i * k`, so a collection
    /// is reproducible and each scenario's seeds are its own.
    pub seed: u64,
    pub judge_failure: JudgeFailurePolicy,
    /// Whether the environment reports a reward of its own without a
    /// `verify` command - an HTTP environment's `/step` does. Only used to
    /// refuse, before any rollout, a collection nothing could grade.
    pub environment_grades: bool,
    /// Recorded on every record: which model generated it.
    pub generator: String,
}

/// Why a trace that came back was not kept, one count per cause.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Rejections {
    /// Cut by a budget: a partial answer is not an answer to learn.
    pub truncated: usize,
    /// Neither the environment nor the judge gave it a reward.
    pub unscored: usize,
    /// It contains a turn with a malformed call or no call where one was
    /// expected. Not taught even when the trace recovers: the turn itself is
    /// what SFT would teach.
    pub invalid_turns: usize,
    pub below_min_reward: usize,
    pub unverified: usize,
    /// The same turns and observations as a trace already kept.
    pub duplicate: usize,
    /// Eligible, but past the per-scenario cap.
    pub over_keep: usize,
    /// It did not export as a valid record - a defect, logged when it happens.
    pub unexportable: usize,
}

/// One scenario's outcome, the pass@k a later run can be planned from.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioReport {
    pub id: String,
    pub attempted: usize,
    /// Traces that met every filter, before deduplication and the cap.
    pub passed: usize,
    pub kept: usize,
    pub pass_rate: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CollectStats {
    pub attempted: usize,
    pub kept: usize,
    pub rejected: Rejections,
    /// Rollouts that died, by cause.
    #[serde(skip)]
    pub failures: RolloutFailures,
    pub scenarios: Vec<ScenarioReport>,
    /// Stopped by an interrupt before the last scenario.
    pub interrupted: bool,
}

impl CollectStats {
    /// The scenarios no attempt solved: the candidates for a stronger
    /// generator.
    pub fn unsolved(&self) -> impl Iterator<Item = &ScenarioReport> {
        self.scenarios.iter().filter(|report| report.passed == 0)
    }
}

/// Rolls `config.k` attempts out per scenario, in file order, grades them the
/// way an update does, and hands every kept trace to `sink` as a record.
///
/// Scenarios run one after the other: a group already fills the decode batch,
/// as it does in training.
pub async fn collect_trajectories(
    engine: &RolloutEngine,
    reward: Option<Arc<dyn RewardBackend>>,
    scenarios: &[Scenario],
    config: &CollectConfig,
    sink: &mut dyn FnMut(ChatExample) -> Result<()>,
) -> Result<CollectStats> {
    check_config(config, reward.is_some(), scenarios)?;
    let mut stats = CollectStats::default();
    for (index, scenario) in scenarios.iter().enumerate() {
        if interrupt::stop_requested() {
            stats.interrupted = true;
            break;
        }
        let base_seed = config
            .seed
            .wrapping_add((index as u64).wrapping_mul(config.k as u64));
        let (attempted, scored) = rollout_and_score(
            engine,
            reward.clone(),
            scenario,
            config,
            base_seed,
            &mut stats,
        )
        .await?;
        stats.attempted += attempted;

        let mut eligible = Vec::new();
        for trajectory in scored {
            match rejection(&trajectory, config) {
                Some(cause) => *cause(&mut stats.rejected) += 1,
                None => eligible.push(trajectory),
            }
        }
        let passed = eligible.len();
        let pass_rate = match attempted {
            0 => 0.0,
            attempted => passed as f32 / attempted as f32,
        };
        // Best first; at equal reward the shorter trace, which says the same
        // thing in fewer tokens; then the member, so the order is total.
        eligible.sort_by(|a, b| {
            b.total_reward()
                .total_cmp(&a.total_reward())
                .then(a.tokens.len().cmp(&b.tokens.len()))
                .then(member(a).cmp(&member(b)))
        });
        let mut seen = HashSet::new();
        let mut kept = 0;
        for trajectory in eligible {
            if !seen.insert(fingerprint(&trajectory)) {
                stats.rejected.duplicate += 1;
                continue;
            }
            if kept == config.keep {
                stats.rejected.over_keep += 1;
                continue;
            }
            let mut example = match to_chat_example(&trajectory, scenario, config.form) {
                Ok(example) => example,
                Err(error) => {
                    tracing::warn!(scenario = %scenario.id, "a kept trace did not export: {error}");
                    stats.rejected.unexportable += 1;
                    continue;
                }
            };
            example.metadata = record_metadata(&trajectory, config, pass_rate);
            sink(example)?;
            kept += 1;
        }
        stats.kept += kept;
        stats.scenarios.push(ScenarioReport {
            id: scenario.id.clone(),
            attempted,
            passed,
            kept,
            pass_rate,
        });
    }
    Ok(stats)
}

/// Refused before the first rollout: a collection that could keep nothing is
/// an hour of rollouts to find out.
fn check_config(config: &CollectConfig, judged: bool, scenarios: &[Scenario]) -> Result<()> {
    if config.k == 0 || config.keep == 0 {
        return Err(Error::invalid("collect needs k and keep of at least one"));
    }
    if scenarios.is_empty() {
        return Err(Error::invalid("collect needs at least one scenario"));
    }
    if config.min_reward.is_some_and(|reward| !reward.is_finite()) {
        return Err(Error::invalid("collect min_reward must be finite"));
    }
    let verified = scenarios.iter().all(|scenario| {
        crate::env::EnvTask::from_scenario(scenario).is_ok_and(|task| task.verify.is_some())
    });
    if !judged && !config.environment_grades && !verified {
        return Err(Error::invalid(
            "nothing grades these trajectories: a scenario declares no metadata.env.verify \
             command, the environment reports no reward of its own, and there is no \
             [agent.judge]. Keeping the best of k needs a reward to rank them by",
        ));
    }
    if config.k == 1 && !config.environment_grades && !verified {
        return Err(Error::invalid(
            "collect with k = 1 needs the environment to grade every scenario: a judge scores a \
             trajectory relative to the others of its group",
        ));
    }
    Ok(())
}

/// One scenario's attempts and the graded traces among them. Rollout failures
/// are counted on `stats`, and so are the traces that came back but could not
/// be graded.
async fn rollout_and_score(
    engine: &RolloutEngine,
    reward: Option<Arc<dyn RewardBackend>>,
    scenario: &Scenario,
    config: &CollectConfig,
    base_seed: u64,
    stats: &mut CollectStats,
) -> Result<(usize, Vec<Trajectory>)> {
    if config.k == 1 {
        return Ok(match engine.rollout(scenario, base_seed).await {
            Ok(trajectory) if trajectory.truncated => {
                stats.rejected.truncated += 1;
                (1, Vec::new())
            }
            Ok(trajectory) if trajectory.reward.is_none() => {
                stats.rejected.unscored += 1;
                (1, Vec::new())
            }
            Ok(trajectory) => (1, vec![trajectory]),
            Err(error) => {
                tracing::warn!(scenario = %scenario.id, "a rollout failed: {error}");
                stats.failures.record(error.failure_kind());
                (1, Vec::new())
            }
        });
    }

    let outcome = engine.rollout_group(scenario, config.k, base_seed).await?;
    stats.failures.merge(outcome.failures);
    if let Some(error) = &outcome.last_error {
        tracing::warn!(scenario = %scenario.id, "{} rollouts failed: {error}", outcome.failures.total());
    }
    let members = outcome
        .group
        .as_ref()
        .map_or(0, |group| group.trajectories.len());
    let truncated = outcome.group.as_ref().map_or(0, |group| {
        group
            .trajectories
            .iter()
            .filter(|trajectory| trajectory.truncated)
            .count()
    });
    let scoring = score_group(outcome.group, reward, Instant::now()).await?;
    let mut groups: Vec<TrajectoryGroup> = Vec::new();
    if let Some((group, result)) = scoring.judged {
        let mut judged = vec![group];
        // The judge's own cap stays off, as in an update: a dropped group is
        // counted below as unscored, and `judge_failure` decides whether a
        // failure stops the collection.
        apply_group_scores(&mut judged, vec![result], config.judge_failure, 1.0, false)?;
        groups.extend(judged);
    }
    groups.extend(scoring.environment_scored);
    let scored = groups
        .into_iter()
        .flat_map(|group| group.trajectories)
        .filter(|trajectory| trajectory.reward.is_some())
        .collect::<Vec<_>>();
    stats.rejected.truncated += truncated;
    stats.rejected.unscored += members.saturating_sub(truncated + scored.len());
    Ok((outcome.attempted, scored))
}

type Cause = fn(&mut Rejections) -> &mut usize;

/// The first filter a graded trace fails, as the counter it lands on.
fn rejection(trajectory: &Trajectory, config: &CollectConfig) -> Option<Cause> {
    if trajectory
        .provenance
        .as_ref()
        .is_some_and(|provenance| provenance.invalid_turns > 0)
    {
        return Some(|rejected| &mut rejected.invalid_turns);
    }
    if config
        .min_reward
        .is_some_and(|minimum| trajectory.total_reward() < minimum)
    {
        return Some(|rejected| &mut rejected.below_min_reward);
    }
    if config.require_verified && verification(trajectory) != Some("passed") {
        return Some(|rejected| &mut rejected.unverified);
    }
    None
}

/// The verdict the environment reported, if it verified this trace at all.
fn verification(trajectory: &Trajectory) -> Option<&str> {
    trajectory
        .metadata
        .get("env_state")?
        .get("metadata")?
        .get("verification")?
        .as_str()
}

fn member(trajectory: &Trajectory) -> usize {
    trajectory
        .provenance
        .as_ref()
        .map_or(usize::MAX, |provenance| provenance.member)
}

/// What makes two traces the same trace: their assistant turns - text and
/// calls - and the observations between them. Canonical JSON, so the same
/// arguments in another key order are the same call.
fn fingerprint(trajectory: &Trajectory) -> [u8; 32] {
    let turns = trajectory
        .messages
        .iter()
        .filter(|message| matches!(message.role, Role::Assistant | Role::Tool))
        .map(|message| {
            json!({
                "role": message.role.as_str(),
                "content": message.content,
                "tool_calls": message.tool_calls,
                "is_error": message.is_error,
            })
        })
        .collect::<Vec<_>>();
    Sha256::digest(Value::Array(turns).to_string().as_bytes()).into()
}

fn record_metadata(
    trajectory: &Trajectory,
    config: &CollectConfig,
    pass_rate: f32,
) -> serde_json::Map<String, Value> {
    let provenance = trajectory.provenance.as_ref();
    let turns = trajectory
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count();
    serde_json::Map::from_iter([
        ("scenario_id".into(), json!(trajectory.scenario_id)),
        (
            "seed".into(),
            json!(provenance.map(|provenance| provenance.seed)),
        ),
        (
            "member".into(),
            json!(provenance.map(|provenance| provenance.member)),
        ),
        ("reward".into(), json!(trajectory.total_reward())),
        ("verification".into(), json!(verification(trajectory))),
        ("turns".into(), json!(turns)),
        ("generator".into(), json!(config.generator)),
        ("form".into(), json!(config.form.as_str())),
        ("pass_rate".into(), json!(pass_rate)),
    ])
}

#[cfg(test)]
mod tests;
