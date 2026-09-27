//! Rejection sampling: roll a group out per scenario, keep what succeeded, and
//! write it as chat records a warm-start can train on - or, in pairs mode, the
//! best trace against the worst as a preference record.
//!
//! Nothing here is new machinery. The rollouts are the engine's, the grading is
//! the update loop's own (`score_group`), and the records are
//! [`to_chat_example`]'s - so a trace is kept on exactly the reward it would
//! have been trained on, and prepares back to the stream it was collected as.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use retrograd_dataset::{ChatExample, ChatMessage, PreferenceExample};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::export::{AssistantForm, to_chat_example};
use crate::grpo::score_group;
use crate::judge::{JudgeFailurePolicy, RewardBackend, apply_group_scores};
use crate::rollout::{RolloutEngine, RolloutFailures};
use crate::trajectory::{Trajectory, TrajectoryGroup};
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
    /// Write one preference pair per scenario instead of the kept traces.
    pub pairs: Option<PairsConfig>,
}

/// How a scenario's group becomes a pair: its best trace that passes every
/// filter, against its worst complete one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairsConfig {
    /// A pair whose rewards are closer than this prefers too little to learn
    /// from, and is not written.
    pub min_gap: f32,
}

/// Why a scenario produced no pair, one count per cause.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PairRejections {
    /// No trace passed the filters, or none earned `min_gap` less than it: a
    /// group that all failed, or all succeeded alike.
    pub no_gap: usize,
    /// Fewer than two graded traces to compare.
    pub single_member: usize,
    /// The two traces do not open on the same prompt.
    pub prompt_mismatch: usize,
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
    /// Pairs mode: the scenarios that gave no pair, by cause.
    pub pairs: PairRejections,
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

/// Where a collection goes: every kept trace as a record, and each scenario's
/// outcome as it finishes.
pub trait CollectSink {
    fn record(&mut self, example: ChatExample) -> Result<()>;

    /// One pair, in pairs mode.
    fn record_pair(&mut self, pair: PreferenceExample) -> Result<()> {
        let _ = pair;
        Err(Error::invalid(
            "this collection sink takes no preference pairs",
        ))
    }

    /// Scenario `index` of `total` is done.
    fn scenario_finished(&mut self, report: &ScenarioReport, index: usize, total: usize) {
        let _ = (report, index, total);
    }
}

/// Keeps the records in memory.
impl CollectSink for Vec<ChatExample> {
    fn record(&mut self, example: ChatExample) -> Result<()> {
        self.push(example);
        Ok(())
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
    sink: &mut dyn CollectSink,
) -> Result<CollectStats> {
    check_config(config, reward.is_some(), scenarios)?;
    let mut stats = CollectStats::default();
    for (index, scenario) in scenarios.iter().enumerate() {
        if interrupt::stop_requested() {
            stats.interrupted = true;
            break;
        }
        let (attempted, scored) = rollout_and_score(
            engine,
            reward.clone(),
            scenario,
            config,
            base_seed(config, index),
            &mut stats,
        )
        .await?;
        let candidates = scored
            .iter()
            .map(|trajectory| Candidate::from_trajectory(trajectory, scenario, config.form))
            .collect();
        keep_best(
            scenario,
            (index, scenarios.len()),
            attempted,
            candidates,
            config,
            &mut stats,
            sink,
        )?;
    }
    Ok(stats)
}

/// Scenario `index`'s attempts start from `seed + index * k`, each member one
/// further: reproducible, and no two scenarios share a seed.
pub(crate) fn base_seed(config: &CollectConfig, index: usize) -> u64 {
    config
        .seed
        .wrapping_add((index as u64).wrapping_mul(config.k as u64))
}

/// One graded attempt, whichever generator produced it.
pub(crate) struct Candidate {
    pub(crate) reward: f32,
    /// What "shorter" means between two traces of the same reward: tokens for
    /// a rollout, characters for a trace that has no tokens.
    pub(crate) length: usize,
    pub(crate) member: usize,
    pub(crate) seed: u64,
    pub(crate) invalid_turns: usize,
    /// The environment's verdict, when it verified the attempt at all.
    pub(crate) verification: Option<String>,
    /// The attempt as a record, or why it could not be written as one.
    pub(crate) record: Result<ChatExample>,
}

impl Candidate {
    fn from_trajectory(trajectory: &Trajectory, scenario: &Scenario, form: AssistantForm) -> Self {
        let provenance = trajectory.provenance.as_ref();
        Self {
            reward: trajectory.total_reward(),
            length: trajectory.tokens.len(),
            member: provenance.map_or(usize::MAX, |provenance| provenance.member),
            seed: provenance.map_or(0, |provenance| provenance.seed),
            invalid_turns: provenance.map_or(0, |provenance| provenance.invalid_turns),
            verification: verification(&trajectory.metadata).map(str::to_owned),
            record: to_chat_example(trajectory, scenario, form),
        }
    }
}

/// Filters one scenario's graded attempts, keeps the best distinct ones up to
/// the cap, and writes them. `position` is the scenario's index and the total,
/// for the progress report.
pub(crate) fn keep_best(
    scenario: &Scenario,
    position: (usize, usize),
    attempted: usize,
    candidates: Vec<Candidate>,
    config: &CollectConfig,
    stats: &mut CollectStats,
    sink: &mut dyn CollectSink,
) -> Result<()> {
    if let Some(pairs) = &config.pairs {
        return keep_pair(
            scenario, position, attempted, candidates, pairs, config, stats, sink,
        );
    }
    stats.attempted += attempted;
    let mut eligible = Vec::new();
    for candidate in candidates {
        match rejection(&candidate, config) {
            Some(cause) => *cause(&mut stats.rejected) += 1,
            None => eligible.push(candidate),
        }
    }
    let passed = eligible.len();
    let pass_rate = match attempted {
        0 => 0.0,
        attempted => passed as f32 / attempted as f32,
    };
    // Best first; at equal reward the shorter trace, which says the same thing
    // in fewer tokens; then the member, so the order is total.
    eligible.sort_by(|a, b| {
        b.reward
            .total_cmp(&a.reward)
            .then(a.length.cmp(&b.length))
            .then(a.member.cmp(&b.member))
    });
    let mut seen = HashSet::new();
    let mut kept = 0;
    for candidate in eligible {
        let Candidate {
            reward,
            member,
            seed,
            verification,
            record,
            ..
        } = candidate;
        let mut example = match record {
            Ok(example) => example,
            Err(error) => {
                tracing::warn!(scenario = %scenario.id, "a passing trace did not export: {error}");
                stats.rejected.unexportable += 1;
                continue;
            }
        };
        if !seen.insert(fingerprint(&example)) {
            stats.rejected.duplicate += 1;
            continue;
        }
        if kept == config.keep {
            stats.rejected.over_keep += 1;
            continue;
        }
        let turns = example
            .messages
            .iter()
            .filter(|message| message.role == "assistant")
            .count();
        example.metadata = serde_json::Map::from_iter([
            ("scenario_id".into(), json!(scenario.id)),
            ("seed".into(), json!(seed)),
            ("member".into(), json!(member)),
            ("reward".into(), json!(reward)),
            ("verification".into(), json!(verification)),
            ("turns".into(), json!(turns)),
            ("generator".into(), json!(config.generator)),
            ("form".into(), json!(config.form.as_str())),
            ("pass_rate".into(), json!(pass_rate)),
        ]);
        sink.record(example)?;
        kept += 1;
    }
    stats.kept += kept;
    let report = ScenarioReport {
        id: scenario.id.clone(),
        attempted,
        passed,
        kept,
        pass_rate,
    };
    sink.scenario_finished(&report, position.0, position.1);
    stats.scenarios.push(report);
    Ok(())
}

/// Pairs mode's [`keep_best`]: the best trace that passes every filter is
/// `chosen`, the worst of all the others is `rejected`. The worst is taken
/// among every graded, complete trace, whatever filter it failed, as long as it
/// can be written as a record - which a trace with a malformed call cannot: its
/// error observation answers no call - while `chosen` meets the same bar a
/// kept trace does.
#[expect(clippy::too_many_arguments)]
fn keep_pair(
    scenario: &Scenario,
    position: (usize, usize),
    attempted: usize,
    candidates: Vec<Candidate>,
    pairs: &PairsConfig,
    config: &CollectConfig,
    stats: &mut CollectStats,
    sink: &mut dyn CollectSink,
) -> Result<()> {
    stats.attempted += attempted;
    let mut passing = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        match rejection(candidate, config) {
            Some(cause) => *cause(&mut stats.rejected) += 1,
            None => passing.push(index),
        }
    }
    let passed = passing.len();
    let pass_rate = match attempted {
        0 => 0.0,
        attempted => passed as f32 / attempted as f32,
    };
    // Best first, as `keep_best` orders them; the worst is the reverse, the
    // longer of two equal failures first.
    let chosen = passing.iter().copied().min_by(|&a, &b| {
        let (a, b) = (&candidates[a], &candidates[b]);
        b.reward
            .total_cmp(&a.reward)
            .then(a.length.cmp(&b.length))
            .then(a.member.cmp(&b.member))
    });
    let rejected = |chosen: usize| {
        (0..candidates.len())
            .filter(|&index| index != chosen && candidates[index].record.is_ok())
            .min_by(|&a, &b| {
                let (a, b) = (&candidates[a], &candidates[b]);
                a.reward
                    .total_cmp(&b.reward)
                    .then(b.length.cmp(&a.length))
                    .then(a.member.cmp(&b.member))
            })
    };
    let mut kept = 0;
    match (candidates.len(), chosen) {
        (0 | 1, _) => stats.pairs.single_member += 1,
        (_, None) => stats.pairs.no_gap += 1,
        (_, Some(chosen)) => match rejected(chosen) {
            None => {
                tracing::warn!(scenario = %scenario.id, "no other trace of the group could be written");
                stats.rejected.unexportable += 1;
            }
            Some(rejected) => {
                let (best, worst) = (&candidates[chosen], &candidates[rejected]);
                if best.reward - worst.reward < pairs.min_gap {
                    stats.pairs.no_gap += 1;
                } else {
                    match (&best.record, &worst.record) {
                        (Ok(best_record), Ok(worst_record)) => {
                            match preference_pair(best_record, worst_record) {
                                None => stats.pairs.prompt_mismatch += 1,
                                Some(mut pair) => {
                                    pair.metadata = serde_json::Map::from_iter([
                                        ("scenario_id".into(), json!(scenario.id)),
                                        ("chosen_reward".into(), json!(best.reward)),
                                        ("rejected_reward".into(), json!(worst.reward)),
                                        ("chosen_seed".into(), json!(best.seed)),
                                        ("rejected_seed".into(), json!(worst.seed)),
                                        ("pass_rate".into(), json!(pass_rate)),
                                        ("generator".into(), json!(config.generator)),
                                        ("form".into(), json!(config.form.as_str())),
                                    ]);
                                    match pair.validate() {
                                        Ok(()) => {
                                            sink.record_pair(pair)?;
                                            kept = 1;
                                        }
                                        Err(error) => {
                                            tracing::warn!(
                                                scenario = %scenario.id,
                                                "a pair did not export: {error}"
                                            );
                                            stats.rejected.unexportable += 1;
                                        }
                                    }
                                }
                            }
                        }
                        (Err(error), _) | (_, Err(error)) => {
                            tracing::warn!(scenario = %scenario.id, "a paired trace did not export: {error}");
                            stats.rejected.unexportable += 1;
                        }
                    }
                }
            }
        },
    }
    stats.kept += kept;
    let report = ScenarioReport {
        id: scenario.id.clone(),
        attempted,
        passed,
        kept,
        pass_rate,
    };
    sink.scenario_finished(&report, position.0, position.1);
    stats.scenarios.push(report);
    Ok(())
}

/// Two traces of one scenario as a pair: the longest run of messages they open
/// with alike is the prompt - cut back so it ends on what the model answers,
/// a user turn or an observation - and each rest is its response. `None` when
/// the two do not share a prompt, or offer different tools.
fn preference_pair(chosen: &ChatExample, rejected: &ChatExample) -> Option<PreferenceExample> {
    if chosen.tools != rejected.tools {
        return None;
    }
    let common = chosen
        .messages
        .iter()
        .zip(&rejected.messages)
        .take_while(|(a, b)| a == b)
        .count();
    let answers_next = |messages: &[ChatMessage], at: usize| {
        messages
            .get(at)
            .is_some_and(|message| message.role == "assistant")
            && at > 0
            && matches!(messages[at - 1].role.as_str(), "user" | "tool")
    };
    let cut = (1..=common)
        .rev()
        .find(|&at| answers_next(&chosen.messages, at) && answers_next(&rejected.messages, at))?;
    Some(PreferenceExample {
        tools: chosen.tools.clone(),
        prompt: chosen.messages[..cut].to_vec(),
        chosen: chosen.messages[cut..].to_vec(),
        rejected: rejected.messages[cut..].to_vec(),
        metadata: Default::default(),
    })
}

/// Refused before the first rollout: a collection that could keep nothing is
/// an hour of rollouts to find out.
fn check_config(config: &CollectConfig, judged: bool, scenarios: &[Scenario]) -> Result<()> {
    if config.k == 0 || config.keep == 0 {
        return Err(Error::invalid("collect needs k and keep of at least one"));
    }
    if let Some(pairs) = &config.pairs {
        if config.k < 2 {
            return Err(Error::invalid(
                "collect --pairs needs at least two attempts per scenario to compare",
            ));
        }
        if config.keep != 1 {
            return Err(Error::invalid(
                "collect --pairs writes one pair per scenario, so it takes no --keep",
            ));
        }
        if !pairs.min_gap.is_finite() || pairs.min_gap < 0.0 {
            return Err(Error::invalid(
                "collect --min-gap must be finite and non-negative",
            ));
        }
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

/// The first filter a graded attempt fails, as the counter it lands on.
fn rejection(candidate: &Candidate, config: &CollectConfig) -> Option<Cause> {
    if candidate.invalid_turns > 0 {
        return Some(|rejected| &mut rejected.invalid_turns);
    }
    if config
        .min_reward
        .is_some_and(|minimum| candidate.reward < minimum)
    {
        return Some(|rejected| &mut rejected.below_min_reward);
    }
    if config.require_verified && candidate.verification.as_deref() != Some("passed") {
        return Some(|rejected| &mut rejected.unverified);
    }
    None
}

/// The verdict an environment reported in its state, under
/// `env_state.metadata.verification`.
pub(crate) fn verification(metadata: &serde_json::Map<String, Value>) -> Option<&str> {
    metadata
        .get("env_state")?
        .get("metadata")?
        .get("verification")?
        .as_str()
}

/// What makes two traces the same trace: the assistant turns - text and calls -
/// and the observations of the record they are written as. Canonical JSON, so
/// the same arguments in another key order are the same call.
fn fingerprint(example: &ChatExample) -> [u8; 32] {
    let turns = example
        .messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "assistant" | "tool"))
        .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
        .collect::<Vec<_>>();
    Sha256::digest(Value::Array(turns).to_string().as_bytes()).into()
}

pub mod api;

#[cfg(test)]
mod tests;
