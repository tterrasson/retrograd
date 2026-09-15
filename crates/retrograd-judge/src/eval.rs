//! Measuring a judge against reference labels.
//!
//! A judge that has never been measured is a hypothesis: it produces numbers
//! that GRPO turns into gradients, and nothing in a training run tells you
//! whether those numbers track quality or the order the trajectories arrived in.
//! This module answers two questions from a fixture file:
//!
//! - **Agreement.** Kendall's τ and Spearman's ρ between the judge's scores and
//!   reference labels. Rank correlations rather than an error on the values,
//!   because GRPO only ever uses a score *relative to its group* - a judge that
//!   scores everything 0.2 too high is not wrong for our purpose.
//! - **Self-consistency.** The same correlation between two independent passes.
//!   It bounds the first: a judge that disagrees with itself cannot agree with
//!   anything else, and the gap between the two numbers separates "the rubric is
//!   wrong" from "the judge is noisy".
//!
//! The fixture shape is deliberately not a serialized `TrajectoryGroup`: a
//! reference file is written by hand, and asking for token ids and log
//! probabilities to grade a transcript would make it unwritable.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use retrograd_agent_core::trajectory::{
    Message, Role, Step, StepKind, Trajectory, TrajectoryGroup,
};
use retrograd_agent_core::{Error, Result};

use crate::RewardBackend;

/// One graded group: a shared opening, then the candidates that answered it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub scenario_id: String,
    /// Messages every candidate shares - the task statement. Rendering sends
    /// them once, exactly as in a real group.
    #[serde(default)]
    pub context: Vec<FixtureMessage>,
    /// Scenario rubric, used in place of the judge's own when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub messages: Vec<FixtureMessage>,
    /// Reference quality. Only the ordering within a fixture is used, so any
    /// consistent scale works.
    pub label: f32,
    /// Terminal environment state, if the reference has one to show.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_state: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureMessage {
    pub role: Role,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

impl Fixture {
    fn validate(&self) -> Result<()> {
        if self.scenario_id.trim().is_empty() {
            return Err(Error::invalid("fixture scenario_id must not be empty"));
        }
        if self.candidates.len() < 2 {
            return Err(Error::invalid(
                "a fixture needs at least two candidates: a relative judge cannot rank one",
            ));
        }
        if self
            .candidates
            .iter()
            .any(|candidate| !candidate.label.is_finite())
        {
            return Err(Error::invalid("fixture labels must all be finite"));
        }
        Ok(())
    }

    /// Builds the group the judge will see. The token stream is synthetic and
    /// minimal - nothing downstream of a `RewardBackend` reads it - but it still
    /// satisfies `Trajectory::validate`, so a fixture cannot smuggle in a shape
    /// the rollout engine would never produce.
    fn to_group(&self, group_id: u64) -> Result<TrajectoryGroup> {
        let trajectories = self
            .candidates
            .iter()
            .map(|candidate| {
                let mut metadata = serde_json::Map::new();
                if let Some(rubric) = &self.rubric {
                    metadata.insert("rubric".into(), serde_json::json!(rubric));
                }
                if let Some(state) = &candidate.env_state {
                    metadata.insert("env_state".into(), serde_json::json!({"summary": state}));
                }
                Trajectory {
                    scenario_id: self.scenario_id.clone(),
                    messages: self
                        .context
                        .iter()
                        .chain(&candidate.messages)
                        .map(|message| Message {
                            role: message.role,
                            content: message.content.clone(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            is_error: message.is_error,
                        })
                        .collect(),
                    tokens: vec![0, 1],
                    old_logprobs: vec![-1.0],
                    train_mask: vec![false, true],
                    steps: vec![
                        Step {
                            kind: StepKind::Context,
                            token_range: (0, 1),
                            reward: None,
                        },
                        Step {
                            kind: StepKind::PolicyAction,
                            token_range: (1, 2),
                            reward: None,
                        },
                    ],
                    reward: None,
                    truncated: false,
                    metadata,
                }
            })
            .collect();
        let group = TrajectoryGroup {
            group_id,
            scenario_id: self.scenario_id.clone(),
            trajectories,
        };
        group.validate()?;
        for trajectory in &group.trajectories {
            trajectory.validate()?;
        }
        Ok(group)
    }
}

pub fn parse_fixtures(source: &str) -> Result<Vec<Fixture>> {
    let mut fixtures = Vec::new();
    for (index, line) in source.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fixture: Fixture = serde_json::from_str(line)
            .map_err(|error| Error::invalid(format!("fixture line {}: {error}", index + 1)))?;
        fixture.validate()?;
        fixtures.push(fixture);
    }
    if fixtures.is_empty() {
        return Err(Error::invalid("fixture file contains no fixtures"));
    }
    Ok(fixtures)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct EvalReport {
    pub fixtures: usize,
    pub scored_fixtures: usize,
    /// Rank agreement with the reference labels, pooled over fixtures.
    pub kendall_tau: f32,
    pub spearman_rho: f32,
    /// Rank agreement of the judge with a second pass of itself. An upper bound
    /// on what agreement with anything else can be.
    pub self_kendall_tau: f32,
    pub self_spearman_rho: f32,
    /// Share of fixtures the judge scored identically across every candidate:
    /// no ordering at all, whatever the labels say.
    pub degenerate_fraction: f32,
}

/// Scores every fixture twice and reports both correlations.
///
/// Two passes over the same input hit the response cache when one is
/// configured, which would make self-consistency look perfect for free. Pass a
/// judge with no `cache_path` to measure it honestly - the report says nothing
/// about caching, so the caller owns that choice.
pub async fn evaluate(backend: Arc<dyn RewardBackend>, fixtures: &[Fixture]) -> Result<EvalReport> {
    let mut report = EvalReport {
        fixtures: fixtures.len(),
        ..Default::default()
    };
    let (mut tau, mut rho, mut self_tau, mut self_rho) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut self_pairs = 0_usize;
    let mut degenerate = 0_usize;
    for (index, fixture) in fixtures.iter().enumerate() {
        let group = fixture.to_group(index as u64)?;
        let first = backend.score_group(&group).await?;
        let second = backend.score_group(&group).await?;
        if first.len() != fixture.candidates.len() || first.iter().any(|score| !score.valid) {
            continue;
        }
        let labels = fixture
            .candidates
            .iter()
            .map(|candidate| candidate.label as f64)
            .collect::<Vec<_>>();
        let values = first
            .iter()
            .map(|score| score.value as f64)
            .collect::<Vec<_>>();
        if crate::is_degenerate(&first) {
            degenerate += 1;
        }
        report.scored_fixtures += 1;
        tau += kendall_tau(&values, &labels);
        rho += spearman_rho(&values, &labels);
        if second.len() == first.len() && second.iter().all(|score| score.valid) {
            let repeat = second
                .iter()
                .map(|score| score.value as f64)
                .collect::<Vec<_>>();
            self_tau += kendall_tau(&values, &repeat);
            self_rho += spearman_rho(&values, &repeat);
            self_pairs += 1;
        }
    }
    if report.scored_fixtures == 0 {
        return Err(Error::Reward(
            "the judge produced no valid scores for any fixture".into(),
        ));
    }
    let scored = report.scored_fixtures as f64;
    report.kendall_tau = (tau / scored) as f32;
    report.spearman_rho = (rho / scored) as f32;
    if self_pairs > 0 {
        report.self_kendall_tau = (self_tau / self_pairs as f64) as f32;
        report.self_spearman_rho = (self_rho / self_pairs as f64) as f32;
    }
    report.degenerate_fraction = degenerate as f32 / scored as f32;
    Ok(report)
}

/// Kendall's τ-b: concordant minus discordant pairs, corrected for ties on
/// either side. τ-a would punish a judge for a tie the labels also declare.
pub fn kendall_tau(left: &[f64], right: &[f64]) -> f64 {
    let (mut concordant, mut discordant, mut left_ties, mut right_ties) =
        (0_i64, 0_i64, 0_i64, 0_i64);
    for i in 0..left.len() {
        for j in (i + 1)..left.len() {
            let a = (left[i] - left[j])
                .partial_cmp(&0.0)
                .map(|order| order as i32);
            let b = (right[i] - right[j])
                .partial_cmp(&0.0)
                .map(|order| order as i32);
            match (a, b) {
                (Some(a), Some(b)) if a == 0 && b == 0 => {
                    left_ties += 1;
                    right_ties += 1;
                }
                (Some(0), Some(_)) => left_ties += 1,
                (Some(_), Some(0)) => right_ties += 1,
                (Some(a), Some(b)) if a == b => concordant += 1,
                (Some(_), Some(_)) => discordant += 1,
                _ => {}
            }
        }
    }
    let pairs = (left.len() * left.len().saturating_sub(1) / 2) as i64;
    let denominator = (((pairs - left_ties) as f64) * ((pairs - right_ties) as f64)).sqrt();
    if denominator <= 0.0 {
        return 0.0;
    }
    (concordant - discordant) as f64 / denominator
}

/// Spearman's ρ: Pearson correlation of the mid-ranks.
pub fn spearman_rho(left: &[f64], right: &[f64]) -> f64 {
    let left = ranks(left);
    let right = ranks(right);
    let n = left.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_left = left.iter().sum::<f64>() / n;
    let mean_right = right.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let (mut variance_left, mut variance_right) = (0.0, 0.0);
    for (a, b) in left.iter().zip(&right) {
        covariance += (a - mean_left) * (b - mean_right);
        variance_left += (a - mean_left).powi(2);
        variance_right += (b - mean_right).powi(2);
    }
    let denominator = (variance_left * variance_right).sqrt();
    if denominator <= 0.0 {
        return 0.0;
    }
    covariance / denominator
}

/// Mid-ranks: tied values share the average of the ranks they span, which is
/// what keeps ρ from depending on the order ties happened to arrive in.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average = (start + end - 1) as f64 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Score;
    use async_trait::async_trait;

    const FIXTURES: &str = concat!(
        r#"{"scenario_id":"s","context":[{"role":"user","content":"solve"}],"#,
        r#""candidates":[{"messages":[{"role":"assistant","content":"good"}],"label":1.0},"#,
        r#"{"messages":[{"role":"assistant","content":"bad"}],"label":0.0}]}"#,
        "\n",
        r#"{"scenario_id":"t","rubric":"be terse","#,
        r#""candidates":[{"messages":[{"role":"assistant","content":"x"}],"label":0.2,"env_state":"diff"},"#,
        r#"{"messages":[{"role":"assistant","content":"y"}],"label":0.9}]}"#,
    );

    /// Scores by position: candidate 0 always best. Agrees with the first
    /// fixture's labels and contradicts the second's.
    struct ByPosition;

    #[async_trait]
    impl RewardBackend for ByPosition {
        async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
            Ok(group
                .trajectories
                .iter()
                .enumerate()
                .map(|(index, _)| Score {
                    value: 1.0 - index as f32 / group.trajectories.len() as f32,
                    valid: true,
                    explanation: None,
                    error: None,
                })
                .collect())
        }
    }

    #[test]
    fn a_fixture_becomes_a_group_the_engine_could_have_produced() {
        let fixtures = parse_fixtures(FIXTURES).unwrap();
        assert_eq!(fixtures.len(), 2);
        let group = fixtures[0].to_group(7).unwrap();
        assert_eq!(group.group_id, 7);
        // Shared opening plus the candidate's own turn, in that order.
        assert_eq!(group.trajectories[0].messages.len(), 2);
        assert_eq!(group.trajectories[0].messages[0].content, "solve");
        // The rubric and the environment state reach the judge the same way a
        // real rollout delivers them: through the trajectory metadata.
        let second = fixtures[1].to_group(0).unwrap();
        assert_eq!(
            second.trajectories[0].metadata["rubric"],
            serde_json::json!("be terse")
        );
        assert_eq!(
            second.trajectories[0].metadata["env_state"]["summary"],
            serde_json::json!("diff")
        );
        assert!(second.trajectories[1].metadata.get("env_state").is_none());
    }

    #[test]
    fn an_unrankable_fixture_is_refused_at_parse_time() {
        for source in [
            // One candidate: nothing to compare it to.
            r#"{"scenario_id":"s","candidates":[{"messages":[],"label":1.0}]}"#,
            r#"{"scenario_id":" ","candidates":[{"messages":[],"label":1.0},{"messages":[],"label":0.0}]}"#,
            r#"{"scenario_id":"s","candidates":[{"messages":[],"label":1.0},{"messages":[],"label":null}]}"#,
            r#"{"scenario_id":"s","candidates":[],"typo":1}"#,
            "",
        ] {
            assert!(parse_fixtures(source).is_err(), "accepted {source}");
        }
    }

    #[tokio::test]
    async fn the_report_separates_agreement_from_self_consistency() {
        let fixtures = parse_fixtures(FIXTURES).unwrap();
        let report = evaluate(Arc::new(ByPosition), &fixtures).await.unwrap();
        assert_eq!(report.scored_fixtures, 2);
        // Right on the first fixture, backwards on the second: they cancel.
        assert!(report.kendall_tau.abs() < 1e-6, "{report:?}");
        assert!(report.spearman_rho.abs() < 1e-6, "{report:?}");
        // A judge that ignores the content is perfectly consistent with itself,
        // which is exactly why the two numbers are reported apart: high
        // self-consistency alone says nothing about quality.
        assert!((report.self_kendall_tau - 1.0).abs() < 1e-6, "{report:?}");
        assert_eq!(report.degenerate_fraction, 0.0);
    }

    #[test]
    fn the_rank_correlations_agree_with_their_textbook_values() {
        let ascending = [1.0, 2.0, 3.0, 4.0];
        let descending = [4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau(&ascending, &ascending) - 1.0).abs() < 1e-9);
        assert!((kendall_tau(&ascending, &descending) + 1.0).abs() < 1e-9);
        assert!((spearman_rho(&ascending, &descending) + 1.0).abs() < 1e-9);
        // A monotone but non-linear relation is a perfect *rank* agreement,
        // which is the whole reason these are rank correlations.
        assert!((spearman_rho(&ascending, &[1.0, 10.0, 100.0, 1000.0]) - 1.0).abs() < 1e-9);
        // All-ties on one side: no ordering to agree with, and no division by
        // zero either.
        assert_eq!(kendall_tau(&[1.0, 1.0, 1.0], &ascending[..3]), 0.0);
        assert_eq!(spearman_rho(&[1.0, 1.0, 1.0], &ascending[..3]), 0.0);
    }
}
