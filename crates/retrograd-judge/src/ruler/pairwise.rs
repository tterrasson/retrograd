//! Two trajectories per request, aggregated into one score each.

use std::sync::atomic::Ordering;

use retrograd_agent_core::{Error, Result};

use crate::Score;
use crate::render::prompt::{self, Aggregation, PairOutcome, RenderedGroup, Winner};
use crate::transport::{JudgeRequest, JudgeTransport};

use super::{Rubrics, RulerCore, cache_key};

/// One planned comparison: the prompt, its cache key, and whether this request
/// is the swapped twin of another.
struct PlannedComparison {
    left: usize,
    right: usize,
    swapped: bool,
    prompt: String,
    key: String,
}

impl RulerCore {
    /// Runs the comparison schedule and aggregates it, as one wave through the
    /// transport. A failed comparison costs its two participants one match, not
    /// the group: as long as every trajectory kept at least one verdict, the
    /// group is still scored.
    pub(super) fn score_pairwise_via(
        &self,
        transport: &mut dyn JudgeTransport,
        rendered: &RenderedGroup,
        max_pairs: Option<usize>,
        both_orders: bool,
        aggregation: Aggregation,
    ) -> Result<Vec<Score>> {
        let size = rendered.trajectories.len();
        let schedule = prompt::pair_schedule(size, max_pairs);
        if schedule.is_empty() {
            return Err(Error::invalid(
                "pairwise judging needs at least two trajectories",
            ));
        }
        let rubric = self.config.pairwise_rubric(rendered);
        // Plan every comparison - and its swapped twin when both orders are
        // asked for - into one wave.
        let mut planned: Vec<PlannedComparison> = Vec::new();
        for &(left, right) in &schedule {
            for swapped in [false, true] {
                if swapped && !both_orders {
                    continue;
                }
                let (a, b) = if swapped {
                    (right, left)
                } else {
                    (left, right)
                };
                let prompt = prompt::pairwise_prompt(&rubric, rendered, a, b)?;
                let key = cache_key(&self.cache_identity(), &prompt);
                planned.push(PlannedComparison {
                    left,
                    right,
                    swapped,
                    prompt,
                    key,
                });
            }
        }
        // Cached comparisons answer without a request; the rest form the wave.
        let mut verdicts: Vec<Option<(Winner, String)>> = vec![None; planned.len()];
        let mut pending: Vec<usize> = Vec::new();
        let mut requests = Vec::new();
        for (index, comparison) in planned.iter().enumerate() {
            if let Some(cached) = self.cached(&comparison.key) {
                let winner = match cached.first().map(|score| score.value) {
                    Some(value) if value > 0.75 => Winner::A,
                    Some(value) if value < 0.25 => Winner::B,
                    Some(_) => Winner::Tie,
                    None => {
                        return Err(Error::Reward("empty cached judge verdict".into()));
                    }
                };
                verdicts[index] = Some((winner, cached[0].explanation.clone().unwrap_or_default()));
            } else {
                pending.push(index);
                requests.push(JudgeRequest::structured(
                    comparison.prompt.clone(),
                    prompt::pairwise_schema(),
                    "ruler_pairwise",
                    self.config.temperature,
                ));
            }
        }
        let mut last_error = None;
        if !pending.is_empty() {
            let replies = self.complete_wave(transport, requests, |_, _, content| {
                prompt::parse_pairwise(content)
            });
            for (&index, reply) in pending.iter().zip(replies) {
                let comparison = &planned[index];
                // A failed comparison costs its pair one match, not the group:
                // the verdict stays empty and the pair is skipped below.
                let (winner, explanation) = match reply {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        last_error = Some(error);
                        continue;
                    }
                };
                // A cached verdict is stored as a one-element score row; a cache
                // that cannot be written costs the pair its match, as it always
                // did, not the group.
                if let Err(error) = self.store(
                    comparison.key.clone(),
                    &[Score {
                        value: match winner {
                            Winner::A => 1.0,
                            Winner::B => 0.0,
                            Winner::Tie => 0.5,
                        },
                        valid: true,
                        explanation: Some(explanation.clone()),
                        error: None,
                    }],
                ) {
                    last_error = Some(error);
                    continue;
                }
                verdicts[index] = Some((winner, explanation));
            }
        }

        // Fold the wave back into per-pair outcomes: the forward verdict, and the
        // swapped one read back through the exchange when both orders ran. A
        // judge that answers by position rather than by content contradicts
        // itself there, and the contradiction is the measurement.
        let mut outcomes = Vec::new();
        let mut by_pair: Vec<Option<(Winner, String)>> = vec![None; schedule.len()];
        let mut by_pair_swapped: Vec<Option<(Winner, String)>> = vec![None; schedule.len()];
        for (index, comparison) in planned.iter().enumerate() {
            let Some((winner, explanation)) = verdicts[index].clone() else {
                continue;
            };
            let pair = schedule
                .iter()
                .position(|&(left, right)| left == comparison.left && right == comparison.right)
                .expect("planned from the schedule");
            if comparison.swapped {
                by_pair_swapped[pair] = Some((winner, explanation));
            } else {
                by_pair[pair] = Some((winner, explanation));
            }
        }
        for (pair, &(left, right)) in schedule.iter().enumerate() {
            let Some((winner, explanation)) = by_pair[pair].clone() else {
                last_error = Some(Error::Reward(format!(
                    "the comparison of trajectories {left} and {right} failed"
                )));
                continue;
            };
            let (winner, explanation) = match by_pair_swapped[pair].clone() {
                Some((swapped, swapped_explanation)) if both_orders => {
                    let (winner, disagreed) = prompt::combine_orders(winner, swapped);
                    self.stats.pairs_judged.fetch_add(1, Ordering::Relaxed);
                    if disagreed {
                        self.stats
                            .position_disagreements
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    (
                        winner,
                        if disagreed {
                            format!(
                                "order-dependent verdict: {explanation} / {swapped_explanation}"
                            )
                        } else {
                            explanation
                        },
                    )
                }
                _ => (winner, explanation),
            };
            outcomes.push(PairOutcome {
                left,
                right,
                winner,
                explanation,
            });
        }
        if outcomes.is_empty() {
            return Err(last_error
                .unwrap_or_else(|| Error::Reward("every pairwise comparison failed".into())));
        }
        Ok(match aggregation {
            Aggregation::WinRate => prompt::scores_from_pairs(size, &outcomes),
            Aggregation::BradleyTerry => prompt::bradley_terry_scores(size, &outcomes),
        })
    }
}
