//! Two trajectories per request, aggregated into one score each.

use std::sync::atomic::Ordering;

use retrograd_agent_core::{Error, Result};

use crate::Score;
use crate::render::prompt::{self, Aggregation, PairOutcome, RenderedGroup, Winner};

use super::Rubrics;
use super::{RulerJudge, cache_key};

impl RulerJudge {
    /// Runs the comparison schedule and aggregates it. A failed comparison costs
    /// its two participants one match, not the group: as long as every
    /// trajectory kept at least one verdict, the group is still scored.
    pub(super) async fn score_pairwise(
        &self,
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
        let verdicts = futures::future::join_all(schedule.iter().map(|&(left, right)| {
            let rubric = rubric.as_str();
            async move {
                let (winner, explanation) = self.compare(rendered, rubric, left, right).await?;
                if !both_orders {
                    return Ok(PairOutcome {
                        left,
                        right,
                        winner,
                        explanation,
                    });
                }
                // The same two trajectories, exchanged. A judge that answers by
                // position rather than by content contradicts itself here, and
                // the contradiction is the measurement.
                let (swapped, swapped_explanation) =
                    self.compare(rendered, rubric, right, left).await?;
                let (winner, disagreed) = prompt::combine_orders(winner, swapped);
                self.stats.pairs_judged.fetch_add(1, Ordering::Relaxed);
                if disagreed {
                    self.stats
                        .position_disagreements
                        .fetch_add(1, Ordering::Relaxed);
                }
                Ok(PairOutcome {
                    left,
                    right,
                    winner,
                    explanation: if disagreed {
                        format!("order-dependent verdict: {explanation} / {swapped_explanation}")
                    } else {
                        explanation
                    },
                })
            }
        }))
        .await;

        let mut outcomes = Vec::new();
        let mut last_error = None;
        for verdict in verdicts {
            match verdict {
                Ok(outcome) => outcomes.push(outcome),
                Err(error) => last_error = Some(error),
            }
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

    /// One comparison, cached as a one-element score row: 1.0 = a wins,
    /// 0.0 = b wins, 0.5 = tie.
    async fn compare(
        &self,
        rendered: &RenderedGroup,
        rubric: &str,
        left: usize,
        right: usize,
    ) -> Result<(Winner, String)> {
        let prompt = prompt::pairwise_prompt(rubric, rendered, left, right)?;
        let key = cache_key(&self.config.model, &prompt);
        if let Some(cached) = self.cached(&key) {
            let winner = match cached.first().map(|score| score.value) {
                Some(value) if value > 0.75 => Winner::A,
                Some(value) if value < 0.25 => Winner::B,
                Some(_) => Winner::Tie,
                None => return Err(Error::Reward("empty cached judge verdict".into())),
            };
            return Ok((winner, cached[0].explanation.clone().unwrap_or_default()));
        }
        let (winner, explanation) = self
            .request(
                &prompt,
                prompt::pairwise_schema(),
                "ruler_pairwise",
                prompt::parse_pairwise,
            )
            .await?;
        self.store(
            key,
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
        )?;
        Ok((winner, explanation))
    }
}
