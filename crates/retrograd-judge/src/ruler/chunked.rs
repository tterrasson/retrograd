//! Groups too large for one request, split and realigned on a shared anchor.

use retrograd_agent_core::Result;

use crate::Score;
use crate::render::prompt::{self, RenderedGroup};

use super::Rubrics;
use super::RulerJudge;

impl RulerJudge {
    /// Splits the group into chunks that fit `max_request_chars` and scores each
    /// one. With `anchor`, the group's first trajectory joins every chunk it is
    /// not already in, and each chunk is shifted so that anchor lands on the
    /// same value everywhere - without it, scores from different requests are
    /// not on a common scale (see [`prompt::align_to_anchor`]).
    pub(super) async fn score_chunked(
        &self,
        rendered: &RenderedGroup,
        anchor: bool,
    ) -> Result<Vec<Score>> {
        let costs = rendered
            .trajectories
            .iter()
            .map(|trajectory| trajectory.chars)
            .collect::<Vec<_>>();
        let overhead = prompt::request_overhead(self.config.listwise_rubric(rendered), rendered);
        let anchor_cost = if anchor {
            costs.first().copied().unwrap_or(0)
        } else {
            0
        };
        let chunks = crate::render::plan_chunks(
            &costs,
            self.config.context.max_request_chars,
            overhead + anchor_cost,
            2,
        );
        if chunks.len() <= 1 {
            return self.score_listwise(rendered).await;
        }

        // The chunk that already contains trajectory 0 owns the reference scale;
        // every other chunk gets the anchor appended and is realigned onto it.
        let anchor_chunk = chunks
            .iter()
            .position(|chunk| chunk.contains(&0))
            .unwrap_or(0);
        let selections = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let mut selection = chunk.clone();
                if anchor && index != anchor_chunk {
                    selection.insert(0, 0);
                }
                selection
            })
            .collect::<Vec<_>>();
        // The request index is the chunk index, so two chunks of the same group
        // get different presentation orders and a member's position is not
        // decided by which chunk it landed in.
        let results = futures_util::future::join_all(
            selections
                .iter()
                .enumerate()
                .map(|(index, selection)| self.score_selection(rendered, selection, index as u64)),
        )
        .await;

        let mut output = vec![
            Score {
                value: 0.0,
                valid: false,
                explanation: None,
                error: Some("chunk request failed".into()),
            };
            rendered.trajectories.len()
        ];
        let mut scored: Vec<Option<Vec<Score>>> = Vec::with_capacity(selections.len());
        for (selection, result) in selections.iter().zip(results) {
            match result {
                Ok(scores) => scored.push(Some(scores)),
                Err(error) => {
                    for &index in selection {
                        output[index].error = Some(error.to_string());
                    }
                    scored.push(None);
                }
            }
        }

        // Write the reference chunk first so its anchor value is available to
        // shift the others. A failed reference chunk leaves the remaining chunks
        // unaligned rather than wrongly aligned: their scores stay as returned,
        // and the anchor itself is reported invalid.
        let reference = scored
            .get(anchor_chunk)
            .and_then(Option::as_ref)
            .and_then(|scores| {
                let position = selections[anchor_chunk]
                    .iter()
                    .position(|&index| index == 0)?;
                scores.get(position).map(|score| score.value)
            });
        for (chunk_index, scores) in scored.iter().enumerate() {
            let Some(scores) = scores else { continue };
            let selection = &selections[chunk_index];
            let mut values = scores.iter().map(|score| score.value).collect::<Vec<_>>();
            if anchor && chunk_index != anchor_chunk {
                let anchor_value = selection
                    .iter()
                    .position(|&index| index == 0)
                    .and_then(|position| values.get(position).copied());
                if let (Some(anchor_value), Some(reference)) = (anchor_value, reference) {
                    prompt::align_to_anchor(&mut values, anchor_value, reference);
                }
            }
            for ((score, value), &index) in scores.iter().zip(values).zip(selection) {
                // The anchor keeps the score from the chunk that owns it.
                if index == 0 && chunk_index != anchor_chunk {
                    continue;
                }
                output[index] = Score {
                    value,
                    ..score.clone()
                };
            }
        }
        Ok(output)
    }
}
