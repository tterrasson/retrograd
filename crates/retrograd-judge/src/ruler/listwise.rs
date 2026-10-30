//! One request covering a set of trajectories, scored against each other.

use retrograd_agent_core::Result;

use crate::Score;
use crate::render::prompt::{self, RenderedGroup};
use crate::transport::{JudgeRequest, JudgeTransport};

use super::{Rubrics, RulerCore, cache_key};

/// One planned listwise request: the presentation order, the prompt, its cache
/// key, and how many scores the answer must carry.
pub(super) struct SelectionRequest {
    pub order: Vec<usize>,
    pub prompt: String,
    pub key: String,
    pub expected: usize,
}

/// Puts scores back in the caller's order: `scores[position]` is the verdict on
/// `selection[order[position]]`.
pub(super) fn restore_order(order: Vec<usize>, scores: Vec<Score>) -> Vec<Score> {
    let mut restored = vec![None; order.len()];
    for (position, score) in order.into_iter().zip(scores) {
        restored[position] = Some(score);
    }
    restored
        .into_iter()
        .map(|score| score.expect("the permutation covers every position"))
        .collect()
}

impl RulerCore {
    /// Plans one listwise request over `selection`.
    ///
    /// The trajectories are presented in a permutation derived from
    /// `(group_id, request_index)` and the scores are put back in the caller's
    /// order before returning, so position bias no longer tracks rollout order.
    /// The permutation is part of the prompt and therefore part of the cache
    /// key, which is why it must be a pure function of those two numbers.
    pub(super) fn plan_selection(
        &self,
        rendered: &RenderedGroup,
        selection: &[usize],
        request_index: u64,
    ) -> Result<SelectionRequest> {
        let order = prompt::presentation_order(rendered.group_id, request_index, selection.len());
        let presented = order
            .iter()
            .map(|&position| selection[position])
            .collect::<Vec<_>>();
        let prompt =
            prompt::listwise_prompt(self.config.listwise_rubric(rendered), rendered, &presented)?;
        let expected = presented.len();
        let key = cache_key(&self.cache_identity(), &prompt);
        Ok(SelectionRequest {
            order,
            prompt,
            key,
            expected,
        })
    }

    /// Builds the wire request for a planned selection, at the judge's
    /// configured temperature.
    pub(super) fn selection_request(&self, planned: &SelectionRequest) -> JudgeRequest {
        JudgeRequest::structured(
            planned.prompt.clone(),
            prompt::listwise_schema(),
            "ruler_scores",
            self.config.temperature,
        )
    }

    /// Executes planned selections as one wave (or one request each when only
    /// one is missing), parses every reply as a listwise answer, caches it, and
    /// returns the scores restored to each selection's order.
    pub(super) fn run_selections(
        &self,
        transport: &mut dyn JudgeTransport,
        planned: Vec<SelectionRequest>,
    ) -> Vec<Result<Vec<Score>>> {
        let mut results: Vec<Option<Result<Vec<Score>>>> =
            (0..planned.len()).map(|_| None).collect();
        let mut pending: Vec<usize> = Vec::new();
        let mut requests = Vec::new();
        for (index, selection) in planned.iter().enumerate() {
            if let Some(scores) = self.cached(&selection.key) {
                results[index] = Some(Ok(restore_order(selection.order.clone(), scores)));
            } else {
                pending.push(index);
                requests.push(self.selection_request(selection));
            }
        }
        if !pending.is_empty() {
            let expected: Vec<usize> = pending.iter().map(|&i| planned[i].expected).collect();
            let replies = self.complete_wave(transport, requests, |slot, _, content| {
                prompt::parse_listwise(content, expected[slot])
            });
            for (&index, reply) in pending.iter().zip(replies) {
                let selection = &planned[index];
                // A verdict is cached before it is handed back: a cache that
                // cannot be written fails the request, as it always did.
                let outcome = match reply {
                    Ok(scores) => self
                        .store(selection.key.clone(), &scores)
                        .map(|()| restore_order(selection.order.clone(), scores)),
                    Err(error) => Err(error),
                };
                results[index] = Some(outcome);
            }
        }
        results
            .into_iter()
            .map(|result| result.expect("every selection ends with a result"))
            .collect()
    }

    pub(super) fn score_listwise_via(
        &self,
        transport: &mut dyn JudgeTransport,
        rendered: &RenderedGroup,
    ) -> Result<Vec<Score>> {
        let selection = (0..rendered.trajectories.len()).collect::<Vec<_>>();
        let planned = vec![self.plan_selection(rendered, &selection, 0)?];
        let mut results = self.run_selections(transport, planned);
        results.pop().expect("one selection planned")
    }
}
