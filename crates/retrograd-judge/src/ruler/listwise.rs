//! One request covering a set of trajectories, scored against each other.

use retrograd_agent_core::Result;

use crate::Score;
use crate::render::prompt::{self, RenderedGroup};

use super::Rubrics;
use super::{RulerJudge, cache_key};

impl RulerJudge {
    /// One listwise request covering `selection`, returning scores in the order
    /// of `selection`.
    ///
    /// The trajectories are presented in a permutation derived from
    /// `(group_id, request_index)` and the scores are put back in the caller's
    /// order before returning, so position bias no longer tracks rollout order.
    /// The permutation is part of the prompt and therefore part of the cache
    /// key, which is why it must be a pure function of those two numbers.
    pub(super) async fn score_selection(
        &self,
        rendered: &RenderedGroup,
        selection: &[usize],
        request_index: u64,
    ) -> Result<Vec<Score>> {
        let order = prompt::presentation_order(rendered.group_id, request_index, selection.len());
        let presented = order
            .iter()
            .map(|&position| selection[position])
            .collect::<Vec<_>>();
        let prompt =
            prompt::listwise_prompt(self.config.listwise_rubric(rendered), rendered, &presented)?;
        let key = cache_key(&self.config.model, &prompt);
        let scores = match self.cached(&key) {
            Some(scores) => scores,
            None => {
                let expected = presented.len();
                let scores = self
                    .request(
                        &prompt,
                        prompt::listwise_schema(),
                        "ruler_scores",
                        |content| prompt::parse_listwise(content, expected),
                    )
                    .await?;
                self.store(key, &scores)?;
                scores
            }
        };
        // `scores[position]` is the verdict on `selection[order[position]]`.
        let mut restored = vec![None; selection.len()];
        for (position, score) in order.into_iter().zip(scores) {
            restored[position] = Some(score);
        }
        Ok(restored
            .into_iter()
            .map(|score| score.expect("the permutation covers every position"))
            .collect())
    }

    pub(super) async fn score_listwise(&self, rendered: &RenderedGroup) -> Result<Vec<Score>> {
        let selection = (0..rendered.trajectories.len()).collect::<Vec<_>>();
        self.score_selection(rendered, &selection, 0).await
    }
}
