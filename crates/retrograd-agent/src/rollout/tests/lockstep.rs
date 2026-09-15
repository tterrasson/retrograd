//! The batched loop: a group must produce, token for token, what the same
//! rollouts run one by one produce.

use std::sync::Arc;

use super::*;

#[derive(Default)]
struct BatchedScoringPolicy {
    score_batches: Mutex<Vec<usize>>,
}

#[async_trait]
impl Policy for BatchedScoringPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|_| PolicyGeneration {
                tokens: vec![11],
                text: "answer".into(),
                stopped_at_eog: true,
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, _train_mask: Vec<bool>) -> Result<Vec<f32>> {
        panic!("the rollout must use the batch scoring surface")
    }

    async fn score_masked_batch(
        &self,
        sequences: Vec<(Vec<i32>, Vec<bool>)>,
    ) -> Result<Vec<Vec<f32>>> {
        self.score_batches.lock().unwrap().push(sequences.len());
        Ok(sequences
            .iter()
            .map(|(_, mask)| logprobs(mask, -0.5))
            .collect())
    }
}

#[tokio::test]
async fn a_group_is_identical_to_the_same_rollouts_run_one_by_one() {
    // The whole point of the lockstep loop: it batches the decoding without
    // changing a single token of what the sequential loop produced.
    let group_size = 8;
    let engine = engine_with(Arc::new(DivergentPolicy::default()), group_limits());
    let group = engine
        .rollout_group(&scenario(), group_size, 40)
        .await
        .unwrap()
        .group
        .unwrap();

    let mut sequential = Vec::new();
    for member in 0..group_size {
        sequential.push(
            engine
                .rollout(&scenario(), 40 + member as u64)
                .await
                .unwrap(),
        );
    }
    for (batched, sequential) in group.trajectories.iter().zip(&sequential) {
        assert_eq!(batched.tokens, sequential.tokens);
        assert_eq!(batched.train_mask, sequential.train_mask);
        assert_eq!(batched.truncated, sequential.truncated);
    }
    // Guards against a vacuous comparison: the members really do diverge,
    // some stopping at turn 0 while others continue into turn 1.
    let lengths = group
        .trajectories
        .iter()
        .map(|trajectory| trajectory.tokens.len())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(lengths.len(), 2, "members must diverge: {lengths:?}");
}

#[tokio::test]
async fn turn_zero_shares_one_prompt_and_later_turns_go_continuous() {
    let policy = Arc::new(DivergentPolicy::default());
    let engine = engine_with(policy.clone(), group_limits());
    engine.rollout_group(&scenario(), 8, 40).await.unwrap();
    assert_eq!(
        *policy.shared_batches.lock().unwrap(),
        [8],
        "turn 0 must decode the shared prompt once for the whole group"
    );
    let continuous = policy.continuous_batches.lock().unwrap().clone();
    assert_eq!(
        continuous.len(),
        1,
        "turn 1 must be a single continuous batch: {continuous:?}"
    );
    assert!(
        continuous[0] > 0 && continuous[0] < 8,
        "only the members still alive belong in turn 1: {continuous:?}"
    );
}

#[tokio::test]
async fn a_group_larger_than_the_sequence_capacity_is_chunked() {
    let group_size = 8;
    let chunked = Arc::new(DivergentPolicy {
        capacity: 3,
        ..Default::default()
    });
    let engine = engine_with(chunked.clone(), group_limits());
    let batched = engine
        .rollout_group(&scenario(), group_size, 40)
        .await
        .unwrap()
        .group
        .unwrap();

    let uncapped = engine_with(Arc::new(DivergentPolicy::default()), group_limits());
    let reference = uncapped
        .rollout_group(&scenario(), group_size, 40)
        .await
        .unwrap()
        .group
        .unwrap();

    assert_eq!(batched.group_id, reference.group_id);
    for (chunked, reference) in batched.trajectories.iter().zip(&reference.trajectories) {
        assert_eq!(chunked.tokens, reference.tokens);
        assert_eq!(chunked.train_mask, reference.train_mask);
    }
    assert_eq!(
        *chunked.shared_batches.lock().unwrap(),
        [3, 3, 2],
        "a group above the sequence capacity must split into waves"
    );
}

#[tokio::test]
async fn completed_group_is_rescored_in_one_batch() {
    let policy = Arc::new(BatchedScoringPolicy::default());
    let engine = engine_with(policy.clone(), group_limits());
    let group = engine
        .rollout_group(&scenario(), 8, 40)
        .await
        .unwrap()
        .group
        .unwrap();
    assert_eq!(group.trajectories.len(), 8);
    assert_eq!(*policy.score_batches.lock().unwrap(), [8]);
}
