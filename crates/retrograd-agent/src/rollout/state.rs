use crate::Error;
use crate::trajectory::{Message, Step, StepKind};

/// Per-member state of a lockstep group rollout. The group loop advances one
/// state per member, applying exactly the same rules turn by turn.
pub(super) struct RolloutState {
    pub(super) seed: u64,
    pub(super) messages: Vec<Message>,
    pub(super) tokens: Vec<i32>,
    /// The template's framing already folded into `tokens`, one piece per gap
    /// around the sampled turns. Re-rendering after a tool turn must reproduce
    /// these byte for byte and add exactly one piece: anything else means the
    /// template rewrote a turn the member has already generated against.
    pub(super) framing: Vec<Vec<i32>>,
    pub(super) train_mask: Vec<bool>,
    pub(super) steps: Vec<Step>,
    /// For each entry of `steps`, the messages it produced.
    pub(super) step_messages: Vec<Vec<usize>>,
    pub(super) old_logprobs: Option<Vec<f32>>,
    pub(super) truncated: bool,
    /// The member no longer generates: it stopped, ran out of budget, or
    /// failed. Only live members enter the next decode batch.
    pub(super) finished: bool,
    /// The environment graded at least one step of this trajectory, so it - not
    /// the judge - decides its reward.
    pub(super) env_rewarded: bool,
    /// The environment, rather than the policy budget, declared the episode
    /// complete. A configured verifier that never reaches this state is a
    /// failed attempt and receives its failure reward.
    pub(super) environment_done: bool,
    /// Turns in a row that named no valid tool call, for
    /// `RolloutLimits::max_failed_turns`. Reset by any turn that did.
    pub(super) failed_turns: usize,
    pub(super) failure: Option<Error>,
}

impl RolloutState {
    pub(super) fn new(seed: u64, messages: Vec<Message>, opening: Vec<i32>) -> Self {
        let context_len = opening.len();
        Self {
            seed,
            step_messages: vec![(0..messages.len()).collect()],
            messages,
            train_mask: vec![false; context_len],
            steps: vec![Step {
                kind: StepKind::Context,
                token_range: (0, context_len),
                reward: None,
            }],
            tokens: opening.clone(),
            framing: vec![opening],
            old_logprobs: None,
            truncated: false,
            finished: false,
            env_rewarded: false,
            environment_done: false,
            failed_turns: 0,
            failure: None,
        }
    }

    pub(super) fn fail(&mut self, error: Error) {
        self.failure = Some(error);
        self.finished = true;
    }

    pub(super) fn push_step(&mut self, step: Step, messages: Vec<usize>) {
        self.steps.push(step);
        self.step_messages.push(messages);
    }

    pub(super) fn stop(&mut self, truncated: bool) {
        self.truncated |= truncated;
        self.finished = true;
    }
}
