use retrograd_dataset::chat_template::TurnPieces;

use crate::Error;
use crate::trajectory::{Message, Step, StepKind};

/// Per-member state of a lockstep group rollout. The group loop advances one
/// state per member, applying exactly the same rules turn by turn.
pub(super) struct RolloutState {
    pub(super) seed: u64,
    pub(super) messages: Vec<Message>,
    pub(super) tokens: Vec<i32>,
    /// What the template writes around each turn the conversation has been
    /// rendered past, checked once against what the turn was sampled after and
    /// as. Every later render must reproduce these byte for byte: anything else
    /// means the template rewrote a turn the member has already generated
    /// against.
    pub(super) checked: Vec<TurnPieces>,
    /// The text of the framing piece last folded into `tokens` - the
    /// generation prompt the turn being sampled follows.
    pub(super) committed: String,
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
    /// For each assistant turn, the prose its parser read out of it.
    pub(super) assistant_prose: Vec<String>,
    /// For each assistant turn, the reasoning block its parser split out.
    pub(super) assistant_reasoning: Vec<String>,
    /// The turn that ran out of tokens was still reasoning: no answer, no
    /// call, only the reasoning block.
    pub(super) cut_while_reasoning: bool,
    /// Assistant turns with a parse error or no call where one was expected,
    /// over the whole trajectory - unlike `failed_turns`, never reset.
    pub(super) invalid_turns: usize,
    /// Of the turns in `checked`, those the template re-renders differently
    /// from how they were sampled, and where the first one parts ways.
    pub(super) normalized_turns: usize,
    pub(super) first_normalization: Option<String>,
    pub(super) failure: Option<Error>,
}

impl RolloutState {
    /// `opening` is the rendered scenario, as tokens and as the text the first
    /// turn is checked against.
    pub(super) fn new(seed: u64, messages: Vec<Message>, opening: (Vec<i32>, String)) -> Self {
        let (opening, committed) = opening;
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
            tokens: opening,
            checked: Vec::new(),
            committed,
            old_logprobs: None,
            truncated: false,
            finished: false,
            env_rewarded: false,
            environment_done: false,
            failed_turns: 0,
            assistant_prose: Vec::new(),
            assistant_reasoning: Vec::new(),
            cut_while_reasoning: false,
            invalid_turns: 0,
            normalized_turns: 0,
            first_normalization: None,
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
