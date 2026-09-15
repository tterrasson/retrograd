//! The one place a running loop can be told to do something else.
//!
//! The constraint this module exists to honour: **the only safe interruption
//! point is the progress callback**. The
//! algorithms already take a `&mut dyn FnMut(&mut Trainer, Progress) ->
//! Result<bool>` whose `false` ends the loop cleanly; everything a control plane
//! wants to do - pause, cancel, checkpoint on demand, adjust a schedule - has to
//! happen inside that callback or not at all. Killing the thread is not an
//! option: it holds the model, the optimizer state and a half-built graph.
//!
//! So a run is polled, once per progress event, through [`RunControl::poll`].
//! The implementation may block (that is what a pause *is*: a callback that does
//! not return), may turn the knobs [`RunControls`] offers, and may ask the loop
//! to stop.
//!
//! **[`RunControls`] is a trait rather than the `Trainer` and `RunController`
//! themselves**, which is the one design decision here worth stating. A control
//! plane that took a `&mut Trainer` could only ever be exercised against a real
//! GGUF on a real device, so the pause/cancel/adjust state machine - the part
//! with all the edge cases - would be testable only in a GPU lane. Behind the
//! trait, the live implementation is a two-field struct in this crate and the
//! server drives the whole of its control logic in `fast-rust`.
//!
//! Two things a control plane deliberately cannot do. It cannot write a
//! checkpoint itself - it asks, with [`RunControls::request_checkpoint`], and the
//! existing boundary logic writes it at the next resumable point, because a
//! snapshot taken mid-epoch would replay or skip work on resume. And it cannot
//! touch anything that fixes the memory footprint: the budget was validated for
//! one geometry, and `n_ctx` or a LoRA rank changing under a live optimizer is a
//! different run, not an adjustment.
//!
//! Two things it *can* do beyond turning knobs, both added in step 9:
//! [`RunControls::evaluate`] and [`RunControls::generate`] read the live model.
//! They are here rather than on a second channel because there is only one
//! `Trainer` and only one thread that may touch it: an evaluation that ran
//! anywhere else would need a second copy of the weights.

use retrograd_config::CheckpointMode;
use retrograd_core::Result;

/// What the loop does after being polled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop the loop cleanly at this callback. The callback still runs to
    /// completion - its evaluation, its checkpoint and its metrics all happen,
    /// so a stop never discards work that was already done.
    Stop,
}

impl Flow {
    pub fn is_continue(self) -> bool {
        self == Self::Continue
    }
}

/// Where the loop is when it asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlPoint {
    /// Epoch (SFT) or absolute update (PPO/GRPO), as the algorithm reports it.
    pub iteration: u32,
    pub global_step: u64,
    /// Whether this callback is at a point the run could be resumed from - the
    /// end of an SFT epoch, or of a rollout update. A checkpoint is only ever
    /// written here, and a "cancel at the boundary" only ever fires here.
    pub at_boundary: bool,
}

/// One out-of-schedule evaluation, in whichever metric the algorithm has: a
/// loss for SFT, a mean reward for PPO and GRPO.
///
/// Deliberately **not** recorded by the controller. `record_evaluation` moves
/// `best`, `stale` and therefore early stopping, and may write the `best`
/// checkpoint; a client that could feed it would be able to end a run - or
/// overwrite its best adapter - just by asking what the loss is now. An ad-hoc
/// evaluation reads, the scheduled one decides.
///
/// The iteration and step this was taken at are not here: the caller polling the
/// control already knows them from its [`ControlPoint`], and a second copy could
/// disagree with the first.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdHocEvaluation {
    /// SFT: mean token-level cross-entropy over the evaluation dataset.
    pub loss: Option<f64>,
    pub perplexity: Option<f64>,
    /// PPO/GRPO: mean reward over the evaluated prompts.
    pub mean_reward: Option<f32>,
    pub reward_min: Option<f32>,
    pub reward_max: Option<f32>,
    /// Rows (SFT) or prompts (rollout) the evaluation covered.
    pub examples: u64,
}

/// One ad-hoc sampling request against the *live* adapter.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub prompt: String,
    /// Wrap `prompt` in the model's own chat template, as a single user turn.
    /// Off means the prompt is tokenized exactly as given, which is what a
    /// base-model completion wants.
    pub chat: bool,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    /// Sample the same prompt a second time with the adapter disabled, so a
    /// caller can see what training has changed. Doubles the cost, hence opt-in.
    pub include_base: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: u64,
    /// Sampled tokens, end-of-generation token included when one stopped it.
    pub tokens: u64,
    /// Present only when [`GenerationRequest::include_base`] asked for it.
    pub base_text: Option<String>,
}

/// The knobs the loop offers a control plane, and the only ones it does.
///
/// Every setter is a *schedule*: when to evaluate, how long to wait for an
/// improvement, when to snapshot, how fast to descend. None changes what is
/// computed or how much memory it takes. The two readers - [`Self::evaluate`]
/// and [`Self::generate`] - change nothing at all.
pub trait RunControls {
    /// Replaces the base learning rate the scheduler multiplies. Takes effect at
    /// the next optimizer step; warm-up and decay keep their shape around the new
    /// base.
    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()>;

    fn set_evaluation_every(&mut self, every_iterations: u32) -> Result<()>;

    fn set_patience(&mut self, patience: Option<u32>) -> Result<()>;

    fn set_checkpoint_every_steps(&mut self, every_steps: u64) -> Result<()>;

    fn set_checkpoint_mode(&mut self, mode: CheckpointMode) -> Result<()>;

    /// Asks for a checkpoint at the next resumable boundary. Never "now".
    fn request_checkpoint(&mut self);

    /// Runs the configured evaluation dataset now, out of schedule.
    ///
    /// Fails on a run that has no evaluation dataset - there is nothing to
    /// evaluate against, and inventing one would report a number about the
    /// training data.
    fn evaluate(&mut self) -> Result<AdHocEvaluation>;

    /// Samples one completion with the weights as they are at this callback.
    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationOutput>;
}

/// Consulted once per progress callback.
pub trait RunControl {
    fn poll(&mut self, controls: &mut dyn RunControls, at: ControlPoint) -> Result<Flow>;
}

/// The control of a run nobody is driving: the CLI's, and the default of
/// [`crate::execute`].
pub struct FreeRunning;

impl RunControl for FreeRunning {
    fn poll(&mut self, _controls: &mut dyn RunControls, _at: ControlPoint) -> Result<Flow> {
        Ok(Flow::Continue)
    }
}
