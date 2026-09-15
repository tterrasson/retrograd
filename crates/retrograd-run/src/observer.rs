//! What a run has to say, without saying how to show it.
//!
//! The epoch loop reports progress through [`RunObserver`] rather than knowing
//! about terminal rendering. The CLI and HTTP server can therefore observe the
//! same orchestration with different renderers.
//!
//! The events carry *values*, never formatted strings - apart from the
//! diagnostics that are already opaque text produced by the runtime. Anything
//! derived from more than one event (a loss delta, a colour threshold, the last
//! evaluation reward shown on a progress bar) is display state and belongs to
//! the observer, not here.

use std::path::Path;
use std::time::Duration;

/// The shape of the loop that is about to start, announced once before the
/// first iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopPlan {
    Sft {
        epochs: u32,
        /// Whether an evaluation dataset was prepared, i.e. whether an
        /// `eval_loss` column will ever be filled.
        has_eval: bool,
        /// Optimizer steps in one epoch, `rows × (n_ctx / n_batch)`. An epoch
        /// is minutes to hours long, so this is what lets a frontend place the
        /// mid-epoch events of [`RunObserver::sft_step`] on a scale.
        steps_per_epoch: u64,
    },
    Rollout {
        algorithm: &'static str,
        updates: u32,
        epochs_per_update: u32,
        /// `updates × epochs_per_update`: the number of optimizer epochs, which
        /// is what the loop actually iterates over.
        total_epochs: u64,
    },
}

/// One completed SFT epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SftEpoch {
    pub epoch: u32,
    pub total_epochs: u32,
    pub global_step: u64,
    pub train_loss: f32,
    /// `NaN` when this epoch was not evaluated.
    pub eval_loss: f32,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
}

/// One completed optimizer step inside an SFT epoch, reported between the
/// [`SftEpoch`] events. An epoch on a real dataset is long enough that a run
/// showing nothing until it ends looks stalled; these carry what is known
/// mid-epoch, which is everything but the evaluation loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SftStep {
    pub epoch: u32,
    pub total_epochs: u32,
    pub global_step: u64,
    /// One-based position of this step inside its epoch.
    pub epoch_step: u64,
    pub steps_per_epoch: u64,
    /// Mean training loss over the epoch so far, i.e. the same running figure
    /// the epoch event reports once the epoch is complete.
    pub train_loss: f32,
    pub learning_rate: f32,
    /// Throughput of the steps since the previous report, not of the run.
    pub tokens_per_second: f32,
}

/// One completed optimizer epoch of a rollout algorithm (PPO or GRPO).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RolloutEpoch {
    /// Absolute, one-based update index as supplied by the algorithm. It
    /// survives a checkpoint resume, unlike a callback counter.
    pub update: u64,
    pub updates: u32,
    /// One-based position of this epoch inside its update.
    pub policy_epoch: u64,
    pub epochs_per_update: u32,
    /// Position in the whole run, `(update - 1) × epochs_per_update + policy_epoch`.
    pub absolute_epoch: u64,
    pub global_step: u64,
    pub train_loss: f32,
    /// `reward/mean`, `policy/kl` and `policy/clip_fraction` pulled out of the
    /// metric values; `NaN` when the algorithm did not report them.
    pub reward: f32,
    pub kl: f32,
    pub clip_fraction: f32,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
}

/// What one evaluation measured, and what it did to the early-stopping state.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluationReport {
    Sft {
        epoch: u32,
        loss: f64,
        perplexity: f64,
        outcome: EvalOutcome,
    },
    Rollout {
        update: u64,
        updates: u32,
        mean_reward: f32,
        reward_min: f32,
        reward_max: f32,
        examples: usize,
        outcome: EvalOutcome,
    },
}

/// What one evaluation did to the early-stopping state.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalOutcome {
    pub improved: bool,
    pub best: f64,
    pub stale: u32,
    pub patience: Option<u32>,
    /// Where the `best` checkpoint was written, when this evaluation improved
    /// and the checkpoint mode keeps one.
    pub saved: Option<std::path::PathBuf>,
    pub keep_training: bool,
}

/// Every point at which a run has something to report.
///
/// All methods default to doing nothing: an observer implements only what it
/// renders. The CLI implements nearly all of them, an HTTP broadcast a
/// different subset.
pub trait RunObserver {
    /// Free-form progress line. Ordering against the other events is
    /// significant and is the run's, not the observer's.
    fn info(&mut self, _message: &str) {}

    /// Opaque multi-line report produced by the runtime (backend report, LoRA
    /// description, preflight). Never parsed by the run itself.
    fn diagnostic(&mut self, _title: &str, _body: &str) {}

    fn model_load_started(&mut self) {}
    fn model_load_failed(&mut self) {}
    fn model_load_finished(&mut self, _elapsed: Duration) {}

    fn loop_started(&mut self, _plan: &LoopPlan) {}
    fn sft_epoch(&mut self, _epoch: &SftEpoch) {}
    /// Mid-epoch progress. Fires once per optimizer step, so an observer that
    /// renders it owns the throttling.
    fn sft_step(&mut self, _step: &SftStep) {}
    fn rollout_epoch(&mut self, _epoch: &RolloutEpoch) {}
    /// A rollout evaluation is about to run. It costs a full generation pass
    /// over the evaluation prompts, so it is worth announcing before it starts.
    fn evaluation_started(&mut self, _update: u64, _updates: u32) {}
    fn evaluation(&mut self, _report: &EvaluationReport) {}
    fn checkpoint_written(&mut self, _path: &Path) {}
    /// A step change in process or device memory, as phrased by
    /// [`retrograd_memory::MemoryTracker`].
    fn memory_note(&mut self, _note: &str) {}
    fn loop_finished(&mut self) {}
}

/// An observer that discards everything, for callers that only want the result.
pub struct SilentObserver;

impl RunObserver for SilentObserver {}
