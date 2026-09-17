//! Live export of what a rollout algorithm trains on: prompts, completions or
//! conversations, rewards, advantages and what the optimizer actually used.
//!
//! The training loops only know [`TrajectoryObserver`] and the plain types of
//! [`batch`]. [`ObserveSink`] implements it with a bounded channel and a
//! writer thread, so a slow or broken disk costs dropped batches, never a
//! stalled update. The directory it writes is described in
//! `docs/training/observe.md`.

pub mod batch;
mod error;
mod record;
mod sink;
#[cfg(test)]
mod tests;
mod writer;

pub use batch::{
    Algorithm, ObserveBatch, ObservedMessage, ObservedPrompt, ObservedRollout, ObservedToolCall,
    OutcomeEntry, RolloutBatch, RolloutContent, RunInfo, SelectionEntry, SkipReason, StepReward,
    UpdateStatus, UpdateSummary,
};
pub use error::ObserveError;
pub use sink::{ObserveSink, SinkConfig};

/// Schema version written on every line of `observe.jsonl`.
pub const SCHEMA_VERSION: u32 = 1;

/// Receives the trajectories of an update. Implemented by the sink; the
/// training loops know nothing else.
pub trait TrajectoryObserver: Send + Sync {
    /// Whether the texts of this one-based update are exported. When it is
    /// false, the loop builds nothing.
    fn wants(&self, update: u32) -> bool;
    /// Never blocks and never fails: a batch that cannot be queued is counted
    /// and dropped.
    fn observe(&self, batch: ObserveBatch);
}
