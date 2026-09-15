use std::future::Future;

// Tokio's clock rather than `std`'s: identical in production, but it lets the
// rollout deadline be tested under a paused clock instead of a real sleep.
use tokio::time::Instant;

use super::state::RolloutState;

/// Awaits `future` unless the rollout deadline lands first; `None` is the
/// deadline winning. Every stage of a rollout goes through here, so a hung
/// `reset`, decode, step, render or rescore costs the rollout its remaining
/// budget instead of running unbounded - a deadline read only between turns
/// bounds how many turns run, not how long they take.
pub(super) async fn before_deadline<T>(
    deadline: Option<Instant>,
    future: impl Future<Output = T>,
) -> Option<T> {
    match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, future).await.ok(),
        None => Some(future.await),
    }
}

pub(super) fn deadline_expired(stage: &str) -> String {
    format!("rollout deadline expired while {stage}")
}

/// Truncates the members a deadline interrupted, leaving the ones that had
/// already stopped or failed with the outcome they had.
///
/// A member cut off before it emitted a single token has no trainable action and
/// is refused by `Trajectory::validate` - so a deadline shorter than one decode
/// yields failures, not empty trajectories.
pub(super) fn truncate(states: &mut [RolloutState], indices: &[usize]) {
    for &index in indices {
        if !states[index].finished {
            states[index].stop(true);
        }
    }
}
