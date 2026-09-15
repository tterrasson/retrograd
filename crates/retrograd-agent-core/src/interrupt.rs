//! What an interrupt means to a run that owns a world.
//!
//! A training process that is killed leaves its containers behind: no `Drop`
//! runs, no [`EnvironmentFactory::shutdown`](crate::EnvironmentFactory::shutdown)
//! is reached, and the pool's own cleanup never happens. The containers of an
//! agentic run do not exit on their own either - they idle on a `sleep` loop,
//! so `AutoRemove` never fires and only the next run's startup sweep collects
//! them.
//!
//! So an interrupt is escalated rather than obeyed:
//!
//! - the **first** one asks the loop to stop at the end of the update it is in.
//!   The update keeps its rollouts, its judging, its evaluation and its
//!   checkpoint, the pool is drained through the ordinary shutdown path, and the
//!   adapter is saved. It is not instant: one update is a full round of rollouts
//!   plus a judge pass, which is minutes;
//! - the **second** one - and any `SIGTERM`, which comes from something that
//!   already decided - gives up on the update and destroys the world now.
//!
//! This module is only the state. Deciding which signal escalates, and running
//! the cleanup, belongs to whoever owns the process: see
//! [`EnvironmentFactory::force_cleanup`](crate::EnvironmentFactory::force_cleanup)
//! for the other half.

use std::sync::atomic::{AtomicU8, Ordering};

/// How far the interrupt has been escalated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Nothing has been asked.
    Running,
    /// Stop at the next update boundary, keeping everything this update did.
    Graceful,
    /// Do not wait for the boundary: tear the world down and exit.
    Force,
}

static LEVEL: AtomicU8 = AtomicU8::new(0);

/// Records one interrupt and says what it asks for.
///
/// The first call asks for a clean stop, every later one for an immediate
/// teardown. `Force` is sticky: once escalated, a run never de-escalates back to
/// a graceful stop.
pub fn escalate() -> Level {
    match LEVEL.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => Level::Graceful,
        Err(_) => force(),
    }
}

/// Escalates straight to [`Level::Force`], skipping the graceful stage.
///
/// For a signal that is not a request to reconsider - `SIGTERM` from an init
/// system or an orchestrator - where waiting out an update would only delay the
/// kill that follows.
pub fn force() -> Level {
    LEVEL.store(2, Ordering::SeqCst);
    Level::Force
}

pub fn level() -> Level {
    match LEVEL.load(Ordering::SeqCst) {
        0 => Level::Running,
        1 => Level::Graceful,
        _ => Level::Force,
    }
}

/// Whether the loop should stop at its next boundary. True from the first
/// interrupt on, at either level.
pub fn stop_requested() -> bool {
    LEVEL.load(Ordering::SeqCst) > 0
}

/// Test-only: the state is process-wide, so a test that escalates has to put it
/// back or the next one starts interrupted.
#[cfg(test)]
fn reset() {
    LEVEL.store(0, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole state machine in one test, because the state is a global: two
    /// tests over it would race each other rather than test it.
    #[test]
    fn the_first_interrupt_is_graceful_and_force_never_goes_back() {
        reset();
        assert_eq!(level(), Level::Running);
        assert!(!stop_requested());

        assert_eq!(escalate(), Level::Graceful);
        assert_eq!(level(), Level::Graceful);
        assert!(stop_requested());

        assert_eq!(escalate(), Level::Force);
        assert_eq!(escalate(), Level::Force);
        assert_eq!(level(), Level::Force);

        reset();
        // A SIGTERM skips the stage a Ctrl+C would have spent.
        assert_eq!(force(), Level::Force);
        assert_eq!(escalate(), Level::Force);
        reset();
    }
}
