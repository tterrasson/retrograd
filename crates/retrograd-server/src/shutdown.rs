//! Stopping the process without losing a run.
//!
//! The naive shutdown - stop accepting, drop everything - loses whatever a live
//! run has computed since its last checkpoint, which on a long job is hours. So
//! the sequence is the one a client would issue by hand:
//!
//! 1. tell every live run to **checkpoint at its next boundary**, so there is a
//!    resumable point on disk;
//! 2. tell it to **stop at that boundary**, which is the ordinary boundary
//!    cancellation and therefore lets the callback finish - evaluation, checkpoint
//!    and metrics all happen;
//! 3. wait, bounded, for the runs to reach a terminal state.
//!
//! A boundary, not `now`: `{"at": "now"}` cannot checkpoint, and a
//! shutdown that stopped mid-epoch would be exactly the lost work this exists to
//! avoid. The cost is the wait - up to one iteration per run - and the bound is
//! what keeps a hung run from making the process unkillable. On timeout the runs
//! are left as they are: the next start reads them back as `interrupted`,
//! which is true and is what `fork_from` picks up.

use std::time::{Duration, Instant};

use crate::dto;
use crate::runtime::control::{CancelAt, RunCommand};
use crate::state::AppState;

/// How often the drain checks whether the runs have finished. A boundary is
/// seconds to minutes away, so a coarse poll costs nothing and a fine one only
/// spins.
const POLL: Duration = Duration::from_millis(200);

/// Asks every live run to checkpoint and stop, then waits up to `bound`.
///
/// Returns the number of runs still alive when it gave up - zero for a clean
/// drain.
pub async fn drain(state: &AppState, bound: Duration) -> usize {
    let live: Vec<_> = state
        .registry
        .live_handles()
        .into_iter()
        .filter(|handle| handle.control().is_some())
        .collect();
    if live.is_empty() {
        return 0;
    }
    tracing::info!(runs = live.len(), "draining live runs before shutdown");
    for handle in &live {
        let Some(control) = handle.control() else {
            continue;
        };
        // Both commands, in this order. `Checkpoint` on its own would not stop the
        // run; `Cancel { checkpoint: true }` alone would be enough, but sending the
        // request first means a run that reaches its boundary between the two
        // messages still writes one.
        let _ = control.send_waiting(RunCommand::Checkpoint).await;
        let _ = control
            .send_waiting(RunCommand::Cancel {
                at: CancelAt::Boundary,
                checkpoint: true,
            })
            .await;
        handle.emit(dto::RunEventPayload::Log {
            message: "the server is shutting down: checkpointing and stopping at the next \
                      boundary"
                .to_string(),
        });
        handle.transition(dto::RunStatus::Cancelling);
    }

    let deadline = Instant::now() + bound;
    loop {
        let remaining = live
            .iter()
            .filter(|handle| !handle.status().is_terminal())
            .count();
        if remaining == 0 {
            tracing::info!("every run stopped at a boundary");
            return 0;
        }
        if Instant::now() >= deadline {
            tracing::warn!(
                runs = remaining,
                "giving up on the drain; the next start will read these runs as interrupted"
            );
            return remaining;
        }
        tokio::time::sleep(POLL).await;
    }
}
