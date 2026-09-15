//! The run runtime: what turns a resolved configuration into a live run.
//!
//! A `Trainer` carries raw FFI
//! pointers and **never crosses an OS-thread boundary**, so a run owns one
//! dedicated thread from the first byte of the model to the last checkpoint;
//! and a run is only interruptible inside its progress callback, so nothing here
//! ever kills a thread.
//!
//! ```text
//!   handler ──▶ RunRegistry::create ──▶ worker::spawn
//!                    │                        │  tokio task: waits for the
//!                    │                        │  device permit (FIFO queue)
//!                    │                        ▼
//!                    │                 std::thread ──▶ RunEngine::execute
//!                    │                        │            │ RunObserver
//!                    ▼                        ▼            ▼
//!               RunHandle ◀───── state ──── journal (run.json, events.jsonl)
//! ```
//!
//! `retrograd_run::execute` is entirely synchronous, so a runtime around it
//! would drive nothing. The constraint is thread affinity, and a plain
//! `std::thread::spawn` gives that with one less moving part. An agentic run
//! that needs to await inside the loop builds its own runtime where it needs it,
//! exactly as the CLI does today.

pub mod control;
pub mod engine;
pub mod journal;
pub mod registry;
pub mod worker;

pub use control::{Adjustments, CancelAt, RunCommand};
pub use engine::{RunEngine, TrainingEngine};
pub use registry::{RunHandle, RunRegistry, RunState};

/// Wall-clock helpers. One place, so every timestamp in the API and in the
/// journal is the same clock in the same unit.
pub fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

pub fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}
