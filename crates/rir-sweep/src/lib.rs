//! `rir-sweep` - the offline schedule search.
//!
//! The schedule table is handwritten, and it should stay handwritten: a block,
//! a width or a reduction topology is a decision a reader has to be able to
//! find, with the measurement that justified it beside it. What is needed is
//! not an autotuner - nothing here runs at dispatch time, and nothing here
//! writes a table - but the loop that a human performs by hand today: build a
//! small product of candidate schedules, time each of them against the pair's
//! current lowering on the same shapes, and say which of them earns a row.
//!
//! The four crates it reaches for are the four the chain already has, and that
//! is deliberate: candidates come out of the same `Schedule` constructors the
//! production table calls, they are lowered by the same `lower`, emitted by the
//! same emitters, and run through the same `rir-runtime` device harness as
//! `device_timing`. A search that reproduced any of those would be searching a
//! different compiler than the one that ships.
//!
//! What it refuses to do matters as much as what it does:
//!
//! - a candidate that does not agree with the lowering it would replace is
//!   refused before any timing is read - two lowerings of one kernel compute the
//!   same bytes or one of them is a bug, and a faster wrong kernel is the one
//!   result a sweep must never propose;
//! - a candidate whose gain does not clear the run-to-run spread of the two
//!   measurements is refused as **noise**, not rounded up;
//! - a candidate that wins on some shapes and loses on others is refused unless
//!   the shapes it wins on are separable by a `ShapeRule` - the registry
//!   publishes a conjunction of extent intervals and nothing else, so a domain
//!   that cannot be written as one is a domain the dispatcher cannot evaluate.
//!
//! And what it produces is a **proposal**: a printed row in the vocabulary of
//! `schedule::table`, to be read, argued with, and committed by hand. Promotion
//! is still `scripts/test-rir.sh` (`docs/engineering/rir/PROMOTION.md`),
//! which compares against the native kernel through ggml; this compares RIR to
//! RIR and decides nothing on its own - the same boundary `device_timing`
//! carries at the top of its file.

pub mod bind;
pub mod candidates;
pub mod measure;
pub mod report;
pub mod run;
pub mod verdict;

pub use bind::{Fixture, Geometry};
pub use candidates::{Candidate, Origin, Subject};
pub use measure::{Built, Footprint, Sample};
pub use run::{Arbitrated, Options, SweepReport, sweep};
pub use verdict::{Measured, Refusal, Verdict, arbitrate};

/// What can go wrong in a sweep, as one type per: every
/// arm below is a `From` of the error the crate that owns it already returns,
/// so no boundary in this crate re-spells a message with `to_string()`.
///
/// A sweep is a tool and not a chain, so it has its own facade rather than
/// reaching for `rir_gen::GenError`: the two share `LowerError` and `EmitError`
/// and nothing else, and a tool that pretended to be the AOT pipeline would
/// publish `RegistryError` and `ScheduleError` variants it can never return.
#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error("lowering: {0}")]
    Lower(#[from] rir_lower::LowerError),
    #[error("emission: {0}")]
    Emit(#[from] rir_emit::EmitError),
    #[error("device: {0}")]
    Runtime(#[from] rir_runtime::RuntimeError),
    #[error("schedule: {0}")]
    Schedule(#[from] rir_lower::ScheduleError),
    #[error("building the kernel graph: {0}")]
    Kernel(#[from] rir_core::ValidateError),
    /// The harness cannot bind this kernel: a dtype it has no fixture for, or
    /// more than one written argument. Stated rather than panicked, because a
    /// kernel outside the harness's reach is a legitimate answer to "sweep
    /// this" and not a failure of the sweep.
    #[error("{kernel}: {why}")]
    Unbindable { kernel: String, why: String },
    /// No kernel of the registry carries this name.
    #[error("no registered kernel named '{0}'")]
    UnknownKernel(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

impl SweepError {
    /// Whether this error means "not measurable here" rather than "wrong":
    /// no device, no compiler, no toolkit. A sweep that skips must say so and
    /// return zero rows, never an empty proposal that reads like a refusal.
    pub fn is_unavailable(&self) -> bool {
        match self {
            SweepError::Runtime(e) => rir_runtime::is_unavailable(e),
            _ => false,
        }
    }
}
