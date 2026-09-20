//! RIR schedules and lowering.
//!
//! Loop IR keeps emitters mechanical. After applying a schedule, loop nests,
//! accumulators, and memory accesses are explicit but still backend-agnostic.
//! If an emitter needs more than a `match` over `Stmt`, a concept is missing
//! here rather than in the emitter.
//!
//! `interp` executes Loop IR directly and provides the stage-zero parity
//! oracle without compiling generated code.

pub mod cse;
pub mod hoist;
pub mod interp;
pub mod loop_ir;
pub mod lower;

#[cfg(test)]
pub mod probe;
pub mod schedule;

pub use loop_ir::*;
pub use lower::{LowerError, can_lower_dequant, lower};
pub use schedule::{
    BLOCKED_SCAN, Backend, FALLBACK_PRIORITY, FAMILY_REFUSED, FLAT_LINEAR, FLAT_ROWS, Family,
    GPU_REFUSED, GPU_TARGETS, GpuBackend, HIER_REDUCE, ParallelMapping, ReductionStrategy,
    SHARED_REDUCE, SUBGROUP_LANES, ScanStrategy, Schedule, ScheduleError, ShapeRule, TILED_SCAN,
    check_schedule, check_schedule_table, family_refusal, schedules_for, targets_for,
};
