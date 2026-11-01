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
    BLOCKED_SCAN, Backend, Claim, Dim, FALLBACK_PRIORITY, FAMILY_REFUSED, FLAT_LINEAR, FLAT_ROWS,
    Family, GPU_REFUSED, GPU_TARGETS, Geometry, GpuBackend, HIER_REDUCE, MATRIX_COLS, MATRIX_ROWS,
    MATRIX_SUBGROUPS, MATRIX_TARGETS, MATRIX_TILE, MAX_STAGED_SHARED_BYTES, PARTITION_TARGETS,
    ParallelMapping, Pick, ReductionStrategy, SHARED_REDUCE, STRIDED_SCAN, SUBGROUP_LANES,
    ScanStrategy, Schedule, ScheduleError, ShapeRule, Space, TILED_SCAN, VariantName,
    check_schedule, check_schedule_table, family_refusal, matrix_targets_for, partitions_subgroups,
    schedules_for, targets_for,
};
