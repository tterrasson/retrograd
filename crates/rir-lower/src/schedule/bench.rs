//! Schedules kept for **comparison only**, out of the production table.
//!
//! A benchmark predecessor is not dead code and it is not a production
//! constructor either: it is the measurement that justified the variant which
//! replaced it. Keeping it is deliberate - a claim that
//! `gpu_tiled_scan` is faster is only checkable against what it replaced, and
//! `device_timing` compares the two on the same shapes in the same session.
//!
//! What this module buys over a paragraph saying the same thing: the **type**
//! says it. `rir_lower::schedule::bench::vulkan_shared_scan()` cannot be read as
//! a production form by accident, `schedules_for` has no reason to reach into
//! this module, and a reviewer sees the path in the diff.
//!
//! The schedule rules apply here unchanged: a schedule in this module is built
//! by the same private constructors, validated by the same `check_shape`, and
//! lowered by the same passes. It is out of the *table*, not out of the
//! contract.

use crate::schedule::{GpuBackend, ReductionStrategy, ScanStrategy, Schedule};

/// Variant name of the workgroup-wide shared-memory scan. Never emitted by the
/// production pipeline: no arm of the schedule table returns this variant, so no
/// artefact carries the name.
pub const SHARED_SCAN: &str = "shared";

/// Workgroup scan: 256 contiguous chunks recombined by the fixed shared-memory
/// tree - the threadgroup lane index on Metal, the workgroup one on Vulkan.
/// Unlike the 32-lane blocked variant it spans the whole workgroup, so it keeps
/// reducing serial depth when one very long row cannot occupy the device on its
/// own.
///
/// The predecessor `gpu_tiled_scan` took the domain of: same 256 lanes, a chunk
/// per lane instead of a coalesced access plan. Identical serial depth, which is
/// exactly what makes the pair worth timing against each other.
pub fn gpu_shared_scan(gpu: GpuBackend) -> Schedule {
    Schedule {
        block: [256, 1, 1],
        scan: ScanStrategy::BlockedLanes,
        reduction: ReductionStrategy::SharedTree,
        ..Schedule::gpu_subgroup(gpu)
    }
    .as_variant(SHARED_SCAN)
}

/// `gpu_shared_scan` on Vulkan.
pub fn vulkan_shared_scan() -> Schedule {
    gpu_shared_scan(GpuBackend::Vulkan)
}

/// `gpu_shared_scan` on Metal.
pub fn metal_shared_scan() -> Schedule {
    gpu_shared_scan(GpuBackend::Metal)
}
