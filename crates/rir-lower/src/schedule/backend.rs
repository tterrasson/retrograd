//! Which backends a family is scheduled on, and which it refuses in writing.
//!
//! `Backend` and `GpuBackend` themselves live in `rir-core`: the manifest a
//! generated kernel publishes names a backend, and `rir-runtime` reads that
//! manifest without depending on a schedule. What stays
//! here is the part that mentions a `Family` - the tables.

pub use rir_core::backend::{Backend, GpuBackend};

use crate::schedule::Family;

/// The GPU backends every arm of `schedules_for` is written for, in the order
/// their artifacts are emitted.
///
/// It is the list a kernel is scheduled on, stated once as **data**. Before it
/// existed the same fact was the conjunction of one `push(Schedule::vulkan_*)`
/// and one `push(Schedule::metal_*)` per lowering: fourteen pairs, and nothing
/// checking that the two halves agreed. A kernel that had lost one half kept
/// CPU-only coverage on that backend, silently: the regression mode where a
/// test does not break, it becomes tautological.
///
/// The order is the emission order, so it is ABI-adjacent: `generated/rir/`
/// carries the Metal aggregate and the Vulkan artifact list in this sequence.
///
/// CUDA joined it later, and it joined at the **end** on
/// purpose: the order is the emission order, so appending a backend leaves the
/// Metal aggregate and the Vulkan artifact list byte for byte as they were.
pub const GPU_TARGETS: &[GpuBackend] = &[GpuBackend::Vulkan, GpuBackend::Metal, GpuBackend::Cuda];

/// The GPU backends the table deliberately schedules **nothing** on, each with
/// the reason.
///
/// A kernel has the right to say "not on this backend"; what it does not have
/// is the right to say it by omitting a line. `GPU_TARGETS` and this list
/// partition `GpuBackend::ALL`, which `gpu_coverage_is_a_partition` checks - so
/// adding a `GpuBackend` variant fails the build until it is either scheduled
/// or refused in writing.
///
/// Empty since CUDA got its own emitter: the one entry it carried was CUDA's, for
/// want of an emitter, and there is one. CUDA refusals are listed
/// **family by family**, below, because the reasons differ between kernels.
pub const GPU_REFUSED: &[(GpuBackend, &str)] = &[];

/// The backends **one family** refuses, each with the reason.
///
/// `GPU_REFUSED` above is the other half of the same right: it is the refusal
/// the whole table carries, and it is the only shape a refusal could take
/// until now. That left one case unsayable - a family that cannot serve a
/// backend the rest of the table serves - and it is exactly the case a new
/// backend produces: hooking CUDA up moves `GpuBackend::Cuda` into
/// `GPU_TARGETS` for every family at once, so a family whose lowering is not
/// ready would have to be served, refused for everyone, or left to fail
/// generation. None of the three is "not on this backend, said once, in
/// writing".
///
/// A refusal here is honoured twice, which is what keeps it from being a
/// comment: `targets_for` does not emit the schedule, and
/// `rir_gen::check_gpu_coverage` requires that the family carry none - and
/// requires a refusal for any backend a family is missing.
///
/// Three entries today, all CUDA's, and they are the shape a new backend was
/// predicted to take: it does not arrive whole. Two of
/// the five it opened with are gone - `row_reduce` and `rms_norm` were refused
/// for want of a collective printer, which RIR now has.
pub const FAMILY_REFUSED: &[(Family, GpuBackend, &str)] = &[
    (
        Family::Cumsum,
        GpuBackend::Cuda,
        "the only family whose refusal survived the collective printer, and the \
         reason no longer mentions the emitter: `CUMSUM` has **zero nodes** in \
         both censused graphs. No promoted pair needs it, so v1 does \
         not carry it - and scheduling it would put three lowerings of a \
         scan in production shape with nothing measuring them",
    ),
    // The two the emitter could print today, refused on their merits.
    (
        Family::OutProd,
        GpuBackend::Cuda,
        "not a target on CUDA, unlike Metal and Vulkan: \
         F32 goes through cuBLAS SGEMM and the quantized path through the tiled \
         kernel with its shared-memory decode (J4.5). Both win largely, and the \
         census puts the op second in the graph - aiming at it would be a \
         mistake already named and refused",
    ),
    (
        Family::MatMulNaive,
        GpuBackend::Cuda,
        "oracle-only kernel: no ggml op, hence no row in `rir_kernel_params.h`, \
         which every generated `.cu` includes to get its parameter struct. \
         There is nothing on CUDA for it to be parity \
         against either - `MUL_MAT` is out of the DSL's domain",
    ),
];

/// The GPU backends the **flattened** variant is written for.
///
/// A shorter list than `GPU_TARGETS`, and the asymmetry is the measurement and
/// not an omission - which is why it is stated here as data, beside the two
/// lists that already are, rather than as a backend named inside an arm.
///
/// The lever answers a deficit that exists on CUDA and does not exist on the
/// other two. the documented constraint is the whole argument: the native CUDA
/// elementwise kernels are *flat* loops over `nelements`, so a three-dimensional
/// RIR grid loses the threads a short row leaves idle in every block; the native
/// Metal and Vulkan kernels dispatch a threadgroup per row, which is the deficit
/// RIR exploits and where it is already ahead. Adding the variant there would
/// publish a claim - a divisor block in the constant buffer, a layout condition
/// at every dispatch site - for a gain nothing has measured.
///
/// The *mechanism* is backend-neutral all the same: it is lowered in the Loop
/// IR, printed by all three GPU emitters, and judged by the oracle. Extending
/// this list is one line and a lane run.
pub const FLAT_TARGETS: &[GpuBackend] = &[GpuBackend::Cuda];

/// `FLAT_TARGETS` less what this family refuses, on the model of `targets_for`.
pub fn flat_targets_for(family: Family) -> impl Iterator<Item = GpuBackend> {
    FLAT_TARGETS
        .iter()
        .copied()
        .filter(move |g| family_refusal(family, *g).is_none())
}

/// The backends whose vectorized lowering keeps the **three-dimensional grid**:
/// `targets_for(family)` less `FLAT_TARGETS`.
///
/// The complement exists because the flattening *replaces* the grid lowering
/// rather than sitting above it, and that is a measurement result rather than a
/// simplification: stacked on top with no shape rule, the flat variant claimed
/// every node of the lane's matrix and left its neighbours at zero dispatches,
/// which the lane reports as "declared variant never dispatched" and refuses,
/// correctly. Two lowerings differing only by a dispatch geometry do not
/// partition a domain; one of them is simply the one this backend uses.
pub fn grid_targets_for(family: Family) -> impl Iterator<Item = GpuBackend> {
    targets_for(family).filter(|g| !FLAT_TARGETS.contains(g))
}

/// Whether `family` refuses `gpu`, and why.
pub fn family_refusal(family: Family, gpu: GpuBackend) -> Option<&'static str> {
    FAMILY_REFUSED
        .iter()
        .find(|(f, g, _)| *f == family && *g == gpu)
        .map(|(_, _, why)| *why)
}

/// The backends the table schedules **this** family on: `GPU_TARGETS` less
/// what the family refuses in writing. Every arm of `schedules_for` iterates
/// it, so a refusal is one line in `FAMILY_REFUSED` and not fourteen arms to
/// audit.
pub fn targets_for(family: Family) -> impl Iterator<Item = GpuBackend> {
    GPU_TARGETS
        .iter()
        .copied()
        .filter(move |g| family_refusal(family, *g).is_none())
}
