//! Lowering errors, and the one predicate the registry asks before it
//! decides which quantized variants to generate.

use rir_core::{AxisId, ValueId};

use crate::schedule::{ParallelMapping, ReductionStrategy, ScheduleError};

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LowerError {
    #[error("{0}")]
    Schedule(#[from] ScheduleError),
    /// At least two axes, including at least one parallel axis.
    #[error("expected at least two axes (including one parallel), {got} declared")]
    UnsupportedAxisCount { got: usize },
    /// V1 supports a single inner reduction or scan axis.
    #[error("v1: only one reduction/scan axis")]
    MultipleInnerAxes,
    /// V1 does not mix reductions and scans in one kernel.
    #[error("v1: reductions and scans are mutually exclusive in one kernel")]
    MixedScanReduce,
    /// V1 requires all scans to have the same direction.
    #[error("v1: all scans in a kernel must have the same direction")]
    MixedScanDirections,
    /// A reduction input depends on another reduction in an unsupported way.
    #[error("nested reduction at %{} - unsupported in v1",.value.0)]
    NestedReduce { value: ValueId },
    /// An `Index` refers to an axis outside the current loop scope.
    #[error("%{} indexes axis #{} outside loop scope",.value.0,.axis.0)]
    AxisOutOfScope { value: ValueId, axis: AxisId },
    /// A quantized tensor `Read` lacks `Dequant`; the builder prevents this.
    #[error("%{}: quantized read without Dequant",.value.0)]
    UnfusedQuantRead { value: ValueId },
    /// The format is `NativeIntrinsic`: RIR owns its
    /// contract but not its decoding formula, so there is nothing to expand.
    /// The backend adapter must bind its own primitive instead.
    #[error(
        "{}: native_intrinsic format, the backend primitive must be linked",
        .format.desc().name
    )]
    NativeIntrinsicQuant { format: rir_core::QuantType },
    /// The format is portable, but its block shape has no Loop IR expansion
    /// yet. Distinct from the case above on purpose: this one is a gap in
    /// lowering, that one is a deliberate policy.
    #[error("{}: block shape without Loop IR expansion",.format.desc().name)]
    UnsupportedQuantShape { format: rir_core::QuantType },
    /// A scan met a collective reduction strategy without asking for one of
    /// the lane scans that go with it.
    #[error(
        "scan under a collective reduction strategy: a lane scan (blocked, tiled or strided) \
         required, or a Serial reduction"
    )]
    ScanRequiresSerial,
    /// The blocked scan needs lanes to recombine chunk totals; a schedule
    /// without a collective cannot run it.
    #[error("blocked scan: collective reduction strategy required (lanes)")]
    BlockedScanRequiresLanes,
    /// A lane strategy requires an inner reduction axis. Elementwise kernels
    /// have nothing to contribute to a collective and need `vulkan_grid`.
    #[error("lane strategy: expected an inner (reduction) axis, elementwise kernel")]
    LanesRequireInnerAxis,
    /// A lane strategy requires `ParallelMapping::Workgroup`, where the whole
    /// workgroup cooperates on one index, and vice versa.
    #[error(
        "strategy {} incompatible with parallel mapping {}",
        .strategy.name(), .mapping.name()
    )]
    MappingMismatch {
        strategy: ReductionStrategy,
        mapping: ParallelMapping,
    },
    /// The fixed shared-memory tree uses the schedule's lane count as its
    /// topology. A power of two keeps every up/down-sweep level total and
    /// deterministic instead of silently dropping a tail.
    #[error("SharedTree: lane count is not a power of two ({lanes})")]
    SharedTreeRequiresPowerOfTwo { lanes: u32 },
    /// `ReductionStrategy::HierarchicalTree` on a geometry its two stages cannot
    /// cover. Same refusal as the three above and for the
    /// same reason: `hierarchical_tree` is published in the manifest, so a
    /// lowering that quietly expanded a flat tree instead would be a kernel
    /// described by a strategy it does not use.
    #[error("hierarchical_tree: {why}")]
    HierarchicalTreeGeometry { why: &'static str },
    /// `vector_width > 1` on a lowering that cannot carry it. Vectorizing means
    /// one invocation owning `w` consecutive elements of the contiguous axis,
    /// which only exists under the invocation mapping of a kernel with no inner
    /// axis; anywhere else the width would have to be silently ignored, and a
    /// schedule field nothing reads is exactly what this refuses.
    #[error("vector_width > 1: {why}")]
    VectorWidthUnsupported { why: &'static str },
    /// `Schedule::flatten` on a lowering that cannot honour it.
    /// Same refusal as the width above and for the same
    /// reason: the flattening is published - in the manifest, in the registry's
    /// `flat[]` and in the constant buffer's divisors - so a lowering that
    /// quietly kept its three-dimensional nest would be dispatched with a grid
    /// computed for a linear one.
    #[error("flatten: {why}")]
    FlattenUnsupported { why: &'static str },
    /// `Schedule::linear_addr` on a kernel whose accesses are not the identity
    /// on the flattened space.
    ///
    /// The *same shape* half of the claim is proved here, on the semantic
    /// graph, and never on a tensor: every access must index every parallel
    /// axis, in dimension order, with nothing folded and nothing read at a
    /// position. A kernel that fails this has no linear address to compute, and
    /// silently keeping the decomposition would publish `linear_addr` for a
    /// shader that does not use it - the failure mode of a silent `vector_width`,
    /// one claim further out.
    #[error("linear_addr: {why}")]
    LinearAddrUnsupported { why: &'static str },
    /// `ReductionStrategy::TiledStage` on a kernel or a schedule whose shape
    /// staging cannot express. Like the width above, a
    /// strategy the lowering cannot honour is an error rather than a silent
    /// downgrade to the untiled form: the manifest would still publish
    /// `tiled_stage`.
    #[error("tiled_stage: {why}")]
    TilingUnsupported { why: &'static str },
    /// A read inside a tiled contraction whose indexing staging cannot describe:
    /// a tile is `rows × depth` of **one** parallel axis and the reduction axis,
    /// so any other index shape has no tile to be staged into.
    #[error("tiled_stage: read %{} - {why}",.value.0)]
    TilingUnsupportedRead { value: ValueId, why: &'static str },
    /// `ScanStrategy::TiledLanes` on a kernel whose shape the tiled scan cannot
    /// express. Same rule as the two above: a strategy
    /// the lowering cannot honour is an error, never a silent fall back to the
    /// blocked scan - the manifest would go on publishing `tiled_lanes`.
    #[error("tiled_lanes: {why}")]
    TiledScanUnsupported { why: &'static str },
    /// `ScanStrategy::StridedLanes` outside the one geometry it has: a single
    /// subgroup, on a kernel that scans.
    #[error("strided_lanes: {why}")]
    StridedScanUnsupported { why: &'static str },
    /// A dense argument whose element type has no memory access type.
    /// F32 and F16 do; `BF16`, `I32` and the rest are
    /// declarable in `rir_core::DType` and are not lowered, so a kernel that
    /// asked for one is an error here rather than an F32 access to bytes that
    /// are not floats.
    #[error(
        "argument {}: element type {} without memory access",
        .arg.0, .dtype.name()
    )]
    UnsupportedElementType {
        arg: rir_core::ArgId,
        dtype: rir_core::DType,
    },
}

/// Whether lowering can expand a format's decoding into Loop IR today.
///
/// Two things this is **not**. It is not the format's `Lowering` policy: a
/// `NativeIntrinsic` format is excluded by design, a portable one only when its
/// block shape has no expansion yet. And it is not a per-kernel decision - it
/// depends on the block layout alone, so the kernel registry can ask it before
/// deciding which variants to generate, instead of discovering the failure at
/// generation time.
///
/// **Derived, and no format name appears below it**: the
/// answer to "can RIR read this format" lives in the canonical table, next to
/// the format. The table carries a `layout` column and this reads it: a row with a description
/// is lowerable, a row without one is not, and adding a format never touches
/// this file.
pub fn can_lower_dequant(format: rir_core::QuantType) -> bool {
    let d = format.desc();
    d.is_portable() && d.layout.is_some()
}
