//! The knobs a schedule turns: reduction strategy, parallel mapping, scan
//! strategy, and the shape rule that decides when a variant may serve.
//!
//! Each knob says what it publishes in the manifest through `published()`, in
//! one exhaustive `match` - the manifest never spells a strategy itself.

use rir_core::manifest;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionStrategy {
    /// Sequential accumulation along the reduction axis.
    Serial,
    /// Shared-memory tree with a fixed, run-to-run deterministic topology.
    SharedTree,
    /// Sequential accumulation, as `Serial`, over operands **staged** in shared
    /// memory a tile at a time (ADR-2 section 6). The workgroup loads a
    /// `tile_depth`-deep slice of each contracted operand cooperatively, then
    /// every invocation of the workgroup accumulates that slice from shared
    /// memory instead of re-reading the same rows from global memory once per
    /// neighbour.
    ///
    /// It is a **memory** strategy, not an arithmetic one: the reduction axis
    /// is still walked in ascending order, one term at a time, so the sum is
    /// the sequential one term for term - which is why it is admissible even
    /// under `ExactOrder`.
    TiledStage,
    /// Subgroup/warp reduction with a fixed, run-to-run deterministic topology.
    SubgroupTree,
    /// **Two stages**: every subgroup reduces its own
    /// lanes with the hardware primitive, one total per subgroup goes to shared
    /// memory, and the first subgroup reduces those totals and broadcasts.
    ///
    /// It is what `SharedTree` is not, on a workgroup wider than a subgroup: the
    /// tree spends `log2(lanes)` barriers and `lanes` words of shared storage,
    /// and its last levels are almost all idle lanes - 256 lanes are eight
    /// barriers whose last five combine 8, 4, 2 and 1 pairs. The native CUDA
    /// reduction reaches the same width in two levels through `warp_reduce_sum`
    /// plus a 32-entry shared stage, which is a deficit no block size
    /// can close.
    ///
    /// Two conditions, checked at lowering rather than assumed: the lane count
    /// is a whole number of subgroups, and there are no more subgroups than one
    /// subgroup can reduce - otherwise the second stage would need a third.
    ///
    /// Like `SubgroupTree` it needs the backend's subgroup primitive, and like
    /// it, it assumes what that primitive already assumes here: a subgroup is a
    /// contiguous run of workgroup-local indices. The manifest publishes the
    /// width as a device requirement (`min_subgroup`).
    HierarchicalTree,
}

impl ReductionStrategy {
    /// The published name of this strategy - the manifest's vocabulary, mapped
    /// here so a new strategy fails to compile until it says what it publishes.
    pub fn published(self) -> manifest::Reduction {
        match self {
            ReductionStrategy::Serial => manifest::Reduction::Serial,
            ReductionStrategy::SharedTree => manifest::Reduction::SharedTree,
            ReductionStrategy::TiledStage => manifest::Reduction::TiledStage,
            ReductionStrategy::SubgroupTree => manifest::Reduction::SubgroupTree,
            ReductionStrategy::HierarchicalTree => manifest::Reduction::HierarchicalTree,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ReductionStrategy::Serial => "serial",
            ReductionStrategy::SharedTree => "shared_tree",
            ReductionStrategy::TiledStage => "tiled_stage",
            ReductionStrategy::SubgroupTree => "subgroup_tree",
            ReductionStrategy::HierarchicalTree => "hierarchical_tree",
        }
    }
}

/// How parallel axes map to hardware. The schedule makes this decision,
/// lowering applies it, and the emitter prints the resulting builtin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelMapping {
    /// One entire workgroup per index (`gl_WorkGroupID`), whose lanes cooperate
    /// on that index. Required by lane-based strategies.
    Workgroup,
    /// One invocation per index (`gl_GlobalInvocationID`), with the workgroup
    /// covering `block[d]` consecutive indices. Used for kernels without
    /// collectives: elementwise operations, per-thread contractions, and
    /// per-row scans.
    Invocation,
}

impl ParallelMapping {
    /// The published name of this mapping. See `ReductionStrategy::published`.
    pub fn published(self) -> manifest::ParallelMapping {
        match self {
            ParallelMapping::Workgroup => manifest::ParallelMapping::Workgroup,
            ParallelMapping::Invocation => manifest::ParallelMapping::Invocation,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ParallelMapping::Workgroup => "workgroup",
            ParallelMapping::Invocation => "invocation",
        }
    }

    /// Number of axis indices covered by one workgroup in dimension `d`; the
    /// dispatcher divides by this value to obtain the workgroup count.
    ///
    /// No `max(1)` on `block[d]`: `Schedule::check_shape` guarantees it, and a
    /// clamp here would be the second half of the bug of the block a shader
    /// declares and the block a dispatcher divides by being normalized at
    /// different sites.
    pub fn per_workgroup(self, block: [u32; 3], d: usize) -> u32 {
        match self {
            ParallelMapping::Workgroup => 1,
            ParallelMapping::Invocation => block[d],
        }
    }
}

/// How a scan axis is traversed. A scan is not a reduction: its result is a
/// value **per element**, so a strategy has to say how partial results
/// recombine, not just how they accumulate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanStrategy {
    /// One invocation traverses the whole axis. The accumulation order is
    /// exactly the sequential one (`ExactOrder`), and the cost is linear in the
    /// axis extent - one invocation does all the work.
    Serial,
    /// The axis is cut into `block[0]` contiguous chunks, one per lane: each
    /// lane totals its chunk, an exclusive lane prefix turns those totals into
    /// per-lane offsets, and each lane re-traverses its chunk from its offset.
    ///
    /// The serial depth drops by the lane count - the point of the strategy.
    /// The price is stated, not hidden: the additions are regrouped, so the
    /// result is `Deterministic` (fixed by the declared lane count, reproduced
    /// by the oracle) and no longer `ExactOrder`.
    BlockedLanes,
    /// The axis is walked in tiles of `block[0] · items` elements, each tile
    /// staged **interleaved** into shared memory, scanned there, and flushed
    /// interleaved (`ScanTiles`, ADR-2).
    ///
    /// Same serial depth as `BlockedLanes` and the same regrouping - hence the
    /// same `Deterministic` semantics - but a different *access plan*: what
    /// `BlockedLanes` leaves on the table is that two neighbouring lanes read
    /// addresses a chunk apart, which was measured
    /// and cannot be fixed by adding lanes.
    ///
    /// `items` is the lane's share of one tile. It is carried by the strategy
    /// rather than by a `Schedule` field because it means nothing for the other
    /// two - the mistake `vector_width` made once already.
    TiledLanes { items: u32 },
}

impl ScanStrategy {
    /// The published name of this strategy, without the tile size `TiledLanes`
    /// carries: that number is a lowering decision, not part of the contract
    /// (`rir_core::manifest::Scan`).
    pub fn published(self) -> manifest::Scan {
        match self {
            ScanStrategy::Serial => manifest::Scan::Serial,
            ScanStrategy::BlockedLanes => manifest::Scan::BlockedLanes,
            ScanStrategy::TiledLanes { .. } => manifest::Scan::TiledLanes,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ScanStrategy::Serial => "serial",
            ScanStrategy::BlockedLanes => "blocked_lanes",
            ScanStrategy::TiledLanes { .. } => "tiled_lanes",
        }
    }
}

/// A shape predicate a variant publishes: the **product** of the extents of
/// the named axes must fall within `[min, max]` inclusive.
///
/// A product rather than a single extent because what decides between two
/// lowerings of the same kernel is rarely one dimension. The blocked scan wins
/// while the *total* number of rows leaves the GPU idle, and that total is
/// `row × plane × batch` - three ggml dimensions the kernel keeps distinct
/// precisely so it can address a view.
///
/// Axes are named, not indexed: the registry emitter resolves the name against
/// the kernel's axis table, so a rule written here fails generation rather than
/// silently constraining the wrong dimension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeRule {
    pub axes: &'static [&'static str],
    pub min: u32,
    pub max: u32,
}

impl ShapeRule {
    /// `product(axes) ≤ max`. A lower bound is written as a literal: the rule
    /// is an interval, and the only variant so far claims a ceiling.
    pub fn at_most(axes: &'static [&'static str], max: u32) -> Self {
        Self { axes, min: 1, max }
    }
}

/// Lanes of one hardware subgroup, as every lane collective in this crate
/// already assumes it.
///
/// It is a warp on CUDA, a SIMD-group on Apple, and the minimum a Vulkan device
/// must publish for a generated kernel to be admitted - the manifest carries it
/// as `min_subgroup`, so a device narrower than this refuses the kernel instead
/// of running a tree that combines the wrong lanes.
///
/// Named here because `HierarchicalTree` made it arithmetic: until then it was a
/// block of `[32, 1, 1]` written by one constructor, which is a geometry, and
/// now it is a divisor of the lane count and the length of a shared array.
pub const SUBGROUP_LANES: u32 = 32;

/// Priority of the fallback variant of a (kernel, backend) pair. Named
/// variants must sit strictly above it, so a shape both accept resolves to the
/// specialization and never to table order.
pub const FALLBACK_PRIORITY: u8 = 80;
