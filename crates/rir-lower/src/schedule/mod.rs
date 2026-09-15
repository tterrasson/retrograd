//! Declarative schedules (ADR-2 section 3), handwritten per kernel and backend.
//! The domain is finite, so there is no autotuning.

pub mod backend;
pub mod bench;
pub mod error;
pub mod strategy;
pub mod table;

#[cfg(test)]
mod tests;

pub use backend::{
    Backend, FAMILY_REFUSED, FLAT_TARGETS, GPU_REFUSED, GPU_TARGETS, GpuBackend, family_refusal,
    flat_targets_for, grid_targets_for, targets_for,
};
pub use error::{ScheduleError, check_schedule, check_schedule_table};
pub use strategy::{
    FALLBACK_PRIORITY, ParallelMapping, ReductionStrategy, SUBGROUP_LANES, ScanStrategy, ShapeRule,
};
pub use table::{Family, schedules_for};

/// A lowering decision for one (kernel, backend) pair.
///
/// **Opaque by design.** The fields are private and the
/// only ways in are the constructors below plus `as_variant`/`unnamed`/
/// `claiming`; the only ways out are the read accessors. Two invariants are
/// therefore not representable rather than merely validated: `grid_dims` is
/// set by the constructor that knows the
/// backend (1 on CPU, 3 on GPU), so nothing can index `block[3]`, and
/// `tile_depth` is non-zero exactly under `ReductionStrategy::TiledStage`
/// because one constructor sets both. What a constructor cannot decide - the
/// numeric arguments it is handed - is checked by `check_shape`, which `lower`
/// calls before reading anything: a zero block is a `ScheduleError`, never a
/// panic and never a silent `max(1)`.
///
/// A schedule assembled field by field does not compile, and neither does a
/// GPU lowering asked for on the CPU:
///
/// ```compile_fail
/// // `Schedule`'s fields are crate-private by design:
/// // `grid_dims: 7` was a value that indexed past `block[2]`.
/// let s = rir_lower::Schedule {
///     grid_dims: 7,
///     ..rir_lower::Schedule::cpu_serial()
/// };
/// ```
///
/// ```compile_fail
/// // `gpu_grid` takes a `GpuBackend`, so there is no CPU workgroup to build.
/// let s = rir_lower::Schedule::gpu_grid(rir_lower::Backend::Cpu, [256, 1, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct Schedule {
    pub(crate) backend: Backend,
    pub(crate) block: [u32; 3],
    /// Number of consecutive elements one invocation handles on the contiguous
    /// axis. `1` is the scalar lowering. A width above one is a **claim**, in
    /// the same sense as a `ShapeRule`: the variant only accepts tensors whose
    /// contiguous stride is one element, so it needs a fallback below it and
    /// the dispatcher has to evaluate that condition before selecting it
    /// (`rir_variant_desc.vector_width`).
    pub(crate) vector_width: u32,
    /// Whether every parallel axis is flattened into **one** grid dimension,
    /// decomposed back by magic-number division.
    ///
    /// Like `vector_width` it is a lowering decision the manifest and the
    /// registry publish, and like it, it is refused rather than ignored when
    /// the rest of the schedule cannot honour it: flattening replaces the grid
    /// nest, so it only means something under `Invocation` mapping and a
    /// `Serial` strategy - a workgroup collective needs its axis to *be* a grid
    /// dimension.
    pub(crate) flatten: bool,
    /// Whether the flattened index addresses memory **directly**, as
    /// `linear · elem_bytes`, instead of being decomposed into axis indices
    /// whose stride products are then summed.
    ///
    /// It is the layout claim: *all bindings contiguous and of the same
    /// shape*. Under it the
    /// decomposition disappears entirely - no magic-number division, no `nb[]`
    /// product per element - which is what `k_bin_bcast` obtains by collapsing
    /// its axes host side, and what the flattened grid alone did not close.
    ///
    /// Like `vector_width` and `flatten` it is published rather than assumed:
    /// the registry carries it, the dispatcher evaluates it against the node's
    /// strides, and the pair keeps a lowering below it for every layout it
    /// declines. The *same shape* half is not a runtime test at all - it is a
    /// property of the kernel, checked at lowering, and the contract's axis
    /// agreement is what makes it hold on the tensors as well.
    pub(crate) linear_addr: bool,
    pub(crate) reduction: ReductionStrategy,
    /// Indices of the reduction axis one staged tile covers, under
    /// `ReductionStrategy::TiledStage`. Zero everywhere else, and a
    /// `LowerError` if the two disagree - a depth nothing reads would be a
    /// silent field.
    pub(crate) tile_depth: u32,
    /// Scan-axis strategy. Independent of `reduction` because v1 forbids a
    /// kernel from carrying both a scan and a reduction.
    pub(crate) scan: ScanStrategy,
    /// Parallel-axis mapping.
    pub(crate) par_map: ParallelMapping,
    /// Number of available grid dimensions. The first `grid_dims` parallel
    /// axes map to the grid; remaining axes stay sequential loops. GPUs use 3;
    /// CPUs use 1, where the "grid" is the parallelizable outer loop.
    pub(crate) grid_dims: usize,
    /// Name of this lowering among the several a (kernel, backend) pair may
    /// carry. `None` is the pair's **fallback**: the one variant that accepts
    /// every shape the contract accepts, and the reason selection is total.
    /// `Some(name)` is a specialization, and it owes the table a shape rule.
    ///
    /// The name is part of the emitted identity - file name, entrypoint,
    /// `variant_id`, SPIR-V symbol - so two variants of one kernel can coexist
    /// as artifacts instead of overwriting each other.
    pub(crate) variant: Option<&'static str>,
    /// Selection priority among the variants eligible for a given shape.
    /// Set by the schedule table, not by the constructor: it is a policy about
    /// which lowering to prefer, not a property of the lowering.
    pub(crate) priority: u8,
    /// Shapes this variant claims. Empty = every shape (the fallback).
    pub(crate) eligible_when: Vec<ShapeRule>,
}

impl Schedule {
    /// Backend this lowering targets.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Workgroup shape, in axis declaration order. Every component is at least
    /// one on a schedule `check_shape` accepted.
    pub fn block(&self) -> [u32; 3] {
        self.block
    }

    /// Elements one invocation covers on the contiguous axis.
    pub fn vector_width(&self) -> u32 {
        self.vector_width
    }

    /// Whether the parallel space is flattened into one grid dimension.
    pub fn flatten(&self) -> bool {
        self.flatten
    }

    /// Whether the flattened index is also the **address**, in units of the
    /// element.
    pub fn linear_addr(&self) -> bool {
        self.linear_addr
    }

    pub fn reduction(&self) -> ReductionStrategy {
        self.reduction
    }

    /// Reduction-axis indices one staged tile covers; zero outside
    /// `ReductionStrategy::TiledStage`.
    pub fn tile_depth(&self) -> u32 {
        self.tile_depth
    }

    pub fn scan(&self) -> ScanStrategy {
        self.scan
    }

    pub fn par_map(&self) -> ParallelMapping {
        self.par_map
    }

    /// Grid dimensions available to this backend: 3 on a GPU, 1 on the CPU.
    /// Never above 3, because no constructor can produce that.
    pub fn grid_dims(&self) -> usize {
        self.grid_dims
    }

    /// Variant name, or `None` for the pair's fallback.
    pub fn variant(&self) -> Option<&'static str> {
        self.variant
    }

    pub fn priority(&self) -> u8 {
        self.priority
    }

    pub fn eligible_when(&self) -> &[ShapeRule] {
        &self.eligible_when
    }

    /// The invariants a constructor's **numeric arguments** can still break,
    /// checked before anything reads them.
    ///
    /// `grid_dims ≤ 3` and "`tile_depth` non-zero iff `TiledStage`" are absent
    /// here on purpose: they are decided by the constructors and therefore not
    /// representable. What a caller supplies is a block, a tile depth, a vector
    /// width and a lane count, and those are what can be zero.
    ///
    /// # Errors
    ///
    /// `ScheduleError::Malformed` naming the field at fault.
    pub fn check_shape(&self) -> Result<(), ScheduleError> {
        // Per dimension rather than "the block is zero", because which one is
        // zero says which grid dimension the caller got wrong.
        const ZERO_BLOCK: [&str; 3] = [
            "block[0] is zero: a workgroup has at least one invocation",
            "block[1] is zero: a workgroup has at least one invocation",
            "block[2] is zero: a workgroup has at least one invocation",
        ];
        for (d, &b) in self.block.iter().enumerate() {
            if b == 0 {
                return Err(ScheduleError::Malformed {
                    backend: self.backend,
                    why: ZERO_BLOCK[d],
                });
            }
        }
        if self.vector_width == 0 {
            return Err(ScheduleError::Malformed {
                backend: self.backend,
                why: "vector_width is zero: an invocation covers at least one element",
            });
        }
        if self.reduction == ReductionStrategy::TiledStage && self.tile_depth == 0 {
            return Err(ScheduleError::Malformed {
                backend: self.backend,
                why: "tile_depth is zero under TiledStage: a staged tile has at least one index",
            });
        }
        if let ScanStrategy::TiledLanes { items } = self.scan {
            if items == 0 {
                return Err(ScheduleError::Malformed {
                    backend: self.backend,
                    why: "TiledLanes with zero items: a lane's share of a tile is at least one \
                          element",
                });
            }
            // The tile is `lanes · items` elements wide and its shared array
            // one word per lane longer (strategies/scan.rs). Both are `u32`
            // there, and both are what the emitters print: an unrepresentable
            // product is a geometry this schedule cannot describe, and it is
            // rejected here rather than discovered as an overflow panic - or,
            // without the overflow checks, as a `ForTiled` of step zero that
            // Vulkan and Metal print literally as an endless loop.
            let lanes = self.block[0];
            if lanes
                .checked_mul(items)
                .and_then(|w| w.checked_add(lanes))
                .is_none()
            {
                return Err(ScheduleError::Malformed {
                    backend: self.backend,
                    why: "TiledLanes geometry overflows a 32-bit tile width: block[0] · items \
                          + block[0] must be representable",
                });
            }
        }
        Ok(())
    }

    /// Names this lowering, making it a specialization rather than the pair's
    /// fallback. Called by the constructors below, which know what they are.
    /// Renames the variant this lowering publishes.
    ///
    /// Two lowerings of one kernel may differ by something other than the
    /// constructor that built them - here two
    /// workgroup widths of the *same* vectorized mapping - and the artifact
    /// name has to say which is which. The default name a constructor gives
    /// describes its lowering ("vec4"); this says what distinguishes it from
    /// its twin.
    pub fn as_variant(self, variant: &'static str) -> Self {
        self.named(variant)
    }

    fn named(mut self, variant: &'static str) -> Self {
        self.variant = Some(variant);
        self
    }

    /// The same lowering with another workgroup shape.
    ///
    /// Not used by the production table, and deliberately so: every schedule it
    /// ships names its block in the constructor that knows why that block. This
    /// exists for the short loop, where a block is the *question* - measuring a
    /// 1 024-lane reduction against a 256-lane one is one line here rather than
    /// a constructor the table would then have to refuse.
    pub fn with_block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    /// Makes this lowering the pair's fallback: it keeps the kernel's own
    /// artifact name and is the row selection falls back to. Only legitimate
    /// when the pair has nothing else - a fallback beside a specialization it
    /// does not outrank would be unreachable.
    pub fn unnamed(mut self) -> Self {
        self.variant = None;
        self.priority = FALLBACK_PRIORITY;
        self
    }

    /// Attaches the selection policy the schedule table decides: which shapes
    /// this variant claims, and how it ranks against the ones that also claim
    /// them. Only a named variant may carry rules - the fallback exists to have
    /// none (`check_schedule_table`).
    ///
    /// The rule list may be empty when the variant claims a **layout** instead
    /// of a shape (`vector_width > 1`); what it may never be is empty on a
    /// variant that claims nothing at all, which would shadow the fallback.
    pub fn claiming(mut self, priority: u8, rules: Vec<ShapeRule>) -> Self {
        self.priority = priority;
        self.eligible_when = rules;
        self
    }

    /// Whether this lowering claims a subset of the pair's domain rather than
    /// accepting everything the contract accepts. Two kinds of claim, and the
    /// table treats them alike: a shape rule (an extent interval, evaluated
    /// from `ne[]`) and a vector width (a layout condition, evaluated from
    /// `nb[0]`). A schedule that claims neither is the pair's fallback.
    pub fn claims(&self) -> bool {
        !self.eligible_when.is_empty() || self.vector_width > 1 || self.linear_addr
    }

    /// Artifact identity of this schedule for `kernel`: the kernel name for the
    /// fallback, `kernel_variant` otherwise. One string behind every per-variant
    /// name the pipeline emits, so they cannot drift apart.
    pub fn artifact_name(&self, kernel: &str) -> String {
        match self.variant {
            Some(v) => format!("{kernel}_{v}"),
            None => kernel.to_string(),
        }
    }

    /// Reference CPU schedule: sequential loops in exact order. The one
    /// constructor with no GPU counterpart, and the reason `grid_dims` is a
    /// constructor's decision: 1 here, 3 in every `gpu_*` below.
    pub fn cpu_serial() -> Self {
        Self {
            backend: Backend::Cpu,
            block: [1, 1, 1],
            vector_width: 1,
            flatten: false,
            linear_addr: false,
            tile_depth: 0,
            scan: ScanStrategy::Serial,
            reduction: ReductionStrategy::Serial,
            par_map: ParallelMapping::Workgroup,
            grid_dims: 1,
            variant: None,
            priority: FALLBACK_PRIORITY,
            eligible_when: Vec::new(),
        }
    }

    // ---------------------------------------------------------------------
    // The `gpu_*` family. One constructor per
    // lowering, the backend a parameter rather than a suffix.
    //
    // The suffix was never carrying a decision: `vulkan_grid_vec4` and
    // `metal_grid_vec4` differed by one enum value and by nothing else, which
    // is why the table needed two `push` lines for one choice - and why a
    // kernel could keep coverage on one GPU backend and not on its neighbour
    // without a word being written. With the backend as a parameter the table
    // names its backends as data (`GPU_TARGETS`), and adding a third one costs
    // an enum variant instead of fourteen lines.
    // ---------------------------------------------------------------------

    /// One row per single-subgroup workgroup (32 lanes), reduced with the
    /// backend's subgroup primitive - `subgroupAdd` on Vulkan, `simd_sum` on
    /// Metal, both of which broadcast the complete reduction without shared
    /// memory or barriers.
    ///
    /// Requires a subgroup at least 32 lanes wide, which Apple GPUs give
    /// exactly and the manifest publishes as a Vulkan feature. Wider blocks
    /// spanning several subgroups need `gpu_shared_reduce` instead.
    pub fn gpu_subgroup(gpu: GpuBackend) -> Self {
        Self {
            backend: gpu.backend(),
            block: [32, 1, 1],
            vector_width: 1,
            flatten: false,
            linear_addr: false,
            tile_depth: 0,
            scan: ScanStrategy::Serial,
            reduction: ReductionStrategy::SubgroupTree,
            par_map: ParallelMapping::Workgroup,
            grid_dims: 3,
            variant: None,
            priority: FALLBACK_PRIORITY,
            eligible_when: Vec::new(),
        }
    }

    /// GPU schedule without collectives: one invocation per point in the
    /// parallel space, with the inner reduction or scan axis traversed
    /// sequentially **within** that invocation. It needs no subgroup feature
    /// and preserves the CPU accumulation order exactly.
    ///
    /// `block` maps parallel axes in declaration order: `par_axes[0]` to x,
    /// `[1]` to y, and `[2]` to z.
    pub fn gpu_grid(gpu: GpuBackend, block: [u32; 3]) -> Self {
        Self {
            backend: gpu.backend(),
            block,
            vector_width: 1,
            flatten: false,
            linear_addr: false,
            tile_depth: 0,
            scan: ScanStrategy::Serial,
            reduction: ReductionStrategy::Serial,
            par_map: ParallelMapping::Invocation,
            grid_dims: 3,
            variant: None,
            priority: FALLBACK_PRIORITY,
            eligible_when: Vec::new(),
        }
    }

    /// One workgroup per row as `gpu_subgroup` does, over a **flattened** row
    /// space: `row`, `plane` and `batch` folded into one
    /// grid dimension and decomposed back inside the kernel.
    ///
    /// It answers what was named beside the reduction topology
    /// and left unbuilt: the two `RMS_NORM_BACK` shapes left to the 32-lane
    /// fallback measure 1.15 and 1.17 "against a native that flattens `nrows`
    /// into a single grid dimension and addresses through one `int64` - no
    /// stride product per element - and reduces in two stages". This is the
    /// first half of that sentence. A three-dimensional grid also puts the row
    /// count on a dimension that stops at 65 535 and leaves the fourth axis a
    /// sequential loop no workgroup shares; a linear row space has neither.
    ///
    /// The reduction is untouched: one row is still one workgroup of 32 lanes
    /// with one subgroup collective, so what changes is where the row index
    /// comes from and nothing about the arithmetic.
    pub fn gpu_subgroup_flat_rows(gpu: GpuBackend) -> Self {
        Self {
            flatten: true,
            ..Self::gpu_subgroup(gpu)
        }
        .named(FLAT_ROWS)
    }

    /// Blocked scan: one row per single-subgroup workgroup, the scanned axis
    /// cut into 32 contiguous chunks recombined by an exclusive lane prefix.
    /// Same hardware requirements as `gpu_subgroup`; what changes is the
    /// strategy applied to the scan axis.
    pub fn gpu_blocked_scan(gpu: GpuBackend) -> Self {
        Self {
            scan: ScanStrategy::BlockedLanes,
            ..Self::gpu_subgroup(gpu)
        }
        .named(BLOCKED_SCAN)
    }

    /// Coalesced tiled scan: `lanes` lanes walking the axis in tiles of
    /// `lanes · items`, staged and flushed interleaved through shared memory
    /// (ADR-2 section 5). The lane count comes from the block, so the same
    /// constructor serves both the very long row and the many-row class; which
    /// widths are worth publishing is the schedule table's decision, and the
    /// bench's.
    pub fn gpu_tiled_scan(gpu: GpuBackend, lanes: u32, items: u32) -> Self {
        Self {
            block: [lanes, 1, 1],
            scan: ScanStrategy::TiledLanes { items },
            reduction: ReductionStrategy::SharedTree,
            ..Self::gpu_subgroup(gpu)
        }
        .named(TILED_SCAN)
    }

    /// Workgroup-wide **reduction**: the same `SharedTree` the scan variants
    /// use, applied to a kernel that only reduces. 256 lanes per row instead of
    /// 32, which is what a row-per-subgroup kernel lacks when the row count
    /// alone cannot occupy the device.
    pub fn gpu_shared_reduce(gpu: GpuBackend) -> Self {
        Self {
            block: [256, 1, 1],
            reduction: ReductionStrategy::SharedTree,
            ..Self::gpu_subgroup(gpu)
        }
        .named(SHARED_REDUCE)
    }

    /// Workgroup-wide reduction in **two stages**: the
    /// same 256 lanes per row `gpu_shared_reduce` puts there, reduced by
    /// subgroup and then across subgroup totals instead of by a full tree.
    ///
    /// What it changes against its neighbour is a count, twice: eight barriers
    /// become two, and 256 words of shared storage per accumulator become eight.
    /// The tree was measured losing to the native precisely
    /// there - "a 1 024-lane tree is ten barriers and 8 KiB of shared storage
    /// whose last five levels are almost all idle lanes; the native reaches the
    /// same width through `warp_reduce_sum` plus a 32-entry shared stage, which
    /// is two levels and not ten" - and named the missing capability rather than
    /// a block size. This is that capability.
    ///
    /// Same lane count as `gpu_shared_reduce` on purpose: what the short loop
    /// then compares is the topology alone, on one row of the same table.
    pub fn gpu_hier_reduce(gpu: GpuBackend) -> Self {
        Self {
            reduction: ReductionStrategy::HierarchicalTree,
            ..Self::gpu_shared_reduce(gpu)
        }
        .named(HIER_REDUCE)
    }

    /// Grid schedule reading and writing **four elements per invocation** on
    /// the contiguous axis.
    ///
    /// The lever the elementwise band was waiting for (ADR-2 section 6):
    /// the native kernels of that band walk their row in `float4` while the
    /// scalar lowering recomputes a full address - four stride products and
    /// three sums - for one `float`. Four is not tuned: it is the width both
    /// native kernels use, so the comparison is against a like-for-like access.
    ///
    /// A specialization, never the fallback: it accepts only a tensor whose
    /// contiguous stride is one element, and the pair must keep a lowering for
    /// the rest instead of handing those nodes back to the native kernel.
    pub fn gpu_grid_vec4(gpu: GpuBackend, block: [u32; 3]) -> Self {
        Self {
            vector_width: 4,
            ..Self::gpu_grid(gpu, block)
        }
        .named(VEC4)
    }

    /// Grid schedule over a **flattened** parallel space: every parallel axis
    /// folded into grid x, decomposed back inside the kernel by magic-number
    /// division.
    ///
    /// The CUDA counterpart of
    /// `vector_width`. Against a native kernel that walks `nelements` flat, the
    /// three-dimensional grid starts with two handicaps that have nothing to do
    /// with the arithmetic: a workgroup covering a row shorter than itself
    /// wastes the difference - 256 lanes for a row of 128 is half the block
    /// idle, on every one of the rows - and the fourth axis stays a sequential
    /// loop no thread shares. Flattening removes both: `ceil(N / block)` full
    /// blocks, nothing sequential, and no grid dimension near its 65 535 ceiling.
    ///
    /// What it costs is the decomposition, and that is why it is not free: one
    /// magic-number division per axis but the last. Native
    /// `k_bin_bcast_unravel` pays exactly this and no more, so the
    /// comparison is like for like.
    ///
    /// A specialization and never the fallback, for the same reason `vec4` is
    /// one: the linear index has to be representable, so the pair keeps a
    /// lowering below it for the shapes it declines
    /// (`ggml_rir_variant_fits_layout`).
    pub fn gpu_grid_flat(gpu: GpuBackend, block: [u32; 3], width: u32) -> Self {
        Self {
            flatten: true,
            vector_width: width,
            ..Self::gpu_grid(gpu, block)
        }
        .named(FLAT)
    }

    /// The flattened grid whose linear index is also the **address**:
    /// `addr = linear · width · elem_bytes`, and no
    /// decomposition at all.
    ///
    /// Flattening
    /// the *grid* removed the idle lanes and the sequential fourth axis, and
    /// left the address exactly where it was: four stride products and three
    /// sums per element, against a native kernel that indexes `base + i ·
    /// elem_size`. What removes those is not another dispatch geometry but a
    /// **layout claim** (every binding contiguous, every binding the same shape),
    /// under which the byte offset of the linear index *is* the linear index,
    /// scaled once.
    ///
    /// The claim's two halves are checked in two different places, and that is
    /// the design rather than an accident:
    ///
    /// - *same shape* is a property of the kernel - every access indexes every
    ///   parallel axis, in dimension order, and nothing folds. `lower` proves it
    ///   on the semantic graph and refuses the schedule otherwise, so no tensor
    ///   is consulted;
    /// - *contiguous* is a property of the tensors, so it is published
    ///   (`rir_variant_desc.linear_addr`) and evaluated at dispatch, exactly as
    ///   `vector_width` publishes `nb[0] == elem_bytes`.
    ///
    /// A specialization, never the fallback: a permuted operand, a view with a
    /// row gap, a row length that is not a whole number of vectors - all keep
    /// the flattened lowering below it, which claims nothing about `nb[]` past
    /// the contiguous stride.
    pub fn gpu_grid_flat_linear(gpu: GpuBackend, block: [u32; 3], width: u32) -> Self {
        Self {
            linear_addr: true,
            ..Self::gpu_grid_flat(gpu, block, width)
        }
        .named(FLAT_LINEAR)
    }

    /// Grid schedule whose contraction reads **staged tiles**
    /// (ADR-2 section 6). The workgroup covers a `block[0] × block[1]`
    /// tile of the output; on each round it stages a `depth`-deep slice of each
    /// contracted operand into shared memory, then every invocation accumulates
    /// that slice from there.
    ///
    /// Same grid geometry as `gpu_grid`, so the dispatch is unchanged: what the
    /// strategy buys is that one row of an operand is read from global memory
    /// once per workgroup rather than once per invocation that needs it.
    ///
    /// `width` above one adds **register tiling**: one invocation owns `width`
    /// consecutive indices of grid x, so the tile it shares is that much
    /// taller and each invocation's share of the cooperative load that much
    /// wider. Both matter, and the second is the one that is easy to miss,
    /// a tile 16 rows wide is staged in 64-byte runs, half a cache line.
    pub fn gpu_grid_tiled(gpu: GpuBackend, block: [u32; 3], depth: u32, width: u32) -> Self {
        Self {
            reduction: ReductionStrategy::TiledStage,
            tile_depth: depth,
            vector_width: width,
            ..Self::gpu_grid(gpu, block)
        }
    }

    // ---------------------------------------------------------------------
    // Named facades. Each is one call to a `gpu_*` constructor and carries no
    // decision of its own. They exist so the ~90 test call sites that name a
    // backend in a constructor remain available to callers; the production
    // table below does not use them.
    // ---------------------------------------------------------------------

    /// `gpu_subgroup` on Vulkan.
    pub fn vulkan_subgroup() -> Self {
        Self::gpu_subgroup(GpuBackend::Vulkan)
    }

    /// `gpu_subgroup` on Metal, where a 32-lane workgroup is exactly one
    /// SIMD-group - hence the name the Metal tests use.
    pub fn metal_simdgroup() -> Self {
        Self::gpu_subgroup(GpuBackend::Metal)
    }

    /// `gpu_grid` on Vulkan.
    pub fn vulkan_grid(block: [u32; 3]) -> Self {
        Self::gpu_grid(GpuBackend::Vulkan, block)
    }

    /// `gpu_grid` on Metal.
    pub fn metal_grid(block: [u32; 3]) -> Self {
        Self::gpu_grid(GpuBackend::Metal, block)
    }

    /// `gpu_blocked_scan` on Vulkan.
    pub fn vulkan_blocked_scan() -> Self {
        Self::gpu_blocked_scan(GpuBackend::Vulkan)
    }

    /// `gpu_blocked_scan` on Metal.
    pub fn metal_blocked_scan() -> Self {
        Self::gpu_blocked_scan(GpuBackend::Metal)
    }

    /// `gpu_tiled_scan` on Vulkan.
    pub fn vulkan_tiled_scan(lanes: u32, items: u32) -> Self {
        Self::gpu_tiled_scan(GpuBackend::Vulkan, lanes, items)
    }

    /// `gpu_tiled_scan` on Metal.
    pub fn metal_tiled_scan(lanes: u32, items: u32) -> Self {
        Self::gpu_tiled_scan(GpuBackend::Metal, lanes, items)
    }

    /// `gpu_shared_reduce` on Vulkan.
    pub fn vulkan_shared_reduce() -> Self {
        Self::gpu_shared_reduce(GpuBackend::Vulkan)
    }

    /// `gpu_shared_reduce` on Metal.
    pub fn metal_shared_reduce() -> Self {
        Self::gpu_shared_reduce(GpuBackend::Metal)
    }

    /// `gpu_grid_vec4` on Vulkan.
    pub fn vulkan_grid_vec4(block: [u32; 3]) -> Self {
        Self::gpu_grid_vec4(GpuBackend::Vulkan, block)
    }

    /// `gpu_grid_vec4` on Metal.
    pub fn metal_grid_vec4(block: [u32; 3]) -> Self {
        Self::gpu_grid_vec4(GpuBackend::Metal, block)
    }

    /// `gpu_grid_tiled` on Vulkan.
    pub fn vulkan_grid_tiled(block: [u32; 3], depth: u32, width: u32) -> Self {
        Self::gpu_grid_tiled(GpuBackend::Vulkan, block, depth, width)
    }

    /// `gpu_grid_tiled` on Metal.
    pub fn metal_grid_tiled(block: [u32; 3], depth: u32, width: u32) -> Self {
        Self::gpu_grid_tiled(GpuBackend::Metal, block, depth, width)
    }
}

/// Variant name of the blocked scan. Named once because it ends up in a file
/// name, an entrypoint, a SPIR-V symbol and a registry row.
pub const BLOCKED_SCAN: &str = "blocked";

/// Variant name of the coalesced tiled scan.
pub const TILED_SCAN: &str = "tiled";

/// Variant name of the workgroup-wide shared-memory reduction. Distinct from
/// the bench module's `SHARED_SCAN` only because the two never appear on the
/// same kernel - v1
/// forbids carrying a scan and a reduction together - but naming them apart
/// keeps an artifact name saying which collective it holds.
pub const SHARED_REDUCE: &str = "shared_reduce";

/// Variant name of the two-stage workgroup reduction.
pub const HIER_REDUCE: &str = "hier_reduce";

pub const VEC4: &str = "vec4";

/// Variant name of the flattened grid. One name whatever
/// the vector width the flattening carries: what distinguishes this variant
/// from its neighbours is the dispatch geometry, and `vector_width` is
/// published beside it.
pub const FLAT: &str = "flat";

/// Variant name of the flattened grid that addresses **linearly**.
/// Distinct from `FLAT` because it is a distinct claim
/// and therefore a distinct artifact: the same body, one address expression
/// instead of a decomposition, and a layout the dispatcher has to check before
/// selecting it.
pub const FLAT_LINEAR: &str = "flat_linear";

/// Variant name of the flattened **row** space. Distinct
/// from `FLAT` because what it flattens is the workgroup grid of a collective
/// and not the invocation grid of an elementwise kernel - one artifact each,
/// and one claim each.
pub const FLAT_ROWS: &str = "flat_rows";
