//! Loop IR statements: the loop forms, the collectives that stayed
//! primitives, and the shared-memory accesses the lowered scans are built
//! from.

use rir_core::{ArgId, AxisId, ReduceOp};

use super::*;

#[derive(Clone, Debug)]
pub struct Inst {
    pub dst: VarId,
    pub expr: LExpr,
}

/// One argument staged into shared memory by a whole workgroup, as a
/// `n_depth × n_rows` tile laid out depth-major - element `(d, r)` sits at
/// `d · n_rows + r`, so the `n_rows` consumers of one depth slice read
/// consecutive words.
///
/// Everything an emitter needs is here, and nothing it would have to invent:
/// the cooperative loop drives `row` and `depth`, binds `row_global` and
/// `depth_global` to `origin + local`, and runs `load` - statements lowering
/// built in terms of those two registers and whatever outer loop variables are
/// in scope, leaving the element in `value`.
///
/// **`load` is a body and not an address**, and that is what lets a tile stage
/// a quantized operand. A plain F32 operand lowers to
/// a single `Load`; a quantized one lowers to the same fused decoder a direct
/// read gets, block arithmetic included. The format's formula therefore stays
/// in lowering, once, and no emitter learns a block layout - which is the rule
/// an `addr` field would have forced all three of them to break.
///
/// An element outside the tensor is staged as **zero**. That is not a
/// convenience: it is only correct because a staged tile feeds a product
/// summed over the depth axis, where a zero factor contributes the sum's
/// identity. Lowering checks that shape before it builds one of these.
#[derive(Clone, Debug)]
pub struct TileStage {
    /// Register naming the shared array. It holds no value of its own - it is
    /// the tile's identity, the way a `WorkgroupReduce`'s `dst` names its slot.
    pub tile: VarId,
    pub arg: ArgId,
    /// Tile-local coordinates, driven by the emitter's cooperative loop.
    pub row: VarId,
    pub depth: VarId,
    /// Global coordinates of the staged element: `row_origin + row` and
    /// `depth_origin + depth`. These are the registers `load` is written in.
    pub row_global: VarId,
    pub depth_global: VarId,
    pub row_origin: VarId,
    pub depth_origin: VarId,
    /// Axes the two coordinates walk, for the in-range test.
    pub row_axis: AxisId,
    pub depth_axis: AxisId,
    /// Linear index of the segment's first element in the tile, bound by the
    /// emitters' cooperative loop. `load` writes through it with `StoreTile`.
    pub slot: VarId,
    /// Consecutive `row` indices one invocation stages. `1` is one element per
    /// invocation, the mapping every F32 tile uses.
    ///
    /// Above one it is what makes a quantized tile affordable:
    /// a block header - an F16 scale, a packed
    /// 6-bit scale pair - is shared by `block_elements` consecutive elements,
    /// and with one element per invocation it was re-read for each of them.
    /// A segment reads it once and lowering puts the per-element work in a
    /// `ForConst` under it. Lowering only chooses a span it can prove keeps
    /// the segment inside one block and inside the tensor.
    pub span: u32,
    /// Statements staging one segment, executed only when its first element is
    /// inside both axes. They write the tile themselves, through `StoreTile`.
    pub load: Vec<Stmt>,
    pub n_rows: u32,
    pub n_depth: u32,
}

impl TileStage {
    /// Elements of shared storage this tile occupies.
    pub fn len(&self) -> u32 {
        self.n_rows * self.n_depth
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One accumulator of a grouped workgroup collective: the combiner, the lane's
/// contribution, and the register every lane receives the total in.
#[derive(Clone, Copy, Debug)]
pub struct GroupRed {
    pub op: ReduceOp,
    pub src: VarId,
    pub dst: VarId,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// Grid-mapped axis: `blockIdx`/`gl_WorkGroupID` on GPU, or an outer
    /// parallelizable loop on CPU.
    Parallel {
        var: VarId,
        axis: AxisId,
        level: HwLevel,
        /// Number of consecutive axis indices this invocation covers. `1` is
        /// the scalar mapping; `w > 1` means `var` is the index of the **first**
        /// element of a `w`-wide vector, so the builtin is multiplied by `w` and
        /// the dispatcher divides the extent by `w` more (`grid_axes`).
        vector: u32,
        /// Whether an invocation past the axis extent may leave immediately.
        ///
        /// True everywhere a workgroup's invocations are independent, which is
        /// the whole grid mapping until staging appears. False when the body
        /// contains a workgroup barrier: an invocation that returned early
        /// would strand the ones still waiting, so the bound moves down to the
        /// writes (`Stmt::InBounds`) and the out-of-range invocations stay
        /// alive to carry their share of the cooperative loads.
        bounded: bool,
        body: Vec<Stmt>,
    },
    /// The **flattened** grid: one linear index over the
    /// product of every parallel axis, decomposed back into one index per axis.
    ///
    /// It replaces the `Parallel` nest, and it is not a spelling of it. The nest
    /// maps three axes to three grid dimensions and leaves the rest sequential,
    /// so a kernel launched on `[128,16,16,1]` with a 256-wide block dispatches
    /// one block of 256 threads for a row of 128 - half the threads idle - plus
    /// a sequential loop over `batch`, plus one stride product per axis and per
    /// element. This dispatches `ceil(N / block)` full blocks over the whole
    /// space, with no dimension wasted and no axis left sequential.
    ///
    /// The decomposition is by **magic-number division** and not by `/` and `%`:
    /// `k_bin_bcast_unravel` - the native
    /// kernel that already does this - precomputes a multiplier per axis host
    /// side (`init_fastdiv_values`) rather than dividing, and that an
    /// integer division per axis and per element is exactly the address algebra
    /// this statement exists to remove. The divisors, their multipliers and
    /// their shifts are therefore **published in the constant buffer**, filled
    /// by the same host code that fills the extents, and every emitter prints
    /// the same `umulhi`-based form the oracle computes exactly.
    ///
    /// `axes` is in decomposition order, fastest first: `axes[0]` walks the
    /// contiguous axis and is the only one a vector width applies to. The
    /// divisor of axis `i` is `ceil(extent(axes[i]) / per_i)`, with `per_0 =
    /// vector` and `per_i = 1` above it - the same `ceil` the grid of a
    /// non-flattened dispatch applies, so a row that is not a whole number of
    /// vectors keeps the `VecTail` branch it already had.
    ParallelFlat {
        /// The linear index: one invocation per point of the flattened space
        /// under `HwLevel::Global`, one **workgroup** per point under
        /// `HwLevel::Grid`.
        linear: VarId,
        /// Which hardware index the linear one is read from.
        ///
        /// `Global(0)` is the flattened *dispatch*: one
        /// invocation per point, for a kernel with no collective. `Grid(0)` is
        /// the flattened *rows*: one workgroup per point,
        /// whose lanes then cooperate on it - which is the only mapping a
        /// collective can have, and the reason the two are one statement with a
        /// level rather than two statements. What flattening buys is the same in
        /// both cases: no axis left sequential, and no grid dimension against
        /// its 65 535 ceiling.
        level: HwLevel,
        /// `(index register, axis)`, fastest axis first.
        ///
        /// The list is what the **host** decomposes - it is where the divisors,
        /// and therefore the total, come from (`LoopKernel::flat_axes`) - and it
        /// stays complete whatever `decompose` says. What `decompose` decides is
        /// whether the *shader* also performs it.
        axes: Vec<(VarId, AxisId)>,
        /// Whether the shader decomposes `linear` back into the index registers
        /// of `axes`.
        ///
        /// False is the linear-addressing lowering: under
        /// the claim that every binding is contiguous and of the same shape,
        /// every address is `linear · width · elem_bytes`, so not one axis index
        /// is read - and computing them would be the magic-number division this
        /// lowering exists to remove. The index registers are then never
        /// assigned, which is safe because lowering never binds an axis to one:
        /// a kernel whose body reads a position fails with `AxisOutOfScope`
        /// rather than reading an undefined register.
        decompose: bool,
        /// Indices of `axes[0]` one invocation covers. `1` is the scalar
        /// mapping.
        vector: u32,
        body: Vec<Stmt>,
    },
    /// The two halves of a vectorized traversal: the vector body when the whole
    /// `width`-wide vector fits inside the axis, the scalar tail otherwise.
    ///
    /// The branch is explicit rather than left to a guard per access because
    /// that is the only way a vector load can be printed at all: a partially
    /// out-of-bounds `float4` has no defined value, and rejecting the shapes
    /// that produce one would hand a quarter of the domain back to the native
    /// kernel. `tail_var` walks `[base, extent)`, so it covers at most `width`
    /// elements - the last, incomplete vector of the row.
    VecTail {
        /// Index of the first element of the vector: the enclosing
        /// `Parallel`'s variable.
        base: VarId,
        axis: AxisId,
        width: u32,
        vec_body: Vec<Stmt>,
        tail_var: VarId,
        tail_body: Vec<Stmt>,
    },
    /// Workgroup/subgroup lanes: the body executes once per lane with `var`
    /// set to the lane index. GPU hardware executes this as SPMD with no
    /// emitted loop; the interpreter simulates it with per-lane registers.
    ParallelLane {
        var: VarId,
        lanes: u32,
        body: Vec<Stmt>,
    },
    /// Sequential loop over a full axis extent. `reverse` traverses the axis
    /// in descending order for backward scans.
    For {
        var: VarId,
        axis: AxisId,
        reverse: bool,
        body: Vec<Stmt>,
    },
    /// Strided sequential loop: `var = start; var < extent(axis); var += step`.
    /// Used to distribute axis traversal across lanes.
    ForStrided {
        var: VarId,
        axis: AxisId,
        start: VarId,
        step: u32,
        body: Vec<Stmt>,
    },
    /// Sequential loop over the **contiguous chunk** of an axis owned by one
    /// lane. With extent `n` and `lanes` lanes, the chunk is `ceil(n / lanes)`
    /// and lane `l` traverses `[l·chunk, min((l+1)·chunk, n))`.
    ///
    /// A scan needs contiguity where a reduction does not: a lane's partial
    /// result must be a prefix of a *contiguous* range, otherwise combining it
    /// with the lanes before it means nothing. `ForStrided`, which interleaves,
    /// is therefore not usable here.
    ///
    /// `reverse` mirrors the assignment: lane `l` takes the `l`-th chunk **from
    /// the end**, traversed descending, so increasing lane order keeps
    /// following the scan direction.
    ForChunk {
        var: VarId,
        axis: AxisId,
        lane: VarId,
        lanes: u32,
        reverse: bool,
        body: Vec<Stmt>,
    },
    InitAcc {
        acc: VarId,
        op: ReduceOp,
    },
    Accum {
        acc: VarId,
        op: ReduceOp,
        value: VarId,
    },
    /// Collective reduction across the enclosing `ParallelLane`. `dst`
    /// receives the result in **every** lane (`subgroupAdd` semantics). Valid
    /// only where every lane reaches it: the top level of a `ParallelLane`
    /// body, or a loop there whose bounds are the same for every lane
    /// (`ForTiled`, `ForConst`, `For`).
    LaneReduce {
        op: ReduceOp,
        src: VarId,
        dst: VarId,
    },
    /// Collective **exclusive prefix** across the enclosing `ParallelLane`:
    /// lane `l` receives the combination of lanes `0..l`, and lane 0 the
    /// identity (`subgroupExclusiveAdd` semantics). Same placement rule as
    /// `LaneReduce`.
    ///
    /// It is what turns per-lane chunk totals into per-lane scan offsets; its
    /// order is fixed by the lane count the schedule declares, not by the
    /// hardware's topology, so it is reproducible - `Deterministic`, not
    /// `ExactOrder`.
    LaneScan {
        op: ReduceOp,
        src: VarId,
        dst: VarId,
    },
    /// Collective reduction across **all** lanes of the enclosing
    /// `ParallelLane`, including workgroups wider than one hardware subgroup.
    /// Emitters implement a fixed shared-memory tree and broadcast the result
    /// to every lane. `lanes` is a power of two fixed by the schedule.
    ///
    /// It carries a **group** of accumulators, not one: two reductions of the
    /// same dependency level have
    /// the same axis, the same lane count and the same tree, so what separated
    /// them was one statement each - and therefore one full series of barriers
    /// each. `rms_norm_back` paid eighteen for two values that could be carried
    /// by the same nine. Emitters apply each level of the tree to *every* entry
    /// before the level's barrier, which is why the group is on the statement
    /// rather than in a pass over statements: a barrier is not something an
    /// emitter may move.
    ///
    /// The topology is unchanged - level `l` still combines lane `i` with lane
    /// `i + 2^l`, in that order, for each accumulator independently - so a
    /// grouped reduction sums exactly what an ungrouped one summed. Grouping is
    /// a synchronisation decision, not an arithmetic one.
    WorkgroupReduce {
        /// One entry per accumulator, reduced under the same barriers. A group
        /// of one is the ungrouped statement.
        reds: Vec<GroupRed>,
        lanes: u32,
        /// Width of the hardware subgroup when the reduction is **hierarchical**
        /// `None` for the flat shared tree.
        ///
        /// `Some(w)` replaces the `log2(lanes)` barriers with two stages: each
        /// subgroup reduces its own lanes with the backend's primitive, lane
        /// zero of each writes one total, and the first subgroup reduces the
        /// `lanes / w` totals. The shared array is that much shorter - sized by
        /// the number of subgroups and not by the lane count, which is the
        /// storage half of the item.
        ///
        /// The group is carried across both stages exactly as it is across the
        /// tree's levels, and for the same reason: what a group shares is the
        /// barriers, so a new stage that ran per accumulator would put the
        /// barriers back one by one.
        subgroup: Option<u32>,
    },
    /// Workgroup barrier: every lane of the enclosing `ParallelLane` waits here
    /// before any of them reads what another wrote
    /// (`barrier()` / `threadgroup_barrier`).
    ///
    /// **The statement that lets an algorithm be lowered instead of printed.**
    /// Until it existed, a collective whose
    /// expansion needs a series of barriers could only live inside an emitter,
    /// which is where `blelloch` and `tiled_scan` lived, twice, one copy per
    /// backend. A barrier is not something an emitter may move, so it is not
    /// something an emitter may own.
    ///
    /// Like every backend's barrier, it must be reached by all lanes or none:
    /// lowering places it at the top level of a lane body or of a loop over a
    /// uniform bound, never inside `Stmt::If`.
    Barrier,
    /// Executes the body when a `Bool` register holds true.
    ///
    /// The general guard, beside the two specialized ones: `InBounds` compares
    /// indices with axis extents, `LaneZero` selects one lane, and this takes
    /// whatever condition an algorithm computed - `idx < lanes` on a tree level.
    If {
        cond: VarId,
        body: Vec<Stmt>,
    },
    /// Assigns an already-declared register: `var = value`.
    ///
    /// Registers are otherwise single-assignment - emitters print a `Compute`
    /// as a declaration - and the two exceptions were both accumulators
    /// (`InitAcc` then `Accum`). A loop-carried value that is *replaced* rather
    /// than combined needs this: the tiled scan's running total is read back
    /// from the tile at the end of each round, and a second declaration would
    /// shadow it instead of updating it.
    Set {
        var: VarId,
        value: VarId,
    },
    /// Only lane 0 executes the body, for uniform per-row writes.
    LaneZero {
        lane: VarId,
        body: Vec<Stmt>,
    },
    /// Sequential loop with a **constant** trip count, `var` walking
    /// `0..count`. The depth of a staged tile is a property of the schedule,
    /// not of a tensor, so it is not an axis extent and must not be printed as
    /// one - the count is what lets a backend unroll it.
    ForConst {
        var: VarId,
        count: u32,
        body: Vec<Stmt>,
    },
    /// Tile loop over an axis: `var = 0; var < extent(axis); var += step`.
    /// `var` is the axis index of the tile's **first** element, so the body
    /// covers `[var, var + step)` - clipped by the staging guard, not by the
    /// loop.
    ForTiled {
        var: VarId,
        axis: AxisId,
        step: u32,
        body: Vec<Stmt>,
    },
    /// Cooperative staging of one or more tiles into shared memory, then the
    /// body that consumes them.
    ///
    /// The expansion is fixed, like the collectives above: barrier, one
    /// cooperative load loop per tile spread over the workgroup's `threads`
    /// invocations, barrier, body, barrier. The leading and trailing barriers
    /// are what make the statement safe to re-enter - a tile loop stages into
    /// the same storage on every round.
    ///
    /// Grouping the tiles in one statement rather than emitting one staging
    /// statement each is what keeps that to **two** barriers per round however
    /// many tiles the contraction reads.
    StageTiles {
        tiles: Vec<TileStage>,
        /// Invocations in the workgroup, over which each cooperative load loop
        /// is spread. From the schedule's block, but read here so an emitter
        /// never recomputes it.
        threads: u32,
        body: Vec<Stmt>,
    },
    /// Writes `value` at element `index` of a **shared** array - the store of
    /// the shared address space, as `Store` is the store of the global one.
    ///
    /// The scan trees write their levels through this statement, not only the
    /// staged tiles. The array is named by the register
    /// that declares it (`LoopKernel::shared`), and the index is an element
    /// index - shared storage is packed by construction, so it needs no stride
    /// and no byte address.
    StoreShared {
        array: VarId,
        index: VarId,
        value: VarId,
    },
    /// Reads element `index` of a shared array. `width > 1` reads `width`
    /// consecutive elements, always contiguous for the same reason.
    LoadShared {
        dst: VarId,
        array: VarId,
        index: VarId,
        width: u32,
    },
    /// Executes the body only for invocations whose indices are inside their
    /// axes. The counterpart of `Parallel { bounded: false }`: what the early
    /// return no longer does, this does at the point where it matters.
    InBounds {
        bounds: Vec<(VarId, AxisId)>,
        body: Vec<Stmt>,
    },
    /// Addressed access whose byte address is the sum of its terms.
    ///
    /// `width > 1` reads `width` consecutive elements starting at that address,
    /// into a `VarKind::Vec` register. "Consecutive" is the registry's job: the
    /// widened dimension must have `nb[0] == elem_bytes`, which the registry
    /// publishes and `supports_op` refuses when it does not hold.
    Load {
        dst: VarId,
        arg: ArgId,
        ty: MemType,
        addr: Vec<AddrTerm>,
        width: u32,
    },
    Store {
        arg: ArgId,
        /// Element type **in memory**. The register that
        /// feeds a store is always F32; this says what it is narrowed to on the
        /// way out, exactly as `Load::ty` says what a register was widened from.
        ///
        /// It is carried rather than derived from the argument's dtype for the
        /// same reason `Load` carries one: an emitter prints a statement and
        /// decides nothing, and a quantized argument already has two access
        /// types on one binding.
        ty: MemType,
        addr: Vec<AddrTerm>,
        value: VarId,
        width: u32,
        /// Axis the widened index walks, when the last vector of a row may run
        /// past its extent. `Some` makes the store per-component: component `c`
        /// is written only if `base + c < extent(axis)`.
        ///
        /// This is `VecTail`'s job done at the write instead of around the whole
        /// body, and it exists because a tiled body cannot be branched: the two
        /// halves would have to contain the barriers, and a workgroup where some
        /// invocations take one half and some the other is a deadlock. Staging
        /// pads with zeros, so the accumulation past the edge is already
        /// harmless; only the write has to be held back.
        bound: Option<(VarId, AxisId)>,
    },
    Compute(Inst),
}

/// The statement lists a statement contains, for a read-only walk.
///
/// The passes have their own `&mut` version; this is the one a test or a tool
/// uses to state a property of a whole nest without matching every arm again,
/// which is what a structural assertion needs to replace a substring search.
pub fn child_blocks_of(s: &Stmt) -> Vec<&Vec<Stmt>> {
    match s {
        Stmt::Parallel { body, .. }
        | Stmt::ParallelFlat { body, .. }
        | Stmt::ParallelLane { body, .. }
        | Stmt::For { body, .. }
        | Stmt::ForStrided { body, .. }
        | Stmt::ForChunk { body, .. }
        | Stmt::ForConst { body, .. }
        | Stmt::ForTiled { body, .. }
        | Stmt::LaneZero { body, .. }
        | Stmt::If { body, .. }
        | Stmt::InBounds { body, .. } => vec![body],
        Stmt::StageTiles { tiles, body, .. } => {
            let mut v: Vec<&Vec<Stmt>> = tiles.iter().map(|t| &t.load).collect();
            v.push(body);
            v
        }
        Stmt::VecTail {
            vec_body,
            tail_body,
            ..
        } => vec![vec_body, tail_body],
        _ => Vec::new(),
    }
}

/// The half-open **ascending** range of an axis owned by `lane` under
/// `ForChunk`. Reverse only mirrors the assignment; the caller still traverses
/// the returned range descending.
///
/// One definition for the interpreter and, in printed form, for every emitter:
/// a chunk geometry that differed between the oracle and the device would make
/// the two disagree on the *result*, not just on the speed.
pub fn chunk_range(n: usize, lane: usize, lanes: u32, reverse: bool) -> (usize, usize) {
    let chunk = n.div_ceil(lanes.max(1) as usize);
    let start = (lane * chunk).min(n);
    let end = ((lane + 1) * chunk).min(n);
    if reverse {
        (n - end, n - start)
    } else {
        (start, end)
    }
}
