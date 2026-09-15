//! The GPU backends' lexicon.
//!
//! **This is a table, not an extension point.** The golden rule of
//! [`crate`] - *emitters make no decisions* - applies here first: every entry
//! below is a word, a constant or a total function of data the printer already
//! holds. None of them inspects a backend tag, none of them chooses between two
//! ways of computing anything, and a `match` on `Self` inside the common
//! skeleton is a regression even when the tests pass.
//!
//! The cartography that produced the list gives the reason each entry
//! exists, rather than repeating it here. What is
//! repeated is the two traps, because they are the ones a reader will otherwise
//! re-introduce: [`Builtin`] has two fields, and [`Dialect::lane_builtin`] is
//! not [`Dialect::WG_LANE_INDEX`].
//!
//! One entry left the table rather than joining it. `FLAT_BUILTIN` spelled the
//! global invocation index a second time, which was harmless while a flattened
//! dispatch was the only flattened thing; flattening rows over
//! *workgroups* means the index then has to follow the level. A duplicate entry
//! that has to disagree with its original is the decision this module may not
//! take, so `Stmt::ParallelFlat` reads [`Dialect::axis_builtin`] like every
//! other mapped index.
//!
//! `cpu.rs` is deliberately absent. The code it emits may depend on no RIR
//! crate - a documented constraint, guarded by
//! `rir-kernels/tests/f16_oracles.rs` - so its copy of the skeleton is an
//! independent witness and not a triplication to absorb.

use rir_core::{CmpOp, LutId, ReduceOp};
use rir_lower::{HwLevel, LoopKernel, VarKind};

/// A hardware index, in the two syntactic positions the skeleton prints it in.
///
/// The second field is the trap. `Stmt::Parallel` prints the builtin alone when
/// the axis is scalar and `{builtin} * {vector}u` when it is not, and CUDA's
/// `Global` builtin is the *sum* `blockIdx.x * blockDim.x + threadIdx.x`:
/// multiplying it needs a parenthesis that Metal's `global_id.x` does not.
///
/// Making that a parenthesis the skeleton adds `if backend == Cuda` would be
/// exactly the decision this module may not take, so it is a second column of
/// the table instead. Where the builtin is a single token, both fields hold the
/// same string.
#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    /// The builtin as a standalone value.
    pub plain: &'static str,
    /// The builtin as the left operand of a multiplication.
    pub as_factor: &'static str,
}

impl Builtin {
    /// A builtin that is a single token: no grouping needed in either position.
    const fn atom(s: &'static str) -> Self {
        Builtin {
            plain: s,
            as_factor: s,
        }
    }

    /// A builtin whose factor position is spelled differently.
    const fn grouped(plain: &'static str, as_factor: &'static str) -> Self {
        Builtin { plain, as_factor }
    }
}

/// The unary float functions the Loop IR builds. Named rather than passed as a
/// string so that adding one to `LExpr` fails to compile here too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryFn {
    Sqrt,
    Exp,
    Tanh,
}

/// How a backend compares two vectors, when it has a form for it at all.
///
/// The three languages disagree on more than a name here, so the table has a
/// second column for the same reason [`Builtin`] does: `splat_operands` is a
/// property of the *function named on the left*, not a branch on the backend.
/// GLSL's `greaterThan` needs both operands at the same width; CUDA's prelude
/// helpers accept a scalar on either side and would print a redundant
/// broadcast. MSL has no entry at all - its comparison operators broadcast, so
/// the scalar spelling is already the vector one.
#[derive(Clone, Copy, Debug)]
pub struct VectorCmp {
    pub name: &'static str,
    pub splat_operands: bool,
}

/// One backend's lexicon.
///
/// Implemented by a zero-sized marker per backend, not by the printer: the
/// table holds no state, and keeping it stateless is what makes "no decisions"
/// checkable by reading the impl rather than by trusting it.
pub trait Dialect {
    /// The name `EmitError::UnsupportedStmt` already carries.
    const BACKEND: &'static str;

    /// The unsigned integer type an index is declared as.
    const UINT: &'static str;

    /// The constant-buffer prefix: `p.` against `pc.`.
    const PARAMS: &'static str;

    /// The float zero a staged tile is padded with.
    const ZERO: &'static str;

    /// The whole barrier statement, semicolon included.
    const BARRIER: &'static str;

    /// The lane index under `Stmt::WorkgroupReduce` and `Stmt::StageTiles`'
    /// cooperative loader.
    ///
    /// Not the same entry as [`Self::lane_builtin`], and Metal is why: it picks
    /// between two spellings for `ParallelLane` and always uses
    /// `threadgroup_lane_id` here.
    const WG_LANE_INDEX: &'static str;

    /// The componentwise `?:` - `rir_vselect` / `select` / `mix`. All three
    /// take `(f, t, cond)` and splat their three arguments identically.
    const SELECT: &'static str;

    /// The grid or global index an axis is mapped to, or `None` when the level
    /// has no hardware index on this backend (`HwLevel::Lane`, which the
    /// skeleton turns into the existing `EmitError`).
    fn axis_builtin(level: HwLevel) -> Option<Builtin>;

    /// The lane index under `Stmt::ParallelLane`. Takes the kernel because
    /// Metal's depends on `LoopKernel::uses_shared`.
    fn lane_builtin(k: &LoopKernel) -> &'static str;

    /// The linear thread index inside a block, as `Stmt::StageTiles` spells it.
    fn local_index(k: &LoopKernel) -> &'static str;

    /// A float literal, in a spelling that does not change the computation:
    /// CUDA's needs an `f` suffix or the expression is promoted to `double`,
    /// and its non-finite constants are spelled by their bits.
    fn float_literal(c: f32) -> String;

    /// The identity of a reduction - three spellings of `0` and of `-inf`.
    fn identity(op: ReduceOp) -> &'static str;

    /// Combining two operands already spelled as text: the shared-memory tree
    /// of `WorkgroupReduce`, whose operands are array slots and not registers.
    fn combine(op: ReduceOp, a: &str, b: &str) -> String;

    /// Combining two operands whose widths are known - `fmaxf` against
    /// `rir_vmax` on CUDA, one width-polymorphic `max` on the two others.
    fn combine_at(op: ReduceOp, width: Option<u32>, a: &str, b: &str) -> String;

    /// What a register of this kind is declared as.
    fn decl_ty(kind: VarKind) -> String;

    /// A vector register built from its components.
    fn vec_ctor(width: u32, comps: &[String]) -> String;

    /// An operand printed at `width`: itself if it is already a vector, its
    /// broadcast otherwise.
    fn splat(kind: VarKind, width: u32, text: &str) -> String;

    /// Component `c` of a vector register, or the register itself when it is
    /// scalar - a scalar under a vector store is a broadcast.
    fn component(kind: VarKind, text: &str, c: usize) -> String;

    /// A unary float function, in the spelling the operand's width requires.
    fn unary_fn(f: UnaryFn, width: Option<u32>) -> &'static str;

    /// The subgroup reduction primitive.
    fn lane_reduce_fn(op: ReduceOp) -> &'static str;

    /// The subgroup exclusive-scan primitive, or `None` where the backend has
    /// none - MSL has no exclusive max prefix, and printing a function that
    /// does not exist would fail at shader compile time, far from the cause.
    fn lane_scan_fn(op: ReduceOp) -> Option<&'static str>;

    /// The name of a constant table at module scope.
    fn lut(k: &LoopKernel, table: LutId) -> String;

    /// How this backend compares two vectors, or `None` when its scalar
    /// spelling already broadcasts.
    fn vector_cmp(op: CmpOp) -> Option<VectorCmp>;
}

/// The comparison symbol, identical in the three languages.
pub fn cmp_symbol(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

/// The artifact-prefixed LUT name CUDA and Metal both use: nothing guarantees
/// that one translation unit will not see two artifacts.
fn artifact_lut(k: &LoopKernel, table: LutId) -> String {
    format!(
        "rir_{}_{}",
        crate::manifest::artifact_name(k),
        table.symbol()
    )
}

// ---------------------------------------------------------------------------

/// CUDA C++.
#[derive(Clone, Copy, Debug)]
pub struct Cuda;

impl Dialect for Cuda {
    const BACKEND: &'static str = "cuda";
    const UINT: &'static str = "uint32_t";
    const PARAMS: &'static str = "p.";
    const ZERO: &'static str = "0.0f";
    const BARRIER: &'static str = "__syncthreads();";
    /// `threadIdx.x` and not the linear index: every schedule that lowers to a
    /// lane collective declares its block as `[lanes, 1, 1]`, which is also
    /// what makes the fixed 32 of the warp prelude meaningful.
    const WG_LANE_INDEX: &'static str = "threadIdx.x";
    const SELECT: &'static str = "rir_vselect";

    /// Every CUDA builtin is parenthesized in factor position, including the
    /// `blockIdx.*` atoms that would not need it.
    ///
    /// That is not an oversight kept for the sake of the byte count: the emitter
    /// wrote `({builtin}) * {vector}u` unconditionally, and the parenthesis is
    /// *load-bearing* on the `Global` levels, whose builtin is a sum. Dropping
    /// it on the grid levels alone would make the entry's shape depend on which
    /// level asked, for no reader's benefit.
    fn axis_builtin(level: HwLevel) -> Option<Builtin> {
        Some(match level {
            HwLevel::Grid(0) => Builtin::grouped("blockIdx.x", "(blockIdx.x)"),
            HwLevel::Grid(1) => Builtin::grouped("blockIdx.y", "(blockIdx.y)"),
            HwLevel::Grid(_) => Builtin::grouped("blockIdx.z", "(blockIdx.z)"),
            HwLevel::Global(0) => Builtin::grouped(
                "blockIdx.x * blockDim.x + threadIdx.x",
                "(blockIdx.x * blockDim.x + threadIdx.x)",
            ),
            HwLevel::Global(1) => Builtin::grouped(
                "blockIdx.y * blockDim.y + threadIdx.y",
                "(blockIdx.y * blockDim.y + threadIdx.y)",
            ),
            HwLevel::Global(_) => Builtin::grouped(
                "blockIdx.z * blockDim.z + threadIdx.z",
                "(blockIdx.z * blockDim.z + threadIdx.z)",
            ),
            HwLevel::Lane => return None,
        })
    }

    fn lane_builtin(_k: &LoopKernel) -> &'static str {
        "threadIdx.x"
    }

    /// Inline rather than declared: a nest may carry two `StageTiles`, and a
    /// `const` at the top of each would be a redefinition in the same scope.
    fn local_index(_k: &LoopKernel) -> &'static str {
        "(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))"
    }

    /// The `f` suffix is not cosmetic: without it the literal is a `double`,
    /// and every expression it takes part in would be promoted - a different
    /// computation from the one the oracle ran, and a slower one.
    fn float_literal(c: f32) -> String {
        if c.is_finite() {
            format!("{c:?}f")
        } else {
            format!("__int_as_float({:#010x})", c.to_bits())
        }
    }

    /// `-inf` is spelled by its bits rather than by `-INFINITY`, exactly as
    /// Metal spells it: `-use_fast_math` implies `--ftz=true`, and a literal
    /// that goes through a constant fold is one more thing to reason about than
    /// a bit pattern that does not.
    fn identity(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "0.0f",
            ReduceOp::Max => "__int_as_float(0xff800000)",
        }
    }

    fn combine(op: ReduceOp, a: &str, b: &str) -> String {
        match op {
            ReduceOp::Sum => format!("{a} + {b}"),
            ReduceOp::Max => format!("fmaxf({a}, {b})"),
        }
    }

    fn combine_at(op: ReduceOp, width: Option<u32>, a: &str, b: &str) -> String {
        match (op, width) {
            (ReduceOp::Sum, _) => format!("{a} + {b}"),
            (ReduceOp::Max, Some(_)) => format!("rir_vmax({a}, {b})"),
            (ReduceOp::Max, None) => format!("fmaxf({a}, {b})"),
        }
    }

    fn decl_ty(kind: VarKind) -> String {
        match kind {
            VarKind::Idx => "uint32_t".to_string(),
            VarKind::F32 => "float".to_string(),
            VarKind::Bool => "bool".to_string(),
            VarKind::Vec(w) => format!("rir_vf<{w}>"),
            VarKind::VecBool(w) => format!("rir_vb<{w}>"),
        }
    }

    /// A braced initializer, not a call: `rir_vf<N>` is an aggregate, and the
    /// declaration next to it already names the type. The doubled braces are
    /// the aggregate's own - one for the struct, one for its `float c[N]`.
    fn vec_ctor(_width: u32, comps: &[String]) -> String {
        format!("{{{{{}}}}}", comps.join(", "))
    }

    fn splat(kind: VarKind, width: u32, text: &str) -> String {
        match kind {
            VarKind::Vec(_) | VarKind::VecBool(_) => text.to_string(),
            VarKind::Bool => format!("rir_vbsplat<{width}>({text})"),
            _ => format!("rir_vsplat<{width}>({text})"),
        }
    }

    fn component(kind: VarKind, text: &str, c: usize) -> String {
        match kind {
            VarKind::Vec(_) => format!("{text}.c[{c}]"),
            _ => text.to_string(),
        }
    }

    /// `sqrtf`/`expf`/`tanhf` and not their double counterparts: an implicit
    /// widening here would be a different function from the one the native
    /// kernel calls, on top of what `-use_fast_math` already substitutes.
    fn unary_fn(f: UnaryFn, width: Option<u32>) -> &'static str {
        match (f, width) {
            (UnaryFn::Sqrt, None) => "sqrtf",
            (UnaryFn::Sqrt, Some(_)) => "rir_vsqrt",
            (UnaryFn::Exp, None) => "expf",
            (UnaryFn::Exp, Some(_)) => "rir_vexp",
            (UnaryFn::Tanh, None) => "tanhf",
            (UnaryFn::Tanh, Some(_)) => "rir_vtanh",
        }
    }

    fn lane_reduce_fn(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "rir_lane_sum",
            ReduceOp::Max => "rir_lane_max",
        }
    }

    fn lane_scan_fn(op: ReduceOp) -> Option<&'static str> {
        Some(match op {
            ReduceOp::Sum => "rir_lane_exclusive_sum",
            ReduceOp::Max => "rir_lane_exclusive_max",
        })
    }

    fn lut(k: &LoopKernel, table: LutId) -> String {
        artifact_lut(k, table)
    }

    /// The prelude's comparisons take a scalar on either side, so a mixed
    /// operand needs no broadcast written at the call site.
    fn vector_cmp(op: CmpOp) -> Option<VectorCmp> {
        Some(VectorCmp {
            name: match op {
                CmpOp::Gt => "rir_vgt",
                CmpOp::Ge => "rir_vge",
                CmpOp::Lt => "rir_vlt",
                CmpOp::Le => "rir_vle",
                CmpOp::Eq => "rir_veq",
                CmpOp::Ne => "rir_vne",
            },
            splat_operands: false,
        })
    }
}

// ---------------------------------------------------------------------------

/// Metal Shading Language.
#[derive(Clone, Copy, Debug)]
pub struct Metal;

impl Dialect for Metal {
    const BACKEND: &'static str = "metal";
    const UINT: &'static str = "uint";
    const PARAMS: &'static str = "pc.";
    const ZERO: &'static str = "0.0f";
    const BARRIER: &'static str = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    const WG_LANE_INDEX: &'static str = "threadgroup_lane_id";
    const SELECT: &'static str = "select";

    fn axis_builtin(level: HwLevel) -> Option<Builtin> {
        Some(match level {
            HwLevel::Grid(0) => Builtin::atom("group_id.x"),
            HwLevel::Grid(1) => Builtin::atom("group_id.y"),
            HwLevel::Grid(_) => Builtin::atom("group_id.z"),
            HwLevel::Global(0) => Builtin::atom("global_id.x"),
            HwLevel::Global(1) => Builtin::atom("global_id.y"),
            HwLevel::Global(_) => Builtin::atom("global_id.z"),
            HwLevel::Lane => return None,
        })
    }

    /// The entrypoint declares one of the two, never both: a kernel using
    /// threadgroup storage takes `thread_index_in_threadgroup`, one that does
    /// not takes `thread_index_in_simdgroup` (`metal/mod.rs`).
    fn lane_builtin(k: &LoopKernel) -> &'static str {
        if k.uses_shared() {
            "threadgroup_lane_id"
        } else {
            "simd_lane_id"
        }
    }

    fn local_index(_k: &LoopKernel) -> &'static str {
        "threadgroup_lane_id"
    }

    fn float_literal(c: f32) -> String {
        format!("{c:?}")
    }

    fn identity(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "0.0f",
            ReduceOp::Max => "as_type<float>(0xff800000u)",
        }
    }

    fn combine(op: ReduceOp, a: &str, b: &str) -> String {
        match op {
            ReduceOp::Sum => format!("{a} + {b}"),
            ReduceOp::Max => format!("max({a}, {b})"),
        }
    }

    /// `max` is width-polymorphic in MSL, so unlike CUDA there is nothing for
    /// the width to select between.
    fn combine_at(op: ReduceOp, _width: Option<u32>, a: &str, b: &str) -> String {
        Self::combine(op, a, b)
    }

    fn decl_ty(kind: VarKind) -> String {
        match kind {
            VarKind::Idx => "uint".to_string(),
            VarKind::F32 => "float".to_string(),
            VarKind::Bool => "bool".to_string(),
            VarKind::Vec(w) => format!("float{w}"),
            VarKind::VecBool(w) => format!("bool{w}"),
        }
    }

    fn vec_ctor(width: u32, comps: &[String]) -> String {
        format!("float{width}({})", comps.join(", "))
    }

    fn splat(kind: VarKind, width: u32, text: &str) -> String {
        match kind {
            VarKind::Vec(_) | VarKind::VecBool(_) => text.to_string(),
            VarKind::Bool => format!("bool{width}({text})"),
            _ => format!("float{width}({text})"),
        }
    }

    fn component(kind: VarKind, text: &str, c: usize) -> String {
        match kind {
            VarKind::Vec(_) => format!("{text}[{c}]"),
            _ => text.to_string(),
        }
    }

    /// `precise::tanh` and not `tanh`: the native Metal GELU uses it, and the
    /// fast variant differs enough on a wide input to move an NMSE.
    fn unary_fn(f: UnaryFn, _width: Option<u32>) -> &'static str {
        match f {
            UnaryFn::Sqrt => "sqrt",
            UnaryFn::Exp => "exp",
            UnaryFn::Tanh => "precise::tanh",
        }
    }

    fn lane_reduce_fn(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "simd_sum",
            ReduceOp::Max => "simd_max",
        }
    }

    fn lane_scan_fn(op: ReduceOp) -> Option<&'static str> {
        match op {
            ReduceOp::Sum => Some("simd_prefix_exclusive_sum"),
            ReduceOp::Max => None,
        }
    }

    fn lut(k: &LoopKernel, table: LutId) -> String {
        artifact_lut(k, table)
    }

    /// MSL broadcasts a scalar across its comparison operators, so the scalar
    /// spelling is already the vector one and there is nothing to name.
    fn vector_cmp(_op: CmpOp) -> Option<VectorCmp> {
        None
    }
}

// ---------------------------------------------------------------------------

/// Compute GLSL.
#[derive(Clone, Copy, Debug)]
pub struct Vulkan;

impl Dialect for Vulkan {
    const BACKEND: &'static str = "vulkan";
    const UINT: &'static str = "uint";
    const PARAMS: &'static str = "pc.";
    const ZERO: &'static str = "0.0";
    const BARRIER: &'static str = "barrier();";
    const WG_LANE_INDEX: &'static str = "gl_LocalInvocationID.x";
    const SELECT: &'static str = "mix";

    fn axis_builtin(level: HwLevel) -> Option<Builtin> {
        Some(match level {
            HwLevel::Grid(0) => Builtin::atom("gl_WorkGroupID.x"),
            HwLevel::Grid(1) => Builtin::atom("gl_WorkGroupID.y"),
            HwLevel::Grid(_) => Builtin::atom("gl_WorkGroupID.z"),
            HwLevel::Global(0) => Builtin::atom("gl_GlobalInvocationID.x"),
            HwLevel::Global(1) => Builtin::atom("gl_GlobalInvocationID.y"),
            HwLevel::Global(_) => Builtin::atom("gl_GlobalInvocationID.z"),
            HwLevel::Lane => return None,
        })
    }

    fn lane_builtin(_k: &LoopKernel) -> &'static str {
        "gl_LocalInvocationID.x"
    }

    fn local_index(_k: &LoopKernel) -> &'static str {
        "gl_LocalInvocationIndex"
    }

    fn float_literal(c: f32) -> String {
        format!("{c:?}")
    }

    fn identity(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "0.0",
            ReduceOp::Max => "uintBitsToFloat(0xFF800000u)",
        }
    }

    fn combine(op: ReduceOp, a: &str, b: &str) -> String {
        match op {
            ReduceOp::Sum => format!("{a} + {b}"),
            ReduceOp::Max => format!("max({a}, {b})"),
        }
    }

    /// `max` is width-polymorphic in GLSL, so unlike CUDA there is nothing for
    /// the width to select between.
    fn combine_at(op: ReduceOp, _width: Option<u32>, a: &str, b: &str) -> String {
        Self::combine(op, a, b)
    }

    fn decl_ty(kind: VarKind) -> String {
        match kind {
            VarKind::Idx => "uint".to_string(),
            VarKind::F32 => "float".to_string(),
            VarKind::Bool => "bool".to_string(),
            VarKind::Vec(w) => format!("vec{w}"),
            VarKind::VecBool(w) => format!("bvec{w}"),
        }
    }

    fn vec_ctor(width: u32, comps: &[String]) -> String {
        format!("vec{width}({})", comps.join(", "))
    }

    fn splat(kind: VarKind, width: u32, text: &str) -> String {
        match kind {
            VarKind::Vec(_) | VarKind::VecBool(_) => text.to_string(),
            VarKind::Bool => format!("bvec{width}({text})"),
            _ => format!("vec{width}({text})"),
        }
    }

    fn component(kind: VarKind, text: &str, c: usize) -> String {
        const LANES: [&str; 4] = ["x", "y", "z", "w"];
        match kind {
            VarKind::Vec(_) => format!("{text}.{}", LANES[c]),
            _ => text.to_string(),
        }
    }

    fn unary_fn(f: UnaryFn, _width: Option<u32>) -> &'static str {
        match f {
            UnaryFn::Sqrt => "sqrt",
            UnaryFn::Exp => "exp",
            UnaryFn::Tanh => "tanh",
        }
    }

    fn lane_reduce_fn(op: ReduceOp) -> &'static str {
        match op {
            ReduceOp::Sum => "subgroupAdd",
            ReduceOp::Max => "subgroupMax",
        }
    }

    fn lane_scan_fn(op: ReduceOp) -> Option<&'static str> {
        Some(match op {
            ReduceOp::Sum => "subgroupExclusiveAdd",
            ReduceOp::Max => "subgroupExclusiveMax",
        })
    }

    fn lut(_k: &LoopKernel, table: LutId) -> String {
        // One SPIR-V module per variant, so the plain name is unambiguous,
        // unlike Metal, whose variants share one translation unit.
        table.symbol().to_string()
    }

    /// GLSL's vector comparisons are named functions and they take both
    /// operands at the same width.
    fn vector_cmp(op: CmpOp) -> Option<VectorCmp> {
        Some(VectorCmp {
            name: match op {
                CmpOp::Gt => "greaterThan",
                CmpOp::Ge => "greaterThanEqual",
                CmpOp::Lt => "lessThan",
                CmpOp::Le => "lessThanEqual",
                CmpOp::Eq => "equal",
                CmpOp::Ne => "notEqual",
            },
            splat_operands: true,
        })
    }
}
