//! Loop IR: a backend-agnostic imperative loop nest.
//!
//! Lowering has already selected the reduction strategy, phase boundaries,
//! logical-axis mapping to loops or lanes, address algebra (including
//! quantized blocking), and register types. Emitters only provide syntax.
//!
//! Shared memory is explicit: an array is **declared** by the kernel
//! (`LoopKernel::shared`), written and read by `Stmt::StoreShared` /
//! `Stmt::LoadShared`, and the lanes that share it meet at `Stmt::Barrier`.
//! What an emitter prints is therefore a statement at a time; what it may not
//! do is *expand an algorithm* - a series of levels and a plan of barriers.
//! `blelloch` and `tiled_scan` lower to these explicit statements. The one
//! collective still expanded by the emitters is `WorkgroupReduce`, with one
//! barrier per level around the group of accumulators chosen by lowering.
//!
//! Vector accesses are explicit too: a `Load`/`Store` carries a `width`, and a
//! register holding such a value is a `VarKind::Vec`. The **width is chosen by
//! lowering**, from the schedule's `vector_width`; emitters only print the
//! vector type of their language. A parallel axis whose invocation covers
//! `vector` consecutive indices says so on `Stmt::Parallel`, and the tail of a
//! row that is not a multiple of that width is an explicit `Stmt::VecTail`
//! branch - never an out-of-bounds vector nobody guards.
//!
//! Shared memory is also used for **staging** (`Stmt::StageTiles`): a
//! workgroup loads a tile of an argument once,
//! cooperatively, and every invocation of that workgroup then consumes it from
//! shared memory. The tile geometry is carried by the statement, and the
//! storage it needs is declared with every other shared array in
//! `LoopKernel::shared`.

use std::collections::BTreeSet;

use rir_core::{Arg, AxisDecl, AxisId, Constraint, ParamDecl, ReductionSemantics};

use crate::schedule::Schedule;

pub mod expr;
pub mod stmt;

#[cfg(test)]
mod tests;

pub use expr::LExpr;
pub use stmt::{GroupRed, Inst, Stmt, TileStage, child_blocks_of, chunk_range};

/// The multiplier and shift of an unsigned division by `d`, as
/// `init_fastdiv_values` computes them in `ggml-cuda/common.cuh`.
///
/// Written here and not in an emitter because three consumers have to agree on
/// it to the bit: the host that fills the constant buffer, the shader that
/// divides, and the oracle that judges the result. Deliberately the *same*
/// formula the native kernel uses rather than a second derivation of the same
/// theorem - a flattening measured against `k_bin_bcast_unravel` must pay the
/// arithmetic it pays, not a variant of it.
///
/// `L = ceil(log2 d)`, `mp = floor(2^32 · (2^L − d) / d) + 1`, and the quotient
/// is `(umulhi(n, mp) + n) >> L`. The addition is done in 32 bits, which is
/// exact while `n < 2^31` - the bound `ggml_rir_flat_total` checks before a
/// flattened variant is selected, and the reason it checks it.
///
/// # Panics
///
/// On `d == 0`: a divisor of zero is an extent of zero, which lowering and the
/// contract both reject long before this. Above `2^31`, `L` reaches 32 and the
/// shift count becomes invalid; the flattened-total contract rejects that
/// range before this helper is called.
pub fn fastdiv_magic(d: u32) -> (u32, u32) {
    assert!(d != 0, "fastdiv by zero");
    assert!(
        d <= 1 << 31,
        "fastdiv by {d}: above the 2^31 the contract admits"
    );
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;
    (mp, l)
}

/// The quotient the emitters print, evaluated exactly as they print it. The
/// oracle uses plain division; this exists for the test that proves the two
/// agree over the range the contract admits.
pub fn fastdiv(n: u32, mp: u32, shift: u32) -> u32 {
    let hi = (((n as u64) * (mp as u64)) >> 32) as u32;
    hi.wrapping_add(n) >> shift
}

/// A Loop IR register. Minted through `IrId::at` like the semantic IR's own
/// identifiers, so the two crates share one checked conversion instead of one
/// cast each.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VarId(pub u32);

rir_core::impl_ir_id!(VarId, "loop variable");

/// Register type. Emitters map it to a source type (`usize`/`uint`,
/// `f32`/`float`, or `bool`); the interpreter uses it to select a register bank.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VarKind {
    /// Index or unsigned integer (loop variables and quantized blocks).
    Idx,
    F32,
    Bool,
    /// `w` F32 lanes in one register (`float4`, `vec4`). Produced only by the
    /// vector widening of lowering, and only for values that flow from a
    /// vector `Load` to a vector `Store` through componentwise arithmetic.
    Vec(u32),
    /// `w` predicates in one register (`bool4`, `bvec4`) - the widened form of
    /// `Bool`.
    ///
    /// It exists because the unary family is written with `Cmp` and `Select`
    /// and its lowering is vectorized: a family whose members are `relu`,
    /// `step` and `sgn` has a comparison per element, and reading one `float`
    /// where the native kernel reads four is the measured deficit this
    /// closes. Both shading languages have the type and the two operations that
    /// go with it; what they do *not* share is the spelling, which is exactly
    /// what an emitter is for.
    VecBool(u32),
}

/// Hardware level to which a parallel axis is mapped. The schedule
/// (`ParallelMapping`) and lowering make this choice; emitters only print the
/// corresponding builtin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HwLevel {
    /// One **workgroup** per axis index (`gl_WorkGroupID`), with the entire
    /// workgroup cooperating on that index (lane-based strategies).
    Grid(u8),
    /// One **invocation** per axis index (`gl_GlobalInvocationID`); the
    /// workgroup covers `block[d]` consecutive indices.
    Global(u8),
    Lane,
}

/// Memory access type. F16 exists only in memory; loading it always produces
/// an F32 register through an explicit contract conversion.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemType {
    F32,
    F16,
    /// Signed byte: `q8_0` payloads.
    I8,
    /// Unsigned byte, loaded into an `Idx` register. Sub-byte formats need the
    /// raw bits before any sign extension - a nibble extracted from a sign
    /// extended byte is wrong for every value above 0x7F.
    U8,
}

impl MemType {
    /// Element size in bytes, used to scale addresses of this type.
    pub fn size_bytes(self) -> u32 {
        match self {
            MemType::F32 => 4,
            MemType::F16 => 2,
            MemType::I8 | MemType::U8 => 1,
        }
    }

    /// View suffix for backends that declare one typed buffer per access type.
    /// F32 has no suffix because it is the default.
    pub fn view_suffix(self) -> &'static str {
        match self {
            MemType::F32 => "",
            MemType::F16 => "_f16",
            MemType::I8 => "_i8",
            MemType::U8 => "_u8",
        }
    }
}

/// Term in a byte-address expression. An access address is the sum of its
/// terms; stride algebra is resolved here, not in emitters.
///
/// `Eq + Hash` because two accesses sharing an address base have to be
/// *recognised* as sharing it, by `crate::hoist` and `crate::cse` alike. A term
/// is a register and two integers, so the derived equality is the structural one
/// those passes need, so neither pass needs to format a key into a `String`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AddrTerm {
    /// `var · nb[dim]`: runtime stride of the accessed argument.
    VarNb { var: VarId, dim: usize },
    /// `var · c`: descriptor constant, such as an intra-block offset.
    VarConst { var: VarId, c: u32 },
    /// Constant byte offset.
    Const(u32),
}

/// A lowered kernel: its loop nest plus everything needed by an emitter or
/// the interpreter (arguments, parameters, axes, and contract).
#[derive(Clone, Debug)]
pub struct LoopKernel {
    pub name: String,
    pub args: Vec<Arg>,
    pub params: Vec<ParamDecl>,
    pub axes: Vec<AxisDecl>,
    /// `arg_axes[a][d]` is the logical axis indexing dimension `d` of argument
    /// `a` (`rir_core::arg_axes`), carried over from the semantic IR because
    /// the manifest publishes it: it is what lets a caller bound the byte range
    /// a dispatch addresses on each buffer.
    pub arg_axes: Vec<Vec<Option<AxisId>>>,
    /// `(index, over)` for each index fold the kernel performs:
    /// position on `index` is replayed modulo the
    /// extent of `over`. Carried over from the semantic IR for the same reason
    /// `arg_axes` is - the registry publishes it, so a dispatcher can require
    /// `extent(index) % extent(over) == 0` on exactly the dimensions the kernel
    /// folds, without replaying lowering.
    pub folds: Vec<(AxisId, AxisId)>,
    pub constraints: Vec<Constraint>,
    /// Kernel reduction semantics, used for the manifest's `determinism` field.
    pub reduction_semantics: Vec<ReductionSemantics>,
    /// `var_names[v]` is the unique name of register `VarId(v)`.
    pub var_names: Vec<String>,
    pub var_kinds: Vec<VarKind>,
    /// Shared arrays the kernel declares, in declaration order: the register
    /// naming each one and its length in elements.
    ///
    /// **Declared, not derived.** The kernel records each array's length once,
    /// because ordinary `StoreShared` statements do not carry that information.
    /// Emitters and the interpreter read the same list.
    pub shared: Vec<(VarId, u32)>,
    pub body: Vec<Stmt>,
    pub schedule: Schedule,
}

impl LoopKernel {
    /// Memory types actually accessed on each argument, in stable order. This
    /// queries the IR rather than making a policy decision. Backends that need
    /// typed buffers per access type and the manifest that derives required
    /// extensions consume **the same** result.
    ///
    /// A quantized argument yields two types (`F16` for the scale and `I8` for
    /// data bytes), reflecting lowering's fused loader at the memory boundary.
    pub fn mem_types(&self) -> Vec<BTreeSet<MemType>> {
        fn walk(stmts: &[Stmt], out: &mut [BTreeSet<MemType>]) {
            for s in stmts {
                match s {
                    Stmt::Load { arg, ty, .. } => {
                        out[arg.0 as usize].insert(*ty);
                    }
                    Stmt::Store { arg, ty, .. } => {
                        out[arg.0 as usize].insert(*ty);
                    }
                    // A staged tile is filled by its loader, so the buffer views
                    // it needs are exactly those a direct read would have
                    // declared - one F32 view for a plain operand, the scale
                    // and payload views of the fused decoder for a quantized
                    // one. Asserting F32 here instead would have declared the
                    // wrong view the day a tile staged anything else.
                    Stmt::StageTiles { tiles, body, .. } => {
                        for t in tiles {
                            walk(&t.load, out);
                        }
                        walk(body, out);
                    }
                    Stmt::Parallel { body, .. }
                    | Stmt::ParallelFlat { body, .. }
                    | Stmt::ParallelLane { body, .. }
                    | Stmt::For { body, .. }
                    | Stmt::ForStrided { body, .. }
                    | Stmt::ForChunk { body, .. }
                    | Stmt::ForConst { body, .. }
                    | Stmt::ForTiled { body, .. }
                    | Stmt::InBounds { body, .. }
                    | Stmt::If { body, .. }
                    | Stmt::LaneZero { body, .. } => walk(body, out),
                    Stmt::VecTail {
                        vec_body,
                        tail_body,
                        ..
                    } => {
                        walk(vec_body, out);
                        walk(tail_body, out);
                    }
                    Stmt::Barrier
                    | Stmt::Set { .. }
                    | Stmt::InitAcc { .. }
                    | Stmt::Accum { .. }
                    | Stmt::LaneReduce { .. }
                    | Stmt::LaneScan { .. }
                    | Stmt::WorkgroupReduce { .. }
                    | Stmt::LoadShared { .. }
                    | Stmt::StoreShared { .. }
                    | Stmt::Compute(_) => {}
                }
            }
        }
        let mut out = vec![BTreeSet::new(); self.args.len()];
        walk(&self.body, &mut out);
        out
    }

    /// Whether the body contains a lane collective (`LaneReduce`, `LaneScan`,
    /// or a hierarchical `WorkgroupReduce`).
    pub fn uses_subgroup(&self) -> bool {
        self.subgroup_width().is_some()
    }

    /// The **width** of the subgroup the body needs, or `None` when it uses no
    /// lane collective.
    ///
    /// Published as `min_subgroup`, and it stopped being derivable from the
    /// workgroup shape when `HierarchicalTree` arrived:
    /// until then every lane collective ran on a `[32, 1, 1]` block, so the
    /// block *was* the width, and a 256-lane hierarchical reduction needs a
    /// 32-lane subgroup and would have published 256 - a requirement no device
    /// meets, on a kernel that does not need it.
    ///
    /// Read from the nest, like every other published property: the widest
    /// collective the body carries, because the requirement is what the kernel
    /// as a whole owes the device.
    pub fn subgroup_width(&self) -> Option<u32> {
        fn walk(stmts: &[Stmt], lanes: u32) -> Option<u32> {
            stmts
                .iter()
                .filter_map(|s| match s {
                    Stmt::LaneReduce { .. } | Stmt::LaneScan { .. } => Some(lanes),
                    Stmt::WorkgroupReduce {
                        subgroup: Some(w), ..
                    } => Some(*w),
                    Stmt::ParallelLane { lanes, body, .. } => walk(body, *lanes),
                    Stmt::Parallel { body, .. }
                    | Stmt::ParallelFlat { body, .. }
                    | Stmt::For { body, .. }
                    | Stmt::ForStrided { body, .. }
                    | Stmt::ForChunk { body, .. }
                    | Stmt::ForConst { body, .. }
                    | Stmt::ForTiled { body, .. }
                    | Stmt::InBounds { body, .. }
                    | Stmt::If { body, .. }
                    | Stmt::StageTiles { body, .. }
                    | Stmt::LaneZero { body, .. } => walk(body, lanes),
                    Stmt::VecTail {
                        vec_body,
                        tail_body,
                        ..
                    } => walk(vec_body, lanes).max(walk(tail_body, lanes)),
                    _ => None,
                })
                .max()
        }
        // Zero is the width outside any `ParallelLane`, where no collective can
        // legitimately appear; a body that carried one there would publish it
        // and the device check would refuse the kernel rather than run it.
        walk(&self.body, 0)
    }

    /// `(index, over)` pairs, as the registry emitter asks for them.
    pub fn folds(&self) -> Vec<(AxisId, AxisId)> {
        self.folds.clone()
    }

    /// Constant tables the body indexes, in stable order. Same query shape as
    /// `shared_collectives` and `shared_tiles`, and for the same reason: what a
    /// backend declares in its preamble is **derived** from the statements it is
    /// about to print, never a list kept in step by hand beside the format
    /// table.
    pub fn luts(&self) -> Vec<rir_core::LutId> {
        fn walk(stmts: &[Stmt], out: &mut Vec<rir_core::LutId>) {
            for s in stmts {
                match s {
                    Stmt::Compute(Inst {
                        expr: LExpr::Lut { table, .. },
                        ..
                    }) => {
                        if !out.contains(table) {
                            out.push(*table);
                        }
                    }
                    Stmt::StageTiles { tiles, body, .. } => {
                        for t in tiles {
                            walk(&t.load, out);
                        }
                        walk(body, out);
                    }
                    Stmt::Parallel { body, .. }
                    | Stmt::ParallelFlat { body, .. }
                    | Stmt::ParallelLane { body, .. }
                    | Stmt::For { body, .. }
                    | Stmt::ForStrided { body, .. }
                    | Stmt::ForChunk { body, .. }
                    | Stmt::ForConst { body, .. }
                    | Stmt::ForTiled { body, .. }
                    | Stmt::InBounds { body, .. }
                    | Stmt::If { body, .. }
                    | Stmt::LaneZero { body, .. } => walk(body, out),
                    Stmt::VecTail {
                        vec_body,
                        tail_body,
                        ..
                    } => {
                        walk(vec_body, out);
                        walk(tail_body, out);
                    }
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.body, &mut out);
        out
    }

    pub fn uses_shared(&self) -> bool {
        !self.shared.is_empty()
    }

    /// Number of consecutive elements one invocation covers on the vectorized
    /// axis, or 1. Read **from the loop nest**, like `grid_axes`: the manifest
    /// and the registry publish what the shader does, not what was requested.
    pub fn vector_width(&self) -> u32 {
        fn walk(stmts: &[Stmt]) -> u32 {
            match stmts {
                [Stmt::Parallel { vector, body, .. }] => (*vector).max(walk(body)),
                [Stmt::ParallelFlat { vector, body, .. }] => (*vector).max(walk(body)),
                _ => 1,
            }
        }
        walk(&self.body).max(1)
    }

    /// The axes a flattened dispatch decomposes its linear index into, fastest
    /// first, with the number of indices one invocation covers on each
    /// Empty for every other lowering.
    ///
    /// Read **from the loop nest** for the reason `grid_axes` is: the constant
    /// buffer and the registry publish what the shader decomposes, not what a
    /// schedule asked for. It is the same `(axis, per_index)` shape the grid
    /// publishes, and deliberately so - the host computes one divisor per entry
    /// as `ceil(extent / per_index)`, which is the `ceil` a grid dimension
    /// already applies.
    pub fn flat_axes(&self) -> Vec<(AxisId, u32)> {
        match self.body.as_slice() {
            [Stmt::ParallelFlat { axes, vector, .. }] => axes
                .iter()
                .enumerate()
                .map(|(i, (_, ax))| (*ax, if i == 0 { (*vector).max(1) } else { 1 }))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Whether the flattened index is also the **address**: the shader
    /// performs no decomposition, because under the claim every access is
    /// `linear · width · elem_bytes`.
    ///
    /// Read from the loop nest, like `flat_axes` and `vector_width`, and for the
    /// same reason: what the manifest and the registry publish is what the
    /// shader does. It is *not* the complement of `flat_axes` - the host
    /// decomposes in both cases, because that is where the total comes from,
    /// so what this answers is narrower: whether the constant block has to carry
    /// the reciprocals, three `uint` per axis but the last, which a linear
    /// variant neither declares nor is ever filled.
    pub fn linear_addr(&self) -> bool {
        matches!(
            self.body.as_slice(),
            [Stmt::ParallelFlat {
                decompose: false,
                ..
            }]
        )
    }
}
