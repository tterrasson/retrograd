//! Semantic IR: a pure tensor SSA graph.
//!
//! Hard rule: no `threadIdx`, `gl_WorkGroupID`, or equivalent exists at this
//! level. Values are functions over a logical index space; hardware mapping
//! is the sole responsibility of the schedule and lowering (`rir-lower`).

use crate::layout::TensorType;
use crate::types::{DType, ScalarType};

/// Identifiers into a kernel's four tables. Each is minted through
/// `IrId::at`, the single checked conversion of `crate::ids` - never by
/// casting a length at the call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ArgId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParamId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AxisId(pub u32);

crate::impl_ir_id!(ValueId, "value");
crate::impl_ir_id!(ArgId, "arg");
crate::impl_ir_id!(ParamId, "param");
crate::impl_ir_id!(AxisId, "axis");

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: String,
    pub ty: TensorType,
    pub access: Access,
}

#[derive(Clone, Debug)]
pub struct ParamDecl {
    pub name: String,
    pub ty: ScalarType,
}

/// Extent of a logical axis: dimension `dim` of an argument (ggml convention,
/// with 0 innermost), resolved from `ggml_tensor.ne[]` at dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Extent {
    Dim { arg: ArgId, dim: usize },
}

#[derive(Clone, Debug)]
pub struct AxisDecl {
    pub name: String,
    pub extent: Extent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    Sum,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOp {
    Sum,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanDirection {
    Forward,
    Backward,
}

impl ScanDirection {
    pub fn flipped(self) -> Self {
        match self {
            ScanDirection::Forward => ScanDirection::Backward,
            ScanDirection::Backward => ScanDirection::Forward,
        }
    }
}

/// Reduction reassociation semantics. A schedule that violates
/// these semantics is a compilation error in `rir-lower`, not a test failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionSemantics {
    /// The schedule may reassociate freely.
    Associative,
    /// Bit-for-bit identical results across runs on a given device.
    Deterministic,
    /// Strictly sequential accumulation order.
    ExactOrder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// SSA graph operations. Memory accesses are logical (symbolic indices, not
/// addresses); lowering resolves stride algebra. Dtype conversions are always
/// explicit operations such as `Cast` and `Dequant`, never implicit ones.
#[derive(Clone, Debug)]
pub enum Op {
    ConstF32(f32),
    /// Current position on a logical axis.
    Index(AxisId),
    /// **Extent** of a logical axis as an F32 value - not a position on it.
    ///
    /// Every backend already carries axis extents (the `n_<axis>` push
    /// constant a loop bound is printed from), so this exposes a datum the
    /// emitted code holds rather than adding one. It is what a mean costs:
    /// `rms_norm_back` divides by the row length, and without it the only way
    /// to obtain `N` would be to reduce a constant `1.0` over the axis - a
    /// second collective computing a number the shader already knows.
    ///
    /// Uniform along its own axis, so it is not a `Read`-like value: it never
    /// makes an expression axis-dependent.
    AxisExtent(AxisId),
    /// Uniform scalar parameter (epsilon, scale, and so on).
    Param(ParamId),

    /// **Index arithmetic**: `index` folded into the extent of `over`.
    /// Its value is `index % extent(over)`.
    ///
    /// It is a node of the *semantic graph* and not a property of an access,
    /// and the choice is `Dequant`'s precedent: the Loop IR guesses nothing, so
    /// a repeated read must be visible where the kernel is written rather than
    /// inferred from two extents that happen to divide. Three consequences all
    /// follow from that one decision - `arg_axes` maps the folded dimension to
    /// `over` (so a dispatcher fills that extent like any other), the axis
    /// agreement of `supports_op` turns into a *divisibility* on exactly those
    /// dimensions, and nothing changes for a kernel that does not use it.
    ///
    /// `over` is an axis declared on the repeated argument's own dimension, so
    /// its extent is that argument's `ne[d]`. It is not a loop axis: no `Index`
    /// names it, and lowering therefore never opens a loop for it.
    RepeatIndex {
        index: ValueId,
        over: AxisId,
    },

    /// `idx[d]` indexes ggml dimension `d` (0 is innermost).
    Read {
        tensor: ArgId,
        idx: Vec<ValueId>,
    },
    Write {
        tensor: ArgId,
        idx: Vec<ValueId>,
        value: ValueId,
    },

    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Sqrt(ValueId),
    Exp(ValueId),
    /// Hyperbolic tangent.
    ///
    /// Written as an operation rather than composed from `Exp` because the
    /// three emitters have it natively and ggml's own `tanhf` is what the
    /// parity is measured against: `1 - 2/(e^{2x}+1)` is the same function and
    /// not the same floating-point result, and a member of the unary family
    /// whose only difference from the native kernel is a rewritten identity is
    /// a difference nobody asked for.
    Tanh(ValueId),
    Cmp {
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Select {
        cond: ValueId,
        t: ValueId,
        f: ValueId,
    },

    /// Explicit dequantization of a `Read` from a quantized tensor. Lowering
    /// always fuses it into its consumer, so no dequantized tensor is ever
    /// materialized.
    Dequant {
        value: ValueId,
        from: crate::quant_table::QuantType,
    },

    Reduce {
        op: ReduceOp,
        axis: AxisId,
        value: ValueId,
        semantics: ReductionSemantics,
    },

    /// Inclusive scan along an axis (`Forward` prefixes or `Backward`
    /// suffixes). For associative sum, the backward is the opposite scan.
    /// V1 schedules execute scans sequentially to preserve exact order.
    Scan {
        op: ScanOp,
        axis: AxisId,
        dir: ScanDirection,
        value: ValueId,
    },
}

/// Constraints checked by `supports_op` against the actual tensor description
/// at dispatch, never by operation-name recognition alone.
#[derive(Clone, Debug)]
pub enum Constraint {
    DType {
        arg: ArgId,
        allowed: Vec<DType>,
    },
    /// **Maximum** effective ggml rank, not an equality: `TensorDesc::rank`
    /// drops trailing extents of 1, so a `[8,1,1,1]` tensor legitimately feeds
    /// a rank-2 kernel. The manifest publishes it as `rank(x)<=N` for that
    /// reason.
    Rank {
        arg: ArgId,
        max: usize,
    },
    Contiguous {
        arg: ArgId,
    },
}

/// A kernel: its SSA graph, arguments, axes, and contract.
///
/// **Read-only from outside this crate.** The
/// fields are `pub(crate)`, so `KernelBuilder` is now what `builder.rs` always
/// claimed to be - the only way to obtain one - and the accessors below are the
/// only way to look inside. Nothing here validates anything: what the privacy
/// buys is that a `Kernel` a caller holds is a `Kernel` the builder made, and
/// therefore one `validate` has seen. The witness of that is
/// [`ValidatedKernel`](crate::ValidatedKernel), which is what `rir_lower::lower` takes.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub(crate) name: String,
    pub(crate) args: Vec<Arg>,
    pub(crate) params: Vec<ParamDecl>,
    pub(crate) axes: Vec<AxisDecl>,
    /// `ops[i]` defines `ValueId(i)`; operands always precede their users.
    pub(crate) ops: Vec<Op>,
    pub(crate) constraints: Vec<Constraint>,
}

/// Whether a value depends on `axis` without passing through a `Reduce`.
/// A reduction result is uniform along its reduced axis, whereas a scan is
/// not. Lowering uses this to place writes; autodiff uses it to transpose
/// broadcasts into reductions and vice versa.
pub fn depends_on_axis(k: &Kernel, v: ValueId, axis: AxisId) -> bool {
    match &k.ops[v.0 as usize] {
        Op::Index(a) => *a == axis,
        // The fold is over an extent, not a position: what it depends on is the
        // axis its *index* walks, exactly like the index itself.
        Op::RepeatIndex { index, .. } => depends_on_axis(k, *index, axis),
        // An extent is a property of the axis, not a position on it: it is the
        // same value at every point, so it never makes a value axis-dependent.
        Op::ConstF32(_) | Op::Param(_) | Op::AxisExtent(_) => false,
        Op::Reduce { .. } => false,
        Op::Scan { axis: a, value, .. } => *a == axis || depends_on_axis(k, *value, axis),
        Op::Read { idx, .. } => idx.iter().any(|&i| depends_on_axis(k, i, axis)),
        Op::Dequant { value, .. } => depends_on_axis(k, *value, axis),
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
            depends_on_axis(k, *a, axis) || depends_on_axis(k, *b, axis)
        }
        Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) => depends_on_axis(k, *a, axis),
        Op::Cmp { lhs, rhs, .. } => {
            depends_on_axis(k, *lhs, axis) || depends_on_axis(k, *rhs, axis)
        }
        Op::Select { cond, t, f } => {
            depends_on_axis(k, *cond, axis)
                || depends_on_axis(k, *t, axis)
                || depends_on_axis(k, *f, axis)
        }
        Op::Write { .. } => unreachable!("Write is not a value"),
    }
}

/// For each argument, the logical axis indexing each of its ggml dimensions,
/// read off the kernel's own accesses. `None` where no access indexes that
/// dimension, or where two accesses disagree on it.
///
/// This is the mapping the rest of the pipeline needs to relate an argument's
/// **extents** to axis extents: `supports_op` uses it to check that two
/// arguments sharing an axis really have the same `ne`, and the manifest
/// publishes it so the runtime can bound the byte range a dispatch addresses.
/// Derived once here rather than restated per kernel.
pub fn arg_axes(k: &Kernel) -> Vec<Vec<Option<AxisId>>> {
    let mut out: Vec<Vec<Option<AxisId>>> = k.args.iter().map(|a| vec![None; a.ty.rank]).collect();
    let mut conflict: Vec<Vec<bool>> = k.args.iter().map(|a| vec![false; a.ty.rank]).collect();

    for op in &k.ops {
        let (tensor, idx) = match op {
            Op::Read { tensor, idx } => (*tensor, idx),
            Op::Write { tensor, idx, .. } => (*tensor, idx),
            _ => continue,
        };
        let a = tensor.0 as usize;
        for (d, &iv) in idx.iter().enumerate() {
            if a >= out.len() || d >= out[a].len() {
                continue;
            }
            // Only a direct `Index` names an axis; a computed subscript (a
            // gather) leaves the dimension unmapped rather than guessing.
            //
            // A `RepeatIndex` is the one computed subscript that *does* name
            // one, and it names `over` rather than the axis it folds: that
            // dimension's extent is what the shader divides by, so it is what a
            // dispatcher must fill and what the contract must check.
            let axis = match k.ops[iv.0 as usize] {
                Op::Index(axis) => axis,
                Op::RepeatIndex { over, .. } => over,
                _ => continue,
            };
            match out[a][d] {
                None if !conflict[a][d] => out[a][d] = Some(axis),
                Some(prev) if prev != axis => {
                    out[a][d] = None;
                    conflict[a][d] = true;
                }
                _ => {}
            }
        }
    }
    out
}

impl Kernel {
    /// Name of this kernel, which is the identity every table joins on.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Arguments in declaration order; `ArgId(i)` indexes this slice.
    pub fn args(&self) -> &[Arg] {
        &self.args
    }

    /// Uniform scalar parameters in declaration order; `ParamId(i)` indexes
    /// this slice.
    pub fn params(&self) -> &[ParamDecl] {
        &self.params
    }

    /// Logical axes in declaration order; `AxisId(i)` indexes this slice.
    pub fn axes(&self) -> &[AxisDecl] {
        &self.axes
    }

    /// The SSA graph: `ops[i]` defines `ValueId(i)`, and operands always
    /// precede their users.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The contract `supports_op` evaluates against a real tensor.
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn op(&self, v: ValueId) -> &Op {
        &self.ops[v.0 as usize]
    }

    pub fn arg(&self, a: ArgId) -> &Arg {
        &self.args[a.0 as usize]
    }

    pub fn axis(&self, a: AxisId) -> &AxisDecl {
        &self.axes[a.0 as usize]
    }
}
