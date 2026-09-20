//! Loop IR interpreter: the stage-zero CPU oracle.
//!
//! Executes a `LoopKernel` on host buffers with arbitrary ggml byte strides,
//! without compiling generated code. This includes GPU-lowered kernels:
//! `ParallelLane` is simulated with per-lane register banks, and `LaneReduce`
//! runs in ascending lane order. Real `subgroupAdd` hardware may use a
//! different fixed topology, so device parity uses a tolerance rather than
//! bitwise equality.
//!
//! Typed registers use an integer bank for `Idx` and a floating-point bank for
//! F32 and Bool (encoded as 0.0/1.0). Quantized F16/I8 loads read byte views.

use rir_core::{Access, Extent};

use crate::loop_ir::*;

pub mod exec;
pub mod f16;
pub mod views;

#[cfg(test)]
mod tests;

pub use f16::{f16_to_f32, f32_to_f16};
pub use views::{BoundArg, TensorView, TensorViewBytes, TensorViewBytesMut, TensorViewMut};

use exec::exec_stmts;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum InterpError {
    #[error("{got} bound arguments, expected {expected}")]
    ArgCountMismatch { expected: usize, got: usize },
    #[error("argument #{arg} is bound with the wrong access direction")]
    AccessMismatch { arg: usize },
    #[error("{got} parameters supplied, expected {expected}")]
    ParamCountMismatch { expected: usize, got: usize },
    #[error("misaligned access at byte {byte_offset} of argument #{arg}")]
    MisalignedAccess { arg: usize, byte_offset: usize },
    #[error("byte {byte_offset} outside argument #{arg} buffer ({len_bytes} bytes)")]
    OutOfBounds {
        arg: usize,
        byte_offset: usize,
        len_bytes: usize,
    },
    /// A typed access is incompatible with the bound view, such as F16/I8 on F32.
    #[error("access type incompatible with argument #{arg} view")]
    BadAccessType { arg: usize },
    /// A lane/workgroup collective occurs outside the top level of a
    /// `ParallelLane`, violating a lowering invariant.
    #[error("lane collective outside a ParallelLane")]
    MisplacedCollective,
    /// A constant-table read past the table. The masks of a bit field make this
    /// unreachable by construction, which is exactly why it is an error and not
    /// a clamp: a saturating read would turn a wrong mask into plausible values.
    #[error("{}: index {index} outside table",.table.symbol())]
    LutOutOfRange {
        table: rir_core::LutId,
        index: usize,
    },
}

/// Register banks for one lane or scalar scope.
///
/// There is no vector bank. A `width`-wide body is executed once per component
/// with `vlane` set, and a widened access reads `vlane` elements further along
/// the contiguous axis - which is what a `float4` load *means*, spelled without
/// a second register file. The oracle stays scalar, and a disagreement with the
/// device can only be about the values, never about how they were packed.
#[derive(Clone)]
pub(crate) struct Regs {
    pub(crate) f: Vec<f32>,
    pub(crate) i: Vec<usize>,
    /// Component of the vector currently being evaluated, or 0.
    pub(crate) vlane: usize,
}

/// Shared memory, one array per declaring register (`LoopKernel::shared`), and
/// **genuinely shared**: registers are cloned per lane, this is not.
///
/// It is separate from per-lane registers because a scan tree can write a slot
/// in one lane and read it in another. Lowering emits the same explicit shared
/// accesses for the oracle and device, with barriers preserving their ordering.
pub(crate) type Shared = Vec<Vec<f32>>;

/// Executes `k` against `args` (one bound view per declared argument, in
/// declaration order) and `params` (one value per declared scalar parameter).
///
/// Checked before a single instruction runs: argument count and access
/// direction, parameter count, and - as each access happens - alignment,
/// bounds, and type compatibility with the bound view. A `LoopKernel` built by
/// `rir_lower::lower` cannot violate these on its own account, so a failure
/// here means the caller bound the wrong shape or the wrong dtype.
pub fn run(k: &LoopKernel, args: &mut [BoundArg], params: &[f32]) -> Result<(), InterpError> {
    if args.len() != k.args.len() {
        return Err(InterpError::ArgCountMismatch {
            expected: k.args.len(),
            got: args.len(),
        });
    }
    for (i, (decl, bound)) in k.args.iter().zip(args.iter()).enumerate() {
        let ok = matches!(
            (decl.access, bound),
            (Access::Read, BoundArg::In(_))
                | (Access::Read, BoundArg::InBytes(_))
                | (Access::Write, BoundArg::Out(_))
                | (Access::Write, BoundArg::OutBytes(_))
        );
        if !ok {
            return Err(InterpError::AccessMismatch { arg: i });
        }
    }
    if params.len() != k.params.len() {
        return Err(InterpError::ParamCountMismatch {
            expected: k.params.len(),
            got: params.len(),
        });
    }

    let n = k.var_names.len();
    let mut regs = Regs {
        f: vec![0f32; n],
        i: vec![0usize; n],
        vlane: 0,
    };
    // The storage the kernel **declares**, allocated exactly as the emitters
    // declare theirs: one list, three consumers.
    let mut shared: Shared = vec![Vec::new(); n];
    for (array, len) in &k.shared {
        shared[array.0 as usize] = vec![0f32; *len as usize];
    }
    exec_stmts(k, &k.body, &mut regs, &mut shared, args, params)
}

pub(crate) fn axis_extent(k: &LoopKernel, axis: rir_core::AxisId, args: &[BoundArg]) -> usize {
    match k.axes[axis.0 as usize].extent {
        Extent::Dim { arg, dim } => args[arg.0 as usize].shape()[dim],
    }
}

pub(crate) fn byte_addr(nb: &[usize; 4], addr: &[AddrTerm], regs: &Regs) -> usize {
    addr.iter()
        .map(|t| match t {
            AddrTerm::VarNb { var, dim } => regs.i[var.0 as usize] * nb[*dim],
            AddrTerm::VarConst { var, c } => regs.i[var.0 as usize] * (*c as usize),
            AddrTerm::Const(c) => *c as usize,
        })
        .sum()
}
