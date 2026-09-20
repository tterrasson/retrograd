//! `supports_op`: variant selection at dispatch.
//!
//! Selection never relies on an operation name alone. It checks kernel
//! constraints against the tensor's **actual shape and layout**: dtype, rank,
//! strides, and quantized-block divisibility. Rejections include a precise
//! reason, making the fallback chain observable instead of silent.

use crate::ids::IrId;

use crate::ir::{ArgId, AxisId, Constraint, Kernel, Op, arg_axes};
use crate::types::DType;

/// Concrete tensor description at dispatch: the subset of `ggml_tensor`
/// needed by `supports_op` (`ne[]` in logical elements, `nb[]` in bytes).
#[derive(Clone, Copy, Debug)]
pub struct TensorDesc {
    pub dtype: DType,
    pub ne: [usize; 4],
    pub nb: [usize; 4],
}

impl TensorDesc {
    /// Effective ggml rank: one plus the last dimension greater than 1,
    /// with a minimum of 1.
    pub fn rank(&self) -> usize {
        (0..4).rev().find(|&d| self.ne[d] > 1).map_or(1, |d| d + 1)
    }

    /// C contiguity as defined by `ggml_is_contiguous`: `nb[0]` equals
    /// `type_size`, and each stride is the preceding stride multiplied by its
    /// extent (in blocks for a quantized dimension).
    pub fn is_contiguous(&self) -> bool {
        let ts = self.dtype.size_bytes();
        let epu = self.dtype.elements_per_unit();
        if self.nb[0] != ts {
            return false;
        }
        let mut expect = ts * self.ne[0].div_ceil(epu);
        for d in 1..4 {
            if self.nb[d] != expect {
                return false;
            }
            expect *= self.ne[d];
        }
        true
    }
}

/// Why a compiled kernel variant refuses the tensors it was handed.
///
/// It is the `Err` of [`supports_op`], so it derives `thiserror::Error` and
/// carries its sentence in `#[error(…)]` - the rule `CLAUDE.md` states, and the
/// one this type was the last in the workspace to break: it had a
/// forty-five-line hand-written `Display` and no `std::error::Error` at all,
/// which made it an error type a caller could print but not propagate.
/// The wordings below are unchanged.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RejectReason {
    #[error("{got} tensors, expected {expected}")]
    ArgCount { expected: usize, got: usize },
    #[error("arg #{}: dtype {} outside {{{}}}",.arg.0,.got.name(), dtype_names(.expected))]
    DType {
        arg: ArgId,
        expected: Vec<DType>,
        got: DType,
    },
    #[error("arg #{}: rank {got} > {max}",.arg.0)]
    Rank { arg: ArgId, max: usize, got: usize },
    #[error("arg #{}: not contiguous",.arg.0)]
    NotContiguous { arg: ArgId },
    /// The quantized dimension is not divisible by the block size.
    #[error("arg #{}: ne[0]={ne0} not divisible by block {block}",.arg.0)]
    QuantBlock {
        arg: ArgId,
        ne0: usize,
        block: usize,
    },
    /// A repeated dimension whose extent does not divide the extent it is
    /// replayed under (`ggml_can_repeat`). Same category
    /// as the disagreement below on purpose: both are `shape`, and both say the
    /// shader would address past the end of one operand.
    #[error(
        "axis #{} with extent {extent} repeated over axis #{} with extent {divisor}: \
         not divisible",
        .axis.0,
        .over.0
    )]
    RepeatNotDivisible {
        axis: AxisId,
        over: AxisId,
        extent: usize,
        divisor: usize,
    },
    /// Two arguments indexed by the same logical axis disagree on its extent.
    /// The kernel reads that axis from one of them, so the other would be
    /// addressed out of its own bounds - the constraint `mat_mul_naive` needs
    /// between `a.ne[0]` and `b.ne[0]`, derived rather than restated.
    #[error("axis #{} with extent {expected}: arg #{} has ne[{dim}]={got}",.axis.0,.arg.0)]
    AxisExtent {
        axis: AxisId,
        arg: ArgId,
        dim: usize,
        expected: usize,
        got: usize,
    },
}

/// `{f16,f32}` - the dtype set a rejected argument was measured against.
fn dtype_names(dtypes: &[DType]) -> String {
    dtypes
        .iter()
        .map(|dtype| dtype.name())
        .collect::<Vec<_>>()
        .join(",")
}

/// Checks whether a compiled `kernel` variant accepts the tensors in `descs`,
/// listed in kernel argument order.
pub fn supports_op(kernel: &Kernel, descs: &[TensorDesc]) -> Result<(), RejectReason> {
    if descs.len() != kernel.args.len() {
        return Err(RejectReason::ArgCount {
            expected: kernel.args.len(),
            got: descs.len(),
        });
    }

    // Each argument's declared dtype is an implicit contract.
    for (i, (arg, desc)) in kernel.args.iter().zip(descs).enumerate() {
        let aid = ArgId::at(i);
        if desc.dtype != arg.ty.dtype {
            return Err(RejectReason::DType {
                arg: aid,
                expected: vec![arg.ty.dtype],
                got: desc.dtype,
            });
        }
        if desc.rank() > arg.ty.rank {
            return Err(RejectReason::Rank {
                arg: aid,
                max: arg.ty.rank,
                got: desc.rank(),
            });
        }
        if let DType::Quant(q) = desc.dtype {
            let block = q.desc().block_elements as usize;
            if desc.ne[0] % block != 0 {
                return Err(RejectReason::QuantBlock {
                    arg: aid,
                    ne0: desc.ne[0],
                    block,
                });
            }
        }
    }

    // Axis agreement. An axis extent comes from **one** argument's dimension
    // (`AxisDecl::extent`), and that single value becomes `n_<axis>` in the
    // dispatch; every other argument indexed by the same axis is addressed
    // with it. They must therefore agree, or the shader walks past the end of
    // one of them. This is checked from the kernel's own accesses, so no
    // kernel has to remember to declare it.
    let mapping = arg_axes(kernel);
    for (ai, dims) in mapping.iter().enumerate() {
        for (d, slot) in dims.iter().enumerate() {
            let Some(axis) = *slot else { continue };
            let crate::ir::Extent::Dim {
                arg: src,
                dim: src_dim,
            } = kernel.axes[axis.0 as usize].extent;
            let expected = descs[src.0 as usize].ne[src_dim];
            let got = descs[ai].ne[d];
            if got != expected {
                return Err(RejectReason::AxisExtent {
                    axis,
                    arg: ArgId::at(ai),
                    dim: d,
                    expected,
                    got,
                });
            }
        }
    }

    // Index arithmetic: a folded dimension must *divide* the one it is replayed
    // under, which is `ggml_can_repeat`. It is derived from the graph's own
    // `RepeatIndex` nodes rather than declared per kernel - the same reading as
    // the axis agreement above, which is why a failure lands in the same
    // `shape` category.
    for op in &kernel.ops {
        let Op::RepeatIndex { index, over } = op else {
            continue;
        };
        let Op::Index(axis) = kernel.ops[index.0 as usize] else {
            continue;
        };
        let extent = |a: AxisId| {
            let crate::ir::Extent::Dim { arg, dim } = kernel.axes[a.0 as usize].extent;
            descs[arg.0 as usize].ne[dim]
        };
        let (e, d) = (extent(axis), extent(*over));
        if d == 0 || e % d != 0 {
            return Err(RejectReason::RepeatNotDivisible {
                axis,
                over: *over,
                extent: e,
                divisor: d,
            });
        }
    }

    for c in &kernel.constraints {
        match c {
            Constraint::DType { arg, allowed } => {
                let got = descs[arg.0 as usize].dtype;
                if !allowed.contains(&got) {
                    return Err(RejectReason::DType {
                        arg: *arg,
                        expected: allowed.clone(),
                        got,
                    });
                }
            }
            // `max`, not an equality: `rank()` drops trailing extents of 1, so
            // a single-row `[8,1,1,1]` tensor is a legitimate input to a
            // rank-2 kernel. See `Constraint::Rank`.
            Constraint::Rank { arg, max } => {
                let got = descs[arg.0 as usize].rank();
                if got > *max {
                    return Err(RejectReason::Rank {
                        arg: *arg,
                        max: *max,
                        got,
                    });
                }
            }
            Constraint::Contiguous { arg } => {
                if !descs[arg.0 as usize].is_contiguous() {
                    return Err(RejectReason::NotContiguous { arg: *arg });
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::KernelBuilder;
    use crate::ir::Extent;
    use crate::layout::TensorType;
    use crate::quant_table::QuantType;

    fn kernel_f32() -> crate::ValidatedKernel {
        let mut k = KernelBuilder::new("copy");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        k.write(y, &[col, row], v);
        k.constrain(Constraint::Rank { arg: x, max: 2 });
        k.constrain(Constraint::Contiguous { arg: x });
        k.finish().unwrap()
    }

    fn f32_desc(ne: [usize; 4], nb: [usize; 4]) -> TensorDesc {
        TensorDesc {
            dtype: DType::F32,
            ne,
            nb,
        }
    }

    #[test]
    fn accepts_a_conforming_shape() {
        let k = kernel_f32();
        let d = f32_desc([8, 3, 1, 1], [4, 32, 96, 96]);
        assert_eq!(supports_op(&k, &[d, d]), Ok(()));
    }

    #[test]
    fn refuses_the_wrong_dtype() {
        let k = kernel_f32();
        let d = f32_desc([8, 3, 1, 1], [4, 32, 96, 96]);
        let bad = TensorDesc {
            dtype: DType::F16,
            ..d
        };
        assert!(matches!(
            supports_op(&k, &[bad, d]),
            Err(RejectReason::DType { .. })
        ));
    }

    #[test]
    fn refuses_a_rank_that_is_too_large() {
        let k = kernel_f32();
        let d = f32_desc([8, 3, 1, 1], [4, 32, 96, 96]);
        let r3 = f32_desc([8, 3, 2, 1], [4, 32, 96, 192]);
        assert!(matches!(
            supports_op(&k, &[r3, d]),
            Err(RejectReason::Rank { .. })
        ));
    }

    #[test]
    fn refuses_a_non_contiguous_stride() {
        let k = kernel_f32();
        let d = f32_desc([8, 3, 1, 1], [4, 32, 96, 96]);
        let padded = f32_desc([8, 3, 1, 1], [4, 40, 120, 120]);
        assert!(matches!(
            supports_op(&k, &[padded, d]),
            Err(RejectReason::NotContiguous { .. })
        ));
    }

    /// `c[i,j] = Σ_k a[k,i]·b[k,j]` takes `n_k` from `a`, then indexes `b`
    /// with it. No kernel declares the equality; it is derived from the shared
    /// axis, and a mismatch is a rejection rather than an out-of-bounds read.
    #[test]
    fn refuses_two_arguments_that_disagree_on_an_axis() {
        let mut kb = KernelBuilder::new("mat_mul");
        let a = kb.input("a", TensorType::f32_2d());
        let b = kb.input("b", TensorType::f32_2d());
        let c = kb.output("c", TensorType::f32_2d());
        let i = kb.axis("i", Extent::Dim { arg: a, dim: 1 });
        let j = kb.axis("j", Extent::Dim { arg: b, dim: 1 });
        let kk = kb.axis("k", Extent::Dim { arg: a, dim: 0 });
        let av = kb.read(a, &[kk, i]);
        let bv = kb.read(b, &[kk, j]);
        let p = kb.mul(av, bv);
        let s = kb.reduce(
            crate::ir::ReduceOp::Sum,
            kk,
            p,
            crate::ir::ReductionSemantics::Deterministic,
        );
        kb.write(c, &[i, j], s);
        let k = kb.finish().unwrap();

        let da = f32_desc([8, 4, 1, 1], [4, 32, 128, 128]); // n_k = 8
        let dc = f32_desc([4, 5, 1, 1], [4, 16, 80, 80]);
        let ok_b = f32_desc([8, 5, 1, 1], [4, 32, 160, 160]);
        assert_eq!(supports_op(&k, &[da, ok_b, dc]), Ok(()));

        let short_b = f32_desc([6, 5, 1, 1], [4, 24, 120, 120]); // n_k = 6
        assert!(matches!(
            supports_op(&k, &[da, short_b, dc]),
            Err(RejectReason::AxisExtent {
                expected: 8,
                got: 6,
                ..
            })
        ));
    }

    /// A trailing extent of 1 lowers the effective rank; a rank-2 kernel must
    /// still accept a single-row tensor.
    #[test]
    fn accepts_a_smaller_effective_rank() {
        let k = kernel_f32();
        let d = f32_desc([8, 1, 1, 1], [4, 32, 32, 32]);
        assert_eq!(supports_op(&k, &[d, d]), Ok(()));
    }

    #[test]
    fn refuses_an_incomplete_quantized_block() {
        let mut kb = KernelBuilder::new("q");
        let ty = TensorType {
            dtype: DType::Quant(QuantType::Q8_0),
            rank: 2,
            layout: crate::layout::Layout::Ggml,
        };
        let x = kb.input("x", ty);
        let y = kb.output("y", TensorType::f32(1));
        let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = kb.read(x, &[col, row]);
        let s = kb.reduce(
            crate::ir::ReduceOp::Sum,
            col,
            v,
            crate::ir::ReductionSemantics::Deterministic,
        );
        kb.write(y, &[row], s);
        let k = kb.finish().unwrap();

        let xq = TensorDesc {
            dtype: DType::Quant(QuantType::Q8_0),
            ne: [33, 2, 1, 1],
            nb: [34, 68, 136, 136],
        };
        let yd = f32_desc([2, 1, 1, 1], [4, 8, 8, 8]);
        assert!(matches!(
            supports_op(&k, &[xq, yd]),
            Err(RejectReason::QuantBlock { .. })
        ));
    }
}
