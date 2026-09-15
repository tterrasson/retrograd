//! Builder DSL: the only way to construct a `Kernel`.
//!
//! The builder guarantees SSA form by construction: every operand exists
//! before it is used. `finish()` runs level-one validation before returning
//! the kernel.

use crate::ids::IrId;
use std::collections::HashMap;

use crate::ir::*;
use crate::layout::TensorType;
use crate::types::ScalarType;
use crate::validate::{ValidateError, ValidatedKernel};

pub struct KernelBuilder {
    kernel: Kernel,
    /// One `Op::Index` per axis, reused by `read` and `write`.
    index_cache: HashMap<AxisId, ValueId>,
}

impl KernelBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            kernel: Kernel {
                name: name.to_string(),
                args: Vec::new(),
                params: Vec::new(),
                axes: Vec::new(),
                ops: Vec::new(),
                constraints: Vec::new(),
            },
            index_cache: HashMap::new(),
        }
    }

    fn push(&mut self, op: Op) -> ValueId {
        let id = ValueId::at(self.kernel.ops.len());
        self.kernel.ops.push(op);
        id
    }

    pub fn input(&mut self, name: &str, ty: TensorType) -> ArgId {
        let id = ArgId::at(self.kernel.args.len());
        self.kernel.args.push(Arg {
            name: name.to_string(),
            ty,
            access: Access::Read,
        });
        id
    }

    pub fn output(&mut self, name: &str, ty: TensorType) -> ArgId {
        let id = ArgId::at(self.kernel.args.len());
        self.kernel.args.push(Arg {
            name: name.to_string(),
            ty,
            access: Access::Write,
        });
        id
    }

    /// Declares a uniform scalar parameter and returns its SSA value.
    pub fn param(&mut self, name: &str, ty: ScalarType) -> ValueId {
        let pid = ParamId::at(self.kernel.params.len());
        self.kernel.params.push(ParamDecl {
            name: name.to_string(),
            ty,
        });
        self.push(Op::Param(pid))
    }

    pub fn axis(&mut self, name: &str, extent: Extent) -> AxisId {
        let id = AxisId::at(self.kernel.axes.len());
        self.kernel.axes.push(AxisDecl {
            name: name.to_string(),
            extent,
        });
        id
    }

    /// Extent of an axis as an F32 value. What a mean is divided by.
    pub fn axis_extent(&mut self, axis: AxisId) -> ValueId {
        self.push(Op::AxisExtent(axis))
    }

    fn index(&mut self, axis: AxisId) -> ValueId {
        if let Some(&v) = self.index_cache.get(&axis) {
            return v;
        }
        let v = self.push(Op::Index(axis));
        self.index_cache.insert(axis, v);
        v
    }

    /// Logical read: `idx[d]` indexes ggml dimension `d` (0 is innermost).
    /// For example, a 2D read is `read(x, &[col, row])`.
    ///
    /// For a quantized tensor, inserts an explicit `Dequant` (the IR never
    /// contains implicit conversions). Lowering fuses it with the `Read` into
    /// a dequantizing load.
    pub fn read(&mut self, tensor: ArgId, idx: &[AxisId]) -> ValueId {
        let idx = idx.iter().map(|&a| self.index(a)).collect();
        let dtype = self.kernel.args[tensor.0 as usize].ty.dtype;
        let read = self.push(Op::Read { tensor, idx });
        match dtype {
            crate::types::DType::Quant(q) => self.push(Op::Dequant {
                value: read,
                from: q,
            }),
            _ => read,
        }
    }

    /// **Repeated** logical read: dimension `d` is indexed by `idx[d]` folded
    /// into the extent of `over[d]` (ADR-1 section 5).
    ///
    /// This is `ggml_can_repeat` written down: `src1` is replayed under `src0`,
    /// so element `i` of the destination reads element `i % ne_src1[d]` of the
    /// repeated operand. `over[d]` must be an axis declared on **this
    /// argument's** dimension `d`, which is what makes its extent the divisor
    /// the shader needs and what lets `arg_axes` publish it.
    ///
    /// A dimension whose `over` axis is the *same* axis as `idx[d]` would be a
    /// fold by its own extent - the identity, written as a modulo. It is
    /// rejected by validation rather than emitted: an identity nobody removes
    /// is a division per element for nothing.
    pub fn read_repeat(&mut self, tensor: ArgId, idx: &[AxisId], over: &[AxisId]) -> ValueId {
        assert_eq!(
            idx.len(),
            over.len(),
            "read_repeat: one `over` axis per dimension"
        );
        let idx: Vec<ValueId> = idx
            .iter()
            .zip(over)
            .map(|(&a, &o)| {
                let index = self.index(a);
                self.push(Op::RepeatIndex { index, over: o })
            })
            .collect();
        let dtype = self.kernel.args[tensor.0 as usize].ty.dtype;
        let read = self.push(Op::Read { tensor, idx });
        match dtype {
            crate::types::DType::Quant(q) => self.push(Op::Dequant {
                value: read,
                from: q,
            }),
            _ => read,
        }
    }

    pub fn write(&mut self, tensor: ArgId, idx: &[AxisId], value: ValueId) {
        let idx = idx.iter().map(|&a| self.index(a)).collect();
        self.push(Op::Write { tensor, idx, value });
    }

    pub fn const_f32(&mut self, v: f32) -> ValueId {
        self.push(Op::ConstF32(v))
    }

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Add(lhs, rhs))
    }

    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Sub(lhs, rhs))
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Mul(lhs, rhs))
    }

    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Div(lhs, rhs))
    }

    pub fn sqrt(&mut self, v: ValueId) -> ValueId {
        self.push(Op::Sqrt(v))
    }

    pub fn exp(&mut self, v: ValueId) -> ValueId {
        self.push(Op::Exp(v))
    }

    pub fn tanh(&mut self, v: ValueId) -> ValueId {
        self.push(Op::Tanh(v))
    }

    pub fn cmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Cmp { op, lhs, rhs })
    }

    pub fn gt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.cmp(CmpOp::Gt, lhs, rhs)
    }

    pub fn select(&mut self, cond: ValueId, t: ValueId, f: ValueId) -> ValueId {
        self.push(Op::Select { cond, t, f })
    }

    pub fn reduce(
        &mut self,
        op: ReduceOp,
        axis: AxisId,
        value: ValueId,
        semantics: ReductionSemantics,
    ) -> ValueId {
        self.push(Op::Reduce {
            op,
            axis,
            value,
            semantics,
        })
    }

    pub fn scan(
        &mut self,
        op: ScanOp,
        axis: AxisId,
        dir: ScanDirection,
        value: ValueId,
    ) -> ValueId {
        self.push(Op::Scan {
            op,
            axis,
            dir,
            value,
        })
    }

    pub fn constrain(&mut self, c: Constraint) {
        self.kernel.constraints.push(c);
    }

    /// Validates the kernel and returns the witness the pipeline takes.
    ///
    /// The only way to obtain a [`ValidatedKernel`], and - since `Kernel`'s
    /// fields became crate-private - the only way to obtain a `Kernel` at all.
    /// The two facts together are what the module docstring above always
    /// asserted and nothing enforced.
    ///
    /// # Errors
    ///
    /// The first level-one violation: SSA form, access permissions, index
    /// arity, output coverage, or the kind of a value.
    pub fn finish(self) -> Result<ValidatedKernel, ValidateError> {
        ValidatedKernel::new(self.kernel)
    }
}
