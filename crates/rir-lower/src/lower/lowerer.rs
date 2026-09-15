//! The lowering context: registers, scopes, addresses, and the walk from a
//! semantic `ValueId` to the register that holds it.

use rir_core::IrId;
use std::collections::HashMap;

use rir_core::{ArgId, AxisId, DType, Kernel, Op, ValueId};

use crate::loop_ir::*;

use crate::lower::LowerError;

pub(crate) struct Lowerer<'k> {
    pub(crate) k: &'k Kernel,
    pub(crate) var_names: Vec<String>,
    pub(crate) var_kinds: Vec<VarKind>,
    /// ValueId-to-register memo valid in the current loop scope.
    pub(crate) env: HashMap<ValueId, VarId>,
    /// Reduction and scan results visible in epilogue scopes.
    pub(crate) results_env: HashMap<ValueId, VarId>,
    /// Logical axis to loop variable in the current scope.
    pub(crate) axis_vars: HashMap<AxisId, VarId>,
    /// Statement buffer for the current scope.
    pub(crate) body: Vec<Stmt>,
    /// Shared arrays the kernel declares, in the order they are created
    /// Copied to `LoopKernel::shared`, which is
    /// what the emitters and the interpreter allocate from.
    pub(crate) shared: Vec<(VarId, u32)>,
    /// Encountering `Reduce`/`Scan` during phase one indicates nesting.
    pub(crate) allow_results: bool,
    /// The linear-addressing register and the number of elements one index of
    /// it covers, under `Schedule::linear_addr`.
    ///
    /// `Some` turns every access address into the single term `linear · width ·
    /// elem_bytes`, and it is set only after `check_linear_addressable` has
    /// proved on the semantic graph that the access *is* that point of the
    /// flattened space. `None` is the ordinary Σ `idx[d] · nb[d]`.
    pub(crate) linear: Option<(VarId, u32)>,
}

impl<'k> Lowerer<'k> {
    pub(crate) fn new_var(&mut self, base: &str, kind: VarKind) -> VarId {
        let id = VarId::at(self.var_names.len());
        self.var_names.push(format!("{}_v{}", base, id.0));
        self.var_kinds.push(kind);
        id
    }

    pub(crate) fn emit(&mut self, base: &str, kind: VarKind, expr: LExpr) -> VarId {
        let dst = self.new_var(base, kind);
        self.body.push(Stmt::Compute(Inst { dst, expr }));
        dst
    }

    /// Opens a fresh evaluation scope with empty memo and statement buffer.
    pub(crate) fn begin_scope(&mut self, allow_results: bool) {
        self.env.clear();
        self.body = Vec::new();
        self.allow_results = allow_results;
    }

    pub(crate) fn take_body(&mut self) -> Vec<Stmt> {
        std::mem::take(&mut self.body)
    }

    pub(crate) fn lower_idx(&mut self, idx: &[ValueId]) -> Result<Vec<VarId>, LowerError> {
        idx.iter().map(|&iv| self.lower_value(iv)).collect()
    }

    /// Memory access type of a **dense** argument, read off its declared dtype
    /// (ADR-3 section 6).
    ///
    /// It is the whole of what F16 costs at this level, and that is the point:
    /// a second element type is a second `MemType` at the memory
    /// boundary, not a second register bank. Every register stays F32 - a
    /// widening load and a narrowing store are the only two places the
    /// difference exists, exactly as they already do for a quantized block's
    /// F16 scale.
    ///
    /// A quantized argument never reaches here: its accesses are built by the
    /// fused decoder, which chooses its own types per field.
    pub(crate) fn elem_mem_type(k: &Kernel, arg: ArgId) -> Result<MemType, LowerError> {
        match k.args()[arg.0 as usize].ty.dtype {
            DType::F32 => Ok(MemType::F32),
            DType::F16 => Ok(MemType::F16),
            dtype => Err(LowerError::UnsupportedElementType { arg, dtype }),
        }
    }

    /// Unblocked element address: Σ idx[d]·nb[d]. Independent of the element
    /// type, because `nb[]` is in bytes - an F16 argument carries `nb[0] == 2`
    /// and the same sum addresses it (ADR-3 section 6).
    pub(crate) fn f32_addr(idx_vars: &[VarId]) -> Vec<AddrTerm> {
        idx_vars
            .iter()
            .enumerate()
            .map(|(d, &v)| AddrTerm::VarNb { var: v, dim: d })
            .collect()
    }

    /// The address of a **dense** access, in whichever of the two forms this
    /// lowering is entitled to.
    ///
    /// Under linear addressing the index registers are not read at all - they
    /// are never even computed - because the claim says the byte offset of the
    /// flattened point is the flattened point itself, scaled by the element:
    /// `nb[0] = elem_bytes` and `nb[d] = nb[d-1] · ne[d-1]` is exactly the
    /// stride sum collapsing into one product. The width multiplies it because
    /// the linear index counts *vectors* on the contiguous axis, and the claim
    /// requires the row to be a whole number of them.
    pub(crate) fn dense_addr(&self, arg: ArgId, idx_vars: &[VarId]) -> Vec<AddrTerm> {
        match self.linear {
            Some((linear, width)) => vec![AddrTerm::VarConst {
                var: linear,
                c: width * self.k.args()[arg.0 as usize].ty.dtype.size_bytes() as u32,
            }],
            None => Self::f32_addr(idx_vars),
        }
    }

    /// `sel < 0.5`, i.e. "the integer selector is zero", in F32.
    ///
    /// The comparison runs in F32 so that one `Cmp`/`Select` pair covers every
    /// backend, with no integer predicate type for an emitter to invent.
    pub(crate) fn lower_is_low(&mut self, sel: VarId, half_c: VarId, name: &str) -> VarId {
        let sel_f = self.emit(&format!("{name}_self"), VarKind::F32, LExpr::IToF(sel));
        self.emit(
            name,
            VarKind::Bool,
            LExpr::Cmp {
                op: rir_core::CmpOp::Lt,
                lhs: sel_f,
                rhs: half_c,
            },
        )
    }

    /// Selects between two `Idx` registers, in F32.
    pub(crate) fn lower_pick(&mut self, name: &str, cond: VarId, t: VarId, f: VarId) -> VarId {
        let tf = self.emit(&format!("{name}_t"), VarKind::F32, LExpr::IToF(t));
        let ff = self.emit(&format!("{name}_f"), VarKind::F32, LExpr::IToF(f));
        self.emit(name, VarKind::F32, LExpr::Select { cond, t: tf, f: ff })
    }

    pub(crate) fn lower_value(&mut self, v: ValueId) -> Result<VarId, LowerError> {
        if let Some(&var) = self.env.get(&v) {
            return Ok(var);
        }
        let op = self.k.ops()[v.0 as usize].clone();
        let var = match op {
            Op::ConstF32(c) => self.emit("c", VarKind::F32, LExpr::ConstF32(c)),
            Op::Index(axis) => *self
                .axis_vars
                .get(&axis)
                .ok_or(LowerError::AxisOutOfScope { value: v, axis })?,
            // One `IModC`-shaped instruction per folded dimension, with a
            // runtime divisor. `hoist` sees it like any other `Idx` compute, so
            // a fold that does not depend on the inner loop leaves it - which is
            // the whole cost question, and it is measured rather than
            // reasoned about.
            Op::RepeatIndex { index, over } => {
                let var = self.lower_value(index)?;
                self.emit("rep", VarKind::Idx, LExpr::IModAxis { var, axis: over })
            }
            Op::AxisExtent(axis) => self.emit("n", VarKind::F32, LExpr::AxisExtent(axis)),
            Op::Param(p) => self.emit("p", VarKind::F32, LExpr::Param(p)),
            Op::Read { tensor, idx } => {
                if matches!(self.k.args()[tensor.0 as usize].ty.dtype, DType::Quant(_)) {
                    return Err(LowerError::UnfusedQuantRead { value: v });
                }
                // Not lowered at all under linear addressing: the address does
                // not read them, and evaluating an `Index` there would bind a
                // register the flattened nest never assigns.
                let idx_vars = match self.linear {
                    Some(_) => Vec::new(),
                    None => self.lower_idx(&idx)?,
                };
                let dst = self.new_var("t", VarKind::F32);
                self.body.push(Stmt::Load {
                    dst,
                    arg: tensor,
                    ty: Self::elem_mem_type(self.k, tensor)?,
                    addr: self.dense_addr(tensor, &idx_vars),
                    width: 1,
                });
                dst
            }
            Op::Dequant { value: rv, from } => {
                let (tensor, idx) = match &self.k.ops()[rv.0 as usize] {
                    Op::Read { tensor, idx } => (*tensor, idx.clone()),
                    _ => unreachable!("validated Dequant: applies to a Read"),
                };
                let idx_vars = self.lower_idx(&idx)?;
                self.lower_dequant_read(tensor, &idx_vars, from)?
            }
            Op::Reduce { .. } | Op::Scan { .. } => {
                if !self.allow_results {
                    return Err(LowerError::NestedReduce { value: v });
                }
                *self
                    .results_env
                    .get(&v)
                    .ok_or(LowerError::NestedReduce { value: v })?
            }
            Op::Add(a, b) => {
                let (a, b) = (self.lower_value(a)?, self.lower_value(b)?);
                self.emit("t", VarKind::F32, LExpr::Add(a, b))
            }
            Op::Sub(a, b) => {
                let (a, b) = (self.lower_value(a)?, self.lower_value(b)?);
                self.emit("t", VarKind::F32, LExpr::Sub(a, b))
            }
            Op::Mul(a, b) => {
                let (a, b) = (self.lower_value(a)?, self.lower_value(b)?);
                self.emit("t", VarKind::F32, LExpr::Mul(a, b))
            }
            Op::Div(a, b) => {
                let (a, b) = (self.lower_value(a)?, self.lower_value(b)?);
                self.emit("t", VarKind::F32, LExpr::Div(a, b))
            }
            Op::Sqrt(a) => {
                let a = self.lower_value(a)?;
                self.emit("t", VarKind::F32, LExpr::Sqrt(a))
            }
            Op::Exp(a) => {
                let a = self.lower_value(a)?;
                self.emit("t", VarKind::F32, LExpr::Exp(a))
            }
            Op::Tanh(a) => {
                let a = self.lower_value(a)?;
                self.emit("t", VarKind::F32, LExpr::Tanh(a))
            }
            Op::Cmp { op, lhs, rhs } => {
                let (lhs, rhs) = (self.lower_value(lhs)?, self.lower_value(rhs)?);
                self.emit("b", VarKind::Bool, LExpr::Cmp { op, lhs, rhs })
            }
            Op::Select { cond, t, f } => {
                let cond = self.lower_value(cond)?;
                let t = self.lower_value(t)?;
                let f = self.lower_value(f)?;
                self.emit("t", VarKind::F32, LExpr::Select { cond, t, f })
            }
            Op::Write { .. } => unreachable!("Write is not a value"),
        };
        self.env.insert(v, var);
        Ok(var)
    }

    pub(crate) fn lower_store(
        &mut self,
        tensor: ArgId,
        idx: &[ValueId],
        value: ValueId,
    ) -> Result<(), LowerError> {
        let idx_vars = match self.linear {
            Some(_) => Vec::new(),
            None => self.lower_idx(idx)?,
        };
        let v = self.lower_value(value)?;
        self.body.push(Stmt::Store {
            arg: tensor,
            ty: Self::elem_mem_type(self.k, tensor)?,
            addr: self.dense_addr(tensor, &idx_vars),
            value: v,
            width: 1,
            bound: None,
        });
        Ok(())
    }
}
