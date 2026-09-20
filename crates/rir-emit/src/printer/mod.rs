//! The printer the three GPU emitters share, parameterized by its dialect.
//!
//! What lives here is the part of an emitter that is *not* a lexicon: the
//! output buffer, the indentation, and the statement skeleton - the shape of
//! the emitted code, which was measured to be the same on CUDA, Metal and
//! Vulkan for 23 of the 25 `Stmt` variants.
//!
//! Everything that differs comes from `D: Dialect` and from nowhere else. A
//! `match` on the backend in this file is the regression the golden rule of
//! [`crate`] names, and the reason the type parameter is a marker rather than a
//! runtime tag: there is no backend value here to match on.
//!
//! Each backend keeps a type alias (`CudaPrinter`, `MetalPrinter`, `VkPrinter`)
//! and the inherent impls of what stays its own - its file preamble, and the
//! two statements whose memory model is genuinely different.

pub(crate) mod expr;
pub(crate) mod memory;
pub(crate) mod stmts;

use std::marker::PhantomData;

use rir_core::ArgId;
use rir_lower::{AddrTerm, LoopKernel, VarId, VarKind};

use crate::dialect::Dialect;

pub(crate) struct Printer<'k, D: Dialect> {
    pub(crate) k: &'k LoopKernel,
    pub(crate) out: String,
    pub(crate) indent: usize,
    _dialect: PhantomData<D>,
}

impl<'k, D: Dialect> Printer<'k, D> {
    pub(crate) fn new(k: &'k LoopKernel) -> Self {
        Printer {
            k,
            out: String::new(),
            indent: 0,
            _dialect: PhantomData,
        }
    }

    pub(crate) fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub(crate) fn var(&self, v: VarId) -> &str {
        &self.k.var_names[v.0 as usize]
    }

    pub(crate) fn kind(&self, v: VarId) -> VarKind {
        self.k.var_kinds[v.0 as usize]
    }

    /// Width of the vector these operands compute in, or `None` when they are
    /// all scalar.
    ///
    /// Shared, but it does not mean the same thing to the three backends, and
    /// that is a property of the *languages* rather than of this function.
    /// GLSL has no scalar broadcast on `greaterThan`/`mix` and CUDA has no
    /// vector arithmetic at all, so both need the width before an operand is
    /// printed; MSL broadcasts across its operators and needs it only where
    /// three arguments must agree. One reading, three uses.
    pub(crate) fn vec_width(&self, vs: &[VarId]) -> Option<u32> {
        vs.iter().find_map(|v| match self.kind(*v) {
            VarKind::Vec(w) | VarKind::VecBool(w) => Some(w),
            _ => None,
        })
    }

    pub(crate) fn shared(&self, dst: VarId) -> String {
        format!("rir_shared_{}", self.var(dst))
    }

    pub(crate) fn extent(&self, axis: rir_core::AxisId) -> String {
        format!("{}n_{}", D::PARAMS, self.k.axes[axis.0 as usize].name)
    }

    /// What a register is declared as.
    pub(crate) fn decl_ty(&self, v: VarId) -> String {
        D::decl_ty(self.kind(v))
    }

    /// An operand printed at `width`: itself if it is already a vector, its
    /// broadcast otherwise.
    pub(crate) fn splat(&self, v: VarId, width: u32) -> String {
        D::splat(self.kind(v), width, self.var(v))
    }

    /// Component selector of a vector register, or the register itself when it
    /// is scalar - a scalar under a vector store is a broadcast.
    pub(crate) fn component(&self, v: VarId, c: usize) -> String {
        D::component(self.kind(v), self.var(v), c)
    }

    pub(crate) fn lut(&self, table: rir_core::LutId) -> String {
        D::lut(self.k, table)
    }

    /// Combining two operands already spelled as text - the shared-memory tree
    /// of `WorkgroupReduce`, whose operands are array slots and not registers.
    pub(crate) fn combine(op: rir_core::ReduceOp, a: &str, b: &str) -> String {
        D::combine(op, a, b)
    }

    /// Combining two registers. The width is read here and handed to the
    /// dialect, which is the only one that knows whether it selects anything:
    /// `fmaxf` against `rir_vmax` on CUDA, one polymorphic `max` elsewhere.
    pub(crate) fn combine_vars(&self, op: rir_core::ReduceOp, a: VarId, b: VarId) -> String {
        D::combine_at(op, self.vec_width(&[a, b]), self.var(a), self.var(b))
    }

    /// A decomposed index scaled back to element units: the contiguous axis of
    /// a flattened dispatch counts *vectors*, every other axis counts elements.
    pub(crate) fn scaled(expr: &str, dim: usize, vector: u32) -> String {
        if dim == 0 && vector > 1 {
            format!("({expr}) * {vector}u")
        } else {
            expr.to_string()
        }
    }

    /// The byte address an `AddrTerm` list prints as, in `ggml_tensor.nb[]`
    /// order.
    ///
    /// One function for the three backends, and the whole difference between
    /// them is `D::PARAMS` - this is the case that was picked to go first
    /// precisely because there is nothing else in it.
    ///
    /// Vulkan reads the result through a typed view and therefore divides it by
    /// an element size afterwards (`vulkan::addr::index_expr`); that conversion
    /// belongs to its memory model, not to the address algebra, and stays
    /// there.
    pub(crate) fn addr(&self, arg: ArgId, addr: &[AddrTerm]) -> String {
        let name = &self.k.args[arg.0 as usize].name;
        let params = D::PARAMS;
        addr.iter()
            .map(|term| match term {
                AddrTerm::VarNb { var, dim } => {
                    format!("{} * {params}{name}_nb{dim}", self.var(*var))
                }
                AddrTerm::VarConst { var, c } if *c == 1 => self.var(*var).to_string(),
                AddrTerm::VarConst { var, c } => format!("{} * {}u", self.var(*var), c),
                AddrTerm::Const(c) => format!("{c}u"),
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
}
