//! One `LExpr` to one expression, for the three GPU backends.
//!
//! No decision is taken here: an expression the lowering did not build cannot
//! be printed, and the match has no wildcard, so an `LExpr` added upstream
//! fails to compile rather than falling into a silent approximation.
//!
//! Of the 32 arms, **19 were already byte-identical** across the three
//! copies. The rest read a word from the dialect,
//! a constant-buffer prefix, a function name, the spelling of a literal.

use rir_core::CmpOp;
use rir_lower::{LExpr, VarId};

use crate::dialect::{Dialect, UnaryFn, cmp_symbol};

use super::Printer;

impl<'k, D: Dialect> Printer<'k, D> {
    pub(crate) fn expr(&self, e: &LExpr) -> String {
        let v = |x: &VarId| self.var(*x).to_string();
        // A unary float function, in the spelling the operand's width requires.
        // CUDA is the reason the width is read at all: GLSL and MSL have a
        // `sqrt` that accepts both a scalar and a vector, and CUDA has neither.
        let map =
            |a: &VarId, f: UnaryFn| format!("{}({})", D::unary_fn(f, self.vec_width(&[*a])), v(a));
        match e {
            LExpr::ConstF32(c) => D::float_literal(*c),
            LExpr::Param(p) => format!("{}{}", D::PARAMS, self.k.params[p.0 as usize].name),
            LExpr::AxisExtent(axis) => format!("float({})", self.extent(*axis)),
            LExpr::Copy(a) => v(a),
            LExpr::Add(a, b) => format!("{} + {}", v(a), v(b)),
            LExpr::Sub(a, b) => format!("{} - {}", v(a), v(b)),
            LExpr::Mul(a, b) => format!("{} * {}", v(a), v(b)),
            LExpr::Div(a, b) => format!("{} / {}", v(a), v(b)),
            LExpr::Sqrt(a) => map(a, UnaryFn::Sqrt),
            LExpr::Exp(a) => map(a, UnaryFn::Exp),
            LExpr::Tanh(a) => map(a, UnaryFn::Tanh),
            LExpr::Cmp { op, lhs, rhs } => self.cmp(*op, *lhs, *rhs),
            // `select(f, t, c)` is `c ? t: f` - the componentwise `?:` none of
            // the three languages has as an operator on a vector. The three
            // names differ; the three call shapes, splats included, do not.
            LExpr::Select { cond, t, f } => match self.vec_width(&[*cond, *t, *f]) {
                Some(w) => format!(
                    "{}({}, {}, {})",
                    D::SELECT,
                    self.splat(*f, w),
                    self.splat(*t, w),
                    self.splat(*cond, w)
                ),
                None => format!("{} ? {} : {}", v(cond), v(t), v(f)),
            },
            LExpr::IDivC(a, c) => format!("{} / {}u", v(a), c),
            LExpr::IModC(a, c) => format!("{} % {}u", v(a), c),
            LExpr::IMulC(a, c) => format!("{} * {}u", v(a), c),
            LExpr::IAdd(a, b) => format!("{} + {}", v(a), v(b)),
            LExpr::IAndC(a, c) => format!("{} & {}u", v(a), c),
            LExpr::IShrC(a, c) => format!("{} >> {}u", v(a), c),
            LExpr::IShr(a, b) => format!("{} >> {}", v(a), v(b)),
            LExpr::IOr(a, b) => format!("{} | {}", v(a), v(b)),
            LExpr::IModAxis { var, axis } => format!("{} % {}", v(var), self.extent(*axis)),
            LExpr::Lut { table, idx } => format!("{}[{}]", self.lut(*table), v(idx)),
            LExpr::IToF(a) => format!("float({})", v(a)),
            LExpr::ConstIdx(c) => format!("{c}u"),
            LExpr::IAddC(a, c) => format!("{} + {c}u", v(a)),
            LExpr::ISubC(a, c) => format!("{} - {c}u", v(a)),
            LExpr::ISub(a, b) => format!("{} - {}", v(a), v(b)),
            LExpr::AxisExtentIdx(axis) => self.extent(*axis),
            LExpr::ICmpC { op, var, c } => format!("{} {} {c}u", v(var), cmp_symbol(*op)),
            LExpr::Combine { op, lhs, rhs } => self.combine_vars(*op, *lhs, *rhs),
            LExpr::AddrSum { arg, terms } => self.addr(*arg, terms),
        }
    }

    /// A comparison, scalar or vector.
    ///
    /// The scalar spelling is one line for the three. The vector one is the
    /// divergence between the three backends, and it is expressed
    /// as two columns of the table rather than three arms: whether the backend
    /// has a named vector form at all, and whether that form wants its operands
    /// broadcast. Reading a column is not deciding - the skeleton never asks
    /// which backend it is printing for.
    fn cmp(&self, op: CmpOp, lhs: VarId, rhs: VarId) -> String {
        if let Some(w) = self.vec_width(&[lhs, rhs])
            && let Some(vector) = D::vector_cmp(op)
        {
            let (a, b) = if vector.splat_operands {
                (self.splat(lhs, w), self.splat(rhs, w))
            } else {
                (self.var(lhs).to_string(), self.var(rhs).to_string())
            };
            return format!("{}({a}, {b})", vector.name);
        }
        format!("{} {} {}", self.var(lhs), cmp_symbol(op), self.var(rhs))
    }
}
