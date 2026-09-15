//! `LExpr` as a Rust expression.
//!
//! The collective forms - shuffles, block reductions, anything that needs more
//! than one lane - have no Rust spelling and print a `compile_error!`: a
//! schedule that reaches here was mapped to the wrong backend, and failing at
//! `rustc` beats emitting something that compiles and computes nothing.

use rir_core::CmpOp;
use rir_lower::{LExpr, VarId};

use super::CpuPrinter;

impl CpuPrinter<'_> {
    pub(super) fn expr(&self, e: &LExpr) -> String {
        let v = |x: &VarId| self.var(*x).to_string();
        match e {
            LExpr::ConstF32(c) => format!("{c:?}f32"),
            LExpr::Param(p) => self.k.params[p.0 as usize].name.clone(),
            LExpr::AxisExtent(axis) => format!("{} as f32", self.extent_name(*axis)),
            LExpr::Copy(a) => v(a),
            LExpr::Add(a, b) => format!("{} + {}", v(a), v(b)),
            LExpr::Sub(a, b) => format!("{} - {}", v(a), v(b)),
            LExpr::Mul(a, b) => format!("{} * {}", v(a), v(b)),
            LExpr::Div(a, b) => format!("{} / {}", v(a), v(b)),
            LExpr::Sqrt(a) => format!("{}.sqrt()", v(a)),
            LExpr::Exp(a) => format!("{}.exp()", v(a)),
            LExpr::Tanh(a) => format!("{}.tanh()", v(a)),
            LExpr::Cmp { op, lhs, rhs } => {
                let sym = match op {
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                };
                format!("{} {} {}", v(lhs), sym, v(rhs))
            }
            LExpr::Select { cond, t, f } => {
                format!("if {} {{ {} }} else {{ {} }}", v(cond), v(t), v(f))
            }
            LExpr::IDivC(a, c) => format!("{} / {}", v(a), c),
            LExpr::IModC(a, c) => format!("{} % {}", v(a), c),
            LExpr::IMulC(a, c) => format!("{} * {}", v(a), c),
            LExpr::IAdd(a, b) => format!("{} + {}", v(a), v(b)),
            LExpr::IAndC(a, c) => format!("{} & {}", v(a), c),
            LExpr::IShrC(a, c) => format!("{} >> {}", v(a), c),
            LExpr::IShr(a, b) => format!("{} >> {}", v(a), v(b)),
            LExpr::IOr(a, b) => format!("{} | {}", v(a), v(b)),
            LExpr::IModAxis { var, axis } => {
                format!("{} % {}", v(var), self.extent_name(*axis))
            }
            LExpr::Lut { table, idx } => format!("{}[{}]", table.symbol().to_uppercase(), v(idx)),
            LExpr::IToF(a) => format!("{} as f32", v(a)),
            // The oracle is scalar and sequential, so nothing lowered for it
            // produces these: they are the index arithmetic and the combiner of
            // the workgroup scan trees, which reach
            // this emitter only if a GPU schedule did. Printing something
            // plausible would hide that; `unreachable_stmt` says it.
            LExpr::ConstIdx(_)
            | LExpr::IAddC(..)
            | LExpr::ISubC(..)
            | LExpr::ISub(..)
            | LExpr::ICmpC { .. }
            | LExpr::Combine { .. }
            | LExpr::AxisExtentIdx(_) => {
                "compile_error!(\"collective expression on CPU\")".to_string()
            }
            LExpr::AddrSum { arg, terms } => self.addr(*arg, terms),
        }
    }
}
