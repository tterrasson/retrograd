//! Vulkan's memory model: a **typed view**, indexed in elements.
//!
//! The statement skeleton is `crate::printer::stmts`, shared by the three GPU
//! backends. What is left here is the pair measured as structurally
//! divergent, and the reason is in `crate::printer::memory`.
//!
//! Vulkan's own share of it is the one that is not a variation on the other
//! two. It never reinterprets a byte pointer: each access type has its own
//! typed view over the same binding (`view_name`, descriptor aliasing), and a
//! Loop IR byte address has to be divided by the element size to index one
//! (`index_expr`). std430 has no unaligned `vec4` view either (an aliased one
//! would need a sixteen-byte address the contract deliberately does not
//! require), so a width-*w* access is *w* indexed accesses from a common base
//! index.

use rir_lower::{MemType, VarKind};

use crate::dialect::Dialect;
use crate::printer::Printer;
use crate::printer::memory::{LoadOp, MemoryModel, StoreOp};

use super::*;

impl MemoryModel for crate::dialect::Vulkan {
    fn load(p: &mut Printer<'_, Self>, op: &LoadOp<'_>) {
        let LoadOp {
            dst,
            arg,
            ty,
            addr,
            width,
        } = *op;
        // GLSL converts `float16_t`/`int8_t` to the F32 `float` register.
        let view = view_name(&p.k.args[arg.0 as usize].name.clone(), ty);
        let a = p.addr(arg, addr);
        let idx = index_expr(&a, ty.size_bytes());
        // The element index is computed **once** and the components read from
        // it: what the width buys here is the amortized address algebra, four
        // stride products and three sums for four elements instead of one.
        if width > 1 {
            let base = format!("{}_i", p.var(dst));
            p.line(&format!("const {} {base} = {idx};", Self::UINT));
            let comps: Vec<String> = (0..width)
                .map(|c| {
                    if c == 0 {
                        format!("{view}[{base}]")
                    } else {
                        format!("{view}[{base} + {c}u]")
                    }
                })
                .collect();
            p.line(&format!(
                "const {} {} = {};",
                Self::decl_ty(VarKind::Vec(width)),
                p.var(dst),
                Self::vec_ctor(width, &comps)
            ));
            return;
        }
        let l = match ty {
            MemType::F32 => format!("const float {} = {view}[{idx}];", p.var(dst)),
            MemType::F16 | MemType::I8 => {
                format!("const float {} = float({view}[{idx}]);", p.var(dst))
            }
            // An unsigned byte lands in an integer register: the reader wants
            // the raw bits, not a converted value.
            MemType::U8 => format!(
                "const {} {} = {}({view}[{idx}]);",
                Self::UINT,
                p.var(dst),
                Self::UINT
            ),
        };
        p.line(&l);
    }

    fn store(p: &mut Printer<'_, Self>, op: &StoreOp<'_>) {
        let StoreOp {
            arg,
            ty,
            addr,
            value,
            width,
            bound,
        } = *op;
        let view = view_name(&p.k.args[arg.0 as usize].name.clone(), ty);
        let a = p.addr(arg, addr);
        let idx = index_expr(&a, ty.size_bytes());
        // The narrowing conversion, printed only when there is one: GLSL has no
        // implicit `float -> float16_t`, and an F32 store must stay the exact
        // line it already was.
        let narrow = |v: String| {
            if ty == MemType::F32 {
                v
            } else {
                format!("float16_t({v})")
            }
        };
        // A bounded vector store: whole while the vector fits, one component at
        // a time on the last, partial one. The `for` is a branch on
        // uniform-per-invocation data, not on a barrier, so it costs nothing
        // the other invocations wait on.
        if let Some((base, axis)) = bound {
            let b = p.var(base).to_string();
            let n = p.extent(axis);
            let v = p.var(value).to_string();
            let at = format!("{v}_o");
            p.line(&format!("const {} {at} = {idx};", Self::UINT));
            p.line(&format!("if ({b} + {width}u <= {n}) {{"));
            for c in 0..width as usize {
                p.line(&format!(
                    "    {view}[{at} + {c}u] = {};",
                    narrow(p.component(value, c))
                ));
            }
            p.line("} else {");
            p.line(&format!(
                "    for ({} c = 0u; c + {b} < {n}; ++c) {{",
                Self::UINT
            ));
            p.line(&format!(
                "        {view}[{at} + c] = {};",
                narrow(format!("{v}[c]"))
            ));
            p.line("    }");
            p.line("}");
            return;
        }
        if width > 1 {
            let base = format!("{}_o", p.var(value));
            p.line(&format!("const {} {base} = {idx};", Self::UINT));
            for c in 0..width as usize {
                let at = if c == 0 {
                    base.clone()
                } else {
                    format!("{base} + {c}u")
                };
                p.line(&format!(
                    "{view}[{at}] = {};",
                    narrow(p.component(value, c))
                ));
            }
            return;
        }
        let l = format!("{view}[{idx}] = {};", narrow(p.var(value).to_string()));
        p.line(&l);
    }
}
