//! CUDA's memory model: a `uint8_t *` reinterpreted at a byte offset.
//!
//! The statement skeleton is `crate::printer::stmts`, shared by the three GPU
//! backends. What is left here is the pair measured as structurally
//! divergent, and the
//! reason is in `crate::printer::memory` rather than repeated per backend.
//!
//! CUDA's own share of it: there is no typed view and no element index, unlike
//! Vulkan, so the Loop IR's byte address is already the address and the access
//! type only decides what it is reinterpreted as. A vector access is *w* scalar
//! accesses from one base address - not a `float4`, which would demand a
//! sixteen-byte address the contract deliberately does not require (ADR-2 section 6).

use rir_lower::MemType;

use crate::dialect::Dialect;
use crate::printer::Printer;
use crate::printer::memory::{LoadOp, MemoryModel, StoreOp};

use super::*;

impl MemoryModel for crate::dialect::Cuda {
    fn load(p: &mut Printer<'_, Self>, op: &LoadOp<'_>) {
        let LoadOp {
            dst,
            arg,
            ty,
            addr,
            width,
        } = *op;
        let arg_name = p.k.args[arg.0 as usize].name.clone();
        let a = p.addr(arg, addr);
        let (mem_ty, elem) = CudaPrinter::mem_ty(ty);
        // One value read at a byte offset from the binding's base, converted to
        // the F32 register every computation uses downstream (ADR-3 section 6).
        let read = |at: String| {
            let raw = format!("*(const {mem_ty} *)({arg_name} + ({at}))");
            match ty {
                MemType::F32 => raw,
                MemType::F16 => format!("__half2float({raw})"),
                _ => format!("float({raw})"),
            }
        };
        if width > 1 {
            // The byte address is computed **once** and the components read
            // from it: what the width buys is the amortized address algebra,
            // four stride products and three sums for four elements instead of
            // one (ADR-2 section 6).
            let base = format!("{}_i", p.var(dst));
            p.line(&format!("const {} {base} = {a};", Self::UINT));
            let comps: Vec<String> = (0..width)
                .map(|c| {
                    if c == 0 {
                        read(base.clone())
                    } else {
                        read(format!("{base} + {}u", c * elem))
                    }
                })
                .collect();
            p.line(&format!(
                "const {} {} = {};",
                Self::decl_ty(rir_lower::VarKind::Vec(width)),
                p.var(dst),
                Self::vec_ctor(width, &comps)
            ));
            return;
        }
        // An unsigned byte lands in an integer register: the reader wants the
        // raw bits, not a converted value.
        if ty == MemType::U8 {
            p.line(&format!(
                "const {} {} = {}(*(const {mem_ty} *)({arg_name} + ({a})));",
                Self::UINT,
                p.var(dst),
                Self::UINT
            ));
            return;
        }
        p.line(&format!(
            "const float {} = {};",
            p.var(dst),
            read(a.clone())
        ));
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
        let arg_name = p.k.args[arg.0 as usize].name.clone();
        let a = p.addr(arg, addr);
        let (mem_ty, elem) = CudaPrinter::mem_ty(ty);
        // The narrowing conversion, printed only when there is one: an F32
        // store must stay the exact line it already was (ADR-3 section 6).
        // `__float2half` is CUDA's round-to-nearest-even, which is what the
        // native kernel storing an F16 result does.
        let narrow = |v: String| match ty {
            MemType::F32 => v,
            MemType::F16 => format!("__float2half({v})"),
            _ => format!("{mem_ty}({v})"),
        };
        let write =
            |at: String, v: String| format!("*({mem_ty} *)({arg_name} + ({at})) = {};", narrow(v));
        // A bounded vector store: whole while the vector fits, one component at
        // a time on the last, partial one. The `for` is a branch on
        // uniform-per-thread data, not on a barrier, so it costs nothing the
        // other threads wait on (ADR-2 section 6).
        if let Some((base, axis)) = bound {
            let b = p.var(base).to_string();
            let n = p.extent(axis);
            let v = p.var(value).to_string();
            let at = format!("{v}_o");
            p.line(&format!("const {} {at} = {a};", Self::UINT));
            p.line(&format!("if ({b} + {width}u <= {n}) {{"));
            for c in 0..width as usize {
                let off = if c == 0 {
                    at.clone()
                } else {
                    format!("{at} + {}u", c as u32 * elem)
                };
                p.line(&format!("    {}", write(off, p.component(value, c))));
            }
            p.line("} else {");
            p.line(&format!(
                "    for ({} c = 0u; c + {b} < {n}; ++c) {{",
                Self::UINT
            ));
            p.line(&format!(
                "        {}",
                write(format!("{at} + c * p.{arg_name}_nb0"), format!("{v}.c[c]"))
            ));
            p.line("    }");
            p.line("}");
            return;
        }
        if width > 1 {
            let base = format!("{}_o", p.var(value));
            p.line(&format!("const {} {base} = {a};", Self::UINT));
            for c in 0..width as usize {
                let off = if c == 0 {
                    base.clone()
                } else {
                    format!("{base} + {}u", c as u32 * elem)
                };
                p.line(&write(off, p.component(value, c)));
            }
            return;
        }
        p.line(&write(a.clone(), p.var(value).to_string()));
    }
}
