//! Metal's memory model: a byte offset, and **one** packed vector access.
//!
//! The statement skeleton is `crate::printer::stmts`, shared by the three GPU
//! backends. What is left here is the pair measured as structurally
//! divergent, and the
//! reason is in `crate::printer::memory`.
//!
//! Metal's own share of it is the vector case, and it is what separates this
//! file from CUDA's rather than from Vulkan's. Both reinterpret a byte pointer,
//! but CUDA expands a width-*w* access into *w* scalar ones while MSL has
//! `packed_float{w}` - a vector type whose alignment is **one element**, so the
//! eligibility condition stays `nb[0] == elem_bytes` and no base offset has to
//! be sixteen-byte aligned. A plain `float4` would demand addresses the
//! dispatcher cannot promise for a view (ADR-2 section 6).

use rir_lower::MemType;

use crate::dialect::Dialect;
use crate::printer::Printer;
use crate::printer::memory::{LoadOp, MemoryModel, StoreOp};

/// The MSL type one element of an access type is read through. Spelled here
/// rather than in the dialect for the reason the whole file exists: it is part
/// of the reinterpretation, and Vulkan - which indexes typed views - has no
/// place to put it.
fn mem_ty(ty: MemType) -> &'static str {
    match ty {
        MemType::F32 => "float",
        MemType::F16 => "half",
        MemType::I8 => "char",
        MemType::U8 => "uchar",
    }
}

impl MemoryModel for crate::dialect::Metal {
    fn load(p: &mut Printer<'_, Self>, op: &LoadOp<'_>) {
        let LoadOp {
            dst,
            arg,
            ty,
            addr,
            width,
        } = *op;
        let arg_name = &p.k.args[arg.0 as usize].name.clone();
        let addr = p.addr(arg, addr);
        let mem_ty = mem_ty(ty);
        if width > 1 {
            // The register is `float{width}` whatever the memory type: a half
            // vector is widened on the way in, the same conversion the scalar
            // path already prints, and every computation downstream stays F32
            // (ADR-3 section 6).
            let load = format!("*(device const packed_{mem_ty}{width} *)({arg_name} + ({addr}))");
            let load = if ty == MemType::F32 {
                load
            } else {
                format!("float{width}({load})")
            };
            p.line(&format!("const float{width} {} = {load};", p.var(dst)));
            return;
        }
        let value = format!("*(device const {mem_ty} *)({arg_name} + ({addr}))");
        // An unsigned byte lands in an integer register: the reader wants the
        // raw bits, not a converted value.
        if ty == MemType::U8 {
            p.line(&format!(
                "const {} {} = {}({value});",
                Self::UINT,
                p.var(dst),
                Self::UINT
            ));
            return;
        }
        let value = if ty == MemType::F32 {
            value
        } else {
            format!("float({value})")
        };
        p.line(&format!("const float {} = {value};", p.var(dst)));
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
        let addr = p.addr(arg, addr);
        let mem_ty = mem_ty(ty);
        // An F32 store writes the register as it stands; only a narrowing one
        // is spelled as a conversion, so adding a second element type leaves
        // every existing shader byte for byte identical. `half(x)` is MSL's own
        // round-to-nearest-even, which is what the native kernel storing an F16
        // result does too.
        let narrow = |v: String| {
            if ty == MemType::F32 {
                v
            } else {
                format!("{mem_ty}({v})")
            }
        };
        // A bounded vector store: whole while the vector fits, one component at
        // a time on the last, partial one. The `for` is a branch on
        // uniform-per-invocation data, not on a barrier, so it costs nothing
        // the other invocations wait on (ADR-2 section 6).
        if let Some((base, axis)) = bound {
            let b = p.var(base).to_string();
            let n = p.extent(axis);
            let v = p.var(value).to_string();
            let nb0 = format!("{}{arg_name}_nb0", Self::PARAMS);
            p.line(&format!("if ({b} + {width}u <= {n}) {{"));
            p.line(&format!(
                "    *(device packed_{mem_ty}{width} *)({arg_name} + ({addr})) = packed_{mem_ty}{width}({v});"
            ));
            p.line("} else {");
            p.line(&format!(
                "    for ({} c = 0u; c + {b} < {n}; ++c) {{",
                Self::UINT
            ));
            p.line(&format!(
                "        *(device {mem_ty} *)({arg_name} + ({addr}) + c * {nb0}) = {};",
                narrow(format!("{v}[c]"))
            ));
            p.line("    }");
            p.line("}");
            return;
        }
        if width > 1 {
            // A scalar value under a vector store is a broadcast: the
            // constructor splats it, which is the semantics the oracle gives it
            // by re-reading the same register once per component.
            p.line(&format!(
                "*(device packed_{mem_ty}{width} *)({arg_name} + ({addr})) = packed_{mem_ty}{width}({});",
                p.var(value)
            ));
            return;
        }
        p.line(&format!(
            "*(device {mem_ty} *)({arg_name} + ({addr})) = {};",
            narrow(p.var(value).to_string())
        ));
    }
}
