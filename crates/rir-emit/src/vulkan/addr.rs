//! The buffer view a Vulkan binding is read through.
//!
//! The byte address itself is `crate::printer::Printer::addr`, shared by the
//! three GPU backends. What stays here is the
//! part that has no counterpart on CUDA or Metal: Vulkan indexes a *typed
//! view* in elements rather than reinterpreting a byte pointer, so a byte
//! address has to be converted.

use rir_lower::MemType;

/// GLSL view name for an argument and access type. A quantized argument has
/// both `_f16` (scale) and `_i8` (data byte) views: two blocks declared on
/// **the same binding**, using descriptor aliasing allowed by Vulkan. F32 has
/// no suffix.
pub(crate) fn view_name(arg: &str, ty: MemType) -> String {
    format!("{arg}{}", ty.view_suffix())
}

/// Converts a Loop IR byte address to an element index for a typed GLSL array.
pub(crate) fn index_expr(addr: &str, elem_bytes: u32) -> String {
    if elem_bytes == 1 {
        addr.to_string()
    } else {
        format!("({addr}) / {elem_bytes}u")
    }
}
