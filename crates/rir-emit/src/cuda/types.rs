//! What CUDA reads a memory access type through.
//!
//! Register declarations, vector widths, shared-array and LUT names, reduction
//! identities and the flattened-index scaling all moved to
//! `crate::printer::Printer` and `crate::dialect::Cuda`: the three copies
//! were one function per
//! entry with a different word in it.
//!
//! What stays is the one thing that is not a word. `mem_ty` belongs to CUDA's
//! memory model - a binding is a `uint8_t *` reinterpreted at a byte offset, so
//! an access type names the C type it is cast to and the size it advances by.
//! Vulkan has no equivalent at all (it indexes a typed view in elements) and
//! Metal spells its own inline.

use super::*;

impl<'k> CudaPrinter<'k> {
    /// The C type one element of an access type is read through, and its size
    /// in bytes.
    pub(crate) fn mem_ty(ty: rir_lower::MemType) -> (&'static str, u32) {
        match ty {
            rir_lower::MemType::F32 => ("float", 4),
            rir_lower::MemType::F16 => ("__half", 2),
            rir_lower::MemType::I8 => ("int8_t", 1),
            rir_lower::MemType::U8 => ("uint8_t", 1),
        }
    }
}
