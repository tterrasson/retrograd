//! Metal type spellings are provided by `crate::printer::Printer` and
//! `crate::dialect::Metal`: register declarations, vector widths, splats,
//! shared-array and LUT names, reduction identities and combiners, axis
//! extents, and flattened-index scaling.
//!
//! Unlike CUDA, there is no `mem_ty`: MSL spells its element type inline
//! at the two places that need it, `Load` and `Store`.
