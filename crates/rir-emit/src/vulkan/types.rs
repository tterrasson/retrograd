//! Vulkan type spellings are provided by `crate::printer::Printer` and
//! `crate::dialect::Vulkan`: register declarations, vector widths, splats,
//! component selectors, shared-array names, reduction identities and
//! combiners, axis extents, and flattened-index scaling.
//!
//! GLSL needs no `mem_ty`: it never reinterprets a byte pointer. Each access
//! type has its own typed view, and what converts a byte address into an index
//! into one is `addr::index_expr`.
