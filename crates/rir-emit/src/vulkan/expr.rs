//! Vulkan expressions use the shared `crate::printer::expr` printer. GLSL-
//! specific spellings come from `crate::dialect::Vulkan`: `mix` as the
//! componentwise `?:`, the plain LUT
//! symbol, and the `greaterThan` family - the one `vector_cmp` entry that wants
//! both operands broadcast to the same width.
