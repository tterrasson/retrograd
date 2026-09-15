//! CUDA expressions use the shared `crate::printer::expr` printer. Backend-
//! specific spellings come from `crate::dialect::Cuda`,
//! `float_literal` and its `f` suffix, `PARAMS`, `unary_fn` and its
//! scalar/vector pair, `SELECT`, `vector_cmp`, `lut`.
