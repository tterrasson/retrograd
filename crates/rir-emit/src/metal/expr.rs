//! Metal expressions use the shared `crate::printer::expr` printer. MSL-specific
//! spellings come from `crate::dialect::Metal`: `precise::tanh` rather than
//! `tanh`, `select` as the
//! componentwise `?:`, and the absence of a `vector_cmp` entry - MSL broadcasts
//! across its comparison operators, so its scalar spelling is already its
//! vector one.
