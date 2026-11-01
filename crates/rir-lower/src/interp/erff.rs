//! The interpreter's `erf`, and the same function as text.
//!
//! Rust's standard library has no `erf`, and the oracle must evaluate what
//! ggml's CPU kernels evaluate - `erff` - rather than a polynomial that merely
//! stays close to it: a GPU emitter's approximation is judged against this
//! function, so this function cannot be an approximation of the same kind.
//!
//! The source lives in `erff_source.rs` and is used twice: compiled here, and
//! exposed as [`SOURCE`] for the CPU emitter to print ahead of a kernel that
//! calls `erff`, since emitted Rust depends on no RIR crate. Both uses are one
//! text, so the oracle and the generated CPU kernel cannot drift apart; what
//! judges the text itself is an `erf` series evaluated in `f64`
//! (`rir-kernels`, `unary`).

/// The item definitions of `erff`, ready to be printed at module level.
pub const SOURCE: &str = include_str!("erff_source.rs");

include!("erff_source.rs");

/// `erff`, as the oracle evaluates it.
pub fn erf(x: f32) -> f32 {
    erff(x)
}
