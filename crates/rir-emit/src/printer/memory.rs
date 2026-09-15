//! The one thing that is not a word: how a backend reaches an element.
//!
//! `Load` and `Store` are the two `Stmt` variants measured as **structurally**
//! divergent - not three spellings of one shape,
//! but three shapes:
//!
//! - **Vulkan** binds each access type as a *typed view* over the same binding
//!   (descriptor aliasing) and indexes it in **elements**, so a Loop IR byte
//!   address has to be divided by the element size. A vector read is *w*
//!   indexed reads from a common base index.
//! - **CUDA** binds a `uint8_t *` and reinterprets it at a **byte** offset.
//!   There is no index division because there is nothing typed to index. A
//!   vector read is *w* scalar reads from a common base address - deliberately
//!   not a `float4`, which would demand a sixteen-byte alignment the contract
//!   does not require (ADR-2 section 6).
//! - **Metal** also reinterprets at a byte offset, but its vector read is
//!   **one** `packed_float{w}` load, whose alignment is one element - so unlike
//!   the two others it has no per-component expansion at all.
//!
//! The narrowing conversions differ with them (`__float2half`, `half(x)`,
//! `float16_t(x)`), and so does the bounded store: CUDA and Metal advance by
//! `{arg}_nb0` bytes, Vulkan by one element index.
//!
//! Merging these would mean the skeleton choosing an addressing mode, which is
//! a decision, and `rir-emit`'s golden rule is that emitters take none. So they
//! stay per backend, behind this trait, and the trait is separate from
//! [`Dialect`](crate::dialect::Dialect) on purpose: everything in that one is a
//! word, and nothing here is.

use rir_core::{ArgId, AxisId};
use rir_lower::{AddrTerm, MemType, VarId};

use crate::dialect::Dialect;

use super::Printer;

/// A read of one element, as the enclosing `Stmt::Load` describes it.
pub(crate) struct LoadOp<'a> {
    pub dst: VarId,
    pub arg: ArgId,
    pub ty: MemType,
    pub addr: &'a [AddrTerm],
    pub width: u32,
}

/// A write of one element, as the enclosing `Stmt::Store` describes it.
pub(crate) struct StoreOp<'a> {
    pub arg: ArgId,
    pub ty: MemType,
    pub addr: &'a [AddrTerm],
    pub value: VarId,
    pub width: u32,
    /// `(first index, axis)` when the vector may run past the axis extent.
    pub bound: Option<(VarId, AxisId)>,
}

/// How a backend reads and writes a binding.
///
/// Bounded by [`Dialect`] rather than being a supertrait of it: the printer is
/// generic over the dialect, and these two methods need a printer of *their
/// own* dialect to write into.
pub(crate) trait MemoryModel: Dialect + Sized {
    fn load(p: &mut Printer<'_, Self>, op: &LoadOp<'_>);
    fn store(p: &mut Printer<'_, Self>, op: &StoreOp<'_>);
}
