//! The one place a `usize` becomes a 32-bit identifier.
//!
//! `ValueId`, `ArgId`, `ParamId`, `AxisId` and `rir-lower`'s `VarId` are `u32`
//! newtypes. Every conversion goes through the checked helpers below, which
//! make the representability limit explicit.
//!
//! Two entry points on purpose:
//!
//! - [`IrId::at`] is what the code calls. It panics past the limit, with the
//!   kind named, because an allocation the host cannot make is not a kernel
//!   limit and the crate's "every limit is an explicit error" rule is
//!   about the latter. A `Result` here would put a `?` on every builder method
//!   and change the DSL for a case no caller can reach;
//! - [`IrId::try_at`] is the same conversion, checked, and it is what makes
//!   the limit **testable** without allocating 64 GiB. That is its whole reason
//!   for existing: a panic nobody can trigger is a claim nobody can check.

/// The largest index an identifier can carry. One identifier is one `u32`, so
/// this is the width of that type and not a policy of its own.
pub const MAX_IDS: usize = u32::MAX as usize;

/// A 32-bit identifier into one of a kernel's tables.
///
/// Implemented by `ValueId`, `ArgId`, `ParamId`, `AxisId` in this crate and by
/// `rir_lower::VarId`, which is why the trait is public: the Loop IR mints its
/// own variables and must not reintroduce a bare cast to do it.
pub trait IrId: Copy + Sized {
    /// What this identifier indexes, for the panic message. `"value"`, `"arg"`,
    /// … - a reader of a backtrace should not have to know which table
    /// overflowed from the type name alone.
    const KIND: &'static str;

    fn from_raw(raw: u32) -> Self;

    fn raw(self) -> u32;

    /// The identifier of the element at `index`, or `None` past [`MAX_IDS`].
    fn try_at(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self::from_raw)
    }

    /// The identifier of the element at `index` - the position it holds now, or
    /// the position it is about to hold when `index` is a length.
    ///
    /// # Panics
    ///
    /// Past [`MAX_IDS`], naming [`IrId::KIND`]. Reaching it means the host
    /// already holds more than `u32::MAX` nodes of that kind, which is tens of
    /// gigabytes for the smallest of them; see the module docs for why this is a
    /// panic and not an error variant.
    fn at(index: usize) -> Self {
        match Self::try_at(index) {
            Some(id) => id,
            None => panic!(
                "{} identifiers are 32-bit: index {index} exceeds {MAX_IDS}",
                Self::KIND
            ),
        }
    }
}

/// Implements [`IrId`] for a `u32` newtype.
///
/// A macro rather than five hand-written impls: the point of this module is that
/// the conversion exists once, and five copies of `fn from_raw` would be five
/// places to get it wrong.
#[macro_export]
macro_rules! impl_ir_id {
    ($ty:ty, $kind:literal) => {
        impl $crate::ids::IrId for $ty {
            const KIND: &'static str = $kind;

            fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgId, AxisId, ParamId, ValueId};

    /// The round trip, on every identifier of this crate: `at` is a position and
    /// `raw` gives it back.
    #[test]
    fn an_identifier_carries_the_index_it_was_minted_from() {
        assert_eq!(ValueId::at(0).raw(), 0);
        assert_eq!(ArgId::at(3).raw(), 3);
        assert_eq!(ParamId::at(7).raw(), 7);
        assert_eq!(AxisId::at(MAX_IDS).raw(), u32::MAX);
    }

    /// The limit itself, which is the whole reason `try_at` exists next to `at`:
    /// the panicking path cannot be reached without allocating the table, so what
    /// is tested is the conversion, at the exact boundary.
    ///
    /// `MAX_IDS + 1` is only a representable index where a `usize` is wider
    /// than a `u32`: on a 32-bit target `MAX_IDS == usize::MAX` and the
    /// addition overflows before the refusal can be observed. The accepting
    /// half of the boundary holds on every target, so it is not conditioned.
    #[test]
    fn the_limit_is_the_width_of_the_identifier() {
        assert!(ValueId::try_at(MAX_IDS).is_some());
        #[cfg(target_pointer_width = "64")]
        assert!(ValueId::try_at(MAX_IDS + 1).is_none());
    }

    /// The panic names its table. A backtrace that says "index too large" and
    /// not *for what* costs a reader the only thing the message had to give.
    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "axis identifiers are 32-bit")]
    fn the_panic_names_the_table_that_overflowed() {
        AxisId::at(MAX_IDS + 1);
    }
}
