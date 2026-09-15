//! Carrying user-supplied sizes into the 64-bit arithmetic of a budget.
//!
//! Most `as` in this repo widens - `u32` to `u64` before a byte product, an
//! integer to `f32` for a metric - and widening *before* the arithmetic is the
//! defence against overflow, not a risk to be converted away. What this module
//! is for is the boundary where a size or a product of sizes, computed from
//! numbers a user supplied, enters a `u64` cost estimate.
//!
//! There, `as u32` is not a conversion but a **truncation**, and the direction
//! it fails in is what makes it dangerous. A workload of `2^32 + 8` rollouts
//! truncates to `8`, the cost model prices eight rollouts, and the planner
//! accepts a run that cannot fit. The number did not become wrong by a little;
//! it became wrong in the direction that removes the refusal.
//!
//! [`saturating_dim`] and [`saturating_dim_product`] preserve that count in the
//! `u64` used by cost estimates, saturating only if the real value exceeds what
//! the estimate can represent. They are deliberately **not** fallible: their
//! callers are cost and capacity estimates, which have no error channel and
//! need none - a saturated estimate is refused by the budget it feeds, which is
//! the outcome an error would have produced anyway, reached without threading
//! a `Result` through the cost model. Where the value feeds a *contract* rather
//! than a budget - a kernel geometry, a wire field, a table index - saturating
//! would be wrong and the caller must return its own typed error instead; see
//! `docs/engineering/CONVERSIONS.md` for the whole rule.

/// A size as a `u64`, saturating at `u64::MAX` rather than truncating.
///
/// For values that feed a cost or capacity estimate: saturating keeps an
/// oversized workload oversized, so the budget downstream refuses it.
pub fn saturating_dim(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// The product of two sizes as a `u64`, saturating at `u64::MAX`.
///
/// The multiplication saturates too, and that is the point of the function
/// existing beside [`saturating_dim`]: `saturating_dim(a * b)` would compute
/// the product in `usize` first, where release builds wrap silently, and hand
/// a small number to a conversion that then has nothing left to saturate.
pub fn saturating_dim_product(a: usize, b: usize) -> u64 {
    saturating_dim(a).saturating_mul(saturating_dim(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sizes_pass_through() {
        assert_eq!(saturating_dim(0), 0);
        assert_eq!(saturating_dim(1024), 1024);
        assert_eq!(saturating_dim_product(3, 7), 21);
    }

    #[test]
    fn values_above_u32_are_preserved() {
        assert_eq!(saturating_dim(u32::MAX as usize), u64::from(u32::MAX));
        assert_eq!(
            saturating_dim(u32::MAX as usize + 1),
            u64::from(u32::MAX) + 1
        );
    }

    /// The case the module exists for: the truncating form turns an impossible
    /// workload into a small one, which is what makes it a silent bug rather
    /// than a wrong number.
    #[test]
    fn saturation_is_not_truncation() {
        let oversized = u32::MAX as usize + 9;
        assert_eq!(oversized as u32, 8, "the cast under replacement");
        assert_eq!(saturating_dim(oversized), u64::from(u32::MAX) + 9);
    }

    /// The operands widen before the product: on a 32-bit target the exact
    /// product can exceed `usize::MAX` while still fitting in `u64`; on a
    /// 64-bit target it saturates at the estimate's own limit.
    #[test]
    fn the_product_widens_before_multiplying_and_saturates_only_at_u64() {
        let (a, b) = (usize::MAX, 2usize);
        let exact = (a as u128) * (b as u128);
        let expected = u64::try_from(exact).unwrap_or(u64::MAX);
        assert_eq!(saturating_dim_product(a, b), expected);
    }
}
