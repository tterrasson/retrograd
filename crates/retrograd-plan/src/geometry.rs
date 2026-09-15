//! Candidate `(n_ctx, n_batch, n_ubatch)` triples that the
//! runtime actually accepts.
//!
//! The divisibilities are enforced in three places already - the C++ runtime
//! (`retro_runtime.cpp:161-166`), `retrograd-config`, and here. This one exists
//! so the resolver never *proposes* a triple the other two would refuse: a 422
//! that says "the resolver produced an invalid config" is a bug report, not an
//! answer.

use serde::Serialize;

/// A candidate batch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Geometry {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
}

impl Geometry {
    /// The two hard divisibilities, checked as the runtime checks them.
    pub fn is_valid(self) -> bool {
        self.n_ctx > 0
            && self.n_batch > 0
            && self.n_ubatch > 0
            && self.n_ctx.is_multiple_of(self.n_batch)
            && self.n_batch.is_multiple_of(self.n_ubatch)
    }

    /// Optimizer steps one full context row produces.
    pub fn steps_per_row(self) -> u64 {
        ((self.n_ctx / self.n_batch.max(1)) as u64).max(1)
    }
}

/// What constrains the enumeration beyond the divisibilities.
#[derive(Clone, Copy, Debug)]
pub struct Constraints {
    pub n_ctx: u32,
    /// GRPO packs `n_seq_max = group_size` sequences, and the runtime refuses
    /// `n_seq_max > n_batch` - so a group size is a floor on `n_batch`.
    pub min_batch: u32,
    /// A rollout algorithm takes one optimizer step per `n_batch` tokens of
    /// every row, and a row is one rollout - so anything below `n_ctx` turns a
    /// single completion into several policy steps and `retrograd-config`
    /// refuses it. The enumeration then has exactly one admissible batch,
    /// whatever `n_ctx` phase 3 settles on.
    pub whole_row_batch: bool,
    /// A field the caller overrode is never re-derived.
    pub fixed_batch: Option<u32>,
    pub fixed_ubatch: Option<u32>,
}

impl Constraints {
    pub fn new(n_ctx: u32) -> Self {
        Self {
            n_ctx,
            min_batch: 1,
            whole_row_batch: false,
            fixed_batch: None,
            fixed_ubatch: None,
        }
    }
}

/// Every geometry satisfying the constraints, largest batch first.
///
/// "Largest first" is the phase-2 rule: keep the biggest one whose estimate
/// fits the budget. Throughput is what a larger batch buys, so the resolver
/// takes it while the budget allows and phase 3 takes it back if it does not.
pub fn candidates(constraints: Constraints) -> Vec<Geometry> {
    let n_ctx = constraints.n_ctx;
    if n_ctx == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for n_batch in divisors(n_ctx) {
        if n_batch < constraints.min_batch {
            continue;
        }
        if constraints.whole_row_batch && n_batch != n_ctx {
            continue;
        }
        if constraints
            .fixed_batch
            .is_some_and(|fixed| fixed != n_batch)
        {
            continue;
        }
        for n_ubatch in divisors(n_batch) {
            if constraints
                .fixed_ubatch
                .is_some_and(|fixed| fixed != n_ubatch)
            {
                continue;
            }
            out.push(Geometry {
                n_ctx,
                n_batch,
                n_ubatch,
            });
        }
    }
    // Largest batch first, then largest ubatch. Total order, so two identical
    // resolutions enumerate identically (invariant 3).
    out.sort_by(|left, right| {
        right
            .n_batch
            .cmp(&left.n_batch)
            .then(right.n_ubatch.cmp(&left.n_ubatch))
    });
    out
}

/// Divisors of `value`, ascending. Bounded by construction: `n_ctx` is a
/// resolver-chosen power of two in the common case, and a few thousand at worst.
///
/// Shared with the candidate search: both enumerate the same physical widths, and
/// two different divisor functions would eventually disagree about which
/// `(batch, ubatch)` pairs exist.
pub(crate) fn divisors(value: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut candidate = 1;
    while candidate <= value {
        if value.is_multiple_of(candidate) {
            out.push(candidate);
        }
        candidate += 1;
    }
    out
}

/// Rounds a token count up to a power of two, which is what phase 1 uses for
/// `n_ctx`: it makes the divisibilities satisfiable by many `(batch, ubatch)`
/// pairs instead of by the few divisors of an arbitrary number.
pub fn round_up_pow2(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    let mut result = 1u32;
    while result < value {
        let Some(next) = result.checked_mul(2) else {
            return u32::MAX;
        };
        result = next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_candidate_satisfies_the_runtime_divisibilities() {
        for n_ctx in [64u32, 128, 512, 2048, 3000] {
            let candidates = candidates(Constraints::new(n_ctx));
            assert!(!candidates.is_empty(), "no geometry for n_ctx={n_ctx}");
            for candidate in candidates {
                assert!(candidate.is_valid(), "{candidate:?}");
                assert_eq!(candidate.n_ctx, n_ctx);
            }
        }
    }

    #[test]
    fn the_largest_batch_comes_first() {
        let candidates = candidates(Constraints::new(512));
        assert_eq!(candidates[0].n_batch, 512);
        assert_eq!(candidates[0].n_ubatch, 512);
        assert_eq!(candidates.last().unwrap().n_batch, 1);
    }

    #[test]
    fn a_group_size_floors_the_batch() {
        let constraints = Constraints {
            min_batch: 64,
            ..Constraints::new(512)
        };
        for candidate in candidates(constraints) {
            assert!(candidate.n_batch >= 64, "{candidate:?}");
        }
    }

    #[test]
    fn an_overridden_batch_is_the_only_batch_offered() {
        let constraints = Constraints {
            fixed_batch: Some(128),
            ..Constraints::new(512)
        };
        for candidate in candidates(constraints) {
            assert_eq!(candidate.n_batch, 128);
        }
        // An override the context cannot honour yields nothing, which is the
        // conflict phase 2 has to report rather than silently move.
        let impossible = Constraints {
            fixed_batch: Some(300),
            ..Constraints::new(512)
        };
        assert!(candidates(impossible).is_empty());
    }

    #[test]
    fn rounding_up_reaches_the_next_power_of_two() {
        assert_eq!(round_up_pow2(0), 1);
        assert_eq!(round_up_pow2(1), 1);
        assert_eq!(round_up_pow2(513), 1024);
        assert_eq!(round_up_pow2(1024), 1024);
    }
}
