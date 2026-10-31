//! What a measured candidate earns: a row, the fallback's place, or a refusal.
//!
//! Pure arithmetic over measured rows - no device, no lowering - so the rules
//! the sweep applies can be tested against shapes rather than argued about:
//!
//! > refuse a variant without a distinct domain or a gain above the noise.
//!
//! Both halves of that sentence are a *comparison against a measurement*, and
//! neither is a constant chosen here. The noise is the run-to-run spread of the
//! two medians being compared ([`Sample::spread`]); the domain is a
//! `Claim` - the only predicate the registry publishes - and it has to
//! separate the shapes the candidate wins on from the ones it loses on
//! *exactly*. A rule that also claims a shape where the candidate loses is an
//! over-claim, which is a failure corrected by hand once seen before: `≤ 64`
//! claimed one shape too many, and the pair stayed in `observe` because of it.

use rir_lower::{Claim, ShapeRule};

use crate::candidates::{AXES, RULE_AXES, rule_product};
use crate::measure::Sample;

/// One shape, timed on both lowerings.
#[derive(Clone, Copy, Debug)]
pub struct Measured {
    pub shape: [usize; 4],
    pub base: Sample,
    pub candidate: Sample,
    /// Whether the candidate returned the same bytes as the lowering it would
    /// replace, within the tolerance of `agreement`.
    pub agrees: bool,
}

impl Measured {
    /// Relative gain of the candidate over the base: `0.25` is "a quarter
    /// faster", negative is slower.
    pub fn gain(&self) -> f64 {
        if self.candidate.median_us <= 0.0 {
            return 0.0;
        }
        self.base.median_us / self.candidate.median_us - 1.0
    }

    /// The noise floor of *this* comparison: the two spreads, added, because a
    /// ratio inherits both. Never below a floor of 2 %, which is not a
    /// measurement but a refusal to believe a Vulkan host timing to the
    /// third digit - the number this harness produces includes one submission
    /// and one fence wait divided by `iters`.
    pub fn noise(&self) -> f64 {
        (self.base.spread + self.candidate.spread).max(0.02)
    }

    /// Wins, loses, or neither.
    pub fn outcome(&self) -> Outcome {
        let (g, n) = (self.gain(), self.noise());
        if g > n {
            Outcome::Wins
        } else if g < -n {
            Outcome::Loses
        } else {
            Outcome::Tied
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Wins,
    Loses,
    /// Inside the noise of the two measurements. Neither claimed nor refused:
    /// a rule that happens to cover it costs nothing, and one that does not
    /// loses nothing.
    Tied,
}

/// Why a candidate earns nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// The candidate did not compute what the lowering it would replace
    /// computes. Checked before any time is read, and fatal: a faster wrong
    /// kernel is the one thing a sweep must never propose.
    Disagrees { shape: [usize; 4] },
    /// It never got outside the noise on any shape.
    NoGain { best_gain: f64, noise: f64 },
    /// It wins somewhere and loses somewhere, and no `Claim` over the
    /// registry's axis groups separates the two sets. The shapes listed are the
    /// losses the closest rule would have claimed.
    NoDistinctDomain { overclaimed: Vec<[usize; 4]> },
    /// Nothing was measured: no shape survived building or binding.
    NotMeasured,
}

/// What the sweep proposes for one candidate.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// It wins somewhere and loses nowhere: it does not need a rule, it needs
    /// the fallback's place. A shape rule here would be a claim narrower than
    /// the measurement, which is the mistake in the other direction from an
    /// over-claim - the pair would keep a lowering it never prefers.
    Replaces {
        best_gain: f64,
    },
    /// It wins on a domain a claim separates.
    Claims {
        claim: Claim,
        best_gain: f64,
        /// Shapes the rule claims and the candidate only ties on. Not a
        /// problem - stated so a reader can see what the rule covers beyond
        /// what it won.
        ties_claimed: Vec<[usize; 4]>,
    },
    Refused(Refusal),
}

/// The verdict for one candidate from its measured rows.
pub fn arbitrate(rows: &[Measured]) -> Verdict {
    arbitrate_with_rule_limit(rows, rir_emit::registry::MAX_SHAPE_RULES)
}

/// Reserve room for rules the caller adds after deriving the measured domain.
pub(crate) fn arbitrate_with_rule_limit(rows: &[Measured], rule_limit: usize) -> Verdict {
    if rows.is_empty() {
        return Verdict::Refused(Refusal::NotMeasured);
    }
    if let Some(bad) = rows.iter().find(|r| !r.agrees) {
        return Verdict::Refused(Refusal::Disagrees { shape: bad.shape });
    }

    let wins: Vec<&Measured> = rows
        .iter()
        .filter(|r| r.outcome() == Outcome::Wins)
        .collect();
    let losses: Vec<&Measured> = rows
        .iter()
        .filter(|r| r.outcome() == Outcome::Loses)
        .collect();
    let best_gain = rows.iter().map(Measured::gain).fold(f64::MIN, f64::max);

    if wins.is_empty() {
        return Verdict::Refused(Refusal::NoGain {
            best_gain,
            noise: rows.iter().map(Measured::noise).fold(f64::MAX, f64::min),
        });
    }
    if losses.is_empty() {
        return Verdict::Replaces { best_gain };
    }

    match separate(&wins, &losses, rule_limit) {
        Some(claim) => {
            let ties_claimed = rows
                .iter()
                .filter(|r| r.outcome() == Outcome::Tied && claims(&claim, r.shape))
                .map(|r| r.shape)
                .collect();
            Verdict::Claims {
                claim,
                best_gain,
                ties_claimed,
            }
        }
        None => Verdict::Refused(Refusal::NoDistinctDomain {
            overclaimed: losses.iter().map(|r| r.shape).collect(),
        }),
    }
}

/// Whether a claim holds on a `[col, row, plane, batch]` shape, evaluated as
/// the dispatcher evaluates it.
pub fn claims(claim: &Claim, shape: [usize; 4]) -> bool {
    claim.holds(|axis| {
        let i = AXES.iter().position(|a| *a == axis)?;
        u64::try_from(shape[i]).ok()
    })
}

/// The smallest conjunction of rules over [`RULE_AXES`] that claims every win
/// and no loss, or `None`.
///
/// Single rules first, then the conjunction of two, because a rule a reader has
/// to hold two intervals in their head to evaluate is worth writing only when
/// one does not do the job - the order the table's own arms are written in.
fn separate(wins: &[&Measured], losses: &[&Measured], rule_limit: usize) -> Option<Claim> {
    if rule_limit == 0 {
        return None;
    }
    let single: Vec<Option<ShapeRule>> = RULE_AXES
        .iter()
        .map(|axes| interval(axes, wins, losses))
        .collect();
    for rule in single.iter().flatten() {
        let claim = Claim(vec![rule.clone()]);
        if losses.iter().all(|l| !claims(&claim, l.shape)) {
            return Some(claim);
        }
    }
    if rule_limit < 2 {
        return None;
    }
    let available: Vec<ShapeRule> = single.into_iter().flatten().collect();
    for (i, first) in available.iter().enumerate() {
        for second in available.iter().skip(i + 1) {
            let pair = Claim(vec![first.clone(), second.clone()]);
            if losses.iter().all(|l| !claims(&pair, l.shape)) {
                return Some(pair);
            }
        }
    }
    // The registry holds at most two rules. A conjunction of all three axis
    // groups could separate the measurements but could never be published.
    None
}

/// The interval over one axis group that covers every win, with each bound
/// pushed out to a round number inside the gap that separates it from the
/// nearest loss.
///
/// The rounding follows a simple rule, mechanized: "nothing in the
/// bench separates 32 from 60, so the bound is stated at the round number
/// inside that gap rather than fitted to a shape". A bound fitted to the
/// extreme measured shape claims to know where the crossover is to the unit,
/// and a sweep of five shapes knows no such thing.
fn interval(axes: &[&str], wins: &[&Measured], losses: &[&Measured]) -> Option<ShapeRule> {
    let w: Vec<u64> = wins.iter().map(|m| rule_product(axes, m.shape)).collect();
    let (lo, hi) = (*w.iter().min()?, *w.iter().max()?);
    let below = losses
        .iter()
        .map(|m| rule_product(axes, m.shape))
        .filter(|p| *p < lo)
        .max();
    let above = losses
        .iter()
        .map(|m| rule_product(axes, m.shape))
        .filter(|p| *p > hi)
        .min();
    let min = match below {
        Some(b) => round_between(b + 1, lo),
        None => 1,
    };
    let max = match above {
        Some(a) => round_between(hi, a - 1),
        None => u64::from(u32::MAX),
    };
    let (min, max) = (u32::try_from(min).ok()?, u32::try_from(max).ok()?);
    Some(ShapeRule::new(axes, min..=max))
}

/// The **roundest** number inside `[lo, hi]`: the one with the most trailing
/// binary zeros, decimal decades among the candidates, `hi` when the gap holds
/// nothing rounder than its own ends.
///
/// This is the same rounding rule, mechanized: "nothing in the bench
/// separates 32 from 60, so the bound is stated at the round number inside that
/// gap rather than fitted to a shape". Sixteen rows win and sixty-four lose;
/// the bound this returns for that gap is 32, which is the bound the table
/// carries.
fn round_between(lo: u64, hi: u64) -> u64 {
    if lo > hi {
        return hi;
    }
    let mut candidates: Vec<u64> = Vec::new();
    let mut p = 1u64;
    while p <= hi {
        candidates.push(p);
        match p.checked_mul(2) {
            Some(n) => p = n,
            None => break,
        }
    }
    let mut d = 1u64;
    while d <= hi {
        for m in [1u64, 2, 5] {
            candidates.push(d * m);
        }
        match d.checked_mul(10) {
            Some(n) => d = n,
            None => break,
        }
    }
    candidates
        .into_iter()
        .filter(|v| *v >= lo && *v <= hi)
        .max_by_key(|v| (v.trailing_zeros(), *v))
        .unwrap_or(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserving_a_rule_for_contraction_depth_refuses_a_two_rule_domain() {
        let row = |shape, us| Measured {
            shape,
            base: Sample {
                median_us: 100.0,
                spread: 0.01,
                groups: [1, 1, 1],
            },
            candidate: Sample {
                median_us: us,
                spread: 0.01,
                groups: [1, 1, 1],
            },
            agrees: true,
        };
        let rows = [
            row([1024, 16, 1, 1], 50.0),
            row([64, 16, 1, 1], 200.0),
            row([1024, 512, 1, 1], 200.0),
        ];
        assert!(matches!(arbitrate(&rows), Verdict::Claims { .. }));
        assert!(matches!(
            arbitrate_with_rule_limit(&rows, rir_emit::registry::MAX_SHAPE_RULES - 1),
            Verdict::Refused(Refusal::NoDistinctDomain { .. })
        ));
    }
}
