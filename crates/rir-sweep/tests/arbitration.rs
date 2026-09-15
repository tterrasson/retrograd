//! What the sweep refuses, pinned on measurements written here.
//!
//! These are the tests that matter for this crate, and the reason is the shape
//! of the tool: the *measuring* half of this tool is checked by running it - a wrong
//! time is visibly wrong - while the *deciding* half is a handful of
//! comparisons that would go on producing plausible-looking proposals while
//! being wrong. So each rule the sweep applies has a case here, and
//! each case is a set of rows a device could have produced.

use rir_sweep::measure::Sample;
use rir_sweep::verdict::{Measured, Refusal, Verdict, arbitrate};

#[test]
fn rule_products_match_the_dispatchers_saturating_u32_arithmetic() {
    assert_eq!(
        rir_sweep::candidates::rule_product(
            &["row", "plane", "batch"],
            [1, u32::MAX as usize, 2, 2],
        ),
        u64::from(u32::MAX)
    );
}

fn sample(us: f64, spread: f64) -> Sample {
    Sample {
        median_us: us,
        spread,
        groups: [1, 1, 1],
    }
}

/// `base` µs against `candidate` µs on `shape`, both measured cleanly (1 %
/// spread, so the 2 % floor is what applies).
fn row(shape: [usize; 4], base: f64, candidate: f64) -> Measured {
    Measured {
        shape,
        base: sample(base, 0.01),
        candidate: sample(candidate, 0.01),
        agrees: true,
    }
}

/// A candidate faster everywhere takes the fallback's place rather than a rule.
/// A shape rule here would claim less than the measurement, and the pair would
/// keep a lowering it never prefers.
#[test]
fn a_candidate_that_never_loses_replaces_the_fallback() {
    let rows = [
        row([1024, 16, 1, 1], 100.0, 50.0),
        row([128, 16, 16, 1], 100.0, 99.0),
    ];
    match arbitrate(&rows) {
        Verdict::Replaces { best_gain } => assert!((best_gain - 1.0).abs() < 1e-9),
        other => panic!("{other:?}"),
    }
}

/// The rule that rejects gains inside measurement noise: a candidate ahead by less
/// than the two measurements' own spread has not been measured to be ahead.
///
/// The same numbers with a tenth of the spread would be a win, which is what
/// makes this a measurement and not a threshold: nothing here knows what 4 %
/// means, only what the device reproduced.
#[test]
fn a_gain_inside_the_spread_is_not_a_gain() {
    let noisy = [Measured {
        shape: [1024, 16, 1, 1],
        base: sample(100.0, 0.20),
        candidate: sample(96.0, 0.20),
        agrees: true,
    }];
    assert!(matches!(
        arbitrate(&noisy),
        Verdict::Refused(Refusal::NoGain { .. })
    ));

    let clean = [row([1024, 16, 1, 1], 100.0, 96.0)];
    assert!(matches!(arbitrate(&clean), Verdict::Replaces { .. }));
}

/// The other half rejects candidates without a distinct winning domain: the
/// rule that claims the wins must exclude every loss, and one that cannot is
/// refused instead of shipped.
///
/// Here the candidate wins on 16 rows and on 256 rows and loses on 128, which no
/// interval over a row count separates, and the row lengths do not separate them
/// either.
#[test]
fn a_domain_no_rule_separates_is_refused() {
    let rows = [
        row([512, 16, 1, 1], 100.0, 50.0),
        row([512, 128, 1, 1], 100.0, 200.0),
        row([512, 256, 1, 1], 100.0, 50.0),
    ];
    match arbitrate(&rows) {
        Verdict::Refused(Refusal::NoDistinctDomain { overclaimed }) => {
            assert_eq!(overclaimed, vec![[512, 128, 1, 1]]);
        }
        other => panic!("{other:?}"),
    }
}

/// A domain a rule does separate: the wins are the few-row shapes, the loss is
/// the many-row one, and the bound is stated **between** them rather than at
/// the last shape measured - which is what keeps a sweep of five
/// shapes from claiming to know where a crossover is to the unit.
#[test]
fn the_bound_of_a_derived_rule_sits_in_the_gap() {
    let rows = [
        row([1024, 16, 1, 1], 100.0, 50.0),
        row([1024, 256, 1, 1], 100.0, 200.0),
    ];
    match arbitrate(&rows) {
        Verdict::Claims { rules, .. } => {
            assert_eq!(rules.len(), 1);
            assert_eq!(rules[0].axes, &["row", "plane", "batch"]);
            assert_eq!(rules[0].min, 1);
            // Strictly between the 16 rows that win and the 256 that lose, and
            // round: 128 is what the gap (16, 256) offers.
            assert_eq!(rules[0].max, 128);
        }
        other => panic!("{other:?}"),
    }
}

/// Two axis groups, one rule each, and the conjunction is what excludes the
/// losses - the shape `few_wide_rows` has in the production table: at most so
/// many rows **and** at least so many columns.
#[test]
fn a_conjunction_is_derived_when_one_rule_does_not_separate() {
    let rows = [
        // Wins: few rows, long rows.
        row([1024, 16, 1, 1], 100.0, 50.0),
        row([1024, 32, 1, 1], 100.0, 50.0),
        // Loses: as few rows, but short ones - and as long rows, but many.
        row([64, 16, 1, 1], 100.0, 200.0),
        row([1024, 512, 1, 1], 100.0, 200.0),
    ];
    match arbitrate(&rows) {
        Verdict::Claims { rules, .. } => {
            assert_eq!(rules.len(), 2);
            let rows_rule = rules.iter().find(|r| r.axes.contains(&"row")).unwrap();
            let cols_rule = rules.iter().find(|r| r.axes == ["col"]).unwrap();
            assert!(rows_rule.max >= 32 && rows_rule.max < 512);
            assert!(cols_rule.min > 64 && cols_rule.min <= 1024);
        }
        other => panic!("{other:?}"),
    }
}

/// Speed is not the first question. A candidate that does not return what the
/// lowering it would replace returns is refused before a time is read, however
/// fast it is - the one proposal a sweep must never make.
#[test]
fn a_faster_candidate_that_disagrees_is_refused_first() {
    let mut rows = [
        row([1024, 16, 1, 1], 100.0, 10.0),
        row([64, 16, 1, 1], 100.0, 10.0),
    ];
    rows[1].agrees = false;
    assert_eq!(
        arbitrate(&rows),
        Verdict::Refused(Refusal::Disagrees {
            shape: [64, 16, 1, 1]
        })
    );
}

/// Nothing measured is its own answer, and not an empty proposal: a kernel the
/// harness could not bind on any shape must not read as a kernel with nothing
/// to gain.
#[test]
fn no_rows_is_not_a_refusal_for_lack_of_gain() {
    assert_eq!(arbitrate(&[]), Verdict::Refused(Refusal::NotMeasured));
}
