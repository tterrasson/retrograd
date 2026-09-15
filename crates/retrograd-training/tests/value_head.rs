//! The PPO critic's public contract, exercised from outside the crate.
//!
//! `retrograd-training` had no `tests/` directory at all: everything was an
//! inline module, where access to private state hides a broken public API
//!  This file sees exactly what a caller sees,
//! `new`, `predict`, `predict_rows`, `fit` - and nothing else.
//!
//! `ValueHead` is the one part of the crate that needs neither a model nor a
//! device, which is what lets it run in the fast lane. The rest needs a model, and is
//! covered by the model lanes; it is not an oversight here.

use retrograd_training::value::ValueHead;

#[test]
fn a_fresh_head_predicts_zero_so_the_first_update_is_unbiased() {
    let head = ValueHead::new(3);
    assert_eq!(head.predict(&[1.0, -2.0, 3.5]), 0.0);
    assert_eq!(
        head.predict_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [0.0, 0.0]
    );
}

#[test]
fn fitting_a_linear_target_reduces_the_error_it_reports() {
    let mut head = ValueHead::new(2);
    // v = 2·x0 - x1 + 1, exactly representable by the probe.
    let features = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let targets = [1.0, 3.0, 0.0, 2.0];

    let before = head.fit(&features, &targets, 0.05, 1).expect("one epoch");
    let after = head
        .fit(&features, &targets, 0.05, 400)
        .expect("four hundred more");

    assert!(after < before, "{after} is not below {before}");
    assert!(
        after < 1.0e-3,
        "the probe did not reach its own target: {after}"
    );
    for (row, expected) in features.as_chunks::<2>().0.iter().zip(targets) {
        assert!((head.predict(row) - expected).abs() < 0.05, "{row:?}");
    }
}

#[test]
fn a_fit_that_cannot_mean_anything_is_refused_rather_than_run() {
    let mut head = ValueHead::new(2);
    let features = [1.0, 2.0];

    // One row of two features against two targets: the caller has mixed up its
    // shapes, and averaging over a mismatched batch would silently train on
    // nothing recognizable.
    assert!(head.fit(&features, &[1.0, 2.0], 0.1, 1).is_err());
    assert!(head.fit(&features, &[], 0.1, 1).is_err());
    assert!(head.fit(&features, &[1.0], 0.0, 1).is_err());
    assert!(head.fit(&features, &[1.0], f32::NAN, 1).is_err());
}
