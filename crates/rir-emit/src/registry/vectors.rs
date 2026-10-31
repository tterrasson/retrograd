//! Claim test vectors: shapes, claims, and the verdict `Claim::holds` gives,
//! published for the C dispatcher to replay.
//!
//! The schedule table, the sweep and the planner evaluate a claim in Rust; the
//! dispatcher evaluates the same claim in C (`ggml_rir_variant_fits_shape`).
//! Nothing but these vectors ties the two together: the expected verdicts are
//! computed here by the Rust evaluator, never written by hand, and the fork's
//! selection selftest fails when its own answer differs.
//!
//! The vectors read the four ggml dimensions of the destination as four axes,
//! `d0` to `d3`, so a replay needs nothing but a probe tensor.

use rir_core::claim::Claim;

use super::{MAX_SHAPE_RULES, RegistryError, pad_rules, rule_initializer};

/// The axes a vector names, axis `i` reading dimension `i`.
const AXES: [&str; 4] = ["d0", "d1", "d2", "d3"];

/// Each case exercises one property of the arithmetic, at both sides of every
/// bound it states.
fn cases() -> Vec<(Claim, Vec<[u64; 4]>)> {
    let huge = 1u64 << 40;
    vec![
        // Inclusive bounds, both ends.
        (
            Claim::any().on("d1", 5..=32),
            vec![[1, 4, 1, 1], [1, 5, 1, 1], [1, 32, 1, 1], [1, 33, 1, 1]],
        ),
        // A product, not one extent.
        (
            Claim::any().product(&["d0", "d1", "d2"], ..=128),
            vec![[4, 8, 4, 1], [4, 8, 5, 1], [128, 1, 1, 7]],
        ),
        // A conjunction: each rule alone is not enough.
        (
            Claim::any().product(&["d1", "d2"], ..=32).on("d0", 256..),
            vec![
                [256, 4, 8, 1],
                [255, 4, 8, 1],
                [256, 4, 9, 1],
                [4096, 1, 1, 1],
            ],
        ),
        // A zero extent fails the rule that reads it, even with a zero bound.
        (
            Claim::any().on("d2", 0..=256),
            vec![[1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 256, 1], [1, 1, 257, 1]],
        ),
        // The product saturates at u32::MAX: an open rule still holds, a rule
        // bounded below it does not.
        (
            Claim::any().product(&["d0", "d1"], 2..),
            vec![[huge, huge, 1, 1], [1, 1, 1, 1]],
        ),
        (
            Claim::any().product(&["d0", "d1"], ..u32::MAX),
            vec![
                [huge, huge, 1, 1],
                [65_535, 65_537, 1, 1],
                [65_535, 65_535, 1, 1],
            ],
        ),
        // A rule over no axis compares the empty product, one.
        (Claim::any().product(&[], 1..=1), vec![[7, 7, 7, 7]]),
        (Claim::any().product(&[], 2..), vec![[7, 7, 7, 7]]),
        // No rule: every shape.
        (Claim::any(), vec![[1, 1, 1, 1], [huge, huge, huge, huge]]),
    ]
}

/// The `rir_shape_rule_vectors` table and its count.
pub(super) fn shape_rule_vectors() -> Result<String, RegistryError> {
    let mut rows = Vec::new();
    for (claim, shapes) in cases() {
        let subject = || RegistryError::new("rir_shape_rule_vectors", format!("{claim:?}"));
        if claim.rules().len() > MAX_SHAPE_RULES {
            return Err(subject());
        }
        let mut rules = Vec::new();
        for rule in claim.rules() {
            let idx = rule
                .axes
                .iter()
                .map(|name| AXES.iter().position(|a| a == name))
                .collect::<Option<Vec<usize>>>()
                .ok_or_else(subject)?;
            rules.push(rule_initializer(&idx, rule));
        }
        let n_rules = rules.len();
        pad_rules(&mut rules);
        for shape in shapes {
            let expected = claim.holds(|axis| {
                let i = AXES.iter().position(|a| *a == axis)?;
                Some(shape[i])
            });
            rows.push(format!(
                "    {{{{{}}}, {n_rules}, {{{}}}, {}}},\n",
                shape.map(|e| e.to_string()).join(", "),
                rules.join(", "),
                u8::from(expected),
            ));
        }
    }
    Ok(format!(
        "\n// Shapes, claims and the verdict the Rust evaluator gives: the dispatcher's\n\
         // selftest replays them through ggml_rir_variant_fits_shape.\n\
         const rir_shape_rule_vector rir_shape_rule_vectors[] = {{\n{}}};\n\
         const uint32_t rir_shape_rule_vector_count = {};\n",
        rows.concat(),
        rows.len()
    ))
}
