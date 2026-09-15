//! `CLAUDE.md`'s rule for the RIR chain, checked rather than intended:
//!
//! > A `rir-*` crate never depends on `retrograd-core`.
//!
//! It was broken at exactly one place, and at the only place it could be: the
//! *schema* of the kernel catalogue lived in `retrograd-core`, so `rir-gen`,
//! the crate that **writes** the catalogue - had to depend on the applicative
//! facade to name what it was writing. The schema now
//! lives in `rir_core::catalog`, which both ends read.
//!
//! The test walks every `rir-*` crate rather than just this one: the rule is
//! about the chain, and the next violation will not be in the crate that
//! carried the last one.

use std::process::Command;

/// Crates from the applicative chain. Any of them under a `rir-*` tree means
/// the arrow has been reversed: the RIR chain is compiled by `rir-gen` at build
/// time and linked into no runtime binary, so nothing it needs can live behind
/// a type that loads a model.
const APPLICATIVE: &[&str] = &["retrograd-core", "retrograd-engine", "retrograd-plan"];

const RIR_CRATES: &[&str] = &[
    "rir-core",
    "rir-lower",
    "rir-emit",
    "rir-kernels",
    "rir-gen",
    "rir-runtime",
    "rir-sweep",
];

fn tree(package: &str) -> Vec<String> {
    let out = Command::new(env!("CARGO"))
        .args(["tree", "-p", package, "-e", "normal", "--prefix", "none"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    assert!(
        out.status.success(),
        "cargo tree -p {package} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

#[test]
fn the_rir_chain_depends_on_nothing_applicative() {
    for package in RIR_CRATES {
        let packages = tree(package);
        // The tree is read, not guessed: an empty one would make the assertion
        // below vacuous.
        assert!(
            packages.iter().any(|name| name == package),
            "cargo tree -p {package} printed no {package}"
        );
        let found: Vec<&&str> = APPLICATIVE
            .iter()
            .filter(|bad| packages.iter().any(|name| name == *bad))
            .collect();
        assert!(
            found.is_empty(),
            "{package} depends on {found:?} - a shared schema has drifted back into the \
             applicative side; it belongs in rir-core, which both chains can read"
        );
    }
}

/// The other half of the same rule: `retrograd-core` may read `rir-core`, and
/// that is the direction the catalogue moved in. Stated here, next to the
/// prohibition, because a reader who finds only the prohibition will conclude
/// the two chains must not meet at all - they meet, in one leaf crate, on
/// purpose.
#[test]
fn the_applicative_side_reads_the_leaf_and_nothing_else_of_the_chain() {
    let packages = tree("retrograd-core");
    assert!(packages.iter().any(|name| name == "rir-core"));
    for forbidden in ["rir-lower", "rir-emit", "rir-kernels", "rir-gen"] {
        assert!(
            !packages.iter().any(|name| name == forbidden),
            "retrograd-core depends on {forbidden}: the applicative side reads the published \
             schema, never the compiler that produces it"
        );
    }
}
