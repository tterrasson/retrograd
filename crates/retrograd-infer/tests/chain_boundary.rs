//! The dependency boundary of the inference-policy crate, checked rather
//! than intended:
//!
//! > `retrograd-infer` never depends on `retrograd-training`, `-agent`,
//! > `-dataset`, `-plan`, `-config` or `-run`.
//!
//! The scheduler and the actor run over a decode backend and know nothing
//! about algorithms, datasets, planning or run registries; the HTTP facade
//! crate that will sit on top of this one must not drag them in either. Same
//! shape of check as `rir-gen`'s `chain_boundary.rs`: read `cargo tree`, do
//! not trust the manifest.

use std::process::Command;

const FORBIDDEN: &[&str] = &[
    "retrograd-training",
    "retrograd-agent",
    "retrograd-dataset",
    "retrograd-plan",
    "retrograd-config",
    "retrograd-run",
];

#[test]
fn the_infer_crate_depends_on_no_training_side_crate() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "retrograd-infer",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    assert!(
        out.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let packages: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    // The tree is read, not guessed: an empty one would make the assertion
    // below vacuous.
    assert!(
        packages.contains(&"retrograd-infer"),
        "cargo tree printed no retrograd-infer"
    );
    assert!(
        packages.contains(&"retrograd-engine"),
        "the decode backends live in retrograd-engine: cargo tree printed no retrograd-engine"
    );
    let found: Vec<&&str> = FORBIDDEN
        .iter()
        .filter(|bad| packages.contains(bad))
        .collect();
    assert!(
        found.is_empty(),
        "retrograd-infer depends on {found:?}: the inference policy must stay free of the \
         training, agent, dataset, plan, config and run crates"
    );
}
