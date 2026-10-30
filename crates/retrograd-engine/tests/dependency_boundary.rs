//! The dependency boundary of the engine crate, checked rather than intended:
//!
//! > The inference path of `retrograd-engine` does not pull `retrograd-dataset`.
//!
//! The SFT methods that take a `PreparedDataset` and the `DatasetBackend` impl
//! for `Trainer` sit behind the `train` feature. An inference-only consumer
//! builds the crate with `--no-default-features` and must not get the dataset
//! crate - the same shape of check as `rir-gen`'s `chain_boundary.rs`, reading
//! `cargo tree` rather than trusting the manifest.

use std::process::Command;

#[test]
fn the_inference_build_of_the_engine_crate_has_no_dataset_dependency() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "retrograd-engine",
            "--no-default-features",
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
        packages.contains(&"retrograd-engine"),
        "cargo tree printed no retrograd-engine"
    );
    assert!(
        !packages.contains(&"retrograd-dataset"),
        "retrograd-dataset reached the --no-default-features build of retrograd-engine: \
         the SFT surface must stay behind the `train` feature"
    );
}
