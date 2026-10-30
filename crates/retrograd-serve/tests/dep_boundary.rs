//! The dependency boundary of the standalone inference binary, checked rather
//! than intended:
//!
//! > `retrograd-serve` (and with it the `retrograd-infer` binary) never depends
//! > on `retrograd-agent`, `retrograd-training`, `retrograd-run`,
//! > `retrograd-config`, `retrograd-env`, `retrograd-judge`,
//! > `retrograd-llm-client`, `rmcp` or `bollard`.
//!
//! The binary serves one model over HTTP; nothing in it trains, runs an agent
//! or parses a run configuration, so those crates must stay out of its tree.
//! Same shape of check as `retrograd-infer`'s `chain_boundary.rs`: read
//! `cargo tree`, do not trust the manifest.

use std::process::Command;

const FORBIDDEN: &[&str] = &[
    "retrograd-agent",
    "retrograd-training",
    "retrograd-run",
    "retrograd-config",
    "retrograd-env",
    "retrograd-judge",
    "retrograd-llm-client",
    "rmcp",
    "bollard",
];

#[test]
fn the_serve_facade_depends_on_no_training_side_crate() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "retrograd-serve",
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
    // The first token of a `--prefix none` line is the bare package name, so
    // an exact match cannot confuse `retrograd-agent` with
    // `retrograd-agent-core`.
    let packages: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    // The tree is read, not guessed: an empty one would make the assertions
    // below vacuous.
    assert!(
        packages.contains(&"retrograd-serve"),
        "cargo tree printed no retrograd-serve"
    );
    assert!(
        packages.contains(&"retrograd-infer"),
        "the serving runtime drives the step-wise policy: cargo tree printed no retrograd-infer"
    );
    assert!(
        packages.contains(&"retrograd-openai"),
        "the protocol comes from retrograd-openai: cargo tree printed no retrograd-openai"
    );
    let found: Vec<&&str> = FORBIDDEN
        .iter()
        .filter(|bad| packages.contains(bad))
        .collect();
    assert!(
        found.is_empty(),
        "retrograd-serve depends on {found:?}: the inference binary must stay free of the agent, \
         training, run, config, environment, judge and llm-client crates and of the MCP and \
         container backends"
    );
}
