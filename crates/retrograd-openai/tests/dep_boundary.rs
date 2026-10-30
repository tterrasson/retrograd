//! The dependency boundary of the OpenAI facade, checked rather than intended:
//!
//! > `retrograd-openai` never depends on `retrograd-agent` - and with it on
//! > `retrograd-training`, `-env`, `-judge`, `-llm-client`, `rmcp` or
//! > `bollard`.
//!
//! The wire contract is served over the rendering and the parsers the rollouts
//! use, and those live in `retrograd-agent-core` next to `retrograd-infer`;
//! nothing here runs a tool, an environment or a judge, so the whole agent
//! stack - and the MCP and container backends behind it - must stay out of
//! this crate's tree. Same shape of check as `retrograd-infer`'s
//! `chain_boundary.rs`: read `cargo tree`, do not trust the manifest.

use std::process::Command;

const FORBIDDEN: &[&str] = &[
    "retrograd-agent",
    "retrograd-training",
    "retrograd-env",
    "retrograd-judge",
    "retrograd-llm-client",
    "rmcp",
    "bollard",
];

#[test]
fn the_openai_crate_depends_on_no_agent_side_crate() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "retrograd-openai",
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
        packages.contains(&"retrograd-openai"),
        "cargo tree printed no retrograd-openai"
    );
    assert!(
        packages.contains(&"retrograd-agent-core"),
        "the rendering and parsers the rollouts use live in retrograd-agent-core: cargo tree \
         printed no retrograd-agent-core"
    );
    assert!(
        packages.contains(&"retrograd-infer"),
        "the inference policy lives in retrograd-infer: cargo tree printed no retrograd-infer"
    );
    let found: Vec<&&str> = FORBIDDEN
        .iter()
        .filter(|bad| packages.contains(bad))
        .collect();
    assert!(
        found.is_empty(),
        "retrograd-openai depends on {found:?}: the serving facade must stay free of the agent, \
         training, environment, judge and llm-client crates and of the MCP and container backends"
    );
}
