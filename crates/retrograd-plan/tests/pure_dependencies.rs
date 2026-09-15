//! The promise in `lib.rs` is that this crate touches no transport, and the
//! promise is checked here rather than intended.
//!
//! `retrograd-plan` depends on `retrograd-config` for the schema it resolves,
//! and `retrograd-config` on `retrograd-spec` for the declarations that schema
//! is made of. The declarations belong to `retrograd-spec`, so parsing a
//! document does not link an HTTP client, a container daemon client, an MCP
//! transport, or an async runtime.

use std::process::Command;

/// What no resolution can need. The list is deliberately spelled by name: a
/// crate added here is a decision, and a crate removed from the tree is a
/// decision too.
///
/// `tokio` is on it, and it is the one that says whether the *arrow* is right
/// rather than whether a feature was forgotten: nothing that merely reads a
/// document has an async runtime to schedule. It disappeared from this tree when
/// the declarations moved into `retrograd-spec` - a run-time crate reappearing
/// under `retrograd-config` is what would bring it back.
const FORBIDDEN: &[&str] = &["reqwest", "rustls", "hyper", "bollard", "rmcp", "tokio"];

#[test]
fn resolving_a_plan_links_no_transport() {
    let out = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "retrograd-plan",
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
    let tree = String::from_utf8_lossy(&out.stdout);

    // Package names sit at the start of a line under `--prefix none`; matching
    // the whole first field avoids counting a path that merely contains one.
    let packages: Vec<&str> = tree
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    let found: Vec<&str> = FORBIDDEN
        .iter()
        .copied()
        .filter(|bad| packages.contains(bad))
        .collect();
    assert!(
        found.is_empty(),
        "retrograd-plan's dependency tree carries {found:?} - a `default-features = false` \
         was dropped in retrograd-config, or a declaration moved into a crate that transports"
    );
    // The tree is read, not guessed: a `cargo tree` that printed nothing would
    // make the assertion above vacuous.
    assert!(
        packages.contains(&"retrograd-config"),
        "the tree does not even contain retrograd-config"
    );
}
