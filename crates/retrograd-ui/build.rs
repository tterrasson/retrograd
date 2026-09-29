//! With `embed`, the UI build must exist before this crate compiles.
//!
//! Bun is never called from here: a Cargo build stays hermetic and offline, and
//! the one thing it does is say which command was forgotten.

use std::path::Path;

fn main() {
    let dist = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EMBED");
    if std::env::var_os("CARGO_FEATURE_EMBED").is_some() && !dist.join("index.html").is_file() {
        panic!(
            "web/dist/index.html is missing: build the UI first: \
             `cd web && bun install && bun run build`"
        );
    }
}
