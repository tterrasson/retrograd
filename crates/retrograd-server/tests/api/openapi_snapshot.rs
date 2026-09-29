//! `web/openapi.json` is the document this build serves.
//!
//! The web client's types are generated from that file. A route or a DTO
//! changed here without regenerating it would compile, pass every Rust test,
//! and break the interface at runtime - so the drift fails here, in the Rust
//! lane, as well as in the web one.

use std::path::Path;

#[test]
fn the_web_snapshot_is_the_document_this_build_serves() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/openapi.json");
    let recorded = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let recorded: serde_json::Value =
        serde_json::from_str(&recorded).expect("the snapshot is JSON");
    assert!(
        recorded == retrograd_server::openapi::render(),
        "web/openapi.json is stale; regenerate it with\n  \
         cargo run -p retrograd-server --bin retrograd-server -- openapi > web/openapi.json\n\
         then `cd web && bun run gen:api`"
    );
}
