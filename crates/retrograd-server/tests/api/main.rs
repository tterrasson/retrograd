//! The HTTP surface of the server, driven through the router in memory.
//!
//! One test binary rather than one per file: each of them statically links the
//! whole server stack, so a dozen binaries cost a dozen links and a dozen copies
//! of it in `target/` for no isolation the tests use - every fixture already
//! lives in a directory of its own. Filter on the module path to run one file:
//! `cargo test -p retrograd-server --test api runs::`.

mod support;

mod artifacts;
mod control;
mod datasets;
mod discovery;
mod error_catalog;
mod events;
mod fork;
mod hardening;
mod inference;
mod openai;
mod plan;
mod runs;
