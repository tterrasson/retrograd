//! `ERRORS.md` documents every `type` (`ProblemKind`) and every
//! `code` (`ErrorCode`) the API can emit. This checks the catalogue against
//! the code, not the other way around: a catalogue that can silently drift
//! from what the server actually emits is worse than no catalogue.

use retrograd_server::error::{ErrorCode, ProblemKind};

fn catalog() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/engineering/server/ERRORS.md"
    );
    std::fs::read_to_string(path).expect("read docs/engineering/server/ERRORS.md")
}

#[test]
fn every_problem_kind_has_an_entry() {
    let text = catalog();
    for kind in ProblemKind::ALL {
        let needle = format!("`{}`", kind.as_str());
        assert!(
            text.contains(&needle),
            "ProblemKind::{kind:?} (`{}`) has no row in ERRORS.md",
            kind.as_str()
        );
    }
}

#[test]
fn every_error_code_has_an_entry() {
    let text = catalog();
    for code in ErrorCode::ALL {
        let needle = format!("`{}`", code.as_str());
        assert!(
            text.contains(&needle),
            "ErrorCode::{code:?} (`{}`) has no row in ERRORS.md",
            code.as_str()
        );
    }
}

/// The other direction, which matters just as much: a `code` in the vocabulary
/// that no handler ever attaches is a promise the server does not keep. A
/// client writes a `match` arm for it, tests it against nothing, and never
/// learns it is dead.
///
/// Checked by scanning the crate's own sources for the variant rather than by
/// exercising every route: the vocabulary is a compile-time list, so the
/// question - "does any call site name this?" - is a compile-time question too.
#[test]
fn every_error_code_is_attached_somewhere() {
    let sources = rust_sources(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    for code in ErrorCode::ALL {
        let needle = format!("ErrorCode::{code:?}");
        assert!(
            sources.iter().any(|(path, text)| {
                // `error.rs` declares the enum; naming a variant there is not
                // emitting it.
                !path.ends_with("error.rs") && text.contains(&needle)
            }),
            "ErrorCode::{code:?} is in the vocabulary and in ERRORS.md, but nothing \
             attaches it: emit it or drop it"
        );
    }
}

/// Every `.rs` under `directory`, as (path, contents).
fn rust_sources(directory: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut pending = vec![std::path::PathBuf::from(directory)];
    while let Some(current) = pending.pop() {
        for entry in std::fs::read_dir(&current).expect("read the crate's sources") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).expect("read a source file");
                found.push((path.to_string_lossy().into_owned(), text));
            }
        }
    }
    assert!(!found.is_empty(), "no sources found under {directory}");
    found
}

// The macro derives the wire spelling and `as_str()` from the same literal.
// The tests above therefore focus on the catalogue and the call sites, which
// the compiler cannot compare with the implementation.
