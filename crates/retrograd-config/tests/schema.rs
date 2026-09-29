//! The published schema and the parser agree.
//!
//! The schema is derived from the types that parse a document, so the two
//! should never disagree; these tests are what makes that a fact rather than
//! an intention. Forward: every document this repository ships - the
//! examples, and every configuration the resolver's snapshots record - is valid
//! against the schema. Backward, on a sample: what the schema refuses, the
//! parser refuses too.

#![cfg(feature = "openapi")]

use std::path::{Path, PathBuf};

use serde_json::Value;

fn validator() -> jsonschema::Validator {
    let schema = retrograd_config::schema::json_schema();
    jsonschema::draft202012::new(&schema).expect("the schema is itself valid")
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn toml_files(directory: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            toml_files(&path, into);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "toml")
        {
            into.push(path);
        }
    }
}

fn errors(validator: &jsonschema::Validator, document: &Value) -> Vec<String> {
    validator
        .iter_errors(document)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect()
}

#[test]
fn every_example_is_valid_against_the_schema() {
    let validator = validator();
    let mut files = Vec::new();
    toml_files(&repository().join("examples"), &mut files);
    assert!(!files.is_empty(), "no example found");
    for path in files {
        let text = std::fs::read_to_string(&path).expect("read");
        let document: Value = toml::from_str(&text).expect("an example is TOML");
        let errors = errors(&validator, &document);
        assert!(errors.is_empty(), "{}: {errors:#?}", path.display());
    }
}

#[test]
fn every_resolved_configuration_is_valid_against_the_schema() {
    let validator = validator();
    let snapshots = repository().join("crates/retrograd-plan/tests/snapshots");
    let mut checked = 0;
    for entry in std::fs::read_dir(&snapshots).expect("snapshots").flatten() {
        let text = std::fs::read_to_string(entry.path()).expect("read");
        let snapshot: Value = serde_json::from_str(&text).expect("JSON");
        let Some(config) = snapshot.get("config") else {
            continue;
        };
        let errors = errors(&validator, config);
        assert!(errors.is_empty(), "{}: {errors:#?}", entry.path().display());
        checked += 1;
    }
    assert!(checked > 0, "no snapshot carried a configuration");
}

/// A small document that parses and builds, to break one field at a time.
const BASE: &str = r#"
[run]
algorithm = "sft"

[model]
path = "model.gguf"

[output]
path = "adapter.gguf"

[lora]
rank = 1
alpha = 2.0
targets = ["q"]

[training]
lr = 0.0001

[sft]
data = "data.jsonl"

[checkpoint]
directory = "checkpoints"
mode = "steps"
every_steps = 10
"#;

fn parser_refuses(text: &str) -> bool {
    retrograd_config::parse_toml(text, "run.toml")
        .and_then(|document| retrograd_config::build(document, Path::new("/")))
        .is_err()
}

#[test]
fn the_base_document_is_accepted_by_both() {
    let document: Value = toml::from_str(BASE).expect("TOML");
    assert!(errors(&validator(), &document).is_empty());
    assert!(
        !parser_refuses(BASE),
        "{:?}",
        retrograd_config::parse_toml(BASE, "run.toml")
            .and_then(|document| retrograd_config::build(document, Path::new("/")))
    );
}

#[test]
fn what_the_schema_refuses_the_parser_refuses() {
    let validator = validator();
    let broken = [
        // A field the document does not have.
        BASE.replace("lr = 0.0001", "lr = 0.0001\nbogus = 1"),
        // A section the document does not have.
        format!("{BASE}\n[nonsense]\nx = 1\n"),
        // A number where a string is expected.
        BASE.replace("algorithm = \"sft\"", "algorithm = 3"),
        // A word outside a closed vocabulary.
        BASE.replace("algorithm = \"sft\"", "algorithm = \"dpo\""),
        BASE.replace("mode = \"steps\"", "mode = \"sometimes\""),
        // A string where a number is expected.
        BASE.replace("lr = 0.0001", "lr = \"fast\""),
        // A negative count.
        BASE.replace("every_steps = 10", "every_steps = -1"),
    ];
    for text in broken {
        let document: Value = toml::from_str(&text).expect("still TOML");
        assert!(
            !errors(&validator, &document).is_empty(),
            "the schema accepts:\n{text}"
        );
        assert!(parser_refuses(&text), "the parser accepts:\n{text}");
    }
}
