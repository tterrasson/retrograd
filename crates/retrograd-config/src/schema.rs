//! The document's JSON Schema, derived from the types that parse it.
//!
//! Every table of [`ConfigDocument`] derives `utoipa::ToSchema` behind this
//! feature, so the schema is the parser's own description of what it accepts:
//! the field names, their types, which ones may be left out,
//! `deny_unknown_fields` as `additionalProperties: false`, and the `///`
//! comments as descriptions. What the types cannot say - the spellings of a
//! string with a closed vocabulary - comes from the constants the parser itself
//! checks against.
//!
//! [`json_schema`] renders it as a standalone JSON Schema 2020-12 document, the
//! form an editor reads for completion of a run's TOML.

use serde_json::{Map, Value, json};
use utoipa::openapi::schema::{Object, ObjectBuilder, SchemaType, Type};

use crate::ConfigDocument;

/// A string with a closed vocabulary, possibly absent.
fn one_of(values: impl IntoIterator<Item = &'static str>, nullable: bool) -> Object {
    let mut spellings: Vec<Value> = values.into_iter().map(Value::from).collect();
    let schema_type = if nullable {
        spellings.push(Value::Null);
        SchemaType::from_iter([Type::String, Type::Null])
    } else {
        SchemaType::new(Type::String)
    };
    ObjectBuilder::new()
        .schema_type(schema_type)
        .enum_values(Some(spellings))
        .build()
}

pub(crate) fn algorithm() -> Object {
    one_of(crate::ALGORITHMS.iter().copied(), false)
}

pub(crate) fn output_kind() -> Object {
    one_of(crate::OutputKind::ALL.map(crate::OutputKind::as_str), true)
}

pub(crate) fn checkpoint_mode() -> Object {
    one_of(crate::CheckpointMode::NAMES, false)
}

pub(crate) fn device() -> Object {
    one_of(retrograd_core::Device::NAMES, true)
}

pub(crate) fn lr_scheduler() -> Object {
    one_of(
        retrograd_core::LrScheduler::ALL.map(retrograd_core::LrScheduler::name),
        true,
    )
}

/// The schema's identity, versioned with the build that describes it.
pub fn schema_id() -> String {
    format!(
        "https://retrograd.dev/schemas/config-{}.json",
        env!("CARGO_PKG_VERSION")
    )
}

/// The components of the document: `ConfigDocument` and every table it
/// references, keyed by name, as OpenAPI 3.1 writes them.
pub fn components() -> Map<String, Value> {
    #[derive(utoipa::OpenApi)]
    #[openapi(components(schemas(ConfigDocument)))]
    struct Document;

    let mut rendered = serde_json::to_value(<Document as utoipa::OpenApi>::openapi())
        .unwrap_or_else(|_| json!({}));
    let mut components = match rendered["components"]["schemas"].take() {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    if let Some(document) = components.get_mut("ConfigDocument") {
        mark_sections(document);
    }
    components
}

/// `x-retrograd-applies-to` on each section that belongs to some algorithms
/// only, so a client does not offer `[ppo]` on a supervised run.
fn mark_sections(document: &mut Value) {
    let Some(properties) = document
        .get_mut("properties")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    for (section, algorithms) in crate::SECTION_ALGORITHMS {
        if let Some(property) = properties.get_mut(*section).and_then(Value::as_object_mut) {
            property.insert("x-retrograd-applies-to".into(), json!(algorithms));
        }
    }
}

/// The document's schema as a standalone JSON Schema 2020-12 document:
/// `ConfigDocument` at the root, every table it uses under `$defs`.
pub fn json_schema() -> Value {
    let mut definitions = components();
    let mut root = definitions
        .remove("ConfigDocument")
        .unwrap_or_else(|| json!({"type": "object"}));
    let mut definitions = Value::Object(definitions);
    rebase(&mut root);
    rebase(&mut definitions);
    if let Value::Object(map) = &mut root {
        let mut document = Map::new();
        document.insert(
            "$schema".into(),
            json!("https://json-schema.org/draft/2020-12/schema"),
        );
        document.insert("$id".into(), json!(schema_id()));
        document.insert("title".into(), json!("retrograd run configuration"));
        document.extend(std::mem::take(map));
        document.insert("$defs".into(), definitions);
        return Value::Object(document);
    }
    root
}

/// Rewrites OpenAPI's component references into the document's own `$defs`.
pub fn rebase(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get_mut("$ref")
                && let Some(name) = reference.strip_prefix("#/components/schemas/")
            {
                *reference = format!("#/$defs/{name}");
            }
            for child in map.values_mut() {
                rebase(child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(rebase),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_is_the_document_and_every_reference_resolves() {
        let schema = json_schema();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert!(schema["properties"]["training"].is_object());
        assert_eq!(schema["additionalProperties"], false);
        let definitions = schema["$defs"].as_object().expect("$defs");
        let mut references = Vec::new();
        collect_references(&schema, &mut references);
        assert!(!references.is_empty());
        for reference in references {
            let name = reference
                .strip_prefix("#/$defs/")
                .unwrap_or_else(|| panic!("{reference} is not local"));
            assert!(definitions.contains_key(name), "{reference} is dangling");
        }
    }

    #[test]
    fn the_vocabularies_are_the_parsers_own() {
        let schema = json_schema();
        let run = &schema["$defs"]["RunToml"]["properties"]["algorithm"]["enum"];
        assert_eq!(run, &json!(crate::ALGORITHMS));
        assert_eq!(
            schema["properties"]["observe"]["x-retrograd-applies-to"],
            json!(["ppo", "grpo", "agent_grpo"])
        );
    }

    fn collect_references(value: &Value, into: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                if let Some(Value::String(reference)) = map.get("$ref") {
                    into.push(reference.clone());
                }
                map.values()
                    .for_each(|child| collect_references(child, into));
            }
            Value::Array(items) => items
                .iter()
                .for_each(|child| collect_references(child, into)),
            _ => {}
        }
    }
}
