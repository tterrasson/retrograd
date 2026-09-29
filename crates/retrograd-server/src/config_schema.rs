//! `GET /v1/config-schema`: the configuration document's JSON Schema, with
//! what this server adds to it.
//!
//! The schema itself is `retrograd-config`'s, derived from the types that parse
//! a document. The server marks two things on it that only the server knows,
//! each from the list it already enforces:
//!
//! - `x-retrograd-server-declared: true` on every field the request guard
//!   refuses from a client ([`crate::guard`]) - a command, an endpoint, a key.
//!   A client does not offer them; the refusal stays the guard's;
//! - `x-retrograd-patchable: true` on every path `PATCH /v1/runs/{id}` accepts
//!   on a live run.
//!
//! The same marks go on the `ConfigDocument` component of the OpenAPI document,
//! so a client generated from either reads the same thing.

use serde_json::{Map, Value};

/// Marks `components` - `ConfigDocument` and the tables it uses - with what
/// this server enforces.
pub fn annotate(components: &mut Map<String, Value>) {
    let declared: Vec<&str> = crate::guard::server_declared_keys().collect();
    for schema in components.values_mut() {
        mark_declared(schema, &declared);
    }
    for path in crate::runtime::control::Adjustments::whitelist() {
        let Some((section, field)) = path.split_once('.') else {
            continue;
        };
        let Some(table) = section_table(components, section) else {
            continue;
        };
        if let Some(property) = components
            .get_mut(&table)
            .and_then(|schema| schema.pointer_mut(&format!("/properties/{field}")))
            .and_then(Value::as_object_mut)
        {
            property.insert("x-retrograd-patchable".into(), Value::Bool(true));
        }
    }
}

/// Every property named like a server-declared key, at any depth.
fn mark_declared(value: &mut Value, declared: &[&str]) {
    match value {
        Value::Object(map) => {
            if let Some(Value::Object(properties)) = map.get_mut("properties") {
                for (name, property) in properties.iter_mut() {
                    if declared.contains(&name.as_str())
                        && let Value::Object(property) = property
                    {
                        property.insert("x-retrograd-server-declared".into(), Value::Bool(true));
                    }
                }
            }
            for child in map.values_mut() {
                mark_declared(child, declared);
            }
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|child| mark_declared(child, declared)),
        _ => {}
    }
}

/// The component a section of `ConfigDocument` refers to: a `$ref`, possibly
/// inside the `oneOf` an optional table is written as.
fn section_table(components: &Map<String, Value>, section: &str) -> Option<String> {
    let property = components
        .get("ConfigDocument")?
        .pointer(&format!("/properties/{section}"))?;
    let mut references = Vec::new();
    collect_references(property, &mut references);
    references
        .into_iter()
        .find_map(|reference| reference.rsplit('/').next().map(str::to_owned))
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

/// The standalone JSON Schema 2020-12 document, marked.
pub fn document() -> Value {
    let mut components = retrograd_config::schema::components();
    annotate(&mut components);
    let mut root = components
        .remove("ConfigDocument")
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));
    let mut definitions = Value::Object(components);
    retrograd_config::schema::rebase(&mut root);
    retrograd_config::schema::rebase(&mut definitions);
    let mut schema = Map::new();
    schema.insert(
        "$schema".into(),
        Value::from("https://json-schema.org/draft/2020-12/schema"),
    );
    schema.insert(
        "$id".into(),
        Value::from(retrograd_config::schema::schema_id()),
    );
    schema.insert("title".into(), Value::from("retrograd run configuration"));
    if let Value::Object(root) = root {
        schema.extend(root);
    }
    schema.insert("$defs".into(), definitions);
    Value::Object(schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_fields_and_patchable_paths_are_marked() {
        let schema = document();
        let grpo = &schema["$defs"]["GrpoToml"]["properties"];
        assert_eq!(grpo["reward_command"]["x-retrograd-server-declared"], true);
        assert!(
            grpo["group_size"]
                .get("x-retrograd-server-declared")
                .is_none()
        );
        let training = &schema["$defs"]["TrainingToml"]["properties"];
        assert_eq!(training["lr"]["x-retrograd-patchable"], true);
        assert!(training["ctx"].get("x-retrograd-patchable").is_none());
        let checkpoint = &schema["$defs"]["CheckpointToml"]["properties"];
        assert_eq!(checkpoint["every_steps"]["x-retrograd-patchable"], true);
        assert_eq!(
            schema["properties"]["ppo"]["x-retrograd-applies-to"],
            serde_json::json!(["ppo"])
        );
    }
}
