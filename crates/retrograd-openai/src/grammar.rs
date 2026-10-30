//! Structured output: from a `response_format` (or a Responses `text.format`)
//! to the GBNF grammar that constrains decoding, through the runtime's
//! JSON Schema -> GBNF converter.
//!
//! Two steps, both at admission (no model needed): the schema is validated
//! against the published subset - every keyword outside it is refused with
//! its path named, never ignored - and the surviving schema is converted once
//! into GBNF, which is what the sampler carries. The grammar is then attached
//! to the request's sampling parameters and compiled by the runtime once per
//! request and choice.
//!
//! The supported subset mirrors what the runtime's converter reads: `type`
//! (string or list), `properties`, `required`, `additionalProperties`,
//! `items`, `prefixItems`, `minItems`, `maxItems`, `pattern`, `minLength`,
//! `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
//! `enum`, `const`, `anyOf`, local `$ref` (`#/definitions/…`, `#/$defs/…`), `definitions`,
//! `$defs`, and the annotations `title`, `description`, `default`, `$schema`,
//! `examples` (read, not enforced). Everything else - `format`, `not`,
//! `if`/`then`/`else`, `contains`, `patternProperties`, `propertyNames`,
//! `dependencies`, `multipleOf`, `uniqueItems`, non-local `$ref`, … - is
//! refused.

use std::sync::Arc;

use serde_json::Value;

use crate::error::OpenAiError;

/// The keywords the converter enforces (produce grammar rules).
const SUPPORTED: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
    "const",
    "anyOf",
    "$ref",
    "definitions",
    "$defs",
];

/// Annotations: read and carried, but they do not shape the grammar.
const ANNOTATIONS: &[&str] = &["title", "description", "default", "$schema", "examples"];

/// Validates a JSON Schema document against the published subset. `path` is
/// the location of `schema` inside the document, used in refusals; `root` is
/// the document itself, where local `$ref` targets are resolved.
pub fn validate_schema(schema: &Value, path: &str) -> Result<(), OpenAiError> {
    validate(schema, path, schema)
}

fn validate(schema: &Value, path: &str, root: &Value) -> Result<(), OpenAiError> {
    let Some(object) = schema.as_object() else {
        return Err(invalid(path, "a schema must be an object"));
    };
    for key in object.keys() {
        if !SUPPORTED.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str()) {
            return Err(OpenAiError::unsupported(
                format!("{path}/{key}"),
                format!("the JSON Schema keyword `{key}` is not supported by this server"),
            ));
        }
    }
    // The converter selects one branch; sibling constraints of these
    // selectors are otherwise silently discarded.
    for selector in ["$ref", "anyOf", "enum", "const"] {
        if object.contains_key(selector) {
            for key in object.keys() {
                if key != selector
                    && SUPPORTED.contains(&key.as_str())
                    && !["definitions", "$defs"].contains(&key.as_str())
                {
                    return Err(OpenAiError::unsupported(
                        format!("{path}/{key}"),
                        format!(
                            "constraints beside `{selector}` are not supported by the grammar converter"
                        ),
                    ));
                }
            }
        }
    }
    for (minimum, maximum) in [("minItems", "maxItems"), ("minLength", "maxLength")] {
        let mut bounds = [None, None];
        for (index, key) in [minimum, maximum].into_iter().enumerate() {
            if let Some(value) = object.get(key) {
                let count = value
                    .as_u64()
                    .filter(|&count| count <= i32::MAX as u64)
                    .ok_or_else(|| {
                        invalid(
                            &format!("{path}/{key}"),
                            "must be a non-negative 32-bit integer",
                        )
                    })?;
                bounds[index] = Some(count);
            }
        }
        if let [Some(min), Some(max)] = bounds
            && min > max
        {
            return Err(invalid(
                &format!("{path}/{maximum}"),
                "must be at least the minimum",
            ));
        }
    }
    if object.contains_key("pattern")
        && ["minLength", "maxLength"]
            .iter()
            .any(|key| object.contains_key(*key))
    {
        return Err(OpenAiError::unsupported(
            format!("{path}/pattern"),
            "pattern and length constraints cannot be combined by the grammar converter",
        ));
    }
    for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
        if let Some(value) = object.get(key) {
            if object.get("type").and_then(Value::as_str) != Some("integer") {
                return Err(OpenAiError::unsupported(
                    format!("{path}/{key}"),
                    "numeric bounds are supported only for type integer",
                ));
            }
            let bound = value.as_i64().ok_or_else(|| {
                invalid(&format!("{path}/{key}"), "must be a signed 64-bit integer")
            })?;
            if key == "exclusiveMinimum" && bound == i64::MAX
                || key == "exclusiveMaximum" && bound == i64::MIN
            {
                return Err(invalid(
                    &format!("{path}/{key}"),
                    "exclusive bound leaves no representable integer",
                ));
            }
        }
    }
    for (inclusive, exclusive) in [
        ("minimum", "exclusiveMinimum"),
        ("maximum", "exclusiveMaximum"),
    ] {
        if object.contains_key(inclusive) && object.contains_key(exclusive) {
            return Err(OpenAiError::unsupported(
                format!("{path}/{exclusive}"),
                "inclusive and exclusive bounds cannot be combined by the grammar converter",
            ));
        }
    }
    let min = object.get("minimum").and_then(Value::as_i64).or_else(|| {
        object
            .get("exclusiveMinimum")
            .and_then(Value::as_i64)
            .map(|v| v + 1)
    });
    let max = object.get("maximum").and_then(Value::as_i64).or_else(|| {
        object
            .get("exclusiveMaximum")
            .and_then(Value::as_i64)
            .map(|v| v - 1)
    });
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        return Err(invalid(path, "numeric bounds leave no valid integer"));
    }
    if let Some(reference) = object.get("$ref") {
        let Some(reference) = reference.as_str() else {
            return Err(invalid(&format!("{path}/$ref"), "`$ref` must be a string"));
        };
        let Some(rest) = reference.strip_prefix("#/") else {
            return Err(OpenAiError::unsupported(
                format!("{path}/$ref"),
                "only local `$ref` of the form `#/definitions/…` or `#/$defs/…` are supported",
            ));
        };
        let mut segments = rest.split('/');
        let (Some(container), Some(name)) = (segments.next(), segments.next()) else {
            return Err(invalid(
                &format!("{path}/$ref"),
                "a local `$ref` names a definition",
            ));
        };
        if segments.next().is_some() {
            return Err(OpenAiError::unsupported(
                format!("{path}/$ref"),
                "only `#/definitions/<name>` and `#/$defs/<name>` references are supported",
            ));
        }
        if !["definitions", "$defs"].contains(&container) {
            return Err(OpenAiError::unsupported(
                format!("{path}/$ref"),
                "only references into definitions or $defs are supported",
            ));
        }
        // The pinned converter resolves literal segments, without JSON Pointer
        // unescaping. Refuse escaped names instead of resolving another target.
        if name.contains('~') {
            return Err(OpenAiError::unsupported(
                format!("{path}/$ref"),
                "escaped reference names are not supported by the grammar converter",
            ));
        }
        let target = root.get(container).and_then(|defs| defs.get(name));
        if target.is_none() {
            return Err(invalid(
                &format!("{path}/$ref"),
                format!("`$ref` `{reference}` does not name a definition of this document"),
            ));
        }
    }
    if let Some(types) = object.get("type") {
        match types {
            Value::String(_) => {}
            Value::Array(items) => {
                for item in items {
                    if item.as_str().is_none() {
                        return Err(invalid(
                            &format!("{path}/type"),
                            "`type` entries must be strings",
                        ));
                    }
                }
            }
            _ => {
                return Err(invalid(
                    &format!("{path}/type"),
                    "`type` must be a string or a list",
                ));
            }
        }
    }
    for key in ["properties", "definitions", "$defs", "patternProperties"] {
        if let Some(map) = object.get(key) {
            let Some(map) = map.as_object() else {
                return Err(invalid(&format!("{path}/{key}"), "must be an object"));
            };
            if key == "patternProperties" {
                return Err(OpenAiError::unsupported(
                    format!("{path}/{key}"),
                    "the JSON Schema keyword `patternProperties` is not supported by this server",
                ));
            }
            for (name, sub) in map {
                validate(sub, &format!("{path}/{key}/{name}"), root)?;
            }
        }
    }
    if let Some(required) = object.get("required") {
        let Some(required) = required.as_array() else {
            return Err(invalid(
                &format!("{path}/required"),
                "`required` must be a list of strings",
            ));
        };
        for name in required {
            if name.as_str().is_none() {
                return Err(invalid(
                    &format!("{path}/required"),
                    "`required` entries must be strings",
                ));
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(sub) = object.get(key)
            && !sub.is_boolean()
        {
            validate(sub, &format!("{path}/{key}"), root)?;
        }
    }
    if let Some(prefix) = object.get("prefixItems") {
        let Some(items) = prefix.as_array() else {
            return Err(invalid(
                &format!("{path}/prefixItems"),
                "`prefixItems` must be a list",
            ));
        };
        if object.contains_key("items") {
            return Err(OpenAiError::unsupported(
                format!("{path}/items"),
                "items and prefixItems cannot be combined by the grammar converter",
            ));
        }
        let length = items.len() as u64;
        if object
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|min| length < min)
            || object
                .get("maxItems")
                .and_then(Value::as_u64)
                .is_some_and(|max| length > max)
        {
            return Err(OpenAiError::unsupported(
                format!("{path}/prefixItems"),
                "tuple bounds must allow the full prefixItems tuple",
            ));
        }
        for (index, sub) in items.iter().enumerate() {
            validate(sub, &format!("{path}/prefixItems/{index}"), root)?;
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(list) = object.get(key) {
            let Some(list) = list.as_array() else {
                return Err(invalid(
                    &format!("{path}/{key}"),
                    "must be a list of schemas",
                ));
            };
            for (index, sub) in list.iter().enumerate() {
                validate(sub, &format!("{path}/{key}/{index}"), root)?;
            }
        }
    }
    Ok(())
}

fn invalid(path: &str, message: impl Into<String>) -> OpenAiError {
    OpenAiError::invalid_param(path, message.into())
}

/// The GBNF grammar for a `response_format` value: `None` for `text`, the
/// any-JSON grammar for `json_object`, the schema's grammar for `json_schema`.
/// The schema is validated at admission before the conversion.
pub fn response_format_grammar(format: &Value) -> Result<Option<Arc<str>>, OpenAiError> {
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| OpenAiError::invalid_param("response_format.type", "`type` is required"))?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(Arc::from(builtin_json_grammar()?))),
        "json_schema" => {
            let schema_def = format.get("json_schema").ok_or_else(|| {
                OpenAiError::invalid_param(
                    "response_format.json_schema",
                    "`json_schema` is required when `type` is `json_schema`",
                )
            })?;
            let name = schema_def.get("name").and_then(Value::as_str);
            if name.is_none_or(|name| name.trim().is_empty()) {
                return Err(OpenAiError::invalid_param(
                    "response_format.json_schema.name",
                    "a `name` is required",
                ));
            }
            let schema = schema_def.get("schema").ok_or_else(|| {
                OpenAiError::invalid_param(
                    "response_format.json_schema.schema",
                    "the `schema` object is required",
                )
            })?;
            validate_schema(schema, "response_format.json_schema.schema")?;
            let gbnf = schema_grammar(schema, "response_format.json_schema.schema")?;
            Ok(Some(Arc::from(gbnf)))
        }
        other => Err(OpenAiError::unsupported(
            "response_format.type",
            format!(
                "unsupported response format `{other}`; expected text, json_object or json_schema"
            ),
        )),
    }
}

/// The GBNF grammar of a Responses `text.format` value: same three forms, with
/// the schema under `text.format.schema` and the name under `text.format.name`.
pub fn text_format_grammar(format: &Value) -> Result<Option<Arc<str>>, OpenAiError> {
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| OpenAiError::invalid_param("text.format.type", "`type` is required"))?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(Arc::from(builtin_json_grammar()?))),
        "json_schema" => {
            let name = format.get("name").and_then(Value::as_str);
            if name.is_none_or(|name| name.trim().is_empty()) {
                return Err(OpenAiError::invalid_param(
                    "text.format.name",
                    "a `name` is required when `type` is `json_schema`",
                ));
            }
            let schema = format.get("schema").ok_or_else(|| {
                OpenAiError::invalid_param(
                    "text.format.schema",
                    "the `schema` object is required when `type` is `json_schema`",
                )
            })?;
            validate_schema(schema, "text.format.schema")?;
            let gbnf = schema_grammar(schema, "text.format.schema")?;
            Ok(Some(Arc::from(gbnf)))
        }
        other => Err(OpenAiError::unsupported(
            "text.format.type",
            format!("unsupported text format `{other}`; expected text, json_object or json_schema"),
        )),
    }
}

fn builtin_json_grammar() -> Result<String, OpenAiError> {
    retrograd_engine::json_object_grammar().map_err(|error| {
        OpenAiError::internal(format!("the built-in JSON grammar failed: {error}"))
    })
}

fn schema_grammar(schema: &Value, param: &str) -> Result<String, OpenAiError> {
    let text = serde_json::to_string(schema).map_err(|error| {
        OpenAiError::internal(format!("the schema does not serialize: {error}"))
    })?;
    retrograd_engine::grammar_from_json_schema(&text).map_err(|error| {
        OpenAiError::invalid_param(
            param,
            format!("the JSON schema is not valid or uses unsupported keywords: {error}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_published_subset_is_accepted() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "kind": {"enum": ["a", "b"]},
                "meta": {"$ref": "#/definitions/meta"}
            },
            "required": ["name"],
            "additionalProperties": false,
            "definitions": {"meta": {"type": "object", "properties": {"x": {"const": 1}}}}
        });
        validate_schema(&schema, "$").unwrap();
    }

    #[test]
    fn unsupported_keywords_are_refused_with_their_path() {
        for keyword in [
            "format",
            "not",
            "if",
            "contains",
            "patternProperties",
            "multipleOf",
        ] {
            let schema = json!({"type": "object", keyword: {}});
            let error = validate_schema(&schema, "$").unwrap_err();
            assert_eq!(
                error.param.as_deref(),
                Some(format!("$/{keyword}").as_str())
            );
            assert_eq!(error.code, Some("unsupported_parameter"));
        }
        // A non-local $ref is refused too.
        let schema = json!({"$ref": "https://example.com/schema.json"});
        let error = validate_schema(&schema, "$").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("$/$ref"));
    }

    #[test]
    fn a_json_object_format_maps_to_the_builtin_grammar() {
        let grammar = response_format_grammar(&json!({"type": "json_object"})).unwrap();
        let gbnf = grammar.expect("json_object constrains to JSON");
        assert!(gbnf.contains("root"), "the GBNF names its root rule");
    }

    #[test]
    fn a_local_ref_must_name_a_definition() {
        let schema = json!({"$ref": "#/definitions/missing"});
        let error = validate_schema(&schema, "$").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("$/$ref"));
    }

    #[test]
    fn a_json_schema_format_is_validated_then_converted() {
        let format = json!({"type": "json_schema", "json_schema": {
            "name": "point", "strict": true,
            "schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
        }});
        let grammar = response_format_grammar(&format)
            .unwrap()
            .expect("a grammar");
        assert!(grammar.contains("root"));

        let bad = json!({"type": "json_schema", "json_schema": {
            "name": "point", "schema": {"type": "object", "format": "email"}
        }});
        let error = response_format_grammar(&bad).unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("response_format.json_schema.schema/format")
        );
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn converter_constraints_that_would_be_ignored_are_refused() {
        for schema in [
            json!({"type":"object","minProperties":1}),
            json!({"type":"object","maxProperties":1}),
            json!({"type":"number","minimum":1}),
            json!({"oneOf":[{"type":"integer"},{"type":"number"}]}),
            json!({"allOf":[{"type":"string"}]}),
            json!({"type":"string","pattern":"^a$","minLength":2}),
            json!({"type":"integer","minimum":0,"exclusiveMinimum":1}),
            json!({"type":"integer","minimum":4,"maximum":2}),
            json!({"type":"array","minItems":4294967296_u64}),
            json!({"const":1,"type":"string"}),
            json!({"type":"array","items":{"type":"string"},"prefixItems":[]}),
        ] {
            assert!(
                validate_schema(&schema, "$").is_err(),
                "must refuse {schema}"
            );
        }
    }
}
