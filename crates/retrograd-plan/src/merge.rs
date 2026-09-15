//! Deep-merging the client's `params` onto a resolved configuration document, and
//! the flat path grammar the whole API shares.
//!
//! One grammar, four users: `params` keys, the paths of `GET /v1/defaults`, the
//! `provenance` map and the `PATCH` whitelist. Dotted, rooted at the document,
//! `training.ctx`, `lora.rank`, `grpo.group_size`.

use std::collections::BTreeSet;

use retrograd_core::PointerPath;
use serde_json::{Map, Value};

/// Why a merge was refused, with the JSON Pointer of the offending field.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{pointer}: {message}")]
pub struct MergeError {
    pub pointer: String,
    pub message: String,
}

/// Deep-merges a partial `params` tree onto `base`, in place.
///
/// Objects merge key by key; every other value - including arrays - replaces
/// wholesale. Merging arrays element-wise would make `targets = ["q"]` mean
/// "replace the first target", which nobody expects.
///
/// An explicit `null` is refused rather than treated as "unset" - the only way
/// back to a default is not to send the field.
pub fn deep_merge(base: &mut Value, params: &Value) -> Result<(), MergeError> {
    merge_at(base, params, &mut PointerPath::default())
}

fn merge_at(
    base: &mut Value,
    overrides: &Value,
    pointer: &mut PointerPath,
) -> Result<(), MergeError> {
    match overrides {
        Value::Null => Err(MergeError {
            pointer: pointer.to_string(),
            message: "an explicit null does not clear a field; omit it to keep the default"
                .to_string(),
        }),
        Value::Object(incoming) => {
            if !base.is_object() {
                *base = Value::Object(Map::new());
            }
            let target = base.as_object_mut().expect("just made an object");
            for (key, value) in incoming {
                let slot = target.entry(key.clone()).or_insert(Value::Null);
                pointer.with_segment(key, |pointer| {
                    if value.is_object() {
                        if slot.is_null() {
                            *slot = Value::Object(Map::new());
                        }
                        merge_at(slot, value, pointer)?;
                    } else if value.is_null() {
                        return Err(MergeError {
                            pointer: pointer.to_string(),
                            message: "an explicit null does not clear a field; omit it to keep the default"
                                .to_string(),
                        });
                    } else {
                        *slot = value.clone();
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
        other => {
            *base = other.clone();
            Ok(())
        }
    }
}

/// Refuses an explicit `null` anywhere in a `params` tree.
///
/// Separate from [`deep_merge`] because the check has to happen *before* the
/// resolver reads any client value: a `null` under `training.ctx` would otherwise
/// be locked as "the caller set this" and reported as an impossible parameter
/// rather than as the malformed request it is.
pub fn reject_nulls(params: &Value) -> Result<(), MergeError> {
    fn walk(value: &Value, pointer: &mut PointerPath) -> Result<(), MergeError> {
        match value {
            Value::Null => Err(MergeError {
                pointer: pointer.to_string(),
                message: "an explicit null does not clear a field; omit it to keep the default"
                    .to_string(),
            }),
            Value::Object(map) => {
                for (key, child) in map {
                    pointer.with_segment(key, |pointer| walk(child, pointer))?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    walk(params, &mut PointerPath::default())
}

/// Every leaf path a `params` tree sets, in the dotted grammar.
///
/// This is what locks a field: a path listed here is the caller's and no phase
/// may re-derive it.
pub fn overridden_paths(params: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    collect(params, &mut String::new(), &mut out);
    out
}

fn collect(value: &Value, prefix: &mut String, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) if !map.is_empty() => {
            for (key, child) in map {
                let restore = prefix.len();
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(key);
                collect(child, prefix, out);
                prefix.truncate(restore);
            }
        }
        // A leaf, an array, or an empty object: the path itself is what was set.
        _ => {
            if !prefix.is_empty() {
                out.insert(prefix.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn objects_merge_and_scalars_replace() {
        let mut base = json!({
            "training": {"ctx": 1024, "micro_batch": 256, "lr": 1e-4},
            "lora": {"rank": 8}
        });
        deep_merge(
            &mut base,
            &json!({"training": {"ctx": 2048}, "lora": {"rank": 32}, "run": {"verbose": true}}),
        )
        .unwrap();
        assert_eq!(base["training"]["ctx"], 2048);
        assert_eq!(
            base["training"]["micro_batch"], 256,
            "untouched keys survive"
        );
        assert_eq!(base["lora"]["rank"], 32);
        assert_eq!(base["run"]["verbose"], true, "a new section is created");
    }

    #[test]
    fn an_array_replaces_rather_than_merging_element_wise() {
        let mut base = json!({"lora": {"targets": ["q", "k", "v"]}});
        deep_merge(&mut base, &json!({"lora": {"targets": ["q"]}})).unwrap();
        assert_eq!(base["lora"]["targets"], json!(["q"]));
    }

    #[test]
    fn an_explicit_null_is_refused_with_a_pointer() {
        let mut base = json!({"training": {"ctx": 1024}});
        let error = deep_merge(&mut base, &json!({"training": {"ctx": null}})).unwrap_err();
        assert_eq!(error.pointer, "/training/ctx");
        assert!(error.message.contains("omit it"), "{}", error.message);
        assert_eq!(base["training"]["ctx"], 1024, "the base is not damaged");
    }

    #[test]
    fn overridden_paths_are_the_leaves_in_dotted_form() {
        let paths = overridden_paths(&json!({
            "training": {"ctx": 2048, "gradient_checkpointing": true},
            "lora": {"targets": ["q"]}
        }));
        assert_eq!(
            paths.into_iter().collect::<Vec<_>>(),
            [
                "lora.targets",
                "training.ctx",
                "training.gradient_checkpointing"
            ]
        );
    }

    #[test]
    fn a_null_is_refused_before_anything_reads_it() {
        let error = reject_nulls(&json!({"training": {"ctx": null}})).unwrap_err();
        assert_eq!(error.pointer, "/training/ctx");
        reject_nulls(&json!({"training": {"ctx": 512}})).expect("a value is fine");
        // The whole tree being absent is not a null override.
        reject_nulls(&json!({})).expect("nothing to refuse");
    }

    #[test]
    fn an_empty_params_tree_locks_nothing() {
        assert!(overridden_paths(&json!({})).is_empty());
        assert!(overridden_paths(&Value::Null).is_empty());
    }
}
