//! The contract boundary: the client references an `id`, never a command, an
//! endpoint or a key.
//!
//! There is no allow-list, no flag and no degraded mode - one path. A request
//! body that carries any of the server-declared keys is refused outright with a
//! JSON Pointer to the offending field, whether it appeared under `config`,
//! under `params`, or nested inside either.
//!
//! Called from the request extractors of `/v1/plan` and `POST /v1/runs`, on the
//! parsed JSON or TOML tree, before anything is typed - see
//! `api::plan::PlanBody::from_request`. `POST /v1/preflight` needs no call: its
//! `deny_unknown_fields` already refuses anything of the sort.
//!
//! Refusing by *key name*, anywhere in the tree, rather than by known location:
//! a location list would have to be extended every time `RunConfig` grows a
//! section, and the one time it was forgotten would be a silent hole. A false
//! positive here costs a caller one clear error message; a false negative costs
//! arbitrary command execution.

use retrograd_core::PointerPath;
use serde_json::Value;

use crate::error::{ApiError, ErrorCode, ProblemKind};

/// Keys only the operator may set, with the reason a client is being refused.
///
/// Every entry is a field that either names a program to execute, an address to
/// reach, or a secret to read.
const SERVER_DECLARED_KEYS: &[(&str, &str)] = &[
    (
        "reward_command",
        "declare it as [[reward]] and reference its id",
    ),
    (
        "command",
        "declare it as [[reward]], [[judge]] or [[mcp_server]] and reference its id",
    ),
    (
        "cwd",
        "the working directory of a reward is part of its server-side declaration",
    ),
    (
        "base_url",
        "declare the judge as [[judge]] and reference its id",
    ),
    (
        "api_key_env",
        "declare the judge as [[judge]] and reference its id",
    ),
    (
        "api_key",
        "the server never accepts a credential from a client",
    ),
    (
        "transport",
        "declare the server as [[mcp_server]] and reference its id",
    ),
    (
        "url",
        "declare the server as [[mcp_server]] and reference its id",
    ),
    (
        "env",
        "the environment of a server-side process is not client-settable",
    ),
    (
        "headers",
        "the headers sent to a server-side endpoint are not client-settable",
    ),
    ("mcp_server", "reference declared servers by id in `tools`"),
    ("mcp_servers", "reference declared servers by id in `tools`"),
    // An image, a mount or a pool size sent by a client would be code and
    // resources of the client's choosing on the operator's machine. They are
    // declared as `[[environment]]` and referenced by id, like everything else.
    (
        "image",
        "declare the sandbox as [[environment]] and reference its id",
    ),
    (
        "mounts",
        "declare the sandbox as [[environment]] and reference its id",
    ),
    (
        "allow_unsandboxed",
        "an unconfined sandbox is an operator decision, never a request field",
    ),
    (
        "pool",
        "declare the sandbox as [[environment]] and reference its id",
    ),
];

/// Rejects a request body that carries a server-declared value.
///
/// `root` is the JSON Pointer prefix of `value` in the request body (`""` for the
/// whole body, `"/params"` for a subtree), so the pointer reported back
/// locates the field in what the client actually sent.
pub fn reject_server_declared(value: &Value, root: &str) -> Result<(), ApiError> {
    let mut pointer = PointerPath::new(root);
    walk(value, &mut pointer)
}

fn walk(value: &Value, pointer: &mut PointerPath) -> Result<(), ApiError> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if let Some((_, hint)) = SERVER_DECLARED_KEYS
                    .iter()
                    .find(|(forbidden, _)| *forbidden == key)
                {
                    return Err(ApiError::new(
                        ProblemKind::ServerDeclared,
                        format!("'{key}' is server-declared, use the id"),
                    )
                    .with_field(
                        pointer.with_segment(key, |pointer| pointer.to_string()),
                        ErrorCode::ServerDeclared,
                        *hint,
                    ));
                }
                pointer.with_segment(key, |pointer| walk(child, pointer))?;
            }
            Ok(())
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                pointer.with_segment(index.to_string(), |pointer| walk(child, pointer))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refusal(value: Value) -> ApiError {
        reject_server_declared(&value, "").expect_err("must be refused")
    }

    #[test]
    fn a_reward_command_is_refused_wherever_it_hides() {
        for (body, expected_pointer) in [
            (
                json!({"config": {"grpo": {"reward_command": ["sh", "-c", "curl evil"]}}}),
                "/config/grpo/reward_command",
            ),
            (
                json!({"params": {"ppo": {"reward_command": ["x"]}}}),
                "/params/ppo/reward_command",
            ),
            (
                json!({"recipe": {"reward": {"command": ["x"]}}}),
                "/recipe/reward/command",
            ),
            (
                json!({"recipe": {"tools": [{"command": ["x"]}]}}),
                "/recipe/tools/0/command",
            ),
        ] {
            let error = refusal(body);
            assert_eq!(error.kind, ProblemKind::ServerDeclared);
            assert_eq!(error.kind.status(), 422);
            assert_eq!(error.errors[0].pointer, expected_pointer);
            assert!(
                error.detail.contains("server-declared"),
                "detail must say why: {}",
                error.detail
            );
        }
    }

    #[test]
    fn judge_endpoints_and_credentials_are_refused() {
        for key in [
            "base_url",
            "api_key_env",
            "api_key",
            "url",
            "headers",
            "env",
        ] {
            let error = refusal(json!({"recipe": {"judge": {key: "anything"}}}));
            assert_eq!(error.kind, ProblemKind::ServerDeclared);
            assert_eq!(error.errors[0].pointer, format!("/recipe/judge/{key}"));
        }
    }

    /// What a client sends decides what runs on the operator's machine only if
    /// this list lets it. An image or a mount is exactly that.
    #[test]
    fn sandbox_images_mounts_and_pools_are_refused() {
        for key in ["image", "mounts", "pool", "allow_unsandboxed"] {
            let error = refusal(json!({"recipe": {"environment": {key: "anything"}}}));
            assert_eq!(error.kind, ProblemKind::ServerDeclared);
            assert_eq!(
                error.errors[0].pointer,
                format!("/recipe/environment/{key}")
            );
        }
    }

    #[test]
    fn a_recipe_that_only_references_ids_passes() {
        // The shape the contract documents, minus everything the operator owns.
        let body = json!({
            "recipe": {
                "objective": "reasoning-rl",
                "model": "models/qwen3-1.7b.gguf",
                "data": {"path": "data/sft.jsonl", "format": "auto"},
                "budget": {"updates": 200},
                "limits": {"vram": "6GiB"},
                "allow": ["truncate_context"],
                "reward": {"id": "sql-exec"},
                "judge": {"id": "ruler-mini", "mode": "pairwise", "max_pairs": 12},
                "tools": ["calc", "search"],
                "environment": {"id": "py-sandbox"}
            },
            "params": {"training": {"ctx": 2048}, "lora": {"rank": 32}},
            "name": "qwen3-sft-v3"
        });
        reject_server_declared(&body, "").expect("a recipe of ids is acceptable");
    }

    #[test]
    fn pointers_escape_the_reserved_characters() {
        let error = refusal(json!({"a/b": {"c~d": {"command": ["x"]}}}));
        assert_eq!(error.errors[0].pointer, "/a~1b/c~0d/command");
    }

    #[test]
    fn a_pointer_prefix_locates_a_subtree_in_the_original_body() {
        let subtree = json!({"training": {"command": ["x"]}});
        let error = reject_server_declared(&subtree, "/params").expect_err("must be refused");
        assert_eq!(error.errors[0].pointer, "/params/training/command");
    }

    #[test]
    fn scalars_and_empty_containers_are_not_refused() {
        for body in [
            json!(null),
            json!(1),
            json!("command"),
            json!([]),
            json!({}),
        ] {
            reject_server_declared(&body, "").expect("nothing to refuse");
        }
    }
}
