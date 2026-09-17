//! Reader for the common `{ "mcpServers": {... } }` exchange format.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use retrograd_agent_core::{Error, Result};
use serde::Deserialize;

use crate::{McpServerConfig, McpTransport};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    #[serde(rename = "mcpServers")]
    servers: BTreeMap<String, ExchangeServer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExchangeServer {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout: Option<u64>,
}

/// Reads `files` (each in the `{ "mcpServers": {...} }` exchange format),
/// merges them, then overlays `policies` on top - a policy field left at its
/// default does not override what a file said, so declaring one setting for a
/// server does not silently reset the others. `${VAR}` references in the
/// merged environment and headers are expanded from the process environment
/// exactly once, after the overlay. Returns the resolved servers plus any
/// non-fatal warnings (a disabled server, a literal secret-looking value).
pub fn load_mcp_config_files(
    files: &[impl AsRef<Path>],
    policies: &[McpServerConfig],
) -> Result<(Vec<McpServerConfig>, Vec<String>)> {
    let mut merged = BTreeMap::<String, McpServerConfig>::new();
    let mut warnings = Vec::new();
    for path in files {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            Error::Tool(format!(
                "read MCP configuration {}: {error}",
                path.display()
            ))
        })?;
        let document: Document = serde_json::from_slice(&bytes).map_err(|error| {
            Error::invalid(format!(
                "{}: invalid MCP configuration: {error}",
                path.display()
            ))
        })?;
        let mut within = HashSet::new();
        for (name, entry) in document.servers {
            if !within.insert(name.clone()) || merged.contains_key(&name) {
                return Err(Error::invalid(format!(
                    "duplicate MCP server '{name}' in configuration files"
                )));
            }
            if entry.disabled {
                warnings.push(format!("MCP server '{name}' is disabled and was ignored"));
                continue;
            }
            let cwd = entry.cwd.as_ref().map(|cwd| {
                let cwd = Path::new(cwd);
                if cwd.is_absolute() {
                    cwd.to_path_buf()
                } else {
                    path.parent().unwrap_or_else(|| Path::new(".")).join(cwd)
                }
                .to_string_lossy()
                .into_owned()
            });
            // One match, not "build a transport, then read it back": a file
            // entry always declares one, so the intermediate value would carry
            // an `Unspecified` case that only the policy overlay below can
            // produce, and the arm for it could only be `unreachable!`.
            let mut config = match (entry.command, entry.url, entry.kind.as_deref()) {
                (Some(command), None, None | Some("stdio")) => {
                    let mut argv = vec![command];
                    argv.extend(entry.args);
                    // Left raw: the single interpolation pass runs after the
                    // policy overlay, over whichever transport survives the
                    // merge. Expanding here too would re-expand the result - a
                    // resolved secret then trips the "use ${VAR}" warning, and a
                    // legitimate `${` inside a value fails the whole load.
                    McpServerConfig::stdio(&name, argv).with_env(entry.env)
                }
                (None, Some(url), None | Some("http")) => {
                    McpServerConfig::http(&name, url).with_headers(entry.headers)
                }
                (_, _, Some("sse")) => {
                    return Err(Error::invalid(format!(
                        "MCP server '{name}' uses SSE, which is not supported in V1"
                    )));
                }
                _ => {
                    return Err(Error::invalid(format!(
                        "MCP server '{name}' must declare one stdio command or HTTP url"
                    )));
                }
            };
            config.cwd = cwd;
            if let Some(timeout) = entry.timeout {
                config.tool_timeout_secs = timeout;
            }
            merged.insert(name, config);
        }
    }
    for policy in policies {
        if let Some(base) = merged.get_mut(&policy.name) {
            // The policy is an overlay, not a replacement: what it does not
            // express stays as the file wrote it. `McpServerConfig` has no
            // "unset" marker, so a field left at its default reads as "not
            // expressed" and the file wins - otherwise declaring `stateless =
            // true` in TOML would silently drop the `cwd` and `timeout` the
            // exchange file gave the same server.
            let mut overlaid = policy.clone();
            if matches!(policy.transport, McpTransport::Unspecified) {
                overlaid.transport = base.transport.clone();
            }
            if policy.cwd.is_none() {
                overlaid.cwd = base.cwd.clone();
            }
            if policy.tool_timeout_secs == retrograd_spec::tools::mcp::default_timeout() {
                overlaid.tool_timeout_secs = base.tool_timeout_secs;
            }
            *base = overlaid;
        } else {
            merged.insert(policy.name.clone(), policy.clone());
        }
    }
    let mut servers = merged.into_values().collect::<Vec<_>>();
    for server in &mut servers {
        interpolate_transport(server, &mut warnings)?;
        server.validate()?;
    }
    Ok((servers, warnings))
}

fn interpolate_transport(server: &mut McpServerConfig, warnings: &mut Vec<String>) -> Result<()> {
    match &mut server.transport {
        McpTransport::Stdio { env, .. } => {
            *env = interpolate_map(std::mem::take(env), &server.name, warnings)?;
        }
        McpTransport::StreamableHttp { headers, .. } => {
            *headers = interpolate_map(std::mem::take(headers), &server.name, warnings)?;
        }
        McpTransport::Unspecified => {}
    }
    Ok(())
}

fn interpolate_map(
    values: BTreeMap<String, String>,
    server: &str,
    warnings: &mut Vec<String>,
) -> Result<BTreeMap<String, String>> {
    values
        .into_iter()
        .map(|(key, value)| {
            let expanded = interpolate(&value).map_err(|error| {
                Error::invalid(format!("MCP server '{server}', value for '{key}': {error}"))
            })?;
            if !value.contains("${") && looks_secret(&value) {
                warnings.push(format!(
                    "MCP server '{server}' contains a literal secret-like value; use ${{VAR}}"
                ));
            }
            Ok((key, expanded))
        })
        .collect()
}

fn interpolate(value: &str) -> Result<String> {
    let mut result = String::new();
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        result.push_str(&rest[..start]);
        let tail = &rest[start + 2..];
        let end = tail
            .find('}')
            .ok_or_else(|| Error::invalid("unclosed ${VAR}"))?;
        let name = &tail[..end];
        if name.is_empty() {
            return Err(Error::invalid("empty environment variable reference"));
        }
        result
            .push_str(&std::env::var(name).map_err(|_| {
                Error::invalid(format!("environment variable '{name}' is not set"))
            })?);
        rest = &tail[end + 1..];
    }
    result.push_str(rest);
    Ok(result)
}

fn looks_secret(value: &str) -> bool {
    value.starts_with("sk-")
        || value.starts_with("ghp_")
        || (value.len() >= 32
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_".contains(c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_format_and_policy_overlay_merge_without_rewriting_transport() {
        let path = std::env::temp_dir().join(format!("retrograd-mcp-{}.json", std::process::id()));
        let exchange = serde_json::json!({
            "mcpServers": {
                "demo": {
                    "command": "demo-cmd",
                    "args": ["--flag"],
                    "cwd": "servers/demo",
                    "timeout": 120
                }
            }
        });
        fs::write(&path, exchange.to_string()).unwrap();
        let policy: McpServerConfig = serde_json::from_value(serde_json::json!({
            "name": "demo", "stateless": true, "denied_tools": ["*_admin"]
        }))
        .unwrap();
        let (servers, warnings) = load_mcp_config_files(&[&path], &[policy]).unwrap();
        fs::remove_file(path).ok();
        assert!(warnings.is_empty());
        assert_eq!(servers.len(), 1);
        assert!(servers[0].stateless);
        assert_eq!(
            servers[0]
                .denied_tools
                .as_ref()
                .and_then(|patterns| patterns.first())
                .map(String::as_str),
            Some("*_admin")
        );
        assert!(
            matches!(&servers[0].transport, McpTransport::Stdio { command, .. }
            if command == &["demo-cmd", "--flag"])
        );
        // A policy that says nothing about where or how long must not undo what
        // the exchange file said about it.
        assert_eq!(servers[0].tool_timeout_secs, 120);
        assert!(
            servers[0]
                .cwd
                .as_deref()
                .is_some_and(|cwd| cwd.ends_with("servers/demo"))
        );
    }

    /// `set_var`/`remove_var` are unsound if another thread can be reading the
    /// process environment at the same time. `resolve_env` (the only other
    /// reader in this crate, above) and `connect_client`'s `env::var_os` in
    /// `mcp.rs` both only run inside this test's own call to
    /// `load_mcp_config_files`, on this same thread -- no other `#[test]` in
    /// `retrograd-tools`'s `--lib` binary touches the process environment. That
    /// binary runs with the default multi-threaded harness (see
    /// `scripts/test-fast-rust.sh`), so this lock is what keeps that true if a
    /// future test is added: take it before adding any other env-touching case.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn file_values_are_interpolated_exactly_once() {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let path =
            std::env::temp_dir().join(format!("retrograd-mcp-interp-{}.json", std::process::id()));
        // SAFETY: guarded by ENV_TEST_LOCK, see its doc comment above.
        unsafe {
            std::env::set_var(
                "RETROGRAD_TEST_MCP_KEY",
                "sk-resolved-secret-value-0123456789",
            )
        };
        // A resolved value that itself contains `${`: a second pass would try to
        // expand it and fail the whole load on an unclosed reference.
        // SAFETY: guarded by ENV_TEST_LOCK, see its doc comment above.
        unsafe { std::env::set_var("RETROGRAD_TEST_MCP_TEMPLATE", "prefix-${unclosed") };
        let exchange = serde_json::json!({
            "mcpServers": {
                "s": {
                    "command": "x",
                    "env": {
                        "KEY": "${RETROGRAD_TEST_MCP_KEY}",
                        "TEMPLATE": "${RETROGRAD_TEST_MCP_TEMPLATE}"
                    }
                }
            }
        });
        fs::write(&path, exchange.to_string()).unwrap();
        let (servers, warnings) = load_mcp_config_files(&[&path], &[]).unwrap();
        fs::remove_file(path).ok();
        // SAFETY: guarded by ENV_TEST_LOCK, see its doc comment above.
        unsafe { std::env::remove_var("RETROGRAD_TEST_MCP_KEY") };
        // SAFETY: guarded by ENV_TEST_LOCK, see its doc comment above.
        unsafe { std::env::remove_var("RETROGRAD_TEST_MCP_TEMPLATE") };
        // A second pass would also warn about the *resolved* secret, which is
        // the one value the operator wrote correctly.
        assert!(warnings.is_empty(), "{warnings:?}");
        let McpTransport::Stdio { env, .. } = &servers[0].transport else {
            panic!("expected stdio");
        };
        assert_eq!(
            env.get("KEY").map(String::as_str),
            Some("sk-resolved-secret-value-0123456789")
        );
        assert_eq!(
            env.get("TEMPLATE").map(String::as_str),
            Some("prefix-${unclosed")
        );
    }

    #[test]
    fn sse_and_unknown_fields_are_refused() {
        let parse = |source: &str| serde_json::from_str::<Document>(source);
        assert!(parse(r#"{"mcpServers":{"x":{"type":"sse","url":"https://x"}}}"#).is_ok());
        assert!(parse(r#"{"mcpServers":{"x":{"command":"x","typo":true}}}"#).is_err());
    }
}
