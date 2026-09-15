//! What an MCP server *is*, independent of the client that talks to it.
//!
//! Split from `retrograd_tools::mcp` so the schema survives `--no-default-features`:
//! `retrograd-config` parses `[[agent.mcp_servers]]` in every build, and a
//! binary without the `mcp` feature must say "not compiled in" rather than
//! "unknown field". Only the transport implementation needs rmcp.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use retrograd_agent_core::{Error, Result};

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpTransport {
    #[serde(skip)]
    Unspecified,
    Stdio {
        command: Vec<String>,
        #[serde(default, skip_serializing)]
        env: BTreeMap<String, String>,
    },
    StreamableHttp {
        url: String,
        #[serde(default, skip_serializing)]
        headers: BTreeMap<String, String>,
    },
}

impl McpTransport {
    fn is_unspecified(&self) -> bool {
        matches!(self, Self::Unspecified)
    }
}

impl std::fmt::Debug for McpTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unspecified => formatter.write_str("Unspecified"),
            Self::Stdio { command, env } => formatter
                .debug_struct("Stdio")
                .field("command", command)
                .field("env_names", &env.keys().collect::<Vec<_>>())
                .finish(),
            Self::StreamableHttp { url, headers } => formatter
                .debug_struct("StreamableHttp")
                .field("url", url)
                .field("header_names", &headers.keys().collect::<Vec<_>>())
                .finish(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(skip_serializing_if = "McpTransport::is_unspecified")]
    pub transport: McpTransport,
    pub tool_timeout_secs: u64,
    /// Tool-name patterns to expose. `None` exposes everything the server
    /// advertises. Patterns may use `*` as a wildcard, so `read_*` keeps a
    /// family of tools without listing each one; a pattern without `*` is an
    /// exact name, which is what plain lists already meant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Patterns to hide, applied after `allowed_tools`. Same wildcard syntax.
    /// Denying is how a mostly-useful server gets adopted without exposing the
    /// one tool that writes to production.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denied_tools: Option<Vec<String>>,
    pub max_tool_result_bytes: usize,
    /// A required server that fails to connect fails the run. An optional one is
    /// skipped with a warning, so a rollout does not die because a nice-to-have
    /// search server is down - but the tool it provided is then simply absent,
    /// which changes what the policy can learn. Required is the default for
    /// exactly that reason.
    pub required: bool,
    pub stateless: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_passthrough: Option<Vec<String>>,
}

pub fn default_timeout() -> u64 {
    30
}

fn default_max_result_bytes() -> usize {
    64 * 1024
}

/// Deserialization shape accepting both spellings of a transport: nested
/// (`transport = { type = "stdio", … }`, what the Python binding sends) and flat
/// (`command = [...]` / `url = "..."`, what the TOML runner has always used).
/// One canonical config type behind both means a frontend adds a server by
/// filling this struct, not by declaring its own.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct McpServerFields {
    name: String,
    #[serde(default)]
    transport: Option<McpTransport>,
    #[serde(default)]
    command: Option<Vec<String>>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default = "default_timeout")]
    tool_timeout_secs: u64,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    denied_tools: Option<Vec<String>>,
    #[serde(default = "default_max_result_bytes")]
    max_tool_result_bytes: usize,
    #[serde(default = "default_required")]
    required: bool,
    #[serde(default)]
    stateless: bool,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    env_passthrough: Option<Vec<String>>,
}

fn default_required() -> bool {
    true
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let fields = McpServerFields::deserialize(deserializer)?;
        let transport = match (fields.transport, fields.command, fields.url) {
            (Some(transport), None, None) => transport,
            (None, Some(command), None) => McpTransport::Stdio {
                command,
                env: fields.env,
            },
            (None, None, Some(url)) => McpTransport::StreamableHttp {
                url,
                headers: fields.headers,
            },
            (None, None, None) => McpTransport::Unspecified,
            _ => {
                return Err(D::Error::custom(format!(
                    "MCP server '{}' must specify exactly one of transport, command, or url",
                    fields.name
                )));
            }
        };
        Ok(Self {
            name: fields.name,
            transport,
            tool_timeout_secs: fields.tool_timeout_secs,
            allowed_tools: fields.allowed_tools,
            denied_tools: fields.denied_tools,
            max_tool_result_bytes: fields.max_tool_result_bytes,
            required: fields.required,
            stateless: fields.stateless,
            cwd: fields.cwd,
            env_passthrough: fields.env_passthrough,
        })
    }
}

impl McpServerConfig {
    /// A stdio server with the default timeout and result cap.
    pub fn stdio<I, S>(name: impl Into<String>, command: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::with_transport(
            name,
            McpTransport::Stdio {
                command: command.into_iter().map(Into::into).collect(),
                env: BTreeMap::new(),
            },
        )
    }

    /// A streamable-HTTP server with the default timeout and result cap.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::with_transport(
            name,
            McpTransport::StreamableHttp {
                url: url.into(),
                headers: BTreeMap::new(),
            },
        )
    }

    fn with_transport(name: impl Into<String>, transport: McpTransport) -> Self {
        Self {
            name: name.into(),
            transport,
            tool_timeout_secs: default_timeout(),
            allowed_tools: None,
            denied_tools: None,
            max_tool_result_bytes: default_max_result_bytes(),
            required: true,
            stateless: false,
            cwd: None,
            env_passthrough: None,
        }
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.tool_timeout_secs = seconds;
        self
    }

    pub fn with_max_result_bytes(mut self, bytes: usize) -> Self {
        self.max_tool_result_bytes = bytes;
        self
    }

    /// Marks the server as skippable when it cannot be reached.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn stateless(mut self) -> Self {
        self.stateless = true;
        self
    }

    pub fn allow<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = Some(patterns.into_iter().map(Into::into).collect());
        self
    }

    pub fn deny<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.denied_tools = Some(patterns.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_env<I, K, V>(mut self, variables: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.extend(
                variables
                    .into_iter()
                    .map(|(key, value)| (key.into(), value.into())),
            );
        }
        self
    }

    pub fn with_headers<I, K, V>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        if let McpTransport::StreamableHttp { headers, .. } = &mut self.transport {
            headers.extend(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.into(), value.into())),
            );
        }
        self
    }

    /// Shape check with no I/O: the loader runs it so a typo in a server
    /// declaration is reported at parse time, not at the first tool call.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.name.contains("__") {
            return Err(Error::invalid(
                "MCP server name must be non-empty and must not contain '__'",
            ));
        }
        if self.tool_timeout_secs == 0 || self.max_tool_result_bytes == 0 {
            return Err(Error::invalid(
                "MCP timeout and max result size must be greater than zero",
            ));
        }
        match &self.transport {
            McpTransport::Unspecified => Err(Error::invalid(format!(
                "MCP server '{}' needs a transport, a command, or a url",
                self.name
            ))),
            McpTransport::Stdio { command, .. } if command.is_empty() => {
                Err(Error::invalid("MCP stdio command must not be empty"))
            }
            McpTransport::StreamableHttp { url, .. } if url.is_empty() => {
                Err(Error::invalid("MCP HTTP url must not be empty"))
            }
            _ => Ok(()),
        }
    }

    /// Whether a remote tool name passes this server's filters.
    pub fn exposes(&self, tool: &str) -> bool {
        let allowed = self.allowed_tools.as_ref().is_none_or(|patterns| {
            patterns
                .iter()
                .any(|pattern| crate::tools::glob_match(pattern, tool))
        });
        let denied = self.denied_tools.as_ref().is_some_and(|patterns| {
            patterns
                .iter()
                .any(|pattern| crate::tools::glob_match(pattern, tool))
        });
        allowed && !denied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_invalid_transports_and_reserved_namespaces() {
        let invalid = [
            McpServerConfig::stdio("bad__name", ["echo"]),
            McpServerConfig::stdio("server", Vec::<String>::new()),
            McpServerConfig::http("server", ""),
            McpServerConfig::stdio("server", ["echo"]).with_timeout(0),
            McpServerConfig::stdio("server", ["echo"]).with_max_result_bytes(0),
        ];
        for config in invalid {
            assert!(config.validate().is_err(), "accepted {}", config.name);
        }
        McpServerConfig::stdio("server", ["echo"])
            .validate()
            .unwrap();
    }

    #[test]
    fn builders_produce_a_usable_server_in_one_expression() {
        let stdio = McpServerConfig::stdio("calc", ["python", "calc.py"])
            .with_env([("KEY", "value")])
            .with_timeout(5)
            .optional()
            .deny(["write_*"]);
        stdio.validate().unwrap();
        assert!(!stdio.required);
        assert_eq!(stdio.tool_timeout_secs, 5);
        assert!(matches!(
            &stdio.transport,
            McpTransport::Stdio { command, env }
                if command == &["python".to_owned(), "calc.py".to_owned()]
                    && env.get("KEY").map(String::as_str) == Some("value")
        ));

        let http = McpServerConfig::http("search", "https://example.test/mcp")
            .with_headers([("authorization", "Bearer x")])
            .allow(["search", "fetch_*"]);
        http.validate().unwrap();
        assert!(http.required, "servers are required unless asked otherwise");
        assert!(matches!(
            &http.transport,
            McpTransport::StreamableHttp { headers, .. }
                if headers.get("authorization").map(String::as_str) == Some("Bearer x")
        ));
    }

    #[test]
    fn both_transport_spellings_deserialize_to_the_same_config() {
        // Flat form: what the TOML runner has always written.
        let flat: McpServerConfig =
            serde_json::from_str(r#"{"name":"calc","command":["python","calc.py"]}"#).unwrap();
        // Nested form: what the Python binding sends.
        let nested: McpServerConfig = serde_json::from_str(
            r#"{"name":"calc","transport":{"type":"stdio","command":["python","calc.py"],"env":{}}}"#,
        )
        .unwrap();
        for config in [&flat, &nested] {
            assert!(matches!(
                &config.transport,
                McpTransport::Stdio { command, .. } if command.len() == 2
            ));
            assert_eq!(config.tool_timeout_secs, 30);
            assert!(config.required);
        }

        let http: McpServerConfig = serde_json::from_str(
            r#"{"name":"s","url":"https://x/mcp","headers":{"a":"b"},"required":false}"#,
        )
        .unwrap();
        assert!(!http.required);
        assert!(matches!(
            &http.transport,
            McpTransport::StreamableHttp { .. }
        ));

        // A transport-less value is the policy overlay shape used beside an
        // mcp.json transport. It parses, but is never runnable before merge.
        let overlay: McpServerConfig = serde_json::from_str(r#"{"name":"calc"}"#).unwrap();
        assert!(overlay.validate().is_err());

        for source in [
            // Both at once is ambiguous.
            r#"{"name":"calc","command":["x"],"url":"https://x/mcp"}"#,
            r#"{"name":"calc","transport":{"type":"stdio","command":["x"]},"command":["y"]}"#,
            // Typos must not be silently ignored.
            r#"{"name":"calc","command":["x"],"tool_timeout":5}"#,
        ] {
            assert!(
                serde_json::from_str::<McpServerConfig>(source).is_err(),
                "accepted {source}"
            );
        }
    }

    #[test]
    fn tool_filters_combine_wildcards_with_deny_winning() {
        let config = McpServerConfig::stdio("s", ["x"])
            .allow(["read_*", "list"])
            .deny(["read_secret"]);
        assert!(config.exposes("read_file"));
        assert!(config.exposes("list"));
        assert!(!config.exposes("read_secret"), "deny must win over allow");
        assert!(!config.exposes("write_file"));
        assert!(
            !config.exposes("listing"),
            "an exact pattern is not a prefix"
        );

        // No filters exposes everything.
        assert!(McpServerConfig::stdio("s", ["x"]).exposes("anything"));
        // Deny alone hides only what it names.
        let deny_only = McpServerConfig::stdio("s", ["x"]).deny(["*_admin"]);
        assert!(deny_only.exposes("read"));
        assert!(!deny_only.exposes("delete_admin"));
    }

    #[test]
    fn glob_matching_handles_anchors_and_inner_wildcards() {
        assert!(crate::tools::glob_match("exact", "exact"));
        assert!(!crate::tools::glob_match("exact", "exactly"));
        assert!(crate::tools::glob_match("*", "anything"));
        assert!(crate::tools::glob_match("read_*", "read_file"));
        assert!(crate::tools::glob_match("*_file", "read_file"));
        assert!(crate::tools::glob_match("read_*_file", "read_big_file"));
        assert!(!crate::tools::glob_match("read_*_file", "read_big_dir"));
        assert!(crate::tools::glob_match("a*b*c", "axxbyyc"));
        assert!(!crate::tools::glob_match("a*b*c", "axxb"));
    }
}
