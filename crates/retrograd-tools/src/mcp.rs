//! [`ToolProvider`] over one or more MCP servers.
//!
//! [`McpToolProvider::connect`] talks to every configured server once, at
//! construction, and merges their tools into one namespaced list (`server__tool`)
//! so a name collision between two servers is a construction-time error rather
//! than a routing ambiguity at call time. A call that hits a closed transport
//! gets one reconnect-and-retry before it is reported as failed.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::future::join_all;
use http::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, ContentBlock, JsonObject};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use tokio::sync::RwLock;

use retrograd_agent_core::tools::{ToolCall, ToolProvider, ToolResult, ToolSpec};
use retrograd_agent_core::{Error, Result};

use crate::ResourceSpec;
pub use retrograd_spec::tools::{McpServerConfig, McpTransport};

type Client = RunningService<RoleClient, ()>;

struct LimitedWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
    truncated: bool,
}

impl LimitedWriter {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(max_bytes.min(64 * 1024)),
            max_bytes,
            truncated: false,
        }
    }
}

impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = self.max_bytes.saturating_sub(self.bytes.len());
        let copy = remaining.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..copy]);
        self.truncated |= copy < bytes.len();
        // Report the whole input consumed so serializers keep walking the
        // value without retrying bytes deliberately discarded by the cap.
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ServerClient {
    config: McpServerConfig,
    client: RwLock<Client>,
}

#[derive(Clone, Debug)]
struct ToolRoute {
    server: usize,
    remote_name: String,
}

/// A [`ToolProvider`] backed by one or more connected MCP servers.
///
/// Built once by [`McpToolProvider::connect`]; each tool's public name is
/// `server__tool`, so [`ToolProvider::call`] can route it back to the server
/// that declared it.
pub struct McpToolProvider {
    servers: Vec<ServerClient>,
    tools: Vec<ToolSpec>,
    routes: HashMap<String, ToolRoute>,
    warnings: Vec<String>,
    resources: Vec<ResourceSpec>,
}

impl McpToolProvider {
    /// Connects to every server in `configs` concurrently. A server marked
    /// optional is skipped (and reported through [`Self::warnings`]) if it
    /// fails to connect; a required one fails the whole call, since silently
    /// training without a tool the configuration asked for changes the task.
    /// Fails if every server is unreachable, or if two servers share a name.
    pub async fn connect(configs: Vec<McpServerConfig>) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::invalid("at least one MCP server is required"));
        }
        let mut names = HashSet::new();
        for config in &configs {
            config.validate()?;
            if !names.insert(config.name.clone()) {
                return Err(Error::invalid(format!(
                    "duplicate MCP server name '{}'",
                    config.name
                )));
            }
        }

        let connected = join_all(configs.into_iter().map(|config| async move {
            let outcome = async {
                let client = connect_client(&config).await?;
                let remote_tools = client.list_all_tools().await.map_err(|error| {
                    Error::Tool(format!("list tools for '{}': {error}", config.name))
                })?;
                let remote_resources = if client
                    .peer_info()
                    .is_some_and(|info| info.capabilities.resources.is_some())
                {
                    client.list_all_resources().await.map_err(|error| {
                        Error::Tool(format!("list resources for '{}': {error}", config.name))
                    })?
                } else {
                    Vec::new()
                };
                Ok::<_, Error>((client, remote_tools, remote_resources))
            }
            .await;
            (config, outcome)
        }))
        .await;

        let mut servers = Vec::new();
        let mut tools = Vec::new();
        let mut routes = HashMap::new();
        let mut warnings = Vec::new();
        let mut resources = Vec::new();
        for (config, outcome) in connected {
            let (client, remote_tools, remote_resources) = match outcome {
                Ok(connected) => connected,
                // An optional server is skipped; a required one fails the run,
                // because silently training without a tool the config asked for
                // changes the task.
                Err(error) if !config.required => {
                    warnings.push(format!(
                        "optional MCP server '{}' is unavailable and was skipped: {error}",
                        config.name
                    ));
                    continue;
                }
                Err(error) => return Err(error),
            };
            let server = servers.len();
            let mut exposed = 0_usize;
            for tool in remote_tools {
                if !config.exposes(tool.name.as_ref()) {
                    continue;
                }
                exposed += 1;
                let public_name = format!("{}__{}", config.name, tool.name);
                let input_schema = serde_json::Value::Object((*tool.input_schema).clone());
                tools.push(ToolSpec {
                    name: public_name.clone(),
                    description: tool.description.unwrap_or_default().into_owned(),
                    input_schema,
                });
                routes.insert(
                    public_name,
                    ToolRoute {
                        server,
                        remote_name: tool.name.into_owned(),
                    },
                );
            }
            if exposed == 0 {
                warnings.push(format!(
                    "MCP server '{}' connected but its filters exposed no tool",
                    config.name
                ));
            }
            resources.extend(remote_resources.into_iter().map(|resource| ResourceSpec {
                server: config.name.clone(),
                uri: resource.uri,
                name: resource.name,
                description: resource.description.unwrap_or_default(),
                mime_type: resource.mime_type,
            }));
            if config.stateless {
                let prefix = format!("{}__", config.name);
                let mut mutators = tools
                    .iter()
                    .filter(|spec| spec.name.starts_with(&prefix))
                    .filter(|spec| {
                        let name = spec.name.to_ascii_lowercase();
                        ["write", "delete", "send", "create", "update"]
                            .iter()
                            .any(|verb| name.contains(verb))
                    })
                    .map(|spec| spec.name.clone())
                    .collect::<Vec<_>>();
                mutators.sort();
                if !mutators.is_empty() {
                    warnings.push(format!(
                        "MCP server '{}' is declared stateless but exposes write-like tools: {}",
                        config.name,
                        mutators.join(", ")
                    ));
                }
            }
            servers.push(ServerClient {
                config,
                client: RwLock::new(client),
            });
        }
        if servers.is_empty() {
            return Err(Error::Tool(format!(
                "every MCP server failed to connect: {}",
                warnings.join("; ")
            )));
        }
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        resources.sort_by(|a, b| (&a.server, &a.uri).cmp(&(&b.server, &b.uri)));
        Ok(Self {
            servers,
            tools,
            routes,
            warnings,
            resources,
        })
    }

    /// Non-fatal problems seen while connecting: skipped optional servers, and
    /// servers whose filters left nothing exposed. Callers should surface these,
    /// a silently toolless rollout is hard to diagnose from metrics alone.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn resources(&self) -> &[ResourceSpec] {
        &self.resources
    }

    async fn call_once(
        client: &Client,
        remote_name: &str,
        arguments: &serde_json::Value,
    ) -> std::result::Result<rmcp::model::CallToolResult, String> {
        let arguments = object_arguments(arguments)?;
        client
            .call_tool(
                CallToolRequestParams::new(remote_name.to_owned())
                    .with_arguments(arguments.into_iter().collect::<JsonObject>()),
            )
            .await
            .map_err(|error| error.to_string())
    }

    fn normalize_result(
        call: &ToolCall,
        result: rmcp::model::CallToolResult,
        max_bytes: usize,
    ) -> ToolResult {
        let mut writer = LimitedWriter::new(max_bytes);
        for (index, block) in result.content.iter().enumerate() {
            if index > 0 {
                let _ = writer.write_all(b"\n");
            }
            match block {
                ContentBlock::Text(text) => {
                    let _ = writer.write_all(text.text.as_bytes());
                }
                other => {
                    let _ = serde_json::to_writer(&mut writer, other);
                }
            }
        }
        if let Some(structured) = result.structured_content {
            if !result.content.is_empty() {
                let _ = writer.write_all(b"\n");
            }
            let _ = serde_json::to_writer(&mut writer, &structured);
        }
        let mut content = String::from_utf8_lossy(&writer.bytes).into_owned();
        if writer.truncated {
            content.push(' ');
        }
        if content.len() > max_bytes {
            truncate_utf8(&mut content, max_bytes);
        }
        ToolResult {
            call_id: call.id.clone(),
            content,
            is_error: result.is_error.unwrap_or(false),
        }
    }

    pub async fn shutdown(&self) {
        for server in &self.servers {
            let _ = server
                .client
                .write()
                .await
                .close_with_timeout(Duration::from_secs(3))
                .await;
        }
    }
}

fn object_arguments(
    arguments: &serde_json::Value,
) -> std::result::Result<serde_json::Map<String, serde_json::Value>, String> {
    arguments
        .as_object()
        .cloned()
        .ok_or_else(|| "tool arguments must be a JSON object".to_owned())
}

#[async_trait]
impl ToolProvider for McpToolProvider {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(self.tools.clone())
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        let Some(route) = self.routes.get(&call.name) else {
            return Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("unknown MCP tool '{}'", call.name),
                is_error: true,
            });
        };
        let server = &self.servers[route.server];
        let timeout = Duration::from_secs(server.config.tool_timeout_secs);
        let first = {
            let client = server.client.read().await;
            tokio::time::timeout(
                timeout,
                Self::call_once(&client, &route.remote_name, &call.arguments),
            )
            .await
        };
        let result = match first {
            Ok(Ok(result)) => result,
            Err(_) => {
                return Ok(ToolResult {
                    call_id: call.id.clone(),
                    content: format!(
                        "MCP tool '{}' timed out after {} seconds",
                        call.name, server.config.tool_timeout_secs
                    ),
                    is_error: true,
                });
            }
            Ok(Err(first_error)) => {
                // One bounded reconnect/retry for a closed transport.
                let reconnect = connect_client(&server.config).await;
                let Ok(new_client) = reconnect else {
                    return Ok(ToolResult {
                        call_id: call.id.clone(),
                        content: format!("MCP call failed: {first_error}; reconnect failed"),
                        is_error: true,
                    });
                };
                let mut client = server.client.write().await;
                *client = new_client;
                match tokio::time::timeout(
                    timeout,
                    Self::call_once(&client, &route.remote_name, &call.arguments),
                )
                .await
                {
                    Ok(Ok(result)) => result,
                    Ok(Err(error)) => {
                        return Ok(ToolResult {
                            call_id: call.id.clone(),
                            content: format!(
                                "MCP call failed after reconnect: {error} (initial error: {first_error})"
                            ),
                            is_error: true,
                        });
                    }
                    Err(_) => {
                        return Ok(ToolResult {
                            call_id: call.id.clone(),
                            content: "MCP call timed out after reconnect".into(),
                            is_error: true,
                        });
                    }
                }
            }
        };
        Ok(Self::normalize_result(
            call,
            result,
            server.config.max_tool_result_bytes,
        ))
    }
}

async fn connect_client(config: &McpServerConfig) -> Result<Client> {
    match &config.transport {
        McpTransport::Unspecified => Err(Error::invalid(format!(
            "MCP server '{}' has no transport after configuration merge",
            config.name
        ))),
        McpTransport::Stdio { command, env } => {
            let (program, args) = command
                .split_first()
                .ok_or_else(|| Error::invalid("MCP stdio command must not be empty"))?;
            let mut process = tokio::process::Command::new(program);
            process.args(args);
            if let Some(names) = &config.env_passthrough {
                process.env_clear();
                // `PATH` and `HOME` are part of what "run this command" means,
                // not part of what the allowlist is protecting: dropping them
                // makes the very common `command = ["bunx", …]` fail to spawn,
                // and the operator reads that as "start MCP server 'x'" with no
                // hint that the allowlist caused it.
                for name in ["PATH", "HOME"]
                    .into_iter()
                    .chain(names.iter().map(String::as_str))
                {
                    if let Some(value) = std::env::var_os(name) {
                        process.env(name, value);
                    }
                }
            }
            process.envs(env);
            if let Some(cwd) = &config.cwd {
                process.current_dir(cwd);
            }
            let transport = TokioChildProcess::new(process).map_err(|error| {
                Error::Tool(format!("start MCP server '{}': {error}", config.name))
            })?;
            ().serve(transport).await.map_err(|error| {
                Error::Tool(format!("initialize MCP server '{}': {error}", config.name))
            })
        }
        McpTransport::StreamableHttp { url, headers } => {
            let custom_headers = headers
                .iter()
                .map(|(name, value)| {
                    let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                        Error::invalid(format!("invalid MCP header name: {error}"))
                    })?;
                    let value = HeaderValue::from_str(value).map_err(|error| {
                        Error::invalid(format!("invalid MCP header value: {error}"))
                    })?;
                    Ok((name, value))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.clone());
            transport_config.custom_headers = custom_headers;
            let transport = StreamableHttpClientTransport::from_config(transport_config);
            ().serve(transport).await.map_err(|error| {
                Error::Tool(format!("initialize MCP server '{}': {error}", config.name))
            })
        }
    }
}

fn truncate_utf8(content: &mut String, max_bytes: usize) {
    retrograd_agent_core::text::truncate_utf8(content, max_bytes, "\n[tool result truncated]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_truncation_preserves_utf8_and_marks_the_cut() {
        // 20 two-byte characters (40 bytes). A naive byte-cut at max_bytes-marker.len()=7
        // would land in the middle of the 4th character; the boundary must back off to 6.
        let mut text = "é".repeat(20);
        truncate_utf8(&mut text, 31);
        assert!(text.is_char_boundary(text.len()));
        assert!(text.starts_with("ééé"));
        assert!(!text.starts_with("éééé"));
        assert!(text.ends_with("[tool result truncated]"));

        let mut text = "abcdefghijklmnopqrstuvwxyz".to_owned();
        truncate_utf8(&mut text, 24);
        assert!(text.ends_with("[tool result truncated]"));
    }

    #[tokio::test]
    async fn an_unreachable_optional_server_is_skipped_but_a_required_one_fails() {
        let missing =
            || McpServerConfig::stdio("gone", ["/definitely/not/a/binary"]).with_timeout(1);
        let error = McpToolProvider::connect(vec![missing()])
            .await
            .err()
            .expect("expected a failure");
        assert!(error.to_string().contains("gone"), "{error}");

        // All-optional and all down is still an error: a rollout with no tools is
        // not the run that was asked for.
        let error = McpToolProvider::connect(vec![missing().optional()])
            .await
            .err()
            .expect("expected a failure");
        assert!(
            error.to_string().contains("every MCP server failed"),
            "{error}"
        );
    }

    #[test]
    fn call_once_requires_object_arguments_before_using_the_transport() {
        let error = object_arguments(&serde_json::json!(["not", "an", "object"])).unwrap_err();
        assert!(error.contains("arguments must be a JSON object"));
    }
}
