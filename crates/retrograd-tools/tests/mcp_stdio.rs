#![allow(clippy::unwrap_used)]
// A helper outside a `#[test]` body, which is what `allow-unwrap-in-tests`
// covers. Same reasoning, said where the configuration cannot reach.

use std::time::Duration;

use retrograd_tools::{McpServerConfig, McpToolProvider, ToolCall, ToolProvider};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};

const HELPER_ENV: &str = "RETROGRAD_MCP_ECHO_HELPER";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct EchoRequest {
    text: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct AddRequest {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone)]
struct EchoServer {
    // Read by the `#[tool_router]`/`#[tool_handler]` macro expansion, never by
    // a line written here - which is what `dead_code` sees.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl EchoServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl EchoServer {
    #[tool(description = "Echo text")]
    fn echo(&self, Parameters(request): Parameters<EchoRequest>) -> String {
        request.text
    }

    #[tool(description = "Add integers")]
    fn add(&self, Parameters(request): Parameters<AddRequest>) -> String {
        (request.a + request.b).to_string()
    }

    #[tool(description = "Sleep for two seconds")]
    async fn slow(&self) -> String {
        tokio::time::sleep(Duration::from_secs(2)).await;
        "awake".into()
    }
}

#[tool_handler]
impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Retrograd MCP integration test server")
    }
}

#[tokio::test]
async fn mcp_echo_helper() {
    if std::env::var(HELPER_ENV).as_deref() != Ok("1") {
        return;
    }
    let server = EchoServer::new()
        .serve(rmcp::transport::stdio())
        .await
        .unwrap();
    server.waiting().await.unwrap();
}

fn server_config(name: &str) -> McpServerConfig {
    let executable = std::env::current_exe().unwrap();
    let config = McpServerConfig::stdio(
        name,
        [
            executable.to_string_lossy().into_owned(),
            "--exact".into(),
            "mcp_echo_helper".into(),
            "--quiet".into(),
            "--nocapture".into(),
            "--test-threads".into(),
            "1".into(),
        ],
    )
    .with_env([(HELPER_ENV, "1")])
    .with_timeout(1)
    .with_max_result_bytes(1024);
    if name == "first" {
        config.allow(["echo"])
    } else {
        config
    }
}

#[tokio::test]
async fn stdio_provider_lists_namespaces_calls_and_times_out() {
    let provider = McpToolProvider::connect(vec![server_config("first"), server_config("second")])
        .await
        .unwrap();
    let mut names = provider
        .list_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
    names.sort();
    assert!(names.contains(&"first__echo".to_owned()));
    assert!(names.contains(&"second__add".to_owned()));
    assert!(
        !names.contains(&"first__add".to_owned()),
        "the server allow-list must be applied before namespacing"
    );

    let result = provider
        .call(&ToolCall {
            id: "a".into(),
            name: "first__echo".into(),
            arguments: serde_json::json!({"text": "hello"}),
        })
        .await
        .unwrap();
    assert_eq!(result.content, "hello");
    assert!(!result.is_error);

    let timeout = provider
        .call(&ToolCall {
            id: "b".into(),
            name: "second__slow".into(),
            arguments: serde_json::json!({}),
        })
        .await
        .unwrap();
    assert!(timeout.is_error);
    assert!(timeout.content.contains("timed out"));

    provider.shutdown().await;
}
