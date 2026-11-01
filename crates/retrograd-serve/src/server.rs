//! The HTTP server: the OpenAI routes over the serving session, plus
//! `GET /v1/engine/stats` - outside the OpenAI contract - reporting the
//! session's scheduler: prefix-cache counters, pool occupancy, queue and live
//! lengths, and the active adapter.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::get;
use retrograd_openai::{
    Endpoint, EngineStats, ModelSource, OpenAiError, Session, SessionOptions, WeightsSpec,
};
use serde::Serialize;

/// Longest a request may wait for its answer. Generous: a CPU answering 512
/// tokens is slow, and a client that wants less says so with its own timeout.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// What the preload callback receives: the load duration, or the error message.
pub type LoadOutcome = std::result::Result<Duration, String>;

/// What `retrograd serve` and the standalone binary both run.
pub struct Serve {
    /// What the model ids name.
    pub source: Arc<dyn ModelSource>,
    /// The weights loaded before the socket answers anything.
    pub preload: WeightsSpec,
    pub session: SessionOptions,
    pub address: SocketAddr,
    /// Bearer token; when set, every route requires it.
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// Largest chat request accepted.
    pub max_body_bytes: usize,
    /// Where `store: true` responses live: `None` for the bounded in-memory
    /// LRU (no write ever touches the disk), a directory for the on-disk store.
    pub responses_store: Option<std::path::PathBuf>,
    /// Called once the preload finished, with the load duration or the error:
    /// what a CLI draws its spinner from.
    pub on_loaded: Option<Arc<dyn Fn(LoadOutcome) + Send + Sync>>,
}

/// The server-sent shape of [`EngineStats`].
#[derive(Serialize)]
pub struct EngineStatsDto {
    pub model: String,
    pub adapter: Option<String>,
    pub adapters: Vec<String>,
    pub queued: usize,
    pub live: usize,
    pub kv: KvStatsDto,
    pub memory: MemoryDto,
}

#[derive(Serialize)]
pub struct KvStatsDto {
    pub hits: u64,
    pub misses: u64,
    pub prompt_cells: u64,
    pub prefilled_cells: u64,
    pub evictions: u64,
    pub swaps: u64,
    pub nodes: usize,
    pub resident_slots: usize,
    pub snapshot_bytes: u64,
}

#[derive(Serialize)]
pub struct MemoryDto {
    pub n_slots: u32,
    pub cells_total: u64,
    pub cells_used: u64,
    pub kv_bytes: u64,
    pub recurrent_bytes: u64,
}

impl From<EngineStats> for EngineStatsDto {
    fn from(stats: EngineStats) -> Self {
        Self {
            model: stats.spec.model.to_string_lossy().into_owned(),
            adapter: stats.adapter,
            adapters: stats.adapters,
            queued: stats.queued,
            live: stats.live,
            kv: KvStatsDto {
                hits: stats.kv.hits,
                misses: stats.kv.misses,
                prompt_cells: stats.kv.prompt_cells,
                prefilled_cells: stats.kv.prefilled_cells,
                evictions: stats.kv.evictions,
                swaps: stats.kv.swaps,
                nodes: stats.kv.nodes,
                resident_slots: stats.kv.resident_slots,
                snapshot_bytes: stats.kv.snapshot_bytes,
            },
            memory: MemoryDto {
                n_slots: stats.memory.n_slots,
                cells_total: stats.memory.cells_total,
                cells_used: stats.memory.cells_used,
                kv_bytes: stats.memory.kv_bytes,
                recurrent_bytes: stats.memory.recurrent_bytes,
            },
        }
    }
}

/// Serves until interrupted. Loads `serve.preload` before the socket accepts,
/// so a model that does not load fails here rather than on a client's first
/// request.
pub async fn serve(serve: Serve) -> Result<(), ServeError> {
    crate::config::check_network_rule(serve.address, serve.api_key.as_deref())?;
    let listener = tokio::net::TcpListener::bind(serve.address).await?;
    let session = Session::new(
        Arc::new(retrograd_openai::EngineLoader),
        Arc::new(retrograd_openai::Unshared),
        serve.session,
    )?;
    let started = std::time::Instant::now();
    let preload = session.preload(serve.preload).await;
    if let Some(on_loaded) = &serve.on_loaded {
        on_loaded(
            preload
                .as_ref()
                .map(|_| started.elapsed())
                .map_err(|error| error.to_string()),
        );
    }
    preload?;
    let session = Arc::new(session);

    let store: std::sync::Arc<dyn retrograd_openai::ResponseStore> = match &serve.responses_store {
        Some(dir) => std::sync::Arc::new(retrograd_openai::FileStore::new(
            dir,
            retrograd_openai::StoreOptions::default(),
        )?),
        None => std::sync::Arc::new(retrograd_openai::MemoryStore::default()),
    };
    let endpoint = Endpoint {
        source: serve.source,
        session: session.clone(),
        timeout: serve.timeout,
        enabled: true,
        store,
    };
    let router = build_router(endpoint, serve.api_key.as_deref(), serve.max_body_bytes);

    tracing::info!(address = %serve.address, "serving");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// Builds the production HTTP surface without loading weights or binding a socket.
pub fn build_router(endpoint: Endpoint, api_key: Option<&str>, max_body_bytes: usize) -> Router {
    let session = endpoint.session.clone();
    let mut v1 = retrograd_openai::router(endpoint);
    // Outside the OpenAI contract: the serving session's own report.
    v1 = v1.merge(
        Router::new()
            .route("/engine/stats", get(engine_stats))
            .with_state(session.clone()),
    );
    if max_body_bytes > 0 {
        v1 = v1.layer(axum::extract::DefaultBodyLimit::max(max_body_bytes));
    }
    if let Some(key) = api_key {
        v1 = v1.layer(axum::middleware::from_fn_with_state(
            Arc::<str>::from(key),
            retrograd_openai::auth::require_api_key,
        ));
    }
    Router::new()
        .nest("/v1", v1)
        .layer(axum::middleware::map_response(async |response| {
            retrograd_openai::error::normalize_response(response)
        }))
}

async fn engine_stats(
    State(session): State<Arc<Session>>,
) -> Result<Json<EngineStatsDto>, OpenAiError> {
    session
        .stats()
        .await
        .map(EngineStatsDto::from)
        .map(Json)
        .ok_or_else(|| OpenAiError::not_found("no model is loaded"))
}

/// The failures of the serving runtime.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error(transparent)]
    OpenAi(#[from] OpenAiError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("config: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("{0}")]
    Usage(String),
}

/// The CLI's façade: an OpenAI error keeps the side of the fault its status
/// code names, a bad configuration stays a configuration error.
impl From<ServeError> for retrograd_core::Error {
    fn from(error: ServeError) -> Self {
        match error {
            ServeError::OpenAi(error) => match error.status.is_client_error() {
                true => Self::invalid(error.message),
                false => Self::runtime(error.message),
            },
            ServeError::Io(error) => Self::Io(error),
            ServeError::Config(error) => Self::config(error.to_string()),
            ServeError::Usage(message) => Self::invalid(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http::{Request, StatusCode, header};
    use http_body_util::BodyExt;
    use retrograd_openai::testing::FakeLoader;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn endpoint() -> Endpoint {
        let session = Session::new(
            Arc::new(FakeLoader::new("hello")),
            Arc::new(retrograd_openai::Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        Endpoint {
            source: Arc::new(retrograd_openai::SingleModel {
                name: "base".into(),
                spec: WeightsSpec {
                    model: "unused.gguf".into(),
                    adapter: None,
                    n_ctx: 4096,
                    device: retrograd_core::Device::Cpu,
                    n_seq_max: None,
                    chat_template_variables: None,
                },
                base: None,
                created: 1,
            }),
            session: Arc::new(session),
            timeout: Duration::from_secs(5),
            enabled: true,
            store: Arc::new(retrograd_openai::MemoryStore::default()),
        }
    }

    async fn send(
        router: &Router,
        request: Request<Body>,
    ) -> (StatusCode, http::HeaderMap, Vec<u8>) {
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        assert!(
            status.is_success()
                || response
                    .extensions()
                    .get::<retrograd_openai::OpenAiEnvelope>()
                    .is_some()
        );
        let headers = response.headers().clone();
        let body = tokio::time::timeout(Duration::from_secs(10), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes()
            .to_vec();
        (status, headers, body)
    }

    fn request(method: &str, path: &str, body: impl Into<Body>) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header(header::AUTHORIZATION, "bEaReR test-key")
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.into())
            .unwrap()
    }

    #[tokio::test]
    async fn routing_and_body_rejections_keep_the_openai_envelope_and_allow_header() {
        let router = build_router(endpoint(), Some("test-key"), 128);
        for (method, path, body, expected) in [
            (
                "GET",
                "/v1/chat/completions",
                "",
                StatusCode::METHOD_NOT_ALLOWED,
            ),
            ("POST", "/v1/responses", "{", StatusCode::BAD_REQUEST),
            ("GET", "/v1/models/%FF", "", StatusCode::BAD_REQUEST),
            ("GET", "/v1/unknown", "", StatusCode::NOT_FOUND),
            (
                "POST",
                "/v1/embeddings",
                &"x".repeat(129),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let (status, headers, bytes) =
                send(&router, request(method, path, body.to_owned())).await;
            assert_eq!(status, expected, "{path}");
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert!(body["error"]["message"].is_string(), "{body}");
            if status == StatusCode::METHOD_NOT_ALLOWED {
                assert_eq!(headers[header::ALLOW], "POST");
            }
        }
        // No Content-Length header: the extractor still bounds a streamed body.
        let body = Body::from_stream(futures_util::stream::iter([
            Ok::<_, std::io::Error>("x".repeat(64)),
            Ok("x".repeat(65)),
        ]));
        let (status, _, _) = send(&router, request("POST", "/v1/chat/completions", body)).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn every_surface_and_engine_stats_require_the_key() {
        let router = build_router(endpoint(), Some("test-key"), 4096);
        for path in [
            "/v1/models",
            "/v1/engine/stats",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/embeddings",
        ] {
            let (status, headers, _) =
                send(&router, Request::get(path).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{path}");
            assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");
        }
    }

    #[tokio::test]
    async fn all_openai_surfaces_work_through_the_production_router() {
        let router = build_router(endpoint(), Some("test-key"), 4096);
        let (status, _, bytes) = send(&router, request("GET", "/v1/models", Body::empty())).await;
        assert_eq!(status, StatusCode::OK);
        let models: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(models["data"][0]["id"], "base");
        for (path, body, object) in [
            (
                "/v1/chat/completions",
                json!({"model":"base", "messages":[{"role":"user", "content":"hi"}]}),
                "chat.completion",
            ),
            (
                "/v1/completions",
                json!({"model":"base", "prompt":"hi"}),
                "text_completion",
            ),
            (
                "/v1/embeddings",
                json!({"model":"base", "input":"hi"}),
                "list",
            ),
            (
                "/v1/responses",
                json!({"model":"base", "input":"hi"}),
                "response",
            ),
        ] {
            let (status, _, bytes) = send(&router, request("POST", path, body.to_string())).await;
            let response: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(status, StatusCode::OK, "{path}: {response}");
            assert_eq!(response["object"], object);
            if object == "response" {
                let id = response["id"].as_str().unwrap();
                let (status, _, _) = send(
                    &router,
                    request("GET", &format!("/v1/responses/{id}"), Body::empty()),
                )
                .await;
                assert_eq!(status, StatusCode::OK);
                let (status, _, bytes) = send(
                    &router,
                    request(
                        "POST",
                        path,
                        json!({"model":"base", "input":"again", "previous_response_id":id})
                            .to_string(),
                    ),
                )
                .await;
                assert_eq!(
                    status,
                    StatusCode::OK,
                    "{}",
                    String::from_utf8_lossy(&bytes)
                );
            }
        }
        let (status, _, _) = send(&router, request("GET", "/v1/engine/stats", Body::empty())).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn generation_streams_finish_through_the_production_router() {
        let router = build_router(endpoint(), Some("test-key"), 4096);
        for (path, body, terminal) in [
            (
                "/v1/chat/completions",
                json!({"model":"base", "messages":[{"role":"user", "content":"hi"}], "stream":true}),
                "data: [DONE]",
            ),
            (
                "/v1/completions",
                json!({"model":"base", "prompt":"hi", "stream":true}),
                "data: [DONE]",
            ),
            (
                "/v1/responses",
                json!({"model":"base", "input":"hi", "stream":true}),
                "event: response.completed",
            ),
        ] {
            let (status, headers, bytes) =
                send(&router, request("POST", path, body.to_string())).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
            let text = String::from_utf8(bytes).unwrap();
            assert!(text.contains(terminal), "{path}: {text}");
        }
    }

    #[tokio::test]
    async fn public_entry_point_checks_network_auth_before_loading_or_binding() {
        let endpoint = endpoint();
        let target = endpoint.source.resolve("base").await.unwrap();
        let retrograd_openai::Target::Cold { spec, .. } = target else {
            panic!("cold")
        };
        let error = serve(Serve {
            source: endpoint.source,
            preload: spec,
            session: SessionOptions::default(),
            address: "0.0.0.0:0".parse().unwrap(),
            api_key: None,
            timeout: Duration::from_secs(1),
            max_body_bytes: 4096,
            responses_store: None,
            on_loaded: None,
        })
        .await
        .unwrap_err();
        assert!(matches!(error, ServeError::Config(_)), "{error}");
    }
}
