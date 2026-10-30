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

    let endpoint = Endpoint {
        source: serve.source,
        session: session.clone(),
        timeout: serve.timeout,
        enabled: true,
    };
    let mut v1 = retrograd_openai::router(endpoint);
    // Outside the OpenAI contract: the serving session's own report.
    v1 = v1.merge(
        Router::new()
            .route("/engine/stats", get(engine_stats))
            .with_state(session.clone()),
    );
    if serve.max_body_bytes > 0 {
        v1 = v1.layer(axum::extract::DefaultBodyLimit::max(serve.max_body_bytes));
    }
    if let Some(key) = serve.api_key.as_deref() {
        v1 = v1.layer(axum::middleware::from_fn_with_state(
            Arc::<str>::from(key),
            retrograd_openai::auth::require_api_key,
        ));
    }
    let router = Router::new().nest("/v1", v1);

    tracing::info!(address = %serve.address, "serving");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
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
