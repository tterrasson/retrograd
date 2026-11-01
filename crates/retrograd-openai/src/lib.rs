//! `retrograd-openai`: the OpenAI-compatible protocol over a Retrograd model.
//!
//! `GET /models`, `GET /models/{model}` and the generation surfaces, for
//! whatever an id names: a checkpoint of a finished run, the adapter a run is
//! training right now, a model given on a command line. The crate knows the
//! protocol and how to answer one request with one model; what an id *means*
//! is a [`ModelSource`], written by the frontend that has ids to give -
//! the run server, or `retrograd serve`.
//!
//! The crate is organized around one internal request,
//! [`GenerationRequest`]: every surface translates its wire body into it, and
//! every surface is served by the same backend - the [`Session`] for weights on
//! disk, a live run's own thread for a run that is training. The surfaces live
//! in their own subtrees: [`chat`] (Chat Completions), [`responses`] (the
//! Responses API, with its store), [`completions`] (the legacy completions)
//! and [`embeddings`]. Beside them, [`systemone`] serves llama-server's typed
//! decisions to a decision model, through the same session.
//!
//! The answer is rendered and parsed by the same code as the agentic rollouts
//! (`retrograd_agent::rendering`), so a model evaluated through this endpoint
//! is evaluated on the prompt format it was trained on.

pub mod auth;
pub mod backend;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod grammar;
pub mod render;
pub mod request;
pub mod responses;
pub mod session;
pub mod stream;
pub mod systemone;
pub mod testing;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::routing::get;

pub use backend::{EngineLoader, Loader, ServeBackend, ServingBackend, adapter_name};
pub use chat::convert::{Prepared, prepare};
pub use chat::wire::{ChatCompletionRequest, ModelCard, ModelList};
pub use error::{ErrorType, OpenAiEnvelope, OpenAiError};
pub use render::{
    ChatModel, Completion, ParserCache, PromptModel, Rendered, Sampled, complete, finish,
    render_prompt,
};
pub use request::GenerationRequest;
pub use responses::store::{
    FileStore, MemoryStore, ResponseStore, StoreOptions, StoreOwner, StoredResponse,
};
pub use session::{
    DecisionOptions, DeviceLease, EngineStats, Lease, ServingJob, Session, SessionOptions,
    Unshared, WeightsSpec,
};

/// The header that reports how many calls the parser could not read, when some
/// could not. The request still succeeds: the raw text is the content, as a
/// rollout keeps it.
pub const PARSE_ERRORS_HEADER: &str = "x-retrograd-tool-parse-errors";

/// What the model ids of one frontend designate.
#[async_trait]
pub trait ModelSource: Send + Sync + 'static {
    /// Every id resolvable now.
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError>;
    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError>;
}

/// Where a request for one id goes.
#[derive(Clone)]
pub enum Target {
    /// Weights on disk, served by the [`Session`].
    Cold {
        spec: WeightsSpec,
        /// Which weights these are, for `system_fingerprint`: a checkpoint id,
        /// `base`, the adapter's name.
        fingerprint: String,
    },
    /// A run that is training: the request is executed on the run's own
    /// thread, at its next progress callback.
    Live {
        runner: Arc<dyn LiveRunner>,
        /// Sample the model the run started from, its weights with the adapter
        /// disabled, never its anchor, instead of the weights being trained.
        base: bool,
    },
}

/// Work handed to the thread that owns a trainer.
pub type TrainerJob = Box<dyn FnOnce(&mut retrograd_engine::Trainer) + Send>;

/// A run that can lend its trainer to one job.
#[async_trait]
pub trait LiveRunner: Send + Sync {
    /// Runs `job` on the live trainer and returns which weights it ran with,
    /// for `system_fingerprint`. An `Err` means the job did not run.
    async fn run(&self, job: TrainerJob) -> Result<String, OpenAiError>;
}

/// The routes, and what they answer with.
#[derive(Clone)]
pub struct Endpoint {
    pub source: Arc<dyn ModelSource>,
    pub session: Arc<Session>,
    /// The longest a request may wait for its answer, load included.
    pub timeout: Duration,
    /// Off, every route answers 404 in the OpenAI envelope.
    pub enabled: bool,
    /// Where `store: true` responses live. The default is a bounded in-memory
    /// LRU ([`MemoryStore`]); a file store is chosen by the frontend's config.
    pub store: Arc<dyn ResponseStore>,
}

/// `GET /models`, `GET /models/{model}`, the generation surfaces
/// (`POST /chat/completions`, `POST /responses`, `POST /completions`,
/// `POST /embeddings`) and the decisions (`POST /systemone`), relative to wherever the caller mounts them - `/v1`
/// for every OpenAI client.
///
/// No timeout layer belongs over these: a stream stays open until its answer
/// exists, and the bound is [`Endpoint::timeout`], applied by the handler.
pub fn router(endpoint: Endpoint) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/models/{*model}", get(get_model))
        .merge(chat::router())
        .merge(responses::router())
        .merge(completions::router())
        .merge(embeddings::router())
        .merge(systemone::router())
        .with_state(endpoint)
        .layer(axum::middleware::map_response(async |response| {
            error::normalize_response(response)
        }))
}

/// The Retrograd extension of `GET /v1/models`: the capabilities this server
/// serves, published only where they are validated (the contract tests). Not
/// part of the standard OpenAI contract; standard fields are untouched.
#[derive(serde::Serialize)]
struct ServerCapabilities {
    /// The Responses API stores responses (bounded in-memory LRU by default).
    responses_store: bool,
    /// `POST /v1/completions` (legacy) is served.
    completions: bool,
    /// `POST /v1/embeddings` is served; whether one model answers embeddings is
    /// decided per request (a model that declares no pooling is refused).
    embeddings: bool,
    /// `POST /v1/systemone` is served; whether one model answers it is decided
    /// per request (a model that is not a decision model is refused, 501).
    systemone: bool,
}

async fn list_models(State(endpoint): State<Endpoint>) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let data = endpoint.source.list().await?;
    Ok(axum::Json(ModelList {
        object: "list",
        data,
        capabilities: Some(
            serde_json::to_value(ServerCapabilities {
                responses_store: true,
                completions: true,
                embeddings: true,
                systemone: true,
            })
            .unwrap_or(serde_json::Value::Null),
        ),
    })
    .into_response())
}

async fn get_model(
    State(endpoint): State<Endpoint>,
    Path(model): Path<String>,
) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let card = endpoint
        .source
        .list()
        .await?
        .into_iter()
        .find(|card| card.id == model)
        .ok_or_else(|| OpenAiError::model_not_found(&model, "it is not listed by /v1/models"))?;
    Ok(axum::Json(card).into_response())
}

impl Endpoint {
    pub(crate) fn check_enabled(&self) -> Result<(), OpenAiError> {
        match self.enabled {
            true => Ok(()),
            false => Err(OpenAiError::disabled()),
        }
    }
}

/// One SSE `data:` line of a JSON document.
pub(crate) fn json_event(value: &impl serde::Serialize) -> axum::response::sse::Event {
    // A document of owned strings and numbers always serializes.
    axum::response::sse::Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

pub(crate) fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// A source with exactly one model, and its base under a second id: what a
/// process serving one adapter from a command line has to offer.
pub struct SingleModel {
    pub name: String,
    pub spec: WeightsSpec,
    /// The same model without its adapter, served as `base`. `None` when the
    /// served model has no adapter to take off.
    pub base: Option<WeightsSpec>,
    pub created: u64,
}

/// The id [`SingleModel`] serves its base under.
pub const BASE_MODEL_ID: &str = "base";

#[async_trait]
impl ModelSource for SingleModel {
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError> {
        let mut cards = vec![ModelCard::new(&self.name, self.created)];
        if self.base.is_some() {
            cards.push(ModelCard::new(BASE_MODEL_ID, self.created));
        }
        Ok(cards)
    }

    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError> {
        match (model, &self.base) {
            (model, _) if model == self.name => Ok(Target::Cold {
                spec: self.spec.clone(),
                fingerprint: self.name.clone(),
            }),
            (BASE_MODEL_ID, Some(base)) => Ok(Target::Cold {
                spec: base.clone(),
                fingerprint: BASE_MODEL_ID.to_owned(),
            }),
            _ => Err(OpenAiError::model_not_found(
                model,
                format!("this server serves `{}`", self.name),
            )),
        }
    }
}
