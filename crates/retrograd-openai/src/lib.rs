//! `retrograd-openai`: the OpenAI chat-completions contract over a Retrograd
//! model.
//!
//! `GET /models`, `GET /models/{model}` and `POST /chat/completions`, for
//! whatever an id names: a checkpoint of a finished run, the adapter a run is
//! training right now, a model given on a command line. The crate knows the
//! protocol and how to answer one request with one model; what an id *means*
//! is a [`ModelSource`], written by the frontend that has ids to give -
//! the run server, or `retrograd serve`.
//!
//! The answer is rendered and parsed by the same code as the agentic rollouts
//! ([`retrograd_agent::rendering`]), so a model evaluated through this endpoint
//! is evaluated on the prompt format it was trained on.

pub mod auth;
pub mod convert;
pub mod error;
pub mod render;
pub mod session;
pub mod sse;
pub mod testing;
pub mod wire;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use http::HeaderValue;
use retrograd_engine::Trainer;
use tokio::sync::oneshot;

pub use convert::{ChatPrompt, Prepared, prepare};
pub use error::{ErrorType, OpenAiEnvelope, OpenAiError};
pub use render::{ChatModel, Completion, ParserCache, Sampled, complete, finish};
pub use session::{
    DeviceLease, Lease, Loader, Session, SessionOptions, TrainerLoader, Unshared, WeightsSpec,
};
pub use wire::{ChatCompletionRequest, ModelCard, ModelList};

/// The header that reports how many calls the parser could not read, when some
/// could not. The request still succeeds: the raw text is the content, as a
/// rollout keeps it.
pub const PARSE_ERRORS_HEADER: &str = "x-retrograd-tool-parse-errors";

/// How often a stream that is still waiting for its answer says so.
const KEEP_ALIVE: Duration = Duration::from_secs(15);

/// What the model ids of one frontend designate.
#[async_trait]
pub trait ModelSource: Send + Sync + 'static {
    /// Every id resolvable now.
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError>;
    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError>;
}

/// Where a request for one id goes.
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
pub type TrainerJob = Box<dyn FnOnce(&mut Trainer) + Send>;

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
}

/// `GET /models`, `GET /models/{model}`, `POST /chat/completions`, relative to
/// wherever the caller mounts them - `/v1` for every OpenAI client.
///
/// No timeout layer belongs over these: a stream stays open until its answer
/// exists, and the bound is [`Endpoint::timeout`], applied by the handler.
pub fn router(endpoint: Endpoint) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/models/{*model}", get(get_model))
        .route("/chat/completions", post(chat_completions))
        .with_state(endpoint)
}

async fn list_models(State(endpoint): State<Endpoint>) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let data = endpoint.source.list().await?;
    Ok(axum::Json(ModelList {
        object: "list",
        data,
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

async fn chat_completions(
    State(endpoint): State<Endpoint>,
    body: Bytes,
) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let request = serde_json::from_slice::<ChatCompletionRequest>(&body).map_err(|error| {
        OpenAiError::invalid(format!(
            "the body is not a chat completion request: {error}"
        ))
    })?;
    let Prepared {
        model,
        prompt,
        stream,
        include_usage,
    } = prepare(request)?;
    // Resolved before anything is sent, so an unknown id is a 404 even on a
    // stream.
    let target = endpoint.source.resolve(&model).await?;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds();
    if stream {
        return Ok(endpoint
            .stream(target, prompt, id, created, model, include_usage)
            .into_response());
    }
    let stop = prompt.stop.clone();
    let (sampled, system_fingerprint) = endpoint.sample(target, prompt).await?;
    let completion = finish(sampled, &stop);
    let parse_errors = completion.parse_errors;
    let body = wire::ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model,
        system_fingerprint,
        choices: vec![wire::Choice {
            index: 0,
            message: wire::AssistantMessage {
                role: "assistant",
                content: completion.content,
                tool_calls: completion.tool_calls,
            },
            finish_reason: completion.finish_reason,
        }],
        usage: completion.usage,
    };
    let mut response = axum::Json(body).into_response();
    if parse_errors > 0 {
        response
            .headers_mut()
            .insert(PARSE_ERRORS_HEADER, HeaderValue::from(parse_errors));
    }
    Ok(response)
}

impl Endpoint {
    fn check_enabled(&self) -> Result<(), OpenAiError> {
        match self.enabled {
            true => Ok(()),
            false => Err(OpenAiError::disabled()),
        }
    }

    /// Samples `prompt` with whatever `target` names, within the bound.
    async fn sample(
        &self,
        target: Target,
        prompt: ChatPrompt,
    ) -> Result<(Sampled, String), OpenAiError> {
        let work = async {
            match target {
                Target::Cold { spec, fingerprint } => {
                    Ok((self.session.complete(spec, prompt).await?, fingerprint))
                }
                Target::Live { runner, base } => {
                    let (reply, answer) = oneshot::channel();
                    let job: TrainerJob = Box::new(move |trainer: &mut Trainer| {
                        let outcome = complete(trainer, &prompt, base, &mut ParserCache::default());
                        let _ = reply.send(outcome);
                    });
                    let fingerprint = runner.run(job).await?;
                    let sampled = answer.await.map_err(|_| {
                        OpenAiError::internal("the run reported the request served, and it was not")
                    })??;
                    Ok((sampled, fingerprint))
                }
            }
        };
        tokio::time::timeout(self.timeout, work)
            .await
            .map_err(|_| {
                OpenAiError::timeout(format!(
                    "no answer within {}s; a live run answers at its next progress callback",
                    self.timeout.as_secs()
                ))
            })?
    }

    /// The answer as a server-sent event stream. The status is already sent
    /// when the answer arrives, so a failure from then on is an `error` event.
    fn stream(
        &self,
        target: Target,
        prompt: ChatPrompt,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    ) -> impl IntoResponse {
        let endpoint = self.clone();
        let stop = prompt.stop.clone();
        let events = async_stream::stream! {
            match endpoint.sample(target, prompt).await {
                Ok((sampled, system_fingerprint)) => {
                    let meta = sse::Meta { id, created, model, system_fingerprint };
                    let completion = finish(sampled, &stop);
                    for chunk in sse::chunks(&meta, &completion, include_usage) {
                        yield Ok::<_, Infallible>(json_event(&chunk));
                    }
                }
                Err(error) => {
                    yield Ok(json_event(&serde_json::json!({"error": error.body()})));
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
    }
}

fn json_event(value: &impl serde::Serialize) -> Event {
    // A document of owned strings and numbers always serializes.
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

fn unix_seconds() -> u64 {
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
