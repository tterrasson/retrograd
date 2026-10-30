//! The Chat Completions surface: `POST /chat/completions`.
//!
//! The wire documents are [`wire`], the translation to the common
//! [`GenerationRequest`](crate::request::GenerationRequest) is [`convert`], and
//! the streamed answer is cut into chunks by [`sse`]. All three branch on the
//! same backend: the [`Session`](crate::session::Session) for weights on disk,
//! the live run's own thread for a run that is training.

pub mod convert;
pub mod sse;
pub mod wire;

pub use convert::{Prepared, prepare};
pub use sse::Meta;
pub use wire::{
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, Choice,
    ChunkChoice, Delta, FunctionOut, ModelCard, ModelList, Stop, ToolCallDelta, ToolCallOut, Usage,
    WireMessage,
};

use std::convert::Infallible;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use http::HeaderValue;
use retrograd_engine::Trainer;
use tokio::sync::oneshot;

use crate::error::OpenAiError;
use crate::render::{ParserCache, Sampled, complete, finish};
use crate::request::GenerationRequest;
use crate::{Endpoint, PARSE_ERRORS_HEADER, Target, TrainerJob, json_event, unix_seconds};

/// How often a stream that is still waiting for its answer says so.
const KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(15);

/// The chat routes, relative to wherever the caller mounts them - `/v1` for
/// every OpenAI client.
pub fn router() -> Router<Endpoint> {
    Router::new().route("/chat/completions", post(chat_completions))
}

async fn chat_completions(
    State(endpoint): State<Endpoint>,
    body: Bytes,
) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let request =
        serde_json::from_slice::<wire::ChatCompletionRequest>(&body).map_err(|error| {
            OpenAiError::invalid(format!(
                "the body is not a chat completion request: {error}"
            ))
        })?;
    let Prepared {
        model,
        request,
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
            .stream(target, request, id, created, model, include_usage)
            .into_response());
    }
    let stop = request.stop.clone();
    let (sampled, system_fingerprint) = endpoint.sample(target, request).await?;
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
    /// Samples `request` with whatever `target` names, within the bound.
    async fn sample(
        &self,
        target: Target,
        request: GenerationRequest,
    ) -> Result<(Sampled, String), OpenAiError> {
        let work = async {
            match target {
                Target::Cold { spec, fingerprint } => {
                    Ok((self.session.complete(spec, request).await?, fingerprint))
                }
                Target::Live { runner, base } => {
                    let (reply, answer) = oneshot::channel();
                    let job: TrainerJob = Box::new(move |trainer: &mut Trainer| {
                        let outcome =
                            complete(trainer, &request, base, &mut ParserCache::default());
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
        request: GenerationRequest,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    ) -> impl IntoResponse {
        let endpoint = self.clone();
        let stop = request.stop.clone();
        let events = async_stream::stream! {
            match endpoint.sample(target, request).await {
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
