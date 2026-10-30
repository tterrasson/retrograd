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
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use http::HeaderValue;
use retrograd_infer::{Event, FinishReason};

use crate::error::OpenAiError;
use crate::render::{ParserCache, Sampled, complete, finish};
use crate::request::GenerationRequest;
use crate::session::{ServingJob, WeightsSpec};
use crate::{Endpoint, PARSE_ERRORS_HEADER, Target, TrainerJob, json_event, unix_seconds};

/// How often a stream that is still waiting for its answer says so.
const KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(15);

/// How many decoded events the forwarder between the session's blocking
/// stream and this async one may buffer.
const EVENT_FORWARD: usize = 64;

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
                    let (reply, answer) = tokio::sync::oneshot::channel();
                    let job: TrainerJob =
                        Box::new(move |trainer: &mut retrograd_engine::Trainer| {
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
    /// when the answer starts, so a failure from then on is an `error` event.
    ///
    /// A cold target is streamed live: the session's event stream is consumed
    /// as the session thread decodes, one chunk per stable delta. A live run
    /// keeps the buffered sequence - its answer exists only once its thread is
    /// free, and slicing it then is what it can offer.
    fn stream(
        &self,
        target: Target,
        request: GenerationRequest,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    ) -> Response {
        match target {
            Target::Cold { spec, fingerprint } => self
                .stream_events(
                    spec,
                    fingerprint,
                    request,
                    id,
                    created,
                    model,
                    include_usage,
                )
                .into_response(),
            Target::Live { .. } => self
                .stream_buffered(target, request, id, created, model, include_usage)
                .into_response(),
        }
    }

    /// The buffered sequence: the whole answer exists, cut by [`sse::chunks`].
    fn stream_buffered(
        &self,
        target: Target,
        request: GenerationRequest,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    ) -> Response {
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
            yield Ok(SseEvent::default().data("[DONE]"));
        };
        Sse::new(events)
            .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
            .into_response()
    }

    /// The live sequence: the session's event stream is consumed as it is
    /// produced, and a client that disconnects cancels the request - dropping
    /// the forwarding receiver makes the forwarder drop the event stream, and
    /// the scheduler removes the request at its next step, returning its slot.
    #[allow(clippy::too_many_arguments)]
    fn stream_events(
        &self,
        spec: WeightsSpec,
        fingerprint: String,
        request: GenerationRequest,
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    ) -> Response {
        let endpoint = self.clone();
        let events = async_stream::stream! {
            let meta = sse::Meta { id, created, model, system_fingerprint: fingerprint };
            let deadline = tokio::time::Instant::now() + endpoint.timeout;
            let opened = tokio::time::timeout_at(deadline, endpoint.session.open(spec, request)).await;
            match opened {
                Err(_) => {
                    yield Ok::<_, Infallible>(json_event(&serde_json::json!({"error": OpenAiError::timeout(format!(
                        "no answer within {}s",
                        endpoint.timeout.as_secs()
                    )).body()})));
                }
                Ok(Err(error)) => {
                    yield Ok(json_event(&serde_json::json!({"error": error.body()})));
                }
                Ok(Ok(job)) => {
                    let ServingJob { stream, prompt_tokens, reader } = job;
                    // The session's stream is blocking; a forwarder moves its
                    // events into this async stream. When the client goes away,
                    // the receiver is dropped, the forwarder's send fails, and
                    // the event stream is dropped with it - which is what
                    // cancels the request at the scheduler's next step.
                    let (tx, mut rx) = tokio::sync::mpsc::channel::<Event>(EVENT_FORWARD);
                    tokio::task::spawn_blocking(move || {
                        for event in stream {
                            if tx.blocking_send(event).is_err() {
                                break;
                            }
                        }
                    });
                    let mut chunker = sse::Chunker::new(meta, reader, prompt_tokens, include_usage);
                    // The role goes out as soon as the request is admitted: the
                    // first byte the client can measure.
                    yield Ok::<_, Infallible>(json_event(&chunker.role()));
                    loop {
                        let next = tokio::time::timeout_at(deadline, rx.recv()).await;
                        match next {
                            Err(_) => {
                                yield Ok(json_event(&serde_json::json!({"error": OpenAiError::timeout(format!(
                                    "no answer within {}s",
                                    endpoint.timeout.as_secs()
                                )).body()})));
                                break;
                            }
                            // Closing without an End event is an interruption,
                            // not a successful answer.
                            Ok(None) => {
                                yield Ok(json_event(&serde_json::json!({"error": OpenAiError::internal(
                                    "the serving thread stopped before the request ended"
                                ).body()})));
                                break;
                            }
                            Ok(Some(event)) => match event {
                                Event::Token(token) if token.index == 0 => {
                                    for chunk in chunker.token(&token.text) {
                                        yield Ok::<_, Infallible>(json_event(&chunk));
                                    }
                                }
                                Event::End { index: 0, reason, text } => {
                                    match reason {
                                        FinishReason::Cancelled => {
                                            yield Ok(json_event(&serde_json::json!({"error": OpenAiError::internal(
                                                "the request was cancelled while it was being served"
                                            ).body()})));
                                        }
                                        _ => {
                                            for chunk in chunker.end(reason, &text) {
                                                yield Ok::<_, Infallible>(json_event(&chunk));
                                            }
                                        }
                                    }
                                    break;
                                }
                                Event::Error(error) => {
                                    yield Ok(json_event(&serde_json::json!({"error": OpenAiError::from(error).body()})));
                                    break;
                                }
                                _ => {}
                            },
                        }
                    }
                    // Do not wait for a blocking backend to produce another
                    // event: the response must close within its own deadline.
                    // The detached forwarder drops the session stream as soon
                    // as its next send observes the closed receiver.
                    drop(rx);
                }
            }
            yield Ok(SseEvent::default().data("[DONE]"));
        };
        Sse::new(events)
            .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
            .into_response()
    }
}
