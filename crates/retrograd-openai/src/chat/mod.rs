//! The Chat Completions surface: `POST /chat/completions`.
//!
//! The wire documents are [`wire`], the translation to the common
//! [`GenerationRequest`] is [`convert`], and
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
    ChunkChoice, Delta, FunctionOut, Logprobs, ModelCard, ModelList, Stop, TokenLogprobs,
    ToolCallDelta, ToolCallOut, TopLogprob, Usage, WireMessage,
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
    let (answers, system_fingerprint) = endpoint.sample(target, request).await?;
    let mut choices = Vec::with_capacity(answers.len());
    let mut usage = wire::Usage::default();
    let mut parse_errors = 0;
    for (index, sampled) in answers.into_iter().enumerate() {
        let completion = finish(sampled, &stop);
        parse_errors += completion.parse_errors;
        usage.prompt_tokens = completion.usage.prompt_tokens;
        usage.completion_tokens += completion.usage.completion_tokens;
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        choices.push(wire::Choice {
            // Bounded by the conversion (n <= 128): the narrowing is exact.
            index: index as u32,
            message: wire::AssistantMessage {
                role: "assistant",
                content: completion.content,
                tool_calls: completion.tool_calls,
            },
            finish_reason: completion.finish_reason,
            logprobs: completion.logprobs,
        });
    }
    let body = wire::ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model,
        system_fingerprint,
        choices,
        usage,
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
    /// One answer per choice when the request asked for `n > 1`.
    pub(crate) async fn sample(
        &self,
        target: Target,
        request: GenerationRequest,
    ) -> Result<(Vec<Sampled>, String), OpenAiError> {
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
                    Ok((vec![sampled], fingerprint))
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
                Ok((answers, system_fingerprint)) => {
                    let meta = sse::Meta { id, created, model, system_fingerprint };
                    // A live run answers a single choice (refused otherwise).
                    for (index, sampled) in answers.into_iter().enumerate() {
                        let completion = finish(sampled, &stop);
                        for chunk in sse::chunks(&meta, index as u32, &completion, include_usage) {
                            yield Ok::<_, Infallible>(json_event(&chunk));
                        }
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
        let expected = request.n as usize;
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
                    // One chunker per choice (`n > 1` streams one `choices`
                    // entry per index); the role goes out as soon as the
                    // request is admitted: the first byte the client can
                    // measure.
                    let mut chunkers: std::collections::HashMap<usize, sse::Chunker> =
                        std::collections::HashMap::new();
                    let mut roles_sent: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut ended: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut completion_tokens = 0_u64;
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
                            // Closing without every choice having ended is an
                            // interruption, not a successful answer.
                            Ok(None) => {
                                if ended.len() < expected {
                                    yield Ok(json_event(&serde_json::json!({"error": OpenAiError::internal(
                                        "the serving thread stopped before the request ended"
                                    ).body()})));
                                }
                                break;
                            }
                            Ok(Some(event)) => match event {
                                Event::Token(token) => {
                                    completion_tokens += 1;
                                    let index = token.index;
                                    if roles_sent.insert(index) {
                                        let chunker = chunkers.entry(index).or_insert_with(|| {
                                            sse::Chunker::new(
                                                meta.clone(),
                                                // Bounded by the conversion (n <= 128).
                                                index as u32,
                                                reader.clone(),
                                                prompt_tokens,
                                                include_usage,
                                            )
                                        });
                                        yield Ok::<_, Infallible>(json_event(&chunker.role()));
                                    }
                                    let logprobs = token.sampled_logprob.map(|logprob| {
                                        wire::TokenLogprobs {
                                            token: String::from_utf8_lossy(&token.piece).into_owned(),
                                            logprob,
                                            bytes: token.piece.clone(),
                                            top_logprobs: token.top_logprobs.iter().map(|top| {
                                                wire::TopLogprob {
                                                    token: String::from_utf8_lossy(&top.piece).into_owned(),
                                                    logprob: top.logprob,
                                                    bytes: top.piece.clone(),
                                                }
                                            }).collect(),
                                        }
                                    });
                                    if let Some(chunker) = chunkers.get_mut(&index) {
                                        for chunk in chunker.token(&token.text, logprobs) {
                                            yield Ok::<_, Infallible>(json_event(&chunk));
                                        }
                                    }
                                }
                                Event::End { index, reason, text } => {
                                    match reason {
                                        FinishReason::Cancelled => {
                                            yield Ok(json_event(&serde_json::json!({"error": OpenAiError::internal(
                                                "the request was cancelled while it was being served"
                                            ).body()})));
                                        }
                                        _ => {
                                            ended.insert(index);
                                            if let Some(chunker) = chunkers.get_mut(&index) {
                                                for chunk in chunker.end(reason, &text) {
                                                    if chunk.choices.is_empty() { continue; }
                                                    yield Ok::<_, Infallible>(json_event(&chunk));
                                                }
                                            }
                                        }
                                    }
                                    if ended.len() >= expected {
                                        if include_usage {
                                            yield Ok(json_event(&wire::ChatCompletionChunk {
                                                id: meta.id.clone(),
                                                object: "chat.completion.chunk",
                                                created: meta.created,
                                                model: meta.model.clone(),
                                                system_fingerprint: meta.system_fingerprint.clone(),
                                                choices: Vec::new(),
                                                usage: Some(Some(wire::Usage {
                                                    prompt_tokens: prompt_tokens as u64,
                                                    completion_tokens,
                                                    total_tokens: prompt_tokens as u64 + completion_tokens,
                                                })),
                                            }));
                                        }
                                        break;
                                    }
                                }
                                Event::Error(error) => {
                                    yield Ok(json_event(&serde_json::json!({"error": OpenAiError::from(error).body()})));
                                    break;
                                }
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
