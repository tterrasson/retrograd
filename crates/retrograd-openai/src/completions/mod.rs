//! The legacy Completions API: `POST /v1/completions`.
//!
//! The prompt is served raw - no chat template ([`GenerationRequest::raw_prompt`]),
//! which is what evaluation harnesses in `local-completions` mode expect - with
//! the same sampling contract as chat and the same backend split: weights on
//! disk stream live through the [`Session`](crate::session::Session), a live run
//! answers buffered on its own thread.

pub mod convert;
pub mod wire;

pub use convert::{Prepared, prepare};
pub use wire::{
    CompletionRequest, LegacyLogprobs, TextChoice, TextCompletion, TextCompletionChunk,
};

use std::collections::HashMap;
use std::convert::Infallible;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use retrograd_infer::{Event, FinishReason};
use serde_json::{Map, Value, json};

use crate::chat::wire::{self as chat_wire, Usage};
use crate::error::OpenAiError;
use crate::render::finish;
use crate::session::{ServingJob, WeightsSpec};
use crate::{Endpoint, Target, json_event, unix_seconds};

/// How often a stream that is still waiting for its answer says so.
const KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(15);

/// How many decoded events the forwarder between the session's blocking
/// stream and this async one may buffer.
const EVENT_FORWARD: usize = 64;

/// The completions routes, relative to wherever the caller mounts them - `/v1`
/// for every OpenAI client.
pub fn router() -> Router<Endpoint> {
    Router::new().route("/completions", post(create))
}

async fn create(State(endpoint): State<Endpoint>, body: Bytes) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let request = serde_json::from_slice::<wire::CompletionRequest>(&body).map_err(|error| {
        OpenAiError::invalid(format!("the body is not a completions request: {error}"))
    })?;
    let prepared = prepare(request)?;
    let target = endpoint.source.resolve(&prepared.model).await?;
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds();
    if prepared.stream {
        return Ok(stream(endpoint, target, prepared, id, created).into_response());
    }
    let mut choices = Vec::new();
    let mut usage = Usage::default();
    let mut index = 0_u32;
    for prompt in &prepared.prompts {
        let request = prepared.request(prompt);
        let stop = request.stop.clone();
        let (answers, _) = endpoint.sample(target.clone(), request).await?;
        // One prefill per prompt, shared by its `n` choices: the prompt's
        // tokens count once, not once per choice.
        if let Some(first) = answers.first() {
            usage.prompt_tokens += first.prompt_tokens as u64;
        }
        for sampled in answers {
            let completion = finish(sampled, &stop);
            let body = completion.content.clone().unwrap_or_default();
            // `echo` prepends the prompt exactly as sent.
            let text = match prepared.echo {
                true => format!("{prompt}{body}"),
                false => body,
            };
            usage.completion_tokens += completion.usage.completion_tokens;
            choices.push(wire::TextChoice {
                index,
                text,
                logprobs: completion.logprobs.as_ref().map(legacy_logprobs),
                finish_reason: completion.finish_reason,
            });
            index += 1;
        }
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    Ok(axum::Json(wire::TextCompletion {
        id,
        object: "text_completion",
        created,
        model: prepared.model,
        choices,
        usage,
    })
    .into_response())
}

/// The legacy logprobs shape of one completion: parallel per-token arrays, the
/// byte offset of each token in the returned text, and a token→logprob map of
/// the top pairs.
fn legacy_logprobs(logprobs: &chat_wire::Logprobs) -> wire::LegacyLogprobs {
    let mut tokens = Vec::with_capacity(logprobs.content.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.content.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.content.len());
    let mut text_offset = Vec::with_capacity(logprobs.content.len());
    let mut offset = 0_usize;
    for entry in &logprobs.content {
        tokens.push(entry.token.clone());
        token_logprobs.push(entry.logprob);
        let map: Map<String, Value> = entry
            .top_logprobs
            .iter()
            .map(|top| {
                (
                    top.token.clone(),
                    json!(if top.logprob.is_finite() {
                        top.logprob
                    } else {
                        -9999.0
                    }),
                )
            })
            .collect();
        top_logprobs.push(map);
        text_offset.push(offset);
        offset += entry.bytes.len();
    }
    wire::LegacyLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// The legacy logprobs shape of one streamed token.
fn legacy_logprobs_chunk(entry: &chat_wire::TokenLogprobs, offset: usize) -> wire::LegacyLogprobs {
    let map: Map<String, Value> = entry
        .top_logprobs
        .iter()
        .map(|top| {
            (
                top.token.clone(),
                json!(if top.logprob.is_finite() {
                    top.logprob
                } else {
                    -9999.0
                }),
            )
        })
        .collect();
    wire::LegacyLogprobs {
        tokens: vec![entry.token.clone()],
        token_logprobs: vec![entry.logprob],
        top_logprobs: vec![map],
        text_offset: vec![offset],
    }
}

/// The streamed answer: every prompt in turn, its `n` choices streamed from
/// the session's event stream, then `[DONE]` - the legacy surface's own
/// terminator, unlike Responses.
fn stream(
    endpoint: Endpoint,
    target: Target,
    prepared: Prepared,
    id: String,
    created: u64,
) -> Response {
    match target {
        Target::Cold { spec, .. } => stream_events(endpoint, spec, prepared, id, created),
        Target::Live { .. } => stream_buffered(endpoint, target, prepared, id, created),
    }
}

fn stream_buffered(
    endpoint: Endpoint,
    target: Target,
    prepared: Prepared,
    id: String,
    created: u64,
) -> Response {
    let events = async_stream::stream! {
        let mut index = 0_u32;
        for prompt in &prepared.prompts {
            let request = prepared.request(prompt);
            let stop = request.stop.clone();
            match endpoint.sample(target.clone(), request).await {
                Ok((answers, _)) => {
                    for sampled in answers {
                        let completion = finish(sampled, &stop);
                        let body = completion.content.clone().unwrap_or_default();
                        let text = match prepared.echo {
                            true => format!("{prompt}{body}"),
                            false => body,
                        };
                        let chunk = wire::TextCompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created,
                            model: prepared.model.clone(),
                            choices: vec![wire::ChunkChoice {
                                index,
                                text,
                                finish_reason: Some(completion.finish_reason),
                                logprobs: completion.logprobs.as_ref().map(legacy_logprobs),
                            }],
                        };
                        index += 1;
                        yield Ok::<_, Infallible>(json_event(&chunk));
                    }
                }
                Err(error) => {
                    yield Ok(json_event(&json!({"error": error.body()})));
                }
            }
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
        .into_response()
}

/// The live sequence: each prompt's event stream is consumed as it is
/// produced, one chunk per decoded piece; a client that disconnects cancels
/// the request (the forwarder drops the session stream).
fn stream_events(
    endpoint: Endpoint,
    spec: WeightsSpec,
    prepared: Prepared,
    id: String,
    created: u64,
) -> Response {
    let events = async_stream::stream! {
        let mut index = 0_u32;
        'prompts: for prompt in &prepared.prompts {
            let request = prepared.request(prompt);
            let expected = request.n as usize;
            let deadline = tokio::time::Instant::now() + endpoint.timeout;
            let opened = tokio::time::timeout_at(
                deadline,
                endpoint.session.open(spec.clone(), request),
            )
            .await;
            let job = match opened {
                Err(_) => {
                    yield Ok::<_, Infallible>(json_event(&json!({"error": OpenAiError::timeout(format!(
                        "no answer within {}s", endpoint.timeout.as_secs()
                    )).body()})));
                    break 'prompts;
                }
                Ok(Err(error)) => {
                    yield Ok(json_event(&json!({"error": error.body()})));
                    break 'prompts;
                }
                Ok(Ok(job)) => job,
            };
            let ServingJob { stream, reader: _, prompt_tokens: _ } = job;
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Event>(EVENT_FORWARD);
            tokio::task::spawn_blocking(move || {
                for event in stream {
                    if tx.blocking_send(event).is_err() {
                        break;
                    }
                }
            });
            // Per choice of this prompt: whether it ended, and its byte offset
            // so far (for the streamed logprobs' text_offset).
            if prepared.echo {
                for choice in 0..prepared.n {
                    yield Ok::<_, Infallible>(json_event(&wire::TextCompletionChunk {
                        id: id.clone(), object: "text_completion", created,
                        model: prepared.model.clone(),
                        choices: vec![wire::ChunkChoice {
                            index: index + choice, text: prompt.clone(),
                            finish_reason: None, logprobs: None,
                        }],
                    }));
                }
            }
            let mut ended: HashMap<usize, bool> = HashMap::new();
            let mut offsets: HashMap<usize, usize> = HashMap::new();
            loop {
                let next = tokio::time::timeout_at(deadline, rx.recv()).await;
                match next {
                    Err(_) => {
                        yield Ok(json_event(&json!({"error": OpenAiError::timeout(format!(
                            "no answer within {}s", endpoint.timeout.as_secs()
                        )).body()})));
                        break 'prompts;
                    }
                    Ok(None) => {
                        if ended.len() < expected {
                            yield Ok(json_event(&json!({"error": OpenAiError::internal(
                                "the serving thread stopped before the request ended"
                            ).body()})));
                        }
                        break;
                    }
                    Ok(Some(event)) => match event {
                        Event::Token(token) => {
                            let choice = token.index;
                            let logprobs = token.sampled_logprob.map(|logprob| {
                                let offset = offsets.entry(choice).or_insert(0);
                                let entry = chat_wire::TokenLogprobs {
                                    token: String::from_utf8_lossy(&token.piece).into_owned(),
                                    logprob,
                                    bytes: token.piece.clone(),
                                    top_logprobs: token.top_logprobs.iter().map(|top| {
                                        chat_wire::TopLogprob {
                                            token: String::from_utf8_lossy(&top.piece).into_owned(),
                                            logprob: top.logprob,
                                            bytes: top.piece.clone(),
                                        }
                                    }).collect(),
                                };
                                let chunk = legacy_logprobs_chunk(&entry, *offset);
                                *offset += entry.bytes.len();
                                chunk
                            });
                            if !token.text.is_empty() || logprobs.is_some() {
                                yield Ok::<_, Infallible>(json_event(&wire::TextCompletionChunk {
                                    id: id.clone(),
                                    object: "text_completion",
                                    created,
                                    model: prepared.model.clone(),
                                    choices: vec![wire::ChunkChoice {
                                        // n <= MAX_N; prompt count <= 256.
                                        index: index + choice as u32,
                                        text: token.text,
                                        finish_reason: None,
                                        logprobs,
                                    }],
                                }));
                            }
                        }
                        Event::End { index: choice, reason, text } => {
                            ended.insert(choice, true);
                            let finish_reason = match reason {
                                FinishReason::Eog | FinishReason::Stop => chat_wire::FinishReason::Stop,
                                _ => chat_wire::FinishReason::Length,
                            };
                            if !text.is_empty() || reason != FinishReason::Cancelled {
                                yield Ok::<_, Infallible>(json_event(&wire::TextCompletionChunk {
                                    id: id.clone(),
                                    object: "text_completion",
                                    created,
                                    model: prepared.model.clone(),
                                    choices: vec![wire::ChunkChoice {
                                        // n <= MAX_N; prompt count <= 256.
                                        index: index + choice as u32,
                                        text,
                                        finish_reason: Some(finish_reason),
                                        logprobs: None,
                                    }],
                                }));
                            }
                            if ended.len() >= expected {
                                break;
                            }
                        }
                        Event::Error(error) => {
                            yield Ok(json_event(&json!({"error": OpenAiError::from(error).body()})));
                            break 'prompts;
                        }
                    },
                }
            }
            // n <= MAX_N; prompt count <= 256.
            index += expected as u32;
            drop(rx);
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
        .into_response()
}
