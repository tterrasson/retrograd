//! The Responses API surface: `POST /v1/responses`, stateless (`store:
//! false` only - storing responses and `previous_response_id` continuations
//! are not served yet, and the request is refused rather than answered as if
//! they were).
//!
//! The wire documents are [`wire`], the translation to the common
//! [`GenerationRequest`](crate::request::GenerationRequest) is [`convert`],
//! and the streamed answer is the typed event sequence of [`events`]. The
//! surface branches on the same backend as chat: the [`Session`](crate::session::Session)
//! for weights on disk (streamed live), a live run's own thread for a run
//! that is training (buffered).

pub mod convert;
pub mod events;
pub mod store;
pub mod wire;

pub use convert::{Prepared, prepare};
pub use events::{ResponseEvent, Sequencer};
pub use wire::{Input, ReasoningParam, Response, ResponsesRequest, Status, Usage};

use std::convert::Infallible;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::routing::post;
use retrograd_infer::{Event, FinishReason};

use crate::error::OpenAiError;
use crate::render::{Completion, finish};
use crate::responses::events::ResponseEvent as Ev;
use crate::responses::wire::{
    EventLogprob, IncompleteDetails, InputTokensDetails, OutputContent, OutputItem,
    OutputTokensDetails, ReasoningTextPart, ResponseError, SummaryPart, TextEcho,
};
use crate::session::{ServingJob, WeightsSpec};
use crate::stream::{StreamDelta, StreamParser};
use crate::{Endpoint, Target, unix_seconds};

fn response_event(event: &Ev) -> axum::response::sse::Event {
    crate::json_event(event).event(event.kind())
}

/// How often a stream that is still waiting for its answer says so.
const KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(15);

/// How many decoded events the forwarder between the session's blocking
/// stream and this async one may buffer.
const EVENT_FORWARD: usize = 64;

/// The responses routes, relative to wherever the caller mounts them - `/v1`
/// for every OpenAI client.
pub fn router() -> Router<Endpoint> {
    Router::new().route("/responses", post(responses_create))
}

/// The fields of the `response` object that do not depend on the answer: the
/// id, the timestamps, the echoes of the effective options.
struct Template {
    id: String,
    created: u64,
    model: String,
    fingerprint: String,
    prepared: convert::Prepared,
}

impl Template {
    fn response(&self, status: Status, output: Vec<OutputItem>, usage: Usage) -> Response {
        let text = output.iter().find_map(|item| match item {
            OutputItem::Message { content, .. } => content.first().map(|part| match part {
                OutputContent::OutputText { text, .. } => text.clone(),
            }),
            _ => None,
        });
        Response {
            id: self.id.clone(),
            object: "response",
            created_at: self.created,
            completed_at: (status == Status::Completed
                || status == Status::Incomplete
                || status == Status::Failed)
                .then_some(self.created),
            status,
            model: self.model.clone(),
            output,
            output_text: text,
            usage,
            error: match status {
                Status::Failed => Some(ResponseError {
                    code: "server_error".into(),
                    message: "the generation failed".into(),
                    r#type: Some("server_error".into()),
                    param: None,
                }),
                _ => None,
            },
            incomplete_details: (status == Status::Incomplete).then_some(IncompleteDetails {
                reason: "max_output_tokens",
            }),
            instructions: self.prepared.instructions.clone(),
            tools: self.prepared.tools.clone(),
            tool_choice: self.prepared.tool_choice.clone(),
            parallel_tool_calls: self.prepared.parallel_tool_calls,
            metadata: self.prepared.metadata.clone(),
            temperature: self.prepared.temperature,
            top_p: self.prepared.top_p,
            top_logprobs: self.prepared.top_logprobs,
            max_output_tokens: self.prepared.max_output_tokens,
            text: TextEcho {
                format: self.prepared.text_format.clone(),
            },
            reasoning: self.prepared.reasoning.clone(),
            store: false,
            background: false,
            previous_response_id: None,
            truncation: "disabled",
            service_tier: "default",
            access_programs: Vec::new(),
        }
    }

    fn usage(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_tokens: usize,
    ) -> Usage {
        let input_tokens = prompt_tokens as u64;
        let output_tokens = completion_tokens as u64;
        Usage {
            input_tokens,
            input_tokens_details: InputTokensDetails {
                // No prefix is pinned across responses yet.
                cached_tokens: 0,
                cache_write_tokens: 0,
            },
            output_tokens,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: reasoning_tokens as u64,
            },
            total_tokens: input_tokens + output_tokens,
        }
    }
}

/// Builds the `output` items of one answer: a `reasoning` item when the
/// parser split a reasoning block out (its raw text is exposed as
/// `reasoning_text` content - a local capability, never an OpenAI summary,
/// which stays empty), the `message` item, and one `function_call` item per
/// call.
fn output_items(completion: &Completion) -> Vec<OutputItem> {
    let mut output = Vec::new();
    if !completion.reasoning.is_empty() {
        output.push(OutputItem::Reasoning {
            id: format!("rs_{}", uuid::Uuid::new_v4().simple()),
            summary: Vec::<SummaryPart>::new(),
            content: vec![ReasoningTextPart {
                kind: "reasoning_text",
                text: completion.reasoning.clone(),
            }],
            status: Some("completed"),
        });
    }
    let show_message = completion.content.is_some() || completion.tool_calls.is_empty();
    if show_message {
        output.push(OutputItem::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            role: "assistant",
            status: "completed",
            content: vec![OutputContent::OutputText {
                text: completion.content.clone().unwrap_or_default(),
                annotations: Vec::new(),
                logprobs: completion
                    .logprobs
                    .as_ref()
                    .map(|logprobs| logprobs.content.clone()),
            }],
        });
    }
    for call in &completion.tool_calls {
        output.push(OutputItem::FunctionCall {
            id: format!("fc_{}", uuid::Uuid::new_v4().simple()),
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
            status: "completed",
        });
    }
    output
}

fn status_of(completion: &Completion) -> Status {
    match completion.finish_reason {
        crate::chat::wire::FinishReason::Length => Status::Incomplete,
        _ => Status::Completed,
    }
}

async fn responses_create(
    State(endpoint): State<Endpoint>,
    body: Bytes,
) -> Result<AxumResponse, OpenAiError> {
    endpoint.check_enabled()?;
    let request = serde_json::from_slice::<wire::ResponsesRequest>(&body).map_err(|error| {
        OpenAiError::invalid(format!("the body is not a responses request: {error}"))
    })?;
    let prepared = prepare(request)?;
    let model = prepared.model.clone();
    let target = endpoint.source.resolve(&model).await?;
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds();
    let template = Template {
        id,
        created,
        model,
        fingerprint: String::new(),
        prepared,
    };
    if template.prepared.stream {
        return Ok(responses_stream(endpoint, target, template).into_response());
    }
    let Prepared { request, .. } = template.prepared.clone();
    let (answers, fingerprint) = endpoint.sample(target, request).await?;
    let mut template = template;
    template.fingerprint = fingerprint;
    // Responses asks for a single completion (`n` is not a field).
    let answer = answers
        .into_iter()
        .next()
        .ok_or_else(|| OpenAiError::internal("the serving thread answered without a request"))?;
    let completion = finish(answer, &[]);
    let status = status_of(&completion);
    let usage = template.usage(
        completion.usage.prompt_tokens as usize,
        completion.usage.completion_tokens as usize,
        completion.reasoning_tokens,
    );
    let output = output_items(&completion);
    Ok(axum::Json(template.response(status, output, usage)).into_response())
}

/// The streamed sequence. A cold target streams live; a live run answers
/// buffered (its answer exists only once its thread is free), and the same
/// event sequence is cut from it whole.
fn responses_stream(endpoint: Endpoint, target: Target, template: Template) -> AxumResponse {
    match target {
        Target::Cold { spec, fingerprint } => {
            let mut template = template;
            template.fingerprint = fingerprint;
            stream_events(endpoint, spec, template).into_response()
        }
        Target::Live { .. } => stream_buffered(endpoint, target, template).into_response(),
    }
}

/// The buffered sequence: the whole answer exists, and the event sequence is
/// cut from it in the protocol's order.
fn stream_buffered(endpoint: Endpoint, target: Target, template: Template) -> AxumResponse {
    let events = async_stream::stream! {
        match endpoint.sample(target, template.prepared.request.clone()).await {
            Ok((answers, fingerprint)) => {
                let mut template = template;
                template.fingerprint = fingerprint;
                let answer = answers.into_iter().next().expect("a live run answers one choice");
                let completion = finish(answer, &[]);
                let status = status_of(&completion);
                let usage = template.usage(
                    completion.usage.prompt_tokens as usize,
                    completion.usage.completion_tokens as usize,
                    completion.reasoning_tokens,
                );
                let output = output_items(&completion);
                let mut sequencer = Sequencer::new();
                for event in buffered_events(&template, &completion, status, output, usage) {
                    if let Some(event) = sequencer.push(event) {
                        yield Ok::<_, Infallible>(response_event(&event));
                    }
                }
            }
            Err(error) => {
                let mut sequencer = Sequencer::new();
                if let Some(event) = sequencer.push(Ev::Error {
                    sequence_number: 0,
                    code: "server_error".into(),
                    message: error.to_string(),
                    param: None,
                }) {
                    yield Ok(response_event(&event));
                }
            }
        }
    };
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
        .into_response()
}

/// The buffered event sequence of one completed answer, in the protocol's
/// order, ending with the single terminal event.
fn buffered_events(
    template: &Template,
    completion: &Completion,
    status: Status,
    output: Vec<OutputItem>,
    usage: Usage,
) -> Vec<Ev> {
    let mut events = Vec::new();
    let mut push = |event: Ev| events.push(event);
    push(Ev::Created {
        sequence_number: 0,
        response: template.response(Status::InProgress, Vec::new(), usage.clone()),
    });
    push(Ev::InProgress {
        sequence_number: 0,
        response: template.response(Status::InProgress, Vec::new(), usage.clone()),
    });
    // Items in order: reasoning, message, calls - one added/done pair each,
    // with the deltas between for the text-bearing ones.
    let mut output_index = 0_u32;
    let reasoning_text = completion.reasoning.clone();
    if !reasoning_text.is_empty() {
        let id = output
            .iter()
            .find_map(|item| match item {
                OutputItem::Reasoning { id, .. } => Some(id.clone()),
                _ => None,
            })
            .expect("output contains the reasoning item");
        push(Ev::OutputItemAdded {
            sequence_number: 0,
            output_index,
            item: OutputItem::Reasoning {
                id: id.clone(),
                summary: Vec::new(),
                content: vec![ReasoningTextPart {
                    kind: "reasoning_text",
                    text: String::new(),
                }],
                status: Some("completed"),
            },
        });
        if !reasoning_text.is_empty() {
            push(Ev::ReasoningTextDelta {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                content_index: 0,
                delta: reasoning_text.clone(),
            });
            push(Ev::ReasoningTextDone {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                content_index: 0,
                text: reasoning_text.clone(),
            });
        }
        push(Ev::OutputItemDone {
            sequence_number: 0,
            output_index,
            item: OutputItem::Reasoning {
                id,
                summary: Vec::new(),
                content: vec![ReasoningTextPart {
                    kind: "reasoning_text",
                    text: reasoning_text,
                }],
                status: Some("completed"),
            },
        });
        output_index += 1;
    }
    if let Some(message) = output
        .iter()
        .find(|item| matches!(item, OutputItem::Message { .. }))
    {
        let OutputItem::Message { id, content, .. } = message else {
            unreachable!()
        };
        let (text, logprobs) = match content.first() {
            Some(OutputContent::OutputText { text, logprobs, .. }) => {
                (text.clone(), logprobs.clone().unwrap_or_default())
            }
            None => (String::new(), Vec::new()),
        };
        push(Ev::OutputItemAdded {
            sequence_number: 0,
            output_index,
            item: message.clone(),
        });
        push(Ev::ContentPartAdded {
            sequence_number: 0,
            item_id: id.clone(),
            output_index,
            content_index: 0,
            part: OutputContent::OutputText {
                text: String::new(),
                annotations: Vec::new(),
                logprobs: None,
            },
        });
        if !text.is_empty() {
            push(Ev::OutputTextDelta {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                content_index: 0,
                delta: text.clone(),
                logprobs: logprobs
                    .iter()
                    .map(|entry| EventLogprob {
                        token: entry.token.clone(),
                        logprob: entry.logprob,
                        top_logprobs: entry
                            .top_logprobs
                            .iter()
                            .map(|top| crate::responses::wire::EventTopLogprob {
                                token: top.token.clone(),
                                logprob: top.logprob,
                            })
                            .collect(),
                    })
                    .collect(),
            });
            push(Ev::OutputTextDone {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                content_index: 0,
                text: text.clone(),
                logprobs: Vec::new(),
            });
        }
        push(Ev::ContentPartDone {
            sequence_number: 0,
            item_id: id.clone(),
            output_index,
            content_index: 0,
            part: OutputContent::OutputText {
                text,
                annotations: Vec::new(),
                logprobs: None,
            },
        });
        push(Ev::OutputItemDone {
            sequence_number: 0,
            output_index,
            item: message.clone(),
        });
        output_index += 1;
    }
    for item in output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall { .. }))
    {
        let OutputItem::FunctionCall { id, arguments, .. } = item else {
            unreachable!()
        };
        push(Ev::OutputItemAdded {
            sequence_number: 0,
            output_index,
            item: item.clone(),
        });
        if !arguments.is_empty() {
            push(Ev::FunctionCallArgumentsDelta {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                delta: arguments.clone(),
            });
            push(Ev::FunctionCallArgumentsDone {
                sequence_number: 0,
                item_id: id.clone(),
                output_index,
                arguments: arguments.clone(),
            });
        }
        push(Ev::OutputItemDone {
            sequence_number: 0,
            output_index,
            item: item.clone(),
        });
        output_index += 1;
    }
    let terminal = match status {
        Status::Completed => Ev::Completed {
            sequence_number: 0,
            response: template.response(status, output, usage),
        },
        Status::Incomplete => Ev::Incomplete {
            sequence_number: 0,
            response: template.response(status, output, usage),
        },
        _ => Ev::Failed {
            sequence_number: 0,
            response: template.response(Status::Failed, Vec::new(), usage),
        },
    };
    push(terminal);
    events
}

/// The live sequence: the session's event stream is consumed as it is
/// produced and each stable delta becomes the protocol's events.
fn stream_events(endpoint: Endpoint, spec: WeightsSpec, template: Template) -> AxumResponse {
    let events = async_stream::stream! {
        let mut builder = StreamBuilder::new(template);
        for event in builder.open() {
            yield Ok::<_, Infallible>(response_event(&event));
        }
        let deadline = tokio::time::Instant::now() + endpoint.timeout;
        let opened = tokio::time::timeout_at(
            deadline,
            endpoint.session.open(spec, builder.template.prepared.request.clone()),
        )
        .await;
        match opened {
            Err(_) => {
                for event in builder.timeout() {
                    yield Ok::<_, Infallible>(response_event(&event));
                }
            }
            Ok(Err(error)) => {
                for event in builder.error(error) {
                    yield Ok::<_, Infallible>(response_event(&event));
                }
            }
            Ok(Ok(job)) => {
                let ServingJob { stream, prompt_tokens, reader } = job;
                let (tx, mut rx) = tokio::sync::mpsc::channel::<Event>(EVENT_FORWARD);
                tokio::task::spawn_blocking(move || {
                    for event in stream {
                        if tx.blocking_send(event).is_err() {
                            break;
                        }
                    }
                });
                let mut parser = reader.map(StreamParser::new);
                let mut tokens = 0_u64;
                let mut reasoning_tokens = 0_u64;
                loop {
                    let next = tokio::time::timeout_at(deadline, rx.recv()).await;
                    match next {
                        Err(_) => {
                            for event in builder.timeout() {
                                yield Ok::<_, Infallible>(response_event(&event));
                            }
                            break;
                        }
                        Ok(None) => {
                            for event in builder.interrupted() {
                                yield Ok::<_, Infallible>(response_event(&event));
                            }
                            break;
                        }
                        Ok(Some(event)) => match event {
                            Event::Token(token) => {
                                tokens += 1;
                                let logprobs = token.sampled_logprob.map(|logprob| {
                                    crate::chat::wire::TokenLogprobs {
                                        token: String::from_utf8_lossy(&token.piece).into_owned(),
                                        logprob,
                                        bytes: token.piece.clone(),
                                        top_logprobs: token.top_logprobs.iter().map(|top| {
                                            crate::chat::wire::TopLogprob {
                                                token: String::from_utf8_lossy(&top.piece).into_owned(),
                                                logprob: top.logprob,
                                                bytes: top.piece.clone(),
                                            }
                                        }).collect(),
                                    }
                                });
                                let deltas = match parser.as_mut() {
                                    Some(parser) => {
                                        let before = parser.parsed_reasoning_len();
                                        let deltas = parser.push(&token.text);
                                        // Attribute this token to the reasoning
                                        // block when the parse grew it.
                                        if parser.parsed_reasoning_len() > before {
                                            reasoning_tokens += 1;
                                        }
                                        deltas
                                    }
                                    None => match token.text.is_empty() {
                                        true => Vec::new(),
                                        false => vec![StreamDelta::Content(token.text.clone())],
                                    },
                                };
                                let mut logprobs = logprobs;
                                for delta in deltas {
                                    let entry = if matches!(delta, StreamDelta::Content(_)) { logprobs.take() } else { None };
                                    for event in builder.push(delta, entry) {
                                        yield Ok::<_, Infallible>(response_event(&event));
                                    }
                                }
                            }
                            Event::End { reason, text, .. } => {
                                let deltas = match parser.as_mut() {
                                    Some(parser) => {
                                        let mut deltas = parser.push(&text);
                                        deltas.extend(parser.finish());
                                        deltas
                                    }
                                    None => match text.is_empty() {
                                        true => Vec::new(),
                                        false => vec![StreamDelta::Content(text.clone())],
                                    },
                                };
                                for delta in deltas {
                                    for event in builder.push(delta, None) {
                                        yield Ok::<_, Infallible>(response_event(&event));
                                    }
                                }
                                let usage = builder.template.usage(
                                    prompt_tokens,
                                    tokens as usize,
                                    reasoning_tokens as usize,
                                );
                                for event in builder.end(reason, usage) {
                                    yield Ok::<_, Infallible>(response_event(&event));
                                }
                                break;
                            }
                            Event::Error(error) => {
                                for event in builder.error(OpenAiError::from(error)) {
                                    yield Ok::<_, Infallible>(response_event(&event));
                                }
                                break;
                            }
                        },
                    }
                }
                drop(rx);
            }
        }
    };
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
        .into_response()
}

/// The live stream's state: the items opened so far (in arrival order), the
/// sequence numbers, and enough of the answer to build the terminal
/// `response`.
struct StreamBuilder {
    template: Template,
    sequencer: Sequencer,
    /// Per output item: its id, its accumulated text (reasoning/message) or
    /// arguments (call), and whether the `added` event went out.
    items: Vec<ItemTrack>,
    /// The message item's accumulated logprobs, when asked for.
    logprobs: Vec<crate::chat::wire::TokenLogprobs>,
    done: bool,
}

struct ItemTrack {
    id: String,
    kind: ItemKind,
    added: bool,
}

enum ItemKind {
    Reasoning(String),
    Message(String),
    Call {
        name: String,
        call_id: String,
        arguments: String,
    },
}

impl ItemTrack {
    fn new(kind: ItemKind) -> Self {
        let prefix = match &kind {
            ItemKind::Reasoning(_) => "rs",
            ItemKind::Message(_) => "msg",
            ItemKind::Call { .. } => "fc",
        };
        Self {
            id: format!("{prefix}_{}", uuid::Uuid::new_v4().simple()),
            kind,
            added: false,
        }
    }

    fn output_item(&self, text: String) -> OutputItem {
        match &self.kind {
            ItemKind::Reasoning(_) => reasoning_item(&self.id, text),
            ItemKind::Message(_) => message_item(&self.id, text, Vec::new()),
            ItemKind::Call {
                name,
                call_id,
                arguments,
            } => call_item(&self.id, name.clone(), call_id.clone(), arguments.clone()),
        }
    }
}

impl StreamBuilder {
    fn new(template: Template) -> Self {
        Self {
            template,
            sequencer: Sequencer::new(),
            items: Vec::new(),
            logprobs: Vec::new(),
            done: false,
        }
    }

    /// `response.created` then `response.in_progress`, with an empty output.
    fn open(&mut self) -> Vec<Ev> {
        let usage = self.template.usage(0, 0, 0);
        vec![
            self.sequencer
                .push(Ev::Created {
                    sequence_number: 0,
                    response: self
                        .template
                        .response(Status::InProgress, Vec::new(), usage.clone()),
                })
                .expect("the sequence is open"),
            self.sequencer
                .push(Ev::InProgress {
                    sequence_number: 0,
                    response: self
                        .template
                        .response(Status::InProgress, Vec::new(), usage),
                })
                .expect("the sequence is open"),
        ]
    }

    /// Feeds one stable delta, emitting the events it produces. Items are
    /// opened on their first delta, in arrival order. `logprobs` is the
    /// sampled token's full logprob entry (with its bytes) when the client
    /// asked for logprobs; the delta event carries the slim form.
    fn push(
        &mut self,
        delta: StreamDelta,
        logprobs: Option<crate::chat::wire::TokenLogprobs>,
    ) -> Vec<Ev> {
        if self.done {
            return Vec::new();
        }
        let mut events = Vec::new();
        match delta {
            StreamDelta::Reasoning(text) => {
                let index = self.track_index(|kind| matches!(kind, ItemKind::Reasoning(_)));
                let index = match index {
                    Some(index) => index,
                    None => {
                        self.items
                            .push(ItemTrack::new(ItemKind::Reasoning(String::new())));
                        self.items.len() - 1
                    }
                };
                let output_index = index as u32;
                let track = &mut self.items[index];
                if !track.added {
                    events.push(
                        self.sequencer
                            .push(Ev::OutputItemAdded {
                                sequence_number: 0,
                                output_index,
                                item: track.output_item(String::new()),
                            })
                            .expect("open"),
                    );
                    track.added = true;
                }
                let ItemKind::Reasoning(existing) = &mut self.items[index].kind else {
                    unreachable!()
                };
                existing.push_str(&text);
                let item_id = self.items[index].id.clone();
                events.push(
                    self.sequencer
                        .push(Ev::ReasoningTextDelta {
                            sequence_number: 0,
                            item_id,
                            output_index,
                            content_index: 0,
                            delta: text,
                        })
                        .expect("open"),
                );
            }
            StreamDelta::Content(text) => {
                let index = match self.track_index(|kind| matches!(kind, ItemKind::Message(_))) {
                    Some(index) => index,
                    None => {
                        self.items
                            .push(ItemTrack::new(ItemKind::Message(String::new())));
                        self.items.len() - 1
                    }
                };
                let output_index = index as u32;
                let track = &mut self.items[index];
                if !track.added {
                    events.push(
                        self.sequencer
                            .push(Ev::OutputItemAdded {
                                sequence_number: 0,
                                output_index,
                                item: track.output_item(String::new()),
                            })
                            .expect("open"),
                    );
                    events.push(
                        self.sequencer
                            .push(Ev::ContentPartAdded {
                                sequence_number: 0,
                                item_id: track.id.clone(),
                                output_index,
                                content_index: 0,
                                part: OutputContent::OutputText {
                                    text: String::new(),
                                    annotations: Vec::new(),
                                    logprobs: None,
                                },
                            })
                            .expect("open"),
                    );
                    track.added = true;
                }
                let ItemKind::Message(existing) = &mut self.items[index].kind else {
                    unreachable!()
                };
                existing.push_str(&text);
                let delta_logprobs = logprobs.as_ref().map(|entry| EventLogprob {
                    token: entry.token.clone(),
                    logprob: entry.logprob,
                    top_logprobs: entry
                        .top_logprobs
                        .iter()
                        .map(|top| crate::responses::wire::EventTopLogprob {
                            token: top.token.clone(),
                            logprob: top.logprob,
                        })
                        .collect(),
                });
                if let Some(logprob) = logprobs {
                    self.logprobs.push(logprob);
                }
                let item_id = self.items[index].id.clone();
                events.push(
                    self.sequencer
                        .push(Ev::OutputTextDelta {
                            sequence_number: 0,
                            item_id,
                            output_index,
                            content_index: 0,
                            delta: text,
                            logprobs: delta_logprobs.into_iter().collect(),
                        })
                        .expect("open"),
                );
            }
            StreamDelta::Call { id, name, .. } => {
                self.items.push(ItemTrack::new(ItemKind::Call {
                    name,
                    call_id: id,
                    arguments: String::new(),
                }));
                let index = self.items.len() - 1;
                let output_index = index as u32;
                let track = &mut self.items[index];
                events.push(
                    self.sequencer
                        .push(Ev::OutputItemAdded {
                            sequence_number: 0,
                            output_index,
                            item: track.output_item(String::new()),
                        })
                        .expect("open"),
                );
                track.added = true;
            }
            StreamDelta::CallArguments { index, delta } => {
                // `index` is the call's position among this turn's calls.
                let Some(position) = self
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| matches!(item.kind, ItemKind::Call { .. }))
                    .map(|(position, _)| position)
                    .nth(index)
                else {
                    return events;
                };
                let output_index = position as u32;
                let track = &mut self.items[position];
                let ItemKind::Call { arguments, .. } = &mut track.kind else {
                    unreachable!()
                };
                arguments.push_str(&delta);
                let item_id = track.id.clone();
                events.push(
                    self.sequencer
                        .push(Ev::FunctionCallArgumentsDelta {
                            sequence_number: 0,
                            item_id,
                            output_index,
                            delta,
                        })
                        .expect("open"),
                );
            }
        }
        events
    }

    /// The completion ended: close every open item, then the single terminal
    /// event carrying the full `response`.
    fn end(&mut self, reason: FinishReason, usage: Usage) -> Vec<Ev> {
        if reason == FinishReason::Cancelled {
            return self.interrupted();
        }
        let mut events = Vec::new();
        for (index, track) in self.items.iter_mut().enumerate() {
            if !track.added {
                continue;
            }
            let output_index = index as u32;
            let item_id = track.id.clone();
            match &mut track.kind {
                ItemKind::Reasoning(text) => {
                    let text = text.clone();
                    events.push(
                        self.sequencer
                            .push(Ev::ReasoningTextDone {
                                sequence_number: 0,
                                item_id,
                                output_index,
                                content_index: 0,
                                text: text.clone(),
                            })
                            .expect("open"),
                    );
                    events.push(
                        self.sequencer
                            .push(Ev::OutputItemDone {
                                sequence_number: 0,
                                output_index,
                                item: reasoning_item(&track.id, text),
                            })
                            .expect("open"),
                    );
                }
                ItemKind::Message(text) => {
                    let text = text.clone();
                    events.push(
                        self.sequencer
                            .push(Ev::OutputTextDone {
                                sequence_number: 0,
                                item_id: item_id.clone(),
                                output_index,
                                content_index: 0,
                                text: text.clone(),
                                logprobs: Vec::new(),
                            })
                            .expect("open"),
                    );
                    events.push(
                        self.sequencer
                            .push(Ev::ContentPartDone {
                                sequence_number: 0,
                                item_id,
                                output_index,
                                content_index: 0,
                                part: OutputContent::OutputText {
                                    text: text.clone(),
                                    annotations: Vec::new(),
                                    logprobs: None,
                                },
                            })
                            .expect("open"),
                    );
                    events.push(
                        self.sequencer
                            .push(Ev::OutputItemDone {
                                sequence_number: 0,
                                output_index,
                                item: message_item(&track.id, text, self.logprobs.clone()),
                            })
                            .expect("open"),
                    );
                }
                ItemKind::Call { arguments, .. } => {
                    let arguments = arguments.clone();
                    events.push(
                        self.sequencer
                            .push(Ev::FunctionCallArgumentsDone {
                                sequence_number: 0,
                                item_id,
                                output_index,
                                arguments: arguments.clone(),
                            })
                            .expect("open"),
                    );
                    let item = track.output_item(String::new());
                    let OutputItem::FunctionCall {
                        arguments: final_arguments,
                        ..
                    } = &item
                    else {
                        unreachable!()
                    };
                    let _ = final_arguments;
                    events.push(
                        self.sequencer
                            .push(Ev::OutputItemDone {
                                sequence_number: 0,
                                output_index,
                                item: {
                                    let mut item = item;
                                    if let OutputItem::FunctionCall {
                                        arguments: slot, ..
                                    } = &mut item
                                    {
                                        *slot = arguments;
                                    }
                                    item
                                },
                            })
                            .expect("open"),
                    );
                }
            }
        }
        let output = self.final_output();
        let status = match reason {
            FinishReason::Eog | FinishReason::Stop => Status::Completed,
            _ => Status::Incomplete,
        };
        let terminal = match status {
            Status::Completed => Ev::Completed {
                sequence_number: 0,
                response: self.template.response(status, output, usage),
            },
            _ => Ev::Incomplete {
                sequence_number: 0,
                response: self.template.response(status, output, usage),
            },
        };
        if let Some(event) = self.sequencer.push(terminal) {
            events.push(event);
        }
        self.done = true;
        events
    }

    fn timeout(&mut self) -> Vec<Ev> {
        self.fail(
            "server_error",
            "no answer within the endpoint's timeout".to_owned(),
        )
    }

    fn interrupted(&mut self) -> Vec<Ev> {
        self.fail(
            "server_error",
            "the serving thread stopped before the request ended".to_owned(),
        )
    }

    fn error(&mut self, error: OpenAiError) -> Vec<Ev> {
        self.fail("server_error", error.to_string())
    }

    fn fail(&mut self, code: &str, message: String) -> Vec<Ev> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        self.sequencer
            .push(Ev::Error {
                sequence_number: 0,
                code: code.to_owned(),
                message,
                param: None,
            })
            .into_iter()
            .collect()
    }

    fn final_output(&self) -> Vec<OutputItem> {
        self.items
            .iter()
            .map(|track| match &track.kind {
                ItemKind::Reasoning(text) => reasoning_item(&track.id, text.clone()),
                ItemKind::Message(text) => {
                    message_item(&track.id, text.clone(), self.logprobs.clone())
                }
                ItemKind::Call {
                    name,
                    call_id,
                    arguments,
                } => call_item(&track.id, name.clone(), call_id.clone(), arguments.clone()),
            })
            .collect()
    }

    fn track_index(&self, matches: impl Fn(&ItemKind) -> bool) -> Option<usize> {
        self.items.iter().position(|item| matches(&item.kind))
    }
}

fn reasoning_item(id: &str, text: String) -> OutputItem {
    OutputItem::Reasoning {
        id: id.to_owned(),
        summary: Vec::new(),
        content: vec![ReasoningTextPart {
            kind: "reasoning_text",
            text,
        }],
        status: Some("completed"),
    }
}

fn message_item(
    id: &str,
    text: String,
    logprobs: Vec<crate::chat::wire::TokenLogprobs>,
) -> OutputItem {
    OutputItem::Message {
        id: id.to_owned(),
        role: "assistant",
        status: "completed",
        content: vec![OutputContent::OutputText {
            text,
            annotations: Vec::new(),
            logprobs: (!logprobs.is_empty()).then_some(logprobs),
        }],
    }
}

fn call_item(id: &str, name: String, call_id: String, arguments: String) -> OutputItem {
    OutputItem::FunctionCall {
        id: id.to_owned(),
        call_id,
        name,
        arguments,
        status: "completed",
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use serde_json::json;

    fn template() -> Template {
        Template {
            id: "resp_test".into(),
            created: 1,
            model: "m".into(),
            fingerprint: String::new(),
            prepared: prepare(
                serde_json::from_value(json!({"model":"m","input":"hi","store":false})).unwrap(),
            )
            .unwrap(),
        }
    }

    #[test]
    fn buffered_reasoning_keeps_its_id_in_the_terminal_response() {
        let completion = Completion {
            content: Some("answer".into()),
            reasoning: "plan".into(),
            tool_calls: Vec::new(),
            finish_reason: crate::chat::wire::FinishReason::Stop,
            usage: Default::default(),
            parse_errors: 0,
            logprobs: None,
            reasoning_tokens: 1,
        };
        let output = output_items(&completion);
        let expected = serde_json::to_value(&output[0]).unwrap()["id"].clone();
        let template = template();
        for event in buffered_events(
            &template,
            &completion,
            Status::Completed,
            output,
            template.usage(1, 2, 1),
        ) {
            let event = serde_json::to_value(event).unwrap();
            if event["item"]["type"] == "reasoning" {
                assert_eq!(event["item"]["id"], expected);
            }
            if event["type"] == "response.completed" {
                assert_eq!(event["response"]["output"][0]["id"], expected);
            }
        }
    }

    #[test]
    fn flushing_a_text_tail_does_not_repeat_the_previous_logprob() {
        let mut builder = StreamBuilder::new(template());
        builder.open();
        builder.push(
            StreamDelta::Content("a".into()),
            Some(crate::chat::wire::TokenLogprobs {
                token: "a".into(),
                bytes: vec![b'a'],
                logprob: -0.5,
                top_logprobs: Vec::new(),
            }),
        );
        let events = builder.push(StreamDelta::Content(" tail".into()), None);
        assert!(events.iter().all(
            |event| !matches!(event, Ev::OutputTextDelta { logprobs, .. } if !logprobs.is_empty())
        ));
    }
}
