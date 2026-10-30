//! The Responses API surface: `POST /v1/responses` with its store
//! (`store: true` is the API default, on the endpoint's [`ResponseStore`]),
//! `previous_response_id` continuations, and the read routes
//! `GET/DELETE /v1/responses/{id}` and `GET /v1/responses/{id}/input_items`.
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
pub use store::{
    FileStore, MemoryStore, ResponseId, ResponseStore, StoreOptions, StoreOwner, StoredResponse,
};
pub use wire::{Input, ReasoningParam, Response, ResponsesRequest, Status, Usage};

use std::convert::Infallible;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Extension, Path, Query, State};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::routing::{get, post};
use retrograd_infer::{Event, FinishReason, Retention};
use serde::Deserialize;
use serde_json::Value;

use crate::error::OpenAiError;
use crate::render::{Completion, finish};
use crate::responses::events::ResponseEvent as Ev;
use crate::responses::store::valid_id;
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
    Router::new()
        .route("/responses", post(responses_create))
        .route(
            "/responses/{id}",
            get(responses_get).delete(responses_delete),
        )
        .route("/responses/{id}/input_items", get(responses_input_items))
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
            store: self.prepared.store,
            background: false,
            previous_response_id: self.prepared.previous_response_id.clone(),
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
        cached_tokens: usize,
    ) -> Usage {
        let input_tokens = prompt_tokens as u64;
        let output_tokens = completion_tokens as u64;
        Usage {
            input_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: cached_tokens as u64,
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

/// What a successful answer needs to be written to the store: the store
/// itself, the ownership and weights identity of the request, and the input
/// items a continuation of this response replays.
struct StoreCtx {
    store: Arc<dyn ResponseStore>,
    owner: Option<String>,
    model: String,
    weights: String,
    fingerprint: String,
    base_model: Option<String>,
    input_items: Vec<Value>,
    previous_response_id: Option<String>,
    store_response: bool,
    created_at: u64,
}

impl StoreCtx {
    /// Writes one finished response to the store. A store that refuses (full,
    /// chain too deep) does not fail the answer: the client got it, and a
    /// later read answers 404 with the eviction message - logged here so the
    /// operator sees the store is the problem.
    fn record(&self, response: &Response) {
        if !self.store_response {
            return;
        }
        let mut input_items = self.input_items.clone();
        ensure_item_ids(&mut input_items);
        let response_value = serde_json::to_value(response).unwrap_or(Value::Null);
        let bytes = response_value.to_string().len()
            + input_items
                .iter()
                .map(|item| item.to_string().len())
                .sum::<usize>();
        let chain_depth = self
            .store
            .get(self.previous_response_id.as_deref().unwrap_or(""))
            .map(|parent| parent.chain_depth + 1)
            .unwrap_or(1);
        let stored = StoredResponse {
            response: response_value,
            input_items,
            created_at: self.created_at,
            bytes,
            previous_response_id: self.previous_response_id.clone(),
            chain_depth,
            model: self.model.clone(),
            weights: self.weights.clone(),
            fingerprint: self.fingerprint.clone(),
            base_model: self.base_model.clone(),
            owner: self.owner.clone(),
        };
        if let Err(error) = self.store.put(stored) {
            tracing::warn!(error = %error, "could not store a response the client asked to keep");
        }
    }
}

/// Gives an `id` to every input item that lacks one, so pagination cursors
/// (`before`/`after`) and the `first_id`/`last_id` of the list envelope can
/// name every item - the way the hosted API assigns item ids server-side.
fn ensure_item_ids(items: &mut [Value]) {
    for (index, item) in items.iter_mut().enumerate() {
        if let Some(object) = item.as_object_mut()
            && !object.contains_key("id")
        {
            object.insert("id".to_owned(), Value::String(format!("item_{index}")));
        }
    }
}

/// The weights identity a target runs on: `cold:<model path>|<adapter>` for
/// weights on disk, `live:<model id>` for a run that is training. A
/// continuation refuses a different base model instead of replaying tokens
/// computed under another one.
fn weights_identity(target: &Target, model: &str) -> (String, Option<String>) {
    match target {
        Target::Cold { spec, .. } => {
            let adapter = spec
                .adapter
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            (
                format!("cold:{}|{adapter}", spec.model.display()),
                Some(spec.model.display().to_string()),
            )
        }
        Target::Live { .. } => (format!("live:{model}"), None),
    }
}

/// Whether `owner` may read a response stored under `stored`'s owner.
fn visible_to(owner: &Option<String>, stored: &StoredResponse) -> bool {
    stored.owner == *owner
}

/// A continuation replays the stored chain only when the model and the base
/// weights are the ones the stored response was produced with. A different
/// adapter or a live run that advanced is fine (its pinned KV would be
/// invalidated - none is pinned yet); a different base model is refused.
fn check_continuation(
    stored: &StoredResponse,
    model: &str,
    target: &Target,
) -> Result<(), OpenAiError> {
    let id = stored
        .response
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("resp_?");
    if stored.model != model {
        return Err(OpenAiError::invalid_param(
            "previous_response_id",
            format!(
                "the response `{id}` was created for model `{}`; continue it with the same model",
                stored.model
            ),
        ));
    }
    if let (Some(stored_base), Target::Cold { spec, .. }) = (&stored.base_model, target)
        && *stored_base != spec.model.display().to_string()
    {
        return Err(OpenAiError::invalid_param(
            "previous_response_id",
            format!(
                "the response `{id}` was produced from the base model `{stored_base}`, but \
                 `{model}` runs on `{}`; a different base model cannot replay its tokens",
                spec.model.display()
            ),
        ));
    }
    Ok(())
}

/// The 404 a `previous_response_id` gets when the store does not know it.
fn evicted_or_unknown(id: &str) -> OpenAiError {
    OpenAiError::response_not_found(
        id,
        "it is unknown to this server's response store, or it was evicted from the in-memory \
         store (it is bounded)",
    )
    .with_param("previous_response_id")
}

async fn responses_create(
    State(endpoint): State<Endpoint>,
    owner: Option<Extension<StoreOwner>>,
    body: Bytes,
) -> Result<AxumResponse, OpenAiError> {
    endpoint.check_enabled()?;
    let owner = owner.map(|Extension(owner)| owner.0);
    let request = serde_json::from_slice::<wire::ResponsesRequest>(&body).map_err(|error| {
        OpenAiError::invalid(format!("the body is not a responses request: {error}"))
    })?;
    let model = request.model.clone();
    let target = endpoint.source.resolve(&model).await?;
    // A continuation merges the stored chain's items in front of the new
    // `input` - without re-injecting the previous request's `instructions`,
    // which stay echoed on the stored response; the new ones apply to the new
    // turn.
    let mut effective = request.clone();
    let mut replay_items: Vec<Value> = Vec::new();
    if let Some(previous_id) = &request.previous_response_id {
        let stored = endpoint
            .store
            .get(previous_id)
            .filter(|stored| visible_to(&owner, stored))
            .ok_or_else(|| evicted_or_unknown(previous_id))?;
        check_continuation(&stored, &model, &target)?;
        let mut items = stored.input_items;
        if let Some(output) = stored.response.get("output").and_then(Value::as_array) {
            items.extend(output.iter().cloned());
        }
        let Some(input) = &request.input else {
            return Err(OpenAiError::invalid_param("input", "`input` is required"));
        };
        items.extend(convert::input_items_of(input));
        replay_items = items;
        effective.input = Some(Input::Items(replay_items.clone()));
    } else if let Some(input) = &request.input {
        replay_items = convert::input_items_of(input);
    }
    let mut prepared = prepare(effective)?;
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    // A stored turn keeps its KV pinned in the prefix cache for the turn that
    // continues it, which releases it in turn: the chain's tail stays warm
    // and its next turn prefills only what is new.
    if prepared.store {
        prepared.request.retention = Some(Retention {
            key: id.clone(),
            release: prepared.previous_response_id.clone(),
        });
    }
    let created = unix_seconds();
    let (weights, base_model) = weights_identity(&target, &model);
    let store_ctx = StoreCtx {
        store: endpoint.store.clone(),
        owner,
        model: model.clone(),
        weights,
        fingerprint: String::new(),
        base_model,
        input_items: replay_items,
        previous_response_id: prepared.previous_response_id.clone(),
        store_response: prepared.store,
        created_at: created,
    };
    let template = Template {
        id,
        created,
        model,
        fingerprint: String::new(),
        prepared,
    };
    if template.prepared.stream {
        return Ok(responses_stream(endpoint, target, template, store_ctx).into_response());
    }
    let Prepared { request, .. } = template.prepared.clone();
    let (answers, fingerprint) = endpoint.sample(target, request).await?;
    let mut template = template;
    template.fingerprint = fingerprint.clone();
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
        completion.usage.prompt_tokens_details.cached_tokens as usize,
    );
    let output = output_items(&completion);
    let response = template.response(status, output, usage);
    let mut store_ctx = store_ctx;
    store_ctx.fingerprint = fingerprint;
    store_ctx.record(&response);
    Ok(axum::Json(response).into_response())
}

/// `GET /v1/responses/{id}`: the stored response object, as it was answered.
async fn responses_get(
    State(endpoint): State<Endpoint>,
    owner: Option<Extension<StoreOwner>>,
    Path(id): Path<String>,
) -> Result<AxumResponse, OpenAiError> {
    endpoint.check_enabled()?;
    if !valid_id(&id) {
        return Err(OpenAiError::response_not_found(&id, "not a response id"));
    }
    let owner = owner.map(|Extension(owner)| owner.0);
    let stored = endpoint
        .store
        .get(&id)
        .filter(|stored| visible_to(&owner, stored))
        .ok_or_else(|| {
            OpenAiError::response_not_found(
                &id,
                "it is unknown, or it was evicted from the in-memory store (it is bounded)",
            )
        })?;
    Ok(axum::Json(stored.response).into_response())
}

/// `DELETE /v1/responses/{id}`: removes the response and releases the pin it
/// held on its ancestor.
async fn responses_delete(
    State(endpoint): State<Endpoint>,
    owner: Option<Extension<StoreOwner>>,
    Path(id): Path<String>,
) -> Result<AxumResponse, OpenAiError> {
    endpoint.check_enabled()?;
    if !valid_id(&id) {
        return Err(OpenAiError::response_not_found(&id, "not a response id"));
    }
    let owner = owner.map(|Extension(owner)| owner.0);
    let visible = endpoint
        .store
        .get(&id)
        .is_some_and(|stored| visible_to(&owner, &stored));
    if !visible {
        return Err(OpenAiError::response_not_found(
            &id,
            "it is unknown, or it was evicted from the in-memory store (it is bounded)",
        ));
    }
    endpoint.store.delete(&id);
    Ok(axum::Json(serde_json::json!({
        "id": id,
        "object": "response",
        "deleted": true,
    }))
    .into_response())
}

/// The query parameters of `GET /v1/responses/{id}/input_items`: the list
/// envelope's pagination.
#[derive(Deserialize)]
struct InputItemsQuery {
    limit: Option<u64>,
    order: Option<String>,
    before: Option<String>,
    after: Option<String>,
}

const MAX_INPUT_ITEMS_PAGE: u64 = 100;
const DEFAULT_INPUT_ITEMS_PAGE: u64 = 20;

/// `GET /v1/responses/{id}/input_items`: the input items of one response -
/// the chain it continues included - in the list envelope, paginated.
async fn responses_input_items(
    State(endpoint): State<Endpoint>,
    owner: Option<Extension<StoreOwner>>,
    Path(id): Path<String>,
    query: Result<Query<InputItemsQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<AxumResponse, OpenAiError> {
    endpoint.check_enabled()?;
    if !valid_id(&id) {
        return Err(OpenAiError::response_not_found(&id, "not a response id"));
    }
    let owner = owner.map(|Extension(owner)| owner.0);
    let stored = endpoint
        .store
        .get(&id)
        .filter(|stored| visible_to(&owner, stored))
        .ok_or_else(|| {
            OpenAiError::response_not_found(
                &id,
                "it is unknown, or it was evicted from the in-memory store (it is bounded)",
            )
        })?;
    let Query(query) = query.map_err(|error| {
        OpenAiError::invalid(format!("invalid input-items query: {}", error.body_text()))
    })?;
    let mut items = stored.input_items;
    let limit = match query.limit {
        None => DEFAULT_INPUT_ITEMS_PAGE,
        Some(0) => {
            return Err(OpenAiError::invalid_param(
                "limit",
                "`limit` must be between 1 and 100",
            ));
        }
        Some(value) if value > MAX_INPUT_ITEMS_PAGE => {
            return Err(OpenAiError::invalid_param(
                "limit",
                format!("`limit` must be at most {MAX_INPUT_ITEMS_PAGE}"),
            ));
        }
        Some(value) => value,
    };
    let descending = match query.order.as_deref() {
        None | Some("asc") => false,
        Some("desc") => true,
        Some(other) => {
            return Err(OpenAiError::invalid_param(
                "order",
                format!("`order` must be `asc` or `desc`; got `{other}`"),
            ));
        }
    };
    if descending {
        items.reverse();
    }
    // `before`/`after` are item ids: the page ends before / starts after the
    // item that id names.
    let position_of = |cursor: &str| -> Option<usize> {
        items
            .iter()
            .position(|item| item.get("id").and_then(Value::as_str) == Some(cursor))
    };
    if let Some(before) = &query.before
        && position_of(before).is_none()
    {
        return Err(OpenAiError::invalid_param(
            "before",
            format!("no input item has id `{before}`"),
        ));
    }
    if let Some(after) = &query.after
        && position_of(after).is_none()
    {
        return Err(OpenAiError::invalid_param(
            "after",
            format!("no input item has id `{after}`"),
        ));
    }
    let mut start = 0_usize;
    let mut end = items.len();
    if let Some(after) = &query.after {
        start = position_of(after).expect("checked above") + 1;
    }
    if let Some(before) = &query.before {
        end = position_of(before).expect("checked above");
    }
    let window = &items[start.min(end)..end];
    // The limit is bounded by MAX_INPUT_ITEMS_PAGE: the narrowing is exact.
    let has_more = window.len() > limit as usize;
    let page: Vec<Value> = window.iter().take(limit as usize).cloned().collect();
    let first_id = page
        .first()
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str);
    let last_id = page
        .last()
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str);
    Ok(axum::Json(serde_json::json!({
        "object": "list",
        "data": page,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    }))
    .into_response())
}

/// The streamed sequence. A cold target streams live; a live run answers
/// buffered (its answer exists only once its thread is free), and the same
/// event sequence is cut from it whole.
fn responses_stream(
    endpoint: Endpoint,
    target: Target,
    template: Template,
    store_ctx: StoreCtx,
) -> AxumResponse {
    match target {
        Target::Cold { spec, fingerprint } => {
            let mut template = template;
            template.fingerprint = fingerprint.clone();
            let mut store_ctx = store_ctx;
            store_ctx.fingerprint = fingerprint;
            stream_events(endpoint, spec, template, store_ctx).into_response()
        }
        Target::Live { .. } => {
            stream_buffered(endpoint, target, template, store_ctx).into_response()
        }
    }
}

/// The buffered sequence: the whole answer exists, and the event sequence is
/// cut from it in the protocol's order.
fn stream_buffered(
    endpoint: Endpoint,
    target: Target,
    template: Template,
    mut store_ctx: StoreCtx,
) -> AxumResponse {
    let events = async_stream::stream! {
        match endpoint.sample(target, template.prepared.request.clone()).await {
            Ok((answers, fingerprint)) => {
                let mut template = template;
                template.fingerprint = fingerprint.clone();
                store_ctx.fingerprint = fingerprint;
                let answer = answers.into_iter().next().expect("a live run answers one choice");
                let completion = finish(answer, &[]);
                let status = status_of(&completion);
                let usage = template.usage(
                    completion.usage.prompt_tokens as usize,
                    completion.usage.completion_tokens as usize,
                    completion.reasoning_tokens,
                    completion.usage.prompt_tokens_details.cached_tokens as usize,
                );
                let output = output_items(&completion);
                let mut sequencer = Sequencer::new();
                // Store before yielding the terminal event: the client may
                // fetch the response immediately or disconnect at that event.
                store_ctx.record(&template.response(status, output.clone(), usage.clone()));
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
fn stream_events(
    endpoint: Endpoint,
    spec: WeightsSpec,
    template: Template,
    store_ctx: StoreCtx,
) -> AxumResponse {
    let events = async_stream::stream! {
        let mut builder = StreamBuilder::new(template, Some(store_ctx));
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
                            Event::End { reason, text, cached_tokens, .. } => {
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
                                    cached_tokens,
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
    /// Where the finished answer goes when the request asked to store it.
    store_ctx: Option<StoreCtx>,
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
    fn new(template: Template, store_ctx: Option<StoreCtx>) -> Self {
        Self {
            template,
            sequencer: Sequencer::new(),
            items: Vec::new(),
            logprobs: Vec::new(),
            store_ctx,
            done: false,
        }
    }

    /// `response.created` then `response.in_progress`, with an empty output.
    fn open(&mut self) -> Vec<Ev> {
        let usage = self.template.usage(0, 0, 0, 0);
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
        let response = self.template.response(status, output, usage);
        let terminal = match status {
            Status::Completed => Ev::Completed {
                sequence_number: 0,
                response: response.clone(),
            },
            _ => Ev::Incomplete {
                sequence_number: 0,
                response: response.clone(),
            },
        };
        if let Some(event) = self.sequencer.push(terminal) {
            events.push(event);
        }
        // The terminal event went out carrying the full response: that is
        // the moment a stored answer exists.
        if let Some(store_ctx) = &self.store_ctx {
            store_ctx.record(&response);
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
            template.usage(1, 2, 1, 0),
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
        let mut builder = StreamBuilder::new(template(), None);
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
