//! From the Responses `input` items to the common
//! [`GenerationRequest`](crate::request::GenerationRequest): `message` items
//! (system/developer/user/assistant, `input_text` and `output_text` parts),
//! re-rendered `reasoning` items, and `function_call` /
//! `function_call_output` items reconciled by their `call_id`.
//!
//! Anything but text - an image, a file - is refused with the field named,
//! never dropped, exactly as the chat conversion does. `store` defaults to
//! `true` (the API default) and writes the response to the endpoint's store;
//! `previous_response_id` replays a stored response's input items and output;
//! `truncation` is still refused - prefix reuse across responses is not served
//! yet.

use retrograd_agent_core::tools::ToolSpec;
use retrograd_agent_core::trajectory::{Message, Role};
use retrograd_core::SamplingParamsV2;
use serde_json::Value;

use crate::chat::convert as chat;
use crate::error::OpenAiError;
use crate::grammar::text_format_grammar;
use crate::request::{GenerationRequest, ToolConstraint};
use crate::responses::wire::{Input, ReasoningParam, ResponsesRequest};

/// The pinned schema's lower bound on `max_output_tokens`.
const MIN_MAX_OUTPUT_TOKENS: u64 = 16;
/// The pinned schema's bounds on `metadata`.
const MAX_METADATA_ENTRIES: usize = 16;
const MAX_METADATA_KEY: usize = 64;
const MAX_METADATA_VALUE: usize = 512;
/// The one `include` value this server serves.
const INCLUDE_LOGPROBS: &str = "message.output_text.logprobs";

/// A validated Responses request: which id to serve, the common
/// [`GenerationRequest`] the surface translated the wire body into, and the
/// echo fields the response object carries back.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub model: String,
    pub request: GenerationRequest,
    pub stream: bool,
    /// The response echoes, as sent (validated).
    pub instructions: Option<String>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel_tool_calls: bool,
    pub metadata: Option<Value>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_logprobs: Option<u32>,
    pub max_output_tokens: Option<u64>,
    pub text_format: Value,
    pub reasoning: Option<Value>,
    /// Whether the response is written to the store (`store`, default `true`).
    pub store: bool,
    /// The stored response this one continues, when it does.
    pub previous_response_id: Option<String>,
}

/// The documented local policy mapping a Responses `reasoning.effort` to a
/// reasoning budget in tokens. It is a local policy, not an equivalence with
/// any hosted model's effort: the budget bounds how long the model may reason
/// before the runtime forces the block closed.
pub fn effort_budget(effort: &str) -> Result<i32, OpenAiError> {
    match effort {
        "none" => Ok(0),
        "minimal" => Ok(256),
        "low" => Ok(1024),
        "medium" => Ok(4096),
        "high" => Ok(16384),
        "xhigh" => Ok(65536),
        "max" => Ok(-1),
        other => Err(OpenAiError::invalid_param(
            "reasoning.effort",
            format!(
                "unsupported effort `{other}`; expected none, minimal, low, medium, high, \
                 xhigh or max"
            ),
        )),
    }
}

pub fn prepare(request: ResponsesRequest) -> Result<Prepared, OpenAiError> {
    refuse_unsupported(&request)?;
    if request.model.trim().is_empty() {
        return Err(OpenAiError::invalid_param("model", "`model` is required"));
    }
    let Some(input) = request.input else {
        return Err(OpenAiError::invalid_param("input", "`input` is required"));
    };
    let mut messages = Vec::new();
    if let Some(instructions) = &request.instructions
        && !instructions.is_empty()
    {
        messages.push(Message {
            role: Role::System,
            content: instructions.clone(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            is_error: false,
            reasoning: None,
        });
    }
    match &input {
        Input::Text(text) => messages.push(user_message(text)),
        Input::Items(items) => {
            let mut calls: Vec<String> = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let at = |field: &str| format!("input[{index}].{field}");
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("role").is_some().then_some("message"))
                    .ok_or_else(|| {
                        OpenAiError::invalid_param(at("type"), "an input item needs a `type`")
                    })?;
                match kind {
                    "message" => messages.push(message_item(index, item)?),
                    "function_call" => {
                        let call = function_call_item(index, item)?;
                        calls.push(call.tool_calls[0].id.clone());
                        messages.push(call);
                    }
                    "function_call_output" => {
                        let message = function_call_output_item(index, item, &calls)?;
                        messages.push(message);
                    }
                    "reasoning" => messages.push(reasoning_item(index, item)?),
                    other => {
                        return Err(OpenAiError::unsupported(
                            at("type"),
                            format!(
                                "input items of type `{other}` are not supported by this server"
                            ),
                        ));
                    }
                }
            }
        }
    }
    let mut merged: Vec<Message> = Vec::with_capacity(messages.len());
    for message in messages {
        if message.role == Role::Assistant
            && let Some(previous) = merged.last_mut().filter(|m| m.role == Role::Assistant)
        {
            previous.content.push_str(&message.content);
            previous.tool_calls.extend(message.tool_calls);
            if let Some(reasoning) = message.reasoning {
                previous
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&reasoning);
            }
        } else {
            merged.push(message);
        }
    }
    let messages = merged;
    if messages.is_empty() {
        return Err(OpenAiError::invalid_param(
            "input",
            "`input` must contain at least one message",
        ));
    }
    let tools_echo = request.tools.clone().unwrap_or_default();
    let tools = request
        .tools
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(index, tool)| convert_tool(index, &tool))
        .collect::<Result<Vec<_>, _>>()?;
    // The Responses forced form is flat: {"type":"function","name":…}. It is
    // normalized to the chat shape the shared constraint logic reads.
    let choice = request.tool_choice.as_ref().map(|choice| match choice {
        Value::Object(object)
            if object.get("type").and_then(Value::as_str) == Some("function")
                && object.get("function").is_none() =>
        {
            let mut normalized = object.clone();
            let name = normalized.remove("name");
            normalized.insert(
                "function".to_owned(),
                serde_json::json!({"name": name.unwrap_or(Value::Null)}),
            );
            Value::Object(normalized)
        }
        other => other.clone(),
    });
    let constraint = chat::tool_constraint(choice.as_ref(), &tools)?;
    let tools = match chat::offers_tools(choice.as_ref()) {
        true => tools,
        false => Vec::new(),
    };
    let constraint = match (request.parallel_tool_calls, constraint) {
        (Some(false), Some(_)) => {
            return Err(OpenAiError::unsupported(
                "parallel_tool_calls",
                "cannot combine `parallel_tool_calls: false` with a `tool_choice` constraint; \
                 send one of them",
            ));
        }
        (Some(false), None) if !tools.is_empty() => Some(ToolConstraint::AtMostOne),
        (Some(false), None) => None,
        (_, existing) => existing,
    };
    let grammar = match &request.text {
        Some(text) => match &text.format {
            Some(format) => text_format_grammar(format)?,
            None => None,
        },
        None => None,
    };
    if grammar.is_some() && (constraint.is_some() || !tools.is_empty()) {
        return Err(OpenAiError::unsupported(
            "text.format",
            "a json text format cannot be combined with tools or a forced tool call; send one \
             of them",
        ));
    }
    // Logprobs are served when `include` asks for them; `top_logprobs` sets
    // how many top pairs each entry carries.
    let include = request.include.unwrap_or_default();
    for value in &include {
        if value != INCLUDE_LOGPROBS {
            return Err(OpenAiError::unsupported(
                "include",
                format!(
                    "unsupported `include` value `{value}`; only `{INCLUDE_LOGPROBS}` is served"
                ),
            ));
        }
    }
    let wants_logprobs = include.iter().any(|value| value == INCLUDE_LOGPROBS);
    let top = request.top_logprobs.unwrap_or(0);
    if top > chat::MAX_TOP_LOGPROBS {
        return Err(OpenAiError::invalid_param(
            "top_logprobs",
            format!("`top_logprobs` must be at most {}", chat::MAX_TOP_LOGPROBS),
        ));
    }
    if top > 0 && !wants_logprobs {
        return Err(OpenAiError::invalid_param(
            "top_logprobs",
            format!("`top_logprobs` requires `include: [\"{INCLUDE_LOGPROBS}\"]`"),
        ));
    }
    // Narrowing checked against MAX_TOP_LOGPROBS above: exact.
    let logprobs = wants_logprobs.then_some(top as u32);
    let reasoning_budget = match &request.reasoning {
        Some(ReasoningParam {
            effort: Some(effort),
            ..
        }) => Some(effort_budget(effort)?),
        _ => None,
    };
    if reasoning_budget.is_some() && grammar.is_some() {
        return Err(OpenAiError::unsupported(
            "reasoning.effort",
            "a reasoning budget cannot be combined with a JSON text format",
        ));
    }
    let max_output_tokens = match request.max_output_tokens {
        Some(value) if value < MIN_MAX_OUTPUT_TOKENS => {
            return Err(OpenAiError::invalid_param(
                "max_output_tokens",
                format!("`max_output_tokens` must be at least {MIN_MAX_OUTPUT_TOKENS}"),
            ));
        }
        Some(value) => Some(value),
        None => None,
    };
    let max_new_tokens = match max_output_tokens {
        Some(value) => u32::try_from(value).map_err(|_| {
            OpenAiError::invalid_param(
                "max_output_tokens",
                format!("`max_output_tokens` must be at most {}", u32::MAX),
            )
        })?,
        None => chat::DEFAULT_MAX_TOKENS,
    };
    let temperature = chat::temperature(request.temperature)?;
    let top_p = chat::top_p(request.top_p)?;
    Ok(Prepared {
        stream: request.stream.unwrap_or(false),
        instructions: request.instructions,
        tools: tools_echo,
        tool_choice: request
            .tool_choice
            .unwrap_or_else(|| Value::String("auto".into())),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        metadata: request.metadata.map(Value::Object),
        temperature: request.temperature,
        top_p: request.top_p,
        top_logprobs: logprobs,
        max_output_tokens,
        text_format: request
            .text
            .and_then(|text| text.format)
            .unwrap_or_else(|| serde_json::json!({"type": "text"})),
        reasoning: request
            .reasoning
            .map(|reasoning| serde_json::to_value(reasoning).unwrap_or(Value::Null)),
        model: request.model.clone(),
        store: request.store.unwrap_or(true),
        previous_response_id: request.previous_response_id.clone(),
        request: GenerationRequest {
            raw_prompt: None,
            messages,
            tools,
            sampling: SamplingParamsV2 {
                temperature,
                top_p,
                max_new_tokens,
                seed: chat::DEFAULT_SEED,
                top_k: 0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                penalty_window: 0,
                logit_bias: Vec::new(),
                grammar_gbnf: grammar,
            },
            stop: Vec::new(),
            n: 1,
            logprobs,
            tool_constraint: constraint,
            reasoning_budget,
        },
    })
}

/// The `input` of a request as a list of input items: a string is one user
/// message item. Used to prepend a stored chain's items on a continuation.
pub fn input_items_of(input: &Input) -> Vec<Value> {
    match input {
        Input::Text(text) => vec![serde_json::json!({
            "type": "message",
            "role": "user",
            "content": text,
        })],
        Input::Items(items) => items.clone(),
    }
}

/// What this server refuses outright, with the field named: the background
/// path, hosted-tool plumbing, prefix truncation (not served yet), and the
/// metadata bounds of the pinned schema.
fn refuse_unsupported(request: &ResponsesRequest) -> Result<(), OpenAiError> {
    if let Some(truncation) = &request.truncation {
        return Err(OpenAiError::unsupported(
            "truncation",
            format!(
                "`truncation: \"{truncation}\"` is not served: prefix reuse across responses is not available yet"
            ),
        ));
    }
    if request.background == Some(true) {
        return Err(OpenAiError::unsupported(
            "background",
            "background (asynchronous) responses are not served",
        ));
    }
    if let Some(conversation) = &request.conversation
        && !conversation.is_null()
    {
        return Err(OpenAiError::unsupported(
            "conversation",
            "hosted conversations are not served: continue a stored response with \
             `previous_response_id` instead",
        ));
    }
    if request.max_tool_calls.is_some() {
        return Err(OpenAiError::unsupported(
            "max_tool_calls",
            "`max_tool_calls` is not served: calls are limited by `tool_choice` and \
             `parallel_tool_calls`, never cut after the fact",
        ));
    }
    if let Some(options) = &request.stream_options {
        let obfuscation = options.get("include_obfuscation").and_then(Value::as_bool);
        if obfuscation == Some(true) {
            return Err(OpenAiError::unsupported(
                "stream_options.include_obfuscation",
                "this server does not obfuscate streamed deltas",
            ));
        }
    }
    if let Some(metadata) = &request.metadata {
        if metadata.len() > MAX_METADATA_ENTRIES {
            return Err(OpenAiError::invalid_param(
                "metadata",
                format!("`metadata` takes at most {MAX_METADATA_ENTRIES} key/value pairs"),
            ));
        }
        for (key, value) in metadata {
            if key.len() > MAX_METADATA_KEY {
                return Err(OpenAiError::invalid_param(
                    "metadata",
                    format!("a metadata key is at most {MAX_METADATA_KEY} characters"),
                ));
            }
            let Some(text) = value.as_str() else {
                return Err(OpenAiError::invalid_param(
                    format!("metadata.{key}"),
                    "a metadata value must be a string",
                ));
            };
            if text.len() > MAX_METADATA_VALUE {
                return Err(OpenAiError::invalid_param(
                    format!("metadata.{key}"),
                    format!("a metadata value is at most {MAX_METADATA_VALUE} characters"),
                ));
            }
        }
    }
    if let Some(text) = &request.text
        && let Some(verbosity) = &text.verbosity
    {
        return Err(OpenAiError::unsupported(
            "text.verbosity",
            format!("`text.verbosity: \"{verbosity}\"` is not served by a local model"),
        ));
    }
    if let Some(reasoning) = &request.reasoning {
        for field in ["summary", "mode", "context"] {
            let value = match field {
                "summary" => &reasoning.summary,
                "mode" => &reasoning.mode,
                _ => &reasoning.context,
            };
            if let Some(value) = value {
                return Err(OpenAiError::unsupported(
                    format!("reasoning.{field}"),
                    format!(
                        "`reasoning.{field}: \"{value}\"` is not served; only `reasoning.effort` is read"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn user_message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: text.to_owned(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        is_error: false,
        reasoning: None,
    }
}

/// A `message` input item: roles system/developer/user/assistant, content a
/// string or a list of parts. `input_text` and `output_text` carry text;
/// `input_image`, `input_file` and anything else are refused, not dropped.
fn message_item(index: usize, item: &Value) -> Result<Message, OpenAiError> {
    let at = |field: &str| format!("input[{index}].{field}");
    let role = match item.get("role").and_then(Value::as_str) {
        Some("system") | Some("developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some(other) => {
            return Err(OpenAiError::invalid_param(
                at("role"),
                format!(
                    "unsupported role `{other}`; expected system, developer, user or assistant"
                ),
            ));
        }
        None => {
            return Err(OpenAiError::invalid_param(
                at("role"),
                "a message item needs a `role`",
            ));
        }
    };
    let content = match item.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for (part, value) in parts.iter().enumerate() {
                let kind = value.get("type").and_then(Value::as_str).ok_or_else(|| {
                    OpenAiError::invalid_param(
                        format!("input[{index}].content[{part}].type"),
                        "a content part needs a `type`",
                    )
                })?;
                match kind {
                    "input_text" | "output_text" => {
                        let Some(part_text) = value.get("text").and_then(Value::as_str) else {
                            return Err(OpenAiError::invalid_param(
                                format!("input[{index}].content[{part}].text"),
                                "a text part needs a `text` field",
                            ));
                        };
                        text.push_str(part_text);
                    }
                    other => {
                        return Err(OpenAiError::unsupported_content(
                            format!("input[{index}].content[{part}]"),
                            format!(
                                "content parts of type `{other}` are not supported by a text model"
                            ),
                        ));
                    }
                }
            }
            text
        }
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                at("content"),
                "`content` must be a string or a list of parts",
            ));
        }
    };
    Ok(Message {
        role,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        is_error: false,
        reasoning: None,
    })
}

/// A `function_call` input item becomes the assistant turn that carried the
/// call, so the template re-renders it as sent.
fn function_call_item(index: usize, item: &Value) -> Result<Message, OpenAiError> {
    let at = |field: &str| format!("input[{index}].{field}");
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            OpenAiError::invalid_param(at("call_id"), "a function_call needs a `call_id`")
        })?;
    if call_id.len() > 64 {
        return Err(OpenAiError::invalid_param(
            at("call_id"),
            "a `call_id` is at most 64 characters",
        ));
    }
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| OpenAiError::invalid_param(at("name"), "a function_call needs a `name`"))?;
    let arguments = chat::arguments(
        &at("arguments"),
        item.get("arguments").and_then(|value| match value {
            Value::String(_) | Value::Object(_) => Some(value),
            _ => None,
        }),
    )?;
    Ok(Message {
        role: Role::Assistant,
        content: String::new(),
        tool_calls: vec![retrograd_agent_core::tools::ToolCall {
            id: call_id.to_owned(),
            name: name.to_owned(),
            arguments,
        }],
        tool_call_id: None,
        is_error: false,
        reasoning: None,
    })
}

/// A `function_call_output` item becomes the tool turn answering the call its
/// `call_id` names - and that call must have been sent earlier in the input.
fn function_call_output_item(
    index: usize,
    item: &Value,
    calls: &[String],
) -> Result<Message, OpenAiError> {
    let at = |field: &str| format!("input[{index}].{field}");
    let call_id = item.get("call_id").and_then(Value::as_str);
    let Some(call_id) = call_id.filter(|id| !id.is_empty()) else {
        return Err(OpenAiError::invalid_param(
            at("call_id"),
            "a function_call_output must name the call it answers in `call_id`",
        ));
    };
    if call_id.len() > 64 {
        return Err(OpenAiError::invalid_param(
            at("call_id"),
            "a `call_id` is at most 64 characters",
        ));
    }
    if !calls.iter().any(|known| known == call_id) {
        return Err(OpenAiError::invalid_param(
            at("call_id"),
            format!("no `function_call` with `call_id` `{call_id}` appears earlier in `input`"),
        ));
    }
    let content = match item.get("output") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for (part, value) in parts.iter().enumerate() {
                match value.get("type").and_then(Value::as_str) {
                    Some("input_text") => {
                        let Some(part_text) = value.get("text").and_then(Value::as_str) else {
                            return Err(OpenAiError::invalid_param(
                                format!("input[{index}].output[{part}].text"),
                                "a text part needs a `text` field",
                            ));
                        };
                        text.push_str(part_text);
                    }
                    Some(other) => {
                        return Err(OpenAiError::unsupported_content(
                            format!("input[{index}].output[{part}]"),
                            format!(
                                "output parts of type `{other}` are not supported by a text model"
                            ),
                        ));
                    }
                    None => {
                        return Err(OpenAiError::invalid_param(
                            format!("input[{index}].output[{part}].type"),
                            "an output part needs a `type`",
                        ));
                    }
                }
            }
            text
        }
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                at("output"),
                "`output` must be a string or a list of parts",
            ));
        }
    };
    Ok(Message {
        role: Role::Tool,
        content,
        tool_calls: Vec::new(),
        tool_call_id: Some(call_id.to_owned()),
        is_error: false,
        reasoning: None,
    })
}

/// A `reasoning` input item becomes the reasoning of an assistant turn, so
/// the template re-renders it. The raw `content` (`reasoning_text`) and the
/// `summary` are joined, in that order, as the turn's reasoning; an
/// `encrypted_content` is refused (it is out of perimeter until stored
/// responses land).
fn reasoning_item(index: usize, item: &Value) -> Result<Message, OpenAiError> {
    let at = |field: &str| format!("input[{index}].{field}");
    if item
        .get("encrypted_content")
        .is_some_and(|value| !value.is_null())
    {
        return Err(OpenAiError::unsupported(
            at("encrypted_content"),
            "encrypted reasoning content is not served",
        ));
    }
    let mut reasoning = String::new();
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for (part, value) in summary.iter().enumerate() {
            match value.get("type").and_then(Value::as_str) {
                Some("summary_text") => {
                    if let Some(text) = value.get("text").and_then(Value::as_str) {
                        reasoning.push_str(text);
                    }
                }
                Some(other) => {
                    return Err(OpenAiError::unsupported(
                        format!("input[{index}].summary[{part}].type"),
                        format!("summary parts of type `{other}` are not supported"),
                    ));
                }
                None => {
                    return Err(OpenAiError::invalid_param(
                        format!("input[{index}].summary[{part}].type"),
                        "a summary part needs a `type`",
                    ));
                }
            }
        }
    }
    if let Some(content) = item.get("content").and_then(Value::as_array) {
        for (part, value) in content.iter().enumerate() {
            match value.get("type").and_then(Value::as_str) {
                Some("reasoning_text") => {
                    if let Some(text) = value.get("text").and_then(Value::as_str) {
                        reasoning.push_str(text);
                    }
                }
                Some(other) => {
                    return Err(OpenAiError::unsupported(
                        format!("input[{index}].content[{part}].type"),
                        format!("reasoning content parts of type `{other}` are not supported"),
                    ));
                }
                None => {
                    return Err(OpenAiError::invalid_param(
                        format!("input[{index}].content[{part}].type"),
                        "a reasoning content part needs a `type`",
                    ));
                }
            }
        }
    }
    Ok(Message {
        role: Role::Assistant,
        content: String::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        is_error: false,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    })
}

/// The flat tool form `{type: "function", name, description?, parameters?,
/// strict?}`. Hosted tools are refused by their type.
fn convert_tool(index: usize, tool: &Value) -> Result<ToolSpec, OpenAiError> {
    let at = |field: &str| format!("tools[{index}].{field}");
    let kind = tool.get("type").and_then(Value::as_str).ok_or_else(|| {
        OpenAiError::invalid_param(format!("tools[{index}].type"), "a tool needs a `type`")
    })?;
    if kind != "function" {
        return Err(OpenAiError::unsupported(
            format!("tools[{index}].type"),
            format!("tools of type `{kind}` are not served; only `function` tools are"),
        ));
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| OpenAiError::invalid_param(at("name"), "a tool needs a `name`"))?;
    Ok(ToolSpec {
        name: name.to_owned(),
        description: tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        input_schema: tool
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).expect("a well-formed request")
    }

    fn base() -> Value {
        json!({"model": "m", "input": "hi", "store": false})
    }

    fn prep(body: Value) -> Result<Prepared, OpenAiError> {
        prepare(request(body))
    }

    #[test]
    fn a_string_input_is_one_user_turn_and_instructions_a_system_turn() {
        let mut body = base();
        body["instructions"] = json!("be brief");
        let prepared = prep(body).unwrap();
        assert_eq!(prepared.request.messages.len(), 2);
        assert_eq!(prepared.request.messages[0].role, Role::System);
        assert_eq!(prepared.request.messages[0].content, "be brief");
        assert_eq!(prepared.request.messages[1].role, Role::User);
        assert_eq!(prepared.request.messages[1].content, "hi");
        assert_eq!(prepared.request.n, 1);
        assert!(!prepared.stream);
    }

    #[test]
    fn item_inputs_are_reconciled_by_call_id() {
        let body = json!({"model": "m", "store": false, "input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c1", "name": "move",
             "arguments": "{\"direction\":\"left\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "moved"},
            {"type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": "thought"}],
             "content": [{"type": "reasoning_text", "text": " more"}]}
        ]});
        let prepared = prep(body).unwrap();
        let messages = &prepared.request.messages;
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].tool_calls[0].id, "c1");
        assert_eq!(
            messages[1].tool_calls[0].arguments,
            json!({"direction": "left"})
        );
        assert_eq!(messages[2].role, Role::Tool);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(messages[2].content, "moved");
        assert_eq!(messages[3].reasoning.as_deref(), Some("thought more"));

        // An output that answers no earlier call is refused, field-localized.
        let body = json!({"model": "m", "store": false, "input": [
            {"type": "function_call_output", "call_id": "nope", "output": "x"}
        ]});
        let error = prep(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input[0].call_id"));
    }

    #[test]
    fn non_text_parts_and_items_are_refused_with_the_type_named() {
        let body = json!({"model": "m", "store": false, "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "https://x/y.png"}]}
        ]});
        let error = prep(body).unwrap_err();
        assert_eq!(error.code, Some("unsupported_content"));
        assert_eq!(error.param.as_deref(), Some("input[0].content[1]"));

        let body = json!({"model": "m", "store": false, "input": [
            {"type": "item_reference", "id": "resp_1"}
        ]});
        let error = prep(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input[0].type"));
    }

    #[test]
    fn store_defaults_to_true_and_hosted_fields_are_refused() {
        // `store: true` (and absent) is the API default and is served by the
        // endpoint's store; `previous_response_id` is looked up there.
        for body in [
            {
                let mut body = base();
                body.as_object_mut().unwrap().remove("store");
                body
            },
            {
                let mut body = base();
                body["store"] = json!(true);
                body
            },
        ] {
            let prepared = prep(body).unwrap();
            assert!(prepared.store);
            assert!(prepared.previous_response_id.is_none());
        }
        let mut body = base();
        body["store"] = json!(false);
        assert!(!prep(body).unwrap().store);

        let mut body = base();
        body["previous_response_id"] = json!("resp_1");
        let prepared = prep(body).unwrap();
        assert_eq!(prepared.previous_response_id.as_deref(), Some("resp_1"));

        for (field, value) in [
            ("truncation", json!("auto")),
            ("background", json!(true)),
            ("max_tool_calls", json!(4)),
        ] {
            let mut body = base();
            body[field] = value;
            let error = prep(body).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field), "{field}");
        }
    }

    #[test]
    fn text_format_json_schema_becomes_a_grammar_and_conflicts_with_tools() {
        let mut body = base();
        body["text"] = json!({"format": {"type": "json_object"}});
        let prepared = prep(body.clone()).unwrap();
        assert!(prepared.request.sampling.grammar_gbnf.is_some());
        assert_eq!(prepared.text_format, json!({"type": "json_object"}));

        body["tools"] = json!([{"type": "function", "name": "move"}]);
        let error = prep(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("text.format"));
    }

    #[test]
    fn reasoning_effort_maps_to_the_local_budget_policy() {
        let mut body = base();
        body["reasoning"] = json!({"effort": "low"});
        let prepared = prep(body).unwrap();
        assert_eq!(prepared.request.reasoning_budget, Some(1024));

        let mut body = base();
        body["reasoning"] = json!({"effort": "max"});
        assert_eq!(prep(body).unwrap().request.reasoning_budget, Some(-1));

        let mut body = base();
        body["reasoning"] = json!({"effort": "turbo"});
        assert_eq!(
            prep(body).unwrap_err().param.as_deref(),
            Some("reasoning.effort")
        );
    }

    #[test]
    fn max_output_tokens_bounds_and_logprobs_include() {
        let mut body = base();
        body["max_output_tokens"] = json!(8);
        let error = prep(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("max_output_tokens"));

        let mut body = base();
        body["max_output_tokens"] = json!(64);
        body["top_logprobs"] = json!(5);
        let error = prep(body.clone()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("top_logprobs"));

        body["include"] = json!([INCLUDE_LOGPROBS]);
        let prepared = prep(body).unwrap();
        assert_eq!(prepared.request.logprobs, Some(5));
        assert_eq!(prepared.request.sampling.max_new_tokens, 64);

        let mut body = base();
        body["include"] = json!(["reasoning.encrypted_content"]);
        assert_eq!(prep(body).unwrap_err().param.as_deref(), Some("include"));
    }

    #[test]
    fn tool_choice_forced_uses_the_flat_responses_form() {
        let mut body = base();
        body["tools"] = json!([{"type": "function", "name": "move"}]);
        body["tool_choice"] = json!({"type": "function", "name": "move"});
        let prepared = prep(body).unwrap();
        assert_eq!(
            prepared.request.tool_constraint,
            Some(ToolConstraint::Forced("move".to_owned()))
        );
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn easy_messages_and_split_assistant_items_form_one_turn() {
        let prepared = prepare(
            serde_json::from_value(json!({
                "model":"m", "store":false, "input":[
                    {"role":"user", "content":"go"},
                    {"type":"reasoning", "content":[{"type":"reasoning_text","text":"plan"}]},
                    {"type":"message","role":"assistant","content":"done"},
                    {"type":"function_call","call_id":"c","name":"move","arguments":"{}"},
                    {"type":"function_call_output","call_id":"c","output":"ok"}
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(prepared.request.messages.len(), 3);
        let assistant = &prepared.request.messages[1];
        assert_eq!(assistant.reasoning.as_deref(), Some("plan"));
        assert_eq!(assistant.content, "done");
        assert_eq!(assistant.tool_calls.len(), 1);
    }
}
