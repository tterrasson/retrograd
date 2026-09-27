//! From the wire request to what the model is asked.
//!
//! The rule this module enforces, field by field: a parameter that would change
//! the sampled distribution is never ignored. It is honoured, accepted in its
//! neutral form (`frequency_penalty: 0`, `logprobs: false`), or refused with the
//! field named. Only the fields that cannot change a token - `user`,
//! `metadata`, `store` - are dropped, because the clients that send them send
//! them on every request.

use retrograd_agent::tools::{ToolCall, ToolSpec};
use retrograd_agent::trajectory::{Message, Role};
use retrograd_core::SamplingParams;
use serde_json::Value;

use crate::error::OpenAiError;
use crate::wire::{ChatCompletionRequest, Content, Stop, WireMessage, WireTool};

pub const DEFAULT_MAX_TOKENS: u32 = 512;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_SEED: u32 = 1;
/// The sampler refuses a zero temperature, and OpenAI's `0` means greedy. Any
/// temperature this small puts all the mass on the largest logit, which is
/// greedy decoding without a second code path.
pub const GREEDY_TEMPERATURE: f32 = 1e-4;
/// Same treatment for `top_p = 0`, which the sampler also refuses: a nucleus
/// this small keeps the most likely token and nothing else.
pub const GREEDY_TOP_P: f32 = 1e-6;
/// OpenAI's own bound on `stop`.
const MAX_STOP: usize = 4;

/// What a chat request asks the model, independent of which weights answer.
#[derive(Clone, Debug)]
pub struct ChatPrompt {
    /// In the order the client sent them. A `tool` message carries the tool's
    /// own output; how it is framed depends on the rendering, decided later.
    pub messages: Vec<Message>,
    /// Empty for `tool_choice: "none"`: with no constrained decoding, a
    /// catalog in the prompt is an invitation to call, so none is rendered.
    pub tools: Vec<ToolSpec>,
    pub sampling: SamplingParams,
    /// Applied to the text after generation, the first match truncating it.
    pub stop: Vec<String>,
}

/// A validated request.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub model: String,
    pub prompt: ChatPrompt,
    pub stream: bool,
    pub include_usage: bool,
}

pub fn prepare(request: ChatCompletionRequest) -> Result<Prepared, OpenAiError> {
    refuse_unsupported(&request)?;
    if request.model.trim().is_empty() {
        return Err(OpenAiError::invalid_param("model", "`model` is required"));
    }
    if request.messages.is_empty() {
        return Err(OpenAiError::invalid_param(
            "messages",
            "`messages` must contain at least one message",
        ));
    }
    let messages = request
        .messages
        .iter()
        .enumerate()
        .map(|(index, message)| convert_message(index, message))
        .collect::<Result<Vec<_>, _>>()?;
    let tools = request
        .tools
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(index, tool)| convert_tool(index, tool))
        .collect::<Result<Vec<_>, _>>()?;
    let tools = match offers_tools(request.tool_choice.as_ref())? {
        true => tools,
        false => Vec::new(),
    };
    // `true` is what an unconstrained sampler does anyway; `false` would need
    // the decoding to stop after one call, and cutting the extra calls off
    // afterwards would answer with a turn the model did not write.
    if request.parallel_tool_calls == Some(false) && !tools.is_empty() {
        return Err(OpenAiError::unsupported(
            "parallel_tool_calls",
            "`parallel_tool_calls: false` needs constrained decoding; send true or omit it",
        ));
    }
    let stream = request.stream.unwrap_or(false);
    Ok(Prepared {
        model: request.model,
        prompt: ChatPrompt {
            messages,
            tools,
            sampling: SamplingParams {
                temperature: temperature(request.temperature)?,
                top_p: top_p(request.top_p)?,
                max_new_tokens: max_tokens(request.max_tokens, request.max_completion_tokens)?,
                seed: seed(request.seed)?,
            },
            stop: stop(request.stop)?,
        },
        stream,
        include_usage: stream
            && request
                .stream_options
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
    })
}

/// Whether a field holds the one value that changes nothing.
type IsNeutral = fn(&Value) -> bool;

/// The fields this server reads only to refuse them, unless they hold the value
/// that changes nothing.
fn refuse_unsupported(request: &ChatCompletionRequest) -> Result<(), OpenAiError> {
    if let Some(n) = request.n
        && n != 1
    {
        return Err(OpenAiError::unsupported(
            "n",
            "only one choice per request is supported (`n = 1`)",
        ));
    }
    for (param, value) in [
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
    ] {
        if value.is_some_and(|value| value != 0.0) {
            return Err(OpenAiError::unsupported(
                param,
                format!("`{param}` is not supported by this server's sampler; send 0 or omit it"),
            ));
        }
    }
    let neutral: [(&str, Option<&Value>, IsNeutral); 6] = [
        ("logprobs", request.logprobs.as_ref(), |value| {
            value == &Value::Bool(false)
        }),
        ("top_logprobs", request.top_logprobs.as_ref(), |value| {
            value.as_u64() == Some(0)
        }),
        ("logit_bias", request.logit_bias.as_ref(), |value| {
            value.as_object().is_some_and(serde_json::Map::is_empty)
        }),
        (
            "response_format",
            request.response_format.as_ref(),
            |value| value.get("type").and_then(Value::as_str) == Some("text"),
        ),
        ("audio", request.audio.as_ref(), |_| false),
        ("modalities", request.modalities.as_ref(), |value| {
            value
                .as_array()
                .is_some_and(|items| items.iter().all(|item| item == "text"))
        }),
    ];
    for (param, value, is_neutral) in neutral {
        if let Some(value) = value
            && !is_neutral(value)
        {
            return Err(OpenAiError::unsupported(
                param,
                format!("`{param}` is not supported by this server"),
            ));
        }
    }
    Ok(())
}

fn convert_message(index: usize, message: &WireMessage) -> Result<Message, OpenAiError> {
    let at = |field: &str| format!("messages[{index}].{field}");
    let role = match message.role.as_str() {
        // `developer` is the newer spelling of the same turn; chat templates
        // only know `system`.
        "system" | "developer" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => {
            return Err(OpenAiError::invalid_param(
                at("role"),
                format!(
                    "unsupported role `{other}`; expected system, developer, user, assistant or \
                     tool"
                ),
            ));
        }
    };
    let tool_calls = match (&message.tool_calls, role) {
        (Some(calls), Role::Assistant) => calls
            .iter()
            .enumerate()
            .map(|(call, wire)| {
                let param = format!("messages[{index}].tool_calls[{call}]");
                if let Some(kind) = wire.kind.as_deref()
                    && kind != "function"
                {
                    return Err(OpenAiError::invalid_param(
                        format!("{param}.type"),
                        format!("unsupported tool call type `{kind}`; expected `function`"),
                    ));
                }
                Ok(ToolCall {
                    id: wire
                        .id
                        .clone()
                        .filter(|id| !id.is_empty())
                        .unwrap_or_else(|| format!("call_{call}")),
                    name: wire.function.name.clone(),
                    arguments: arguments(
                        &format!("{param}.function.arguments"),
                        wire.function.arguments.as_ref(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        (Some(calls), _) if !calls.is_empty() => {
            return Err(OpenAiError::invalid_param(
                at("tool_calls"),
                "only an assistant message carries tool calls",
            ));
        }
        _ => Vec::new(),
    };
    let content = match &message.content {
        Some(content) => text(index, content)?,
        // A turn that is nothing but calls has no content, by the contract.
        None if role == Role::Assistant => String::new(),
        None => {
            return Err(OpenAiError::invalid_param(
                at("content"),
                "`content` is required on this message",
            ));
        }
    };
    let tool_call_id = match role {
        Role::Tool => Some(
            message
                .tool_call_id
                .clone()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    OpenAiError::invalid_param(
                        at("tool_call_id"),
                        "a tool message must name the call it answers in `tool_call_id`",
                    )
                })?,
        ),
        _ => None,
    };
    Ok(Message {
        role,
        content,
        tool_calls,
        tool_call_id,
        is_error: false,
    })
}

/// The text of a message: the string itself, or its text parts concatenated
/// as sent, with nothing between them.
/// Anything else - an image, audio, a file - is refused rather than dropped,
/// which would answer a question the client did not ask.
fn text(index: usize, content: &Content) -> Result<String, OpenAiError> {
    match content {
        Content::Text(text) => Ok(text.clone()),
        Content::Parts(parts) => parts
            .iter()
            .enumerate()
            .map(|(part, value)| match (value.kind.as_str(), &value.text) {
                ("text", Some(text)) => Ok(text.as_str()),
                ("text", None) => Err(OpenAiError::invalid_param(
                    format!("messages[{index}].content[{part}].text"),
                    "a text part needs a `text` field",
                )),
                (kind, _) => Err(OpenAiError::unsupported_content(
                    format!("messages[{index}].content[{part}]"),
                    format!("content parts of type `{kind}` are not supported by a text model"),
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.concat()),
    }
}

/// A call's arguments, which the contract sends as a JSON document inside a
/// string. The templates iterate over them as an object, so an object is what
/// they must be.
fn arguments(param: &str, value: Option<&Value>) -> Result<Value, OpenAiError> {
    let parsed = match value {
        None | Some(Value::Null) => return Ok(Value::Object(Default::default())),
        Some(Value::String(text)) if text.trim().is_empty() => {
            return Ok(Value::Object(Default::default()));
        }
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).map_err(|error| {
            OpenAiError::invalid_param(param, format!("the arguments are not valid JSON: {error}"))
        })?,
        Some(other) => other.clone(),
    };
    if !parsed.is_object() {
        return Err(OpenAiError::invalid_param(
            param,
            "the arguments must be a JSON object",
        ));
    }
    Ok(parsed)
}

fn convert_tool(index: usize, tool: WireTool) -> Result<ToolSpec, OpenAiError> {
    if tool.kind != "function" {
        return Err(OpenAiError::unsupported(
            format!("tools[{index}].type"),
            format!("unsupported tool type `{}`; expected `function`", tool.kind),
        ));
    }
    if tool.function.name.trim().is_empty() {
        return Err(OpenAiError::invalid_param(
            format!("tools[{index}].function.name"),
            "a tool needs a name",
        ));
    }
    Ok(ToolSpec {
        name: tool.function.name,
        description: tool.function.description.unwrap_or_default(),
        input_schema: tool
            .function
            .parameters
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
    })
}

/// Whether the catalog is offered to the model. There is no constrained
/// decoding, so a choice that would *force* a call cannot be honoured and is
/// refused.
fn offers_tools(choice: Option<&Value>) -> Result<bool, OpenAiError> {
    match choice {
        None => Ok(true),
        Some(Value::String(choice)) if choice == "auto" => Ok(true),
        Some(Value::String(choice)) if choice == "none" => Ok(false),
        Some(_) => Err(OpenAiError::unsupported(
            "tool_choice",
            "only `auto` and `none` are supported: forcing a call needs constrained decoding",
        )),
    }
}

/// `max_completion_tokens` is the newer name and wins when both are sent.
fn max_tokens(legacy: Option<u64>, current: Option<u64>) -> Result<u32, OpenAiError> {
    let (param, value) = match (current, legacy) {
        (Some(value), _) => ("max_completion_tokens", value),
        (None, Some(value)) => ("max_tokens", value),
        (None, None) => return Ok(DEFAULT_MAX_TOKENS),
    };
    if value == 0 {
        return Err(OpenAiError::invalid_param(
            param,
            format!("`{param}` must be greater than zero"),
        ));
    }
    // A contract, not a budget: a value that does not fit is refused rather
    // than silently lowered.
    u32::try_from(value).map_err(|_| {
        OpenAiError::invalid_param(param, format!("`{param}` must be at most {}", u32::MAX))
    })
}

fn temperature(value: Option<f64>) -> Result<f32, OpenAiError> {
    let Some(value) = value else {
        return Ok(DEFAULT_TEMPERATURE);
    };
    if !value.is_finite() || !(0.0..=2.0).contains(&value) {
        return Err(OpenAiError::invalid_param(
            "temperature",
            "`temperature` must be between 0 and 2",
        ));
    }
    // Narrowing a value already bounded to [0, 2].
    Ok((value as f32).max(GREEDY_TEMPERATURE))
}

fn top_p(value: Option<f64>) -> Result<f32, OpenAiError> {
    let Some(value) = value else {
        return Ok(DEFAULT_TOP_P);
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(OpenAiError::invalid_param(
            "top_p",
            "`top_p` must be between 0 and 1",
        ));
    }
    // Narrowing a value already bounded to [0, 1].
    Ok((value as f32).max(GREEDY_TOP_P))
}

/// OpenAI's seed is a signed 64-bit integer and the sampler's an unsigned
/// 32-bit one. A value outside the second is refused, not folded into it: two
/// seeds that sampled the same thing would make `seed` a lie.
fn seed(value: Option<i64>) -> Result<u32, OpenAiError> {
    match value {
        None => Ok(DEFAULT_SEED),
        Some(value) => u32::try_from(value).map_err(|_| {
            OpenAiError::invalid_param("seed", format!("`seed` must be between 0 and {}", u32::MAX))
        }),
    }
}

fn stop(value: Option<Stop>) -> Result<Vec<String>, OpenAiError> {
    let stops = match value {
        None => Vec::new(),
        Some(Stop::One(stop)) => vec![stop],
        Some(Stop::Many(stops)) => stops,
    };
    if stops.len() > MAX_STOP {
        return Err(OpenAiError::invalid_param(
            "stop",
            format!("`stop` takes at most {MAX_STOP} sequences"),
        ));
    }
    // An empty sequence would match at the first byte and answer nothing.
    Ok(stops.into_iter().filter(|stop| !stop.is_empty()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(body: Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("a well-formed request")
    }

    fn prepared(body: Value) -> Result<Prepared, OpenAiError> {
        prepare(request(body))
    }

    fn base() -> Value {
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    }

    fn with(field: &str, value: Value) -> Value {
        let mut body = base();
        body[field] = value;
        body
    }

    #[test]
    fn defaults_fill_what_the_client_left_out() {
        let prepared = prepared(base()).unwrap();
        assert_eq!(prepared.prompt.sampling.max_new_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(prepared.prompt.sampling.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(prepared.prompt.sampling.top_p, DEFAULT_TOP_P);
        assert_eq!(prepared.prompt.sampling.seed, DEFAULT_SEED);
        assert!(!prepared.stream);
    }

    #[test]
    fn parts_concatenate_their_text_and_refuse_everything_else() {
        let body = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]});
        assert_eq!(
            prepared(body).unwrap().prompt.messages[0].content,
            "one two"
        );

        let body = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:"}}]}]});
        let error = prepared(body).unwrap_err();
        assert_eq!(error.code, Some("unsupported_content"));
        assert_eq!(error.param.as_deref(), Some("messages[0].content[1]"));
    }

    #[test]
    fn developer_is_a_system_turn() {
        let body = json!({"model": "m", "messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "hi"}]});
        assert_eq!(
            prepared(body).unwrap().prompt.messages[0].role,
            Role::System
        );
    }

    #[test]
    fn an_assistant_call_keeps_its_id_and_its_arguments_as_an_object() {
        let body = json!({"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "move", "arguments": "{\"direction\":\"left\"}"}},
                {"type": "function", "function": {"name": "look", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "moved"}]});
        let prompt = prepared(body).unwrap().prompt;
        let calls = &prompt.messages[1].tool_calls;
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].arguments, json!({"direction": "left"}));
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments, json!({}));
        assert_eq!(prompt.messages[2].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn malformed_arguments_and_orphan_observations_are_located() {
        let body = json!({"model": "m", "messages": [
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "move", "arguments": "{oops"}}]}]});
        let error = prepared(body).unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("messages[0].tool_calls[0].function.arguments")
        );

        let body = json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]});
        let error = prepared(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("messages[0].tool_call_id"));

        let body = json!({"model": "m", "messages": [{"role": "narrator", "content": "x"}]});
        assert_eq!(
            prepared(body).unwrap_err().param.as_deref(),
            Some("messages[0].role")
        );
    }

    #[test]
    fn numbers_at_their_bounds_are_refused_rather_than_folded() {
        for (field, value) in [
            ("seed", json!(-1)),
            ("seed", json!(4_294_967_296_u64)),
            ("max_tokens", json!(4_294_967_296_u64)),
            ("max_tokens", json!(0)),
            ("max_completion_tokens", json!(0)),
            ("temperature", json!(-0.1)),
            ("temperature", json!(2.5)),
            ("top_p", json!(1.5)),
        ] {
            let error = prepared(with(field, value.clone())).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field), "{field} = {value}");
        }
        let prompt = prepared(with("seed", json!(4_294_967_295_u64)))
            .unwrap()
            .prompt;
        assert_eq!(prompt.sampling.seed, u32::MAX);
    }

    #[test]
    fn the_newer_token_limit_wins() {
        let mut body = with("max_tokens", json!(10));
        body["max_completion_tokens"] = json!(20);
        assert_eq!(prepared(body).unwrap().prompt.sampling.max_new_tokens, 20);
    }

    #[test]
    fn zero_means_greedy_to_a_sampler_that_refuses_zero() {
        let mut body = with("temperature", json!(0));
        body["top_p"] = json!(0);
        let sampling = prepared(body).unwrap().prompt.sampling;
        assert_eq!(sampling.temperature, GREEDY_TEMPERATURE);
        assert_eq!(sampling.top_p, GREEDY_TOP_P);
    }

    #[test]
    fn stop_takes_one_or_up_to_four() {
        assert_eq!(
            prepared(with("stop", json!("\n"))).unwrap().prompt.stop,
            ["\n"]
        );
        assert_eq!(
            prepared(with("stop", json!(["a", "", "b"])))
                .unwrap()
                .prompt
                .stop,
            ["a", "b"]
        );
        let error = prepared(with("stop", json!(["a", "b", "c", "d", "e"]))).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("stop"));
    }

    /// One row per field of the contract that is read: honoured in its neutral
    /// form, refused otherwise, and the refusal names the field.
    #[test]
    fn a_parameter_that_would_change_the_sample_is_never_ignored() {
        for (field, neutral, refused) in [
            ("n", json!(1), json!(2)),
            ("frequency_penalty", json!(0), json!(0.5)),
            ("presence_penalty", json!(0.0), json!(-1)),
            ("logprobs", json!(false), json!(true)),
            ("top_logprobs", json!(0), json!(5)),
            ("logit_bias", json!({}), json!({"42": 100})),
            (
                "response_format",
                json!({"type": "text"}),
                json!({"type": "json_object"}),
            ),
            ("modalities", json!(["text"]), json!(["text", "audio"])),
            ("audio", Value::Null, json!({"voice": "alloy"})),
            ("tool_choice", json!("auto"), json!("required")),
        ] {
            prepared(with(field, neutral.clone()))
                .unwrap_or_else(|error| panic!("{field} = {neutral}: {error}"));
            let error = prepared(with(field, refused.clone())).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field), "{field} = {refused}");
            assert_eq!(error.code, Some("unsupported_parameter"), "{field}");
        }
        let error = prepared(with(
            "tool_choice",
            json!({"type": "function", "function": {"name": "move"}}),
        ))
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tool_choice"));
    }

    #[test]
    fn fields_that_cannot_change_a_token_are_dropped() {
        let mut body = base();
        for (field, value) in [
            ("user", json!("someone")),
            ("metadata", json!({"k": "v"})),
            ("store", json!(false)),
            ("a_field_from_next_year", json!(1)),
        ] {
            body[field] = value;
        }
        prepared(body).unwrap();
    }

    #[test]
    fn tool_choice_none_offers_no_catalog() {
        let mut body = with("tool_choice", json!("none"));
        body["tools"] = json!([{"type": "function", "function": {"name": "move"}}]);
        assert!(prepared(body).unwrap().prompt.tools.is_empty());

        let body = with(
            "tools",
            json!([{"type": "function", "function": {"name": "move"}}]),
        );
        let prompt = prepared(body).unwrap().prompt;
        assert_eq!(prompt.tools.len(), 1);
        assert_eq!(prompt.tools[0].input_schema["type"], "object");
    }

    #[test]
    fn one_call_at_most_is_refused_where_tools_are_offered() {
        let tools = json!([{"type": "function", "function": {"name": "move"}}]);
        let mut body = with("parallel_tool_calls", json!(false));
        prepared(body.clone()).expect("no catalog, nothing to limit");
        body["tools"] = tools.clone();
        let error = prepared(body.clone()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("parallel_tool_calls"));
        assert_eq!(error.code, Some("unsupported_parameter"));
        body["tool_choice"] = json!("none");
        prepared(body).expect("no catalog offered, nothing to limit");

        let mut body = with("parallel_tool_calls", json!(true));
        body["tools"] = tools;
        prepared(body).unwrap();
    }

    #[test]
    fn usage_is_only_streamed_when_asked_for() {
        let mut body = with("stream", json!(true));
        body["stream_options"] = json!({"include_usage": true});
        let prepared = prepared(body).unwrap();
        assert!(prepared.stream && prepared.include_usage);
    }
}
