//! From the chat wire request to the common [`GenerationRequest`].
//!
//! The rule this module enforces, field by field: a parameter that would change
//! the sampled distribution is never ignored. It is honoured, accepted in its
//! neutral form (`frequency_penalty: 0`, `logprobs: false`), or refused with the
//! field named. Only the fields that cannot change a token - `user`,
//! `metadata`, `store` - are dropped, because the clients that send them send
//! them on every request.

use retrograd_agent_core::tools::{ToolCall, ToolSpec};
use retrograd_agent_core::trajectory::{Message, Role};
use retrograd_core::SamplingParamsV2;
use serde_json::Value;

use crate::chat::wire::{ChatCompletionRequest, Content, Stop, WireMessage, WireTool};
use crate::error::OpenAiError;
use crate::grammar::response_format_grammar;
use crate::request::{GenerationRequest, ToolConstraint};

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
/// How many completions of one prompt one request may ask for. The scheduler
/// gives each choice its own slot of the pool, so the bound is a contract,
/// not a capacity: a request beyond it is refused, not queued.
pub(crate) const MAX_N: u64 = 128;
/// OpenAI's own bound on `top_logprobs`.
pub(crate) const MAX_TOP_LOGPROBS: u64 = 20;
/// OpenAI's own bounds on the penalties.
const PENALTY_RANGE: std::ops::RangeInclusive<f64> = -2.0..=2.0;
/// OpenAI's own bounds on a logit bias.
const LOGIT_BIAS_RANGE: std::ops::RangeInclusive<f64> = -100.0..=100.0;
/// Cap on the entries of one `logit_bias`, mirroring the runtime's.
const MAX_LOGIT_BIAS: usize = 4096;

/// A validated request: which id to serve, and the common
/// [`GenerationRequest`] the surface translated the wire body into.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub model: String,
    pub request: GenerationRequest,
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
    let choice = request.tool_choice.as_ref();
    let constraint = tool_constraint(choice, &tools)?;
    // `tool_choice: "none"` offers no catalog: with no constrained decoding,
    // a catalog in the prompt is only an invitation to call.
    let tools = match offers_tools(choice) {
        true => tools,
        false => Vec::new(),
    };
    // `parallel_tool_calls: false` limits the calls to one through the
    // template's own grammar - never by cutting extra calls off afterwards,
    // which would answer with a turn the model did not write.
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
    let grammar = match request.response_format.as_ref() {
        Some(format) => response_format_grammar(format)?,
        None => None,
    };
    // A structured output and a tool constraint are two grammars over the
    // same tokens: refusing the combination is the only honest answer, since
    // one constraint cannot be derived from the other.
    if grammar.is_some() && (constraint.is_some() || !tools.is_empty()) {
        return Err(OpenAiError::unsupported(
            "response_format",
            "a json response format cannot be combined with tools or a forced tool call; \
             send one of them",
        ));
    }
    let stream = request.stream.unwrap_or(false);
    let n = request.n.unwrap_or(1);
    if n == 0 || n > MAX_N {
        return Err(OpenAiError::invalid_param(
            "n",
            format!("`n` must be between 1 and {MAX_N}"),
        ));
    }
    // Narrowing checked against MAX_N above: cannot lose anything.
    let n = u32::try_from(n).map_err(|_| OpenAiError::invalid_param("n", "`n` does not fit"))?;
    let logprobs = logprobs(request.logprobs.as_ref(), request.top_logprobs.as_ref())?;
    Ok(Prepared {
        model: request.model,
        request: GenerationRequest {
            raw_prompt: None,
            messages,
            tools,
            sampling: SamplingParamsV2 {
                temperature: temperature(request.temperature)?,
                top_p: top_p(request.top_p)?,
                max_new_tokens: max_tokens(request.max_tokens, request.max_completion_tokens)?,
                seed: seed(request.seed)?,
                top_k: top_k(request.top_k)?,
                min_p: min_p(request.min_p)?,
                repeat_penalty: repeat_penalty(request.repeat_penalty)?,
                frequency_penalty: penalty("frequency_penalty", request.frequency_penalty)?,
                presence_penalty: penalty("presence_penalty", request.presence_penalty)?,
                penalty_window: penalty_window(request.penalty_window)?,
                logit_bias: logit_bias(request.logit_bias.as_ref())?,
                grammar_gbnf: grammar,
            },
            stop: stop(request.stop)?,
            n,
            logprobs,
            tool_constraint: constraint,
            reasoning_budget: None,
        },
        stream,
        include_usage: stream
            && request
                .stream_options
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
    })
}

/// What remains refused outright: the non-text modalities of a multimodal
/// request. Everything that can change a token is honoured or refused with a
/// reason in `prepare` - a parameter is never silently ignored.
fn refuse_unsupported(request: &ChatCompletionRequest) -> Result<(), OpenAiError> {
    if let Some(audio) = &request.audio
        && !audio.is_null()
    {
        return Err(OpenAiError::unsupported(
            "audio",
            "this server serves text models; `audio` is not supported",
        ));
    }
    if let Some(modalities) = &request.modalities {
        let text_only = modalities
            .as_array()
            .is_some_and(|items| items.iter().all(|item| item == "text"));
        if !text_only {
            return Err(OpenAiError::unsupported(
                "modalities",
                "only the `text` modality is supported by this server",
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
        reasoning: None,
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
pub(crate) fn arguments(param: &str, value: Option<&Value>) -> Result<Value, OpenAiError> {
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

/// Whether the catalog is offered to the model at all (`auto`, or a
/// constraint - `none` offers none).
pub(crate) fn offers_tools(choice: Option<&Value>) -> bool {
    !matches!(choice, Some(Value::String(choice)) if choice == "none")
}

/// The tool-calling constraint a `tool_choice` asks for, when it is one:
/// `required` forces at least one call, the object form forces exactly the
/// named function. Both are realized as a grammar derived from the template
/// at rendering time, never by deleting calls after the fact.
pub(crate) fn tool_constraint(
    choice: Option<&Value>,
    tools: &[ToolSpec],
) -> Result<Option<ToolConstraint>, OpenAiError> {
    match choice {
        None => Ok(None),
        Some(Value::String(choice)) if choice == "auto" || choice == "none" => Ok(None),
        Some(Value::String(choice)) if choice == "required" => {
            if tools.is_empty() {
                return Err(OpenAiError::invalid_param(
                    "tool_choice",
                    "`tool_choice: \"required\"` needs a `tools` catalog to choose from",
                ));
            }
            Ok(Some(ToolConstraint::Required))
        }
        Some(Value::String(other)) => Err(OpenAiError::unsupported(
            "tool_choice",
            format!(
                "unsupported `tool_choice` `{other}`; expected auto, none, required or a forced function"
            ),
        )),
        Some(Value::Object(object)) => {
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err(OpenAiError::invalid_param(
                    "tool_choice.type",
                    "a forced tool choice must be of type `function`",
                ));
            }
            let name = object
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    OpenAiError::invalid_param(
                        "tool_choice.function.name",
                        "the forced function needs a `name`",
                    )
                })?;
            if tools.is_empty() {
                return Err(OpenAiError::invalid_param(
                    "tool_choice",
                    "a forced function needs a `tools` catalog to choose from",
                ));
            }
            if !tools.iter().any(|tool| tool.name == name) {
                return Err(OpenAiError::invalid_param(
                    "tool_choice.function.name",
                    format!("the forced function `{name}` is not in the `tools` catalog"),
                ));
            }
            Ok(Some(ToolConstraint::Forced(name.to_owned())))
        }
        Some(_) => Err(OpenAiError::invalid_param(
            "tool_choice",
            "`tool_choice` must be a string or a function object",
        )),
    }
}

/// `logprobs`/`top_logprobs` as the count of top pairs to return, when
/// logprobs are asked for at all. `top_logprobs` without `logprobs` is the
/// one combination the contract refuses.
pub(crate) fn logprobs(
    logprobs: Option<&Value>,
    top_logprobs: Option<&Value>,
) -> Result<Option<u32>, OpenAiError> {
    let asked = match logprobs {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                "logprobs",
                "`logprobs` must be a boolean",
            ));
        }
    };
    let top = match top_logprobs {
        None | Some(Value::Null) => 0,
        Some(Value::Number(value)) => value.as_u64().ok_or_else(|| {
            OpenAiError::invalid_param(
                "top_logprobs",
                "`top_logprobs` must be a non-negative integer",
            )
        })?,
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                "top_logprobs",
                "`top_logprobs` must be an integer",
            ));
        }
    };
    if top > 0 && !asked {
        return Err(OpenAiError::invalid_param(
            "top_logprobs",
            "`top_logprobs` requires `logprobs: true`",
        ));
    }
    if top > MAX_TOP_LOGPROBS {
        return Err(OpenAiError::invalid_param(
            "top_logprobs",
            format!("`top_logprobs` must be at most {MAX_TOP_LOGPROBS}"),
        ));
    }
    // Narrowing checked against MAX_TOP_LOGPROBS above: exact.
    Ok(asked.then_some(top as u32))
}

/// `top_k`: -1 (or absent) disables; anything below -1 or zero is refused.
pub(crate) fn top_k(value: Option<i64>) -> Result<i32, OpenAiError> {
    match value {
        None => Ok(0),
        Some(-1) => Ok(0),
        Some(value) if value > 0 => i32::try_from(value).map_err(|_| {
            OpenAiError::invalid_param("top_k", format!("`top_k` must be at most {}", i32::MAX))
        }),
        Some(_) => Err(OpenAiError::invalid_param(
            "top_k",
            "`top_k` must be -1 (disabled) or a positive integer",
        )),
    }
}

pub(crate) fn min_p(value: Option<f64>) -> Result<f32, OpenAiError> {
    let Some(value) = value else {
        return Ok(0.0);
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(OpenAiError::invalid_param(
            "min_p",
            "`min_p` must be between 0 and 1",
        ));
    }
    // Narrowing a value already bounded to [0, 1]; 0 disables.
    Ok(value as f32)
}

/// The repeat penalty has no wire field in the OpenAI contract; it is a local
/// extension, so its bound is the sampler's own: finite and > 0, 1.0 off.
pub(crate) fn repeat_penalty(value: Option<f64>) -> Result<f32, OpenAiError> {
    let Some(value) = value else {
        return Ok(1.0);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(OpenAiError::invalid_param(
            "repeat_penalty",
            "`repeat_penalty` must be finite and greater than zero (1.0 disables it)",
        ));
    }
    // A contract value must remain finite and positive after narrowing.
    let narrowed = value as f32;
    if !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(OpenAiError::invalid_param(
            "repeat_penalty",
            "`repeat_penalty` must fit a finite positive f32",
        ));
    }
    Ok(narrowed)
}

/// The frequency/presence penalties, bounded as OpenAI bounds them; 0 is the
/// neutral value and is accepted.
pub(crate) fn penalty(param: &str, value: Option<f64>) -> Result<f32, OpenAiError> {
    let Some(value) = value else {
        return Ok(0.0);
    };
    if !value.is_finite() || !PENALTY_RANGE.contains(&value) {
        return Err(OpenAiError::invalid_param(
            param,
            format!(
                "`{param}` must be between {} and {}",
                PENALTY_RANGE.start(),
                PENALTY_RANGE.end()
            ),
        ));
    }
    // Narrowing a value already bounded to [-2, 2].
    Ok(value as f32)
}

/// The window the penalties look back over: a local extension, `0` (the
/// default) penalizes over the whole generated history.
pub(crate) fn penalty_window(value: Option<u64>) -> Result<u32, OpenAiError> {
    match value {
        None => Ok(0),
        Some(value) => u32::try_from(value).map_err(|_| {
            OpenAiError::invalid_param(
                "penalty_window",
                format!("`penalty_window` must be at most {}", u32::MAX),
            )
        }),
    }
}

/// `logit_bias` as `(token id, bias)` pairs: the keys must be token ids, the
/// values bounded as OpenAI bounds them.
pub(crate) fn logit_bias(value: Option<&Value>) -> Result<Vec<(i32, f32)>, OpenAiError> {
    let Some(object) = value else {
        return Ok(Vec::new());
    };
    let Some(object) = object.as_object() else {
        return Err(OpenAiError::invalid_param(
            "logit_bias",
            "`logit_bias` must be an object mapping token ids to biases",
        ));
    };
    if object.len() > MAX_LOGIT_BIAS {
        return Err(OpenAiError::invalid_param(
            "logit_bias",
            format!("`logit_bias` takes at most {MAX_LOGIT_BIAS} entries"),
        ));
    }
    object
        .iter()
        .map(|(id, bias)| {
            let id = id.parse::<i32>().map_err(|_| {
                OpenAiError::invalid_param(
                    format!("logit_bias.{id}"),
                    "a logit bias key must be a token id (an integer)",
                )
            })?;
            let bias = bias.as_f64().ok_or_else(|| {
                OpenAiError::invalid_param(
                    format!("logit_bias.{id}"),
                    "a logit bias must be a number",
                )
            })?;
            if !bias.is_finite() || !LOGIT_BIAS_RANGE.contains(&bias) {
                return Err(OpenAiError::invalid_param(
                    format!("logit_bias.{id}"),
                    format!(
                        "a logit bias must be between {} and {}",
                        LOGIT_BIAS_RANGE.start(),
                        LOGIT_BIAS_RANGE.end()
                    ),
                ));
            }
            // Narrowing a value already bounded to [-100, 100].
            Ok((id, bias as f32))
        })
        .collect()
}

/// `max_completion_tokens` is the newer name and wins when both are sent.
pub(crate) fn max_tokens(legacy: Option<u64>, current: Option<u64>) -> Result<u32, OpenAiError> {
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

pub(crate) fn temperature(value: Option<f64>) -> Result<f32, OpenAiError> {
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

pub(crate) fn top_p(value: Option<f64>) -> Result<f32, OpenAiError> {
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
pub(crate) fn seed(value: Option<i64>) -> Result<u32, OpenAiError> {
    match value {
        None => Ok(DEFAULT_SEED),
        Some(value) => u32::try_from(value).map_err(|_| {
            OpenAiError::invalid_param("seed", format!("`seed` must be between 0 and {}", u32::MAX))
        }),
    }
}

pub(crate) fn stop(value: Option<Stop>) -> Result<Vec<String>, OpenAiError> {
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
        assert_eq!(prepared.request.sampling.max_new_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(prepared.request.sampling.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(prepared.request.sampling.top_p, DEFAULT_TOP_P);
        assert_eq!(prepared.request.sampling.seed, DEFAULT_SEED);
        assert!(!prepared.stream);
    }

    #[test]
    fn parts_concatenate_their_text_and_refuse_everything_else() {
        let body = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]});
        assert_eq!(
            prepared(body).unwrap().request.messages[0].content,
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
            prepared(body).unwrap().request.messages[0].role,
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
        let request = prepared(body).unwrap().request;
        let calls = &request.messages[1].tool_calls;
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].arguments, json!({"direction": "left"}));
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments, json!({}));
        assert_eq!(request.messages[2].tool_call_id.as_deref(), Some("c1"));
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
        let request = prepared(with("seed", json!(4_294_967_295_u64)))
            .unwrap()
            .request;
        assert_eq!(request.sampling.seed, u32::MAX);
    }

    #[test]
    fn the_newer_token_limit_wins() {
        let mut body = with("max_tokens", json!(10));
        body["max_completion_tokens"] = json!(20);
        assert_eq!(prepared(body).unwrap().request.sampling.max_new_tokens, 20);
    }

    #[test]
    fn zero_means_greedy_to_a_sampler_that_refuses_zero() {
        let mut body = with("temperature", json!(0));
        body["top_p"] = json!(0);
        let sampling = prepared(body).unwrap().request.sampling;
        assert_eq!(sampling.temperature, GREEDY_TEMPERATURE);
        assert_eq!(sampling.top_p, GREEDY_TOP_P);
    }

    #[test]
    fn stop_takes_one_or_up_to_four() {
        assert_eq!(
            prepared(with("stop", json!("\n"))).unwrap().request.stop,
            ["\n"]
        );
        assert_eq!(
            prepared(with("stop", json!(["a", "", "b"])))
                .unwrap()
                .request
                .stop,
            ["a", "b"]
        );
        let error = prepared(with("stop", json!(["a", "b", "c", "d", "e"]))).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("stop"));
    }

    /// One row per field of the contract that is read: honoured in its neutral
    /// form, refused with the field named when the value is out of bounds,
    /// and never silently ignored.
    #[test]
    fn a_parameter_that_would_change_the_sample_is_never_ignored() {
        for (field, neutral, refused, param) in [
            ("n", json!(1), json!(0), "n"),
            ("n", json!(1), json!(129), "n"),
            (
                "frequency_penalty",
                json!(0),
                json!(2.5),
                "frequency_penalty",
            ),
            (
                "presence_penalty",
                json!(0.0),
                json!(-2.5),
                "presence_penalty",
            ),
            ("logprobs", json!(false), json!("yes"), "logprobs"),
            ("top_logprobs", json!(0), json!(21), "top_logprobs"),
            ("top_logprobs", json!(0), json!(-1), "top_logprobs"),
            ("logit_bias", json!({}), json!({"x": 1}), "logit_bias.x"),
            ("logit_bias", json!({}), json!({"42": 101}), "logit_bias.42"),
            ("top_k", json!(-1), json!(0), "top_k"),
            ("min_p", json!(0.0), json!(1.5), "min_p"),
            ("repeat_penalty", json!(1.0), json!(0.0), "repeat_penalty"),
            (
                "modalities",
                json!(["text"]),
                json!(["text", "audio"]),
                "modalities",
            ),
            ("audio", Value::Null, json!({"voice": "alloy"}), "audio"),
        ] {
            prepared(with(field, neutral.clone()))
                .unwrap_or_else(|error| panic!("{field} = {neutral}: {error}"));
            let error = prepared(with(field, refused.clone())).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param), "{field} = {refused}");
        }
        // `top_logprobs` without `logprobs` is refused.
        let error = prepared(with("top_logprobs", json!(5))).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("top_logprobs"));
    }

    /// The sampling fields now land in the request's extended parameters.
    #[test]
    fn sampling_fields_reach_the_extended_parameters() {
        let mut body = with("top_k", json!(40));
        body["min_p"] = json!(0.05);
        body["frequency_penalty"] = json!(0.5);
        body["presence_penalty"] = json!(-0.5);
        body["repeat_penalty"] = json!(1.1);
        body["penalty_window"] = json!(64);
        body["logit_bias"] = json!({"42": -10});
        body["n"] = json!(3);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(5);
        let request = prepared(body).unwrap().request;
        assert_eq!(request.sampling.top_k, 40);
        assert_eq!(request.sampling.min_p, 0.05);
        assert_eq!(request.sampling.frequency_penalty, 0.5);
        assert_eq!(request.sampling.presence_penalty, -0.5);
        assert_eq!(request.sampling.repeat_penalty, 1.1);
        assert_eq!(request.sampling.penalty_window, 64);
        assert_eq!(request.sampling.logit_bias, vec![(42, -10.0)]);
        assert_eq!(request.n, 3);
        assert_eq!(request.logprobs, Some(5));
        assert!(request.sampling.grammar_gbnf.is_none());
        assert!(!request.sampling.is_v1());
    }

    #[test]
    fn a_json_object_format_constrains_the_output_to_json() {
        let request = prepared(with("response_format", json!({"type": "json_object"})))
            .unwrap()
            .request;
        let grammar = request.sampling.grammar_gbnf.expect("a grammar");
        assert!(grammar.contains("root"), "the GBNF names its root rule");
    }

    #[test]
    fn a_json_schema_format_is_validated_at_admission() {
        let ok = with(
            "response_format",
            json!({"type": "json_schema", "json_schema": {
                "name": "point", "schema": {"type": "object", "properties": {"x": {"type": "number"}}}
            }}),
        );
        let request = prepared(ok).unwrap().request;
        assert!(request.sampling.grammar_gbnf.is_some());

        let bad = with(
            "response_format",
            json!({"type": "json_schema", "json_schema": {
                "name": "point", "schema": {"type": "object", "patternProperties": {}}
            }}),
        );
        let error = prepared(bad).unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("response_format.json_schema.schema/patternProperties")
        );
    }

    #[test]
    fn a_structured_output_and_tools_are_refused_together() {
        let mut body = with("response_format", json!({"type": "json_object"}));
        body["tools"] = json!([{"type": "function", "function": {"name": "move"}}]);
        let error = prepared(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("response_format"));
    }

    #[test]
    fn tool_choice_required_and_forced_become_constraints() {
        let tools = json!([{"type": "function", "function": {"name": "move"}}]);
        let mut body = with("tool_choice", json!("required"));
        body["tools"] = tools.clone();
        let request = prepared(body).unwrap().request;
        assert_eq!(request.tool_constraint, Some(ToolConstraint::Required));
        assert_eq!(request.tools.len(), 1);

        let mut body = with(
            "tool_choice",
            json!({"type": "function", "function": {"name": "move"}}),
        );
        body["tools"] = tools.clone();
        let request = prepared(body).unwrap().request;
        assert_eq!(
            request.tool_constraint,
            Some(ToolConstraint::Forced("move".to_owned()))
        );

        // Forcing a function outside the catalog is refused, field-localized.
        let mut body = with(
            "tool_choice",
            json!({"type": "function", "function": {"name": "fly"}}),
        );
        body["tools"] = tools.clone();
        let error = prepared(body).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tool_choice.function.name"));

        // `required` without a catalog is refused.
        let error = prepared(with("tool_choice", json!("required"))).unwrap_err();
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
        assert!(prepared(body).unwrap().request.tools.is_empty());

        let body = with(
            "tools",
            json!([{"type": "function", "function": {"name": "move"}}]),
        );
        let request = prepared(body).unwrap().request;
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].input_schema["type"], "object");
    }

    #[test]
    fn one_call_at_most_becomes_a_constraint_where_tools_are_offered() {
        let tools = json!([{"type": "function", "function": {"name": "move"}}]);
        let mut body = with("parallel_tool_calls", json!(false));
        prepared(body.clone()).expect("no catalog, nothing to limit");
        body["tools"] = tools.clone();
        let request = prepared(body.clone()).unwrap().request;
        assert_eq!(request.tool_constraint, Some(ToolConstraint::AtMostOne));
        // Combined with `required`/`forced`: refused, one constraint at a time.
        body["tool_choice"] = json!("required");
        let error = prepared(body.clone()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("parallel_tool_calls"));
        body["tool_choice"] = json!("none");
        let request = prepared(body).unwrap().request;
        assert!(
            request.tools.is_empty(),
            "no catalog offered, nothing to limit"
        );
        assert_eq!(request.tool_constraint, None);

        let mut body = with("parallel_tool_calls", json!(true));
        body["tools"] = tools;
        assert!(prepared(body).unwrap().request.tool_constraint.is_none());
    }

    #[test]
    fn usage_is_only_streamed_when_asked_for() {
        let mut body = with("stream", json!(true));
        body["stream_options"] = json!({"include_usage": true});
        let prepared = prepared(body).unwrap();
        assert!(prepared.stream && prepared.include_usage);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn repeat_penalties_must_survive_the_native_narrowing() {
        for value in [1e-100, 1e100] {
            let error = repeat_penalty(Some(value)).unwrap_err();
            assert_eq!(error.param.as_deref(), Some("repeat_penalty"));
        }
        assert_eq!(repeat_penalty(Some(1.1)).unwrap(), 1.1_f32);
    }
}
