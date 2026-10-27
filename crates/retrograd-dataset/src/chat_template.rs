//! The contract a chat template is driven through when a conversation has
//! tools: what the template is handed, how its output is cut around the
//! assistant turns, how the catalog and the observations are written, and how
//! an assistant turn is read back into calls.
//!
//! Two consumers need it byte for byte: the agentic rollout, which samples the
//! assistant turns and renders only the framing around them, and SFT on tool
//! conversations, which has to rebuild the exact token stream that rollout
//! produces and parses. One implementation, here, is what keeps them from
//! drifting apart - a warm-start that teaches a format the rollout does not
//! read back teaches nothing. The module is pure: no model, no FFI, no async.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// Why a conversation could not be handed to, or cut out of, a chat template.
#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    #[error("serialize {what}: {source}")]
    Serialize {
        what: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "chat template did not render assistant turn {turn} of {turns}: a template that drops or \
         reorders a turn cannot be used for multi-turn rollouts"
    )]
    DroppedTurn { turn: usize, turns: usize },
    /// The inference server re-renders the history the same way, so the turns
    /// sampled after this one would continue a conversation it never shows.
    #[error(
        "chat template drops the reasoning of assistant turn {turn} once the conversation \
         continues past it, so the turns sampled after it continue a history inference never \
         renders: such a template cannot be used for multi-turn rollouts of a model that reasons"
    )]
    DroppedReasoning { turn: usize },
    #[error("chat template renders the reasoning of assistant turn {turn} after its content")]
    ReasoningAfterContent { turn: usize },
    #[error("chat template rendered the {part} of assistant turn {turn} more than once")]
    RepeatedSentinel { turn: usize, part: SpanPart },
    /// [`parse_hermes`] would cut the call short at the first closing tag, so
    /// writing it would teach a call that does not read back.
    #[error(
        "tool call '{name}' cannot be written in the <tool_call> convention: its arguments \
         contain the closing tag"
    )]
    UnwritableCall { name: String },
}

impl From<ChatTemplateError> for retrograd_core::Error {
    fn from(error: ChatTemplateError) -> Self {
        retrograd_core::Error::invalid(error.to_string())
    }
}

type Result<T> = std::result::Result<T, ChatTemplateError>;

/// The placeholder an assistant turn's text is replaced by before the template
/// sees it.
///
/// Deliberately plain: no whitespace at either edge and no markup, so a template
/// that trims content or splits it on `</think>` - Qwen's does both - passes it
/// through byte for byte and stays findable in the output. Suffixed rather than
/// prefixed with the index so no sentinel is a prefix of another.
pub fn assistant_span(index: usize) -> String {
    format!("retro_span_{index}_9d41c7")
}

/// The placeholder an assistant turn's reasoning is handed to the template as,
/// for a turn that has some. Same shape as [`assistant_span`], under a prefix
/// neither shares with the other.
///
/// The template writes a turn's reasoning in a block of its own, with framing
/// around it that a turn without reasoning does not get - and that the
/// generation prompt may already have opened. Handing it over apart from the
/// content is what lets that framing be cut out and compared, instead of being
/// written around a turn that was sampled without it.
pub fn reasoning_span(index: usize) -> String {
    format!("retro_reasoning_{index}_9d41c7")
}

/// Which of a turn's placeholders a cut refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanPart {
    Reasoning,
    Content,
}

impl std::fmt::Display for SpanPart {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Reasoning => "reasoning",
            Self::Content => "content",
        })
    }
}

/// The placeholders one assistant turn was handed to the template as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnSentinels {
    /// For a turn whose reasoning is not empty.
    pub reasoning: Option<String>,
    pub content: String,
}

impl TurnSentinels {
    fn parts(&self) -> impl Iterator<Item = (SpanPart, &str)> {
        self.reasoning
            .as_deref()
            .map(|reasoning| (SpanPart::Reasoning, reasoning))
            .into_iter()
            .chain([(SpanPart::Content, self.content.as_str())])
    }
}

/// What a template writes around one sampled turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnPieces {
    /// From the end of the previous turn - or the start of the text - to the
    /// turn's first placeholder: its reasoning when it has some, its content
    /// otherwise.
    pub pre: String,
    /// Between the reasoning and the content; `None` for a turn handed over
    /// without reasoning.
    pub mid: Option<String>,
}

/// A rendered conversation cut around its sampled turns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Framing {
    pub turns: Vec<TurnPieces>,
    /// After the last turn's content: what closes it, what follows it, and the
    /// generation prompt when one was asked for. The whole text for a
    /// conversation with no assistant turn yet.
    pub tail: String,
}

/// One tool call, as a template renders it and as a parser reads it back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemplateCall {
    pub id: String,
    pub name: String,
    /// Always an object: both parsers refuse anything else.
    pub arguments: Value,
}

/// One message as the template is handed it. A view that the rollout's
/// messages and a dataset record both fill, so neither owns the serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateMessage<'a> {
    pub role: &'a str,
    pub content: Cow<'a, str>,
    pub tool_call_id: Option<&'a str>,
    /// Only ever rendered for the assistant turn [`template_messages_revealing`]
    /// reveals; a sentinelled turn never shows its calls.
    pub tool_calls: &'a [TemplateCall],
    /// Handed over as `reasoning_content`: for real on a revealed turn, as a
    /// [`reasoning_span`] on a sentinelled one - when it is not empty.
    pub reasoning: Option<&'a str>,
}

impl<'a> TemplateMessage<'a> {
    pub fn text(role: &'a str, content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: &[],
            reasoning: None,
        }
    }
}

impl SystemTurn for TemplateMessage<'_> {
    fn system_content(&mut self) -> Option<&mut String> {
        (self.role == "system").then(|| self.content.to_mut())
    }

    fn new_system(content: String) -> Self {
        Self::text("system", content)
    }
}

/// Serializes a conversation for the model's chat template, replacing every
/// assistant turn's content with a sentinel - and its reasoning, when it has
/// some, with another. Returns the JSON and each turn's sentinels in turn
/// order.
///
/// The template never sees a sampled turn, on purpose. It cannot be trusted with
/// one: it is free to trim the content, to lift a `<think>` block out of it and
/// re-emit it somewhere else, or to render its own serialization of the tool
/// calls - and even a template that does none of that only gets the *text* back,
/// while what is being trained on is the *tokens*, which sampling does not
/// produce in canonical form (a model that sampled `"]]` + `])` re-tokenizes as
/// `"]` + `]])`). Splitting the rendered text on the sentinels yields the framing
/// around the sampled turns, which is the only part of a multi-turn prompt the
/// template gets to decide.
///
/// An assistant turn's `tool_calls` are not handed over either, for the same
/// reason: the calls are already inside the sampled tokens. Its reasoning is,
/// as [`reasoning_span`], because where the template puts it - and what it
/// writes around it - is framing the sampled turn has to be checked against.
pub fn template_messages(messages: &[TemplateMessage<'_>]) -> Result<(String, Vec<TurnSentinels>)> {
    serialize_messages(messages, Reveal::None)
}

/// [`template_messages`] with assistant turn `reveal` (counted among assistant
/// turns) handed over for real - its content and its calls, arguments as an
/// object, the HF convention - and every other one still a sentinel.
///
/// Rendered next to the all-sentinel form, the difference between the two is
/// exactly what the template writes for that one turn, which is how a
/// structured turn is given the text the model's own template would write.
pub fn template_messages_revealing(
    messages: &[TemplateMessage<'_>],
    reveal: usize,
) -> Result<String> {
    serialize_messages(messages, Reveal::One(reveal)).map(|(json, _)| json)
}

/// The conversation as a client sent it: every assistant turn handed over for
/// real, content and calls, and no sentinel anywhere.
///
/// For a caller that has no sampled tokens to preserve - a server answering a
/// chat request only has the text the client sends back, so the template is the
/// only thing that can decide how an earlier assistant turn reads.
pub fn template_messages_verbatim(messages: &[TemplateMessage<'_>]) -> Result<String> {
    serialize_messages(messages, Reveal::All).map(|(json, _)| json)
}

/// Which assistant turns [`serialize_messages`] hands over instead of a sentinel.
#[derive(Clone, Copy)]
enum Reveal {
    None,
    One(usize),
    All,
}

impl Reveal {
    fn reveals(self, index: usize) -> bool {
        match self {
            Self::None => false,
            Self::One(revealed) => revealed == index,
            Self::All => true,
        }
    }
}

fn serialize_messages(
    messages: &[TemplateMessage<'_>],
    reveal: Reveal,
) -> Result<(String, Vec<TurnSentinels>)> {
    let mut sentinels = Vec::new();
    let mut assistant = 0;
    let rendered = messages
        .iter()
        .map(|message| {
            let mut object = Map::new();
            object.insert("role".into(), Value::String(message.role.into()));
            let content = if message.role == "assistant" {
                let index = assistant;
                assistant += 1;
                if reveal.reveals(index) {
                    if !message.tool_calls.is_empty() {
                        object.insert("tool_calls".into(), openai_calls(message.tool_calls));
                    }
                    if let Some(reasoning) = message.reasoning {
                        object.insert("reasoning_content".into(), Value::String(reasoning.into()));
                    }
                    message.content.to_string()
                } else {
                    let reasoning = message
                        .reasoning
                        .filter(|reasoning| !reasoning.is_empty())
                        .map(|_| reasoning_span(index));
                    if let Some(reasoning) = &reasoning {
                        object.insert("reasoning_content".into(), Value::String(reasoning.clone()));
                    }
                    let content = assistant_span(index);
                    sentinels.push(TurnSentinels {
                        reasoning,
                        content: content.clone(),
                    });
                    content
                }
            } else {
                message.content.to_string()
            };
            object.insert("content".into(), Value::String(content));
            if let Some(call_id) = message.tool_call_id {
                object.insert("tool_call_id".into(), Value::String(call_id.into()));
            }
            Value::Object(object)
        })
        .collect::<Vec<_>>();
    let json = serde_json::to_string(&rendered).map_err(|source| ChatTemplateError::Serialize {
        what: "chat messages",
        source,
    })?;
    Ok((json, sentinels))
}

fn openai_calls(calls: &[TemplateCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect(),
    )
}

/// Cuts a rendered conversation into the framing around the sampled turns.
///
/// Every sentinel must be rendered exactly once, a turn's reasoning before its
/// content, and the turns in order: a template that drops, repeats or
/// reorders one is caught here rather than corrupting the token stream
/// downstream.
pub fn split_turns(rendered: &str, sentinels: &[TurnSentinels]) -> Result<Framing> {
    for (turn, turn_sentinels) in sentinels.iter().enumerate() {
        for (part, sentinel) in turn_sentinels.parts() {
            match (rendered.matches(sentinel).count(), part) {
                (1, _) => {}
                (0, SpanPart::Content) => {
                    return Err(ChatTemplateError::DroppedTurn {
                        turn,
                        turns: sentinels.len(),
                    });
                }
                (0, SpanPart::Reasoning) => {
                    return Err(ChatTemplateError::DroppedReasoning { turn });
                }
                _ => return Err(ChatTemplateError::RepeatedSentinel { turn, part }),
            }
        }
    }
    let mut turns = Vec::with_capacity(sentinels.len());
    let mut rest = rendered;
    for (turn, turn_sentinels) in sentinels.iter().enumerate() {
        // Each sentinel is in the text once; one that is not in what is left
        // was rendered before an earlier turn.
        let Some(content) = rest.find(&turn_sentinels.content) else {
            return Err(ChatTemplateError::DroppedTurn {
                turn,
                turns: sentinels.len(),
            });
        };
        let pieces = match &turn_sentinels.reasoning {
            Some(reasoning) => {
                let Some(start) = rest.find(reasoning.as_str()) else {
                    return Err(ChatTemplateError::DroppedTurn {
                        turn,
                        turns: sentinels.len(),
                    });
                };
                if start > content {
                    return Err(ChatTemplateError::ReasoningAfterContent { turn });
                }
                TurnPieces {
                    pre: rest[..start].to_owned(),
                    mid: Some(rest[start + reasoning.len()..content].to_owned()),
                }
            }
            None => TurnPieces {
                pre: rest[..content].to_owned(),
                mid: None,
            },
        };
        turns.push(pieces);
        rest = &rest[content + turn_sentinels.content.len()..];
    }
    Ok(Framing {
        turns,
        tail: rest.to_owned(),
    })
}

/// One entry of a tool catalog.
///
/// Read in the OpenAI function shape, `{"type": "function", "function": {name,
/// description, parameters}}`, or in the flat one HF datasets use; always
/// written in the first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ToolWire", into = "OpenAiTool")]
pub struct TemplateTool {
    pub name: String,
    pub description: String,
    /// The JSON Schema of the arguments.
    pub parameters: Value,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionWire {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "no_parameters")]
    parameters: Value,
}

fn no_parameters() -> Value {
    json!({"type": "object", "properties": {}})
}

#[derive(Serialize)]
struct OpenAiTool {
    r#type: &'static str,
    function: FunctionWire,
}

impl From<TemplateTool> for OpenAiTool {
    fn from(tool: TemplateTool) -> Self {
        Self {
            r#type: "function",
            function: FunctionWire {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolWire {
    r#type: Option<String>,
    function: Option<FunctionWire>,
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
}

impl TryFrom<ToolWire> for TemplateTool {
    type Error = String;

    fn try_from(wire: ToolWire) -> std::result::Result<Self, String> {
        if let Some(kind) = wire.r#type.as_deref()
            && kind != "function"
        {
            return Err(format!(
                "unsupported tool type '{kind}', expected 'function'"
            ));
        }
        match wire.function {
            Some(function) => {
                if wire.name.is_some() || wire.description.is_some() || wire.parameters.is_some() {
                    return Err(
                        "a tool is either {type, function} or {name, description, parameters}, \
                         not both"
                            .into(),
                    );
                }
                Ok(Self {
                    name: function.name,
                    description: function.description,
                    parameters: function.parameters,
                })
            }
            None => Ok(Self {
                name: wire
                    .name
                    .ok_or("a tool needs a name, or a function object carrying one")?,
                description: wire.description.unwrap_or_default(),
                parameters: wire.parameters.unwrap_or_else(no_parameters),
            }),
        }
    }
}

/// Serializes the tool catalog in the OpenAI function shape, the one HF chat
/// templates iterate over.
pub fn template_tools(tools: &[TemplateTool]) -> Result<String> {
    let rendered = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&rendered).map_err(|source| ChatTemplateError::Serialize {
        what: "tool definitions",
        source,
    })
}

/// How a conversation's tools reach the model. The payload each side needs - a
/// parser, the catalog text - stays with that side; this is the decision alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolRenderingKind {
    /// The template renders the catalog and frames the observations in its own
    /// `tool` role; the parser derived from it reads the calls back.
    Native,
    /// The catalog is described in the system turn and the calls are read back
    /// in the `<tool_call>` convention of [`parse_hermes`].
    Prompt,
    /// No tools at all.
    #[default]
    None,
}

/// The rendering a catalog of `declared` tools gets.
///
/// Native only when the template renders a catalog *and* a parser for its
/// format could be derived: rendering natively without a way to read the
/// answer is the asymmetry the fallback exists to prevent, so both halves fall
/// back together.
pub fn decide_rendering(
    declared: usize,
    supports_native: bool,
    has_parser: bool,
) -> ToolRenderingKind {
    match (declared, supports_native && has_parser) {
        (0, _) => ToolRenderingKind::None,
        (_, true) => ToolRenderingKind::Native,
        (_, false) => ToolRenderingKind::Prompt,
    }
}

/// How one catalog entry is spelled in the prompt-described convention. Field
/// order is the text the model reads, so it is a struct and not a map.
#[derive(Serialize)]
struct PromptTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a Value,
}

/// The fallback rendering's catalog: the tools, sorted by name, in the
/// `<tool_call>` convention [`parse_hermes`] reads back.
pub fn prompt_tool_instructions(tools: &[TemplateTool]) -> Result<String> {
    let mut canonical = tools
        .iter()
        .map(|tool| PromptTool {
            name: &tool.name,
            description: &tool.description,
            input_schema: &tool.parameters,
        })
        .collect::<Vec<_>>();
    canonical.sort_by(|a, b| a.name.cmp(b.name));
    let definitions =
        serde_json::to_string(&canonical).map_err(|source| ChatTemplateError::Serialize {
            what: "tool definitions",
            source,
        })?;
    Ok(format!(
        "Available tools (JSON): {definitions}\n\
         Call a tool with <tool_call>{{\"name\":\"tool_name\",\"arguments\":{{...}}}}</tool_call>."
    ))
}

/// A message list the prompt-described catalog can be written into.
pub trait SystemTurn: Sized {
    /// The content of this message when it is a system turn.
    fn system_content(&mut self) -> Option<&mut String>;
    fn new_system(content: String) -> Self;
}

/// Writes the catalog into the first system turn, after a blank line, or into a
/// system turn of its own at the head of the conversation.
pub fn inject_tool_instructions<M: SystemTurn>(messages: &mut Vec<M>, instructions: &str) {
    if let Some(system) = messages.iter_mut().find_map(SystemTurn::system_content) {
        system.push_str("\n\n");
        system.push_str(instructions);
    } else {
        messages.insert(0, M::new_system(instructions.to_owned()));
    }
}

/// The text an observation is rendered as.
///
/// Under native rendering the template has a `tool` role of its own and the call
/// id travels as `tool_call_id`, so the content is the tool's output and nothing
/// else. Without it, the same information has to survive inside free text, which
/// is what the `id: content` shape is for. The error marker stays in the content
/// either way: no chat template has a concept of a failed tool result, and the
/// policy needs to read the failure to react to it.
pub fn observation_text(
    kind: ToolRenderingKind,
    call_id: &str,
    content: &str,
    is_error: bool,
) -> String {
    let error_marker = if is_error { "ERROR: " } else { "" };
    match kind {
        ToolRenderingKind::Native => format!("{error_marker}{content}"),
        ToolRenderingKind::Prompt | ToolRenderingKind::None => {
            format!("{call_id}: {error_marker}{content}")
        }
    }
}

/// The exact inverse of [`observation_text`]: the tool's own output, or `None`
/// when `text` is not something it wrote for this call.
pub fn observation_payload(
    kind: ToolRenderingKind,
    call_id: &str,
    text: &str,
    is_error: bool,
) -> Option<String> {
    let text = match kind {
        ToolRenderingKind::Native => text,
        ToolRenderingKind::Prompt | ToolRenderingKind::None => text
            .strip_prefix(call_id)
            .and_then(|rest| rest.strip_prefix(": "))?,
    };
    let text = match is_error {
        true => text.strip_prefix("ERROR: ")?,
        false => text,
    };
    Some(text.to_owned())
}

/// What a parser read out of an assistant turn: the prose, the reasoning block
/// the template's own parser split from it, the calls, and each call it could
/// not read with its position among the turn's calls.
#[derive(Debug, Default)]
pub struct ParsedCalls {
    pub content: String,
    /// The text of the template's reasoning block, without its tags. Empty for
    /// a turn that did not reason, and for a parser - Hermes - that knows no
    /// such block.
    pub reasoning: String,
    pub calls: Vec<TemplateCall>,
    pub errors: Vec<(usize, ToolCallParseError)>,
}

/// Why one tool call could not be read.
///
/// The message is what the policy reads back as an observation, so it is part
/// of the training data: a variant's wording does not change without a reason.
#[derive(Debug, thiserror::Error)]
pub enum ToolCallParseError {
    #[error("unterminated <tool_call> block")]
    UnterminatedToolCall,
    #[error("invalid tool-call JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("invalid tool-call arguments: {0}")]
    InvalidArguments(#[source] serde_json::Error),
    #[error("tool call must be a JSON object")]
    NotAnObject,
    #[error("tool call requires a non-empty string name")]
    MissingName,
    #[error("tool call arguments must be a JSON object")]
    ArgumentsNotAnObject,
    #[error("tool call must be a JSON object or a <function=name> block")]
    UnknownForm,
    #[error("unterminated <function= tag")]
    UnterminatedFunctionTag,
    #[error("tool call requires a non-empty function name")]
    MissingFunctionName,
    #[error("unterminated <function> block")]
    UnterminatedFunctionBlock,
    #[error("unterminated <parameter= tag")]
    UnterminatedParameterTag,
    #[error("tool call parameter requires a non-empty name")]
    MissingParameterName,
    #[error("unterminated <parameter={0}> block")]
    UnterminatedParameterBlock(String),
}

const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";

/// Reads the `<tool_call>...</tool_call>` conventions out of free text.
///
/// Two bodies are accepted, because the family split one: the original
/// Hermes/Qwen2.5 form is a JSON object, while Qwen3's template instructs the
/// model to nest `<function=name>` and `<parameter=name>` tags instead. Which
/// one a model emits is decided by its own template, not by configuration, so
/// reading only one of them turns every tool call of the other family into a
/// parse error.
pub fn parse_hermes(output: &str) -> ParsedCalls {
    let mut parsed = ParsedCalls::default();
    let mut remainder = output;
    let mut call_index = 0_usize;
    while let Some(open) = remainder.find(HERMES_OPEN) {
        parsed.content.push_str(&remainder[..open]);
        let body_start = open + HERMES_OPEN.len();
        let Some(relative_close) = remainder[body_start..].find(HERMES_CLOSE) else {
            parsed
                .errors
                .push((call_index, ToolCallParseError::UnterminatedToolCall));
            parsed.content.push_str(&remainder[open..]);
            remainder = "";
            break;
        };
        let close = body_start + relative_close;
        let body = remainder[body_start..close].trim();
        match parse_call(body, call_index) {
            Ok(call) => parsed.calls.push(call),
            Err(error) => parsed.errors.push((call_index, error)),
        }
        call_index += 1;
        remainder = &remainder[close + HERMES_CLOSE.len()..];
    }
    parsed.content.push_str(remainder);
    parsed.content = parsed.content.trim().to_owned();
    parsed
}

/// How one call is spelled in the `<tool_call>` JSON body. A struct rather than
/// a map, so `name` comes first - the order the prompt tells the model to use.
#[derive(Serialize)]
struct HermesBody<'a> {
    name: &'a str,
    arguments: &'a Value,
}

/// Writes calls in the JSON form of the `<tool_call>` convention, one block per
/// call joined by newlines. The inverse of [`parse_hermes`] for the calls'
/// names and arguments; ids are positional on the way back.
pub fn hermes_call_text(calls: &[TemplateCall]) -> Result<String> {
    let mut blocks = Vec::with_capacity(calls.len());
    for call in calls {
        let body = serde_json::to_string(&HermesBody {
            name: &call.name,
            arguments: &call.arguments,
        })
        .map_err(|source| ChatTemplateError::Serialize {
            what: "tool call",
            source,
        })?;
        // A JSON string may hold the closing tag verbatim, and the reader stops
        // at the first one it finds.
        if body.contains(HERMES_CLOSE) {
            return Err(ChatTemplateError::UnwritableCall {
                name: call.name.clone(),
            });
        }
        blocks.push(format!("{HERMES_OPEN}{body}{HERMES_CLOSE}"));
    }
    Ok(blocks.join("\n"))
}

fn parse_call(body: &str, index: usize) -> std::result::Result<TemplateCall, ToolCallParseError> {
    match body.starts_with('{') {
        true => parse_json_call(body, index),
        false => parse_tagged_call(body, index),
    }
}

fn parse_json_call(
    body: &str,
    index: usize,
) -> std::result::Result<TemplateCall, ToolCallParseError> {
    let value: Value = serde_json::from_str(body).map_err(ToolCallParseError::InvalidJson)?;
    let object = value.as_object().ok_or(ToolCallParseError::NotAnObject)?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or(ToolCallParseError::MissingName)?;
    let arguments = object
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    if !arguments.is_object() {
        return Err(ToolCallParseError::ArgumentsNotAnObject);
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| positional_id(index));
    Ok(TemplateCall {
        id,
        name: name.to_owned(),
        arguments,
    })
}

/// Reads the tag form: `<function=name>` wrapping one `<parameter=name>` block
/// per argument. The call carries no id of its own, so it gets the positional
/// one the JSON form falls back to.
fn parse_tagged_call(
    body: &str,
    index: usize,
) -> std::result::Result<TemplateCall, ToolCallParseError> {
    let after = body
        .strip_prefix("<function=")
        .ok_or(ToolCallParseError::UnknownForm)?;
    let (name, body) = after
        .split_once('>')
        .ok_or(ToolCallParseError::UnterminatedFunctionTag)?;
    if name.is_empty() {
        return Err(ToolCallParseError::MissingFunctionName);
    }
    let body = body
        .trim_end()
        .strip_suffix("</function>")
        .ok_or(ToolCallParseError::UnterminatedFunctionBlock)?;

    let mut arguments = Map::new();
    let mut remainder = body;
    while let Some(open) = remainder.find("<parameter=") {
        let after = &remainder[open + "<parameter=".len()..];
        let (argument, after) = after
            .split_once('>')
            .ok_or(ToolCallParseError::UnterminatedParameterTag)?;
        if argument.is_empty() {
            return Err(ToolCallParseError::MissingParameterName);
        }
        let (value, after) = after
            .split_once("</parameter>")
            .ok_or_else(|| ToolCallParseError::UnterminatedParameterBlock(argument.to_owned()))?;
        arguments.insert(argument.to_owned(), parameter_value(value));
        remainder = after;
    }
    Ok(TemplateCall {
        id: positional_id(index),
        name: name.to_owned(),
        arguments: Value::Object(arguments),
    })
}

/// Recovers a parameter's type the way the template encoded it: an object or an
/// array was written as JSON, and everything else - including a number or a
/// boolean - was stringified. Inverting more than that would turn a path named
/// `123` into an integer.
fn parameter_value(raw: &str) -> Value {
    // The template puts the value on its own lines, so exactly one newline on
    // each side belongs to the framing rather than to the value.
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    let trimmed = value.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(json) = serde_json::from_str(trimmed)
    {
        return json;
    }
    Value::String(value.to_owned())
}

/// The id a call gets when its format carries none. Shared by every reader, so
/// a tool result can be matched to its call whichever parser produced it.
pub fn positional_id(index: usize) -> String {
    format!("call_{index}")
}

/// Why the runtime's parsed-assistant document could not be read at all - a
/// runtime defect, not a model output.
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    #[error("invalid parsed-assistant document: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("parsed assistant must be a JSON object")]
    NotAnObject,
    #[error("parsed assistant must carry a tool_calls array")]
    NoToolCalls,
}

/// Reads the document a template-derived parser produces,
/// `{"content", "reasoning_content", "tool_calls": [{"id", "name", "arguments"}]}`
/// with `arguments` a JSON string.
///
/// Only a malformed *document* is an `Err` here - that would be a runtime bug,
/// not a model output. A malformed `arguments` string is the model's doing and
/// becomes one entry of [`ParsedCalls::errors`], leaving the other calls of the
/// same turn usable.
pub fn decode_parsed_assistant(document: &str) -> std::result::Result<ParsedCalls, DocumentError> {
    let value: Value = serde_json::from_str(document).map_err(DocumentError::InvalidJson)?;
    let object = value.as_object().ok_or(DocumentError::NotAnObject)?;
    let field = |name: &str| {
        object
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    // The runtime splits the reasoning block out with the tags llama.cpp
    // derived from the template; nothing here knows what they are.
    let mut parsed = ParsedCalls {
        content: field("content"),
        reasoning: field("reasoning_content"),
        ..Default::default()
    };
    let calls = object
        .get("tool_calls")
        .and_then(Value::as_array)
        .ok_or(DocumentError::NoToolCalls)?;
    for (index, call) in calls.iter().enumerate() {
        match decode_call(call, index) {
            Ok(call) => parsed.calls.push(call),
            Err(error) => parsed.errors.push((index, error)),
        }
    }
    Ok(parsed)
}

fn decode_call(
    call: &Value,
    index: usize,
) -> std::result::Result<TemplateCall, ToolCallParseError> {
    let object = call.as_object().ok_or(ToolCallParseError::NotAnObject)?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or(ToolCallParseError::MissingName)?;
    let raw = object
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    // An absent argument list is an empty one: a template may render a
    // no-argument call with nothing between its markers.
    let arguments = match raw.is_empty() {
        true => Value::Object(Default::default()),
        false => serde_json::from_str(raw).map_err(ToolCallParseError::InvalidArguments)?,
    };
    if !arguments.is_object() {
        return Err(ToolCallParseError::ArgumentsNotAnObject);
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| positional_id(index));
    Ok(TemplateCall {
        id,
        name: name.to_owned(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> TemplateCall {
        TemplateCall {
            id: String::new(),
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn an_observation_reads_back_as_the_tool_output_it_was_written_from() {
        for kind in [
            ToolRenderingKind::Native,
            ToolRenderingKind::Prompt,
            ToolRenderingKind::None,
        ] {
            for is_error in [false, true] {
                // Content that itself looks like the framing must come back
                // untouched: only the one prefix the writer added is removed.
                for content in ["plain", "", "ERROR: already", "call_0: echoed", "call_0: "] {
                    let text = observation_text(kind, "call_0", content, is_error);
                    assert_eq!(
                        observation_payload(kind, "call_0", &text, is_error).as_deref(),
                        Some(content),
                        "{kind:?} is_error={is_error} {content:?} -> {text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn text_the_writer_did_not_produce_has_no_payload() {
        assert_eq!(
            observation_payload(ToolRenderingKind::Prompt, "call_0", "call_1: x", false),
            None
        );
        assert_eq!(
            observation_payload(ToolRenderingKind::Native, "call_0", "boom", true),
            None
        );
    }

    #[test]
    fn written_calls_read_back_with_and_without_prose() {
        let calls = vec![
            call("run", json!({"cmd": "pytest -q", "nested": {"a": [1, 2]}})),
            call("submit", json!({})),
        ];
        let text = hermes_call_text(&calls).unwrap();
        for turn in [text.clone(), format!("Let me look.\n{text}")] {
            let parsed = parse_hermes(&turn);
            assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
            assert_eq!(
                parsed
                    .calls
                    .iter()
                    .map(|call| (call.name.as_str(), &call.arguments))
                    .collect::<Vec<_>>(),
                calls
                    .iter()
                    .map(|call| (call.name.as_str(), &call.arguments))
                    .collect::<Vec<_>>()
            );
            assert_eq!(parsed.calls[1].id, "call_1", "ids are positional");
        }
        assert_eq!(
            parse_hermes(&format!("Let me look.\n{text}")).content,
            "Let me look."
        );
    }

    #[test]
    fn a_call_that_would_not_read_back_is_refused_rather_than_written() {
        let error =
            hermes_call_text(&[call("echo", json!({"text": "a </tool_call> b"}))]).unwrap_err();
        assert!(
            matches!(error, ChatTemplateError::UnwritableCall { .. }),
            "{error}"
        );
        // The opening tag alone does not end a block and reads back fine.
        let text = hermes_call_text(&[call("echo", json!({"text": "<tool_call>"}))]).unwrap();
        assert_eq!(
            parse_hermes(&text).calls[0].arguments["text"],
            "<tool_call>"
        );
    }

    #[test]
    fn the_catalog_reads_both_shapes_and_writes_the_openai_one() {
        let openai: TemplateTool = serde_json::from_value(json!({
            "type": "function",
            "function": {"name": "run", "description": "d", "parameters": {"type": "object"}},
        }))
        .unwrap();
        let flat: TemplateTool = serde_json::from_value(json!({
            "name": "run", "description": "d", "parameters": {"type": "object"},
        }))
        .unwrap();
        assert_eq!(openai, flat);
        assert_eq!(
            serde_json::to_value(&flat).unwrap(),
            json!({
                "type": "function",
                "function": {"name": "run", "description": "d", "parameters": {"type": "object"}},
            })
        );

        for invalid in [
            json!({"type": "retrieval", "function": {"name": "run"}}),
            json!({"description": "no name"}),
            json!({"function": {"name": "run"}, "name": "run"}),
            json!({"name": "run", "extra": 1}),
        ] {
            assert!(
                serde_json::from_value::<TemplateTool>(invalid.clone()).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn the_rendering_falls_back_to_the_prompt_when_either_native_half_is_missing() {
        assert_eq!(decide_rendering(0, true, true), ToolRenderingKind::None);
        assert_eq!(decide_rendering(2, true, true), ToolRenderingKind::Native);
        assert_eq!(decide_rendering(2, true, false), ToolRenderingKind::Prompt);
        assert_eq!(decide_rendering(2, false, true), ToolRenderingKind::Prompt);
    }

    fn sentinels(reasoning: bool) -> TurnSentinels {
        TurnSentinels {
            reasoning: reasoning.then(|| reasoning_span(0)),
            content: assistant_span(0),
        }
    }

    #[test]
    fn a_turn_that_reasons_is_handed_over_as_two_sentinels() {
        let messages = [
            TemplateMessage::text("user", "go"),
            TemplateMessage {
                reasoning: Some("left looks open"),
                ..TemplateMessage::text("assistant", "left")
            },
            TemplateMessage {
                reasoning: Some(""),
                ..TemplateMessage::text("assistant", "up")
            },
        ];
        let (json, sentinels) = template_messages(&messages).unwrap();
        let json: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(json[1]["reasoning_content"], reasoning_span(0));
        assert_eq!(json[1]["content"], assistant_span(0));
        assert!(
            json[2].get("reasoning_content").is_none(),
            "empty reasoning is no reasoning"
        );
        assert_eq!(
            sentinels,
            [
                TurnSentinels {
                    reasoning: Some(reasoning_span(0)),
                    content: assistant_span(0),
                },
                TurnSentinels {
                    reasoning: None,
                    content: assistant_span(1),
                },
            ]
        );
    }

    #[test]
    fn a_turn_is_cut_before_its_reasoning_between_it_and_its_content_and_after() {
        let (rs, cs) = (reasoning_span(0), assistant_span(0));
        let framing = split_turns(
            &format!("<a><think>{rs}</think>\n{cs}</a><b>"),
            &[sentinels(true)],
        )
        .unwrap();
        assert_eq!(
            framing,
            Framing {
                turns: vec![TurnPieces {
                    pre: "<a><think>".into(),
                    mid: Some("</think>\n".into()),
                }],
                tail: "</a><b>".into(),
            }
        );

        let framing = split_turns(&format!("<a>{cs}</a><b>"), &[sentinels(false)]).unwrap();
        assert_eq!(
            framing.turns,
            [TurnPieces {
                pre: "<a>".into(),
                mid: None,
            }]
        );
        assert_eq!(framing.tail, "</a><b>");

        let empty = split_turns("<a>", &[]).unwrap();
        assert!(empty.turns.is_empty());
        assert_eq!(empty.tail, "<a>");
    }

    #[test]
    fn a_template_that_drops_repeats_or_moves_the_reasoning_is_refused() {
        let (rs, cs) = (reasoning_span(0), assistant_span(0));
        let error = split_turns(&format!("<a>{cs}</a>"), &[sentinels(true)]).unwrap_err();
        assert!(
            matches!(error, ChatTemplateError::DroppedReasoning { turn: 0 }),
            "{error}"
        );
        assert!(error.to_string().contains("drops the reasoning"), "{error}");

        let error = split_turns(&format!("<a>{rs}{rs}{cs}</a>"), &[sentinels(true)]).unwrap_err();
        assert!(
            matches!(
                error,
                ChatTemplateError::RepeatedSentinel {
                    turn: 0,
                    part: SpanPart::Reasoning
                }
            ),
            "{error}"
        );
        let error = split_turns(&format!("<a>{cs}{cs}</a>"), &[sentinels(false)]).unwrap_err();
        assert!(
            matches!(
                error,
                ChatTemplateError::RepeatedSentinel {
                    turn: 0,
                    part: SpanPart::Content
                }
            ),
            "{error}"
        );

        let error = split_turns(&format!("<a>{cs}{rs}</a>"), &[sentinels(true)]).unwrap_err();
        assert!(
            matches!(error, ChatTemplateError::ReasoningAfterContent { turn: 0 }),
            "{error}"
        );
    }

    #[test]
    fn turns_are_cut_in_order() {
        let both = [
            sentinels(false),
            TurnSentinels {
                reasoning: Some(reasoning_span(1)),
                content: assistant_span(1),
            },
        ];
        let (cs0, rs1, cs1) = (assistant_span(0), reasoning_span(1), assistant_span(1));
        let framing = split_turns(&format!("a{cs0}b{rs1}c{cs1}d"), &both).unwrap();
        assert_eq!(framing.turns[1].pre, "b");
        assert_eq!(framing.turns[1].mid.as_deref(), Some("c"));
        assert_eq!(framing.tail, "d");

        let error = split_turns(&format!("a{cs1}b{cs0}c{rs1}"), &both).unwrap_err();
        assert!(
            matches!(error, ChatTemplateError::DroppedTurn { turn: 1, turns: 2 }),
            "{error}"
        );
    }

    #[test]
    fn a_revealed_turn_carries_its_calls_and_the_others_stay_sentinels() {
        let calls = [TemplateCall {
            id: "call_0".into(),
            name: "run".into(),
            arguments: json!({"cmd": "ls"}),
        }];
        let messages = [
            TemplateMessage::text("user", "go"),
            TemplateMessage {
                tool_calls: &calls,
                ..TemplateMessage::text("assistant", "looking")
            },
            TemplateMessage {
                tool_call_id: Some("call_0"),
                ..TemplateMessage::text("tool", "a b")
            },
            TemplateMessage::text("assistant", "done"),
        ];
        let (hidden, sentinels) = template_messages(&messages).unwrap();
        let hidden: Value = serde_json::from_str(&hidden).unwrap();
        assert!(hidden[1].get("tool_calls").is_none());
        assert_eq!(hidden[1]["content"], sentinels[0].content);

        let revealed: Value =
            serde_json::from_str(&template_messages_revealing(&messages, 0).unwrap()).unwrap();
        assert_eq!(revealed[1]["content"], "looking");
        assert_eq!(revealed[1]["tool_calls"][0]["function"]["name"], "run");
        assert_eq!(
            revealed[1]["tool_calls"][0]["function"]["arguments"],
            json!({"cmd": "ls"}),
            "arguments go to the template as an object"
        );
        assert_eq!(
            revealed[3]["content"], sentinels[1].content,
            "the other turns keep the sentinel they have in the hidden form"
        );
    }
}
