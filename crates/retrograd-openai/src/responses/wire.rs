//! The Responses API documents, as the OpenAI clients write and read them:
//! the `POST /v1/responses` request (model, input as a string or a list of
//! items, instructions, tools, text format, budgets, streaming and store
//! options) and the `response` object with its `output` items.
//!
//! The contract is pinned against the published OpenAPI description of the
//! Responses API (see the crate docs for the pinned revision): the request
//! fields and their bounds, the input item shapes that carry text only, the
//! response object's required keys, and the streamed event types with their
//! `sequence_number`. Anything the pinned schema offers but this server does
//! not serve - non-text content parts, hosted tools, background mode, hosted
//! conversations, prefix truncation - is declared here and refused with the
//! field named, never ignored. `store` and `previous_response_id` are served:
//! they go to the endpoint's store.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::wire::{TokenLogprobs, TopLogprob};

/// `POST /v1/responses`.
///
/// Read without `deny_unknown_fields`, like the chat request: SDKs send
/// fields newer than this server. The known fields that cannot be honoured
/// are refused in `convert`, with the field named.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    /// A string, or a list of input items (parsed as raw values so a refused
    /// item can be named by its `type`).
    pub input: Option<Input>,
    pub instructions: Option<String>,
    /// The flat tool form `{type: "function", name, description?, parameters?,
    /// strict?}`.
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub text: Option<TextParam>,
    pub max_output_tokens: Option<u64>,
    /// Declared only to be refused: limiting calls after the fact would answer
    /// with a turn the model did not write.
    pub max_tool_calls: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_logprobs: Option<u64>,
    pub reasoning: Option<ReasoningParam>,
    pub metadata: Option<serde_json::Map<String, Value>>,
    pub include: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub stream_options: Option<Value>,
    /// `true` (the API default) needs a store; this server serves the
    /// stateless path only, so anything but `false` is refused.
    pub store: Option<bool>,
    /// Declared only to be refused: KV pinning across responses is not served
    /// yet.
    pub truncation: Option<String>,
    /// The stored response this one continues.
    pub previous_response_id: Option<String>,
    /// Background (asynchronous) responses are out of perimeter.
    pub background: Option<bool>,
    /// Declared only to be refused: hosted conversations are out of perimeter.
    pub conversation: Option<Value>,
}

/// `input`: a string, or a list of items.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<Value>),
}

/// `text`: the format (and verbosity, refused as unsupported).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct TextParam {
    pub format: Option<Value>,
    pub verbosity: Option<String>,
}

/// `reasoning`: only `effort` is read; it maps to a local budget policy.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ReasoningParam {
    pub effort: Option<String>,
    pub summary: Option<String>,
    pub mode: Option<String>,
    pub context: Option<String>,
}

/// The `response` object, as the schema pins it: the required keys are always
/// present (null when empty), the echoes carry the effective options.
#[derive(Clone, Debug, Serialize)]
pub struct Response {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub status: Status,
    pub model: String,
    pub output: Vec<OutputItem>,
    /// The aggregated `output_text`, an SDK convenience.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
    pub usage: Usage,
    /// Always present; `null` on success.
    pub error: Option<ResponseError>,
    /// Always present; `null` unless the answer was cut.
    pub incomplete_details: Option<IncompleteDetails>,
    pub instructions: Option<String>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel_tool_calls: bool,
    pub metadata: Option<Value>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_logprobs: Option<u32>,
    pub max_output_tokens: Option<u64>,
    pub text: TextEcho,
    pub reasoning: Option<Value>,
    pub store: bool,
    pub background: bool,
    pub previous_response_id: Option<String>,
    pub truncation: &'static str,
    pub service_tier: &'static str,
    pub access_programs: Vec<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

/// One `output` item. The discriminating `type` is carried by the tag.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    /// A reasoning item. The local model's raw reasoning is exposed as
    /// `content` (`reasoning_text`), never as an OpenAI-style `summary`:
    /// summaries are a hosted-model capability, and passing raw reasoning off
    /// as one would be a lie. `summary` is therefore always empty.
    Reasoning {
        id: String,
        summary: Vec<SummaryPart>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        content: Vec<ReasoningTextPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<&'static str>,
    },
    Message {
        id: String,
        role: &'static str,
        status: &'static str,
        content: Vec<OutputContent>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct SummaryPart {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

/// The local extension: the model's raw reasoning text, tags removed.
#[derive(Clone, Debug, Serialize)]
pub struct ReasoningTextPart {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    OutputText {
        text: String,
        annotations: Vec<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<TokenLogprobs>>,
    },
}

/// The `text` echo of a response: the format that was in effect.
#[derive(Clone, Debug, Serialize)]
pub struct TextEcho {
    pub format: Value,
}

/// `usage`, with the details objects the schema requires. `cached_tokens` is
/// the number of input tokens copied from the prefix cache instead of
/// prefilled (a stored turn keeps its KV pinned for its continuation, as a
/// hint); `cache_write_tokens` stays 0, nothing is billed for writing it.
#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: u64,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

/// One `logprobs` entry of a streamed `response.output_text.delta`.
#[derive(Clone, Debug, Serialize)]
pub struct EventLogprob {
    pub token: String,
    #[serde(serialize_with = "crate::render::serialize_logprob")]
    pub logprob: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<EventTopLogprob>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EventTopLogprob {
    pub token: String,
    #[serde(serialize_with = "crate::render::serialize_logprob")]
    pub logprob: f32,
}

/// Re-exported so the events module spells the same shapes as the response.
pub type ResponseTokenLogprobs = TokenLogprobs;
pub type ResponseTopLogprob = TopLogprob;
