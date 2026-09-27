//! The documents of the contract, as the OpenAI clients write and read them.
//!
//! Requests are read without `deny_unknown_fields`, unlike the rest of
//! Retrograd's API: the SDKs send fields newer than this server, and refusing a
//! request over a field it does not know would break every client on each of
//! their releases. What a known field *means* is decided in `convert`, where a
//! field that would change what is sampled is refused rather than ignored.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `POST /v1/chat/completions`.
///
/// Every field a client may send `null` for is an `Option`: `null` and absent
/// mean the same thing to every OpenAI client.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub max_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<i64>,
    pub stop: Option<Stop>,
    pub n: Option<u64>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    // Read only to be refused, or accepted in their neutral form.
    pub logprobs: Option<Value>,
    pub top_logprobs: Option<Value>,
    pub logit_bias: Option<Value>,
    pub response_format: Option<Value>,
    pub audio: Option<Value>,
    pub modalities: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: Option<Content>,
    pub tool_calls: Option<Vec<WireToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WireToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub function: WireFunctionCall,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WireFunctionCall {
    pub name: String,
    /// A JSON document in a string, per the contract. Some clients send the
    /// object itself; both are read.
    pub arguments: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WireTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunction,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WireFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

/// The answer to a non-streamed request.
#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    /// `null`, not absent, when the turn is only calls: the clients read the
    /// key.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOut>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionOut,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionOut {
    pub name: String,
    /// Always a string: the JSON of the arguments, as the contract has it.
    pub arguments: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// One event of a streamed answer.
#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChunkChoice>,
    /// Only on the last chunk, and only when `stream_options.include_usage`
    /// asked for it - where it is `null` on every other chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(flatten)]
    pub call: ToolCallOut,
}

/// `GET /v1/models` and `GET /v1/models/{model}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// An extension: this id is served by a run that is training, so an answer
    /// waits for its next progress callback.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub live: bool,
}

impl ModelCard {
    pub fn new(id: impl Into<String>, created: u64) -> Self {
        Self {
            id: id.into(),
            object: "model",
            created,
            owned_by: "retrograd",
            live: false,
        }
    }

    pub fn live(mut self) -> Self {
        self.live = true;
        self
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}
