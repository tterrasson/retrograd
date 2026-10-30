//! The legacy Completions API documents: `POST /v1/completions` and its
//! `text_completion` answer. Read without `deny_unknown_fields`, like the chat
//! request: the SDKs send fields newer than this server, and what a known field
//! *means* is decided in `convert`, where a field that would change what is
//! sampled is refused with the field named.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::chat::wire::{FinishReason, Stop, Usage};

/// `POST /v1/completions`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// A string, or a list of strings (one prompt per element). Token arrays
    /// are refused in `convert`, with the field named.
    pub prompt: Option<Value>,
    /// Declared only to be refused: a suffix would be generated after the
    /// completion and joined to it, which this server does not do.
    pub suffix: Option<Value>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub seed: Option<i64>,
    pub stop: Option<Stop>,
    pub n: Option<u64>,
    pub stream: Option<bool>,
    /// The legacy form: an integer 0..=20, how many top logprobs each token
    /// carries. `0` asks for the sampled token's logprob without top pairs.
    pub logprobs: Option<Value>,
    /// Prepend the prompt to the returned text.
    pub echo: Option<bool>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub penalty_window: Option<u64>,
    pub logit_bias: Option<Value>,
    /// Declared only to be refused: `best_of` would generate more than `n`
    /// completions and pick one, which is a different answer than the `n` the
    /// client asked for.
    pub best_of: Option<u64>,
    /// Cannot change a token: dropped, as in the chat request.
    pub user: Option<String>,
}

/// The answer to a non-streamed request: one choice per prompt per `n`.
#[derive(Clone, Debug, Serialize)]
pub struct TextCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
pub struct TextChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LegacyLogprobs>,
    pub finish_reason: FinishReason,
}

/// The legacy logprobs shape: parallel arrays per sampled token, and
/// `top_logprobs[i]` a map of token to logprob.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LegacyLogprobs {
    pub tokens: Vec<String>,
    #[serde(serialize_with = "crate::render::serialize_logprob_vec")]
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<Map<String, Value>>,
    pub text_offset: Vec<usize>,
}

/// One event of a streamed answer. The legacy chunk carries the piece in
/// `choices[].text` (there is no `delta` in this object).
#[derive(Clone, Debug, Serialize)]
pub struct TextCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LegacyLogprobs>,
}
