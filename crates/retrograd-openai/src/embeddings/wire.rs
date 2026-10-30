//! The Embeddings API documents: `POST /v1/embeddings` and its `list` answer.
//! Read without `deny_unknown_fields`, like every OpenAI request here: what a
//! known field *means* is decided in `convert`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `POST /v1/embeddings`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    /// A string, a list of strings, a list of token ids, or a list of token
    /// id lists - parsed as raw values so a malformed element can be named.
    pub input: Option<Value>,
    /// `float` (the default) or `base64` (little-endian float32, documented).
    pub encoding_format: Option<String>,
    /// Declared only to be refused: the model's own embedding width is
    /// returned, truncating it would silently change the vectors.
    pub dimensions: Option<u64>,
    /// Cannot change a vector: dropped, as in the chat request.
    pub user: Option<String>,
}

/// How the vectors are encoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncodingFormat {
    #[default]
    Float,
    /// Base64 of the little-endian float32 bytes of the vector.
    Base64,
}

/// The answer: one `embedding` object per input, in order.
#[derive(Clone, Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Clone, Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: u32,
    pub embedding: EmbeddingValue,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}
