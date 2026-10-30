//! The Embeddings API: `POST /v1/embeddings`.
//!
//! One prefill pass per input on the serving session's thread - no generation
//! slot is taken, and the KV is cleared around the extraction, so an embedding
//! never shares KV with a generation whose attention mode differs. Only
//! weights on disk answer (a live run's trainer has no embedding path); a
//! model that declares no pooling is refused rather than given an embedding
//! invented by averaging.

pub mod convert;
pub mod wire;

pub use convert::{Prepared, prepare};
pub use wire::{
    EmbeddingObject, EmbeddingUsage, EmbeddingValue, EmbeddingsRequest, EmbeddingsResponse,
    EncodingFormat,
};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;

use crate::error::OpenAiError;
use crate::{Endpoint, Target};

/// The embeddings routes, relative to wherever the caller mounts them - `/v1`
/// for every OpenAI client.
pub fn router() -> Router<Endpoint> {
    Router::new().route("/embeddings", post(create))
}

async fn create(State(endpoint): State<Endpoint>, body: Bytes) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    let request = serde_json::from_slice::<wire::EmbeddingsRequest>(&body).map_err(|error| {
        OpenAiError::invalid(format!("the body is not an embeddings request: {error}"))
    })?;
    let prepared = prepare(request)?;
    let target = endpoint.source.resolve(&prepared.model).await?;
    // Embeddings are prefill-only on the serving session; a live run's
    // trainer has no embedding path, so the request is refused, not narrowed.
    let Target::Cold { spec, .. } = target else {
        return Err(OpenAiError::unsupported(
            "embeddings",
            "embeddings are served by the weights on disk; a run that is training does not \
             answer them",
        ));
    };
    let outcome = tokio::time::timeout(
        endpoint.timeout,
        endpoint.session.embed(spec, prepared.inputs),
    )
    .await
    .map_err(|_| {
        OpenAiError::timeout(format!("no answer within {}s", endpoint.timeout.as_secs()))
    })??;
    let data: Vec<wire::EmbeddingObject> = outcome
        .embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| wire::EmbeddingObject {
            object: "embedding",
            // Bounded by MAX_EMBED_INPUTS: the narrowing is exact.
            index: index as u32,
            embedding: match prepared.encoding {
                wire::EncodingFormat::Float => wire::EmbeddingValue::Float(embedding.clone()),
                wire::EncodingFormat::Base64 => {
                    wire::EmbeddingValue::Base64(base64_encode(&f32_le_bytes(embedding)))
                }
            },
        })
        .collect();
    // Widening: token counts of one request.
    let prompt_tokens = outcome.prompt_tokens.iter().sum::<usize>() as u64;
    Ok(axum::Json(wire::EmbeddingsResponse {
        object: "list",
        data,
        model: prepared.model,
        usage: wire::EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
    .into_response())
}

/// The little-endian float32 bytes of a vector.
fn f32_le_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding - the `base64` encoding format of the
/// contract, over the little-endian float32 bytes of the vector.
pub(crate) fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_ALPHABET[(triple >> 18 & 63) as usize] as char);
        out.push(BASE64_ALPHABET[(triple >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            BASE64_ALPHABET[(triple >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64_ALPHABET[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_is_the_standard_alphabet_with_padding() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // One f32: 1.0f32 = 00 00 80 3F -> "AACAPw==".
        assert_eq!(base64_encode(&1.0_f32.to_le_bytes()), "AACAPw==");
    }
}
