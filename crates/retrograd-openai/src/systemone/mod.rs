//! Typed decisions: `POST /v1/systemone`, the contract of llama-server's
//! decision models.
//!
//! The body is `{state, questions, images?, files?}` plus the `model` id this
//! server routes on. It is read with the order of its keys kept, handed to
//! the serving session, validated there against the readout of the loaded
//! model, and answered in one forward pass per question - no generation
//! slot is taken. The answer and the error statuses are llama-server's: 400
//! for a body it would refuse, 501 for a model that is not a decision model
//! (or one whose head this server does not implement) and for images.
//!
//! Only weights on disk answer: a run that is training has no decision path.

pub mod wire;

pub use wire::{SystemOneResponse, SystemOneUsage};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use retrograd_decision::Json;

use crate::error::OpenAiError;
use crate::{Endpoint, Target};

/// The decision route, relative to wherever the caller mounts it - `/v1`, as
/// llama-server serves it.
pub fn router() -> Router<Endpoint> {
    Router::new().route("/systemone", post(create))
}

async fn create(State(endpoint): State<Endpoint>, body: Bytes) -> Result<Response, OpenAiError> {
    endpoint.check_enabled()?;
    // The message is llama-server's, for the same failure.
    let body: Json = serde_json::from_slice(&body)
        .map_err(|error| OpenAiError::invalid(format!("invalid JSON: {error}")))?;
    if !body.is_object() {
        return Err(OpenAiError::invalid("the body must be a JSON object"));
    }
    let model = model_id(&endpoint, &body).await?;
    let target = endpoint.source.resolve(&model).await?;
    let Target::Cold { spec, .. } = target else {
        return Err(OpenAiError::unsupported(
            "model",
            "decisions are served by the weights on disk; a run that is training does not \
             answer them",
        ));
    };
    let decision = tokio::time::timeout(endpoint.timeout, endpoint.session.decide(spec, body))
        .await
        .map_err(|_| {
            OpenAiError::timeout(format!("no answer within {}s", endpoint.timeout.as_secs()))
        })??;
    Ok(axum::Json(SystemOneResponse {
        model,
        answers: decision.answers,
        usage: SystemOneUsage {
            // Widening: a token count.
            input_tokens: decision.input_tokens as u64,
            output_tokens: 0,
        },
    })
    .into_response())
}

/// The id the request names. llama-server serves one model and reads no
/// `model` field, so a body without one is served when this server has one
/// model to offer; with several, the id is required.
async fn model_id(endpoint: &Endpoint, body: &Json) -> Result<String, OpenAiError> {
    match body.get("model") {
        Some(Json::String(model)) if !model.trim().is_empty() => Ok(model.clone()),
        None | Some(Json::Null) => {
            let mut cards = endpoint.source.list().await?;
            match cards.len() {
                1 => Ok(cards.remove(0).id),
                _ => Err(OpenAiError::invalid_param(
                    "model",
                    "`model` is required: this server serves several models",
                )),
            }
        }
        Some(_) => Err(OpenAiError::invalid_param(
            "model",
            "`model` must be a non-empty string",
        )),
    }
}
