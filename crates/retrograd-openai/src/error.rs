//! The error document of the OpenAI contract.
//!
//! One type, no variants: this is a wire document, not a domain error. What a
//! client branches on is the triple `status` / `type` / `code`, and each named
//! constructor below fixes that triple once, so two call sites cannot spell the
//! same failure two ways.

use axum::response::{IntoResponse, Response};
use http::{HeaderValue, StatusCode, header};
use serde::Serialize;

retrograd_core::wire_enum! {
    /// The `error.type` of the envelope. The spellings are OpenAI's where it has
    /// one; `server_busy` and `timeout` are the two a local server needs and the
    /// public API has no word for, and `not_supported_error` is llama-server's,
    /// for a model that cannot answer a route at all.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ErrorType {
        InvalidRequest = "invalid_request_error",
        Authentication = "authentication_error",
        NotFound = "not_found_error",
        ServerBusy = "server_busy",
        Timeout = "timeout",
        NotSupported = "not_supported_error",
        Api = "api_error",
    }
}

/// Marks a response as an OpenAI error envelope, so a layer that rewrites every
/// other error into its own document leaves this one alone.
#[derive(Clone, Copy, Debug)]
pub struct OpenAiEnvelope;

/// Gives routing and extractor rejections the same envelope as handler errors.
/// Attach this outside any body-limit layer supplied by the frontend as well.
pub fn normalize_response(response: Response) -> Response {
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error())
        || response.extensions().get::<OpenAiEnvelope>().is_some()
    {
        return response;
    }
    let detail = match status {
        StatusCode::PAYLOAD_TOO_LARGE => "the request body is larger than this server accepts",
        StatusCode::METHOD_NOT_ALLOWED => "that method is not allowed on this path",
        StatusCode::NOT_FOUND => "no route or resource exists at this path",
        _ => "the request could not be served",
    };
    let error = if status.is_client_error() {
        OpenAiError::invalid(detail)
    } else {
        OpenAiError::internal(detail)
    };
    let mut normalized = error.with_status(status).into_response();
    // Method routing supplies Allow; capacity middleware may supply Retry-After.
    for name in [header::ALLOW, header::RETRY_AFTER, header::WWW_AUTHENTICATE] {
        if let Some(value) = response.headers().get(&name) {
            normalized.headers_mut().insert(name, value.clone());
        }
    }
    normalized
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct OpenAiError {
    pub status: StatusCode,
    pub kind: ErrorType,
    pub message: String,
    /// The request field at fault, in the dotted form OpenAI uses:
    /// `messages[2].content`, `max_tokens`.
    pub param: Option<String>,
    pub code: Option<&'static str>,
    /// Seconds, for the `Retry-After` of a refusal that is only a matter of
    /// waiting.
    pub retry_after: Option<u64>,
}

impl OpenAiError {
    fn new(status: StatusCode, kind: ErrorType, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            param: None,
            code: None,
            retry_after: None,
        }
    }

    /// A request that is malformed or out of range. 400 and not 422: the SDKs
    /// retry a 5xx and give up on a 400, which is the behaviour a bad request
    /// wants.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, ErrorType::InvalidRequest, message)
    }

    /// [`Self::invalid`], located on one field.
    pub fn invalid_param(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::invalid(message).with_param(param)
    }

    /// A parameter this server understands and will not honour: one that would
    /// change what is sampled, or a feature it does not implement.
    pub fn unsupported(param: impl Into<String>, message: impl Into<String>) -> Self {
        let mut error = Self::invalid_param(param, message);
        error.code = Some("unsupported_parameter");
        error
    }

    /// A content part this server cannot hand to a text model.
    pub fn unsupported_content(param: impl Into<String>, message: impl Into<String>) -> Self {
        let mut error = Self::invalid_param(param, message);
        error.code = Some("unsupported_content");
        error
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, ErrorType::NotFound, message)
    }

    pub fn model_not_found(model: &str, detail: impl std::fmt::Display) -> Self {
        let mut error = Self::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            format!("the model `{model}` does not exist: {detail}"),
        )
        .with_param("model");
        error.code = Some("model_not_found");
        error
    }

    /// A stored response id that is unknown, evicted, or owned by someone
    /// else. 404 either way: which of the three it was is not information the
    /// caller is entitled to.
    pub fn response_not_found(id: &str, detail: impl std::fmt::Display) -> Self {
        let mut error = Self::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            format!("the response `{id}` does not exist: {detail}"),
        );
        error.code = Some("response_not_found");
        error
    }

    /// The code lm-eval and Open WebUI recognise, raised before anything is
    /// sampled.
    pub fn context_length_exceeded(prompt: usize, requested: u32, context: usize) -> Self {
        let mut error = Self::invalid(format!(
            "this model's context is {context} tokens, and the request asks for {prompt} \
             prompt tokens plus {requested} completion tokens"
        ))
        .with_param("messages");
        error.code = Some("context_length_exceeded");
        error
    }

    /// The model exists and cannot answer in the state it is in.
    pub fn conflict(message: impl Into<String>) -> Self {
        let mut error = Self::new(StatusCode::CONFLICT, ErrorType::InvalidRequest, message);
        error.code = Some("model_unavailable");
        error
    }

    /// Refused for now, served later: the device or the queue is taken.
    pub fn busy(code: &'static str, message: impl Into<String>, retry_after: u64) -> Self {
        let mut error = Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::ServerBusy,
            message,
        );
        error.code = Some(code);
        error.retry_after = Some(retry_after);
        error
    }

    /// The model cannot answer this route, whatever the request says: a model
    /// that is not a decision model asked a decision, a decision model asked
    /// to generate. 501, with llama-server's `not_supported_error`, so a client
    /// written for that server branches the same way here.
    pub fn not_supported(message: impl Into<String>) -> Self {
        let mut error = Self::new(
            StatusCode::NOT_IMPLEMENTED,
            ErrorType::NotSupported,
            message,
        );
        error.code = Some("not_supported");
        error
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, ErrorType::Timeout, message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        let mut error = Self::new(StatusCode::UNAUTHORIZED, ErrorType::Authentication, message);
        error.code = Some("invalid_api_key");
        error
    }

    /// A route this server has, switched off by its operator.
    pub fn disabled() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            "the OpenAI-compatible endpoint is disabled on this server",
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, ErrorType::Api, message)
    }

    /// Replaces the status, keeping the rest - for a translation that knows the
    /// status better than the constructor it went through.
    pub fn with_status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// The `error` object alone, for a stream that already sent its status.
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "message": self.message,
            "type": self.kind.as_str(),
            "param": self.param,
            "code": self.code,
        })
    }
}

/// What the caller can fix by sending something else is theirs (400); every
/// other failure is the server's (500).
impl From<retrograd_core::Error> for OpenAiError {
    fn from(error: retrograd_core::Error) -> Self {
        if error.is_user_error() {
            Self::invalid(error.to_string())
        } else {
            Self::internal(error.to_string())
        }
    }
}

/// The agent-core errors reach this one through the applicative façade, which
/// already decides what is the caller's fault.
impl From<retrograd_agent_core::Error> for OpenAiError {
    fn from(error: retrograd_agent_core::Error) -> Self {
        retrograd_core::Error::from(error).into()
    }
}

/// The engine's errors reach this one through the applicative façade, except
/// a decision failure, which keeps the status only its own variant knows.
impl From<retrograd_engine::EngineError> for OpenAiError {
    fn from(error: retrograd_engine::EngineError) -> Self {
        match error {
            retrograd_engine::EngineError::Decision(error) => error.into(),
            error => retrograd_core::Error::from(error).into(),
        }
    }
}

/// A model that cannot answer a decision at all - no decision head, a head of
/// a type this server does not implement, images it has no place for - is a
/// 501, as llama-server answers it. Request validation keeps the protocol's
/// own message (400); other errors go through the applicative façade.
impl From<retrograd_decision::DecisionError> for OpenAiError {
    fn from(error: retrograd_decision::DecisionError) -> Self {
        use retrograd_decision::DecisionError;
        match error {
            DecisionError::InvalidRequest(message) => Self::invalid(message),
            DecisionError::NotDecisionModel
            | DecisionError::ImagesUnsupported
            | DecisionError::UnknownType { .. } => Self::not_supported(error.to_string()),
            error => retrograd_core::Error::from(error).into(),
        }
    }
}

/// The inference policy's errors, translated where the policy's structured
/// variants decide the status code: a full KV pool or admission queue is a
/// capacity condition (503 + `Retry-After`), never a caller mistake.
impl From<retrograd_infer::InferError> for OpenAiError {
    fn from(error: retrograd_infer::InferError) -> Self {
        use retrograd_infer::InferError;
        match error {
            InferError::KvFull { .. } => Self::busy("kv_full", error.to_string(), 1),
            InferError::QueueFull { .. } => Self::busy("queue_full", error.to_string(), 1),
            InferError::InvalidArgument(_) => Self::invalid(error.to_string()),
            InferError::Overflow(_) | InferError::Failed(_) | InferError::Cancelled => {
                Self::internal(error.to_string())
            }
            InferError::Backend(error) => error.into(),
            InferError::Core(error) => error.into(),
        }
    }
}

#[derive(Serialize)]
struct Envelope {
    error: serde_json::Value,
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            [(header::CONTENT_TYPE, "application/json")],
            axum::Json(Envelope { error: self.body() }),
        )
            .into_response();
        if let Some(seconds) = self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response.extensions_mut().insert(OpenAiEnvelope);
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_user_error_is_a_bad_request_and_the_rest_is_ours() {
        let user = OpenAiError::from(retrograd_core::Error::invalid("x"));
        assert_eq!(user.status, StatusCode::BAD_REQUEST);
        assert_eq!(user.kind, ErrorType::InvalidRequest);
        let ours = OpenAiError::from(retrograd_core::Error::runtime("x"));
        assert_eq!(ours.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(ours.kind, ErrorType::Api);
    }

    #[test]
    fn a_model_that_cannot_decide_is_not_implemented_and_a_bad_body_is_the_callers() {
        use retrograd_decision::DecisionError;
        for error in [
            DecisionError::NotDecisionModel,
            DecisionError::ImagesUnsupported,
            DecisionError::UnknownType {
                name: "other".into(),
            },
        ] {
            let error = OpenAiError::from(retrograd_engine::EngineError::from(error));
            assert_eq!(error.status, StatusCode::NOT_IMPLEMENTED);
            assert_eq!(error.kind, ErrorType::NotSupported);
        }
        let error = OpenAiError::from(DecisionError::InvalidRequest(
            "\"state\" must be provided".into(),
        ));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "\"state\" must be provided");
        let error = OpenAiError::from(DecisionError::Layout);
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn the_envelope_carries_the_triple_and_its_marker() {
        let response = OpenAiError::busy("device_busy", "held", 30).into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "30");
        assert!(response.extensions().get::<OpenAiEnvelope>().is_some());
    }
}
