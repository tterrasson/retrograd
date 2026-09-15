//! `ApiError` and its rendering as RFC 9457 `application/problem+json`.
//!
//! One error type for the whole API, with a closed set of `type` URIs. The URI
//! is the part of the contract a client tests against; `title` and `detail` are
//! prose and may be reworded.

use axum::response::{IntoResponse, Response};
use http::{StatusCode, header};
use serde::Serialize;

/// Base of the problem-type URIs. Not a resolvable URL today, but a stable
/// namespace, which is what RFC 9457 asks of `type`.
const PROBLEM_BASE: &str = "https://retrograd.dev/problems";

retrograd_core::wire_enum! {
    /// The closed set of failure kinds the V1 API reports.
    ///
    /// `as_str` is the slug the problem-type URI is built from
    /// (`{PROBLEM_BASE}/{slug}`), so the string beside each variant *is* the
    /// part of the contract a client tests against. It is declared once here
    /// rather than restated by a parallel `slug()`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ProblemKind {
        /// The request is malformed or fails the same validation `config::load`
        /// applies to a TOML.
        InvalidRequest = "invalid-request",
        /// The client supplied something only the operator may declare: a reward
        /// command, a judge endpoint or key, an MCP server command.
        ServerDeclared = "server-declared",
        /// A catalogue identifier the operator has not declared.
        UnknownCatalogId = "unknown-catalog-id",
        /// No arrangement of the memory levers fits the budget.
        InsufficientMemory = "insufficient-memory",
        /// A path outside every allowed root.
        ForbiddenPath = "forbidden-path",
        NotFound = "not-found",
        /// A legal request against a resource whose state forbids it.
        Conflict = "conflict",
        /// The body is larger than this server accepts.
        PayloadTooLarge = "payload-too-large",
        /// The body did not arrive as JSON. Every route but `POST /v1/plan` (which
        /// also reads TOML) and the NDJSON upload wants `application/json`; parsing
        /// whatever arrived anyway would mean a client that mislabels its body finds
        /// out only when a field goes missing.
        UnsupportedMediaType = "unsupported-media-type",
        /// The path exists but not for this method.
        MethodNotAllowed = "method-not-allowed",
        /// The device is saturated: every run slot is taken.
        DeviceBusy = "device-busy",
        /// The run did not answer a request-shaped command in time. Only `evaluate`
        /// and `generate` can produce it: they are the only routes that wait for the
        /// loop to reach its next callback.
        Timeout = "timeout",
        /// The caller is not authenticated, or its token is wrong.
        Unauthorized = "unauthorized",
        /// A route that exists in the V1 contract but is not wired yet. Explicit,
        /// so a client sees "not built" instead of a 404 that reads as "wrong URL".
        NotImplemented = "not-implemented",
        Internal = "internal",
    }
}

impl ProblemKind {
    fn title(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid request",
            Self::ServerDeclared => "server-declared value",
            Self::UnknownCatalogId => "unknown catalog id",
            Self::InsufficientMemory => "insufficient memory",
            Self::ForbiddenPath => "forbidden path",
            Self::NotFound => "not found",
            Self::Conflict => "conflict",
            Self::PayloadTooLarge => "payload too large",
            Self::UnsupportedMediaType => "unsupported media type",
            Self::MethodNotAllowed => "method not allowed",
            Self::DeviceBusy => "device busy",
            Self::Timeout => "timeout",
            Self::Unauthorized => "unauthorized",
            Self::NotImplemented => "not implemented",
            Self::Internal => "internal error",
        }
    }

    pub fn status(self) -> StatusCode {
        match self {
            Self::InvalidRequest
            | Self::ServerDeclared
            | Self::UnknownCatalogId
            | Self::InsufficientMemory => StatusCode::UNPROCESSABLE_ENTITY,
            Self::ForbiddenPath => StatusCode::FORBIDDEN,
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Conflict => StatusCode::CONFLICT,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
            Self::DeviceBusy => StatusCode::SERVICE_UNAVAILABLE,
            Self::Timeout => StatusCode::GATEWAY_TIMEOUT,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

retrograd_core::wire_enum! {
    /// The closed vocabulary a client's `match` is written against.
    ///
    /// `message` and `hint` are prose and may be reworded release to release;
    /// `code` is the part of the contract, tested (`tests/error_catalog.rs`)
    /// against `ERRORS.md`.
    ///
    /// The spelling beside each variant is the one serde writes *and* the one
    /// `as_str()` returns - one token, so the two cannot drift and no test has
    /// to check them against each other.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    pub enum ErrorCode: serde {
        /// A required field was absent.
        MissingField = "missing_field",
        /// `deny_unknown_fields` rejected a field the schema does not have.
        UnknownField = "unknown_field",
        /// The field is present but its value does not fit (wrong shape, wrong
        /// enum member, wrong combination with a sibling field).
        InvalidValue = "invalid_value",
        /// A numeric field is outside the range the field accepts.
        OutOfRange = "out_of_range",
        /// A file or media-type field named something this server does not parse.
        UnsupportedFormat = "unsupported_format",
        /// A path field names something that does not exist on disk.
        PathNotFound = "path_not_found",
        /// A path field names something outside every root this server serves.
        ForbiddenPath = "forbidden_path",
        /// A catalogue reference (`reward`, `judge`, an MCP server) the operator
        /// has not declared.
        UnknownCatalogId = "unknown_catalog_id",
        /// The request needs a degradation from `Allow` and did not opt into it.
        NeedsOptIn = "needs_opt_in",
        /// A client-supplied parameter conflicts with the rest of the request.
        OverrideConflict = "override_conflict",
        /// A legal request against a resource whose current state forbids it.
        Conflict = "conflict",
        /// The caller's credentials are absent or wrong.
        Unauthorized = "unauthorized",
        /// The referenced resource does not exist.
        NotFound = "not_found",
        /// The field is reserved for the operator; a client may not set it.
        ServerDeclared = "server_declared",
    }
}

/// One field-level problem, located by a JSON Pointer (RFC 6901) into the
/// request body.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct FieldError {
    pub pointer: String,
    pub code: ErrorCode,
    pub message: String,
    /// What to do about it, only when there is a short, concrete answer. A
    /// vague hint is worse than none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("{}: {detail}",.kind.title())]
pub struct ApiError {
    pub kind: ProblemKind,
    pub detail: String,
    pub errors: Vec<FieldError>,
    /// The structured payload specific to `kind`, e.g. the dominant memory
    /// posts of `insufficient-memory`, so that numbers never travel through
    /// `errors[]`.
    pub meta: Option<serde_json::Value>,
}

impl ApiError {
    pub fn new(kind: ProblemKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            errors: Vec::new(),
            meta: None,
        }
    }

    /// Attaches the structured, `kind`-specific payload. Overwrites any
    /// previous `meta`; only one problem type produces this per response.
    pub fn with_meta(mut self, meta: serde_json::Value) -> Self {
        self.meta = Some(meta);
        self
    }

    pub fn invalid(detail: impl Into<String>) -> Self {
        Self::new(ProblemKind::InvalidRequest, detail)
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self::new(ProblemKind::NotFound, detail)
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(ProblemKind::Internal, detail)
    }

    /// Attaches a field-level pointer. Several may be attached; they are
    /// reported in insertion order so the response is deterministic.
    pub fn with_field(
        mut self,
        pointer: impl Into<String>,
        code: ErrorCode,
        message: impl Into<String>,
    ) -> Self {
        self.errors.push(FieldError {
            pointer: pointer.into(),
            code,
            message: message.into(),
            hint: None,
        });
        self
    }

    /// Sets the `hint` of the last field error attached. Panics if none has
    /// been attached yet - a hint with nothing to hint about is a bug at the
    /// call site, not a runtime condition.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.errors
            .last_mut()
            .expect("with_hint called before with_field")
            .hint = Some(hint.into());
        self
    }
}

tokio::task_local! {
    /// The current request's trace id, set by [`crate::trace_id_middleware`]
    /// for the duration of the request future. A test that builds an
    /// `ApiError` outside that scope simply gets no `trace_id` - the field is
    /// optional on the wire for exactly that reason.
    pub(crate) static TRACE_ID: String;
}

/// The trace id of the request currently being handled, if any.
pub fn current_trace_id() -> Option<String> {
    TRACE_ID.try_with(Clone::clone).ok()
}

/// Wire shape of a problem document. Serialized by hand rather than derived
/// from `ApiError` so the field order - and therefore the bytes - is fixed.
#[derive(Serialize)]
struct ProblemDocument<'a> {
    #[serde(rename = "type")]
    problem_type: String,
    title: &'a str,
    status: u16,
    detail: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: &'a Vec<FieldError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta: &'a Option<serde_json::Value>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.kind.status();
        let document = ProblemDocument {
            problem_type: format!("{PROBLEM_BASE}/{}", self.kind.as_str()),
            title: self.kind.title(),
            status: status.as_u16(),
            detail: &self.detail,
            trace_id: current_trace_id(),
            errors: &self.errors,
            meta: &self.meta,
        };
        let body = serde_json::to_vec(&document).unwrap_or_else(|_| {
            // Serializing a struct of owned strings cannot fail; a minimal
            // hand-written document is still better than a panic in a handler.
            br#"{
                "type": "https://retrograd.dev/problems/internal",
                "title": "internal error",
                "status": 500,
                "detail": "failed to render the problem document"
            }"#
            .to_vec()
        });
        let mut response = (
            status,
            [(header::CONTENT_TYPE, "application/problem+json")],
            body,
        )
            .into_response();
        // RFC 9110 section 11.6.1: a 401 without a challenge is not a 401. `Bearer` is
        // the only scheme this server accepts.
        if self.kind == ProblemKind::Unauthorized {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static("Bearer"),
            );
        }
        response
    }
}

/// Replaces every absolute path in a message with its last component.
///
/// `retrograd_core::Error` messages carry the paths they failed on, which is what
/// makes them useful but may expose the server's filesystem. Trim them from
/// external responses and retain them in logs.
///
/// Redaction depends on whether the response leaves the machine, not on authentication:
/// [`crate::ServerConfig::validate`] refuses an unauthenticated non-loopback
/// bind, so no-authentication implies a caller who can already read the filesystem
/// and for whom redaction buys nothing while costing every diagnostic. The
/// boundary that matters is the *machine*: what leaves it is redacted, whoever
/// asked. A shared bearer token is not an identity.
pub fn redact_paths(message: &str) -> String {
    let mut out = String::with_capacity(message.len());
    let mut rest = message;
    while let Some(start) = rest.find('/') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        // A path ends at the first character that cannot be in one. Quotes and
        // parentheses matter: the messages wrap paths in both.
        let end = tail
            .find(|character: char| {
                character.is_whitespace() || matches!(character, '\'' | '"' | ')' | ',' | ';')
            })
            .unwrap_or(tail.len());
        let (path, remainder) = tail.split_at(end);
        // A single `/`, or something with no second component, is not a path worth
        // hiding - and `and/or` should not become `or`.
        let components = path.split('/').filter(|part| !part.is_empty()).count();
        if components >= 2 && !path.contains(' ') {
            out.push_str("<path>/");
            out.push_str(path.rsplit('/').next().unwrap_or_default());
        } else {
            out.push_str(path);
        }
        rest = remainder;
    }
    out.push_str(rest);
    out
}

impl ApiError {
    /// The same problem with every absolute path trimmed (see [`redact_paths`]).
    pub fn redacted(mut self) -> Self {
        self.detail = redact_paths(&self.detail);
        for error in &mut self.errors {
            error.message = redact_paths(&error.message);
            if let Some(hint) = &error.hint {
                error.hint = Some(redact_paths(hint));
            }
        }
        if let Some(meta) = &self.meta {
            self.meta = Some(redact_paths_in_value(meta));
        }
        self
    }
}

/// Applies [`redact_paths`] to every string leaf of a JSON value, recursively.
/// `meta` is typed per problem kind, so this is cheaper and less error-prone
/// than teaching every producer to redact its own strings.
pub(crate) fn redact_paths_in_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::String(text) => serde_json::Value::String(redact_paths(text)),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(redact_paths_in_value).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), redact_paths_in_value(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Converts a core error into an API error. What the caller can fix by writing
/// something else is theirs (422), everything else is ours (500).
///
/// The detail is passed through as-is here; trimming absolute paths out of it is
/// [`redact_paths`], applied once on the way out rather than in every conversion.
impl From<retrograd_core::Error> for ApiError {
    fn from(error: retrograd_core::Error) -> Self {
        let detail = error.to_string();
        // `is_user_error` rather than one variant: a malformed configuration and
        // a malformed dataset record are the client's fault exactly like a bad
        // argument, and must answer 422 rather than being grouped with runtime
        // failures.
        if error.is_user_error() {
            Self::invalid(detail)
        } else {
            Self::internal(detail)
        }
    }
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_keeps_its_shape_and_loses_the_server_s_filesystem() {
        assert_eq!(
            redact_paths("could not read /Users/someone/models/qwen3.gguf: not found"),
            "could not read <path>/qwen3.gguf: not found"
        );
        assert_eq!(
            redact_paths("path '/srv/data/sft.jsonl' cannot be resolved"),
            "path '<path>/sft.jsonl' cannot be resolved"
        );
        // Not paths: a bare separator, a prose slash, a relative name.
        assert_eq!(
            redact_paths("use steps and/or best_eval"),
            "use steps and/or best_eval"
        );
        assert_eq!(
            redact_paths("nothing to redact here"),
            "nothing to redact here"
        );
        assert_eq!(
            redact_paths("mode is steps / best_eval"),
            "mode is steps / best_eval"
        );
        // Two of them in one message.
        assert_eq!(
            redact_paths("copying /a/b/one.gguf to /c/d/two.gguf failed"),
            "copying <path>/one.gguf to <path>/two.gguf failed"
        );
    }

    #[test]
    fn a_redacted_problem_trims_its_field_messages_too() {
        let problem = ApiError::invalid("could not read /srv/models/m.gguf")
            .with_field(
                "/model",
                ErrorCode::PathNotFound,
                "no such file: /srv/models/m.gguf",
            )
            .redacted();
        assert_eq!(problem.detail, "could not read <path>/m.gguf");
        assert_eq!(problem.errors[0].message, "no such file: <path>/m.gguf");
        // The pointer is the client's own JSON path and is never redacted.
        assert_eq!(problem.errors[0].pointer, "/model");
    }
}
