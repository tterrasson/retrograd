//! `retrograd-server`: an HTTP control plane for training runs.
//!
//! The crate exposes [`build_router`], which returns a plain `axum::Router` over
//! an [`AppState`]. Everything the API does is reachable through that router with
//! no socket, no model and no GPU, which is what lets the whole HTTP surface be
//! tested in the fast lane (`tower::ServiceExt::oneshot`).

pub mod api;
pub mod auth;
pub mod catalog;
pub mod datasets;
pub mod defaults;
pub mod dto;
pub mod error;
pub mod extract;
pub mod guard;
mod lock;
pub mod openapi;
pub mod resolve;
pub mod runtime;
pub mod shutdown;
pub mod state;

use axum::Router;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use http::{StatusCode, header};
use tower::limit::GlobalConcurrencyLimitLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

pub use catalog::Catalog;
pub use error::{ApiError, ApiResult, ProblemKind};
pub use runtime::{RunEngine, RunRegistry, TrainingEngine};
pub use state::{AppState, EngineProbe, ModelProbe, ServerConfig};

/// Builds the `/v1` router.
///
/// Kept separate from the binary so tests drive the same routes the process
/// serves: a route registered in `main` and not here would be untested by
/// construction.
///
/// The layering is where the hardening of the API lives, and the order is the point.
/// From the outside in: every error becomes a problem document, then the
/// concurrency limit (so an excessive request is turned away before anything
/// buffers it), then tracing, then authentication - which sits outside every
/// handler, because an unauthenticated caller must not reach one.
///
/// Two deliberate asymmetries, both per-branch rather than global:
///
/// - **The request timeout** wraps the routes that answer and finish, and
///   **not** the event streams or dataset ingestion: an SSE connection is meant
///   to stay open for the length of a training run, and an upload over a slow
///   link legitimately outlasts a plan. A timeout over either would cut it and
///   report it as a failure.
/// - **The body limit** is sized to what each branch actually accepts - the
///   small-request ceiling everywhere, `max_dataset_bytes` on `POST
///   /v1/datasets`, none at all on the streams, which carry no body.
pub fn build_router(state: AppState) -> Router {
    let config = state.config.clone();

    // Streams: no timeout, by design (see above). No body limit either - a GET
    // carries no body, and the blanket body limit is not applied to this router.
    let streams = Router::new()
        .route("/runs/{id}/events", get(api::events::stream))
        .route("/events", get(api::events::stream_all))
        .with_state(state.clone());

    // Dataset ingestion streams up to `max_dataset_bytes` - an operator setting
    // that starts at 1 GiB, far past the small-request ceiling every other
    // route is held to - and skips the request timeout
    // for the same reason streams do: uploading a large file over a slow link
    // legitimately takes longer than a plan or a control command.
    let datasets = Router::new()
        .route(
            "/datasets",
            post(api::datasets::create).get(api::datasets::list),
        )
        .layer(RequestBodyLimitLayer::new(
            config.max_dataset_bytes() as usize
        ))
        .with_state(state.clone());

    let v1 = Router::new()
        .route("/health", get(api::discovery::health))
        .route("/capabilities", get(api::discovery::capabilities))
        .route("/defaults", get(api::discovery::defaults))
        .route("/rewards", get(api::discovery::rewards))
        .route("/judges", get(api::discovery::judges))
        .route("/mcp-servers", get(api::discovery::mcp_servers))
        .route("/environments", get(api::discovery::environments))
        .route("/openapi.json", get(openapi::document))
        .route("/preflight", post(api::discovery::preflight))
        .route("/plan", post(api::plan::plan))
        .route("/runs", post(api::runs::create).get(api::runs::list))
        .route(
            "/runs/{id}",
            get(api::runs::get)
                .patch(api::control::patch)
                .delete(api::artifacts::delete),
        )
        .route("/runs/{id}/pause", post(api::control::pause))
        .route("/runs/{id}/resume", post(api::control::resume))
        .route("/runs/{id}/cancel", post(api::control::cancel))
        .route(
            "/runs/{id}/checkpoints",
            post(api::control::checkpoint).get(api::artifacts::checkpoints),
        )
        .route("/runs/{id}/evaluate", post(api::inference::evaluate))
        .route("/runs/{id}/generate", post(api::inference::generate))
        .route("/runs/{id}/artifacts", get(api::artifacts::list))
        .route("/runs/{id}/artifacts/{name}", get(api::artifacts::download))
        .route("/runs/{id}/metrics", get(api::events::metrics))
        .route(
            "/datasets/{id}",
            get(api::datasets::get).delete(api::datasets::delete),
        )
        .route("/datasets/{id}/preview", get(api::datasets::preview))
        .route("/datasets/{id}/tokenize", post(api::datasets::tokenize))
        // `evaluate` and `generate` wait for the run's next callback, which is
        // legitimately longer than a request timeout; they carry their own bound
        // (`command_timeout_seconds`) and answer 504 themselves.
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            config.command_timeout().max(config.request_timeout()),
        ))
        .layer(RequestBodyLimitLayer::new(config.max_body_bytes()))
        .with_state(state.clone())
        .merge(streams)
        .merge(datasets);

    Router::new()
        .nest("/v1", v1)
        // An unknown path must answer with a problem document like every other
        // failure, not with axum's empty 404 body.
        .fallback(not_found)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_token,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state,
            redact_problem_paths,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(GlobalConcurrencyLimitLayer::new(
            config.max_concurrent_requests(),
        ))
        // Outermost, so it also covers what the layers below it produce: the
        // timeout's 408 is generated by middleware and would otherwise be the
        // only failure in the API without a body. Each branch above sets its
        // own body limit, sized to what it actually accepts.
        .layer(axum::middleware::from_fn(as_problem_document))
        // Absolute outermost: every response, including the ones the layers
        // above invent, gets a `trace_id`, and every log line written while
        // handling this request carries the same one. This lets
        // `redact_problem_paths` drop the filesystem detail without losing the
        // operator's ability to find it in the logs.
        .layer(axum::middleware::from_fn(trace_id_middleware))
}

/// Generates one trace id per request, puts it on the tracing span every log
/// line in this request writes through, and lets [`error::current_trace_id`]
/// read it back when a problem document is rendered.
async fn trace_id_middleware(request: Request, next: axum::middleware::Next) -> Response {
    let trace_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!("request", trace_id = %trace_id);
    error::TRACE_ID
        .scope(trace_id, next.run(request).instrument(span))
        .await
}

async fn not_found(uri: http::Uri) -> ApiError {
    ApiError::not_found(format!("no route for {}", uri.path()))
}

/// Replaces any error response that is not already a problem document with one.
///
/// The handlers all answer through [`ApiError`], so this exists for the responses
/// *middleware* invents: `413` from the body limit, `408` from the timeout, `405`
/// from axum's method routing. Every failure uses `application/problem+json`,
/// so a client parsing one shape never meets a second.
async fn as_problem_document(request: Request, next: axum::middleware::Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    if !status.is_client_error() && !status.is_server_error() {
        return response;
    }
    let is_problem = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/problem+json"));
    if is_problem {
        return response;
    }
    let detail = match status {
        StatusCode::PAYLOAD_TOO_LARGE => "the request body is larger than this server accepts",
        StatusCode::REQUEST_TIMEOUT => "the request took longer than this server allows",
        StatusCode::METHOD_NOT_ALLOWED => "that method is not allowed on this path",
        StatusCode::SERVICE_UNAVAILABLE => {
            "the server is already serving as many requests as it accepts"
        }
        _ => "the request could not be served",
    };
    problem_from_rejection(status, detail).into_response()
}

/// Trims absolute paths out of problem documents on the way out.
///
/// Done here rather than at every construction site because there are dozens of
/// those and one of them will always be forgotten; a problem document is small, so
/// re-reading and re-rendering one costs nothing on a path that is already an
/// error. The untrimmed message goes to the log, where an operator can read it.
async fn redact_problem_paths(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let response = next.run(request).await;
    if !state.config.redact_error_paths() {
        return response;
    }
    let is_problem = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/problem+json"));
    if !is_problem {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 64 * 1024).await else {
        // Unreadable, so it cannot be redacted; an empty problem body is a worse
        // answer than none, so the response is rebuilt as a bare status.
        return (parts.status, "").into_response();
    };
    let Ok(mut document) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return (parts, bytes).into_response();
    };
    if let Some(detail) = document.get("detail").and_then(|value| value.as_str()) {
        let redacted = error::redact_paths(detail);
        if redacted != detail {
            tracing::info!(detail, "redacted the paths in an error response");
            document["detail"] = serde_json::Value::String(redacted);
        }
    }
    if let Some(errors) = document
        .get_mut("errors")
        .and_then(|value| value.as_array_mut())
    {
        for error in errors {
            if let Some(message) = error.get("message").and_then(|value| value.as_str()) {
                let redacted = error::redact_paths(message);
                error["message"] = serde_json::Value::String(redacted);
            }
            if let Some(hint) = error.get("hint").and_then(|value| value.as_str()) {
                let redacted = error::redact_paths(hint);
                error["hint"] = serde_json::Value::String(redacted);
            }
        }
    }
    if let Some(meta) = document.get_mut("meta") {
        *meta = error::redact_paths_in_value(meta);
    }
    let body = serde_json::to_vec(&document).unwrap_or_else(|_| bytes.to_vec());
    (parts, body).into_response()
}

/// Axum's own rejections (a malformed JSON body, a missing content type) do not
/// go through [`ApiError`], so they would answer in plain text. This converts the
/// ones a client can trigger into problem documents.
///
/// Exposed for the handlers that need it and for the tests that assert every
/// error is a problem document.
///
/// Every status a middleware can produce has a [`ProblemKind`] that maps back to
/// it, so the document's `status` field and the response's are the same number.
/// A kind whose status differed would make a client that reads one disagree with a
/// client that reads the other.
pub fn problem_from_rejection(status: StatusCode, detail: impl Into<String>) -> ApiError {
    let kind = match status {
        StatusCode::UNPROCESSABLE_ENTITY | StatusCode::BAD_REQUEST => ProblemKind::InvalidRequest,
        StatusCode::PAYLOAD_TOO_LARGE => ProblemKind::PayloadTooLarge,
        StatusCode::METHOD_NOT_ALLOWED => ProblemKind::MethodNotAllowed,
        StatusCode::NOT_FOUND => ProblemKind::NotFound,
        StatusCode::UNAUTHORIZED => ProblemKind::Unauthorized,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ProblemKind::Timeout,
        StatusCode::SERVICE_UNAVAILABLE => ProblemKind::DeviceBusy,
        StatusCode::CONFLICT => ProblemKind::Conflict,
        _ => ProblemKind::Internal,
    };
    ApiError::new(kind, detail)
}
