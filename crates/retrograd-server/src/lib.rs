//! `retrograd-server`: an HTTP control plane for training runs.
//!
//! The crate exposes [`build_router`], which returns a plain `axum::Router` over
//! an [`AppState`]. Everything the API does is reachable through that router with
//! no socket, no model and no GPU, which is what lets the whole HTTP surface be
//! tested in the fast lane (`tower::ServiceExt::oneshot`).

pub mod api;
pub mod auth;
pub mod catalog;
#[cfg(feature = "openapi")]
pub mod config_schema;
pub mod datasets;
pub mod defaults;
pub mod dto;
pub mod error;
pub mod extract;
pub mod guard;
pub mod links;
mod lock;
pub mod model_files;
pub mod openapi;
pub mod resolve;
pub mod runtime;
pub mod shutdown;
pub mod state;
pub mod viewer;

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
pub use viewer::build_viewer_router;

/// Builds the `/v1` router, and under `/` the web interface when this build
/// embeds it ([`ServerConfig::serves_ui`]).
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
        // A reverse proxy that buffers a stream delivers it when the run is
        // over; nginx reads this header as "pass it through as it comes".
        .layer(axum::middleware::map_response(
            |mut response: Response| async move {
                response.headers_mut().insert(
                    http::HeaderName::from_static("x-accel-buffering"),
                    http::HeaderValue::from_static("no"),
                );
                response
            },
        ))
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

    // The OpenAI contract over the runs' weights. No timeout layer, like the
    // streams: an answer from a live run waits for its next progress callback,
    // and a stream stays open until then. The bound is `command_timeout`,
    // applied by the handler. Its own body limit, because a chat history
    // outgrows the small-request ceiling.
    let openai = retrograd_openai::router(retrograd_openai::Endpoint {
        source: std::sync::Arc::new(api::openai::RunModels::new(state.clone())),
        session: state.serving.clone(),
        timeout: config.command_timeout(),
        enabled: config.serving.enabled(),
    })
    .layer(RequestBodyLimitLayer::new(config.serving.max_body_bytes()));

    let v1 = Router::new()
        .route("/health", get(api::discovery::health))
        .route("/capabilities", get(api::discovery::capabilities))
        .route("/defaults", get(api::discovery::defaults))
        .route("/rewards", get(api::discovery::rewards))
        .route("/judges", get(api::discovery::judges))
        .route("/mcp-servers", get(api::discovery::mcp_servers))
        .route("/environments", get(api::discovery::environments))
        .route("/openapi.json", get(openapi::document))
        .route("/config-schema", get(openapi::config_schema))
        .route("/preflight", post(api::discovery::preflight))
        .route("/model-files", get(api::discovery::model_files))
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
        .route(
            "/runs/{id}/artifacts/{name}/link",
            post(api::artifacts::link),
        )
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
        .merge(datasets)
        .merge(openai)
        .merge(api::trajectories::router(
            api::trajectories::RegistrySource(state.registry.clone()),
            config.request_timeout(),
        ))
        // An unknown path under `/v1` answers with a problem document like
        // every other failure of the API, not with axum's empty 404 body -
        // whether or not the web interface owns the rest of the paths.
        .fallback(not_found);

    let root = Router::new().nest("/v1", v1);
    let root = mount_ui(root, &config);
    harden(root, state)
}

/// `/` and every path outside `/v1`: the web interface when this build embeds
/// it and the operator left it on, a problem document otherwise.
fn mount_ui(root: Router, config: &ServerConfig) -> Router {
    #[cfg(feature = "ui")]
    if config.serves_ui() {
        return root.fallback_service(retrograd_ui::router());
    }
    #[cfg(not(feature = "ui"))]
    let _ = config;
    root.fallback(not_found)
}

/// The layers every router of this crate is served through; see
/// [`build_router`] for why they are in this order.
fn harden(root: Router, state: AppState) -> Router {
    let config = state.config.clone();
    root.layer(axum::middleware::from_fn_with_state(
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
pub(crate) async fn trace_id_middleware(
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let trace_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!("request", trace_id = %trace_id);
    error::TRACE_ID
        .scope(trace_id, next.run(request).instrument(span))
        .await
}

pub(crate) async fn not_found(
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> ApiError {
    ApiError::not_found(format!("no route for {}", uri.path()))
}

/// Replaces any error response that is not already a problem document with one.
///
/// The handlers all answer through [`ApiError`], so this exists for the responses
/// *middleware* invents: `413` from the body limit, `408` from the timeout, `405`
/// from axum's method routing. Every failure uses `application/problem+json`,
/// so a client parsing one shape never meets a second.
///
/// The OpenAI routes are the exception: their clients read an OpenAI error
/// envelope and nothing else, so an envelope passes through untouched and a
/// middleware's own failure on one of those routes becomes an envelope too.
///
/// Only under `/v1`: the web interface's own `404` for a missing file is a
/// plain one, and turning it into JSON would make a browser report a media-type
/// error for a script that is simply absent.
pub(crate) async fn as_problem_document(
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let path = request.uri().path();
    let openai = api::openai::is_openai_path(path);
    let api = auth::is_api_path(path) || !serves_ui_paths();
    let response = next.run(request).await;
    if !api {
        return response;
    }
    let status = response.status();
    if !status.is_client_error() && !status.is_server_error() {
        return response;
    }
    as_problem(response, status, openai)
}

/// Whether paths outside `/v1` can belong to the web interface at all in this
/// build. Without it they are the API's unknown paths, answered like any
/// other.
fn serves_ui_paths() -> bool {
    cfg!(feature = "ui")
}

fn as_problem(response: Response, status: StatusCode, openai: bool) -> Response {
    if response
        .extensions()
        .get::<retrograd_openai::OpenAiEnvelope>()
        .is_some()
    {
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
    let problem = problem_from_rejection(status, detail);
    if openai {
        return retrograd_openai::OpenAiError::from(problem)
            .with_status(status)
            .into_response();
    }
    problem.into_response()
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
    if response
        .extensions()
        .get::<retrograd_openai::OpenAiEnvelope>()
        .is_some()
    {
        return redact_openai_envelope(response).await;
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

/// [`redact_problem_paths`] for the OpenAI envelope: its one prose field is
/// `error.message`, and a model that failed to load names its file there.
async fn redact_openai_envelope(response: Response) -> Response {
    let (parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 64 * 1024).await else {
        return (parts.status, "").into_response();
    };
    let Ok(mut document) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Response::from_parts(parts, axum::body::Body::from(bytes));
    };
    if let Some(message) = document
        .pointer("/error/message")
        .and_then(|value| value.as_str())
    {
        let redacted = error::redact_paths(message);
        if redacted != message {
            tracing::info!(message, "redacted the paths in an error response");
            document["error"]["message"] = serde_json::Value::String(redacted);
        }
    }
    let body = serde_json::to_vec(&document).unwrap_or_else(|_| bytes.to_vec());
    Response::from_parts(parts, axum::body::Body::from(body))
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
