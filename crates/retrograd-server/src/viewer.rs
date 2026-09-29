//! `retrograd-server view <directory>`: one run's trajectories, read-only.
//!
//! The web interface over a single `[observe]` directory - what a run started
//! by the command line writes - with nothing else behind it: no registry, no
//! catalogue, no engine, no token. It serves `GET /v1/health`,
//! `GET /v1/capabilities` (whose `features.mode` is `viewer`, which is how the
//! interface knows to show this one run and no navigation), and the three
//! trajectory routes for the run `local`. Every other path under `/v1` is the
//! API's own 404, and every write a 405: there is nothing here to write to.
//!
//! No token because it only ever binds loopback: the binary refuses anything
//! else before calling [`build_viewer_router`].

use std::path::PathBuf;
use std::time::Duration;

use axum::routing::get;
use axum::{Json, Router};
use tower_http::trace::TraceLayer;

use crate::api::trajectories::{self, DirectorySource};
use crate::error::ApiResult;
use crate::{dto, state};

/// How long a read may take. The routes are bounded by their pagination, so
/// this only has to be generous.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// The viewer over `directory`.
pub fn build_viewer_router(directory: PathBuf) -> Router {
    let v1 = Router::new()
        .route("/health", get(health))
        .route("/capabilities", get(capabilities))
        .merge(trajectories::router(
            DirectorySource(directory),
            REQUEST_TIMEOUT,
        ))
        .fallback(crate::not_found);
    let root = Router::new().nest("/v1", v1);
    #[cfg(feature = "ui")]
    let root = root.fallback_service(retrograd_ui::router());
    #[cfg(not(feature = "ui"))]
    let root = root.fallback(crate::not_found);
    root.layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn(crate::as_problem_document))
        .layer(axum::middleware::from_fn(crate::trace_id_middleware))
}

async fn health() -> Json<dto::Health> {
    Json(dto::Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        backends: state::compiled_backends(),
    })
}

async fn capabilities() -> ApiResult<Json<dto::Capabilities>> {
    let baseline = state::measure_baseline();
    let budgets = retrograd_plan::Budgets::resolve(
        baseline,
        Default::default(),
        (None, None),
        Default::default(),
    );
    Ok(Json(crate::api::discovery::describe(
        state::compiled_backends(),
        baseline.unified,
        budgets,
        dto::Features {
            openapi: cfg!(feature = "openapi"),
            max_concurrent_runs: 0,
            serving_enabled: false,
            ui: cfg!(feature = "ui"),
            auth: false,
            mode: dto::ServerMode::Viewer,
        },
    )?))
}
