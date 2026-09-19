//! `POST /v1/runs`, `GET /v1/runs`, `GET /v1/runs/{id}`.
//!
//! Creation is a resolution followed by a registration: the same three body
//! forms as `/v1/plan`, the same guard, the same budget check, and then - if the
//! caller did not ask for a dry run - a journal on disk and a worker queued
//! behind the device.
//!
//! The answer comes back before the run starts, on purpose. Loading a model is
//! seconds and training is hours; a `201` that waited for either would be a
//! request a client has to hold open for a job it is going to poll anyway.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use http::{HeaderMap, StatusCode};
use serde::Deserialize;
use serde_json::value::{RawValue, to_raw_value};
use uuid::Uuid;

use super::plan::{PlanBody, resolve_body};
use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::resolve::Resolved;
use crate::runtime::control;
use crate::runtime::registry::{RunArtifacts, RunControls, RunFilter, RunRecord};
use crate::runtime::worker;
use crate::state::AppState;

/// Default page size of `GET /v1/runs`.
const DEFAULT_PAGE: usize = 50;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateQuery {
    /// Resolve and answer, create nothing.
    #[serde(default)]
    pub dry_run: bool,
    /// Accept a configuration whose estimate is over budget.
    #[serde(default)]
    pub force: bool,
    /// Measure the candidate. Defaults to the server's `calibrate_runs`, which
    /// is itself on: the model is about to be loaded either way.
    #[serde(default)]
    pub calibrate: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListQuery {
    #[serde(default)]
    pub status: Option<dto::RunStatus>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
}

/// `POST /v1/runs`
pub async fn create(
    State(state): State<AppState>,
    Query(query): Query<CreateQuery>,
    headers: HeaderMap,
    body: PlanBody,
) -> ApiResult<axum::response::Response> {
    let idempotency_key = idempotency_key(&headers)?;
    // A repeat of a request that already created a run answers that run rather
    // than starting a second one. Checked before the resolution: the point of
    // the header is to make a retry after a timeout free.
    if let Some(key) = &idempotency_key
        && let Some(existing) = state.registry.by_idempotency_key(key)
    {
        return Ok((StatusCode::OK, Json(existing.view())).into_response());
    }

    let calibrate = match (query.dry_run, query.calibrate) {
        (_, Some(explicit)) => explicit,
        // A dry run stays a plan: it must not load a model onto the device
        // unless the caller asked for exactly that.
        (true, None) => false,
        (false, None) => state.config.calibrate_runs(),
    };
    let resolved = resolve_body(&state, body, query.force, calibrate).await?;

    if query.dry_run {
        // The same payload a creation returns, minus the identity - there is
        // nothing to identify.
        return Ok((StatusCode::OK, Json(resolved.response)).into_response());
    }

    let Resolved {
        mut response,
        mut config,
        iterations,
        managed_adapter,
    } = resolved;
    let id = Uuid::new_v4();
    if managed_adapter {
        let output = state
            .registry
            .state_dir()
            .join(id.to_string())
            .join("adapter.gguf");
        config.output.path = output.clone();
        response.effective_config.output = Some(retrograd_config::OutputToml {
            path: output,
            // The resolver drafts LoRA runs, so the managed placeholder keeps
            // the default kind rather than pinning one the client never chose.
            kind: None,
        });
    }
    // Rendered once, from the typed resolution, and then reused verbatim by the
    // response, the journal and every later `GET`. Serializing the typed value is
    // also what keeps an `f32` reading as the number that was chosen
    let record = RunRecord {
        effective_config: render(&response.effective_config)?,
        provenance: render(&response.provenance)?,
        plan: render(&response.plan)?,
        iterations,
        // Read off the resolved configuration, not off the JSON it renders to:
        // this is what lets `PATCH` refuse "evaluate more often" on a run with
        // no evaluation dataset without re-parsing a document.
        controls: RunControls {
            has_evaluation: config.evaluation.is_some(),
            has_checkpoint: config.checkpoint.is_some(),
        },
        // Same reasoning, and stronger: the inventory has to answer for a run
        // this process only read back from disk, so these paths go into
        // `run.json` instead of being re-derived from the rendered document.
        artifacts: artifacts_of(&config),
    };
    let (sender, commands) = control::channel();
    let handle = state
        .registry
        .create_with_id(id, response.name.clone(), record, idempotency_key, sender)
        .map_err(ApiError::from)?;
    let view = handle.view();
    worker::spawn(
        state.engine.clone(),
        state.device.clone(),
        handle,
        config,
        commands,
    );
    Ok((StatusCode::CREATED, Json(view)).into_response())
}

/// `GET /v1/runs`
pub async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> ApiResult<Json<dto::RunListing>> {
    let page = state.registry.list(&RunFilter {
        status: query.status,
        name: query.name,
        limit: query.limit.unwrap_or(DEFAULT_PAGE),
        cursor: query.cursor,
    });
    Ok(Json(dto::RunListing {
        runs: page.runs,
        next_cursor: page.next_cursor,
    }))
}

/// `GET /v1/runs/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::RunView>> {
    let handle = lookup(&state, &id)?;
    Ok(Json(handle.view()))
}

pub(crate) fn lookup(state: &AppState, id: &str) -> ApiResult<Arc<crate::runtime::RunHandle>> {
    // A malformed id is a 404 and not a 422: to a client the id is opaque
    //, so "not a UUID" and "no such run" are the same answer.
    let uuid = id
        .parse::<Uuid>()
        .map_err(|_| ApiError::not_found(format!("no run with id {id}")))?;
    state
        .registry
        .get(&uuid)
        .ok_or_else(|| ApiError::not_found(format!("no run with id {id}")))
}

/// Where this run's outputs will land, read off the resolved configuration.
fn artifacts_of(config: &retrograd_config::RunConfig) -> RunArtifacts {
    RunArtifacts {
        adapter: (config.output.kind == retrograd_config::OutputKind::Adapter)
            .then(|| config.output.path.clone()),
        checkpoint_directory: config
            .checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.directory.clone()),
        tensorboard_directory: config.metrics.tensorboard_dir.clone(),
        wandb_export_directory: config.metrics.wandb_export_dir.clone(),
        observe: config
            .observe
            .as_ref()
            .map(|observe| observe.directory.clone()),
    }
}

fn render<T: serde::Serialize>(value: &T) -> ApiResult<Box<RawValue>> {
    to_raw_value(value)
        .map_err(|error| ApiError::internal(format!("could not render the run record: {error}")))
}

/// The `Idempotency-Key` header, if the client sent one.
fn idempotency_key(headers: &HeaderMap) -> ApiResult<Option<String>> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let key = value
        .to_str()
        .map_err(|_| {
            ApiError::invalid("the Idempotency-Key header must be ASCII").with_field(
                "/headers/idempotency-key",
                ErrorCode::InvalidValue,
                "not ASCII",
            )
        })?
        .trim();
    if key.is_empty() {
        return Err(
            ApiError::invalid("the Idempotency-Key header is empty").with_field(
                "/headers/idempotency-key",
                ErrorCode::InvalidValue,
                "empty",
            ),
        );
    }
    if key.len() > 200 {
        return Err(ApiError::new(
            ProblemKind::InvalidRequest,
            "the Idempotency-Key is too long",
        )
        .with_field(
            "/headers/idempotency-key",
            ErrorCode::OutOfRange,
            "at most 200 characters",
        ));
    }
    Ok(Some(key.to_string()))
}
