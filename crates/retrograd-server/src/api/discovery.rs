//! Discovery endpoints: what this server is, what it can do, and what the
//! operator declared. All read-only, none of them touches a run.

use axum::Json;
use axum::extract::State;
use retrograd_core::{Device, TargetSet};

use crate::defaults;
use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::extract::Json as ApiJson;
use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> Json<dto::Health> {
    Json(dto::Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        backends: state.backends.as_ref().clone(),
    })
}

pub async fn capabilities(State(state): State<AppState>) -> ApiResult<Json<dto::Capabilities>> {
    let catalog = crate::resolve::kernel_catalog()?;
    let (rir_mode, rir_policy_latched) = retrograd_engine::rir_runtime_policy()
        .map_err(|error| ApiError::internal(format!("could not read RIR policy: {error}")))?;
    let rir_mode = match rir_mode {
        retrograd_core::RirMode::Off => "off",
        retrograd_core::RirMode::Observe => "observe",
        retrograd_core::RirMode::Prefer => "prefer",
        retrograd_core::RirMode::Require => "require",
    };
    Ok(Json(dto::Capabilities {
        execution_profile_schema_version: retrograd_core::EXECUTION_PROFILE_VERSION,
        kernel_catalog_fingerprint: catalog.fingerprint.clone(),
        rir_mode: rir_mode.to_string(),
        rir_policy_latched,
        backends: state.backends.as_ref().clone(),
        devices: devices(),
        unified_memory: state.baseline.unified,
        budgets: state.budgets(),
        features: dto::Features {
            openapi: cfg!(feature = "openapi"),
            max_concurrent_runs: state.config.concurrency(),
        },
    }))
}

/// The device list, with the memory budget attached to the first GPU.
///
/// Only the first GPU carries a budget because that is the only one
/// `retro_device_memory` reads - the runtime itself picks the first registered
/// GPU device. Reporting a budget on the others would be an invention.
fn devices() -> Vec<dto::DeviceInfo> {
    let Ok(list) = retrograd_engine::backend_list() else {
        return Vec::new();
    };
    let device = retrograd_memory::device_snapshot();
    let mut budget_assigned = false;
    let mut devices = Vec::new();
    for line in list.lines() {
        let mut fields = line.split('\t');
        let kind = fields.next().unwrap_or("other").to_string();
        let name = fields.next().unwrap_or_default().to_string();
        let description = fields.next().unwrap_or_default().to_string();
        let carries_budget = kind == "gpu" && !budget_assigned;
        if carries_budget {
            budget_assigned = true;
        }
        devices.push(dto::DeviceInfo {
            kind,
            name,
            description,
            total_bytes: carries_budget
                .then(|| device.map(|device| device.total))
                .flatten(),
            free_bytes: carries_budget
                .then(|| device.map(|device| device.free))
                .flatten(),
        });
    }
    devices
}

/// `GET /v1/defaults` - the rules the resolver applies when `params` is silent,
/// and the settings it turns on proactively.
///
/// Rendered from the resolver's own tables, so a client reading this reads what
/// will actually happen rather than a copy of it.
pub async fn defaults() -> Json<dto::Defaults> {
    Json(defaults::listing())
}

pub async fn rewards(State(state): State<AppState>) -> Json<dto::Rewards> {
    Json(state.catalog.reward_listing())
}

pub async fn judges(State(state): State<AppState>) -> Json<dto::Judges> {
    Json(state.catalog.judge_listing())
}

pub async fn mcp_servers(State(state): State<AppState>) -> Json<dto::McpServers> {
    Json(state.catalog.mcp_listing())
}

pub async fn environments(State(state): State<AppState>) -> Json<dto::Environments> {
    Json(state.catalog.environment_listing())
}

/// `POST /v1/preflight` - builds the training graph for a throwaway adapter and
/// returns the runtime's report.
///
/// Serialized with the runs by the shared device semaphore, and run on a blocking
/// thread: a `Trainer` carries raw FFI pointers, never crosses an OS-thread
/// boundary, and loading a model is seconds of synchronous work that must not sit
/// on an async worker.
pub async fn preflight(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<dto::PreflightRequest>,
) -> ApiResult<Json<dto::PreflightResponse>> {
    let model = state.resolve_path(&request.model, "/model")?;
    let device = parse_device(request.device.as_deref())?;
    let targets = parse_targets(request.targets.as_deref())?;
    let profile = crate::resolve::execution_profile(&state, &model, device).await?;
    let profile_fingerprint = profile.fingerprint();

    let permit = state
        .device
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the device queue is closed"))?;
    let probe = state.probe.clone();
    let report = tokio::task::spawn_blocking(move || {
        let outcome = probe.preflight(&model, device, targets, profile_fingerprint);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("preflight task failed: {error}")))??;

    report
        .validate()
        .map_err(|error| ApiError::internal(format!("invalid preflight report: {error}")))?;
    Ok(Json(dto::PreflightResponse { report }))
}

fn parse_device(value: Option<&str>) -> ApiResult<Device> {
    match value {
        None => Ok(Device::Auto),
        // Reuses the CLI's own parser, so `--device` and this field cannot
        // diverge on what they accept.
        Some(text) => text.parse::<Device>().map_err(|error| {
            ApiError::from(error).with_field(
                "/device",
                ErrorCode::InvalidValue,
                "expected auto, cpu, or gpu",
            )
        }),
    }
}

fn parse_targets(values: Option<&[String]>) -> ApiResult<TargetSet> {
    match values {
        None => Ok(TargetSet::Auto),
        Some([]) => Err(
            ApiError::invalid("targets must not be an empty list").with_field(
                "/targets",
                ErrorCode::InvalidValue,
                "omit the field to use the automatic profile",
            ),
        ),
        Some(values) => retrograd_config::parse_targets(values).map_err(|error| {
            ApiError::from(error).with_field("/targets", ErrorCode::InvalidValue, "unknown target")
        }),
    }
}
