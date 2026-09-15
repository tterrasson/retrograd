//! Driving a live run: pause, resume, cancel, checkpoint, adjust.
//!
//! Every route here is the same three steps: check that the run's state allows
//! what is being asked (409 if not), queue a command, and answer with where the
//! run is and when the request lands. None of them waits for the run to act.
//! That is not a shortcut - a rollout update can be minutes long, and a request
//! held open for one would time out on the client long before the run reached its
//! next callback. The effect is observable where every other effect of a run is:
//! in `GET /v1/runs/{id}` and in the event stream.
//!
//! Two properties the tests pin down. **Repeating a command is not an error**: a
//! client that retried after a timeout must not get a 409 for a pause that
//! already took, so re-asserting a state is a 200. And **a cancellation is never
//! weakened** by a later, gentler one - that rule lives in the control channel,
//! where the two meet.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use retrograd_config::CheckpointMode;

use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::extract::Json as RequestJson;
use crate::runtime::RunHandle;
use crate::runtime::control::{Adjustments, CancelAt, RunCommand};
use crate::state::AppState;

/// `POST /v1/runs/{id}/pause`
///
/// The run stops at its next progress callback and stays there, **holding the
/// device**: weights, KV caches and optimizer state are all still allocated.
/// That is the deliberate V1 trade - resuming is instant and the memory is
/// not available to anything else meanwhile - and `holds_device` in `GET
/// /v1/runs/{id}` is how a client sees it.
pub async fn pause(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::CommandAccepted>> {
    let handle = super::runs::lookup(&state, &id)?;
    match handle.status() {
        // Already there, or already on the way. A retry is not a conflict.
        dto::RunStatus::Paused | dto::RunStatus::Pausing => return Ok(Json(accepted(&handle))),
        dto::RunStatus::Running | dto::RunStatus::Starting => {}
        status => return Err(illegal("pause", status)),
    }
    send(&handle, RunCommand::Pause)?;
    handle.transition(dto::RunStatus::Pausing);
    Ok(Json(accepted(&handle)))
}

/// `POST /v1/runs/{id}/resume`
pub async fn resume(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::CommandAccepted>> {
    let handle = super::runs::lookup(&state, &id)?;
    match handle.status() {
        dto::RunStatus::Running | dto::RunStatus::Starting => return Ok(Json(accepted(&handle))),
        dto::RunStatus::Paused | dto::RunStatus::Pausing => {}
        status => return Err(illegal("resume", status)),
    }
    send(&handle, RunCommand::Resume)?;
    // The run itself moves back to `running` when its callback wakes; announcing
    // it here would show a state the loop has not reached.
    Ok(Json(accepted(&handle)))
}

/// `POST /v1/runs/{id}/cancel`
pub async fn cancel(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Bytes,
) -> ApiResult<Json<dto::CommandAccepted>> {
    // An empty body is the common case - "stop this run" - and reading the body
    // as bytes is what lets it stay empty: an extractor would demand `{}` from
    // every client, or a `Content-Type` from a `curl -X POST` that has neither.
    let request = if body.iter().all(u8::is_ascii_whitespace) {
        dto::CancelRequest::default()
    } else {
        crate::extract::from_json_slice(&body)?
    };
    if request.checkpoint && request.at == dto::CancelAt::Now {
        return Err(ApiError::invalid(
            "a checkpoint needs a resumable boundary; cancel with at=\"boundary\" to keep one",
        )
        .with_field(
            "/at",
            ErrorCode::Conflict,
            "cannot checkpoint an immediate cancellation",
        ));
    }
    let handle = super::runs::lookup(&state, &id)?;
    let status = handle.status();
    match status {
        dto::RunStatus::Cancelling | dto::RunStatus::Cancelled => {
            return Ok(Json(accepted(&handle)));
        }
        // Nothing is loaded yet, so there is nothing to unwind and no callback to
        // wait for. The queued worker sees the terminal state and returns.
        dto::RunStatus::Queued | dto::RunStatus::Resolving => {
            handle.transition(dto::RunStatus::Cancelled);
            handle.emit(dto::RunEventPayload::Terminal {
                status: dto::RunStatus::Cancelled,
                error: None,
            });
            return Ok(Json(dto::CommandAccepted {
                applies_at_iteration: None,
                ..accepted(&handle)
            }));
        }
        dto::RunStatus::Starting
        | dto::RunStatus::Running
        | dto::RunStatus::Pausing
        | dto::RunStatus::Paused => {}
        status => return Err(illegal("cancel", status)),
    }
    send(
        &handle,
        RunCommand::Cancel {
            at: match request.at {
                dto::CancelAt::Boundary => CancelAt::Boundary,
                dto::CancelAt::Now => CancelAt::Now,
            },
            checkpoint: request.checkpoint,
        },
    )?;
    handle.transition(dto::RunStatus::Cancelling);
    Ok(Json(accepted(&handle)))
}

/// `POST /v1/runs/{id}/checkpoints`
///
/// Asks for a checkpoint at the next boundary - never "now": a snapshot taken
/// between two boundaries would replay or skip work on resume. The
/// `checkpoint` event says where it landed.
pub async fn checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::CommandAccepted>> {
    let handle = super::runs::lookup(&state, &id)?;
    require_live(&handle, "checkpoint")?;
    if !handle.record.controls.has_checkpoint {
        return Err(ApiError::new(
            ProblemKind::Conflict,
            "this run has no checkpoint directory, so there is nowhere to write one",
        )
        .with_field(
            "/checkpoint",
            ErrorCode::Conflict,
            "absent from the run's configuration",
        ));
    }
    send(&handle, RunCommand::Checkpoint)?;
    Ok(Json(accepted(&handle)))
}

/// `PATCH /v1/runs/{id}`
///
/// The whitelist is the request type: anything outside it fails to deserialize.
/// What is left to check here is whether *this* run has the thing being adjusted
/// (a cadence needs an evaluation dataset, a step schedule needs a checkpoint
/// directory), and that is why the resolved configuration's capabilities are kept
/// beside the run rather than re-parsed out of its rendered JSON.
pub async fn patch(
    State(state): State<AppState>,
    Path(id): Path<String>,
    RequestJson(request): RequestJson<dto::PatchRequest>,
) -> ApiResult<Json<dto::CommandAccepted>> {
    let handle = super::runs::lookup(&state, &id)?;
    require_live(&handle, "adjust")?;
    let adjustments = validate(&handle, &request)?;
    if adjustments.is_empty() {
        return Err(ApiError::invalid("the request adjusts nothing").with_field(
            "",
            ErrorCode::InvalidValue,
            "at least one whitelisted field is required",
        ));
    }
    let applied = adjustments.paths();
    send(&handle, RunCommand::Adjust(adjustments))?;
    Ok(Json(dto::CommandAccepted {
        applied,
        ..accepted(&handle)
    }))
}

/// Turns the request into the typed whitelist, refusing what this run cannot do.
fn validate(handle: &RunHandle, request: &dto::PatchRequest) -> ApiResult<Adjustments> {
    let controls = handle.record.controls;
    let mut adjustments = Adjustments::default();

    if let Some(training) = request.training
        && let Some(rate) = training.lr
    {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(
                ApiError::invalid("training.lr must be greater than zero").with_field(
                    "/training/lr",
                    ErrorCode::OutOfRange,
                    "not a positive, finite number",
                ),
            );
        }
        adjustments.learning_rate = Some(rate);
    }

    if let Some(evaluation) = request.evaluation {
        if (evaluation.every_iterations.is_some() || evaluation.patience.is_some())
            && !controls.has_evaluation
        {
            return Err(ApiError::new(
                ProblemKind::Conflict,
                "this run has no evaluation dataset, so its evaluation schedule cannot be changed",
            )
            .with_field(
                "/evaluation",
                ErrorCode::Conflict,
                "absent from the run's configuration",
            ));
        }
        if let Some(every) = evaluation.every_iterations {
            if every == 0 {
                return Err(ApiError::invalid(
                    "evaluation.every_iterations must be greater than zero",
                )
                .with_field(
                    "/evaluation/every_iterations",
                    ErrorCode::OutOfRange,
                    "not a positive integer",
                ));
            }
            adjustments.evaluation_every_iterations = Some(every);
        }
        adjustments.evaluation_patience = evaluation.patience;
    }

    if let Some(checkpoint) = &request.checkpoint {
        if (checkpoint.every_steps.is_some() || checkpoint.mode.is_some())
            && !controls.has_checkpoint
        {
            return Err(ApiError::new(
                ProblemKind::Conflict,
                "this run writes no checkpoints, so its checkpoint schedule cannot be changed",
            )
            .with_field(
                "/checkpoint",
                ErrorCode::Conflict,
                "absent from the run's configuration",
            ));
        }
        if let Some(every) = checkpoint.every_steps {
            if every == 0 {
                return Err(
                    ApiError::invalid("checkpoint.every_steps must be greater than zero")
                        .with_field(
                            "/checkpoint/every_steps",
                            ErrorCode::OutOfRange,
                            "not a positive integer",
                        ),
                );
            }
            adjustments.checkpoint_every_steps = Some(every);
        }
        if let Some(mode) = &checkpoint.mode {
            let parsed = match mode.as_str() {
                "steps" => CheckpointMode::Steps,
                "best_eval" => CheckpointMode::BestEval,
                "steps_and_best_eval" => CheckpointMode::StepsAndBestEval,
                _ => {
                    return Err(ApiError::invalid(
                        "checkpoint.mode must be steps, best_eval, or steps_and_best_eval",
                    )
                    .with_field(
                        "/checkpoint/mode",
                        ErrorCode::InvalidValue,
                        "unknown mode",
                    ));
                }
            };
            if parsed.includes_best_eval() && !controls.has_evaluation {
                return Err(ApiError::new(
                    ProblemKind::Conflict,
                    "checkpoint.mode includes best_eval but this run has no evaluation dataset",
                )
                .with_field(
                    "/checkpoint/mode",
                    ErrorCode::Conflict,
                    "requires an evaluation dataset",
                ));
            }
            adjustments.checkpoint_mode = Some(parsed);
        }
    }

    Ok(adjustments)
}

/// A run with a thread to command. A queued one has none yet, and a finished one
/// never will.
fn require_live(handle: &RunHandle, action: &str) -> ApiResult<()> {
    match handle.status() {
        dto::RunStatus::Running | dto::RunStatus::Pausing | dto::RunStatus::Paused => Ok(()),
        status => Err(illegal(action, status)),
    }
}

fn send(handle: &Arc<RunHandle>, command: RunCommand) -> ApiResult<()> {
    let sent = handle
        .control()
        .is_some_and(|control| control.send(command));
    if sent {
        return Ok(());
    }
    // The channel is closed, so the run's thread is gone even though its state
    // has not caught up yet. Reporting success would promise something nothing
    // is left to do.
    Err(ApiError::new(
        ProblemKind::Conflict,
        "this run is no longer accepting commands",
    ))
}

/// 409 with the state that refused, plus the escape hatch: what
/// cannot be changed on a live run can be changed on a fork of it.
fn illegal(action: &str, status: dto::RunStatus) -> ApiError {
    ApiError::new(
        ProblemKind::Conflict,
        format!("cannot {action} a run that is {}", status.as_str()),
    )
    .with_field(
        "/status",
        ErrorCode::Conflict,
        "start a new run with fork_from instead",
    )
}

fn accepted(handle: &RunHandle) -> dto::CommandAccepted {
    let state = handle.state();
    dto::CommandAccepted {
        id: handle.id.to_string(),
        status: state.status,
        // The next iteration: a run is only interruptible at its progress
        // callback, so nothing a client asks for lands before the one after this.
        applies_at_iteration: Some(state.progress.iteration.saturating_add(1)),
        applied: Vec::new(),
    }
}
