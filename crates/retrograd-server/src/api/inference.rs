//! Asking a live run a question: `evaluate` and `generate`.
//!
//! These are the only two routes that **wait**. Everything else in `control.rs`
//! queues a command and answers where the run is, because everything else has an
//! observable effect. An evaluation has a result, and a result has to come back
//! on the request that asked for it.
//!
//! Both go down the same command channel the control plane uses, and are executed
//! inside the progress callback, for this reason: there is one `Trainer`,
//! it never leaves its thread, and nothing else may touch the weights while the
//! loop has them. So the latency is bounded by one iteration - an SFT epoch, or a
//! rollout update - and the reply carries the `global_step` it was taken at, so a
//! caller knows *which* adapter answered rather than assuming it was the latest.
//!
//! One consequence worth stating: a paused run answers both. The model is loaded
//! and idle, which is the cheapest moment there is to ask it something.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use retrograd_run::GenerationRequest;
use tokio::sync::oneshot;

use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::extract::Json as RequestJson;
use crate::runtime::RunHandle;
use crate::runtime::control::{At, Refused, RunCommand};
use crate::state::AppState;

/// Sampling defaults for an ad-hoc generation.
///
/// Fixed here rather than taken from the run's own sampling configuration: an SFT
/// run has none, and reading a rollout run's would make the same request behave
/// differently per algorithm. A caller who cares sends the values.
const DEFAULT_MAX_NEW_TOKENS: u32 = 128;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_SEED: u32 = 1;

/// `POST /v1/runs/{id}/evaluate`
pub async fn evaluate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Bytes,
) -> ApiResult<Json<dto::EvaluationResult>> {
    // An empty body is the request. Parsing one when it is there anyway keeps
    // `deny_unknown_fields` in force, so a client that sent `{"dataset": …}`
    // thinking it would be honoured gets told it would not.
    if !body.iter().all(u8::is_ascii_whitespace) {
        crate::extract::from_json_slice::<dto::EvaluateRequest>(&body)?;
    }
    let handle = super::runs::lookup(&state, &id)?;
    require_live(&handle, "evaluate")?;
    if !handle.record.controls.has_evaluation {
        return Err(ApiError::new(
            ProblemKind::Conflict,
            "this run has no evaluation dataset, so there is nothing to evaluate against",
        )
        .with_field(
            "/evaluation",
            ErrorCode::Conflict,
            "absent from the run's configuration",
        ));
    }

    let (reply, answer) = oneshot::channel();
    dispatch(&handle, RunCommand::Evaluate(reply))?;
    let (at, evaluation) = wait(&state, answer, "evaluation").await?;
    Ok(Json(dto::EvaluationResult {
        id: handle.id.to_string(),
        iteration: at.iteration,
        global_step: at.global_step,
        loss: evaluation.loss,
        perplexity: evaluation.perplexity,
        mean_reward: evaluation.mean_reward,
        reward_min: evaluation.reward_min,
        reward_max: evaluation.reward_max,
        examples: evaluation.examples,
    }))
}

/// `POST /v1/runs/{id}/generate`
pub async fn generate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    RequestJson(request): RequestJson<dto::GenerateRequest>,
) -> ApiResult<Json<dto::GenerationResult>> {
    let handle = super::runs::lookup(&state, &id)?;
    require_live(&handle, "generate from")?;
    let request = validate(request)?;

    let (reply, answer) = oneshot::channel();
    dispatch(&handle, RunCommand::Generate(Box::new(request), reply))?;
    let (at, output) = wait(&state, answer, "generation").await?;
    Ok(Json(dto::GenerationResult {
        id: handle.id.to_string(),
        iteration: at.iteration,
        global_step: at.global_step,
        text: output.text,
        prompt_tokens: output.prompt_tokens,
        tokens: output.tokens,
        base_text: output.base_text,
    }))
}

/// Fills in the defaults and refuses what the sampler cannot use.
///
/// Bounds checked here rather than in the runtime because a rejected request must
/// be a 422 the caller can fix, not an error surfacing from inside a training
/// callback several seconds later.
fn validate(request: dto::GenerateRequest) -> ApiResult<GenerationRequest> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid("the prompt is empty").with_field(
            "/prompt",
            ErrorCode::InvalidValue,
            "at least one non-blank character",
        ));
    }
    let max_new_tokens = request.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if max_new_tokens == 0 {
        return Err(
            ApiError::invalid("max_new_tokens must be greater than zero").with_field(
                "/max_new_tokens",
                ErrorCode::OutOfRange,
                "not a positive integer",
            ),
        );
    }
    let temperature = request.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(
            ApiError::invalid("temperature must be a finite, non-negative number").with_field(
                "/temperature",
                ErrorCode::OutOfRange,
                "out of range",
            ),
        );
    }
    let top_p = request.top_p.unwrap_or(DEFAULT_TOP_P);
    if !(0.0..=1.0).contains(&top_p) {
        return Err(
            ApiError::invalid("top_p must be between 0 and 1").with_field(
                "/top_p",
                ErrorCode::OutOfRange,
                "out of range",
            ),
        );
    }
    Ok(GenerationRequest {
        prompt: request.prompt,
        chat: request.chat,
        max_new_tokens,
        temperature,
        top_p,
        seed: request.seed.unwrap_or(DEFAULT_SEED),
        include_base: request.include_base,
    })
}

/// Only a run whose loop is turning, or paused inside its callback, can answer.
///
/// A `starting` run is deliberately excluded even though it has a thread: its
/// model is still loading and its command would sit unanswered until the first
/// callback, which on a large model is a minute of a held-open request.
fn require_live(handle: &RunHandle, action: &str) -> ApiResult<()> {
    match handle.status() {
        dto::RunStatus::Running | dto::RunStatus::Pausing | dto::RunStatus::Paused => Ok(()),
        status => Err(ApiError::new(
            ProblemKind::Conflict,
            format!("cannot {action} a run that is {}", status.as_str()),
        )
        .with_field(
            "/status",
            ErrorCode::Conflict,
            "the run is not executing its loop",
        )),
    }
}

fn dispatch(handle: &Arc<RunHandle>, command: RunCommand) -> ApiResult<()> {
    match handle.control().map(|control| control.send(command)) {
        Some(Ok(())) => return Ok(()),
        // Commands are arriving faster than the run reads them; the run itself
        // is healthy, so this is a retry rather than a refusal.
        Some(Err(Refused::Saturated)) => {
            return Err(ApiError::new(
                ProblemKind::DeviceBusy,
                "this run has too many pending commands",
            ));
        }
        Some(Err(Refused::Gone)) | None => {}
    }
    Err(ApiError::new(
        ProblemKind::Conflict,
        "this run is no longer accepting commands",
    ))
}

/// Waits for the run's next callback, and no longer than the configured bound.
///
/// The timeout is what keeps this route from being a way to pin a connection for
/// hours: a rollout update is minutes, a stuck one is forever. Dropping the
/// receiver on the way out is not just tidiness - the run checks whether anyone is
/// still listening before spending a forward pass, so an abandoned request costs
/// nothing.
async fn wait<T>(
    state: &AppState,
    answer: oneshot::Receiver<retrograd_core::Result<(At, T)>>,
    what: &str,
) -> ApiResult<(At, T)> {
    let bound = state.config.command_timeout();
    match tokio::time::timeout(bound, answer).await {
        Ok(Ok(Ok(result))) => Ok(result),
        // The run itself refused: no evaluation dataset by the time it looked, a
        // prompt that tokenized to nothing, a sampler error.
        Ok(Ok(Err(error))) => Err(ApiError::from(error)),
        // The sender was dropped without answering: the run ended between the
        // command being queued and its next callback.
        Ok(Err(_)) => Err(ApiError::new(
            ProblemKind::Conflict,
            format!("the run finished before it could answer the {what} request"),
        )),
        Err(_) => Err(ApiError::new(
            ProblemKind::Timeout,
            format!(
                "the run did not reach a progress callback within {}s, so the {what} request \
                 was not served; it is applied at a callback and nowhere else",
                bound.as_secs()
            ),
        )),
    }
}
