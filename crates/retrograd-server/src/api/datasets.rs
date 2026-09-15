//! `POST/GET /v1/datasets`, `GET /v1/datasets/{id}`, `GET.../preview`,
//! `POST.../tokenize`, `DELETE /v1/datasets/{id}`.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use futures_util::StreamExt;
use http::StatusCode;
use serde::Deserialize;

use crate::datasets::{BodyError, DatasetMeta, IngestError, IngestOptions, Ingested};
use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::extract::Json as RequestJson;
use crate::state::AppState;

const DEFAULT_PAGE: usize = 50;
const DEFAULT_PREVIEW: usize = 5;
/// Enough to fix a file in one pass without shipping back a body sized to
/// however wrong the file is.
const MAX_LINE_ERRORS_SHOWN: usize = 20;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateQuery {
    #[serde(default)]
    pub name: Option<String>,
    /// `auto` (default) | `text` | `jsonl`.
    #[serde(default)]
    pub format: Option<String>,
}

/// `POST /v1/datasets`
///
/// The body is the dataset itself - `application/x-ndjson` or `text/plain`,
/// streamed straight to disk while its `sha256` is computed. `name` and
/// `format` travel as query parameters: there is no separate metadata channel
/// for a raw-body upload, and putting them in the body would make the body no
/// longer be the dataset.
pub async fn create(
    State(state): State<AppState>,
    Query(query): Query<CreateQuery>,
    headers: http::HeaderMap,
    body: axum::body::Body,
) -> ApiResult<axum::response::Response> {
    let format =
        crate::datasets::parse_format_hint(query.format.as_deref()).map_err(ingest_problem)?;
    let stream = body
        .into_data_stream()
        .map(|chunk| chunk.map_err(body_error));
    let outcome = state
        .datasets
        .ingest(
            stream,
            IngestOptions {
                format,
                name: query.name,
                declared_bytes: declared_bytes(&headers),
                max_dataset_bytes: Some(state.config.max_dataset_bytes()),
                max_datasets_bytes: Some(state.config.max_datasets_bytes()),
                idle_timeout: Some(state.config.upload_idle_timeout()),
            },
        )
        .await
        .map_err(ingest_problem)?;

    let status = match &outcome {
        // A repeat of the same content is answered like any other idempotent
        // creation ('s convention for `POST /v1/runs`'s own
        // `Idempotency-Key`): `200`, not `201` - nothing new was made.
        Ingested::Created(_) => StatusCode::CREATED,
        Ingested::AlreadyExists(_) => StatusCode::OK,
    };
    Ok((status, Json(dto::DatasetView::from(outcome.meta()))).into_response())
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
}

/// `GET /v1/datasets`
pub async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> ApiResult<Json<dto::DatasetListing>> {
    let mut items = state.datasets.list();
    if let Some(cursor) = &query.cursor {
        // Everything strictly after the cursor. An unknown cursor yields an
        // empty page rather than the first one, the same rule `GET /v1/runs`
        // follows.
        match items.iter().position(|item| &item.id == cursor) {
            Some(index) => {
                items.drain(..=index);
            }
            None => {
                items.drain(..);
            }
        }
    }
    let limit = query.limit.unwrap_or(DEFAULT_PAGE).clamp(1, 500);
    let next_cursor = (items.len() > limit)
        .then(|| items.get(limit - 1).map(|item| item.id.clone()))
        .flatten();
    items.truncate(limit);
    Ok(Json(dto::DatasetListing {
        datasets: items.iter().map(dto::DatasetView::from).collect(),
        next_cursor,
    }))
}

/// `GET /v1/datasets/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::DatasetView>> {
    let meta = lookup(&state, &id)?;
    Ok(Json(dto::DatasetView::from(&meta)))
}

/// `DELETE /v1/datasets/{id}`
///
/// Deletion is unconditional because there is no reverse index from a dataset
/// to the runs that use it.
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let meta = lookup(&state, &id)?;
    state.datasets.delete(&meta.id);
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /v1/datasets/{id}/preview`
pub async fn preview(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<PreviewQuery>,
) -> ApiResult<Json<dto::DatasetPreview>> {
    let meta = lookup(&state, &id)?;
    let limit = query.limit.unwrap_or(DEFAULT_PREVIEW).clamp(1, 100);
    let path = state.datasets.data_path(&meta);
    let examples = preview_examples(&path, meta.data_format(), limit).map_err(|error| {
        ApiError::from(error).with_field("/id", ErrorCode::InvalidValue, "could not be read")
    })?;
    Ok(Json(dto::DatasetPreview {
        id: meta.id,
        format: meta.format,
        examples,
    }))
}

/// `POST /v1/datasets/{id}/tokenize`
///
/// Measures the dataset's real per-example lengths with `model`'s own tokenizer
/// and chat template, caches them on the dataset's card, and answers what a few
/// common contexts would truncate. This is the answer to
/// "what `n_ctx` do I need?" before there is a plan to read it off - and it is
/// what lets a later `/v1/plan` drop the character heuristic's deliberate
/// over-count.
///
/// Runs behind the *probe* pool rather than the device semaphore, for the same
/// reason a geometry read does: `tokenize_lengths` loads the model on the CPU
/// and never touches a device, so serializing it with the runs would block
/// every measurement behind a training job it has nothing to do with.
pub async fn tokenize(
    State(state): State<AppState>,
    Path(id): Path<String>,
    RequestJson(request): RequestJson<dto::TokenizeRequest>,
) -> ApiResult<Json<dto::DatasetTokenization>> {
    let meta = lookup(&state, &id)?;
    let format = meta.data_format().ok_or_else(|| {
        ApiError::invalid(format!(
            "dataset {id} was stored as '{}', which this server can no longer read",
            meta.format
        ))
        .with_field("/id", ErrorCode::UnsupportedFormat, "unknown stored format")
    })?;
    let model_path = state.resolve_path(&request.model.to_string_lossy(), "/model")?;
    let data_path = state.datasets.data_path(&meta);

    // The geometry decides the cache key, so it is read first - and it is
    // cheap next to the tokenization that follows.
    let model = crate::resolve::geometry(&state, &model_path, retrograd_core::Device::Cpu).await?;
    let model_fingerprint = state.model_fingerprint(&model_path).await?;
    let tokenizer_key = crate::datasets::tokenizer_key(&model, &model_fingerprint);
    if let Some(cached) = meta.tokenized_for(&tokenizer_key) {
        // Already measured against this tokenizer: the lengths cannot have
        // changed, because the content behind a `ds_` id cannot.
        return Ok(Json(tokenization(&meta.id, &tokenizer_key, cached)));
    }

    let permit = state
        .probes
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the probe queue is closed"))?;
    let probe = state.probe.clone();
    let lengths = tokio::task::spawn_blocking(move || {
        let outcome = probe.tokenize_lengths(&model_path, &data_path, format);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("the tokenizer task failed: {error}")))?
    .map_err(|error| {
        ApiError::from(error).with_field(
            "/model",
            ErrorCode::InvalidValue,
            "could not tokenize this dataset",
        )
    })?;

    let stored = state
        .datasets
        .record_tokenized(&meta.id, &tokenizer_key, lengths)
        .map_err(|error| ApiError::internal(format!("could not store the measurement: {error}")))?
        .ok_or_else(|| {
            ApiError::new(
                ProblemKind::Conflict,
                format!("dataset {id} was deleted while it was being tokenized"),
            )
        })?;
    Ok(Json(tokenization(&meta.id, &tokenizer_key, &stored)))
}

fn tokenization(
    id: &str,
    tokenizer_key: &str,
    stored: &crate::datasets::TokenizedStats,
) -> dto::DatasetTokenization {
    dto::DatasetTokenization {
        id: id.to_string(),
        tokenizer: tokenizer_key.to_string(),
        stats: dto::DatasetStatsView {
            measured: stored.stats.measured,
            total_tokens: stored.stats.total,
            p50: stored.stats.p50,
            p90: stored.stats.p90,
            p99: stored.stats.p99,
            max: stored.stats.max,
        },
        truncation: stored.truncation.clone(),
    }
}

/// The first `limit` non-empty lines, read one at a time: a preview of five
/// examples must not depend on whether the file behind them is five kilobytes
/// or a gigabyte.
fn preview_examples(
    path: &std::path::Path,
    format: Option<retrograd_dataset::DataFormat>,
    limit: usize,
) -> retrograd_core::Result<Vec<serde_json::Value>> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)?;
    let mut examples = Vec::with_capacity(limit);
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        examples.push(match format {
            Some(retrograd_dataset::DataFormat::Text) => serde_json::Value::String(line),
            // Chat-JSONL, or an unsupported format: every line here already
            // survived ingestion's
            // validation, so parsing it back as JSON is expected to work - and
            // a line that somehow does not is shown raw rather than hidden.
            _ => serde_json::from_str(&line).unwrap_or(serde_json::Value::String(line)),
        });
        if examples.len() == limit {
            break;
        }
    }
    Ok(examples)
}

/// The size the client announced, when it announced one and it parses.
///
/// A claim, not a measurement - `ingest` treats it as a way to refuse early,
/// never as the size it records.
fn declared_bytes(headers: &http::HeaderMap) -> Option<u64> {
    headers
        .get(http::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Classifies a failed body chunk *here*, where the transport's own error type
/// is still in hand.
///
/// The body-limit layer rejects an oversized upload by failing the stream, so
/// without this the only trace of a `413` downstream would be the wording of a
/// message `http-body-util` is free to change. `LengthLimitError` is looked for
/// through the whole source chain: axum wraps it, and how deeply is not part of
/// anyone's contract.
fn body_error(error: axum::Error) -> BodyError {
    let error = error.into_inner();
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error.as_ref());
    while let Some(current) = source {
        if current.is::<http_body_util::LengthLimitError>() {
            return BodyError::TooLarge;
        }
        source = current.source();
    }
    BodyError::Failed(error.to_string())
}

fn lookup(state: &AppState, id: &str) -> ApiResult<DatasetMeta> {
    state
        .datasets
        .get(id)
        .ok_or_else(|| ApiError::not_found(format!("no dataset with id {id}")))
}

/// Maps a storage failure onto its public API problem type.
fn ingest_problem(error: IngestError) -> ApiError {
    match error {
        IngestError::TooLarge { max_bytes } => ApiError::new(
            ProblemKind::PayloadTooLarge,
            format!("the dataset is larger than the {max_bytes} byte limit this server accepts"),
        ),
        IngestError::QuotaExceeded {
            requested,
            used,
            quota,
        } => ApiError::new(ProblemKind::PayloadTooLarge, error.to_string()).with_meta(
            serde_json::json!({
                "requested_bytes": requested,
                "used_bytes": used,
                "quota_bytes": quota,
            }),
        ),
        IngestError::UnsupportedFormat(message) => ApiError::invalid(message).with_field(
            "/format",
            ErrorCode::UnsupportedFormat,
            "expected auto, text or jsonl",
        ),
        IngestError::Unreadable(inner) => ApiError::from(inner).with_field(
            "/",
            ErrorCode::InvalidValue,
            "the dataset could not be read",
        ),
        IngestError::Body(message) => {
            ApiError::invalid(format!("the upload was interrupted: {message}"))
        }
        IngestError::Inactive { .. } => ApiError::new(ProblemKind::Timeout, error.to_string()),
        IngestError::Io(inner) => ApiError::internal(inner.to_string()),
        IngestError::Invalid(validation) => {
            let total = validation.total;
            let shown: Vec<_> = validation
                .errors
                .into_iter()
                .take(MAX_LINE_ERRORS_SHOWN)
                .map(|error| {
                    serde_json::json!({
                        "line": error.line,
                        "code": error.kind.as_str(),
                        "message": error.message,
                    })
                })
                .collect();
            let errors_shown = shown.len();
            ApiError::new(ProblemKind::InvalidRequest, "the dataset failed validation").with_meta(
                serde_json::json!({
                    "line_errors": shown,
                    "errors_shown": errors_shown,
                    "errors_total": total,
                }),
            )
        }
    }
}
