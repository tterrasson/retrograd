//! `GET /v1/runs/{id}/trajectories…`: what a rollout run generated, read back
//! from its `observe.jsonl`.
//!
//! The handlers do not know where an export lives: they ask a
//! [`TrajectorySource`]. The control plane answers with the run's `[observe]`
//! directory; `retrograd-server view` answers with the one directory it was
//! given. That is the whole difference between the two, so the routes are the
//! same code in both.
//!
//! Every read goes through an [`ObserveIndex`] kept per directory and refreshed
//! on each request - the cost of a request is the bytes appended since the last
//! one, plus the lines it actually shows.

use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::routing::get;
use axum::{Json, Router};
use http::StatusCode;
use retrograd_observe::reader::{ObserveIndex, ReadError};
use serde::Deserialize;
use tower_http::timeout::TimeoutLayer;

use crate::dto;
use crate::error::{ApiError, ApiResult, ProblemKind};
use crate::lock::Recover as _;
use crate::runtime::RunRegistry;
use crate::runtime::registry::model_file_name;

/// How many export indexes are kept. Each holds offsets and a few numbers per
/// rollout; eight open runs is more than one person looks at.
const CACHED_INDEXES: usize = 8;
/// Groups per page of `…/updates/{update}`.
const DEFAULT_GROUPS: usize = 50;
const MAX_GROUPS: usize = 500;
/// Characters of each prompt message in a listing.
const DEFAULT_PREVIEW_CHARS: usize = 400;

/// Where a run's export is.
pub trait TrajectorySource: Send + Sync + 'static {
    /// The `[observe]` directory of `run`: `Ok(None)` for a run that exports
    /// nothing, an error for a run that does not exist.
    fn observe_dir(&self, run: &str) -> ApiResult<Option<PathBuf>>;
}

/// The control plane's runs.
pub struct RegistrySource(pub Arc<RunRegistry>);

impl TrajectorySource for RegistrySource {
    fn observe_dir(&self, run: &str) -> ApiResult<Option<PathBuf>> {
        let handle = run
            .parse::<uuid::Uuid>()
            .ok()
            .and_then(|id| self.0.get(&id))
            .ok_or_else(|| ApiError::not_found(format!("no run with id {run}")))?;
        Ok(handle.record.artifacts.observe.clone())
    }
}

/// One directory, served as the run [`LOCAL_RUN`].
pub struct DirectorySource(pub PathBuf);

/// The id of the one run `retrograd-server view` serves.
pub const LOCAL_RUN: &str = "local";

impl TrajectorySource for DirectorySource {
    /// A directory without a log has nothing to show - it is not an export,
    /// whatever it was meant to be.
    fn observe_dir(&self, run: &str) -> ApiResult<Option<PathBuf>> {
        if run == LOCAL_RUN {
            Ok(self
                .0
                .join("observe.jsonl")
                .is_file()
                .then(|| self.0.clone()))
        } else {
            Err(ApiError::not_found(format!(
                "no run with id {run}; this viewer serves '{LOCAL_RUN}'"
            )))
        }
    }
}

type SharedIndex = Arc<Mutex<ObserveIndex>>;

/// The handlers' state: where exports are, and the indexes already open.
#[derive(Clone)]
pub struct Trajectories {
    source: Arc<dyn TrajectorySource>,
    /// Most recently used last.
    indexes: Arc<Mutex<Vec<(PathBuf, SharedIndex)>>>,
}

impl Trajectories {
    pub fn new(source: impl TrajectorySource) -> Self {
        Self {
            source: Arc::new(source),
            indexes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The index of `directory`, opened or taken from the cache, refreshed.
    /// Blocking: called from the blocking pool.
    fn index(&self, directory: &FsPath) -> Result<SharedIndex, ReadError> {
        let cached = {
            let mut indexes = self.indexes.lock().recover();
            let position = indexes.iter().position(|(path, _)| path == directory);
            position.map(|position| {
                let entry = indexes.remove(position);
                let index = entry.1.clone();
                indexes.push(entry);
                index
            })
        };
        let index = match cached {
            Some(index) => {
                index.lock().recover().refresh()?;
                index
            }
            None => {
                let index = Arc::new(Mutex::new(ObserveIndex::open(directory)?));
                let mut indexes = self.indexes.lock().recover();
                if indexes.len() == CACHED_INDEXES {
                    indexes.remove(0);
                }
                indexes.push((directory.to_path_buf(), index.clone()));
                index
            }
        };
        Ok(index)
    }

    /// Runs `read` over the refreshed index of `run`'s export, off the async
    /// workers. `None` when the run exports nothing.
    async fn read<T: Send + 'static>(
        &self,
        run: &str,
        read: impl FnOnce(&ObserveIndex) -> ApiResult<T> + Send + 'static,
    ) -> ApiResult<Option<T>> {
        let Some(directory) = self.source.observe_dir(run)? else {
            return Ok(None);
        };
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let index = this.index(&directory)?;
            let index = index.lock().recover();
            read(&index).map(Some)
        })
        .await
        .map_err(|error| ApiError::internal(format!("the trajectory read failed: {error}")))?
    }
}

impl From<ReadError> for ApiError {
    fn from(error: ReadError) -> Self {
        // The file is there and cannot be read as an export of this version:
        // a state of the resource, not a malformed request. The paths it names
        // are trimmed on the way out like any other.
        ApiError::new(ProblemKind::Conflict, error.to_string())
    }
}

/// The three routes over `source`, with the request timeout every
/// finite route has.
pub fn router(source: impl TrajectorySource, timeout: Duration) -> Router {
    Router::new()
        .route("/runs/{id}/trajectories", get(overview))
        .route("/runs/{id}/trajectories/updates/{update}", get(update))
        .route(
            "/runs/{id}/trajectories/updates/{update}/groups/{group}",
            get(group),
        )
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            timeout,
        ))
        .with_state(Trajectories::new(source))
}

/// `GET /v1/runs/{id}/trajectories`
pub async fn overview(
    State(trajectories): State<Trajectories>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::TrajectoryOverview>> {
    let overview = trajectories
        .read(&id, |index| {
            let run = index.run();
            Ok(dto::TrajectoryOverview {
                observed: true,
                algorithm: run.map(|run| run.algorithm.clone()),
                model: run.map(|run| model_file_name(FsPath::new(&run.model))),
                every: run.and_then(|run| run.every),
                segments: u32::try_from(index.runs().len()).unwrap_or(u32::MAX),
                skipped: index.skipped(),
                updates: index
                    .updates()
                    .into_iter()
                    .map(dto::UpdateSummary::from)
                    .collect(),
            })
        })
        .await?;
    Ok(Json(overview.unwrap_or(dto::TrajectoryOverview {
        observed: false,
        algorithm: None,
        model: None,
        every: None,
        segments: 0,
        skipped: 0,
        updates: Vec::new(),
    })))
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateQuery {
    /// The index of the first group of the page, as the last page returned it.
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    /// Characters of each prompt message; `0` keeps them whole.
    #[serde(default)]
    pub preview_chars: Option<usize>,
}

/// `GET /v1/runs/{id}/trajectories/updates/{update}`
pub async fn update(
    State(trajectories): State<Trajectories>,
    Path((id, update)): Path<(String, String)>,
    Query(query): Query<UpdateQuery>,
) -> ApiResult<Json<dto::UpdateDetail>> {
    let number = update_number(&update)?;
    let start = match &query.cursor {
        Some(cursor) => cursor.parse::<usize>().map_err(|_| {
            ApiError::invalid("the cursor is not one this server returned").with_field(
                "/query/cursor",
                crate::error::ErrorCode::InvalidValue,
                "not a cursor",
            )
        })?,
        None => 0,
    };
    let limit = query.limit.unwrap_or(DEFAULT_GROUPS).clamp(1, MAX_GROUPS);
    let preview = match query.preview_chars.unwrap_or(DEFAULT_PREVIEW_CHARS) {
        0 => None,
        chars => Some(chars),
    };
    let detail = trajectories
        .read(&id, move |index| {
            let (entry, groups) = index.update(number).ok_or_else(|| missing_update(number))?;
            let segment = entry.segment;
            let next_cursor = (groups.len() > start + limit).then(|| (start + limit).to_string());
            let groups = groups
                .into_iter()
                .skip(start)
                .take(limit)
                .map(|group| {
                    let prompt = match index.prompt(segment, &group.prompt)? {
                        Some(prompt) => dto::PromptView::from_read(prompt, preview),
                        None => dto::PromptView::missing(group.prompt.clone()),
                    };
                    let members = group
                        .members
                        .into_iter()
                        .map(dto::MemberSummary::from)
                        .collect();
                    Ok(dto::GroupSummary::new(group.group, prompt, members))
                })
                .collect::<ApiResult<Vec<_>>>()?;
            Ok(dto::UpdateDetail {
                summary: entry.into(),
                groups,
                next_cursor,
            })
        })
        .await?
        .ok_or_else(|| not_observed(&id))?;
    Ok(Json(detail))
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupQuery {
    /// Only this member: a group of long trajectories can weigh megabytes.
    #[serde(default)]
    pub member: Option<usize>,
}

/// `GET /v1/runs/{id}/trajectories/updates/{update}/groups/{group}`
///
/// `{group}` is the group's index, or `-` for PPO, whose rollouts have none.
pub async fn group(
    State(trajectories): State<Trajectories>,
    Path((id, update, group)): Path<(String, String, String)>,
    Query(query): Query<GroupQuery>,
) -> ApiResult<Json<dto::GroupDetail>> {
    let number = update_number(&update)?;
    let group =
        match group.as_str() {
            "-" => None,
            other => Some(other.parse::<usize>().map_err(|_| {
                ApiError::not_found(format!("no group '{other}' in update {number}"))
            })?),
        };
    let detail = trajectories
        .read(&id, move |index| {
            let (entry, groups) = index.update(number).ok_or_else(|| missing_update(number))?;
            let found = groups
                .iter()
                .find(|candidate| candidate.group == group)
                .ok_or_else(|| {
                    ApiError::not_found(format!(
                        "update {number} has no group {}",
                        group.map_or_else(|| "-".to_string(), |group| group.to_string())
                    ))
                })?;
            let prompt = match index.prompt(entry.segment, &found.prompt)? {
                Some(prompt) => dto::PromptView::from_read(prompt, None),
                None => dto::PromptView::missing(found.prompt.clone()),
            };
            let members = index
                .rollouts(number, group, query.member)?
                .into_iter()
                .map(dto::MemberDetail::from)
                .collect();
            Ok(dto::GroupDetail {
                update: number,
                group,
                prompt,
                members,
            })
        })
        .await?
        .ok_or_else(|| not_observed(&id))?;
    Ok(Json(detail))
}

fn update_number(raw: &str) -> ApiResult<u32> {
    raw.parse::<u32>()
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| ApiError::not_found(format!("no update '{raw}'; updates count from 1")))
}

fn missing_update(number: u32) -> ApiError {
    ApiError::not_found(format!(
        "update {number} is not in this export, or a resume superseded it"
    ))
}

fn not_observed(id: &str) -> ApiError {
    ApiError::not_found(format!(
        "run {id} exports no trajectories; its overview says `observed: false`"
    ))
}
