//! What a run left on disk: checkpoints, artefacts, and the purge.
//!
//! Everything here is a *read of the filesystem*, not of the run's memory. That
//! is deliberate and it is what makes these routes useful: the run whose outputs
//! a client comes back for is a finished one, often from a previous process. So
//! the paths come from `run.json` - typed, restored at startup - and the contents
//! come from the directory as it is now. A checkpoint deleted by hand disappears
//! from the listing with nothing having to be told.
//!
//! The download route takes a **name from a closed inventory**, never a path.
//! `GET.../artifacts/adapter` resolves through the run's own record; there is no
//! parameter a client could point somewhere else, which is the only way this is
//! not an arbitrary file reader.

use std::path::{Path as FsPath, PathBuf};

use axum::Json;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use http::{StatusCode, header};
use retrograd_checkpoint as checkpoint;

use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::runtime::registry::RunArtifacts;
use crate::state::AppState;

/// Largest artefact served inline. An adapter is megabytes and an observe log
/// can be tens; a multi-gigabyte file is a path a client should read directly, and
/// streaming one through the control plane would tie up a connection for minutes.
const MAX_DOWNLOAD_BYTES: u64 = 512 * 1024 * 1024;

/// `GET /v1/runs/{id}/checkpoints`
pub async fn checkpoints(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::CheckpointListing>> {
    let handle = super::runs::lookup(&state, &id)?;
    let Some(directory) = handle.record.artifacts.checkpoint_directory.clone() else {
        // Not a 404: the run exists and has an answer, and the answer is that it
        // writes none. An empty list with no directory says exactly that.
        return Ok(Json(dto::CheckpointListing {
            directory: None,
            checkpoints: Vec::new(),
            latest: None,
        }));
    };
    let checkpoints = scan_checkpoints(&directory);
    // The same resolution `--resume` performs, so what a fork picks by default is
    // visible before the fork is made.
    let latest = retrograd_run::latest_checkpoint(&directory)
        .ok()
        .and_then(|path| checkpoint_id(&path));
    Ok(Json(dto::CheckpointListing {
        directory: Some(directory.display().to_string()),
        checkpoints,
        latest,
    }))
}

/// `GET /v1/runs/{id}/artifacts`
pub async fn list(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<dto::ArtifactListing>> {
    let handle = super::runs::lookup(&state, &id)?;
    let artifacts = inventory(&handle.record.artifacts, handle.state_directory())
        .into_iter()
        .map(|item| item.describe())
        .collect();
    Ok(Json(dto::ArtifactListing { artifacts }))
}

/// `GET /v1/runs/{id}/artifacts/{name}`
pub async fn download(
    State(state): State<AppState>,
    Path((id, name)): Path<(String, String)>,
) -> ApiResult<Response> {
    let handle = super::runs::lookup(&state, &id)?;
    let inventory = inventory(&handle.record.artifacts, handle.state_directory());
    let item = inventory
        .iter()
        .find(|item| item.name == name)
        .ok_or_else(|| {
            let known: Vec<&str> = inventory.iter().map(|item| item.name).collect();
            ApiError::not_found(format!(
                "this run has no artifact named '{name}'; it has {}",
                known.join(", ")
            ))
        })?;
    if item.directory {
        return Err(ApiError::new(
            ProblemKind::Conflict,
            format!("'{name}' is a directory; GET /v1/runs/{id}/checkpoints lists its contents"),
        ));
    }
    let bytes = item.bytes().ok_or_else(|| {
        ApiError::not_found(format!(
            "'{name}' is not on disk; a run that has not finished has not written its adapter yet"
        ))
    })?;
    if bytes > MAX_DOWNLOAD_BYTES {
        return Err(ApiError::new(
            ProblemKind::Conflict,
            format!(
                "'{name}' is {bytes} bytes, over the {MAX_DOWNLOAD_BYTES} byte inline limit; \
                 read it from the path the listing reports"
            ),
        ));
    }
    let body = std::fs::read(&item.path).map_err(|error| {
        ApiError::internal(format!("could not read the artifact '{name}': {error}"))
    })?;
    let filename = item
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| name.clone());
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, item.content_type.to_string()),
            // `attachment`, always: an artefact is a file to save, and serving a
            // client-influenced payload inline is how a control plane becomes an
            // XSS vector for whoever opens it in a browser.
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}\"", sanitize(&filename)),
            ),
        ],
        body,
    )
        .into_response())
}

/// `DELETE /v1/runs/{id}` - forget a finished run.
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let handle = super::runs::lookup(&state, &id)?;
    if !handle.status().is_terminal() {
        return Err(ApiError::new(
            ProblemKind::Conflict,
            format!(
                "cannot delete a run that is {}; cancel it first",
                handle.status().as_str()
            ),
        )
        .with_field("/status", ErrorCode::Conflict, "not terminal"));
    }
    // Only the server's own bookkeeping. Adapters, checkpoints and TensorBoard
    // logs are at paths the client chose and may well be shared between runs;
    // deleting a run's record must not delete a client's work.
    state.registry.purge(&handle.id);
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// The inventory
// ---------------------------------------------------------------------------

/// One named member of a run's closed artefact set.
struct Item {
    name: &'static str,
    path: PathBuf,
    directory: bool,
    content_type: &'static str,
}

impl Item {
    fn bytes(&self) -> Option<u64> {
        std::fs::metadata(&self.path)
            .ok()
            .filter(|meta| meta.is_file())
            .map(|meta| meta.len())
    }

    fn describe(&self) -> dto::ArtifactEntry {
        let bytes = if self.directory {
            directory_bytes(&self.path)
        } else {
            self.bytes()
        };
        dto::ArtifactEntry {
            name: self.name,
            path: self.path.display().to_string(),
            kind: if self.directory { "directory" } else { "file" },
            present: if self.directory {
                self.path.is_dir()
            } else {
                self.path.is_file()
            },
            downloadable: !self.directory && bytes.is_some_and(|bytes| bytes <= MAX_DOWNLOAD_BYTES),
            bytes,
        }
    }
}

/// The whole inventory, in a fixed order.
///
/// Fixed because the listing is a contract: a client scripting `artifacts[0]` is
/// wrong, but a client diffing two listings is not, and a set that reordered
/// itself would make every diff noise.
fn inventory(artifacts: &RunArtifacts, run_dir: &FsPath) -> Vec<Item> {
    let mut items = Vec::new();
    if let Some(adapter) = &artifacts.adapter {
        items.push(Item {
            name: "adapter",
            path: adapter.clone(),
            directory: false,
            content_type: "application/octet-stream",
        });
    }
    // The server's own two files. Always present, and the only artefacts that are
    // there for a run that never got as far as writing anything of its own.
    items.push(Item {
        name: "run",
        path: run_dir.join("run.json"),
        directory: false,
        content_type: "application/json",
    });
    items.push(Item {
        name: "events",
        path: run_dir.join("events.jsonl"),
        directory: false,
        content_type: "application/x-ndjson",
    });
    if let Some(observe) = &artifacts.observe {
        items.push(Item {
            name: "observe_log",
            path: observe.join("observe.jsonl"),
            directory: false,
            content_type: "application/x-ndjson",
        });
    }
    for (name, path) in [
        ("observe", &artifacts.observe),
        ("checkpoints", &artifacts.checkpoint_directory),
        ("tensorboard", &artifacts.tensorboard_directory),
        ("wandb_export", &artifacts.wandb_export_directory),
    ] {
        if let Some(path) = path {
            items.push(Item {
                name,
                path: path.clone(),
                directory: true,
                content_type: "application/octet-stream",
            });
        }
    }
    items
}

/// Sum of the regular files directly under `dir`, one level down included.
///
/// Not a full recursive walk: a checkpoint directory is one level of `.state`
/// directories, and an unbounded walk on a path a client chose is an unbounded
/// amount of work on a route that is polled.
fn directory_bytes(dir: &FsPath) -> Option<u64> {
    if !dir.is_dir() {
        return None;
    }
    let mut total = 0;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        match entry.metadata() {
            Ok(meta) if meta.is_file() => total += meta.len(),
            Ok(meta) if meta.is_dir() => {
                for child in std::fs::read_dir(entry.path()).into_iter().flatten() {
                    if let Ok(meta) = child.and_then(|child| child.metadata())
                        && meta.is_file()
                    {
                        total += meta.len();
                    }
                }
            }
            _ => {}
        }
    }
    Some(total)
}

/// Every `*.state` directory under `directory`, newest step first.
fn scan_checkpoints(directory: &FsPath) -> Vec<dto::CheckpointEntry> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(id) = checkpoint_id(&path) else {
            continue;
        };
        // Manifest only: `Checkpoint::read` would deserialize every AdamW moment
        // in the file, which for a listing of twenty snapshots is hundreds of
        // megabytes to print their step numbers.
        let manifest = checkpoint::read_manifest(&path).ok();
        let adapter = path.with_extension("gguf");
        found.push(dto::CheckpointEntry {
            kind: if id == "best" { "best" } else { "step" },
            // The step from the manifest when there is one, and from the name
            // otherwise: an incomplete checkpoint still has a name.
            global_step: manifest
                .as_ref()
                .map(|manifest| manifest.global_step)
                .or_else(|| step_from_id(&id))
                .unwrap_or(0),
            bytes: directory_bytes(&path).unwrap_or(0),
            written_at: modified_seconds(&path.join(checkpoint::MANIFEST_FILE)),
            adapter: adapter.is_file().then(|| adapter.display().to_string()),
            complete: manifest.is_some(),
            path: path.display().to_string(),
            id,
        });
    }
    // Newest first: by step, then by id so `best` and a step of the same number
    // do not swap places between two calls.
    found.sort_by(|left, right| {
        right
            .global_step
            .cmp(&left.global_step)
            .then(left.id.cmp(&right.id))
    });
    found
}

/// The checkpoint id a `*.state` directory carries, or `None` for anything else
/// in the directory - including the `.tmp-` and `.backup` directories a write in
/// flight leaves behind, which are not checkpoints a client may resume from.
fn checkpoint_id(path: &FsPath) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    if name.starts_with('.') {
        return None;
    }
    Some(name.strip_suffix(".state")?.to_string())
}

fn step_from_id(id: &str) -> Option<u64> {
    id.strip_prefix("step-")?.parse().ok()
}

fn modified_seconds(path: &FsPath) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|elapsed| elapsed.as_secs())
}

/// Keeps a filename usable inside a quoted `Content-Disposition`. The names come
/// from paths a client configured, so a quote or a newline in one would otherwise
/// let it write its own header.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|character| match character {
            '"' | '\\' | '\r' | '\n' => '_',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_state_directories_are_checkpoints() {
        assert_eq!(
            checkpoint_id(FsPath::new("/c/step-000000000010.state")).as_deref(),
            Some("step-000000000010")
        );
        assert_eq!(
            checkpoint_id(FsPath::new("/c/best.state")).as_deref(),
            Some("best")
        );
        assert!(checkpoint_id(FsPath::new("/c/step-10.gguf")).is_none());
        // A write in flight, and the backup a replacement leaves: neither is a
        // checkpoint a client may name.
        assert!(checkpoint_id(FsPath::new("/c/.best.state.tmp-1")).is_none());
        assert!(checkpoint_id(FsPath::new("/c/.step-1.state.backup")).is_none());
        assert_eq!(step_from_id("step-000000000042"), Some(42));
        assert_eq!(step_from_id("best"), None);
    }

    #[test]
    fn an_observed_run_lists_its_log_and_its_directory_in_order() {
        let artifacts = RunArtifacts {
            adapter: Some("/r/adapter.gguf".into()),
            checkpoint_directory: Some("/r/ckpt".into()),
            observe: Some("/r/observe".into()),
            ..Default::default()
        };
        let items = inventory(&artifacts, FsPath::new("/runs/1"));
        let names = items.iter().map(|item| item.name).collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "adapter",
                "run",
                "events",
                "observe_log",
                "observe",
                "checkpoints"
            ]
        );
        let log = &items[3];
        assert_eq!(log.path, FsPath::new("/r/observe/observe.jsonl"));
        assert!(!log.directory);
        assert!(items[4].directory);
    }

    #[test]
    fn a_filename_cannot_write_its_own_header() {
        assert_eq!(sanitize("adapter.gguf"), "adapter.gguf");
        assert_eq!(sanitize("a\"b\r\nc"), "a_b__c");
    }
}
