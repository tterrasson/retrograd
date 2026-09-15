//! `--state-dir` on disk: one directory per run, `run.json` for what the
//! run *is* and `events.jsonl` for what happened to it.
//!
//! Two different write patterns on purpose. `run.json` is rewritten whole at
//! every state change - it is small, rare, and a client reading a torn one would
//! be worse than a rewrite. `events.jsonl` is append-only and never rewritten:
//! an append is what survives a kill -9 with the prefix intact, which is exactly
//! what a replay needs.
//!
//! Nothing here is allowed to fail a run. A state directory that cannot be
//! written is a degraded server, not a broken training job, so every I/O error
//! is logged and swallowed. The one place that does propagate is creation, where
//! the caller can still answer the request with an error.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use retrograd_core::{Error, Result as CoreResult};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::dto;

/// `run.json`: everything needed to answer `GET /v1/runs/{id}` without the run.
///
/// The three heavy fields are `RawValue`, so what the file holds is the bytes
/// the API already answered - no re-serialization, and no float re-rendered
/// differently by a round trip through a typed value.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredRun {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub status: dto::RunStatus,
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<u64>,
    #[serde(default)]
    pub progress: dto::RunProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Every state this run went through, in order: a run that failed at 03:12
    /// after four hours of `running` reads that way, instead of just `failed`.
    #[serde(default)]
    pub transitions: Vec<Transition>,
    pub effective_config: Box<RawValue>,
    pub provenance: Box<RawValue>,
    pub plan: Box<RawValue>,
    /// Where this run's outputs are. Typed rather than re-derived from
    /// `effective_config`, so `GET /v1/runs/{id}/artifacts` answers for a run this
    /// process only read back from disk - which is most of them.
    #[serde(default)]
    pub artifacts: super::registry::RunArtifacts,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Transition {
    pub status: dto::RunStatus,
    pub at: u64,
}

/// The on-disk home of one run.
pub struct Journal {
    dir: PathBuf,
    /// Kept open for the life of the run: an append per event, not an open per
    /// event. A run reports several events a second at a small batch size.
    events: Mutex<File>,
}

impl Journal {
    /// Creates `<state_dir>/<id>/` and opens the event log.
    pub fn create(state_dir: &Path, id: &str) -> CoreResult<Self> {
        let dir = state_dir.join(id);
        std::fs::create_dir_all(&dir).map_err(|error| {
            Error::runtime(format!(
                "could not create the run directory {}: {error}",
                dir.display()
            ))
        })?;
        let events = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("events.jsonl"))
            .map_err(|error| {
                Error::runtime(format!("could not open the run's event log: {error}"))
            })?;
        Ok(Self {
            dir,
            events: Mutex::new(events),
        })
    }

    pub fn directory(&self) -> &Path {
        &self.dir
    }

    /// Rewrites `run.json`.
    pub fn write_document(&self, document: &StoredRun) {
        let path = self.dir.join("run.json");
        match serde_json::to_vec_pretty(document) {
            Ok(bytes) => {
                if let Err(error) = std::fs::write(&path, bytes) {
                    tracing::warn!(path = %path.display(), %error, "could not write run.json");
                }
            }
            Err(error) => tracing::warn!(%error, "could not render run.json"),
        }
    }

    /// Appends one event. One line, flushed: a reader tailing the file must see
    /// whole lines, and a crash must not lose the last minute of a run.
    pub fn append(&self, event: &dto::RunEvent) {
        let Ok(mut line) = serde_json::to_vec(event) else {
            return;
        };
        line.push(b'\n');
        let Ok(mut file) = self.events.lock() else {
            return;
        };
        if let Err(error) = file.write_all(&line).and_then(|()| file.flush()) {
            tracing::warn!(%error, "could not append to the run's event log");
        }
    }
}

/// Reads one run's events back, oldest first, keeping only those after `since`.
///
/// The append-only log is what makes this honest: a torn last line is a line that
/// was being written when the process died, and skipping it loses nothing a
/// reader could have seen anyway. Everything before it is intact by construction,
/// which is exactly the guarantee a replay needs.
pub fn read_events(dir: &Path, since: u64) -> Vec<dto::RunEvent> {
    let Ok(text) = std::fs::read_to_string(dir.join("events.jsonl")) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<dto::RunEvent>(line).ok())
        .filter(|event| event.seq > since)
        .collect()
}

/// Reads back every run in the state directory, newest first.
///
/// A run found in a live state was interrupted by whatever stopped the previous
/// process. It is marked `interrupted` **on disk**, so the history says
/// so even if this process dies too: the V1 does not re-attach, and a run that
/// still claimed to be `running` would be a permanent lie.
pub fn scan(state_dir: &Path) -> Vec<StoredRun> {
    let Ok(entries) = std::fs::read_dir(state_dir) else {
        return Vec::new();
    };
    let mut runs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path().join("run.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut run: StoredRun = match serde_json::from_str(&text) {
            Ok(run) => run,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "ignoring an unreadable run.json");
                continue;
            }
        };
        if !run.status.is_terminal() {
            run.status = dto::RunStatus::Interrupted;
            run.finished_at = Some(run.finished_at.unwrap_or_else(super::unix_seconds));
            run.transitions.push(Transition {
                status: dto::RunStatus::Interrupted,
                at: super::unix_seconds(),
            });
            if let Ok(bytes) = serde_json::to_vec_pretty(&run) {
                let _ = std::fs::write(&path, bytes);
            }
        }
        runs.push(run);
    }
    // Newest first, ties broken by id so two runs created in the same second
    // list in a fixed order.
    runs.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then(left.id.cmp(&right.id))
    });
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(text: &str) -> Box<RawValue> {
        RawValue::from_string(text.to_string()).expect("valid JSON")
    }

    fn stored(id: &str, status: dto::RunStatus, created_at: u64) -> StoredRun {
        StoredRun {
            id: id.to_string(),
            name: None,
            status,
            created_at,
            started_at: None,
            finished_at: None,
            progress: dto::RunProgress::default(),
            error: None,
            transitions: Vec::new(),
            effective_config: raw(r#"{"training":{"lr":0.0001}}"#),
            provenance: raw("{}"),
            plan: raw("{}"),
            artifacts: Default::default(),
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("retrograd-journal-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create");
        dir
    }

    #[test]
    fn a_run_survives_the_process_that_wrote_it() {
        let dir = temp_dir("roundtrip");
        let journal = Journal::create(&dir, "abc").expect("create the journal");
        journal.write_document(&stored("abc", dto::RunStatus::Completed, 10));
        journal.append(&dto::RunEvent {
            seq: 1,
            at: 5,
            payload: dto::RunEventPayload::Log {
                message: "hello".into(),
            },
        });
        journal.append(&dto::RunEvent {
            seq: 2,
            at: 6,
            payload: dto::RunEventPayload::Terminal {
                status: dto::RunStatus::Completed,
                error: None,
            },
        });

        let events = std::fs::read_to_string(dir.join("abc/events.jsonl")).expect("read");
        assert_eq!(events.lines().count(), 2, "one line per event: {events}");
        assert!(events.contains(r#""type":"log""#), "{events}");

        let restored = scan(&dir);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].status, dto::RunStatus::Completed);
        // The float came back as it was written, not widened by a round trip.
        assert!(
            restored[0].effective_config.get().contains("0.0001"),
            "{}",
            restored[0].effective_config.get()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_still_marked_running_is_interrupted_on_disk() {
        let dir = temp_dir("interrupted");
        let journal = Journal::create(&dir, "live").expect("create the journal");
        journal.write_document(&stored("live", dto::RunStatus::Running, 10));

        let restored = scan(&dir);
        assert_eq!(restored[0].status, dto::RunStatus::Interrupted);
        assert!(restored[0].finished_at.is_some());
        // And on disk, so a second restart does not have to rediscover it.
        let again = scan(&dir);
        assert_eq!(again[0].status, dto::RunStatus::Interrupted);
        assert_eq!(
            again[0].transitions.len(),
            1,
            "a terminal run is not re-marked on every scan"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_history_is_newest_first_and_skips_what_it_cannot_read() {
        let dir = temp_dir("ordering");
        for (id, created) in [("old", 10_u64), ("new", 30), ("mid", 20)] {
            let journal = Journal::create(&dir, id).expect("create");
            journal.write_document(&stored(id, dto::RunStatus::Completed, created));
        }
        std::fs::create_dir_all(dir.join("broken")).unwrap();
        std::fs::write(dir.join("broken/run.json"), "{not json").unwrap();

        let ids: Vec<String> = scan(&dir).into_iter().map(|run| run.id).collect();
        assert_eq!(ids, ["new", "mid", "old"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
