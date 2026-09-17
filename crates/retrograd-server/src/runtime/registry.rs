//! The registry of runs and the state machine each one obeys.
//!
//! One handle per run, live or historical. A handle is the only thing that
//! writes a run's state, which is what makes the transition table below the
//! single source of truth about what a run may do next - a handler asking for an
//! impossible transition gets a 409 rather than a registry that quietly agrees.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use retrograd_core::Result as CoreResult;
use serde_json::value::RawValue;
use tokio::sync::broadcast;
use uuid::Uuid;

use super::control::ControlSender;
use super::journal::{self, Journal, StoredRun, Transition};
use super::{unix_millis, unix_seconds};
use crate::dto;
use crate::lock::Recover as _;

/// How many recent events a run keeps in memory for replay.
///
/// The ring is an optimization, not the record: anything older is read back from
/// `events.jsonl`, so a client reconnecting after a long gap is served correctly
/// and only more slowly. That is why the number can be modest.
const EVENT_RING: usize = 512;

/// How many events a live subscriber may fall behind before it is told it
/// lagged. A slow client must never slow the training loop, so the
/// broadcast drops rather than blocks and the client catches up with `?since=`.
const BROADCAST_CAPACITY: usize = 1024;

/// One event, tagged with the run it belongs to, for the aggregate stream.
#[derive(Clone, Debug)]
pub struct TaggedEvent {
    pub run: String,
    pub event: Arc<dto::RunEvent>,
}

/// The immutable half of a run: what was resolved, rendered once.
#[derive(Clone, Debug)]
pub struct RunRecord {
    pub effective_config: Box<RawValue>,
    pub provenance: Box<RawValue>,
    pub plan: Box<RawValue>,
    /// Epochs or updates the plan expects, so progress is a fraction and not a
    /// bare counter.
    pub iterations: u64,
    /// What the resolved configuration makes adjustable at all.
    pub controls: RunControls,
    /// The files this run writes, so the inventory is a directory read rather
    /// than a re-parse of the rendered configuration.
    pub artifacts: RunArtifacts,
}

/// Where a run's outputs land, read off the resolved configuration at creation.
///
/// Kept in `run.json` (and therefore restored at startup) for one reason: the run
/// whose artefacts a client most wants to list is a *finished* one, possibly from
/// a previous process. Deriving these paths from the rendered configuration would
/// work only while the run was alive, and re-parsing a document to answer "where
/// is the adapter" is how the two copies drift apart.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct RunArtifacts {
    /// The adapter the run saves when it finishes.
    pub adapter: Option<PathBuf>,
    pub checkpoint_directory: Option<PathBuf>,
    pub tensorboard_directory: Option<PathBuf>,
    pub wandb_export_directory: Option<PathBuf>,
    /// The `[observe]` directory: the viewer, its feed and `observe.jsonl`.
    pub observe: Option<PathBuf>,
}

/// Which halves of the `PATCH` whitelist this run actually has.
///
/// Captured from the resolved configuration at creation, so a handler can refuse
/// "evaluate more often" on a run with no evaluation dataset *before* queueing a
/// command nobody would ever apply. Cheap, typed, and not read back from the
/// rendered JSON - parsing a document to answer a yes/no question is how the two
/// drift apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunControls {
    pub has_evaluation: bool,
    pub has_checkpoint: bool,
}

/// The mutable half: where the run is now.
#[derive(Clone, Debug)]
pub struct RunState {
    pub status: dto::RunStatus,
    pub progress: dto::RunProgress,
    pub error: Option<String>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub transitions: Vec<Transition>,
}

/// Whether `to` may follow `from`.
///
/// The base runtime drives `queued → starting → running → completed | failed`;
/// `pause`, `resume` and `cancel` add `pausing`, `paused` and `cancelling` on
/// top, reachable from `starting` onward so a command sent while the model is
/// still loading is honoured at the run's first callback rather than ignored.
pub fn transition_allowed(from: dto::RunStatus, to: dto::RunStatus) -> bool {
    use dto::RunStatus::*;
    if from == to {
        // Re-asserting the current state is a no-op, not a conflict: a command
        // repeated after a timeout must be idempotent.
        return true;
    }
    match from {
        Queued => matches!(to, Resolving | Starting | Cancelled | Failed),
        Resolving => matches!(to, Starting | Failed | Cancelled),
        // `Pausing` is reachable from `Starting`: a client may pause a run while
        // its model is still loading, and the loop honours it at its first
        // callback rather than ignoring a command that was legal when sent.
        Starting => matches!(to, Running | Pausing | Failed | Cancelled | Cancelling),
        Running => matches!(to, Pausing | Cancelling | Completed | Failed | Cancelled),
        Pausing => matches!(to, Paused | Cancelling | Completed | Failed | Cancelled),
        Paused => matches!(to, Running | Cancelling | Cancelled | Failed),
        Cancelling => matches!(to, Cancelled | Completed | Failed),
        // A terminal state is final. A run that finished does not start again;
        // that is what `fork_from` is for.
        Completed | Failed | Cancelled | Interrupted => false,
    }
}

/// One run: its identity, what was resolved for it, where it is, and where it is
/// written down.
pub struct RunHandle {
    pub id: Uuid,
    pub name: Option<String>,
    pub created_at: u64,
    /// Creation order in this process. Restored runs get zero and are ordered by
    /// `created_at` alone, which is all the disk knows.
    pub sequence: u64,
    pub record: RunRecord,
    state: RwLock<RunState>,
    next_seq: AtomicU64,
    /// Absent for a run restored from disk: its journal is complete and must not
    /// be appended to by a process that is not running it.
    journal: Option<Journal>,
    /// Where this run's `events.jsonl` is, whether or not this process wrote it.
    /// A restored run has no journal but still has a directory, and that is what
    /// lets its history be replayed.
    dir: PathBuf,
    /// Absent for a restored run: there is no thread to command.
    control: Option<ControlSender>,
    /// The last [`EVENT_RING`] events, newest last, for replay without disk.
    ring: RwLock<VecDeque<Arc<dto::RunEvent>>>,
    live: broadcast::Sender<Arc<dto::RunEvent>>,
    all: broadcast::Sender<TaggedEvent>,
}

impl RunHandle {
    pub fn state(&self) -> RunState {
        self.state.read().recover().clone()
    }

    pub fn status(&self) -> dto::RunStatus {
        self.state.read().recover().status
    }

    /// Moves the run to `status`, emits the matching event and rewrites
    /// `run.json`. Returns false - and changes nothing - when the transition is
    /// not allowed.
    pub fn transition(&self, status: dto::RunStatus) -> bool {
        let previous;
        {
            let mut state = self.state.write().recover();
            if !transition_allowed(state.status, status) {
                return false;
            }
            if state.status == status {
                return true;
            }
            previous = state.status;
            let now = unix_seconds();
            state.status = status;
            state.transitions.push(Transition { status, at: now });
            if status == dto::RunStatus::Running && state.started_at.is_none() {
                state.started_at = Some(now);
            }
            if status.is_terminal() {
                state.finished_at = Some(now);
            }
        }
        tracing::info!(
            target: "retrograd::run::state",
            run_id = %self.id,
            from = ?previous,
            to = ?status,
            "run state changed"
        );
        self.emit(dto::RunEventPayload::Status { status });
        self.persist();
        true
    }

    /// Records the terminal outcome: the message goes into the state *before*
    /// the transition, so a client that sees `failed` always sees why.
    pub fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        {
            let mut state = self.state.write().recover();
            state.error = Some(message.clone());
        }
        let status = if self.transition(dto::RunStatus::Failed) {
            dto::RunStatus::Failed
        } else {
            self.status()
        };
        self.emit(dto::RunEventPayload::Terminal {
            status,
            error: Some(message),
        });
        self.persist();
    }

    pub fn complete(&self) {
        self.transition(dto::RunStatus::Completed);
        self.emit(dto::RunEventPayload::Terminal {
            status: dto::RunStatus::Completed,
            error: None,
        });
    }

    /// The run stopped because it was asked to. Distinct from [`Self::complete`]
    /// even though both leave a clean adapter on disk: one reached its last
    /// iteration and the other did not.
    pub fn cancelled(&self) {
        self.transition(dto::RunStatus::Cancelled);
        self.emit(dto::RunEventPayload::Terminal {
            status: dto::RunStatus::Cancelled,
            error: None,
        });
    }

    /// Updates the progress snapshot and emits it.
    pub fn progress(&self, update: impl FnOnce(&mut dto::RunProgress)) {
        let snapshot = {
            let mut state = self.state.write().recover();
            update(&mut state.progress);
            state.progress.iterations = state.progress.iterations.max(self.record.iterations);
            state.progress
        };
        self.emit(dto::RunEventPayload::Progress(snapshot));
    }

    /// Appends one event to the journal, numbering it.
    ///
    /// The sequence is allocated here and nowhere else, so it is monotonic per
    /// run whatever order the observer's callbacks arrive in - which is what a
    /// replaying client relies on.
    pub fn emit(&self, payload: dto::RunEventPayload) {
        let event = Arc::new(dto::RunEvent {
            seq: self.next_seq.fetch_add(1, Ordering::Relaxed),
            at: unix_millis(),
            payload,
        });
        // Disk first. The journal is the record a replay falls back on and the
        // only copy that survives this process; the ring and the broadcast are
        // both caches of it.
        if let Some(journal) = &self.journal {
            journal.append(&event);
        }
        {
            let mut ring = self.ring.write().recover();
            if ring.len() == EVENT_RING {
                ring.pop_front();
            }
            ring.push_back(event.clone());
        }
        // Both sends fail only when nobody is subscribed, which is the normal
        // case for a run nobody is watching.
        let _ = self.live.send(event.clone());
        let _ = self.all.send(TaggedEvent {
            run: self.id.to_string(),
            event,
        });
    }

    /// A live subscription. Taken *before* replaying, so nothing emitted between
    /// the two is lost - the duplicates that causes are removed by `seq`.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<dto::RunEvent>> {
        self.live.subscribe()
    }

    /// Every event after `since`, oldest first.
    ///
    /// Served from the ring when it reaches back far enough and from
    /// `events.jsonl` otherwise, so `?since=0` replays a whole run and a client
    /// that was away longer than the ring is not silently given a hole. A run
    /// restored from disk has only the second path, which is the whole reason
    /// the directory is kept beside the journal.
    pub fn replay(&self, since: u64) -> Vec<Arc<dto::RunEvent>> {
        {
            let ring = self.ring.read().recover();
            let covers = ring
                .front()
                .is_some_and(|oldest| oldest.seq <= since.saturating_add(1));
            if covers {
                return ring
                    .iter()
                    .filter(|event| event.seq > since)
                    .cloned()
                    .collect();
            }
        }
        journal::read_events(&self.dir, since)
            .into_iter()
            .map(Arc::new)
            .collect()
    }

    /// The seq the next event will carry minus one: what a client that has seen
    /// everything would pass as `?since=`.
    pub fn last_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed).saturating_sub(1)
    }

    /// The command channel of a live run, or `None` when there is no thread to
    /// command - a run restored from disk, or one already over.
    pub fn control(&self) -> Option<&ControlSender> {
        self.control.as_ref()
    }

    /// Whether this run is holding the device: weights, KV caches and optimizer
    /// state are allocated.
    pub fn holds_device(&self) -> bool {
        matches!(
            self.status(),
            dto::RunStatus::Starting
                | dto::RunStatus::Running
                | dto::RunStatus::Pausing
                | dto::RunStatus::Paused
                | dto::RunStatus::Cancelling
        )
    }

    pub fn summary(&self) -> dto::RunSummary {
        let state = self.state();
        dto::RunSummary {
            id: self.id.to_string(),
            name: self.name.clone(),
            status: state.status,
            created_at: self.created_at,
            started_at: state.started_at,
            finished_at: state.finished_at,
            progress: state.progress,
            error: state.error,
        }
    }

    pub fn view(&self) -> dto::RunView {
        let state = self.state();
        dto::RunView {
            id: self.id.to_string(),
            name: self.name.clone(),
            status: state.status,
            created_at: self.created_at,
            started_at: state.started_at,
            finished_at: state.finished_at,
            progress: state.progress,
            error: state.error,
            holds_device: self.holds_device(),
            effective_config: self.record.effective_config.clone(),
            provenance: self.record.provenance.clone(),
            plan: self.record.plan.clone(),
        }
    }

    /// Where this run's artefacts live, when it has a directory.
    pub fn directory(&self) -> Option<&Path> {
        self.journal.as_ref().map(Journal::directory)
    }

    /// This run's directory under the state dir, journal or no journal. A
    /// restored run has no journal to append to but still has a `run.json` and an
    /// `events.jsonl` a client may fetch.
    pub fn state_directory(&self) -> &Path {
        &self.dir
    }

    fn persist(&self) {
        let Some(journal) = &self.journal else {
            return;
        };
        journal.write_document(&self.document());
    }

    fn document(&self) -> StoredRun {
        let state = self.state();
        StoredRun {
            id: self.id.to_string(),
            name: self.name.clone(),
            status: state.status,
            created_at: self.created_at,
            started_at: state.started_at,
            finished_at: state.finished_at,
            progress: state.progress,
            error: state.error,
            transitions: state.transitions,
            effective_config: self.record.effective_config.clone(),
            provenance: self.record.provenance.clone(),
            plan: self.record.plan.clone(),
            artifacts: self.record.artifacts.clone(),
        }
    }
}

/// How a listing is narrowed.
#[derive(Clone, Debug)]
pub struct RunFilter {
    pub status: Option<dto::RunStatus>,
    /// Exact match, not a substring: a name is a label a client chose, and a
    /// substring filter would make two runs called `sft` and `sft-v2`
    /// indistinguishable to a paging client.
    pub name: Option<String>,
    pub limit: usize,
    /// The `id` of the last run of the previous page.
    pub cursor: Option<String>,
}

impl Default for RunFilter {
    /// A page, not an empty one. `limit: 0` would clamp to a single run, which
    /// is the kind of default that turns a count into an off-by-everything.
    fn default() -> Self {
        Self {
            status: None,
            name: None,
            limit: 50,
            cursor: None,
        }
    }
}

pub struct Page {
    pub runs: Vec<dto::RunSummary>,
    pub next_cursor: Option<String>,
}

/// Every run this process knows about, live or read back from `--state-dir`.
pub struct RunRegistry {
    state_dir: PathBuf,
    inner: RwLock<Inner>,
    /// Every run's events, merged: what `GET /v1/events` streams.
    all: broadcast::Sender<TaggedEvent>,
}

#[derive(Default)]
struct Inner {
    runs: BTreeMap<Uuid, Arc<RunHandle>>,
    /// Creation order, newest last. Kept beside the map because a `Uuid` v4 has
    /// no time component to sort on.
    order: Vec<Uuid>,
    /// `Idempotency-Key` → the run it created.
    keys: HashMap<String, Uuid>,
    next_sequence: u64,
}

impl RunRegistry {
    /// Builds the registry and reads the state directory back.
    ///
    /// Restoring at startup is what makes `GET /v1/runs` show the history rather
    /// than only what this process happened to launch.
    pub fn open(state_dir: impl Into<PathBuf>) -> Self {
        let state_dir = state_dir.into();
        let (all, _) = broadcast::channel(BROADCAST_CAPACITY);
        let mut inner = Inner::default();
        // Oldest first, so the creation order the listing renders matches the
        // order of the timestamps.
        for stored in journal::scan(&state_dir).into_iter().rev() {
            let Ok(id) = stored.id.parse::<Uuid>() else {
                continue;
            };
            let (live, _) = broadcast::channel(BROADCAST_CAPACITY);
            let handle = Arc::new(RunHandle {
                id,
                name: stored.name.clone(),
                created_at: stored.created_at,
                sequence: 0,
                record: RunRecord {
                    effective_config: stored.effective_config.clone(),
                    provenance: stored.provenance.clone(),
                    plan: stored.plan.clone(),
                    iterations: stored.progress.iterations,
                    // A restored run is terminal, so nothing about it is
                    // adjustable; its artefacts, on the other hand, are exactly
                    // what a client comes back for.
                    controls: RunControls::default(),
                    artifacts: stored.artifacts.clone(),
                },
                state: RwLock::new(RunState {
                    status: stored.status,
                    progress: stored.progress,
                    error: stored.error,
                    started_at: stored.started_at,
                    finished_at: stored.finished_at,
                    transitions: stored.transitions,
                }),
                next_seq: AtomicU64::new(1),
                journal: None,
                dir: state_dir.join(&stored.id),
                control: None,
                ring: RwLock::new(VecDeque::new()),
                live,
                all: all.clone(),
            });
            inner.order.push(id);
            inner.runs.insert(id, handle);
        }
        Self {
            state_dir,
            inner: RwLock::new(inner),
            all,
        }
    }

    /// A subscription to every run's events.
    pub fn subscribe_all(&self) -> broadcast::Receiver<TaggedEvent> {
        self.all.subscribe()
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Registers a new run in `queued`, with its journal already on disk.
    ///
    /// The journal is written before the worker starts on purpose: a run that
    /// exists to the API but not to the filesystem would vanish on a restart
    /// while its worker was still holding the device.
    pub fn create(
        &self,
        name: Option<String>,
        record: RunRecord,
        idempotency_key: Option<String>,
        control: ControlSender,
    ) -> CoreResult<Arc<RunHandle>> {
        self.create_with_id(Uuid::new_v4(), name, record, idempotency_key, control)
    }

    /// Registers a run whose id was allocated by the caller. Run creation uses
    /// this to derive managed artifact paths before the effective configuration
    /// and journal are rendered.
    pub fn create_with_id(
        &self,
        id: Uuid,
        name: Option<String>,
        record: RunRecord,
        idempotency_key: Option<String>,
        control: ControlSender,
    ) -> CoreResult<Arc<RunHandle>> {
        let created_at = unix_seconds();
        let sequence = {
            let mut inner = self.inner.write().recover();
            inner.next_sequence += 1;
            inner.next_sequence
        };
        let journal = Journal::create(&self.state_dir, &id.to_string())?;
        let (live, _) = broadcast::channel(BROADCAST_CAPACITY);
        let handle = Arc::new(RunHandle {
            id,
            name,
            created_at,
            sequence,
            record,
            state: RwLock::new(RunState {
                status: dto::RunStatus::Queued,
                progress: dto::RunProgress::default(),
                error: None,
                started_at: None,
                finished_at: None,
                transitions: vec![Transition {
                    status: dto::RunStatus::Queued,
                    at: created_at,
                }],
            }),
            next_seq: AtomicU64::new(1),
            journal: Some(journal),
            dir: self.state_dir.join(id.to_string()),
            control: Some(control),
            ring: RwLock::new(VecDeque::new()),
            live,
            all: self.all.clone(),
        });
        handle.persist();

        let mut inner = self.inner.write().recover();
        if let Some(key) = idempotency_key {
            inner.keys.insert(key, id);
        }
        inner.order.push(id);
        inner.runs.insert(id, handle.clone());
        Ok(handle)
    }

    /// Forgets a run and deletes its state directory (`DELETE /v1/runs/{id}`).
    ///
    /// Only the *run's own* directory goes: `run.json`, `events.jsonl` and
    /// whatever else the server wrote under it. Checkpoints, adapters and
    /// TensorBoard logs live wherever the configuration put them - paths the
    /// client chose, often shared between runs - and deleting those would make a
    /// purge of a run's bookkeeping a destructive operation on a client's
    /// filesystem.
    ///
    /// The idempotency key is dropped with the run, so retrying the creation that
    /// made it starts a new run rather than resurrecting a deleted one.
    pub fn purge(&self, id: &Uuid) -> Option<Arc<RunHandle>> {
        let handle = {
            let mut inner = self.inner.write().recover();
            let handle = inner.runs.remove(id)?;
            inner.order.retain(|known| known != id);
            inner.keys.retain(|_, known| known != id);
            handle
        };
        let dir = handle.state_directory().to_path_buf();
        // Guard against ever removing the state directory itself, which a run
        // built with an empty id would point at.
        if dir.starts_with(&self.state_dir)
            && dir != self.state_dir
            && let Err(error) = std::fs::remove_dir_all(&dir)
        {
            tracing::warn!(path = %dir.display(), %error, "could not remove the run directory");
        }
        Some(handle)
    }

    pub fn get(&self, id: &Uuid) -> Option<Arc<RunHandle>> {
        self.inner.read().recover().runs.get(id).cloned()
    }

    /// The run a previous request with this `Idempotency-Key` created, if any.
    pub fn by_idempotency_key(&self, key: &str) -> Option<Arc<RunHandle>> {
        let inner = self.inner.read().recover();
        inner
            .keys
            .get(key)
            .and_then(|id| inner.runs.get(id))
            .cloned()
    }

    /// Every run this registry knows about, live and restored.
    pub fn len(&self) -> usize {
        self.inner.read().recover().runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn count_with(&self, status: dto::RunStatus) -> usize {
        self.inner
            .read()
            .recover()
            .runs
            .values()
            .filter(|handle| handle.status() == status)
            .count()
    }

    /// How many runs are holding, or about to hold, the device. What a caller
    /// needs to know before queueing another one behind them.
    pub fn active(&self) -> usize {
        self.inner
            .read()
            .recover()
            .runs
            .values()
            .filter(|handle| !handle.status().is_terminal())
            .count()
    }

    /// Every run that has not finished, handle and all. What a graceful shutdown
    /// needs: the summaries a listing returns cannot be commanded.
    pub fn live_handles(&self) -> Vec<Arc<RunHandle>> {
        self.inner
            .read()
            .recover()
            .runs
            .values()
            .filter(|handle| !handle.status().is_terminal())
            .cloned()
            .collect()
    }

    /// Newest first, filtered, one page at a time.
    pub fn list(&self, filter: &RunFilter) -> Page {
        let inner = self.inner.read().recover();
        let mut handles: Vec<&Arc<RunHandle>> = inner
            .order
            .iter()
            .filter_map(|id| inner.runs.get(id))
            .collect();
        // Newest first: creation order within this process, then the recorded
        // timestamp for runs restored from disk.
        handles.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then(right.sequence.cmp(&left.sequence))
                .then(left.id.cmp(&right.id))
        });

        let mut summaries: Vec<dto::RunSummary> = handles
            .into_iter()
            .map(|handle| handle.summary())
            .filter(|summary| filter.status.is_none_or(|status| summary.status == status))
            .filter(|summary| {
                filter
                    .name
                    .as_ref()
                    .is_none_or(|name| summary.name.as_deref() == Some(name.as_str()))
            })
            .collect();

        if let Some(cursor) = &filter.cursor {
            // Everything strictly after the cursor. An unknown cursor yields an
            // empty page rather than the first one: silently restarting a
            // pagination would make a client loop forever.
            match summaries.iter().position(|run| &run.id == cursor) {
                Some(index) => summaries.drain(..=index),
                None => summaries.drain(..),
            };
        }
        let limit = filter.limit.clamp(1, 500);
        let next_cursor = (summaries.len() > limit)
            .then(|| summaries.get(limit - 1).map(|run| run.id.clone()))
            .flatten();
        summaries.truncate(limit);
        Page {
            runs: summaries,
            next_cursor,
        }
    }
}

/// Handles built without a registry, for the unit tests of the control channel.
#[cfg(test)]
pub mod testing {
    use super::*;

    /// A handle attached to nothing: no journal, no directory, no registry.
    ///
    /// The control state machine needs *a* handle to report its transitions to,
    /// and giving it a real one would drag a temporary directory into every test
    /// of a state machine that never touches disk.
    pub fn detached_handle() -> Arc<RunHandle> {
        let (live, _) = broadcast::channel(16);
        let (all, _) = broadcast::channel(16);
        Arc::new(RunHandle {
            id: Uuid::new_v4(),
            name: None,
            created_at: 0,
            sequence: 0,
            record: RunRecord {
                effective_config: RawValue::from_string("{}".into()).expect("valid JSON"),
                provenance: RawValue::from_string("{}".into()).expect("valid JSON"),
                plan: RawValue::from_string("{}".into()).expect("valid JSON"),
                iterations: 0,
                controls: RunControls::default(),
                artifacts: RunArtifacts::default(),
            },
            state: RwLock::new(RunState {
                status: dto::RunStatus::Running,
                progress: dto::RunProgress::default(),
                error: None,
                started_at: None,
                finished_at: None,
                transitions: Vec::new(),
            }),
            next_seq: AtomicU64::new(1),
            journal: None,
            dir: PathBuf::new(),
            control: None,
            ring: RwLock::new(VecDeque::new()),
            live,
            all,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw() -> Box<RawValue> {
        RawValue::from_string("{}".to_string()).expect("valid JSON")
    }

    fn record() -> RunRecord {
        RunRecord {
            effective_config: raw(),
            provenance: raw(),
            plan: raw(),
            iterations: 3,
            controls: RunControls::default(),
            artifacts: RunArtifacts::default(),
        }
    }

    /// A control sender whose receiver is dropped straight away: these tests are
    /// about the registry, and no worker is listening.
    fn control() -> ControlSender {
        super::super::control::channel().0
    }

    fn registry(label: &str) -> RunRegistry {
        let dir =
            std::env::temp_dir().join(format!("retrograd-registry-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        RunRegistry::open(dir)
    }

    #[test]
    fn the_state_machine_refuses_to_restart_a_finished_run() {
        use dto::RunStatus::*;
        assert!(transition_allowed(Queued, Starting));
        assert!(transition_allowed(Starting, Running));
        assert!(transition_allowed(Running, Completed));
        assert!(transition_allowed(Running, Failed));
        // Idempotent: the same state twice is not a conflict.
        assert!(transition_allowed(Running, Running));
        // And the ones that must never happen.
        assert!(!transition_allowed(Completed, Running));
        assert!(!transition_allowed(Failed, Queued));
        assert!(!transition_allowed(Interrupted, Running));
        assert!(!transition_allowed(Queued, Completed));
    }

    #[test]
    fn a_handle_records_its_progress_and_its_terminal_reason() {
        let registry = registry("progress");
        let handle = registry
            .create(Some("a".into()), record(), None, control())
            .unwrap();
        assert_eq!(handle.status(), dto::RunStatus::Queued);
        assert!(!handle.holds_device());

        assert!(handle.transition(dto::RunStatus::Starting));
        assert!(handle.holds_device());
        assert!(handle.transition(dto::RunStatus::Running));
        handle.progress(|progress| {
            progress.iteration = 1;
            progress.global_step = 12;
            progress.train_loss = Some(0.5);
        });
        let state = handle.state();
        assert_eq!(state.progress.iteration, 1);
        assert_eq!(
            state.progress.iterations, 3,
            "the plan's iteration count fills in, so progress is a fraction"
        );
        assert!(state.started_at.is_some());

        handle.fail("model could not be loaded");
        let state = handle.state();
        assert_eq!(state.status, dto::RunStatus::Failed);
        assert_eq!(state.error.as_deref(), Some("model could not be loaded"));
        assert!(state.finished_at.is_some());
        // Terminal is terminal.
        assert!(!handle.transition(dto::RunStatus::Running));
        let _ = std::fs::remove_dir_all(registry.state_dir());
    }

    #[test]
    fn a_listing_is_newest_first_filtered_and_paged() {
        let registry = registry("listing");
        let first = registry
            .create(Some("a".into()), record(), None, control())
            .unwrap();
        let second = registry
            .create(Some("b".into()), record(), None, control())
            .unwrap();
        let third = registry
            .create(Some("a".into()), record(), None, control())
            .unwrap();
        second.transition(dto::RunStatus::Starting);
        second.transition(dto::RunStatus::Running);
        second.complete();

        let page = registry.list(&RunFilter {
            limit: 10,
            ..Default::default()
        });
        let ids: Vec<&str> = page.runs.iter().map(|run| run.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                third.id.to_string(),
                second.id.to_string(),
                first.id.to_string()
            ]
        );
        assert!(page.next_cursor.is_none());

        let completed = registry.list(&RunFilter {
            status: Some(dto::RunStatus::Completed),
            limit: 10,
            ..Default::default()
        });
        assert_eq!(completed.runs.len(), 1);
        assert_eq!(completed.runs[0].id, second.id.to_string());

        let named = registry.list(&RunFilter {
            name: Some("a".into()),
            limit: 10,
            ..Default::default()
        });
        assert_eq!(named.runs.len(), 2);

        let page = registry.list(&RunFilter {
            limit: 2,
            ..Default::default()
        });
        assert_eq!(page.runs.len(), 2);
        let cursor = page.next_cursor.clone().expect("a second page exists");
        let page = registry.list(&RunFilter {
            limit: 2,
            cursor: Some(cursor),
            ..Default::default()
        });
        assert_eq!(page.runs.len(), 1);
        assert_eq!(page.runs[0].id, first.id.to_string());
        assert!(page.next_cursor.is_none());

        // An unknown cursor is an empty page, never a silent restart.
        let page = registry.list(&RunFilter {
            limit: 2,
            cursor: Some("nope".into()),
            ..Default::default()
        });
        assert!(page.runs.is_empty());
        let _ = std::fs::remove_dir_all(registry.state_dir());
    }

    #[test]
    fn a_registry_reopens_onto_the_history_it_left() {
        let registry = registry("reopen");
        let dir = registry.state_dir().to_path_buf();
        let live = registry
            .create(Some("live".into()), record(), None, control())
            .unwrap();
        live.transition(dto::RunStatus::Starting);
        live.transition(dto::RunStatus::Running);
        let done = registry
            .create(Some("done".into()), record(), None, control())
            .unwrap();
        done.transition(dto::RunStatus::Starting);
        done.transition(dto::RunStatus::Running);
        done.complete();
        drop(registry);

        let reopened = RunRegistry::open(&dir);
        assert_eq!(
            reopened.get(&done.id).unwrap().status(),
            dto::RunStatus::Completed
        );
        assert_eq!(
            reopened.get(&live.id).unwrap().status(),
            dto::RunStatus::Interrupted,
            "the V1 does not re-attach: a run left running is interrupted (§6.5)"
        );
        assert_eq!(reopened.active(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_idempotency_key_finds_the_run_it_created() {
        let registry = registry("idempotency");
        let handle = registry
            .create(None, record(), Some("key-1".into()), control())
            .unwrap();
        assert_eq!(
            registry.by_idempotency_key("key-1").map(|run| run.id),
            Some(handle.id)
        );
        assert!(registry.by_idempotency_key("key-2").is_none());
        let _ = std::fs::remove_dir_all(registry.state_dir());
    }
}
