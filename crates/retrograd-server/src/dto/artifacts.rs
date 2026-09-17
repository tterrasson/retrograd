//! Checkpoints and artefacts

use super::*;

schema! {
/// One checkpoint found on disk.
///
/// A directory read, not a run's memory: a run restored from the state directory
/// lists its checkpoints exactly like a live one, and a checkpoint deleted by
/// hand disappears from the listing without anything having to be told.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CheckpointEntry {
    /// `step-000000000400` or `best`, the checkpoint's own id.
    pub id: String,
    pub path: String,
    /// `step` | `best`.
    pub kind: &'static str,
    pub global_step: u64,
    /// Total size of the state directory.
    pub bytes: u64,
    /// Unix seconds of the manifest, i.e. when the checkpoint was published.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub written_at: Option<u64>,
    /// The adapter GGUF exported beside the state directory, when it is there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// False for a directory whose manifest is missing or of another format
    /// version: it is listed rather than hidden, because a client that asked to
    /// resume from it deserves to know why it cannot.
    pub complete: bool,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CheckpointListing {
    /// Absent when the run writes no checkpoints, which is not the same as an
    /// empty directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    /// Newest first.
    pub checkpoints: Vec<CheckpointEntry>,
    /// The id a resume would pick, i.e. what `fork_from` resolves to by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest: Option<String>,
}
}

schema! {
/// One artefact of a run, by a fixed name rather than by path.
///
/// The name is what `GET /v1/runs/{id}/artifacts/{name}` takes. Naming the
/// members of a closed inventory is what keeps the download route from being a
/// path parameter - and therefore from being an arbitrary file reader.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ArtifactEntry {
    /// `adapter` | `run` | `events` | `observe_log` | `observe` |
    /// `checkpoints` | `tensorboard` | `wandb_export`.
    pub name: &'static str,
    pub path: String,
    /// `file` | `directory`.
    pub kind: &'static str,
    /// Whether it is on disk right now. A run that has not finished has no
    /// adapter yet, and that is worth saying rather than omitting.
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Whether `GET.../artifacts/{name}` will serve the bytes. False for a
    /// directory, and for a file over the download limit.
    pub downloadable: bool,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ArtifactListing {
    pub artifacts: Vec<ArtifactEntry>,
}
}
