//! On-disk format of a training checkpoint.
//!
//! A checkpoint is a `.state` directory containing its authoritative adapter,
//! plus a sibling GGUF export for ordinary cold-adapter loading:
//!
//! ```text
//! checkpoints/
//! ├── step-000000000123.gguf     # convenience LoRA export
//! └── step-000000000123.state/  # authoritative atomic unit
//!     ├── adapter.gguf
//!     ├── manifest.msgpack
//!     ├── progress.msgpack
//!     ├── scheduler.msgpack
//!     ├── optimizer.msgpack
//!     ├── rng.msgpack
//!     ├── dataset.msgpack
//!     └── artifacts/
//! ```
//!
//! The GGUF is never enriched with resume state: it stays a portable adapter
//! export that every helper already knows how to load. Each `.msgpack` file
//! holds exactly one versioned Rust structure, so a schema can evolve without
//! turning the checkpoint into an opaque blob.
//!
//! The manifest is written last and is the only marker of a complete
//! checkpoint. Loading refuses an unknown version, a missing declared file, or
//! any mismatch that would silently change the training trajectory.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use retrograd_core::{ArtifactPolicy, Dataset, Progress};
use retrograd_core::{Error, Result};

/// Bumped whenever a structure below changes in a way older readers cannot
/// interpret. Loading a checkpoint with a different value is a hard error.
pub const FORMAT_VERSION: u32 = 3;

/// Directory suffix of the state directory that accompanies `<name>.gguf`.
pub const STATE_SUFFIX: &str = "state";

pub const MANIFEST_FILE: &str = "manifest.msgpack";
pub const PROGRESS_FILE: &str = "progress.msgpack";
pub const SCHEDULER_FILE: &str = "scheduler.msgpack";
pub const OPTIMIZER_FILE: &str = "optimizer.msgpack";
pub const RNG_FILE: &str = "rng.msgpack";
pub const DATASET_FILE: &str = "dataset.msgpack";
pub const ARTIFACTS_DIR: &str = "artifacts";
pub const ADAPTER_FILE: &str = "adapter.gguf";

/// The state files that must exist at the root of a `.state` directory. The
/// manifest is excluded: it is the marker, not a member.
pub const REQUIRED_FILES: [&str; 5] = [
    PROGRESS_FILE,
    SCHEDULER_FILE,
    OPTIMIZER_FILE,
    RNG_FILE,
    DATASET_FILE,
];

/// Written last; its presence means every other file landed intact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    /// `step-000000000123` or `best`; matches the GGUF stem.
    pub checkpoint_id: String,
    pub global_step: u64,
    /// Relative name of the adapter GGUF, so the directory stays movable.
    pub adapter: String,
    /// State files the loader must find. Ordered for a stable encoding.
    pub files: Vec<String>,
    pub app_version: String,
    pub llama_cpp_commit: String,
    /// Architecture and shape hyperparameters, from the runtime.
    pub model_signature: String,
    /// Base-model file size, retained for a precise mismatch diagnostic.
    pub model_bytes: u64,
    /// Fingerprint of the complete base-model file. Shape and size are useful
    /// diagnostics, but only content identity prevents a same-sized GGUF with
    /// different weights from being accepted on resume.
    #[serde(default)]
    pub model_fingerprint: String,
    pub algorithm: String,
    /// Fingerprint of every run setting that can change the training
    /// trajectory (algorithm geometry, sampling, objective and evaluation).
    pub trajectory_signature: String,
    /// Coarsest unit a resume may restart at, e.g. `epoch` or `update`.
    pub resume_boundary: String,
    /// Optional artifacts and their policy, keyed by file name.
    #[serde(default)]
    pub artifacts: BTreeMap<String, ArtifactPolicy>,
}

/// Learning-rate schedule. A run whose configuration disagrees with this is
/// refused rather than resumed onto a different trajectory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scheduler {
    pub version: u32,
    pub step: u64,
    pub total_steps: u64,
    pub last_learning_rate: f32,
    /// `constant`, `linear`, or `cosine`.
    pub kind: String,
    pub learning_rate: f32,
    pub warmup_steps: u64,
}

/// Optimizer state. `has_moments` is false both for a checkpoint taken before
/// the optimizer graph existed and for an optimizer that keeps no per-parameter
/// state; `graph_ready` is what tells those two apart.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Optimizer {
    pub version: u32,
    /// `adamw` or `sgd`.
    pub kind: String,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    /// Bias-correction counter; ggml starts it at 1.
    pub iter: i64,
    pub has_moments: bool,
    /// Whether the optimizer graph existed when this checkpoint was written,
    /// and so whether `iter` and the RNG state mean anything.
    ///
    /// Defaulted rather than required: every checkpoint written before this
    /// field existed came from AdamW, where `has_moments` answered the same
    /// question, and [`Optimizer::graph_was_ready`] is the reader that says so.
    #[serde(default)]
    pub graph_ready: bool,
    /// One entry per trainable parameter, keyed by its stable tensor name.
    /// Iteration order is never used to match them back.
    pub moments: Vec<Moments>,
}

impl Optimizer {
    /// Whether the optimizer graph existed, reading an older record correctly.
    ///
    /// Before `graph_ready` existed the only selectable optimizer was AdamW, so
    /// "carries moments" and "had a graph" were the same fact and the absent
    /// field is recoverable rather than unknown.
    pub fn graph_was_ready(&self) -> bool {
        self.graph_ready || self.has_moments
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub name: String,
    /// Four ggml dimensions, so a shape change is caught before any write.
    pub shape: [i64; 4],
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

/// RNG states, one entry per domain so components cannot collide.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rng {
    pub version: u32,
    /// mt19937 state of the ggml optimizer, as its serialized word list.
    pub runtime_mt19937: Option<String>,
    /// Seeds of the Rust-side samplers, keyed by domain (`sampling`, `lora`…).
    #[serde(default)]
    pub seeds: BTreeMap<String, u64>,
}

/// Everything one checkpoint holds, assembled by the trainer and written as
/// one atomic unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub manifest: Manifest,
    pub progress: Progress,
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub rng: Rng,
    pub dataset: Dataset,
    /// Extra per-algorithm files, written under `artifacts/`.
    pub artifacts: BTreeMap<String, Vec<u8>>,
}

/// Reads only the manifest of a checkpoint directory.
///
/// [`Checkpoint::read`] loads the optimizer moments too - one `f32` pair per
/// trainable parameter - which is right for a resume and far too much for a
/// caller that wants to *describe* a checkpoint: listing a directory of twenty
/// snapshots would deserialize hundreds of megabytes to print their step
/// numbers. The manifest is written last, so its presence still means the
/// directory is complete.
pub fn read_manifest(state_dir: &Path) -> Result<Manifest> {
    let state_dir = readable_state_dir(state_dir)?;
    let manifest: Manifest = read_file(&state_dir, MANIFEST_FILE).map_err(|_| {
        Error::checkpoint(format!(
            "{} is not a complete checkpoint: {MANIFEST_FILE} is missing or unreadable",
            state_dir.display()
        ))
    })?;
    expect_version(manifest.format_version, "manifest")?;
    Ok(manifest)
}

/// Resolves the state directory that belongs to `path`, which may be either the
/// directory itself or the adapter GGUF next to it.
pub fn state_dir_for(path: &Path) -> PathBuf {
    if path.extension().and_then(|value| value.to_str()) == Some(STATE_SUFFIX) {
        return path.to_path_buf();
    }
    path.with_extension(STATE_SUFFIX)
}

/// The adapter GGUF that belongs to a state directory.
pub fn adapter_for(state_dir: &Path, manifest: &Manifest) -> PathBuf {
    readable_state_dir(state_dir)
        .unwrap_or_else(|_| state_dir.to_path_buf())
        .join(&manifest.adapter)
}

fn backup_dir_for(state_dir: &Path) -> Result<PathBuf> {
    let parent = state_dir.parent().ok_or_else(|| {
        Error::checkpoint("a checkpoint state directory must have a parent directory")
    })?;
    let name = state_dir
        .file_name()
        .ok_or_else(|| Error::checkpoint("a checkpoint state directory must have a name"))?
        .to_string_lossy();
    Ok(parent.join(format!(".{name}.backup")))
}

/// Returns the complete authoritative directory. If a process died between
/// moving the previous checkpoint aside and publishing its replacement, the
/// backup is still a valid checkpoint and is used transparently.
fn readable_state_dir(state_dir: &Path) -> Result<PathBuf> {
    if state_dir.is_dir() {
        return Ok(state_dir.to_path_buf());
    }
    let backup = backup_dir_for(state_dir)?;
    if backup.is_dir() {
        return Ok(backup);
    }
    Ok(state_dir.to_path_buf())
}

fn encode<T: Serialize>(value: &T, what: &str) -> Result<Vec<u8>> {
    rmp_serde::to_vec_named(value)
        .map_err(|error| Error::checkpoint(format!("failed to encode checkpoint {what}: {error}")))
}

fn decode<T: DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T> {
    rmp_serde::from_slice(bytes)
        .map_err(|error| Error::checkpoint(format!("failed to decode checkpoint {what}: {error}")))
}

fn read_file<T: DeserializeOwned>(state_dir: &Path, name: &str) -> Result<T> {
    let path = state_dir.join(name);
    let bytes = fs::read(&path).map_err(|error| {
        Error::checkpoint(format!(
            "checkpoint file {} is unreadable: {error}",
            path.display()
        ))
    })?;
    decode(&bytes, name)
}

fn expect_version(actual: u32, what: &str) -> Result<()> {
    if actual == FORMAT_VERSION {
        return Ok(());
    }
    Err(Error::checkpoint(format!(
        "checkpoint {what} has schema version {actual}, expected {FORMAT_VERSION}"
    )))
}

impl Checkpoint {
    /// Writes the checkpoint atomically: everything lands in a sibling
    /// temporary directory, the manifest is written last, then the directory is
    /// renamed into place. An interrupted write leaves no directory a loader
    /// would accept.
    ///
    /// `write_adapter` receives a path inside the temporary directory. The
    /// adapter and state files therefore become visible as one directory
    /// transaction; a replacement keeps the prior directory as a recoverable
    /// backup until the new one has landed.
    pub fn write(
        &self,
        state_dir: &Path,
        write_adapter: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let span = tracing::info_span!(
            target: "retrograd::checkpoint",
            "checkpoint",
            operation = "write",
            path = %state_dir.display()
        );
        let _entered = span.enter();
        let parent = state_dir.parent().ok_or_else(|| {
            Error::checkpoint("a checkpoint state directory must have a parent directory")
        })?;
        fs::create_dir_all(parent)?;
        static TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let token = format!(
            "{}-{}",
            std::process::id(),
            TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let stem = state_dir
            .file_name()
            .ok_or_else(|| Error::checkpoint("a checkpoint state directory must have a name"))?
            .to_string_lossy()
            .into_owned();
        let temporary_dir = parent.join(format!(".{stem}.tmp-{token}"));
        let temporary_adapter = temporary_dir.join(&self.manifest.adapter);

        let result = self.write_into(&temporary_dir, &temporary_adapter, write_adapter);
        if result.is_err() {
            let _ = fs::remove_dir_all(&temporary_dir);
            return result;
        }

        let backup = backup_dir_for(state_dir)?;
        // A backup can only remain after an interrupted replacement. If the
        // canonical directory exists it is newer, so the backup is stale.
        if state_dir.is_dir() && backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        if state_dir.exists() {
            fs::rename(state_dir, &backup)?;
        }
        if let Err(error) = fs::rename(&temporary_dir, state_dir) {
            if backup.is_dir() && !state_dir.exists() {
                let _ = fs::rename(&backup, state_dir);
            }
            let _ = fs::remove_dir_all(&temporary_dir);
            return Err(error.into());
        }
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }

        // Publish a conventional sibling GGUF as a non-authoritative export.
        // Resume always reads the copy inside `.state`, so an interrupted copy
        // can never pair new weights with old optimizer state.
        let export = state_dir.with_extension("gguf");
        let temporary_export = parent.join(format!(".{stem}.export-{token}.gguf"));
        fs::copy(state_dir.join(&self.manifest.adapter), &temporary_export)?;
        if export.exists() {
            fs::remove_file(&export)?;
        }
        fs::rename(&temporary_export, export)?;
        tracing::info!(target: "retrograd::checkpoint", "checkpoint published");
        Ok(())
    }

    fn write_into(
        &self,
        temporary_dir: &Path,
        temporary_adapter: &Path,
        write_adapter: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let _ = fs::remove_dir_all(temporary_dir);
        fs::create_dir_all(temporary_dir)?;
        write_adapter(temporary_adapter)?;

        fs::write(
            temporary_dir.join(PROGRESS_FILE),
            encode(&self.progress, "progress")?,
        )?;
        fs::write(
            temporary_dir.join(SCHEDULER_FILE),
            encode(&self.scheduler, "scheduler")?,
        )?;
        fs::write(
            temporary_dir.join(OPTIMIZER_FILE),
            encode(&self.optimizer, "optimizer")?,
        )?;
        fs::write(temporary_dir.join(RNG_FILE), encode(&self.rng, "rng")?)?;
        fs::write(
            temporary_dir.join(DATASET_FILE),
            encode(&self.dataset, "dataset")?,
        )?;
        if !self.artifacts.is_empty() {
            let artifacts = temporary_dir.join(ARTIFACTS_DIR);
            fs::create_dir_all(&artifacts)?;
            for (name, bytes) in &self.artifacts {
                fs::write(artifacts.join(name), bytes)?;
            }
        }
        // Last, and only after everything above landed.
        fs::write(
            temporary_dir.join(MANIFEST_FILE),
            encode(&self.manifest, "manifest")?,
        )?;
        Ok(())
    }

    /// Reads a checkpoint back. Fails on an unknown schema version, a missing
    /// declared state file, or a missing required artifact; a partially written
    /// directory is never loaded piecemeal.
    pub fn read(state_dir: &Path) -> Result<Self> {
        let span = tracing::info_span!(
            target: "retrograd::checkpoint",
            "checkpoint",
            operation = "read",
            path = %state_dir.display()
        );
        let _entered = span.enter();
        let state_dir = readable_state_dir(state_dir)?;
        let manifest: Manifest = read_file(&state_dir, MANIFEST_FILE).map_err(|_| {
            Error::checkpoint(format!(
                "{} is not a complete checkpoint: {MANIFEST_FILE} is missing or unreadable",
                state_dir.display()
            ))
        })?;
        expect_version(manifest.format_version, "manifest")?;
        for name in &manifest.files {
            if !state_dir.join(name).is_file() {
                return Err(Error::checkpoint(format!(
                    "checkpoint {} declares {name} but the file is missing",
                    state_dir.display()
                )));
            }
        }

        let progress: Progress = read_file(&state_dir, PROGRESS_FILE)?;
        expect_version(progress.version, "progress")?;
        let scheduler: Scheduler = read_file(&state_dir, SCHEDULER_FILE)?;
        expect_version(scheduler.version, "scheduler")?;
        let optimizer: Optimizer = read_file(&state_dir, OPTIMIZER_FILE)?;
        expect_version(optimizer.version, "optimizer")?;
        let rng: Rng = read_file(&state_dir, RNG_FILE)?;
        expect_version(rng.version, "rng")?;
        let dataset: Dataset = read_file(&state_dir, DATASET_FILE)?;
        expect_version(dataset.version, "dataset")?;

        let mut artifacts = BTreeMap::new();
        for (name, policy) in &manifest.artifacts {
            let path = state_dir.join(ARTIFACTS_DIR).join(name);
            match fs::read(&path) {
                Ok(bytes) => {
                    artifacts.insert(name.clone(), bytes);
                }
                Err(error) if *policy == ArtifactPolicy::Required => {
                    return Err(Error::checkpoint(format!(
                        "checkpoint artifact {} is required but unreadable: {error}",
                        path.display()
                    )));
                }
                Err(_) => {}
            }
        }

        Ok(Self {
            manifest,
            progress,
            scheduler,
            optimizer,
            rng,
            dataset,
            artifacts,
        })
    }

    /// Refuses a resume whose run configuration would change the trajectory.
    /// Everything compared here is something a mismatch would silently corrupt:
    /// the model, the dataset, and the schedule.
    pub fn check_compatible(&self, expected: &Compatibility) -> Result<()> {
        let mismatch = |field: &str, saved: String, found: String| {
            Err(Error::checkpoint(format!(
                "checkpoint is incompatible with this run: {field} was {saved}, now {found}"
            )))
        };
        if self.manifest.model_signature != expected.model_signature {
            return mismatch(
                "model",
                self.manifest.model_signature.clone(),
                expected.model_signature.clone(),
            );
        }
        if self.manifest.model_bytes != expected.model_bytes {
            return mismatch(
                "model size",
                self.manifest.model_bytes.to_string(),
                expected.model_bytes.to_string(),
            );
        }
        if self.manifest.model_fingerprint != expected.model_fingerprint {
            return mismatch(
                "model content",
                self.manifest.model_fingerprint.clone(),
                expected.model_fingerprint.clone(),
            );
        }
        if self.manifest.algorithm != expected.algorithm {
            return mismatch(
                "algorithm",
                self.manifest.algorithm.clone(),
                expected.algorithm.clone(),
            );
        }
        if self.manifest.trajectory_signature != expected.trajectory_signature {
            return mismatch(
                "training trajectory",
                self.manifest.trajectory_signature.clone(),
                expected.trajectory_signature.clone(),
            );
        }
        if self.dataset.fingerprint != expected.dataset_fingerprint {
            return mismatch(
                "dataset",
                self.dataset.fingerprint.clone(),
                expected.dataset_fingerprint.clone(),
            );
        }
        if self.scheduler.kind != expected.scheduler_kind
            || self.scheduler.learning_rate != expected.learning_rate
            || self.scheduler.warmup_steps != expected.warmup_steps
            || expected
                .total_steps
                .is_some_and(|total| total != self.scheduler.total_steps)
        {
            return mismatch(
                "learning-rate schedule",
                format!(
                    "{} lr={} warmup={} total={}",
                    self.scheduler.kind,
                    self.scheduler.learning_rate,
                    self.scheduler.warmup_steps,
                    self.scheduler.total_steps
                ),
                format!(
                    "{} lr={} warmup={} total={:?}",
                    expected.scheduler_kind,
                    expected.learning_rate,
                    expected.warmup_steps,
                    expected.total_steps
                ),
            );
        }
        // The optimizer itself, before its hyperparameters: AdamW moments mean
        // nothing to an SGD step, and an SGD checkpoint resumed under AdamW
        // would pair a warm iteration counter with cold moments. Skipped for a
        // checkpoint whose graph never existed - it pins no trajectory.
        if self.optimizer.graph_was_ready() && self.optimizer.kind != expected.optimizer_kind {
            return mismatch(
                "optimizer",
                self.optimizer.kind.clone(),
                expected.optimizer_kind.clone(),
            );
        }
        if self.optimizer.weight_decay != expected.weight_decay
            || self.optimizer.max_grad_norm != expected.max_grad_norm
        {
            return mismatch(
                "optimizer hyperparameters",
                format!(
                    "weight_decay={} max_grad_norm={}",
                    self.optimizer.weight_decay, self.optimizer.max_grad_norm
                ),
                format!(
                    "weight_decay={} max_grad_norm={}",
                    expected.weight_decay, expected.max_grad_norm
                ),
            );
        }
        if self.progress.global_step != self.scheduler.step {
            return Err(Error::checkpoint(format!(
                "checkpoint is inconsistent: progress step {} does not match scheduler step {}",
                self.progress.global_step, self.scheduler.step
            )));
        }
        Ok(())
    }
}

/// The run-side values a checkpoint is validated against.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Compatibility {
    pub model_signature: String,
    pub model_bytes: u64,
    pub model_fingerprint: String,
    pub algorithm: String,
    pub trajectory_signature: String,
    pub dataset_fingerprint: String,
    pub scheduler_kind: String,
    pub learning_rate: f32,
    pub warmup_steps: u64,
    /// Total optimizer steps of the schedule horizon. `None` skips the check:
    /// the horizon is a function of the algorithm, the dataset and the epoch
    /// counts, all of which are pinned by the fields above, so a caller that
    /// cannot recompute it without duplicating the runtime's arithmetic may
    /// leave it out rather than risk a wrong expectation.
    pub total_steps: Option<u64>,
    /// `adamw` or `sgd`, as the run configures it. Compared with what the
    /// checkpoint recorded rather than with what the runtime happens to build.
    pub optimizer_kind: String,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

/// Stable 128-bit content fingerprint, rendered as hex: two 64-bit FNV-1a-style
/// lanes, the second run over the bitwise complement of each byte so the two
/// lanes do not trivially agree. Used for the dataset identity in a manifest,
/// where a cryptographic digest would be overkill and an extra dependency.
pub fn fingerprint(bytes: &[u8]) -> String {
    let mut low: u64 = 0xcbf2_9ce4_8422_2325;
    let mut high: u64 = 0x9dcf_1a8b_4c37_5f11;
    for byte in bytes {
        low = (low ^ *byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        high = (high ^ !*byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{low:016x}{high:016x}")
}

/// [`fingerprint_file`] behind a process-wide cache keyed by path, size and
/// modification time.
///
/// A training run writes one checkpoint per interval and each of them records
/// the base model's content identity; hashing a multi-gigabyte GGUF every time
/// would put tens of seconds of I/O on the training thread for an answer that
/// cannot have changed. The cached entry is used only while size and mtime
/// still agree, so a model replaced under a live run is re-read rather than
/// asserted from memory.
pub fn fingerprint_file_cached(path: &std::path::Path) -> retrograd_core::Result<String> {
    type Cache = std::sync::Mutex<
        std::collections::HashMap<std::path::PathBuf, (u64, Option<std::time::SystemTime>, String)>,
    >;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();

    let metadata = std::fs::metadata(path)?;
    let identity = (metadata.len(), metadata.modified().ok());
    let cache = CACHE.get_or_init(Default::default);
    let cached = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(path)
        .filter(|(bytes, modified, _)| (*bytes, *modified) == identity)
        .map(|(_, _, fingerprint)| fingerprint.clone());
    if let Some(fingerprint) = cached {
        return Ok(fingerprint);
    }
    let fingerprint = fingerprint_file(path)?;
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(
            path.to_path_buf(),
            (identity.0, identity.1, fingerprint.clone()),
        );
    Ok(fingerprint)
}

/// SHA-256 of a large file, streamed so model identity does not require holding
/// a GGUF in memory.
pub fn fingerprint_file(path: &std::path::Path) -> retrograd_core::Result<String> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(retrograd_core::hex_lower(&digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Checkpoint {
        Checkpoint {
            manifest: Manifest {
                format_version: FORMAT_VERSION,
                checkpoint_id: "step-000000000042".into(),
                global_step: 42,
                adapter: ADAPTER_FILE.into(),
                files: REQUIRED_FILES.iter().map(|name| name.to_string()).collect(),
                app_version: "0.1.0".into(),
                llama_cpp_commit: "deadbeef".into(),
                model_signature: "arch=llama n_embd=64".into(),
                model_bytes: 1234,
                model_fingerprint: "model-content".into(),
                algorithm: "sft".into(),
                trajectory_signature: "trajectory-1".into(),
                resume_boundary: "epoch".into(),
                artifacts: BTreeMap::new(),
            },
            progress: Progress {
                version: FORMAT_VERSION,
                epoch: 2,
                global_step: 42,
                cursor: 7,
                algorithm: "sft".into(),
                phase: "train".into(),
                best_eval: Some(0.25),
                stale_evaluations: 1,
                kl_multiplier: None,
            },
            scheduler: Scheduler {
                version: FORMAT_VERSION,
                step: 42,
                total_steps: 100,
                last_learning_rate: 1.0e-4,
                kind: "cosine".into(),
                learning_rate: 2.0e-4,
                warmup_steps: 5,
            },
            optimizer: Optimizer {
                version: FORMAT_VERSION,
                kind: "adamw".into(),
                learning_rate: 2.0e-4,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                iter: 43,
                has_moments: true,
                graph_ready: true,
                moments: vec![Moments {
                    name: "blk.0.attn_q.weight.lora_a".into(),
                    shape: [4, 8, 1, 1],
                    m: vec![0.5; 32],
                    v: vec![0.25; 32],
                }],
            },
            rng: Rng {
                version: FORMAT_VERSION,
                runtime_mt19937: Some("1 2 3".into()),
                seeds: BTreeMap::from([("sampling".into(), 99_u64)]),
            },
            dataset: Dataset {
                version: FORMAT_VERSION,
                path: "data/train.jsonl".into(),
                fingerprint: fingerprint(b"rows"),
                examples: 10,
                row_width: 256,
                format: "chat".into(),
                permutation: vec![2, 0, 1],
                cursor: 7,
            },
            artifacts: BTreeMap::new(),
        }
    }

    fn compatibility() -> Compatibility {
        Compatibility {
            model_signature: "arch=llama n_embd=64".into(),
            model_bytes: 1234,
            model_fingerprint: "model-content".into(),
            algorithm: "sft".into(),
            trajectory_signature: "trajectory-1".into(),
            dataset_fingerprint: fingerprint(b"rows"),
            scheduler_kind: "cosine".into(),
            learning_rate: 2.0e-4,
            warmup_steps: 5,
            total_steps: Some(100),
            optimizer_kind: "adamw".into(),
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }

    #[test]
    fn every_state_file_round_trips_through_messagepack() {
        let checkpoint = sample();
        // Each file is its own schema, so each one is checked on its own.
        assert_eq!(
            decode::<Manifest>(&encode(&checkpoint.manifest, "m").unwrap(), "m").unwrap(),
            checkpoint.manifest
        );
        assert_eq!(
            decode::<Progress>(&encode(&checkpoint.progress, "p").unwrap(), "p").unwrap(),
            checkpoint.progress
        );
        assert_eq!(
            decode::<Scheduler>(&encode(&checkpoint.scheduler, "s").unwrap(), "s").unwrap(),
            checkpoint.scheduler
        );
        assert_eq!(
            decode::<Optimizer>(&encode(&checkpoint.optimizer, "o").unwrap(), "o").unwrap(),
            checkpoint.optimizer
        );
        assert_eq!(
            decode::<Rng>(&encode(&checkpoint.rng, "r").unwrap(), "r").unwrap(),
            checkpoint.rng
        );
        assert_eq!(
            decode::<Dataset>(&encode(&checkpoint.dataset, "d").unwrap(), "d").unwrap(),
            checkpoint.dataset
        );
    }

    #[test]
    fn a_written_checkpoint_reads_back_identically() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let checkpoint = sample();
        checkpoint
            .write(&state, |path| {
                std::fs::write(path, b"GGUF").map_err(Into::into)
            })
            .unwrap();
        // Resume owns an atomic adapter inside the directory, while the sibling
        // remains a conventional cold-load export.
        assert!(state.join(ADAPTER_FILE).is_file());
        assert!(root.join("step-000000000042.gguf").is_file());
        assert_eq!(Checkpoint::read(&state).unwrap(), checkpoint);
    }

    #[test]
    fn replacing_a_checkpoint_never_mixes_adapter_and_state() {
        let root = tempdir();
        let state = root.join("best.state");
        let first = sample();
        first
            .write(&state, |path| {
                std::fs::write(path, b"FIRST").map_err(Into::into)
            })
            .unwrap();

        let mut second = sample();
        second.progress.best_eval = Some(0.5);
        second
            .write(&state, |path| {
                std::fs::write(path, b"SECOND").map_err(Into::into)
            })
            .unwrap();

        assert_eq!(Checkpoint::read(&state).unwrap(), second);
        assert_eq!(std::fs::read(state.join(ADAPTER_FILE)).unwrap(), b"SECOND");
        assert_eq!(std::fs::read(root.join("best.gguf")).unwrap(), b"SECOND");
        assert!(!backup_dir_for(&state).unwrap().exists());
    }

    #[test]
    fn an_interrupted_replacement_falls_back_to_the_complete_backup() {
        let root = tempdir();
        let state = root.join("best.state");
        let checkpoint = sample();
        checkpoint
            .write(&state, |path| {
                std::fs::write(path, b"FIRST").map_err(Into::into)
            })
            .unwrap();
        let backup = backup_dir_for(&state).unwrap();
        std::fs::rename(&state, &backup).unwrap();

        assert_eq!(Checkpoint::read(&state).unwrap(), checkpoint);
        assert_eq!(
            std::fs::read(adapter_for(&state, &checkpoint.manifest)).unwrap(),
            b"FIRST"
        );
    }

    #[test]
    fn a_failing_adapter_write_leaves_nothing_behind() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let error = sample()
            .write(&state, |_| Err(Error::runtime("disk on fire")))
            .unwrap_err();
        assert!(error.to_string().contains("disk on fire"));
        assert!(!state.exists());
        assert!(!root.join("step-000000000042.gguf").exists());
        // No temporary residue either: an interrupted write is invisible.
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    }

    #[test]
    fn a_directory_without_a_manifest_is_not_a_checkpoint() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let checkpoint = sample();
        checkpoint
            .write(&state, |path| {
                std::fs::write(path, b"GGUF").map_err(Into::into)
            })
            .unwrap();
        std::fs::remove_file(state.join(MANIFEST_FILE)).unwrap();
        let error = Checkpoint::read(&state).unwrap_err();
        assert!(error.to_string().contains("not a complete checkpoint"));
    }

    #[test]
    fn a_declared_state_file_that_is_missing_is_refused() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        sample()
            .write(&state, |path| {
                std::fs::write(path, b"GGUF").map_err(Into::into)
            })
            .unwrap();
        std::fs::remove_file(state.join(OPTIMIZER_FILE)).unwrap();
        let error = Checkpoint::read(&state).unwrap_err();
        assert!(error.to_string().contains(OPTIMIZER_FILE));
    }

    #[test]
    fn an_unknown_schema_version_is_refused() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let mut checkpoint = sample();
        checkpoint.manifest.format_version = FORMAT_VERSION + 1;
        checkpoint
            .write(&state, |path| {
                std::fs::write(path, b"GGUF").map_err(Into::into)
            })
            .unwrap();
        let error = Checkpoint::read(&state).unwrap_err();
        assert!(error.to_string().contains("schema version"));
    }

    #[test]
    fn a_required_artifact_that_is_missing_is_refused_but_a_recreatable_one_is_not() {
        let root = tempdir();
        let mut checkpoint = sample();
        checkpoint
            .manifest
            .artifacts
            .insert("ppo_value.msgpack".into(), ArtifactPolicy::Required);
        checkpoint
            .manifest
            .artifacts
            .insert("metrics.msgpack".into(), ArtifactPolicy::Recreatable);
        checkpoint
            .artifacts
            .insert("ppo_value.msgpack".into(), b"value".to_vec());
        checkpoint
            .artifacts
            .insert("metrics.msgpack".into(), b"metrics".to_vec());

        let state = root.join("step-000000000042.state");
        checkpoint
            .write(&state, |path| {
                std::fs::write(path, b"GGUF").map_err(Into::into)
            })
            .unwrap();

        std::fs::remove_file(state.join(ARTIFACTS_DIR).join("metrics.msgpack")).unwrap();
        let loaded = Checkpoint::read(&state).unwrap();
        assert!(!loaded.artifacts.contains_key("metrics.msgpack"));

        std::fs::remove_file(state.join(ARTIFACTS_DIR).join("ppo_value.msgpack")).unwrap();
        let error = Checkpoint::read(&state).unwrap_err();
        assert!(error.to_string().contains("required"));
    }

    /// "Keeps no state" and "was never initialized" are different states, and a
    /// resume must not read the first as the second.
    #[test]
    fn a_zero_slot_optimizer_still_counts_as_initialized() {
        let mut checkpoint = sample();
        checkpoint.optimizer.kind = "sgd".into();
        checkpoint.optimizer.has_moments = false;
        checkpoint.optimizer.graph_ready = true;
        checkpoint.optimizer.moments.clear();
        assert!(checkpoint.optimizer.graph_was_ready());

        let mut expected = compatibility();
        expected.optimizer_kind = "sgd".into();
        checkpoint.check_compatible(&expected).unwrap();
        // ... and the optimizer it names is still compared, precisely because
        // the empty slot list no longer says which one wrote it.
        let error = checkpoint
            .check_compatible(&compatibility())
            .expect_err("adamw cannot resume an sgd trajectory");
        assert!(error.to_string().contains("optimizer was sgd"), "{error}");
    }

    /// A record written before `graph_ready` existed came from AdamW, where
    /// carrying moments and having a graph were the same fact.
    #[test]
    fn an_older_record_recovers_its_missing_graph_flag() {
        let mut checkpoint = sample();
        checkpoint.optimizer.graph_ready = false;
        assert!(checkpoint.optimizer.has_moments);
        assert!(checkpoint.optimizer.graph_was_ready());

        checkpoint.optimizer.has_moments = false;
        checkpoint.optimizer.moments.clear();
        assert!(!checkpoint.optimizer.graph_was_ready());
        // Nothing pins a trajectory, so the optimizer name is not compared.
        let mut renamed = checkpoint.clone();
        renamed.optimizer.kind = "sgd".into();
        renamed.check_compatible(&compatibility()).unwrap();
    }

    #[test]
    fn a_matching_configuration_is_compatible() {
        sample().check_compatible(&compatibility()).unwrap();
    }

    #[test]
    fn every_trajectory_changing_difference_is_refused() {
        // Each case is something a silent resume would corrupt: a different
        // model, dataset, schedule, or optimizer regularization.
        type Case = (&'static str, fn(&mut Compatibility));
        let cases: [Case; 10] = [
            ("model", |c| c.model_signature = "arch=qwen3".into()),
            ("model size", |c| c.model_bytes = 999),
            ("model content", |c| c.model_fingerprint = "other".into()),
            ("algorithm", |c| c.algorithm = "grpo".into()),
            ("training trajectory", |c| {
                c.trajectory_signature = "other".into()
            }),
            ("dataset", |c| c.dataset_fingerprint = "0".into()),
            ("learning-rate schedule", |c| c.learning_rate = 1.0e-3),
            ("learning-rate schedule", |c| c.total_steps = Some(200)),
            ("optimizer hyperparameters", |c| c.weight_decay = 0.5),
            ("optimizer", |c| c.optimizer_kind = "sgd".into()),
        ];
        for (field, mutate) in cases {
            let mut expected = compatibility();
            mutate(&mut expected);
            let error = sample().check_compatible(&expected).unwrap_err();
            assert!(
                error.to_string().contains(field),
                "expected a {field} mismatch, got {error}"
            );
        }
    }

    #[test]
    fn a_progress_and_scheduler_step_disagreement_is_refused() {
        let mut checkpoint = sample();
        checkpoint.progress.global_step = 41;
        let error = checkpoint.check_compatible(&compatibility()).unwrap_err();
        assert!(error.to_string().contains("inconsistent"));
    }

    #[test]
    fn state_directories_resolve_from_either_the_adapter_or_themselves() {
        let gguf = Path::new("/runs/step-000000000042.gguf");
        let state = Path::new("/runs/step-000000000042.state");
        assert_eq!(state_dir_for(gguf), state);
        assert_eq!(state_dir_for(state), state);
        // The manifest keeps a relative name, so a moved directory still
        // resolves its own adapter.
        assert_eq!(
            adapter_for(state, &sample().manifest),
            state.join(ADAPTER_FILE)
        );
    }

    #[test]
    fn fingerprints_separate_content_and_order() {
        assert_eq!(fingerprint(b"abc"), fingerprint(b"abc"));
        assert_ne!(fingerprint(b"abc"), fingerprint(b"abd"));
        assert_ne!(fingerprint(b"abc"), fingerprint(b"cba"));
        assert_ne!(fingerprint(b""), fingerprint(b"\0"));
    }

    #[test]
    fn a_cached_file_fingerprint_answers_the_hash_and_follows_the_content() {
        let dir = tempdir();
        let model = dir.join("model.gguf");
        fs::write(&model, b"weights-a").unwrap();
        let direct = fingerprint_file(&model).unwrap();
        assert_eq!(fingerprint_file_cached(&model).unwrap(), direct);
        // Same call twice is the cache hit the checkpoint loop depends on.
        assert_eq!(fingerprint_file_cached(&model).unwrap(), direct);

        // A different length invalidates the entry whatever the clock did.
        fs::write(&model, b"weights-a-longer").unwrap();
        let replaced = fingerprint_file_cached(&model).unwrap();
        assert_ne!(replaced, direct);
        assert_eq!(replaced, fingerprint_file(&model).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Unique scratch directory; removed by the OS, not worth a dependency.
    fn tempdir() -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "retrograd-checkpoint-{}-{index}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }
}
