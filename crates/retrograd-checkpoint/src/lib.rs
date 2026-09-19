//! On-disk format of a training checkpoint.
//!
//! A checkpoint is a `.state` directory holding everything a resume needs,
//! plus - for a run that has an adapter - a sibling GGUF export for ordinary
//! cold-adapter loading:
//!
//! ```text
//! checkpoints/
//! ├── step-000000000123.gguf     # convenience LoRA export, adapter runs only
//! └── step-000000000123.state/  # authoritative atomic unit
//!     ├── adapter.gguf           # present when the run trains an adapter
//!     ├── trainable.gguf         # present when the run trains base tensors
//!     ├── optimizer-state.bin    # slot payloads, addressed by the manifest
//!     ├── manifest.msgpack
//!     ├── progress.msgpack
//!     ├── scheduler.msgpack
//!     ├── optimizer.msgpack
//!     ├── rng.msgpack
//!     ├── dataset.msgpack
//!     └── artifacts/
//! ```
//!
//! The sibling GGUF is never enriched with resume state: it stays a portable
//! adapter export that every helper already knows how to load. Each `.msgpack`
//! file holds exactly one versioned Rust structure, so a schema can evolve
//! without turning the checkpoint into an opaque blob.
//!
//! Optimizer state is the one thing that is *not* msgpack. A slot payload is a
//! parameter-sized buffer, and a run has one per slot per parameter; encoding
//! them into the same document would make reading a step number cost the whole
//! optimizer. They are concatenated into `optimizer-state.bin` in the order
//! the manifest lists them, each slot a `(offset, n_bytes)` range, so both ends
//! stream through a bounded staging buffer instead of materializing the state.
//! Payload bytes are canonical little-endian, as the backends produce them.
//!
//! What a run *leaves behind* depends on what it trained, and the two halves
//! are independent: an adapter run writes `adapter.gguf`, a base-weight run
//! writes `trainable.gguf` with the absolute trained values, and a hybrid run
//! writes both. Absolute values rather than deltas - a delta is interpretable
//! only beside the exact GGUF it came from, and the base model's fingerprint
//! is already compared on resume.
//!
//! The manifest is written last and is the only marker of a complete
//! checkpoint. Loading refuses an unknown version, a missing declared file, or
//! any mismatch that would silently change the training trajectory.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use retrograd_core::{ArtifactPolicy, Dataset, Progress};
use retrograd_core::{Error, Result};

/// Bumped whenever a structure below changes in a way older readers cannot
/// interpret. Loading a checkpoint with a different value is a hard error.
///
/// Version 1 describes a directory whose optimizer state is a typed slot table
/// with its own shared scope, and whose trained values are an optional adapter
/// beside an optional trainable bundle. The two belong to one version because
/// both describe what the directory contains, and a reader that knew only one
/// of them would accept a checkpoint it cannot restore.
pub const FORMAT_VERSION: u32 = 1;

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
/// Absolute values of the base tensors a run trained.
pub const TRAINABLE_FILE: &str = "trainable.gguf";
/// Concatenated optimizer slot payloads, addressed by [`StateSlot`].
pub const OPTIMIZER_STATE_FILE: &str = "optimizer-state.bin";

/// Largest host staging buffer a streamed slot transfer uses, in bytes.
///
/// The point of the streaming contract is that neither end holds the whole
/// optimizer state, so the constant is the *bound*, not a tuning knob: a
/// 4 GiB AdamW state moves through 8 MiB of host memory either way.
pub const STAGING_CHUNK_BYTES: usize = 8 * 1024 * 1024;

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
    /// Absent for a run that trains base tensors and no adapter: there is no
    /// adapter-shaped record of what such a run produced, and writing an empty
    /// one would claim there is.
    pub adapter: Option<String>,
    /// The trained base tensors, when the run trains any. Independent of
    /// `adapter`: a hybrid run carries both, a LoRA run neither.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainable: Option<TrainableBundle>,
    /// `lora`, `full`, `partial` or `hybrid`, as the run resolved it. Recorded
    /// beside the files rather than derived from which ones exist, so a resume
    /// refuses a policy change instead of inferring one.
    pub trainable_policy: String,
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

/// Optimizer state: the scalars, the slot table, and which optimizer owns each
/// trainable parameter.
///
/// An empty `slots` list is a fact about the optimizer, not about the
/// checkpoint - SGD keeps no per-parameter state and still has a step counter,
/// a schedule and an RNG state. `graph_ready` is what tells "initialized with
/// zero slots" from "never initialized", and nothing here may use
/// `slots.is_empty()` to decide that the counters should restart.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Optimizer {
    pub version: u32,
    /// `adamw` or `sgd`.
    pub kind: String,
    /// The optimizer's slot layout version. Recorded beside the name because
    /// the name alone does not pin what a slot payload means.
    #[serde(default)]
    pub layout_version: u32,
    /// Every hyperparameter the update read, rendered `name=value` in
    /// declaration order. Includes the optimizer's own defaults, not only the
    /// three a document spells, so a changed coefficient is caught on resume.
    #[serde(default)]
    pub hyperparameters: Vec<String>,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    /// Bias-correction counter; ggml starts it at 1.
    pub iter: i64,
    /// Whether the optimizer graph existed when this checkpoint was written,
    /// and so whether `iter` and the RNG state mean anything.
    pub graph_ready: bool,
    /// Every persistent slot, parameter-scoped and shared, in the order their
    /// payloads are concatenated into [`OPTIMIZER_STATE_FILE`]. Matching on
    /// restore is by `(scope, owner, slot)`; the order is a file layout and
    /// never an identity.
    pub slots: Vec<StateSlot>,
    /// Which optimizer owns each trainable parameter. One entry per parameter
    /// even under a single-optimizer run, and even for a parameter with no
    /// slots: the table is what a mixed run's resume compares, and a table that
    /// only listed parameters with state could not express "this one fell back".
    pub assignment: Vec<ParameterAssignment>,
    /// Total payload length, checked against the file rather than trusted.
    /// A truncated transfer is otherwise a silent partial restore.
    pub state_bytes: u64,
}

impl Optimizer {
    /// Slots of one scope, in file order.
    pub fn slots_in(&self, scope: SlotScope) -> impl Iterator<Item = &StateSlot> {
        self.slots.iter().filter(move |slot| slot.scope == scope)
    }

    /// One recorded hyperparameter by name.
    pub fn hyperparameter(&self, name: &str) -> Option<&str> {
        self.hyperparameters
            .iter()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
    }

    /// Compare parameter ownership and layout before restoring any slots.
    pub fn check_assignment(&self, expected: &[ParameterAssignment]) -> Result<()> {
        let mut saved: Vec<_> = self.assignment.iter().collect();
        let mut found: Vec<_> = expected.iter().collect();
        saved.sort_by(|a, b| a.parameter.cmp(&b.parameter));
        found.sort_by(|a, b| a.parameter.cmp(&b.parameter));
        if saved != found {
            return Err(Error::checkpoint(
                "checkpoint optimizer parameter assignment or layout differs from this run",
            ));
        }
        Ok(())
    }

    /// The `(m, v)` pair of one parameter, for a reader that still thinks in
    /// AdamW momenta.
    ///
    /// Temporary, and deliberately narrow: it refuses any other optimizer
    /// rather than handing back slots 0 and 1 of whatever is there. Reading an
    /// arbitrary layout as moments is the exact confusion the slot table
    /// replaced.
    pub fn adamw_moments(&self, parameter: &str) -> Result<(&StateSlot, &StateSlot)> {
        if self.kind != "adamw" {
            return Err(Error::checkpoint(format!(
                "this checkpoint was written by {}, which keeps no AdamW momenta",
                self.kind
            )));
        }
        let find = |name: &str| {
            self.slots
                .iter()
                .find(|slot| {
                    slot.scope == SlotScope::Parameter
                        && slot.owner == parameter
                        && slot.slot == name
                })
                .ok_or_else(|| {
                    Error::checkpoint(format!(
                        "the checkpoint holds no '{name}' slot for parameter '{parameter}'"
                    ))
                })
        };
        Ok((find("m")?, find("v")?))
    }
}

/// Which namespace a slot belongs to.
///
/// Separate namespaces because the allocation rules differ: a parameter slot
/// exists once per trainable tensor, a shared slot once per owner the
/// optimizer declares - a codebook belongs to the optimizer, not to any one
/// parameter, and allocating it per parameter would be a different algorithm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotScope {
    #[default]
    Parameter,
    Shared,
}

impl SlotScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::Shared => "shared",
        }
    }
}

/// One persistent optimizer tensor, described well enough to be validated
/// against the live layout before a single byte is written back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateSlot {
    pub scope: SlotScope,
    /// Trainable tensor name for a parameter slot, the optimizer-declared
    /// owner for a shared one.
    pub owner: String,
    /// Slot name inside the optimizer's layout: `m`, `v`, `momentum`, ...
    pub slot: String,
    /// ggml type name of the payload (`F32`, `I8`, ...). Stored because the
    /// next optimizer's slots are not all floats, and a payload restored under
    /// the wrong dtype is indistinguishable from a corrupt one.
    pub dtype: String,
    /// Four ggml dimensions, so a shape change is caught before any write.
    pub shape: [i64; 4],
    /// Byte range inside [`OPTIMIZER_STATE_FILE`].
    pub offset: u64,
    pub n_bytes: u64,
}

/// Which optimizer updates one trainable parameter.
///
/// A single-optimizer run records the same name on every row, which is the
/// point: the day a run mixes Muon with an AdamW fallback, the resume compares
/// a table it already had rather than inventing one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterAssignment {
    pub parameter: String,
    pub optimizer: String,
    /// The layout version of the optimizer that wrote this row's slots, not of
    /// the run's chosen one: a run with a fallback carries two layouts in one
    /// table.
    #[serde(default)]
    pub layout_version: u32,
}

/// The trained base tensors of a checkpoint, and how to check the file is the
/// one the manifest describes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainableBundle {
    /// Relative name of the GGUF, so the directory stays movable.
    pub file: String,
    pub bytes: u64,
    /// Content fingerprint of that file.
    pub fingerprint: String,
    /// Fingerprint of the resolved trainable set's canonical manifest, which is
    /// what a resume compares. Deliberately not the file's: two runs that
    /// trained the same tensors to different values share this signature, and
    /// that is exactly the comparison a resume wants.
    pub signature: String,
    pub tensors: Vec<TrainableTensor>,
}

/// One tensor inside a trainable bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainableTensor {
    pub name: String,
    /// `base`, `lora_a` or `lora_b` - a bundle holds base values today, and the
    /// field exists so a composite hybrid bundle does not need a new schema.
    pub role: String,
    pub dtype: String,
    pub shape: [i64; 4],
    pub n_elements: u64,
    pub n_bytes: u64,
    /// Other names the loader accepts for this storage, so a renamed alias is
    /// a diagnosable mismatch rather than a missing tensor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
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

/// The adapter GGUF that belongs to a state directory, when the run has one.
pub fn adapter_for(state_dir: &Path, manifest: &Manifest) -> Option<PathBuf> {
    let adapter = manifest.adapter.as_ref()?;
    Some(
        readable_state_dir(state_dir)
            .unwrap_or_else(|_| state_dir.to_path_buf())
            .join(adapter),
    )
}

/// The trainable bundle that belongs to a state directory, when the run
/// trained base tensors.
pub fn trainable_for(state_dir: &Path, manifest: &Manifest) -> Option<PathBuf> {
    let bundle = manifest.trainable.as_ref()?;
    Some(
        readable_state_dir(state_dir)
            .unwrap_or_else(|_| state_dir.to_path_buf())
            .join(&bundle.file),
    )
}

/// The files a checkpoint's payload writer has to produce, resolved inside the
/// temporary directory the write is staged in.
///
/// Passed rather than derived by the caller because the directory is
/// transactional: a payload written anywhere else would become visible on its
/// own schedule, and an interrupted write could pair new weights with old
/// optimizer state. Each field is `Some` exactly when the manifest declares it.
#[derive(Clone, Debug)]
pub struct CheckpointPaths {
    pub adapter: Option<PathBuf>,
    pub trainable: Option<PathBuf>,
    /// Slot payloads, concatenated in [`Optimizer::slots`] order. `None` when
    /// the optimizer keeps no state at all.
    pub optimizer_state: Option<PathBuf>,
}

/// Streams one slot's payload out of a checkpoint, in bounded chunks.
///
/// A reader rather than a `read_slot(&self) -> Vec<u8>`: the whole point of the
/// separate payload file is that restoring a multi-gigabyte optimizer state
/// never needs a host copy of it, and an API that returned the bytes would
/// undo that at the first call site.
pub struct OptimizerStateReader {
    file: fs::File,
    len: u64,
    path: PathBuf,
}

impl OptimizerStateReader {
    /// Opens the payload file of a checkpoint directory, refusing a length that
    /// disagrees with what the optimizer record declares.
    pub fn open(state_dir: &Path, optimizer: &Optimizer) -> Result<Self> {
        let state_dir = readable_state_dir(state_dir)?;
        let path = state_dir.join(OPTIMIZER_STATE_FILE);
        let file = fs::File::open(&path).map_err(|error| {
            Error::checkpoint(format!(
                "checkpoint optimizer state {} is unreadable: {error}",
                path.display()
            ))
        })?;
        let len = file.metadata()?.len();
        if len != optimizer.state_bytes {
            return Err(Error::checkpoint(format!(
                "checkpoint optimizer state {} is {len} bytes, the manifest declares {}",
                path.display(),
                optimizer.state_bytes
            )));
        }
        Ok(Self { file, len, path })
    }

    /// Feeds `apply` successive chunks of one slot's payload, each with its
    /// offset inside the slot. `staging` bounds the transfer; a slot larger
    /// than it takes several calls rather than a larger buffer.
    pub fn stream(
        &mut self,
        slot: &StateSlot,
        staging: &mut [u8],
        mut apply: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        if staging.is_empty() {
            return Err(Error::checkpoint("a staging buffer must not be empty"));
        }
        let end = slot.offset.checked_add(slot.n_bytes).ok_or_else(|| {
            Error::checkpoint(format!(
                "slot '{}' of '{}' declares a byte range that overflows",
                slot.slot, slot.owner
            ))
        })?;
        if end > self.len {
            return Err(Error::checkpoint(format!(
                "slot '{}' of '{}' ends at {end} in a {} byte payload file {}",
                slot.slot,
                slot.owner,
                self.len,
                self.path.display()
            )));
        }
        self.file.seek(SeekFrom::Start(slot.offset))?;
        let mut done = 0_u64;
        while done < slot.n_bytes {
            let remaining = slot.n_bytes - done;
            let want = remaining.min(staging.len() as u64) as usize;
            self.file
                .read_exact(&mut staging[..want])
                .map_err(|error| {
                    Error::checkpoint(format!(
                        "checkpoint optimizer state {} is truncated: {error}",
                        self.path.display()
                    ))
                })?;
            apply(done, &staging[..want])?;
            done += want as u64;
        }
        Ok(())
    }
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

/// The slot table describes a file layout, so its own consistency is checked
/// before anything reads through it: one row per `(scope, owner, slot)`, ranges
/// that tile the payload without overlapping, and a declared total that matches
/// the rows. A duplicate row is the interesting case - it would restore one
/// slot twice and leave another cold.
fn validate_slot_table(optimizer: &Optimizer) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    let mut next = 0_u64;
    for slot in &optimizer.slots {
        if !seen.insert((slot.scope, slot.owner.as_str(), slot.slot.as_str())) {
            return Err(Error::checkpoint(format!(
                "the checkpoint holds two '{}' slots for {} '{}'",
                slot.slot,
                slot.scope.as_str(),
                slot.owner
            )));
        }
        if slot.offset != next {
            return Err(Error::checkpoint(format!(
                "slot '{}' of '{}' starts at {} where the payload continues at {next}",
                slot.slot, slot.owner, slot.offset
            )));
        }
        next = next.checked_add(slot.n_bytes).ok_or_else(|| {
            Error::checkpoint("the optimizer slot table overflows its byte range")
        })?;
    }
    if next != optimizer.state_bytes {
        return Err(Error::checkpoint(format!(
            "the optimizer slot table covers {next} bytes, the record declares {}",
            optimizer.state_bytes
        )));
    }
    Ok(())
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
    /// `write_payloads` receives paths inside the temporary directory, one per
    /// file the manifest declares. Those files therefore become visible as one
    /// directory transaction; a replacement keeps the prior directory as a
    /// recoverable backup until the new one has landed.
    pub fn write(
        &self,
        state_dir: &Path,
        write_payloads: impl FnOnce(&CheckpointPaths) -> Result<()>,
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

        let result = self.write_into(&temporary_dir, write_payloads);
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
        //
        // Only for a run that has an adapter. A base-weight run's result is a
        // bundle no adapter loader accepts, and publishing it under the name
        // every helper reads as "the adapter" would be a file that loads into
        // the wrong thing rather than a missing convenience.
        if let Some(adapter) = &self.manifest.adapter {
            let export = state_dir.with_extension("gguf");
            let temporary_export = parent.join(format!(".{stem}.export-{token}.gguf"));
            fs::copy(state_dir.join(adapter), &temporary_export)?;
            if export.exists() {
                fs::remove_file(&export)?;
            }
            fs::rename(&temporary_export, export)?;
        }
        tracing::info!(target: "retrograd::checkpoint", "checkpoint published");
        Ok(())
    }

    /// The files the payload writer is asked for, inside `directory`.
    fn payload_paths(&self, directory: &Path) -> CheckpointPaths {
        CheckpointPaths {
            adapter: self
                .manifest
                .adapter
                .as_ref()
                .map(|name| directory.join(name)),
            trainable: self
                .manifest
                .trainable
                .as_ref()
                .map(|bundle| directory.join(&bundle.file)),
            optimizer_state: (self.optimizer.state_bytes > 0)
                .then(|| directory.join(OPTIMIZER_STATE_FILE)),
        }
    }

    fn write_into(
        &self,
        temporary_dir: &Path,
        write_payloads: impl FnOnce(&CheckpointPaths) -> Result<()>,
    ) -> Result<()> {
        let _ = fs::remove_dir_all(temporary_dir);
        fs::create_dir_all(temporary_dir)?;
        let paths = self.payload_paths(temporary_dir);
        write_payloads(&paths)?;
        // Declared and produced are two different things, and the manifest is
        // written from the first. A payload writer that silently skipped a file
        // would publish a directory whose own manifest describes a checkpoint
        // that is not there.
        for (path, what) in [
            (paths.adapter.as_ref(), "adapter"),
            (paths.trainable.as_ref(), "trainable bundle"),
            (paths.optimizer_state.as_ref(), "optimizer state"),
        ] {
            let Some(path) = path else { continue };
            if !path.is_file() {
                return Err(Error::checkpoint(format!(
                    "the checkpoint declares a {what} but none was written to {}",
                    path.display()
                )));
            }
        }
        if let Some(path) = &paths.optimizer_state {
            let written = fs::metadata(path)?.len();
            if written != self.optimizer.state_bytes {
                return Err(Error::checkpoint(format!(
                    "the optimizer state is {written} bytes, the slot table declares {}",
                    self.optimizer.state_bytes
                )));
            }
        }

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
        let mut manifest = self.manifest.clone();
        if let Some(bundle) = &mut manifest.trainable {
            let path = temporary_dir.join(&bundle.file);
            bundle.bytes = fs::metadata(&path)?.len();
            bundle.fingerprint = fingerprint_file(&path)?;
        }
        fs::write(
            temporary_dir.join(MANIFEST_FILE),
            encode(&manifest, "manifest")?,
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

        validate_slot_table(&optimizer)?;
        // Declared payload files have to exist, exactly like the state files
        // above. `manifest.files` lists the msgpack members; these two are
        // named by their own fields and would otherwise go unchecked until a
        // restore tried to open them.
        for path in [
            manifest.adapter.as_ref().map(|name| state_dir.join(name)),
            manifest
                .trainable
                .as_ref()
                .map(|bundle| state_dir.join(&bundle.file)),
            (optimizer.state_bytes > 0).then(|| state_dir.join(OPTIMIZER_STATE_FILE)),
        ]
        .into_iter()
        .flatten()
        {
            if !path.is_file() {
                return Err(Error::checkpoint(format!(
                    "checkpoint {} declares {} but the file is missing",
                    state_dir.display(),
                    path.display()
                )));
            }
        }

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
        // What the run trains, before what updates it. A policy change moves
        // the parameters themselves, so every optimizer comparison below is
        // about a different set of them.
        if self.manifest.trainable_policy != expected.trainable_policy {
            return mismatch(
                "trainable policy",
                self.manifest.trainable_policy.clone(),
                expected.trainable_policy.clone(),
            );
        }
        // And *which* tensors that policy resolved to. Two `partial` runs over
        // different layer ranges agree on the policy and share nothing else;
        // the signature is over the canonical manifest of the resolved set, so
        // a selector change is refused rather than restored into.
        let saved_signature = self
            .manifest
            .trainable
            .as_ref()
            .map(|bundle| bundle.signature.clone())
            .unwrap_or_default();
        if saved_signature != expected.trainable_signature {
            return mismatch(
                "trainable set",
                saved_signature,
                expected.trainable_signature.clone(),
            );
        }
        // The optimizer itself, before its hyperparameters: AdamW slots mean
        // nothing to an SGD step, and an SGD checkpoint resumed under AdamW
        // would pair a warm iteration counter with cold state. Skipped for a
        // checkpoint whose graph never existed - it pins no trajectory.
        if self.optimizer.graph_ready && self.optimizer.kind != expected.optimizer_kind {
            return mismatch(
                "optimizer",
                self.optimizer.kind.clone(),
                expected.optimizer_kind.clone(),
            );
        }
        // Then the layout and the coefficients it declares, both skipped for a
        // cold checkpoint: nothing has run, so nothing has been committed to.
        if self.optimizer.graph_ready
            && self.optimizer.layout_version != expected.optimizer_layout_version
        {
            return mismatch(
                "optimizer layout version",
                self.optimizer.layout_version.to_string(),
                expected.optimizer_layout_version.to_string(),
            );
        }
        if self.optimizer.graph_ready
            && self.optimizer.hyperparameters != expected.optimizer_hyperparameters
        {
            // Name the row that differs so the error says which coefficient.
            let (saved, found) = first_difference(
                &self.optimizer.hyperparameters,
                &expected.optimizer_hyperparameters,
            );
            return mismatch("optimizer hyperparameters", saved, found);
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

/// The first `name=value` row two vectors disagree on, as `(saved, found)`.
/// A row present on one side only is reported against `absent`.
fn first_difference(saved: &[String], found: &[String]) -> (String, String) {
    let missing = || "absent".to_string();
    for index in 0..saved.len().max(found.len()) {
        let left = saved.get(index);
        let right = found.get(index);
        if left != right {
            return (
                left.cloned().unwrap_or_else(missing),
                right.cloned().unwrap_or_else(missing),
            );
        }
    }
    (saved.join(" "), found.join(" "))
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
    /// That optimizer's declared layout version, compared beside the name.
    pub optimizer_layout_version: u32,
    /// This run's hyperparameter vector, in the same rendering the checkpoint
    /// stores. Empty on both sides compares equal, which keeps records written
    /// before the field existed resumable.
    pub optimizer_hyperparameters: Vec<String>,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    /// `lora`, `full`, `partial` or `hybrid`.
    pub trainable_policy: String,
    /// Fingerprint of the resolved trainable set's canonical manifest, empty
    /// for a run that trains no base tensor. Empty on both sides is the LoRA
    /// case and compares equal, which is what keeps this field additive for
    /// every run that existed before it.
    pub trainable_signature: String,
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

/// Disk size of one published checkpoint, member by member.
///
/// A total would hide which member grows: the adapter and the bundle follow
/// the trainable set, the payload follows the optimizer, and the sibling export
/// is a copy of the adapter that lands outside the directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointFootprint {
    /// `adapter.gguf`, when the run trains an adapter.
    pub adapter_bytes: u64,
    /// `trainable.gguf`, when the run trains base tensors.
    pub trainable_bytes: u64,
    /// `optimizer-state.bin`: the concatenated slot payloads.
    pub optimizer_state_bytes: u64,
}

/// Headroom for the files a footprint does not size: the msgpack documents and
/// the artifacts directory. A bound, not a measurement: those files are a few
/// kilobytes next to gigabytes of weights.
pub const CHECKPOINT_METADATA_BYTES: u64 = 4 * 1024 * 1024;

impl CheckpointFootprint {
    /// Total bytes one checkpoint costs.
    ///
    /// The adapter is counted twice: the directory holds `adapter.gguf` and a
    /// sibling copy is published beside it.
    pub fn bytes(&self) -> u64 {
        let export_bytes = self.adapter_bytes;
        [
            self.adapter_bytes,
            self.trainable_bytes,
            self.optimizer_state_bytes,
            export_bytes,
            CHECKPOINT_METADATA_BYTES,
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
    }
}

/// What a run needs free on the filesystem holding its checkpoint directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskBudget {
    pub footprint: CheckpointFootprint,
    /// Complete checkpoints the directory holds at once. Nothing deletes one,
    /// so for a whole run this is every checkpoint its schedule writes; for a
    /// single save it is 0, since what already landed is already out of the
    /// free figure.
    pub retained: u64,
}

impl DiskBudget {
    /// The retained checkpoints plus the one being written. The extra one is
    /// not slack: publication stages the whole directory beside its final name
    /// and renames it, so both copies exist at once.
    pub fn required_bytes(&self) -> u64 {
        self.footprint
            .bytes()
            .saturating_mul(self.retained.saturating_add(1))
    }

    /// How many complete checkpoints `free` bytes hold. Zero means even one
    /// save cannot land.
    pub fn affordable(&self, free: u64) -> u64 {
        // Never zero: every checkpoint carries its metadata.
        free / self.footprint.bytes().max(1)
    }

    /// Refuses when the filesystem holding `directory` cannot take this
    /// budget. `what` names the moment: before a run or before a save.
    pub fn check(&self, directory: &Path, what: &str) -> Result<()> {
        let free = free_space(directory)?;
        let required = self.required_bytes();
        if free >= required {
            return Ok(());
        }
        Err(Error::checkpoint(format!(
            "{what}: {} has {free} bytes free and needs {required} - {} checkpoint(s) at \
             {} bytes each (adapter {}, trainable {}, optimizer state {}, sibling export {}, \
             metadata {CHECKPOINT_METADATA_BYTES}), of which {} fit. A wider checkpoint \
             cadence lowers how often they are written, not how much the retained ones occupy",
            directory.display(),
            self.retained.saturating_add(1),
            self.footprint.bytes(),
            self.footprint.adapter_bytes,
            self.footprint.trainable_bytes,
            self.footprint.optimizer_state_bytes,
            self.footprint.adapter_bytes,
            self.affordable(free),
        )))
    }
}

/// Bytes free on the filesystem holding `path`, for an unprivileged writer.
///
/// Resolved against the nearest existing ancestor, so a not-yet-existing
/// subdirectory is measured against its filesystem.
pub fn free_space(path: &Path) -> Result<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;

    let mut probe = path;
    while !probe.exists() {
        probe = probe.parent().ok_or_else(|| {
            Error::checkpoint(format!(
                "no existing directory above {} to measure free space on",
                path.display()
            ))
        })?;
    }
    let target = CString::new(probe.as_os_str().as_bytes()).map_err(|_| {
        Error::checkpoint(format!(
            "{} cannot be passed to the filesystem: it contains a NUL byte",
            probe.display()
        ))
    })?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `target` is a NUL-terminated path that lives across the call, and
    // `stat` is a correctly aligned, writable `statvfs` the call either fills
    // or leaves untouched - and it is only read on success.
    if unsafe { libc::statvfs(target.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: `statvfs` returned zero, so it wrote the whole structure.
    let stat = unsafe { stat.assume_init() };
    // `f_bavail`, not `f_bfree`: the latter includes a reserve a training job
    // cannot write into.
    Ok(widen(stat.f_bavail).saturating_mul(widen(stat.f_frsize)))
}

/// Widens a C integer that is 32 bits on Darwin and 64 on Linux.
/// `Into<u64>` handles both without a per-platform cast.
fn widen(value: impl Into<u64>) -> u64 {
    value.into()
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
                adapter: Some(ADAPTER_FILE.into()),
                trainable: None,
                trainable_policy: "lora".into(),
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
                layout_version: 1,
                hyperparameters: sample_hyperparameters(),
                learning_rate: 2.0e-4,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
                iter: 43,
                graph_ready: true,
                slots: vec![
                    StateSlot {
                        scope: SlotScope::Parameter,
                        owner: "blk.0.attn_q.weight.lora_a".into(),
                        slot: "m".into(),
                        dtype: "F32".into(),
                        shape: [4, 8, 1, 1],
                        offset: 0,
                        n_bytes: 128,
                    },
                    StateSlot {
                        scope: SlotScope::Parameter,
                        owner: "blk.0.attn_q.weight.lora_a".into(),
                        slot: "v".into(),
                        dtype: "F32".into(),
                        shape: [4, 8, 1, 1],
                        offset: 128,
                        n_bytes: 128,
                    },
                ],
                assignment: vec![ParameterAssignment {
                    parameter: "blk.0.attn_q.weight.lora_a".into(),
                    optimizer: "adamw".into(),
                    layout_version: 1,
                }],
                state_bytes: 256,
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

    /// AdamW's declared vector, spelled out here rather than imported so a
    /// layout change fails the comparisons in this crate.
    fn sample_hyperparameters() -> Vec<String> {
        [
            "learning_rate=0.0002",
            "beta1=0.9",
            "beta2=0.999",
            "eps=1e-8",
            "weight_decay=0.01",
            "max_grad_norm=1.0",
        ]
        .iter()
        .map(|line| (*line).to_string())
        .collect()
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
            optimizer_layout_version: 1,
            optimizer_hyperparameters: sample_hyperparameters(),
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            trainable_policy: "lora".into(),
            trainable_signature: String::new(),
        }
    }

    /// Writes the payloads the sample manifest declares: a stand-in adapter and
    /// a slot payload of exactly the declared length.
    fn write_sample_payloads(marker: &[u8]) -> impl FnOnce(&CheckpointPaths) -> Result<()> + '_ {
        move |paths: &CheckpointPaths| {
            if let Some(path) = &paths.adapter {
                fs::write(path, marker)?;
            }
            if let Some(path) = &paths.trainable {
                fs::write(path, b"TRAINABLE")?;
            }
            if let Some(path) = &paths.optimizer_state {
                fs::write(path, vec![7_u8; 256])?;
            }
            Ok(())
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
            .write(&state, write_sample_payloads(b"GGUF"))
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
            .write(&state, write_sample_payloads(b"FIRST"))
            .unwrap();

        let mut second = sample();
        second.progress.best_eval = Some(0.5);
        second
            .write(&state, write_sample_payloads(b"SECOND"))
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
            .write(&state, write_sample_payloads(b"FIRST"))
            .unwrap();
        let backup = backup_dir_for(&state).unwrap();
        std::fs::rename(&state, &backup).unwrap();

        assert_eq!(Checkpoint::read(&state).unwrap(), checkpoint);
        assert_eq!(
            std::fs::read(adapter_for(&state, &checkpoint.manifest).unwrap()).unwrap(),
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
            .write(&state, write_sample_payloads(b"GGUF"))
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
            .write(&state, write_sample_payloads(b"GGUF"))
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
            .write(&state, write_sample_payloads(b"GGUF"))
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
            .write(&state, write_sample_payloads(b"GGUF"))
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
        checkpoint.optimizer.graph_ready = true;
        checkpoint.optimizer.slots.clear();
        checkpoint.optimizer.state_bytes = 0;
        checkpoint.optimizer.assignment[0].optimizer = "sgd".into();

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

    /// A cold checkpoint pins no trajectory, so the optimizer it names is not
    /// compared: nothing was allocated under it and nothing counted steps.
    #[test]
    fn a_checkpoint_taken_before_the_graph_existed_pins_no_optimizer() {
        let mut checkpoint = sample();
        checkpoint.optimizer.graph_ready = false;
        checkpoint.optimizer.slots.clear();
        checkpoint.optimizer.state_bytes = 0;
        checkpoint.optimizer.kind = "sgd".into();
        checkpoint.check_compatible(&compatibility()).unwrap();
    }

    /// The slot table addresses a byte range in a separate file, so its own
    /// consistency is checked before anything reads through it.
    #[test]
    fn a_slot_table_that_does_not_tile_its_payload_is_refused() {
        let mut gapped = sample().optimizer;
        gapped.slots[1].offset = 192;
        let error = validate_slot_table(&gapped).unwrap_err();
        assert!(
            error.to_string().contains("payload continues at 128"),
            "{error}"
        );

        let mut duplicated = sample().optimizer;
        duplicated.slots[1].slot = "m".into();
        duplicated.slots[1].offset = 128;
        let error = validate_slot_table(&duplicated).unwrap_err();
        assert!(error.to_string().contains("two 'm' slots"), "{error}");

        let mut miscounted = sample().optimizer;
        miscounted.state_bytes = 512;
        let error = validate_slot_table(&miscounted).unwrap_err();
        assert!(error.to_string().contains("covers 256 bytes"), "{error}");
    }

    /// The temporary AdamW accessor exists for callers that still think in
    /// momenta, and refuses every other optimizer rather than reading slots 0
    /// and 1 of an unknown layout as a pair.
    #[test]
    fn the_adamw_accessor_refuses_another_optimizers_slots() {
        let checkpoint = sample();
        let (m, v) = checkpoint
            .optimizer
            .adamw_moments("blk.0.attn_q.weight.lora_a")
            .unwrap();
        assert_eq!((m.slot.as_str(), v.slot.as_str()), ("m", "v"));
        assert_eq!((m.offset, v.offset), (0, 128));

        let mut renamed = checkpoint.optimizer.clone();
        renamed.kind = "sgd".into();
        let error = renamed
            .adamw_moments("blk.0.attn_q.weight.lora_a")
            .unwrap_err();
        assert!(
            error.to_string().contains("keeps no AdamW momenta"),
            "{error}"
        );
    }

    /// A slot payload is read back in bounded chunks, and a range outside the
    /// file is an error rather than a short read.
    #[test]
    fn a_slot_payload_streams_back_through_a_bounded_buffer() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let checkpoint = sample();
        checkpoint
            .write(&state, write_sample_payloads(b"GGUF"))
            .unwrap();

        let mut reader = OptimizerStateReader::open(&state, &checkpoint.optimizer).unwrap();
        let mut staging = vec![0_u8; 48];
        let mut seen = Vec::new();
        reader
            .stream(
                &checkpoint.optimizer.slots[1],
                &mut staging,
                |offset, chunk| {
                    seen.push((offset, chunk.len()));
                    assert!(chunk.iter().all(|byte| *byte == 7));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(seen, [(0, 48), (48, 48), (96, 32)]);

        let past_the_end = StateSlot {
            offset: 200,
            n_bytes: 128,
            ..checkpoint.optimizer.slots[0].clone()
        };
        let error = reader
            .stream(&past_the_end, &mut staging, |_, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("ends at 328"), "{error}");
    }

    /// The shared scope at a length above zero. Every built-in optimizer
    /// declares none, so this synthetic row is the only way to exercise it:
    #[test]
    fn a_shared_slot_round_trips_beside_the_parameter_ones() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let mut checkpoint = sample();
        // 256 F32 codebook entries, owned by the optimizer, not by any
        // parameter.
        let codebook = StateSlot {
            scope: SlotScope::Shared,
            owner: "gefen".into(),
            slot: "codebook".into(),
            dtype: "F32".into(),
            shape: [256, 1, 1, 1],
            offset: 256,
            n_bytes: 1024,
        };
        checkpoint.optimizer.slots.push(codebook.clone());
        checkpoint.optimizer.state_bytes = 256 + 1024;
        checkpoint
            .write(&state, |paths| {
                if let Some(path) = &paths.adapter {
                    fs::write(path, b"GGUF")?;
                }
                let path = paths
                    .optimizer_state
                    .as_ref()
                    .expect("the record declares a payload");
                let mut payload = vec![7_u8; 256];
                payload.extend(std::iter::repeat_n(3_u8, 1024));
                fs::write(path, payload)?;
                Ok(())
            })
            .unwrap();

        let read = Checkpoint::read(&state).unwrap();
        assert_eq!(read.optimizer.slots, checkpoint.optimizer.slots);
        // Shared and parameter rows tile one payload, so a shared row is
        // streamed like a parameter one.
        let mut reader = OptimizerStateReader::open(&state, &read.optimizer).unwrap();
        let mut staging = vec![0_u8; 512];
        let mut seen = Vec::new();
        reader
            .stream(&codebook, &mut staging, |offset, chunk| {
                seen.push((offset, chunk.len()));
                assert!(chunk.iter().all(|byte| *byte == 3));
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, [(0, 512), (512, 512)]);
        let _ = fs::remove_dir_all(&root);
    }

    /// The scope is part of a slot's identity, so the table rules hold across
    /// scopes, not within each.
    #[test]
    fn the_scope_is_part_of_a_slots_identity_in_the_table() {
        let mut duplicated = sample().optimizer;
        // Same owner and slot name in both scopes: two rows, accepted.
        duplicated.slots.push(StateSlot {
            scope: SlotScope::Shared,
            owner: "blk.0.attn_q.weight.lora_a".into(),
            slot: "m".into(),
            dtype: "F32".into(),
            shape: [4, 8, 1, 1],
            offset: 256,
            n_bytes: 128,
        });
        duplicated.state_bytes = 384;
        validate_slot_table(&duplicated).unwrap();
        // The same row twice within the shared scope is not.
        let mut twice = duplicated.clone();
        twice.slots.push(StateSlot {
            offset: 384,
            ..duplicated.slots[2].clone()
        });
        twice.state_bytes = 512;
        let error = validate_slot_table(&twice).unwrap_err().to_string();
        assert!(error.contains("two 'm' slots for shared"), "{error}");
        // A shared row that does not continue the payload is a gap, as with a
        // parameter row.
        let mut gapped = duplicated.clone();
        gapped.slots[2].offset = 300;
        let error = validate_slot_table(&gapped).unwrap_err().to_string();
        assert!(error.contains("continues at 256"), "{error}");
    }

    /// A run that trains base weights leaves a bundle and no adapter, and the
    /// sibling GGUF export - which every helper reads as an adapter - is not
    /// published for it.
    #[test]
    fn a_base_weight_checkpoint_carries_a_bundle_and_publishes_no_adapter() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let mut checkpoint = sample();
        checkpoint.manifest.adapter = None;
        checkpoint.manifest.trainable_policy = "partial".into();
        checkpoint.manifest.trainable = Some(TrainableBundle {
            file: TRAINABLE_FILE.into(),
            // Filled in by the writer: the size and identity are only known
            // once the payload exists.
            bytes: 0,
            fingerprint: String::new(),
            signature: "set-1".into(),
            tensors: vec![TrainableTensor {
                name: "blk.0.attn_norm.weight".into(),
                role: "base".into(),
                dtype: "F32".into(),
                shape: [64, 1, 1, 1],
                n_elements: 64,
                n_bytes: 256,
                aliases: Vec::new(),
            }],
        });
        checkpoint
            .write(&state, write_sample_payloads(b"GGUF"))
            .unwrap();

        assert!(state.join(TRAINABLE_FILE).is_file());
        assert!(!state.join(ADAPTER_FILE).exists());
        assert!(!root.join("step-000000000042.gguf").exists());
        assert_eq!(adapter_for(&state, &checkpoint.manifest), None);
        assert_eq!(
            trainable_for(&state, &checkpoint.manifest),
            Some(state.join(TRAINABLE_FILE))
        );

        // Publication finalizes the bundle's size and identity from the file
        // the payload writer produced.
        let published = Checkpoint::read(&state).unwrap();
        let bundle = published.manifest.trainable.as_ref().unwrap();
        assert_eq!(bundle.bytes, 9);
        assert_eq!(
            bundle.fingerprint,
            fingerprint_file(&state.join(TRAINABLE_FILE)).unwrap()
        );
        // ... and nothing else moved.
        let mut expected = checkpoint.clone();
        let declared = expected.manifest.trainable.as_mut().unwrap();
        declared.bytes = bundle.bytes;
        declared.fingerprint.clone_from(&bundle.fingerprint);
        assert_eq!(published, expected);

        // And the resolved set is compared, not just the policy name.
        let mut expected = compatibility();
        expected.trainable_policy = "partial".into();
        expected.trainable_signature = "set-1".into();
        checkpoint.check_compatible(&expected).unwrap();
        expected.trainable_signature = "set-2".into();
        let error = checkpoint.check_compatible(&expected).unwrap_err();
        assert!(error.to_string().contains("trainable set"), "{error}");
    }

    /// A declared payload the writer did not produce is refused at write time:
    /// the manifest is written from the declaration, so publishing anyway would
    /// describe a checkpoint that is not there.
    #[test]
    fn a_payload_the_writer_skipped_is_refused_before_publication() {
        let root = tempdir();
        let state = root.join("step-000000000042.state");
        let error = sample()
            .write(&state, |paths| {
                fs::write(paths.adapter.as_ref().unwrap(), b"GGUF")?;
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("optimizer state"), "{error}");
        assert!(!state.exists());

        let error = sample()
            .write(&state, |paths| {
                fs::write(paths.adapter.as_ref().unwrap(), b"GGUF")?;
                fs::write(paths.optimizer_state.as_ref().unwrap(), vec![0_u8; 8])?;
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("8 bytes"), "{error}");
        assert!(!state.exists());
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

    /// The name, the layout version and the vector pin three different things;
    /// a resume comparing only the first would restore a payload written under
    /// different arithmetic.
    #[test]
    fn a_changed_layout_or_coefficient_is_refused_by_the_row_that_moved() {
        let mut expected = compatibility();
        expected.optimizer_layout_version = 2;
        let error = sample().check_compatible(&expected).unwrap_err();
        assert!(error.to_string().contains("layout version"), "{error}");

        // A coefficient no document spells moved: the run configuration is
        // unchanged, the trajectory is not.
        let mut expected = compatibility();
        expected.optimizer_hyperparameters[1] = "beta1=0.95".into();
        let error = sample().check_compatible(&expected).unwrap_err();
        assert!(error.to_string().contains("beta1=0.9,"), "{error}");
        assert!(error.to_string().contains("beta1=0.95"), "{error}");

        // A layout that lost a knob is reported against its absence.
        let mut expected = compatibility();
        expected.optimizer_hyperparameters.pop();
        let error = sample().check_compatible(&expected).unwrap_err();
        assert!(error.to_string().contains("absent"), "{error}");
    }

    /// Like the name, a cold checkpoint commits to no layout and no
    /// coefficients: it allocated nothing and counted no step.
    #[test]
    fn a_cold_checkpoint_pins_neither_layout_nor_coefficients() {
        let mut checkpoint = sample();
        checkpoint.optimizer.graph_ready = false;
        checkpoint.optimizer.layout_version = 7;
        checkpoint.optimizer.hyperparameters = vec!["beta1=0.5".into()];
        checkpoint.check_compatible(&compatibility()).unwrap();
    }

    #[test]
    fn parameter_assignment_pins_owners_and_layouts_without_requiring_order() {
        let optimizer = sample().optimizer;
        optimizer.check_assignment(&optimizer.assignment).unwrap();
        for field in ["optimizer", "layout", "parameter"] {
            let mut changed = optimizer.assignment.clone();
            match field {
                "optimizer" => changed[0].optimizer = "sgd".into(),
                "layout" => changed[0].layout_version += 1,
                _ => changed[0].parameter = "another.weight".into(),
            }
            assert!(optimizer.check_assignment(&changed).is_err());
        }
        assert!(optimizer.check_assignment(&[]).is_err());
        let mut two = optimizer.clone();
        let mut second = two.assignment[0].clone();
        second.parameter = "another.weight".into();
        two.assignment.push(second);
        let mut reversed = two.assignment.clone();
        reversed.reverse();
        two.check_assignment(&reversed).unwrap();
        reversed[0] = reversed[1].clone();
        assert!(two.check_assignment(&reversed).is_err());
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
            Some(state.join(ADAPTER_FILE))
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

    #[test]
    fn a_footprint_pays_for_the_adapter_twice_and_the_bundle_once() {
        // The adapter is copied out beside the directory; a bundle has no
        // sibling.
        let adapter = CheckpointFootprint {
            adapter_bytes: 1_000,
            trainable_bytes: 0,
            optimizer_state_bytes: 4_000,
        };
        assert_eq!(adapter.bytes(), 2_000 + 4_000 + CHECKPOINT_METADATA_BYTES);
        let base = CheckpointFootprint {
            adapter_bytes: 0,
            trainable_bytes: 1_000,
            optimizer_state_bytes: 4_000,
        };
        assert_eq!(base.bytes(), 1_000 + 4_000 + CHECKPOINT_METADATA_BYTES);
    }

    #[test]
    fn a_budget_charges_the_retained_checkpoints_and_the_one_being_written() {
        let budget = DiskBudget {
            footprint: CheckpointFootprint {
                adapter_bytes: 0,
                trainable_bytes: 6 * 1024 * 1024,
                optimizer_state_bytes: 0,
            },
            retained: 3,
        };
        let per_checkpoint = 6 * 1024 * 1024 + CHECKPOINT_METADATA_BYTES;
        assert_eq!(budget.required_bytes(), per_checkpoint * 4);
        assert_eq!(budget.affordable(per_checkpoint * 4), 4);
        // One byte short of a fourth checkpoint fits three.
        assert_eq!(budget.affordable(per_checkpoint * 4 - 1), 3);
    }

    #[test]
    fn a_budget_larger_than_the_filesystem_is_refused_and_names_what_fits() {
        let dir = tempdir();
        let free = free_space(&dir).unwrap();
        assert!(free > 0, "a writable scratch directory has free space");
        let budget = DiskBudget {
            footprint: CheckpointFootprint {
                adapter_bytes: free,
                trainable_bytes: 0,
                optimizer_state_bytes: 0,
            },
            retained: 0,
        };
        let error = budget.check(&dir, "cannot write this checkpoint").unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, Error::Checkpoint(_)), "{message}");
        assert!(message.contains("cannot write this checkpoint"), "{message}");
        assert!(message.contains("of which 0 fit"), "{message}");
        // Widening the cadence is not a fix; the message says so.
        assert!(message.contains("cadence"), "{message}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_footprint_is_affordable_and_a_free_figure_comes_from_an_ancestor() {
        let dir = tempdir();
        // A not-yet-existing subdirectory is measured against its filesystem.
        // The figure is not compared: it is a live machine's, and it moves
        // between two calls.
        let unborn = dir.join("checkpoints").join("deeper");
        assert!(free_space(&unborn).unwrap() > 0);
        // Even the empty footprint carries the metadata.
        let empty = DiskBudget::default();
        assert_eq!(empty.required_bytes(), CHECKPOINT_METADATA_BYTES);
        assert_eq!(empty.affordable(0), 0);
        empty.check(&unborn, "nothing to write").unwrap();
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
