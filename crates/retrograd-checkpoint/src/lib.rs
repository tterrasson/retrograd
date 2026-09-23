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
    /// Fingerprint of the fixed-reference model, empty when the run declares
    /// no anchor. Old checkpoints that lack the field read as empty, which is
    /// their actual state: their anchor was their own frozen weights.
    #[serde(default)]
    pub reference_fingerprint: String,
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

    /// The `(m, v)` pair of one parameter of an AdamW checkpoint.
    ///
    /// Refuses any other optimizer rather than handing back slots 0 and 1 of
    /// whatever is there.
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
        if self.manifest.reference_fingerprint != expected.reference_fingerprint {
            return mismatch(
                "fixed reference",
                describe_anchor(&self.manifest.reference_fingerprint),
                describe_anchor(&expected.reference_fingerprint),
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

/// Renders a fingerprint for an error message; empty means no anchor.
fn describe_anchor(fingerprint: &str) -> String {
    if fingerprint.is_empty() {
        "none".to_string()
    } else {
        fingerprint.to_string()
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
    /// Fingerprint of this run's anchor, empty when none is declared. Changing
    /// the anchor mid-run is a different objective, so it is compared on
    /// resume like the base model's.
    pub reference_fingerprint: String,
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

mod budget;
mod fingerprint;

pub use budget::{CHECKPOINT_METADATA_BYTES, CheckpointFootprint, DiskBudget, free_space};
pub use fingerprint::{Fingerprinter, fingerprint, fingerprint_file, fingerprint_file_cached};

#[cfg(test)]
mod tests;
