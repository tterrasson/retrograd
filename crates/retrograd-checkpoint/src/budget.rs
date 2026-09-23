//! The disk space a checkpoint schedule needs, checked before it is spent.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt as _;
use std::path::Path;

use retrograd_core::{Error, Result};

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
