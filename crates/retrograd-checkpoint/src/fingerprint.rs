//! Content identities: a cheap one for datasets, SHA-256 for model files.

use std::collections::HashMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use retrograd_core::Result;
use sha2::{Digest as _, Sha256};

/// Stable 128-bit content fingerprint, rendered as hex: two 64-bit FNV-1a-style
/// lanes, the second run over the bitwise complement of each byte so the two
/// lanes do not trivially agree. Used for the dataset identity in a manifest,
/// where a cryptographic digest would be overkill and an extra dependency.
pub fn fingerprint(bytes: &[u8]) -> String {
    let mut fingerprinter = Fingerprinter::new();
    fingerprinter.update(bytes);
    fingerprinter.finish()
}

/// [`fingerprint`] fed in pieces: the result depends only on the concatenated
/// bytes, never on where they were split. Lets a caller hash a prepared dataset
/// in place instead of first serializing a copy the size of the dataset.
#[derive(Clone, Debug)]
pub struct Fingerprinter {
    low: u64,
    high: u64,
}

impl Default for Fingerprinter {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprinter {
    pub fn new() -> Self {
        Self {
            low: 0xcbf2_9ce4_8422_2325,
            high: 0x9dcf_1a8b_4c37_5f11,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.low = (self.low ^ *byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
            self.high = (self.high ^ !*byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn finish(&self) -> String {
        format!("{:016x}{:016x}", self.low, self.high)
    }
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
pub fn fingerprint_file_cached(path: &Path) -> Result<String> {
    type Cache = Mutex<HashMap<PathBuf, (u64, Option<SystemTime>, String)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();

    let metadata = fs::metadata(path)?;
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
pub fn fingerprint_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
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
