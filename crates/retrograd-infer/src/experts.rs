//! Persistent routing counts, independent of the decode backend. Disk reads,
//! hashing and merging run on the history worker; the serving thread submits
//! only the counters accumulated since its last submission.

use std::fs::{self, File, Metadata};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &str = "retrograd-expert-history";
const VERSION: u32 = 1;
const HASH_DOMAIN: &[u8] = b"retrograd-gguf-shards-v1\0";

/// Readback can synchronize the device, so history uses sparse samples by
/// default. A caller can request an exact count with an interval of one.
pub const DEFAULT_ROUTE_SAMPLE_EVERY: NonZeroU32 = NonZeroU32::new(64).unwrap();

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertHistoryOptions {
    pub dir: PathBuf,
    pub sample_every: NonZeroU32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bank {
    pub layer: u32,
    pub role: String,
    pub n_expert: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ExpertHistoryError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("invalid expert history document: {0}")]
    Format(#[from] serde_json::Error),
    #[error("unsupported expert history header: magic {magic:?}, version {version}")]
    Header { magic: String, version: u32 },
    #[error("expert history content identity differs: file {found}, model {expected}")]
    Identity { found: String, expected: String },
    #[error("expert history bank layout differs: file {found:?}, model {expected:?}")]
    Banks {
        found: Vec<Bank>,
        expected: Vec<Bank>,
    },
    #[error("expert history counter dimensions do not match the bank layout")]
    Shape,
    #[error("model shard changed while preparing expert history: {}", .0.display())]
    Changed(PathBuf),
    #[error("expert history requires at least one model shard")]
    NoShards,
    #[error("expert history directory cannot be the model directory: {}", .0.display())]
    ModelDirectory(PathBuf),
    #[error("expert history worker has stopped")]
    WorkerStopped,
}

impl From<ExpertHistoryError> for retrograd_core::Error {
    fn from(error: ExpertHistoryError) -> Self {
        match error {
            ExpertHistoryError::Io(error) => Self::Io(error),
            error @ ExpertHistoryError::ModelDirectory(_) => Self::config(error.to_string()),
            error => Self::runtime(error.to_string()),
        }
    }
}

type Result<T> = std::result::Result<T, ExpertHistoryError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ShardKey {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
    #[cfg(unix)]
    file_id: (u64, u64),
    #[cfg(unix)]
    changed: (i64, i64),
}

impl ShardKey {
    fn read(path: PathBuf, meta: &Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;
        Ok(Self {
            path,
            size: meta.len(),
            modified: meta.modified()?,
            #[cfg(unix)]
            file_id: (meta.dev(), meta.ino()),
            #[cfg(unix)]
            changed: (meta.ctime(), meta.ctime_nsec()),
        })
    }
}

/// The ordered files passed to the model loader, captured before it opens
/// them. Recheck both the paths and opened files before trusting a digest.
pub struct ModelShards(Vec<(PathBuf, ShardKey)>);

impl ModelShards {
    pub fn capture(paths: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let entries = paths
            .into_iter()
            .map(|path| Ok((path.clone(), Self::key(&path)?)))
            .collect::<io::Result<Vec<_>>>()?;
        if entries.is_empty() {
            return Err(ExpertHistoryError::NoShards);
        }
        Ok(Self(entries))
    }

    fn key(path: &Path) -> io::Result<ShardKey> {
        let canonical = path.canonicalize()?;
        ShardKey::read(canonical, &fs::metadata(path)?)
    }

    /// Check that the paths still identify the same files.
    pub fn verify(&self) -> Result<()> {
        for (path, key) in &self.0 {
            if Self::key(path)? != *key {
                return Err(ExpertHistoryError::Changed(path.clone()));
            }
        }
        Ok(())
    }

    fn identity(&self, dir: &Path) -> Result<String> {
        self.verify()?;
        let keys: Vec<_> = self.0.iter().map(|(_, key)| key.clone()).collect();
        let cache_key = hex(&Sha256::digest(keys[0].path.as_os_str().as_encoded_bytes()));
        let cache = dir.join(format!("model-{cache_key}.json"));
        // An inode and device identify the actual file, including replacement
        // with a same-length file. Platforms without them always rehash.
        if cfg!(unix)
            && let Ok(record) = read_json::<DigestRecord>(&cache)
            && record.version == VERSION
            && record.shards == keys
            && record.identity.len() == 64
            && record.identity.bytes().all(|c| c.is_ascii_hexdigit())
        {
            self.verify()?;
            return Ok(record.identity);
        }

        let mut hash = Sha256::new();
        hash.update(HASH_DOMAIN);
        let mut buffer = vec![0_u8; 1 << 20];
        for (path, key) in &self.0 {
            let mut file = File::open(path)?;
            if ShardKey::read(key.path.clone(), &file.metadata()?)? != *key {
                return Err(ExpertHistoryError::Changed(path.clone()));
            }
            // Frame the shards so a different partition of the same bytes
            // cannot produce the same digest.
            hash.update(key.size.to_le_bytes());
            let mut read = 0_u64;
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                // Each read is bounded by the 1 MiB buffer.
                read += n as u64;
                hash.update(&buffer[..n]);
            }
            if read != key.size || ShardKey::read(key.path.clone(), &file.metadata()?)? != *key {
                return Err(ExpertHistoryError::Changed(path.clone()));
            }
        }
        self.verify()?;
        let identity = hex(&hash.finalize());
        if cfg!(unix)
            && let Err(error) = atomic_json(
                &cache,
                &DigestRecord {
                    version: VERSION,
                    shards: keys,
                    identity: identity.clone(),
                },
            )
        {
            tracing::warn!(path = %cache.display(), %error, "could not cache model identity");
        }
        Ok(identity)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Serialize, Deserialize)]
struct DigestRecord {
    version: u32,
    shards: Vec<ShardKey>,
    identity: String,
}

#[derive(Serialize, Deserialize)]
struct History {
    magic: String,
    version: u32,
    identity: String,
    banks: Vec<Bank>,
    counts: Vec<Vec<u32>>,
}

impl History {
    fn empty(identity: String, banks: Vec<Bank>) -> Self {
        let counts = banks.iter().map(|bank| vec![0; bank.n_expert]).collect();
        Self {
            magic: MAGIC.into(),
            version: VERSION,
            identity,
            banks,
            counts,
        }
    }

    fn load(path: &Path, identity: &str, banks: &[Bank]) -> Result<Self> {
        let file: Self = read_json(path)?;
        if file.magic != MAGIC || file.version != VERSION {
            return Err(ExpertHistoryError::Header {
                magic: file.magic,
                version: file.version,
            });
        }
        if file.identity != identity {
            return Err(ExpertHistoryError::Identity {
                found: file.identity,
                expected: identity.to_owned(),
            });
        }
        if file.banks != banks {
            return Err(ExpertHistoryError::Banks {
                found: file.banks,
                expected: banks.to_vec(),
            });
        }
        check_shape(&file.counts, banks)?;
        Ok(file)
    }
}

fn check_shape(counts: &[Vec<u32>], banks: &[Bank]) -> Result<()> {
    if counts.len() != banks.len()
        || counts
            .iter()
            .zip(banks)
            .any(|(row, b)| row.len() != b.n_expert)
    {
        return Err(ExpertHistoryError::Shape);
    }
    Ok(())
}

fn merge(into: &mut [Vec<u32>], delta: &[Vec<u32>]) {
    for (row, added) in into.iter_mut().zip(delta) {
        for (count, add) in row.iter_mut().zip(added) {
            *count = count.saturating_add(*add);
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file)).map_err(|error| {
        if let Some(kind) = error.io_error_kind() {
            ExpertHistoryError::Io(io::Error::new(kind, error))
        } else {
            error.into()
        }
    })
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut temp =
        tempfile::NamedTempFile::new_in(path.parent().expect("state file has a parent"))?;
    let mut writer = BufWriter::new(temp.as_file_mut());
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    drop(writer);
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn flush(path: &Path, identity: &str, banks: &[Bank], delta: &[Vec<u32>]) -> Result<()> {
    // Lock a separate, stable file: replacing the history must not replace
    // the inode on which other processes are waiting.
    let lock = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path.with_extension("lock"))?;
    lock.lock()?;
    let cold = || History::empty(identity.to_owned(), banks.to_vec());
    let mut history = match History::load(path, identity, banks) {
        Ok(history) => history,
        Err(ExpertHistoryError::Io(error)) if error.kind() == io::ErrorKind::NotFound => cold(),
        Err(error @ ExpertHistoryError::Io(_)) => return Err(error),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "starting expert history cold");
            cold()
        }
    };
    merge(&mut history.counts, delta);
    atomic_json(path, &history)
}

type Pending = Arc<Mutex<Option<Vec<Vec<u32>>>>>;

/// A bounded handoff to the worker. Repeated submissions during a large
/// model hash are merged in memory, without queuing a copy per submission.
pub struct ExpertHistory {
    banks: Vec<Bank>,
    pending: Pending,
    wake: Option<mpsc::SyncSender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ExpertHistory {
    pub fn start(dir: PathBuf, shards: ModelShards, banks: Vec<Bank>) -> Result<Self> {
        let pending = Pending::default();
        let (wake, inbox) = mpsc::sync_channel(1);
        let queue = Arc::clone(&pending);
        let layout = banks.clone();
        let worker = thread::Builder::new()
            .name("expert-history".into())
            .spawn(move || {
                if let Err(error) = run(&dir, &shards, &layout, &queue, inbox) {
                    tracing::warn!(path = %dir.display(), %error, "expert history unavailable; using in-memory telemetry");
                }
            })?;
        Ok(Self {
            banks,
            pending,
            wake: Some(wake),
            worker: Some(worker),
        })
    }

    /// Close the producer and return the worker so a caller can wait for its
    /// last flush later, without delaying an inference model switch.
    pub fn close(mut self) -> JoinHandle<()> {
        self.wake.take();
        self.worker.take().expect("history worker exists")
    }

    pub fn submit(&self, delta: Vec<Vec<u32>>) -> Result<()> {
        check_shape(&delta, &self.banks)?;
        if !delta.iter().flatten().any(|&count| count != 0) {
            return Ok(());
        }
        let mut pending = self.pending.lock().expect("pending expert deltas");
        add_pending(&mut pending, delta);
        match self
            .wake
            .as_ref()
            .expect("history worker is live")
            .try_send(())
        {
            Ok(()) | Err(mpsc::TrySendError::Full(())) => Ok(()),
            Err(mpsc::TrySendError::Disconnected(())) => Err(ExpertHistoryError::WorkerStopped),
        }
    }
}

impl Drop for ExpertHistory {
    fn drop(&mut self) {
        self.wake.take();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            tracing::warn!("expert history worker panicked");
        }
    }
}

fn add_pending(pending: &mut Option<Vec<Vec<u32>>>, delta: Vec<Vec<u32>>) {
    if let Some(counts) = pending {
        merge(counts, &delta);
    } else {
        *pending = Some(delta);
    }
}

fn run(
    dir: &Path,
    shards: &ModelShards,
    banks: &[Bank],
    queue: &Pending,
    inbox: mpsc::Receiver<()>,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let dir = dir.canonicalize()?;
    if shards
        .0
        .iter()
        .any(|(_, key)| key.path.parent() == Some(&dir))
    {
        return Err(ExpertHistoryError::ModelDirectory(dir));
    }
    let identity = shards.identity(&dir)?;
    let path = dir.join(format!("routing-{identity}.json"));
    let mut retry = None;
    loop {
        let closed = inbox.recv().is_err();
        if let Some(delta) = queue.lock().expect("pending expert deltas").take() {
            add_pending(&mut retry, delta);
        }
        if let Some(delta) = &retry {
            match flush(&path, &identity, banks, delta) {
                Ok(()) => retry = None,
                Err(error) if closed => return Err(error),
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "could not flush expert history; retaining deltas");
                }
            }
        }
        if closed {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn banks() -> Vec<Bank> {
        vec![Bank {
            layer: 0,
            role: "up".into(),
            n_expert: 2,
        }]
    }

    #[test]
    fn history_validates_identity_layout_and_recovers_from_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing.json");
        let banks = banks();
        flush(&path, "abc", &banks, &[vec![7, u32::MAX]]).unwrap();
        flush(&path, "abc", &banks, &[vec![2, 1]]).unwrap();
        assert_eq!(
            History::load(&path, "abc", &banks).unwrap().counts,
            [vec![9, u32::MAX]]
        );
        assert!(matches!(
            History::load(&path, "def", &banks),
            Err(ExpertHistoryError::Identity { .. })
        ));
        let different = [Bank {
            role: "down".into(),
            ..banks[0].clone()
        }];
        assert!(matches!(
            History::load(&path, "abc", &different),
            Err(ExpertHistoryError::Banks { .. })
        ));

        fs::write(&path, b"{\"magic\":").unwrap();
        assert!(matches!(
            History::load(&path, "abc", &banks),
            Err(ExpertHistoryError::Format(_))
        ));
        flush(&path, "abc", &banks, &[vec![1, 0]]).unwrap();
        assert_eq!(
            History::load(&path, "abc", &banks).unwrap().counts,
            [vec![1, 0]]
        );
    }

    #[test]
    fn identity_covers_every_shard_and_rejects_replaced_files() {
        let dir = tempfile::tempdir().unwrap();
        let shard1 = dir.path().join("one.gguf");
        let shard2 = dir.path().join("two.gguf");
        fs::write(&shard1, b"first").unwrap();
        fs::write(&shard2, b"second").unwrap();
        let paths = [shard1, shard2.clone()];
        let model = ModelShards::capture(paths.clone()).unwrap();
        let first = model.identity(dir.path()).unwrap();
        assert_eq!(
            first,
            ModelShards::capture(paths.clone())
                .unwrap()
                .identity(dir.path())
                .unwrap()
        );

        let replacement = dir.path().join("replacement");
        fs::write(&replacement, b"edited").unwrap();
        fs::rename(replacement, shard2).unwrap();
        assert!(matches!(
            model.identity(dir.path()),
            Err(ExpertHistoryError::Changed(_))
        ));
        let second = ModelShards::capture(paths)
            .unwrap()
            .identity(dir.path())
            .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn submit_refuses_a_mismatched_shape_and_skips_empty_deltas() {
        let model = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let shard = model.path().join("one.gguf");
        fs::write(&shard, b"model weights").unwrap();
        let shards = ModelShards::capture([shard]).unwrap();
        let history = ExpertHistory::start(state.path().into(), shards, banks()).unwrap();
        // A delta that does not match the bank layout is refused, not merged.
        assert!(matches!(
            history.submit(vec![vec![1, 2, 3]]),
            Err(ExpertHistoryError::Shape)
        ));
        assert!(matches!(
            history.submit(vec![vec![]]),
            Err(ExpertHistoryError::Shape)
        ));
        // An all-zero delta is accepted but never reaches the disk.
        history.submit(vec![vec![0, 0]]).unwrap();
        history.close().join().unwrap();
        assert!(
            fs::read_dir(state.path())
                .unwrap()
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("routing-")),
            "a zero delta writes no history"
        );
    }

    #[test]
    fn independent_workers_merge_deltas_once_and_drain_on_shutdown() {
        let model = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let shard = model.path().join("one.gguf");
        fs::write(&shard, b"model weights").unwrap();
        let make_shards = || ModelShards::capture([shard.clone()]).unwrap();
        let banks = banks();
        let start =
            || ExpertHistory::start(state.path().into(), make_shards(), banks.clone()).unwrap();
        let a = start();
        let b = start();
        for _ in 0..20 {
            a.submit(vec![vec![1, 0]]).unwrap();
            b.submit(vec![vec![0, 2]]).unwrap();
        }
        drop(a);
        drop(b);
        let a = start();
        a.submit(vec![vec![1, 3]]).unwrap();
        drop(a);

        let identity = make_shards().identity(state.path()).unwrap();
        let path = state.path().join(format!("routing-{identity}.json"));
        assert_eq!(
            History::load(&path, &identity, &banks).unwrap().counts,
            [vec![21, 43]]
        );
    }
}
