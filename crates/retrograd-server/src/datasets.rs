//! Datasets as a content-addressed resource.
//!
//! A dataset is uploaded once, streamed to disk while its `sha256` is computed,
//! and stored under `state_dir/datasets/<id>/`, where `id = "ds_" + sha256`.
//! Two uploads of the same bytes land on the same id and write nothing twice,
//! the idempotence comes from the address, not from a header.
//!
//! No HTTP here: this module is the storage layer only (layout, hashing,
//! ingestion, the in-memory index rebuilt from `meta.json` at startup). The
//! routes that call it live in `api::datasets`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use futures_util::StreamExt;
use retrograd_core::ModelInfo;
use retrograd_dataset::{DataFormat, Validation};
use retrograd_plan::DatasetStats;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::lock::Recover as _;

/// A file this large or larger is refused before it is ever written whole.
/// The operator's own `max_dataset_bytes` is the real limit in production;
/// this is only the fallback when none is configured.
const DEFAULT_MAX_DATASET_BYTES: u64 = 1024 * 1024 * 1024;

/// Length statistics as they are written to `meta.json`. A flattened,
/// serializable copy of [`DatasetStats`] - the plan type keeps every length to
/// answer an exact truncation query; the stored copy only needs the five
/// numbers a client reads off a dataset's card.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct StoredStats {
    pub measured: bool,
    pub total: u64,
    pub p50: u32,
    pub p90: u32,
    pub p99: u32,
    pub max: u32,
}

impl From<&DatasetStats> for StoredStats {
    fn from(stats: &DatasetStats) -> Self {
        Self {
            measured: stats.measured,
            total: stats.total_tokens,
            p50: stats.percentile(0.5),
            p90: stats.percentile(0.9),
            p99: stats.percentile(0.99),
            max: stats.max_length(),
        }
    }
}

/// Real, tokenized lengths against one model, cached under the model's own key.
/// Empty until `POST /v1/datasets/{id}/tokenize` fills it in.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TokenizedStats {
    pub stats: StoredStats,
    /// Truncation fraction at a handful of common contexts, keyed by the
    /// context as a decimal string (`"512"`, `"1024"`, …) - the answer to
    /// "what `n_ctx` do I need?" without a plan.
    #[serde(default)]
    pub truncation: BTreeMap<String, f64>,
    /// Every per-example length, ascending, so a resolution can rebuild the
    /// exact `DatasetStats` this measurement produced.
    ///
    /// Redundant with `stats` and `truncation`, deliberately: those two are
    /// what a *client* reads and are computed once, while the resolver reports
    /// the truncation rate at whatever `n_ctx` it lands on - a number invariant
    /// 4 requires to be exact, and one no set of summaries can recover. The
    /// cost is four bytes an example on disk, against re-running a tokenizer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lengths: Vec<u32>,
}

impl TokenizedStats {
    /// The measurement, back in the shape the resolver consumes.
    pub fn dataset_stats(&self) -> Option<DatasetStats> {
        (!self.lengths.is_empty()).then(|| DatasetStats::measured(self.lengths.clone()))
    }
}

/// `state_dir/datasets/<id>/meta.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetMeta {
    pub id: String,
    pub sha256: String,
    /// `"chat-jsonl"` | `"text"`.
    pub format: String,
    /// The data file's name inside the dataset's own directory
    /// (`"data.jsonl"` | `"data.txt"`), so a future layout (a split's derived
    /// files, say) is not forced to reuse this one's name.
    pub data_file: String,
    pub bytes: u64,
    pub examples: u64,
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub stats: StoredStats,
    /// Keyed by the model's own hash. A `BTreeMap` so the file's bytes are
    /// deterministic.
    #[serde(default)]
    pub tokenized: BTreeMap<String, TokenizedStats>,
}

impl DatasetMeta {
    pub fn data_format(&self) -> Option<DataFormat> {
        match self.format.as_str() {
            "chat-jsonl" => Some(DataFormat::ChatJsonl),
            "text" => Some(DataFormat::Text),
            _ => None,
        }
    }

    /// The measured lengths for `model`, if this dataset has already been
    /// tokenized against it.
    pub fn tokenized_for(&self, key: &str) -> Option<&TokenizedStats> {
        self.tokenized.get(key)
    }
}

/// The key a dataset's measured lengths are cached under.
///
/// The model fingerprint is the full GGUF SHA-256, cached by [`AppState`](crate::AppState) while
/// size and mtime agree. It therefore covers the vocabulary, merges, special
/// tokens and chat template instead of treating architecture and file size as
/// tokenizer identity.
pub fn tokenizer_key(model: &ModelInfo, model_fingerprint: &str) -> String {
    let architecture: String = model
        .architecture
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("{architecture}/v{}/m{model_fingerprint}", model.n_vocab)
}

/// The contexts a tokenization reports a truncation rate at - the answer
/// to "what `n_ctx` do I need?" before a plan exists.
pub const REPORTED_CONTEXTS: [u32; 4] = [512, 1024, 2048, 4096];

/// What an ingestion produced.
#[derive(Debug)]
pub enum Ingested {
    /// New content: written, hashed, measured, indexed.
    Created(DatasetMeta),
    /// The same bytes were already stored under this id - idempotent, and
    /// nothing was written twice.
    AlreadyExists(DatasetMeta),
}

impl Ingested {
    pub fn meta(&self) -> &DatasetMeta {
        match self {
            Self::Created(meta) | Self::AlreadyExists(meta) => meta,
        }
    }
}

/// Why a chunk of the uploaded body never arrived.
///
/// Typed rather than a string because the two cases answer different statuses,
/// and the distinction is made where it is *known* - in the handler, off the
/// transport's own error type - instead of being recovered downstream by
/// matching on prose the transport is free to reword.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BodyError {
    /// The HTTP body-limit layer cut the stream: the upload is over
    /// `max_dataset_bytes`.
    #[error("the upload exceeds the configured body limit")]
    TooLarge,
    /// Anything else - a client that hung up, a broken connection.
    #[error("the upload failed: {0}")]
    Failed(String),
}

/// Everything [`DatasetStore::ingest`] needs besides the bytes themselves.
#[derive(Clone, Debug, Default)]
pub struct IngestOptions {
    /// `None` asks [`DataFormat::infer`] to determine it from the content.
    pub format: Option<DataFormat>,
    pub name: Option<String>,
    /// The size the client announced (`Content-Length`), when it did. Lets both
    /// limits be enforced before a single byte is written; an upload that
    /// declares nothing is still checked as it streams.
    pub declared_bytes: Option<u64>,
    pub max_dataset_bytes: Option<u64>,
    pub max_datasets_bytes: Option<u64>,
    /// Maximum time between two body chunks. `None` is useful to direct callers
    /// with no transport; the HTTP route always supplies the operator limit.
    pub idle_timeout: Option<Duration>,
}

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("could not write the dataset: {0}")]
    Io(#[from] std::io::Error),
    /// The streamed body itself failed (a client that hung up mid-upload).
    #[error("the upload failed: {0}")]
    Body(String),
    #[error("the dataset is larger than {max_bytes} bytes")]
    TooLarge { max_bytes: u64 },
    #[error(
        "storing {requested} more bytes would exceed the {quota} byte dataset \
         quota ({used} already used)"
    )]
    QuotaExceeded {
        requested: u64,
        used: u64,
        quota: u64,
    },
    #[error("the upload made no progress for {seconds} seconds")]
    Inactive { seconds: u64 },
    #[error("{0}")]
    UnsupportedFormat(String),
    /// The format could not be determined, or the file could not be read at
    /// all - an empty upload, for instance.
    #[error("{0}")]
    Unreadable(#[from] retrograd_core::Error),
    /// Chat-JSONL failed its per-record validation. Every problem found, not
    /// just the first - narrowing the list to a display
    /// window is the caller's job, since that is a response-shaping decision,
    /// not a storage one.
    #[error("the dataset failed validation: {} problem(s) found",.0.total)]
    Invalid(Validation),
}

fn data_file_name(format: DataFormat) -> &'static str {
    match format {
        DataFormat::ChatJsonl => "data.jsonl",
        DataFormat::Text => "data.txt",
    }
}

fn format_name(format: DataFormat) -> &'static str {
    match format {
        DataFormat::ChatJsonl => "chat-jsonl",
        DataFormat::Text => "text",
    }
}

/// Parses the `format` field/query param a client may send. `None` or
/// `"auto"` defers to [`DataFormat::infer`], the same vocabulary
/// `resolve::plan_recipe` accepts for a recipe's `data.format`.
pub fn parse_format_hint(hint: Option<&str>) -> Result<Option<DataFormat>, IngestError> {
    match hint {
        None | Some("auto") | Some("") => Ok(None),
        Some("text") | Some("txt") => Ok(Some(DataFormat::Text)),
        Some("jsonl") | Some("chat") | Some("chat-jsonl") => Ok(Some(DataFormat::ChatJsonl)),
        Some(other) => Err(IngestError::UnsupportedFormat(format!(
            "unknown dataset format '{other}'; use auto, text or jsonl"
        ))),
    }
}

/// The datasets this server has stored, live in memory and backed by
/// `state_dir/datasets/`.
pub struct DatasetStore {
    root: PathBuf,
    index: RwLock<BTreeMap<String, DatasetMeta>>,
    /// Serializes the short commit phase only: hashing and validation remain
    /// concurrent, while deduplication, quota accounting and installation are
    /// one transaction with respect to other mutations.
    mutation: Mutex<()>,
}

impl DatasetStore {
    /// Opens the store, rebuilding the index from every `<id>/meta.json` on
    /// disk, which is the source of truth, and sweeping `tmp/` of
    /// whatever an interrupted upload left behind.
    ///
    /// A `meta.json` that fails to parse isolates the one dataset it belongs
    /// to - logged and skipped - rather than stopping the server from starting:
    /// a single corrupted card is not a reason to make every other dataset
    /// unreachable.
    pub fn open(state_dir: impl AsRef<Path>) -> Self {
        let root = state_dir.as_ref().join("datasets");
        let _ = std::fs::create_dir_all(root.join("tmp"));
        let mut index = BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || path.file_name().is_some_and(|name| name == "tmp") {
                    continue;
                }
                let meta_path = path.join("meta.json");
                let text = match std::fs::read_to_string(&meta_path) {
                    Ok(text) => text,
                    Err(_) => continue,
                };
                match serde_json::from_str::<DatasetMeta>(&text) {
                    Ok(meta) => {
                        index.insert(meta.id.clone(), meta);
                    }
                    Err(error) => {
                        tracing::warn!(
                            path = %meta_path.display(),
                            %error,
                            "ignoring a dataset with an unreadable meta.json"
                        );
                    }
                }
            }
        }
        // Orphaned temporaries - an upload that never finished, from a process
        // that did not shut down cleanly - do not belong to any id and are
        // simply gone.
        if let Ok(entries) = std::fs::read_dir(root.join("tmp")) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Self {
            root,
            index: RwLock::new(index),
            mutation: Mutex::new(()),
        }
    }

    pub fn get(&self, id: &str) -> Option<DatasetMeta> {
        self.index.read().recover().get(id).cloned()
    }

    /// Newest first.
    pub fn list(&self) -> Vec<DatasetMeta> {
        let mut all: Vec<DatasetMeta> = self.index.read().recover().values().cloned().collect();
        all.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then(right.id.cmp(&left.id))
        });
        all
    }

    pub fn total_bytes(&self) -> u64 {
        self.index
            .read()
            .recover()
            .values()
            .map(|meta| meta.bytes)
            .sum()
    }

    /// The dataset's data file, when it exists.
    pub fn data_path(&self, meta: &DatasetMeta) -> PathBuf {
        self.root.join(&meta.id).join(&meta.data_file)
    }

    /// Forgets a dataset and removes its directory. `false` when there was
    /// nothing to remove.
    ///
    /// Unconditional: whether a live run still references this id is the caller's
    /// question, not this store's.
    pub fn delete(&self, id: &str) -> bool {
        let _mutation = self.mutation.lock().recover();
        let removed = self.index.write().recover().remove(id);
        if removed.is_none() {
            return false;
        }
        let dir = self.root.join(id);
        if dir.starts_with(&self.root)
            && dir != self.root
            && let Err(error) = std::fs::remove_dir_all(&dir)
        {
            tracing::warn!(path = %dir.display(), %error, "could not remove a dataset directory");
        }
        true
    }

    /// Writes `meta` back after an in-place update (e.g. a fresh `tokenized`
    /// entry) and refreshes the in-memory copy.
    pub fn save(&self, meta: DatasetMeta) -> std::io::Result<()> {
        let _mutation = self.mutation.lock().recover();
        self.save_unlocked(meta)
    }

    fn save_unlocked(&self, meta: DatasetMeta) -> std::io::Result<()> {
        let path = self.root.join(&meta.id).join("meta.json");
        let bytes = serde_json::to_vec_pretty(&meta)?;
        let temporary = path.with_extension(format!("json.{}.tmp", uuid::Uuid::new_v4()));
        {
            let mut file = std::fs::File::create(&temporary)?;
            std::io::Write::write_all(&mut file, &bytes)?;
            file.sync_all()?;
        }
        if let Err(error) = std::fs::rename(&temporary, &path) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }
        self.index.write().recover().insert(meta.id.clone(), meta);
        Ok(())
    }

    /// Caches the real, tokenized lengths of `id` against `model` and
    /// answers what was stored.
    ///
    /// `None` when the dataset went away between the measurement and this call:
    /// a `DELETE` that raced an in-flight tokenization, whose result there is
    /// nowhere left to put.
    pub fn record_tokenized(
        &self,
        id: &str,
        tokenizer_key: &str,
        lengths: Vec<u32>,
    ) -> std::io::Result<Option<TokenizedStats>> {
        let _mutation = self.mutation.lock().recover();
        let stats = DatasetStats::measured(lengths);
        let tokenized = TokenizedStats {
            stats: StoredStats::from(&stats),
            truncation: REPORTED_CONTEXTS
                .iter()
                .map(|context| {
                    (
                        context.to_string(),
                        // Two decimals, like every other fraction on the wire
                        // a rate is read, not computed on.
                        (stats.truncation_fraction(*context) * 100.0).round() / 100.0,
                    )
                })
                .collect(),
            lengths: stats.lengths().to_vec(),
        };
        let Some(mut meta) = self.get(id) else {
            return Ok(None);
        };
        meta.tokenized
            .insert(tokenizer_key.to_string(), tokenized.clone());
        self.save_unlocked(meta)?;
        Ok(Some(tokenized))
    }

    /// Streams `body` to a temporary file while hashing it, then - once the
    /// content is known - either discards the temporary (the content already
    /// exists) or measures it, moves it into place and indexes it.
    ///
    /// `max_dataset_bytes` is enforced here as well as by the HTTP body-limit
    /// layer that wraps this route: the layer is sized from the same
    /// operator setting, but a store used directly (a test, a future CLI
    /// import) gets the same guarantee without depending on that layer.
    ///
    /// Both limits are checked twice on purpose. A client that declares its
    /// `Content-Length` is refused before the first byte is written; one that
    /// does not is caught as it streams, which is the only moment its size
    /// becomes known. The quota therefore admits a transient overshoot - at
    /// most one `max_dataset_bytes` of temporary, for an undeclared upload,
    /// released the moment it is refused. Holding the index lock across a
    /// whole upload is the alternative, and it would serialize every
    /// ingestion against every listing.
    pub async fn ingest(
        &self,
        mut body: impl futures_core::Stream<Item = Result<bytes::Bytes, BodyError>> + Unpin,
        options: IngestOptions,
    ) -> Result<Ingested, IngestError> {
        let IngestOptions {
            format: format_hint,
            name,
            declared_bytes,
            max_dataset_bytes,
            max_datasets_bytes,
            idle_timeout,
        } = options;
        let max_bytes = max_dataset_bytes.unwrap_or(DEFAULT_MAX_DATASET_BYTES);

        // What the client says it is about to send is enough to refuse it
        // without touching the disk. It stays a claim, so it never *replaces*
        // the checks below - it only spares an upload nobody could accept.
        if let Some(declared) = declared_bytes {
            if declared > max_bytes {
                return Err(IngestError::TooLarge { max_bytes });
            }
            if let Some(quota) = max_datasets_bytes {
                let used = self.total_bytes();
                if used.saturating_add(declared) > quota {
                    return Err(IngestError::QuotaExceeded {
                        requested: declared,
                        used,
                        quota,
                    });
                }
            }
        }

        let tmp_name = uuid::Uuid::new_v4().to_string();
        let tmp_path = self.root.join("tmp").join(&tmp_name);

        let (bytes_written, sha256) = {
            let mut file = tokio::fs::File::create(&tmp_path).await?;
            let mut hasher = Sha256::new();
            let mut total: u64 = 0;
            let mut failure = None;
            loop {
                let next = match idle_timeout {
                    Some(limit) => match tokio::time::timeout(limit, body.next()).await {
                        Ok(next) => next,
                        Err(_) => {
                            failure = Some(IngestError::Inactive {
                                seconds: limit.as_secs(),
                            });
                            break;
                        }
                    },
                    None => body.next().await,
                };
                let Some(chunk) = next else { break };
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(BodyError::TooLarge) => {
                        failure = Some(IngestError::TooLarge { max_bytes });
                        break;
                    }
                    Err(BodyError::Failed(message)) => {
                        failure = Some(IngestError::Body(message));
                        break;
                    }
                };
                total += chunk.len() as u64;
                if total > max_bytes {
                    failure = Some(IngestError::TooLarge { max_bytes });
                    break;
                }
                hasher.update(&chunk);
                if let Err(error) = file.write_all(&chunk).await {
                    failure = Some(IngestError::Io(error));
                    break;
                }
            }
            if let Some(error) = failure {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(error);
            }
            if let Err(error) = file.flush().await {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(error.into());
            }
            (total, retrograd_core::hex_lower(&hasher.finalize()))
        };

        // From here on every exit but the last must take the temporary with it:
        // the sweep in `open` only catches what a crash left behind, not what a
        // refused upload does.
        let outcome = self
            .place(
                &tmp_path,
                bytes_written,
                sha256,
                format_hint,
                name,
                max_datasets_bytes,
            )
            .await;
        if outcome.is_err() {
            let _ = tokio::fs::remove_file(&tmp_path).await;
        }
        outcome
    }

    /// The second half of [`Self::ingest`]: the content is on disk and hashed,
    /// so its id - and therefore whether there is anything left to do - is
    /// known. Split out so that one `remove_file` at the call site covers every
    /// way this can fail.
    async fn place(
        &self,
        tmp_path: &Path,
        bytes_written: u64,
        sha256: String,
        format_hint: Option<DataFormat>,
        name: Option<String>,
        max_datasets_bytes: Option<u64>,
    ) -> Result<Ingested, IngestError> {
        let id = format!("ds_{sha256}");
        let dir = self.root.join(&id);
        // No await below this guard: a standard mutex is sufficient and keeps
        // the commit indivisible without holding a lock while bytes arrive.
        let _mutation = self.mutation.lock().recover();
        if dir.exists() {
            // Nothing to write, so the temporary goes now rather than through
            // the caller's failure path.
            let _ = std::fs::remove_file(tmp_path);
            if let Some(meta) = self.get(&id) {
                return Ok(Ingested::AlreadyExists(meta));
            }
            // The directory is there but the index lost track of it (a
            // concurrent delete, most likely): read the card straight off
            // disk rather than fail an upload of content that is, in fact,
            // already stored.
            let text = std::fs::read_to_string(dir.join("meta.json"))?;
            let meta: DatasetMeta = serde_json::from_str(&text).map_err(|error| {
                IngestError::Unreadable(retrograd_core::Error::runtime(format!(
                    "existing dataset {id} has an unreadable meta.json: {error}"
                )))
            })?;
            return Ok(Ingested::AlreadyExists(meta));
        }

        let format = match format_hint {
            Some(format) => format,
            None => DataFormat::infer(tmp_path)?,
        };

        if format == DataFormat::ChatJsonl {
            let validation = retrograd_dataset::validate_chat_jsonl(tmp_path)?;
            if !validation.is_valid() {
                return Err(IngestError::Invalid(validation));
            }
        }

        let stats = DatasetStats::estimate_from_file(tmp_path, format)?;

        if let Some(quota) = max_datasets_bytes {
            let used = self.total_bytes();
            if used.saturating_add(bytes_written) > quota {
                return Err(IngestError::QuotaExceeded {
                    requested: bytes_written,
                    used,
                    quota,
                });
            }
        }

        std::fs::create_dir_all(&dir)?;
        let data_file = data_file_name(format);
        let data_path = dir.join(data_file);
        if let Err(error) = std::fs::rename(tmp_path, &data_path) {
            // The commit lock guarantees this directory was created by this
            // attempt, never by a concurrent successful upload.
            let _ = std::fs::remove_dir_all(&dir);
            return Err(error.into());
        }

        let meta = DatasetMeta {
            id: id.clone(),
            sha256,
            format: format_name(format).to_string(),
            data_file: data_file.to_string(),
            bytes: bytes_written,
            examples: stats.examples,
            created_at: crate::runtime::unix_seconds(),
            name,
            stats: StoredStats::from(&stats),
            tokenized: BTreeMap::new(),
        };
        if let Err(error) = self.save_unlocked(meta.clone()) {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(error.into());
        }
        Ok(Ingested::Created(meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn store(label: &str) -> DatasetStore {
        let dir =
            std::env::temp_dir().join(format!("retrograd-datasets-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        DatasetStore::open(&dir)
    }

    fn body(
        bytes: &[u8],
    ) -> impl futures_core::Stream<Item = Result<bytes::Bytes, BodyError>> + use<> {
        stream::iter(vec![Ok(bytes::Bytes::copy_from_slice(bytes))])
    }

    #[tokio::test]
    async fn the_same_content_twice_yields_the_same_id_and_writes_nothing_twice() {
        let store = store("dedup");
        let content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yo\"}]}\n";
        let first = store
            .ingest(
                body(content),
                IngestOptions {
                    name: Some("a".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let id = match first {
            Ingested::Created(meta) => meta.id,
            Ingested::AlreadyExists(_) => panic!("must be new"),
        };
        assert_eq!(store.list().len(), 1);

        let second = store
            .ingest(body(content), IngestOptions::default())
            .await
            .unwrap();
        match second {
            Ingested::AlreadyExists(meta) => assert_eq!(meta.id, id),
            Ingested::Created(_) => panic!("must be a repeat"),
        }
        assert_eq!(store.list().len(), 1, "no second directory was written");
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn concurrent_identical_uploads_commit_one_dataset_without_deleting_it() {
        let store = store("concurrent-dedup");
        let content = b"one complete dataset\n";
        let options = || IngestOptions {
            format: Some(DataFormat::Text),
            ..Default::default()
        };
        let (left, right) = tokio::join!(
            store.ingest(body(content), options()),
            store.ingest(body(content), options()),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert!(matches!(
            left,
            Ingested::Created(_) | Ingested::AlreadyExists(_)
        ));
        assert!(matches!(
            right,
            Ingested::Created(_) | Ingested::AlreadyExists(_)
        ));
        assert_eq!(store.list().len(), 1);
        let meta = store.list().pop().unwrap();
        assert!(store.data_path(&meta).is_file());
        assert!(store.root.join(&meta.id).join("meta.json").is_file());
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn concurrent_uploads_cannot_both_spend_the_same_quota() {
        let store = store("concurrent-quota");
        let options = || IngestOptions {
            format: Some(DataFormat::Text),
            max_datasets_bytes: Some(10),
            ..Default::default()
        };
        let (left, right) = tokio::join!(
            store.ingest(body(b"123456"), options()),
            store.ingest(body(b"abcdef"), options()),
        );
        assert_eq!(usize::from(left.is_ok()) + usize::from(right.is_ok()), 1);
        assert_eq!(store.total_bytes(), 6);
        assert!(matches!(
            left.err().or_else(|| right.err()),
            Some(IngestError::QuotaExceeded { .. })
        ));
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn an_idle_upload_is_stopped_and_its_temporary_is_removed() {
        let store = store("idle-timeout");
        let pending = futures_util::stream::pending::<Result<bytes::Bytes, BodyError>>();
        let error = store
            .ingest(
                pending,
                IngestOptions {
                    idle_timeout: Some(Duration::from_millis(10)),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IngestError::Inactive { .. }));
        assert_eq!(
            std::fs::read_dir(store.root.join("tmp")).unwrap().count(),
            0
        );
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn an_oversized_upload_is_refused_and_leaves_no_temporary() {
        let store = store("oversize");
        let content = vec![b'a'; 100];
        let error = store
            .ingest(
                body(&content),
                IngestOptions {
                    format: Some(DataFormat::Text),
                    max_dataset_bytes: Some(10),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IngestError::TooLarge { max_bytes: 10 }));
        let tmp: Vec<_> = std::fs::read_dir(store.root.join("tmp")).unwrap().collect();
        assert!(
            tmp.is_empty(),
            "an oversized upload must not leave a temporary file behind"
        );
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn a_malformed_chat_jsonl_reports_every_problem() {
        let store = store("invalid");
        let content =
            b"not json\nnot json either\n{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}\n";
        let error = store
            .ingest(
                body(content),
                IngestOptions {
                    format: Some(DataFormat::ChatJsonl),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        match error {
            IngestError::Invalid(validation) => {
                assert_eq!(validation.total, 3, "{validation:?}");
                assert_eq!(validation.errors.len(), 3, "{validation:?}");
            }
            other => panic!("expected a validation failure, got {other:?}"),
        }
        assert_eq!(store.list().len(), 0);
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn the_quota_refuses_new_content_but_never_a_repeat() {
        let store = store("quota");
        let a = b"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]}\n";
        let first = store
            .ingest(
                body(a),
                IngestOptions {
                    max_datasets_bytes: Some(1_000_000),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let bytes_used = first.meta().bytes;

        let b = b"{\"messages\":[{\"role\":\"user\",\"content\":\"c\"},{\"role\":\"assistant\",\"content\":\"d\"}]}\n";
        let error = store
            .ingest(
                body(b),
                IngestOptions {
                    max_datasets_bytes: Some(bytes_used),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IngestError::QuotaExceeded { .. }));

        // The same content again is a no-op, not a quota check: nothing new is
        // written.
        let repeat = store
            .ingest(
                body(a),
                IngestOptions {
                    max_datasets_bytes: Some(bytes_used),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(matches!(repeat, Ingested::AlreadyExists(_)));
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn a_declared_size_is_refused_before_anything_is_written() {
        let store = store("declared");
        let content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]}\n";

        // Over the per-dataset limit, and over the quota: both are answered
        // from the claim alone.
        let error = store
            .ingest(
                body(content),
                IngestOptions {
                    declared_bytes: Some(10_000),
                    max_dataset_bytes: Some(1_000),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IngestError::TooLarge { max_bytes: 1_000 }));

        let error = store
            .ingest(
                body(content),
                IngestOptions {
                    declared_bytes: Some(10_000),
                    max_datasets_bytes: Some(1_000),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IngestError::QuotaExceeded { .. }));

        // Refused before the disk was touched, so not even a temporary exists.
        let tmp: Vec<_> = std::fs::read_dir(store.root.join("tmp")).unwrap().collect();
        assert!(tmp.is_empty(), "nothing should have been written");
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn a_body_cut_by_the_limit_layer_is_a_size_failure_not_a_transport_one() {
        let store = store("body-too-large");
        let cut = stream::iter(vec![
            Ok(bytes::Bytes::from_static(b"{\"messages\":[]}")),
            Err(BodyError::TooLarge),
        ]);
        let error = store
            .ingest(cut, IngestOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, IngestError::TooLarge { .. }),
            "the limit layer's rejection must not read as a hung-up client"
        );
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }

    #[tokio::test]
    async fn a_tokenization_is_cached_on_the_card_and_survives_a_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "retrograd-datasets-tokenize-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = DatasetStore::open(&dir);
        let content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]}\n";
        let id = store
            .ingest(body(content), IngestOptions::default())
            .await
            .unwrap()
            .meta()
            .id
            .clone();

        let model = ModelInfo {
            architecture: "qwen3".into(),
            n_vocab: 151_936,
            file_size_bytes: 4096,
            ..Default::default()
        };
        let key = tokenizer_key(&model, "model-a");
        // Two examples astride 1024: exactly half of them truncate there, and
        // none of them at 2048.
        let stored = store
            .record_tokenized(&id, &key, vec![600, 1500])
            .unwrap()
            .expect("the dataset is still there");
        assert!(stored.stats.measured);
        assert_eq!(stored.stats.max, 1500);
        assert_eq!(stored.truncation["512"], 1.0);
        assert_eq!(stored.truncation["1024"], 0.5);
        assert_eq!(stored.truncation["2048"], 0.0);

        // The lengths come back exactly, which is what lets a resolution report
        // an exact truncation rate at an `n_ctx` nobody listed up front.
        let reopened = DatasetStore::open(&dir);
        let meta = reopened.get(&id).expect("the card survives");
        let cached = meta.tokenized_for(&key).expect("the measurement survives");
        let stats = cached.dataset_stats().expect("lengths were persisted");
        assert!(stats.measured);
        assert_eq!(stats.truncation_fraction(700), 0.5);

        // A different tokenizer is a different key, so nothing is reused.
        let other = tokenizer_key(&model, "model-b");
        assert!(meta.tokenized_for(&other).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tokenizer_key_uses_the_vocabulary_and_complete_model_identity() {
        let model = ModelInfo {
            architecture: "qwen3".into(),
            n_vocab: 151_936,
            file_size_bytes: 4096,
            n_layer: 24,
            ..Default::default()
        };
        // The number of layers does not change how text tokenizes.
        let deeper = ModelInfo {
            n_layer: 48,
            ..model.clone()
        };
        assert_eq!(
            tokenizer_key(&model, "content"),
            tokenizer_key(&deeper, "content")
        );
        // The vocabulary and content identity both matter.
        let other_vocab = ModelInfo {
            n_vocab: 32_000,
            ..model.clone()
        };
        assert_ne!(
            tokenizer_key(&model, "content"),
            tokenizer_key(&other_vocab, "content")
        );
        assert_ne!(
            tokenizer_key(&model, "content"),
            tokenizer_key(&model, "other-content")
        );
    }

    #[tokio::test]
    async fn a_store_reopened_sees_what_the_previous_one_wrote() {
        let dir =
            std::env::temp_dir().join(format!("retrograd-datasets-reopen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = DatasetStore::open(&dir);
        let content = b"plain text corpus with more than a few words in it\n".repeat(20);
        let ingested = store
            .ingest(
                body(&content),
                IngestOptions {
                    format: Some(DataFormat::Text),
                    name: Some("corpus".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let id = ingested.meta().id.clone();
        drop(store);

        let reopened = DatasetStore::open(&dir);
        let meta = reopened.get(&id).expect("the dataset survives a reopen");
        assert_eq!(meta.name.as_deref(), Some("corpus"));
        assert_eq!(meta.format, "text");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_corrupted_meta_json_is_isolated_rather_than_failing_startup() {
        let dir =
            std::env::temp_dir().join(format!("retrograd-datasets-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("datasets/ds_broken")).unwrap();
        std::fs::write(dir.join("datasets/ds_broken/meta.json"), "not json").unwrap();
        let store = DatasetStore::open(&dir);
        assert!(store.list().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_reports_whether_there_was_anything_to_remove() {
        let store = store("delete");
        assert!(!store.delete("ds_does_not_exist"));
        let _ = std::fs::remove_dir_all(store.root.parent().unwrap());
    }
}
