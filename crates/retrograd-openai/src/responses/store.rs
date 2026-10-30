//! Where a stored response lives: the [`ResponseStore`] trait, with a bounded
//! in-memory LRU as the default and a directory of one JSON per response as
//! the on-disk option.
//!
//! `store: true` (the API default) writes to the store the endpoint was built
//! with; `store: false` writes nothing. A `previous_response_id` that points
//! at an unknown or evicted response is a 404 in the OpenAI envelope whose
//! message says it may have been evicted from the in-memory store.
//!
//! Two protections keep a chain usable while it is alive: an LRU eviction
//! never removes an entry a living descendant still points at (each stored
//! response pins its `previous_response_id`), and a continuation chain is
//! bounded in depth. Reads, deletions and continuations are isolated by
//! owner: when the request carries a [`StoreOwner`] extension, a stored
//! response owned by someone else answers 404 like an unknown one.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;

/// The authenticated identity a stored response belongs to, inserted as a
/// request extension by the frontend that has principals. A stored response
/// with an owner is readable, deletable and continuable only by the same
/// owner; a request without one sees only ownerless responses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreOwner(pub String);

/// The `resp_…` id of one stored response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseId(pub String);

/// One stored response: the object the API returns, the input items a
/// `previous_response_id` continuation replays (the whole chain before this
/// response's own output), and the bookkeeping the store's protections need.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StoredResponse {
    /// The `response` object as the API returns it.
    pub response: serde_json::Value,
    /// The input items a continuation replays before appending the new
    /// `input`: the chain's earlier input items plus every earlier output.
    pub input_items: Vec<serde_json::Value>,
    pub created_at: u64,
    /// Approximate size on the store's budget: the serialized response plus
    /// its input items.
    pub bytes: usize,
    /// The response this one continues, when it does: kept pinned in the
    /// store while this entry lives.
    pub previous_response_id: Option<String>,
    /// 1 for a first turn, one more per continuation.
    pub chain_depth: usize,
    /// The model id the response was produced for.
    pub model: String,
    /// Which weights produced it: `cold:<model path>` (plus the adapter when
    /// there is one) for weights on disk, `live:<model id>` for a run that was
    /// training. A continuation refuses a different base model clearly
    /// instead of replaying tokens computed under another one.
    pub weights: String,
    /// The weights' fingerprint (checkpoint id, live step), kept for the KV
    /// pinning a later scheduler may add: a chain prolonged on other weights
    /// is served, but its pinned KV is invalidated.
    pub fingerprint: String,
    /// The base model path when the target was weights on disk.
    pub base_model: Option<String>,
    /// The owner from the request that created it, when it carried one.
    pub owner: Option<String>,
}

/// Why a `put` was refused.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(
        "the response store is full ({max_responses} responses, {max_bytes} bytes) and every \
         remaining entry is pinned by a live continuation"
    )]
    Full {
        max_responses: usize,
        max_bytes: usize,
    },
    #[error("the continuation chain would be longer than {max_chain_depth} responses")]
    ChainTooDeep { max_chain_depth: usize },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization: {0}")]
    Json(#[from] serde_json::Error),
}

/// The bounds of a store.
#[derive(Clone, Copy, Debug)]
pub struct StoreOptions {
    /// Most responses kept. Default 1024.
    pub max_responses: usize,
    /// Most bytes kept (serialized responses plus input items). Default 64 MiB.
    pub max_bytes: usize,
    /// Longest continuation chain. Default 64.
    pub max_chain_depth: usize,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            max_responses: 1024,
            max_bytes: 64 * 1024 * 1024,
            max_chain_depth: 64,
        }
    }
}

/// Where stored responses live.
pub trait ResponseStore: Send + Sync {
    /// Stores one response and returns its id.
    fn put(&self, response: StoredResponse) -> Result<ResponseId, StoreError>;
    /// The response `id` names, or `None` when it is unknown or was evicted.
    fn get(&self, id: &str) -> Option<StoredResponse>;
    /// Removes one response, releasing the pin it held on its ancestor.
    fn delete(&self, id: &str) -> bool;
    /// The input items of one response, for `GET /v1/responses/{id}/input_items`.
    fn input_items(&self, id: &str) -> Option<Vec<serde_json::Value>>;
    /// Removes the responses created before `older_than_unix` that no living
    /// descendant pins, and returns how many went. A store without a time
    /// budget (the file store) leaves this to the operator.
    fn purge(&self, older_than_unix: u64) -> usize;
}

/// An id is a path segment of a file store: refuse anything that could escape.
pub(crate) fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

struct Entry {
    stored: StoredResponse,
}

struct MemoryInner {
    entries: HashMap<String, Entry>,
    /// LRU order, oldest first.
    order: VecDeque<String>,
    /// How many living descendants pin each ancestor.
    pins: HashMap<String, usize>,
    bytes: usize,
}

/// The default store: a bounded in-memory LRU. No write ever touches the
/// disk; when the bound is reached the least recently used *unpinned* entry
/// goes, and a store whose every entry is pinned refuses the put.
pub struct MemoryStore {
    inner: Mutex<MemoryInner>,
    options: StoreOptions,
}

impl MemoryStore {
    pub fn new(options: StoreOptions) -> Self {
        Self {
            inner: Mutex::new(MemoryInner {
                entries: HashMap::new(),
                order: VecDeque::new(),
                pins: HashMap::new(),
                bytes: 0,
            }),
            options,
        }
    }

    fn remove(inner: &mut MemoryInner, id: &str) -> Option<StoredResponse> {
        let entry = inner.entries.remove(id)?;
        inner.order.retain(|other| other != id);
        inner.bytes = inner.bytes.saturating_sub(entry.stored.bytes);
        if let Some(ancestor) = &entry.stored.previous_response_id
            && let Some(count) = inner.pins.get_mut(ancestor)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                inner.pins.remove(ancestor);
            }
        }
        Some(entry.stored)
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new(StoreOptions::default())
    }
}

impl ResponseStore for MemoryStore {
    fn put(&self, mut response: StoredResponse) -> Result<ResponseId, StoreError> {
        let id = response
            .response
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                StoreError::Io(std::io::Error::other("a stored response needs an id"))
            })?;
        let mut inner = self.inner.lock().expect("store lock");
        if inner.entries.contains_key(&id) {
            return Err(StoreError::Io(std::io::Error::other(
                "a stored response id already exists",
            )));
        }
        if response.bytes > self.options.max_bytes || self.options.max_responses == 0 {
            return Err(StoreError::Full {
                max_responses: self.options.max_responses,
                max_bytes: self.options.max_bytes,
            });
        }
        response.chain_depth = 1;
        if let Some(ancestor) = &response.previous_response_id {
            let Some(parent) = inner.entries.get(ancestor) else {
                return Err(StoreError::Io(std::io::Error::other(format!(
                    "the previous response `{ancestor}` is gone from the store"
                ))));
            };
            if parent.stored.chain_depth >= self.options.max_chain_depth {
                return Err(StoreError::ChainTooDeep {
                    max_chain_depth: self.options.max_chain_depth,
                });
            }
            response.chain_depth = parent.stored.chain_depth + 1;
            *inner.pins.entry(ancestor.clone()).or_insert(0) += 1;
        }
        let bytes = response.bytes;
        inner.bytes += bytes;
        inner.entries.insert(id.clone(), Entry { stored: response });
        inner.order.push_back(id.clone());
        // Evict the least recently used unpinned entries until within bounds.
        while inner.entries.len() > self.options.max_responses
            || inner.bytes > self.options.max_bytes
        {
            let Some(victim) = inner
                .order
                .iter()
                .find(|candidate| **candidate != id && !inner.pins.contains_key(*candidate))
                .cloned()
            else {
                // Everything left is pinned: the put itself fails rather than
                // breaking a chain a client may still continue.
                Self::remove(&mut inner, &id);
                return Err(StoreError::Full {
                    max_responses: self.options.max_responses,
                    max_bytes: self.options.max_bytes,
                });
            };
            Self::remove(&mut inner, &victim);
        }
        Ok(ResponseId(id))
    }

    fn get(&self, id: &str) -> Option<StoredResponse> {
        let mut inner = self.inner.lock().expect("store lock");
        let stored = inner.entries.get(id)?.stored.clone();
        // A read makes the entry recently used.
        inner.order.retain(|other| other != id);
        inner.order.push_back(id.to_owned());
        Some(stored)
    }

    fn delete(&self, id: &str) -> bool {
        Self::remove(&mut self.inner.lock().expect("store lock"), id).is_some()
    }

    fn input_items(&self, id: &str) -> Option<Vec<serde_json::Value>> {
        self.get(id).map(|stored| stored.input_items)
    }

    fn purge(&self, older_than_unix: u64) -> usize {
        let mut inner = self.inner.lock().expect("store lock");
        let victims: Vec<String> = inner
            .entries
            .iter()
            .filter(|(id, entry)| {
                entry.stored.created_at < older_than_unix && !inner.pins.contains_key(*id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        let count = victims.len();
        for victim in victims {
            Self::remove(&mut inner, &victim);
        }
        count
    }
}

/// The on-disk store: one JSON file per response in `dir`, written atomically
/// (a temporary file renamed over the target). Bounds are the operator's
/// (disk quota, TTL purge); the chain-depth bound still applies, and a purge
/// never removes a file a living descendant points at.
pub struct FileStore {
    dir: PathBuf,
    max_chain_depth: usize,
    // Serialize puts with deletes and purges so an ancestor cannot disappear
    // between validation and publication of its continuation.
    mutation: Mutex<()>,
}

impl FileStore {
    pub fn new(dir: impl Into<PathBuf>, options: StoreOptions) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            max_chain_depth: options.max_chain_depth,
            mutation: Mutex::new(()),
        })
    }

    fn path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!("{id}.json")))
    }
}

impl ResponseStore for FileStore {
    fn put(&self, mut response: StoredResponse) -> Result<ResponseId, StoreError> {
        let _guard = self.mutation.lock().expect("store lock");
        response.chain_depth = 1;
        let id = response
            .response
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                StoreError::Io(std::io::Error::other("a stored response needs an id"))
            })?;
        if let Some(ancestor) = &response.previous_response_id {
            let parent_path = self
                .path(ancestor)
                .ok_or_else(|| StoreError::Io(std::io::Error::other("an invalid previous id")))?;
            let parent: StoredResponse =
                serde_json::from_str(&std::fs::read_to_string(&parent_path).map_err(|error| {
                    StoreError::Io(std::io::Error::other(format!(
                        "the previous response `{ancestor}` is gone from the store: {error}"
                    )))
                })?)
                .map_err(|error| {
                    StoreError::Io(std::io::Error::other(format!(
                        "the previous response `{ancestor}` is unreadable: {error}"
                    )))
                })?;
            response.chain_depth = parent.chain_depth + 1;
            if parent.chain_depth >= self.max_chain_depth {
                return Err(StoreError::ChainTooDeep {
                    max_chain_depth: self.max_chain_depth,
                });
            }
        }
        let path = self
            .path(&id)
            .ok_or_else(|| StoreError::Io(std::io::Error::other("an invalid response id")))?;
        if path.exists() {
            return Err(StoreError::Io(std::io::Error::other(
                "a stored response id already exists",
            )));
        }
        let payload = serde_json::to_string(&response)?;
        let temporary = self
            .dir
            .join(format!(".{id}.{}.tmp", uuid::Uuid::new_v4().simple()));
        std::fs::write(&temporary, payload)?;
        std::fs::rename(&temporary, &path)?;
        Ok(ResponseId(id))
    }

    fn get(&self, id: &str) -> Option<StoredResponse> {
        let path = self.path(id)?;
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn delete(&self, id: &str) -> bool {
        let _guard = self.mutation.lock().expect("store lock");
        match self.path(id) {
            Some(path) => std::fs::remove_file(path).is_ok(),
            None => false,
        }
    }

    fn input_items(&self, id: &str) -> Option<Vec<serde_json::Value>> {
        self.get(id).map(|stored| stored.input_items)
    }

    fn purge(&self, older_than_unix: u64) -> usize {
        let _guard = self.mutation.lock().expect("store lock");
        let Ok(read) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        // A file another living response continues is referenced by its
        // `previous_response_id`; the purge recomputes the references rather
        // than trusting pins, because a delete may have orphaned one.
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut candidates: Vec<(PathBuf, String, u64)> = Vec::new();
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(stored) = serde_json::from_str::<StoredResponse>(&text) else {
                continue;
            };
            if let Some(ancestor) = &stored.previous_response_id {
                referenced.insert(ancestor.clone());
            }
            if stored.created_at < older_than_unix {
                let id = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_owned();
                candidates.push((path, id, stored.created_at));
            }
        }
        let mut removed = 0;
        for (path, id, _) in candidates {
            if referenced.contains(&id) {
                continue;
            }
            if std::fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stored(id: &str, previous: Option<&str>, depth: usize, created_at: u64) -> StoredResponse {
        StoredResponse {
            response: json!({"id": id, "object": "response"}),
            input_items: vec![json!({"type": "message", "role": "user", "content": "hi"})],
            created_at,
            bytes: 100,
            previous_response_id: previous.map(str::to_owned),
            chain_depth: depth,
            model: "m".into(),
            weights: "cold:base.gguf".into(),
            fingerprint: "fp".into(),
            base_model: Some("base.gguf".into()),
            owner: None,
        }
    }

    #[test]
    fn failed_oversized_puts_and_duplicate_ids_preserve_existing_entries() {
        let store = MemoryStore::new(StoreOptions {
            max_responses: 2,
            max_bytes: 200,
            max_chain_depth: 64,
        });
        store.put(stored("resp_a", None, 1, 1)).unwrap();
        assert!(store.put(stored("resp_a", None, 1, 2)).is_err());
        let mut large = stored("resp_large", None, 1, 3);
        large.bytes = 201;
        assert!(matches!(store.put(large), Err(StoreError::Full { .. })));
        assert_eq!(store.get("resp_a").unwrap().created_at, 1);
        store.put(stored("resp_b", None, 1, 4)).unwrap();
        assert!(store.get("resp_a").is_some());
    }

    #[test]
    fn an_lru_eviction_never_takes_a_pinned_ancestor() {
        let store = MemoryStore::new(StoreOptions {
            max_responses: 2,
            max_bytes: usize::MAX,
            max_chain_depth: 64,
        });
        store.put(stored("resp_a", None, 1, 1)).unwrap();
        store.put(stored("resp_b", Some("resp_a"), 2, 2)).unwrap();
        // resp_c evicts the LRU entry - resp_a is pinned by resp_b, so resp_b
        // would go instead... it is the only unpinned one, so it goes and
        // resp_a survives.
        store.put(stored("resp_c", None, 1, 3)).unwrap();
        assert!(
            store.get("resp_a").is_some(),
            "the pinned ancestor survives"
        );
        assert!(store.get("resp_b").is_none(), "the unpinned LRU entry goes");
        assert!(store.get("resp_c").is_some());
    }

    #[test]
    fn a_full_store_of_pinned_entries_refuses_the_put() {
        // Two slots, and a chain of three: every entry left is pinned by its
        // descendant, so the third put fails rather than breaking the chain.
        let store = MemoryStore::new(StoreOptions {
            max_responses: 2,
            max_bytes: usize::MAX,
            max_chain_depth: 64,
        });
        store.put(stored("resp_a", None, 1, 1)).unwrap();
        store.put(stored("resp_b", Some("resp_a"), 2, 2)).unwrap();
        let error = store
            .put(stored("resp_c", Some("resp_b"), 3, 3))
            .unwrap_err();
        assert!(matches!(error, StoreError::Full { .. }), "{error}");
        assert!(store.get("resp_a").is_some());
        assert!(store.get("resp_b").is_some());
        assert!(
            store.get("resp_c").is_none(),
            "the refused put wrote nothing"
        );
    }

    #[test]
    fn a_delete_releases_the_pin_on_its_ancestor() {
        let store = MemoryStore::new(StoreOptions {
            max_responses: 2,
            max_bytes: usize::MAX,
            max_chain_depth: 64,
        });
        store.put(stored("resp_a", None, 1, 1)).unwrap();
        store.put(stored("resp_b", Some("resp_a"), 2, 2)).unwrap();
        assert!(store.delete("resp_b"));
        // With the pin released, resp_a is evictable again: two more puts
        // push the store past its bound and the LRU entry (resp_a) goes.
        store.put(stored("resp_c", None, 1, 3)).unwrap();
        store.put(stored("resp_d", None, 1, 4)).unwrap();
        assert!(store.get("resp_a").is_none());
        assert!(store.get("resp_c").is_some());
        assert!(store.get("resp_d").is_some());
        assert!(!store.delete("resp_missing"));
    }

    #[test]
    fn the_chain_depth_is_bounded() {
        let store = MemoryStore::new(StoreOptions {
            max_responses: 100,
            max_bytes: usize::MAX,
            max_chain_depth: 2,
        });
        store.put(stored("resp_a", None, 1, 1)).unwrap();
        store.put(stored("resp_b", Some("resp_a"), 2, 2)).unwrap();
        let error = store
            .put(stored("resp_c", Some("resp_b"), 3, 3))
            .unwrap_err();
        assert!(matches!(error, StoreError::ChainTooDeep { .. }), "{error}");
    }

    #[test]
    fn a_purge_keeps_pinned_ancestors_and_releases_the_rest() {
        let store = MemoryStore::new(StoreOptions::default());
        store.put(stored("resp_old", None, 1, 10)).unwrap();
        store
            .put(stored("resp_new", Some("resp_old"), 2, 20))
            .unwrap();
        store.put(stored("resp_other", None, 1, 5)).unwrap();
        let removed = store.purge(15);
        assert_eq!(removed, 1, "only the unpinned old entry goes");
        assert!(store.get("resp_old").is_some());
        assert!(store.get("resp_new").is_some());
        assert!(store.get("resp_other").is_none());
    }

    #[test]
    fn the_file_store_round_trips_and_purges_around_references() {
        let dir = std::env::temp_dir().join(format!("retrograd-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir, StoreOptions::default()).unwrap();
        store.put(stored("resp_a", None, 1, 10)).unwrap();
        store.put(stored("resp_b", Some("resp_a"), 2, 20)).unwrap();
        let got = store.get("resp_a").expect("stored");
        assert_eq!(got.model, "m");
        assert_eq!(store.input_items("resp_b").unwrap().len(), 1);
        assert_eq!(store.purge(15), 0, "resp_a is referenced by resp_b");
        store.delete("resp_b");
        assert_eq!(store.purge(15), 1, "once unreferenced, the old file goes");
        assert!(store.get("resp_a").is_none());
        assert!(!valid_id("../escape"));
        assert!(valid_id("resp_0123abcd"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
