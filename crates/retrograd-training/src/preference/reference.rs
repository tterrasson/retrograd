//! The reference a preference loss compares the policy with: two summed
//! log-probabilities per pair, scored once before the first step, and the file
//! that keeps them across a resume.
//!
//! Scoring once is what makes the *initial* policy a reference at all: it is
//! gone after the first step, and no resume can reconstruct it. The cache is
//! therefore not an optimization for that source but the only way to continue
//! the run.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use retrograd_checkpoint as checkpoint;
use retrograd_config::ReferenceSource;
use retrograd_core::{Error, Result};
use retrograd_engine::Trainer;
use serde::{Deserialize, Serialize};

use super::{Pair, PreparedPreference, score_pair};

/// File name of the cache, at the root of the checkpoint directory.
pub const CACHE_FILE: &str = "preference-reference.bin";

const MAGIC: &[u8; 8] = b"RGPREFRF";
const VERSION: u32 = 1;

/// Reference sums, `[chosen, rejected]`, one per pair of the training set and
/// of the evaluation set, in file order.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceTable {
    pub train: Vec<[f64; 2]>,
    pub eval: Vec<[f64; 2]>,
}

impl ReferenceTable {
    /// Content hash of the table, over the bits of every score.
    pub fn fingerprint(&self) -> String {
        let mut fingerprinter = checkpoint::Fingerprinter::new();
        for value in self.train.iter().chain(&self.eval).flatten() {
            fingerprinter.update(&value.to_bits().to_le_bytes());
        }
        fingerprinter.finish()
    }
}

/// Scores every pair of `prepared` under `source`. `Initial` scores the trainer
/// as it is, so it must run before anything has trained it.
pub fn compute(
    trainer: &mut Trainer,
    prepared: &PreparedPreference,
    source: ReferenceSource,
) -> Result<ReferenceTable> {
    let span = tracing::info_span!(
        target: "retrograd::training::preference",
        "reference",
        source = source.as_str(),
        pairs = prepared.pairs.len() + prepared.eval.len()
    );
    let _entered = span.enter();
    let score_all = |trainer: &mut Trainer| -> Result<ReferenceTable> {
        let mut score = |pairs: &[Pair]| {
            pairs
                .iter()
                .map(|pair| score_pair(trainer, pair))
                .collect::<Result<Vec<_>>>()
        };
        Ok(ReferenceTable {
            train: score(&prepared.pairs)?,
            eval: score(&prepared.eval)?,
        })
    };
    match source {
        ReferenceSource::Initial => score_all(trainer),
        ReferenceSource::Base | ReferenceSource::Model => trainer.with_reference_policy(score_all),
    }
}

/// What a cache must agree with before its scores are used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheHeader {
    pub pairs: u64,
    pub eval_pairs: u64,
    /// Fingerprint of the prepared pairs the scores are aligned with.
    pub corpus: String,
    /// Which reference scored them, and the identity of what it was loaded
    /// from.
    pub reference: String,
    pub loss: String,
}

/// Why a cache cannot stand for this run's reference.
#[derive(Debug, thiserror::Error)]
pub enum ReferenceCacheError {
    #[error(
        "{path} is missing: the reference of this run is its initial policy, which no longer \
         exists once training has started, so the run cannot be resumed - start it again \
         without checkpoint.resume_from"
    )]
    MissingInitial { path: PathBuf },
    #[error("{path} is not a preference reference cache")]
    NotACache { path: PathBuf },
    #[error("{path} has format version {found}, and this build reads version {VERSION}")]
    Version { path: PathBuf, found: u32 },
    #[error("{path} is truncated: its header announces more scores than it holds")]
    Truncated { path: PathBuf },
    #[error("{path} has an unreadable header: {message}")]
    Header { path: PathBuf, message: String },
    #[error(
        "{path} was scored for another run: its {field} is {found}, and this run's is {expected}"
    )]
    Mismatch {
        path: PathBuf,
        field: &'static str,
        expected: String,
        found: String,
    },
    #[error("cannot write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl From<ReferenceCacheError> for Error {
    fn from(error: ReferenceCacheError) -> Self {
        Error::checkpoint(error.to_string())
    }
}

/// The cache of one run: where it lives and what it must agree with.
#[derive(Clone, Debug)]
pub struct ReferenceCache {
    pub path: PathBuf,
    pub header: CacheHeader,
}

impl ReferenceCache {
    pub fn new(directory: &Path, header: CacheHeader) -> Self {
        Self {
            path: directory.join(CACHE_FILE),
            header,
        }
    }

    /// Writes `table` next to its header, through a temporary file renamed over
    /// the target, so a crash leaves the previous cache or none - never half of
    /// one.
    pub fn write(&self, table: &ReferenceTable) -> Result<()> {
        let error = |source| ReferenceCacheError::Write {
            path: self.path.clone(),
            source,
        };
        let header = serde_json::to_vec(&self.header)
            .map_err(|source| error(std::io::Error::other(source)))?;
        let header_len = u32::try_from(header.len())
            .map_err(|_| Error::overflow("the reference cache header does not fit in u32"))?;
        let mut bytes = Vec::with_capacity(16 + header.len() + 16 * table.train.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in table.train.iter().chain(&table.eval).flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(error)?;
        }
        let temporary = self.path.with_extension("bin.tmp");
        let mut file = fs::File::create(&temporary).map_err(error)?;
        file.write_all(&bytes).map_err(error)?;
        file.sync_all().map_err(error)?;
        fs::rename(&temporary, &self.path).map_err(error)?;
        Ok(())
    }

    /// The cached table, `None` when there is no file, or an error when the
    /// file exists and disagrees with this run.
    pub fn read(&self) -> Result<Option<ReferenceTable>> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let path = || self.path.clone();
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(ReferenceCacheError::NotACache { path: path() }.into());
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        let version = word(8);
        if version != VERSION {
            return Err(ReferenceCacheError::Version {
                path: path(),
                found: version,
            }
            .into());
        }
        // A `u32` header length always fits a `usize` on the targets this
        // runs on; the slice bound below is what guards the file itself.
        let header_end = 16 + word(12) as usize;
        let header = bytes
            .get(16..header_end)
            .ok_or_else(|| ReferenceCacheError::Truncated { path: path() })?;
        let header: CacheHeader =
            serde_json::from_slice(header).map_err(|error| ReferenceCacheError::Header {
                path: path(),
                message: error.to_string(),
            })?;
        let expected = &self.header;
        for (field, expected, found) in [
            ("loss", expected.loss.clone(), header.loss.clone()),
            (
                "reference",
                expected.reference.clone(),
                header.reference.clone(),
            ),
            ("corpus", expected.corpus.clone(), header.corpus.clone()),
            (
                "pair count",
                expected.pairs.to_string(),
                header.pairs.to_string(),
            ),
            (
                "evaluation pair count",
                expected.eval_pairs.to_string(),
                header.eval_pairs.to_string(),
            ),
        ] {
            if expected != found {
                return Err(ReferenceCacheError::Mismatch {
                    path: path(),
                    field,
                    expected,
                    found,
                }
                .into());
            }
        }
        let (words, rest) = bytes[header_end..].as_chunks::<8>();
        let scores = words
            .iter()
            .map(|word| f64::from_le_bytes(*word))
            .collect::<Vec<_>>();
        let train = usize::try_from(header.pairs)
            .map_err(|_| Error::overflow("the reference cache pair count exceeds usize"))?;
        let eval = usize::try_from(header.eval_pairs)
            .map_err(|_| Error::overflow("the reference cache pair count exceeds usize"))?;
        if scores.len() != 2 * (train + eval) || !rest.is_empty() {
            return Err(ReferenceCacheError::Truncated { path: path() }.into());
        }
        let pairs = scores.as_chunks::<2>().0.to_vec();
        let (train, eval) = pairs.split_at(train);
        Ok(Some(ReferenceTable {
            train: train.to_vec(),
            eval: eval.to_vec(),
        }))
    }

    /// [`Self::read`] for a source that cannot be scored again: a missing file
    /// is an error.
    pub fn read_initial(&self) -> Result<ReferenceTable> {
        self.read()?.ok_or_else(|| {
            ReferenceCacheError::MissingInitial {
                path: self.path.clone(),
            }
            .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> CacheHeader {
        CacheHeader {
            pairs: 2,
            eval_pairs: 1,
            corpus: "corpus".into(),
            reference: "initial|model".into(),
            loss: "dpo".into(),
        }
    }

    fn table() -> ReferenceTable {
        ReferenceTable {
            train: vec![[-1.5, -2.25], [-3.0, -0.125]],
            eval: vec![[-7.0, -8.5]],
        }
    }

    fn directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "retrograd-preference-cache-{label}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn a_table_survives_the_trip_and_its_header_is_checked_field_by_field() {
        let root = directory("round-trip");
        let cache = ReferenceCache::new(&root, header());
        assert_eq!(cache.read().unwrap(), None);
        assert!(matches!(
            cache.read_initial().unwrap_err(),
            Error::Checkpoint(message) if message.contains("cannot be resumed")
        ));
        cache.write(&table()).unwrap();
        assert_eq!(cache.read().unwrap(), Some(table()));
        assert_eq!(cache.read_initial().unwrap(), table());

        type Change = fn(&mut CacheHeader);
        let changes: [(&str, Change); 5] = [
            ("loss", |header| header.loss = "ipo".into()),
            ("reference", |header| header.reference = "base|model".into()),
            ("corpus", |header| header.corpus = "other".into()),
            ("pair count", |header| header.pairs = 3),
            ("evaluation pair count", |header| header.eval_pairs = 0),
        ];
        for (field, change) in changes {
            let mut expected = header();
            change(&mut expected);
            let error = ReferenceCache::new(&root, expected).read().unwrap_err();
            assert!(
                matches!(&error, Error::Checkpoint(message) if message.contains(&format!("its {field} is"))),
                "{field}: {error}"
            );
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_file_that_is_not_a_cache_is_refused() {
        let root = directory("garbage");
        let cache = ReferenceCache::new(&root, header());
        fs::create_dir_all(&root).unwrap();
        fs::write(&cache.path, b"not a cache at all").unwrap();
        assert!(cache.read().unwrap_err().to_string().contains("is not a"));

        cache.write(&table()).unwrap();
        let bytes = fs::read(&cache.path).unwrap();
        fs::write(&cache.path, &bytes[..bytes.len() - 8]).unwrap();
        assert!(cache.read().unwrap_err().to_string().contains("truncated"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_fingerprint_follows_the_scores() {
        let mut other = table();
        assert_eq!(table().fingerprint(), other.fingerprint());
        other.eval[0][1] = -8.25;
        assert_ne!(table().fingerprint(), other.fingerprint());
    }
}
