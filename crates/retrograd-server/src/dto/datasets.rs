//! Datasets

use super::*;

schema! {
/// Length statistics as they ride along a dataset's own card - the flattened
/// view of `retrograd_plan::DatasetStats` a client reads without fetching a
/// plan.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct DatasetStatsView {
    /// False when these came from the character heuristic rather than a real
    /// tokenizer - true once `POST /v1/datasets/{id}/tokenize` has run.
    pub measured: bool,
    pub total_tokens: u64,
    pub p50: u32,
    pub p90: u32,
    pub p99: u32,
    pub max: u32,
}
}

schema! {
/// `POST /v1/datasets` and `GET /v1/datasets/{id}`.
///
/// `id = "ds_" + sha256(content)`: two uploads of the same bytes answer the
/// same card, which is where the idempotence of `POST /v1/datasets` comes
/// from.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetView {
    pub id: String,
    pub sha256: String,
    /// `chat-jsonl` | `text`.
    pub format: String,
    pub bytes: u64,
    pub examples: u64,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub stats: DatasetStatsView,
}
}

impl From<&crate::datasets::DatasetMeta> for DatasetView {
    fn from(meta: &crate::datasets::DatasetMeta) -> Self {
        Self {
            id: meta.id.clone(),
            sha256: meta.sha256.clone(),
            format: meta.format.clone(),
            bytes: meta.bytes,
            examples: meta.examples,
            created_at: meta.created_at,
            name: meta.name.clone(),
            stats: DatasetStatsView {
                measured: meta.stats.measured,
                total_tokens: meta.stats.total,
                p50: meta.stats.p50,
                p90: meta.stats.p90,
                p99: meta.stats.p99,
                max: meta.stats.max,
            },
        }
    }
}

schema! {
/// `GET /v1/datasets`
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetListing {
    pub datasets: Vec<DatasetView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}
}

schema! {
/// `GET /v1/datasets/{id}/preview`
///
/// Examples exactly as they are read off disk: a `chat-jsonl` dataset answers
/// parsed JSON objects, a `text` dataset answers raw lines - never
/// re-formatted, since the point of a preview is to show what training would
/// actually see.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetPreview {
    pub id: String,
    pub format: String,
    #[cfg_attr(feature = "openapi", schema(value_type = Vec<Object>))]
    pub examples: Vec<serde_json::Value>,
}
}

schema! {
/// `POST /v1/datasets/{id}/tokenize`
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    /// The model whose tokenizer and chat template decide the lengths. Subject
    /// to the same `path_roots` check as `recipe.model`.
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub model: std::path::PathBuf,
}
}

schema! {
/// `POST /v1/datasets/{id}/tokenize` - the measured lengths, and what they mean
/// for a context you have not chosen yet.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetTokenization {
    pub id: String,
    /// The cache key these lengths were stored under: the vocabulary and chat
    /// template that produced them, not the file that carried them. Every model
    /// answering the same key reuses this measurement.
    pub tokenizer: String,
    /// `measured` is `true` here by construction - a real tokenizer ran.
    pub stats: DatasetStatsView,
    /// The share of examples that would be truncated at each of a few common
    /// contexts, keyed by the context as a decimal string.
    pub truncation: std::collections::BTreeMap<String, f64>,
}
}
