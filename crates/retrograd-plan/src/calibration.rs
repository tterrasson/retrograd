//! Shape-specific measurements from this machine, used so the next resolution is
//! less wrong.
//!
//! Analysis cannot predict llama.cpp's compute buffers to the MiB. The cost
//! model is deliberately conservative instead, which means it is *systematically*
//! off on a given machine, and a systematic error is exactly the kind a table of
//! correction factors fixes. Each measured resolution writes its ratio here,
//! keyed by what the ratio depends on; every later resolution on the same key
//! starts from it.
//!
//! Pure data, like the rest of this crate: measuring is the caller's job (it
//! needs a `Trainer`, therefore the FFI). What lives here is the schema of
//! `calibration.json`, the key, and the rule for folding an observation into it.
//!
//! Two properties the rule has to keep:
//!
//! - **Conservative.** A factor never falls below one. Measuring less than the
//!   estimate is good news, and it is recorded - but it must not let the *next*
//!   resolution promise more than the model can justify.
//! - **Deterministic.** Same table plus same inputs means the same output
//!   (invariant 3). So the stored factor is the running maximum of the observed
//!   ratios, not an average that would depend on how many runs happened and in
//!   which order.

use std::collections::BTreeMap;
use std::path::Path;

use retrograd_config::{Algorithm, RunConfig};
use retrograd_core::{
    Error, ExecutionProfile, ModelInfo, Result, SharedPrefixFanout, TrainableSet,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cost::{COST_MODEL_VERSION, Calibration, MemoryEstimate};

/// Schema version of `calibration.json`. A file written by a future version is
/// ignored rather than misread: a wrong correction factor is worse than none.
pub const CALIBRATION_VERSION: u32 = 2;

/// Complete identity of one measured graph class. It is deliberately more
/// specific than a model/backend key: a factor learned at one context or
/// ubatch must never be applied to a graph that selects different kernels or
/// has different activation lifetimes.
///
/// The key includes the geometry that determines compute reserves and
/// dequantization scratch. Each geometry starts uncorrected; only the measured
/// one carries a factor, and the resolver re-resolves from scratch after every
/// measurement up to `MAX_CALIBRATION_PASSES`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationKey {
    pub cost_model_version: u32,
    pub engine_fingerprint: String,
    pub kernel_catalog_fingerprint: String,
    pub profile_fingerprint: String,
    pub backend: String,
    pub device_id: String,
    pub model: CalibrationModelClass,
    pub shape: CalibrationShapeClass,
    pub graph: CalibrationGraphClass,
    /// What a base-weight run trains. `None` for an adapter run and skipped in
    /// the JSON, so pre-existing adapter keys stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainable: Option<CalibrationTrainableClass>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationModelClass {
    pub architecture: String,
    pub layers: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub kv_width: u32,
    pub vocabulary: u32,
    pub weight_dtype: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationShapeClass {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub generation_concurrency: u32,
    pub generation_batch: u32,
    pub packed_fanout: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationGraphClass {
    pub algorithm: String,
    pub chunked_cross_entropy: bool,
    pub chunked_ce_tiles: u32,
    pub chunked_ce_seq_chunk: u32,
    pub gradient_checkpointing: bool,
    pub checkpoint_every_n_layers: u32,
    pub checkpoint_dtype: String,
    pub fast_generation_context: bool,
    pub kv_dtype: String,
    pub require_gpu_resident: bool,
}

/// What a `full`, `partial` or `hybrid` run trains, as the key reads it.
///
/// The backward stops at the lowest trainable block, so two selections of the
/// same model build graphs of different heights and need different factors.
/// `manifest_digest` identifies the set; the other fields are carried because
/// each is a term of the estimate the factor corrects.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTrainableClass {
    pub policy: String,
    pub optimizer: String,
    /// `None` when the set reaches outside every block, i.e. the whole
    /// backward runs.
    pub lowest_trainable_layer: Option<u32>,
    /// The set carries the vocabulary projection, which builds a different,
    /// larger loss graph.
    pub trains_loss_head: bool,
    /// `None` when the caller had no resolved set; kept representable so it
    /// cannot read back as an empty set.
    pub manifest_digest: Option<String>,
}

impl CalibrationTrainableClass {
    /// `None` for an adapter policy, whatever set was resolved: its key must
    /// stay unchanged.
    pub fn of(
        training: &retrograd_core::TrainableRunConfig,
        set: Option<&TrainableSet>,
    ) -> Option<Self> {
        if !training.policy.trains_base_weights() {
            return None;
        }
        Some(Self {
            policy: training.policy.as_str().to_string(),
            optimizer: training.optimizer.as_str().to_string(),
            lowest_trainable_layer: set.and_then(TrainableSet::lowest_trainable_layer),
            trains_loss_head: set.is_some_and(TrainableSet::trains_loss_head),
            manifest_digest: set.map(|set| {
                let manifest = set.manifest_lines().join("\n");
                retrograd_core::hex_lower(&Sha256::digest(manifest.as_bytes()))
            }),
        })
    }
}

/// What a measurement on this machine taught about one geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalibrationEntry {
    /// Correction applied to [`MemoryEstimate::optimizer_compute_bytes`] and to
    /// the generation compute buffers. Never below one.
    pub compute_scale: f64,
    /// Correction applied to [`MemoryEstimate::dequant_scratch_bytes`], which is
    /// what the backend's own scratch measures against.
    pub scratch_scale: f64,
    /// The last ratio actually observed, factor floor included or not. This is
    /// the *gap*, recorded even when nothing changes - an estimate
    /// running at half the truth and one running at twice it are both worth
    /// knowing, and only one of them moves a factor.
    pub last_compute_ratio: f64,
    pub last_scratch_ratio: f64,
    /// How many measurements folded into this entry.
    pub samples: u64,
}

impl CalibrationEntry {
    fn factors(&self) -> Calibration {
        Calibration {
            compute_scale: sane(self.compute_scale),
            scratch_scale: sane(self.scratch_scale),
        }
    }
}

/// A factor read back from disk is data a user can edit. Anything that is not a
/// finite number at or above one is read as "no correction".
fn sane(value: f64) -> f64 {
    if value.is_finite() && value >= 1.0 {
        value
    } else {
        1.0
    }
}

/// `calibration.json`, one entry per geometry this machine has measured.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalibrationStore {
    pub version: u32,
    /// `BTreeMap` so the file is byte-stable across writes: a state directory
    /// under version control must not churn because a hash map reordered.
    pub entries: BTreeMap<String, CalibrationEntry>,
}

/// What folding one measurement into the table did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub compute_ratio: f64,
    pub scratch_ratio: f64,
    /// The factors in force after the fold, i.e. what the next resolution on
    /// this key will use.
    pub factors: Calibration,
    /// Whether a factor moved. False means the estimate was already at or above
    /// what the machine did - the good case, and the one that changes nothing.
    pub raised: bool,
}

impl CalibrationStore {
    /// Reads the table, treating every failure as an empty one.
    ///
    /// A missing file is the normal first-run case; a corrupt or future-versioned
    /// one is an operator problem that must not take the server down, because the
    /// server works fine without any calibration at all. The caller logs it.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let Ok(text) = std::fs::read_to_string(path.as_ref()) else {
            return Self::empty();
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(store) if store.version == CALIBRATION_VERSION => store,
            _ => Self::empty(),
        }
    }

    pub fn empty() -> Self {
        Self {
            version: CALIBRATION_VERSION,
            entries: BTreeMap::new(),
        }
    }

    /// Writes the table, creating the parent directory if needed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| Error::runtime(format!("could not create {parent:?}: {error}")))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|error| {
            Error::runtime(format!("could not render the calibration: {error}"))
        })?;
        std::fs::write(path, text)
            .map_err(|error| Error::runtime(format!("could not write {path:?}: {error}")))
    }

    /// The factors to resolve with. Identity for a geometry never measured here.
    pub fn factors(&self, key: &str) -> Calibration {
        self.entries
            .get(key)
            .map(CalibrationEntry::factors)
            .unwrap_or_default()
    }

    pub fn entry(&self, key: &str) -> Option<&CalibrationEntry> {
        self.entries.get(key)
    }

    /// Folds one measurement in and returns what it did.
    ///
    /// `estimated` must be the estimate produced *without* this key's factors,
    /// otherwise the ratio would be measured against an already-corrected number
    /// and the factor would compound with itself on every run.
    pub fn observe(
        &mut self,
        key: &str,
        estimated: &MemoryEstimate,
        measured: &MemoryEstimate,
        measured_scratch_bytes: u64,
    ) -> Observation {
        let compute_ratio = ratio(
            estimated.optimizer_compute_bytes + estimated.generation_compute_bytes,
            measured.optimizer_compute_bytes + measured.generation_compute_bytes,
        );
        let scratch_ratio = ratio(estimated.dequant_scratch_bytes, measured_scratch_bytes);

        let entry = self.entries.entry(key.to_string()).or_default();
        let before = entry.factors();
        // Running maximum, floored at one: a factor that could fall would let a
        // lucky measurement erase an unlucky one, and the next run would be
        // planned on the optimistic half of the evidence.
        entry.compute_scale = sane(entry.compute_scale).max(sane(compute_ratio));
        entry.scratch_scale = sane(entry.scratch_scale).max(sane(scratch_ratio));
        entry.last_compute_ratio = compute_ratio;
        entry.last_scratch_ratio = scratch_ratio;
        entry.samples += 1;
        let factors = entry.factors();
        Observation {
            compute_ratio,
            scratch_ratio,
            factors,
            raised: factors != before,
        }
    }
}

/// Zero on either side means "not comparable", which reads as a ratio of one:
/// the runtime reports zero for a post it has not filled in yet, and dividing by
/// it would turn "unknown" into a correction factor.
fn ratio(estimated: u64, measured: u64) -> f64 {
    if estimated == 0 || measured == 0 {
        return 1.0;
    }
    measured as f64 / estimated as f64
}

/// Stable V2 key for one measured execution class. The readable prefix helps
/// operators inspect the file; the digest covers every structured field and
/// avoids ambiguous ad-hoc separators.
/// `base_trainable` is the resolved set for this configuration, or `None` for
/// an adapter run. A base run passed `None` still gets its own key (no
/// manifest), so it never collides with an adapter key.
pub fn calibration_key(
    profile: &ExecutionProfile,
    model: &ModelInfo,
    config: &RunConfig,
    base_trainable: Option<&TrainableSet>,
) -> String {
    calibration_key_for(
        profile,
        model,
        algorithm_slug(&config.algorithm),
        &config.training,
        base_trainable,
    )
}

/// The single place an algorithm becomes a token of the calibration key. The
/// resolver reads factors before it has a `RunConfig` to hand, and a second
/// mapping there would silently split the read key from the written one.
pub fn algorithm_slug(algorithm: &Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Sft(_) => "sft",
        Algorithm::Ppo(_) => "ppo",
        Algorithm::Grpo(_) => "grpo",
        Algorithm::Distill(_) => "distill",
        Algorithm::AgentGrpo(_) => "agent_grpo",
    }
}

/// Variant used while the resolver is evaluating a `TrainConfig` candidate,
/// before it has rendered the final `RunConfig` document.
pub fn calibration_key_for(
    profile: &ExecutionProfile,
    model: &ModelInfo,
    algorithm: &str,
    training: &retrograd_core::TrainConfig,
    base_trainable: Option<&TrainableSet>,
) -> String {
    let key = CalibrationKey {
        cost_model_version: COST_MODEL_VERSION,
        engine_fingerprint: profile.engine_fingerprint.clone(),
        kernel_catalog_fingerprint: profile.kernel_catalog_fingerprint.clone(),
        profile_fingerprint: profile.fingerprint(),
        backend: profile.device.backend.clone(),
        device_id: profile.device.stable_id.clone(),
        model: CalibrationModelClass {
            architecture: model.architecture.clone(),
            layers: model.n_layer,
            embedding: model.n_embd,
            feed_forward: model.n_ff,
            heads: model.n_head,
            kv_heads: model.n_head_kv,
            kv_width: model.n_embd_k_gqa + model.n_embd_v_gqa,
            vocabulary: model.n_vocab,
            weight_dtype: model.dominant_weight_type.clone(),
        },
        shape: CalibrationShapeClass {
            n_ctx: training.n_ctx,
            n_batch: training.n_batch,
            n_ubatch: training.n_ubatch,
            generation_concurrency: training.generation_concurrency,
            generation_batch: training.generation_batch,
            packed_fanout: match training.shared_prefix_fanout {
                SharedPrefixFanout::Auto => "auto".to_string(),
                SharedPrefixFanout::Off => "off".to_string(),
                SharedPrefixFanout::Max => "max".to_string(),
                SharedPrefixFanout::Exact(value) => format!("exact:{value}"),
            },
        },
        graph: CalibrationGraphClass {
            algorithm: algorithm.to_string(),
            chunked_cross_entropy: training.chunked_cross_entropy,
            chunked_ce_tiles: training.chunked_ce_tiles,
            chunked_ce_seq_chunk: training.chunked_ce_seq_chunk,
            gradient_checkpointing: training.gradient_checkpointing,
            checkpoint_every_n_layers: training.checkpoint_every_n_layers,
            checkpoint_dtype: format!("{:?}", training.checkpoint_dtype).to_ascii_lowercase(),
            fast_generation_context: training.fast_generation_context,
            kv_dtype: format!("{:?}", training.kv_dtype).to_ascii_lowercase(),
            require_gpu_resident: training.require_gpu_resident,
        },
        trainable: CalibrationTrainableClass::of(&training.trainable, base_trainable),
    };
    let encoded = serde_json::to_vec(&key).expect("CalibrationKey is serializable");
    let digest = retrograd_core::hex_lower(&Sha256::digest(encoded));
    format!(
        "v{}/{}:{}",
        COST_MODEL_VERSION, profile.device.backend, digest
    )
}

#[cfg(test)]
mod tests {
    use retrograd_core::{TrainablePolicy, TrainableSet};

    use super::*;

    fn estimate(compute: u64, scratch: u64) -> MemoryEstimate {
        MemoryEstimate {
            optimizer_compute_bytes: compute,
            dequant_scratch_bytes: scratch,
            ..Default::default()
        }
    }

    #[test]
    fn an_underestimate_raises_the_factor_and_an_overestimate_does_not() {
        let mut store = CalibrationStore::empty();
        let observation = store.observe("k", &estimate(100, 50), &estimate(150, 0), 25);
        assert_eq!(observation.compute_ratio, 1.5);
        assert!(observation.raised);
        assert_eq!(store.factors("k").compute_scale, 1.5);
        // The scratch came in *under* the estimate: recorded, never applied.
        assert_eq!(observation.scratch_ratio, 0.5);
        assert_eq!(store.factors("k").scratch_scale, 1.0);
        assert_eq!(store.entry("k").unwrap().last_scratch_ratio, 0.5);

        // A later, kinder measurement does not undo the first one.
        let observation = store.observe("k", &estimate(100, 50), &estimate(110, 0), 25);
        assert!(!observation.raised);
        assert_eq!(store.factors("k").compute_scale, 1.5);
        assert_eq!(store.entry("k").unwrap().last_compute_ratio, 1.1);
        assert_eq!(store.entry("k").unwrap().samples, 2);
    }

    #[test]
    fn an_unmeasured_post_is_not_a_correction_of_zero() {
        let mut store = CalibrationStore::empty();
        let observation = store.observe("k", &estimate(100, 50), &estimate(0, 0), 0);
        assert_eq!(observation.compute_ratio, 1.0);
        assert_eq!(observation.scratch_ratio, 1.0);
        assert!(!observation.raised);
    }

    #[test]
    fn an_unknown_key_resolves_with_the_identity() {
        let store = CalibrationStore::empty();
        assert_eq!(store.factors("nothing"), Calibration::default());
    }

    #[test]
    fn a_file_that_is_missing_corrupt_or_from_the_future_reads_as_empty() {
        let dir =
            std::env::temp_dir().join(format!("retrograd-calibration-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("calibration.json");
        assert!(CalibrationStore::load(&path).entries.is_empty());

        let mut store = CalibrationStore::empty();
        store.observe("k", &estimate(100, 0), &estimate(200, 0), 0);
        store.save(&path).expect("write");
        let reloaded = CalibrationStore::load(&path);
        assert_eq!(reloaded.factors("k").compute_scale, 2.0);

        std::fs::write(&path, "{not json").unwrap();
        assert!(CalibrationStore::load(&path).entries.is_empty());

        std::fs::write(&path, r#"{"version":99,"entries":{}}"#).unwrap();
        assert!(CalibrationStore::load(&path).entries.is_empty());

        // A hand-edited nonsense factor is ignored, not applied.
        std::fs::write(
            &path,
            r#"{"version":2,"entries":{"k":{"compute_scale":-3.0,"scratch_scale":0.1}}}"#,
        )
        .unwrap();
        assert_eq!(
            CalibrationStore::load(&path).factors("k"),
            Calibration::default()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_key_changes_with_profile_shape_and_graph_class() {
        let model = ModelInfo {
            n_layer: 24,
            n_embd: 1024,
            n_ff: 4096,
            n_head: 16,
            n_embd_k_gqa: 512,
            n_embd_v_gqa: 512,
            n_vocab: 151_936,
            dominant_weight_type: "Q4_K".into(),
            ..Default::default()
        };
        let mut profile = ExecutionProfile {
            schema_version: retrograd_core::EXECUTION_PROFILE_VERSION,
            engine_fingerprint: "engine-a".to_string(),
            kernel_catalog_fingerprint: "catalog-a".to_string(),
            device: retrograd_core::DeviceProfile {
                stable_id: "metal:0".to_string(),
                backend: "metal".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let source = r#"
            [run]
            algorithm = "sft"
            [model]
            path = "model.gguf"
            [output]
            path = "adapter.gguf"
            [lora]
            [training]
            ctx = 1024
            micro_batch = 128
            [sft]
            data = "data.txt"
        "#;
        let document: retrograd_config::ConfigDocument = toml::from_str(source).unwrap();
        let mut config = retrograd_config::build(document, Path::new(".")).unwrap();

        let key = calibration_key(&profile, &model, &config, None);
        assert!(
            key.starts_with(&format!("v{COST_MODEL_VERSION}/metal:")),
            "{key}"
        );
        assert_eq!(calibration_key(&profile, &model, &config, None), key);

        config.training.n_ubatch /= 2;
        assert_ne!(calibration_key(&profile, &model, &config, None), key);
        config.training.n_ubatch *= 2;
        profile.kernel_catalog_fingerprint = "catalog-b".to_string();
        assert_ne!(calibration_key(&profile, &model, &config, None), key);
    }

    /// An adapter and two base selections of the same model and shape must all
    /// key apart, or one would borrow another's correction factor.
    #[test]
    fn a_base_run_and_an_adapter_run_of_the_same_shape_key_apart() {
        let model = ModelInfo {
            n_layer: 24,
            n_embd: 1024,
            n_ff: 4096,
            n_head: 16,
            n_embd_k_gqa: 512,
            n_embd_v_gqa: 512,
            n_vocab: 151_936,
            dominant_weight_type: "F16".into(),
            ..Default::default()
        };
        let profile = ExecutionProfile {
            schema_version: retrograd_core::EXECUTION_PROFILE_VERSION,
            engine_fingerprint: "engine-a".to_string(),
            kernel_catalog_fingerprint: "catalog-a".to_string(),
            device: retrograd_core::DeviceProfile {
                stable_id: "metal:0".to_string(),
                backend: "metal".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let document = |policy: &str, sections: &str| -> RunConfig {
            let source = format!(
                r#"
                [run]
                algorithm = "sft"
                [model]
                path = "model.gguf"
                [output]
                path = "out.gguf"
                [training]
                ctx = 1024
                micro_batch = 128
                trainable = "{policy}"
                [sft]
                data = "data.txt"
                {sections}
                "#
            );
            let document: retrograd_config::ConfigDocument = toml::from_str(&source).unwrap();
            retrograd_config::build(document, Path::new(".")).unwrap()
        };

        let adapter = document("lora", "[lora]");
        let base = document("partial", "[trainable]\n                norms = true");
        assert_eq!(base.training.trainable.policy, TrainablePolicy::Partial);

        let low = base_set(TrainablePolicy::Partial, &["blk.0.attn_norm.weight"]);
        let high = base_set(TrainablePolicy::Partial, &["blk.11.attn_norm.weight"]);

        let adapter_key = calibration_key(&profile, &model, &adapter, None);
        let low_key = calibration_key(&profile, &model, &base, Some(&low));
        let high_key = calibration_key(&profile, &model, &base, Some(&high));

        assert_ne!(adapter_key, low_key);
        assert_ne!(low_key, high_key);
        assert_ne!(adapter_key, high_key);
        // Deterministic, like every other key.
        assert_eq!(
            calibration_key(&profile, &model, &base, Some(&low)),
            low_key
        );
        // A base document with no set is a third key, not the adapter's.
        assert!(
            ![adapter_key, low_key, high_key]
                .contains(&calibration_key(&profile, &model, &base, None))
        );
    }

    fn base_set(policy: TrainablePolicy, names: &[&str]) -> TrainableSet {
        TrainableSet {
            policy,
            entries: names
                .iter()
                .map(|name| retrograd_core::TrainableEntry {
                    name: (*name).to_string(),
                    role: retrograd_core::TensorRole::Base,
                    ne: [1024, 1, 1, 1],
                    dtype: retrograd_core::TensorDtype::F32,
                    n_elements: 1024,
                    n_bytes: 4096,
                    storage_id: 0,
                })
                .collect(),
            exclusions: Vec::new(),
        }
    }

    fn base_training(policy: TrainablePolicy) -> retrograd_core::TrainableRunConfig {
        retrograd_core::TrainableRunConfig {
            policy,
            selector: retrograd_core::TrainableSelector {
                norms: true,
                ..Default::default()
            },
            optimizer: retrograd_core::OptimizerKind::AdamW,
        }
    }

    /// An adapter run has no trainable class, and the field is skipped rather
    /// than serialized as null, keeping pre-existing keys valid.
    #[test]
    fn an_adapter_key_carries_no_trainable_class_and_is_unchanged_by_one() {
        let lora = retrograd_core::TrainableRunConfig::default();
        assert_eq!(lora.policy, TrainablePolicy::Lora);
        // The policy decides, not the set the caller found.
        let set = base_set(TrainablePolicy::Partial, &["blk.3.attn_norm.weight"]);
        assert_eq!(CalibrationTrainableClass::of(&lora, Some(&set)), None);

        let encoded = serde_json::to_string(&CalibrationKey {
            cost_model_version: COST_MODEL_VERSION,
            engine_fingerprint: String::new(),
            kernel_catalog_fingerprint: String::new(),
            profile_fingerprint: String::new(),
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            model: CalibrationModelClass {
                architecture: "qwen2".to_string(),
                layers: 4,
                embedding: 64,
                feed_forward: 128,
                heads: 4,
                kv_heads: 2,
                kv_width: 64,
                vocabulary: 260,
                weight_dtype: "F32".to_string(),
            },
            shape: CalibrationShapeClass {
                n_ctx: 32,
                n_batch: 32,
                n_ubatch: 16,
                generation_concurrency: 1,
                generation_batch: 1,
                packed_fanout: "auto".to_string(),
            },
            graph: CalibrationGraphClass {
                algorithm: "sft".to_string(),
                chunked_cross_entropy: false,
                chunked_ce_tiles: 0,
                chunked_ce_seq_chunk: 0,
                gradient_checkpointing: false,
                checkpoint_every_n_layers: 0,
                checkpoint_dtype: "f32".to_string(),
                fast_generation_context: false,
                kv_dtype: "f16".to_string(),
                require_gpu_resident: false,
            },
            trainable: None,
        })
        .unwrap();
        assert!(!encoded.contains("trainable"), "{encoded}");
    }

    /// Two base selections of the same model build backward graphs of
    /// different heights, so they must not share a factor.
    #[test]
    fn a_base_selection_keys_on_the_set_it_resolved() {
        let training = base_training(TrainablePolicy::Partial);
        let low = base_set(TrainablePolicy::Partial, &["blk.0.attn_norm.weight"]);
        let high = base_set(TrainablePolicy::Partial, &["blk.11.attn_norm.weight"]);

        let of = |set: Option<&TrainableSet>| CalibrationTrainableClass::of(&training, set);
        let low_class = of(Some(&low)).expect("a base policy has a class");
        let high_class = of(Some(&high)).expect("a base policy has a class");
        assert_eq!(low_class.lowest_trainable_layer, Some(0));
        assert_eq!(high_class.lowest_trainable_layer, Some(11));
        assert_ne!(low_class.manifest_digest, high_class.manifest_digest);

        // The digest covers the set's content, not listing order.
        let again = base_set(TrainablePolicy::Partial, &["blk.0.attn_norm.weight"]);
        assert_eq!(of(Some(&again)), Some(low_class.clone()));

        // No resolved set is its own case, not the adapter one.
        let unresolved = of(None).expect("a base policy has a class");
        assert_eq!(unresolved.manifest_digest, None);
        assert_ne!(unresolved, low_class);

        // The optimizer is part of the key: different optimizers keep
        // different amounts of state.
        let mut sgd = training.clone();
        sgd.optimizer = retrograd_core::OptimizerKind::Sgd;
        assert_ne!(
            CalibrationTrainableClass::of(&sgd, Some(&low)),
            Some(low_class)
        );
    }
}
