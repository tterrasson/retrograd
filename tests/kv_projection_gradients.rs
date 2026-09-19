//! The K and V projections are written into the KV cache with ggml_set_rows().
//! If the attention reads the cache buffer through a plain view that merely
//! aliases that store instead of depending on it, the backward pass finds no
//! route from the attention back to k_cur/v_cur, and a LoRA targeting attn_k or
//! attn_v trains as a silent no-op: finite loss, moving metrics, and AdamW
//! momenta that stay exactly zero forever.
//!
//! `llama_context_params::kv_differentiable` restores that route. These tests
//! pin it down through the observable that cannot lie -- the momenta -- because
//! the loss curve looks perfectly healthy either way.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{CheckpointMetadata, Device, KvDtype, LoraConfig, TargetSet, TrainConfig, Trainer};

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
);

/// Every projection in one attention block, so a single run separates the ones
/// that reach the optimizer from the ones that do not. The block index is
/// resolved from the loaded model rather than pinned to zero: the CPU fixture
/// is a hybrid `lfm2`, whose block 0 is a shortconv block with no attention
/// projection at all.
const FAMILIES: [&str; 4] = ["attn_q", "attn_k", "attn_v", "attn_output"];

fn config(device: Device, kv_dtype: KvDtype) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-3,
        device,
        kv_dtype,
        ..TrainConfig::default()
    }
}

fn lora(trainer: &Trainer) -> LoraConfig {
    let targets = common::block_targets(trainer, &FAMILIES)
        .expect("the model has a block carrying every attention projection");
    let mut config = LoraConfig::qv(2, 4.0);
    config.seed = 7;
    config.targets = TargetSet::Patterns(targets);
    config
}

fn metadata(model: &Path) -> CheckpointMetadata {
    CheckpointMetadata {
        checkpoint_id: "step-000000000001".into(),
        algorithm: "sft".into(),
        trajectory_signature: "test-kv-grad-v1".into(),
        resume_boundary: "epoch".into(),
        scheduler_kind: "constant".into(),
        warmup_steps: 0,
        progress: checkpoint::Progress {
            version: checkpoint::FORMAT_VERSION,
            epoch: 1,
            global_step: 1,
            cursor: 0,
            algorithm: "sft".into(),
            phase: "train".into(),
            best_eval: None,
            stale_evaluations: 0,
            kl_multiplier: None,
        },
        dataset: checkpoint::Dataset {
            version: checkpoint::FORMAT_VERSION,
            path: "inline".into(),
            fingerprint: checkpoint::fingerprint(TRAIN_TEXT.as_bytes()),
            examples: 1,
            row_width: 32,
            format: "text".into(),
            permutation: Vec::new(),
            cursor: 0,
        },
        seeds: BTreeMap::from([("sampling".to_string(), 7_u64)]),
        artifacts: BTreeMap::new(),
        model_path: model.to_path_buf(),
    }
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("retrograd-kvgrad-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

/// Trains one step on every attention projection and returns each LoRA tensor
/// paired with the total magnitude of its AdamW second moment. A tensor that
/// never received a gradient has `v` identically zero.
fn second_moment_magnitudes(
    model: &Path,
    device: Device,
    kv_dtype: KvDtype,
    scratch_name: &str,
) -> Vec<(String, f64)> {
    let root = scratch(scratch_name);
    let mut trainer = Trainer::new(model, config(device, kv_dtype)).expect("load trainer");
    if kv_dtype == KvDtype::F16 {
        let report = trainer.backend_report().expect("F16 KV backend report");
        assert!(report.contains("training_kv_dtype: F16"), "{report}");
        assert!(report.contains("training_kv_f16: supported"), "{report}");
    }
    let lora = lora(&trainer);
    trainer.create_lora(&lora).expect("create lora");
    // Enough tokens for several optimizer steps: B is initialised to zero, so
    // grad_A = B^T. grad is legitimately zero on the very first step and the
    // A momenta only become meaningful from the second one on.
    let tokens = trainer
        .tokenize_text(&TRAIN_TEXT.repeat(32))
        .expect("tokenize");
    trainer.train_tokens(&tokens).expect("training run");
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(model))
        .expect("save checkpoint");
    drop(trainer);

    let saved = Checkpoint::read(&state).expect("read checkpoint");
    assert!(
        saved.optimizer.graph_ready,
        "the run took no optimizer step"
    );
    assert!(
        saved.optimizer.iter > 2,
        "need more than one optimizer step for the A momenta to be meaningful, got iter={}",
        saved.optimizer.iter
    );
    // The second moment, read back through the slot table and streamed out of
    // the state file.
    let mut reader = checkpoint::OptimizerStateReader::open(&state, &saved.optimizer)
        .expect("open optimizer payload");
    let mut staging = vec![0_u8; 1 << 16];
    let mut out: Vec<(String, f64)> = saved
        .optimizer
        .assignment
        .iter()
        .map(|row| {
            let (_, v) = saved
                .optimizer
                .adamw_moments(&row.parameter)
                .expect("an adamw run keeps m and v for every parameter");
            let mut total = 0.0_f64;
            reader
                .stream(v, &mut staging, |_, bytes| {
                    for chunk in bytes.as_chunks::<4>().0 {
                        total += f32::from_le_bytes(*chunk).abs() as f64;
                    }
                    Ok(())
                })
                .expect("stream the second moment");
            (row.parameter.clone(), total)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    let _ = std::fs::remove_dir_all(&root);
    out
}

#[test]
fn every_attention_projection_receives_a_gradient() {
    let _guard = common::serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };

    let moments = second_moment_magnitudes(&model, Device::Cpu, KvDtype::F32, "moments-cpu");
    assert_eq!(
        moments.len(),
        FAMILIES.len() * 2,
        "expected an a/b pair per target, got {moments:?}"
    );

    let dead: Vec<&str> = moments
        .iter()
        .filter(|(_, total)| *total == 0.0)
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        dead.is_empty(),
        "these LoRA tensors trained as no-ops (zero AdamW momenta): {dead:?}\n\
         attn_k/attn_v regressing here means the attention is reading the KV \
         cache buffer instead of the ggml_set_rows() that fills it; check \
         kv_differentiable and llm_graph_context::build_attn"
    );

    // A gradient that is merely non-zero could still be numerical noise, so
    // require K and V to be within a few orders of magnitude of Q rather than
    // vanishingly small.
    let magnitude = |needle: &str| -> f64 {
        moments
            .iter()
            .filter(|(name, _)| name.contains(needle))
            .map(|(_, total)| *total)
            .sum()
    };
    let q = magnitude("attn_q");
    assert!(q > 0.0, "the query projection itself has no gradient");
    for projection in ["attn_k", "attn_v"] {
        let value = magnitude(projection);
        assert!(
            value > q * 1.0e-4,
            "{projection} gradient {value} is negligible next to attn_q {q}"
        );
    }
}

/// Fixed tolerance for F16 KV storage, chosen before comparing the results.
/// The second Adam moment is proportional to squared gradients, so 15% here is
/// stricter than a roughly 8% relative difference in gradient magnitude.
#[cfg(retro_vulkan)]
const F16_KV_MOMENT_REL_TOL: f64 = 0.15;

#[cfg(retro_vulkan)]
#[test]
fn f16_training_kv_preserves_kv_projection_gradients() {
    let _guard = common::serialize_models();
    if !common::gpu_device_present() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let Some(model) = common::vulkan_model_path_if_available() else {
        eprintln!(
            "skipping: no Vulkan test model at {}",
            common::vulkan_model_path().display()
        );
        return;
    };

    let f32 = second_moment_magnitudes(&model, Device::Gpu, KvDtype::F32, "moments-vk-f32");
    let f16 = second_moment_magnitudes(&model, Device::Gpu, KvDtype::F16, "moments-vk-f16");
    assert_eq!(f32.len(), f16.len());
    for ((f32_name, reference), (f16_name, actual)) in f32.iter().zip(&f16) {
        assert_eq!(f32_name, f16_name);
        assert!(
            reference.is_finite() && actual.is_finite(),
            "{f32_name}: {reference} vs {actual}"
        );
        if *reference <= 1.0e-16 {
            assert!(
                *actual <= 1.0e-14,
                "near-zero gradient changed for {f32_name}: F32={reference}, F16={actual}"
            );
        } else {
            let relative = (actual - reference).abs() / reference;
            assert!(
                relative <= F16_KV_MOMENT_REL_TOL,
                "gradient moment parity failed for {f32_name}: F32={reference}, F16={actual}, rel={relative}"
            );
        }
    }

    for projection in ["attn_k", "attn_v"] {
        let total: f64 = f16
            .iter()
            .filter(|(name, _)| name.contains(projection))
            .map(|(_, value)| *value)
            .sum();
        assert!(total > 0.0, "F16 KV made {projection} a silent no-op");
    }
}
