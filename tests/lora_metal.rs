//! Verifies that a trainable LoRA adapter is allocated on the Metal buffer
//! when the model is GPU-offloaded, and that a short training step updates the
//! LoRA tensors (and only the LoRA tensors) while running with Metal active.

mod common;

// Every test in this binary is `#[cfg(retro_metal)]`, and so is everything they
// use. A helper compiled without Metal is not dead code, it is off-topic code:
// gating it here is what keeps a CPU-only build free of warnings
// without a blanket `allow(dead_code)` that would also hide a real one.
#[cfg(retro_metal)]
use retrograd::Trainer;
#[cfg(retro_metal)]
use retrograd::{Device, LoraConfig, TargetSet, TrainConfig};

#[cfg(retro_metal)]
fn small_config(device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-3,
        device,
        ..TrainConfig::default()
    }
}

/// A short paragraph that tokenizes to well over n_ctx+1 tokens.
#[cfg(retro_metal)]
const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
    "The five boxing wizards jump quickly. ",
    "Jackdaws love my big sphinx of quartz. ",
);

#[cfg(retro_metal)]
fn single_layer_lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    // Keep the graph tiny: one attention projection is enough to prove the path.
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

#[cfg(retro_metal)]
#[test]
fn lora_tensors_are_allocated_on_metal() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut trainer = Trainer::new(&model, small_config(Device::Gpu)).expect("load gpu trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");

    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- GPU LoRA report ---\n{report}");
    assert!(
        common::section_has_metal(&report, "lora_tensors_by_buffer"),
        "LoRA tensors must be allocated on a Metal buffer when GPU is active:\n{report}"
    );
}

#[cfg(retro_metal)]
#[test]
fn short_training_step_updates_lora_on_metal() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut trainer = Trainer::new(&model, small_config(Device::Gpu)).expect("load gpu trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");

    // llama rounds the context up (to 256 for this model), and the dataset needs
    // more than n_ctx+1 tokens, so repeat the paragraph to comfortably exceed it.
    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    assert!(
        tokens.len() > 257,
        "need > n_ctx+1 tokens, got {}",
        tokens.len()
    );

    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one epoch on metal");
    eprintln!(
        "metal train: train_loss={} eval_loss={} tok/s={}",
        metrics.train_loss, metrics.eval_loss, metrics.tokens_per_second
    );
    assert!(metrics.train_loss.is_finite(), "train_loss must be finite");
    assert!(metrics.eval_loss.is_finite(), "eval_loss must be finite");
}

/// The two halves of the fused-CE fallback diagnostic must agree: `cap_fused_sparse_ce`
/// answers "would the fused CE run on this device", the preflight counts what
/// actually falls back. An "unavailable" capability with a fused-CE graph
/// therefore *has* to show up as at least one CPU fallback node - that is the
/// silent degradation the field exists to name. The converse is deliberately
/// not asserted: other ops may fall back for unrelated reasons.
#[cfg(retro_metal)]
#[test]
fn fused_ce_capability_agrees_with_the_metal_preflight() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    let mut trainer = Trainer::new(&model, config).expect("load fused-CE Metal trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");

    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- Metal fused-CE report ---\n{report}");
    // The fixture is quantized, so both base-weight lines are meaningful here
    // rather than degenerate. Which quantization it uses is the fixture's
    // business -- pinning a specific family made this assert fail the day the
    // fixture moved from Q8_0 to Q4_K_M, without anything being wrong.
    assert!(report.contains("base_weight_types:"), "{report}");
    let quantized: Vec<&str> = common::section_lines(&report, "base_weight_types")
        .into_iter()
        .filter(|line| line.starts_with('q'))
        .collect();
    assert!(
        !quantized.is_empty(),
        "the fixture must carry quantized weights for this path to mean anything:\n{report}"
    );
    assert!(
        report.contains("quantized_backward_path: native"),
        "Metal dequantizes inside the out_prod kernel, so it owns no scratch:\n{report}"
    );
    assert!(
        report.contains("chunked_cross_entropy: enabled"),
        "{report}"
    );

    let preflight = trainer.train_preflight().expect("train preflight");
    eprintln!("--- Metal fused-CE preflight ---\n{preflight}");
    let fallbacks: u32 = preflight
        .lines()
        .find_map(|line| line.trim().strip_prefix("active_device_fallback_nodes: "))
        .expect("preflight reports the active device's fallback count")
        .trim()
        .parse()
        .expect("fallback count is a number");

    if report.contains("cap_fused_sparse_ce: unavailable") {
        assert!(
            fallbacks > 0,
            "an unavailable fused CE must appear as a CPU fallback:\n{preflight}"
        );
        assert!(
            report.contains("chunked_cross_entropy_status: cpu_fallback"),
            "{report}"
        );
    }
}

/// `require_gpu_resident` turns exactly the fallbacks the preflight already
/// counts into a hard error. Asserted as an implication rather than a fixed
/// expectation so the test keeps its meaning as Metal's op coverage grows: the
/// day nothing falls back, the guard must stop firing, not start lying.
#[cfg(retro_metal)]
#[test]
fn require_gpu_resident_rejects_a_metal_cpu_fallback() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut permissive = small_config(Device::Gpu);
    permissive.chunked_cross_entropy = true;
    let mut trainer = Trainer::new(&model, permissive).expect("load permissive trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");
    let preflight = trainer.train_preflight().expect("train preflight");
    let has_fallback = !preflight.contains("active_device_fallback_nodes: 0");
    drop(trainer);

    let mut strict = small_config(Device::Gpu);
    strict.chunked_cross_entropy = true;
    strict.require_gpu_resident = true;
    let mut trainer = Trainer::new(&model, strict).expect("load strict trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");
    let result = trainer.train_preflight();

    if has_fallback {
        let error = result.expect_err("require_gpu_resident must refuse a CPU fallback");
        let message = error.to_string();
        assert!(
            message.contains("require_gpu_resident") && message.contains("fall back to the CPU"),
            "{message}"
        );
    } else {
        result.expect("nothing falls back, so the guard must stay quiet");
    }
}

/// A real training step with the fused cross-entropy, pinned GPU-resident.
///
/// The probes in `tests/fused_ce.rs` prove the Metal kernels match the CPU
/// oracle on loss and `grad_h`; this proves the training graph actually
/// dispatches them - with `require_gpu_resident` set, a single node handed back
/// to the CPU fails the preflight instead of quietly costing a transfer of the
/// hidden states and of the whole projection head per token chunk.
#[cfg(retro_metal)]
#[test]
fn fused_cross_entropy_trains_gpu_resident_on_metal() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    config.require_gpu_resident = true;
    let mut trainer = Trainer::new(&model, config).expect("load fused-CE Metal trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");

    let report = trainer.backend_report().expect("backend report");
    assert!(
        report.contains("cap_fused_sparse_ce: supported"),
        "{report}"
    );

    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one epoch with the fused CE on Metal");
    eprintln!(
        "metal fused-CE train: train_loss={} eval_loss={}",
        metrics.train_loss, metrics.eval_loss
    );
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

/// `kv_dtype = "f16"` halves the KV cache only if the *backward* Flash Attention
/// kernel is available, since that is what keeps the cache differentiable. This
/// asserts the Metal port of FLASH_ATTN_BACK lights the capability up rather than
/// silently falling back to the materialized F32 attention graph, and that a real
/// LoRA step through the F16 KV path still trains with a finite loss. Mirrors
/// `f16_training_kv_is_effective_and_differentiable_on_cuda`.
#[cfg(retro_metal)]
#[test]
fn f16_training_kv_is_effective_and_differentiable_on_metal() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let mut config = small_config(Device::Gpu);
    config.kv_dtype = retrograd::KvDtype::F16;
    let mut trainer = Trainer::new(&model, config).expect("load F16-KV Metal trainer");

    let report = trainer.backend_report().expect("F16-KV backend report");
    eprintln!("--- Metal F16-KV report ---\n{report}");
    assert!(
        report.contains("cap_flash_attn_back: supported"),
        "{report}"
    );
    assert!(report.contains("training_kv_dtype: F16"), "{report}");
    assert!(report.contains("training_kv_f16: supported"), "{report}");

    // K/V is where the F16 cache actually changes the gradient path. The block
    // index comes from the model: a hybrid architecture has no attention
    // projection in block 0.
    let targets = common::block_targets(&trainer, &["attn_k", "attn_v"])
        .expect("the model has an attention block");
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(targets);
    trainer.create_lora(&lora).expect("create K/V LoRA");

    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one epoch on Metal with F16 KV");
    eprintln!(
        "metal f16-kv train: train_loss={} eval_loss={}",
        metrics.train_loss, metrics.eval_loss
    );
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
}

/// Wall-clock smoke benchmark for the fused cross-entropy on Metal, not a
/// correctness test: `#[ignore]`d so no lane pays for it. Run with
/// `cargo test --release --test lora_metal -- --ignored --nocapture fused_ce_metal_timing`.
///
/// Compare it against a build where `supports_op` refuses `FUSED_SPARSE_CE` to
/// measure what the CPU fallback costs.
#[cfg(retro_metal)]
#[test]
#[ignore]
fn fused_ce_metal_timing() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local model");
        return;
    };

    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create lora");
    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");

    let start = std::time::Instant::now();
    let metrics = trainer.train_tokens(&tokens).expect("train");
    eprintln!(
        "metal fused-CE epoch: {:.3} s, {:.1} tok/s, loss={}",
        start.elapsed().as_secs_f64(),
        metrics.tokens_per_second,
        metrics.train_loss
    );
}
