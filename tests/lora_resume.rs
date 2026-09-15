//! Resuming training from an exported LoRA adapter: train + save, reload the
//! GGUF in a fresh trainer, and confirm the loaded adapter trains again with
//! the base model still frozen, then exports again.

mod common;

use retrograd::{Device, LoraConfig, TargetSet, TrainConfig, Trainer};

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
    "The five boxing wizards jump quickly. ",
    "Jackdaws love my big sphinx of quartz. ",
);

fn config(device: Device) -> TrainConfig {
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

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

/// Trains a fresh adapter for one short epoch and exports it.
fn train_and_export(device: Device, out_path: &std::path::Path) {
    let model = common::model_path();
    let mut trainer = Trainer::new(&model, config(device)).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    trainer.train_tokens(&tokens).expect("first training run");
    trainer.save_lora(out_path).expect("save lora");
}

fn resume_and_train(device: Device, adapter_path: &std::path::Path) {
    let model = common::model_path();
    let mut trainer = Trainer::new(&model, config(device)).expect("load trainer");
    trainer.load_lora(adapter_path).expect("load lora");

    let before = trainer.describe_lora().expect("describe loaded lora");
    assert!(before.contains("loaded_from_file: true"), "{before}");

    // The preflight promotes the loaded adapter to trainable and must succeed.
    let preflight = trainer.train_preflight().expect("train preflight");
    assert!(
        preflight.contains("missing_gradient_rules: 0"),
        "[{device:?}] {preflight}"
    );

    // Promotion flags every A/B pair and leaves the base model frozen.
    let after = trainer.describe_lora().expect("describe promoted lora");
    assert!(after.contains("base_trainable_tensors: 0"), "{after}");
    assert!(after.contains("trainable=true"), "{after}");
    assert!(!after.contains("trainable=false"), "{after}");

    // The resumed adapter must actually learn: its scores change and the
    // second export still round-trips through llama_adapter_lora_init().
    let probe = trainer.tokenize_text(TRAIN_TEXT).expect("tokenize probe");
    let scores_before = trainer.score_tokens(&probe).expect("score before");

    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    let metrics = trainer.train_tokens(&tokens).expect("resumed training run");
    assert!(
        metrics.train_loss.is_finite(),
        "[{device:?}] resumed train_loss is not finite"
    );

    let scores_after = trainer.score_tokens(&probe).expect("score after");
    assert!(
        scores_before
            .iter()
            .zip(&scores_after)
            .any(|(before, after)| before != after),
        "[{device:?}] resumed training did not change the adapter"
    );

    let resumed_out = std::env::temp_dir().join(format!("retrograd-resume-out-{device:?}.gguf"));
    trainer.save_lora(&resumed_out).expect("save resumed lora");
    let _ = std::fs::remove_file(&resumed_out);
}

#[test]
fn cpu_resume_trains_a_loaded_adapter() {
    let _guard = common::serialize_models();
    if common::model_path_if_available().is_none() {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    }
    let adapter = std::env::temp_dir().join("retrograd-resume-cpu.gguf");
    train_and_export(Device::Cpu, &adapter);
    resume_and_train(Device::Cpu, &adapter);
    let _ = std::fs::remove_file(&adapter);
}

#[cfg(retro_metal)]
#[test]
fn metal_resume_trains_a_loaded_adapter() {
    let _guard = common::serialize_models();
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    if common::model_path_if_available().is_none() {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    }
    let adapter = std::env::temp_dir().join("retrograd-resume-metal.gguf");
    train_and_export(Device::Gpu, &adapter);
    resume_and_train(Device::Gpu, &adapter);
    let _ = std::fs::remove_file(&adapter);
}

#[cfg(retro_cuda)]
#[test]
fn cuda_resume_trains_a_loaded_adapter() {
    let _guard = common::serialize_models();
    if !common::cuda_registered()
        || !retrograd::gpu_runtime_available()
        || common::model_path_if_available().is_none()
    {
        eprintln!("skipping: CUDA device or local model unavailable");
        return;
    }
    let adapter = std::env::temp_dir().join("retrograd-resume-cuda.gguf");
    train_and_export(Device::Gpu, &adapter);
    resume_and_train(Device::Gpu, &adapter);
    let _ = std::fs::remove_file(&adapter);
}

/// The optimizer captures the trainable tensors once, so swapping the adapter
/// afterwards must still be refused.
#[test]
fn loading_an_adapter_after_training_is_refused() {
    let _guard = common::serialize_models();
    if common::model_path_if_available().is_none() {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    }
    let adapter = std::env::temp_dir().join("retrograd-resume-guard.gguf");
    train_and_export(Device::Cpu, &adapter);

    let model = common::model_path();
    let mut trainer = Trainer::new(&model, config(Device::Cpu)).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    trainer.train_tokens(&tokens).expect("train");

    let error = trainer
        .load_lora(&adapter)
        .expect_err("adapter swap after optimizer init must fail");
    assert!(
        error.to_string().contains("optimizer"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_file(&adapter);
}
