//! End-to-end short training on CPU and Metal, including the
//! save/reload path. Confirms the GPU path is functionally equivalent to the
//! CPU path (base frozen, LoRA updated, adapter re-loadable) and that the two
//! losses stay in the same ballpark.

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

struct Outcome {
    train_loss: f32,
    eval_loss: f32,
}

/// Runs one training epoch on the given device and saves+reloads the adapter.
fn run_once(device: Device, out_path: &std::path::Path) -> Outcome {
    let model = common::model_path();
    let mut trainer = Trainer::new(&model, config(device)).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");

    // An explicit preflight must leave the optimizer in a state where the
    // subsequent training run still works (it is also run implicitly there).
    let preflight = trainer.train_preflight().expect("train preflight");
    assert!(
        preflight.contains("missing_gradient_rules: 0"),
        "[{device:?}] {preflight}"
    );

    let text = TRAIN_TEXT.repeat(12);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    let metrics = trainer.train_tokens(&tokens).expect("train");

    // Save writes a standalone GGUF and reloads it via llama_adapter_lora_init(),
    // so a successful save proves the exported adapter is valid on this backend.
    trainer.save_lora(out_path).expect("save lora");
    assert!(out_path.exists(), "[{device:?}] adapter file not written");

    Outcome {
        train_loss: metrics.train_loss,
        eval_loss: metrics.eval_loss,
    }
}

#[test]
fn cpu_end_to_end() {
    if common::model_path_if_available().is_none() {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    }
    let out = std::env::temp_dir().join("retrograd-parity-cpu.gguf");
    let cpu = run_once(Device::Cpu, &out);
    eprintln!(
        "CPU: train_loss={} eval_loss={}",
        cpu.train_loss, cpu.eval_loss
    );
    assert!(cpu.train_loss.is_finite() && cpu.eval_loss.is_finite());
    let _ = std::fs::remove_file(&out);
}

#[cfg(retro_metal)]
#[test]
fn metal_matches_cpu_end_to_end() {
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

    let cpu_out = std::env::temp_dir().join("retrograd-parity-cpu2.gguf");
    let gpu_out = std::env::temp_dir().join("retrograd-parity-metal.gguf");
    let cpu = run_once(Device::Cpu, &cpu_out);
    let gpu = run_once(Device::Gpu, &gpu_out);

    eprintln!(
        "CPU:   train_loss={} eval_loss={}\nMetal: train_loss={} eval_loss={}",
        cpu.train_loss, cpu.eval_loss, gpu.train_loss, gpu.eval_loss
    );

    assert!(gpu.train_loss.is_finite() && gpu.eval_loss.is_finite());

    // Same seed and data: the initial forward loss should be close on both
    // backends. Allow a generous tolerance for fp ordering / hybrid execution.
    let rel = (cpu.train_loss - gpu.train_loss).abs() / cpu.train_loss.abs().max(1e-6);
    assert!(
        rel < 0.05,
        "CPU vs Metal train_loss diverged: cpu={} metal={} rel={rel}",
        cpu.train_loss,
        gpu.train_loss
    );

    let _ = std::fs::remove_file(&cpu_out);
    let _ = std::fs::remove_file(&gpu_out);
}

#[cfg(retro_cuda)]
#[test]
fn cuda_matches_cpu_end_to_end() {
    let _guard = common::serialize_models();
    if !common::cuda_registered()
        || !retrograd::gpu_runtime_available()
        || common::model_path_if_available().is_none()
    {
        eprintln!("skipping: CUDA device or local model unavailable");
        return;
    }
    let cpu_out = std::env::temp_dir().join("retrograd-parity-cpu-cuda.gguf");
    let gpu_out = std::env::temp_dir().join("retrograd-parity-cuda.gguf");
    let cpu = run_once(Device::Cpu, &cpu_out);
    let cuda = run_once(Device::Gpu, &gpu_out);
    assert!(cuda.train_loss.is_finite() && cuda.eval_loss.is_finite());
    let rel = (cpu.train_loss - cuda.train_loss).abs() / cpu.train_loss.abs().max(1e-6);
    assert!(
        rel < 0.05,
        "CPU vs CUDA train_loss diverged: cpu={} cuda={} rel={rel}",
        cpu.train_loss,
        cuda.train_loss
    );
    let _ = std::fs::remove_file(&cpu_out);
    let _ = std::fs::remove_file(&gpu_out);
}
