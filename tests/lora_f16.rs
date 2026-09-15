mod common;

use retrograd::{
    Device, LoraConfig, LoraDtype, ProbeInputs, ProbeOp, TargetSet, TrainConfig, Trainer, probe_op,
};

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

#[test]
fn cpu_f16_adamw_matches_the_f32_formula_with_f16_rounding() {
    let weights = [0.5, -1.0, 2.0, -0.25];
    let gradients = [0.25, -0.5, 1.0, -2.0];
    let alpha = 1.0e-2;
    let wd = 0.1;
    let actual = probe_op(
        ProbeOp::OptStepAdamwF16,
        false,
        ProbeInputs::pair([4, 1, 1, 1], &weights, [4, 1, 1, 1], &gradients),
        [alpha, wd],
        4,
    )
    .expect("run CPU F16 AdamW probe");
    for ((actual, weight), gradient) in actual.iter().zip(weights).zip(gradients) {
        let expected = weight * (1.0 - alpha * wd) - alpha * gradient / (gradient.abs() + 1.0e-8);
        assert!(
            (actual - expected).abs() <= 1.0e-3,
            "{actual} vs {expected}"
        );
    }
}

/// The AdamW kernels fold gradient clipping in through `params[7]` instead of
/// materializing `scale * grad`. That is only equivalent if the kernel applies
/// the scale to the gradient *before* the moment update, so scaling in the
/// kernel and pre-scaling the gradient by hand must agree exactly.
#[test]
fn cpu_f16_adamw_folds_the_clipping_scale_into_the_gradient() {
    let gradients = [0.25, -0.5, 1.0, -2.0];
    let weights = [0.5, -1.0, 2.0, -0.25];
    let scale = 0.375_f32;
    let run = |gradients: &[f32], scale: Option<f32>| {
        let scale = scale.map(|value| ([1, 1, 1, 1], vec![value]));
        probe_op(
            ProbeOp::OptStepAdamwF16,
            false,
            ProbeInputs::pair([4, 1, 1, 1], &weights, [4, 1, 1, 1], gradients)
                .with_src2(scale.as_ref().map(|(ne, data)| (*ne, data.as_slice()))),
            [1.0e-2, 0.1],
            4,
        )
        .expect("run CPU F16 AdamW probe")
    };
    let prescaled: Vec<f32> = gradients.iter().map(|value| value * scale).collect();
    assert_eq!(run(&gradients, Some(scale)), run(&prescaled, None));
    // A neutral scale must be a no-op, otherwise the fold silently rescales
    // every unclipped step.
    assert_eq!(run(&gradients, Some(1.0)), run(&gradients, None));
}

/// Chains `steps` single AdamW updates, feeding each result back in and moving
/// the rounding seed on, the way a real run does. Every call resets the moments,
/// so `mh / vh` is exactly `sign(g)` and each step moves the weight by `alpha`.
fn chained_adamw_f16(
    gpu: bool,
    weights: &[f32],
    gradients: &[f32],
    alpha: f32,
    wd: f32,
    steps: u32,
) -> Vec<f32> {
    let n = weights.len();
    let ne = [n as i64, 1, 1, 1];
    let mut current = weights.to_vec();
    for step in 0..steps {
        let scale_and_seed = vec![1.0_f32, step as f32];
        current = probe_op(
            ProbeOp::OptStepAdamwF16,
            gpu,
            ProbeInputs::pair(ne, &current, ne, gradients)
                .with_src2(Some(([2, 1, 1, 1], scale_and_seed.as_slice()))),
            [alpha, wd],
            n,
        )
        .expect("run chained F16 AdamW probe");
    }
    current
}

/// The point of stochastic rounding: an update smaller than half an F16 ULP
/// must still accumulate. Round-to-nearest would discard every one of these
/// steps and leave the weights at exactly 1.0 forever.
#[test]
fn cpu_f16_adamw_accumulates_updates_below_the_f16_resolution() {
    let n = 256;
    let alpha = 1.0e-5;
    let steps = 300;
    let weights = vec![1.0_f32; n];
    let gradients = vec![1.0_f32; n];
    // Half an ULP just below 1.0 is 2^-12; the per-step update is ~50x smaller.
    assert!(
        alpha < 2.0_f32.powi(-12),
        "the test must round away each step"
    );

    let after = chained_adamw_f16(false, &weights, &gradients, alpha, 0.0, steps);
    assert!(
        after.iter().all(|value| value.is_finite()),
        "stochastic rounding must not manufacture inf/NaN"
    );
    assert!(
        after.iter().any(|value| *value != 1.0),
        "every weight is still exactly 1.0: the updates were rounded away"
    );

    // Unbiased means the average lands on the exact arithmetic, not merely
    // somewhere below 1.0. Over 256 elements the spread is ~1e-4.
    let mean = after.iter().sum::<f32>() / n as f32;
    let expected = 1.0 - alpha * steps as f32;
    assert!(
        (mean - expected).abs() < 5.0e-4,
        "mean {mean} drifted from the unbiased {expected}"
    );
}

/// Decoupled weight decay is the other victim of round-to-nearest: `w * keep`
/// moves the weight by far less than an ULP, so the decay never applied.
#[test]
fn cpu_f16_adamw_applies_weight_decay_below_the_f16_resolution() {
    let n = 256;
    let (alpha, wd, steps) = (1.0e-3_f32, 0.05_f32, 300);
    let weights = vec![1.0_f32; n];
    let gradients = vec![0.0_f32; n];
    assert!(
        alpha * wd < 2.0_f32.powi(-12),
        "decay must round away each step"
    );

    let after = chained_adamw_f16(false, &weights, &gradients, alpha, wd, steps);
    let mean = after.iter().sum::<f32>() / n as f32;
    let expected = (1.0 - alpha * wd).powi(steps as i32);
    assert!(
        (mean - expected).abs() < 5.0e-4,
        "mean {mean} does not follow the decay curve {expected}"
    );
}

fn train_config(device: Device) -> TrainConfig {
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

fn f16_lora(rank: u32) -> LoraConfig {
    let mut config = LoraConfig::qv(rank, 2.0 * rank as f32);
    config.seed = 7;
    config.dtype = LoraDtype::F16;
    config.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    config
}

fn create_train_save_reload(device: Device, rank: u32, label: &str) {
    let model = common::model_path();
    let mut trainer = Trainer::new(&model, train_config(device)).expect("load trainer");
    trainer
        .create_lora(&f16_lora(rank))
        .expect("create F16 LoRA");

    // The CLI's verbose backend diagnostic is emitted at this exact lifecycle
    // point, so its F16 fields must already reflect the created adapter.
    let backend = trainer.backend_report().expect("report F16 backend");
    assert!(backend.contains("lora_dtype: F16"), "{backend}");
    assert!(backend.contains("optimizer_f16: supported"), "{backend}");

    let before = trainer.describe_lora().expect("describe F16 LoRA");
    assert!(before.contains("lora_dtype: F16"), "{before}");
    assert!(before.contains("dtype=f16"), "{before}");
    assert!(before.contains("base_trainable_tensors: 0"), "{before}");
    assert!(
        before.contains(&format!("A=32x{rank}")) || before.contains(&format!("x{rank}")),
        "{before}"
    );

    let text = TRAIN_TEXT.repeat(20);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    let metrics = trainer.train_tokens(&tokens).expect("train F16 LoRA");
    assert!(metrics.train_loss.is_finite());
    assert!(metrics.eval_loss.is_finite());

    let output = std::env::temp_dir().join(format!(
        "retrograd-lora-f16-{}-{label}-{rank}.gguf",
        std::process::id()
    ));
    trainer.save_lora(&output).expect("save F16 LoRA");
    drop(trainer);

    let mut reloaded = Trainer::new(&model, train_config(device)).expect("reload trainer");
    reloaded.load_lora(&output).expect("reload F16 LoRA");
    let report = reloaded
        .describe_lora()
        .expect("describe reloaded F16 LoRA");
    assert!(report.contains("loaded_from_file: true"), "{report}");
    assert!(report.contains("lora_dtype: F16"), "{report}");
    reloaded
        .train_preflight()
        .expect("promote reloaded F16 LoRA");
    let _ = std::fs::remove_file(output);
}

#[test]
fn cpu_f16_lora_trains_and_round_trips_for_edge_ranks() {
    let _guard = common::serialize_models();
    if common::model_path_if_available().is_none() {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    }
    create_train_save_reload(Device::Cpu, 1, "cpu");
    create_train_save_reload(Device::Cpu, 32, "cpu");
}

#[cfg(retro_metal)]
#[test]
fn metal_f16_lora_optimizer_stays_supported() {
    let _guard = common::serialize_models();
    if !common::gpu_device_present() || common::model_path_if_available().is_none() {
        eprintln!("skipping: Metal device or local model unavailable");
        return;
    }
    create_train_save_reload(Device::Gpu, 1, "metal");
}

#[cfg(retro_cuda)]
#[test]
fn cuda_f16_lora_trains_and_round_trips() {
    let _guard = common::serialize_models();
    if !common::cuda_registered()
        || !retrograd::gpu_runtime_available()
        || common::model_path_if_available().is_none()
    {
        eprintln!("skipping: CUDA device or local model unavailable");
        return;
    }
    create_train_save_reload(Device::Gpu, 1, "cuda");
}
