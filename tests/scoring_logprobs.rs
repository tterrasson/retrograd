//! Parity guard for the device gather of behavior log-probabilities.
//!
//! The shared-prefix scorer defines the GRPO behavior policy, so its output is
//! what the initial ratios are measured against. When the active device can run
//! the gather, `log softmax(logits)[target]` is computed inside the decode graph
//! and one float per position comes back instead of a full vocabulary row. That
//! reduction is a device F32 softmax, whereas the host oracle accumulates its
//! exponentials in double - so the two agree closely but not bit for bit, and
//! this file pins how closely.
//!
//! Runs on whatever GPU is registered (Metal, CUDA, Vulkan); skipped without
//! one, since the host path is then the only path and the comparison would be
//! against itself.

mod common;

use common::serialize_models;
use retrograd::{Device, TrainConfig, Trainer};

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 256,
        // Two sequence slots are what lets the scorer branch off the shared
        // prefix instead of rolling it back.
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Gpu,
        ..TrainConfig::default()
    }
}

/// Absolute tolerance on one behavior log-probability. This is a numerical
/// parity bound, not a training tolerance: a GRPO ratio is `exp(new - old)`, so
/// 1e-4 on the log-probability is 1e-4 of relative ratio error against a clip
/// range of 0.2.
const TOLERANCE: f32 = 1.0e-4;

#[test]
fn device_logprob_gather_matches_the_host_oracle() {
    let _guard = serialize_models();
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let report = trainer.backend_report().expect("backend report");
    if !report.contains("cap_device_logprobs: supported") {
        eprintln!("skipping: the active device does not run the logprob gather");
        return;
    }

    let prompt = trainer
        .tokenize_text("The quick brown fox jumps over the lazy dog and keeps going")
        .expect("tokenize");
    // Two completions of the same prompt, so this exercises the branch loop and
    // not just a single sequence.
    let rows = [7usize, 11]
        .map(|extra| {
            let mut tokens = prompt.clone();
            tokens.extend(prompt.iter().cycle().take(extra));
            tokens
        })
        .to_vec();
    let inputs = rows
        .iter()
        .map(|tokens| (tokens.as_slice(), prompt.len()))
        .collect::<Vec<_>>();

    let before = trainer.scoring_stats().expect("scoring stats");
    let device = trainer
        .score_token_suffix_batch(&inputs)
        .expect("device-gathered scores");
    let stats = trainer
        .scoring_stats()
        .expect("scoring stats")
        .delta_since(before);
    // Without this the test would still pass if the gather silently never ran.
    assert_eq!(
        stats.device_logprob_positions, stats.scored_positions,
        "the device gather did not cover every scored position"
    );

    let _env = common::EnvGuard::set("RETRO_DEVICE_LOGPROBS", "0");
    let host = trainer
        .score_token_suffix_batch(&inputs)
        .expect("host-reduced scores");

    let mut worst = 0.0_f32;
    for (device_row, host_row) in device.iter().zip(&host) {
        assert_eq!(device_row.len(), host_row.len());
        for (gathered, reduced) in device_row.iter().zip(host_row) {
            assert!(
                gathered.is_finite(),
                "device gather produced {gathered} against oracle {reduced}"
            );
            worst = worst.max((gathered - reduced).abs());
        }
    }
    eprintln!("worst |device - host| log-probability gap: {worst:e}");
    assert!(
        worst <= TOLERANCE,
        "device gather drifts from the host oracle by {worst:e} (> {TOLERANCE:e})"
    );
}

/// The single-sequence scorer takes the same gather. It is the one GRPO calls
/// most: once per rollout per optimizer step for the current-policy ratio, and
/// once per rollout for the fixed reference - against once per update for the
/// shared-prefix behavior pass covered above.
#[test]
fn single_sequence_suffix_scoring_takes_the_same_device_gather() {
    let _guard = serialize_models();
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let report = trainer.backend_report().expect("backend report");
    if !report.contains("cap_device_logprobs: supported") {
        eprintln!("skipping: the active device does not run the logprob gather");
        return;
    }

    let prompt = trainer
        .tokenize_text("The quick brown fox jumps over the lazy dog and keeps going")
        .expect("tokenize");
    let mut tokens = prompt.clone();
    tokens.extend(prompt.iter().cycle().take(9));

    let before = trainer.scoring_stats().expect("scoring stats");
    let device = trainer
        .score_token_suffix(&tokens, prompt.len())
        .expect("device-gathered scores");
    let stats = trainer
        .scoring_stats()
        .expect("scoring stats")
        .delta_since(before);
    assert_eq!(device.len(), tokens.len() - prompt.len());
    assert_eq!(
        stats.device_logprob_positions, stats.scored_positions,
        "the single-sequence scorer did not gather on the device"
    );
    assert_eq!(
        stats.scored_positions as usize,
        device.len(),
        "every scored position must be counted once"
    );

    let _env = common::EnvGuard::set("RETRO_DEVICE_LOGPROBS", "0");
    let host = trainer
        .score_token_suffix(&tokens, prompt.len())
        .expect("host-reduced scores");
    let mut worst = 0.0_f32;
    for (gathered, reduced) in device.iter().zip(&host) {
        assert!(gathered.is_finite(), "device gather produced {gathered}");
        worst = worst.max((gathered - reduced).abs());
    }
    eprintln!("worst |device - host| gap on the suffix path: {worst:e}");
    assert!(
        worst <= TOLERANCE,
        "the suffix-path gather drifts from the host oracle by {worst:e} (> {TOLERANCE:e})"
    );
}
