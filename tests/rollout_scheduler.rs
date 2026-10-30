//! The rollout driver over the step-wise scheduler, on the CPU fixture: in
//! `Lockstep` mode (the default admission), a heterogeneous wave of
//! (prompt, params) rows through the driver must reproduce
//! `Trainer::generate_sampled_continuous` bit for bit - tokens and forced
//! mask - and a shared-prompt group of n members must reproduce
//! `Trainer::generate_tokens_batch` bit for bit. Skipped when no local model
//! exists (override with `RETRO_CPU_FIXTURE`).

mod common;

use common::serialize_models;
use retrograd::training::rollout::driver::run_scheduled_rollout_groups;
use retrograd::{Device, SamplingParams, TrainConfig, Trainer};

/// The trainer config the parity tests run with: four sequence slots, so the
/// dedicated generation context exists with a known geometry.
fn train_config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn params(seed: u32, max_new_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens,
        seed,
    }
}

#[test]
fn lockstep_wave_matches_generate_tokens_continuous_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, train_config()).expect("load trainer");

    // Two rows share `first` so the wave exercises the merged prefill + seq_cp
    // group, and the prompts differ so the canonical order is not the caller
    // order. Submitted in a scrambled order on purpose.
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    let wave: Vec<(Vec<i32>, SamplingParams)> = vec![
        (second.clone(), params(50, 6)),
        (first.clone(), params(51, 6)),
        (first.clone(), params(52, 6)),
    ];
    let rows: Vec<(&[i32], SamplingParams)> = wave
        .iter()
        .map(|(prompt, sampling)| (prompt.as_slice(), *sampling))
        .collect();
    let reference = trainer
        .generate_sampled_continuous(&rows)
        .expect("continuous reference");

    // One single-member group per wave row: the driver's Lockstep path submits
    // exactly these requests as one wave.
    let groups: Vec<(&[i32], Vec<SamplingParams>)> = rows
        .iter()
        .map(|(prompt, sampling)| (*prompt, vec![*sampling]))
        .collect();
    let (out, stats) = run_scheduled_rollout_groups(&mut trainer, &train_config(), &groups, 4)
        .expect("driver run");
    assert_eq!(out.len(), wave.len());
    for (index, group) in out.iter().enumerate() {
        assert_eq!(group.len(), 1);
        assert_eq!(
            group[0].tokens, reference[index].tokens,
            "row {index}: tokens bit for bit"
        );
        assert_eq!(
            group[0].forced, reference[index].forced,
            "row {index}: forced mask bit for bit"
        );
    }
    // The driver sampled the scheduler on real steps.
    assert!(stats.steps > 0);
    assert!(stats.slot_occupancy_mean(4) > 0.0);
    assert!(stats.slot_occupancy_mean(4) <= 1.0);
}

#[test]
fn shared_prompt_group_matches_generate_tokens_batch_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, train_config()).expect("load trainer");

    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let sampling: Vec<SamplingParams> = vec![params(61, 5), params(62, 5), params(63, 5)];
    let reference = trainer
        .generate_tokens_batch(&prompt, &sampling)
        .expect("batch reference");

    // One group with three members: the scheduler admits it as one prefill
    // plus seq_cp, the generate_tokens_batch shape.
    let groups: Vec<(&[i32], Vec<SamplingParams>)> = vec![(prompt.as_slice(), sampling.clone())];
    let (out, _stats) = run_scheduled_rollout_groups(&mut trainer, &train_config(), &groups, 4)
        .expect("driver run");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 3);
    for (member, completion) in out[0].iter().enumerate() {
        assert_eq!(
            completion.tokens, reference[member],
            "member {member}: tokens bit for bit"
        );
        assert!(
            completion.forced.iter().all(|&forced| !forced),
            "member {member}: single-turn sampling forces nothing"
        );
    }
}
