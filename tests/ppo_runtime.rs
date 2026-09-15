//! Independent validation of each PPO runtime building block against a real
//! GGUF model: rollout generation, teacher-forced logprob scoring, and the
//! weighted differentiable training step. Skipped when no local model exists
//! (override with `RETRO_TEST_MODEL`).

mod common;

use retrograd::{
    Device, LoraConfig, SamplingParams, TargetSet, TrainConfig, Trainer, WeightedBatch,
};

use common::serialize_models;

const PROMPT: &str = "The quick brown fox";

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 32,
        epochs: 1,
        learning_rate: 1.0e-4,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    // The final ShortConv projection keeps the sign test's gradient path
    // short while still exercising LFM2's recurrent architecture.
    cfg.targets = TargetSet::Patterns(vec!["blk.13.shortconv.out_proj.weight".to_string()]);
    cfg
}

fn sampling(seed: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        max_new_tokens: 8,
        seed,
    }
}

fn trainer() -> Option<Trainer> {
    let model = common::model_path_if_available()?;
    Some(Trainer::new(model, config()).expect("load trainer"))
}

#[test]
fn generation_scoring_and_hidden_states_share_one_loaded_model() {
    let _guard = serialize_models();
    let Some(mut trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let prompt = trainer.tokenize_text(PROMPT).expect("tokenize");

    let first = trainer.generate(&prompt, &sampling(42)).expect("generate");
    let second = trainer.generate(&prompt, &sampling(42)).expect("generate");
    assert_eq!(first.tokens, second.tokens, "same seed, same rollout");
    assert_eq!(first.logprobs, second.logprobs);

    assert!(!first.tokens.is_empty());
    assert!(first.tokens.len() <= 8);
    assert_eq!(first.tokens.len(), first.logprobs.len());
    for &logprob in &first.logprobs {
        assert!(logprob.is_finite() && logprob <= 0.0, "logprob {logprob}");
    }

    // The completion detokenizes (possibly to an empty string if the model
    // immediately emitted a special stop token).
    let text = trainer
        .detokenize(&first.tokens, false)
        .expect("detokenize");
    assert!(text.len() < 512);

    let prompt = trainer.tokenize_text(PROMPT).expect("tokenize");
    let generation = trainer.generate(&prompt, &sampling(7)).expect("generate");

    let mut sequence = prompt.clone();
    sequence.extend_from_slice(&generation.tokens);
    let scored = trainer.score_tokens(&sequence).expect("score");
    assert_eq!(scored.len(), sequence.len() - 1);

    // The teacher-forced path must be deterministic: PPO relies on identical
    // rescoring producing ratio 1 on the first epoch.
    let rescored = trainer.score_tokens(&sequence).expect("score again");
    assert_eq!(scored, rescored, "scoring must be deterministic");

    // Incremental (kv-cached, single-token) rollout decoding and batched
    // re-scoring hit different quantized kernels, so their logprobs drift by
    // up to ~0.1 on this Q8_0 model. PPO recomputes its old logprobs through
    // the batched path precisely so this drift never enters a ratio; here it
    // only bounds gross disagreement (wrong position/token alignment would be
    // off by whole units).
    let completion_scores = &scored[prompt.len() - 1..];
    assert_eq!(completion_scores.len(), generation.logprobs.len());
    for (i, (&rescored, &sampled)) in completion_scores
        .iter()
        .zip(&generation.logprobs)
        .enumerate()
    {
        assert!(
            (rescored - sampled).abs() < 0.25,
            "token {i}: rescored {rescored} too far from rollout {sampled}"
        );
    }
    let tokens = trainer.tokenize_text(PROMPT).expect("tokenize");
    let dim = trainer.hidden_size().expect("hidden size");
    assert!(dim > 0);

    let features = trainer.hidden_states(&tokens).expect("hidden states");
    assert_eq!(features.len(), tokens.len() * dim);
    assert!(features.iter().all(|value| value.is_finite()));
    // Rows must differ across positions (a broken extraction that repeats one
    // row would silently cripple the value head).
    assert_ne!(features[..dim], features[dim..2 * dim]);

    // Deterministic: the critic's features must be stable across calls.
    let again = trainer.hidden_states(&tokens).expect("hidden states again");
    assert_eq!(features, again);

    // Extraction must not disturb the scoring path (both directions).
    let scored = trainer.score_tokens(&tokens).expect("score");
    let _ = trainer
        .hidden_states(&tokens)
        .expect("hidden states after score");
    let rescored = trainer.score_tokens(&tokens).expect("score again");
    assert_eq!(scored, rescored, "hidden-state extraction leaked state");
}

/// Builds aligned SFT-style rows (labels = next token) over the test text.
fn packed_rows(trainer: &Trainer, n_rows: usize) -> (Vec<i32>, Vec<i32>) {
    let n_ctx = trainer.context_size().expect("context");
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(120);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    assert!(
        tokens.len() > n_rows * n_ctx + 1,
        "test text too short: {} tokens for {} rows of {}",
        tokens.len(),
        n_rows,
        n_ctx
    );
    let mut row_tokens = Vec::new();
    let mut labels = Vec::new();
    for row in 0..n_rows {
        let start = row * n_ctx;
        row_tokens.extend_from_slice(&tokens[start..start + n_ctx]);
        labels.extend_from_slice(&tokens[start + 1..start + n_ctx + 1]);
    }
    (row_tokens, labels)
}

#[test]
fn unit_weights_reproduce_the_sft_loss() {
    let _guard = serialize_models();
    let Some(mut weighted_trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut sft_trainer = Trainer::new(common::model_path(), config()).expect("load trainer");
    weighted_trainer.create_lora(&lora()).expect("create lora");
    sft_trainer.create_lora(&lora()).expect("create lora");

    let n_ctx = weighted_trainer.context_size().expect("context");
    let (tokens, labels) = packed_rows(&weighted_trainer, 2);

    let weighted = weighted_trainer
        .train_weighted(
            &WeightedBatch {
                tokens: tokens.clone(),
                labels: labels.clone(),
                weights: vec![1.0; tokens.len()],
                n_rows: 2,
                n_ctx,
                n_topk: 1,
            },
            0,
        )
        .expect("weighted step");

    let prepared = retrograd::dataset::PreparedDataset {
        tokens,
        labels,
        n_ctx,
        examples: 2,
        supervised_tokens: 2 * n_ctx,
    };
    let sft = sft_trainer
        .train_sft_with_progress(&prepared, None, |_| {})
        .expect("sft step");

    // Identical rows, identical LoRA seed, identical optimizer: the weighted
    // objective with all-ones weights IS the SFT objective. Both losses must
    // also be real (a silently masked-out batch would compare 0 == 0).
    assert!(
        weighted.train_loss > 0.1,
        "weighted loss {} is suspiciously low",
        weighted.train_loss
    );
    assert!(
        (weighted.train_loss - sft.train_loss).abs() < 1e-4,
        "weighted loss {} != sft loss {}",
        weighted.train_loss,
        sft.train_loss
    );
}

#[test]
fn sft_train_and_eval_use_distinct_external_dataset_segments() {
    let _guard = serialize_models();
    let Some(mut trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    trainer.create_lora(&lora()).expect("create lora");
    let n_ctx = trainer.context_size().expect("context");
    let (tokens, labels) = packed_rows(&trainer, 3);
    let split = 2 * n_ctx;
    let train = retrograd::dataset::PreparedDataset {
        tokens: tokens[..split].to_vec(),
        labels: labels[..split].to_vec(),
        n_ctx,
        examples: 2,
        supervised_tokens: split,
    };
    let eval = retrograd::dataset::PreparedDataset {
        tokens: tokens[split..].to_vec(),
        labels: labels[split..].to_vec(),
        n_ctx,
        examples: 1,
        supervised_tokens: n_ctx,
    };

    let metrics = trainer
        .train_sft_with_progress(&train, Some(&eval), |_| {})
        .expect("train and evaluate split external dataset");
    assert!(metrics.train_loss.is_finite() && metrics.train_loss > 0.0);
    assert!(metrics.eval_loss.is_finite() && metrics.eval_loss > 0.0);
}

#[test]
fn fixed_reference_ignores_lora_updates_and_restores_the_policy() {
    let _guard = serialize_models();
    let Some(mut trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    trainer.create_lora(&lora()).expect("create lora");
    let n_ctx = trainer.context_size().expect("context");
    let (tokens, labels) = packed_rows(&trainer, 1);
    let window = (config().n_ctx as usize).min(n_ctx);
    let mut sequence = tokens[..window].to_vec();
    sequence.push(labels[window - 1]);

    let reference_before = trainer
        .score_reference_tokens(&sequence)
        .expect("score fixed reference");
    let policy_before = trainer.score_tokens(&sequence).expect("score policy");
    // A reference score must leave the active LoRA attached.
    assert_eq!(reference_before, policy_before);

    trainer
        .train_weighted(
            &WeightedBatch {
                tokens: tokens.clone(),
                labels,
                weights: vec![1.0; tokens.len()],
                n_rows: 1,
                n_ctx,
                n_topk: 1,
            },
            0,
        )
        .expect("weighted step");

    let policy_after = trainer
        .score_tokens(&sequence)
        .expect("score updated policy");
    let reference_after = trainer
        .score_reference_tokens(&sequence)
        .expect("score fixed reference again");
    assert_ne!(
        policy_before, policy_after,
        "LoRA update had no policy effect"
    );
    assert_eq!(
        reference_before, reference_after,
        "the base-model GRPO reference drifted after a LoRA update"
    );
    // Scoring the reference must restore the updated policy, not leave LoRA
    // disabled on the context.
    assert_eq!(
        policy_after,
        trainer
            .score_tokens(&sequence)
            .expect("score restored policy")
    );
}

/// End-to-end differentiability: signed weights must produce signed losses,
/// positive supervision must raise its target probability, and either sign
/// must update the policy while leaving the base model frozen.
#[test]
fn weight_sign_steers_the_policy() {
    let _guard = serialize_models();
    let Some(_) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut outcomes = Vec::new();
    for weight in [1.0_f32, -1.0_f32] {
        // One moderate step is enough for the deterministic direction check
        // without overshooting on low-bit base models.
        let mut train_config = config();
        train_config.learning_rate = 1.0e-4;
        let mut trainer = Trainer::new(common::model_path(), train_config).expect("load trainer");
        trainer.create_lora(&lora()).expect("create lora");
        let n_ctx = trainer.context_size().expect("context");
        let (tokens, labels) = packed_rows(&trainer, 1);

        // Measure exactly the single next-token transition carrying the
        // non-zero coefficient. Keeping this sequence within n_ctx avoids a
        // scoring-window shift on recurrent models.
        let sequence = vec![tokens[0], labels[0]];
        let before = trainer.score_tokens(&sequence).expect("score")[0];

        let mut weights = vec![0.0; tokens.len()];
        weights[0] = weight;
        let batch = WeightedBatch {
            tokens: tokens.clone(),
            labels: labels.clone(),
            weights,
            n_rows: 1,
            n_ctx,
            n_topk: 1,
        };
        let metrics = trainer.train_weighted(&batch, 0).expect("weighted step");
        assert!(
            metrics.train_loss.signum() == weight.signum(),
            "weight {weight}: weighted loss had unexpected sign {}",
            metrics.train_loss
        );

        let after = trainer.score_tokens(&sequence).expect("score")[0];
        assert_ne!(after, before, "weight {weight}: policy did not move");
        outcomes.push((weight, before, after));
    }
    assert_eq!(outcomes[0].1, outcomes[1].1, "initial policies differ");
    assert!(
        outcomes[0].2 > outcomes[0].1,
        "positive weight moved target logprob from {} to {}",
        outcomes[0].1,
        outcomes[0].2
    );
}

/// Full PPO loop: prompts -> rollouts -> external rewards -> whitened
/// advantages -> weighted differentiable updates, with the base model frozen.
#[test]
fn ppo_runs_end_to_end() {
    let _guard = serialize_models();
    let Some(mut trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    trainer.create_lora(&lora()).expect("create lora");

    let dir = std::env::temp_dir().join(format!(
        "retrograd-ppo-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let prompts = dir.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox\"}]}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Pack my box with\"}]}\n"
        ),
    )
    .unwrap();
    // Reward: completion length, so the batch rewards differ and whitening
    // produces non-trivial advantages.
    let reward = dir.join("reward.sh");
    std::fs::write(
        &reward,
        format!(
            r#"IFS= read -r hello || exit 1
printf '{{"protocol":"{}"}}\n'
while IFS= read -r line; do
  case "$line" in
    *'"_retrograd_batch_end"'*) printf '%s\n' "$line";;
    *)
      batch=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_batch":\([0-9][0-9]*\).*/\1/p')
      index=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_index":\([0-9][0-9]*\).*/\1/p')
      printf '{{"reward":%d,"_retrograd_batch":%s,"_retrograd_index":%s}}\n' "${{#line}}" "$batch" "$index"
      ;;
  esac
done
"#,
            retrograd::training::REWARD_PROTOCOL_VERSION
        ),
    )
    .unwrap();

    let ppo_config = retrograd::config::PpoConfig {
        prompts,
        reward_command: vec!["/bin/sh".into(), reward.display().to_string()],
        // The default: one worker for the whole loop, which the script above
        // answers the handshake for.
        reward_protocol: retrograd::RewardProtocol::default(),
        updates: 1,
        rollout_batch_size: 1,
        // One epoch still exercises the complete PPO update. Repeating the
        // exact same one-sample rollout only adds a redundant, highly
        // correlated step and can trip the intended KL trust-region guard.
        ppo_epochs: 1,
        clip_range: 0.2,
        kl_coefficient: 0.1,
        // Default critic: linear-probe value head + GAE per-token advantages.
        critic: retrograd::config::CriticConfig::default(),
        sampling: retrograd::SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            max_new_tokens: 8,
            seed: 42,
        },
    };

    let mut progress_events = 0;
    let metrics =
        retrograd::training::ppo::run(&mut trainer, &ppo_config, &config(), &mut |event| {
            progress_events += 1;
            assert!(event.values.iter().any(|value| value.name == "reward/mean"));
            // The default critic is enabled, so every update reports its
            // regression loss and it must stay finite.
            let value_loss = event
                .values
                .iter()
                .find(|value| value.name == "policy/value_loss")
                .expect("critic value loss metric");
            assert!(value_loss.value.is_finite());
        })
        .expect("ppo run");

    assert_eq!(progress_events, 1, "one progress event per ppo epoch");
    assert!(metrics.train_loss.is_finite());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn zero_weights_are_a_legitimate_noop_step() {
    let _guard = serialize_models();
    let Some(mut trainer) = trainer() else {
        eprintln!("skipping: no local test model");
        return;
    };
    trainer.create_lora(&lora()).expect("create lora");
    let n_ctx = trainer.context_size().expect("context");
    let (tokens, labels) = packed_rows(&trainer, 1);

    // A fully clipped PPO batch produces all-zero weights; that must succeed
    // with a zero loss instead of erroring like SFT would.
    let metrics = trainer
        .train_weighted(
            &WeightedBatch {
                tokens,
                labels,
                weights: vec![0.0; n_ctx],
                n_rows: 1,
                n_ctx,
                n_topk: 1,
            },
            0,
        )
        .expect("zero-weight step");
    assert!(
        metrics.train_loss.abs() < 1e-6,
        "loss {}",
        metrics.train_loss
    );
}
