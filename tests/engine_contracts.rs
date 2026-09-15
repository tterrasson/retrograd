//! Error-path contracts for `retrograd::Trainer` that `capabilities.rs` and
//! the algorithm-specific integration suites don't exercise: a missing model
//! path, and the validation guards on masked/suffix scoring and bounded
//! generation. These are worth locking down independently of whether the
//! surrounding training loop is correct.

mod common;

use retrograd::{Device, LoraConfig, SamplingParams, TargetSet, TrainConfig, Trainer};

fn cpu_config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 32,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

/// No model needs to load for this one, so it runs in every lane, including
/// `abi`.
#[test]
fn trainer_new_reports_a_clean_error_for_a_missing_model_path() {
    let error = match Trainer::new("tests/fixtures/does-not-exist.gguf", cpu_config()) {
        Ok(_) => panic!("a missing model path must not construct a trainer"),
        Err(error) => error,
    };
    assert!(
        !error.to_string().is_empty(),
        "runtime error must carry a message"
    );
}

/// The candidate listing is what makes a LoRA target resolvable without
/// assuming a block layout, and the failure it replaces is what a caller sees
/// when the assumption is wrong. Both halves are asserted together so the
/// listing cannot drift away from the patterns the resolver actually matches.
#[test]
fn lora_candidate_targets_resolve_a_block_and_a_miss_names_the_alternatives() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(&model, cpu_config()).expect("load model");

    let candidates = trainer
        .lora_candidate_targets()
        .expect("candidate targets are available before an adapter exists");
    assert!(
        !candidates.is_empty(),
        "a loaded model must expose at least one LoRA-eligible tensor"
    );
    assert!(
        candidates.windows(2).all(|pair| pair[0] < pair[1]),
        "candidates must be sorted and unique: {candidates:?}"
    );

    // Every listed name has to be a name the resolver accepts, otherwise the
    // listing would send callers to targets that then fail to match.
    let first = candidates.first().expect("non-empty").clone();
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(vec![first.clone()]);
    trainer
        .create_lora(&lora)
        .unwrap_or_else(|error| panic!("candidate {first} did not resolve: {error}"));

    // A pattern naming a block that carries a different family is the failure
    // this listing exists to answer, so the error has to point at it.
    let mut trainer = Trainer::new(&model, cpu_config()).expect("reload model");
    let mut missing = LoraConfig::qv(2, 4.0);
    missing.seed = 7;
    missing.targets = TargetSet::Patterns(vec!["blk.0.no_such_projection.weight".to_string()]);
    let error = trainer
        .create_lora(&missing)
        .expect_err("a target matching nothing must not create an adapter")
        .to_string();
    assert!(
        error.contains("candidate patterns: ["),
        "the failure must list what the model does carry: {error}"
    );
}

#[test]
fn train_tokens_rejects_a_single_token_sequence() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(&model, cpu_config()).expect("load model");

    let error = trainer
        .train_tokens(&[1])
        .expect_err("training requires at least two tokens");
    assert!(error.to_string().contains("at least two tokens"));
}

#[test]
fn generation_is_bounded_and_model_info_matches_the_loaded_trainer() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(&model, cpu_config()).expect("load model");
    let prompt = trainer.tokenize_text("Hello").expect("tokenize prompt");

    for &bound in &[1_usize, 3, 8] {
        let sampling = SamplingParams {
            temperature: 1.0e-3,
            top_p: 1.0,
            max_new_tokens: bound as u32,
            seed: 1,
        };
        let generation = trainer
            .generate(&prompt, &sampling)
            .expect("bounded generation should succeed");
        assert!(
            generation.tokens.len() <= bound,
            "generated {} tokens for a bound of {bound}",
            generation.tokens.len()
        );
    }

    // `model_info` is the resolver's cheap first input; it must describe the
    // same geometry as the already-loaded trainer.
    let info = retrograd::model_info(&model, Device::Cpu).expect("read model geometry");
    assert!(info.n_layer > 0, "a model has at least one layer");
    assert!(info.n_vocab > 0);
    assert!(info.n_ctx_train > 0);
    // Recurrent/hybrid architectures can report zero aggregate KV geometry
    // even though their attention layers still have heads.
    assert!(info.n_head_kv == 0 || info.n_head_kv <= info.n_head);
    assert!(info.n_head_kv == 0 || (info.n_embd_k_gqa > 0 && info.n_embd_v_gqa > 0));
    assert!(info.model_size_bytes > 0 && info.n_params > 0);
    assert_eq!(
        info.file_size_bytes,
        std::fs::metadata(&model).expect("stat fixture").len()
    );
    assert!(!info.architecture.is_empty());
    // The dominant type must account for a real share of the weights, not be a
    // stray F32 norm tensor.
    assert!(!info.dominant_weight_type.is_empty());
    assert!(info.dominant_weight_bytes * 2 > info.model_size_bytes);

    assert_eq!(
        info.n_embd as usize,
        trainer.hidden_size().expect("hidden size")
    );
    assert_eq!(
        info.n_vocab as usize,
        trainer.vocab_size().expect("vocab size")
    );

    // The KV formula has to match what llama.cpp actually allocated. The
    // optimizer context is F32 by default and holds `n_ctx` tokens; the runtime
    // pads its cache, so the analytic figure is a lower bound, not an equality.
    let report = trainer.memory_report().expect("memory report");
    let expected_kv = info.kv_cache_bytes(cpu_config().n_ctx as u64, 4);
    if expected_kv > 0 {
        assert!(
            report.optimizer_kv_bytes >= expected_kv,
            "runtime KV {} is below the analytic {expected_kv}",
            report.optimizer_kv_bytes
        );
        assert!(
            report.optimizer_kv_bytes <= expected_kv * 4,
            "runtime KV {} is more than 4x the analytic {expected_kv}",
            report.optimizer_kv_bytes
        );
    } else {
        assert!(
            report.optimizer_kv_bytes > 0,
            "a hybrid model still allocates recurrent/attention state"
        );
    }

    // The structured report and the text one are rendered from one computation,
    // so every byte figure must be the same on both sides.
    let text = trainer.backend_report().expect("backend report");
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(&format!("{name}: ")))
            .and_then(|value| value.trim().parse().ok())
    };
    assert_eq!(field("model_weight_bytes"), Some(report.model_weight_bytes));
    assert_eq!(
        field("training_kv_cache_bytes"),
        Some(report.optimizer_kv_bytes)
    );
    assert_eq!(field("memory_device_bytes"), Some(report.device_bytes));
    assert_eq!(field("memory_host_bytes"), Some(report.host_bytes));
    assert_eq!(
        text.lines()
            .find_map(|line| line.trim().strip_prefix("model_weight_dtype: "))
            .map(str::trim),
        Some(info.dominant_weight_type.as_str()),
        "the dominant weight type must not depend on which accessor asked"
    );
}
