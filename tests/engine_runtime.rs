//! Engine vs trainer parity on the CPU fixture: the inference engine runs the
//! same generation core as the trainer's `retro_trainer_generate_*`, so at
//! equal geometry, seeds and weights its output must be bit-identical. The
//! engine's memory report must also show no training state (zero optimizer
//! state, zero gradient), which is the property that distinguishes the handle
//! from a trainer. Skipped when no local model exists (override with
//! `RETRO_CPU_FIXTURE`).

mod common;

use common::serialize_models;
use retrograd::{
    Device, Engine, EngineConfig, EngineKvDtype, LoraConfig, SamplingParams, TargetSet,
    TrainConfig, Trainer,
};

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

/// The engine geometry that reproduces the trainer's dedicated generation
/// context exactly: `n_ctx * effective_generation_concurrency` cells, one slot
/// per concurrent sequence, the generation batch, and the fast context's F16
/// KV with flash attention (the `TrainConfig` defaults here).
fn engine_config(train: &TrainConfig) -> EngineConfig {
    let concurrency = train.effective_generation_concurrency();
    let generation_batch = if train.generation_batch != 0 {
        train.generation_batch
    } else {
        train.n_ctx.min(512)
    };
    let generation_batch = generation_batch.max(concurrency);
    EngineConfig {
        n_ctx: train.n_ctx * concurrency,
        n_seq_max: concurrency,
        n_batch: generation_batch,
        n_ubatch: generation_batch,
        kv_dtype: if train.fast_generation_context {
            match train.generation_kv_dtype {
                retrograd::GenerationKvDtype::F16 => EngineKvDtype::F16,
                retrograd::GenerationKvDtype::Q8_0 => EngineKvDtype::Q8_0,
                retrograd::GenerationKvDtype::Q4_0 => EngineKvDtype::Q4_0,
            }
        } else {
            EngineKvDtype::F32
        },
        flash_attn: train.fast_generation_context,
        device: Device::Cpu,
        threads: 0,
        verbose: false,
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

fn params(seed: u32) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 6,
        seed,
    }
}

/// The full parity check: shared-prompt batches, single generation and
/// heterogeneous continuous waves must all agree bit for bit, tokens and
/// logprobs alike.
fn assert_parity(trainer: &mut Trainer, engine: &mut Engine, prompt: &[i32], rows: &[&[i32]]) {
    // Shared-prompt batch, logprobs included. `Generation` has no PartialEq,
    // so compare the fields: tokens exactly, logprobs bit for bit.
    let sampling = [params(31), params(32), params(33)];
    let trainer_batch = trainer
        .generate_batch(prompt, &sampling)
        .expect("trainer generate_batch");
    let engine_batch = engine
        .generate_batch(prompt, &sampling)
        .expect("engine generate_batch");
    assert_eq!(trainer_batch.len(), engine_batch.len());
    for (left, right) in trainer_batch.iter().zip(&engine_batch) {
        assert_eq!(left.tokens, right.tokens, "shared-prompt batch tokens");
        assert_eq!(
            left.logprobs, right.logprobs,
            "shared-prompt batch logprobs"
        );
    }

    // Single generation.
    let trainer_one = trainer
        .generate(prompt, &params(41))
        .expect("trainer generate");
    let engine_one = engine
        .generate(prompt, &params(41))
        .expect("engine generate");
    assert_eq!(
        trainer_one.tokens, engine_one.tokens,
        "single generation tokens"
    );
    assert_eq!(
        trainer_one.logprobs, engine_one.logprobs,
        "single generation logprobs"
    );

    // Heterogeneous continuous wave, tokens and forced masks.
    let wave: Vec<(&[i32], SamplingParams)> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| (*row, params(50 + i as u32)))
        .collect();
    let trainer_rows = trainer
        .generate_sampled_continuous(&wave)
        .expect("trainer continuous");
    let engine_rows = engine
        .generate_sampled_continuous(&wave)
        .expect("engine continuous");
    assert_eq!(trainer_rows, engine_rows, "continuous wave parity");

    // Logprob-free paths agree too.
    let trainer_tokens = trainer
        .generate_tokens_batch(prompt, &sampling)
        .expect("trainer tokens batch");
    let engine_tokens = engine
        .generate_tokens_batch(prompt, &sampling)
        .expect("engine tokens batch");
    assert_eq!(trainer_tokens, engine_tokens);
    let trainer_wave = trainer
        .generate_tokens_continuous(&wave)
        .expect("trainer tokens continuous");
    let engine_wave = engine
        .generate_tokens_continuous(&wave)
        .expect("engine tokens continuous");
    assert_eq!(trainer_wave, engine_wave);
}

#[test]
fn engine_generate_matches_trainer_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");

    // Tokenization agrees, so the prompts below are the same token stream.
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    assert_eq!(first, engine.tokenize_text("The quick brown fox").unwrap());
    assert_eq!(second, engine.tokenize_text("Pack my box with").unwrap());
    assert_eq!(trainer.eos_token().unwrap(), engine.eos_token().unwrap());
    assert_eq!(
        trainer.vocab_size().unwrap(),
        engine.vocab_size().unwrap() as usize
    );
    assert_eq!(
        engine.context_size().unwrap() as u32,
        train.n_ctx * train.effective_generation_concurrency(),
        "the engine context must match the trainer's generation context width"
    );

    // Base weights first.
    assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);

    // Then with a LoRA adapter attached on both sides: the trainer creates and
    // saves one, the engine loads the same file.
    trainer.create_lora(&lora()).expect("create lora");
    let tmp = common::scratch("engine-parity-lora");
    let adapter = tmp.path().join("adapter.gguf");
    trainer.save_lora(&adapter).expect("save lora");
    engine.load_lora(&adapter).expect("engine load lora");
    assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);

    // Disabling the adapter on the engine must restore base-model parity
    // against a fresh adapter-free engine.
    let mut engine_base = Engine::new(&model, &engine_config(&train)).expect("load base engine");
    engine.set_lora_enabled(false).expect("engine disable lora");
    let trainer_side = engine_base
        .generate(&second, &params(41))
        .expect("base engine");
    let disabled = engine
        .generate(&second, &params(41))
        .expect("disabled engine");
    assert_eq!(trainer_side.tokens, disabled.tokens);
    assert_eq!(trainer_side.logprobs, disabled.logprobs);

    // The engine is deterministic across repeated calls.
    let again = engine
        .generate(&first, &params(41))
        .expect("engine generate");
    let once = engine
        .generate(&first, &params(41))
        .expect("engine generate again");
    assert_eq!(again.tokens, once.tokens);
    assert_eq!(again.logprobs, once.logprobs);
}

#[test]
fn engine_generation_matches_exact_and_quantized_trainer_contexts() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        return;
    };
    for dtype in [EngineKvDtype::F32, EngineKvDtype::Q8_0, EngineKvDtype::Q4_0] {
        let mut train = train_config();
        train.fast_generation_context = dtype != EngineKvDtype::F32;
        train.generation_kv_dtype = match dtype {
            EngineKvDtype::Q8_0 => retrograd::GenerationKvDtype::Q8_0,
            EngineKvDtype::Q4_0 => retrograd::GenerationKvDtype::Q4_0,
            _ => retrograd::GenerationKvDtype::F16,
        };
        let mut trainer = Trainer::new(&model, train.clone()).unwrap();
        let mut engine = Engine::new(&model, &engine_config(&train)).unwrap();
        let first = trainer.tokenize_text("The quick brown fox").unwrap();
        let second = trainer.tokenize_text("Pack my box with").unwrap();
        assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);
    }
}

#[test]
fn engine_memory_report_carries_no_training_state() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    // Configured to train for real: the optimizer graph exists.
    trainer.prepare_optimizer().expect("prepare optimizer");
    let trainer_report = trainer.memory_report().expect("trainer memory report");

    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");
    let tmp = common::scratch("engine-memory-lora");
    let adapter = tmp.path().join("adapter.gguf");
    trainer.save_lora(&adapter).expect("save lora");
    engine.load_lora(&adapter).expect("engine load lora");
    let engine_report = engine.memory_report().expect("engine memory report");

    // The engine has no optimizer and no backward graph: those fields are a
    // fact, not a gap.
    assert_eq!(engine_report.optimizer_state_bytes, 0);
    assert_eq!(engine_report.trainable_gradient_bytes, 0);
    assert_eq!(engine_report.optimizer_kv_bytes, 0);
    assert_eq!(engine_report.optimizer_compute_bytes, 0);
    assert_eq!(engine_report.checkpoint_count, 0);
    assert!(engine_report.has_generation_context);
    assert!(engine_report.model_weight_bytes > 0);
    assert!(engine_report.generation_kv_bytes > 0);
    // The same adapter is attached on both sides.
    assert_eq!(
        engine_report.trainable_parameter_bytes,
        trainer_report.trainable_parameter_bytes
    );
    assert!(engine_report.trainable_parameter_bytes > 0);

    // Against a trainer actually configured to train, the engine's footprint is
    // strictly smaller: no optimizer context KV, no compute buffers, no
    // optimizer state, no gradient.
    assert!(trainer_report.optimizer_state_bytes > 0);
    assert!(trainer_report.trainable_gradient_bytes > 0);
    let engine_total = engine_report.device_bytes + engine_report.host_bytes;
    let trainer_total = trainer_report.device_bytes + trainer_report.host_bytes;
    assert!(
        engine_total < trainer_total,
        "engine {engine_total} bytes should be below the training-configured trainer's \
         {trainer_total} bytes"
    );
    eprintln!(
        "memory: engine total {engine_total} B vs trainer {trainer_total} B \
         (optimizer_state {} B, gradient {} B, optimizer kv {} B)",
        trainer_report.optimizer_state_bytes,
        trainer_report.trainable_gradient_bytes,
        trainer_report.optimizer_kv_bytes,
    );

    // Generation counters are reported on the engine too; the continuous path
    // is the one that counts calls.
    let stats = engine.generation_stats().expect("engine generation stats");
    assert_eq!(stats.calls, 0);
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    engine
        .generate_tokens_continuous(&[(&prompt, params(7))])
        .expect("engine continuous");
    let stats = engine.generation_stats().expect("engine generation stats");
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.sequences, 1);
}

#[test]
fn engine_refuses_a_missing_model_with_a_clean_error() {
    let _guard = serialize_models();
    let config = engine_config(&train_config());
    let Err(error) = Engine::new("/nonexistent/model.gguf", &config) else {
        panic!("a missing model path must fail");
    };
    // Converted through the From into the core facade, it stays a runtime error.
    let core: retrograd::Error = error.into();
    assert!(matches!(core, retrograd::Error::Runtime(_)));
    assert!(core.to_string().contains("failed to load model GGUF"));
}

#[test]
fn engine_rejects_sequence_slots_larger_than_the_effective_native_batch() {
    let _guard = serialize_models();
    let Some(model) = common::tiny_model_path_if_available() else {
        return;
    };
    // llama.cpp clamps n_batch to the requested n_ctx before padding the
    // context to 256 cells. These dimensions would produce two sequence slots
    // but a physical batch of one, aborting during the output reservation.
    let mut config = EngineConfig {
        n_ctx: 1,
        n_seq_max: 2,
        n_batch: 2,
        n_ubatch: 2,
        device: Device::Cpu,
        ..EngineConfig::default()
    };
    assert!(matches!(
        Engine::new(&model, &config),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
    config.n_ctx = 2;
    let mut engine = Engine::new(&model, &config).unwrap();
    engine
        .generate_batch(&[1], &[params(1), params(2)])
        .unwrap();
    assert!(matches!(
        engine.generate_batch(&[1], &[params(1), params(2), params(3)]),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.generate_sampled_continuous(&[
            (&[1], params(1)),
            (&[1], params(2)),
            (&[1], params(3))
        ]),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
}

#[test]
fn engine_large_token_budgets_are_bounded_by_the_context() {
    let _guard = serialize_models();
    let Some(model) = common::tiny_model_path_if_available() else {
        return;
    };
    let config = EngineConfig {
        n_ctx: 256,
        n_seq_max: 2,
        n_batch: 64,
        n_ubatch: 64,
        device: Device::Cpu,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&model, &config).unwrap();
    let prompt = vec![1; 252];
    let mut unlimited = params(2);
    unlimited.max_new_tokens = u32::MAX;
    let mut bounded = unlimited.clone();
    bounded.max_new_tokens = 4;
    let expected = engine.generate(&prompt, &bounded).unwrap();
    let actual = engine.generate(&prompt, &unlimited).unwrap();
    assert_eq!(actual.tokens, expected.tokens);
    assert_eq!(actual.logprobs, expected.logprobs);
    for length in [252, 250] {
        let prompt = &prompt[..length];
        let mut bounded = unlimited.clone();
        // The test prompt lengths are below the 256-cell context.
        bounded.max_new_tokens = u32::try_from(256 - length).unwrap();
        let expected = engine
            .generate_sampled_continuous(&[(prompt, bounded)])
            .unwrap();
        let actual = engine
            .generate_sampled_continuous(&[(prompt, unlimited.clone())])
            .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(
        engine.generation_stats().unwrap().evictions > 0,
        "dormant near-full prefixes must yield their KV cells to the next call"
    );
}
