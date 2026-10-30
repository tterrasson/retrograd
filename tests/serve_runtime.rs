//! The serving session over a real engine, on the CPU fixture: a chat
//! completion served through `retrograd-openai`'s session (which drives the
//! `retrograd-infer` scheduler over the inference engine) must match the
//! engine's own generation bit for bit, and switching LoRA checkpoints must
//! activate the adapter without reloading the base model. Skipped when no
//! local model exists (override with `RETRO_CPU_FIXTURE`).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::serialize_models;
use retrograd::{Device, Engine, EngineConfig, EngineKvDtype, LoraConfig, TargetSet, Trainer};
use retrograd_infer::AdmissionMode;
use retrograd_openai::{
    EngineLoader, ParserCache, Session, SessionOptions, Unshared, WeightsSpec, prepare,
    render_prompt,
};

/// The engine geometry of `infer_runtime`: 4 slots over a 256-cell pool,
/// generation batch 64, F16 KV with flash attention, CPU.
fn engine_config() -> EngineConfig {
    EngineConfig {
        n_ctx: 256,
        n_seq_max: 4,
        n_batch: 64,
        n_ubatch: 64,
        kv_dtype: EngineKvDtype::F16,
        flash_attn: true,
        kv_unified: true,
        device: Device::Cpu,
        threads: 0,
        verbose: false,
    }
}

fn spec(fixture: &Path, adapter: Option<PathBuf>) -> WeightsSpec {
    WeightsSpec {
        model: fixture.to_path_buf(),
        adapter,
        n_ctx: 256,
        device: Device::Cpu,
        n_seq_max: Some(4),
        chat_template_variables: None,
    }
}

fn chat_prompt() -> retrograd_openai::GenerationRequest {
    prepare(
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 8,
            "seed": 7,
        }))
        .expect("a valid request"),
    )
    .expect("the request prepares")
    .request
}

fn completion_text(engine: &Engine, tokens: &[i32], unparse_special: bool) -> String {
    let mut end = tokens.len();
    while end > 0 && engine.is_eog_token(tokens[end - 1]).unwrap() {
        end -= 1;
    }
    engine.detokenize(&tokens[..end], unparse_special).unwrap()
}

fn session() -> Session {
    Session::new(
        Arc::new(EngineLoader),
        Arc::new(Unshared),
        SessionOptions {
            idle: None,
            admission: AdmissionMode::Continuous,
            ..SessionOptions::default()
        },
    )
    .expect("the serving thread starts")
}

#[tokio::test]
async fn a_served_completion_matches_the_engines_own_generation_bit_for_bit() {
    let _guard = serialize_models();
    let Some(fixture) = common::model_path_if_available() else {
        eprintln!("no CPU fixture; skipping");
        return;
    };
    let base = spec(&fixture, None);
    let prompt = chat_prompt();

    // What the engine itself would answer for the same rendered prompt.
    let mut engine = Engine::new(&fixture, &engine_config()).expect("engine loads");
    let mut parsers = ParserCache::default();
    let rendered = render_prompt(&engine, &prompt, &mut parsers).expect("prompt renders");
    let params = retrograd_core::SamplingParams {
        temperature: prompt.sampling.temperature,
        top_p: prompt.sampling.top_p,
        max_new_tokens: prompt.sampling.max_new_tokens,
        seed: prompt.sampling.seed,
    };
    let expected = engine
        .generate(&rendered.tokens, &params)
        .expect("engine generates");
    let expected_text = completion_text(&engine, &expected.tokens, rendered.unparse_special);

    let session = session();
    let sampled = session
        .complete(base.clone(), prompt.clone())
        .await
        .expect("the session serves the request")
        .pop()
        .expect("one choice");
    assert_eq!(sampled.text, expected_text);
    assert_eq!(sampled.completion_tokens, expected.tokens.len());
    assert_eq!(sampled.prompt_tokens, rendered.tokens.len());
    assert_eq!(
        sampled.stopped_at_eog,
        expected
            .tokens
            .last()
            .is_some_and(|t| engine.is_eog_token(*t).unwrap())
    );

    // The stats endpoint's view: one model loaded, no adapter, one served
    // request done.
    let stats = session.stats().await.expect("a model is loaded");
    assert_eq!(stats.spec.model, fixture);
    assert_eq!(stats.adapter, None);
    assert!(stats.adapters.is_empty());
}

#[tokio::test]
async fn switching_checkpoints_activates_the_adapter_without_reloading_the_base() {
    let _guard = serialize_models();
    let Some(fixture) = common::model_path_if_available() else {
        eprintln!("no CPU fixture; skipping");
        return;
    };
    // Two LoRA checkpoints of the fixture, written by a trainer.
    let mut trainer = Trainer::new(&fixture, Default::default()).expect("trainer loads");
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(
        common::block_targets(&trainer, &["attn_q"]).expect("the fixture has attention blocks"),
    );
    trainer.create_lora(&lora).expect("create lora");
    let scratch = common::scratch("serve-runtime-lora");
    let adapter_a = scratch.path().join("a.gguf");
    let adapter_b = scratch.path().join("b.gguf");
    trainer.save_lora(&adapter_a).expect("save adapter a");
    // A second, different adapter: a different seed draws different factors.
    lora.seed = 8;
    trainer.create_lora(&lora).expect("recreate lora");
    trainer.save_lora(&adapter_b).expect("save adapter b");

    let session = session();
    let prompt = chat_prompt();

    // Under adapter A, the served answer matches an engine with A attached.
    let mut engine_a = Engine::new(&fixture, &engine_config()).expect("engine loads");
    engine_a.load_lora(&adapter_a).expect("load A");
    let mut parsers = ParserCache::default();
    let rendered = render_prompt(&engine_a, &prompt, &mut parsers).expect("prompt renders");
    let params = retrograd_core::SamplingParams {
        temperature: prompt.sampling.temperature,
        top_p: prompt.sampling.top_p,
        max_new_tokens: prompt.sampling.max_new_tokens,
        seed: prompt.sampling.seed,
    };
    let expected_a = engine_a
        .generate(&rendered.tokens, &params)
        .expect("generate with A");
    let expected_text_a = completion_text(&engine_a, &expected_a.tokens, rendered.unparse_special);

    let sampled_a = session
        .complete(spec(&fixture, Some(adapter_a.clone())), prompt.clone())
        .await
        .expect("served under A")
        .pop()
        .expect("one choice");
    assert_eq!(sampled_a.text, expected_text_a);

    // Switch to B: no reload of the base, the adapter is registered and
    // activated per request - the answer must match an engine with B attached.
    let mut engine_b = Engine::new(&fixture, &engine_config()).expect("engine loads");
    engine_b.load_lora(&adapter_b).expect("load B");
    let expected_b = engine_b
        .generate(&rendered.tokens, &params)
        .expect("generate with B");
    let expected_text_b = completion_text(&engine_b, &expected_b.tokens, rendered.unparse_special);
    let sampled_b = session
        .complete(spec(&fixture, Some(adapter_b.clone())), prompt.clone())
        .await
        .expect("served under B")
        .pop()
        .expect("one choice");
    assert_eq!(sampled_b.text, expected_text_b, "B is the active adapter");
    let stats = session.stats().await.expect("loaded");
    assert_eq!(stats.adapters.len(), 2, "both checkpoints registered");
    assert_eq!(
        stats.adapter.as_deref(),
        Some(adapter_b.to_string_lossy().as_ref()),
        "B is active after its request"
    );
    // And back to the base model.
    let sampled_base = session
        .complete(spec(&fixture, None), prompt.clone())
        .await
        .expect("served under the base")
        .pop()
        .expect("one choice");
    let mut engine_base = Engine::new(&fixture, &engine_config()).expect("engine loads");
    let expected_base = engine_base
        .generate(&rendered.tokens, &params)
        .expect("generate base");
    let expected_text_base = completion_text(
        &engine_base,
        &expected_base.tokens,
        rendered.unparse_special,
    );
    assert_eq!(sampled_base.text, expected_text_base);
}
