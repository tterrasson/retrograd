//! The inference scheduler over a real engine, on the CPU fixture: in
//! `Lockstep` mode, submitting the same heterogeneous wave (duplicate prompts
//! included) through the actor must reproduce `Engine::generate_sampled_continuous`
//! bit for bit - tokens and forced mask - and an `n > 1` request must
//! reproduce `Engine::generate_batch` (tokens and logprobs). Skipped when no
//! local model exists (override with `RETRO_CPU_FIXTURE`).

mod common;

use std::time::Duration;

use common::serialize_models;
use retrograd::{Device, Engine, EngineConfig, EngineKvDtype, SamplingParams};
use retrograd_infer::{
    ActorHandle, ActorOptions, AdmissionMode, Event, FinishReason, Request, SchedulerOptions,
};

/// The engine geometry that reproduces the trainer's dedicated generation
/// context of `engine_runtime`: 4 slots over a 256-cell pool, generation
/// batch 64, F16 KV with flash attention, CPU.
fn engine_config() -> EngineConfig {
    EngineConfig {
        n_ctx: 256,
        n_seq_max: 4,
        n_batch: 64,
        n_ubatch: 64,
        kv_dtype: EngineKvDtype::F16,
        flash_attn: true,
        device: Device::Cpu,
        threads: 0,
        verbose: false,
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

fn actor_options(mode: AdmissionMode) -> ActorOptions {
    ActorOptions {
        scheduler: SchedulerOptions {
            mode,
            n_batch: 64,
            queue_capacity: 16,
            event_capacity: 256,
            slow_client_timeout: Duration::from_secs(5),
        },
    }
}

/// Splits one request's collected events per completion index into
/// (tokens, logprobs, forced, reason).
fn summarize(
    events: &[Event],
    index: usize,
) -> (Vec<i32>, Vec<f32>, Vec<bool>, Option<FinishReason>) {
    let mut tokens = Vec::new();
    let mut logprobs = Vec::new();
    let mut forced = Vec::new();
    let mut reason = None;
    for event in events {
        match event {
            Event::Token(token) if token.index == index => {
                tokens.push(token.token);
                logprobs.push(token.logprob);
                forced.push(token.forced);
            }
            Event::End {
                index: i,
                reason: r,
                ..
            } if *i == index => {
                reason = Some(*r);
            }
            Event::Error(error) => panic!("request failed: {error}"),
            _ => {}
        }
    }
    (tokens, logprobs, forced, reason)
}

#[test]
fn lockstep_scheduler_matches_generate_continuous_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };

    let mut reference_engine = Engine::new(&model, &engine_config()).expect("reference engine");

    // Two rows share `first` so the wave exercises the merged prefill + seq_cp
    // group, and the prompts differ so the canonical order is not the caller
    // order. Submitted in a scrambled order on purpose.
    let first = reference_engine
        .tokenize_text("The quick brown fox")
        .unwrap();
    let second = reference_engine.tokenize_text("Pack my box with").unwrap();
    let wave: Vec<(&[i32], SamplingParams)> = vec![
        (&second, params(50, 6)),
        (&first, params(51, 6)),
        (&first, params(52, 6)),
    ];

    let reference = reference_engine
        .generate_sampled_continuous(&wave)
        .expect("continuous reference");

    // The engine is built on the actor thread: only the path and the config
    // cross the boundary, the handle itself never does.
    let actor_model = model.clone();
    let (start, ready) = std::sync::mpsc::channel();
    let handle = ActorHandle::spawn(
        move || {
            ready.recv().expect("release the full lockstep wave");
            Engine::new(&actor_model, &engine_config())
                .map_err(retrograd_infer::InferError::Backend)
        },
        actor_options(AdmissionMode::Lockstep),
    )
    .expect("actor spawns");
    let mut streams = Vec::new();
    for (prompt, sampling) in &wave {
        streams.push(
            handle
                .submit(Request::single(prompt.to_vec(), *sampling))
                .expect("submit"),
        );
    }
    start.send(()).unwrap();
    for (index, stream) in streams.into_iter().enumerate() {
        let events: Vec<Event> = stream.collect();
        let (tokens, _logprobs, forced, reason) = summarize(&events, 0);
        assert_eq!(
            tokens, reference[index].tokens,
            "request {index}: tokens bit for bit"
        );
        assert_eq!(
            forced, reference[index].forced,
            "request {index}: forced mask bit for bit"
        );
        assert_eq!(reason, Some(FinishReason::MaxTokens));
    }
    handle.shutdown();
    handle.join().expect("the actor drains cleanly");
}

#[test]
fn n_greater_than_one_request_matches_generate_batch_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };

    let mut reference_engine = Engine::new(&model, &engine_config()).expect("reference engine");

    let prompt = reference_engine
        .tokenize_text("The quick brown fox")
        .unwrap();
    let sampling: Vec<SamplingParams> = vec![params(61, 5), params(62, 5), params(63, 5)];
    let reference = reference_engine
        .generate_batch(&prompt, &sampling)
        .expect("batch reference");

    let actor_model = model.clone();
    let handle = ActorHandle::spawn(
        move || {
            Engine::new(&actor_model, &engine_config())
                .map_err(retrograd_infer::InferError::Backend)
        },
        actor_options(AdmissionMode::Continuous),
    )
    .expect("actor spawns");
    let mut request = Request::single(prompt.clone(), sampling[0]);
    request.sampling = sampling.clone();
    let stream = handle.submit(request).expect("submit");
    let events: Vec<Event> = stream.collect();
    for index in 0..3 {
        let (tokens, logprobs, _forced, reason) = summarize(&events, index);
        assert_eq!(tokens, reference[index].tokens, "member {index}: tokens");
        assert_eq!(
            logprobs, reference[index].logprobs,
            "member {index}: logprobs bit for bit"
        );
        assert_eq!(reason, Some(FinishReason::MaxTokens));
    }
    handle.shutdown();
    handle.join().expect("the actor drains cleanly");
}

#[test]
fn trainer_backend_reports_its_generation_context_instead_of_its_training_context() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let config = retrograd::TrainConfig {
        n_ctx: 128,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 4,
        generation_concurrency: 4,
        device: Device::Cpu,
        ..retrograd::TrainConfig::default()
    };
    let mut trainer = retrograd::Trainer::new(&model, config).unwrap();
    let training_context = trainer.context_size().unwrap();
    let generation_context = trainer.memory().unwrap().cells_total;
    assert!(generation_context > training_context as u64);
    let backend = retrograd_infer::TrainerBackend(&mut trainer);
    assert_eq!(
        u64::from(retrograd_infer::DecodeBackend::context_size(&backend).unwrap()),
        generation_context
    );
}
