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
        kv_unified: true,
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
            ..SchedulerOptions::default()
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
    // The KV capability and weights-version queries the prefix cache keys on.
    assert!(trainer.generation_kv_unified().unwrap());
    let version = trainer.generation_weights_version().unwrap();
    assert!(trainer.model_fingerprint().unwrap() != 0);
    // A weight mutation (adapter load/enable, optimizer step, import) bumps
    // the version through forget_generation_kv, which drops the prefix tree.
    trainer.create_lora(&retrograd::LoraConfig::qv(2, 4.0)).unwrap();
    let tmp = common::scratch("infer-trainer-lora");
    let adapter = tmp.path().join("adapter.gguf");
    trainer.save_lora(&adapter).unwrap();
    trainer.load_lora(&adapter).unwrap();
    assert!(trainer.generation_weights_version().unwrap() > version);
}

/// A tiny dense model (no recurrent state): exercises the dense prefix-cache
/// path on a real backend, including a fully cached prompt served by a
/// one-token replay of the last prompt token.
#[test]
fn dense_prefix_cache_full_prompt_hit_matches_cold_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::tiny_model_path_if_available() else {
        eprintln!("skipping: no tiny fixture");
        return;
    };
    let config = EngineConfig {
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
    };
    let mut reference = Engine::new(&model, &config).expect("reference engine");
    assert!(!reference.is_hybrid().unwrap(), "dense fixture");

    let shared_full = reference
        .tokenize_text(&"You are a terse assistant. Answer in one short sentence. ".repeat(8))
        .unwrap();
    let shared = shared_full[..96.min(shared_full.len())].to_vec();
    let suffix = reference.tokenize_fragment("Capital of France?").unwrap();
    let prompt_a: Vec<i32> = [shared.as_slice(), suffix.as_slice()].concat();

    let ref_a = reference
        .generate_sampled_continuous(&[(&prompt_a, params(81, 5))])
        .unwrap();
    // Wave 2: a fresh suffix on the shared prefix, and the shared prefix alone
    // (fully cached prompt).
    let suffix_b = reference.tokenize_fragment("Capital of Japan?").unwrap();
    let prompt_b: Vec<i32> = [shared.as_slice(), suffix_b.as_slice()].concat();
    let wave2: Vec<(Vec<i32>, SamplingParams)> =
        vec![(prompt_b.clone(), params(82, 5)), (shared.clone(), params(83, 5))];
    let ref_b = reference
        .generate_sampled_continuous(
            &wave2
                .iter()
                .map(|(p, s)| (p.as_slice(), *s))
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let actor_model = model.clone();
    let handle = ActorHandle::spawn(
        move || Engine::new(&actor_model, &config).map_err(retrograd_infer::InferError::Backend),
        actor_options(AdmissionMode::Lockstep),
    )
    .expect("actor spawns");

    let stream = handle
        .submit(Request::single(prompt_a.clone(), params(81, 5)))
        .unwrap();
    let events: Vec<Event> = stream.collect();
    let (tokens, _lp, forced, _reason) = summarize(&events, 0);
    assert_eq!(tokens, ref_a[0].tokens);
    assert_eq!(forced, ref_a[0].forced);
    assert_eq!(handle.kv_stats().unwrap().nodes, 1);

    let mut streams = Vec::new();
    for (prompt, sampling) in &wave2 {
        streams.push(handle.submit(Request::single(prompt.clone(), *sampling)).unwrap());
    }
    for (index, stream) in streams.into_iter().enumerate() {
        let events: Vec<Event> = stream.collect();
        let (tokens, _lp, forced, _reason) = summarize(&events, 0);
        assert_eq!(
            tokens, ref_b[index].tokens,
            "dense wave 2 request {index}: tokens bit for bit (request 1 is a full-prompt hit)"
        );
        assert_eq!(forced, ref_b[index].forced);
    }
    let stats = handle.kv_stats().unwrap();
    assert_eq!(stats.hits, 2, "{stats:?}");
    handle.shutdown();
    handle.join().expect("the actor drains cleanly");
}

#[test]
fn prefix_cache_reuse_matches_cold_generation_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };

    let mut reference_engine = Engine::new(&model, &engine_config()).expect("reference engine");
    // The pool is unified and the weights version is queryable: the prefix
    // cache's capability checks read these, they are not assumed.
    assert!(reference_engine.kv_unified().expect("kv_unified"));
    let version_before = reference_engine.weights_version().expect("weights_version");
    let fingerprint = reference_engine.model_fingerprint().expect("fingerprint");
    assert_ne!(fingerprint, 0);
    let recurrent = reference_engine.is_hybrid().unwrap();

    // Learn the SWA window behavior of this fixture: generate a long probe and
    // read the resident span of its slot. A slot whose pos_min is past 0 has
    // evicted its low positions (windowed SWA), so cross-slot sharing needs
    // sequences shorter than the window.
    let probe = reference_engine
        .tokenize_text(&"The quick brown fox jumps over the lazy dog. ".repeat(12))
        .unwrap();
    reference_engine
        .generate_sampled_continuous(&[(&probe, params(70, 4))])
        .unwrap();
    let (probe_min, probe_max) = reference_engine.seq_pos(0).unwrap();
    let window = if probe_min > 0 {
        (probe_max - probe_min + 1) as usize
    } else {
        usize::MAX
    };
    // Keep every sequence (prompt + generation) strictly inside the window so
    // pos_min stays 0 and the slot is shareable from position 0.
    let budget = 12; // 6 generated + suffix room
    let shared_len = if window == usize::MAX {
        80
    } else {
        window.saturating_sub(budget).max(16)
    };
    if window < 24 {
        eprintln!("skipping: SWA window {window} too small to share a prefix");
        return;
    }

    // A long shared prefix (> 80 % of every request's prompt in wave 2).
    let shared_full = reference_engine
        .tokenize_text(&"You are a terse assistant. Answer in one short sentence. ".repeat(12))
        .unwrap();
    let shared = shared_full[..shared_len.min(shared_full.len())].to_vec();
    let suffix_a = reference_engine
        .tokenize_fragment("Capital of France?")
        .unwrap();
    let suffix_b = reference_engine
        .tokenize_fragment("Capital of Japan?")
        .unwrap();
    let prompt_a: Vec<i32> = [shared.as_slice(), suffix_a.as_slice()].concat();

    // Cold references for both waves, on the engine that never sees the tree.
    let ref_a = reference_engine
        .generate_sampled_continuous(&[(&prompt_a, params(71, 6))])
        .expect("wave 1 reference");

    // Wave 2 depends on the model family: a dense model can reuse a
    // mid-sequence prefix (partial copy + replay for a fully cached prompt);
    // a hybrid only reuses exact boundaries, so its wave-2 prompt extends the
    // previous response, the multi-turn shape.
    let wave2: Vec<(Vec<i32>, SamplingParams)> = if recurrent {
        vec![(
            [
                prompt_a.as_slice(),
                &ref_a[0].tokens[..ref_a[0].tokens.len() - 1],
                suffix_b.as_slice(),
            ]
            .concat(),
            params(72, 6),
        )]
    } else {
        vec![
            ([shared.as_slice(), suffix_b.as_slice()].concat(), params(72, 6)),
            // The whole prompt is cached: only the last token is replayed.
            (shared.clone(), params(73, 6)),
        ]
    };
    let ref_b = reference_engine
        .generate_sampled_continuous(&wave2.iter().map(|(p, s)| (p.as_slice(), *s)).collect::<Vec<_>>())
        .expect("wave 2 reference");

    let actor_model = model.clone();
    let handle = ActorHandle::spawn(
        move || {
            Engine::new(&actor_model, &engine_config()).map_err(retrograd_infer::InferError::Backend)
        },
        actor_options(AdmissionMode::Lockstep),
    )
    .expect("actor spawns");

    // Wave 1: cold request; its finished slot becomes a prefix-tree node.
    let stream = handle
        .submit(Request::single(prompt_a.clone(), params(71, 6)))
        .expect("submit wave 1");
    let events: Vec<Event> = stream.collect();
    let (tokens, _logprobs, forced, reason) = summarize(&events, 0);
    assert_eq!(tokens, ref_a[0].tokens, "wave 1: tokens bit for bit");
    assert_eq!(forced, ref_a[0].forced, "wave 1: forced bit for bit");
    let expected_reason = if reference_engine
        .is_eog_token(*ref_a[0].tokens.last().expect("a token"))
        .unwrap()
    {
        FinishReason::Eog
    } else {
        FinishReason::MaxTokens
    };
    assert_eq!(reason, Some(expected_reason), "wave 1: finish reason");
    let stats_after_wave1 = handle.kv_stats().expect("stats");
    assert_eq!(stats_after_wave1.misses, 1);
    assert_eq!(stats_after_wave1.nodes, 1, "the finished slot is a dormant node");

    // Wave 2: shares the prefix with the node.
    let mut streams = Vec::new();
    for (prompt, sampling) in &wave2 {
        streams.push(
            handle
                .submit(Request::single(prompt.clone(), *sampling))
                .expect("submit wave 2"),
        );
    }
    for (index, stream) in streams.into_iter().enumerate() {
        let events: Vec<Event> = stream.collect();
        let (tokens, _logprobs, forced, reason) = summarize(&events, 0);
        assert_eq!(
            tokens, ref_b[index].tokens,
            "wave 2 request {index}: tokens bit for bit despite the cache hit"
        );
        assert_eq!(
            forced, ref_b[index].forced,
            "wave 2 request {index}: forced bit for bit"
        );
        let expected_reason = if reference_engine
            .is_eog_token(*ref_b[index].tokens.last().expect("a token"))
            .unwrap()
        {
            FinishReason::Eog
        } else {
            FinishReason::MaxTokens
        };
        assert_eq!(reason, Some(expected_reason), "wave 2 request {index}: finish reason");
    }

    let stats = handle.kv_stats().expect("stats");
    assert_eq!(
        stats.hits,
        wave2.len() as u64,
        "every wave-2 request hit the prefix cache: {stats:?}"
    );
    // Sortie criterion on the warm wave: the shared prefix is > 80 % of each
    // prompt, so prefilled/prompt after warm-up is well under 0.2.
    let warm_prompt = stats.prompt_cells - stats_after_wave1.prompt_cells;
    let warm_prefilled = stats.prefilled_cells - stats_after_wave1.prefilled_cells;
    assert!(
        warm_prefilled * 5 < warm_prompt,
        "prefilled/prompt < 0.2 on the shared load: prefilled={warm_prefilled} prompt={warm_prompt}"
    );

    // The actor's backend reports a stable key across the two waves (no weight
    // mutation happened), so the tree survived.
    assert_eq!(version_before, reference_engine.weights_version().unwrap());

    handle.shutdown();
    handle.join().expect("the actor drains cleanly");
}
