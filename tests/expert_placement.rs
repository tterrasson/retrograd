//! Expert placement: the bank inventory of a MoE model, the explicit
//! `host`/`device` placement of its expert banks through the engine, and the
//! device-cache telemetry. On one backend the placement must not change a
//! single routed id or token; the refusals (not a MoE, architecture not on the
//! validated list, cache without a GPU, cache under the routing floor) must be
//! typed errors, not runtime strings.
//!
//! The tiny MoE fixture (OLMoE, one bank set per layer) is the validated
//! architecture; the tiny dense fixture is the not-a-MoE refusal. GPU cases
//! are gated on Metal like `model_offload.rs`.

mod common;

use std::path::PathBuf;

use common::serialize_models;
use retrograd::{
    Device, Engine, EngineConfig, EngineKvDtype, EngineOptions, Error, ExpertPlacement,
    ExpertPlacementMode, SamplingParams, read_expert_banks,
};

fn moe_model() -> Option<PathBuf> {
    common::tiny_moe_model_path_if_available()
}

fn dense_model() -> Option<PathBuf> {
    common::tiny_model_path_if_available()
}

fn engine_config(device: Device) -> EngineConfig {
    EngineConfig {
        n_ctx: 128,
        n_seq_max: 2,
        n_batch: 32,
        n_ubatch: 32,
        kv_dtype: EngineKvDtype::F16,
        flash_attn: true,
        kv_unified: true,
        device,
        threads: 0,
        verbose: false,
    }
}

fn host_experts(host_layers: u32, device_cache_bytes: u64) -> ExpertPlacement {
    ExpertPlacement {
        mode: ExpertPlacementMode::Host,
        host_layers,
        device_cache_bytes,
    }
}

fn sampling() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 8,
        seed: 7,
    }
}

/// `expect_err` for `Engine`, which has no `Debug` impl.
fn expect_engine_err(
    result: Result<Engine, retrograd::EngineError>,
    message: &str,
) -> retrograd::EngineError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

fn generate_tokens(engine: &mut Engine, prompt: &str) -> Vec<i32> {
    let tokens = engine.tokenize_text(prompt).expect("tokenize");
    engine
        .generate(&tokens, &sampling())
        .expect("generate")
        .tokens
}

#[test]
fn the_tiny_moe_fixture_inventory_is_validated() {
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let banks = read_expert_banks(&model).expect("read expert banks");
    assert!(banks.moe, "the tiny MoE fixture has expert banks");
    assert!(
        banks.validated,
        "olmoe is on the validated architecture list"
    );
    // Two layers, one bank set of three roles each.
    let layers: std::collections::BTreeSet<u32> = banks.banks.iter().map(|b| b.layer).collect();
    assert_eq!(layers, [0, 1].into_iter().collect());
    assert_eq!(banks.banks.len(), 6);
    for bank in &banks.banks {
        assert!(
            matches!(bank.role.as_str(), "gate" | "up" | "down"),
            "role {}",
            bank.role
        );
        assert_eq!(bank.n_expert(), 8, "the fixture routes over 8 experts");
        assert!(bank.bytes_per_expert() > 0);
        assert!(bank.name.contains("_exps"));
    }
}

#[test]
fn a_dense_model_has_no_expert_banks_and_refuses_host_placement() {
    let Some(model) = dense_model() else {
        eprintln!("skipping: no tiny dense fixture");
        return;
    };
    let banks = read_expert_banks(&model).expect("read expert banks");
    assert!(!banks.moe);
    assert!(banks.banks.is_empty());

    let error = expect_engine_err(
        Engine::new_with_options(
            &model,
            &engine_config(Device::Cpu),
            &EngineOptions {
                experts: host_experts(0, 0),
            },
        ),
        "host placement on a dense model must be refused",
    );
    let error: Error = error.into();
    assert!(
        matches!(error, Error::InvalidArgument(_)),
        "a non-MoE model is a caller error, got: {error}"
    );
    assert!(error.to_string().contains("no expert banks"), "{error}");
    let mut engine = Engine::new(&model, &engine_config(Device::Cpu)).unwrap();
    let error = engine.set_expert_route_sampling(1).unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
}

#[test]
fn host_layers_beyond_the_model_are_refused() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let error = expect_engine_err(
        Engine::new_with_options(
            &model,
            &engine_config(Device::Cpu),
            &EngineOptions {
                experts: host_experts(99, 0),
            },
        ),
        "host_layers past the MoE layers must be refused",
    );
    let error: Error = error.into();
    assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
    assert!(
        error.to_string().contains("exceeds the 2 MoE layers"),
        "{error}"
    );
}

#[test]
fn device_placement_is_the_default_and_bit_identical() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let config = engine_config(Device::Cpu);
    let mut plain = Engine::new(&model, &config).expect("default engine");
    let mut explicit = Engine::new_with_options(
        &model,
        &config,
        &EngineOptions {
            experts: ExpertPlacement::device(),
        },
    )
    .expect("device placement engine");
    assert_eq!(
        generate_tokens(&mut plain, "The quick brown fox"),
        generate_tokens(&mut explicit, "The quick brown fox"),
        "device placement must be the historical behaviour"
    );
    // No cache was configured: the stats read zero.
    let stats = plain.expert_stats().expect("expert stats");
    assert_eq!(stats.groups.len(), 0);
    assert_eq!(stats.small.hits + stats.small.misses, 0);
}

#[test]
fn host_placement_on_cpu_matches_device_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let config = engine_config(Device::Cpu);
    let mut device = Engine::new(&model, &config).expect("device engine");
    let mut host = Engine::new_with_options(
        &model,
        &config,
        &EngineOptions {
            experts: host_experts(0, 0),
        },
    )
    .expect("host engine");
    // Same backend (CPU) on both: the placement must not change a single
    // token - identical ids, identical greedy path.
    assert_eq!(
        generate_tokens(&mut device, "Pack my box with"),
        generate_tokens(&mut host, "Pack my box with"),
    );
    // The engine's bank inventory is the path-based one.
    let from_engine = host.expert_banks().expect("engine expert banks");
    let from_path = read_expert_banks(&model).expect("path expert banks");
    assert_eq!(from_engine, from_path.banks);
}

#[test]
fn a_device_cache_without_a_gpu_is_refused() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let error = expect_engine_err(
        Engine::new_with_options(
            &model,
            &engine_config(Device::Cpu),
            &EngineOptions {
                experts: host_experts(0, 1 << 20),
            },
        ),
        "a device cache on a CPU context must be refused",
    );
    assert!(error.to_string().contains("requires a GPU"), "{error}");
}

#[test]
fn a_cache_on_device_placement_is_refused() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let error = expect_engine_err(
        Engine::new_with_options(
            &model,
            &engine_config(Device::Cpu),
            &EngineOptions {
                experts: ExpertPlacement {
                    mode: ExpertPlacementMode::Device,
                    host_layers: 0,
                    device_cache_bytes: 1 << 20,
                },
            },
        ),
        "a device cache with device placement must be refused",
    );
    assert!(error.to_string().contains("host placement"), "{error}");
}

/// Three banks, each with two routed slots plus one padding slot.
#[cfg(retro_metal)]
const FIXTURE_CACHE_FLOOR: u64 = 3 * 3 * 64 * 64 * 4;

#[cfg(retro_metal)]
#[test]
fn a_cache_under_the_routing_floor_is_refused_on_gpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let error = expect_engine_err(
        Engine::new_with_options(
            &model,
            &engine_config(Device::Gpu),
            &EngineOptions {
                experts: host_experts(0, FIXTURE_CACHE_FLOOR - 1),
            },
        ),
        "a cache under the floor must be refused before the load",
    );
    assert!(error.to_string().contains("does not cover"), "{error}");
}

#[cfg(retro_metal)]
#[test]
fn host_with_a_device_cache_on_gpu_matches_device_and_reports_stats() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let _guard = serialize_models();
    let Some(model) = moe_model() else {
        eprintln!("skipping: no tiny MoE fixture");
        return;
    };
    let config = engine_config(Device::Gpu);
    let mut device = Engine::new(&model, &config).expect("device engine on gpu");
    let cache_bytes = FIXTURE_CACHE_FLOOR;
    let mut host = Engine::new_with_options(
        &model,
        &config,
        &EngineOptions {
            experts: host_experts(0, cache_bytes),
        },
    )
    .expect("host + cache engine on gpu");

    let prompt = "The quick brown fox jumps over";
    let device_tokens = generate_tokens(&mut device, prompt);
    let host_tokens = generate_tokens(&mut host, prompt);
    // Same compute backend (GPU) on both sides: the cache changes where the
    // expert weights are read from, not what is computed - identical tokens.
    assert_eq!(device_tokens, host_tokens, "host+cache must match device");

    // The cache was actually used: at least one group, and every routed
    // expert was either a hit or a miss.
    let stats = host.expert_stats().expect("expert stats");
    assert!(!stats.groups.is_empty(), "the cache has a group: {stats:?}");
    let group = &stats.groups[0];
    assert!(
        group.n_slots >= 2,
        "at least n_expert_used slots: {group:?}"
    );
    assert_eq!(group.n_expert, 8);
    assert!(group.device_bytes > 0);
    let routed = stats.small.hits + stats.small.misses + stats.large.hits + stats.large.misses;
    assert!(routed > 0, "the decode routed through the cache: {stats:?}");
    assert_eq!(group.device_bytes, FIXTURE_CACHE_FLOOR);
}

/// Submit a single logical batch that spans several micro-batches, then feed
/// back greedy tokens so the test also sees a reused single-token graph.
fn route_probe(engine: &mut Engine, prompt_len: usize) -> Vec<(i32, u32)> {
    use retrograd::{StepOutput, StepRow};
    use retrograd_core::SamplingParamsV2;

    let prompt: Vec<_> = engine
        .tokenize_text("The quick brown fox")
        .expect("the fixture tokenizes")
        .into_iter()
        .cycle()
        .take(prompt_len)
        .collect();
    let mut sampling = SamplingParamsV2::from(sampling());
    sampling.top_k = 1;
    let mut row = StepRow::prefill(0, 0, &prompt, 0);
    row.sample = true;
    row.sampling = sampling.clone();
    row.n_top_logprobs = 1;
    let mut output = StepOutput::with_capacity(1, 1);
    engine.step(&[row], &mut output).expect("prefill steps");
    let mut token = output.row(0).token;
    let mut result = vec![(token, output.row(0).logprob.to_bits())];
    for pos in prompt_len..prompt_len + 3 {
        let mut row =
            StepRow::decode(0, 0, token, i32::try_from(pos).expect("pos fits in i32"), &sampling);
        row.n_top_logprobs = 1;
        row.want_logprob = true;
        engine.step(&[row], &mut output).expect("decode steps");
        token = output.row(0).token;
        result.push((token, output.row(0).logprob.to_bits()));
    }
    engine.seq_rm(0, -1, -1).expect("sequence removed");
    engine.request_free(0).expect("slot freed");
    result
}

fn assert_fixture_routes(counts: &retrograd::ExpertRouteCounts, tokens_by_layer: [u32; 2]) {
    assert_eq!(counts.banks.len(), 6);
    for bank in &counts.banks {
        // The concentrated fixture routes every token computed by this bank
        // to experts 0 and 1. OLMoE skips non-output rows in the final FFN.
        let routed_tokens = tokens_by_layer[usize::try_from(bank.bank.layer).expect("layer fits")];
        assert_eq!(
            bank.counts,
            [routed_tokens, routed_tokens, 0, 0, 0, 0, 0, 0],
            "{}",
            bank.bank.name
        );
    }
}

#[test]
fn cpu_routing_readback_counts_each_ubatch_and_preserves_greedy_output() {
    let _guard = serialize_models();
    let Some(model) = moe_model() else { return };
    let mut config = engine_config(Device::Cpu);
    config.n_ubatch = 3;
    for placement in [ExpertPlacement::device(), host_experts(0, 0)] {
        let mut engine =
            Engine::new_with_options(&model, &config, &EngineOptions { experts: placement })
                .unwrap();
        let baseline = route_probe(&mut engine, 9);
        assert!(engine.expert_route_counts().unwrap().banks.is_empty());
        engine.set_expert_route_sampling(1).unwrap();
        assert_eq!(route_probe(&mut engine, 9), baseline);
        let counts = engine.expert_route_counts().unwrap();
        assert_fixture_routes(&counts, [12, 4]);
        assert_eq!(counts.paths, [0, 0, 10]);
        engine.reset_expert_route_counts().unwrap();
        assert_fixture_routes(&engine.expert_route_counts().unwrap(), [0, 0]);

        engine.set_expert_route_sampling(2).unwrap();
        assert_eq!(route_probe(&mut engine, 9), baseline);
        let counts = engine.expert_route_counts().unwrap();
        // Widths 3,3,3,1,1,1; the first, third and fifth are sampled.
        assert_fixture_routes(&counts, [7, 2]);
        assert_eq!(counts.paths, [0, 0, 5]);
        engine.set_expert_route_sampling(0).unwrap();
        assert_eq!(route_probe(&mut engine, 9), baseline);
    }
}

#[cfg(retro_metal)]
#[test]
fn metal_routing_counts_cover_device_slots_and_large_copies_without_double_counting() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let _guard = serialize_models();
    let Some(model) = moe_model() else { return };
    // With only two slots, one token is always admitted. Width 4 exceeds
    // capacity and normally computes on the host. Width 32 also exceeds
    // capacity and reaches Metal's default host-weight offload threshold.
    for (width, prompt, experts, paths) in [
        (4, 8, ExpertPlacement::device(), [0, 0, 9]),
        (1, 8, host_experts(0, FIXTURE_CACHE_FLOOR), [15, 0, 0]),
        (32, 64, host_experts(0, FIXTURE_CACHE_FLOOR), [7, 2, 0]),
    ] {
        let mut config = engine_config(Device::Gpu);
        config.n_batch = 64;
        config.n_ubatch = width;
        let mut engine =
            Engine::new_with_options(&model, &config, &EngineOptions { experts }).unwrap();
        let baseline = route_probe(&mut engine, prompt);
        engine.set_expert_route_sampling(1).unwrap();
        assert_eq!(route_probe(&mut engine, prompt), baseline);
        let counts = engine.expert_route_counts().unwrap();
        assert_fixture_routes(&counts, [u32::try_from(prompt).unwrap() + 3, 4]);
        assert_eq!(counts.paths, paths);
    }
}
