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
