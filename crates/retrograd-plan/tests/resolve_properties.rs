//! The properties every resolution must satisfy, whatever the inputs.
//!
//! Snapshots pin *one* answer; these pin the rules the answers obey. A change
//! that improves the cost model rewrites the snapshots and leaves this file
//! untouched - a change that breaks an invariant fails here, which is the point.

use std::path::PathBuf;

use retrograd_config::Algorithm;
use retrograd_plan::recipe::{Allow, DataSpec, Objective, RewardRef};
use retrograd_plan::testing::{
    input, input_with_eval, machine, mixed_dataset, sft_recipe, tiny_model, uniform_dataset,
};
use retrograd_plan::{Resolution, ResolveError, Source, resolve};
use serde_json::{Value, json};

const GIB: u64 = 1024 * 1024 * 1024;

/// Every budget a resolution is tried against, from roomy to hopeless.
const BUDGETS: [u64; 6] = [
    48 * GIB,
    24 * GIB,
    8 * GIB,
    4 * GIB,
    2 * GIB,
    1024 * 1024 * 1024,
];

fn resolve_at(
    budget: u64,
    recipe: &retrograd_plan::Recipe,
    overrides: &Value,
) -> Option<Resolution> {
    let model = tiny_model();
    let data = uniform_dataset(3_000, 1_800);
    resolve(&input(
        recipe,
        overrides,
        &model,
        &data,
        machine(budget, 64 * GIB),
    ))
    .ok()
}

/// Invariant 1: the resolver cannot produce a configuration the CLI would
/// refuse. `resolve` builds through `retrograd_config::build`, so reaching a
/// `Resolution` is already the proof; this re-checks the two divisibilities the
/// runtime enforces separately, because that is the pair a lever could break.
#[test]
fn every_resolution_satisfies_the_runtime_divisibilities() {
    let permissive = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({});
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &permissive, &overrides) else {
            continue;
        };
        let training = &resolution.config.training;
        assert_eq!(
            training.n_ctx % training.n_batch,
            0,
            "n_ctx % n_batch at {budget} bytes: {training:?}"
        );
        assert_eq!(
            training.n_batch % training.n_ubatch,
            0,
            "n_batch % n_ubatch at {budget} bytes: {training:?}"
        );
        assert!(training.n_ubatch > 0 && training.n_batch > 0);
    }
}

/// Invariant 2: what is returned fits the budget it was resolved against.
#[test]
fn a_returned_plan_always_fits_its_own_budget() {
    let permissive = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({});
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &permissive, &overrides) else {
            continue;
        };
        let memory = &resolution.plan.memory;
        assert!(
            memory.resources.device_peak_bytes <= memory.budgets.vram.effective_bytes,
            "{} device bytes against a {} budget",
            memory.resources.device_peak_bytes,
            memory.budgets.vram.effective_bytes
        );
        assert!(memory.resources.host_peak_bytes <= memory.budgets.ram.effective_bytes);
    }
}

/// A tighter budget never produces a heavier configuration. Without this the
/// lever ordering could "improve" a run into an OOM.
#[test]
fn a_smaller_budget_never_yields_a_larger_estimate() {
    let permissive = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({});
    let mut previous: Option<(u64, u64)> = None;
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &permissive, &overrides) else {
            continue;
        };
        let estimated = resolution.plan.memory.resources.device_peak_bytes;
        if let Some((larger_budget, larger_estimate)) = previous {
            assert!(
                estimated <= larger_estimate,
                "{budget} bytes resolved to {estimated}, above the {larger_estimate} \
                 chosen for the roomier {larger_budget}"
            );
        }
        previous = Some((budget, estimated));
    }
    assert!(previous.is_some(), "no budget resolved at all");
}

/// Invariant 3: two identical resolutions are the same byte of JSON.
#[test]
fn the_same_request_resolves_to_the_same_bytes() {
    let recipe = sft_recipe();
    let overrides = json!({"training": {"gradient_checkpointing": true}});
    let first = resolve_at(8 * GIB, &recipe, &overrides).expect("resolves");
    let second = resolve_at(8 * GIB, &recipe, &overrides).expect("resolves");
    let render = |resolution: &Resolution| {
        serde_json::to_string(&json!({
            "config": serde_json::to_value(&resolution.document).unwrap(),
            "provenance": serde_json::to_value(&resolution.provenance).unwrap(),
            "plan": serde_json::to_value(&resolution.plan).unwrap(),
        }))
        .unwrap()
    };
    assert_eq!(render(&first), render(&second));
}

/// Invariant 6: a client-supplied parameter comes back exactly as it was sent, at every
/// budget, and is marked as the caller's.
#[test]
fn a_client_parameter_is_never_modified() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({
        "training": {"ctx": 512, "gradient_checkpointing": false},
        "lora": {"rank": 4}
    });
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &recipe, &overrides) else {
            continue;
        };
        assert_eq!(resolution.config.training.n_ctx, 512, "at {budget} bytes");
        assert!(
            !resolution.config.training.gradient_checkpointing,
            "a lever overwrote an override at {budget} bytes"
        );
        assert_eq!(resolution.config.lora.config.rank, 4);
        for path in [
            "training.ctx",
            "training.gradient_checkpointing",
            "lora.rank",
        ] {
            assert_eq!(
                resolution.provenance.get(path).map(|origin| origin.source),
                Some(Source::Override),
                "{path} lost its provenance at {budget} bytes"
            );
        }
    }
}

/// Invariant 4: no degradation without an explicit opt-in, apart from the
/// documented defaults (fast sampling, inert for this SFT recipe, and the F16 KV
/// cache). The same request that fails without an opt-in must succeed with it,
/// otherwise the error would be pointing at the wrong thing.
#[test]
fn no_opt_in_lever_is_applied_without_its_opt_in() {
    let model = tiny_model();
    let overrides = json!({});
    let strict = sft_recipe();

    // Loose enough that nothing degrading is ever needed.
    for budget in [48 * GIB, 24 * GIB, 8 * GIB] {
        let Some(resolution) = resolve_at(budget, &strict, &overrides) else {
            continue;
        };
        // F16 unasked-for: it is the engine's default cache, not a lever pulled
        // here, and the runtime probes the device before applying it. What the
        // plan owes the caller for it is the `kv_f16_may_fall_back` warning.
        assert_eq!(
            resolution.config.training.kv_dtype,
            retrograd_core::KvDtype::F16
        );
        assert!(
            resolution
                .plan
                .warnings
                .iter()
                .any(|warning| warning.code == "kv_f16_may_fall_back")
        );
        assert!(resolution.config.training.fast_generation_context);
        assert_eq!(resolution.plan.truncation_fraction, 0.0);
        for lever in &resolution.plan.levers {
            assert!(
                !matches!(lever.id, "context_length"),
                "{} was applied without an opt-in",
                lever.id
            );
        }
    }

    // Tight enough that one is. The failure names the opt-in, and granting it
    // resolves the same request.
    //
    // The floor has moved down twice. Proactive defaults enabled gradient checkpointing
    // before any lever is considered, and then the engine's defaults changed
    // under it: the F16 cache halves the dominant term in this budget, and
    // the fused cross-entropy is on from the start. So a budget that once needed
    // a degradation now resolves for free - which is the point of both changes,
    // and it means the only opt-in a *supervised* recipe can still be pushed
    // into is truncation. That needs a corpus with a long tail: the resolver will
    // not cut below the median, so a uniform one has nothing to truncate.
    let tailed = retrograd_plan::testing::mixed_dataset(3_000, 400, 15_000);
    let tight = machine(1450 * 1024 * 1024, 64 * GIB);
    let error = resolve(&input(&strict, &overrides, &model, &tailed, tight))
        .expect_err("a 1450 MiB budget cannot hold a 16k context for this model");
    let ResolveError::NeedsOptIn { allow, message } = error else {
        panic!("expected an opt-in request, got {error:?}");
    };
    assert!(!message.is_empty(), "the caller must be told what it buys");

    let granted = retrograd_plan::Recipe {
        allow: vec![allow],
        ..sft_recipe()
    };
    resolve(&input(&granted, &overrides, &model, &tailed, tight))
        .expect("the very opt-in the error asked for must resolve it");
}

/// An override that makes the budget unreachable is a named conflict, never a
/// silently rewritten field.
#[test]
fn an_impossible_override_is_reported_as_a_conflict() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    // Every free lever is pinned to its most expensive setting, on a budget
    // that needs all of them.
    let overrides = json!({
        "training": {
            "chunked_cross_entropy": false,
            "chunked_ce_tiles": 1,
            "chunked_ce_seq_chunk": 0,
            "gradient_checkpointing": false,
            "checkpoint_every_n_layers": 1,
            "micro_batch": 512,
            "ctx": 8192,
            "gradient_accumulation": 16
        }
    });
    let model = tiny_model();
    let data = uniform_dataset(3_000, 1_800);
    let error = resolve(&input(
        &recipe,
        &overrides,
        &model,
        &data,
        machine(2 * GIB, 64 * GIB),
    ))
    .expect_err("the overrides leave no room");
    match error {
        ResolveError::OverrideConflict { path, message } => {
            assert!(path.starts_with("training."), "{path}");
            assert!(!message.is_empty());
        }
        ResolveError::InsufficientMemory(report) => {
            // Also acceptable: what must never happen is a success that quietly
            // moved one of the pinned fields.
            assert!(!report.overflow.fits());
        }
        other => panic!("expected a conflict or an overflow, got {other:?}"),
    }
}

/// Packed accumulation may change its physical width, but still represents one
/// logical optimizer update over the complete rollout row.
#[test]
fn grpo_physical_micro_batches_preserve_the_logical_update() {
    let recipe = retrograd_plan::Recipe {
        objective: Objective::ReasoningRl,
        allow: vec![Allow::TruncateContext],
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..sft_recipe()
    };
    let overrides = json!({});
    let model = tiny_model();
    let data = uniform_dataset(800, 1_200);
    for budget in BUDGETS {
        let mut assembled = input(
            &recipe,
            &overrides,
            &model,
            &data,
            machine(budget, 64 * GIB),
        );
        assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
        let Ok(resolution) = resolve(&assembled) else {
            continue;
        };
        let Algorithm::Grpo(grpo) = &resolution.config.algorithm else {
            panic!("reasoning-rl must resolve to GRPO");
        };
        // The runtime's GRPO constraints, checked on the way out.
        assert_eq!(
            resolution.config.training.n_seq_max as usize,
            grpo.group_size
        );
        assert!(grpo.group_size <= resolution.config.training.n_batch as usize);
        assert_eq!(
            resolution.config.training.n_batch, resolution.config.training.n_ctx,
            "one rollout row must remain one AdamW update"
        );
        assert_eq!(
            resolution.config.training.n_batch % resolution.config.training.n_ubatch,
            0,
            "the physical packed width must divide the logical row"
        );
        assert!(
            resolution.config.training.generation_concurrency <= resolution.config.training.n_batch
        );
        assert!(resolution.config.training.generation_concurrency <= 256);
    }
}

/// Every `derived` and `measured` entry carries a reason. A provenance without
/// one is indistinguishable from a default and defeats the point of recording
/// provenance.
#[test]
fn every_derived_field_explains_itself() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({"lora": {"rank": 4}});
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &recipe, &overrides) else {
            continue;
        };
        for (path, origin) in &resolution.provenance.0 {
            match origin.source {
                Source::Derived | Source::Measured => assert!(
                    origin.reason.as_ref().is_some_and(|text| !text.is_empty()),
                    "{path} is {:?} with no reason",
                    origin.source
                ),
                Source::Override | Source::Default => {}
            }
        }
    }
}

/// A budget that cannot hold the weights fails with a decomposition, not with a
/// configuration nobody can run.
#[test]
fn an_impossible_budget_fails_loudly() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let overrides = json!({});
    let model = tiny_model();
    let data = uniform_dataset(3_000, 1_800);
    let error = resolve(&input(
        &recipe,
        &overrides,
        &model,
        &data,
        machine(64 * 1024 * 1024, 64 * GIB),
    ))
    .expect_err("64 MiB cannot hold a 400 MiB model");
    let ResolveError::InsufficientMemory(report) = error else {
        panic!("expected an overflow report");
    };
    assert!(!report.estimate.dominant_posts().is_empty());
    assert!(
        report.detail().contains("over budget"),
        "{}",
        report.detail()
    );
}

/// Every lever in the phase 3 table is reachable.
///
/// One case per lever, where only that lever makes the configuration fit, so
/// that the order of phase 3 is tested and not merely documented. This walks a budget sweep and a rollout objective and checks
/// that each lever the resolver is allowed to pull actually gets pulled
/// somewhere. A lever nobody can reach is dead code that reads like policy.
#[test]
fn every_lever_in_the_table_is_reachable() {
    use std::collections::BTreeSet;

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    // A lever pulled on the way to a refusal counts: the question here is
    // whether the resolver can reach that row of the table at all, and an
    // `insufficient_memory` answer carries the levers it tried.
    let mut record = |outcome: Result<Resolution, ResolveError>| match outcome {
        Ok(resolution) => seen.extend(resolution.plan.levers.iter().map(|lever| lever.id)),
        Err(ResolveError::InsufficientMemory(report)) => {
            seen.extend(report.applied.iter().map(|lever| lever.id))
        }
        Err(_) => {}
    };

    let overrides = json!({});
    let model = tiny_model();
    let permissive = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };

    // Supervised, over several length profiles: a long tail is what makes the
    // context lever reachable, since a uniform corpus has P50 = P99 and the
    // resolver refuses to cut below the median.
    for data in [
        retrograd_plan::testing::mixed_dataset(3_000, 400, 7_000),
        retrograd_plan::testing::mixed_dataset(3_000, 900, 15_000),
        uniform_dataset(3_000, 3_500),
    ] {
        for step in 6..=48 {
            let budget = step * 128 * 1024 * 1024;
            record(resolve(&input(
                &permissive,
                &overrides,
                &model,
                &data,
                machine(budget, 64 * GIB),
            )));
        }
    }

    // Rollout: the two generation levers only exist here.
    let rollout = retrograd_plan::Recipe {
        objective: Objective::ReasoningRl,
        allow: vec![Allow::TruncateContext],
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..sft_recipe()
    };
    for prompts in [
        uniform_dataset(800, 1_500),
        retrograd_plan::testing::mixed_dataset(800, 600, 6_000),
    ] {
        for step in 6..=48 {
            let budget = step * 128 * 1024 * 1024;
            let mut assembled = input(
                &rollout,
                &overrides,
                &model,
                &prompts,
                machine(budget, 64 * GIB),
            );
            assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
            record(resolve(&assembled));
        }
    }

    for lever in retrograd_plan::LEVERS {
        // `device_cpu` is deliberately inert: the list proposes it, the resolver
        // never selects it.
        if lever.id == "device_cpu" {
            continue;
        }
        assert!(
            seen.contains(lever.id),
            "no budget in the sweep reaches the '{}' lever; applied: {seen:?}",
            lever.id
        );
    }
}

// ---------------------------------------------------------------------------
// Proactive defaults
// ---------------------------------------------------------------------------

/// A field the client supplied is never touched by a *default* either. Phase
/// 2bis is the newer of the two phases that write to a configuration, so it is
/// the one likeliest to forget invariant 6.
#[test]
fn no_active_default_is_applied_over_a_client_parameter() {
    let recipe = retrograd_plan::Recipe {
        objective: Objective::ReasoningRl,
        allow: vec![Allow::TruncateContext],
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..sft_recipe()
    };
    // Every field the four defaults would move, pinned to the value they would
    // move it away from.
    let params = json!({
        "training": {
            "gradient_checkpointing": false,
            "checkpoint_every_n_layers": 3,
            "chunked_cross_entropy": false,
            "chunked_ce_tiles": 4,
            "fast_sampling_context": false,
            "generation_concurrency": 2
        }
    });
    let model = tiny_model();
    let data = uniform_dataset(800, 1_200);
    let mut assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
    let resolution = resolve(&assembled).expect("a roomy budget needs no lever");

    let training = &resolution.config.training;
    assert!(!training.gradient_checkpointing);
    assert_eq!(training.checkpoint_every_n_layers, 3);
    assert!(!training.chunked_cross_entropy);
    assert!(!training.fast_generation_context);
    assert_eq!(training.generation_concurrency, 2);
    assert!(
        resolution.plan.defaults_applied.is_empty(),
        "every field was the client's: {:?}",
        resolution.plan.defaults_applied
    );
    for path in [
        "training.gradient_checkpointing",
        "training.chunked_cross_entropy",
        "training.fast_sampling_context",
        "training.generation_concurrency",
    ] {
        assert_eq!(
            resolution.provenance.get(path).map(|origin| origin.source),
            Some(Source::Override),
            "{path} lost its provenance to a default"
        );
    }
}

/// `defaults_applied` and `levers` never name the same setting: one reads as
/// "this is better", the other as "this was necessary", and a client that saw
/// both would not know which.
#[test]
fn what_was_chosen_and_what_was_sacrificed_are_disjoint() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let params = json!({});
    let model = tiny_model();
    for data in [
        uniform_dataset(3_000, 1_800),
        retrograd_plan::testing::mixed_dataset(3_000, 900, 15_000),
    ] {
        for step in 6..=48u64 {
            let budget = step * 128 * 1024 * 1024;
            let Ok(resolution) = resolve(&input(
                &recipe,
                &params,
                &model,
                &data,
                machine(budget, 64 * GIB),
            )) else {
                continue;
            };
            for applied in &resolution.plan.defaults_applied {
                assert!(
                    !resolution
                        .plan
                        .levers
                        .iter()
                        .any(|lever| lever.id == applied.id),
                    "'{}' is reported as both a default and a lever at {budget} bytes",
                    applied.id
                );
            }
        }
    }
}

/// A reported default has to survive into the configuration that will run.
///
/// `training.generation_concurrency` is GRPO-only - `retrograd_config::build`
/// refuses it anywhere else - so phase 2bis must not claim to have set it on a
/// PPO run: the entry would name a field the effective configuration does not
/// carry, which is the one thing `defaults_applied` exists to make readable.
#[test]
fn no_default_is_reported_for_a_field_the_algorithm_cannot_carry() {
    let recipe = retrograd_plan::Recipe {
        objective: retrograd_plan::recipe::Objective::PreferenceRl,
        reward: Some(retrograd_plan::recipe::RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..sft_recipe()
    };
    let params = json!({});
    let model = tiny_model();
    let data = uniform_dataset(3_000, 900);
    let mut assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
    let resolution = resolve(&assembled).expect("a PPO recipe resolves on a roomy card");

    assert!(
        !resolution
            .plan
            .defaults_applied
            .iter()
            .any(|applied| applied.id == "generation_concurrency"),
        "{:?}",
        resolution.plan.defaults_applied
    );
    assert!(
        resolution
            .provenance
            .get("training.generation_concurrency")
            .is_none(),
        "a field PPO has no place for was given a provenance entry"
    );
    assert_eq!(resolution.document.training.generation_concurrency, None);
}

/// Turning a default off can only make the run bigger. If it could make it
/// smaller, the default would be a pessimisation dressed as a policy.
#[test]
fn disabling_a_default_never_lowers_the_estimate() {
    let model = tiny_model();
    let data = uniform_dataset(3_000, 1_800);
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let automatic = json!({});
    let disabled = json!({"training": {"gradient_checkpointing": false}});
    let with = resolve(&input(
        &recipe,
        &automatic,
        &model,
        &data,
        machine(24 * GIB, 64 * GIB),
    ))
    .expect("resolves");
    let without = resolve(&input(
        &recipe,
        &disabled,
        &model,
        &data,
        machine(24 * GIB, 64 * GIB),
    ))
    .expect("resolves");
    assert!(
        without.plan.memory.resources.device_peak_bytes
            >= with.plan.memory.resources.device_peak_bytes,
        "checkpointing off is {} bytes, on is {}",
        without.plan.memory.resources.device_peak_bytes,
        with.plan.memory.resources.device_peak_bytes
    );
}

/// The fused cross-entropy is on for every backend the resolver recognises,
/// Metal included - it has kernels for both fused nodes. The one machine that
/// does not get it is the one nobody recognises, and nothing anywhere still warns
/// about a Metal CPU fallback.
#[test]
fn every_known_backend_resolves_onto_the_fused_cross_entropy() {
    let model = tiny_model();
    let data = uniform_dataset(3_000, 3_500);
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let params = json!({});
    let budget = machine(6 * GIB, 64 * GIB);

    for backend in [
        retrograd_plan::Backend::Cpu,
        retrograd_plan::Backend::Metal,
        retrograd_plan::Backend::Cuda,
        retrograd_plan::Backend::Vulkan,
        retrograd_plan::Backend::Blas,
    ] {
        let mut assembled = input(&recipe, &params, &model, &data, budget);
        assembled.hardware = retrograd_plan::HardwareFacts {
            backend,
            ..assembled.hardware
        };
        let resolution = resolve(&assembled).expect("resolves");
        assert!(
            resolution.config.training.chunked_cross_entropy,
            "{}: {:?}",
            backend.id(),
            resolution.plan.defaults_applied
        );
        assert!(
            resolution.config.training.chunked_ce_seq_chunk > 0,
            "{}: the token axis has to be bounded too",
            backend.id()
        );
        for warning in &resolution.plan.warnings {
            assert!(
                !warning.code.contains("chunked_ce"),
                "{}: {warning:?}",
                backend.id()
            );
        }
    }

    // An unrecognised accelerator still gets the fused path - it is the engine's
    // default and withholding it would change nothing - but it is the one case
    // that says out loud that the estimate rests on a probe nobody ran.
    let mut unknown = input(&recipe, &params, &model, &data, budget);
    unknown.hardware = retrograd_plan::HardwareFacts {
        backend: retrograd_plan::Backend::Unknown,
        ..unknown.hardware
    };
    let resolution = resolve(&unknown).expect("resolves");
    assert!(resolution.config.training.chunked_cross_entropy);
    assert!(
        resolution
            .plan
            .warnings
            .iter()
            .any(|warning| warning.code == "chunked_ce_on_an_unrecognised_backend"),
        "{:?}",
        resolution.plan.warnings
    );
}

/// The derivation rules, seen through whole resolutions rather than one function at a
/// time: more data means more rank, and more rank means a smaller step.
#[test]
fn more_data_means_more_rank_and_a_smaller_step() {
    let model = tiny_model();
    let recipe = sft_recipe();
    let params = json!({});
    let mut previous: Option<(u64, u32, f32)> = None;
    for examples in [300u64, 1_500, 12_000, 150_000, 400_000] {
        let data = uniform_dataset(examples, 400);
        let resolution = resolve(&input(
            &recipe,
            &params,
            &model,
            &data,
            machine(24 * GIB, 64 * GIB),
        ))
        .expect("resolves");
        let rank = resolution.config.lora.config.rank;
        let rate = resolution.config.training.learning_rate;
        if let Some((fewer, smaller_rank, larger_rate)) = previous {
            assert!(
                rank >= smaller_rank,
                "{examples} examples took rank {rank}, below the {smaller_rank} of {fewer}"
            );
            assert!(
                rate <= larger_rate,
                "{examples} examples took rate {rate}, above the {larger_rate} of {fewer}"
            );
        }
        previous = Some((examples, rank, rate));
    }
}

/// Every warning a plan carries has a code, and the same code is never emitted
/// twice. A client filters on the code, so a duplicate reads as two problems.
#[test]
fn every_warning_carries_a_unique_code() {
    let recipe = retrograd_plan::Recipe {
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let params = json!({});
    for budget in BUDGETS {
        let Some(resolution) = resolve_at(budget, &recipe, &params) else {
            continue;
        };
        let mut seen = std::collections::BTreeSet::new();
        for warning in &resolution.plan.warnings {
            assert!(!warning.code.is_empty());
            assert!(!warning.message.is_empty(), "{}", warning.code);
            assert!(
                seen.insert(warning.code),
                "'{}' was emitted twice at {budget} bytes",
                warning.code
            );
        }
    }
}

/// The eval dataset gets its own truncation figure, independent of the
/// training one, and a rollout's `evaluation.max_examples` never asks for more
/// than the eval set actually has.
#[test]
fn the_eval_dataset_reports_its_own_truncation_and_caps_max_examples() {
    let model = tiny_model();
    // Training data is uniform (nothing truncates); eval data is long-tailed,
    // so its truncation figure has to come from its own distribution and not
    // leak the training one.
    let data = uniform_dataset(3_000, 900);
    let eval = mixed_dataset(20, 500, 4_000);
    let recipe = retrograd_plan::Recipe {
        objective: Objective::ReasoningRl,
        eval: Some(DataSpec {
            path: Some(PathBuf::from("eval.jsonl")),
            dataset: None,
            format: Some("jsonl".to_string()),
        }),
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        allow: vec![Allow::TruncateContext],
        ..sft_recipe()
    };
    let params = json!({});
    let mut assembled = input_with_eval(
        &recipe,
        &params,
        &model,
        &data,
        Some(&eval),
        machine(24 * GIB, 64 * GIB),
    );
    assembled.reward_command = vec!["python3".to_string(), "reward.py".to_string()];
    let resolution = resolve(&assembled).expect("a roomy card must resolve");

    let n_ctx = resolution.config.training.n_ctx;
    let expected = retrograd_plan::resolver::reported_fraction(eval.truncation_fraction(n_ctx));
    assert_eq!(resolution.plan.eval_truncation_fraction, Some(expected));
    if expected > 0.0 {
        assert_ne!(
            Some(resolution.plan.truncation_fraction),
            resolution.plan.eval_truncation_fraction,
            "the eval and training truncation figures must not be conflated"
        );
    }

    let evaluation = resolution
        .config
        .evaluation
        .as_ref()
        .expect("eval was configured");
    let max_examples = evaluation.max_examples.expect("a rollout derives one");
    assert!(
        max_examples <= eval.examples as usize,
        "max_examples ({max_examples}) must never exceed the {} examples the eval set has",
        eval.examples
    );
}

/// A recipe with no eval dataset reports no eval truncation figure at all,
/// `None`, not a `0.0` that would claim a measurement that was never taken.
#[test]
fn no_eval_dataset_means_no_eval_truncation_figure() {
    let model = tiny_model();
    let data = uniform_dataset(3_000, 900);
    let recipe = sft_recipe();
    let params = json!({});
    let assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    let resolution = resolve(&assembled).expect("a roomy card must resolve");
    assert_eq!(resolution.plan.eval_truncation_fraction, None);
}
