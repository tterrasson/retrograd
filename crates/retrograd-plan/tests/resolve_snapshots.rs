//! Snapshot coverage of the resolver.
//!
//! A table of synthetic model geometries × budgets × dataset sizes, resolved and
//! compared against a recorded answer. No model, no GPU, no device.
//!
//! A resolution depends on the data, the budget, and the machine, so those are
//! the axes the table varies.
//!
//! The recorded answers live in `tests/snapshots/`. Regenerate them with
//! `RETRO_UPDATE_SNAPSHOTS=1 cargo test -p retrograd-plan`, and *read the diff*:
//! this is the file that turns a silent regression in the cost model into a
//! visible one, which is the whole reason it exists.

#![allow(clippy::unwrap_used)]
// Helpers outside a `#[test]` body, which is what `allow-unwrap-in-tests`
// covers. Same reasoning, said where the configuration cannot reach.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd_plan::recipe::{Allow, Objective, RewardRef, TrainingBudget};
use retrograd_plan::testing::{input, large_model, machine, tiny_model, uniform_dataset};
use retrograd_plan::{Recipe, resolve};
use serde::Serialize;
use serde_json::json;

const GIB: u64 = 1024 * 1024 * 1024;

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
}

/// Compares `value` against the recorded snapshot named `name`.
///
/// Serialized **straight from the typed value**, never through
/// `serde_json::Value`: `to_value` widens every `f32` to `f64`, which turns the
/// document's `lr = 1e-4` into `0.00009999999747378752`. That is not the number
/// the resolver chose, not the number the API returns (axum serializes the typed
/// response the same way this does), and not a diff anybody can read.
fn assert_snapshot<T: Serialize>(name: &str, value: &T) {
    let path = snapshot_dir().join(format!("{name}.json"));
    let rendered = format!("{}\n", serde_json::to_string_pretty(value).unwrap());
    if std::env::var_os("RETRO_UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshot_dir()).unwrap();
        std::fs::write(&path, &rendered).unwrap();
        return;
    }
    let recorded = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "missing snapshot {}: {error}\n\
             regenerate with RETRO_UPDATE_SNAPSHOTS=1 cargo test -p retrograd-plan",
            path.display()
        )
    });
    assert_eq!(
        recorded, rendered,
        "snapshot {name} changed; inspect the diff before regenerating"
    );
}

/// The parts of a resolution that form the contract: the effective document, the
/// provenance and the plan. Borrowed, so nothing is re-serialized on the way in.
#[derive(Serialize)]
struct Snapshot<'a> {
    config: &'a retrograd_config::ConfigDocument,
    plan: &'a retrograd_plan::PlanSummary,
    provenance: &'a retrograd_plan::Provenance,
}

fn render(resolution: &retrograd_plan::Resolution) -> Snapshot<'_> {
    Snapshot {
        config: &resolution.document,
        plan: &resolution.plan,
        provenance: &resolution.provenance,
    }
}

#[test]
fn an_sft_recipe_on_a_roomy_card() {
    let recipe = retrograd_plan::testing::sft_recipe();
    let params = json!({});
    let model = tiny_model();
    let data = uniform_dataset(4_000, 900);
    let resolution = resolve(&input(
        &recipe,
        &params,
        &model,
        &data,
        machine(24 * GIB, 64 * GIB),
    ))
    .expect("a small model in 24 GiB must resolve");

    // P99 selects the context without truncating this distribution.
    assert_eq!(resolution.config.training.n_ctx, 1024, "P99 of 900 tokens");
    assert_eq!(resolution.plan.truncation_fraction, 0.0);
    assert!(
        resolution.plan.levers.is_empty(),
        "nothing should be *needed* at this budget: {:?}",
        resolution.plan.levers
    );
    // …and yet a default is on, which is the whole point of phase 2bis: the
    // vocabulary logits are tiled even though this run would have fitted without
    assert!(
        resolution
            .plan
            .defaults_applied
            .iter()
            .any(|entry| entry.id == "chunked_cross_entropy"),
        "{:?}",
        resolution.plan.defaults_applied
    );
    assert!(resolution.config.training.chunked_cross_entropy);
    // Checkpointing, by contrast, is *not* on, and that is the fix rather than a
    // This run's activations are a rounding error against 24 GiB, so a long
    // context alone must not trigger an extra forward pass.
    assert!(!resolution.config.training.gradient_checkpointing);
    assert_snapshot("sft_roomy", &render(&resolution));
}

/// One snapshot per dataset family. `rank`, `lr`, `epochs` and the scheduler are
/// all derived from the amount of data, so this is the table that shows the derivation
/// rules acting together rather than one at a time.
#[test]
fn the_derivations_move_with_the_size_of_the_dataset() {
    let model = tiny_model();
    let params = json!({});
    #[derive(Serialize)]
    struct Row {
        examples: u64,
        rank: u32,
        alpha: f32,
        targets: String,
        epochs: u32,
        /// Kept as `f32` all the way to the writer: this is the number the
        /// engine gets, and widening it to `f64` here would record a different
        /// one.
        learning_rate: f32,
        weight_decay: f32,
        scheduler: &'static str,
        warmup_steps: u64,
        total_steps: u64,
    }
    let mut rendered: BTreeMap<String, Row> = BTreeMap::new();
    for (label, examples, tokens) in [
        ("1_tiny", 300u64, 300u32),
        ("2_small", 1_500, 700),
        ("3_medium", 12_000, 900),
        ("4_large", 250_000, 400),
    ] {
        let recipe = retrograd_plan::testing::sft_recipe();
        let data = uniform_dataset(examples, tokens);
        let resolution = resolve(&input(
            &recipe,
            &params,
            &model,
            &data,
            machine(24 * GIB, 64 * GIB),
        ))
        .expect("resolves");
        rendered.insert(
            label.to_string(),
            Row {
                examples,
                rank: resolution.config.lora.config.rank,
                alpha: resolution.config.lora.config.alpha,
                targets: format!("{:?}", resolution.config.lora.config.targets),
                epochs: resolution.config.training.epochs,
                learning_rate: resolution.config.training.learning_rate,
                weight_decay: resolution.config.training.weight_decay,
                scheduler: resolution.config.training.lr_scheduler.name(),
                warmup_steps: resolution.config.training.warmup_steps,
                total_steps: resolution.plan.total_steps,
            },
        );
    }
    assert_snapshot("datasets", &rendered);
}

#[test]
fn a_tight_budget_walks_down_the_lever_table() {
    let recipe = Recipe {
        allow: vec![Allow::TruncateContext],
        ..retrograd_plan::testing::sft_recipe()
    };
    let params = json!({});
    let model = tiny_model();
    let data = uniform_dataset(1_000, 3_500);
    let resolution = resolve(&input(
        &recipe,
        &params,
        &model,
        &data,
        // Barely more than the weights, the dequantization scratch and a
        // 4096-token F16 KV cache: every lever has to earn its place. The figure
        // came down when the F16 cache and the fused cross-entropy became
        // defaults. The budget remains below the fit point so this test still
        // exercises both opt-ins.
        machine(1100 * 1024 * 1024, 8 * GIB),
    ))
    .expect("with both opt-ins granted it must fit");

    assert!(
        !resolution.plan.levers.is_empty(),
        "a budget below the fixed floor cannot be free"
    );
    // Cheapest first: whatever was applied, the free levers come before the
    // ones that cost fidelity.
    let ids: Vec<&str> = resolution
        .plan
        .levers
        .iter()
        .map(|lever| lever.id)
        .collect();
    let first_costly = ids.iter().position(|id| matches!(*id, "context_length"));
    if let Some(index) = first_costly {
        assert!(
            ids[index..]
                .iter()
                .all(|id| matches!(*id, "context_length")),
            "a free lever was applied after a degrading one: {ids:?}"
        );
    }
    assert_snapshot("sft_tight_budget", &render(&resolution));
}

#[test]
fn a_grpo_recipe_resolves_its_rollout_geometry() {
    let recipe = Recipe {
        objective: Objective::ReasoningRl,
        budget: Some(TrainingBudget {
            epochs: None,
            updates: Some(40),
            minutes: None,
        }),
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..retrograd_plan::testing::sft_recipe()
    };
    let params = json!({});
    let model = tiny_model();
    let data = uniform_dataset(500, 700);
    let mut assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
    let resolution = resolve(&assembled).expect("resolves");

    let retrograd_config::Algorithm::Grpo(grpo) = &resolution.config.algorithm else {
        panic!("a reasoning-rl objective must resolve to GRPO");
    };
    assert_eq!(grpo.updates, 40, "the requested budget is honoured");
    assert_eq!(
        resolution.config.training.n_seq_max as usize, grpo.group_size,
        "the packed optimizer width is the group size"
    );
    assert!(resolution.config.training.generation_concurrency >= 1);
    assert!(
        resolution.config.training.generation_concurrency <= resolution.config.training.n_batch
    );
    // The sampling context is fast by default on a rollout, and the plan
    // says so in a code a client can match on.
    assert!(resolution.config.training.fast_generation_context);
    assert!(
        resolution
            .plan
            .warnings
            .iter()
            .any(|warning| warning.code == "sampling_distribution_approximated"),
        "{:?}",
        resolution.plan.warnings
    );
    assert_snapshot("grpo_rollout", &render(&resolution));
}

#[test]
fn a_model_that_cannot_fit_says_which_post_overflows() {
    let recipe = retrograd_plan::testing::sft_recipe();
    let params = json!({});
    let model = large_model();
    let data = uniform_dataset(1_000, 1_500);
    let error = resolve(&input(
        &recipe,
        &params,
        &model,
        &data,
        // The weights alone are 8 GiB.
        machine(4 * GIB, 16 * GIB),
    ))
    .expect_err("8 GiB of weights cannot fit in 4 GiB");

    match error {
        retrograd_plan::ResolveError::InsufficientMemory(report) => {
            assert!(!report.overflow.fits());
            let detail = report.detail();
            assert!(
                detail.contains("model_weight_bytes"),
                "the weights are what overflows here: {detail}"
            );
            assert!(
                report.unlocks.contains(&Allow::TruncateContext),
                "the answer must say what would unlock more"
            );
        }
        other => panic!("expected insufficient memory, got {other:?}"),
    }
}
