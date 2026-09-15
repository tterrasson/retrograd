//! The derivation table and the resolver cannot drift apart.
//!
//! `GET /v1/defaults` is generated from [`retrograd_plan::DERIVATIONS`] and
//! [`retrograd_plan::ACTIVE_DEFAULTS`], not recopied from the document. That only
//! buys anything if the tables describe what the resolver really does - a row
//! for a field nobody derives is a promise the server does not keep, and a
//! derived field with no row is a decision a client cannot anticipate.
//!
//! So: resolve one recipe per algorithm, with held-out data and a checkpoint
//! directory so every optional section exists, and compare the provenance map
//! against the table in both directions.

use std::collections::BTreeSet;
use std::path::PathBuf;

use retrograd_plan::recipe::{DataSpec, Objective, RewardRef};
use retrograd_plan::testing::{execution_profile, input, machine, tiny_model, uniform_dataset};
use retrograd_plan::tuning::Scope;
use retrograd_plan::{DERIVATIONS, Provenance, Recipe, resolve};
use serde_json::json;

const GIB: u64 = 1024 * 1024 * 1024;

/// A recipe with every optional section filled, so no row of the table is
/// skipped for want of a dataset to evaluate on or a directory to write to.
fn complete(objective: Objective) -> Recipe {
    Recipe {
        objective,
        eval: Some(DataSpec {
            path: Some(PathBuf::from("eval.jsonl")),
            dataset: None,
            format: Some("jsonl".to_string()),
        }),
        checkpoint_dir: Some(PathBuf::from("checkpoints")),
        reward: objective.is_rollout().then(|| RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..retrograd_plan::testing::sft_recipe()
    }
}

fn resolve_complete(objective: Objective) -> Provenance {
    let recipe = complete(objective);
    let params = json!({});
    let model = tiny_model();
    let data = uniform_dataset(4_000, 900);
    // With a profile, as every server resolution has one: rows that key on a
    // published capability - the packed fanout - are only decided by the planner
    // when the engine has answered, and would otherwise read as stale table
    // entries.
    let profile = execution_profile(true);
    let mut assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    assembled.execution_profile = Some(&profile);
    if objective.is_rollout() {
        assembled.reward_command = vec!["python".to_string(), "reward.py".to_string()];
    }
    resolve(&assembled)
        .unwrap_or_else(|error| panic!("{objective:?} must resolve on a roomy card: {error:?}"))
        .provenance
}

/// Whether a row is expected to appear for this algorithm. Every recipe here
/// carries held-out data and a checkpoint directory, so those two scopes always
/// apply.
fn applies(scope: Scope, objective: Objective) -> bool {
    let algorithm = objective.algorithm();
    match scope {
        Scope::Any | Scope::Evaluation | Scope::Checkpoint => true,
        Scope::Sft => algorithm == "sft",
        // `Scope::Grpo` is documented as covering `agentic` too, which also
        // resolves to a GRPO objective.
        Scope::Grpo => algorithm == "grpo" || algorithm == "agent_grpo",
        Scope::Ppo => algorithm == "ppo",
        Scope::Rollout => objective.is_rollout(),
    }
}

const OBJECTIVES: [Objective; 3] = [
    Objective::InstructionTuning,
    Objective::ReasoningRl,
    Objective::PreferenceRl,
];

/// Every row of the table names a field the resolver really decides.
#[test]
fn every_documented_rule_shows_up_in_a_real_resolution() {
    for objective in OBJECTIVES {
        let provenance = resolve_complete(objective);
        for derivation in DERIVATIONS {
            if !applies(derivation.scope, objective) {
                continue;
            }
            assert!(
                provenance.get(derivation.path).is_some(),
                "GET /v1/defaults documents '{}' for {objective:?}, but no resolution \
                 records it; either the rule moved or the table is stale",
                derivation.path
            );
        }
    }
}

/// And every field the resolver decides on the client's behalf has a row.
///
/// The exclusions are the two phases that describe themselves elsewhere: what
/// phase 2bis turned on is listed by `ACTIVE_DEFAULTS`, and what phase 3 had to
/// sacrifice is in `plan.levers` with its own cost - neither is a rule a client
/// can read off in advance, which is what this table is for.
#[test]
fn every_derived_field_is_documented() {
    let from_defaults: BTreeSet<&str> = retrograd_plan::ACTIVE_DEFAULTS
        .iter()
        .flat_map(|entry| entry.touches.iter().copied())
        .collect();
    let from_levers: BTreeSet<&str> = retrograd_plan::LEVERS
        .iter()
        .flat_map(|lever| lever.touches.iter().copied())
        .collect();
    let documented: BTreeSet<&str> = DERIVATIONS
        .iter()
        .map(|derivation| derivation.path)
        .collect();

    for objective in OBJECTIVES {
        let provenance = resolve_complete(objective);
        for (path, origin) in &provenance.0 {
            if from_defaults.contains(path.as_str()) || from_levers.contains(path.as_str()) {
                continue;
            }
            assert!(
                documented.contains(path.as_str()),
                "{objective:?} decided '{path}' ({origin:?}) and GET /v1/defaults says \
                 nothing about it"
            );
        }
    }
}

/// A row whose scope claims an algorithm that does not have the field would make
/// the endpoint describe an impossible parameter.
#[test]
fn a_row_scoped_to_an_algorithm_never_appears_under_another() {
    for objective in OBJECTIVES {
        let provenance = resolve_complete(objective);
        for derivation in DERIVATIONS {
            if applies(derivation.scope, objective) {
                continue;
            }
            // A narrower scope may still be a *subset* of what is recorded - a
            // field both algorithms happen to carry. What must not happen is a
            // path that belongs to the other algorithm's section.
            let other_section = ["grpo.", "ppo.", "sft."]
                .iter()
                .any(|prefix| derivation.path.starts_with(prefix));
            if !other_section {
                continue;
            }
            assert!(
                provenance.get(derivation.path).is_none(),
                "'{}' is scoped away from {objective:?} yet a {objective:?} resolution \
                 records it",
                derivation.path
            );
        }
    }
}
