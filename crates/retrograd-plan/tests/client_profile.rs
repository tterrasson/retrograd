//! A profile, client-side: a GRPO configuration amputated of everything the
//! server can work out on its own - no `[run]`, no `[model]`, no paths.
//!
//! The claim a profile makes is strong: post this and the server changes
//! nothing in it. That is worth a test rather than a paragraph - a rule added to
//! [`retrograd_plan::tuning`] that forgot to check the lock would break the
//! promise silently, and a profile is what a client copies.

use std::path::PathBuf;

use retrograd_plan::recipe::{DataSpec, Objective, RewardRef};
use retrograd_plan::testing::{input, machine, tiny_model, uniform_dataset};
use retrograd_plan::{Recipe, Source, resolve};
use serde_json::Value;

const GIB: u64 = 1024 * 1024 * 1024;

/// Every leaf is locked - no phase may re-derive it.
const PROFILE: &str = r#"
[lora]
rank = 8
alpha = 16.0
targets = ["q", "k", "v", "o", "ffn_up", "ffn_down", "ffn_gate"]
dtype = "f16"

[training]
# A fact about the dataset - its longest prompt plus the generation budget - so
# it is stated rather than derived.
ctx = 1024
micro_batch = 16
lr = 1e-5
lr_scheduler = "constant"
warmup_steps = 0
weight_decay = 0.005
max_grad_norm = 1.0
generation_concurrency = 8
kv_dtype = "f16"

[grpo]
updates = 100
prompts_per_update = 4
group_size = 8
grpo_epochs = 3
clip_range_low = 0.2
clip_range_high = 0.28
kl_coefficient = 0.0
mask_truncated = true

[grpo.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 128
seed = 42

[evaluation]
every_iterations = 2
patience = 10
min_delta = 0.0
"#;

fn example_params() -> Value {
    toml::from_str(PROFILE).expect("the profile is valid TOML")
}

/// Every leaf of the tree, in the dotted grammar, paired with its value.
fn leaves(tree: &Value) -> Vec<(String, Value)> {
    fn walk(value: &Value, prefix: &mut String, out: &mut Vec<(String, Value)>) {
        match value {
            Value::Object(map) if !map.is_empty() => {
                for (key, child) in map {
                    let restore = prefix.len();
                    if !prefix.is_empty() {
                        prefix.push('.');
                    }
                    prefix.push_str(key);
                    walk(child, prefix, out);
                    prefix.truncate(restore);
                }
            }
            other => out.push((prefix.clone(), other.clone())),
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut String::new(), &mut out);
    out
}

/// Same value, comparing floats at the width the document holds them in.
///
/// The configuration stores `clip_range_high` as an `f32`, and
/// `serde_json::to_value` widens it on the way into this comparison: `0.28`
/// becomes `0.2800000011920929`. That is the `f64` rendering of the same bits, not
/// a value the resolver touched - the API never serializes through a `Value`, and
/// `api_plan.rs` has its own test for that. Comparing at `f32` asks the question
/// this test is actually about.
fn same_value(left: &Value, right: &Value) -> bool {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => left as f32 == right as f32,
        _ => left == right,
    }
}

#[test]
fn the_example_profile_comes_back_exactly_as_it_was_sent() {
    let params = example_params();
    let recipe = Recipe {
        objective: Objective::ReasoningRl,
        eval: Some(DataSpec {
            path: Some(PathBuf::from("eval.jsonl")),
            dataset: None,
            format: Some("jsonl".to_string()),
        }),
        reward: Some(RewardRef {
            id: "sql-exec".to_string(),
        }),
        ..retrograd_plan::testing::sft_recipe()
    };
    let model = tiny_model();
    // Short enough that the pinned `ctx = 1024` truncates nothing - the example
    // pins the context because it knows its own dataset, and a fixture that
    // contradicted it would test the truncation opt-in instead.
    let data = uniform_dataset(600, 800);
    let mut assembled = input(&recipe, &params, &model, &data, machine(24 * GIB, 64 * GIB));
    assembled.reward_command = vec!["python3".to_string(), "reward_sql.py".to_string()];
    let resolution = resolve(&assembled).expect("the example must resolve as written");

    let effective = serde_json::to_value(&resolution.document).expect("serializes");
    for (path, sent) in leaves(&params) {
        let pointer = format!("/{}", path.replace('.', "/"));
        let got = effective
            .pointer(&pointer)
            .unwrap_or_else(|| panic!("{path} is missing from the effective configuration"));
        assert!(
            same_value(got, &sent),
            "{path} was changed: {got} vs {sent}"
        );
        assert_eq!(
            resolution.provenance.get(&path).map(|origin| origin.source),
            Some(Source::Override),
            "{path} is not marked as the caller's"
        );
    }

    // And nothing the example pinned shows up as something the server chose.
    let pinned: Vec<String> = leaves(&params).into_iter().map(|(path, _)| path).collect();
    for applied in &resolution.plan.defaults_applied {
        let entry = retrograd_plan::ACTIVE_DEFAULTS
            .iter()
            .find(|entry| entry.id == applied.id)
            .expect("a reported default is one of the table's");
        for path in entry.touches {
            assert!(
                !pinned.contains(&path.to_string()),
                "'{}' moved {path}, which the example pinned",
                applied.id
            );
        }
    }
}
