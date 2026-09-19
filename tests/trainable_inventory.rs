//! The tensor inventory and the trainable-set resolution, against a real GGUF.
//!
//! The unit tests in `retrograd-core` resolve against a hand-written inventory,
//! which proves the rules and nothing about whether the runtime produces the
//! names those rules expect. This binary closes that gap: it reads the CPU
//! fixture's own tensor table and resolves selections over it, so a change in
//! how llama.cpp spells a tensor fails here rather than silently selecting an
//! empty set.
//!
//! The fixture is a Q4_K_M quantization, which is the interesting case: full
//! training must refuse it, and a norms-only partial selection must still
//! resolve - "validate the selected tensors, not the dominant dtype".

mod common;

use retrograd::{
    Device, OptimizerKind, TENSOR_INVENTORY_VERSION, TensorDtype, TrainablePolicy,
    TrainableSelector, resolve_base, tensor_inventory,
};

fn inventory() -> Option<retrograd::TensorInventory> {
    let model = common::model_path_if_available()?;
    let _guard = common::serialize_models();
    Some(tensor_inventory(model, Device::Cpu).expect("read the fixture's tensor inventory"))
}

macro_rules! fixture_inventory {
    () => {
        match inventory() {
            Some(inventory) => inventory,
            None => {
                eprintln!(
                    "skipping: no local model at {}",
                    common::model_path().display()
                );
                return;
            }
        }
    };
}

#[test]
fn the_inventory_describes_the_tensors_the_loader_actually_has() {
    let inventory = fixture_inventory!();
    assert_eq!(inventory.version, TENSOR_INVENTORY_VERSION);
    assert_eq!(inventory.architecture, "lfm2");
    assert!(inventory.n_layer > 0, "{}", inventory.n_layer);
    assert!(!inventory.tensors.is_empty());

    // The two tensors no policy may ever take, and the one every inventory has.
    let embedding = inventory
        .get("token_embd.weight")
        .expect("every model has an input embedding");
    assert!(embedding.n_elements > 0);
    assert_eq!(embedding.n_bytes, embedding.n_bytes.max(1));

    // Names are unique, and so the inventory is a table rather than a list.
    let mut names: Vec<&str> = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let before = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), before, "the inventory repeats a tensor name");

    // Sorted, so two reads of one file give one order.
    let read_order: Vec<&str> = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(read_order, names);
}

#[test]
fn full_training_refuses_a_quantized_fixture_by_name() {
    let inventory = fixture_inventory!();
    let error = resolve_base(
        &inventory,
        TrainablePolicy::Full,
        &TrainableSelector::default(),
    )
    .expect_err("a Q4_K_M model has no full-training path");
    assert!(error.is_user_error(), "{error}");
    let message = error.to_string();
    assert!(message.contains("F32-only"), "{message}");
    // The refusal quotes the tensors, not just a count: a user has to be able
    // to see *which* family is out of reach.
    assert!(message.contains("blk."), "{message}");
}

#[test]
fn a_quantized_fixture_still_has_trainable_f32_norms() {
    let inventory = fixture_inventory!();
    let selector = TrainableSelector {
        norms: true,
        ..Default::default()
    };
    let set = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
        .expect("norms-only resolves on a quantized model");

    assert!(!set.is_empty());
    for entry in set.base_entries() {
        assert_eq!(entry.dtype, TensorDtype::F32, "{}", entry.name);
        assert!(
            entry.name.contains("norm"),
            "a norms-only selection took '{}'",
            entry.name
        );
    }
    // Norms are cheap, and the accounting has to say so rather than repeat the
    // model's own weight bytes.
    assert_eq!(set.parameter_bytes_on_top(), 0);
    assert_eq!(set.gradient_bytes(), set.n_parameters() * 4);
    assert_eq!(
        OptimizerKind::AdamW.state_bytes(&set),
        set.n_parameters() * 8
    );
    assert_eq!(OptimizerKind::Sgd.state_bytes(&set), 0);
}

#[test]
fn the_always_frozen_tensors_are_never_selected_and_are_named_when_excluded() {
    let inventory = fixture_inventory!();
    // `full` is the only policy that would sweep them up, and it refuses this
    // quantized fixture before it gets there - so the exclusion is checked on
    // the selection path instead, where a user can actually ask for them.
    let selector = TrainableSelector {
        modules: vec!["token_embd.weight".into()],
        ..Default::default()
    };
    let error = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
        .expect_err("the input embedding is never trainable");
    assert!(error.to_string().contains("input embedding"), "{error}");
}

#[test]
fn the_layer_range_lands_on_the_fixtures_own_block_indices() {
    let inventory = fixture_inventory!();
    let last = inventory.n_layer - 1;
    let selector = TrainableSelector {
        layers: retrograd::LayerRange::Last(1),
        norms: true,
        ..Default::default()
    };
    let set = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
        .expect("the last block's norms resolve");
    let prefix = format!("blk.{last}.");
    for entry in set.base_entries() {
        assert!(
            entry.name.starts_with(&prefix) || !entry.name.starts_with("blk."),
            "'{}' is outside the last block",
            entry.name
        );
    }
    assert!(
        set.base_entries()
            .any(|entry| entry.name.starts_with(&prefix)),
        "the last block carries no norm"
    );
}

#[test]
fn a_selection_never_issues_two_updates_against_one_allocation() {
    let inventory = fixture_inventory!();
    let selector = TrainableSelector {
        norms: true,
        biases: true,
        ..Default::default()
    };
    let set = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
        .expect("norms and biases resolve");
    let mut identities: Vec<u64> = set
        .base_entries()
        .map(|entry| entry.storage_id)
        .filter(|id| *id != 0)
        .collect();
    let before = identities.len();
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(
        identities.len(),
        before,
        "two entries share one allocation: the second update would read weights \
         the first had already moved"
    );
}
