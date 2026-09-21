//! The tensor inventory and the trainable-set resolution, against a real GGUF.
//!
//! The unit tests in `retrograd-core` resolve against a hand-written inventory,
//! which proves the rules and nothing about whether the runtime produces the
//! names those rules expect. This binary closes that gap: it reads the CPU
//! fixture's own tensor table and resolves selections over it, so a change in
//! how llama.cpp spells a tensor fails here rather than silently selecting an
//! empty set.
//!
//! The download fixture is a Q4_K_M quantization: full training must refuse
//! it, and a norms-only partial selection must still resolve. The generated
//! fixture is its complement: F32 and untied.

mod common;

use retrograd::{
    Device, OptimizerKind, TENSOR_INVENTORY_VERSION, TensorDtype, TrainablePolicy,
    TrainableSelector, resolve_base, tensor_inventory,
};

fn inventory() -> Option<retrograd::TensorInventory> {
    let model = common::model_path_if_available()?;
    read_inventory(&model)
}

fn tiny_inventory() -> Option<retrograd::TensorInventory> {
    let model = common::tiny_model_path_if_available()?;
    read_inventory(&model)
}

fn read_inventory(model: &std::path::Path) -> Option<retrograd::TensorInventory> {
    let _guard = common::serialize_models();
    Some(tensor_inventory(model, Device::Cpu).expect("read the fixture's tensor inventory"))
}

macro_rules! tiny_inventory {
    () => {
        match tiny_inventory() {
            Some(inventory) => inventory,
            None => {
                eprintln!(
                    "skipping: no generated fixture at {}",
                    common::tiny_model_path().display()
                );
                return;
            }
        }
    };
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
    // The refusal names what base training *does* admit, from the table
    // rather than from a sentence: the list grows with the rows.
    assert!(message.contains("admits F32, F16"), "{message}");
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

/// The capability row checked against the file: every tensor, minus the ones
/// frozen for every model, must be a family the row names. Otherwise the
/// table silently freezes real parameters.
fn assert_row_covers(inventory: &retrograd::TensorInventory) {
    let row = retrograd::architecture_capability(&inventory.architecture)
        .unwrap_or_else(|| panic!("no capability row for '{}'", inventory.architecture));

    for tensor in &inventory.tensors {
        if retrograd::ALWAYS_FROZEN.contains(&tensor.name.as_str()) || tensor.is_rotary_constant() {
            assert!(
                !row.admits(&tensor.name),
                "'{}' is frozen for every model and must not be listed",
                tensor.name
            );
            continue;
        }
        assert!(
            row.admits(&tensor.name),
            "'{}' is a parameter of {} and the row does not name family '{}'",
            tensor.name,
            inventory.architecture,
            retrograd::tensor_family(&tensor.name),
        );
    }
}

#[test]
fn the_fixtures_architecture_row_covers_every_parameter_it_carries() {
    let inventory = fixture_inventory!();
    assert_row_covers(&inventory);
}

#[test]
fn the_generated_fixtures_architecture_row_covers_every_parameter_it_carries() {
    let inventory = tiny_inventory!();
    assert_row_covers(&inventory);
}

/// The hybrid row, against a real Qwen3.5 file rather than against names typed
/// from memory: the family list is what `full` derives its set from, and a
/// family spelled wrong here silently freezes a parameter instead of failing.
///
/// Opt-in: no Qwen3.5 fixture is generated or downloaded, so this skips unless
/// `RETRO_QWEN3NEXT_TEST_MODEL` points at one.
#[test]
fn the_hybrid_rows_families_are_the_ones_a_real_qwen35_file_carries() {
    let family = common::RECURRENT_FAMILIES
        .iter()
        .find(|family| family.family == "gated_delta_net")
        .expect("the gated delta net fixture definition");
    let Some(model) = family.path_if_available() else {
        eprintln!(
            "skipping the qwen35 row: no model at {} (set {})",
            family.path().display(),
            family.env
        );
        return;
    };
    let inventory = read_inventory(&model).expect("read the qwen35 inventory");
    assert_eq!(inventory.architecture, "qwen35");
    assert_row_covers(&inventory);

    // Both kinds of block are present, so the row is checked against the
    // hybrid and not against whichever half a truncated file kept.
    let names: Vec<&str> = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert!(
        names.iter().any(|name| name.ends_with(".ssm_out.weight")),
        "no gated-delta-net block"
    );
    assert!(
        names
            .iter()
            .any(|name| name.ends_with(".attn_output.weight")),
        "no full-attention block"
    );
}

/// The file the row was written from stores its matrices in BF16, which the
/// base-dtype table now carries: `full` selects them instead of refusing them,
/// and the selection is the model's own weights at their own precision, with
/// nothing dropped to a frozen F32 fallback.
#[test]
fn full_training_selects_a_bf16_qwen35_at_its_own_precision() {
    let family = common::RECURRENT_FAMILIES
        .iter()
        .find(|family| family.family == "gated_delta_net")
        .expect("the gated delta net fixture definition");
    let Some(model) = family.path_if_available() else {
        eprintln!("skipping the qwen35 dtype case: set {}", family.env);
        return;
    };
    let inventory = read_inventory(&model).expect("read the qwen35 inventory");
    let bf16 = inventory
        .tensors
        .iter()
        .filter(|tensor| tensor.dtype == TensorDtype::BF16)
        .count();
    if bf16 == 0 {
        eprintln!("skipping: the qwen35 model at hand stores nothing in BF16");
        return;
    }
    let set = resolve_base(
        &inventory,
        TrainablePolicy::Full,
        &TrainableSelector::default(),
    )
    .expect("a BF16 model has a full-training path");
    let selected = set
        .entries
        .iter()
        .filter(|entry| entry.dtype == TensorDtype::BF16)
        .count();
    assert!(
        selected > 0,
        "{bf16} BF16 tensor(s) in the file and none of them selected"
    );
    assert!(
        set.entries
            .iter()
            .all(|entry| entry.dtype == TensorDtype::F32 || entry.dtype == TensorDtype::BF16),
        "the selection carries a dtype the file does not store"
    );
}

/// The properties the generated fixture must hold, asserted rather than
/// assumed by the runs that depend on them.
#[test]
fn the_generated_fixture_is_f32_throughout_and_does_not_tie_its_head() {
    let inventory = tiny_inventory!();
    assert_eq!(inventory.architecture, "qwen2");
    assert!(
        !inventory.tied_embeddings,
        "the generated fixture ties its head"
    );

    let head = inventory
        .get(retrograd::OUTPUT_HEAD)
        .expect("an untied projection head");
    let bias = inventory
        .get(retrograd::OUTPUT_HEAD_BIAS)
        .expect("the head's bias, so the head is two parameters");
    let embedding = inventory
        .get("token_embd.weight")
        .expect("every model has an input embedding");
    assert_ne!(
        head.storage_id, embedding.storage_id,
        "the head and the embedding share one allocation"
    );
    for tensor in &inventory.tensors {
        assert_eq!(tensor.dtype, TensorDtype::F32, "{}", tensor.name);
    }
    assert_eq!(head.dtype, TensorDtype::F32);
    assert_eq!(bias.dtype, TensorDtype::F32);
}

/// `full` derives its own set, run against the generated fixture's inventory.
#[test]
fn full_training_derives_a_set_from_the_generated_fixtures_row() {
    let inventory = tiny_inventory!();
    let set = resolve_base(
        &inventory,
        TrainablePolicy::Full,
        &TrainableSelector::default(),
    )
    .expect("an F32 model of an architecture with a row derives a full set");

    assert!(!set.is_empty());
    let names: Vec<&str> = set
        .base_entries()
        .map(|entry| entry.name.as_str())
        .collect();
    for frozen in retrograd::ALWAYS_FROZEN {
        assert!(
            !names.contains(&frozen),
            "'{frozen}' was derived as trainable"
        );
    }
    // The untied head is in it: on a tied model the head is frozen by
    // storage identity.
    assert!(names.contains(&retrograd::OUTPUT_HEAD));
    assert!(names.contains(&retrograd::OUTPUT_HEAD_BIAS));
    assert!(set.trains_loss_head());
    // qwen2 carries biases on the projections.
    assert!(names.iter().any(|name| name.ends_with("attn_q.bias")));
}

// --- the BF16 fixture family --------------------------------------------------

fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn u64_le(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

/// The byte span of one GGUF metadata value. All the reader below needs of the
/// metadata: it reads no key, it only has to step over them.
fn gguf_value_len(bytes: &[u8], at: usize, kind: u32) -> usize {
    match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => 8 + u64_le(bytes, at) as usize,
        9 => {
            let element = u32_le(bytes, at);
            let count = u64_le(bytes, at + 4);
            let mut span = 12;
            for _ in 0..count {
                span += gguf_value_len(bytes, at + span, element);
            }
            span
        }
        other => panic!("unknown GGUF value type {other}"),
    }
}

/// Every tensor's values, read straight out of a GGUF.
///
/// A dtype with no row is never marked, so the trainer has nothing to read
/// here; the claim is about the file the generator wrote. Only the two
/// storages this fixture family uses are decoded.
fn gguf_values(path: &std::path::Path) -> std::collections::BTreeMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).expect("read the fixture");
    assert_eq!(u32_le(&bytes, 0), 0x4655_4747, "not a GGUF");
    assert_eq!(u32_le(&bytes, 4), 3, "GGUF version");
    let n_tensors = u64_le(&bytes, 8);
    let n_kv = u64_le(&bytes, 16);
    let mut at = 24;

    for _ in 0..n_kv {
        at += 8 + u64_le(&bytes, at) as usize;
        let kind = u32_le(&bytes, at);
        at += 4;
        at += gguf_value_len(&bytes, at, kind);
    }

    let mut table = Vec::new();
    for _ in 0..n_tensors {
        let name_len = u64_le(&bytes, at) as usize;
        at += 8;
        let name = String::from_utf8(bytes[at..at + name_len].to_vec()).expect("a tensor name");
        at += name_len;
        let n_dims = u32_le(&bytes, at);
        at += 4;
        let mut elements = 1_usize;
        for _ in 0..n_dims {
            elements *= u64_le(&bytes, at) as usize;
            at += 8;
        }
        let dtype = u32_le(&bytes, at);
        let offset = u64_le(&bytes, at + 4) as usize;
        at += 12;
        table.push((name, elements, dtype, offset));
    }
    // `general.alignment`, which this generator pins at 32.
    let data = at.next_multiple_of(32);

    table
        .into_iter()
        .map(|(name, elements, dtype, offset)| {
            let start = data + offset;
            let values: Vec<f32> = match dtype {
                0 => bytes[start..start + 4 * elements]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect(),
                30 => bytes[start..start + 2 * elements]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16))
                    .collect(),
                other => panic!("{name} is stored as ggml type {other}"),
            };
            (name, values)
        })
        .collect()
}

/// The BF16 fixture and its control are one model at two storage precisions.
///
/// The F32 fixture above cannot play that role: it is snapped to the F16
/// grid, and an F16-grid value is not a BF16 value. This pair is generated
/// with `--grid bf16`, so the only difference between the two files is the
/// storage.
#[test]
fn the_bf16_fixture_and_its_control_are_one_model_at_two_storage_precisions() {
    let (Some(bf16), Some(control)) = (
        common::tiny_bf16_model_path_if_available(),
        common::tiny_bf16_control_model_path_if_available(),
    ) else {
        eprintln!(
            "skipping: no BF16 fixture family at {}",
            common::tiny_bf16_model_path().display()
        );
        return;
    };

    let left = read_inventory(&bf16).expect("read the BF16 inventory");
    let right = read_inventory(&control).expect("read the control inventory");
    assert_eq!(left.architecture, right.architecture);
    assert_eq!(left.tensors.len(), right.tensors.len());

    let mut matrices = 0_usize;
    let mut vectors = 0_usize;
    for (a, b) in left.tensors.iter().zip(&right.tensors) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.ne, b.ne, "{}", a.name);
        assert_eq!(b.dtype, TensorDtype::F32, "{}", b.name);
        if a.ne[1] > 1 {
            assert_eq!(a.dtype, TensorDtype::BF16, "{}", a.name);
            assert_eq!(a.n_bytes, b.n_bytes / 2, "{}", a.name);
            matrices += 1;
        } else {
            // Vectors stay F32 in both, as a real BF16 GGUF does.
            assert_eq!(a.dtype, TensorDtype::F32, "{}", a.name);
            vectors += 1;
        }
    }
    assert!(matrices > 0 && vectors > 0, "{matrices} / {vectors}");

    // And the numbers themselves, which the inventory does not carry:
    let stored = gguf_values(&bf16);
    let reference = gguf_values(&control);
    assert_eq!(stored.len(), reference.len());
    for (name, values) in &stored {
        let expected = reference.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(values, expected, "{name} differs between the two files");
    }
}
