//! Snapshot coverage of the expert placement proposal: a table of bank
//! inventories × memory budgets, proposed and compared against a recorded
//! answer. No model, no GPU, no device.
//!
//! The recorded answers live in `tests/snapshots/`. Regenerate them with
//! `RETRO_UPDATE_SNAPSHOTS=1 cargo test -p retrograd-plan`, and *read the diff*.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use retrograd_core::{ExpertBank, TensorDtype};
use retrograd_plan::experts::{PlacementInput, propose};
use serde::Serialize;

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
}

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

/// A bank of `n_expert` experts of `per_expert` bytes, F32, on one layer.
fn bank(layer: u32, role: &str, n_expert: i64, per_expert: u64) -> ExpertBank {
    ExpertBank {
        layer,
        role: role.to_string(),
        name: format!("blk.{layer}.ffn_{role}_exps.weight"),
        ne: [64, 64, n_expert, 1],
        dtype: TensorDtype::F32,
        n_bytes: per_expert * n_expert as u64,
    }
}

/// 32 layers of 3 banks x 128 experts x 1 MiB = 12 GiB of expert banks,
/// beside 8 GiB of dense weights.
fn big_moe() -> PlacementInput {
    let banks = (0..32)
        .flat_map(|layer| {
            ["gate", "up", "down"]
                .iter()
                .map(move |role| bank(layer, role, 128, MIB))
        })
        .collect();
    PlacementInput {
        banks,
        n_expert_used: 8,
        model_total_bytes: 20 * GIB,
        device_free_bytes: Some(16 * GIB),
        host_available_bytes: 64 * GIB,
    }
}

#[test]
fn a_moe_larger_than_the_device_is_proposed_host_with_a_capped_cache() {
    let proposal = propose(&big_moe());
    assert_eq!(
        proposal.placement.mode,
        retrograd_core::ExpertPlacementMode::Host
    );
    assert_snapshot("expert_placement_host_with_cache", &proposal);
}

#[test]
fn a_moe_that_fits_the_device_is_proposed_device() {
    let mut input = big_moe();
    input.device_free_bytes = Some(24 * GIB);
    let proposal = propose(&input);
    assert_eq!(
        proposal.placement,
        retrograd_core::ExpertPlacement::device()
    );
    assert_snapshot("expert_placement_device", &proposal);
}

#[test]
fn a_cpu_only_machine_proposes_host_without_a_cache() {
    let mut input = big_moe();
    input.device_free_bytes = None;
    let proposal = propose(&input);
    assert_eq!(proposal.placement.device_cache_bytes, 0);
    assert_snapshot("expert_placement_host_no_device", &proposal);
}
