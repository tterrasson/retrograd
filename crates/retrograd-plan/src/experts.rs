//! Expert placement proposals: given a model's expert bank inventory and the
//! machine's memory, propose where the routed expert banks live and explain
//! the choice in the report's form.
//!
//! Pure like the rest of this crate: no FFI, no model loaded, no device
//! touched. Everything the proposal needs - the banks, the routing width, the
//! memory totals - is an input, which is what makes it testable in the fast
//! lane.
//!
//! The proposal decides only *where the weights sit*, never what the model
//! computes: whatever the placement, the routed ids and the greedy tokens are
//! the same.

use std::collections::BTreeMap;

use retrograd_core::{ExpertBank, ExpertPlacement, ExpertPlacementMode};
use serde::Serialize;

/// Everything a placement proposal needs, all of it an input.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlacementInput {
    /// Every expert bank of the model (all layers, all roles), as the tensor
    /// inventory reports them.
    pub banks: Vec<ExpertBank>,
    /// The routing width: how many experts one token selects (`n_expert_used`).
    pub n_expert_used: u32,
    /// Total bytes of the model, all tensors.
    pub model_total_bytes: u64,
    /// Free device memory when a discrete GPU reports it; `None` on a unified
    /// or CPU-only machine, where the host figure already counts everything
    /// once.
    pub device_free_bytes: Option<u64>,
    /// Host memory available for weights (unified memory counted once).
    pub host_available_bytes: u64,
}

#[derive(Default)]
struct LayerMemory {
    bytes: u64,
    cache_floor: u64,
}

/// A proposed placement with its explanation, in the resolver's report form:
/// one line per decision, with the bytes spelled out.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertProposal {
    pub placement: ExpertPlacement,
    /// One line per fact the operator needs: the level each block lands on,
    /// the bytes, and the expected bottleneck.
    pub explanation: Vec<String>,
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Proposes a placement for the input. The rules, in order:
///
/// 1. everything fits the free device memory ⇒ `device`, no cache;
/// 2. otherwise, propose `host` for every MoE layer and report any host
///    memory shortfall;
/// 3. size the cache from the device memory left after its weights, or
///    disable it when the routing floor does not fit.
pub fn propose(input: &PlacementInput) -> ExpertProposal {
    propose_placement(input, false, 0)
}

/// Proposes a cache budget for a specific host placement.
pub fn propose_host(input: &PlacementInput, host_layers: u32) -> ExpertProposal {
    propose_placement(input, true, host_layers)
}

fn propose_placement(input: &PlacementInput, force_host: bool, host_layers: u32) -> ExpertProposal {
    let mut layers = BTreeMap::<_, LayerMemory>::new();
    let slots = u64::from(input.n_expert_used) + 1;
    for bank in &input.banks {
        let layer = layers.entry(bank.layer).or_default();
        layer.bytes = layer.bytes.saturating_add(bank.n_bytes);
        // Every bank needs all routed slots plus the allocator's padding slot.
        layer.cache_floor = layer
            .cache_floor
            .saturating_add(bank.bytes_per_expert().saturating_mul(slots));
    }
    let expert_bytes = layers
        .values()
        .fold(0u64, |sum, layer| sum.saturating_add(layer.bytes));
    let dense_bytes = input.model_total_bytes.saturating_sub(expert_bytes);
    let mut explanation = Vec::new();
    explanation.push(format!(
        "expert banks: {} across {} MoE layer(s), model total {}",
        human_bytes(expert_bytes),
        layers.len(),
        human_bytes(input.model_total_bytes)
    ));

    // Rule 1: everything on the device.
    if !force_host && let Some(free) = input.device_free_bytes {
        if dense_bytes.saturating_add(expert_bytes) <= free {
            explanation.push(format!(
                "device: the whole model ({} dense + {}) fits the {} free on the device",
                human_bytes(dense_bytes),
                human_bytes(expert_bytes),
                human_bytes(free)
            ));
            explanation.push("placement: device, no expert cache".to_string());
            return ExpertProposal {
                placement: ExpertPlacement::device(),
                explanation,
            };
        }
        explanation.push(format!(
            "device: the model needs {} but only {} are free on the device",
            human_bytes(dense_bytes.saturating_add(expert_bytes)),
            human_bytes(free)
        ));
    }

    let kept_layers = if host_layers == 0 {
        layers.len()
    } else {
        (host_layers as usize).min(layers.len())
    };
    let (host_bytes, floor) =
        layers
            .values()
            .rev()
            .take(kept_layers)
            .fold((0u64, 0), |(bytes, floor), layer| {
                (
                    bytes.saturating_add(layer.bytes),
                    floor.max(layer.cache_floor),
                )
            });
    let device_weights = dense_bytes.saturating_add(expert_bytes.saturating_sub(host_bytes));
    let host_fits = host_bytes <= input.host_available_bytes;
    if host_fits {
        explanation.push(format!(
            "host: the {} of expert banks fit the {} available on the host",
            human_bytes(host_bytes),
            human_bytes(input.host_available_bytes)
        ));
    } else {
        explanation.push(format!(
            "host: the {} of expert banks exceed the {} available on the host - the model does \
             not fit under this placement either",
            human_bytes(host_bytes),
            human_bytes(input.host_available_bytes)
        ));
    }

    // The device cache, when a device has free memory beside the host banks.
    let cache = match input
        .device_free_bytes
        .map(|free| free.saturating_sub(device_weights))
    {
        Some(free) if floor > 0 && free >= floor => {
            let cache = free.min(host_bytes.max(floor));
            explanation.push(format!(
                "device cache: {} of the {} free after device weights (floor for {} experts per group: {})",
                human_bytes(cache),
                human_bytes(free),
                input.n_expert_used,
                human_bytes(floor)
            ));
            cache
        }
        Some(free) => {
            explanation.push(format!(
                "device cache: disabled - the {} free after device weights do not cover the floor of {} \
                 ({} experts per group)",
                human_bytes(free),
                human_bytes(floor),
                input.n_expert_used
            ));
            0
        }
        None => {
            explanation.push(
                "device cache: disabled - no device reports free memory (unified or CPU-only)"
                    .to_string(),
            );
            0
        }
    };

    explanation.push(format!(
        "placement: host, {kept_layers} MoE layer(s) on the host, cache {}",
        if cache == 0 {
            "off".to_string()
        } else {
            human_bytes(cache)
        }
    ));
    let bottleneck = if cache == 0 {
        "expected bottleneck: access to host-resident expert weights"
    } else {
        "expected bottleneck: expert uploads on cache misses"
    };
    explanation.push(bottleneck.to_string());
    ExpertProposal {
        placement: ExpertPlacement {
            mode: ExpertPlacementMode::Host,
            host_layers,
            device_cache_bytes: cache,
        },
        explanation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bank(layer: u32, role: &str, n_expert: i64, per_expert: u64) -> ExpertBank {
        ExpertBank {
            layer,
            role: role.to_string(),
            name: format!("blk.{layer}.ffn_{role}_exps.weight"),
            ne: [64, 64, n_expert, 1],
            dtype: retrograd_core::TensorDtype::F32,
            n_bytes: per_expert * n_expert as u64,
        }
    }

    fn input(
        banks: Vec<ExpertBank>,
        n_expert_used: u32,
        model_total: u64,
        device_free: Option<u64>,
        host: u64,
    ) -> PlacementInput {
        PlacementInput {
            banks,
            n_expert_used,
            model_total_bytes: model_total,
            device_free_bytes: device_free,
            host_available_bytes: host,
        }
    }

    #[test]
    fn everything_on_device_when_it_fits() {
        // 2 layers x 3 banks x 8 experts x 1 KiB = 48 KiB of banks, 1 MiB model.
        let banks: Vec<ExpertBank> = (0..2)
            .flat_map(|layer| {
                ["gate", "up", "down"]
                    .iter()
                    .map(move |role| bank(layer, role, 8, 1024))
            })
            .collect();
        let proposal = propose(&input(banks, 2, 1 << 20, Some(8 << 30), 64 << 30));
        assert_eq!(proposal.placement, ExpertPlacement::device());
        assert!(!proposal.explanation.is_empty());
        assert!(
            proposal
                .explanation
                .iter()
                .any(|line| line.contains("device"))
        );
    }

    #[test]
    fn host_with_a_capped_cache_when_the_device_is_too_small() {
        let banks: Vec<ExpertBank> = (0..2)
            .flat_map(|layer| {
                ["gate", "up", "down"]
                    .iter()
                    .map(move |role| bank(layer, role, 8, 1 << 20))
            })
            .collect();
        // 48 MiB of banks plus 4 MiB dense. 16 MiB free leaves 12 MiB for cache.
        let proposal = propose(&input(banks, 2, 52 << 20, Some(16 << 20), 64 << 30));
        assert_eq!(proposal.placement.mode, ExpertPlacementMode::Host);
        assert_eq!(proposal.placement.host_layers, 0);
        // The cache never exceeds the free device memory, and covers the floor.
        let free = 16u64 << 20;
        assert_eq!(proposal.placement.device_cache_bytes, free - (4 << 20));
        assert!(
            proposal
                .explanation
                .iter()
                .any(|line| line.contains("device cache"))
        );
    }

    #[test]
    fn explicit_host_cache_reserves_device_resident_expert_banks() {
        let banks = vec![bank(2, "down", 8, 1 << 20), bank(5, "down", 8, 1 << 20)];
        let input = input(banks, 2, 20 << 20, Some(16 << 20), 64 << 30);
        let proposal = propose_host(&input, 1);
        assert_eq!(proposal.placement.mode, ExpertPlacementMode::Host);
        assert_eq!(proposal.placement.host_layers, 1);
        // 4 MiB dense + 8 MiB for layer 2 leaves 4 MiB for layer 5's cache.
        assert_eq!(proposal.placement.device_cache_bytes, 4 << 20);
    }

    #[test]
    fn cache_dropped_below_the_floor_and_on_a_cpu_only_machine() {
        let banks = vec![bank(0, "down", 8, 1 << 20)];
        // Free device memory below the floor (2 experts x 1 MiB).
        let proposal = propose(&input(banks.clone(), 2, 16 << 20, Some(1 << 20), 64 << 30));
        assert_eq!(proposal.placement.mode, ExpertPlacementMode::Host);
        assert_eq!(proposal.placement.device_cache_bytes, 0);
        assert!(
            proposal
                .explanation
                .iter()
                .any(|line| line.contains("do not cover the floor"))
        );
        // No device memory reported at all.
        let proposal = propose(&input(banks, 2, 16 << 20, None, 64 << 30));
        assert_eq!(proposal.placement.device_cache_bytes, 0);
        assert!(
            proposal
                .explanation
                .iter()
                .any(|line| line.contains("no device reports free memory"))
        );
    }

    #[test]
    fn the_shortfall_is_spelled_out_when_nothing_fits() {
        let banks = vec![bank(0, "down", 8, 1 << 30)];
        let proposal = propose(&input(banks, 2, 9 << 30, Some(1 << 30), 4 << 30));
        assert_eq!(proposal.placement.mode, ExpertPlacementMode::Host);
        assert!(
            proposal
                .explanation
                .iter()
                .any(|line| line.contains("does not fit"))
        );
    }

    #[test]
    fn properties_hold_over_a_grid_of_inputs() {
        for n_layers in [1u32, 2, 8, 32] {
            for experts in [4i64, 8, 64, 128] {
                for used in [1u32, 2, 4] {
                    for free_mb in [0u64, 64, 512, 4096, 65536] {
                        for host_gib in [1u64, 8, 64, 512] {
                            let per_expert = 1 << 20;
                            let banks: Vec<ExpertBank> = (0..n_layers)
                                .flat_map(|layer| {
                                    ["gate", "up", "down"]
                                        .iter()
                                        .map(move |role| bank(layer, role, experts, per_expert))
                                })
                                .collect();
                            let expert_bytes = n_layers as u64 * 3 * experts as u64 * per_expert;
                            let model_total = expert_bytes + (1 << 24);
                            let proposal = propose(&input(
                                banks,
                                used,
                                model_total,
                                Some(free_mb << 20),
                                host_gib << 30,
                            ));
                            let placement = &proposal.placement;
                            // The cache budget is never exceeded and never sits
                            // between 0 and the floor.
                            if placement.device_cache_bytes > 0 {
                                assert_eq!(placement.mode, ExpertPlacementMode::Host);
                                assert!(
                                    placement.device_cache_bytes <= (free_mb << 20) - (1 << 24)
                                );
                                assert!(
                                    placement.device_cache_bytes
                                        >= (u64::from(used) + 1) * 3 * per_expert
                                );
                            }
                            // Everything fits ⇒ device, no cache.
                            if model_total <= free_mb << 20 {
                                assert_eq!(placement.mode, ExpertPlacementMode::Device);
                                assert_eq!(placement.device_cache_bytes, 0);
                            }
                            // An explanation is always present and names the
                            // placement.
                            assert!(!proposal.explanation.is_empty());
                            assert!(
                                proposal
                                    .explanation
                                    .iter()
                                    .any(|line| line.contains("placement:"))
                            );
                        }
                    }
                }
            }
        }
    }
}
