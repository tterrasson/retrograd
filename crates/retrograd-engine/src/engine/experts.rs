//! Expert banks and expert placement: what the runtime exposes about the
//! routed expert banks of a MoE model and where the placement fields of
//! [`super::Engine`] land.
//!
//! The placement is an inference option passed beside the geometry, never part
//! of it: [`EngineOptions`] carries it to the constructor, and a different
//! placement is a different set of resident weights rather than a different
//! request.

use std::path::Path;

use retrograd_core::{
    ExpertBank, ExpertCacheGroupStats, ExpertCachePathStats, ExpertCacheStats, ExpertPlacement,
    ExpertPlacementMode, TensorDtype,
};
use retrograd_ffi as ffi;

use crate::{EngineError, ffi_out_vec, path_to_cstring, probe::fixed_name, runtime_error};

/// Options of an [`super::Engine`] beside its geometry: the expert placement
/// of the loaded model. The default is `device` - every tensor follows the
/// device offload, exactly the historical behaviour.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineOptions {
    pub experts: ExpertPlacement,
}

/// Maps a resolved placement to the three wire fields of `retro_engine_config`.
pub(crate) fn expert_placement_to_ffi(experts: &ExpertPlacement) -> (std::ffi::c_int, u32, u64) {
    let placement = match experts.mode {
        ExpertPlacementMode::Device => ffi::RETRO_EXPERT_PLACEMENT_DEVICE,
        ExpertPlacementMode::Host => ffi::RETRO_EXPERT_PLACEMENT_HOST,
    };
    (placement, experts.host_layers, experts.device_cache_bytes)
}

/// The expert banks of a GGUF, read through the runtime's tensor inventory,
/// with the flags saying whether the model is a MoE at all and whether its
/// architecture is validated for explicit expert placement.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertBanks {
    /// The model has at least one expert bank.
    pub moe: bool,
    /// The architecture is on the validated list: every bank of every layer is
    /// a single `ffn_*_exps` bank.
    pub validated: bool,
    pub banks: Vec<ExpertBank>,
}

fn bank_from_ffi(desc: &ffi::RetroExpertBankDesc) -> ExpertBank {
    ExpertBank {
        layer: desc.layer,
        role: fixed_name(&desc.role),
        name: fixed_name(&desc.tensor.name),
        ne: desc.tensor.ne,
        dtype: TensorDtype::from_ggml_name(&fixed_name(&desc.tensor.type_name)),
        n_bytes: desc.tensor.n_bytes,
    }
}

/// Reads the expert banks of the GGUF at `path` without loading any weight: the
/// model is opened mapped and host-only. This is the inventory a placement
/// decision or a validation is taken against before a model is loaded.
pub fn read_expert_banks(model_path: impl AsRef<Path>) -> Result<ExpertBanks, EngineError> {
    let c_path = path_to_cstring(model_path.as_ref())?;
    let mut flags = 0_u32;
    let mut count = 0_usize;
    // SAFETY: the size probe writes only `flags` and `count`.
    let code = unsafe {
        ffi::retro_read_expert_banks(
            c_path.as_ptr(),
            &mut flags,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if code != 0 {
        return Err(EngineError::Runtime(ffi::last_error()));
    }
    // SAFETY: `count` entries are initialized by the runtime on success.
    let banks = unsafe {
        ffi_out_vec(count, |out| {
            let mut written = 0_usize;
            let code =
                ffi::retro_read_expert_banks(c_path.as_ptr(), &mut flags, out, count, &mut written);
            if code != 0 {
                return Err(runtime_error());
            }
            Ok(written)
        })?
    };
    Ok(ExpertBanks {
        moe: flags & ffi::RETRO_EXPERT_BANKS_MOE != 0,
        validated: flags & ffi::RETRO_EXPERT_BANKS_VALIDATED != 0,
        banks: banks.iter().map(bank_from_ffi).collect(),
    })
}

impl super::Engine {
    /// The expert banks of this engine's loaded model, in the same order as
    /// [`read_expert_banks`] on its path.
    pub fn expert_banks(&self) -> Result<Vec<ExpertBank>, EngineError> {
        let mut count = 0_usize;
        // SAFETY: the size probe writes only `count`.
        let code = unsafe {
            ffi::retro_engine_expert_banks(self.raw.as_ptr(), std::ptr::null_mut(), 0, &mut count)
        };
        if code != 0 {
            return Err(EngineError::Runtime(ffi::last_error()));
        }
        // SAFETY: `count` entries are initialized by the runtime on success.
        Ok(unsafe {
            ffi_out_vec(count, |out| {
                let mut written = 0_usize;
                let code =
                    ffi::retro_engine_expert_banks(self.raw.as_ptr(), out, count, &mut written);
                if code != 0 {
                    return Err(runtime_error());
                }
                Ok(written)
            })?
        }
        .iter()
        .map(bank_from_ffi)
        .collect())
    }

    /// Read-only counters of this engine context's MoE expert device cache,
    /// monotonic since the context was created; all zero when the context was
    /// built without a cache.
    pub fn expert_stats(&self) -> Result<ExpertCacheStats, EngineError> {
        let mut stats = ffi::RetroExpertCacheStats::default();
        // SAFETY: the `Engine` invariant holds and `stats` is a valid out-pointer.
        let code = unsafe { ffi::retro_engine_expert_stats(self.raw.as_ptr(), &mut stats) };
        if code != 0 {
            return Err(EngineError::Runtime(ffi::last_error()));
        }
        let path = |raw: ffi::RetroExpertCachePathStats| ExpertCachePathStats {
            hits: raw.hits,
            misses: raw.misses,
            bytes: raw.bytes,
        };
        let n_groups = usize::try_from(stats.n_groups)
            .ok()
            .filter(|n| *n <= stats.groups.len())
            .unwrap_or(0);
        Ok(ExpertCacheStats {
            small: path(stats.small),
            large: path(stats.large),
            copy: path(stats.copy),
            groups: stats.groups[..n_groups]
                .iter()
                .map(|group| ExpertCacheGroupStats {
                    n_slots: group.n_slots,
                    n_layers: group.n_layers,
                    n_expert: group.n_expert,
                    device_bytes: group.device_bytes,
                })
                .collect(),
        })
    }
}

/// Counts for the expert IDs that index one bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertRouteBank {
    pub bank: ExpertBank,
    pub counts: Vec<u32>,
}

/// A snapshot since telemetry was enabled or the counters were reset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertRouteCounts {
    pub banks: Vec<ExpertRouteBank>,
    /// Number of layer micro-batches observed by the slot planner, copy
    /// callback, and explicit readback, respectively.
    pub paths: [u64; 3],
}

impl super::Engine {
    /// Opt into routing telemetry on a validated MoE. Count one micro-batch in
    /// `sample_every`, starting with the next one. Zero disables telemetry.
    /// Reconfiguring discards any previous counts.
    pub fn set_expert_route_sampling(&mut self, sample_every: u32) -> retrograd_core::Result<()> {
        let banks = if sample_every == 0 {
            Vec::new()
        } else {
            self.expert_banks()?
        };
        // SAFETY: the engine is live and exclusively borrowed.
        match unsafe { ffi::retro_engine_expert_route_enable(self.raw.as_ptr(), sample_every) } {
            0 => self.route_banks = banks,
            -2 => return Err(retrograd_core::Error::invalid(ffi::last_error())),
            _ => return Err(runtime_error()),
        }
        Ok(())
    }

    pub fn expert_route_counts(&self) -> Result<ExpertRouteCounts, EngineError> {
        let mut result = ExpertRouteCounts::default();
        if self.route_banks.is_empty() {
            return Ok(result);
        }
        let mut layers = std::collections::BTreeMap::new();
        for bank in &self.route_banks {
            use std::collections::btree_map::Entry;
            let counts = match layers.entry(bank.layer) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let n_expert = usize::try_from(bank.n_expert()).map_err(|_| {
                        EngineError::Runtime("expert bank does not fit in memory".into())
                    })?;
                    let mut counts = vec![0; n_expert];
                    // SAFETY: this buffer has exactly n_expert writable entries.
                    let code = unsafe {
                        ffi::retro_engine_expert_route_counts(
                            self.raw.as_ptr(),
                            bank.layer,
                            counts.as_mut_ptr(),
                            n_expert,
                        )
                    };
                    if code != 0 {
                        return Err(EngineError::Runtime(ffi::last_error()));
                    }
                    entry.insert(counts)
                }
            };
            result.banks.push(ExpertRouteBank {
                bank: bank.clone(),
                counts: counts.clone(),
            });
        }
        // SAFETY: paths has three writable entries as required by the ABI.
        let code = unsafe {
            ffi::retro_engine_expert_route_paths(self.raw.as_ptr(), result.paths.as_mut_ptr())
        };
        if code != 0 {
            return Err(EngineError::Runtime(ffi::last_error()));
        }
        Ok(result)
    }

    pub fn reset_expert_route_counts(&mut self) -> Result<(), EngineError> {
        // SAFETY: the engine is live and exclusively borrowed.
        let code = unsafe { ffi::retro_engine_expert_route_reset(self.raw.as_ptr()) };
        if code != 0 {
            return Err(EngineError::Runtime(ffi::last_error()));
        }
        Ok(())
    }
}
