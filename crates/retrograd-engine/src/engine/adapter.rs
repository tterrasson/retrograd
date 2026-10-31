//! Named LoRA adapters on the inference engine: several checkpoints of one
//! base model can be resident at once, and exactly one is *active* at a time
//! (a decode batch stays homogeneous). Switching the active adapter clears the
//! KV and bumps the weights version on the runtime side, so no prefix computed
//! under another adapter is ever reused.
//!
//! Per-row adapters are the other mode: loaded under ids that are never
//! reused, they are selected row by row through [`StepRow::adapter`], so one
//! decode mixes several checkpoints, the base model and the active adapter.
//! Loading or unloading one leaves the resident KV alone; the KV of a slot
//! belongs to the adapter its rows were decoded under, which is the caller's
//! bookkeeping (a prefix cache partitions by adapter id).
//!
//! [`StepRow::adapter`]: crate::StepRow::adapter

use std::ffi::CString;
use std::path::Path;

use retrograd_ffi as ffi;

use crate::{nul_error, read_string_sized_lossy};

use super::{Engine, EngineError};
use crate::step::AdapterId;

/// What the per-row adapters of a context cost. They are stacked per target
/// weight in F32, each zero-padded to the largest rank among them (exact: the
/// padded factors contribute nothing, and each keeps its own alpha/rank).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterRowsInfo {
    pub n_adapters: u32,
    /// The rank every stacked adapter is padded to.
    pub max_rank: u32,
    /// Bytes of the stacks, padding included, beside the adapters' own
    /// factors.
    pub stacked_bytes: u64,
    /// What the stacks would take without the padding.
    pub unpadded_bytes: u64,
}

impl AdapterRowsInfo {
    /// The bytes the padding to the largest rank costs.
    pub fn padding_bytes(&self) -> u64 {
        self.stacked_bytes.saturating_sub(self.unpadded_bytes)
    }
}

impl From<ffi::RetroAdapterRowsInfo> for AdapterRowsInfo {
    fn from(info: ffi::RetroAdapterRowsInfo) -> Self {
        Self {
            n_adapters: info.n_adapters,
            max_rank: info.max_rank,
            stacked_bytes: info.stacked_bytes,
            unpadded_bytes: info.unpadded_bytes,
        }
    }
}

impl Engine {
    /// Registers a LoRA adapter loaded from `adapter_path` under `name`,
    /// keeping it resident beside the others. The adapter is *not* applied
    /// unless `name` is already active; use [`Engine::activate_lora`] to
    /// apply one.
    pub fn load_lora_named(
        &mut self,
        name: &str,
        adapter_path: impl AsRef<Path>,
    ) -> Result<(), EngineError> {
        let name = adapter_name(name)?;
        let adapter_path = crate::path_to_cstring(adapter_path.as_ref())?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_load_lora_named(
                self.raw.as_ptr(),
                name.as_ptr(),
                adapter_path.as_ptr(),
            )
        })
    }

    /// Applies the adapter registered under `name` to the context, or the
    /// base model when `name` is `None`. Every activation clears the KV and
    /// bumps the weights version (the runtime does it), which is what
    /// namespaces the prefix cache per adapter.
    pub fn activate_lora(&mut self, name: Option<&str>) -> Result<(), EngineError> {
        let name = name.map(adapter_name).transpose()?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_activate_lora(
                self.raw.as_ptr(),
                name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr()),
            )
        })
    }

    /// The name of the active adapter, or `None` when the base model is
    /// active.
    pub fn active_adapter(&self) -> Result<Option<String>, EngineError> {
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        let name = unsafe {
            read_string_sized_lossy(64, |buffer, n_buffer, out| {
                ffi::retro_engine_active_adapter(self.raw.as_ptr(), buffer, n_buffer, out)
            })?
        };
        Ok((!name.is_empty()).then_some(name))
    }
}

impl Engine {
    /// Loads a LoRA checkpoint of the engine's base model as a per-row
    /// adapter, applied at `scale` (times its own alpha/rank) to the rows that
    /// select it. Restacks every per-row adapter; refused for an activated
    /// LoRA and for token-embedding or expert targets.
    pub fn adapter_load(
        &mut self,
        adapter_path: impl AsRef<Path>,
        scale: f32,
    ) -> Result<AdapterId, EngineError> {
        if !scale.is_finite() {
            return Err(EngineError::invalid("the adapter scale must be finite"));
        }
        let adapter_path = crate::path_to_cstring(adapter_path.as_ref())?;
        let mut id = 0_i32;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_adapter_load(self.raw.as_ptr(), adapter_path.as_ptr(), scale, &mut id)
        })?;
        AdapterId::new(id)
            .ok_or_else(|| EngineError::Runtime(format!("the runtime returned adapter id {id}")))
    }

    /// Unloads a per-row adapter. The caller first finishes or cancels the
    /// rows decoding under it and drops their slots: the runtime does not
    /// know which KV belongs to which adapter.
    pub fn adapter_unload(&mut self, id: AdapterId) -> Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_adapter_unload(self.raw.as_ptr(), id.get()) })
    }

    /// What the per-row adapters cost (zero when none is loaded).
    pub fn adapter_rows_info(&self) -> Result<AdapterRowsInfo, EngineError> {
        let mut info = ffi::RetroAdapterRowsInfo::default();
        // SAFETY: the `Engine` invariant holds and `info` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_adapter_rows_info(self.raw.as_ptr(), &mut info) })?;
        Ok(info.into())
    }
}

/// An adapter name as the runtime takes it: non-empty, NUL-free.
fn adapter_name(name: &str) -> Result<CString, EngineError> {
    if name.is_empty() {
        return Err(EngineError::invalid("an adapter name must be non-empty"));
    }
    Ok(CString::new(name).map_err(nul_error)?)
}
