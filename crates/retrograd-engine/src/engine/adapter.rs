//! Named LoRA adapters on the inference engine: several checkpoints of one
//! base model can be resident at once, and exactly one is *active* at a time
//! (a decode batch stays homogeneous). Switching the active adapter clears the
//! KV and bumps the weights version on the runtime side, so no prefix computed
//! under another adapter is ever reused.

use std::ffi::CString;
use std::path::Path;

use retrograd_ffi as ffi;

use crate::{nul_error, read_string_sized_lossy};

use super::{Engine, EngineError};

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

/// An adapter name as the runtime takes it: non-empty, NUL-free.
fn adapter_name(name: &str) -> Result<CString, EngineError> {
    if name.is_empty() {
        return Err(EngineError::invalid("an adapter name must be non-empty"));
    }
    Ok(CString::new(name).map_err(nul_error)?)
}
