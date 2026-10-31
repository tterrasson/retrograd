//! Per-row adapters of the trainer's generation context: frozen in-memory
//! copies of the trained adapter (a judge's snapshot of the policy), selected
//! per step row or as the default of the batched generators, on the same base
//! weights as the policy. The training context never sees them.

use super::*;
use crate::engine::AdapterRowsInfo;
use crate::step::{AdapterId, RowAdapter};

impl Trainer {
    /// Copies the trained LoRA adapter, device to device and without a file,
    /// into a per-row adapter of the generation context, and returns its id.
    /// `replace` is unloaded first, so a refresh never holds two snapshots.
    /// Refused without a dedicated generation context, without a LoRA
    /// adapter, and when the run trains base weights (a copied adapter does
    /// not freeze a policy whose base moves).
    pub fn generation_adapter_snapshot(
        &mut self,
        replace: Option<AdapterId>,
        scale: f32,
    ) -> Result<AdapterId> {
        if !scale.is_finite() {
            return Err(Error::invalid("the adapter scale must be finite"));
        }
        let mut id = 0_i32;
        // SAFETY: the `Trainer` invariant holds and `id` is a valid out-pointer.
        self.check(unsafe {
            ffi::retro_trainer_generation_adapter_snapshot(
                self.raw.as_ptr(),
                replace.map_or(0, AdapterId::get),
                scale,
                &mut id,
            )
        })?;
        AdapterId::new(id)
            .ok_or_else(|| Error::runtime(format!("the runtime returned adapter id {id}")))
    }

    /// Unloads a per-row adapter of the generation context.
    pub fn generation_adapter_unload(&mut self, id: AdapterId) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds.
        self.check(unsafe {
            ffi::retro_trainer_generation_adapter_unload(self.raw.as_ptr(), id.get())
        })
    }

    /// What the generation context's per-row adapters cost (zero without any).
    pub fn generation_adapter_rows_info(&self) -> Result<AdapterRowsInfo> {
        let mut info = ffi::RetroAdapterRowsInfo::default();
        // SAFETY: the `Trainer` invariant holds and `info` is a valid out-pointer.
        self.check(unsafe {
            ffi::retro_trainer_generation_adapter_rows_info(self.raw.as_ptr(), &mut info)
        })?;
        Ok(info.into())
    }

    /// Whether the trainer has the dedicated generation context per-row
    /// adapters live on.
    pub fn supports_generation_adapters(&self) -> bool {
        self.generation_adapter().is_ok()
    }

    fn generation_adapter(&self) -> Result<RowAdapter> {
        let mut adapter = 0;
        // SAFETY: the trainer and the out-pointer are valid for this call.
        self.check(unsafe {
            ffi::retro_trainer_get_generation_adapter(self.raw.as_ptr(), &mut adapter)
        })?;
        match adapter {
            0 => Ok(RowAdapter::Context),
            ffi::RETRO_STEP_ADAPTER_BASE => Ok(RowAdapter::Base),
            id => AdapterId::new(id)
                .map(RowAdapter::Loaded)
                .ok_or_else(|| Error::runtime(format!("the runtime returned adapter id {id}"))),
        }
    }

    fn set_generation_adapter(&mut self, adapter: RowAdapter) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds.
        self.check(unsafe {
            ffi::retro_trainer_set_generation_adapter(self.raw.as_ptr(), adapter.to_ffi())
        })
    }

    /// Runs `operation` with the generation context decoding under `adapter`
    /// (its batched generators, and its step rows that name no adapter),
    /// then restores the previous selection, even when `operation` fails. The
    /// trained adapter stays applied to everything else, so nothing is
    /// toggled and no resident KV is invalidated; the teacher-forced scorers,
    /// which run on the training context, are not affected.
    pub fn with_generation_adapter<T>(
        &mut self,
        adapter: RowAdapter,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let previous = self.generation_adapter()?;
        self.set_generation_adapter(adapter)?;
        let result = operation(self);
        let restore = self.set_generation_adapter(previous);
        match (result, restore) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (_, Err(restore_error)) => Err(restore_error),
        }
    }
}
