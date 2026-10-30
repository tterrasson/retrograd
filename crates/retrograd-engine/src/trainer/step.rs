//! The trainer's step-wise generation API, on its dedicated generation
//! context (the main context when the run has none): the same operations as
//! [`Engine::step`](crate::Engine::step), through the same shared runtime core,
//! so a trainer can be driven step by step - from its own thread - exactly
//! like an engine.

use super::*;
use crate::step::{EngineMemory, StepOutput, StepRow, step_with, validate_snapshot_flags};

impl Trainer {
    /// Runs one step over `rows` on the generation context; see
    /// [`Engine::step`](crate::Engine::step) for the full contract.
    pub fn step(&mut self, rows: &[StepRow], out: &mut StepOutput) -> Result<()> {
        Ok(step_with(
            rows,
            out,
            |ffi_rows, n_rows, out_rows, top_ids, top_logprobs, stride| {
                // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
                unsafe {
                    ffi::retro_trainer_generation_step(
                        self.raw.as_ptr(),
                        ffi_rows,
                        n_rows,
                        out_rows,
                        top_ids,
                        top_logprobs,
                        stride,
                    )
                }
            },
        )?)
    }

    /// Removes the positions `[p0, p1)` of `slot` from the generation context's
    /// KV (`-1, -1` removes the whole sequence).
    pub fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds.
        self.check(unsafe { ffi::retro_trainer_generation_seq_rm(self.raw.as_ptr(), slot, p0, p1) })
    }

    /// Copies the positions `[p0, p1)` of `src` into `dst` in the generation
    /// context; the copied range of `dst` must be empty.
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds.
        self.check(unsafe {
            ffi::retro_trainer_generation_seq_cp(self.raw.as_ptr(), src, dst, p0, p1)
        })
    }

    /// The size a [`Trainer::seq_save_into`] buffer needs for `slot`.
    pub fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize> {
        validate_snapshot_flags(flags)?;
        let mut needed = 0_usize;
        // SAFETY: the `Trainer` invariant holds and `needed` is a valid out-pointer; a null buffer only reports the size.
        self.check(unsafe {
            ffi::retro_trainer_generation_seq_save(
                self.raw.as_ptr(),
                slot,
                flags,
                std::ptr::null_mut(),
                0,
                &mut needed,
            )
        })?;
        Ok(needed)
    }

    /// Saves a host snapshot (flags 0 or `RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY`)
    /// of `slot` into `buffer`; returns the bytes written.
    pub fn seq_save_into(&self, slot: i32, flags: u32, buffer: &mut [u8]) -> Result<usize> {
        validate_snapshot_flags(flags)?;
        let mut written = 0_usize;
        // SAFETY: the `Trainer` invariant holds and `buffer`/`written` are valid for this synchronous call.
        let code = unsafe {
            ffi::retro_trainer_generation_seq_save(
                self.raw.as_ptr(),
                slot,
                flags,
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut written,
            )
        };
        if code == -2 {
            return Err(Error::invalid(format!(
                "save buffer too small: {written} bytes required"
            )));
        }
        self.check(code)?;
        Ok(written)
    }

    /// Restores a snapshot into `slot`; returns the bytes read.
    pub fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize> {
        validate_snapshot_flags(flags)?;
        let mut read = 0_usize;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_generation_seq_load(
                self.raw.as_ptr(),
                slot,
                flags,
                data.as_ptr(),
                data.len(),
                &mut read,
            )
        })?;
        Ok(read)
    }

    /// The resident position range of `slot` in the generation context; both
    /// `-1` when the slot is empty.
    pub fn seq_pos(&self, slot: i32) -> Result<(i32, i32)> {
        let mut pos_min = 0_i32;
        let mut pos_max = 0_i32;
        // SAFETY: the `Trainer` invariant holds and both are valid out-pointers.
        self.check(unsafe {
            ffi::retro_trainer_generation_seq_pos(
                self.raw.as_ptr(),
                slot,
                &mut pos_min,
                &mut pos_max,
            )
        })?;
        Ok((pos_min, pos_max))
    }

    /// The KV-pool occupancy of the generation context.
    pub fn memory(&self) -> Result<EngineMemory> {
        let mut info = ffi::RetroEngineMemoryInfo::default();
        // SAFETY: the `Trainer` invariant holds and `info` is a valid out-pointer.
        self.check(unsafe { ffi::retro_trainer_generation_memory(self.raw.as_ptr(), &mut info) })?;
        Ok(EngineMemory::from(info))
    }

    /// Releases the per-request state of `request`.
    pub fn request_free(&mut self, request: i32) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds.
        self.check(unsafe {
            ffi::retro_trainer_generation_request_free(self.raw.as_ptr(), request)
        })
    }
}
