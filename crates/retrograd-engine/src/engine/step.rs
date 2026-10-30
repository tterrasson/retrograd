//! The engine's step-wise generation API: one decode per call over a
//! caller-built plan, plus the raw sequence operations and the pool occupancy
//! report. Same semantics as the trainer's counterpart, over the engine's own
//! FFI entry points.

use super::*;
use crate::step::{EngineMemory, StepOutput, StepRow, step_with, validate_snapshot_flags};

impl Engine {
    /// Runs one step: a single decode over `rows`, with the per-request
    /// sampler/budget state kept by the runtime. `out` is reused across calls;
    /// see [`StepOutput`].
    ///
    /// After a successful call every input token of every row is resident in
    /// the KV of its slot; a sampled token is not resident until it is fed
    /// back by a later step. A failed call may have made a prefix of the
    /// submitted tokens resident - query [`Engine::seq_pos`] before
    /// resubmitting.
    pub fn step(&mut self, rows: &[StepRow], out: &mut StepOutput) -> Result<(), EngineError> {
        step_with(
            rows,
            out,
            |ffi_rows, n_rows, out_rows, top_ids, top_logprobs, stride| {
                // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
                unsafe {
                    ffi::retro_engine_step(
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
        )
    }

    /// Removes the positions `[p0, p1)` of `slot` from the KV (`-1, -1` removes
    /// the whole sequence; a negative end is open).
    pub fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_seq_rm(self.raw.as_ptr(), slot, p0, p1) })
    }

    /// Copies the positions `[p0, p1)` of `src` into `dst`; the copied range of
    /// `dst` must be empty (remove it with [`Engine::seq_rm`] first).
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_seq_cp(self.raw.as_ptr(), src, dst, p0, p1) })
    }

    /// The size a [`Engine::seq_save_into`] buffer needs for `slot`.
    pub fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, EngineError> {
        validate_snapshot_flags(flags)?;
        let mut needed = 0_usize;
        // SAFETY: the `Engine` invariant holds and `needed` is a valid out-pointer; a null buffer only reports the size.
        self.check(unsafe {
            ffi::retro_engine_seq_save(
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
    /// of `slot` into `buffer` (see `seq_save_len` for the
    /// size); returns the bytes written. `flags` is a combination of the
    /// `RETRO_STATE_SEQ_FLAGS_*` constants.
    pub fn seq_save_into(
        &self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, EngineError> {
        validate_snapshot_flags(flags)?;
        let mut written = 0_usize;
        // SAFETY: the `Engine` invariant holds and `buffer`/`written` are valid for this synchronous call.
        let code = unsafe {
            ffi::retro_engine_seq_save(
                self.raw.as_ptr(),
                slot,
                flags,
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut written,
            )
        };
        if code == -2 {
            return Err(EngineError::invalid(format!(
                "save buffer too small: {written} bytes required"
            )));
        }
        self.check(code)?;
        Ok(written)
    }

    /// Restores a snapshot written by [`Engine::seq_save_into`] into `slot`
    /// (empty it first with [`Engine::seq_rm`]); returns the bytes read.
    pub fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, EngineError> {
        validate_snapshot_flags(flags)?;
        let mut read = 0_usize;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_seq_load(
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

    /// The resident position range of `slot`; both `-1` when the slot is
    /// empty. This is how a caller learns how far a failed step got.
    pub fn seq_pos(&self, slot: i32) -> Result<(i32, i32), EngineError> {
        let mut pos_min = 0_i32;
        let mut pos_max = 0_i32;
        // SAFETY: the `Engine` invariant holds and both are valid out-pointers.
        self.check(unsafe {
            ffi::retro_engine_seq_pos(self.raw.as_ptr(), slot, &mut pos_min, &mut pos_max)
        })?;
        Ok((pos_min, pos_max))
    }

    /// The KV-pool occupancy of the engine's context.
    pub fn memory(&self) -> Result<EngineMemory, EngineError> {
        let mut info = ffi::RetroEngineMemoryInfo::default();
        // SAFETY: the `Engine` invariant holds and `info` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_memory(self.raw.as_ptr(), &mut info) })?;
        Ok(EngineMemory::from(info))
    }

    /// Releases the per-request state (sampler RNG, reasoning-budget machine)
    /// of `request`. The KV of its slot is untouched.
    pub fn request_free(&mut self, request: i32) -> Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_request_free(self.raw.as_ptr(), request) })
    }
}
