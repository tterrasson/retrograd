//! The step-wise generation API shared by [`Engine`](crate::Engine) and
//! [`Trainer`](crate::Trainer): the plan of a step comes from Rust, its
//! execution from the runtime - one decode per call, raw sequence operations
//! beside it. The batched generators (`generate_batch`,
//! `generate_sampled_continuous`, ...) share its sampling primitives, so a
//! caller-side loop over [`Engine::step`](crate::Engine::step) reproduces them bit for bit when it
//! composes steps in the canonical order.
//!
//! Output buffers are reused: [`StepOutput`] is allocated once and grown only
//! when a call needs more.

use std::borrow::Cow;
use std::mem::size_of;

use retrograd_core::SamplingParams;
use retrograd_ffi as ffi;

use crate::EngineError;

pub(crate) fn validate_snapshot_flags(flags: u32) -> Result<(), EngineError> {
    if flags & !ffi::RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0 {
        return Err(EngineError::invalid(
            "only host snapshots (flags 0 or PARTIAL_ONLY) are supported",
        ));
    }
    Ok(())
}

/// One row of a step plan: `tokens` are decoded into `slot` at the absolute
/// positions `pos0..pos0 + tokens.len() - 1`, and when `sample` is set the last
/// token's logits are sampled through the same oracle as the batched
/// generators.
///
/// `request` keys the per-request state (sampler RNG, reasoning-budget
/// machine), independent of `slot`: a request keeps its state across slot
/// reassignments and releases it with `request_free`. It is required on
/// sampling rows and should be carried from a request's first row so the
/// budget can be primed from the prompt tail.
#[derive(Clone, Debug)]
pub struct StepRow<'a> {
    pub request: i32,
    pub slot: i32,
    pub tokens: Cow<'a, [i32]>,
    pub pos0: i32,
    pub sample: bool,
    pub sampling: SamplingParams,
    pub n_top_logprobs: u32,
}

impl<'a> StepRow<'a> {
    /// A prefill row: decode `tokens` at `pos0`, no sampling.
    pub fn prefill(request: i32, slot: i32, tokens: &'a [i32], pos0: i32) -> Self {
        Self {
            request,
            slot,
            tokens: Cow::Borrowed(tokens),
            pos0,
            sample: false,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens: 1,
                seed: 0,
            },
            n_top_logprobs: 0,
        }
    }

    /// A decode row: feed back one token at `pos0` and sample the next one.
    pub fn decode(
        request: i32,
        slot: i32,
        token: i32,
        pos0: i32,
        sampling: &SamplingParams,
    ) -> Self {
        Self {
            request,
            slot,
            tokens: Cow::Owned(vec![token]),
            pos0,
            sample: true,
            sampling: *sampling,
            n_top_logprobs: 0,
        }
    }

    fn to_ffi(&self) -> Result<ffi::RetroStepRow, EngineError> {
        if self.tokens.is_empty() {
            return Err(EngineError::invalid(
                "a step row requires at least one token",
            ));
        }
        if self.tokens.len() > u32::MAX as usize {
            return Err(EngineError::Overflow(
                "a step row's token count does not fit u32".into(),
            ));
        }
        if self.pos0 < 0 {
            return Err(EngineError::invalid("a step row's pos0 must be >= 0"));
        }
        // pos0 + n must stay a valid position: checked, not wrapped.
        self.pos0
            .checked_add(i32::try_from(self.tokens.len()).map_err(|_| {
                EngineError::Overflow("a step row's position range overflows i32".into())
            })?)
            .ok_or_else(|| {
                EngineError::Overflow("a step row's position range overflows i32".into())
            })?;
        if self.sample && self.request < 0 {
            return Err(EngineError::invalid(
                "a sampling step row requires a request id >= 0",
            ));
        }
        Ok(ffi::RetroStepRow {
            struct_size: size_of::<ffi::RetroStepRow>() as u32,
            request: self.request,
            slot: self.slot,
            tokens: self.tokens.as_ptr(),
            n: self.tokens.len() as u32,
            pos0: self.pos0,
            sample: self.sample,
            sampling: crate::sampling_params_to_ffi(&self.sampling),
            n_top_logprobs: self.n_top_logprobs,
            // Reserved for per-line adapters; 0 is the active adapter.
            adapter: 0,
        })
    }
}

/// What one step produced for one row. `token`/`logprob`/`forced`/`n_top` are
/// meaningful only when `sampled` is set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepOutRow {
    pub token: i32,
    pub logprob: f32,
    pub forced: bool,
    pub sampled: bool,
    pub n_top: usize,
}

/// KV-pool occupancy of a generation context, as reported by
/// [`Engine::memory`](crate::Engine::memory) / [`Trainer::memory`](crate::Trainer::memory).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineMemory {
    /// Sequence slots of the context.
    pub n_slots: u32,
    /// KV cells of the pool.
    pub cells_total: u64,
    /// Occupied cells: the sum of the resident position spans of non-empty slots.
    pub cells_used: u64,
    /// Bytes of the attention KV buffers.
    pub kv_bytes: u64,
    /// Partial state bytes summed over all slots (recurrent state, plus the SWA
    /// KV of a sliding-window layer where the model has one).
    pub recurrent_bytes: u64,
}

impl From<ffi::RetroEngineMemoryInfo> for EngineMemory {
    fn from(info: ffi::RetroEngineMemoryInfo) -> Self {
        Self {
            n_slots: info.n_slots,
            cells_total: info.cells_total,
            cells_used: info.cells_used,
            kv_bytes: info.kv_bytes,
            recurrent_bytes: info.recurrent_bytes,
        }
    }
}

/// The reused buffers of a [`Engine::step`](crate::Engine::step) / [`Trainer::step`](crate::Trainer::step) call: the FFI
/// input staging, one output row per step row, and the row-major top-logprob
/// arrays of stride `n_top_stride`. Allocated by [`StepOutput::with_capacity`],
/// grown only when a call needs more.
pub struct StepOutput {
    ffi_rows: Vec<ffi::RetroStepRow>,
    rows: Vec<ffi::RetroStepOutRow>,
    top_ids: Vec<i32>,
    top_logprobs: Vec<f32>,
    n_top_stride: usize,
    n_rows: usize,
}

impl StepOutput {
    /// Reserves room for `n_rows` step rows and a top-logprob stride of
    /// `n_top_stride` per row.
    pub fn with_capacity(n_rows: usize, n_top_stride: usize) -> Self {
        Self {
            ffi_rows: Vec::with_capacity(n_rows),
            rows: vec![ffi::RetroStepOutRow::default(); n_rows],
            top_ids: vec![0; n_rows.saturating_mul(n_top_stride)],
            top_logprobs: vec![0.0; n_rows.saturating_mul(n_top_stride)],
            n_top_stride,
            n_rows: 0,
        }
    }

    /// Grows the buffers (never shrinks) to fit `n_rows` rows and the given
    /// stride. In steady state - same shape every step - this is a no-op.
    fn ensure(&mut self, n_rows: usize, n_top_stride: usize) -> Result<(), EngineError> {
        if n_rows > self.rows.len() {
            self.rows.resize(n_rows, ffi::RetroStepOutRow::default());
        }
        if self.ffi_rows.len() < n_rows {
            // Zeroed staging rows are fully overwritten by `step_with` before
            // the call; `Default` is null pointers and zero scalars.
            self.ffi_rows.resize(n_rows, ffi::RetroStepRow::default());
        }
        if n_top_stride > self.n_top_stride {
            self.n_top_stride = n_top_stride;
        }
        let needed = n_rows
            .checked_mul(self.n_top_stride)
            .ok_or_else(|| EngineError::Overflow("step output shape overflows usize".into()))?;
        if needed > self.top_ids.len() {
            self.top_ids.resize(needed, 0);
            self.top_logprobs.resize(needed, 0.0);
        }
        Ok(())
    }

    /// Records a step's outputs into the reused buffers, for a backend
    /// implemented in Rust (a fake in tests): the FFI entry points fill the
    /// same buffers through the C ABI instead. `top_ids`/`top_logprobs` are
    /// the row-major arrays of `rows.len() * stride` entries that
    /// [`StepOutput::top`] reads back.
    pub fn fill(
        &mut self,
        rows: Vec<StepOutRow>,
        stride: usize,
        top_ids: Vec<i32>,
        top_logprobs: Vec<f32>,
    ) -> Result<(), EngineError> {
        let n_rows = rows.len();
        let needed = n_rows
            .checked_mul(stride)
            .ok_or_else(|| EngineError::Overflow("step output shape overflows usize".into()))?;
        if top_ids.len() != needed || top_logprobs.len() != needed {
            return Err(EngineError::invalid(
                "fill top arrays must hold rows.len() * stride entries",
            ));
        }
        // Validate every row before changing the last successful output.
        for row in &rows {
            u32::try_from(row.n_top)
                .map_err(|_| EngineError::Overflow("n_top overflows u32".into()))?;
            if row.n_top > stride {
                return Err(EngineError::invalid("n_top exceeds the top-array stride"));
            }
        }
        // Grow-only, like `ensure`: in steady state this reuses the buffers.
        if self.rows.len() < n_rows {
            self.rows.resize(n_rows, ffi::RetroStepOutRow::default());
        }
        if self.ffi_rows.len() < n_rows {
            self.ffi_rows.resize(n_rows, ffi::RetroStepRow::default());
        }
        if self.top_ids.len() < needed {
            self.top_ids.resize(needed, 0);
            self.top_logprobs.resize(needed, 0.0);
        }
        self.n_top_stride = stride;
        for (index, row) in rows.iter().enumerate() {
            self.rows[index] = ffi::RetroStepOutRow {
                struct_size: size_of::<ffi::RetroStepOutRow>() as u32,
                token: row.token,
                logprob: row.logprob,
                // Checked for every row before mutating any buffer above.
                n_top: row.n_top as u32,
                forced: u8::from(row.forced),
                sampled: u8::from(row.sampled),
            };
        }
        self.top_ids[..needed].copy_from_slice(&top_ids);
        self.top_logprobs[..needed].copy_from_slice(&top_logprobs);
        self.n_rows = n_rows;
        Ok(())
    }

    /// The output row of step row `index` in the latest successful call.
    /// Panics if `index` is outside that call's rows.
    pub fn row(&self, index: usize) -> StepOutRow {
        assert!(index < self.n_rows, "step output row index out of bounds");
        let row = self.rows[index];
        StepOutRow {
            token: row.token,
            logprob: row.logprob,
            forced: row.forced != 0,
            sampled: row.sampled != 0,
            n_top: row.n_top as usize,
        }
    }

    /// Every output row of the latest successful call, in step-row order.
    /// Empty before the first call and after a failed call.
    pub fn rows(&self) -> impl Iterator<Item = StepOutRow> + '_ {
        self.rows[..self.n_rows].iter().map(|row| StepOutRow {
            token: row.token,
            logprob: row.logprob,
            forced: row.forced != 0,
            sampled: row.sampled != 0,
            n_top: row.n_top as usize,
        })
    }

    /// The top-logprob `(id, logprob)` pairs of step row `index`, in decreasing
    /// probability order, ties broken by ascending id.
    pub fn top(&self, index: usize) -> impl Iterator<Item = (i32, f32)> + '_ {
        assert!(index < self.n_rows, "step output row index out of bounds");
        let n_top = self.rows[index].n_top as usize;
        let base = index * self.n_top_stride;
        self.top_ids[base..base + n_top]
            .iter()
            .copied()
            .zip(self.top_logprobs[base..base + n_top].iter().copied())
    }
}

/// Validates `rows`, stages them into `out`, and runs the step through `call`.
/// Shared by the engine and trainer wrappers, which only differ in the FFI
/// entry point and the handle.
pub(crate) fn step_with(
    rows: &[StepRow],
    out: &mut StepOutput,
    call: impl FnOnce(
        *const ffi::RetroStepRow,
        usize,
        *mut ffi::RetroStepOutRow,
        *mut i32,
        *mut f32,
        usize,
    ) -> i32,
) -> Result<(), EngineError> {
    // A failed call must not expose outputs from an earlier successful step.
    out.n_rows = 0;
    if rows.is_empty() {
        return Err(EngineError::invalid("a step requires at least one row"));
    }
    let stride = rows
        .iter()
        .map(|row| row.n_top_logprobs as usize)
        .max()
        .unwrap_or(0);
    out.ensure(rows.len(), stride)?;
    for (index, row) in rows.iter().enumerate() {
        let ffi_row = row.to_ffi()?;
        if ffi_row.request >= 0
            && out.ffi_rows[..index]
                .iter()
                .any(|previous| previous.request == ffi_row.request)
        {
            return Err(EngineError::invalid(
                "a request id appears twice in one step",
            ));
        }
        if ffi_row.n_top_logprobs as usize > out.n_top_stride {
            return Err(EngineError::invalid(
                "a step row's n_top_logprobs exceeds the output stride",
            ));
        }
        out.ffi_rows[index] = ffi_row;
    }
    let (top_ids, top_logprobs) = if out.n_top_stride == 0 {
        (std::ptr::null_mut(), std::ptr::null_mut())
    } else {
        (out.top_ids.as_mut_ptr(), out.top_logprobs.as_mut_ptr())
    };
    // SAFETY: the handle invariant of the caller holds; every borrowed pointer
    // (rows, tokens, output buffers) lives through this synchronous call, and
    // `out.ensure` sized the output buffers for `rows.len()` rows and the stride.
    let code = call(
        out.ffi_rows.as_ptr(),
        rows.len(),
        out.rows.as_mut_ptr(),
        top_ids,
        top_logprobs,
        out.n_top_stride,
    );
    if code == -3 {
        return Err(EngineError::invalid(ffi::last_error()));
    }
    if code != 0 {
        return Err(EngineError::Runtime(ffi::last_error()));
    }
    out.n_rows = rows.len();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(seed: u32) -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed,
        }
    }

    #[test]
    fn step_row_validation_catches_bad_shapes_before_the_ffi() {
        let tokens = [1, 2, 3];
        let err = StepRow::prefill(0, 0, &[], 0).to_ffi().unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));
        let err = StepRow::prefill(0, 0, &tokens, -1).to_ffi().unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));
        let err = StepRow {
            request: -1,
            slot: 0,
            tokens: Cow::Borrowed(&tokens[..]),
            pos0: i32::MAX - 1,
            sample: true,
            sampling: sampling(1),
            n_top_logprobs: 0,
        }
        .to_ffi()
        .unwrap_err();
        assert!(matches!(err, EngineError::Overflow(_)));
        let err = StepRow {
            request: -1,
            ..StepRow::decode(0, 0, 1, 0, &sampling(1))
        }
        .to_ffi()
        .unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));

        let ok = StepRow::prefill(7, 2, &tokens[..], 10).to_ffi().unwrap();
        assert_eq!(ok.struct_size as usize, size_of::<ffi::RetroStepRow>());
        assert_eq!(ok.request, 7);
        assert_eq!(ok.slot, 2);
        assert_eq!(ok.n, 3);
        assert_eq!(ok.pos0, 10);
        assert!(!ok.sample);
        assert_eq!(ok.adapter, 0);
    }

    #[test]
    fn step_output_grows_once_and_then_reuses() {
        let mut out = StepOutput::with_capacity(2, 4);
        out.ensure(2, 4).unwrap();
        let ids_ptr = out.top_ids.as_ptr();
        let rows_ptr = out.rows.as_ptr();
        // Same shape again: no reallocation.
        out.ensure(2, 4).unwrap();
        assert_eq!(out.rows.as_ptr(), rows_ptr);
        assert_eq!(out.top_ids.as_ptr(), ids_ptr);
        // A wider stride grows the top buffers once.
        out.ensure(2, 8).unwrap();
        assert!(out.top_ids.as_ptr() != ids_ptr || out.top_ids.len() == 16);
        assert_eq!(out.top_ids.len(), 16);
        assert_eq!(out.n_top_stride, 8);
        // A smaller request keeps the buffers.
        out.ensure(1, 2).unwrap();
        assert_eq!(out.rows.len(), 2);
        assert_eq!(out.n_top_stride, 8);
    }

    #[test]
    fn fill_records_outputs_for_a_rust_side_backend() {
        let mut out = StepOutput::with_capacity(2, 2);
        let rows = vec![
            StepOutRow {
                token: 7,
                logprob: -0.5,
                forced: false,
                sampled: true,
                n_top: 2,
            },
            StepOutRow {
                token: 8,
                logprob: -1.0,
                forced: true,
                sampled: true,
                n_top: 1,
            },
        ];
        out.fill(rows, 2, vec![1, 2, 3, 4], vec![-0.1, -0.2, -0.3, -0.4])
            .unwrap();
        assert_eq!(out.rows().count(), 2);
        assert_eq!(out.row(0).token, 7);
        assert!(!out.row(0).forced);
        assert!(out.row(1).forced);
        assert_eq!(out.top(0).collect::<Vec<_>>(), vec![(1, -0.1), (2, -0.2)]);
        assert_eq!(out.top(1).collect::<Vec<_>>(), vec![(3, -0.3)]);
        // A second fill of the same shape reuses the buffers.
        let rows_ptr = out.rows.as_ptr();
        out.fill(
            vec![StepOutRow {
                token: 9,
                logprob: -2.0,
                forced: false,
                sampled: true,
                n_top: 0,
            }],
            2,
            vec![0, 0],
            vec![0.0, 0.0],
        )
        .unwrap();
        assert_eq!(out.rows.as_ptr(), rows_ptr);
        assert_eq!(out.rows().count(), 1);
        assert_eq!(out.row(0).token, 9);
        // Mismatched top arrays are refused, and a failed fill exposes nothing.
        assert!(out.fill(vec![], 2, vec![0], vec![]).is_err());
        assert_eq!(out.rows().count(), 1);
    }

    #[test]
    fn fill_rejects_rows_wider_than_the_stride_without_changing_previous_outputs() {
        let mut out = StepOutput::with_capacity(1, 0);
        let row = StepOutRow {
            token: 7,
            sampled: true,
            ..StepOutRow::default()
        };
        out.fill(vec![row], 0, vec![], vec![]).unwrap();
        let invalid = StepOutRow {
            token: 9,
            n_top: 2,
            ..row
        };
        assert!(out.fill(vec![invalid], 1, vec![1], vec![-0.1]).is_err());
        assert_eq!(out.row(0), row);
        assert_eq!(out.top(0).count(), 0);
    }

    #[test]
    fn step_with_rejects_an_empty_plan_and_duplicate_requests() {
        let mut out = StepOutput::with_capacity(2, 0);
        let err = step_with(&[], &mut out, |_, _, _, _, _, _| 0).unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));

        let tokens = [1];
        let rows = [
            StepRow::prefill(3, 0, &tokens[..], 0),
            StepRow::prefill(3, 1, &tokens[..], 0),
        ];
        let err = step_with(&rows, &mut out, |_, _, _, _, _, _| 0).unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));
        assert!(err.to_string().contains("twice"));
    }

    #[test]
    fn output_exposes_only_the_latest_successful_step() {
        let mut out = StepOutput::with_capacity(4, 2);
        assert_eq!(out.rows().count(), 0);
        let tokens = [1];
        let rows = [
            StepRow::prefill(1, 0, &tokens, 0),
            StepRow::prefill(2, 1, &tokens, 0),
        ];
        step_with(&rows, &mut out, |_, _, _, _, _, _| 0).unwrap();
        assert_eq!(out.rows().count(), 2);
        step_with(&rows[..1], &mut out, |_, _, _, _, _, _| 0).unwrap();
        assert_eq!(out.rows().count(), 1);
        assert!(std::panic::catch_unwind(|| out.row(1)).is_err());
        assert!(std::panic::catch_unwind(|| out.top(1).count()).is_err());

        let err = step_with(&rows, &mut out, |_, _, _, _, _, _| -3).unwrap_err();
        assert!(matches!(err, EngineError::InvalidArgument(_)));
        assert_eq!(out.rows().count(), 0);
        step_with(&rows, &mut out, |_, _, _, _, _, _| 0).unwrap();
        assert!(step_with(&[], &mut out, |_, _, _, _, _, _| 0).is_err());
        assert_eq!(out.rows().count(), 0);
    }
}
