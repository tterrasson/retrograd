//! A deterministic in-memory [`DecodeBackend`](crate::DecodeBackend), for
//! tests of the policy that need no model. It models just enough of the
//! runtime to exercise the scheduler: slot spans in a fixed-size cell pool,
//! per-request sampler state (a pure function of the request id and the
//! number of tokens already generated), a piece table for detokenization,
//! and a log of every step, copy and removal so tests can assert on the
//! composed plans.

use std::collections::HashMap;
use std::sync::Arc;

use retrograd_core::SamplingParams;
use retrograd_engine::{EngineMemory, StepOutRow, StepOutput, StepRow};

use crate::{DecodeBackend, InferError};

/// The fake's geometry and script.
pub struct FakeBackend {
    n_slots: u32,
    n_ctx_seq: u32,
    cells_total: u64,
    vocab: u32,
    eos: i32,
    pieces: Vec<String>,
    /// Emit the EOG token after this many generated tokens of a request.
    eog_after: Option<usize>,
    /// When set, sampling returns these tokens in order (per request), so a
    /// test controls the generated text exactly.
    script: Option<Vec<i32>>,
    /// Resident span `[pos_min, pos_max]` of each slot, `None` when empty.
    slots: Vec<Option<(i32, i32)>>,
    /// request id -> tokens already generated (the sampler state).
    requests: HashMap<i32, usize>,
    /// Every `step` call, as the `(slot, n_tokens, request)` rows it carried
    /// (`request` is `Some` on sampling rows).
    pub steps: Vec<Vec<(i32, u32, Option<i32>)>>,
    /// Every `seq_cp(src, dst)` call.
    pub copies: Vec<(i32, i32)>,
    /// Every slot `seq_rm`'d.
    pub removed: Vec<i32>,
    /// First-seen order of request ids in sampling rows.
    pub request_order: Vec<i32>,
}

impl FakeBackend {
    /// A pool of `n_slots` slots, `n_ctx_seq` positions per sequence,
    /// `cells_total` cells in total, and a vocabulary of `vocab` tokens where
    /// `eos` ends generation. `pieces[i]` is the text of token `i`.
    pub fn new(
        n_slots: u32,
        n_ctx_seq: u32,
        cells_total: u64,
        vocab: u32,
        eos: i32,
        pieces: Vec<String>,
    ) -> Self {
        Self {
            n_slots,
            n_ctx_seq,
            cells_total,
            vocab,
            eos,
            pieces,
            eog_after: None,
            script: None,
            slots: vec![None; n_slots as usize],
            requests: HashMap::new(),
            steps: Vec::new(),
            copies: Vec::new(),
            removed: Vec::new(),
            request_order: Vec::new(),
        }
    }

    /// Makes every request emit the EOG token after `count` generated tokens.
    pub fn with_eog_after(mut self, count: usize) -> Self {
        self.eog_after = Some(count);
        self
    }

    /// Makes sampling return `tokens` in order (per request), so a test
    /// controls the generated text exactly.
    pub fn with_script(mut self, tokens: Vec<i32>) -> Self {
        self.script = Some(tokens);
        self
    }

    /// Registers the per-request sampler state a step's sampling rows rely on.
    /// The scheduler does this implicitly by handing out request ids; the
    /// fake accepts any id on a sampling row and starts it at zero.
    fn cells_used(&self) -> u64 {
        self.slots
            .iter()
            .flatten()
            .map(|(lo, hi)| (hi - lo + 1) as u64)
            .sum()
    }

    /// The deterministic "sampled" token of a request's `generated`-th draw.
    fn sample_token(&self, request: i32, generated: usize) -> i32 {
        if self.eog_after == Some(generated) {
            return self.eos;
        }
        if let Some(script) = &self.script {
            return script[generated % script.len()];
        }
        let vocab = self.vocab.saturating_sub(1).max(1) as i64;
        let mixed = (request as i64)
            .wrapping_mul(31)
            .wrapping_add(generated as i64 * 7 + 1);
        (mixed.rem_euclid(vocab)) as i32
    }
}

impl DecodeBackend for FakeBackend {
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
        if rows.is_empty() {
            return Err(InferError::invalid("a step requires at least one row"));
        }
        let total: u64 = rows.iter().map(|row| row.tokens.len() as u64).sum();
        if total == 0 {
            return Err(InferError::invalid("a step row requires tokens"));
        }
        self.steps.push(
            rows.iter()
                .map(|row| {
                    (
                        row.slot,
                        row.tokens.len() as u32,
                        if row.sample { Some(row.request) } else { None },
                    )
                })
                .collect(),
        );
        // Validate spans and capacity before mutating anything.
        let mut used = self.cells_used();
        for row in rows {
            let slot = row.slot;
            if slot < 0 || slot >= self.n_slots as i32 {
                return Err(InferError::invalid("step row slot out of range"));
            }
            if row.tokens.is_empty() {
                return Err(InferError::invalid(
                    "a step row requires at least one token",
                ));
            }
            let end = row
                .pos0
                .checked_add(i32::try_from(row.tokens.len()).map_err(|_| {
                    InferError::Overflow("a step row's token count does not fit i32".into())
                })?)
                .ok_or_else(|| InferError::Overflow("position range overflows i32".into()))?;
            if end > self.n_ctx_seq as i32 {
                return Err(InferError::invalid(
                    "a step row decodes past the per-sequence context",
                ));
            }
            let span = &self.slots[slot as usize];
            match span {
                None => used += row.tokens.len() as u64,
                Some((_, hi)) if row.pos0 == hi + 1 => used += row.tokens.len() as u64,
                Some((lo, hi)) if row.pos0 >= *lo && end - 1 <= *hi => {}
                _ => {
                    return Err(InferError::invalid(
                        "a step row leaves a gap in its slot's resident span",
                    ));
                }
            }
            if row.sample && row.request < 0 {
                return Err(InferError::invalid(
                    "a sampling step row requires a request id >= 0",
                ));
            }
        }
        if used > self.cells_total {
            return Err(InferError::Backend(retrograd_engine::EngineError::Runtime(
                "the fake KV pool is full".into(),
            )));
        }
        // Commit: extend spans, then sample.
        let stride = rows
            .iter()
            .map(|row| row.n_top_logprobs as usize)
            .max()
            .unwrap_or(0);
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut top_ids = vec![0_i32; rows.len() * stride];
        let mut top_logprobs = vec![0.0_f32; rows.len() * stride];
        for (index, row) in rows.iter().enumerate() {
            let end = row.pos0 + row.tokens.len() as i32;
            let slot = &mut self.slots[row.slot as usize];
            *slot = Some(match *slot {
                None => (row.pos0, end - 1),
                Some((lo, hi)) => (lo.min(row.pos0), hi.max(end - 1)),
            });
            if !row.sample {
                out_rows.push(StepOutRow::default());
                continue;
            }
            let generated = *self.requests.get(&row.request).unwrap_or(&0);
            if !self.request_order.contains(&row.request) {
                self.request_order.push(row.request);
            }
            let token = self.sample_token(row.request, generated);
            self.requests.insert(row.request, generated + 1);
            let n_top = (row.n_top_logprobs as usize).min(self.vocab as usize);
            for k in 0..n_top {
                top_ids[index * stride + k] = k as i32;
                top_logprobs[index * stride + k] = -1.0 - k as f32;
            }
            out_rows.push(StepOutRow {
                token,
                logprob: -0.5,
                forced: false,
                sampled: true,
                n_top,
            });
        }
        out.fill(out_rows, stride, top_ids, top_logprobs)?;
        Ok(())
    }

    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        if slot < 0 || slot >= self.n_slots as i32 {
            return Err(InferError::invalid("seq_rm slot out of range"));
        }
        self.removed.push(slot);
        if p0 == -1 && p1 == -1 {
            self.slots[slot as usize] = None;
        } else if let Some((lo, hi)) = self.slots[slot as usize] {
            // The scheduler only ever removes whole sequences; partial ranges
            // shrink the span.
            let lo = if p0 <= lo { p1 } else { lo };
            let hi = if p1 >= hi || p1 < 0 { lo - 1 } else { hi };
            self.slots[slot as usize] = if lo > hi { None } else { Some((lo, hi)) };
        }
        Ok(())
    }

    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        if src < 0 || src >= self.n_slots as i32 || dst < 0 || dst >= self.n_slots as i32 {
            return Err(InferError::invalid("seq_cp slot out of range"));
        }
        self.copies.push((src, dst));
        let span = self.slots[src as usize]
            .ok_or_else(|| InferError::invalid("seq_cp source slot is empty"))?;
        if self.slots[dst as usize].is_some() {
            return Err(InferError::invalid("seq_cp destination slot is not empty"));
        }
        if p0 == -1 && p1 == -1 {
            self.slots[dst as usize] = Some(span);
        }
        Ok(())
    }

    fn memory(&self) -> Result<EngineMemory, InferError> {
        Ok(EngineMemory {
            n_slots: self.n_slots,
            cells_total: self.cells_total,
            cells_used: self.cells_used(),
            kv_bytes: self.cells_used() * 16,
            recurrent_bytes: 0,
        })
    }

    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        self.requests.remove(&request);
        Ok(())
    }

    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        Ok(token == self.eos)
    }

    fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError> {
        Ok(tokens
            .iter()
            .flat_map(|&token| {
                usize::try_from(token)
                    .ok()
                    .and_then(|index| self.pieces.get(index).cloned())
                    .unwrap_or_else(|| format!("<{token}>"))
                    .into_bytes()
            })
            .collect())
    }

    fn context_size(&self) -> Result<u32, InferError> {
        Ok(self.n_ctx_seq)
    }

    fn vocab_size(&self) -> Result<u32, InferError> {
        Ok(self.vocab)
    }
}

/// A sampling-params builder for tests: temperature 1, top-p 1.
pub fn sampling(seed: u32, max_new_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens,
        seed,
    }
}

/// A [`FakeBackend`] behind a mutex, so a test can keep a handle to it and
/// inspect the recorded steps after the actor thread has taken ownership.
pub struct SharedFake(pub Arc<std::sync::Mutex<FakeBackend>>);

impl SharedFake {
    pub fn new(backend: FakeBackend) -> Self {
        Self(Arc::new(std::sync::Mutex::new(backend)))
    }

    /// Locks the fake for inspection.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, FakeBackend> {
        self.0.lock().expect("the fake lock is not poisoned")
    }
}

impl DecodeBackend for SharedFake {
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
        self.lock().step(rows, out)
    }

    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        self.lock().seq_rm(slot, p0, p1)
    }

    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        self.lock().seq_cp(src, dst, p0, p1)
    }

    fn memory(&self) -> Result<EngineMemory, InferError> {
        self.lock().memory()
    }

    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        self.lock().request_free(request)
    }

    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        self.lock().is_eog_token(token)
    }

    fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError> {
        self.lock().detokenize_bytes(tokens)
    }

    fn context_size(&self) -> Result<u32, InferError> {
        self.lock().context_size()
    }

    fn vocab_size(&self) -> Result<u32, InferError> {
        self.lock().vocab_size()
    }
}
