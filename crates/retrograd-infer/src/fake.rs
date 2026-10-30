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
use retrograd_engine::{
    EngineMemory, STATE_SEQ_FLAGS_PARTIAL_ONLY, StepOutRow, StepOutput, StepRow,
};

use crate::kv::KvCacheKey;
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
    /// Every `seq_cp(src, dst, p0, p1)` call.
    pub copies: Vec<(i32, i32, i32, i32)>,
    /// Every slot `seq_rm`'d.
    pub removed: Vec<i32>,
    /// First-seen order of request ids in sampling rows.
    pub request_order: Vec<i32>,
    /// Every `seq_save(slot, flags)` call with the byte length it reported.
    pub saves: Vec<(i32, u32, usize)>,
    /// Every `seq_load(slot, flags)` call with the byte length it read.
    pub loads: Vec<(i32, u32, usize)>,
    /// Whether the pool is unified (partial-range `seq_cp` allowed).
    pub kv_unified: bool,
    /// Whether the model has a recurrent state (hybrid).
    pub recurrent: bool,
    /// The weights/adapter generation, bumped by `bump_weights_version`.
    pub weights_version: u64,
    /// The model fingerprint (fixed for a fake).
    pub fingerprint: u64,
    /// Named adapters registered by `load_adapter` (name -> path).
    pub adapters: HashMap<String, std::path::PathBuf>,
    /// The active adapter (`None` = base model).
    pub active_adapter: Option<String>,
    /// Every adapter operation, for tests: `("load", name)` / `("activate", name)`.
    pub adapter_ops: Vec<(String, Option<String>)>,
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
            saves: Vec::new(),
            loads: Vec::new(),
            kv_unified: true,
            recurrent: false,
            weights_version: 0,
            fingerprint: 0xFA_CE,
            adapters: HashMap::new(),
            active_adapter: None,
            adapter_ops: Vec::new(),
        }
    }

    /// A partitioned pool: partial-range `seq_cp` is refused, like the
    /// backend's cross-stream assertion.
    pub fn with_kv_unified(mut self, unified: bool) -> Self {
        self.kv_unified = unified;
        self
    }

    /// Marks the fake model as hybrid (recurrent state present).
    pub fn with_recurrent(mut self, recurrent: bool) -> Self {
        self.recurrent = recurrent;
        self
    }

    /// Simulates a weight mutation (optimizer step, adapter change): bumps the
    /// weights version, which invalidates every prefix-cache entry.
    pub fn bump_weights_version(&mut self) {
        self.weights_version += 1;
    }

    /// Registers a named adapter without applying it (the fake's
    /// `load_adapter`).
    pub fn register_adapter(&mut self, name: &str, path: std::path::PathBuf) {
        self.adapters.insert(name.to_owned(), path);
        self.adapter_ops
            .push(("load".into(), Some(name.to_owned())));
    }

    /// Applies the named adapter (or the base model for `None`): like the real
    /// runtime, activation clears the pool and bumps the weights version.
    pub fn apply_adapter(&mut self, name: Option<&str>) -> Result<(), InferError> {
        match name {
            None => {
                self.active_adapter = None;
            }
            Some(name) if self.adapters.contains_key(name) => {
                self.active_adapter = Some(name.to_owned());
            }
            Some(name) => {
                return Err(InferError::invalid(format!(
                    "no adapter registered under '{name}'"
                )));
            }
        }
        self.adapter_ops
            .push(("activate".into(), self.active_adapter.clone()));
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
        self.bump_weights_version();
        Ok(())
    }

    /// Snapshot encoding: a magic, the flags, and the resident span - enough
    /// for a load to restore the span exactly.
    fn encode_snapshot(&self, slot: i32, flags: u32) -> Option<Vec<u8>> {
        let (lo, hi) = self.slots.get(slot as usize).copied().flatten()?;
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&0x5EED_u32.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&lo.to_le_bytes());
        buf.extend_from_slice(&hi.to_le_bytes());
        Some(buf)
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
        // A partial range on a partitioned pool is what asserts in the real
        // backend: refuse it here so tests exercise the fallback.
        let partial = !(p0 == -1 && p1 == -1);
        if partial && !self.kv_unified {
            return Err(InferError::invalid(
                "seq_cp: a partial range between streams of a partitioned pool is unsupported",
            ));
        }
        self.copies.push((src, dst, p0, p1));
        let span = self.slots[src as usize]
            .ok_or_else(|| InferError::invalid("seq_cp source slot is empty"))?;
        if self.slots[dst as usize].is_some() {
            return Err(InferError::invalid("seq_cp destination slot is not empty"));
        }
        if p0 == -1 && p1 == -1 {
            self.slots[dst as usize] = Some(span);
        } else {
            // `p1` is exclusive, mirroring the backend's `pos_in`.
            let lo = span.0.max(if p0 < 0 { 0 } else { p0 });
            let hi = if p1 < 0 { span.1 } else { span.1.min(p1 - 1) };
            self.slots[dst as usize] = (lo <= hi).then_some((lo, hi));
        }
        Ok(())
    }

    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
        if slot < 0 || slot >= self.n_slots as i32 {
            return Err(InferError::invalid("seq_pos slot out of range"));
        }
        Ok(self.slots[slot as usize].unwrap_or((-1, -1)))
    }

    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
        if slot < 0 || slot >= self.n_slots as i32 {
            return Err(InferError::invalid("seq_save slot out of range"));
        }
        // PARTIAL_ONLY on a dense fake is empty; on a hybrid fake it is a
        // fixed-size recurrent-state marker.
        if flags == STATE_SEQ_FLAGS_PARTIAL_ONLY && self.recurrent {
            return Ok(8);
        }
        if flags != 0 && flags != STATE_SEQ_FLAGS_PARTIAL_ONLY {
            return Err(InferError::invalid("unknown snapshot flags"));
        }
        Ok(self
            .encode_snapshot(slot, flags)
            .map(|buf| buf.len())
            .unwrap_or(0))
    }

    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError> {
        let encoded = if flags == STATE_SEQ_FLAGS_PARTIAL_ONLY && self.recurrent {
            let mut buf = Vec::with_capacity(8);
            buf.extend_from_slice(&0x5EED_u32.to_le_bytes());
            buf.extend_from_slice(&STATE_SEQ_FLAGS_PARTIAL_ONLY.to_le_bytes());
            buf
        } else {
            self.encode_snapshot(slot, flags)
                .ok_or_else(|| InferError::invalid("seq_save slot is empty"))?
        };
        if buffer.len() < encoded.len() {
            return Err(InferError::invalid("save buffer too small"));
        }
        buffer[..encoded.len()].copy_from_slice(&encoded);
        self.saves.push((slot, flags, encoded.len()));
        Ok(encoded.len())
    }

    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
        if slot < 0 || slot >= self.n_slots as i32 {
            return Err(InferError::invalid("seq_load slot out of range"));
        }
        self.loads.push((slot, flags, data.len()));
        if flags == STATE_SEQ_FLAGS_PARTIAL_ONLY && self.recurrent {
            // A recurrent-state restore: the span is untouched (the fake keeps
            // spans as the attention KV; the recurrent state is implicit).
            return Ok(data.len());
        }
        if data.len() < 16 || u32::from_le_bytes(data[0..4].try_into().expect("4 bytes")) != 0x5EED
        {
            return Err(InferError::invalid("corrupted snapshot"));
        }
        let lo = i32::from_le_bytes(data[8..12].try_into().expect("4 bytes"));
        let hi = i32::from_le_bytes(data[12..16].try_into().expect("4 bytes"));
        self.slots[slot as usize] = Some((lo, hi));
        Ok(data.len())
    }

    fn kv_unified(&self) -> Result<bool, InferError> {
        Ok(self.kv_unified)
    }

    fn weights_version(&self) -> Result<u64, InferError> {
        Ok(self.weights_version)
    }

    fn model_fingerprint(&self) -> Result<u64, InferError> {
        Ok(self.fingerprint)
    }

    fn is_hybrid(&self) -> Result<bool, InferError> {
        Ok(self.recurrent)
    }

    fn cache_key(&self) -> Result<KvCacheKey, InferError> {
        Ok(KvCacheKey::new(
            self.fingerprint,
            self.weights_version,
            u64::from(self.kv_unified),
        ))
    }

    fn memory(&self) -> Result<EngineMemory, InferError> {
        Ok(EngineMemory {
            n_slots: self.n_slots,
            cells_total: self.cells_total,
            cells_used: self.cells_used(),
            kv_bytes: self.cells_used() * 16,
            recurrent_bytes: if self.recurrent {
                self.cells_used() * 8
            } else {
                0
            },
        })
    }

    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        self.requests.remove(&request);
        Ok(())
    }

    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        Ok(token == self.eos)
    }

    fn detokenize_bytes(
        &self,
        tokens: &[i32],
        unparse_special: bool,
    ) -> Result<Vec<u8>, InferError> {
        Ok(tokens
            .iter()
            .flat_map(|&token| {
                // Like the runtime's detokenize with remove_special: the EOG
                // token renders as text only when special tokens are unparsed.
                if token == self.eos && !unparse_special {
                    return Vec::new();
                }
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

    fn load_adapter(&mut self, name: &str, path: &std::path::Path) -> Result<(), InferError> {
        if name.is_empty() {
            return Err(InferError::invalid("adapter name must not be empty"));
        }
        self.register_adapter(name, path.to_owned());
        Ok(())
    }

    fn activate_adapter(&mut self, name: Option<&str>) -> Result<(), InferError> {
        self.apply_adapter(name)
    }

    fn active_adapter(&self) -> Result<Option<String>, InferError> {
        Ok(self.active_adapter.clone())
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

    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
        self.lock().seq_pos(slot)
    }

    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
        self.lock().seq_save_len(slot, flags)
    }

    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError> {
        self.lock().seq_save_into(slot, flags, buffer)
    }

    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
        self.lock().seq_load(slot, flags, data)
    }

    fn kv_unified(&self) -> Result<bool, InferError> {
        self.lock().kv_unified()
    }

    fn weights_version(&self) -> Result<u64, InferError> {
        self.lock().weights_version()
    }

    fn model_fingerprint(&self) -> Result<u64, InferError> {
        self.lock().model_fingerprint()
    }

    fn is_hybrid(&self) -> Result<bool, InferError> {
        self.lock().is_hybrid()
    }

    fn cache_key(&self) -> Result<KvCacheKey, InferError> {
        self.lock().cache_key()
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

    fn detokenize_bytes(
        &self,
        tokens: &[i32],
        unparse_special: bool,
    ) -> Result<Vec<u8>, InferError> {
        self.lock().detokenize_bytes(tokens, unparse_special)
    }

    fn context_size(&self) -> Result<u32, InferError> {
        self.lock().context_size()
    }

    fn vocab_size(&self) -> Result<u32, InferError> {
        self.lock().vocab_size()
    }

    fn load_adapter(&mut self, name: &str, path: &std::path::Path) -> Result<(), InferError> {
        self.lock().load_adapter(name, path)
    }

    fn activate_adapter(&mut self, name: Option<&str>) -> Result<(), InferError> {
        self.lock().activate_adapter(name)
    }

    fn active_adapter(&self) -> Result<Option<String>, InferError> {
        self.lock().active_adapter()
    }
}
