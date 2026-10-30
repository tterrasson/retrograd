//! The decode backend: what the scheduler drives. The plan of a step comes
//! from the policy in this crate; its execution belongs to the backend - one
//! decode per step plus the raw sequence operations, the pool occupancy
//! report, and the two read-only queries (EOG, detokenization) the emit path
//! needs.
//!
//! [`Engine`](retrograd_engine::Engine) implements the trait directly, a
//! `&mut Trainer` through [`TrainerBackend`] (so rollouts can be driven on
//! the trainer's own thread), and [`FakeBackend`](crate::FakeBackend) is a
//! deterministic in-memory implementation for tests that need no model.

use retrograd_engine::{Engine, EngineMemory, StepOutput, StepRow, Trainer};

use crate::InferError;
use crate::kv::{KvCacheKey, mix};

/// One decode per `step` call, over a caller-built plan, plus the sequence
/// operations beside it. Mirrors the contract of
/// [`Engine::step`](retrograd_engine::Engine::step): after a successful call
/// every input token is resident in its slot's KV, a sampled token is not
/// resident until fed back, and a failed call may have made a prefix of the
/// submitted tokens resident.
pub trait DecodeBackend {
    /// Runs one step (a single decode) over `rows`, writing into the reused
    /// `out` buffers.
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError>;
    /// Removes the positions `[p0, p1)` of `slot` from the KV (`-1, -1`
    /// removes the whole sequence).
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError>;
    /// Copies the positions `[p0, p1)` of `src` into `dst` (`-1, -1` copies the
    /// whole resident range; `p1` is exclusive). A *partial* range is only
    /// supported on a unified pool - check [`DecodeBackend::kv_unified`] first;
    /// on a partitioned pool only whole-sequence copies are supported by the
    /// backend.
    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError>;
    /// The resident position range of `slot`; both `-1` when empty.
    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError>;
    /// The size a [`DecodeBackend::seq_save_into`] buffer needs for `slot`.
    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError>;
    /// Saves a host snapshot (flags 0 = full sequence state, or
    /// `RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY` for the recurrent + SWA state) of
    /// `slot` into `buffer`; returns the bytes written.
    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError>;
    /// Restores a snapshot written by [`DecodeBackend::seq_save_into`] into
    /// `slot`; returns the bytes read.
    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError>;
    /// Whether the pool is unified (any-range `seq_cp` is metadata-only) or
    /// partitioned (whole-sequence copies only). Read back from the built
    /// context, not assumed.
    fn kv_unified(&self) -> Result<bool, InferError>;
    /// The generation of the weights/adapters the resident KV was computed
    /// under; bumped by every invalidation on the backend side.
    fn weights_version(&self) -> Result<u64, InferError>;
    /// Stable fingerprint of the loaded model (weights + tokenizer).
    fn model_fingerprint(&self) -> Result<u64, InferError>;
    /// The key every prefix-cache entry is valid under. The default folds the
    /// fingerprint, the weights version and the pool geometry together.
    fn cache_key(&self) -> Result<KvCacheKey, InferError> {
        let fingerprint = self.model_fingerprint()?;
        let version = self.weights_version()?;
        let memory = self.memory()?;
        let unified = self.kv_unified()?;
        let mut settings = mix(0x9e3779b97f4a7c15, memory.cells_total);
        settings = mix(settings, memory.n_slots as u64);
        settings = mix(settings, u64::from(unified));
        settings = mix(settings, u64::from(self.context_size()?));
        Ok(KvCacheKey::new(fingerprint, version, settings))
    }
    /// Whether the model is hybrid (recurrent state beside the attention KV).
    /// The recurrent store is never quantized and cannot be truncated
    /// mid-sequence, so prefix reuse for hybrids is strict (exact boundaries
    /// only).
    fn is_hybrid(&self) -> Result<bool, InferError>;
    /// The KV-pool occupancy report.
    fn memory(&self) -> Result<EngineMemory, InferError>;
    /// Releases the per-request state (sampler RNG, budget machine) of
    /// `request`. The KV of its slot is untouched.
    fn request_free(&mut self, request: i32) -> Result<(), InferError>;
    /// Whether generation stops on `token` for this model.
    fn is_eog_token(&self, token: i32) -> Result<bool, InferError>;
    /// Renders tokens back to their raw piece bytes, without UTF-8 decoding:
    /// a piece may end mid-character, and the incremental detokenizer splits
    /// characters itself. Special tokens are unparsed.
    fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError>;
    /// The per-sequence context size, in positions.
    fn context_size(&self) -> Result<u32, InferError>;
    /// The model's vocabulary size.
    fn vocab_size(&self) -> Result<u32, InferError>;
}

impl DecodeBackend for Engine {
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
        Ok(Engine::step(self, rows, out)?)
    }

    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        Ok(Engine::seq_rm(self, slot, p0, p1)?)
    }

    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        Ok(Engine::seq_cp(self, src, dst, p0, p1)?)
    }

    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
        Ok(Engine::seq_pos(self, slot)?)
    }

    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
        Ok(Engine::seq_save_len(self, slot, flags)?)
    }

    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError> {
        Ok(Engine::seq_save_into(self, slot, flags, buffer)?)
    }

    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
        Ok(Engine::seq_load(self, slot, flags, data)?)
    }

    fn kv_unified(&self) -> Result<bool, InferError> {
        Ok(Engine::kv_unified(self)?)
    }

    fn weights_version(&self) -> Result<u64, InferError> {
        Ok(Engine::weights_version(self)?)
    }

    fn model_fingerprint(&self) -> Result<u64, InferError> {
        Ok(Engine::model_fingerprint(self)?)
    }

    fn is_hybrid(&self) -> Result<bool, InferError> {
        Ok(Engine::is_hybrid(self)?)
    }

    fn memory(&self) -> Result<EngineMemory, InferError> {
        Ok(Engine::memory(self)?)
    }

    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        Ok(Engine::request_free(self, request)?)
    }

    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        Ok(Engine::is_eog_token(self, token)?)
    }

    fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError> {
        Ok(Engine::detokenize_bytes(self, tokens, false)?)
    }

    fn context_size(&self) -> Result<u32, InferError> {
        Ok(Engine::context_size(self)?)
    }

    fn vocab_size(&self) -> Result<u32, InferError> {
        Ok(Engine::vocab_size(self)?)
    }
}

/// A `&mut Trainer` as a decode backend, over its generation context: the
/// same operations as [`Engine`], so rollouts can be driven step by step from
/// the trainer's own thread - the backend never leaves that thread.
pub struct TrainerBackend<'a>(pub &'a mut Trainer);

impl DecodeBackend for TrainerBackend<'_> {
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
        Ok(Trainer::step(self.0, rows, out)?)
    }

    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        Ok(Trainer::seq_rm(self.0, slot, p0, p1)?)
    }

    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        Ok(Trainer::seq_cp(self.0, src, dst, p0, p1)?)
    }

    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
        Ok(Trainer::seq_pos(self.0, slot)?)
    }

    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
        Ok(Trainer::seq_save_len(self.0, slot, flags)?)
    }

    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError> {
        Ok(Trainer::seq_save_into(self.0, slot, flags, buffer)?)
    }

    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
        Ok(Trainer::seq_load(self.0, slot, flags, data)?)
    }

    fn kv_unified(&self) -> Result<bool, InferError> {
        Ok(Trainer::generation_kv_unified(self.0)?)
    }

    fn weights_version(&self) -> Result<u64, InferError> {
        Ok(Trainer::generation_weights_version(self.0)?)
    }

    fn model_fingerprint(&self) -> Result<u64, InferError> {
        Ok(Trainer::model_fingerprint(self.0)?)
    }

    fn is_hybrid(&self) -> Result<bool, InferError> {
        Ok(Trainer::is_hybrid(self.0)?)
    }

    fn memory(&self) -> Result<EngineMemory, InferError> {
        Ok(Trainer::memory(self.0)?)
    }

    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        Ok(Trainer::request_free(self.0, request)?)
    }

    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        Ok(Trainer::is_eog_token(self.0, token)?)
    }

    fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError> {
        Ok(Trainer::detokenize_bytes(self.0, tokens, false)?)
    }

    fn context_size(&self) -> Result<u32, InferError> {
        // Both native trainer contexts use `kv_unified = true`, so their
        // per-sequence limit equals their full pool size. The existing
        // Trainer::context_size getter queries the separate training context.
        u32::try_from(Trainer::memory(self.0)?.cells_total).map_err(|_| {
            InferError::Overflow("trainer generation context size does not fit u32".into())
        })
    }

    fn vocab_size(&self) -> Result<u32, InferError> {
        u32::try_from(Trainer::vocab_size(self.0)?)
            .map_err(|_| InferError::Overflow("trainer vocab size does not fit u32".into()))
    }
}
