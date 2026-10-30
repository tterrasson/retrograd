//! Embeddings on the engine's context: the pooled sequence embedding a model
//! that declares a pooling mode produces, and a clear refusal for one that
//! declares none (a plain causal chat model gets no embedding invented by
//! averaging its hidden states).
//!
//! The call is prefill-only: the runtime clears the KV (and with it the
//! prefix cache) before and after, so an embedding never shares KV with a
//! generation whose attention mode differs.

use super::*;

/// The pooling mode a model declares, mirroring `llama_pooling_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingType {
    /// `LLAMA_POOLING_TYPE_NONE`: no sequence embedding exists.
    None,
    Mean,
    Cls,
    Last,
    /// Reranking models: the pooled output is the classification score(s).
    Rank,
}

impl Engine {
    /// The pooling mode the loaded model declares.
    pub fn pooling_type(&self) -> std::result::Result<PoolingType, EngineError> {
        let mut pooling = 0_i32;
        // SAFETY: the `Engine` invariant holds and `pooling` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_pooling_type(self.raw.as_ptr(), &mut pooling) })?;
        Ok(match pooling {
            0 => PoolingType::None,
            1 => PoolingType::Mean,
            2 => PoolingType::Cls,
            3 => PoolingType::Last,
            4 => PoolingType::Rank,
            other => {
                return Err(EngineError::Runtime(format!(
                    "unknown pooling type {other}"
                )));
            }
        })
    }

    /// Whether the loaded model can produce sequence embeddings at all.
    pub fn supports_embeddings(&self) -> std::result::Result<bool, EngineError> {
        Ok(self.pooling_type()? != PoolingType::None)
    }

    /// The width of the pooled embedding the loaded model produces.
    pub fn embedding_size(&self) -> std::result::Result<usize, EngineError> {
        let mut n_embd = 0_usize;
        // SAFETY: the `Engine` invariant holds; a null buffer only reports the
        // size.
        self.check(unsafe {
            ffi::retro_engine_embed(
                self.raw.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                &mut n_embd,
            )
        })?;
        Ok(n_embd)
    }

    /// The pooled embedding of one token sequence. Refused with
    /// [`EngineError::Runtime`] when the model declares no pooling, when the
    /// sequence exceeds the context or the physical batch, or on a decode
    /// failure.
    pub fn embed(&mut self, tokens: &[i32]) -> std::result::Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::InvalidArgument(
                "embedding requires a non-empty token sequence".into(),
            ));
        }
        let n_embd = self.embedding_size()?;
        let mut out = vec![0.0_f32; n_embd];
        let mut written = n_embd;
        // SAFETY: the `Engine` invariant holds and every borrowed argument
        // lives through this synchronous call.
        let code = unsafe {
            ffi::retro_engine_embed(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut written,
            )
        };
        if code == ffi::RETRO_EMBED_NO_POOLING {
            return Err(EngineError::InvalidArgument(
                "the model declares no pooling (a causal chat model): embeddings are not served \
                 for it"
                    .into(),
            ));
        }
        if code == ffi::RETRO_EMBED_INVALID_INPUT {
            return Err(EngineError::InvalidArgument(ffi::last_error()));
        }
        self.check(code)?;
        Ok(out)
    }
}
