//! The inference KV model: what one token of KV costs, what a serving context
//! therefore holds, and which KV storage types a geometry admits.
//!
//! The per-token figure is block-aware, so a quantized cache is not
//! under-counted by its block padding; for F32/F16 it agrees with
//! [`ModelInfo::kv_cache_bytes`]. The support matrix for quantized KV storage
//! refuses an invalid combination *before* allocation rather than letting the
//! backend fail mid-context-creation.

use crate::Error;
use crate::types::{EngineKvDtype, ModelInfo};

impl ModelInfo {
    /// Attention-KV bytes for **one token**, stored in `dtype`.
    ///
    /// Quantized types store blocks, not elements: the per-token cost is the
    /// per-layer element count times the block ratio, rounded up per layer
    /// (ggml blocks a row along the head dimension, which every supported
    /// attention geometry divides into whole blocks - the same assumption
    /// [`crate::KvDtype::block`] documents).
    pub fn kv_bytes_per_token(&self, dtype: EngineKvDtype) -> u64 {
        let (block_bytes, block_elements) = dtype_block(dtype);
        let per_layer_elements =
            (self.n_embd_k_gqa as u64).saturating_add(self.n_embd_v_gqa as u64);
        per_layer_elements
            .div_ceil(block_elements)
            .saturating_mul(block_bytes)
            .saturating_mul(self.n_layer as u64)
    }
}

/// `(bytes, elements)` of one storage block; a plain float type is a block of
/// one element.
fn dtype_block(dtype: EngineKvDtype) -> (u64, u64) {
    match dtype {
        EngineKvDtype::F32 => (4, 1),
        EngineKvDtype::F16 => (2, 1),
        EngineKvDtype::Q8_0 => (34, 32),
        EngineKvDtype::Q4_0 => (18, 32),
    }
}

/// The footprint of one forward-only inference context, in bytes, split the
/// way a memory budget needs it: weights, attention KV, recurrent state (never
/// quantized - the recurrent store is not the attention KV), and the compute
/// buffers (logits plus one wave of activations).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InferenceFootprint {
    /// Mapped model weights.
    pub weight_bytes: u64,
    /// Attention KV cache for `n_ctx` cells.
    pub kv_bytes: u64,
    /// Recurrent/partial state per slot, summed over `n_seq_max` slots. Fixed
    /// per slot: it does not scale with `n_ctx`.
    pub recurrent_bytes: u64,
    /// The context's output (logits) buffer for `n_batch` tokens.
    pub logits_bytes: u64,
    /// One wave of activations for `n_batch` tokens.
    pub activation_bytes: u64,
    pub n_ctx: u64,
    pub n_seq_max: u64,
}

impl InferenceFootprint {
    pub fn compute_bytes(&self) -> u64 {
        self.logits_bytes.saturating_add(self.activation_bytes)
    }

    pub fn total_bytes(&self) -> u64 {
        self.weight_bytes
            .saturating_add(self.kv_bytes)
            .saturating_add(self.recurrent_bytes)
            .saturating_add(self.compute_bytes())
    }

    /// How many sequences of `tokens_per_seq` positions the pool holds:
    /// bounded by the cell pool (`n_ctx / tokens_per_seq`) and by the slot
    /// count. Zero when one sequence alone does not fit.
    pub fn sequences_of(&self, tokens_per_seq: u64) -> u64 {
        sequences_of_len(self.n_ctx, self.n_seq_max, tokens_per_seq)
    }
}

/// "Holds N sequences of L tokens": `min(n_seq_max, n_ctx / L)`, saturating.
pub fn sequences_of_len(n_ctx: u64, n_seq_max: u64, tokens_per_seq: u64) -> u64 {
    if tokens_per_seq == 0 {
        return 0;
    }
    (n_ctx / tokens_per_seq).min(n_seq_max)
}

/// Number of activation tensors retained per layer in one forward wave.
const ACTIVATION_TENSORS_PER_LAYER: u64 = 7;

/// Estimates the footprint of one inference engine context. Mirrors what the
/// runtime allocates for a forward-only context: the KV pool of `n_ctx` cells
/// at `kv_dtype`, the per-slot recurrent state of a hybrid model (F32, never
/// quantized), and the logits plus activation buffers of `n_batch`.
pub fn inference_footprint(
    model: &ModelInfo,
    n_ctx: u64,
    n_seq_max: u64,
    n_batch: u64,
    kv_dtype: EngineKvDtype,
) -> InferenceFootprint {
    let weight_bytes = if model.model_size_bytes > 0 {
        model.model_size_bytes
    } else {
        model.file_size_bytes
    };
    let kv_bytes = model.kv_bytes_per_token(kv_dtype).saturating_mul(n_ctx);
    // Recurrent state: n_embd_r + n_embd_s per layer per slot, F32. It is its
    // own store in the runtime and is deliberately not scaled by kv_dtype.
    let recurrent_per_slot = (model.n_embd_r as u64)
        .saturating_add(model.n_embd_s as u64)
        .saturating_mul(model.n_layer.max(1) as u64)
        .saturating_mul(4);
    let recurrent_bytes = if model.is_recurrent {
        recurrent_per_slot.saturating_mul(n_seq_max.max(1))
    } else {
        0
    };
    let n_batch = n_batch.max(1);
    let logits_bytes = n_batch
        .saturating_mul(model.n_vocab.max(1) as u64)
        .saturating_mul(4);
    let activation_bytes = n_batch
        .saturating_mul(model.n_embd.max(1) as u64)
        .saturating_mul(ACTIVATION_TENSORS_PER_LAYER)
        .saturating_mul(4);
    InferenceFootprint {
        weight_bytes,
        kv_bytes,
        recurrent_bytes,
        logits_bytes,
        activation_bytes,
        n_ctx,
        n_seq_max,
    }
}

/// The KV-storage support matrix, checked before any allocation.
///
/// What the runtime actually constrains (see the fork's KV cache): a quantized
/// K or V is only *read* through the flash-attention path, so a quantized
/// `kv_dtype` requires `flash_attn`, and its attention head dimensions must be
/// a positive multiple of 32 within the FA ceiling of 256. A float cache has
/// no such constraint. The recurrent state of a hybrid model is a separate
/// store that `type_k`/`type_v` never quantize - so a quantized dtype is a
/// statement about the attention KV only, never about the recurrent state.
pub fn validate_kv_dtype_support(
    kv_dtype: EngineKvDtype,
    head_dim_k: u32,
    head_dim_v: u32,
    flash_attn: bool,
) -> Result<(), Error> {
    if !kv_dtype.is_quantized() {
        return Ok(());
    }
    if !flash_attn {
        return Err(Error::config(format!(
            "kv_dtype {} is quantized and is only read through the flash-attention path; \
             enable flash_attn",
            kv_dtype.as_str()
        )));
    }
    for (name, head_dim) in [("n_embd_head_k", head_dim_k), ("n_embd_head_v", head_dim_v)] {
        if head_dim == 0 || head_dim % 32 != 0 || head_dim > 256 {
            return Err(Error::config(format!(
                "{name} = {head_dim} is not a supported attention head dimension \
                 for a quantized KV cache (positive multiple of 32, at most 256)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ModelInfo {
        ModelInfo {
            n_layer: 4,
            n_embd: 64,
            n_embd_k_gqa: 32,
            n_embd_v_gqa: 32,
            n_embd_r: 16,
            n_embd_s: 8,
            n_vocab: 128,
            model_size_bytes: 1000,
            file_size_bytes: 1000,
            is_recurrent: true,
            ..Default::default()
        }
    }

    #[test]
    fn kv_bytes_per_token_matches_the_blocked_cache_total() {
        let model = fixture();
        // F16: 4 layers * (32 + 32) elements * 2 bytes.
        assert_eq!(model.kv_bytes_per_token(EngineKvDtype::F16), 4 * 64 * 2);
        // Q8_0: per layer 64 elements -> 2 blocks of 32 -> 2 * 34 bytes.
        assert_eq!(model.kv_bytes_per_token(EngineKvDtype::Q8_0), 4 * 2 * 34);
        // The per-token figure times n_ctx equals the blocked total.
        let n_ctx = 100u64;
        assert_eq!(
            model.kv_bytes_per_token(EngineKvDtype::Q8_0) * n_ctx,
            inference_footprint(&model, n_ctx, 1, 1, EngineKvDtype::Q8_0).kv_bytes
        );
        // Float types agree with the training-side cache formula.
        assert_eq!(
            model.kv_bytes_per_token(EngineKvDtype::F16) * n_ctx,
            model.kv_cache_bytes(n_ctx, 2)
        );
        // Element counts round up to whole blocks per layer: 33 elements are
        // 2 Q8_0 blocks (34 bytes each) per layer.
        let mut odd = fixture();
        odd.n_embd_k_gqa = 33;
        odd.n_embd_v_gqa = 0;
        assert_eq!(odd.kv_bytes_per_token(EngineKvDtype::Q8_0), 4 * 2 * 34);
    }

    #[test]
    fn footprint_splits_kv_recurrent_compute_and_answers_capacity() {
        let model = fixture();
        let fp = inference_footprint(&model, 256, 4, 64, EngineKvDtype::F16);
        assert_eq!(fp.weight_bytes, 1000);
        assert_eq!(fp.kv_bytes, 4 * 64 * 2 * 256);
        // Recurrent: (16 + 8) * 4 layers * 4 B per slot, never quantized.
        assert_eq!(fp.recurrent_bytes, 24 * 4 * 4 * 4);
        assert_eq!(fp.logits_bytes, 64 * 128 * 4);
        assert!(fp.total_bytes() > fp.kv_bytes);
        assert_eq!(fp.sequences_of(64), 4);
        assert_eq!(fp.sequences_of(100), 2);
        assert_eq!(fp.sequences_of(1000), 0);
        assert_eq!(sequences_of_len(256, 8, 32), 8);
        assert_eq!(sequences_of_len(256, 8, 64), 4);
        assert_eq!(sequences_of_len(256, 8, 0), 0);
    }

    #[test]
    fn quantized_kv_requires_flash_attention_and_sane_head_dims() {
        assert!(validate_kv_dtype_support(EngineKvDtype::Q8_0, 128, 128, false).is_err());
        assert!(validate_kv_dtype_support(EngineKvDtype::Q4_0, 128, 128, true).is_ok());
        assert!(validate_kv_dtype_support(EngineKvDtype::F16, 128, 128, false).is_ok());
        // Head dims outside the FA-supported set are refused even with FA on.
        assert!(validate_kv_dtype_support(EngineKvDtype::Q8_0, 100, 128, true).is_err());
        assert!(validate_kv_dtype_support(EngineKvDtype::Q8_0, 512, 128, true).is_err());
        assert!(validate_kv_dtype_support(EngineKvDtype::Q8_0, 0, 128, true).is_err());
        assert!(validate_kv_dtype_support(EngineKvDtype::Q8_0, 64, 256, true).is_ok());
        // A float cache does not constrain the head dimension.
        assert!(validate_kv_dtype_support(EngineKvDtype::F16, 80, 80, true).is_ok());
    }
}
