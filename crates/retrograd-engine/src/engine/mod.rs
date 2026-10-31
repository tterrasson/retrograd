//! The safe inference-engine API over the C++ runtime.
//!
//! [`Engine`] is [`Trainer`](crate::Trainer)'s sibling for inference: the same
//! loaded model and the *same* generation code in the runtime (one shared
//! core), but no optimizer, no training result cache and no trainable flags.
//! Its generation is bit-identical to a trainer's generation context at equal
//! geometry, seeds and weights; [`Engine::memory_report`] is how a caller
//! checks the "no training state" property (zero optimizer state, zero
//! gradient).

mod adapter;
mod chat;
mod embed;
mod generate;
mod score;
mod step;

pub use adapter::AdapterRowsInfo;
pub use embed::PoolingType;

use std::ffi::CString;
use std::mem::size_of;
use std::path::Path;
use std::ptr::NonNull;

use retrograd_core::{
    EngineConfig, EngineKvDtype, Generation, GenerationKvDtype, MemoryReport, SamplingParams,
    SamplingParamsV2, TrainConfig,
};
use retrograd_ffi as ffi;

use crate::{
    GenerationStats, SampledRow, ffi_out_vec, nul_error, path_to_cstring, read_bytes_sized,
    read_string_sized_lossy, runtime_error, sampling_params_to_ffi,
};

/// The `n_batch` of a run's dedicated generation context, resolved exactly as
/// the runtime resolves it when it builds that context: an explicit
/// `generation_batch` wins; zero derives the smaller of the context and a
/// 512-token launch, capped by what a 64 MiB logits budget affords the
/// vocabulary; the result is then raised to one row per concurrent sequence,
/// since a decode wave submits one row per live sequence and has to fit a
/// single launch. A scheduler composing steps over that context must use this
/// same budget to reproduce the batched generators' step shapes.
pub fn generation_batch(training: &TrainConfig, n_vocab: u32) -> u32 {
    let batch = match training.generation_batch {
        0 => {
            // The quotient is at most 2^24 (64 MiB over a four-byte row), so
            // the conversion back to u32 cannot fail; the min with 512 bounds
            // the result regardless.
            let affordable = ((64_u64 << 20) / (u64::from(n_vocab.max(1)) * 4)).max(1);
            let affordable = u32::try_from(affordable).unwrap_or(u32::MAX);
            training.n_ctx.min(512).min(affordable)
        }
        explicit => explicit,
    };
    batch.max(training.effective_generation_concurrency())
}

/// The [`EngineConfig`] that reproduces a run's dedicated generation context:
/// `n_ctx * effective_generation_concurrency` KV cells, one slot per concurrent
/// sequence, the [`generation_batch`] geometry for the model's `n_vocab`, and
/// the fast context's KV dtype with flash attention when the run asks for one.
/// At this geometry the engine's generation is bit-identical to the trainer's
/// (see `tests/engine_runtime.rs`), which is what lets generation-only commands
/// run on the lightweight handle without changing a single token they print.
pub fn engine_config_from_train_config(training: &TrainConfig, n_vocab: u32) -> EngineConfig {
    let concurrency = training.effective_generation_concurrency();
    let n_batch = generation_batch(training, n_vocab);
    EngineConfig {
        n_ctx: training.n_ctx.saturating_mul(concurrency),
        n_seq_max: concurrency,
        n_batch,
        n_ubatch: n_batch,
        kv_dtype: if training.fast_generation_context {
            match training.generation_kv_dtype {
                GenerationKvDtype::F16 => EngineKvDtype::F16,
                GenerationKvDtype::Q8_0 => EngineKvDtype::Q8_0,
                GenerationKvDtype::Q4_0 => EngineKvDtype::Q4_0,
            }
        } else {
            EngineKvDtype::F32
        },
        flash_attn: training.fast_generation_context,
        kv_unified: true,
        device: training.device,
        threads: training.threads,
        verbose: training.verbose,
    }
}

/// Errors of the inference engine. Converted into
/// [`retrograd_core::Error`](retrograd_core::Error) by the `From` impl below,
/// which lives here - in the crate that owns this error - rather than in
/// `retrograd-core`.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// What the caller passed is wrong: an argument out of range, a shape the
    /// engine cannot honour.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A computed size, count or product left the range of its type.
    #[error("arithmetic overflow: {0}")]
    Overflow(String),
    /// The runtime refused or failed while running.
    #[error("engine runtime error: {0}")]
    Runtime(String),
    /// An error from the shared core/ffi helpers (tokenization guards, buffer
    /// helpers), carried unchanged.
    #[error(transparent)]
    Core(#[from] retrograd_core::Error),
}

impl EngineError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }
}

impl From<EngineError> for retrograd_core::Error {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::InvalidArgument(message) => {
                retrograd_core::Error::InvalidArgument(message)
            }
            EngineError::Overflow(message) => retrograd_core::Error::Overflow(message),
            EngineError::Runtime(message) => retrograd_core::Error::Runtime(message),
            EngineError::Core(error) => error,
        }
    }
}

fn engine_config_to_ffi(
    config: &EngineConfig,
) -> std::result::Result<ffi::RetroEngineConfig, EngineError> {
    if config.n_ctx == 0 || config.n_seq_max == 0 {
        return Err(EngineError::invalid(
            "engine n_ctx and n_seq_max must be greater than zero",
        ));
    }
    if config.n_batch == 0 || config.n_ubatch == 0 || config.n_ubatch > config.n_batch {
        return Err(EngineError::invalid(
            "engine n_ubatch must be in (0, n_batch]",
        ));
    }
    if [
        config.n_ctx,
        config.n_batch,
        config.n_ubatch,
        config.threads,
    ]
    .into_iter()
    .any(|value| value > i32::MAX as u32)
    {
        return Err(EngineError::invalid(
            "engine context, batch and thread counts must fit in int32_t",
        ));
    }
    // llama.cpp's LLAMA_MAX_SEQ is 256; each live sequence also needs a row
    // in the physical decode batch.
    if config.n_seq_max > 256 || config.n_seq_max > config.n_batch.min(config.n_ctx) {
        return Err(EngineError::invalid(
            "engine n_seq_max must not exceed 256, n_batch or n_ctx",
        ));
    }
    if config.kv_dtype.is_quantized() && !config.flash_attn {
        return Err(EngineError::invalid(
            "a quantized engine kv_dtype requires flash_attn",
        ));
    }
    Ok(ffi::RetroEngineConfig {
        // The fixed C layout is checked by the FFI ABI tests.
        struct_size: size_of::<ffi::RetroEngineConfig>() as u32,
        n_ctx: config.n_ctx,
        n_seq_max: config.n_seq_max,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        kv_dtype: config.kv_dtype.as_ffi(),
        flash_attn: config.flash_attn,
        device: config.device.as_ffi(),
        threads: config.threads,
        verbose: config.verbose,
        kv_unified: u32::from(config.kv_unified),
    })
}

/// A loaded model with one forward-only context, for inference. The handle
/// owns no optimizer and no training state; see the module docs.
pub struct Engine {
    // Invariant for every FFI call made by the engine modules: `raw` is the
    // unique non-null handle returned by `retro_engine_new`, remains live for
    // the whole Rust value, and is freed exactly once by `Drop`. All slice,
    // string and out-parameter pointers passed beside it are borrowed only for
    // the duration of the synchronous call.
    raw: NonNull<ffi::RetroEngine>,
    // Fixed at construction, bounded by LLAMA_MAX_SEQ in config validation.
    n_seq_max: usize,
}

impl Engine {
    /// Loads a GGUF model with the geometry of `training`'s dedicated
    /// generation context ([`engine_config_from_train_config`]). The model's
    /// vocabulary, which that geometry depends on, is read first from the
    /// GGUF metadata alone - no weights are offloaded for it.
    pub fn from_train_config(
        model_path: impl AsRef<Path>,
        training: &TrainConfig,
    ) -> retrograd_core::Result<Self> {
        let model_path = model_path.as_ref();
        let n_vocab = crate::model_info(model_path, training.device)?.n_vocab;
        Ok(Self::new(
            model_path,
            &engine_config_from_train_config(training, n_vocab),
        )?)
    }

    /// Loads a GGUF model (mapped, never written) and builds one forward-only
    /// context with the requested geometry. No adapter is attached; call
    /// [`Engine::load_lora`] to attach one.
    pub fn new(
        model_path: impl AsRef<Path>,
        config: &EngineConfig,
    ) -> std::result::Result<Self, EngineError> {
        let model_path = path_to_cstring(model_path.as_ref())?;
        let ffi_config = engine_config_to_ffi(config)?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        let raw = unsafe { ffi::retro_engine_new(model_path.as_ptr(), &ffi_config) };
        NonNull::new(raw)
            .ok_or_else(|| EngineError::Runtime(ffi::last_error()))
            .map(|raw| Self {
                raw,
                n_seq_max: config.n_seq_max as usize,
            })
    }

    /// Attaches a LoRA adapter loaded from a GGUF file, replacing any adapter
    /// already attached. No tensor is flagged as an optimizer parameter: the
    /// engine has no optimizer.
    pub fn load_lora(
        &mut self,
        adapter_path: impl AsRef<Path>,
    ) -> std::result::Result<(), EngineError> {
        let adapter_path = path_to_cstring(adapter_path.as_ref())?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_engine_load_lora(self.raw.as_ptr(), adapter_path.as_ptr()) })
    }

    /// Enables or disables the attached adapter. Disabling without an adapter
    /// is a no-op; enabling without one is an error.
    pub fn set_lora_enabled(&mut self, enabled: bool) -> std::result::Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_set_lora_enabled(self.raw.as_ptr(), enabled) })
    }

    /// Sets JSON chat-template variables; see the trainer's counterpart for
    /// the accepted shape.
    pub fn set_chat_template_variables(
        &mut self,
        variables_json: &str,
    ) -> std::result::Result<(), EngineError> {
        let variables = CString::new(variables_json).map_err(nul_error)?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_set_chat_template_variables(self.raw.as_ptr(), variables.as_ptr())
        })
    }

    /// Caps the tokens of a reasoning block; -1 (the default) lets the model
    /// decide. Forced tokens are reported through the generators'
    /// `out_forced`, i.e. [`Engine::generate_sampled_continuous`].
    pub fn set_reasoning_budget(&mut self, budget: i32) -> std::result::Result<(), EngineError> {
        // SAFETY: the `Engine` invariant holds.
        self.check(unsafe { ffi::retro_engine_set_reasoning_budget(self.raw.as_ptr(), budget) })
    }

    /// Tokenizes a complete text, prepending BOS.
    pub fn tokenize_text(&self, text: &str) -> std::result::Result<Vec<i32>, EngineError> {
        self.tokenize(text, false)
    }

    /// Tokenizes a fragment that continues an existing token stream: no BOS is
    /// prepended. An empty fragment tokenizes to nothing.
    pub fn tokenize_fragment(&self, text: &str) -> std::result::Result<Vec<i32>, EngineError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.tokenize(text, true)
    }

    fn tokenize(&self, text: &str, fragment: bool) -> std::result::Result<Vec<i32>, EngineError> {
        let tokenize = if fragment {
            ffi::retro_engine_tokenize_fragment
        } else {
            ffi::retro_engine_tokenize_text
        };
        let text = CString::new(text).map_err(nul_error)?;
        let mut needed = 0_usize;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        let code = unsafe {
            tokenize(
                self.raw.as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if code != 0 {
            return Err(EngineError::Runtime(ffi::last_error()));
        }
        if needed == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the `Engine` invariant holds, all borrowed arguments live
        // through this synchronous call, and `tokenize` writes the `written`
        // ids it reports.
        Ok(unsafe {
            ffi_out_vec(needed, |tokens_out| {
                let mut written = 0_usize;
                let code = tokenize(
                    self.raw.as_ptr(),
                    text.as_ptr(),
                    tokens_out,
                    needed,
                    &mut written,
                );
                if code != 0 {
                    return Err(runtime_error());
                }
                Ok(written)
            })?
        })
    }

    /// The model's end-of-sequence token.
    pub fn eos_token(&self) -> std::result::Result<i32, EngineError> {
        let mut token = 0_i32;
        // SAFETY: the `Engine` invariant holds and `token` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_eos_token(self.raw.as_ptr(), &mut token) })?;
        Ok(token)
    }

    /// The model's vocabulary size.
    pub fn vocab_size(&self) -> std::result::Result<u32, EngineError> {
        let mut size = 0_u32;
        // SAFETY: the `Engine` invariant holds and `size` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_vocab_size(self.raw.as_ptr(), &mut size) })?;
        Ok(size)
    }

    /// Whether generation stops on `token` for this model.
    pub fn is_eog_token(&self, token: i32) -> std::result::Result<bool, EngineError> {
        let mut is_eog = false;
        // SAFETY: the `Engine` invariant holds and `is_eog` is a valid out-pointer.
        self.check(unsafe {
            ffi::retro_engine_is_eog_token(self.raw.as_ptr(), token, &mut is_eog)
        })?;
        Ok(is_eog)
    }

    /// The context size of the engine's context, per sequence.
    pub fn context_size(&self) -> std::result::Result<u32, EngineError> {
        let mut n_ctx = 0_u32;
        // SAFETY: the `Engine` invariant holds and `n_ctx` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_context_size(self.raw.as_ptr(), &mut n_ctx) })?;
        Ok(n_ctx)
    }

    /// Renders tokens back to text; see [`crate::Trainer::detokenize`] for the
    /// `unparse_special` choice. Decoded leniently: sampling can stop
    /// mid-character.
    pub fn detokenize(
        &self,
        tokens: &[i32],
        unparse_special: bool,
    ) -> std::result::Result<String, EngineError> {
        let size_hint = tokens.len().saturating_mul(16).max(64);
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        Ok(unsafe {
            read_string_sized_lossy(size_hint, |buffer, n_buffer, out| {
                ffi::retro_engine_detokenize(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    unparse_special,
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    /// Renders tokens back to their raw piece bytes, with no UTF-8 decoding:
    /// a piece may end mid-character, and a streaming caller that
    /// detokenizes token by token accumulates the bytes and splits characters
    /// itself - decoding each piece alone would replace the partial trailing
    /// character with U+FFFD.
    pub fn detokenize_bytes(
        &self,
        tokens: &[i32],
        unparse_special: bool,
    ) -> std::result::Result<Vec<u8>, EngineError> {
        let size_hint = tokens.len().saturating_mul(16).max(64);
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        Ok(unsafe {
            read_bytes_sized(size_hint, |buffer, n_buffer, out| {
                ffi::retro_engine_detokenize(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    unparse_special,
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    /// The structured memory breakdown of this engine: weights, KV and compute
    /// buffers of its one context, plus the attached adapter if any. The
    /// optimizer-state and gradient fields are always zero - that is the
    /// point of the handle - so comparing them against a trainer's report is
    /// the memory check.
    pub fn memory_report(&self) -> std::result::Result<MemoryReport, EngineError> {
        let mut report = ffi::RetroMemoryReport::default();
        // SAFETY: the `Engine` invariant holds and `report` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_memory_report(self.raw.as_ptr(), &mut report) })?;
        Ok(crate::memory_report_from_ffi(report))
    }

    /// Monotonic prefix-reuse counters of the engine's context, same contract
    /// as [`crate::Trainer::generation_stats`].
    pub fn generation_stats(&self) -> std::result::Result<GenerationStats, EngineError> {
        let mut stats = ffi::RetroGenerationStats::default();
        // SAFETY: the `Engine` invariant holds and `stats` is a valid out-pointer.
        self.check(unsafe { ffi::retro_engine_generation_stats(self.raw.as_ptr(), &mut stats) })?;
        Ok(GenerationStats {
            calls: stats.calls,
            sequences: stats.sequences,
            prompt_tokens: stats.prompt_tokens,
            prefilled_tokens: stats.prefilled_tokens,
            reused_tokens: stats.reused_tokens,
            hits: stats.hits,
            evictions: stats.evictions,
        })
    }

    fn check(&self, code: i32) -> std::result::Result<(), EngineError> {
        if code == 0 {
            Ok(())
        } else {
            Err(EngineError::Runtime(ffi::last_error()))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: the `Engine` invariant holds.
        unsafe {
            ffi::retro_engine_free(self.raw.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_core::{Device, EngineKvDtype};

    #[test]
    fn engine_error_maps_into_core_error_variant_by_variant() {
        let invalid: retrograd_core::Error = EngineError::invalid("bad shape").into();
        assert!(matches!(invalid, retrograd_core::Error::InvalidArgument(_)));
        assert!(invalid.is_user_error());

        let overflow: retrograd_core::Error = EngineError::Overflow("too big".into()).into();
        assert!(matches!(overflow, retrograd_core::Error::Overflow(_)));

        let runtime: retrograd_core::Error = EngineError::Runtime("decode failed".into()).into();
        assert!(matches!(runtime, retrograd_core::Error::Runtime(_)));
        assert!(!runtime.is_user_error());

        // A core error carried through the Core variant comes back unchanged.
        let core = retrograd_core::Error::tokenize("empty prompt");
        let round: retrograd_core::Error = EngineError::from(core).into();
        assert!(matches!(round, retrograd_core::Error::Tokenize(_)));
    }

    #[test]
    fn generation_batch_follows_the_runtime_default_and_the_logits_budget() {
        let training = TrainConfig {
            n_ctx: 1024,
            generation_concurrency: 4,
            ..TrainConfig::default()
        };
        // A small vocabulary: the 512-token launch binds.
        assert_eq!(generation_batch(&training, 1000), 512);
        // A 151 936-token vocabulary affords 110 rows of logits in 64 MiB.
        assert_eq!(generation_batch(&training, 151_936), 110);
        // A context below 512 binds first.
        let short = TrainConfig {
            n_ctx: 64,
            ..training.clone()
        };
        assert_eq!(generation_batch(&short, 1000), 64);
        // An explicit value wins, but never below one row per sequence.
        let explicit = TrainConfig {
            generation_batch: 2,
            ..training.clone()
        };
        assert_eq!(generation_batch(&explicit, 151_936), 4);

        let config = engine_config_from_train_config(&training, 151_936);
        assert_eq!((config.n_batch, config.n_ubatch), (110, 110));
        assert_eq!(config.n_ctx, 1024 * 4);
        assert_eq!(config.n_seq_max, 4);
    }

    #[test]
    fn engine_config_as_ffi_maps_every_field_and_sets_the_evolution_key() {
        let config = EngineConfig {
            n_ctx: 256,
            n_seq_max: 4,
            n_batch: 64,
            n_ubatch: 32,
            kv_dtype: EngineKvDtype::Q8_0,
            flash_attn: true,
            kv_unified: false,
            device: Device::Cpu,
            threads: 6,
            verbose: true,
        };
        let ffi = engine_config_to_ffi(&config).unwrap();
        assert_eq!(
            ffi.struct_size as usize,
            size_of::<ffi::RetroEngineConfig>()
        );
        assert_eq!(ffi.n_ctx, 256);
        assert_eq!(ffi.n_seq_max, 4);
        assert_eq!(ffi.n_batch, 64);
        assert_eq!(ffi.n_ubatch, 32);
        assert_eq!(ffi.kv_dtype, ffi::RETRO_ENGINE_KV_DTYPE_Q8_0);
        assert!(ffi.flash_attn);
        assert_eq!(ffi.device, 1);
        assert_eq!(ffi.threads, 6);
        assert!(ffi.verbose);
        assert_eq!(ffi.kv_unified, 0);
    }

    #[test]
    fn engine_config_validation_refuses_bad_geometry_and_quantized_without_fa() {
        let base = EngineConfig::default();
        assert!(engine_config_to_ffi(&base).is_ok());

        let zero_ctx = EngineConfig {
            n_ctx: 0,
            ..base.clone()
        };
        let error = engine_config_to_ffi(&zero_ctx).unwrap_err();
        assert!(matches!(error, EngineError::InvalidArgument(_)));

        let wide_ubatch = EngineConfig {
            n_ubatch: 1024,
            n_batch: 512,
            ..base.clone()
        };
        assert!(engine_config_to_ffi(&wide_ubatch).is_err());

        for invalid in [
            EngineConfig {
                n_ctx: u32::MAX,
                ..base.clone()
            },
            EngineConfig {
                n_batch: u32::MAX,
                ..base.clone()
            },
            EngineConfig {
                n_ubatch: u32::MAX,
                n_batch: u32::MAX,
                ..base.clone()
            },
            EngineConfig {
                threads: u32::MAX,
                ..base.clone()
            },
            EngineConfig {
                n_seq_max: 257,
                ..base.clone()
            },
            EngineConfig {
                n_ctx: 1,
                n_seq_max: 2,
                ..base.clone()
            },
            EngineConfig {
                n_seq_max: 65,
                n_batch: 64,
                n_ubatch: 64,
                ..base.clone()
            },
        ] {
            assert!(matches!(
                engine_config_to_ffi(&invalid),
                Err(EngineError::InvalidArgument(_))
            ));
        }

        let quantized_no_fa = EngineConfig {
            kv_dtype: EngineKvDtype::Q4_0,
            flash_attn: false,
            ..base.clone()
        };
        let error = engine_config_to_ffi(&quantized_no_fa).unwrap_err();
        assert!(matches!(error, EngineError::InvalidArgument(_)));
        assert!(error.to_string().contains("flash_attn"));
    }
}
