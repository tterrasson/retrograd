use super::*;

impl Trainer {
    /// Loads a GGUF model and builds a training context for it. No LoRA
    /// adapter is created yet; call [`Trainer::create_lora`] or
    /// [`Trainer::load_lora`] before the first optimizer step.
    pub fn new(model_path: impl AsRef<Path>, config: TrainConfig) -> Result<Self> {
        let started = Instant::now();
        let model_path = path_to_cstring(model_path.as_ref())?;
        let ffi_config = train_config_to_ffi(&config);
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        let raw = unsafe { ffi::retro_trainer_new(model_path.as_ptr(), &ffi_config) };
        let result = NonNull::new(raw)
            .ok_or_else(runtime_error)
            .map(|raw| Self { raw });
        test_timing("model_load", started);
        let trainer = result?;
        // Only when one was asked for. `None` is the unthrottled path and must
        // not cross the FFI boundary at all: the setter would install a limiter
        // that reports `requested = 1.0` and never sleeps, which is the same
        // behaviour by a longer route and one more thing to keep true.
        //
        // The setting is deliberately not a field of `RetroTrainConfig`: that
        // layout is published and guarded by exact size and offset tests, and an
        // execution policy that can change between two operations does not
        // belong in the document describing the training problem.
        if let Some(fraction) = config.max_gpu_duty_cycle {
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            trainer.check(unsafe {
                ffi::retro_trainer_set_max_gpu_duty_cycle(trainer.raw.as_ptr(), fraction)
            })?;
        }
        Ok(trainer)
    }

    /// Attaches a fresh, randomly initialized LoRA adapter to the model,
    /// targeting the tensors matched by `config.targets`.
    pub fn create_lora(&mut self, config: &LoraConfig) -> Result<()> {
        if config.rank == 0 {
            return Err(Error::invalid("LoRA rank must be greater than zero"));
        }

        let patterns = config.targets.patterns();
        let c_patterns = patterns
            .iter()
            .map(|pattern| CString::new(pattern.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(nul_error)?;
        let pattern_ptrs = c_patterns
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let ffi_config = ffi::RetroLoraConfig {
            rank: config.rank,
            alpha: config.alpha,
            dropout: config.dropout,
            seed: config.seed,
            target_patterns: pattern_ptrs.as_ptr(),
            n_target_patterns: pattern_ptrs.len(),
            dtype: config.dtype.as_ffi(),
        };

        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_create_lora(self.raw.as_ptr(), &ffi_config) })
    }

    /// Attaches a LoRA adapter loaded from a GGUF file, replacing any adapter
    /// already attached.
    pub fn load_lora(&mut self, adapter_path: impl AsRef<Path>) -> Result<()> {
        let adapter_path = path_to_cstring(adapter_path.as_ref())?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_load_lora(self.raw.as_ptr(), adapter_path.as_ptr())
        })
    }

    /// Bounds the fraction of wall time this trainer spends waiting on GPU work
    /// it submitted, so another workload can use the device in between. `1.0`
    /// means no limit and clears every window and counter the limiter holds.
    ///
    /// Safe between trainer operations, never during one. Deliberately a call
    /// rather than a [`TrainConfig`] field the runtime reads: the published
    /// `retro_train_config` layout is guarded by exact size and offset tests,
    /// and an execution policy that may change between two operations does not
    /// belong in the document describing the training problem. It is also the
    /// shape a live server adjustment would need, without another ABI change.
    pub fn set_max_gpu_duty_cycle(&mut self, fraction: f32) -> Result<()> {
        // Validated here as well as at the ABI, so a caller of *this* API gets
        // `InvalidArgument` - which frontends branch on for 422 vs 500 - rather
        // than the runtime error the C guard produces for its own callers.
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(Error::invalid(
                "max_gpu_duty_cycle must be finite and in (0, 1]",
            ));
        }
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_max_gpu_duty_cycle(self.raw.as_ptr(), fraction)
        })
    }

    fn check(&self, code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(runtime_error())
        }
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        unsafe {
            ffi::retro_trainer_free(self.raw.as_ptr());
        }
    }
}

impl retrograd_dataset::DatasetBackend for Trainer {
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        Trainer::tokenize_text(self, text)
    }

    fn eos_token(&self) -> Result<i32> {
        Trainer::eos_token(self)
    }

    fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
        Trainer::format_chat(self, messages, add_assistant)
    }
}

mod generate;
mod introspect;
#[path = "checkpoint.rs"]
mod persistence;
mod score;
mod tokenize;

/// Chat-template and tool-call-parsing helpers that take a template or parser
/// directly instead of a loaded model; see [`Trainer::format_chat_messages`]
/// and [`Trainer::tool_call_parser`] for the model-backed equivalents.
pub use tokenize::{
    parse_assistant_output, render_chat_template_source, tool_call_parser_from_source,
};
mod train;
