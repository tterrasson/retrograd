use super::*;

impl Trainer {
    /// Loads a GGUF model and builds a training context for it. No LoRA
    /// adapter is created yet; call [`Trainer::create_lora`] or
    /// [`Trainer::load_lora`] before the first optimizer step.
    pub fn new(model_path: impl AsRef<Path>, config: TrainConfig) -> Result<Self> {
        if !config.trainable.optimizer.is_implemented() {
            return Err(Error::invalid(format!(
                "optimizer {} is not available in this build; use adamw or sgd",
                config.trainable.optimizer
            )));
        }
        let started = Instant::now();
        let model_path = path_to_cstring(model_path.as_ref())?;
        let ffi_config = train_config_to_ffi(&config)?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        let raw = unsafe { ffi::retro_trainer_new(model_path.as_ptr(), &ffi_config) };
        let result = NonNull::new(raw).ok_or_else(runtime_error).map(|raw| Self {
            raw,
            trainable_policy: config.trainable.policy,
            declared_trainable: None,
            declared_assignment: Vec::new(),
            reference: None,
            reference_path: None,
        });
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

    /// Declares the resolved base trainable set, by canonical tensor name.
    ///
    /// Resolution happens outside the runtime, against the GGUF's tensor
    /// inventory, so a frontend can answer "what would this train, and what
    /// would it cost" before a model is open. This is where the answer is
    /// handed over, and the runtime validates it against the model it actually
    /// loaded: a name no tensor carries is an error rather than a silently
    /// smaller update.
    ///
    /// Must be called before the first training step or
    /// [`Trainer::prepare_optimizer`], and only on a trainer whose
    /// [`TrainConfig`] names a base-weight policy.
    /// Declares the resolved trainable set: its base names reach the runtime's
    /// parameter filter, and the set itself is kept for the checkpoint
    /// signature, which a resume needs before the optimizer graph exists.
    pub fn declare_trainable_set(&mut self, set: &retrograd_core::TrainableSet) -> Result<()> {
        let names: Vec<String> = set.base_entries().map(|entry| entry.name.clone()).collect();
        self.set_trainable_base(&names)?;
        self.declared_trainable = Some(set.clone());
        Ok(())
    }

    pub fn set_trainable_base(&mut self, names: &[String]) -> Result<()> {
        let owned = names
            .iter()
            .map(|name| CString::new(name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(nul_error)?;
        let pointers: Vec<*const c_char> = owned.iter().map(|name| name.as_ptr()).collect();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_trainable_base(
                self.raw.as_ptr(),
                pointers.as_ptr(),
                pointers.len(),
            )
        })?;
        self.declared_trainable = None;
        Ok(())
    }

    /// Declares which optimizer owns each marked parameter. Empty for a
    /// single-optimizer run: the optimizer the [`TrainConfig`] names owns
    /// everything.
    ///
    /// Must be called before [`Trainer::prepare_optimizer`]. A row naming a
    /// parameter this run does not train is refused once the marked set
    /// exists.
    pub fn set_optimizer_assignment(
        &mut self,
        assignment: &[(String, retrograd_core::OptimizerKind)],
    ) -> Result<()> {
        let owned = assignment
            .iter()
            .map(|(name, _)| CString::new(name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(nul_error)?;
        let pointers: Vec<*const c_char> = owned.iter().map(|name| name.as_ptr()).collect();
        let optimizers = assignment
            .iter()
            .map(|(_, optimizer)| optimizer.as_ffi())
            .collect::<Result<Vec<i32>>>()?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_optimizer_assignment(
                self.raw.as_ptr(),
                pointers.as_ptr(),
                optimizers.as_ptr(),
                pointers.len(),
            )
        })?;
        self.declared_assignment = assignment.to_vec();
        Ok(())
    }

    /// The assignment an [`OptimizerPlan`](retrograd_core::OptimizerPlan)
    /// declares, ready for [`Trainer::set_optimizer_assignment`]. A parameter
    /// the plan leaves unwritable is refused.
    pub fn assignment_from_plan(
        plan: &retrograd_core::OptimizerPlan,
    ) -> Result<Vec<(String, retrograd_core::OptimizerKind)>> {
        let unwritable = plan.unwritable();
        if !unwritable.is_empty() {
            return Err(Error::invalid(format!(
                "no optimizer in this run can write {} parameter(s): [{}]",
                unwritable.len(),
                unwritable.join(", ")
            )));
        }
        Ok(plan
            .parameters
            .iter()
            .filter_map(|parameter| {
                parameter
                    .optimizer
                    .map(|optimizer| (parameter.name.clone(), optimizer))
            })
            .collect())
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
mod reference;
mod score;
mod tokenize;

/// Chat-template and tool-call-parsing helpers that take a template or parser
/// directly instead of a loaded model; see [`Trainer::format_chat_messages`]
/// and [`Trainer::tool_call_parser`] for the model-backed equivalents.
pub use reference::{VOCABULARY_WITNESSES, VocabularyMismatch};
pub use tokenize::{
    parse_assistant_output, render_chat_template_source, tool_call_parser_from_source,
};
mod train;
