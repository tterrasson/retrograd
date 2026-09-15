use super::*;

impl Trainer {
    /// Runs one causal-LM optimizer step over a single token sequence, using
    /// every position but the first as a teacher-forced target.
    pub fn train_tokens(&mut self, tokens: &[i32]) -> Result<TrainMetrics> {
        validate_train_tokens(tokens)?;

        let started = Instant::now();
        let mut metrics = ffi::RetroTrainMetrics::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_train_tokens(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                &mut metrics,
            )
        })?;
        let metrics = train_metrics_from_ffi(metrics);
        test_timing("train_tokens", started);
        Ok(metrics)
    }

    fn run_with_callback<F, C>(&mut self, on_progress: F, call: C) -> Result<(TrainMetrics, bool)>
    where
        F: FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
        C: FnOnce(
            *mut ffi::RetroTrainer,
            *mut ffi::RetroTrainMetrics,
            Option<ffi::RetroTrainProgressCallback>,
            *mut c_void,
        ) -> i32,
    {
        let mut metrics = ffi::RetroTrainMetrics::default();
        let trainer = self as *mut Trainer;
        let mut state = FfiCallbackState {
            trainer,
            callback: on_progress,
            error: None,
            keep_training: true,
        };
        let user_data = &mut state as *mut FfiCallbackState<F> as *mut c_void;
        let runtime_result = self.check(call(
            self.raw.as_ptr(),
            &mut metrics,
            Some(ffi_callback_trampoline::<F>),
            user_data,
        ));
        if let Some(error) = state.error {
            return Err(error);
        }
        runtime_result?;
        Ok((train_metrics_from_ffi(metrics), state.keep_training))
    }

    /// Runs the SFT algorithm over rows prepared by [`retrograd_dataset::prepare`].
    pub fn train_sft_with_progress<F>(
        &mut self,
        train: &PreparedDataset,
        eval: Option<&PreparedDataset>,
        mut on_progress: F,
    ) -> Result<TrainMetrics>
    where
        F: FnMut(TrainMetrics),
    {
        self.train_sft_controlled(train, eval, |_, metrics| {
            on_progress(metrics);
            Ok(true)
        })
    }

    /// Runs SFT with a synchronous callback that can inspect the trainer at
    /// safe callback points and stop the loop by returning `false`.
    pub fn train_sft_controlled<F>(
        &mut self,
        train: &PreparedDataset,
        eval: Option<&PreparedDataset>,
        on_progress: F,
    ) -> Result<TrainMetrics>
    where
        F: FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
    {
        let span = tracing::info_span!(
            target: "retrograd::engine::update",
            "update",
            algorithm = "sft",
            rows = train.rows()
        );
        let _entered = span.enter();
        if train.is_empty() {
            return Err(Error::invalid("training dataset must not be empty"));
        }
        if train.n_ctx == 0 {
            return Err(Error::invalid("training dataset context must not be zero"));
        }
        if train.tokens.len() != train.rows() * train.n_ctx
            || train.labels.len() != train.tokens.len()
        {
            return Err(Error::invalid("invalid prepared training dataset shape"));
        }
        if let Some(eval) = eval {
            if eval.n_ctx != train.n_ctx {
                return Err(Error::invalid(
                    "training and evaluation contexts must match",
                ));
            }
            if eval.tokens.len() != eval.rows() * eval.n_ctx
                || eval.labels.len() != eval.tokens.len()
            {
                return Err(Error::invalid("invalid prepared evaluation dataset shape"));
            }
        }

        let train_ffi = ffi::RetroSftDataset {
            tokens: train.tokens.as_ptr(),
            labels: train.labels.as_ptr(),
            n_rows: train.rows(),
            n_ctx: train.n_ctx as u32,
        };
        let eval_ffi = eval.map(|dataset| ffi::RetroSftDataset {
            tokens: dataset.tokens.as_ptr(),
            labels: dataset.labels.as_ptr(),
            n_rows: dataset.rows(),
            n_ctx: dataset.n_ctx as u32,
        });
        let (metrics, _) = self.run_with_callback(
            on_progress,
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            |trainer, metrics, callback, user_data| unsafe {
                ffi::retro_trainer_train_sft(
                    trainer,
                    &train_ffi,
                    eval_ffi.as_ref().map_or(std::ptr::null(), |value| value),
                    metrics,
                    callback,
                    user_data,
                )
            },
        )?;
        Ok(metrics)
    }

    /// Scores every non-ignored label in fixed-size SFT rows without creating
    /// an optimizer. This works with the base model alone and automatically
    /// includes a loaded LoRA adapter when present.
    pub fn eval_sft(&mut self, data: &PreparedDataset) -> Result<EvalMetrics> {
        if data.is_empty() || data.n_ctx == 0 {
            return Err(Error::invalid("evaluation dataset must contain rows"));
        }
        let expected = data
            .rows()
            .checked_mul(data.n_ctx)
            .ok_or_else(|| Error::overflow("evaluation dataset shape overflows usize"))?;
        if data.tokens.len() != expected || data.labels.len() != expected {
            return Err(Error::invalid("invalid prepared evaluation dataset shape"));
        }
        let ffi_data = ffi::RetroSftDataset {
            tokens: data.tokens.as_ptr(),
            labels: data.labels.as_ptr(),
            n_rows: data.rows(),
            n_ctx: data.n_ctx as u32,
        };
        let mut metrics = ffi::RetroEvalMetrics::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_eval_sft(self.raw.as_ptr(), &ffi_data, &mut metrics)
        })?;
        Ok(EvalMetrics {
            negative_log_likelihood: metrics.negative_log_likelihood,
            supervised_tokens: metrics.supervised_tokens,
        })
    }

    /// Runs one weighted optimization step over a [`WeightedBatch`] - a
    /// per-token target/weight batch used by policy-gradient objectives (PPO)
    /// and top-k distillation alike. `scheduler_total_steps` pins the
    /// LR-schedule horizon across repeated calls; `0` extends it to just this
    /// call's steps.
    pub fn train_weighted(
        &mut self,
        batch: &WeightedBatch,
        scheduler_total_steps: u64,
    ) -> Result<TrainMetrics> {
        self.train_weighted_controlled(batch, scheduler_total_steps, |_, _| Ok(true))
            .map(|(metrics, _)| metrics)
    }

    /// Weighted optimization with a callback after every optimizer step.
    pub fn train_weighted_controlled<F>(
        &mut self,
        batch: &WeightedBatch,
        scheduler_total_steps: u64,
        on_progress: F,
    ) -> Result<(TrainMetrics, bool)>
    where
        F: FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
    {
        let span = tracing::info_span!(
            target: "retrograd::engine::update",
            "update",
            algorithm = "weighted",
            rows = batch.n_rows
        );
        let _entered = span.enter();
        batch.validate()?;
        let data = ffi::RetroWeightedDataset {
            tokens: batch.tokens.as_ptr(),
            labels: batch.labels.as_ptr(),
            weights: batch.weights.as_ptr(),
            n_rows: batch.n_rows,
            n_ctx: batch.n_ctx as u32,
            n_topk: batch.n_topk as u32,
        };
        self.run_with_callback(
            on_progress,
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            |trainer, metrics, callback, user_data| unsafe {
                ffi::retro_trainer_train_weighted(
                    trainer,
                    &data,
                    scheduler_total_steps,
                    metrics,
                    callback,
                    user_data,
                )
            },
        )
    }

    /// Runs one optimizer step over a packed set of independent sequences.
    /// Tokens may belong to several sequences, which is how prompt prefixes
    /// remain shared differentiable nodes across their completion branches.
    pub fn train_packed_sequences_controlled<F>(
        &mut self,
        batch: &PackedSequenceBatch,
        scheduler_total_steps: u64,
        accumulation_steps: u32,
        on_progress: F,
    ) -> Result<(TrainMetrics, bool)>
    where
        F: FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
    {
        let span = tracing::info_span!(
            target: "retrograd::engine::update",
            "update",
            algorithm = "packed",
            sequences = batch.n_sequences
        );
        let _entered = span.enter();
        batch.validate()?;
        let data = ffi::RetroPackedSequenceBatch {
            tokens: batch.tokens.as_ptr(),
            labels: batch.labels.as_ptr(),
            weights: batch.weights.as_ptr(),
            positions: batch.positions.as_ptr(),
            seq_offsets: batch.seq_offsets.as_ptr(),
            seq_ids: batch.seq_ids.as_ptr(),
            n_tokens: batch.tokens.len(),
            n_seq_ids: batch.seq_ids.len(),
            n_sequences: batch.n_sequences as u32,
            n_topk: batch.n_topk as u32,
        };
        self.run_with_callback(
            on_progress,
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            |trainer, metrics, callback, user_data| unsafe {
                ffi::retro_trainer_train_packed_sequences(
                    trainer,
                    &data,
                    scheduler_total_steps,
                    accumulation_steps,
                    metrics,
                    callback,
                    user_data,
                )
            },
        )
    }

    /// Consumes logical learning-rate scheduler slots without optimizing.
    /// GRPO uses this when filtering rollouts so the configured schedule does
    /// not stretch as the effective training fraction falls.
    pub fn advance_scheduler_steps(&mut self, steps: u64) -> Result<u64> {
        let mut global_step = 0_u64;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_advance_scheduler_steps(self.raw.as_ptr(), steps, &mut global_step)
        })?;
        Ok(global_step)
    }

    /// Replaces the base learning rate the schedule multiplies, taking effect at
    /// the next optimizer step.
    ///
    /// The step counter, the horizon and the AdamW momenta are untouched, so the
    /// warm-up and decay curve keeps its shape around the new base. That is what
    /// makes this callable from a progress callback - the one safe interruption
    /// point a run has - rather than only at construction.
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_learning_rate(self.raw.as_ptr(), learning_rate)
        })
    }
}
