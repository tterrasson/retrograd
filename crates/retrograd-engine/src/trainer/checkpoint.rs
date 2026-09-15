use super::*;

impl Trainer {
    /// Declares that `completed_epochs` SFT epochs are already done, so the
    /// next SFT run starts there and keeps the restored scheduler step. The
    /// schedule horizon stays the full configured run. Rollout algorithms do
    /// not need this: their scheduler step already accumulates across calls.
    pub fn set_resume_point(&mut self, completed_epochs: u32) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_resume_point(self.raw.as_ptr(), completed_epochs)
        })
    }

    /// Builds the optimizer graph if it does not exist yet, so the AdamW
    /// momenta are allocated and can be restored before the first step.
    pub fn prepare_optimizer(&mut self) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_prepare_optimizer(self.raw.as_ptr()) })
    }

    fn optimizer_state(&mut self) -> Result<ffi::RetroOptimizerState> {
        let mut state = ffi::RetroOptimizerState::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_optimizer_state(self.raw.as_ptr(), &mut state) })?;
        Ok(state)
    }

    /// Reads every AdamW momenta pair, keyed by the parameter's stable tensor
    /// name. Empty before the optimizer graph exists.
    fn read_moments(&mut self) -> Result<Vec<checkpoint::Moments>> {
        let mut count = 0_usize;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_momenta_count(self.raw.as_ptr(), &mut count) })?;
        let mut moments = Vec::with_capacity(count);
        for index in 0..count {
            let mut shape = [0_i64; 4];
            let mut n_elements = 0_usize;
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            let name = read_string(|buffer, n_buffer, out| unsafe {
                ffi::retro_trainer_momenta_info(
                    self.raw.as_ptr(),
                    index,
                    buffer,
                    n_buffer,
                    out,
                    shape.as_mut_ptr(),
                    &mut n_elements,
                )
            })?;
            let mut m = vec![0.0_f32; n_elements];
            let mut v = vec![0.0_f32; n_elements];
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            self.check(unsafe {
                ffi::retro_trainer_momenta_read(
                    self.raw.as_ptr(),
                    index,
                    m.as_mut_ptr(),
                    v.as_mut_ptr(),
                    n_elements,
                )
            })?;
            moments.push(checkpoint::Moments { name, shape, m, v });
        }
        Ok(moments)
    }

    /// Restores momenta by name. Requires [`Trainer::prepare_optimizer`]; a
    /// name, shape, or length that does not match the live parameter is an
    /// error rather than a partial restore.
    fn write_moments(&mut self, moments: &[checkpoint::Moments]) -> Result<()> {
        let live = self.read_moments()?;
        if live.len() != moments.len() {
            return Err(Error::runtime(format!(
                "checkpoint holds momenta for {} parameters, the model has {}",
                moments.len(),
                live.len()
            )));
        }
        for entry in moments {
            let matching = live
                .iter()
                .find(|candidate| candidate.name == entry.name)
                .ok_or_else(|| {
                    Error::runtime(format!(
                        "checkpoint holds momenta for unknown parameter '{}'",
                        entry.name
                    ))
                })?;
            if matching.shape != entry.shape {
                return Err(Error::runtime(format!(
                    "parameter '{}' has shape {:?} in the checkpoint and {:?} in the model",
                    entry.name, entry.shape, matching.shape
                )));
            }
            if entry.m.len() != entry.v.len() {
                return Err(Error::runtime(format!(
                    "parameter '{}' has mismatched m and v lengths",
                    entry.name
                )));
            }
            let name = CString::new(entry.name.as_str()).map_err(nul_error)?;
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            self.check(unsafe {
                ffi::retro_trainer_momenta_write(
                    self.raw.as_ptr(),
                    name.as_ptr(),
                    entry.m.as_ptr(),
                    entry.v.as_ptr(),
                    entry.m.len(),
                )
            })?;
        }
        Ok(())
    }

    read_string_method!(private rng_state, retro_trainer_rng_state);

    fn set_rng_state(&mut self, state: &str) -> Result<()> {
        let state = CString::new(state).map_err(nul_error)?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_set_rng_state(self.raw.as_ptr(), state.as_ptr()) })
    }

    /// Writes a complete training checkpoint. The authoritative LoRA GGUF and
    /// every piece of resume state land atomically inside `.state`; a plain
    /// sibling GGUF is then published for ordinary cold-adapter loading.
    ///
    /// `state_dir` may be given as either the state directory or the adapter
    /// GGUF path; the sibling is derived from it.
    pub fn save_checkpoint(
        &mut self,
        state_dir: impl AsRef<Path>,
        metadata: &CheckpointMetadata,
    ) -> Result<()> {
        let state_dir = checkpoint::state_dir_for(state_dir.as_ref());
        let adapter = checkpoint::ADAPTER_FILE.to_string();

        let optimizer_state = self.optimizer_state()?;
        // Before the first step the optimizer graph does not exist, so there
        // are no momenta to save and none to demand back on resume.
        let moments = if optimizer_state.has_momenta {
            self.read_moments()?
        } else {
            Vec::new()
        };
        let runtime_mt19937 = optimizer_state
            .has_momenta
            .then(|| self.rng_state())
            .transpose()?;
        let model_bytes = std::fs::metadata(&metadata.model_path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let model_fingerprint = checkpoint::fingerprint_file_cached(&metadata.model_path)?;
        let signature = self.model_signature()?;

        let mut artifacts = std::collections::BTreeMap::new();
        let mut policies = std::collections::BTreeMap::new();
        for (name, (policy, bytes)) in &metadata.artifacts {
            policies.insert(name.clone(), *policy);
            artifacts.insert(name.clone(), bytes.clone());
        }

        let record = checkpoint::Checkpoint {
            manifest: checkpoint::Manifest {
                format_version: checkpoint::FORMAT_VERSION,
                checkpoint_id: metadata.checkpoint_id.clone(),
                global_step: metadata.progress.global_step,
                adapter,
                files: checkpoint::REQUIRED_FILES
                    .iter()
                    .map(|name| name.to_string())
                    .collect(),
                app_version: env!("CARGO_PKG_VERSION").to_string(),
                llama_cpp_commit: ffi::LLAMA_CPP_COMMIT.to_string(),
                model_signature: signature,
                model_bytes,
                model_fingerprint,
                algorithm: metadata.algorithm.clone(),
                trajectory_signature: metadata.trajectory_signature.clone(),
                resume_boundary: metadata.resume_boundary.clone(),
                artifacts: policies,
            },
            progress: metadata.progress.clone(),
            scheduler: checkpoint::Scheduler {
                version: checkpoint::FORMAT_VERSION,
                step: optimizer_state.scheduler_step,
                total_steps: optimizer_state.scheduler_total_steps,
                last_learning_rate: optimizer_state.last_learning_rate,
                kind: metadata.scheduler_kind.clone(),
                learning_rate: optimizer_state.learning_rate,
                warmup_steps: metadata.warmup_steps,
            },
            optimizer: checkpoint::Optimizer {
                version: checkpoint::FORMAT_VERSION,
                kind: if optimizer_state.optimizer == 1 {
                    "sgd".into()
                } else {
                    "adamw".into()
                },
                learning_rate: optimizer_state.learning_rate,
                weight_decay: optimizer_state.weight_decay,
                max_grad_norm: optimizer_state.max_grad_norm,
                iter: optimizer_state.iter,
                has_moments: optimizer_state.has_momenta,
                moments,
            },
            rng: checkpoint::Rng {
                version: checkpoint::FORMAT_VERSION,
                runtime_mt19937,
                seeds: metadata.seeds.clone(),
            },
            dataset: metadata.dataset.clone(),
            artifacts,
        };
        record.write(&state_dir, |path| self.save_lora(path))
    }

    /// Restores a checkpoint: loads its adapter, validates it against the run
    /// configuration, then applies the optimizer, scheduler, and RNG state.
    /// Nothing is written into the trainer before validation succeeds.
    pub fn load_checkpoint(
        &mut self,
        state_dir: impl AsRef<Path>,
        expected: &checkpoint::Compatibility,
    ) -> Result<ResumeInfo> {
        let state_dir = checkpoint::state_dir_for(state_dir.as_ref());
        let record = checkpoint::Checkpoint::read(&state_dir)?;
        record.check_compatible(expected)?;

        let adapter = checkpoint::adapter_for(&state_dir, &record.manifest);
        if !adapter.is_file() {
            return Err(Error::runtime(format!(
                "checkpoint adapter {} is missing",
                adapter.display()
            )));
        }
        self.load_lora(&adapter)?;

        if record.optimizer.has_moments {
            // The momenta only exist once the optimizer graph is built, which
            // is why the restore is deferred to here rather than done at read.
            self.prepare_optimizer()?;
            self.write_moments(&record.optimizer.moments)?;
            if let Some(rng) = &record.rng.runtime_mt19937 {
                self.set_rng_state(rng)?;
            }
        }
        let state = ffi::RetroOptimizerState {
            iter: record.optimizer.iter,
            has_momenta: record.optimizer.has_moments,
            optimizer: if record.optimizer.kind == "sgd" { 1 } else { 0 },
            learning_rate: record.optimizer.learning_rate,
            weight_decay: record.optimizer.weight_decay,
            max_grad_norm: record.optimizer.max_grad_norm,
            scheduler_step: record.scheduler.step,
            scheduler_total_steps: record.scheduler.total_steps,
            last_learning_rate: record.scheduler.last_learning_rate,
        };
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_restore_optimizer_state(self.raw.as_ptr(), &state)
        })?;

        Ok(ResumeInfo {
            adapter,
            progress: record.progress,
            dataset: record.dataset,
            seeds: record.rng.seeds,
            artifacts: record.artifacts,
            had_moments: record.optimizer.has_moments,
        })
    }
}
