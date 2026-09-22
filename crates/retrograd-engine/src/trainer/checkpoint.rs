use super::*;

use retrograd_core::{HyperparameterVector, TensorDtype, TensorRole, TrainableEntry, TrainableSet};

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

    /// Builds the optimizer graph if it does not exist yet, so any persistent
    /// slots are allocated and can be restored before the first step.
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

    /// The hyperparameters this run's optimizer reads, at the values it reads
    /// them.
    ///
    /// The declared layout filled in from the live runtime, never from the
    /// document: a configuration spells only part of the vector, the rest is
    /// ggml's own. Recorded in the checkpoint and compared on resume.
    /// The optimizer the runtime is running, at the layout this run declared.
    ///
    /// The wire carries the optimizer and, for Gefen, the variant - which is
    /// what decides *which* slot table. The rest of a layout (the block size,
    /// the fallback threshold) decides the table's shapes and who owns what,
    /// and it is the configuration's: the threshold never crosses at all,
    /// because its answer crosses as the assignment instead. So the run's own
    /// value is used whenever the two name the same optimizer under the same
    /// variant, and the live one whenever they do not - which is the
    /// disagreement `check_live` is there to report.
    fn live_optimizer_kind(&self, state: &ffi::RetroOptimizerState) -> Result<OptimizerKind> {
        let live = OptimizerKind::from_ffi(state.optimizer, state.gefen_variant)?;
        match (live, self.chosen_optimizer()) {
            (OptimizerKind::Gefen(reported), OptimizerKind::Gefen(declared)) => {
                if reported.variant == declared.variant {
                    Ok(OptimizerKind::Gefen(declared))
                } else {
                    // `live` only carries the variant the wire reports, filled
                    // out with `GefenLayout::default()` - not this run's real
                    // block_size/min_numel. Returning it here would make the
                    // caller's slot-size math compare against a layout neither
                    // side is actually running, surfacing as an opaque
                    // byte-size mismatch instead of this direct diagnosis.
                    Err(Error::runtime(format!(
                        "the runtime is running Gefen variant {}, but this run declared \
                         variant {}",
                        reported.variant, declared.variant
                    )))
                }
            }
            _ => Ok(live),
        }
    }

    pub fn optimizer_hyperparameters(&mut self) -> Result<HyperparameterVector> {
        let state = self.optimizer_state()?;
        let kind = self.live_optimizer_kind(&state)?;
        let mut vector = kind.declared_hyperparameters();
        vector.set_scalar("learning_rate", state.learning_rate)?;
        vector.set_scalar("weight_decay", state.weight_decay)?;
        vector.set_scalar("max_grad_norm", state.max_grad_norm)?;
        // Each optimizer's own rows, from the same live state as the three
        // above. A row the layout does not declare is never set: the vector
        // refuses it, and that refusal is the check that this match and the
        // declaration agree.
        match kind {
            OptimizerKind::AdamW => {
                vector.set_scalar("beta1", state.adamw_beta1)?;
                vector.set_scalar("beta2", state.adamw_beta2)?;
                vector.set_scalar("eps", state.adamw_eps)?;
            }
            OptimizerKind::Sgd => {}
            OptimizerKind::Muon => {
                vector.set_scalar("momentum", state.muon_momentum)?;
                vector.set_scalar("ns_epsilon", state.muon_ns_epsilon)?;
                vector.set_scalar("fallback_learning_rate", state.muon_fallback_learning_rate)?;
                vector.set(
                    "ns_steps",
                    retrograd_core::HyperparameterValue::Structural(state.muon_ns_steps.into()),
                )?;
                vector.set(
                    "nesterov",
                    retrograd_core::HyperparameterValue::Toggle(state.muon_nesterov),
                )?;
            }
            OptimizerKind::Gefen(layout) => {
                vector.set_scalar("beta1", state.gefen_beta1)?;
                vector.set_scalar("beta2", state.gefen_beta2)?;
                vector.set_scalar("eps", state.gefen_eps)?;
                vector.set(
                    "block_size",
                    retrograd_core::HyperparameterValue::Structural(state.gefen_block_size.into()),
                )?;
                // The threshold is the configuration's, for the reason
                // `live_optimizer_kind` gives.
                vector.set(
                    "min_numel",
                    retrograd_core::HyperparameterValue::Structural(
                        i64::try_from(layout.min_numel).unwrap_or(i64::MAX),
                    ),
                )?;
            }
        }
        Ok(vector)
    }

    /// Every tensor the optimizer marked trainable, as a resolved set.
    ///
    /// Read from the runtime rather than from the selector that produced it:
    /// this is what the update step actually writes, and the two are equal only
    /// because rule 6 compares them. Empty before the optimizer graph exists.
    pub fn marked_trainable_set(&mut self) -> Result<TrainableSet> {
        let mut count = 0_usize;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_marked_parameter_count(self.raw.as_ptr(), &mut count)
        })?;
        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let mut desc = ffi::RetroTensorDesc::default();
            // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
            self.check(unsafe {
                ffi::retro_trainer_marked_parameter_info(self.raw.as_ptr(), index, &mut desc)
            })?;
            let name = fixed_string(&desc.name)?;
            let dtype = TensorDtype::from_ggml_name(&fixed_string(&desc.type_name)?);
            entries.push(TrainableEntry {
                role: role_of(&name),
                name,
                ne: desc.ne,
                dtype,
                n_elements: desc.n_elements,
                n_bytes: desc.n_bytes,
                storage_id: desc.storage_id,
            });
        }
        Ok(TrainableSet {
            policy: self.trainable_policy(),
            entries,
            exclusions: Vec::new(),
        })
    }

    /// Copies one marked parameter's bytes, starting at `offset`, in the
    /// parameter's stored dtype (see [`Self::marked_trainable_set`]).
    pub fn read_marked_parameter(
        &mut self,
        index: usize,
        offset: u64,
        out: &mut [u8],
    ) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_marked_parameter_read(
                self.raw.as_ptr(),
                index,
                offset,
                out.as_mut_ptr().cast(),
                out.len(),
            )
        })
    }

    /// Describes the gradient accumulator of one marked parameter. Always F32
    /// and parameter-shaped; `role` is the parameter's, since that is how a
    /// caller identifies it.
    pub fn parameter_gradient_info(&mut self, index: usize) -> Result<TrainableEntry> {
        let mut desc = ffi::RetroTensorDesc::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_parameter_gradient_info(self.raw.as_ptr(), index, &mut desc)
        })?;
        let name = fixed_string(&desc.name)?;
        let dtype = TensorDtype::from_ggml_name(&fixed_string(&desc.type_name)?);
        Ok(TrainableEntry {
            role: role_of(&name),
            name,
            ne: desc.ne,
            dtype,
            n_elements: desc.n_elements,
            n_bytes: desc.n_bytes,
            storage_id: desc.storage_id,
        })
    }

    /// Copies bytes of the parameter's gradient accumulator, starting at
    /// `offset`.
    ///
    /// The accumulators outlive the graph that wrote them, so this answers
    /// after the step. That makes slot-less optimizers (SGD) checkable: the
    /// gradient is the only other input to their update.
    pub fn read_parameter_gradient(
        &mut self,
        index: usize,
        offset: u64,
        out: &mut [u8],
    ) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_parameter_gradient_read(
                self.raw.as_ptr(),
                index,
                offset,
                out.as_mut_ptr().cast(),
                out.len(),
            )
        })
    }

    /// Fingerprint of the marked set's canonical manifest, empty when the run
    /// trains no base tensor.
    ///
    /// Empty rather than a digest of the adapter on purpose: a LoRA run's
    /// identity is already pinned by the adapter file and the trajectory
    /// signature, and an extra comparison would refuse every resume written
    /// before this field existed without catching anything new.
    pub fn trainable_signature(&mut self) -> Result<String> {
        if !self.trains_base_weights() {
            return Ok(String::new());
        }
        // The declared set first: a resume compares this signature before the
        // optimizer graph exists, so the marked set is not available yet. Once
        // it is, rule 6 has already proved the two agree.
        if let Some(declared) = self.declared_trainable_set() {
            return Ok(set_signature(declared));
        }
        let signature = set_signature(&self.marked_trainable_set()?);
        if signature.is_empty() {
            // Neither source can answer, and an empty signature would compare
            // equal to a LoRA run's - which is the one comparison that must not
            // pass. `declare_trainable_set` is what resolves it.
            return Err(Error::invalid(
                "this run trains base weights but has not declared its resolved trainable \
                 set, so its checkpoint identity cannot be computed",
            ));
        }
        Ok(signature)
    }

    /// The slot table of the live optimizer, in file order, with the byte
    /// offsets a checkpoint addresses its payload by. Public so a declared
    /// plan can be compared against the live table without writing a
    /// checkpoint.
    pub fn state_slots(&mut self) -> Result<Vec<checkpoint::StateSlot>> {
        let mut parameter_slots = 0_usize;
        let mut shared_slots = 0_usize;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_state_slot_count(
                self.raw.as_ptr(),
                &mut parameter_slots,
                &mut shared_slots,
            )
        })?;
        let mut slots = Vec::with_capacity(parameter_slots + shared_slots);
        let mut offset = 0_u64;
        for (scope, ffi_scope, count) in [
            (
                checkpoint::SlotScope::Parameter,
                ffi::SLOT_SCOPE_PARAMETER,
                parameter_slots,
            ),
            (
                checkpoint::SlotScope::Shared,
                ffi::SLOT_SCOPE_SHARED,
                shared_slots,
            ),
        ] {
            for index in 0..count {
                let mut info = ffi::RetroOptimizerSlot::default();
                // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
                self.check(unsafe {
                    ffi::retro_trainer_state_slot_info(
                        self.raw.as_ptr(),
                        ffi_scope,
                        index,
                        &mut info,
                    )
                })?;
                slots.push(checkpoint::StateSlot {
                    scope,
                    owner: fixed_string(&info.owner)?,
                    slot: fixed_string(&info.slot)?,
                    dtype: fixed_string(&info.type_name)?,
                    shape: info.ne,
                    offset,
                    n_bytes: info.n_bytes,
                });
                offset = offset.checked_add(info.n_bytes).ok_or_else(|| {
                    Error::runtime("the optimizer state exceeds the addressable byte range")
                })?;
            }
        }
        Ok(slots)
    }

    /// Streams every slot payload into `sink`, in `slots` order, through one
    /// bounded staging buffer.
    ///
    /// Never a `Vec<Vec<u8>>`: the state of a fully trained model is the size
    /// of two more models, and collecting it before writing would make saving a
    /// checkpoint cost more host memory than training does.
    fn write_state_payload(
        &mut self,
        slots: &[checkpoint::StateSlot],
        sink: &mut impl std::io::Write,
    ) -> Result<()> {
        let mut staging = vec![0_u8; checkpoint::STAGING_CHUNK_BYTES];
        for (index, slot) in slots.iter().enumerate() {
            let scope = ffi_scope(slot.scope);
            // Index within the scope, which is what the runtime enumerates.
            let scoped_index = slots[..index]
                .iter()
                .filter(|earlier| earlier.scope == slot.scope)
                .count();
            let mut done = 0_u64;
            while done < slot.n_bytes {
                let want = (slot.n_bytes - done).min(staging.len() as u64) as usize;
                // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
                self.check(unsafe {
                    ffi::retro_trainer_state_slot_read(
                        self.raw.as_ptr(),
                        scope,
                        scoped_index,
                        done,
                        staging.as_mut_ptr().cast(),
                        want,
                    )
                })?;
                sink.write_all(&staging[..want])?;
                done += want as u64;
            }
        }
        Ok(())
    }

    /// Restores slot payloads by `(scope, owner, slot)`, validating the live
    /// layout against the record before a single byte is written.
    fn restore_state_payload(
        &mut self,
        state_dir: &Path,
        optimizer: &checkpoint::Optimizer,
    ) -> Result<()> {
        let live_state = self.optimizer_state()?;
        let kind = self.live_optimizer_kind(&live_state)?;
        let marked = self.marked_trainable_set()?;
        let plan = self.optimizer_plan(kind, &marked);
        optimizer.check_assignment(&assignment_of(&plan))?;
        let live = self.state_slots()?;
        if live.len() != optimizer.slots.len() {
            return Err(Error::runtime(format!(
                "the checkpoint holds {} optimizer slots, this run allocates {}",
                optimizer.slots.len(),
                live.len()
            )));
        }
        for saved in &optimizer.slots {
            let matching = live
                .iter()
                .find(|candidate| {
                    candidate.scope == saved.scope
                        && candidate.owner == saved.owner
                        && candidate.slot == saved.slot
                })
                .ok_or_else(|| {
                    Error::runtime(format!(
                        "the checkpoint holds a '{}' slot for {} '{}' that this run does not \
                         allocate",
                        saved.slot,
                        saved.scope.as_str(),
                        saved.owner
                    ))
                })?;
            if matching.shape != saved.shape
                || matching.dtype != saved.dtype
                || matching.n_bytes != saved.n_bytes
            {
                return Err(Error::runtime(format!(
                    "slot '{}' of '{}' is {} {:?} ({} bytes) in the checkpoint and {} {:?} \
                     ({} bytes) in this run",
                    saved.slot,
                    saved.owner,
                    saved.dtype,
                    saved.shape,
                    saved.n_bytes,
                    matching.dtype,
                    matching.shape,
                    matching.n_bytes
                )));
            }
        }

        if optimizer.slots.is_empty() {
            return Ok(());
        }
        let mut reader = checkpoint::OptimizerStateReader::open(state_dir, optimizer)?;
        let mut staging = vec![0_u8; checkpoint::STAGING_CHUNK_BYTES];
        for saved in &optimizer.slots {
            let owner = CString::new(saved.owner.as_str()).map_err(nul_error)?;
            let slot = CString::new(saved.slot.as_str()).map_err(nul_error)?;
            let scope = ffi_scope(saved.scope);
            let raw = self.raw.as_ptr();
            reader.stream(saved, &mut staging, |offset, chunk| {
                // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
                let code = unsafe {
                    ffi::retro_trainer_state_slot_write(
                        raw,
                        scope,
                        owner.as_ptr(),
                        slot.as_ptr(),
                        offset,
                        chunk.as_ptr().cast(),
                        chunk.len(),
                    )
                };
                if code == 0 {
                    Ok(())
                } else {
                    Err(runtime_error())
                }
            })?;
        }
        Ok(())
    }

    /// Writes the trained base tensors, by absolute value, as a GGUF bundle.
    pub fn save_trainable(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_save_trainable(self.raw.as_ptr(), path.as_ptr()) })
    }

    /// Restores base tensor values from such a bundle onto the live model.
    pub fn load_trainable(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_load_trainable(self.raw.as_ptr(), path.as_ptr()) })
    }

    /// Writes the whole model back out as a standalone GGUF, trained weights
    /// included. The result needs neither the source model nor this loader.
    ///
    /// Refused for a run that changes no base tensor or carries an adapter.
    /// Check [`retrograd_core::architecture_exports_model`] before training.
    pub fn save_model(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_save_model(self.raw.as_ptr(), path.as_ptr()) })
    }

    read_string_method!(private rng_state, retro_trainer_rng_state);

    fn set_rng_state(&mut self, state: &str) -> Result<()> {
        let state = CString::new(state).map_err(nul_error)?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_set_rng_state(self.raw.as_ptr(), state.as_ptr()) })
    }

    /// Disk size of one published checkpoint of this run.
    ///
    /// Read from the live state, and available before the optimizer graph is
    /// built, so a run can be refused for disk before it trains. The
    /// optimizer half is the state the run will keep, not the zero a
    /// checkpoint taken before the first step holds.
    pub fn checkpoint_footprint(&mut self) -> Result<checkpoint::CheckpointFootprint> {
        let report = self.memory_report()?;
        // Only the adapter half is copied out as a sibling GGUF, so the report
        // has to be split: the marked set is the exact split and exists once
        // the graph does; before that the declared set is all a preflight has,
        // and it is base-only by construction.
        let marked = self.marked_trainable_set()?;
        let trainable_bytes = if marked.is_empty() {
            self.declared_trainable_set()
                .map_or(0, |set| set.parameter_bytes())
        } else {
            marked
                .parameter_bytes()
                .saturating_sub(marked.parameter_bytes_on_top())
        };
        Ok(checkpoint::CheckpointFootprint {
            adapter_bytes: report
                .trainable_parameter_bytes
                .saturating_sub(trainable_bytes),
            trainable_bytes,
            optimizer_state_bytes: report.optimizer_state_bytes,
        })
    }

    /// Writes a complete training checkpoint: whatever the run produced - an
    /// adapter, a trainable bundle, or both - and every piece of resume state,
    /// landing atomically inside `.state`. A plain sibling GGUF is then
    /// published for ordinary cold-adapter loading, for a run that has one.
    ///
    /// `state_dir` may be given as either the state directory or the adapter
    /// GGUF path; the sibling is derived from it.
    pub fn save_checkpoint(
        &mut self,
        state_dir: impl AsRef<Path>,
        metadata: &CheckpointMetadata,
    ) -> Result<()> {
        let state_dir = checkpoint::state_dir_for(state_dir.as_ref());
        // Before anything is staged, so a too-small filesystem fails here
        // rather than mid-bundle, leaving a temporary directory.
        checkpoint::DiskBudget {
            footprint: self.checkpoint_footprint()?,
            retained: 0,
        }
        .check(&state_dir, "cannot write this checkpoint")?;
        let trains_base = self.trains_base_weights();
        // `full` and `partial` train base tensors and create no adapter;
        // `hybrid` trains both. Read off the policy rather than off whether an
        // adapter happens to exist, so a checkpoint taken before one is created
        // still declares the file the run will produce.
        let has_adapter = matches!(
            self.trainable_policy(),
            retrograd_core::TrainablePolicy::Lora | retrograd_core::TrainablePolicy::Hybrid
        );

        let optimizer_state = self.optimizer_state()?;
        // Before the first step the optimizer graph does not exist, so there
        // are no slots to save and none to demand back on resume. An optimizer
        // that keeps none is a different case: its graph exists, and its RNG
        // state and iteration counter are still worth saving.
        let slots = if optimizer_state.graph_ready {
            self.state_slots()?
        } else {
            Vec::new()
        };
        let state_bytes = slots.iter().map(|slot| slot.n_bytes).sum::<u64>();
        let marked = if optimizer_state.graph_ready {
            self.marked_trainable_set()?
        } else {
            TrainableSet::default()
        };
        let optimizer_kind = self.live_optimizer_kind(&optimizer_state)?;
        // Filled in from the same live state as the scalars above, so the
        // record and the resume comparison read one answer.
        let hyperparameters = self.optimizer_hyperparameters()?;
        // The declared state table, built before the live one is read, so the
        // two can be compared.
        let plan = self.optimizer_plan(optimizer_kind, &marked);
        let unwritable = plan.unwritable();
        if !unwritable.is_empty() {
            return Err(Error::invalid(format!(
                "optimizer {optimizer_kind} has no update step for {} marked parameter(s), \
                 starting with '{}': a checkpoint that recorded another optimizer for them \
                 would describe a trajectory this run did not take",
                unwritable.len(),
                unwritable[0]
            )));
        }
        if optimizer_state.graph_ready {
            plan.check_live(&live_rows(&slots, checkpoint::SlotScope::Parameter))?;
            plan.check_live_shared(&live_rows(&slots, checkpoint::SlotScope::Shared))?;
        }
        // The same source a resume reads, so the two sides compare what they
        // both can see rather than two renderings of the same set.
        let trainable_signature = self.trainable_signature()?;
        let runtime_mt19937 = optimizer_state
            .graph_ready
            .then(|| self.rng_state())
            .transpose()?;
        let model_bytes = std::fs::metadata(&metadata.model_path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let model_fingerprint = checkpoint::fingerprint_file_cached(&metadata.model_path)?;
        let reference_fingerprint = self.reference_fingerprint()?;
        let signature = self.model_signature()?;

        let mut artifacts = std::collections::BTreeMap::new();
        let mut policies = std::collections::BTreeMap::new();
        for (name, (policy, bytes)) in &metadata.artifacts {
            policies.insert(name.clone(), *policy);
            artifacts.insert(name.clone(), bytes.clone());
        }

        // The bundle's own description is assembled before the file exists: its
        // size and fingerprint are filled in after the payload writer has run,
        // which is the only moment both are knowable.
        let trainable = trains_base.then(|| checkpoint::TrainableBundle {
            file: checkpoint::TRAINABLE_FILE.to_string(),
            bytes: 0,
            fingerprint: String::new(),
            signature: trainable_signature,
            tensors: marked
                .base_entries()
                .map(|entry| checkpoint::TrainableTensor {
                    name: entry.name.clone(),
                    role: entry.role.as_str().to_string(),
                    dtype: entry.dtype.name().to_string(),
                    shape: entry.ne,
                    n_elements: entry.n_elements,
                    n_bytes: entry.n_bytes,
                    // No aliases here: these entries come from the marked set,
                    // and no-aliasing is the resolver's guarantee, not this writer's.
                    aliases: Vec::new(),
                })
                .collect(),
        });

        let record = checkpoint::Checkpoint {
            manifest: checkpoint::Manifest {
                format_version: checkpoint::FORMAT_VERSION,
                checkpoint_id: metadata.checkpoint_id.clone(),
                global_step: metadata.progress.global_step,
                adapter: has_adapter.then(|| checkpoint::ADAPTER_FILE.to_string()),
                trainable,
                trainable_policy: self.trainable_policy().as_str().to_string(),
                files: checkpoint::REQUIRED_FILES
                    .iter()
                    .map(|name| name.to_string())
                    .collect(),
                app_version: env!("CARGO_PKG_VERSION").to_string(),
                llama_cpp_commit: ffi::LLAMA_CPP_COMMIT.to_string(),
                model_signature: signature,
                model_bytes,
                model_fingerprint,
                reference_fingerprint,
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
                kind: optimizer_kind.to_string(),
                // The run's table, not the optimizer's alone: a master copy
                // adds a slot to it, and a payload written with one is not
                // readable without one.
                layout_version: optimizer_kind.layout_version_with_master(plan.keeps_master_copy()),
                hyperparameters: hyperparameters.lines(),
                learning_rate: optimizer_state.learning_rate,
                weight_decay: optimizer_state.weight_decay,
                max_grad_norm: optimizer_state.max_grad_norm,
                iter: optimizer_state.iter,
                graph_ready: optimizer_state.graph_ready,
                slots: slots.clone(),
                assignment: assignment_of(&plan),
                state_bytes,
            },
            rng: checkpoint::Rng {
                version: checkpoint::FORMAT_VERSION,
                runtime_mt19937,
                seeds: metadata.seeds.clone(),
            },
            dataset: metadata.dataset.clone(),
            artifacts,
        };

        record.write(&state_dir, |paths| {
            if let Some(path) = &paths.adapter {
                self.save_lora(path)?;
            }
            if let Some(path) = &paths.trainable {
                self.save_trainable(path)?;
            }
            if let Some(path) = &paths.optimizer_state {
                let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                self.write_state_payload(&slots, &mut file)?;
                std::io::Write::flush(&mut file)?;
            }
            Ok(())
        })?;

        Ok(())
    }

    /// Restores a checkpoint: loads whatever the run produced, validates it
    /// against the run configuration, then applies the optimizer, scheduler,
    /// and RNG state. Nothing is written into the trainer before validation
    /// succeeds.
    pub fn load_checkpoint(
        &mut self,
        state_dir: impl AsRef<Path>,
        expected: &checkpoint::Compatibility,
    ) -> Result<ResumeInfo> {
        let state_dir = checkpoint::state_dir_for(state_dir.as_ref());
        let record = checkpoint::Checkpoint::read(&state_dir)?;
        record.check_compatible(expected)?;
        let optimizer = OptimizerKind::parse(&record.optimizer.kind)?.as_ffi()?;

        let adapter = checkpoint::adapter_for(&state_dir, &record.manifest);
        if let Some(adapter) = &adapter {
            if !adapter.is_file() {
                return Err(Error::runtime(format!(
                    "checkpoint adapter {} is missing",
                    adapter.display()
                )));
            }
            self.load_lora(adapter)?;
        }

        let trainable = checkpoint::trainable_for(&state_dir, &record.manifest);
        if let Some(bundle) = &trainable {
            if !bundle.is_file() {
                return Err(Error::runtime(format!(
                    "checkpoint trainable bundle {} is missing",
                    bundle.display()
                )));
            }
            // The manifest records what the bundle was when it was published.
            // A file that no longer matches is a corrupted or swapped payload,
            // and restoring it would resume from weights nobody trained.
            if let Some(declared) = &record.manifest.trainable {
                let found = checkpoint::fingerprint_file(bundle)?;
                if found != declared.fingerprint {
                    return Err(Error::checkpoint(format!(
                        "checkpoint trainable bundle {} does not match the manifest",
                        bundle.display()
                    )));
                }
            }
            self.load_trainable(bundle)?;
        }

        let mut restored_slots = 0;
        if record.optimizer.graph_ready {
            // Per-parameter state only exists once the optimizer graph is
            // built, which is why the restore is deferred to here rather than
            // done at read. An optimizer with no slots still needs the graph:
            // that is what its iteration counter and RNG state belong to.
            self.prepare_optimizer()?;
            self.restore_state_payload(&state_dir, &record.optimizer)?;
            restored_slots = record.optimizer.slots.len();
            if let Some(rng) = &record.rng.runtime_mt19937 {
                self.set_rng_state(rng)?;
            }
        }
        let state = ffi::RetroOptimizerState {
            iter: record.optimizer.iter,
            has_persistent_state: !record.optimizer.slots.is_empty(),
            graph_ready: record.optimizer.graph_ready,
            optimizer,
            learning_rate: record.optimizer.learning_rate,
            weight_decay: record.optimizer.weight_decay,
            max_grad_norm: record.optimizer.max_grad_norm,
            scheduler_step: record.scheduler.step,
            scheduler_total_steps: record.scheduler.total_steps,
            last_learning_rate: record.scheduler.last_learning_rate,
            // Reported, never restored: the coefficients belong to the run
            // configuration, and the compatibility check above refuses a
            // record that disagrees with it.
            ..ffi::RetroOptimizerState::default()
        };
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_restore_optimizer_state(self.raw.as_ptr(), &state)
        })?;

        Ok(ResumeInfo {
            adapter,
            trainable,
            progress: record.progress,
            dataset: record.dataset,
            seeds: record.rng.seeds,
            artifacts: record.artifacts,
            had_optimizer_graph: record.optimizer.graph_ready,
            restored_optimizer_slots: restored_slots,
        })
    }
}

fn ffi_scope(scope: checkpoint::SlotScope) -> i32 {
    match scope {
        checkpoint::SlotScope::Parameter => ffi::SLOT_SCOPE_PARAMETER,
        checkpoint::SlotScope::Shared => ffi::SLOT_SCOPE_SHARED,
    }
}

/// Reads a NUL-terminated fixed-size contract field.
fn fixed_string(field: &[std::os::raw::c_char]) -> Result<String> {
    let bytes: Vec<u8> = field
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| *byte as u8)
        .collect();
    String::from_utf8(bytes)
        .map_err(|error| Error::runtime(format!("the runtime returned a non-UTF-8 name: {error}")))
}

/// Which family a marked tensor belongs to, from its name.
///
/// The runtime enumerates adapter factors and base tensors through the same
/// call because they are the same kind of thing to the update step; the suffix
/// is what the LoRA loader itself appends, so this reads back its own fact.
fn role_of(name: &str) -> TensorRole {
    if name.ends_with(".lora_a") {
        TensorRole::LoraA
    } else if name.ends_with(".lora_b") {
        TensorRole::LoraB
    } else {
        TensorRole::Base
    }
}

/// Fingerprint of a resolved set's canonical manifest.
///
/// Over [`TrainableSet::manifest_lines`] rather than over the entry order,
/// because that order carries the optimizer's update sequence and identity must
/// not depend on it.
fn set_signature(set: &TrainableSet) -> String {
    if set.base_entries().next().is_none() {
        return String::new();
    }
    checkpoint::fingerprint(set.manifest_lines().join("\n").as_bytes())
}

/// One row per marked parameter, naming the optimizer that updates it.
///
/// Written even when every row is the same name: the table is what a mixed run
/// compares on resume, and one that listed only the parameters an optimizer
/// accepted could not express a fallback.
fn assignment_of(plan: &retrograd_core::OptimizerPlan) -> Vec<checkpoint::ParameterAssignment> {
    plan.parameters
        .iter()
        .map(|parameter| checkpoint::ParameterAssignment {
            parameter: parameter.name.clone(),
            // `None` is refused before this is reached, so the fallback is
            // unreachable.
            optimizer: parameter
                .optimizer
                .unwrap_or(OptimizerKind::AdamW)
                .to_string(),
            layout_version: parameter
                .layout_version()
                .unwrap_or_else(|| OptimizerKind::AdamW.layout_version()),
        })
        .collect()
}

/// The live slot table of one scope, as `(owner, slot, bytes)`.
fn live_rows(
    slots: &[checkpoint::StateSlot],
    scope: checkpoint::SlotScope,
) -> Vec<(String, String, u64)> {
    slots
        .iter()
        .filter(|slot| slot.scope == scope)
        .map(|slot| (slot.owner.clone(), slot.slot.clone(), slot.n_bytes))
        .collect()
}
