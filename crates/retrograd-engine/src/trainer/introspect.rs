use super::*;
use crate::probe::fixed_name;

impl Trainer {
    /// Snapshots the runtime's monotonic optimizer counters. These counters
    /// deliberately survive optimizer calls so GRPO can attribute a whole
    /// update even when it contains several packed chunks.
    pub fn optimizer_timing(&self) -> Result<OptimizerTiming> {
        let mut timing = ffi::RetroOptimizerTiming::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_optimizer_timing(self.raw.as_ptr(), &mut timing) })?;
        Ok(OptimizerTiming {
            graph_build_seconds: timing.graph_build_seconds,
            allocation_seconds: timing.allocation_seconds,
            execution_seconds: timing.execution_seconds,
        })
    }

    /// Snapshots the runtime's monotonic behavior-scoring counters.
    pub fn scoring_stats(&self) -> Result<ScoringStats> {
        let mut stats = ffi::RetroScoringStats::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_scoring_stats(self.raw.as_ptr(), &mut stats) })?;
        Ok(ScoringStats {
            calls: stats.calls,
            prefix_decodes: stats.prefix_decodes,
            prefix_reprefills: stats.prefix_reprefills,
            branch_evictions_refused: stats.branch_evictions_refused,
            scored_positions: stats.scored_positions,
            device_logprob_positions: stats.device_logprob_positions,
        })
    }

    /// Snapshots the runtime's monotonic generation prefix-reuse counters.
    pub fn generation_stats(&self) -> Result<GenerationStats> {
        let mut stats = ffi::RetroGenerationStats::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_generation_stats(self.raw.as_ptr(), &mut stats) })?;
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

    /// Snapshots the GPU duty-cycle limiter's wall-clock accounting. Never
    /// fails on a trainer that has throttled nothing: the seconds are zero and
    /// `active` is false.
    pub fn duty_cycle_stats(&self) -> Result<DutyCycleStats> {
        let mut stats = ffi::RetroDutyCycleStats::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_duty_cycle_stats(self.raw.as_ptr(), &mut stats) })?;
        Ok(DutyCycleStats {
            requested_fraction: stats.requested_fraction,
            active: stats.active,
            compute_seconds: stats.compute_seconds,
            idle_seconds: stats.idle_seconds,
            wall_seconds: stats.wall_seconds,
        })
    }

    /// Reads the device-memory high-water the runtime measured inside its
    /// optimizer steps. Unlike [`Trainer::optimizer_timing`] the peak fields are
    /// running maxima rather than accumulators, so a delta between two snapshots
    /// is not meaningful; read it after the work you want to characterize.
    pub fn optimizer_memory(&self) -> Result<OptimizerMemory> {
        let mut memory = ffi::RetroOptimizerMemory::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_optimizer_memory(self.raw.as_ptr(), &mut memory) })?;
        Ok(OptimizerMemory {
            device_used_bytes: memory.device_used_bytes,
            device_total_bytes: memory.device_total_bytes,
            device_peak_used_bytes: memory.device_peak_used_bytes,
            scratch_bytes: memory.scratch_bytes,
            scratch_peak_bytes: memory.scratch_peak_bytes,
            n_samples: memory.n_samples,
        })
    }

    /// Reads the structured memory breakdown of this trainer: the same figures
    /// the textual backend report carries, as data.
    ///
    /// Prefer this over parsing [`Trainer::backend_report`]. The text is a
    /// human-facing diagnostic whose layout is free to change; this is the
    /// contract. Compute and measured fields fill in as the run progresses, so
    /// read it after the preflight to see the activation buffers.
    pub fn memory_report(&self) -> Result<MemoryReport> {
        let mut report = ffi::RetroMemoryReport::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_memory_report(self.raw.as_ptr(), &mut report) })?;
        Ok(MemoryReport {
            model_weight_bytes: report.model_weight_bytes,
            optimizer_kv_bytes: report.optimizer_kv_bytes,
            optimizer_compute_bytes: report.optimizer_compute_bytes,
            generation_kv_bytes: report.generation_kv_bytes,
            generation_compute_bytes: report.generation_compute_bytes,
            has_generation_context: report.has_generation_context,
            lora_parameter_bytes: report.lora_parameter_bytes,
            lora_gradient_bytes: report.lora_gradient_bytes,
            adamw_momenta_bytes: report.adamw_momenta_bytes,
            lora_on_host: report.lora_on_host,
            device_bytes: report.device_bytes,
            host_bytes: report.host_bytes,
            device_total_bytes: report.device_total_bytes,
            device_used_bytes: report.device_used_bytes,
            device_peak_used_bytes: report.device_peak_used_bytes,
            backend_scratch_bytes: report.backend_scratch_bytes,
            backend_scratch_peak_bytes: report.backend_scratch_peak_bytes,
            device_memory_samples: report.device_memory_samples,
            checkpoint_count: report.checkpoint_count,
            checkpoint_retained_bytes: report.checkpoint_retained_bytes,
            checkpoint_live_peak_bytes: report.checkpoint_live_peak_bytes,
            checkpoint_live_peak_count: report.checkpoint_live_peak_count,
            checkpoint_long_lived_bytes: report.checkpoint_long_lived_bytes,
            checkpoint_long_lived_count: report.checkpoint_long_lived_count,
            checkpoint_graph_nodes: report.checkpoint_graph_nodes,
            checkpoint_max_span_nodes: report.checkpoint_max_span_nodes,
            checkpoint_total_span_nodes: report.checkpoint_total_span_nodes,
        })
    }

    read_string_method!(describe_lora, retro_trainer_describe_lora);

    read_string_method!(
        #[doc = "Reports which backend buffers the model and LoRA tensors live on, plus the resolved device. Useful for verifying GPU offload."]
        backend_report,
        retro_trainer_backend_report
    );

    read_string_method!(
        #[doc = "Describes model, target-profile and backend compatibility before training."]
        capability_report,
        retro_trainer_capability_report
    );

    /// Every tensor a LoRA adapter can attach to, sorted, with layer indices
    /// intact. Lets a caller resolve a concrete target from the model instead
    /// of assuming a layout - on a hybrid architecture (`lfm2`, `falcon-h1`)
    /// the block families interleave, so `blk.0` need not be an attention
    /// block. Available before a LoRA adapter is created.
    pub fn lora_candidate_targets(&self) -> Result<Vec<String>> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        let listing = unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_trainer_lora_candidate_targets(self.raw.as_ptr(), buffer, n_buffer, out)
            })
        }?;
        Ok(listing
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect())
    }

    read_string_method!(
        #[doc = "Builds the exact training graph for a representative micro-batch without running it and reports, per backend device, unsupported ops and ops without gradient rules. Requires a created LoRA adapter; the same check runs automatically before the first optimizer step."]
        train_preflight,
        retro_trainer_train_preflight,
        mut
    );

    /// Machine-readable counterpart of [`Trainer::train_preflight`]. The text
    /// report is still produced for diagnostics, but no planning fact is parsed
    /// from it.
    pub fn structured_preflight(&mut self, profile_fingerprint: String) -> Result<PreflightReport> {
        self.train_preflight()?;
        let mut summary = ffi::RetroPreflightSummary::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_preflight_summary(self.raw.as_ptr(), &mut summary)
        })?;
        let graph_fingerprint = fixed_name(&summary.graph_fingerprint);
        let fallback_nodes = summary.active_device_fallback_nodes;
        let mut warnings = Vec::new();
        if summary.missing_gradient_rules > 0 {
            warnings.push(PreflightWarning {
                code: "missing_gradient_rules".to_string(),
                message: format!(
                    "{} training graph operation(s) have no gradient rule",
                    summary.missing_gradient_rules
                ),
            });
        }
        if fallback_nodes > 0 {
            warnings.push(PreflightWarning {
                code: "cpu_fallback".to_string(),
                message: format!("{fallback_nodes} training graph node(s) fall back to the CPU"),
            });
        }
        let placements = if fallback_nodes == 0 {
            Vec::new()
        } else {
            vec![OpPlacement {
                ggml_op: "aggregate".to_string(),
                shape_class: "training_graph".to_string(),
                nodes: fallback_nodes,
                backend: "cpu".to_string(),
                implementation: "native".to_string(),
                rejection: Some("device_unsupported".to_string()),
                native_fallback: true,
                ..Default::default()
            }]
        };
        Ok(PreflightReport {
            schema_version: retrograd_core::PREFLIGHT_REPORT_VERSION,
            profile_fingerprint,
            graph_fingerprint,
            placements,
            kernel_summary: Vec::new(),
            memory: self.memory_report()?,
            warnings,
        })
    }

    pub fn save_lora(&mut self, adapter_path: impl AsRef<Path>) -> Result<()> {
        let adapter_path = path_to_cstring(adapter_path.as_ref())?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_save_lora(self.raw.as_ptr(), adapter_path.as_ptr())
        })
    }

    read_string_method!(
        #[doc = "Architecture and shape hyperparameters of the loaded model, recorded in a checkpoint manifest and compared on resume."]
        model_signature,
        retro_trainer_model_signature,
        mut
    );

    /// Structured model/device capabilities used by the planner. The text
    /// report remains diagnostic output and is never parsed here.
    pub fn model_capabilities(&mut self) -> Result<retrograd_core::ModelCapabilities> {
        let mut capabilities = ffi::RetroModelCapabilities::default();
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_model_capabilities(self.raw.as_ptr(), &mut capabilities)
        })?;
        Ok(retrograd_core::ModelCapabilities {
            shared_prefix_packed_training: capabilities.shared_prefix_packed_training,
            fused_sparse_cross_entropy: capabilities.fused_sparse_cross_entropy,
            differentiable_flash_attention: capabilities.differentiable_flash_attention,
            ..Default::default()
        })
    }

    pub fn supports_shared_prefix_packed_training(&mut self) -> Result<bool> {
        Ok(self.model_capabilities()?.shared_prefix_packed_training)
    }
}
