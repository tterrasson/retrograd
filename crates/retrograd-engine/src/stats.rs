//! Counters and measurements the runtime reports about its own work.

/// Monotonic optimizer timings supplied by the runtime. Take two snapshots
/// and call [`OptimizerTiming::delta_since`] to attribute one logical update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OptimizerTiming {
    pub graph_build_seconds: f64,
    pub allocation_seconds: f64,
    pub execution_seconds: f64,
}

impl OptimizerTiming {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            graph_build_seconds: (self.graph_build_seconds - before.graph_build_seconds).max(0.0),
            allocation_seconds: (self.allocation_seconds - before.allocation_seconds).max(0.0),
            execution_seconds: (self.execution_seconds - before.execution_seconds).max(0.0),
        }
    }

    pub fn total_seconds(self) -> f64 {
        self.graph_build_seconds + self.allocation_seconds + self.execution_seconds
    }
}

/// Monotonic counters of the shared-prefix behavior scorer. Like
/// [`OptimizerTiming`], take two snapshots and difference them to attribute one
/// update. They answer two questions a wall clock cannot: whether the shared
/// prefix had to be re-decoded per branch (`prefix_decodes` above `calls`), and
/// whether the target log-probabilities were gathered on the device or reduced
/// on the host (`device_logprob_positions` against `scored_positions`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScoringStats {
    pub calls: u64,
    pub prefix_decodes: u64,
    pub prefix_reprefills: u64,
    pub branch_evictions_refused: u64,
    pub scored_positions: u64,
    pub device_logprob_positions: u64,
}

impl ScoringStats {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(before.calls),
            prefix_decodes: self.prefix_decodes.saturating_sub(before.prefix_decodes),
            prefix_reprefills: self
                .prefix_reprefills
                .saturating_sub(before.prefix_reprefills),
            branch_evictions_refused: self
                .branch_evictions_refused
                .saturating_sub(before.branch_evictions_refused),
            scored_positions: self
                .scored_positions
                .saturating_sub(before.scored_positions),
            device_logprob_positions: self
                .device_logprob_positions
                .saturating_sub(before.device_logprob_positions),
        }
    }
}

/// Monotonic counters of the generation context's prefix reuse. Same contract as
/// [`ScoringStats`]: difference two snapshots to attribute one update.
///
/// The pair that matters is `prompt_tokens` against `prefilled_tokens` - what a
/// rollout asked to be resident against what had to be decoded to get there.
/// They are equal on a single-turn run, where every prompt is new. On a
/// multi-turn one their ratio is the cost model of the whole rollout: without
/// reuse a trajectory re-decodes its entire prefix every turn, so its prefill
/// grows with the *square* of its turn count, and the same trajectory with reuse
/// pays the length of one turn each time.
///
/// `evictions` separates the two ways the ratio can be bad. Zero, with few
/// `hits`, means the prompts are genuinely new. Non-zero means the context has
/// fewer sequences than the rollout has live trajectories - a cache that works
/// and does not fit, which `generation_concurrency` fixes and nothing else does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationStats {
    pub calls: u64,
    pub sequences: u64,
    pub prompt_tokens: u64,
    pub prefilled_tokens: u64,
    pub reused_tokens: u64,
    pub hits: u64,
    pub evictions: u64,
}

/// Wall-clock accounting of the GPU duty-cycle limiter, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DutyCycleStats {
    /// What the run asked for. `1.0` means no limit was requested.
    pub requested_fraction: f32,
    /// Whether the limiter engaged. `requested_fraction < 1.0` with `active`
    /// false has exactly one cause: the active backend is the CPU.
    pub active: bool,
    /// Synchronized GPU work charged to the limiter.
    pub compute_seconds: f64,
    /// Time it deliberately slept.
    pub idle_seconds: f64,
    /// Wall time since the limiter was enabled.
    pub wall_seconds: f64,
}

impl DutyCycleStats {
    /// Compute over the accounted windows alone. Every accounted window is by
    /// construction followed by its own repayment, so this converges to the
    /// requested fraction whenever the limiter is working, and mostly echoes
    /// the setting back. `None` before the first accounted window - a zero
    /// would read as "throttled to nothing".
    pub fn observed(&self) -> Option<f64> {
        let total = self.compute_seconds + self.idle_seconds;
        (total > 0.0).then(|| self.compute_seconds / total)
    }

    /// Compute over the whole trainer wall clock. This is the number an
    /// operator is actually asking for: the gap between it and
    /// [`Self::observed`] is the unaccounted host time - data loading, judging,
    /// tokenization, checkpoint I/O. A run reporting `observed = 0.50` and
    /// `wall_share = 0.20` is not misconfigured, it is CPU-bound, and no duty
    /// cycle will free the compute its operator hoped to release.
    pub fn wall_share(&self) -> Option<f64> {
        (self.compute_seconds > 0.0 && self.wall_seconds > 0.0)
            .then(|| self.compute_seconds / self.wall_seconds)
    }
}

impl GenerationStats {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(before.calls),
            sequences: self.sequences.saturating_sub(before.sequences),
            prompt_tokens: self.prompt_tokens.saturating_sub(before.prompt_tokens),
            prefilled_tokens: self
                .prefilled_tokens
                .saturating_sub(before.prefilled_tokens),
            reused_tokens: self.reused_tokens.saturating_sub(before.reused_tokens),
            hits: self.hits.saturating_sub(before.hits),
            evictions: self.evictions.saturating_sub(before.evictions),
        }
    }
}

/// Device memory the runtime measured while running optimizer steps, in bytes.
///
/// This is the counterpart to the byte breakdown in the backend report, and the
/// two are not interchangeable: the breakdown sums `ggml_backend_buffer`s, which
/// excludes the backends' own scratch (CUDA pool, Vulkan `prealloc_*`) and the
/// graph allocator's transient reserve. Every VRAM lever in the project lands in
/// that gap. Sampling from here between steps would not help either - the scratch
/// is handed back when the step returns - so the runtime samples inside the step
/// and this only reads back what it recorded. Scope: the high-water *retained* by
/// the instrumented stores (the CUDA pools and the Vulkan `prealloc_*` buffers keep
/// theirs). An allocation taken and released entirely within one evaluation,
/// without going through those pools, is not covered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptimizerMemory {
    /// Most recent device-wide usage. Includes other processes: meaningful as a
    /// delta between comparable runs, not as "this run's VRAM".
    pub device_used_bytes: u64,
    pub device_total_bytes: u64,
    /// High-water of `device_used_bytes` across every sample.
    pub device_peak_used_bytes: u64,
    /// Backend-owned scratch, attributable to this process.
    pub scratch_bytes: u64,
    pub scratch_peak_bytes: u64,
    /// Zero when no non-CPU device was active, which is what distinguishes an
    /// unavailable measurement from a measured zero.
    pub n_samples: u64,
}

impl OptimizerMemory {
    /// Whether the runtime actually took a measurement. CPU-only runs and
    /// backends that do not report a budget answer `false`, and every byte field
    /// is then zero.
    pub fn is_measured(self) -> bool {
        self.n_samples > 0
    }
}
