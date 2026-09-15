use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::{Error, Result};

/// Selects the execution backend for the training loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// Use a compiled GPU backend (Metal, Vulkan, or CUDA) when available,
    /// otherwise fall back to CPU.
    #[default]
    Auto,
    /// Force CPU-only execution.
    Cpu,
    /// Force GPU offload; trainer creation fails if no GPU backend is present.
    Gpu,
}

/// Storage precision requested for the differentiable training KV cache.
/// F16 is enabled only when the active device reports the differentiable Flash
/// Attention forward and backward ops for the model's head geometry; every
/// other device falls back explicitly to F32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KvDtype {
    #[default]
    F32,
    F16,
}

impl KvDtype {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
        }
    }
}

/// Storage precision for the layer outputs retained by gradient checkpointing.
///
/// A 16-bit checkpoint halves that term, and would halve the host-transfer traffic
/// if activation offloading were combined with it. It is opt-in, and not merely
/// because it breaks bit-parity: measurement on the CPU fixture shows a 16-bit
/// checkpoint perturbs the LoRA update by roughly its own relative precision
/// (~1e-3), which is about four orders of magnitude more than the F32 recompute's
/// reassociation error. Neither F16 nor BF16 was clearly better at that scale. So
/// this trades a real amount of gradient fidelity for memory - appropriate when a
/// run would not otherwise fit, not a free win.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointDtype {
    /// Keep the checkpoints as the forward built them. No casts are inserted, so
    /// the recompute stays bit-exact.
    #[default]
    F32,
    /// 10-bit mantissa, 5-bit exponent.
    F16,
    /// 8-bit mantissa with F32's exponent range, so it cannot overflow on a wide
    /// residual stream the way F16 can on a larger model than the test fixture.
    Bf16,
}

impl CheckpointDtype {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }
}

/// Storage precision for the PPO critic's host-side feature matrix.
///
/// The critic keeps one row of `hidden_dim` features per completion state for the
/// whole update - predictions, GAE and `value_epochs` passes of full-batch Adam all
/// read the same matrix - so the buffer is `total_completion_states * hidden_dim`
/// and it lives in host RAM, not on the device. A 16-bit row halves it; the fit
/// still runs in F32, one chunk converted at a time, so the arithmetic is unchanged
/// and only the stored rows are rounded.
///
/// Opt-in, and the reason is the same one [`CheckpointDtype`] gives: rounding the
/// features perturbs the value regression. That perturbation is benign in a way the
/// gradient one is not - the baseline only reduces the variance of the policy
/// gradient and never biases the update - but "benign" is not "absent", and the
/// current default stays until a measurement asks for the memory back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureDtype {
    /// Store the hidden states exactly as the runtime produced them.
    #[default]
    F32,
    /// 10-bit mantissa, 5-bit exponent. The most precise of the two 16-bit forms
    /// and the one to prefer while the residual stream stays inside its range.
    F16,
    /// 8-bit mantissa with F32's exponent range. Coarser, but a late-layer residual
    /// stream on a large model can exceed F16's 65504 and this cannot.
    Bf16,
}

impl FeatureDtype {
    /// Bytes one stored feature occupies.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

impl Device {
    pub fn as_ffi(self) -> i32 {
        match self {
            Device::Auto => 0,
            Device::Cpu => 1,
            Device::Gpu => 2,
        }
    }
}

impl std::str::FromStr for Device {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            _ => Err(Error::invalid(format!(
                "unknown device '{value}'; use auto, cpu, or gpu"
            ))),
        }
    }
}

/// Learning-rate schedule evaluated once for every AdamW update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrScheduler {
    Constant,
    Linear,
    Cosine,
}

impl LrScheduler {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::Constant => 0,
            Self::Linear => 1,
            Self::Cosine => 2,
        }
    }

    /// The configuration-file spelling of this scheduler.
    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Cosine => "cosine",
        }
    }
}

/// Token chunk the fused cross-entropy is bounded to when nothing else says.
///
/// Large enough that a short run never splits (the chunk is a ceiling, not a
/// quantum), small enough that an 8k-token step stops carrying a tile
/// intermediate proportional to the whole context.
pub const DEFAULT_CE_SEQ_CHUNK: u32 = 512;

/// Checkpoint stride for a model whose depth is not known yet.
///
/// `checkpoint_stride_for(16)`. Callers that do know the depth - the resolver
/// does - should use that function instead; this is the value the struct
/// default has to name before any model is loaded.
pub const DEFAULT_CHECKPOINT_STRIDE: u32 = 4;

/// Layers per activation checkpoint that minimise the retained term.
///
/// With one checkpoint every `s` layers, the backward holds `n_layer / s` layer
/// boundaries plus the working set of the one segment it is recomputing, so the
/// peak goes as `n_layer / s + s`. That is smallest at `s = sqrt(n_layer)`, and
/// the extra compute is one forward pass either side of it - a stride is a
/// memory choice, not a compute one: values near one maximize the boundary term
/// for the same amount of recomputation.
pub fn checkpoint_stride_for(n_layer: u32) -> u32 {
    (n_layer as f64).sqrt().round().max(1.0) as u32
}

/// Physical number of completions that may share one prompt copy in a
/// differentiable optimizer graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedPrefixFanout {
    #[default]
    Auto,
    Off,
    Max,
    Exact(u32),
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Maximum number of independent sequences packed by the optimizer.
    /// GRPO sets this to its group size.
    pub n_seq_max: u32,
    pub shared_prefix_fanout: SharedPrefixFanout,
    /// Maximum number of rollout sequences decoded concurrently. This is
    /// independent from `n_seq_max`: GRPO groups may be sampled over several
    /// waves without changing their statistical meaning. Zero preserves the
    /// legacy programmatic API by inheriting `n_seq_max`; TOML configs always
    /// resolve this to a positive value.
    pub generation_concurrency: u32,
    /// Fast sampling context: builds the dedicated generation context with an
    /// F16 KV cache and flash-attention instead of the optimizer context's exact
    /// F32 / no-FA settings. Halves the KV footprint and speeds up decoding;
    /// the sampling distribution differs from the trained policy by ulp-level
    /// rounding. Enabled by default and explicitly disableable for bit-exact
    /// sampling.
    pub fast_generation_context: bool,
    /// KV-cache storage requested for the differentiable optimizer context.
    ///
    /// F16 is the default: it halves the term that grows with the context, and
    /// the runtime never applies it blind - `retro_backend` probes the device
    /// for differentiable Flash Attention at this model's head geometry and
    /// rebuilds the context on F32 when the probe declines, reporting which of
    /// the two it got as `training_kv_f16`. Pin `f32` when the KV cache must be
    /// bit-identical to the forward the estimate assumed.
    pub kv_dtype: KvDtype,
    /// CPU worker threads. Zero automatically selects performance cores when
    /// available. `RETRO_THREADS` takes precedence at runtime.
    pub threads: u32,
    pub epochs: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    /// Maximum global L2 norm of all trainable gradients before an optimizer
    /// step. Gradients above this norm are scaled together, preserving their
    /// relative direction.
    pub max_grad_norm: f32,
    pub lr_scheduler: LrScheduler,
    pub warmup_steps: u64,
    pub verbose: bool,
    pub device: Device,
    /// Fused/chunked vocabulary cross-entropy for the packed GRPO optimizer
    /// step. The loss and its gradient are computed by streaming the vocabulary
    /// in tiles, so the full `[n_vocab, n_tokens]` logits are never
    /// materialized.
    ///
    /// On by default. It is not a memory-for-speed trade: never building the
    /// `[n_vocab, n_tokens]` tensor removes the write *and* the read of the
    /// largest tensor in the step, which is bandwidth the tile recompute does
    /// not spend back. Every backend the project ships carries both fused nodes
    /// (`FUSED_SPARSE_CE` and `_BACK`), and the runtime probes them for the real
    /// head geometry before the run: `cap_fused_sparse_ce` in the report says
    /// whether this device took the fused path, and `require_gpu_resident`
    /// turns a decline into a failure instead of a quiet CPU tail. Only the
    /// packed path is affected.
    pub chunked_cross_entropy: bool,
    /// Number of vocabulary tiles `C` for the fused cross-entropy. Higher `C`
    /// lowers the peak logits footprint (~`n_vocab / C`) at the cost of more
    /// recomputation. Ignored unless `chunked_cross_entropy` is set.
    pub chunked_ce_tiles: u32,
    /// Flattened `(batch × seq)` token chunk size for the fused cross-entropy.
    /// `0` processes all tokens at once; `> 0` bounds the tiled logits
    /// intermediate to this many tokens, capping the peak footprint
    /// independently of the sequence length. Ignored unless
    /// `chunked_cross_entropy` is set.
    ///
    /// Defaults to [`DEFAULT_CE_SEQ_CHUNK`] rather than `0`: vocabulary tiling
    /// alone leaves the tile intermediate growing as `tile × n_tokens`, so a
    /// long context walks the peak back up the axis the tiles do not cover.
    /// A chunk at or above the token count is one chunk, i.e. the old
    /// behaviour, so short runs pay nothing for the default.
    ///
    /// When a chunk bounds the staging buffer, the backward pass writes `grad_h`
    /// over the hidden states instead of into a second `[n_embd, n_tokens]`
    /// buffer. The engine derives this behavior from the chunk size.
    pub chunked_ce_seq_chunk: u32,
    /// Recompute transformer-layer activations during the packed PPO/GRPO
    /// backward pass, retaining only selected layer outputs as checkpoints.
    /// This trades additional forward computation for a lower activation peak.
    pub gradient_checkpointing: bool,
    /// Keep one activation checkpoint every N transformer layers. Ignored
    /// unless `gradient_checkpointing` is enabled.
    ///
    /// Not `1`. A stride of one stores *every* layer boundary - the largest
    /// retained term checkpointing can produce - and still recomputes each
    /// layer's internals, so it buys the least memory for the same extra
    /// forward. The peak is `boundaries + one segment`, i.e. proportional to
    /// `n_layer / stride + stride`, which bottoms out at `sqrt(n_layer)`;
    /// [`checkpoint_stride_for`] is that rule, and the resolver narrows it to
    /// the smallest stride that actually clears the budget.
    pub checkpoint_every_n_layers: u32,
    /// Precision the retained checkpoints are held in. `F16` halves that term at
    /// the cost of recompute bit-parity and ~1e-3 of gradient fidelity, so it
    /// stays opt-in. Rejected at load time unless `gradient_checkpointing` is
    /// enabled - it has no effect without it, and silently ignoring a request
    /// for lower precision reads as if it had been honoured.
    pub checkpoint_dtype: CheckpointDtype,
    /// Fail the training preflight instead of letting the scheduler send a
    /// training-graph op back to the CPU. Off by default: a fallback is correct,
    /// it only costs a scheduler split and a device↔host round trip per node.
    /// Turn it on when that cost is what is being measured or guarded against,
    /// e.g. `chunked_cross_entropy` on Metal, where both fused-CE nodes fall
    /// back silently.
    pub require_gpu_resident: bool,
    /// Batch/ubatch geometry of the dedicated generation context. Sampling has
    /// no backward graph, so it has no reason to inherit `n_batch`/`n_ubatch`,
    /// which size the optimizer's activations: a prompt prefilled in 16-token
    /// chunks pays one full sweep of the quantized weights per chunk. Zero
    /// derives the largest value that keeps the output-logits buffer within a
    /// 64 MiB budget, capped by `n_ctx` and 512, then raises it to
    /// `generation_concurrency` so a decode wave always fits one launch.
    /// Generation compute memory grows with this value. See
    /// `docs/engineering/optims/SAMPLING.md`.
    pub generation_batch: u32,
    /// Permute the SFT training rows at the start of every epoch. On by
    /// default: an SFT file is almost always ordered - by source, by length, by
    /// whatever the generator emitted last - and replaying that same order
    /// every epoch both correlates the rows inside a micro-batch and makes the
    /// per-epoch metrics an artefact of the file rather than of the model.
    /// Only the training part is permuted; the evaluation rows of a split
    /// dataset keep their order, so the eval loss stays comparable.
    ///
    /// Turn it off (`sft.shuffle = false`) to reproduce an unshuffled run, or when
    /// the row order is itself the curriculum.
    pub shuffle_dataset: bool,
    /// Seeds that permutation; the CLI fills it from `lora.seed`. Epoch `N`'s
    /// permutation is a pure function of `(shuffle_seed, N)`, so a resumed run
    /// draws exactly what an uninterrupted one would have drawn - the shuffle
    /// does not depend on the RNG state a checkpoint carries.
    pub shuffle_seed: u64,
    /// Upper bound on the fraction of wall time this trainer spends with GPU
    /// work submitted and in flight, so another workload can use the device in
    /// between. `None` - the default, and what an omitted or explicit `1.0`
    /// normalizes to - is the unthrottled path: no clock read, no sleep and, in
    /// particular, no added decode synchronization.
    ///
    /// It releases *compute* time, not device memory: weights, KV caches,
    /// retained activations and optimizer state stay allocated while the
    /// trainer sleeps. A neighbour that needs VRAM still needs a smaller
    /// `n_ubatch` or a lower generation concurrency.
    ///
    /// Deliberately absent from the trajectory signature: sleeping cannot alter
    /// the training problem or the serialized optimizer state, and a checkpoint
    /// may be resumed under a different duty cycle. It is still part of resolved
    /// configuration, so an observed throughput remains explainable.
    ///
    /// Enabling it costs the overlap between host-side sampling and the previous
    /// decode, once, before any sleep - so `0.99` is not approximately `1.0`.
    /// It is worth its overhead at `0.75` and below.
    pub max_gpu_duty_cycle: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_ctx: 128,
            n_batch: 128,
            n_ubatch: 32,
            n_seq_max: 1,
            shared_prefix_fanout: SharedPrefixFanout::Auto,
            generation_concurrency: 0,
            fast_generation_context: true,
            kv_dtype: KvDtype::F16,
            threads: 0,
            epochs: 1,
            learning_rate: 1.0e-4,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            lr_scheduler: LrScheduler::Constant,
            warmup_steps: 0,
            verbose: false,
            device: Device::Auto,
            chunked_cross_entropy: true,
            chunked_ce_tiles: 8,
            chunked_ce_seq_chunk: DEFAULT_CE_SEQ_CHUNK,
            gradient_checkpointing: false,
            checkpoint_every_n_layers: DEFAULT_CHECKPOINT_STRIDE,
            checkpoint_dtype: CheckpointDtype::F32,
            require_gpu_resident: false,
            generation_batch: 0,
            shuffle_dataset: true,
            // Matches the `lora.seed` default, so a programmatic run that never
            // names a seed shuffles the same way a configured one does.
            shuffle_seed: 42,
            max_gpu_duty_cycle: None,
        }
    }
}

impl TrainConfig {
    /// Micro-batches accumulated before one optimizer step - the configuration
    /// spells this as `training.gradient_accumulation`, the runtime stores the
    /// product it implies. Zero-safe so it can be used inside diagnostics.
    pub fn gradient_accumulation(&self) -> u32 {
        self.n_batch / self.n_ubatch.max(1)
    }

    /// Geometry rules every training path shares, whatever frontend built the
    /// configuration.
    ///
    /// `n_ubatch` is the physical forward/backward width (`training.micro_batch`
    /// in a configuration file, TRL's `per_device_train_batch_size`), `n_batch`
    /// the token window one optimizer step trains - `micro_batch *
    /// gradient_accumulation` - and `n_ctx` the trained window. Hence the two
    /// divisibility rules: a step is a whole number of micro-batches, and a row
    /// is a whole number of steps.
    pub fn validate_geometry(&self) -> Result<()> {
        if self.n_ctx == 0 || self.n_batch == 0 || self.n_ubatch == 0 {
            return Err(Error::invalid(
                "training.ctx, training.micro_batch and training.gradient_accumulation must all \
                 be greater than zero",
            ));
        }
        if !self.n_batch.is_multiple_of(self.n_ubatch) {
            return Err(Error::invalid(format!(
                "one optimizer step spans {} tokens, which is not a whole number of \
                 training.micro_batch ({}) micro-batches",
                self.n_batch, self.n_ubatch,
            )));
        }
        if !self.n_ctx.is_multiple_of(self.n_batch) {
            return Err(Error::invalid(format!(
                "training.ctx ({}) must be a multiple of the optimizer window \
                 training.micro_batch * training.gradient_accumulation ({} * {} = {})",
                self.n_ctx,
                self.n_ubatch,
                self.gradient_accumulation(),
                self.n_batch,
            )));
        }
        Ok(())
    }

    /// The additional rule a rollout objective (PPO, GRPO, agentic GRPO)
    /// imposes on top of [`Self::validate_geometry`]. Every frontend that
    /// builds a rollout run must go through this - the TOML loader, the
    /// planner, and the agentic binary - because it is the objective's
    /// assumption, not one loader's policy.
    ///
    /// A packed row is one rollout and the runtime takes one optimizer step per
    /// `n_batch` tokens of it, so an optimizer window below `n_ctx` turns a
    /// single completion into several AdamW steps - most of them over prompt
    /// positions that carry no label at all - and the policy leaves the trust
    /// region the clipped surrogate assumes inside the first update. That window
    /// is not the memory lever: it only sizes llama.cpp's host-side output
    /// buffer, while `n_ubatch` is what bounds activation memory.
    pub fn validate_rollout_geometry(&self) -> Result<()> {
        self.validate_geometry()?;
        if self.n_batch != self.n_ctx {
            return Err(Error::invalid(format!(
                "a rollout algorithm takes exactly one optimizer step per rollout, so the step \
                 must span the whole context: training.micro_batch * \
                 training.gradient_accumulation is {} * {} = {} but training.ctx is {}, which \
                 silently takes {} steps per completion. Omit training.gradient_accumulation - it \
                 is pinned to ctx / micro_batch = {} here - and lower training.micro_batch to \
                 bound activation memory instead.",
                self.n_ubatch,
                self.gradient_accumulation(),
                self.n_batch,
                self.n_ctx,
                self.n_ctx.div_ceil(self.n_batch),
                self.n_ctx.div_ceil(self.n_ubatch),
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetSet {
    /// Resolve architecture-specific default targets in the C++ runtime.
    Auto,
    /// Attention query and value projections only (`attn_q`, `attn_v`).
    QV,
    /// Explicit glob patterns over tensor names.
    Patterns(Vec<String>),
}

/// Storage precision for trainable LoRA A/B matrices. Gradients and optimizer
/// state remain F32 for both variants.
///
/// Deserializes from `"f32"` / `"f16"` so every TOML frontend spells it the
/// same way and none of them can silently drop the field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LoraDtype {
    F32,
    /// The default: gradients and optimizer state stay F32, so halving the A/B
    /// storage costs nothing that shows up in training quality.
    #[default]
    F16,
}

impl LoraDtype {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
        }
    }
}

impl TargetSet {
    pub fn patterns(&self) -> Vec<String> {
        match self {
            TargetSet::Auto => Vec::new(),
            TargetSet::QV => vec![
                "blk.*.attn_q.weight".to_string(),
                "blk.*.attn_v.weight".to_string(),
            ],
            TargetSet::Patterns(patterns) => patterns.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraConfig {
    pub rank: u32,
    pub alpha: f32,
    pub dropout: f32,
    pub seed: u32,
    pub targets: TargetSet,
    pub dtype: LoraDtype,
}

impl LoraConfig {
    pub fn auto(rank: u32, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            seed: 42,
            targets: TargetSet::Auto,
            dtype: LoraDtype::default(),
        }
    }

    pub fn qv(rank: u32, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            seed: 42,
            targets: TargetSet::QV,
            dtype: LoraDtype::default(),
        }
    }
}

/// Geometry of a GGUF model, read without building a training context.
///
/// This is the resolver's first input: a memory estimate needs the layer count,
/// the KV width and the vocabulary size *before* anything is allocated, and
/// `Trainer` only exposed `hidden_size` / `vocab_size` / `context_size`, which
/// is not enough to size a KV cache or an activation buffer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelInfo {
    pub n_layer: u32,
    pub n_embd: u32,
    /// Maximum feed-forward width across transformer layers. Zero means an old
    /// producer did not publish it; consumers then use a conservative fallback.
    pub n_ff: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    /// Per-layer KV width in elements, maximised across layers. Sliding-window
    /// and hybrid models vary it per layer; a maximum over-estimates, which is
    /// the direction a memory budget must err in.
    pub n_embd_k_gqa: u32,
    pub n_embd_v_gqa: u32,
    /// Recurrent/state widths, non-zero only for recurrent or hybrid models.
    /// Unlike the KV cache, this memory does not scale with `n_ctx`.
    pub n_embd_r: u32,
    pub n_embd_s: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_params: u64,
    /// Bytes the weights occupy once loaded, which is the figure a budget needs
    /// - not the file length, which `file_size_bytes` carries separately.
    pub model_size_bytes: u64,
    pub file_size_bytes: u64,
    /// Bytes held by tensors of [`Self::dominant_weight_type`].
    pub dominant_weight_bytes: u64,
    /// Whether the output projection reuses the token-embedding tensor. When it
    /// does, the vocabulary head costs no weights of its own.
    pub tied_embeddings: bool,
    pub is_recurrent: bool,
    pub has_encoder: bool,
    pub architecture: String,
    /// ggml name of the type accounting for the most weight bytes (`"Q4_K"`,
    /// `"F16"`, …). The dequantization scratch cost depends on it.
    pub dominant_weight_type: String,
}

impl ModelInfo {
    /// KV-cache bytes for `n_ctx` tokens at `element_bytes` per element,
    /// summed over layers and over both K and V.
    ///
    /// Kept here rather than in the resolver because it is the one formula that
    /// has to agree with what llama.cpp actually allocates, and it belongs next
    /// to the widths it multiplies.
    pub fn kv_cache_bytes(&self, n_ctx: u64, element_bytes: u64) -> u64 {
        // Saturating for the reason `retrograd_plan::cost` states at length: a
        // cache figure that wrapped would be *smaller* than the truth, and the
        // budget that reads it would accept a run it exists to refuse.
        let per_token = (self.n_embd_k_gqa as u64).saturating_add(self.n_embd_v_gqa as u64);
        per_token
            .saturating_mul(self.n_layer as u64)
            .saturating_mul(n_ctx)
            .saturating_mul(element_bytes)
    }
}

/// Where a trainer's memory actually went, per component.
///
/// The structured form of the byte fields in the textual backend report. The
/// runtime renders both from one computation, so they cannot disagree.
///
/// `*_compute_bytes` only becomes non-zero once a graph has been reserved (after
/// the preflight); before that it is a genuine zero, not a missing measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryReport {
    pub model_weight_bytes: u64,
    pub optimizer_kv_bytes: u64,
    pub optimizer_compute_bytes: u64,
    pub generation_kv_bytes: u64,
    pub generation_compute_bytes: u64,
    /// False when generation shares the optimizer context, in which case both
    /// `generation_*` fields are zero rather than unknown.
    pub has_generation_context: bool,
    pub lora_parameter_bytes: u64,
    pub lora_gradient_bytes: u64,
    pub adamw_momenta_bytes: u64,
    /// Which budget the LoRA parameters, gradients and AdamW moments draw on.
    pub lora_on_host: bool,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub device_total_bytes: u64,
    pub device_used_bytes: u64,
    pub device_peak_used_bytes: u64,
    pub backend_scratch_bytes: u64,
    pub backend_scratch_peak_bytes: u64,
    /// Zero when the runtime never sampled the device budget: every measured
    /// field above is then zero because it is unavailable, not because it is nil.
    pub device_memory_samples: u64,

    /// Number of retained activation checkpoints in the last backward graph.
    /// Zero when the run is not checkpointing or has not built one yet, which is
    /// what makes every `checkpoint_*` field below unavailable rather than nil,
    /// the same convention `device_memory_samples` carries for the block above.
    pub checkpoint_count: u64,
    /// Bytes the checkpoints hold *as held*: a run with `checkpoint_dtype = f16`
    /// reports the 16-bit copies, not the F32 tensors the forward built. It is the
    /// numerator of `checkpoint_share_of_device_peak`.
    pub checkpoint_retained_bytes: u64,
    /// Most checkpoint bytes alive at any one point of the backward, and how many
    /// tensors that was. Below `checkpoint_retained_bytes` when the lifetimes do
    /// not all overlap - which is the only case where an offload ring of finite
    /// depth has anything to gain.
    pub checkpoint_live_peak_bytes: u64,
    pub checkpoint_live_peak_count: u64,
    /// Bytes held by the checkpoints alive across at least half the backward: the
    /// subset an activation offload would move first, since they are the ones whose host
    /// round-trip has compute to hide behind.
    pub checkpoint_long_lived_bytes: u64,
    pub checkpoint_long_lived_count: u64,
    /// Lifetime spans in backward-graph node positions, **not** seconds. Nodes
    /// differ by orders of magnitude in cost, so these rank checkpoints against
    /// each other and say nothing about durations.
    pub checkpoint_graph_nodes: u64,
    pub checkpoint_max_span_nodes: u64,
    pub checkpoint_total_span_nodes: u64,
}

impl MemoryReport {
    /// Whether the measured device fields carry a real measurement.
    pub fn is_measured(&self) -> bool {
        self.device_memory_samples > 0
    }

    /// Sum of the LoRA parameters, their gradients and the AdamW moments: the
    /// whole trainable-state cost, which always lives on one side of the
    /// device/host split.
    pub fn lora_total_bytes(&self) -> u64 {
        self.lora_parameter_bytes + self.lora_gradient_bytes + self.adamw_momenta_bytes
    }

    /// Whether the retained checkpoints were measured at all. False means the run
    /// is not checkpointing or has not built a backward graph yet; every
    /// `checkpoint_*` field is then zero because it is unavailable.
    pub fn has_checkpoint_profile(&self) -> bool {
        self.checkpoint_count > 0
    }

    /// Share of the measured device peak the retained checkpoints account for.
    /// Below 0.20, an activation offload ring is not worth designing.
    ///
    /// `None` without both terms, because a ratio against an unmeasured peak is
    /// not a small share, it is no answer. Note the peak is device-wide, so on a
    /// shared GPU this reads *low*: it is a floor on the share, and a floor is the
    /// safe direction for a trigger that opens work.
    pub fn checkpoint_share_of_device_peak(&self) -> Option<f64> {
        (self.has_checkpoint_profile() && self.device_peak_used_bytes > 0)
            .then(|| self.checkpoint_retained_bytes as f64 / self.device_peak_used_bytes as f64)
    }

    /// Mean checkpoint lifetime as a fraction of the backward graph, in node
    /// positions. `None` without a profile. See `checkpoint_graph_nodes` for why
    /// this is not a duration.
    pub fn checkpoint_mean_span_fraction(&self) -> Option<f64> {
        (self.has_checkpoint_profile() && self.checkpoint_graph_nodes > 0).then(|| {
            self.checkpoint_total_span_nodes as f64
                / (self.checkpoint_count as f64 * self.checkpoint_graph_nodes as f64)
        })
    }

    /// How much the measured device peak exceeds the summed allocations: the
    /// backends' own scratch and the graph allocator's transient reserve, i.e.
    /// the gap no `ggml_backend_buffer` accounts for. `None` without a
    /// measurement, and zero rather than negative when the peak sits below the
    /// sum (device-wide readings include other processes).
    pub fn unaccounted_device_bytes(&self) -> Option<u64> {
        self.is_measured().then(|| {
            self.device_peak_used_bytes
                .saturating_sub(self.device_bytes)
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainMetrics {
    pub epoch: u32,
    pub epoch_complete: bool,
    pub global_step: u64,
    pub train_loss: f32,
    pub eval_loss: f32,
    pub tokens_per_second: f32,
    pub learning_rate: f32,
}

/// Aggregate forward-only score over an SFT dataset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvalMetrics {
    pub negative_log_likelihood: f64,
    pub supervised_tokens: u64,
}

impl EvalMetrics {
    /// Mean token-level cross-entropy over non-ignored labels.
    pub fn loss(self) -> f64 {
        self.negative_log_likelihood / self.supervised_tokens as f64
    }

    /// Exponential of [`Self::loss`].
    pub fn perplexity(self) -> f64 {
        self.loss().exp()
    }
}

crate::wire_enum! {
    /// How the reward command is spoken to.
    ///
    /// The protocol on the wire is the same in both: one JSON request per line
    /// in, one JSON response per line out, in order. What differs is the
    /// lifetime of the process that speaks it - and with it, who pays the
    /// command's startup.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
    pub enum RewardMode: serde {
        /// One process for the whole loop, kept alive between batches after a
        /// version handshake. The default: a rollout algorithm calls its reward
        /// once per sampling wave - up to `max_resample_factor` times per
        /// update - so a one-shot command pays its interpreter, its imports and
        /// whatever model or dataset it loads several hundred times a run, for
        /// work that is identical every time.
        ///
        /// The command must write each response line *and flush it*: nothing
        /// closes its stdin between batches, so an answer still sitting in a
        /// buffer is an answer the trainer never receives.
        #[default]
        Persistent = "persistent",
        /// One process per batch, stdin closed after the requests. What every
        /// reward command did before the persistent mode existed, and what a
        /// command that cannot hold state - or cannot flush per line - still
        /// wants.
        OneShot = "oneshot",
    }
}

/// How the reward command is spoken to, and for how long it may take.
///
/// The timeout bounds one batch: the whole exchange in [`RewardMode::OneShot`],
/// and - because a persistent worker's startup is paid inside its first
/// exchange - the handshake plus the batch on the first call of
/// [`RewardMode::Persistent`]. A reward that loads a model must therefore be
/// given a timeout that covers loading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewardProtocol {
    pub mode: RewardMode,
    pub timeout: std::time::Duration,
}

/// Default batch timeout for a reward command. Generous on purpose: it is a
/// deadlock guard, not a performance budget, and the first call of a persistent
/// worker also pays whatever the command loads at startup.
pub const DEFAULT_REWARD_TIMEOUT_SECONDS: u64 = 300;

impl Default for RewardProtocol {
    fn default() -> Self {
        Self {
            mode: RewardMode::default(),
            timeout: std::time::Duration::from_secs(DEFAULT_REWARD_TIMEOUT_SECONDS),
        }
    }
}

/// Rollout sampling parameters. Temperature and top-p only shape exploration;
/// reported logprobs always describe the temperature-1 model policy that PPO
/// optimizes.
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: u32,
    pub seed: u32,
}

/// A sampled completion plus the rollout-time policy logprob of each token.
#[derive(Clone, Debug)]
pub struct Generation {
    pub tokens: Vec<i32>,
    pub logprobs: Vec<f32>,
}

/// Ceiling on the number of sparse targets a position may carry, mirroring
/// `RETRO_FUSED_CE_K_MAX` / `GGML_FUSED_SPARSE_CE_K_MAX`. The bound is the
/// scratch the Vulkan and Metal kernels reserve per position, so it is part of
/// the operator's contract rather than a preference of this crate.
pub const FUSED_CE_K_MAX: usize = 32;

/// Fixed-shape weighted training rows: `labels` mirrors `tokens` shifted by
/// one (-1 = ignored) and `weights` scales each label position's loss
/// contribution (0 = ignored). The runtime gradient per position is
/// `weight * (softmax - one_hot)`, which is the differentiable PPO objective
/// once the weights carry the detached clipped-surrogate coefficients.
#[derive(Clone, Debug)]
pub struct WeightedBatch {
    pub tokens: Vec<i32>,
    pub labels: Vec<i32>,
    pub weights: Vec<f32>,
    pub n_rows: usize,
    pub n_ctx: usize,
    /// Sparse targets per position. `1` is one target per
    /// position - the policy-gradient objective above. Above `1`, `labels` and
    /// `weights` hold `n_rows * n_ctx * n_topk` values, entry `j` of position
    /// `p` of row `r` at `(r * n_ctx + p) * n_topk + j`, and the weights are the
    /// teacher's renormalized probabilities: the objective is then the sparse
    /// cross-entropy against that distribution.
    pub n_topk: usize,
}

impl WeightedBatch {
    pub fn validate(&self) -> Result<()> {
        if self.n_rows == 0 || self.n_ctx == 0 {
            return Err(Error::invalid("weighted batch must contain rows"));
        }
        let expected = self
            .n_rows
            .checked_mul(self.n_ctx)
            .ok_or_else(|| Error::overflow("weighted batch shape overflows usize"))?;
        if self.n_topk == 0 || self.n_topk > FUSED_CE_K_MAX {
            return Err(Error::invalid(format!(
                "weighted batch n_topk must be in 1..={FUSED_CE_K_MAX}, got {}",
                self.n_topk
            )));
        }
        let entries = expected
            .checked_mul(self.n_topk)
            .ok_or_else(|| Error::overflow("weighted batch shape overflows usize"))?;
        if self.tokens.len() != expected
            || self.labels.len() != entries
            || self.weights.len() != entries
        {
            return Err(Error::invalid("invalid weighted batch shape"));
        }
        Ok(())
    }
}

/// Logical position in the training run. Counters are those of the *next* step
/// to execute, so a resume neither replays nor skips work.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Progress {
    pub version: u32,
    /// Next epoch (SFT) or next update (PPO/GRPO), zero-based.
    pub epoch: u64,
    /// Next optimizer step.
    pub global_step: u64,
    /// Next row/prompt cursor inside the dataset pass.
    pub cursor: u64,
    pub algorithm: String,
    pub phase: String,
    /// Best held-out value seen so far, for early stopping.
    pub best_eval: Option<f64>,
    pub stale_evaluations: u32,
    /// State of GRPO's adaptive KL controller at the next update boundary.
    /// Other algorithms leave it unset.
    #[serde(default)]
    pub kl_multiplier: Option<f32>,
}

/// Dataset identity and reading position. A different dataset or a different
/// preparation is refused, not silently resumed.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Dataset {
    pub version: u32,
    /// Logical path as configured, for diagnostics.
    pub path: String,
    /// Content fingerprint of the prepared dataset.
    pub fingerprint: String,
    pub examples: u64,
    /// Row width for SFT, generation budget for rollout algorithms.
    pub row_width: u64,
    pub format: String,
    /// Current shuffle permutation when the algorithm uses one.
    #[serde(default)]
    pub permutation: Vec<u32>,
    pub cursor: u64,
}

/// What the loader does when an `artifacts/` entry is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPolicy {
    /// Resuming without it is an error.
    Required,
    /// The algorithm can rebuild it.
    Recreatable,
    /// Reporting only.
    Ignorable,
}

/// What the caller contributes to a checkpoint: everything the runtime cannot
/// know on its own. The optimizer, scheduler, and RNG state are read from the
/// runtime instead.
#[derive(Clone, Debug)]
pub struct CheckpointMetadata {
    /// `step-000000000123` or `best`; also names the adapter GGUF.
    pub checkpoint_id: String,
    pub algorithm: String,
    /// Fingerprint of all settings that can change the continued trajectory.
    pub trajectory_signature: String,
    /// Coarsest unit a resume may restart at, e.g. `epoch` or `update`.
    pub resume_boundary: String,
    /// `constant`, `linear`, or `cosine`.
    pub scheduler_kind: String,
    pub warmup_steps: u64,
    pub progress: Progress,
    pub dataset: Dataset,
    pub seeds: BTreeMap<String, u64>,
    /// Per-algorithm extras written under `artifacts/`, with the policy the
    /// loader must apply when one is missing.
    pub artifacts: BTreeMap<String, (ArtifactPolicy, Vec<u8>)>,
    /// Base model, for the size recorded in the manifest.
    pub model_path: PathBuf,
}

/// What a restored checkpoint tells the driver: where to resume, and with what.
#[derive(Clone, Debug)]
pub struct ResumeInfo {
    /// The adapter GGUF that was loaded.
    pub adapter: PathBuf,
    pub progress: Progress,
    pub dataset: Dataset,
    pub seeds: BTreeMap<String, u64>,
    pub artifacts: BTreeMap<String, Vec<u8>>,
    /// False when the checkpoint predates the first optimizer step, so the
    /// resume starts from a cold optimizer.
    pub had_moments: bool,
}

impl ResumeInfo {
    /// Next optimizer step to execute.
    pub fn global_step(&self) -> u64 {
        self.progress.global_step
    }

    /// Next epoch (SFT) or update (PPO/GRPO), zero-based.
    pub fn epoch(&self) -> u64 {
        self.progress.epoch
    }
}

/// A single ggml training-backward op that can be run in isolation on a chosen
/// backend, for GPU-vs-CPU correctness testing via `probe_op`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOp {
    /// SILU backward: `dst = dy * s * (1 + x*(1-s))`, `s = sigmoid(x)`.
    /// `src0 = dy` (grad), `src1 = x` (forward input), same shape.
    SiluBack,
    /// RMS-norm backward. `src0 = dy` (grad), `src1 = x` (forward input), same
    /// shape; `params[0] = eps`. Output shape == `src0`.
    RmsNormBack,
    /// Out-prod / weight-gradient GEMM: `dst[i0,i1] = Σ_k src0[i0,k]·src1[i1,k]`
    /// per batch. Output shape is `[src0.ne0, src1.ne0,...]`, so size the output
    /// buffer accordingly.
    OutProd,
    /// Soft-max backward: `dx = (dy - dot(y, dy)) * y * scale` per row.
    /// `src0 = dy` (grad of softmax output), `src1 = y` (softmax output), same
    /// shape; `params[0] = scale`, `params[1] = max_bias` (must be 0).
    SoftMaxBack,
    /// Cross-entropy loss forward: `src0 = logits`, `src1 = labels` (same
    /// shape); output is a single scalar (`out_len = 1`).
    CrossEntropyLoss,
    /// Cross-entropy loss backward: `src0 = grad` of the loss (scalar, shape
    /// `[1,1,1,1]`), `src1 = logits`, `src2 = labels`. Output has the logits
    /// shape.
    CrossEntropyLossBack,
    /// Get-rows backward (scatter-add): `src0 = grad rows [n_embd, n_rows]`,
    /// `src1 = row indices` as floats (cast to i32 internally, shape
    /// `[n_rows,1,1,1]`), `src2 = shape template [n_embd, n_vocab]` (data
    /// unused). Output has the `src2` shape; duplicate indices accumulate.
    GetRowsBack,
    /// Out-prod with a Q8_0-quantized `src0` (the activation-gradient case
    /// where `src0` is a quantized model weight). `src0` is passed as F32 and
    /// quantized to Q8_0 internally, so both backends dequantize identical
    /// blocks; `ne_src0[0]` must be a multiple of 32 (the Q8_0 block size).
    OutProdQ80,
    /// Out-prod with a Q5_0-quantized `src0`.
    OutProdQ50,
    /// Out-prod with a Q4_0-quantized `src0`.
    OutProdQ40,
    /// Out-prod with a Q4_1-quantized `src0`.
    OutProdQ41,
    /// Out-prod with a Q5_1-quantized `src0`.
    OutProdQ51,
    /// Out-prod with a Q2_K-quantized `src0` (256-value blocks).
    OutProdQ2K,
    /// Out-prod with a Q3_K-quantized `src0` (256-value blocks).
    OutProdQ3K,
    /// Out-prod with a Q4_K-quantized `src0` (256-value blocks).
    OutProdQ4K,
    /// Out-prod with a Q5_K-quantized `src0` (256-value blocks).
    OutProdQ5K,
    /// Out-prod with a Q6_K-quantized `src0` (256-value blocks).
    OutProdQ6K,
    /// SSM convolution backward; `src2` is the convolution-output gradient.
    SsmConvBack,
    /// SSM scan backward probe; see `retro_probe_op` for its packed input.
    SsmScanBack,
    /// One AdamW update of an F16 parameter with F32 gradient and moments.
    /// `params = [learning_rate, weight_decay]` and output shape == `src0`.
    /// `src2`, when given, holds the single gradient-clipping scale the kernel
    /// folds into the gradient; it defaults to 1.
    OptStepAdamwF16,
    /// Streaming Vulkan Flash Attention backward (packed probe ABI).
    FlashAttnBack,
    /// L2-norm backward. `src0 = dy` (grad), `src1 = x` (forward input), same
    /// shape; `params[0] = eps`. Output shape == `src0`.
    L2NormBack,
    /// Gated delta net (Qwen3-Next / KDA) backward probe; see
    /// `retro_probe_op` for its packed input layout.
    GatedDeltaNetBack,
    /// Recurrent-state rollback snapshot gather. `src0 = conv_input`
    /// `[kernel_m1 + n_seq_tokens, n_channels, n_seqs]`,
    /// `params = [kernel_m1, K]`; `src1` is ignored but must be non-null.
    /// Output shape is `[kernel_m1 * n_channels, n_seqs, K]`.
    ConvRsGather,
    /// Repeat backward: `src0` is the broadcast gradient, `src1` the shape
    /// template it reduces onto (its data is unused, only its shape). Every
    /// `src1` dimension must divide the matching `src0` one. Output has the
    /// `src1` shape.
    RepeatBack,
    /// Out-prod with a non-F32 `src0`, the type given by `params[0]` as a
    /// `ggml_type` id; enumerate the valid ids with `retrograd_engine::dequant_types`.
    /// Supersedes the `OutProd<T>` variants above, which remain for the
    /// existing targeted tests: one op parameterized by type rather than one
    /// variant per type, so a type added to `GGML_RETRO_DEQUANT_TYPES` needs no new
    /// ABI. `src0` is passed as F32 and converted internally, so both backends
    /// decode identical bytes; `ne_src0[0]` must be a multiple of the type's block
    /// size (and of 16, which every block size here already implies).
    OutProdQuant,
    /// Inclusive prefix sum along `ne0`. `src0 = x`; `src1` is ignored but must
    /// be passed (the probe ABI always carries two inputs). Output shape ==
    /// `src0`. The second op with a RIR variant.
    Cumsum,
}

impl ProbeOp {
    pub fn id(self) -> i32 {
        match self {
            ProbeOp::SiluBack => 1,
            ProbeOp::RmsNormBack => 2,
            ProbeOp::OutProd => 3,
            ProbeOp::SoftMaxBack => 4,
            ProbeOp::CrossEntropyLoss => 5,
            ProbeOp::CrossEntropyLossBack => 6,
            ProbeOp::GetRowsBack => 7,
            ProbeOp::OutProdQ80 => 8,
            ProbeOp::OutProdQ50 => 9,
            ProbeOp::SsmConvBack => 10,
            ProbeOp::SsmScanBack => 11,
            ProbeOp::OutProdQ2K => 12,
            ProbeOp::OutProdQ3K => 13,
            ProbeOp::OutProdQ4K => 14,
            ProbeOp::OutProdQ5K => 15,
            ProbeOp::OutProdQ6K => 16,
            ProbeOp::OutProdQ40 => 17,
            ProbeOp::OutProdQ41 => 18,
            ProbeOp::OutProdQ51 => 19,
            ProbeOp::OptStepAdamwF16 => 20,
            ProbeOp::FlashAttnBack => 21,
            ProbeOp::L2NormBack => 22,
            ProbeOp::GatedDeltaNetBack => 23,
            ProbeOp::ConvRsGather => 24,
            ProbeOp::RepeatBack => 25,
            ProbeOp::OutProdQuant => 26,
            ProbeOp::Cumsum => 27,
        }
    }
}

/// Which kernel implementation a probe asks for, and which one ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelImpl {
    /// Whatever the configured RIR mode selects; a rejected contract falls back
    /// to the native kernel before launch and is reported, never hidden.
    Auto,
    /// Force the native kernel for this run, whatever the configured mode is.
    Native,
    /// Require the RIR variant. The probe fails instead of falling back, so a
    /// green "RIR" test cannot be exercising the native kernel.
    Rir,
}

impl KernelImpl {
    pub fn id(self) -> i32 {
        match self {
            KernelImpl::Auto => 0,
            KernelImpl::Native => 1,
            KernelImpl::Rir => 2,
        }
    }

    pub fn from_id(id: i32) -> Option<Self> {
        match id {
            0 => Some(KernelImpl::Auto),
            1 => Some(KernelImpl::Native),
            2 => Some(KernelImpl::Rir),
            _ => None,
        }
    }
}

/// Buckets `RirCounters::reject_by_reason` carries: one per [`KernelReject`]
/// value, `Matched` included. Mirrors `RETRO_KERNEL_REJECT_COUNT` in
/// `retro_lora_train.h` and `GGML_RIR_REJECT_COUNT` in ggml.
pub const KERNEL_REJECT_COUNT: usize = 14;

/// Why a RIR variant did not run. Mirrors `ggml_rir_reject`; the values are ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelReject {
    /// The contract matched. Paired with [`KernelImpl::Native`] it means the
    /// mode did not dispatch (observe), or the op has no registered variant.
    Matched,
    WrongOp,
    Dtype,
    Rank,
    Shape,
    Stride,
    QuantBlock,
    IntegerRange,
    MissingFeature,
    Pipeline,
    PolicyNative,
    DeviceGrid,
    DeviceAlignment,
    /// The node names a member of an op family no registered kernel writes.
    OpVariant,
    /// A value this build does not know; carried through rather than dropped.
    Unknown(i32),
}

impl KernelReject {
    pub fn from_id(id: i32) -> Self {
        match id {
            0 => KernelReject::Matched,
            1 => KernelReject::WrongOp,
            2 => KernelReject::Dtype,
            3 => KernelReject::Rank,
            4 => KernelReject::Shape,
            5 => KernelReject::Stride,
            6 => KernelReject::QuantBlock,
            7 => KernelReject::IntegerRange,
            8 => KernelReject::MissingFeature,
            9 => KernelReject::Pipeline,
            10 => KernelReject::PolicyNative,
            11 => KernelReject::DeviceGrid,
            12 => KernelReject::DeviceAlignment,
            13 => KernelReject::OpVariant,
            other => KernelReject::Unknown(other),
        }
    }
}

/// What a probe run actually did. See `retrograd_engine::probe_op_ex`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRunInfo {
    pub requested: KernelImpl,
    /// Always [`KernelImpl::Native`] or [`KernelImpl::Rir`], never `Auto`.
    pub executed: KernelImpl,
    pub reject: KernelReject,
    /// `variant_id` of the RIR variant that ran; empty when native ran.
    pub variant: String,
}

/// The RIR activation policy. Fixed before any backend
/// context is created, because the mode is what decides which pipelines get
/// built; changing it afterwards could only make a report lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RirMode {
    /// Native path only. The default, and it must stay the same default as the
    /// runtime's (`retro_lora_train.h`: `RETRO_RIR_MODE_OFF` is what
    /// `retro_runtime_config` starts from). This value is what a profile falls
    /// back to when nobody probed the engine, so disagreeing with the runtime
    /// would make a plan claim `prefer` for a run that executes `off`. Flipping
    /// it is a live open point with an unmet prerequisite (changing the default
    /// from `off` to `prefer`) - not a default to change on the Rust side alone.
    #[default]
    Off,
    /// Decide eligibility and count it, but always run the native kernel.
    Observe,
    /// Run the RIR variant when the contract matches; fall back before launch.
    Prefer,
    /// `Prefer` plus a graph preflight: a backend refuses to run a graph in
    /// which any node with an integrated `(op, backend)` is outside its
    /// variant's contract, instead of falling back to the native kernel. The
    /// failure surfaces as an error before anything is encoded, so
    /// `native_dispatched` stays at zero.
    ///
    /// Only nodes the registry actually targets are required: an op with no
    /// variant on that backend, or one the policy table pins to native, still
    /// runs natively under `Require`.
    Require,
}

impl RirMode {
    pub fn id(self) -> i32 {
        match self {
            RirMode::Off => 0,
            RirMode::Observe => 1,
            RirMode::Prefer => 2,
            RirMode::Require => 3,
        }
    }

    /// Anything unrecognized maps to `Off`: an unknown policy must never
    /// silently enable a variant.
    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => RirMode::Observe,
            2 => RirMode::Prefer,
            3 => RirMode::Require,
            _ => RirMode::Off,
        }
    }
}

/// Snapshot of the process-wide RIR dispatch counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RirCounters {
    /// Configured mode: 0 off, 1 observe, 2 prefer, 3 require.
    pub mode: i32,
    /// Ops with a registered variant that reached a dispatch site.
    pub ops_seen: u64,
    /// Of those, how many matched the variant's contract.
    pub rir_eligible: u64,
    /// Of those, how many were actually encoded as the RIR pipeline.
    pub rir_dispatched: u64,
    /// Native kernel ran for an op that has a variant.
    pub native_dispatched: u64,
    pub fallback_contract: u64,
    pub fallback_feature: u64,
    pub fallback_pipeline: u64,
    /// The same rejections bucketed by [`KernelReject`] discriminant, in the
    /// enum's declaration order. The aggregates above say how much fell back;
    /// this says what to widen first on a real graph.
    pub reject_by_reason: [u64; KERNEL_REJECT_COUNT],
}

/// Loss and grad-of-`h` from both the full-vocab path and the fused/tiled
/// reference, for the same inputs. See `fused_sparse_ce_probe`.
#[derive(Debug, Clone)]
pub struct FusedCeProbe {
    /// Scalar loss from the current path (`mul_mat` → weighted cross-entropy).
    pub loss_full: f32,
    /// Scalar loss from the fused/tiled reference.
    pub loss_fused: f32,
    /// `d loss / d h` (`[n_embd, n_tokens]`, token-major) from the full path.
    pub grad_h_full: Vec<f32>,
    /// `d loss / d h` from the fused/tiled reference.
    pub grad_h_fused: Vec<f32>,
}

/// Projection-head storage exercised by the fused path in `fused_sparse_ce_probe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedCeWeightType {
    /// Full-precision head.
    F32,
    /// Q8_0-quantized head (the fused operator dequantizes each row on the fly).
    Q8_0,
    /// Q6_K-quantized head - the type of a tied `token_embd`/`output` weight in
    /// common k-quant models (e.g. Qwen3 Q4_K_M).
    Q6K,
    /// Q4_0-quantized head - a legacy quant (block 32, `QUANT_R == 2`), exercising
    /// the other dequant interleave of the generic path.
    Q4_0,
    /// Q4_K-quantized head (k-quant, block 256).
    Q4K,
    /// Q5_K-quantized head (k-quant, block 256).
    Q5K,
    /// F16 head (converted to F32 for the CPU oracle).
    F16,
    /// Any head type, named by its raw `ggml_type` id. Lets a test sweep every
    /// entry of `retrograd_engine::dequant_types` without a variant per type; the
    /// named variants above stay for the readable targeted tests.
    Ggml(i32),
}

impl FusedCeWeightType {
    pub fn id(self) -> i32 {
        match self {
            FusedCeWeightType::F32 => 0,
            FusedCeWeightType::Q8_0 => 1,
            FusedCeWeightType::Q6K => 2,
            FusedCeWeightType::Q4_0 => 3,
            FusedCeWeightType::Q4K => 4,
            FusedCeWeightType::Q5K => 5,
            FusedCeWeightType::F16 => 6,
            // Must match RETRO_FUSED_CE_W_TYPE_GGML_BASE in retro_lora_train.h.
            FusedCeWeightType::Ggml(type_id) => Self::GGML_BASE + type_id,
        }
    }

    /// Offset that distinguishes a raw `ggml_type` id from the shorthand wire
    /// values above.
    pub const GGML_BASE: i32 = 1000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_ffi_values_match_the_c_header() {
        // These must stay in lock-step with retro_device in
        // runtime/include/retro_lora_train.h (AUTO=0, CPU=1, GPU=2). A silent
        // drift here would mis-select the backend at runtime.
        assert_eq!(Device::Auto.as_ffi(), 0);
        assert_eq!(Device::Cpu.as_ffi(), 1);
        assert_eq!(Device::Gpu.as_ffi(), 2);
    }

    #[test]
    fn rir_device_reject_ids_stay_distinct_from_portable_rejects() {
        assert_eq!(KernelReject::from_id(7), KernelReject::IntegerRange);
        assert_eq!(KernelReject::from_id(11), KernelReject::DeviceGrid);
        assert_eq!(KernelReject::from_id(12), KernelReject::DeviceAlignment);
        assert_eq!(KernelReject::from_id(13), KernelReject::OpVariant);
    }

    /// The bucket count is ABI: the C side copies `GGML_RIR_REJECT_COUNT`
    /// entries into an array this constant sizes, so a taxonomy that grows
    /// without this growing writes past the end. The
    /// `static_assert` in retro_probe.cpp holds C against ggml; this holds Rust
    /// against the enum it mirrors - every id below `KERNEL_REJECT_COUNT` must
    /// be a value this build knows.
    #[test]
    fn every_reject_bucket_has_a_known_reason() {
        for id in 0..KERNEL_REJECT_COUNT as i32 {
            assert!(
                !matches!(KernelReject::from_id(id), KernelReject::Unknown(_)),
                "bucket {id} has no known reason: \
                 KERNEL_REJECT_COUNT and KernelReject diverged"
            );
        }
        assert!(matches!(
            KernelReject::from_id(KERNEL_REJECT_COUNT as i32),
            KernelReject::Unknown(_)
        ));
    }

    #[test]
    fn device_default_is_auto() {
        assert_eq!(Device::default(), Device::Auto);
    }

    #[test]
    fn eval_metrics_report_mean_loss_and_perplexity() {
        let metrics = EvalMetrics {
            negative_log_likelihood: 6.0,
            supervised_tokens: 3,
        };
        assert_eq!(metrics.loss(), 2.0);
        assert!((metrics.perplexity() - 2.0_f64.exp()).abs() < 1.0e-12);
    }

    #[test]
    fn device_parser_does_not_promise_a_backend_it_cannot_select() {
        for alias in ["metal", "Vulkan", "cuda", "CUDA"] {
            assert!(alias.parse::<Device>().is_err(), "{alias}");
        }
        assert_eq!(" auto ".parse::<Device>().unwrap(), Device::Auto);
        assert_eq!("CPU".parse::<Device>().unwrap(), Device::Cpu);
        assert!("tpu".parse::<Device>().is_err());
    }

    #[test]
    fn qv_target_set_expands_to_q_and_v_projections() {
        let patterns = TargetSet::QV.patterns();
        assert_eq!(
            patterns,
            vec![
                "blk.*.attn_q.weight".to_string(),
                "blk.*.attn_v.weight".to_string(),
            ]
        );
    }

    #[test]
    fn explicit_patterns_are_passed_through_verbatim() {
        let custom = vec!["blk.0.ffn_up.weight".to_string()];
        let patterns = TargetSet::Patterns(custom.clone()).patterns();
        assert_eq!(patterns, custom);
    }

    #[test]
    fn lora_qv_constructor_sets_expected_defaults() {
        let cfg = LoraConfig::qv(8, 16.0);
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.dropout, 0.0);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.dtype, LoraDtype::F16);
        assert!(matches!(cfg.targets, TargetSet::QV));
    }
}
