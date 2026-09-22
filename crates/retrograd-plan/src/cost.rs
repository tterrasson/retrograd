//! The memory cost model.
//!
//! A pure function of the model geometry, the training configuration and the
//! workload. No device, no model file, no allocation - which is what lets the
//! bulk of the resolver's coverage live in the fast lane, and what lets the CLI
//! and the profiler ask the same question without a server.
//!
//! **It is deliberately conservative.** At equal uncertainty it overestimates: a
//! run that starts smaller than necessary is a disappointment, an OOM at update
//! 300 costs hours. Where a constant below is a judgement call rather
//! than a formula, that is said in place, and the calibration pass (step 6)
//! replaces it with what this machine actually did.
//!
//! **Every byte figure below is computed with `product` and `total`**, and
//! that is what makes the paragraph above true at the edges as well as in the
//! middle. See their documentation for why plain `*` was the one arithmetic in
//! this file that could round the wrong way.

use serde::Serialize;

use retrograd_core::{
    CheckpointDtype, KvDtype, LoraConfig, LoraDtype, MasterWeights, MemoryReport, ModelInfo,
    TargetSet, TensorDtype, TrainConfig, TrainableSet,
};

/// The product of a byte formula's factors, saturating at `u64::MAX`.
///
/// Every term here is a count of bytes built from three to five geometry
/// numbers, each of which reaches this file as a `u32` from a config file or a
/// model header. Nothing bounds their product: `n_ctx`, `n_ubatch` and `n_head`
/// near their own maxima overflow a `u64` between them, and a release build
/// wraps silently.
///
/// The direction is what makes that a defect rather than a large number. A
/// wrapped product is *small*, so the budget accepts a run it should have
/// refused - precisely the OOM at update 300 the module header says this file
/// exists to prevent, arrived at by the model that was supposed to prevent it.
/// Saturating keeps an impossible configuration impossible, which the caller
/// already knows how to refuse.
///
/// Written as a function over an array rather than a chain of `saturating_mul`
/// so that a five-factor formula still reads as a formula.
fn product<const N: usize>(factors: [u64; N]) -> u64 {
    factors.iter().fold(1u64, |acc, f| acc.saturating_mul(*f))
}

/// The sum of byte figures, saturating at `u64::MAX` - the companion of
/// [`product`], and saturating for the same reason: a total that wrapped would
/// be smaller than one of its own terms.
fn total<const N: usize>(terms: [u64; N]) -> u64 {
    terms.iter().fold(0u64, |acc, t| acc.saturating_add(*t))
}

/// Bytes per element of a KV cache entry.
fn kv_element_bytes(dtype: KvDtype) -> u64 {
    match dtype {
        KvDtype::F32 => 4,
        KvDtype::F16 => 2,
    }
}

fn checkpoint_element_bytes(dtype: CheckpointDtype) -> u64 {
    match dtype {
        CheckpointDtype::F32 => 4,
        CheckpointDtype::F16 | CheckpointDtype::Bf16 => 2,
    }
}

fn lora_element_bytes(dtype: LoraDtype) -> u64 {
    match dtype {
        LoraDtype::F32 => 4,
        LoraDtype::F16 => 2,
    }
}

/// Whether this run keeps an F32 master copy, resolved the way the runtime
/// resolves it: `auto` asks the *base* half of the trainable set, because that
/// is the only place a storage grid is coarser than the step it is given.
///
/// The budget and the run must agree here or the estimate is four bytes per
/// trained element short of what is allocated.
fn keeps_master_copy(training: &TrainConfig, base: Option<&TrainableSet>) -> bool {
    match training.master_weights {
        MasterWeights::Off => false,
        MasterWeights::F32 => true,
        MasterWeights::Auto => base.is_some_and(|set| {
            set.base_entries()
                .any(|entry| matches!(entry.dtype, TensorDtype::F16 | TensorDtype::BF16))
        }),
    }
}

/// Live tensors the backward pass keeps per transformer layer, in units of
/// `n_tokens × n_embd × 4`.
///
/// A judgement call, not a derivation: the residual stream is retained at the
/// layer boundary, the norm inputs, the attention projections and the FFN
/// intermediate all live at once inside a layer, and each has a gradient
/// counterpart. Twelve is above what a careful count gives on a plain
/// llama-style block, which is the side to err on. Calibration corrects it.
const ACTIVATION_TENSORS_PER_LAYER: u64 = 12;

/// Layers whose activations are live at once when gradient checkpointing is on:
/// the retained checkpoints, plus one segment being recomputed.
const RECOMPUTED_SEGMENTS: u64 = 1;

/// Increment whenever a cost formula, a lifetime assumption *or a post
/// name* changes. Calibration keys include this value, so stale measurements
/// cannot silently correct a different model - and `dominant_posts` is
/// published on the wire, so a reader that keys on a post name must be able to
/// tell an answer from an older model version rather than silently miss a
/// renamed line.
pub const COST_MODEL_VERSION: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum EstimateOrigin {
    Exact,
    Analytical,
    Catalog,
    Calibrated,
    Measured,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum EstimateBound {
    Exact,
    Upper,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ResourcePost {
    pub name: &'static str,
    pub bytes: u64,
    pub origin: EstimateOrigin,
    pub bound: EstimateBound,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PhaseResources {
    pub bytes: u64,
    pub posts: Vec<ResourcePost>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ResourceEstimate {
    pub cost_model_version: u32,
    pub persistent_device: PhaseResources,
    pub persistent_host: PhaseResources,
    pub optimizer_transient: PhaseResources,
    pub generation_transient: PhaseResources,
    pub kernel: PhaseResources,
    /// Common persistent allocations plus the larger mutually-exclusive phase.
    pub device_peak_bytes: u64,
    pub host_peak_bytes: u64,
    pub lower_bound_device_bytes: u64,
    pub upper_bound_device_bytes: u64,
}

/// What the run does with the model, beyond what [`TrainConfig`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadKind {
    /// One teacher-forced pass per row. No sampling, no rollout buffers.
    Sft,
    /// PPO or GRPO: sampling into a dedicated generation context, then a packed
    /// optimizer step over the rollouts.
    ///
    /// How many sequences decode at once is `training.generation_concurrency`,
    /// read from the configuration rather than duplicated here: it is a lever,
    /// and two copies of a lever's value drift.
    Rollout {
        /// Rollouts held in host memory for one update.
        rollouts_per_update: u64,
    },
}

impl WorkloadKind {
    /// Whether the optimizer runs on the packed differentiable path, where
    /// `n_ubatch` is part of the calculation and not a free memory lever.
    pub fn is_packed(self) -> bool {
        matches!(self, Self::Rollout { .. })
    }
}

/// Everything the cost model needs that is neither geometry nor `TrainConfig`.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    pub kind: WorkloadKind,
    /// Prepared rows held on the host: `tokens` and `labels`, both `i32`.
    pub examples: u64,
    /// Device bytes the *other* models occupy beside the one being sized,
    /// for the whole run: distillation's teacher, the fixed-reference anchor,
    /// or both.
    ///
    /// A flat term rather than a second geometry, because both are
    /// forward-only: no adapter, no backward graph, no optimizer state. Their
    /// cost is weights plus KV, computed by the caller from each model's
    /// `ModelInfo`; the cost model here only ever sees one model.
    pub co_resident_bytes: u64,
}

/// Per-machine correction factors learned by calibration and applied to the two
/// terms static analysis cannot predict to the MiB.
///
/// One is the identity, which is what a machine with no calibration file yet
/// uses. They only ever scale up an estimate that measurement found too low; a
/// factor below one is honoured too, but the resolver never *lowers* a
/// configuration because of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub compute_scale: f64,
    pub scratch_scale: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            compute_scale: 1.0,
            scratch_scale: 1.0,
        }
    }
}

/// The estimate, one field per line item.
///
/// The names that also exist in [`MemoryReport`] are spelled identically, so
/// `estimated` and `measured` can be subtracted field by field instead of
/// aligned. The fields with no measured
/// counterpart (`dequant_scratch_bytes`, the activation/logit split, the host
/// lines) are the estimate's own detail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct MemoryEstimate {
    pub model_weight_bytes: u64,
    /// The models held beside this one for the whole run: a distillation
    /// teacher, a fixed-reference anchor, or both. Separate from
    /// `model_weight_bytes` so an estimate can be compared with a
    /// `MemoryReport` byte by byte.
    pub co_resident_bytes: u64,
    /// Transient dequantization buffers the backend allocates for a quantized
    /// weight. Several GiB have been observed on CUDA.
    pub dequant_scratch_bytes: u64,
    pub optimizer_kv_bytes: u64,
    /// `activation_bytes + attention_bytes + logits_bytes`, the figure
    /// [`MemoryReport::optimizer_compute_bytes`] measures.
    pub optimizer_compute_bytes: u64,
    pub activation_bytes: u64,
    /// The materialized attention scores. The optimizer context has no flash
    /// attention on the differentiable path, so this term is real and grows
    /// with `n_ctx`.
    pub attention_bytes: u64,
    pub logits_bytes: u64,
    pub generation_kv_bytes: u64,
    pub generation_compute_bytes: u64,
    /// The resolved trainable set - LoRA factors today. Named for the role
    /// rather than for one policy, so a partial or hybrid estimate lands in the
    /// same posts instead of inventing parallel ones.
    pub trainable_parameter_bytes: u64,
    pub trainable_gradient_bytes: u64,
    /// Persistent optimizer state of that set: `8N` for AdamW, `0` for SGD,
    /// `4N` for a Muon-eligible matrix.
    pub optimizer_state_bytes: u64,
    /// Whether `trainable_parameter_bytes` is already inside
    /// `model_weight_bytes`, i.e. selected out of the loaded weights rather
    /// than allocated on top of them. The device rollup adds the parameters
    /// only when this is false.
    pub trainable_parameters_are_model_subset: bool,
    /// Prepared dataset rows: `tokens` and `labels`, `i32` each.
    pub host_dataset_bytes: u64,
    /// Rollout trajectories held for one update.
    pub host_rollout_bytes: u64,
}

impl MemoryEstimate {
    /// Everything drawn from the device budget.
    pub fn device_bytes(&self) -> u64 {
        total([
            self.model_weight_bytes,
            self.co_resident_bytes,
            self.dequant_scratch_bytes,
            self.optimizer_kv_bytes,
            self.optimizer_compute_bytes,
            self.generation_kv_bytes,
            self.generation_compute_bytes,
            self.trainable_parameter_bytes_on_top(),
            self.trainable_gradient_bytes,
            self.optimizer_state_bytes,
        ])
    }

    /// The trainable parameters *added* to the device budget: zero when they
    /// are a subset of the model weights already counted above. Never add
    /// `trainable_parameter_bytes` to `model_weight_bytes` directly.
    fn trainable_parameter_bytes_on_top(&self) -> u64 {
        if self.trainable_parameters_are_model_subset {
            0
        } else {
            self.trainable_parameter_bytes
        }
    }

    /// Everything drawn from the host budget.
    pub fn host_bytes(&self) -> u64 {
        total([self.host_dataset_bytes, self.host_rollout_bytes])
    }

    /// The line items, largest first, as `(name, bytes)`. What an
    /// `insufficient_memory` answer quotes so the caller sees *which* post
    /// overflows and by how much, not just that it does.
    pub fn dominant_posts(&self) -> Vec<(&'static str, u64)> {
        let mut posts = vec![
            ("model_weight_bytes", self.model_weight_bytes),
            ("dequant_scratch_bytes", self.dequant_scratch_bytes),
            ("optimizer_kv_bytes", self.optimizer_kv_bytes),
            ("activation_bytes", self.activation_bytes),
            ("attention_bytes", self.attention_bytes),
            ("logits_bytes", self.logits_bytes),
            ("generation_kv_bytes", self.generation_kv_bytes),
            ("generation_compute_bytes", self.generation_compute_bytes),
            (
                "trainable_parameter_bytes",
                self.trainable_parameter_bytes_on_top(),
            ),
            ("trainable_gradient_bytes", self.trainable_gradient_bytes),
            ("optimizer_state_bytes", self.optimizer_state_bytes),
            ("host_dataset_bytes", self.host_dataset_bytes),
            ("host_rollout_bytes", self.host_rollout_bytes),
        ];
        posts.retain(|(_, bytes)| *bytes > 0);
        // Ties broken by name: two posts of equal size must not reorder between
        // two identical resolutions (invariant 3, byte-identical output).
        posts.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(right.0)));
        posts
    }

    /// V2 lifetime-aware view of the same line items: which phase each post is
    /// reserved for, with its origin and its bound.
    ///
    /// The phases are a reporting split, not an exclusion: the runtime creates
    /// the generation context alongside the optimizer one and holds both for the
    /// trainer's lifetime, so `memory_totals` sums the two compute reserves and
    /// the device peak has to as well. Splitting them with `max` would let the
    /// resolver admit a rollout configuration the runtime then reports as over
    /// budget in phase 4.
    pub fn resources(&self) -> ResourceEstimate {
        let calibrated = |name, bytes| ResourcePost {
            name,
            bytes,
            origin: EstimateOrigin::Calibrated,
            bound: EstimateBound::Upper,
        };
        let analytical = |name, bytes| ResourcePost {
            name,
            bytes,
            origin: EstimateOrigin::Analytical,
            bound: EstimateBound::Upper,
        };
        let exact = |name, bytes| ResourcePost {
            name,
            bytes,
            origin: EstimateOrigin::Exact,
            bound: EstimateBound::Exact,
        };
        let phase = |posts: Vec<ResourcePost>| {
            let bytes = posts.iter().map(|post| post.bytes).sum();
            PhaseResources { bytes, posts }
        };

        let persistent_device = phase(vec![
            exact("model_weight_bytes", self.model_weight_bytes),
            analytical("optimizer_kv_bytes", self.optimizer_kv_bytes),
            analytical("generation_kv_bytes", self.generation_kv_bytes),
            analytical(
                "trainable_parameter_bytes",
                self.trainable_parameter_bytes_on_top(),
            ),
            analytical("trainable_gradient_bytes", self.trainable_gradient_bytes),
            analytical("optimizer_state_bytes", self.optimizer_state_bytes),
        ]);
        let persistent_host = phase(vec![
            analytical("host_dataset_bytes", self.host_dataset_bytes),
            analytical("host_rollout_bytes", self.host_rollout_bytes),
        ]);
        let optimizer_transient = phase(vec![
            calibrated("dequant_scratch_bytes", self.dequant_scratch_bytes),
            calibrated("optimizer_compute_bytes", self.optimizer_compute_bytes),
        ]);
        let generation_transient = phase(vec![
            calibrated("dequant_scratch_bytes", self.dequant_scratch_bytes),
            calibrated("generation_compute_bytes", self.generation_compute_bytes),
        ]);
        let kernel = PhaseResources::default();
        // Both transients are live at once, and `dequant_scratch_bytes` appears
        // in both phase breakdowns because either graph may use it - it is one
        // buffer, so it is added once here.
        let transient_peak = self
            .dequant_scratch_bytes
            .saturating_add(self.optimizer_compute_bytes)
            .saturating_add(self.generation_compute_bytes)
            .saturating_add(kernel.bytes);
        let device_peak_bytes = persistent_device.bytes.saturating_add(transient_peak);
        ResourceEstimate {
            cost_model_version: COST_MODEL_VERSION,
            persistent_device,
            persistent_host: persistent_host.clone(),
            optimizer_transient,
            generation_transient,
            kernel,
            device_peak_bytes,
            host_peak_bytes: persistent_host.bytes,
            // The analytical estimate is intentionally an upper bound; without
            // a preflight there is no defensible tighter lower bound than the
            // exact persistent weights and trainable allocations.
            lower_bound_device_bytes: self
                .model_weight_bytes
                .saturating_add(self.trainable_parameter_bytes_on_top())
                .saturating_add(self.trainable_gradient_bytes)
                .saturating_add(self.optimizer_state_bytes),
            upper_bound_device_bytes: device_peak_bytes,
        }
    }
}

impl From<&MemoryReport> for MemoryEstimate {
    /// The measured counterpart, in the same shape. The posts the runtime does
    /// not separate (the activation/attention/logit split, the dequantization
    /// scratch, the host lines) stay zero rather than being invented.
    fn from(report: &MemoryReport) -> Self {
        Self {
            model_weight_bytes: report.model_weight_bytes,
            optimizer_kv_bytes: report.optimizer_kv_bytes,
            optimizer_compute_bytes: report.optimizer_compute_bytes,
            generation_kv_bytes: report.generation_kv_bytes,
            generation_compute_bytes: report.generation_compute_bytes,
            trainable_parameter_bytes: report.trainable_parameter_bytes,
            trainable_gradient_bytes: report.trainable_gradient_bytes,
            optimizer_state_bytes: report.optimizer_state_bytes,
            trainable_parameters_are_model_subset: report.trainable_parameters_are_model_subset,
            ..Default::default()
        }
    }
}

/// Why a base-weight policy cannot be planned without the model's tensor
/// table, as the message a refusal carries.
///
/// The cost model sizes an adapter from the model's geometry, but a base
/// trainable set is priced per tensor and needs the GGUF inventory. Without it
/// the budget would be short by the largest term the run pays.
pub fn base_training_unpriced(policy: retrograd_core::TrainablePolicy) -> String {
    format!(
        "training.trainable = '{policy}' cannot be planned without the model's tensor \
         inventory: the gradients and optimizer state of a base trainable set are resolved \
         per tensor rather than derived from the model's aggregate geometry. Point the \
         planner at a readable [model].path, or train an adapter"
    )
}

/// The base half of this configuration's trainable set, resolved against the
/// model's own tensor table.
///
/// `None` for a `lora` document. A base-weight policy without an inventory is
/// refused by [`base_training_unpriced`]. One entry point for the resolver and
/// the server so the same set is not resolved twice under two rules.
pub fn resolve_trainable_set(
    training: &retrograd_core::TrainableRunConfig,
    inventory: Option<&retrograd_core::TensorInventory>,
) -> retrograd_core::Result<Option<TrainableSet>> {
    if !training.policy.trains_base_weights() {
        return Ok(None);
    }
    let Some(inventory) = inventory else {
        return Err(retrograd_core::Error::config(base_training_unpriced(
            training.policy,
        )));
    };
    retrograd_core::resolve_base(inventory, training.policy, &training.selector).map(Some)
}

/// What a run trains, as the estimate reads it.
///
/// Two independent halves because a run may have either or both: the adapter
/// it creates, sized analytically, and the base tensors it resolved, sized per
/// tensor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trainable<'a> {
    pub adapter: Option<&'a LoraConfig>,
    /// The resolved base set, base entries only: the adapter's factors do not
    /// exist in the GGUF and are sized by the adapter half.
    pub base: Option<&'a TrainableSet>,
}

impl<'a> Trainable<'a> {
    /// A LoRA run: an adapter and no base tensor.
    pub fn adapter(lora: &'a LoraConfig) -> Self {
        Self {
            adapter: Some(lora),
            base: None,
        }
    }

    /// A `full` or `partial` run: base tensors and no adapter.
    pub fn base(set: &'a TrainableSet) -> Self {
        Self {
            adapter: None,
            base: Some(set),
        }
    }

    /// How many of the model's `n_layer` blocks the backward graph reaches.
    ///
    /// The whole model unless a base-only selection has a lowest trainable
    /// block above zero. An adapter never prunes: its factors sit in every
    /// block it targets. A `full` policy answers "all" through the same call.
    pub fn backward_layers(&self, n_layer: u64) -> u64 {
        if self.adapter.is_some() {
            return n_layer;
        }
        let Some(set) = self.base.filter(|set| !set.is_empty()) else {
            return n_layer;
        };
        set.lowest_trainable_layer()
            .map_or(n_layer, |lowest| n_layer.saturating_sub(u64::from(lowest)))
            .clamp(1, n_layer)
    }
}

impl<'a> From<Option<&'a LoraConfig>> for Trainable<'a> {
    fn from(adapter: Option<&'a LoraConfig>) -> Self {
        Self {
            adapter,
            base: None,
        }
    }
}

/// Estimates the footprint of one configuration.
///
/// `trainable` carries whichever halves the policy has; a base policy missing
/// its set is refused upstream by [`crate::base_training_unpriced`].
pub fn estimate(
    model: &ModelInfo,
    training: &TrainConfig,
    trainable: Trainable<'_>,
    workload: &Workload,
    calibration: Calibration,
) -> MemoryEstimate {
    let lora = trainable.adapter;
    let n_ctx = training.n_ctx as u64;
    let n_ubatch = training.n_ubatch.max(1) as u64;
    let n_embd = model.n_embd as u64;
    let n_layer = model.n_layer.max(1) as u64;
    let n_vocab = model.n_vocab as u64;

    let model_weight_bytes = if model.model_size_bytes > 0 {
        model.model_size_bytes
    } else {
        model.file_size_bytes
    };

    // --- Optimizer KV ------------------------------------------------------
    // `kv_unified = true` in the runtime, so the cache holds `n_ctx` cells in
    // total whatever `n_seq_max` is.
    let optimizer_kv_bytes = total([
        model.kv_cache_bytes(n_ctx, kv_element_bytes(training.kv_dtype)),
        recurrent_state_bytes(model, training.n_seq_max.max(1) as u64),
    ]);

    // --- Activations -------------------------------------------------------
    // Retained layer outputs, plus the segment currently being recomputed, over
    // the layers the backward actually reaches. The blocks below the lowest
    // trainable one carry no backward node, so their forward tensors are freed
    // as they are consumed (see the `the_backward_prunes_the_blocks_below_...`
    // test in `tests/base_training.rs`).
    let backward_layers = trainable.backward_layers(n_layer);
    let per_layer_tokens = product([n_ubatch, n_embd]);
    let activation_bytes = if training.gradient_checkpointing {
        let every = training.checkpoint_every_n_layers.max(1) as u64;
        let retained = backward_layers.div_ceil(every);
        let checkpoints = product([
            retained,
            per_layer_tokens,
            checkpoint_element_bytes(training.checkpoint_dtype),
        ]);
        let recompute = product([
            product([every, RECOMPUTED_SEGMENTS]).min(backward_layers),
            per_layer_tokens,
            ACTIVATION_TENSORS_PER_LAYER,
            4,
        ]);
        total([checkpoints, recompute])
    } else {
        product([
            backward_layers,
            per_layer_tokens,
            ACTIVATION_TENSORS_PER_LAYER,
            4,
        ])
    };

    // --- Attention scores --------------------------------------------------
    // `[n_ctx, n_ubatch, n_head]` forward plus its gradient. The differentiable
    // optimizer context runs without flash attention, so this is materialized.
    let attention_bytes = product([2, model.n_head.max(1) as u64, n_ubatch, n_ctx, 4]);

    // --- Vocabulary logits -------------------------------------------------
    // The loss graph, not the option: a run that trains the projection head
    // takes the dense path, because the fused loss differentiates only its
    // hidden-state input.
    let fused_loss = training.chunked_cross_entropy
        && !trainable.base.is_some_and(TrainableSet::trains_loss_head);
    let logits_bytes = logits_estimate(model, training, n_ubatch, fused_loss);

    let optimizer_compute_bytes = scale(
        total([activation_bytes, attention_bytes, logits_bytes]),
        calibration.compute_scale,
    );

    // --- Generation context (RL only) --------------------------------------
    let (generation_kv_bytes, generation_compute_bytes, host_rollout_bytes) = match workload.kind {
        WorkloadKind::Sft => (0, 0, 0),
        WorkloadKind::Rollout {
            rollouts_per_update,
        } => {
            let concurrency = training.generation_concurrency.max(1) as u64;
            if !has_generation_context(training, concurrency) {
                // Sampling shares the optimizer context: no second cache, and
                // the rollouts still sit on the host.
                (0, 0, rollout_host_bytes(rollouts_per_update, n_ctx))
            } else {
                let element = if training.fast_generation_context {
                    2
                } else {
                    kv_element_bytes(training.kv_dtype)
                };
                let kv = total([
                    model.kv_cache_bytes(product([n_ctx, concurrency]), element),
                    recurrent_state_bytes(model, concurrency),
                ]);
                let batch = generation_batch(training, model, concurrency);
                // The output buffer llama.cpp reserves for the context, plus one
                // wave of activations.
                let compute = total([
                    product([batch, n_vocab, 4]),
                    product([batch, n_embd, ACTIVATION_TENSORS_PER_LAYER, 4]),
                ]);
                (
                    kv,
                    scale(compute, calibration.compute_scale),
                    rollout_host_bytes(rollouts_per_update, n_ctx),
                )
            }
        }
    };

    // --- Trainable set and optimizer state ---------------------------------
    // Two halves, summed. The adapter's factors sit *next to* the model and
    // add parameter bytes to the budget; the resolved base tensors are already
    // *inside* its weights, so only their gradients and optimizer state are
    // new. That is what `trainable_parameters_are_model_subset` records.
    let adapter_elements = trainable_parameters(model, lora);
    let adapter_parameter_bytes = product([
        adapter_elements,
        lora.map_or(0, |lora| lora_element_bytes(lora.dtype)),
    ]);
    let base = trainable.base.filter(|set| !set.is_empty());
    let base_elements = base.map_or(0, TrainableSet::n_parameters);
    let trainable_parameter_bytes = total([
        adapter_parameter_bytes,
        base.map_or(0, TrainableSet::parameter_bytes),
    ]);
    let trainable_gradient_bytes = product([total([adapter_elements, base_elements]), 4]);
    // The optimizer's own formula rather than AdamW's `8N`: an SGD run planned
    // against `8N` is over-budgeted by the whole optimizer. The adapter half
    // goes through the element-count entry point; the base half through the
    // resolved set, which is what Muon's shape rule needs.
    let optimizer = training.trainable.optimizer;
    let master_weights = keeps_master_copy(training, base);
    // The adapter's factors keep a master copy only when the run named one:
    // `auto` looks at the base tensors, and a rank-16 factor sits far above
    // its own grid anyway.
    let adapter_master_bytes =
        if master_weights && lora.is_some_and(|lora| lora_element_bytes(lora.dtype) == 2) {
            product([adapter_elements, 4])
        } else {
            0
        };
    let optimizer_state_bytes = total([
        optimizer.adapter_state_bytes(adapter_elements),
        adapter_master_bytes,
        base.map_or(0, |set| {
            optimizer.state_bytes_with_master(set, master_weights)
        }),
    ]);
    // Neither answer is right for a hybrid set, and the conservative one is
    // "added on top": the runtime's own report makes the same choice, so a
    // hybrid budget over-counts its base half in both places rather than
    // disagreeing with itself.
    let trainable_parameters_are_model_subset = base.is_some() && adapter_parameter_bytes == 0;

    MemoryEstimate {
        model_weight_bytes,
        co_resident_bytes: workload.co_resident_bytes,
        dequant_scratch_bytes: scale(dequant_scratch(model, training), calibration.scratch_scale),
        optimizer_kv_bytes,
        optimizer_compute_bytes,
        activation_bytes,
        attention_bytes,
        logits_bytes,
        generation_kv_bytes,
        generation_compute_bytes,
        trainable_parameter_bytes,
        trainable_gradient_bytes,
        optimizer_state_bytes,
        trainable_parameters_are_model_subset,
        // `tokens` and `labels`, one `i32` each per position.
        host_dataset_bytes: product([workload.examples, n_ctx, 8]),
        host_rollout_bytes,
    }
}

fn scale(bytes: u64, factor: f64) -> u64 {
    if factor == 1.0 {
        return bytes;
    }
    ((bytes as f64) * factor.max(0.0)) as u64
}

/// Whether the runtime builds a dedicated generation context, in exactly the
/// conditions `retro_backend.cpp` builds one.
fn has_generation_context(training: &TrainConfig, concurrency: u64) -> bool {
    concurrency > 1 || training.fast_generation_context || training.generation_batch != 0
}

/// The generation context's batch, mirroring the runtime's derivation: the
/// largest value keeping the reserved output buffer inside 64 MiB, capped by
/// `n_ctx` and 512, then raised to the concurrency so one decode wave fits a
/// single launch.
///
/// Public because the first memory lever needs it: `generation_batch = 0` means
/// "let the runtime derive it", and a lever cannot lower a number it cannot see.
pub fn generation_batch(training: &TrainConfig, model: &ModelInfo, concurrency: u64) -> u64 {
    let derived = if training.generation_batch != 0 {
        training.generation_batch as u64
    } else {
        let budget = 64u64 << 20;
        let affordable = (budget / (model.n_vocab.max(1) as u64 * 4)).max(1);
        (training.n_ctx as u64).min(512).min(affordable)
    };
    derived.max(concurrency)
}

/// Trajectories held on the host for one update: tokens, logprobs and the
/// weights the packed step consumes, at four bytes a position each.
fn rollout_host_bytes(rollouts_per_update: u64, n_ctx: u64) -> u64 {
    product([rollouts_per_update, n_ctx, 4, 4])
}

/// What a forward-only second model costs on the device for a whole run:
/// its weights, plus the KV cache the context it runs in reserves.
///
/// Weights plus KV and nothing else, because a `Trainer` with no adapter has
/// no backward graph, no gradients and no AdamW moments (see `distill::Teacher`
/// and `tests/distill_runtime.rs`). The compute term is
/// left out because a forward pass through a scoring batch peaks below the
/// optimizer graph this budget is already sized for.
///
/// The model inherits `n_ctx`, `n_seq_max` and `kv_dtype` from the training
/// configuration and nothing else.
pub fn co_resident_model_bytes(model: &ModelInfo, training: &TrainConfig) -> u64 {
    co_resident_model_bytes_at(model, training, training.n_ctx)
}

/// Like [`co_resident_model_bytes`] but at a context width other than the
/// training one. Only the KV term moves with it; the weights do not.
pub fn co_resident_model_bytes_at(model: &ModelInfo, training: &TrainConfig, n_ctx: u32) -> u64 {
    let weights = if model.model_size_bytes > 0 {
        model.model_size_bytes
    } else {
        model.file_size_bytes
    };
    total([
        weights,
        model.kv_cache_bytes(n_ctx as u64, kv_element_bytes(training.kv_dtype)),
        recurrent_state_bytes(model, training.n_seq_max.max(1) as u64),
    ])
}

/// Recurrent / SSM state, which is per-sequence and does *not* grow with
/// `n_ctx`. Zero for a plain transformer.
fn recurrent_state_bytes(model: &ModelInfo, sequences: u64) -> u64 {
    product([
        total([model.n_embd_r as u64, model.n_embd_s as u64]),
        model.n_layer as u64,
        sequences,
        4,
    ])
}

/// The vocabulary logits, the post that explodes on a large vocabulary.
///
/// Two terms: the output buffer llama.cpp reserves for one physical micro-batch
/// (`n_ubatch × n_vocab × 4` - the graph is evaluated one ubatch at a time, and
/// `n_batch` only sets how many of them accumulate before an optimizer step), and
/// the loss graph's own logits plus their gradient, which is what
/// `chunked_cross_entropy` bounds.
fn logits_estimate(
    model: &ModelInfo,
    training: &TrainConfig,
    n_ubatch: u64,
    fused_loss: bool,
) -> u64 {
    let n_vocab = model.n_vocab as u64;
    let reserved_outputs = product([n_ubatch, n_vocab, 4]);
    let graph = if fused_loss {
        let tiles = training.chunked_ce_tiles.max(1) as u64;
        let tokens = if training.chunked_ce_seq_chunk > 0 {
            (training.chunked_ce_seq_chunk as u64).min(n_ubatch)
        } else {
            n_ubatch
        };
        // One tile of logits and its gradient, plus the staging buffer for the
        // hidden-state gradient - which a bounded token chunk removes: the
        // backward then writes `grad_h` over the hidden states one chunk at a
        // time instead of into a second `[n_embd, n_tokens]` buffer. The engine
        // derives that in-place write from the chunk, so the chunk is the whole
        // condition here too.
        let staging = if training.chunked_ce_seq_chunk > 0 {
            0
        } else {
            product([n_ubatch, model.n_embd as u64, 4])
        };
        total([product([2, tokens, n_vocab.div_ceil(tiles), 4]), staging])
    } else {
        product([2, n_ubatch, n_vocab, 4])
    };
    total([reserved_outputs, graph])
}

/// Transient dequantization buffers, sized by the largest weight matrix the
/// backend has to expand.
///
/// Zero on CPU (which reads quantized blocks in place) and for an unquantized
/// model. Otherwise the widest matmul - the vocabulary head, or the FFN - held
/// in F32.
fn dequant_scratch(model: &ModelInfo, training: &TrainConfig) -> u64 {
    if training.device == retrograd_core::Device::Cpu {
        return 0;
    }
    let quantized = !matches!(
        model.dominant_weight_type.to_ascii_uppercase().as_str(),
        "F32" | "F16" | "BF16" | ""
    );
    if !quantized {
        return 0;
    }
    let n_embd = model.n_embd as u64;
    let widest = (model.n_vocab as u64).max(feed_forward_width(model));
    // Never more than the weights themselves: a dequantization buffer holds an
    // expanded *tensor*, so a figure above the whole model is arithmetic that
    // ran away, not a budget.
    let weights = if model.model_size_bytes > 0 {
        model.model_size_bytes
    } else {
        model.file_size_bytes
    };
    product([widest, n_embd, 4]).min(weights)
}

/// Feed-forward width published by the runtime, with a conservative fallback
/// for old/synthetic profiles that still carry zero.
pub fn feed_forward_width(model: &ModelInfo) -> u64 {
    if model.n_ff > 0 {
        return model.n_ff as u64;
    }
    let n_embd = model.n_embd as u64;
    let n_layer = model.n_layer.max(1) as u64;
    let fallback = 4 * n_embd;
    if n_embd == 0 || model.n_params == 0 {
        return fallback;
    }
    let embeddings = product([
        model.n_vocab as u64,
        n_embd,
        if model.tied_embeddings { 1 } else { 2 },
    ]);
    let attention = product([n_layer, attention_parameters(model)]);
    let experts = model.n_expert.max(1) as u64;
    let Some(remaining) = model
        .n_params
        .checked_sub(embeddings)
        .and_then(|value| value.checked_sub(attention))
    else {
        return fallback;
    };
    // Gated FFN: `up`, `gate` and `down`, each `n_embd × n_ff`, per expert.
    let divisor = product([n_layer, 3, n_embd, experts]);
    if divisor == 0 {
        return fallback;
    }
    let width = remaining / divisor;
    if (n_embd / 2..=16 * n_embd).contains(&width) {
        width
    } else {
        fallback
    }
}

fn attention_parameters(model: &ModelInfo) -> u64 {
    let n_embd = model.n_embd as u64;
    let q_out = product([model.n_head as u64, model.n_embd_head_k as u64]);
    let o_in = product([model.n_head as u64, model.n_embd_head_v as u64]);
    total([
        product([n_embd, q_out]),
        product([n_embd, model.n_embd_k_gqa as u64]),
        product([n_embd, model.n_embd_v_gqa as u64]),
        product([o_in, n_embd]),
    ])
}

/// `Σ_targets (in + out) × rank`, summed over layers.
///
/// [`TargetSet::Auto`] is resolved in the C++ runtime per architecture, so the
/// estimate assumes the widest plausible set - every attention and FFN
/// projection. Over-counting a term that is a rounding error next to the KV
/// cache is the right side to be wrong on.
pub fn trainable_parameters(model: &ModelInfo, lora: Option<&LoraConfig>) -> u64 {
    let Some(lora) = lora else {
        // No adapter, so no adapter parameters. The base tensors such a run
        // trains are a subset of the model weights and are sized from a
        // resolved set, which this function does not receive.
        return 0;
    };
    let n_embd = model.n_embd as u64;
    let n_ff = feed_forward_width(model);
    let q_out = product([model.n_head as u64, model.n_embd_head_k as u64]);
    let o_in = product([model.n_head as u64, model.n_embd_head_v as u64]);
    let widths = |name: &str| -> Option<(u64, u64)> {
        Some(match name {
            "attn_q" => (n_embd, q_out),
            "attn_k" => (n_embd, model.n_embd_k_gqa as u64),
            "attn_v" => (n_embd, model.n_embd_v_gqa as u64),
            "attn_output" => (o_in, n_embd),
            "ffn_up" | "ffn_gate" => (n_embd, n_ff),
            "ffn_down" => (n_ff, n_embd),
            _ => return None,
        })
    };
    const ALL: &[&str] = &[
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_up",
        "ffn_gate",
        "ffn_down",
    ];
    let selected: Vec<&str> = match &lora.targets {
        TargetSet::Auto => ALL.to_vec(),
        TargetSet::QV => vec!["attn_q", "attn_v"],
        TargetSet::Patterns(patterns) => {
            let mut names = Vec::new();
            for pattern in patterns {
                if let Some(name) = ALL.iter().find(|name| pattern.contains(*name)) {
                    // A pattern naming a single layer (`blk.0.attn_q.weight`)
                    // still counts as the whole stack: distinguishing them would
                    // need the layer index, and under-counting is the wrong
                    // direction.
                    names.push(*name);
                }
            }
            names.sort_unstable();
            names.dedup();
            names
        }
    };
    let per_layer: u64 = selected
        .iter()
        .filter_map(|name| widths(name))
        .map(|(input, output)| product([total([input, output]), lora.rank as u64]))
        .fold(0u64, u64::saturating_add);
    product([per_layer, model.n_layer.max(1) as u64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{sft_workload, tiny_model};

    fn rollout_training(n_ctx: u32) -> TrainConfig {
        TrainConfig {
            generation_concurrency: 8,
            ..training(n_ctx)
        }
    }

    /// A distillation workload: the same rollout shape, plus a resident teacher.
    fn distill_workload(teacher_bytes: u64) -> Workload {
        Workload {
            co_resident_bytes: teacher_bytes,
            ..rollout_workload()
        }
    }

    fn rollout_workload() -> Workload {
        Workload {
            kind: WorkloadKind::Rollout {
                rollouts_per_update: 16,
            },
            examples: 0,
            co_resident_bytes: 0,
        }
    }

    /// The unoptimised baseline, spelled out.
    ///
    /// Every test below measures what one setting *changes*, so the fixture pins
    /// the side of each one it treats as the starting point. Inheriting the
    /// defaults instead would leave the fused cross-entropy and the F16 cache on
    /// both sides of the comparison and make it vacuous - which is exactly how
    /// a cost model stops noticing that a term stopped shrinking.
    fn training(n_ctx: u32) -> TrainConfig {
        TrainConfig {
            n_ctx,
            n_batch: n_ctx,
            n_ubatch: 32,
            device: retrograd_core::Device::Gpu,
            kv_dtype: KvDtype::F32,
            chunked_cross_entropy: false,
            chunked_ce_seq_chunk: 0,
            ..Default::default()
        }
    }

    /// Two F32 norms, plus the vocabulary projection when `with_head`.
    fn base_set(with_head: bool) -> TrainableSet {
        use retrograd_core::{OUTPUT_HEAD, TensorDtype, TensorRole, TrainableEntry};
        let entry = |name: &str, ne: [i64; 4]| {
            let n_elements = ne.iter().product::<i64>() as u64;
            TrainableEntry {
                name: name.to_string(),
                role: TensorRole::Base,
                ne,
                dtype: TensorDtype::F32,
                n_elements,
                n_bytes: n_elements * 4,
                storage_id: 0,
            }
        };
        let mut entries = vec![
            entry("blk.0.attn_norm.weight", [512, 1, 1, 1]),
            entry("blk.1.attn_norm.weight", [512, 1, 1, 1]),
        ];
        if with_head {
            entries.push(entry(OUTPUT_HEAD, [512, 32_000, 1, 1]));
        }
        TrainableSet {
            policy: retrograd_core::TrainablePolicy::Partial,
            entries,
            exclusions: Vec::new(),
        }
    }

    /// A base set is priced per tensor and its parameters are a slice of the
    /// model's weights, not added on top of them.
    #[test]
    fn a_base_set_is_priced_per_tensor_and_never_added_on_top_of_the_model_weights() {
        let set = base_set(false);
        let priced = estimate(
            &tiny_model(),
            &training(1024),
            Trainable::base(&set),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(priced.trainable_parameter_bytes, 2 * 512 * 4);
        assert_eq!(priced.trainable_gradient_bytes, 2 * 512 * 4);
        assert_eq!(priced.optimizer_state_bytes, 2 * 512 * 8);
        assert!(priced.trainable_parameters_are_model_subset);

        // The device rollup honours that.
        let adapter = estimate(
            &tiny_model(),
            &training(1024),
            Trainable::adapter(&LoraConfig::auto(8, 16.0)),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(!adapter.trainable_parameters_are_model_subset);
        assert_eq!(
            priced.device_bytes() + priced.trainable_parameter_bytes,
            estimate(
                &tiny_model(),
                &training(1024),
                Trainable::default(),
                &sft_workload(),
                Calibration::default(),
            )
            .device_bytes()
                + priced.trainable_gradient_bytes
                + priced.optimizer_state_bytes
                + priced.trainable_parameter_bytes
        );
    }

    /// A partial selection that starts above the first block is charged only
    /// for the blocks its backward reaches: the activation term is over the
    /// span, not over the model.
    #[test]
    fn a_partial_selection_is_charged_for_the_layers_its_backward_reaches() {
        use retrograd_core::{TensorDtype, TensorRole, TrainableEntry};
        let norm = |name: &str| {
            let n_elements = 512_u64;
            TrainableEntry {
                name: name.to_string(),
                role: TensorRole::Base,
                ne: [512, 1, 1, 1],
                dtype: TensorDtype::F32,
                n_elements,
                n_bytes: n_elements * 4,
                storage_id: 0,
            }
        };
        let of = |entries: Vec<TrainableEntry>| TrainableSet {
            policy: retrograd_core::TrainablePolicy::Partial,
            entries,
            exclusions: Vec::new(),
        };
        let priced = |set: &TrainableSet| {
            estimate(
                &tiny_model(),
                &training(1024),
                Trainable::base(set),
                &sft_workload(),
                Calibration::default(),
            )
            .activation_bytes
        };

        // 24 blocks, so the last four cost a sixth of all of them.
        let bottom = of(vec![norm("blk.0.attn_norm.weight")]);
        let top = of(vec![norm("blk.20.attn_norm.weight")]);
        assert_eq!(priced(&bottom), priced(&top) * 6);

        // The lowest selected block decides, not the highest.
        let both = of(vec![
            norm("blk.0.attn_norm.weight"),
            norm("blk.20.attn_norm.weight"),
        ]);
        assert_eq!(priced(&both), priced(&bottom));

        // A tensor outside every block gets the conservative answer.
        let with_global = of(vec![
            norm("blk.20.attn_norm.weight"),
            norm("output_norm.weight"),
        ]);
        assert_eq!(priced(&with_global), priced(&bottom));

        // An adapter never prunes.
        let hybrid = Trainable {
            adapter: Some(&LoraConfig::auto(8, 16.0)),
            base: Some(&top),
        };
        assert_eq!(
            estimate(
                &tiny_model(),
                &training(1024),
                hybrid,
                &sft_workload(),
                Calibration::default(),
            )
            .activation_bytes,
            priced(&bottom)
        );
    }

    /// The optimizer state follows the chosen optimizer's own formula.
    #[test]
    fn a_base_sets_optimizer_state_follows_the_chosen_optimizer() {
        let set = base_set(false);
        let mut config = training(1024);
        config.trainable.optimizer = retrograd_core::OptimizerKind::Sgd;
        let sgd = estimate(
            &tiny_model(),
            &config,
            Trainable::base(&set),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(sgd.optimizer_state_bytes, 0);
        assert_eq!(sgd.trainable_gradient_bytes, 2 * 512 * 4);
    }

    /// A run that trains the projection head takes the dense path even with
    /// `chunked_cross_entropy` on.
    #[test]
    fn training_the_head_prices_dense_logits_even_with_chunked_cross_entropy_on() {
        let mut config = training(1024);
        config.chunked_cross_entropy = true;
        config.chunked_ce_tiles = 8;
        let norms = base_set(false);
        let with_head = base_set(true);
        let tiled = estimate(
            &tiny_model(),
            &config,
            Trainable::base(&norms),
            &sft_workload(),
            Calibration::default(),
        );
        let dense = estimate(
            &tiny_model(),
            &config,
            Trainable::base(&with_head),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(
            dense.logits_bytes > tiled.logits_bytes,
            "dense {} is not above tiled {}",
            dense.logits_bytes,
            tiled.logits_bytes
        );
        // Same figure the option-off run pays: same graph.
        let off = estimate(
            &tiny_model(),
            &training(1024),
            Trainable::base(&with_head),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(dense.logits_bytes, off.logits_bytes);
    }

    /// A base policy without an inventory is refused rather than priced.
    #[test]
    fn a_base_policy_without_an_inventory_is_refused_rather_than_priced() {
        let mut trainable = retrograd_core::TrainableRunConfig {
            policy: retrograd_core::TrainablePolicy::Partial,
            ..Default::default()
        };
        let error = resolve_trainable_set(&trainable, None).unwrap_err();
        assert!(error.to_string().contains("tensor inventory"), "{error}");
        // A LoRA document needs none and gets none.
        trainable.policy = retrograd_core::TrainablePolicy::Lora;
        assert!(resolve_trainable_set(&trainable, None).unwrap().is_none());
    }

    #[test]
    fn the_saturating_helpers_keep_a_total_above_its_terms() {
        assert_eq!(product([2, 3, 7]), 42);
        assert_eq!(total([1, 2, 3]), 6);
        assert_eq!(product([u64::MAX, 2]), u64::MAX);
        assert_eq!(total([u64::MAX, 1]), u64::MAX);
        // The property the model rests on, and the one plain `*` breaks: a
        // figure is never smaller than one of its own factors.
        let huge = product([u64::MAX / 3, 9, 4]);
        assert!(huge >= u64::MAX / 3);
    }

    #[test]
    fn rollout_counts_above_u32_are_not_underpriced() {
        let rollouts = u64::from(u32::MAX) + 9;
        assert_eq!(rollout_host_bytes(rollouts, 1), rollouts * 16);
        assert!(rollout_host_bytes(rollouts, 1) > rollout_host_bytes(u64::from(u32::MAX), 1));
    }

    /// Geometry whose factors can wrap. Each factor is inside its own `u32`
    /// field, so nothing upstream refuses this configuration; the product of
    /// four of them is not.
    ///
    /// With wrapping arithmetic the attention post comes out at a few gigabytes
    /// and the plan is accepted. What is asserted here is not a number but a *direction*:
    /// an absurd geometry must not price below a merely large one.
    #[test]
    fn an_overflowing_geometry_prices_above_a_large_one_instead_of_below_it() {
        let mut model = tiny_model();
        model.n_head = u32::MAX;
        let mut config = training(u32::MAX);
        config.n_ubatch = u32::MAX;

        let absurd = estimate(
            &model,
            &config,
            Trainable::adapter(&LoraConfig::auto(8, 16.0)),
            &sft_workload(),
            Calibration::default(),
        );
        let large = estimate(
            &tiny_model(),
            &training(131_072),
            Trainable::adapter(&LoraConfig::auto(8, 16.0)),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(
            absurd.attention_bytes >= large.attention_bytes,
            "attention: {} < {}",
            absurd.attention_bytes,
            large.attention_bytes
        );
        assert!(absurd.device_bytes() >= large.device_bytes());
    }

    #[test]
    fn the_kv_cache_grows_linearly_with_the_context_and_halves_in_f16() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let small = estimate(
            &model,
            &training(1024),
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        let large = estimate(
            &model,
            &training(2048),
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(large.optimizer_kv_bytes, 2 * small.optimizer_kv_bytes);

        let half = estimate(
            &model,
            &TrainConfig {
                kv_dtype: KvDtype::F16,
                ..training(2048)
            },
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(half.optimizer_kv_bytes, large.optimizer_kv_bytes / 2);
    }

    #[test]
    fn sgd_removes_only_the_persistent_optimizer_state_from_the_budget() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let mut config = training(1024);
        let adamw = estimate(
            &model,
            &config,
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        config.trainable.optimizer = retrograd_core::OptimizerKind::Sgd;
        let sgd = estimate(
            &model,
            &config,
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(adamw.optimizer_state_bytes > 0);
        assert_eq!(sgd.optimizer_state_bytes, 0);
        assert_eq!(
            adamw.device_bytes() - sgd.device_bytes(),
            adamw.optimizer_state_bytes
        );
    }

    #[test]
    fn chunked_cross_entropy_divides_the_logit_term_by_the_tile_count() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let plain = estimate(
            &model,
            &training(1024),
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        let chunked = estimate(
            &model,
            &TrainConfig {
                chunked_cross_entropy: true,
                chunked_ce_tiles: 8,
                ..training(1024)
            },
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(
            chunked.logits_bytes < plain.logits_bytes,
            "{} is not below {}",
            chunked.logits_bytes,
            plain.logits_bytes
        );
        // More tiles never costs more memory.
        let more = estimate(
            &model,
            &TrainConfig {
                chunked_cross_entropy: true,
                chunked_ce_tiles: 32,
                ..training(1024)
            },
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(more.logits_bytes <= chunked.logits_bytes);
    }

    #[test]
    fn a_resident_teacher_is_charged_to_the_device_and_not_to_the_host() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let training = training(1024);
        let alone = estimate(
            &model,
            &training,
            Trainable::adapter(&lora),
            &rollout_workload(),
            Calibration::default(),
        );
        const TEACHER: u64 = 4 * 1024 * 1024 * 1024;
        let with_teacher = estimate(
            &model,
            &training,
            Trainable::adapter(&lora),
            &distill_workload(TEACHER),
            Calibration::default(),
        );

        // Exactly the term, added exactly once, and to the device only: a
        // teacher's weights and KV are device residency, and charging them to
        // the host budget too would refuse machines that run.
        assert_eq!(with_teacher.co_resident_bytes, TEACHER);
        assert_eq!(alone.co_resident_bytes, 0);
        assert_eq!(with_teacher.device_bytes(), alone.device_bytes() + TEACHER);
        assert_eq!(with_teacher.host_bytes(), alone.host_bytes());
        // And nothing else moved: the student's own posts are a function of its
        // geometry, which the teacher does not touch.
        assert_eq!(with_teacher.model_weight_bytes, alone.model_weight_bytes);
        assert_eq!(with_teacher.optimizer_kv_bytes, alone.optimizer_kv_bytes);
    }

    #[test]
    fn a_forward_only_teacher_costs_its_weights_and_its_kv() {
        let model = tiny_model();
        let narrow = training(1024);
        let bytes = co_resident_model_bytes(&model, &narrow);
        let weights = model.model_size_bytes.max(model.file_size_bytes);

        // Weights plus a cache, never less than the weights and never the whole
        // optimizer footprint: a teacher has no adapter, so no gradients and no
        // AdamW moments exist for it to pay for.
        assert!(bytes > weights, "{bytes} is not above {weights}");
        let student = estimate(
            &model,
            &narrow,
            Trainable::adapter(&LoraConfig::auto(8, 16.0)),
            &rollout_workload(),
            Calibration::default(),
        );
        assert!(bytes < student.device_bytes());

        // A wider context is a bigger teacher, because the KV is the only term
        // that reads one.
        let wide = co_resident_model_bytes(&model, &training(4096));
        assert!(wide > bytes, "{wide} is not above {bytes}");
    }

    #[test]
    fn gradient_checkpointing_lowers_the_activation_term() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let plain = estimate(
            &model,
            &training(1024),
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        let checkpointed = estimate(
            &model,
            &TrainConfig {
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 4,
                ..training(1024)
            },
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(checkpointed.activation_bytes < plain.activation_bytes);

        // An F16 checkpoint halves the retained part, never raises the total.
        let f16 = estimate(
            &model,
            &TrainConfig {
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 4,
                checkpoint_dtype: CheckpointDtype::F16,
                ..training(1024)
            },
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert!(f16.activation_bytes < checkpointed.activation_bytes);
    }

    #[test]
    fn sft_pays_for_no_generation_context() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let sft = estimate(
            &model,
            &training(1024),
            Trainable::adapter(&lora),
            &sft_workload(),
            Calibration::default(),
        );
        assert_eq!(sft.generation_kv_bytes, 0);
        assert_eq!(sft.host_rollout_bytes, 0);

        let rollout = estimate(
            &model,
            &rollout_training(1024),
            Trainable::adapter(&lora),
            &rollout_workload(),
            Calibration::default(),
        );
        assert!(rollout.generation_kv_bytes > 0);
        assert!(rollout.host_rollout_bytes > 0);
    }

    #[test]
    fn fast_generation_halves_the_sampling_cache() {
        let model = tiny_model();
        let lora = LoraConfig::auto(8, 16.0);
        let workload = rollout_workload();
        let plain = estimate(
            &model,
            &TrainConfig {
                fast_generation_context: false,
                ..rollout_training(1024)
            },
            Trainable::adapter(&lora),
            &workload,
            Calibration::default(),
        );
        let fast = estimate(
            &model,
            &rollout_training(1024),
            Trainable::adapter(&lora),
            &workload,
            Calibration::default(),
        );
        assert_eq!(fast.generation_kv_bytes, plain.generation_kv_bytes / 2);
    }

    #[test]
    fn trainable_parameters_follow_the_target_set_and_the_rank() {
        let model = tiny_model();
        let qv = trainable_parameters(&model, Some(&LoraConfig::qv(8, 16.0)));
        let auto = trainable_parameters(&model, Some(&LoraConfig::auto(8, 16.0)));
        assert!(qv < auto, "QV must be cheaper than every projection");
        let rank16 = trainable_parameters(&model, Some(&LoraConfig::qv(16, 16.0)));
        assert_eq!(rank16, 2 * qv);

        let mut patterns = LoraConfig::auto(8, 16.0);
        patterns.targets = TargetSet::Patterns(vec![
            "blk.*.attn_q.weight".to_string(),
            "blk.*.attn_v.weight".to_string(),
        ]);
        assert_eq!(trainable_parameters(&model, Some(&patterns)), qv);
    }

    #[test]
    fn the_feed_forward_width_is_solved_from_the_parameter_count() {
        // A model whose parameters are exactly embeddings + attention + a
        // 4× gated FFN must read back that width.
        let mut model = tiny_model();
        model.n_expert = 0;
        let n_ff = 4 * model.n_embd as u64;
        model.n_params = model.n_vocab as u64 * model.n_embd as u64
            + model.n_layer as u64 * attention_parameters(&model)
            + model.n_layer as u64 * 3 * model.n_embd as u64 * n_ff;
        model.tied_embeddings = true;
        assert_eq!(feed_forward_width(&model), n_ff);

        // Nonsense arithmetic falls back rather than producing a silly width.
        model.n_params = 1;
        assert_eq!(feed_forward_width(&model), 4 * model.n_embd as u64);
    }

    #[test]
    fn dominant_posts_are_ordered_and_stable() {
        let estimate = MemoryEstimate {
            model_weight_bytes: 10,
            optimizer_kv_bytes: 30,
            logits_bytes: 20,
            ..Default::default()
        };
        assert_eq!(
            estimate.dominant_posts(),
            vec![
                ("optimizer_kv_bytes", 30),
                ("logits_bytes", 20),
                ("model_weight_bytes", 10)
            ]
        );
    }

    #[test]
    fn a_measured_report_maps_onto_the_same_posts() {
        let report = MemoryReport {
            model_weight_bytes: 7,
            optimizer_kv_bytes: 9,
            optimizer_compute_bytes: 11,
            device_memory_samples: 1,
            ..Default::default()
        };
        let measured = MemoryEstimate::from(&report);
        assert_eq!(measured.model_weight_bytes, 7);
        assert_eq!(measured.optimizer_kv_bytes, 9);
        assert_eq!(measured.optimizer_compute_bytes, 11);
        // Not measured is not zero-because-nil; the resolver must not read the
        // split as "the runtime says the activations are free".
        assert_eq!(measured.activation_bytes, 0);
    }

    #[test]
    fn the_device_peak_counts_both_compute_reserves_once_each() {
        let estimate = MemoryEstimate {
            model_weight_bytes: 1_000,
            optimizer_kv_bytes: 100,
            generation_kv_bytes: 200,
            optimizer_compute_bytes: 400,
            generation_compute_bytes: 700,
            dequant_scratch_bytes: 50,
            trainable_parameter_bytes: 1,
            trainable_gradient_bytes: 2,
            optimizer_state_bytes: 4,
            host_dataset_bytes: 9,
            ..Default::default()
        };
        let resources = estimate.resources();
        // The runtime holds both contexts at once, so the peak is the sum. A
        // `max` here would hide 400 bytes and admit a run the device refuses.
        assert_eq!(resources.device_peak_bytes, estimate.device_bytes());
        assert_eq!(
            resources.device_peak_bytes,
            1_000 + 100 + 200 + 400 + 700 + 50 + 7
        );
        // The scratch appears in both phase breakdowns because either graph may
        // use it, and it is one buffer: counted once in the peak.
        assert_eq!(resources.optimizer_transient.bytes, 450);
        assert_eq!(resources.generation_transient.bytes, 750);
        assert_eq!(resources.host_peak_bytes, estimate.host_bytes());
        assert!(resources.lower_bound_device_bytes <= resources.device_peak_bytes);
        assert_eq!(
            resources.upper_bound_device_bytes,
            resources.device_peak_bytes
        );
    }
}
