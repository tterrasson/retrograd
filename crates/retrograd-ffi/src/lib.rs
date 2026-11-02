//! Raw bindings to the C++ runtime (`runtime/include/retro_lora_train.h`).
//!
//! Every `#[repr(C)]` struct here mirrors the published header, which
//! `c_struct_layouts_match_the_published_header` checks field by field.
//! `build.rs` compiles the vendored llama.cpp fork and the runtime for the
//! backends selected by Cargo features (`metal`, `vulkan`, `cuda`, `platform-gpu`).
//! The safe API over these calls is `retrograd-engine`.

use std::ffi::{CStr, c_char, c_double, c_float, c_int, c_void};
use std::mem::size_of;

/// Commit of the vendored `llama.cpp` fork this crate was built against, set
/// by `build.rs` (`cargo:rustc-env=RETRO_LLAMA_CPP_COMMIT`).
pub const LLAMA_CPP_COMMIT: &str = env!("RETRO_LLAMA_CPP_COMMIT");

/// The chat template is valid but llama.cpp cannot derive a parser for its
/// assistant tool-call format. Unlike `-1`, callers may handle this status by
/// switching both rendering and parsing to the prompt-described convention.
pub const RETRO_CHAT_PARSER_UNAVAILABLE: c_int = -3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroLoraConfig {
    pub rank: u32,
    pub alpha: c_float,
    pub dropout: c_float,
    pub seed: u32,
    pub target_patterns: *const *const c_char,
    pub n_target_patterns: usize,
    pub dtype: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroTrainConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub generation_concurrency: u32,
    pub fast_generation_context: bool,
    pub kv_dtype: c_int,
    pub threads: u32,
    pub epochs: u32,
    pub learning_rate: c_float,
    pub weight_decay: c_float,
    pub max_grad_norm: c_float,
    pub lr_scheduler: i32,
    pub warmup_steps: u64,
    pub verbose: bool,
    pub device: i32,
    pub chunked_cross_entropy: bool,
    pub chunked_ce_tiles: u32,
    pub chunked_ce_seq_chunk: u32,
    pub chunked_ce_offload_logsoftmax: bool,
    pub gradient_checkpointing: bool,
    pub checkpoint_every_n_layers: u32,
    pub checkpoint_dtype: c_int,
    pub require_gpu_resident: bool,
    pub generation_batch: u32,
    pub shuffle_dataset: bool,
    pub optimizer: i32,
    pub shuffle_seed: u64,
    pub trainable: i32,
    /// Muon, read only when `optimizer` names it. Zero in any of these selects
    /// the runtime's frozen v1 default, so a caller that fills none of them
    /// gets the declared layout rather than zeros.
    pub muon_momentum: c_float,
    pub muon_ns_epsilon: c_float,
    pub muon_fallback_learning_rate: c_float,
    pub muon_ns_steps: u32,
    pub muon_nesterov: bool,
    /// Gefen, read only when `optimizer` names it. `gefen_variant` is one of
    /// `RETRO_GEFEN_*`.
    pub gefen_variant: i32,
    pub gefen_block_size: u32,
    pub gefen_beta1: c_float,
    pub gefen_beta2: c_float,
    pub gefen_eps: c_float,
    /// One of `RETRO_MASTER_WEIGHTS_*`: whether a half-precision parameter is
    /// trained through an F32 master copy.
    pub master_weights: c_int,
    /// One of `RETRO_GENERATION_KV_DTYPE_*`: the dedicated generation context's
    /// KV-cache storage.
    pub generation_kv_dtype: c_int,
}

/// Which fixed-block state a Gefen run keeps, mirroring `retro_gefen_variant`.
/// It selects a slot table rather than scaling an update, so it moves the
/// layout version a checkpoint records.
pub const RETRO_GEFEN_SHARED_V: i32 = 0;
pub const RETRO_GEFEN_QUANTIZED_M: i32 = 1;

/// Implementation choice for [`retro_probe_op_run_ex`].
pub const RETRO_KERNEL_IMPL_AUTO: i32 = 0;
pub const RETRO_KERNEL_IMPL_NATIVE: i32 = 1;
pub const RETRO_KERNEL_IMPL_RIR: i32 = 2;

/// What [`retro_probe_op_run_ex`] actually did. `struct_size` must be set to
/// `size_of::<RetroKernelRunInfo>()` before the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroKernelRunInfo {
    pub struct_size: u32,
    pub requested_impl: i32,
    pub executed_impl: i32,
    pub reject_reason: i32,
    pub variant: [c_char; 64],
}

impl Default for RetroKernelRunInfo {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            requested_impl: RETRO_KERNEL_IMPL_AUTO,
            executed_impl: RETRO_KERNEL_IMPL_NATIVE,
            reject_reason: 0,
            variant: [0; 64],
        }
    }
}

/// Mirrors `retro_grouped_gradient_probe` in `retro_lora_train.h`: the gradient
/// of the grouped softmax cross-entropy of one F32 tensor against finite
/// differences. Set `struct_size` before the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroGroupedGradientProbe {
    pub struct_size: u32,
    pub model_path: *const c_char,
    pub use_gpu: i32,
    pub flash_attn: i32,
    pub param_name: *const c_char,
    pub tokens: *const i32,
    pub positions: *const i32,
    pub seq_ids: *const i32,
    pub outputs: *const i8,
    pub n_tokens: i32,
    pub n_groups: u32,
    pub n_opt_max: u32,
    pub rows: *const i32,
    pub columns: *const i32,
    pub targets: *const c_float,
    pub inv_temperature: c_float,
    pub elements: *const i64,
    pub n_elements: i32,
    pub step: c_float,
    pub n_missing_backward: i32,
    pub loss_step: c_double,
    pub loss_decode: c_double,
    pub grad_analytic: *mut c_float,
    pub grad_numeric: *mut c_double,
}

/// The two fixed-block Gefen ops driven directly, with no model and no
/// training graph. Every pointer is host memory in the caller's layout; the
/// probe uploads it, runs `n_steps` full updates on the selected device and
/// reads the state back into the same buffers.
///
/// Which pointers are required depends on the variant: `shared_v` keeps
/// `moment` and no `indices`/`scales`/`codebook`, `quantized_m` the reverse.
/// Set `struct_size` before the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroGefenProbe {
    pub struct_size: u32,
    pub use_gpu: i32,
    pub variant: i32,
    pub block_size: i32,
    pub n_elements: i64,
    pub n_blocks: i64,
    pub levels: i32,
    pub n_steps: i32,
    pub weights: *mut c_float,
    pub grad: *const c_float,
    pub moment: *mut c_float,
    pub indices: *mut u8,
    pub scales: *mut c_float,
    pub v: *mut c_float,
    pub codebook: *const c_float,
    pub pars: *const c_float,
    pub stats: *mut c_float,
}

impl Default for RetroGefenProbe {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            use_gpu: 0,
            variant: RETRO_GEFEN_SHARED_V,
            block_size: 0,
            n_elements: 0,
            n_blocks: 0,
            levels: 0,
            n_steps: 1,
            weights: std::ptr::null_mut(),
            grad: std::ptr::null(),
            moment: std::ptr::null_mut(),
            indices: std::ptr::null_mut(),
            scales: std::ptr::null_mut(),
            v: std::ptr::null_mut(),
            codebook: std::ptr::null(),
            pars: std::ptr::null(),
            stats: std::ptr::null_mut(),
        }
    }
}

/// Process-wide RIR dispatch counters. Set `struct_size` before the call.
/// `reject_by_reason` has one bucket per `retro_kernel_reject` value, including
/// `MATCHED`; its size must stay synchronized with the C and ggml definitions.
pub const KERNEL_REJECT_COUNT: usize = 14;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroRirCounters {
    pub struct_size: u32,
    pub mode: i32,
    pub ops_seen: u64,
    pub rir_eligible: u64,
    pub rir_dispatched: u64,
    pub native_dispatched: u64,
    pub fallback_contract: u64,
    pub fallback_feature: u64,
    pub fallback_pipeline: u64,
    pub reject_by_reason: [u64; KERNEL_REJECT_COUNT],
}

impl Default for RetroRirCounters {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            mode: 0,
            ops_seen: 0,
            rir_eligible: 0,
            rir_dispatched: 0,
            native_dispatched: 0,
            fallback_contract: 0,
            fallback_feature: 0,
            fallback_pipeline: 0,
            reject_by_reason: [0; KERNEL_REJECT_COUNT],
        }
    }
}

/// Process-wide runtime policy, applied before any backend context exists.
/// Versioned by `struct_size` rather than by widening
/// [`RetroTrainConfig`], whose size is part of the existing ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroRuntimeConfig {
    pub struct_size: u32,
    /// 0 off, 1 observe, 2 prefer, 3 require. Takes precedence over
    /// `RETRO_RIR_MODE`.
    pub rir_mode: i32,
}

impl Default for RetroRuntimeConfig {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            // `off`: RIR stays opt-in, so a caller that does not ask for it
            // cannot get it by upgrading.
            rir_mode: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroTrainMetrics {
    pub epoch: u32,
    pub epoch_complete: bool,
    pub global_step: u64,
    pub train_loss: c_float,
    pub eval_loss: c_float,
    pub tokens_per_second: c_float,
    pub learning_rate: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroOptimizerTiming {
    pub graph_build_seconds: c_double,
    pub allocation_seconds: c_double,
    pub execution_seconds: c_double,
}

/// Device-memory counters for the optimizer path, sampled by the runtime inside
/// each step. `device_*` come from the backend's device-wide budget (other
/// processes included); `scratch_*` are this process's backend-owned scratch,
/// the CUDA pool and Vulkan `prealloc_*` buffers, which belong to no
/// `ggml_backend_buffer` and are therefore invisible to the byte breakdown.
/// `n_samples == 0` means unavailable, not "measured zero".
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroOptimizerMemory {
    pub device_used_bytes: u64,
    pub device_total_bytes: u64,
    pub device_peak_used_bytes: u64,
    pub scratch_bytes: u64,
    pub scratch_peak_bytes: u64,
    pub n_samples: u64,
}

/// Capacity of the fixed-size name fields in [`RetroModelInfo`], mirroring
/// `RETRO_MODEL_INFO_NAME_MAX`.
pub const MODEL_INFO_NAME_MAX: usize = 64;

/// Geometry of a GGUF model, read without creating a training context. See
/// `retro_model_info` in the C header for the meaning of each field; in
/// particular `n_embd_k_gqa`/`n_embd_v_gqa` are the per-layer KV widths
/// maximised across layers, which is what a KV-cache estimate needs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroModelInfo {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    pub n_embd_k_gqa: u32,
    pub n_embd_v_gqa: u32,
    pub n_embd_r: u32,
    pub n_embd_s: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_params: u64,
    pub model_size_bytes: u64,
    pub file_size_bytes: u64,
    pub dominant_weight_bytes: u64,
    pub tied_embeddings: bool,
    pub is_recurrent: bool,
    pub has_encoder: bool,
    pub architecture: [c_char; MODEL_INFO_NAME_MAX],
    pub dominant_weight_type: [c_char; MODEL_INFO_NAME_MAX],
}

/// Schema version of the tensor inventory, mirroring
/// `RETRO_TENSOR_INVENTORY_VERSION`.
pub const TENSOR_INVENTORY_VERSION: u32 = 1;

/// Mirrors `RETRO_TENSOR_NAME_MAX`.
pub const TENSOR_NAME_MAX: usize = 128;

/// One tensor as the loader sees it. The per-tensor counterpart of
/// [`RetroModelInfo`], which carries only aggregate geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroTensorDesc {
    pub name: [c_char; TENSOR_NAME_MAX],
    pub ne: [i64; 4],
    pub n_elements: u64,
    pub n_bytes: u64,
    pub storage_id: u64,
    pub type_name: [c_char; MODEL_INFO_NAME_MAX],
}

impl Default for RetroTensorDesc {
    fn default() -> Self {
        Self {
            name: [0; TENSOR_NAME_MAX],
            ne: [0; 4],
            n_elements: 0,
            n_bytes: 0,
            storage_id: 0,
            type_name: [0; MODEL_INFO_NAME_MAX],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroModelCapabilities {
    pub struct_size: u32,
    pub shared_prefix_packed_training: bool,
    pub fused_sparse_cross_entropy: bool,
    pub differentiable_flash_attention: bool,
}

pub const PREFLIGHT_FINGERPRINT_MAX: usize = 65;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroPreflightSummary {
    pub struct_size: u32,
    pub missing_gradient_rules: i32,
    pub active_device_fallback_nodes: u64,
    pub graph_fingerprint: [c_char; PREFLIGHT_FINGERPRINT_MAX],
}

impl Default for RetroPreflightSummary {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            missing_gradient_rules: 0,
            active_device_fallback_nodes: 0,
            graph_fingerprint: [0; PREFLIGHT_FINGERPRINT_MAX],
        }
    }
}

impl Default for RetroModelCapabilities {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            shared_prefix_packed_training: false,
            fused_sparse_cross_entropy: false,
            differentiable_flash_attention: false,
        }
    }
}

impl Default for RetroModelInfo {
    fn default() -> Self {
        // The struct is a plain C value with two fixed char arrays, so a zeroed
        // instance is both valid and the natural "nothing read yet" state.
        unsafe { std::mem::zeroed() }
    }
}

/// Measured host<->device transfer rates, pinned and pageable. `pinned_is_pageable`
/// says the backend has no dedicated host buffer type, so the two pairs are the
/// same measurement rather than a comparison.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroTransferRates {
    pub bytes_per_transfer: u64,
    pub iterations: u64,
    pub pinned_h2d_bytes_per_second: f64,
    pub pinned_d2h_bytes_per_second: f64,
    pub pageable_h2d_bytes_per_second: f64,
    pub pageable_d2h_bytes_per_second: f64,
    pub pinned_is_pageable: bool,
    pub device_buffer_is_host: bool,
}

/// Structured counterpart of the byte fields in the textual backend report.
/// Both are rendered from one computation in the runtime, so they cannot
/// disagree. `device_memory_samples == 0` means the measured fields are
/// unavailable, not measured zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroMemoryReport {
    pub model_weight_bytes: u64,
    pub optimizer_kv_bytes: u64,
    pub optimizer_compute_bytes: u64,
    pub generation_kv_bytes: u64,
    pub generation_compute_bytes: u64,
    pub has_generation_context: bool,
    pub trainable_parameter_bytes: u64,
    pub trainable_gradient_bytes: u64,
    pub optimizer_state_bytes: u64,
    pub trainable_parameters_are_model_subset: bool,
    pub adapter_on_host: bool,
    pub base_trainable_on_host: bool,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub device_total_bytes: u64,
    pub device_used_bytes: u64,
    pub device_peak_used_bytes: u64,
    pub backend_scratch_bytes: u64,
    pub backend_scratch_peak_bytes: u64,
    pub device_memory_samples: u64,
    pub checkpoint_count: u64,
    pub checkpoint_retained_bytes: u64,
    pub checkpoint_live_peak_bytes: u64,
    pub checkpoint_live_peak_count: u64,
    pub checkpoint_long_lived_bytes: u64,
    pub checkpoint_long_lived_count: u64,
    pub checkpoint_graph_nodes: u64,
    pub checkpoint_max_span_nodes: u64,
    pub checkpoint_total_span_nodes: u64,
}

/// Monotonic counters for the shared-prefix behavior scorer. `prefix_decodes`
/// greater than `calls` indicates that a branch could not be evicted and the
/// shared prefix had to be rebuilt. `device_logprob_positions` less than
/// `scored_positions` indicates host-side log-probability reduction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroScoringStats {
    pub calls: u64,
    pub prefix_decodes: u64,
    pub prefix_reprefills: u64,
    pub branch_evictions_refused: u64,
    pub scored_positions: u64,
    pub device_logprob_positions: u64,
}

/// Monotonic counters of the generation context's prefix reuse. `prompt_tokens`
/// is what the caller asked to be resident before sampling; `prefilled_tokens`
/// is what had to be decoded to get there.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroGenerationStats {
    pub calls: u64,
    pub sequences: u64,
    pub prompt_tokens: u64,
    pub prefilled_tokens: u64,
    pub reused_tokens: u64,
    pub hits: u64,
    pub evictions: u64,
}

/// Wall-clock accounting of the GPU duty-cycle limiter, in seconds. The pair
/// `requested_fraction` / `active` distinguishes "not requested" from
/// "requested on a CPU backend", which neither field alone can.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroDutyCycleStats {
    pub requested_fraction: c_float,
    pub active: bool,
    pub compute_seconds: c_double,
    pub idle_seconds: c_double,
    pub wall_seconds: c_double,
}

/// One step of a [`retro_probe_duty_cycle`] script. `micros` is simulated
/// microseconds, ignored by the kinds that do not span time.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroDutyCycleEvent {
    pub kind: i32,
    pub micros: u64,
}

/// `micros` of synchronized GPU work, accounted at its completion.
pub const RETRO_DUTY_CYCLE_EVENT_WORK: i32 = 0;
/// One idle boundary: repay what is owed, up to the internal sleep cap.
pub const RETRO_DUTY_CYCLE_EVENT_IDLE: i32 = 1;
/// `micros` of host time the limiter did not choose, between two accounted
/// windows - a progress callback, a run-control pause.
pub const RETRO_DUTY_CYCLE_EVENT_HOST: i32 = 2;
/// An explicit window reset.
pub const RETRO_DUTY_CYCLE_EVENT_RESET: i32 = 3;

/// Outcome of a [`retro_probe_duty_cycle`] replay. `clock_reads` and
/// `sleep_calls` count the fake clock's use: a disabled replay must leave both
/// at zero, which is the zero-overhead property and is not observable from the
/// seconds alone.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroDutyCycleProbe {
    pub stats: RetroDutyCycleStats,
    pub clock_reads: u64,
    pub sleep_calls: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroEvalMetrics {
    pub negative_log_likelihood: c_double,
    pub supervised_tokens: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroSftDataset {
    pub tokens: *const i32,
    pub labels: *const i32,
    pub n_rows: usize,
    pub n_ctx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroSamplingParams {
    pub temperature: c_float,
    pub top_p: c_float,
    pub max_new_tokens: u32,
    pub seed: u32,
}

#[repr(C)]
pub struct RetroTokenSuffixSequence {
    pub tokens: *const i32,
    pub n_tokens: usize,
    pub n_prompt: usize,
}

#[repr(C)]
pub struct RetroGenerationSequence {
    pub prompt_tokens: *const i32,
    pub n_prompt: usize,
    pub sampling: RetroSamplingParams,
}

/// Mirror of `retro_generation_sequence_v2`: one independently sampled prompt
/// with extended (v2) sampling, grammar handle included.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RetroGenerationSequenceV2 {
    pub prompt_tokens: *const i32,
    pub n_prompt: usize,
    pub sampling: RetroSamplingParamsV2,
}

/// Mirror of `RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY`: save/load only the partial
/// state (recurrent state, plus the SWA KV of a sliding-window layer).
pub const RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY: u32 = 1;
/// Mirror of `RETRO_STATE_SEQ_FLAGS_ON_DEVICE`.
pub const RETRO_STATE_SEQ_FLAGS_ON_DEVICE: u32 = 2;

/// Mirror of `retro_step_row`, versioned by `struct_size`: one row of a step
/// plan in `retro_*_step`. See the published header for the full contract
/// (residency, logits validity, partial mutation on failure).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroStepRow {
    /// Filled by the caller: the evolution key. Must be `size_of::<Self>()`.
    pub struct_size: u32,
    /// Per-request state id (sampler RNG, reasoning-budget machine); >= 0 on
    /// rows that join a request, required on sampling rows.
    pub request: i32,
    /// Sequence id in the context's KV pool.
    pub slot: i32,
    pub tokens: *const i32,
    pub n: u32,
    /// Absolute position of `tokens[0]`.
    pub pos0: i32,
    /// Request logits on the last token and sample it.
    pub sample: bool,
    /// Read when `sample` is set (`seed` seeds the request's RNG once).
    pub sampling: RetroSamplingParams,
    pub n_top_logprobs: u32,
    /// The adapter this row decodes under: 0 for the context's own adapter
    /// set, [`RETRO_STEP_ADAPTER_BASE`] for the base model alone, or a
    /// per-row adapter id. Rows of one slot in one step must agree.
    pub adapter: i32,
}

/// Mirror of `RETRO_STEP_ADAPTER_BASE`: a step row decoding under the base
/// model alone, whatever adapter the context applies to its other rows.
pub const RETRO_STEP_ADAPTER_BASE: i32 = -1;

/// Mirror of `retro_adapter_rows_info`: what the per-row adapters of a
/// context cost (F32 stacks, zero-padded to the largest rank).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetroAdapterRowsInfo {
    /// Filled by the runtime.
    pub struct_size: u32,
    pub n_adapters: u32,
    /// The rank every stacked adapter is padded to.
    pub max_rank: u32,
    /// Bytes of the stacks, padding included.
    pub stacked_bytes: u64,
    /// What the same adapters would take unpadded, in F32.
    pub unpadded_bytes: u64,
}

/// Mirror of `retro_step_out_row`: what one step produced for one row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroStepOutRow {
    /// Filled by the runtime.
    pub struct_size: u32,
    /// The emitted token; valid when `sampled` is set.
    pub token: i32,
    /// Temperature-1 policy logprob of `token`, as the batched generators
    /// report it.
    pub logprob: c_float,
    /// Top entries written at the row's slice of the top output buffers.
    pub n_top: u32,
    /// The reasoning budget forced `token`.
    pub forced: u8,
    /// A sample was produced for this row.
    pub sampled: u8,
}

/// Mirror of `retro_grammar` (opaque): an owned GBNF string. See [`Grammar`].
#[repr(C)]
pub struct RetroGrammar {
    _private: [u8; 0],
}

/// Mirror of `RETRO_TOOL_GRAMMAR_UNAVAILABLE`: the chat template cannot
/// express the requested tool-call grammar mode.
pub const RETRO_TOOL_GRAMMAR_UNAVAILABLE: c_int = -2;

/// Mirror of `RETRO_EMBED_NO_POOLING`: the model declares no pooling (a causal
/// chat model), so `retro_engine_embed` refuses rather than inventing an
/// embedding.
pub const RETRO_EMBED_NO_POOLING: c_int = -3;

/// Mirror of `RETRO_EMBED_INVALID_INPUT`: invalid tokens or an input that
/// exceeds the embedding context or micro-batch capacity.
pub const RETRO_EMBED_INVALID_INPUT: c_int = -4;

/// Mirror of the `RETRO_DECIDE_*` readouts of `retro_engine_decide`:
/// `embeddings[column]` at each read position.
pub const RETRO_DECIDE_MARKER_COLUMN: i32 = 0;
/// The logits of each read token id, at the last position.
pub const RETRO_DECIDE_LABEL_LOGITS: i32 = 1;
/// `q.k / sqrt(n/2)`: `q` the first half of the embeddings at `pointer`, `k`
/// the second half at each read position.
pub const RETRO_DECIDE_POINTER: i32 = 2;
/// Mirror of `RETRO_DECIDE_INVALID_INPUT`: a prompt the context cannot
/// evaluate, or a read out of range.
pub const RETRO_DECIDE_INVALID_INPUT: c_int = -4;

/// Mirror of `retro_decision_prompt`: the tokens of one question as the model
/// reads them, and where its scores are read.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroDecisionPrompt {
    pub tokens: *const i32,
    pub n_tokens: usize,
    /// Positions, or label token ids for `RETRO_DECIDE_LABEL_LOGITS`.
    pub reads: *const i32,
    pub n_reads: usize,
    /// `RETRO_DECIDE_MARKER_COLUMN` only.
    pub column: i32,
    /// `RETRO_DECIDE_POINTER` only: the position of the query.
    pub pointer: i32,
}

/// Mirror of `retro_grouped_batch`: one physical step of the grouped softmax
/// cross-entropy, questions as sequences and the groups of scores the loss
/// reads from the embeddings output.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroGroupedBatch {
    pub tokens: *const i32,
    /// Per sequence.
    pub positions: *const i32,
    pub seq_ids: *const i32,
    /// 1 exactly where a group reads the token.
    pub outputs: *const i8,
    pub n_tokens: usize,
    pub n_groups: u32,
    pub n_opt_max: u32,
    /// `[n_groups * n_opt_max]`, batch token index or -1.
    pub rows: *const i32,
    /// `[n_groups]`.
    pub columns: *const i32,
    /// `[n_groups * n_opt_max]`, each group summing to 1.
    pub targets: *const f32,
    pub inv_temperature: f32,
    pub accumulation_steps: u32,
}

/// Mirror of `retro_sampling_params_v2`: extended sampling parameters for v2
/// step rows. See the published header for the field contracts and the pinned
/// transformation order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroSamplingParamsV2 {
    /// Filled by the caller: the evolution key. Must be `size_of::<Self>()`.
    pub struct_size: u32,
    /// Finite, > 0.
    pub temperature: c_float,
    /// In (0, 1].
    pub top_p: c_float,
    /// > 0; the caller's stop criterion, not read by the runtime.
    pub max_new_tokens: u32,
    /// Seeds the request's RNG once, at its first sampling row.
    pub seed: u32,
    /// <= 0: disabled.
    pub top_k: i32,
    /// In [0, 1]; <= 0: disabled.
    pub min_p: c_float,
    /// > 0 finite; 1.0: disabled.
    pub repeat_penalty: c_float,
    /// Finite; 0.0: disabled.
    pub frequency_penalty: c_float,
    /// Finite; 0.0: disabled.
    pub presence_penalty: c_float,
    /// Last N accepted tokens the penalties see; 0 = the whole history.
    pub penalty_window: u32,
    /// 0 = none.
    pub n_logit_bias: u32,
    /// `n_logit_bias` entries, each a valid token id.
    pub logit_bias_ids: *const i32,
    /// `n_logit_bias` finite values.
    pub logit_bias_values: *const c_float,
    /// NULL: unconstrained.
    pub grammar: *const RetroGrammar,
}

/// Mirror of `retro_step_row_v2`, versioned by `struct_size`: one row of a v2
/// step plan, same row contracts as [`RetroStepRow`] plus extended sampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroStepRowV2 {
    /// Filled by the caller: the evolution key. Must be `size_of::<Self>()`.
    pub struct_size: u32,
    /// Per-request state id; >= 0 required on sampling rows.
    pub request: i32,
    /// Sequence id in the context's KV pool.
    pub slot: i32,
    pub tokens: *const i32,
    pub n: u32,
    /// Absolute position of `tokens[0]`.
    pub pos0: i32,
    /// Request logits on the last token and sample it.
    pub sample: bool,
    /// Read when `sample` is set.
    pub sampling: RetroSamplingParamsV2,
    pub n_top_logprobs: u32,
    /// Also fill `sampled_logprob`.
    pub want_logprob: u8,
    /// Same contract as [`RetroStepRow::adapter`].
    pub adapter: i32,
}

/// Mirror of `retro_step_out_row_v2`: what one v2 step produced for one row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroStepOutRowV2 {
    /// Filled by the runtime.
    pub struct_size: u32,
    /// The emitted token; valid when `sampled` is set.
    pub token: i32,
    /// Temperature-1 policy logprob of `token` (unchanged contract).
    pub logprob: c_float,
    /// Top entries written at the row's slice of the top output buffers.
    pub n_top: u32,
    /// The reasoning budget forced `token`.
    pub forced: u8,
    /// A sample was produced for this row.
    pub sampled: u8,
    /// Post-transformation logprob of `token`; 0 when `want_logprob` was 0.
    pub sampled_logprob: c_float,
}

/// Safe handle to a `retro_grammar`: an owned GBNF string. Dropping it frees
/// the handle; the runtime copies the GBNF into the per-request sampler state
/// at the request's first sampling row, so a handle may be dropped as soon as
/// the step call that carried it has returned.
pub struct Grammar {
    raw: *mut RetroGrammar,
}

impl Grammar {
    /// The raw handle, for embedding in a `RetroSamplingParamsV2` row for the
    /// duration of one FFI call (see the struct's safety note above).
    pub fn as_ptr(&self) -> *mut RetroGrammar {
        self.raw
    }

    /// Converts a JSON schema to a GBNF grammar (`retro_grammar_from_json_schema`).
    pub fn from_json_schema(schema: &str) -> Result<Grammar, String> {
        let schema = std::ffi::CString::new(schema).map_err(|err| err.to_string())?;
        let raw = unsafe { retro_grammar_from_json_schema(schema.as_ptr()) };
        if raw.is_null() {
            return Err(last_error());
        }
        Ok(Grammar { raw })
    }

    /// Takes the GBNF text as-is (`retro_grammar_from_gbnf`).
    pub fn from_gbnf(gbnf: &str) -> Result<Grammar, String> {
        let gbnf = std::ffi::CString::new(gbnf).map_err(|err| err.to_string())?;
        let raw = unsafe { retro_grammar_from_gbnf(gbnf.as_ptr()) };
        if raw.is_null() {
            return Err(last_error());
        }
        Ok(Grammar { raw })
    }

    /// The built-in any-JSON grammar (`retro_grammar_json_object`).
    pub fn json_object() -> Grammar {
        let raw = unsafe { retro_grammar_json_object() };
        assert!(
            !raw.is_null(),
            "retro_grammar_json_object failed: {}",
            last_error()
        );
        Grammar { raw }
    }

    /// The owned GBNF text (`retro_grammar_gbnf`).
    pub fn gbnf(&self) -> String {
        unsafe {
            let ptr = retro_grammar_gbnf(self.raw);
            if ptr.is_null() {
                return String::new();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe { retro_grammar_free(self.raw) }
    }
}

/// Mirror of `retro_engine_memory_info`: KV-pool occupancy of a generation
/// context.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroEngineMemoryInfo {
    /// Filled by the runtime.
    pub struct_size: u32,
    /// Sequence slots of the context (n_seq_max).
    pub n_slots: u32,
    /// KV cells of the pool.
    pub cells_total: u64,
    /// Occupied cells: sum of the resident position spans of non-empty slots.
    pub cells_used: u64,
    /// Bytes of the attention KV buffers.
    pub kv_bytes: u64,
    /// Partial state bytes summed over all slots (recurrent state, plus SWA KV
    /// where the model has a sliding window).
    pub recurrent_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// `n_topk` sparse targets per position instead of one.
/// `labels` and `weights` then hold `n_rows * n_ctx * n_topk` values, entry `j`
/// of position `p` of row `r` at `((r * n_ctx + p) * n_topk + j)`. `0` and `1`
/// are the same one-target layout and the same run, bit for bit.
pub struct RetroWeightedDataset {
    pub tokens: *const i32,
    pub labels: *const i32,
    pub weights: *const c_float,
    pub n_rows: usize,
    pub n_ctx: u32,
    pub n_topk: u32,
}

#[repr(C)]
pub struct RetroPackedSequenceBatch {
    pub tokens: *const i32,
    pub labels: *const i32,
    pub weights: *const c_float,
    pub positions: *const i32,
    pub seq_offsets: *const usize,
    pub seq_ids: *const i32,
    pub n_tokens: usize,
    pub n_seq_ids: usize,
    pub n_sequences: u32,
    /// Same layout and meaning as [`RetroWeightedDataset::n_topk`].
    pub n_topk: u32,
}

/// Scope of an optimizer state slot, mirroring `RETRO_SLOT_SCOPE_*`.
///
/// Two namespaces rather than one list: a parameter slot is owned by the
/// tensor it updates, a shared slot by whatever the optimizer declares, and a
/// shared slot exists once per owner rather than once per parameter.
pub const SLOT_SCOPE_PARAMETER: i32 = 0;
pub const SLOT_SCOPE_SHARED: i32 = 1;

/// One persistent optimizer tensor: what owns it, what it is called inside the
/// optimizer's layout, and the exact payload a checkpoint round-trips.
///
/// Deliberately not a pair of momenta. AdamW happens to contribute `m` and `v`
/// per parameter; SGD contributes nothing and keeps a step counter anyway, and
/// an optimizer with block-shaped state contributes rows of other dtypes. Only
/// the enumeration is contract.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroOptimizerSlot {
    pub owner: [c_char; TENSOR_NAME_MAX],
    pub slot: [c_char; MODEL_INFO_NAME_MAX],
    pub type_name: [c_char; MODEL_INFO_NAME_MAX],
    pub ne: [i64; 4],
    pub n_elements: u64,
    pub n_bytes: u64,
}

impl Default for RetroOptimizerSlot {
    fn default() -> Self {
        Self {
            owner: [0; TENSOR_NAME_MAX],
            slot: [0; MODEL_INFO_NAME_MAX],
            type_name: [0; MODEL_INFO_NAME_MAX],
            ne: [0; 4],
            n_elements: 0,
            n_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroOptimizerState {
    pub iter: i64,
    pub has_persistent_state: bool,
    /// Whether the optimizer graph exists. Separate from
    /// `has_persistent_state`, which is false both for a cold optimizer and
    /// for one that keeps no state.
    pub graph_ready: bool,
    pub optimizer: i32,
    pub learning_rate: c_float,
    pub weight_decay: c_float,
    pub max_grad_norm: c_float,
    pub scheduler_step: u64,
    pub scheduler_total_steps: u64,
    pub last_learning_rate: c_float,
    /// AdamW's coefficients as the update step reads them. Reported, not
    /// configured: a checkpoint records the values that ran.
    pub adamw_beta1: c_float,
    pub adamw_beta2: c_float,
    pub adamw_eps: c_float,
    /// One of `RETRO_GEFEN_*`, and meaningless for any other optimizer. The
    /// name alone does not say which slot table was allocated, so the variant
    /// travels beside it.
    pub gefen_variant: i32,
    /// The chosen optimizer's own coefficients, reported the same way the
    /// AdamW three are: a checkpoint records the values that ran.
    pub muon_momentum: c_float,
    pub muon_ns_epsilon: c_float,
    pub muon_fallback_learning_rate: c_float,
    pub muon_ns_steps: u32,
    pub muon_nesterov: bool,
    pub gefen_beta1: c_float,
    pub gefen_beta2: c_float,
    pub gefen_eps: c_float,
    pub gefen_block_size: u32,
}

#[repr(C)]
pub struct RetroTrainer {
    _private: [u8; 0],
}

/// The inference engine handle: a mapped model with one forward-only context,
/// no optimizer. Opaque, like [`RetroTrainer`].
#[repr(C)]
pub struct RetroEngine {
    _private: [u8; 0],
}

/// One of `RETRO_ENGINE_KV_DTYPE_*`: the engine context's KV-cache storage.
pub const RETRO_ENGINE_KV_DTYPE_F32: c_int = 0;
pub const RETRO_ENGINE_KV_DTYPE_F16: c_int = 1;
pub const RETRO_ENGINE_KV_DTYPE_Q8_0: c_int = 2;
pub const RETRO_ENGINE_KV_DTYPE_Q4_0: c_int = 3;

/// Mirror of `retro_engine_config`. Versioned by `struct_size`: the runtime
/// reads only the prefix a caller's `struct_size` covers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RetroEngineConfig {
    /// Filled by the caller: the evolution key. Must be `size_of::<Self>()`.
    pub struct_size: u32,
    /// Total KV cells across all sequences.
    pub n_ctx: u32,
    /// Sequence slots.
    pub n_seq_max: u32,
    /// Physical batch: prefill chunk width.
    pub n_batch: u32,
    /// Micro-batch: rows per emitting decode.
    pub n_ubatch: u32,
    /// One of `RETRO_ENGINE_KV_DTYPE_*`.
    pub kv_dtype: c_int,
    /// Request flash attention; required for a quantized `kv_dtype`.
    pub flash_attn: bool,
    /// One of the `RETRO_DEVICE_*` values.
    pub device: i32,
    /// 0 resolves like the trainer (RETRO_THREADS included).
    pub threads: u32,
    pub verbose: bool,
    /// One unified KV stream across all slots (0 or 1); see the C header.
    /// A u32 so the field lands past the old 40-byte layout and the runtime's
    /// struct_size versioning detects callers compiled before it existed.
    pub kv_unified: u32,
}

impl Default for RetroEngineConfig {
    fn default() -> Self {
        Self {
            // The ABI layout is checked in contract_tests below.
            struct_size: size_of::<Self>() as u32,
            n_ctx: 4096,
            n_seq_max: 8,
            n_batch: 512,
            n_ubatch: 512,
            kv_dtype: RETRO_ENGINE_KV_DTYPE_F16,
            flash_attn: true,
            device: 0, // RETRO_DEVICE_AUTO
            threads: 0,
            verbose: false,
            kv_unified: 1,
        }
    }
}

pub type RetroTrainProgressCallback =
    unsafe extern "C" fn(u32, *const RetroTrainMetrics, *mut c_void) -> bool;

unsafe extern "C" {
    pub fn retro_trainer_new(
        model_path: *const c_char,
        train_config: *const RetroTrainConfig,
    ) -> *mut RetroTrainer;

    /// The policy in force, and whether it can still be changed. `struct_size`
    /// must be set by the caller.
    pub fn retro_runtime_config_effective(
        out: *mut RetroRuntimeConfig,
        out_latched: *mut bool,
    ) -> c_int;

    /// Applies the policy without creating a trainer. Idempotent for the value
    /// already in force, an error for a different one once it has been read.
    pub fn retro_runtime_config_apply(config: *const RetroRuntimeConfig) -> c_int;

    pub fn retro_trainer_create_lora(
        trainer: *mut RetroTrainer,
        lora_config: *const RetroLoraConfig,
    ) -> c_int;

    pub fn retro_trainer_load_lora(
        trainer: *mut RetroTrainer,
        adapter_path: *const c_char,
    ) -> c_int;

    pub fn retro_trainer_set_trainable_base(
        trainer: *mut RetroTrainer,
        names: *const *const c_char,
        n_names: usize,
    ) -> c_int;

    pub fn retro_optimizer_slot_initial_bytes(
        dtype: i32,
        init: i32,
        code: u8,
        n_elements: u64,
        out_bytes: *mut c_void,
        n_bytes: usize,
    ) -> c_int;

    pub fn retro_trainer_set_optimizer_assignment(
        trainer: *mut RetroTrainer,
        names: *const *const c_char,
        optimizers: *const i32,
        n_rows: usize,
    ) -> c_int;

    pub fn retro_trainer_tokenize_text(
        trainer: *mut RetroTrainer,
        text: *const c_char,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_tokenize_with(
        trainer: *mut RetroTrainer,
        text: *const c_char,
        add_special: bool,
        parse_special: bool,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> i32;
    pub fn retro_trainer_tokenize_fragment(
        trainer: *mut RetroTrainer,
        text: *const c_char,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_eos_token(trainer: *mut RetroTrainer, out_token: *mut i32) -> c_int;
    pub fn retro_trainer_vocab_size(trainer: *mut RetroTrainer, out_n_vocab: *mut u32) -> c_int;
    pub fn retro_trainer_is_eog_token(
        trainer: *mut RetroTrainer,
        token: i32,
        out_is_eog: *mut bool,
    ) -> c_int;
    pub fn retro_trainer_context_size(trainer: *mut RetroTrainer, out_n_ctx: *mut u32) -> c_int;

    pub fn retro_trainer_format_chat(
        trainer: *mut RetroTrainer,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: usize,
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_format_chat_messages(
        trainer: *mut RetroTrainer,
        messages_json: *const c_char,
        tools_json: *const c_char,
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Extra variables every later render hands to the chat template, as a JSON object.
    pub fn retro_trainer_set_chat_template_variables(
        trainer: *mut RetroTrainer,
        variables_json: *const c_char,
    ) -> c_int;

    pub fn retro_trainer_set_reasoning_budget(trainer: *mut RetroTrainer, budget: i32) -> c_int;

    pub fn retro_trainer_chat_template_supports_tools(
        trainer: *mut RetroTrainer,
        out_supports: *mut bool,
    ) -> c_int;

    pub fn retro_trainer_tool_call_parser(
        trainer: *mut RetroTrainer,
        tools_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Model-free counterparts of the model-backed chat functions, using a
    /// template supplied as Jinja source. Neither opens a model.
    pub fn retro_chat_template_render(
        template_src: *const c_char,
        messages_json: *const c_char,
        tools_json: *const c_char,
        variables_json: *const c_char,
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_chat_template_tool_call_parser(
        template_src: *const c_char,
        tools_json: *const c_char,
        variables_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Takes no trainer: the serialized parser is self-sufficient, so this runs
    /// on the rollout thread while the trainer generates.
    pub fn retro_chat_parse_assistant(
        parser_blob: *const c_char,
        n_parser_blob: usize,
        text: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Same document as [`retro_chat_parse_assistant`], read with
    /// `is_partial=true`: what the parser recognized of a turn still being
    /// generated. `content` is the committed prefix so far and a call's
    /// `arguments` may be an incomplete string.
    pub fn retro_chat_parse_assistant_partial(
        parser_blob: *const c_char,
        n_parser_blob: usize,
        text: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_describe_lora(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_backend_report(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_capability_report(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_model_capabilities(
        trainer: *mut RetroTrainer,
        out_capabilities: *mut RetroModelCapabilities,
    ) -> c_int;

    pub fn retro_trainer_lora_candidate_targets(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_train_preflight(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_preflight_summary(
        trainer: *mut RetroTrainer,
        out_summary: *mut RetroPreflightSummary,
    ) -> c_int;

    pub fn retro_backend_list(
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_gpu_runtime_probe() -> c_int;

    pub fn retro_device_memory(out_free: *mut usize, out_total: *mut usize) -> c_int;

    pub fn retro_transfer_probe(
        bytes: usize,
        iterations: u32,
        out_rates: *mut RetroTransferRates,
    ) -> c_int;

    pub fn retro_probe_op_run(
        op: i32,
        use_gpu: i32,
        ne_src0: *const i64,
        src0: *const c_float,
        ne_src1: *const i64,
        src1: *const c_float,
        ne_src2: *const i64,
        src2: *const c_float,
        param0: c_float,
        param1: c_float,
        dst: *mut c_float,
        dst_len: usize,
    ) -> c_int;

    /// `retro_probe_op_run` with an explicit implementation choice and a report
    /// of what ran. Requesting [`RETRO_KERNEL_IMPL_RIR`] fails rather than
    /// falling back, so a successful RIR test cannot exercise the native kernel.
    pub fn retro_probe_op_run_ex(
        op: i32,
        use_gpu: i32,
        ne_src0: *const i64,
        src0: *const c_float,
        ne_src1: *const i64,
        src1: *const c_float,
        ne_src2: *const i64,
        src2: *const c_float,
        param0: c_float,
        param1: c_float,
        dst: *mut c_float,
        dst_len: usize,
        implementation: i32,
        info: *mut RetroKernelRunInfo,
    ) -> c_int;

    /// Runs both fixed-block Gefen phases over caller-supplied state. Fails
    /// when the selected device declines either phase: they are admitted
    /// together, so a partial answer is never returned.
    pub fn retro_probe_gefen_run(probe: *mut RetroGefenProbe) -> c_int;

    /// The grouped softmax cross-entropy gradient of one F32 tensor of a model
    /// file, analytic against finite differences; see
    /// `RetroGroupedGradientProbe`.
    pub fn retro_probe_grouped_gradient(probe: *mut RetroGroupedGradientProbe) -> c_int;

    /// Whether the device declares both Gefen phases for this variant and block
    /// size. The predicate is the backend's own, so it is the answer a run's
    /// preflight gets.
    pub fn retro_probe_gefen_supported(
        use_gpu: i32,
        variant: i32,
        block_size: i32,
        out_supported: *mut i32,
    ) -> c_int;

    /// Fills a snapshot of the process-wide RIR dispatch counters. `struct_size`
    /// must be set by the caller.
    pub fn retro_rir_counters_get(out: *mut RetroRirCounters) -> c_int;

    /// One line per compiled RIR variant and per natively-locked op; same
    /// two-call buffer contract as [`retro_backend_list`].
    pub fn retro_rir_variant_report(
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// The per-`(ggml_op, backend)` census of the graphs computed so far, plus
    /// the per-chain `census-pattern` rows that rank fusion candidates. Empty
    /// unless `RETRO_RIR_CENSUS=1` was set before the first graph. Same
    /// two-call buffer contract as [`retro_backend_list`].
    pub fn retro_rir_census_report(
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Runs the registry's selection rule on synthetic tables. Returns 0 when
    /// every case held, otherwise a bitmask of the failing ones. Needs no
    /// device, so it belongs to the model-free lane.
    pub fn retro_rir_selection_selftest() -> u32;

    pub fn retro_dequant_types(out_types: *mut i32, cap: usize) -> usize;

    pub fn retro_ggml_type_name(ty: i32) -> *const std::os::raw::c_char;

    /// `blck_size`/`type_size` of a ggml type, the two numbers the RIR quant
    /// table restates.
    pub fn retro_quant_traits(
        ty: i32,
        out_block_elements: *mut i64,
        out_block_bytes: *mut usize,
    ) -> c_int;

    /// Quantizes with ggml's reference quantizer and decodes with ggml's own
    /// `to_float`, returning both the bytes and the decoded values.
    pub fn retro_quant_roundtrip(
        ty: i32,
        n: i64,
        src: *const c_float,
        out_bytes: *mut u8,
        out_bytes_cap: usize,
        out_dequant: *mut c_float,
    ) -> c_int;

    pub fn retro_fused_sparse_ce_probe(
        n_embd: i32,
        n_tokens: i32,
        n_vocab: i32,
        n_topk: i32,
        n_tiles: i32,
        seq_chunk: i32,
        offload_h: i32,
        w_type: i32,
        use_gpu: i32,
        h: *const c_float,
        w: *const c_float,
        targets: *const i32,
        weights: *const c_float,
        bias: *const c_float,
        grad_loss: c_float,
        out_loss_full: *mut c_float,
        out_loss_fused: *mut c_float,
        out_grad_h_full: *mut c_float,
        out_grad_h_fused: *mut c_float,
    ) -> c_int;

    pub fn retro_probe_token_logprob(
        logits: *const c_float,
        n_vocab: usize,
        token: i32,
        vectorized: bool,
        out_logprob: *mut c_float,
    ) -> c_int;

    pub fn retro_trainer_train_tokens(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        out_metrics: *mut RetroTrainMetrics,
    ) -> c_int;

    pub fn retro_trainer_train_sft(
        trainer: *mut RetroTrainer,
        train: *const RetroSftDataset,
        eval: *const RetroSftDataset,
        out_metrics: *mut RetroTrainMetrics,
        progress_callback: Option<RetroTrainProgressCallback>,
        progress_user_data: *mut c_void,
    ) -> c_int;

    pub fn retro_trainer_eval_sft(
        trainer: *mut RetroTrainer,
        data: *const RetroSftDataset,
        out_metrics: *mut RetroEvalMetrics,
    ) -> c_int;

    pub fn retro_trainer_generate_batch(
        trainer: *mut RetroTrainer,
        prompt_tokens: *const i32,
        n_prompt: usize,
        sampling: *const RetroSamplingParams,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
        out_forced: *mut u8,
    ) -> c_int;

    pub fn retro_trainer_generate_continuous_batch(
        trainer: *mut RetroTrainer,
        sequences: *const RetroGenerationSequence,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
        out_forced: *mut u8,
    ) -> c_int;

    /// Mirror of `retro_trainer_generate_continuous_batch_v2`: v2 rows with
    /// extended sampling (grammar included), sampled by the CPU oracle.
    pub fn retro_trainer_generate_continuous_batch_v2(
        trainer: *mut RetroTrainer,
        sequences: *const RetroGenerationSequenceV2,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
        out_forced: *mut u8,
    ) -> c_int;

    pub fn retro_trainer_generation_step(
        trainer: *mut RetroTrainer,
        rows: *const RetroStepRow,
        n_rows: usize,
        out_rows: *mut RetroStepOutRow,
        out_top_ids: *mut i32,
        out_top_logprobs: *mut c_float,
        n_top_stride: usize,
    ) -> c_int;

    pub fn retro_trainer_generation_seq_rm(
        trainer: *mut RetroTrainer,
        slot: i32,
        p0: i32,
        p1: i32,
    ) -> c_int;

    pub fn retro_trainer_generation_seq_cp(
        trainer: *mut RetroTrainer,
        src: i32,
        dst: i32,
        p0: i32,
        p1: i32,
    ) -> c_int;

    pub fn retro_trainer_generation_seq_save(
        trainer: *mut RetroTrainer,
        slot: i32,
        flags: u32,
        buffer: *mut u8,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_generation_seq_load(
        trainer: *mut RetroTrainer,
        slot: i32,
        flags: u32,
        buffer: *const u8,
        n_bytes: usize,
        out_n_read: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_generation_seq_pos(
        trainer: *mut RetroTrainer,
        slot: i32,
        out_pos_min: *mut i32,
        out_pos_max: *mut i32,
    ) -> c_int;

    pub fn retro_trainer_generation_memory(
        trainer: *mut RetroTrainer,
        out: *mut RetroEngineMemoryInfo,
    ) -> c_int;

    pub fn retro_trainer_generation_request_free(trainer: *mut RetroTrainer, request: i32)
    -> c_int;

    /// Mirror of `retro_trainer_generation_kv_unified`.
    pub fn retro_trainer_generation_kv_unified(
        trainer: *mut RetroTrainer,
        out_unified: *mut bool,
    ) -> c_int;

    /// Mirror of `retro_trainer_generation_weights_version`: bumped by every
    /// `forget_generation_kv` (optimizer step, import, adapter change).
    pub fn retro_trainer_generation_weights_version(
        trainer: *mut RetroTrainer,
        out_version: *mut u64,
    ) -> c_int;

    /// Mirror of `retro_trainer_model_fingerprint`.
    pub fn retro_trainer_model_fingerprint(
        trainer: *mut RetroTrainer,
        out_fingerprint: *mut u64,
    ) -> c_int;

    /// Mirror of `retro_trainer_is_hybrid`.
    pub fn retro_trainer_is_hybrid(trainer: *mut RetroTrainer, out_is_hybrid: *mut bool) -> c_int;

    pub fn retro_trainer_score_tokens(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        out_logprobs: *mut c_float,
    ) -> c_int;

    pub fn retro_trainer_score_token_suffix(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        n_prompt: usize,
        out_logprobs: *mut c_float,
    ) -> c_int;

    pub fn retro_trainer_top_logprobs_suffix(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        n_prompt: usize,
        k: usize,
        out_ids: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
    ) -> c_int;

    pub fn retro_trainer_score_token_suffix_batch(
        trainer: *mut RetroTrainer,
        sequences: *const RetroTokenSuffixSequence,
        n_sequences: usize,
        out_logprobs: *mut c_float,
        out_stride: usize,
        out_n_logprobs: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_set_lora_enabled(trainer: *mut RetroTrainer, enabled: bool) -> c_int;

    pub fn retro_trainer_hidden_size(trainer: *mut RetroTrainer, out_n_embd: *mut u32) -> c_int;

    pub fn retro_trainer_hidden_states(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        out_features: *mut c_float,
        n_features_max: usize,
    ) -> c_int;

    pub fn retro_trainer_score_token_suffix_and_hidden_states(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        n_prompt: usize,
        out_logprobs: *mut c_float,
        out_features: *mut c_float,
        n_features_max: usize,
    ) -> c_int;

    pub fn retro_trainer_detokenize(
        trainer: *mut RetroTrainer,
        tokens: *const i32,
        n_tokens: usize,
        unparse_special: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_train_weighted(
        trainer: *mut RetroTrainer,
        data: *const RetroWeightedDataset,
        scheduler_total_steps: u64,
        out_metrics: *mut RetroTrainMetrics,
        progress_callback: Option<RetroTrainProgressCallback>,
        progress_user_data: *mut c_void,
    ) -> c_int;

    pub fn retro_trainer_train_packed_sequences(
        trainer: *mut RetroTrainer,
        data: *const RetroPackedSequenceBatch,
        scheduler_total_steps: u64,
        accumulation_steps: u32,
        out_metrics: *mut RetroTrainMetrics,
        progress_callback: Option<RetroTrainProgressCallback>,
        progress_user_data: *mut c_void,
    ) -> c_int;

    pub fn retro_trainer_optimizer_timing(
        trainer: *mut RetroTrainer,
        out_timing: *mut RetroOptimizerTiming,
    ) -> c_int;

    pub fn retro_read_model_info(
        model_path: *const c_char,
        device: i32,
        out_info: *mut RetroModelInfo,
    ) -> c_int;

    pub fn retro_read_tensor_inventory(
        model_path: *const c_char,
        out_version: *mut u32,
        out_tensors: *mut RetroTensorDesc,
        n_max: usize,
        out_count: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_memory_report(
        trainer: *mut RetroTrainer,
        out_report: *mut RetroMemoryReport,
    ) -> c_int;

    pub fn retro_trainer_optimizer_memory(
        trainer: *mut RetroTrainer,
        out_memory: *mut RetroOptimizerMemory,
    ) -> c_int;

    pub fn retro_trainer_scoring_stats(
        trainer: *mut RetroTrainer,
        out_stats: *mut RetroScoringStats,
    ) -> c_int;

    pub fn retro_trainer_generation_stats(
        trainer: *mut RetroTrainer,
        out_stats: *mut RetroGenerationStats,
    ) -> c_int;

    pub fn retro_trainer_set_max_gpu_duty_cycle(
        trainer: *mut RetroTrainer,
        fraction: c_float,
    ) -> c_int;

    pub fn retro_trainer_duty_cycle_stats(
        trainer: *mut RetroTrainer,
        out_stats: *mut RetroDutyCycleStats,
    ) -> c_int;

    pub fn retro_probe_duty_cycle(
        fraction: c_float,
        events: *const RetroDutyCycleEvent,
        n_events: usize,
        out_probe: *mut RetroDutyCycleProbe,
    ) -> c_int;

    pub fn retro_trainer_advance_scheduler_steps(
        trainer: *mut RetroTrainer,
        steps: u64,
        out_global_step: *mut u64,
    ) -> c_int;

    pub fn retro_trainer_set_learning_rate(trainer: *mut RetroTrainer, learning_rate: f32)
    -> c_int;

    pub fn retro_trainer_save_lora(
        trainer: *mut RetroTrainer,
        adapter_path: *const c_char,
    ) -> c_int;

    pub fn retro_trainer_save_trainable(
        trainer: *mut RetroTrainer,
        trainable_path: *const c_char,
    ) -> c_int;

    pub fn retro_trainer_load_trainable(
        trainer: *mut RetroTrainer,
        trainable_path: *const c_char,
    ) -> c_int;

    pub fn retro_trainer_save_model(trainer: *mut RetroTrainer, model_path: *const c_char)
    -> c_int;

    pub fn retro_trainer_optimizer_state(
        trainer: *mut RetroTrainer,
        out_state: *mut RetroOptimizerState,
    ) -> c_int;

    pub fn retro_trainer_restore_optimizer_state(
        trainer: *mut RetroTrainer,
        state: *const RetroOptimizerState,
    ) -> c_int;

    pub fn retro_trainer_set_resume_point(
        trainer: *mut RetroTrainer,
        completed_epochs: u32,
    ) -> c_int;

    pub fn retro_trainer_prepare_optimizer(trainer: *mut RetroTrainer) -> c_int;

    pub fn retro_trainer_marked_parameter_count(
        trainer: *mut RetroTrainer,
        out_count: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_marked_parameter_info(
        trainer: *mut RetroTrainer,
        index: usize,
        out_tensor: *mut RetroTensorDesc,
    ) -> c_int;

    pub fn retro_trainer_marked_parameter_read(
        trainer: *mut RetroTrainer,
        index: usize,
        offset: u64,
        out_bytes: *mut c_void,
        n_bytes: usize,
    ) -> c_int;

    pub fn retro_trainer_parameter_gradient_info(
        trainer: *mut RetroTrainer,
        index: usize,
        out_tensor: *mut RetroTensorDesc,
    ) -> c_int;

    pub fn retro_trainer_parameter_gradient_read(
        trainer: *mut RetroTrainer,
        index: usize,
        offset: u64,
        out_bytes: *mut c_void,
        n_bytes: usize,
    ) -> c_int;

    pub fn retro_trainer_state_slot_count(
        trainer: *mut RetroTrainer,
        out_parameter_slots: *mut usize,
        out_shared_slots: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_state_slot_info(
        trainer: *mut RetroTrainer,
        scope: i32,
        index: usize,
        out_slot: *mut RetroOptimizerSlot,
    ) -> c_int;

    pub fn retro_trainer_state_slot_read(
        trainer: *mut RetroTrainer,
        scope: i32,
        index: usize,
        offset: u64,
        out_bytes: *mut c_void,
        n_bytes: usize,
    ) -> c_int;

    pub fn retro_trainer_state_slot_write(
        trainer: *mut RetroTrainer,
        scope: i32,
        owner: *const c_char,
        slot: *const c_char,
        offset: u64,
        bytes: *const c_void,
        n_bytes: usize,
    ) -> c_int;

    pub fn retro_trainer_rng_state(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_set_rng_state(trainer: *mut RetroTrainer, state: *const c_char) -> c_int;

    pub fn retro_trainer_model_signature(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_free(trainer: *mut RetroTrainer);

    pub fn retro_engine_new(
        model_path: *const c_char,
        config: *const RetroEngineConfig,
    ) -> *mut RetroEngine;

    pub fn retro_engine_free(engine: *mut RetroEngine);

    pub fn retro_engine_generate_batch(
        engine: *mut RetroEngine,
        prompt_tokens: *const i32,
        n_prompt: usize,
        sampling: *const RetroSamplingParams,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
        out_forced: *mut u8,
    ) -> c_int;

    pub fn retro_engine_generate_continuous_batch(
        engine: *mut RetroEngine,
        sequences: *const RetroGenerationSequence,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
        out_forced: *mut u8,
    ) -> c_int;

    pub fn retro_engine_tokenize_text(
        engine: *mut RetroEngine,
        text: *const c_char,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

    pub fn retro_engine_tokenize_fragment(
        engine: *mut RetroEngine,
        text: *const c_char,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

    pub fn retro_engine_eos_token(engine: *mut RetroEngine, out_token: *mut i32) -> c_int;
    pub fn retro_engine_vocab_size(engine: *mut RetroEngine, out_n_vocab: *mut u32) -> c_int;
    pub fn retro_engine_is_eog_token(
        engine: *mut RetroEngine,
        token: i32,
        out_is_eog: *mut bool,
    ) -> c_int;
    pub fn retro_engine_context_size(engine: *mut RetroEngine, out_n_ctx: *mut u32) -> c_int;

    pub fn retro_engine_detokenize(
        engine: *mut RetroEngine,
        tokens: *const i32,
        n_tokens: usize,
        unparse_special: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_load_lora(engine: *mut RetroEngine, adapter_path: *const c_char) -> c_int;
    pub fn retro_engine_set_lora_enabled(engine: *mut RetroEngine, enabled: bool) -> c_int;

    /// Mirror of `retro_engine_load_lora_named`: register an adapter under
    /// `name` beside the others, without applying it.
    pub fn retro_engine_load_lora_named(
        engine: *mut RetroEngine,
        name: *const c_char,
        adapter_path: *const c_char,
    ) -> c_int;

    /// Mirror of `retro_engine_activate_lora`: apply the adapter registered
    /// under `name` (null/empty = base model); clears KV and bumps the weights
    /// version.
    pub fn retro_engine_activate_lora(engine: *mut RetroEngine, name: *const c_char) -> c_int;

    /// Mirror of `retro_trainer_generation_adapter_snapshot`: copy the
    /// trained adapter into a per-row adapter of the generation context.
    pub fn retro_trainer_generation_adapter_snapshot(
        trainer: *mut RetroTrainer,
        replace: i32,
        scale: c_float,
        out_id: *mut i32,
    ) -> c_int;

    /// Mirror of `retro_trainer_generation_adapter_unload`.
    pub fn retro_trainer_generation_adapter_unload(trainer: *mut RetroTrainer, id: i32) -> c_int;

    /// Mirror of `retro_trainer_get_generation_adapter`.
    pub fn retro_trainer_get_generation_adapter(
        trainer: *mut RetroTrainer,
        out_adapter: *mut i32,
    ) -> c_int;

    /// Mirror of `retro_trainer_set_generation_adapter`.
    pub fn retro_trainer_set_generation_adapter(trainer: *mut RetroTrainer, adapter: i32) -> c_int;

    /// Mirror of `retro_trainer_generation_adapter_rows_info`.
    pub fn retro_trainer_generation_adapter_rows_info(
        trainer: *mut RetroTrainer,
        out_info: *mut RetroAdapterRowsInfo,
    ) -> c_int;

    /// Mirror of `retro_engine_adapter_load`: load a LoRA checkpoint of the
    /// base model as a per-row adapter under a fresh id, never reused.
    pub fn retro_engine_adapter_load(
        engine: *mut RetroEngine,
        adapter_path: *const c_char,
        scale: c_float,
        out_id: *mut i32,
    ) -> c_int;

    /// Mirror of `retro_engine_adapter_unload`.
    pub fn retro_engine_adapter_unload(engine: *mut RetroEngine, id: i32) -> c_int;

    /// Mirror of `retro_engine_adapter_rows_info`.
    pub fn retro_engine_adapter_rows_info(
        engine: *mut RetroEngine,
        out_info: *mut RetroAdapterRowsInfo,
    ) -> c_int;

    /// Mirror of `retro_engine_graphs_reused`.
    pub fn retro_engine_graphs_reused(engine: *mut RetroEngine, out_n_reused: *mut u64) -> c_int;

    /// Mirror of `retro_engine_active_adapter`: writes the active adapter's
    /// name (empty = base), two-call sized-string contract.
    pub fn retro_engine_active_adapter(
        engine: *mut RetroEngine,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_set_chat_template_variables(
        engine: *mut RetroEngine,
        variables_json: *const c_char,
    ) -> c_int;
    pub fn retro_engine_set_reasoning_budget(engine: *mut RetroEngine, budget: i32) -> c_int;

    /// Mirror of `retro_engine_format_chat`: render messages through the
    /// GGUF's chat template, same contract as `retro_trainer_format_chat`.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_engine_format_chat(
        engine: *mut RetroEngine,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: usize,
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Mirror of `retro_engine_format_chat_messages`, the JSON-messages
    /// counterpart of `retro_trainer_format_chat_messages`.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_engine_format_chat_messages(
        engine: *mut RetroEngine,
        messages_json: *const c_char,
        tools_json: *const c_char,
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Mirror of `retro_engine_chat_template_supports_tools`.
    pub fn retro_engine_chat_template_supports_tools(
        engine: *mut RetroEngine,
        out_supports: *mut bool,
    ) -> c_int;

    /// Mirror of `retro_engine_tool_call_parser`; returns
    /// `RETRO_CHAT_PARSER_UNAVAILABLE` when the template yields no parser.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_engine_tool_call_parser(
        engine: *mut RetroEngine,
        tools_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_memory_report(
        engine: *mut RetroEngine,
        out_report: *mut RetroMemoryReport,
    ) -> c_int;

    pub fn retro_engine_generation_stats(
        engine: *mut RetroEngine,
        out_stats: *mut RetroGenerationStats,
    ) -> c_int;

    pub fn retro_engine_step(
        engine: *mut RetroEngine,
        rows: *const RetroStepRow,
        n_rows: usize,
        out_rows: *mut RetroStepOutRow,
        out_top_ids: *mut i32,
        out_top_logprobs: *mut c_float,
        n_top_stride: usize,
    ) -> c_int;

    /// Mirror of `retro_engine_step_v2`: the v2 step entry point over
    /// `RetroStepRowV2` / `RetroStepOutRowV2` (extended sampling).
    #[allow(clippy::too_many_arguments)]
    pub fn retro_engine_step_v2(
        engine: *mut RetroEngine,
        rows: *const RetroStepRowV2,
        n_rows: usize,
        out_rows: *mut RetroStepOutRowV2,
        out_top_ids: *mut i32,
        out_top_logprobs: *mut c_float,
        n_top_stride: usize,
    ) -> c_int;

    /// Mirror of `retro_trainer_generation_step_v2`.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_trainer_generation_step_v2(
        trainer: *mut RetroTrainer,
        rows: *const RetroStepRowV2,
        n_rows: usize,
        out_rows: *mut RetroStepOutRowV2,
        out_top_ids: *mut i32,
        out_top_logprobs: *mut c_float,
        n_top_stride: usize,
    ) -> c_int;

    /// Mirror of `retro_grammar_from_json_schema`; NULL on failure with
    /// `retro_last_error()` set.
    pub fn retro_grammar_from_json_schema(json_schema: *const c_char) -> *mut RetroGrammar;

    /// Mirror of `retro_grammar_from_gbnf`; NULL on failure with
    /// `retro_last_error()` set.
    pub fn retro_grammar_from_gbnf(gbnf: *const c_char) -> *mut RetroGrammar;

    /// Mirror of `retro_grammar_json_object` (the built-in any-JSON grammar).
    pub fn retro_grammar_json_object() -> *mut RetroGrammar;

    /// Mirror of `retro_grammar_gbnf`; owned by the handle, NUL-terminated.
    pub fn retro_grammar_gbnf(grammar: *const RetroGrammar) -> *const c_char;

    /// Mirror of `retro_grammar_free`.
    pub fn retro_grammar_free(grammar: *mut RetroGrammar);

    /// Mirror of `retro_engine_tool_call_grammar`; returns
    /// `RETRO_TOOL_GRAMMAR_UNAVAILABLE` when the template cannot express the
    /// mode.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_engine_tool_call_grammar(
        engine: *mut RetroEngine,
        tools_json: *const c_char,
        mode: i32,
        forced_name: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Mirror of `retro_trainer_tool_call_grammar`.
    #[allow(clippy::too_many_arguments)]
    pub fn retro_trainer_tool_call_grammar(
        trainer: *mut RetroTrainer,
        tools_json: *const c_char,
        mode: i32,
        forced_name: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_seq_rm(engine: *mut RetroEngine, slot: i32, p0: i32, p1: i32) -> c_int;

    pub fn retro_engine_seq_cp(
        engine: *mut RetroEngine,
        src: i32,
        dst: i32,
        p0: i32,
        p1: i32,
    ) -> c_int;

    pub fn retro_engine_seq_save(
        engine: *mut RetroEngine,
        slot: i32,
        flags: u32,
        buffer: *mut u8,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_seq_load(
        engine: *mut RetroEngine,
        slot: i32,
        flags: u32,
        buffer: *const u8,
        n_bytes: usize,
        out_n_read: *mut usize,
    ) -> c_int;

    pub fn retro_engine_seq_pos(
        engine: *mut RetroEngine,
        slot: i32,
        out_pos_min: *mut i32,
        out_pos_max: *mut i32,
    ) -> c_int;

    pub fn retro_engine_memory(engine: *mut RetroEngine, out: *mut RetroEngineMemoryInfo) -> c_int;

    pub fn retro_engine_request_free(engine: *mut RetroEngine, request: i32) -> c_int;

    /// Mirror of `retro_engine_kv_unified`: whether the engine context's KV
    /// pool is unified (any-range seq_cp is metadata-only) or partitioned.
    pub fn retro_engine_kv_unified(engine: *mut RetroEngine, out_unified: *mut bool) -> c_int;

    /// Mirror of `retro_engine_weights_version`: generation of the weights the
    /// resident KV was computed under.
    pub fn retro_engine_weights_version(engine: *mut RetroEngine, out_version: *mut u64) -> c_int;

    /// Mirror of `retro_engine_model_fingerprint`.
    pub fn retro_engine_model_fingerprint(
        engine: *mut RetroEngine,
        out_fingerprint: *mut u64,
    ) -> c_int;

    /// Mirror of `retro_engine_is_hybrid`.
    pub fn retro_engine_is_hybrid(engine: *mut RetroEngine, out_is_hybrid: *mut bool) -> c_int;

    /// Mirror of `retro_engine_pooling_type`: the model's declared pooling
    /// (`llama_pooling_type`: 0 = none, 1 = mean, 2 = cls, 3 = last, 4 = rank).
    pub fn retro_engine_pooling_type(engine: *mut RetroEngine, out_pooling: *mut i32) -> c_int;

    /// Mirror of `retro_engine_embed`: the pooled embedding of one token
    /// sequence, or `RETRO_EMBED_NO_POOLING` when the model declares no pooling.
    /// With a null `out_features`, only `out_n_embd` is filled (size query).
    pub fn retro_engine_embed(
        engine: *mut RetroEngine,
        tokens: *const i32,
        n_tokens: usize,
        out_features: *mut f32,
        n_embd_max: usize,
        out_n_embd: *mut usize,
    ) -> c_int;

    /// What a model declares about its decision head, as a JSON object; see
    /// the C header for the fields. Opens the model mapped, with no context.
    pub fn retro_read_decision_info(
        model_path: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_decision_info(
        engine: *mut RetroEngine,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Renders `tokenizer.chat_template.<name>` over a JSON object that is the
    /// template's whole input.
    pub fn retro_engine_render_named_template(
        engine: *mut RetroEngine,
        name: *const c_char,
        context_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Renders `source` in place of the model's template `name`; a null source
    /// goes back to the model's.
    pub fn retro_engine_set_named_template(
        engine: *mut RetroEngine,
        name: *const c_char,
        source: *const c_char,
    ) -> c_int;

    pub fn retro_template_render_context(
        template_src: *const c_char,
        context_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_engine_tokenize(
        engine: *mut RetroEngine,
        text: *const c_char,
        add_special: bool,
        parse_special: bool,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

    pub fn retro_engine_token_piece(
        engine: *mut RetroEngine,
        token: i32,
        special: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    /// Evaluates decision prompts, `n_reads` scores per prompt in order, at
    /// most `n_pack_max` (0 counts as 1) per decode.
    pub fn retro_engine_decide(
        engine: *mut RetroEngine,
        readout: i32,
        prompts: *const RetroDecisionPrompt,
        n_prompts: usize,
        n_pack_max: usize,
        out_scores: *mut f32,
        n_scores_max: usize,
    ) -> c_int;

    /// `retro_engine_decision_info` for a trainer's model.
    pub fn retro_trainer_decision_info(
        trainer: *mut RetroTrainer,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_render_named_template(
        trainer: *mut RetroTrainer,
        name: *const c_char,
        context_json: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_set_named_template(
        trainer: *mut RetroTrainer,
        name: *const c_char,
        source: *const c_char,
    ) -> c_int;

    /// `retro_engine_decide` under the trainer's current weights.
    pub fn retro_trainer_decide(
        trainer: *mut RetroTrainer,
        readout: i32,
        prompts: *const RetroDecisionPrompt,
        n_prompts: usize,
        n_pack_max: usize,
        out_scores: *mut f32,
        n_scores_max: usize,
    ) -> c_int;

    /// One physical step of the grouped softmax cross-entropy.
    pub fn retro_trainer_train_grouped(
        trainer: *mut RetroTrainer,
        batch: *const RetroGroupedBatch,
        scheduler_total_steps: u64,
        out_metrics: *mut RetroTrainMetrics,
    ) -> c_int;

    /// Copies a GGUF with metadata entries set, replaced or removed (a JSON
    /// object by key, null removing), its tensors byte for byte.
    pub fn retro_gguf_rewrite_metadata(
        src_path: *const c_char,
        dst_path: *const c_char,
        entries_json: *const c_char,
    ) -> c_int;

    /// The metadata entries of a GGUF under a key prefix, as a JSON object of
    /// the texts llama.cpp prints.
    pub fn retro_gguf_read_metadata(
        path: *const c_char,
        prefix: *const c_char,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_last_error() -> *const c_char;
}

pub fn last_error() -> String {
    unsafe {
        let ptr = retro_last_error();
        if ptr.is_null() {
            return "unknown runtime error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

// These tests deliberately call the C ABI directly.  The safe `Trainer` API
// has its own tests; this module is the guardrail for the assumptions it makes
// about return values, output buffers, and the layout of the C structs.
#[cfg(test)]
mod contract_tests {
    use super::*;
    use std::ffi::CString;
    use std::mem::{align_of, offset_of, size_of};
    use std::ptr;

    fn assert_null_trainer(code: c_int) {
        assert_eq!(code, -1);
        assert_eq!(last_error(), "trainer is null");
    }

    /// Replays a duty-cycle script on the runtime's fake clock. `script` is
    /// `(kind, micros)` pairs; nothing here waits in real time.
    fn duty_cycle(fraction: f32, script: &[(i32, u64)]) -> RetroDutyCycleProbe {
        let events: Vec<RetroDutyCycleEvent> = script
            .iter()
            .map(|&(kind, micros)| RetroDutyCycleEvent { kind, micros })
            .collect();
        let mut probe = RetroDutyCycleProbe::default();
        // SAFETY: `events` outlives the synchronous call and `probe` is a valid out-pointer.
        let code =
            unsafe { retro_probe_duty_cycle(fraction, events.as_ptr(), events.len(), &mut probe) };
        assert_eq!(code, 0, "{}", last_error());
        probe
    }

    const WORK: i32 = RETRO_DUTY_CYCLE_EVENT_WORK;
    const IDLE: i32 = RETRO_DUTY_CYCLE_EVENT_IDLE;
    const HOST: i32 = RETRO_DUTY_CYCLE_EVENT_HOST;
    const RESET: i32 = RETRO_DUTY_CYCLE_EVENT_RESET;

    #[test]
    fn a_duty_cycle_repays_one_over_fraction_minus_one_per_unit_of_work() {
        // 200 ms of work at 50% earns 200 ms of idle, at 75% earns ~66.7 ms,
        // at 25% earns 600 ms - which exceeds the 100 ms sleep cap, so the
        // repayment is spread over the boundaries that follow.
        for (fraction, expected_idle_micros, boundaries) in [
            (0.5_f32, 200_000_u64, 2_usize),
            (0.75, 66_666, 1),
            (0.25, 600_000, 6),
        ] {
            let mut script = vec![(WORK, 200_000_u64)];
            script.extend(std::iter::repeat_n((IDLE, 0), boundaries));
            let probe = duty_cycle(fraction, &script);
            let idle = probe.stats.idle_seconds;
            assert!(
                (idle - expected_idle_micros as f64 / 1e6).abs() < 2e-6,
                "fraction {fraction}: idle {idle}s, expected {expected_idle_micros}us"
            );
            assert_eq!(probe.stats.compute_seconds, 0.2);
            assert_eq!(probe.stats.requested_fraction, fraction);
            assert!(probe.stats.active);
        }
    }

    #[test]
    fn a_capped_sleep_carries_its_remainder_to_the_next_boundary() {
        // 1 s of work at 25% owes 3 s. No single sleep may exceed 100 ms, so
        // one boundary repays exactly that and the debt survives to the next.
        let one = duty_cycle(0.25, &[(WORK, 1_000_000), (IDLE, 0)]);
        assert_eq!(one.stats.idle_seconds, 0.1);
        assert_eq!(one.sleep_calls, 1);

        let two = duty_cycle(0.25, &[(WORK, 1_000_000), (IDLE, 0), (IDLE, 0)]);
        assert_eq!(two.stats.idle_seconds, 0.2);
        assert_eq!(two.sleep_calls, 2);

        // Thirty boundaries repay the whole 3 s and no more: the debt is
        // repaid, not renewed, so the sleeper stops being called.
        let all = duty_cycle(
            0.25,
            &std::iter::once((WORK, 1_000_000))
                .chain(std::iter::repeat_n((IDLE, 0), 40))
                .collect::<Vec<_>>(),
        );
        assert_eq!(all.stats.idle_seconds, 3.0);
        assert_eq!(all.sleep_calls, 30);
    }

    #[test]
    fn short_debts_coalesce_across_boundaries_instead_of_being_lost() {
        // 1 ms of work at 50% owes 1 ms, below the 2 ms coalescing floor: a
        // sleep that short is host scheduler noise rather than released
        // compute, so the sleeper is not called.
        let single = duty_cycle(0.5, &[(WORK, 1_000), (IDLE, 0)]);
        assert_eq!(single.sleep_calls, 0);
        assert_eq!(single.stats.idle_seconds, 0.0);

        // Carried, not dropped: the second window crosses the floor and both
        // milliseconds are repaid at once. Four windows repay four, in two
        // sleeps of two - the ledger stays exact whatever the slice length.
        let owed: Vec<(usize, f64)> = (1..=4)
            .map(|windows| {
                let script: Vec<(i32, u64)> =
                    std::iter::repeat_n([(WORK, 1_000), (IDLE, 0)], windows)
                        .flatten()
                        .collect();
                let probe = duty_cycle(0.5, &script);
                (probe.sleep_calls as usize, probe.stats.idle_seconds)
            })
            .collect();
        assert_eq!(owed, vec![(0, 0.0), (1, 0.002), (1, 0.002), (2, 0.004)]);
    }

    #[test]
    fn a_host_gap_shorter_than_the_stale_window_keeps_the_pending_debt() {
        // An ordinary progress row must not be mistaken for a pause.
        let kept = duty_cycle(0.5, &[(WORK, 200_000), (HOST, 499_000), (IDLE, 0)]);
        assert_eq!(kept.stats.idle_seconds, 0.1);

        // At the threshold the debt is discarded: the host blocked for a reason
        // the limiter did not choose, the device was free throughout, and
        // repaying now would idle for compute the neighbour has already had.
        let discarded = duty_cycle(0.5, &[(WORK, 200_000), (HOST, 500_000), (IDLE, 0)]);
        assert_eq!(discarded.stats.idle_seconds, 0.0);
        assert_eq!(discarded.sleep_calls, 0);
        // The compute counter is a lifetime total and survives: it describes
        // the run, not the window that was dropped.
        assert_eq!(discarded.stats.compute_seconds, 0.2);
    }

    #[test]
    fn host_time_between_two_boundaries_is_never_charged_as_gpu_work() {
        // The regression that would over-throttle a fast micro-batch: a Rust
        // progress callback, a reward subprocess or a judge sits between an
        // accounting boundary and the next submission. Leaving the window open
        // across it charges that host time as compute and manufactures debt
        // from it.
        // 100 ms of work, then 400 ms of host time (under the stale window, so
        // the debt survives), then 100 ms more work: compute must be exactly
        // 200 ms and the debt exactly 200 ms, not 600 ms.
        let probe = duty_cycle(
            0.5,
            &[
                (WORK, 100_000),
                (HOST, 400_000),
                (WORK, 100_000),
                (IDLE, 0),
                (IDLE, 0),
                (IDLE, 0),
            ],
        );
        assert_eq!(probe.stats.compute_seconds, 0.2);
        assert_eq!(probe.stats.idle_seconds, 0.2);

        // Same with a debt too small to sleep on at the first boundary: the
        // window still has to reopen, or the host time lands in the next
        // accounted window instead.
        let coalescing = duty_cycle(
            0.5,
            &[
                (WORK, 1_000),
                (IDLE, 0),
                (HOST, 400_000),
                (WORK, 1_000),
                (IDLE, 0),
            ],
        );
        assert_eq!(coalescing.stats.compute_seconds, 0.002);
        assert_eq!(coalescing.stats.idle_seconds, 0.002);
    }

    #[test]
    fn a_later_operation_does_not_repay_an_earlier_one_across_a_long_host_phase() {
        // The GRPO shape: generation leaves debt the optimizer would otherwise
        // inherit across a reward subprocess or a judge. The device was free
        // for that whole phase, so the neighbour has already had the compute
        // and idling for it now would be paying twice.
        // 1 s of work at 25% owes 3 s and one boundary repays 100 ms of it.
        let carried = duty_cycle(0.25, &[(WORK, 1_000_000), (IDLE, 0), (IDLE, 0)]);
        assert_eq!(carried.stats.idle_seconds, 0.2);

        // Insert a 5 s judge between the two boundaries and the second one has
        // nothing left to repay.
        let dropped = duty_cycle(
            0.25,
            &[(WORK, 1_000_000), (IDLE, 0), (HOST, 5_000_000), (IDLE, 0)],
        );
        assert_eq!(dropped.stats.idle_seconds, 0.1);
        assert_eq!(dropped.sleep_calls, 1);
    }

    #[test]
    fn a_reset_discards_the_debt_and_the_window_but_not_the_counters() {
        let probe = duty_cycle(
            0.5,
            &[
                (WORK, 200_000),
                (RESET, 0),
                (IDLE, 0),
                (WORK, 100_000),
                (IDLE, 0),
            ],
        );
        // Only the second window's 100 ms is repaid; the first was reset away.
        assert_eq!(probe.stats.idle_seconds, 0.1);
        assert_eq!(probe.stats.compute_seconds, 0.3);
    }

    #[test]
    fn unaccounted_host_time_separates_the_wall_share_from_the_observed_share() {
        // Two 200 ms windows at 50%, each fully repaid, with 400 ms of
        // CPU-only work in between: 400 ms of compute and 400 ms of idle over
        // a 1.2 s wall clock. `observed` echoes the setting back at 0.5;
        // `wall_share` is the 1/3 that says the run is CPU-bound and that no
        // duty cycle will free the compute its operator hoped to release.
        let probe = duty_cycle(
            0.5,
            &[
                (WORK, 200_000),
                (IDLE, 0),
                (IDLE, 0),
                (HOST, 400_000),
                (WORK, 200_000),
                (IDLE, 0),
                (IDLE, 0),
            ],
        );
        let compute = probe.stats.compute_seconds;
        let observed = compute / (compute + probe.stats.idle_seconds);
        let wall_share = compute / probe.stats.wall_seconds;
        assert!((observed - 0.5).abs() < 1e-9, "observed {observed}");
        assert!(
            wall_share < observed,
            "wall {wall_share} vs observed {observed}"
        );
        assert!(
            (wall_share - 1.0 / 3.0).abs() < 1e-9,
            "wall_share {wall_share}"
        );
    }

    #[test]
    fn nothing_is_reported_before_the_first_accounted_window() {
        let probe = duty_cycle(0.5, &[(IDLE, 0)]);
        assert_eq!(probe.stats.compute_seconds, 0.0);
        assert_eq!(probe.stats.idle_seconds, 0.0);
        assert_eq!(probe.sleep_calls, 0);
    }

    #[test]
    fn a_disabled_limiter_reads_no_clock_and_calls_no_sleeper() {
        // The zero-overhead property at this level. `1.0` is how the C ABI
        // spells "no limit", and a replay of a full script under it must leave
        // the fake clock and the fake sleeper untouched.
        let probe = duty_cycle(
            1.0,
            &[
                (WORK, 200_000),
                (IDLE, 0),
                (HOST, 900_000),
                (WORK, 200_000),
                (RESET, 0),
                (IDLE, 0),
            ],
        );
        assert_eq!(probe.clock_reads, 0);
        assert_eq!(probe.sleep_calls, 0);
        assert_eq!(probe.stats.compute_seconds, 0.0);
        assert_eq!(probe.stats.idle_seconds, 0.0);
        assert_eq!(probe.stats.wall_seconds, 0.0);
        assert_eq!(probe.stats.requested_fraction, 1.0);
        assert!(!probe.stats.active);
    }

    #[test]
    fn an_out_of_range_duty_cycle_is_refused_at_the_c_boundary() {
        // Zero is not a spelling for pause: run control already has one.
        for fraction in [0.0_f32, -0.5, 1.5, f32::NAN, f32::INFINITY] {
            let mut probe = RetroDutyCycleProbe::default();
            // SAFETY: a null event pointer is legal for a zero-length script.
            let code = unsafe { retro_probe_duty_cycle(fraction, ptr::null(), 0, &mut probe) };
            assert_eq!(code, -1, "fraction {fraction} was accepted");
        }
        let mut probe = RetroDutyCycleProbe::default();
        // SAFETY: `probe` is a valid out-pointer; the script is deliberately malformed.
        let unknown = RetroDutyCycleEvent {
            kind: 99,
            micros: 0,
        };
        let code = unsafe { retro_probe_duty_cycle(0.5, &unknown, 1, &mut probe) };
        assert_eq!(code, -1);
        assert!(last_error().contains("event kind"));
    }

    #[test]
    fn every_training_owned_decode_in_the_rollout_goes_through_the_limiter() {
        // A source-level coverage assertion, because a new raw `llama_decode`
        // in the generation core would be a silent hole in the duty cycle
        // rather than a compile error: generation and scoring would keep
        // running unthrottled while the report said the limiter was active.
        // The two lines the helper itself owns are the only exemption; probe,
        // preflight and calibration decodes live in other files and stay
        // unthrottled on purpose. The core lives in retro_gen_core.cpp since the
        // engine extraction; retro_rollout.cpp keeps only wrappers and must
        // hold no decode of its own.
        let core = include_str!("../runtime/src/retro_gen_core.cpp");
        // `llama_process` is the extended-batch spelling of the same decode, so
        // it is scanned too: a raw call of either spelling outside the two
        // limiter helpers would be a silent hole in the duty cycle.
        let raw: Vec<&str> = core
            .lines()
            .filter(|line| {
                (line.contains("llama_decode(") || line.contains("llama_process("))
                    && !line.trim_start().starts_with("//")
            })
            .collect();
        assert_eq!(
            raw.len(),
            4,
            "raw decode/process calls outside gen_core::decode / decode_ext: {raw:#?}"
        );
        for line in &raw {
            assert!(
                line.contains("return llama_decode(ctx, batch);")
                    || line.contains("const int status = llama_decode(ctx, batch);")
                    || line
                        .contains("return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, &batch);")
                    || line.contains(
                        "const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, &batch);"
                    ),
                "unexpected raw decode: {line}"
            );
        }
        let rollout = include_str!("../runtime/src/retro_rollout.cpp");
        let rollout_raw: Vec<&str> = rollout
            .lines()
            .filter(|line| {
                (line.contains("llama_decode(") || line.contains("llama_process("))
                    && !line.trim_start().starts_with("//")
            })
            .collect();
        assert!(
            rollout_raw.is_empty(),
            "retro_rollout.cpp must delegate decodes to gen_core: {rollout_raw:#?}"
        );
    }

    #[test]
    fn c_struct_layouts_match_the_published_header() {
        // Keep these values in sync with runtime/include/retro_lora_train.h.
        // A layout drift can otherwise turn a valid Rust configuration into
        // unrelated C++ options without a compiler error.
        assert_eq!(size_of::<RetroLoraConfig>(), 40);
        assert_eq!(align_of::<RetroLoraConfig>(), 8);
        assert_eq!(offset_of!(RetroLoraConfig, target_patterns), 16);
        assert_eq!(offset_of!(RetroLoraConfig, dtype), 32);

        assert_eq!(size_of::<RetroTrainConfig>(), 176);
        assert_eq!(align_of::<RetroTrainConfig>(), 8);
        assert_eq!(offset_of!(RetroTrainConfig, generation_concurrency), 16);
        assert_eq!(offset_of!(RetroTrainConfig, kv_dtype), 24);
        assert_eq!(offset_of!(RetroTrainConfig, learning_rate), 36);
        assert_eq!(offset_of!(RetroTrainConfig, warmup_steps), 56);
        assert_eq!(offset_of!(RetroTrainConfig, device), 68);
        assert_eq!(offset_of!(RetroTrainConfig, chunked_ce_seq_chunk), 80);
        assert_eq!(
            offset_of!(RetroTrainConfig, chunked_ce_offload_logsoftmax),
            84
        );
        assert_eq!(offset_of!(RetroTrainConfig, checkpoint_every_n_layers), 88);
        // checkpoint_dtype lands in what was tail padding, so the struct size is
        // unchanged; the `size_of::<RetroTrainConfig>()` assertion guards this.
        assert_eq!(offset_of!(RetroTrainConfig, checkpoint_dtype), 92);
        // require_gpu_resident opened a new 8-byte tail slot (one bool plus
        // padding), increasing the struct size from 96 to 104 bytes.
        assert_eq!(offset_of!(RetroTrainConfig, require_gpu_resident), 96);
        // generation_batch lands in that slot's remaining padding, so the
        // struct size is unchanged; the size assertion guards this layout.
        assert_eq!(offset_of!(RetroTrainConfig, generation_batch), 100);
        // The slot ending at 104 was already full, so the shuffle pair opens
        // two more: the bool alone in one because the u64 seed behind it forces
        // 8-byte alignment. That is what took the struct from 104 to 120.
        assert_eq!(offset_of!(RetroTrainConfig, shuffle_dataset), 104);
        // `optimizer` costs nothing: the three bytes after that bool were
        // padding the u64 seed forced anyway, so it lands at 108 and
        // `shuffle_seed` does not move. `trainable` did open a new tail slot,
        // which is what took the struct from 120 to 128.
        assert_eq!(offset_of!(RetroTrainConfig, optimizer), 108);
        assert_eq!(offset_of!(RetroTrainConfig, shuffle_seed), 112);
        assert_eq!(offset_of!(RetroTrainConfig, trainable), 120);
        // The optimizer-specific tail. `trainable` left four bytes of padding
        // at 124, which is where the first Muon float lands, so the block below
        // costs what it holds and nothing for alignment until `muon_nesterov`,
        // whose three trailing bytes the next i32 would have forced anyway.
        assert_eq!(offset_of!(RetroTrainConfig, muon_momentum), 124);
        assert_eq!(offset_of!(RetroTrainConfig, muon_ns_steps), 136);
        assert_eq!(offset_of!(RetroTrainConfig, muon_nesterov), 140);
        assert_eq!(offset_of!(RetroTrainConfig, gefen_variant), 144);
        assert_eq!(offset_of!(RetroTrainConfig, gefen_eps), 160);
        // master_weights lands in the tail padding after gefen_eps, so the
        // struct size is unchanged.
        assert_eq!(offset_of!(RetroTrainConfig, master_weights), 164);
        // master_weights filled the slot ending at 168, so generation_kv_dtype
        // opens a new one: the struct goes from 168 to 176, four bytes of it
        // tail padding.
        assert_eq!(offset_of!(RetroTrainConfig, generation_kv_dtype), 168);

        assert_eq!(size_of::<RetroScoringStats>(), 48);
        assert_eq!(align_of::<RetroScoringStats>(), 8);
        assert_eq!(offset_of!(RetroScoringStats, scored_positions), 32);

        assert_eq!(size_of::<RetroGenerationStats>(), 56);
        assert_eq!(align_of::<RetroGenerationStats>(), 8);
        assert_eq!(offset_of!(RetroGenerationStats, prefilled_tokens), 24);
        assert_eq!(offset_of!(RetroGenerationStats, reused_tokens), 32);

        // The engine config is versioned by struct_size, so its layout is part
        // of the ABI like every other caller-filled struct here. `kv_unified`
        // was appended after `verbose` as a u32: callers compiled against the
        // older 40-byte layout pass struct_size 40, the runtime reads only
        // that prefix and keeps the default.
        assert_eq!(size_of::<RetroEngineConfig>(), 44);
        assert_eq!(align_of::<RetroEngineConfig>(), 4);
        assert_eq!(offset_of!(RetroEngineConfig, struct_size), 0);
        assert_eq!(offset_of!(RetroEngineConfig, n_ctx), 4);
        assert_eq!(offset_of!(RetroEngineConfig, n_seq_max), 8);
        assert_eq!(offset_of!(RetroEngineConfig, n_batch), 12);
        assert_eq!(offset_of!(RetroEngineConfig, n_ubatch), 16);
        assert_eq!(offset_of!(RetroEngineConfig, kv_dtype), 20);
        assert_eq!(offset_of!(RetroEngineConfig, flash_attn), 24);
        assert_eq!(offset_of!(RetroEngineConfig, device), 28);
        assert_eq!(offset_of!(RetroEngineConfig, threads), 32);
        assert_eq!(offset_of!(RetroEngineConfig, verbose), 36);
        assert_eq!(offset_of!(RetroEngineConfig, kv_unified), 40);

        // These introspection structs are caller-allocated C values, so their
        // layout is part of the ABI. The fixed name arrays remain at the end so
        // numeric fields can be added without moving either string.
        assert_eq!(size_of::<RetroModelInfo>(), 232);
        assert_eq!(align_of::<RetroModelInfo>(), 8);
        assert_eq!(offset_of!(RetroModelInfo, n_params), 64);
        assert_eq!(offset_of!(RetroModelInfo, architecture), 99);
        assert_eq!(offset_of!(RetroModelInfo, dominant_weight_type), 163);

        assert_eq!(size_of::<RetroModelCapabilities>(), 8);
        assert_eq!(align_of::<RetroModelCapabilities>(), 4);
        assert_eq!(
            offset_of!(RetroModelCapabilities, shared_prefix_packed_training),
            4
        );

        assert_eq!(size_of::<RetroPreflightSummary>(), 88);
        assert_eq!(align_of::<RetroPreflightSummary>(), 8);
        assert_eq!(
            offset_of!(RetroPreflightSummary, active_device_fallback_nodes),
            8
        );
        assert_eq!(offset_of!(RetroPreflightSummary, graph_fingerprint), 16);

        assert_eq!(size_of::<RetroTransferRates>(), 56);
        assert_eq!(align_of::<RetroTransferRates>(), 8);
        assert_eq!(
            offset_of!(RetroTransferRates, pinned_h2d_bytes_per_second),
            16
        );
        assert_eq!(offset_of!(RetroTransferRates, pinned_is_pageable), 48);

        assert_eq!(size_of::<RetroMemoryReport>(), 216);
        assert_eq!(align_of::<RetroMemoryReport>(), 8);
        assert_eq!(offset_of!(RetroMemoryReport, trainable_parameter_bytes), 48);
        // The three placement booleans fit the padding the single `lora_on_host`
        // once occupied: `device_bytes` is the assertion that pins the layout.
        assert_eq!(
            offset_of!(RetroMemoryReport, trainable_parameters_are_model_subset),
            72
        );
        assert_eq!(offset_of!(RetroMemoryReport, base_trainable_on_host), 74);
        assert_eq!(offset_of!(RetroMemoryReport, device_bytes), 80);
        assert_eq!(offset_of!(RetroMemoryReport, device_memory_samples), 136);
        assert_eq!(offset_of!(RetroMemoryReport, checkpoint_count), 144);
        assert_eq!(
            offset_of!(RetroMemoryReport, checkpoint_total_span_nodes),
            208
        );

        assert_eq!(size_of::<RetroSamplingParams>(), 16);
        assert_eq!(offset_of!(RetroSamplingParams, max_new_tokens), 8);
        assert_eq!(size_of::<RetroSftDataset>(), 32);

        // The step-wise API structs: caller-filled rows are versioned by
        // struct_size like every other caller-filled struct, and the output
        // structs are pinned so the runtime and this crate cannot drift.
        assert_eq!(size_of::<RetroStepRow>(), 64);
        assert_eq!(align_of::<RetroStepRow>(), 8);
        assert_eq!(offset_of!(RetroStepRow, struct_size), 0);
        assert_eq!(offset_of!(RetroStepRow, request), 4);
        assert_eq!(offset_of!(RetroStepRow, slot), 8);
        assert_eq!(offset_of!(RetroStepRow, tokens), 16);
        assert_eq!(offset_of!(RetroStepRow, n), 24);
        assert_eq!(offset_of!(RetroStepRow, pos0), 28);
        assert_eq!(offset_of!(RetroStepRow, sample), 32);
        assert_eq!(offset_of!(RetroStepRow, sampling), 36);
        assert_eq!(offset_of!(RetroStepRow, n_top_logprobs), 52);
        assert_eq!(offset_of!(RetroStepRow, adapter), 56);
        assert_eq!(size_of::<RetroAdapterRowsInfo>(), 32);
        assert_eq!(align_of::<RetroAdapterRowsInfo>(), 8);
        assert_eq!(offset_of!(RetroAdapterRowsInfo, max_rank), 8);
        assert_eq!(offset_of!(RetroAdapterRowsInfo, stacked_bytes), 16);
        assert_eq!(offset_of!(RetroAdapterRowsInfo, unpadded_bytes), 24);
        assert_eq!(size_of::<RetroStepOutRow>(), 20);
        assert_eq!(align_of::<RetroStepOutRow>(), 4);
        assert_eq!(offset_of!(RetroStepOutRow, struct_size), 0);
        assert_eq!(offset_of!(RetroStepOutRow, token), 4);
        assert_eq!(offset_of!(RetroStepOutRow, logprob), 8);
        assert_eq!(offset_of!(RetroStepOutRow, n_top), 12);
        assert_eq!(offset_of!(RetroStepOutRow, forced), 16);
        assert_eq!(offset_of!(RetroStepOutRow, sampled), 17);

        // The v2 step structs, pinned the same way: sizes and offsets computed
        // from the C layout (64-bit, 8-byte pointers).
        assert_eq!(size_of::<RetroSamplingParamsV2>(), 72);
        assert_eq!(align_of::<RetroSamplingParamsV2>(), 8);
        assert_eq!(offset_of!(RetroSamplingParamsV2, struct_size), 0);
        assert_eq!(offset_of!(RetroSamplingParamsV2, temperature), 4);
        assert_eq!(offset_of!(RetroSamplingParamsV2, top_p), 8);
        assert_eq!(offset_of!(RetroSamplingParamsV2, max_new_tokens), 12);
        assert_eq!(offset_of!(RetroSamplingParamsV2, seed), 16);
        assert_eq!(offset_of!(RetroSamplingParamsV2, top_k), 20);
        assert_eq!(offset_of!(RetroSamplingParamsV2, min_p), 24);
        assert_eq!(offset_of!(RetroSamplingParamsV2, repeat_penalty), 28);
        assert_eq!(offset_of!(RetroSamplingParamsV2, frequency_penalty), 32);
        assert_eq!(offset_of!(RetroSamplingParamsV2, presence_penalty), 36);
        assert_eq!(offset_of!(RetroSamplingParamsV2, penalty_window), 40);
        assert_eq!(offset_of!(RetroSamplingParamsV2, n_logit_bias), 44);
        assert_eq!(offset_of!(RetroSamplingParamsV2, logit_bias_ids), 48);
        assert_eq!(offset_of!(RetroSamplingParamsV2, logit_bias_values), 56);
        assert_eq!(offset_of!(RetroSamplingParamsV2, grammar), 64);
        assert_eq!(size_of::<RetroStepRowV2>(), 128);
        assert_eq!(align_of::<RetroStepRowV2>(), 8);
        assert_eq!(offset_of!(RetroStepRowV2, struct_size), 0);
        assert_eq!(offset_of!(RetroStepRowV2, request), 4);
        assert_eq!(offset_of!(RetroStepRowV2, slot), 8);
        assert_eq!(offset_of!(RetroStepRowV2, tokens), 16);
        assert_eq!(offset_of!(RetroStepRowV2, n), 24);
        assert_eq!(offset_of!(RetroStepRowV2, pos0), 28);
        assert_eq!(offset_of!(RetroStepRowV2, sample), 32);
        assert_eq!(offset_of!(RetroStepRowV2, sampling), 40);
        assert_eq!(offset_of!(RetroStepRowV2, n_top_logprobs), 112);
        assert_eq!(offset_of!(RetroStepRowV2, want_logprob), 116);
        assert_eq!(offset_of!(RetroStepRowV2, adapter), 120);
        assert_eq!(size_of::<RetroStepOutRowV2>(), 24);
        assert_eq!(align_of::<RetroStepOutRowV2>(), 4);
        assert_eq!(offset_of!(RetroStepOutRowV2, struct_size), 0);
        assert_eq!(offset_of!(RetroStepOutRowV2, token), 4);
        assert_eq!(offset_of!(RetroStepOutRowV2, logprob), 8);
        assert_eq!(offset_of!(RetroStepOutRowV2, n_top), 12);
        assert_eq!(offset_of!(RetroStepOutRowV2, forced), 16);
        assert_eq!(offset_of!(RetroStepOutRowV2, sampled), 17);
        assert_eq!(offset_of!(RetroStepOutRowV2, sampled_logprob), 20);
        assert_eq!(RETRO_TOOL_GRAMMAR_UNAVAILABLE, -2);
        assert_eq!(size_of::<RetroEngineMemoryInfo>(), 40);
        assert_eq!(align_of::<RetroEngineMemoryInfo>(), 8);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, struct_size), 0);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, n_slots), 4);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, cells_total), 8);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, cells_used), 16);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, kv_bytes), 24);
        assert_eq!(offset_of!(RetroEngineMemoryInfo, recurrent_bytes), 32);
        // `n_topk` is appended, and lands in the tail padding the 40-byte layout
        // already carried, so the struct keeps its size. The offset assertion is
        // what pins that; the size alone would not have noticed the field.
        assert_eq!(size_of::<RetroWeightedDataset>(), 40);
        assert_eq!(offset_of!(RetroWeightedDataset, n_ctx), 32);
        assert_eq!(offset_of!(RetroWeightedDataset, n_topk), 36);
        assert_eq!(size_of::<RetroPackedSequenceBatch>(), 72);
        assert_eq!(offset_of!(RetroPackedSequenceBatch, n_sequences), 64);
        assert_eq!(offset_of!(RetroPackedSequenceBatch, n_topk), 68);

        // The inventory contract. The name array is the first field so its
        // length is what the offsets below pin: a header that widened
        // RETRO_TENSOR_NAME_MAX without this crate following would move every
        // numeric field and be read as garbage shapes, not as a size mismatch.
        assert_eq!(size_of::<RetroTensorDesc>(), 248);
        assert_eq!(align_of::<RetroTensorDesc>(), 8);
        assert_eq!(offset_of!(RetroTensorDesc, ne), 128);
        assert_eq!(offset_of!(RetroTensorDesc, n_elements), 160);
        assert_eq!(offset_of!(RetroTensorDesc, storage_id), 176);
        assert_eq!(offset_of!(RetroTensorDesc, type_name), 184);

        // The optimizer slot contract, pinned the same way and for the same
        // reason: the two name arrays come first, so a header that resized
        // either one without this crate following would move `ne` and be read
        // as a shape rather than as a size mismatch.
        assert_eq!(size_of::<RetroOptimizerSlot>(), 304);
        assert_eq!(align_of::<RetroOptimizerSlot>(), 8);
        assert_eq!(offset_of!(RetroOptimizerSlot, slot), 128);
        assert_eq!(offset_of!(RetroOptimizerSlot, type_name), 192);
        assert_eq!(offset_of!(RetroOptimizerSlot, ne), 256);
        assert_eq!(offset_of!(RetroOptimizerSlot, n_elements), 288);
        assert_eq!(offset_of!(RetroOptimizerSlot, n_bytes), 296);

        // `graph_ready` sits in the padding beside `has_persistent_state` and the three
        // AdamW coefficients are a tail addition, so nothing before them moved:
        // `optimizer` at 12 pins that, and the size catches a header resized on
        // one side.
        assert_eq!(offset_of!(RetroOptimizerState, has_persistent_state), 8);
        assert_eq!(offset_of!(RetroOptimizerState, graph_ready), 9);
        assert_eq!(offset_of!(RetroOptimizerState, optimizer), 12);
        assert_eq!(offset_of!(RetroOptimizerState, last_learning_rate), 48);
        assert_eq!(offset_of!(RetroOptimizerState, adamw_beta1), 52);
        assert_eq!(offset_of!(RetroOptimizerState, adamw_eps), 60);
        // The variant opens a new 8-byte tail slot, and the per-optimizer
        // coefficients fill it and four more.
        assert_eq!(offset_of!(RetroOptimizerState, gefen_variant), 64);
        assert_eq!(offset_of!(RetroOptimizerState, muon_momentum), 68);
        assert_eq!(offset_of!(RetroOptimizerState, muon_nesterov), 84);
        assert_eq!(offset_of!(RetroOptimizerState, gefen_beta1), 88);
        assert_eq!(offset_of!(RetroOptimizerState, gefen_block_size), 100);
        assert_eq!(size_of::<RetroOptimizerState>(), 104);

        assert_eq!(offset_of!(RetroDecisionPrompt, tokens), 0);
        assert_eq!(offset_of!(RetroDecisionPrompt, n_tokens), 8);
        assert_eq!(offset_of!(RetroDecisionPrompt, reads), 16);
        assert_eq!(offset_of!(RetroDecisionPrompt, n_reads), 24);
        assert_eq!(offset_of!(RetroDecisionPrompt, column), 32);
        assert_eq!(offset_of!(RetroDecisionPrompt, pointer), 36);
        assert_eq!(size_of::<RetroDecisionPrompt>(), 40);

        assert_eq!(offset_of!(RetroGroupedBatch, tokens), 0);
        assert_eq!(offset_of!(RetroGroupedBatch, positions), 8);
        assert_eq!(offset_of!(RetroGroupedBatch, seq_ids), 16);
        assert_eq!(offset_of!(RetroGroupedBatch, outputs), 24);
        assert_eq!(offset_of!(RetroGroupedBatch, n_tokens), 32);
        assert_eq!(offset_of!(RetroGroupedBatch, n_groups), 40);
        assert_eq!(offset_of!(RetroGroupedBatch, n_opt_max), 44);
        assert_eq!(offset_of!(RetroGroupedBatch, rows), 48);
        assert_eq!(offset_of!(RetroGroupedBatch, columns), 56);
        assert_eq!(offset_of!(RetroGroupedBatch, targets), 64);
        assert_eq!(offset_of!(RetroGroupedBatch, inv_temperature), 72);
        assert_eq!(offset_of!(RetroGroupedBatch, accumulation_steps), 76);
        assert_eq!(size_of::<RetroGroupedBatch>(), 80);
    }

    #[test]
    fn decision_entry_points_refuse_a_null_engine() {
        let mut needed = 0_usize;
        // SAFETY: a null engine is refused before any pointer is read.
        let code =
            unsafe { retro_engine_decision_info(ptr::null_mut(), ptr::null_mut(), 0, &mut needed) };
        assert_eq!((code, last_error().as_str()), (-1, "engine is required"));
        let mut score = 0.0_f32;
        // SAFETY: as above.
        let code = unsafe {
            retro_engine_decide(
                ptr::null_mut(),
                RETRO_DECIDE_MARKER_COLUMN,
                ptr::null(),
                0,
                1,
                &mut score,
                1,
            )
        };
        assert_eq!((code, last_error().as_str()), (-1, "engine is required"));
    }

    /// A named template is rendered over its context alone, keys in the
    /// order given: no messages, no bos/eos token.
    #[test]
    fn a_template_renders_over_its_context_alone() {
        let source = CString::new(
            "{{ id }}|{% for key, value in state.items() %}{{ key }}={{ value }};{% endfor %}\
             |{{ bos_token is defined }}",
        )
        .unwrap();
        let context = CString::new(r#"{"id": "q", "state": {"z": 1, "a": 2}}"#).unwrap();
        let mut needed = 0_usize;
        // SAFETY: both strings outlive the synchronous calls; the second call
        // gets a buffer of the size the first one reported plus the NUL.
        let code = unsafe {
            retro_template_render_context(
                source.as_ptr(),
                context.as_ptr(),
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        assert_eq!(code, 0, "{}", last_error());
        let mut buffer = vec![0_u8; needed + 1];
        let code = unsafe {
            retro_template_render_context(
                source.as_ptr(),
                context.as_ptr(),
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut needed,
            )
        };
        assert_eq!(code, 0, "{}", last_error());
        assert_eq!(
            std::str::from_utf8(&buffer[..needed]).unwrap(),
            "q|z=1;a=2;|False"
        );

        let not_object = CString::new("[1]").unwrap();
        // SAFETY: as above.
        let code = unsafe {
            retro_template_render_context(
                source.as_ptr(),
                not_object.as_ptr(),
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        assert_eq!(
            (code, last_error().as_str()),
            (-1, "the template context must be a JSON object")
        );
    }

    #[test]
    fn v2_step_structs_default_to_a_null_grammar_and_zeroed_fields() {
        let sampling = RetroSamplingParamsV2::default();
        assert_eq!(sampling.struct_size, 0);
        assert!(sampling.grammar.is_null());
        assert!(sampling.logit_bias_ids.is_null());
        assert!(sampling.logit_bias_values.is_null());
        let row = RetroStepRowV2::default();
        assert_eq!(row.struct_size, 0);
        assert!(row.tokens.is_null());
        assert!(!row.sample);
        assert_eq!(row.adapter, 0);
        assert!(row.sampling.grammar.is_null());
        let out = RetroStepOutRowV2::default();
        assert_eq!(out.struct_size, 0);
        assert_eq!(out.token, 0);
        assert_eq!(out.sampled, 0);
        assert_eq!(out.sampled_logprob, 0.0);
    }

    #[test]
    fn grammar_from_json_schema_round_trips_and_rejects_invalid_input() {
        let grammar = Grammar::from_json_schema(
            r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#,
        )
        .expect("a small object schema converts");
        let gbnf = grammar.gbnf();
        assert!(!gbnf.is_empty());
        assert!(gbnf.contains("root"), "GBNF names its root rule: {gbnf}");

        assert!(Grammar::from_json_schema("not json at all").is_err());
        assert!(Grammar::from_json_schema(r#"{"type":"definitely-not-a-type"}"#).is_err());
        assert!(Grammar::from_gbnf("").is_err());
    }

    #[test]
    fn grammar_from_gbnf_keeps_the_text_and_json_object_is_any_json() {
        let grammar = Grammar::from_gbnf("root ::= \"a\"").expect("non-empty GBNF is accepted");
        assert_eq!(grammar.gbnf(), "root ::= \"a\"");

        let json = Grammar::json_object();
        let gbnf = json.gbnf();
        // The embedded grammars/json.gbnf names its value rule.
        assert!(
            gbnf.contains("root"),
            "any-JSON grammar has a root rule: {gbnf}"
        );
        assert!(
            gbnf.contains("value"),
            "any-JSON grammar has a value rule: {gbnf}"
        );
    }

    /// `retro_read_tensor_inventory` on its error paths only, for the same
    /// reason as the model-info test below: no model is loaded, so its contract
    /// belongs in the ABI lane. That contract is the required out-pointers, and
    /// a buffer shorter than the count refused rather than truncated.
    #[test]
    fn tensor_inventory_requires_its_out_pointers() {
        let mut version = 0_u32;
        let mut count = usize::MAX;
        let missing = CString::new("does-not-exist.gguf").expect("path");
        unsafe {
            assert_eq!(
                retro_read_tensor_inventory(
                    ptr::null(),
                    &mut version,
                    ptr::null_mut(),
                    0,
                    &mut count
                ),
                -1
            );
            assert_eq!(last_error(), "model_path is required");

            assert_eq!(
                retro_read_tensor_inventory(
                    missing.as_ptr(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    &mut count
                ),
                -1
            );
            assert_eq!(last_error(), "out_version and out_count are required");

            assert_eq!(
                retro_read_tensor_inventory(
                    missing.as_ptr(),
                    &mut version,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut()
                ),
                -1
            );
            assert_eq!(last_error(), "out_version and out_count are required");

            // A path that does not resolve fails on the load, after the version
            // has been published: a caller learns the schema it would have got.
            assert_eq!(
                retro_read_tensor_inventory(
                    missing.as_ptr(),
                    &mut version,
                    ptr::null_mut(),
                    0,
                    &mut count
                ),
                -1
            );
        }
        assert_eq!(version, TENSOR_INVENTORY_VERSION);
        assert_eq!(count, 0);
    }

    /// `retro_read_model_info` on its error paths only: no model is loaded, so
    /// this belongs in the `abi` lane alongside the other contract checks. A
    /// blank path and a null out-pointer must both fail with a message rather
    /// than write through a null pointer or half-fill the struct.
    #[test]
    fn model_info_rejects_blank_paths_and_null_outputs() {
        let mut info = RetroModelInfo::default();
        let empty = CString::new("").expect("empty path");
        let missing = CString::new("does-not-exist.gguf").expect("path");
        unsafe {
            assert_eq!(retro_read_model_info(ptr::null(), 0, &mut info), -1);
            assert_eq!(last_error(), "model_path is required");

            assert_eq!(retro_read_model_info(empty.as_ptr(), 0, &mut info), -1);
            assert_eq!(last_error(), "model_path is required");

            assert_eq!(
                retro_read_model_info(missing.as_ptr(), 0, ptr::null_mut()),
                -1
            );
            assert_eq!(last_error(), "out_info is required");
        }
        // A rejected call must not have written anything.
        assert_eq!(info.n_layer, 0);
        assert_eq!(info.architecture[0], 0);
    }

    #[test]
    fn backend_list_obeys_the_two_pass_string_contract() {
        let mut needed = usize::MAX;
        unsafe {
            assert_eq!(retro_backend_list(ptr::null_mut(), 0, &mut needed), 0);
        }
        assert_ne!(needed, usize::MAX);

        let mut too_small = vec![b'x'; needed];
        let mut reported = 0;
        unsafe {
            assert_eq!(
                retro_backend_list(
                    too_small.as_mut_ptr().cast(),
                    too_small.len(),
                    &mut reported
                ),
                -2
            );
        }
        assert_eq!(reported, needed);
        assert!(too_small.iter().all(|&byte| byte == b'x'));
        assert_eq!(last_error(), "output buffer is too small");

        let mut buffer = vec![0_u8; needed + 1];
        unsafe {
            assert_eq!(
                retro_backend_list(buffer.as_mut_ptr().cast(), buffer.len(), &mut reported),
                0
            );
        }
        assert_eq!(reported, needed);
        assert_eq!(buffer[needed], 0);
        assert_eq!(
            CStr::from_bytes_with_nul(&buffer).unwrap().to_bytes().len(),
            needed
        );

        unsafe {
            assert_eq!(retro_backend_list(ptr::null_mut(), 0, ptr::null_mut()), -1);
        }
        assert_eq!(last_error(), "out_n_bytes is required");
    }

    #[test]
    fn gpu_runtime_probe_never_reports_success_without_a_registered_gpu() {
        let has_gpu = unsafe {
            let mut needed = 0;
            if retro_backend_list(ptr::null_mut(), 0, &mut needed) != 0 || needed == 0 {
                false
            } else {
                let mut buffer = vec![0_u8; needed + 1];
                let mut reported = 0;
                retro_backend_list(buffer.as_mut_ptr().cast(), buffer.len(), &mut reported) == 0
                    && CStr::from_bytes_with_nul(&buffer)
                        .map(|list| {
                            list.to_string_lossy()
                                .lines()
                                .any(|line| line.starts_with("gpu\t"))
                        })
                        .unwrap_or(false)
            }
        };
        let probe = unsafe { retro_gpu_runtime_probe() };
        if !has_gpu {
            assert_eq!(probe, -1);
        }
    }

    #[test]
    fn null_trainer_is_rejected_consistently_by_export_families() {
        let mut count = 0_usize;
        let mut token = 0_i32;
        let mut width = 0_u32;
        let mut step = 0_u64;
        let mut metrics = RetroTrainMetrics::default();
        let mut eval_metrics = RetroEvalMetrics::default();
        let mut timing = RetroOptimizerTiming::default();
        let mut memory = RetroOptimizerMemory::default();
        let mut scoring = RetroScoringStats::default();
        let mut generation = RetroGenerationStats::default();
        let mut duty_cycle = RetroDutyCycleStats::default();
        let mut state = RetroOptimizerState::default();
        let mut slot = RetroOptimizerSlot::default();
        let mut tensor = RetroTensorDesc::default();
        let mut bytes = 0_usize;
        let text = CString::new("x").unwrap();
        let sampling = RetroSamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 1,
            seed: 0,
        };

        let mut is_eog = false;

        unsafe {
            assert_null_trainer(retro_trainer_load_lora(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_vocab_size(ptr::null_mut(), &mut width));
            assert_null_trainer(retro_trainer_is_eog_token(ptr::null_mut(), 0, &mut is_eog));
            assert_null_trainer(retro_trainer_context_size(ptr::null_mut(), &mut width));
            assert_null_trainer(retro_trainer_set_lora_enabled(ptr::null_mut(), false));
            assert_null_trainer(retro_trainer_train_tokens(
                ptr::null_mut(),
                ptr::null(),
                0,
                &mut metrics,
            ));
            assert_null_trainer(retro_trainer_eval_sft(
                ptr::null_mut(),
                ptr::null(),
                &mut eval_metrics,
            ));
            assert_null_trainer(retro_trainer_generate_continuous_batch(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut count,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_set_reasoning_budget(ptr::null_mut(), -1));
            assert_null_trainer(retro_trainer_tokenize_text(
                ptr::null_mut(),
                text.as_ptr(),
                ptr::null_mut(),
                0,
                &mut count,
            ));
            assert_null_trainer(retro_trainer_detokenize(
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_create_lora(ptr::null_mut(), ptr::null()));
            assert_null_trainer(retro_trainer_format_chat(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                0,
                false,
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_format_chat_messages(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                false,
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            let mut supports = false;
            assert_null_trainer(retro_trainer_chat_template_supports_tools(
                ptr::null_mut(),
                &mut supports,
            ));
            assert_null_trainer(retro_trainer_tool_call_parser(
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            // retro_chat_parse_assistant takes no trainer, so the null it has
            // to refuse is the parser blob.
            assert_eq!(
                retro_chat_parse_assistant(
                    ptr::null(),
                    0,
                    text.as_ptr(),
                    ptr::null_mut(),
                    0,
                    &mut bytes,
                ),
                -1
            );
            assert_eq!(last_error(), "tool-call parser blob is required");
            // Same refusal on the partial entry point.
            assert_eq!(
                retro_chat_parse_assistant_partial(
                    ptr::null(),
                    0,
                    text.as_ptr(),
                    ptr::null_mut(),
                    0,
                    &mut bytes,
                ),
                -1
            );
            assert_eq!(last_error(), "tool-call parser blob is required");
            assert_null_trainer(retro_trainer_describe_lora(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_generate_batch(
                ptr::null_mut(),
                ptr::null(),
                0,
                &sampling,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut count,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_score_tokens(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_score_token_suffix(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_top_logprobs_suffix(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_score_token_suffix_batch(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                &mut count,
            ));
            assert_null_trainer(retro_trainer_hidden_states(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_hidden_size(ptr::null_mut(), &mut width));
            assert_null_trainer(retro_trainer_score_token_suffix_and_hidden_states(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_train_sft(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                &mut metrics,
                None,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_train_weighted(
                ptr::null_mut(),
                ptr::null(),
                0,
                &mut metrics,
                None,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_train_packed_sequences(
                ptr::null_mut(),
                ptr::null(),
                0,
                1,
                &mut metrics,
                None,
                ptr::null_mut(),
            ));
            assert_null_trainer(retro_trainer_train_preflight(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_backend_report(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_capability_report(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_lora_candidate_targets(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_optimizer_timing(ptr::null_mut(), &mut timing));
            assert_null_trainer(retro_trainer_optimizer_memory(ptr::null_mut(), &mut memory));
            assert_null_trainer(retro_trainer_scoring_stats(ptr::null_mut(), &mut scoring));
            assert_null_trainer(retro_trainer_set_max_gpu_duty_cycle(ptr::null_mut(), 0.5));
            assert_null_trainer(retro_trainer_duty_cycle_stats(
                ptr::null_mut(),
                &mut duty_cycle,
            ));
            assert_null_trainer(retro_trainer_generation_stats(
                ptr::null_mut(),
                &mut generation,
            ));
            assert_null_trainer(retro_trainer_advance_scheduler_steps(
                ptr::null_mut(),
                0,
                &mut step,
            ));
            assert_null_trainer(retro_trainer_set_learning_rate(ptr::null_mut(), 1.0e-4));
            assert_null_trainer(retro_trainer_save_lora(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_optimizer_state(ptr::null_mut(), &mut state));
            assert_null_trainer(retro_trainer_restore_optimizer_state(
                ptr::null_mut(),
                &state,
            ));
            assert_null_trainer(retro_trainer_set_resume_point(ptr::null_mut(), 0));
            assert_null_trainer(retro_trainer_prepare_optimizer(ptr::null_mut()));
            assert_null_trainer(retro_trainer_marked_parameter_count(
                ptr::null_mut(),
                &mut count,
            ));
            assert_null_trainer(retro_trainer_marked_parameter_info(
                ptr::null_mut(),
                0,
                &mut tensor,
            ));
            assert_null_trainer(retro_trainer_marked_parameter_read(
                ptr::null_mut(),
                0,
                0,
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_parameter_gradient_info(
                ptr::null_mut(),
                0,
                &mut tensor,
            ));
            assert_null_trainer(retro_trainer_parameter_gradient_read(
                ptr::null_mut(),
                0,
                0,
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_state_slot_count(
                ptr::null_mut(),
                &mut count,
                &mut count,
            ));
            assert_null_trainer(retro_trainer_state_slot_info(
                ptr::null_mut(),
                SLOT_SCOPE_PARAMETER,
                0,
                &mut slot,
            ));
            assert_null_trainer(retro_trainer_state_slot_read(
                ptr::null_mut(),
                SLOT_SCOPE_PARAMETER,
                0,
                0,
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_state_slot_write(
                ptr::null_mut(),
                SLOT_SCOPE_SHARED,
                text.as_ptr(),
                text.as_ptr(),
                0,
                ptr::null(),
                0,
            ));
            assert_null_trainer(retro_trainer_save_trainable(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_load_trainable(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_save_model(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_rng_state(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_set_rng_state(ptr::null_mut(), text.as_ptr()));
            assert_null_trainer(retro_trainer_model_signature(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut bytes,
            ));
            assert_null_trainer(retro_trainer_eos_token(ptr::null_mut(), &mut token));
        }
    }

    #[test]
    fn engine_creation_rejects_a_truncated_config_before_reading_its_fields() {
        let path = CString::new("definitely-not-a-model.gguf").unwrap();
        let header = Box::new(size_of::<u32>() as u32);
        // SAFETY: the allocation contains the complete struct_size prefix. The
        // versioned ABI must reject it before reading any subsequent field.
        let engine = unsafe { retro_engine_new(path.as_ptr(), (&*header as *const u32).cast()) };
        assert!(engine.is_null());
        assert!(last_error().contains("struct_size is too small"));
    }

    #[test]
    fn engine_creation_validates_native_dimensions_before_loading_a_model() {
        let path = CString::new("definitely-not-a-model.gguf").unwrap();
        let base = RetroEngineConfig::default();
        for config in [
            RetroEngineConfig {
                n_ctx: u32::MAX,
                ..base
            },
            RetroEngineConfig {
                n_batch: u32::MAX,
                ..base
            },
            RetroEngineConfig {
                n_ubatch: u32::MAX,
                n_batch: u32::MAX,
                ..base
            },
            RetroEngineConfig {
                threads: u32::MAX,
                ..base
            },
            RetroEngineConfig {
                n_seq_max: 257,
                ..base
            },
            RetroEngineConfig {
                n_ctx: 1,
                n_seq_max: 2,
                ..base
            },
            RetroEngineConfig {
                n_seq_max: 65,
                n_batch: 64,
                n_ubatch: 64,
                ..base
            },
        ] {
            // SAFETY: both pointers remain valid for this synchronous call.
            let engine = unsafe { retro_engine_new(path.as_ptr(), &config) };
            assert!(engine.is_null());
            let error = last_error();
            assert!(
                error.contains("int32_t") || error.contains("n_seq_max"),
                "{error}"
            );
        }
    }

    #[test]
    fn trainer_creation_validates_config_before_loading_a_model() {
        let path = CString::new("definitely-not-a-model.gguf").unwrap();
        let invalid = RetroTrainConfig {
            n_ctx: 0,
            n_batch: 128,
            n_ubatch: 32,
            n_seq_max: 1,
            generation_concurrency: 1,
            fast_generation_context: false,
            kv_dtype: 0,
            threads: 0,
            epochs: 1,
            learning_rate: 1.0e-4,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            lr_scheduler: 0,
            warmup_steps: 0,
            verbose: false,
            device: 1,
            chunked_cross_entropy: false,
            chunked_ce_tiles: 8,
            chunked_ce_seq_chunk: 0,
            chunked_ce_offload_logsoftmax: false,
            gradient_checkpointing: false,
            checkpoint_every_n_layers: 1,
            checkpoint_dtype: 0,
            require_gpu_resident: false,
            generation_batch: 0,
            shuffle_dataset: true,
            optimizer: 0,
            shuffle_seed: 42,
            trainable: 0,
            muon_momentum: 0.0,
            muon_ns_epsilon: 0.0,
            muon_fallback_learning_rate: 0.0,
            muon_ns_steps: 0,
            muon_nesterov: true,
            gefen_variant: RETRO_GEFEN_SHARED_V,
            gefen_block_size: 0,
            gefen_beta1: 0.0,
            gefen_beta2: 0.0,
            gefen_eps: 0.0,
            master_weights: 0,
            generation_kv_dtype: 0,
        };
        unsafe {
            assert!(retro_trainer_new(path.as_ptr(), &invalid).is_null());
        }
        assert_eq!(last_error(), "n_ctx must be greater than zero");

        // A null config selects the C++ defaults. It must get as far as model
        // loading, rather than failing a default-value validation check.
        unsafe {
            assert!(retro_trainer_new(path.as_ptr(), ptr::null()).is_null());
        }
        assert!(!last_error().contains("must be greater than zero"));
    }
}

// Exercise build-time platform rules in the normal fast lane.
#[cfg(test)]
#[expect(dead_code)]
#[path = "../build/backend_selection.rs"]
mod backend_selection;
