//! Raw bindings to the C++ runtime (`runtime/include/retro_lora_train.h`).
//!
//! Every `#[repr(C)]` struct here mirrors the published header, which
//! `c_struct_layouts_match_the_published_header` checks field by field.
//! `build.rs` compiles the vendored llama.cpp fork and the runtime for the
//! backends selected by Cargo features (`metal`, `vulkan`, `cuda`, `platform-gpu`).
//! The safe API over these calls is `retrograd-engine`.

use std::ffi::{CStr, c_char, c_double, c_float, c_int, c_void};

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
    pub shuffle_seed: u64,
}

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
    pub lora_parameter_bytes: u64,
    pub lora_gradient_bytes: u64,
    pub adamw_momenta_bytes: u64,
    pub lora_on_host: bool,
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
#[derive(Clone, Copy, Debug)]
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetroOptimizerState {
    pub iter: i64,
    pub has_momenta: bool,
    pub optimizer: i32,
    pub learning_rate: c_float,
    pub weight_decay: c_float,
    pub max_grad_norm: c_float,
    pub scheduler_step: u64,
    pub scheduler_total_steps: u64,
    pub last_learning_rate: c_float,
}

#[repr(C)]
pub struct RetroTrainer {
    _private: [u8; 0],
}

pub type RetroTrainProgressCallback =
    unsafe extern "C" fn(u32, *const RetroTrainMetrics, *mut c_void) -> bool;

unsafe extern "C" {
    pub fn retro_trainer_new(
        model_path: *const c_char,
        train_config: *const RetroTrainConfig,
    ) -> *mut RetroTrainer;

    /// [`retro_trainer_new`] plus the process-wide runtime policy, which must be
    /// fixed before any backend context exists. Fails
    /// rather than silently ignoring a policy that disagrees with one already in
    /// force. `runtime_config` may be null.
    pub fn retro_trainer_new_ex(
        model_path: *const c_char,
        train_config: *const RetroTrainConfig,
        runtime_config: *const RetroRuntimeConfig,
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

    pub fn retro_trainer_tokenize_text(
        trainer: *mut RetroTrainer,
        text: *const c_char,
        tokens: *mut i32,
        n_tokens_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

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
        add_assistant: bool,
        buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
    ) -> c_int;

    pub fn retro_chat_template_tool_call_parser(
        template_src: *const c_char,
        tools_json: *const c_char,
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
    ) -> c_int;

    pub fn retro_trainer_generate_continuous_batch(
        trainer: *mut RetroTrainer,
        sequences: *const RetroGenerationSequence,
        n_sequences: usize,
        out_tokens: *mut i32,
        out_logprobs: *mut c_float,
        n_out_max: usize,
        out_n_tokens: *mut usize,
    ) -> c_int;

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

    pub fn retro_trainer_momenta_count(trainer: *mut RetroTrainer, out_count: *mut usize) -> c_int;

    pub fn retro_trainer_momenta_info(
        trainer: *mut RetroTrainer,
        index: usize,
        name_buffer: *mut c_char,
        n_buffer: usize,
        out_n_bytes: *mut usize,
        out_ne: *mut i64,
        out_n_elements: *mut usize,
    ) -> c_int;

    pub fn retro_trainer_momenta_read(
        trainer: *mut RetroTrainer,
        index: usize,
        out_m: *mut c_float,
        out_v: *mut c_float,
        n_values: usize,
    ) -> c_int;

    pub fn retro_trainer_momenta_write(
        trainer: *mut RetroTrainer,
        name: *const c_char,
        m: *const c_float,
        v: *const c_float,
        n_values: usize,
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
        // in retro_rollout.cpp would be a silent hole in the duty cycle rather
        // than a compile error: generation and scoring would keep running
        // unthrottled while the report said the limiter was active.
        // The three lines the helper itself owns are the only exemption; probe,
        // preflight and calibration decodes live in other files and stay
        // unthrottled on purpose.
        let source = include_str!("../runtime/src/retro_rollout.cpp");
        let raw: Vec<&str> = source
            .lines()
            .filter(|line| line.contains("llama_decode(") && !line.trim_start().starts_with("//"))
            .collect();
        assert_eq!(
            raw.len(),
            2,
            "raw llama_decode calls outside decode_with_duty_cycle: {raw:#?}"
        );
        for line in &raw {
            assert!(
                line.contains("return llama_decode(ctx, batch);")
                    || line.contains("const int status = llama_decode(ctx, batch);"),
                "unexpected raw decode: {line}"
            );
        }
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

        assert_eq!(size_of::<RetroTrainConfig>(), 120);
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
        assert_eq!(offset_of!(RetroTrainConfig, shuffle_seed), 112);

        assert_eq!(size_of::<RetroScoringStats>(), 48);
        assert_eq!(align_of::<RetroScoringStats>(), 8);
        assert_eq!(offset_of!(RetroScoringStats, scored_positions), 32);

        assert_eq!(size_of::<RetroGenerationStats>(), 56);
        assert_eq!(align_of::<RetroGenerationStats>(), 8);
        assert_eq!(offset_of!(RetroGenerationStats, prefilled_tokens), 24);
        assert_eq!(offset_of!(RetroGenerationStats, reused_tokens), 32);

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
        assert_eq!(offset_of!(RetroMemoryReport, lora_parameter_bytes), 48);
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
        // `n_topk` is appended, and lands in the tail padding the 40-byte layout
        // already carried, so the struct keeps its size. The offset assertion is
        // what pins that; the size alone would not have noticed the field.
        assert_eq!(size_of::<RetroWeightedDataset>(), 40);
        assert_eq!(offset_of!(RetroWeightedDataset, n_ctx), 32);
        assert_eq!(offset_of!(RetroWeightedDataset, n_topk), 36);
        assert_eq!(size_of::<RetroPackedSequenceBatch>(), 72);
        assert_eq!(offset_of!(RetroPackedSequenceBatch, n_sequences), 64);
        assert_eq!(offset_of!(RetroPackedSequenceBatch, n_topk), 68);
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
        let registered = unsafe {
            let mut needed = 0;
            retro_backend_list(ptr::null_mut(), 0, &mut needed) == 0 && needed > 0
        };
        let has_gpu = if registered {
            unsafe {
                let mut needed = 0;
                if retro_backend_list(ptr::null_mut(), 0, &mut needed) != 0 {
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
            }
        } else {
            false
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
        let mut shape = [0_i64; 4];
        let mut metrics = RetroTrainMetrics::default();
        let mut eval_metrics = RetroEvalMetrics::default();
        let mut timing = RetroOptimizerTiming::default();
        let mut memory = RetroOptimizerMemory::default();
        let mut scoring = RetroScoringStats::default();
        let mut generation = RetroGenerationStats::default();
        let mut duty_cycle = RetroDutyCycleStats::default();
        let mut state = RetroOptimizerState::default();
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
            ));
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
            assert_null_trainer(retro_trainer_momenta_count(ptr::null_mut(), &mut count));
            assert_null_trainer(retro_trainer_momenta_info(
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                &mut bytes,
                shape.as_mut_ptr(),
                &mut count,
            ));
            assert_null_trainer(retro_trainer_momenta_read(
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ));
            assert_null_trainer(retro_trainer_momenta_write(
                ptr::null_mut(),
                text.as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
            ));
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
            shuffle_seed: 42,
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
#[allow(dead_code)]
#[path = "../build/backend_selection.rs"]
mod backend_selection;
