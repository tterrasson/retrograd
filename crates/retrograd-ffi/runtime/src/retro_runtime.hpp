#pragma once

#include "retro_lora_train.h"
#include "retro_duty_cycle.hpp"

#include "ggml-backend.h"
#include "ggml-rir/ggml-rir.h"
#include "ggml-opt.h"
#include "gguf.h"
#include "llama-adapter.h"
#include "llama.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include <memory>
#include <string>
#include <vector>

#ifndef RETRO_LLAMA_CPP_COMMIT
#define RETRO_LLAMA_CPP_COMMIT "unknown"
#endif

#ifndef RETRO_LLAMA_CPP_UPSTREAM_COMMIT
#define RETRO_LLAMA_CPP_UPSTREAM_COMMIT "unknown"
#endif

namespace retro {

extern thread_local std::string g_last_error;

struct llama_model_deleter {
    void operator()(llama_model * model) const;
};

struct llama_context_deleter {
    void operator()(llama_context * ctx) const;
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) const;
};

struct ggml_opt_dataset_deleter {
    void operator()(ggml_opt_dataset * dataset) const;
};

struct ggml_opt_result_deleter {
    void operator()(ggml_opt_result * result) const;
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) const;
};

// Opaque: the parsed jinja program lives in retro_chat_template.cpp so jinja
// headers don't leak into every translation unit that includes this header.
struct chat_template_state;
struct chat_template_state_deleter {
    void operator()(chat_template_state * state) const;
};

using model_ptr = std::unique_ptr<llama_model, llama_model_deleter>;
using context_ptr = std::unique_ptr<llama_context, llama_context_deleter>;
using lora_adapter_ptr = std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter>;
using dataset_ptr = std::unique_ptr<ggml_opt_dataset, ggml_opt_dataset_deleter>;
using opt_result_ptr = std::unique_ptr<ggml_opt_result, ggml_opt_result_deleter>;
using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;
using chat_template_state_ptr = std::unique_ptr<chat_template_state, chat_template_state_deleter>;

struct trainer_state {
    std::string model_path;
    retro_train_config train_config {};
    model_ptr model;
    context_ptr ctx;
    // Dedicated multi-sequence inference context. The optimizer context stays
    // single-sequence because llama_opt interprets its total KV width as one
    // training datapoint.
    context_ptr generation_ctx;
    lora_adapter_ptr adapter;
    bool has_lora = false;
    bool loaded_lora = false;
    // loaded adapter validated and its tensors flagged as optimizer params
    bool lora_promoted = false;
    // llama_opt_init ran (it must run exactly once per context)
    bool opt_created = false;
    // optimizer ready for training: opt context created and preflight passed
    bool opt_initialized = false;
    // training-graph preflight cache; missing < 0 means it has not run yet
    int32_t preflight_missing = -1;
    std::string preflight_report;
    uint32_t lora_rank = 0;
    float lora_alpha = 0.0f;
    int32_t lora_dtype = RETRO_LORA_DTYPE_F32;
    std::vector<std::string> target_patterns;
    ggml_opt_optimizer_params optimizer_params {};
    // Fixed-shape PPO/GRPO calls reuse these host-side ggml containers instead
    // of allocating a dataset and two result objects for every rollout step.
    dataset_ptr weighted_dataset_cache;
    opt_result_ptr weighted_train_result_cache;
    opt_result_ptr weighted_eval_result_cache;
    size_t weighted_cache_rows = 0;
    uint32_t weighted_cache_ctx = 0;
    // Extra variables handed to the chat template on every render, as a JSON object, empty for
    // none.
    std::string chat_template_variables;
    uint64_t scheduler_step = 0;
    uint64_t scheduler_total_steps = 0;
    // Checkpoint resume point. When active, the SFT epoch loop starts at
    // `resume_epoch` and keeps the restored scheduler step instead of
    // restarting the schedule from zero.
    uint32_t resume_epoch = 0;
    bool resume_active = false;
    float last_learning_rate = 0.0f;
    int32_t requested_device = 0;
    bool gpu_active = false;
    int32_t n_gpu_layers = 0;
    uint32_t effective_threads = 0;
    std::string backend_name = "CPU";
    int32_t effective_kv_dtype = RETRO_KV_DTYPE_F32;
    std::string training_kv_status = "not_requested";
    // Active-device training capabilities, resolved from ggml supports_op probes
    // at load time rather than from the backend name. `backend_name` is reserved
    // for diagnostics; these decide behavior.
    bool cap_flash_attn_back = false;
    bool cap_device_sampling = false;
    // Whether the behavior scorer can gather log p(target) inside the decode
    // graph instead of pulling an n_vocab logits row per scored position.
    bool cap_device_logprobs = false;
    // Micro-batch the training context was actually built with. Equal to the
    // requested n_ubatch unless the load-time finiteness probe escalated it.
    uint32_t effective_ubatch = 0;
    std::string ubatch_finite_status = "not_probed";
    // Whether this (model, device) pair may train on packed multi-sequence
    // micro-batches: llama_model_supports_packed_seq() for the graph,
    // downgraded by the load-time equivalence probe for the driver.
    bool cap_packed_seq_training = false;
    std::string packed_seq_status = "not_probed";
    // Recurrent-state rollback: what the derivation asked for and what
    // llama_context ended up applying (llama_n_rs_seq).
    std::string recurrent_rollback_status = "0 (not derived)";
    uint32_t effective_rs_seq = 0;
    // Whether the active device runs the fused sparse cross-entropy pair for
    // this model's head. Only consulted when chunked_cross_entropy is set, but
    // resolved unconditionally so the report can warn before the option is.
    bool cap_fused_sparse_ce = false;
    // Number of training-graph nodes the preflight saw the active GPU decline
    // (forward + backward). Zero once the preflight has run and found none;
    // meaningless before it runs (preflight_missing < 0).
    int32_t preflight_device_fallbacks = 0;
    // Monotonic counters of the shared-prefix behavior scorer, reported by
    // retro_trainer_scoring_stats. Accumulated for the trainer's lifetime so a
    // caller can attribute one update by differencing two snapshots.
    retro_scoring_stats scoring_stats {};
    // What each sequence of the *generation* context currently holds, so a
    // multi-turn rollout can extend a prompt instead of re-decoding it.
    // `generation_kv[slot]` is exactly the tokens occupying positions
    // `0.. generation_kv[slot].size()` of that sequence. The invariant is the
    // whole design: it is what lets the next call decide, by comparing tokens,
    // that a prompt extends what is already resident and that only the tail has
    // to be decoded. Every path that can break it - an explicit clear, a weight
    // change, an error - goes through `forget_generation_kv`, because a stale
    // entry is not a slow cache but a wrong one: it would attend over cells
    // computed for other tokens, or under other weights.
    std::vector<std::vector<int32_t>> generation_kv;
    // Last call that touched each slot, for the eviction order. A rollout with
    // more live trajectories than the context has sequences has to lose some;
    // losing the least recently extended one costs one prefill, and losing a
    // trajectory that is still advancing every turn would cost one per turn.
    std::vector<uint64_t> generation_kv_used_at;
    uint64_t generation_kv_clock = 0;
    // Monotonic counters of that reuse, reported by
    // retro_trainer_generation_stats. Same contract as `scoring_stats`:
    // lifetime totals, differenced by the caller around one update.
    retro_generation_stats generation_stats {};
    // Execution policy, not training geometry: how much of its own wall time
    // this trainer is allowed to spend waiting on submitted GPU work, so
    // another workload can use the device in between. Disabled unless
    // retro_trainer_set_max_gpu_duty_cycle asked for it, and inert on a CPU
    // backend. Its owner is this state; the rollout sites reach it through the
    // `trainer_state &` they already take, and the optimizer callback - whose
    // vendored signature has no user-data slot - through the `thread_local`
    // step_progress pointer that is already there. One owner, two access paths.
    duty_cycle_limiter duty_cycle;
    std::string lora_description_cache;
    std::string backend_report_cache;
    // llama_opt_memory sample count folded into backend_report_cache when it was
    // rendered. Those measurements move during training while nothing else in the
    // report does, so structural invalidation alone would keep serving the first
    // step's numbers. The counter is the right key rather than any single value:
    // it increments on every sample, so an unchanged counter is the only proof
    // that *none* of device_used / peak / scratch / scratch_peak has moved.
    uint64_t backend_report_memory_samples = 0;
    std::string capability_report_cache;
    // RIR counters as they stood when this trainer was created. The counters
    // themselves are process-wide, so the capability report differences against
    // this to answer "what did *this* trainer do" rather than inheriting a
    // previous trainer's coverage.
    ggml_rir_counters rir_baseline {};
    // Derived from the loaded model's tensor table alone, so this value never
    // goes stale and is deliberately left out of
    // invalidate_report_caches().
    std::string lora_candidate_targets_cache;
    // Lexed/parsed once per trainer (the model's chat template never changes
    // for its lifetime) and reused across every format_chat call.
    chat_template_state_ptr chat_template_cache;
    // Serialized PEG tool-call parser and the catalog it was derived from. The
    // generated grammar may name the tools, so the catalog is the cache key and
    // not merely a detail: a parser built for one catalog does not read another.
    std::string tool_call_parser_cache;
    std::string tool_call_parser_tools;

    // Single owner of report-cache coherence: every mutation that can change
    // any report calls this. Recomputing a still-valid report is harmless,
    // serving a stale one is not.
    void invalidate_report_caches() {
        lora_description_cache.clear();
        backend_report_cache.clear();
        capability_report_cache.clear();
    }

    // Single owner of generation-KV coherence. Called by every mutation that
    // can invalidate a resident prefix: a cleared context, and - the one that
    // is easy to miss - any change to the weights. A prefix decoded under the
    // adapter of update N is not the prefix of update N+1, and reusing it would
    // sample from a model that no longer exists rather than merely sample
    // badly.
    // The cells go with the bookkeeping. A slot whose record is empty but whose
    // cells are still allocated is not wrong - the next placement removes the
    // sequence before decoding into it - but it is only removed when that slot
    // is claimed again, and a unified cache prices every attention row over its
    // whole occupied span, so cells nobody will extend would keep taxing every
    // decode step until the next full clear. Metadata only: a sequence that
    // starts over in a cell is zeroed by the graph, never by a memset, which is
    // the same property the per-slot eviction already relies on.
    void forget_generation_kv() {
        if (generation_ctx) {
            if (llama_memory_t memory = llama_get_memory(generation_ctx.get())) {
                llama_memory_clear(memory, false);
            }
        }
        for (auto & resident : generation_kv) {
            resident.clear();
        }
        std::fill(generation_kv_used_at.begin(), generation_kv_used_at.end(), 0);
    }

};

void set_error(std::string message);
void clear_error();
bool is_blank(const char * value);
void ensure_backend_initialized();
void configure_runtime_logging(bool verbose);
retro_train_config default_train_config();
const char * device_kind_name(int32_t device);
bool validate_train_config(const retro_train_config & config);
trainer_state * checked(retro_trainer * trainer);
std::string join_patterns(const std::vector<std::string> & patterns);
int copy_string_out(const std::string & text, char * buffer, size_t n_buffer, size_t * out_n_bytes);

// Zero-copy sibling of copy_string_out with the identical two-call contract: a
// null/empty buffer only reports the required size, a short buffer fails with
// -2, and successful writes are NUL-terminated. render(destination, capacity)
// writes at most capacity bytes and returns the full untruncated size in
// bytes, or a negative value on failure (reported as render_error).
template <typename Render>
int render_string_out(
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes,
        const char * render_error,
        Render && render) {
    char * destination = buffer && n_buffer > 0 ? buffer : nullptr;
    const int32_t capacity = destination
            ? static_cast<int32_t>(std::min<size_t>(n_buffer - 1, INT32_MAX))
            : 0;
    const int32_t needed = render(destination, capacity);
    if (needed < 0) {
        set_error(render_error);
        return -1;
    }
    *out_n_bytes = static_cast<size_t>(needed);
    if (!destination) {
        return 0;
    }
    if (needed > capacity) {
        set_error("output buffer is too small");
        return -2;
    }
    destination[needed] = '\0';
    return 0;
}

// Replaces chat-template variables; null or `{}` clears them. Reserved keys are rejected.
int set_chat_template_variables(trainer_state & state, const char * variables_json);

// Renders `messages` through the model's own tokenizer.chat_template by
// actually executing its Jinja program (defined in retro_chat_template.cpp),
// rather than matching it against a fixed list of known template families.
// Follows the render_string_out two-call contract.
int render_chat_template(
        trainer_state & state,
        const char * const * roles,
        const char * const * contents,
        size_t n_messages,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

// Same renderer, given the messages as a JSON array instead of parallel
// role/content arrays, so a message can carry the structured fields a chat
// template expects of it (tool_call_id, name, tool_calls) and the tool catalog
// can be passed as `tools`. `tools_json` may be null.
int render_chat_messages(
        trainer_state & state,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

// Same renderer over a template given as source rather than taken from a
// model, so a `.jinja` fixture can be rendered without a GGUF. `bos_token` and
// `eos_token` render empty, there being no vocabulary to draw them from.
int render_chat_template_source(
        const char * template_src,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

// Whether the model's own chat template renders a tool catalog, probed once by
// rendering with a sentinel tool and looking for it in the output. Templates
// that ignore `tools` - most non-agentic ones - report false, and the caller
// falls back to describing the tools in the system prompt.
int chat_template_supports_tools(trainer_state & state, bool * out_supports);

// Derives the PEG tool-call parser of the model's own chat template for this
// catalog and writes it serialized (defined in retro_chat_parser.cpp). Follows
// the render_string_out two-call contract. `tools_json` may be null.
int build_tool_call_parser(
        trainer_state & state,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

// Same derivation over a template given as source. The counterpart of
// render_chat_template_source: together they let the agreement between what a
// template writes and what its parser reads be checked without a model.
int build_tool_call_parser_from_source(
        const char * template_src,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

// Runs an already-serialized parser over one assistant output. Deliberately
// free of trainer_state: the blob is self-sufficient, which is what lets the
// rollout parse on its own thread instead of routing every turn through the
// policy actor.
int parse_assistant_output(
        const char * parser_blob,
        size_t n_parser_blob,
        const char * text,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

ggml_backend_dev_t first_gpu_device();
bool load_model_and_context(trainer_state & state);
// Single computation behind both memory views: the text `backend_report` renders
// it for humans, `retro_trainer_memory_report` hands it to callers as data. Two
// independent computations would drift, and the drift would only surface as a
// resolver estimate that disagrees with the report it was validated against.
retro_memory_report memory_totals(const trainer_state & state);
int read_model_info_impl(const char * model_path, int32_t device, retro_model_info * out_info);
std::string backend_report(const trainer_state & state);
std::string capability_report(const trainer_state & state);
bool shared_prefix_packed_training(const trainer_state & state);
// Appended to capability_report at call time rather than folded into it: the
// counters it carries move on every graph, and capability_report is cached for
// the trainer's lifetime.
std::string rir_capability_section(const trainer_state & state);
// RIR counters broken down by (ggml_op, backend, variant_id), tab-separated.
// Defined next to the probe because that is where the registry is already
// included; shared so the FFI variant report and capability_report cannot
// disagree about what ran.
std::string rir_site_lines();
int backend_list_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes);
int gpu_runtime_probe_impl();
int device_memory_impl(size_t * out_free, size_t * out_total);
// retro delta: transfer timing used by the activation-offload bandwidth gate.
int transfer_probe_impl(size_t bytes, uint32_t iterations, retro_transfer_rates * out_rates);

// retro delta: whether a training kernel can decode this type in place on every
// backend. Shared by the probe and preflight to keep rejection causes consistent.
bool is_retro_dequant_type(enum ggml_type type);
bool is_retro_out_prod_type(enum ggml_type type);

bool is_param_tensor(const ggml_tensor * tensor);
const char * lora_dtype_name(int32_t dtype);
int64_t count_lora_parameters(const trainer_state & state);
size_t count_lora_parameter_bytes(const trainer_state & state);
// Automatic LoRA target profile, detected from the tensors actually present in
// the model instead of a per-architecture table. Single source of truth for
// target resolution and the capability report.
struct lora_profile {
    std::string name;
    std::vector<std::string> patterns;
};
bool detect_lora_profile(const llama_model & model, lora_profile & out);
// Sorted wildcard families of the model's 2D block weights (layer indices
// collapsed to '*'), the candidates suggested when no profile is detected.
std::vector<std::string> lora_candidate_patterns(const llama_model & model);
// Sorted concrete names of every tensor a LoRA adapter can attach to. Unlike
// lora_candidate_patterns() the layer index is kept, so a caller can resolve
// "the first layer carrying attn_k" instead of assuming a layout. Hybrid
// architectures (lfm2, falcon-h1) interleave block families, so blk.0 is not
// guaranteed to be an attention block.
std::vector<std::string> lora_candidate_targets(const llama_model & model);
std::string describe_lora(const trainer_state & state);
bool validate_lora_tensor_pairs(const llama_adapter_lora & adapter);
bool save_lora_adapter_gguf(const trainer_state & state, const char * adapter_path);
// One-line model identity recorded in a checkpoint manifest (retro_checkpoint.cpp).
std::string model_signature(const trainer_state & state);
bool create_trainable_lora_adapter(trainer_state & state, uint32_t seed);
// Validates a file-loaded adapter for training and flags its tensors as
// optimizer params. Idempotent; must run before llama_opt_init.
bool promote_loaded_lora_to_trainable(trainer_state & state);
bool resolve_lora_targets(trainer_state & state, std::vector<const ggml_tensor *> & targets);
bool assert_lora_only_params(const trainer_state & state);
bool opt_param_filter_none(const ggml_tensor * tensor, void * userdata);

// Whether a training-graph node falling back to the CPU should fail the
// preflight for this run: the config field, or RETRO_REQUIRE_GPU_RESIDENT set to
// anything but "0"/"".
// The env override exists because "no node in this lane runs on the CPU" is a
// property of a *lane*, not of a run: it is the guard that would have caught the
// fused cross-entropy sitting on the CPU tail on a backend whose kernels had
// since landed, and no per-run fixture was ever going to notice that. Config
// wins when it is already set, so a run that asks for the guard keeps it whatever
// the environment says.
bool require_gpu_resident(const trainer_state & state);

bool ensure_lora_optimizer_initialized(trainer_state & state);
// Creates the llama.cpp optimizer context once (llama_opt_init) and verifies
// that only LoRA tensors are trainable. Does not run the preflight.
bool ensure_opt_context(trainer_state & state);
// Runs the training-graph preflight once and caches its report and the number
// of ops without a gradient rule on the state. Requires ensure_opt_context.
bool ensure_train_preflight(trainer_state & state);
int train_preflight_impl(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);
int preflight_summary_impl(retro_trainer * trainer, retro_preflight_summary * out_summary);
ggml_opt_optimizer_params scheduled_optimizer_params(void * userdata);
int train_tokens_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        retro_train_metrics * out_metrics);
int train_sft_impl(
        retro_trainer * trainer,
        const retro_sft_dataset * train,
        const retro_sft_dataset * eval,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data);
int train_weighted_impl(
        retro_trainer * trainer,
        const retro_weighted_dataset * data,
        uint64_t scheduler_total_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data);
int train_packed_sequences_impl(
        retro_trainer * trainer,
        const retro_packed_sequence_batch * data,
        uint64_t scheduler_total_steps,
        uint32_t accumulation_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data);

int generate_batch_impl(
        retro_trainer * trainer,
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens);
int generate_continuous_batch_impl(
        retro_trainer * trainer,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens);
int score_tokens_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_logprobs);
int eval_sft_impl(
        retro_trainer * trainer,
        const retro_sft_dataset * data,
        retro_eval_metrics * out_metrics);
int score_token_suffix_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs);
int top_logprobs_suffix_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        size_t k,
        int32_t * out_ids,
        float * out_logprobs,
        size_t n_out_max);
int score_token_suffix_batch_impl(
        retro_trainer * trainer,
        const retro_token_suffix_sequence * sequences,
        size_t n_sequences,
        float * out_logprobs,
        size_t out_stride,
        size_t * out_n_logprobs);
int set_lora_enabled_impl(retro_trainer * trainer, bool enabled);
int hidden_size_impl(retro_trainer * trainer, uint32_t * out_n_embd);
int hidden_states_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_features,
        size_t n_features_max);
int score_token_suffix_and_hidden_states_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs,
        float * out_features,
        size_t n_features_max);
int detokenize_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        bool unparse_special,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes);

int probe_op_run_impl(
        int32_t op,
        int32_t use_gpu,
        const int64_t * ne_src0,
        const float * src0,
        const int64_t * ne_src1,
        const float * src1,
        const int64_t * ne_src2,
        const float * src2,
        float param0,
        float param1,
        float * dst,
        size_t dst_len);
// retro delta: probe with an explicit native or RIR implementation choice.
int probe_op_run_ex_impl(
        int32_t op,
        int32_t use_gpu,
        const int64_t * ne_src0,
        const float * src0,
        const int64_t * ne_src1,
        const float * src1,
        const int64_t * ne_src2,
        const float * src2,
        float param0,
        float param1,
        float * dst,
        size_t dst_len,
        int32_t implementation,
        retro_kernel_run_info * info);
int rir_counters_impl(retro_rir_counters * out);
int rir_variant_report_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes);
int rir_census_report_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes);
// The duty-cycle limiter's two C bridges, defined next to its arithmetic in
// retro_duty_cycle.cpp so the class itself stays free of the ABI header.
retro_duty_cycle_stats duty_cycle_snapshot(const duty_cycle_limiter & limiter);
int duty_cycle_probe_impl(
        float fraction,
        const retro_duty_cycle_event * events,
        size_t n_events,
        retro_duty_cycle_probe * out_probe);

int probe_token_logprob_impl(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        bool vectorized,
        float * out_logprob);

int fused_sparse_ce_probe_impl(
        int32_t         n_embd,
        int32_t         n_tokens,
        int32_t         n_vocab,
        int32_t         n_topk, // retro delta (DISTILL D6.5)
        int32_t         n_tiles,
        int32_t         seq_chunk,
        int32_t         offload_h,
        int32_t         w_type,
        int32_t         use_gpu,
        const float   * h,
        const float   * w,
        const int32_t * targets,
        const float   * weights,
        const float   * bias,
        float           grad_loss,
        float         * out_loss_full,
        float         * out_loss_fused,
        float         * out_grad_h_full,
        float         * out_grad_h_fused);

template <typename Fn>
int boundary(Fn && fn) {
    try {
        clear_error();
        return fn();
    } catch (const std::exception & err) {
        set_error(err.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

} // namespace retro
