#pragma once

// The single generation code path of the runtime, shared by the trainer
// (retro_trainer_generate_* / score_*) and the inference engine
// (retro_engine_*). gen_core knows only a model, one context and the
// adapters attached to that context - plus the owner's ambient execution
// policy (duty-cycle limiter, device capabilities) and prefix-cache
// bookkeeping, both handed over by pointer so they stay owned by the state
// that outlives the calls. It holds no optimizer, no training result cache
// and no trainable flags: everything forward-only lives here, and the two
// owners differ only in what they wrap around it.

#include "retro_runtime.hpp"

#include <random>

namespace retro {

// Outcome of one scoring pass. A device-gathered log-probability that came back
// non-finite is not an error: the pass reports it so the caller can redo the
// whole request on the host oracle, which keeps its reduction in double.
enum class score_pass { ok, needs_host_reduction, failed };

namespace gen_detail {

struct batch_guard {
    llama_batch batch;
    explicit batch_guard(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {}
    ~batch_guard() { llama_batch_free(batch); }
    batch_guard(const batch_guard &) = delete;
    batch_guard & operator=(const batch_guard &) = delete;
};

// Owning guard for a llama_batch_ext allocated for one call; the step API
// reuses a caller-lent batch instead (see gen_core::step).
struct batch_ext_guard {
    llama_batch_ext * batch;
    explicit batch_ext_guard(llama_context * ctx) : batch(ctx ? llama_batch_ext_init(ctx) : nullptr) {}
    ~batch_ext_guard() { if (batch) llama_batch_ext_free(batch); }
    batch_ext_guard(const batch_ext_guard &) = delete;
    batch_ext_guard & operator=(const batch_ext_guard &) = delete;
};

// Vocab-sized sampling buffers, reused across the tokens of one generation so
// they are allocated once per call instead of once per sampled token.
struct sampler_scratch {
    std::vector<double> probs;
    std::vector<int32_t> order;
    std::vector<int32_t> selected;
    float max = -INFINITY;
    double total = 0.0;
};

} // namespace gen_detail

// The shared generation core. A view, not an owner: `model`, `ctx` and `cache`
// must outlive it, and all output buffers are caller-owned for the duration of
// a call.
struct gen_core {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    gen_core_env env;
    gen_cache * cache = nullptr;

    gen_core(llama_model * model, llama_context * ctx, gen_core_env env, gen_cache * cache)
        : model(model), ctx(ctx), env(env), cache(cache) {}

    const llama_vocab * vocab() const { return llama_model_get_vocab(model); }
    size_t n_vocab() const {
        return static_cast<size_t>(llama_vocab_n_tokens(vocab()));
    }

    // Every decode owned by the core goes through here, so the owner's
    // duty-cycle limiter sees each of them. Disabled (env.duty_cycle == null
    // or the limiter off), this is exactly `llama_decode(ctx, batch)` - one
    // predictable branch, no clock read and, above all, no fence.
    //
    // Enabled, it has to add one. `llama_decode` returning 0 means *submitted*:
    // llama_context::decode ends with its `synchronize()` call commented out and
    // hands the outputs back through `ggml_backend_tensor_get_async`. Sites whose
    // next act is `llama_get_logits_ith`, `llama_get_embeddings_ith` or the fork's
    // `llama_get_target_logprob_ith` are already synchronized by those accessors,
    // but a prefill emitting no logits row is not, and idle time may only be
    // charged against work that is actually finished. Fencing every accounted
    // decode is uniform and correct; it costs the overlap between host-side
    // sampling and the previous decode still in flight, which is the price the
    // contract names as "0.99 is not approximately 1.0".
    int decode(llama_batch & batch);

    // The extended-batch counterpart, for the step-wise API: one
    // llama_process per call, through the same owner policy (duty-cycle
    // limiter included) as decode().
    int decode_ext(llama_batch_ext & batch);

    // The step-wise generation primitive. One llama_process over `rows`
    // (n_rows), sampled rows read back through the batch index their last
    // token was added at - which is exactly what llama_get_logits_ith (and the
    // sampled-token / target-logprob accessors) accept after llama_process.
    // `requests` is owned by the caller's state and outlives the call;
    // `ext_batch` is a reusable batch lent by the owner (cleared and refilled
    // here), or null to allocate one for the call. Returns 0 on success, -1
    // on error (retro_last_error() set). See retro_lora_train.h for the full
    // residency, logits-validity and partial-mutation contract.
    int step(
            const retro_step_row * rows,
            size_t n_rows,
            retro_step_out_row * out_rows,
            int32_t * out_top_ids,
            float * out_top_logprobs,
            size_t n_top_stride,
            step_request_table & requests,
            const reasoning_budget_plan * plan,
            llama_batch_ext * ext_batch);

    // Raw sequence operations on the context's memory, for callers that manage
    // placement themselves (the step API does). Each one invalidates the
    // prefix-cache bookkeeping: the mutation happened outside its protocol.
    // seq_rm returns false (after set_error) when the memory cannot perform a
    // partial removal; removing a whole sequence never fails.
    bool seq_rm(int32_t slot, int32_t p0, int32_t p1);
    bool seq_cp(int32_t src, int32_t dst, int32_t p0, int32_t p1);
    // Two-call save contract; -2 means the buffer is too small and
    // *out_n_bytes carries the required size.
    int seq_save(int32_t slot, uint32_t flags, uint8_t * buffer, size_t n_buffer, size_t * out_n_bytes);
    int seq_load(int32_t slot, uint32_t flags, const uint8_t * buffer, size_t n_bytes, size_t * out_n_read);
    // Resident position range of `slot`, both -1 when empty.
    int seq_pos(int32_t slot, int32_t * out_pos_min, int32_t * out_pos_max);
    int memory(retro_engine_memory_info * out);

    // Clears the context's sequence memory (the `true` variant: embeddings too).
    void clear_ctx_memory();
    // Clears the context's sequence memory and the prefix-cache bookkeeping.
    // The pair is never split: residency that outlives the cells it describes
    // is the one way this cache can be wrong rather than merely cold.
    void clear_generation_memory();

    bool validate_tokens(const int32_t * tokens, size_t n_tokens) const;
    bool validate_sampling(const retro_sampling_params & sampling) const;

    // Decodes tokens at absolute positions pos0.. in n_batch chunks and returns
    // a borrowed pointer to the final requested logits row, through out_logits.
    bool decode_span_last_logits(
            const int32_t * tokens,
            size_t n_tokens,
            int64_t pos0,
            int64_t logits_from,
            llama_batch & batch,
            const float * & out_logits);
    bool decode_span_no_logits(
            const int32_t * tokens,
            size_t n_tokens,
            int64_t pos0,
            llama_batch & batch);

    // Scores targets [first_target, n_tokens): each emitted position is reduced
    // to the scalar log-probability of the next token, on the device when the
    // owner's capabilities allow it, else on the host oracle.
    bool score_token_range(
            const int32_t * tokens,
            size_t n_tokens,
            size_t first_target,
            float * out_logprobs);

    // One shared-prompt generation call: the prompt is decoded once, its memory
    // copied to one sequence per member, and all live members advance in one
    // decode batch per token step. Outputs are row-major with stride n_out_max.
    // Returns 0 on success, -1 on error (retro_last_error() set).
    int generate_batch(
            const int32_t * prompt_tokens,
            size_t n_prompt,
            const retro_sampling_params * sampling,
            size_t n_sequences,
            int32_t * out_tokens,
            float * out_logprobs,
            size_t n_out_max,
            size_t * out_n_tokens,
            uint8_t * out_forced,
            const reasoning_budget_plan * plan);

    // Continuous counterpart for heterogeneous prompts: one KV sequence per
    // request, advanced together, with content-ordered placement and prefix
    // reuse. Same output layout and error contract as generate_batch().
    int generate_continuous_batch(
            const retro_generation_sequence * sequences,
            size_t n_sequences,
            int32_t * out_tokens,
            float * out_logprobs,
            size_t n_out_max,
            size_t * out_n_tokens,
            uint8_t * out_forced,
            const reasoning_budget_plan * plan);
};

// The device-path switches of a gen_core's owner, read per call rather than
// cached: RETRO_DEVICE_LOGPROBS=0 / RETRO_DEVICE_SAMPLING=0 are both escape
// hatches and how the two reductions are compared on a given model, including
// from a test that flips them.
bool gen_env_device_logprob_gather(const gen_core_env & env);
bool gen_env_device_sampling(const gen_core_env & env, const float * out_logprobs);

// Temperature-1 log-softmax value of `token` under a logits row. The host
// oracle's reduction: max and log-sum in double, spelled as
// `logit - max - log_sum` so every path that compares against it bit for bit
// sees the same rounding.
float token_logprob(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        std::vector<double> & work);
float token_logprob_scalar(const float * logits, size_t n_vocab, int32_t token);

// The `k` largest entries of one logits row, as ids and log-probabilities, in
// decreasing probability order, ties broken by ascending id. Sorted rather
// than merely selected: every consumer reads column 0 as the argmax, so the
// order is part of the contract.
void top_token_logprobs(
        const float * logits,
        size_t n_vocab,
        size_t k,
        std::vector<int32_t> & order,
        std::vector<double> & work,
        int32_t * out_ids,
        float * out_logprobs);

// Decodes states [0, n_states) at absolute positions 0.., emitting one output
// row per state >= first_state, while keeping peak temporary outputs bounded
// by one physical ubatch. Prefix states are decoded in n_batch chunks without
// outputs; output-emitting chunks are capped at n_ubatch so callers can reduce
// each row before the next chunk, rather than materializing
// sequence_length * vocabulary floats. on_output(state_index, batch_index) is
// invoked for every emitted row and returns false (after set_error) to abort.
// on_before_outputs(first_state_of_chunk, count) runs immediately before an
// emitting decode, which is the only point where a per-decode request such as
// llama_set_target_logprobs can be installed.
template <typename OnBeforeOutputs, typename OnOutput>
bool decode_states_emitting_outputs(
        gen_core & core,
        const int32_t * tokens,
        size_t n_states,
        size_t first_state,
        const char * decode_error,
        OnBeforeOutputs && on_before_outputs,
        OnOutput && on_output) {
    llama_context * ctx = core.ctx;
    const uint32_t n_batch = llama_n_batch(ctx);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);

    gen_detail::batch_guard guard(static_cast<int32_t>(n_batch));
    size_t offset = 0;
    while (offset < n_states) {
        const bool emit_outputs = offset >= first_state;
        const size_t boundary = emit_outputs ? n_states : first_state;
        const size_t chunk_limit = emit_outputs ? n_ubatch : n_batch;
        const uint32_t count = static_cast<uint32_t>(
                std::min<size_t>(chunk_limit, boundary - offset));
        llama_batch & batch = guard.batch;
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = static_cast<llama_pos>(offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = emit_outputs;
        }
        if (emit_outputs && !on_before_outputs(offset, count)) {
            return false;
        }
        if (core.decode(batch) != 0) {
            set_error(decode_error);
            return false;
        }
        if (emit_outputs) {
            for (uint32_t i = 0; i < count; ++i) {
                if (!on_output(offset + i, i)) {
                    return false;
                }
            }
        }
        offset += count;
    }
    return true;
}

} // namespace retro
