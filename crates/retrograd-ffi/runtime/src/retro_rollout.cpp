// Rollout support for policy-gradient training: sampled generation,
// teacher-forced logprob scoring, and detokenization. Everything forward-only
// lives in the shared gen_core (retro_gen_core.cpp), which the engine runs
// too; this file keeps only the FFI-shaped wrappers: they validate the trainer
// handle, pick the context the call runs on, resolve the reasoning budget and
// delegate. The differentiable objective lives in retro_training.cpp.
#include "retro_gen_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace retro {

namespace {

using gen_detail::batch_guard;

// The trainer's ambient policy as a gen_core sees it: the duty-cycle limiter,
// the device capabilities probed at load time, and the scoring counters the
// score_* wrappers below report through retro_trainer_scoring_stats.
gen_core_env gen_env_of(trainer_state & state) {
    return gen_core_env{
        &state.duty_cycle,
        state.gpu_active,
        state.cap_device_sampling,
        state.cap_device_logprobs,
        &state.scoring_stats,
    };
}

// A gen_core over the context a call runs on. The prefix-cache bookkeeping
// stays owned by the trainer so it survives across calls.
gen_core core_over(trainer_state & state, llama_context * ctx) {
    return gen_core(state.model.get(), ctx, gen_env_of(state), &state.generation_cache);
}

void clear_context_memory(llama_context * ctx) {
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory) {
        llama_memory_clear(memory, true);
    }
}

void clear_context_memory(trainer_state & state) {
    clear_context_memory(state.ctx.get());
    // Conservative on purpose: on a trainer with no dedicated generation
    // context the two are the same context, and the cost of forgetting a
    // residency that was in fact untouched is one prefill, against a wrong
    // attention if it was not.
    state.forget_generation_kv();
}

} // namespace

int generate_batch_impl(
        retro_trainer * trainer,
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!prompt_tokens || n_prompt == 0) {
            set_error("prompt tokens are required");
            return -1;
        }
        if (!sampling || n_sequences == 0 || !out_tokens || !out_n_tokens || n_out_max == 0) {
            set_error("sampling params and output buffers are required");
            return -1;
        }
        // The dedicated generation context exists as soon as the trainer was
        // configured for multi-sequence sampling or for the fast (F16 KV +
        // flash-attention) sampling path; single-sequence exact sampling keeps
        // running on the optimizer context.
        llama_context * ctx = state->generation_ctx
                ? state->generation_ctx.get()
                : state->ctx.get();
        if (n_sequences > 1 && !state->generation_ctx) {
            set_error("trainer was not configured for multi-sequence generation");
            return -1;
        }
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        gen_core core = core_over(*state, ctx);
        return core.generate_batch(prompt_tokens, n_prompt, sampling, n_sequences,
                out_tokens, out_logprobs, n_out_max, out_n_tokens, out_forced, plan);
    });
}

int generate_continuous_batch_impl(
        retro_trainer * trainer,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) return -1;
        if (!sequences || n_sequences == 0 || !out_tokens || !out_n_tokens || n_out_max == 0) {
            set_error("continuous generation inputs and output buffers are required");
            return -1;
        }
        llama_context * ctx = state->generation_ctx ? state->generation_ctx.get() : state->ctx.get();
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        gen_core core = core_over(*state, ctx);
        return core.generate_continuous_batch(sequences, n_sequences, out_tokens,
                out_logprobs, n_out_max, out_n_tokens, out_forced, plan);
    });
}

// The context the step-wise generation API runs on: the dedicated generation
// context when the run has one, the main context otherwise - the same rule the
// batched generators use.
llama_context * generation_ctx_of(trainer_state & state) {
    return state.generation_ctx ? state.generation_ctx.get() : state.ctx.get();
}

int generation_step_impl(
        retro_trainer * trainer,
        const retro_step_row * rows,
        size_t n_rows,
        retro_step_out_row * out_rows,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t n_top_stride) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        llama_context * ctx = generation_ctx_of(*state);
        gen_core core = core_over(*state, ctx);
        return core.step(rows, n_rows, out_rows, out_top_ids, out_top_logprobs, n_top_stride,
                state->step_requests, plan, state->step_batch.get(ctx));
    });
}

int generation_seq_rm_impl(retro_trainer * trainer, int32_t slot, int32_t p0, int32_t p1) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.seq_rm(slot, p0, p1) ? 0 : -1;
    });
}

int generation_seq_cp_impl(
        retro_trainer * trainer, int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.seq_cp(src, dst, p0, p1) ? 0 : -1;
    });
}

int generation_seq_save_impl(
        retro_trainer * trainer,
        int32_t slot,
        uint32_t flags,
        uint8_t * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.seq_save(slot, flags, buffer, n_buffer, out_n_bytes);
    });
}

int generation_seq_load_impl(
        retro_trainer * trainer,
        int32_t slot,
        uint32_t flags,
        const uint8_t * buffer,
        size_t n_bytes,
        size_t * out_n_read) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.seq_load(slot, flags, buffer, n_bytes, out_n_read);
    });
}

int generation_seq_pos_impl(
        retro_trainer * trainer, int32_t slot, int32_t * out_pos_min, int32_t * out_pos_max) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.seq_pos(slot, out_pos_min, out_pos_max);
    });
}

int generation_memory_impl(retro_trainer * trainer, retro_engine_memory_info * out) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        gen_core core = core_over(*state, generation_ctx_of(*state));
        return core.memory(out);
    });
}

int generation_request_free_impl(retro_trainer * trainer, int32_t request) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        state->step_requests.requests.erase(request);
        return 0;
    });
}

int score_tokens_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_logprobs) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!tokens || n_tokens < 2) {
            set_error("scoring requires at least two tokens");
            return -1;
        }
        if (!out_logprobs) {
            set_error("out_logprobs is required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }

        clear_context_memory(*state);
        if (!core.score_token_range(tokens, n_tokens, 1, out_logprobs)) {
            clear_context_memory(*state);
            return -1;
        }
        clear_context_memory(*state);
        return 0;
    });
}

int eval_sft_impl(
        retro_trainer * trainer,
        const retro_sft_dataset * data,
        retro_eval_metrics * out_metrics) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!data || !data->tokens || !data->labels || data->n_rows == 0 || data->n_ctx == 0) {
            set_error("evaluation SFT dataset must contain rows");
            return -1;
        }
        if (!out_metrics) {
            set_error("out_metrics is required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (data->n_ctx > n_ctx) {
            std::ostringstream message;
            message << "evaluation row context " << data->n_ctx
                    << " exceeds model context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        if (data->n_rows > SIZE_MAX / data->n_ctx) {
            set_error("evaluation SFT dataset size overflows size_t");
            return -1;
        }

        gen_core core = core_over(*state, state->ctx.get());
        const size_t n_vocab = core.n_vocab();
        double negative_log_likelihood = 0.0;
        uint64_t supervised_tokens = 0;
        std::vector<double> logprob_work;

        for (size_t row = 0; row < data->n_rows; ++row) {
            const int32_t * tokens = data->tokens + row * data->n_ctx;
            const int32_t * labels = data->labels + row * data->n_ctx;
            if (!core.validate_tokens(tokens, data->n_ctx)) {
                clear_context_memory(*state);
                return -1;
            }
            for (size_t i = 0; i < data->n_ctx; ++i) {
                if (labels[i] < -1 || labels[i] >= static_cast<int32_t>(n_vocab)) {
                    std::ostringstream message;
                    message << "label " << labels[i] << " at row " << row
                            << ", index " << i << " is outside the vocabulary of size "
                            << n_vocab;
                    set_error(message.str());
                    clear_context_memory(*state);
                    return -1;
                }
            }

            clear_context_memory(*state);
            const bool ok = decode_states_emitting_outputs(
                    core, tokens, data->n_ctx, 0,
                    "llama_decode failed during SFT evaluation",
                    // Dense per-position labels, several of them ignored: this
                    // path reduces the rows it needs on the host and installs no
                    // per-decode request.
                    [](size_t, uint32_t) { return true; },
                    [&](size_t state_index, uint32_t batch_index) {
                        const int32_t label = labels[state_index];
                        if (label == -1) {
                            return true;
                        }
                        const float * logits = llama_get_logits_ith(
                                state->ctx.get(), static_cast<int32_t>(batch_index));
                        if (!logits) {
                            set_error("decode produced no logits for SFT evaluation");
                            return false;
                        }
                        negative_log_likelihood -= token_logprob(
                                logits, n_vocab, label, logprob_work);
                        ++supervised_tokens;
                        return true;
                    });
            clear_context_memory(*state);
            if (!ok) {
                return -1;
            }
        }

        if (supervised_tokens == 0) {
            set_error("evaluation SFT dataset contains no supervised tokens");
            return -1;
        }
        out_metrics->negative_log_likelihood = negative_log_likelihood;
        out_metrics->supervised_tokens = supervised_tokens;
        return 0;
    });
}

int score_token_suffix_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!tokens || n_prompt == 0 || n_prompt >= n_tokens) {
            set_error("suffix scoring requires a prompt and completion tokens");
            return -1;
        }
        if (!out_logprobs) {
            set_error("out_logprobs is required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }

        clear_context_memory(*state);
        if (!core.score_token_range(tokens, n_tokens, n_prompt, out_logprobs)) {
            clear_context_memory(*state);
            return -1;
        }
        clear_context_memory(*state);
        return 0;
    });
}

int top_logprobs_suffix_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        size_t k,
        int32_t * out_ids,
        float * out_logprobs,
        size_t n_out_max) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!tokens || n_prompt == 0 || n_prompt >= n_tokens) {
            set_error("top-k suffix scoring requires a prompt and completion tokens");
            return -1;
        }
        if (!out_ids || !out_logprobs) {
            set_error("out_ids and out_logprobs are required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        const size_t n_vocab = core.n_vocab();
        if (k == 0 || k > n_vocab) {
            std::ostringstream message;
            message << "top-k suffix scoring needs k in [1, " << n_vocab << "], got " << k;
            set_error(message.str());
            return -1;
        }
        const size_t rows = n_tokens - n_prompt;
        if (rows > SIZE_MAX / k || rows * k > n_out_max) {
            std::ostringstream message;
            message << "top-k suffix scoring needs " << rows << " x " << k
                    << " output entries, the caller provided " << n_out_max;
            set_error(message.str());
            return -1;
        }
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }

        const size_t first_state = n_prompt - 1;
        std::vector<double> logprob_work;
        std::vector<int32_t> order;
        clear_context_memory(*state);
        // No device gather here, and not as a fallback: the gather returns the
        // scalar log-probability of one requested target, and what this call is
        // for is the rest of the row. The n_vocab logits row therefore comes
        // back to the host, which is the cost the primitive is priced at - one
        // nth of a percent of the forward that produced it.
        const bool decoded = decode_states_emitting_outputs(
                core, tokens, n_tokens - 1, first_state,
                "llama_decode failed during top-k suffix scoring",
                [](size_t, uint32_t) { return true; },
                [&](size_t state_index, uint32_t batch_index) {
                    const float * logits = llama_get_logits_ith(
                            state->ctx.get(), static_cast<int32_t>(batch_index));
                    if (!logits) {
                        set_error("decode produced no logits for top-k suffix scoring");
                        return false;
                    }
                    const size_t row = state_index - first_state;
                    top_token_logprobs(logits, n_vocab, k, order, logprob_work,
                            out_ids + row * k, out_logprobs + row * k);
                    state->scoring_stats.scored_positions++;
                    return true;
                });
        clear_context_memory(*state);
        return decoded ? 0 : -1;
    });
}

int score_token_suffix_batch_impl(
        retro_trainer * trainer,
        const retro_token_suffix_sequence * sequences,
        size_t n_sequences,
        float * out_logprobs,
        size_t out_stride,
        size_t * out_n_logprobs) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!sequences || n_sequences == 0 || !out_logprobs || !out_n_logprobs || out_stride == 0) {
            set_error("suffix scoring batch inputs and output buffers are required");
            return -1;
        }
        llama_context * ctx = state->ctx.get();
        gen_core core = core_over(*state, ctx);
        // Only the sequence width matters: the branches are scored one at a
        // time (a private copy of the shared prefix at a time), so no
        // launch ever carries more than one completion and the physical batch
        // capacity is irrelevant to how many sequences a call may score.
        if (n_sequences > llama_n_seq_max(ctx)) {
            set_error("suffix scoring batch exceeds the configured sequence capacity");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx_seq(ctx);
        const size_t n_prompt = sequences[0].n_prompt;
        const int32_t * prompt = sequences[0].tokens;
        if (!prompt || n_prompt == 0 || n_prompt >= sequences[0].n_tokens) {
            set_error("suffix scoring requires a prompt and completion tokens");
            return -1;
        }
        if (!core.validate_tokens(prompt, sequences[0].n_tokens)) {
            return -1;
        }
        if (sequences[0].n_tokens > n_ctx) {
            set_error("suffix scoring sequence exceeds context");
            return -1;
        }
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            const retro_token_suffix_sequence & item = sequences[sequence];
            if (!item.tokens || item.n_prompt != n_prompt || n_prompt >= item.n_tokens
                    || item.n_tokens > n_ctx || item.n_tokens - n_prompt > out_stride) {
                set_error("invalid suffix scoring batch geometry");
                return -1;
            }
            if (std::memcmp(item.tokens, prompt, n_prompt*sizeof(int32_t)) != 0) {
                set_error("suffix scoring batch sequences must share an identical prompt");
                return -1;
            }
            if (!core.validate_tokens(item.tokens, item.n_tokens)) {
                return -1;
            }
            out_n_logprobs[sequence] = item.n_tokens - n_prompt;
        }

        const size_t n_vocab = core.n_vocab();
        std::vector<double> logprob_work;
        state->scoring_stats.calls++;
        clear_context_memory(*state);
        // Decode only the non-emitting prefix. The final prompt state belongs
        // to each branch's emitting decode, matching score_token_range's
        // graph boundaries exactly (and therefore its F32 reductions).
        batch_guard prefix_batch(static_cast<int32_t>(llama_n_batch(ctx)));
        const size_t prefix_len = n_prompt - 1;
        // With at least two sequence slots, keep the shared prefix in sequence
        // zero and score each branch in a copied sequence that is removed after
        // use. Single-sequence contexts retain the rollback path. Set
        // RETRO_SCORING_BRANCH_ISOLATION=0 to select that path explicitly.
        const char * isolation_env = std::getenv("RETRO_SCORING_BRANCH_ISOLATION");
        const bool isolation_disabled = isolation_env
                && (std::strcmp(isolation_env, "0") == 0 || std::strcmp(isolation_env, "false") == 0);
        const bool isolated_branches = !isolation_disabled && llama_n_seq_max(ctx) >= 2;
        const llama_seq_id prefix_seq = 0;
        const llama_seq_id branch_seq = isolated_branches ? 1 : 0;
        // Redecodes the shared, non-emitting prefix into `prefix_seq` from an
        // empty cache. Used for the initial decode and, on a context that
        // cannot isolate branches, as the fallback when llama_memory_seq_rm
        // refuses to roll a branch back to prefix_len.
        auto decode_shared_prefix = [&]() -> bool {
            state->scoring_stats.prefix_decodes++;
            size_t prefix_offset = 0;
            while (prefix_offset < prefix_len) {
                const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
                        llama_n_batch(ctx), prefix_len - prefix_offset));
                llama_batch & batch = prefix_batch.batch;
                batch.n_tokens = static_cast<int32_t>(count);
                for (uint32_t i = 0; i < count; ++i) {
                    batch.token[i] = prompt[prefix_offset + i];
                    batch.pos[i] = static_cast<llama_pos>(prefix_offset + i);
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = prefix_seq;
                    batch.logits[i] = false;
                }
                if (core.decode(batch) != 0) {
                    set_error("llama_decode failed during shared-prefix suffix scoring");
                    return false;
                }
                prefix_offset += count;
            }
            return true;
        };
        llama_memory_t memory = llama_get_memory(ctx);
        if (!memory) {
            set_error("scoring context has no sequence memory");
            clear_context_memory(*state);
            return -1;
        }
        // Gather one target log-probability per position on the device when
        // supported, instead of copying an entire vocabulary row. Set
        // RETRO_DEVICE_LOGPROBS=0 to use the host reference reduction.
        const bool device_gather_available = gen_env_device_logprob_gather(core.env);
        std::vector<int32_t> gather_targets;
        using pass_result = score_pass;
        // Score each completion against the shared prefix one sequence at a
        // time, branching off sequence zero's already-decoded prefix cells and
        // dropping each completion before the next. This deliberately uses the
        // same forward geometry as scalar scoring: some backends change F32
        // attention reduction order when unrelated sequences share a decode
        // call or coexist in the cache, which would make the initial GRPO
        // ratios drift from 1. Keeping only the prefix plus a single
        // completion resident also bounds the KV cache by one sequence -- the
        // whole group at once would overflow the training context's n_ctx
        // cells (prompt + group_size * completion greatly exceeds n_ctx).
        batch_guard completion_batch(static_cast<int32_t>(llama_n_ubatch(ctx)));
        // One full scoring pass over every branch, from an empty cache.
        // `device_gather` selects where log p(target) is reduced; the two are
        // meant to agree to within F32 noise, and the pass reports back rather
        // than emitting a non-finite score if the device path ever hands out a
        // probability that underflowed to zero.
        auto score_all_branches = [&](bool device_gather) -> pass_result {
            clear_context_memory(*state);
            if (!decode_shared_prefix()) {
                return pass_result::failed;
            }
            for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
                if (isolated_branches) {
                    // Full-stream copy only: a partial one would make the windowed
                    // SWA caches of other contexts unsound, and nothing here needs
                    // it (the branch always starts at the end of the prefix).
                    llama_memory_seq_cp(memory, prefix_seq, branch_seq, -1, -1);
                }
                size_t completion_index = 0;
                while (completion_index < out_n_logprobs[sequence]) {
                    const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
                            llama_n_ubatch(ctx), out_n_logprobs[sequence] - completion_index));
                    llama_batch & batch = completion_batch.batch;
                    batch.n_tokens = static_cast<int32_t>(count);
                    for (uint32_t row = 0; row < count; ++row) {
                        const size_t source = n_prompt - 1 + completion_index + row;
                        batch.token[row] = sequences[sequence].tokens[source];
                        batch.pos[row] = static_cast<llama_pos>(source);
                        batch.n_seq_id[row] = 1;
                        batch.seq_id[row][0] = branch_seq;
                        batch.logits[row] = true;
                    }
                    if (device_gather) {
                        // Targets are indexed by batch token, and every row of this
                        // batch emits, so the arrays line up one to one.
                        gather_targets.assign(
                                sequences[sequence].tokens + n_prompt + completion_index,
                                sequences[sequence].tokens + n_prompt + completion_index + count);
                        if (!llama_set_target_logprobs(ctx, gather_targets.data(), count)) {
                            set_error("shared-prefix suffix scoring could not request target logprobs");
                            return pass_result::failed;
                        }
                    }
                    if (core.decode(batch) != 0) {
                        set_error("llama_decode failed during shared-prefix suffix scoring");
                        return pass_result::failed;
                    }
                    for (uint32_t row = 0; row < count; ++row) {
                        float logprob = 0.0f;
                        if (device_gather) {
                            logprob = llama_get_target_logprob_ith(ctx, static_cast<int32_t>(row));
                            if (!std::isfinite(logprob)) {
                                // Only reachable when the gathered probability
                                // underflowed F32 (p < ~1e-38 for a token the policy
                                // itself produced). Redo the call on the host
                                // oracle, which keeps its reduction in double,
                                // rather than feed -inf into a GRPO ratio.
                                return pass_result::needs_host_reduction;
                            }
                        } else {
                            const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(row));
                            if (!logits) {
                                set_error("shared-prefix suffix scoring produced no logits");
                                return pass_result::failed;
                            }
                            logprob = token_logprob(logits, n_vocab,
                                    sequences[sequence].tokens[n_prompt + completion_index + row],
                                    logprob_work);
                        }
                        out_logprobs[sequence*out_stride + completion_index + row] = logprob;
                    }
                    state->scoring_stats.scored_positions += count;
                    if (device_gather) {
                        state->scoring_stats.device_logprob_positions += count;
                    }
                    completion_index += count;
                }
                // Evict this branch. Dropping a whole sequence is unconditionally
                // supported; the mid-sequence rollback of the single-slot path is
                // not, and when it is refused it fails before mutating anything,
                // which leaves the attention cache holding this completion's cells.
                // Ignoring that let stale cells accumulate across branches until a
                // later decode's position bookkeeping went inconsistent with the KV
                // cache, so the recovery is to rebuild the prefix from a clean
                // cache -- correct, at one prompt prefill per branch. The counters
                // make that degradation visible instead of merely slow.
                const bool evicted = isolated_branches
                        ? llama_memory_seq_rm(memory, branch_seq, -1, -1)
                        : llama_memory_seq_rm(memory, prefix_seq, static_cast<llama_pos>(prefix_len), -1);
                if (!evicted) {
                    state->scoring_stats.branch_evictions_refused++;
                    clear_context_memory(*state);
                    const bool is_last_sequence = sequence + 1 == n_sequences;
                    if (!is_last_sequence) {
                        state->scoring_stats.prefix_reprefills++;
                        if (!decode_shared_prefix()) {
                            return pass_result::failed;
                        }
                    }
                }
            }
            return pass_result::ok;
        };

        // A pass that gives up part way through has already counted the
        // positions it reduced, the prefix it decoded and any eviction it was
        // refused, and the host retry redoes every one of them from an empty
        // cache. Rewinding to the pre-pass values (taken after `calls++`, so the
        // call itself stays counted once) keeps the retry the only pass this
        // call reports: otherwise `device_logprob_fraction` credits abandoned
        // device positions and `prefix_decodes` / `prefix_reprefills` report a
        // degradation twice per affected call.
        const retro_scoring_stats before = state->scoring_stats;
        pass_result result = score_all_branches(device_gather_available);
        if (result == pass_result::needs_host_reduction) {
            state->scoring_stats = before;
            result = score_all_branches(false);
        }
        clear_context_memory(*state);
        return result == pass_result::ok ? 0 : -1;
    });
}

int set_lora_enabled_impl(retro_trainer * trainer, bool enabled) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        // Adapter changes invalidate decoded sequence memory.
        clear_context_memory(*state);
        if (state->generation_ctx) {
            clear_context_memory(state->generation_ctx.get());
        }
        if (!enabled) {
            llama_set_adapters_lora(state->ctx.get(), nullptr, 0, nullptr);
            if (state->generation_ctx) {
                llama_set_adapters_lora(state->generation_ctx.get(), nullptr, 0, nullptr);
            }
            return 0;
        }
        if (!state->adapter) {
            set_error("cannot enable LoRA: no adapter exists");
            return -1;
        }
        llama_adapter_lora * adapters[] = { state->adapter.get() };
        float scales[] = { 1.0f };
        llama_set_adapters_lora(state->ctx.get(), adapters, 1, scales);
        if (state->generation_ctx) {
            llama_set_adapters_lora(state->generation_ctx.get(), adapters, 1, scales);
        }
        return 0;
    });
}

int hidden_size_impl(retro_trainer * trainer, uint32_t * out_n_embd) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state || !out_n_embd) {
            if (state) set_error("out_n_embd is required");
            return -1;
        }
        const int32_t n_embd = llama_model_n_embd_out(state->model.get());
        if (n_embd <= 0) {
            set_error("model reports no output embedding width");
            return -1;
        }
        *out_n_embd = static_cast<uint32_t>(n_embd);
        return 0;
    });
}

int hidden_states_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_features,
        size_t n_features_max) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!tokens || n_tokens == 0) {
            set_error("hidden-state extraction requires tokens");
            return -1;
        }
        if (!out_features) {
            set_error("out_features is required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }
        const size_t n_embd =
                static_cast<size_t>(llama_model_n_embd_out(state->model.get()));
        if (n_features_max < n_tokens * n_embd) {
            set_error("out_features buffer is too small");
            return -1;
        }

        llama_context * ctx = state->ctx.get();
        // Embedding output changes the decode graph; restore the plain-logits
        // configuration on every exit path so generation, scoring, and the
        // optimizer keep their usual graphs.
        llama_set_embeddings(ctx, true);
        clear_context_memory(*state);

        const bool ok = decode_states_emitting_outputs(
                core, tokens, n_tokens, 0,
                "llama_decode failed during hidden-state extraction",
                // Hidden states, not logits: nothing to request per decode.
                [](size_t, uint32_t) { return true; },
                [&](size_t state_index, uint32_t batch_index) {
                    const float * row = llama_get_embeddings_ith(
                            ctx, static_cast<int32_t>(batch_index));
                    if (!row) {
                        set_error("decode produced no hidden state for a requested position");
                        return false;
                    }
                    std::memcpy(out_features + state_index * n_embd,
                            row, n_embd * sizeof(float));
                    return true;
                });

        clear_context_memory(*state);
        llama_set_embeddings(ctx, false);
        return ok ? 0 : -1;
    });
}

int score_token_suffix_and_hidden_states_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs,
        float * out_features,
        size_t n_features_max) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!tokens || n_prompt == 0 || n_prompt >= n_tokens) {
            set_error("combined scoring requires a prompt and completion tokens");
            return -1;
        }
        if (!out_logprobs || !out_features) {
            set_error("combined scoring output buffers are required");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx(state->ctx.get());
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }

        const size_t n_embd =
                static_cast<size_t>(llama_model_n_embd_out(state->model.get()));
        const size_t n_completion = n_tokens - n_prompt;
        if (n_embd == 0 || n_completion > n_features_max / n_embd) {
            set_error("combined hidden-state output buffer is too small");
            return -1;
        }
        const size_t n_vocab = core.n_vocab();
        const size_t first_state = n_prompt - 1;
        const size_t n_states = n_tokens - 1;
        std::vector<double> logprob_work;

        llama_context * ctx = state->ctx.get();
        // Embedding output changes the decode graph; restore the plain-logits
        // configuration on every exit path so generation, scoring, and the
        // optimizer keep their usual graphs.
        llama_set_embeddings(ctx, true);
        clear_context_memory(*state);

        // One decode carries both outputs only when the model keeps its LM
        // head in the embeddings graph: the LFM2 family drops t_logits there,
        // so llama_get_logits_ith comes back null for every row. Probe that
        // once on a single token. When the head is gone, the logprobs come
        // from a second pass on the plain-logits graph - the very graph the
        // single-purpose scoring path runs - and the hidden states from the
        // embeddings pass, so both outputs stay bit-identical to their
        // single-purpose counterparts whichever branch runs.
        bool combined_graph = false;
        {
            batch_guard probe(1);
            llama_batch & batch = probe.batch;
            batch.n_tokens = 1;
            batch.token[0] = tokens[0];
            batch.pos[0] = 0;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            combined_graph = core.decode(batch) == 0
                    && llama_get_logits_ith(ctx, 0) != nullptr;
            clear_context_memory(*state);
        }

        if (combined_graph) {
            const bool ok = decode_states_emitting_outputs(
                    core, tokens, n_states, first_state,
                    "llama_decode failed during combined scoring",
                    // This pass needs the hidden states of the same rows, so the
                    // logits have to come back anyway: no device gather here.
                    [](size_t, uint32_t) { return true; },
                    [&](size_t state_index, uint32_t batch_index) {
                        const float * logits =
                                llama_get_logits_ith(ctx, static_cast<int32_t>(batch_index));
                        const float * features = llama_get_embeddings_ith(
                                ctx, static_cast<int32_t>(batch_index));
                        if (!logits || !features) {
                            set_error("combined decode did not produce all requested outputs");
                            return false;
                        }
                        const size_t output_index = state_index - first_state;
                        out_logprobs[output_index] = token_logprob(
                                logits, n_vocab, tokens[state_index + 1], logprob_work);
                        std::memcpy(out_features + output_index * n_embd,
                                features, n_embd * sizeof(float));
                        return true;
                    });

            clear_context_memory(*state);
            llama_set_embeddings(ctx, false);
            return ok ? 0 : -1;
        }

        // No head in the embeddings graph: score the suffix on the
        // plain-logits graph first (host reduction, spelled exactly like the
        // scoring path's fallback), then re-decode for the hidden states.
        llama_set_embeddings(ctx, false);
        clear_context_memory(*state);
        const bool scored = decode_states_emitting_outputs(
                core, tokens, n_states, first_state,
                "llama_decode failed during combined scoring",
                [](size_t, uint32_t) { return true; },
                [&](size_t state_index, uint32_t batch_index) {
                    const float * logits =
                            llama_get_logits_ith(ctx, static_cast<int32_t>(batch_index));
                    if (!logits) {
                        set_error("combined decode did not produce logits");
                        return false;
                    }
                    out_logprobs[state_index - first_state] = token_logprob(
                            logits, n_vocab, tokens[state_index + 1], logprob_work);
                    return true;
                });
        if (!scored) {
            clear_context_memory(*state);
            return -1;
        }
        clear_context_memory(*state);

        llama_set_embeddings(ctx, true);
        clear_context_memory(*state);
        const bool featured = decode_states_emitting_outputs(
                core, tokens, n_states, first_state,
                "llama_decode failed during combined scoring",
                [](size_t, uint32_t) { return true; },
                [&](size_t state_index, uint32_t batch_index) {
                    const float * features = llama_get_embeddings_ith(
                            ctx, static_cast<int32_t>(batch_index));
                    if (!features) {
                        set_error("combined decode did not produce hidden states");
                        return false;
                    }
                    std::memcpy(out_features + (state_index - first_state) * n_embd,
                            features, n_embd * sizeof(float));
                    return true;
                });

        clear_context_memory(*state);
        llama_set_embeddings(ctx, false);
        return featured ? 0 : -1;
    });
}

int probe_token_logprob_impl(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        bool vectorized,
        float * out_logprob) {
    return boundary([&]() -> int {
        if (!logits || !out_logprob || n_vocab == 0
                || token < 0 || static_cast<size_t>(token) >= n_vocab) {
            set_error("invalid token-logprob probe inputs");
            return -1;
        }
        std::vector<double> work;
        *out_logprob = vectorized
                ? token_logprob(logits, n_vocab, token, work)
                : token_logprob_scalar(logits, n_vocab, token);
        return 0;
    });
}

int detokenize_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        bool unparse_special,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        if (n_tokens > 0 && !tokens) {
            set_error("tokens are required");
            return -1;
        }
        if (n_tokens > static_cast<size_t>(INT32_MAX)) {
            set_error("too many tokens to detokenize");
            return -1;
        }
        gen_core core = core_over(*state, state->ctx.get());
        if (!core.validate_tokens(tokens, n_tokens)) {
            return -1;
        }

        if (n_tokens == 0) {
            *out_n_bytes = 0;
            if (buffer && n_buffer > 0) {
                buffer[0] = '\0';
            }
            return 0;
        }

        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        return render_string_out(
                buffer, n_buffer, out_n_bytes, "detokenization failed",
                [&](char * destination, int32_t capacity) {
                    const int32_t written = llama_detokenize(
                            vocab, tokens, static_cast<int32_t>(n_tokens),
                            destination, capacity,
                            /*remove_special =*/ true, unparse_special);
                    // A negative result only reports the required size.
                    return written < 0 ? -written : written;
                });
    });
}

} // namespace retro
