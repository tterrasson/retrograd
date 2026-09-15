// Rollout support for policy-gradient training: sampled generation,
// teacher-forced logprob scoring, and detokenization. Everything here is
// forward-only; the differentiable objective lives in retro_training.cpp.
#include "retro_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace retro {

namespace {

struct batch_guard {
    llama_batch batch;
    explicit batch_guard(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {}
    ~batch_guard() { llama_batch_free(batch); }
    batch_guard(const batch_guard &) = delete;
    batch_guard & operator=(const batch_guard &) = delete;
};

// Every training-owned decode in this file goes through here, so the duty-cycle
// limiter sees each of them: generation, prefill, behavior and reference
// scoring, re-scoring between rollout epochs, and the forward-only evaluation
// paths built on the same helpers. A raw `llama_decode` left in this file is a
// coverage hole rather than a style question, and `rollout_decode_sites` in
// tests/duty_cycle_coverage.rs is what says so.
//
// Disabled, this is exactly `llama_decode(ctx, batch)` - one predictable branch
// on a member of the state the caller already holds, no clock read and, above
// all, no fence.
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
int decode_with_duty_cycle(
        trainer_state & state,
        llama_context * ctx,
        llama_batch & batch) {
    if (!state.duty_cycle.enabled()) {
        return llama_decode(ctx, batch);
    }
    state.duty_cycle.begin_window();
    const int status = llama_decode(ctx, batch);
    // Fence whatever was submitted, failure included: the caller is about to
    // tear the operation down and must not do so with work still in flight.
    llama_synchronize(ctx);
    state.duty_cycle.account_window();
    // The debt is recorded either way - the device did the work - but a failing
    // decode is unwinding into an error the caller is waiting on, so the repayment
    // moves to the next boundary rather than delaying it.
    if (status == 0) {
        state.duty_cycle.idle_if_needed();
    }
    return status;
}

bool validate_tokens(const trainer_state & state, const int32_t * tokens, size_t n_tokens) {
    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t i = 0; i < n_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) {
            std::ostringstream message;
            message << "token " << tokens[i] << " at index " << i
                    << " is outside the vocabulary of size " << n_vocab;
            set_error(message.str());
            return false;
        }
    }
    return true;
}

// Decodes tokens at absolute positions pos0.. in n_batch chunks and returns a
// borrowed pointer to the final requested logits row. The pointer remains valid
// until the next llama_decode on this context, which is exactly the lifetime the
// token sampler needs. The caller owns the batch so incremental generation can
// reuse its allocations across every sampled token.
bool decode_span_last_logits(
        trainer_state & state,
        llama_context * ctx,
        const int32_t * tokens,
        size_t n_tokens,
        int64_t pos0,
        int64_t logits_from,
        llama_batch & batch,
        const float * & out_logits) {
    const uint32_t n_batch = llama_n_batch(ctx);

    out_logits = nullptr;
    size_t offset = 0;
    while (offset < n_tokens) {
        const uint32_t count =
                static_cast<uint32_t>(std::min<size_t>(n_batch, n_tokens - offset));
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t pos = pos0 + static_cast<int64_t>(offset) + i;
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = static_cast<llama_pos>(pos);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // `out_logits` is a reference to the result pointer, not an enable
            // flag. Logits are requested from logits_from through the end.
            batch.logits[i] = pos >= logits_from;
        }
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
            set_error("llama_decode failed during rollout");
            return false;
        }
        const int64_t last_pos = pos0 + static_cast<int64_t>(offset + count - 1);
        if (last_pos >= logits_from) {
            out_logits = llama_get_logits_ith(ctx, static_cast<int32_t>(count - 1));
            if (!out_logits) {
                set_error("decode produced no logits for the requested positions");
                return false;
            }
        }
        offset += count;
    }
    if (!out_logits) {
        set_error("decode produced no final logits row");
        return false;
    }
    return true;
}

// The two terms of a row's log-softmax denominator, kept apart rather than
// summed: every reduction below spells the value as `logit - max - log_sum`,
// and folding the pair into one normalizer would move the rounding of a
// log-probability that other paths compare against bit for bit.
struct log_softmax_terms {
    float max;
    double log_sum;
};

// Scalar reference for the vectorized scoring path and the portable fallback.
log_softmax_terms log_softmax_terms_scalar(const float * logits, size_t n_vocab) {
    float max = -INFINITY;
    for (size_t i = 0; i < n_vocab; ++i) {
        max = std::max(max, logits[i]);
    }
    double sum = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - max);
    }
    return { max, std::log(sum) };
}

// Denominator of the temperature-1 log-softmax of one logits row. Accelerate
// evaluates exp and the final double reduction vectorially on Apple hosts;
// callers reuse `work` across every logits row in a scoring operation.
log_softmax_terms row_log_softmax_terms(
        const float * logits,
        size_t n_vocab,
        std::vector<double> & work) {
#if defined(__APPLE__)
    if (n_vocab <= static_cast<size_t>(INT_MAX)) {
        float max = -INFINITY;
        vDSP_maxv(logits, 1, &max, n_vocab);
        work.resize(n_vocab);
        vDSP_vspdp(logits, 1, work.data(), 1, n_vocab);
        const double shift = -static_cast<double>(max);
        vDSP_vsaddD(work.data(), 1, &shift, work.data(), 1, n_vocab);
        const int count = static_cast<int>(n_vocab);
        vvexp(work.data(), work.data(), &count);
        double sum = 0.0;
        vDSP_sveD(work.data(), 1, &sum, n_vocab);
        return { max, std::log(sum) };
    }
#else
    (void) work;
#endif
    return log_softmax_terms_scalar(logits, n_vocab);
}

float token_logprob_scalar(const float * logits, size_t n_vocab, int32_t token) {
    const log_softmax_terms terms = log_softmax_terms_scalar(logits, n_vocab);
    return static_cast<float>(
            static_cast<double>(logits[token]) - terms.max - terms.log_sum);
}

// Temperature-1 log-softmax value of `token` under a logits row.
float token_logprob(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        std::vector<double> & work) {
    const log_softmax_terms terms = row_log_softmax_terms(logits, n_vocab, work);
    return static_cast<float>(
            static_cast<double>(logits[token]) - terms.max - terms.log_sum);
}

// The `k` largest entries of one logits row, as ids and log-probabilities, in
// decreasing probability order. `order` is the reusable index permutation, so a
// whole suffix costs one vocabulary-sized allocation rather than one per
// position.
//
// Sorted rather than merely selected: `std::nth_element` would answer "which k"
// for the same price, but every consumer of this row reads column 0 as the
// argmax - top-1 agreement, and the reference column of an offline sidecar - so
// the order is part of the contract and not a convenience.
//
// Ties are broken by ascending id. Two equal logits are common on a quantized
// model, and without a total order the argmax of a row would depend on the
// pivot the selection happened to pick.
void top_token_logprobs(
        const float * logits,
        size_t n_vocab,
        size_t k,
        std::vector<int32_t> & order,
        std::vector<double> & work,
        int32_t * out_ids,
        float * out_logprobs) {
    const log_softmax_terms terms = row_log_softmax_terms(logits, n_vocab, work);
    order.resize(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    const auto by_logit = [&](int32_t left, int32_t right) {
        if (logits[left] != logits[right]) {
            return logits[left] > logits[right];
        }
        return left < right;
    };
    std::partial_sort(order.begin(), order.begin() + k, order.end(), by_logit);
    for (size_t j = 0; j < k; ++j) {
        const int32_t id = order[j];
        out_ids[j] = id;
        // The same expression the scalar path spells, so a k = 1 row is the
        // log-probability `score_token_suffix` would return for that id and not
        // a value that merely rounds to it.
        out_logprobs[j] = static_cast<float>(
                static_cast<double>(logits[id]) - terms.max - terms.log_sum);
    }
}

// Vocab-sized sampling buffers, reused across the tokens of one generation so
// they are allocated once per call instead of once per sampled token.
struct sampler_scratch {
    std::vector<double> probs;
    std::vector<int32_t> order;
    std::vector<int32_t> selected;
    float max = -INFINITY;
    double total = 0.0;
};

bool fast_top_p_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("RETRO_FAST_TOP_P");
        return value && value[0] != '\0' && std::strcmp(value, "0") != 0
                && std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

int32_t sample_token(
        const float * logits,
        size_t n_vocab,
        float temperature,
        float top_p,
        std::mt19937 & rng,
        sampler_scratch & scratch) {
    float max = -INFINITY;
    for (size_t i = 0; i < n_vocab; ++i) {
        max = std::max(max, logits[i]);
    }
    scratch.max = max;
    std::vector<double> & probs = scratch.probs;
    probs.resize(n_vocab);
    double total = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((static_cast<double>(logits[i]) - max) / temperature);
        total += probs[i];
    }
    scratch.total = total;

    // Strictly on-policy Dr. GRPO uses top_p == 1. Avoid sorting the entire
    // vocabulary in that common path; a cumulative categorical draw is exact.
    if (top_p == 1.0f) {
        std::uniform_real_distribution<double> uniform(0.0, total);
        const double draw = uniform(rng);
        double acc = 0.0;
        for (size_t i = 0; i < n_vocab; ++i) {
            acc += probs[i];
            if (draw <= acc) {
                return static_cast<int32_t>(i);
            }
        }
        return static_cast<int32_t>(n_vocab - 1);
    }

    std::vector<int32_t> & order = scratch.order;
    order.resize(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    const double target = static_cast<double>(top_p) * total;
    double kept = 0.0;
    size_t n_keep = 0;
    const std::vector<int32_t> * candidates = &order;
    if (fast_top_p_enabled()) {
        // Opt-in heap selection avoids sorting the unused vocabulary tail.
        // Ties use token id as a deterministic secondary key; this can differ
        // from the fallback unstable sort and is therefore never default.
        auto less_prob = [&](int32_t a, int32_t b) {
            return probs[a] != probs[b] ? probs[a] < probs[b] : a > b;
        };
        std::make_heap(order.begin(), order.end(), less_prob);
        std::vector<int32_t> & selected = scratch.selected;
        selected.clear();
        auto heap_end = order.end();
        while (heap_end != order.begin() && kept < target) {
            std::pop_heap(order.begin(), heap_end, less_prob);
            --heap_end;
            selected.push_back(*heap_end);
            kept += probs[*heap_end];
        }
        candidates = &selected;
        n_keep = selected.size();
    } else {
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return probs[a] > probs[b];
        });
        while (n_keep < n_vocab) {
            kept += probs[order[n_keep]];
            ++n_keep;
            if (kept >= target) {
                break;
            }
        }
    }

    std::uniform_real_distribution<double> uniform(0.0, kept);
    const double draw = uniform(rng);
    double acc = 0.0;
    for (size_t i = 0; i < n_keep; ++i) {
        acc += probs[(*candidates)[i]];
        if (draw <= acc) {
            return (*candidates)[i];
        }
    }
    return (*candidates)[n_keep - 1];
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

// Clearing a context the generation cache tracks. The pair is never split:
// residency that outlives the cells it describes is the one way this cache can
// be wrong rather than merely cold.
void clear_generation_memory(trainer_state & state, llama_context * ctx) {
    clear_context_memory(ctx);
    state.forget_generation_kv();
}

// Outcome of one scoring pass. A device-gathered log-probability that came back
// non-finite is not an error: the pass reports it so the caller can redo the
// whole request on the host oracle, which keeps its reduction in double.
enum class score_pass { ok, needs_host_reduction, failed };

// Whether the active device can gather log softmax(logits)[target] inside the
// decode graph (docs/engineering/optims/SAMPLING.md S6). Read per call rather than cached:
// RETRO_DEVICE_LOGPROBS=0 is both the escape hatch and how the two reductions
// are compared on a given model, including from a test that flips it.
bool device_logprob_gather_enabled(const trainer_state & state) {
    const char * value = std::getenv("RETRO_DEVICE_LOGPROBS");
    const bool disabled = value
            && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0);
    return !disabled && state.gpu_active && state.cap_device_logprobs;
}

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
        trainer_state & state,
        const int32_t * tokens,
        size_t n_states,
        size_t first_state,
        const char * decode_error,
        OnBeforeOutputs && on_before_outputs,
        OnOutput && on_output) {
    llama_context * ctx = state.ctx.get();
    const uint32_t n_batch = llama_n_batch(ctx);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);

    batch_guard guard(static_cast<int32_t>(n_batch));
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
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
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

// Scores targets [first_target, n_tokens): each emitted position is reduced to
// the scalar log-probability of the next token.
// `device_gather` decides where that reduction happens. On the device the
// decode returns one float per scored position instead of an n_vocab logits row
// (608 KiB at Qwen's vocabulary), which is the same S6 saving the shared-prefix
// scorer takes; this is the path the per-optimizer-step re-scoring and the
// fixed-reference pass go through, and they run far more often than the
// behavior pass does.
score_pass score_token_range_pass(
        trainer_state & state,
        const int32_t * tokens,
        size_t n_tokens,
        size_t first_target,
        bool device_gather,
        float * out_logprobs) {
    llama_context * ctx = state.ctx.get();
    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    const size_t first_state = first_target - 1;
    std::vector<double> logprob_work;
    std::vector<int32_t> gather_targets;
    score_pass result = score_pass::ok;

    const bool decoded = decode_states_emitting_outputs(
            state, tokens, n_tokens - 1, first_state,
            "llama_decode failed during token scoring",
            [&](size_t chunk_first_state, uint32_t count) {
                if (!device_gather) {
                    return true;
                }
                // Targets are indexed by batch token and every row of an
                // emitting chunk requests logits, so the arrays line up one to
                // one: state s predicts tokens[s + 1].
                gather_targets.assign(
                        tokens + chunk_first_state + 1,
                        tokens + chunk_first_state + 1 + count);
                if (!llama_set_target_logprobs(ctx, gather_targets.data(), count)) {
                    set_error("token scoring could not request target logprobs");
                    return false;
                }
                return true;
            },
            [&](size_t state_index, uint32_t batch_index) {
                float logprob = 0.0f;
                if (device_gather) {
                    logprob = llama_get_target_logprob_ith(
                            ctx, static_cast<int32_t>(batch_index));
                    if (!std::isfinite(logprob)) {
                        // Only reachable when the gathered probability
                        // underflowed F32 (p < ~1e-38 for a token the policy
                        // itself produced). Redo the request on the host oracle
                        // rather than feed -inf into a policy ratio.
                        result = score_pass::needs_host_reduction;
                        return false;
                    }
                } else {
                    const float * row =
                            llama_get_logits_ith(ctx, static_cast<int32_t>(batch_index));
                    if (!row) {
                        set_error("decode produced no logits for token scoring");
                        return false;
                    }
                    logprob = token_logprob(
                            row, n_vocab, tokens[state_index + 1], logprob_work);
                }
                out_logprobs[state_index - first_state] = logprob;
                state.scoring_stats.scored_positions++;
                if (device_gather) {
                    state.scoring_stats.device_logprob_positions++;
                }
                return true;
            });
    if (!decoded) {
        return result == score_pass::needs_host_reduction ? result : score_pass::failed;
    }
    return score_pass::ok;
}

bool score_token_range(
        trainer_state & state,
        const int32_t * tokens,
        size_t n_tokens,
        size_t first_target,
        float * out_logprobs) {
    // The device pass counts each position as it reduces it, and the underflow
    // that sends it to the host is only discovered part way through. Rewinding
    // the counters to their pre-pass values makes the retry the only pass this
    // call reports, so `device_logprob_fraction` reads as "the host reduced this
    // range" instead of crediting the abandoned device positions twice.
    const retro_scoring_stats before = state.scoring_stats;
    score_pass result = score_token_range_pass(
            state, tokens, n_tokens, first_target,
            device_logprob_gather_enabled(state), out_logprobs);
    if (result == score_pass::needs_host_reduction) {
        state.scoring_stats = before;
        // The retry re-decodes from position zero, so it needs the same empty
        // cache the first pass started from.
        clear_context_memory(state);
        result = score_token_range_pass(
                state, tokens, n_tokens, first_target, /*device_gather =*/ false, out_logprobs);
    }
    return result == score_pass::ok;
}

bool validate_sampling(const retro_sampling_params & sampling) {
    if (!(sampling.temperature > 0.0f) || !std::isfinite(sampling.temperature)) {
        set_error("sampling temperature must be finite and greater than zero");
        return false;
    }
    if (!(sampling.top_p > 0.0f && sampling.top_p <= 1.0f)) {
        set_error("sampling top_p must be in (0, 1]");
        return false;
    }
    if (sampling.max_new_tokens == 0) {
        set_error("sampling max_new_tokens must be greater than zero");
        return false;
    }
    return true;
}

// On by default wherever the capability is present: sampling on the device
// keeps the n_vocab logits row there instead of copying it to the host for
// every sampled token. RETRO_DEVICE_SAMPLING=0 (or "false") is the escape
// hatch. Callers never have to handle a refusal - when llama.cpp cannot attach
// the chain, backend_sampler_scope reports it unavailable and, if a single
// sampler op stayed on the host mid-generation, sample_backend_token returns 0
// and the caller replays the whole request through the CPU oracle.
bool device_sampling_requested(const trainer_state & state, const float * out_logprobs) {
    const char * value = std::getenv("RETRO_DEVICE_SAMPLING");
    const bool disabled = value
            && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0);
    // Generation logprobs describe the unmodified temperature-1 policy. The
    // public API that requests them (retro_trainer_generate*) therefore remains
    // on the exact CPU oracle; GRPO rollout collection passes nullptr because it
    // scores behavior separately, so it takes this path.
    return !disabled && out_logprobs == nullptr && state.gpu_active
            && state.cap_device_sampling;
}

// One backend sampler chain per row of a call, attached to the sequence id
// the row decodes in. `slots[row]` is that id: the caller's index only while no
// prefix cache is in play, and whatever slot the placement handed the row
// otherwise. A chain on any other id samples the row with another row's seed,
// or - for a slot past the row count - not at all, which llama.cpp reports as
// a null token and this runtime reads as "fall back to the CPU oracle", at the
// price of every resident prefix.
class backend_sampler_scope {
public:
    backend_sampler_scope(
            llama_context * ctx,
            const retro_sampling_params * params,
            std::vector<llama_seq_id> slots)
        : ctx_(ctx), slots_(std::move(slots)) {
        chains_.reserve(slots_.size());
        llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        chain_params.no_perf = true;
        for (size_t row = 0; row < slots_.size(); ++row) {
            llama_sampler * chain = llama_sampler_chain_init(chain_params);
            llama_sampler_chain_add(chain, llama_sampler_init_temp(params[row].temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(params[row].top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(params[row].seed));
            chains_.push_back(chain);
            if (!llama_set_sampler(ctx_, slots_[row], chain)) {
                available_ = false;
                break;
            }
            attached_++;
        }
    }

    ~backend_sampler_scope() {
        for (size_t row = 0; row < attached_; ++row) {
            llama_set_sampler(ctx_, slots_[row], nullptr);
        }
        for (llama_sampler * chain : chains_) {
            llama_sampler_free(chain);
        }
    }

    backend_sampler_scope(const backend_sampler_scope &) = delete;
    backend_sampler_scope & operator=(const backend_sampler_scope &) = delete;

    bool available() const { return available_ && attached_ == chains_.size(); }

private:
    llama_context * ctx_;
    std::vector<llama_seq_id> slots_;
    std::vector<llama_sampler *> chains_;
    size_t attached_ = 0;
    bool available_ = true;
};

// The slots of a call that places row `i` in sequence `i`: the shared-prompt
// paths, which clear the context on entry and own every sequence id.
std::vector<llama_seq_id> identity_slots(size_t n_sequences) {
    std::vector<llama_seq_id> slots(n_sequences);
    std::iota(slots.begin(), slots.end(), llama_seq_id{0});
    return slots;
}

bool decode_span_no_logits(
        trainer_state & state,
        llama_context * ctx,
        const int32_t * tokens,
        size_t n_tokens,
        int64_t pos0,
        llama_batch & batch) {
    const uint32_t n_batch = llama_n_batch(ctx);
    size_t offset = 0;
    while (offset < n_tokens) {
        const uint32_t count = static_cast<uint32_t>(
                std::min<size_t>(n_batch, n_tokens - offset));
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = static_cast<llama_pos>(pos0 + offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
            set_error("llama_decode failed during device-sampling prompt prefill");
            return false;
        }
        offset += count;
    }
    return true;
}

// Orders the sequences of a continuous-generation call by content rather than
// by caller slot: prompt length, prompt tokens, then sampling parameters.
// Every physical launch - prefill and decode alike - is then built in this
// order, so permuting the caller's rows permutes only the output mapping and
// leaves the decodes bit-identical. That is what makes each row's tokens depend
// on its own prompt and seed rather than on the slot it happens to occupy
// (`continuous_generation_preserves_per_prompt_seeded_rows`): once a prompt is
// prefilled in its own launches, the KV layout and the attention tiling a row
// sees follow the order the rows were submitted in, so that order has to come
// from the content.
std::vector<size_t> canonical_prefill_order(
        const retro_generation_sequence * sequences,
        size_t n_sequences) {
    std::vector<size_t> order(n_sequences);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        const retro_generation_sequence & a = sequences[left];
        const retro_generation_sequence & b = sequences[right];
        if (a.n_prompt != b.n_prompt) return a.n_prompt < b.n_prompt;
        for (size_t i = 0; i < a.n_prompt; ++i) {
            if (a.prompt_tokens[i] != b.prompt_tokens[i]) {
                return a.prompt_tokens[i] < b.prompt_tokens[i];
            }
        }
        if (a.sampling.seed != b.sampling.seed) return a.sampling.seed < b.sampling.seed;
        if (a.sampling.max_new_tokens != b.sampling.max_new_tokens) {
            return a.sampling.max_new_tokens < b.sampling.max_new_tokens;
        }
        if (a.sampling.temperature != b.sampling.temperature) {
            return a.sampling.temperature < b.sampling.temperature;
        }
        return a.sampling.top_p < b.sampling.top_p;
    });
    return order;
}

bool same_prompt(
        const retro_generation_sequence & left,
        const retro_generation_sequence & right) {
    return left.n_prompt == right.n_prompt
            && std::equal(left.prompt_tokens, left.prompt_tokens + left.n_prompt,
                    right.prompt_tokens);
}

// Where one requested sequence is going to live in the generation context, and
// how much of its prompt is already there.
// `slot` is a sequence id of that context, not the caller's index: the two
// coincided while every call started from an empty cache, and they must not now
// - the whole point is that a slot outlives the call that filled it.
struct kv_placement {
    llama_seq_id slot = 0;
    // Resident prompt tokens an earlier call already decoded. Always strictly
    // less than `n_prompt`: the terminal prompt position has to be decoded by
    // *this* call, since that is the row the first sampled token reads its
    // logits from.
    size_t reuse = 0;
};

std::vector<llama_seq_id> placement_slots(const std::vector<kv_placement> & placement) {
    std::vector<llama_seq_id> slots;
    slots.reserve(placement.size());
    for (const kv_placement & item : placement) {
        slots.push_back(item.slot);
    }
    return slots;
}

// Whether the generation prefix cache is on. Off restores the previous
// behaviour exactly - every call clears the context and prefills every prompt
// in full - which is what makes it the reference the reuse path is compared
// against on a given model, the same role RETRO_DEVICE_LOGPROBS=0 plays for the
// scorer's reduction.
bool generation_prefix_cache_enabled() {
    const char * value = std::getenv("RETRO_GENERATION_PREFIX_CACHE");
    return !(value && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0));
}

// How much of `prompt` the slot already holds, or zero when it holds something
// else.
// A hit is an exact *extension*: the resident tokens are a strict prefix of the
// prompt. Nothing less is accepted, and that is a constraint from the model
// rather than a simplification - dropping a diverged tail would mean removing a
// range from the middle of a sequence, which llama.cpp does not support for the
// recurrent half of a hybrid architecture, while dropping a whole sequence
// always works. So a divergence costs the slot, not a partial rewind.
size_t resident_prefix(
        const std::vector<int32_t> & resident,
        const retro_generation_sequence & item) {
    // `>=` and not `>`: a resident block as long as the prompt would leave the
    // terminal position already decoded, and this call needs its logits.
    if (resident.empty() || resident.size() >= item.n_prompt) {
        return 0;
    }
    return std::equal(resident.begin(), resident.end(), item.prompt_tokens)
            ? resident.size()
            : 0;
}

// Assigns every requested sequence a slot, reusing what each one can and
// dropping what it cannot.
// Walking `order` rather than the caller's slots keeps the assignment a pure
// function of the request's *content* and of the cache, which is the same
// property `canonical_prefill_order` exists to give the launches: two runs that
// issue the same calls place the same tokens in the same slots.
std::vector<kv_placement> place_generation_sequences(
        trainer_state & state,
        llama_context * ctx,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        const std::vector<size_t> & order,
        bool cache_enabled) {
    const size_t n_slots = static_cast<size_t>(llama_n_seq_max(ctx));
    std::vector<kv_placement> placement(n_sequences);
    if (!cache_enabled) {
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            placement[sequence].slot = static_cast<llama_seq_id>(sequence);
            placement[sequence].reuse = 0;
        }
        return placement;
    }
    state.generation_kv.resize(n_slots);
    state.generation_kv_used_at.resize(n_slots, 0);

    llama_memory_t memory = llama_get_memory(ctx);
    std::vector<bool> claimed(n_slots, false);
    for (const size_t sequence : order) {
        const retro_generation_sequence & item = sequences[sequence];
        size_t best_slot = n_slots;
        size_t best_reuse = 0;
        for (size_t slot = 0; slot < n_slots; ++slot) {
            if (claimed[slot]) {
                continue;
            }
            const size_t reuse = resident_prefix(state.generation_kv[slot], item);
            if (best_slot == n_slots || reuse > best_reuse) {
                best_slot = slot;
                best_reuse = reuse;
                continue;
            }
            if (reuse == best_reuse && reuse == 0) {
                // No candidate carries anything worth keeping, so the choice is
                // which slot to sacrifice: the one that has gone longest
                // without being extended.
                if (state.generation_kv_used_at[slot]
                        < state.generation_kv_used_at[best_slot]) {
                    best_slot = slot;
                }
            }
        }
        claimed[best_slot] = true;
        placement[sequence].slot = static_cast<llama_seq_id>(best_slot);
        placement[sequence].reuse = best_reuse;
        if (best_reuse == 0) {
            if (!state.generation_kv[best_slot].empty()) {
                state.generation_stats.evictions++;
            }
            // Dropping a whole sequence is unconditionally supported, unlike the
            // mid-sequence removal a partial rewind would need.
            if (memory) {
                llama_memory_seq_rm(memory, static_cast<llama_seq_id>(best_slot), -1, -1);
            }
            state.generation_kv[best_slot].clear();
        } else {
            state.generation_stats.hits++;
        }
        state.generation_stats.sequences++;
        state.generation_stats.prompt_tokens += item.n_prompt;
        state.generation_stats.reused_tokens += best_reuse;
        // `prefilled_tokens` is counted where positions are actually pushed: a
        // member of a copy group receives its prefix by copy and decodes one
        // position, not `n_prompt - reuse` of them.
    }
    return placement;
}

// Records what each slot holds once a call has finished sampling.
// The prompt, plus every sampled token *but the last*. That exception is the
// shape of the decode loop rather than an off-by-one: a token is decoded to
// produce the next one, so the token a sequence stopped on was emitted and
// never fed back - its position is not in the cache. Recording it would make
// the next call start its prefill one position past what the memory module
// holds, which llama.cpp refuses outright ("sequence positions must remain
// consecutive") instead of quietly attending over a hole.
void remember_generation_kv(
        trainer_state & state,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        const std::vector<kv_placement> & placement,
        const int32_t * out_tokens,
        size_t n_out_max,
        const size_t * out_n_tokens) {
    if (state.generation_kv.empty()) {
        return;
    }
    ++state.generation_kv_clock;
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        const retro_generation_sequence & item = sequences[sequence];
        const size_t slot = static_cast<size_t>(placement[sequence].slot);
        std::vector<int32_t> & resident = state.generation_kv[slot];
        const size_t decoded = out_n_tokens[sequence] > 0 ? out_n_tokens[sequence] - 1 : 0;
        resident.assign(item.prompt_tokens, item.prompt_tokens + item.n_prompt);
        resident.insert(resident.end(), out_tokens + sequence*n_out_max,
                out_tokens + sequence*n_out_max + decoded);
        state.generation_kv_used_at[slot] = state.generation_kv_clock;
    }
}

// Prefills every continuous-generation prompt in chunks of `n_batch` positions
// instead of one position per sequence and per launch. Positions stay strictly
// increasing from 0 within a sequence, which is what llama.cpp requires.
// Sequences that share a prompt are prefilled once. The canonical order already
// puts them side by side, so a group is a run of adjacent entries: its prefix is
// decoded into the group's leader, copied to the other members with
// `llama_memory_seq_cp`, and only the terminal token is then decoded per member
// - each one needs its own logits row, and on the device path its own sampler
// has to fire there. A GRPO wave of `prompts × group_size` sequences therefore
// costs `prompts` prefixes, not `prompts × group_size`.
// Logits are requested on terminal prompt rows only - the output projection over
// every prompt position was pure waste. `on_terminal(sequence, batch_row)` runs
// right after the decode that produced that row, since the next launch
// invalidates llama's output buffer.
template <typename OnTerminal>
bool prefill_continuous_prompts(
        trainer_state & state,
        llama_context * ctx,
        const retro_generation_sequence * sequences,
        const std::vector<size_t> & order,
        const std::vector<kv_placement> & placement,
        llama_batch & batch,
        const char * decode_error,
        OnTerminal on_terminal) {
    const size_t n_batch = static_cast<size_t>(llama_n_batch(ctx));
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("generation context has no sequence memory");
        return false;
    }
    // Rows of the pending launch whose logits a caller is waiting on.
    std::vector<std::pair<size_t, int32_t>> terminals;
    size_t count = 0;

    auto flush = [&]() -> bool {
        if (count == 0) {
            return true;
        }
        batch.n_tokens = static_cast<int32_t>(count);
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
            set_error(decode_error);
            return false;
        }
        for (const auto & terminal : terminals) {
            if (!on_terminal(terminal.first, terminal.second)) {
                return false;
            }
        }
        terminals.clear();
        count = 0;
        return true;
    };

    auto push = [&](size_t sequence, size_t position, int32_t token, bool logits) -> bool {
        if (count == n_batch && !flush()) {
            return false;
        }
        batch.token[count] = token;
        batch.pos[count] = static_cast<llama_pos>(position);
        batch.n_seq_id[count] = 1;
        batch.seq_id[count][0] = placement[sequence].slot;
        batch.logits[count] = logits;
        if (logits) {
            terminals.emplace_back(sequence, static_cast<int32_t>(count));
        }
        ++count;
        ++state.generation_stats.prefilled_tokens;
        return true;
    };

    size_t group_start = 0;
    while (group_start < order.size()) {
        const size_t leader = order[group_start];
        const retro_generation_sequence & item = sequences[leader];
        const size_t reuse = placement[leader].reuse;
        size_t group_end = group_start + 1;
        // Only a cold prompt may lead a copy group. A slot that already holds a
        // prefix must not be overwritten by someone else's whole-sequence copy,
        // and copying into it would throw away precisely what the call came here
        // to keep; a warm sequence is therefore its own group of one.
        if (reuse == 0) {
            while (group_end < order.size()
                    && placement[order[group_end]].reuse == 0
                    && same_prompt(sequences[order[group_end]], item)) {
                ++group_end;
            }
        }

        for (size_t position = reuse; position + 1 < item.n_prompt; ++position) {
            if (!push(leader, position, item.prompt_tokens[position], false)) {
                return false;
            }
        }
        // Each group gets its own launches: packing several prompts into one
        // would make a row's ubatch tiling depend on which prompts happen to
        // share its launch, which is what the per-slot independence guarantee
        // forbids. Within a group, the prefix has to be resident before it can
        // be copied, so a group with members flushes before the copies. A group
        // of one - every warm sequence is one - has nothing to copy and decodes
        // its prefix and its terminal in a single launch: the flush that used
        // to sit between them was the copy's, not the terminal's, and at one
        // sequence per group it doubled the launches of a whole turn.
        const bool copies = group_end > group_start + 1;
        if (copies) {
            if (!flush()) {
                return false;
            }
            for (size_t member = group_start + 1; member < group_end; ++member) {
                // Whole-sequence copy. `swa_full = false` on this context is
                // only sound while every copy here spans (-1, -1); a windowed
                // copy would invalidate that choice.
                llama_memory_seq_cp(memory, placement[leader].slot,
                        placement[order[member]].slot, -1, -1);
            }
        }

        const size_t terminal_position = item.n_prompt - 1;
        const int32_t terminal_token = item.prompt_tokens[terminal_position];
        for (size_t member = group_start; member < group_end; ++member) {
            if (!push(order[member], terminal_position, terminal_token, true)) {
                return false;
            }
        }
        if (!flush()) {
            return false;
        }
        group_start = group_end;
    }
    return true;
}

int sample_backend_token(
        llama_context * ctx,
        int32_t batch_index,
        size_t n_vocab,
        int32_t & out_token) {
    const llama_token token = llama_get_sampled_token_ith(ctx, batch_index);
    if (token == LLAMA_TOKEN_NULL) {
        // At least one sampler op stayed on the host. Re-run the request with
        // Retrograd's exact CPU oracle instead of mixing sampler implementations.
        return 0;
    }
    if (token < 0 || static_cast<size_t>(token) >= n_vocab) {
        set_error("device sampler produced an invalid token");
        return -1;
    }
    out_token = token;
    return 1;
}

// Returns 1 when the device path completed, 0 when llama.cpp could not attach
// the sampler (caller must use the CPU oracle), and -1 on a generation error.
int generate_batch_device(
        trainer_state & state,
        llama_context * ctx,
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        size_t n_out_max,
        size_t * out_n_tokens) {
    backend_sampler_scope samplers(ctx, sampling, identity_slots(n_sequences));
    if (!samplers.available()) {
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    clear_generation_memory(state, ctx);

    batch_guard prefill(static_cast<int32_t>(llama_n_batch(ctx)));
    if (n_prompt > 1 && !decode_span_no_logits(
            state, ctx, prompt_tokens, n_prompt - 1, 0, prefill.batch)) {
        clear_generation_memory(state, ctx);
        return -1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("generation context has no sequence memory");
        clear_generation_memory(state, ctx);
        return -1;
    }
    for (size_t sequence = 1; sequence < n_sequences; ++sequence) {
        llama_memory_seq_cp(memory, 0, static_cast<llama_seq_id>(sequence), -1, -1);
    }

    batch_guard decode_batch(static_cast<int32_t>(n_sequences));
    llama_batch & first = decode_batch.batch;
    first.n_tokens = static_cast<int32_t>(n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        first.token[sequence] = prompt_tokens[n_prompt - 1];
        first.pos[sequence] = static_cast<llama_pos>(n_prompt - 1);
        first.n_seq_id[sequence] = 1;
        first.seq_id[sequence][0] = static_cast<llama_seq_id>(sequence);
        first.logits[sequence] = true;
        out_n_tokens[sequence] = 0;
    }
    if (decode_with_duty_cycle(state, ctx, first) != 0) {
        set_error("llama_decode failed during device-sampling prompt terminal");
        clear_generation_memory(state, ctx);
        return -1;
    }

    std::vector<size_t> limits(n_sequences);
    std::vector<bool> live(n_sequences, true);
    std::vector<int32_t> sample_indices(n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        limits[sequence] = std::min<size_t>(
                std::min<size_t>(n_out_max, sampling[sequence].max_new_tokens),
                n_ctx - n_prompt);
        sample_indices[sequence] = static_cast<int32_t>(sequence);
    }
    std::vector<size_t> active;
    active.reserve(n_sequences);
    size_t n_live = n_sequences;
    while (n_live > 0) {
        active.clear();
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            if (!live[sequence]) continue;
            int32_t token = LLAMA_TOKEN_NULL;
            const int sample_result = sample_backend_token(
                    ctx, sample_indices[sequence], n_vocab, token);
            if (sample_result <= 0) {
                clear_generation_memory(state, ctx);
                return sample_result;
            }
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            out_n_tokens[sequence] = count + 1;
            if (llama_vocab_is_eog(vocab, token) || out_n_tokens[sequence] == limits[sequence]) {
                live[sequence] = false;
                --n_live;
            } else {
                active.push_back(sequence);
            }
        }
        if (active.empty()) break;
        llama_batch & batch = decode_batch.batch;
        batch.n_tokens = static_cast<int32_t>(active.size());
        for (size_t row = 0; row < active.size(); ++row) {
            const size_t sequence = active[row];
            const size_t count = out_n_tokens[sequence];
            batch.token[row] = out_tokens[sequence*n_out_max + count - 1];
            batch.pos[row] = static_cast<llama_pos>(n_prompt + count - 1);
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = static_cast<llama_seq_id>(sequence);
            batch.logits[row] = true;
        }
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
            set_error("llama_decode failed during device-sampled generation");
            clear_generation_memory(state, ctx);
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            sample_indices[active[row]] = static_cast<int32_t>(row);
        }
    }
    clear_generation_memory(state, ctx);
    return 1;
}

int generate_continuous_batch_device(
        trainer_state & state,
        llama_context * ctx,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        size_t n_out_max,
        size_t * out_n_tokens) {
    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        out_n_tokens[sequence] = 0;
    }
    const bool cache = generation_prefix_cache_enabled();
    if (!cache) {
        clear_generation_memory(state, ctx);
    }
    state.generation_stats.calls++;

    const std::vector<size_t> order = canonical_prefill_order(sequences, n_sequences);
    const std::vector<kv_placement> placement = place_generation_sequences(
            state, ctx, sequences, n_sequences, order, cache);
    // Samplers after the placement, because a chain is attached to the slot its
    // row decodes in and the placement is what decides that. Declining here
    // leaves the context exactly as the placement left it - evictions applied,
    // nothing decoded - so the oracle takes over without a clear.
    // Each sampler is activated exactly once, on its prompt's terminal row.
    // Earlier prompt positions produce no output, so they neither transfer
    // logits nor advance that sequence's random stream.
    std::vector<retro_sampling_params> params;
    params.reserve(n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        params.push_back(sequences[sequence].sampling);
    }
    backend_sampler_scope samplers(ctx, params.data(), placement_slots(placement));
    if (!samplers.available()) {
        return 0;
    }
    batch_guard prefill_batch(static_cast<int32_t>(llama_n_batch(ctx)));
    std::vector<int32_t> pending_tokens(n_sequences, LLAMA_TOKEN_NULL);
    int prefill_status = 1;
    if (!prefill_continuous_prompts(
            state, ctx, sequences, order, placement, prefill_batch.batch,
            "llama_decode failed during device-sampling continuous prefill",
            [&](size_t sequence, int32_t row) {
                const int sample_result = sample_backend_token(
                        ctx, row, n_vocab, pending_tokens[sequence]);
                if (sample_result <= 0) {
                    prefill_status = sample_result;
                    return false;
                }
                return true;
            })) {
        clear_generation_memory(state, ctx);
        return prefill_status <= 0 ? prefill_status : -1;
    }

    std::vector<size_t> limits(n_sequences);
    std::vector<bool> live(n_sequences, true);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        limits[sequence] = std::min<size_t>(
                std::min<size_t>(n_out_max, sequences[sequence].sampling.max_new_tokens),
                n_ctx - sequences[sequence].n_prompt);
    }
    batch_guard decode_batch(static_cast<int32_t>(n_sequences));
    std::vector<size_t> active;
    active.reserve(n_sequences);
    size_t n_live = n_sequences;
    while (n_live > 0) {
        active.clear();
        // Canonical order here too: decode rows must not carry caller slots
        // into the physical batch. See canonical_prefill_order.
        for (const size_t sequence : order) {
            if (!live[sequence]) continue;
            const int32_t token = pending_tokens[sequence];
            if (token == LLAMA_TOKEN_NULL) {
                set_error("device sampler has no pending token");
                clear_generation_memory(state, ctx);
                return -1;
            }
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            out_n_tokens[sequence] = count + 1;
            if (llama_vocab_is_eog(vocab, token) || out_n_tokens[sequence] == limits[sequence]) {
                live[sequence] = false;
                --n_live;
            } else {
                active.push_back(sequence);
            }
        }
        if (active.empty()) break;
        llama_batch & batch = decode_batch.batch;
        batch.n_tokens = static_cast<int32_t>(active.size());
        for (size_t row = 0; row < active.size(); ++row) {
            const size_t sequence = active[row];
            const size_t count = out_n_tokens[sequence];
            batch.token[row] = out_tokens[sequence*n_out_max + count - 1];
            batch.pos[row] = static_cast<llama_pos>(
                    sequences[sequence].n_prompt + count - 1);
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = placement[sequence].slot;
            batch.logits[row] = true;
        }
        if (decode_with_duty_cycle(state, ctx, batch) != 0) {
            set_error("llama_decode failed during continuous device sampling");
            clear_generation_memory(state, ctx);
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            const size_t sequence = active[row];
            const int sample_result = sample_backend_token(
                    ctx, static_cast<int32_t>(row), n_vocab, pending_tokens[sequence]);
            if (sample_result <= 0) {
                clear_generation_memory(state, ctx);
                return sample_result;
            }
        }
    }
    if (cache) {
        remember_generation_kv(state, sequences, n_sequences, placement,
                out_tokens, n_out_max, out_n_tokens);
    } else {
        clear_generation_memory(state, ctx);
    }
    return 1;
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
        size_t * out_n_tokens) {
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
        if (n_sequences > llama_n_seq_max(ctx)) {
            std::ostringstream message;
            message << "generation batch has " << n_sequences
                    << " sequences but the context supports "
                    << llama_n_seq_max(ctx);
            set_error(message.str());
            return -1;
        }
        if (n_sequences > SIZE_MAX / n_out_max) {
            set_error("generation batch output shape overflows size_t");
            return -1;
        }
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            if (!validate_sampling(sampling[sequence])) {
                return -1;
            }
        }
        if (!validate_tokens(*state, prompt_tokens, n_prompt)) {
            return -1;
        }
        const size_t n_ctx = llama_n_ctx_seq(ctx);
        if (n_prompt >= n_ctx) {
            std::ostringstream message;
            message << "prompt of " << n_prompt
                    << " tokens leaves no room to generate in context " << n_ctx;
            set_error(message.str());
            return -1;
        }

        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
        if (device_sampling_requested(*state, out_logprobs)) {
            const int result = generate_batch_device(*state, ctx, prompt_tokens, n_prompt,
                    sampling, n_sequences, out_tokens, n_out_max, out_n_tokens);
            if (result != 0) {
                return result > 0 ? 0 : -1;
            }
            // Unsupported backend sampler configuration: use the scalar sampler
            // with the same inputs as a transparent fallback.
        }
        batch_guard decode_batch(static_cast<int32_t>(llama_n_batch(ctx)));
        const float * logits = nullptr;
        clear_generation_memory(*state, ctx);
        if (!decode_span_last_logits(*state, ctx, prompt_tokens, n_prompt, 0,
                static_cast<int64_t>(n_prompt) - 1, decode_batch.batch, logits)) {
            clear_generation_memory(*state, ctx);
            return -1;
        }

        // The next decode invalidates llama's logits buffer, so keep the one
        // shared prompt row until every member has drawn its first token.
        std::vector<float> prompt_logits(logits, logits + n_vocab);
        llama_memory_t memory = llama_get_memory(ctx);
        if (!memory) {
            set_error("generation context has no sequence memory");
            clear_generation_memory(*state, ctx);
            return -1;
        }
        for (size_t sequence = 1; sequence < n_sequences; ++sequence) {
            llama_memory_seq_cp(memory, 0, static_cast<llama_seq_id>(sequence),
                    -1, -1);
        }

        std::vector<std::mt19937> rngs;
        rngs.reserve(n_sequences);
        std::vector<sampler_scratch> scratches(n_sequences);
        std::vector<const float *> sequence_logits(n_sequences, prompt_logits.data());
        std::vector<size_t> limits(n_sequences);
        std::vector<bool> live(n_sequences, true);
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            rngs.emplace_back(sampling[sequence].seed);
            limits[sequence] = std::min<size_t>(
                    std::min<size_t>(n_out_max, sampling[sequence].max_new_tokens),
                    n_ctx - n_prompt);
            out_n_tokens[sequence] = 0;
        }

        batch_guard member_batch(static_cast<int32_t>(n_sequences));
        std::vector<size_t> decoded_sequences;
        decoded_sequences.reserve(n_sequences);
        size_t n_live = n_sequences;
        while (n_live > 0) {
            decoded_sequences.clear();
            for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
                if (!live[sequence]) {
                    continue;
                }
                sampler_scratch & scratch = scratches[sequence];
                const retro_sampling_params & params = sampling[sequence];
                const int32_t token = sample_token(sequence_logits[sequence], n_vocab,
                        params.temperature, params.top_p, rngs[sequence], scratch);
                const size_t count = out_n_tokens[sequence];
                const size_t output = sequence * n_out_max + count;
                out_tokens[output] = token;
                if (out_logprobs) {
                    out_logprobs[output] = params.temperature == 1.0f
                            ? static_cast<float>(
                                    static_cast<double>(sequence_logits[sequence][token])
                                    - scratch.max - std::log(scratch.total))
                            : token_logprob(sequence_logits[sequence], n_vocab,
                                    token, scratch.probs);
                }
                out_n_tokens[sequence] = count + 1;
                if (llama_vocab_is_eog(vocab, token)
                        || out_n_tokens[sequence] == limits[sequence]) {
                    live[sequence] = false;
                    --n_live;
                    continue;
                }
                decoded_sequences.push_back(sequence);
            }
            if (decoded_sequences.empty()) {
                break;
            }

            llama_batch & batch = member_batch.batch;
            batch.n_tokens = static_cast<int32_t>(decoded_sequences.size());
            for (size_t row = 0; row < decoded_sequences.size(); ++row) {
                const size_t sequence = decoded_sequences[row];
                const size_t count = out_n_tokens[sequence];
                batch.token[row] = out_tokens[sequence * n_out_max + count - 1];
                batch.pos[row] = static_cast<llama_pos>(n_prompt + count - 1);
                batch.n_seq_id[row] = 1;
                batch.seq_id[row][0] = static_cast<llama_seq_id>(sequence);
                batch.logits[row] = true;
            }
            if (decode_with_duty_cycle(*state, ctx, batch) != 0) {
                set_error("llama_decode failed during batched generation");
                clear_generation_memory(*state, ctx);
                return -1;
            }
            for (size_t row = 0; row < decoded_sequences.size(); ++row) {
                const float * row_logits = llama_get_logits_ith(
                        ctx, static_cast<int32_t>(row));
                if (!row_logits) {
                    set_error("batched decode produced no logits for a live sequence");
                    clear_generation_memory(*state, ctx);
                    return -1;
                }
                sequence_logits[decoded_sequences[row]] = row_logits;
            }
        }
        clear_generation_memory(*state, ctx);
        return 0;
    });
}

int generate_continuous_batch_impl(
        retro_trainer * trainer,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) return -1;
        if (!sequences || n_sequences == 0 || !out_tokens || !out_n_tokens || n_out_max == 0) {
            set_error("continuous generation inputs and output buffers are required");
            return -1;
        }
        llama_context * ctx = state->generation_ctx ? state->generation_ctx.get() : state->ctx.get();
        if (n_sequences > llama_n_seq_max(ctx) || n_sequences > llama_n_batch(ctx)) {
            set_error("continuous generation batch exceeds the configured sequence or batch capacity");
            return -1;
        }
        const size_t n_ctx = llama_n_ctx_seq(ctx);
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            const retro_generation_sequence & item = sequences[sequence];
            if (!item.prompt_tokens || item.n_prompt == 0 || item.n_prompt >= n_ctx
                    || !validate_sampling(item.sampling)
                    || !validate_tokens(*state, item.prompt_tokens, item.n_prompt)) {
                if (g_last_error.empty()) set_error("invalid continuous generation prompt");
                return -1;
            }
            out_n_tokens[sequence] = 0;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
        // out_logprobs, not nullptr: this path cannot produce generation
        // logprobs, so a caller that asks for them must stay on the CPU oracle.
        // It was harmless while the device sampler was opt-in and no caller
        // combined the two; with S4 turning it on by default it would have
        // returned uninitialized logprobs.
        if (device_sampling_requested(*state, out_logprobs)) {
            const retro_generation_stats before_device = state->generation_stats;
            const int result = generate_continuous_batch_device(*state, ctx, sequences,
                    n_sequences, out_tokens, n_out_max, out_n_tokens);
            if (result != 0) {
                return result > 0 ? 0 : -1;
            }
            // If the sampler cannot attach, continue through the CPU oracle. The
            // context is consistent either way it declined: before any decode,
            // with the placement's evictions applied, or at a terminal row that
            // came back without a token, after which the device path cleared
            // it. The counters are not - they describe an attempt that produced
            // nothing - so they go back to what they were.
            state->generation_stats = before_device;
            std::fill(out_n_tokens, out_n_tokens + n_sequences, 0);
        }
        const bool cache = generation_prefix_cache_enabled();
        if (!cache) {
            clear_generation_memory(*state, ctx);
        }
        state->generation_stats.calls++;

        // Prefill prompt by prompt in chunks of n_batch positions, instead of
        // one position per sequence and per launch. Only terminal prompt rows
        // request logits, and each is copied out before the next launch.
        const std::vector<size_t> order = canonical_prefill_order(sequences, n_sequences);
        const std::vector<kv_placement> placement = place_generation_sequences(
                *state, ctx, sequences, n_sequences, order, cache);
        batch_guard prefill_batch(static_cast<int32_t>(llama_n_batch(ctx)));
        std::vector<std::vector<float>> sequence_logits(n_sequences);
        if (!prefill_continuous_prompts(
                *state, ctx, sequences, order, placement, prefill_batch.batch,
                "llama_decode failed during continuous prompt prefill",
                [&](size_t sequence, int32_t row) {
                    const float * logits = llama_get_logits_ith(ctx, row);
                    if (!logits) {
                        set_error("continuous prompt prefill produced no logits");
                        return false;
                    }
                    sequence_logits[sequence].assign(logits, logits + n_vocab);
                    return true;
                })) {
            clear_generation_memory(*state, ctx);
            return -1;
        }

        std::vector<std::mt19937> rngs;
        rngs.reserve(n_sequences);
        std::vector<sampler_scratch> scratches(n_sequences);
        std::vector<size_t> limits(n_sequences);
        std::vector<bool> live(n_sequences, true);
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            rngs.emplace_back(sequences[sequence].sampling.seed);
            limits[sequence] = std::min<size_t>(
                    std::min<size_t>(n_out_max, sequences[sequence].sampling.max_new_tokens),
                    n_ctx - sequences[sequence].n_prompt);
        }
        batch_guard decode_batch(static_cast<int32_t>(n_sequences));
        std::vector<size_t> active;
        active.reserve(n_sequences);
        size_t n_live = n_sequences;
        while (n_live > 0) {
            active.clear();
            // Canonical order here too: decode rows must not carry caller slots
            // into the physical batch. See canonical_prefill_order.
            for (const size_t sequence : order) {
                if (!live[sequence]) continue;
                sampler_scratch & scratch = scratches[sequence];
                const retro_sampling_params & params = sequences[sequence].sampling;
                const int32_t token = sample_token(sequence_logits[sequence].data(), n_vocab,
                        params.temperature, params.top_p, rngs[sequence], scratch);
                const size_t count = out_n_tokens[sequence];
                out_tokens[sequence*n_out_max + count] = token;
                if (out_logprobs) {
                    out_logprobs[sequence*n_out_max + count] = params.temperature == 1.0f
                            ? static_cast<float>(static_cast<double>(sequence_logits[sequence][token])
                                    - scratch.max - std::log(scratch.total))
                            : token_logprob(sequence_logits[sequence].data(), n_vocab, token, scratch.probs);
                }
                out_n_tokens[sequence] = count + 1;
                if (llama_vocab_is_eog(vocab, token) || out_n_tokens[sequence] == limits[sequence]) {
                    live[sequence] = false;
                    --n_live;
                } else {
                    active.push_back(sequence);
                }
            }
            if (active.empty()) break;
            llama_batch & batch = decode_batch.batch;
            batch.n_tokens = static_cast<int32_t>(active.size());
            for (size_t row = 0; row < active.size(); ++row) {
                const size_t sequence = active[row];
                const size_t count = out_n_tokens[sequence];
                batch.token[row] = out_tokens[sequence*n_out_max + count - 1];
                batch.pos[row] = static_cast<llama_pos>(sequences[sequence].n_prompt + count - 1);
                batch.n_seq_id[row] = 1;
                batch.seq_id[row][0] = placement[sequence].slot;
                batch.logits[row] = true;
            }
            if (decode_with_duty_cycle(*state, ctx, batch) != 0) {
                set_error("llama_decode failed during continuous generation");
                clear_generation_memory(*state, ctx);
                return -1;
            }
            for (size_t row = 0; row < active.size(); ++row) {
                const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(row));
                if (!logits) {
                    set_error("continuous generation produced no logits");
                    clear_generation_memory(*state, ctx);
                    return -1;
                }
                sequence_logits[active[row]].assign(logits, logits + n_vocab);
            }
        }
        if (cache) {
            remember_generation_kv(*state, sequences, n_sequences, placement,
                    out_tokens, n_out_max, out_n_tokens);
        } else {
            clear_generation_memory(*state, ctx);
        }
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
            return -1;
        }

        clear_context_memory(*state);
        if (!score_token_range(*state, tokens, n_tokens, 1, out_logprobs)) {
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

        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
        double negative_log_likelihood = 0.0;
        uint64_t supervised_tokens = 0;
        std::vector<double> logprob_work;

        for (size_t row = 0; row < data->n_rows; ++row) {
            const int32_t * tokens = data->tokens + row * data->n_ctx;
            const int32_t * labels = data->labels + row * data->n_ctx;
            if (!validate_tokens(*state, tokens, data->n_ctx)) {
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
                    *state, tokens, data->n_ctx, 0,
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
            return -1;
        }

        clear_context_memory(*state);
        if (!score_token_range(*state, tokens, n_tokens, n_prompt, out_logprobs)) {
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
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
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
                *state, tokens, n_tokens - 1, first_state,
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
        if (!validate_tokens(*state, prompt, sequences[0].n_tokens)) {
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
            if (!validate_tokens(*state, item.tokens, item.n_tokens)) {
                return -1;
            }
            out_n_logprobs[sequence] = item.n_tokens - n_prompt;
        }

        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
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
                if (decode_with_duty_cycle(*state, ctx, batch) != 0) {
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
        const bool device_gather_available = device_logprob_gather_enabled(*state);
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
                    if (decode_with_duty_cycle(*state, ctx, batch) != 0) {
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
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
                *state, tokens, n_tokens, 0,
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
            return -1;
        }

        const size_t n_embd =
                static_cast<size_t>(llama_model_n_embd_out(state->model.get()));
        const size_t n_completion = n_tokens - n_prompt;
        if (n_embd == 0 || n_completion > n_features_max / n_embd) {
            set_error("combined hidden-state output buffer is too small");
            return -1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
        const size_t first_state = n_prompt - 1;
        const size_t n_states = n_tokens - 1;
        std::vector<double> logprob_work;

        llama_context * ctx = state->ctx.get();
        llama_set_embeddings(ctx, true);
        clear_context_memory(*state);

        const bool ok = decode_states_emitting_outputs(
                *state, tokens, n_states, first_state,
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
        if (!validate_tokens(*state, tokens, n_tokens)) {
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
