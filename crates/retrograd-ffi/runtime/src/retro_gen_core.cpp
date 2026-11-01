// The shared generation core: batch construction, duty-cycle-aware decodes,
// the CPU sampling oracle and the device sampler chain, logprobs/top-k,
// sequence placement and prefix reuse. Extracted from retro_rollout.cpp so the
// trainer and the engine run the exact same code; retro_rollout.cpp keeps only
// the FFI-shaped wrappers that validate their handle and pick a context.
#include "retro_gen_core.hpp"

#include "reasoning-budget.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <utility>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace retro {

namespace {

using gen_detail::batch_guard;
using gen_detail::sampler_scratch;

// ---------------------------------------------------------------------------
// Log-softmax reductions. The two terms of a row's denominator are kept apart
// rather than summed: every reduction below spells the value as
// `logit - max - log_sum`, and folding the pair into one normalizer would move
// the rounding of a log-probability that other paths compare against bit for
// bit.

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

// ---------------------------------------------------------------------------
// The CPU sampling oracle.

bool fast_top_p_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("RETRO_FAST_TOP_P");
        return value && value[0] != '\0' && std::strcmp(value, "0") != 0
                && std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

// Forward: the draw half of the oracle, shared verbatim by sample_token (v1)
// and sample_token_v2 (v2).
int32_t draw_from_probs(
        std::vector<double> & probs,
        size_t n_vocab,
        float top_p,
        std::mt19937 & rng,
        sampler_scratch & scratch,
        double & out_denom,
        bool truncate = false);

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
    if (!std::isfinite(max)) {
        set_error("sampling transformations left no finite logits");
        return -1;
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
    double denom = 0.0;
    return draw_from_probs(probs, n_vocab, top_p, rng, scratch, denom);
}

// The draw half of the oracle, shared verbatim by sample_token (v1) and
// sample_token_v2 (v2, over the transformed probs): the top_p truncation and
// the categorical draw are the same code, so a v2 row with neutral extras
// behaves exactly like v1's temperature + top_p. `out_denom` receives the mass
// the draw was normalized over (the full total on the top_p == 1 fast path,
// the kept mass otherwise), which is the renormalizer of the post-transformation
// distribution the v2 outputs report.
int32_t draw_from_probs(
        std::vector<double> & probs,
        size_t n_vocab,
        float top_p,
        std::mt19937 & rng,
        sampler_scratch & scratch,
        double & out_denom,
        bool truncate) {
    const double total = scratch.total;

    // Strictly on-policy Dr. GRPO uses top_p == 1. Avoid sorting the entire
    // vocabulary in that common path; a cumulative categorical draw is exact.
    if (top_p == 1.0f) {
        std::uniform_real_distribution<double> uniform(0.0, total);
        const double draw = uniform(rng);
        double acc = 0.0;
        for (size_t i = 0; i < n_vocab; ++i) {
            acc += probs[i];
            if (probs[i] > 0.0 && draw <= acc) {
                out_denom = total;
                return static_cast<int32_t>(i);
            }
        }
        out_denom = total;
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

    if (truncate) {
        // Preserve exactly the nucleus used by the draw for serving logprobs.
        const size_t begin = fast_top_p_enabled() ? 0 : n_keep;
        const size_t end = fast_top_p_enabled() ? n_vocab - n_keep : n_vocab;
        for (size_t i = begin; i < end; ++i) {
            probs[order[i]] = 0.0;
        }
    }

    std::uniform_real_distribution<double> uniform(0.0, kept);
    const double draw = uniform(rng);
    double acc = 0.0;
    for (size_t i = 0; i < n_keep; ++i) {
        acc += probs[(*candidates)[i]];
        if (draw <= acc) {
            out_denom = kept;
            return (*candidates)[i];
        }
    }
    out_denom = kept;
    return (*candidates)[n_keep - 1];
}

// ---------------------------------------------------------------------------
// Device-side sampling: one backend sampler chain per row of a call.

// Whether the active device can gather log softmax(logits)[target] inside the
// decode graph. Read per call rather than cached: RETRO_DEVICE_LOGPROBS=0 is
// both the escape hatch and how the two reductions are compared on a given
// model, including from a test that flips it.
bool device_logprob_gather_enabled(const gen_core_env & env) {
    const char * value = std::getenv("RETRO_DEVICE_LOGPROBS");
    const bool disabled = value
            && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0);
    return !disabled && env.gpu_active && env.cap_device_logprobs;
}

// On by default wherever the capability is present: sampling on the device
// keeps the n_vocab logits row there instead of copying it to the host for
// every sampled token. RETRO_DEVICE_SAMPLING=0 (or "false") is the escape
// hatch. Callers never have to handle a refusal - when llama.cpp cannot attach
// the chain, backend_sampler_scope reports it unavailable and, if a single
// sampler op stayed on the host mid-generation, sample_backend_token returns 0
// and the caller replays the whole request through the CPU oracle.
bool device_sampling_requested(const gen_core_env & env, const float * out_logprobs) {
    const char * value = std::getenv("RETRO_DEVICE_SAMPLING");
    const bool disabled = value
            && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0);
    // Generation logprobs describe the unmodified temperature-1 policy. The
    // public API that requests them (retro_trainer_generate*) therefore remains
    // on the exact CPU oracle; GRPO rollout collection passes nullptr because it
    // scores behavior separately, so it takes this path.
    return !disabled && out_logprobs == nullptr && env.gpu_active
            && env.cap_device_sampling;
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

// ---------------------------------------------------------------------------
// Continuous-batching placement and prefix reuse.

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
        gen_core & core,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        const std::vector<size_t> & order,
        size_t n_out_max,
        bool cache_enabled) {
    llama_context * ctx = core.ctx;
    gen_cache & bookkeeping = *core.cache;
    const size_t n_slots = static_cast<size_t>(llama_n_seq_max(ctx));
    std::vector<kv_placement> placement(n_sequences);
    if (!cache_enabled) {
        for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
            placement[sequence].slot = static_cast<llama_seq_id>(sequence);
            placement[sequence].reuse = 0;
        }
        return placement;
    }
    bookkeeping.slots.resize(n_slots);
    bookkeeping.used_at.resize(n_slots, 0);

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
            const size_t reuse = resident_prefix(bookkeeping.slots[slot], item);
            if (best_slot == n_slots || reuse > best_reuse) {
                best_slot = slot;
                best_reuse = reuse;
                continue;
            }
            if (reuse == best_reuse && reuse == 0) {
                // No candidate carries anything worth keeping, so the choice is
                // which slot to sacrifice: the one that has gone longest
                // without being extended.
                if (bookkeeping.used_at[slot] < bookkeeping.used_at[best_slot]) {
                    best_slot = slot;
                }
            }
        }
        claimed[best_slot] = true;
        placement[sequence].slot = static_cast<llama_seq_id>(best_slot);
        placement[sequence].reuse = best_reuse;
        if (best_reuse == 0) {
            if (!bookkeeping.slots[best_slot].empty()) {
                bookkeeping.stats.evictions++;
            }
            // Dropping a whole sequence is unconditionally supported, unlike the
            // mid-sequence removal a partial rewind would need.
            if (memory) {
                llama_memory_seq_rm(memory, static_cast<llama_seq_id>(best_slot), -1, -1);
            }
            bookkeeping.slots[best_slot].clear();
        } else {
            bookkeeping.stats.hits++;
        }
        bookkeeping.stats.sequences++;
        bookkeeping.stats.prompt_tokens += item.n_prompt;
        bookkeeping.stats.reused_tokens += best_reuse;
        // `prefilled_tokens` is counted where positions are actually pushed: a
        // member of a copy group receives its prefix by copy and decodes one
        // position, not `n_prompt - reuse` of them.
    }
    // Unrequested trajectories stay cached between waves, but share the same
    // finite KV pool as the current requests. Reserving a previously empty
    // slot does not reclaim their cells: without pressure eviction, even one
    // near-context-sized prompt can fail on its second call. Budget the final
    // decoded positions (the last emitted token is never decoded), then evict
    // only dormant slots, oldest first. Counting copied prefixes per sequence
    // is conservative: it can cool an idle slot, never discard an active one.
    size_t remaining = llama_n_ctx(ctx);
    const size_t n_ctx_seq = llama_n_ctx_seq(ctx);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        const auto & item = sequences[sequence];
        const size_t emitted = std::min<size_t>(
                std::min<size_t>(n_out_max, item.sampling.max_new_tokens),
                n_ctx_seq - item.n_prompt);
        const size_t decoded = item.n_prompt + emitted - 1;
        remaining -= std::min(remaining, decoded);
    }
    // At most LLAMA_MAX_SEQ resident vectors, each bounded by the u32 context
    // width; their conservative sum fits in u64 even on a 32-bit host.
    uint64_t dormant = 0;
    for (size_t slot = 0; slot < n_slots; ++slot) {
        if (!claimed[slot]) {
            dormant += bookkeeping.slots[slot].size();
        }
    }
    while (dormant > remaining) {
        size_t oldest = n_slots;
        for (size_t slot = 0; slot < n_slots; ++slot) {
            if (!claimed[slot] && !bookkeeping.slots[slot].empty()
                    && (oldest == n_slots || bookkeeping.used_at[slot] < bookkeeping.used_at[oldest])) {
                oldest = slot;
            }
        }
        if (oldest == n_slots) {
            break;
        }
        if (memory) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(oldest), -1, -1);
        }
        dormant -= bookkeeping.slots[oldest].size();
        bookkeeping.slots[oldest].clear();
        bookkeeping.used_at[oldest] = 0;
        bookkeeping.stats.evictions++;
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
        gen_core & core,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        const std::vector<kv_placement> & placement,
        const int32_t * out_tokens,
        size_t n_out_max,
        const size_t * out_n_tokens) {
    gen_cache & bookkeeping = *core.cache;
    if (bookkeeping.slots.empty()) {
        return;
    }
    ++bookkeeping.clock;
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        const retro_generation_sequence & item = sequences[sequence];
        const size_t slot = static_cast<size_t>(placement[sequence].slot);
        std::vector<int32_t> & resident = bookkeeping.slots[slot];
        const size_t decoded = out_n_tokens[sequence] > 0 ? out_n_tokens[sequence] - 1 : 0;
        resident.assign(item.prompt_tokens, item.prompt_tokens + item.n_prompt);
        resident.insert(resident.end(), out_tokens + sequence*n_out_max,
                out_tokens + sequence*n_out_max + decoded);
        bookkeeping.used_at[slot] = bookkeeping.clock;
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
// has to fire there. A GRPO wave of `prompts x group_size` sequences therefore
// costs `prompts` prefixes, not `prompts x group_size`.
// Logits are requested on terminal prompt rows only - the output projection over
// every prompt position was pure waste. `on_terminal(sequence, batch_row)` runs
// right after the decode that produced that row, since the next launch
// invalidates llama's output buffer.
template <typename OnTerminal>
bool prefill_continuous_prompts(
        gen_core & core,
        const retro_generation_sequence * sequences,
        const std::vector<size_t> & order,
        const std::vector<kv_placement> & placement,
        llama_batch & batch,
        const char * decode_error,
        OnTerminal on_terminal) {
    llama_context * ctx = core.ctx;
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
        if (core.decode(batch) != 0) {
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
        ++core.cache->stats.prefilled_tokens;
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

} // namespace

// ---------------------------------------------------------------------------
// The v2 CPU sampling oracle: extended sampling on the step path. The
// transformation order is pinned (retro_lora_train.h):
//   logit_bias -> penalties(window) -> grammar mask -> top_k -> min_p
//   -> temperature -> top_p -> draw.
// Each stage mirrors the vendored llama-sampler.cpp apply it replaces.

int32_t sample_token_v2(
        const float * logits,
        size_t n_vocab,
        const retro_sampling_params_v2 & params,
        std::mt19937 & rng,
        gen_detail::sampler_scratch & scratch,
        const std::vector<int32_t> & history,
        llama_sampler * grammar_sampler,
        float * out_sampled_logprob,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t k) {
    // (1) Copy the logits and apply the logit bias.
    std::vector<float> & t = scratch.logits_v2;
    t.resize(n_vocab);
    std::memcpy(t.data(), logits, n_vocab * sizeof(float));
    for (uint32_t i = 0; i < params.n_logit_bias; ++i) {
        t[params.logit_bias_ids[i]] += params.logit_bias_values[i];
    }

    // (2) Penalties over the last penalty_window entries of the history
    // (0 = the whole history). Mirrors llama_sampler_penalties_apply exactly:
    // repeat multiplies non-positive logits and divides positive ones, then
    // frequency and presence subtract count-scaled terms.
    const bool any_penalty = params.repeat_penalty != 1.0f
            || params.frequency_penalty != 0.0f
            || params.presence_penalty != 0.0f;
    if (any_penalty && !history.empty()) {
        const size_t window = params.penalty_window == 0
                ? history.size()
                : std::min<size_t>(params.penalty_window, history.size());
        std::unordered_map<int32_t, int32_t> & counts = scratch.penalty_counts;
        counts.clear();
        for (size_t i = history.size() - window; i < history.size(); ++i) {
            counts[history[i]]++;
        }
        for (const auto & entry : counts) {
            const int32_t id = entry.first;
            const int32_t count = entry.second;
            if (t[id] <= 0.0f) {
                t[id] *= params.repeat_penalty;
            } else {
                t[id] /= params.repeat_penalty;
            }
            t[id] -= static_cast<float>(count) * params.frequency_penalty
                    + static_cast<float>(count > 0) * params.presence_penalty;
        }
    }

    // (3) Grammar mask. Candidates are built exactly as llama.cpp builds them
    // from a logits row ({id, logit, 0}); the grammar apply sets the logit of
    // every disallowed token to -INFINITY in place, so read it back.
    if (grammar_sampler) {
        std::vector<llama_token_data> & candidates = scratch.candidates;
        candidates.resize(n_vocab);
        for (size_t i = 0; i < n_vocab; ++i) {
            candidates[i] = llama_token_data{
                    static_cast<llama_token>(i), t[i], 0.0f};
        }
        llama_token_data_array arr = {
            /* .data     = */ candidates.data(),
            /* .size     = */ candidates.size(),
            /* .selected = */ -1,
            /* .sorted   = */ false,
        };
        llama_sampler_apply(grammar_sampler, &arr);
        bool any_allowed = false;
        for (size_t i = 0; i < n_vocab; ++i) {
            t[i] = arr.data[i].logit;
            if (t[i] != -INFINITY) {
                any_allowed = true;
            }
        }
        if (!any_allowed) {
            set_error("grammar allows no token at this position");
            return -1;
        }
    }

    // (4) top_k: keep the top_k tokens by transformed logit, mirroring
    // llama_sampler_top_k_impl (partial sort by logit descending, truncate).
    if (params.top_k > 0 && static_cast<size_t>(params.top_k) < n_vocab) {
        std::vector<int32_t> & order = scratch.order;
        order.resize(n_vocab);
        std::iota(order.begin(), order.end(), 0);
        const size_t keep = static_cast<size_t>(params.top_k);
        const auto by_logit = [&](int32_t a, int32_t b) {
            return t[a] > t[b];
        };
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), by_logit);
        for (size_t i = keep; i < n_vocab; ++i) {
            t[order[i]] = -INFINITY;
        }
    }

    // (5) min_p: mirror llama_sampler_min_p_apply with min_keep = 1 (the first
    // token always matches, so the filtered set is never empty here): the
    // threshold is max_logit + log(min_p), tokens below it are dropped.
    if (params.min_p > 0.0f) {
        float max_logit = -INFINITY;
        for (size_t i = 0; i < n_vocab; ++i) {
            max_logit = std::max(max_logit, t[i]);
        }
        const float min_logit = max_logit + std::log(params.min_p);
        for (size_t i = 0; i < n_vocab; ++i) {
            if (t[i] < min_logit) {
                t[i] = -INFINITY;
            }
        }
    }

    // (6) Temperature, in the oracle's existing style: probs[i] =
    // exp((logit - max) / temp) in double, masked tokens contribute 0.
    float max = -INFINITY;
    for (size_t i = 0; i < n_vocab; ++i) {
        max = std::max(max, t[i]);
    }
    scratch.max = max;
    std::vector<double> & probs = scratch.probs;
    probs.resize(n_vocab);
    double total = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((static_cast<double>(t[i]) - max) / params.temperature);
        total += probs[i];
    }
    scratch.total = total;

    // (7) top_p truncation + draw: the exact v1 code path over the transformed
    // probs (including the top_p == 1.0 fast path).
    double denom = 0.0;
    const int32_t token = draw_from_probs(probs, n_vocab, params.top_p, rng, scratch, denom, true);
    scratch.total = denom;

    // (8) Report under the post-transformation distribution (renormalized over
    // the mass the draw used). The caller advances history and grammar with
    // the emitted token after the reasoning budget has chosen it.
    if (k > 0 && out_top_ids && out_top_logprobs) {
        std::vector<int32_t> & order = scratch.order;
        order.resize(n_vocab);
        std::iota(order.begin(), order.end(), 0);
        const auto by_logit = [&](int32_t a, int32_t b) {
            if (probs[a] != probs[b]) {
                return probs[a] > probs[b];
            }
            return a < b;
        };
        std::partial_sort(order.begin(), order.begin() + k, order.end(), by_logit);
        for (size_t j = 0; j < k; ++j) {
            const int32_t id = order[j];
            out_top_ids[j] = id;
            out_top_logprobs[j] = static_cast<float>(std::log(probs[id] / denom));
        }
    }
    if (out_sampled_logprob) {
        *out_sampled_logprob = static_cast<float>(std::log(probs[token] / denom));
    }
    return token;
}

// ---------------------------------------------------------------------------
// The reasoning budget: one llama.cpp state machine per row of a batched call,
// or per request of a step-wise one.

// Retrograd samples on its own - the CPU oracle or a device chain - so the
// machine is used for its state and not as a sampler: once it is forcing, its
// token replaces the sampled one, which is the only token its masked
// distribution could have drawn. Rows and requests report which of their
// tokens were forced, since those are the runtime's choice and not the
// policy's. budget_machine is declared in retro_runtime.hpp (step_request_state
// holds one), so its definitions sit at namespace scope, outside the anonymous
// block below.

budget_machine::budget_machine(const llama_vocab * vocab, const reasoning_budget_plan * plan) {
    if (!plan || !plan->active) {
        return;
    }
    plan_ = plan;
    sampler_ = common_reasoning_budget_init(
            vocab, { plan->start }, plan->ends, plan->forced, plan->budget);
}

budget_machine::~budget_machine() {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }
}

budget_machine::budget_machine(budget_machine && other) noexcept
    : plan_(other.plan_), sampler_(other.sampler_), force_pos_(other.force_pos_) {
    other.plan_ = nullptr;
    other.sampler_ = nullptr;
    other.force_pos_ = 0;
}

budget_machine & budget_machine::operator=(budget_machine && other) noexcept {
    if (this != &other) {
        if (sampler_) {
            llama_sampler_free(sampler_);
        }
        plan_ = other.plan_;
        sampler_ = other.sampler_;
        force_pos_ = other.force_pos_;
        other.plan_ = nullptr;
        other.sampler_ = nullptr;
        other.force_pos_ = 0;
    }
    return *this;
}

// One past the earliest end tag in [from, to), or null if none closes there.
const int32_t * budget_machine::closing_tag_end(const int32_t * from, const int32_t * to) const {
    const int32_t * closed = nullptr;
    const int32_t * earliest = to;
    for (const auto & tag : plan_->ends) {
        const int32_t * found = std::search(from, to, tag.begin(), tag.end());
        if (found != to && found < earliest) {
            earliest = found;
            closed = found + tag.size();
        }
    }
    return closed;
}

void budget_machine::accept(int32_t token) {
    const auto before = common_reasoning_budget_get_state(sampler_);
    llama_sampler_accept(sampler_, token);
    if (common_reasoning_budget_get_state(sampler_) == REASONING_BUDGET_FORCING
            && before != REASONING_BUDGET_FORCING) {
        force_pos_ = 0;
    }
}

// Replays the end of the prompt, as long as the template's generation
// prompt, so a prompt that already opened the reasoning block starts the
// machine counting and one that closed it starts it done.
//
// A machine that starts forcing on a prompt token - a budget of zero on an
// opening tag, or one spent inside the prompt - stops the replay there: the
// machine reads every token it accepts while forcing as the next forced
// one, so the prompt's newline after the opening tag would stand for the
// end tag and the tag would never be emitted. The rest of the prompt
// belongs to the prompt, unless it closes the block itself (an empty block
// written by the template): then the forced sequence is accepted in its
// place, which leaves the machine done, and the replay resumes after the tag.
void budget_machine::prime(const int32_t * prompt, size_t n_prompt) {
    if (!plan_ || n_prompt == 0) {
        return;
    }
    const int32_t * end = prompt + n_prompt;
    const int32_t * at = prompt + (n_prompt - std::min(plan_->n_prompt_tail, n_prompt));
    while (at != end) {
        accept(*at++);
        if (common_reasoning_budget_get_state(sampler_) != REASONING_BUDGET_FORCING) {
            continue;
        }
        const int32_t * closed = closing_tag_end(at, end);
        if (!closed) {
            return;
        }
        for (const int32_t token : plan_->forced) {
            accept(token);
        }
        at = closed;
    }
}

// The token to emit in place of `sampled`, and whether the budget forced it.
int32_t budget_machine::next(int32_t sampled, bool & forced) {
    forced = false;
    if (!plan_) {
        return sampled;
    }
    int32_t token = sampled;
    if (common_reasoning_budget_get_state(sampler_) == REASONING_BUDGET_FORCING) {
        // The machine forces exactly `forced`, one token per accept, so a
        // forcing machine whose local position ran past it is out of step.
        GGML_ASSERT(force_pos_ < plan_->forced.size());
        token = plan_->forced[force_pos_++];
        forced = true;
    }
    accept(token);
    // The forced sequence is emitted whole: the machine is done exactly
    // when the last forced token went out, and still forcing before.
    if (forced) {
        const bool done = force_pos_ == plan_->forced.size();
        const auto after = common_reasoning_budget_get_state(sampler_);
        GGML_ASSERT(done ? after == REASONING_BUDGET_DONE : after == REASONING_BUDGET_FORCING);
    }
    return token;
}

namespace {

// Whether two derived budget plans describe the same machine configuration.
bool same_reasoning_plan(const reasoning_budget_plan & left, const reasoning_budget_plan & right) {
    return left.active == right.active
            && left.budget == right.budget
            && left.start == right.start
            && left.ends == right.ends
            && left.forced == right.forced
            && left.n_prompt_tail == right.n_prompt_tail;
}

// One budget machine per row of a batched call.
class reasoning_budget_rows {
public:
    reasoning_budget_rows(const llama_vocab * vocab, const reasoning_budget_plan * plan, size_t n_rows) {
        if (!plan || !plan->active) {
            return;
        }
        rows_.reserve(n_rows);
        for (size_t row = 0; row < n_rows; ++row) {
            rows_.emplace_back(vocab, plan);
        }
    }

    reasoning_budget_rows(const reasoning_budget_rows &) = delete;
    reasoning_budget_rows & operator=(const reasoning_budget_rows &) = delete;

    void prime(size_t row, const int32_t * prompt, size_t n_prompt) {
        if (rows_.empty()) {
            return;
        }
        rows_[row].prime(prompt, n_prompt);
    }

    int32_t next(size_t row, int32_t sampled, bool & forced) {
        return rows_.empty() ? sampled : rows_[row].next(sampled, forced);
    }

private:
    std::vector<budget_machine> rows_;
};

// Writes one emitted token's forced flag when the caller asked for them.
void record_forced(uint8_t * out_forced, size_t at, bool forced) {
    if (out_forced) {
        out_forced[at] = forced ? 1 : 0;
    }
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
        gen_core & core,
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced,
        const reasoning_budget_plan * plan) {
    llama_context * ctx = core.ctx;
    backend_sampler_scope samplers(ctx, sampling, identity_slots(n_sequences));
    if (!samplers.available()) {
        return 0;
    }

    const llama_vocab * vocab = core.vocab();
    reasoning_budget_rows budget(vocab, plan, n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        budget.prime(sequence, prompt_tokens, n_prompt);
    }
    const size_t n_vocab = core.n_vocab();
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    core.clear_generation_memory();

    batch_guard prefill(static_cast<int32_t>(llama_n_batch(ctx)));
    if (n_prompt > 1 && !core.decode_span_no_logits(
            prompt_tokens, n_prompt - 1, 0, prefill.batch)) {
        core.clear_generation_memory();
        return -1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("generation context has no sequence memory");
        core.clear_generation_memory();
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
    if (core.decode(first) != 0) {
        set_error("llama_decode failed during device-sampling prompt terminal");
        core.clear_generation_memory();
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
            int32_t sampled = LLAMA_TOKEN_NULL;
            const int sample_result = sample_backend_token(
                    ctx, sample_indices[sequence], n_vocab, sampled);
            if (sample_result <= 0) {
                core.clear_generation_memory();
                return sample_result;
            }
            bool forced = false;
            const int32_t token = budget.next(sequence, sampled, forced);
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            record_forced(out_forced, sequence*n_out_max + count, forced);
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
        if (core.decode(batch) != 0) {
            set_error("llama_decode failed during device-sampled generation");
            core.clear_generation_memory();
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            sample_indices[active[row]] = static_cast<int32_t>(row);
        }
    }
    core.clear_generation_memory();
    return 1;
}

int generate_continuous_batch_device(
        gen_core & core,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced,
        const reasoning_budget_plan * plan) {
    llama_context * ctx = core.ctx;
    const llama_vocab * vocab = core.vocab();
    const size_t n_vocab = core.n_vocab();
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        out_n_tokens[sequence] = 0;
    }
    reasoning_budget_rows budget(vocab, plan, n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        budget.prime(sequence, sequences[sequence].prompt_tokens, sequences[sequence].n_prompt);
    }
    const bool cache = generation_prefix_cache_enabled();
    if (!cache) {
        core.clear_generation_memory();
    }
    core.cache->stats.calls++;

    const std::vector<size_t> order = canonical_prefill_order(sequences, n_sequences);
    const std::vector<kv_placement> placement = place_generation_sequences(
            core, sequences, n_sequences, order, n_out_max, cache);
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
            core, sequences, order, placement, prefill_batch.batch,
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
        core.clear_generation_memory();
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
            const int32_t sampled = pending_tokens[sequence];
            if (sampled == LLAMA_TOKEN_NULL) {
                set_error("device sampler has no pending token");
                core.clear_generation_memory();
                return -1;
            }
            bool forced = false;
            const int32_t token = budget.next(sequence, sampled, forced);
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            record_forced(out_forced, sequence*n_out_max + count, forced);
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
        if (core.decode(batch) != 0) {
            set_error("llama_decode failed during continuous device sampling");
            core.clear_generation_memory();
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            const size_t sequence = active[row];
            const int sample_result = sample_backend_token(
                    ctx, static_cast<int32_t>(row), n_vocab, pending_tokens[sequence]);
            if (sample_result <= 0) {
                core.clear_generation_memory();
                return sample_result;
            }
        }
    }
    if (cache) {
        remember_generation_kv(core, sequences, n_sequences, placement,
                out_tokens, n_out_max, out_n_tokens);
    } else {
        core.clear_generation_memory();
    }
    return 1;
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
        gen_core & core,
        const int32_t * tokens,
        size_t n_tokens,
        size_t first_target,
        bool device_gather,
        float * out_logprobs) {
    llama_context * ctx = core.ctx;
    const size_t n_vocab = core.n_vocab();
    const size_t first_state = first_target - 1;
    std::vector<double> logprob_work;
    std::vector<int32_t> gather_targets;
    score_pass result = score_pass::ok;

    const bool decoded = decode_states_emitting_outputs(
            core, tokens, n_tokens - 1, first_state,
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
                if (core.env.scoring_stats) {
                    core.env.scoring_stats->scored_positions++;
                    if (device_gather) {
                        core.env.scoring_stats->device_logprob_positions++;
                    }
                }
                return true;
            });
    if (!decoded) {
        return result == score_pass::needs_host_reduction ? result : score_pass::failed;
    }
    return score_pass::ok;
}

} // namespace

// Public spelling of the two device-path switches, for the wrappers in
// retro_rollout.cpp and retro_engine.cpp: the definitions above stay in the
// anonymous namespace, and these forward to them within this TU.
bool gen_env_device_logprob_gather(const gen_core_env & env) {
    return device_logprob_gather_enabled(env);
}
bool gen_env_device_sampling(const gen_core_env & env, const float * out_logprobs) {
    return device_sampling_requested(env, out_logprobs);
}

// ---------------------------------------------------------------------------
// gen_core

int gen_core::decode(llama_batch & batch) {
    // The owner's default adapter, for this decode's every sequence.
    struct route_scope {
        llama_context * ctx = nullptr;
        ~route_scope() { if (ctx) llama_clear_seq_lora_routes(ctx); }
    } routes;
    if (env.lora_router && env.lora_router->default_adapter != 0) {
        lora_row_router & router = *env.lora_router;
        if (!router.installed && !lora_router_install(router, ctx)) {
            return -1;
        }
        routes.ctx = ctx;
        const int32_t route = lora_router_route(router, router.default_adapter);
        for (uint32_t seq = 0; seq < llama_n_seq_max(ctx); ++seq) {
            llama_set_seq_lora_route(ctx, static_cast<llama_seq_id>(seq), route);
        }
    }
    if (!env.duty_cycle || !env.duty_cycle->enabled()) {
        return llama_decode(ctx, batch);
    }
    env.duty_cycle->begin_window();
    const int status = llama_decode(ctx, batch);
    // Fence whatever was submitted, failure included: the caller is about to
    // tear the operation down and must not do so with work still in flight.
    llama_synchronize(ctx);
    env.duty_cycle->account_window();
    // The debt is recorded either way - the device did the work - but a failing
    // decode is unwinding into an error the caller is waiting on, so the repayment
    // moves to the next boundary rather than delaying it.
    if (status == 0) {
        env.duty_cycle->idle_if_needed();
    }
    return status;
}

int gen_core::decode_ext(llama_batch_ext & batch) {
    // Same owner policy as decode(), over the extended batch the step API
    // builds: one llama_process per step call.
    if (!env.duty_cycle || !env.duty_cycle->enabled()) {
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, &batch);
    }
    env.duty_cycle->begin_window();
    const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, &batch);
    // Fence whatever was submitted, failure included: see decode().
    llama_synchronize(ctx);
    env.duty_cycle->account_window();
    if (status == 0) {
        env.duty_cycle->idle_if_needed();
    }
    return status;
}

void gen_core::clear_ctx_memory() {
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory) {
        llama_memory_clear(memory, true);
    }
}

void gen_core::clear_generation_memory() {
    clear_ctx_memory();
    if (cache) {
        cache->forget_bookkeeping();
    }
}

// Generation decodes into sequence memory and samples from logits; a context
// without memory is an encoder, which evaluates a prompt and generates
// nothing. Refused up front, before anything is decoded.
bool gen_core::require_sequence_memory() const {
    if (llama_get_memory(ctx)) {
        return true;
    }
    set_error("the model has no sequence memory (an encoder, such as a decision model): "
              "it does not generate text");
    return false;
}

bool gen_core::validate_tokens(const int32_t * tokens, size_t n_tokens) const {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab());
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

bool gen_core::validate_sampling(const retro_sampling_params & sampling) const {
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

bool gen_core::validate_sampling_v2(const retro_sampling_params_v2 & sampling) const {
    // Read the evolution key before any other field.
    if (sampling.struct_size < sizeof(retro_sampling_params_v2)) {
        set_error("retro_sampling_params_v2.struct_size is too small for this build");
        return false;
    }
    if (!(sampling.temperature > 0.0f) || !std::isfinite(sampling.temperature)) {
        set_error("sampling temperature must be finite and greater than zero");
        return false;
    }
    if (!(sampling.top_p > 0.0f && sampling.top_p <= 1.0f)) {
        set_error("sampling top_p must be in (0, 1]");
        return false;
    }
    if (!(sampling.repeat_penalty > 0.0f) || !std::isfinite(sampling.repeat_penalty)) {
        set_error("sampling repeat_penalty must be finite and greater than zero");
        return false;
    }
    if (!(sampling.min_p >= 0.0f && sampling.min_p <= 1.0f) || !std::isfinite(sampling.min_p)) {
        set_error("sampling min_p must be in [0, 1]");
        return false;
    }
    if (!std::isfinite(sampling.frequency_penalty)) {
        set_error("sampling frequency_penalty must be finite");
        return false;
    }
    if (!std::isfinite(sampling.presence_penalty)) {
        set_error("sampling presence_penalty must be finite");
        return false;
    }
    if (sampling.n_logit_bias > 4096) {
        set_error("sampling n_logit_bias must not exceed 4096");
        return false;
    }
    if (sampling.n_logit_bias > 0) {
        if (!sampling.logit_bias_ids || !sampling.logit_bias_values) {
            set_error("logit_bias ids and values are required when n_logit_bias is nonzero");
            return false;
        }
        if (!validate_tokens(sampling.logit_bias_ids, sampling.n_logit_bias)) {
            return false;
        }
        for (uint32_t i = 0; i < sampling.n_logit_bias; ++i) {
            if (!std::isfinite(sampling.logit_bias_values[i])) {
                set_error("logit_bias values must be finite");
                return false;
            }
        }
    }
    return true;
}

// Decodes tokens at absolute positions pos0.. in n_batch chunks and returns a
// borrowed pointer to the final requested logits row. The pointer remains valid
// until the next llama_decode on this context, which is exactly the lifetime the
// token sampler needs. The caller owns the batch so incremental generation can
// reuse its allocations across every sampled token.
bool gen_core::decode_span_last_logits(
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
        if (decode(batch) != 0) {
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

bool gen_core::decode_span_no_logits(
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
        if (decode(batch) != 0) {
            set_error("llama_decode failed during device-sampling prompt prefill");
            return false;
        }
        offset += count;
    }
    return true;
}

bool gen_core::score_token_range(
        const int32_t * tokens,
        size_t n_tokens,
        size_t first_target,
        float * out_logprobs) {
    // The device pass counts each position as it reduces it, and the underflow
    // that sends it to the host is only discovered part way through. Rewinding
    // the counters to their pre-pass values makes the retry the only pass this
    // call reports, so `device_logprob_fraction` reads as "the host reduced this
    // range" instead of crediting the abandoned device positions twice.
    const retro_scoring_stats before = env.scoring_stats ? *env.scoring_stats : retro_scoring_stats{};
    score_pass result = score_token_range_pass(
            *this, tokens, n_tokens, first_target,
            device_logprob_gather_enabled(env), out_logprobs);
    if (result == score_pass::needs_host_reduction) {
        if (env.scoring_stats) {
            *env.scoring_stats = before;
        }
        // The retry re-decodes from position zero, so it needs the same empty
        // cache the first pass started from.
        clear_generation_memory();
        result = score_token_range_pass(
                *this, tokens, n_tokens, first_target, /*device_gather =*/ false, out_logprobs);
    }
    return result == score_pass::ok;
}

int gen_core::generate_batch(
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced,
        const reasoning_budget_plan * plan) {
    if (!require_sequence_memory()) {
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
    if (n_sequences > llama_n_seq_max(ctx)) {
        std::ostringstream message;
        message << "generation batch has " << n_sequences
                << " sequences but the context supports "
                << llama_n_seq_max(ctx);
        set_error(message.str());
        return -1;
    }
    // Each live sequence emits a row in the same decode. llama_decode asserts
    // on an oversized physical batch rather than returning an error.
    if (n_sequences > llama_n_batch(ctx)) {
        set_error("generation batch exceeds the configured batch capacity");
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
    if (!validate_tokens(prompt_tokens, n_prompt)) {
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

    const llama_vocab * vocab = this->vocab();
    const size_t n_vocab = this->n_vocab();
    if (device_sampling_requested(env, out_logprobs)) {
        const int result = generate_batch_device(*this, prompt_tokens, n_prompt,
                sampling, n_sequences, out_tokens, n_out_max, out_n_tokens, out_forced, plan);
        if (result != 0) {
            return result > 0 ? 0 : -1;
        }
        // Unsupported backend sampler configuration: use the scalar sampler
        // with the same inputs as a transparent fallback.
    }
    batch_guard decode_batch(static_cast<int32_t>(llama_n_batch(ctx)));
    const float * logits = nullptr;
    clear_generation_memory();
    if (!decode_span_last_logits(prompt_tokens, n_prompt, 0,
            static_cast<int64_t>(n_prompt) - 1, decode_batch.batch, logits)) {
        clear_generation_memory();
        return -1;
    }

    // The next decode invalidates llama's logits buffer, so keep the one
    // shared prompt row until every member has drawn its first token.
    std::vector<float> prompt_logits(logits, logits + n_vocab);
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("generation context has no sequence memory");
        clear_generation_memory();
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

    reasoning_budget_rows budget(vocab, plan, n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        budget.prime(sequence, prompt_tokens, n_prompt);
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
            bool forced = false;
            const int32_t token = budget.next(sequence,
                    sample_token(sequence_logits[sequence], n_vocab,
                            params.temperature, params.top_p, rngs[sequence], scratch),
                    forced);
            const size_t count = out_n_tokens[sequence];
            const size_t output = sequence * n_out_max + count;
            out_tokens[output] = token;
            record_forced(out_forced, output, forced);
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
        if (decode(batch) != 0) {
            set_error("llama_decode failed during batched generation");
            clear_generation_memory();
            return -1;
        }
        for (size_t row = 0; row < decoded_sequences.size(); ++row) {
            const float * row_logits = llama_get_logits_ith(
                    ctx, static_cast<int32_t>(row));
            if (!row_logits) {
                set_error("batched decode produced no logits for a live sequence");
                clear_generation_memory();
                return -1;
            }
            sequence_logits[decoded_sequences[row]] = row_logits;
        }
    }
    clear_generation_memory();
    return 0;
}

int gen_core::generate_continuous_batch(
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced,
        const reasoning_budget_plan * plan) {
    if (!require_sequence_memory()) {
        return -1;
    }
    if (!sequences || n_sequences == 0 || !out_tokens || !out_n_tokens || n_out_max == 0) {
        set_error("continuous generation inputs and output buffers are required");
        return -1;
    }
    if (n_sequences > llama_n_seq_max(ctx) || n_sequences > llama_n_batch(ctx)) {
        set_error("continuous generation batch exceeds the configured sequence or batch capacity");
        return -1;
    }
    if (n_sequences > SIZE_MAX / n_out_max) {
        set_error("continuous generation output shape overflows size_t");
        return -1;
    }
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        const retro_generation_sequence & item = sequences[sequence];
        if (!item.prompt_tokens || item.n_prompt == 0 || item.n_prompt >= n_ctx
                || !validate_sampling(item.sampling)
                || !validate_tokens(item.prompt_tokens, item.n_prompt)) {
            if (g_last_error.empty()) set_error("invalid continuous generation prompt");
            return -1;
        }
        out_n_tokens[sequence] = 0;
    }
    const llama_vocab * vocab = this->vocab();
    const size_t n_vocab = this->n_vocab();
    // out_logprobs, not nullptr: this path cannot produce generation
    // logprobs, so a caller that asks for them must stay on the CPU oracle.
    // It was harmless while the device sampler was opt-in and no caller
    // combined the two; with the device sampler on by default it would have
    // returned uninitialized logprobs.
    if (device_sampling_requested(env, out_logprobs)) {
        const retro_generation_stats before_device = cache->stats;
        const int result = generate_continuous_batch_device(*this, sequences,
                n_sequences, out_tokens, n_out_max, out_n_tokens, out_forced, plan);
        if (result != 0) {
            return result > 0 ? 0 : -1;
        }
        // If the sampler cannot attach, continue through the CPU oracle. The
        // context is consistent either way it declined: before any decode,
        // with the placement's evictions applied, or at a terminal row that
        // came back without a token, after which the device path cleared
        // it. The counters are not - they describe an attempt that produced
        // nothing - so they go back to what they were.
        cache->stats = before_device;
        std::fill(out_n_tokens, out_n_tokens + n_sequences, 0);
    }
    const bool cache = generation_prefix_cache_enabled();
    if (!cache) {
        clear_generation_memory();
    }
    this->cache->stats.calls++;

    // Prefill prompt by prompt in chunks of n_batch positions, instead of
    // one position per sequence and per launch. Only terminal prompt rows
    // request logits, and each is copied out before the next launch.
    const std::vector<size_t> order = canonical_prefill_order(sequences, n_sequences);
    const std::vector<kv_placement> placement = place_generation_sequences(
            *this, sequences, n_sequences, order, n_out_max, cache);
    batch_guard prefill_batch(static_cast<int32_t>(llama_n_batch(ctx)));
    std::vector<std::vector<float>> sequence_logits(n_sequences);
    if (!prefill_continuous_prompts(
            *this, sequences, order, placement, prefill_batch.batch,
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
        clear_generation_memory();
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
    reasoning_budget_rows budget(vocab, plan, n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        budget.prime(sequence, sequences[sequence].prompt_tokens, sequences[sequence].n_prompt);
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
            bool forced = false;
            const int32_t token = budget.next(sequence,
                    sample_token(sequence_logits[sequence].data(), n_vocab,
                            params.temperature, params.top_p, rngs[sequence], scratch),
                    forced);
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            record_forced(out_forced, sequence*n_out_max + count, forced);
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
        if (decode(batch) != 0) {
            set_error("llama_decode failed during continuous generation");
            clear_generation_memory();
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(row));
            if (!logits) {
                set_error("continuous generation produced no logits");
                clear_generation_memory();
                return -1;
            }
            sequence_logits[active[row]].assign(logits, logits + n_vocab);
        }
    }
    if (cache) {
        remember_generation_kv(*this, sequences, n_sequences, placement,
                out_tokens, n_out_max, out_n_tokens);
    } else {
        clear_generation_memory();
    }
    return 0;
}

int gen_core::generate_continuous_batch_v2(
        const retro_generation_sequence_v2 * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced,
        const reasoning_budget_plan * plan) {
    if (!require_sequence_memory()) {
        return -1;
    }
    if (!sequences || n_sequences == 0 || !out_tokens || !out_n_tokens || n_out_max == 0) {
        set_error("continuous v2 generation inputs and output buffers are required");
        return -1;
    }
    if (n_sequences > llama_n_seq_max(ctx) || n_sequences > llama_n_batch(ctx)) {
        set_error("continuous v2 generation batch exceeds the configured sequence or batch capacity");
        return -1;
    }
    if (n_sequences > SIZE_MAX / n_out_max) {
        set_error("continuous v2 generation output shape overflows size_t");
        return -1;
    }
    const size_t n_ctx = llama_n_ctx_seq(ctx);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        const retro_generation_sequence_v2 & item = sequences[sequence];
        if (!item.prompt_tokens || item.n_prompt == 0 || item.n_prompt >= n_ctx
                || !validate_sampling_v2(item.sampling)
                || item.sampling.max_new_tokens == 0
                || !validate_tokens(item.prompt_tokens, item.n_prompt)) {
            if (g_last_error.empty()) set_error("invalid continuous v2 generation prompt");
            return -1;
        }
        out_n_tokens[sequence] = 0;
    }
    const llama_vocab * vocab = this->vocab();
    const size_t n_vocab = this->n_vocab();
    // The placement, prefill and KV-bookkeeping helpers read only the prompt
    // of each row, so they run over a v1 view of the v2 rows; sampling itself
    // is v2-only and happens below, through the CPU oracle.
    std::vector<retro_generation_sequence> view(n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        view[sequence].prompt_tokens = sequences[sequence].prompt_tokens;
        view[sequence].n_prompt = sequences[sequence].n_prompt;
        view[sequence].sampling = {};
    }
    const bool cache = generation_prefix_cache_enabled();
    if (!cache) {
        clear_generation_memory();
    }
    this->cache->stats.calls++;

    const std::vector<size_t> order = canonical_prefill_order(view.data(), n_sequences);
    const std::vector<kv_placement> placement = place_generation_sequences(
            *this, view.data(), n_sequences, order, n_out_max, cache);
    batch_guard prefill_batch(static_cast<int32_t>(llama_n_batch(ctx)));
    std::vector<std::vector<float>> sequence_logits(n_sequences);
    if (!prefill_continuous_prompts(
            *this, view.data(), order, placement, prefill_batch.batch,
            "llama_decode failed during continuous v2 prompt prefill",
            [&](size_t sequence, int32_t row) {
                const float * logits = llama_get_logits_ith(ctx, row);
                if (!logits) {
                    set_error("continuous v2 prompt prefill produced no logits");
                    return false;
                }
                sequence_logits[sequence].assign(logits, logits + n_vocab);
                return true;
            })) {
        clear_generation_memory();
        return -1;
    }

    // Per-sequence oracle state: RNG, scratch, accepted-token history (the
    // penalties' window source) and the compiled grammar, rebuilt when the
    // row's GBNF text changes. The handle itself is copied here, so the caller
    // may free it as soon as this call returns.
    using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    std::vector<std::mt19937> rngs;
    rngs.reserve(n_sequences);
    std::vector<sampler_scratch> scratches(n_sequences);
    std::vector<sampler_scratch> raw_scratches(n_sequences);
    std::vector<std::vector<int32_t>> histories(n_sequences);
    std::vector<sampler_ptr> grammars;
    grammars.reserve(n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        grammars.emplace_back(nullptr, llama_sampler_free);
    }
    std::vector<size_t> limits(n_sequences);
    std::vector<bool> live(n_sequences, true);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        rngs.emplace_back(sequences[sequence].sampling.seed);
        limits[sequence] = std::min<size_t>(
                std::min<size_t>(n_out_max, sequences[sequence].sampling.max_new_tokens),
                n_ctx - sequences[sequence].n_prompt);
        if (sequences[sequence].sampling.grammar) {
            const char * gbnf = retro_grammar_gbnf(sequences[sequence].sampling.grammar);
            if (!gbnf) {
                set_error("the v2 sampling row's grammar handle is invalid");
                clear_generation_memory();
                return -1;
            }
            grammars[sequence].reset(llama_sampler_init_grammar(vocab, gbnf, "root"));
            if (!grammars[sequence]) {
                set_error("failed to compile the row's grammar (invalid GBNF)");
                clear_generation_memory();
                return -1;
            }
        }
    }
    reasoning_budget_rows budget(vocab, plan, n_sequences);
    for (size_t sequence = 0; sequence < n_sequences; ++sequence) {
        budget.prime(sequence, sequences[sequence].prompt_tokens, sequences[sequence].n_prompt);
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
            const retro_sampling_params_v2 & params = sequences[sequence].sampling;
            float sampled_logprob = 0.0f;
            const int32_t drawn = sample_token_v2(
                    sequence_logits[sequence].data(), n_vocab, params, rngs[sequence], scratch,
                    histories[sequence], grammars[sequence].get(),
                    &sampled_logprob, nullptr, nullptr, 0);
            if (drawn < 0) {
                // A grammar failure surfaces here; it is never faked.
                clear_generation_memory();
                return -1;
            }
            bool forced = false;
            const int32_t token = budget.next(sequence, drawn, forced);
            if (forced && grammars[sequence] && scratch.probs[token] == 0.0) {
                set_error("the reasoning budget forced a token excluded by the grammar");
                clear_generation_memory();
                return -1;
            }
            const size_t count = out_n_tokens[sequence];
            out_tokens[sequence*n_out_max + count] = token;
            record_forced(out_forced, sequence*n_out_max + count, forced);
            if (out_logprobs) {
                // Raw temperature-1 policy logprob, the v1 contract: computed
                // from the untransformed logits row, not from the oracle's
                // post-transformation distribution.
                out_logprobs[sequence*n_out_max + count] = token_logprob(
                        sequence_logits[sequence].data(), n_vocab, token,
                        raw_scratches[sequence].probs);
            }
            out_n_tokens[sequence] = count + 1;
            histories[sequence].push_back(token);
            if (grammars[sequence]) {
                llama_sampler_accept(grammars[sequence].get(), token);
            }
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
        if (decode(batch) != 0) {
            set_error("llama_decode failed during continuous v2 generation");
            clear_generation_memory();
            return -1;
        }
        for (size_t row = 0; row < active.size(); ++row) {
            const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(row));
            if (!logits) {
                set_error("continuous v2 generation produced no logits");
                clear_generation_memory();
                return -1;
            }
            sequence_logits[active[row]].assign(logits, logits + n_vocab);
        }
    }
    if (cache) {
        remember_generation_kv(*this, view.data(), n_sequences, placement,
                out_tokens, n_out_max, out_n_tokens);
    } else {
        clear_generation_memory();
    }
    return 0;
}

// ---------------------------------------------------------------------------
// The step-wise API: one llama_process per call, raw sequence operations, and
// the pool occupancy report. The plan comes from the caller; the execution is
// this core's. See retro_lora_train.h for the published contract.

void llama_batch_ext_deleter::operator()(llama_batch_ext * batch) const {
    if (batch) {
        llama_batch_ext_free(batch);
    }
}

llama_batch_ext * step_batch_cache::get(llama_context * target) {
    if (!target) {
        return nullptr;
    }
    if (ctx != target || !batch) {
        batch.reset(llama_batch_ext_init(target));
        ctx = target;
    }
    return batch.get();
}

int gen_core::step(
        const retro_step_row * rows,
        size_t n_rows,
        retro_step_out_row * out_rows,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t n_top_stride,
        step_request_table & requests,
        const reasoning_budget_plan * plan,
        llama_batch_ext * ext_batch) {
    if (!require_sequence_memory()) {
        return -1;
    }
    if (!rows || n_rows == 0 || !out_rows) {
        set_error("step rows and output rows are required");
        return -3;
    }
    std::vector<gen_detail::effective_step_row> effective(n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const retro_step_row & row = rows[r];
        if (row.struct_size < offsetof(retro_step_row, adapter) + sizeof(int32_t)) {
            set_error("retro_step_row.struct_size is too small for this build");
            return -3;
        }
        gen_detail::effective_step_row & out = effective[r];
        out.struct_size = row.struct_size;
        out.request = row.request;
        out.slot = row.slot;
        out.tokens = row.tokens;
        out.n = row.n;
        out.pos0 = row.pos0;
        out.sample = row.sample;
        out.sampling.v1 = row.sampling;
        out.sampling.v2 = nullptr;
        out.n_top_logprobs = row.n_top_logprobs;
        out.want_logprob = false;
        out.adapter = row.adapter;
        out.grammar = nullptr;
    }
    std::vector<gen_detail::step_out_row_internal> internal(n_rows);
    const int status = step_impl(effective.data(), n_rows, internal.data(), out_top_ids,
            out_top_logprobs, n_top_stride, requests, plan, ext_batch);
    if (status != 0) {
        return status;
    }
    // v1 output layout: the internal row minus the v2-only sampled_logprob.
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::step_out_row_internal & src = internal[r];
        retro_step_out_row & out = out_rows[r];
        out.struct_size = sizeof(retro_step_out_row);
        out.token = src.token;
        out.logprob = src.logprob;
        out.n_top = src.n_top;
        out.forced = src.forced;
        out.sampled = src.sampled;
    }
    return 0;
}

int gen_core::step_v2(
        const retro_step_row_v2 * rows,
        size_t n_rows,
        retro_step_out_row_v2 * out_rows,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t n_top_stride,
        step_request_table & requests,
        const reasoning_budget_plan * plan,
        llama_batch_ext * ext_batch) {
    if (!require_sequence_memory()) {
        return -1;
    }
    if (!rows || n_rows == 0 || !out_rows) {
        set_error("step rows and output rows are required");
        return -3;
    }
    std::vector<gen_detail::effective_step_row> effective(n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const retro_step_row_v2 & row = rows[r];
        if (row.struct_size < offsetof(retro_step_row_v2, adapter) + sizeof(int32_t)) {
            set_error("retro_step_row_v2.struct_size is too small for this build");
            return -3;
        }
        if (row.sampling.struct_size < sizeof(retro_sampling_params_v2)) {
            set_error("retro_sampling_params_v2.struct_size is too small for this build");
            return -3;
        }
        gen_detail::effective_step_row & out = effective[r];
        out.struct_size = row.struct_size;
        out.request = row.request;
        out.slot = row.slot;
        out.tokens = row.tokens;
        out.n = row.n;
        out.pos0 = row.pos0;
        out.sample = row.sample;
        out.sampling.v1 = retro_sampling_params{};
        out.sampling.v2 = &row.sampling;
        out.n_top_logprobs = row.n_top_logprobs;
        out.want_logprob = row.want_logprob != 0;
        out.adapter = row.adapter;
        out.grammar = row.sampling.grammar;
    }
    std::vector<gen_detail::step_out_row_internal> internal(n_rows);
    const int status = step_impl(effective.data(), n_rows, internal.data(), out_top_ids,
            out_top_logprobs, n_top_stride, requests, plan, ext_batch);
    if (status != 0) {
        return status;
    }
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::step_out_row_internal & src = internal[r];
        retro_step_out_row_v2 & out = out_rows[r];
        out.struct_size = sizeof(retro_step_out_row_v2);
        out.token = src.token;
        out.logprob = src.logprob;
        out.n_top = src.n_top;
        out.forced = src.forced;
        out.sampled = src.sampled;
        out.sampled_logprob = src.sampled_logprob;
    }
    return 0;
}

int gen_core::step_impl(
        const gen_detail::effective_step_row * rows,
        size_t n_rows,
        gen_detail::step_out_row_internal * out_rows,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t n_top_stride,
        step_request_table & requests,
        const reasoning_budget_plan * plan,
        llama_batch_ext * ext_batch) {
    if (!rows || n_rows == 0 || !out_rows) {
        set_error("step rows and output rows are required");
        return -3;
    }
    const int32_t n_seq_max = llama_n_seq_max(ctx);
    const int32_t n_batch = llama_n_batch(ctx);
    const int64_t n_ctx_seq = llama_n_ctx_seq(ctx);
    if (n_rows > static_cast<size_t>(n_batch)
            || (n_top_stride != 0 && n_rows > SIZE_MAX / n_top_stride)) {
        set_error("the step row count or output shape exceeds capacity");
        return -3;
    }
    // Read the evolution key before any other field: an older or truncated
    // caller may not own sizeof(retro_step_row[_v2]) bytes.
    constexpr size_t row_prefix_v1 = offsetof(retro_step_row, adapter) + sizeof(int32_t);
    constexpr size_t row_prefix_v2 = offsetof(retro_step_row_v2, adapter) + sizeof(int32_t);
    std::vector<int32_t> seen_requests;
    size_t total = 0;
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::effective_step_row & row = rows[r];
        const bool is_v2 = row.sampling.v2 != nullptr;
        if (row.struct_size < (is_v2 ? row_prefix_v2 : row_prefix_v1)) {
            set_error(is_v2
                    ? "retro_step_row_v2.struct_size is too small for this build"
                    : "retro_step_row.struct_size is too small for this build");
            return -3;
        }
        if (!row.tokens || row.n == 0) {
            set_error("a step row requires at least one token");
            return -3;
        }
        if (row.slot < 0 || row.slot >= n_seq_max) {
            set_error("step row slot is outside the context's sequence range");
            return -3;
        }
        if (row.adapter != 0) {
            if (!env.lora_router) {
                set_error("this context does not support per-row adapters: step row adapter must be 0");
                return -3;
            }
            if (row.adapter != RETRO_STEP_ADAPTER_BASE
                    && env.lora_router->adapters.find(row.adapter) == env.lora_router->adapters.end()) {
                set_error("step row adapter " + std::to_string(row.adapter) + " is not a loaded per-row adapter");
                return -3;
            }
        }
        if (row.pos0 < 0 || static_cast<int64_t>(row.pos0) + row.n > n_ctx_seq) {
            set_error("step row positions exceed the per-sequence context");
            return -3;
        }
        for (size_t previous = 0; previous < r; ++previous) {
            const gen_detail::effective_step_row & other = rows[previous];
            if (row.slot == other.slot
                    && row.pos0 < static_cast<int64_t>(other.pos0) + other.n
                    && other.pos0 < static_cast<int64_t>(row.pos0) + row.n) {
                set_error("step rows overlap in the same sequence slot");
                return -3;
            }
            if (row.slot == other.slot && row.adapter != other.adapter) {
                set_error("step rows of one sequence slot select different adapters");
                return -3;
            }
        }
        if (!validate_tokens(row.tokens, row.n)) {
            return -3;
        }
        if (row.n > static_cast<uint32_t>(n_batch)
                || total + row.n > static_cast<size_t>(n_batch)) {
            set_error("the step's tokens exceed the context's physical batch (n_batch)");
            return -3;
        }
        total += row.n;
        if (row.n_top_logprobs > 0 && (!out_top_ids || !out_top_logprobs)) {
            set_error("top logprobs requested but the output buffers are missing");
            return -3;
        }
        if (row.n_top_logprobs > n_top_stride) {
            set_error("a step row's n_top_logprobs exceeds the output stride");
            return -3;
        }
        if (row.sample) {
            if (row.request < 0) {
                set_error("a sampling step row requires a request id");
                return -3;
            }
            if (row.sampling.v2) {
                if (!validate_sampling_v2(*row.sampling.v2)) {
                    return -3;
                }
            } else {
                // The caller owns stopping; only the distribution parameters
                // participate in this primitive's sampling contract.
                retro_sampling_params sampling = row.sampling.v1;
                sampling.max_new_tokens = 1;
                if (!validate_sampling(sampling)) {
                    return -3;
                }
            }
        }
        if (row.request >= 0) {
            if (std::find(seen_requests.begin(), seen_requests.end(), row.request)
                    != seen_requests.end()) {
                set_error("a request id appears twice in one step");
                return -3;
            }
            seen_requests.push_back(row.request);
        }
    }

    using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    std::vector<sampler_ptr> compiled_grammars;
    compiled_grammars.reserve(n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const auto & row = rows[r];
        sampler_ptr compiled(nullptr, llama_sampler_free);
        if (row.sample && row.grammar) {
            const char * gbnf = retro_grammar_gbnf(row.grammar);
            const auto previous = requests.requests.find(row.request);
            if (previous == requests.requests.end() || !previous->second.grammar
                    || previous->second.grammar_gbnf != gbnf) {
                compiled.reset(llama_sampler_init_grammar(this->vocab(), gbnf, "root"));
                if (!compiled) {
                    set_error("failed to compile the request's grammar (invalid GBNF)");
                    return -3;
                }
            }
        }
        compiled_grammars.push_back(std::move(compiled));
    }

    // Per-row adapters: route each row's slot for this decode only. The
    // routes are cleared on every exit, so the batched generators, which know
    // nothing of them, always decode under the context's own adapter set.
    // A row with adapter 0 follows the owner's default (0 unless set).
    const int32_t default_adapter = env.lora_router ? env.lora_router->default_adapter : 0;
    const bool routed = default_adapter != 0 || std::any_of(rows, rows + n_rows,
            [](const gen_detail::effective_step_row & row) { return row.adapter != 0; });
    struct route_reset {
        llama_context * ctx = nullptr;
        ~route_reset() { if (ctx) llama_clear_seq_lora_routes(ctx); }
    } reset_routes;
    if (routed) {
        lora_row_router & router = *env.lora_router;
        if (!router.installed && !lora_router_install(router, ctx)) {
            return -1;
        }
        reset_routes.ctx = ctx;
        for (size_t r = 0; r < n_rows; ++r) {
            const int32_t adapter = rows[r].adapter != 0 ? rows[r].adapter : default_adapter;
            const int32_t route = lora_router_route(router, adapter);
            if (llama_set_seq_lora_route(ctx, rows[r].slot, route) != 0) {
                set_error("failed to route a step row to its adapter");
                return -1;
            }
        }
    }

    // The step mutates raw KV outside the prefix-cache protocol: forget the
    // bookkeeping so a later batched call never trusts a stale residency.
    if (cache) {
        cache->forget_bookkeeping();
    }

    gen_detail::batch_ext_guard owned(ext_batch ? nullptr : ctx);
    llama_batch_ext * ext = ext_batch ? ext_batch : owned.batch;
    if (!ext) {
        set_error("failed to allocate the step batch");
        return -1;
    }
    llama_batch_ext_clear(ext);
    std::vector<int32_t> last_index(n_rows, -1);
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::effective_step_row & row = rows[r];
        for (uint32_t i = 0; i < row.n; ++i) {
            const int32_t index = llama_batch_ext_add_token(
                    ext, static_cast<llama_seq_id>(row.slot),
                    static_cast<llama_token>(row.tokens[i]));
            if (index < 0) {
                set_error("the step batch is full or the slot is invalid");
                return -1;
            }
            const llama_pos pos = static_cast<llama_pos>(static_cast<int64_t>(row.pos0) + i);
            if (!llama_batch_ext_set_pos(ext, index, &pos)) {
                set_error("failed to set a step token's position");
                return -1;
            }
            if (i + 1 == row.n) {
                last_index[r] = index;
                if (row.sample && !llama_batch_ext_set_output_logits(ext, index, true)) {
                    set_error("failed to request logits for a sampling step row");
                    return -1;
                }
            }
        }
    }
    const int decode_status = decode_ext(*ext);
    const llama_vocab * vocab = this->vocab();
    const size_t n_vocab = this->n_vocab();
    // Record only input tokens actually decoded. On a partial failure the
    // caller resumes after seq_pos(), and the undecoded suffix must not have
    // primed the request's budget ahead of its KV state.
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::effective_step_row & row = rows[r];
        if (row.request < 0) {
            continue;
        }
        step_request_state & request = requests.requests[row.request];
        if (!request.plan_known || !same_reasoning_plan(request.plan_snapshot, *plan)) {
            request.budget = budget_machine();
            request.plan_snapshot = *plan;
            // Own the plan the machine reads: a later set_reasoning_budget()
            // replaces the model's plan, including its forced-token vector.
            request.budget = budget_machine(vocab, &request.plan_snapshot);
            request.plan_known = true;
            request.primed = false;
            request.prompt_tail.clear();
        }
        if (!plan->active || request.primed) {
            continue;
        }
        size_t decoded = row.n;
        if (decode_status != 0) {
            const llama_memory_t memory = llama_get_memory(ctx);
            const int64_t last = memory ? llama_memory_seq_pos_max(memory, row.slot) : -1;
            decoded = last < row.pos0 ? 0 : std::min<size_t>(row.n, last - row.pos0 + 1);
        }
        // Keep exactly the final n_prompt_tail tokens across chunk boundaries.
        request.prompt_tail.insert(request.prompt_tail.end(), row.tokens, row.tokens + decoded);
        if (request.prompt_tail.size() > plan->n_prompt_tail) {
            request.prompt_tail.erase(request.prompt_tail.begin(),
                    request.prompt_tail.end() - plan->n_prompt_tail);
        }
    }
    if (decode_status != 0) {
        set_error("llama_process failed during step: the KV may hold a prefix of the "
                  "submitted tokens (query it with the seq_pos entry point) - do not "
                  "resubmit tokens that are already resident");
        return -1;
    }

    gen_detail::sampler_scratch scratch;
    std::vector<int32_t> order;
    std::vector<double> work;
    for (size_t r = 0; r < n_rows; ++r) {
        const gen_detail::effective_step_row & row = rows[r];
        gen_detail::step_out_row_internal & out = out_rows[r];
        out.token = 0;
        out.logprob = 0.0f;
        out.n_top = 0;
        out.forced = 0;
        out.sampled = 0;
        out.sampled_logprob = 0.0f;
        if (!row.sample) {
            continue;
        }
        // The index llama_batch_ext_add_token returned resolves through the
        // context's output map after llama_process: exactly what
        // llama_get_logits_ith (and the sampled-token / target-logprob
        // accessors) accept for an extended batch.
        const float * logits = llama_get_logits_ith(ctx, last_index[r]);
        if (!logits) {
            set_error("the step decode produced no logits for a sampling row");
            return -1;
        }
        step_request_state & request = requests.requests[row.request];
        if (!request.seeded) {
            request.rng.seed(row.sampling.v2 ? row.sampling.v2->seed : row.sampling.v1.seed);
            request.seeded = true;
        }
        if (request.budget.active() && !request.primed) {
            request.budget.prime(request.prompt_tail.data(), request.prompt_tail.size());
            request.primed = true;
        }
        if (row.sampling.v2) {
            // v2 row: extended sampling through sample_token_v2. The grammar
            // handle is copied into the per-request sampler state here (at the
            // request's first sampling row, rebuilt when the GBNF text
            // changes), so the caller may free the handle once this call
            // returns.
            const retro_sampling_params_v2 & params = *row.sampling.v2;
            if (row.grammar) {
                const char * gbnf = retro_grammar_gbnf(row.grammar);
                if (!gbnf) {
                    set_error("the sampling row's grammar handle is invalid");
                    return -1;
                }
                if (!request.grammar || request.grammar_gbnf != gbnf) {
                    llama_sampler * compiled = compiled_grammars[r].release();
                    if (!compiled) {
                        set_error("failed to compile the request's grammar (invalid GBNF)");
                        return -1;
                    }
                    if (request.grammar) {
                        llama_sampler_free(request.grammar);
                    }
                    request.grammar = compiled;
                    request.grammar_gbnf = gbnf;
                }
            } else if (request.grammar) {
                llama_sampler_free(request.grammar);
                request.grammar = nullptr;
                request.grammar_gbnf.clear();
            }
            const uint32_t k = std::min<uint32_t>(
                    row.n_top_logprobs,
                    static_cast<uint32_t>(std::min<size_t>(n_top_stride, n_vocab)));
            float sampled_logprob = 0.0f;
            const int32_t sampled = sample_token_v2(
                    logits, n_vocab, params, request.rng, scratch, request.history,
                    request.grammar,
                    row.want_logprob ? &sampled_logprob : nullptr,
                    k > 0 ? out_top_ids + r * n_top_stride : nullptr,
                    k > 0 ? out_top_logprobs + r * n_top_stride : nullptr,
                    k);
            if (sampled < 0) {
                // A grammar failure surfaces here; it is never faked.
                return -1;
            }
            bool forced = false;
            const int32_t token = request.budget.next(sampled, forced);
            out.token = token;
            out.forced = forced ? 1 : 0;
            out.sampled = 1;
            if (forced && request.grammar && scratch.probs[token] == 0.0) {
                set_error("the reasoning budget forced a token excluded by the grammar");
                return -1;
            }
            request.history.push_back(token);
            if (request.grammar) {
                llama_sampler_accept(request.grammar, token);
            }
            out.sampled_logprob = !row.want_logprob ? 0.0f : forced
                    ? static_cast<float>(std::log(scratch.probs[token] / scratch.total))
                    : sampled_logprob;
            out.n_top = k;
            // The temperature-1 policy logprob of the emitted token, spelled
            // exactly like the v1 temperature-1 path so both versions report
            // the same value for the same token.
            const log_softmax_terms terms = log_softmax_terms_scalar(logits, n_vocab);
            out.logprob = static_cast<float>(
                    static_cast<double>(logits[token]) - terms.max - terms.log_sum);
        } else {
            // v1 row: the untouched oracle path.
            const retro_sampling_params & params = row.sampling.v1;
            const int32_t sampled = sample_token(
                    logits, n_vocab, params.temperature, params.top_p, request.rng, scratch);
            bool forced = false;
            const int32_t token = request.budget.next(sampled, forced);
            out.token = token;
            out.forced = forced ? 1 : 0;
            out.sampled = 1;
            // The same expression the batched generators report, so a step-wise
            // loop reproduces their logprobs bit for bit.
            out.logprob = params.temperature == 1.0f
                    ? static_cast<float>(
                            static_cast<double>(logits[token]) - scratch.max - std::log(scratch.total))
                    : token_logprob(logits, n_vocab, token, scratch.probs);
            const uint32_t k = std::min<uint32_t>(
                    row.n_top_logprobs,
                    static_cast<uint32_t>(std::min<size_t>(n_top_stride, n_vocab)));
            if (k > 0) {
                top_token_logprobs(logits, n_vocab, k, order, work,
                        out_top_ids + r * n_top_stride, out_top_logprobs + r * n_top_stride);
                out.n_top = k;
            }
        }
    }
    return 0;
}

bool gen_core::seq_rm(int32_t slot, int32_t p0, int32_t p1) {
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("the context has no sequence memory");
        return false;
    }
    if (slot < 0 || slot >= llama_n_seq_max(ctx)) {
        set_error("slot is outside the context's sequence range");
        return false;
    }
    if (!llama_memory_seq_rm(memory, static_cast<llama_seq_id>(slot), p0, p1)) {
        set_error("the memory cannot remove that partial range; "
                  "remove the whole sequence instead");
        return false;
    }
    if (cache) {
        cache->forget_bookkeeping();
    }
    return true;
}

bool gen_core::seq_cp(int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("the context has no sequence memory");
        return false;
    }
    const int32_t n_seq_max = llama_n_seq_max(ctx);
    if (src < 0 || src >= n_seq_max || dst < 0 || dst >= n_seq_max) {
        set_error("seq_cp slots are outside the context's sequence range");
        return false;
    }
    llama_memory_seq_cp(memory, static_cast<llama_seq_id>(src), static_cast<llama_seq_id>(dst), p0, p1);
    if (cache) {
        cache->forget_bookkeeping();
    }
    return true;
}

int gen_core::seq_save(
        int32_t slot, uint32_t flags, uint8_t * buffer, size_t n_buffer, size_t * out_n_bytes) {
    if (!out_n_bytes) {
        set_error("out_n_bytes is required");
        return -1;
    }
    constexpr uint32_t known_flags = RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    if (flags & ~known_flags) {
        set_error("only host snapshots (flags 0 or PARTIAL_ONLY) are supported");
        return -1;
    }
    if (slot < 0 || slot >= llama_n_seq_max(ctx)) {
        set_error("slot is outside the context's sequence range");
        return -1;
    }
    const size_t required = llama_state_seq_get_size_ext(ctx, slot, flags);
    if (required == 0) {
        set_error("failed to determine the sequence snapshot size");
        return -1;
    }
    *out_n_bytes = required;
    if (!buffer) {
        return 0;
    }
    if (n_buffer < required) {
        set_error("the save buffer is too small for this sequence's state");
        return -2;
    }
    const size_t written = llama_state_seq_get_data_ext(ctx, buffer, required, slot, flags);
    if (written == 0) {
        set_error("saving the sequence state produced no data");
        return -1;
    }
    *out_n_bytes = written;
    return 0;
}

int gen_core::seq_load(
        int32_t slot, uint32_t flags, const uint8_t * buffer, size_t n_bytes, size_t * out_n_read) {
    if (!out_n_read) {
        set_error("out_n_read is required");
        return -1;
    }
    constexpr uint32_t known_flags = RETRO_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    if (flags & ~known_flags) {
        set_error("only host snapshots (flags 0 or PARTIAL_ONLY) are supported");
        return -1;
    }
    if (slot < 0 || slot >= llama_n_seq_max(ctx)) {
        set_error("slot is outside the context's sequence range");
        return -1;
    }
    if (!buffer || n_bytes == 0) {
        set_error("a snapshot buffer is required");
        return -1;
    }
    if (cache) {
        cache->forget_bookkeeping();
    }
    const size_t read = llama_state_seq_set_data_ext(ctx, buffer, n_bytes, slot, flags);
    if (read == 0) {
        set_error("the snapshot was refused (wrong size, flags or corrupted data)");
        return -1;
    }
    *out_n_read = read;
    return 0;
}

int gen_core::seq_pos(int32_t slot, int32_t * out_pos_min, int32_t * out_pos_max) {
    if (!out_pos_min || !out_pos_max) {
        set_error("the position outputs are required");
        return -1;
    }
    if (slot < 0 || slot >= llama_n_seq_max(ctx)) {
        set_error("slot is outside the context's sequence range");
        return -1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        set_error("the context has no sequence memory");
        return -1;
    }
    *out_pos_min = llama_memory_seq_pos_min(memory, static_cast<llama_seq_id>(slot));
    *out_pos_max = llama_memory_seq_pos_max(memory, static_cast<llama_seq_id>(slot));
    return 0;
}

int gen_core::memory(retro_engine_memory_info * out) {
    if (!out) {
        set_error("out is required");
        return -1;
    }
    out->struct_size = sizeof(retro_engine_memory_info);
    out->n_slots = static_cast<uint32_t>(llama_n_seq_max(ctx));
    out->cells_total = static_cast<uint64_t>(llama_n_ctx(ctx));
    llama_memory_t memory = llama_get_memory(ctx);
    uint64_t used = 0;
    uint64_t recurrent = 0;
    for (int32_t slot = 0; slot < llama_n_seq_max(ctx); ++slot) {
        if (memory) {
            const llama_pos pos_min = llama_memory_seq_pos_min(memory, static_cast<llama_seq_id>(slot));
            const llama_pos pos_max = llama_memory_seq_pos_max(memory, static_cast<llama_seq_id>(slot));
            if (pos_min >= 0) {
                used += static_cast<uint64_t>(pos_max - pos_min + 1);
            }
        }
        // The fork's PARTIAL_ONLY state: recurrent state, plus the SWA KV of a
        // sliding-window layer where the model has one.
        recurrent += llama_state_seq_get_size_ext(
                ctx, static_cast<llama_seq_id>(slot), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    out->cells_used = used;
    uint64_t kv = 0;
    for (const auto & item : context_memory_by_buffer(ctx)) {
        kv += item.second.kv;
    }
    out->kv_bytes = kv;
    out->recurrent_bytes = recurrent;
    return 0;
}

// ---------------------------------------------------------------------------
// Logprob / top-k helpers shared with the trainer's scoring wrappers.

float token_logprob_scalar(const float * logits, size_t n_vocab, int32_t token) {
    const log_softmax_terms terms = log_softmax_terms_scalar(logits, n_vocab);
    return static_cast<float>(
            static_cast<double>(logits[token]) - terms.max - terms.log_sum);
}

float token_logprob(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        std::vector<double> & work) {
    const log_softmax_terms terms = row_log_softmax_terms(logits, n_vocab, work);
    return static_cast<float>(
            static_cast<double>(logits[token]) - terms.max - terms.log_sum);
}

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

bool lora_router_install(lora_row_router & router, llama_context * ctx) {
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> scales;
    std::map<int32_t, int32_t> route_of;
    for (const auto & [id, entry] : router.adapters) {
        adapters.push_back(entry.adapter.get());
        scales.push_back(entry.scale);
        route_of.emplace(id, static_cast<int32_t>(adapters.size()));
    }
    router.installed = false;
    router.route_of.clear();
    std::string err;
    if (llama_set_adapters_lora_rows(ctx, adapters.data(), adapters.size(), scales.data(), &err) != 0) {
        set_error("failed to stack the per-row adapters: " + err);
        return false;
    }
    router.installed = true;
    router.route_of = std::move(route_of);
    return true;
}

void lora_router_info(llama_context * ctx, retro_adapter_rows_info & out) {
    out = retro_adapter_rows_info{};
    out.struct_size = sizeof(retro_adapter_rows_info);
    llama_adapter_lora_rows_info info {};
    if (ctx && llama_get_adapters_lora_rows_info(ctx, &info)) {
        out.n_adapters = static_cast<uint32_t>(info.n_adapters);
        out.max_rank = static_cast<uint32_t>(info.r_max);
        out.stacked_bytes = info.n_bytes;
        out.unpadded_bytes = info.n_bytes_unpadded;
    }
}

lora_adapter_ptr clone_lora_adapter(llama_model * model, const llama_adapter_lora & source) {
    lora_adapter_ptr copy(new llama_adapter_lora(model));
    copy->alpha = source.alpha;
    copy->gguf_kv = source.gguf_kv;
    std::map<ggml_backend_buffer_type_t, std::vector<std::pair<std::string, llama_adapter_lora_weight>>> by_buft;
    for (const auto & [name, weight] : source.ab_map) {
        by_buft[ggml_backend_buffer_get_type(weight.a->buffer)].emplace_back(name, weight);
    }
    std::vector<std::pair<llama_adapter_lora_weight, llama_adapter_lora_weight>> pairs;
    for (const auto & [buft, weights] : by_buft) {
        ggml_init_params params {
            /*.mem_size   =*/ weights.size() * 2 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            set_error("failed to allocate the adapter copy's metadata");
            return nullptr;
        }
        copy->ctxs.emplace_back(ctx);
        for (const auto & [name, weight] : weights) {
            ggml_tensor * a = ggml_dup_tensor(ctx, weight.a);
            ggml_tensor * b = ggml_dup_tensor(ctx, weight.b);
            ggml_set_name(a, weight.a->name);
            ggml_set_name(b, weight.b->name);
            copy->ab_map.emplace(name, llama_adapter_lora_weight(a, b));
            pairs.emplace_back(weight, llama_adapter_lora_weight(a, b));
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
        if (!buffer) {
            set_error("failed to allocate the adapter copy on " + std::string(ggml_backend_buft_name(buft)));
            return nullptr;
        }
        copy->bufs.emplace_back(std::move(buffer));
    }
    for (const auto & [from, to] : pairs) {
        ggml_backend_tensor_copy(from.a, to.a);
        ggml_backend_tensor_copy(from.b, to.b);
    }
    model->loras.insert(copy.get());
    return copy;
}

} // namespace retro
