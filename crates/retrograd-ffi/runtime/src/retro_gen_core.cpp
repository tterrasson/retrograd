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

// ---------------------------------------------------------------------------
// The reasoning budget: one llama.cpp state machine per row of a call.

// One llama.cpp reasoning-budget state machine per row of a call. Retrograd
// samples on its own - the CPU oracle or a device chain - so the machine is
// used for its state and not as a sampler: once it is forcing, its token
// replaces the sampled one, which is the only token its masked distribution
// could have drawn. Rows report which of their tokens were forced, since those
// are the runtime's choice and not the policy's.
class reasoning_budget_rows {
public:
    reasoning_budget_rows(const llama_vocab * vocab, const reasoning_budget_plan * plan, size_t n_rows) {
        if (!plan || !plan->active) {
            return;
        }
        plan_ = plan;
        rows_.resize(n_rows);
        for (row_state & row : rows_) {
            row.sampler = common_reasoning_budget_init(
                    vocab, { plan->start }, plan->ends, plan->forced, plan->budget);
        }
    }

    ~reasoning_budget_rows() {
        for (row_state & row : rows_) {
            if (row.sampler) {
                llama_sampler_free(row.sampler);
            }
        }
    }

    reasoning_budget_rows(const reasoning_budget_rows &) = delete;
    reasoning_budget_rows & operator=(const reasoning_budget_rows &) = delete;

    // Replays the end of the prompt, as long as the template's generation
    // prompt, so a prompt that already opened the reasoning block starts the
    // row counting and one that closed it starts it done.
    //
    // A machine that starts forcing on a prompt token - a budget of zero on an
    // opening tag, or one spent inside the prompt - stops the replay there: the
    // machine reads every token it accepts while forcing as the next forced
    // one, so the prompt's newline after the opening tag would stand for the
    // end tag and the tag would never be emitted. The rest of the prompt
    // belongs to the prompt, unless it closes the block itself (an empty block
    // written by the template): then the forced sequence is accepted in its
    // place, which leaves the machine done, and the replay resumes after the tag.
    void prime(size_t row, const int32_t * prompt, size_t n_prompt) {
        if (!plan_) {
            return;
        }
        const int32_t * end = prompt + n_prompt;
        const int32_t * at = prompt + (n_prompt - std::min(plan_->n_prompt_tail, n_prompt));
        while (at != end) {
            accept(row, *at++);
            if (common_reasoning_budget_get_state(rows_[row].sampler) != REASONING_BUDGET_FORCING) {
                continue;
            }
            const int32_t * closed = closing_tag_end(at, end);
            if (!closed) {
                return;
            }
            for (const int32_t token : plan_->forced) {
                accept(row, token);
            }
            at = closed;
        }
    }

    // The token the row emits in place of `sampled`, and whether the budget
    // forced it.
    int32_t next(size_t row, int32_t sampled, bool & forced) {
        forced = false;
        if (!plan_) {
            return sampled;
        }
        row_state & state = rows_[row];
        int32_t token = sampled;
        if (common_reasoning_budget_get_state(state.sampler) == REASONING_BUDGET_FORCING) {
            // The machine forces exactly `forced`, one token per accept, so a
            // forcing machine whose local position ran past it is out of step.
            GGML_ASSERT(state.force_pos < plan_->forced.size());
            token = plan_->forced[state.force_pos++];
            forced = true;
        }
        accept(row, token);
        // The forced sequence is emitted whole: the machine is done exactly
        // when the last forced token went out, and still forcing before.
        if (forced) {
            const bool done = state.force_pos == plan_->forced.size();
            const auto after = common_reasoning_budget_get_state(state.sampler);
            GGML_ASSERT(done ? after == REASONING_BUDGET_DONE : after == REASONING_BUDGET_FORCING);
        }
        return token;
    }

private:
    struct row_state {
        llama_sampler * sampler = nullptr;
        // Position in `forced`, which the machine keeps to itself.
        size_t force_pos = 0;
    };

    // One past the earliest end tag in [from, to), or null if none closes there.
    const int32_t * closing_tag_end(const int32_t * from, const int32_t * to) const {
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

    void accept(size_t row, int32_t token) {
        row_state & state = rows_[row];
        const auto before = common_reasoning_budget_get_state(state.sampler);
        llama_sampler_accept(state.sampler, token);
        if (common_reasoning_budget_get_state(state.sampler) == REASONING_BUDGET_FORCING
                && before != REASONING_BUDGET_FORCING) {
            state.force_pos = 0;
        }
    }

    const reasoning_budget_plan * plan_ = nullptr;
    std::vector<row_state> rows_;
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

} // namespace retro
