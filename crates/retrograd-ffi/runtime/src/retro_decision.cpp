// Decision models on the runtime side: what a GGUF declares about its decision
// head, the context such a model needs, and the evaluation of decision prompts.
//
// A decision model answers typed questions in one forward pass. Everything
// around that pass - the request, the template context, the layout of the
// tokens, the softmax and the temperatures - is pure logic and lives in Rust;
// this file only reads metadata, evaluates token sequences and reads scores
// where the readout says.
#include "retro_gen_core.hpp"

#include "json.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace retro {

namespace {

using json = common_json;

// A metadata value as llama.cpp prints it, or false when the key is absent.
bool meta_string(const llama_model * model, const std::string & key, std::string & out) {
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), nullptr, 0);
    if (n < 0) {
        return false;
    }
    out.assign(static_cast<size_t>(n) + 1, '\0');
    llama_model_meta_val_str(model, key.c_str(), out.data(), out.size());
    out.resize(static_cast<size_t>(n));
    return true;
}

bool meta_at(const llama_model * model, int32_t i, std::string & key, std::string & value) {
    const int32_t n_key = llama_model_meta_key_by_index(model, i, nullptr, 0);
    const int32_t n_value = llama_model_meta_val_str_by_index(model, i, nullptr, 0);
    if (n_key < 0 || n_value < 0) {
        return false;
    }
    key.assign(static_cast<size_t>(n_key) + 1, '\0');
    value.assign(static_cast<size_t>(n_value) + 1, '\0');
    llama_model_meta_key_by_index(model, i, key.data(), key.size());
    llama_model_meta_val_str_by_index(model, i, value.data(), value.size());
    key.resize(static_cast<size_t>(n_key));
    value.resize(static_cast<size_t>(n_value));
    return true;
}

// `<arch>.decision.type`, or false when the model has none.
bool decision_type(const llama_model * model, std::string & out) {
    std::string arch;
    return meta_string(model, "general.architecture", arch)
            && meta_string(model, arch + ".decision.type", out);
}

json token_or_null(llama_token token) {
    return token == LLAMA_TOKEN_NULL ? json(nullptr) : json(static_cast<int64_t>(token));
}

}  // namespace

std::string describe_decision_model(const llama_model * model) {
    json out = json::object();
    std::string arch;
    const bool has_arch = meta_string(model, "general.architecture", arch);
    out["architecture"] = has_arch ? json(arch) : json(nullptr);

    std::string type;
    out["decision_type"] = has_arch && meta_string(model, arch + ".decision.type", type)
            ? json(type)
            : json(nullptr);

    json metadata = json::object();
    if (has_arch) {
        const std::string prefix = arch + ".decision.";
        const int32_t n_meta = llama_model_meta_count(model);
        std::string key;
        std::string value;
        for (int32_t i = 0; i < n_meta; ++i) {
            if (meta_at(model, i, key, value) && key.size() > prefix.size()
                    && key.compare(0, prefix.size(), prefix) == 0) {
                metadata[key.substr(prefix.size())] = value;
            }
        }
    }
    out["metadata"] = metadata;

    bool has_default = false;
    json templates = json::array();
    for (const std::string & name : model_template_names(model, &has_default)) {
        templates.push_back(name);
    }
    out["templates"] = templates;
    out["has_default_template"] = has_default;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    out["mask_token"] = token_or_null(llama_vocab_mask(vocab));
    out["sep_token"] = token_or_null(llama_vocab_sep(vocab));
    out["n_embd_out"] = static_cast<int64_t>(llama_model_n_embd_out(model));
    out["n_cls_out"] = static_cast<int64_t>(llama_model_n_cls_out(model));

    json labels = json::array();
    for (uint32_t i = 0; llama_model_decision_label(model, i) != nullptr; ++i) {
        labels.push_back(std::string(llama_model_decision_label(model, i)));
    }
    out["decision_labels"] = labels;

    // The texts an input may spell of the tokens that are not plain text,
    // which some decision types break or quote in their inputs.
    json special = json::array();
    const llama_token n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        const auto attr = llama_vocab_get_attr(vocab, id);
        const bool control = (attr & LLAMA_TOKEN_ATTR_CONTROL) != 0;
        const bool user_defined = (attr & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;
        if (!control && !user_defined) {
            continue;
        }
        json token = json::object();
        token["text"] = std::string(llama_vocab_get_text(vocab, id));
        token["control"] = control;
        token["user_defined"] = user_defined;
        special.push_back(token);
    }
    out["special_tokens"] = special;
    // A token text need not be UTF-8: such bytes are written as U+FFFD.
    return out.dump_safe();
}

bool gguf_is_decision_model(const char * model_path) {
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * gguf = gguf_init_from_file(model_path, params);
    if (!gguf) {
        // The load that follows reports the unreadable file.
        return false;
    }
    bool found = false;
    const int64_t arch = gguf_find_key(gguf, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(gguf, arch) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(gguf, arch)) + ".decision.type";
        found = gguf_find_key(gguf, key.c_str()) >= 0;
    }
    gguf_free(gguf);
    return found;
}

// The decision types whose readout is the embeddings output, as llama.cpp's
// common_init_from_params lists them: the context is created in embeddings
// mode, with no pooling for the ones read per token and the pooling of the
// GGUF for the one read pooled.
bool decision_context_pooling(const llama_model * model, enum llama_pooling_type * out_pooling) {
    std::string type;
    if (!decision_type(model, type)) {
        return false;
    }
    static const char * const per_token[] = {
        "laya", "kev", "clef", "lfm2-d1-omni", "vllm-decision2",
    };
    for (const char * name : per_token) {
        if (type == name) {
            *out_pooling = LLAMA_POOLING_TYPE_NONE;
            return true;
        }
    }
    if (type == "openbmb-minicpm-ocr-jev") {
        *out_pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
        return true;
    }
    return false;
}

namespace {

struct read_plan {
    // Positions whose output this prompt needs, ascending.
    std::vector<int32_t> outputs;
    // What each needed output holds once decoded, by position: one float for a
    // marker column, the labels' logits, or the whole row for a pointer.
    std::map<int32_t, std::vector<float>> captured;
};

std::string describe_prompt(size_t index) {
    return "decision prompt " + std::to_string(index);
}

// The scores a prompt writes.
size_t prompt_scores(int32_t readout, const retro_decision_prompt & prompt) {
    if (readout == RETRO_DECIDE_JOINT || readout == RETRO_DECIDE_POOLED) {
        return prompt.n_scores;
    }
    if (readout == RETRO_DECIDE_LABEL_LOGITS && prompt.label_groups) {
        return prompt.n_label_groups;
    }
    return prompt.n_reads;
}

// The readouts that see a whole prompt at once: a joint head reads the spans
// of every question of its micro-batch, a pooled output the whole sequence.
// Each prompt is the only sequence of its decode, in one micro-batch, every
// token an output, as llama-server evaluates them.
int decide_alone(
        llama_model * model,
        llama_context * ctx,
        int32_t readout,
        const retro_decision_prompt * prompts,
        size_t n_prompts,
        float * out_scores,
        size_t n_scores_max) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_embd_out = llama_model_n_embd_out(model);
    const size_t n_micro = static_cast<size_t>(std::min(llama_n_batch(ctx), llama_n_ubatch(ctx)));
    llama_memory_t memory = llama_get_memory(ctx);

    size_t n_scores = 0;
    for (size_t p = 0; p < n_prompts; ++p) {
        const retro_decision_prompt & prompt = prompts[p];
        if (!prompt.tokens || prompt.n_tokens == 0) {
            set_error(describe_prompt(p) + " has no tokens");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        for (size_t i = 0; i < prompt.n_tokens; ++i) {
            if (prompt.tokens[i] < 0 || prompt.tokens[i] >= n_vocab) {
                set_error(describe_prompt(p) + " has a token outside the vocabulary");
                return RETRO_DECIDE_INVALID_INPUT;
            }
        }
        if (prompt.n_tokens > n_micro) {
            set_error(describe_prompt(p) + " is " + std::to_string(prompt.n_tokens)
                    + " tokens, but it must fit one micro-batch of " + std::to_string(n_micro));
            return RETRO_DECIDE_INVALID_INPUT;
        }
        if (readout == RETRO_DECIDE_JOINT) {
            if (!prompt.reads || prompt.n_reads != prompt.n_tokens) {
                set_error(describe_prompt(p) + " needs one decision order per token");
                return RETRO_DECIDE_INVALID_INPUT;
            }
            for (size_t i = 0; i < prompt.n_reads; ++i) {
                if (prompt.reads[i] < LLAMA_DECISION_ORDER_NONE || prompt.reads[i] > LLAMA_DECISION_ORDER_OPTION) {
                    set_error(describe_prompt(p) + " has an unknown decision order");
                    return RETRO_DECIDE_INVALID_INPUT;
                }
            }
            if (prompt.n_scores == 0 || prompt.n_scores > prompt.n_tokens) {
                set_error(describe_prompt(p) + " reads more scores than it has rows");
                return RETRO_DECIDE_INVALID_INPUT;
            }
        } else if (n_embd_out <= 0 || prompt.n_scores == 0
                || prompt.n_scores > static_cast<size_t>(n_embd_out)) {
            set_error(describe_prompt(p) + " reads more scores than the pooled output has");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        if (prompt.n_scores > std::numeric_limits<size_t>::max() - n_scores) {
            set_error("too many decision scores");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        n_scores += prompt.n_scores;
    }
    if (!out_scores || n_scores_max < n_scores) {
        set_error("out_scores buffer is too small");
        return RETRO_DECIDE_INVALID_INPUT;
    }

    llama_set_embeddings(ctx, true);
    struct batch_ext_guard {
        llama_batch_ext * batch;
        ~batch_ext_guard() {
            llama_batch_ext_free(batch);
        }
    } guard { llama_batch_ext_init(ctx) };

    size_t written = 0;
    for (size_t p = 0; p < n_prompts; ++p) {
        const retro_decision_prompt & prompt = prompts[p];
        if (memory) {
            llama_memory_clear(memory, true);
        }
        llama_batch_ext * batch = guard.batch;
        llama_batch_ext_clear(batch);
        for (size_t i = 0; i < prompt.n_tokens; ++i) {
            const int32_t index = llama_batch_ext_add_token(batch, 0, prompt.tokens[i]);
            const llama_pos pos = static_cast<llama_pos>(i);
            if (index < 0 || !llama_batch_ext_set_pos(batch, index, &pos)
                    || !llama_batch_ext_set_output_embd(batch, index, true)) {
                set_error("the decision batch could not take " + describe_prompt(p));
                return -1;
            }
            if (readout == RETRO_DECIDE_JOINT && prompt.reads[i] != LLAMA_DECISION_ORDER_NONE
                    && !llama_batch_ext_set_decision_order(
                            batch, index, static_cast<llama_decision_order>(prompt.reads[i]))) {
                set_error("the decision batch could not take the order of " + describe_prompt(p));
                return -1;
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            set_error("llama_decode failed while evaluating decision prompts");
            return -1;
        }
        if (readout == RETRO_DECIDE_JOINT) {
            // Row i of the output is the score of option i of the prompt.
            for (size_t i = 0; i < prompt.n_scores; ++i) {
                const float * embd = llama_get_embeddings_ith(ctx, static_cast<int32_t>(i));
                if (!embd) {
                    set_error("the decode produced no embeddings for a joint decision head");
                    return -1;
                }
                out_scores[written++] = embd[0];
            }
        } else {
            const float * embd = llama_get_embeddings_seq(ctx, 0);
            if (!embd) {
                set_error("the decode produced no pooled embeddings for a decision readout");
                return -1;
            }
            for (size_t i = 0; i < prompt.n_scores; ++i) {
                out_scores[written++] = embd[i];
            }
        }
    }
    if (memory) {
        llama_memory_clear(memory, true);
    }
    return 0;
}

}  // namespace

int decide_prompts(
        llama_model * model,
        llama_context * ctx,
        int32_t readout,
        const retro_decision_prompt * prompts,
        size_t n_prompts,
        size_t n_pack_max,
        bool share_prefix,
        float * out_scores,
        size_t n_scores_max) {
    if (readout != RETRO_DECIDE_MARKER_COLUMN && readout != RETRO_DECIDE_LABEL_LOGITS
            && readout != RETRO_DECIDE_POINTER && readout != RETRO_DECIDE_JOINT
            && readout != RETRO_DECIDE_POOLED) {
        set_error("decision readout must be one of RETRO_DECIDE_*");
        return RETRO_DECIDE_INVALID_INPUT;
    }
    if (!prompts || n_prompts == 0) {
        set_error("decision prompts are required");
        return RETRO_DECIDE_INVALID_INPUT;
    }
    if (readout == RETRO_DECIDE_JOINT || readout == RETRO_DECIDE_POOLED) {
        return decide_alone(model, ctx, readout, prompts, n_prompts, out_scores, n_scores_max);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_embd_out = llama_model_n_embd_out(model);
    llama_memory_t memory = llama_get_memory(ctx);
    // An encoder evaluates a whole prompt in one micro-batch; a context with
    // memory splits a batch into micro-batches itself.
    const size_t n_batch = memory
            ? static_cast<size_t>(llama_n_batch(ctx))
            : static_cast<size_t>(std::min(llama_n_batch(ctx), llama_n_ubatch(ctx)));
    const size_t n_seq_max = std::min(static_cast<size_t>(llama_n_seq_max(ctx)), std::max<size_t>(n_pack_max, 1));
    const size_t n_ctx = static_cast<size_t>(llama_n_ctx(ctx));
    const size_t n_ctx_seq = static_cast<size_t>(llama_n_ctx_seq(ctx));
    // A partitioned pool gives each sequence its own cells, a unified one
    // shares all of them.
    const bool unified_cells = n_ctx_seq >= n_ctx;

    size_t n_scores = 0;
    for (size_t p = 0; p < n_prompts; ++p) {
        const retro_decision_prompt & prompt = prompts[p];
        if (!prompt.tokens || prompt.n_tokens == 0) {
            set_error(describe_prompt(p) + " has no tokens");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        if (!prompt.reads || prompt.n_reads == 0) {
            set_error(describe_prompt(p) + " reads nothing");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        for (size_t i = 0; i < prompt.n_tokens; ++i) {
            if (prompt.tokens[i] < 0 || prompt.tokens[i] >= n_vocab) {
                set_error(describe_prompt(p) + " has a token outside the vocabulary");
                return RETRO_DECIDE_INVALID_INPUT;
            }
        }
        const size_t limit = memory ? n_ctx_seq : n_batch;
        if (prompt.n_tokens > limit) {
            std::ostringstream message;
            message << describe_prompt(p) << " is " << prompt.n_tokens << " tokens, but "
                    << (memory ? "a sequence of the context holds " : "a non-causal prompt must fit one batch of ")
                    << limit;
            set_error(message.str());
            return RETRO_DECIDE_INVALID_INPUT;
        }
        if (readout == RETRO_DECIDE_LABEL_LOGITS && prompt.label_groups) {
            size_t n_grouped = 0;
            for (size_t g = 0; g < prompt.n_label_groups; ++g) {
                if (prompt.label_groups[g] <= 0) {
                    n_grouped = 0;
                    break;
                }
                n_grouped += static_cast<size_t>(prompt.label_groups[g]);
            }
            if (prompt.n_label_groups == 0 || n_grouped != prompt.n_reads) {
                set_error(describe_prompt(p) + " has label groups that do not cover its labels");
                return RETRO_DECIDE_INVALID_INPUT;
            }
        }
        for (size_t r = 0; r < prompt.n_reads; ++r) {
            const int32_t read = prompt.reads[r];
            const bool in_range = readout == RETRO_DECIDE_LABEL_LOGITS
                    ? read >= 0 && read < n_vocab
                    : read >= 0 && static_cast<size_t>(read) < prompt.n_tokens;
            if (!in_range) {
                set_error(describe_prompt(p) + (readout == RETRO_DECIDE_LABEL_LOGITS
                        ? " reads a label outside the vocabulary"
                        : " reads a position outside its tokens"));
                return RETRO_DECIDE_INVALID_INPUT;
            }
        }
        if (readout == RETRO_DECIDE_MARKER_COLUMN && (prompt.column < 0 || prompt.column >= n_embd_out)) {
            set_error(describe_prompt(p) + " reads a column outside the embeddings output");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        if (readout == RETRO_DECIDE_POINTER
                && (prompt.pointer < 0 || static_cast<size_t>(prompt.pointer) >= prompt.n_tokens)) {
            set_error(describe_prompt(p) + " has its query outside its tokens");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        const size_t n_prompt_scores = prompt_scores(readout, prompt);
        if (n_prompt_scores > std::numeric_limits<size_t>::max() - n_scores) {
            set_error("too many decision scores");
            return RETRO_DECIDE_INVALID_INPUT;
        }
        n_scores += n_prompt_scores;
    }
    if (readout == RETRO_DECIDE_POINTER && (n_embd_out <= 0 || n_embd_out % 2 != 0)) {
        set_error("a pointer readout needs an even embeddings output width");
        return -1;
    }
    if (readout != RETRO_DECIDE_LABEL_LOGITS && n_embd_out <= 0) {
        set_error("model reports no output embedding width");
        return -1;
    }
    if (!out_scores || n_scores_max < n_scores) {
        set_error("out_scores buffer is too small");
        return RETRO_DECIDE_INVALID_INPUT;
    }

    llama_set_embeddings(ctx, readout != RETRO_DECIDE_LABEL_LOGITS);
    const size_t row_width = static_cast<size_t>(n_embd_out);
    size_t written = 0;

    size_t first = 0;
    while (first < n_prompts) {
        // The prompts of this pack: one sequence each.
        size_t last = first + 1;
        size_t n_cells = prompts[first].n_tokens;
        while (last < n_prompts && last - first < n_seq_max) {
            const size_t next = prompts[last].n_tokens;
            if (memory ? (unified_cells && n_cells + next > n_ctx) : n_cells + next > n_batch) {
                break;
            }
            n_cells += next;
            ++last;
        }
        const size_t n_pack = last - first;

        std::vector<read_plan> plans(n_pack);
        for (size_t k = 0; k < n_pack; ++k) {
            const retro_decision_prompt & prompt = prompts[first + k];
            std::vector<int32_t> & outputs = plans[k].outputs;
            if (readout == RETRO_DECIDE_LABEL_LOGITS) {
                outputs.push_back(static_cast<int32_t>(prompt.n_tokens - 1));
            } else {
                outputs.assign(prompt.reads, prompt.reads + prompt.n_reads);
                if (readout == RETRO_DECIDE_POINTER) {
                    outputs.push_back(prompt.pointer);
                }
            }
            std::sort(outputs.begin(), outputs.end());
            outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
        }

        // The prefix every prompt of the pack starts with, decoded once and
        // copied to the other sequences. It stops before the first position a
        // prompt reads, so every read output is decoded in its own sequence.
        size_t prefix = 0;
        if (memory && share_prefix && n_pack > 1) {
            prefix = prompts[first].n_tokens;
            for (size_t k = 0; k < n_pack; ++k) {
                const retro_decision_prompt & prompt = prompts[first + k];
                size_t common = 0;
                const size_t n = std::min(prefix, prompt.n_tokens);
                while (common < n && prompt.tokens[common] == prompts[first].tokens[common]) {
                    ++common;
                }
                prefix = std::min({ common, static_cast<size_t>(plans[k].outputs.front()) });
            }
        }

        struct entry {
            llama_token token;
            llama_pos pos;
            llama_seq_id seq;
            bool output;
            size_t plan;
        };
        std::vector<entry> entries;
        entries.reserve(n_cells);
        for (size_t i = 0; i < prefix; ++i) {
            entries.push_back(entry{ prompts[first].tokens[i], static_cast<llama_pos>(i), 0, false, 0 });
        }
        const size_t n_prefix_entries = entries.size();
        for (size_t k = 0; k < n_pack; ++k) {
            const retro_decision_prompt & prompt = prompts[first + k];
            const std::vector<int32_t> & outputs = plans[k].outputs;
            for (size_t i = prefix; i < prompt.n_tokens; ++i) {
                const auto pos = static_cast<int32_t>(i);
                // An encoder outputs every position anyway; marking them all
                // keeps llama.cpp from overriding the flags with a warning.
                const bool output = !memory || std::binary_search(outputs.begin(), outputs.end(), pos);
                entries.push_back(entry{ prompt.tokens[i], pos, static_cast<llama_seq_id>(k), output, k });
            }
        }

        auto capture = [&](const entry & item, int32_t batch_index) -> bool {
            read_plan & plan = plans[item.plan];
            if (!std::binary_search(plan.outputs.begin(), plan.outputs.end(), item.pos)) {
                return true;
            }
            const retro_decision_prompt & prompt = prompts[first + item.plan];
            std::vector<float> & slot = plan.captured[item.pos];
            if (readout == RETRO_DECIDE_LABEL_LOGITS) {
                const float * logits = llama_get_logits_ith(ctx, batch_index);
                if (!logits) {
                    set_error("the decode produced no logits for a decision label readout");
                    return false;
                }
                for (size_t r = 0; r < prompt.n_reads; ++r) {
                    slot.push_back(logits[prompt.reads[r]]);
                }
                return true;
            }
            const float * embd = llama_get_embeddings_ith(ctx, batch_index);
            if (!embd) {
                set_error("the decode produced no embeddings at a decision read position");
                return false;
            }
            if (readout == RETRO_DECIDE_MARKER_COLUMN) {
                slot.push_back(embd[prompt.column]);
            } else {
                slot.assign(embd, embd + row_width);
            }
            return true;
        };

        // Encoders take the whole pack at once, a context with memory takes
        // it in batches, the shared prefix first and alone.
        size_t offset = 0;
        while (offset < entries.size()) {
            const size_t end = offset < n_prefix_entries
                    ? std::min(n_prefix_entries, offset + n_batch)
                    : std::min(entries.size(), offset + n_batch);
            const auto n_tokens = static_cast<int32_t>(end - offset);
            gen_detail::batch_guard guard(n_tokens);
            llama_batch & batch = guard.batch;
            batch.n_tokens = n_tokens;
            for (int32_t i = 0; i < n_tokens; ++i) {
                const entry & item = entries[offset + static_cast<size_t>(i)];
                batch.token[i] = item.token;
                batch.pos[i] = item.pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = item.seq;
                batch.logits[i] = item.output ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                set_error("llama_decode failed while evaluating decision prompts");
                return -1;
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                const entry & item = entries[offset + static_cast<size_t>(i)];
                if (item.output && !capture(item, i)) {
                    return -1;
                }
            }
            if (end == n_prefix_entries && prefix > 0) {
                for (size_t k = 1; k < n_pack; ++k) {
                    llama_memory_seq_cp(memory, 0, static_cast<llama_seq_id>(k), -1, -1);
                }
            }
            offset = end;
        }

        for (size_t k = 0; k < n_pack; ++k) {
            const retro_decision_prompt & prompt = prompts[first + k];
            const read_plan & plan = plans[k];
            if (readout == RETRO_DECIDE_LABEL_LOGITS && prompt.label_groups) {
                // The largest logit of each group of labels, as llama-server
                // reads the labels of a d1 model.
                const std::vector<float> & logits = plan.captured.at(static_cast<int32_t>(prompt.n_tokens - 1));
                size_t start = 0;
                for (size_t g = 0; g < prompt.n_label_groups; ++g) {
                    const size_t end = start + static_cast<size_t>(prompt.label_groups[g]);
                    out_scores[written++] = *std::max_element(logits.begin() + start, logits.begin() + end);
                    start = end;
                }
                continue;
            }
            for (size_t r = 0; r < prompt.n_reads; ++r) {
                float score = 0.0f;
                if (readout == RETRO_DECIDE_LABEL_LOGITS) {
                    score = plan.captured.at(static_cast<int32_t>(prompt.n_tokens - 1))[r];
                } else if (readout == RETRO_DECIDE_MARKER_COLUMN) {
                    score = plan.captured.at(prompt.reads[r])[0];
                } else {
                    // Summed in order in f32, as llama-server does.
                    const std::vector<float> & query = plan.captured.at(prompt.pointer);
                    const std::vector<float> & key = plan.captured.at(prompt.reads[r]);
                    const size_t half = row_width / 2;
                    float dot = 0.0f;
                    for (size_t i = 0; i < half; ++i) {
                        dot += query[i] * key[half + i];
                    }
                    // Widening: `half` is a model width, far below 2^24.
                    score = dot / std::sqrt(static_cast<float>(half));
                }
                out_scores[written++] = score;
            }
        }

        if (memory) {
            llama_memory_clear(memory, true);
        }
        first = last;
    }
    return 0;
}

extern "C" int retro_read_decision_info(const char * model_path, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        if (is_blank(model_path)) {
            set_error("model_path is required");
            return -1;
        }
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        ensure_backend_initialized();
        model_ptr model = load_model_metadata_only(model_path);
        if (!model) {
            set_error(std::string("failed to load model GGUF: ") + model_path);
            return -1;
        }
        return copy_string_out(describe_decision_model(model.get()), buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_decision_info(retro_trainer * trainer, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        return copy_string_out(describe_decision_model(state->model.get()), buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_render_named_template(
        retro_trainer * trainer,
        const char * name,
        const char * context_json,
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
        return render_named_template(state->chat, name, context_json, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_set_named_template(retro_trainer * trainer, const char * name, const char * source) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        return set_named_template(state->chat, name, source);
    });
}

extern "C" int retro_trainer_decide(
        retro_trainer * trainer,
        int32_t readout,
        const retro_decision_prompt * prompts,
        size_t n_prompts,
        size_t n_pack_max,
        float * out_scores,
        size_t n_scores_max) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (state->training_failed) {
            set_error("a previous training step failed on the backend; create a new trainer");
            return -1;
        }
        // The training context reads its loss from the embeddings output
        // whatever this flag says, and evaluates nothing else: the flag is
        // restored so a later forward pass finds the context as it was built.
        llama_context * ctx = state->ctx.get();
        struct decide_guard {
            llama_context * ctx;
            ~decide_guard() {
                if (llama_memory_t memory = llama_get_memory(ctx)) {
                    llama_memory_clear(memory, true);
                }
                llama_set_embeddings(ctx, false);
            }
        } cleanup { ctx };
        if (llama_memory_t memory = llama_get_memory(ctx)) {
            llama_memory_clear(memory, true);
        }
        // The optimizer context is single-sequence unless the run widened it,
        // and its cells are not partitioned.
        return decide_prompts(state->model.get(), ctx, readout, prompts, n_prompts,
                              n_pack_max, /*share_prefix =*/ true, out_scores, n_scores_max);
    });
}

extern "C" int retro_template_render_context(
        const char * template_src,
        const char * context_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        return render_template_source_context(template_src, context_json, buffer, n_buffer, out_n_bytes);
    });
}

}  // namespace retro
