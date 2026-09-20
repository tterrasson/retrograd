#include "retro_runtime.hpp"

#include "ggml-retro-quant.h"

#include <cmath>
#include <cstddef>
#include <sstream>

namespace {
// Applies the versioned runtime policy before any backend context exists.
// Returns false with an error when the requested policy cannot be honored.
bool apply_runtime_config(const retro_runtime_config * cfg) {
    if (!cfg) {
        return true;
    }
    // struct_size is the evolution key: anything shorter than the prefix a field
    // lives in was compiled before that field existed and must not be read.
    const uint32_t need = offsetof(retro_runtime_config, rir_mode) + sizeof(int32_t);
    if (cfg->struct_size < need) {
        retro::set_error("retro_runtime_config.struct_size is too small for this build");
        return false;
    }
    if (cfg->rir_mode < RETRO_RIR_MODE_OFF || cfg->rir_mode > RETRO_RIR_MODE_REQUIRE) {
        retro::set_error("retro_runtime_config.rir_mode is out of range");
        return false;
    }
    if (ggml_rir_set_mode((ggml_rir_mode) cfg->rir_mode) != 0) {
        retro::set_error("the RIR policy is already in force with a different value; "
                         "it is fixed before the first backend context is created and "
                         "cannot be changed for a second trainer in the same process");
        return false;
    }
    return true;
}

retro_trainer * trainer_new_impl(
        const char * model_path,
        const retro_train_config * train_config,
        const retro_runtime_config * runtime_config) {
    try {
        retro::clear_error();
        if (!apply_runtime_config(runtime_config)) {
            return nullptr;
        }
        if (retro::is_blank(model_path)) {
            retro::set_error("model_path is required");
            return nullptr;
        }

        retro_train_config config = train_config ? *train_config : retro::default_train_config();
        if (!retro::validate_train_config(config)) {
            return nullptr;
        }
        retro::configure_runtime_logging(config.verbose);

        std::unique_ptr<retro::trainer_state> state(new retro::trainer_state());
        // Before load_model_and_context: that is what creates the backend
        // contexts, so anything counted after this point belongs to this
        // trainer.
        state->rir_baseline = ggml_rir_counters_snapshot();
        state->model_path = model_path;
        state->train_config = config;
        if (!retro::load_model_and_context(*state)) {
            return nullptr;
        }
        return reinterpret_cast<retro_trainer *>(state.release());
    } catch (const std::exception & err) {
        retro::set_error(err.what());
        return nullptr;
    } catch (...) {
        retro::set_error("unknown C++ exception");
        return nullptr;
    }
}
} // namespace

extern "C" retro_trainer * retro_trainer_new(
        const char * model_path,
        const retro_train_config * train_config) {
    return trainer_new_impl(model_path, train_config, nullptr);
}

extern "C" retro_trainer * retro_trainer_new_ex(
        const char * model_path,
        const retro_train_config * train_config,
        const retro_runtime_config * runtime_config) {
    return trainer_new_impl(model_path, train_config, runtime_config);
}

extern "C" int retro_runtime_config_apply(const retro_runtime_config * config) {
    return retro::boundary([&]() -> int {
        retro::clear_error();
        if (!config) {
            retro::set_error("retro_runtime_config_apply requires a non-null config");
            return -1;
        }
        return apply_runtime_config(config) ? 0 : -1;
    });
}

extern "C" int retro_runtime_config_effective(retro_runtime_config * out, bool * out_latched) {
    return retro::boundary([&]() -> int {
        if (!out) {
            retro::set_error("retro_runtime_config_effective requires a non-null out");
            return -1;
        }
        if (out->struct_size != sizeof(retro_runtime_config)) {
            retro::set_error("retro_runtime_config.struct_size does not match this build");
            return -1;
        }
        // Reading the mode is what latches it, so ask about the latch first,
        // otherwise this query would itself freeze the policy it reports.
        const bool latched = ggml_rir_mode_is_latched();
        out->rir_mode = (int32_t) ggml_rir_get_mode();
        if (out_latched) {
            *out_latched = latched;
        }
        return 0;
    });
}

extern "C" int retro_trainer_create_lora(
        retro_trainer * trainer,
        const retro_lora_config * lora_config) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!lora_config) {
            retro::set_error("lora_config is required");
            return -1;
        }
        if (lora_config->rank == 0) {
            retro::set_error("LoRA rank must be greater than zero");
            return -1;
        }
        if (lora_config->alpha <= 0.0f) {
            retro::set_error("LoRA alpha must be greater than zero");
            return -1;
        }
        if (lora_config->dropout != 0.0f) {
            retro::set_error("LoRA dropout is not implemented");
            return -1;
        }
        if (lora_config->dtype != RETRO_LORA_DTYPE_F32
                && lora_config->dtype != RETRO_LORA_DTYPE_F16) {
            retro::set_error("LoRA dtype must be F32 or F16");
            return -1;
        }
        if (lora_config->n_target_patterns != 0 && !lora_config->target_patterns) {
            retro::set_error("target_patterns is required when n_target_patterns is non-zero");
            return -1;
        }

        std::vector<std::string> patterns;
        patterns.reserve(lora_config->n_target_patterns);
        for (size_t i = 0; i < lora_config->n_target_patterns; ++i) {
            const char * pattern = lora_config->target_patterns[i];
            if (retro::is_blank(pattern)) {
                retro::set_error("LoRA target patterns must not be empty");
                return -1;
            }
            patterns.emplace_back(pattern);
        }

        if (state->opt_created) {
            retro::set_error("cannot replace LoRA adapter after optimizer initialization");
            return -1;
        }

        state->adapter.reset();
        state->invalidate_report_caches();
        state->forget_generation_kv();
        state->has_lora = true;
        state->loaded_lora = false;
        state->lora_promoted = false;
        state->lora_rank = lora_config->rank;
        state->lora_alpha = lora_config->alpha;
        state->lora_dtype = lora_config->dtype;
        state->target_patterns = std::move(patterns);
        if (!retro::create_trainable_lora_adapter(*state, lora_config->seed)) {
            state->has_lora = false;
            return -1;
        }
        return 0;
    });
}

extern "C" int retro_trainer_set_optimizer_assignment(
        retro_trainer * trainer,
        const char * const * names,
        const int32_t * optimizers,
        size_t n_rows) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (n_rows != 0 && (!names || !optimizers)) {
            retro::set_error("names and optimizers are required when n_rows is non-zero");
            return -1;
        }
        if (state->opt_created) {
            retro::set_error(
                    "cannot change the optimizer assignment after optimizer initialization");
            return -1;
        }

        // Validated into a local first, so a rejected call leaves the
        // previous assignment intact.
        std::vector<std::pair<std::string, int32_t>> resolved;
        resolved.reserve(n_rows);
        for (size_t i = 0; i < n_rows; ++i) {
            const char * name = names[i];
            if (retro::is_blank(name)) {
                retro::set_error("assigned parameter names must not be empty");
                return -1;
            }
            const int32_t optimizer = optimizers[i];
            if (optimizer != RETRO_OPTIMIZER_ADAMW && optimizer != RETRO_OPTIMIZER_SGD) {
                retro::set_error(
                        std::string("parameter '") + name + "' is assigned optimizer "
                        + std::to_string(optimizer)
                        + ", which this build cannot build an update step for");
                return -1;
            }
            const auto duplicate = std::find_if(
                    resolved.begin(),
                    resolved.end(),
                    [&](const std::pair<std::string, int32_t> & row) { return row.first == name; });
            if (duplicate != resolved.end()) {
                retro::set_error(
                        std::string("duplicate optimizer assignment for '") + name + "'");
                return -1;
            }
            resolved.emplace_back(name, optimizer);
        }
        state->optimizer_assignment = std::move(resolved);
        state->invalidate_report_caches();
        return 0;
    });
}

extern "C" int retro_trainer_set_trainable_base(
        retro_trainer * trainer,
        const char * const * names,
        size_t n_names) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (n_names != 0 && !names) {
            retro::set_error("names is required when n_names is non-zero");
            return -1;
        }
        if (state->opt_created) {
            retro::set_error("cannot change the trainable set after optimizer initialization");
            return -1;
        }
        if (!retro::trains_base_weights(*state) && n_names != 0) {
            retro::set_error(
                    "this run's policy trains no base tensor, so it takes no trainable set");
            return -1;
        }

        if (retro::trains_base_weights(*state) && n_names == 0) {
            retro::set_error("a base-weight policy requires a non-empty trainable set");
            return -1;
        }

        // Validated and copied in one pass, into a local, so a rejected call
        // leaves the previous set intact rather than half-replaced.
        std::vector<std::string> resolved;
        resolved.reserve(n_names);
        for (size_t i = 0; i < n_names; ++i) {
            const char * name = names[i];
            if (retro::is_blank(name)) {
                retro::set_error("trainable tensor names must not be empty");
                return -1;
            }
            // The name has to exist in *this* model: the resolver read the
            // GGUF's metadata, and a tensor the loader renamed or never
            // allocated would otherwise be silently absent from the update.
            if (state->model->tensors_by_name.end()
                    == std::find_if(
                            state->model->tensors_by_name.begin(),
                            state->model->tensors_by_name.end(),
                            [&](const std::pair<std::string, ggml_tensor *> & item) {
                                return item.first == name;
                            })) {
                retro::set_error(
                        std::string("the model declares no tensor named '") + name + "'");
                return -1;
            }
            if (std::find(resolved.begin(), resolved.end(), name) != resolved.end()) {
                retro::set_error(std::string("duplicate trainable tensor: ") + name);
                return -1;
            }
            resolved.emplace_back(name);
        }
        state->trainable_base = std::move(resolved);
        state->trainable_base_set = true;
        state->invalidate_report_caches();
        return 0;
    });
}

extern "C" int retro_trainer_load_lora(
        retro_trainer * trainer,
        const char * adapter_path) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(adapter_path)) {
            retro::set_error("adapter_path is required");
            return -1;
        }
        // Refuse before touching the context: once the adapter is applied, an
        // error return would leave the context pointing at a freed adapter.
        if (state->opt_created) {
            retro::set_error("cannot replace LoRA adapter after optimizer initialization");
            return -1;
        }

        retro::lora_adapter_ptr adapter(llama_adapter_lora_init(state->model.get(), adapter_path));
        if (!adapter) {
            retro::set_error("failed to load LoRA adapter: " + std::string(adapter_path));
            return -1;
        }
        if (!retro::validate_lora_tensor_pairs(*adapter)) {
            return -1;
        }

        llama_adapter_lora * adapters[] = { adapter.get() };
        float scales[] = { 1.0f };
        const int32_t status = llama_set_adapters_lora(state->ctx.get(), adapters, 1, scales);
        if (status != 0) {
            retro::set_error("failed to apply LoRA adapter to context");
            return -1;
        }
        if (state->generation_ctx
                && llama_set_adapters_lora(
                        state->generation_ctx.get(), adapters, 1, scales) != 0) {
            llama_set_adapters_lora(state->ctx.get(), nullptr, 0, nullptr);
            retro::set_error("failed to apply LoRA adapter to generation context");
            return -1;
        }

        state->adapter = std::move(adapter);
        state->has_lora = true;
        state->loaded_lora = true;
        state->lora_promoted = false;
        state->lora_rank = 0;
        state->lora_alpha = state->adapter->alpha;
        if (!state->adapter->ab_map.empty() && state->adapter->ab_map.begin()->second.a) {
            state->lora_dtype = state->adapter->ab_map.begin()->second.a->type == GGML_TYPE_F16
                    ? RETRO_LORA_DTYPE_F16 : RETRO_LORA_DTYPE_F32;
        }
        state->target_patterns.clear();
        state->invalidate_report_caches();
        state->forget_generation_kv();
        return 0;
    });
}

namespace {

int tokenize_into(
        retro_trainer * trainer,
        const char * text,
        bool add_special,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->model) {
            retro::set_error("model is not loaded");
            return -1;
        }
        if (retro::is_blank(text)) {
            retro::set_error("text is required");
            return -1;
        }
        if (!out_n_tokens) {
            retro::set_error("out_n_tokens is required");
            return -1;
        }
        const size_t text_size = std::strlen(text);
        if (text_size > static_cast<size_t>(INT32_MAX)) {
            retro::set_error("text is too large to tokenize");
            return -1;
        }

        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const int32_t text_len = static_cast<int32_t>(text_size);
        const int32_t cap = n_tokens_max > static_cast<size_t>(INT32_MAX)
                ? INT32_MAX
                : static_cast<int32_t>(n_tokens_max);
        llama_token * token_buf = reinterpret_cast<llama_token *>(tokens);
        const int32_t result = llama_tokenize(
                vocab,
                text,
                text_len,
                token_buf,
                cap,
                add_special,
                true);

        if (result == INT32_MIN) {
            retro::set_error("tokenization overflow");
            return -1;
        }
        if (result < 0) {
            *out_n_tokens = static_cast<size_t>(-result);
            return tokens == nullptr || n_tokens_max == 0 ? 0 : -2;
        }

        *out_n_tokens = static_cast<size_t>(result);
        return 0;
    });
}

}  // namespace

extern "C" int retro_trainer_tokenize_text(
        retro_trainer * trainer,
        const char * text,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return tokenize_into(trainer, text, /*add_special=*/true, tokens, n_tokens_max, out_n_tokens);
}

extern "C" int retro_trainer_tokenize_fragment(
        retro_trainer * trainer,
        const char * text,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return tokenize_into(trainer, text, /*add_special=*/false, tokens, n_tokens_max, out_n_tokens);
}

extern "C" int retro_trainer_eos_token(retro_trainer * trainer, int32_t * out_token) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_token) {
            if (state) retro::set_error("out_token is required");
            return -1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        *out_token = llama_vocab_eos(vocab);
        return 0;
    });
}

extern "C" int retro_trainer_vocab_size(retro_trainer * trainer, uint32_t * out_n_vocab) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_n_vocab) {
            if (state) retro::set_error("out_n_vocab is required");
            return -1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        const int32_t size = llama_vocab_n_tokens(vocab);
        if (size < 0) {
            retro::set_error("model reported a negative vocabulary size");
            return -1;
        }
        *out_n_vocab = static_cast<uint32_t>(size);
        return 0;
    });
}

extern "C" int retro_trainer_is_eog_token(retro_trainer * trainer, int32_t token, bool * out_is_eog) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_is_eog) {
            if (state) retro::set_error("out_is_eog is required");
            return -1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
            retro::set_error("token is outside the model vocabulary");
            return -1;
        }
        *out_is_eog = llama_vocab_is_eog(vocab, token);
        return 0;
    });
}

extern "C" int retro_trainer_context_size(retro_trainer * trainer, uint32_t * out_n_ctx) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_n_ctx) {
            if (state) retro::set_error("out_n_ctx is required");
            return -1;
        }
        *out_n_ctx = llama_n_ctx(state->ctx.get());
        return 0;
    });
}

extern "C" int retro_trainer_format_chat(
        retro_trainer * trainer,
        const char * const * roles,
        const char * const * contents,
        size_t n_messages,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_n_bytes) {
            if (state) retro::set_error("out_n_bytes is required");
            return -1;
        }
        if (n_messages > 0 && (!roles || !contents)) {
            retro::set_error("roles and contents are required");
            return -1;
        }
        for (size_t i = 0; i < n_messages; ++i) {
            if (retro::is_blank(roles[i]) || contents[i] == nullptr) {
                retro::set_error("chat roles and contents must not be empty");
                return -1;
            }
        }
        return retro::render_chat_template(
                *state, roles, contents, n_messages, add_assistant,
                buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_format_chat_messages(
        retro_trainer * trainer,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_n_bytes) {
            if (state) retro::set_error("out_n_bytes is required");
            return -1;
        }
        if (retro::is_blank(messages_json)) {
            retro::set_error("chat messages JSON is required");
            return -1;
        }
        return retro::render_chat_messages(
                *state, messages_json, tools_json, add_assistant,
                buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_set_chat_template_variables(
        retro_trainer * trainer,
        const char * variables_json) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        return retro::set_chat_template_variables(*state, variables_json);
    });
}

extern "C" int retro_trainer_chat_template_supports_tools(
        retro_trainer * trainer,
        bool * out_supports) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_supports) {
            if (state) retro::set_error("out_supports is required");
            return -1;
        }
        return retro::chat_template_supports_tools(*state, out_supports);
    });
}

extern "C" int retro_trainer_tool_call_parser(
        retro_trainer * trainer,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_n_bytes) {
            if (state) retro::set_error("out_n_bytes is required");
            return -1;
        }
        return retro::build_tool_call_parser(*state, tools_json, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_chat_template_render(
        const char * template_src,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        if (!out_n_bytes) {
            retro::set_error("out_n_bytes is required");
            return -1;
        }
        if (retro::is_blank(template_src)) {
            retro::set_error("chat template source is required");
            return -1;
        }
        if (retro::is_blank(messages_json)) {
            retro::set_error("chat messages JSON is required");
            return -1;
        }
        return retro::render_chat_template_source(
                template_src, messages_json, tools_json, add_assistant,
                buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_chat_template_tool_call_parser(
        const char * template_src,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        if (!out_n_bytes) {
            retro::set_error("out_n_bytes is required");
            return -1;
        }
        if (retro::is_blank(template_src)) {
            retro::set_error("chat template source is required");
            return -1;
        }
        return retro::build_tool_call_parser_from_source(
                template_src, tools_json, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_chat_parse_assistant(
        const char * parser_blob,
        size_t n_parser_blob,
        const char * text,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        if (!out_n_bytes) {
            retro::set_error("out_n_bytes is required");
            return -1;
        }
        if (!parser_blob || n_parser_blob == 0) {
            retro::set_error("tool-call parser blob is required");
            return -1;
        }
        if (!text) {
            retro::set_error("assistant text is required");
            return -1;
        }
        return retro::parse_assistant_output(
                parser_blob, n_parser_blob, text, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_describe_lora(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (state->lora_description_cache.empty()) {
            state->lora_description_cache = retro::describe_lora(*state);
        }
        return retro::copy_string_out(
                state->lora_description_cache, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_lora_candidate_targets(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->model) {
            retro::set_error("model is not loaded");
            return -1;
        }
        if (state->lora_candidate_targets_cache.empty()) {
            std::ostringstream out;
            for (const std::string & name : retro::lora_candidate_targets(*state->model)) {
                out << name << "\n";
            }
            state->lora_candidate_targets_cache = out.str();
        }
        return retro::copy_string_out(
                state->lora_candidate_targets_cache, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_backend_report(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->model) {
            retro::set_error("model is not loaded");
            return -1;
        }
        // Memory measurements change during optimization, so invalidate the cache
        // when the sample count changes rather than relying on structural changes.
        llama_opt_memory measured {};
        llama_opt_get_memory(state->ctx.get(), &measured);
        if (state->backend_report_memory_samples != measured.n_samples) {
            state->backend_report_memory_samples = measured.n_samples;
            state->backend_report_cache.clear();
        }
        if (state->backend_report_cache.empty()) {
            state->backend_report_cache = retro::backend_report(*state);
        }
        return retro::copy_string_out(
                state->backend_report_cache, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_capability_report(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->model) {
            retro::set_error("model is not loaded");
            return -1;
        }
        if (state->capability_report_cache.empty()) {
            state->capability_report_cache = retro::capability_report(*state);
        }
        // Rebuild the RIR section because its counters change on every graph.
        const std::string report =
                state->capability_report_cache + retro::rir_capability_section(*state);
        return retro::copy_string_out(report, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_model_capabilities(
        retro_trainer * trainer,
        retro_model_capabilities * out_capabilities) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_capabilities) {
            retro::set_error("out_capabilities is required");
            return -1;
        }
        if (out_capabilities->struct_size != sizeof(retro_model_capabilities)) {
            retro::set_error("retro_model_capabilities.struct_size does not match this build");
            return -1;
        }
        out_capabilities->shared_prefix_packed_training =
                retro::shared_prefix_packed_training(*state);
        out_capabilities->fused_sparse_cross_entropy = state->cap_fused_sparse_ce;
        out_capabilities->differentiable_flash_attention = state->cap_flash_attn_back;
        return 0;
    });
}

extern "C" int retro_probe_op_run(
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
        size_t dst_len) {
    return retro::probe_op_run_impl(op, use_gpu, ne_src0, src0, ne_src1, src1,
            ne_src2, src2, param0, param1, dst, dst_len);
}

extern "C" int retro_probe_op_run_ex(
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
        retro_kernel_run_info * info) {
    return retro::probe_op_run_ex_impl(op, use_gpu, ne_src0, src0, ne_src1, src1,
            ne_src2, src2, param0, param1, dst, dst_len, implementation, info);
}

extern "C" int retro_rir_counters_get(retro_rir_counters * out) {
    return retro::rir_counters_impl(out);
}

extern "C" int retro_rir_variant_report(char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return retro::rir_variant_report_impl(buffer, n_buffer, out_n_bytes);
}

extern "C" int retro_rir_census_report(char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return retro::rir_census_report_impl(buffer, n_buffer, out_n_bytes);
}

extern "C" uint32_t retro_rir_selection_selftest(void) {
    return ggml_rir_selftest_selection();
}

namespace retro {

bool is_retro_dequant_type(enum ggml_type type) {
    switch (type) {
#define GGML_RETRO_CASE(TYPE, BLK, NL, NAME, VKNAME) case TYPE: return true;
        GGML_RETRO_DEQUANT_TYPES(GGML_RETRO_CASE)
#undef GGML_RETRO_CASE
        default: return false;
    }
}

bool is_retro_out_prod_type(enum ggml_type type) {
    switch (type) {
#define GGML_RETRO_CASE(TYPE, BLK, NL, NAME, VKNAME) case TYPE: return true;
        GGML_RETRO_OUT_PROD_TYPES(GGML_RETRO_CASE)
#undef GGML_RETRO_CASE
        default: return false;
    }
}

} // namespace retro

extern "C" size_t retro_dequant_types(int32_t * out_types, size_t cap) {
    static const int32_t types[] = {
#define GGML_RETRO_TYPE_ID(TYPE, BLK, NL, NAME, VKNAME) (int32_t) TYPE,
        GGML_RETRO_DEQUANT_TYPES(GGML_RETRO_TYPE_ID)
#undef GGML_RETRO_TYPE_ID
    };
    static const size_t n = sizeof(types)/sizeof(types[0]);
    static_assert(n == GGML_RETRO_DEQUANT_TYPE_COUNT,
            "GGML_RETRO_DEQUANT_TYPE_COUNT is out of sync with the table");
    if (out_types && cap >= n) {
        std::copy(types, types + n, out_types);
    }
    return n;
}

extern "C" int retro_quant_traits(
        int32_t type, int64_t * out_block_elements, size_t * out_block_bytes) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return -1;
    }
    const ggml_type_traits * t = ggml_get_type_traits((ggml_type) type);
    if (out_block_elements) {
        *out_block_elements = t->blck_size;
    }
    if (out_block_bytes) {
        *out_block_bytes = t->type_size;
    }
    return 0;
}

extern "C" int retro_quant_roundtrip(
        int32_t         type,
        int64_t         n,
        const float   * src,
        uint8_t       * out_bytes,
        size_t          out_bytes_cap,
        float         * out_dequant) {
    if (type < 0 || type >= GGML_TYPE_COUNT || !src || !out_bytes || !out_dequant) {
        return -1;
    }
    const ggml_type_traits * t = ggml_get_type_traits((ggml_type) type);
    if (n <= 0 || t->blck_size <= 0 || n % t->blck_size != 0 || !t->to_float) {
        return -1;
    }
    const size_t need = (size_t) (n / t->blck_size) * t->type_size;
    if (out_bytes_cap < need) {
        return -1;
    }
    // Return bytes produced by the reference quantizer so parity tests compare
    // ggml's decoder with the other decoder, not two independent quantizations.
    if (ggml_quantize_chunk((ggml_type) type, src, out_bytes, 0, 1, n, nullptr) != need) {
        return -1;
    }
    t->to_float(out_bytes, out_dequant, n);
    return 0;
}

extern "C" const char * retro_ggml_type_name(int32_t type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return "?";
    }
    return ggml_type_name((ggml_type) type);
}

extern "C" int retro_probe_token_logprob(
        const float * logits,
        size_t n_vocab,
        int32_t token,
        bool vectorized,
        float * out_logprob) {
    return retro::probe_token_logprob_impl(
            logits, n_vocab, token, vectorized, out_logprob);
}

extern "C" int retro_probe_duty_cycle(
        float fraction,
        const retro_duty_cycle_event * events,
        size_t n_events,
        retro_duty_cycle_probe * out_probe) {
    return retro::duty_cycle_probe_impl(fraction, events, n_events, out_probe);
}

extern "C" int retro_fused_sparse_ce_probe(
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
        float         * out_grad_h_fused) {
    return retro::fused_sparse_ce_probe_impl(
            n_embd, n_tokens, n_vocab, n_topk, n_tiles, seq_chunk, offload_h, w_type, use_gpu, h, w, targets, weights,
            bias, grad_loss,
            out_loss_full, out_loss_fused, out_grad_h_full, out_grad_h_fused);
}

extern "C" int retro_backend_list(
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::backend_list_impl(buffer, n_buffer, out_n_bytes);
}

extern "C" int retro_gpu_runtime_probe(void) {
    return retro::gpu_runtime_probe_impl();
}

extern "C" int retro_device_memory(
        size_t * out_free,
        size_t * out_total) {
    return retro::device_memory_impl(out_free, out_total);
}

extern "C" int retro_transfer_probe(
        size_t bytes,
        uint32_t iterations,
        retro_transfer_rates * out_rates) {
    return retro::transfer_probe_impl(bytes, iterations, out_rates);
}

extern "C" int retro_trainer_train_preflight(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::train_preflight_impl(trainer, buffer, n_buffer, out_n_bytes);
}

extern "C" int retro_trainer_preflight_summary(
        retro_trainer * trainer,
        retro_preflight_summary * out_summary) {
    return retro::preflight_summary_impl(trainer, out_summary);
}

namespace {

// Every resident generation prefix was decoded under the weights of the moment,
// so a step that moves them makes those cells describe a model that no longer
// exists. Forgetting is the whole invalidation: the cells stay where they are
// and the next call simply decodes over them.
// Called at the ABI boundary rather than inside each `*_impl`, so the set of
// weight mutations is one greppable block instead of a convention every future
// training entry point has to be told about. No `checked()`: this must not set
// an error over the one the call itself is reporting.
void forget_generation_kv(retro_trainer * trainer) {
    if (trainer) {
        reinterpret_cast<retro::trainer_state *>(trainer)->forget_generation_kv();
    }
}

}  // namespace

extern "C" int retro_trainer_train_tokens(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        retro_train_metrics * out_metrics) {
    const int status = retro::train_tokens_impl(trainer, tokens, n_tokens, out_metrics);
    forget_generation_kv(trainer);
    return status;
}

extern "C" int retro_trainer_train_sft(
        retro_trainer * trainer,
        const retro_sft_dataset * train,
        const retro_sft_dataset * eval,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    const int status = retro::train_sft_impl(
            trainer, train, eval, out_metrics, progress_callback, progress_user_data);
    forget_generation_kv(trainer);
    return status;
}

extern "C" int retro_trainer_generate_batch(
        retro_trainer * trainer,
        const int32_t * prompt_tokens,
        size_t n_prompt,
        const retro_sampling_params * sampling,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens) {
    return retro::generate_batch_impl(trainer, prompt_tokens, n_prompt, sampling,
            n_sequences, out_tokens, out_logprobs, n_out_max, out_n_tokens);
}

extern "C" int retro_trainer_generate_continuous_batch(
        retro_trainer * trainer,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens) {
    return retro::generate_continuous_batch_impl(
            trainer, sequences, n_sequences, out_tokens, out_logprobs, n_out_max, out_n_tokens);
}

extern "C" int retro_trainer_score_tokens(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_logprobs) {
    return retro::score_tokens_impl(trainer, tokens, n_tokens, out_logprobs);
}

extern "C" int retro_trainer_eval_sft(
        retro_trainer * trainer,
        const retro_sft_dataset * data,
        retro_eval_metrics * out_metrics) {
    return retro::eval_sft_impl(trainer, data, out_metrics);
}

extern "C" int retro_trainer_score_token_suffix(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs) {
    return retro::score_token_suffix_impl(
            trainer, tokens, n_tokens, n_prompt, out_logprobs);
}

extern "C" int retro_trainer_top_logprobs_suffix(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        size_t k,
        int32_t * out_ids,
        float * out_logprobs,
        size_t n_out_max) {
    return retro::top_logprobs_suffix_impl(
            trainer, tokens, n_tokens, n_prompt, k, out_ids, out_logprobs, n_out_max);
}

extern "C" int retro_trainer_score_token_suffix_batch(
        retro_trainer * trainer,
        const retro_token_suffix_sequence * sequences,
        size_t n_sequences,
        float * out_logprobs,
        size_t out_stride,
        size_t * out_n_logprobs) {
    return retro::score_token_suffix_batch_impl(
            trainer, sequences, n_sequences, out_logprobs, out_stride, out_n_logprobs);
}

extern "C" int retro_trainer_set_lora_enabled(retro_trainer * trainer, bool enabled) {
    const int status = retro::set_lora_enabled_impl(trainer, enabled);
    forget_generation_kv(trainer);
    return status;
}

extern "C" int retro_trainer_hidden_size(retro_trainer * trainer, uint32_t * out_n_embd) {
    return retro::hidden_size_impl(trainer, out_n_embd);
}

extern "C" int retro_trainer_hidden_states(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_features,
        size_t n_features_max) {
    return retro::hidden_states_impl(trainer, tokens, n_tokens, out_features, n_features_max);
}

extern "C" int retro_trainer_score_token_suffix_and_hidden_states(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        size_t n_prompt,
        float * out_logprobs,
        float * out_features,
        size_t n_features_max) {
    return retro::score_token_suffix_and_hidden_states_impl(
            trainer, tokens, n_tokens, n_prompt,
            out_logprobs, out_features, n_features_max);
}

extern "C" int retro_trainer_detokenize(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        bool unparse_special,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::detokenize_impl(trainer, tokens, n_tokens, unparse_special, buffer, n_buffer, out_n_bytes);
}

extern "C" int retro_trainer_train_weighted(
        retro_trainer * trainer,
        const retro_weighted_dataset * data,
        uint64_t scheduler_total_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    const int status = retro::train_weighted_impl(trainer, data, scheduler_total_steps,
            out_metrics, progress_callback, progress_user_data);
    forget_generation_kv(trainer);
    return status;
}

extern "C" int retro_trainer_train_packed_sequences(
        retro_trainer * trainer,
        const retro_packed_sequence_batch * data,
        uint64_t scheduler_total_steps,
        uint32_t accumulation_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    const int status = retro::train_packed_sequences_impl(trainer, data, scheduler_total_steps,
            accumulation_steps, out_metrics,
            progress_callback, progress_user_data);
    forget_generation_kv(trainer);
    return status;
}

extern "C" int retro_trainer_optimizer_timing(
        retro_trainer * trainer,
        retro_optimizer_timing * out_timing) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_timing) {
            if (state) {
                retro::set_error("out_timing is required");
            }
            return -1;
        }
        llama_opt_timing timing {};
        llama_opt_get_timing(state->ctx.get(), &timing);
        out_timing->graph_build_seconds = timing.graph_build_seconds;
        out_timing->allocation_seconds = timing.allocation_seconds;
        out_timing->execution_seconds = timing.execution_seconds;
        return 0;
    });
}

extern "C" int retro_read_model_info(
        const char * model_path,
        int32_t device,
        retro_model_info * out_info) {
    return retro::read_model_info_impl(model_path, device, out_info);
}

extern "C" int retro_trainer_memory_report(
        retro_trainer * trainer,
        retro_memory_report * out_report) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_report) {
            if (state) {
                retro::set_error("out_report is required");
            }
            return -1;
        }
        *out_report = retro::memory_totals(*state);
        return 0;
    });
}

extern "C" int retro_trainer_optimizer_memory(
        retro_trainer * trainer,
        retro_optimizer_memory * out_memory) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_memory) {
            if (state) {
                retro::set_error("out_memory is required");
            }
            return -1;
        }
        llama_opt_memory memory {};
        llama_opt_get_memory(state->ctx.get(), &memory);
        out_memory->device_used_bytes      = memory.device_used_bytes;
        out_memory->device_total_bytes     = memory.device_total_bytes;
        out_memory->device_peak_used_bytes = memory.device_peak_used_bytes;
        out_memory->scratch_bytes          = memory.scratch_bytes;
        out_memory->scratch_peak_bytes     = memory.scratch_peak_bytes;
        out_memory->n_samples              = memory.n_samples;
        return 0;
    });
}

extern "C" int retro_trainer_scoring_stats(
        retro_trainer * trainer,
        retro_scoring_stats * out_stats) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_stats) {
            if (state) {
                retro::set_error("out_stats is required");
            }
            return -1;
        }
        *out_stats = state->scoring_stats;
        return 0;
    });
}

extern "C" int retro_trainer_generation_stats(
        retro_trainer * trainer,
        retro_generation_stats * out_stats) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_stats) {
            if (state) {
                retro::set_error("out_stats is required");
            }
            return -1;
        }
        *out_stats = state->generation_stats;
        return 0;
    });
}

extern "C" int retro_trainer_set_max_gpu_duty_cycle(
        retro_trainer * trainer,
        float fraction) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!retro::duty_cycle_limiter::is_valid_fraction(fraction)) {
            // Zero is rejected rather than read as "pause": the run-control
            // pause operation is already the safe way to stop a live run, and a
            // limiter that never repays its debt is a stall, not a pause.
            retro::set_error(
                    "max_gpu_duty_cycle must be finite and in (0, 1]");
            return -1;
        }
        state->duty_cycle.configure(fraction, state->gpu_active);
        // The report names the requested fraction and whether the limiter
        // engaged, so a setting change has to invalidate it like any other.
        state->invalidate_report_caches();
        return 0;
    });
}

extern "C" int retro_trainer_duty_cycle_stats(
        retro_trainer * trainer,
        retro_duty_cycle_stats * out_stats) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state || !out_stats) {
            if (state) {
                retro::set_error("out_stats is required");
            }
            return -1;
        }
        *out_stats = retro::duty_cycle_snapshot(state->duty_cycle);
        return 0;
    });
}

extern "C" int retro_trainer_advance_scheduler_steps(
        retro_trainer * trainer,
        uint64_t steps,
        uint64_t * out_global_step) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_global_step) {
            retro::set_error("out_global_step is required");
            return -1;
        }
        if (steps > UINT64_MAX - state->scheduler_step) {
            retro::set_error("scheduler step count overflows uint64");
            return -1;
        }
        state->scheduler_step += steps;
        *out_global_step = state->scheduler_step;
        return 0;
    });
}

extern "C" int retro_trainer_set_learning_rate(
        retro_trainer * trainer,
        float learning_rate) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
            retro::set_error("learning_rate must be finite and greater than zero");
            return -1;
        }
        // The scheduler reads this base rate before each step and applies its
        // existing warm-up or decay factor.
        state->train_config.learning_rate = learning_rate;
        return 0;
    });
}

extern "C" int retro_trainer_save_lora(
        retro_trainer * trainer,
        const char * adapter_path) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora) {
            retro::set_error("no LoRA adapter exists");
            return -1;
        }
        if (retro::is_blank(adapter_path)) {
            retro::set_error("adapter_path is required");
            return -1;
        }
        if (!retro::save_lora_adapter_gguf(*state, adapter_path)) {
            return -1;
        }
        return 0;
    });
}

extern "C" void retro_trainer_free(retro_trainer * trainer) {
    retro::trainer_state * state = reinterpret_cast<retro::trainer_state *>(trainer);
    delete state;
}

extern "C" const char * retro_last_error(void) {
    if (retro::g_last_error.empty()) {
        return "no runtime error";
    }
    return retro::g_last_error.c_str();
}
