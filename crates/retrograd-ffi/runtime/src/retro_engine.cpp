// The inference engine behind the opaque retro_engine handle: a mapped model,
// one forward-only context and the shared gen_core (retro_gen_core.cpp) - the
// exact same generation code the trainer's retro_trainer_generate_* run, so
// the two agree bit for bit. What the engine deliberately does not have: a
// llama_opt context, a training result cache, trainable flags, a duty-cycle
// limiter. retro_engine_memory_report proves the first three by reporting
// zero optimizer state and zero gradient.
#include "retro_gen_core.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>

namespace retro {

namespace {

struct engine_state {
    std::string model_path;
    retro_engine_config config {};
    model_ptr model;
    // The single forward-only context. Built with the same parameters as the
    // trainer's dedicated generation context, so a geometry that matches a
    // trainer's generation context produces identical outputs.
    context_ptr ctx;
    // Prefix-cache bookkeeping, owned here and handed to the gen_core.
    gen_cache cache;
    // Per-request state and reusable extended batch of the step-wise API
    // (retro_engine_step & co.).
    step_request_table step_requests;
    step_batch_cache step_batch;
    // No duty-cycle limiter, no scoring counters: the engine is unthrottled.
    gen_core_env env;
    lora_adapter_ptr adapter;
    bool lora_enabled = false;
    chat_model_state chat;
    int32_t requested_device = 0;
    bool gpu_active = false;
    int32_t n_gpu_layers = 0;
    // Active-device capabilities, probed at load time; they feed gen_core_env.
    bool cap_device_sampling = false;
    bool cap_device_logprobs = false;
    uint32_t effective_threads = 0;
    std::string backend_name = "CPU";
    std::string backend_registry = "CPU";
    std::string weight_storage = "mapped";
    int32_t effective_kv_dtype = RETRO_KV_DTYPE_F32;
    ggml_rir_counters rir_baseline {};
};

engine_state * checked_engine(retro_engine * engine) {
    if (!engine) {
        set_error("engine is required");
        return nullptr;
    }
    return reinterpret_cast<engine_state *>(engine);
}

ggml_type engine_kv_type(int32_t dtype) {
    switch (dtype) {
        case RETRO_ENGINE_KV_DTYPE_F32:  return GGML_TYPE_F32;
        case RETRO_ENGINE_KV_DTYPE_Q8_0: return GGML_TYPE_Q8_0;
        case RETRO_ENGINE_KV_DTYPE_Q4_0: return GGML_TYPE_Q4_0;
        default:                         return GGML_TYPE_F16;
    }
}

retro_engine_config default_engine_config() {
    retro_engine_config config {};
    config.struct_size = sizeof(retro_engine_config);
    config.n_ctx = 4096;
    config.n_seq_max = 8;
    config.n_batch = 512;
    config.n_ubatch = 512;
    config.kv_dtype = RETRO_ENGINE_KV_DTYPE_F16;
    config.flash_attn = true;
    config.device = RETRO_DEVICE_AUTO;
    config.threads = 0;
    config.verbose = false;
    return config;
}

bool validate_engine_config(const retro_engine_config & config) {
    if (config.n_ctx == 0 || config.n_seq_max == 0) {
        set_error("engine n_ctx and n_seq_max must be greater than zero");
        return false;
    }
    if (config.n_batch == 0 || config.n_ubatch == 0 || config.n_ubatch > config.n_batch) {
        set_error("engine n_ubatch must be in (0, n_batch]");
        return false;
    }
    if (config.n_ctx > INT32_MAX || config.n_batch > INT32_MAX || config.n_ubatch > INT32_MAX
            || config.threads > INT32_MAX) {
        set_error("engine context, batch and thread counts must fit in int32_t");
        return false;
    }
    // llama.cpp clamps the physical batch to n_ctx before padding its KV
    // pool. Its initial output reservation asserts if the sequence count no
    // longer fits that effective batch.
    if (config.n_seq_max > LLAMA_MAX_SEQ
            || config.n_seq_max > std::min(config.n_batch, config.n_ctx)) {
        set_error("engine n_seq_max must not exceed LLAMA_MAX_SEQ, n_batch or n_ctx");
        return false;
    }
    if (config.kv_dtype < RETRO_ENGINE_KV_DTYPE_F32 || config.kv_dtype > RETRO_ENGINE_KV_DTYPE_Q4_0) {
        set_error("engine kv_dtype must be one of retro_engine_kv_dtype");
        return false;
    }
    if ((config.kv_dtype == RETRO_ENGINE_KV_DTYPE_Q8_0 || config.kv_dtype == RETRO_ENGINE_KV_DTYPE_Q4_0)
            && !config.flash_attn) {
        set_error("a quantized engine kv_dtype requires flash_attn");
        return false;
    }
    if (config.device < RETRO_DEVICE_AUTO || config.device > RETRO_DEVICE_GPU) {
        set_error("engine device must be one of retro_device");
        return false;
    }
    return true;
}

// Loads the model mapped and builds the single forward-only context. The
// context parameters mirror the trainer's dedicated generation context (see
// load_model_and_context), which is what makes engine and trainer generation
// bit-identical at equal geometry; what is absent here is everything the
// optimizer needs.
bool load_engine_model_and_context(engine_state & state) {
    ensure_backend_initialized();

    const int32_t requested = state.config.device;
    ggml_backend_dev_t gpu_device = first_gpu_device();
    const bool gpu_available = gpu_device != nullptr;
    bool use_gpu = false;
    switch (requested) {
        case RETRO_DEVICE_CPU:
            use_gpu = false;
            break;
        case RETRO_DEVICE_GPU:
            if (!gpu_available) {
                set_error("device=gpu requested but no GPU backend is available; "
                          "rebuild with Cargo features metal (Apple hardware), "
                          "vulkan (Vulkan SDK), or "
                          "cuda (NVIDIA CUDA toolkit), and confirm a "
                          "supported GPU is present");
                return false;
            }
            use_gpu = true;
            break;
        case RETRO_DEVICE_AUTO:
        default:
            use_gpu = gpu_available;
            break;
    }

    state.requested_device = requested;
    state.gpu_active = use_gpu;
    state.n_gpu_layers = use_gpu ? 999 : 0;
    const char * backend_name = use_gpu ? device_name(gpu_device) : nullptr;
    state.backend_name = backend_name ? backend_name : "CPU";
    ggml_backend_reg_t active_reg = use_gpu && gpu_device
            ? ggml_backend_dev_backend_reg(gpu_device)
            : nullptr;
    const char * registry_name = active_reg ? ggml_backend_reg_name(active_reg) : nullptr;
    state.backend_registry = use_gpu
            ? (registry_name && registry_name[0] ? registry_name : state.backend_name)
            : "CPU";

    llama_model_params model_params = llama_model_default_params();
    // mmap the weights when they stay on the host, and skip it for a GPU load
    // where every tensor is copied into device memory anyway. The engine never
    // writes base weights, so the mapping is always read-only here - the
    // trainer forces owned buffers only for a base-weight training policy.
    model_params.load_mode = use_gpu ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
    state.weight_storage = model_params.load_mode == LLAMA_LOAD_MODE_MMAP ? "mapped" : "owned";
    model_params.use_extra_bufts = false;
    model_params.n_gpu_layers = state.n_gpu_layers;
    // Same reason as the trainer's CPU path: an explicit empty device list
    // keeps the scheduler CPU-only, which is what device=cpu asked for anyway.
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (!use_gpu) {
        model_params.devices = no_devices;
    }

    model_ptr model(llama_model_load_from_file(state.model_path.c_str(), model_params));
    if (!model) {
        set_error("failed to load model GGUF: " + state.model_path);
        return false;
    }
    state.chat.model = model.get();

    uint32_t n_threads = 0;
    retro_train_config thread_config {};
    thread_config.threads = state.config.threads;
    if (!resolve_thread_count(thread_config, n_threads)
            || n_threads > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        if (g_last_error.empty()) {
            set_error("thread count exceeds the llama.cpp limit");
        }
        return false;
    }
    state.effective_threads = n_threads;

    const ggml_type kv_type = engine_kv_type(state.config.kv_dtype);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = state.config.n_ctx;
    ctx_params.n_batch = state.config.n_batch;
    ctx_params.n_ubatch = state.config.n_ubatch;
    ctx_params.n_seq_max = state.config.n_seq_max;
    // Same as the trainer's contexts: one unified KV pool, no recurrent
    // rollback (sampling never rolls a sequence back), windowed SWA, and no
    // differentiable store - forward only.
    ctx_params.kv_unified = true;
    ctx_params.n_rs_seq = 0;
    ctx_params.swa_full = false;
    ctx_params.kv_differentiable = false;
    ctx_params.no_fused_gdn = false;
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    ctx_params.flash_attn_type = state.config.flash_attn
            ? LLAMA_FLASH_ATTN_TYPE_AUTO
            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_threads = static_cast<int32_t>(n_threads);
    ctx_params.n_threads_batch = static_cast<int32_t>(n_threads);
    ctx_params.no_perf = true;
    ctx_params.offload_kqv = use_gpu;
    ctx_params.op_offload = use_gpu;

    context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    if (!ctx) {
        set_error("failed to create the engine context for model: " + state.model_path);
        return false;
    }

    state.effective_kv_dtype = state.config.kv_dtype == RETRO_ENGINE_KV_DTYPE_F32
            ? RETRO_KV_DTYPE_F32
            : RETRO_KV_DTYPE_F16;
    state.cap_device_sampling = use_gpu && supports_device_sampling(gpu_device);
    state.cap_device_logprobs = use_gpu && supports_device_logprob_gather(gpu_device);
    state.env = gen_core_env{
        nullptr,
        use_gpu,
        state.cap_device_sampling,
        state.cap_device_logprobs,
        nullptr,
    };

    state.model = std::move(model);
    state.ctx = std::move(ctx);
    return true;
}

// A gen_core over the engine's context. The engine owns its prefix-cache
// bookkeeping, so a multi-turn call reuses what an earlier one left resident.
gen_core engine_core(engine_state & state) {
    return gen_core(state.model.get(), state.ctx.get(), state.env, &state.cache);
}

// Adapter changes invalidate decoded sequence memory: a prefix decoded under
// the adapter of the moment is not the prefix of the next one.
void forget_engine_kv(engine_state & state) {
    if (llama_memory_t memory = llama_get_memory(state.ctx.get())) {
        llama_memory_clear(memory, false);
    }
    state.cache.forget_bookkeeping();
}

// Same two-call contract and same guards as the trainer's tokenize_into.
int tokenize_into(
        engine_state & state,
        const char * text,
        bool add_special,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    if (is_blank(text)) {
        set_error("text is required");
        return -1;
    }
    if (!out_n_tokens) {
        set_error("out_n_tokens is required");
        return -1;
    }
    const size_t text_size = std::strlen(text);
    if (text_size > static_cast<size_t>(INT32_MAX)) {
        set_error("text is too large to tokenize");
        return -1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model.get());
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
        set_error("tokenization overflow");
        return -1;
    }
    if (result < 0) {
        *out_n_tokens = static_cast<size_t>(-result);
        return tokens == nullptr || n_tokens_max == 0 ? 0 : -2;
    }

    *out_n_tokens = static_cast<size_t>(result);
    return 0;
}

// The engine's memory report, from the same per-buffer computation as the
// trainer's (context_memory_by_buffer). Weights are counted once - the
// engine has exactly one context - and land in the generation buckets. There
// is no optimizer context, so optimizer_kv/compute are zero and the measured
// device fields stay at "unavailable" (samples = 0); the trainable set, when
// an adapter is attached, is the adapter alone, with zero gradient and zero
// optimizer state, which is the property the report exists to prove.
retro_memory_report engine_memory_totals(const engine_state & state) {
    retro_memory_report report {};
    report.has_generation_context = true;
    uint64_t adapter_bytes = 0;
    bool adapter_on_host = true;
    if (state.adapter) {
        for (const auto & item : state.adapter->ab_map) {
            if (item.second.a) {
                adapter_bytes += ggml_nbytes(item.second.a);
            }
            if (item.second.b) {
                adapter_bytes += ggml_nbytes(item.second.b);
            }
        }
        const ggml_tensor * a = state.adapter->ab_map.empty()
                ? nullptr
                : state.adapter->ab_map.begin()->second.a;
        if (a && a->buffer) {
            adapter_on_host = ggml_backend_buft_is_host(ggml_backend_buffer_get_type(a->buffer));
        }
    }
    report.trainable_parameter_bytes = adapter_bytes;
    // Adapter factors are allocated on top of the loaded model, never a slice
    // of it.
    report.trainable_parameters_are_model_subset = false;
    report.adapter_on_host = adapter_on_host;
    report.base_trainable_on_host = adapter_on_host;

    for (const auto & item : context_memory_by_buffer(state.ctx.get())) {
        report.model_weight_bytes += item.second.model;
        report.generation_kv_bytes += item.second.kv;
        report.generation_compute_bytes += item.second.compute;
        const uint64_t total = item.second.model + item.second.kv + item.second.compute;
        (item.second.is_host ? report.host_bytes : report.device_bytes) += total;
    }
    (adapter_on_host ? report.host_bytes : report.device_bytes) += adapter_bytes;
    // optimizer_state_bytes and trainable_gradient_bytes stay zero: the engine
    // has no optimizer and no backward graph. That is a fact about the handle,
    // not a missing measurement.
    return report;
}

} // namespace

} // namespace retro

namespace retro {

extern "C" retro_engine * retro_engine_new(
        const char * model_path,
        const retro_engine_config * config) {
    try {
        clear_error();
        if (is_blank(model_path)) {
            set_error("model_path is required");
            return nullptr;
        }
        retro_engine_config resolved = default_engine_config();
        if (config) {
            // Read the evolution key before copying any other field: an old or
            // truncated caller may not own sizeof(retro_engine_config) bytes.
            const uint32_t size = config->struct_size;
            constexpr size_t required = offsetof(retro_engine_config, verbose) + sizeof(bool);
            if (size < required) {
                set_error("retro_engine_config.struct_size is too small for this build");
                return nullptr;
            }
            std::memcpy(&resolved, config, std::min<size_t>(size, sizeof(resolved)));
        }
        if (!validate_engine_config(resolved)) {
            return nullptr;
        }
        configure_runtime_logging(resolved.verbose);

        std::unique_ptr<engine_state> state(new engine_state());
        // Before the model load: that is what creates the backend contexts, so
        // anything counted after this point belongs to this engine.
        state->rir_baseline = ggml_rir_counters_snapshot();
        state->model_path = model_path;
        state->config = resolved;
        if (!load_engine_model_and_context(*state)) {
            return nullptr;
        }
        return reinterpret_cast<retro_engine *>(state.release());
    } catch (const std::exception & err) {
        set_error(err.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

extern "C" void retro_engine_free(retro_engine * engine) {
    delete reinterpret_cast<engine_state *>(engine);
}

extern "C" int retro_engine_generate_batch(
        retro_engine * engine,
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
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.generate_batch(prompt_tokens, n_prompt, sampling, n_sequences,
                out_tokens, out_logprobs, n_out_max, out_n_tokens, out_forced, plan);
    });
}

extern "C" int retro_engine_generate_continuous_batch(
        retro_engine * engine,
        const retro_generation_sequence * sequences,
        size_t n_sequences,
        int32_t * out_tokens,
        float * out_logprobs,
        size_t n_out_max,
        size_t * out_n_tokens,
        uint8_t * out_forced) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.generate_continuous_batch(sequences, n_sequences, out_tokens,
                out_logprobs, n_out_max, out_n_tokens, out_forced, plan);
    });
}

extern "C" int retro_engine_tokenize_text(
        retro_engine * engine,
        const char * text,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return tokenize_into(*state, text, /*add_special =*/ true, tokens, n_tokens_max, out_n_tokens);
    });
}

extern "C" int retro_engine_tokenize_fragment(
        retro_engine * engine,
        const char * text,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return tokenize_into(*state, text, /*add_special =*/ false, tokens, n_tokens_max, out_n_tokens);
    });
}

extern "C" int retro_engine_eos_token(retro_engine * engine, int32_t * out_token) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_token) {
            if (state) set_error("out_token is required");
            return -1;
        }
        *out_token = llama_vocab_eos(llama_model_get_vocab(state->model.get()));
        return 0;
    });
}

extern "C" int retro_engine_vocab_size(retro_engine * engine, uint32_t * out_n_vocab) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_vocab) {
            if (state) set_error("out_n_vocab is required");
            return -1;
        }
        const int32_t size = llama_vocab_n_tokens(llama_model_get_vocab(state->model.get()));
        if (size < 0) {
            set_error("model reported a negative vocabulary size");
            return -1;
        }
        *out_n_vocab = static_cast<uint32_t>(size);
        return 0;
    });
}

extern "C" int retro_engine_is_eog_token(retro_engine * engine, int32_t token, bool * out_is_eog) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_is_eog) {
            if (state) set_error("out_is_eog is required");
            return -1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
            set_error("token is outside the vocabulary");
            return -1;
        }
        *out_is_eog = llama_vocab_is_eog(vocab, token);
        return 0;
    });
}

extern "C" int retro_engine_context_size(retro_engine * engine, uint32_t * out_n_ctx) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_ctx) {
            if (state) set_error("out_n_ctx is required");
            return -1;
        }
        *out_n_ctx = llama_n_ctx_seq(state->ctx.get());
        return 0;
    });
}

extern "C" int retro_engine_detokenize(
        retro_engine * engine,
        const int32_t * tokens,
        size_t n_tokens,
        bool unparse_special,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
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
        gen_core core = engine_core(*state);
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

extern "C" int retro_engine_load_lora(retro_engine * engine, const char * adapter_path) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (is_blank(adapter_path)) {
            set_error("adapter_path is required");
            return -1;
        }
        // Refuse before touching the context: once the adapter is applied, an
        // error return would leave the context pointing at a freed adapter.
        lora_adapter_ptr adapter(llama_adapter_lora_init(state->model.get(), adapter_path));
        if (!adapter) {
            set_error("failed to load LoRA adapter: " + std::string(adapter_path));
            return -1;
        }
        if (!validate_lora_tensor_pairs(*adapter)) {
            return -1;
        }
        llama_adapter_lora * adapters[] = { adapter.get() };
        float scales[] = { 1.0f };
        if (llama_set_adapters_lora(state->ctx.get(), adapters, 1, scales) != 0) {
            set_error("failed to apply LoRA adapter to the engine context");
            return -1;
        }
        state->adapter = std::move(adapter);
        state->lora_enabled = true;
        // No tensor is flagged as an optimizer parameter: the engine has no
        // optimizer, and the load path above is where a trainer's would.
        forget_engine_kv(*state);
        return 0;
    });
}

extern "C" int retro_engine_set_lora_enabled(retro_engine * engine, bool enabled) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        // Adapter changes invalidate decoded sequence memory.
        if (llama_memory_t memory = llama_get_memory(state->ctx.get())) {
            llama_memory_clear(memory, true);
        }
        state->cache.forget_bookkeeping();
        if (!enabled) {
            llama_set_adapters_lora(state->ctx.get(), nullptr, 0, nullptr);
            state->lora_enabled = false;
            return 0;
        }
        if (!state->adapter) {
            set_error("cannot enable LoRA: no adapter exists");
            return -1;
        }
        llama_adapter_lora * adapters[] = { state->adapter.get() };
        float scales[] = { 1.0f };
        if (llama_set_adapters_lora(state->ctx.get(), adapters, 1, scales) != 0) {
            set_error("failed to apply LoRA adapter to the engine context");
            return -1;
        }
        state->lora_enabled = true;
        return 0;
    });
}

extern "C" int retro_engine_set_chat_template_variables(
        retro_engine * engine,
        const char * variables_json) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return set_chat_template_variables(state->chat, variables_json);
    });
}

extern "C" int retro_engine_set_reasoning_budget(retro_engine * engine, int32_t budget) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return set_reasoning_budget(state->chat, budget);
    });
}

extern "C" int retro_engine_memory_report(
        retro_engine * engine,
        retro_memory_report * out_report) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_report) {
            if (state) set_error("out_report is required");
            return -1;
        }
        *out_report = engine_memory_totals(*state);
        return 0;
    });
}

extern "C" int retro_engine_generation_stats(
        retro_engine * engine,
        retro_generation_stats * out_stats) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_stats) {
            if (state) set_error("out_stats is required");
            return -1;
        }
        *out_stats = state->cache.stats;
        return 0;
    });
}

extern "C" int retro_engine_step(
        retro_engine * engine,
        const retro_step_row * rows,
        size_t n_rows,
        retro_step_out_row * out_rows,
        int32_t * out_top_ids,
        float * out_top_logprobs,
        size_t n_top_stride) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        const reasoning_budget_plan * plan = resolve_reasoning_budget(state->chat);
        if (!plan) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.step(rows, n_rows, out_rows, out_top_ids, out_top_logprobs, n_top_stride,
                state->step_requests, plan, state->step_batch.get(state->ctx.get()));
    });
}

extern "C" int retro_engine_seq_rm(retro_engine * engine, int32_t slot, int32_t p0, int32_t p1) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.seq_rm(slot, p0, p1) ? 0 : -1;
    });
}

extern "C" int retro_engine_seq_cp(
        retro_engine * engine, int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.seq_cp(src, dst, p0, p1) ? 0 : -1;
    });
}

extern "C" int retro_engine_seq_save(
        retro_engine * engine,
        int32_t slot,
        uint32_t flags,
        uint8_t * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.seq_save(slot, flags, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_engine_seq_load(
        retro_engine * engine,
        int32_t slot,
        uint32_t flags,
        const uint8_t * buffer,
        size_t n_bytes,
        size_t * out_n_read) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.seq_load(slot, flags, buffer, n_bytes, out_n_read);
    });
}

extern "C" int retro_engine_seq_pos(
        retro_engine * engine, int32_t slot, int32_t * out_pos_min, int32_t * out_pos_max) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.seq_pos(slot, out_pos_min, out_pos_max);
    });
}

extern "C" int retro_engine_memory(retro_engine * engine, retro_engine_memory_info * out) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        gen_core core = engine_core(*state);
        return core.memory(out);
    });
}

extern "C" int retro_engine_request_free(retro_engine * engine, int32_t request) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        state->step_requests.requests.erase(request);
        return 0;
    });
}

} // namespace retro
