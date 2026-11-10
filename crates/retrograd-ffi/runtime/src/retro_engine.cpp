// The inference engine behind the opaque retro_engine handle: a mapped model,
// one forward-only context and the shared gen_core (retro_gen_core.cpp) - the
// exact same generation code the trainer's retro_trainer_generate_* run, so
// the two agree bit for bit. What the engine deliberately does not have: a
// llama_opt context, a training result cache, trainable flags, a duty-cycle
// limiter. retro_engine_memory_report proves the first three by reporting
// zero optimizer state and zero gradient.
#include "retro_gen_core.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>

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
    // Every LoRA adapter registered by name, resident beside the base model.
    // `active_adapter` names the one applied to the context (empty = base).
    std::map<std::string, lora_adapter_ptr> adapters;
    std::string active_adapter;
    bool lora_enabled = false;
    // Per-row adapters, selected by the step rows' adapter field. Declared
    // after `model`: freeing an adapter unregisters it from its model.
    lora_row_router lora_rows;
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
    // Whether the context outputs embeddings outside a call that sets them for
    // itself: a decision model that reads its embeddings output is created
    // in that mode, like llama-server creates it.
    bool embeddings_default = false;
    // Bumped by forget_engine_kv(): every event that invalidates resident KV
    // (adapter load/enable, explicit forget) moves the weights generation, so
    // a prefix cache keyed on it can drop stale entries.
    uint64_t weights_version = 0;
    // Expert placement, probed before the load (one metadata-only read) and
    // kept for the load: the bank inventory validates the request and the
    // host overrides pin the chosen layers' banks to the CPU buffer type.
    // Empty when the placement is DEVICE with no cache - the default, which
    // loads exactly as before.
    std::vector<std::string> expert_host_patterns;
    std::vector<llama_model_tensor_buft_override> expert_host_overrides;
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
    config.kv_unified = 1;
    config.expert_placement = RETRO_EXPERT_PLACEMENT_DEVICE;
    config.expert_host_layers = 0;
    config.expert_device_cache_bytes = 0;
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
    if (config.expert_placement != RETRO_EXPERT_PLACEMENT_DEVICE
            && config.expert_placement != RETRO_EXPERT_PLACEMENT_HOST) {
        set_error("engine expert_placement must be one of retro_expert_placement");
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
    // Host expert placement is the exception: the host-resident banks must stay
    // mapped even with a GPU, so the load mode is MMAP there too.
    const bool host_experts = state.config.expert_placement == RETRO_EXPERT_PLACEMENT_HOST
            && !state.expert_host_overrides.empty();
    model_params.load_mode = (use_gpu && !host_experts) ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
    state.weight_storage = model_params.load_mode == LLAMA_LOAD_MODE_MMAP ? "mapped" : "owned";
    // Host placement pins the chosen layers' expert banks to the CPU buffer
    // type; every other tensor follows the device offload as usual.
    if (host_experts) {
        model_params.tensor_buft_overrides = state.expert_host_overrides.data();
    }
    // The host weights keep their file layout, as the trainer's do, so the
    // engine's generation is the trainer's to the bit. A decision model is
    // the exception: it generates nothing, and llama-server reads its Q8_0
    // weights repacked into the CPU's extra buffer types, which changes the
    // arithmetic of every matrix product. Serving it with the same kernels is
    // what serving it with the same numbers takes.
    model_params.use_extra_bufts = !use_gpu && gguf_is_decision_model(state.model_path.c_str());
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
    // Same as the trainer's contexts by default: one unified KV pool (the
    // config may partition it), no recurrent rollback (sampling never rolls a sequence back), windowed SWA,
    // and no differentiable store - forward only.
    ctx_params.kv_unified = state.config.kv_unified != 0;
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
    // The MoE expert device cache, for host-resident banks on a GPU context.
    // The budget was validated against the routing width before the load.
    ctx_params.moe_cache_size = state.config.expert_device_cache_bytes;
    ctx_params.no_perf = true;
    ctx_params.offload_kqv = use_gpu;
    ctx_params.op_offload = use_gpu;
    // A decision model that reads its embeddings output gets the context
    // llama-server gives it: embeddings on, its pooling, an output per batch
    // row (each option of a question is read at its own position).
    enum llama_pooling_type decision_pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    if (decision_context_pooling(model.get(), &decision_pooling)) {
        ctx_params.embeddings = true;
        ctx_params.pooling_type = decision_pooling;
        ctx_params.n_outputs_max = ctx_params.n_batch;
        state.embeddings_default = true;
    }

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
        &state.lora_rows,
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
    state.weights_version += 1;
}

// Same two-call contract and same guards as the trainer's tokenize_into.
int tokenize_into(
        engine_state & state,
        const char * text,
        bool add_special,
        bool parse_special,
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
            parse_special);

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
    // The report describes the applied adapter; the registered-but-inactive
    // ones also occupy memory, so they are counted in as well.
    for (const auto & [name, adapter] : state.adapters) {
        if (!adapter) {
            continue;
        }
        for (const auto & item : adapter->ab_map) {
            if (item.second.a) {
                adapter_bytes += ggml_nbytes(item.second.a);
            }
            if (item.second.b) {
                adapter_bytes += ggml_nbytes(item.second.b);
            }
        }
        if (name == state.active_adapter) {
            const ggml_tensor * a = adapter->ab_map.empty()
                    ? nullptr
                    : adapter->ab_map.begin()->second.a;
            if (a && a->buffer) {
                adapter_on_host =
                        ggml_backend_buft_is_host(ggml_backend_buffer_get_type(a->buffer));
            }
        }
    }
    // Per-row adapters hold their own factors plus the padded stacks the
    // context gathers from; both stay resident while they are loaded.
    for (const auto & [id, entry] : state.lora_rows.adapters) {
        for (const auto & item : entry.adapter->ab_map) {
            adapter_bytes += ggml_nbytes(item.second.a) + ggml_nbytes(item.second.b);
        }
    }
    retro_adapter_rows_info rows {};
    lora_router_info(state.ctx.get(), rows);
    adapter_bytes += rows.stacked_bytes;
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
    return retro_engine_new_checked(model_path, config, nullptr);
}

extern "C" retro_engine * retro_engine_new_checked(
        const char * model_path,
        const retro_engine_config * config,
        bool * out_invalid_placement) {
    if (out_invalid_placement) {
        *out_invalid_placement = false;
    }
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
            uint32_t size = 0;
            std::memcpy(&size, config, sizeof(size));
            constexpr size_t required = offsetof(retro_engine_config, verbose) + sizeof(bool);
            if (size < required) {
                set_error("retro_engine_config.struct_size is too small for this build");
                return nullptr;
            }
            // A caller compiled against the 40-byte layout stops before
            // kv_unified, which then keeps its default.
            static_assert(offsetof(retro_engine_config, kv_unified) == 40);
            // The expert placement fields land past the 44-byte layout; the
            // u64 cache budget is padded to the 8-byte boundary.
            static_assert(offsetof(retro_engine_config, expert_placement) == 44);
            static_assert(offsetof(retro_engine_config, expert_host_layers) == 48);
            static_assert(offsetof(retro_engine_config, expert_device_cache_bytes) == 56);
            static_assert(sizeof(retro_engine_config) == 64);
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
        // A non-default placement is validated against the model's expert bank
        // inventory before the real load: a refused request must not create a
        // backend context. The default (DEVICE, no cache) skips the probe and
        // loads exactly as before.
        expert_placement_probe probe;
        if (resolved.expert_placement == RETRO_EXPERT_PLACEMENT_HOST
                && !probe_expert_banks(model_path, probe)) {
            return nullptr;
        }
        const bool cache_gpu = resolved.expert_device_cache_bytes > 0
                && resolved.device != RETRO_DEVICE_CPU && first_gpu_device() != nullptr;
        if (!validate_expert_placement(resolved, probe, cache_gpu)
                || (resolved.expert_placement == RETRO_EXPERT_PLACEMENT_HOST
                    && !build_host_buft_overrides(
                        probe, resolved.expert_host_layers,
                        state->expert_host_patterns, state->expert_host_overrides))) {
            if (out_invalid_placement) {
                *out_invalid_placement = true;
            }
            return nullptr;
        }
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
        return tokenize_into(*state, text, /*add_special =*/ true, /*parse_special =*/ true, tokens, n_tokens_max, out_n_tokens);
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
        return tokenize_into(*state, text, /*add_special =*/ false, /*parse_special =*/ true, tokens, n_tokens_max, out_n_tokens);
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

// Applies the active adapter to the context (nullptr when the base model is
// active), so exactly one adapter is ever applied: a decode batch stays
// homogeneous. Returns false with the error set when the apply fails.
bool apply_active_adapter(engine_state & state) {
    if (state.active_adapter.empty()) {
        llama_set_adapters_lora(state.ctx.get(), nullptr, 0, nullptr);
        state.lora_enabled = false;
        return true;
    }
    auto found = state.adapters.find(state.active_adapter);
    if (found == state.adapters.end()) {
        set_error("the active LoRA adapter is not registered");
        return false;
    }
    llama_adapter_lora * adapters[] = { found->second.get() };
    float scales[] = { 1.0f };
    if (llama_set_adapters_lora(state.ctx.get(), adapters, 1, scales) != 0) {
        set_error("failed to apply LoRA adapter to the engine context");
        return false;
    }
    state.lora_enabled = true;
    return true;
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
        // The historical single-adapter contract: every other adapter goes.
        state->adapters.clear();
        state->adapters.emplace(adapter_path, std::move(adapter));
        state->active_adapter = adapter_path;
        if (!apply_active_adapter(*state)) {
            return -1;
        }
        // No tensor is flagged as an optimizer parameter: the engine has no
        // optimizer, and the load path above is where a trainer's would.
        forget_engine_kv(*state);
        return 0;
    });
}

extern "C" int retro_engine_load_lora_named(
        retro_engine * engine, const char * name, const char * adapter_path) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (is_blank(name)) {
            set_error("name is required");
            return -1;
        }
        if (is_blank(adapter_path)) {
            set_error("adapter_path is required");
            return -1;
        }
        lora_adapter_ptr adapter(llama_adapter_lora_init(state->model.get(), adapter_path));
        if (!adapter) {
            set_error("failed to load LoRA adapter: " + std::string(adapter_path));
            return -1;
        }
        if (!validate_lora_tensor_pairs(*adapter)) {
            return -1;
        }
        // Registered beside the others, not applied: the context still runs
        // the active adapter. Only a reload of the active one re-applies it,
        // because the context would otherwise point at the freed old adapter.
        const bool was_active = state->active_adapter == name;
        state->adapters.insert_or_assign(name, std::move(adapter));
        if (was_active) {
            if (!apply_active_adapter(*state)) {
                return -1;
            }
            forget_engine_kv(*state);
        }
        return 0;
    });
}

extern "C" int retro_engine_activate_lora(retro_engine * engine, const char * name) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        const std::string wanted = name ? name : "";
        if (!wanted.empty() && state->adapters.find(wanted) == state->adapters.end()) {
            set_error("no LoRA adapter registered under '" + wanted + "'");
            return -1;
        }
        state->active_adapter = wanted;
        if (!apply_active_adapter(*state)) {
            return -1;
        }
        // Adapter changes invalidate decoded sequence memory: a prefix decoded
        // under the adapter of the moment is not the prefix of the next one.
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
        if (!enabled) {
            state->active_adapter.clear();
        } else if (state->active_adapter.empty()) {
            set_error("cannot enable LoRA: no adapter is active");
            return -1;
        }
        if (!apply_active_adapter(*state)) {
            return -1;
        }
        // Adapter changes invalidate decoded sequence memory, in both
        // directions: disabling is as much a weights change as enabling.
        forget_engine_kv(*state);
        return 0;
    });
}

extern "C" int retro_engine_adapter_load(
        retro_engine * engine, const char * adapter_path, float scale, int32_t * out_id) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (is_blank(adapter_path)) {
            set_error("adapter_path is required");
            return -1;
        }
        if (!out_id) {
            set_error("out_id is required");
            return -1;
        }
        if (!std::isfinite(scale)) {
            set_error("the adapter scale must be finite");
            return -1;
        }
        lora_row_router & router = state->lora_rows;
        if (router.next_id == INT32_MAX) {
            set_error("per-row adapter ids are exhausted");
            return -1;
        }
        lora_adapter_ptr adapter(llama_adapter_lora_init(state->model.get(), adapter_path));
        if (!adapter) {
            set_error("failed to load LoRA adapter: " + std::string(adapter_path));
            return -1;
        }
        if (!validate_lora_tensor_pairs(*adapter)) {
            return -1;
        }
        const int32_t id = router.next_id++;
        router.adapters.emplace(id, lora_row_router::entry{ std::move(adapter), scale });
        if (!lora_router_install(router, state->ctx.get())) {
            // Restack what was there before: a refused adapter (aLoRA, an
            // embedding or expert target) must not take the others down.
            const std::string error = retro_last_error();
            router.adapters.erase(id);
            lora_router_install(router, state->ctx.get());
            set_error(error);
            return -1;
        }
        *out_id = id;
        return 0;
    });
}

extern "C" int retro_engine_adapter_unload(retro_engine * engine, int32_t id) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        lora_row_router & router = state->lora_rows;
        auto found = router.adapters.find(id);
        if (found == router.adapters.end()) {
            set_error("no per-row adapter is loaded under id " + std::to_string(id));
            return -1;
        }
        // Restack first: the context's stacks are copies, so the adapter can
        // only go once the routed set no longer names it.
        lora_adapter_ptr adapter = std::move(found->second.adapter);
        router.adapters.erase(found);
        if (!lora_router_install(router, state->ctx.get())) {
            return -1;
        }
        return 0;
    });
}

extern "C" int retro_engine_adapter_rows_info(
        retro_engine * engine, retro_adapter_rows_info * out_info) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (!out_info) {
            set_error("out_info is required");
            return -1;
        }
        lora_router_info(state->ctx.get(), *out_info);
        return 0;
    });
}

extern "C" int retro_engine_graphs_reused(retro_engine * engine, uint64_t * out_n_reused) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (!out_n_reused) {
            set_error("out_n_reused is required");
            return -1;
        }
        // Counted whether or not perf timing is enabled, and never reset here:
        // nothing in the runtime calls llama_perf_context_reset.
        *out_n_reused = (uint64_t) std::max(0, llama_perf_context(state->ctx.get()).n_reused);
        return 0;
    });
}

extern "C" int retro_engine_active_adapter(
        retro_engine * engine, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        return render_string_out(
                buffer, n_buffer, out_n_bytes, "active adapter query failed",
                [&](char * destination, int32_t capacity) -> int32_t {
                    const std::string & name = state->active_adapter;
                    const auto n = static_cast<int32_t>(
                            std::min<size_t>(name.size(), static_cast<size_t>(INT32_MAX)));
                    if (destination && capacity > 0) {
                        const auto to_copy = static_cast<size_t>(std::min<int32_t>(n, capacity));
                        std::memcpy(destination, name.data(), to_copy);
                    }
                    return n;
                });
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

extern "C" int retro_engine_format_chat(
        retro_engine * engine,
        const char * const * roles,
        const char * const * contents,
        size_t n_messages,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_bytes) {
            if (state) set_error("out_n_bytes is required");
            return -1;
        }
        if (n_messages > 0 && (!roles || !contents)) {
            set_error("roles and contents are required");
            return -1;
        }
        for (size_t i = 0; i < n_messages; ++i) {
            if (is_blank(roles[i]) || contents[i] == nullptr) {
                set_error("chat roles and contents must not be empty");
                return -1;
            }
        }
        return render_chat_template(
                state->chat, roles, contents, n_messages, add_assistant,
                buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_engine_format_chat_messages(
        retro_engine * engine,
        const char * messages_json,
        const char * tools_json,
        bool add_assistant,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_bytes) {
            if (state) set_error("out_n_bytes is required");
            return -1;
        }
        if (is_blank(messages_json)) {
            set_error("chat messages JSON is required");
            return -1;
        }
        return render_chat_messages(
                state->chat, messages_json, tools_json, add_assistant,
                buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_engine_chat_template_supports_tools(
        retro_engine * engine,
        bool * out_supports) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_supports) {
            if (state) set_error("out_supports is required");
            return -1;
        }
        return chat_template_supports_tools(state->chat, out_supports);
    });
}

extern "C" int retro_engine_tool_call_parser(
        retro_engine * engine,
        const char * tools_json,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_bytes) {
            if (state) set_error("out_n_bytes is required");
            return -1;
        }
        return build_tool_call_parser(state->chat, tools_json, buffer, n_buffer, out_n_bytes);
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

extern "C" int retro_engine_step_v2(
        retro_engine * engine,
        const retro_step_row_v2 * rows,
        size_t n_rows,
        retro_step_out_row_v2 * out_rows,
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
        return core.step_v2(rows, n_rows, out_rows, out_top_ids, out_top_logprobs, n_top_stride,
                state->step_requests, plan, state->step_batch.get(state->ctx.get()));
    });
}

extern "C" int retro_engine_tool_call_grammar(
        retro_engine * engine,
        const char * tools_json,
        int32_t mode,
        const char * forced_name,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_n_bytes) {
            if (state) set_error("out_n_bytes is required");
            return -1;
        }
        return build_tool_call_grammar(
                state->chat, tools_json, mode, forced_name, buffer, n_buffer, out_n_bytes);
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

extern "C" int retro_engine_kv_unified(retro_engine * engine, bool * out_unified) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_unified) {
            if (state) set_error("out_unified is required");
            return -1;
        }
        // The configuration the context was built from.
        *out_unified = state->config.kv_unified != 0;
        return 0;
    });
}

extern "C" int retro_engine_weights_version(retro_engine * engine, uint64_t * out_version) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_version) {
            if (state) set_error("out_version is required");
            return -1;
        }
        *out_version = state->weights_version;
        return 0;
    });
}

extern "C" int retro_engine_is_hybrid(retro_engine * engine, bool * out_is_hybrid) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_is_hybrid) {
            if (state) set_error("out_is_hybrid is required");
            return -1;
        }
        *out_is_hybrid = llama_model_is_hybrid(state->model.get());
        return 0;
    });
}

extern "C" int retro_engine_model_fingerprint(retro_engine * engine, uint64_t * out_fingerprint) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_fingerprint) {
            if (state) set_error("out_fingerprint is required");
            return -1;
        }
        *out_fingerprint = model_fingerprint(state->model_path, state->model.get());
        return 0;
    });
}

extern "C" int retro_engine_pooling_type(retro_engine * engine, int32_t * out_pooling) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_pooling) {
            if (state) set_error("out_pooling is required");
            return -1;
        }
        *out_pooling = static_cast<int32_t>(llama_pooling_type(state->ctx.get()));
        return 0;
    });
}

extern "C" int retro_engine_embed(
        retro_engine * engine,
        const int32_t * tokens,
        size_t n_tokens,
        float * out_features,
        size_t n_embd_max,
        size_t * out_n_embd) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (!out_n_embd) {
            set_error("out_n_embd is required");
            return -1;
        }
        llama_context * ctx = state->ctx.get();
        const size_t n_embd = llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_RANK
                ? llama_model_n_cls_out(state->model.get())
                : static_cast<size_t>(llama_model_n_embd_out(state->model.get()));
        *out_n_embd = n_embd;
        if (!out_features) {
            // Size query only: no decode.
            if (n_embd == 0) {
                set_error("model reports no output embedding width");
                return -1;
            }
            return 0;
        }
        if (!tokens || n_tokens == 0) {
            set_error("embedding extraction requires tokens");
            return RETRO_EMBED_INVALID_INPUT;
        }
        const size_t n_ctx = llama_n_ctx(ctx);
        if (n_tokens > n_ctx) {
            std::ostringstream message;
            message << "sequence of " << n_tokens << " tokens exceeds context " << n_ctx;
            set_error(message.str());
            return RETRO_EMBED_INVALID_INPUT;
        }
        gen_core core = engine_core(*state);
        if (!core.validate_tokens(tokens, n_tokens)) {
            return RETRO_EMBED_INVALID_INPUT;
        }
        // A model that declares no pooling is a plain causal chat model: no
        // embedding is invented for it by averaging hidden states.
        if (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE) {
            set_error("the model declares no pooling (a causal chat model): embeddings are not "
                      "served for it");
            return RETRO_EMBED_NO_POOLING;
        }
        if (n_embd == 0) {
            set_error("model reports no output embedding width");
            return -1;
        }
        if (n_embd_max < n_embd) {
            set_error("out_features buffer is too small");
            return -1;
        }
        // The pooled output is computed over the whole sequence, so the input
        // must fit one physical batch.
        const uint32_t n_batch = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
        if (n_tokens > n_batch) {
            std::ostringstream message;
            message << "the input is " << n_tokens << " tokens, but a pooled embedding needs "
                    << "the whole sequence in one batch of " << n_batch << "; shorten the input";
            set_error(message.str());
            return RETRO_EMBED_INVALID_INPUT;
        }
        // The embedding graph differs from the decode graph: apply the active
        // adapter, clear the KV (and with it the prefix cache) on every exit
        // path, and restore the plain-logits configuration afterwards.
        if (!apply_active_adapter(*state)) {
            return -1;
        }
        struct embedding_guard {
            engine_state & state;
            ~embedding_guard() {
                forget_engine_kv(state);
                llama_set_embeddings(state.ctx.get(), state.embeddings_default);
            }
        } cleanup { *state };
        llama_set_embeddings(ctx, true);
        forget_engine_kv(*state);
        int status = 0;
        {
            gen_detail::batch_guard guard(static_cast<int32_t>(n_tokens));
            llama_batch & batch = guard.batch;
            batch.n_tokens = static_cast<int32_t>(n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // Embeddings mode outputs every token regardless; no logits.
                batch.logits[i] = 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                set_error("llama_decode failed during embedding extraction");
                status = -1;
            }
        }
        if (status == 0) {
            const float * emb = llama_get_embeddings_seq(ctx, 0);
            if (!emb) {
                set_error("the decode produced no pooled embedding");
                status = -1;
            } else {
                std::memcpy(out_features, emb, n_embd * sizeof(float));
            }
        }
        return status;
    });
}

extern "C" int retro_engine_decision_info(retro_engine * engine, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
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

extern "C" int retro_engine_render_named_template(
        retro_engine * engine,
        const char * name,
        const char * context_json,
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
        return render_named_template(state->chat, name, context_json, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_engine_set_named_template(retro_engine * engine, const char * name, const char * source) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return set_named_template(state->chat, name, source);
    });
}

extern "C" int retro_engine_tokenize(
        retro_engine * engine,
        const char * text,
        bool add_special,
        bool parse_special,
        int32_t * tokens,
        size_t n_tokens_max,
        size_t * out_n_tokens) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        return tokenize_into(*state, text, add_special, parse_special, tokens, n_tokens_max, out_n_tokens);
    });
}

extern "C" int retro_engine_token_piece(
        retro_engine * engine,
        int32_t token,
        bool special,
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
        const llama_vocab * vocab = llama_model_get_vocab(state->model.get());
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
            set_error("token is outside the vocabulary");
            return -1;
        }
        return render_string_out(
                buffer, n_buffer, out_n_bytes, "failed to render the token",
                [&](char * destination, int32_t capacity) -> int32_t {
                    // llama_token_to_piece writes nothing and returns minus
                    // the size when the buffer is short.
                    char probe[1];
                    const int32_t n = llama_token_to_piece(
                            vocab, token, destination ? destination : probe,
                            destination ? capacity : 0, 0, special);
                    return n < 0 ? -n : n;
                });
    });
}

extern "C" int retro_engine_decide(
        retro_engine * engine,
        int32_t readout,
        const retro_decision_prompt * prompts,
        size_t n_prompts,
        size_t n_pack_max,
        float * out_scores,
        size_t n_scores_max) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        // Same discipline as an embedding: the active adapter applies, the KV
        // (and with it the prefix cache) is cleared before and after, and the
        // context's embeddings configuration is restored on every exit path.
        if (!apply_active_adapter(*state)) {
            return -1;
        }
        struct decide_guard {
            engine_state & state;
            ~decide_guard() {
                forget_engine_kv(state);
                llama_set_embeddings(state.ctx.get(), state.embeddings_default);
            }
        } cleanup { *state };
        forget_engine_kv(*state);
        // A prefix is shared by copying it between sequences, which a
        // partitioned pool supports only for whole sequences.
        const bool share_prefix = state->config.kv_unified != 0;
        return decide_prompts(state->model.get(), state->ctx.get(), readout, prompts, n_prompts,
                              n_pack_max, share_prefix, out_scores, n_scores_max);
    });
}

extern "C" int retro_engine_expert_banks(
        retro_engine * engine,
        retro_expert_bank_desc * out_banks,
        size_t n_max,
        size_t * out_count) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state) {
            return -1;
        }
        if (!out_count) {
            set_error("out_count is required");
            return -1;
        }
        *out_count = 0;
        expert_placement_probe probe;
        if (!collect_expert_banks(*state->model, probe)) {
            return -1;
        }
        return copy_expert_banks(probe, out_banks, n_max, out_count);
    });
}

extern "C" int retro_engine_expert_stats(
        retro_engine * engine,
        retro_expert_cache_stats * out_stats) {
    return boundary([&]() -> int {
        engine_state * state = checked_engine(engine);
        if (!state || !out_stats) {
            if (state) set_error("out_stats is required");
            return -1;
        }
        *out_stats = {};
        struct llama_moe_cache_stats stats {};
        llama_moe_cache_stats(state->ctx.get(), &stats);
        out_stats->small.hits = stats.small.hits;
        out_stats->small.misses = stats.small.misses;
        out_stats->small.bytes = stats.small.bytes;
        out_stats->large.hits = stats.large.hits;
        out_stats->large.misses = stats.large.misses;
        out_stats->large.bytes = stats.large.bytes;
        out_stats->copy.hits = stats.copy.hits;
        out_stats->copy.misses = stats.copy.misses;
        out_stats->copy.bytes = stats.copy.bytes;
        out_stats->n_groups = std::min(stats.n_groups, RETRO_EXPERT_CACHE_MAX_GROUPS);
        for (int32_t i = 0; i < out_stats->n_groups; ++i) {
            out_stats->groups[i].n_slots = stats.groups[i].n_slots;
            out_stats->groups[i].n_layers = stats.groups[i].n_layers;
            out_stats->groups[i].n_expert = stats.groups[i].n_expert;
            out_stats->groups[i].device_bytes = stats.groups[i].device_bytes;
        }
        return 0;
    });
}

extern "C" int retro_engine_expert_route_enable(retro_engine * engine, uint32_t sample_every) {
    return boundary([&]() -> int {
        auto * state = checked_engine(engine);
        if (!state) return -1;
        if (sample_every != 0) {
            expert_placement_probe probe;
            if (!collect_expert_banks(*state->model, probe)) return -1;
            if (!(probe.flags & RETRO_EXPERT_BANKS_VALIDATED)) {
                set_error("routing telemetry is not validated for architecture " + probe.arch);
                return -2;
            }
        }
        if (!llama_moe_route_enable(state->ctx.get(), sample_every)) {
            set_error("the context does not support routing telemetry");
            return -2;
        }
        return 0;
    });
}

extern "C" int retro_engine_expert_route_counts(
        retro_engine * engine, uint32_t layer, uint32_t * out, size_t n_expert) {
    return boundary([&]() -> int {
        auto * state = checked_engine(engine);
        if (!state) return -1;
        if (!out || !n_expert || llama_moe_route_counts(state->ctx.get(), layer, nullptr, 0) != n_expert) {
            set_error("routing counter buffer does not match the expert bank");
            return -2;
        }
        llama_moe_route_counts(state->ctx.get(), layer, out, n_expert);
        return 0;
    });
}

extern "C" int retro_engine_expert_route_paths(retro_engine * engine, uint64_t out[3]) {
    return boundary([&]() -> int {
        auto * state = checked_engine(engine);
        if (!state || !out) {
            if (state) set_error("routing paths output is required");
            return -1;
        }
        llama_moe_route_paths(state->ctx.get(), out);
        return 0;
    });
}

extern "C" int retro_engine_expert_route_reset(retro_engine * engine) {
    return boundary([&]() -> int {
        auto * state = checked_engine(engine);
        if (!state) return -1;
        llama_moe_route_reset(state->ctx.get());
        return 0;
    });
}

} // namespace retro
