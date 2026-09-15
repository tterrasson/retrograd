#include "retro_runtime.hpp"

#include "log.h"

#include <cmath>
#include <sstream>

namespace retro {

thread_local std::string g_last_error;

void llama_model_deleter::operator()(llama_model * model) const {
    if (model) {
        llama_model_free(model);
    }
}

void llama_context_deleter::operator()(llama_context * ctx) const {
    if (ctx) {
        llama_free(ctx);
    }
}

void llama_adapter_lora_deleter::operator()(llama_adapter_lora * adapter) const {
    if (adapter) {
        llama_adapter_lora_free(adapter);
    }
}

void ggml_opt_dataset_deleter::operator()(ggml_opt_dataset * dataset) const {
    if (dataset) {
        ggml_opt_dataset_free(dataset);
    }
}

void ggml_opt_result_deleter::operator()(ggml_opt_result * result) const {
    if (result) {
        ggml_opt_result_free(result);
    }
}

void gguf_context_deleter::operator()(gguf_context * ctx) const {
    if (ctx) {
        gguf_free(ctx);
    }
}

void set_error(std::string message) {
    g_last_error = std::move(message);
}

void clear_error() {
    g_last_error.clear();
}

bool is_blank(const char * value) {
    return value == nullptr || value[0] == '\0';
}

void ensure_backend_initialized() {
    static const bool initialized = [] {
        llama_backend_init();
        return true;
    }();
    (void) initialized;
}

namespace {

void filtered_runtime_log_callback(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

} // namespace

void configure_runtime_logging(bool verbose) {
    if (verbose) {
        llama_log_set(nullptr, nullptr);
        ggml_log_set(nullptr, nullptr);
    } else {
        llama_log_set(filtered_runtime_log_callback, nullptr);
        ggml_log_set(filtered_runtime_log_callback, nullptr);
    }
    common_log_set_verbosity_thold(verbose ? LOG_DEFAULT_LLAMA : LOG_LEVEL_ERROR);
}

retro_train_config default_train_config() {
    retro_train_config config {};
    config.n_ctx = 128;
    config.n_batch = 128;
    config.n_ubatch = 32;
    config.n_seq_max = 1;
    config.generation_concurrency = 1;
    config.fast_generation_context = false;
    config.kv_dtype = RETRO_KV_DTYPE_F32;
    config.threads = 0;
    config.epochs = 1;
    config.learning_rate = 1.0e-4f;
    config.weight_decay = 0.0f;
    config.max_grad_norm = 1.0f;
    config.lr_scheduler = 0;
    config.warmup_steps = 0;
    config.verbose = false;
    config.device = 0;
    config.chunked_cross_entropy = false;
    config.chunked_ce_tiles = 8;
    config.chunked_ce_seq_chunk = 0;
    config.chunked_ce_offload_logsoftmax = false;
    config.gradient_checkpointing = false;
    config.checkpoint_every_n_layers = 1;
    config.checkpoint_dtype = RETRO_CHECKPOINT_DTYPE_F32;
    config.require_gpu_resident = false;
    // Keep the default aligned with the Rust configuration so omitted configs
    // still shuffle training rows.
    config.shuffle_dataset = true;
    config.shuffle_seed = 42;
    return config;
}

const char * device_kind_name(int32_t device) {
    switch (device) {
        case RETRO_DEVICE_AUTO: return "auto";
        case RETRO_DEVICE_CPU:  return "cpu";
        case RETRO_DEVICE_GPU:  return "gpu";
        default:                return "unknown";
    }
}

bool validate_train_config(const retro_train_config & config) {
    if (config.n_ctx == 0) {
        set_error("n_ctx must be greater than zero");
        return false;
    }
    if (config.kv_dtype != RETRO_KV_DTYPE_F32 && config.kv_dtype != RETRO_KV_DTYPE_F16) {
        set_error("kv_dtype must be f32 or f16");
        return false;
    }
    if (config.n_batch == 0) {
        set_error("n_batch must be greater than zero");
        return false;
    }
    if (config.n_ubatch == 0) {
        set_error("n_ubatch must be greater than zero");
        return false;
    }
    if (config.n_seq_max == 0 || config.n_seq_max > 256) {
        set_error("n_seq_max must be between 1 and 256");
        return false;
    }
    if (config.n_seq_max > config.n_batch) {
        set_error("n_seq_max must not exceed n_batch");
        return false;
    }
    if (config.generation_concurrency == 0 || config.generation_concurrency > 256) {
        set_error("generation_concurrency must be between 1 and 256");
        return false;
    }
    if (config.generation_concurrency > config.n_batch) {
        set_error("generation_concurrency must not exceed n_batch");
        return false;
    }
    if (config.n_ctx > UINT32_MAX / config.generation_concurrency) {
        set_error("n_ctx * generation_concurrency overflows uint32_t");
        return false;
    }
    if (config.n_batch % config.n_ubatch != 0) {
        set_error("n_batch must be divisible by n_ubatch");
        return false;
    }
    if (config.n_ctx % config.n_batch != 0) {
        set_error("n_ctx must be divisible by n_batch");
        return false;
    }
    if (config.epochs == 0) {
        set_error("epochs must be greater than zero");
        return false;
    }
    if (config.learning_rate <= 0.0f) {
        set_error("learning_rate must be greater than zero");
        return false;
    }
    if (!std::isfinite(config.max_grad_norm) || config.max_grad_norm <= 0.0f) {
        set_error("max_grad_norm must be finite and greater than zero");
        return false;
    }
    if (config.lr_scheduler < 0 || config.lr_scheduler > 2) {
        set_error("lr_scheduler must be constant, linear, or cosine");
        return false;
    }
    if (config.checkpoint_every_n_layers == 0) {
        set_error("checkpoint_every_n_layers must be greater than zero");
        return false;
    }
    if (config.checkpoint_dtype != RETRO_CHECKPOINT_DTYPE_F32 &&
            config.checkpoint_dtype != RETRO_CHECKPOINT_DTYPE_F16 &&
            config.checkpoint_dtype != RETRO_CHECKPOINT_DTYPE_BF16) {
        set_error("checkpoint_dtype must be f32, f16, or bf16");
        return false;
    }
    return true;
}

trainer_state * checked(retro_trainer * trainer) {
    if (!trainer) {
        set_error("trainer is null");
        return nullptr;
    }
    return reinterpret_cast<trainer_state *>(trainer);
}

std::string join_patterns(const std::vector<std::string> & patterns) {
    std::ostringstream out;
    for (size_t i = 0; i < patterns.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << patterns[i];
    }
    return out.str();
}

int copy_string_out(
        const std::string & text, char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    if (!out_n_bytes) {
        set_error("out_n_bytes is required");
        return -1;
    }
    *out_n_bytes = text.size();
    if (!buffer || n_buffer == 0) {
        return 0;
    }
    if (n_buffer <= text.size()) {
        set_error("output buffer is too small");
        return -2;
    }
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return 0;
}

} // namespace retro
