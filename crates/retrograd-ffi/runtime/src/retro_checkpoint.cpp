// Training-checkpoint state access.
// This file is deliberately format-free: it moves optimizer scalars, AdamW
// momenta, RNG state and a model signature across the FFI boundary and lets
// Rust own the on-disk representation. The adapter GGUF written next to a
// checkpoint stays a pure LoRA export.

#include "retro_runtime.hpp"

#include <sstream>

namespace retro {

namespace {

// The optimizer context only exists after llama_opt_init. Callers that merely
// read state must tolerate its absence (a checkpoint taken before the first
// step legitimately has no momenta), so this returns null without an error.
ggml_opt_context_t opt_context(trainer_state & state) {
    if (!state.opt_created || !state.ctx) {
        return nullptr;
    }
    return llama_opt_context(state.ctx.get());
}

// Momenta live in the optimizer's static buffer, allocated by the first graph
// build. Before that the count is zero and there is nothing to save.
ggml_opt_context_t momenta_context(trainer_state & state) {
    ggml_opt_context_t opt = opt_context(state);
    if (!opt || ggml_opt_momenta_count(opt) == 0) {
        return nullptr;
    }
    return opt;
}

bool momenta_entry(
        trainer_state & state,
        size_t index,
        ggml_tensor ** out_m,
        ggml_tensor ** out_v,
        const char ** out_name) {
    ggml_opt_context_t opt = momenta_context(state);
    if (!opt) {
        set_error("the optimizer has no momenta yet; call retro_trainer_prepare_optimizer first");
        return false;
    }
    if (index >= static_cast<size_t>(ggml_opt_momenta_count(opt))) {
        set_error("momenta index is out of range");
        return false;
    }
    const int64_t i = static_cast<int64_t>(index);
    ggml_tensor * m = ggml_opt_momenta_m(opt, i);
    ggml_tensor * v = ggml_opt_momenta_v(opt, i);
    const char * name = ggml_opt_momenta_name(opt, i);
    if (!m || !v || !name) {
        set_error("momenta entry is incomplete");
        return false;
    }
    if (out_m) {
        *out_m = m;
    }
    if (out_v) {
        *out_v = v;
    }
    if (out_name) {
        *out_name = name;
    }
    return true;
}

} // namespace

std::string model_signature(const trainer_state & state) {
    if (!state.model) {
        return "unloaded";
    }
    const llama_model * model = state.model.get();
    std::ostringstream out;
    out << "arch=" << state.model->arch_name()
        << " n_embd=" << llama_model_n_embd(model)
        << " n_layer=" << llama_model_n_layer(model)
        << " n_vocab=" << llama_vocab_n_tokens(llama_model_get_vocab(model))
        << " n_params=" << llama_model_n_params(model);
    return out.str();
}

} // namespace retro

extern "C" int retro_trainer_optimizer_state(
        retro_trainer * trainer,
        retro_optimizer_state * out_state) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_state) {
            retro::set_error("out_state is required");
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        *out_state = retro_optimizer_state {};
        out_state->iter = opt ? ggml_opt_iter(opt) : 1;
        out_state->has_momenta = opt && ggml_opt_momenta_count(opt) > 0;
        out_state->optimizer = opt
                ? static_cast<int32_t>(ggml_opt_context_optimizer_type(opt))
                : 0;
        out_state->learning_rate = state->train_config.learning_rate;
        out_state->weight_decay = state->train_config.weight_decay;
        out_state->max_grad_norm = state->train_config.max_grad_norm;
        out_state->scheduler_step = state->scheduler_step;
        out_state->scheduler_total_steps = state->scheduler_total_steps;
        out_state->last_learning_rate = state->last_learning_rate;
        return 0;
    });
}

extern "C" int retro_trainer_restore_optimizer_state(
        retro_trainer * trainer,
        const retro_optimizer_state * saved) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!saved) {
            retro::set_error("state is required");
            return -1;
        }
        if (saved->iter < 1) {
            retro::set_error("optimizer iter must be at least 1");
            return -1;
        }
        // The momenta counter only makes sense once the graph exists; restoring
        // it into a cold optimizer would corrupt the first bias correction.
        if (saved->has_momenta) {
            ggml_opt_context_t opt = retro::opt_context(*state);
            if (!opt || ggml_opt_momenta_count(opt) == 0) {
                retro::set_error(
                        "cannot restore an optimizer iteration without momenta; "
                        "call retro_trainer_prepare_optimizer first");
                return -1;
            }
            ggml_opt_set_iter(opt, saved->iter);
        }
        state->scheduler_step = saved->scheduler_step;
        state->scheduler_total_steps = saved->scheduler_total_steps;
        state->last_learning_rate = saved->last_learning_rate;
        return 0;
    });
}

extern "C" int retro_trainer_set_resume_point(
        retro_trainer * trainer,
        uint32_t completed_epochs) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (completed_epochs >= state->train_config.epochs) {
            retro::set_error("the resume point is at or past the configured number of epochs");
            return -1;
        }
        state->resume_epoch = completed_epochs;
        state->resume_active = true;
        return 0;
    });
}

extern "C" int retro_trainer_prepare_optimizer(retro_trainer * trainer) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora) {
            retro::set_error("create or load a LoRA adapter before preparing the optimizer");
            return -1;
        }
        // The preflight run by this call builds the full optimizer graph once,
        // which is what allocates and zeroes the momenta.
        if (!retro::ensure_lora_optimizer_initialized(*state)) {
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        if (!opt || ggml_opt_momenta_count(opt) == 0) {
            retro::set_error("the optimizer graph did not allocate AdamW momenta");
            return -1;
        }
        return 0;
    });
}

extern "C" int retro_trainer_momenta_count(retro_trainer * trainer, size_t * out_count) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_count) {
            retro::set_error("out_count is required");
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        *out_count = opt ? static_cast<size_t>(ggml_opt_momenta_count(opt)): 0;
        return 0;
    });
}

extern "C" int retro_trainer_momenta_info(
        retro_trainer * trainer,
        size_t index,
        char * name_buffer,
        size_t n_buffer,
        size_t * out_n_bytes,
        int64_t * out_ne,
        size_t * out_n_elements) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes || !out_ne || !out_n_elements) {
            retro::set_error("out_n_bytes, out_ne, and out_n_elements are required");
            return -1;
        }
        ggml_tensor * m = nullptr;
        const char * name = nullptr;
        if (!retro::momenta_entry(*state, index, &m, nullptr, &name)) {
            return -1;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            out_ne[i] = m->ne[i];
        }
        *out_n_elements = static_cast<size_t>(ggml_nelements(m));
        return retro::copy_string_out(name, name_buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_momenta_read(
        retro_trainer * trainer,
        size_t index,
        float * out_m,
        float * out_v,
        size_t n_values) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_m || !out_v) {
            retro::set_error("out_m and out_v are required");
            return -1;
        }
        ggml_tensor * m = nullptr;
        ggml_tensor * v = nullptr;
        if (!retro::momenta_entry(*state, index, &m, &v, nullptr)) {
            return -1;
        }
        if (n_values != static_cast<size_t>(ggml_nelements(m))) {
            retro::set_error("momenta buffer length does not match the parameter");
            return -1;
        }
        ggml_backend_tensor_get(m, out_m, 0, n_values * sizeof(float));
        ggml_backend_tensor_get(v, out_v, 0, n_values * sizeof(float));
        return 0;
    });
}

extern "C" int retro_trainer_momenta_write(
        retro_trainer * trainer,
        const char * name,
        const float * m_values,
        const float * v_values,
        size_t n_values) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(name) || !m_values || !v_values) {
            retro::set_error("name, m, and v are required");
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        const int64_t count = opt ? ggml_opt_momenta_count(opt) : 0;
        if (count == 0) {
            retro::set_error(
                    "the optimizer has no momenta yet; "
                    "call retro_trainer_prepare_optimizer first");
            return -1;
        }
        for (int64_t i = 0; i < count; ++i) {
            const char * candidate = ggml_opt_momenta_name(opt, i);
            if (!candidate || std::strcmp(candidate, name) != 0) {
                continue;
            }
            ggml_tensor * m = ggml_opt_momenta_m(opt, i);
            ggml_tensor * v = ggml_opt_momenta_v(opt, i);
            if (n_values != static_cast<size_t>(ggml_nelements(m))) {
                retro::set_error(
                        std::string("momenta length does not match parameter '") + name + "'");
                return -1;
            }
            ggml_backend_tensor_set(m, m_values, 0, n_values * sizeof(float));
            ggml_backend_tensor_set(v, v_values, 0, n_values * sizeof(float));
            return 0;
        }
        retro::set_error(std::string("no trainable parameter named '") + name + "'");
        return -1;
    });
}

extern "C" int retro_trainer_rng_state(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes) {
            retro::set_error("out_n_bytes is required");
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        if (!opt) {
            retro::set_error("the optimizer context does not exist yet");
            return -1;
        }
        const size_t needed = ggml_opt_rng_state(opt, nullptr, 0);
        std::string rendered(needed, '\0');
        ggml_opt_rng_state(opt, rendered.data(), needed + 1);
        return retro::copy_string_out(rendered, buffer, n_buffer, out_n_bytes);
    });
}

extern "C" int retro_trainer_set_rng_state(retro_trainer * trainer, const char * rng_state) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(rng_state)) {
            retro::set_error("rng state is required");
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        if (!opt) {
            retro::set_error("the optimizer context does not exist yet");
            return -1;
        }
        if (!ggml_opt_set_rng_state(opt, rng_state)) {
            retro::set_error("the saved RNG state is not a valid mt19937 state");
            return -1;
        }
        return 0;
    });
}

extern "C" int retro_trainer_model_signature(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_n_bytes) {
            retro::set_error("out_n_bytes is required");
            return -1;
        }
        return retro::copy_string_out(
                retro::model_signature(*state), buffer, n_buffer, out_n_bytes);
    });
}
