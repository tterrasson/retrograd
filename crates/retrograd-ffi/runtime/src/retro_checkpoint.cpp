// Training-checkpoint state access.
// This file is deliberately format-free: it moves optimizer scalars, the
// marked parameter list, the persistent state slots, RNG state and a model
// signature across the FFI boundary and lets Rust own the on-disk
// representation. The adapter GGUF written next to a checkpoint stays a pure
// LoRA export.

#include "retro_runtime.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace retro {

namespace {

// The optimizer context only exists after llama_opt_init. Callers that merely
// read state must tolerate its absence (a checkpoint taken before the first
// step legitimately has no slots), so this returns null without an error.
ggml_opt_context_t opt_context(trainer_state & state) {
    if (!state.opt_created || !state.ctx) {
        return nullptr;
    }
    return llama_opt_context(state.ctx.get());
}

// One persistent optimizer tensor and the pair of names that identify it.
// Built from whatever the live optimizer declares, so the table is empty for
// an optimizer that keeps no state and would gain rows - not a new shape -
// for one that keeps block-shaped or shared state.
struct slot_ref {
    std::string owner;
    std::string slot;
    ggml_tensor * tensor;
};

// The per-parameter slots of the live optimizer, in enumeration order:
// a parameter's slots together, parameters in the order the optimizer holds
// them. AdamW contributes "m" and "v" per parameter; SGD contributes nothing.
//
// Reading the table through ggml's momenta accessors is the only shape
// available today, and it is deliberately the only place that assumes it: the
// FFI above and the checkpoint above that see slots and never a pair.
std::vector<slot_ref> parameter_slots(trainer_state & state) {
    std::vector<slot_ref> slots;
    ggml_opt_context_t opt = opt_context(state);
    if (!opt) {
        return slots;
    }
    const int64_t count = ggml_opt_momenta_count(opt);
    slots.reserve(static_cast<size_t>(count) * 2);
    for (int64_t i = 0; i < count; ++i) {
        const char * name = ggml_opt_momenta_name(opt, i);
        ggml_tensor * m = ggml_opt_momenta_m(opt, i);
        ggml_tensor * v = ggml_opt_momenta_v(opt, i);
        if (!name || !m || !v) {
            continue;
        }
        slots.push_back(slot_ref { name, "m", m });
        slots.push_back(slot_ref { name, "v", v });
    }
    return slots;
}

// State an optimizer keeps once rather than once per parameter - a codebook,
// a shared second moment. Neither AdamW nor SGD has any, so the table is empty
// and the enumeration exists to be enumerable: a checkpoint that round-trips
// an empty shared scope is a checkpoint that will round-trip a full one.
std::vector<slot_ref> shared_slots(trainer_state & state) {
    (void) state;
    return {};
}

bool slot_table(trainer_state & state, int32_t scope, std::vector<slot_ref> & out) {
    switch (scope) {
        case RETRO_SLOT_SCOPE_PARAMETER:
            out = parameter_slots(state);
            return true;
        case RETRO_SLOT_SCOPE_SHARED:
            out = shared_slots(state);
            return true;
        default:
            set_error("optimizer slot scope must be parameter or shared");
            return false;
    }
}

// Every marked parameter, in the order the resolved trainable set publishes:
// adapter factors first, in the adapter's registration order, then base
// tensors by name. Empty before the optimizer graph exists - a flag is set by
// llama_opt_init, so asking earlier would answer about a set nobody built.
std::vector<const ggml_tensor *> marked_parameters(trainer_state & state) {
    std::vector<const ggml_tensor *> marked;
    if (!opt_context(state) || !state.model) {
        return marked;
    }
    if (state.adapter) {
        for (const auto & item : state.adapter->ab_map) {
            if (is_param_tensor(item.second.a)) {
                marked.push_back(item.second.a);
            }
            if (is_param_tensor(item.second.b)) {
                marked.push_back(item.second.b);
            }
        }
    }
    const size_t adapter_end = marked.size();
    for (const auto & item : state.model->tensors_by_name) {
        if (is_param_tensor(item.second)) {
            marked.push_back(item.second);
        }
    }
    std::sort(
            marked.begin() + static_cast<std::ptrdiff_t>(adapter_end),
            marked.end(),
            [](const ggml_tensor * left, const ggml_tensor * right) {
                return std::strcmp(left->name, right->name) < 0;
            });
    return marked;
}

// A byte range is inside the payload, or it is an error. Never a short read:
// a caller that staged fewer bytes than it asked for would write a truncated
// slot into a checkpoint and call it complete.
bool slot_range_ok(const ggml_tensor * tensor, uint64_t offset, size_t n_bytes) {
    const uint64_t total = static_cast<uint64_t>(ggml_nbytes(tensor));
    if (offset > total || static_cast<uint64_t>(n_bytes) > total - offset) {
        set_error("optimizer slot byte range is outside the payload");
        return false;
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
        out_state->graph_ready = opt != nullptr;
        // Before the graph exists there is no context to ask, so the answer is
        // the configured optimizer rather than a default that would record
        // AdamW for an SGD run whose first step has not happened yet.
        out_state->optimizer = opt
                ? static_cast<int32_t>(ggml_opt_context_optimizer_type(opt))
                : state->train_config.optimizer;
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
        // An optimizer the run cannot build is refused rather than resumed onto
        // a different trajectory: the momenta of an AdamW checkpoint mean
        // nothing to an SGD step, and an SGD checkpoint resumed under AdamW
        // would start with cold moments and a warm iteration counter.
        if (saved->graph_ready && saved->optimizer != state->train_config.optimizer) {
            retro::set_error(
                    "the checkpoint was written by a different optimizer than this run "
                    "configures; resume with the optimizer the run used");
            return -1;
        }
        // The iteration counter belongs to the graph. Restoring it into a cold
        // optimizer would corrupt the first bias correction - but "no momenta"
        // is not "no graph": an optimizer with zero slots still counts steps.
        if (saved->graph_ready) {
            ggml_opt_context_t opt = retro::opt_context(*state);
            if (!opt) {
                retro::set_error(
                        "cannot restore an optimizer iteration before the optimizer graph "
                        "exists; call retro_trainer_prepare_optimizer first");
                return -1;
            }
            if (saved->has_momenta && ggml_opt_momenta_count(opt) == 0) {
                retro::set_error(
                        "the checkpoint carries momenta but this optimizer keeps none");
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
        if (!state->has_lora && !retro::trains_base_weights(*state)) {
            retro::set_error("create or load a LoRA adapter before preparing the optimizer");
            return -1;
        }
        // The preflight run by this call builds the full optimizer graph once,
        // which is what allocates and zeroes any per-parameter state.
        if (!retro::ensure_optimizer_initialized(*state)) {
            return -1;
        }
        ggml_opt_context_t opt = retro::opt_context(*state);
        if (!opt) {
            retro::set_error("the optimizer graph was not built");
            return -1;
        }
        // No assertion on the momenta count: an optimizer that keeps no state
        // is prepared once its graph exists, and demanding momenta here is
        // exactly the confusion between "zero slots" and "uninitialized".
        return 0;
    });
}

extern "C" int retro_trainer_marked_parameter_count(
        retro_trainer * trainer,
        size_t * out_count) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_count) {
            retro::set_error("out_count is required");
            return -1;
        }
        *out_count = retro::marked_parameters(*state).size();
        return 0;
    });
}

extern "C" int retro_trainer_marked_parameter_info(
        retro_trainer * trainer,
        size_t index,
        retro_tensor_desc * out_tensor) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_tensor) {
            retro::set_error("out_tensor is required");
            return -1;
        }
        const std::vector<const ggml_tensor *> marked = retro::marked_parameters(*state);
        if (index >= marked.size()) {
            retro::set_error("marked parameter index is out of range");
            return -1;
        }
        const ggml_tensor * tensor = marked[index];
        *out_tensor = retro_tensor_desc {};
        const char * type_name = ggml_type_name(tensor->type);
        if (!retro::copy_fixed_field(tensor->name, out_tensor->name, sizeof(out_tensor->name))
                || !retro::copy_fixed_field(
                        type_name ? type_name : "",
                        out_tensor->type_name,
                        sizeof(out_tensor->type_name))) {
            return -1;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            out_tensor->ne[d] = tensor->ne[d];
        }
        out_tensor->n_elements = static_cast<uint64_t>(ggml_nelements(tensor));
        out_tensor->n_bytes = static_cast<uint64_t>(ggml_nbytes(tensor));
        // Same contract as the inventory's: the data address is the allocation
        // identity, and 0 means "no identity recorded" rather than "shared".
        out_tensor->storage_id = tensor->data
                ? reinterpret_cast<uint64_t>(tensor->data)
                : 0;
        return 0;
    });
}

extern "C" int retro_trainer_state_slot_count(
        retro_trainer * trainer,
        size_t * out_parameter_slots,
        size_t * out_shared_slots) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (out_parameter_slots) {
            *out_parameter_slots = retro::parameter_slots(*state).size();
        }
        if (out_shared_slots) {
            *out_shared_slots = retro::shared_slots(*state).size();
        }
        return 0;
    });
}

extern "C" int retro_trainer_state_slot_info(
        retro_trainer * trainer,
        int32_t scope,
        size_t index,
        retro_optimizer_slot * out_slot) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_slot) {
            retro::set_error("out_slot is required");
            return -1;
        }
        std::vector<retro::slot_ref> slots;
        if (!retro::slot_table(*state, scope, slots)) {
            return -1;
        }
        if (index >= slots.size()) {
            retro::set_error("optimizer slot index is out of range");
            return -1;
        }
        const retro::slot_ref & entry = slots[index];
        *out_slot = retro_optimizer_slot {};
        // A truncated owner is not a shorter name, it is a name that would
        // restore into the wrong parameter. Same contract as the inventory.
        if (!retro::copy_fixed_field(entry.owner, out_slot->owner, sizeof(out_slot->owner))
                || !retro::copy_fixed_field(entry.slot, out_slot->slot, sizeof(out_slot->slot))) {
            return -1;
        }
        const char * type_name = ggml_type_name(entry.tensor->type);
        if (!retro::copy_fixed_field(
                    type_name ? type_name : "", out_slot->type_name, sizeof(out_slot->type_name))) {
            return -1;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            out_slot->ne[d] = entry.tensor->ne[d];
        }
        out_slot->n_elements = static_cast<uint64_t>(ggml_nelements(entry.tensor));
        out_slot->n_bytes = static_cast<uint64_t>(ggml_nbytes(entry.tensor));
        return 0;
    });
}

extern "C" int retro_trainer_state_slot_read(
        retro_trainer * trainer,
        int32_t scope,
        size_t index,
        uint64_t offset,
        void * out_bytes,
        size_t n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (n_bytes != 0 && !out_bytes) {
            retro::set_error("out_bytes is required");
            return -1;
        }
        std::vector<retro::slot_ref> slots;
        if (!retro::slot_table(*state, scope, slots)) {
            return -1;
        }
        if (index >= slots.size()) {
            retro::set_error("optimizer slot index is out of range");
            return -1;
        }
        ggml_tensor * tensor = slots[index].tensor;
        if (!retro::slot_range_ok(tensor, offset, n_bytes)) {
            return -1;
        }
        if (n_bytes != 0) {
            ggml_backend_tensor_get(tensor, out_bytes, offset, n_bytes);
        }
        return 0;
    });
}

extern "C" int retro_trainer_state_slot_write(
        retro_trainer * trainer,
        int32_t scope,
        const char * owner,
        const char * slot,
        uint64_t offset,
        const void * bytes,
        size_t n_bytes) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(owner) || retro::is_blank(slot)) {
            retro::set_error("owner and slot are required");
            return -1;
        }
        if (n_bytes != 0 && !bytes) {
            retro::set_error("bytes is required");
            return -1;
        }
        std::vector<retro::slot_ref> slots;
        if (!retro::slot_table(*state, scope, slots)) {
            return -1;
        }
        for (const retro::slot_ref & entry : slots) {
            if (entry.owner != owner || entry.slot != slot) {
                continue;
            }
            if (!retro::slot_range_ok(entry.tensor, offset, n_bytes)) {
                return -1;
            }
            if (n_bytes != 0) {
                ggml_backend_tensor_set(entry.tensor, bytes, offset, n_bytes);
            }
            return 0;
        }
        retro::set_error(
                std::string("the optimizer keeps no slot '") + slot + "' for '" + owner + "'");
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
