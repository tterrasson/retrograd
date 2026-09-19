// The trainable bundle: the base tensors a run trained, by absolute value.
//
// A LoRA run leaves an adapter behind; a base-weight run leaves the weights
// themselves, and there is no adapter-shaped record that can stand in for
// them. This file writes and reads that record as a plain GGUF, which is the
// format the rest of the stack already knows how to open - but never as an
// adapter: it carries no `adapter.type`, and llama_adapter_lora_init() refuses
// it, which is what keeps the two kinds from being confused for one another.
//
// Values, not deltas. A delta is only interpretable beside the exact GGUF it
// was subtracted from, while the values are what the run produced; the base
// model's fingerprint is recorded by the caller and compared on restore, so
// the pairing is checked rather than assumed.
//
// The third writer is the standalone model export: the source file's metadata
// and tensor list, copied rather than reconstructed, with the live values in
// place of the stored ones. Copying the metadata is what makes the result
// loadable by whatever loaded the input.

#include "retro_runtime.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace retro {

namespace {

// The bundle's own schema version, separate from the checkpoint format that
// carries it: the file is also an export target, and a reader that finds an
// unknown version must refuse rather than guess at the tensor list.
constexpr uint32_t TRAINABLE_BUNDLE_VERSION = 1;
constexpr const char * TRAINABLE_BUNDLE_TYPE = "retrograd.trainable";

ggml_tensor * model_tensor(const trainer_state & state, const std::string & name) {
    for (const auto & item : state.model->tensors_by_name) {
        if (item.first == name) {
            return item.second;
        }
    }
    return nullptr;
}

// The resolved set, or a refusal naming why there is nothing to write. Both
// conditions are the same mistake seen from two sides: a bundle for a run that
// trained no base tensor would claim a result the run does not have.
bool resolved_base_set(trainer_state & state, std::vector<std::string> & out) {
    if (!trains_base_weights(state)) {
        set_error(
                "this run trains no base tensor, so it has no trainable bundle to write; "
                "a LoRA run's result is its adapter");
        return false;
    }
    if (!state.trainable_base_set || state.trainable_base.empty()) {
        set_error("the trainable set has not been resolved yet");
        return false;
    }
    out = state.trainable_base;
    return true;
}

} // namespace

bool save_trainable_bundle(trainer_state & state, const char * path) {
    std::vector<std::string> names;
    if (!resolved_base_set(state, names)) {
        return false;
    }

    // Host copies, because a tensor may live on a device and gguf_add_tensor
    // reads `->data` directly. The context is sized for exactly this set, so
    // the staging cost is the trainable bytes and never the model's.
    size_t payload = 0;
    for (const std::string & name : names) {
        ggml_tensor * tensor = model_tensor(state, name);
        if (!tensor) {
            set_error(std::string("the model declares no tensor named '") + name + "'");
            return false;
        }
        payload += ggml_nbytes(tensor) + GGML_MEM_ALIGN;
    }
    ggml_init_params params {};
    params.mem_size = payload + ggml_tensor_overhead() * names.size() + GGML_MEM_ALIGN;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * staging = ggml_init(params);
    if (!staging) {
        set_error("failed to allocate the trainable bundle staging context");
        return false;
    }

    gguf_context_ptr gguf(gguf_init_empty());
    if (!gguf) {
        ggml_free(staging);
        set_error("failed to allocate the trainable bundle GGUF context");
        return false;
    }
    gguf_set_val_str(gguf.get(), "general.type", TRAINABLE_BUNDLE_TYPE);
    gguf_set_val_str(gguf.get(), "general.architecture", state.model->arch_name().c_str());
    gguf_set_val_u32(gguf.get(), "trainable.version", TRAINABLE_BUNDLE_VERSION);

    for (const std::string & name : names) {
        ggml_tensor * source = model_tensor(state, name);
        ggml_tensor * copy = ggml_new_tensor(
                staging, source->type, GGML_MAX_DIMS, source->ne);
        if (!copy) {
            ggml_free(staging);
            set_error(std::string("failed to stage trainable tensor '") + name + "'");
            return false;
        }
        ggml_set_name(copy, name.c_str());
        ggml_backend_tensor_get(source, copy->data, 0, ggml_nbytes(source));
        gguf_add_tensor(gguf.get(), copy);
    }

    const bool written = gguf_write_to_file(gguf.get(), path, false);
    ggml_free(staging);
    if (!written) {
        set_error("failed to write the trainable bundle: " + std::string(path));
        return false;
    }
    return true;
}

bool load_trainable_bundle(trainer_state & state, const char * path) {
    std::vector<std::string> names;
    if (!resolved_base_set(state, names)) {
        return false;
    }

    ggml_context * data = nullptr;
    gguf_init_params params {};
    params.no_alloc = false;
    params.ctx = &data;
    gguf_context_ptr gguf(gguf_init_from_file(path, params));
    if (!gguf) {
        set_error("failed to read the trainable bundle: " + std::string(path));
        return false;
    }
    struct data_guard {
        ggml_context * ctx;
        ~data_guard() { if (ctx) { ggml_free(ctx); } }
    } guard { data };

    const int64_t type_key = gguf_find_key(gguf.get(), "general.type");
    if (type_key < 0 || gguf_get_kv_type(gguf.get(), type_key) != GGUF_TYPE_STRING
            || std::string(gguf_get_val_str(gguf.get(), type_key)) != TRAINABLE_BUNDLE_TYPE) {
        set_error(
                "this file is not a trainable bundle: " + std::string(path)
                + " (a LoRA adapter is loaded with retro_trainer_load_lora)");
        return false;
    }
    const int64_t version_key = gguf_find_key(gguf.get(), "trainable.version");
    if (version_key >= 0 && gguf_get_kv_type(gguf.get(), version_key) != GGUF_TYPE_UINT32) {
        set_error("trainable.version must be a uint32");
        return false;
    }
    const uint32_t version = version_key < 0
            ? 0
            : gguf_get_val_u32(gguf.get(), version_key);
    if (version != TRAINABLE_BUNDLE_VERSION) {
        std::ostringstream message;
        message << "trainable bundle " << path << " has schema version " << version
                << ", expected " << TRAINABLE_BUNDLE_VERSION;
        set_error(message.str());
        return false;
    }

    // Set equality both ways, before any write. A bundle missing a tensor the
    // run trains would resume from a model that is neither the checkpoint's
    // nor the base's, and a bundle carrying one the run does not train would
    // silently move a weight nobody selected.
    const int64_t n_tensors = gguf_get_n_tensors(gguf.get());
    std::vector<std::string> in_file;
    in_file.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        in_file.emplace_back(gguf_get_tensor_name(gguf.get(), i));
    }
    std::vector<std::string> expected = names;
    std::sort(expected.begin(), expected.end());
    std::vector<std::string> found = in_file;
    std::sort(found.begin(), found.end());
    if (expected != found) {
        std::vector<std::string> missing;
        std::vector<std::string> extra;
        std::set_difference(
                expected.begin(), expected.end(), found.begin(), found.end(),
                std::back_inserter(missing));
        std::set_difference(
                found.begin(), found.end(), expected.begin(), expected.end(),
                std::back_inserter(extra));
        std::ostringstream message;
        message << "the trainable bundle does not match this run's trainable set";
        if (!missing.empty()) {
            message << "; missing (" << missing.size() << "): [" << join_patterns(missing) << "]";
        }
        if (!extra.empty()) {
            message << "; unexpected (" << extra.size() << "): [" << join_patterns(extra) << "]";
        }
        set_error(message.str());
        return false;
    }

    for (const std::string & name : in_file) {
        ggml_tensor * source = ggml_get_tensor(data, name.c_str());
        ggml_tensor * target = model_tensor(state, name);
        if (!source || !target) {
            set_error(std::string("the trainable bundle entry '") + name + "' has no data");
            return false;
        }
        if (source->type != target->type) {
            set_error(
                    std::string("trainable bundle entry '") + name + "' is "
                    + ggml_type_name(source->type) + " and the model's tensor is "
                    + ggml_type_name(target->type));
            return false;
        }
        if (!ggml_are_same_shape(source, target)) {
            set_error(
                    std::string("trainable bundle entry '") + name
                    + "' has a different shape than the model's tensor");
            return false;
        }
    }

    // Validate every entry before changing any live weight.
    for (const std::string & name : in_file) {
        ggml_tensor * source = ggml_get_tensor(data, name.c_str());
        ggml_tensor * target = model_tensor(state, name);
        ggml_backend_tensor_set(target, source->data, 0, ggml_nbytes(source));
    }

    // The weights just moved, so every resident prefix was decoded under
    // weights that no longer exist - the same reason a training step forgets
    // them.
    state.forget_generation_kv();
    state.invalidate_report_caches();
    return true;
}

bool save_model_gguf(trainer_state & state, const char * path) {
    // Without changed base weights, the file would be a copy of the input.
    if (!trains_base_weights(state)) {
        set_error(
                "this run trains no base tensor, so a model export would be a copy of the "
                "model it was loaded from; a LoRA run's result is its adapter");
        return false;
    }
    // An adapter does not live in the weights, and folding it in is a merge
    // this build does not do.
    if (state.has_lora) {
        set_error(
                "this run carries a LoRA adapter, which a standalone model GGUF cannot hold: "
                "merging an adapter into the weights it multiplies is not implemented, so "
                "export the composite trainable bundle instead");
        return false;
    }

    // Re-read the source file for its metadata; it is copied, not reconstructed.
    ggml_context * meta = nullptr;
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context_ptr source(gguf_init_from_file(state.model_path.c_str(), params));
    if (!source) {
        set_error("failed to re-read the model file for its metadata: " + state.model_path);
        return false;
    }
    struct meta_guard {
        ggml_context * ctx;
        ~meta_guard() { if (ctx) { ggml_free(ctx); } }
    } guard { meta };

    const int64_t split_key = gguf_find_key(source.get(), "split.count");
    if (split_key >= 0
            && (gguf_get_kv_type(source.get(), split_key) != GGUF_TYPE_UINT16
                || gguf_get_val_u16(source.get(), split_key) > 1)) {
        set_error(
                "this model is stored as several GGUF files and a model export writes one; "
                "the trainable bundle carries the tensors this run changed");
        return false;
    }

    gguf_context_ptr out(gguf_init_empty());
    if (!out) {
        set_error("failed to allocate the model export GGUF context");
        return false;
    }
    gguf_set_kv(out.get(), source.get());
    // gguf_set_kv copies the key and not the alignment it implies, so a file
    // that sets its own would be written at one alignment and read at another.
    if (gguf_get_alignment(out.get()) != gguf_get_alignment(source.get())) {
        set_error(
                "this model declares a GGUF alignment this build does not reproduce; "
                "the trainable bundle carries the tensors this run changed");
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(source.get());
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(source.get(), i);
        ggml_tensor * live = model_tensor(state, name);
        if (!live) {
            set_error(
                    std::string("the loaded model carries no tensor named '") + name
                    + "', which the model file declares; a model export writes the file's "
                      "own tensor list");
            return false;
        }
        const ggml_tensor * declared = ggml_get_tensor(meta, name);
        if (!declared || live->type != declared->type
                || !ggml_are_same_shape(live, declared)) {
            set_error(
                    std::string("tensor '") + name
                    + "' is not the shape or dtype the model file declares for it");
            return false;
        }
        // Weights on a device are copied out one tensor at a time.
        gguf_add_tensor(out.get(), live);
    }

    if (!gguf_write_to_file(out.get(), path, false)) {
        set_error("failed to write the model export: " + std::string(path));
        return false;
    }
    return true;
}

} // namespace retro

extern "C" int retro_trainer_save_trainable(
        retro_trainer * trainer,
        const char * trainable_path) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(trainable_path)) {
            retro::set_error("trainable_path is required");
            return -1;
        }
        return retro::save_trainable_bundle(*state, trainable_path) ? 0 : -1;
    });
}

extern "C" int retro_trainer_load_trainable(
        retro_trainer * trainer,
        const char * trainable_path) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(trainable_path)) {
            retro::set_error("trainable_path is required");
            return -1;
        }
        return retro::load_trainable_bundle(*state, trainable_path) ? 0 : -1;
    });
}

extern "C" int retro_trainer_save_model(
        retro_trainer * trainer,
        const char * model_path) {
    return retro::boundary([&]() -> int {
        retro::trainer_state * state = retro::checked(trainer);
        if (!state) {
            return -1;
        }
        if (retro::is_blank(model_path)) {
            retro::set_error("model_path is required");
            return -1;
        }
        return retro::save_model_gguf(*state, model_path) ? 0 : -1;
    });
}
