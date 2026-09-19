// The per-tensor inventory: what a GGUF declares, as the loader sees it.
//
// retro_model_info answers "how big is this model"; this answers "which tensors
// is it made of, under which names, shapes and dtypes". Selection, optimizer
// allocation, checkpoint manifests and the planner all resolve against the
// second question, and none of them can be answered from an aggregate.
//
// The model is opened the same way retro_read_model_info opens it - mapped,
// host-only, no context - so enumerating a 70B model costs a page table and not
// a memory budget.

#include "retro_runtime.hpp"

#include <algorithm>
#include <vector>

namespace retro {

namespace {

model_ptr load_model_metadata_only(const char * model_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.use_extra_bufts = false;
    model_params.n_gpu_layers = 0;
    // An empty device list keeps the load host-only rather than leaving a GPU
    // in the model's device list, exactly as read_model_info_impl does.
    ggml_backend_dev_t no_devices[] = { nullptr };
    model_params.devices = no_devices;
    return model_ptr(llama_model_load_from_file(model_path, model_params));
}

// The storage identity two aliases agree on. A mapped model's tensors point
// into the mapping, so the data address *is* the allocation identity; a tensor
// with no data yet reports 0, which the resolver reads as "no identity
// recorded" rather than as "shares storage with every other such tensor".
uint64_t storage_id_of(const ggml_tensor * tensor) {
    return tensor && tensor->data ? reinterpret_cast<uint64_t>(tensor->data) : 0;
}

} // namespace

int read_tensor_inventory_impl(
        const char * model_path,
        uint32_t * out_version,
        retro_tensor_desc * out_tensors,
        size_t n_max,
        size_t * out_count) {
    return boundary([&]() -> int {
        if (is_blank(model_path)) {
            set_error("model_path is required");
            return -1;
        }
        if (!out_version || !out_count) {
            set_error("out_version and out_count are required");
            return -1;
        }
        *out_version = RETRO_TENSOR_INVENTORY_VERSION;
        *out_count = 0;

        ensure_backend_initialized();
        model_ptr model = load_model_metadata_only(model_path);
        if (!model) {
            set_error("failed to load model GGUF: " + std::string(model_path));
            return -1;
        }

        // Sorted by name so two reads of one file produce the same order. The
        // loader's map order is already deterministic, but the inventory is a
        // published contract and must not inherit a container's guarantees.
        std::vector<const ggml_tensor *> tensors;
        tensors.reserve(model->tensors_by_name.size());
        for (const auto & item : model->tensors_by_name) {
            if (item.second) {
                tensors.push_back(item.second);
            }
        }
        std::sort(
                tensors.begin(),
                tensors.end(),
                [](const ggml_tensor * left, const ggml_tensor * right) {
                    return std::strcmp(left->name, right->name) < 0;
                });

        *out_count = tensors.size();
        if (!out_tensors || n_max == 0) {
            return 0;
        }
        if (n_max < tensors.size()) {
            set_error("tensor inventory buffer is too small");
            return -2;
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            const ggml_tensor * tensor = tensors[i];
            retro_tensor_desc & desc = out_tensors[i];
            desc = retro_tensor_desc {};

            // A truncated name is not a shorter name, it is a name that selects
            // the wrong tensor. Refuse instead.
            const size_t name_len = std::strlen(tensor->name);
            if (name_len + 1 > RETRO_TENSOR_NAME_MAX) {
                set_error(
                        std::string("tensor name does not fit the inventory contract: ")
                        + tensor->name);
                return -1;
            }
            std::memcpy(desc.name, tensor->name, name_len + 1);

            const char * type_name = ggml_type_name(tensor->type);
            const size_t type_len = type_name ? std::strlen(type_name) : 0;
            if (type_len + 1 > RETRO_MODEL_INFO_NAME_MAX) {
                set_error(std::string("ggml type name does not fit: ") + type_name);
                return -1;
            }
            std::memcpy(desc.type_name, type_name, type_len + 1);

            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                desc.ne[d] = tensor->ne[d];
            }
            desc.n_elements = static_cast<uint64_t>(ggml_nelements(tensor));
            desc.n_bytes = static_cast<uint64_t>(ggml_nbytes(tensor));
            desc.storage_id = storage_id_of(tensor);
        }
        return 0;
    });
}

} // namespace retro

extern "C" int retro_read_tensor_inventory(
        const char * model_path,
        uint32_t * out_version,
        retro_tensor_desc * out_tensors,
        size_t n_max,
        size_t * out_count) {
    return retro::read_tensor_inventory_impl(
            model_path, out_version, out_tensors, n_max, out_count);
}
