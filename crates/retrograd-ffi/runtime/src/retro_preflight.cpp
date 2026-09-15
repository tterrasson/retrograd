#include "retro_runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

namespace retro {

namespace {

// One deduplicated op signature with an occurrence count and an example
// tensor, so a 36-layer model reports one line per op/shape, not 36.
struct issue_group {
    size_t count = 0;
    std::string example;
};

using issue_map = std::map<std::string, issue_group>;

struct preflight_data {
    issue_map missing_grad;
    std::map<std::string, issue_map> device_forward;
    std::map<std::string, issue_map> device_backward;
    // retro delta: per-device quantized types that caused an op to fall back
    // because the device has no in-place decoder. This exposes the cause and
    // suggested remedy directly in the report.
    std::map<std::string, std::set<enum ggml_type>> device_undecodable;
};

// The tensor an op reads through a per-type decoder, or null for ops that have
// none. These are the ops driven by GGML_RETRO_DEQUANT_TYPES.
const ggml_tensor * decoded_src(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_OUT_PROD:              return node->src[0];
        case GGML_OP_FUSED_SPARSE_CE:       return node->src[1];
        case GGML_OP_FUSED_SPARSE_CE_BACK:  return node->src[2];
        default:                            return nullptr;
    }
}

std::string tensor_brief(const ggml_tensor * tensor) {
    std::ostringstream out;
    out << ggml_type_name(tensor->type) << "[";
    const int n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < n_dims; ++i) {
        out << (i > 0 ? "," : "") << tensor->ne[i];
    }
    out << "]";
    return out.str();
}

// Op identity for deduplication: op name plus destination and source
// types/shapes. The tensor name is kept separately as an example.
std::string node_signature(const ggml_tensor * node) {
    std::ostringstream out;
    out << ggml_op_desc(node) << ": dst=" << tensor_brief(node);
    for (int i = 0; i < GGML_MAX_SRC && node->src[i]; ++i) {
        out << " src" << i << "=" << tensor_brief(node->src[i]);
    }
    return out.str();
}

void record_issue(issue_map & issues, const ggml_tensor * node) {
    issue_group & group = issues[node_signature(node)];
    if (group.count == 0) {
        group.example = node->name;
    }
    ++group.count;
}

void record_undecodable(
        preflight_data & data, ggml_backend_dev_t dev, const ggml_tensor * node) {
    const ggml_tensor * src = decoded_src(node);
    // Only when the type is genuinely absent from GGML_RETRO_DEQUANT_TYPES. A listed
    // type that still falls back fell back for another reason -- a shape, a stride, a
    // missing device feature, a memory limit -- and blaming quantization for it would
    // send the reader off to requantize a model that is already fine.
    if (src && src->type != GGML_TYPE_F32 && !is_retro_dequant_type(src->type)) {
        data.device_undecodable[ggml_backend_dev_name(dev)].insert(src->type);
    }
}

void preflight_callback(
        int32_t check, ggml_backend_dev_t dev, const ggml_tensor * node, void * userdata) {
    preflight_data * data = static_cast<preflight_data *>(userdata);
    switch (check) {
        case LLAMA_OPT_PREFLIGHT_MISSING_GRAD:
            record_issue(data->missing_grad, node);
            break;
        case LLAMA_OPT_PREFLIGHT_DEVICE_FORWARD:
            record_issue(data->device_forward[ggml_backend_dev_name(dev)], node);
            record_undecodable(*data, dev, node);
            break;
        case LLAMA_OPT_PREFLIGHT_DEVICE_BACKWARD:
            record_issue(data->device_backward[ggml_backend_dev_name(dev)], node);
            record_undecodable(*data, dev, node);
            break;
        default:
            break;
    }
}

// The devices the preflight checked: every registered CPU/GPU device, in
// registry order. Mirrors the device filter inside llama_opt_preflight.
std::vector<std::string> preflight_device_names() {
    std::vector<std::string> names;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU ||
                type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            const char * name = ggml_backend_dev_name(dev);
            names.push_back(name ? name : "unknown");
        }
    }
    return names;
}

// Requantization targets to suggest: a short preference order, filtered through
// GGML_RETRO_DEQUANT_TYPES and through the types that just failed. Both filters
// matter -- a hardcoded list drifts from the table, and suggesting a type that is
// itself in the undecodable set reads as nonsense to whoever has to act on it.
std::string suggested_quant_types(const std::set<enum ggml_type> & undecodable) {
    static const enum ggml_type preferred[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    };
    std::string list;
    for (const enum ggml_type type : preferred) {
        if (!is_retro_dequant_type(type) || undecodable.count(type)) {
            continue;
        }
        if (!list.empty()) {
            list += ", ";
        }
        list += ggml_type_name(type);
    }
    return list.empty() ? "a type listed in GGML_RETRO_DEQUANT_TYPES" : list;
}

void append_issues(std::ostringstream & out, const issue_map & issues, const char * indent) {
    for (const auto & item : issues) {
        out << indent << "- " << item.first
            << " (x" << item.second.count
            << ", e.g. " << item.second.example << ")\n";
    }
}

std::string format_preflight_report(
        const trainer_state & state,
        const preflight_data & data,
        int32_t n_missing) {
    std::ostringstream out;
    out << "training preflight\n";
    out << "  architecture: " << state.model->arch_name() << "\n";
    out << "  lora_tensor_pairs: " << (state.adapter ? state.adapter->ab_map.size() : 0) << "\n";
    if (!state.target_patterns.empty()) {
        out << "  lora_target_patterns: [" << join_patterns(state.target_patterns) << "]\n";
    }
    out << "  missing_gradient_rules: " << n_missing << "\n";
    if (!data.missing_grad.empty()) {
        append_issues(out, data.missing_grad, "    ");
    }

    out << "  devices:\n";
    for (const std::string & device : preflight_device_names()) {
        const auto forward = data.device_forward.find(device);
        const auto backward = data.device_backward.find(device);
        const bool forward_ok = forward == data.device_forward.end() || forward->second.empty();
        const bool backward_ok = backward == data.device_backward.end() || backward->second.empty();
        out << "    " << device << ": ";
        if (n_missing != 0) {
            out << (forward_ok ? "forward ready, backward blocked by missing gradient rules"
                               : "incomplete (backward blocked by missing gradient rules)") << "\n";
        } else if (forward_ok && backward_ok) {
            out << "training graph ready\n";
        } else {
            out << "incomplete (the scheduler falls back to another backend for these ops)\n";
        }
        if (!forward_ok) {
            out << "      forward:\n";
            append_issues(out, forward->second, "        ");
        }
        if (n_missing == 0 && !backward_ok) {
            out << "      backward:\n";
            append_issues(out, backward->second, "        ");
        }
        // retro delta: identify quantization fallbacks and suggest a supported
        // replacement type; the generic op signature is not enough to act on.
        const auto undecodable = data.device_undecodable.find(device);
        if (!(forward_ok && backward_ok) && undecodable != data.device_undecodable.end()) {
            out << "      undecodable_types:";
            for (const enum ggml_type type : undecodable->second) {
                out << " " << ggml_type_name(type);
            }
            out << "\n        this device has no in-place decoder for them; requantize the "
                   "model to " << suggested_quant_types(undecodable->second)
                << ", or add the type to GGML_RETRO_DEQUANT_TYPES\n";
        }
    }

    // Count nodes the active training device hands back to the CPU. Zero means
    // the graph remains device-resident; non-zero values imply scheduler splits.
    if (state.gpu_active && n_missing == 0) {
        out << "  active_device_fallback_nodes: " << state.preflight_device_fallbacks << "\n";
    }

    if (n_missing != 0) {
        out << "  status: not trainable; the listed ops have no gradient rule in "
               "ggml_compute_backward\n";
    } else {
        out << "  status: backward graph builds; unsupported device ops fall back at dispatch\n";
    }
    return out.str();
}

// Nodes the active GPU declined in forward or backward. Other registered devices
// are irrelevant because this report describes the device used by this trainer.
size_t count_device_fallbacks(
        const preflight_data & data, const std::string & device, std::ostringstream & detail) {
    size_t total = 0;
    for (const auto * issues : { &data.device_forward, &data.device_backward }) {
        const auto found = issues->find(device);
        if (found == issues->end()) {
            continue;
        }
        for (const auto & item : found->second) {
            total += item.second.count;
            detail << "\n  - " << item.first << " (x" << item.second.count << ")";
        }
    }
    return total;
}

} // namespace

bool require_gpu_resident(const trainer_state & state) {
    if (state.train_config.require_gpu_resident) {
        return true;
    }
    const char * override_value = std::getenv("RETRO_REQUIRE_GPU_RESIDENT");
    return override_value != nullptr && override_value[0] != '\0'
            && std::strcmp(override_value, "0") != 0;
}

bool ensure_train_preflight(trainer_state & state) {
    if (state.preflight_missing >= 0) {
        return true;
    }

    preflight_data data;
    const int32_t n_missing = llama_opt_preflight(state.ctx.get(), preflight_callback, &data);
    if (n_missing < 0) {
        set_error("training preflight failed to build the forward graph");
        return false;
    }

    // Count the active GPU's fallbacks before the report is formatted, so both
    // the report and require_gpu_resident read the same number.
    state.preflight_device_fallbacks = 0;
    if (state.gpu_active && n_missing == 0) {
        std::ostringstream detail;
        const size_t fallbacks = count_device_fallbacks(data, state.backend_name, detail);
        state.preflight_device_fallbacks = (int32_t) fallbacks;
        if (fallbacks > 0 && require_gpu_resident(state)) {
            set_error("training.require_gpu_resident is set but " + std::to_string(fallbacks)
                    + " training-graph node(s) fall back to the CPU on " + state.backend_name
                    + ":" + detail.str());
            return false;
        }
    }

    state.preflight_missing = n_missing;
    state.preflight_report = format_preflight_report(state, data, n_missing);
    state.invalidate_report_caches();
    return true;
}

int train_preflight_impl(
        retro_trainer * trainer,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora) {
            set_error("create or load a LoRA adapter before running the training preflight");
            return -1;
        }
        if (!state->adapter) {
            set_error("LoRA adapter is not initialized");
            return -1;
        }
        if (state->loaded_lora && !promote_loaded_lora_to_trainable(*state)) {
            return -1;
        }
        if (!ensure_opt_context(*state)) {
            return -1;
        }
        if (!ensure_train_preflight(*state)) {
            return -1;
        }
        return copy_string_out(state->preflight_report, buffer, n_buffer, out_n_bytes);
    });
}

int preflight_summary_impl(retro_trainer * trainer, retro_preflight_summary * out_summary) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!out_summary) {
            set_error("out_summary is required");
            return -1;
        }
        if (out_summary->struct_size != sizeof(retro_preflight_summary)) {
            set_error("retro_preflight_summary.struct_size does not match this build");
            return -1;
        }
        if (state->preflight_missing < 0) {
            set_error("run the training preflight before requesting its summary");
            return -1;
        }

        // FNV-1a is sufficient here: this is an invalidation identity, not a
        // security boundary. Hash the deterministic rendering produced from the
        // structured traversal; no caller parses that rendering.
        uint64_t fingerprint = UINT64_C(14695981039346656037);
        for (unsigned char byte : state->preflight_report) {
            fingerprint ^= byte;
            fingerprint *= UINT64_C(1099511628211);
        }
        std::ostringstream rendered;
        rendered << "fnv1a64:" << std::hex << fingerprint;
        const std::string text = rendered.str();

        out_summary->missing_gradient_rules = state->preflight_missing;
        out_summary->active_device_fallback_nodes =
                static_cast<uint64_t>(state->preflight_device_fallbacks);
        std::memset(out_summary->graph_fingerprint, 0, RETRO_PREFLIGHT_FINGERPRINT_MAX);
        std::memcpy(out_summary->graph_fingerprint, text.data(),
                std::min(text.size(), static_cast<size_t>(RETRO_PREFLIGHT_FINGERPRINT_MAX - 1)));
        return 0;
    });
}

} // namespace retro
