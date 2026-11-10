// Expert banks and expert placement: which tensors of a MoE model are routed
// expert banks, whether its architecture is one the runtime can place
// explicitly, and the validation of a placement request before the real model
// load (a refused request must not create a backend context).
//
// A bank is a layer tensor whose name carries an `_exps` stem. The runtime
// places explicitly only the architectures whose every bank sits under the
// `ffn_{gate_up,gate,up,down}_exps` motif and is routed through the shared MoE
// FFN path with no id remapping: a bank outside that motif (a second bank set,
// a per-expert norm, an attention-side bank) means the placement contract does
// not describe this model, and the architecture is reported as not validated.
// The validated list below is that contract; adding an architecture is adding
// a line here plus a fixture whose inventory a test checks.

#include "retro_runtime.hpp"

#include <charconv>
#include <map>
#include <regex>
#include <set>

namespace retro {

namespace {

// Architectures validated for explicit expert placement. Each entry is covered
// by a test that loads its fixture and checks the bank inventory.
const char * const validated_expert_archs[] = {
    "olmoe",
};

// `blk.<layer>.<stem>` or `blk.<layer>.<stem>.<suffix>`.
const std::regex & block_tensor_pattern() {
    static const std::regex pattern(R"(^blk\.(\d+)\.([A-Za-z0-9_]+)(\..+)?$)");
    return pattern;
}

// The four bank stems the placement contract covers.
const std::regex & bank_stem_pattern() {
    static const std::regex pattern(R"(^(ffn_(gate_up|gate|up|down)_exps)$)");
    return pattern;
}

// Any layer tensor with an `_exps` stem, bank or not.
const std::regex & any_exps_pattern() {
    static const std::regex pattern(R"(^blk\.\d+\..*_exps)");
    return pattern;
}

bool parse_block_tensor(const std::string & name, uint32_t & layer, std::string & stem) {
    std::smatch match;
    if (!std::regex_match(name, match, block_tensor_pattern())) {
        return false;
    }
    const auto index = match[1].str();
    if (std::from_chars(index.data(), index.data() + index.size(), layer).ec != std::errc{}) {
        return false;
    }
    stem = match[2].str();
    return true;
}

std::string regex_escape(const std::string & literal) {
    std::string out;
    out.reserve(literal.size() * 2);
    for (char c : literal) {
        if (std::string(".^$|()[]{}*+?\\").find(c) != std::string::npos) {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

} // namespace

bool arch_validated_for_expert_placement(const std::string & arch) {
    for (const char * validated : validated_expert_archs) {
        if (arch == validated) {
            return true;
        }
    }
    return false;
}

bool collect_expert_banks(const llama_model & model, expert_placement_probe & out) {
    out = expert_placement_probe {};
    out.arch = model.arch_name();
    out.n_expert_used = model.hparams.n_expert_used_max();

    bool saw_exps_outside_motif = false;
    for (const auto & item : model.tensors_by_name) {
        if (!item.second) {
            continue;
        }
        const std::string name(item.second->name);
        if (!std::regex_search(name, any_exps_pattern())) {
            continue;
        }
        uint32_t layer = 0;
        std::string stem;
        if (!parse_block_tensor(name, layer, stem) || !std::regex_match(stem, bank_stem_pattern())) {
            // A bank outside the placement motif (a second bank set, a
            // per-expert norm, an attention-side bank) invalidates the
            // architecture rather than being listed.
            saw_exps_outside_motif = true;
            continue;
        }
        retro_expert_bank_desc & bank = out.banks.emplace_back();
        bank = retro_expert_bank_desc {};
        bank.layer = layer;
        // The role is the stem without the `ffn_` prefix and the `_exps`
        // suffix: "gate_up" | "gate" | "up" | "down".
        const std::string role = stem.substr(4, stem.size() - 4 - 5);
        std::strncpy(bank.role, role.c_str(), sizeof(bank.role) - 1);
        if (!describe_tensor(item.second, bank.tensor)) {
            return false;
        }
    }

    std::sort(out.banks.begin(), out.banks.end(), [](const auto & a, const auto & b) {
        return std::strcmp(a.tensor.name, b.tensor.name) < 0;
    });
    if (out.n_expert_used > 0) {
        out.flags |= RETRO_EXPERT_BANKS_MOE;
    }
    if (!out.banks.empty() && arch_validated_for_expert_placement(out.arch) && !saw_exps_outside_motif) {
        out.flags |= RETRO_EXPERT_BANKS_VALIDATED;
    }
    return true;
}

int copy_expert_banks(
        const expert_placement_probe & probe,
        retro_expert_bank_desc * out_banks,
        size_t n_max,
        size_t * out_count) {
    *out_count = probe.banks.size();
    if (!out_banks || probe.banks.empty()) {
        return 0;
    }
    if (n_max < probe.banks.size()) {
        set_error("expert bank buffer is too small");
        return -2;
    }
    std::copy(probe.banks.begin(), probe.banks.end(), out_banks);
    return 0;
}

static int expert_banks_impl(
        const char * model_path,
        uint32_t * out_flags,
        retro_expert_bank_desc * out_banks,
        size_t n_max,
        size_t * out_count) {
    return boundary([&]() -> int {
        if (!out_flags || !out_count) {
            set_error("out_flags and out_count are required");
            return -1;
        }
        *out_flags = 0;
        *out_count = 0;
        expert_placement_probe probe;
        if (!probe_expert_banks(model_path, probe)) {
            return -1;
        }
        *out_flags = probe.flags;
        return copy_expert_banks(probe, out_banks, n_max, out_count);
    });
}

bool probe_expert_banks(const char * model_path, expert_placement_probe & out) {
    if (is_blank(model_path)) {
        set_error("model_path is required");
        return false;
    }
    ensure_backend_initialized();
    model_ptr model = load_model_metadata_only(model_path);
    if (!model) {
        set_error("failed to load model GGUF: " + std::string(model_path));
        return false;
    }
    return collect_expert_banks(*model, out);
}

// Lower bound for one group: each bank needs all routed slots plus the
// padding slot used by the cache allocator.
static uint64_t expert_cache_floor_bytes(const expert_placement_probe & probe) {
    std::map<uint32_t, uint64_t> layer_bytes;
    uint64_t floor = 0;
    for (const retro_expert_bank_desc & bank : probe.banks) {
        const int64_t n_expert = bank.tensor.ne[2];
        if (n_expert > 0) {
            const uint64_t stride = bank.tensor.n_bytes / static_cast<uint64_t>(n_expert);
            const uint64_t bytes = stride * (uint64_t{probe.n_expert_used} + 1);
            floor = std::max(floor, layer_bytes[bank.layer] += bytes);
        }
    }
    return floor;
}

static size_t expert_moe_layer_count(const expert_placement_probe & probe) {
    std::set<uint32_t> layers;
    for (const retro_expert_bank_desc & bank : probe.banks) {
        layers.insert(bank.layer);
    }
    return layers.size();
}

bool validate_expert_placement(
        const retro_engine_config & config,
        const expert_placement_probe & probe,
        bool gpu_available) {
    const bool cache_requested = config.expert_device_cache_bytes > 0;
    if (config.expert_placement == RETRO_EXPERT_PLACEMENT_DEVICE) {
        if (cache_requested) {
            set_error("an expert device cache requires host placement: with device placement every expert bank already sits on the device");
            return false;
        }
        if (config.expert_host_layers != 0) {
            set_error("expert_host_layers only applies to host placement");
            return false;
        }
        return true;
    }
    // HOST placement.
    if (!(probe.flags & RETRO_EXPERT_BANKS_MOE)) {
        set_error("host expert placement requested for a model with no expert banks (not a MoE model)");
        return false;
    }
    if (!(probe.flags & RETRO_EXPERT_BANKS_VALIDATED)) {
        set_error("architecture '" + probe.arch + "' is not validated for expert placement: its expert banks are not all single `ffn_*_exps` banks per layer");
        return false;
    }
    const size_t n_moe_layers = expert_moe_layer_count(probe);
    if (config.expert_host_layers > n_moe_layers) {
        set_error("expert_host_layers = " + std::to_string(config.expert_host_layers)
                + " exceeds the " + std::to_string(n_moe_layers) + " MoE layers of this model");
        return false;
    }
    if (cache_requested) {
        if (!gpu_available) {
            set_error("an expert device cache requires a GPU backend; rebuild with Cargo features metal (Apple hardware), vulkan (Vulkan SDK), or cuda (NVIDIA CUDA toolkit)");
            return false;
        }
        const uint64_t floor_bytes = expert_cache_floor_bytes(probe);
        if (config.expert_device_cache_bytes < floor_bytes) {
            set_error("expert device cache budget of " + std::to_string(config.expert_device_cache_bytes)
                    + " bytes does not cover " + std::to_string(probe.n_expert_used)
                    + " experts per group (needs at least " + std::to_string(floor_bytes) + " bytes)");
            return false;
        }
    }
    return true;
}

bool build_host_buft_overrides(
        const expert_placement_probe & probe,
        uint32_t host_layers,
        std::vector<std::string> & patterns,
        std::vector<llama_model_tensor_buft_override> & overrides) {
    std::set<uint32_t> layers;
    for (const retro_expert_bank_desc & bank : probe.banks) {
        layers.insert(bank.layer);
    }
    // The last `host_layers` MoE layers (0 = all of them) keep their banks on
    // the host; the overrides pin exactly their bank tensors to the CPU buffer
    // type, the rest follows the device offload.
    std::vector<uint32_t> ordered(layers.begin(), layers.end());
    const size_t keep = host_layers == 0 ? ordered.size() : std::min<size_t>(host_layers, ordered.size());
    std::set<uint32_t> host_set(
            ordered.end() - static_cast<std::ptrdiff_t>(keep), ordered.end());

    patterns.clear();
    overrides.clear();
    patterns.reserve(probe.banks.size());
    for (const retro_expert_bank_desc & bank : probe.banks) {
        if (!host_set.count(bank.layer)) {
            continue;
        }
        patterns.push_back("^" + regex_escape(bank.tensor.name) + "$");
        llama_model_tensor_buft_override entry {};
        entry.pattern = patterns.back().c_str();
        entry.buft = ggml_backend_cpu_buffer_type();
        overrides.push_back(entry);
    }
    if (overrides.size() >= llama_max_tensor_buft_overrides()) {
        set_error("too many expert bank tensors for the tensor_buft_overrides list");
        return false;
    }
    llama_model_tensor_buft_override terminator {};
    terminator.pattern = nullptr;
    overrides.push_back(terminator);
    return true;
}

} // namespace retro

extern "C" int retro_read_expert_banks(
        const char * model_path,
        uint32_t * out_flags,
        retro_expert_bank_desc * out_banks,
        size_t n_max,
        size_t * out_count) {
    return retro::expert_banks_impl(model_path, out_flags, out_banks, n_max, out_count);
}
