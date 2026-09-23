#include "retro_runtime.hpp"

#include <cfloat>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>

namespace retro {

bool wildcard_match(const char * pattern, const char * value) {
    if (*pattern == '\0') {
        return *value == '\0';
    }
    if (*pattern == '*') {
        while (*(pattern + 1) == '*') {
            ++pattern;
        }
        for (const char * it = value; ; ++it) {
            if (wildcard_match(pattern + 1, it)) {
                return true;
            }
            if (*it == '\0') {
                return false;
            }
        }
    }
    if (*value == '\0' || *pattern != *value) {
        return false;
    }
    return wildcard_match(pattern + 1, value + 1);
}

bool matches_any_pattern(const std::string & name, const std::vector<std::string> & patterns) {
    for (const std::string & pattern : patterns) {
        if (wildcard_match(pattern.c_str(), name.c_str())) {
            return true;
        }
    }
    return false;
}

int64_t tensor_param_count(const ggml_tensor * tensor) {
    return tensor ? ggml_nelements(tensor) : 0;
}

std::string tensor_shape_2d(const ggml_tensor * tensor) {
    if (!tensor) {
        return "null";
    }
    std::ostringstream out;
    out << tensor->ne[0] << "x" << tensor->ne[1];
    return out.str();
}

uint32_t inferred_lora_rank(const trainer_state & state) {
    if (state.lora_rank != 0) {
        return state.lora_rank;
    }
    if (!state.adapter || state.adapter->ab_map.empty()) {
        return 0;
    }
    const ggml_tensor * tensor_a = state.adapter->ab_map.begin()->second.a;
    return tensor_a && tensor_a->ne[1] > 0 ? static_cast<uint32_t>(tensor_a->ne[1]) : 0;
}

size_t count_lora_param_tensors(const trainer_state & state) {
    if (!state.adapter) {
        return 0;
    }

    size_t count = 0;
    for (const auto & item : state.adapter->ab_map) {
        if (is_param_tensor(item.second.a)) {
            ++count;
        }
        if (is_param_tensor(item.second.b)) {
            ++count;
        }
    }
    return count;
}

size_t count_base_param_tensors(const trainer_state & state) {
    size_t count = 0;
    for (const auto & item : state.model->tensors_by_name) {
        if (is_param_tensor(item.second)) {
            ++count;
        }
    }
    return count;
}

int64_t count_lora_parameters(const trainer_state & state) {
    if (!state.adapter) {
        return 0;
    }

    int64_t count = 0;
    for (const auto & item : state.adapter->ab_map) {
        count += tensor_param_count(item.second.a);
        count += tensor_param_count(item.second.b);
    }
    return count;
}

size_t count_lora_parameter_bytes(const trainer_state & state) {
    if (!state.adapter) {
        return 0;
    }
    size_t bytes = 0;
    for (const auto & item : state.adapter->ab_map) {
        bytes += item.second.a ? ggml_nbytes(item.second.a) : 0;
        bytes += item.second.b ? ggml_nbytes(item.second.b) : 0;
    }
    return bytes;
}

const char * lora_dtype_name(int32_t dtype) {
    return dtype == RETRO_LORA_DTYPE_F16 ? "F16" : "F32";
}

std::string tensor_pair_error(const std::string & name) {
    return "LoRA tensor pair for '" + name + "' is incomplete or invalid";
}

bool validate_lora_tensor_pairs(const llama_adapter_lora & adapter) {
    if (adapter.ab_map.empty()) {
        set_error("LoRA adapter has no trainable tensor pairs");
        return false;
    }

    ggml_type adapter_type = GGML_TYPE_COUNT;
    for (const auto & item : adapter.ab_map) {
        const ggml_tensor * tensor_a = item.second.a;
        const ggml_tensor * tensor_b = item.second.b;
        if (!tensor_a || !tensor_b) {
            set_error(tensor_pair_error(item.first));
            return false;
        }
        const bool supported = tensor_a->type == GGML_TYPE_F32 || tensor_a->type == GGML_TYPE_F16;
        if (!supported || tensor_b->type != tensor_a->type) {
            set_error("LoRA adapter requires matching F32 or F16 tensor pairs");
            return false;
        }
        if (adapter_type == GGML_TYPE_COUNT) {
            adapter_type = tensor_a->type;
        } else if (tensor_a->type != adapter_type) {
            set_error("LoRA adapter must use one consistent dtype for every tensor pair");
            return false;
        }
        if (!ggml_is_contiguous(tensor_a) || !ggml_is_contiguous(tensor_b)) {
            set_error("LoRA adapter requires contiguous tensors");
            return false;
        }
    }

    return true;
}

namespace {

// LoRA adapters wrap a 2D weight in an A/B matrix pair, so only true matrices
// are valid targets (a 3D MoE expert stack needs per-expert handling).
bool is_lora_target_matrix(const ggml_tensor * tensor) {
    return tensor && ggml_n_dims(tensor) == 2 && tensor->ne[0] > 1 && tensor->ne[1] > 1;
}

bool model_has_matrix_matching(const llama_model & model, const char * pattern) {
    for (const auto & item : model.tensors_by_name) {
        if (is_lora_target_matrix(item.second) && wildcard_match(pattern, item.first.c_str())) {
            return true;
        }
    }
    return false;
}

// Collapses a per-layer tensor name into its wildcard family, e.g.
// "blk.17.ssm_in.weight" -> "blk.*.ssm_in.weight".
std::string tensor_name_family(const std::string & name) {
    if (name.compare(0, 4, "blk.") != 0) {
        return name;
    }
    size_t digits = name.find_first_not_of("0123456789", 4);
    if (digits == 4 || digits == std::string::npos || name[digits] != '.') {
        return name;
    }
    return "blk.*" + name.substr(digits);
}

} // namespace

bool detect_lora_profile(const llama_model & model, lora_profile & out) {
    // Attention families cover every transformer variant llama.cpp loads;
    // detection keys on the tensors actually present, not the arch name, so a
    // new architecture with standard GGUF tensor names works out of the box.
    if (model_has_matrix_matching(model, "blk.*.attn_q.weight") &&
            model_has_matrix_matching(model, "blk.*.attn_v.weight")) {
        out.name = "qv-attention";
        out.patterns = { "blk.*.attn_q.weight", "blk.*.attn_v.weight" };
        return true;
    }
    if (model_has_matrix_matching(model, "blk.*.attn_qkv.weight")) {
        out.name = "fused-qkv";
        out.patterns = { "blk.*.attn_qkv.weight" };
        return true;
    }
    return false;
}

std::vector<std::string> lora_candidate_patterns(const llama_model & model) {
    std::set<std::string> families;
    for (const auto & item : model.tensors_by_name) {
        if (is_lora_target_matrix(item.second)
                && item.first.size() >= 7
                && item.first.compare(item.first.size() - 7, 7, ".weight") == 0) {
            families.insert(tensor_name_family(item.first));
        }
    }
    return std::vector<std::string>(families.begin(), families.end());
}

std::vector<std::string> lora_candidate_targets(const llama_model & model) {
    std::set<std::string> names;
    for (const auto & item : model.tensors_by_name) {
        if (is_lora_target_matrix(item.second)) {
            names.insert(item.first);
        }
    }
    return std::vector<std::string>(names.begin(), names.end());
}

bool resolve_lora_targets(trainer_state & state, std::vector<const ggml_tensor *> & targets) {
    if (state.target_patterns.empty()) {
        lora_profile profile;
        if (!detect_lora_profile(*state.model, profile)) {
            std::vector<std::string> candidates = lora_candidate_patterns(*state.model);
            std::ostringstream message;
            message << "no automatic LoRA target profile detected for architecture '"
                    << state.model->arch_name()
                    << "'; pass --targets explicitly";
            if (!candidates.empty()) {
                message << "; candidate patterns: [" << join_patterns(candidates) << "]";
            }
            set_error(message.str());
            return false;
        }
        state.target_patterns = profile.patterns;
    }

    targets.clear();
    size_t matched_non_matrix = 0;
    for (const auto & item : state.model->tensors_by_name) {
        if (!item.second || !matches_any_pattern(item.first, state.target_patterns)) {
            continue;
        }
        if (!is_lora_target_matrix(item.second)) {
            ++matched_non_matrix;
            continue;
        }
        targets.push_back(item.second);
    }
    if (targets.empty()) {
        std::ostringstream message;
        message << "no base model 2D weight matrices matched LoRA target patterns: ["
                << join_patterns(state.target_patterns) << "]";
        if (matched_non_matrix > 0) {
            message << " (" << matched_non_matrix
                    << " matching tensors were skipped because they are not 2D matrices)";
        }
        // A pattern that names a layer index is the common way to miss on a
        // hybrid architecture, where blk.N carries a different block family
        // than the caller assumed. Naming the families that do exist turns a
        // dead end into a one-line fix.
        std::vector<std::string> candidates = lora_candidate_patterns(*state.model);
        if (!candidates.empty()) {
            message << "; candidate patterns: [" << join_patterns(candidates) << "]";
        }
        set_error(message.str());
        return false;
    }
    return true;
}

bool trains_base_weights(const trainer_state & state) {
    return state.train_config.trainable != RETRO_TRAINABLE_LORA;
}

bool trains_loss_head(const trainer_state & state) {
    // The two names the fused loss unpacks from the logits node; a tied head
    // never reaches the resolved set, so the name match is enough.
    for (const std::string & name : state.trainable_base) {
        if (name == "output.weight" || name == "output.bias") {
            return true;
        }
    }
    return false;
}

bool fused_loss_enabled(const trainer_state & state) {
    // Not a preference: the fused backward asserts on a projection that needs
    // a gradient, so training the head must take the dense path, and the dense
    // backward already produces the head's gradient.
    return state.train_config.chunked_cross_entropy && !trains_loss_head(state);
}

bool opt_param_filter_trainable(const ggml_tensor * tensor, void * userdata) {
    const trainer_state * state = static_cast<const trainer_state *>(userdata);
    if (!tensor || !state || state->trainable_base.empty()) {
        return false;
    }
    // Linear over a set that is at most a few hundred names, walked once per
    // tensor at graph construction: a map would buy nothing measurable and
    // would need its own lifetime beside the vector the callback already owns.
    for (const std::string & name : state->trainable_base) {
        if (name == tensor->name) {
            return true;
        }
    }
    return false;
}

bool is_param_tensor(const ggml_tensor * tensor) {
    return tensor && ((tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0);
}

std::string describe_lora(const trainer_state & state) {
    std::ostringstream out;
    out << "LoRA adapter\n";
    if (!state.adapter) {
        out << "  status: not initialized\n";
        return out.str();
    }

    out << "  loaded_from_file: " << (state.loaded_lora ? "true" : "false") << "\n";
    out << "  rank: " << inferred_lora_rank(state) << "\n";
    out << "  alpha: " << state.adapter->alpha << "\n";
    out << "  lora_dtype: " << lora_dtype_name(state.lora_dtype) << "\n";
    if (!state.target_patterns.empty()) {
        out << "  target_patterns: [" << join_patterns(state.target_patterns) << "]\n";
    }
    out << "  tensor_pairs: " << state.adapter->ab_map.size() << "\n";
    out << "  lora_parameter_count: " << count_lora_parameters(state) << "\n";
    out << "  lora_parameter_bytes: " << count_lora_parameter_bytes(state) << "\n";
    out << "  lora_trainable_tensors: " << count_lora_param_tensors(state) << "\n";
    out << "  base_trainable_tensors: " << count_base_param_tensors(state) << "\n";
    out << "  tensors:\n";
    std::vector<std::string> tensor_names;
    tensor_names.reserve(state.adapter->ab_map.size());
    for (const auto & item : state.adapter->ab_map) {
        tensor_names.push_back(item.first);
    }
    std::sort(tensor_names.begin(), tensor_names.end());
    for (const std::string & tensor_name : tensor_names) {
        const auto item = state.adapter->ab_map.find(tensor_name);
        if (item == state.adapter->ab_map.end()) {
            continue;
        }
        const ggml_tensor * tensor_a = item->second.a;
        const ggml_tensor * tensor_b = item->second.b;
        out << "    " << item->first
            << ": A=" << tensor_shape_2d(tensor_a)
            << " B=" << tensor_shape_2d(tensor_b)
            << " dtype=" << (tensor_a ? ggml_type_name(tensor_a->type) : "unknown")
            << " trainable="
            << ((is_param_tensor(tensor_a) && is_param_tensor(tensor_b)) ? "true" : "false")
            << "\n";
    }
    return out.str();
}

// The name a refusal spells, from the wire value the config carries.
const char * optimizer_name(int32_t optimizer) {
    switch (optimizer) {
        case RETRO_OPTIMIZER_SGD:   return "sgd";
        case RETRO_OPTIMIZER_ADAMW: return "adamw";
        case RETRO_OPTIMIZER_MUON:  return "muon";
        case RETRO_OPTIMIZER_GEFEN: return "gefen";
        default:                    return "unknown";
    }
}

// retro_optimizer and ggml_opt_optimizer_type are one enumeration written
// twice, and these two functions are the only place that is relied upon: a
// value outside the table is the run's default rather than a cast.
ggml_opt_optimizer_type ggml_optimizer_of(int32_t optimizer) {
    switch (optimizer) {
        case RETRO_OPTIMIZER_SGD:   return GGML_OPT_OPTIMIZER_TYPE_SGD;
        case RETRO_OPTIMIZER_MUON:  return GGML_OPT_OPTIMIZER_TYPE_MUON;
        case RETRO_OPTIMIZER_GEFEN: return GGML_OPT_OPTIMIZER_TYPE_GEFEN;
        default:                    return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
}

int32_t retro_optimizer_of(ggml_opt_optimizer_type optimizer) {
    switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_SGD:   return RETRO_OPTIMIZER_SGD;
        case GGML_OPT_OPTIMIZER_TYPE_MUON:  return RETRO_OPTIMIZER_MUON;
        case GGML_OPT_OPTIMIZER_TYPE_GEFEN: return RETRO_OPTIMIZER_GEFEN;
        default:                            return RETRO_OPTIMIZER_ADAMW;
    }
}

int opt_step_dtype_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return RETRO_OPT_STEP_DTYPE_F16;
        case GGML_TYPE_BF16: return RETRO_OPT_STEP_DTYPE_BF16;
        default:             return -1;
    }
}

ggml_type opt_step_dtype_of(int index) {
    return index == RETRO_OPT_STEP_DTYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
}

bool optimizer_kernel_writes_dtype(int32_t optimizer, ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;  // the floor, under every optimizer
    }
    switch (optimizer) {
        // Both store their update with rounding, so both take the two
        // half-precision grids.
        case RETRO_OPTIMIZER_ADAMW:
        case RETRO_OPTIMIZER_SGD:
            return opt_step_dtype_index(type) >= 0;
        // F32 only: both steps already approximate (Muon's Newton-Schulz,
        // Gefen's first moment), and a rounded store would add a second.
        case RETRO_OPTIMIZER_MUON:
        case RETRO_OPTIMIZER_GEFEN:
            return false;
        default:
            return false;
    }
}

bool optimizer_supports_dtype(const trainer_state & state, int32_t optimizer, ggml_type type) {
    // The kernel table first, then the device probe: a pair the table carries
    // can still be refused by the backend. F32 never asks the device.
    if (!optimizer_kernel_writes_dtype(optimizer, type)) {
        return false;
    }
    if (type == GGML_TYPE_F32) {
        return true;
    }
    const int slot = opt_step_dtype_index(type);
    // The load-time probe recorded which half-precision parameters this
    // device's step takes.
    return slot >= 0 && optimizer >= 0 && optimizer < 4
            && state.cap_opt_step_dtypes[slot][optimizer];
}

ggml_opt_optimizer_type opt_param_optimizer(const ggml_tensor * tensor, void * userdata) {
    const trainer_state * state = static_cast<const trainer_state *>(userdata);
    int32_t optimizer = state ? state->train_config.optimizer : RETRO_OPTIMIZER_ADAMW;
    if (state && tensor) {
        // The table is at most one row per marked parameter and usually empty.
        for (const auto & row : state->optimizer_assignment) {
            if (row.first == tensor->name) {
                optimizer = row.second;
                break;
            }
        }
    }
    return ggml_optimizer_of(optimizer);
}

// Every marked parameter, adapter factors first and base tensors after, with
// the optimizer that owns each.
static void for_each_marked(
        const trainer_state & state,
        const std::function<void(const ggml_tensor *, int32_t)> & visit) {
    auto owner_of = [&](const ggml_tensor * tensor) {
        return retro_optimizer_of(
                opt_param_optimizer(tensor, const_cast<trainer_state *>(&state)));
    };
    auto offer = [&](const ggml_tensor * tensor) {
        if (is_param_tensor(tensor)) {
            visit(tensor, owner_of(tensor));
        }
    };
    if (state.adapter) {
        for (const auto & item : state.adapter->ab_map) {
            offer(item.second.a);
            offer(item.second.b);
        }
    }
    for (const auto & item : state.model->tensors_by_name) {
        offer(item.second);
    }
}

bool optimizer_supports_marked_dtypes(const trainer_state & state) {
    // Asked per parameter, against the table of the optimizer that owns it.
    // An F16 adapter is the normal case, since it is the default storage: an
    // SGD-owned F16 parameter would abort in the middle of the first step.
    std::vector<std::string> unsupported;
    std::vector<std::string> offenders;
    // Whether at least one refusal is the backend declining a dtype its
    // optimizer's kernel table does carry; that gets its own message.
    bool backend_refused = false;
    std::vector<std::string> refused_dtypes;
    for_each_marked(state, [&](const ggml_tensor * tensor, int32_t optimizer) {
        if (optimizer_supports_dtype(state, optimizer, tensor->type)) {
            return;
        }
        // The kernel carries the precision and the device turned it down:
        // a different refusal from an optimizer that writes F32 only.
        if (optimizer_kernel_writes_dtype(optimizer, tensor->type)) {
            backend_refused = true;
            if (std::find(refused_dtypes.begin(), refused_dtypes.end(),
                        ggml_type_name(tensor->type)) == refused_dtypes.end()) {
                refused_dtypes.emplace_back(ggml_type_name(tensor->type));
            }
        }
        unsupported.push_back(std::string(tensor->name) + " (" + ggml_type_name(tensor->type)
                + ", " + optimizer_name(optimizer) + ")");
        if (std::find(offenders.begin(), offenders.end(), optimizer_name(optimizer))
                == offenders.end()) {
            offenders.push_back(optimizer_name(optimizer));
        }
    });
    if (unsupported.empty()) {
        return true;
    }
    std::sort(unsupported.begin(), unsupported.end());
    const size_t total = unsupported.size();
    if (unsupported.size() > 6) {
        const size_t rest = unsupported.size() - 6;
        unsupported.resize(6);
        unsupported.push_back("and " + std::to_string(rest) + " more");
    }
    // Two different refusals: a kernel that cannot write the dtype (fix the
    // optimizer or the storage) and a device that declines it (fix the
    // backend). Saying "F32-only" in the second case would send the reader to
    // fix the wrong thing.
    set_error(backend_refused
            ? "the " + join_patterns(offenders) + " update step does not write "
                      + join_patterns(refused_dtypes) + " on " + state.backend_registry
                      + ", and this run marks " + std::to_string(total)
                      + " parameter(s) it cannot write: [" + join_patterns(unsupported)
                      + "]. Train these tensors in F32, or run on a backend whose "
                        "update step carries the precision"
            : "the " + join_patterns(offenders) + " update step is F32-only, and this run marks "
                      + std::to_string(total) + " parameter(s) it cannot write: ["
                      + join_patterns(unsupported)
                      + "]. Store the adapter as F32 (lora.dtype = \"f32\"), or use adamw or "
                        "sgd, whose steps round a half-precision store");
    return false;
}

// One admitted (dtype, optimizer, backend, master) combination for a *base*
// weight. Mirrors BASE_DTYPE_TABLE in retrograd-core; only a measured backend
// is admitted, and the device probe is asked as well.
// tests/f16_base_training.rs checks the two answers agree.
//
// `master` is part of the key: with a copy the step runs in F32 and casts
// once, without one it rounds the sum back into the store.
struct base_dtype_row {
    ggml_type    type;
    int32_t      optimizer;
    const char * backend;  // ggml_backend_reg_name's spelling
    bool         master;
};

static const base_dtype_row BASE_DTYPE_TABLE[] = {
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "CPU",  false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "CUDA", false },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_ADAMW, "CPU",  false },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_ADAMW, "CUDA", false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "CPU",  false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "CUDA", false },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_SGD,   "CPU",  false },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_SGD,   "CUDA", false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "CPU",  true  },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_ADAMW, "CPU",  true  },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "CPU",  true  },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_SGD,   "CPU",  true  },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "CUDA", true  },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_ADAMW, "CUDA", true  },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "CUDA", true  },
    { GGML_TYPE_BF16, RETRO_OPTIMIZER_SGD,   "CUDA", true  },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "MTL",  false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "MTL",  false },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_ADAMW, "MTL",  true  },
    { GGML_TYPE_F16,  RETRO_OPTIMIZER_SGD,   "MTL",  true  },
};

static bool base_dtype_is_tabled(
        ggml_type type, int32_t optimizer, const std::string & backend, bool master) {
    for (const base_dtype_row & row : BASE_DTYPE_TABLE) {
        if (row.type == type && row.optimizer == optimizer && backend == row.backend
                && row.master == master) {
            return true;
        }
    }
    return false;
}

// The backends the table carries for one (dtype, optimizer, master) triple, so
// a refusal can name where the combination does run.
static std::vector<std::string> base_dtype_backends(
        ggml_type type, int32_t optimizer, bool master) {
    std::vector<std::string> backends;
    for (const base_dtype_row & row : BASE_DTYPE_TABLE) {
        if (row.type == type && row.optimizer == optimizer && row.master == master) {
            backends.emplace_back(row.backend);
        }
    }
    return backends;
}

// The same admission over the declared base set, before llama_opt_init.
// A dtype llama_set_param does not admit is never marked, so the marked-set
// check above would never see it and rule 6 would report it as "declared but
// not marked". This check names the tensor, its dtype, its owner and the
// backend, which is the message that says what to change.
bool declared_base_dtypes_are_admitted(const trainer_state & state) {
    if (!trains_base_weights(state) || state.trainable_base.empty()) {
        return true;
    }
    std::vector<std::string> unsupported;
    std::vector<std::string> offenders;
    // The refused (dtype, optimizer) pairs, so the message can name the
    // backends that do carry them.
    std::vector<std::pair<ggml_type, int32_t>> refused;
    const bool master = master_weights_enabled(state);
    for (const std::string & name : state.trainable_base) {
        const ggml_tensor * tensor = nullptr;
        for (const auto & item : state.model->tensors_by_name) {
            if (item.first == name) {
                tensor = item.second;
                break;
            }
        }
        if (!tensor) {
            continue;  // resolved against the same file; the marked-set check owns this
        }
        const int32_t optimizer = retro_optimizer_of(
                opt_param_optimizer(tensor, const_cast<trainer_state *>(&state)));
        // Table first, device probe right after; either can refuse.
        // LoRA keeps its own policy in optimizer_supports_marked_dtypes.
        const bool admitted = tensor->type == GGML_TYPE_F32
                || base_dtype_is_tabled(tensor->type, optimizer, state.backend_registry, master);
        // With a master copy the step writes the F32 copy, so the probe is
        // asked about F32; the cast into the store exists on every backend.
        const ggml_type stepped = master ? GGML_TYPE_F32 : tensor->type;
        if (admitted && optimizer_supports_dtype(state, optimizer, stepped)) {
            continue;
        }
        unsupported.push_back(name + " (" + ggml_type_name(tensor->type) + ", "
                + optimizer_name(optimizer) + ")");
        if (std::find(offenders.begin(), offenders.end(), optimizer_name(optimizer))
                == offenders.end()) {
            offenders.push_back(optimizer_name(optimizer));
        }
        const std::pair<ggml_type, int32_t> pair { tensor->type, optimizer };
        if (std::find(refused.begin(), refused.end(), pair) == refused.end()) {
            refused.push_back(pair);
        }
    }
    if (unsupported.empty()) {
        return true;
    }
    std::sort(unsupported.begin(), unsupported.end());
    const size_t total = unsupported.size();
    if (unsupported.size() > 6) {
        const size_t rest = unsupported.size() - 6;
        unsupported.resize(6);
        unsupported.push_back("and " + std::to_string(rest) + " more");
    }
    // Where each refused combination does run, from the same table.
    std::vector<std::string> elsewhere;
    for (const auto & pair : refused) {
        // A F32-only kernel refuses on every backend, so it says so instead
        // of naming an empty list.
        if (!optimizer_kernel_writes_dtype(pair.second, pair.first)) {
            elsewhere.push_back(std::string(optimizer_name(pair.second))
                    + "'s update step writes F32 only, on every backend");
            continue;
        }
        const std::vector<std::string> backends =
                base_dtype_backends(pair.first, pair.second, master);
        elsewhere.push_back(std::string(ggml_type_name(pair.first)) + " under "
                + optimizer_name(pair.second)
                + (backends.empty()
                        ? " is measured on no backend yet"
                        : " is measured on " + join_patterns(backends)));
    }
    set_error("the declared trainable set carries " + std::to_string(total)
            + " base tensor(s) not admitted for " + join_patterns(offenders) + " on "
            + state.backend_registry + ": [" + join_patterns(unsupported) + "]. "
            + join_patterns(elsewhere) + "; store these tensors as F32 to train them here");
    return false;
}

// Copies of MIN_BASE_STEP_ULPS and MAX_BASE_STEP_UNDERFLOW_SHARE from
// retrograd-core/src/base_dtype.rs; tests/capabilities.rs checks they agree.
static const float MIN_BASE_STEP_ULPS = 0.125f;
static const float MAX_BASE_STEP_UNDERFLOW_SHARE = 0.5f;

// The floor, exposed for the capability report.
float base_step_min_ulps() {
    return MIN_BASE_STEP_ULPS;
}

// Significand bits below the leading one, or 0 for types this rule does not
// grade (F32 and every quantization).
static int store_significand_bits(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return 10;
        case GGML_TYPE_BF16: return 7;
        default:             return 0;
    }
}

// One ulp of `type`'s grid at `value`; zero and subnormals use the smallest
// normal's ulp.
static float storage_ulp(ggml_type type, float value, int significand_bits) {
    const float smallest_normal = type == GGML_TYPE_F16 ? 6.1035156e-5f : FLT_MIN;
    const float magnitude = std::fabs(value);
    if (!std::isfinite(magnitude) || magnitude < smallest_normal) {
        return std::ldexp(1.0f, std::ilogb(smallest_normal) - significand_bits);
    }
    return std::ldexp(1.0f, std::ilogb(magnitude) - significand_bits);
}

// Three significant digits, because the values here span ten orders of
// magnitude.
std::string format_significant(float value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3g", value);
    return std::string(buffer);
}

// The mean ulp of a sample of a tensor's weights, or false when the sample
// carries nothing to average. Sampled, not read whole: the marked set of a
// full run is the size of the model. The read goes through the backend
// because a base weight may be device-resident.
static bool sampled_mean_ulp(const ggml_tensor * tensor, int significand_bits, float & out_ulp) {
    const int64_t n_elements = ggml_nelements(tensor);
    if (n_elements <= 0 || !ggml_is_contiguous(tensor)) {
        return false;
    }
    // Widening: a count clamped to 4096 fits every size type below.
    const size_t sampled = static_cast<size_t>(std::min<int64_t>(n_elements, 4096));
    std::vector<uint16_t> half(sampled);
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), half.data(), 0,
            half.size() * sizeof(uint16_t));
    double total = 0.0;
    int64_t counted = 0;
    for (const uint16_t bits : half) {
        const float value = tensor->type == GGML_TYPE_BF16
                ? ggml_bf16_to_fp32(ggml_bf16_t { bits })
                : ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(bits));
        if (!std::isfinite(value)) {
            continue;
        }
        total += static_cast<double>(storage_ulp(tensor->type, value, significand_bits));
        ++counted;
    }
    if (counted == 0) {
        return false;
    }
    out_ulp = static_cast<float>(total / static_cast<double>(counted));
    return out_ulp > 0.0f;
}

// One marked half-precision tensor's grid and the rate it is stepped at.
struct base_step_entry {
    std::string  name;
    ggml_type    type;
    float        ulp;
    float        step;
    int64_t      n_elements;
};

// Whether the per-element update this run intends is large enough for the
// half-precision stores it is written to. Without a master copy a step below
// one ulp is a whole ulp taken with probability step/ulp, so the weights
// follow the rounding rather than the gradient. With a master copy there is
// no in-place store, so the run is reported but never refused.
//
// Only optimizers whose step is about alpha per element are graded (AdamW,
// Gefen). SGD's step is alpha*|gradient|, which no preflight can know.
bool declared_base_steps_are_representable(trainer_state & state) {
    state.base_step_ulps = -1.0f;
    if (!trains_base_weights(state) || state.trainable_base.empty()) {
        return true;
    }
    const retro_train_config & config = state.train_config;
    std::vector<base_step_entry> entries;
    int64_t total_elements = 0;
    int64_t under_elements = 0;
    double step_ulps_total = 0.0;
    for (const std::string & name : state.trainable_base) {
        const ggml_tensor * tensor = nullptr;
        for (const auto & item : state.model->tensors_by_name) {
            if (item.first == name) {
                tensor = item.second;
                break;
            }
        }
        if (!tensor) {
            continue;  // the marked-set check owns names that resolve to nothing
        }
        const int significand_bits = store_significand_bits(tensor->type);
        if (significand_bits == 0) {
            continue;  // F32, or a dtype the admission above already refused
        }
        const int32_t optimizer = retro_optimizer_of(opt_param_optimizer(tensor, &state));
        if (optimizer != RETRO_OPTIMIZER_ADAMW && optimizer != RETRO_OPTIMIZER_GEFEN) {
            continue;
        }
        // A Muon run steps the parameters Muon declines at its fallback rate.
        const bool muon_fallback = optimizer == RETRO_OPTIMIZER_ADAMW
                && config.optimizer == RETRO_OPTIMIZER_MUON
                && config.muon_fallback_learning_rate > 0.0f;
        const float step = muon_fallback
                ? config.muon_fallback_learning_rate
                : config.learning_rate;
        float ulp = 0.0f;
        if (!sampled_mean_ulp(tensor, significand_bits, ulp)) {
            continue;
        }
        const int64_t n_elements = ggml_nelements(tensor);
        entries.push_back(base_step_entry { name, tensor->type, ulp, step, n_elements });
        total_elements += n_elements;
        step_ulps_total += static_cast<double>(step / ulp) * static_cast<double>(n_elements);
        if (step < MIN_BASE_STEP_ULPS * ulp) {
            under_elements += n_elements;
        }
    }
    if (total_elements == 0) {
        return true;  // nothing graded here
    }
    // One number for the whole marked set, weighted by element count; the
    // report prints it whether or not the run is refused.
    state.base_step_ulps =
            static_cast<float>(step_ulps_total / static_cast<double>(total_elements));
    if (master_weights_enabled(state)) {
        // The number above is still reported, but it bounds nothing: the sum
        // is formed in F32 and the store is written by one cast.
        return true;
    }
    const double share = static_cast<double>(under_elements) / static_cast<double>(total_elements);
    if (share <= static_cast<double>(MAX_BASE_STEP_UNDERFLOW_SHARE)) {
        return true;
    }
    // The tensor whose step is the smallest fraction of its own grid.
    const base_step_entry * worst = &entries.front();
    for (const base_step_entry & entry : entries) {
        if (entry.step / entry.ulp < worst->step / worst->ulp) {
            worst = &entry;
        }
    }
    // The rate that would bring the share back under the cap: MIN_BASE_STEP_ULPS
    // ulps of the grid at the tolerated quantile.
    std::vector<base_step_entry> by_ulp = entries;
    std::sort(by_ulp.begin(), by_ulp.end(),
            [](const base_step_entry & a, const base_step_entry & b) { return a.ulp < b.ulp; });
    const double carried_target = static_cast<double>(total_elements)
            * (1.0 - static_cast<double>(MAX_BASE_STEP_UNDERFLOW_SHARE));
    double carried = 0.0;
    float quantile_ulp = by_ulp.back().ulp;
    for (const base_step_entry & entry : by_ulp) {
        carried += static_cast<double>(entry.n_elements);
        if (carried >= carried_target) {
            quantile_ulp = entry.ulp;
            break;
        }
    }
    const int percent = static_cast<int>(share * 100.0 + 0.5);
    // If the run explicitly turned the master copy off, stop suggesting it.
    const bool declined = config.master_weights == RETRO_MASTER_WEIGHTS_OFF;
    const std::string remedy = declined
            ? std::string("Set training.master_weights = 'f32' to accumulate the update in an "
                          "F32 copy (four bytes per trained element), raise training.lr to ")
            : std::string("Set training.master_weights, train from an F32 model file, raise "
                          "training.lr to ");
    set_error("the learning rate " + format_significant(config.learning_rate)
            + " is below what this run's " + ggml_type_name(worst->type)
            + " base weights can carry: " + std::to_string(percent)
            + "% of the trained parameters take an update under "
            + format_significant(MIN_BASE_STEP_ULPS) + " ulp of their own store (worst "
            + worst->name + ": ulp " + format_significant(worst->ulp) + ", update "
            + format_significant(worst->step / worst->ulp) + " ulp). This run has no F32 "
              "master copy: the update step rounds the sum back onto the same grid, so a step "
              "under one ulp is a whole ulp taken at random and the weights follow the rounding "
              "rather than the gradient. "
            + remedy
            + format_significant(MIN_BASE_STEP_ULPS * quantile_ulp)
            + " or above, or train a LoRA adapter, whose factors are stored F32");
    return false;
}

bool assert_assignment_covers_marked_set(const trainer_state & state) {
    if (state.optimizer_assignment.empty()) {
        return true;
    }
    std::vector<std::string> marked;
    for_each_marked(state, [&](const ggml_tensor * tensor, int32_t) {
        marked.emplace_back(tensor->name);
    });
    std::vector<std::string> unmatched;
    for (const auto & row : state.optimizer_assignment) {
        if (std::find(marked.begin(), marked.end(), row.first) == marked.end()) {
            unmatched.push_back(row.first);
        }
    }
    if (unmatched.empty()) {
        return true;
    }
    std::sort(unmatched.begin(), unmatched.end());
    set_error("the optimizer assignment names " + std::to_string(unmatched.size())
              + " parameter(s) this run does not train: [" + join_patterns(unmatched)
              + "]");
    return false;
}

bool assert_marked_set_is_resolved(const trainer_state & state) {
    // The adapter half first, unchanged: a LoRA run with no flagged factor has
    // nothing to train whatever the base policy says.
    const bool expects_adapter = state.adapter != nullptr;
    if (!expects_adapter && !trains_base_weights(state)) {
        set_error("LoRA adapter is not initialized");
        return false;
    }
    if (expects_adapter && count_lora_param_tensors(state) == 0) {
        set_error("no LoRA tensors are marked trainable");
        return false;
    }

    // The base half. `marked` is what llama_opt_init actually flagged;
    // `state.trainable_base` is what the resolver said it would. Equality both
    // ways is the point: a missing name means a tensor was selected and never
    // reached - llama_set_param refuses a non-F32 tensor and opt_init visits
    // only the members it enumerates - and an extra one means the filter
    // admitted something nobody asked for.
    std::set<std::string> marked;
    for (const auto & item : state.model->tensors_by_name) {
        if (is_param_tensor(item.second)) {
            marked.insert(item.first);
        }
    }
    const std::set<std::string> resolved(
            state.trainable_base.begin(), state.trainable_base.end());
    if (marked == resolved) {
        return true;
    }

    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    for (const std::string & name : resolved) {
        if (marked.find(name) == marked.end()) {
            missing.push_back(name);
        }
    }
    for (const std::string & name : marked) {
        if (resolved.find(name) == resolved.end()) {
            unexpected.push_back(name);
        }
    }
    std::ostringstream message;
    message << "the trainable set the optimizer marked is not the one that was resolved";
    if (!missing.empty()) {
        message << "; selected but not marked (" << missing.size() << "): ["
                << join_patterns(missing) << "]";
    }
    if (!unexpected.empty()) {
        message << "; marked but not selected (" << unexpected.size() << "): ["
                << join_patterns(unexpected) << "]";
    }
    set_error(message.str());
    return false;
}

bool save_lora_adapter_gguf(const trainer_state & state, const char * adapter_path) {
    if (!state.adapter) {
        set_error("LoRA adapter is not initialized");
        return false;
    }
    if (!validate_lora_tensor_pairs(*state.adapter)) {
        return false;
    }

    gguf_context_ptr gguf(gguf_init_empty());
    if (!gguf) {
        set_error("failed to allocate GGUF export context");
        return false;
    }

    gguf_set_val_str(gguf.get(), "general.type", "adapter");
    gguf_set_val_str(gguf.get(), "general.architecture", state.model->arch_name().c_str());
    gguf_set_val_str(gguf.get(), "adapter.type", "lora");
    gguf_set_val_f32(gguf.get(), "adapter.lora.alpha", state.adapter->alpha);

    for (const auto & item : state.adapter->ab_map) {
        gguf_add_tensor(gguf.get(), item.second.a);
        gguf_add_tensor(gguf.get(), item.second.b);
    }

    if (!gguf_write_to_file(gguf.get(), adapter_path, false)) {
        set_error("failed to write LoRA GGUF adapter: " + std::string(adapter_path));
        return false;
    }

    lora_adapter_ptr reloaded(llama_adapter_lora_init(state.model.get(), adapter_path));
    if (!reloaded) {
        set_error("wrote LoRA GGUF but could not reload it with llama_adapter_lora_init(): "
                  + std::string(adapter_path));
        return false;
    }

    return true;
}

bool promote_loaded_lora_to_trainable(trainer_state & state) {
    if (state.lora_promoted) {
        return true;
    }
    if (!state.adapter) {
        set_error("LoRA adapter is not initialized");
        return false;
    }
    if (!state.adapter->alora_invocation_tokens.empty()) {
        set_error("training activated LoRA (aLoRA) adapters is not supported");
        return false;
    }
    if (state.adapter->ab_map.empty()) {
        set_error("loaded LoRA adapter has no tensor pairs");
        return false;
    }
    if (!validate_lora_tensor_pairs(*state.adapter)) {
        return false;
    }

    auto ends_with = [](const std::string & name, const char * suffix) {
        const size_t n = std::strlen(suffix);
        return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
    };

    for (const auto & item : state.adapter->ab_map) {
        const ggml_tensor * tensor_a = item.second.a;
        const ggml_tensor * tensor_b = item.second.b;
        if (!tensor_a || !tensor_b) {
            set_error(tensor_pair_error(item.first));
            return false;
        }
        // The embedding pair carries flipped A/B semantics in the forward
        // graph (see llm_build_inp_embd), which the training path does not
        // model; the create path never produces it either.
        if (ends_with(item.first, "token_embd.weight")) {
            set_error("training a LoRA adapter that targets token_embd.weight "
                      "is not supported");
            return false;
        }
    }

    for (auto & item : state.adapter->ab_map) {
        ggml_set_param(item.second.a);
        ggml_set_param(item.second.b);
    }

    const ggml_tensor * first_a = state.adapter->ab_map.begin()->second.a;
    state.lora_rank = static_cast<uint32_t>(first_a->ne[1]);
    state.lora_alpha = state.adapter->alpha;
    state.lora_dtype = first_a->type == GGML_TYPE_F16
            ? RETRO_LORA_DTYPE_F16 : RETRO_LORA_DTYPE_F32;
    state.lora_promoted = true;
    state.invalidate_report_caches();
    return true;
}

bool create_trainable_lora_adapter(trainer_state & state, uint32_t seed) {
    const uint32_t rank = state.lora_rank;
    const ggml_type lora_type = state.lora_dtype == RETRO_LORA_DTYPE_F16
            ? GGML_TYPE_F16 : GGML_TYPE_F32;
    std::vector<const ggml_tensor *> targets;
    if (!resolve_lora_targets(state, targets)) {
        return false;
    }

    std::unique_ptr<llama_adapter_lora> adapter(new llama_adapter_lora(state.model.get()));
    adapter->alpha = state.lora_alpha;
    adapter->gguf_kv.emplace("general.type", "adapter");
    adapter->gguf_kv.emplace("adapter.type", "lora");

    auto buft_of = [](const ggml_tensor * tensor) -> ggml_backend_buffer_type_t {
        if (tensor->buffer) {
            return ggml_backend_buffer_get_type(tensor->buffer);
        }
        return ggml_backend_cpu_buffer_type();
    };

    std::map<ggml_backend_buffer_type_t, std::vector<const ggml_tensor *>> targets_by_buft;
    for (const ggml_tensor * model_tensor : targets) {
        targets_by_buft[buft_of(model_tensor)].push_back(model_tensor);
    }

    for (const auto & group : targets_by_buft) {
        ggml_backend_buffer_type_t buft = group.first;
        const std::vector<const ggml_tensor *> & group_targets = group.second;

        ggml_init_params params {
            /*.mem_size   =*/ group_targets.size() * 2 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            set_error("failed to allocate LoRA tensor metadata context");
            return false;
        }
        adapter->ctxs.emplace_back(ctx);

        for (const ggml_tensor * model_tensor : group_targets) {
            ggml_tensor * tensor_a =
                    ggml_new_tensor_2d(ctx, lora_type, model_tensor->ne[0], rank);
            ggml_tensor * tensor_b =
                    ggml_new_tensor_2d(ctx, lora_type, rank, model_tensor->ne[1]);
            if (!tensor_a || !tensor_b) {
                set_error("failed to allocate LoRA tensor metadata");
                return false;
            }
            ggml_format_name(tensor_a, "%s.lora_a", model_tensor->name);
            ggml_format_name(tensor_b, "%s.lora_b", model_tensor->name);
            ggml_set_param(tensor_a);
            ggml_set_param(tensor_b);
            adapter->ab_map.emplace(
                    model_tensor->name, llama_adapter_lora_weight(tensor_a, tensor_b));
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
        if (!buffer) {
            set_error("failed to allocate LoRA tensor buffer for buft: "
                      + std::string(ggml_backend_buft_name(buft)));
            return false;
        }
        ggml_backend_buffer_clear(buffer.get(), 0);
        adapter->bufs.emplace_back(std::move(buffer));
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    std::vector<float> values;
    std::vector<ggml_fp16_t> values_f16;
    for (auto & item : adapter->ab_map) {
        ggml_tensor * tensor_a = item.second.a;
        values.resize(static_cast<size_t>(ggml_nelements(tensor_a)));
        for (float & value : values) {
            value = dist(rng);
        }
        if (lora_type == GGML_TYPE_F16) {
            values_f16.resize(values.size());
            ggml_fp32_to_fp16_row(values.data(), values_f16.data(), values.size());
            ggml_backend_tensor_set(
                    tensor_a, values_f16.data(), 0, values_f16.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(tensor_a, values.data(), 0, values.size() * sizeof(float));
        }
    }

    state.model->loras.insert(adapter.get());

    llama_adapter_lora * adapters[] = { adapter.get() };
    float scales[] = { 1.0f };
    const int32_t status = llama_set_adapters_lora(state.ctx.get(), adapters, 1, scales);
    if (status != 0) {
        state.model->loras.erase(adapter.get());
        set_error("failed to apply trainable LoRA adapter to context");
        return false;
    }
    if (state.generation_ctx
            && llama_set_adapters_lora(state.generation_ctx.get(), adapters, 1, scales) != 0) {
        llama_set_adapters_lora(state.ctx.get(), nullptr, 0, nullptr);
        state.model->loras.erase(adapter.get());
        set_error("failed to apply trainable LoRA adapter to generation context");
        return false;
    }

    state.adapter.reset(adapter.release());
    return true;
}

} // namespace retro
