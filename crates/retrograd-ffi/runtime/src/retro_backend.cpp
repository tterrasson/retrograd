#include "retro_runtime.hpp"
#include "llama-context.h"
#include "ggml-rir/ggml-rir.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace retro {

ggml_backend_dev_t first_gpu_device() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return dev;
            }
        }
    }
    return nullptr;
}

namespace {

uint32_t automatic_thread_count() {
#if defined(__APPLE__)
    uint32_t performance_cores = 0;
    size_t size = sizeof(performance_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &size, nullptr, 0) == 0
            && performance_cores > 0) {
        return performance_cores;
    }
#endif
    const uint32_t hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? hardware : 4;
}

bool resolve_thread_count(const retro_train_config & config, uint32_t & out) {
    out = config.threads > 0 ? config.threads : automatic_thread_count();
    const char * override_value = std::getenv("RETRO_THREADS");
    if (!override_value || override_value[0] == '\0') {
        return true;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(override_value, &end, 10);
    if (errno != 0 || end == override_value || *end != '\0' || parsed == 0
            || parsed > std::numeric_limits<uint32_t>::max()) {
        set_error("RETRO_THREADS must be an integer greater than zero");
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

const char * device_name(ggml_backend_dev_t device) {
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    if (name && name[0] != '\0') {
        return name;
    }
    ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return reg_name && reg_name[0] != '\0' ? reg_name : "unknown";
}

const char * buffer_type_label(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) {
        return "unallocated";
    }
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
}

// Byte-level memory accounting for one llama_context, aggregated per backend
// buffer type. `model` is the weight allocation (shared with any other context
// built on the same model, so it must be counted once), `kv` is the KV / recurrent
// cache, `compute` is the scheduler's reserved activation buffer (only non-zero
// once a graph has been reserved -- i.e. after preflight).
struct context_memory {
    size_t model   = 0;
    size_t kv      = 0;
    size_t compute = 0;
    bool   is_host = false;
};

// Keyed by backend buffer-type label (e.g. "CUDA0", "CPU", "Metal"). Ordered so
// the report is stable across runs.
std::map<std::string, context_memory> context_memory_by_buffer(const llama_context * ctx) {
    std::map<std::string, context_memory> by_label;
    if (!ctx) {
        return by_label;
    }
    for (const auto & item : ctx->memory_breakdown()) {
        const char * name = ggml_backend_buft_name(item.first);
        context_memory & entry = by_label[name && name[0] ? name : "unknown"];
        entry.model   += item.second.model;
        entry.kv      += item.second.context;
        entry.compute += item.second.compute;
        entry.is_host  = ggml_backend_buft_is_host(item.first);
    }
    return by_label;
}

// Ask the active device whether both differentiable Flash Attention operations
// support every attention layer's real head geometry and the requested KV type.
// Capability probing avoids backend-name assumptions and makes unsupported
// shapes fall back explicitly.
bool supports_flash_attn_back(
        ggml_backend_dev_t device, const llama_model & model, uint32_t n_tokens,
        ggml_type kv_type) {
    if (!device) {
        return false;
    }
    bool saw_attention_layer = false;
    const uint32_t tokens = std::max<uint32_t>(n_tokens, 1);
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        const int64_t hsk = model.hparams.n_embd_head_k(il);
        const int64_t hsv = model.hparams.n_embd_head_v(il);
        const int64_t n_head = model.hparams.n_head(il);
        const int64_t n_head_kv = model.hparams.n_head_kv(il);
        // Recurrent-only layers do not build multi-head attention.
        if (hsk == 0 || hsv == 0 || n_head == 0 || n_head_kv == 0) {
            continue;
        }
        saw_attention_layer = true;

        ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 16,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> probe(ggml_init(params), &ggml_free);
        if (!probe) {
            return false;
        }
        // Use the padded KV-cache length from the real graph. Backend support can
        // depend on that shape, so probing only the current ubatch may reject a
        // model whose actual graph is supported.
        const int64_t n_kv = GGML_PAD((int64_t) tokens, 256);
        ggml_tensor * q = ggml_new_tensor_4d(
                probe.get(), GGML_TYPE_F32, hsk, tokens, n_head, 1);
        ggml_tensor * k = ggml_new_tensor_4d(
                probe.get(), kv_type, hsk, n_kv, n_head_kv, 1);
        ggml_tensor * v = ggml_new_tensor_4d(
                probe.get(), kv_type, hsv, n_kv, n_head_kv, 1);
        ggml_tensor * mask = ggml_new_tensor_4d(
                probe.get(), GGML_TYPE_F16, n_kv, tokens, 1, 1);
        ggml_tensor * dout = ggml_new_tensor_4d(
                probe.get(), GGML_TYPE_F32, hsv, n_head, tokens, 1);
        ggml_tensor * out = ggml_flash_attn_ext(
                probe.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        // Probe the dense backward: the KV gradient window is an optional extra
        // input the same kernels accept, so a device that supports this supports
        // the windowed form too.
        ggml_tensor * back = ggml_flash_attn_ext_back(
                probe.get(), q, k, v, mask, out, dout, nullptr,
                nullptr, nullptr, nullptr, 0, 0,
                GGML_FLASH_ATTN_BACK_GRAD_Q | GGML_FLASH_ATTN_BACK_GRAD_K | GGML_FLASH_ATTN_BACK_GRAD_V,
                1.0f, 0.0f, 0.0f);
        if (!out || !back || !ggml_backend_dev_supports_op(device, out)
                || !ggml_backend_dev_supports_op(device, back)) {
            return false;
        }
    }
    // A model with no attention layers (pure recurrent) cannot use this path.
    return saw_attention_layer;
}

// The model's projection head, i.e. the `w` the fused cross-entropy consumes.
// Null when the model ties the head to the token embedding, in which case the
// embedding matrix is what the graph reads.
const ggml_tensor * projection_head(const llama_model & model) {
    return model.output ? model.output : model.tok_embd;
}

// Whether the active device can run the fused sparse cross-entropy pair for
// this model's real head geometry and head weight type, at the tiling the
// config requests. Both nodes are probed because `chunked_cross_entropy` emits
// both, and a device that declines either sends the whole tail to the CPU --
// which then transfers the hidden states and the entire (quantized) head per
// token chunk, i.e. the memory profile of the fused mode with the traffic of
// the dense one. That fallback is silent today; this is the value that lets the
// report say it.
bool supports_fused_sparse_ce(
        ggml_backend_dev_t device, const llama_model & model,
        const retro_train_config & config, uint32_t n_tokens) {
    if (!device) {
        return false;
    }
    const ggml_tensor * head = projection_head(model);
    if (!head) {
        return false;
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return false;
    }
    const int64_t n_embd  = head->ne[0];
    const int64_t n_vocab = head->ne[1];
    const int64_t tokens  = std::max<int64_t>(n_tokens, 1);
    ggml_tensor * h   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, tokens);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx.get(), head->type, n_embd, n_vocab);
    // retro delta (plan DISTILL D6.5): [K, n_tokens]. The capability this probe
    // asks about is the operator's, not a particular k, and every backend gate
    // accepts k = 1; a run that then asks for more falls back per node.
    ggml_tensor * tgt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, tokens);
    ggml_tensor * wgt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, tokens);
    // A head bias is part of the op signature and backends disagree on it
    // (Vulkan refuses it outright), so the probe carries one exactly when the
    // model has one.
    ggml_tensor * bias = model.output_b
            ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab)
            : nullptr;
    if (!h || !w || !tgt || !wgt || (model.output_b && !bias)) {
        return false;
    }
    const int n_tiles = (int) std::max<uint32_t>(config.chunked_ce_tiles, 1);
    const int seq_chunk = (int) config.chunked_ce_seq_chunk;
    const int offload = config.chunked_ce_offload_logsoftmax ? 1 : 0;
    ggml_tensor * grad = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * loss = ggml_fused_sparse_ce(
            ctx.get(), h, w, tgt, wgt, bias, n_tiles, seq_chunk, offload);
    ggml_tensor * back = grad ? ggml_fused_sparse_ce_back(
            ctx.get(), grad, h, w, tgt, wgt, bias, n_tiles, seq_chunk, offload) : nullptr;
    return loss && back
            && ggml_backend_dev_supports_op(device, loss)
            && ggml_backend_dev_supports_op(device, back);
}

// How the active backend feeds a quantized base weight to the backward's
// `OUT_PROD` (the gradient of a frozen projection's input). Unlike every
// other `cap_*` helpers, this is not a capability probe: the three strategies are
// compile-time properties of the backend's kernels, not something ggml can be
// asked. It is reported because it is the difference between a backward that
// touches no scratch and one that allocates an F32 copy of the weight, and
// because without it a regression to the pre-A1 full-tensor scratch is
// invisible in every counter the project has.
//   native        -- the kernel dequantizes each block inline; zero scratch
//                    (Vulkan reads the quantized rows in the shader, Metal in
//                    kernel_out_prod_k)
//   tiled_dequant -- an F32 scratch bounded by a tile budget (CUDA, A1)
//   full_dequant  -- an F32 copy of the whole weight
// The probe still decides the *reachability*: a type the device declines never
// reaches any of these paths, it goes to the CPU.
const char * quantized_backward_path(
        ggml_backend_dev_t device, const llama_model & model) {
    ggml_type quantized = GGML_TYPE_COUNT;
    size_t quantized_bytes = 0;
    for (const auto & item : model.tensors_by_name) {
        const ggml_tensor * tensor = item.second;
        if (!tensor || !ggml_is_quantized(tensor->type)) {
            continue;
        }
        const size_t bytes = ggml_nbytes(tensor);
        if (bytes > quantized_bytes) {
            quantized_bytes = bytes;
            quantized = tensor->type;
        }
    }
    if (quantized == GGML_TYPE_COUNT) {
        return "not_applicable";  // float base weights: no dequantization at all
    }
    if (!device) {
        return "cpu";
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return "unknown";
    }
    // out_prod(w, dy): w is the frozen quantized weight, dy the output gradient.
    const int64_t block = ggml_blck_size(quantized);
    ggml_tensor * w  = ggml_new_tensor_2d(ctx.get(), quantized, block * 2, 8);
    ggml_tensor * dy = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    ggml_tensor * dx = w && dy ? ggml_out_prod(ctx.get(), w, dy) : nullptr;
    if (!dx || !ggml_backend_dev_supports_op(device, dx)) {
        return "cpu_fallback";
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    const std::string backend = reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "";
    // Registry names, not device names: GGML_CUDA_NAME / GGML_VK_NAME /
    // GGML_METAL_NAME as the backends spell them ("MTL" for Metal).
    if (backend == "CUDA" || backend == "ROCm" || backend == "MUSA") {
        return "tiled_dequant";
    }
    if (backend == "MTL" || backend == "Vulkan") {
        return "native";
    }
    return "unknown";
}

// Whether the active device can run the token-sampling ops (argsort for
// top-k/top-p ordering, plus a row softmax) on device. On-device sampling keeps
// the logits on the GPU instead of copying the vocabulary row to the host every
// step. Probed rather than inferred from the backend name.
bool supports_device_sampling(ggml_backend_dev_t device) {
    if (!device) {
        return false;
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return false;
    }
    const int64_t n_vocab = 256;
    ggml_tensor * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
    ggml_tensor * softmax = logits ? ggml_soft_max(ctx.get(), logits) : nullptr;
    ggml_tensor * order = logits ? ggml_argsort(ctx.get(), logits, GGML_SORT_ORDER_DESC) : nullptr;
    return softmax && order
            && ggml_backend_dev_supports_op(device, softmax)
            && ggml_backend_dev_supports_op(device, order);
}

// Whether the active device can gather a target log-probability inside the
// decode graph (llama_set_target_logprobs): a row softmax over the logits plus
// a get_rows on the flattened result. Both are ordinary ops, but a device that
// declines either would have the scheduler stream the full vocabulary back to
// the host to run them there -- strictly worse than the host reduction this
// replaces, so it is probed rather than assumed.
bool supports_device_logprob_gather(ggml_backend_dev_t device) {
    if (!device) {
        return false;
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return false;
    }
    const int64_t n_vocab = 256;
    const int64_t n_rows = 4;
    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_vocab, n_rows);
    ggml_tensor * softmax = logits ? ggml_soft_max(ctx.get(), logits) : nullptr;
    ggml_tensor * flat = softmax
            ? ggml_reshape_2d(ctx.get(), softmax, 1, ggml_nelements(softmax))
            : nullptr;
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_rows);
    ggml_tensor * gathered = flat && rows ? ggml_get_rows(ctx.get(), flat, rows) : nullptr;
    return softmax && gathered
            && ggml_backend_dev_supports_op(device, softmax)
            && ggml_backend_dev_supports_op(device, gathered);
}

// Whether `device` can run `optimizer`'s update step on a parameter of `type`.
// Asking ggml rather than reading a name keeps the answer true when a kernel
// is added or a device changes. `nullptr` is the CPU device; it is looked up
// rather than assumed supported.
bool supports_opt_step_dtype(
        ggml_backend_dev_t device, int32_t optimizer, ggml_type type) {
    if (!device) {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!device) {
        return false;
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return false;
    }
    // A non-degenerate shape: per-row or per-simdgroup kernels refuse degenerate
    // shapes for reasons unrelated to the dtype.
    ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), type, 32, 2);
    if (!w) {
        return false;
    }
    // ggml_opt_step_* assert on a parameter without the flag. No_alloc is set
    // above, so nothing is allocated.
    ggml_set_param(w);
    ggml_tensor * g = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_tensor * step = nullptr;
    switch (optimizer) {
        case RETRO_OPTIMIZER_SGD: {
            ggml_tensor * pars = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
            if (!g || !pars) {
                return false;
            }
            // The kernel is F32-only; anything else would abort in its CPU path.
            if (type != GGML_TYPE_F32) {
                return false;
            }
            step = ggml_opt_step_sgd(ctx.get(), w, g, pars);
        } break;
        case RETRO_OPTIMIZER_ADAMW: {
            ggml_tensor * m = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
            ggml_tensor * v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
            // The declared parameters plus the gradient-clipping scale that
            // ggml_opt_build appends for AdamW; the step asserts on the total.
            ggml_tensor * pars = ggml_new_tensor_1d(
                    ctx.get(), GGML_TYPE_F32,
                    ggml_opt_optimizer_n_params(GGML_OPT_OPTIMIZER_TYPE_ADAMW) + 1);
            if (!g || !m || !v || !pars) {
                return false;
            }
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
                return false;  // ggml_opt_step_adamw asserts on anything else
            }
            step = ggml_opt_step_adamw(ctx.get(), w, g, m, v, pars);
        } break;
        default:
            return false;
    }
    return step && ggml_backend_dev_supports_op(device, step);
}

// Diagnostic: scan each computed node for non-finite values and log the first
// offender. Enabled with RETRO_CUDA_NAN_SCAN; used to localize a NaN to a
// specific graph op. Only F32 nodes are downloaded and scanned.
bool nan_scan_eval_callback(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return true; // request every node
    }
    if (!t || t->type != GGML_TYPE_F32 || !t->buffer) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || n > (int64_t) (64 * 1024 * 1024)) {
        return true;
    }
    static std::vector<float> host;
    host.resize((size_t) n);
    ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(host[i])) {
            fprintf(stderr,
                    "[nan-scan] first non-finite: op=%s name=%s shape=[%lld,%lld,%lld,%lld] at %lld = %g\n",
                    ggml_op_name(t->op), t->name,
                    (long long) t->ne[0], (long long) t->ne[1],
                    (long long) t->ne[2], (long long) t->ne[3],
                    (long long) i, host[i]);
            // Report the input tensors' finiteness/range so a NaN produced by an
            // op from finite inputs is distinguished from one merely propagated.
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * src = t->src[s];
                if (!src || src->type != GGML_TYPE_F32 || !src->buffer) {
                    continue;
                }
                const int64_t sn = ggml_nelements(src);
                if (sn <= 0 || sn > (int64_t) (64 * 1024 * 1024)) {
                    continue;
                }
                std::vector<float> sh((size_t) sn);
                ggml_backend_tensor_get(src, sh.data(), 0, ggml_nbytes(src));
                int64_t bad = 0;
                float lo = INFINITY, hi = -INFINITY;
                for (int64_t j = 0; j < sn; ++j) {
                    if (!std::isfinite(sh[j])) { ++bad; continue; }
                    lo = std::min(lo, sh[j]);
                    hi = std::max(hi, sh[j]);
                }
                fprintf(stderr,
                        "[nan-scan]   src[%d] op=%s name=%s type=%s ne=[%lld,%lld,%lld,%lld] nonfinite=%lld range=[%g,%g]\n",
                        s, ggml_op_name(src->op), src->name, ggml_type_name(src->type),
                        (long long) src->ne[0], (long long) src->ne[1],
                        (long long) src->ne[2], (long long) src->ne[3],
                        (long long) bad, lo, hi);
            }
            return false; // stop after the first offender
        }
    }
    return true;
}

// --- packed-sequence and finiteness probes ---------------------------------
// Both answer a question that used to be answered by an architecture name.
// llama_model_supports_packed_seq() states what the *graph* can be built on,
// which is a property of llama.cpp and belongs there. Neither it nor any name
// states what the *driver* then computes, and a driver is exactly where a
// supported graph shape still comes back as NaN. So the declaration is checked
// on the device at hand, once per load, and may only ever be downgraded.

// Frees the batch it owns whatever path leaves the probe.
struct probe_batch {
    llama_batch batch;
    probe_batch(int32_t n_tokens, int32_t n_seq_max)
        : batch(llama_batch_init(n_tokens, 0, n_seq_max)) {}
    ~probe_batch() { llama_batch_free(batch); }
    probe_batch(const probe_batch &) = delete;
    probe_batch & operator=(const probe_batch &) = delete;
};

// A synthetic token run. What these probes measure is the graph's arithmetic
// and its memory split, not what the model has to say, so the ids only have to
// be inside the vocabulary -- decode rejects the batch otherwise -- and to
// differ from each other, so that two packed sequences cannot agree by
// carrying the same tokens.
std::vector<llama_token> probe_token_run(
        const llama_model & model, uint32_t n_tokens, uint32_t stride) {
    const llama_vocab * vocab = llama_model_get_vocab(&model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens;
    if (n_vocab <= 0) {
        return tokens;
    }
    const llama_token bos = llama_vocab_bos(vocab);
    tokens.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (i == 0 && bos >= 0 && bos < n_vocab) {
            tokens.push_back(bos);
            continue;
        }
        tokens.push_back(static_cast<llama_token>((i * stride + 11) % n_vocab));
    }
    return tokens;
}

// Empties a probe context's sequence memory. A probe that left decoded cells
// behind would move the first real decode's position bookkeeping.
void clear_probe_memory(llama_context * ctx) {
    if (llama_memory_t memory = llama_get_memory(ctx)) {
        llama_memory_clear(memory, true);
    }
}

// Decodes one physical micro-batch of `n_tokens` and reports how many of the
// logits it produced are not finite.
// This replaces a workaround keyed on (architecture name, macOS, Vulkan,
// n_batch >= 16). What that rule encoded is a graph shape a driver evaluates to
// NaN for a short micro-batch; that is a property of the (model, driver,
// micro-batch) triple and of nothing else, so it is measured on the triple at
// hand and the rule follows from the measurement.
bool decode_logits_are_finite(
        llama_context * ctx, const llama_model & model,
        uint32_t n_tokens, uint64_t & out_nonfinite) {
    out_nonfinite = 0;
    if (n_tokens == 0) {
        return true;
    }
    const std::vector<llama_token> tokens = probe_token_run(model, n_tokens, 37);
    if (tokens.size() != n_tokens) {
        return true;  // no vocabulary to probe with; not a failure of the graph
    }
    const llama_vocab * vocab = llama_model_get_vocab(&model);
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));

    // Three emitting rows, not all of them. `llama_decode` sizes the context's
    // output buffer to the number of rows that emit and keeps it, so asking
    // every token for its vocabulary row would add n_ubatch * n_vocab floats to
    // this context for the rest of the run -- 300 MiB at a 512-token
    // micro-batch on a 150k vocabulary, to answer a yes/no question. The first
    // row depends on no other, the last depends on all of them (through
    // attention over the earlier keys, or through the recurrent state), and the
    // middle one is there so a defect confined to one half still has a witness.
    const uint32_t emitting[3] = { 0, n_tokens / 2, n_tokens - 1 };

    clear_probe_memory(ctx);
    probe_batch guard(static_cast<int32_t>(n_tokens), 1);
    guard.batch.n_tokens = static_cast<int32_t>(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        guard.batch.token[i] = tokens[i];
        guard.batch.pos[i] = static_cast<llama_pos>(i);
        guard.batch.n_seq_id[i] = 1;
        guard.batch.seq_id[i][0] = 0;
        guard.batch.logits[i] = i == emitting[0] || i == emitting[1] || i == emitting[2];
    }
    const bool decoded = llama_decode(ctx, guard.batch) == 0;
    if (decoded) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t row = emitting[k];
            // A micro-batch of one or two tokens collapses the three rows onto
            // the same one; counting it once keeps the reported total a count
            // of non-finite values and not of visits.
            if ((k > 0 && row == emitting[0]) || (k > 1 && row == emitting[1])) {
                continue;
            }
            const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(row));
            if (!logits) {
                out_nonfinite += n_vocab;
                continue;
            }
            for (size_t j = 0; j < n_vocab; ++j) {
                if (!std::isfinite(logits[j])) {
                    ++out_nonfinite;
                }
            }
        }
    }
    clear_probe_memory(ctx);
    return decoded && out_nonfinite == 0;
}

// Whether a packed multi-sequence forward keeps its sequences apart on this
// device, and whether the packed graph is computing this model at all.
// Two comparisons, because they fail differently and neither subsumes the other:
//   * isolation - the same run decoded alone through the packed path, then
//     beside a second run in one micro-batch. Same nodes, same kernels, same
//     shapes per token; the only difference is the neighbour. A state that
//     leaks across sequences moves these logits and nothing else can, so the
//     tolerance here is tight.
//   * sanity - the packed path against the ordinary equal-sequence decode.
//     These are two different graphs (LFM2 gathers its ShortConv window with
//     GET_ROWS where the equal-sequence path runs SSM_CONV), reduced in a
//     different order over quantized weights, so their logits differ by
//     percents on a healthy model. Comparing them numerically would fail on
//     arithmetic; comparing which token they rank first would not, and that is
//     what catches an indexed path computing something else entirely.
// `ctx` must have at least two sequence slots and a micro-batch wide enough for
// both runs, and the model must already be declared packable: llama_decode_packed
// refuses a model that needs equal-sequence ubatches, and a refused decode
// measures nothing rather than downgrading anything.
bool packed_forward_keeps_sequences_apart(
        llama_context * ctx, const llama_model & model,
        uint32_t n_seq_tokens, float & out_max_delta) {
    out_max_delta = 0.0f;
    const llama_vocab * vocab = llama_model_get_vocab(&model);
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    if (n_seq_tokens < 2 || n_vocab == 0) {
        return true;
    }
    // Two runs that share no token past the first: a packed graph that leaked
    // one sequence's state into the other has to change an output to be caught.
    const std::vector<llama_token> run_a = probe_token_run(model, n_seq_tokens, 37);
    const std::vector<llama_token> run_b = probe_token_run(model, n_seq_tokens, 131);
    if (run_a.size() != n_seq_tokens || run_b.size() != n_seq_tokens) {
        return true;
    }

    // Decodes `tokens` as sequence `seq`, optionally beside `other`, and copies
    // out the logits of `seq`'s last token. Returns false only when nothing was
    // measured, which never downgrades the declaration.
    auto decode_run = [&](const std::vector<llama_token> & tokens,
                          const std::vector<llama_token> * other,
                          bool packed,
                          std::vector<float> & out) -> bool {
        const uint32_t n_tokens = other ? 2 * n_seq_tokens : n_seq_tokens;
        clear_probe_memory(ctx);
        probe_batch guard(static_cast<int32_t>(n_tokens), other ? 2 : 1);
        guard.batch.n_tokens = static_cast<int32_t>(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const bool second = i >= n_seq_tokens;
            const uint32_t within = second ? i - n_seq_tokens : i;
            guard.batch.token[i] = second ? (*other)[within] : tokens[within];
            guard.batch.pos[i] = static_cast<llama_pos>(within);
            guard.batch.n_seq_id[i] = 1;
            guard.batch.seq_id[i][0] = second ? 1 : 0;
            guard.batch.logits[i] = within + 1 == n_seq_tokens;
        }
        const int32_t status = packed
                ? llama_decode_packed(ctx, guard.batch)
                : llama_decode(ctx, guard.batch);
        if (status != 0) {
            clear_probe_memory(ctx);
            return false;
        }
        // llama_get_logits_ith indexes the batch, and only each run's last
        // token was asked to emit.
        const float * logits = llama_get_logits_ith(
                ctx, static_cast<int32_t>(n_seq_tokens - 1));
        if (!logits) {
            clear_probe_memory(ctx);
            return false;
        }
        out.assign(logits, logits + n_vocab);
        clear_probe_memory(ctx);
        return true;
    };

    auto argmax = [&](const std::vector<float> & row) -> size_t {
        size_t best = 0;
        for (size_t j = 1; j < row.size(); ++j) {
            if (row[j] > row[best]) {
                best = j;
            }
        }
        return best;
    };

    for (int run = 0; run < 2; ++run) {
        const std::vector<llama_token> & self = run == 0 ? run_a : run_b;
        const std::vector<llama_token> & neighbour = run == 0 ? run_b : run_a;

        std::vector<float> alone;
        std::vector<float> together;
        std::vector<float> classic;
        if (!decode_run(self, nullptr, true, alone)
                || !decode_run(self, &neighbour, true, together)
                || !decode_run(self, nullptr, false, classic)) {
            return true;  // nothing measured; do not downgrade on a decode error
        }

        float scale = 1.0f;
        for (size_t j = 0; j < n_vocab; ++j) {
            if (!std::isfinite(alone[j]) || !std::isfinite(together[j])) {
                return false;
            }
            scale = std::max(scale, std::fabs(alone[j]));
        }
        for (size_t j = 0; j < n_vocab; ++j) {
            out_max_delta = std::max(out_max_delta, std::fabs(together[j] - alone[j]));
        }
        if (out_max_delta > 1.0e-3f * scale) {
            return false;
        }
        if (argmax(alone) != argmax(classic)) {
            return false;
        }
    }
    return true;
}

// The recurrent-state rollback depth this training context needs, in tokens.
// It used to be a hardwired zero for every model. The need behind it is narrow
// but real: the shared-prefix behavior scorer evicts each scored branch, and on
// a recurrent memory llama_memory_seq_rm only accepts that eviction when it
// drops a whole sequence or rolls back at most n_rs_seq tokens. With two
// sequence slots the scorer drops a whole sequence and no snapshot buys
// anything; with one it rolls back to the end of the prompt, and a refused
// rollback costs a full prompt re-prefill per branch (counted by
// retro_trainer_scoring_stats as prefix_reprefills).
// Three bounds, all of them hard:
//   - the longest rollback the scorer can ask for is n_ctx - 1 tokens;
//   - split_equal asserts n_ubatch > n_rs_seq + 1, because the trailing
//     (1 + n_rs_seq) tokens of a sequence may not be split across micro-batches;
//   - llama_memory_recurrent allocates (1 + n_rs_seq) copies of the whole
//     per-sequence state, so the depth is also whatever a byte budget allows.
uint32_t derive_recurrent_rollback(
        const llama_model & model, const retro_train_config & config,
        std::string & out_reason) {
    if (!llama_model_is_recurrent(&model) && !llama_model_is_hybrid(&model)) {
        out_reason = "0 (no recurrent state to snapshot)";
        return 0;
    }
    if (config.n_seq_max >= 2) {
        out_reason = "0 (branch isolation drops a whole sequence, which needs no snapshot)";
        return 0;
    }
    // Opt-in until the counters above have been read on a real run: this is a
    // lever that changes micro-batch splitting and device memory for every
    // decode of the context, and the repo's rule is that an unmeasured lever
    // ships off. The derivation itself is reported either way.
    const char * mode = std::getenv("RETRO_RECURRENT_ROLLBACK");
    const bool enabled = mode && std::strcmp(mode, "auto") == 0;

    const auto & hparams = model.hparams;
    const uint64_t state_elements = static_cast<uint64_t>(hparams.n_embd_r() + hparams.n_embd_s())
            * hparams.n_layer();
    const uint64_t per_level_bytes = state_elements
            * std::max<uint32_t>(config.n_seq_max, 1) * sizeof(float);

    uint64_t budget_bytes = 64ull * 1024 * 1024;
    if (const char * budget = std::getenv("RETRO_RECURRENT_ROLLBACK_BUDGET_MB")) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(budget, &end, 10);
        if (end && *end == '\0') {
            budget_bytes = static_cast<uint64_t>(parsed) * 1024 * 1024;
        }
    }

    const uint64_t window = config.n_ctx > 1 ? config.n_ctx - 1 : 0;
    const uint64_t split_bound = config.n_ubatch > 2 ? config.n_ubatch - 2 : 0;
    const uint64_t budget_bound = per_level_bytes > 0 ? budget_bytes / per_level_bytes : window;
    const uint64_t derived = std::min({ window, split_bound, budget_bound });

    std::ostringstream reason;
    reason << derived << " (scorer window " << window
           << ", micro-batch bound " << split_bound
           << ", budget bound " << budget_bound
           << "; " << (enabled ? "applied" : "not applied: set RETRO_RECURRENT_ROLLBACK=auto")
           << ")";
    out_reason = reason.str();
    return enabled ? static_cast<uint32_t>(derived) : 0;
}

} // namespace

bool load_model_and_context(trainer_state & state) {
    ensure_backend_initialized();

    const int32_t requested = state.train_config.device;
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
    // The registry behind the device, which per-backend claims key on. A GPU
    // device with no registry name keeps the device name.
    ggml_backend_reg_t active_reg = use_gpu && gpu_device
            ? ggml_backend_dev_backend_reg(gpu_device)
            : nullptr;
    const char * registry_name = active_reg ? ggml_backend_reg_name(active_reg) : nullptr;
    state.backend_registry = use_gpu
            ? (registry_name && registry_name[0] ? registry_name : state.backend_name)
            : "CPU";

    llama_model_params model_params = llama_model_default_params();
    // Upstream replaced the use_mmap/use_mlock booleans with llama_load_mode.
    // Same intent as before: mmap the weights when they stay on the host, and
    // skip it for a GPU load where every tensor is copied into device memory
    // anyway and the mapping only costs page cache.
    //
    // A base-weight policy takes the same path for a different reason: the
    // mapping is PROT_READ (llama-mmap.cpp), so the first optimizer step would
    // fault rather than drift. LLAMA_LOAD_MODE_NONE reads the file into owned
    // writable buffers, which is also what keeps the source GGUF's bytes
    // untouched. It costs the host the weights it used to share with the page
    // cache, which is what memory_totals() has to report.
    const bool writable_weights = trains_base_weights(state);
    model_params.load_mode = (use_gpu || writable_weights)
            ? LLAMA_LOAD_MODE_NONE
            : LLAMA_LOAD_MODE_MMAP;
    state.weight_storage = model_params.load_mode == LLAMA_LOAD_MODE_MMAP ? "mapped" : "owned";
    model_params.use_extra_bufts = false;
    model_params.n_gpu_layers = state.n_gpu_layers;

    // n_gpu_layers=0 keeps the weights on the host, but by itself it still
    // leaves the GPU in the model's device list, so llama_context builds a
    // scheduler whose backend 0 is the GPU. ggml_opt allocates the AdamW
    // moments on backend 0 while the parameters and the (GPU-unsupported)
    // OPT_STEP_ADAMW nodes stay on the CPU, so every LoRA parameter feeds two
    // cross-backend copies into a single CPU split and the split overruns
    // GGML_SCHED_MAX_SPLIT_INPUTS. An explicit empty device list keeps the
    // scheduler CPU-only, which is what device=cpu asked for anyway.
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (!use_gpu) {
        model_params.devices = no_devices;
    }

    model_ptr model(llama_model_load_from_file(state.model_path.c_str(), model_params));
    if (!model) {
        set_error("failed to load model GGUF: " + state.model_path);
        return false;
    }

    // The requested micro-batch is used as asked. Whether this device evaluates
    // it to finite values is measured below, once the context exists, and the
    // escalation to a full logical batch follows that measurement rather than
    // an architecture name (see decode_logits_are_finite).
    uint32_t effective_ubatch = state.train_config.n_ubatch;

    llama_context_params ctx_params = llama_context_default_params();
    uint32_t n_threads = 0;
    if (!resolve_thread_count(state.train_config, n_threads)
            || n_threads > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        if (g_last_error.empty()) {
            set_error("thread count exceeds the llama.cpp limit");
        }
        return false;
    }
    ctx_params.n_ctx = state.train_config.n_ctx;
    ctx_params.n_batch = state.train_config.n_batch;
    ctx_params.n_ubatch = effective_ubatch;
    // GRPO's optimizer can build a single teacher-forced graph with a prompt
    // shared by several completion sequences. Ordinary SFT/PPO still use
    // sequence zero only.
    ctx_params.n_seq_max = state.train_config.n_seq_max;
    // Shared-prefix training stores one prompt cell set carrying several
    // sequence ids. A partitioned KV cache would divide `n_ctx` by the group
    // size and reject an otherwise valid full-width training row.
    ctx_params.kv_unified = true;
    // Recurrent-state rollback depth, derived from what the shared-prefix
    // scorer can ask of this geometry and from what the snapshots cost, rather
    // than pinned at zero for every model (see derive_recurrent_rollback).
    // llama_context clamps it again for an architecture whose graph cannot
    // produce the snapshots, so the value that ends up applied is read back
    // from the context below.
    ctx_params.n_rs_seq = derive_recurrent_rollback(
            *model, state.train_config, state.recurrent_rollback_status);
    ctx_params.n_threads = static_cast<int32_t>(n_threads);
    ctx_params.n_threads_batch = static_cast<int32_t>(n_threads);
    state.effective_threads = n_threads;
    // Differentiable Flash Attention is enabled per device capability, not per
    // backend name: the active device must report both the forward and backward
    // flash-attention ops for this model's head geometry. AUTO retains the
    // materialized graph as a safe fallback for ordinary F32 training on
    // devices/shapes not covered by that kernel.
    const uint32_t fa_probe_tokens = std::min(effective_ubatch, state.train_config.n_ctx);
    const bool differentiable_flash_attn = use_gpu
            && std::getenv("RETRO_DISABLE_DIFF_FLASH_ATTN") == nullptr
            && supports_flash_attn_back(gpu_device, *model, fa_probe_tokens, GGML_TYPE_F32);
    state.cap_flash_attn_back = differentiable_flash_attn;
    state.cap_device_sampling = use_gpu && supports_device_sampling(gpu_device);
    state.cap_device_logprobs = use_gpu && supports_device_logprob_gather(gpu_device);
    // Probed for every optimizer, not only the one this run asked for: a mixed
    // assignment can hand a parameter to either, and the report describes the
    // device rather than the run.
    ggml_backend_dev_t step_device = use_gpu ? gpu_device : nullptr;
    state.cap_opt_step_f16[RETRO_OPTIMIZER_ADAMW] =
            supports_opt_step_dtype(step_device, RETRO_OPTIMIZER_ADAMW, GGML_TYPE_F16);
    state.cap_opt_step_f16[RETRO_OPTIMIZER_SGD] =
            supports_opt_step_dtype(step_device, RETRO_OPTIMIZER_SGD, GGML_TYPE_F16);
    state.cap_fused_sparse_ce = use_gpu
            && supports_fused_sparse_ce(gpu_device, *model, state.train_config, fa_probe_tokens);
    ctx_params.flash_attn_type = differentiable_flash_attn
            ? LLAMA_FLASH_ATTN_TYPE_AUTO
            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Without this the attention reads the KV cache buffer directly, aliasing
    // the store that filled it rather than depending on it, so no gradient ever
    // reaches attn_k/attn_v and a LoRA targeting them trains as a silent no-op.
    // RETRO_DISABLE_KV_DIFF is a diagnostic escape hatch (K/V LoRA becomes a
    // no-op) used to isolate differentiable-store issues from the rest of the
    // training graph.
    ctx_params.kv_differentiable = std::getenv("RETRO_DISABLE_KV_DIFF") == nullptr;
    const bool requested_f16_kv = state.train_config.kv_dtype == RETRO_KV_DTYPE_F16;
    const bool candidate_f16_kv = requested_f16_kv && differentiable_flash_attn
            && supports_flash_attn_back(gpu_device, *model, fa_probe_tokens, GGML_TYPE_F16);
    ctx_params.type_k = candidate_f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ctx_params.type_v = candidate_f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;
    // The fused Gated Delta Net kernel (Qwen3-Next / Qwen3.5) now has a real
    // analytic backward (ggml_gated_delta_net_back), so training keeps the
    // fast, rollback-capable fused path like generation does. no_fused_gdn
    // stays available as an escape hatch (and exercises the differentiable
    // chunking graph in delta-net-base.cpp) but training no longer needs it.
    ctx_params.no_fused_gdn = std::getenv("RETRO_GDN_UNFUSED") != nullptr;
    ctx_params.no_perf = true;
    ctx_params.offload_kqv = use_gpu;
    ctx_params.op_offload = use_gpu;
    if (std::getenv("RETRO_CUDA_NAN_SCAN") != nullptr) {
        ctx_params.cb_eval = nan_scan_eval_callback;
        ctx_params.cb_eval_user_data = nullptr;
    }

    context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    if (!ctx) {
        set_error("failed to create llama context for model: " + state.model_path);
        return false;
    }

    state.effective_kv_dtype = RETRO_KV_DTYPE_F32;
    state.training_kv_status = requested_f16_kv
            ? "fallback_f32: differentiable Flash Attention unavailable on this device"
            : "not_requested";
    if (candidate_f16_kv) {
        // AUTO can still reject Flash Attention for a model architecture or
        // head shape. Do not leave such a graph on F16 cache storage: rebuild
        // the optimizer context with the stable F32 cache and report the
        // fallback explicitly.
        if (ctx->get_cparams().flash_attn) {
            state.effective_kv_dtype = RETRO_KV_DTYPE_F16;
            state.training_kv_status = "supported";
        } else {
            ctx.reset();
            ctx_params.type_k = GGML_TYPE_F32;
            ctx_params.type_v = GGML_TYPE_F32;
            ctx.reset(llama_init_from_model(model.get(), ctx_params));
            if (!ctx) {
                set_error("failed to rebuild the llama context with F32 training KV fallback: "
                        + state.model_path);
                return false;
            }
            state.training_kv_status = "fallback_f32: Flash Attention rejected the model shape";
        }
    }


    state.effective_rs_seq = llama_n_rs_seq(ctx.get());
    state.effective_ubatch = effective_ubatch;

    // Finiteness of the requested micro-batch on this device, measured.
    // A backend that evaluates a short micro-batch of this model's graph to NaN
    // makes every step of the run meaningless, and it is silent: the loss is
    // NaN, nothing errors. The check costs one forward pass over the tokens the
    // first real step would have used anyway. RETRO_UBATCH_FINITE_CHECK=0 skips
    // it for a run that would rather find out later.
    const char * finite_check = std::getenv("RETRO_UBATCH_FINITE_CHECK");
    const bool finite_check_disabled = finite_check
            && (std::strcmp(finite_check, "0") == 0 || std::strcmp(finite_check, "false") == 0);
    if (!use_gpu) {
        state.ubatch_finite_status = "not_probed: cpu has no driver to disagree with";
    } else if (finite_check_disabled) {
        state.ubatch_finite_status = "skipped: RETRO_UBATCH_FINITE_CHECK=0";
    } else {
        const uint32_t probe_tokens = std::min(effective_ubatch, state.train_config.n_ctx);
        uint64_t nonfinite = 0;
        if (decode_logits_are_finite(ctx.get(), *model, probe_tokens, nonfinite)) {
            state.ubatch_finite_status =
                    "pass at " + std::to_string(probe_tokens) + " tokens";
        } else {
            // The escalation the old Falcon-H1/MoltenVK rule performed by name:
            // a full logical batch is a different graph shape, and it is the
            // only larger one this configuration allows. Both the trigger and
            // the width now come from the measurement.
            const uint32_t escalated = std::min(
                    state.train_config.n_batch, state.train_config.n_ctx);
            uint64_t escalated_nonfinite = 0;
            bool recovered = false;
            if (escalated > probe_tokens) {
                ctx.reset();
                ctx_params.n_ubatch = escalated;
                ctx.reset(llama_init_from_model(model.get(), ctx_params));
                if (!ctx) {
                    set_error("failed to rebuild the llama context at the escalated micro-batch of "
                            + std::to_string(escalated) + " tokens: " + state.model_path);
                    return false;
                }
                recovered = decode_logits_are_finite(
                        ctx.get(), *model, escalated, escalated_nonfinite);
                if (recovered) {
                    effective_ubatch = escalated;
                    state.effective_ubatch = escalated;
                    state.effective_rs_seq = llama_n_rs_seq(ctx.get());
                    state.ubatch_finite_status = "escalated: " + std::to_string(probe_tokens)
                            + " -> " + std::to_string(escalated) + " tokens ("
                            + std::to_string(nonfinite) + " non-finite logits at "
                            + std::to_string(probe_tokens) + ")";
                }
            }
            if (!recovered) {
                std::ostringstream message;
                message << "the training graph of " << model->arch_name()
                        << " produces non-finite logits on " << state.backend_name
                        << " at a micro-batch of " << probe_tokens << " tokens ("
                        << nonfinite << " non-finite values)";
                if (escalated > probe_tokens) {
                    message << " and at " << escalated << " tokens ("
                            << escalated_nonfinite << " non-finite values), the widest"
                               " micro-batch this configuration allows";
                } else {
                    message << ", and the optimizer window (training.micro_batch *"
                               " training.gradient_accumulation = " << state.train_config.n_batch
                            << ") leaves no wider micro-batch to escalate to";
                }
                message << "; raise the optimizer window, or select device=cpu."
                           " Set RETRO_UBATCH_FINITE_CHECK=0 to train anyway";
                set_error(message.str());
                return false;
            }
        }
    }

    // Packed multi-sequence training: the declaration, then the verification.
    // llama_model_supports_packed_seq() answers for the graph llama.cpp builds
    // and is maintained beside the arch table the packed entry points honour.
    // It cannot answer for the driver that evaluates that graph, so on a model
    // carrying recurrent state the declaration is checked numerically against
    // the isolated forwards it claims to reproduce, and may only be downgraded.
    const bool declared_packed = llama_model_supports_packed_seq(model.get());
    const bool carries_state = llama_model_is_recurrent(model.get())
            || llama_model_is_hybrid(model.get());
    const char * packed_probe = std::getenv("RETRO_PACKED_SEQ_PROBE");
    const bool packed_probe_disabled = packed_probe
            && (std::strcmp(packed_probe, "0") == 0 || std::strcmp(packed_probe, "false") == 0);
    if (!declared_packed) {
        state.cap_packed_seq_training = false;
        state.packed_seq_status = std::string("unsupported: the ")
                + model->arch_name()
                + " graph reads its recurrent state by physical adjacency";
    } else if (!carries_state) {
        state.cap_packed_seq_training = true;
        state.packed_seq_status = "supported: attention-only graph, history addressed by sequence id";
    } else if (packed_probe_disabled) {
        state.cap_packed_seq_training = true;
        state.packed_seq_status = "declared: RETRO_PACKED_SEQ_PROBE=0, not verified on this device";
    } else {
        // A context of its own: the optimizer context may carry a single
        // sequence slot and a micro-batch too narrow to hold both runs, and a
        // probe has no business resizing the geometry the caller asked for.
        const uint32_t n_seq_tokens = 4;
        llama_context_params probe_params = ctx_params;
        probe_params.n_ctx = 4 * n_seq_tokens;
        probe_params.n_batch = 2 * n_seq_tokens;
        probe_params.n_ubatch = 2 * n_seq_tokens;
        probe_params.n_seq_max = 2;
        probe_params.n_rs_seq = 0;
        context_ptr probe_ctx(llama_init_from_model(model.get(), probe_params));
        float max_delta = 0.0f;
        if (!probe_ctx) {
            state.cap_packed_seq_training = false;
            state.packed_seq_status = "unsupported: the packed-equivalence probe context could not be created";
        } else if (packed_forward_keeps_sequences_apart(
                probe_ctx.get(), *model, n_seq_tokens, max_delta)) {
            state.cap_packed_seq_training = true;
            std::ostringstream status;
            status << "verified on " << state.backend_name
                   << ": a neighbouring sequence moves this one's logits by at"
                      " most " << max_delta;
            state.packed_seq_status = status.str();
        } else {
            state.cap_packed_seq_training = false;
            std::ostringstream status;
            status << "unsupported: " << model->arch_name() << " declares packed"
                      " multi-sequence support, but on " << state.backend_name
                   << " a packed micro-batch does not keep its sequences apart"
                      " (a neighbour moved this sequence's logits by " << max_delta << ")";
            state.packed_seq_status = status.str();
        }
    }

    const bool fast_generation = state.train_config.fast_generation_context;
    context_ptr generation_ctx;
    // An explicit generation_batch is itself a request for the dedicated
    // forward-only context. Without this, PPO/SFT callers using the default
    // single-sequence geometry would accept the setting but silently keep the
    // optimizer context's batch geometry.
    if (state.train_config.generation_concurrency > 1 || fast_generation
            || state.train_config.generation_batch != 0) {
        llama_context_params generation_params = ctx_params;
        // Sampling never differentiates, so it keeps the cheaper transposed V
        // cache and the plain aliasing view of it.
        generation_params.kv_differentiable = false;
        // Generation never differentiates: keep the fast fused GDN kernel.
        generation_params.no_fused_gdn = false;
        // Recurrent-state rollback is only needed by the training context's
        // shared-prefix GRPO scorer (score_token_suffix_batch_impl), which
        // never runs against this context. Rollback snapshots widen the
        // recurrent-state cache by a (1 + n_rs_seq) factor per sequence, and
        // this context already multiplies n_ctx/n_seq_max by
        // generation_concurrency, so inheriting it here would multiply that
        // cost again for no benefit.
        generation_params.n_rs_seq = 0;
        // Sampling never rolls a sequence back, so a window-sized SWA cache is
        // sufficient here and avoids allocating the full history per stream.
        generation_params.swa_full = false;
        generation_params.n_ctx =
                state.train_config.n_ctx * state.train_config.generation_concurrency;
        generation_params.n_seq_max = state.train_config.generation_concurrency;
        // This forward-only context uses independent batch geometry; inheriting
        // the optimizer's small activation budget would slow prompt prefill.
        {
            uint32_t generation_batch = state.train_config.generation_batch;
            if (generation_batch == 0) {
                // Bound the default output buffer to a fixed budget; callers can
                // raise it explicitly for workloads with many short prompts.
                const size_t n_vocab = static_cast<size_t>(
                        llama_vocab_n_tokens(llama_model_get_vocab(model.get())));
                const size_t logits_budget = size_t{64} << 20;
                const uint32_t affordable = static_cast<uint32_t>(std::max<size_t>(
                        1, logits_budget / std::max<size_t>(1, n_vocab * sizeof(float))));
                generation_batch = std::min(
                        std::min<uint32_t>(state.train_config.n_ctx, 512), affordable);
            }
            // A decode wave submits one row per concurrent sequence and has to
            // fit a single launch; retro_rollout.cpp rejects the call otherwise.
            generation_batch = std::max(generation_batch,
                    state.train_config.generation_concurrency);
            generation_params.n_batch = generation_batch;
            generation_params.n_ubatch = generation_batch;
        }
        if (fast_generation) {
            // Forward-only context: an F16 KV cache halves the cache footprint
            // and flash-attention removes the materialized attention matrix.
            // The optimizer may independently opt into differentiable F16 KV
            // on devices/shapes that report Flash Attention backward. Sampling
            // can always use this smaller cache because it has no backward graph.
            generation_params.type_k = GGML_TYPE_F16;
            generation_params.type_v = GGML_TYPE_F16;
            generation_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        generation_ctx.reset(llama_init_from_model(model.get(), generation_params));
        if (!generation_ctx) {
            set_error("failed to create the dedicated generation context for model: "
                    + state.model_path);
            return false;
        }
    }

    state.model = std::move(model);
    state.ctx = std::move(ctx);
    state.generation_ctx = std::move(generation_ctx);
    return true;
}

// Whether the AdamW step for an F16 parameter actually runs on the active
// device. The optimizer op is what decides it: a backend that declines it sends
// every step back to the CPU, with a cross-backend copy of each parameter on
// every iteration. Ask the device rather than guess from its name -- Vulkan
// declines F16 unless the device reports fp16 support, and a name match would
// report "supported" on a device that silently falls back.
const char * optimizer_f16_status(const trainer_state & state) {
    if (state.lora_dtype != RETRO_LORA_DTYPE_F16) {
        return "not_requested";
    }
    if (!state.gpu_active) {
        return "supported";  // the CPU kernel handles F16 parameters
    }
    ggml_backend_dev_t dev = first_gpu_device();
    if (!dev) {
        return "cpu_fallback";
    }
    ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) {
        return "cpu_fallback";
    }
    ggml_tensor * w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 32);
    ggml_tensor * g = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_tensor * m = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_tensor * v = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_tensor * p = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 9);
    if (!w || !g || !m || !v || !p) {
        return "cpu_fallback";
    }
    ggml_set_param(w);
    ggml_tensor * step = ggml_opt_step_adamw(ctx.get(), w, g, m, v, p);
    return step && ggml_backend_dev_supports_op(dev, step) ? "supported" : "cpu_fallback";
}

retro_memory_report memory_totals(const trainer_state & state) {
    retro_memory_report report {};

    // The resolved trainable set: the adapter's factors, allocated next to the
    // model, plus the base tensors, which are already inside its weights. Both
    // carry a gradient and optimizer state; only the first adds parameter bytes.
    size_t trainable_elements = static_cast<size_t>(count_lora_parameters(state));
    const size_t adapter_elements = trainable_elements;
    auto optimizer_bytes = [&](const ggml_tensor * tensor) -> uint64_t {
        if (!tensor) {
            return 0;
        }
        const auto owner = opt_param_optimizer(tensor, const_cast<trainer_state *>(&state));
        int64_t count = 0;
        const auto * slots = ggml_opt_optimizer_slots(owner, &count);
        uint64_t bytes = 0;
        for (int64_t i = 0; i < count; ++i) {
            bytes += static_cast<uint64_t>(ggml_opt_slot_n_elements(&slots[i], ggml_nelements(tensor)))
                    * ggml_type_size(slots[i].type);
        }
        return bytes;
    };
    uint64_t adapter_state_bytes = 0;
    if (state.adapter) {
        for (const auto & item : state.adapter->ab_map) {
            adapter_state_bytes += optimizer_bytes(item.second.a) + optimizer_bytes(item.second.b);
        }
    }
    report.optimizer_state_bytes = adapter_state_bytes;
    uint64_t base_host_training_bytes = 0;
    uint64_t base_device_training_bytes = 0;
    report.trainable_parameter_bytes = count_lora_parameter_bytes(state);
    const uint64_t adapter_parameter_bytes = report.trainable_parameter_bytes;
    const ggml_tensor * base_sample = nullptr;
    for (const std::string & name : state.trainable_base) {
        const auto found = std::find_if(
                state.model->tensors_by_name.begin(),
                state.model->tensors_by_name.end(),
                [&](const std::pair<std::string, ggml_tensor *> & item) {
                    return item.first == name;
                });
        if (found == state.model->tensors_by_name.end() || !found->second) {
            continue;
        }
        trainable_elements += static_cast<size_t>(ggml_nelements(found->second));
        report.trainable_parameter_bytes += ggml_nbytes(found->second);
        const bool on_host = found->second->buffer && ggml_backend_buft_is_host(
                ggml_backend_buffer_get_type(found->second->buffer));
        const uint64_t state_bytes = optimizer_bytes(found->second);
        report.optimizer_state_bytes += state_bytes;
        (on_host ? base_host_training_bytes : base_device_training_bytes) +=
                static_cast<uint64_t>(ggml_nelements(found->second)) * sizeof(float) + state_bytes;
        if (!base_sample) {
            base_sample = found->second;
        }
    }
    report.trainable_gradient_bytes  = trainable_elements * sizeof(float);
    // Whether `trainable_parameter_bytes` may be added on top of the model's
    // weights or is a slice of them. Neither answer is right for a hybrid set,
    // so the rollup below adds the adapter half explicitly instead of reading
    // this flag for both.
    report.trainable_parameters_are_model_subset =
            !state.trainable_base.empty() && adapter_parameter_bytes == 0;

    // The model weights are shared between the optimizer and the generation
    // context, so they are summed once (from the optimizer context) and never
    // double-counted; each context only adds its own KV cache and compute
    // buffer on top.
    const auto optimizer_mem = context_memory_by_buffer(state.ctx.get());
    const auto generation_mem = context_memory_by_buffer(state.generation_ctx.get());
    for (const auto & item : optimizer_mem) {
        report.model_weight_bytes += item.second.model;
        // The breakdown's "context" bucket is the memory object of the
        // context, i.e. the KV cache buffers - the only term kv_dtype resizes.
        report.optimizer_kv_bytes += item.second.kv;
        report.optimizer_compute_bytes += item.second.compute;
    }
    report.has_generation_context = state.generation_ctx != nullptr;
    for (const auto & item : generation_mem) {
        report.generation_kv_bytes += item.second.kv;
        report.generation_compute_bytes += item.second.compute;
    }

    // Where the trainable parameters (and therefore their F32 gradient and the
    // optimizer state ggml_opt allocates alongside) physically live. Base
    // placement follows the selected base tensors once there are any; until
    // then it tracks the adapter so the rollup below has one answer.
    if (state.adapter && !state.adapter->ab_map.empty()) {
        const ggml_tensor * a = state.adapter->ab_map.begin()->second.a;
        if (a && a->buffer) {
            report.adapter_on_host =
                    ggml_backend_buft_is_host(ggml_backend_buffer_get_type(a->buffer));
        }
    }
    report.base_trainable_on_host = base_sample && base_sample->buffer
            ? ggml_backend_buft_is_host(ggml_backend_buffer_get_type(base_sample->buffer))
            : report.adapter_on_host;
    // Only what is *not* already inside model_weight_bytes. See the field's
    // comment in retro_lora_train.h: base tensors are a slice of the weights
    // and must never be added to them, adapter factors sit on top.
    const uint64_t adapter_total_bytes = adapter_parameter_bytes
            + adapter_elements * sizeof(float) + adapter_state_bytes;

    // Roll the memory-breakdown-visible allocations up into a device vs host
    // split so the report says which budget each part draws on. The optimizer
    // state (params + grad + state) is bucketed by the trainable parameters'
    // device.
    for (const auto & item : optimizer_mem) {
        const uint64_t total = item.second.model + item.second.kv + item.second.compute;
        (item.second.is_host ? report.host_bytes : report.device_bytes) += total;
    }
    for (const auto & item : generation_mem) {
        const uint64_t total = item.second.kv + item.second.compute;
        (item.second.is_host ? report.host_bytes : report.device_bytes) += total;
    }
    // Each base tensor contributes where it lives; an absent adapter's default
    // placement must not send CPU base gradients into the device budget.
    report.host_bytes += base_host_training_bytes;
    report.device_bytes += base_device_training_bytes;
    (report.adapter_on_host ? report.host_bytes : report.device_bytes) += adapter_total_bytes;

    // Measured device memory is separate from the buffer totals: it also includes
    // backend scratch and the graph allocator's transient reserve. Report it only
    // after the runtime has sampled it, so zero is not mistaken for a measurement.
    llama_opt_memory measured {};
    llama_opt_get_memory(state.ctx.get(), &measured);
    report.device_memory_samples = measured.n_samples;
    if (measured.n_samples > 0) {
        report.device_total_bytes = measured.device_total_bytes;
        report.device_used_bytes = measured.device_used_bytes;
        report.device_peak_used_bytes = measured.device_peak_used_bytes;
        report.backend_scratch_bytes = measured.scratch_bytes;
        report.backend_scratch_peak_bytes = measured.scratch_peak_bytes;
    }

    // Read checkpoint lifetimes from the built backward graph; the checkpoint
    // list contains shapes but not the lifetime data needed by this report.
    ggml_opt_checkpoint_profile checkpoints {};
    if (llama_opt_get_checkpoint_profile(state.ctx.get(), &checkpoints)) {
        report.checkpoint_count             = (uint64_t) checkpoints.n_checkpoints;
        report.checkpoint_retained_bytes    = checkpoints.retained_bytes;
        report.checkpoint_live_peak_bytes   = checkpoints.live_peak_bytes;
        report.checkpoint_live_peak_count   = (uint64_t) checkpoints.live_peak_count;
        report.checkpoint_long_lived_bytes  = checkpoints.long_lived_bytes;
        report.checkpoint_long_lived_count  = (uint64_t) checkpoints.n_long_lived;
        report.checkpoint_graph_nodes       = (uint64_t) checkpoints.n_nodes;
        report.checkpoint_max_span_nodes    = (uint64_t) checkpoints.max_span_nodes;
        report.checkpoint_total_span_nodes  = (uint64_t) checkpoints.total_span_nodes;
    }
    return report;
}

namespace {

void copy_name_out(const std::string & value, char (&destination)[RETRO_MODEL_INFO_NAME_MAX]) {
    const size_t length = std::min(value.size(), sizeof(destination) - 1);
    std::memcpy(destination, value.data(), length);
    destination[length] = '\0';
}

uint64_t file_size_bytes(const char * path) {
    std::FILE * file = std::fopen(path, "rb");
    if (!file) {
        return 0;
    }
    uint64_t size = 0;
    if (std::fseek(file, 0, SEEK_END) == 0) {
        const long position = std::ftell(file);
        if (position >= 0) {
            size = static_cast<uint64_t>(position);
        }
    }
    std::fclose(file);
    return size;
}

} // namespace

int read_model_info_impl(const char * model_path, int32_t device, retro_model_info * out_info) {
    return boundary([&]() -> int {
        if (is_blank(model_path)) {
            set_error("model_path is required");
            return -1;
        }
        if (!out_info) {
            set_error("out_info is required");
            return -1;
        }
        ensure_backend_initialized();
        // The device request is validated but never honoured with an offload:
        // reading geometry must not allocate device memory, which is the whole
        // point of asking before a run is planned. Validating it anyway means a
        // caller planning a GPU run learns here that there is no GPU, rather
        // than at trainer creation.
        if (device == RETRO_DEVICE_GPU && !first_gpu_device()) {
            set_error("device=gpu requested but no GPU backend is available");
            return -1;
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.use_extra_bufts = false;
        model_params.n_gpu_layers = 0;
        // Same reason as load_model_and_context's CPU path: an empty device list
        // keeps the load host-only instead of leaving a GPU in the model's
        // device list.
        ggml_backend_dev_t no_devices[] = { nullptr };
        model_params.devices = no_devices;

        model_ptr model(llama_model_load_from_file(model_path, model_params));
        if (!model) {
            set_error("failed to load model GGUF: " + std::string(model_path));
            return -1;
        }

        const llama_hparams & hparams = model->hparams;
        retro_model_info info {};
        info.n_layer       = hparams.n_layer();
        info.n_embd        = hparams.n_embd;
        for (uint32_t il = 0; il < info.n_layer; ++il) {
            info.n_ff = std::max(info.n_ff, hparams.n_ff(il));
        }
        info.n_head        = hparams.n_head();
        info.n_head_kv     = hparams.n_head_kv();
        info.n_embd_head_k = hparams.n_embd_head_k();
        info.n_embd_head_v = hparams.n_embd_head_v();
        info.n_embd_k_gqa  = hparams.n_embd_k_gqa_max();
        info.n_embd_v_gqa  = hparams.n_embd_v_gqa_max();
        info.n_embd_r      = hparams.n_embd_r();
        info.n_embd_s      = hparams.n_embd_s();
        info.n_ctx_train   = hparams.n_ctx_train;
        info.n_expert      = hparams.n_expert;
        info.n_expert_used = hparams.n_expert_used_max();

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
        info.n_vocab = n_vocab > 0 ? static_cast<uint32_t>(n_vocab) : 0;

        info.n_params         = llama_model_n_params(model.get());
        info.model_size_bytes = llama_model_size(model.get());
        info.file_size_bytes  = file_size_bytes(model_path);
        info.is_recurrent     = llama_model_is_recurrent(model.get());
        info.has_encoder      = llama_model_has_encoder(model.get());
        // A tied head reuses the embedding tensor, so the vocabulary projection
        // costs no weights of its own - a term worth several hundred MiB on a
        // large vocabulary, and one an estimator must not add twice.
        info.tied_embeddings  = model->output == nullptr || model->output == model->tok_embd;

        // Same rule as the backend report's `model_weight_dtype`: the type that
        // accounts for the most bytes, not the most tensors. A quantized model
        // mixes its dominant type with a handful of F32 norms too small to
        // matter, and the dequantization scratch depends on the dominant one.
        std::map<std::string, uint64_t> type_bytes;
        for (const auto & item : model->tensors_by_name) {
            if (item.second) {
                type_bytes[ggml_type_name(item.second->type)] += ggml_nbytes(item.second);
            }
        }
        std::string dominant_type = "unknown";
        uint64_t dominant_bytes = 0;
        for (const auto & item : type_bytes) {
            if (item.second > dominant_bytes) {
                dominant_bytes = item.second;
                dominant_type = item.first;
            }
        }
        info.dominant_weight_bytes = dominant_bytes;
        copy_name_out(model->arch_name(), info.architecture);
        copy_name_out(dominant_type, info.dominant_weight_type);

        *out_info = info;
        return 0;
    });
}

std::string backend_report(const trainer_state & state) {
    std::ostringstream out;
    out << "backend report\n";
    out << "  requested_device: " << device_kind_name(state.requested_device) << "\n";
    out << "  gpu_active: " << (state.gpu_active ? "true" : "false") << "\n";
    out << "  backend: " << state.backend_name << "\n";
    out << "  backend_registry: " << state.backend_registry << "\n";
    out << "  weight_storage: " << state.weight_storage << "\n";
    out << "  trainable_policy: " << trainable_policy_name(state.train_config.trainable) << "\n";
    out << "  trainable_base_tensors: " << state.trainable_base.size() << "\n";
    out << "  optimizer: "
        << (state.train_config.optimizer == RETRO_OPTIMIZER_SGD ? "sgd" : "adamw") << "\n";
    if (state.gpu_active) {
        ggml_backend_dev_t dev = first_gpu_device();
        const char * desc = dev ? ggml_backend_dev_description(dev) : nullptr;
        out << "  gpu_device: " << (desc && desc[0] ? desc : "unknown") << "\n";
    }
    // Training capabilities are resolved from ggml supports_op probes, never from
    // the backend name; a false value means the corresponding path falls back.
    out << "  cap_flash_attn_back: "
        << (state.cap_flash_attn_back ? "supported" : "unavailable") << "\n";
    out << "  cap_device_sampling: "
        << (state.cap_device_sampling ? "supported" : "unavailable") << "\n";
    out << "  cap_device_logprobs: "
        << (state.cap_device_logprobs ? "supported" : "unavailable") << "\n";
    // Reported whether or not chunked_cross_entropy is on: "unavailable" is a
    // warning about what enabling it would cost, not a description of this run.
    out << "  cap_fused_sparse_ce: "
        << (state.cap_fused_sparse_ce ? "supported" : "unavailable") << "\n";
    // Which optimizers this device can run an F16 update step for; empty means
    // F32 base weights only.
    out << "  cap_opt_step_f16: ";
    {
        std::vector<std::string> writers;
        if (state.cap_opt_step_f16[RETRO_OPTIMIZER_ADAMW]) {
            writers.emplace_back("adamw");
        }
        if (state.cap_opt_step_f16[RETRO_OPTIMIZER_SGD]) {
            writers.emplace_back("sgd");
        }
        out << (writers.empty() ? std::string("none") : join_patterns(writers)) << "\n";
    }
    out << "  chunked_cross_entropy: "
        << (state.train_config.chunked_cross_entropy ? "enabled" : "disabled") << "\n";
    // The loss graph this run builds, on its own line because a reader
    // comparing budget to run must see which of the two graphs was priced.
    out << "  loss_path: " << (fused_loss_enabled(state) ? "fused" : "dense") << "\n";
    if (state.train_config.chunked_cross_entropy && trains_loss_head(state)) {
        out << "  loss_path_status: dense_fallback (this run trains the projection head, "
               "and the fused cross-entropy differentiates only its hidden-state input; "
               "the dense path produces the head's gradient, at the cost of materializing "
               "the whole vocabulary)\n";
    } else if (state.train_config.chunked_cross_entropy && state.gpu_active
            && !state.cap_fused_sparse_ce) {
        out << "  chunked_cross_entropy_status: cpu_fallback (the active device has no "
               "FUSED_SPARSE_CE kernel; the hidden states and the whole projection head "
               "cross the bus per token chunk)\n";
    }
    // The effective value, not the config field: RETRO_REQUIRE_GPU_RESIDENT can
    // turn it on for a whole lane, and a report that said "false" while the
    // preflight was enforcing it would be describing a different run.
    out << "  require_gpu_resident: " << (require_gpu_resident(state) ? "true" : "false") << "\n";
    out << "  effective_batch: " << llama_n_batch(state.ctx.get()) << "\n";
    out << "  effective_ubatch: " << llama_n_ubatch(state.ctx.get()) << "\n";
    out << "  optimizer_max_sequences: " << llama_n_seq_max(state.ctx.get()) << "\n";
    out << "  generation_concurrency: " << (state.generation_ctx
            ? llama_n_seq_max(state.generation_ctx.get()) : 1) << "\n";
    // Reported separately from effective_batch: the generation context no
    // longer inherits the optimizer's geometry (docs/engineering/optims/SAMPLING.md, S3).
    out << "  generation_batch: " << (state.generation_ctx
            ? llama_n_batch(state.generation_ctx.get())
            : llama_n_batch(state.ctx.get())) << "\n";
    out << "  generation_ubatch: " << (state.generation_ctx
            ? llama_n_ubatch(state.generation_ctx.get())
            : llama_n_ubatch(state.ctx.get())) << "\n";
    out << "  fast_sampling_context: "
        << (state.train_config.fast_generation_context ? "true" : "false") << "\n";
    out << "  training_kv_requested_dtype: "
        << (state.train_config.kv_dtype == RETRO_KV_DTYPE_F16 ? "F16" : "F32") << "\n";
    out << "  training_kv_dtype: "
        << (state.effective_kv_dtype == RETRO_KV_DTYPE_F16 ? "F16" : "F32") << "\n";
    out << "  training_kv_f16: " << state.training_kv_status << "\n";
    out << "  gradient_checkpointing: "
        << (state.train_config.gradient_checkpointing ? "enabled" : "disabled") << "\n";
    out << "  gradient_checkpointing_scope: packed_sequences\n";
    out << "  checkpoint_every_n_layers: "
        << state.train_config.checkpoint_every_n_layers << "\n";
    // The one checkpointing option that is not numerically inert, so it is named
    // in the report rather than left implicit: an F16 recompute has its own
    // tolerances and must never be mistaken for the bit-exact default.
    out << "  checkpoint_dtype: "
        << (state.train_config.checkpoint_dtype == RETRO_CHECKPOINT_DTYPE_F16  ? "F16"
          : state.train_config.checkpoint_dtype == RETRO_CHECKPOINT_DTYPE_BF16 ? "BF16"
                                                                              : "F32")
        << "\n";
    out << "  optimizer_flash_attention: "
        << (state.ctx->get_cparams().flash_attn ? "enabled" : "disabled") << "\n";
    // The requested value and whether the limiter engaged, never one without
    // the other: a run whose device resolved to CPU accepts the setting and
    // throttles nothing, and a report naming only the request would be
    // promising GPU throttling that is not happening. Static enough to belong
    // in a cached report; the seconds move on every boundary and are read
    // through retro_trainer_duty_cycle_stats instead.
    const retro_duty_cycle_stats duty_cycle = duty_cycle_snapshot(state.duty_cycle);
    out << "  gpu_duty_cycle_requested: " << duty_cycle.requested_fraction << "\n";
    out << "  gpu_duty_cycle_active: " << (duty_cycle.active ? "true" : "false") << "\n";
    if (state.duty_cycle.inactive_on_cpu()) {
        out << "  gpu_duty_cycle_reason: cpu_backend\n";
    }
    out << "  threads: " << state.effective_threads << "\n";
    out << "  n_gpu_layers: " << state.n_gpu_layers << "\n";
    out << "  lora_dtype: " << lora_dtype_name(state.lora_dtype) << "\n";
    out << "  optimizer_f16: " << optimizer_f16_status(state) << "\n";
    // Every byte figure in this report comes from `memory_totals`, the same computation
    // `retro_trainer_memory_report` hands to callers as data. The report renders
    // it; it does not recompute it.
    const retro_memory_report totals = memory_totals(state);
    out << "  trainable_parameter_bytes: " << totals.trainable_parameter_bytes << "\n";
    out << "  trainable_gradient_bytes: " << totals.trainable_gradient_bytes << "\n";
    out << "  optimizer_state_bytes: " << totals.optimizer_state_bytes << "\n";
    out << "  trainable_parameters_are_model_subset: "
        << (totals.trainable_parameters_are_model_subset ? 1 : 0) << "\n";
    out << "  lora_f32_master_copy: false\n";

    // Byte-accurate breakdown per backend buffer type, kept for the per-buffer
    // detail block at the end of the report.
    const auto optimizer_mem = context_memory_by_buffer(state.ctx.get());
    const auto generation_mem = context_memory_by_buffer(state.generation_ctx.get());

    out << "  model_weight_bytes: " << totals.model_weight_bytes << "\n";
    out << "  compute_buffer_bytes: " << totals.optimizer_compute_bytes << "\n";
    out << "  optimizer_compute_bytes: " << totals.optimizer_compute_bytes << "\n";
    out << "  training_kv_cache_bytes: " << totals.optimizer_kv_bytes << "\n";
    if (totals.has_generation_context) {
        out << "  generation_kv_cache_bytes: " << totals.generation_kv_bytes << "\n";
        out << "  generation_compute_bytes: " << totals.generation_compute_bytes << "\n";
    }
    out << "  adapter_buffer_is_host: " << (totals.adapter_on_host ? 1 : 0) << "\n";
    out << "  base_trainable_buffer_is_host: "
        << (totals.base_trainable_on_host ? 1 : 0) << "\n";
    out << "  memory_device_bytes: " << totals.device_bytes << "\n";
    out << "  memory_host_bytes: " << totals.host_bytes << "\n";

    // Measured device memory includes backend scratch and transient graph
    // allocations that are absent from the buffer totals. Emit the values only
    // after sampling so zero is not mistaken for an observation.
    out << "  device_memory_samples: " << totals.device_memory_samples << "\n";
    if (totals.device_memory_samples > 0) {
        out << "  device_total_bytes: " << totals.device_total_bytes << "\n";
        // Device-wide: other processes are included. Deltas between two runs of
        // the same shape are meaningful, the absolute value is not "this run".
        out << "  device_used_bytes: " << totals.device_used_bytes << "\n";
        out << "  device_peak_used_bytes: " << totals.device_peak_used_bytes << "\n";
        // Attributable to this process's backends, unlike device_*.
        out << "  backend_scratch_bytes: " << totals.backend_scratch_bytes << "\n";
        out << "  backend_scratch_peak_bytes: " << totals.backend_scratch_peak_bytes << "\n";
    }

    // The retained activation checkpoints, which O7 has to size against the
    // measured peak before an offload ring can be justified. `checkpoint_count: 0`
    // means the run is not checkpointing or has not built a backward graph yet;
    // optional checkpoint lines are then absent rather than zero, so an unmeasured run and a run
    // that genuinely retains nothing do not read alike.
    out << "  checkpoint_count: " << totals.checkpoint_count << "\n";
    if (totals.checkpoint_count > 0) {
        out << "  checkpoint_retained_bytes: " << totals.checkpoint_retained_bytes << "\n";
        out << "  checkpoint_live_peak_bytes: " << totals.checkpoint_live_peak_bytes << "\n";
        out << "  checkpoint_live_peak_count: " << totals.checkpoint_live_peak_count << "\n";
        out << "  checkpoint_long_lived_bytes: " << totals.checkpoint_long_lived_bytes << "\n";
        out << "  checkpoint_long_lived_count: " << totals.checkpoint_long_lived_count << "\n";
        // Positions in the backward graph's node array, not durations.
        out << "  checkpoint_graph_nodes: " << totals.checkpoint_graph_nodes << "\n";
        out << "  checkpoint_max_span_nodes: " << totals.checkpoint_max_span_nodes << "\n";
        out << "  checkpoint_total_span_nodes: " << totals.checkpoint_total_span_nodes << "\n";
    }

    // Finest granularity: one line per (context, buffer type) with its model /
    // KV / compute bytes and whether that buffer is host memory. `compute` is
    // only populated once a graph has been reserved (after preflight); before
    // that it reads zero. The `generation` rows repeat the shared model bytes so
    // each context line is self-describing; only the optimizer model is summed
    // into `model_weight_bytes`.
    out << "  memory_by_buffer (bytes):\n";
    for (const auto & item : optimizer_mem) {
        out << "    optimizer " << item.first
            << " host=" << (item.second.is_host ? 1 : 0)
            << " model=" << item.second.model
            << " kv=" << item.second.kv
            << " compute=" << item.second.compute << "\n";
    }
    for (const auto & item : generation_mem) {
        out << "    generation " << item.first
            << " host=" << (item.second.is_host ? 1 : 0)
            << " model=" << item.second.model
            << " kv=" << item.second.kv
            << " compute=" << item.second.compute << "\n";
    }

    std::map<std::string, size_t> model_buffers;
    for (const auto & item : state.model->tensors_by_name) {
        model_buffers[buffer_type_label(item.second)] += 1;
    }
    out << "  model_tensors_by_buffer:\n";
    for (const auto & item : model_buffers) {
        out << "    " << item.first << ": " << item.second << "\n";
    }

    // Quantized models mix a dominant weight type (for example Q6_K) with small
    // F32 norm/bias tensors. Report the type that accounts for the most bytes.
    std::map<std::string, size_t> model_type_bytes;
    std::map<std::string, size_t> model_type_counts;
    for (const auto & item : state.model->tensors_by_name) {
        const ggml_tensor * tensor = item.second;
        if (tensor) {
            model_type_bytes[ggml_type_name(tensor->type)] += ggml_nbytes(tensor);
            model_type_counts[ggml_type_name(tensor->type)] += 1;
        }
    }
    std::string dominant_model_type = "unknown";
    size_t dominant_model_bytes = 0;
    for (const auto & item : model_type_bytes) {
        if (item.second > dominant_model_bytes) {
            dominant_model_bytes = item.second;
            dominant_model_type = item.first;
        }
    }
    out << "  model_weight_dtype: " << dominant_model_type << "\n";

    // Full per-type breakdown of the frozen base weights, not just the dominant
    // one: the quantized backward paths are per-type (a model mixing Q4_K blocks
    // with an IQ4_XS head takes two different routes), so the dominant type alone
    // cannot explain why a backend is slow on a given base.
    out << "  base_weight_types:\n";
    for (const auto & item : model_type_bytes) {
        out << "    " << item.first
            << ": tensors=" << model_type_counts[item.first]
            << " bytes=" << item.second << "\n";
    }
    out << "  quantized_backward_path: "
        << quantized_backward_path(state.gpu_active ? first_gpu_device() : nullptr, *state.model)
        << "\n";

    if (state.adapter) {
        std::map<std::string, size_t> lora_buffers;
        for (const auto & item : state.adapter->ab_map) {
            lora_buffers[buffer_type_label(item.second.a)] += 1;
            lora_buffers[buffer_type_label(item.second.b)] += 1;
        }
        out << "  lora_tensors_by_buffer:\n";
        for (const auto & item : lora_buffers) {
            out << "    " << item.first << ": " << item.second << "\n";
        }
    }
    return out.str();
}

// Whether the shared-prefix trainer may put several sequences in one physical
// micro-batch for this trainer.
// This was a list of architecture names maintained here, next to no code that
// implements the property and next to nothing that could fail when the list
// went stale -- and it had: mamba, mamba2, rwkv, minimax-01, kimi-k3,
// bailingmoe3 and qwen4exp all abort on a packed micro-batch and none of them
// was named. The answer now comes from the two places that can give it: the
// graph declares it (llama_model_supports_packed_seq, honoured by the fork at
// the entries to the packed split) and the device is measured against that
// declaration at
// load time (packed_forward_keeps_sequences_apart). `state.packed_seq_status`
// carries which of the two decided, for the capability report.
bool shared_prefix_packed_training(const trainer_state & state) {
    return state.model != nullptr && state.cap_packed_seq_training;
}

std::string capability_report(const trainer_state & state) {
    const std::string architecture = state.model ? state.model->arch_name() : "unloaded";
    lora_profile profile;
    bool has_auto_profile = false;
    std::vector<std::string> candidates;
    std::map<std::string, size_t> tensor_types;
    if (state.model) {
        has_auto_profile = detect_lora_profile(*state.model, profile);
        candidates = lora_candidate_patterns(*state.model);
        for (const auto & item : state.model->tensors_by_name) {
            const ggml_tensor * tensor = item.second;
            if (tensor) {
                tensor_types[ggml_type_name(tensor->type)] += 1;
            }
        }
    }

    std::ostringstream out;
    out << "model capabilities\n";
    out << "  architecture: " << architecture << "\n";
    out << "  llama_cpp_fork_commit: " << RETRO_LLAMA_CPP_COMMIT << "\n";
    out << "  llama_cpp_upstream_commit: " << RETRO_LLAMA_CPP_UPSTREAM_COMMIT << "\n";
    out << "  requested_device: " << device_kind_name(state.requested_device) << "\n";
    out << "  effective_device: " << (state.gpu_active ? "gpu" : "cpu") << "\n";
    out << "  lora_dtype: " << lora_dtype_name(state.lora_dtype) << "\n";
    out << "  optimizer_f16: " << optimizer_f16_status(state) << "\n";
    out << "  automatic_target_profile: " << (has_auto_profile ? profile.name : "none") << "\n";
    if (has_auto_profile) {
        out << "  automatic_target_patterns: [" << join_patterns(profile.patterns) << "]\n";
    }
    out << "  lora_candidate_patterns: " << candidates.size() << "\n";
    for (const std::string & candidate : candidates) {
        out << "    " << candidate << "\n";
    }
    out << "  model_tensor_types:\n";
    for (const auto & item : tensor_types) {
        out << "    " << item.first << ": " << item.second << "\n";
    }
    out << "  training_graph:\n";
    out << "    shared_prefix_packed_training: "
        << (shared_prefix_packed_training(state) ? "yes" : "no") << "\n";
    // Why, and who decided: the architecture's declaration or this device's
    // measurement. A "no" that names neither is a support question nobody can
    // answer from a log.
    out << "    packed_seq: " << state.packed_seq_status << "\n";
    out << "    micro_batch: " << state.effective_ubatch
        << " (requested " << state.train_config.n_ubatch << ")\n";
    out << "    micro_batch_finite_check: " << state.ubatch_finite_status << "\n";
    out << "    recurrent_rollback_derived: " << state.recurrent_rollback_status << "\n";
    out << "    recurrent_rollback_applied: " << state.effective_rs_seq << "\n";
    if (state.preflight_missing >= 0) {
        out << "    preflight: " << (state.preflight_missing == 0 ? "passed" : "failed") << "\n";
        out << "    missing_gradient_rules: " << state.preflight_missing << "\n";
    } else {
        out << "    preflight: pending (runs automatically before the first optimizer step)\n";
    }
    out << "    status: "
        << (state.opt_initialized ? "initialized" : "pending_lora_preflight") << "\n";
    if (!has_auto_profile) {
        out << "    hint: pass --targets explicitly; see lora_candidate_patterns\n";
    }
    return out.str();
}

// The RIR half of the capability report. Deliberately
// *not* part of capability_report(): that one is cached for the trainer's
// lifetime because nothing in it moves, whereas these counters move on every
// graph. Serving them from the cache would report the first step forever.
std::string rir_capability_section(const trainer_state & state) {
    const ggml_rir_mode mode = ggml_rir_get_mode();
    const char * mode_name = "off";
    switch (mode) {
        case GGML_RIR_MODE_OFF:     mode_name = "off";     break;
        case GGML_RIR_MODE_OBSERVE: mode_name = "observe"; break;
        case GGML_RIR_MODE_PREFER:  mode_name = "prefer";  break;
        case GGML_RIR_MODE_REQUIRE: mode_name = "require"; break;
    }

    // Which backends this build can actually dispatch on. A variant whose
    // backend is not registered here is declared but unreachable, and saying so
    // is the difference between "RIR is off" and "RIR had nowhere to run".
    // The registry's own spelling is not the rir_backend spelling: ggml names
    // the Metal registration "MTL". Matching on ggml_rir_backend_name alone
    // reported backend_not_built for a backend that was dispatching every node,
    // so the aliases are explicit and the graph-coverage test asserts the row.
    std::map<std::string, bool> backend_registered;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) {
            continue;
        }
        const char * name = ggml_backend_reg_name(reg);
        if (!name) {
            continue;
        }
        std::string lower(name);
        std::transform(lower.begin(), lower.end(), lower.begin(),
                [](unsigned char c) { return (char) std::tolower(c); });
        backend_registered[lower] = true;
    }
    auto backend_is_registered = [&](uint8_t backend) {
        switch ((rir_backend) backend) {
            case RIR_BACKEND_CUDA:   return backend_registered.count("cuda") != 0;
            case RIR_BACKEND_VULKAN: return backend_registered.count("vulkan") != 0;
            case RIR_BACKEND_METAL:  return backend_registered.count("mtl") != 0 ||
                                            backend_registered.count("metal") != 0;
            case RIR_BACKEND_CPU:    return backend_registered.count("cpu") != 0;
        }
        return false;
    };

    std::ostringstream out;
    out << "  rir:\n";
    out << "    mode: " << mode_name << "\n";
    out << "    registry_schema: " << RIR_REGISTRY_SCHEMA << "\n";
    out << "    variants:\n";
    for (uint32_t i = 0; i < rir_variant_count; ++i) {
        const rir_variant_desc & v = rir_variants[i];
        const char * backend = ggml_rir_backend_name(v.backend);
        const bool reachable = backend_is_registered(v.backend);
        out << "      " << v.ggml_op << "/" << backend << ": " << v.variant_id
            << " priority=" << (unsigned) v.priority;
        if (!reachable) {
            out << " unavailable=backend_not_built";
        } else if (mode == GGML_RIR_MODE_OFF) {
            out << " unavailable=mode_off";
        } else {
            out << " available";
        }
        out << "\n";
    }
    out << "    native_only:\n";
    for (uint32_t i = 0; i < rir_op_policy_count; ++i) {
        const rir_op_policy & p = rir_op_policies[i];
        if (p.policy == RIR_POLICY_NATIVE_ONLY) {
            out << "      " << p.ggml_op << "/" << ggml_rir_backend_name(p.backend) << "\n";
        }
    }
    // The opposite end of the same ladder: pairs with
    // no native kernel left at all. An operator reading this report needs it for
    // one concrete reason - on these pairs `RETRO_RIR_MODE=off` does not restore
    // the default behavior, it moves the op to the CPU.
    out << "    native_retired:\n";
    for (uint32_t i = 0; i < rir_op_policy_count; ++i) {
        const rir_op_policy & p = rir_op_policies[i];
        if (p.native_retired) {
            out << "      " << p.ggml_op << "/" << ggml_rir_backend_name(p.backend) << "\n";
        }
    }

    // Counters since this trainer was created, by differencing the baseline it
    // captured. The underlying counters are process-wide, so without the
    // baseline a second trainer would inherit the first one's coverage.
    const ggml_rir_counters now = ggml_rir_counters_snapshot();
    const ggml_rir_counters & base = state.rir_baseline;
    out << "    counters_since_trainer:\n";
    out << "      ops_seen: "          << (now.ops_seen          - base.ops_seen)          << "\n";
    out << "      rir_eligible: "      << (now.rir_eligible      - base.rir_eligible)      << "\n";
    out << "      rir_dispatched: "    << (now.rir_dispatched    - base.rir_dispatched)    << "\n";
    out << "      native_dispatched: " << (now.native_dispatched - base.native_dispatched) << "\n";
    for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
        const uint64_t delta = now.reject_by_reason[i] - base.reject_by_reason[i];
        if (delta != 0) {
            out << "      reject_" << ggml_rir_reject_name(i) << ": " << delta << "\n";
        }
    }

    // The per-(op, backend, variant) rows, process-wide. Not differenced: the
    // site rows are the axis a second op is added against, and a wrong baseline
    // there would hide exactly what they exist to show.
    const std::string sites = rir_site_lines();
    if (!sites.empty()) {
        out << "    sites (process-wide):\n";
        std::istringstream lines(sites);
        std::string line;
        while (std::getline(lines, line)) {
            out << "      " << line << "\n";
        }
    }
    return out.str();
}

int backend_list_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        ensure_backend_initialized();
        static const std::string report = [] {
            std::ostringstream out;
            const size_t n = ggml_backend_dev_count();
            for (size_t i = 0; i < n; ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (!dev) {
                    continue;
                }
                const char * type = "other";
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:   type = "cpu";   break;
                    case GGML_BACKEND_DEVICE_TYPE_GPU:
                    case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "gpu";   break;
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel"; break;
                    default:                             type = "other"; break;
                }
                const char * name = device_name(dev);
                const char * desc = ggml_backend_dev_description(dev);
                out << type << "\t"
                    << (name ? name : "") << "\t"
                    << (desc ? desc : "") << "\n";
            }
            return out.str();
        }();
        return copy_string_out(report, buffer, n_buffer, out_n_bytes);
    });
}

int device_memory_impl(size_t * out_free, size_t * out_total) {
    return boundary([&]() -> int {
        ensure_backend_initialized();
        ggml_backend_dev_t dev = first_gpu_device();
        if (!dev) {
            set_error("no GPU device is registered in this build");
            return -1;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (out_free) {
            *out_free = free_bytes;
        }
        if (out_total) {
            *out_total = total_bytes;
        }
        return 0;
    });
}


// retro delta: measure the cost of a host<->device copy for pinned and pageable
// buffers on this machine.
// The measurement is deliberately unoverlapped: every copy is followed by a
// synchronize, so what is timed is one round trip end to end with no queue depth
// hiding any of it. An offload ring would overlap its copies with compute, so a
// rate measured this way understates what a ring achieves - which is the safe
// direction for a gate that decides whether to open work.
int transfer_probe_impl(size_t bytes, uint32_t iterations, retro_transfer_rates * out_rates) {
    return boundary([&]() -> int {
        ensure_backend_initialized();
        if (!out_rates) {
            set_error("transfer probe needs an output struct");
            return -1;
        }
        if (bytes == 0 || iterations == 0) {
            set_error("transfer probe needs a non-zero size and iteration count");
            return -1;
        }
        ggml_backend_dev_t dev = first_gpu_device();
        if (!dev) {
            set_error("no GPU device is registered in this build");
            return -1;
        }
        ggml_backend_ptr backend { ggml_backend_dev_init(dev, nullptr) };
        if (!backend) {
            set_error("the GPU device could not be initialized");
            return -1;
        }

        // A one-dimensional I8 tensor of exactly `bytes`: the probe measures a
        // transfer, and giving it a shape it does not need would only add ways for
        // the allocation to differ from the size being reported.
        ggml_init_params meta { ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
        ggml_context_ptr ctx { ggml_init(meta) };
        if (!ctx) {
            set_error("transfer probe could not allocate its metadata context");
            return -1;
        }
        ggml_tensor * target = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) bytes);
        if (!target) {
            set_error("transfer probe could not build its device tensor");
            return -1;
        }
        ggml_backend_buffer_ptr device_buffer {
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()) };
        if (!device_buffer) {
            set_error("transfer probe could not allocate " + std::to_string(bytes)
                    + " bytes on the device");
            return -1;
        }

        // Pageable staging: an ordinary heap allocation, the baseline the pinned
        // row is only interesting relative to.
        std::vector<uint8_t> pageable(bytes, 0x5a);

        // Pinned staging: the backend's own host buffer type. Absent on backends
        // that have no such concept, which is recorded rather than papered over.
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        ggml_backend_buffer_ptr host_buffer;
        uint8_t * pinned = nullptr;
        if (host_buft) {
            host_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, bytes));
            if (host_buffer) {
                pinned = (uint8_t *) ggml_backend_buffer_get_base(host_buffer.get());
            }
        }
        out_rates->pinned_is_pageable = pinned == nullptr;
        if (!pinned) {
            pinned = pageable.data();
        }

        const auto time_round_trips = [&](uint8_t * staging, double & h2d, double & d2h) {
            // One discarded warm-up: the first copy pays for lazily created
            // command queues, pipeline objects and first-touch page faults, none
            // of which an offload ring would pay per transfer.
            ggml_backend_tensor_set(target, staging, 0, bytes);
            ggml_backend_tensor_get(target, staging, 0, bytes);
            ggml_backend_synchronize(backend.get());

            auto started = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                ggml_backend_tensor_set(target, staging, 0, bytes);
                ggml_backend_synchronize(backend.get());
            }
            const double h2d_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();

            started = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                ggml_backend_tensor_get(target, staging, 0, bytes);
                ggml_backend_synchronize(backend.get());
            }
            const double d2h_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();

            const double moved = (double) bytes * (double) iterations;
            h2d = h2d_seconds > 0.0 ? moved / h2d_seconds : 0.0;
            d2h = d2h_seconds > 0.0 ? moved / d2h_seconds : 0.0;
        };

        time_round_trips(pinned, out_rates->pinned_h2d_bytes_per_second,
                out_rates->pinned_d2h_bytes_per_second);
        if (out_rates->pinned_is_pageable) {
            out_rates->pageable_h2d_bytes_per_second = out_rates->pinned_h2d_bytes_per_second;
            out_rates->pageable_d2h_bytes_per_second = out_rates->pinned_d2h_bytes_per_second;
        } else {
            time_round_trips(pageable.data(), out_rates->pageable_h2d_bytes_per_second,
                    out_rates->pageable_d2h_bytes_per_second);
        }
        out_rates->bytes_per_transfer = bytes;
        out_rates->iterations = iterations;
        out_rates->device_buffer_is_host =
                ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(dev));
        return 0;
    });
}

} // namespace retro
