#include "retro_runtime.hpp"

#include <cmath>
#include <limits>

// The grouped softmax cross-entropy probe: see retro_grouped_gradient_probe.

namespace retro {

namespace {

struct model_deleter {
    void operator()(llama_model * model) const { llama_model_free(model); }
};
struct context_deleter {
    void operator()(llama_context * ctx) const { llama_free(ctx); }
};
struct batch_owner {
    llama_batch batch;
    explicit batch_owner(int32_t n) : batch(llama_batch_init(n, 0, 1)) {}
    ~batch_owner() { llama_batch_free(batch); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};
struct result_owner {
    ggml_opt_result_t result = ggml_opt_result_init();
    ~result_owner() { ggml_opt_result_free(result); }
    result_owner() = default;
    result_owner(const result_owner &) = delete;
    result_owner & operator=(const result_owner &) = delete;
};

bool marks_named(const ggml_tensor * tensor, void * userdata) {
    return std::strcmp(tensor->name, static_cast<const char *>(userdata)) == 0;
}

// A zero learning rate: the step computes the gradient and leaves the weights
// as they are, so the numeric side differentiates the same point. No clipping.
ggml_opt_optimizer_params frozen_optimizer(void *) {
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.max_grad_norm = std::numeric_limits<float>::max();
    params.adamw.alpha   = 0.0f;
    params.adamw.wd      = 0.0f;
    return params;
}

void preflight_ignore(int32_t, ggml_backend_dev_t, const ggml_tensor *, void *) {}

// The loss of the groups, from the embeddings of a decode, in double: the
// same masked, tempered softmax cross-entropy the training graph builds,
// averaged over the groups.
bool decode_loss(llama_context * ctx, const llama_batch & batch, const retro_grouped_gradient_probe & p, double & loss) {
    // a causal model keeps the previous decode in its memory
    if (llama_memory_t memory = llama_get_memory(ctx)) {
        llama_memory_clear(memory, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        set_error("llama_decode failed");
        return false;
    }
    double total = 0.0;
    for (uint32_t g = 0; g < p.n_groups; ++g) {
        std::vector<double> scores;
        std::vector<double> targets;
        for (uint32_t j = 0; j < p.n_opt_max; ++j) {
            const size_t  k   = (size_t) g*p.n_opt_max + j;
            const int32_t row = p.rows[k];
            if (row < 0) {
                continue;
            }
            const float * embd = llama_get_embeddings_ith(ctx, row);
            if (!embd) {
                set_error("no embeddings at a read token");
                return false;
            }
            if (p.pointer_rows) {
                // q.k / sqrt(d), q the first half of the query's row, k the
                // second half of the option's
                const float * query = llama_get_embeddings_ith(ctx, p.pointer_rows[g]);
                if (!query) {
                    set_error("no embeddings at a query token");
                    return false;
                }
                const int32_t d = llama_model_n_embd_out(llama_get_model(ctx)) / 2;
                double dot = 0.0;
                for (int32_t i = 0; i < d; ++i) {
                    dot += (double) query[i] * (double) embd[d + i];
                }
                scores.push_back(dot / std::sqrt((double) d) * (double) p.inv_temperature);
                targets.push_back(p.targets[k]);
                continue;
            }
            scores.push_back((double) embd[p.columns[g]] * (double) p.inv_temperature);
            targets.push_back(p.targets[k]);
        }
        double max = -std::numeric_limits<double>::infinity();
        for (const double s : scores) {
            max = std::max(max, s);
        }
        double sum = 0.0;
        for (const double s : scores) {
            sum += std::exp(s - max);
        }
        const double log_z = max + std::log(sum);
        for (size_t j = 0; j < scores.size(); ++j) {
            total -= targets[j] * (scores[j] - log_z);
        }
    }
    loss = total / p.n_groups;
    return true;
}

} // namespace

int grouped_gradient_probe_impl(retro_grouped_gradient_probe * probe) {
    return boundary([&]() -> int {
        if (!probe || probe->struct_size != sizeof(retro_grouped_gradient_probe)) {
            set_error("retro_grouped_gradient_probe: struct_size mismatch");
            return -1;
        }
        retro_grouped_gradient_probe & p = *probe;
        if (!p.model_path || !p.param_name || !p.tokens || !p.positions || !p.seq_ids || !p.outputs ||
                p.n_tokens <= 0 || p.n_elements < 0 || !std::isfinite(p.step) ||
                p.flash_attn < -1 || p.flash_attn > 1 || !p.rows || (!p.columns && !p.pointer_rows) || !p.targets || p.n_groups == 0 || p.n_opt_max == 0 ||
                (p.n_elements > 0 && (!p.elements || !p.grad_analytic || !p.grad_numeric)) || !(p.step > 0.0f)) {
            set_error("retro_grouped_gradient_probe: missing or invalid field");
            return -1;
        }
        ensure_backend_initialized();

        llama_model_params mparams = llama_model_default_params();
        ggml_backend_dev_t no_devices[] = { nullptr };
        if (p.use_gpu) {
            mparams.n_gpu_layers = -1;
        } else {
            mparams.devices      = no_devices;
            mparams.n_gpu_layers = 0;
        }
        // the weights are written in place by the finite differences
        mparams.load_mode       = LLAMA_LOAD_MODE_NONE;
        mparams.use_extra_bufts = false;
        std::unique_ptr<llama_model, model_deleter> model(llama_model_load_from_file(p.model_path, mparams));
        if (!model) {
            set_error(std::string("cannot load ") + p.model_path);
            return -1;
        }
        ggml_tensor * param = nullptr;
        for (const auto & [name, tensor] : model->tensors_by_name) {
            if (name == p.param_name) {
                param = tensor;
            }
        }
        if (!param || param->type != GGML_TYPE_F32) {
            set_error(std::string("no F32 tensor named ") + p.param_name);
            return -1;
        }
        for (int32_t i = 0; i < p.n_elements; ++i) {
            if (p.elements[i] < 0 || p.elements[i] >= ggml_nelements(param)) {
                set_error("an element is outside the tensor");
                return -1;
            }
        }

        int32_t n_seq = 1;
        for (int32_t i = 0; i < p.n_tokens; ++i) {
            // Bound sequence IDs before adding one or sizing the context.
            if (p.seq_ids[i] < 0 || p.seq_ids[i] >= p.n_tokens || p.positions[i] < 0) {
                set_error("invalid sequence ID or position");
                return -1;
            }
            n_seq = std::max(n_seq, p.seq_ids[i] + 1);
        }
        // One micro-batch that holds the whole batch, and a context that is a
        // multiple of it, as llama_opt_init requires.
        const uint32_t n_ctx = 256;
        if ((uint32_t) p.n_tokens > n_ctx) {
            set_error("the probe batch is limited to 256 tokens");
            return -1;
        }
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = n_ctx;
        cparams.n_batch         = n_ctx;
        cparams.n_ubatch        = n_ctx;
        cparams.n_seq_max       = (uint32_t) n_seq;
        cparams.embeddings      = true;
        cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
        cparams.flash_attn_type = (llama_flash_attn_type) p.flash_attn;
        std::unique_ptr<llama_context, context_deleter> ctx(llama_init_from_model(model.get(), cparams));
        if (!ctx) {
            set_error("cannot create a context");
            return -1;
        }

        llama_opt_params oparams {};
        oparams.n_ctx_train      = 0;
        oparams.param_filter     = marks_named;
        oparams.param_filter_ud  = const_cast<char *>(p.param_name);
        oparams.get_opt_pars     = frozen_optimizer;
        oparams.get_opt_pars_ud  = nullptr;
        oparams.optimizer_type   = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        oparams.n_ce_tiles       = 1;
        oparams.checkpoint_every_n_layers = 1;
        oparams.checkpoint_type  = GGML_TYPE_COUNT;
        oparams.optimizer_layout = ggml_opt_default_optimizer_layout();
        oparams.grouped_loss     = true;
        llama_opt_init(ctx.get(), model.get(), oparams);

        p.n_missing_backward = llama_opt_preflight(ctx.get(), preflight_ignore, nullptr);
        if (p.n_missing_backward != 0) {
            return 0;
        }

        batch_owner owned(p.n_tokens);
        llama_batch & batch = owned.batch;
        batch.n_tokens = p.n_tokens;
        for (int32_t i = 0; i < p.n_tokens; ++i) {
            batch.token[i]     = p.tokens[i];
            batch.pos[i]       = p.positions[i];
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = p.seq_ids[i];
            batch.logits[i]    = p.outputs[i] ? 1 : 0;
        }

        llama_grouped_target target {};
        target.n_groups           = p.n_groups;
        target.n_opt_max          = p.n_opt_max;
        target.rows               = p.rows;
        target.columns            = p.columns;
        target.targets            = p.targets;
        target.inv_temperature    = p.inv_temperature;
        target.accumulation_steps = 1;
        target.pointer_rows       = p.pointer_rows;

        result_owner result;
        if (!llama_opt_step_grouped(ctx.get(), batch, &target, result.result)) {
            set_error("llama_opt_step_grouped refused the batch");
            return -1;
        }
        double unc = 0.0;
        ggml_opt_result_loss(result.result, &p.loss_step, &unc);

        ggml_tensor * grad = ggml_opt_grad_acc_by_name(llama_opt_context(ctx.get()), p.param_name);
        if (!grad || grad->type != GGML_TYPE_F32 || ggml_nelements(grad) != ggml_nelements(param)) {
            set_error("no F32 gradient accumulator for the parameter");
            return -1;
        }
        for (int32_t i = 0; i < p.n_elements; ++i) {
            ggml_backend_tensor_get(grad, &p.grad_analytic[i], (size_t) p.elements[i]*sizeof(float), sizeof(float));
        }

        // the decode reads every token: its outputs are all the positions
        for (int32_t i = 0; i < p.n_tokens; ++i) {
            batch.logits[i] = 1;
        }
        if (!decode_loss(ctx.get(), batch, p, p.loss_decode)) {
            return -1;
        }

        for (int32_t i = 0; i < p.n_elements; ++i) {
            const size_t offset = (size_t) p.elements[i]*sizeof(float);
            float w0 = 0.0f;
            ggml_backend_tensor_get(param, &w0, offset, sizeof(float));
            const double h = p.step;
            double f[4] = {}; // w0 + h, w0 + h/2, w0 - h/2, w0 - h
            const double shifts[4] = { h, 0.5*h, -0.5*h, -h };
            for (int k = 0; k < 4; ++k) {
                const float w = (float) (w0 + shifts[k]);
                ggml_backend_tensor_set(param, &w, offset, sizeof(float));
                if (!decode_loss(ctx.get(), batch, p, f[k])) {
                    ggml_backend_tensor_set(param, &w0, offset, sizeof(float));
                    return -1;
                }
            }
            ggml_backend_tensor_set(param, &w0, offset, sizeof(float));
            p.grad_numeric[i] = (8.0*f[1] + f[3] - 8.0*f[2] - f[0]) / (6.0*h);
        }
        return 0;
    });
}

} // namespace retro
