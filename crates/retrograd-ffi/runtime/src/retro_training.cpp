#include "retro_runtime.hpp"

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <random>
#include <sstream>

namespace retro {

namespace {

// Maps the requested checkpoint precision to the type used by the graph rewrite.
// F32 means no casts, represented by GGML_TYPE_COUNT. The
// RETRO_CHECKPOINT_ROUNDTRIP_F32 override forces an F32 cast round-trip so the
// rewrite can be tested for structural parity without rounding differences.
ggml_type checkpoint_ggml_type(int32_t requested) {
    const char * roundtrip = std::getenv("RETRO_CHECKPOINT_ROUNDTRIP_F32");
    if (roundtrip && roundtrip[0] != '\0' && roundtrip[0] != '0') {
        return GGML_TYPE_F32;
    }
    switch (requested) {
        case RETRO_CHECKPOINT_DTYPE_F16:  return GGML_TYPE_F16;
        case RETRO_CHECKPOINT_DTYPE_BF16: return GGML_TYPE_BF16;
        default:                          return GGML_TYPE_COUNT;
    }
}

struct step_progress {
    trainer_state * state = nullptr;
    uint32_t epoch = 0;
    uint64_t last_step = 0;
    retro_train_progress_callback callback = nullptr;
    void * user_data = nullptr;
    bool continue_training = true;
    // Timestamp of the previous mid-iteration report, used for interval throughput.
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
    // Loss and datapoint count at the previous report, used to derive the
    // interval mean instead of exposing the running epoch mean.
    double last_weighted_sum = 0.0;
    int64_t last_ndata = 0;
};

thread_local step_progress * g_step_progress = nullptr;

// Both optimizer callbacks below are duty-cycle accounting boundaries, and
// neither adds a fence, unlike the decode sites of retro_rollout.cpp:
// `ggml_opt_eval` computes through `ggml_backend_sched_graph_compute`, which is
// `ggml_backend_sched_graph_compute_async` immediately followed by
// `ggml_backend_sched_synchronize` (ggml-backend.cpp). The optimizer path is
// synchronous by construction on every backend, before any result is touched.
// That is a property of the vendored function rather than of this loop, so it
// is written down here to survive the next fork update.
//
// It is also what keeps the sleep out of `execution_seconds`: llama-context.cpp
// stops that timer immediately after `ggml_opt_eval` and before invoking the
// callback, so a profile still tells a slower kernel from deliberate idling.

// Installed only while the limiter is enabled: it derives no metric, and the
// evaluation loop of llama_opt_epoch runs the same graphs on the same device as
// the training loop, so an epoch with a large eval split would otherwise
// overshoot its duty cycle at every epoch boundary.
void on_optimizer_eval(
        bool,
        ggml_opt_context_t,
        ggml_opt_dataset_t,
        ggml_opt_result_t,
        int64_t,
        int64_t,
        int64_t) {
    step_progress * progress = g_step_progress;
    if (!progress) {
        return;
    }
    progress->state->duty_cycle.account_window();
    if (!progress->continue_training) {
        // Same reason as the training callback: the epoch is unwinding a
        // cancellation and must not sleep on the way out.
        progress->state->duty_cycle.reset_window();
        return;
    }
    progress->state->duty_cycle.idle_if_needed();
}

void on_optimizer_step(
        bool train,
        ggml_opt_context_t,
        ggml_opt_dataset_t,
        ggml_opt_result_t result,
        int64_t,
        int64_t,
        int64_t) {
    step_progress * progress = g_step_progress;
    if (!progress) {
        return;
    }
    // Above the progress guard, deliberately. This callback fires on every
    // physical micro-batch while the guard below filters down to logical
    // optimizer steps; accounting under it is the silent failure where the
    // limiter is installed, reports active, and never sleeps during gradient
    // accumulation.
    // The order is the run-control contract: finish and account the GPU work,
    // deliver a due progress callback so cancellation is not hidden behind
    // avoidable idle time, then repay. A micro-batch with no logical callback
    // accounts and sleeps in one go, and a cancelled run repays nothing at
    // all.
    duty_cycle_limiter & limiter = progress->state->duty_cycle;
    const bool throttling = limiter.enabled();
    if (throttling) {
        limiter.account_window();
    }
    if (!progress->continue_training) {
        // A cancelled run owes nothing. The vendored callback returns void, so
        // the native epoch keeps going until its own loop ends: sleeping
        // through every remaining micro-batch would add the whole rest of the
        // epoch's debt to a cancellation the caller is already waiting on.
        if (throttling) {
            limiter.reset_window();
        }
        return;
    }
    if (!train || !progress->callback
            || progress->state->scheduler_step <= progress->last_step) {
        if (throttling) {
            limiter.idle_if_needed();
        }
        return;
    }
    // One optimizer step consumes `n_batch` tokens, so the steps since the
    // previous report divided by their wall time is the live throughput.
    const uint64_t steps = progress->state->scheduler_step - progress->last_step;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - progress->last_report).count();
    progress->last_report = now;
    progress->last_step = progress->state->scheduler_step;
    double mean = NAN;
    double unc = 0.0;
    ggml_opt_result_loss(result, &mean, &unc);
    int64_t ndata = 0;
    ggml_opt_result_ndata(result, &ndata);
    // Derive the mean over new datapoints since the previous callback. If the
    // result did not grow, retain the current reported value.
    double loss = mean;
    const double weighted_sum = mean * (double) ndata;
    if (ndata > progress->last_ndata) {
        loss = (weighted_sum - progress->last_weighted_sum)
                / (double) (ndata - progress->last_ndata);
    }
    progress->last_weighted_sum = weighted_sum;
    progress->last_ndata = ndata;
    retro_train_metrics metrics {};
    metrics.epoch = progress->epoch;
    metrics.epoch_complete = false;
    metrics.global_step = progress->last_step;
    metrics.train_loss = static_cast<float>(loss);
    metrics.eval_loss = NAN;
    metrics.tokens_per_second = seconds > 0.0
            ? static_cast<float>((double) steps * (double) progress->state->train_config.n_batch
                    / seconds)
            : 0.0f;
    metrics.learning_rate = progress->state->last_learning_rate;
    progress->continue_training = progress->callback(
            metrics.epoch, &metrics, progress->user_data);
    if (throttling) {
        if (!progress->continue_training) {
            // The callback just cancelled. Drop the debt here rather than
            // repay it: the epoch still has to unwind, and every boundary left
            // in it would otherwise sleep on the way out.
            limiter.reset_window();
            return;
        }
        // Reopen before repaying. Whatever the callback took - a progress row,
        // a metrics write, a run-control pause acknowledged inside it - is host
        // time the limiter did not choose, and leaving the window open across
        // it would charge it as GPU work at the next boundary and manufacture
        // debt from it. `begin_window` also applies the stale rule, so a real
        // pause does not carry seconds of debt into the resume.
        limiter.begin_window();
        limiter.idle_if_needed();
    }
}

dataset_ptr allocate_sft_dataset(
        llama_context * ctx,
        size_t n_rows,
        uint32_t n_ctx) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    if (n_rows == 0) {
        set_error("SFT dataset rows are required");
        return nullptr;
    }
    if (n_ctx != static_cast<uint32_t>(ne_datapoint)) {
        std::ostringstream message;
        message << "SFT dataset context " << n_ctx
                << " does not match effective model context " << ne_datapoint;
        set_error(message.str());
        return nullptr;
    }
    if (n_rows > static_cast<size_t>(INT64_MAX)) {
        set_error("too many SFT rows for dataset construction");
        return nullptr;
    }
    const int64_t ndata = static_cast<int64_t>(n_rows);
    dataset_ptr dataset(ggml_opt_dataset_init(
            GGML_TYPE_I32,
            GGML_TYPE_I32,
            ne_datapoint,
            ne_datapoint,
            ndata,
            1));

    if (!dataset) {
        set_error("failed to allocate training dataset");
        return nullptr;
    }

    return dataset;
}

dataset_ptr view_sft_dataset(llama_context * ctx, const retro_sft_dataset & data) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    if (data.n_rows == 0 || data.n_ctx != static_cast<uint32_t>(ne_datapoint)
            || data.n_rows > static_cast<size_t>(INT64_MAX)) {
        set_error("invalid SFT dataset shape for zero-copy view");
        return nullptr;
    }
    dataset_ptr dataset(ggml_opt_dataset_init_external(
            GGML_TYPE_I32,
            GGML_TYPE_I32,
            ne_datapoint,
            ne_datapoint,
            static_cast<int64_t>(data.n_rows),
            1,
            const_cast<int32_t *>(data.tokens),
            const_cast<int32_t *>(data.labels)));
    if (!dataset) {
        set_error("failed to create zero-copy SFT dataset view");
    }
    return dataset;
}

dataset_ptr view_split_sft_dataset(
        llama_context * ctx,
        const retro_sft_dataset & train,
        const retro_sft_dataset & eval) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    if (train.n_rows == 0 || eval.n_rows == 0 || train.n_ctx != eval.n_ctx
            || train.n_ctx != static_cast<uint32_t>(ne_datapoint)
            || train.n_rows > static_cast<size_t>(INT64_MAX)
            || eval.n_rows > static_cast<size_t>(INT64_MAX)
            || train.n_rows > static_cast<size_t>(INT64_MAX) - eval.n_rows) {
        set_error("invalid split SFT dataset shape for zero-copy view");
        return nullptr;
    }
    dataset_ptr dataset(ggml_opt_dataset_init_external_split(
            GGML_TYPE_I32,
            GGML_TYPE_I32,
            ne_datapoint,
            ne_datapoint,
            static_cast<int64_t>(train.n_rows),
            static_cast<int64_t>(eval.n_rows),
            1,
            const_cast<int32_t *>(train.tokens),
            const_cast<int32_t *>(train.labels),
            const_cast<int32_t *>(eval.tokens),
            const_cast<int32_t *>(eval.labels)));
    if (!dataset) {
        set_error("failed to create zero-copy split SFT dataset view");
    }
    return dataset;
}

// SplitMix64 (Steele et al., OOPSLA 2014), the same mixer the Rust side uses in
// retrograd-training/src/grpo.rs. Its avalanche makes successive epoch seeds
// draw unrelated permutations.
uint64_t splitmix64(uint64_t & state) {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

// Seeds the optimizer RNG from (seed, epoch), making each epoch's permutation
// independent of prior RNG draws and reproducible after checkpoint resume.
bool seed_epoch_shuffle(ggml_opt_context_t opt, uint64_t seed, uint32_t epoch) {
    uint64_t state = seed * 0x9e3779b97f4a7c15ull ^ ((uint64_t) (epoch + 1) << 1);
    std::mt19937 rng(static_cast<uint32_t>(splitmix64(state) >> 32));
    std::ostringstream stream;
    stream << rng;
    return ggml_opt_set_rng_state(opt, stream.str().c_str());
}

} // namespace

ggml_opt_optimizer_params scheduled_optimizer_params(void * userdata) {
    trainer_state * state = static_cast<trainer_state *>(userdata);
    ggml_opt_optimizer_params params = state->optimizer_params;
    const uint64_t step = state->scheduler_step++;
    const uint64_t total = std::max<uint64_t>(1, state->scheduler_total_steps);
    float factor = 1.0f;
    if (state->train_config.warmup_steps > 0 && step < state->train_config.warmup_steps) {
        factor = static_cast<float>(step + 1)
                / static_cast<float>(state->train_config.warmup_steps);
    } else if (state->train_config.lr_scheduler != 0) {
        const uint64_t decay_start = std::min(state->train_config.warmup_steps, total);
        const uint64_t decay_steps = std::max<uint64_t>(1, total - decay_start);
        const float progress = std::min(
                1.0f,
                static_cast<float>(step - std::min(step, decay_start))
                        / static_cast<float>(decay_steps));
        if (state->train_config.lr_scheduler == 1) {
            factor = 1.0f - progress;
        } else {
            factor = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
        }
    }
    state->last_learning_rate = state->train_config.learning_rate * factor;
    params.adamw.alpha = state->last_learning_rate;
    params.sgd.alpha = state->last_learning_rate;
    return params;
}

bool ensure_opt_context(trainer_state & state) {
    if (state.train_config.trainable == RETRO_TRAINABLE_HYBRID && !state.adapter) {
        set_error("a hybrid policy requires a LoRA adapter and a base trainable set");
        return false;
    }
    if ((state.train_config.trainable == RETRO_TRAINABLE_FULL
            || state.train_config.trainable == RETRO_TRAINABLE_PARTIAL) && state.adapter) {
        set_error("a full or partial policy trains only base weights; use hybrid to train an adapter too");
        return false;
    }
    if (trains_base_weights(state) && !state.trainable_base_set) {
        set_error("this run trains base weights but no trainable set was declared; "
                  "call retro_trainer_set_trainable_base first");
        return false;
    }
    if (state.opt_created) {
        return assert_marked_set_is_resolved(state)
                && optimizer_supports_marked_dtypes(state);
    }

    state.optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    state.optimizer_params.max_grad_norm = state.train_config.max_grad_norm;
    state.optimizer_params.adamw.alpha = state.train_config.learning_rate;
    state.optimizer_params.adamw.wd = state.train_config.weight_decay;
    state.optimizer_params.sgd.alpha = state.train_config.learning_rate;
    state.optimizer_params.sgd.wd = state.train_config.weight_decay;

    // The optimizer the document asked for, not the one this function used to
    // hard-code. Validated at trainer creation, so anything else here would be
    // a value validate_train_config let through.
    const ggml_opt_optimizer_type optimizer_type =
            state.train_config.optimizer == RETRO_OPTIMIZER_SGD
                    ? GGML_OPT_OPTIMIZER_TYPE_SGD
                    : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    llama_opt_params params {
        /*n_ctx_train     =*/ state.train_config.n_ctx,
        /*param_filter    =*/ opt_param_filter_trainable,
        /*param_filter_ud =*/ &state,
        /*get_opt_pars    =*/ scheduled_optimizer_params,
        /*get_opt_pars_ud =*/ &state,
        /*optimizer_type  =*/ optimizer_type,
        /*fused_sparse_ce =*/ state.train_config.chunked_cross_entropy,
        /*n_ce_tiles      =*/ (int32_t) (state.train_config.chunked_ce_tiles > 0
                ? state.train_config.chunked_ce_tiles : 1),
        /*n_ce_seq_chunk  =*/ (int32_t) state.train_config.chunked_ce_seq_chunk,
        /*ce_offload_logsoftmax =*/ state.train_config.chunked_ce_offload_logsoftmax,
        /*gradient_checkpointing    =*/ state.train_config.gradient_checkpointing,
        /*checkpoint_every_n_layers =*/ state.train_config.checkpoint_every_n_layers,
        /*checkpoint_type =*/ checkpoint_ggml_type(state.train_config.checkpoint_dtype),
    };
    llama_opt_init(state.ctx.get(), state.model.get(), params);
    // Recorded before the checks below, not after: llama_opt_init asserts that
    // the context has no optimizer yet, so a caller retrying after a refusal
    // would abort the process instead of getting the same error twice.
    state.opt_created = true;
    // Rule 6 of the trainable-set contract: a flag without a gradient is not a
    // successful selection, so the marked set is compared with the resolved one
    // here rather than assumed to follow from the filter.
    if (!assert_marked_set_is_resolved(state)) {
        return false;
    }
    // Between the graph build and the first step: the update kernels abort on a
    // dtype they do not carry, so the refusal has to come from here.
    if (!optimizer_supports_marked_dtypes(state)) {
        return false;
    }
    return true;
}

bool ensure_optimizer_initialized(trainer_state & state) {
    if (state.opt_initialized) {
        return true;
    }
    // An adapter is required exactly when the run has one to train. A
    // base-weight policy resolves its own parameters and needs no LoRA at all.
    if (!state.adapter && !trains_base_weights(state)) {
        set_error("create or load a LoRA adapter before training");
        return false;
    }
    // ... and a base-weight policy needs its resolved set, which only the
    // caller can produce. An empty one here would train nothing while
    // reporting a policy that says otherwise.
    if (trains_base_weights(state) && !state.trainable_base_set) {
        set_error("this run trains base weights but no trainable set was declared; "
                  "call retro_trainer_set_trainable_base first");
        return false;
    }
    if (state.loaded_lora && !promote_loaded_lora_to_trainable(state)) {
        return false;
    }
    if (!ensure_opt_context(state)) {
        return false;
    }
    // The backward graph aborts the process on the first op without a gradient
    // rule, so refuse to train with the precise diagnosis instead.
    if (!ensure_train_preflight(state)) {
        return false;
    }
    if (state.preflight_missing != 0) {
        set_error("training graph is not supported for this model:\n" + state.preflight_report);
        return false;
    }
    state.opt_initialized = true;
    state.invalidate_report_caches();
    return true;
}

int train_sft_impl(
        retro_trainer * trainer,
        const retro_sft_dataset * train,
        const retro_sft_dataset * eval,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora && !trains_base_weights(*state)) {
            set_error("create or load a LoRA adapter before training");
            return -1;
        }
        if (!train) {
            set_error("training SFT dataset is required");
            return -1;
        }
        if (!out_metrics) {
            set_error("out_metrics is required");
            return -1;
        }

        if (!train->tokens || !train->labels || train->n_rows == 0) {
            set_error("training SFT dataset must contain tokens, labels, and rows");
            return -1;
        }
        if (eval
                && (!eval->tokens || !eval->labels || eval->n_rows == 0
                        || eval->n_ctx != train->n_ctx)) {
            set_error("evaluation SFT dataset is invalid or incompatible");
            return -1;
        }
        const size_t eval_rows = eval ? eval->n_rows : 0;
        if (train->n_rows > SIZE_MAX - eval_rows) {
            set_error("too many combined SFT rows");
            return -1;
        }
        const size_t total_rows = train->n_rows + eval_rows;
        const size_t row_width = train->n_ctx;
        if (row_width == 0 || total_rows > SIZE_MAX / row_width) {
            set_error("invalid SFT dataset shape");
            return -1;
        }
        // The caller-owned Rust buffers stay alive for this synchronous call.
        // A split dataset is represented as a logical concatenation whose
        // shard lookup switches from the train buffers to the eval buffers.
        dataset_ptr dataset = eval
                ? view_split_sft_dataset(state->ctx.get(), *train, *eval)
                : view_sft_dataset(state->ctx.get(), *train);
        if (!dataset) {
            return -1;
        }
        const uint64_t steps_per_row = static_cast<uint64_t>(state->train_config.n_ctx) /
                state->train_config.n_batch;
        // A resumed run keeps the scheduler step restored from its checkpoint;
        // the configured horizon is the full run either way, so warm-up
        // and decay stay on the trajectory the first launch started.
        if (!state->resume_active) {
            state->scheduler_step = 0;
        }
        state->scheduler_total_steps = static_cast<uint64_t>(train->n_rows) * steps_per_row *
                state->train_config.epochs;
        if (state->resume_epoch >= state->train_config.epochs) {
            set_error("the resume point is at or past the configured number of epochs");
            return -1;
        }
        if (state->train_config.warmup_steps > state->scheduler_total_steps) {
            set_error("warmup_steps exceeds the total number of optimizer steps");
            return -1;
        }
        state->last_learning_rate = state->train_config.learning_rate;
        if (!ensure_optimizer_initialized(*state)) {
            return -1;
        }

        *out_metrics = retro_train_metrics {};
        opt_result_ptr result_train(ggml_opt_result_init());
        if (!result_train) {
            set_error("failed to allocate training metrics result");
            return -1;
        }
        opt_result_ptr result_eval(ggml_opt_result_init());
        if (!result_eval) {
            set_error("failed to allocate eval metrics result");
            return -1;
        }

        // The internal optimizer callback is what the duty-cycle limiter
        // accounts on, so it is installed for throttling as well as for
        // progress: without this the limiter would report active and never
        // sleep during a run that asked for no progress rows.
        const bool throttling = state->duty_cycle.enabled();
        const bool needs_optimizer_callback = progress_callback || throttling;

        const int64_t idata_split = static_cast<int64_t>(train->n_rows);
        const auto t0 = std::chrono::steady_clock::now();
        double train_loss = NAN;
        double eval_loss = NAN;
        uint32_t completed_epochs = state->resume_epoch;
        for (uint32_t epoch = state->resume_epoch; epoch < state->train_config.epochs; ++epoch) {
            if (state->train_config.shuffle_dataset) {
                ggml_opt_context_t opt = llama_opt_context(state->ctx.get());
                if (!opt || !seed_epoch_shuffle(
                                    opt, state->train_config.shuffle_seed, epoch)) {
                    set_error("failed to seed the SFT shuffle for this epoch");
                    return -1;
                }
                // `idata_split`, never -1: -1 permutes the whole permutation,
                // which on a split dataset mixes the evaluation rows into
                // training. The shuffle only reorders shard indices, so the
                // zero-copy views over the Rust-owned buffers are untouched.
                ggml_opt_dataset_shuffle(opt, dataset.get(), idata_split);
            }
            ggml_opt_result_reset(result_train.get());
            ggml_opt_result_reset(result_eval.get());
            const auto epoch_t0 = std::chrono::steady_clock::now();
            step_progress step_events {
                state,
                epoch + 1,
                state->scheduler_step,
                progress_callback,
                progress_user_data,
            };
            g_step_progress = &step_events;
            state->duty_cycle.begin_window();
            llama_opt_epoch(
                    state->ctx.get(),
                    dataset.get(),
                    result_train.get(),
                    result_eval.get(),
                    idata_split,
                    needs_optimizer_callback ? on_optimizer_step : nullptr,
                    throttling ? on_optimizer_eval : nullptr);
            g_step_progress = nullptr;
            if (!step_events.continue_training) {
                // A stopped run owes nothing: the debt describes compute this
                // trainer is about to stop taking.
                state->duty_cycle.reset_window();
                break;
            }
            double unc = 0.0;
            ggml_opt_result_loss(result_train.get(), &train_loss, &unc);
            ggml_opt_result_loss(result_eval.get(), &eval_loss, &unc);
            out_metrics->epoch = epoch + 1;
            out_metrics->epoch_complete = true;
            out_metrics->global_step = state->scheduler_step;
            out_metrics->train_loss = static_cast<float>(train_loss);
            out_metrics->eval_loss = eval ? static_cast<float>(eval_loss) : NAN;
            // The throughput of *this* epoch, not a zero the caller has to
            // explain: the run-wide average only exists once the loop is
            // over, which is too late for a progress row.
            const double epoch_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_t0)
                            .count();
            out_metrics->tokens_per_second = epoch_seconds > 0.0
                    ? static_cast<float>((double) train->n_rows * (double) state->train_config.n_ctx
                            / epoch_seconds)
                    : 0.0f;
            out_metrics->learning_rate = state->last_learning_rate;
            completed_epochs = epoch + 1;
            if (progress_callback) {
                if (!progress_callback(epoch + 1, out_metrics, progress_user_data)) {
                    break;
                }
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        out_metrics->train_loss = static_cast<float>(train_loss);
        out_metrics->eval_loss = eval ? static_cast<float>(eval_loss) : NAN;
        out_metrics->tokens_per_second = seconds > 0.0
                ? static_cast<float>(
                        (double) train->n_rows * (double) state->train_config.n_ctx
                        * (double) completed_epochs / seconds)
                : 0.0f;
        out_metrics->global_step = state->scheduler_step;
        out_metrics->learning_rate = state->last_learning_rate;

        return 0;
    });
}

int train_weighted_impl(
        retro_trainer * trainer,
        const retro_weighted_dataset * data,
        uint64_t scheduler_total_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora && !trains_base_weights(*state)) {
            set_error("create or load a LoRA adapter before training");
            return -1;
        }
        if (!data || !out_metrics) {
            set_error("weighted dataset and out_metrics are required");
            return -1;
        }
        if (!data->tokens || !data->labels || !data->weights || data->n_rows == 0) {
            set_error("weighted dataset must contain tokens, labels, weights, and rows");
            return -1;
        }
        const size_t row_width = data->n_ctx;
        if (row_width == 0 || data->n_rows > SIZE_MAX / row_width) {
            set_error("invalid weighted dataset shape");
            return -1;
        }
        // retro delta (plan DISTILL D6.5): k targets per position.
        const uint32_t n_topk = data->n_topk == 0 ? 1u : data->n_topk;
        if (n_topk > RETRO_FUSED_CE_K_MAX) {
            set_error("weighted dataset n_topk exceeds RETRO_FUSED_CE_K_MAX");
            return -1;
        }
        const size_t n_values = data->n_rows * row_width;
        if (n_values > SIZE_MAX / n_topk) {
            set_error("invalid weighted dataset shape");
            return -1;
        }
        const size_t n_entries = n_values * n_topk;
        for (size_t i = 0; i < n_entries; ++i) {
            if (!std::isfinite(data->weights[i])) {
                set_error("weighted dataset weights must be finite");
                return -1;
            }
        }

        // Re-run the context-width validation of allocate_sft_dataset on every
        // call, so a cache hit cannot bypass it if the context is ever
        // recreated with a different width.
        const uint32_t live_ctx = static_cast<uint32_t>(llama_n_ctx(state->ctx.get()));
        if (!state->weighted_dataset_cache
                || state->weighted_cache_rows != data->n_rows
                || state->weighted_cache_ctx != data->n_ctx
                || data->n_ctx != live_ctx) {
            state->weighted_dataset_cache = allocate_sft_dataset(
                    state->ctx.get(), data->n_rows, data->n_ctx);
            if (!state->weighted_dataset_cache) {
                state->weighted_cache_rows = 0;
                state->weighted_cache_ctx = 0;
                return -1;
            }
            state->weighted_cache_rows = data->n_rows;
            state->weighted_cache_ctx = data->n_ctx;
        }
        ggml_opt_dataset * dataset = state->weighted_dataset_cache.get();
        std::memcpy(ggml_opt_dataset_data(dataset)->data,
                data->tokens, n_values * sizeof(llama_token));
        // retro delta (plan DISTILL D6.5): the dataset holds one label per
        // position by construction, so with k targets it takes the first
        // column - the producer writes the teacher's argmax there. It is what
        // the callback-facing dataset shows; the objective reads the k entries
        // below.
        if (n_topk == 1) {
            std::memcpy(ggml_opt_dataset_labels(dataset)->data,
                    data->labels, n_values * sizeof(llama_token));
        } else {
            int32_t * row0 = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
            for (size_t i = 0; i < n_values; ++i) {
                row0[i] = data->labels[i*n_topk];
            }
        }
        const llama_opt_topk_labels topk_labels = {
            reinterpret_cast<const llama_token *>(data->labels),
            data->weights,
            n_topk,
        };

        // PPO drives many short weighted epochs through one trainer, so the
        // scheduler step accumulates across calls instead of resetting.
        // Rows are trained only up to their last active label (the weighted
        // epoch skips trailing padded ubatches), so full-width steps_per_row
        // is an upper bound used solely as the fallback schedule horizon.
        const uint64_t steps_per_row = static_cast<uint64_t>(state->train_config.n_ctx) /
                state->train_config.n_batch;
        const uint64_t steps_this_call = static_cast<uint64_t>(data->n_rows) * steps_per_row;
        state->scheduler_total_steps = scheduler_total_steps != 0
                ? scheduler_total_steps
                : state->scheduler_step + steps_this_call;
        if (state->train_config.warmup_steps > state->scheduler_total_steps) {
            set_error("warmup_steps exceeds the total number of optimizer steps");
            return -1;
        }
        if (!state->opt_initialized) {
            state->last_learning_rate = state->train_config.learning_rate;
        }
        if (!ensure_optimizer_initialized(*state)) {
            return -1;
        }

        *out_metrics = retro_train_metrics {};
        if (!state->weighted_train_result_cache) {
            state->weighted_train_result_cache.reset(ggml_opt_result_init());
        }
        if (!state->weighted_eval_result_cache) {
            state->weighted_eval_result_cache.reset(ggml_opt_result_init());
        }
        if (!state->weighted_train_result_cache || !state->weighted_eval_result_cache) {
            set_error("failed to allocate training metrics result");
            return -1;
        }
        ggml_opt_result_reset(state->weighted_train_result_cache.get());
        ggml_opt_result_reset(state->weighted_eval_result_cache.get());
        ggml_opt_result * result_train = state->weighted_train_result_cache.get();
        ggml_opt_result * result_eval = state->weighted_eval_result_cache.get();

        // Mirror the weighted epoch's early exit for the throughput metric:
        // each row evaluates only up to its last active label, rounded up to
        // a physical ubatch, and the total is padded back to an accumulation-
        // period boundary so every optimizer step closes inside the call.
        uint64_t evaluated_tokens = 0;
        {
            const uint32_t n_ctx_train = std::min<uint32_t>(
                    state->train_config.n_ctx, static_cast<uint32_t>(row_width));
            const uint32_t n_batch_train = std::min(state->train_config.n_batch, n_ctx_train);
            const uint32_t n_ubatch_train =
                    std::min(state->train_config.n_ubatch, n_batch_train);
            const uint32_t evals_cap  = n_ctx_train / n_ubatch_train;
            const uint32_t opt_period = n_batch_train / n_ubatch_train;
            uint64_t total_evals = 0;
            for (size_t i = 0; i < data->n_rows; ++i) {
                const int32_t * row_labels  = data->labels  + i*row_width*n_topk;
                const float   * row_weights = data->weights + i*row_width*n_topk;
                int64_t last = -1;
                for (uint32_t j = 0; j < n_ctx_train; ++j) {
                    // retro delta (plan DISTILL D6.5): a position is active when
                    // any of its k entries is, mirroring the runtime's own
                    // predicate; reading only the first column would cut a row
                    // short wherever the argmax entry happens to be masked.
                    bool active = false;
                    for (uint32_t e = 0; e < n_topk && !active; ++e) {
                        active = row_labels[j*n_topk + e] >= 0
                                && row_weights[j*n_topk + e] != 0.0f;
                    }
                    if (active) {
                        last = j;
                    }
                }
                const uint32_t evals = last < 0
                        ? 1
                        : static_cast<uint32_t>(last)/n_ubatch_train + 1;
                total_evals += std::min(evals, evals_cap);
            }
            total_evals += (opt_period - total_evals % opt_period) % opt_period;
            evaluated_tokens = total_evals * n_ubatch_train;
        }

        // The internal optimizer callback is what the duty-cycle limiter
        // accounts on, so it is installed for throttling as well as for
        // progress: without this the limiter would report active and never
        // sleep during a run that asked for no progress rows.
        const bool throttling = state->duty_cycle.enabled();
        const bool needs_optimizer_callback = progress_callback || throttling;

        const auto t0 = std::chrono::steady_clock::now();
        step_progress step_events {
            state,
            1,
            state->scheduler_step,
            progress_callback,
            progress_user_data,
        };
        g_step_progress = &step_events;
        state->duty_cycle.begin_window();
        llama_opt_epoch_weighted(
                state->ctx.get(),
                dataset,
                result_train,
                result_eval,
                static_cast<int64_t>(data->n_rows),
                needs_optimizer_callback ? on_optimizer_step : nullptr,
                throttling ? on_optimizer_eval : nullptr,
                data->weights,
                &topk_labels);
        g_step_progress = nullptr;

        double train_loss = NAN;
        double unc = 0.0;
        ggml_opt_result_loss(result_train, &train_loss, &unc);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        out_metrics->epoch = 1;
        out_metrics->epoch_complete = true;
        out_metrics->global_step = state->scheduler_step;
        out_metrics->train_loss = static_cast<float>(train_loss);
        out_metrics->eval_loss = NAN;
        out_metrics->tokens_per_second = seconds > 0.0
                ? static_cast<float>((double) evaluated_tokens / seconds)
                : 0.0f;
        out_metrics->learning_rate = state->last_learning_rate;

        return 0;
    });
}

int train_packed_sequences_impl(
        retro_trainer * trainer,
        const retro_packed_sequence_batch * data,
        uint64_t scheduler_total_steps,
        uint32_t accumulation_steps,
        retro_train_metrics * out_metrics,
        retro_train_progress_callback progress_callback,
        void * progress_user_data) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) {
            return -1;
        }
        if (!state->has_lora && !trains_base_weights(*state)) {
            set_error("create or load a LoRA adapter before training");
            return -1;
        }
        if (!data || !out_metrics || accumulation_steps == 0 || !data->tokens || !data->labels ||
                !data->weights || !data->positions || !data->seq_offsets ||
                !data->seq_ids) {
            set_error("packed-sequence batch and out_metrics are required");
            return -1;
        }
        if (data->n_tokens != state->train_config.n_ubatch ||
                data->n_seq_ids < data->n_tokens ||
                data->n_sequences == 0 ||
                data->n_sequences > state->train_config.n_seq_max) {
            set_error("invalid packed-sequence optimizer geometry");
            return -1;
        }
        if (data->seq_offsets[0] != 0 ||
                data->seq_offsets[data->n_tokens] != data->n_seq_ids) {
            set_error("invalid packed-sequence CSR offsets");
            return -1;
        }
        // retro delta (plan DISTILL D6.5): k targets per position.
        const uint32_t n_topk = data->n_topk == 0 ? 1u : data->n_topk;
        if (n_topk > RETRO_FUSED_CE_K_MAX) {
            set_error("packed-sequence n_topk exceeds RETRO_FUSED_CE_K_MAX");
            return -1;
        }
        for (size_t i = 0; i < data->n_tokens; ++i) {
            if (data->positions[i] < 0 ||
                    data->seq_offsets[i] >= data->seq_offsets[i + 1]) {
                set_error("invalid packed-sequence weight, position, or CSR row");
                return -1;
            }
            for (uint32_t j = 0; j < n_topk; ++j) {
                if (!std::isfinite(data->weights[i*n_topk + j])) {
                    set_error("invalid packed-sequence weight, position, or CSR row");
                    return -1;
                }
            }
        }
        for (size_t i = 0; i < data->n_seq_ids; ++i) {
            if (data->seq_ids[i] < 0 ||
                    static_cast<uint32_t>(data->seq_ids[i]) >= data->n_sequences) {
                set_error("invalid packed-sequence id");
                return -1;
            }
        }

        // The callback API expects a dataset handle. Keep a one-row mirror of
        // the packed graph; the graph itself consumes the explicit sequence
        // metadata fields.
        const uint32_t dataset_ctx = llama_n_ctx(state->ctx.get());
        if (!state->weighted_dataset_cache
                || state->weighted_cache_rows != 1
                || state->weighted_cache_ctx != dataset_ctx) {
            state->weighted_dataset_cache = allocate_sft_dataset(
                    state->ctx.get(), 1, dataset_ctx);
            if (!state->weighted_dataset_cache) {
                return -1;
            }
            state->weighted_cache_rows = 1;
            state->weighted_cache_ctx = dataset_ctx;
        }
        ggml_opt_dataset * dataset = state->weighted_dataset_cache.get();
        std::memcpy(ggml_opt_dataset_data(dataset)->data,
                data->tokens, data->n_tokens*sizeof(llama_token));
        // retro delta (plan DISTILL D6.5): first column, same reason as the
        // weighted path - the dataset mirror holds one label per position.
        std::vector<int32_t> labels_first;
        const int32_t * labels_scalar = data->labels;
        if (n_topk > 1) {
            labels_first.resize(data->n_tokens);
            for (size_t i = 0; i < data->n_tokens; ++i) {
                labels_first[i] = data->labels[i*n_topk];
            }
            labels_scalar = labels_first.data();
        }
        std::memcpy(ggml_opt_dataset_labels(dataset)->data,
                labels_scalar, data->n_tokens*sizeof(llama_token));
        const llama_opt_topk_labels topk_labels = {
            reinterpret_cast<const llama_token *>(data->labels),
            data->weights,
            n_topk,
        };

        state->scheduler_total_steps = scheduler_total_steps != 0
                ? scheduler_total_steps : state->scheduler_step + 1;
        if (state->train_config.warmup_steps > state->scheduler_total_steps) {
            set_error("warmup_steps exceeds the total number of optimizer steps");
            return -1;
        }
        if (!state->opt_initialized) {
            state->last_learning_rate = state->train_config.learning_rate;
        }
        if (!ensure_optimizer_initialized(*state)) {
            return -1;
        }

        if (!state->weighted_train_result_cache) {
            state->weighted_train_result_cache.reset(ggml_opt_result_init());
        }
        if (!state->weighted_train_result_cache) {
            set_error("failed to allocate training metrics result");
            return -1;
        }
        ggml_opt_result_reset(state->weighted_train_result_cache.get());
        ggml_opt_result * result = state->weighted_train_result_cache.get();
        *out_metrics = retro_train_metrics {};

        // Same reason as the SFT and weighted sites: the limiter accounts on
        // this callback, so throttling installs it even without progress rows.
        // There is no eval counterpart here - the packed step has a single
        // callback and no evaluation loop.
        const bool needs_optimizer_callback =
                progress_callback || state->duty_cycle.enabled();

        const auto t0 = std::chrono::steady_clock::now();
        step_progress step_events {
            state, 1, state->scheduler_step, progress_callback, progress_user_data,
        };
        g_step_progress = &step_events;
        state->duty_cycle.begin_window();
        const bool ok = llama_opt_step_packed_sequences(
                state->ctx.get(), dataset, result,
                reinterpret_cast<const llama_token *>(data->tokens),
                reinterpret_cast<const llama_token *>(labels_scalar),
                data->weights,
                &topk_labels,
                reinterpret_cast<const llama_pos *>(data->positions),
                data->seq_offsets,
                reinterpret_cast<const llama_seq_id *>(data->seq_ids),
                static_cast<uint32_t>(data->n_tokens),
                data->n_seq_ids,
                data->n_sequences,
                accumulation_steps,
                needs_optimizer_callback ? on_optimizer_step : nullptr);
        g_step_progress = nullptr;
        if (!ok) {
            set_error("failed to build the packed-sequence optimizer graph");
            return -1;
        }

        double train_loss = NAN;
        double unc = 0.0;
        ggml_opt_result_loss(result, &train_loss, &unc);
        const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        out_metrics->epoch = 1;
        out_metrics->epoch_complete = true;
        out_metrics->global_step = state->scheduler_step;
        out_metrics->train_loss = static_cast<float>(train_loss);
        out_metrics->eval_loss = NAN;
        out_metrics->tokens_per_second = seconds > 0.0
                ? static_cast<float>(data->n_tokens / seconds) : 0.0f;
        out_metrics->learning_rate = state->last_learning_rate;
        return 0;
    });
}

int train_tokens_impl(
        retro_trainer * trainer,
        const int32_t * tokens,
        size_t n_tokens,
        retro_train_metrics * out_metrics) {
    return boundary([&]() -> int {
        trainer_state * state = checked(trainer);
        if (!state) return -1;
        const uint32_t n_ctx = llama_n_ctx(state->ctx.get());
        if (!tokens || n_tokens <= n_ctx) {
            set_error("training requires more tokens than the effective context");
            return -1;
        }
        const size_t stride = std::max<size_t>(1, n_ctx / 2);
        const size_t rows = 1 + (n_tokens - n_ctx - 1) / stride;
        std::vector<int32_t> data(rows * n_ctx);
        std::vector<int32_t> labels(rows * n_ctx);
        for (size_t row = 0; row < rows; ++row) {
            const size_t offset = row * stride;
            std::memcpy(data.data() + row*n_ctx, tokens + offset, n_ctx*sizeof(int32_t));
            std::memcpy(labels.data() + row*n_ctx, tokens + offset + 1, n_ctx*sizeof(int32_t));
        }
        // Preserve the legacy API's implicit final-row validation split. The
        // structured SFT API deliberately does not do this: it only evaluates
        // when the caller supplies an explicit dataset.
        const size_t train_rows = rows > 1 ? rows - 1 : rows;
        const retro_sft_dataset train { data.data(), labels.data(), train_rows, n_ctx };
        const retro_sft_dataset eval {
            data.data() + train_rows*n_ctx,
            labels.data() + train_rows*n_ctx,
            rows - train_rows,
            n_ctx,
        };
        const int code = train_sft_impl(
                trainer, &train, rows > 1 ? &eval : nullptr, out_metrics, nullptr, nullptr);
        if (code == 0 && rows == 1) {
            out_metrics->eval_loss = out_metrics->train_loss;
        }
        return code;
    });
}

} // namespace retro
