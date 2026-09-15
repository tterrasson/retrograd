// Probe for chunked cross-entropy parity. It compares the full-vocabulary ggml
// path with the fused/tiled operators for the same inputs and checks loss and
// hidden-state gradients. The fused path streams the vocabulary in `n_tiles`
// tiles instead of materializing the full logits matrix.
// The projection head `w` may be quantized (w_type != F32), matching the real
// training setup (e.g. Q8_0 lm_head). In that case the full path runs on the
// dequantized head so it stays exact, while the fused path consumes the quantized
// head and dequantizes each row on demand inside the operator - so parity proves
// the operator's on-the-fly dequantization.

#include "retro_runtime.hpp"

#include <algorithm>
#include <vector>

namespace retro {

namespace {

// w_type wire values kept independent of ggml's enum so the Rust side needs no
// ggml constants: 0 = F32 (no quantization), 1 = Q8_0, 2 = Q6_K, 3 = Q4_0,
// 4 = Q4_K, 5 = Q5_K, 6 = F16. Any supported ggml quant works: the fused op
// dequantizes generically, hence the GGML_BASE escape hatch for callers that
// sweep retro_dequant_types() instead of naming one type.
ggml_type fused_ce_probe_w_type(int32_t w_type) {
    if (w_type >= RETRO_FUSED_CE_W_TYPE_GGML_BASE) {
        const int32_t id = w_type - RETRO_FUSED_CE_W_TYPE_GGML_BASE;
        return id >= 0 && id < GGML_TYPE_COUNT ? (ggml_type) id : GGML_TYPE_COUNT;
    }
    switch (w_type) {
        case 0:  return GGML_TYPE_F32;
        case 1:  return GGML_TYPE_Q8_0;
        case 2:  return GGML_TYPE_Q6_K;
        case 3:  return GGML_TYPE_Q4_0;
        case 4:  return GGML_TYPE_Q4_K;
        case 5:  return GGML_TYPE_Q5_K;
        case 6:  return GGML_TYPE_F16;
        default: return GGML_TYPE_COUNT;
    }
}

} // namespace

int fused_sparse_ce_probe_impl(
        int32_t         n_embd,
        int32_t         n_tokens,
        int32_t         n_vocab,
        int32_t         n_topk, // retro delta (DISTILL D6.5)
        int32_t         n_tiles,
        int32_t         seq_chunk,
        int32_t         offload_h,
        int32_t         w_type,
        int32_t         use_gpu,
        const float   * h,
        const float   * w,
        const int32_t * targets,
        const float   * weights,
        const float   * bias, // [n_vocab] F32, or NULL for no bias
        float           grad_loss,
        float         * out_loss_full,
        float         * out_loss_fused,
        float         * out_grad_h_full,
        float         * out_grad_h_fused) {
    return boundary([&]() -> int {
        ensure_backend_initialized();
        if (n_embd <= 0 || n_tokens <= 0 || n_vocab <= 0 || n_tiles <= 0 || seq_chunk < 0 ||
                !h || !w || !targets || !weights ||
                !out_loss_full || !out_loss_fused ||
                !out_grad_h_full || !out_grad_h_fused) {
            set_error("fused CE probe requires positive dims and non-null buffers");
            return -1;
        }

        const ggml_type wt = fused_ce_probe_w_type(w_type);
        if (wt == GGML_TYPE_COUNT) {
            set_error("fused CE probe: unsupported w_type");
            return -1;
        }
        if (wt != GGML_TYPE_F32 && (n_embd % (int32_t) ggml_blck_size(wt)) != 0) {
            set_error("fused CE probe: n_embd must be a multiple of the quant block size");
            return -1;
        }

        // retro delta (plan DISTILL D6.5): k entries per position, [K, n_tokens].
        const int32_t topk = n_topk <= 0 ? 1 : n_topk;
        if (topk > RETRO_FUSED_CE_K_MAX) {
            set_error("fused CE probe: n_topk exceeds RETRO_FUSED_CE_K_MAX");
            return -1;
        }
        for (int32_t i = 0; i < n_tokens*topk; ++i) {
            if (targets[i] >= n_vocab) {
                set_error("fused CE probe: target token index out of range");
                return -1;
            }
        }

        // Projection head for the FULL path is always F32: when the requested head
        // is quantized, quantize then dequantize so the full path stays exact and
        // uses the same values the fused operator will reconstruct on the fly.
        std::vector<uint8_t> w_quant;
        std::vector<float>   w_full((size_t) n_embd * n_vocab);
        if (wt == GGML_TYPE_F32) {
            std::copy(w, w + (size_t) n_embd * n_vocab, w_full.begin());
        } else if (wt == GGML_TYPE_F16) {
            w_quant.resize(sizeof(ggml_fp16_t) * (size_t) n_embd * n_vocab);
            ggml_fp32_to_fp16_row(w, (ggml_fp16_t *) w_quant.data(), (int64_t) n_embd*n_vocab);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) w_quant.data(), w_full.data(),
                    (int64_t) n_embd*n_vocab);
        } else {
            w_quant.resize(ggml_row_size(wt, n_embd) * (size_t) n_vocab);
            // A few IQ types assert on a null importance matrix. Uniform importance
            // is a deterministic stand-in and cannot skew parity: the full path is
            // built by dequantizing these very bytes.
            std::vector<float> imatrix;
            if (ggml_quantize_requires_imatrix(wt)) {
                imatrix.assign((size_t) n_embd, 1.0f);
            }
            ggml_quantize_chunk(wt, w, w_quant.data(), 0, n_vocab, n_embd,
                    imatrix.empty() ? nullptr : imatrix.data());
            ggml_to_float_t const to_float = ggml_get_type_traits(wt)->to_float;
            for (int64_t v = 0; v < n_vocab; ++v) {
                to_float(w_quant.data() + v*ggml_row_size(wt, n_embd),
                        w_full.data() + v*n_embd, n_embd);
            }
        }

        ggml_backend_dev_t dev =
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            set_error("no CPU device available for fused CE probe");
            return -1;
        }

        // The full-vocab path is always the CPU oracle; the fused path runs on
        // the requested device (CPU, or the registered GPU when use_gpu != 0), so
        // a GPU run proves the device kernel matches the CPU reference.
        ggml_backend_dev_t fused_dev = dev;
        if (use_gpu) {
            // first_gpu_device() rather than dev_by_type(GPU): a unified-memory
            // device registers as GGML_BACKEND_DEVICE_TYPE_IGPU, which is what
            // Vulkan-on-Apple reports. Asking for GPU alone silently found nothing
            // there, so the whole fused-CE lane errored out instead of covering the
            // Vulkan kernels. Same helper the capability probes use.
            fused_dev = first_gpu_device();
            if (!fused_dev) {
                set_error("fused CE probe: use_gpu set but no GPU device is registered");
                return -1;
            }
        }

        // ---- Full-vocab path: the current training tail, real fork ggml ops ----
        // logits = mul_mat(w, h); loss = cross_entropy_loss(logits, labels);
        // grad_logits = cross_entropy_loss_back(...); grad_h = mul_mat(w^T, grad_logits).
        {
            ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
            if (!backend) {
                set_error("failed to initialize CPU backend for fused CE probe");
                return -1;
            }
            const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096;
            ggml_init_params params { mem, nullptr, true };
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                    ggml_init(params), &ggml_free);
            if (!ctx) {
                set_error("fused CE probe context allocation failed");
                return -1;
            }

            ggml_tensor * h_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
            ggml_tensor * w_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_vocab);
            ggml_tensor * labels = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_vocab, n_tokens);
            ggml_tensor * grad = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
            ggml_tensor * bias_t = bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab) : nullptr;
            if (!h_t || !w_t || !labels || !grad || (bias && !bias_t)) {
                set_error("fused CE probe tensor allocation failed");
                return -1;
            }

            ggml_tensor * logits = ggml_mul_mat(ctx.get(), w_t, h_t);
            if (bias_t) {
                logits = ggml_add(ctx.get(), logits, bias_t);
            }
            ggml_tensor * loss = ggml_cross_entropy_loss(ctx.get(), logits, labels);
            ggml_tensor * grad_logits =
                    ggml_cross_entropy_loss_back(ctx.get(), grad, logits, labels);
            ggml_tensor * w_tr = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), w_t));
            ggml_tensor * grad_h = ggml_mul_mat(ctx.get(), w_tr, grad_logits);
            if (!logits || !loss || !grad_logits || !w_tr || !grad_h) {
                set_error("fused CE probe full-path graph construction failed");
                return -1;
            }

            // A position counts once however many entries it carries: the mean
            // is over positions, and the fused operator normalizes the same way.
            int32_t n_active = 0;
            for (int32_t t = 0; t < n_tokens; ++t) {
                for (int32_t j = 0; j < topk; ++j) {
                    const int32_t v = targets[t*topk + j];
                    if (v >= 0 && v < n_vocab && weights[t*topk + j] != 0.0f) {
                        ++n_active;
                        break;
                    }
                }
            }
            if (n_active > 0) {
                loss->op_params[0] = n_active;
                loss->op_params[1] = 1;
                grad_logits->op_params[0] = n_active;
                grad_logits->op_params[1] = 1;
            }

            ggml_backend_buffer_ptr buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error("fused CE probe backend buffer allocation failed");
                return -1;
            }

            ggml_backend_tensor_set(h_t, h, 0, ggml_nbytes(h_t));
            ggml_backend_tensor_set(w_t, w_full.data(), 0, ggml_nbytes(w_t));
            ggml_backend_tensor_set(grad, &grad_loss, 0, sizeof(float));
            if (bias_t) {
                ggml_backend_tensor_set(bias_t, bias, 0, ggml_nbytes(bias_t));
            }

            // The dense label row *is* a distribution - writing k entries into it
            // is the whole of the offline-KD forward on this path, and is what
            // makes it an oracle the fused operator has to match at K > 1.
            std::vector<float> dense_labels((size_t) n_vocab * n_tokens, 0.0f);
            for (int32_t t = 0; t < n_tokens; ++t) {
                for (int32_t j = 0; j < topk; ++j) {
                    const int32_t v = targets[t*topk + j];
                    const float   c = weights[t*topk + j];
                    if (v >= 0 && v < n_vocab && c != 0.0f) {
                        dense_labels[(size_t) t * n_vocab + v] = c;
                    }
                }
            }
            ggml_backend_tensor_set(labels, dense_labels.data(), 0, ggml_nbytes(labels));

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, loss);
            ggml_build_forward_expand(gf, grad_h);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("fused CE probe full-path graph compute failed");
                return -1;
            }

            ggml_backend_tensor_get(loss, out_loss_full, 0, sizeof(float));
            ggml_backend_tensor_get(grad_h, out_grad_h_full, 0,
                    (size_t) n_embd * n_tokens * sizeof(float));
        }

        // ---- Fused / tiled path: the real ggml operators ----
        {
            ggml_backend_ptr backend(ggml_backend_dev_init(fused_dev, nullptr));
            if (!backend) {
                set_error("failed to initialize backend for fused CE probe");
                return -1;
            }
            const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096;
            ggml_init_params params { mem, nullptr, true };
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                    ggml_init(params), &ggml_free);
            if (!ctx) {
                set_error("fused CE probe context allocation failed");
                return -1;
            }

            ggml_tensor * h_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
            ggml_tensor * w_t = ggml_new_tensor_2d(ctx.get(), wt, n_embd, n_vocab);
            ggml_tensor * tgt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, topk, n_tokens);
            ggml_tensor * wgt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, topk, n_tokens);
            ggml_tensor * grad = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
            ggml_tensor * bias_t = bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab) : nullptr;
            if (!h_t || !w_t || !tgt || !wgt || !grad || (bias && !bias_t)) {
                set_error("fused CE probe tensor allocation failed");
                return -1;
            }

            ggml_tensor * loss = ggml_fused_sparse_ce(
                    ctx.get(), h_t, w_t, tgt, wgt, bias_t, n_tiles, seq_chunk, offload_h);
            ggml_tensor * grad_h = ggml_fused_sparse_ce_back(
                    ctx.get(), grad, h_t, w_t, tgt, wgt, bias_t, n_tiles, seq_chunk, offload_h);
            if (!loss || !grad_h) {
                set_error("fused CE probe fused-path graph construction failed");
                return -1;
            }

            ggml_backend_buffer_ptr buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error("fused CE probe backend buffer allocation failed");
                return -1;
            }

            // In the training graph, `offload_h` lets ggml-alloc reuse `h` for
            // grad_h. This probe allocates tensors statically, so reproduce that
            // placement by hand; otherwise the flag would not exercise aliasing.
            // Safe because the graph is built loss-first: the forward reads `h`
            // before the backward writes over it. grad_h's own allocation is simply
            // abandoned inside the buffer.
            if (offload_h) {
                grad_h->data = h_t->data;
            }

            ggml_backend_tensor_set(h_t, h, 0, ggml_nbytes(h_t));
            if (wt == GGML_TYPE_F32) {
                ggml_backend_tensor_set(w_t, w_full.data(), 0, ggml_nbytes(w_t));
            } else {
                ggml_backend_tensor_set(w_t, w_quant.data(), 0, ggml_nbytes(w_t));
            }
            ggml_backend_tensor_set(tgt, targets, 0, ggml_nbytes(tgt));
            ggml_backend_tensor_set(wgt, weights, 0, ggml_nbytes(wgt));
            ggml_backend_tensor_set(grad, &grad_loss, 0, sizeof(float));
            if (bias_t) {
                ggml_backend_tensor_set(bias_t, bias, 0, ggml_nbytes(bias_t));
            }

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, loss);
            ggml_build_forward_expand(gf, grad_h);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("fused CE probe fused-path graph compute failed");
                return -1;
            }

            ggml_backend_tensor_get(loss, out_loss_fused, 0, sizeof(float));
            ggml_backend_tensor_get(grad_h, out_grad_h_fused, 0,
                    (size_t) n_embd * n_tokens * sizeof(float));
        }

        return 0;
    });
}

} // namespace retro
