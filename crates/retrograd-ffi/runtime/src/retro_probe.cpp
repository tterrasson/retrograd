#include "retro_runtime.hpp"

// retro delta: RIR policy, counters and last-dispatch decision.
#include "ggml-rir/ggml-rir.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>

namespace retro {

namespace {
// Probes create and destroy a backend for each call. Vulkan destroys its device
// with the last context, so concurrent probes can use a destroyed device. A
// process-wide lock covers the full create/run/destroy window and the associated
// process-wide dispatch state.
std::mutex & probe_device_mutex() {
    static std::mutex m;
    return m;
}
} // namespace

int gpu_runtime_probe_impl() {
    return boundary([&]() -> int {
        std::lock_guard<std::mutex> serialize(probe_device_mutex());
        ensure_backend_initialized();
        ggml_backend_dev_t dev = first_gpu_device();
        if (!dev) {
            set_error("no GPU device available");
            return -1;
        }

        // Device initialization is where Metal creates its command queue and
        // Vulkan creates its device/queue. Synchronizing makes this a runtime
        // availability probe, not merely a registry lookup.
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            set_error("failed to initialize GPU backend");
            return -1;
        }
        ggml_backend_synchronize(backend.get());
        return 0;
    });
}

// The probe body, with the device lock already held. `probe_op_run_impl` takes
// it; `probe_op_run_ex_impl` takes it earlier, because its decision record has
// to cover the same window.
static int probe_op_run_locked(
        int32_t op,
        int32_t use_gpu,
        const int64_t * ne_src0,
        const float * src0,
        const int64_t * ne_src1,
        const float * src1,
        const int64_t * ne_src2,
        const float * src2,
        float param0,
        float param1,
        float * dst,
        size_t dst_len) {
    return boundary([&]() -> int {
        ensure_backend_initialized();
        if (!ne_src0 || !src0 || !ne_src1 || !src1 || !dst) {
            set_error("probe requires non-null shapes and buffers");
            return -1;
        }

        // Ops that consume a third tensor. GET_ROWS_BACK only uses src2 for its
        // shape; CROSS_ENTROPY_LOSS_BACK reads its data (labels).
        const bool needs_src2 = op == RETRO_PROBE_OP_CROSS_ENTROPY_LOSS_BACK ||
                                op == RETRO_PROBE_OP_GET_ROWS_BACK ||
                                op == RETRO_PROBE_OP_SSM_CONV_BACK;
        if (needs_src2 && (!ne_src2 || (op != RETRO_PROBE_OP_GET_ROWS_BACK && !src2))) {
            set_error("probe op requires a third input (ne_src2/src2)");
            return -1;
        }

        ggml_backend_dev_t dev = use_gpu
                ? first_gpu_device()
                : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            set_error(use_gpu ? "no GPU device available for probe"
                              : "no CPU device available for probe");
            return -1;
        }
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            set_error("failed to initialize backend for probe");
            return -1;
        }

        const size_t mem = ggml_tensor_overhead() * 24 + ggml_graph_overhead() + 1024;
        ggml_init_params params {
            /*.mem_size   =*/ mem,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
        if (!ctx) {
            set_error("probe context allocation failed");
            return -1;
        }

        if (op == RETRO_PROBE_OP_FLASH_ATTN_BACK) {
            const int64_t hsk    = ne_src0[0];
            const int64_t hsv    = ne_src0[1];
            const int64_t nq     = ne_src0[2];
            const int64_t nkv    = ne_src0[3];
            const int64_t nhead  = ne_src1[0];
            const int64_t nheadk = ne_src1[1];
            const int64_t nbatch = ne_src1[2];
            const bool causal    = ne_src1[3] != 0;
            // Head-dim ceiling comes from ggml.h rather than a literal repeated
            // here: every backend cap (FA_BACK_MAX_D / VK_FA_BACK_MAX_D /
            // GGML_METAL_FA_BACK_MAX_D) static_asserts against the same
            // constant, so widening a kernel can no longer leave this harness
            // silently unable to exercise the new ceiling.
            const int64_t hs_cap = GGML_FLASH_ATTN_BACK_MAX_HEAD_DIM;
            if (hsk <= 0 || hsv <= 0 || hsk > hs_cap || hsv > hs_cap || nq <= 0 ||
                    nkv < nq || nhead <= 0 || nheadk <= 0 || nbatch <= 0 ||
                    nhead % nheadk != 0) {
                set_error("invalid packed FLASH_ATTN_BACK probe dimensions");
                return -1;
            }

            // Optional KV gradient window: ne_src2 = [nwin, 0, 0, 0], src2 holds
            // nwin*nbatch cache row indices (one per window row and batch, in
            // batch-major order). dK/dV then cover only the window rows. Without
            // it (nwin == 0) the dense case runs: dK/dV span the whole cache.
            const bool has_window = ne_src2 && ne_src2[0] > 0;
            const int64_t nwin = has_window ? ne_src2[0] : nkv;
            // ne_src2[1] != 0 stores K/V as F32 instead of F16. The capability
            // probe in retro_backend.cpp builds the op with F32 K/V (that is the
            // gate `kv_dtype = "f16"` has to clear first), so backends carrying
            // both element types need parity coverage for both. No src2 data is
            // read for this, so `[0, 1, 0, 0]` means "no window, F32 K/V".
            const bool kv_f32 = ne_src2 && ne_src2[1] != 0;
            const ggml_type kv_type = kv_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
            // ne_src2[2] != 0 attaches attention sinks: one extra logit per head
            // that joins the softmax denominator without contributing a V row.
            // Its `nhead` values are appended to the src0 pack after dO. Vulkan is
            // the only backend that advertises this path, so it is the only one a
            // probe can exercise.
            const bool has_sinks = ne_src2 && ne_src2[2] != 0;
            // Test-only selector for the native CPU backend. The normal CPU
            // probe remains the independent analytic oracle used by every GPU
            // parity test; ne_src2[3] lets the ABI lane exercise the real CPU
            // GGML_OP_FLASH_ATTN_BACK implementation against that oracle.
            const bool native_cpu = ne_src2 && ne_src2[3] != 0;
            std::vector<int32_t> kv_idxs;
            if (has_window) {
                if (nwin > nkv || !src2) {
                    set_error("invalid FLASH_ATTN_BACK window");
                    return -1;
                }
                kv_idxs.resize(static_cast<size_t>(nwin*nbatch));
                for (size_t i = 0; i < kv_idxs.size(); ++i) {
                    kv_idxs[i] = (int32_t) std::lround(src2[i]);
                }
            }
            // Production packs the window index as (row + kv_stride*stream); the
            // probe keeps one cache per batch, so stride == nkv and stream == ib.
            const int32_t kv_stride  = (int32_t) nkv;
            const int32_t kv_stream0 = 0;

            const size_t n_q  = static_cast<size_t>(hsk*nq*nhead*nbatch);
            const size_t n_k  = static_cast<size_t>(hsk*nkv*nheadk*nbatch);
            const size_t n_v  = static_cast<size_t>(hsv*nkv*nheadk*nbatch);
            const size_t n_do = static_cast<size_t>(hsv*nhead*nq*nbatch);
            const size_t n_gk = static_cast<size_t>(hsk*nwin*nheadk*nbatch);
            const size_t n_gv = static_cast<size_t>(hsv*nwin*nheadk*nbatch);
            const size_t n_g  = n_q + n_gk + n_gv;
            if (dst_len < n_g) {
                set_error("probe dst buffer is too small for FLASH_ATTN_BACK");
                return -1;
            }

            const float * q_data  = src0;
            const float * k_data  = q_data + n_q;
            const float * v_data  = k_data + n_k;
            const float * do_data = v_data + n_v;
            const float * s_data  = has_sinks ? do_data + n_do : nullptr;

            // With an F16 KV cache the production graph casts differentiable F32
            // K/V to F16 before Flash Attention, so the reference operands are
            // rounded identically. With an F32 cache there is no rounding at all
            // and the reference reads the raw inputs.
            std::vector<ggml_fp16_t> k16(n_k), v16(n_v);
            std::vector<float> kf(n_k), vf(n_v);
            if (kv_f32) {
                std::copy(k_data, k_data + n_k, kf.begin());
                std::copy(v_data, v_data + n_v, vf.begin());
            } else {
                ggml_fp32_to_fp16_row(k_data, k16.data(), n_k);
                ggml_fp32_to_fp16_row(v_data, v16.data(), n_v);
                ggml_fp16_to_fp32_row(k16.data(), kf.data(), n_k);
                ggml_fp16_to_fp32_row(v16.data(), vf.data(), n_v);
            }

            const size_t n_mask = static_cast<size_t>(nkv*nq*nbatch);
            std::vector<float> maskf(n_mask, 0.0f);
            const int64_t n_past = nkv - nq;
            if (causal) {
                for (int64_t ib = 0; ib < nbatch; ++ib) {
                    for (int64_t iq = 0; iq < nq; ++iq) {
                        for (int64_t ik = n_past + iq + 1; ik < nkv; ++ik) {
                            maskf[static_cast<size_t>((ib*nq + iq)*nkv + ik)] =
                                    -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            std::vector<ggml_fp16_t> mask16(n_mask);
            ggml_fp32_to_fp16_row(maskf.data(), mask16.data(), n_mask);

            std::vector<float> out_ref(n_do, 0.0f);
            std::vector<float> grad_ref(n_g, 0.0f);
            const int64_t ratio = nhead/nheadk;
            const float scale = param0;
            const float softcap = param1;

            // For each cache row, the window slot that carries its gradient
            // (-1 if none). Dense case: identity. This is the inverse of the
            // kv_idxs mapping the kernel applies.
            std::vector<std::vector<int64_t>> win_of(
                    static_cast<size_t>(nbatch), std::vector<int64_t>(static_cast<size_t>(nkv), -1));
            for (int64_t ib = 0; ib < nbatch; ++ib) {
                for (int64_t j = 0; j < nwin; ++j) {
                    int64_t ik = has_window
                            ? (int64_t) kv_idxs[static_cast<size_t>(ib*nwin + j)] - kv_stride*(kv_stream0 + ib)
                            : j;
                    if (ik >= 0 && ik < nkv) {
                        win_of[static_cast<size_t>(ib)][static_cast<size_t>(ik)] = j;
                    }
                }
            }

            for (int64_t ib = 0; ib < nbatch; ++ib) {
                for (int64_t ih = 0; ih < nhead; ++ih) {
                    const int64_t ikh = ih/ratio;
                    for (int64_t iq = 0; iq < nq; ++iq) {
                        const size_t qbase = static_cast<size_t>(((ib*nhead + ih)*nq + iq)*hsk);
                        const size_t obase = static_cast<size_t>(((ib*nq + iq)*nhead + ih)*hsv);
                        std::vector<float> scores(static_cast<size_t>(nkv));
                        float smax = -std::numeric_limits<float>::infinity();
                        for (int64_t ik = 0; ik < nkv; ++ik) {
                            const size_t kbase = static_cast<size_t>(((ib*nheadk + ikh)*nkv + ik)*hsk);
                            float dot = 0.0f;
                            for (int64_t id = 0; id < hsk; ++id) {
                                dot += q_data[qbase + id]*kf[kbase + id];
                            }
                            float score = softcap != 0.0f
                                    ? softcap*std::tanh(dot*scale/softcap)
                                    : dot*scale;
                            score += maskf[static_cast<size_t>((ib*nq + iq)*nkv + ik)];
                            scores[static_cast<size_t>(ik)] = score;
                            smax = std::max(smax, score);
                        }
                        // The sink logit raises the denominator (and the row max)
                        // but has no V row, so it only shrinks every probability.
                        // Its own gradient is not part of the requested mask.
                        if (has_sinks) {
                            smax = std::max(smax, s_data[static_cast<size_t>(ih)]);
                        }
                        float sum = 0.0f;
                        for (float & score : scores) {
                            score = std::exp(score - smax);
                            sum += score;
                        }
                        if (has_sinks) {
                            sum += std::exp(s_data[static_cast<size_t>(ih)] - smax);
                        }
                        for (float & probability : scores) {
                            probability /= sum;
                        }

                        for (int64_t ik = 0; ik < nkv; ++ik) {
                            const size_t vbase = static_cast<size_t>(((ib*nheadk + ikh)*nkv + ik)*hsv);
                            const float probability = scores[static_cast<size_t>(ik)];
                            for (int64_t id = 0; id < hsv; ++id) {
                                out_ref[obase + id] += probability*vf[vbase + id];
                            }
                        }

                        float delta = 0.0f;
                        for (int64_t id = 0; id < hsv; ++id) {
                            delta += do_data[obase + id]*out_ref[obase + id];
                        }
                        for (int64_t ik = 0; ik < nkv; ++ik) {
                            const size_t kbase = static_cast<size_t>(((ib*nheadk + ikh)*nkv + ik)*hsk);
                            const size_t vbase = static_cast<size_t>(((ib*nheadk + ikh)*nkv + ik)*hsv);
                            float dot_qk = 0.0f;
                            for (int64_t id = 0; id < hsk; ++id) {
                                dot_qk += q_data[qbase + id]*kf[kbase + id];
                            }
                            float dot_dv = 0.0f;
                            for (int64_t id = 0; id < hsv; ++id) {
                                dot_dv += do_data[obase + id]*vf[vbase + id];
                            }
                            float derivative = scale;
                            if (softcap != 0.0f) {
                                const float t = std::tanh(dot_qk*scale/softcap);
                                derivative *= 1.0f - t*t;
                            }
                            const float probability = scores[static_cast<size_t>(ik)];
                            const float ds = probability*(dot_dv - delta)*derivative;
                            // dQ spans the whole cache; dK/dV land in the window
                            // slot for this cache row (skipped if outside it).
                            for (int64_t id = 0; id < hsk; ++id) {
                                grad_ref[qbase + id] += ds*kf[kbase + id];
                            }
                            const int64_t jwin = win_of[static_cast<size_t>(ib)][static_cast<size_t>(ik)];
                            if (jwin < 0) {
                                continue;
                            }
                            const size_t gkbase = static_cast<size_t>(((ib*nheadk + ikh)*nwin + jwin)*hsk);
                            const size_t gvbase = static_cast<size_t>(((ib*nheadk + ikh)*nwin + jwin)*hsv);
                            for (int64_t id = 0; id < hsk; ++id) {
                                grad_ref[n_q + gkbase + id] += ds*q_data[qbase + id];
                            }
                            for (int64_t id = 0; id < hsv; ++id) {
                                grad_ref[n_q + n_gk + gvbase + id] += probability*do_data[obase + id];
                            }
                        }
                    }
                }
            }

            if (!use_gpu && !native_cpu) {
                memcpy(dst, grad_ref.data(), n_g*sizeof(float));
                return 0;
            }

            ggml_tensor * tq = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hsk, nq, nhead, nbatch);
            ggml_tensor * tk = ggml_new_tensor_4d(ctx.get(), kv_type, hsk, nkv, nheadk, nbatch);
            ggml_tensor * tv = ggml_new_tensor_4d(ctx.get(), kv_type, hsv, nkv, nheadk, nbatch);
            ggml_tensor * tm = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, nkv, nq, 1, nbatch);
            ggml_tensor * td = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hsv, nhead, nq, nbatch);
            ggml_tensor * to = ggml_flash_attn_ext(ctx.get(), tq, tk, tv, tm, scale, 0.0f, softcap);
            ggml_prec_set_acc(to, GGML_PREC_F32);
            ggml_tensor * ts = nullptr;
            if (has_sinks) {
                ts = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nhead);
                ggml_flash_attn_ext_add_sinks(to, ts);
            }

            // KV gradient window operands. The backward kernels never read the
            // k_cur/v_cur values (dK/dV are ds*q and prob*dO); the tensors only
            // supply the window shape and wiring, so their contents are left at
            // zero. kv_idxs carries the cache-row mapping the kernel applies.
            ggml_tensor * tkc = nullptr;
            ggml_tensor * tvc = nullptr;
            ggml_tensor * tix = nullptr;
            if (has_window) {
                tkc = ggml_new_tensor_4d(ctx.get(), kv_type, hsk, nwin, nheadk, nbatch);
                tvc = ggml_new_tensor_4d(ctx.get(), kv_type, hsv, nwin, nheadk, nbatch);
                tix = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, nwin*nbatch);
                ggml_flash_attn_ext_set_grad_window(to, tkc, tvc, tix, kv_stride, kv_stream0);
            }

            const int32_t grad_mask = GGML_FLASH_ATTN_BACK_GRAD_Q |
                    GGML_FLASH_ATTN_BACK_GRAD_K | GGML_FLASH_ATTN_BACK_GRAD_V;
            ggml_tensor * back = ggml_flash_attn_ext_back(
                    ctx.get(), tq, tk, tv, tm, to, td, ts,
                    tkc, tvc, tix, kv_stride, kv_stream0, grad_mask, scale, 0.0f, softcap);

            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error("FLASH_ATTN_BACK probe backend allocation failed");
                return -1;
            }
            ggml_backend_tensor_set(tq, q_data, 0, n_q*sizeof(float));
            if (kv_f32) {
                ggml_backend_tensor_set(tk, kf.data(), 0, n_k*sizeof(float));
                ggml_backend_tensor_set(tv, vf.data(), 0, n_v*sizeof(float));
            } else {
                ggml_backend_tensor_set(tk, k16.data(), 0, n_k*sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(tv, v16.data(), 0, n_v*sizeof(ggml_fp16_t));
            }
            ggml_backend_tensor_set(tm, mask16.data(), 0, n_mask*sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(td, do_data, 0, n_do*sizeof(float));
            if (has_sinks) {
                ggml_backend_tensor_set(ts, s_data, 0, nhead*sizeof(float));
            }
            if (has_window) {
                ggml_backend_tensor_memset(tkc, 0, 0, ggml_nbytes(tkc));
                ggml_backend_tensor_memset(tvc, 0, 0, ggml_nbytes(tvc));
                ggml_backend_tensor_set(tix, kv_idxs.data(), 0, kv_idxs.size()*sizeof(int32_t));
            }

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, back);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("FLASH_ATTN_BACK probe graph compute failed");
                return -1;
            }
            size_t off_k = 0;
            size_t off_v = 0;
            ggml_flash_attn_back_offsets(back, nullptr, &off_k, &off_v, nullptr);
            ggml_backend_tensor_get(back, dst, 0, n_q*sizeof(float));
            ggml_backend_tensor_get(back, dst + n_q, off_k, n_gk*sizeof(float));
            ggml_backend_tensor_get(back, dst + n_q + n_gk, off_v, n_gv*sizeof(float));
            return 0;
        }

        if (op == RETRO_PROBE_OP_CAST_STORE_F16 || op == RETRO_PROBE_OP_CAST_STORE_BF16) {
            // The F32 -> F16/BF16 cast alone on the device. Two answers come
            // back: what CPY wrote, and the reference row conversion of the
            // same input, so a truncating cast shows up here instead of as
            // drift in a long run.
            const bool bf16 = op == RETRO_PROBE_OP_CAST_STORE_BF16;
            const ggml_type store = bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
            const char * label = bf16 ? "CAST_STORE_BF16" : "CAST_STORE_F16";
            ggml_tensor * source = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                    ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
            ggml_tensor * stored = ggml_new_tensor_4d(ctx.get(), store,
                    ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
            if (!source || !stored) {
                set_error(std::string(label) + " probe tensor allocation failed");
                return -1;
            }
            ggml_tensor * out = ggml_cpy(ctx.get(), source, stored);
            ggml_backend_buffer_ptr buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error(std::string(label) + " probe backend allocation failed");
                return -1;
            }
            const size_t n = static_cast<size_t>(ggml_nelements(source));
            if (dst_len < 2*n) {
                set_error(std::string("probe dst buffer is too small for ") + label
                        + "; it carries the device result and the reference one");
                return -1;
            }
            ggml_backend_tensor_set(source, src0, 0, n*sizeof(float));
            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error(std::string(label) + " probe graph compute failed");
                return -1;
            }
            std::vector<uint16_t> bits(n);
            ggml_backend_tensor_get(stored, bits.data(), 0, n*sizeof(uint16_t));
            if (bf16) {
                ggml_bf16_to_fp32_row(
                        reinterpret_cast<const ggml_bf16_t *>(bits.data()), dst, n);
                ggml_fp32_to_bf16_row_ref(src0, reinterpret_cast<ggml_bf16_t *>(bits.data()), n);
                ggml_bf16_to_fp32_row(
                        reinterpret_cast<const ggml_bf16_t *>(bits.data()), dst + n, n);
            } else {
                ggml_fp16_to_fp32_row(
                        reinterpret_cast<const ggml_fp16_t *>(bits.data()), dst, n);
                ggml_fp32_to_fp16_row(src0, reinterpret_cast<ggml_fp16_t *>(bits.data()), n);
                ggml_fp16_to_fp32_row(
                        reinterpret_cast<const ggml_fp16_t *>(bits.data()), dst + n, n);
            }
            return 0;
        }

        if (op == RETRO_PROBE_OP_OPT_STEP_ADAMW_F16 ||
                op == RETRO_PROBE_OP_OPT_STEP_ADAMW_BF16 ||
                op == RETRO_PROBE_OP_OPT_STEP_SGD_F16 ||
                op == RETRO_PROBE_OP_OPT_STEP_SGD_BF16) {
            // One id per (optimizer, storage): the four are separate kernel
            // implementations, so a test that names the wrong one fails
            // instead of measuring another.
            const bool bf16 = op == RETRO_PROBE_OP_OPT_STEP_ADAMW_BF16
                    || op == RETRO_PROBE_OP_OPT_STEP_SGD_BF16;
            const bool sgd = op == RETRO_PROBE_OP_OPT_STEP_SGD_F16
                    || op == RETRO_PROBE_OP_OPT_STEP_SGD_BF16;
            const ggml_type wtype = bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
            const char * label = sgd
                    ? (bf16 ? "SGD BF16" : "SGD F16")
                    : (bf16 ? "AdamW BF16" : "AdamW F16");
            if (!ggml_are_same_shape(
                        ggml_new_tensor_4d(ctx.get(), wtype,
                            ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]),
                        ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                            ne_src1[0], ne_src1[1], ne_src1[2], ne_src1[3]))) {
                set_error(std::string(label)
                        + " probe requires matching weight and gradient shapes");
                return -1;
            }
            // Recreate named pointers after the shape-only validation.
            ggml_tensor * w = ggml_new_tensor_4d(ctx.get(), wtype,
                    ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
            ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                    ne_src1[0], ne_src1[1], ne_src1[2], ne_src1[3]);
            ggml_tensor * m = sgd ? nullptr : ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                    ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
            ggml_tensor * v = sgd ? nullptr : ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                    ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
            ggml_tensor * p = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, sgd ? 3 : 9);
            ggml_set_param(w);
            ggml_tensor * out = sgd
                    ? ggml_opt_step_sgd(ctx.get(), w, g, p)
                    : ggml_opt_step_adamw(ctx.get(), w, g, m, v, p);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error(std::string(label) + " probe backend allocation failed");
                return -1;
            }
            const size_t n = static_cast<size_t>(ggml_nelements(w));
            if (dst_len < n) {
                set_error(std::string("probe dst buffer is too small for ") + label);
                return -1;
            }
            std::vector<uint16_t> stored(n);
            if (bf16) {
                ggml_fp32_to_bf16_row_ref(src0, reinterpret_cast<ggml_bf16_t *>(stored.data()), n);
            } else {
                ggml_fp32_to_fp16_row(src0, reinterpret_cast<ggml_fp16_t *>(stored.data()), n);
            }
            // AdamW: pars[7] seeds the stochastic rounding of the half
            // precision store, pars[8] is the clipping scale. SGD: pars[2]
            // is the seed and the gradient is pre-scaled. src2, when given,
            // carries {scale, seed} so a test can pin them down.
            const float gscale = src2 ? src2[0] : 1.0f;
            const float sr_seed = (src2 && ne_src2[0] > 1) ? src2[1] : 0.0f;
            ggml_backend_tensor_set(w, stored.data(), 0, n*sizeof(uint16_t));
            if (sgd) {
                // The SGD kernel takes an already-scaled gradient, as the
                // graph multiplies it in.
                std::vector<float> scaled(n);
                for (size_t i = 0; i < n; ++i) {
                    scaled[i] = src1[i] * gscale;
                }
                const float pars[3] = { param0, param1, sr_seed };
                ggml_backend_tensor_set(g, scaled.data(), 0, n*sizeof(float));
                ggml_backend_tensor_set(p, pars, 0, sizeof(pars));
            } else {
                const float pars[9] = {
                    param0, 0.9f, 0.999f, 1.0e-8f, param1,
                    1.0f / (1.0f - 0.9f), 1.0f / (1.0f - 0.999f), sr_seed, gscale,
                };
                ggml_backend_tensor_set(g, src1, 0, n*sizeof(float));
                ggml_backend_tensor_set(p, pars, 0, sizeof(pars));
                ggml_backend_tensor_memset(m, 0, 0, ggml_nbytes(m));
                ggml_backend_tensor_memset(v, 0, 0, ggml_nbytes(v));
            }
            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error(std::string(label) + " probe graph compute failed");
                return -1;
            }
            ggml_backend_tensor_get(w, stored.data(), 0, n*sizeof(uint16_t));
            if (bf16) {
                ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(stored.data()), dst, n);
            } else {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(stored.data()), dst, n);
            }
            return 0;
        }

        // SSM_SCAN_BACK has eight inputs, so the generic three-input probe ABI
        // carries them in one deterministic packed F32 buffer. This path is
        // intentionally test-only; production graphs construct the op normally.
        if (op == RETRO_PROBE_OP_SSM_SCAN_BACK) {
            const int64_t nc    = ne_src0[0];
            const int64_t nr    = ne_src0[1];
            const int64_t nh    = ne_src0[2];
            const int64_t nslot = ne_src0[3];
            const int64_t ng    = ne_src1[0];
            const int64_t nt    = ne_src1[1];
            const int64_t ns    = ne_src1[2];
            const int64_t nA0   = ne_src1[3];
            if (nc <= 0 || nr <= 0 || nh <= 0 || nslot <= 0 || ng <= 0 ||
                    nt <= 0 || ns <= 0 || nh % ng != 0 || (nA0 != 1 && nA0 != nc)) {
                set_error("invalid packed SSM_SCAN_BACK probe dimensions");
                return -1;
            }

            ggml_tensor * ts   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, nc, nr, nh, nslot);
            ggml_tensor * tx   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, nr, nh, nt, ns);
            ggml_tensor * tdt  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nh, nt, ns);
            ggml_tensor * tA   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nA0, nh);
            ggml_tensor * tB   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, nc, ng, nt, ns);
            ggml_tensor * tC   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, nc, ng, nt, ns);
            ggml_tensor * tids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ns);
            ggml_tensor * tds  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32,
                    nr*nh*nt*ns + nc*nr*nh*ns);
            ggml_tensor * out = ggml_ssm_scan_back(ctx.get(), ts, tx, tdt, tA, tB, tC, tids, tds);
            if (!out) {
                set_error("SSM_SCAN_BACK probe graph construction failed");
                return -1;
            }

            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error("SSM_SCAN_BACK probe backend allocation failed");
                return -1;
            }

            size_t off = 0;
            const auto upload_f32 = [&](ggml_tensor * t) {
                const size_t n = static_cast<size_t>(ggml_nelements(t));
                ggml_backend_tensor_set(t, src0 + off, 0, n*sizeof(float));
                off += n;
            };
            upload_f32(ts);
            upload_f32(tx);
            upload_f32(tdt);
            upload_f32(tA);
            upload_f32(tB);
            upload_f32(tC);
            std::vector<int32_t> idata(static_cast<size_t>(ns));
            for (int64_t i = 0; i < ns; ++i) {
                idata[static_cast<size_t>(i)] = static_cast<int32_t>(src0[off + i]);
            }
            off += static_cast<size_t>(ns);
            ggml_backend_tensor_set(tids, idata.data(), 0, idata.size()*sizeof(int32_t));
            upload_f32(tds);

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("SSM_SCAN_BACK probe graph compute failed");
                return -1;
            }
            const size_t n_out = static_cast<size_t>(ggml_nelements(out));
            if (dst_len < n_out) {
                set_error("probe dst buffer is too small for SSM_SCAN_BACK");
                return -1;
            }
            ggml_backend_tensor_get(out, dst, 0, n_out*sizeof(float));
            return 0;
        }

        // GATED_DELTA_NET_BACK has seven inputs of varying shape, packed the
        // using the same packed representation as SSM_SCAN_BACK. No q/k broadcast against v is
        // exercised here (q/k head count is pinned to H). param0 pins which of
        // the two formulations runs (see ggml_gated_delta_net_back_chunked):
        // 0 backend default, negative sequential, positive chunkwise with that
        // chunk length -- so one process can compare them against each other.
        if (op == RETRO_PROBE_OP_GATED_DELTA_NET_BACK) {
            const int64_t S_v      = ne_src0[0];
            const int64_t H        = ne_src0[1];
            const int64_t n_tokens = ne_src0[2];
            const int64_t n_seqs   = ne_src0[3];
            const int64_t K        = ne_src1[0];
            const bool    kda      = ne_src1[1] != 0;
            if (S_v <= 0 || H <= 0 || n_tokens <= 0 || n_seqs <= 0 || K < 1) {
                set_error("invalid packed GATED_DELTA_NET_BACK probe dimensions");
                return -1;
            }

            ggml_tensor * tq    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
            ggml_tensor * tk    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
            ggml_tensor * tv    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
            ggml_tensor * tg    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kda ? S_v : 1, H, n_tokens, n_seqs);
            ggml_tensor * tbeta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
            ggml_tensor * tstate = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, S_v, H, n_seqs);
            const int64_t n_grad = S_v * H * n_tokens * n_seqs + K * S_v * S_v * H * n_seqs;
            ggml_tensor * tgrad = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_grad);

            ggml_tensor * out = ggml_gated_delta_net_back_chunked(
                    ctx.get(), tq, tk, tv, tg, tbeta, tstate, tgrad, K,
                    static_cast<int32_t>(param0));
            if (!out) {
                set_error("GATED_DELTA_NET_BACK probe graph construction failed");
                return -1;
            }

            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            if (!buffer) {
                set_error("GATED_DELTA_NET_BACK probe backend allocation failed");
                return -1;
            }

            size_t off = 0;
            const auto upload_f32 = [&](ggml_tensor * t) {
                const size_t n = static_cast<size_t>(ggml_nelements(t));
                ggml_backend_tensor_set(t, src0 + off, 0, n*sizeof(float));
                off += n;
            };
            upload_f32(tq);
            upload_f32(tk);
            upload_f32(tv);
            upload_f32(tg);
            upload_f32(tbeta);
            upload_f32(tstate);
            upload_f32(tgrad);

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("GATED_DELTA_NET_BACK probe graph compute failed");
                return -1;
            }
            const size_t n_out = static_cast<size_t>(ggml_nelements(out));
            if (dst_len < n_out) {
                set_error("probe dst buffer is too small for GATED_DELTA_NET_BACK");
                return -1;
            }
            ggml_backend_tensor_get(out, dst, 0, n_out*sizeof(float));
            return 0;
        }

        // The non-F32 src0 out-prod probes: the caller's F32 data is converted to
        // `quant_type`, so both backends decode the exact same bytes.
        ggml_type quant_type = GGML_TYPE_COUNT;
        switch (op) {
            case RETRO_PROBE_OP_OUT_PROD_QUANT: {
                const int type_id = static_cast<int>(param0);
                // Membership in GGML_RETRO_OUT_PROD_TYPES, not merely "a valid enum
                // value": an unlisted type reaches ggml_out_prod, where the CPU
                // reference GGML_ABORTs (BF16 does exactly that) and a GPU backend
                // aborts on the missing pipeline. An error return is something a
                // test can assert on; an abort takes the process down.
                if (type_id <= GGML_TYPE_F32 || type_id >= GGML_TYPE_COUNT ||
                        !is_retro_out_prod_type(static_cast<ggml_type>(type_id))) {
                    set_error("OUT_PROD_QUANT: param0 is not a type in "
                              "GGML_RETRO_OUT_PROD_TYPES");
                    return -1;
                }
                quant_type = static_cast<ggml_type>(type_id);
                break;
            }
            case RETRO_PROBE_OP_OUT_PROD_Q8_0: quant_type = GGML_TYPE_Q8_0; break;
            case RETRO_PROBE_OP_OUT_PROD_Q4_0: quant_type = GGML_TYPE_Q4_0; break;
            case RETRO_PROBE_OP_OUT_PROD_Q4_1: quant_type = GGML_TYPE_Q4_1; break;
            case RETRO_PROBE_OP_OUT_PROD_Q5_0: quant_type = GGML_TYPE_Q5_0; break;
            case RETRO_PROBE_OP_OUT_PROD_Q5_1: quant_type = GGML_TYPE_Q5_1; break;
            case RETRO_PROBE_OP_OUT_PROD_Q2_K: quant_type = GGML_TYPE_Q2_K; break;
            case RETRO_PROBE_OP_OUT_PROD_Q3_K: quant_type = GGML_TYPE_Q3_K; break;
            case RETRO_PROBE_OP_OUT_PROD_Q4_K: quant_type = GGML_TYPE_Q4_K; break;
            case RETRO_PROBE_OP_OUT_PROD_Q5_K: quant_type = GGML_TYPE_Q5_K; break;
            case RETRO_PROBE_OP_OUT_PROD_Q6_K: quant_type = GGML_TYPE_Q6_K; break;
            default: break;
        }
        const bool quant_src0 = quant_type != GGML_TYPE_COUNT;
        if (quant_src0 && ne_src0[0] % ggml_blck_size(quant_type) != 0) {
            set_error("quantized OUT_PROD requires ne_src0[0] to be a multiple of "
                      "the quantization block size");
            return -1;
        }
        ggml_tensor * a = ggml_new_tensor_4d(ctx.get(),
                quant_src0 ? quant_type : GGML_TYPE_F32,
                ne_src0[0], ne_src0[1], ne_src0[2], ne_src0[3]);
        // GET_ROWS_BACK takes I32 row indices as its second input; every other
        // probe op takes F32. The FFI stays all-float: indices arrive as floats
        // and are cast before upload.
        const ggml_type b_type = op == RETRO_PROBE_OP_GET_ROWS_BACK
                ? GGML_TYPE_I32
                : GGML_TYPE_F32;
        ggml_tensor * b = ggml_new_tensor_4d(ctx.get(), b_type,
                ne_src1[0], ne_src1[1], ne_src1[2], ne_src1[3]);
        ggml_tensor * c = nullptr;
        if (needs_src2) {
            c = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                    ne_src2[0], ne_src2[1], ne_src2[2], ne_src2[3]);
        }
        if (!a || !b || (needs_src2 && !c)) {
            set_error("probe input tensor allocation failed");
            return -1;
        }

        ggml_tensor * out = nullptr;
        switch (op) {
            case RETRO_PROBE_OP_SILU_BACK:
                out = ggml_silu_back(ctx.get(), a, b);
                break;
            case RETRO_PROBE_OP_RMS_NORM_BACK:
                out = ggml_rms_norm_back(ctx.get(), a, b, param0);
                break;
            case RETRO_PROBE_OP_L2_NORM_BACK:
                out = ggml_l2_norm_back(ctx.get(), a, b, param0);
                break;
            case RETRO_PROBE_OP_CUMSUM:
                // a = x; b is ignored (the ABI always carries two inputs).
                out = ggml_cumsum(ctx.get(), a);
                break;
            case RETRO_PROBE_OP_OUT_PROD:
            case RETRO_PROBE_OP_OUT_PROD_QUANT:
            case RETRO_PROBE_OP_OUT_PROD_Q8_0:
            case RETRO_PROBE_OP_OUT_PROD_Q4_0:
            case RETRO_PROBE_OP_OUT_PROD_Q4_1:
            case RETRO_PROBE_OP_OUT_PROD_Q5_0:
            case RETRO_PROBE_OP_OUT_PROD_Q5_1:
            case RETRO_PROBE_OP_OUT_PROD_Q2_K:
            case RETRO_PROBE_OP_OUT_PROD_Q3_K:
            case RETRO_PROBE_OP_OUT_PROD_Q4_K:
            case RETRO_PROBE_OP_OUT_PROD_Q5_K:
            case RETRO_PROBE_OP_OUT_PROD_Q6_K:
                out = ggml_out_prod(ctx.get(), a, b);
                break;
            case RETRO_PROBE_OP_REPEAT_BACK:
                // a = broadcast gradient, b = shape template to reduce onto
                out = ggml_repeat_back(ctx.get(), a, b);
                break;
            case RETRO_PROBE_OP_SSM_CONV_BACK:
                out = ggml_ssm_conv_back(ctx.get(), a, b, c);
                break;
            case RETRO_PROBE_OP_CONV_RS_GATHER:
                // a = conv_input; b is ignored. param0 = kernel_m1, param1 = K.
                out = ggml_conv_rs_gather(ctx.get(), a,
                        static_cast<int64_t>(param0), static_cast<int64_t>(param1));
                break;
            case RETRO_PROBE_OP_SOFT_MAX_BACK:
                // a = dy, b = y, param0 = scale, param1 = max_bias
                out = ggml_soft_max_ext_back(ctx.get(), a, b, param0, param1);
                break;
            case RETRO_PROBE_OP_CROSS_ENTROPY_LOSS:
                // a = logits, b = labels -> scalar loss
                out = ggml_cross_entropy_loss(ctx.get(), a, b);
                break;
            case RETRO_PROBE_OP_CROSS_ENTROPY_LOSS_BACK:
                // a = grad of the loss (scalar), b = logits, c = labels
                out = ggml_cross_entropy_loss_back(ctx.get(), a, b, c);
                break;
            case RETRO_PROBE_OP_GET_ROWS_BACK:
                // a = grad rows, b = I32 indices, c = shape template for dst
                out = ggml_get_rows_back(ctx.get(), a, b, c);
                break;
            default:
                set_error("unknown probe op");
                return -1;
        }
        if (!out) {
            set_error("probe op graph construction failed");
            return -1;
        }

        // For the cross-entropy ops, a positive param1 pins the active-row
        // count that the training path would provide through
        // ggml_opt_set_loss_active_rows(). The CPU backend derives the count
        // from the label data itself, so passing the true count keeps CPU and
        // Metal normalization comparable in weighted/masked-label tests.
        if ((op == RETRO_PROBE_OP_CROSS_ENTROPY_LOSS ||
                    op == RETRO_PROBE_OP_CROSS_ENTROPY_LOSS_BACK) && param1 > 0.0f) {
            out->op_params[0] = static_cast<int32_t>(param1);
            out->op_params[1] = 1;
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            set_error("probe backend buffer allocation failed");
            return -1;
        }

        if (quant_src0) {
            const int64_t n_per_row = ne_src0[0];
            const int64_t n_rows    = ggml_nelements(a) / n_per_row;
            std::vector<uint8_t> qdata(ggml_nbytes(a));
            if (quant_type == GGML_TYPE_F16) {
                // Not a quantized type, so ggml_quantize_chunk does not handle it.
                ggml_fp32_to_fp16_row(src0, (ggml_fp16_t *) qdata.data(),
                        static_cast<int64_t>(ggml_nelements(a)));
            } else {
                // A few IQ types assert on a null importance matrix. Uniform
                // importance is the natural stand-in and keeps the probe
                // deterministic; it cannot affect parity either way, since the same
                // quantized bytes are uploaded to both backends.
                std::vector<float> imatrix;
                if (ggml_quantize_requires_imatrix(quant_type)) {
                    imatrix.assign(static_cast<size_t>(n_per_row), 1.0f);
                }
                ggml_quantize_chunk(quant_type, src0, qdata.data(),
                        0, n_rows, n_per_row, imatrix.empty() ? nullptr : imatrix.data());
            }
            ggml_backend_tensor_set(a, qdata.data(), 0, ggml_nbytes(a));
        } else {
            ggml_backend_tensor_set(a, src0, 0, ggml_nbytes(a));
        }
        if (b_type == GGML_TYPE_I32) {
            const size_t n_idx = static_cast<size_t>(ggml_nelements(b));
            std::vector<int32_t> indices(n_idx);
            for (size_t i = 0; i < n_idx; ++i) {
                indices[i] = static_cast<int32_t>(src1[i]);
            }
            ggml_backend_tensor_set(b, indices.data(), 0, ggml_nbytes(b));
        } else {
            ggml_backend_tensor_set(b, src1, 0, ggml_nbytes(b));
        }
        // GET_ROWS_BACK's third input is a pure shape template: never read its
        // data (callers may pass an empty or dummy buffer).
        if (c && src2 && (op == RETRO_PROBE_OP_CROSS_ENTROPY_LOSS_BACK ||
                         op == RETRO_PROBE_OP_SSM_CONV_BACK)) {
            ggml_backend_tensor_set(c, src2, 0, ggml_nbytes(c));
        }

        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
            // A `require` preflight failure is the one graph-compute error with
            // a precise cause to report; anything else is genuinely opaque here.
            char rir_msg[512];
            ggml_rir_violation_format(rir_msg, sizeof(rir_msg));
            set_error(rir_msg[0] != '\0'
                    ? rir_msg
                    : "probe graph compute failed on the selected backend");
            return -1;
        }

        const size_t n_out = static_cast<size_t>(ggml_nelements(out));
        if (dst_len < n_out) {
            set_error("probe dst buffer is too small for the op output");
            return -1;
        }
        ggml_backend_tensor_get(out, dst, 0, n_out * sizeof(float));
        return 0;
    });
}

int probe_op_run_impl(
        int32_t op,
        int32_t use_gpu,
        const int64_t * ne_src0,
        const float * src0,
        const int64_t * ne_src1,
        const float * src1,
        const int64_t * ne_src2,
        const float * src2,
        float param0,
        float param1,
        float * dst,
        size_t dst_len) {
    std::lock_guard<std::mutex> serialize(probe_device_mutex());
    return probe_op_run_locked(op, use_gpu, ne_src0, src0, ne_src1, src1,
            ne_src2, src2, param0, param1, dst, dst_len);
}

// ---------------------------------------------------------------------------
// retro delta: the two fixed-block Gefen ops, driven directly
// ---------------------------------------------------------------------------

namespace {

// The state tensors of one Gefen variant, built once so the probe and the
// support predicate below cannot describe different graphs.
struct gefen_graph {
    ggml_tensor * w        = nullptr;
    ggml_tensor * grad     = nullptr;
    ggml_tensor * moment   = nullptr;
    ggml_tensor * scales   = nullptr;
    ggml_tensor * v        = nullptr;
    ggml_tensor * codebook = nullptr;
    ggml_tensor * pars     = nullptr;
    ggml_tensor * stats    = nullptr;
    ggml_tensor * step     = nullptr;
};

// `n_blocks` the way both phases compute it, so a caller that passed a
// different one is refused rather than silently reshaped.
int64_t gefen_block_count(int64_t n_elements, int64_t block_size) {
    return n_elements/block_size + (n_elements % block_size != 0);
}

bool gefen_build(
        ggml_context * ctx, gefen_graph & g, int32_t variant,
        int64_t n_elements, int64_t block_size, int64_t levels) {
    const bool quantized = variant == GGML_OPT_GEFEN_VARIANT_QUANTIZED_M;
    const int64_t n_blocks = gefen_block_count(n_elements, block_size);

    g.w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    if (!g.w) {
        return false;
    }
    // Phase B asserts the flag: it writes the parameter in place, and the flag
    // is what says a tensor may be written in place.
    ggml_set_param(g.w);
    g.grad   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    g.moment = ggml_new_tensor_1d(ctx, quantized ? GGML_TYPE_I8 : GGML_TYPE_F32, n_elements);
    g.v      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_blocks);
    g.pars   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    if (quantized) {
        g.scales   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_blocks);
        g.codebook = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, levels);
    }
    if (!g.grad || !g.moment || !g.v || !g.pars || (quantized && (!g.scales || !g.codebook))) {
        return false;
    }

    g.stats = ggml_opt_step_gefen_stats(
            ctx, g.grad, g.moment, g.scales, g.v, g.codebook, g.pars,
            variant, static_cast<int>(block_size));
    if (!g.stats) {
        return false;
    }
    g.step = ggml_opt_step_gefen(
            ctx, g.w, g.grad, g.moment, g.scales, g.v, g.stats, g.codebook, g.pars,
            variant, static_cast<int>(block_size));
    return g.step != nullptr;
}

// A device carries this step only when it carries *both* phases: one of the two
// on a fallback backend would mutate a copy of the state and leave the
// device-resident slot stale.
bool gefen_device_supports(ggml_backend_dev_t dev, const gefen_graph & g) {
    return ggml_backend_dev_supports_op(dev, g.stats)
            && ggml_backend_dev_supports_op(dev, g.step);
}

} // namespace

int gefen_probe_supported_impl(
        int32_t use_gpu, int32_t variant, int32_t block_size, int32_t * out_supported) {
    return boundary([&]() -> int {
        std::lock_guard<std::mutex> serialize(probe_device_mutex());
        ensure_backend_initialized();
        if (!out_supported) {
            set_error("out_supported is required");
            return -1;
        }
        if (block_size <= 0) {
            set_error("gefen probe block_size must be positive");
            return -1;
        }
        ggml_backend_dev_t dev = use_gpu
                ? first_gpu_device()
                : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            set_error(use_gpu ? "no GPU device available for probe"
                              : "no CPU device available for probe");
            return -1;
        }
        ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 16,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
        if (!ctx) {
            set_error("gefen probe context allocation failed");
            return -1;
        }
        gefen_graph g;
        // Two full blocks and a partial third: the shape a per-block kernel is
        // allowed to refuse for reasons that have nothing to do with the dtype.
        if (!gefen_build(ctx.get(), g, variant, 2*block_size + 1, block_size,
                    GGML_OPT_GEFEN_CODEBOOK_LEVELS)) {
            set_error("gefen probe graph construction failed");
            return -1;
        }
        *out_supported = gefen_device_supports(dev, g) ? 1 : 0;
        return 0;
    });
}

int gefen_probe_run_impl(retro_gefen_probe * probe) {
    return boundary([&]() -> int {
        std::lock_guard<std::mutex> serialize(probe_device_mutex());
        ensure_backend_initialized();
        if (!probe) {
            set_error("retro_gefen_probe is required");
            return -1;
        }
        if (probe->struct_size != sizeof(retro_gefen_probe)) {
            set_error("retro_gefen_probe.struct_size does not match this build");
            return -1;
        }
        const bool quantized = probe->variant == GGML_OPT_GEFEN_VARIANT_QUANTIZED_M;
        if (!quantized && probe->variant != GGML_OPT_GEFEN_VARIANT_SHARED_V) {
            set_error("unknown gefen variant");
            return -1;
        }
        if (probe->block_size <= 0 || probe->n_elements <= 0 || probe->n_steps <= 0) {
            set_error("gefen probe needs a positive block size, element count and step count");
            return -1;
        }
        if (gefen_block_count(probe->n_elements, probe->block_size) != probe->n_blocks) {
            set_error("gefen probe n_blocks is not ceil(n_elements / block_size)");
            return -1;
        }
        if (!probe->weights || !probe->grad || !probe->v || !probe->pars) {
            set_error("gefen probe requires weights, grad, v and pars");
            return -1;
        }
        if (quantized) {
            if (!probe->indices || !probe->scales || !probe->codebook || probe->levels < 2) {
                set_error("quantized_m needs indices, scales and a codebook of at least two entries");
                return -1;
            }
            if (probe->moment) {
                set_error("quantized_m keeps no F32 first moment; pass indices instead");
                return -1;
            }
        } else {
            if (!probe->moment) {
                set_error("shared_v needs an F32 first moment");
                return -1;
            }
            if (probe->indices || probe->scales || probe->codebook) {
                set_error("shared_v keeps no indices, scales or codebook");
                return -1;
            }
        }

        ggml_backend_dev_t dev = probe->use_gpu
                ? first_gpu_device()
                : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            set_error(probe->use_gpu ? "no GPU device available for probe"
                                     : "no CPU device available for probe");
            return -1;
        }
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            set_error("failed to initialize backend for gefen probe");
            return -1;
        }

        ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
        if (!ctx) {
            set_error("gefen probe context allocation failed");
            return -1;
        }

        gefen_graph g;
        if (!gefen_build(ctx.get(), g, probe->variant, probe->n_elements,
                    probe->block_size, probe->levels)) {
            set_error("gefen probe graph construction failed");
            return -1;
        }
        if (!gefen_device_supports(dev, g)) {
            set_error("the selected device does not carry both fixed-block Gefen phases");
            return -1;
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            set_error("gefen probe backend buffer allocation failed");
            return -1;
        }

        ggml_backend_tensor_set(g.w,    probe->weights, 0, ggml_nbytes(g.w));
        ggml_backend_tensor_set(g.grad, probe->grad,    0, ggml_nbytes(g.grad));
        ggml_backend_tensor_set(g.v,    probe->v,       0, ggml_nbytes(g.v));
        ggml_backend_tensor_set(g.pars, probe->pars,    0, ggml_nbytes(g.pars));
        if (quantized) {
            ggml_backend_tensor_set(g.moment,   probe->indices,  0, ggml_nbytes(g.moment));
            ggml_backend_tensor_set(g.scales,   probe->scales,   0, ggml_nbytes(g.scales));
            ggml_backend_tensor_set(g.codebook, probe->codebook, 0, ggml_nbytes(g.codebook));
        } else {
            ggml_backend_tensor_set(g.moment, probe->moment, 0, ggml_nbytes(g.moment));
        }

        // One graph, computed n_steps times: the state tensors are the same
        // allocations across steps, which is exactly what a run does between
        // two optimizer steps.
        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, g.step);
        for (int32_t step = 0; step < probe->n_steps; ++step) {
            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                set_error("gefen probe graph compute failed on the selected backend");
                return -1;
            }
        }

        ggml_backend_tensor_get(g.w, probe->weights, 0, ggml_nbytes(g.w));
        ggml_backend_tensor_get(g.v, probe->v,       0, ggml_nbytes(g.v));
        if (quantized) {
            ggml_backend_tensor_get(g.moment, probe->indices, 0, ggml_nbytes(g.moment));
            ggml_backend_tensor_get(g.scales, probe->scales,  0, ggml_nbytes(g.scales));
        } else {
            ggml_backend_tensor_get(g.moment, probe->moment, 0, ggml_nbytes(g.moment));
        }
        if (probe->stats) {
            ggml_backend_tensor_get(g.stats, probe->stats, 0, ggml_nbytes(g.stats));
        }
        return 0;
    });
}

// ---------------------------------------------------------------------------
// RIR: implementation selection and reporting
// ---------------------------------------------------------------------------

namespace {

// The ggml op a probe id builds, for registry lookups. Only the ids that can
// have a RIR variant need an entry: an unmapped id simply has no variant, which
// is the honest answer for "can RIR run this?".
bool probe_ggml_op(int32_t op, enum ggml_op * out) {
    switch (op) {
        case RETRO_PROBE_OP_L2_NORM_BACK: *out = GGML_OP_L2_NORM_BACK; return true;
        case RETRO_PROBE_OP_CUMSUM:       *out = GGML_OP_CUMSUM;       return true;
        default:                          return false;
    }
}

// The registry backend a ggml device belongs to. Returns false for a backend
// RIR does not model, which keeps "unknown backend" distinct from "known
// backend, no variant".
bool rir_backend_of_device(ggml_backend_dev_t dev, rir_backend * out) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!name) {
        return false;
    }
    // The spellings ggml publishes for its registries: GGML_METAL_NAME is
    // "MTL", GGML_VK_NAME is "Vulkan", and GGML_CUDA_NAME follows the toolkit
    // the build targets.
    if (std::strcmp(name, "MTL")    == 0) { *out = RIR_BACKEND_METAL;  return true; }
    if (std::strcmp(name, "Vulkan") == 0) { *out = RIR_BACKEND_VULKAN; return true; }
    if (std::strcmp(name, "CUDA")   == 0 || std::strcmp(name, "ROCm") == 0 ||
            std::strcmp(name, "MUSA") == 0) {
        *out = RIR_BACKEND_CUDA;
        return true;
    }
    if (std::strcmp(name, "CPU")    == 0) { *out = RIR_BACKEND_CPU;    return true; }
    return false;
}

// The variant a probe would use on `dev`, or nullptr. Selection goes through
// the registry's own rule, so the probe sees exactly what the backend will see
// - including a pair the policy table registers for observation only, which
// `probe_policy` then separates from a dispatchable one.
const rir_variant_desc * probe_variant(int32_t op, ggml_backend_dev_t dev) {
    enum ggml_op ggml_op;
    rir_backend backend;
    if (!probe_ggml_op(op, &ggml_op) || !rir_backend_of_device(dev, &backend)) {
        return nullptr;
    }
    return ggml_rir_find_variant_for_op((int32_t) ggml_op, backend);
}

// The registry policy for this probe op on `dev`.
rir_policy probe_policy(int32_t op, ggml_backend_dev_t dev) {
    enum ggml_op ggml_op;
    rir_backend backend;
    if (!probe_ggml_op(op, &ggml_op) || !rir_backend_of_device(dev, &backend)) {
        return RIR_POLICY_NATIVE_ONLY;
    }
    return ggml_rir_op_policy((int32_t) ggml_op, backend);
}

// Whether the native kernel of this probe op is gone on `dev`.
// Keyed by the registry spelling, like the report.
bool probe_native_retired(int32_t op, ggml_backend_dev_t dev) {
    enum ggml_op ggml_op;
    rir_backend backend;
    if (!probe_ggml_op(op, &ggml_op) || !rir_backend_of_device(dev, &backend)) {
        return false;
    }
    return ggml_rir_op_native_retired(ggml_op_name(ggml_op), backend);
}

// Scoped force-native: the only way to make a NATIVE request unfalsifiable
// while the process runs in prefer mode. Restores on every exit path.
struct force_native_guard {
    bool active;
    explicit force_native_guard(bool on) : active(on) {
        if (active) {
            ggml_rir_set_force_native(true);
        }
    }
    ~force_native_guard() {
        if (active) {
            ggml_rir_set_force_native(false);
        }
    }
};

// One spelling of a rir_backend for every report; ggml owns it so the stats
// line, this report and capability_report cannot drift apart.
const char * backend_spelling(uint8_t backend) {
    return ggml_rir_backend_name(backend);
}

} // namespace

int probe_op_run_ex_impl(
        int32_t op,
        int32_t use_gpu,
        const int64_t * ne_src0,
        const float * src0,
        const int64_t * ne_src1,
        const float * src1,
        const int64_t * ne_src2,
        const float * src2,
        float param0,
        float param1,
        float * dst,
        size_t dst_len,
        int32_t implementation,
        retro_kernel_run_info * info) {
    if (info) {
        if (info->struct_size != sizeof(retro_kernel_run_info)) {
            set_error("retro_kernel_run_info.struct_size does not match this build");
            return -1;
        }
        info->requested_impl = implementation;
        info->executed_impl  = RETRO_KERNEL_IMPL_NATIVE;
        info->reject_reason  = RETRO_KERNEL_MATCHED;
        info->variant[0]     = '\0';
    }
    if (implementation != RETRO_KERNEL_IMPL_AUTO &&
            implementation != RETRO_KERNEL_IMPL_NATIVE &&
            implementation != RETRO_KERNEL_IMPL_RIR) {
        set_error("unknown retro_kernel_impl");
        return -1;
    }

    // Held from here: the pre-check in this function also initializes the backend, and on
    // Vulkan a concurrent probe destroying the device between that check and
    // the run is exactly the abort this lock exists to prevent.
    std::lock_guard<std::mutex> serialize(probe_device_mutex());

    // Refuse a NATIVE request when that backend has retired the native kernel.
    // Quietly falling back to the CPU would make a "GPU native" parity result
    // indistinguishable from a real native execution.
    if (implementation == RETRO_KERNEL_IMPL_NATIVE && use_gpu) {
        const int rc = boundary([&]() -> int {
            ensure_backend_initialized();
            ggml_backend_dev_t dev = first_gpu_device();
            if (dev && probe_native_retired(op, dev)) {
                set_error("the native kernel of this op has been retired on this backend: "
                          "the generated variant is the only "
                          "implementation, and the CPU probe is the independent reference");
                return -1;
            }
            return 0;
        });
        if (rc != 0) {
            if (info) {
                info->reject_reason = RETRO_KERNEL_REJECT_PIPELINE;
            }
            return rc;
        }
    }

    // A RIR request is refused before running anything when the answer is
    // already known: no variant in this build, or a mode that never built the
    // pipelines. Both are configuration errors, and running the native kernel
    // then reporting "RIR" is exactly what this entry point exists to prevent.
    if (implementation == RETRO_KERNEL_IMPL_RIR) {
        const int rc = boundary([&]() -> int {
            ensure_backend_initialized();
            ggml_backend_dev_t dev = use_gpu
                    ? first_gpu_device()
                    : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                set_error(use_gpu ? "no GPU device available for probe"
                                  : "no CPU device available for probe");
                return -1;
            }
            if (!probe_variant(op, dev)) {
                set_error("no RIR variant is compiled for this op on this backend");
                return -1;
            }
            // A variant registered for observation only will never be encoded,
            // whatever the mode: saying so here, by name, is the difference
            // between a configuration error and a mysterious "the native
            // kernel ran" after the fact.
            if (probe_policy(op, dev) != RIR_POLICY_PREFER_GENERATED) {
                set_error("the RIR variant for this op is registered as "
                          "observe_generated on this backend: its contract is "
                          "evaluated and counted, but the native kernel runs");
                return -1;
            }
            if (ggml_rir_get_mode() < GGML_RIR_MODE_PREFER) {
                set_error("RIR was requested but RETRO_RIR_MODE is not prefer/require; "
                          "the policy must be set before the backend context is created");
                return -1;
            }
            return 0;
        });
        if (rc != 0) {
            if (info) {
                info->reject_reason = RETRO_KERNEL_REJECT_PIPELINE;
            }
            return rc;
        }
    }

    ggml_rir_decision_reset();
    ggml_rir_violation_reset();
    force_native_guard guard(implementation == RETRO_KERNEL_IMPL_NATIVE);

    const int rc = probe_op_run_locked(op, use_gpu, ne_src0, src0, ne_src1, src1,
            ne_src2, src2, param0, param1, dst, dst_len);

    const ggml_rir_decision decision = ggml_rir_decision_last();
    // Anything that is not a recorded RIR dispatch ran the native kernel: an op
    // without a variant records nothing, and force-native cannot dispatch.
    const bool ran_rir = decision.impl == GGML_RIR_IMPL_RIR;
    if (info) {
        info->executed_impl = ran_rir ? RETRO_KERNEL_IMPL_RIR : RETRO_KERNEL_IMPL_NATIVE;
        info->reject_reason = guard.active ? RETRO_KERNEL_REJECT_POLICY_NATIVE
                                           : (int32_t) decision.reject;
        if (ran_rir && decision.variant) {
            std::snprintf(info->variant, sizeof(info->variant), "%s", decision.variant);
        }
    }
    if (rc != 0) {
        return rc;
    }
    if (implementation == RETRO_KERNEL_IMPL_RIR && !ran_rir) {
        // A RIR probe never falls back silently.
        set_error("the RIR variant was requested but the native kernel ran "
                  "(the op is outside the variant's contract)");
        return -1;
    }
    return 0;
}

int rir_counters_impl(retro_rir_counters * out) {
    if (!out) {
        set_error("retro_rir_counters_get requires a non-null out");
        return -1;
    }
    if (out->struct_size != sizeof(retro_rir_counters)) {
        set_error("retro_rir_counters.struct_size does not match this build");
        return -1;
    }
    const ggml_rir_counters c = ggml_rir_counters_snapshot();
    out->mode              = (int32_t) ggml_rir_get_mode();
    out->ops_seen          = c.ops_seen;
    out->rir_eligible      = c.rir_eligible;
    out->rir_dispatched    = c.rir_dispatched;
    out->native_dispatched = c.native_dispatched;
    out->fallback_contract = c.fallback_contract;
    out->fallback_feature  = c.fallback_feature;
    out->fallback_pipeline = c.fallback_pipeline;
    // The copy loop uses the *ggml* constant, so the two sides must
    // agree exactly: a taxonomy that grew without this header growing with it
    // would write past `out->reject_by_reason` rather than truncate.
    static_assert(GGML_RIR_REJECT_COUNT == RETRO_KERNEL_REJECT_COUNT,
            "retro_rir_counters.reject_by_reason must match the ggml taxonomy");
    for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
        out->reject_by_reason[i] = c.reject_by_reason[i];
    }
    return 0;
}

// Tab-separated breakdown of the RIR counters by (ggml_op, backend, variant),
// shared by the FFI variant report and capability_report so both read the same
// rows. Empty while nothing has been dispatched, which
// is itself the answer under `off`.
std::string rir_site_lines() {
    std::ostringstream out;
    const uint32_t n_sites = ggml_rir_site_count();
    for (uint32_t s = 0; s < n_sites; ++s) {
        const ggml_rir_site_counters row = ggml_rir_site_snapshot(s);
        if (!row.ggml_op) {
            continue;
        }
        const char * backend = backend_spelling(row.backend);
        out << "site\t" << row.ggml_op << "\t" << backend
            << "\tseen=" << row.counters.ops_seen
            << "\teligible=" << row.counters.rir_eligible
            << "\trir=" << row.counters.rir_dispatched
            << "\tnative=" << row.counters.native_dispatched << "\n";
        for (uint32_t v = 0; v < row.n_variants; ++v) {
            out << "site-variant\t" << row.ggml_op << "\t" << backend << "\t"
                << (row.variant_id[v] ? row.variant_id[v] : "?") << "\t"
                << row.variant_dispatched[v] << "\n";
        }
        if (row.variants_overflow != 0) {
            // Never folded into a named variant: a lossy axis has to say so.
            out << "site-variant\t" << row.ggml_op << "\t" << backend
                << "\t<overflow>\t" << row.variants_overflow << "\n";
        }
        for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
            if (row.counters.reject_by_reason[i] != 0) {
                out << "site-reject\t" << row.ggml_op << "\t" << backend << "\t"
                    << ggml_rir_reject_name(i) << "\t"
                    << row.counters.reject_by_reason[i] << "\n";
            }
        }
        // The domain the registry publishes for this pair, one row per declared
        // restriction. A `site-reject` whose reason has
        // no matching `site-domain` is a node the kernel claimed and did not
        // serve - which is what makes a coverage number on a real graph an
        // assertion rather than a log line.
        // Emitted even when the mask is empty, as nothing: a pair that claims
        // its whole ggml op is the case where *any* site-reject is a defect,
        // and a reader must not have to guess whether the rows are missing or
        // the domain is.
        const uint32_t domain = ggml_rir_op_assumed_domain(row.ggml_op, (rir_backend) row.backend);
        for (int i = 0; i < GGML_RIR_REJECT_COUNT; ++i) {
            if (domain & (1u << i)) {
                out << "site-domain\t" << row.ggml_op << "\t" << backend << "\t"
                    << ggml_rir_reject_name(i) << "\n";
            }
        }
        // And whether this pair still has a native kernel at all.
        // It is the strongest thing this report can
        // say about a promotion: not "the generated variant was preferred" but
        // "there is nothing else here". Published rather than inferred from
        // `native=0`, which a lucky graph produces on a pair whose native is
        // very much still there.
        if (ggml_rir_op_native_retired(row.ggml_op, (rir_backend) row.backend)) {
            out << "site-retired\t" << row.ggml_op << "\t" << backend << "\n";
        }
    }
    return out.str();
}

int rir_variant_report_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        // The registry half is fixed at link time; the site half is not, so the
        // whole report is rebuilt per call. It is a diagnostic entry point, not
        // a hot path.
        static const std::string registry = [] {
            std::ostringstream out;
            for (uint32_t i = 0; i < rir_variant_count; ++i) {
                const rir_variant_desc & v = rir_variants[i];
                out << "variant\t" << v.kernel << "\t" << v.ggml_op << "\t"
                    << v.variant_id << "\t" << backend_spelling(v.backend) << "\t"
                    << (unsigned) v.priority << "\n";
            }
            // Report policy rows as well as variants: a variant can exist without
            // being dispatchable under the current policy.
            for (uint32_t i = 0; i < rir_op_policy_count; ++i) {
                const rir_op_policy & p = rir_op_policies[i];
                // `prefer-only` means the generated variant is the only available
                // implementation for the pair.
                const char * kind =
                    p.policy == RIR_POLICY_NATIVE_ONLY       ? "native-only" :
                    p.policy == RIR_POLICY_OBSERVE_GENERATED ? "observe-only" :
                    p.native_retired                         ? "prefer-only"  :
                                                               "prefer";
                out << kind << "\t" << p.ggml_op << "\t"
                    << backend_spelling(p.backend) << "\n";
            }
            return out.str();
        }();

        std::ostringstream out;
        out << registry << rir_site_lines();
        const std::string report = out.str();
        return copy_string_out(report, buffer, n_buffer, out_n_bytes);
    });
}

int rir_census_report_impl(char * buffer, size_t n_buffer, size_t * out_n_bytes) {
    return boundary([&]() -> int {
        // Unsorted, and deliberately so: the rows are the measurement, the
        // ranking is a reading of it. Sorting here would force every consumer
        // to accept this file's idea of what "biggest" means, when the caller
        // may well rank by node count instead.
        std::ostringstream out;
        const uint32_t n = ggml_rir_census_count();
        for (uint32_t i = 0; i < n; ++i) {
            const ggml_rir_census_row row = ggml_rir_census_snapshot(i);
            if (row.ggml_op == nullptr) {
                continue;
            }
            const char * backend = backend_spelling(row.backend);
            out << "census\t" << row.ggml_op << "\t" << backend << "\t"
                << row.n_nodes << "\t" << row.n_bytes << "\t" << row.n_elements << "\t"
                << (row.registered ? "registered" : "uncovered") << "\n";
            for (uint32_t s = 0; s < row.n_shapes; ++s) {
                const ggml_rir_census_shape & sh = row.shapes[s];
                out << "census-shape\t" << row.ggml_op << "\t" << backend << "\t"
                    << ggml_type_name((ggml_type) sh.type) << "\t"
                    << sh.ne[0] << "," << sh.ne[1] << "," << sh.ne[2] << "," << sh.ne[3]
                    << "\t" << sh.n_nodes << "\n";
            }
            if (row.shapes_overflow != 0) {
                // Keep overflow on its own line: the listed shapes do not cover
                // the complete op count.
                out << "census-shape-overflow\t" << row.ggml_op << "\t" << backend << "\t"
                    << row.shapes_overflow << "\n";
            }
        }
        // The subgraph chains use their own line kind. They rank an *edge* and
        // not an op, so folding them into the rows
        // combining them would put two different units in one column.
        const uint32_t n_patterns = ggml_rir_census_pattern_count();
        for (uint32_t i = 0; i < n_patterns; ++i) {
            const ggml_rir_census_pattern pat = ggml_rir_census_pattern_snapshot(i);
            if (pat.n_ops == 0) {
                continue;
            }
            std::string ops;
            for (uint32_t k = 0; k < pat.n_ops; ++k) {
                if (k != 0) {
                    ops += ">";
                }
                ops += ggml_op_name((ggml_op) pat.ops[k]);
            }
            out << "census-pattern\t" << ops << "\t" << backend_spelling(pat.backend) << "\t"
                << pat.n_occurrences << "\t" << pat.n_dispatches << "\t"
                << pat.n_bytes_intermediate << "\t"
                << (pat.registered ? "registered" : "uncovered") << "\n";
        }
        const uint64_t dropped = ggml_rir_census_pattern_overflow();
        if (dropped != 0) {
            out << "census-pattern-overflow\t" << dropped << "\n";
        }
        const std::string report = out.str();
        return copy_string_out(report, buffer, n_buffer, out_n_bytes);
    });
}

} // namespace retro
