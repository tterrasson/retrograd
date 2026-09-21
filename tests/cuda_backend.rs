//! CUDA validation split by layer: registration, individual ggml ops, model
//! offload, LoRA placement, then one minimal optimizer step.
//!
//! Structured as a mirror of `tests/vulkan_backend.rs`, but the op coverage
//! starts from the kernels ggml-cuda already ships (SILU/RMS-norm/soft-max/
//! get-rows backward, F32 OUT_PROD, cross-entropy backward). Flash Attention
//! backward, SSM backward, quantized OUT_PROD, F16 AdamW and the fused sparse
//! cross-entropy are ported in later milestones and gain their probes then.

mod common;

#[cfg(retro_cuda)]
use retrograd::training::batch::train_grpo_batch;
#[cfg(retro_cuda)]
use retrograd::{
    Device, FusedCeProbeInputs, FusedCeProbeShape, FusedCeWeightType, GrpoBatchParams, LoraConfig,
    ProbeInputs, ProbeOp, TargetSet, TrainConfig, TrainSequence, Trainer, WeightedBatch,
    fused_sparse_ce_probe, probe_op,
};

#[cfg(retro_cuda)]
fn cuda_available() -> bool {
    common::cuda_registered() && retrograd::gpu_runtime_available()
}

#[cfg(retro_cuda)]
fn pseudo_random(n: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
            let unit = (bits as f32) / ((1_u32 << 24) as f32);
            (unit * 2.0 - 1.0) * range
        })
        .collect()
}

#[cfg(retro_cuda)]
fn softmax_rows(values: &[f32], width: usize) -> Vec<f32> {
    values
        .chunks(width)
        .flat_map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<_> = row.iter().map(|value| (value - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.into_iter().map(move |value| value / sum)
        })
        .collect()
}

#[cfg(retro_cuda)]
fn assert_close(op: ProbeOp, cpu: &[f32], cuda: &[f32], tolerance: f32) {
    assert_eq!(cpu.len(), cuda.len());
    let mut max_abs = 0.0_f32;
    for (index, (&expected, &actual)) in cpu.iter().zip(cuda).enumerate() {
        let difference = (expected - actual).abs();
        max_abs = max_abs.max(difference);
        assert!(
            difference <= tolerance,
            "{op:?} differs at {index}: cpu={expected}, cuda={actual}, diff={difference}, tolerance={tolerance}"
        );
    }
    eprintln!("{op:?}: max|cpu-cuda|={max_abs:e} (tolerance {tolerance:e})");
}

#[cfg(retro_cuda)]
fn compare_binary_op(
    op: ProbeOp,
    shape: [i64; 4],
    src0: &[f32],
    src1: &[f32],
    params: [f32; 2],
    tolerance: f32,
) {
    let n = shape.iter().product::<i64>() as usize;
    let cpu = probe_op(
        op,
        false,
        ProbeInputs::pair(shape, src0, shape, src1),
        params,
        n,
    )
    .expect("CPU reference probe");
    let cuda = probe_op(
        op,
        true,
        ProbeInputs::pair(shape, src0, shape, src1),
        params,
        n,
    )
    .expect("CUDA probe");
    assert_close(op, &cpu, &cuda, tolerance);
}

#[cfg(retro_cuda)]
#[test]
fn cuda_build_registers_a_cuda_gpu() {
    let list = retrograd::backend_list().expect("backend list");
    eprintln!("registered ggml devices:\n{list}");
    // A retro_cuda build always links ggml-cuda, so the device must register.
    // Whether it initializes is checked separately by cuda_available().
    assert!(
        common::cuda_registered(),
        "a CUDA build must register a CUDA GPU device:\n{list}"
    );
}

#[cfg(retro_cuda)]
#[test]
fn silu_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let shape = [1000, 1, 1, 1];
    compare_binary_op(
        ProbeOp::SiluBack,
        shape,
        &pseudo_random(1000, 0x1234, 1.0),
        &pseudo_random(1000, 0x5678, 6.0),
        [0.0, 0.0],
        1.0e-5,
    );
}

#[cfg(retro_cuda)]
#[test]
fn rms_norm_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let shape = [100, 4, 1, 1];
    compare_binary_op(
        ProbeOp::RmsNormBack,
        shape,
        &pseudo_random(400, 0x2234, 1.0),
        &pseudo_random(400, 0x6678, 2.0),
        [1.0e-5, 0.0],
        2.0e-4,
    );
}

#[cfg(retro_cuda)]
#[test]
fn l2_norm_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let shape = [100, 4, 1, 1];
    compare_binary_op(
        ProbeOp::L2NormBack,
        shape,
        &pseudo_random(400, 0x2244, 1.0),
        &pseudo_random(400, 0x6688, 2.0),
        [1.0e-5, 0.0],
        2.0e-4,
    );
}

#[cfg(retro_cuda)]
#[test]
fn soft_max_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let shape = [128, 5, 1, 1];
    let dy = pseudo_random(640, 0x3234, 1.0);
    let y = softmax_rows(&pseudo_random(640, 0x7678, 3.0), 128);
    compare_binary_op(
        ProbeOp::SoftMaxBack,
        shape,
        &dy,
        &y,
        [0.088_388_35, 0.0],
        1.0e-5,
    );
}

#[cfg(retro_cuda)]
#[test]
fn get_rows_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let grad_shape = [32, 4, 1, 1];
    let index_shape = [4, 1, 1, 1];
    let output_shape = [32, 7, 1, 1];
    let grad = pseudo_random(128, 0x4234, 1.0);
    let indices = [1.0, 5.0, 1.0, 3.0];
    let output_template = vec![0.0; 32 * 7];
    let cpu = probe_op(
        ProbeOp::GetRowsBack,
        false,
        ProbeInputs::pair(grad_shape, &grad, index_shape, &indices)
            .with_src2(Some((output_shape, &output_template))),
        [0.0, 0.0],
        output_template.len(),
    )
    .expect("CPU get_rows_back probe");
    let cuda = probe_op(
        ProbeOp::GetRowsBack,
        true,
        ProbeInputs::pair(grad_shape, &grad, index_shape, &indices)
            .with_src2(Some((output_shape, &output_template))),
        [0.0, 0.0],
        output_template.len(),
    )
    .expect("CUDA get_rows_back probe");
    assert_close(ProbeOp::GetRowsBack, &cpu, &cuda, 1.0e-6);
}

// dst[i0,i1] = Sum_k src0[i0,k] * src1[i1,k]; src0 is [ne00,k], src1 is [ne10,k].
#[cfg(retro_cuda)]
fn compare_out_prod(ne00: i64, k: i64, ne10: i64, tolerance: f32) {
    let ne_src0 = [ne00, k, 1, 1];
    let ne_src1 = [ne10, k, 1, 1];
    let src0 = pseudo_random((ne00 * k) as usize, 0x0a11, 1.0);
    let src1 = pseudo_random((ne10 * k) as usize, 0xb0b0, 1.0);
    let out_len = (ne00 * ne10) as usize;
    let cpu = probe_op(
        ProbeOp::OutProd,
        false,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CPU out_prod probe");
    let cuda = probe_op(
        ProbeOp::OutProd,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CUDA out_prod probe");
    assert_close(ProbeOp::OutProd, &cpu, &cuda, tolerance);
}

#[cfg(retro_cuda)]
#[test]
fn out_prod_f32_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // ggml-cuda routes F32 OUT_PROD through cuBLAS SGEMM, compiled with
    // -use_fast_math; its fused-multiply-add accumulation order differs from the
    // scalar CPU reference, so a k=48 reduction lands ~2e-3 away on the worst
    // element (relative error ~1e-3). A per-op tolerance of 5e-3 covers that
    // without hiding a real kernel regression.
    compare_out_prod(64, 48, 32, 5.0e-3);
}

// dst[i0,i1] = Sum_k dequant(src0[i0,k]) * src1[i1,k]; `op` selects the quantized
// src0 kernel. src0 is passed as F32 and quantized internally, so both backends
// dequantize identical blocks (J4.3: quantized weight gradient during LoRA).
#[cfg(retro_cuda)]
fn compare_out_prod_quant(op: ProbeOp, ne_src0: [i64; 4], ne_src1: [i64; 4], tolerance: f32) {
    let src0 = pseudo_random(
        ne_src0.iter().product::<i64>() as usize,
        0x0a11 ^ op as u64 ^ ne_src0[0] as u64,
        1.0,
    );
    let src1 = pseudo_random(
        ne_src1.iter().product::<i64>() as usize,
        0xb0b0 ^ op as u64 ^ ne_src1[2] as u64,
        1.0,
    );
    let out_len = (ne_src0[0] * ne_src1[0] * ne_src1[2] * ne_src1[3]) as usize;
    let cpu = probe_op(
        op,
        false,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CPU quantized out_prod probe");
    let cuda = probe_op(
        op,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CUDA quantized out_prod probe");
    assert_close(op, &cpu, &cuda, tolerance);
}

#[cfg(retro_cuda)]
#[test]
fn out_prod_q8_0_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // ne00 must be a multiple of the Q8_0 block size (32).
    compare_out_prod_quant(ProbeOp::OutProdQ80, [64, 40, 1, 1], [16, 40, 1, 1], 2.0e-3);
}

/// Small OUT_PROD reductions still decode into the shared-memory tile. The
/// scratch budget must remain inert there, even when the weight is large enough
/// that dequantize+SGEMM would split it.
#[cfg(retro_cuda)]
#[test]
fn out_prod_quant_cuda_native_is_independent_of_the_dequant_budget() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    common::assert_out_prod_quant_budget_independent(true, "cuda");
}

#[cfg(retro_cuda)]
#[test]
fn out_prod_k_quants_cuda_match_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    for op in [
        ProbeOp::OutProdQ2K,
        ProbeOp::OutProdQ3K,
        ProbeOp::OutProdQ4K,
        ProbeOp::OutProdQ5K,
        ProbeOp::OutProdQ6K,
    ] {
        // One partial output tile, then two 256-value K blocks over two batches.
        compare_out_prod_quant(op, [256, 7, 1, 1], [11, 7, 1, 1], 2.0e-3);
        compare_out_prod_quant(op, [512, 9, 2, 1], [5, 9, 2, 1], 2.0e-3);
    }
}

/// Every type in the fork's decodable-types table, not a hand-picked subset. See
/// `common::assert_out_prod_all_dequant_types_match_cpu`. F16 is the entry worth
/// watching: it is neither F32 nor `ggml_is_quantized`, so a gate on either
/// would send every `out_prod` of a plain F16 GGUF back to the CPU.
#[cfg(retro_cuda)]
#[test]
fn out_prod_all_dequant_types_cuda_match_cpu() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    common::assert_out_prod_all_dequant_types_match_cpu("cuda");
    common::assert_out_prod_extra_types_match_cpu(true, "cuda");
}

#[cfg(retro_cuda)]
#[test]
fn cross_entropy_loss_forward_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let nc = 128_i64;
    let nr = 8_i64;
    let logits = pseudo_random((nc * nr) as usize, 0xce10, 3.0);
    let mut labels = vec![0.0_f32; (nc * nr) as usize];
    for row in 0..nr {
        let hot = (row as usize * 11 + 5) % nc as usize;
        labels[row as usize * nc as usize + hot] = 1.0;
    }
    let shape = [nc, nr, 1, 1];
    let cpu = probe_op(
        ProbeOp::CrossEntropyLoss,
        false,
        ProbeInputs::pair(shape, &logits, shape, &labels),
        [0.0, 0.0],
        1,
    )
    .expect("CPU cross_entropy_loss forward probe");
    let cuda = probe_op(
        ProbeOp::CrossEntropyLoss,
        true,
        ProbeInputs::pair(shape, &logits, shape, &labels),
        [0.0, 0.0],
        1,
    )
    .expect("CUDA cross_entropy_loss forward probe");
    assert_close(ProbeOp::CrossEntropyLoss, &cpu, &cuda, 1.0e-4);
}

#[cfg(retro_cuda)]
#[test]
fn cross_entropy_loss_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let nc = 64_i64;
    let nr = 4_i64;
    let logits = pseudo_random((nc * nr) as usize, 0xce11, 3.0);
    let mut labels = vec![0.0_f32; (nc * nr) as usize];
    for row in 0..nr {
        let hot = (row as usize * 7 + 3) % nc as usize;
        labels[row as usize * nc as usize + hot] = 1.0;
    }
    let grad = [1.5_f32];
    let out_len = (nc * nr) as usize;
    let cpu = probe_op(
        ProbeOp::CrossEntropyLossBack,
        false,
        ProbeInputs::pair([1, 1, 1, 1], &grad, [nc, nr, 1, 1], &logits)
            .with_src2(Some(([nc, nr, 1, 1], &labels))),
        [0.0, 0.0],
        out_len,
    )
    .expect("CPU cross_entropy_loss_back probe");
    let cuda = probe_op(
        ProbeOp::CrossEntropyLossBack,
        true,
        ProbeInputs::pair([1, 1, 1, 1], &grad, [nc, nr, 1, 1], &logits)
            .with_src2(Some(([nc, nr, 1, 1], &labels))),
        [0.0, 0.0],
        out_len,
    )
    .expect("CUDA cross_entropy_loss_back probe");
    assert_close(ProbeOp::CrossEntropyLossBack, &cpu, &cuda, 1.0e-5);
}

// Differentiable Flash Attention backward (J4.1): packed dQ/dK/dV against the
// streaming CPU reference. K/V are rounded to F16 as in production.
#[cfg(retro_cuda)]
fn compare_flash_attn_back(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    causal: bool,
    tolerance: f32,
) {
    compare_flash_attn_back_window(
        hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, causal, None, tolerance,
    );
}

// `window`, when Some((nwin, kv_off)), exercises the KV gradient window: dK/dV
// cover `nwin` cache rows starting at `kv_off` (per batch), instead of all
// `nkv`. This is the case n_kv > n_window that the dense tests never reach.
#[cfg(retro_cuda)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_window(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    causal: bool,
    window: Option<(usize, usize)>,
    tolerance: f32,
) {
    let n_q = hsk * nq * nhead * nbatch;
    let n_k = hsk * nkv * nhead_kv * nbatch;
    let n_v = hsv * nkv * nhead_kv * nbatch;
    let n_do = hsv * nhead * nq * nbatch;
    let mut packed = Vec::with_capacity(n_q + n_k + n_v + n_do);
    packed.extend(pseudo_random(n_q, 0xf1a5_0001, 0.4));
    packed.extend(pseudo_random(n_k, 0xf1a5_0002, 0.4));
    packed.extend(pseudo_random(n_v, 0xf1a5_0003, 0.4));
    packed.extend(pseudo_random(n_do, 0xf1a5_0004, 0.7));

    let dims = [hsk as i64, hsv as i64, nq as i64, nkv as i64];
    let layout = [nhead as i64, nhead_kv as i64, nbatch as i64, causal as i64];
    let layout_data = [0.0f32; 4];
    let params = [1.0 / (hsk as f32).sqrt(), 0.0];

    // src2 carries the window: ne_src2[0] = nwin, data = per-batch cache rows.
    // stride == nkv, stream0 == 0 (see the probe), so row r of batch ib maps to
    // index r + nkv*ib.
    let (nwin, idx_data) = match window {
        Some((nwin, kv_off)) => {
            let mut idxs = Vec::with_capacity(nwin * nbatch);
            for ib in 0..nbatch {
                for j in 0..nwin {
                    idxs.push((kv_off + j + nkv * ib) as f32);
                }
            }
            (nwin, idxs)
        }
        None => (nkv, Vec::new()),
    };
    let out_len = n_q + hsk * nwin * nhead_kv * nbatch + hsv * nwin * nhead_kv * nbatch;
    let src2 = window.map(|(nwin, _)| ([nwin as i64, 0, 0, 0], idx_data.as_slice()));

    let cpu = probe_op(
        ProbeOp::FlashAttnBack,
        false,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("streaming CPU Flash Attention backward reference");
    let cuda = probe_op(
        ProbeOp::FlashAttnBack,
        true,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("CUDA Flash Attention backward probe");
    assert_close(ProbeOp::FlashAttnBack, &cpu, &cuda, tolerance);

    // The retained scalar F32-accumulation path is the comparison/recovery
    // switch for F2 even when the production cache itself is F16.
    let scalar = {
        let _env = common::EnvGuard::set("GGML_CUDA_FA_BACK_MMA", "0");
        probe_op(
            ProbeOp::FlashAttnBack,
            true,
            ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
            params,
            out_len,
        )
        .expect("scalar CUDA Flash Attention backward probe")
    };
    assert_close(ProbeOp::FlashAttnBack, &scalar, &cuda, tolerance);

    // The kernel rounds Q/dO to F16 before MMA, so repeated execution is a
    // tolerance contract rather than a bit-level one. Every dK/dV row still has
    // one owner, which avoids atomic-order drift.
    let repeated = probe_op(
        ProbeOp::FlashAttnBack,
        true,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("repeated CUDA Flash Attention backward probe");
    assert_close(ProbeOp::FlashAttnBack, &cuda, &repeated, tolerance);
}

#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_matches_streaming_cpu_reference() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // Non-causal, GQA (nhead != nhead_kv), nq != nkv. Head dim 64: the probe
    // builds a real ggml_flash_attn_ext forward dependency, and CUDA's FA forward
    // supports head dims 64/128 (not 32), so we exercise the backward on the head
    // dimensions used in production rather than the Vulkan test's 32.
    compare_flash_attn_back(64, 64, 5, 8, 4, 2, 1, false, 1.0e-3);
}

#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_covers_causal_gqa_and_head_dims() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // causal, multi-head/batch, head dims 64 and 128, MHA and GQA.
    compare_flash_attn_back(64, 64, 6, 6, 4, 4, 2, true, 1.0e-3);
    compare_flash_attn_back(128, 128, 4, 10, 8, 2, 1, true, 1.5e-3);
    compare_flash_attn_back(64, 64, 8, 8, 6, 3, 1, false, 1.0e-3);
}

#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_covers_wide_head_dims() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // Head dim 256 (Qwen3.5/GDN, Gemma2) selects the deeper register bucket in
    // flash-attn-back.cu. Before the bucket dispatch the supports check rejected
    // anything above 128, so these models silently trained on the materialized
    // F32 backward graph and `kv_dtype = "f16"` was a no-op for them.
    compare_flash_attn_back(256, 256, 4, 8, 4, 2, 1, true, 2.0e-3);
    compare_flash_attn_back(256, 256, 5, 5, 2, 2, 2, false, 2.0e-3);
}

// Head dim 512: Gemma-4's global-attention layers (1 in 6, `key_length = 512`
// against `key_length_swa = 256`). The FA_BACK_REGS_XWIDE bucket covers them;
// without it a single such layer failed the model-wide capability probe and cost
// Gemma-4 the fused path *and* the F16 KV cache on every layer.
//
// The shapes here are constrained by the *forward* pass this probe builds as a
// dependency, not by the backward kernel: CUDA only offers head dims above 256
// when its `gqa_opt_applies` holds, i.e. n_head/n_head_kv >= 2 and
// n_kv % FATTN_KQ_STRIDE (256) == 0. Gemma-4's global layers are 16 heads over 1
// KV head against a cache padded to 256, so they satisfy it; a test at
// nkv = 8 would abort in the forward instead of exercising this bucket.
#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_covers_xwide_head_dims() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // Causal GQA, then Gemma-4's own 16:1 head ratio over two streams. Tolerance
    // steps once more from the 256 bucket: twice the head dimension is twice the
    // accumulated warp-shuffle-reduction error per dot product.
    compare_flash_attn_back(512, 512, 4, 256, 4, 2, 1, true, 3.0e-3);
    compare_flash_attn_back(512, 512, 3, 256, 16, 1, 2, false, 3.0e-3);
    // The KV gradient window was never exercised at this head dimension (nothing
    // in the window logic is head-dim-specific, but that is an argument for
    // testing it, not for assuming it).
    compare_flash_attn_back_window(512, 512, 4, 256, 4, 2, 1, false, Some((4, 12)), 3.0e-3);
}

// KV gradient window: dK/dV cover only the rows written at this step, so n_kv
// (what the forward reads) exceeds n_window (what receives a gradient). This is
// the shape production hits at a long context and the dense tests never do --
// including kv_off > 0, which catches a window indexed off by a row.
#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_covers_kv_gradient_window() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // Non-causal, GQA, window strictly inside a larger cache, offset window.
    compare_flash_attn_back_window(64, 64, 4, 32, 4, 2, 1, false, Some((4, 0)), 1.0e-3);
    compare_flash_attn_back_window(64, 64, 4, 32, 4, 2, 1, false, Some((4, 12)), 1.0e-3);
    // Multi-stream (nbatch > 1) with a per-stream offset window, wide head dim.
    compare_flash_attn_back_window(256, 256, 5, 40, 4, 2, 2, false, Some((5, 7)), 2.0e-3);
    // Degenerate case: window == whole cache must match the dense path exactly.
    compare_flash_attn_back_window(64, 64, 6, 6, 4, 2, 1, true, Some((6, 0)), 1.0e-3);
}

/// Shapes that exercise what the shared-memory K/V tiling changed.
///
/// The backward now runs one block per (query tile, head, batch) with the block's
/// warps cooperating on a K/V tile, instead of one flat warp per query row reading
/// K/V from global. Three things moved and each has a shape that catches it:
///
/// * `nq` below the warps per block, and `nq` not a multiple of them, leave part of
///   the block with no query row. Those warps must still reach every barrier and
///   contribute nothing - the failure mode is a hang or a corrupted tile, not a
///   small numeric drift.
/// * `nkv` not a multiple of the tile depth exercises the partial trailing tile.
/// * A wider head dimension shrinks the tile depth, so the tile loop runs more,
///   shorter iterations. Head dim 512 drops it to the floor of 8 but cannot show a
///   partial tile there (see the case below); 256 gives depth 16, which can.
///
/// Tolerances match the dense tests above on purpose. The tiling preserves the
/// order of the online-softmax recurrence and of the dQ accumulation, so it has no
/// licence to be less accurate; a widened tolerance here would mean the rewrite
/// changed the arithmetic rather than just where the operands are read from.
#[cfg(retro_cuda)]
#[test]
fn flash_attn_back_cuda_covers_kv_tile_boundaries() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // nq = 1: a single active warp in a block of eight.
    compare_flash_attn_back(64, 64, 1, 40, 4, 2, 1, false, 1.0e-3);
    // nq just over one block, so the second block is almost entirely inactive.
    compare_flash_attn_back(64, 64, 9, 33, 4, 2, 1, true, 1.0e-3);
    // nq well past a block, several full query tiles.
    compare_flash_attn_back(64, 64, 40, 48, 4, 2, 1, true, 1.0e-3);
    // nkv one past a tile boundary at the shallowest depth a partial tile can
    // actually be reached at. Tile depth is 32*1024/((hsk+hsv)*4) capped at 32, so
    // 32 rows at head dim 64/128, 16 at 256 and 8 at 512. The shallowest bucket
    // (512) cannot reach a partial tile at all: the forward this probe builds only
    // offers head dims above 256 when n_kv % FATTN_KQ_STRIDE (256) == 0, and 256 is
    // a multiple of 8, so the trailing tile there is always full -- the nkv = 256
    // cases in `covers_xwide_head_dims` are the whole story at that head dim. Depth
    // 16 is therefore the real boundary case, and head dim 256 takes no gqa_opt
    // detour, so nkv is free: 33 = 2*16 + 1 leaves a one-row trailing tile.
    compare_flash_attn_back(256, 256, 3, 33, 2, 1, 1, false, 2.0e-3);
    // Inactive warps combined with a gradient window, where the atomics into
    // dK/dV must stay confined to the warps that actually own a query row.
    compare_flash_attn_back_window(64, 64, 1, 32, 4, 2, 1, false, Some((4, 12)), 1.0e-3);
    compare_flash_attn_back_window(64, 64, 9, 40, 4, 2, 2, false, Some((5, 7)), 1.0e-3);
}

// F16 AdamW (J4.4): the F16 store uses the fork's stochastic rounding, which the
// CUDA kernel reproduces bit-for-bit, so it must match the CPU oracle exactly.
#[cfg(retro_cuda)]
#[test]
fn f16_adamw_cuda_kernel_matches_cpu() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let weights = [0.5, -1.0, 2.0, -0.25, 0.125];
    let gradients = [0.25, -0.5, 1.0, -2.0, 0.75];
    let run = |gpu| {
        probe_op(
            ProbeOp::OptStepAdamwF16,
            gpu,
            ProbeInputs::pair([5, 1, 1, 1], &weights, [5, 1, 1, 1], &gradients),
            [1.0e-2, 0.1],
            5,
        )
        .expect("run F16 AdamW probe")
    };
    let cpu = run(false);
    let gpu = run(true);
    assert_eq!(cpu, gpu, "CUDA F16 AdamW must match CPU after F16 rounding");
}

// The same contract at 8 significand bits: BF16 rounds on its own grid
// through its own conversion.
#[cfg(retro_cuda)]
#[test]
fn bf16_adamw_cuda_kernel_matches_cpu() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let weights = [0.5, -1.0, 2.0, -0.25, 0.125];
    let gradients = [0.25, -0.5, 1.0, -2.0, 0.75];
    let run = |gpu| {
        probe_op(
            ProbeOp::OptStepAdamwBf16,
            gpu,
            ProbeInputs::pair([5, 1, 1, 1], &weights, [5, 1, 1, 1], &gradients),
            [1.0e-2, 0.1],
            5,
        )
        .expect("run BF16 AdamW probe")
    };
    let cpu = run(false);
    let gpu = run(true);
    assert_eq!(
        cpu, gpu,
        "CUDA BF16 AdamW must match CPU after BF16 rounding"
    );
}

// One step shares a single rounding seed; this moves the seed the way a run
// does, so the rounding formula, not the arithmetic, decides the result.
#[cfg(retro_cuda)]
#[test]
fn bf16_adamw_cuda_matches_cpu_across_chained_steps() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let n = 256;
    // Updates well below half a BF16 ulp at 1.0 (2^-8), so every step is
    // decided purely by the stochastic rounding.
    let alpha = 1.0e-5_f32;
    let steps = 64;
    let gradients = vec![1.0_f32; n];

    let chained = |gpu: bool| {
        let ne = [n as i64, 1, 1, 1];
        let mut current = vec![1.0_f32; n];
        for step in 0..steps {
            let scale_and_seed = [1.0_f32, step as f32];
            current = probe_op(
                ProbeOp::OptStepAdamwBf16,
                gpu,
                ProbeInputs::pair(ne, &current, ne, &gradients)
                    .with_src2(Some(([2, 1, 1, 1], scale_and_seed.as_slice()))),
                [alpha, 0.0],
                n,
            )
            .expect("run chained BF16 AdamW probe");
        }
        current
    };

    let cpu = chained(false);
    let gpu = chained(true);
    assert!(
        cpu.iter().any(|value| *value != 1.0),
        "the fixture rounded every update away; it proves nothing"
    );
    assert_eq!(
        cpu, gpu,
        "CUDA BF16 AdamW must track the CPU's stochastic rounding step for step"
    );
}

// The same contract for SGD, whose store is the rounding itself: it keeps no
// moments.
#[cfg(retro_cuda)]
#[test]
fn half_precision_sgd_cuda_kernel_matches_cpu() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let weights = [0.5, -1.0, 2.0, -0.25, 0.125];
    let gradients = [0.25, -0.5, 1.0, -2.0, 0.75];
    for op in [ProbeOp::OptStepSgdF16, ProbeOp::OptStepSgdBf16] {
        let run = |gpu| {
            probe_op(
                op,
                gpu,
                ProbeInputs::pair([5, 1, 1, 1], &weights, [5, 1, 1, 1], &gradients),
                [1.0e-2, 0.1],
                5,
            )
            .expect("run the half-precision SGD probe")
        };
        assert_eq!(
            run(false),
            run(true),
            "CUDA {op:?} must match CPU after the rounded store"
        );
    }
}

// The same moving-seed chain for SGD.
#[cfg(retro_cuda)]
#[test]
fn half_precision_sgd_cuda_matches_cpu_across_chained_steps() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let n = 256;
    // Below half a BF16 ulp at 1.0 (2^-8), so every step is decided purely by
    // the stochastic rounding on either grid.
    let alpha = 1.0e-5_f32;
    let steps = 64;
    let gradients = vec![1.0_f32; n];

    for op in [ProbeOp::OptStepSgdF16, ProbeOp::OptStepSgdBf16] {
        let chained = |gpu: bool| {
            let ne = [n as i64, 1, 1, 1];
            let mut current = vec![1.0_f32; n];
            for step in 0..steps {
                let scale_and_seed = [1.0_f32, step as f32];
                current = probe_op(
                    op,
                    gpu,
                    ProbeInputs::pair(ne, &current, ne, &gradients)
                        .with_src2(Some(([2, 1, 1, 1], scale_and_seed.as_slice()))),
                    [alpha, 0.0],
                    n,
                )
                .expect("run the chained half-precision SGD probe");
            }
            current
        };
        let cpu = chained(false);
        assert!(
            cpu.iter().any(|value| *value != 1.0),
            "the fixture rounded every update away; it proves nothing"
        );
        assert_eq!(
            cpu,
            chained(true),
            "CUDA {op:?} must track the CPU's stochastic rounding step for step"
        );
    }
}

// SSM backward (J4.2) for recurrent models, validated against the CPU oracle.
#[cfg(retro_cuda)]
#[test]
fn ssm_conv_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let d_conv = 4_i64;
    let n_t = 6_i64;
    let ncs = d_conv - 1 + n_t;
    let d_inner = 5_i64;
    let n_s = 2_i64;
    let ne_sx = [ncs, d_inner, n_s, 1];
    let ne_c = [d_conv, d_inner, 1, 1];
    let ne_dy = [d_inner, n_t, n_s, 1];
    let sx = pseudo_random((ncs * d_inner * n_s) as usize, 0x5c04, 1.0);
    let c = pseudo_random((d_conv * d_inner) as usize, 0x5c05, 1.0);
    let dy = pseudo_random((d_inner * n_t * n_s) as usize, 0x5c06, 1.0);
    let out_len = (ncs * d_inner * n_s + d_conv * d_inner) as usize;
    let cpu = probe_op(
        ProbeOp::SsmConvBack,
        false,
        ProbeInputs::pair(ne_sx, &sx, ne_c, &c).with_src2(Some((ne_dy, &dy))),
        [0.0, 0.0],
        out_len,
    )
    .expect("CPU ssm_conv_back probe");
    let cuda = probe_op(
        ProbeOp::SsmConvBack,
        true,
        ProbeInputs::pair(ne_sx, &sx, ne_c, &c).with_src2(Some((ne_dy, &dy))),
        [0.0, 0.0],
        out_len,
    )
    .expect("CUDA ssm_conv_back probe");
    assert_close(ProbeOp::SsmConvBack, &cpu, &cuda, 1.0e-5);
}

#[cfg(retro_cuda)]
#[test]
fn ssm_scan_back_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // n_a0 == 1 is Mamba2's scalar per-head A; n_a0 == nc is Mamba1's per-channel
    // A. They are separate branches in the kernel, so both must be exercised.
    ssm_scan_back_case(1);
    ssm_scan_back_case(16);
}

#[cfg(retro_cuda)]
fn conv_rs_gather_case(kernel_m1: i64, n_seq_tokens: i64, k: i64) {
    let n_channels = 5_i64;
    let n_seqs = 3_i64;
    let ne0 = kernel_m1 + n_seq_tokens;
    let ne_src0 = [ne0, n_channels, n_seqs, 1];
    let src0 = pseudo_random((ne0 * n_channels * n_seqs) as usize, 0x5c07, 1.0);
    // src1 is ignored by the op but the probe ABI requires a non-null buffer.
    let ne_src1 = [1_i64, 1, 1, 1];
    let src1 = vec![0.0_f32];
    let out_len = (kernel_m1 * n_channels * n_seqs * k) as usize;
    let params = [kernel_m1 as f32, k as f32];
    let cpu = probe_op(
        ProbeOp::ConvRsGather,
        false,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        params,
        out_len,
    )
    .expect("CPU conv_rs_gather probe");
    let cuda = probe_op(
        ProbeOp::ConvRsGather,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        params,
        out_len,
    )
    .expect("CUDA conv_rs_gather probe");
    // A pure gather: the two backends must agree bit for bit.
    assert_close(ProbeOp::ConvRsGather, &cpu, &cuda, 0.0);
}

#[cfg(retro_cuda)]
#[test]
fn conv_rs_gather_cuda_matches_cpu_in_isolation() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // K == 1 is the single-snapshot case; K > 1 walks the overlapping windows.
    conv_rs_gather_case(3, 8, 1);
    conv_rs_gather_case(3, 8, 5);
    // n_seq_tokens < K exercises the clamp that collapses the trailing slots
    // onto slot 0's window.
    conv_rs_gather_case(3, 2, 6);
}

#[cfg(retro_cuda)]
fn ssm_scan_back_case(n_a0: i64) {
    let (nc, nr, nh, nslot) = (16_i64, 8_i64, 4_i64, 2_i64);
    // Forty tokens cross the 32-token chunk boundary of the two-pass reference.
    let (ng, nt, ns) = (2_i64, 40_i64, 2_i64);
    assert!(n_a0 == 1 || n_a0 == nc);
    let ne_src0 = [nc, nr, nh, nslot];
    let ne_src1 = [ng, nt, ns, n_a0];

    let n_s = nc * nr * nh * nslot;
    let n_x = nr * nh * nt * ns;
    let n_dt = nh * nt * ns;
    let n_a = n_a0 * nh;
    let n_b = nc * ng * nt * ns;
    let n_c = nc * ng * nt * ns;
    let n_ds = nr * nh * nt * ns + nc * nr * nh * ns;

    let mut packed = Vec::new();
    packed.extend(pseudo_random(n_s as usize, 0x5501, 0.5));
    packed.extend(pseudo_random(n_x as usize, 0x5502, 0.5));
    packed.extend(pseudo_random(n_dt as usize, 0x5503, 0.5));
    packed.extend(
        pseudo_random(n_a as usize, 0x5504, 0.5)
            .iter()
            .map(|v| -v.abs()),
    );
    packed.extend(pseudo_random(n_b as usize, 0x5505, 0.5));
    packed.extend(pseudo_random(n_c as usize, 0x5506, 0.5));
    packed.extend((0..ns).map(|i| i as f32));
    packed.extend(pseudo_random(n_ds as usize, 0x5507, 0.5));

    let out_len = (n_x + n_dt + n_a + n_b + n_c + n_s) as usize;
    let dummy_src1 = vec![0.0_f32; (ng * nt * ns * n_a0) as usize];
    let cpu = probe_op(
        ProbeOp::SsmScanBack,
        false,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CPU ssm_scan_back probe");
    let cuda = probe_op(
        ProbeOp::SsmScanBack,
        true,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("CUDA ssm_scan_back probe");
    assert_close(ProbeOp::SsmScanBack, &cpu, &cuda, 5.0e-3);
}

/// One gated-delta-net backward shape. `gate` scales the log-gate, which is the
/// input that decides what the chunkwise CUDA path actually does: it
/// normalises by the cumulative decay, so a gate
/// strong enough to put a chunk's cumulative decay past the F32 bound sends that
/// (chunk, unit) pair to the per-token fallback instead. Every case is therefore
/// a gate case as much as a shape case.
///
/// `spike` overrides the gate of a chosen head, in the *first* half of the
/// tokens only, with a value nothing can normalise. That is how a node gets a
/// mixed layout -- some (chunk, unit) pairs on the batched path, some on the
/// fallback, in both directions of the state chain.
#[cfg(retro_cuda)]
#[derive(Clone, Copy, Debug)]
struct GdnCase {
    kda: bool,
    k: i64,
    s_v: i64,
    h: i64,
    n_tokens: i64,
    n_seqs: i64,
    gate: f32,
    /// (head, log-gate) forced on the first half of that head's tokens.
    spike: Option<(i64, f32)>,
    beta: Option<f32>,
}

#[cfg(retro_cuda)]
impl GdnCase {
    fn new(n_tokens: i64) -> Self {
        Self {
            kda: false,
            k: 1,
            s_v: 32,
            h: 4,
            n_tokens,
            n_seqs: 2,
            gate: 0.3,
            spike: None,
            beta: None,
        }
    }

    fn packed(&self) -> (Vec<f32>, [i64; 4], [i64; 4], Vec<(&'static str, usize)>) {
        let Self {
            kda,
            k,
            s_v,
            h,
            n_tokens,
            n_seqs,
            gate,
            spike,
            beta,
        } = *self;
        let n_qkv = s_v * h * n_tokens * n_seqs;
        let n_g = (if kda { s_v } else { 1 }) * h * n_tokens * n_seqs;
        let n_beta = h * n_tokens * n_seqs;
        let n_state = s_v * s_v * h * n_seqs;
        let n_grad = n_qkv + k * n_state;

        let mut packed = Vec::new();
        packed.extend(pseudo_random(n_qkv as usize, 0x6601, 0.5)); // q
        packed.extend(pseudo_random(n_qkv as usize, 0x6602, 0.5)); // k
        packed.extend(pseudo_random(n_qkv as usize, 0x6603, 0.5)); // v
        let g_width = if kda { s_v } else { 1 };
        let mut gates: Vec<f32> = pseudo_random(n_g as usize, 0x6604, 1.0)
            .iter()
            .map(|v| -v.abs() * gate - 0.02 * gate) // log-gate, decaying
            .collect();
        if let Some((spike_head, spike_gate)) = spike {
            // g is [g_width, h, n_tokens, n_seqs].
            for i_seq in 0..n_seqs {
                for t in 0..n_tokens / 2 {
                    let row = (i_seq * n_tokens + t) * h + spike_head;
                    for c in 0..g_width {
                        gates[(row * g_width + c) as usize] = spike_gate;
                    }
                }
            }
        }
        packed.extend(gates);
        packed.extend(
            pseudo_random(n_beta as usize, 0x6605, 0.5)
                .iter()
                .map(|v| beta.unwrap_or(v.abs())), // non-negative mixing coefficient
        );
        packed.extend(pseudo_random(n_state as usize, 0x6606, 0.5)); // s0
        packed.extend(pseudo_random(n_grad as usize, 0x6607, 0.5)); // upstream grad

        let blocks = vec![
            ("grad_q", n_qkv as usize),
            ("grad_k", n_qkv as usize),
            ("grad_v", n_qkv as usize),
            ("grad_g", n_g as usize),
            ("grad_beta", n_beta as usize),
            ("grad_state", n_state as usize),
        ];
        (
            packed,
            [s_v, h, n_tokens, n_seqs],
            [k, if kda { 1 } else { 0 }, 1, 1],
            blocks,
        )
    }
}

/// `chunk` pins the formulation, exactly as `ggml_gated_delta_net_back_chunked`
/// documents it: negative for the sequential scan, positive for the chunkwise
/// form with that chunk length, zero for the backend default.
#[cfg(retro_cuda)]
fn gdn_probe(case: &GdnCase, cuda: bool, chunk: f32) -> Vec<f32> {
    let (packed, ne_src0, ne_src1, blocks) = case.packed();
    let out_len = blocks.iter().map(|(_, n)| n).sum();
    let dummy_src1 = vec![0.0_f32; 1];
    probe_op(
        ProbeOp::GatedDeltaNetBack,
        cuda,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [chunk, 0.0],
        out_len,
    )
    .expect("gated_delta_net_back probe")
}

/// Compares gradient block by gradient block, scaled by the reference's own
/// magnitude. The two formulations group the same sums differently and cuBLAS
/// picks its own order on top of that, so the contract is relative agreement;
/// reporting per block is what turns a failure into a diagnosis, since a wrong
/// adjoint lands in one gradient and a wrong chunk layout in `grad_state`.
#[cfg(retro_cuda)]
fn gdn_assert_agrees(case: &GdnCase, reference: &[f32], actual: &[f32], tolerance: f32) {
    let (_, _, _, blocks) = case.packed();
    assert_eq!(reference.len(), actual.len());
    let mut offset = 0;
    let mut worst = 0.0_f32;
    let mut worst_block = "";
    for (name, count) in blocks {
        let mut scale = 0.0_f32;
        let mut max_diff = 0.0_f32;
        let mut where_at = 0;
        for index in 0..count {
            let expected = reference[offset + index];
            let got = actual[offset + index];
            assert!(
                got.is_finite(),
                "{case:?}: {name}[{index}] is not finite: {got}"
            );
            scale = scale.max(expected.abs());
            if (expected - got).abs() > max_diff {
                max_diff = (expected - got).abs();
                where_at = index;
            }
        }
        let relative = max_diff / scale.max(1.0e-3);
        assert!(
            relative <= tolerance,
            "{case:?}: {name} differs at {where_at}: reference={}, actual={}, \
             |diff|={max_diff:e}, scale={scale:e}, relative={relative:e} > {tolerance:e}",
            reference[offset + where_at],
            actual[offset + where_at],
        );
        if relative > worst {
            worst = relative;
            worst_block = name;
        }
        offset += count;
    }
    eprintln!("{case:?}: worst relative diff {worst:e} ({worst_block})");
}

/// The CPU sequential scan is the oracle for both CUDA formulations; the CUDA
/// sequential kernel has to stay reachable and correct because it is also the
/// chunkwise path's numerical fallback.
#[cfg(retro_cuda)]
fn gdn_case(case: &GdnCase, tolerance: f32) {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let cpu = gdn_probe(case, false, -1.0);
    gdn_assert_agrees(case, &cpu, &gdn_probe(case, true, 0.0), tolerance);
}

#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_matches_cpu_scalar_gate_k1() {
    let mut case = GdnCase::new(20);
    case.gate = 2.0;
    gdn_case(&case, 1.0e-4);
}

#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_matches_cpu_kda_gate_with_snapshots() {
    let mut case = GdnCase::new(20);
    case.kda = true;
    case.k = 3;
    case.gate = 2.0;
    gdn_case(&case, 1.0e-4);
}

/// The default CUDA path is now the chunkwise one, so the whole-sequence
/// sequential kernel would rot unnoticed -- and both formulations now share the
/// same two per-token steps, so a break in it is a break in the chunkwise
/// path's numerical fallback too. `GGML_CUDA_GDN_BACK_CHUNK=0` selects it in
/// production; a negative chunk parameter is the same switch from a test.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_sequential_kernel_stays_a_tested_path() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let case = GdnCase::new(70);
    let cpu = gdn_probe(&case, false, -1.0);
    gdn_assert_agrees(&case, &cpu, &gdn_probe(&case, true, -1.0), 1.0e-4);
    // And the two CUDA formulations against each other, which is the comparison
    // that isolates the chunking from everything else about the backend.
    let sequential = gdn_probe(&case, true, -1.0);
    gdn_assert_agrees(&case, &sequential, &gdn_probe(&case, true, 64.0), 1.0e-4);
}

/// Token counts around the chunk length: a partial tail chunk is the first thing
/// a chunkwise kernel drops, and a single token is the degenerate layout.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_covers_chunk_boundaries() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    for n_tokens in [1, 2, 63, 64, 65, 127, 129] {
        let case = GdnCase::new(n_tokens);
        let cpu = gdn_probe(&case, false, -1.0);
        gdn_assert_agrees(&case, &cpu, &gdn_probe(&case, true, 64.0), 1.0e-4);
    }
    // A chunk longer than the sequence, and chunks of one token.
    for chunk in [4096.0, 1.0] {
        let case = GdnCase::new(40);
        let cpu = gdn_probe(&case, false, -1.0);
        gdn_assert_agrees(&case, &cpu, &gdn_probe(&case, true, chunk), 1.0e-4);
    }
}

/// One head and one sequence: the grid collapses to a single unit, so every
/// batched GEMM has batch 1 and every head/sequence stride is exercised at its
/// boundary value.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_handles_a_degenerate_grid() {
    let mut case = GdnCase::new(65);
    case.h = 1;
    case.n_seqs = 1;
    gdn_case(&case, 1.0e-4);
}

/// The gate regimes, from mildest to worst: no decay at all, decay the chunkwise
/// normalisation absorbs, and decay past the F32 bound, which sends every
/// (chunk, unit) pair to the per-token fallback. The last two are where a
/// missing guard would not produce a small error but an infinity.
/// `GGML_CUDA_GDN_BACK_DEBUG=1` shows which regime each case actually hit.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_survives_adverse_gates() {
    for gate in [0.0, 1.0, 4.0, 200.0] {
        let mut case = GdnCase::new(96);
        case.gate = gate;
        gdn_case(&case, 1.0e-4);
        let mut kda = case;
        kda.kda = true;
        gdn_case(&kda, 1.0e-4);
    }
}

/// A gate spike confined to one head and to the first half of the sequence: the
/// layout is global to the node, so those tokens shorten every unit's chunks
/// while the rest stay long, and the same node therefore carries both short and
/// full-length chunks. `gated_delta_net_back_cuda_survives_adverse_gates` only
/// ever makes the whole node uniform.
///
/// This is also the shape that rules out a device-side variant of the guard:
/// a per-unit check cannot shorten a chunk, only fall back.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_handles_a_gate_spike_on_one_head() {
    for kda in [false, true] {
        for n_tokens in [96, 129] {
            let mut case = GdnCase::new(n_tokens);
            case.kda = kda;
            case.spike = Some((2, -80.0));
            gdn_case(&case, 1.0e-4);
        }
    }
}

/// `beta = 0` freezes the state, making the `(I+T)` solve the identity; `beta = 1`
/// replaces the state outright, which is where that system is worst conditioned.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_covers_the_extremes_of_beta() {
    for beta in [0.0, 1.0] {
        let mut case = GdnCase::new(96);
        case.beta = Some(beta);
        gdn_case(&case, 1.0e-4);
    }
}

/// The shape the production profile was taken on:
/// `S_v = 128`, 16 value heads, 128 tokens per sequence. This is the only case
/// that runs the kernel at the head dimension production uses, where one gate
/// channel per thread fills half a block.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_matches_cpu_at_the_production_head_dim() {
    let mut case = GdnCase::new(128);
    case.s_v = 128;
    case.h = 16;
    case.n_seqs = 2;
    gdn_case(&case, 1.0e-4);
}

/// O3's BF16-input/F32-accumulator GEMMs are an explicitly different numerical
/// regime. Pin the accepted relative error here instead of weakening the F32
/// oracle cases above: 3e-2 is the mode's contract, while F32 stays at 1e-4.
/// The first sweep covers the 16/32/64/128-unit batches the optimization targets;
/// the second covers every supported chunk size. Pre-Ampere NVIDIA devices keep
/// the F32 path, so the same test remains valid there without pretending they
/// executed BF16 tensor-core instructions.
#[cfg(retro_cuda)]
#[test]
fn gated_delta_net_back_cuda_bf16_gemm_mode_has_bounded_error() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }

    const TOLERANCE: f32 = 3.0e-2;
    for n_seqs in [1, 2, 4, 8] {
        let mut case = GdnCase::new(33);
        case.s_v = 32;
        case.h = 16;
        case.n_seqs = n_seqs;
        let f32 = {
            let _mode = common::EnvGuard::set("GGML_CUDA_GDN_BACK_MMA", "0");
            gdn_probe(&case, true, 64.0)
        };
        let bf16 = {
            let _mode = common::EnvGuard::set("GGML_CUDA_GDN_BACK_MMA", "1");
            gdn_probe(&case, true, 64.0)
        };
        gdn_assert_agrees(&case, &f32, &bf16, TOLERANCE);
    }

    let mut case = GdnCase::new(65);
    case.s_v = 128;
    case.h = 16;
    case.n_seqs = 1;
    for chunk in [16.0, 32.0, 64.0, 128.0] {
        let f32 = {
            let _mode = common::EnvGuard::set("GGML_CUDA_GDN_BACK_MMA", "0");
            gdn_probe(&case, true, chunk)
        };
        let bf16 = {
            let _mode = common::EnvGuard::set("GGML_CUDA_GDN_BACK_MMA", "1");
            gdn_probe(&case, true, chunk)
        };
        gdn_assert_agrees(&case, &f32, &bf16, TOLERANCE);
    }
}

#[cfg(retro_cuda)]
#[test]
fn fused_sparse_ce_cuda_matches_cpu_for_float_and_quant_heads() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let _guard = common::serialize_models();
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    let h = pseudo_random(NE * NT, 0xCE01, 1.2);
    let w = pseudo_random(NE * NV, 0xCE02, 0.8);
    let targets = [3, -1, 17, 8, 299];
    let weights = [1.0_f32, 0.9, 0.0, -0.6, 1.4];

    for w_type in [
        FusedCeWeightType::F32,
        FusedCeWeightType::F16,
        FusedCeWeightType::Q8_0,
        FusedCeWeightType::Q4K,
        FusedCeWeightType::Q5K,
        FusedCeWeightType::Q6K,
    ] {
        let probe = fused_sparse_ce_probe(
            FusedCeProbeShape {
                n_embd: NE,
                n_tokens: NT,
                n_vocab: NV,
                n_tiles: 8,
                seq_chunk: 0,
                w_type,
            },
            true,
            FusedCeProbeInputs {
                h: &h,
                w: &w,
                targets: &targets,
                weights: &weights,
                bias: None,
            },
            1.5,
        )
        .expect("CUDA fused sparse CE probe");
        let loss_gap = (probe.loss_full - probe.loss_fused).abs();
        assert!(
            loss_gap < 5.0e-4,
            "{w_type:?}: loss gap {loss_gap}: CPU={} CUDA={}",
            probe.loss_full,
            probe.loss_fused,
        );
        let max_grad_gap = probe
            .grad_h_full
            .iter()
            .zip(&probe.grad_h_fused)
            .map(|(cpu, cuda)| (cpu - cuda).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_grad_gap < 3.0e-3,
            "{w_type:?}: max grad_h gap {max_grad_gap}"
        );
        eprintln!(
            "FUSED_SPARSE_CE {w_type:?}: loss gap={loss_gap:e}, max grad gap={max_grad_gap:e}"
        );
    }
}

/// The in-kernel decode of a quantized head is a second numerical regime for the
/// same operator: the dot products are accumulated in F32 by a shared-memory tile
/// instead of by cuBLAS, so the summation order differs and the gaps are pinned
/// here rather than by weakening the default path's case above. `n_embd` is a
/// whole number of 256-element tiles, which is the path's own precondition - a
/// head that is not falls back and would silently measure the default path.
///
/// The second half is the output that matters: with no logits and no
/// F32 view of the head, `chunked_ce_tiles` and `chunked_ce_seq_chunk` size
/// nothing, so the result must be *identical* across them, not merely close.
#[cfg(retro_cuda)]
#[test]
fn fused_sparse_ce_cuda_decodes_a_quantized_head_in_kernel() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    let _guard = common::serialize_models();
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    let h = pseudo_random(NE * NT, 0xCE11, 1.2);
    let w = pseudo_random(NE * NV, 0xCE12, 0.8);
    let targets = [3, -1, 17, 8, 299];
    let weights = [1.0_f32, 0.9, 0.0, -0.6, 1.4];
    let inputs = FusedCeProbeInputs {
        h: &h,
        w: &w,
        targets: &targets,
        weights: &weights,
        bias: None,
    };
    let shape = |n_tiles: usize, seq_chunk: usize, w_type: FusedCeWeightType| FusedCeProbeShape {
        n_embd: NE,
        n_tokens: NT,
        n_vocab: NV,
        n_tiles,
        seq_chunk,
        w_type,
    };

    for w_type in [
        FusedCeWeightType::F16,
        FusedCeWeightType::Q4_0,
        FusedCeWeightType::Q8_0,
        FusedCeWeightType::Q4K,
        FusedCeWeightType::Q5K,
        FusedCeWeightType::Q6K,
    ] {
        let _mode = common::EnvGuard::set("GGML_CUDA_CE_QHEAD", "1");
        let probe = fused_sparse_ce_probe(shape(8, 0, w_type), true, inputs, 1.5)
            .expect("CUDA in-kernel decode fused sparse CE probe");
        let loss_gap = (probe.loss_full - probe.loss_fused).abs();
        assert!(
            loss_gap < 5.0e-4,
            "{w_type:?}: loss gap {loss_gap}: full={} decode={}",
            probe.loss_full,
            probe.loss_fused,
        );
        let max_grad_gap = probe
            .grad_h_full
            .iter()
            .zip(&probe.grad_h_fused)
            .map(|(full, decode)| (full - decode).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_grad_gap < 3.0e-3,
            "{w_type:?}: max grad_h gap {max_grad_gap}"
        );

        // Every tiling knob of the default path is inert on this one.
        for (n_tiles, seq_chunk) in [(1, 0), (32, 0), (8, 1), (8, 2)] {
            let other = fused_sparse_ce_probe(shape(n_tiles, seq_chunk, w_type), true, inputs, 1.5)
                .expect("CUDA in-kernel decode fused sparse CE probe");
            assert_eq!(
                other.loss_fused, probe.loss_fused,
                "{w_type:?}: loss moved with n_tiles={n_tiles} seq_chunk={seq_chunk}"
            );
            assert_eq!(
                other.grad_h_fused, probe.grad_h_fused,
                "{w_type:?}: grad_h moved with n_tiles={n_tiles} seq_chunk={seq_chunk}"
            );
        }
        eprintln!(
            "FUSED_SPARSE_CE decode {w_type:?}: loss gap={loss_gap:e}, max grad gap={max_grad_gap:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Model-dependent placement/training tests. They require a small GGUF fixture
// (RETRO_CUDA_TEST_MODEL, defaulting to the in-repo CPU fixture) and skip when
// it is not present.
// ---------------------------------------------------------------------------

#[cfg(retro_cuda)]
fn small_config(device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-3,
        device,
        ..TrainConfig::default()
    }
}

#[cfg(retro_cuda)]
fn single_layer_lora() -> LoraConfig {
    let mut config = LoraConfig::qv(2, 4.0);
    config.seed = 7;
    config.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    config
}

#[cfg(retro_cuda)]
fn cuda_model_or_skip() -> Option<std::path::PathBuf> {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device available");
        return None;
    }
    let model = common::cuda_model_path_if_available();
    if model.is_none() {
        eprintln!(
            "skipping: no CUDA test model at {}",
            common::cuda_model_path().display()
        );
    }
    model
}

#[cfg(retro_cuda)]
#[test]
fn model_is_offloaded_to_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load model on CUDA");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("{report}");
    assert!(report.contains("gpu_active: true"), "{report}");
    assert!(report.contains("backend: CUDA"), "{report}");
    assert!(
        common::section_has_cuda(&report, "model_tensors_by_buffer"),
        "model tensors were not offloaded to CUDA:\n{report}"
    );
}

#[cfg(retro_cuda)]
#[test]
fn lora_tensors_are_allocated_on_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load model on CUDA");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create CUDA LoRA");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("{report}");
    assert!(
        common::section_has_cuda(&report, "lora_tensors_by_buffer"),
        "LoRA tensors were not allocated on CUDA:\n{report}"
    );
}

#[cfg(retro_cuda)]
#[test]
fn training_preflight_reports_cuda_device_support() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut trainer =
        Trainer::new(model, small_config(Device::Gpu)).expect("load the preflight model on CUDA");
    trainer
        .create_lora(&LoraConfig::auto(2, 4.0))
        .expect("create CUDA preflight LoRA");

    let report = trainer.train_preflight().expect("run CUDA preflight");
    eprintln!("{report}");
    assert!(report.contains("missing_gradient_rules: 0"), "{report}");
    assert!(
        report
            .lines()
            .any(|line| line.trim_start().starts_with("CUDA") && line.contains(':')),
        "preflight did not report the registered CUDA device:\n{report}"
    );
    // With quantized OUT_PROD ported (J4.3), the Qwen3 Q8_0 training graph is
    // fully GPU-resident: CUDA0 must report "training graph ready" with no
    // fallback nodes listed for it.
    assert!(
        report.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("CUDA") && line.ends_with(": training graph ready")
        }),
        "CUDA0 still falls back for some training ops:\n{report}"
    );
}

#[cfg(retro_cuda)]
#[test]
fn chunked_cross_entropy_trains_without_cuda_fallback() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.n_ctx = 256;
    config.n_batch = 256;
    config.n_ubatch = 256;
    config.n_seq_max = 4;
    config.chunked_cross_entropy = true;
    config.chunked_ce_tiles = 8;
    {
        let mut preflight =
            Trainer::new(&model, config.clone()).expect("load chunked-CE CUDA model");
        preflight
            .create_lora(&single_layer_lora())
            .expect("create CUDA LoRA for chunked CE preflight");
        let report = preflight
            .train_preflight()
            .expect("chunked-CE CUDA preflight");
        eprintln!("{report}");
        assert!(
            report.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("CUDA") && line.ends_with(": training graph ready")
            }),
            "fused CE graph contains a CUDA fallback:\n{report}"
        );
    }
    let mut trainer = Trainer::new(model, config.clone()).expect("reload chunked-CE CUDA model");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create CUDA LoRA for chunked CE training");

    let mut sequences = Vec::new();
    for (text, reward) in [
        ("A CUDA fused cross entropy answer with enough tokens.", 0.0),
        (
            "Another CUDA fused cross entropy answer for the group.",
            1.0,
        ),
    ] {
        let tokens = trainer
            .tokenize_text(text)
            .expect("tokenize chunked-CE row");
        let mut train_mask = vec![false; tokens.len()];
        let last = tokens.len() - 1;
        train_mask[last] = true;
        let old_logprobs = trainer
            .score_masked_tokens(&tokens, &train_mask)
            .expect("score chunked-CE row");
        sequences.push(TrainSequence {
            tokens,
            old_logprobs,
            train_mask,
            reward,
            group_id: 7,
            intermediate_returns: vec![1.0],
        });
    }
    let params = GrpoBatchParams {
        epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 2,
        seed: 42,
        scheduler_total_rollouts: None,
    };
    let metrics = train_grpo_batch(&mut trainer, &sequences, &params, &config, &mut |_| {})
        .expect("train one chunked-CE CUDA step");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_cuda)]
#[test]
fn chunked_cross_entropy_sft_trains_on_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    config.chunked_ce_tiles = 8;
    let mut trainer = Trainer::new(model, config).expect("load chunked-CE SFT CUDA model");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create CUDA LoRA for chunked SFT");
    let text = "Fused sparse cross entropy trains SFT entirely on CUDA. ".repeat(40);
    let tokens = trainer
        .tokenize_text(&text)
        .expect("tokenize chunked SFT text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train chunked-CE SFT on CUDA");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_cuda)]
#[test]
fn chunked_cross_entropy_weighted_ppo_step_trains_on_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    config.chunked_ce_tiles = 8;
    let mut trainer = Trainer::new(model, config).expect("load weighted CUDA model");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create CUDA LoRA for weighted step");
    let encoded = trainer
        .tokenize_text(&"A weighted PPO update remains on CUDA. ".repeat(40))
        .expect("tokenize weighted CUDA text");
    let n_ctx = trainer.context_size().expect("CUDA context size");
    assert!(encoded.len() > n_ctx);
    let batch = WeightedBatch {
        tokens: encoded[..n_ctx].to_vec(),
        labels: encoded[1..=n_ctx].to_vec(),
        weights: (0..n_ctx)
            .map(|i| if i % 5 == 0 { -0.2 } else { 0.8 })
            .collect(),
        n_rows: 1,
        n_ctx,
        n_topk: 1,
    };
    let metrics = trainer
        .train_weighted(&batch, 0)
        .expect("train weighted PPO-style CUDA step");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
}

// A minimal LoRA step on the quantized fixture trains with a finite loss on
// CUDA via the materialized F32 attention fallback (cap_flash_attn_back is not
// yet available). This exercises the whole training graph - forward, backward
// through the CPU-fallback quantized OUT_PROD, and the AdamW step - end to end.
#[cfg(retro_cuda)]
#[test]
fn f16_training_kv_is_effective_and_differentiable_on_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.kv_dtype = retrograd::KvDtype::F16;
    let mut trainer = Trainer::new(model, config).expect("load F16-KV CUDA model");
    let report = trainer.backend_report().expect("F16-KV backend report");
    // FLASH_ATTN_BACK (J4.1) makes the differentiable F16 KV path available on
    // CUDA for a supported head geometry.
    assert!(
        report.contains("cap_flash_attn_back: supported"),
        "{report}"
    );
    assert!(report.contains("training_kv_dtype: F16"), "{report}");
    assert!(report.contains("training_kv_f16: supported"), "{report}");

    // Resolved from the model: the default CUDA fixture is a hybrid `lfm2`,
    // whose block 0 is a shortconv block with no attention projection.
    let targets = common::block_targets(&trainer, &["attn_k", "attn_v"])
        .expect("the model has an attention block");
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(targets);
    trainer.create_lora(&lora).expect("create K/V LoRA");
    let text = "Differentiable F16 keys and values train on CUDA. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize F16-KV text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train F16-KV K/V LoRA");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_cuda)]
#[test]
fn f16_lora_optimizer_runs_on_cuda_without_cpu_fallback() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut lora = single_layer_lora();
    lora.dtype = retrograd::LoraDtype::F16;
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load model on CUDA");
    trainer.create_lora(&lora).expect("create F16 CUDA LoRA");
    let report = trainer.backend_report().expect("backend report");
    assert!(report.contains("lora_dtype: F16"), "{report}");
    // J4.4: the F16 AdamW step runs on device, not as a CPU fallback.
    assert!(report.contains("optimizer_f16: supported"), "{report}");
    let text = "CUDA performs the F16 AdamW update on device. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer.train_tokens(&tokens).expect("train F16 CUDA LoRA");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_cuda)]
#[test]
fn minimal_lora_step_updates_only_the_adapter_on_cuda() {
    let _lock = common::serialize_models();
    let Some(model) = cuda_model_or_skip() else {
        return;
    };
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load model on CUDA");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create CUDA LoRA");
    let text = "CUDA trains a small adapter while the base model stays frozen. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("run minimal CUDA LoRA step");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite());
}
