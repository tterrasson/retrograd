//! Vulkan validation split by layer: registration, individual ggml ops, model
//! offload, LoRA placement, then one minimal optimizer step.

mod common;

#[cfg(retro_vulkan)]
use retrograd::{
    Device, LoraConfig, ProbeInputs, ProbeOp, SamplingParams, TargetSet, TrainConfig, Trainer,
    probe_op,
};

#[cfg(retro_vulkan)]
fn vulkan_registered() -> bool {
    retrograd::backend_list()
        .map(|list| list.lines().any(|line| line.starts_with("gpu\tVulkan")))
        .unwrap_or(false)
}

#[cfg(retro_vulkan)]
fn vulkan_available() -> bool {
    vulkan_registered() && retrograd::gpu_runtime_available()
}

#[cfg(retro_vulkan)]
fn vulkan_icd_was_explicitly_configured() -> bool {
    std::env::var_os("VK_ICD_FILENAMES").is_some() || std::env::var_os("VK_DRIVER_FILES").is_some()
}

#[cfg(retro_vulkan)]
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

#[cfg(retro_vulkan)]
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

#[cfg(retro_vulkan)]
fn assert_close(op: ProbeOp, cpu: &[f32], vulkan: &[f32], tolerance: f32) {
    assert_eq!(cpu.len(), vulkan.len());
    let mut max_abs = 0.0_f32;
    for (index, (&expected, &actual)) in cpu.iter().zip(vulkan).enumerate() {
        let difference = (expected - actual).abs();
        max_abs = max_abs.max(difference);
        assert!(
            difference <= tolerance,
            "{op:?} differs at {index}: cpu={expected}, vulkan={actual}, diff={difference}, tolerance={tolerance}"
        );
    }
    eprintln!("{op:?}: max|cpu-vulkan|={max_abs:e}");
}

#[cfg(retro_vulkan)]
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
    let vulkan = probe_op(
        op,
        true,
        ProbeInputs::pair(shape, src0, shape, src1),
        params,
        n,
    )
    .expect("Vulkan probe");
    assert_close(op, &cpu, &vulkan, tolerance);
}

#[cfg(retro_vulkan)]
#[test]
fn vulkan_build_registers_a_vulkan_gpu() {
    let list = retrograd::backend_list().expect("backend list");
    eprintln!("registered ggml devices:\n{list}");
    if !vulkan_registered() {
        assert!(
            !vulkan_icd_was_explicitly_configured(),
            "a Vulkan ICD was explicitly configured, but no Vulkan device was registered:\n{list}"
        );
        eprintln!("skipping runtime registration check: no Vulkan ICD/device available");
        return;
    }
    assert!(
        vulkan_registered(),
        "a Vulkan build must register a Vulkan GPU device:\n{list}"
    );
}

#[cfg(retro_vulkan)]
#[test]
fn silu_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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

#[cfg(retro_vulkan)]
#[test]
fn rms_norm_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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

#[cfg(retro_vulkan)]
#[test]
fn l2_norm_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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

#[cfg(retro_vulkan)]
#[test]
fn soft_max_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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

#[cfg(retro_vulkan)]
fn compare_flash_attn_back(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    tolerance: f32,
) {
    compare_flash_attn_back_window(hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, None, tolerance);
}

// `kv_f32` stores K/V as F32 rather than F16, selecting the other shader variant.
// Both matter: `cap_flash_attn_back` is probed with an F32 cache, and clearing
// that gate is the precondition for `kv_dtype = "f16"` engaging at all (see
// retro_backend.cpp).
#[cfg(retro_vulkan)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_kv_f32(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    window: Option<(usize, usize)>,
    tolerance: f32,
) {
    compare_flash_attn_back_impl(
        hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, window, true, false, tolerance,
    );
}

// `window`, when Some((nwin, kv_off)), exercises the KV gradient window: dK/dV
// cover `nwin` cache rows starting at `kv_off` (per batch), instead of all
// `nkv` -- the n_kv > n_window case the dense tests never reach.
#[cfg(retro_vulkan)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_window(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    window: Option<(usize, usize)>,
    tolerance: f32,
) {
    compare_flash_attn_back_impl(
        hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, window, false, false, tolerance,
    );
}

// Attention sinks: one extra logit per head in the softmax denominator, with no
// V row of its own. Vulkan is the only backend that accepts them in
// FLASH_ATTN_BACK (CUDA and Metal refuse, commented) -- so per the project rule
// that a `supports_op` only advertises what a probe exercises, this is the probe
// that entitles Vulkan to keep saying yes.
#[cfg(retro_vulkan)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_sinks(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    kv_f32: bool,
    tolerance: f32,
) {
    compare_flash_attn_back_impl(
        hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, None, kv_f32, true, tolerance,
    );
}

#[cfg(retro_vulkan)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_impl(
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    window: Option<(usize, usize)>,
    kv_f32: bool,
    sinks: bool,
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
    if sinks {
        // One sink logit per head, appended after dO (see the probe). Spread
        // around the score scale so the sink genuinely takes softmax mass: a sink
        // far below every score would leave the probabilities untouched and the
        // test would pass whether or not the shader reads it at all.
        packed.extend(pseudo_random(nhead, 0xf1a5_0005, 1.5));
    }

    let dims = [hsk as i64, hsv as i64, nq as i64, nkv as i64];
    let layout = [nhead as i64, nhead_kv as i64, nbatch as i64, 1];
    let layout_data = [0.0f32; 4];
    let params = [1.0 / (hsk as f32).sqrt(), 0.0];

    // src2 carries the window: ne_src2[0] = nwin, data = per-batch cache rows
    // (row r of batch ib maps to index r + nkv*ib; see the probe). ne_src2[1] is
    // the F32-KV flag, so a src2 of [0, 1, 0, 0] with no data means "no window,
    // F32 K/V".
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
    let ne_src2 = [
        if window.is_some() { nwin as i64 } else { 0 },
        kv_f32 as i64,
        sinks as i64,
        0,
    ];
    let src2 = if window.is_some() || kv_f32 || sinks {
        Some((ne_src2, idx_data.as_slice()))
    } else {
        None
    };

    let cpu = probe_op(
        ProbeOp::FlashAttnBack,
        false,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("streaming CPU Flash Attention backward reference");
    let vulkan = probe_op(
        ProbeOp::FlashAttnBack,
        true,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("Vulkan Flash Attention backward probe");
    assert_close(ProbeOp::FlashAttnBack, &cpu, &vulkan, tolerance);

    let scalar = {
        let _env = common::EnvGuard::set("GGML_VK_FA_BACK_MMA", "0");
        probe_op(
            ProbeOp::FlashAttnBack,
            true,
            ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
            params,
            out_len,
        )
        .expect("scalar Vulkan Flash Attention backward probe")
    };
    assert_close(ProbeOp::FlashAttnBack, &scalar, &vulkan, tolerance);

    // The kernel rounds Q/dO to F16 before cooperative-matrix products, so
    // repeated execution is validated by the same numerical tolerance as the
    // CPU oracle. Each gradient KV tile still has one workgroup.
    let repeated = probe_op(
        ProbeOp::FlashAttnBack,
        true,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("repeated Vulkan Flash Attention backward probe");
    assert_close(ProbeOp::FlashAttnBack, &vulkan, &repeated, tolerance);
}

/// The folded log-sum-exp is the same algebra in a different summation order,
/// so the two regimes are held to a tolerance against each other and not to a
/// bit-for-bit equality. `GGML_VK_FA_BACK_FUSED_LSE=1` selects the single sweep
/// on the same build -- which is also what makes a before/after timing possible
/// in one session. The default stays the two-sweep form until a measurement
/// clears the other one.
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_lse_regimes_agree() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    for (window, sinks) in [(None, false), (Some((4, 2)), false), (None, true)] {
        let two_pass = fa_back_vulkan_probe(FaBackTimingShape::small(), window, sinks);
        let fused = {
            let _env = common::EnvGuard::set("GGML_VK_FA_BACK_FUSED_LSE", "1");
            fa_back_vulkan_probe(FaBackTimingShape::small(), window, sinks)
        };
        assert_close(ProbeOp::FlashAttnBack, &two_pass, &fused, 7.5e-4);
    }
}

/// The shape a timing run uses, and the one the parity case above reuses so the
/// two cannot drift apart.
#[cfg(retro_vulkan)]
#[derive(Clone, Copy)]
struct FaBackTimingShape {
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
}

#[cfg(retro_vulkan)]
impl FaBackTimingShape {
    fn small() -> Self {
        Self {
            hsk: 32,
            hsv: 32,
            nq: 5,
            nkv: 8,
            nhead: 4,
            nhead_kv: 2,
            nbatch: 1,
        }
    }
}

#[cfg(retro_vulkan)]
fn fa_back_vulkan_probe(
    shape: FaBackTimingShape,
    window: Option<(usize, usize)>,
    sinks: bool,
) -> Vec<f32> {
    fa_back_probe(shape, window, sinks, true)
}

#[cfg(retro_vulkan)]
fn fa_back_probe(
    shape: FaBackTimingShape,
    window: Option<(usize, usize)>,
    sinks: bool,
    use_gpu: bool,
) -> Vec<f32> {
    let FaBackTimingShape {
        hsk,
        hsv,
        nq,
        nkv,
        nhead,
        nhead_kv,
        nbatch,
    } = shape;
    let n_q = hsk * nq * nhead * nbatch;
    let n_k = hsk * nkv * nhead_kv * nbatch;
    let n_v = hsv * nkv * nhead_kv * nbatch;
    let n_do = hsv * nhead * nq * nbatch;
    let mut packed = Vec::with_capacity(n_q + n_k + n_v + n_do);
    packed.extend(pseudo_random(n_q, 0xf1a5_0001, 0.4));
    packed.extend(pseudo_random(n_k, 0xf1a5_0002, 0.4));
    packed.extend(pseudo_random(n_v, 0xf1a5_0003, 0.4));
    packed.extend(pseudo_random(n_do, 0xf1a5_0004, 0.7));
    if sinks {
        packed.extend(pseudo_random(nhead, 0xf1a5_0005, 1.5));
    }
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
    let ne_src2 = [
        if window.is_some() { nwin as i64 } else { 0 },
        0,
        sinks as i64,
        0,
    ];
    let src2 = (window.is_some() || sinks).then_some((ne_src2, idx_data.as_slice()));
    let out_len = n_q + hsk * nwin * nhead_kv * nbatch + hsv * nwin * nhead_kv * nbatch;
    probe_op(
        ProbeOp::FlashAttnBack,
        use_gpu,
        ProbeInputs::pair(
            [hsk as i64, hsv as i64, nq as i64, nkv as i64],
            &packed,
            [nhead as i64, nhead_kv as i64, nbatch as i64, 1],
            &[0.0f32; 4],
        )
        .with_src2(src2),
        [1.0 / (hsk as f32).sqrt(), 0.0],
        out_len,
    )
    .expect("Vulkan Flash Attention backward probe")
}

/// The before/after of a schedule change has to be taken in one
/// session, on one machine, without a rebuild. Gated by `RETRO_FA_TIME=1` the way
/// the RIR loop is gated by `RIR_TIME`, because it is a measurement and not an
/// assertion: it prints, it does not decide.
///
/// What it does *not* separate is the dQ dispatch from the dK/dV one -- the probe
/// always asks for all three gradients. The two dispatches are named
/// (`flash_attn_back_{q,kv}_*`), so a GPU profiler splits them without new code;
/// splitting them here would mean extending the probe ABI with a gradient mask.
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_lse_regime_timing() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    if std::env::var("RETRO_FA_TIME").is_err() {
        eprintln!("skipping: set RETRO_FA_TIME=1 to time the two log-sum-exp regimes");
        return;
    }
    let shapes = [
        FaBackTimingShape {
            hsk: 128,
            hsv: 128,
            nq: 256,
            nkv: 256,
            nhead: 8,
            nhead_kv: 2,
            nbatch: 1,
        },
        FaBackTimingShape {
            hsk: 128,
            hsv: 128,
            nq: 128,
            nkv: 1024,
            nhead: 8,
            nhead_kv: 2,
            nbatch: 1,
        },
        FaBackTimingShape {
            hsk: 64,
            hsv: 64,
            nq: 512,
            nkv: 512,
            nhead: 8,
            nhead_kv: 8,
            nbatch: 1,
        },
    ];
    let median = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    for shape in shapes {
        // The floor this harness cannot see under: `probe_op` computes its own
        // analytic CPU reference on every call, GPU or not, so a run's wall clock
        // is that reference plus the graph plus the two dispatches. Printed rather
        // than hidden -- a regime delta far below it is not a measurement.
        let oracle = median(
            (0..3)
                .map(|_| {
                    let start = std::time::Instant::now();
                    let _ = fa_back_probe(shape, None, false, false);
                    start.elapsed().as_secs_f64() * 1.0e3
                })
                .collect(),
        );
        let mut medians = Vec::new();
        for fused in [false, true] {
            let _env = fused.then(|| common::EnvGuard::set("GGML_VK_FA_BACK_FUSED_LSE", "1"));
            let mut samples = Vec::new();
            for run in 0..6 {
                let start = std::time::Instant::now();
                let _ = fa_back_vulkan_probe(shape, None, false);
                if run > 0 {
                    samples.push(start.elapsed().as_secs_f64() * 1.0e3);
                }
            }
            medians.push(median(samples));
        }
        eprintln!(
            "fa_back timing hsk={} nq={} nkv={} nhead={}/{}: oracle={:.2} ms two-pass={:.2} ms fused={:.2} ms ({:+.1} % of the run)",
            shape.hsk,
            shape.nq,
            shape.nkv,
            shape.nhead,
            shape.nhead_kv,
            oracle,
            medians[0],
            medians[1],
            (medians[1] - medians[0]) / medians[0] * 100.0,
        );
    }
}

#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_matches_streaming_cpu_reference() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    compare_flash_attn_back(32, 32, 5, 8, 4, 2, 1, 7.5e-4);
}

// The probe that entitles the Vulkan `supports_op` to accept attention sinks.
// `flash_attn_back_q.comp` folds the sink into the
// row's `lse` and `flash_attn_back_kv.comp` re-reads that `lse`, so a sink read
// on only one of the two sides shows up as a dQ/dK mismatch rather than a
// uniformly scaled gradient. Both KV element types run: the sink lives in the F32
// score domain, but the two variants reduce differently.
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_matches_cpu_with_attention_sinks() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    compare_flash_attn_back_sinks(32, 32, 5, 8, 4, 2, 1, false, 7.5e-4);
    compare_flash_attn_back_sinks(32, 32, 5, 8, 4, 2, 1, true, 5.0e-5);
    // Grouped heads over two batches: the sink is indexed by head, not by KV head,
    // so a shader reading it at the KV-head index diverges here and not above.
    compare_flash_attn_back_sinks(64, 64, 3, 6, 8, 2, 2, false, 1.5e-3);
}

#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_covers_wide_head_dims() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // Head dim 256 (Qwen3.5/GDN, Gemma2) selects the deeper shader variant; 128
    // stays on the shallow one. See the bucket table in ggml-vulkan.cpp.
    compare_flash_attn_back(128, 128, 4, 10, 8, 2, 1, 1.5e-3);
    compare_flash_attn_back(256, 256, 4, 8, 4, 2, 1, 2.0e-3);
}

// Head dim 512: Gemma-4's global-attention layers (`key_length = 512` against
// `key_length_swa = 256`), the FA_BACK_BUCKET_512 shader variant. Without it a
// single such layer failed the model-wide capability probe and cost the whole
// model the fused path and the F16 KV cache. Unlike CUDA, the Vulkan forward has
// no alignment precondition at this head dimension, so the shapes stay small.
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_covers_xwide_head_dims() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // Dense, then Gemma-4's own 16:1 head ratio, then the F32 cache variant and
    // an offset gradient window at this head dimension. Tolerance steps once more
    // from the 256 bucket: twice the head dimension is twice the accumulated
    // subgroup-reduction error per dot product.
    compare_flash_attn_back(512, 512, 4, 8, 4, 2, 1, 3.0e-3);
    compare_flash_attn_back(512, 512, 3, 6, 16, 1, 2, 3.0e-3);
    compare_flash_attn_back_kv_f32(512, 512, 4, 8, 4, 2, 1, None, 5.0e-5);
    compare_flash_attn_back_window(512, 512, 4, 24, 4, 2, 1, Some((4, 8)), 3.0e-3);
}

// KV gradient window: dK/dV cover only the rows written at this step, so n_kv
// (what the forward reads) exceeds n_window (what receives a gradient). This is
// the shape production hits at a long context and the dense tests never do,
// including kv_off > 0 which catches a window indexed off by a row.
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_covers_kv_gradient_window() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // Window strictly inside a larger cache, with and without an offset.
    compare_flash_attn_back_window(32, 32, 4, 24, 4, 2, 1, Some((4, 0)), 7.5e-4);
    compare_flash_attn_back_window(32, 32, 4, 24, 4, 2, 1, Some((4, 8)), 7.5e-4);
    // Multi-stream (nbatch > 1) offset window, wide head dim.
    compare_flash_attn_back_window(256, 256, 5, 40, 4, 2, 2, Some((5, 7)), 2.0e-3);
    // Degenerate case: window == whole cache must match the dense path.
    compare_flash_attn_back_window(32, 32, 5, 5, 4, 2, 1, Some((5, 0)), 7.5e-4);
}

// F32 K/V selects the other shader variant, and it is the one the capability probe
// actually builds (retro_backend.cpp probes with GGML_TYPE_F32), so the F16 path is
// only reachable once this one is correct. No F16 rounding in the operands, so the
// tolerance is ~15x tighter than the F16 cases above -- but not arbitrarily so:
// measured diffs here are 5e-6..2.1e-5, i.e. the driver's own accumulation order is
// the floor, not the cache dtype. (Metal, same cases, lands at ~2e-7.)
#[cfg(retro_vulkan)]
#[test]
fn flash_attn_back_vulkan_covers_f32_kv_cache() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    compare_flash_attn_back_kv_f32(32, 32, 5, 8, 4, 2, 1, None, 5.0e-5);
    compare_flash_attn_back_kv_f32(128, 128, 4, 10, 8, 2, 1, None, 5.0e-5);
    compare_flash_attn_back_kv_f32(256, 256, 4, 8, 4, 2, 1, None, 5.0e-5);
    // Offset gradient window, multi-stream.
    compare_flash_attn_back_kv_f32(128, 128, 4, 24, 4, 2, 2, Some((4, 8)), 5.0e-5);
}

#[cfg(retro_vulkan)]
#[test]
fn get_rows_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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
    let vulkan = probe_op(
        ProbeOp::GetRowsBack,
        true,
        ProbeInputs::pair(grad_shape, &grad, index_shape, &indices)
            .with_src2(Some((output_shape, &output_template))),
        [0.0, 0.0],
        output_template.len(),
    )
    .expect("Vulkan get_rows_back probe");
    assert_close(ProbeOp::GetRowsBack, &cpu, &vulkan, 1.0e-6);
}

// dst[i0,i1] = Sum_k src0[i0,k] * src1[i1,k]; src0 is [ne00,k], src1 is [ne10,k],
// so the output is [ne00, ne10]. `op` selects the F32 or a quantized-src0 kernel.
#[cfg(retro_vulkan)]
fn compare_out_prod_shapes(op: ProbeOp, ne_src0: [i64; 4], ne_src1: [i64; 4], tolerance: f32) {
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
    .expect("CPU out_prod probe");
    let vulkan = probe_op(
        op,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("Vulkan out_prod probe");
    assert_close(op, &cpu, &vulkan, tolerance);
}

#[cfg(retro_vulkan)]
fn compare_out_prod(op: ProbeOp, ne00: i64, k: i64, ne10: i64, tolerance: f32) {
    compare_out_prod_shapes(op, [ne00, k, 1, 1], [ne10, k, 1, 1], tolerance);
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    compare_out_prod(ProbeOp::OutProd, 64, 48, 32, 1.0e-4);
}

/// Shapes that exercise the boundaries of the tiled `out_prod` shader.
///
/// The shader computes a 64x16 dst tile per workgroup with a 16-deep reduction
/// slice. That puts a boundary everywhere: dst rows do not align to the tile, dst
/// columns below 16 leave part
/// of the tile inactive, and a reduction length that is not a multiple of the slice
/// depth relies on the padded lanes contributing exactly zero.
///
/// `ne10 = 8` is the case that matters most in practice - it is the nominal
/// `n_ubatch`, so it is the shape the training graph actually dispatches.
///
/// Tolerances are deliberately the same as the aligned cases above: the rewrite
/// preserves the summation order exactly, so it has no licence to be less accurate.
/// A widened tolerance here would mean the rewrite changed the arithmetic.
#[cfg(retro_vulkan)]
#[test]
fn out_prod_vulkan_matches_cpu_on_tile_boundaries() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // (ne00, k, ne10): thin dst, unaligned rows, reduction shorter/longer than a
    // slice, and a single row/column degenerate case.
    for (ne00, k, ne10) in [
        (64, 16, 8),   // nominal ubatch: half the tile's columns are inactive
        (130, 17, 8),  // dst rows unaligned to the 64-row tile, k unaligned to 16
        (64, 1, 1),    // reduction and dst of length one
        (33, 5, 3),    // every dimension below its tile extent
        (256, 129, 8), // several full row tiles, reduction just past a slice
        (64, 48, 40),  // more dst columns than one tile
    ] {
        compare_out_prod(ProbeOp::OutProd, ne00, k, ne10, 1.0e-4);
    }
    // Batched, including src0 broadcast over the batch dimension.
    compare_out_prod_shapes(ProbeOp::OutProd, [96, 33, 2, 1], [8, 33, 2, 1], 1.0e-4);
    compare_out_prod_shapes(ProbeOp::OutProd, [96, 33, 1, 1], [8, 33, 2, 1], 1.0e-4);
}

/// The same boundary sweep on a quantized `src0`, where the tiling also moved the
/// per-element block/`iqs` selection into the cooperative load.
///
/// Q8_0 and Q4_K cover both `QUANT_R` branches of that selection, which is the part
/// of the shader most likely to break under retiling: for `QUANT_R == 2` the two
/// values a `dequantize()` call returns are half a block apart rather than adjacent.
#[cfg(retro_vulkan)]
#[test]
fn out_prod_quant_vulkan_matches_cpu_on_tile_boundaries() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    for op in [ProbeOp::OutProdQ80, ProbeOp::OutProdQ4K] {
        // ne00 must stay a multiple of the block size (32 for Q8_0, 256 for K).
        let block = if op == ProbeOp::OutProdQ4K { 256 } else { 32 };
        for (rows, k, ne10) in [(2, 17, 8), (1, 5, 3), (3, 129, 8), (2, 16, 40)] {
            compare_out_prod(op, rows * block, k, ne10, 2.0e-3);
        }
        // Batched, but *not* src0-broadcast over the batch: the F32 CPU out_prod
        // accepts a broadcast src0 (it asserts ne2 % ne02 == 0 and derives dps2, as
        // the F32 case above exercises), while the quantized CPU out_prod asserts
        // ne02 == ne12 outright. So [.., 1, 1] x [.., 2, 1] has no CPU oracle to
        // compare against for a quantized src0 -- it aborts in the reference, not in
        // the shader. Matching ne02 to ne12 still covers what the retiling moved
        // here, since the batch is a dispatch dimension and the tile decomposition
        // is per-plane either way.
        compare_out_prod_shapes(op, [2 * block, 33, 2, 1], [8, 33, 2, 1], 2.0e-3);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_q8_0_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // ne00 must be a multiple of the Q8_0 block size (32).
    compare_out_prod(ProbeOp::OutProdQ80, 64, 40, 16, 1.0e-3);
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_q5_0_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // ne00 must be a multiple of the Q5_0 block size (32).
    compare_out_prod(ProbeOp::OutProdQ50, 64, 40, 16, 1.0e-3);
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_q4_q5_legacy_quants_vulkan_match_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    for op in [
        ProbeOp::OutProdQ40,
        ProbeOp::OutProdQ41,
        ProbeOp::OutProdQ51,
    ] {
        compare_out_prod(op, 64, 40, 17, 2.0e-3);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_k_quants_vulkan_match_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let ops = [
        ProbeOp::OutProdQ2K,
        ProbeOp::OutProdQ3K,
        ProbeOp::OutProdQ4K,
        ProbeOp::OutProdQ5K,
        ProbeOp::OutProdQ6K,
    ];
    for op in ops {
        // One partial output tile, then two K blocks over two batch planes.
        compare_out_prod_shapes(op, [256, 7, 1, 1], [11, 7, 1, 1], 2.0e-3);
        compare_out_prod_shapes(op, [512, 9, 2, 1], [5, 9, 2, 1], 2.0e-3);
    }
}

/// Every type in the fork's decodable-types table, not a hand-picked subset. See
/// `common::assert_out_prod_all_dequant_types_match_cpu`.
#[cfg(retro_vulkan)]
#[test]
fn out_prod_all_dequant_types_vulkan_match_cpu() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    common::assert_out_prod_all_dequant_types_match_cpu("vulkan");
    common::assert_out_prod_extra_types_match_cpu(true, "vulkan");
}

#[cfg(retro_vulkan)]
#[test]
fn out_prod_quant_vulkan_native_is_independent_of_the_cuda_budget() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    common::assert_out_prod_quant_budget_independent(true, "vulkan");
}

#[cfg(retro_vulkan)]
#[test]
fn cross_entropy_loss_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }

    // The dense CE forward. Logits random, labels a
    // valid distribution per row. The row widths pick the three regimes of the
    // shader's strided reduction: a vocab-sized row, a row narrower than the
    // 128-wide workgroup, and a width that is not a multiple of it. Output is one
    // scalar accumulated with atomic adds, so this also checks the host zero-fill.
    let shapes: [[i64; 4]; 4] = [
        [4096, 7, 1, 1],
        [151, 3, 1, 1],
        [33, 2, 1, 1],
        [128, 1, 1, 1],
    ];
    for (k, ne) in shapes.iter().enumerate() {
        let n = ne.iter().product::<i64>() as usize;
        let width = ne[0] as usize;
        let logits = pseudo_random(n, 0x4444 + k as u64, 4.0);
        let labels = softmax_rows(&pseudo_random(n, 0x8888 + k as u64, 8.0), width);
        let cpu = probe_op(
            ProbeOp::CrossEntropyLoss,
            false,
            ProbeInputs::pair(*ne, &logits, *ne, &labels),
            [0.0, 0.0],
            1,
        )
        .expect("CPU cross_entropy_loss probe");
        let vulkan = probe_op(
            ProbeOp::CrossEntropyLoss,
            true,
            ProbeInputs::pair(*ne, &logits, *ne, &labels),
            [0.0, 0.0],
            1,
        )
        .expect("Vulkan cross_entropy_loss probe");
        // Guard against a vacuous pass on a zero loss.
        assert!(
            cpu[0].abs() > 1e-3,
            "CrossEntropyLoss case {k}: CPU loss unexpectedly ~0 ({})",
            cpu[0]
        );
        assert_close(ProbeOp::CrossEntropyLoss, &cpu, &vulkan, 1.0e-4);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn cross_entropy_loss_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let nc = 64_i64;
    let nr = 4_i64;
    let logits = pseudo_random((nc * nr) as usize, 0xce11, 3.0);
    // one-hot labels: every row active, so nactive == nr on both backends.
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
    let vulkan = probe_op(
        ProbeOp::CrossEntropyLossBack,
        true,
        ProbeInputs::pair([1, 1, 1, 1], &grad, [nc, nr, 1, 1], &logits)
            .with_src2(Some(([nc, nr, 1, 1], &labels))),
        [0.0, 0.0],
        out_len,
    )
    .expect("Vulkan cross_entropy_loss_back probe");
    assert_close(ProbeOp::CrossEntropyLossBack, &cpu, &vulkan, 1.0e-5);
}

#[cfg(retro_vulkan)]
#[test]
fn ssm_conv_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let d_conv = 4_i64;
    let n_t = 6_i64;
    let ncs = d_conv - 1 + n_t; // 9
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
    let vulkan = probe_op(
        ProbeOp::SsmConvBack,
        true,
        ProbeInputs::pair(ne_sx, &sx, ne_c, &c).with_src2(Some((ne_dy, &dy))),
        [0.0, 0.0],
        out_len,
    )
    .expect("Vulkan ssm_conv_back probe");
    assert_close(ProbeOp::SsmConvBack, &cpu, &vulkan, 1.0e-5);
}

#[cfg(retro_vulkan)]
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
    let vulkan = probe_op(
        ProbeOp::ConvRsGather,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        params,
        out_len,
    )
    .expect("Vulkan conv_rs_gather probe");
    // A pure gather: the two backends must agree bit for bit.
    assert_close(ProbeOp::ConvRsGather, &cpu, &vulkan, 0.0);
}

#[cfg(retro_vulkan)]
#[test]
fn conv_rs_gather_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // K == 1 is the single-snapshot case; K > 1 walks the overlapping windows.
    conv_rs_gather_case(3, 8, 1);
    conv_rs_gather_case(3, 8, 5);
    // n_seq_tokens < K exercises the clamp that collapses the trailing slots
    // onto slot 0's window.
    conv_rs_gather_case(3, 2, 6);
}

#[cfg(retro_vulkan)]
#[test]
fn ssm_scan_back_vulkan_matches_cpu_in_isolation() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    // dims: nc=d_state, nr=head_dim, nh=n_head, nslot; ng=n_group, nt, ns, nA0
    let (nc, nr, nh, nslot) = (16_i64, 8_i64, 4_i64, 2_i64);
    // Forty tokens cross the 32-token chunk boundary of the two-pass kernel.
    let (ng, nt, ns, n_a0) = (2_i64, 40_i64, 2_i64, 1_i64);
    let ne_src0 = [nc, nr, nh, nslot];
    let ne_src1 = [ng, nt, ns, n_a0];

    let n_s = nc * nr * nh * nslot;
    let n_x = nr * nh * nt * ns;
    let n_dt = nh * nt * ns;
    let n_a = n_a0 * nh;
    let n_b = nc * ng * nt * ns;
    let n_c = nc * ng * nt * ns;
    let n_ds = nr * nh * nt * ns + nc * nr * nh * ns;

    // packed src0: [ s | x | dt | A | B | C | ids | ds ]
    let mut packed = Vec::new();
    packed.extend(pseudo_random(n_s as usize, 0x5501, 0.5));
    packed.extend(pseudo_random(n_x as usize, 0x5502, 0.5));
    packed.extend(pseudo_random(n_dt as usize, 0x5503, 0.5));
    // keep A negative for a stable decay exp(softplus(dt)*A)
    packed.extend(
        pseudo_random(n_a as usize, 0x5504, 0.5)
            .iter()
            .map(|v| -v.abs()),
    );
    packed.extend(pseudo_random(n_b as usize, 0x5505, 0.5));
    packed.extend(pseudo_random(n_c as usize, 0x5506, 0.5));
    packed.extend((0..ns).map(|i| i as f32)); // ids: one slot per sequence
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
    let vulkan = probe_op(
        ProbeOp::SsmScanBack,
        true,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [0.0, 0.0],
        out_len,
    )
    .expect("Vulkan ssm_scan_back probe");
    assert_close(ProbeOp::SsmScanBack, &cpu, &vulkan, 5.0e-3);
}

#[cfg(retro_vulkan)]
#[derive(Clone, Copy, Debug)]
struct GdnCase {
    kda: bool,
    k: i64,
    s_v: i64,
    h: i64,
    n_tokens: i64,
    n_seqs: i64,
    gate: f32,
    beta: Option<f32>,
}

// The production-shape case reserves a sizeable Vulkan scratch buffer. Running
// all GDN cases concurrently creates and destroys several backend devices at
// once and can exhaust/tear down the driver's test contexts. Other Vulkan tests
// already serialize model-owning cases; keep this op family serialized too.
#[cfg(retro_vulkan)]
static GDN_VULKAN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(retro_vulkan)]
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
            beta: None,
        }
    }

    fn packed(&self) -> (Vec<f32>, [i64; 4], [i64; 4], Vec<(&'static str, usize)>) {
        let n_qkv = self.s_v * self.h * self.n_tokens * self.n_seqs;
        let n_g = (if self.kda { self.s_v } else { 1 }) * self.h * self.n_tokens * self.n_seqs;
        let n_beta = self.h * self.n_tokens * self.n_seqs;
        let n_state = self.s_v * self.s_v * self.h * self.n_seqs;
        let n_grad = n_qkv + self.k * n_state;
        let mut packed = Vec::new();
        packed.extend(pseudo_random(n_qkv as usize, 0x6601, 0.5));
        packed.extend(pseudo_random(n_qkv as usize, 0x6602, 0.5));
        packed.extend(pseudo_random(n_qkv as usize, 0x6603, 0.5));
        packed.extend(
            pseudo_random(n_g as usize, 0x6604, 1.0)
                .iter()
                .map(|v| -v.abs() * self.gate - 0.02 * self.gate),
        );
        packed.extend(
            pseudo_random(n_beta as usize, 0x6605, 0.5)
                .iter()
                .map(|v| self.beta.unwrap_or(v.abs())),
        );
        packed.extend(pseudo_random(n_state as usize, 0x6606, 0.5));
        packed.extend(pseudo_random(n_grad as usize, 0x6607, 0.5));
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
            [self.s_v, self.h, self.n_tokens, self.n_seqs],
            [self.k, if self.kda { 1 } else { 0 }, 1, 1],
            blocks,
        )
    }
}

#[cfg(retro_vulkan)]
fn gdn_probe(case: &GdnCase, vulkan: bool, chunk: f32) -> Vec<f32> {
    let (packed, ne_src0, ne_src1, blocks) = case.packed();
    let dummy = [0.0_f32];
    probe_op(
        ProbeOp::GatedDeltaNetBack,
        vulkan,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy),
        [chunk, 0.0],
        blocks.iter().map(|(_, n)| n).sum(),
    )
    .expect("gated_delta_net_back probe")
}

#[cfg(retro_vulkan)]
fn gdn_assert_agrees(case: &GdnCase, reference: &[f32], actual: &[f32], tolerance: f32) {
    let (_, _, _, blocks) = case.packed();
    assert_eq!(reference.len(), actual.len());
    let mut offset = 0;
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
            "{case:?}: {name}[{where_at}] differs: reference={}, actual={}, relative={relative:e} > {tolerance:e}",
            reference[offset + where_at],
            actual[offset + where_at],
        );
        offset += count;
    }
}

#[cfg(retro_vulkan)]
fn gated_delta_net_back_case(case: &GdnCase, tolerance: f32) {
    let _guard = GDN_VULKAN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let cpu = gdn_probe(case, false, -1.0);
    gdn_assert_agrees(case, &cpu, &gdn_probe(case, true, 0.0), tolerance);
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_matches_cpu_scalar_gate_k1() {
    let mut case = GdnCase::new(20);
    case.gate = 2.0;
    gated_delta_net_back_case(&case, 1.0e-4);
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_matches_cpu_kda_gate_with_snapshots() {
    let mut case = GdnCase::new(20);
    case.kda = true;
    case.k = 3;
    case.gate = 2.0;
    gated_delta_net_back_case(&case, 1.0e-4);
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_keeps_the_sequential_path() {
    let _guard = GDN_VULKAN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !vulkan_available() {
        return;
    }
    let case = GdnCase::new(70);
    let cpu = gdn_probe(&case, false, -1.0);
    let sequential = gdn_probe(&case, true, -1.0);
    gdn_assert_agrees(&case, &cpu, &sequential, 1.0e-4);
    gdn_assert_agrees(&case, &sequential, &gdn_probe(&case, true, 64.0), 1.0e-4);
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_covers_chunk_boundaries() {
    let _guard = GDN_VULKAN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !vulkan_available() {
        return;
    }
    for n_tokens in [1, 2, 63, 64, 65, 129] {
        let case = GdnCase::new(n_tokens);
        let cpu = gdn_probe(&case, false, -1.0);
        gdn_assert_agrees(&case, &cpu, &gdn_probe(&case, true, 64.0), 1.0e-4);
    }
    let case = GdnCase::new(40);
    let cpu = gdn_probe(&case, false, -1.0);
    for chunk in [1.0, 4096.0] {
        gdn_assert_agrees(&case, &cpu, &gdn_probe(&case, true, chunk), 1.0e-4);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_survives_adverse_gates() {
    for gate in [0.0, 1.0, 4.0, 200.0] {
        let mut case = GdnCase::new(96);
        case.gate = gate;
        gated_delta_net_back_case(&case, 1.0e-4);
        case.kda = true;
        gated_delta_net_back_case(&case, 1.0e-4);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_covers_beta_and_degenerate_grid() {
    for beta in [0.0, 1.0] {
        let mut case = GdnCase::new(65);
        case.h = 1;
        case.n_seqs = 1;
        case.beta = Some(beta);
        gated_delta_net_back_case(&case, 1.0e-4);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn gated_delta_net_back_vulkan_matches_cpu_at_production_head_dim() {
    let mut case = GdnCase::new(128);
    case.s_v = 128;
    case.h = 16;
    case.n_seqs = 2;
    gated_delta_net_back_case(&case, 1.0e-4);
}

#[cfg(retro_vulkan)]
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

#[cfg(retro_vulkan)]
fn single_layer_lora() -> LoraConfig {
    let mut config = LoraConfig::qv(2, 4.0);
    config.seed = 7;
    config.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    config
}

#[cfg(retro_vulkan)]
fn vulkan_model_or_skip() -> Option<std::path::PathBuf> {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return None;
    }
    let model = common::vulkan_model_path_if_available();
    if model.is_none() {
        eprintln!(
            "skipping: no Vulkan test model at {}",
            common::vulkan_model_path().display()
        );
    }
    model
}

#[cfg(retro_vulkan)]
#[test]
fn device_sampling_is_reproducible_for_heterogeneous_prompts() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.n_seq_max = 2;
    let mut trainer = Trainer::new(model, config).expect("load sampling model on Vulkan");
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    let params = |seed| SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        max_new_tokens: 4,
        seed,
    };

    let _env = common::EnvGuard::set("RETRO_DEVICE_SAMPLING", "1");
    let requests = [(&first[..], params(41)), (&second[..], params(42))];
    let first_run = trainer
        .generate_tokens_continuous(&requests)
        .expect("first on-device sample");
    let second_run = trainer
        .generate_tokens_continuous(&requests)
        .expect("repeat on-device sample");
    assert_eq!(
        first_run, second_run,
        "fixed seeds must reproduce device samples"
    );
    assert!(first_run.iter().all(|tokens| !tokens.is_empty()));
}

/// The micro-batch escalation is not keyed on this model's name: the
/// runtime decodes the requested micro-batch, and escalates to the full logical
/// batch only when that decode came back non-finite. So this test does not
/// assert a width - it asserts that whatever width was chosen, the report says
/// which measurement chose it, and that a machine whose driver has been fixed
/// keeps the width the caller asked for instead of paying for a model-name rule.
#[cfg(all(retro_vulkan, target_os = "macos"))]
#[test]
fn falcon_h1_micro_batch_follows_the_finiteness_measurement_on_moltenvk() {
    let _lock = common::serialize_models();
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let Some(model) = common::falcon_h1_model_path_if_available() else {
        eprintln!(
            "skipping: no Falcon-H1 regression model at {}",
            common::falcon_h1_model_path().display()
        );
        return;
    };
    let config = TrainConfig {
        n_ctx: 64,
        n_batch: 16,
        n_ubatch: 8,
        device: Device::Gpu,
        ..TrainConfig::default()
    };
    let trainer = Trainer::new(model, config).expect("load Falcon-H1 on Vulkan");
    let report = trainer.backend_report().expect("backend report");
    let capabilities = trainer.capability_report().expect("capability report");
    assert!(report.contains("effective_batch: 16"), "{report}");

    let escalated = capabilities.contains("micro_batch_finite_check: escalated");
    let passed = capabilities.contains("micro_batch_finite_check: pass");
    assert!(
        escalated || passed,
        "the micro-batch must be decided by a measurement, not assumed\n{capabilities}"
    );
    if escalated {
        assert!(
            report.contains("effective_ubatch: 16"),
            "an escalation must reach the full logical batch\n{report}\n{capabilities}"
        );
    } else {
        assert!(
            report.contains("effective_ubatch: 8"),
            "a driver that computes the requested micro-batch must keep it\n{report}"
        );
    }
}

#[cfg(retro_vulkan)]
#[test]
fn falcon_h1_minimal_lora_step_is_finite_on_vulkan() {
    let _lock = common::serialize_models();
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let Some(model) = common::falcon_h1_model_path_if_available() else {
        eprintln!(
            "skipping: no Falcon-H1 regression model at {}",
            common::falcon_h1_model_path().display()
        );
        return;
    };
    let config = TrainConfig {
        n_ctx: 64,
        n_batch: 16,
        n_ubatch: 8,
        epochs: 1,
        learning_rate: 1.0e-4,
        device: Device::Gpu,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(model, config).expect("load Falcon-H1 on Vulkan");
    trainer
        .create_lora(&LoraConfig::auto(2, 4.0))
        .expect("create Falcon-H1 LoRA");
    let text = "Falcon trains a finite adapter update on Vulkan. ".repeat(32);
    let tokens = trainer
        .tokenize_text(&text)
        .expect("tokenize Falcon smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("run Falcon-H1 Vulkan LoRA step");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_vulkan)]
#[test]
fn gemma_model_is_offloaded_to_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load Gemma on Vulkan");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("{report}");
    assert!(report.contains("gpu_active: true"), "{report}");
    assert!(report.contains("backend: Vulkan"), "{report}");
    assert!(
        common::section_has_vulkan(&report, "model_tensors_by_buffer"),
        "Gemma tensors were not offloaded to Vulkan:\n{report}"
    );
}

#[cfg(retro_vulkan)]
#[test]
fn training_preflight_reports_vulkan_device_support() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut trainer =
        Trainer::new(model, small_config(Device::Gpu)).expect("load the preflight model on Vulkan");
    trainer
        .create_lora(&LoraConfig::auto(2, 4.0))
        .expect("create Vulkan preflight LoRA");

    let report = trainer.train_preflight().expect("run Vulkan preflight");
    eprintln!("{report}");
    assert!(report.contains("missing_gradient_rules: 0"), "{report}");
    assert!(
        report
            .lines()
            .any(|line| line.trim_start().starts_with("Vulkan") && line.contains(':')),
        "preflight did not report the registered Vulkan device:\n{report}"
    );
    assert!(
        report.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("Vulkan") && line.ends_with(": training graph ready")
        }),
        "{report}"
    );
}

#[cfg(retro_vulkan)]
#[test]
fn f16_training_kv_is_effective_and_differentiable_on_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.kv_dtype = retrograd::KvDtype::F16;
    let mut trainer = Trainer::new(model, config).expect("load F16-KV Vulkan model");
    let report = trainer.backend_report().expect("F16-KV backend report");
    // The F16 cache is gated behind the *F32* backward Flash Attention probe
    // (retro_backend.cpp), so assert that capability explicitly: an F16-only
    // shader variant leaves this unavailable and silently falls back to F32.
    assert!(
        report.contains("cap_flash_attn_back: supported"),
        "{report}"
    );
    assert!(
        report.contains("training_kv_requested_dtype: F16"),
        "{report}"
    );
    assert!(report.contains("training_kv_dtype: F16"), "{report}");
    assert!(report.contains("training_kv_f16: supported"), "{report}");

    // Resolved from the model: a hybrid architecture has no attention
    // projection in block 0.
    let targets = common::block_targets(&trainer, &["attn_k", "attn_v"])
        .expect("the model has an attention block");
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(targets);
    trainer.create_lora(&lora).expect("create K/V LoRA");
    let preflight = trainer.train_preflight().expect("F16-KV preflight");
    assert!(
        preflight.contains("missing_gradient_rules: 0"),
        "{preflight}"
    );
    assert!(
        preflight.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("Vulkan") && line.ends_with(": training graph ready")
        }),
        "{preflight}"
    );

    let text = "Differentiable F16 keys and values train on Vulkan. ".repeat(40);
    let tokens = trainer
        .tokenize_text(&text)
        .expect("tokenize F16-KV smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train F16-KV K/V LoRA");
    assert!(metrics.train_loss.is_finite());
    assert!(metrics.eval_loss.is_finite());
}

#[cfg(retro_vulkan)]
#[test]
fn lora_tensors_are_allocated_on_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load Gemma on Vulkan");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create Vulkan LoRA");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("{report}");
    assert!(
        common::section_has_vulkan(&report, "lora_tensors_by_buffer"),
        "LoRA tensors were not allocated on Vulkan:\n{report}"
    );
}

#[cfg(retro_vulkan)]
#[test]
fn minimal_lora_step_updates_only_the_adapter_on_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load Gemma on Vulkan");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create Vulkan LoRA");
    let text = "Vulkan trains a small adapter while the base model stays frozen. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("run minimal Vulkan LoRA step");
    assert!(metrics.train_loss.is_finite());
    assert!(metrics.eval_loss.is_finite());
}

/// A real training step with the fused cross-entropy, pinned GPU-resident - the
/// Vulkan pendant of `fused_cross_entropy_trains_gpu_resident_on_metal`, and what
/// closes the remaining fused-CE fallback coverage gap.
///
/// `tests/fused_ce.rs` proves the Vulkan kernels match the CPU oracle on loss and
/// `grad_h`; this proves the training graph actually dispatches them. With
/// `require_gpu_resident` set, a single node handed back to the CPU fails the
/// preflight instead of quietly costing a transfer of the hidden states and of
/// the whole projection head per token chunk.
#[cfg(retro_vulkan)]
#[test]
fn fused_cross_entropy_trains_gpu_resident_on_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.chunked_cross_entropy = true;
    config.require_gpu_resident = true;
    let mut trainer = Trainer::new(model, config).expect("load fused-CE Vulkan trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create Vulkan LoRA");

    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- Vulkan fused-CE report ---\n{report}");
    assert!(
        report.contains("cap_fused_sparse_ce: supported"),
        "{report}"
    );
    assert!(
        report.contains("chunked_cross_entropy: enabled"),
        "{report}"
    );

    let text = "Vulkan computes the fused cross-entropy without leaving the device. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one epoch with the fused CE on Vulkan");
    eprintln!(
        "vulkan fused-CE train: train_loss={} eval_loss={}",
        metrics.train_loss, metrics.eval_loss
    );
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

/// The dense cross-entropy forward, GPU-resident.
/// This is the *default* configuration - `chunked_cross_entropy` is off - and
/// before the `cross_entropy_loss.comp` shader existed it meant a scheduler split
/// plus a device→host transfer of the dense `[n_vocab, n_ubatch]` logits at every
/// ubatch evaluation. `require_gpu_resident` is what makes that regression loud:
/// were the forward to stop being dispatched, this fails instead of getting slow.
#[cfg(retro_vulkan)]
#[test]
fn dense_cross_entropy_trains_gpu_resident_on_vulkan() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut config = small_config(Device::Gpu);
    config.require_gpu_resident = true;
    let mut trainer = Trainer::new(model, config).expect("load dense-CE Vulkan trainer");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create Vulkan LoRA");

    let preflight = trainer
        .train_preflight()
        .expect("the dense CE path must be fully device-resident on Vulkan");
    eprintln!("--- Vulkan dense-CE preflight ---\n{preflight}");
    assert!(
        preflight.contains("active_device_fallback_nodes: 0"),
        "{preflight}"
    );

    let text = "Vulkan computes the dense cross-entropy on device. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one epoch with the dense CE on Vulkan");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
    assert!(metrics.eval_loss.is_finite(), "{metrics:?}");
}

#[cfg(retro_vulkan)]
#[test]
fn f16_lora_optimizer_runs_on_vulkan_without_cpu_fallback() {
    let _lock = common::serialize_models();
    let Some(model) = vulkan_model_or_skip() else {
        return;
    };
    let mut lora = single_layer_lora();
    lora.dtype = retrograd::LoraDtype::F16;
    let mut trainer = Trainer::new(model, small_config(Device::Gpu)).expect("load Gemma on Vulkan");
    trainer.create_lora(&lora).expect("create F16 Vulkan LoRA");
    let report = trainer.backend_report().expect("backend report");
    assert!(report.contains("lora_dtype: F16"), "{report}");
    assert!(report.contains("optimizer_f16: supported"), "{report}");
    assert!(
        common::section_has_vulkan(&report, "lora_tensors_by_buffer"),
        "F16 LoRA tensors were not allocated on Vulkan:\n{report}"
    );
    let text = "Vulkan performs the F16 AdamW update on device. ".repeat(40);
    let tokens = trainer.tokenize_text(&text).expect("tokenize smoke text");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train F16 Vulkan LoRA");
    assert!(metrics.train_loss.is_finite());
    assert!(metrics.eval_loss.is_finite());
}

#[cfg(retro_vulkan)]
#[test]
fn f16_adamw_vulkan_kernel_matches_cpu() {
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
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
    assert_eq!(
        cpu, gpu,
        "Vulkan F16 AdamW must match CPU after F16 rounding"
    );
}

/// A wide training step with a real vocabulary head, on the geometry that used
/// to lose the device on MoltenVK: `n_ctx = 1024` split into 256-token physical
/// micro-batches, so one `FUSED_SPARSE_CE` node covers 256 tokens against a
/// 65536-wide Q6_K head. Before the token tile a single such dispatch ran for seconds
/// and Metal aborted the command buffer - `VK_ERROR_DEVICE_LOST`, reported as a
/// fence wait failure with no hint of which node was at fault.
///
/// The fixture is the CPU one: it is a hybrid model with a tied Q6_K head, which
/// is exactly the shape that made the kernel expensive.
#[cfg(retro_vulkan)]
#[test]
fn wide_micro_batch_trains_without_losing_the_device_on_vulkan() {
    let _lock = common::serialize_models();
    if !vulkan_available() {
        eprintln!("skipping: no Vulkan device available");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no CPU fixture, run scripts/fetch-cpu-fixture.sh");
        return;
    };
    const N_CTX: usize = 1024;
    const N_ROWS: usize = 2;
    let config = TrainConfig {
        n_ctx: N_CTX as u32,
        n_batch: N_CTX as u32,
        n_ubatch: 256,
        n_seq_max: 8,
        epochs: 1,
        learning_rate: 1.0e-5,
        device: Device::Gpu,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(model, config).expect("load the fixture on Vulkan");
    trainer
        .create_lora(&single_layer_lora())
        .expect("create Vulkan LoRA");
    let report = trainer.backend_report().expect("backend report");
    assert!(
        report.contains("cap_fused_sparse_ce: supported"),
        "the fused CE is what this guards; without it the test proves nothing:\n{report}"
    );

    // One row per sequence, trainable only over the last 128 positions: a long
    // prompt followed by a short completion, like a GRPO rollout.
    let vocab = trainer.vocab_size().expect("vocab size");
    let mut tokens = Vec::with_capacity(N_ROWS * N_CTX);
    let mut labels = Vec::with_capacity(N_ROWS * N_CTX);
    let mut weights = Vec::with_capacity(N_ROWS * N_CTX);
    for row in 0..N_ROWS {
        for pos in 0..N_CTX {
            let token = ((row * 7 + pos * 13) % (vocab - 16) + 8) as i32;
            tokens.push(token);
            let trainable = pos + 128 >= N_CTX && pos + 1 < N_CTX;
            labels.push(if trainable { token } else { -1 });
            weights.push(if trainable {
                1.0 / (N_ROWS * 128) as f32
            } else {
                0.0
            });
        }
    }
    let batch = retrograd::WeightedBatch {
        tokens,
        labels,
        weights,
        n_rows: N_ROWS,
        n_ctx: N_CTX,
        n_topk: 1,
    };
    let metrics = trainer
        .train_weighted(&batch, 0)
        .expect("a 256-token fused CE node must not lose the device");
    assert!(metrics.train_loss.is_finite(), "{metrics:?}");
}
