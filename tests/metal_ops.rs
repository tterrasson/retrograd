//! Reusable Metal-vs-CPU correctness harness for hand-written ggml training ops.
//!
//! Each op is validated in isolation: identical random inputs are run once on the
//! CPU backend (reference) and once on Metal (our kernel), and the outputs must
//! match within a tight tolerance. New ops plug in by adding a case here.

mod common;

#[cfg(retro_metal)]
use retrograd::{ProbeInputs, ProbeOp, probe_op};

// The helpers below serve only `#[cfg(retro_metal)]` tests, so they carry the
// same gate. A helper compiled without Metal is not dead code, it is off-topic
// code - gating it is what keeps a CPU-only build warning-free
// without a blanket `allow(dead_code)` that would also hide a real one.

/// Deterministic pseudo-random f32s in [-range, range] (no rand dependency).
#[cfg(retro_metal)]
fn pseudo_random(n: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D) >> 40; // 24 bits
            let unit = (bits as f32) / ((1u32 << 24) as f32); // [0,1)
            (unit * 2.0 - 1.0) * range
        })
        .collect()
}

/// Normalises each `width`-wide row of `v` to sum to 1 (a valid softmax output /
/// label distribution). Inputs are shifted to positive via exp first.
#[cfg(retro_metal)]
fn softmax_rows(v: &[f32], width: usize) -> Vec<f32> {
    assert_eq!(v.len() % width, 0);
    let mut out = Vec::with_capacity(v.len());
    for row in v.chunks(width) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e / sum));
    }
    out
}

/// Asserts elementwise closeness of two probe outputs and logs the max diff.
#[cfg(retro_metal)]
fn assert_close(op: ProbeOp, case: usize, cpu: &[f32], gpu: &[f32], tol: f32) {
    let mut max_abs = 0.0_f32;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let d = (c - g).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= tol,
            "{op:?} case {case} mismatch at {i}: cpu={c} metal={g} diff={d} (tol={tol})"
        );
    }
    eprintln!(
        "{op:?} case {case}: {} elems, max|cpu-metal| = {max_abs:e}",
        cpu.len()
    );
}

/// Runs a same-shape binary op on both backends and asserts closeness.
#[cfg(retro_metal)]
fn assert_metal_matches_cpu(
    op: ProbeOp,
    ne: [i64; 4],
    src0: &[f32],
    src1: &[f32],
    params: [f32; 2],
    tol: f32,
) {
    let n = ne.iter().product::<i64>() as usize;
    let cpu =
        probe_op(op, false, ProbeInputs::pair(ne, src0, ne, src1), params, n).expect("cpu probe");
    let gpu =
        probe_op(op, true, ProbeInputs::pair(ne, src0, ne, src1), params, n).expect("metal probe");
    assert_close(op, 0, &cpu, &gpu, tol);
}

#[cfg(retro_metal)]
#[test]
fn silu_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // A few shapes: a flat vector, a 2D matrix, and a non-power-of-two size that
    // forces a partial final threadgroup.
    let shapes: [[i64; 4]; 3] = [[4096, 1, 1, 1], [512, 33, 1, 1], [1000, 1, 1, 1]];
    for (k, ne) in shapes.iter().enumerate() {
        let n = ne.iter().product::<i64>() as usize;
        let dy = pseudo_random(n, 0x1234 + k as u64, 1.0);
        let x = pseudo_random(n, 0x9999 + k as u64, 6.0); // wide range exercises the sigmoid tails
        assert_metal_matches_cpu(ProbeOp::SiluBack, *ne, &dy, &x, [0.0, 0.0], 1e-5);
    }
}

#[cfg(retro_metal)]
#[test]
fn rms_norm_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // [ne0 (row width), nrows,...]. Include non-multiple-of-32 row widths to
    // exercise the partial simdgroup reduction (33 is the regression case for
    // the nth-clamp dispatch bug), and multiple rows/batches.
    let shapes: [[i64; 4]; 4] = [
        [1024, 8, 1, 1],
        [100, 4, 1, 1],
        [256, 3, 2, 1],
        [33, 5, 1, 1],
    ];
    let eps = 1e-5_f32;
    for (k, ne) in shapes.iter().enumerate() {
        let n = ne.iter().product::<i64>() as usize;
        let dy = pseudo_random(n, 0x2222 + k as u64, 1.0);
        let x = pseudo_random(n, 0x5555 + k as u64, 2.0);
        // Reduction order differs from CPU -> looser tolerance.
        assert_metal_matches_cpu(ProbeOp::RmsNormBack, *ne, &dy, &x, [eps, 0.0], 2e-4);
    }
}

#[cfg(retro_metal)]
#[test]
fn l2_norm_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    let ne = [100_i64, 4, 1, 1];
    let n = ne.iter().product::<i64>() as usize;
    let dy = pseudo_random(n, 0x2244, 1.0);
    let x = pseudo_random(n, 0x6688, 2.0);
    assert_metal_matches_cpu(ProbeOp::L2NormBack, ne, &dy, &x, [1e-5, 0.0], 2e-4);

    // Exercise the forward norm clamp and its constant-scale derivative.
    let tiny_x = vec![1e-9_f32; n];
    assert_metal_matches_cpu(ProbeOp::L2NormBack, ne, &dy, &tiny_x, [0.1, 0.0], 2e-4);
}

#[cfg(retro_metal)]
#[test]
fn out_prod_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // out_prod(a, b): a=[M,K], b=[N,K] -> dst=[M,N]; dst[m,n] = Σ_k a[m,k]*b[n,k].
    // Test a plain matrix, a batched case (ne2>1), and the rank-1 LoRA shape.
    struct Case {
        ne_a: [i64; 4],
        ne_b: [i64; 4],
        out_len: usize,
    }
    let cases = [
        Case {
            ne_a: [4, 3, 1, 1],
            ne_b: [5, 3, 1, 1],
            out_len: 4 * 5,
        },
        Case {
            ne_a: [8, 7, 2, 1],
            ne_b: [6, 7, 2, 1],
            out_len: 8 * 6 * 2,
        },
        Case {
            ne_a: [1024, 16, 1, 1],
            ne_b: [1, 16, 1, 1],
            out_len: 1024,
        }, // rank-1 LoRA
    ];
    for (i, c) in cases.iter().enumerate() {
        let na = c.ne_a.iter().product::<i64>() as usize;
        let nb = c.ne_b.iter().product::<i64>() as usize;
        let a = pseudo_random(na, 0x71 + i as u64, 1.0);
        let b = pseudo_random(nb, 0x91 + i as u64, 1.0);
        let cpu = probe_op(
            ProbeOp::OutProd,
            false,
            ProbeInputs::pair(c.ne_a, &a, c.ne_b, &b),
            [0.0, 0.0],
            c.out_len,
        )
        .expect("cpu out_prod");
        let gpu = probe_op(
            ProbeOp::OutProd,
            true,
            ProbeInputs::pair(c.ne_a, &a, c.ne_b, &b),
            [0.0, 0.0],
            c.out_len,
        )
        .expect("metal out_prod");
        assert_close(ProbeOp::OutProd, i, &cpu, &gpu, 1e-3);
    }
}

#[cfg(retro_metal)]
#[test]
fn out_prod_q8_0_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // Same contraction as OutProd but src0 is quantized to Q8_0 inside the
    // probe (both backends dequantize identical blocks). ne_a[0] must be a
    // multiple of 32. Cover: single block, several blocks per row, a batched
    // case, and the real activation-gradient shape dx = out_prod(W, dy) with a
    // weight-sized src0 and a skinny src1.
    struct Case {
        ne_a: [i64; 4],
        ne_b: [i64; 4],
        out_len: usize,
    }
    let cases = [
        Case {
            ne_a: [32, 3, 1, 1],
            ne_b: [5, 3, 1, 1],
            out_len: 32 * 5,
        },
        Case {
            ne_a: [96, 7, 1, 1],
            ne_b: [6, 7, 1, 1],
            out_len: 96 * 6,
        },
        Case {
            ne_a: [64, 5, 2, 1],
            ne_b: [4, 5, 2, 1],
            out_len: 64 * 4 * 2,
        },
        Case {
            ne_a: [1024, 512, 1, 1],
            ne_b: [16, 512, 1, 1],
            out_len: 1024 * 16,
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        let na = c.ne_a.iter().product::<i64>() as usize;
        let nb = c.ne_b.iter().product::<i64>() as usize;
        let a = pseudo_random(na, 0xd1 + i as u64, 1.0);
        let b = pseudo_random(nb, 0xe1 + i as u64, 1.0);
        let cpu = probe_op(
            ProbeOp::OutProdQ80,
            false,
            ProbeInputs::pair(c.ne_a, &a, c.ne_b, &b),
            [0.0, 0.0],
            c.out_len,
        )
        .expect("cpu out_prod q8_0");
        let gpu = probe_op(
            ProbeOp::OutProdQ80,
            true,
            ProbeInputs::pair(c.ne_a, &a, c.ne_b, &b),
            [0.0, 0.0],
            c.out_len,
        )
        .expect("metal out_prod q8_0");
        assert!(
            cpu.iter().any(|v| v.abs() > 1e-3),
            "OutProdQ80 case {i}: CPU output unexpectedly all ~0"
        );
        assert_close(ProbeOp::OutProdQ80, i, &cpu, &gpu, 1e-3);
    }
}

#[cfg(retro_metal)]
#[test]
fn out_prod_q5_0_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // Q5_0 is the quantization used by gemma-3-270m's activation-gradient
    // OUT_PROD nodes. Include multiple blocks, batches, and a skinny dy.
    let cases = [
        ([32, 3, 1, 1], [5, 3, 1, 1]),
        ([96, 7, 2, 1], [6, 7, 2, 1]),
        ([640, 128, 1, 1], [32, 128, 1, 1]),
    ];
    for (i, (ne_a, ne_b)) in cases.into_iter().enumerate() {
        let a = pseudo_random(ne_a.iter().product::<i64>() as usize, 0xf1 + i as u64, 1.0);
        let b = pseudo_random(ne_b.iter().product::<i64>() as usize, 0x101 + i as u64, 1.0);
        let out_len = (ne_a[0] * ne_b[0] * ne_b[2] * ne_b[3]) as usize;
        let cpu = probe_op(
            ProbeOp::OutProdQ50,
            false,
            ProbeInputs::pair(ne_a, &a, ne_b, &b),
            [0.0, 0.0],
            out_len,
        )
        .expect("cpu out_prod q5_0");
        let gpu = probe_op(
            ProbeOp::OutProdQ50,
            true,
            ProbeInputs::pair(ne_a, &a, ne_b, &b),
            [0.0, 0.0],
            out_len,
        )
        .expect("metal out_prod q5_0");
        assert_close(ProbeOp::OutProdQ50, i, &cpu, &gpu, 1e-3);
    }
}

#[cfg(retro_metal)]
#[test]
fn out_prod_k_quants_metal_match_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    let quant_ops = [
        ProbeOp::OutProdQ2K,
        ProbeOp::OutProdQ3K,
        ProbeOp::OutProdQ4K,
        ProbeOp::OutProdQ5K,
        ProbeOp::OutProdQ6K,
    ];
    // Exercise two K-blocks, a partial output-row tile, and batched planes.
    let cases = [
        ([256, 7, 1, 1], [11, 7, 1, 1]),
        ([512, 9, 2, 1], [5, 9, 2, 1]),
    ];
    for (op_index, op) in quant_ops.into_iter().enumerate() {
        for (case_index, (ne_a, ne_b)) in cases.into_iter().enumerate() {
            let a = pseudo_random(
                ne_a.iter().product::<i64>() as usize,
                0x120 + 17 * op_index as u64 + case_index as u64,
                1.0,
            );
            let b = pseudo_random(
                ne_b.iter().product::<i64>() as usize,
                0x220 + 17 * op_index as u64 + case_index as u64,
                1.0,
            );
            let out_len = (ne_a[0] * ne_b[0] * ne_b[2] * ne_b[3]) as usize;
            let cpu = probe_op(
                op,
                false,
                ProbeInputs::pair(ne_a, &a, ne_b, &b),
                [0.0, 0.0],
                out_len,
            )
            .expect("cpu K-quant out_prod");
            let gpu = probe_op(
                op,
                true,
                ProbeInputs::pair(ne_a, &a, ne_b, &b),
                [0.0, 0.0],
                out_len,
            )
            .expect("metal K-quant out_prod");
            assert_close(op, case_index, &cpu, &gpu, 2e-3);
        }
    }
}

/// Every type the fork claims a training kernel can decode in place must produce
/// the CPU's `out_prod` on Metal -- not a hand-picked subset. See
/// `common::assert_out_prod_all_dequant_types_match_cpu` for why this is driven by
/// the fork's own table and why it is not vacuous.
#[cfg(retro_metal)]
#[test]
fn out_prod_all_dequant_types_metal_match_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    common::assert_out_prod_all_dequant_types_match_cpu("metal");
}

/// The other half of the sweep above: a type the table does not list must be refused,
/// not aborted on. See `common::assert_out_prod_rejects_unlisted_type`.
#[test]
fn out_prod_quant_rejects_a_type_outside_the_table() {
    common::assert_out_prod_rejects_unlisted_type(false);
    if common::gpu_device_present() {
        common::assert_out_prod_rejects_unlisted_type(true);
    }
}

#[cfg(retro_metal)]
#[test]
fn ssm_conv_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // [ncs,d_inner,n_s], [d_conv,d_inner], [d_inner,n_t,n_s]. Cover multiple
    // sequences and both short and Falcon-H1-like 4-tap kernels.
    let cases = [(3_i64, 4_i64, 5_i64, 1_i64), (4, 7, 16, 2)];
    for (i, (d_conv, d_inner, n_t, n_s)) in cases.into_iter().enumerate() {
        let ncs = d_conv - 1 + n_t;
        let ne_sx = [ncs, d_inner, n_s, 1];
        let ne_c = [d_conv, d_inner, 1, 1];
        let ne_dy = [d_inner, n_t, n_s, 1];
        let sx = pseudo_random(
            ne_sx.iter().product::<i64>() as usize,
            0x121 + i as u64,
            1.0,
        );
        let c = pseudo_random(ne_c.iter().product::<i64>() as usize, 0x141 + i as u64, 1.0);
        let dy = pseudo_random(
            ne_dy.iter().product::<i64>() as usize,
            0x161 + i as u64,
            1.0,
        );
        let out_len = sx.len() + c.len();
        let cpu = probe_op(
            ProbeOp::SsmConvBack,
            false,
            ProbeInputs::pair(ne_sx, &sx, ne_c, &c).with_src2(Some((ne_dy, &dy))),
            [0.0, 0.0],
            out_len,
        )
        .expect("cpu ssm_conv_back");
        let gpu = probe_op(
            ProbeOp::SsmConvBack,
            true,
            ProbeInputs::pair(ne_sx, &sx, ne_c, &c).with_src2(Some((ne_dy, &dy))),
            [0.0, 0.0],
            out_len,
        )
        .expect("metal ssm_conv_back");
        assert_close(ProbeOp::SsmConvBack, i, &cpu, &gpu, 2e-4);
    }
}

#[cfg(retro_metal)]
#[test]
fn ssm_scan_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // Mamba-1 uses A=[d_state,n_head]; Mamba-2/Falcon-H1 uses A=[1,n_head].
    // Small dimensions keep this test quick, while grouped heads and permuted
    // state slots exercise every accumulator. The third case crosses the
    // 32-token chunk boundary of the two-pass Metal kernel (checkpoint replay
    // and lambda carry between chunk dispatches).
    let cases = [
        (3_i64, 1_i64, 2_i64, 1_i64, 3_i64, 2_i64, 3_i64),
        (4, 2, 4, 2, 4, 2, 1),
        (4, 2, 4, 2, 40, 2, 1),
    ];
    for (case, (nc, nr, nh, ng, nt, ns, n_a0)) in cases.into_iter().enumerate() {
        let nslot = ns;
        let n_s = (nc * nr * nh * nslot) as usize;
        let n_x = (nr * nh * nt * ns) as usize;
        let n_dt = (nh * nt * ns) as usize;
        let n_a = (n_a0 * nh) as usize;
        let n_bc = (nc * ng * nt * ns) as usize;
        let n_ds = n_x + (nc * nr * nh * ns) as usize;

        let mut packed = Vec::new();
        packed.extend(pseudo_random(n_s, 0x201 + case as u64, 0.3));
        packed.extend(pseudo_random(n_x, 0x221 + case as u64, 0.3));
        packed.extend(pseudo_random(n_dt, 0x241 + case as u64, 1.0));
        let mut a = pseudo_random(n_a, 0x261 + case as u64, 0.5);
        for v in &mut a {
            *v = -v.abs() - 0.05;
        }
        packed.extend(a);
        packed.extend(pseudo_random(n_bc, 0x281 + case as u64, 0.3));
        packed.extend(pseudo_random(n_bc, 0x2a1 + case as u64, 0.3));
        for i in 0..ns {
            packed.push((ns - 1 - i) as f32);
        }
        packed.extend(pseudo_random(n_ds, 0x2c1 + case as u64, 0.3));

        let out_len = n_x + n_dt + n_a + 2 * n_bc + n_s;
        let dims = [nc, nr, nh, nslot];
        let meta = [ng, nt, ns, n_a0];
        let dummy = [0.0_f32];
        let cpu = probe_op(
            ProbeOp::SsmScanBack,
            false,
            ProbeInputs::pair(dims, &packed, meta, &dummy),
            [0.0, 0.0],
            out_len,
        )
        .expect("cpu ssm_scan_back");
        let gpu = probe_op(
            ProbeOp::SsmScanBack,
            true,
            ProbeInputs::pair(dims, &packed, meta, &dummy),
            [0.0, 0.0],
            out_len,
        )
        .expect("metal ssm_scan_back");
        // Atomic accumulation changes summation order relative to CPU.
        assert_close(ProbeOp::SsmScanBack, case, &cpu, &gpu, 5e-4);
    }
}

/// `geometry` is `(state_width, heads, tokens, sequences)`. The Metal grid's x
/// axis is `ceil(state_width / GDN_BACK_COLS)` column blocks, so the geometry
/// decides whether grad_q/grad_k/grad_g/grad_beta accumulate in one threadgroup
/// or across several via device atomics: pass a geometry that yields several
/// blocks to cover the atomic path.
/// Returns the packed `[q|k|v|g|beta|state|grad]` input the probe takes, the two
/// shape vectors that describe it, and the length of the packed gradient it
/// produces. Shared by the comparison cases and the timing benchmark.
#[cfg(retro_metal)]
fn gated_delta_net_back_inputs(
    kda: bool,
    snapshots: i64,
    geometry: (i64, i64, i64, i64),
) -> ([i64; 4], [i64; 4], Vec<f32>, usize) {
    let (state_width, heads, tokens, sequences) = geometry;
    let ne_src0 = [state_width, heads, tokens, sequences];
    let ne_src1 = [snapshots, if kda { 1 } else { 0 }, 1, 1];
    let n_qkv = state_width * heads * tokens * sequences;
    let n_gate = (if kda { state_width } else { 1 }) * heads * tokens * sequences;
    let n_beta = heads * tokens * sequences;
    let n_state = state_width * state_width * heads * sequences;
    let n_grad = n_qkv + snapshots * n_state;

    let mut packed = Vec::new();
    packed.extend(pseudo_random(n_qkv as usize, 0x6601, 0.5)); // q
    packed.extend(pseudo_random(n_qkv as usize, 0x6602, 0.5)); // k
    packed.extend(pseudo_random(n_qkv as usize, 0x6603, 0.5)); // v
    packed.extend(
        pseudo_random(n_gate as usize, 0x6604, 2.0)
            .iter()
            .map(|v| -v.abs() - 0.1),
    );
    packed.extend(
        pseudo_random(n_beta as usize, 0x6605, 0.5)
            .iter()
            .map(|v| v.abs()),
    );
    packed.extend(pseudo_random(n_state as usize, 0x6606, 0.5));
    packed.extend(pseudo_random(n_grad as usize, 0x6607, 0.5));

    let out_len = (3 * n_qkv + n_gate + n_beta + n_state) as usize;
    (ne_src0, ne_src1, packed, out_len)
}

#[cfg(retro_metal)]
fn gated_delta_net_back_case(kda: bool, snapshots: i64, geometry: (i64, i64, i64, i64)) {
    let (ne_src0, ne_src1, packed, out_len) = gated_delta_net_back_inputs(kda, snapshots, geometry);
    let dummy = [0.0_f32];
    let cpu = probe_op(
        ProbeOp::GatedDeltaNetBack,
        false,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy),
        [0.0, 0.0],
        out_len,
    )
    .expect("cpu gated_delta_net_back");
    let metal = probe_op(
        ProbeOp::GatedDeltaNetBack,
        true,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy),
        [0.0, 0.0],
        out_len,
    )
    .expect("metal gated_delta_net_back");
    assert_close(
        ProbeOp::GatedDeltaNetBack,
        snapshots as usize,
        &cpu,
        &metal,
        5.0e-4,
    );
}

#[cfg(retro_metal)]
#[test]
fn gated_delta_net_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    gated_delta_net_back_case(false, 1, (32, 4, 20, 2));
    gated_delta_net_back_case(true, 3, (32, 4, 20, 2));
}

/// The head dimension Qwen3.5 and Qwen3-Next train at, and the widest column
/// split the kernel takes: 128 columns over blocks of `GDN_BACK_COLS`, so every
/// cross-column output is accumulated by 32 threadgroups at once.
#[cfg(retro_metal)]
#[test]
fn gated_delta_net_back_metal_matches_cpu_at_the_production_head_dim() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    gated_delta_net_back_case(false, 1, (128, 4, 12, 1));
    gated_delta_net_back_case(true, 2, (128, 4, 12, 1));
}

/// A state width the split does not divide, so the last column block is short.
/// 17 columns become four blocks of `GDN_BACK_COLS` and a tail of one: that
/// tail is the only place where a thread can walk past the end of its slice.
#[cfg(retro_metal)]
#[test]
fn gated_delta_net_back_metal_matches_cpu_on_a_ragged_column_block() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    gated_delta_net_back_case(false, 1, (17, 3, 9, 2));
    gated_delta_net_back_case(true, 3, (17, 3, 9, 2));
}

/// A state too narrow to split, which is the pre-split kernel: one threadgroup
/// owns every column, and the atomics on grad_g/grad_beta have a single
/// contributor. Keeps that path tested now that it is no longer the only one.
#[cfg(retro_metal)]
#[test]
fn gated_delta_net_back_metal_keeps_the_unsplit_path() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    gated_delta_net_back_case(false, 1, (4, 3, 9, 2));
    gated_delta_net_back_case(true, 2, (4, 3, 9, 2));
}

/// Wall-clock smoke benchmark for the backward scan, not a correctness test:
/// `#[ignore]`d so no lane pays for it. Run it with
/// `cargo test --release --test metal_ops -- --ignored --nocapture gated_delta_net_back_metal_timing`.
///
/// The shapes are the two a Qwen3.5 optimizer step dispatches: one sequence
/// (the model cannot pack) and four (the packed case). They differ only on the
/// grid's second axis. The one-sequence row is what `GDN_BACK_COLS` was sized
/// for: without the column split its grid is 16 threadgroups.
#[cfg(retro_metal)]
#[test]
#[ignore]
fn gated_delta_net_back_metal_timing() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    // (state width, heads, tokens, sequences): Qwen3.5 at ctx 512.
    for geometry in [(128_i64, 16_i64, 512_i64, 1_i64), (128, 16, 512, 4)] {
        let (ne_src0, ne_src1, packed, out_len) = gated_delta_net_back_inputs(false, 1, geometry);
        let dummy = [0.0_f32];
        let run = || {
            probe_op(
                ProbeOp::GatedDeltaNetBack,
                true,
                ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy),
                [0.0, 0.0],
                out_len,
            )
            .expect("metal gated_delta_net_back")
        };
        // One warm-up run so pipeline compilation is not in the measurement.
        run();

        let runs = 5;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            run();
        }
        let elapsed = start.elapsed();
        let (s_v, h, t, n) = geometry;
        eprintln!(
            "gated_delta_net_back metal [S_v={s_v} H={h} tokens={t} seqs={n}]: \
             {:.1} ms/run over {runs} runs",
            elapsed.as_secs_f64() * 1000.0 / f64::from(runs)
        );
    }
}

#[cfg(retro_metal)]
fn conv_rs_gather_case(kernel_m1: i64, n_seq_tokens: i64, snapshots: i64) {
    let (channels, sequences) = (5_i64, 3_i64);
    let ne0 = kernel_m1 + n_seq_tokens;
    let ne_src0 = [ne0, channels, sequences, 1];
    let src0 = pseudo_random((ne0 * channels * sequences) as usize, 0x5c07, 1.0);
    let ne_src1 = [1_i64, 1, 1, 1];
    let dummy = [0.0_f32];
    let out_len = (kernel_m1 * channels * sequences * snapshots) as usize;
    let params = [kernel_m1 as f32, snapshots as f32];
    let cpu = probe_op(
        ProbeOp::ConvRsGather,
        false,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &dummy),
        params,
        out_len,
    )
    .expect("cpu conv_rs_gather");
    let metal = probe_op(
        ProbeOp::ConvRsGather,
        true,
        ProbeInputs::pair(ne_src0, &src0, ne_src1, &dummy),
        params,
        out_len,
    )
    .expect("metal conv_rs_gather");
    assert_close(ProbeOp::ConvRsGather, snapshots as usize, &cpu, &metal, 0.0);
}

#[cfg(retro_metal)]
#[test]
fn conv_rs_gather_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    conv_rs_gather_case(3, 8, 1);
    conv_rs_gather_case(3, 8, 5);
    conv_rs_gather_case(3, 2, 6);
}

// Streaming Flash Attention backward - the kernel that makes an F16 KV cache
// differentiable (`kv_dtype = "f16"`). `window`, when Some((nwin, kv_off)),
// exercises the KV gradient window: dK/dV cover `nwin` cache rows starting at
// `kv_off` (per batch) instead of all `nkv` - the n_kv > n_window case the dense
// tests never reach. Mirrors the Vulkan/CUDA cases in tests/{vulkan,cuda}_backend.rs.
#[cfg(retro_metal)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back(
    case: usize,
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
    compare_flash_attn_back_kv(
        case, hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch, window, false, tolerance,
    );
}

// `kv_f32` stores K/V as F32 rather than F16, selecting the other kernel variant.
// Both matter: `cap_flash_attn_back` is probed with an F32 cache, and clearing
// that gate is the precondition for `kv_dtype = "f16"` engaging at all (see
// retro_backend.cpp).
#[cfg(retro_metal)]
#[allow(clippy::too_many_arguments)]
fn compare_flash_attn_back_kv(
    case: usize,
    hsk: usize,
    hsv: usize,
    nq: usize,
    nkv: usize,
    nhead: usize,
    nhead_kv: usize,
    nbatch: usize,
    window: Option<(usize, usize)>,
    kv_f32: bool,
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
        0,
        0,
    ];
    let src2 = if window.is_some() || kv_f32 {
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
    let metal = probe_op(
        ProbeOp::FlashAttnBack,
        true,
        ProbeInputs::pair(dims, &packed, layout, &layout_data).with_src2(src2),
        params,
        out_len,
    )
    .expect("Metal Flash Attention backward probe");
    assert_close(ProbeOp::FlashAttnBack, case, &cpu, &metal, tolerance);
}

#[cfg(retro_metal)]
#[test]
fn flash_attn_back_metal_matches_streaming_cpu_reference() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    compare_flash_attn_back(0, 32, 32, 5, 8, 4, 2, 1, None, 7.5e-4);
}

#[cfg(retro_metal)]
#[test]
fn flash_attn_back_metal_covers_wide_head_dims() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    // Head dim 256 (Qwen3.5/GDN, Gemma2) selects the deeper kernel variant; 128
    // stays on the shallow one. See ggml_metal_fa_back_bucket_suffix().
    compare_flash_attn_back(1, 128, 128, 4, 10, 8, 2, 1, None, 1.5e-3);
    compare_flash_attn_back(2, 256, 256, 4, 8, 4, 2, 1, None, 2.0e-3);
}

// KV gradient window: dK/dV cover only the rows written at this step, so n_kv
// (what the forward reads) exceeds n_window (what receives a gradient). This is
// the shape production hits at a long context and the dense tests never do,
// including kv_off > 0 which catches a window indexed off by a row.
#[cfg(retro_metal)]
#[test]
fn flash_attn_back_metal_covers_kv_gradient_window() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    // Window strictly inside a larger cache, with and without an offset.
    compare_flash_attn_back(3, 32, 32, 4, 24, 4, 2, 1, Some((4, 0)), 7.5e-4);
    compare_flash_attn_back(4, 32, 32, 4, 24, 4, 2, 1, Some((4, 8)), 7.5e-4);
    // Multi-stream (nbatch > 1) offset window, wide head dim.
    compare_flash_attn_back(5, 256, 256, 5, 40, 4, 2, 2, Some((5, 7)), 2.0e-3);
    // Degenerate case: window == whole cache must match the dense path.
    compare_flash_attn_back(6, 32, 32, 5, 5, 4, 2, 1, Some((5, 0)), 7.5e-4);
}

// F32 K/V selects the other kernel variant, and it is the one the capability
// probe actually builds (retro_backend.cpp probes with GGML_TYPE_F32), so the F16
// path is only reachable once this one is correct. No F16 rounding anywhere in the
// operands, hence a much tighter tolerance than the F16 cases above.
#[cfg(retro_metal)]
#[test]
fn flash_attn_back_metal_covers_f32_kv_cache() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    compare_flash_attn_back_kv(7, 32, 32, 5, 8, 4, 2, 1, None, true, 1.0e-5);
    compare_flash_attn_back_kv(8, 128, 128, 4, 10, 8, 2, 1, None, true, 2.0e-5);
    compare_flash_attn_back_kv(9, 256, 256, 4, 8, 4, 2, 1, None, true, 2.0e-5);
    // Offset gradient window, multi-stream.
    compare_flash_attn_back_kv(10, 128, 128, 4, 24, 4, 2, 2, Some((4, 8)), true, 2.0e-5);
}

#[cfg(retro_metal)]
#[test]
fn soft_max_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // y must be a valid softmax output (rows sum to 1). Sweep the attention
    // scale, and include non-multiple-of-32 row widths (33 exercises the
    // partial-simdgroup reduction path).
    let shapes: [[i64; 4]; 4] = [
        [128, 128, 4, 1],
        [512, 16, 1, 1],
        [100, 7, 1, 1],
        [33, 5, 1, 1],
    ];
    let scales = [1.0_f32, 0.088_388_35]; // 1/sqrt(128) is the Qwen3 head scale
    for (k, ne) in shapes.iter().enumerate() {
        let n = ne.iter().product::<i64>() as usize;
        let width = ne[0] as usize;
        let dy = pseudo_random(n, 0x3333 + k as u64, 1.0);
        let y = softmax_rows(&pseudo_random(n, 0x7777 + k as u64, 3.0), width);
        for &scale in &scales {
            assert_metal_matches_cpu(ProbeOp::SoftMaxBack, *ne, &dy, &y, [scale, 0.0], 1e-5);
        }
    }
}

#[cfg(retro_metal)]
#[test]
fn cross_entropy_loss_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // Logits random, labels a valid distribution per row (softmax of noise,
    // close to one-hot for a wide range). Vocab-sized rows exercise the strided
    // per-row reduction; 33 the partial-simdgroup path. Output is one scalar.
    let shapes: [[i64; 4]; 3] = [[4096, 7, 1, 1], [151, 3, 1, 1], [33, 2, 1, 1]];
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
        .expect("cpu cross_entropy_loss");
        let gpu = probe_op(
            ProbeOp::CrossEntropyLoss,
            true,
            ProbeInputs::pair(*ne, &logits, *ne, &labels),
            [0.0, 0.0],
            1,
        )
        .expect("metal cross_entropy_loss");
        assert!(
            cpu[0].abs() > 1e-3,
            "CrossEntropyLoss case {k}: CPU loss unexpectedly ~0 ({})",
            cpu[0]
        );
        assert_close(ProbeOp::CrossEntropyLoss, k, &cpu, &gpu, 1e-4);
    }
}

#[cfg(retro_metal)]
#[test]
fn cross_entropy_loss_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // src0 = grad of the loss (scalar), src1 = logits, src2 = labels.
    let shapes: [[i64; 4]; 3] = [[4096, 7, 1, 1], [151, 3, 1, 1], [33, 2, 1, 1]];
    let ne_grad: [i64; 4] = [1, 1, 1, 1];
    for (k, ne) in shapes.iter().enumerate() {
        let n = ne.iter().product::<i64>() as usize;
        let width = ne[0] as usize;
        let grad = [0.5_f32 + k as f32]; // non-trivial upstream gradient
        let logits = pseudo_random(n, 0x6666 + k as u64, 4.0);
        let labels = softmax_rows(&pseudo_random(n, 0xaaaa + k as u64, 8.0), width);
        let cpu = probe_op(
            ProbeOp::CrossEntropyLossBack,
            false,
            ProbeInputs::pair(ne_grad, &grad, *ne, &logits)
                .with_src2(Some((*ne, labels.as_slice()))),
            [0.0, 0.0],
            n,
        )
        .expect("cpu cross_entropy_loss_back");
        let gpu = probe_op(
            ProbeOp::CrossEntropyLossBack,
            true,
            ProbeInputs::pair(ne_grad, &grad, *ne, &logits)
                .with_src2(Some((*ne, labels.as_slice()))),
            [0.0, 0.0],
            n,
        )
        .expect("metal cross_entropy_loss_back");
        assert!(
            cpu.iter().any(|v| v.abs() > 1e-6),
            "CrossEntropyLossBack case {k}: CPU grad unexpectedly all ~0"
        );
        assert_close(ProbeOp::CrossEntropyLossBack, k, &cpu, &gpu, 1e-5);
    }
}

#[cfg(retro_metal)]
#[test]
fn get_rows_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }

    // src0 = grad rows [n_embd, n_rows], src1 = row indices (with deliberate
    // duplicates, so the atomic scatter-add accumulation is exercised),
    // src2 = shape template [n_embd, n_vocab] (data unused).
    struct Case {
        n_embd: i64,
        n_rows: i64,
        n_vocab: i64,
        indices: Vec<f32>,
    }
    let cases = [
        Case {
            n_embd: 64,
            n_rows: 16,
            n_vocab: 100,
            // duplicates: rows 3 and 42 are hit multiple times
            indices: vec![
                3.0, 42.0, 0.0, 99.0, 3.0, 17.0, 42.0, 3.0, 55.0, 42.0, 1.0, 98.0, 3.0, 0.0, 77.0,
                42.0,
            ],
        },
        Case {
            n_embd: 33, // non-multiple-of-32 row width
            n_rows: 4,
            n_vocab: 10,
            indices: vec![7.0, 7.0, 7.0, 7.0], // all rows collapse into one
        },
    ];
    for (k, c) in cases.iter().enumerate() {
        assert_eq!(c.indices.len(), c.n_rows as usize);
        let ne_a: [i64; 4] = [c.n_embd, c.n_rows, 1, 1];
        let ne_b: [i64; 4] = [c.n_rows, 1, 1, 1];
        let ne_c: [i64; 4] = [c.n_embd, c.n_vocab, 1, 1];
        let n_out = (c.n_embd * c.n_vocab) as usize;
        let grads = pseudo_random((c.n_embd * c.n_rows) as usize, 0xbbbb + k as u64, 1.0);
        let template: Vec<f32> = Vec::new(); // shape-only input, never read
        let cpu = probe_op(
            ProbeOp::GetRowsBack,
            false,
            ProbeInputs::pair(ne_a, &grads, ne_b, &c.indices)
                .with_src2(Some((ne_c, template.as_slice()))),
            [0.0, 0.0],
            n_out,
        )
        .expect("cpu get_rows_back");
        let gpu = probe_op(
            ProbeOp::GetRowsBack,
            true,
            ProbeInputs::pair(ne_a, &grads, ne_b, &c.indices)
                .with_src2(Some((ne_c, template.as_slice()))),
            [0.0, 0.0],
            n_out,
        )
        .expect("metal get_rows_back");
        assert!(
            cpu.iter().any(|v| v.abs() > 1e-6),
            "GetRowsBack case {k}: CPU output unexpectedly all ~0"
        );
        assert_close(ProbeOp::GetRowsBack, k, &cpu, &gpu, 1e-5);
    }
}

#[cfg(retro_metal)]
#[test]
fn silu_back_is_not_all_zero() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    // Guards against a kernel that "matches" only because both sides are zero.
    let ne = [256_i64, 1, 1, 1];
    let dy = pseudo_random(256, 7, 1.0);
    let x = pseudo_random(256, 11, 3.0);
    let out = probe_op(
        ProbeOp::SiluBack,
        true,
        ProbeInputs::pair(ne, &dy, ne, &x),
        [0.0, 0.0],
        256,
    )
    .expect("metal probe");
    assert!(
        out.iter().any(|v| v.abs() > 1e-4),
        "SILU_BACK output is unexpectedly all near-zero"
    );
}

// F16 AdamW: the F16 store rounds stochastically, and the Metal kernel
// reproduces `ggml_stochastic_round_f16` bit for bit - so the assertion is
// equality, not a tolerance. This is the trap the tolerance-based tests above
// cannot catch: an off-by-one-ulp rounding is invisible in the loss and fatal in
// convergence, because the whole point of stochastic rounding is that updates
// below half an ULP must still accumulate. CUDA and Vulkan already assert it
// (`f16_adamw_{cuda,vulkan}_kernel_matches_cpu`); Metal was the one backend
// where the kernel existed with nothing checking it. The BF16 cases below
// repeat the same checks on the other grid.
#[cfg(retro_metal)]
#[test]
fn f16_adamw_metal_kernel_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
    assert_eq!(
        run(false),
        run(true),
        "Metal F16 AdamW must match CPU after F16 rounding"
    );
}

// The single step above shares one rounding seed; this one moves the seed the
// way a real run does (`params[8]`, one value per optimizer step) over enough
// steps that the rounding decisions, not the arithmetic, dominate. A kernel that
// derived its randomness from anything but the CPU's exact formula - thread id,
// a different hash, the seed applied after the update - passes the first test
// and fails this one.
#[cfg(retro_metal)]
#[test]
fn f16_adamw_metal_matches_cpu_across_chained_steps() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let n = 256;
    // Updates ~50x below half an ULP at 1.0: every step is decided purely by the
    // stochastic rounding, so agreement here is agreement on the rounding itself.
    let alpha = 1.0e-5_f32;
    let steps = 64;
    let gradients = vec![1.0_f32; n];

    let chained = |gpu: bool| {
        let ne = [n as i64, 1, 1, 1];
        let mut current = vec![1.0_f32; n];
        for step in 0..steps {
            let scale_and_seed = [1.0_f32, step as f32];
            current = probe_op(
                ProbeOp::OptStepAdamwF16,
                gpu,
                ProbeInputs::pair(ne, &current, ne, &gradients)
                    .with_src2(Some(([2, 1, 1, 1], scale_and_seed.as_slice()))),
                [alpha, 0.0],
                n,
            )
            .expect("run chained F16 AdamW probe");
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
        "Metal F16 AdamW must track the CPU's stochastic rounding step for step"
    );
}

// The same two contracts at 8 significand bits: BF16 rounds on its own grid
// through its own conversion.
#[cfg(retro_metal)]
#[test]
fn bf16_adamw_metal_kernel_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
    assert_eq!(
        run(false),
        run(true),
        "Metal BF16 AdamW must match CPU after BF16 rounding"
    );
}

#[cfg(retro_metal)]
#[test]
fn bf16_adamw_metal_matches_cpu_across_chained_steps() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
        "Metal BF16 AdamW must track the CPU's stochastic rounding step for step"
    );
}

// The same contract for SGD, whose store is the rounding itself: it keeps no
// moments.
#[cfg(retro_metal)]
#[test]
fn half_precision_sgd_metal_kernel_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
            "Metal {op:?} must match CPU after the rounded store"
        );
    }
}

// The same moving-seed chain for SGD.
#[cfg(retro_metal)]
#[test]
fn half_precision_sgd_metal_matches_cpu_across_chained_steps() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
            "Metal {op:?} must track the CPU's stochastic rounding step for step"
        );
    }
}

/// Runs one `out_prod` shape on both backends and asserts closeness. Shared by
/// the tile-boundary sweeps below; `op` selects the F32 or a quantized-src0
/// kernel, which differ only in how the tile is filled.
#[cfg(retro_metal)]
fn compare_out_prod_shapes(op: ProbeOp, ne_a: [i64; 4], ne_b: [i64; 4], tolerance: f32) {
    let a = pseudo_random(
        ne_a.iter().product::<i64>() as usize,
        0x0a11 ^ op as u64 ^ ne_a[0] as u64,
        1.0,
    );
    let b = pseudo_random(
        ne_b.iter().product::<i64>() as usize,
        0xb0b0 ^ op as u64 ^ ne_b[2] as u64,
        1.0,
    );
    let out_len = (ne_a[0] * ne_b[0] * ne_b[2] * ne_b[3]) as usize;
    let cpu = probe_op(
        op,
        false,
        ProbeInputs::pair(ne_a, &a, ne_b, &b),
        [0.0, 0.0],
        out_len,
    )
    .expect("cpu out_prod");
    let gpu = probe_op(
        op,
        true,
        ProbeInputs::pair(ne_a, &a, ne_b, &b),
        [0.0, 0.0],
        out_len,
    )
    .expect("metal out_prod");
    assert_close(op, 0, &cpu, &gpu, tolerance);
}

#[cfg(retro_metal)]
fn compare_out_prod(op: ProbeOp, ne00: i64, k: i64, ne10: i64, tolerance: f32) {
    compare_out_prod_shapes(op, [ne00, k, 1, 1], [ne10, k, 1, 1], tolerance);
}

/// Shapes that exercise the boundaries of the tiled `out_prod` kernel (ported
/// from the Vulkan shader).
///
/// The kernel computes a 64x16 dst tile per threadgroup with a 16-deep
/// reduction slice. That puts a boundary everywhere: dst rows do not align to
/// the tile, dst columns below 16
/// leave part of the tile inactive, and a reduction length that is not a
/// multiple of the slice depth relies on the padded lanes contributing exactly
/// zero.
///
/// `ne10 = 8` is the case that matters most in practice - it is the nominal
/// `n_ubatch`, so it is the shape the training graph actually dispatches.
///
/// Tolerances are deliberately those of the aligned cases above: the rewrite
/// preserves the summation order exactly, so it has no licence to be less
/// accurate. Widening one here would mean the arithmetic changed.
#[cfg(retro_metal)]
#[test]
fn out_prod_metal_matches_cpu_on_tile_boundaries() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
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
    // Batched, including src0 broadcast over the batch dimension (dps2 > 1).
    compare_out_prod_shapes(ProbeOp::OutProd, [96, 33, 2, 1], [8, 33, 2, 1], 1.0e-4);
    compare_out_prod_shapes(ProbeOp::OutProd, [96, 33, 1, 1], [8, 33, 2, 1], 1.0e-4);
}

/// The same boundary sweep on a quantized `src0`, where the tiling also moved
/// the block/lane selection into the cooperative load.
///
/// Q8_0 and Q4_K cover the two shapes that selection takes: Q8_0 decodes one
/// element per thread, while the K-quants decode a whole 16-value chunk per
/// thread - the branch most likely to break under retiling, since it is the one
/// that assumes a chunk never straddles the tensor end or a QK_K block.
#[cfg(retro_metal)]
#[test]
fn out_prod_quant_metal_matches_cpu_on_tile_boundaries() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    for op in [ProbeOp::OutProdQ80, ProbeOp::OutProdQ4K] {
        // ne00 must stay a multiple of the block size (32 for Q8_0, 256 for K).
        let block = if op == ProbeOp::OutProdQ4K { 256 } else { 32 };
        for (rows, k, ne10) in [(2, 17, 8), (1, 5, 3), (3, 129, 8), (2, 16, 40)] {
            compare_out_prod(op, rows * block, k, ne10, 2.0e-3);
        }
        // Batched with ne02 == ne12: the quantized CPU out_prod asserts that
        // equality outright (unlike the F32 one, which derives a broadcast
        // factor), so a src0-broadcast case would abort in the oracle rather
        // than test the kernel.
        compare_out_prod_shapes(op, [2 * block, 33, 2, 1], [8, 33, 2, 1], 2.0e-3);
    }
}

/// Wall-clock smoke benchmark for the tiled `out_prod`, not a correctness test:
/// `#[ignore]`d so no lane pays for it. Run it with
/// `cargo test --release --test metal_ops -- --ignored --nocapture out_prod_metal_timing`.
///
/// The shape is deliberately reduction-heavy (the axis a per-k kernel
/// synchronizes on, two threadgroup barriers per k) and thin in dst columns (the
/// training regime, where an 8x8 tile leaves most of itself idle).
#[cfg(retro_metal)]
#[test]
#[ignore]
fn out_prod_metal_timing() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    for (ne00, k, ne10) in [(4096_i64, 4096_i64, 8_i64), (1024, 1024, 16)] {
        let a = pseudo_random((ne00 * k) as usize, 0x51, 1.0);
        let b = pseudo_random((ne10 * k) as usize, 0x52, 1.0);
        let ne_a = [ne00, k, 1, 1];
        let ne_b = [ne10, k, 1, 1];
        let out_len = (ne00 * ne10) as usize;
        // One warm-up run so pipeline compilation is not in the measurement.
        probe_op(
            ProbeOp::OutProd,
            true,
            ProbeInputs::pair(ne_a, &a, ne_b, &b),
            [0.0, 0.0],
            out_len,
        )
        .expect("warm-up");

        let runs = 20;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            probe_op(
                ProbeOp::OutProd,
                true,
                ProbeInputs::pair(ne_a, &a, ne_b, &b),
                [0.0, 0.0],
                out_len,
            )
            .expect("timed run");
        }
        let elapsed = start.elapsed();
        eprintln!(
            "out_prod metal [{ne00}x{k}] x [{ne10}x{k}]: {:.3} ms/run over {runs} runs",
            elapsed.as_secs_f64() * 1000.0 / runs as f64
        );
    }
}

/// `REPEAT_BACK` is the gradient of a broadcast `ADD`/`MUL`/`REPEAT` input: it
/// sums the destination-sized tiles of the incoming gradient. Metal had no
/// kernel for it, so any graph emitting one fell back to the CPU silently.
///
/// The cases below cover each axis broadcasting on its own and all four at
/// once, because the kernel walks the outer axes by stride and the inner one by
/// thread - a transposed pair of loop bounds only shows up when the axes differ.
#[cfg(retro_metal)]
#[test]
fn repeat_back_metal_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    // (grad shape, shape it reduces onto)
    let cases = [
        ([8_i64, 1, 1, 1], [4_i64, 1, 1, 1]), // dim 0 only
        ([4, 6, 1, 1], [4, 3, 1, 1]),         // dim 1 only
        ([4, 3, 4, 1], [4, 3, 1, 1]),         // dim 2 only
        ([4, 3, 2, 6], [4, 3, 2, 2]),         // dim 3 only
        ([8, 6, 4, 2], [2, 3, 2, 1]),         // every axis at once
        ([37, 5, 1, 1], [37, 5, 1, 1]),       // no broadcast: a plain copy
        ([1024, 4, 1, 1], [1024, 1, 1, 1]),   // a realistic bias-gradient row
    ];
    for (case, (ne_a, ne_b)) in cases.into_iter().enumerate() {
        let a = pseudo_random(
            ne_a.iter().product::<i64>() as usize,
            0x5e + case as u64,
            1.0,
        );
        let b = vec![0.0_f32; ne_b.iter().product::<i64>() as usize];
        let out_len = b.len();
        let cpu = probe_op(
            ProbeOp::RepeatBack,
            false,
            ProbeInputs::pair(ne_a, &a, ne_b, &b),
            [0.0, 0.0],
            out_len,
        )
        .expect("cpu repeat_back");
        let gpu = probe_op(
            ProbeOp::RepeatBack,
            true,
            ProbeInputs::pair(ne_a, &a, ne_b, &b),
            [0.0, 0.0],
            out_len,
        )
        .expect("metal repeat_back");
        assert!(
            cpu.iter().any(|v| v.abs() > 1.0e-6),
            "case {case}: the CPU reference is all ~0, the comparison proves nothing"
        );
        assert_close(ProbeOp::RepeatBack, case, &cpu, &gpu, 1.0e-6);
    }
}
