//! Fused sparse cross-entropy contracts and parity coverage.
//!
//! The chunked/fused vocab cross-entropy must reproduce, bit-for-bit within F32
//! tolerance, the quantity and the `grad_h` the current full-vocab training tail
//! computes: `logits = mul_mat(W, h)` → weighted `ggml_cross_entropy_loss` →
//! autodiff back to the hidden states. These tests pin that contract *before* the
//! fused operator is wired into the graph, so the memory-saving rewrite has an
//! oracle to match. The full path runs the real fork ggml ops; the fused path is
//! a tiled CPU reference that never materializes the `[n_vocab, n_tokens]` logits.
//!
//! Parity must hold for every tile count, so the peak-logits divisor `C` is a
//! pure memory knob with no effect on the math (the plan's non-negotiable).

mod common;

use retrograd::{
    FusedCeProbeInputs, FusedCeProbeShape, FusedCeWeightType, fused_sparse_ce_probe,
    fused_sparse_ce_probe_offloaded,
};

const N_EMBD: usize = 32;
const N_VOCAB: usize = 40;
const N_TOKENS: usize = 6;

/// Deterministic, non-degenerate hidden states `[n_embd, n_tokens]`.
fn hidden() -> Vec<f32> {
    (0..N_EMBD * N_TOKENS)
        .map(|i| (((i * 31 + 7) % 23) as f32) * 0.11 - 1.2)
        .collect()
}

/// Deterministic projection head `[n_embd, n_vocab]` (the frozen lm_head).
fn head() -> Vec<f32> {
    (0..N_EMBD * N_VOCAB)
        .map(|i| (((i * 17 + 5) % 29) as f32) * 0.07 - 1.0)
        .collect()
}

/// Targets and GRPO coefficients: a masked token (target < 0), a masked token
/// (weight 0), a negative-weight token (KL can drive weights negative), and
/// ordinary positive-weight tokens.
fn labels() -> (Vec<i32>, Vec<f32>) {
    let targets = vec![3, -1, 17, 8, 25, 31];
    let weights = vec![1.0, 0.9, 0.0, -0.6, 1.4, 0.3];
    (targets, weights)
}

fn assert_parity_wt(n_tiles: usize, w_type: FusedCeWeightType) {
    assert_parity_wt_dev(n_tiles, w_type, false);
}

fn assert_parity_wt_dev(n_tiles: usize, w_type: FusedCeWeightType, use_gpu: bool) {
    // The Vulkan backend is not safe to initialize/use from several threads at
    // once, so serialize every probe that may touch a device.
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let grad_loss = 1.5_f32;

    let probe = fused_sparse_ce_probe(
        FusedCeProbeShape {
            n_embd: N_EMBD,
            n_tokens: N_TOKENS,
            n_vocab: N_VOCAB,
            n_tiles,
            seq_chunk: 0,
            w_type,
        },
        use_gpu,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: None,
        },
        grad_loss,
    )
    .expect("fused CE parity probe");

    // Loss: absolute tolerance sized for F32 rounding differences between the
    // full log-sum-exp (single max, one reduction) and the fused online one.
    // CUDA uses cuBLAS SGEMM (TF32/fast-math on supported NVIDIA devices), so
    // its large dot products need the same backend-specific envelope as the
    // OUT_PROD probe. Vulkan and CPU retain the tighter scalar-F32 threshold.
    let loss_tol = if use_gpu && cfg!(retro_cuda) {
        5.0e-3
    } else {
        2.0e-4
    };
    assert!(
        (probe.loss_full - probe.loss_fused).abs() < loss_tol,
        "C={n_tiles} {w_type:?}: loss full {} != fused {}",
        probe.loss_full,
        probe.loss_fused,
    );

    // grad_h: same F32 tolerance, scaled by magnitude for the larger entries.
    for (i, (&full, &fused)) in probe
        .grad_h_full
        .iter()
        .zip(&probe.grad_h_fused)
        .enumerate()
    {
        let tol = if use_gpu && cfg!(retro_cuda) {
            5.0e-4 + 3.0e-3 * full.abs()
        } else {
            2.0e-4 + 1.0e-3 * full.abs()
        };
        assert!(
            (full - fused).abs() < tol,
            "C={n_tiles} {w_type:?}: grad_h[{i}] full {full} != fused {fused}",
        );
    }
}

fn assert_parity(n_tiles: usize) {
    assert_parity_wt(n_tiles, FusedCeWeightType::F32);
}

#[test]
fn fused_matches_full_path_two_tiles() {
    assert_parity(2);
}

#[test]
fn fused_matches_full_path_four_tiles() {
    assert_parity(4);
}

#[test]
fn fused_matches_full_path_eight_tiles() {
    assert_parity(8);
}

/// One tile is the degenerate "no tiling" case and must also match: it is the
/// same math as `C > 1`, just a single pass over the vocabulary.
#[test]
fn fused_matches_full_path_single_tile() {
    assert_parity(1);
}

/// The real training head is quantized (Q8_0). The fused operator dequantizes
/// each projection-head row on the fly; parity against the full path on the
/// dequantized head proves that path matches the current CE tail.
#[test]
fn fused_matches_full_path_quantized_head() {
    for &c in &[1usize, 2, 4, 8] {
        assert_parity_wt(c, FusedCeWeightType::Q8_0);
    }
}

/// Feature 1 - flattened (batch × seq) token chunking.
/// `seq_chunk` caps how many tokens are processed at once, bounding the tiled
/// logits intermediate to `[n_vocab/C, seq_chunk]`. Like `C`, it is a pure memory
/// knob: the loss and `grad_h` must be identical for every chunk size, and every
/// chunk size must still match the full-vocab oracle. Covers chunks below, equal
/// to, and above `N_TOKENS`, crossed with several tile counts. The CPU reference
/// streams one token at a time and is invariant to `seq_chunk` by construction, so
/// this pins the plumbing and the contract; the chunked intermediate itself is
/// exercised by the GPU sweep in `gpu_fused_ce_seq_chunk_matches_cpu`.
#[test]
fn fused_ce_seq_chunk_is_a_pure_memory_knob() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let grad_loss = 1.5_f32;

    for &c in &[1usize, 4] {
        // Reference fused run with no token chunking (seq_chunk = 0 = all tokens).
        let base = fused_sparse_ce_probe(
            FusedCeProbeShape {
                n_embd: N_EMBD,
                n_tokens: N_TOKENS,
                n_vocab: N_VOCAB,
                n_tiles: c,
                seq_chunk: 0,
                w_type: FusedCeWeightType::F32,
            },
            false,
            FusedCeProbeInputs {
                h: &h,
                w: &w,
                targets: &targets,
                weights: &weights,
                bias: None,
            },
            grad_loss,
        )
        .expect("fused CE base probe");

        for &seq_chunk in &[1usize, 2, 3, N_TOKENS, N_TOKENS + 4] {
            let probe = fused_sparse_ce_probe(
                FusedCeProbeShape {
                    n_embd: N_EMBD,
                    n_tokens: N_TOKENS,
                    n_vocab: N_VOCAB,
                    n_tiles: c,
                    seq_chunk,
                    w_type: FusedCeWeightType::F32,
                },
                false,
                FusedCeProbeInputs {
                    h: &h,
                    w: &w,
                    targets: &targets,
                    weights: &weights,
                    bias: None,
                },
                grad_loss,
            )
            .expect("fused CE seq-chunk probe");

            // Each chunk size still matches the full-vocab CPU oracle.
            assert!(
                (probe.loss_full - probe.loss_fused).abs() < 2.0e-4,
                "C={c} seq_chunk={seq_chunk}: loss full {} != fused {}",
                probe.loss_full,
                probe.loss_fused,
            );
            // And the fused result does not move with seq_chunk: a pure knob.
            assert_eq!(
                base.loss_fused, probe.loss_fused,
                "C={c} seq_chunk={seq_chunk}: fused loss drifted with the chunk size",
            );
            for (i, (&b, &p)) in base
                .grad_h_fused
                .iter()
                .zip(&probe.grad_h_fused)
                .enumerate()
            {
                assert_eq!(
                    b, p,
                    "C={c} seq_chunk={seq_chunk}: grad_h[{i}] drifted with the chunk size",
                );
            }
        }
    }
}

/// The GPU token-chunked kernel must match the CPU cross-entropy oracle for every
/// (tile count, chunk size), including chunks smaller than, equal to and larger
/// than the token count. This is where the flattened-sequence chunking actually
/// bounds the materialized logits intermediate. Skipped when no GPU device is
/// registered.
///
/// Crossed with a quantized head, because the CUDA kernel materializes such a head
/// one vocab tile at a time (a whole-head F32 copy would cost `n_embd*n_vocab*4`
/// bytes: 3.75 GiB on a 262k-vocab, 3840-wide head). The target row each token
/// needs for `grad_h` is therefore captured out of whichever tile holds it, once
/// per token chunk - so a chunk sweep on a quantized head is exactly what pins
/// that capture down.
#[test]
fn gpu_fused_ce_seq_chunk_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let grad_loss = 1.5_f32;

    // Q8_0 has a 32-value block and N_EMBD is 32, so a vocab row is exactly one
    // block: every tile boundary is a block boundary, as the kernel requires.
    for w_type in [FusedCeWeightType::F32, FusedCeWeightType::Q8_0] {
        for &c in &[1usize, 4] {
            for &seq_chunk in &[1usize, 3, N_TOKENS, N_TOKENS + 4] {
                let probe = fused_sparse_ce_probe(
                    FusedCeProbeShape {
                        n_embd: N_EMBD,
                        n_tokens: N_TOKENS,
                        n_vocab: N_VOCAB,
                        n_tiles: c,
                        seq_chunk,
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
                    grad_loss,
                )
                .expect("GPU fused CE seq-chunk probe");
                let loss_tol = if cfg!(retro_cuda) { 5.0e-3 } else { 2.0e-4 };
                assert!(
                    (probe.loss_full - probe.loss_fused).abs() < loss_tol,
                    "{w_type:?} C={c} seq_chunk={seq_chunk}: GPU loss full {} != fused {}",
                    probe.loss_full,
                    probe.loss_fused,
                );
                for (i, (&full, &fused)) in probe
                    .grad_h_full
                    .iter()
                    .zip(&probe.grad_h_fused)
                    .enumerate()
                {
                    let tol = if cfg!(retro_cuda) {
                        5.0e-4 + 3.0e-3 * full.abs()
                    } else {
                        2.0e-4 + 1.0e-3 * full.abs()
                    };
                    assert!(
                        (full - fused).abs() < tol,
                        "{w_type:?} C={c} seq_chunk={seq_chunk}: GPU grad_h[{i}] full {full} != fused {fused}",
                    );
                }
            }
        }
    }
}

/// Feature 3 - offloading the log-softmax activations.
/// With the flag set, the graph allocator hands `grad_h` the very buffer holding
/// `h`, so the backward writes over the hidden states it is still reading; the
/// operators evict one token chunk at a time to stay correct. The probe
/// reproduces that aliasing by hand, so this is the test that would catch a
/// backward that clobbers its own input: the loss and `grad_h` must come out
/// bit-identical to the non-aliased run, for every (tile count, chunk size),
/// including chunk sizes that do not divide `N_TOKENS`.
#[test]
fn fused_ce_offload_logsoftmax_is_a_pure_memory_knob() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let bias = bias();
    let grad_loss = 1.5_f32;

    for &c in &[1usize, 4] {
        for &seq_chunk in &[0usize, 1, 4, N_TOKENS] {
            for bias in [None, Some(bias.as_slice())] {
                let base = fused_sparse_ce_probe_offloaded(
                    FusedCeProbeShape {
                        n_embd: N_EMBD,
                        n_tokens: N_TOKENS,
                        n_vocab: N_VOCAB,
                        n_tiles: c,
                        seq_chunk,
                        w_type: FusedCeWeightType::F32,
                    },
                    false,
                    false,
                    FusedCeProbeInputs {
                        h: &h,
                        w: &w,
                        targets: &targets,
                        weights: &weights,
                        bias,
                    },
                    grad_loss,
                )
                .expect("fused CE probe without offload");
                let probe = fused_sparse_ce_probe_offloaded(
                    FusedCeProbeShape {
                        n_embd: N_EMBD,
                        n_tokens: N_TOKENS,
                        n_vocab: N_VOCAB,
                        n_tiles: c,
                        seq_chunk,
                        w_type: FusedCeWeightType::F32,
                    },
                    true,
                    false,
                    FusedCeProbeInputs {
                        h: &h,
                        w: &w,
                        targets: &targets,
                        weights: &weights,
                        bias,
                    },
                    grad_loss,
                )
                .expect("fused CE probe with offload");

                let case = format!("C={c} seq_chunk={seq_chunk} bias={}", bias.is_some());
                // Still the full-vocab oracle, aliasing or not.
                assert!(
                    (probe.loss_full - probe.loss_fused).abs() < 2.0e-4,
                    "{case}: loss full {} != fused {}",
                    probe.loss_full,
                    probe.loss_fused,
                );
                assert_eq!(
                    base.loss_fused, probe.loss_fused,
                    "{case}: fused loss moved when grad_h aliased h",
                );
                for (i, (&b, &p)) in base
                    .grad_h_fused
                    .iter()
                    .zip(&probe.grad_h_fused)
                    .enumerate()
                {
                    assert_eq!(b, p, "{case}: grad_h[{i}] moved when grad_h aliased h");
                }
            }
        }
    }
}

/// Same aliasing contract on the GPU kernel, where the staging buffer that makes
/// it safe actually lives (`grad_stage` in `fused-sparse-ce.cu`). The CPU column
/// copy is invariant to the aliasing by construction; the CUDA backward, which
/// reads whole `[n_embd, chunk]` blocks of `h` across every vocab tile before
/// writing, is not. Skipped when no GPU device is registered.
#[test]
fn gpu_fused_ce_offload_logsoftmax_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let grad_loss = 1.5_f32;

    for &c in &[1usize, 4] {
        for &seq_chunk in &[1usize, 4, N_TOKENS] {
            let probe = fused_sparse_ce_probe_offloaded(
                FusedCeProbeShape {
                    n_embd: N_EMBD,
                    n_tokens: N_TOKENS,
                    n_vocab: N_VOCAB,
                    n_tiles: c,
                    seq_chunk,
                    w_type: FusedCeWeightType::F32,
                },
                true,
                true,
                FusedCeProbeInputs {
                    h: &h,
                    w: &w,
                    targets: &targets,
                    weights: &weights,
                    bias: None,
                },
                grad_loss,
            )
            .expect("GPU fused CE offload probe");
            let loss_tol = if cfg!(retro_cuda) { 5.0e-3 } else { 2.0e-4 };
            assert!(
                (probe.loss_full - probe.loss_fused).abs() < loss_tol,
                "C={c} seq_chunk={seq_chunk}: GPU offload loss full {} != fused {}",
                probe.loss_full,
                probe.loss_fused,
            );
            for (i, (&full, &fused)) in probe
                .grad_h_full
                .iter()
                .zip(&probe.grad_h_fused)
                .enumerate()
            {
                let tol = if cfg!(retro_cuda) {
                    5.0e-4 + 3.0e-3 * full.abs()
                } else {
                    2.0e-4 + 1.0e-3 * full.abs()
                };
                assert!(
                    (full - fused).abs() < tol,
                    "C={c} seq_chunk={seq_chunk}: GPU offload grad_h[{i}] full {full} != fused {fused}",
                );
            }
        }
    }
}

/// The fused operator on the GPU backend must match the CPU cross-entropy oracle
/// (the probe's full path always runs on CPU). Covers the frozen head in both
/// F32 and the real training layout (Q8_0, dequantized on the fly in the shader),
/// for every tile count - parity must be independent of `C`. Skipped when no GPU
/// device is registered.
#[test]
fn vulkan_fused_ce_matches_cpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    for &c in &[1usize, 2, 4, 8] {
        assert_parity_wt_dev(c, FusedCeWeightType::F32, true);
        assert_parity_wt_dev(c, FusedCeWeightType::Q8_0, true);
    }
}

/// The real training head for a tied-embedding k-quant model (e.g. Qwen3
/// Q4_K_M) is Q6_K. The fused operator dequantizes each Q6_K block on the fly;
/// parity against the CPU cross-entropy oracle proves that path, on CPU always
/// and on the GPU backend when present. Q6_K blocks are 256 wide, so n_embd is a
/// multiple of 256 here.
#[test]
fn fused_matches_full_path_q6_k_head() {
    let _guard = common::serialize_models();
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    let h: Vec<f32> = (0..NE * NT)
        .map(|i| (((i * 31 + 7) % 23) as f32) * 0.11 - 1.2)
        .collect();
    let w: Vec<f32> = (0..NE * NV)
        .map(|i| (((i * 17 + 5) % 29) as f32) * 0.07 - 1.0)
        .collect();
    let targets = vec![3, -1, 17, 8, 299];
    let weights = vec![1.0_f32, 0.9, 0.0, -0.6, 1.4];

    let gpu = common::gpu_device_present();
    for &c in &[1usize, 2, 8] {
        for &use_gpu in if gpu {
            &[false, true][..]
        } else {
            &[false][..]
        } {
            let probe = fused_sparse_ce_probe(
                FusedCeProbeShape {
                    n_embd: NE,
                    n_tokens: NT,
                    n_vocab: NV,
                    n_tiles: c,
                    seq_chunk: 0,
                    w_type: FusedCeWeightType::Q6K,
                },
                use_gpu,
                FusedCeProbeInputs {
                    h: &h,
                    w: &w,
                    targets: &targets,
                    weights: &weights,
                    bias: None,
                },
                1.5,
            )
            .expect("fused CE Q6_K parity probe");
            let loss_tol = if use_gpu && cfg!(retro_cuda) {
                5.0e-3
            } else {
                2.0e-4
            };
            assert!(
                (probe.loss_full - probe.loss_fused).abs() < loss_tol,
                "C={c} gpu={use_gpu}: Q6_K loss full {} != fused {}",
                probe.loss_full,
                probe.loss_fused,
            );
            for (i, (&full, &fused)) in probe
                .grad_h_full
                .iter()
                .zip(&probe.grad_h_fused)
                .enumerate()
            {
                let tol = if use_gpu && cfg!(retro_cuda) {
                    5.0e-4 + 3.0e-3 * full.abs()
                } else {
                    2.0e-4 + 1.0e-3 * full.abs()
                };
                assert!(
                    (full - fused).abs() < tol,
                    "C={c} gpu={use_gpu}: Q6_K grad_h[{i}] full {full} != fused {fused}",
                );
            }
        }
    }
}

/// The generic in-shader dequant covers every ggml quant with one shader source
/// (compiled per DATA_A_* type). This exercises the two dequant interleaves - a
/// legacy quant (Q4_0, block 32, `QUANT_R == 2`) and more k-quants (Q4_K, Q5_K,
/// block 256) - against the CPU cross-entropy oracle, on CPU always and on the
/// GPU backend when present. n_embd is a multiple of 256 to satisfy every block
/// size at once.
#[test]
fn fused_matches_full_path_quant_family() {
    let _guard = common::serialize_models();
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    let h: Vec<f32> = (0..NE * NT)
        .map(|i| (((i * 31 + 7) % 23) as f32) * 0.11 - 1.2)
        .collect();
    let w: Vec<f32> = (0..NE * NV)
        .map(|i| (((i * 17 + 5) % 29) as f32) * 0.07 - 1.0)
        .collect();
    let targets = vec![3, -1, 17, 8, 299];
    let weights = vec![1.0_f32, 0.9, 0.0, -0.6, 1.4];

    let gpu = common::gpu_device_present();
    for wt in [
        FusedCeWeightType::Q4_0,
        FusedCeWeightType::Q4K,
        FusedCeWeightType::Q5K,
    ] {
        for &c in &[1usize, 2, 8] {
            for &use_gpu in if gpu {
                &[false, true][..]
            } else {
                &[false][..]
            } {
                let probe = fused_sparse_ce_probe(
                    FusedCeProbeShape {
                        n_embd: NE,
                        n_tokens: NT,
                        n_vocab: NV,
                        n_tiles: c,
                        seq_chunk: 0,
                        w_type: wt,
                    },
                    use_gpu,
                    FusedCeProbeInputs {
                        h: &h,
                        w: &w,
                        targets: &targets,
                        weights: &weights,
                        bias: None,
                    },
                    1.5,
                )
                .expect("fused CE quant-family parity probe");
                let loss_tol = if use_gpu && cfg!(retro_cuda) {
                    5.0e-3
                } else {
                    2.0e-4
                };
                assert!(
                    (probe.loss_full - probe.loss_fused).abs() < loss_tol,
                    "{wt:?} C={c} gpu={use_gpu}: loss full {} != fused {}",
                    probe.loss_full,
                    probe.loss_fused,
                );
                for (i, (&full, &fused)) in probe
                    .grad_h_full
                    .iter()
                    .zip(&probe.grad_h_fused)
                    .enumerate()
                {
                    let tol = if use_gpu && cfg!(retro_cuda) {
                        5.0e-4 + 3.0e-3 * full.abs()
                    } else {
                        2.0e-4 + 1.0e-3 * full.abs()
                    };
                    assert!(
                        (full - fused).abs() < tol,
                        "{wt:?} C={c} gpu={use_gpu}: grad_h[{i}] full {full} != fused {fused}",
                    );
                }
            }
        }
    }
}

/// The fused CE must match the oracle for *every* head type the fork claims it can
/// decode, not the three the test above hand-picks.
///
/// Same rationale as `common::assert_out_prod_all_dequant_types_match_cpu`: the type
/// list comes from `retrograd::dequant_types()`, so it is the fork's table rather
/// than a copy of it, and the head types of `FUSED_SPARSE_CE` and the frozen-weight
/// types of `OUT_PROD` are guaranteed to be the same set.
///
/// Tolerances match the hand-picked test: the full path is built by dequantizing the
/// very bytes the fused path consumes, so a coarse head type is compared against its
/// own dequantization and has no licence to be less accurate.
#[test]
fn fused_matches_full_path_every_dequant_type() {
    let _guard = common::serialize_models();
    // A multiple of 256 satisfies every block size in the table at once.
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    let h: Vec<f32> = (0..NE * NT)
        .map(|i| (((i * 31 + 7) % 23) as f32) * 0.11 - 1.2)
        .collect();
    let w: Vec<f32> = (0..NE * NV)
        .map(|i| (((i * 17 + 5) % 29) as f32) * 0.07 - 1.0)
        .collect();
    let targets = vec![3, -1, 17, 8, 299];
    let weights = vec![1.0_f32, 0.9, 0.0, -0.6, 1.4];

    let types = retrograd::dequant_types();
    assert!(
        types.len() >= 19,
        "dequant_types() returned {} entries, expected the full table",
        types.len()
    );

    let gpu = common::gpu_device_present();
    for (type_id, type_name) in &types {
        let wt = FusedCeWeightType::Ggml(*type_id);
        for &use_gpu in if gpu {
            &[false, true][..]
        } else {
            &[false][..]
        } {
            let probe = fused_sparse_ce_probe(
                FusedCeProbeShape {
                    n_embd: NE,
                    n_tokens: NT,
                    n_vocab: NV,
                    n_tiles: 2,
                    seq_chunk: 0,
                    w_type: wt,
                },
                use_gpu,
                FusedCeProbeInputs {
                    h: &h,
                    w: &w,
                    targets: &targets,
                    weights: &weights,
                    bias: None,
                },
                1.5,
            )
            .unwrap_or_else(|e| panic!("fused CE probe for {type_name} gpu={use_gpu}: {e}"));
            let loss_tol = if use_gpu && cfg!(retro_cuda) {
                5.0e-3
            } else {
                2.0e-4
            };
            assert!(
                (probe.loss_full - probe.loss_fused).abs() < loss_tol,
                "{type_name} gpu={use_gpu}: loss full {} != fused {}",
                probe.loss_full,
                probe.loss_fused,
            );
            for (i, (&full, &fused)) in probe
                .grad_h_full
                .iter()
                .zip(&probe.grad_h_fused)
                .enumerate()
            {
                let tol = if use_gpu && cfg!(retro_cuda) {
                    5.0e-4 + 3.0e-3 * full.abs()
                } else {
                    2.0e-4 + 1.0e-3 * full.abs()
                };
                assert!(
                    (full - fused).abs() < tol,
                    "{type_name} gpu={use_gpu}: grad_h[{i}] full {full} != fused {fused}",
                );
            }
        }
    }
}

/// Guards vigilance point #1: the on-device log-sum-exp must accumulate in F32.
/// Hidden states and head are scaled so the logits span a wide dynamic range;
/// an accidental F16 reduction of the 151k-term sum would visibly diverge from
/// the CPU F32 oracle. Skipped when no GPU device is registered.
#[test]
fn vulkan_fused_ce_reduction_stays_f32() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    let _guard = common::serialize_models();
    const NE: usize = 64;
    const NT: usize = 4;
    const NV: usize = 4096;

    // Large-magnitude, heterogeneous hidden states and head → logits with a wide
    // spread, the regime where a low-precision reduction biases the softmax.
    let h: Vec<f32> = (0..NE * NT)
        .map(|i| (((i * 13 + 3) % 41) as f32) * 0.35 - 7.0)
        .collect();
    let w: Vec<f32> = (0..NE * NV)
        .map(|i| (((i * 19 + 7) % 53) as f32) * 0.21 - 5.0)
        .collect();
    let targets = vec![10, 2000, -1, 4095];
    let weights = vec![1.2_f32, -0.7, 0.5, 0.9];

    for &c in &[2usize, 8] {
        let probe = fused_sparse_ce_probe(
            FusedCeProbeShape {
                n_embd: NE,
                n_tokens: NT,
                n_vocab: NV,
                n_tiles: c,
                seq_chunk: 0,
                w_type: FusedCeWeightType::F32,
            },
            true,
            FusedCeProbeInputs {
                h: &h,
                w: &w,
                targets: &targets,
                weights: &weights,
                bias: None,
            },
            1.0,
        )
        .expect("fused CE parity probe (large vocab)");
        let loss_tol = if cfg!(retro_cuda) { 1.0e-2 } else { 5.0e-4 };
        assert!(
            (probe.loss_full - probe.loss_fused).abs() < loss_tol,
            "C={c}: large-vocab loss cpu {} != gpu {}",
            probe.loss_full,
            probe.loss_fused,
        );
        for (i, (&full, &fused)) in probe
            .grad_h_full
            .iter()
            .zip(&probe.grad_h_fused)
            .enumerate()
        {
            let tol = 1.0e-3 + 3.0e-3 * full.abs();
            assert!(
                (full - fused).abs() < tol,
                "C={c}: large-vocab grad_h[{i}] cpu {full} != gpu {fused}",
            );
        }
    }
}

/// Masked tokens (target < 0 or weight 0) must contribute exactly zero gradient
/// on both paths - the training mask must survive the fusion untouched.
#[test]
fn masked_tokens_have_zero_gradient() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let probe = fused_sparse_ce_probe(
        FusedCeProbeShape {
            n_embd: N_EMBD,
            n_tokens: N_TOKENS,
            n_vocab: N_VOCAB,
            n_tiles: 4,
            seq_chunk: 0,
            w_type: FusedCeWeightType::F32,
        },
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: None,
        },
        1.5,
    )
    .expect("fused CE parity probe");

    for t in 0..N_TOKENS {
        let masked = targets[t] < 0 || weights[t] == 0.0;
        if !masked {
            continue;
        }
        let col = &probe.grad_h_fused[t * N_EMBD..(t + 1) * N_EMBD];
        assert!(
            col.iter().all(|&v| v == 0.0),
            "masked token {t} has non-zero fused gradient: {col:?}",
        );
        let col_full = &probe.grad_h_full[t * N_EMBD..(t + 1) * N_EMBD];
        for (i, &v) in col_full.iter().enumerate() {
            assert!(v.abs() < 1.0e-6, "masked token {t} full grad[{i}] = {v}");
        }
    }
}

/// A fixed, deterministic per-vocab bias - the `ggml_fused_sparse_ce` contract
/// gemma4's `ADD(MUL_MAT(w, h), bias)` output head relies on. `full` path adds the bias with a
/// plain `ggml_add` before `mul_mat -> cross_entropy_loss`; the fused path
/// passes it as the operator's native bias input. Parity of both loss and
/// `grad_h` proves the bias participates in the softmax identically to the
/// oracle while never receiving a gradient itself.
fn bias() -> Vec<f32> {
    (0..N_VOCAB)
        .map(|v| (((v * 13 + 3) % 17) as f32) * 0.23 - 1.5)
        .collect()
}

fn assert_bias_parity_wt(n_tiles: usize, w_type: FusedCeWeightType) {
    assert_bias_parity_wt_dev(n_tiles, w_type, false);
}

fn assert_bias_parity_wt_dev(n_tiles: usize, w_type: FusedCeWeightType, use_gpu: bool) {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let b = bias();
    let grad_loss = 1.5_f32;

    let probe = fused_sparse_ce_probe(
        FusedCeProbeShape {
            n_embd: N_EMBD,
            n_tokens: N_TOKENS,
            n_vocab: N_VOCAB,
            n_tiles,
            seq_chunk: 0,
            w_type,
        },
        use_gpu,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: Some(&b),
        },
        grad_loss,
    )
    .expect("fused CE bias parity probe");

    let loss_tol = if use_gpu && cfg!(retro_cuda) {
        5.0e-3
    } else {
        2.0e-4
    };
    assert!(
        (probe.loss_full - probe.loss_fused).abs() < loss_tol,
        "C={n_tiles} {w_type:?} bias gpu={use_gpu}: loss full {} != fused {}",
        probe.loss_full,
        probe.loss_fused,
    );
    for (i, (&full, &fused)) in probe
        .grad_h_full
        .iter()
        .zip(&probe.grad_h_fused)
        .enumerate()
    {
        let tol = if use_gpu && cfg!(retro_cuda) {
            5.0e-4 + 3.0e-3 * full.abs()
        } else {
            2.0e-4 + 1.0e-3 * full.abs()
        };
        assert!(
            (full - fused).abs() < tol,
            "C={n_tiles} {w_type:?} bias gpu={use_gpu}: grad_h[{i}] full {full} != fused {fused}",
        );
    }
}

#[test]
fn fused_matches_full_path_with_bias_f32_head() {
    for &c in &[1usize, 2, 4, 8] {
        assert_bias_parity_wt(c, FusedCeWeightType::F32);
    }
}

#[test]
fn fused_matches_full_path_with_bias_quantized_head() {
    for &c in &[1usize, 2, 4, 8] {
        assert_bias_parity_wt(c, FusedCeWeightType::Q8_0);
    }
}

/// Same bias contract as the CPU-only tests above, but on whichever GPU device is
/// registered - it is the parity coverage for the biased fused-CE head on all
/// three backends. Skipped when no GPU is registered.
#[test]
fn fused_matches_full_path_with_bias_gpu() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    for &c in &[1usize, 2, 4, 8] {
        assert_bias_parity_wt_dev(c, FusedCeWeightType::F32, true);
        assert_bias_parity_wt_dev(c, FusedCeWeightType::Q8_0, true);
    }
}

/// The bias never receives a gradient: masked-token columns of `grad_h` stay
/// exactly zero even with a bias present, same contract as the no-bias case.
#[test]
fn bias_present_masked_tokens_still_zero_gradient() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let b = bias();
    let probe = fused_sparse_ce_probe(
        FusedCeProbeShape {
            n_embd: N_EMBD,
            n_tokens: N_TOKENS,
            n_vocab: N_VOCAB,
            n_tiles: 4,
            seq_chunk: 0,
            w_type: FusedCeWeightType::F32,
        },
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: Some(&b),
        },
        1.5,
    )
    .expect("fused CE bias parity probe");

    for t in 0..N_TOKENS {
        let masked = targets[t] < 0 || weights[t] == 0.0;
        if !masked {
            continue;
        }
        let col = &probe.grad_h_fused[t * N_EMBD..(t + 1) * N_EMBD];
        assert!(
            col.iter().all(|&v| v == 0.0),
            "masked token {t} has non-zero fused gradient with bias: {col:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// `k` sparse targets per position.
//
// The operator's contract past this line is that `targets` and `weights` are
// `[K, n_tokens]` and the loss a position carries is
// `sum_j weights[j,t] * (logsumexp - z[targets[j,t]])`. The oracle is the same
// one the K = 1 tests use - the dense `ggml_cross_entropy_loss` over a full
// `[n_vocab, n_tokens]` label row - which is exactly what makes it an oracle
// here: writing `k` entries into that row is the *definition* of the sparse
// distribution the fused path has to reproduce, not a re-implementation of it.

/// A teacher-shaped top-`k` block per position: `k` distinct vocabulary ids and
/// a renormalized probability each. Two positions are masked, one by a negative
/// id in every entry and one by zero weights, so the inactive-position path is
/// exercised at `k > 1` as well.
fn topk_labels(k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut targets = Vec::with_capacity(N_TOKENS * k);
    let mut weights = Vec::with_capacity(N_TOKENS * k);
    for token in 0..N_TOKENS {
        let masked = token == 1 || token == 4;
        let mut mass = 0.0_f32;
        let mut row = Vec::with_capacity(k);
        for entry in 0..k {
            // Distinct ids inside a position: two entries on the same row would
            // be a producer bug, and the dense oracle would silently keep only
            // the last of them.
            let id = ((token * 7 + entry * 11 + 3) % N_VOCAB) as i32;
            let probability = 1.0 / ((entry + 1) as f32).powf(1.3);
            row.push((id, probability));
            mass += probability;
        }
        row.sort_by_key(|&(id, _)| id);
        row.dedup_by_key(|&mut (id, _)| id);
        while row.len() < k {
            let next = row.iter().map(|&(id, _)| id).max().unwrap_or(0) + 1;
            row.push((next % N_VOCAB as i32, 0.0));
        }
        for (index, (id, probability)) in row.into_iter().enumerate() {
            let _ = index;
            targets.push(if masked && token == 1 { -1 } else { id });
            weights.push(if masked { 0.0 } else { probability / mass });
        }
    }
    (targets, weights)
}

fn assert_topk_parity(k: usize, n_tiles: usize, w_type: FusedCeWeightType, use_gpu: bool) {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = topk_labels(k);
    let probe = fused_sparse_ce_probe(
        FusedCeProbeShape {
            n_embd: N_EMBD,
            n_tokens: N_TOKENS,
            n_vocab: N_VOCAB,
            n_tiles,
            seq_chunk: 0,
            w_type,
        },
        use_gpu,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: None,
        },
        1.5,
    )
    .expect("fused CE top-k parity probe");

    let loss_tol = if use_gpu && cfg!(retro_cuda) {
        5.0e-3
    } else {
        2.0e-4
    };
    assert!(
        (probe.loss_full - probe.loss_fused).abs() < loss_tol,
        "k={k} C={n_tiles} {w_type:?}: loss full {} != fused {}",
        probe.loss_full,
        probe.loss_fused,
    );
    for (i, (&full, &fused)) in probe
        .grad_h_full
        .iter()
        .zip(&probe.grad_h_fused)
        .enumerate()
    {
        let tol = if use_gpu && cfg!(retro_cuda) {
            5.0e-4 + 3.0e-3 * full.abs()
        } else {
            2.0e-4 + 1.0e-3 * full.abs()
        };
        assert!(
            (full - fused).abs() < tol,
            "k={k} C={n_tiles} {w_type:?}: grad_h[{i}] full {full} != fused {fused}",
        );
    }
    // A position whose every entry is masked owns its slice of grad_h and it
    // must be zero - the same contract as at K = 1, and the one that keeps a
    // prompt token out of the update.
    for token in [1_usize, 4] {
        let column = &probe.grad_h_fused[token * N_EMBD..(token + 1) * N_EMBD];
        assert!(
            column.iter().all(|&value| value == 0.0),
            "k={k}: masked position {token} has a non-zero fused gradient: {column:?}",
        );
    }
}

#[test]
fn topk_matches_the_dense_distribution_for_every_tile_count() {
    for n_tiles in [1, 2, 4, 8] {
        assert_topk_parity(4, n_tiles, FusedCeWeightType::F32, false);
    }
}

/// The same parity on the registered GPU backend. The
/// oracle is the CPU dense path in both cases, so this is the device kernel
/// against the reference and not one device against another.
#[test]
fn gpu_topk_matches_the_dense_distribution() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered");
        return;
    }
    for k in [1, 2, 8] {
        for n_tiles in [1, 4] {
            assert_topk_parity(k, n_tiles, FusedCeWeightType::F32, true);
        }
    }
    assert_topk_parity(8, 4, FusedCeWeightType::F16, true);
}

#[test]
fn topk_matches_the_dense_distribution_for_every_k() {
    for k in [1, 2, 3, 8, 16] {
        assert_topk_parity(k, 4, FusedCeWeightType::F32, false);
    }
}

/// A quantized head at `k > 1`: the entries are read through the same on-the-fly
/// dequantization the scalar path uses, once per target row instead of once.
/// `n_embd = 256` so every block size divides it, as in the quant-family test.
#[test]
fn topk_matches_the_dense_distribution_on_a_quantized_head() {
    let _guard = common::serialize_models();
    const NE: usize = 256;
    const NT: usize = 5;
    const NV: usize = 300;
    const K: usize = 8;
    let h: Vec<f32> = (0..NE * NT)
        .map(|i| (((i * 31 + 7) % 23) as f32) * 0.11 - 1.2)
        .collect();
    let w: Vec<f32> = (0..NE * NV)
        .map(|i| (((i * 17 + 5) % 29) as f32) * 0.07 - 1.0)
        .collect();
    let mut targets = Vec::with_capacity(NT * K);
    let mut weights = Vec::with_capacity(NT * K);
    for token in 0..NT {
        let masked = token == 2;
        let mass: f32 = (0..K).map(|entry| 1.0 / ((entry + 1) as f32)).sum();
        for entry in 0..K {
            targets.push(((token * 13 + entry * 29 + 5) % NV) as i32);
            weights.push(if masked {
                0.0
            } else {
                (1.0 / ((entry + 1) as f32)) / mass
            });
        }
    }
    for w_type in [
        FusedCeWeightType::F16,
        FusedCeWeightType::Q8_0,
        FusedCeWeightType::Q4K,
    ] {
        let probe = fused_sparse_ce_probe(
            FusedCeProbeShape {
                n_embd: NE,
                n_tokens: NT,
                n_vocab: NV,
                n_tiles: 4,
                seq_chunk: 0,
                w_type,
            },
            false,
            FusedCeProbeInputs {
                h: &h,
                w: &w,
                targets: &targets,
                weights: &weights,
                bias: None,
            },
            1.5,
        )
        .expect("fused CE quantized top-k parity probe");
        assert!(
            (probe.loss_full - probe.loss_fused).abs() < 2.0e-4,
            "{w_type:?}: loss full {} != fused {}",
            probe.loss_full,
            probe.loss_fused,
        );
        for (index, (&full, &fused)) in probe
            .grad_h_full
            .iter()
            .zip(&probe.grad_h_fused)
            .enumerate()
        {
            assert!(
                (full - fused).abs() < 2.0e-4 + 1.0e-3 * full.abs(),
                "{w_type:?}: grad_h[{index}] full {full} != fused {fused}",
            );
        }
    }
}

/// The invariant that makes the generalization safe for the rest of the
/// repository: at `k = 1` the operator is the one that existed before, bit for
/// bit. Not "within tolerance" - the same bits, so nothing already measured on
/// this operator moves.
#[test]
fn k_equal_to_one_is_bit_identical_to_the_scalar_operator() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let shape = FusedCeProbeShape {
        n_embd: N_EMBD,
        n_tokens: N_TOKENS,
        n_vocab: N_VOCAB,
        n_tiles: 4,
        seq_chunk: 0,
        w_type: FusedCeWeightType::F32,
    };
    // `[n_tokens]` and `[1, n_tokens]` are the same bytes and the same shape:
    // the probe reads `k` off the length, so this is the scalar call. The
    // comparison that matters is against the dense oracle computed from the
    // same inputs, which is what the loop below asserts to the last bit.
    let probe = fused_sparse_ce_probe(
        shape,
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: None,
        },
        1.5,
    )
    .expect("scalar probe");
    // Now the same objective spelled as a k = 2 block whose second entry is
    // absent. The operator must skip it exactly, not average it in.
    let mut padded_targets = Vec::with_capacity(N_TOKENS * 2);
    let mut padded_weights = Vec::with_capacity(N_TOKENS * 2);
    for token in 0..N_TOKENS {
        padded_targets.push(targets[token]);
        padded_weights.push(weights[token]);
        padded_targets.push(-1);
        padded_weights.push(0.0);
    }
    let padded = fused_sparse_ce_probe(
        shape,
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &padded_targets,
            weights: &padded_weights,
            bias: None,
        },
        1.5,
    )
    .expect("padded probe");
    assert_eq!(
        probe.loss_fused.to_bits(),
        padded.loss_fused.to_bits(),
        "an absent entry changed the loss: {} vs {}",
        probe.loss_fused,
        padded.loss_fused
    );
    for (index, (&scalar, &padded)) in probe
        .grad_h_fused
        .iter()
        .zip(&padded.grad_h_fused)
        .enumerate()
    {
        assert_eq!(
            scalar.to_bits(),
            padded.to_bits(),
            "an absent entry changed grad_h[{index}]: {scalar} vs {padded}"
        );
    }
}

/// The normalization pitfall, checked on the operator itself: the mean is over
/// active *positions*, so widening `k` on a position that already had one
/// target must not divide the loss by `k`. A block whose whole mass sits on one
/// entry is the same objective as that entry alone.
#[test]
fn widening_k_normalizes_over_positions_and_not_over_entries() {
    let _guard = common::serialize_models();
    let h = hidden();
    let w = head();
    let (targets, weights) = labels();
    let shape = FusedCeProbeShape {
        n_embd: N_EMBD,
        n_tokens: N_TOKENS,
        n_vocab: N_VOCAB,
        n_tiles: 2,
        seq_chunk: 0,
        w_type: FusedCeWeightType::F32,
    };
    let scalar = fused_sparse_ce_probe(
        shape,
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &targets,
            weights: &weights,
            bias: None,
        },
        1.0,
    )
    .expect("scalar probe");
    // The same weight split over four entries that all name the same token: the
    // dense label row is unchanged, so the loss must be too.
    let k = 4;
    let mut wide_targets = Vec::with_capacity(N_TOKENS * k);
    let mut wide_weights = Vec::with_capacity(N_TOKENS * k);
    for token in 0..N_TOKENS {
        for _ in 0..k {
            wide_targets.push(targets[token]);
            wide_weights.push(weights[token] / k as f32);
        }
    }
    let wide = fused_sparse_ce_probe(
        shape,
        false,
        FusedCeProbeInputs {
            h: &h,
            w: &w,
            targets: &wide_targets,
            weights: &wide_weights,
            bias: None,
        },
        1.0,
    )
    .expect("wide probe");
    assert!(
        (scalar.loss_fused - wide.loss_fused).abs() < 2.0e-5,
        "spreading one target's weight over {k} entries changed the loss: {} vs {}",
        scalar.loss_fused,
        wide.loss_fused
    );
    for (index, (&one, &many)) in scalar
        .grad_h_fused
        .iter()
        .zip(&wide.grad_h_fused)
        .enumerate()
    {
        assert!(
            (one - many).abs() < 2.0e-5 + 1.0e-4 * one.abs(),
            "spreading the weight changed grad_h[{index}]: {one} vs {many}"
        );
    }
}
