//! The chunkwise gated-delta-net backward against the sequential one, on CPU.
//!
//! `GGML_OP_GATED_DELTA_NET_BACK` has two implementations of the same
//! gradients: the per-token reverse scan
//! (`ggml_compute_forward_gated_delta_net_back_f32`) and a chunkwise form that
//! replaces the scan inside a chunk of C tokens by six matrix products and a
//! unit-triangular solve (`..._back_chunked_f32`). The second is what makes the
//! CUDA kernel possible -- the token scan there is 88% of all GPU time on a
//! hybrid model -- and this binary is the
//! milestone that de-risks it: the adjoints are checked in plain CPU
//! arithmetic, on shapes the CUDA grid cannot even express, before any kernel
//! exists. If these disagree, the derivation is wrong and no kernel can save
//! it.
//!
//! Both paths run in one process through the `probe_op` harness, selected by
//! its `param0` (see `ggml_gated_delta_net_back_chunked`): negative pins the
//! sequential reference, positive pins the chunkwise form with that chunk
//! length. The sequential path is never the thing under test here -- it is the
//! oracle, unchanged and still the default on CPU.

use retrograd::{ProbeInputs, ProbeOp, probe_op};

const S_V: i64 = 32;

/// Deterministic non-degenerate values in `[-range, range]` (xorshift, so the
/// stream is stable across platforms and reruns).
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

/// What the log-gate looks like. The chunkwise form normalises by the
/// cumulative decay (`khat = k/A`), so the gate is the one input that decides
/// whether it takes its fast path, halves the chunk, or falls back to a single
/// sequential token -- which is why every case below is a gate case.
#[derive(Clone, Copy, Debug)]
enum Gate {
    /// Ordinary decay, a few tenths per token: the fast path.
    Mild,
    /// No decay at all, `A == 1` exactly: the identity the normalisation must
    /// not disturb.
    Zero,
    /// Strong decay: `1/A` outgrows the F32 bound within the requested chunk,
    /// so the guard must halve.
    Strong,
    /// One token whose gate alone defeats any normalisation: the guard must
    /// reach a single token and hand it to the sequential step.
    Extreme,
    /// Per-channel mix of growth and decay, only meaningful for a KDA gate:
    /// `A` is a vector and the bound has to hold channel by channel.
    MixedSign,
}

/// What beta looks like. `beta = 0` freezes the state (the `(I+T)` solve
/// becomes the identity), `beta = 1` replaces it outright.
#[derive(Clone, Copy, Debug)]
enum Beta {
    Random,
    Zero,
    One,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kda: bool,
    k: i64,
    n_tokens: i64,
    h: i64,
    n_seqs: i64,
    chunk: i32,
    gate: Gate,
    beta: Beta,
}

impl Case {
    fn new(n_tokens: i64, chunk: i32) -> Self {
        Self {
            kda: false,
            k: 1,
            n_tokens,
            h: 4,
            n_seqs: 2,
            chunk,
            gate: Gate::Mild,
            beta: Beta::Random,
        }
    }
}

fn gate_values(case: &Case, n: usize, width: i64) -> Vec<f32> {
    let base = pseudo_random(n, 0x6604, 1.0);
    base.iter()
        .enumerate()
        .map(|(index, &value)| match case.gate {
            Gate::Mild => -value.abs() * 0.3 - 0.02,
            Gate::Zero => 0.0,
            Gate::Strong => -value.abs() - 2.5,
            // One token in three carries a gate no chunk can normalise; the
            // rest stay mild so the same case also covers a chunk boundary
            // forced next to ordinary tokens.
            Gate::Extreme => {
                if (index as i64 / width) % 3 == 1 {
                    -120.0
                } else {
                    -value.abs() * 0.3 - 0.02
                }
            }
            Gate::MixedSign => {
                if index % 2 == 0 {
                    value.abs() * 0.2
                } else {
                    -value.abs() * 0.4 - 0.05
                }
            }
        })
        .collect()
}

/// Runs one case through both formulations and returns (sequential, chunkwise)
/// together with the element counts of each gradient block.
fn run_case(case: &Case) -> (Vec<f32>, Vec<f32>, Vec<(&'static str, usize)>) {
    let Case {
        kda,
        k,
        n_tokens,
        h,
        n_seqs,
        chunk,
        beta,
        ..
    } = *case;

    let ne_src0 = [S_V, h, n_tokens, n_seqs];
    let ne_src1 = [k, if kda { 1 } else { 0 }, 1, 1];

    let n_qkv = S_V * h * n_tokens * n_seqs;
    let g_width = if kda { S_V } else { 1 };
    let n_g = g_width * h * n_tokens * n_seqs;
    let n_beta = h * n_tokens * n_seqs;
    let n_state = S_V * S_V * h * n_seqs;
    let n_grad = n_qkv + k * n_state;

    let mut packed = Vec::new();
    packed.extend(pseudo_random(n_qkv as usize, 0x6601, 0.5)); // q
    packed.extend(pseudo_random(n_qkv as usize, 0x6602, 0.5)); // k
    packed.extend(pseudo_random(n_qkv as usize, 0x6603, 0.5)); // v
    packed.extend(gate_values(case, n_g as usize, g_width));
    packed.extend(
        pseudo_random(n_beta as usize, 0x6605, 0.5)
            .iter()
            .map(|value| match beta {
                Beta::Random => value.abs(),
                Beta::Zero => 0.0,
                Beta::One => 1.0,
            }),
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
    let out_len = blocks.iter().map(|(_, n)| n).sum();
    let dummy_src1 = vec![0.0_f32; 1];

    let sequential = probe_op(
        ProbeOp::GatedDeltaNetBack,
        false,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [-1.0, 0.0],
        out_len,
    )
    .expect("sequential gated_delta_net_back probe");
    let chunked = probe_op(
        ProbeOp::GatedDeltaNetBack,
        false,
        ProbeInputs::pair(ne_src0, &packed, ne_src1, &dummy_src1),
        [chunk as f32, 0.0],
        out_len,
    )
    .expect("chunked gated_delta_net_back probe");

    (sequential, chunked, blocks)
}

/// Compares block by block, scaled by the reference's own magnitude: the two
/// paths group the same sums differently, so the contract is relative
/// agreement, not bit equality. Reporting per block is what turns a failure
/// into a diagnosis -- a wrong adjoint shows up in one gradient, a wrong chunk
/// layout in `grad_state`.
fn assert_agrees(case: &Case, tolerance: f32) {
    let (sequential, chunked, blocks) = run_case(case);
    assert_eq!(sequential.len(), chunked.len());

    let mut offset = 0;
    let mut worst = 0.0_f32;
    let mut worst_block = "";
    for (name, count) in blocks {
        let mut scale = 0.0_f32;
        let mut max_diff = 0.0_f32;
        let mut where_at = 0;
        for index in 0..count {
            let expected = sequential[offset + index];
            let actual = chunked[offset + index];
            assert!(
                actual.is_finite(),
                "{case:?}: {name}[{index}] is not finite: {actual}"
            );
            scale = scale.max(expected.abs());
            let diff = (expected - actual).abs();
            if diff > max_diff {
                max_diff = diff;
                where_at = index;
            }
        }
        let relative = max_diff / scale.max(1.0e-3);
        assert!(
            relative <= tolerance,
            "{case:?}: {name} differs at {where_at}: sequential={}, chunked={}, \
             |diff|={max_diff:e}, scale={scale:e}, relative={relative:e} > {tolerance:e}",
            sequential[offset + where_at],
            chunked[offset + where_at],
        );
        if relative > worst {
            worst = relative;
            worst_block = name;
        }
        offset += count;
    }
    eprintln!("{case:?}: worst relative diff {worst:e} ({worst_block})");
}

/// The default chunk length against a token count that is a clean multiple of
/// it, i.e. the shape the CUDA kernel will actually see.
#[test]
fn chunked_matches_sequential_on_whole_chunks() {
    assert_agrees(&Case::new(128, 64), 1.0e-5);
}

/// Token counts that straddle the chunk length in every direction, which is
/// where a partial tail chunk is either handled or silently dropped.
#[test]
fn chunked_matches_sequential_on_partial_tail_chunks() {
    for n_tokens in [1, 2, 15, 16, 17, 31, 33, 63, 65, 127, 129] {
        assert_agrees(&Case::new(n_tokens, 16), 1.0e-5);
    }
}

/// A chunk longer than the sequence, and a chunk of one token: the two
/// degenerate layouts. Chunk length 1 makes the chunkwise form structurally
/// equivalent to the sequential one, so it must also be numerically equivalent.
#[test]
fn chunked_matches_sequential_on_degenerate_chunk_lengths() {
    assert_agrees(&Case::new(20, 4096), 1.0e-5);
    assert_agrees(&Case::new(20, 1), 1.0e-5);
}

/// One head, one sequence: the grid the CUDA kernel will collapse to, and the
/// case where an off-by-one in the head or sequence stride hides.
#[test]
fn chunked_matches_sequential_on_a_degenerate_grid() {
    let mut case = Case::new(65, 32);
    case.h = 1;
    case.n_seqs = 1;
    assert_agrees(&case, 1.0e-5);
}

/// The gate is the input that decides the chunk layout, so every regime gets a
/// case: no decay, decay strong enough to force the guard to halve the chunk,
/// and a token no normalisation survives (which must land on the sequential
/// step rather than on an infinity).
#[test]
fn chunked_matches_sequential_for_adverse_gates() {
    for gate in [Gate::Zero, Gate::Strong, Gate::Extreme] {
        let mut case = Case::new(96, 64);
        case.gate = gate;
        assert_agrees(&case, 1.0e-5);
    }
}

/// `beta = 0` freezes the state and `beta = 1` replaces it: the two ends of the
/// `(I+T)` solve, where the triangular system is respectively the identity and
/// at its worst conditioned.
#[test]
fn chunked_matches_sequential_at_the_extremes_of_beta() {
    for beta in [Beta::Zero, Beta::One] {
        let mut case = Case::new(96, 64);
        case.beta = beta;
        assert_agrees(&case, 1.0e-5);
    }
}

/// A per-channel (KDA) gate turns `A` into a vector, which is the part of the
/// derivation the scalar cases cannot exercise: the decay bound, the reverse
/// cumulative sum into `grad_g`, and the row scaling all become per-channel.
#[test]
fn chunked_matches_sequential_for_a_per_channel_gate() {
    for gate in [Gate::Mild, Gate::MixedSign, Gate::Extreme] {
        let mut case = Case::new(96, 64);
        case.kda = true;
        case.gate = gate;
        assert_agrees(&case, 1.0e-5);
    }
}

/// `K > 1` injects a state gradient in the middle of a chunk (slot `s` is the
/// state `s` tokens back), so the chunk layout has to end a chunk at every one
/// of those tokens. Wrong boundaries do not crash -- they silently drop or
/// misplace a snapshot gradient, which only `grad_state` and the last tokens'
/// `grad_*` would show.
#[test]
fn chunked_matches_sequential_with_state_snapshots() {
    for (k, kda) in [(2, false), (3, true), (5, false)] {
        let mut case = Case::new(40, 16);
        case.k = k;
        case.kda = kda;
        assert_agrees(&case, 1.0e-5);
    }
}

/// `K` covering the whole sequence: every token is a snapshot, so every chunk
/// is one token long and the chunk layout degenerates to the sequential one.
#[test]
fn chunked_matches_sequential_when_every_token_is_a_snapshot() {
    let mut case = Case::new(12, 64);
    case.k = 12;
    assert_agrees(&case, 1.0e-5);
}
