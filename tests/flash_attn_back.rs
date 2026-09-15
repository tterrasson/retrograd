//! Model-free coverage for the native CPU Flash Attention backward.
//!
//! GPU tests keep using the analytic CPU probe as their oracle. The fourth
//! source-layout flag selects the actual CPU GGML op here, which lets this lane
//! distinguish a shared kernel bug from an oracle bug and pins the determinism of repeated runs.

use retrograd::{ProbeInputs, ProbeOp, probe_op};

fn pseudo_random(n: usize, mut state: u64, range: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / ((1u32 << 24) as f32) * 2.0 - 1.0) * range
        })
        .collect()
}

fn run(native_cpu: bool, kv_f32: bool, sinks: bool, window: Option<(usize, usize)>) -> Vec<f32> {
    let (hsk, hsv, nq, nkv, nhead, nhead_kv, nbatch) =
        (32usize, 24usize, 5usize, 9usize, 4usize, 2usize, 2usize);
    let n_q = hsk * nq * nhead * nbatch;
    let n_k = hsk * nkv * nhead_kv * nbatch;
    let n_v = hsv * nkv * nhead_kv * nbatch;
    let n_do = hsv * nhead * nq * nbatch;
    let mut packed = Vec::with_capacity(n_q + n_k + n_v + n_do);
    packed.extend(pseudo_random(n_q, 0xb001_0001, 0.4));
    packed.extend(pseudo_random(n_k, 0xb001_0002, 0.4));
    packed.extend(pseudo_random(n_v, 0xb001_0003, 0.4));
    packed.extend(pseudo_random(n_do, 0xb001_0004, 0.7));
    if sinks {
        packed.extend(pseudo_random(nhead, 0xb001_0005, 1.5));
    }

    let dims = [hsk as i64, hsv as i64, nq as i64, nkv as i64];
    let layout = [nhead as i64, nhead_kv as i64, nbatch as i64, 1];
    let (nwin, idxs) = if let Some((nwin, offset)) = window {
        let mut idxs = Vec::with_capacity(nwin * nbatch);
        for ib in 0..nbatch {
            for j in 0..nwin {
                idxs.push((ib * nkv + offset + j) as f32);
            }
        }
        (nwin, idxs)
    } else {
        (nkv, Vec::new())
    };
    let flags = [
        window.map_or(0, |(nwin, _)| nwin as i64),
        kv_f32 as i64,
        sinks as i64,
        native_cpu as i64,
    ];
    probe_op(
        ProbeOp::FlashAttnBack,
        false,
        ProbeInputs::pair(dims, &packed, layout, &[0.0; 4]).with_src2(
            (native_cpu || kv_f32 || sinks || window.is_some()).then_some((flags, idxs.as_slice())),
        ),
        [1.0 / (hsk as f32).sqrt(), 0.75],
        n_q + hsk * nwin * nhead_kv * nbatch + hsv * nwin * nhead_kv * nbatch,
    )
    .expect("CPU Flash Attention backward probe")
}

#[test]
fn flash_attn_back_cpu_matches_the_analytic_streaming_oracle() {
    for (kv_f32, sinks, window) in [
        (false, false, None),
        (true, false, None),
        (false, true, None),
        (false, false, Some((3, 4))),
    ] {
        let expected = run(false, kv_f32, sinks, window);
        let actual = run(true, kv_f32, sinks, window);
        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 2.0e-5,
            "CPU FLASH_ATTN_BACK max diff {max_diff:e} for kv_f32={kv_f32}, sinks={sinks}, window={window:?}"
        );
    }
}

#[test]
fn flash_attn_back_cpu_is_bit_deterministic() {
    let first = run(true, false, false, Some((3, 4)));
    let second = run(true, false, false, Some((3, 4)));
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
