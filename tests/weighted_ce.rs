//! Independent validation of the fork's weighted cross-entropy loss, the
//! differentiable core of the PPO objective. The probe harness runs the exact
//! ggml ops the training graph uses; outputs are compared against an analytic
//! Rust reference for one-hot, weighted, masked, and negative-weight rows, and
//! Metal is compared against CPU when a GPU is present.

mod common;

use retrograd::{ProbeInputs, ProbeOp, probe_op};

const NC: usize = 8; // classes (row width)
const NR: usize = 4; // rows

/// Deterministic, non-degenerate logits.
fn logits() -> Vec<f32> {
    (0..NC * NR)
        .map(|i| ((i * 37 + 11) % 17) as f32 * 0.25 - 2.0)
        .collect()
}

/// Rows: a one-hot SFT row, a positively weighted row, a masked row, and a
/// negatively weighted row (PPO weights may be negative).
fn labels() -> (Vec<f32>, usize) {
    let mut labels = vec![0.0_f32; NC * NR];
    labels[2] = 1.0; // row 0: classic one-hot
    labels[NC + 5] = 0.7; // row 1: weight 0.7
    // row 2: all zero = masked
    labels[3 * NC + 1] = -0.4; // row 3: weight -0.4
    (labels, 3) // 3 active rows
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |m, v| m.max(v as f64));
    let sum: f64 = row.iter().map(|&v| ((v as f64) - max).exp()).sum();
    row.iter().map(|&v| (v as f64) - max - sum.ln()).collect()
}

/// Analytic loss: -(sum over active rows of labels. log_softmax) / n_active.
fn expected_loss(logits: &[f32], labels: &[f32], n_active: usize) -> f32 {
    let mut total = 0.0_f64;
    for row in 0..NR {
        let ls = log_softmax(&logits[row * NC..(row + 1) * NC]);
        let l = &labels[row * NC..(row + 1) * NC];
        if l.iter().any(|&v| v != 0.0) {
            total += l
                .iter()
                .zip(&ls)
                .map(|(&w, &lp)| w as f64 * lp)
                .sum::<f64>();
        }
    }
    (-total / n_active as f64) as f32
}

/// Analytic gradient: (sum(labels) * softmax - labels) * grad / n_active for
/// active rows, zero for masked rows.
fn expected_grad(logits: &[f32], labels: &[f32], grad: f32, n_active: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; NC * NR];
    for row in 0..NR {
        let l = &labels[row * NC..(row + 1) * NC];
        if l.iter().all(|&v| v == 0.0) {
            continue;
        }
        let mass: f64 = l.iter().map(|&v| v as f64).sum();
        let ls = log_softmax(&logits[row * NC..(row + 1) * NC]);
        for i in 0..NC {
            let softmax = ls[i].exp();
            out[row * NC + i] =
                ((mass * softmax - l[i] as f64) * grad as f64 / n_active as f64) as f32;
        }
    }
    out
}

fn shape(ne0: usize, ne1: usize) -> [i64; 4] {
    [ne0 as i64, ne1 as i64, 1, 1]
}

fn run_forward(use_gpu: bool) -> f32 {
    let (labels, n_active) = labels();
    probe_op(
        ProbeOp::CrossEntropyLoss,
        use_gpu,
        ProbeInputs::pair(shape(NC, NR), &logits(), shape(NC, NR), &labels),
        [0.0, n_active as f32],
        1,
    )
    .expect("cross-entropy forward probe")[0]
}

fn run_backward(use_gpu: bool, grad: f32) -> Vec<f32> {
    let (labels, n_active) = labels();
    probe_op(
        ProbeOp::CrossEntropyLossBack,
        use_gpu,
        ProbeInputs::pair(shape(1, 1), &[grad], shape(NC, NR), &logits())
            .with_src2(Some((shape(NC, NR), labels.as_slice()))),
        [0.0, n_active as f32],
        NC * NR,
    )
    .expect("cross-entropy backward probe")
}

#[test]
fn weighted_forward_matches_the_analytic_loss_on_cpu() {
    let (labels, n_active) = labels();
    let expected = expected_loss(&logits(), &labels, n_active);
    let actual = run_forward(false);
    assert!(
        (actual - expected).abs() < 1e-5,
        "loss {actual} != expected {expected}"
    );
}

#[test]
fn weighted_backward_matches_the_analytic_gradient_on_cpu() {
    let grad = 1.5_f32;
    let (labels, n_active) = labels();
    let expected = expected_grad(&logits(), &labels, grad, n_active);
    let actual = run_backward(false, grad);
    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-5, "grad[{i}] {a} != expected {e}");
    }
    // The masked row (row 2) must be exactly zero.
    assert!(actual[2 * NC..3 * NC].iter().all(|&v| v == 0.0));
}

#[test]
fn one_hot_rows_keep_the_classic_gradient() {
    // With a single one-hot row the generalization must be a no-op:
    // grad = (softmax - one_hot) * d.
    let logits: Vec<f32> = (0..NC).map(|i| i as f32 * 0.3 - 1.0).collect();
    let mut labels = vec![0.0_f32; NC];
    labels[4] = 1.0;
    let actual = probe_op(
        ProbeOp::CrossEntropyLossBack,
        false,
        ProbeInputs::pair(shape(1, 1), &[1.0], shape(NC, 1), &logits)
            .with_src2(Some((shape(NC, 1), labels.as_slice()))),
        [0.0, 1.0],
        NC,
    )
    .expect("one-hot backward probe");
    let ls = log_softmax(&logits);
    for i in 0..NC {
        let expected = (ls[i].exp() - labels[i] as f64) as f32;
        assert!((actual[i] - expected).abs() < 1e-5, "grad[{i}]");
    }
}

#[test]
fn metal_weighted_cross_entropy_matches_cpu() {
    if !common::metal_compiled() || !common::gpu_device_present() {
        eprintln!("skipping: Metal backend unavailable");
        return;
    }
    let cpu_loss = run_forward(false);
    let gpu_loss = run_forward(true);
    assert!(
        (cpu_loss - gpu_loss).abs() < 1e-4,
        "forward loss cpu {cpu_loss} != metal {gpu_loss}"
    );

    let cpu_grad = run_backward(false, 1.5);
    let gpu_grad = run_backward(true, 1.5);
    for (i, (&c, &g)) in cpu_grad.iter().zip(&gpu_grad).enumerate() {
        assert!((c - g).abs() < 1e-4, "grad[{i}] cpu {c} != metal {g}");
    }
}
