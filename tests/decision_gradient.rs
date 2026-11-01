//! The gradient of a decision head, end to end.
//!
//! A decision model reads its answer from the embeddings output: one score per
//! option, read at the marker token that opens the option, in the column of
//! the question type, and a softmax over the options of each question. Its
//! training loss is that softmax's cross-entropy, built over the embeddings
//! output by `llama_opt_step_grouped`. This file checks that loss and its
//! gradient on the generated decision model, through every op family of the
//! graph - LayerNorm with and without bias, the fused erf GEGLU, RoPE and a
//! sliding window, the head evaluated once per question type over a row of
//! `token_types`, the scorer's erf GELU - against an oracle that shares none
//! of the training graph: the same loss recomputed from `llama_decode`
//! embeddings in F64, and four-point central differences on the weights.
//!
//! The model is F32 throughout, so a disagreement is a gradient rule and not a
//! storage rounding.

mod common;

use retrograd::{Device, GroupedGradientCase, grouped_gradient_probe};

const BOS: i32 = 2;
const EOS: i32 = 1;
const MASK: i32 = 4;

/// A byte-fallback token: the vocabulary has five specials before the bytes.
fn byte(b: u8) -> i32 {
    5 + i32::from(b)
}

fn text(s: &str) -> Vec<i32> {
    s.bytes().map(byte).collect()
}

/// One question laid out the way the `laya` readout reads it:
/// `[bos] question [eos] ([mask] option)* [eos] state [eos]`.
/// Returns the tokens and the positions of the markers.
fn question(instructions: &str, options: &[&str], state: &str) -> (Vec<i32>, Vec<usize>) {
    let mut tokens = vec![BOS];
    tokens.extend(text(instructions));
    tokens.push(EOS);
    let mut markers = Vec::new();
    for option in options {
        markers.push(tokens.len());
        tokens.push(MASK);
        tokens.extend(text(option));
    }
    tokens.push(EOS);
    tokens.extend(text(state));
    tokens.push(EOS);
    (tokens, markers)
}

struct Batch {
    tokens: Vec<i32>,
    positions: Vec<i32>,
    seq_ids: Vec<i32>,
    outputs: Vec<bool>,
    rows: Vec<i32>,
    columns: Vec<i32>,
    targets: Vec<f32>,
}

const N_OPT_MAX: u32 = 3;

/// Two questions packed as two sequences, three groups over them: a choice
/// with a one-hot target, a score with a soft one read from the same markers,
/// and a two-option noul padded to three. Both sequences are longer than the
/// sliding window of the encoder, so the window masks something.
fn batch() -> Batch {
    let (q0, m0) = question("route?", &["bill", "ship", "tech"], "charged twice");
    let (q1, m1) = question("angry?", &["no", "yes"], "nobody replied");

    let mut b = Batch {
        tokens: Vec::new(),
        positions: Vec::new(),
        seq_ids: Vec::new(),
        outputs: Vec::new(),
        rows: Vec::new(),
        columns: Vec::new(),
        targets: Vec::new(),
    };
    let mut starts = Vec::new();
    for (seq, q) in [&q0, &q1].into_iter().enumerate() {
        starts.push(b.tokens.len());
        for (pos, &token) in q.iter().enumerate() {
            b.tokens.push(token);
            b.positions.push(pos as i32);
            b.seq_ids.push(seq as i32);
            b.outputs.push(false);
        }
    }
    let mut group = |start: usize, markers: &[usize], column: i32, targets: &[f32]| {
        assert_eq!(markers.len(), targets.len());
        for (&m, &target) in markers.iter().zip(targets) {
            b.outputs[start + m] = true;
            b.rows.push((start + m) as i32);
            b.targets.push(target);
        }
        for _ in markers.len()..N_OPT_MAX as usize {
            b.rows.push(-1);
            b.targets.push(0.0);
        }
        b.columns.push(column);
    };
    group(starts[0], &m0, 0, &[1.0, 0.0, 0.0]); // choice
    group(starts[0], &m0, 1, &[0.2, 0.5, 0.3]); // score, soft
    group(starts[1], &m1, 2, &[0.3, 0.7]); // noul, soft, padded
    b
}

/// Each tensor family of the graph, with elements spread over it: both halves
/// of the fused GEGLU projection, each row of `token_types`.
const TENSORS: &[(&str, &[i64])] = &[
    ("token_embd_norm.weight", &[0, 17, 63]),
    ("blk.0.attn_qkv.weight", &[3, 64 * 70 + 5, 64 * 191 + 63]),
    ("blk.0.ffn_up.weight", &[1, 64 * 50 + 9, 64 * 150 + 33]),
    ("blk.0.ffn_down.weight", &[0, 96 * 31 + 7]),
    ("blk.1.attn_norm.weight", &[2, 40]),
    ("blk.1.attn_output.weight", &[5, 64 * 63 + 1]),
    ("blk.1.ffn_up.weight", &[64 * 120 + 11]),
    ("output_norm.weight", &[4, 60]),
    ("token_types.weight", &[3, 64 + 3, 128 + 3]),
    ("blk.2.attn_norm.weight", &[6]),
    ("blk.2.attn_norm.bias", &[0, 50]),
    ("blk.2.attn_qkv.bias", &[1, 64 + 2, 128 + 3]),
    ("blk.2.attn_output.bias", &[9]),
    ("blk.2.ffn_up.weight", &[64 * 100 + 1]),
    ("blk.2.ffn_up.bias", &[0, 127]),
    ("blk.2.ffn_down.bias", &[8]),
    ("cls.norm.weight", &[0, 33]),
    ("cls.norm.bias", &[0, 33]),
    ("cls.weight", &[0, 64 * 40 + 2]),
    ("cls.bias", &[7]),
    ("cls.output.weight", &[0, 63]),
    ("cls.output.bias", &[0]),
];

fn case(b: &Batch, flash_attn: bool) -> GroupedGradientCase<'_> {
    GroupedGradientCase {
        tokens: &b.tokens,
        positions: &b.positions,
        seq_ids: &b.seq_ids,
        outputs: &b.outputs,
        n_opt_max: N_OPT_MAX,
        rows: &b.rows,
        columns: &b.columns,
        targets: &b.targets,
        inv_temperature: 0.8,
        flash_attn: Some(flash_attn),
    }
}

/// Checks the analytic gradient of the graph built with or without flash
/// attention. The finite differences always come from the graph without it:
/// the CPU flash-attention kernel reads K and V in F16, which is noise at the
/// step the differences need, and the analytic gradient of either graph is a
/// gradient of the same F32 function.
fn check(flash_attn: bool) {
    let Some(model) = common::tiny_decision_model_path_if_available() else {
        eprintln!("skipping: tiny decision fixture not present (scripts/fetch-cpu-fixture.sh)");
        return;
    };
    let b = batch();
    let mut failures = Vec::new();
    for &(name, elements) in TENSORS {
        let reference = grouped_gradient_probe(
            &model,
            Device::Cpu,
            name,
            &case(&b, false),
            elements,
            1.0e-2,
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        let g = if flash_attn {
            grouped_gradient_probe(&model, Device::Cpu, name, &case(&b, true), elements, 1.0e-2)
                .unwrap_or_else(|err| panic!("{name}: {err}"))
        } else {
            reference.clone()
        };
        assert_eq!(
            g.n_missing_backward, 0,
            "{name}: the training graph has ops without a gradient rule"
        );
        assert!(
            (g.loss_step - g.loss_decode).abs() <= 1.0e-5 * g.loss_decode.abs().max(1.0),
            "{name}: the grouped loss graph gives {} and the decoded embeddings {}",
            g.loss_step,
            g.loss_decode
        );
        for (i, &element) in elements.iter().enumerate() {
            let analytic = f64::from(g.analytic[i]);
            let numeric = reference.numeric[i];
            let tolerance = 3.0e-5 + 3.0e-2 * numeric.abs();
            if !analytic.is_finite()
                || !numeric.is_finite()
                || (analytic - numeric).abs() > tolerance
            {
                failures.push(format!(
                    "{name}[{element}]: analytic {analytic:e}, numeric {numeric:e}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n{}",
        failures.join("\n")
    );
}

#[test]
fn grouped_loss_gradient_matches_finite_differences() {
    check(false);
}

/// Both packed sequences and the sliding window leave rows whose first keys
/// are masked, which is where a flash-attention backward that folds a masked
/// key into an empty running maximum goes wrong.
#[test]
fn grouped_loss_gradient_matches_finite_differences_with_flash_attention() {
    check(true);
}
