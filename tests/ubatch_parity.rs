//! The effect of `n_ubatch` on the non-packed training path.
//!
//! The VRAM sweep assumes the physical micro-batch is a memory knob: split a
//! logical batch into `n_batch / n_ubatch` micro-batches, accumulate their
//! gradients, take the same optimizer step, pay less peak memory. These tests
//! measure that assumption instead of trusting it, with the tolerance fixed
//! before the comparison.
//!
//! Measured result (Qwen3-0.6B-Q8_0, CPU, `lr = 1e-2`, one epoch over the same
//! tokens, `n_ctx` padded by llama.cpp to 256):
//!
//! | batch | ubatch | loss | Σ logprob of the probe |
//! |------:|-------:|-----:|-----------------------:|
//! |    64 |     64 | 3.73 |                 -88.12 |
//! |    64 |     32 | 4.37 |                 -86.49 |
//! |    64 |     16 | 4.12 |                 -86.94 |
//! |    32 |     32 | 6.66 |                 -83.55 |
//!
//! The loss already differs at `lr = 1e-9`, i.e. before any weight moves, so
//! this is a forward-path difference and not gradient-accumulation rounding:
//! on this path a micro-batch is not a transparent slice of the logical batch.
//! Both regimes therefore train *different* objectives, and a VRAM comparison
//! across `n_ubatch` values compares runs that are not doing the same thing.
//!
//! These tests pin that coupling. If a future change makes the micro-batch
//! transparent (uniform per-token loss normalization across an accumulation
//! period, or a differentiable KV shared across micro-batches), the second test
//! fails: that is the signal to flip it into a parity assertion and to rerun
//! the VRAM sweep, whose recommendation is blocked on exactly this property.

mod common;

use retrograd::{Device, LoraConfig, TargetSet, TrainConfig, Trainer};

/// Absolute tolerance on the loss between two micro-batch sizes, chosen for
/// gradient-accumulation rounding in F32 -- the only difference a transparent
/// micro-batch would be allowed to produce.
const LOSS_TOLERANCE: f32 = 1.0e-3;
/// Same, on a per-token logprob of the trained adapter.
const LOGPROB_TOLERANCE: f32 = 5.0e-3;

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

const PROBE_TEXT: &str = "The five boxing wizards jump quickly, judging my vow.";

/// Enough tokens to fill whole rows, so no micro-batch is short of active
/// labels: padding would be a second, independent reason for a divergence.
const TRAIN_TOKENS: usize = 513;

fn config(n_ubatch: u32) -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch,
        epochs: 1,
        learning_rate: 1.0e-2,
        // The CPU path is the reproducible reference: it keeps backend
        // scheduling out of the comparison.
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    // Dropout stays at its default 0.0: any noise source would make the runs
    // differ for a reason that has nothing to do with the micro-batch.
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

struct Outcome {
    train_loss: f32,
    /// Per-token logprobs of a fixed probe under the trained adapter. Two runs
    /// that applied the same gradient produce the same numbers here, so this
    /// compares trained weights without tensor introspection.
    probe: Vec<f32>,
}

fn train_with_ubatch(n_ubatch: u32) -> Outcome {
    let model = common::model_path();
    let mut trainer = Trainer::new(&model, config(n_ubatch)).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");

    let text = TRAIN_TEXT.repeat(24);
    let tokens = trainer.tokenize_text(&text).expect("tokenize");
    assert!(tokens.len() >= TRAIN_TOKENS, "fixture text is too short");
    let metrics = trainer
        .train_tokens(&tokens[..TRAIN_TOKENS])
        .expect("train");

    let probe_tokens = trainer.tokenize_text(PROBE_TEXT).expect("tokenize probe");
    let probe = trainer.score_tokens(&probe_tokens).expect("score probe");

    Outcome {
        train_loss: metrics.train_loss,
        probe,
    }
}

/// One sweep pins both properties: every supported micro-batch trains finitely,
/// and reducing it still changes the characterized objective. Keeping the
/// assertions in one scenario avoids training the same three configurations
/// twice.
#[test]
fn every_micro_batch_is_finite_and_preserves_the_characterized_difference() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let outcomes = [64, 32, 16].map(|n_ubatch| (n_ubatch, train_with_ubatch(n_ubatch)));
    for (n_ubatch, outcome) in &outcomes {
        assert!(
            outcome.train_loss.is_finite() && outcome.train_loss > 0.0,
            "ubatch={n_ubatch} produced a non-finite loss: {}",
            outcome.train_loss
        );
        assert!(
            outcome.probe.iter().all(|value| value.is_finite()),
            "ubatch={n_ubatch} produced non-finite probe scores"
        );
    }

    let reference = &outcomes[0].1;
    for (n_ubatch, outcome) in &outcomes[1..] {
        let loss_gap = (outcome.train_loss - reference.train_loss).abs();
        assert_eq!(
            outcome.probe.len(),
            reference.probe.len(),
            "ubatch={n_ubatch} probe length"
        );
        let probe_gap = outcome
            .probe
            .iter()
            .zip(reference.probe.iter())
            .map(|(value, expected)| (value - expected).abs())
            .fold(0.0_f32, f32::max);
        // Transparency requires both the forward loss and the trained policy
        // to agree. A model may expose the coupling predominantly in either
        // observable, so fail only when both fall inside their tolerances.
        assert!(
            loss_gap > LOSS_TOLERANCE || probe_gap > LOGPROB_TOLERANCE,
            "ubatch={n_ubatch} now matches the full-width batch \
             (loss gap {loss_gap}, probe gap {probe_gap}): the micro-batch has \
             become transparent, so turn this into a parity assertion and \
             rerun the VRAM sweep"
        );
    }
}
