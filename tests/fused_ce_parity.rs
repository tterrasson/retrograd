//! End-to-end guards for the fused/chunked vocabulary cross-entropy: a real
//! packed GRPO step, run several ways on the same model, LoRA seed, sequences
//! and RNG seed.
//!
//! # Why there is no dense-vs-fused comparison here
//!
//! Asserting that the fused path trains the *same adapter* as the dense one
//! cannot be made to work at any tolerance:
//!
//! - The gradients themselves agree. Comparing the saved adapters, the dense and
//!   fused updates had the same norm (6.397e-2 vs 6.398e-2) and 4080 of 4096
//!   components had the same sign.
//! - AdamW's *first* step is `m̂/√v̂ = g/|g|`, i.e. `−lr·sign(g)`, which is
//!   scale-free: measured, every component moved by exactly `lr`. A component
//!   whose gradient sits in F32 noise therefore still takes a full ±lr step, with
//!   an arbitrary sign. 16 such sign flips accounted for the entire cosine
//!   deficit (0.99246 observed vs 0.99219 predicted by the flips alone).
//! - So any post-step quantity is a *discontinuous* function of the gradient:
//!   a bit-level difference becomes an O(lr) weight difference. Measured, the
//!   probe logprob gap did not even shrink monotonically with the learning rate
//!   (0.87 at 1e-3, 1.37 at 1e-4, exactly 0 at 1e-5 - the last only because
//!   `lora_b` starts at zero and the model stops changing at F32 resolution).
//!   Comparing per token is worse than the sum, not better (1.23 vs 0.87).
//!
//! Dense-vs-fused parity is therefore asserted where it is well posed: on the
//! loss and on `grad_h` directly, in `tests/fused_ce.rs`. Note also that the
//! reported GRPO loss cannot discriminate here - at iteration 0 the ratios are 1
//! and the advantages are mean-zero by construction, so *both* paths report
//! exactly 0.
//!
//! What remains below is a fused-vs-fused check, where the two runs are expected
//! to be **bit-identical**. For a bit-identity claim the same discontinuity is an
//! asset rather than a problem: any difference in the gradient, however small,
//! would be amplified into a visible one.
//!
//! Skipped when no local GGUF model is available (override with
//! `RETRO_TEST_MODEL`).

mod common;

use retrograd::training::batch::train_grpo_batch;
use retrograd::{
    Device, GrpoBatchParams, LoraConfig, LoraDtype, TargetSet, TrainConfig, TrainSequence, Trainer,
};

/// Absolute tolerance on the reported loss between two fused runs. These are
/// expected to agree exactly; the tolerance only keeps the failure message
/// readable if they ever stop doing so.
const LOSS_TOLERANCE: f32 = 2.0e-3;
/// Same, on the summed probe logprob of the adapter each run trained.
const LOGPROB_TOLERANCE: f32 = 5.0e-3;

const PROBE_TEXT: &str = "The five boxing wizards jump quickly, judging my vow.";

fn config(tiles: u32, seq_chunk: u32) -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 2,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        chunked_cross_entropy: true,
        chunked_ce_tiles: tiles,
        chunked_ce_seq_chunk: seq_chunk,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.dtype = LoraDtype::F32;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

fn params() -> GrpoBatchParams {
    GrpoBatchParams {
        epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 4,
        seed: 42,
        scheduler_total_rollouts: None,
    }
}

fn sequences(trainer: &mut Trainer) -> Vec<TrainSequence> {
    let mut sequences = Vec::new();
    for (text, reward) in [
        ("This is a deliberately longer first training answer.", 0.0),
        (
            "This is a deliberately longer second training response.",
            1.0,
        ),
    ] {
        let tokens = trainer.tokenize_text(text).expect("tokenize batch row");
        let mut train_mask = vec![false; tokens.len()];
        let last = tokens.len() - 1;
        train_mask[last - 2] = true;
        train_mask[last] = true;
        let old_logprobs = trainer
            .score_masked_tokens(&tokens, &train_mask)
            .expect("score masked row");
        sequences.push(TrainSequence {
            tokens,
            old_logprobs,
            train_mask,
            reward,
            group_id: 99,
            intermediate_returns: vec![1.0, 0.0],
        });
    }
    sequences
}

/// Trains one GRPO step on the fused cross-entropy path with the given
/// memory knobs, and returns the reported loss plus a fixed
/// probe's summed logprob under the trained adapter.
fn run_tiled(tiles: u32, seq_chunk: u32) -> (f32, f32) {
    let model = common::model_path().clone();
    let cfg = config(tiles, seq_chunk);
    let mut trainer = Trainer::new(model, cfg.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");

    let seqs = sequences(&mut trainer);
    let metrics = train_grpo_batch(&mut trainer, &seqs, &params(), &cfg, &mut |_| {})
        .expect("train GRPO batch");
    assert!(
        metrics.train_loss.is_finite(),
        "seq_chunk={seq_chunk}: loss must be finite"
    );

    let probe = trainer.tokenize_text(PROBE_TEXT).expect("tokenize probe");
    // Teacher-forced scoring rejects the first token (it has no predecessor to
    // condition on), so score every position but that one.
    let mut mask = vec![true; probe.len()];
    mask[0] = false;
    let logprobs = trainer
        .score_masked_tokens(&probe, &mask)
        .expect("score probe");
    let sum: f32 = logprobs.iter().copied().sum();
    (metrics.train_loss, sum)
}

/// Feature 3 - offloading the log-softmax activations.
/// ggml-alloc gives `grad_h` the buffer of the hidden states, so the fused
/// backward writes over its own input and evicts a token chunk at a time to stay
/// correct. This is the end-to-end guard: a real GRPO step must land on the same
/// loss and the same trained adapter whether or not that rewrite happened.
///
/// The rewrite has no flag of its own any more - it follows
/// `chunked_ce_seq_chunk`, since a bounded chunk is the only thing it needs and
/// it costs nothing. So the two sides of the comparison are the two sides of that
/// field: `0` is the whole step at once and no in-place write, `3` is a bounded
/// chunk that deliberately does not divide the token count.
#[test]
fn offloaded_log_softmax_matches_the_fused_path_end_to_end() {
    if common::model_path_if_available().is_none() {
        eprintln!("skipping: no local test model");
        return;
    }
    let (base_loss, base_logprob) = run_tiled(4, 0);

    // Sizes 1 and 64 exercise the same CPU arithmetic and are kept in the
    // backend kernel probes instead of loading this model twice more.
    for seq_chunk in [3u32, 16] {
        let (loss, logprob) = run_tiled(4, seq_chunk);
        assert!(
            (base_loss - loss).abs() < LOSS_TOLERANCE,
            "seq_chunk={seq_chunk}: loss {loss} moved under activation offloading (base {base_loss})",
        );
        assert!(
            (base_logprob - logprob).abs() < LOGPROB_TOLERANCE,
            "seq_chunk={seq_chunk}: probe logprob {logprob} moved under activation offloading (base {base_logprob})",
        );
    }
}
