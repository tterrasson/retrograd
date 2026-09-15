//! End-to-end parity for activation recomputation (memory plan 02).
//!
//! The packed multi-sequence optimizer is run from identical LoRA weights with
//! and without checkpointing. Matching loss and post-update scores prove that
//! the recomputed forward feeds the same LoRA gradient. Rank 1 covers the
//! smallest adapter geometry; rank 32 exercises the wider contraction and two
//! consecutive optimizer steps.

mod common;

use retrograd::training::batch::train_grpo_batch;
use retrograd::{
    CheckpointDtype, Device, GrpoBatchParams, LoraConfig, LoraDtype, TargetSet, TrainConfig,
    TrainSequence, Trainer,
};

const LOSS_TOLERANCE: f32 = 1.0e-5;
const LOGPROB_TOLERANCE: f32 = 1.0e-4;
const PROBE_TEXT: &str = "The five boxing wizards jump quickly, judging my vow.";

fn config(checkpointing: bool, chunked_cross_entropy: bool, device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 2,
        epochs: 1,
        learning_rate: 1.0e-3,
        device,
        gradient_checkpointing: checkpointing,
        checkpoint_every_n_layers: 2,
        chunked_cross_entropy,
        chunked_ce_tiles: 4,
        ..TrainConfig::default()
    }
}

fn lora(rank: u32) -> LoraConfig {
    let mut cfg = LoraConfig::qv(rank, 2.0 * rank as f32);
    cfg.seed = 7;
    cfg.dtype = LoraDtype::F32;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

fn sequences(trainer: &mut Trainer) -> Vec<TrainSequence> {
    let prefix = trainer
        .tokenize_text("A shared checkpointing prompt asks for")
        .expect("tokenize shared prefix");
    ["the first answer.", "a different second answer."]
        .into_iter()
        .enumerate()
        .map(|(index, suffix)| {
            let suffix = trainer
                .tokenize_text(suffix)
                .expect("tokenize packed suffix");
            assert!(suffix.len() >= 2);
            let mut tokens = prefix.clone();
            tokens.extend_from_slice(&suffix[suffix.len() - 2..]);
            let mut train_mask = vec![false; tokens.len()];
            let last = tokens.len() - 1;
            train_mask[last - 1] = true;
            train_mask[last] = true;
            let old_logprobs = trainer
                .score_masked_tokens(&tokens, &train_mask)
                .expect("score packed row");
            TrainSequence {
                tokens,
                old_logprobs,
                train_mask,
                reward: index as f32,
                group_id: 17,
                intermediate_returns: vec![1.0, 0.0],
            }
        })
        .collect()
}

fn report_usize(report: &str, key: &str) -> usize {
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix(key))
        .and_then(|value| value.trim_start_matches(':').trim().parse().ok())
        .unwrap_or_else(|| panic!("missing {key} in backend report:\n{report}"))
}

fn run(
    rank: u32,
    steps: u32,
    checkpointing: bool,
    chunked_cross_entropy: bool,
    device: Device,
) -> (f32, Vec<f32>, usize, bool) {
    let cfg = config(checkpointing, chunked_cross_entropy, device);
    let mut trainer = Trainer::new(common::model_path(), cfg.clone()).expect("load trainer");
    trainer.create_lora(&lora(rank)).expect("create lora");

    let report = trainer.backend_report().expect("backend report");
    let expected = if checkpointing { "enabled" } else { "disabled" };
    assert!(
        report.contains(&format!("gradient_checkpointing: {expected}")),
        "{report}"
    );
    assert!(report.contains("checkpoint_every_n_layers: 2"), "{report}");
    #[cfg(retro_vulkan)]
    if device == Device::Gpu && std::env::var_os("RETRO_DISABLE_DIFF_FLASH_ATTN").is_none() {
        assert!(
            report.contains("optimizer_flash_attention: enabled"),
            "{report}"
        );
    }

    let sequences = sequences(&mut trainer);
    let params = GrpoBatchParams {
        epochs: steps,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 4,
        seed: 42,
        scheduler_total_rollouts: None,
    };
    let metrics = train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {})
        .expect("train packed batch");
    assert!(metrics.train_loss.is_finite());
    let compute_buffer_bytes = report_usize(
        &trainer.backend_report().expect("post-train backend report"),
        "compute_buffer_bytes",
    );

    let probe = trainer.tokenize_text(PROBE_TEXT).expect("tokenize probe");
    let scores = trainer.score_tokens(&probe).expect("score trained adapter");
    (
        metrics.train_loss,
        scores,
        compute_buffer_bytes,
        trainer
            .supports_shared_prefix_packed_training()
            .expect("model signature"),
    )
}

#[test]
fn recompute_matches_the_packed_gradient_at_rank_1_and_32() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    for (rank, steps) in [(1, 1), (32, 2)] {
        let reference = run(rank, steps, false, false, Device::Cpu);
        let checkpointed = run(rank, steps, true, false, Device::Cpu);
        assert!(
            (reference.0 - checkpointed.0).abs() <= LOSS_TOLERANCE,
            "rank={rank}, steps={steps}: loss {} != {}",
            reference.0,
            checkpointed.0
        );
        assert_eq!(reference.1.len(), checkpointed.1.len());
        let max_gap = reference
            .1
            .iter()
            .zip(&checkpointed.1)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_gap <= LOGPROB_TOLERANCE,
            "rank={rank}, steps={steps}: post-update score gap {max_gap}"
        );
        if reference.3 {
            assert!(
                checkpointed.2 < reference.2,
                "rank={rank}, steps={steps}: checkpointed compute buffer {} is not smaller than {}",
                checkpointed.2,
                reference.2
            );
        } else {
            assert_eq!(
                checkpointed.2, reference.2,
                "the independent-row fallback should not rewrite its graph"
            );
        }
    }
}

/// Trains one packed update with the checkpoints held in `checkpoint_dtype`.
///
/// Separate from `run` because this is the only axis where the two configurations
/// are *not* expected to agree bit for bit, so it carries its own tolerances and
/// must not be folded into the parity helper above.
fn run_with_checkpoint_dtype(
    rank: u32,
    steps: u32,
    checkpoint_dtype: CheckpointDtype,
) -> (f32, Vec<f32>, usize) {
    let cfg = TrainConfig {
        checkpoint_dtype,
        ..config(true, false, Device::Cpu)
    };
    let mut trainer = Trainer::new(common::model_path(), cfg.clone()).expect("load trainer");
    trainer.create_lora(&lora(rank)).expect("create lora");

    let report = trainer.backend_report().expect("backend report");
    let expected = match checkpoint_dtype {
        CheckpointDtype::F32 => "F32",
        CheckpointDtype::F16 => "F16",
        CheckpointDtype::Bf16 => "BF16",
    };
    assert!(
        report.contains(&format!("checkpoint_dtype: {expected}")),
        "the report must name the effective checkpoint precision:\n{report}"
    );

    let sequences = sequences(&mut trainer);
    let params = GrpoBatchParams {
        epochs: steps,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 4,
        seed: 42,
        scheduler_total_rollouts: None,
    };
    let metrics = train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {})
        .expect("train packed batch");
    assert!(
        metrics.train_loss.is_finite(),
        "an F16 checkpoint round-trip must not produce a non-finite loss"
    );
    let compute_buffer_bytes = report_usize(
        &trainer.backend_report().expect("post-train backend report"),
        "compute_buffer_bytes",
    );
    let probe = trainer.tokenize_text(PROBE_TEXT).expect("tokenize probe");
    let scores = trainer.score_tokens(&probe).expect("score trained adapter");
    (metrics.train_loss, scores, compute_buffer_bytes)
}

/// The checkpoint-narrowing rewrite must be exact when it narrows to F32.
///
/// This is the structural test for that rewrite. `RETRO_CHECKPOINT_ROUNDTRIP_F32`
/// forces the two casts to be inserted with F32 on both sides: every part of the
/// machinery runs - the redirection of value reads to the round-tripped copy, the
/// separation of that from the gradient chain, the node ordering, the backend
/// pinning - while the arithmetic is required to be the identity. So the run must
/// reproduce the unnarrowed one *bit for bit*, and any structural mistake shows up
/// with no rounding to hide behind.
///
/// A 16-bit run cannot make this distinction: there a difference is expected, and
/// a broken gradient chain would look exactly like coarse rounding. That is not a
/// hypothetical - the difference a 16-bit checkpoint actually produces on the
/// trained scores is O(1) logprob, which would mask almost any bug.
#[test]
fn narrowing_the_checkpoints_to_f32_reproduces_the_unnarrowed_run_exactly() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let reference = run_with_checkpoint_dtype(32, 2, CheckpointDtype::F32);
    let round_tripped = {
        // Scoped so the variable cannot leak into the sibling tests, which share
        // this process.
        let _env = common::EnvGuard::set("RETRO_CHECKPOINT_ROUNDTRIP_F32", "1");
        run_with_checkpoint_dtype(32, 2, CheckpointDtype::F16)
    };

    assert_eq!(
        reference.0, round_tripped.0,
        "an F32 round-trip changed the loss, so the rewrite is not value-preserving"
    );
    assert_eq!(
        reference.1, round_tripped.1,
        "an F32 round-trip changed the trained adapter"
    );
}

/// A 16-bit checkpoint must still train - and the report must say which precision.
///
/// Deliberately *not* a parity test. Measurement on this fixture puts the update
/// difference at roughly the checkpoint type's own relative precision (~1e-3),
/// about four orders of magnitude above the F32 recompute's reassociation error,
/// which shows up as O(1) logprob on the trained scores. Asserting a tight bound
/// here would mean inventing a tolerance that hides the real cost of the mode;
/// asserting a loose one would prove nothing. What is worth pinning is that the
/// path runs, stays finite, and is named - the exactness of the machinery is
/// covered by the F32 round-trip above, and the cost is documented on
/// `CheckpointDtype`.
#[test]
fn narrowed_checkpoints_train_finitely_and_are_named_in_the_report() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    for dtype in [CheckpointDtype::F16, CheckpointDtype::Bf16] {
        let (loss, scores, compute_bytes) = run_with_checkpoint_dtype(32, 2, dtype);
        assert!(loss.is_finite(), "{dtype:?}: loss {loss} is not finite");
        assert!(
            scores.iter().all(|score| score.is_finite()),
            "{dtype:?}: a narrowed checkpoint produced a non-finite trained score"
        );
        // Not asserted: that the compute buffer shrinks. The narrowed copies are
        // appended after the forward rather than spliced into it, so the original
        // activations are still pinned to the end of the forward and the graph
        // allocator's reserve need not move. The saving is in what is held *across
        // the backward*, which only the device peak of tests/device_memory.rs sees.
        assert!(compute_bytes > 0);
    }
}

/// The checkpoint profile must measure the checkpoints, not the checkpoint list.
///
/// Activation offloading is gated on two numbers the plan cannot supply from a
/// design: how many bytes the retained checkpoints hold, and how long they are
/// held. Both come from a walk of the built backward
/// graph, so what is worth pinning is not a byte count - that is the fixture's,
/// and it moves with the model - but the four relations a wrong walk breaks first.
///
/// The narrowing case is the one that distinguishes a correct walk from a
/// plausible one: with `checkpoint_dtype = f16` each checkpoint is three tensors
/// (the F32 original, its 16-bit store, its widened fetch), and a walk that
/// followed only the original would report *double* the bytes actually retained.
/// So the F16 run must report about half the F32 run's, not the same figure.
#[test]
fn the_checkpoint_profile_measures_what_is_retained_and_for_how_long() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let profile = |checkpointing: bool, checkpoint_dtype: CheckpointDtype| {
        let cfg = TrainConfig {
            checkpoint_dtype,
            ..config(checkpointing, false, Device::Cpu)
        };
        let mut trainer = Trainer::new(common::model_path(), cfg.clone()).expect("load trainer");
        trainer.create_lora(&lora(4)).expect("create lora");
        let sequences = sequences(&mut trainer);
        let params = GrpoBatchParams {
            epochs: 1,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            loss_denominator: 4,
            seed: 42,
            scheduler_total_rollouts: None,
        };
        train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {})
            .expect("train packed batch");
        trainer.memory_report().expect("memory report")
    };

    // A run that retains nothing must report nothing, not zeroes that read like a
    // measurement. This is the whole point of `has_checkpoint_profile`.
    let without = profile(false, CheckpointDtype::F32);
    assert!(
        !without.has_checkpoint_profile(),
        "a run without checkpointing reported {} checkpoints",
        without.checkpoint_count
    );
    assert_eq!(without.checkpoint_retained_bytes, 0);

    let wide = profile(true, CheckpointDtype::F32);
    assert!(
        wide.has_checkpoint_profile(),
        "a checkpointed run reported no profile"
    );
    // Every span is a position inside the graph it was measured against, so a
    // span longer than the graph means the walk left the graph.
    assert!(wide.checkpoint_graph_nodes > 0);
    assert!(
        wide.checkpoint_max_span_nodes <= wide.checkpoint_graph_nodes,
        "a checkpoint outlived the graph: {} > {}",
        wide.checkpoint_max_span_nodes,
        wide.checkpoint_graph_nodes
    );
    // The simultaneous peak cannot exceed the total, and cannot be zero while
    // something is retained: both directions are one comparison in the sweep.
    assert!(wide.checkpoint_live_peak_bytes > 0);
    assert!(
        wide.checkpoint_live_peak_bytes <= wide.checkpoint_retained_bytes,
        "more bytes alive at once ({}) than retained in total ({})",
        wide.checkpoint_live_peak_bytes,
        wide.checkpoint_retained_bytes
    );
    // The long-lived subset is a subset.
    assert!(wide.checkpoint_long_lived_bytes <= wide.checkpoint_retained_bytes);
    assert!(wide.checkpoint_long_lived_count <= wide.checkpoint_count);

    let narrow = profile(true, CheckpointDtype::F16);
    assert_eq!(
        narrow.checkpoint_count, wide.checkpoint_count,
        "narrowing changed how many checkpoints are retained, which it must not"
    );
    // Held as 16-bit copies: about half, and definitely not the same or double.
    // The bound is loose because the checkpoint tensors need not all be F32,
    // what a wrong walk gets wrong is the factor, not the last percent.
    let ratio = narrow.checkpoint_retained_bytes as f64 / wide.checkpoint_retained_bytes as f64;
    assert!(
        (0.4..0.75).contains(&ratio),
        "a narrowed run retained {} of the F32 run's bytes; the walk is not \
         following the store/fetch chain",
        ratio
    );
}

#[test]
fn recompute_matches_the_fused_cross_entropy_path() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let reference = run(1, 1, false, true, Device::Cpu);
    let checkpointed = run(1, 1, true, true, Device::Cpu);
    assert!(
        (reference.0 - checkpointed.0).abs() <= LOSS_TOLERANCE,
        "fused loss {} != {}",
        reference.0,
        checkpointed.0
    );
    let max_gap = reference
        .1
        .iter()
        .zip(&checkpointed.1)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_gap <= LOGPROB_TOLERANCE, "score gap {max_gap}");
    if reference.3 {
        assert!(checkpointed.2 < reference.2);
    } else {
        assert_eq!(checkpointed.2, reference.2);
    }
}

#[cfg(retro_vulkan)]
#[test]
fn recompute_matches_differentiable_vulkan_attention() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no Vulkan device");
        return;
    }
    let _guard = common::serialize_models();

    let reference = run(1, 1, false, false, Device::Gpu);
    let checkpointed = run(1, 1, true, false, Device::Gpu);
    assert!((reference.0 - checkpointed.0).abs() <= 2.0e-3);
    let max_gap = reference
        .1
        .iter()
        .zip(&checkpointed.1)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_gap <= 5.0e-3, "Vulkan score gap {max_gap}");
    if reference.3 {
        assert!(checkpointed.2 < reference.2);
    } else {
        assert_eq!(checkpointed.2, reference.2);
    }
}
