//! The separate fixed reference: the anchor a run declares when its own frozen
//! weights can no longer serve as one.
//!
//! Covered properties:
//!
//! - A copy of the model reproduces the adapter-disabled scores exactly.
//! - A base-weight run is refused without an anchor and scores with one; its
//!   reference scores stay fixed across an update to the trained model.
//! - A different tokenizer is refused at attach time.
//! - The checkpoint records the anchor's fingerprint.
//!
//! The generated F32 fixture covers the low-level contracts. The downloaded
//! fixture supplies a different vocabulary and a chat template for distillation.

mod common;

use std::path::{Path, PathBuf};

use retrograd::checkpoint;
use retrograd::{
    Device, OptimizerKind, TrainConfig, TrainablePolicy, TrainableRunConfig, TrainableSelector,
    Trainer, resolve_base, tensor_inventory,
};

const TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
);

macro_rules! tiny_fixture {
    () => {
        match common::tiny_model_path_if_available() {
            Some(model) => model,
            None => {
                eprintln!("skipping: generated fixture not available");
                return;
            }
        }
    };
}

fn scratch(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("retrograd-reference-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

fn config(policy: TrainablePolicy) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        // Large enough that one step moves the scores well past any rounding.
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        trainable: TrainableRunConfig {
            policy,
            optimizer: OptimizerKind::AdamW,
            selector: TrainableSelector {
                norms: true,
                ..Default::default()
            },
        },
        ..TrainConfig::default()
    }
}

/// A second file with the same bytes, so the anchor is identified by content
/// rather than by path.
fn copy_of(model: &Path, root: &Path) -> PathBuf {
    let path = root.join("anchor.gguf");
    std::fs::copy(model, &path).expect("copy the fixture");
    path
}

fn tokens(trainer: &Trainer) -> Vec<i32> {
    trainer.tokenize_text(TEXT).expect("tokenize")
}

/// Declares the norm set of `model`, the way a run declares it.
fn declare_norms(trainer: &mut Trainer, model: &Path) {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    let set = resolve_base(
        &inventory,
        TrainablePolicy::Partial,
        &TrainableSelector {
            norms: true,
            ..Default::default()
        },
    )
    .expect("the norms resolve");
    trainer
        .declare_trainable_set(&set)
        .expect("declare the set");
}

/// A LoRA run can score through its own frozen weights or through a declared
/// anchor; with the same weights the two must give the same numbers.
#[test]
fn an_anchor_that_copies_the_model_reproduces_the_frozen_scores_exactly() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("exact-anchor");

    let mut trainer =
        Trainer::new(&model, config(TrainablePolicy::Lora)).expect("load the trained model");
    // The two answers being compared come from two different mechanisms, so
    // there must be an adapter to disable.
    trainer
        .create_lora(&retrograd::LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            seed: 7,
            targets: retrograd::TargetSet::Auto,
            dtype: retrograd::LoraDtype::F32,
        })
        .expect("create an adapter");
    let sequence = tokens(&trainer);
    // The reference a LoRA run has without declaring one.
    let frozen = trainer
        .score_reference_tokens(&sequence)
        .expect("a lora run scores against its own frozen weights");

    let anchor = copy_of(&model, &root);
    trainer
        .attach_reference(&anchor, &config(TrainablePolicy::Lora), None)
        .expect("a copy of the model is a valid anchor");
    assert_eq!(trainer.reference_path(), Some(anchor.as_path()));
    let declared = trainer
        .score_reference_tokens(&sequence)
        .expect("the declared anchor scores");

    assert_eq!(
        declared, frozen,
        "the same weights under two routes must produce one answer"
    );
}

/// The anchor's scores stay fixed across an update to the trained model.
#[test]
fn a_base_run_gains_a_reference_that_does_not_move_with_it() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("base-anchor");
    let anchor = copy_of(&model, &root);

    let mut trainer =
        Trainer::new(&model, config(TrainablePolicy::Partial)).expect("load the trained model");
    declare_norms(&mut trainer, &model);
    let sequence = tokens(&trainer);

    // Before the anchor: the refusal.
    let error = trainer
        .score_reference_tokens(&sequence)
        .expect_err("a base run has no frozen model of its own");
    assert!(
        error.to_string().contains("separate frozen model"),
        "{error}"
    );

    trainer
        .attach_reference(&anchor, &config(TrainablePolicy::Partial), None)
        .expect("the anchor is a separate frozen model");
    let before = trainer
        .score_reference_tokens(&sequence)
        .expect("the anchor scores a base-weight run");
    let trained_before = trainer.score_tokens(&sequence).expect("score the policy");

    trainer.prepare_optimizer().expect("build the update");
    let mut row = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    row.truncate(257);
    trainer.train_tokens(&row).expect("one optimizer step");

    let trained_after = trainer.score_tokens(&sequence).expect("score the policy");
    assert_ne!(
        trained_before, trained_after,
        "the step has to move the model for this test to mean anything"
    );
    let after = trainer
        .score_reference_tokens(&sequence)
        .expect("the anchor still scores");
    assert_eq!(
        before, after,
        "the anchor is fixed: an update to the trained weights cannot reach it"
    );
}

/// With an anchor, a base-weight run can sample from the reference policy;
/// without one, it is told so.
#[test]
fn reference_generation_routes_to_the_anchor_or_refuses() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("anchor-generation");
    let anchor = copy_of(&model, &root);

    let mut trainer =
        Trainer::new(&model, config(TrainablePolicy::Partial)).expect("load the trained model");
    declare_norms(&mut trainer, &model);
    let prompt = tokens(&trainer);
    let sampling = retrograd::SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 7,
    };

    let error = trainer
        .generate_base(&prompt[..4], &sampling)
        .expect_err("there is no frozen model to sample from");
    assert!(
        error.to_string().contains("separate frozen model"),
        "{error}"
    );

    trainer
        .attach_reference(&anchor, &config(TrainablePolicy::Partial), None)
        .expect("attach the anchor");
    let generated = trainer
        .generate_base(&prompt[..4], &sampling)
        .expect("the anchor samples");
    assert!(!generated.tokens.is_empty());
}

/// An anchor with a different tokenizer is refused at attach time.
#[test]
fn an_anchor_with_another_tokenizer_is_refused() {
    let model = tiny_fixture!();
    let other = match common::model_path_if_available() {
        Some(path) => path,
        None => {
            eprintln!("skipping: downloaded fixture not available");
            return;
        }
    };
    let _guard = common::serialize_models();

    let mut trainer =
        Trainer::new(&model, config(TrainablePolicy::Lora)).expect("load the trained model");
    let error = trainer
        .attach_reference(&other, &config(TrainablePolicy::Lora), None)
        .expect_err("two vocabularies are not one anchor");
    let message = error.to_string();
    assert!(message.contains("tokenizer"), "{message}");
    assert!(trainer.reference_path().is_none(), "nothing was attached");

    // And a missing path is a configuration error, not a load failure an hour
    // in.
    let error = trainer
        .attach_reference(
            "tests/fixtures/there-is-no-such-anchor.gguf",
            &config(TrainablePolicy::Lora),
            None,
        )
        .expect_err("an absent anchor is refused");
    assert!(
        error.to_string().contains("reference model not found"),
        "{error}"
    );
}

/// The anchor is part of what a resume must find unchanged.
#[test]
fn a_checkpoint_records_the_anchor_it_was_taken_against() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("anchor-checkpoint");
    let anchor = copy_of(&model, &root);

    let mut trainer =
        Trainer::new(&model, config(TrainablePolicy::Lora)).expect("load the trained model");
    assert_eq!(
        trainer.reference_fingerprint().expect("no anchor, no hash"),
        "",
        "a run with no anchor records none"
    );
    trainer
        .attach_reference(&anchor, &config(TrainablePolicy::Lora), None)
        .expect("attach the anchor");
    let fingerprint = trainer.reference_fingerprint().expect("fingerprint");
    assert_eq!(
        fingerprint,
        checkpoint::fingerprint_file(&anchor).expect("fingerprint the anchor"),
        "the recorded hash is the anchor file's own"
    );
    assert!(!fingerprint.is_empty());
}

#[test]
fn distillation_with_a_kl_term_scores_the_attached_reference() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: downloaded fixture not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("distill-anchor");
    let prompts = root.join("prompts.txt");
    std::fs::write(
        &prompts,
        concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox\"}]}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Pack my box\"}]}\n",
        ),
    )
    .expect("write prompts");
    let training = TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 2,
        ..config(TrainablePolicy::Partial)
    };
    let distill = retrograd::config::DistillConfig {
        mode: retrograd::config::DistillMode::OnPolicy,
        teacher_path: model.clone(),
        prompts,
        updates: 1,
        prompts_per_update: 2,
        samples_per_prompt: 2,
        distill_epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        weight_clip: 5.0,
        kl_coefficient: 0.1,
        mask_truncated: false,
        prompt_order: retrograd::config::PromptOrder::Sequential,
        sampling: retrograd::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 4,
            seed: 7,
        },
    };
    let mut trainer = Trainer::new(&model, training.clone()).expect("load policy");
    declare_norms(&mut trainer, &model);
    let error = retrograd::training::distill::run(&mut trainer, &distill, &training, &mut |_| {})
        .expect_err("the KL term must request a frozen reference");
    assert!(
        error.to_string().contains("separate frozen model"),
        "{error}"
    );
    trainer
        .attach_reference(&model, &training, None)
        .expect("attach anchor");
    let metrics = retrograd::training::distill::run(&mut trainer, &distill, &training, &mut |_| {})
        .expect("KL scores align with the live members of both prompt groups");
    assert!(metrics.global_step > 0);
}

/// What a *quantized* anchor costs in accuracy.
///
/// A run's KL term scores against a frozen reference, and nothing says that
/// reference has to be stored at the precision the policy is: a Q8_0 anchor is
/// a quarter of the weights on the device. Whether its scores are close enough
/// to an F32 anchor's is a measurement, and it needs two files holding *one*
/// model - which is what the generated fixtures are, the same numbers snapped
/// to the F16 grid and then stored three ways.
///
/// The comparison is on identical tokens through identical geometry, so the
/// only difference between the two runs is the anchor's storage. The bound is
/// deliberately on the *divergence of the scores*, not on a per-token
/// tolerance: a KL term reads a sum, and a handful of tokens disagreeing by
/// more than the rest is what a quantized anchor actually does.
#[test]
fn a_quantized_anchor_scores_within_a_measured_distance_of_its_f32_twin() {
    let model = tiny_fixture!();
    let Some(quantized) = common::tiny_q8_model_path_if_available() else {
        eprintln!("skipping: the Q8_0 fixture is not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("quantized-anchor");

    let mut trainer = Trainer::new(&model, config(TrainablePolicy::Partial)).expect("load trainer");
    declare_norms(&mut trainer, &model);
    let ids = tokens(&trainer);

    // The F32 anchor is a copy of this run's own file, so it is the control:
    // the only reason its scores could differ from the quantized anchor's is
    // the quantization.
    trainer
        .attach_reference(
            copy_of(&model, &root),
            &config(TrainablePolicy::Partial),
            None,
        )
        .expect("attach the F32 anchor");
    let exact = trainer
        .score_reference_tokens(&ids)
        .expect("the F32 anchor scores");

    trainer
        .attach_reference(&quantized, &config(TrainablePolicy::Partial), None)
        .expect("attach the Q8_0 anchor");
    let approximate = trainer
        .score_reference_tokens(&ids)
        .expect("the Q8_0 anchor scores");

    assert_eq!(
        exact.len(),
        approximate.len(),
        "two anchors over one token sequence score the same positions"
    );
    assert!(!exact.is_empty(), "the fixture scores at least one token");

    // Both are log-probabilities whatever the storage: a quantized anchor that
    // stopped producing them would be a different failure from an inaccurate
    // one, and it is worth separating.
    assert!(
        approximate
            .iter()
            .all(|score| score.is_finite() && *score <= 0.0),
        "a Q8_0 anchor still scores log-probabilities"
    );

    let mean_absolute = exact
        .iter()
        .zip(&approximate)
        .map(|(left, right)| f64::from(*left - *right).abs())
        .sum::<f64>()
        / exact.len() as f64;
    let worst = exact
        .iter()
        .zip(&approximate)
        .map(|(left, right)| f64::from(*left - *right).abs())
        .fold(0.0_f64, f64::max);
    // The quantity a KL term actually reads: the sum of the per-token
    // differences, which is what the penalty is built out of.
    let total = (exact.iter().map(|score| f64::from(*score)).sum::<f64>()
        - approximate
            .iter()
            .map(|score| f64::from(*score))
            .sum::<f64>())
    .abs();

    eprintln!(
        "quantized anchor over {} tokens: mean |delta| {mean_absolute:.6}, \
         worst {worst:.6}, |sum delta| {total:.6}",
        exact.len()
    );

    // Measured on this fixture - mean 0.005 nats per token, worst 0.017 - and
    // bounded an order of magnitude above that. The bounds are a ceiling rather
    // than a claim: what they pin is that a Q8_0 anchor is a perturbation of
    // its F32 twin and not a different model. A run that needs a tighter bound
    // has to measure its own pair.
    assert!(
        mean_absolute < 0.05,
        "a Q8_0 anchor of the same model diverges by {mean_absolute} per token on average"
    );
    assert!(
        worst < 0.2,
        "a Q8_0 anchor of the same model diverges by {worst} on its worst token"
    );

    // The control, asserted last so a vacuous pass is impossible: the two
    // anchors are *not* the same file, and the scores must not be identical.
    // If they were, the run would be scoring against the same weights twice and
    // the measurement above would mean nothing.
    assert!(
        exact
            .iter()
            .zip(&approximate)
            .any(|(left, right)| left != right),
        "the two anchors produced identical scores, so the quantized one was never read"
    );

    let _ = std::fs::remove_dir_all(&root);
}
