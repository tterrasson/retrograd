//! Base-weight training and the optimizer choice, against a real GGUF.
//!
//! Two properties the unit tests cannot reach, because both are about what the
//! *runtime* does with an answer resolved outside it:
//!
//! - A resolved base set is marked, carries a gradient, and moves the model -
//!   on a fixture whose weights are mapped read-only for every other run, so a
//!   write that was not preceded by the writable-buffer load would fault rather
//!   than drift. The source GGUF's bytes are checked afterwards: owning the
//!   buffers is what keeps the file out of the update.
//! - The optimizer named by the configuration is the one the graph builds.
//!   SGD is the case worth running: its kernels exist on every backend, it
//!   keeps no per-parameter state at all, and "keeps none" must not be read by
//!   a resume as "was never initialized".
//!
//! The fixture is Q4_K_M, so only its F32 norms are eligible - which is the
//! selection §2 of the contract exists for: validate the selected tensors, not
//! the model's dominant dtype.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{
    CheckpointMetadata, Device, LoraConfig, LoraDtype, OptimizerKind, TargetSet, TrainConfig,
    TrainableEntry, TrainablePolicy, TrainableRunConfig, TrainableSelector, Trainer, resolve_base,
    tensor_inventory,
};

const TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

fn base_config(policy: TrainablePolicy, optimizer: OptimizerKind) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        // Large enough that a norms-only update is visible in the scores, and
        // still small enough to stay finite over one step.
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        trainable: TrainableRunConfig {
            policy,
            selector: TrainableSelector {
                norms: true,
                ..Default::default()
            },
            optimizer,
        },
        ..TrainConfig::default()
    }
}

/// The norms of the fixture, resolved the way a run would resolve them.
fn resolved_norms(model: &Path) -> Vec<TrainableEntry> {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    let selector = TrainableSelector {
        norms: true,
        ..Default::default()
    };
    let set = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
        .expect("a quantized fixture still has F32 norms");
    set.entries
}

fn names(entries: &[TrainableEntry]) -> Vec<String> {
    entries.iter().map(|entry| entry.name.clone()).collect()
}

/// SGD's update kernel is F32-only, and F16 is the default adapter storage, so
/// an SGD run has to say which one it wants. The refusal is asserted in
/// `sgd_refuses_an_adapter_its_update_step_cannot_write`.
fn f32_lora() -> LoraConfig {
    let mut lora = LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    lora.dtype = LoraDtype::F32;
    lora.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    lora
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("retrograd-base-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

fn scores(trainer: &mut Trainer) -> Vec<f32> {
    let tokens = trainer.tokenize_text(TEXT).expect("tokenize");
    trainer.score_tokens(&tokens).expect("score")
}

fn deviation(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn train_once(trainer: &mut Trainer) -> u64 {
    let tokens = trainer.tokenize_text(&TEXT.repeat(12)).expect("tokenize");
    trainer
        .train_tokens(&tokens)
        .expect("training run")
        .global_step
}

macro_rules! fixture {
    () => {
        match common::model_path_if_available() {
            Some(model) => model,
            None => {
                eprintln!("skipping: test model not available");
                return;
            }
        }
    };
}

#[test]
fn a_resolved_norm_set_is_marked_and_moves_the_model_without_touching_the_gguf() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let before_file = checkpoint::fingerprint_file(&model).expect("fingerprint the fixture");

    let entries = resolved_norms(&model);
    assert!(!entries.is_empty());
    let selected = names(&entries);

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load a trainer with writable weights");
    // The mapping a LoRA run gets is PROT_READ, so a base update would fault on
    // it rather than drift. The load mode is invisible in every other field,
    // which is why the report names it.
    let report_text = trainer.backend_report().expect("backend report");
    assert!(
        report_text.contains("weight_storage: owned"),
        "the weights are still mapped read-only:\n{report_text}"
    );
    assert!(
        report_text.contains("trainable_policy: partial"),
        "{report_text}"
    );
    trainer
        .set_trainable_base(&selected)
        .expect("the resolver's names are the loader's names");

    trainer
        .train_preflight()
        .expect("preflight supports base weights");

    // Rule 6: the graph build compares what it marked with what was resolved,
    // and fails here rather than by training a quietly smaller set.
    trainer
        .prepare_optimizer()
        .expect("the marked set is the resolved set");

    let report = trainer.memory_report().expect("memory report");
    // Base weights are a slice of the loaded model, never an allocation on top.
    assert!(report.trainable_parameters_are_model_subset);
    assert!(report.trainable_parameter_bytes > 0);
    assert!(report.base_trainable_on_host);
    assert_eq!(
        report.device_bytes, 0,
        "CPU training must not consume a GPU budget"
    );
    assert_eq!(
        report.trainable_gradient_bytes,
        entries.iter().map(|entry| entry.n_elements).sum::<u64>() * 4
    );
    assert_eq!(
        report.optimizer_state_bytes,
        report.trainable_gradient_bytes * 2
    );

    let before = scores(&mut trainer);
    train_once(&mut trainer);
    let after = scores(&mut trainer);
    assert!(
        deviation(&before, &after) > 1.0e-4,
        "a norms-only update left the model where it was"
    );
    drop(trainer);

    // The mapping this run replaced is the one that made the file safe. With
    // owned buffers the update lands in memory and the GGUF is still the GGUF.
    assert_eq!(
        before_file,
        checkpoint::fingerprint_file(&model).expect("fingerprint the fixture again"),
        "base training rewrote the source model"
    );
}

#[test]
fn a_base_policy_refuses_to_train_before_its_set_is_declared() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    let error = trainer
        .train_preflight()
        .expect_err("preflight needs a declared set");
    assert!(error.to_string().contains("trainable set"), "{error}");
    let error = trainer
        .prepare_optimizer()
        .expect_err("an undeclared set is not an empty one");
    assert!(error.to_string().contains("trainable set"), "{error}");
}

#[test]
fn a_name_the_model_does_not_carry_is_refused_at_the_boundary() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    let error = trainer
        .set_trainable_base(&["blk.0.attn_nope.weight".to_string()])
        .expect_err("the runtime validates the resolver's answer against the model");
    assert!(error.to_string().contains("attn_nope"), "{error}");
}

#[test]
fn a_lora_run_takes_no_base_set() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Lora, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    let error = trainer
        .set_trainable_base(&["output_norm.weight".to_string()])
        .expect_err("a lora policy trains no base tensor");
    assert!(error.to_string().contains("no base tensor"), "{error}");
    // The counterpart of the base run's assertion: a LoRA run keeps the
    // read-only mapping, which is what makes it safe to share the page cache.
    let report_text = trainer.backend_report().expect("backend report");
    assert!(
        report_text.contains("weight_storage: mapped"),
        "a lora run paid for owned weight buffers:\n{report_text}"
    );
    // ... and the empty declaration every caller may make is not an error.
    trainer.set_trainable_base(&[]).expect("nothing to declare");
}

#[test]
fn sgd_is_the_optimizer_the_graph_builds_and_the_checkpoint_records() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("sgd");

    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::Sgd);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    trainer.create_lora(&f32_lora()).expect("create lora");

    let before = scores(&mut trainer);
    let step = train_once(&mut trainer);
    let after = scores(&mut trainer);
    assert!(
        deviation(&before, &after) > 0.0,
        "an SGD step left the adapter where it was"
    );

    let state = root.join(format!("step-{step:012}.state"));
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save an SGD checkpoint");
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the SGD checkpoint");
    let mut reader = Trainer::new(
        &model,
        base_config(TrainablePolicy::Lora, OptimizerKind::Sgd),
    )
    .expect("a second trainer, only to sign the model");
    assert_eq!(record.optimizer.kind, "sgd");
    // Zero slots, and initialized all the same. A reader that concluded "no
    // moments, so nothing ran" would restart the schedule from zero.
    assert!(!record.optimizer.has_moments);
    assert!(record.optimizer.moments.is_empty());
    assert!(record.optimizer.graph_ready);
    assert!(record.optimizer.graph_was_ready());
    assert!(record.rng.runtime_mt19937.is_some());

    // And the trajectory it names is checked: AdamW cannot resume it.
    let mut expected = compatibility_for(&mut reader, &model, "sgd");
    record
        .check_compatible(&expected)
        .expect("an SGD run resumes its own checkpoint");
    expected.optimizer_kind = "adamw".into();
    let error = record
        .check_compatible(&expected)
        .expect_err("AdamW moments mean nothing to an SGD step");
    assert!(error.to_string().contains("optimizer was sgd"), "{error}");
}

#[test]
fn an_sgd_checkpoint_restores_its_schedule_into_a_fresh_trainer() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("sgd-resume");

    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::Sgd);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer.create_lora(&f32_lora()).expect("create lora");
    let step = train_once(&mut trainer);
    let state = root.join(format!("step-{step:012}.state"));
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save");
    let saved = scores(&mut trainer);
    drop(trainer);

    let saved_record = Checkpoint::read(&state).expect("read");
    assert!(saved_record.scheduler.step > 0);

    let mut resumed = Trainer::new(&model, config).expect("load a fresh trainer");
    let expected = compatibility_for(&mut resumed, &model, "sgd");
    let info = resumed
        .load_checkpoint(&state, &expected)
        .expect("restore a zero-slot optimizer");
    // The adapter came back, so the scores do too - the assertion that the
    // restore reached the weights and not only the counters.
    assert!(!info.had_moments);
    assert_eq!(deviation(&saved, &scores(&mut resumed)), 0.0);
    let restored_state = root.join("restored.state");
    resumed
        .save_checkpoint(&restored_state, &metadata(&model, step))
        .expect("save restored state");
    let restored = Checkpoint::read(&restored_state).expect("read restored state");
    assert_eq!(restored.optimizer, saved_record.optimizer);
    assert_eq!(restored.scheduler, saved_record.scheduler);
    assert_eq!(
        restored.rng.runtime_mt19937,
        saved_record.rng.runtime_mt19937
    );
}

fn metadata(model: &Path, global_step: u64) -> CheckpointMetadata {
    CheckpointMetadata {
        checkpoint_id: format!("step-{global_step:012}"),
        algorithm: "sft".into(),
        trajectory_signature: "test-base-v1".into(),
        resume_boundary: "epoch".into(),
        scheduler_kind: "constant".into(),
        warmup_steps: 0,
        progress: checkpoint::Progress {
            version: checkpoint::FORMAT_VERSION,
            epoch: 1,
            global_step,
            cursor: 0,
            algorithm: "sft".into(),
            phase: "train".into(),
            best_eval: None,
            stale_evaluations: 0,
            kl_multiplier: None,
        },
        dataset: checkpoint::Dataset {
            version: checkpoint::FORMAT_VERSION,
            path: "inline".into(),
            fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
            examples: 1,
            row_width: 32,
            format: "text".into(),
            permutation: Vec::new(),
            cursor: 0,
        },
        seeds: BTreeMap::from([("sampling".to_string(), 7_u64)]),
        artifacts: BTreeMap::new(),
        model_path: model.to_path_buf(),
    }
}

fn compatibility_for(
    trainer: &mut Trainer,
    model: &Path,
    optimizer: &str,
) -> checkpoint::Compatibility {
    let config = base_config(TrainablePolicy::Lora, OptimizerKind::Sgd);
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-base-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: config.learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: optimizer.into(),
        weight_decay: config.weight_decay,
        max_grad_norm: config.max_grad_norm,
    }
}

/// The kernel table, not the optimizer name, is what decides. An F16 adapter is
/// the default storage, so this is the ordinary way to reach for SGD - and the
/// alternative to this refusal is `GGML_ABORT` in the middle of the first step.
#[test]
fn sgd_refuses_an_adapter_its_update_step_cannot_write() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::Sgd);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config).expect("load trainer");

    let mut f16 = f32_lora();
    f16.dtype = LoraDtype::F16;
    trainer.create_lora(&f16).expect("create an f16 adapter");
    let error = trainer
        .prepare_optimizer()
        .expect_err("the sgd kernel carries no F16 path");
    let message = error.to_string();
    assert!(message.contains("F32-only"), "{message}");
    assert!(
        message.contains("lora_a") || message.contains("lora_b"),
        "{message}"
    );
    let retry = trainer
        .prepare_optimizer()
        .expect_err("retry must remain safe");
    assert!(retry.to_string().contains("F32-only"), "{retry}");
}

#[test]
fn unsupported_optimizers_are_refused_before_loading_a_model() {
    for optimizer in [OptimizerKind::Muon, OptimizerKind::Gefen] {
        let result = Trainer::new(
            "missing-model.gguf",
            base_config(TrainablePolicy::Lora, optimizer),
        );
        let error = result.err().expect("unsupported optimizer must be refused");
        assert!(error.to_string().contains("not available"), "{error}");
    }
}

#[test]
fn a_base_set_must_be_nonempty_and_unique() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    let error = trainer.set_trainable_base(&[]).expect_err("empty base set");
    assert!(error.to_string().contains("non-empty"), "{error}");
    let name = names(&resolved_norms(&model))[0].clone();
    let error = trainer
        .set_trainable_base(&[name.clone(), name])
        .expect_err("duplicate base tensor");
    assert!(error.to_string().contains("duplicate"), "{error}");
}

#[test]
fn the_policy_checks_whether_an_adapter_belongs_to_the_trainable_set() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let selected = names(&resolved_norms(&model));
    for policy in [TrainablePolicy::Hybrid, TrainablePolicy::Partial] {
        let mut trainer =
            Trainer::new(&model, base_config(policy, OptimizerKind::AdamW)).expect("load trainer");
        trainer.set_trainable_base(&selected).expect("select norms");
        let error = trainer
            .with_lora_disabled(|_| Ok(()))
            .expect_err("disabling LoRA cannot freeze trainable base weights");
        assert!(
            error.to_string().contains("separate frozen model"),
            "{error}"
        );
        if policy == TrainablePolicy::Partial {
            trainer.create_lora(&f32_lora()).expect("create adapter");
        }
        for _ in 0..2 {
            let error = trainer
                .prepare_optimizer()
                .expect_err("adapter and policy disagree");
            assert!(error.to_string().contains("hybrid"), "{error}");
        }
        if policy == TrainablePolicy::Hybrid {
            trainer
                .create_lora(&f32_lora())
                .expect("supply the missing adapter");
            trainer
                .prepare_optimizer()
                .expect("both parts can now train");
        }
    }
}

#[test]
fn a_hybrid_run_cannot_publish_an_incomplete_checkpoint() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer.create_lora(&f32_lora()).expect("create adapter");
    let root = scratch("hybrid-checkpoint");
    let state = root.join("checkpoint.state");
    let error = trainer
        .save_checkpoint(&state, &metadata(&model, 0))
        .expect_err("adapter-only checkpoint would lose the base weights");
    assert!(error.to_string().contains("base weights"), "{error}");
    assert!(!state.exists());
    let expected = compatibility_for(&mut trainer, &model, "adamw");
    let error = trainer
        .load_checkpoint(&state, &expected)
        .expect_err("base-weight restore is unsupported");
    assert!(error.to_string().contains("base weights"), "{error}");
}
