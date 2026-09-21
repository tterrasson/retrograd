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
//! The download fixture is Q4_K_M, so only its F32 norms are eligible:
//! selections are validated against the selected tensors, not the model's
//! dominant dtype. It also ties its vocabulary projection, so the cases that
//! need the head run against the generated F32 fixture instead.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{
    CheckpointMetadata, Device, GefenLayout, GefenVariant, LoraConfig, LoraDtype, OptimizerKind,
    TargetSet, TensorDtype, TrainConfig, TrainableEntry, TrainablePolicy, TrainableRunConfig,
    TrainableSelector, TrainableSet, Trainer, resolve_base, tensor_inventory,
};

const TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

fn base_config(policy: TrainablePolicy, optimizer: OptimizerKind) -> TrainConfig {
    base_config_on(Device::Cpu, policy, optimizer)
}

/// [`base_config`] on a chosen device.
fn base_config_on(
    device: Device,
    policy: TrainablePolicy,
    optimizer: OptimizerKind,
) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        // Large enough that a norms-only update is visible in the scores, and
        // still small enough to stay finite over one step.
        learning_rate: 1.0e-3,
        device,
        trainable: TrainableRunConfig {
            policy,
            selector: TrainableSelector {
                norms: true,
                ..Default::default()
            },
            optimizer,
        },
        // The declared vector of whichever optimizer the case names, which is
        // what a built document would carry.
        optimizer_hyperparameters: optimizer.declared_hyperparameters(),
        ..TrainConfig::default()
    }
}

/// The norms of the fixture, resolved the way a run would resolve them.
fn resolved_norm_set(model: &Path, policy: TrainablePolicy) -> TrainableSet {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    let selector = TrainableSelector {
        norms: true,
        ..Default::default()
    };
    resolve_base(&inventory, policy, &selector).expect("a quantized fixture still has F32 norms")
}

fn resolved_norms(model: &Path) -> Vec<TrainableEntry> {
    resolved_norm_set(model, TrainablePolicy::Partial).entries
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

/// The generated F32 fixture: untied head, F32 throughout.
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

/// Exactly one training row, and therefore exactly one optimizer step.
///
/// One row is `n_ctx + 1` tokens: one context plus the label shift.
/// `train_once` feeds a whole text instead, which on this fixture's
/// byte-fallback vocabulary is eighteen steps, while the arithmetic cases
/// below compare against a cold optimizer's first update.
fn train_one_row(trainer: &mut Trainer, n_tokens: usize) -> u64 {
    let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    assert!(
        tokens.len() >= n_tokens,
        "a byte-fallback vocabulary produced only {} tokens",
        tokens.len()
    );
    tokens.truncate(n_tokens);
    trainer
        .train_tokens(&tokens)
        .expect("training run")
        .global_step
}

/// One training row of the generated fixture, whose effective context is its
/// declared `n_ctx_train`.
const TINY_ONE_ROW_TOKENS: usize = 257;

/// A run that selects the projection head, and nothing else.
fn head_config(chunked: bool) -> TrainConfig {
    let mut config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
    config.trainable.selector = TrainableSelector {
        output_head: true,
        ..Default::default()
    };
    config.chunked_cross_entropy = chunked;
    config
}

/// The head set of the generated fixture, resolved the way a run resolves it.
fn resolved_head_set(model: &Path) -> TrainableSet {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    resolve_base(
        &inventory,
        TrainablePolicy::Partial,
        &TrainableSelector {
            output_head: true,
            ..Default::default()
        },
    )
    .expect("an untied head resolves")
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

/// The preflight reports each selected family in the selector's vocabulary,
/// and the loss path it preflighted. The fixture ties its projection, so the
/// fused path is the only one it reaches.
#[test]
fn the_preflight_audits_each_selected_family_and_names_the_loss_path() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let selected = names(&resolved_norms(&model));

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load a trainer with writable weights");
    trainer
        .set_trainable_base(&selected)
        .expect("the resolver's names are the loader's names");

    let backend = trainer.backend_report().expect("backend report");
    assert!(backend.contains("loss_path: fused"), "{backend}");

    let preflight = trainer.train_preflight().expect("preflight the base graph");
    assert!(preflight.contains("loss_path: fused"), "{preflight}");
    assert!(preflight.contains("trainable_families:"), "{preflight}");
    assert!(
        preflight.contains(&format!(
            "norms: {} tensor(s), backward ready",
            selected.len()
        )),
        "{preflight}"
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
    assert!(record.optimizer.slots.is_empty());
    assert_eq!(record.optimizer.state_bytes, 0);
    assert!(record.optimizer.graph_ready);
    // ... and the parameters are still enumerated, each naming its optimizer.
    // A table that only listed parameters carrying state would be empty here,
    // and a mixed run could not tell "SGD owns it" from "it was not selected".
    assert!(!record.optimizer.assignment.is_empty());
    assert!(
        record
            .optimizer
            .assignment
            .iter()
            .all(|row| row.optimizer == "sgd" && row.layout_version == 1)
    );
    assert!(record.rng.runtime_mt19937.is_some());

    // The recorded vector is SGD's own: SGD declares no coefficients, so
    // AdamW's betas are absent.
    assert_eq!(record.optimizer.layout_version, 1);
    assert_eq!(
        record.optimizer.hyperparameters,
        vec![
            "learning_rate=0.001",
            "weight_decay=0.0",
            "max_grad_norm=1.0"
        ]
    );
    assert_eq!(record.optimizer.hyperparameter("beta1"), None);

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
    assert!(info.had_optimizer_graph);
    assert_eq!(info.restored_optimizer_slots, 0);
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
    // From the trainer, the same side that records it.
    let hyperparameters = trainer
        .optimizer_hyperparameters()
        .expect("optimizer hyperparameters");
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("fingerprint"),
        reference_fingerprint: trainer.reference_fingerprint().expect("anchor fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-base-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: config.learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: optimizer.into(),
        optimizer_layout_version: hyperparameters.optimizer().layout_version(),
        optimizer_hyperparameters: hyperparameters.lines(),
        weight_decay: config.weight_decay,
        max_grad_norm: config.max_grad_norm,
        trainable_policy: trainer.trainable_policy().as_str().to_string(),
        trainable_signature: trainer.trainable_signature().expect("trainable signature"),
    }
}

/// The kernel table, not the optimizer name, is what decides: the default
/// F16 adapter is refused for Muon at load time rather than by `GGML_ABORT`
/// in the middle of the first step. AdamW and SGD are not refused.
#[test]
fn an_f32_only_optimizer_refuses_an_adapter_its_update_step_cannot_write() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::Muon);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config).expect("load trainer");

    let mut f16 = f32_lora();
    f16.dtype = LoraDtype::F16;
    trainer.create_lora(&f16).expect("create an f16 adapter");
    let error = trainer
        .prepare_optimizer()
        .expect_err("the muon kernel carries no F16 path");
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

/// The other side of the same table: SGD's step rounds a half-precision
/// store, so the default F16 adapter is marked rather than refused.
#[test]
fn sgd_marks_the_default_f16_adapter() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::Sgd);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config).expect("load trainer");

    let mut f16 = f32_lora();
    f16.dtype = LoraDtype::F16;
    trainer.create_lora(&f16).expect("create an f16 adapter");
    trainer
        .prepare_optimizer()
        .expect("sgd writes an f16 adapter");
}

/// Every declared optimizer now has an update step, so the refusal that used
/// to greet Muon and Gefen before the model was opened is gone: what remains is
/// the model error, and the *device* refusal, which cannot be answered without
/// one. A name this build cannot honour would still be refused by `parse`.
#[test]
fn every_declared_optimizer_reaches_the_model_load() {
    for optimizer in [
        OptimizerKind::AdamW,
        OptimizerKind::Sgd,
        OptimizerKind::Muon,
        OptimizerKind::Gefen(GefenLayout::default()),
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::QuantizedM,
            ..GefenLayout::default()
        }),
    ] {
        assert!(optimizer.is_implemented(), "{optimizer}");
        let result = Trainer::new(
            "missing-model.gguf",
            base_config(TrainablePolicy::Lora, optimizer),
        );
        let error = result.err().expect("a missing model is still a failure");
        let message = error.to_string();
        assert!(
            !message.contains("not available"),
            "{optimizer} was refused for itself rather than for the model: {message}"
        );
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
            .with_reference_policy(|_| Ok(()))
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

/// The acceptance of the persistence half: train, save, resume, and land where
/// the run left off - not train alone.
///
/// A base run's result is the weights themselves, so the checkpoint carries a
/// trainable bundle of absolute values and no adapter, and a fresh trainer that
/// restores it reproduces the scores of the run that wrote it.
#[test]
fn a_partial_run_checkpoints_its_base_weights_and_resumes_where_it_stopped() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("partial-resume");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let selected = names(&set.entries);

    let config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    let step = train_once(&mut trainer);
    let state = root.join(format!("step-{step:012}.state"));
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("a base run publishes its weights");
    let saved = scores(&mut trainer);
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the base checkpoint");
    assert_eq!(record.manifest.trainable_policy, "partial");
    // No adapter, and no sibling GGUF either: the name every helper reads as
    // "the adapter" must not resolve to a file no adapter loader accepts.
    assert!(record.manifest.adapter.is_none());
    assert!(!state.join(checkpoint::ADAPTER_FILE).exists());
    assert!(!root.join(format!("step-{step:012}.gguf")).exists());

    let bundle = record.manifest.trainable.as_ref().expect("a base bundle");
    assert!(!bundle.signature.is_empty());
    assert_eq!(
        bundle.bytes,
        std::fs::metadata(state.join(&bundle.file)).unwrap().len()
    );
    assert_eq!(
        bundle.fingerprint,
        checkpoint::fingerprint_file(&state.join(&bundle.file)).unwrap()
    );
    let mut declared: Vec<&str> = bundle.tensors.iter().map(|t| t.name.as_str()).collect();
    declared.sort();
    let mut expected_names: Vec<&str> = selected.iter().map(String::as_str).collect();
    expected_names.sort();
    assert_eq!(declared, expected_names);
    assert!(
        bundle
            .tensors
            .iter()
            .all(|t| t.role == "base" && t.dtype == "F32")
    );

    // The restore is the assertion: a fresh trainer holds the fixture's own
    // weights until the bundle is applied, so identical scores mean the values
    // travelled rather than the counters.
    let mut resumed = Trainer::new(&model, config).expect("load a fresh trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let cold = scores(&mut resumed);
    assert!(
        deviation(&saved, &cold) > 0.0,
        "the fixture was already trained"
    );
    let expected = compatibility_for(&mut resumed, &model, "adamw");
    let info = resumed
        .load_checkpoint(&state, &expected)
        .expect("restore a base-weight checkpoint");
    assert_eq!(info.adapter, None);
    assert_eq!(info.trainable, Some(state.join(&bundle.file)));
    assert!(info.had_optimizer_graph);
    assert_eq!(info.restored_optimizer_slots, record.optimizer.slots.len());
    assert_eq!(deviation(&saved, &scores(&mut resumed)), 0.0);

    // And a different selection is refused rather than restored into: two
    // `partial` runs agree on the policy and can share nothing else.
    let mut narrowed = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load a third trainer");
    let mut narrower = set.clone();
    narrower.entries.truncate(1);
    narrowed
        .declare_trainable_set(&narrower)
        .expect("a narrower selection");
    let expected = compatibility_for(&mut narrowed, &model, "adamw");
    let error = narrowed
        .load_checkpoint(&state, &expected)
        .expect_err("a different resolved set is a different trajectory");
    assert!(error.to_string().contains("trainable set"), "{error}");
}

/// A hybrid run trains both halves, and its checkpoint carries both. The
/// adapter alone would be a file a loader cannot tell is incomplete.
#[test]
fn a_hybrid_checkpoint_carries_the_adapter_and_the_base_bundle() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("hybrid-checkpoint");
    let set = resolved_norm_set(&model, TrainablePolicy::Hybrid);
    let selected = names(&set.entries);

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the norms");
    trainer.create_lora(&f32_lora()).expect("create adapter");
    let step = train_once(&mut trainer);
    let state = root.join("checkpoint.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("a hybrid run publishes both halves");
    let saved = scores(&mut trainer);
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the hybrid checkpoint");
    assert_eq!(record.manifest.trainable_policy, "hybrid");
    assert!(record.manifest.adapter.is_some());
    assert!(record.manifest.trainable.is_some());
    assert!(state.join(checkpoint::ADAPTER_FILE).is_file());
    assert!(state.join(checkpoint::TRAINABLE_FILE).is_file());
    // The optimizer slots cover both families, which is what makes the
    // composite restore meaningful rather than an adapter reload beside it.
    let owners: Vec<&str> = record
        .optimizer
        .slots_in(checkpoint::SlotScope::Parameter)
        .map(|slot| slot.owner.as_str())
        .collect();
    assert!(owners.iter().any(|owner| owner.contains("lora")));
    assert!(
        owners
            .iter()
            .any(|owner| selected.iter().any(|name| name == owner))
    );

    let mut resumed = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load a fresh trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut resumed, &model, "adamw");
    let info = resumed
        .load_checkpoint(&state, &expected)
        .expect("restore both halves");
    assert!(info.adapter.is_some());
    assert!(info.trainable.is_some());
    assert_eq!(deviation(&saved, &scores(&mut resumed)), 0.0);
}

/// The composite export, reloaded. This proves the *published* pair carries
/// both halves: the bundle plus the adapter written beside it, loaded into a
/// trainer that never saw the run.
///
/// Each half is loaded alone on purpose: either reproduces neither the run nor
/// the base model, which is why an adapter-only export of a hybrid run is
/// refused rather than written.
#[test]
fn a_hybrid_composite_export_reloads_into_the_run_it_came_from() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("hybrid-export");
    let set = resolved_norm_set(&model, TrainablePolicy::Hybrid);

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the norms");
    trainer.create_lora(&f32_lora()).expect("create adapter");
    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(
        deviation(&cold, &trained) > 0.0,
        "the run has to move the model for the reload to mean anything"
    );

    // The `output.kind = "trainable"` shape for a policy with an adapter:
    // the bundle at the configured path, the adapter beside it.
    let bundle = root.join("result.gguf");
    let adapter = retrograd::run::trainable_adapter_sibling(&bundle);
    trainer.save_trainable(&bundle).expect("write the bundle");
    trainer.save_lora(&adapter).expect("write the adapter");
    drop(trainer);
    assert!(bundle.is_file() && adapter.is_file());

    let mut reloaded = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load a fresh trainer");
    reloaded
        .declare_trainable_set(&set)
        .expect("the same selection");
    // Base half alone: a bundle without its sibling is a different model.
    reloaded.load_trainable(&bundle).expect("load the bundle");
    let base_only = scores(&mut reloaded);
    assert!(deviation(&base_only, &trained) > 0.0);
    assert!(deviation(&base_only, &cold) > 0.0);

    reloaded.load_lora(&adapter).expect("load the adapter");
    assert_eq!(deviation(&trained, &scores(&mut reloaded)), 0.0);

    // The other half alone is equally incomplete: the adapter, into a trainer
    // whose norms are the base model's.
    let mut adapter_only = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load a fresh trainer");
    adapter_only
        .declare_trainable_set(&set)
        .expect("the same selection");
    adapter_only.load_lora(&adapter).expect("load the adapter");
    assert!(deviation(&scores(&mut adapter_only), &trained) > 0.0);
}

/// A bundle whose tensor list is not the run's resolved set is refused before
/// any weight moves: a partial restore resumes from a model that is neither the
/// checkpoint's nor the base's.
#[test]
fn a_trainable_bundle_must_match_the_run_it_is_restored_into() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("bundle-mismatch");
    let selected = names(&resolved_norms(&model));
    assert!(selected.len() > 1, "the fixture carries several norms");

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer.set_trainable_base(&selected).expect("select");
    train_once(&mut trainer);
    let bundle = root.join("trainable.gguf");
    trainer.save_trainable(&bundle).expect("write the bundle");
    drop(trainer);

    let mut narrowed = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    narrowed
        .set_trainable_base(&selected[..1])
        .expect("select fewer");
    let error = narrowed
        .load_trainable(&bundle)
        .expect_err("the bundle carries tensors this run does not train");
    assert!(error.to_string().contains("unexpected"), "{error}");

    // A LoRA run has no bundle to write at all.
    let mut lora_only = Trainer::new(
        &model,
        base_config(TrainablePolicy::Lora, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    lora_only.create_lora(&f32_lora()).expect("create adapter");
    let error = lora_only
        .save_trainable(root.join("empty.gguf"))
        .expect_err("a lora run trains no base tensor");
    assert!(error.to_string().contains("no base tensor"), "{error}");

    // And the bundle is not an adapter: loading it as one is refused rather
    // than producing an adapter of whatever happened to parse.
    let error = lora_only
        .load_lora(&bundle)
        .expect_err("a bundle is not an adapter");
    assert!(!error.to_string().is_empty());
}

/// What a frozen prefix costs. Same-size selections (one block's norms) at
/// different heights: the parameter, gradient and state halves are identical
/// by construction, so anything that differs is the backward graph.
/// `norms = true` cannot ask this, since the model-wide norms sit below every
/// block.
///
/// Only the shape is asserted: monotonic, and at least a factor of two between
/// the ends. The per-block coefficient is printed, not checked, since it is a
/// property of one architecture's block width.
fn assert_the_frozen_prefix_is_pruned(model: &Path, label: &str) {
    let inventory = tensor_inventory(model, Device::Cpu).expect("inventory");
    let last = inventory.n_layer - 1;
    assert!(last >= 3, "{label} is too shallow to have a prefix");

    // One measurement per height.
    let measure = |block: u32| -> (usize, u64, u64) {
        let mut config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
        config.trainable.selector = TrainableSelector {
            modules: vec![format!("blk.{block}.*norm.weight")],
            ..Default::default()
        };
        let set = resolve_base(
            &inventory,
            TrainablePolicy::Partial,
            &config.trainable.selector,
        )
        .expect("one block's norms resolve");
        let mut trainer = Trainer::new(model, config).expect("load trainer");
        trainer
            .set_trainable_base(&names(&set.entries))
            .expect("declare the set");
        trainer.prepare_optimizer().expect("build the graph");
        let report = trainer.memory_report().expect("memory report");
        (
            set.entries.len(),
            report.trainable_gradient_bytes,
            report.optimizer_compute_bytes,
        )
    };

    let (bottom_tensors, bottom_gradient, bottom_compute) = measure(0);
    let (top_tensors, top_gradient, top_compute) = measure(last);
    // Same work: if the tensors or gradients differ, the comparison below is
    // measuring the selection instead.
    assert_eq!(bottom_tensors, top_tensors, "{label}");
    assert_eq!(bottom_gradient, top_gradient, "{label}");
    assert!(
        top_compute * 2 < bottom_compute,
        "{label}: the frozen prefix was not pruned: {bottom_compute} bytes at \
         block 0, {top_compute} at block {last}"
    );

    // A middle selection costs strictly between the two ends.
    let middle = last / 2;
    let (_, _, middle_compute) = measure(middle);
    assert!(
        top_compute < middle_compute && middle_compute < bottom_compute,
        "{label}: compute is not monotonic in the lowest trainable block: \
         {bottom_compute} / {middle_compute} / {top_compute}"
    );

    let per_block = (bottom_compute - top_compute) as f64 / f64::from(last);
    eprintln!(
        "{label}: {} blocks, optimizer compute {bottom_compute} B at block 0, \
         {middle_compute} B at block {middle}, {top_compute} B at block {last} \
         - {per_block:.0} B per frozen block",
        inventory.n_layer
    );
}

/// On the download fixture, 14 blocks of a real width. Measured here: the
/// optimizer compute buffer falls from about 25 MB at block 0 to about 4.4 MB
/// at the last block, roughly linearly.
#[test]
fn the_backward_prunes_the_blocks_below_the_lowest_trainable_one() {
    let model = fixture!();
    let _guard = common::serialize_models();
    assert_the_frozen_prefix_is_pruned(&model, "lfm2 fixture");
}

/// The same check on a second architecture: a model whose prefix did not
/// prune would make the span multiplier in `retrograd_plan::cost` an
/// under-estimate.
#[test]
fn the_backward_prunes_the_prefix_on_a_second_architecture() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    assert_the_frozen_prefix_is_pruned(&model, "generated qwen2 fixture");
}

/// Whole-tensor reads because the fixture's tensors are small; the FFI
/// surface is byte ranges for the ones that are not.
fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_ne_bytes)
        .collect()
}

fn parameter_values(trainer: &mut Trainer, index: usize, n_bytes: u64) -> Vec<f32> {
    let mut bytes = vec![0_u8; usize::try_from(n_bytes).expect("a host-sized tensor")];
    trainer
        .read_marked_parameter(index, 0, &mut bytes)
        .expect("read a marked parameter");
    read_f32(&bytes)
}

fn parameter_gradient(trainer: &mut Trainer, index: usize) -> Vec<f32> {
    let info = trainer
        .parameter_gradient_info(index)
        .expect("describe a gradient");
    // F32 and parameter-shaped, whatever the parameter's own storage: both
    // are what the arithmetic below assumes.
    assert_eq!(info.dtype, TensorDtype::F32, "{}", info.name);
    assert_eq!(info.n_bytes, info.n_elements * 4, "{}", info.name);
    let mut bytes = vec![0_u8; usize::try_from(info.n_bytes).expect("a host-sized tensor")];
    trainer
        .read_parameter_gradient(index, 0, &mut bytes)
        .expect("read a gradient");
    read_f32(&bytes)
}

/// The update is the arithmetic it claims to be, checked on real weights and
/// a real gradient.
///
/// SGD is the interesting case: it keeps no persistent state, so without the
/// gradient its step could only be compared with itself. With the gradient
/// readable after the step, every input and the result are observable.
#[test]
fn an_sgd_step_is_the_arithmetic_it_claims_to_be() {
    let model = fixture!();
    let _guard = common::serialize_models();

    let mut config = base_config(TrainablePolicy::Partial, OptimizerKind::Sgd);
    // Non-zero, so the decay term is actually exercised.
    config.weight_decay = 0.1;
    // A ceiling no gradient reaches leaves the clipping scale at exactly one;
    // the norm is measured below rather than assumed.
    config.max_grad_norm = 1.0e9;

    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    trainer
        .prepare_optimizer()
        .expect("the marked set is the resolved set");

    let marked = trainer
        .marked_trainable_set()
        .expect("the marked parameters");
    assert!(!marked.entries.is_empty());
    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();

    // Exactly one optimizer step: the equation below relates one gradient to
    // one update.
    let tokens = trainer.tokenize_text(&TEXT.repeat(12)).expect("tokenize");
    let step = trainer.train_tokens(&tokens).expect("train").global_step;
    assert_eq!(step, 1, "the equation below describes exactly one step");

    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();
    let gradients: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_gradient(&mut trainer, index))
        .collect();

    // The norm over every trainable gradient is what the clipping scale is
    // computed from; under the ceiling, the scale is one.
    let norm = gradients
        .iter()
        .flatten()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.0, "the backward produced no gradient at all");
    assert!(norm < 1.0e9, "the gradient was clipped: {norm}");

    let knobs = trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    let scalar = |name: &str| match knobs.get(name) {
        Some(retrograd::HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    };
    let alpha = scalar("learning_rate");
    let decay = scalar("weight_decay");
    let keep = 1.0_f32 - alpha * decay;
    assert!(alpha > 0.0 && decay > 0.0 && keep < 1.0);

    let mut compared = 0_usize;
    let mut worst = 0.0_f32;
    // Each term is checked by dropping it: with small steps, an equation
    // satisfied by any small update would pass. Both rejections must fail the
    // tolerance the full expression passes.
    let mut without_gradient = 0.0_f32;
    let mut without_decay = 0.0_f32;
    for (index, entry) in marked.entries.iter().enumerate() {
        assert_eq!(before[index].len(), after[index].len(), "{}", entry.name);
        assert_eq!(
            before[index].len(),
            gradients[index].len(),
            "{}",
            entry.name
        );
        for ((w0, w1), g) in before[index]
            .iter()
            .zip(&after[index])
            .zip(&gradients[index])
        {
            let error =
                |expected: f32| (expected - w1).abs() / expected.abs().max(w1.abs()).max(1.0e-6);
            worst = worst.max(error(w0 * keep - alpha * g));
            without_gradient = without_gradient.max(error(w0 * keep));
            without_decay = without_decay.max(error(w0 - alpha * g));
            compared += 1;
        }
    }
    assert!(compared > 0);
    // Not bit-exact: the kernel may fuse the multiply and the subtract.
    const TOLERANCE: f32 = 1.0e-5;
    assert!(
        worst < TOLERANCE,
        "the step is not `w * (1 - lr * wd) - lr * g`: worst relative error {worst}"
    );
    assert!(
        without_gradient > TOLERANCE,
        "the gradient term changes nothing, so the comparison proves nothing"
    );
    assert!(
        without_decay > TOLERANCE,
        "the decay term changes nothing, so the comparison proves nothing"
    );
}

/// The standalone model export: the exported GGUF is the run, opened with the
/// exported path and nothing else. Both directions are asserted: reloading it
/// must match the trained run, and the source must still match the cold one.
#[test]
fn a_model_export_is_the_run_and_needs_nothing_beside_it() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("model-export");
    let config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(
        deviation(&cold, &trained) > 0.0,
        "the run has to move the model for the reload to mean anything"
    );

    let exported = root.join("trained-model.gguf");
    trainer.save_model(&exported).expect("write the model");
    drop(trainer);
    assert!(exported.is_file());
    // The source file is untouched; only its metadata is re-read.
    assert_ne!(
        checkpoint::fingerprint_file(&exported).expect("fingerprint the export"),
        checkpoint::fingerprint_file(&model).expect("fingerprint the fixture"),
    );

    // Opened as a model in its own right; the selection resolves against the
    // exported file's own inventory.
    let reloaded_set = resolved_norm_set(&exported, TrainablePolicy::Partial);
    assert_eq!(names(&reloaded_set.entries), names(&set.entries));
    let mut reloaded = Trainer::new(&exported, config.clone()).expect("load the exported model");
    reloaded
        .declare_trainable_set(&reloaded_set)
        .expect("the same selection");
    assert_eq!(deviation(&trained, &scores(&mut reloaded)), 0.0);
    drop(reloaded);

    // And it is not the model it was trained from.
    let mut source = Trainer::new(&model, config).expect("load the fixture again");
    source
        .declare_trainable_set(&set)
        .expect("the same selection");
    assert_eq!(deviation(&cold, &scores(&mut source)), 0.0);
}

/// `lora` and `hybrid` are refused: a standalone GGUF holds weights, and an
/// adapter is not in them.
#[test]
fn a_model_export_refuses_what_it_would_have_to_leave_out() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("model-export-refusals");

    let mut lora_only = Trainer::new(
        &model,
        base_config(TrainablePolicy::Lora, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    lora_only.create_lora(&f32_lora()).expect("create adapter");
    let error = lora_only
        .save_model(root.join("lora.gguf"))
        .expect_err("a lora run changes no weight");
    assert!(error.to_string().contains("no base tensor"), "{error}");

    let mut hybrid = Trainer::new(
        &model,
        base_config(TrainablePolicy::Hybrid, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    hybrid
        .declare_trainable_set(&resolved_norm_set(&model, TrainablePolicy::Hybrid))
        .expect("select the norms");
    hybrid.create_lora(&f32_lora()).expect("create adapter");
    let error = hybrid
        .save_model(root.join("hybrid.gguf"))
        .expect_err("the adapter would be dropped");
    assert!(error.to_string().contains("merging an adapter"), "{error}");
    assert!(!root.join("hybrid.gguf").exists());
}

/// AdamW's update, on the same fixture and with the same method as SGD's.
///
/// On a cold optimizer's first step the bias correction cancels: `m` is the
/// gradient and `sqrt(v)` its magnitude, so the step reduces to
/// `-lr * g/(|g| + eps)` on top of the decay.
#[test]
fn an_adamw_step_is_the_arithmetic_it_claims_to_be() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("adamw-arithmetic");

    let mut config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
    config.weight_decay = 0.1;
    config.max_grad_norm = 1.0e9;

    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    trainer
        .prepare_optimizer()
        .expect("the marked set is the resolved set");

    let marked = trainer
        .marked_trainable_set()
        .expect("the marked parameters");
    assert!(!marked.entries.is_empty());
    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();

    let tokens = trainer.tokenize_text(&TEXT.repeat(12)).expect("tokenize");
    let step = trainer.train_tokens(&tokens).expect("train").global_step;
    assert_eq!(step, 1, "the arithmetic below is the first step's");

    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();
    let gradients: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_gradient(&mut trainer, index))
        .collect();
    let norm = gradients
        .iter()
        .flatten()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.0, "the backward produced no gradient at all");
    assert!(norm < 1.0e9, "the gradient was clipped: {norm}");

    // The state the step kept, read back the way a resume reads it.
    let state = root.join("step.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("publish the state this step wrote");
    let knobs = trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the checkpoint");
    let mut reader = checkpoint::OptimizerStateReader::open(&state, &record.optimizer)
        .expect("open the payload");
    let mut slot_values = |slot: &checkpoint::StateSlot| -> Vec<f32> {
        let mut bytes = Vec::with_capacity(slot.n_bytes as usize);
        let mut staging = vec![0_u8; checkpoint::STAGING_CHUNK_BYTES];
        reader
            .stream(slot, &mut staging, |_, chunk| {
                bytes.extend_from_slice(chunk);
                Ok(())
            })
            .expect("read a slot payload");
        read_f32(&bytes)
    };

    let scalar = |name: &str| match knobs.get(name) {
        Some(retrograd::HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    };
    let alpha = scalar("learning_rate");
    let decay = scalar("weight_decay");
    let beta1 = scalar("beta1");
    let beta2 = scalar("beta2");
    let eps = scalar("eps");
    let keep = 1.0_f32 - alpha * decay;
    assert!(alpha > 0.0 && decay > 0.0 && keep < 1.0);
    assert!(beta1 > 0.0 && beta1 < 1.0 && beta2 > 0.0 && beta2 < 1.0 && eps > 0.0);

    let mut compared = 0_usize;
    let mut worst_moments = 0.0_f32;
    let mut worst = 0.0_f32;
    let mut without_gradient = 0.0_f32;
    let mut without_decay = 0.0_f32;
    let mut without_normalization = 0.0_f32;
    for (index, entry) in marked.entries.iter().enumerate() {
        let (m_slot, v_slot) = record
            .optimizer
            .adamw_moments(&entry.name)
            .expect("both moments of a marked parameter");
        let moment = slot_values(m_slot);
        let second = slot_values(v_slot);
        assert_eq!(moment.len(), before[index].len(), "{}", entry.name);
        assert_eq!(second.len(), before[index].len(), "{}", entry.name);

        for (position, ((w0, w1), g)) in before[index]
            .iter()
            .zip(&after[index])
            .zip(&gradients[index])
            .enumerate()
        {
            let error =
                |expected: f32| (expected - w1).abs() / expected.abs().max(w1.abs()).max(1.0e-6);
            // On a cold first step, m and v are the gradient and its square,
            // scaled by what the moving average keeps of a new sample.
            let relative = |expected: f32, actual: f32| {
                (expected - actual).abs() / expected.abs().max(actual.abs()).max(1.0e-12)
            };
            worst_moments = worst_moments
                .max(relative(g * (1.0 - beta1), moment[position]))
                .max(relative(g * g * (1.0 - beta2), second[position]));
            // Both bias corrections cancel, leaving a normalized step.
            worst = worst.max(error(w0 * keep - alpha * g / (g.abs() + eps)));
            without_gradient = without_gradient.max(error(w0 * keep));
            without_decay = without_decay.max(error(w0 - alpha * g / (g.abs() + eps)));
            // What separates this step from SGD's: AdamW steps along the
            // gradient's direction at a fixed size.
            without_normalization = without_normalization.max(error(w0 * keep - alpha * g));
            compared += 1;
        }
    }
    assert!(compared > 0);
    const TOLERANCE: f32 = 1.0e-5;
    assert!(
        worst_moments < 1.0e-4,
        "m and v are not the first step's moments: worst relative error {worst_moments}"
    );
    assert!(
        worst < TOLERANCE,
        "the step is not `w * (1 - lr * wd) - lr * g / (|g| + eps)`: worst relative error {worst}"
    );
    assert!(
        without_gradient > TOLERANCE,
        "the gradient term changes nothing, so the comparison proves nothing"
    );
    assert!(
        without_decay > TOLERANCE,
        "the decay term changes nothing, so the comparison proves nothing"
    );
    assert!(
        without_normalization > TOLERANCE,
        "the normalization changes nothing, so this passes for an SGD step"
    );
}

/// The dense loss path, executed. The fused backward asserts on a projection
/// that needs a gradient, inside the graph build: building the optimizer graph
/// is the gate, and the arithmetic below proves the head actually got a
/// gradient.
#[test]
fn a_run_that_trains_the_head_takes_the_dense_loss_path_and_moves_both_its_tensors() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let set = resolved_head_set(&model);
    let selected = names(&set.entries);
    assert!(
        selected.contains(&retrograd::OUTPUT_HEAD.to_string())
            && selected.contains(&retrograd::OUTPUT_HEAD_BIAS.to_string()),
        "rule 3 selects the head's weight and its bias together: {selected:?}"
    );

    // The option is on and loses to the selection; both reports say so.
    let mut trainer = Trainer::new(&model, head_config(true)).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("declare the head");
    let backend = trainer.backend_report().expect("backend report");
    assert!(
        backend.contains("chunked_cross_entropy: enabled"),
        "{backend}"
    );
    assert!(backend.contains("loss_path: dense"), "{backend}");
    assert!(
        backend.contains("loss_path_status: dense_fallback"),
        "{backend}"
    );
    let preflight = trainer.train_preflight().expect("preflight the base graph");
    assert!(preflight.contains("loss_path: dense"), "{preflight}");

    // The abort this case exists to rule out happens here, in the build.
    trainer
        .prepare_optimizer()
        .expect("the dense backward differentiates the head");

    let marked = trainer
        .marked_trainable_set()
        .expect("the marked parameters");
    let index_of = |name: &str| {
        marked
            .entries
            .iter()
            .position(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("'{name}' is not marked"))
    };
    let head = index_of(retrograd::OUTPUT_HEAD);
    let bias = index_of(retrograd::OUTPUT_HEAD_BIAS);
    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = [head, bias]
        .iter()
        .map(|index| parameter_values(&mut trainer, *index, sizes[*index]))
        .collect();

    train_once(&mut trainer);

    let after: Vec<Vec<f32>> = [head, bias]
        .iter()
        .map(|index| parameter_values(&mut trainer, *index, sizes[*index]))
        .collect();
    let gradients: Vec<Vec<f32>> = [head, bias]
        .iter()
        .map(|index| parameter_gradient(&mut trainer, *index))
        .collect();

    // Both halves move: a run where only one moved would mean the selection
    // and the graph disagree about what the head is.
    for (position, name) in [retrograd::OUTPUT_HEAD, retrograd::OUTPUT_HEAD_BIAS]
        .into_iter()
        .enumerate()
    {
        assert!(
            deviation(&before[position], &after[position]) > 0.0,
            "'{name}' did not move"
        );
        assert!(
            gradients[position].iter().any(|value| *value != 0.0),
            "'{name}' has an all-zero gradient, so the update was a no-op the \
             comparison above cannot distinguish from a fused-path run"
        );
        assert!(
            gradients[position].iter().all(|value| value.is_finite()),
            "'{name}' has a non-finite gradient"
        );
    }
}

/// The same selection with `chunked_cross_entropy` off: the same graph, and a
/// report that does not claim a fallback nothing fell back from.
#[test]
fn the_head_selection_decides_the_loss_path_whatever_the_option_says() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let set = resolved_head_set(&model);

    let mut trainer = Trainer::new(&model, head_config(false)).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("declare the head");
    let backend = trainer.backend_report().expect("backend report");
    assert!(
        backend.contains("chunked_cross_entropy: disabled"),
        "{backend}"
    );
    assert!(backend.contains("loss_path: dense"), "{backend}");
    assert!(
        !backend.contains("loss_path_status:"),
        "nothing fell back, so nothing should say so:\n{backend}"
    );
    trainer.prepare_optimizer().expect("build the dense graph");
}

/// A document that trains the head is priced against the dense buffer, not
/// the tiled one, asserted against the arithmetic rather than the fused
/// figure alone.
#[test]
fn the_planner_prices_the_dense_logits_buffer_for_a_head_training_document() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let info = retrograd::model_info(&model, Device::Cpu).expect("model info");
    let head = resolved_head_set(&model);
    let norms = resolved_norm_set(&model, TrainablePolicy::Partial);
    assert!(head.trains_loss_head() && !norms.trains_loss_head());

    let config = head_config(true);
    let workload = retrograd_plan::cost::Workload {
        kind: retrograd_plan::cost::WorkloadKind::Sft,
        examples: 1,
        co_resident_bytes: 0,
    };
    let estimate = |set: &TrainableSet| {
        retrograd_plan::cost::estimate(
            &info,
            &config,
            retrograd_plan::cost::Trainable::base(set),
            &workload,
            retrograd_plan::cost::Calibration::default(),
        )
    };

    // `[n_ubatch, n_vocab]` reserved outputs, plus the dense logits and their
    // gradient: the whole vocabulary, materialized.
    let n_ubatch = u64::from(config.n_ubatch.max(1));
    let n_vocab = info.n_vocab as u64;
    let dense = 3 * n_ubatch * n_vocab * 4;
    assert_eq!(estimate(&head).logits_bytes, dense);
    // ... and the same document, differing only in whether its selection
    // reaches the head, is priced against the tiled buffer.
    assert!(
        estimate(&norms).logits_bytes < dense,
        "the option is priced for a selection that does not train the head"
    );
}

/// The published base-only export, reloaded into a fresh trainer.
#[test]
fn a_published_base_bundle_reloads_into_a_trainer_that_never_saw_the_run() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("partial-export");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(
        deviation(&cold, &trained) > 0.0,
        "the run has to move the model for the reload to mean anything"
    );

    let bundle = root.join("result.gguf");
    trainer.save_trainable(&bundle).expect("write the bundle");
    drop(trainer);
    assert!(bundle.is_file());
    // No adapter beside it: this run has none.
    assert!(!retrograd::run::trainable_adapter_sibling(&bundle).exists());

    let mut reloaded = Trainer::new(&model, config).expect("load a fresh trainer");
    reloaded
        .declare_trainable_set(&set)
        .expect("the same selection");
    assert!(
        deviation(&cold, &scores(&mut reloaded)) == 0.0,
        "a fresh trainer already differs from the cold run"
    );
    reloaded.load_trainable(&bundle).expect("load the bundle");
    assert_eq!(
        deviation(&trained, &scores(&mut reloaded)),
        0.0,
        "the published bundle is not the run it came from"
    );
}

/// The model export on the generated fixture: what the `qwen2` row's
/// `exports_model = true` claims.
#[test]
fn a_model_export_of_the_generated_fixture_is_the_run_it_came_from() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("tiny-model-export");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);

    let inventory = tensor_inventory(&model, Device::Cpu).expect("inventory");
    assert!(
        retrograd::architecture_exports_model(&inventory.architecture),
        "the row for '{}' does not grant a model export",
        inventory.architecture
    );

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(deviation(&cold, &trained) > 0.0);

    let exported = root.join("trained-model.gguf");
    trainer.save_model(&exported).expect("write the model");
    drop(trainer);
    assert!(exported.is_file());

    let reloaded_set = resolved_norm_set(&exported, TrainablePolicy::Partial);
    assert_eq!(names(&reloaded_set.entries), names(&set.entries));
    let mut reloaded = Trainer::new(&exported, config.clone()).expect("load the exported model");
    reloaded
        .declare_trainable_set(&reloaded_set)
        .expect("the same selection");
    assert_eq!(deviation(&trained, &scores(&mut reloaded)), 0.0);
    drop(reloaded);

    let mut source = Trainer::new(&model, config).expect("load the fixture again");
    source
        .declare_trainable_set(&set)
        .expect("the same selection");
    assert_eq!(deviation(&cold, &scores(&mut source)), 0.0);
}

/// The same export, from a run whose weights live on the device: every weight
/// the writer emits must come back off the device.
///
/// Both trainers run on the device: scoring is not bit-identical across
/// backends, so the reload happens on the device too.
#[test]
fn a_model_export_of_a_device_resident_run_is_the_run_it_came_from() {
    let model = tiny_fixture!();
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();
    let root = scratch("tiny-model-export-gpu");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let config = base_config_on(Device::Gpu, TrainablePolicy::Partial, OptimizerKind::AdamW);

    let inventory = tensor_inventory(&model, Device::Gpu).expect("inventory");
    assert!(
        retrograd::architecture_exports_model(&inventory.architecture),
        "the row for '{}' does not grant a model export",
        inventory.architecture
    );

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    // The selected weights are on the device, which is the point of the case.
    let report = trainer.memory_report().expect("memory report");
    assert!(
        !report.base_trainable_on_host,
        "the selected base tensors did not land on the device: {report:?}"
    );

    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(
        deviation(&cold, &trained) > 0.0,
        "the run has to move the model for the export to mean anything"
    );

    let exported = root.join("trained-model.gguf");
    trainer.save_model(&exported).expect("write the model");
    drop(trainer);
    assert!(exported.is_file());

    let reloaded_set = resolved_norm_set(&exported, TrainablePolicy::Partial);
    assert_eq!(names(&reloaded_set.entries), names(&set.entries));
    let mut reloaded = Trainer::new(&exported, config.clone()).expect("load the exported model");
    reloaded
        .declare_trainable_set(&reloaded_set)
        .expect("the same selection");
    assert_eq!(
        deviation(&trained, &scores(&mut reloaded)),
        0.0,
        "the exported model is not the run it came from"
    );
    drop(reloaded);

    // The source file is untouched.
    let mut source = Trainer::new(&model, config).expect("load the fixture again");
    source
        .declare_trainable_set(&set)
        .expect("the same selection");
    assert_eq!(deviation(&cold, &scores(&mut source)), 0.0);
}

/// The trainable bundle from a device-resident run: a different writer from
/// the model export, so the device read is a separate claim.
#[test]
fn a_trainable_bundle_of_a_device_resident_run_is_the_run_it_came_from() {
    let model = tiny_fixture!();
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();
    let root = scratch("tiny-bundle-gpu");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let config = base_config_on(Device::Gpu, TrainablePolicy::Partial, OptimizerKind::AdamW);

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    let cold = scores(&mut trainer);
    train_once(&mut trainer);
    let trained = scores(&mut trainer);
    assert!(deviation(&cold, &trained) > 0.0);

    let bundle = root.join("result.gguf");
    trainer.save_trainable(&bundle).expect("write the bundle");
    drop(trainer);

    let mut reloaded = Trainer::new(&model, config).expect("load a fresh trainer");
    reloaded
        .declare_trainable_set(&set)
        .expect("the same selection");
    assert_eq!(deviation(&cold, &scores(&mut reloaded)), 0.0);
    reloaded.load_trainable(&bundle).expect("load the bundle");
    assert_eq!(
        deviation(&trained, &scores(&mut reloaded)),
        0.0,
        "the published bundle is not the run it came from"
    );
}

/// Two optimizers in one run, on a cold first step where both closed forms
/// are exact: AdamW's is `w * (1 - lr * wd) - lr * g / (|g| + eps)`, SGD's is
/// `w * (1 - lr * wd) - lr * g`. Each parameter has to *fail* the other
/// optimizer's, or the case would pass for a run that used one optimizer
/// twice.
#[test]
fn two_optimizers_side_by_side_each_keep_their_own_state_and_arithmetic() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();

    let mut config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);
    config.weight_decay = 0.1;
    config.max_grad_norm = 1.0e9;
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    assert!(set.entries.len() >= 4, "the fixture carries several norms");

    // Every other norm to SGD.
    let assignment: Vec<(String, OptimizerKind)> = set
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let optimizer = if index % 2 == 0 {
                OptimizerKind::Sgd
            } else {
                OptimizerKind::AdamW
            };
            (entry.name.clone(), optimizer)
        })
        .collect();

    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("select the fixture's norms");
    trainer
        .set_optimizer_assignment(&assignment)
        .expect("two implemented optimizers can coexist");
    trainer
        .prepare_optimizer()
        .expect("the marked set is the resolved set");

    let marked = trainer
        .marked_trainable_set()
        .expect("the marked parameters");
    let owner_of = |name: &str| {
        assignment
            .iter()
            .find(|(assigned, _)| assigned == name)
            .map(|(_, optimizer)| *optimizer)
            .expect("every marked parameter was assigned")
    };

    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();

    let step = train_one_row(&mut trainer, TINY_ONE_ROW_TOKENS);
    assert_eq!(step, 1, "the arithmetic below is the first step's");

    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();
    let gradients: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_gradient(&mut trainer, index))
        .collect();
    let norm = gradients
        .iter()
        .flatten()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.0 && norm < 1.0e9, "gradient norm {norm}");

    let knobs = trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    let scalar = |name: &str| match knobs.get(name) {
        Some(retrograd::HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    };
    let alpha = scalar("learning_rate");
    let decay = scalar("weight_decay");
    let eps = scalar("eps");
    let keep = 1.0_f32 - alpha * decay;

    // The live slot table: two rows per AdamW-owned parameter, none for the
    // SGD-owned ones.
    let slots = trainer.state_slots().expect("the live slot table");
    assert_eq!(
        trainer
            .memory_report()
            .expect("memory report")
            .optimizer_state_bytes,
        slots.iter().map(|slot| slot.n_bytes).sum::<u64>(),
        "the memory report must price each parameter's actual owner"
    );
    for entry in &marked.entries {
        let owned: Vec<&str> = slots
            .iter()
            .filter(|slot| {
                slot.scope == checkpoint::SlotScope::Parameter && slot.owner == entry.name
            })
            .map(|slot| slot.slot.as_str())
            .collect();
        match owner_of(&entry.name) {
            OptimizerKind::AdamW => assert_eq!(owned, ["m", "v"], "{}", entry.name),
            OptimizerKind::Sgd => assert!(owned.is_empty(), "{}: {owned:?}", entry.name),
            other => panic!("{other} was not assigned"),
        }
    }

    const TOLERANCE: f32 = 1.0e-5;
    let mut checked = [0_usize; 2];
    for (index, entry) in marked.entries.iter().enumerate() {
        let mut own = 0.0_f32;
        let mut other = 0.0_f32;
        for ((w0, w1), g) in before[index]
            .iter()
            .zip(&after[index])
            .zip(&gradients[index])
        {
            let error =
                |expected: f32| (expected - w1).abs() / expected.abs().max(w1.abs()).max(1.0e-6);
            let adamw = error(w0 * keep - alpha * g / (g.abs() + eps));
            let sgd = error(w0 * keep - alpha * g);
            let (mine, theirs) = match owner_of(&entry.name) {
                OptimizerKind::AdamW => (adamw, sgd),
                _ => (sgd, adamw),
            };
            own = own.max(mine);
            other = other.max(theirs);
        }
        assert!(
            own < TOLERANCE,
            "'{}' did not take its own optimizer's step: worst relative error {own}",
            entry.name
        );
        assert!(
            other > TOLERANCE,
            "'{}' is indistinguishable from the other optimizer's step, so the \
             comparison proves nothing",
            entry.name
        );
        checked[usize::from(owner_of(&entry.name) == OptimizerKind::Sgd)] += 1;
    }
    assert!(
        checked[0] > 0 && checked[1] > 0,
        "the run has to carry both optimizers: {checked:?}"
    );
}

#[test]
fn mixed_adapter_memory_follows_assignment_before_and_after_preparation() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    for default in [OptimizerKind::AdamW, OptimizerKind::Sgd] {
        let mut config = base_config(TrainablePolicy::Lora, default);
        config.trainable.selector = TrainableSelector::default();
        let mut trainer = Trainer::new(&model, config).expect("load trainer");
        let mut lora = f32_lora();
        lora.targets = TargetSet::Patterns(vec!["blk.0.attn_q.weight".to_string()]);
        trainer.create_lora(&lora).expect("create adapter");
        let original = trainer.memory_report().expect("initial memory");
        let override_kind = if default == OptimizerKind::AdamW {
            OptimizerKind::Sgd
        } else {
            OptimizerKind::AdamW
        };
        trainer
            .set_optimizer_assignment(&[("blk.0.attn_q.weight.lora_a".into(), override_kind)])
            .expect("override one factor");
        let declared = trainer.memory_report().expect("declared memory");
        trainer
            .prepare_optimizer()
            .expect("prepare mixed optimizer");
        let slots = trainer.state_slots().expect("live slots");
        let expected = slots.iter().map(|slot| slot.n_bytes).sum::<u64>();
        assert!(expected > 0);
        assert_eq!(declared.optimizer_state_bytes, expected);
        assert_eq!(
            trainer
                .memory_report()
                .expect("prepared memory")
                .optimizer_state_bytes,
            expected
        );
        assert_eq!(
            i128::from(declared.host_bytes) - i128::from(original.host_bytes),
            i128::from(expected) - i128::from(original.optimizer_state_bytes),
            "the CPU host budget must track the changed slots"
        );
    }
}

/// The mixed run's checkpoint, restored under the same assignment.
#[test]
fn a_mixed_run_checkpoints_both_owners_and_resumes_under_the_same_assignment() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("mixed-optimizers");
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);
    let config = base_config(TrainablePolicy::Partial, OptimizerKind::AdamW);

    let assignment: Vec<(String, OptimizerKind)> = set
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let optimizer = if index % 2 == 0 {
                OptimizerKind::Sgd
            } else {
                OptimizerKind::AdamW
            };
            (entry.name.clone(), optimizer)
        })
        .collect();

    let mut trainer = Trainer::new(&model, config.clone()).expect("load trainer");
    trainer.declare_trainable_set(&set).expect("select");
    trainer
        .set_optimizer_assignment(&assignment)
        .expect("declare the assignment");
    let step = train_once(&mut trainer);
    let state = root.join("mixed.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save a mixed checkpoint");
    let saved = scores(&mut trainer);
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the mixed checkpoint");
    let owners: Vec<&str> = record
        .optimizer
        .assignment
        .iter()
        .map(|row| row.optimizer.as_str())
        .collect();
    assert!(
        owners.contains(&"adamw") && owners.contains(&"sgd"),
        "{owners:?}"
    );
    assert!(
        record
            .optimizer
            .assignment
            .iter()
            .all(|row| row.layout_version == 1),
        "each row carries its own owner's layout, and both are 1 today"
    );
    // The state bytes are the AdamW half's alone.
    let adamw_rows = record
        .optimizer
        .assignment
        .iter()
        .filter(|row| row.optimizer == "adamw")
        .count();
    let slot_owners: Vec<&str> = record
        .optimizer
        .slots_in(checkpoint::SlotScope::Parameter)
        .map(|slot| slot.owner.as_str())
        .collect();
    assert_eq!(slot_owners.len(), adamw_rows * 2);

    let mut resumed = Trainer::new(&model, config.clone()).expect("load a fresh trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    resumed
        .set_optimizer_assignment(&assignment)
        .expect("the same assignment");
    let expected = compatibility_for(&mut resumed, &model, "adamw");
    resumed
        .load_checkpoint(&state, &expected)
        .expect("restore a mixed checkpoint");
    assert_eq!(deviation(&saved, &scores(&mut resumed)), 0.0);
    drop(resumed);

    // A different assignment is a different trajectory.
    let mut single = Trainer::new(&model, config).expect("load a third trainer");
    single
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut single, &model, "adamw");
    let error = single
        .load_checkpoint(&state, &expected)
        .expect_err("an all-AdamW run cannot resume a mixed one");
    assert!(error.to_string().contains("assignment"), "{error}");
}

/// An assignment row naming a parameter this run does not train is refused
/// once the marked set exists.
#[test]
fn an_assignment_row_that_names_no_trained_parameter_is_refused() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let set = resolved_norm_set(&model, TrainablePolicy::Partial);

    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer.declare_trainable_set(&set).expect("select");
    trainer
        .set_optimizer_assignment(&[("blk.0.attn_q.weight".to_string(), OptimizerKind::Sgd)])
        .expect("the row is validated against the marked set, which does not exist yet");
    let error = trainer
        .prepare_optimizer()
        .expect_err("the row names a tensor this run does not train");
    assert!(error.to_string().contains("attn_q"), "{error}");
    drop(trainer);

    // Every declared optimizer now has an update step, so a row naming one is
    // a row the allocator can honour; the refusal above is about the
    // *parameter*, which is the only thing an assignment can get wrong.
    let mut trainer = Trainer::new(
        &model,
        base_config(TrainablePolicy::Partial, OptimizerKind::AdamW),
    )
    .expect("load trainer");
    trainer.declare_trainable_set(&set).expect("select");
    trainer
        .set_optimizer_assignment(&[(set.entries[0].name.clone(), OptimizerKind::Muon)])
        .expect("a muon row is one the allocator can honour");
}

/// The dtype refusal follows the owner of each parameter: the run is AdamW,
/// but the refusal comes from the parameter assigned to Muon.
#[test]
fn the_dtype_refusal_follows_the_owner_of_each_parameter() {
    let model = tiny_fixture!();
    let _guard = common::serialize_models();
    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::AdamW);
    config.trainable.selector = TrainableSelector::default();

    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    let mut f16 = f32_lora();
    f16.dtype = LoraDtype::F16;
    f16.targets = TargetSet::Patterns(vec!["blk.0.attn_q.weight".to_string()]);
    trainer.create_lora(&f16).expect("create an f16 adapter");
    // Unassigned, the whole run is AdamW's and F16 is writable.
    trainer
        .prepare_optimizer()
        .expect("adamw writes an f16 adapter");
    drop(trainer);

    let mut config = base_config(TrainablePolicy::Lora, OptimizerKind::AdamW);
    config.trainable.selector = TrainableSelector::default();
    let mut trainer = Trainer::new(&model, config).expect("load trainer");
    trainer.create_lora(&f16).expect("create an f16 adapter");
    trainer
        .set_optimizer_assignment(&[(
            "blk.0.attn_q.weight.lora_a".to_string(),
            OptimizerKind::Muon,
        )])
        .expect("declare one factor onto muon");
    let error = trainer
        .prepare_optimizer()
        .expect_err("the muon kernel carries no F16 path");
    let message = error.to_string();
    assert!(message.contains("muon"), "{message}");
    assert!(message.contains("F32-only"), "{message}");
}
