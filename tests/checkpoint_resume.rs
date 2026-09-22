//! Complete training checkpoints: save the optimizer state next to a pure LoRA
//! GGUF and restore it into a fresh trainer, and continue the run from there
//! without a numerical seam - see `exact_numerical_continuity_after_a_resume`.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, ArtifactPolicy, Checkpoint};
use retrograd::{
    CheckpointMetadata, Device, LoraConfig, TargetSet, TrainConfig, TrainMetrics, Trainer,
};

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut config = LoraConfig::qv(2, 4.0);
    config.seed = 7;
    config.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    config
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("retrograd-ckpt-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

fn metadata(model: &Path, global_step: u64) -> CheckpointMetadata {
    CheckpointMetadata {
        checkpoint_id: format!("step-{global_step:012}"),
        algorithm: "sft".into(),
        trajectory_signature: "test-sft-v1".into(),
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
            fingerprint: checkpoint::fingerprint(TRAIN_TEXT.as_bytes()),
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

fn compatibility(trainer: &mut Trainer, model: &Path) -> checkpoint::Compatibility {
    // From the trainer: the descriptor declares more coefficients than a
    // document spells.
    let hyperparameters = trainer
        .optimizer_hyperparameters()
        .expect("optimizer hyperparameters");
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("model fingerprint"),
        reference_fingerprint: trainer.reference_fingerprint().expect("anchor fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-sft-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TRAIN_TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: config().learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: config().trainable.optimizer.to_string(),
        optimizer_layout_version: trainer
            .optimizer_layout_version()
            .expect("the layout version"),
        optimizer_hyperparameters: hyperparameters.lines(),
        weight_decay: config().weight_decay,
        max_grad_norm: config().max_grad_norm,
        trainable_policy: trainer.trainable_policy().as_str().to_string(),
        trainable_signature: trainer.trainable_signature().expect("trainable signature"),
    }
}

/// The payload bytes of one slot, read back through the streaming reader.
///
/// The checkpoint deliberately does not hand out a slot as a `Vec`: the whole
/// point of the separate payload file is that a restore never needs a host copy
/// of the optimizer state. A test that wants to compare values assembles its own.
fn slot_payload(
    state_dir: &Path,
    optimizer: &checkpoint::Optimizer,
    slot: &checkpoint::StateSlot,
) -> Vec<u8> {
    let mut reader =
        checkpoint::OptimizerStateReader::open(state_dir, optimizer).expect("open the payload");
    let mut staging = vec![0_u8; 64 * 1024];
    let mut out = Vec::with_capacity(slot.n_bytes as usize);
    reader
        .stream(slot, &mut staging, |_, chunk| {
            out.extend_from_slice(chunk);
            Ok(())
        })
        .expect("stream the slot");
    out
}

fn train_once(trainer: &mut Trainer) -> TrainMetrics {
    let tokens = trainer
        .tokenize_text(&TRAIN_TEXT.repeat(12))
        .expect("tokenize");
    trainer.train_tokens(&tokens).expect("training run")
}

/// Teacher-forced log-probabilities: the observable that says where a run
/// landed, independently of any internal counter.
fn scores(trainer: &mut Trainer) -> Vec<f32> {
    let tokens = trainer.tokenize_text(TRAIN_TEXT).expect("tokenize");
    trainer.score_tokens(&tokens).expect("score")
}

fn deviation(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn a_checkpoint_taken_before_the_first_step_declares_no_optimizer_state() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("cold");
    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");

    let state = root.join("step-000000000000.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, 0))
        .expect("save cold checkpoint");

    let record = Checkpoint::read(&state).expect("read cold checkpoint");
    // The slots are allocated by the first optimizer graph build, which has
    // not happened: the checkpoint must say so instead of inventing zeros.
    assert!(!record.optimizer.graph_ready);
    assert!(record.optimizer.slots.is_empty());
    assert_eq!(record.optimizer.state_bytes, 0);
    assert!(record.rng.runtime_mt19937.is_none());
    assert_eq!(record.optimizer.iter, 1);
    assert!(root.join("step-000000000000.gguf").is_file());
}

#[test]
fn a_checkpoint_after_a_step_carries_named_slots_that_restore_verbatim() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("moments");

    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let metrics = train_once(&mut trainer);
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, metrics.global_step))
        .expect("save warm checkpoint");
    drop(trainer);

    let saved = Checkpoint::read(&state).expect("read warm checkpoint");
    assert!(saved.optimizer.graph_ready);
    assert!(!saved.optimizer.slots.is_empty());
    // AdamW keeps two slots per parameter, and every LoRA a/b tensor is one.
    // The shared scope is empty and enumerated anyway: "this optimizer keeps
    // no shared state" is a fact, not a missing section.
    assert_eq!(
        saved
            .optimizer
            .slots_in(checkpoint::SlotScope::Shared)
            .count(),
        0
    );
    for slot in saved.optimizer.slots_in(checkpoint::SlotScope::Parameter) {
        assert!(slot.owner.contains("lora"), "{}", slot.owner);
        assert!(["m", "v"].contains(&slot.slot.as_str()), "{}", slot.slot);
        assert_eq!(slot.dtype, "f32", "{}", slot.owner);
        assert_eq!(
            slot.n_bytes as i64,
            slot.shape.iter().product::<i64>() * 4,
            "{} shape {:?}",
            slot.owner,
            slot.shape
        );
    }
    // Recorded layout and coefficients: three of the six are the run's own,
    // `beta1`, `beta2` and `eps` are the runtime's.
    assert_eq!(saved.optimizer.layout_version, 1);
    assert_eq!(saved.optimizer.hyperparameter("beta1"), Some("0.9"));
    assert_eq!(saved.optimizer.hyperparameter("beta2"), Some("0.999"));
    assert_eq!(saved.optimizer.hyperparameter("eps"), Some("1e-8"));
    assert_eq!(
        saved.optimizer.hyperparameter("learning_rate"),
        Some(config().learning_rate.to_string().as_str())
    );

    // Every marked parameter is named by the assignment table, including the
    // optimizer that owns it - the row a mixed run would compare on resume.
    assert!(!saved.optimizer.assignment.is_empty());
    assert!(
        saved
            .optimizer
            .assignment
            .iter()
            .all(|row| row.optimizer == "adamw")
    );
    assert_eq!(
        saved.optimizer.state_bytes,
        saved
            .optimizer
            .slots
            .iter()
            .map(|slot| slot.n_bytes)
            .sum::<u64>()
    );
    // A non-trivial optimizer state is what makes the restore meaningful.
    let second_moment = saved
        .optimizer
        .slots
        .iter()
        .find(|slot| slot.slot == "v")
        .expect("adamw keeps a second moment");
    assert!(
        slot_payload(&state, &saved.optimizer, second_moment)
            .iter()
            .any(|byte| *byte != 0)
    );
    assert!(saved.optimizer.iter > 1);

    // Restoring into a fresh trainer must reproduce the state bit for bit,
    // matched by parameter name rather than by graph order.
    let mut resumed = Trainer::new(&model, config()).expect("load trainer");
    let expected = compatibility(&mut resumed, &model);
    let info = resumed
        .load_checkpoint(&state, &expected)
        .expect("load checkpoint");
    assert!(info.had_optimizer_graph);
    assert_eq!(info.restored_optimizer_slots, saved.optimizer.slots.len());
    assert_eq!(info.global_step(), metrics.global_step);
    assert_eq!(info.epoch(), 1);

    let reread = root.join("roundtrip.state");
    let mut echo = metadata(&model, metrics.global_step);
    echo.checkpoint_id = "roundtrip".into();
    resumed
        .save_checkpoint(&reread, &echo)
        .expect("re-save the restored state");
    let after = Checkpoint::read(&reread).expect("read the re-saved checkpoint");
    assert_eq!(after.optimizer.iter, saved.optimizer.iter);
    assert_eq!(after.optimizer.state_bytes, saved.optimizer.state_bytes);
    for slot in &saved.optimizer.slots {
        let restored = after
            .optimizer
            .slots
            .iter()
            .find(|candidate| {
                candidate.scope == slot.scope
                    && candidate.owner == slot.owner
                    && candidate.slot == slot.slot
            })
            .unwrap_or_else(|| panic!("slot {} of {} survived the restore", slot.slot, slot.owner));
        assert_eq!(restored.shape, slot.shape);
        assert_eq!(restored.dtype, slot.dtype);
        assert_eq!(
            slot_payload(&reread, &after.optimizer, restored),
            slot_payload(&state, &saved.optimizer, slot),
            "{} {}",
            slot.owner,
            slot.slot
        );
    }
    assert_eq!(after.rng.runtime_mt19937, saved.rng.runtime_mt19937);
}

#[test]
fn a_checkpoint_gguf_holds_only_lora_tensors_and_loads_as_a_cold_adapter() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("gguf");

    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let metrics = train_once(&mut trainer);
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, metrics.global_step))
        .expect("save checkpoint");
    drop(trainer);

    // The GGUF is an ordinary adapter export: no optimizer state leaks into it,
    // and loading it alone is a cold adapter load, never a resume.
    let adapter = root.join("step-000000000001.gguf");
    let bytes = std::fs::read(&adapter).expect("read adapter gguf");
    for forbidden in ["AdamW", "manifest", "grad_m", "optimizer"] {
        assert!(
            !bytes
                .windows(forbidden.len())
                .any(|window| window == forbidden.as_bytes()),
            "the adapter GGUF must not mention {forbidden}"
        );
    }

    let mut cold = Trainer::new(&model, config()).expect("load trainer");
    cold.load_lora(&adapter).expect("load adapter gguf");
    let described = cold.describe_lora().expect("describe");
    assert!(described.contains("loaded_from_file: true"), "{described}");
}

/// Restoring the optimizer state must measurably change where the run lands:
/// a resumed run has to track the uninterrupted reference more closely than the
/// same reload with a cold optimizer does.
///
/// This is deliberately a *relative* assertion; the absolute one is
/// `exact_numerical_continuity_after_a_resume` below. Both failed until the
/// gradient accumulators stopped carrying every earlier step's gradients into
/// the next one, which the restored trainer had no way to reproduce.
#[test]
fn restoring_the_optimizer_state_tracks_the_uninterrupted_run_more_closely() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("continuity");

    // Reference: two consecutive training runs in one process.
    let mut straight = Trainer::new(&model, config()).expect("load trainer");
    straight.create_lora(&lora()).expect("create lora");
    train_once(&mut straight);
    train_once(&mut straight);
    let reference = scores(&mut straight);
    drop(straight);

    // One run, checkpointed.
    let mut first = Trainer::new(&model, config()).expect("load trainer");
    first.create_lora(&lora()).expect("create lora");
    let interrupted = train_once(&mut first);
    let state = root.join("step-000000000001.state");
    first
        .save_checkpoint(&state, &metadata(&model, interrupted.global_step))
        .expect("save checkpoint");
    drop(first);

    // Continued from the full checkpoint: adapter plus optimizer state.
    let mut warm = Trainer::new(&model, config()).expect("load trainer");
    let expected = compatibility(&mut warm, &model);
    warm.load_checkpoint(&state, &expected)
        .expect("load checkpoint");
    train_once(&mut warm);
    let warm_deviation = deviation(&scores(&mut warm), &reference);
    drop(warm);

    // Continued from the adapter alone: the same weights, a cold optimizer.
    let mut cold = Trainer::new(&model, config()).expect("load trainer");
    cold.load_lora(root.join("step-000000000001.gguf"))
        .expect("load adapter");
    train_once(&mut cold);
    let cold_deviation = deviation(&scores(&mut cold), &reference);

    assert!(
        warm_deviation < cold_deviation,
        "restoring the optimizer state must help: warm {warm_deviation}, cold {cold_deviation}"
    );
}

/// `N` steps, save, resume, `N` steps must match `2N` uninterrupted steps.
///
/// What used to break it was not in the checkpoint at all: `ggml_opt_alloc`
/// cleared the gradient accumulators through the *previous* evaluation's
/// graph, which the dynamic-graph path drops at the end of every
/// `ggml_opt_eval`. The clear therefore ran on a null graph and never
/// happened, so each optimizer step of an uninterrupted run consumed the sum
/// of its own gradients and of every window before it. A resumed trainer
/// starts from empty accumulators - live state no checkpoint carries - and so
/// took a different first step while agreeing on every value that was saved.
///
/// The bit-for-bit form of the same statement, on base weights and the
/// generated fixture, is
/// `an_interrupted_run_lands_bit_for_bit_where_an_uninterrupted_one_does` in
/// `tests/f16_base_training.rs`.
#[test]
fn exact_numerical_continuity_after_a_resume() {
    let Some(model) = common::model_path_if_available() else {
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("exact");

    let mut straight = Trainer::new(&model, config()).expect("load trainer");
    straight.create_lora(&lora()).expect("create lora");
    train_once(&mut straight);
    train_once(&mut straight);
    let reference = scores(&mut straight);
    drop(straight);

    let mut first = Trainer::new(&model, config()).expect("load trainer");
    first.create_lora(&lora()).expect("create lora");
    let interrupted = train_once(&mut first);
    let state = root.join("step-000000000001.state");
    first
        .save_checkpoint(&state, &metadata(&model, interrupted.global_step))
        .expect("save checkpoint");
    drop(first);

    let mut resumed = Trainer::new(&model, config()).expect("load trainer");
    let expected = compatibility(&mut resumed, &model);
    resumed
        .load_checkpoint(&state, &expected)
        .expect("load checkpoint");
    train_once(&mut resumed);

    let deviation = deviation(&scores(&mut resumed), &reference);
    assert!(
        deviation < 1.0e-3,
        "resumed run diverged from the uninterrupted one by {deviation}"
    );
}

#[test]
fn an_incompatible_run_is_refused_before_anything_is_written_into_the_trainer() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("refuse");

    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let metrics = train_once(&mut trainer);
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, metrics.global_step))
        .expect("save checkpoint");
    drop(trainer);

    let mut resumed = Trainer::new(&model, config()).expect("load trainer");
    let base = compatibility(&mut resumed, &model);

    let mut wrong_dataset = base.clone();
    wrong_dataset.dataset_fingerprint = checkpoint::fingerprint(b"other corpus");
    let error = resumed
        .load_checkpoint(&state, &wrong_dataset)
        .expect_err("a different dataset must be refused");
    assert!(error.to_string().contains("dataset"), "{error}");

    let mut wrong_lr = base.clone();
    wrong_lr.learning_rate = 1.0;
    let error = resumed
        .load_checkpoint(&state, &wrong_lr)
        .expect_err("a different learning rate must be refused");
    assert!(error.to_string().contains("schedule"), "{error}");

    // A coefficient no document spells, moved: everything else still agrees.
    let mut wrong_beta = base.clone();
    wrong_beta.optimizer_hyperparameters = wrong_beta
        .optimizer_hyperparameters
        .iter()
        .map(|line| {
            if line.starts_with("beta2=") {
                "beta2=0.95".to_string()
            } else {
                line.clone()
            }
        })
        .collect();
    let error = resumed
        .load_checkpoint(&state, &wrong_beta)
        .expect_err("a different second moment decay must be refused");
    assert!(error.to_string().contains("beta2=0.95"), "{error}");

    // A refusal is total: the adapter was never loaded, so the trainer is still
    // the blank one it was before.
    let described = resumed.describe_lora().expect("describe");
    assert!(described.contains("status: not initialized"), "{described}");

    // The same checkpoint still loads once the expectation matches.
    resumed
        .load_checkpoint(&state, &base)
        .expect("the compatible resume still works");
}

#[test]
fn an_interrupted_write_leaves_no_loadable_checkpoint() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("atomic");

    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let metrics = train_once(&mut trainer);
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, metrics.global_step))
        .expect("save checkpoint");

    // Truncating the state directory is what an interrupted write looks like
    // from the loader's side: the manifest is written last.
    std::fs::remove_file(state.join(checkpoint::MANIFEST_FILE)).expect("remove manifest");
    let mut resumed = Trainer::new(&model, config()).expect("load trainer");
    let expected = compatibility(&mut resumed, &model);
    let error = resumed
        .load_checkpoint(&state, &expected)
        .expect_err("a manifest-less directory must not load");
    assert!(error.to_string().contains("not a complete checkpoint"));
}

#[test]
fn an_artifact_travels_with_the_checkpoint_under_its_declared_policy() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: test model not available");
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("artifacts");

    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let metrics = train_once(&mut trainer);

    let mut with_artifacts = metadata(&model, metrics.global_step);
    with_artifacts.artifacts.insert(
        "metrics.msgpack".into(),
        (ArtifactPolicy::Recreatable, b"reward history".to_vec()),
    );
    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &with_artifacts)
        .expect("save checkpoint");
    drop(trainer);

    let mut resumed = Trainer::new(&model, config()).expect("load trainer");
    let expected = compatibility(&mut resumed, &model);
    let info = resumed
        .load_checkpoint(&state, &expected)
        .expect("load checkpoint");
    assert_eq!(
        info.artifacts.get("metrics.msgpack").map(Vec::as_slice),
        Some(b"reward history".as_slice())
    );
}
