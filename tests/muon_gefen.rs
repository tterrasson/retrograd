//! Muon and fixed-block Gefen, against independent oracles.
//!
//! The update is checked the only way an update can be: every input observed -
//! the weights before, the gradient the backward actually produced, the
//! coefficients the step read - and the result recomputed here, in slow F64,
//! by an implementation that shares no code with the one being checked.
//!
//! - **Muon** is a graph of ordinary ops, so the oracle is the algorithm's
//!   own arithmetic: EMA momentum, Nesterov, five quintic Newton-Schulz
//!   iterations, orientation so rows <= columns, and the shape factor on the
//!   original logical dimensions. Square, wide and tall matrices are all in the
//!   selection, so all three orientation branches run in one step.
//! - **Gefen** is the same fixed-block algorithm written straight: per-block
//!   second moments, and under `quantized_m` a first moment that survives the
//!   step as one unsigned byte per element against a shared codebook.
//!
//! Approximation quality is deliberately *not* asserted here. A cosine against
//! AdamW is a statement about whether these optimizers are useful; what this
//! file decides is whether they compute what they say they compute.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::gefen::gefen_step;
use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{
    CheckpointMetadata, Device, GEFEN_CODEBOOK_LEVELS, GEFEN_ZERO_BLOCK_INDEX, GefenLayout,
    GefenVariant, HyperparameterValue, HyperparameterVector, OptimizerKind, TensorDtype,
    TrainConfig, TrainablePolicy, TrainableRunConfig, TrainableSelector, TrainableSet, Trainer,
    resolve_base, tensor_inventory,
};

const TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

/// One context plus the label shift: exactly one optimizer step, which is what
/// relates one gradient to one update.
const ONE_ROW_TOKENS: usize = 257;

/// A square matrix and two rectangular ones of opposite aspect, so Muon's
/// orientation branch is exercised in both directions and its identity branch
/// once. All three are hidden base matrices, which is what its eligibility
/// rule admits.
const MATRIX_MODULES: [&str; 3] = ["attn_q", "ffn_up", "ffn_down"];

macro_rules! fixture {
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

fn matrix_selector() -> TrainableSelector {
    TrainableSelector {
        modules: MATRIX_MODULES.iter().map(|m| (*m).to_string()).collect(),
        ..Default::default()
    }
}

/// The matrices plus every norm: the norms are one-dimensional, so Muon
/// declines them by role and Gefen by element count, and the run is mixed
/// without anyone declaring a mixed assignment.
fn mixed_selector() -> TrainableSelector {
    TrainableSelector {
        norms: true,
        ..matrix_selector()
    }
}

fn config(optimizer: OptimizerKind, selector: TrainableSelector) -> TrainConfig {
    TrainConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 64,
        epochs: 1,
        learning_rate: 1.0e-3,
        weight_decay: 0.0,
        // A ceiling no gradient of this fixture reaches, so the clipping scale
        // is exactly one; the norm is measured below rather than assumed.
        max_grad_norm: 1.0e9,
        device: Device::Cpu,
        trainable: TrainableRunConfig {
            policy: TrainablePolicy::Partial,
            selector,
            optimizer,
        },
        optimizer_hyperparameters: optimizer.declared_hyperparameters(),
        ..TrainConfig::default()
    }
}

fn resolved(model: &Path, selector: &TrainableSelector) -> TrainableSet {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    resolve_base(&inventory, TrainablePolicy::Partial, selector)
        .expect("the selected tensors resolve")
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("retrograd-opt-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

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
    assert_eq!(info.dtype, TensorDtype::F32, "{}", info.name);
    let mut bytes = vec![0_u8; usize::try_from(info.n_bytes).expect("a host-sized tensor")];
    trainer
        .read_parameter_gradient(index, 0, &mut bytes)
        .expect("read a gradient");
    read_f32(&bytes)
}

fn train_one_row(trainer: &mut Trainer) -> u64 {
    let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    assert!(tokens.len() >= ONE_ROW_TOKENS, "{} tokens", tokens.len());
    tokens.truncate(ONE_ROW_TOKENS);
    trainer
        .train_tokens(&tokens)
        .expect("training run")
        .global_step
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

/// The clipping scale the step applied, computed from the same norm the graph
/// computes it from: one norm over every trainable gradient.
fn clip_scale(gradients: &[Vec<f32>], max_grad_norm: f32) -> f32 {
    let norm = gradients
        .iter()
        .flatten()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    assert!(norm > 0.0, "the backward produced no gradient at all");
    (f64::from(max_grad_norm) / norm).clamp(0.0, 1.0) as f32
}

fn scalar(vector: &HyperparameterVector, name: &str) -> f32 {
    match vector.get(name) {
        Some(HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    }
}

// --- the oracles -------------------------------------------------------------

/// A dense matrix in the same terms the update is specified in: `rows` is
/// ggml's `ne[1]` and `cols` is `ne[0]`, which is the fastest axis.
#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols);
        Self { rows, cols, values }
    }

    fn at(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    fn transposed(&self) -> Self {
        let mut values = vec![0.0; self.values.len()];
        for row in 0..self.rows {
            for col in 0..self.cols {
                values[col * self.rows + row] = self.at(row, col);
            }
        }
        Self::new(self.cols, self.rows, values)
    }

    fn times(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut values = vec![0.0; self.rows * other.cols];
        for row in 0..self.rows {
            for inner in 0..self.cols {
                let left = self.at(row, inner);
                for col in 0..other.cols {
                    values[row * other.cols + col] += left * other.at(inner, col);
                }
            }
        }
        Self::new(self.rows, other.cols, values)
    }

    fn scaled(&self, factor: f64) -> Self {
        Self::new(
            self.rows,
            self.cols,
            self.values.iter().map(|value| value * factor).collect(),
        )
    }

    fn plus(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Self::new(
            self.rows,
            self.cols,
            self.values
                .iter()
                .zip(&other.values)
                .map(|(left, right)| left + right)
                .collect(),
        )
    }

    fn frobenius(&self) -> f64 {
        self.values
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }
}

/// The orthogonalized direction Muon's update is `-lr * scale *` times, from
/// the specification alone.
#[allow(clippy::too_many_arguments)]
fn muon_direction(
    momentum: &mut [f64],
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mu: f64,
    nesterov: bool,
    ns_steps: u32,
    ns_epsilon: f64,
) -> Matrix {
    for (state, value) in momentum.iter_mut().zip(gradient) {
        *state = mu * *state + (1.0 - mu) * value;
    }
    let combined: Vec<f64> = if nesterov {
        gradient
            .iter()
            .zip(momentum.iter())
            .map(|(g, m)| (1.0 - mu) * g + mu * m)
            .collect()
    } else {
        momentum.to_vec()
    };

    // Orientation: the Gram matrix below is rows x rows, so rows must be the
    // smaller dimension.
    let transposed = rows > cols;
    let oriented = if transposed {
        Matrix::new(rows, cols, combined).transposed()
    } else {
        Matrix::new(rows, cols, combined)
    };

    let mut x = oriented.scaled(1.0 / (oriented.frobenius() + ns_epsilon));
    for _ in 0..ns_steps {
        let a = x.times(&x.transposed());
        let a2 = a.times(&a);
        let p = a.scaled(-4.7750).plus(&a2.scaled(2.0315));
        x = x.scaled(3.4445).plus(&p.times(&x));
    }
    let undone = if transposed { x.transposed() } else { x };
    // The factor reads the *original* logical dimensions, so it does not
    // follow whichever orientation the iteration chose.
    undone.scaled((1.0_f64).max(rows as f64 / cols as f64).sqrt())
}

// --- Muon --------------------------------------------------------------------

/// The step is the arithmetic the specification names, on real weights and a
/// real gradient, over a square matrix and two rectangular ones.
#[test]
fn a_muon_step_is_the_orthogonalized_update_it_claims_to_be() {
    let model = fixture!();
    let _guard = common::serialize_models();

    let mut trainer =
        Trainer::new(&model, config(OptimizerKind::Muon, matrix_selector())).expect("load trainer");
    let set = resolved(&model, &matrix_selector());
    trainer.declare_trainable_set(&set).expect("select");
    trainer.prepare_optimizer().expect("build the muon graph");

    let marked = trainer.marked_trainable_set().expect("the marked set");
    assert!(!marked.entries.is_empty());
    // Every row is Muon's: the selection is matrices only, and the assignment
    // the trainer declared says so.
    let owners: Vec<OptimizerKind> = trainer
        .declared_optimizer_assignment()
        .iter()
        .map(|(_, optimizer)| *optimizer)
        .collect();
    assert!(
        owners.iter().all(|owner| *owner == OptimizerKind::Muon),
        "{owners:?}"
    );
    // One F32 momentum per parameter and nothing else.
    let slots = trainer.state_slots().expect("the live slot table");
    assert_eq!(slots.len(), marked.entries.len());
    assert!(
        slots
            .iter()
            .all(|slot| slot.slot == "momentum" && slot.dtype == "f32")
    );

    let shapes: Vec<(usize, usize)> = marked
        .entries
        .iter()
        .map(|entry| (entry.ne[1] as usize, entry.ne[0] as usize))
        .collect();
    assert!(
        shapes.iter().any(|(rows, cols)| rows == cols)
            && shapes.iter().any(|(rows, cols)| rows < cols)
            && shapes.iter().any(|(rows, cols)| rows > cols),
        "the selection must cover all three orientations: {shapes:?}"
    );

    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();
    assert_eq!(train_one_row(&mut trainer), 1, "exactly one step");
    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_values(&mut trainer, index, sizes[index]))
        .collect();
    let gradients: Vec<Vec<f32>> = (0..sizes.len())
        .map(|index| parameter_gradient(&mut trainer, index))
        .collect();

    let knobs = trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    assert_eq!(knobs.optimizer(), OptimizerKind::Muon);
    let alpha = f64::from(scalar(&knobs, "learning_rate"));
    let mu = f64::from(scalar(&knobs, "momentum"));
    let ns_epsilon = f64::from(scalar(&knobs, "ns_epsilon"));
    let decay = f64::from(scalar(&knobs, "weight_decay"));
    let ns_steps = match knobs.get("ns_steps") {
        Some(HyperparameterValue::Structural(value)) => value as u32,
        other => panic!("ns_steps is {other:?}"),
    };
    let nesterov = matches!(
        knobs.get("nesterov"),
        Some(HyperparameterValue::Toggle(true))
    );
    assert!(nesterov && ns_steps == 5, "the frozen v1 is what ran");

    let scale = f64::from(clip_scale(&gradients, 1.0e9));
    let keep = 1.0 - alpha * decay;

    let mut worst = 0.0_f64;
    let mut worst_without_orthogonalization = 0.0_f64;
    for (index, entry) in marked.entries.iter().enumerate() {
        let rows = entry.ne[1] as usize;
        let cols = entry.ne[0] as usize;
        let gradient: Vec<f64> = gradients[index]
            .iter()
            .map(|value| f64::from(*value) * scale)
            .collect();
        // The momentum starts at the declared zero, which is what the
        // initializer wrote before the first step.
        let mut momentum = vec![0.0; gradient.len()];
        let direction = muon_direction(
            &mut momentum,
            &gradient,
            rows,
            cols,
            mu,
            nesterov,
            ns_steps,
            ns_epsilon,
        );
        // The direction is normalized and then orthogonalized, so its scale is
        // O(1) whatever the gradient's was; the error is measured against it.
        let unit = direction
            .values
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1.0e-12);
        for (offset, (w0, w1)) in before[index].iter().zip(&after[index]).enumerate() {
            let expected = f64::from(*w0) * keep - alpha * direction.values[offset];
            worst = worst.max((expected - f64::from(*w1)).abs() / (alpha * unit));
            // The same equation with the raw momentum in place of the
            // orthogonalized direction. It has to fail, or the tolerance above
            // is satisfied by any small update rather than by this one.
            let naive = f64::from(*w0) * keep - alpha * momentum[offset];
            worst_without_orthogonalization = worst_without_orthogonalization
                .max((naive - f64::from(*w1)).abs() / (alpha * unit));
        }
    }
    assert!(
        worst < 2.0e-3,
        "the muon step is not the oracle's: relative error {worst}"
    );
    assert!(
        worst_without_orthogonalization > 1.0e-1,
        "an unorthogonalized update would have passed too: {worst_without_orthogonalization}"
    );
}

/// A run with matrices and norms is mixed without anyone saying so: Muon
/// declines a one-dimensional tensor by role, and the fallback owns it, keeps
/// AdamW's two slots for it and is priced at its own declared rate.
#[test]
fn muon_falls_back_by_role_and_the_two_tables_live_side_by_side() {
    let model = fixture!();
    let _guard = common::serialize_models();

    let mut trainer =
        Trainer::new(&model, config(OptimizerKind::Muon, mixed_selector())).expect("load trainer");
    let set = resolved(&model, &mixed_selector());
    trainer.declare_trainable_set(&set).expect("select");
    trainer.prepare_optimizer().expect("build a mixed graph");

    let assignment = trainer.declared_optimizer_assignment().to_vec();
    let muon = assignment
        .iter()
        .filter(|(_, owner)| *owner == OptimizerKind::Muon)
        .count();
    let adamw = assignment
        .iter()
        .filter(|(_, owner)| *owner == OptimizerKind::AdamW)
        .count();
    assert!(muon > 0 && adamw > 0, "{assignment:?}");

    let slots = trainer.state_slots().expect("the live slot table");
    assert_eq!(
        slots.iter().filter(|slot| slot.slot == "momentum").count(),
        muon
    );
    assert_eq!(slots.iter().filter(|slot| slot.slot == "m").count(), adamw);
    assert_eq!(slots.iter().filter(|slot| slot.slot == "v").count(), adamw);

    let before = scores(&mut trainer);
    train_one_row(&mut trainer);
    let after = scores(&mut trainer);
    assert!(deviation(&before, &after) > 0.0, "the step changed nothing");
    assert!(after.iter().all(|value| value.is_finite()));
}

/// The momentum survives a checkpoint, and the run continues from it exactly.
#[test]
fn a_muon_momentum_round_trips_and_the_resume_is_exact() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("muon-resume");

    let mut trainer =
        Trainer::new(&model, config(OptimizerKind::Muon, matrix_selector())).expect("load trainer");
    let set = resolved(&model, &matrix_selector());
    trainer.declare_trainable_set(&set).expect("select");
    let step = train_one_row(&mut trainer);
    let state = root.join("muon.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save a muon checkpoint");
    // The trajectory the resume has to reproduce: one more step from here.
    let continued = train_one_row(&mut trainer);
    let expected = scores(&mut trainer);
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the muon checkpoint");
    assert_eq!(record.optimizer.kind, "muon");
    assert_eq!(record.optimizer.state_bytes, 4 * total_elements(&set));
    assert!(
        record
            .optimizer
            .slots_in(checkpoint::SlotScope::Parameter)
            .all(|slot| slot.slot == "momentum"),
        "muon keeps one slot per parameter and it is not a moment pair"
    );
    assert_eq!(
        record
            .optimizer
            .slots_in(checkpoint::SlotScope::Shared)
            .count(),
        0,
        "muon declares no shared state"
    );

    let mut resumed = Trainer::new(&model, config(OptimizerKind::Muon, matrix_selector()))
        .expect("load a fresh trainer");
    resumed.declare_trainable_set(&set).expect("the same set");
    let compatibility = compatibility_for(&mut resumed, &model, OptimizerKind::Muon);
    resumed
        .load_checkpoint(&state, &compatibility)
        .expect("restore a muon checkpoint");
    assert_eq!(train_one_row(&mut resumed), continued);
    assert_eq!(
        deviation(&expected, &scores(&mut resumed)),
        0.0,
        "the resumed step did not land where the uninterrupted one did"
    );
}

// --- Gefen -------------------------------------------------------------------

fn gefen(variant: GefenVariant, block_size: u64) -> OptimizerKind {
    OptimizerKind::Gefen(GefenLayout {
        variant,
        block_size,
        ..GefenLayout::default()
    })
}

fn total_elements(set: &TrainableSet) -> u64 {
    set.entries.iter().map(|entry| entry.n_elements).sum()
}

/// Both variants, against the oracle, at a block size that divides nothing:
/// every tensor ends in a partial trailing block.
///
/// The oracle is fed the inputs the run itself observed (pre-step weights,
/// the run's own backward gradient), so the same body checks any backend
/// without comparing two backends' arithmetic.
fn assert_gefen_matches_the_oracle(model: &Path, device: Device) {
    // Per-device bound: the relative error is dominated by a near-zero
    // weight, and a GPU reduction in a different order lands a small multiple
    // above the CPU figure. Both stay four orders of magnitude under the
    // negative control.
    let tolerance = match device {
        Device::Cpu => 1.0e-4,
        _ => 1.0e-3,
    };
    for variant in [GefenVariant::SharedV, GefenVariant::QuantizedM] {
        // Small enough that every matrix spans many blocks. The selection adds
        // the norms and drops the fallback threshold so Gefen owns them too:
        // they are shorter than one block, which is the partial-trailing-block
        // case, and the only way to reach it on a fixture whose matrices are
        // all powers of two.
        let block_size = 512_u64;
        let kind = OptimizerKind::Gefen(GefenLayout {
            variant,
            block_size,
            min_numel: 1,
        });
        let mut run = config(kind, mixed_selector());
        run.device = device;
        let mut trainer = Trainer::new(model, run).expect("load trainer");
        let set = resolved(model, &mixed_selector());
        trainer.declare_trainable_set(&set).expect("select");
        trainer.prepare_optimizer().expect("build the gefen graph");

        let marked = trainer.marked_trainable_set().expect("the marked set");
        let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
        assert!(
            marked
                .entries
                .iter()
                .any(|entry| entry.n_elements % block_size != 0),
            "no partial trailing block in this selection"
        );

        let before: Vec<Vec<f32>> = (0..sizes.len())
            .map(|index| parameter_values(&mut trainer, index, sizes[index]))
            .collect();
        assert_eq!(train_one_row(&mut trainer), 1, "exactly one step");
        let after: Vec<Vec<f32>> = (0..sizes.len())
            .map(|index| parameter_values(&mut trainer, index, sizes[index]))
            .collect();
        let gradients: Vec<Vec<f32>> = (0..sizes.len())
            .map(|index| parameter_gradient(&mut trainer, index))
            .collect();

        let knobs = trainer
            .optimizer_hyperparameters()
            .expect("the values the update read");
        assert_eq!(knobs.optimizer(), kind);
        let alpha = f64::from(scalar(&knobs, "learning_rate"));
        let beta1 = f64::from(scalar(&knobs, "beta1"));
        let beta2 = f64::from(scalar(&knobs, "beta2"));
        let eps = f64::from(scalar(&knobs, "eps"));
        let decay = f64::from(scalar(&knobs, "weight_decay"));
        let scale = f64::from(clip_scale(&gradients, 1.0e9));

        let mut worst = 0.0_f64;
        let mut worst_without_block_scaling = 0.0_f64;
        for (index, entry) in marked.entries.iter().enumerate() {
            let n = entry.n_elements as usize;
            let blocks = n.div_ceil(block_size as usize);
            let gradient: Vec<f64> = gradients[index]
                .iter()
                .map(|value| f64::from(*value) * scale)
                .collect();
            let weights: Vec<f64> = before[index]
                .iter()
                .map(|value| f64::from(*value))
                .collect();
            let step = gefen_step(
                variant,
                block_size as usize,
                &weights,
                &gradient,
                &vec![0.0; n],
                &vec![GEFEN_ZERO_BLOCK_INDEX; n],
                &vec![0.0; blocks],
                &vec![0.0; blocks],
                alpha,
                beta1,
                beta2,
                eps,
                decay,
                1,
            );
            for (offset, w1) in after[index].iter().enumerate() {
                let expected = step.weights[offset];
                let magnitude = expected.abs().max(f64::from(*w1)).max(1.0e-6);
                worst = worst.max((expected - f64::from(*w1)).abs() / magnitude);
            }
            // The same step with one global second moment instead of one per
            // block: it has to fail, or the block structure is unasserted.
            let global = gefen_step(
                variant,
                n,
                &weights,
                &gradient,
                &vec![0.0; n],
                &vec![GEFEN_ZERO_BLOCK_INDEX; n],
                &[0.0],
                &[0.0],
                alpha,
                beta1,
                beta2,
                eps,
                decay,
                1,
            );
            for (offset, w1) in after[index].iter().enumerate() {
                let naive = global.weights[offset];
                let magnitude = naive.abs().max(f64::from(*w1)).max(1.0e-6);
                worst_without_block_scaling =
                    worst_without_block_scaling.max((naive - f64::from(*w1)).abs() / magnitude);
            }
        }
        assert!(
            worst < tolerance,
            "{variant} on {device:?}: the gefen step is not the oracle's: \
             relative error {worst}"
        );
        assert!(
            worst_without_block_scaling > 1000.0 * tolerance,
            "{variant} on {device:?}: one global second moment would have passed \
             too: {worst_without_block_scaling}"
        );
    }
}

#[test]
fn a_gefen_step_is_the_fixed_block_algorithm_it_claims_to_be() {
    let model = fixture!();
    let _guard = common::serialize_models();
    assert_gefen_matches_the_oracle(&model, Device::Cpu);
}

/// The same oracle, on the GPU: the F64 arithmetic is the algorithm oracle,
/// the inputs are the GPU run's own, not the CPU run's output.
#[test]
fn a_gefen_step_on_the_gpu_is_the_same_fixed_block_algorithm() {
    let model = fixture!();
    if !retrograd::gpu_runtime_available() {
        eprintln!("skipping: no GPU runtime to run the step on");
        return;
    }
    let _guard = common::serialize_models();
    assert_gefen_matches_the_oracle(&model, Device::Gpu);
}

/// At `B = 1` shared-v keeps one second moment per element, which is AdamW's,
/// so the two agree on the cheapest available correctness anchor. The first
/// moments agree too, and the updates differ only by the square of one number
/// against the number itself - so this compares the state rather than the
/// weights.
#[test]
fn shared_v_at_one_element_per_block_keeps_adamws_second_moment() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("gefen-b1");

    let state_of = |kind: OptimizerKind, name: &str| {
        let mut trainer =
            Trainer::new(&model, config(kind, matrix_selector())).expect("load trainer");
        let set = resolved(&model, &matrix_selector());
        trainer.declare_trainable_set(&set).expect("select");
        let step = train_one_row(&mut trainer);
        let path = root.join(name);
        trainer
            .save_checkpoint(&path, &metadata(&model, step))
            .expect("save");
        drop(trainer);
        path
    };

    let adamw_path = state_of(OptimizerKind::AdamW, "adamw.state");
    let gefen_path = state_of(gefen(GefenVariant::SharedV, 1), "gefen.state");

    let adamw = Checkpoint::read(&adamw_path).expect("read");
    let gefen_record = Checkpoint::read(&gefen_path).expect("read");
    assert_eq!(gefen_record.optimizer.kind, "gefen");
    // shared_v is layout 1 and quantized_m is layout 2: a payload written under
    // one must not restore into the other.
    assert_eq!(
        gefen_record
            .optimizer
            .assignment
            .iter()
            .find(|row| row.optimizer == "gefen")
            .expect("a gefen row")
            .layout_version,
        1
    );

    let adamw_v = payload(&adamw_path, &adamw, "v");
    let gefen_v = payload(&gefen_path, &gefen_record, "v");
    assert_eq!(adamw_v.len(), gefen_v.len(), "B = 1 is one row per element");
    let worst = adamw_v
        .iter()
        .zip(&gefen_v)
        .map(|(left, right)| (left - right).abs() / left.abs().max(1.0e-12))
        .fold(0.0_f32, f32::max);
    assert!(
        worst < 1.0e-5,
        "shared-v at B = 1 is not adamw's second moment: {worst}"
    );
    let adamw_m = payload(&adamw_path, &adamw, "m");
    let gefen_m = payload(&gefen_path, &gefen_record, "m");
    assert_eq!(adamw_m, gefen_m, "the first moments are the same recursion");
}

/// The quantized variant's whole state crosses the checkpoint boundary,
/// including the shared codebook - which is the first byte anything has ever
/// written or read in the shared scope.
#[test]
fn quantized_m_round_trips_its_indices_scales_and_shared_codebook() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("gefen-quantized");
    let kind = gefen(GefenVariant::QuantizedM, 512);

    let mut trainer = Trainer::new(&model, config(kind, matrix_selector())).expect("load trainer");
    let set = resolved(&model, &matrix_selector());
    trainer.declare_trainable_set(&set).expect("select");
    let step = train_one_row(&mut trainer);
    let state = root.join("quantized.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save a quantized checkpoint");
    let continued = train_one_row(&mut trainer);
    let expected = scores(&mut trainer);
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read");
    assert_eq!(record.optimizer.kind, "gefen");
    assert_eq!(
        record
            .optimizer
            .assignment
            .iter()
            .find(|row| row.optimizer == "gefen")
            .expect("a gefen row")
            .layout_version,
        2,
        "the variant is what moves the layout version"
    );

    // One shared codebook, for one owner, whatever the parameter count.
    let shared: Vec<&checkpoint::StateSlot> = record
        .optimizer
        .slots_in(checkpoint::SlotScope::Shared)
        .collect();
    assert_eq!(shared.len(), 1, "{shared:?}");
    assert_eq!(shared[0].owner, "gefen");
    assert_eq!(shared[0].slot, "codebook");
    assert_eq!(shared[0].n_bytes, GEFEN_CODEBOOK_LEVELS * 4);

    let codebook = payload(&state, &record, "codebook");
    assert_eq!(codebook.len() as u64, GEFEN_CODEBOOK_LEVELS);
    assert_eq!(codebook[0], -1.0);
    assert_eq!(codebook[codebook.len() - 1], 1.0);
    assert!(
        codebook.iter().all(|value| *value != 0.0),
        "the uniform codebook has no exact zero; a zero block decodes through \
         its zero scale instead"
    );

    // The indices are unsigned bytes over the whole range, and the step used
    // more than the one code the initializer wrote.
    let indices = raw_payload(&state, &record, "indices");
    assert!(
        indices.iter().any(|code| *code != GEFEN_ZERO_BLOCK_INDEX),
        "nothing was quantized"
    );
    assert!(
        indices.iter().any(|code| *code > 127),
        "no index above the signed range; the payload is being read as signed"
    );
    let distinct = indices
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert!(distinct > 64, "only {distinct} of 256 codes were used");

    let scales = payload(&state, &record, "scales");
    assert!(
        scales
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );

    let mut resumed = Trainer::new(&model, config(kind, matrix_selector())).expect("fresh trainer");
    resumed.declare_trainable_set(&set).expect("the same set");
    let compatibility = compatibility_for(&mut resumed, &model, kind);
    resumed
        .load_checkpoint(&state, &compatibility)
        .expect("restore indices, scales, second moments and the codebook");
    assert_eq!(train_one_row(&mut resumed), continued);
    assert_eq!(
        deviation(&expected, &scores(&mut resumed)),
        0.0,
        "the resumed step did not land where the uninterrupted one did"
    );
}

/// A payload written under one variant is not readable as the other: the
/// layout version moves with the variant, and the slot names do too.
#[test]
fn a_shared_v_payload_does_not_restore_into_a_quantized_m_run() {
    let model = fixture!();
    let _guard = common::serialize_models();
    let root = scratch("gefen-variants");

    let mut trainer = Trainer::new(
        &model,
        config(gefen(GefenVariant::SharedV, 512), matrix_selector()),
    )
    .expect("load trainer");
    let set = resolved(&model, &matrix_selector());
    trainer.declare_trainable_set(&set).expect("select");
    let step = train_one_row(&mut trainer);
    let state = root.join("shared.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("save");
    drop(trainer);

    let other = gefen(GefenVariant::QuantizedM, 512);
    let mut resumed =
        Trainer::new(&model, config(other, matrix_selector())).expect("fresh trainer");
    resumed.declare_trainable_set(&set).expect("the same set");
    let compatibility = compatibility_for(&mut resumed, &model, other);
    let error = resumed
        .load_checkpoint(&state, &compatibility)
        .expect_err("two variants are two layouts");
    let message = error.to_string();
    assert!(
        message.contains("layout") || message.contains("slot"),
        "{message}"
    );
}

/// A state mutation must never be answered on a fallback backend: the
/// scheduler would update a copy and leave the device-resident slot stale.
/// The device decides whether it carries the step, at load time; this asserts
/// that `cap_opt_step_device` and the preflight refusal agree.
#[test]
fn the_device_decides_whether_a_gefen_run_may_start() {
    let model = fixture!();
    if !retrograd::gpu_runtime_available() {
        eprintln!("skipping: no GPU runtime to ask");
        return;
    }
    let _guard = common::serialize_models();

    let mut run = config(gefen(GefenVariant::SharedV, 512), matrix_selector());
    run.device = Device::Gpu;
    let mut trainer = Trainer::new(&model, run).expect("load trainer");
    let set = resolved(&model, &matrix_selector());
    trainer.declare_trainable_set(&set).expect("select");

    let report = trainer.backend_report().expect("a backend report");
    let probed = report
        .lines()
        .find_map(|line| line.trim().strip_prefix("cap_opt_step_device: "))
        .expect("the report declares whether the device carries this step")
        .trim()
        .to_string();

    match probed.as_str() {
        "supported" => {
            trainer
                .prepare_optimizer()
                .expect("the device declared the step and must then build it");
            assert_eq!(train_one_row(&mut trainer), 1);
            assert!(scores(&mut trainer).iter().all(|value| value.is_finite()));
        }
        "unavailable" => {
            let error = trainer
                .prepare_optimizer()
                .expect_err("the device declared no step and must then refuse");
            let message = error.to_string();
            assert!(message.contains("gefen"), "{message}");
            assert!(message.contains("no update step"), "{message}");
        }
        other => panic!("cap_opt_step_device is '{other}'"),
    }

    // Muon is built out of ops every backend carries, so no device declines
    // it: the case above is about one optimizer's kernels, not about training
    // on a device at all.
    let mut run = config(OptimizerKind::Muon, matrix_selector());
    run.device = Device::Gpu;
    let mut trainer = Trainer::new(&model, run).expect("load trainer");
    trainer.declare_trainable_set(&set).expect("select");
    trainer
        .prepare_optimizer()
        .expect("muon is a graph of ordinary ops");
    assert_eq!(train_one_row(&mut trainer), 1);
    assert!(scores(&mut trainer).iter().all(|value| value.is_finite()));
}

// --- shared helpers ----------------------------------------------------------

/// Every payload of one slot name, decoded as F32 and concatenated in file
/// order. The checkpoint's own offsets are what address it, so this reads what
/// the writer wrote rather than what the runtime still holds.
fn payload(state: &Path, record: &Checkpoint, slot: &str) -> Vec<f32> {
    read_f32(&raw_payload(state, record, slot))
}

fn raw_payload(state: &Path, record: &Checkpoint, slot: &str) -> Vec<u8> {
    let bytes = std::fs::read(state.join(checkpoint::OPTIMIZER_STATE_FILE))
        .expect("the optimizer state file");
    let mut out = Vec::new();
    for row in record.optimizer.slots.iter().filter(|row| row.slot == slot) {
        let first = usize::try_from(row.offset).expect("a host-sized offset");
        let last = first + usize::try_from(row.n_bytes).expect("a host-sized slot");
        out.extend_from_slice(&bytes[first..last]);
    }
    assert!(!out.is_empty(), "no '{slot}' slot in this checkpoint");
    out
}

fn metadata(model: &Path, global_step: u64) -> CheckpointMetadata {
    CheckpointMetadata {
        checkpoint_id: format!("step-{global_step:012}"),
        algorithm: "sft".into(),
        trajectory_signature: "test-optimizers-v1".into(),
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
    optimizer: OptimizerKind,
) -> checkpoint::Compatibility {
    let run = config(optimizer, matrix_selector());
    let hyperparameters = trainer
        .optimizer_hyperparameters()
        .expect("optimizer hyperparameters");
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("fingerprint"),
        reference_fingerprint: trainer.reference_fingerprint().expect("anchor fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-optimizers-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: run.learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: optimizer.as_str().into(),
        optimizer_layout_version: optimizer.layout_version(),
        optimizer_hyperparameters: hyperparameters.lines(),
        weight_decay: run.weight_decay,
        max_grad_norm: run.max_grad_norm,
        trainable_policy: trainer.trainable_policy().as_str().to_string(),
        trainable_signature: trainer.trainable_signature().expect("trainable signature"),
    }
}
