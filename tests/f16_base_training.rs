//! F16 base weights: the admission, the parity, the stability and the resume.
//!
//! - **Admission**: a dtype is admitted under an *optimizer* whose update
//!   kernel writes it, on a *backend* whose implementation accepts it. Both
//!   refusals are exercised, and the declared table is compared with what the
//!   live device says.
//! - **Parity**: the same step, on the same numbers, at two storage
//!   precisions. Meaningful only because the two fixtures hold *identical*
//!   values (every weight is snapped to the F16 grid before storage), so a
//!   difference is a difference in precision, not in the draw.
//! - **Stability**: a long run of the number of steps the row claims, with the
//!   run kept finite and bounded over all of them.
//! - **Resume**: an F16 AdamW step rounds stochastically from a stream seeded
//!   by the optimizer's iteration counter, which the checkpoint already holds.
//!   One case checks restored weights, iteration and forward loss; a second
//!   one checks the whole of it, on the F32 fixture - an interrupted `n + m`
//!   step run landing bit-for-bit where an uninterrupted one does.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{
    BASE_DTYPE_TABLE, BaseDtypeCapability, CheckpointMetadata, Device, OptimizerKind, TensorDtype,
    TrainConfig, TrainablePolicy, TrainableRunConfig, TrainableSelector, TrainableSet, Trainer,
    base_dtype_admits, base_dtype_capability, resolve_base, tensor_inventory,
};

const TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

/// One training row of the generated fixtures: one context plus the label
/// shift, and therefore exactly one optimizer step.
const ONE_ROW_TOKENS: usize = 257;

/// The tensors both parity runs train: two attention projections in the middle
/// of the stack, matrices rather than vectors (the generator keeps every
/// vector F32), and far enough from the head that the gradient reaching them
/// has been through something.
const PARITY_MODULES: [&str; 2] = ["attn_q", "attn_v"];

macro_rules! f32_fixture {
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

macro_rules! f16_fixture {
    () => {
        match common::tiny_f16_model_path_if_available() {
            Some(model) => model,
            None => {
                eprintln!("skipping: generated F16 fixture not available");
                return;
            }
        }
    };
}

fn selector() -> TrainableSelector {
    TrainableSelector {
        modules: PARITY_MODULES.iter().map(|m| (*m).to_string()).collect(),
        ..Default::default()
    }
}

fn config(optimizer: OptimizerKind) -> TrainConfig {
    TrainConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 64,
        epochs: 1,
        // Above an F16 ulp of these weights (~1e-4), so one step is visible
        // in the stored value rather than rounded away.
        learning_rate: 1.0e-3,
        weight_decay: 0.0,
        device: Device::Cpu,
        trainable: TrainableRunConfig {
            policy: TrainablePolicy::Partial,
            selector: selector(),
            optimizer,
        },
        ..TrainConfig::default()
    }
}

fn resolved_set(model: &Path) -> TrainableSet {
    let inventory = tensor_inventory(model, Device::Cpu).expect("read the tensor inventory");
    resolve_base(&inventory, TrainablePolicy::Partial, &selector())
        .expect("the selected projections resolve")
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("retrograd-f16-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

/// One optimizer step, on exactly one row.
fn train_one_row(trainer: &mut Trainer) -> u64 {
    let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    assert!(tokens.len() >= ONE_ROW_TOKENS, "{} tokens", tokens.len());
    tokens.truncate(ONE_ROW_TOKENS);
    trainer
        .train_tokens(&tokens)
        .expect("training run")
        .global_step
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

/// F16 bits, widened. The parameter read returns the parameter's own storage;
/// comparing it is comparing what an F16 run keeps.
fn read_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(|pair| f16_to_f32(u16::from_ne_bytes(pair)))
        .collect()
}

/// IEEE binary16 to binary32, written out rather than pulled in: the test
/// compares storage, so the decode has to be the standard one.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x03FF);
    if exponent == 0 {
        // Subnormal (or zero): the F16 grid is uniform here, so the value is
        // the mantissa in units of the smallest one.
        let magnitude = f32::from(mantissa as u16) * 5.960_464_5e-8;
        return f32::from_bits(sign) + if sign == 0 { magnitude } else { -magnitude };
    }
    if exponent == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}

/// One unit in the last place of the F16 grid an F16 parameter is stored on.
/// The natural unit for comparing two update trajectories when one is
/// quantized; a relative error would count a correct rounding near zero as an
/// error of one.
fn f16_ulp(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude < 6.103_515_6e-5 {
        // Subnormal range: the grid is uniform.
        return 5.960_464_5e-8;
    }
    let exponent = magnitude.log2().floor();
    (exponent - 10.0).exp2()
}

fn parameter_bytes(trainer: &mut Trainer, index: usize, n_bytes: u64) -> Vec<u8> {
    let mut bytes = vec![0_u8; usize::try_from(n_bytes).expect("a host-sized tensor")];
    trainer
        .read_marked_parameter(index, 0, &mut bytes)
        .expect("read a marked parameter");
    bytes
}

/// A marked parameter's values, decoded from whatever it is stored as.
fn parameter_values(
    trainer: &mut Trainer,
    index: usize,
    entry_dtype: &TensorDtype,
    n_bytes: u64,
) -> Vec<f32> {
    let bytes = parameter_bytes(trainer, index, n_bytes);
    match entry_dtype {
        TensorDtype::F16 => read_f16(&bytes),
        _ => read_f32(&bytes),
    }
}

fn parameter_gradient(trainer: &mut Trainer, index: usize) -> Vec<f32> {
    let info = trainer
        .parameter_gradient_info(index)
        .expect("describe a gradient");
    // F32 whatever the parameter's own storage: no master copy, and the two
    // runs compared below produce gradients in the same units.
    assert_eq!(info.dtype, TensorDtype::F32, "{}", info.name);
    let mut bytes = vec![0_u8; usize::try_from(info.n_bytes).expect("a host-sized tensor")];
    trainer
        .read_parameter_gradient(index, 0, &mut bytes)
        .expect("read a gradient");
    read_f32(&bytes)
}

/// The ggml registry this process trains on, read from the trainer's own
/// report rather than guessed from the build features.
fn backend_registry(trainer: &mut Trainer) -> String {
    let report = trainer.backend_report().expect("a backend report");
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix("backend_registry: "))
        .expect("the report names the active registry")
        .trim()
        .to_string()
}

fn report_line(trainer: &mut Trainer, key: &str) -> String {
    let report = trainer.backend_report().expect("a backend report");
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("the report has no '{key}' line"))
        .trim()
        .to_string()
}

/// The row that admitted this combination here, or a skip: a device with no
/// row is a combination nobody measured, not a failure of this lane.
fn row_for(trainer: &mut Trainer) -> Option<&'static BaseDtypeCapability> {
    let registry = backend_registry(trainer);
    base_dtype_capability(&TensorDtype::F16, OptimizerKind::AdamW, &registry)
}

// --- admission ----------------------------------------------------------------

/// The fixture pair is a control, asserted rather than assumed: same names,
/// same shapes, same *numbers*, different storage.
#[test]
fn the_two_fixtures_are_one_model_at_two_storage_precisions() {
    let f32_model = f32_fixture!();
    let f16_model = f16_fixture!();

    let left = tensor_inventory(&f32_model, Device::Cpu).expect("read the F32 inventory");
    let right = tensor_inventory(&f16_model, Device::Cpu).expect("read the F16 inventory");
    assert_eq!(left.architecture, right.architecture);
    assert_eq!(left.n_layer, right.n_layer);
    assert_eq!(left.tensors.len(), right.tensors.len());

    let mut matrices = 0_usize;
    let mut vectors = 0_usize;
    for (a, b) in left.tensors.iter().zip(&right.tensors) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.ne, b.ne, "{}", a.name);
        assert_eq!(a.n_elements, b.n_elements, "{}", a.name);
        assert_eq!(a.dtype, TensorDtype::F32, "{}", a.name);
        if a.ne[1] > 1 {
            assert_eq!(b.dtype, TensorDtype::F16, "{}", b.name);
            assert_eq!(b.n_bytes, a.n_bytes / 2, "{}", b.name);
            matrices += 1;
        } else {
            // Vectors stay F32 in both, as a real F16 GGUF does, which makes
            // the F16 fixture the mixed case: one run marks both precisions.
            assert_eq!(b.dtype, TensorDtype::F32, "{}", b.name);
            vectors += 1;
        }
    }
    assert!(matrices > 0 && vectors > 0, "{matrices} / {vectors}");
}

/// The dtype screen, and the terms it does *not* answer alone.
#[test]
fn the_dtype_screen_is_a_screen_and_the_row_is_the_admission() {
    assert!(TensorDtype::F32.is_trainable_base());
    assert!(TensorDtype::F16.is_trainable_base());
    assert!(!TensorDtype::BF16.is_trainable_base());
    assert!(!TensorDtype::from_ggml_name("Q4_K").is_trainable_base());

    // Passing the screen is not admission: the optimizer and the backend are
    // asked separately and either can refuse.
    assert!(!base_dtype_admits(
        &TensorDtype::F16,
        OptimizerKind::Sgd,
        "CPU"
    ));
    assert!(!base_dtype_admits(
        &TensorDtype::F16,
        OptimizerKind::AdamW,
        "a-backend-no-lane-has-run"
    ));
    assert!(!BASE_DTYPE_TABLE.is_empty(), "the widening has no row");
}

/// An F16 base tensor is marked, carries an F32 gradient, and moves.
#[test]
fn an_f16_base_tensor_is_marked_and_the_update_moves_it() {
    let model = f16_fixture!();
    let _guard = common::serialize_models();

    let set = resolved_set(&model);
    let mut trainer = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    let Some(_row) = row_for(&mut trainer) else {
        eprintln!(
            "skipping: no BASE_DTYPE_TABLE row for F16/adamw on {}",
            backend_registry(&mut trainer)
        );
        return;
    };
    // The declared table and the live device agree.
    assert!(
        report_line(&mut trainer, "cap_opt_step_f16").contains("adamw"),
        "the row claims F16 AdamW and the device's own probe declines it"
    );

    trainer
        .declare_trainable_set(&set)
        .expect("the selected projections");
    trainer
        .prepare_optimizer()
        .expect("an F16 base set is marked");

    let marked = trainer.marked_trainable_set().expect("the marked set");
    assert_eq!(
        marked.entries.len(),
        set.entries.len(),
        "the declared set and the marked set are the same set"
    );
    let f16_entries = marked
        .entries
        .iter()
        .filter(|entry| entry.dtype == TensorDtype::F16)
        .count();
    assert_eq!(
        f16_entries,
        marked.entries.len(),
        "the selection was supposed to be matrices only"
    );

    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|i| parameter_values(&mut trainer, i, &TensorDtype::F16, sizes[i]))
        .collect();
    assert_eq!(train_one_row(&mut trainer), 1);
    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|i| parameter_values(&mut trainer, i, &TensorDtype::F16, sizes[i]))
        .collect();

    let mut moved = 0_usize;
    for (index, entry) in marked.entries.iter().enumerate() {
        let gradient = parameter_gradient(&mut trainer, index);
        assert_eq!(gradient.len(), before[index].len(), "{}", entry.name);
        assert!(
            gradient.iter().all(|value| value.is_finite()),
            "{} has a non-finite gradient",
            entry.name
        );
        assert!(
            gradient.iter().any(|value| *value != 0.0),
            "{} was marked and got no gradient",
            entry.name
        );
        assert!(
            after[index].iter().all(|value| value.is_finite()),
            "{} left the representable range",
            entry.name
        );
        moved += before[index]
            .iter()
            .zip(&after[index])
            .filter(|(a, b)| a != b)
            .count();
    }
    assert!(moved > 0, "no F16 element changed in the step");
}

/// The admission follows the optimizer, not the dtype: SGD's update step is
/// F32-only, so the same selection that trains under AdamW is refused under
/// SGD by name, before a graph is built.
#[test]
fn an_f16_base_set_is_refused_by_an_optimizer_whose_step_cannot_write_it() {
    let model = f16_fixture!();
    let _guard = common::serialize_models();

    let set = resolved_set(&model);
    let mut trainer = Trainer::new(&model, config(OptimizerKind::Sgd)).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("the same selection resolves whatever the optimizer");
    let error = trainer
        .prepare_optimizer()
        .expect_err("sgd cannot write an F16 parameter");
    let message = error.to_string();
    assert!(message.contains("sgd"), "{message}");
    // `ggml_type_name`'s spelling, which is what the runtime quotes.
    assert!(message.contains("f16"), "{message}");
    // The refusal is the dtype one, not rule 6's "declared but not marked".
    assert!(
        !message.contains("marked"),
        "the refusal arrived after the marking: {message}"
    );
    // It names the backend, because the answer is per backend.
    let registry = backend_registry(&mut trainer);
    assert!(message.contains(&registry), "{message}");
}

// --- parity -------------------------------------------------------------------

/// One step, twice: the same tokens, the same configuration and the same
/// weights, stored once as F32 and once as F16. Per tensor and per element,
/// the gradient and the update are held to the tolerances the row publishes,
/// which come from the table so the row's combination has to keep meeting
/// them rather than describing a one-off measurement.
#[test]
fn an_f16_step_matches_the_f32_step_within_the_published_tolerance() {
    let f32_model = f32_fixture!();
    let f16_model = f16_fixture!();
    let _guard = common::serialize_models();

    let mut f32_trainer =
        Trainer::new(&f32_model, config(OptimizerKind::AdamW)).expect("load the F32 trainer");
    let Some(row) = row_for(&mut f32_trainer) else {
        eprintln!("skipping: no row for this backend");
        return;
    };
    f32_trainer
        .declare_trainable_set(&resolved_set(&f32_model))
        .expect("the F32 selection");
    f32_trainer.prepare_optimizer().expect("mark the F32 set");

    let mut f16_trainer =
        Trainer::new(&f16_model, config(OptimizerKind::AdamW)).expect("load the F16 trainer");
    f16_trainer
        .declare_trainable_set(&resolved_set(&f16_model))
        .expect("the F16 selection");
    f16_trainer.prepare_optimizer().expect("mark the F16 set");

    let left = f32_trainer.marked_trainable_set().expect("marked F32");
    let right = f16_trainer.marked_trainable_set().expect("marked F16");
    assert_eq!(left.entries.len(), right.entries.len());
    assert!(!left.entries.is_empty());

    let before_f32: Vec<Vec<f32>> = (0..left.entries.len())
        .map(|i| {
            parameter_values(
                &mut f32_trainer,
                i,
                &TensorDtype::F32,
                left.entries[i].n_bytes,
            )
        })
        .collect();
    let before_f16: Vec<Vec<f32>> = (0..right.entries.len())
        .map(|i| {
            parameter_values(
                &mut f16_trainer,
                i,
                &TensorDtype::F16,
                right.entries[i].n_bytes,
            )
        })
        .collect();

    // The control, asserted before anything is compared against it: if the two
    // files did not hold the same numbers, every difference below would be a
    // difference in the draw.
    for (index, entry) in left.entries.iter().enumerate() {
        assert_eq!(
            before_f32[index], before_f16[index],
            "{} differs before the step: the fixtures are not one model",
            entry.name
        );
    }

    assert_eq!(train_one_row(&mut f32_trainer), 1);
    assert_eq!(train_one_row(&mut f16_trainer), 1);

    // AdamW's own bound on one step: `|delta| <= alpha * (1 + wd)`, so two
    // trajectories are at most twice that apart, plus the grid the F16 result
    // lands on (an ulp at the largest weight in the set).
    let knobs = f16_trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    let scalar = |name: &str| match knobs.get(name) {
        Some(retrograd::HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    };
    let alpha = scalar("learning_rate");
    let decay = scalar("weight_decay");
    let largest = before_f16
        .iter()
        .flatten()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let ceiling = 2.0 * alpha * (1.0 + decay) + f16_ulp(largest);
    assert!(alpha > 0.0 && ceiling.is_finite());

    let mut worst_gradient = 0.0_f32;
    let mut worst_gradient_at = String::new();
    let mut worst_update_ulps = 0.0_f32;
    let mut worst_update_at = String::new();
    let mut compared = 0_usize;
    let mut over = 0_usize;
    let mut worst_divergence = 0.0_f32;

    for (index, entry) in left.entries.iter().enumerate() {
        assert_eq!(entry.name, right.entries[index].name);
        let g32 = parameter_gradient(&mut f32_trainer, index);
        let g16 = parameter_gradient(&mut f16_trainer, index);
        assert_eq!(g32.len(), g16.len(), "{}", entry.name);

        let after_f32 = parameter_values(
            &mut f32_trainer,
            index,
            &TensorDtype::F32,
            left.entries[index].n_bytes,
        );
        let after_f16 = parameter_values(
            &mut f16_trainer,
            index,
            &TensorDtype::F16,
            right.entries[index].n_bytes,
        );

        // The gradient scale of this tensor, so an element at the tensor's
        // noise floor is not asked to agree relatively with a number that is
        // nearly zero.
        let scale = g32
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()))
            .max(f32::MIN_POSITIVE);

        for element in 0..g32.len() {
            assert!(
                g32[element].is_finite()
                    && g16[element].is_finite()
                    && after_f32[element].is_finite()
                    && after_f16[element].is_finite(),
                "{}[{element}] has a non-finite gradient or parameter",
                entry.name
            );
            let difference = (g32[element] - g16[element]).abs();
            let relative = difference / scale;
            if relative > worst_gradient {
                worst_gradient = relative;
                worst_gradient_at = format!("{}[{element}]", entry.name);
            }

            // The update, in units of the grid the F16 parameter is stored on;
            // an F16 run cannot land between two grid points.
            let delta32 = after_f32[element] - before_f32[index][element];
            let delta16 = after_f16[element] - before_f16[index][element];
            let scale = before_f16[index][element]
                .abs()
                .max(after_f32[element].abs());
            let ulps = (delta32 - delta16).abs() / f16_ulp(scale);
            if ulps > worst_update_ulps {
                worst_update_ulps = ulps;
                worst_update_at = format!("{}[{element}]", entry.name);
            }
            if ulps > row.update_tolerance {
                over += 1;
            }
            worst_divergence = worst_divergence.max((delta32 - delta16).abs());
            compared += 1;
        }
    }

    assert!(compared > 0);
    let outliers = over as f32 / compared as f32;
    eprintln!(
        "F16 parity over {compared} elements: gradient {worst_gradient:.3e} \
         (at {worst_gradient_at}), update worst {worst_update_ulps:.1} ulp \
         (at {worst_update_at}), {over} over {:.1} ulp ({outliers:.2e} of the set), \
         largest divergence {worst_divergence:.3e} against a step ceiling of {ceiling:.3e}; \
         the row publishes {:.3e} / {:.1} ulp / {:.2e}",
        row.update_tolerance,
        row.gradient_tolerance,
        row.update_tolerance,
        row.update_outlier_fraction
    );
    assert!(
        worst_gradient <= row.gradient_tolerance,
        "the gradient parity the row publishes is not met: {worst_gradient:.3e} \
         at {worst_gradient_at}, allowed {:.3e}",
        row.gradient_tolerance
    );
    assert!(
        outliers <= row.update_outlier_fraction,
        "{outliers:.2e} of the elements take an update more than {:.1} grid point(s) from \
         the F32 trajectory (worst {worst_update_ulps:.1} ulp at {worst_update_at}), and the \
         row admits {:.2e}",
        row.update_tolerance,
        row.update_outlier_fraction
    );
    // The bound that holds for *every* element, outliers included: an AdamW
    // step is bounded by alpha in magnitude, so two trajectories cannot be
    // further apart than two steps plus the grid the F16 result is stored on.
    assert!(
        worst_divergence <= ceiling,
        "an element diverged by {worst_divergence:.3e}, beyond the optimizer's own step \
         ceiling of {ceiling:.3e}: F16 storage amplified the update rather than rounding it"
    );
    // Two bit-identical runs would prove nothing about precision.
    assert!(
        worst_update_ulps > 0.0,
        "the two runs were bit-identical: the F16 fixture is not F16"
    );
}

// --- stability ------------------------------------------------------------------

/// The row claims a number of steps; this runs them and asserts what a long
/// F16 run can actually lose: finiteness, a weight that has not walked out of
/// the representable range, and a loss that has not diverged from where it
/// started.
///
/// `RETRO_F16_STABILITY_STEPS` shortens the run while iterating but cannot
/// lengthen the claim: the row is what the lane ran.
#[test]
fn f16_base_training_stays_finite_and_bounded_over_the_claimed_run() {
    let model = f16_fixture!();
    let _guard = common::serialize_models();

    let mut trainer = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    let Some(row) = row_for(&mut trainer) else {
        eprintln!("skipping: no row for this backend");
        return;
    };
    let steps = std::env::var("RETRO_F16_STABILITY_STEPS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(row.stability_steps)
        .min(row.stability_steps);
    assert!(
        steps > LATE,
        "RETRO_F16_STABILITY_STEPS must exceed the {LATE}-step late window"
    );
    drop(trainer);

    let f32_model = f32_fixture!();
    let f16 = run_stability(&model, steps, &TensorDtype::F16);
    let f32 = run_stability(&f32_model, steps, &TensorDtype::F32);

    eprintln!(
        "F16 stability over {steps} steps: loss {:.5} -> {:.5} (F32: {:.5} -> {:.5}), \
         |w|max {:.5} -> {:.5} (F32: {:.5}), {} of {} elements moved, \
         last {LATE} steps moved {}",
        f16.first_loss,
        f16.last_loss,
        f32.first_loss,
        f32.last_loss,
        f16.largest_initial,
        f16.largest_final,
        f32.largest_final,
        f16.moved,
        f16.elements,
        f16.moved_late
    );

    // Per-step finiteness is asserted inside `run_stability`; this is the end
    // state.
    assert!(f16.largest_final.is_finite() && f16.last_loss.is_finite());
    assert!(f16.moved > 0, "{steps} steps changed nothing");

    // How far the weights travelled, against the F32 control on identical data.
    // What F16 storage can do is inflate the drift (a rounding with an
    // accumulating bias walks a weight outward every step), and the F32 run is
    // the reference that shows it. The margin is small on purpose: a
    // combination that starts drifting has changed.
    let envelope = f32.largest_final * 1.05 + f16_ulp(f32.largest_final);
    assert!(
        f16.largest_final <= envelope,
        "the largest weight reached {} after {steps} steps against F32's {}: F16 \
         storage inflated the drift rather than rounding it",
        f16.largest_final,
        f32.largest_final
    );
    // And stayed far from where F16 stops being a number: 65504 is the largest
    // finite F16, and within an order of magnitude of it is one bad step from
    // infinity.
    assert!(
        f16.largest_final < 6550.0,
        "the weights reached {}, within an order of magnitude of F16's ceiling",
        f16.largest_final
    );

    // An update smaller than half a grid point rounds away, so an F16 run can
    // stop moving *silently*. The late window is where it would show: by then
    // the weights have grown, their grid is coarser, and the step has not
    // changed.
    assert!(
        f16.moved_late > 0,
        "nothing moved in the last {LATE} steps: the updates rounded away"
    );

    // Against F32 on the same steps: not bit-parity, which a long normalized
    // run cannot have, but the same outcome. A run that quietly stopped
    // training looks identical on every check above and differs here.
    assert!(
        f16.last_loss < f16.first_loss,
        "the F16 run did not train: {} -> {}",
        f16.first_loss,
        f16.last_loss
    );
    let loss_gap =
        (f16.last_loss - f32.last_loss).abs() / f32.last_loss.abs().max(f32::MIN_POSITIVE);
    assert!(
        loss_gap <= row.stability_loss_tolerance,
        "after {steps} steps the F16 run's loss is {} against F32's {} - a relative \
         gap of {loss_gap:.3e}, and the row admits {:.3e}",
        f16.last_loss,
        f32.last_loss,
        row.stability_loss_tolerance
    );
}

/// Steps of the long run whose movement is checked on its own, at the end,
/// where an F16 update is most likely to have rounded away.
const LATE: u32 = 20;

struct StabilityRun {
    first_loss: f32,
    last_loss: f32,
    largest_initial: f32,
    largest_final: f32,
    moved: usize,
    moved_late: usize,
    elements: usize,
}

/// `steps` optimizer steps on one row, with everything the assertions above
/// read recorded as it goes. Run once per fixture: a claim about F16 that is
/// not also measured on F32 is a claim about this model, not about the
/// precision.
fn run_stability(model: &Path, steps: u32, dtype: &TensorDtype) -> StabilityRun {
    let mut trainer = Trainer::new(model, config(OptimizerKind::AdamW)).expect("load trainer");
    trainer
        .declare_trainable_set(&resolved_set(model))
        .expect("the selected projections");
    trainer.prepare_optimizer().expect("mark the set");
    let marked = trainer.marked_trainable_set().expect("the marked set");
    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let values = |trainer: &mut Trainer| -> Vec<Vec<f32>> {
        (0..sizes.len())
            .map(|i| parameter_values(trainer, i, dtype, sizes[i]))
            .collect()
    };
    let largest = |values: &[Vec<f32>]| {
        values
            .iter()
            .flatten()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()))
    };

    let initial = values(&mut trainer);
    let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    tokens.truncate(ONE_ROW_TOKENS);

    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    let mut late_start: Vec<Vec<f32>> = Vec::new();
    for step in 0..steps {
        if step + LATE == steps {
            late_start = values(&mut trainer);
        }
        let metrics = trainer.train_tokens(&tokens).expect("a training step");
        assert!(
            metrics.train_loss.is_finite(),
            "{}: step {step} produced a non-finite loss",
            model.display()
        );
        if step == 0 {
            first_loss = metrics.train_loss;
        }
        last_loss = metrics.train_loss;
    }

    let final_values = values(&mut trainer);
    let mut moved = 0_usize;
    let mut elements = 0_usize;
    for (index, entry) in marked.entries.iter().enumerate() {
        for (before, after) in initial[index].iter().zip(&final_values[index]) {
            assert!(
                after.is_finite(),
                "{}: {} left the representable range after {steps} steps",
                model.display(),
                entry.name
            );
            elements += 1;
            if before != after {
                moved += 1;
            }
        }
    }
    let moved_late = late_start
        .iter()
        .zip(&final_values)
        .map(|(before, after)| {
            before
                .iter()
                .zip(after)
                .filter(|(before, after)| before != after)
                .count()
        })
        .sum();

    StabilityRun {
        first_loss,
        last_loss,
        largest_initial: largest(&initial),
        largest_final: largest(&final_values),
        moved,
        moved_late,
        elements,
    }
}

// --- resume --------------------------------------------------------------------

fn metadata(model: &Path, global_step: u64) -> CheckpointMetadata {
    CheckpointMetadata {
        checkpoint_id: format!("step-{global_step:012}"),
        algorithm: "sft".into(),
        trajectory_signature: "test-f16-v1".into(),
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

fn compatibility_for(trainer: &mut Trainer, model: &Path) -> checkpoint::Compatibility {
    let reference = config(OptimizerKind::AdamW);
    let hyperparameters = trainer
        .optimizer_hyperparameters()
        .expect("optimizer hyperparameters");
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-f16-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: reference.learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: "adamw".into(),
        optimizer_layout_version: hyperparameters.optimizer().layout_version(),
        optimizer_hyperparameters: hyperparameters.lines(),
        weight_decay: reference.weight_decay,
        max_grad_norm: reference.max_grad_norm,
        trainable_policy: trainer.trainable_policy().as_str().to_string(),
        trainable_signature: trainer.trainable_signature().expect("trainable signature"),
    }
}

/// Restores the F16 bytes and the optimizer iteration, then checks the next
/// forward loss. Forward-loss parity does not establish parity of the
/// subsequent update or RNG draws.
#[test]
fn an_f16_checkpoint_restores_weights_iteration_and_forward_loss() {
    let model = f16_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("resume");
    let set = resolved_set(&model);

    // Long enough that the momenta are no longer their initial zeros.
    const BEFORE: u32 = 3;

    let mut trainer = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    if row_for(&mut trainer).is_none() {
        eprintln!("skipping: no row for this backend");
        return;
    }
    trainer
        .declare_trainable_set(&set)
        .expect("the selected projections");
    let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
    tokens.truncate(ONE_ROW_TOKENS);
    // Each `train_tokens` call is one run of its own, so the checkpoint's
    // progress has to be the one the trainer actually holds, not a count of
    // the calls.
    let mut step = 0;
    for _ in 0..BEFORE {
        step = trainer
            .train_tokens(&tokens)
            .expect("a training step")
            .global_step;
    }
    // The loss of the step *after* the checkpoint, from the uninterrupted run.
    let marked = trainer.marked_trainable_set().expect("the marked set");
    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let stored: Vec<Vec<u8>> = (0..sizes.len())
        .map(|i| parameter_bytes(&mut trainer, i, sizes[i]))
        .collect();

    let state = root.join("step-000000000001.state");
    trainer
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("checkpoint an F16 base run");
    let next_loss = trainer
        .train_tokens(&tokens)
        .expect("the step after the checkpoint")
        .train_loss;
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the checkpoint");
    let bundle = record.manifest.trainable.as_ref().expect("a base bundle");
    // An F16 run that published F32 values would restore a *different* model:
    // every weight moved to the nearest F32, not where the F16 run left it.
    assert!(
        bundle.tensors.iter().all(|tensor| tensor.dtype == "F16"),
        "the bundle widened the values it was given"
    );
    let saved_iter = record.optimizer.iter;
    assert!(saved_iter > 1, "the counter never advanced: {saved_iter}");

    let mut resumed = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut resumed, &model);
    resumed
        .load_checkpoint(&state, &expected)
        .expect("restore an F16 base checkpoint");

    // The weights came back bit-for-bit, in F16.
    for (index, entry) in marked.entries.iter().enumerate() {
        assert_eq!(
            parameter_bytes(&mut resumed, index, sizes[index]),
            stored[index],
            "{} did not come back as it was stored",
            entry.name
        );
    }

    // And so did the counter the rounding stream is seeded by, read back
    // through a checkpoint taken from the *restored* trainer: the live value,
    // not the one the file was written with.
    let echo = root.join("echo.state");
    resumed
        .save_checkpoint(&echo, &metadata(&model, step))
        .expect("checkpoint the restored trainer");
    let echoed = Checkpoint::read(&echo).expect("read the echo");
    assert_eq!(
        echoed.optimizer.iter, saved_iter,
        "the optimizer iteration - and with it the seed of the rounding stream - \
         did not survive the restore"
    );

    // The forward reads the restored weights; it does not exercise the
    // rounding of the update that follows it.
    let resumed_loss = resumed
        .train_tokens(&tokens)
        .expect("the first step after the restore")
        .train_loss;
    assert_eq!(
        resumed_loss, next_loss,
        "the restored run did not reproduce the step it was interrupted before"
    );

    // No new checkpoint field was needed for any of this; the format version
    // is unchanged.
    assert_eq!(record.manifest.format_version, checkpoint::FORMAT_VERSION);
}

/// An interrupted run and an uninterrupted one land on the same weights, to
/// the bit: `n` steps, checkpoint, fresh trainer, `m` steps must equal `n + m`
/// steps in one process.
///
/// The case the resume test above deliberately stops short of, and the one the
/// `partial` case in `tests/base_training.rs` cannot see because it compares
/// scores after zero further steps. What it caught: the gradient accumulators
/// are live state no checkpoint holds, and `ggml_opt_alloc` was failing to
/// clear them between accumulation windows, so an uninterrupted run fed every
/// step the sum of every window before it while a restored one started from
/// zero. The two agreed on the restored weights and on the next forward loss -
/// only the update after them differed, which is why nothing shorter than this
/// saw it.
///
/// On the F32 fixture: the finding is about what the optimizer carries between
/// steps, not about storage precision, and F32 removes stochastic rounding
/// from the comparison.
#[test]
fn an_interrupted_run_lands_bit_for_bit_where_an_uninterrupted_one_does() {
    let model = f32_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("continuity");
    let set = resolved_set(&model);

    // Both halves are longer than one step: with `n_batch / n_ubatch` micro
    // batches per step, a single step on each side would accumulate one window
    // and never compare a second one against what the first left behind.
    const BEFORE: u32 = 3;
    const AFTER: u32 = 3;

    let row = |trainer: &mut Trainer| -> Vec<i32> {
        let mut tokens = trainer.tokenize_text(&TEXT.repeat(8)).expect("tokenize");
        tokens.truncate(ONE_ROW_TOKENS);
        tokens
    };
    let weights = |trainer: &mut Trainer| -> Vec<Vec<u8>> {
        let marked = trainer.marked_trainable_set().expect("the marked set");
        marked
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| parameter_bytes(trainer, index, entry.n_bytes))
            .collect()
    };

    // The reference: every step in one process.
    let mut straight = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    straight
        .declare_trainable_set(&set)
        .expect("the selected projections");
    let tokens = row(&mut straight);
    for _ in 0..BEFORE + AFTER {
        straight.train_tokens(&tokens).expect("a training step");
    }
    let reference = weights(&mut straight);
    drop(straight);

    // The same run, interrupted after `BEFORE`.
    let mut first = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    first
        .declare_trainable_set(&set)
        .expect("the same selection");
    let mut step = 0;
    for _ in 0..BEFORE {
        step = first
            .train_tokens(&tokens)
            .expect("a training step")
            .global_step;
    }
    let state = root.join("interrupted.state");
    first
        .save_checkpoint(&state, &metadata(&model, step))
        .expect("checkpoint the interrupted run");
    drop(first);

    let mut resumed = Trainer::new(&model, config(OptimizerKind::AdamW)).expect("load trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut resumed, &model);
    resumed
        .load_checkpoint(&state, &expected)
        .expect("restore the checkpoint");
    for _ in 0..AFTER {
        resumed.train_tokens(&tokens).expect("a training step");
    }
    let continued = weights(&mut resumed);

    let marked = resumed.marked_trainable_set().expect("the marked set");
    assert_eq!(continued.len(), reference.len());
    for (index, entry) in marked.entries.iter().enumerate() {
        let differing = reference[index]
            .iter()
            .zip(&continued[index])
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(
            differing,
            0,
            "{} differs in {differing} of {} bytes between the interrupted run \
             and the uninterrupted one",
            entry.name,
            reference[index].len()
        );
    }
}
