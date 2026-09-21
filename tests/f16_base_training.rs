//! Half-precision base weights: the admission, the parity, the stability and
//! the resume.
//!
//! - **Admission**: a dtype is admitted under an *optimizer* whose update
//!   kernel writes it, on a *backend* whose implementation accepts it. Both
//!   refusals are exercised, and the declared table is compared with what the
//!   live device says.
//! - **Parity**: the same step, on the same numbers, at two storage
//!   precisions. Meaningful only because a fixture and its control hold
//!   *identical* values (every weight is snapped to the storage grid before
//!   either file is written), so a difference is a difference in precision,
//!   not in the draw.
//! - **Stability**: a long run of the number of steps the row claims, with the
//!   run kept finite and bounded over all of them.
//! - **Resume**: a half-precision AdamW step rounds stochastically from a
//!   stream seeded by the optimizer's iteration counter, which the checkpoint
//!   already holds. One case checks restored weights, iteration and forward
//!   loss; a second checks the whole of it on an F32 fixture, an interrupted
//!   `n + m` step run landing bit-for-bit where an uninterrupted one does.
//!
//! Every lane runs once per (storage, optimizer, device) triple, which is
//! what a row is indexed by; a combination with no row is skipped, not
//! failed.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use retrograd::checkpoint::{self, Checkpoint};
use retrograd::{
    BASE_DTYPE_TABLE, BaseDtypeCapability, CheckpointMetadata, Device, OptimizerKind, ProbeInputs,
    ProbeOp, TensorDtype, TrainConfig, TrainablePolicy, TrainableRunConfig, TrainableSelector,
    TrainableSet, Trainer, base_dtype_admits, base_dtype_capability, probe_op, resolve_base,
    tensor_inventory,
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

/// One storage precision the lane asks its four questions of, with the
/// control those questions are answered against.
///
/// The F16 and BF16 grids are not nested, so each storage carries its own
/// control rather than sharing the F32 fixture.
struct Storage {
    dtype: TensorDtype,
    model: fn() -> Option<PathBuf>,
    control: fn() -> Option<PathBuf>,
}

fn storages() -> Vec<Storage> {
    vec![
        Storage {
            dtype: TensorDtype::F16,
            model: common::tiny_f16_model_path_if_available,
            control: common::tiny_model_path_if_available,
        },
        Storage {
            dtype: TensorDtype::BF16,
            model: common::tiny_bf16_model_path_if_available,
            control: common::tiny_bf16_control_model_path_if_available,
        },
    ]
}

/// The optimizers whose update step writes a half-precision parameter; Muon
/// and Gefen are F32-only and the refusal test covers them.
fn optimizers() -> [OptimizerKind; 2] {
    [OptimizerKind::AdamW, OptimizerKind::Sgd]
}

impl Storage {
    /// The fixture pair, or `None` with a skip printed: a missing fixture
    /// cannot run the lane, and that is not a failure.
    fn pair(&self) -> Option<(PathBuf, PathBuf)> {
        match ((self.model)(), (self.control)()) {
            (Some(model), Some(control)) => Some((model, control)),
            _ => {
                eprintln!("skipping: the {} fixture pair is not available", self.dtype);
                None
            }
        }
    }

    /// The update step driven on its own, without a model: one probe id per
    /// (optimizer, storage), because that is one kernel each.
    fn probe(&self, optimizer: OptimizerKind) -> ProbeOp {
        match (optimizer, &self.dtype) {
            (OptimizerKind::Sgd, TensorDtype::BF16) => ProbeOp::OptStepSgdBf16,
            (OptimizerKind::Sgd, _) => ProbeOp::OptStepSgdF16,
            (_, TensorDtype::BF16) => ProbeOp::OptStepAdamwBf16,
            (_, _) => ProbeOp::OptStepAdamwF16,
        }
    }
}

fn selector() -> TrainableSelector {
    TrainableSelector {
        modules: PARITY_MODULES.iter().map(|m| (*m).to_string()).collect(),
        ..Default::default()
    }
}

/// The devices this build can run a lane on: the CPU, plus the GPU when the
/// runtime can create a context on it.
fn devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if common::gpu_device_present() {
        devices.push(Device::Gpu);
    }
    devices
}

/// Runs `lane` once per device, naming the device on the way in so a failure
/// says which backend produced it.
fn per_device(lane: impl Fn(Device)) {
    for device in devices() {
        eprintln!("--- device {device:?} ---");
        lane(device);
    }
}

/// Runs `lane` once per (storage, optimizer, device) triple, naming all
/// three on the way in.
fn per_case(lane: impl Fn(&Storage, OptimizerKind, Device)) {
    for storage in storages() {
        for optimizer in optimizers() {
            for device in devices() {
                eprintln!(
                    "--- {} under {optimizer} on device {device:?} ---",
                    storage.dtype
                );
                lane(&storage, optimizer, device);
            }
        }
    }
}

/// The step size each optimizer's lane runs at. AdamW normalizes its step to
/// roughly `alpha`; SGD's step is the raw gradient, two orders of magnitude
/// smaller here, so it needs a larger rate to leave a visible trace.
fn learning_rate(optimizer: OptimizerKind) -> f32 {
    match optimizer {
        OptimizerKind::Sgd => 1.0e-1,
        _ => 1.0e-3,
    }
}

fn config(optimizer: OptimizerKind, device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 64,
        epochs: 1,
        learning_rate: learning_rate(optimizer),
        weight_decay: 0.0,
        device,
        trainable: TrainableRunConfig {
            policy: TrainablePolicy::Partial,
            selector: selector(),
            optimizer,
        },
        ..TrainConfig::default()
    }
}

fn resolved_set(model: &Path, device: Device) -> TrainableSet {
    let inventory = tensor_inventory(model, device).expect("read the tensor inventory");
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
/// comparing it is comparing what a half-precision run keeps.
fn read_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(|pair| f16_to_f32(u16::from_ne_bytes(pair)))
        .collect()
}

/// BF16 bits, widened. BF16 is the top half of the F32 with the same value, so
/// the decode is a shift and there is no subnormal or infinity special case.
fn read_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(|pair| f32::from_bits(u32::from(u16::from_ne_bytes(pair)) << 16))
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

/// One unit in the last place of the grid a parameter of `dtype` is stored on.
/// The natural unit for comparing two update trajectories when one is
/// quantized; a relative error would count a correct rounding near zero as an
/// error of one.
fn storage_ulp(dtype: &TensorDtype, value: f32) -> f32 {
    match dtype {
        TensorDtype::BF16 => bf16_ulp(value),
        _ => f16_ulp(value),
    }
}

fn f16_ulp(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude < 6.103_515_6e-5 {
        // Subnormal range: the grid is uniform.
        return 5.960_464_5e-8;
    }
    let exponent = magnitude.log2().floor();
    (exponent - 10.0).exp2()
}

/// BF16 carries 8 significand bits against F16's 11 and F32's exponent range,
/// so its grid is F32's shifted by 16 bits: one ulp is 2^(e-7), and the
/// smallest normal is F32's.
fn bf16_ulp(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude < f32::MIN_POSITIVE {
        return (-133.0_f32).exp2();
    }
    let exponent = magnitude.log2().floor();
    (exponent - 7.0).exp2()
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
        TensorDtype::BF16 => read_bf16(&bytes),
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

/// The row that admitted this combination here, or `None`: a pair with no
/// row was never measured and is skipped.
fn row_for(
    trainer: &mut Trainer,
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
) -> Option<&'static BaseDtypeCapability> {
    let registry = backend_registry(trainer);
    base_dtype_capability(dtype, optimizer, &registry)
}

/// The live device's own answer, read from the `cap_opt_step` line, which
/// spells the matrix as `adamw{f16, bf16}, sgd{f16, bf16}`.
fn device_probe_admits(
    trainer: &mut Trainer,
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
) -> bool {
    let line = report_line(trainer, "cap_opt_step");
    let named = format!("{optimizer}{{");
    let Some(rest) = line.split(&named).nth(1) else {
        return false;
    };
    let storages = rest.split('}').next().unwrap_or_default();
    storages
        .split(',')
        .any(|storage| storage.trim().eq_ignore_ascii_case(dtype.name()))
}

// --- admission ----------------------------------------------------------------

/// Each fixture pair is a control, asserted rather than assumed: same names,
/// same shapes, same *numbers*, different storage.
#[test]
fn each_fixture_pair_is_one_model_at_two_storage_precisions() {
    for storage in storages() {
        let Some((model, control)) = storage.pair() else {
            continue;
        };
        let left = tensor_inventory(&control, Device::Cpu).expect("read the control inventory");
        let right = tensor_inventory(&model, Device::Cpu).expect("read the stored inventory");
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
                assert_eq!(b.dtype, storage.dtype, "{}", b.name);
                assert_eq!(b.n_bytes, a.n_bytes / 2, "{}", b.name);
                matrices += 1;
            } else {
                // Vectors stay F32 in both, as a real half-precision GGUF does,
                // which makes each fixture the mixed case: one run marks both
                // precisions.
                assert_eq!(b.dtype, TensorDtype::F32, "{}", b.name);
                vectors += 1;
            }
        }
        assert!(matrices > 0 && vectors > 0, "{matrices} / {vectors}");
    }
}

/// The dtype screen, and the terms it does *not* answer alone.
#[test]
fn the_dtype_screen_is_a_screen_and_the_row_is_the_admission() {
    assert!(TensorDtype::F32.is_trainable_base());
    assert!(TensorDtype::F16.is_trainable_base());
    assert!(TensorDtype::BF16.is_trainable_base());
    assert!(!TensorDtype::from_ggml_name("Q4_K").is_trainable_base());

    // Passing the screen is not admission: the optimizer and the backend are
    // asked separately and either can refuse.
    assert!(!base_dtype_admits(
        &TensorDtype::F16,
        OptimizerKind::Muon,
        "CPU"
    ));
    assert!(!base_dtype_admits(
        &TensorDtype::F16,
        OptimizerKind::AdamW,
        "a-backend-no-lane-has-run"
    ));
    assert!(!BASE_DTYPE_TABLE.is_empty(), "the widening has no row");
}

/// The declared table and the running backend answer the same question in
/// both directions: a row means the marking succeeds, no row means it is
/// refused by name. Only the backend this process runs on can be asked, which
/// is also why a row is written per measured backend.
#[test]
fn the_row_and_the_running_backend_agree_on_whether_a_storage_is_admitted() {
    per_case(table_agrees_with_backend);
}

fn table_agrees_with_backend(storage: &Storage, optimizer: OptimizerKind, device: Device) {
    let Some((model, _)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();

    let set = resolved_set(&model, device);
    let mut trainer = Trainer::new(&model, config(optimizer, device)).expect("load trainer");
    let registry = backend_registry(&mut trainer);
    let tabled = base_dtype_capability(&storage.dtype, optimizer, &registry).is_some();
    // The device's own probe, which is neither the table nor the refusal.
    let probed = device_probe_admits(&mut trainer, &storage.dtype, optimizer);
    assert_eq!(
        tabled, probed,
        "{registry}: the table says {tabled} and the device's {optimizer} {} probe says {probed}",
        storage.dtype
    );

    trainer
        .declare_trainable_set(&set)
        .expect("the selected projections");
    let named = storage.dtype.name().to_ascii_lowercase();
    match trainer.prepare_optimizer() {
        Ok(()) => assert!(
            tabled,
            "{registry} marked a {} base set under {optimizer} that no row admits",
            storage.dtype
        ),
        Err(error) => {
            assert!(
                !tabled,
                "{registry} refused a combination it has a row for: {error}"
            );
            let message = error.to_string();
            // The refusal names all three coordinates, because all three are
            // what a reader has to change.
            assert!(message.contains(&named), "{message}");
            assert!(message.contains(optimizer.as_str()), "{message}");
            assert!(message.contains(&registry), "{message}");
        }
    }
}

/// A half-precision base tensor is marked, carries an F32 gradient, and moves.
#[test]
fn a_half_precision_base_tensor_is_marked_and_the_update_moves_it() {
    per_case(marked_and_moved);
}

fn marked_and_moved(storage: &Storage, optimizer: OptimizerKind, device: Device) {
    let Some((model, _)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();

    let set = resolved_set(&model, device);
    let mut trainer = Trainer::new(&model, config(optimizer, device)).expect("load trainer");
    let Some(_row) = row_for(&mut trainer, &storage.dtype, optimizer) else {
        eprintln!(
            "skipping: no BASE_DTYPE_TABLE row for {}/{optimizer} on {}",
            storage.dtype,
            backend_registry(&mut trainer)
        );
        return;
    };
    // The declared table and the live device agree.
    assert!(
        device_probe_admits(&mut trainer, &storage.dtype, optimizer),
        "the row claims {} under {optimizer} and the device's own probe declines it",
        storage.dtype
    );

    trainer
        .declare_trainable_set(&set)
        .expect("the selected projections");
    trainer
        .prepare_optimizer()
        .expect("a half-precision base set is marked");

    let marked = trainer.marked_trainable_set().expect("the marked set");
    assert_eq!(
        marked.entries.len(),
        set.entries.len(),
        "the declared set and the marked set are the same set"
    );
    let stored_entries = marked
        .entries
        .iter()
        .filter(|entry| entry.dtype == storage.dtype)
        .count();
    assert_eq!(
        stored_entries,
        marked.entries.len(),
        "the selection was supposed to be matrices only"
    );

    let sizes: Vec<u64> = marked.entries.iter().map(|entry| entry.n_bytes).collect();
    let before: Vec<Vec<f32>> = (0..sizes.len())
        .map(|i| parameter_values(&mut trainer, i, &storage.dtype, sizes[i]))
        .collect();
    assert_eq!(train_one_row(&mut trainer), 1);
    let after: Vec<Vec<f32>> = (0..sizes.len())
        .map(|i| parameter_values(&mut trainer, i, &storage.dtype, sizes[i]))
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
    assert!(moved > 0, "no stored element changed in the step");
}

/// Muon's update step is F32-only, so a set assigned to it is refused by
/// name before a graph is built. The assignment is named per parameter,
/// which overrides Muon's AdamW fallback; the refusal says the step is
/// F32-only rather than naming an empty list of backends.
#[test]
fn an_f32_only_optimizer_refuses_a_half_precision_base_set() {
    for storage in storages() {
        for device in devices() {
            eprintln!("--- {} under muon on device {device:?} ---", storage.dtype);
            refused_by_muon(&storage, device);
        }
    }
}

fn refused_by_muon(storage: &Storage, device: Device) {
    let Some((model, _)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();

    assert!(
        !OptimizerKind::Muon.supports_dtype(&storage.dtype),
        "muon grew a half-precision store; this case needs another optimizer"
    );
    let set = resolved_set(&model, device);
    let mut trainer =
        Trainer::new(&model, config(OptimizerKind::Muon, device)).expect("load trainer");
    trainer
        .declare_trainable_set(&set)
        .expect("the same selection resolves whatever the optimizer");
    let assignment: Vec<(String, OptimizerKind)> = set
        .base_entries()
        .map(|entry| (entry.name.clone(), OptimizerKind::Muon))
        .collect();
    assert!(!assignment.is_empty());
    trainer
        .set_optimizer_assignment(&assignment)
        .expect("muon is named for every selected parameter");
    let error = trainer
        .prepare_optimizer()
        .expect_err("muon cannot write a half-precision parameter");
    let message = error.to_string();
    assert!(message.contains("muon"), "{message}");
    // `ggml_type_name`'s spelling, which is what the runtime quotes.
    assert!(
        message.contains(&storage.dtype.name().to_ascii_lowercase()),
        "{message}"
    );
    assert!(message.contains("F32 only"), "{message}");
    // The refusal is the dtype one, not rule 6's "declared but not marked".
    assert!(
        !message.contains("marked"),
        "the refusal arrived after the marking: {message}"
    );
    // It names the backend, because the answer is per backend.
    let registry = backend_registry(&mut trainer);
    assert!(message.contains(&registry), "{message}");
}

/// The rounding on its own, without a model: an update below half a grid
/// point still has to accumulate, which is the reason the store is
/// stochastic. Run through the probe, which is also what a backend's kernel
/// is compared against.
#[test]
fn an_update_below_half_a_grid_point_still_accumulates() {
    const N: usize = 256;
    const STEPS: u32 = 300;
    const ALPHA: f32 = 1.0e-5;

    for storage in storages() {
        for optimizer in optimizers() {
            let grid = storage_ulp(&storage.dtype, 1.0);
            assert!(
                ALPHA < grid / 2.0,
                "{}: a step of {ALPHA} is not below half a grid point of {grid}",
                storage.dtype
            );
            let ne = [N as i64, 1, 1, 1];
            let gradients = vec![1.0_f32; N];
            let mut weights = vec![1.0_f32; N];
            for step in 0..STEPS {
                // Both optimizers move by ALPHA here: AdamW's moments reset
                // each probe call, and SGD's step is the gradient itself.
                // src2 carries {clipping scale, rounding seed}; the moving
                // seed mirrors a run's iteration counter.
                let scale_and_seed = [1.0_f32, step as f32];
                weights = probe_op(
                    storage.probe(optimizer),
                    false,
                    ProbeInputs::pair(ne, &weights, ne, &gradients)
                        .with_src2(Some(([2, 1, 1, 1], scale_and_seed.as_slice()))),
                    [ALPHA, 0.0],
                    N,
                )
                .expect("run one chained update through the probe");
            }

            assert!(
                weights.iter().all(|value| value.is_finite()),
                "{}: stochastic rounding manufactured a non-finite weight",
                storage.dtype
            );
            assert!(
                weights.iter().any(|value| *value != 1.0),
                "{}: every weight is still exactly 1.0, so the updates rounded away",
                storage.dtype
            );

            // Unbiased means the mean lands on the exact arithmetic, not merely
            // somewhere below where it started. The spread is one grid point over
            // the root of the population, which is what makes this a bound rather
            // than a fitted number.
            let mean = weights.iter().sum::<f32>() / N as f32;
            let expected = 1.0 - ALPHA * STEPS as f32;
            let spread = grid / (N as f32).sqrt();
            eprintln!(
                "{} under {optimizer} rounding over {N} elements and {STEPS} steps: mean \
             {mean:.6} against the exact {expected:.6}, one grid point being {grid:.3e}",
                storage.dtype
            );
            assert!(
                (mean - expected).abs() < 4.0 * spread,
                "{}: the mean drifted to {mean} from the unbiased {expected}, beyond \
             four times the {spread:.3e} spread one grid point allows",
                storage.dtype
            );
        }
    }
}

// --- parity -------------------------------------------------------------------

/// One step, twice: the same tokens, the same configuration and the same
/// weights, stored once as F32 and once at the storage under test. Per tensor
/// and per element, the gradient and the update are held to the tolerances the
/// row publishes, which come from the table so the row's combination has to
/// keep meeting them rather than describing a one-off measurement.
#[test]
fn a_half_precision_step_matches_the_f32_step_within_the_published_tolerance() {
    per_case(step_parity);
}

fn step_parity(storage: &Storage, optimizer: OptimizerKind, device: Device) {
    let Some((stored_model, f32_model)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();

    let mut f32_trainer =
        Trainer::new(&f32_model, config(optimizer, device)).expect("load the F32 trainer");
    let Some(row) = row_for(&mut f32_trainer, &storage.dtype, optimizer) else {
        eprintln!("skipping: no row for this combination");
        return;
    };
    f32_trainer
        .declare_trainable_set(&resolved_set(&f32_model, device))
        .expect("the F32 selection");
    f32_trainer.prepare_optimizer().expect("mark the F32 set");

    let mut stored_trainer = Trainer::new(&stored_model, config(optimizer, device))
        .expect("load the half-precision trainer");
    stored_trainer
        .declare_trainable_set(&resolved_set(&stored_model, device))
        .expect("the half-precision selection");
    stored_trainer
        .prepare_optimizer()
        .expect("mark the half-precision set");

    let left = f32_trainer.marked_trainable_set().expect("marked F32");
    let right = stored_trainer
        .marked_trainable_set()
        .expect("marked half-precision");
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
    let before_stored: Vec<Vec<f32>> = (0..right.entries.len())
        .map(|i| {
            parameter_values(
                &mut stored_trainer,
                i,
                &storage.dtype,
                right.entries[i].n_bytes,
            )
        })
        .collect();

    // The control, asserted before anything is compared against it: if the two
    // files did not hold the same numbers, every difference below would be a
    // difference in the draw.
    for (index, entry) in left.entries.iter().enumerate() {
        assert_eq!(
            before_f32[index], before_stored[index],
            "{} differs before the step: the fixtures are not one model",
            entry.name
        );
    }

    assert_eq!(train_one_row(&mut f32_trainer), 1);
    assert_eq!(train_one_row(&mut stored_trainer), 1);

    // Each optimizer's own bound on one step, so two trajectories are at most
    // twice that apart, plus the grid the stored result lands on. AdamW
    // normalizes its step: `|delta| <= alpha * (1 + wd)`, whatever the
    // gradient. SGD does not - its step *is* the gradient - so the bound is
    // read from the gradient of the element being compared, which the clipping
    // scale can only shrink.
    let knobs = stored_trainer
        .optimizer_hyperparameters()
        .expect("the values the update read");
    let scalar = |name: &str| match knobs.get(name) {
        Some(retrograd::HyperparameterValue::Scalar(value)) => value,
        other => panic!("{name} is {other:?}"),
    };
    let alpha = scalar("learning_rate");
    let decay = scalar("weight_decay");
    let step_ceiling = |gradient: f32, weight: f32| {
        let step = match optimizer {
            OptimizerKind::Sgd => alpha * (gradient.abs() + decay * weight.abs()),
            _ => alpha * (1.0 + decay),
        };
        2.0 * step + storage_ulp(&storage.dtype, weight)
    };
    assert!(alpha > 0.0);

    let mut worst_gradient = 0.0_f32;
    let mut worst_gradient_at = String::new();
    let mut worst_update_ulps = 0.0_f32;
    let mut worst_update_at = String::new();
    let mut compared = 0_usize;
    let mut over = 0_usize;
    let mut worst_divergence = 0.0_f32;
    // The divergence of the worst element as a fraction of what that element's
    // own step could move it: one number for a bound that is per element.
    let mut worst_excess = 0.0_f32;

    for (index, entry) in left.entries.iter().enumerate() {
        assert_eq!(entry.name, right.entries[index].name);
        let g32 = parameter_gradient(&mut f32_trainer, index);
        let g16 = parameter_gradient(&mut stored_trainer, index);
        assert_eq!(g32.len(), g16.len(), "{}", entry.name);

        let after_f32 = parameter_values(
            &mut f32_trainer,
            index,
            &TensorDtype::F32,
            left.entries[index].n_bytes,
        );
        let after_stored = parameter_values(
            &mut stored_trainer,
            index,
            &storage.dtype,
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
                    && after_stored[element].is_finite(),
                "{}[{element}] has a non-finite gradient or parameter",
                entry.name
            );
            let difference = (g32[element] - g16[element]).abs();
            let relative = difference / scale;
            if relative > worst_gradient {
                worst_gradient = relative;
                worst_gradient_at = format!("{}[{element}]", entry.name);
            }

            // The update, in units of the grid the parameter is stored on; a
            // half-precision run cannot land between two grid points.
            let delta32 = after_f32[element] - before_f32[index][element];
            let delta16 = after_stored[element] - before_stored[index][element];
            let scale = before_stored[index][element]
                .abs()
                .max(after_f32[element].abs());
            let ulps = (delta32 - delta16).abs() / storage_ulp(&storage.dtype, scale);
            if ulps > worst_update_ulps {
                worst_update_ulps = ulps;
                worst_update_at = format!("{}[{element}]", entry.name);
            }
            if ulps > row.update_tolerance {
                over += 1;
            }
            let divergence = (delta32 - delta16).abs();
            worst_divergence = worst_divergence.max(divergence);
            let ceiling = step_ceiling(
                g32[element].abs().max(g16[element].abs()),
                before_stored[index][element]
                    .abs()
                    .max(after_f32[element].abs()),
            );
            assert!(ceiling.is_finite() && ceiling > 0.0);
            worst_excess = worst_excess.max(divergence / ceiling);
            compared += 1;
        }
    }

    assert!(compared > 0);
    let outliers = over as f32 / compared as f32;
    let registry = backend_registry(&mut stored_trainer);
    eprintln!(
        "{} under {optimizer} parity on {registry} over {compared} elements: gradient \
         {worst_gradient:.3e} (at {worst_gradient_at}), update worst {worst_update_ulps:.1} ulp \
         (at {worst_update_at}), {over} over {:.1} ulp ({outliers:.2e} of the set), \
         largest divergence {worst_divergence:.3e} at {worst_excess:.2} of its own step \
         ceiling; the row publishes {:.3e} / {:.1} ulp / {:.2e}",
        storage.dtype,
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
    // The bound that holds for *every* element, outliers included: two
    // trajectories cannot be further apart than two of that element's steps
    // plus the grid the half-precision result is stored on.
    assert!(
        worst_excess <= 1.0,
        "an element diverged by {worst_divergence:.3e}, {worst_excess:.2} times its own \
         step ceiling: half-precision storage amplified the update rather than rounding it"
    );
    // Two bit-identical runs would prove nothing about precision.
    assert!(
        worst_update_ulps > 0.0,
        "the two runs were bit-identical: the {} fixture is not {}",
        storage.dtype,
        storage.dtype
    );
}

// --- stability ------------------------------------------------------------------

/// The row claims a number of steps; this runs them and asserts what a long
/// half-precision run can actually lose: finiteness, a weight that has not
/// walked out of the representable range, and a loss that has not diverged
/// from where it started.
///
/// `RETRO_F16_STABILITY_STEPS` shortens the run while iterating but cannot
/// lengthen the claim: the row is what the lane ran.
#[test]
fn half_precision_base_training_stays_finite_and_bounded_over_the_claimed_run() {
    per_case(stability);
}

fn stability(storage: &Storage, optimizer: OptimizerKind, device: Device) {
    let Some((model, f32_model)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();

    let mut trainer = Trainer::new(&model, config(optimizer, device)).expect("load trainer");
    let Some(row) = row_for(&mut trainer, &storage.dtype, optimizer) else {
        eprintln!("skipping: no row for this combination");
        return;
    };
    let registry = backend_registry(&mut trainer);
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

    let f16 = run_stability(&model, steps, &storage.dtype, optimizer, device);
    let f32 = run_stability(&f32_model, steps, &TensorDtype::F32, optimizer, device);

    eprintln!(
        "{} under {optimizer} stability on {registry} over {steps} steps: loss {:.5} -> {:.5} \
         (F32: {:.5} -> {:.5}), \
         |w|max {:.5} -> {:.5} (F32: {:.5}), |w|rms {:.5} (F32: {:.5}), \
         {} of {} elements moved, last {LATE} steps moved {}",
        storage.dtype,
        f16.first_loss,
        f16.last_loss,
        f32.first_loss,
        f32.last_loss,
        f16.largest_initial,
        f16.largest_final,
        f32.largest_final,
        f16.rms_final,
        f32.rms_final,
        f16.moved,
        f16.elements,
        f16.moved_late
    );

    // Per-step finiteness is asserted inside `run_stability`; this is the end
    // state.
    assert!(f16.largest_final.is_finite() && f16.last_loss.is_finite());
    assert!(f16.moved > 0, "{steps} steps changed nothing");

    // How far the weights travelled, against the F32 control on identical
    // data. Bounded on the RMS over the whole set: a storage bias moves every
    // element, while the single largest is the noisiest summary of the run,
    // and the F32 controls alone spread over 20% between fixture and backend.
    let envelope = f32.rms_final * 1.02;
    assert!(
        f16.rms_final <= envelope,
        "the weights ended at an RMS of {} after {steps} steps against F32's {}: {} \
         storage inflated the drift rather than rounding it",
        f16.rms_final,
        f32.rms_final,
        storage.dtype
    );
    // And stayed far from where the storage stops being a number: 65504 is the
    // largest finite F16 (BF16 reaches F32's range, so this is the tighter of
    // the two), and within an order of magnitude of it is one bad step from
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
        "the {} run did not train: {} -> {}",
        storage.dtype,
        f16.first_loss,
        f16.last_loss
    );
    let loss_gap =
        (f16.last_loss - f32.last_loss).abs() / f32.last_loss.abs().max(f32::MIN_POSITIVE);
    assert!(
        loss_gap <= row.stability_loss_tolerance,
        "after {steps} steps the {} run's loss is {} against F32's {} - a relative \
         gap of {loss_gap:.3e}, and the row admits {:.3e}",
        storage.dtype,
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
    rms_final: f32,
    moved: usize,
    moved_late: usize,
    elements: usize,
}

/// `steps` optimizer steps on one row, with everything the assertions above
/// read recorded as it goes. Run once per fixture: a claim about a storage
/// precision that is not also measured on F32 is a claim about this model, not
/// about the precision.
fn run_stability(
    model: &Path,
    steps: u32,
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
    device: Device,
) -> StabilityRun {
    let mut trainer = Trainer::new(model, config(optimizer, device)).expect("load trainer");
    trainer
        .declare_trainable_set(&resolved_set(model, device))
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
    let rms = |values: &[Vec<f32>]| {
        let count = values.iter().flatten().count();
        let sum: f64 = values
            .iter()
            .flatten()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum();
        (sum / count.max(1) as f64).sqrt() as f32
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
        rms_final: rms(&final_values),
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

fn compatibility_for(
    trainer: &mut Trainer,
    model: &Path,
    optimizer: OptimizerKind,
    device: Device,
) -> checkpoint::Compatibility {
    let reference = config(optimizer, device);
    let hyperparameters = trainer
        .optimizer_hyperparameters()
        .expect("optimizer hyperparameters");
    checkpoint::Compatibility {
        model_signature: trainer.model_signature().expect("model signature"),
        model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
        model_fingerprint: checkpoint::fingerprint_file(model).expect("fingerprint"),
        reference_fingerprint: trainer.reference_fingerprint().expect("anchor fingerprint"),
        algorithm: "sft".into(),
        trajectory_signature: "test-f16-v1".into(),
        dataset_fingerprint: checkpoint::fingerprint(TEXT.as_bytes()),
        scheduler_kind: "constant".into(),
        learning_rate: reference.learning_rate,
        warmup_steps: 0,
        total_steps: None,
        optimizer_kind: optimizer.as_str().into(),
        optimizer_layout_version: hyperparameters.optimizer().layout_version(),
        optimizer_hyperparameters: hyperparameters.lines(),
        weight_decay: reference.weight_decay,
        max_grad_norm: reference.max_grad_norm,
        trainable_policy: trainer.trainable_policy().as_str().to_string(),
        trainable_signature: trainer.trainable_signature().expect("trainable signature"),
    }
}

/// Restores the stored bytes and the optimizer iteration, then checks the next
/// forward loss. Forward-loss parity does not establish parity of the
/// subsequent update or RNG draws.
#[test]
fn a_half_precision_checkpoint_restores_weights_iteration_and_forward_loss() {
    per_case(checkpoint_restores);
}

fn checkpoint_restores(storage: &Storage, optimizer: OptimizerKind, device: Device) {
    let Some((model, _)) = storage.pair() else {
        return;
    };
    let _guard = common::serialize_models();
    let root = scratch("resume");
    let set = resolved_set(&model, device);

    // Long enough that the momenta are no longer their initial zeros.
    const BEFORE: u32 = 3;

    let mut trainer = Trainer::new(&model, config(optimizer, device)).expect("load trainer");
    if row_for(&mut trainer, &storage.dtype, optimizer).is_none() {
        eprintln!("skipping: no row for this combination");
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
        .expect("checkpoint a half-precision base run");
    let next_loss = trainer
        .train_tokens(&tokens)
        .expect("the step after the checkpoint")
        .train_loss;
    drop(trainer);

    let record = Checkpoint::read(&state).expect("read the checkpoint");
    let bundle = record.manifest.trainable.as_ref().expect("a base bundle");
    // A half-precision run that published F32 values would restore a
    // *different* model: every weight moved to the nearest F32, not where the
    // run left it.
    assert!(
        bundle
            .tensors
            .iter()
            .all(|tensor| tensor.dtype == storage.dtype.name()),
        "the bundle widened the values it was given"
    );
    let saved_iter = record.optimizer.iter;
    assert!(saved_iter > 1, "the counter never advanced: {saved_iter}");

    let mut resumed = Trainer::new(&model, config(optimizer, device)).expect("load trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut resumed, &model, optimizer, device);
    resumed
        .load_checkpoint(&state, &expected)
        .expect("restore a half-precision base checkpoint");

    // The weights came back bit-for-bit, at their storage precision.
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
    per_device(interrupted_run_continuity);
}

fn interrupted_run_continuity(device: Device) {
    let model = f32_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("continuity");
    let set = resolved_set(&model, device);

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
    let mut straight =
        Trainer::new(&model, config(OptimizerKind::AdamW, device)).expect("load trainer");
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
    let mut first =
        Trainer::new(&model, config(OptimizerKind::AdamW, device)).expect("load trainer");
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

    let mut resumed =
        Trainer::new(&model, config(OptimizerKind::AdamW, device)).expect("load trainer");
    resumed
        .declare_trainable_set(&set)
        .expect("the same selection");
    let expected = compatibility_for(&mut resumed, &model, OptimizerKind::AdamW, device);
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
