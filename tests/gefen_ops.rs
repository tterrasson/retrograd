//! The two fixed-block Gefen ops driven directly, with no model and no
//! training graph.
//!
//! A Gefen step reached through a run is a step whose inputs a backward pass
//! produced: one gradient, one state, one arithmetic path per fixture. The
//! cases the algorithm actually has to survive - a block whose gradient is
//! exactly zero, a block of one repeated magnitude, every one of the 256 index
//! codes, a partial trailing block, a step that only decays - are not reachable
//! that way at all, and driving them through a training step would be both slow
//! and indirect.
//!
//! Every case here states its own state, runs both phases on a named device and
//! checks the answer against the F64 oracle in `common::gefen`, which is the
//! same oracle the model-driven test uses. The device cases run the *same*
//! inputs on the CPU and on the GPU, so they are a parity assertion and not a
//! second tolerance.

mod common;

use common::gefen::{gefen_code, gefen_step};
use retrograd::{
    Device, GEFEN_CODEBOOK_LEVELS, GEFEN_ZERO_BLOCK_INDEX, GefenParams, GefenState, GefenVariant,
    gefen_codebook, gefen_step_probe, gefen_step_supported,
};

const BOTH: [GefenVariant; 2] = [GefenVariant::SharedV, GefenVariant::QuantizedM];

/// One place where the coefficients of these cases are chosen, so a case is a
/// gradient and a state and nothing else.
fn params(step: u32) -> GefenParams {
    GefenParams::at_step(1.0e-2, 0.9, 0.999, 1.0e-8, step)
}

fn oracle(
    variant: GefenVariant,
    block_size: usize,
    state: &GefenState,
    gradient: &[f32],
    knobs: GefenParams,
) -> common::gefen::GefenStep {
    let blocks = state.weights.len().div_ceil(block_size);
    let as_f64 = |values: &[f32]| {
        values
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>()
    };
    // The oracle derives the bias corrections from the step count, so the one
    // it is given has to be the one the coefficients were built at.
    let iteration = bias_corrected_step(knobs);
    gefen_step(
        variant,
        block_size,
        &as_f64(&state.weights),
        &gradient
            .iter()
            .map(|value| f64::from(*value) * f64::from(knobs.grad_scale))
            .collect::<Vec<_>>(),
        &as_f64(&state.moment),
        &if state.indices.is_empty() {
            vec![0_u8; state.weights.len()]
        } else {
            state.indices.clone()
        },
        &if state.scales.is_empty() {
            vec![0.0; blocks]
        } else {
            as_f64(&state.scales)
        },
        &as_f64(&state.v),
        f64::from(knobs.learning_rate),
        f64::from(knobs.beta1),
        f64::from(knobs.beta2),
        f64::from(knobs.eps),
        f64::from(knobs.weight_decay),
        iteration,
    )
}

/// The oracle's next step, from the oracle's own previous answer: the F64
/// trajectory advances on F64 state, so it is a trajectory and not ten
/// independent first steps.
fn oracle_from(
    variant: GefenVariant,
    block_size: usize,
    previous: &common::gefen::GefenStep,
    gradient: &[f32],
    knobs: GefenParams,
) -> common::gefen::GefenStep {
    gefen_step(
        variant,
        block_size,
        &previous.weights,
        &gradient
            .iter()
            .map(|value| f64::from(*value) * f64::from(knobs.grad_scale))
            .collect::<Vec<_>>(),
        &previous.moment,
        &previous.indices,
        &previous.scales,
        &previous.second,
        f64::from(knobs.learning_rate),
        f64::from(knobs.beta1),
        f64::from(knobs.beta2),
        f64::from(knobs.eps),
        f64::from(knobs.weight_decay),
        bias_corrected_step(knobs),
    )
}

/// The step count `beta1h` was built at, recovered so the oracle and the
/// coefficients cannot describe different steps.
fn bias_corrected_step(knobs: GefenParams) -> i64 {
    let mut step = 1_i64;
    while step < 64 {
        let expected = 1.0 / (1.0 - f64::from(knobs.beta1).powi(step as i32));
        if (expected - f64::from(knobs.beta1h)).abs() <= 1.0e-6 * expected {
            return step;
        }
        step += 1;
    }
    panic!("no step count produces this bias correction");
}

/// Relative error against a magnitude floor, so a near-zero expected value does
/// not turn a correct answer into an enormous ratio.
fn worst_relative(expected: &[f64], actual: &[f32]) -> f64 {
    assert_eq!(expected.len(), actual.len(), "length mismatch");
    expected
        .iter()
        .zip(actual)
        .map(|(want, got)| {
            let magnitude = want.abs().max(f64::from(*got)).max(1.0e-6);
            (want - f64::from(*got)).abs() / magnitude
        })
        .fold(0.0_f64, f64::max)
}

/// Runs one case on `device` and asserts the whole state against the oracle:
/// the weights, the second moments, and - under `quantized_m` - the scales and
/// every stored index.
fn assert_case(
    label: &str,
    device: Device,
    variant: GefenVariant,
    block_size: usize,
    weights: Vec<f32>,
    gradient: Vec<f32>,
    knobs: GefenParams,
) -> GefenState {
    let state = GefenState::initial(variant, weights, block_size);
    let expected = oracle(variant, block_size, &state, &gradient, knobs);
    let actual = gefen_step_probe(
        device,
        variant,
        block_size as u64,
        &state,
        &gradient,
        knobs,
        1,
    )
    .expect("the device runs both gefen phases");

    let tolerance = match device {
        Device::Cpu => 1.0e-5,
        _ => 1.0e-4,
    };
    assert!(
        worst_relative(&expected.weights, &actual.weights) < tolerance,
        "{label} / {variant} on {device:?}: the weights are not the oracle's \
         ({})",
        worst_relative(&expected.weights, &actual.weights)
    );
    assert!(
        worst_relative(&expected.second, &actual.v) < tolerance,
        "{label} / {variant} on {device:?}: the second moments are not the \
         oracle's ({})",
        worst_relative(&expected.second, &actual.v)
    );
    match variant {
        GefenVariant::SharedV => assert!(
            worst_relative(&expected.moment, &actual.moment) < tolerance,
            "{label} on {device:?}: the first moments are not the oracle's"
        ),
        GefenVariant::QuantizedM => {
            assert!(
                worst_relative(&expected.scales, &actual.scales) < tolerance,
                "{label} on {device:?}: the scales are not the oracle's"
            );
            // The indices are a discrete answer: a tolerance would let a kernel
            // that rounds the other way at every midpoint pass.
            assert_eq!(
                expected.indices, actual.indices,
                "{label} on {device:?}: the stored indices are not the oracle's"
            );
        }
    }
    actual
}

fn devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if retrograd::gpu_runtime_available() {
        devices.push(Device::Gpu);
    }
    devices
}

/// A deterministic spread with no library PRNG, so a failure reproduces from
/// the seed alone.
fn spread(n: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
            let unit = (bits as f32) / ((1_u32 << 24) as f32);
            (unit * 2.0 - 1.0) * range
        })
        .collect()
}

// --- The gradient shapes 4.4 names -------------------------------------------

/// A block whose gradient is exactly zero has no direction to quantize. Its
/// scale is zero, its stored index is the canonical one, and the weights move
/// only by whatever the old moment carried - here, nothing.
#[test]
fn a_zero_gradient_block_stores_the_canonical_index_and_a_zero_scale() {
    for device in devices() {
        for variant in BOTH {
            let n = 96;
            let out = assert_case(
                "zero",
                device,
                variant,
                32,
                spread(n, 7, 0.5),
                vec![0.0; n],
                params(1),
            );
            assert!(
                out.v.iter().all(|value| *value == 0.0),
                "a zero gradient leaves a nonzero second moment on {device:?}"
            );
            if variant == GefenVariant::QuantizedM {
                assert!(
                    out.scales.iter().all(|value| *value == 0.0),
                    "a zero gradient leaves a nonzero scale on {device:?}"
                );
                assert!(
                    out.indices
                        .iter()
                        .all(|code| *code == GEFEN_ZERO_BLOCK_INDEX),
                    "a zero block did not store the canonical index on {device:?}"
                );
            }
        }
    }
}

/// One repeated magnitude: every element of a block quantizes to the same
/// index, and the block's scale is that magnitude scaled by `1 - beta1`.
#[test]
fn a_constant_gradient_quantizes_to_one_index_for_the_whole_block() {
    for device in devices() {
        for variant in BOTH {
            let n = 256;
            let out = assert_case(
                "constant",
                device,
                variant,
                64,
                vec![0.25; n],
                vec![0.5; n],
                params(1),
            );
            if variant == GefenVariant::QuantizedM {
                let first = out.indices[0];
                assert!(
                    out.indices.iter().all(|code| *code == first),
                    "a constant gradient stored more than one index on {device:?}"
                );
            }
        }
    }
}

/// Alternating signs: the block's maximum is the same on both sides, so the
/// codebook is used symmetrically and the two indices straddle its middle.
#[test]
fn an_alternating_gradient_uses_the_codebook_symmetrically() {
    for device in devices() {
        for variant in BOTH {
            let n = 128;
            let gradient: Vec<f32> = (0..n)
                .map(|index| if index % 2 == 0 { 0.75 } else { -0.75 })
                .collect();
            let out = assert_case(
                "alternating",
                device,
                variant,
                32,
                spread(n, 11, 0.1),
                gradient,
                params(1),
            );
            if variant == GefenVariant::QuantizedM {
                let codes: std::collections::BTreeSet<u8> = out.indices.iter().copied().collect();
                assert_eq!(
                    codes.len(),
                    2,
                    "an alternating gradient stored {} distinct indices on {device:?}",
                    codes.len()
                );
                let lowest = *codes.first().expect("a stored index");
                let highest = *codes.last().expect("a stored index");
                assert_eq!(
                    u32::from(lowest) + u32::from(highest),
                    (GEFEN_CODEBOOK_LEVELS - 1) as u32,
                    "the two indices are not symmetric about the codebook's middle"
                );
            }
        }
    }
}

/// One element many orders of magnitude above the rest of its block: the scale
/// follows the largest, so the small ones quantize towards the codebook's
/// middle. This is 4.4's adversarial block, and it is where a shared scale
/// costs the most.
#[test]
fn an_adversarial_block_is_scaled_by_its_largest_element() {
    for device in devices() {
        for variant in BOTH {
            let block = 64;
            let n = 4 * block;
            let mut gradient = vec![1.0e-4_f32; n];
            for index in (0..n).step_by(block) {
                gradient[index] = 1.0e3;
            }
            let out = assert_case(
                "adversarial",
                device,
                variant,
                block,
                spread(n, 13, 0.2),
                gradient,
                params(1),
            );
            if variant == GefenVariant::QuantizedM {
                for offset in (0..n).step_by(block) {
                    let large = out.indices[offset];
                    let small = out.indices[offset + 1];
                    assert_eq!(
                        large,
                        (GEFEN_CODEBOOK_LEVELS - 1) as u8,
                        "the block's largest element is not at the codebook's top"
                    );
                    // The small ones are within one step of the codebook's
                    // middle: a scale that had followed them instead would have
                    // spread them across the whole range.
                    let middle = i32::from(GEFEN_ZERO_BLOCK_INDEX);
                    assert!(
                        (i32::from(small) - middle).abs() <= 1,
                        "a value 10^7 below the block's scale did not land at \
                         the codebook's middle: {small}"
                    );
                }
            }
        }
    }
}

/// Extreme finite values in one block, at both ends of the range whose
/// *square* F32 still holds: the second moment is a mean of squares, so
/// `1e19` is the largest magnitude that keeps the kernel's own arithmetic in
/// range, and an oracle in F64 would otherwise be compared against an F32
/// overflow.
#[test]
fn extreme_finite_values_stay_on_the_oracle() {
    for device in devices() {
        for variant in BOTH {
            let n = 64;
            let mut gradient = spread(n, 17, 1.0);
            gradient[0] = 1.0e19;
            gradient[1] = -1.0e19;
            gradient[2] = 1.0e-19;
            gradient[3] = -1.0e-19;
            let out = assert_case(
                "extreme",
                device,
                variant,
                32,
                spread(n, 19, 1.0),
                gradient,
                params(1),
            );
            assert!(
                out.weights.iter().all(|value| value.is_finite()),
                "an extreme finite gradient produced a nonfinite weight on {device:?}"
            );
        }
    }
}

/// A gradient whose square overflows F32 is where the kernel and an F64 oracle
/// genuinely part company: the block's second moment saturates to infinity, the
/// denominator with it, and the update term goes to zero. What must not happen
/// is a NaN in the weights or in the stored state, because that is the value a
/// checkpoint would carry into every later step.
#[test]
fn a_gradient_whose_square_overflows_leaves_no_nan_behind() {
    for device in devices() {
        for variant in BOTH {
            let n = 64;
            let mut gradient = spread(n, 61, 1.0);
            gradient[0] = 3.0e38;
            gradient[1] = -3.0e38;
            let weights = spread(n, 67, 1.0);
            let state = GefenState::initial(variant, weights.clone(), 32);
            let out = gefen_step_probe(device, variant, 32, &state, &gradient, params(1), 1)
                .expect("the device runs both gefen phases");
            assert!(
                out.weights.iter().all(|value| !value.is_nan()),
                "an overflowing gradient left a NaN weight on {device:?}"
            );
            assert!(
                out.v.iter().all(|value| !value.is_nan()),
                "an overflowing gradient left a NaN second moment on {device:?}"
            );
            // The block holding the overflow is the first one; the others are
            // ordinary and must be untouched by it.
            for (index, (before, after)) in weights.iter().zip(&out.weights).enumerate().skip(32) {
                assert_ne!(
                    before, after,
                    "element {index} of an unaffected block did not move on {device:?}"
                );
            }
        }
    }
}

/// A trailing block shorter than `block_size`: its mean is over the elements it
/// has, so it is not diluted by padding it does not have.
#[test]
fn a_partial_trailing_block_averages_over_the_elements_it_has() {
    for device in devices() {
        for variant in BOTH {
            // Two full blocks and a third holding five elements.
            let block = 32;
            let n = 2 * block + 5;
            let out = assert_case(
                "partial",
                device,
                variant,
                block,
                spread(n, 23, 0.4),
                vec![0.5; n],
                params(1),
            );
            assert_eq!(
                out.v.len(),
                3,
                "a partial block did not get a block of its own"
            );
            // A constant gradient means every full block has the same second
            // moment, and so does the partial one - which is the assertion: a
            // mean over 32 slots with 5 filled would be 5/32 of it.
            let full = f64::from(out.v[0]);
            let partial = f64::from(out.v[2]);
            assert!(
                (full - partial).abs() <= 1.0e-6 * full,
                "the partial block's second moment is {partial} against {full}: \
                 it was averaged over the padding"
            );
        }
    }
}

/// Every one of the 256 codes, stored and read back. The gradient is built so
/// that the recomputed first moments land on each code in turn, which is the
/// only way to cover a codebook exhaustively.
#[test]
fn every_index_code_round_trips_through_the_stored_state() {
    let levels = GEFEN_CODEBOOK_LEVELS as usize;
    let codebook = gefen_codebook();
    for device in devices() {
        // One block holding exactly the codebook: element k is meant to land on
        // code k, so the block's scale is the largest magnitude and each
        // element is that magnitude times the codebook entry.
        let knobs = params(1);
        let gradient: Vec<f32> = (0..levels)
            .map(|code| codebook[code] / (1.0 - knobs.beta1))
            .collect();
        let state = GefenState::initial(GefenVariant::QuantizedM, vec![0.0; levels], levels);
        let out = gefen_step_probe(
            device,
            GefenVariant::QuantizedM,
            levels as u64,
            &state,
            &gradient,
            knobs,
            1,
        )
        .expect("the device runs both gefen phases");

        let scale = f64::from(out.scales[0]);
        let expected: Vec<u8> = gradient
            .iter()
            .map(|value| gefen_code(f64::from(*value) * f64::from(1.0 - knobs.beta1) / scale))
            .collect();
        assert_eq!(
            out.indices, expected,
            "the stored indices are not the codebook's own on {device:?}"
        );
        let distinct: std::collections::BTreeSet<u8> = out.indices.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            levels,
            "only {} of the {levels} codes were reachable on {device:?}",
            distinct.len()
        );
    }
}

/// A step with no gradient at all and a nonzero weight decay moves the weights
/// by exactly the decay factor: the decay is decoupled and reads the *old*
/// weight, so applying it to an already-updated one would show up here.
#[test]
fn a_decay_only_step_multiplies_the_old_weight() {
    for device in devices() {
        for variant in BOTH {
            let n = 64;
            let weights = spread(n, 29, 1.0);
            let mut knobs = params(1);
            knobs.weight_decay = 0.5;
            let state = GefenState::initial(variant, weights.clone(), 32);
            let out = gefen_step_probe(device, variant, 32, &state, &vec![0.0; n], knobs, 1)
                .expect("the device runs both gefen phases");
            let keep = 1.0 - knobs.learning_rate * knobs.weight_decay;
            for (before, after) in weights.iter().zip(&out.weights) {
                let expected = f64::from(*before) * f64::from(keep);
                let magnitude = expected.abs().max(1.0e-6);
                assert!(
                    (expected - f64::from(*after)).abs() / magnitude < 1.0e-5,
                    "a decay-only step on {device:?} did not multiply the old \
                     weight: {before} -> {after}, expected {expected}"
                );
            }
        }
    }
}

/// The global clip scale multiplies the gradient inside the step, so a run that
/// clipped by half and a run given half the gradient are the same step.
#[test]
fn the_clip_scale_is_applied_to_the_gradient_the_step_reads() {
    for device in devices() {
        for variant in BOTH {
            let n = 64;
            let weights = spread(n, 31, 0.3);
            let gradient = spread(n, 37, 2.0);
            let halved: Vec<f32> = gradient.iter().map(|value| value * 0.5).collect();

            let mut clipped = params(1);
            clipped.grad_scale = 0.5;
            let state = GefenState::initial(variant, weights, 32);
            let scaled = gefen_step_probe(device, variant, 32, &state, &gradient, clipped, 1)
                .expect("the device runs both gefen phases");
            let prescaled = gefen_step_probe(device, variant, 32, &state, &halved, params(1), 1)
                .expect("the device runs both gefen phases");
            assert_eq!(
                scaled.weights, prescaled.weights,
                "clipping by half is not the same step as half the gradient on {device:?}"
            );
        }
    }
}

/// Bias correction is a function of the step count and nothing else: the same
/// state and gradient at step 1 and at step 10 differ by the ratio of the two
/// corrections, and the correction at a late step is one.
#[test]
fn the_bias_correction_is_the_step_counts_and_no_other_state() {
    for device in devices() {
        for variant in BOTH {
            let n = 64;
            let weights = vec![0.0_f32; n];
            let gradient = spread(n, 41, 1.0);
            let state = GefenState::initial(variant, weights, 32);

            let early = gefen_step_probe(device, variant, 32, &state, &gradient, params(1), 1)
                .expect("the device runs both gefen phases");
            let late = gefen_step_probe(device, variant, 32, &state, &gradient, params(10000), 1)
                .expect("the device runs both gefen phases");

            // From a zero state, the two differ only through beta1h/beta2h, and
            // the late step's corrections are both one.
            let knobs = params(1);
            let ratio = f64::from(knobs.beta1h) / f64::from(knobs.beta2h).sqrt();
            for (early_value, late_value) in early.weights.iter().zip(&late.weights) {
                let expected = f64::from(*late_value) * ratio;
                let magnitude = expected.abs().max(1.0e-6);
                assert!(
                    (expected - f64::from(*early_value)).abs() / magnitude < 1.0e-4,
                    "the two steps do not differ by their bias corrections on {device:?}"
                );
            }
        }
    }
}

/// Ten steps over the same gradient, against ten oracle steps. A state that
/// survives one step and drifts over ten is a state whose round trip is wrong,
/// and the quantized variant is where that would show.
#[test]
fn ten_steps_stay_on_the_oracles_trajectory() {
    for device in devices() {
        for variant in BOTH {
            let block = 64;
            let n = 4 * block;
            let gradient = spread(n, 47, 1.0);
            let start = GefenState::initial(variant, spread(n, 43, 0.5), block);

            let mut state = start.clone();
            let mut expected = oracle(variant, block, &start, &gradient, params(1));
            for step in 2..=10_u32 {
                state = gefen_step_probe(
                    device,
                    variant,
                    block as u64,
                    &state,
                    &gradient,
                    params(step - 1),
                    1,
                )
                .expect("the device runs both gefen phases");
                expected = oracle_from(variant, block, &expected, &gradient, params(step));
            }
            state = gefen_step_probe(
                device,
                variant,
                block as u64,
                &state,
                &gradient,
                params(10),
                1,
            )
            .expect("the device runs both gefen phases");

            // Over ten steps the comparison that means something is how far
            // the F32 trajectory is from the F64 one *relative to how far it
            // moved*: a per-element ratio would be dominated by whichever
            // weight happened to pass near zero, which says nothing about the
            // kernel.
            let deviation = expected
                .weights
                .iter()
                .zip(&state.weights)
                .map(|(want, got)| (want - f64::from(*got)).abs())
                .fold(0.0_f64, f64::max);
            let displacement = expected
                .weights
                .iter()
                .zip(&start.weights)
                .map(|(want, from)| (want - f64::from(*from)).abs())
                .fold(0.0_f64, f64::max);
            assert!(displacement > 1.0e-3, "ten steps moved nothing to compare");
            let drift = deviation / displacement;
            let tolerance = match device {
                Device::Cpu => 1.0e-4,
                _ => 1.0e-3,
            };
            assert!(
                drift < tolerance,
                "{variant} on {device:?}: ten steps left the oracle's trajectory                  by {deviation:e} against a displacement of {displacement:e}"
            );
        }
    }
}

// --- Backend parity ----------------------------------------------------------

/// The same inputs on the CPU and on the GPU. Not a second tolerance against
/// the oracle: the two kernels see one state and one gradient, so anything they
/// disagree about is the kernels disagreeing.
#[test]
fn the_gpu_kernels_are_the_cpu_kernels_on_the_same_inputs() {
    if !retrograd::gpu_runtime_available() {
        eprintln!("skipping: no GPU runtime to compare against");
        return;
    }
    for variant in BOTH {
        let block = 128;
        // Three full blocks and a partial fourth, so the reduction tail and the
        // block boundary are both inside the comparison.
        let n = 3 * block + 37;
        let state = GefenState::initial(variant, spread(n, 53, 0.5), block);
        let gradient = spread(n, 59, 1.5);
        let knobs = params(3);

        let cpu = gefen_step_probe(
            Device::Cpu,
            variant,
            block as u64,
            &state,
            &gradient,
            knobs,
            4,
        )
        .expect("the CPU runs both gefen phases");
        let gpu = gefen_step_probe(
            Device::Gpu,
            variant,
            block as u64,
            &state,
            &gradient,
            knobs,
            4,
        )
        .expect("the GPU runs both gefen phases");

        // Normalized by how far the four steps moved the weights, not by each
        // weight's own size: a per-element ratio would be dominated by whichever
        // coordinate happened to pass near zero, which says nothing about
        // whether the two kernels agree.
        let deviation = cpu
            .weights
            .iter()
            .zip(&gpu.weights)
            .map(|(left, right)| f64::from(*left - *right).abs())
            .fold(0.0_f64, f64::max);
        let displacement = cpu
            .weights
            .iter()
            .zip(&state.weights)
            .map(|(after, before)| f64::from(*after - *before).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            displacement > 1.0e-6,
            "the four steps moved nothing to compare"
        );
        assert!(
            deviation / displacement < 1.0e-3,
            "{variant}: the GPU weights differ from the CPU's by {deviation:e}              against a displacement of {displacement:e}"
        );
        // The indices are the part a different reduction order can move, and
        // the part a checkpoint carries across backends. A single code apart on
        // a handful of elements is the rounding boundary; a wholesale
        // disagreement is a different algorithm.
        if variant == GefenVariant::QuantizedM {
            let apart = cpu
                .indices
                .iter()
                .zip(&gpu.indices)
                .filter(|(left, right)| left != right)
                .count();
            let far = cpu
                .indices
                .iter()
                .zip(&gpu.indices)
                .filter(|(left, right)| i32::from(**left).abs_diff(i32::from(**right)) > 1)
                .count();
            assert_eq!(
                far, 0,
                "{apart} indices differ, {far} of them by more than one code"
            );
        }
    }
}

/// The support predicate and the probe agree in both directions: a device that
/// declares the step runs it, and a device that declines refuses rather than
/// answering from a fallback backend.
#[test]
fn the_device_predicate_and_the_probe_give_the_same_answer() {
    for device in devices() {
        for variant in BOTH {
            let declared = gefen_step_supported(device, variant, 128)
                .expect("the device answers whether it carries the step");
            let n = 256;
            let state = GefenState::initial(variant, vec![0.25_f32; n], 128);
            let outcome = gefen_step_probe(
                device,
                variant,
                128,
                &state,
                &vec![0.5_f32; n],
                params(1),
                1,
            );
            assert_eq!(
                declared,
                outcome.is_ok(),
                "{variant} on {device:?}: the predicate says {declared} and the \
                 probe says {}",
                outcome.is_ok()
            );
        }
    }
}
