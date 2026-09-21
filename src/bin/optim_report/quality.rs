//! Gefen's approximation quality, measured against the update it approximates.
//!
//! Correctness and quality are separate questions and this file only asks the
//! second: every figure here compares Gefen's update with AdamW's on *the same*
//! weights, gradient and coefficients, which is the only comparison in which a
//! difference is the approximation and not the arithmetic.
//!
//! A universal cosine above 0.99 is not the target and would not be evidence:
//! a shared second moment changes coordinate scaling by design, so the
//! interesting number is how the cosine moves with the block size, and where
//! the error comes from when it moves.

use retrograd::{
    Device, GEFEN_CODEBOOK_LEVELS, GefenParams, GefenState, GefenVariant, gefen_codebook,
    gefen_step_probe, gefen_step_supported,
};

/// The block sizes the report walks. One element per block is AdamW's own
/// second moment, so it is the anchor the others are read against; 1024 is the
/// frozen v1 default.
const BLOCK_SIZES: [usize; 5] = [1, 64, 256, 1024, 4096];

/// Large enough that 4096-element blocks are not one block, and that a cosine
/// over it is not dominated by a handful of coordinates.
const N_ELEMENTS: usize = 1 << 16;

/// Steps of the synthetic trajectory. The quantized first moment's error
/// compounds through its own state, so a single step would measure the best
/// case and call it the figure.
const N_STEPS: u32 = 16;

pub fn run(device: Device) {
    println!("\n=== Gefen approximation quality ===");
    println!(
        "  {N_ELEMENTS} elements, {N_STEPS} steps, device {device:?}, \
         gradient: a stationary log-normal magnitude with a drifting sign"
    );

    let gradients: Vec<Vec<f32>> = (0..N_STEPS)
        .map(|step| gradient(N_ELEMENTS, 0x51ED_0000 + u64::from(step)))
        .collect();
    let weights = spread(N_ELEMENTS, 0xBEEF, 0.05);

    println!(
        "\n  {:<12} {:>7} {:>11} {:>11} {:>11} {:>11} {:>10}",
        "variant", "block", "cosine", "rel.error", "recon.err", "v.error", "bytes/par"
    );
    for variant in [GefenVariant::SharedV, GefenVariant::QuantizedM] {
        for block in BLOCK_SIZES {
            if !gefen_step_supported(device, variant, block as u64).unwrap_or(false) {
                println!("  {:<12} {block:>7} {:>11}", variant.as_str(), "declined");
                continue;
            }
            let row = measure(device, variant, block, &weights, &gradients);
            println!(
                "  {:<12} {block:>7} {:>11.6} {:>11.3e} {:>11.3e} {:>11.3e} {:>10.4}",
                variant.as_str(),
                row.cosine,
                row.relative_error,
                row.reconstruction_error,
                row.second_moment_error,
                row.bytes_per_parameter
            );
        }
    }
    println!(
        "\n  cosine      of Gefen's update against AdamW's, over the whole parameter\n  \
           rel.error   ||gefen - adamw|| / ||adamw||, the same update\n  \
           recon.err   ||decode(stored) - m|| / ||m||, the quantized first moment alone\n  \
           v.error     mean |sqrt(v_block) - sqrt(v_element)| / sqrt(v_element), the \
           coordinate\n              scaling a shared second moment gives up by design\n  \
           bytes/par   persistent state per eligible parameter, the shared codebook \
           excluded"
    );

    adversarial(device);
}

struct Row {
    cosine: f64,
    relative_error: f64,
    reconstruction_error: f64,
    second_moment_error: f64,
    bytes_per_parameter: f64,
}

fn measure(
    device: Device,
    variant: GefenVariant,
    block: usize,
    weights: &[f32],
    gradients: &[Vec<f32>],
) -> Row {
    let n = weights.len();
    let blocks = n.div_ceil(block);
    let mut state = GefenState::initial(variant, weights.to_vec(), block);
    let mut reference = AdamW::new(weights);

    // The two run the same trajectory: each reads the weights the *same*
    // optimizer produced on the previous step, so the comparison is between two
    // trajectories rather than between two single steps off a shared state.
    let mut gefen_before = weights.to_vec();
    let mut adamw_before: Vec<f64> = weights.iter().map(|value| f64::from(*value)).collect();
    let mut gefen_update = Vec::new();
    let mut adamw_update = Vec::new();

    for (index, gradient) in gradients.iter().enumerate() {
        let step = index as u32 + 1;
        let knobs = GefenParams::at_step(1.0e-3, 0.9, 0.999, 1.0e-8, step);
        gefen_before.clone_from(&state.weights);
        state = gefen_step_probe(device, variant, block as u64, &state, gradient, knobs, 1)
            .expect("the device runs both gefen phases");
        gefen_update = state
            .weights
            .iter()
            .zip(&gefen_before)
            .map(|(after, before)| f64::from(*after) - f64::from(*before))
            .collect();

        adamw_before.clone_from(&reference.weights);
        reference.step(gradient, knobs);
        adamw_update = reference
            .weights
            .iter()
            .zip(&adamw_before)
            .map(|(after, before)| *after - *before)
            .collect();
    }

    // The reconstruction error of the state that survived the last step: what a
    // decode of the stored bytes gives back, against the first moment the step
    // actually used. Under shared_v nothing is quantized and it is exactly zero.
    let reconstruction_error = match variant {
        GefenVariant::SharedV => 0.0,
        GefenVariant::QuantizedM => {
            let codebook = gefen_codebook();
            let decoded: Vec<f64> = (0..n)
                .map(|index| {
                    f64::from(state.scales[index / block])
                        * f64::from(codebook[usize::from(state.indices[index])])
                })
                .collect();
            relative_norm(&decoded, &reference.moment)
        }
    };

    // The coordinate scaling a shared second moment gives up: the block's
    // denominator against the per-element one AdamW keeps.
    let second_moment_error = {
        let mut total = 0.0;
        for index in 0..n {
            let element = reference.second[index].max(0.0).sqrt();
            let shared = f64::from(state.v[index / block]).max(0.0).sqrt();
            total += (shared - element).abs() / element.max(1.0e-12);
        }
        total / n as f64
    };

    Row {
        cosine: cosine(&gefen_update, &adamw_update),
        relative_error: relative_norm(&gefen_update, &adamw_update),
        reconstruction_error,
        second_moment_error,
        bytes_per_parameter: bytes_per_parameter(variant, n, blocks),
    }
}

/// 4.4's adversarial block: one element many orders of magnitude above the rest
/// of its block, which is where a shared scale costs the most. Reported apart
/// because averaging it into a uniform gradient would hide exactly the case it
/// exists to expose.
fn adversarial(device: Device) {
    println!("\n  --- an adversarial block: one element 10^6 above its neighbours ---");
    let block = 1024;
    let n = 16 * block;
    let weights = spread(n, 0xADD5, 0.05);
    let mut gradient = spread(n, 0x1EAF, 1.0e-4);
    for index in (0..n).step_by(block) {
        gradient[index] = 1.0e2;
    }
    let knobs = GefenParams::at_step(1.0e-3, 0.9, 0.999, 1.0e-8, 1);

    let mut reference = AdamW::new(&weights);
    reference.step(&gradient, knobs);
    let adamw: Vec<f64> = reference
        .weights
        .iter()
        .zip(&weights)
        .map(|(after, before)| *after - f64::from(*before))
        .collect();

    println!(
        "  {:<12} {:>11} {:>11} {:>14} {:>14}",
        "variant", "cosine", "rel.error", "cos(large)", "cos(small)"
    );
    for variant in [GefenVariant::SharedV, GefenVariant::QuantizedM] {
        if !gefen_step_supported(device, variant, block as u64).unwrap_or(false) {
            continue;
        }
        let state = GefenState::initial(variant, weights.clone(), block);
        let after = gefen_step_probe(device, variant, block as u64, &state, &gradient, knobs, 1)
            .expect("the device runs both gefen phases");
        let update: Vec<f64> = after
            .weights
            .iter()
            .zip(&weights)
            .map(|(after, before)| f64::from(*after) - f64::from(*before))
            .collect();

        // The whole-parameter cosine is dominated by the many small
        // coordinates; split it so the outlier's own direction is visible.
        let large: Vec<usize> = (0..n).step_by(block).collect();
        let small: Vec<usize> = (0..n).filter(|index| index % block != 0).collect();
        println!(
            "  {:<12} {:>11.6} {:>11.3e} {:>14.6} {:>14.6}",
            variant.as_str(),
            cosine(&update, &adamw),
            relative_norm(&update, &adamw),
            cosine_at(&update, &adamw, &large),
            cosine_at(&update, &adamw, &small)
        );
    }
}

/// AdamW in F64, the update Gefen approximates. Written here rather than read
/// off a run: the comparison has to be against the same gradient and the same
/// coefficients, and a run would supply neither.
struct AdamW {
    weights: Vec<f64>,
    moment: Vec<f64>,
    second: Vec<f64>,
}

impl AdamW {
    /// From the same weights Gefen starts at: the two trajectories are only
    /// comparable if their first step reads the same parameter.
    fn new(weights: &[f32]) -> Self {
        Self {
            weights: weights.iter().map(|value| f64::from(*value)).collect(),
            moment: vec![0.0; weights.len()],
            second: vec![0.0; weights.len()],
        }
    }

    fn step(&mut self, gradient: &[f32], knobs: GefenParams) {
        let alpha = f64::from(knobs.learning_rate);
        let beta1 = f64::from(knobs.beta1);
        let beta2 = f64::from(knobs.beta2);
        let eps = f64::from(knobs.eps);
        let keep = 1.0 - alpha * f64::from(knobs.weight_decay);
        let beta1h = f64::from(knobs.beta1h);
        let beta2h = f64::from(knobs.beta2h);
        for (index, value) in gradient.iter().enumerate() {
            let g = f64::from(*value) * f64::from(knobs.grad_scale);
            self.moment[index] = beta1 * self.moment[index] + (1.0 - beta1) * g;
            self.second[index] = beta2 * self.second[index] + (1.0 - beta2) * g * g;
            let denominator = (self.second[index] * beta2h).sqrt() + eps;
            self.weights[index] =
                self.weights[index] * keep - alpha * (self.moment[index] * beta1h) / denominator;
        }
    }
}

/// The persistent state one eligible parameter costs, the shared codebook
/// excluded: it is one allocation per owner and does not scale with the
/// parameter count, so folding it in would make the ratio depend on the model.
fn bytes_per_parameter(variant: GefenVariant, n: usize, blocks: usize) -> f64 {
    let bytes = match variant {
        // F32 first moment per element, F32 second moment per block.
        GefenVariant::SharedV => 4 * n + 4 * blocks,
        // One byte per element, F32 scale and second moment per block.
        GefenVariant::QuantizedM => n + 8 * blocks,
    };
    bytes as f64 / n as f64
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    cosine_at(left, right, &(0..left.len()).collect::<Vec<_>>())
}

fn cosine_at(left: &[f64], right: &[f64], indices: &[usize]) -> f64 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in indices {
        dot += left[*index] * right[*index];
        left_norm += left[*index] * left[*index];
        right_norm += right[*index] * right[*index];
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    // A zero update has no direction; reporting 1.0 would claim agreement that
    // was never tested.
    if denominator <= 0.0 {
        return f64::NAN;
    }
    dot / denominator
}

fn relative_norm(left: &[f64], right: &[f64]) -> f64 {
    let mut difference = 0.0;
    let mut magnitude = 0.0;
    for (a, b) in left.iter().zip(right) {
        difference += (a - b) * (a - b);
        magnitude += b * b;
    }
    if magnitude <= 0.0 {
        return f64::NAN;
    }
    (difference / magnitude).sqrt()
}

/// A gradient whose magnitudes span several orders of magnitude, which is what
/// a real one does and what makes a shared block scale a question at all. A
/// uniform gradient would flatter every block size equally.
fn gradient(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (bits as f64) / f64::from(1_u32 << 24)
    };
    (0..n)
        .map(|_| {
            let magnitude = 10.0_f64.powf(-4.0 + 4.0 * next());
            let sign = if next() < 0.5 { -1.0 } else { 1.0 };
            (sign * magnitude) as f32
        })
        .collect()
}

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

/// The codebook has no exact zero, so a decode never returns one: stated here
/// because the reconstruction error above is read against that fact.
#[allow(dead_code)]
const CODEBOOK_LEVELS: u64 = GEFEN_CODEBOOK_LEVELS;
