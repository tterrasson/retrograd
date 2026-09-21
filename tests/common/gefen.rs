//! The fixed-block Gefen algorithm written straight, in slow F64.
//!
//! The oracle of every Gefen assertion in this repository, shared rather than
//! repeated: one written for the model-driven step and another for the
//! op-driven one would be two definitions of the algorithm, and a test suite
//! that agrees with itself while disagreeing with the specification.

use retrograd::{GEFEN_CODEBOOK_LEVELS, GEFEN_ZERO_BLOCK_INDEX, GefenVariant};

/// One Gefen step over one parameter, in slow F64: the new weights, the new
/// first-moment state and the new per-block scales and second moments.
pub struct GefenStep {
    pub weights: Vec<f64>,
    pub moment: Vec<f64>,
    pub indices: Vec<u8>,
    pub scales: Vec<f64>,
    pub second: Vec<f64>,
}

pub fn gefen_codebook() -> Vec<f64> {
    let last = (GEFEN_CODEBOOK_LEVELS - 1) as f64;
    (0..GEFEN_CODEBOOK_LEVELS)
        .map(|index| -1.0 + 2.0 * index as f64 / last)
        .collect()
}

/// Nearest entry, ties to the lower index, clamped. Bytes, never signed
/// offsets.
pub fn gefen_code(value: f64) -> u8 {
    let last = (GEFEN_CODEBOOK_LEVELS - 1) as f64;
    let position = (value + 1.0) * 0.5 * last;
    // NaN has no nearest entry to speak of, and lands on the floor with
    // everything else below the codebook's first value.
    if position.is_nan() || position <= 0.0 {
        return 0;
    }
    if position >= last {
        return (GEFEN_CODEBOOK_LEVELS - 1) as u8;
    }
    (position - 0.5).ceil() as u8
}

#[allow(clippy::too_many_arguments)]
pub fn gefen_step(
    variant: GefenVariant,
    block_size: usize,
    weights: &[f64],
    gradient: &[f64],
    moment: &[f64],
    indices: &[u8],
    scales: &[f64],
    second: &[f64],
    alpha: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    iteration: i64,
) -> GefenStep {
    let codebook = gefen_codebook();
    let n = weights.len();
    let blocks = n.div_ceil(block_size);
    let beta1h = 1.0 / (1.0 - beta1.powi(iteration as i32));
    let beta2h = 1.0 / (1.0 - beta2.powi(iteration as i32));
    let keep = 1.0 - alpha * weight_decay;

    let mut out = GefenStep {
        weights: weights.to_vec(),
        moment: moment.to_vec(),
        indices: indices.to_vec(),
        scales: scales.to_vec(),
        second: second.to_vec(),
    };

    for block in 0..blocks {
        let first = block * block_size;
        let last = (first + block_size).min(n);
        // Phase A, pure: the new scale and the new second moment of the whole
        // block, read against the old state alone.
        let mut sum_sq = 0.0;
        let mut max_abs = 0.0_f64;
        let recomputed: Vec<f64> = (first..last)
            .map(|index| {
                let g = gradient[index];
                sum_sq += g * g;
                match variant {
                    GefenVariant::SharedV => beta1 * moment[index] + (1.0 - beta1) * g,
                    GefenVariant::QuantizedM => {
                        let decoded = scales[block] * codebook[usize::from(indices[index])];
                        beta1 * decoded + (1.0 - beta1) * g
                    }
                }
            })
            .collect();
        if variant == GefenVariant::QuantizedM {
            for value in &recomputed {
                max_abs = max_abs.max(value.abs());
            }
        }
        // The mean is over the block's actual elements, so a partial trailing
        // block is not diluted by padding it does not have.
        let mean_sq = sum_sq / (last - first) as f64;
        let v_new = beta2 * second[block] + (1.0 - beta2) * mean_sq;
        let denominator = (v_new * beta2h).sqrt() + eps;

        // Phase B: the update uses the unquantized recomputed moment and
        // stores its representation for the next step.
        for (offset, index) in (first..last).enumerate() {
            let m = recomputed[offset];
            out.weights[index] = weights[index] * keep - alpha * (m * beta1h) / denominator;
            match variant {
                GefenVariant::SharedV => out.moment[index] = m,
                GefenVariant::QuantizedM => {
                    out.indices[index] = if max_abs > 0.0 {
                        gefen_code(m / max_abs)
                    } else {
                        GEFEN_ZERO_BLOCK_INDEX
                    };
                }
            }
        }
        if variant == GefenVariant::QuantizedM {
            out.scales[block] = max_abs;
        }
        out.second[block] = v_new;
    }
    out
}
