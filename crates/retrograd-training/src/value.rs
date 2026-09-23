//! Linear-probe value head: the PPO critic.
//!
//! A single linear layer `V(s) = w·h(s) + b` over the frozen final-layer
//! hidden states extracted by the runtime, trained in pure Rust with
//! full-batch Adam on a mean-squared-error regression toward the TD(lambda)
//! returns. The baseline only reduces the variance of the policy gradient,
//! its quality never biases the PPO update - so a modest probe is a sound
//! starting point, and its closed-form training target makes it exactly
//! testable.

use crate::features::FeatureStore;
use retrograd_core::{Error, Result};

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

#[derive(Clone, Debug)]
pub struct ValueHead {
    dim: usize,
    weights: Vec<f32>,
    bias: f32,
    m_weights: Vec<f32>,
    v_weights: Vec<f32>,
    m_bias: f32,
    v_bias: f32,
    step: u64,
}

impl ValueHead {
    /// Zero initialization: the head predicts 0 everywhere, so before its
    /// first fit the PPO advantages reduce to plain (whitened) returns.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            weights: vec![0.0; dim],
            bias: 0.0,
            m_weights: vec![0.0; dim],
            v_weights: vec![0.0; dim],
            m_bias: 0.0,
            v_bias: 0.0,
            step: 0,
        }
    }

    /// `V(s)` for one feature row.
    pub fn predict(&self, features: &[f32]) -> f32 {
        debug_assert_eq!(features.len(), self.dim);
        self.bias + dot(&self.weights, features)
    }

    /// `V(s)` for `features.len() / dim` row-major feature rows.
    pub fn predict_rows(&self, features: &[f32]) -> Vec<f32> {
        features
            .chunks_exact(self.dim)
            .map(|row| self.predict(row))
            .collect()
    }

    /// Full-batch Adam on the MSE toward `targets` (one per feature row).
    /// Returns the post-fit MSE, the critic's own training diagnostic.
    pub fn fit(&mut self, features: &[f32], targets: &[f32], lr: f32, epochs: u32) -> Result<f32> {
        let mut store = FeatureStore::with_capacity(
            self.dim,
            retrograd_core::FeatureDtype::F32,
            targets.len(),
        )?;
        store.push_rows(features)?;
        self.fit_store(&mut store, targets, lr, epochs)
    }

    /// [`Self::fit`] over a [`FeatureStore`], which is what PPO calls.
    ///
    /// Identical arithmetic to the slice form: the gradient is accumulated over
    /// every row before a single Adam update per epoch, so the result does not
    /// depend on how the rows were chunked. Only the rows' *storage* precision can
    /// change what this computes, and that is the store's decision, not this
    /// function's.
    pub fn fit_store(
        &mut self,
        features: &mut FeatureStore,
        targets: &[f32],
        lr: f32,
        epochs: u32,
    ) -> Result<f32> {
        if self.dim == 0 || features.rows() != targets.len() {
            return Err(Error::invalid("value head features do not match targets"));
        }
        if targets.is_empty() {
            return Err(Error::invalid("value head fit requires at least one row"));
        }
        if !(lr > 0.0 && lr.is_finite()) {
            return Err(Error::invalid("value head learning rate must be positive"));
        }
        let n = targets.len() as f32;
        let dim = self.dim;
        let mut grad_weights = vec![0.0_f32; self.dim];
        for _ in 0..epochs {
            grad_weights.fill(0.0);
            let mut grad_bias = 0.0_f32;
            // `head` is an immutable reborrow so the closure can call `predict`
            // while `features` is borrowed mutably and the two gradient
            // accumulators are borrowed mutably - three disjoint paths, which the
            // borrow checker only sees once `self` is named separately.
            let head = &*self;
            features.for_each_chunk(|first_row, rows| {
                for (row, &target) in rows.chunks_exact(dim).zip(&targets[first_row..]) {
                    let error = 2.0 * (head.predict(row) - target) / n;
                    grad_bias += error;
                    for (grad, &x) in grad_weights.iter_mut().zip(row) {
                        *grad += error * x;
                    }
                }
                Ok::<_, Error>(())
            })?;

            self.step += 1;
            let bias_correction1 = 1.0 - BETA1.powi(self.step as i32);
            let bias_correction2 = 1.0 - BETA2.powi(self.step as i32);
            let adam = |param: &mut f32, m: &mut f32, v: &mut f32, grad: f32| {
                *m = BETA1 * *m + (1.0 - BETA1) * grad;
                *v = BETA2 * *v + (1.0 - BETA2) * grad * grad;
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                *param -= lr * m_hat / (v_hat.sqrt() + EPSILON);
            };
            for (((weight, m), v), &grad) in self
                .weights
                .iter_mut()
                .zip(&mut self.m_weights)
                .zip(&mut self.v_weights)
                .zip(&grad_weights)
            {
                adam(weight, m, v, grad);
            }
            adam(
                &mut self.bias,
                &mut self.m_bias,
                &mut self.v_bias,
                grad_bias,
            );
        }

        let mut squared_error = 0.0_f32;
        let head = &*self;
        features.for_each_chunk(|first_row, rows| {
            for (row, &target) in rows.chunks_exact(dim).zip(&targets[first_row..]) {
                let diff = head.predict(row) - target;
                squared_error += diff * diff;
            }
            Ok::<_, Error>(())
        })?;
        let mse = squared_error / n;
        if !mse.is_finite() {
            return Err(Error::invalid(
                "value head diverged (non-finite loss); lower ppo.value_lr",
            ));
        }
        Ok(mse)
    }
}

/// Independent partial sums of [`dot`]: enough to fill one 256-bit register,
/// or two NEON ones.
const DOT_LANES: usize = 8;

/// `a · b` over `DOT_LANES` independent accumulators.
///
/// A single running `f32` sum is a serial dependency chain the compiler may
/// not reorder, so it runs one add per FP latency; the critic evaluates this
/// once per completion state per epoch, over the full hidden width, which makes
/// it the fit's inner loop. Fixed lanes and a fixed reduction tree keep the
/// result deterministic - it rounds differently from the serial sum, not
/// differently from run to run.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a_chunks, a_tail) = a.as_chunks::<DOT_LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<DOT_LANES>();
    let tail = a_tail.iter().zip(b_tail).map(|(&x, &y)| x * y).sum::<f32>();
    let mut lanes = [0.0_f32; DOT_LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..DOT_LANES {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
        + tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_core::FeatureDtype;

    #[test]
    fn dot_covers_the_lanes_and_the_tail() {
        // Integers are exact in f32, so any dropped or doubled term shows.
        for len in [0, 1, 7, 8, 9, 16, 4099] {
            let a = (0..len).map(|i| (i % 13) as f32 - 6.0).collect::<Vec<_>>();
            let b = (0..len).map(|i| (i % 5) as f32 + 1.0).collect::<Vec<_>>();
            let expected = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
            assert_eq!(dot(&a, &b), expected, "len {len}");
        }
    }

    #[test]
    fn a_narrow_store_fits_identically_on_values_it_represents_exactly() {
        // Halves and quarters are exact in binary16, so the only thing left that
        // could differ between the two stores is the chunked accumulation - and
        // it must not, because the gradient is summed over every row before a
        // single Adam update.
        let features: Vec<f32> = (0..24).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
        let targets: Vec<f32> = (0..12).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
        let mut wide = FeatureStore::with_capacity(2, FeatureDtype::F32, 12).unwrap();
        let mut narrow = FeatureStore::with_capacity(2, FeatureDtype::F16, 12).unwrap();
        wide.push_rows(&features).unwrap();
        narrow.push_rows(&features).unwrap();

        let mut wide_head = ValueHead::new(2);
        let mut narrow_head = ValueHead::new(2);
        let wide_mse = wide_head.fit_store(&mut wide, &targets, 0.05, 25).unwrap();
        let narrow_mse = narrow_head
            .fit_store(&mut narrow, &targets, 0.05, 25)
            .unwrap();
        assert_eq!(wide_mse.to_bits(), narrow_mse.to_bits());
        assert_eq!(wide_head.weights, narrow_head.weights);
        assert_eq!(wide_head.bias.to_bits(), narrow_head.bias.to_bits());
        // Halving the storage is the whole point of the narrow store.
        assert_eq!(narrow.allocated_bytes() * 2, wide.allocated_bytes());
    }

    #[test]
    fn a_narrow_store_stays_a_usable_critic_on_values_it_rounds() {
        // Rounded features are still a sound regression: the fit must converge to
        // the same function within the format's own precision, not diverge.
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 0..64 {
            let x0 = (i % 8) as f32 * 0.137 - 0.5;
            let x1 = (i / 8) as f32 * 0.211 - 0.7;
            features.extend_from_slice(&[x0, x1]);
            targets.push(2.0 * x0 - x1 + 0.5);
        }
        let mut narrow = FeatureStore::with_capacity(2, FeatureDtype::F16, 64).unwrap();
        narrow.push_rows(&features).unwrap();
        let mut head = ValueHead::new(2);
        let mse = head.fit_store(&mut narrow, &targets, 0.05, 2000).unwrap();
        assert!(mse < 1e-4, "mse {mse}");
        assert!((head.predict(&[0.25, -0.5]) - (2.0 * 0.25 + 0.5 + 0.5)).abs() < 0.05);
    }
}
