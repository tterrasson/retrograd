//! Which base-weight precisions this build trains, and on what.
//!
//! F32 is the floor and is deliberately not a row here: a row is a measured
//! claim, and F32 is the baseline those claims are measured against.
//!
//! Anything wider is admitted per (dtype, optimizer, backend) combination: the
//! model file decides the dtype, the update step is a kernel with its own
//! dtype table, and each backend reimplements that kernel. A row records the
//! evidence that admitted it, and a lane reads its tolerances from the row
//! instead of restating them.
//!
//! The backend column names the ggml registry ("CPU", "MTL", "CUDA",
//! "Vulkan", as `ggml_backend_reg_name` spells them), not a device name. The
//! table declares; the runtime asks the live device whether it can run the
//! step, and `tests/f16_base_training.rs` checks the two agree.

use crate::optimizer::OptimizerKind;
use crate::trainable::TensorDtype;

/// One admitted (dtype, optimizer, backend) combination, with its evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseDtypeCapability {
    /// The dtype as [`TensorDtype::name`] spells it. Never `"F32"`.
    pub dtype: &'static str,
    /// The optimizer whose update step writes a parameter of that dtype.
    pub optimizer: OptimizerKind,
    /// The ggml registry the kernel belongs to.
    pub backend: &'static str,
    /// Largest relative per-element difference the parity lane allows between
    /// this combination's gradient and the F32 run's on the fixture.
    /// Gradients are F32 in both runs; the weights the forward read differ.
    pub gradient_tolerance: f32,
    /// Bound on the per-element parameter delta of one step, in units of the
    /// F16 grid the parameter is stored on. A quantized trajectory cannot land
    /// between grid points, so one grid point is the closest agreement the
    /// storage allows; a relative error would count a correct rounding near
    /// zero as an error of one.
    pub update_tolerance: f32,
    /// Fraction of elements allowed to exceed [`Self::update_tolerance`].
    ///
    /// Measured, not slack: near the optimizer's epsilon an absolute gradient
    /// difference of 1e-7 (meaningless against the tensor's scale) is the gap
    /// between a full step and almost none, and those elements form a real
    /// minority. What is bounded for every element, asserted separately, is
    /// that the divergence never exceeds one whole optimizer step.
    pub update_outlier_fraction: f32,
    /// Steps the stability lane ran before this row was written. Zero is not
    /// allowed: a row with no long lane is a row nobody measured.
    pub stability_steps: u32,
    /// Largest relative gap allowed between this combination's loss after
    /// [`Self::stability_steps`] steps and the F32 run's on the same data. Not
    /// a bit-parity bound, which a long normalized run cannot have: it bounds
    /// the outcome, so a run that quietly stopped training is caught.
    pub stability_loss_tolerance: f32,
}

/// Every base dtype beyond F32 this build admits. One row per combination a
/// lane has actually executed; adding a backend means running the lane, not
/// editing a string.
pub const BASE_DTYPE_TABLE: &[BaseDtypeCapability] = &[BaseDtypeCapability {
    dtype: "F16",
    optimizer: OptimizerKind::AdamW,
    backend: "CPU",
    // An F16 weight carries 11 significant bits, so the forward it feeds
    // differs from the F32 forward by ~5e-4 per operand; the gradient is
    // allowed an order of magnitude more than that.
    gradient_tolerance: 5.0e-2,
    update_tolerance: 1.0,
    // Measured 1.8e-3 over 24576 elements; the headroom is deliberate and
    // modest, so a doubled outlier population fails the lane.
    update_outlier_fraction: 5.0e-3,
    stability_steps: 2000,
    stability_loss_tolerance: 2.0e-1,
}];

/// Whether any row admits this dtype, under some optimizer, on some backend.
/// A screen the resolver can apply, not an admission: it says the question is
/// worth asking where the optimizer and the device are known. F32 is the
/// floor and is not answered here.
pub fn base_dtype_is_tabled(dtype: &TensorDtype) -> bool {
    BASE_DTYPE_TABLE.iter().any(|row| row.dtype == dtype.name())
}

/// The row for one full combination, or `None` when nothing here has measured
/// it. F32 has no row by construction.
pub fn base_dtype_capability(
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
    backend: &str,
) -> Option<&'static BaseDtypeCapability> {
    BASE_DTYPE_TABLE.iter().find(|row| {
        row.dtype == dtype.name() && row.optimizer == optimizer && row.backend == backend
    })
}

/// Whether this build admits marking a base tensor of `dtype` under
/// `optimizer` on `backend`. F32 is admitted everywhere; everything else
/// needs a row.
pub fn base_dtype_admits(dtype: &TensorDtype, optimizer: OptimizerKind, backend: &str) -> bool {
    matches!(dtype, TensorDtype::F32) || base_dtype_capability(dtype, optimizer, backend).is_some()
}

/// The backends a refusal can name for a (dtype, optimizer) pair, in table
/// order.
pub fn base_dtype_backends(
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
) -> impl Iterator<Item = &'static str> {
    let name = dtype.name().to_string();
    BASE_DTYPE_TABLE
        .iter()
        .filter(move |row| row.dtype == name && row.optimizer == optimizer)
        .map(|row| row.backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_is_not_a_row_and_is_admitted_everywhere() {
        assert!(
            !base_dtype_is_tabled(&TensorDtype::F32),
            "F32 is the baseline the rows are measured against, not a claim"
        );
        assert!(base_dtype_admits(
            &TensorDtype::F32,
            OptimizerKind::Sgd,
            "a-backend-with-no-row"
        ));
    }

    #[test]
    fn a_row_admits_its_own_combination_and_nothing_beside_it() {
        assert!(base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::AdamW,
            "CPU"
        ));
        // Same dtype and optimizer, a backend no lane has run.
        assert!(!base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::AdamW,
            "MTL"
        ));
        // Same dtype and backend, an optimizer whose kernel is F32-only.
        assert!(!base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::Sgd,
            "CPU"
        ));
        assert!(!base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "CPU"
        ));
    }

    /// A row whose optimizer cannot write its dtype would admit a step that
    /// aborts. The optimizer's dtype table is a necessary condition.
    #[test]
    fn no_row_outruns_its_optimizers_dtype_table() {
        for row in BASE_DTYPE_TABLE {
            let dtype = TensorDtype::from_ggml_name(row.dtype);
            assert!(
                row.optimizer.supports_dtype(&dtype),
                "{} claims {} which its update step cannot write",
                row.optimizer,
                row.dtype
            );
            assert_ne!(row.dtype, "F32", "the floor is not a row");
            assert!(!row.backend.is_empty());
            assert!(row.gradient_tolerance > 0.0);
            assert!(row.update_tolerance > 0.0);
            assert!(
                row.update_outlier_fraction > 0.0 && row.update_outlier_fraction < 0.05,
                "{}/{}/{} admits {} of its elements as outliers, which is not a minority",
                row.dtype,
                row.optimizer,
                row.backend,
                row.update_outlier_fraction
            );
            assert!(row.stability_loss_tolerance > 0.0);
            assert!(
                row.stability_steps > 0,
                "{}/{}/{} has no long lane behind it",
                row.dtype,
                row.optimizer,
                row.backend
            );
        }
    }

    #[test]
    fn a_combination_appears_at_most_once() {
        let mut seen: Vec<(&str, OptimizerKind, &str)> = BASE_DTYPE_TABLE
            .iter()
            .map(|row| (row.dtype, row.optimizer, row.backend))
            .collect();
        let before = seen.len();
        seen.sort_unstable_by_key(|(dtype, optimizer, backend)| {
            (*dtype, optimizer.as_str(), *backend)
        });
        seen.dedup();
        assert_eq!(seen.len(), before, "two rows for one combination");
    }

    #[test]
    fn a_refusal_can_name_the_backends_that_do_carry_it() {
        let backends: Vec<&str> =
            base_dtype_backends(&TensorDtype::F16, OptimizerKind::AdamW).collect();
        assert_eq!(backends, vec!["CPU"]);
        assert_eq!(
            base_dtype_backends(&TensorDtype::F16, OptimizerKind::Sgd).count(),
            0
        );
    }
}
