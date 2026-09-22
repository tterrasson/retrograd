//! Which base-weight precisions this build trains, and on what.
//!
//! F32 is the floor and is deliberately not a row here: a row is a measured
//! claim, and F32 is the baseline those claims are measured against.
//!
//! Anything wider is admitted per (dtype, optimizer, backend, master)
//! combination: the model file decides the dtype, the update kernel has its
//! own dtype table, each backend reimplements it, and the master copy decides
//! whether the step that runs is the half-precision one at all. A row records
//! the measurement that admitted the combination, and a lane reads its
//! tolerances from the row.
//!
//! The master column is what lets a backend be admitted without a
//! half-precision update kernel: with a master copy the step is the F32 one
//! and a cast writes the store, and both exist on every backend.
//!
//! The backend column names the ggml registry ("CPU", "MTL", "CUDA",
//! "Vulkan", as `ggml_backend_reg_name` spells them), not a device name. The
//! table declares; the runtime asks the live device whether it can run the
//! step, and `tests/f16_base_training.rs` checks the two agree.

use crate::optimizer::OptimizerKind;
use crate::trainable::TensorDtype;

/// One admitted (dtype, optimizer, backend, master) combination, with its
/// evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseDtypeCapability {
    /// The dtype as [`TensorDtype::name`] spells it. Never `"F32"`.
    pub dtype: &'static str,
    /// The optimizer whose update step writes a parameter of that dtype.
    pub optimizer: OptimizerKind,
    /// The ggml registry the kernel belongs to.
    pub backend: &'static str,
    /// Whether this row was measured with an F32 master copy in the loop.
    ///
    /// Two rows per (dtype, optimizer, backend) are the rule, not a duplicate:
    /// without a master the update kernel rounds back onto the grid it read,
    /// with one the F32 step runs on the copy and a cast writes the store.
    pub master: bool,
    /// Largest relative per-element difference the parity lane allows between
    /// this combination's gradient and the F32 run's on the fixture.
    /// Gradients are F32 in both runs; the weights the forward read differ.
    pub gradient_tolerance: f32,
    /// Bound on the per-element parameter delta of one step, in units of the
    /// storage grid the parameter is kept on. A quantized trajectory cannot
    /// land between grid points, so one grid point is the closest agreement
    /// the storage allows; a relative error would count a correct rounding
    /// near zero as an error of one. Being in grid units is what lets one
    /// value stand for every dtype.
    ///
    /// A master row keeps the same one grid point, not the cast's half: the
    /// gap left with the F32 run is the forward's difference, which a master
    /// copy does not touch. What it removes is the in-place rounding, and the
    /// assertion for that is an equality: `tests/f16_base_training.rs`
    /// compares the store against the run's own master copy.
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
    ///
    /// A master row's lane runs at the same rate as the in-place row beside
    /// it, so at this fixture's scale the two paths differ by the fixture's
    /// own noise. A master copy buys the rates below the in-place floor,
    /// which the in-place path refuses and therefore has no row.
    pub stability_loss_tolerance: f32,
}

/// Every base dtype beyond F32 this build admits. One row per combination a
/// lane has actually executed; adding a backend means running the lane, not
/// editing a string.
pub const BASE_DTYPE_TABLE: &[BaseDtypeCapability] = &[
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::AdamW,
        backend: "CPU",
        master: false,
        // The F16 forward differs from F32 by ~5e-4 per operand; the bound
        // sits an order of magnitude above that.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 1.8e-3, bound 5e-3.
        update_outlier_fraction: 5.0e-3,
        stability_steps: 2000,
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::AdamW,
        backend: "CUDA",
        master: false,
        // Measured 1.7e-3, three times the CPU row's 5.5e-4: the forward
        // reduction order differs. The bound is set by the dtype, not the
        // backend.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 2.1e-3 against CPU's 1.8e-3.
        update_outlier_fraction: 6.0e-3,
        stability_steps: 2000,
        // Measured 7.9e-4 after 2000 steps. The bound catches a run whose
        // training quietly stopped, not bit parity.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::AdamW,
        backend: "CPU",
        master: false,
        // BF16 carries 8 significant bits, so an operand costs about eight
        // times more than F16. Measured 5.5e-3.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 2.3e-3; the coarser grid puts more elements a grid point
        // away from the F32 trajectory.
        update_outlier_fraction: 5.0e-3,
        stability_steps: 2000,
        // Measured 5.9e-3 after 2000 steps; the bound is the same on every
        // row.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::AdamW,
        backend: "CUDA",
        master: false,
        // Same bound as the CPU row. Measured 4.9e-3 against 5.5e-3 there: at
        // this grid the storage error dominates the reduction-order one.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 2.2e-3 against the CPU row's 2.3e-3.
        update_outlier_fraction: 5.0e-3,
        stability_steps: 2000,
        // Measured 1.5e-3 after 2000 steps.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::Sgd,
        backend: "CPU",
        master: false,
        // Same bound as AdamW: the gradient comes out of the forward, which
        // does not know which optimizer will read it. Measured 5.5e-4.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 5.7e-4, half the AdamW row: an SGD step is smaller than
        // AdamW's normalized one.
        update_outlier_fraction: 2.0e-3,
        stability_steps: 2000,
        // Measured 2.2e-3 after 2000 steps.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::Sgd,
        backend: "CPU",
        master: false,
        // The BF16 operand cost, as in the AdamW row. Measured 5.5e-3.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 8.5e-4 against the F16 SGD row's 5.7e-4: the coarser grid
        // puts more elements a grid point away.
        update_outlier_fraction: 3.0e-3,
        stability_steps: 2000,
        // Measured 6.3e-3 after 2000 steps.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::Sgd,
        backend: "CUDA",
        master: false,
        // The F16 bound. Measured 1.7e-3, the AdamW CUDA number exactly: the
        // forward does not know which optimizer will read it.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 3.3e-3, six times the CPU row: an SGD step is about one
        // grid point wide, so the backend's forward difference decides which
        // side of a grid point the update lands on.
        update_outlier_fraction: 1.0e-2,
        stability_steps: 2000,
        // Measured 2.3e-5 after 2000 steps.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::Sgd,
        backend: "CUDA",
        master: false,
        // The BF16 bound. Measured 4.9e-3, the AdamW CUDA number.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 6.5e-4, below the F16 SGD row: the step is well inside one
        // grid point, so the forward difference flips fewer elements across
        // one.
        update_outlier_fraction: 2.5e-3,
        stability_steps: 2000,
        // Measured 5.0e-3 after 2000 steps.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::AdamW,
        backend: "CPU",
        master: true,
        // The forward still reads the F16 store, so the gradient bound is the
        // store's, as in the in-place row.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 1.1e-3 against the in-place row's 1.2e-3.
        update_outlier_fraction: 2.0e-3,
        stability_steps: 2000,
        // Measured 7.4e-4 after 2000 steps, against the in-place row's 2.1e-4.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::AdamW,
        backend: "CPU",
        master: true,
        // The forward still reads the BF16 store, so the gradient bound is the
        // store's, as in the in-place row.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 2.2e-3 against the in-place row's 2.3e-3.
        update_outlier_fraction: 5.0e-3,
        stability_steps: 2000,
        // Measured 4.0e-3 after 2000 steps, against the in-place row's 5.9e-3.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::Sgd,
        backend: "CPU",
        master: true,
        // The forward still reads the F16 store, so the gradient bound is the
        // store's, as in the in-place row.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 3.3e-4, half the in-place row's 5.7e-4: the in-place
        // rounding is gone.
        update_outlier_fraction: 1.0e-3,
        stability_steps: 2000,
        // Measured 4.9e-3 after 2000 steps, against the in-place row's 2.2e-3.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::Sgd,
        backend: "CPU",
        master: true,
        // The forward still reads the BF16 store, so the gradient bound is the
        // store's, as in the in-place row.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 6.1e-4 against the in-place row's 8.5e-4.
        update_outlier_fraction: 2.0e-3,
        stability_steps: 2000,
        // Measured 1.8e-3 after 2000 steps, against the in-place row's 6.2e-3.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::AdamW,
        backend: "CUDA",
        master: true,
        // The store's bound, as on every master row: measured 1.7e-3, as the
        // in-place CUDA row measures.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 1.9e-3 against the in-place CUDA row's 2.1e-3.
        update_outlier_fraction: 4.0e-3,
        stability_steps: 2000,
        // Measured 1.9e-4 after 2000 steps, against the in-place row's 7.9e-4.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::AdamW,
        backend: "CUDA",
        master: true,
        // The store's bound, as on every master row. Measured 4.9e-3, the
        // in-place CUDA row's number.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 2.0e-3 against the in-place CUDA row's 2.2e-3.
        update_outlier_fraction: 5.0e-3,
        stability_steps: 2000,
        // Measured 2.0e-3 after 2000 steps, against the in-place row's 1.5e-3.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "F16",
        optimizer: OptimizerKind::Sgd,
        backend: "CUDA",
        master: true,
        // The F16 bound, measured 1.7e-3, the in-place CUDA row's number.
        gradient_tolerance: 5.0e-2,
        update_tolerance: 1.0,
        // Measured 2.6e-3 against the in-place CUDA row's 3.3e-3: an SGD step
        // here is about one grid point wide, so removing the in-place rounding
        // moves some elements back off the wrong side of one.
        update_outlier_fraction: 6.0e-3,
        stability_steps: 2000,
        // Measured 5.8e-3 after 2000 steps, against the in-place row's 2.3e-5.
        stability_loss_tolerance: 2.0e-1,
    },
    BaseDtypeCapability {
        dtype: "BF16",
        optimizer: OptimizerKind::Sgd,
        backend: "CUDA",
        master: true,
        // The BF16 bound, measured 4.9e-3, the in-place CUDA row's number.
        gradient_tolerance: 4.0e-1,
        update_tolerance: 1.0,
        // Measured 7.3e-4 against the in-place CUDA row's 6.5e-4: the step is
        // well inside one grid point, so the forward's own difference puts
        // these elements a grid point away, not the rounding.
        update_outlier_fraction: 2.5e-3,
        stability_steps: 2000,
        // Measured 5.2e-3 after 2000 steps, against the in-place row's 5.0e-3.
        stability_loss_tolerance: 2.0e-1,
    },
];

/// Smallest per-element update, in ulps of the store it is written to, a
/// half-precision base weight is trained at here.
///
/// The rows above admit a storage but say nothing about how far one step
/// moves. The update kernels have no F32 master copy: they read the parameter,
/// add the step and round the sum back onto the same grid. A step below one
/// ulp is a whole ulp taken with probability `step / ulp`, so the weights
/// follow the rounding rather than the gradient. At one eighth of an ulp the
/// gradient still dominates the rounding noise within a few hundred steps.
pub const MIN_BASE_STEP_ULPS: f32 = 0.125;

/// Share of the trained half-precision *parameters* allowed to sit under
/// [`MIN_BASE_STEP_ULPS`] before the run is refused.
///
/// The grid is relative, so the count is by element: a norm vector near 1.0
/// has a much coarser ulp than the matrices, and a minority of under-represented
/// elements is tolerated.
pub const MAX_BASE_STEP_UNDERFLOW_SHARE: f32 = 0.5;

/// One ulp of the grid `dtype` stores `value` on, or `None` for dtypes with no
/// half-precision grid (F32 and every quantization).
///
/// Subnormals answer with the smallest normal's ulp, so a zero weight still
/// gets a positive grid.
pub fn storage_ulp(dtype: &TensorDtype, value: f32) -> Option<f32> {
    // Significand bits below the leading one, and the smallest normal.
    let (significand_bits, smallest_normal) = match dtype {
        TensorDtype::F16 => (f32::from(10_u8), 6.103_515_6e-5_f32),
        TensorDtype::BF16 => (f32::from(7_u8), f32::MIN_POSITIVE),
        TensorDtype::F32 | TensorDtype::Other(_) => return None,
    };
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude < smallest_normal {
        return Some((smallest_normal.log2() - significand_bits).exp2());
    }
    Some((magnitude.log2().floor() - significand_bits).exp2())
}

/// Whether one per-element update is large enough to be carried by a grid of
/// `ulp`. The step is the update the optimizer intends, not the learning rate.
pub fn base_step_is_representable(step: f32, ulp: f32) -> bool {
    step.is_finite() && ulp > 0.0 && step >= MIN_BASE_STEP_ULPS * ulp
}

/// Whether any row admits this dtype, under some optimizer, on some backend.
/// A screen the resolver can apply, not an admission: it says the question is
/// worth asking where the optimizer and the device are known. F32 is the
/// floor and is not answered here.
pub fn base_dtype_is_tabled(dtype: &TensorDtype) -> bool {
    BASE_DTYPE_TABLE.iter().any(|row| row.dtype == dtype.name())
}

/// The row for one full combination, or `None` when nothing here has measured
/// it. F32 has no row by construction.
///
/// `master` is part of the key: the two paths take different steps, so a row
/// measured on one says nothing about the other.
pub fn base_dtype_capability(
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
    backend: &str,
    master: bool,
) -> Option<&'static BaseDtypeCapability> {
    BASE_DTYPE_TABLE.iter().find(|row| {
        row.dtype == dtype.name()
            && row.optimizer == optimizer
            && row.backend == backend
            && row.master == master
    })
}

/// Whether this build admits marking a base tensor of `dtype` under
/// `optimizer` on `backend`, with or without a master copy. F32 is admitted
/// everywhere; everything else needs a row.
pub fn base_dtype_admits(
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
    backend: &str,
    master: bool,
) -> bool {
    matches!(dtype, TensorDtype::F32)
        || base_dtype_capability(dtype, optimizer, backend, master).is_some()
}

/// The backends a refusal can name for a (dtype, optimizer, master) triple, in
/// table order.
pub fn base_dtype_backends(
    dtype: &TensorDtype,
    optimizer: OptimizerKind,
    master: bool,
) -> impl Iterator<Item = &'static str> {
    let name = dtype.name().to_string();
    BASE_DTYPE_TABLE
        .iter()
        .filter(move |row| row.dtype == name && row.optimizer == optimizer && row.master == master)
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
            OptimizerKind::Muon,
            "a-backend-with-no-row",
            false
        ));
    }

    #[test]
    fn a_row_admits_its_own_combination_and_nothing_beside_it() {
        assert!(base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::AdamW,
            "CPU",
            false
        ));
        // Same dtype and optimizer, a backend no lane has run.
        assert!(!base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::AdamW,
            "MTL",
            false
        ));
        // Same dtype and backend, an optimizer whose kernel is F32-only.
        assert!(!base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::Muon,
            "CPU",
            false
        ));
        // And one whose kernel writes it, on the backend that measured it.
        assert!(base_dtype_admits(
            &TensorDtype::F16,
            OptimizerKind::Sgd,
            "CPU",
            false
        ));
        assert!(base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "CPU",
            false
        ));
        // A backend that carries the kernel and has no lane behind it.
        assert!(!base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "Vulkan",
            false
        ));
    }

    /// A row measured on the master path admits only that path.
    #[test]
    fn a_master_row_admits_only_the_master_path() {
        assert!(base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "CPU",
            true
        ));
        assert!(base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "CUDA",
            true
        ));
        // No lane has run the master path on Metal, so no row.
        assert!(!base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::AdamW,
            "MTL",
            true
        ));
        // F32-only kernels stay F32-only: the master copy changes what the
        // step writes, not which dtypes the kernel accepts.
        assert!(!base_dtype_admits(
            &TensorDtype::BF16,
            OptimizerKind::Muon,
            "CPU",
            true
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
        let mut seen: Vec<(&str, OptimizerKind, &str, bool)> = BASE_DTYPE_TABLE
            .iter()
            .map(|row| (row.dtype, row.optimizer, row.backend, row.master))
            .collect();
        let before = seen.len();
        seen.sort_unstable_by_key(|(dtype, optimizer, backend, master)| {
            (*dtype, optimizer.as_str(), *backend, *master)
        });
        seen.dedup();
        assert_eq!(seen.len(), before, "two rows for one combination");
    }

    #[test]
    fn a_refusal_can_name_the_backends_that_do_carry_it() {
        let backends: Vec<&str> =
            base_dtype_backends(&TensorDtype::F16, OptimizerKind::AdamW, false).collect();
        assert_eq!(backends, vec!["CPU", "CUDA"]);
        let backends: Vec<&str> =
            base_dtype_backends(&TensorDtype::BF16, OptimizerKind::AdamW, false).collect();
        assert_eq!(backends, vec!["CPU", "CUDA"]);
        let backends: Vec<&str> =
            base_dtype_backends(&TensorDtype::F16, OptimizerKind::Sgd, false).collect();
        assert_eq!(backends, vec!["CPU", "CUDA"]);
        // The master path has its own list of measured backends.
        let backends: Vec<&str> =
            base_dtype_backends(&TensorDtype::BF16, OptimizerKind::AdamW, true).collect();
        assert_eq!(backends, vec!["CPU", "CUDA"]);
        // An optimizer that writes F32 only has no backend to name, ever.
        assert_eq!(
            base_dtype_backends(&TensorDtype::F16, OptimizerKind::Muon, false).count(),
            0
        );
    }

    #[test]
    fn the_grid_of_a_store_is_relative_and_f32_has_no_row_here_either() {
        // 8e-3 sits in [2^-7, 2^-6): the ulp is 2^-7 / 2^7 for BF16.
        let ulp = storage_ulp(&TensorDtype::BF16, 8.0e-3).expect("BF16 has a grid");
        assert!((ulp - 6.103_515_6e-5).abs() < 1.0e-9, "{ulp:e}");
        // F16 has four more significand bits: eight times finer.
        let fine = storage_ulp(&TensorDtype::F16, 8.0e-3).expect("F16 has a grid");
        assert!((fine - ulp / 8.0).abs() < 1.0e-12, "{fine:e}");
        assert_eq!(storage_ulp(&TensorDtype::F32, 8.0e-3), None);
        assert_eq!(storage_ulp(&TensorDtype::from_ggml_name("Q4_K"), 1.0), None);
        // Zero answers with a positive ulp, not zero.
        assert!(storage_ulp(&TensorDtype::BF16, 0.0).expect("a grid") > 0.0);
    }

    #[test]
    fn the_rate_the_rows_were_measured_at_is_representable_and_an_rl_rate_is_not() {
        // Tens of ulps per step, the regime the table was measured at.
        let ulp = storage_ulp(&TensorDtype::BF16, 8.0e-3).expect("a grid");
        assert!(base_step_is_representable(1.0e-3, ulp));
        // 1e-6 is a fortieth of an ulp: too small.
        assert!(!base_step_is_representable(1.0e-6, ulp));
        // The boundary is the constant, not a nearby value.
        assert!(base_step_is_representable(MIN_BASE_STEP_ULPS * ulp, ulp));
        assert!(!base_step_is_representable(
            MIN_BASE_STEP_ULPS * ulp * 0.99,
            ulp
        ));
        // F16 is eight times finer: the same rate is representable there.
        let fine = storage_ulp(&TensorDtype::F16, 8.0e-3).expect("a grid");
        assert!(base_step_is_representable(1.0e-6 * 8.0, fine));
        assert!(!base_step_is_representable(f32::NAN, ulp));
        const { assert!(MAX_BASE_STEP_UNDERFLOW_SHARE > 0.0 && MAX_BASE_STEP_UNDERFLOW_SHARE < 1.0) };
    }
}
