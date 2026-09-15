//! Measures a candidate, compares it with the estimate, and stores the
//! gap.
//!
//! Everything before this point is arithmetic. The cost model is deliberately
//! conservative, which makes it safe and makes it *wrong* - always in the same
//! direction on a given machine. So the last phase loads the model with the
//! configuration the first three produced, reserves the optimizer graph, and
//! reads what that actually cost.
//!
//! What is done with the answer:
//!
//! - the ratio goes into `calibration.json`, keyed by the geometry it is valid
//!   for, and corrects every later resolution on this machine;
//! - if the measurement raised a factor, the resolver runs again with the real
//!   numbers - that is "re-enter phase 3", and it is the caller in
//!   [`super::plan_recipe`] who loops;
//! - if the measurement came in *under* the estimate, nothing shrinks. The plan
//!   stays conservative and the gap is recorded, which is what makes the next
//!   resolution better without making this one optimistic.
//!
//! The measurement runs behind the same device semaphore as the runs:
//! two resolutions each loading a 4 GiB model would cause exactly the overflow
//! the resolver exists to prevent.

use std::path::Path;

use retrograd_config::RunConfig;
use retrograd_core::{MemoryReport, ModelInfo};
use retrograd_plan::cost::{Calibration, MemoryEstimate};
use retrograd_plan::provenance::Provenance;
use retrograd_plan::resolver::PlanSummary;
use retrograd_plan::{DatasetStats, Observation};

use crate::error::{ApiError, ApiResult, ProblemKind};
use crate::lock::Recover as _;
use crate::state::AppState;

/// What one measurement produced.
pub struct Measurement {
    pub key: String,
    /// The runtime's own breakdown, kept whole: the budget check needs
    /// `device_bytes`, and the scratch ratio needs a field the estimate has no
    /// counterpart for.
    pub report: MemoryReport,
    /// The measured posts, in the estimate's shape, so the two subtract field by
    /// field.
    pub measured: MemoryEstimate,
    pub observation: Observation,
}

impl Measurement {
    /// Device bytes this configuration really needs.
    ///
    /// The runtime's summed allocations, not its device-wide peak: the peak
    /// includes every other process on the card, and the budget already had that
    /// baseline subtracted from it. Counting it twice would refuse
    /// configurations that fit.
    pub fn device_bytes(&self) -> u64 {
        self.report.device_bytes
    }
}

/// Loads `config` for real and folds what it cost into the calibration table.
pub async fn measure(
    state: &AppState,
    model_path: &Path,
    model: &ModelInfo,
    config: &RunConfig,
    key: &str,
) -> ApiResult<Measurement> {
    // The ratio has to be taken against an *uncorrected* estimate. Comparing the
    // measurement to an already-corrected one would fold the factor into itself,
    // and three measured runs would triple it.
    let (uncorrected, _, _) = retrograd_plan::assess(
        config,
        model,
        // Deliberately absent, even on a `distill` document: the measurement
        // this estimate is divided by comes from `probe.measure`, which loads
        // the student alone. Charging the estimate for a teacher the
        // measurement never held would drive the correction factor below one on
        // every distillation run.
        None,
        &DatasetStats::default(),
        state.baseline,
        state.server_budgets(),
        state.config.margin,
        Calibration::default(),
    );

    let permit = state
        .device
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the device queue is closed"))?;
    let probe = state.probe.clone();
    let model_path = model_path.to_path_buf();
    let candidate = config.clone();
    let report = tokio::task::spawn_blocking(move || {
        let outcome = probe.measure(&model_path, &candidate);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("the calibration task failed: {error}")))??;

    let measured = MemoryEstimate::from(&report);
    // The backend's own scratch is what the estimate calls dequantization
    // scratch: the peak, because the estimate is a high-water figure too.
    let scratch = report
        .backend_scratch_peak_bytes
        .max(report.backend_scratch_bytes);
    let observation = {
        let mut store = state.calibration.write().recover();
        let observation = store.observe(key, &uncorrected, &measured, scratch);
        // Written straight away rather than on shutdown: a server that is killed
        // must not lose what it just learned, and the file is a few hundred bytes.
        if let Err(error) = store.save(state.calibration_path()) {
            tracing::warn!(%error, "could not write calibration.json");
        }
        observation
    };
    tracing::info!(
        key,
        compute_ratio = observation.compute_ratio,
        scratch_ratio = observation.scratch_ratio,
        raised = observation.raised,
        "measured a candidate configuration"
    );
    Ok(Measurement {
        key: key.to_string(),
        report,
        measured,
        observation,
    })
}

/// Writes the measurement into the plan a client will read.
pub fn record(measurement: &Measurement, plan: &mut PlanSummary, provenance: &mut Provenance) {
    plan.memory.measured = Some(measurement.measured);
    provenance.promote_derived_to_measured(&format!(
        "confirmed by loading the model on this machine (calibration key {})",
        measurement.key
    ));
    let estimated = plan.memory.resources.device_peak_bytes;
    let measured = measurement.device_bytes();
    if measured > estimated {
        plan.warnings.push(retrograd_plan::PlanWarning {
            code: "measurement_above_estimate",
            field: None,
            message: format!(
                "the measured device footprint ({measured} bytes) is above the estimate \
                 ({estimated} bytes); the correction is recorded and applies from the next \
                 resolution on"
            ),
        });
    }
    if !measurement.report.is_measured() {
        plan.warnings.push(retrograd_plan::PlanWarning {
            code: "device_budget_never_sampled",
            field: None,
            message: "the runtime never sampled the device budget, so the measured device \
                      totals are unavailable rather than nil"
                .to_string(),
        });
    }
}

/// The budget check, run against what was measured rather than what was guessed.
///
/// The host side stays estimated: nothing in the runtime's report describes the
/// prepared dataset or the rollout buffers, and inventing a measurement for them
/// would be exactly the lie a plan must not tell.
pub fn over_budget(measurement: &Measurement, plan: &PlanSummary) -> Option<ApiError> {
    let overflow = plan.memory.budgets.overflow(
        measurement.device_bytes(),
        plan.memory.resources.host_peak_bytes,
    );
    if overflow.fits() {
        return None;
    }
    Some(
        ApiError::new(
            ProblemKind::InsufficientMemory,
            format!(
                "measured {} bytes on the device, {} over the effective budget, after \
                 re-resolving with the measured figures",
                measurement.device_bytes(),
                overflow.total()
            ),
        )
        .with_meta(serde_json::json!({
            "overflow_bytes": overflow.total(),
            "measured_device_bytes": measurement.device_bytes(),
        })),
    )
}
