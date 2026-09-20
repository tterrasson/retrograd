//! `retrograd-plan`: budgets, memory cost model and configuration resolver.
//!
//! The pure half of the resolver:
//! it turns an intention ([`Recipe`]) plus a machine's budgets into a complete,
//! validated configuration, with the provenance of every field and a decomposed
//! memory estimate.
//!
//! **It depends on no FFI.** No model is loaded, no device is touched, nothing is
//! allocated on a backend. Everything it needs about the machine - the model
//! geometry, the memory totals, the dataset lengths - is an input. That is what
//! makes the whole thing testable in the fast lane, and what lets the HTTP
//! server, the CLI and any other frontend share one answer to "will this fit,
//! and what would you run?" instead of three.
//!
//! ```no_run
//! use retrograd_plan::{budget, cost, resolver, DatasetStats};
//!
//! let baseline = budget::MemoryBaseline {
//!     device_total: Some(8 << 30),
//!     host_total: Some(32u64 << 30),
//!     ..Default::default()
//! };
//! let budgets = budget::Budgets::resolve(
//!     baseline,
//!     (budget::BudgetRequest::All, budget::BudgetRequest::All),
//!     (None, None),
//!     budget::MarginPolicy::default(),
//! );
//! println!("{} bytes of VRAM to spend", budgets.vram.effective_bytes);
//! ```

pub mod budget;
pub mod calibration;
pub mod candidate;
pub mod cost;
pub mod dataset;
pub mod defaults;
pub mod execution;
pub mod geometry;
pub mod levers;
pub mod merge;
pub mod packing_tuning;
pub mod provenance;
pub mod recipe;
pub mod resolver;
mod rules;
pub mod tuning;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

use serde::Serialize;

pub use budget::{Budget, BudgetRequest, Budgets, MarginPolicy, MemoryBaseline, Overflow};
pub use calibration::{
    CalibrationEntry, CalibrationGraphClass, CalibrationKey, CalibrationModelClass,
    CalibrationShapeClass, CalibrationStore, Observation, calibration_key, calibration_key_for,
};
pub use candidate::{
    Candidate, CandidateEvaluation, EnumerateError, MAX_CANDIDATES, NormalizedIntent,
    eliminate_dominated, enumerate, evaluate, normalize, select,
};
pub use cost::{
    COST_MODEL_VERSION, Calibration, EstimateBound, EstimateOrigin, MemoryEstimate, PhaseResources,
    ResourceEstimate, ResourcePost, Trainable, Workload, WorkloadKind, base_training_unpriced,
    estimate, resolve_trainable_set,
};
pub use dataset::DatasetStats;
pub use defaults::{ACTIVE_DEFAULTS, ActiveDefault, AppliedDefault};
pub use execution::{
    Assumption, Confidence, ExecutionPlan, FallbackSummary, KernelCoverage,
    RejectedCandidateSummary,
};
pub use levers::{AppliedLever, LEVERS};
pub use provenance::{Origin, Provenance, Source};
pub use recipe::{Allow, DataSpec, JudgeRef, Limits, Objective, Recipe, RewardRef};
pub use resolver::{
    CoResident, InsufficientMemory, MemoryPlan, PlanSummary, PlanWarning, Resolution, ResolveError,
    ResolveInput, assess, resolve,
};
pub use rules::AppliedRule;
pub use tuning::{DERIVATIONS, Derivation};

/// The compute backend a run will actually use.
///
/// Not the list of backends the build supports - the one that will run this
/// configuration. Decisions that key on it would be wrong if made from a
/// compiled-in feature flag.
///
/// It is deliberately coarse, and no rule here should ask it a question only a
/// device can answer. `cap_fused_sparse_ce` probes the device at the model's
/// real head geometry. What a backend name is still good for is
/// [`Backend::Unknown`] - an accelerator this crate has never heard of,
/// where every rule takes its conservative branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Backend {
    /// Nothing was reported. Every rule that keys on the backend takes its
    /// conservative branch, which is what an unknown machine deserves.
    #[default]
    Unknown,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    Blas,
}

impl Backend {
    /// From the family name `retrograd-server` reduces a device to (`"metal"`,
    /// `"cuda"`, …). An unrecognised name is [`Backend::Unknown`], never
    /// silently mapped onto a backend whose quirks it may not share.
    pub fn from_family(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "metal" => Self::Metal,
            "cuda" => Self::Cuda,
            "vulkan" => Self::Vulkan,
            "blas" => Self::Blas,
            _ => Self::Unknown,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Blas => "blas",
        }
    }

    /// Whether this backend drives a device with its own memory and its own
    /// launch queue - the case where a narrow launch leaves the hardware idle
    /// instead of leaving memory for something else.
    ///
    /// Vulkan says yes here and can still be an iGPU, so the caller pairs this
    /// with [`HardwareFacts::unified_memory`], which is what actually settles
    /// whether there are two pools. `Unknown` says no: an unrecognised
    /// accelerator gets the conservative width.
    pub fn is_discrete(self) -> bool {
        matches!(self, Self::Cuda | Self::Vulkan)
    }
}

/// What the resolver knows about the machine beyond its memory totals.
///
/// Plain data, like [`MemoryBaseline`]: whoever can read a device fills it in -
/// the server does, from its device list and its startup probe - and this crate
/// stays free of any FFI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardwareFacts {
    pub backend: Backend,
    /// Device and host memory are the same physical pool (Metal, iGPU).
    pub unified_memory: bool,
    /// `None` when the platform reports no total, which is not zero.
    pub device_total_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_family_maps_onto_the_enum_or_onto_unknown() {
        assert_eq!(Backend::from_family("metal"), Backend::Metal);
        assert_eq!(Backend::from_family("CUDA"), Backend::Cuda);
        assert_eq!(Backend::from_family(" vulkan "), Backend::Vulkan);
        assert_eq!(Backend::from_family("sycl"), Backend::Unknown);
        assert_eq!(Backend::default(), Backend::Unknown);
        for backend in [
            Backend::Unknown,
            Backend::Cpu,
            Backend::Metal,
            Backend::Cuda,
            Backend::Vulkan,
            Backend::Blas,
        ] {
            assert_eq!(
                serde_json::to_string(&backend).unwrap(),
                format!("\"{}\"", backend.id())
            );
        }
    }
}
