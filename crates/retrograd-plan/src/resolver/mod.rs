//! Resolves an intention into a complete, executable configuration.
//!
//! Resolution collects inputs, derives semantic choices, searches feasible
//! geometry, applies proactive defaults, and finally uses memory levers while
//! the estimate exceeds its budget. Measured calibration enters through
//! [`ResolveInput::calibrations`] and [`crate::provenance::Source::Measured`].
//!
//! Client parameters remain locked throughout resolution. Any lever that loses
//! information requires an explicit [`Allow`]. The sole degrading default,
//! `fast_sampling_context`, emits `sampling_distribution_approximated` and can
//! be disabled through `params`.

use std::path::PathBuf;

use retrograd_config::{
    CheckpointToml, ConfigDocument, EvaluationToml, GrpoToml, LoraToml, MetricsToml, ModelToml,
    OutputToml, PpoToml, RunConfig, RunToml, SamplingToml, SftToml, SharedPrefixFanoutToml,
    TrainingToml, build as build_run_config,
};
use retrograd_core::{
    ExecutionProfile, LrScheduler, ModelInfo, PreflightReport, RewardProtocol, SharedPrefixFanout,
    TargetSet, TrainConfig, saturating_dim, saturating_dim_product,
};
use retrograd_dataset::DataFormat;
use serde::Serialize;
use serde_json::Value;

use crate::budget::{BudgetRequest, Budgets, MarginPolicy, MemoryBaseline, Overflow};
use crate::cost::{
    Calibration, MemoryEstimate, Trainable, Workload, WorkloadKind, estimate, resolve_trainable_set,
};
use crate::dataset::DatasetStats;
use crate::defaults::{self, ACTIVE_DEFAULTS, AppliedDefault, Verdict};
use crate::execution::{ExecutionPlan, RejectedCandidateSummary, explain_execution};
use crate::geometry::{Constraints, Geometry, candidates, round_up_pow2};
use crate::levers::{AppliedLever, LEVERS, Limits, unlocking_opt_ins};
use crate::merge::{MergeError, deep_merge, overridden_paths};
use crate::provenance::Provenance;
use crate::recipe::{Allow, Objective, Recipe};
use crate::rules::Rule;
use crate::tuning;
use crate::{Backend, HardwareFacts};

mod document_round_trip;
mod feasible_geometry;
mod memory_recovery;
mod proactive_defaults;
mod reporting;
/// Percentile of the example lengths `n_ctx` is chosen from.
mod semantic_choices;

use document_round_trip::*;
use feasible_geometry::*;
use memory_recovery::*;
use proactive_defaults::*;
pub use reporting::*;
use semantic_choices::*;

const CONTEXT_PERCENTILE: f64 = 0.99;

/// Smallest context the truncation lever may fall to. Below the median the run
/// would be training on fragments; a caller who wants that says so with an
/// explicit `training.ctx` parameter.
const MIN_CONTEXT: u32 = 64;

/// Rough training throughput, in tokens per second, used only for a `minutes`
/// budget. Backed by nothing but an order of magnitude - a forward and backward
/// pass costs about `6 × n_params` FLOP per token, against an effective couple
/// hundred GFLOP/s. Every plan that uses it carries a warning saying so, and the
/// resolved `epochs`/`updates` are what the run actually obeys.
const EFFECTIVE_FLOPS: f64 = 2.0e11;

/// Everything phase 0 collected.
pub struct ResolveInput<'a> {
    pub recipe: &'a Recipe,
    /// The client's own parameters: a partial document tree, same schema as
    /// `config` and as the CLI's TOML. Its leaves are locked.
    pub params: &'a Value,
    pub model: &'a ModelInfo,
    /// Geometry of the distillation teacher, when the document names one. It is
    /// the only second model a run holds, and the budget below is optimistic by
    /// its weights plus its KV without it - which is why a `distill` document
    /// that arrives without one is warned about rather than quietly sized as if
    /// one model were resident.
    pub teacher: Option<&'a ModelInfo>,
    /// The model's own tensor table, when the caller could read one.
    ///
    /// What a base-weight policy is priced from; a `lora` document never needs
    /// it, and a base document without one is refused by
    /// [`crate::base_training_unpriced`].
    pub inventory: Option<&'a retrograd_core::TensorInventory>,
    pub data: &'a DatasetStats,
    pub eval: Option<&'a DatasetStats>,
    pub data_format: DataFormat,
    pub baseline: MemoryBaseline,
    /// The backend and memory facts a rule may key on.
    pub hardware: HardwareFacts,
    /// Versioned snapshot produced by the engine. `None` is accepted for pure
    /// analytical callers, and is reported as such rather than guessed.
    pub execution_profile: Option<&'a ExecutionProfile>,
    /// Concrete graph report, only trusted when its profile fingerprint matches.
    pub preflight: Option<&'a PreflightReport>,
    pub server_budgets: (BudgetRequest, BudgetRequest),
    pub margin: MarginPolicy,
    /// Shape-specific measurements. Without a profile or a matching entry the
    /// analytical identity factor is used.
    pub calibrations: Option<&'a crate::CalibrationStore>,
    /// Packed optimizer timings taken for this request only. Never persisted:
    /// they do not transfer to another workload, device state or adapter.
    pub packing_measurements: Option<&'a crate::packing_tuning::PackingMeasurements>,
    /// The reward command the catalogue resolved for `recipe.reward`. Empty for
    /// a supervised objective. The resolver never sees an id-to-command mapping;
    /// the caller substitutes, and redacts again on the way out.
    pub reward_command: Vec<String>,
    /// How that command is spoken to, as the catalogue declared it: the client
    /// never sees the command, so it cannot be the one to say whether the
    /// command can hold a persistent worker. `None` leaves the document on the
    /// schema's own defaults.
    pub reward_protocol: Option<RewardProtocol>,
    /// Base directory relative paths in the document resolve against.
    pub root: PathBuf,
}

/// What the resolution produced.
#[derive(Clone, Debug)]
pub struct Resolution {
    /// Bounded, fitting packing geometries for an optional measured planning
    /// pass. Internal execution work, not part of the public plan schema.
    pub packing_probes: Vec<crate::packing_tuning::PackingProbe>,
    /// The effective configuration, in the document schema - the same schema the
    /// CLI reads from TOML.
    pub document: ConfigDocument,
    /// The same, already validated into what the engine will run.
    pub config: RunConfig,
    pub provenance: Provenance,
    pub plan: PlanSummary,
}

/// One warning, with a code a client can match on.
///
/// A string alone was enough while every warning was advice for a human. It is
/// not enough now that one of them - `sampling_distribution_approximated` - is
/// how the API discharges its side of invariant 4: a client that has to *detect*
/// an approximation cannot be asked to grep prose.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PlanWarning {
    /// Stable identifier, `snake_case`. Part of the contract.
    pub code: &'static str,
    /// The dotted configuration path this warning is about, when there is one
    /// field that is the reason for it. Absent for a
    /// warning about the request as a whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<&'static str>,
    /// One sentence for a human, and free to be reworded.
    pub message: String,
}

fn warn(
    warnings: &mut Vec<PlanWarning>,
    code: &'static str,
    field: Option<&'static str>,
    message: impl Into<String>,
) {
    let message = message.into();
    // Two phases can legitimately reach the same conclusion (a default applied,
    // then the final configuration read back). One warning per code is what a
    // client wants to see.
    if warnings.iter().any(|warning| warning.code == code) {
        return;
    }
    warnings.push(PlanWarning {
        code,
        field,
        message,
    });
}

/// The `plan` half of the response.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PlanSummary {
    pub schema_version: u32,
    pub total_steps: u64,
    /// Epochs for SFT, updates for PPO/GRPO.
    pub iterations: u64,
    pub memory: MemoryPlan,
    pub execution: ExecutionPlan,
    /// What phase 2bis turned on because it is better on this machine at this
    /// context - not because the run was cornered. Disjoint from [`Self::levers`]
    /// by construction: a setting phase 3 pushed further is a sacrifice and
    /// appears there instead.
    pub defaults_applied: Vec<AppliedDefault>,
    /// The levers phase 3 applied, in the order it applied them.
    pub levers: Vec<AppliedLever>,
    /// Share of examples longer than the resolved context, rounded to four
    /// decimals by [`reported_fraction`] so the payload carries no float the
    /// caller cannot read.
    pub truncation_fraction: f64,
    /// The same figure for the eval dataset, when there is one - an evaluation
    /// that truncates 40% of its examples does not measure what it claims to,
    /// and that is invisible unless it is reported next to the training figure
    /// rather than folded into it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_truncation_fraction: Option<f64>,
    pub judge_calls_expected: u64,
    pub warnings: Vec<PlanWarning>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct MemoryPlan {
    pub resources: crate::cost::ResourceEstimate,
    /// Filled by phase 4 (step 6). Absent means "not measured yet", which is not
    /// the same as measured-at-zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured: Option<MemoryEstimate>,
    pub budgets: Budgets,
}

/// Why a resolution failed. Each variant maps to one problem type in the API.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ResolveError {
    /// The request cannot be satisfied as written. `path` is the dotted field at
    /// fault when there is one.
    #[error("{message}")]
    Invalid {
        message: String,
        path: Option<String>,
    },
    /// A degradation would be needed and was not opted into: a degradation is never silent.
    /// Says what it would have done, so the caller can decide.
    #[error("{message}")]
    NeedsOptIn { allow: Allow, message: String },
    /// A client-supplied parameter cannot be honoured together with the rest of
    /// the request. It is never silently moved.
    #[error("{path}: {message}")]
    OverrideConflict { path: String, message: String },
    /// Every lever was exhausted and the estimate still overflows.
    #[error("{}",.0.detail())]
    InsufficientMemory(Box<InsufficientMemory>),
}

#[derive(Clone, Debug)]
pub struct InsufficientMemory {
    pub estimate: MemoryEstimate,
    pub budgets: Budgets,
    pub overflow: Overflow,
    pub applied: Vec<AppliedLever>,
    /// Opt-ins that would unlock at least one more lever.
    pub unlocks: Vec<Allow>,
}

impl InsufficientMemory {
    /// A one-line summary naming the dominant post and the shortfall, because
    /// "it does not fit" without *what* does not fit is not an answer.
    pub fn detail(&self) -> String {
        let posts = self.estimate.dominant_posts();
        let dominant = posts
            .first()
            .map(|(name, bytes)| format!("{name} at {}", human_bytes(*bytes)))
            .unwrap_or_else(|| "nothing".to_string());
        format!(
            "{} over budget after {} lever(s); largest post is {dominant}",
            human_bytes(self.overflow.total()),
            self.applied.len()
        )
    }
}

/// Rounds a ratio to four decimals for reporting.
///
/// `3 / 7` serializes as `0.42857142857142855` - seventeen digits of a number
/// nobody asked to that precision, and one whose last bits depend on the
/// division order. Four decimals round-trip exactly through `f64`, so the
/// payload stays byte-identical across resolutions (invariant 3).
///
/// A non-zero share never rounds to zero: `plan.truncation_fraction = 0.0`
/// beside a run that truncates would be a lie, and it is the field the caller
/// opted into a degradation for.
pub fn reported_fraction(value: f64) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return 0.0;
    }
    let rounded = (value * 10_000.0).round() / 10_000.0;
    if rounded == 0.0 { 0.0001 } else { rounded }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

impl From<MergeError> for ResolveError {
    fn from(error: MergeError) -> Self {
        Self::Invalid {
            message: error.message,
            path: Some(error.pointer.trim_start_matches('/').replace('/', ".")),
        }
    }
}

impl From<retrograd_core::Error> for ResolveError {
    fn from(error: retrograd_core::Error) -> Self {
        Self::Invalid {
            message: error.to_string(),
            path: None,
        }
    }
}

/// `algorithm` comes from [`crate::calibration::algorithm_slug`] on the built
/// configuration - never from the recipe's objective. The written key is derived
/// the same way, and two mappings would drift into never matching each other.
fn calibration_for(
    input: &ResolveInput<'_>,
    algorithm: &str,
    training: &TrainConfig,
) -> Calibration {
    let (Some(profile), Some(store)) = (input.execution_profile, input.calibrations) else {
        return Calibration::default();
    };
    let key = crate::calibration_key_for(profile, input.model, algorithm, training);
    store.factors(&key)
}

/// Runs the phases.
pub fn resolve(input: &ResolveInput<'_>) -> Result<Resolution, ResolveError> {
    let span = tracing::info_span!(target: "retrograd::plan::resolve", "resolve");
    let _entered = span.enter();
    let recipe = input.recipe;
    if let Some(profile) = input.execution_profile {
        profile.validate().map_err(|error| ResolveError::Invalid {
            message: error.to_string(),
            path: Some("execution_profile".to_string()),
        })?;
    }
    if let Some(preflight) = input.preflight {
        preflight
            .validate()
            .map_err(|error| ResolveError::Invalid {
                message: error.to_string(),
                path: Some("preflight".to_string()),
            })?;
    }
    // Before anything reads a parameter: a `null` is a malformed request, and
    // locking it first would report it as an impossible field instead.
    crate::merge::reject_nulls(input.params)?;
    let locked = overridden_paths(input.params);
    let is_locked = |path: &str| locked.contains(path);
    let mut provenance = Provenance::new();
    for path in &locked {
        provenance.overridden(path);
    }
    let mut warnings = Vec::new();

    // Resolve budgets.
    let budgets = Budgets::resolve(
        input.baseline,
        input.server_budgets,
        (recipe.limits.vram, recipe.limits.ram),
        input.margin,
    );
    if budgets.vram.capped || budgets.ram.capped {
        warn(
            &mut warnings,
            "limits_capped_to_server_budget",
            Some("limits"),
            "the requested limits were above the server budget and were lowered to it",
        );
    }
    if !input.data.measured {
        warn(
            &mut warnings,
            "example_lengths_estimated",
            Some("training.ctx"),
            "example lengths are estimated from character counts, not tokenized; \
             the resolved context is deliberately generous",
        );
    }

    // Derive semantic choices.
    let n_ctx = choose_context(input, &mut provenance, &is_locked)?;
    let truncation_fraction = input.data.truncation_fraction(n_ctx);
    if truncation_fraction > 0.0 && !recipe.allows(Allow::TruncateContext) {
        return Err(ResolveError::NeedsOptIn {
            allow: Allow::TruncateContext,
            message: format!(
                "a context of {n_ctx} truncates {:.2}% of the examples \
                 (the longest is {} tokens)",
                truncation_fraction * 100.0,
                input.data.max_length()
            ),
        });
    }

    let (mut document, rollout) = draft_document(
        input,
        n_ctx,
        &budgets,
        &is_locked,
        &mut provenance,
        &mut warnings,
    )?;
    apply_params(&mut document, input.params)?;
    let run_config = build_run_config(document.clone(), &input.root)?;
    // The base half of the trainable set, resolved once and priced everywhere
    // downstream, so the budget and the run it describes cannot disagree.
    let base_trainable = resolve_trainable_set(&run_config.training.trainable, input.inventory)
        .map_err(|error| ResolveError::Invalid {
            message: error.to_string(),
            path: Some("training.trainable".to_string()),
        })?;

    // Search geometry, apply defaults, and recover from memory pressure.
    let workload = workload_of(input, &run_config);
    let algorithm = crate::calibration::algorithm_slug(&run_config.algorithm);
    let mut training = run_config.training.clone();
    // Either half may be absent: `full` and `partial` create no adapter, and
    // `lora` resolves no base tensor.
    let lora = run_config.lora.as_ref().map(|lora| lora.config.clone());
    let mut limits = Limits {
        min_batch: training.n_seq_max.max(1),
        min_ctx: MIN_CONTEXT.max(round_up_pow2(input.data.percentile(0.5))),
        // Raised by the candidate search once a packed subgroup is selected.
        min_ubatch: 1,
        // PPO and GRPO only: one optimizer step per rollout is part of the
        // objective, not a tuning choice.
        whole_row_batch: matches!(workload.kind, WorkloadKind::Rollout { .. }),
        generation_batch_default: 0,
    };

    // These borrows are shared by candidate search, defaults, and levers. Build
    // them once after the
    // run config exists, because `algorithm`, `trainable` and `workload` are
    // read off it.
    let trainable = Trainable {
        adapter: lora.as_ref(),
        base: base_trainable.as_ref(),
    };
    let sizing = Sizing {
        input,
        algorithm,
        budgets: &budgets,
        trainable,
        workload: &workload,
        is_locked: &is_locked,
    };

    choose_geometry(&mut training, &sizing, &limits, &mut provenance);

    let mut defaults_applied = apply_defaults(
        &mut training,
        &sizing,
        rollout.group_size,
        &mut provenance,
        &mut warnings,
    );

    // The first lever halves `generation_batch`, and it cannot halve a number the
    // configuration does not spell out - so what the runtime *would* derive has
    // to be read after phase 2bis, which is what sets the concurrency it depends
    // on.
    if workload.kind.is_packed() {
        limits.generation_batch_default = crate::cost::generation_batch(
            &training,
            input.model,
            training.generation_concurrency.max(1) as u64,
        ) as u32;
    }

    let CandidateSearch {
        min_ubatch,
        applied: mut candidate_applied,
        alternatives: candidate_alternatives,
        packing_probes,
        packing_note,
    } = search_candidates(
        &mut training,
        &sizing,
        &run_config,
        rollout.group_size,
        &locked,
        &mut provenance,
    )?;
    if let Some(message) = packing_note {
        warnings.push(PlanWarning {
            code: "packing_geometry_measured",
            field: None,
            message,
        });
    }
    // The rescue levers run after this search. Without the floor, the
    // `micro_batch` lever would halve the physical width below the packed
    // subgroup the search just validated and emit a graph the runtime refuses.
    limits.min_ubatch = min_ubatch;

    let LeverOutcome {
        mut applied,
        blocked,
    } = apply_levers(&mut training, &sizing, &limits, &mut provenance)?;
    candidate_applied.append(&mut applied);
    let mut applied = candidate_applied;

    // A setting phase 2bis turned on and phase 3 then pushed further is a
    // user-visible change: the run is paying for it. The entry moves into
    // `plan.levers` carrying the *pre-2bis* setting as its `from`, so the two
    // lists never name the same one and neither loses where it started.
    for lever in &mut applied {
        if let Some(index) = defaults_applied
            .iter()
            .position(|entry| entry.id == lever.id)
        {
            lever.from = defaults_applied.remove(index).from;
        }
    }

    // --- Rebuild and validate ---------------------------------------------
    rebuild_and_validate(
        Tuned {
            document,
            training,
            provenance,
            warnings,
            applied,
            defaults_applied,
            alternatives: candidate_alternatives,
            packing_probes,
            blocked,
        },
        input,
        budgets,
        &workload,
        algorithm,
        &is_locked,
    )
}

/// Everything phases 1 to 3 produced, on the way to a [`Resolution`].
///
/// Owned rather than borrowed: [`rebuild_and_validate`] either moves these into
/// the resolution or into the [`InsufficientMemory`] refusal, and the two
/// destinations are what makes the split worth a struct.
struct Tuned {
    packing_probes: Vec<crate::packing_tuning::PackingProbe>,
    document: ConfigDocument,
    training: TrainConfig,
    provenance: Provenance,
    warnings: Vec<PlanWarning>,
    applied: Vec<AppliedLever>,
    defaults_applied: Vec<AppliedDefault>,
    alternatives: Vec<RejectedCandidateSummary>,
    /// Levers a client parameter took off the table, in the order they would
    /// have been tried. The first one names the refusal when the run still
    /// overflows.
    blocked: Vec<&'static str>,
}

/// The last phase: write the tuned geometry back into the document, build it
/// through the same loader the CLI uses, and either answer with the plan or
/// refuse on budget.
///
/// This is where invariant 1 is enforced - the resolver never emits a document
/// `config::load` would reject - so it deliberately goes through
/// `build_run_config` a second time rather than trusting the first build.
fn rebuild_and_validate(
    tuned: Tuned,
    input: &ResolveInput<'_>,
    budgets: Budgets,
    workload: &Workload,
    algorithm: &str,
    is_locked: &dyn Fn(&str) -> bool,
) -> Result<Resolution, ResolveError> {
    let Tuned {
        packing_probes,
        mut document,
        training,
        mut provenance,
        mut warnings,
        applied,
        defaults_applied,
        alternatives,
        blocked,
    } = tuned;
    let recipe = input.recipe;
    write_back(&mut document, &training, &is_locked);
    apply_params(&mut document, input.params)?;
    let config = build_run_config(document.clone(), &input.root).map_err(|error| {
        // Reaching here means the resolver produced something `config::load`
        // refuses - invariant 1. The message names the field so the failure is
        // diagnosable instead of mysterious.
        ResolveError::Invalid {
            message: format!("the resolved configuration is not valid: {error}"),
            path: None,
        }
    })?;

    // Re-resolved against the rebuilt configuration, not carried in from the
    // first build.
    let base_trainable = resolve_trainable_set(&config.training.trainable, input.inventory)
        .map_err(|error| ResolveError::Invalid {
            message: error.to_string(),
            path: Some("training.trainable".to_string()),
        })?;
    let final_estimate = estimate(
        input.model,
        &config.training,
        Trainable {
            adapter: config.lora.as_ref().map(|lora| &lora.config),
            base: base_trainable.as_ref(),
        },
        workload,
        calibration_for(input, algorithm, &config.training),
    );
    let resources = final_estimate.resources();
    let overflow = budgets.overflow(resources.device_peak_bytes, resources.host_peak_bytes);
    if !overflow.fits() {
        let unlocks = unlocking_opt_ins(&recipe.allow);
        if let Some(path) = blocked.first() {
            return Err(ResolveError::OverrideConflict {
                path: (*path).to_string(),
                message: format!(
                    "the parameter leaves this memory lever unavailable and the run is \
                     {} over budget",
                    human_bytes(overflow.total())
                ),
            });
        }
        return Err(ResolveError::InsufficientMemory(Box::new(
            InsufficientMemory {
                estimate: final_estimate,
                budgets,
                overflow,
                applied,
                unlocks,
            },
        )));
    }

    collect_warnings(
        &config,
        input.teacher,
        input.hardware.backend,
        &mut warnings,
    );
    let mut execution = explain_execution(
        input.execution_profile,
        input.preflight,
        input.hardware.backend.id(),
    );
    execution.alternatives = alternatives;
    if config.training.require_gpu_resident
        && execution
            .cpu_fallbacks
            .iter()
            .any(|fallback| fallback.nodes > 0)
    {
        return Err(ResolveError::Invalid {
            message:
                "require_gpu_resident is incompatible with CPU placements reported by preflight"
                    .to_string(),
            path: Some("training.require_gpu_resident".to_string()),
        });
    }
    let (iterations, total_steps) = step_counts(input, &config);
    // Four cadences only make sense against the step count, which is only known
    // once the geometry is. None of them can make a valid configuration invalid,
    // any positive cadence and any `warmup_steps` are accepted, and a scheduler
    // costs no memory - so they are written into both the document and the built
    // configuration rather than triggering a third build.
    let mut config = config;
    write_cadences(
        &mut document,
        &mut config,
        iterations,
        total_steps,
        &is_locked,
        &mut provenance,
    );
    let plan = PlanSummary {
        schema_version: 2,
        total_steps,
        iterations,
        memory: MemoryPlan {
            resources: resources.clone(),
            measured: None,
            budgets,
        },
        execution,
        defaults_applied,
        levers: applied,
        truncation_fraction: reported_fraction(
            input.data.truncation_fraction(config.training.n_ctx),
        ),
        eval_truncation_fraction: input
            .eval
            .map(|eval| reported_fraction(eval.truncation_fraction(config.training.n_ctx))),
        judge_calls_expected: judge_calls(input, &config),
        warnings,
    };

    Ok(Resolution {
        packing_probes,
        document,
        config,
        provenance,
        plan,
    })
}

/// Estimates a complete configuration supplied by the caller.
///
/// The semantic phases are skipped - the caller chose - but the budget check is
/// not: an overflow is still a 422 unless the caller forces it.
// Eight now that a distillation run has a second model to size, and every one
// of them is a distinct fact about the run being assessed.
#[expect(clippy::too_many_arguments)]
pub fn assess(
    config: &RunConfig,
    model: &ModelInfo,
    teacher: Option<&ModelInfo>,
    // The resolved base trainable set, from `crate::resolve_trainable_set`; `None`
    // for a LoRA document.
    base: Option<&retrograd_core::TrainableSet>,
    data: &DatasetStats,
    baseline: MemoryBaseline,
    server_budgets: (BudgetRequest, BudgetRequest),
    margin: MarginPolicy,
    calibration: Calibration,
) -> (MemoryEstimate, Budgets, Overflow) {
    let budgets = Budgets::resolve(baseline, server_budgets, (None, None), margin);
    let workload = Workload {
        kind: workload_kind_of(config),
        examples: data.examples,
        co_resident_bytes: co_resident_bytes(config, teacher),
    };
    let estimate = estimate(
        model,
        &config.training,
        Trainable {
            adapter: config.lora.as_ref().map(|lora| &lora.config),
            base,
        },
        &workload,
        calibration,
    );
    let resources = estimate.resources();
    let overflow = budgets.overflow(resources.device_peak_bytes, resources.host_peak_bytes);
    (estimate, budgets, overflow)
}

/// The immutable half of a sizing pass: what phases 2, 2bis and 3 all *read*,
/// and none of what they write.
///
/// These six borrows travelled together, in the same order, through four
/// signatures - they were already a struct, spelled out four times. What is
/// deliberately **not** in here is the mutable half (`training`, `provenance`,
/// `warnings`): keeping the two apart is what makes each pass's signature say
/// which of them it may change, which a struct holding both would hide.
///
/// Every pass destructures it on entry, so the bodies read exactly as they did
/// when these were free parameters.
struct Sizing<'a, 'i> {
    input: &'a ResolveInput<'i>,
    algorithm: &'a str,
    budgets: &'a Budgets,
    /// What this run trains: the adapter and/or the resolved base set.
    trainable: Trainable<'a>,
    workload: &'a Workload,
    is_locked: &'a dyn Fn(&str) -> bool,
}

/// What the V2 candidate search decided, on top of the `training` it mutated.
struct CandidateSearch {
    packing_probes: Vec<crate::packing_tuning::PackingProbe>,
    packing_note: Option<String>,
    /// Floor the rescue levers must not push the physical width below: the
    /// packed subgroup the winning candidate was validated on.
    min_ubatch: u32,
    applied: Vec<AppliedLever>,
    alternatives: Vec<RejectedCandidateSummary>,
}

/// Searches fanout, physical ubatch, and checkpointing as one Pareto problem.
/// These choices are evaluated before the remaining memory levers, and table
/// order does not determine the winner.
fn search_candidates(
    training: &mut TrainConfig,
    sizing: &Sizing<'_, '_>,
    run_config: &RunConfig,
    group_size: u32,
    locked: &std::collections::BTreeSet<String>,
    provenance: &mut Provenance,
) -> Result<CandidateSearch, ResolveError> {
    let &Sizing {
        input,
        algorithm,
        budgets,
        trainable,
        workload,
        is_locked,
    } = sizing;
    let intent = crate::candidate::normalize(
        training,
        group_size.max(1),
        input.data.percentile(0.5).min(training.n_ctx),
        completion_bound(run_config),
        workload.kind.is_packed(),
        locked.iter().cloned(),
        input.execution_profile,
    );
    let candidate_values = crate::candidate::enumerate(&intent, input.model).map_err(|error| {
        match error {
            crate::candidate::EnumerateError::LimitReached { limit } => ResolveError::Invalid {
                message: format!("candidate search reached its explicit limit of {limit}"),
                path: None,
            },
            crate::candidate::EnumerateError::LockedFanoutUnsupported => {
                ResolveError::OverrideConflict {
                    path: "training.shared_prefix_fanout".to_string(),
                    message: "the locked fanout requires shared-prefix packed training, but the execution profile does not publish that capability".to_string(),
                }
            }
        }
    })?;
    let mut evaluated: Vec<_> = candidate_values
        .into_iter()
        .map(|candidate| {
            let candidate_training = candidate.applied_to(training);
            crate::candidate::evaluate(
                candidate,
                &intent,
                input.model,
                trainable,
                workload,
                budgets,
                calibration_for(input, algorithm, &candidate_training),
            )
        })
        .collect();
    let shape = crate::packing_tuning::PackingShape::from_intent(&intent);
    let empty_measurements = crate::packing_tuning::PackingMeasurements::new();
    let measurements = input.packing_measurements.unwrap_or(&empty_measurements);
    crate::packing_tuning::apply_measurements(&mut evaluated, shape, measurements, budgets);
    let packing_probes = if matches!(
        &run_config.algorithm,
        retrograd_config::Algorithm::Grpo(_) | retrograd_config::Algorithm::AgentGrpo(_)
    ) {
        crate::packing_tuning::shortlist(&evaluated, shape)
    } else {
        Vec::new()
    };
    let chosen = crate::packing_tuning::select_measured(&evaluated, shape, measurements);
    if chosen.is_none() && !measurements.is_empty() {
        return Err(ResolveError::Invalid {
            message: "no fitting execution candidate remains after the packed optimizer measurements; widen the memory budget or relax the locked geometry".into(),
            path: None,
        });
    }
    let min_ubatch = chosen.map_or(1, |evaluation| evaluation.min_ubatch.max(1));
    let selected = chosen.map(|value| value.candidate.clone());
    let packing_note = selected.as_ref().and_then(|candidate| {
        let sample = measurements.get(&shape.key(candidate))?;
        let (low, high) = sample.interval()?;
        Some(format!(
            "selected {} with packed optimizer timings {low:.6}..{high:.6} seconds per logical update \
             on synthetic prompt/completion/group shape {}/{}/{}; one warm-up and three timed updates \
             per measured candidate; switching requires a gain above 5% with non-overlapping timings; \
             generation and reward evaluation are not timed",
            candidate.canonical_key(), shape.prompt_tokens, shape.completion_tokens, shape.group_size
        ))
    });
    let mut applied = Vec::new();
    if let Some(selected) = &selected {
        let before = training.clone();
        selected.apply(training);
        record_candidate_transition(&before, training, &is_locked, provenance, &mut applied);
    }
    let mut alternatives_candidates = if measurements.is_empty() {
        crate::candidate::eliminate_dominated(evaluated)
    } else {
        evaluated
    };
    if !measurements.is_empty() {
        alternatives_candidates.sort_by_key(|candidate| {
            (
                !measurements.contains_key(&shape.key(&candidate.candidate)),
                candidate.candidate.canonical_key(),
            )
        });
    }
    let alternatives = alternatives_candidates
        .iter()
        .filter(|candidate| Some(&candidate.candidate) != selected.as_ref())
        .take(5)
        .map(|candidate| RejectedCandidateSummary {
            code: candidate
                .rejection
                .unwrap_or(
                    if measurements.contains_key(&shape.key(&candidate.candidate)) {
                        "packing_timing_alternative"
                    } else {
                        "cost_superior"
                    },
                )
                .to_string(),
            candidate: candidate.candidate.canonical_key(),
            reason: if let Some(sample) = measurements.get(&shape.key(&candidate.candidate)) {
                if let Some(error) = &sample.failure {
                    format!("packing benchmark failed: {error}")
                } else {
                    format!(
                        "packed optimizer benchmark: seconds={:?}, device_bytes={}, status={}",
                        sample.seconds,
                        sample.device_bytes,
                        candidate.rejection.unwrap_or("within_budget")
                    )
                }
            } else if candidate.valid {
                format!("predicted relative cost {}", candidate.execution_cost)
            } else {
                candidate.rejection.unwrap_or("invalid").to_string()
            },
        })
        .collect::<Vec<_>>();
    Ok(CandidateSearch {
        packing_probes,
        packing_note,
        min_ubatch,
        applied,
        alternatives,
    })
}
