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
    PpoToml, RunConfig, RunToml, SamplingToml, SftToml, SharedPrefixFanoutToml, TrainingToml,
    build as build_run_config,
};
use retrograd_core::{
    ExecutionProfile, LoraConfig, LrScheduler, ModelInfo, PreflightReport, RewardProtocol,
    SharedPrefixFanout, TargetSet, TrainConfig, saturating_dim, saturating_dim_product,
};
use retrograd_dataset::DataFormat;
use serde::Serialize;
use serde_json::Value;

use crate::budget::{BudgetRequest, Budgets, MarginPolicy, MemoryBaseline, Overflow};
use crate::cost::{Calibration, MemoryEstimate, Workload, WorkloadKind, estimate};
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

/// Percentile of the example lengths `n_ctx` is chosen from.
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

    // Search geometry, apply defaults, and recover from memory pressure.
    let workload = workload_of(input, &run_config);
    let algorithm = crate::calibration::algorithm_slug(&run_config.algorithm);
    let mut training = run_config.training.clone();
    let lora = run_config.lora.config.clone();
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
    // run config exists, because `algorithm`, `lora` and `workload` are read
    // off it.
    let sizing = Sizing {
        input,
        algorithm,
        budgets: &budgets,
        lora: &lora,
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

    let final_estimate = estimate(
        input.model,
        &config.training,
        &config.lora.config,
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
        &config.lora.config,
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
    lora: &'a LoraConfig,
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
        lora,
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
                lora,
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

// ---------------------------------------------------------------------------
// Semantic choices
// ---------------------------------------------------------------------------

fn choose_context(
    input: &ResolveInput<'_>,
    provenance: &mut Provenance,
    is_locked: &dyn Fn(&str) -> bool,
) -> Result<u32, ResolveError> {
    if is_locked("training.ctx") {
        let value = param_u32(input.params, "training", "ctx").ok_or_else(|| {
            ResolveError::OverrideConflict {
                path: "training.ctx".to_string(),
                message: "training.ctx must be a positive integer".to_string(),
            }
        })?;
        if value == 0 {
            return Err(ResolveError::OverrideConflict {
                path: "training.ctx".to_string(),
                message: "training.ctx must be greater than zero".to_string(),
            });
        }
        return Ok(value);
    }

    let n_ctx_train = input.model.n_ctx_train.max(1);
    // A text corpus is windowed into rows by the dataset preparation, so its
    // "example length" is the whole file and a percentile of it is meaningless.
    // The trained context is the honest default there.
    let wanted = match input.data_format {
        DataFormat::Text => n_ctx_train.min(2048),
        DataFormat::ChatJsonl => round_up_pow2(input.data.percentile(CONTEXT_PERCENTILE).max(1)),
    };
    let n_ctx = if wanted > n_ctx_train {
        if input.recipe.allows(Allow::ExceedTrainContext) {
            provenance.derived(
                "training.ctx",
                format!(
                    "P{:.0} of the example lengths, above the model's trained \
                     context of {n_ctx_train} (opted in)",
                    CONTEXT_PERCENTILE * 100.0
                ),
            );
            return Ok(wanted);
        }
        provenance.derived(
            "training.ctx",
            format!("capped at the model's trained context of {n_ctx_train}"),
        );
        n_ctx_train
    } else {
        provenance.derived(
            "training.ctx",
            format!(
                "P{:.0} of the example lengths, rounded up to a power of two",
                CONTEXT_PERCENTILE * 100.0
            ),
        );
        wanted
    };
    Ok(n_ctx.max(MIN_CONTEXT))
}

/// The derived rollout width. Proactive defaults need the group size to size
/// the generation concurrency, and it is not readable back off the document
/// without knowing which algorithm section to look in.
#[derive(Clone, Copy, Debug, Default)]
struct RolloutShape {
    /// Zero for anything but GRPO - the only algorithm whose configuration has
    /// a `training.generation_concurrency` to size.
    group_size: u32,
}

fn draft_document(
    input: &ResolveInput<'_>,
    n_ctx: u32,
    budgets: &Budgets,
    is_locked: &dyn Fn(&str) -> bool,
    provenance: &mut Provenance,
    warnings: &mut Vec<PlanWarning>,
) -> Result<(ConfigDocument, RolloutShape), ResolveError> {
    let recipe = input.recipe;
    let facts = tuning::Facts {
        examples: input.data.examples,
        total_tokens: input.data.total_tokens,
        vram_bytes: budgets.vram.effective_bytes,
    };

    // --- LoRA --------------------------------------------------------------
    let LoraDraft {
        targets,
        rank,
        alpha,
        learning_rate,
        weight_decay,
    } = draft_lora(input, &facts, provenance);

    let epochs = budgeted_epochs(input, n_ctx, &facts, provenance, warnings);

    // --- Rollout width -----------------------------------------------------
    let (
        rollout,
        RolloutWidth {
            group_size,
            prompts_per_update,
            updates,
            inner_epochs,
        },
    ) = draft_rollout_width(input, n_ctx, budgets, provenance);

    let training = TrainingToml {
        ctx: Some(n_ctx),
        // On the packed path `micro_batch` is pinned, not enumerated, so
        // the draft has to set a value the geometry phase will keep - which makes
        // this the *final* width for a rollout, not a starting point.
        micro_batch: Some(if recipe.objective.is_rollout() {
            tuning::packed_micro_batch(
                n_ctx,
                input.hardware.backend.is_discrete() && !input.hardware.unified_memory,
            )
        } else {
            n_ctx
        }),
        // The draft is a *valid* geometry, not the final one: phase 2 picks the
        // optimizer window. A rollout objective leaves it unset so the loader
        // pins it to `ctx / micro_batch` - spelling a count here would go stale
        // the moment `params` lock a different `micro_batch`, since the count is
        // relative to it. Off the rollout path the window starts at the whole
        // context, which keeps every intermediate document loadable, so a
        // validation failure always means a real bug.
        gradient_accumulation: (!recipe.objective.is_rollout()).then_some(1),
        epochs: Some(epochs),
        lr: Some(learning_rate),
        // A placeholder: the rule keys on the step count, which needs the
        // geometry. `constant` is the engine's own default, so the draft still
        // validates and nothing depends on this value surviving.
        lr_scheduler: Some("constant".to_string()),
        weight_decay: Some(weight_decay),
        max_grad_norm: Some(1.0),
        warmup_steps: Some(0),
        ..Default::default()
    };

    let algorithm = recipe.objective.algorithm();
    let sampling = SamplingToml {
        // Dr. GRPO is strictly on-policy; PPO inherits the same defaults so the
        // reported logprobs describe the policy being optimized.
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: (n_ctx / 2).max(16),
        seed: recipe.seed.unwrap_or(42),
    };
    if recipe.objective.is_rollout() {
        provenance.derived(
            "sampling.max_new_tokens",
            "half the resolved context, so a prompt and its completion both fit",
        );
    }

    let (sft, ppo, grpo) = match recipe.objective {
        Objective::InstructionTuning => (
            Some(SftToml {
                data: recipe.data.path.clone().unwrap_or_default(),
                data_format: recipe.data.format.clone(),
                // Left unset so the generated config carries the default
                // rather than pinning it: the planner has no reason to hold an
                // opinion on the row order.
                shuffle: None,
            }),
            None,
            None,
        ),
        Objective::PreferenceRl => (
            None,
            Some(PpoToml {
                prompts: recipe.data.path.clone().unwrap_or_default(),
                reward_command: input.reward_command.clone(),
                reward_mode: input.reward_protocol.map(|protocol| protocol.mode),
                reward_timeout_seconds: input
                    .reward_protocol
                    .map(|protocol| protocol.timeout.as_secs()),
                updates,
                rollout_batch_size: prompts_per_update,
                ppo_epochs: inner_epochs,
                clip_range: 0.2,
                kl_coefficient: 0.02,
                critic: Default::default(),
                sampling,
            }),
            None,
        ),
        Objective::ReasoningRl | Objective::Agentic => (
            None,
            None,
            Some(GrpoToml {
                prompts: recipe.data.path.clone().unwrap_or_default(),
                reward_command: input.reward_command.clone(),
                reward_mode: input.reward_protocol.map(|protocol| protocol.mode),
                reward_timeout_seconds: input
                    .reward_protocol
                    .map(|protocol| protocol.timeout.as_secs()),
                updates,
                prompts_per_update,
                group_size,
                grpo_epochs: inner_epochs,
                clip_range_low: 0.2,
                // DAPO Clip-Higher: a looser upper range fights entropy collapse.
                clip_range_high: 0.28,
                // Verifiable rewards do not need a KL anchor; it mostly slows
                // learning there.
                kl_coefficient: 0.0,
                mask_truncated: true,
                baseline: None,
                prompt_order: Some("shuffled".to_string()),
                overlong_penalty: None,
                kl_schedule: None,
                dynamic_sampling: None,
                // A judge is a server-side declaration referenced by id
                // (`[[judge]]`), never something the planner invents from a
                // recipe: it names an endpoint and a credential.
                judge: None,
                judge_weight: None,
                judge_failure: None,
                max_judge_dropped_fraction: None,
                max_stalled_updates: None,
                sampling,
            }),
        ),
    };
    if recipe.objective.is_rollout() && input.reward_command.is_empty() {
        return Err(ResolveError::Invalid {
            message: "a reinforcement objective needs a reward declared server-side".to_string(),
            path: Some("recipe.reward".to_string()),
        });
    }

    let evaluation = recipe.eval.as_ref().map(|spec| {
        // SFT evaluation is a forward pass over the whole file and ignores
        // `max_examples` (`retrograd_config::EvaluationConfig`); deriving a
        // number there would be a provenance entry for a field that does
        // nothing. A rollout evaluation, on the other hand, generates one
        // completion per prompt, so its cost scales with the count - and now
        // that the eval dataset has been read, the
        // count can be capped at what actually exists instead of a guess.
        let max_examples = recipe.objective.is_rollout().then(|| {
            let eval_examples = input.eval.map(|eval| eval.examples).unwrap_or(0);
            tuning::eval_max_examples(prompts_per_update, eval_examples)
        });
        if let Some(choice) = &max_examples
            && !is_locked("evaluation.max_examples")
        {
            provenance.derived("evaluation.max_examples", choice.reason.clone());
        }
        let patience = tuning::patience();
        if !is_locked("evaluation.patience") {
            provenance.derived("evaluation.patience", patience.reason);
        }
        EvaluationToml {
            data: spec.path.clone().unwrap_or_default(),
            // A placeholder, like the scheduler: ten evaluations over the run
            // needs the iteration count.
            every_iterations: Some(1),
            patience: Some(patience.value),
            min_delta: Some(0.0),
            max_examples: max_examples.map(|choice| choice.value),
        }
    });

    let checkpoint = recipe.checkpoint_dir.as_ref().map(|directory| {
        provenance.derived(
            "checkpoint.mode",
            if evaluation.is_some() {
                "held-out data is available, so the best evaluation is kept alongside step checkpoints"
            } else {
                "no held-out data: checkpoints are taken on step count only"
            },
        );
        CheckpointToml {
            directory: directory.clone(),
            mode: if evaluation.is_some() {
                "steps_and_best_eval".to_string()
            } else {
                "steps".to_string()
            },
            // A placeholder: the real cadence needs the step count, which needs
            // the geometry. Positive so the draft still validates.
            every_steps: Some(1),
            resume_from: None,
        }
    });

    Ok((
        ConfigDocument {
            run: RunToml {
                algorithm: algorithm.to_string(),
                verbose: false,
            },
            model: ModelToml {
                path: Some(recipe.model.clone()),
                device: None,
            },
            lora: LoraToml {
                output: recipe
                    .output
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("adapter.gguf")),
                rank: Some(rank),
                alpha: Some(alpha),
                seed: Some(recipe.seed.unwrap_or(42)),
                targets,
                init_adapter: None,
                dtype: None,
            },
            training,
            metrics: MetricsToml::default(),
            evaluation,
            checkpoint,
            sft,
            ppo,
            grpo,
            // No `Objective` resolves to `distill`, so the planner never
            // derives one. A distillation document reaches the resolver only
            // from a frontend that wrote it, and the budget it gets back is
            // optimistic by the teacher's weights and KV cache - see
            // `workload_kind_of`.
            distill: None,
            // The planner sizes single-turn runs: an agentic recipe is not one
            // of the shapes it derives, so the section is never emitted.
            agent: None,
            observe: None,
        },
        rollout,
    ))
}

/// The LoRA half of the draft: the shape of the adapter, and the two optimizer
/// settings that follow from it.
///
/// Everything here is a function of the dataset and the budget alone - no
/// geometry, no candidate search - which is why it can be decided before phase
/// 2 even starts.
fn draft_lora(
    input: &ResolveInput<'_>,
    facts: &tuning::Facts,
    provenance: &mut Provenance,
) -> LoraDraft {
    // Few examples means narrow targets and a low rank: the extra parameters
    // would mostly memorise. The rank is then capped so the adapter stays a
    // rounding error next to the weights.
    // An empty target list is the tuner's way of saying "let the runtime pick",
    // which the emitted document has to spell as `['auto']`: an *absent*
    // `lora.targets` now means every projection, and the estimate below would
    // then describe a different adapter than the config it ships with.
    let mut targets = tuning::targets(facts);
    let target_set = if targets.value.is_empty() {
        targets.value = vec!["auto".to_string()];
        TargetSet::Auto
    } else {
        TargetSet::QV
    };
    let dtype = retrograd_core::LoraDtype::default();
    let rank = tuning::rank(
        facts,
        tuning::bytes_per_rank(input.model, &target_set, dtype),
    );
    let alpha = tuning::alpha(rank.value);
    let learning_rate = tuning::learning_rate(rank.value);
    let weight_decay = tuning::weight_decay(facts);

    provenance.derived("lora.rank", rank.reason);
    provenance.derived("lora.targets", targets.reason);
    provenance.derived("lora.alpha", alpha.reason);
    provenance.derived("training.lr", learning_rate.reason);
    provenance.derived("training.weight_decay", weight_decay.reason);
    // Two fields nobody derives: they are the engine's own, and saying so is
    // more useful than an invented reason.
    provenance.defaulted("lora.dtype");
    provenance.defaulted("training.max_grad_norm");

    LoraDraft {
        targets: targets.value,
        rank: rank.value,
        alpha: alpha.value,
        learning_rate: learning_rate.value,
        weight_decay: weight_decay.value,
    }
}

/// What [`draft_lora`] decided, with the reasons already recorded in the
/// provenance map.
struct LoraDraft {
    targets: Vec<String>,
    rank: u32,
    alpha: f32,
    learning_rate: f32,
    weight_decay: f32,
}

/// The rollout half of the draft: how wide one update is.
///
/// `group_size` and `prompts_per_update` are bounded by what the *generation*
/// KV cache can hold, so they are derived from the budget and not tabulated.
/// Every value is zero outside a rollout objective - those fields do not exist
/// in the document, and recording them would put paths nobody can set into the
/// provenance map.
fn draft_rollout_width(
    input: &ResolveInput<'_>,
    n_ctx: u32,
    budgets: &Budgets,
    provenance: &mut Provenance,
) -> (RolloutShape, RolloutWidth) {
    let recipe = input.recipe;
    // `group_size` and `prompts_per_update` are bounded by what the *generation*
    // KV cache can hold, so they are derived from the budget and not tabulated.
    let mut rollout = RolloutShape::default();
    let (group_size, prompts_per_update, updates, inner_epochs) = if recipe.objective.is_rollout() {
        // The generation cache is F16 unless the client turned the fast sampling
        // context off, in which case it inherits `kv_dtype` - which is F16 too
        // now, so only an explicit `f32` widens this.
        let inherits_optimizer_cache =
            param_bool(input.params, "training", "fast_sampling_context") == Some(false);
        let element_bytes = if inherits_optimizer_cache
            && param_str(input.params, "training", "kv_dtype") == Some("f32")
        {
            4
        } else {
            2
        };
        let capacity = tuning::generation_capacity(
            input.model,
            n_ctx,
            budgets.vram.effective_bytes,
            element_bytes,
        );
        let algorithm = recipe.objective.algorithm();
        let is_grpo = algorithm == "grpo";
        // PPO does not group: one rollout per prompt, so there is nothing to
        // divide the generation capacity by and no `group_size` field to fill.
        let group = if !is_grpo {
            1
        } else {
            match param_u64(input.params, algorithm, "group_size") {
                Some(pinned) => pinned.max(1) as usize,
                None => {
                    let derived = tuning::group_size(capacity);
                    provenance.derived(format!("{algorithm}.group_size"), derived.reason);
                    derived.value
                }
            }
        };
        let prompts_key = if is_grpo {
            "prompts_per_update"
        } else {
            "rollout_batch_size"
        };
        let prompts = match param_u64(input.params, algorithm, prompts_key) {
            Some(pinned) => pinned.max(1) as usize,
            None => {
                let derived = tuning::prompts_per_update(group, capacity);
                provenance.derived(format!("{algorithm}.{prompts_key}"), derived.reason);
                derived.value
            }
        };
        let updates = budgeted_updates(input, prompts, provenance);
        let inner = tuning::inner_epochs(updates);
        let inner_key = if is_grpo { "grpo_epochs" } else { "ppo_epochs" };
        provenance.derived(format!("{algorithm}.{inner_key}"), inner.reason);
        // What caps the generation concurrency. Zero outside GRPO on purpose:
        // `training.generation_concurrency` is a GRPO-only field
        // (`retrograd_config::build` refuses it elsewhere, and `write_back`
        // therefore never writes it), so letting phase 2bis set it for PPO
        // would report a `defaults_applied` entry the effective configuration
        // does not carry.
        rollout.group_size = if is_grpo { group as u32 } else { 0 };
        (group, prompts, updates, inner.value)
    } else {
        // Not a rollout: these fields do not exist in the document, and recording
        // them would put paths nobody can set into the provenance map.
        (0, 0, 0, 0)
    };
    (
        rollout,
        RolloutWidth {
            group_size,
            prompts_per_update,
            updates,
            inner_epochs,
        },
    )
}

/// One update's shape, as the draft document spells it.
struct RolloutWidth {
    group_size: usize,
    prompts_per_update: usize,
    updates: u32,
    inner_epochs: u32,
}

fn budgeted_epochs(
    input: &ResolveInput<'_>,
    n_ctx: u32,
    facts: &tuning::Facts,
    provenance: &mut Provenance,
    warnings: &mut Vec<PlanWarning>,
) -> u32 {
    let derive = |provenance: &mut Provenance| {
        let choice = tuning::epochs(facts);
        provenance.derived("training.epochs", choice.reason);
        choice.value
    };
    let Some(budget) = input.recipe.budget else {
        return derive(provenance);
    };
    if let Some(epochs) = budget.epochs {
        // `Derived`, not `Override`: the caller named `budget.epochs`, not
        // `params.training.epochs`, and `Override` is what locks a field against
        // every later phase. The reason names the field it came from, which is
        // the part a reader of the provenance cannot guess.
        provenance.derived(
            "training.epochs",
            format!("budget.epochs = {epochs}, as the recipe asked"),
        );
        return epochs;
    }
    if let Some(minutes) = budget.minutes {
        let tokens_per_second =
            EFFECTIVE_FLOPS / (6.0 * (input.model.n_params.max(1) as f64)).max(1.0);
        let tokens_per_epoch = (input.data.examples * n_ctx as u64) as f64;
        let epochs = ((minutes as f64 * 60.0 * tokens_per_second) / tokens_per_epoch.max(1.0))
            .floor()
            .clamp(1.0, u32::MAX as f64) as u32;
        provenance.derived(
            "training.epochs",
            format!(
                "{minutes} minutes at an estimated {tokens_per_second:.0} tokens/s \
                 for a {}-parameter model",
                input.model.n_params
            ),
        );
        // The throughput figure is an order of magnitude, not a measurement, and
        // this is the only place a plan depends on one.
        warn(
            warnings,
            "duration_budget_is_estimated",
            Some("budget.minutes"),
            "a minutes budget was converted with an assumed throughput, not a measured \
             one; the resolved epochs are what the run obeys",
        );
        return epochs;
    }
    derive(provenance)
}

fn budgeted_updates(
    input: &ResolveInput<'_>,
    prompts_per_update: usize,
    provenance: &mut Provenance,
) -> u32 {
    let algorithm = input.recipe.objective.algorithm();
    match input.recipe.budget.and_then(|budget| budget.updates) {
        Some(updates) => {
            // Same reasoning as `budgeted_epochs`: the recipe's budget is not a
            // `params` leaf, so it derives the field rather than locking it.
            provenance.derived(
                format!("{algorithm}.updates"),
                format!("budget.updates = {updates}, as the recipe asked"),
            );
            updates
        }
        None => {
            let choice = tuning::updates(input.data.examples, prompts_per_update);
            provenance.derived(format!("{algorithm}.updates"), choice.reason);
            choice.value
        }
    }
}

// ---------------------------------------------------------------------------
// Feasible geometry
// ---------------------------------------------------------------------------

/// Picks the geometry.
///
/// The objective is "the largest geometry whose estimate fits **with the safety
/// margin**", not "the largest that fits the raw budget".
/// That is what [`Budgets::overflow`] already compares against - `effective_bytes`
/// is the total less the baseline *and* less [`MarginPolicy`] - so the search
/// below stops at the first candidate inside the margin, which is where a run
/// that has to survive fragmentation wants to be.
fn choose_geometry(
    training: &mut TrainConfig,
    sizing: &Sizing<'_, '_>,
    limits: &Limits,
    provenance: &mut Provenance,
) {
    let &Sizing {
        input,
        algorithm,
        budgets,
        lora,
        workload,
        is_locked,
    } = sizing;
    let constraints = Constraints {
        n_ctx: training.n_ctx,
        min_batch: limits.min_batch,
        // A rollout run needs one optimizer step per rollout, so its batch is
        // the whole row and never a memory lever.
        whole_row_batch: limits.whole_row_batch,
        fixed_batch: is_locked("training.gradient_accumulation").then_some(training.n_batch),
        fixed_ubatch: is_locked("training.micro_batch").then_some(training.n_ubatch),
    };
    let candidates = candidates(constraints);
    let footprint = |candidate: &Geometry| {
        let mut probe = training.clone();
        probe.n_batch = candidate.n_batch;
        probe.n_ubatch = candidate.n_ubatch;
        let usage = estimate(
            input.model,
            &probe,
            lora,
            workload,
            calibration_for(input, algorithm, &probe),
        );
        let resources = usage.resources();
        (resources.device_peak_bytes, resources.host_peak_bytes)
    };
    // Largest first, so the first one that fits is the biggest one that does.
    let mut chosen = candidates.iter().copied().find(|candidate| {
        let (device, host) = footprint(candidate);
        budgets.overflow(device, host).fits()
    });
    if chosen.is_none() {
        // Nothing fits yet. Hand phase 3 the *largest* geometry, not the
        // smallest: shrinking the micro-batch is a late lever, and it is
        // deliberately ranked below chunked cross-entropy and gradient
        // checkpointing. Jumping straight to `n_ubatch = 1` here would apply a
        // late lever before an early one and hand back a configuration that
        // trades far more throughput than it had to.
        chosen = candidates.first().copied();
    }
    let Some(geometry) = chosen else {
        // The constraints are unsatisfiable; the rebuild will report it as the
        // validation error it is rather than guessing here.
        return;
    };
    if !is_locked("training.gradient_accumulation") {
        training.n_batch = geometry.n_batch;
        provenance.derived(
            "training.gradient_accumulation",
            "largest optimizer window whose estimate fits the budget and its safety margin",
        );
    }
    if !is_locked("training.micro_batch") {
        training.n_ubatch = geometry.n_ubatch;
        provenance.derived(
            "training.micro_batch",
            "divides the resolved optimizer window",
        );
    }
}

// ---------------------------------------------------------------------------
// Proactive defaults
// ---------------------------------------------------------------------------

/// Applies the active defaults, in order, on what phase 2 chose.
///
/// The estimate is recomputed before each row, so a default that lowers the
/// activation term is visible to the one that reads it. Nothing here is
/// conditional on the run overflowing: that is exactly the difference from phase
/// 3.
fn apply_defaults(
    training: &mut TrainConfig,
    sizing: &Sizing<'_, '_>,
    group_size: u32,
    provenance: &mut Provenance,
    warnings: &mut Vec<PlanWarning>,
) -> Vec<AppliedDefault> {
    let &Sizing {
        input,
        algorithm,
        budgets,
        lora,
        workload,
        is_locked,
    } = sizing;
    let mut applied = Vec::new();
    for entry in ACTIVE_DEFAULTS {
        if entry.is_locked_by(is_locked) {
            continue;
        }
        let usage = estimate(
            input.model,
            training,
            lora,
            workload,
            calibration_for(input, algorithm, training),
        );
        let element_bytes = if training.fast_generation_context {
            2
        } else {
            match training.kv_dtype {
                retrograd_core::KvDtype::F16 => 2,
                retrograd_core::KvDtype::F32 => 4,
            }
        };
        let context = defaults::Context {
            n_layer: input.model.n_layer,
            n_ctx: training.n_ctx,
            vram_bytes: budgets.vram.effective_bytes,
            activation_bytes: usage.activation_bytes,
            logits_bytes: usage.logits_bytes,
            backend: input.hardware.backend,
            is_rollout: input.recipe.objective.is_rollout(),
            group_size,
            generation_capacity: tuning::generation_capacity(
                input.model,
                training.n_ctx,
                budgets.vram.effective_bytes,
                element_bytes,
            ),
        };
        match entry.verdict(&context) {
            Verdict::Skip => {}
            Verdict::Unavailable(message) => {
                if let Some(code) = entry.unavailable_warning {
                    warn(warnings, code, entry.touches.first().copied(), message);
                }
            }
            Verdict::Apply => {
                let Some(step) = entry.apply(training, &context, is_locked) else {
                    continue;
                };
                for path in entry.touches {
                    if is_locked(path) {
                        continue;
                    }
                    // Keyed by field, and the reason is the *condition* - which
                    // is the answer to "why is this on when I did not ask for
                    // it", the question this phase exists to be able to answer.
                    provenance.derived(*path, format!("active by default: {}", entry.condition));
                }
                applied.push(step);
            }
        }
    }
    applied
}

// ---------------------------------------------------------------------------
// Memory recovery
// ---------------------------------------------------------------------------

/// What phase 3 did: the levers it pulled, and the ones it could not.
struct LeverOutcome {
    applied: Vec<AppliedLever>,
    /// Levers a lock forbade - reported so the refusal can say the run was
    /// over budget *and* which rescue the caller had ruled out.
    blocked: Vec<&'static str>,
}

fn apply_levers(
    training: &mut TrainConfig,
    sizing: &Sizing<'_, '_>,
    limits: &Limits,
    provenance: &mut Provenance,
) -> Result<LeverOutcome, ResolveError> {
    let &Sizing {
        input,
        algorithm,
        budgets,
        lora,
        workload,
        is_locked,
    } = sizing;
    let mut applied = Vec::new();
    let mut blocked: Vec<&'static str> = Vec::new();
    let fits = |training: &TrainConfig| {
        let usage = estimate(
            input.model,
            training,
            lora,
            workload,
            calibration_for(input, algorithm, training),
        );
        budgets
            .overflow(
                usage.resources().device_peak_bytes,
                usage.resources().host_peak_bytes,
            )
            .fits()
    };
    if fits(training) {
        return Ok(LeverOutcome { applied, blocked });
    }

    for lever in LEVERS {
        if fits(training) {
            break;
        }
        if lever.is_locked_by(is_locked) {
            // Note it rather than skip it silently: if the resolution ends up
            // failing, this is the honest reason, and a client-supplied
            // parameter is never moved to make room.
            blocked.extend(lever.touches.iter().copied().filter(|path| is_locked(path)));
            continue;
        }
        let overflows = |training: &TrainConfig| !fits(training);
        if let Some(required) = lever.requires
            && !input.recipe.allows(required)
        {
            // Only propose an opt-in that actually closes the gap: one
            // suggested for nothing is noise, and the caller would accept a
            // degradation for no benefit.
            let mut probe = training.clone();
            let moved = lever.pull_while(&mut probe, limits, &overflows).is_some();
            if moved && fits(&probe) {
                return Err(ResolveError::NeedsOptIn {
                    allow: required,
                    message: format!(
                        "'{}' would make the run fit, at the cost of: {}",
                        lever.id, lever.cost
                    ),
                });
            }
            continue;
        }
        // Exhaust this lever before reaching for a costlier one.
        let Some(step) = lever.pull_while(training, limits, &overflows) else {
            continue;
        };
        for path in lever.touches {
            // Keyed by field, so the reason names the lever rather than
            // whichever of its steps happened to be last: how far it went is in
            // `plan.levers`.
            provenance.derived(*path, format!("set by the '{}' memory lever", lever.id));
        }
        applied.push(step);
    }
    Ok(LeverOutcome { applied, blocked })
}

// ---------------------------------------------------------------------------
// Document round-trip
// ---------------------------------------------------------------------------

fn apply_params(document: &mut ConfigDocument, params: &Value) -> Result<(), ResolveError> {
    if params.is_null() || params.as_object().is_some_and(|map| map.is_empty()) {
        return Ok(());
    }
    let mut tree = serde_json::to_value(&*document).map_err(|error| ResolveError::Invalid {
        message: format!("the resolved configuration could not be serialized: {error}"),
        path: None,
    })?;
    deep_merge(&mut tree, params)?;
    *document = serde_json::from_value(tree).map_err(|error| ResolveError::Invalid {
        message: format!("params do not fit the configuration schema: {error}"),
        path: None,
    })?;
    Ok(())
}

/// Copies phases 2, 2bis and 3 back into the document, skipping every locked
/// field.
fn write_back(
    document: &mut ConfigDocument,
    training: &TrainConfig,
    is_locked: &dyn Fn(&str) -> bool,
) {
    let section = &mut document.training;
    macro_rules! set {
        ($path:literal, $field:ident, $value:expr_2021) => {
            if !is_locked($path) {
                section.$field = Some($value);
            }
        };
    }
    set!("training.ctx", ctx, training.n_ctx);
    set!("training.micro_batch", micro_batch, training.n_ubatch);
    // The document spells the optimizer window as a micro-batch count, so what
    // phase 2 resolved as `n_batch` tokens is written back divided.
    set!(
        "training.gradient_accumulation",
        gradient_accumulation,
        training.gradient_accumulation()
    );
    set!("training.kv_dtype", kv_dtype, training.kv_dtype);
    set!(
        "training.fast_sampling_context",
        fast_sampling_context,
        training.fast_generation_context
    );
    set!(
        "training.chunked_cross_entropy",
        chunked_cross_entropy,
        training.chunked_cross_entropy
    );
    set!(
        "training.chunked_ce_tiles",
        chunked_ce_tiles,
        training.chunked_ce_tiles
    );
    set!(
        "training.chunked_ce_seq_chunk",
        chunked_ce_seq_chunk,
        training.chunked_ce_seq_chunk
    );
    set!(
        "training.gradient_checkpointing",
        gradient_checkpointing,
        training.gradient_checkpointing
    );
    set!(
        "training.checkpoint_every_n_layers",
        checkpoint_every_n_layers,
        training.checkpoint_every_n_layers
    );
    set!(
        "training.checkpoint_dtype",
        checkpoint_dtype,
        training.checkpoint_dtype
    );
    // The candidate search costs a fanout for every rollout objective, and
    // `retrograd-config` validates the field for `[grpo]` *and* `[agent]`
    // (both reach `grpo_geometry`). Writing it for `[grpo]` only would emit an
    // agentic document that runs a different packing than the plan describes.
    let is_grpo = document.grpo.is_some();
    let packs_a_group = is_grpo || document.agent.is_some();
    if packs_a_group && !is_locked("training.shared_prefix_fanout") {
        document.training.shared_prefix_fanout = Some(match training.shared_prefix_fanout {
            SharedPrefixFanout::Auto => SharedPrefixFanoutToml::Name("auto".to_string()),
            SharedPrefixFanout::Off => SharedPrefixFanoutToml::Name("off".to_string()),
            SharedPrefixFanout::Max => SharedPrefixFanoutToml::Name("max".to_string()),
            SharedPrefixFanout::Exact(value) => SharedPrefixFanoutToml::Exact(value),
        });
    }
    // `generation_concurrency` and `generation_batch` are GRPO-only: setting
    // either anywhere else is a validation error, not a lever.
    if is_grpo && !is_locked("training.generation_concurrency") {
        document.training.generation_concurrency = Some(training.generation_concurrency.max(1));
    }
    if is_grpo && !is_locked("training.generation_batch") && training.generation_batch != 0 {
        document.training.generation_batch = Some(training.generation_batch);
    }
}

/// The four cadences that can only be derived once the step count exists.
///
/// Written into the document *and* into the already-built configuration: none of
/// them can turn a valid configuration invalid, and a third `config::build` to
/// carry four scalars would be the expensive way to learn nothing.
fn write_cadences(
    document: &mut ConfigDocument,
    config: &mut RunConfig,
    iterations: u64,
    total_steps: u64,
    is_locked: &dyn Fn(&str) -> bool,
    provenance: &mut Provenance,
) {
    if !is_locked("training.warmup_steps") {
        let choice = tuning::warmup_steps(total_steps);
        document.training.warmup_steps = Some(choice.value);
        config.training.warmup_steps = choice.value;
        provenance.derived("training.warmup_steps", choice.reason);
    }
    if !is_locked("training.lr_scheduler") {
        let choice = tuning::lr_scheduler(total_steps);
        // Mapped rather than parsed: `retrograd_config`'s parser is private, and
        // the rule only ever produces these two spellings - a third would fail
        // to compile here rather than resolve to something silently different.
        let scheduler = match choice.value {
            "cosine" => LrScheduler::Cosine,
            _ => LrScheduler::Constant,
        };
        document.training.lr_scheduler = Some(choice.value.to_string());
        config.training.lr_scheduler = scheduler;
        provenance.derived("training.lr_scheduler", choice.reason);
    }
    if !is_locked("evaluation.every_iterations")
        && let Some(evaluation) = document.evaluation.as_mut()
    {
        let choice = tuning::eval_every(iterations);
        evaluation.every_iterations = Some(choice.value);
        if let Some(built) = config.evaluation.as_mut() {
            built.every_iterations = choice.value;
        }
        provenance.derived("evaluation.every_iterations", choice.reason);
    }
    if !is_locked("checkpoint.every_steps")
        && let Some(checkpoint) = document.checkpoint.as_mut()
    {
        let choice = tuning::checkpoint_every(total_steps);
        checkpoint.every_steps = Some(choice.value);
        if let Some(built) = config.checkpoint.as_mut() {
            built.every_steps = Some(choice.value);
        }
        provenance.derived("checkpoint.every_steps", choice.reason);
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

// The rollout count is a `usize` product of two numbers the config file
// supplies, carried as `u64` into the memory budget. Truncating it to `u32` is
// the cast shape the numeric-conversion convention calls out: `2^32 + 8`
// rollouts became `8`. Capping it at `u32::MAX` would still underprice every
// larger workload, so the conversion preserves the full count and saturates
// only at `u64::MAX`.
fn workload_kind_of(config: &RunConfig) -> WorkloadKind {
    match &config.algorithm {
        retrograd_config::Algorithm::Sft(_) => WorkloadKind::Sft,
        retrograd_config::Algorithm::Ppo(ppo) => WorkloadKind::Rollout {
            rollouts_per_update: saturating_dim(ppo.rollout_batch_size),
        },
        retrograd_config::Algorithm::Grpo(grpo) => WorkloadKind::Rollout {
            rollouts_per_update: saturating_dim_product(grpo.prompts_per_update, grpo.group_size),
        },
        // Same shape as GRPO's, `samples_per_prompt` playing `group_size`. What
        // this figure does *not* carry is the teacher, which occupies its own
        // weights and KV cache alongside the student for the whole run; the
        // memory budget below is therefore optimistic by exactly that much.
        // Offline top-k distillation generates nothing - it is a supervised pass
        // over a fixed corpus whose targets happen to be sparse distributions -
        // so it is sized as SFT is.
        retrograd_config::Algorithm::Distill(distill) if !distill.mode.is_rollout() => {
            WorkloadKind::Sft
        }
        retrograd_config::Algorithm::Distill(distill) => WorkloadKind::Rollout {
            rollouts_per_update: saturating_dim_product(
                distill.prompts_per_update,
                distill.samples_per_prompt,
            ),
        },
        // A trajectory is a rollout, so the shape is the same; what differs is
        // its length, and the resolver already reads that from the geometry.
        retrograd_config::Algorithm::AgentGrpo(agent) => WorkloadKind::Rollout {
            rollouts_per_update: saturating_dim_product(
                agent.config.scenarios_per_update,
                agent.config.group_size,
            ),
        },
    }
}

fn completion_bound(config: &RunConfig) -> u32 {
    match &config.algorithm {
        retrograd_config::Algorithm::Sft(_) => 0,
        retrograd_config::Algorithm::Ppo(ppo) => ppo.sampling.max_new_tokens,
        retrograd_config::Algorithm::Grpo(grpo) => grpo.sampling.max_new_tokens,
        // Offline distillation completes nothing; its bound is SFT's.
        retrograd_config::Algorithm::Distill(distill) if !distill.mode.is_rollout() => 0,
        retrograd_config::Algorithm::Distill(distill) => distill.sampling.max_new_tokens,
        retrograd_config::Algorithm::AgentGrpo(agent) => {
            agent.config.limits.max_new_tokens_per_turn
        }
    }
}

fn fanout_state(value: SharedPrefixFanout) -> String {
    match value {
        SharedPrefixFanout::Auto => "auto".to_string(),
        SharedPrefixFanout::Off => "off".to_string(),
        SharedPrefixFanout::Max => "max".to_string(),
        SharedPrefixFanout::Exact(value) => value.to_string(),
    }
}

fn record_candidate_transition(
    before: &TrainConfig,
    after: &TrainConfig,
    is_locked: &dyn Fn(&str) -> bool,
    provenance: &mut Provenance,
    applied: &mut Vec<AppliedLever>,
) {
    if before.shared_prefix_fanout != after.shared_prefix_fanout
        && !is_locked("training.shared_prefix_fanout")
    {
        provenance.derived(
            "training.shared_prefix_fanout",
            "Pareto selection across packed physical width, passes and memory peak",
        );
        applied.push(AppliedLever {
            id: "shared_prefix_fanout",
            from: fanout_state(before.shared_prefix_fanout),
            to: fanout_state(after.shared_prefix_fanout),
            note: "the shared prompt is copied once per physical subgroup",
            cost: "peak memory versus packed recomputation passes",
        });
    }
    if before.n_ubatch != after.n_ubatch && !is_locked("training.micro_batch") {
        provenance.derived(
            "training.micro_batch",
            "Pareto selection across physical passes, memory peak and packed fanout",
        );
        applied.push(AppliedLever {
            id: "micro_batch",
            from: before.n_ubatch.to_string(),
            to: after.n_ubatch.to_string(),
            note: "a shorter physical micro-batch",
            cost: "additional physical passes",
        });
    }
    // A locked field cannot have moved, and reporting a lever for one would
    // credit the search with a decision the caller made.
    if (before.gradient_checkpointing != after.gradient_checkpointing
        || before.checkpoint_every_n_layers != after.checkpoint_every_n_layers)
        && !is_locked("training.gradient_checkpointing")
        && !is_locked("training.checkpoint_every_n_layers")
    {
        provenance.derived(
            "training.gradient_checkpointing",
            "Pareto selection across activation memory, recompute and fidelity",
        );
        applied.push(AppliedLever {
            id: "gradient_checkpointing",
            from: if before.gradient_checkpointing {
                format!("every {} layers", before.checkpoint_every_n_layers)
            } else {
                "off".to_string()
            },
            to: if after.gradient_checkpointing {
                format!("every {} layers", after.checkpoint_every_n_layers)
            } else {
                "off".to_string()
            },
            note: "layer activations are recomputed instead of retained",
            cost: "one extra forward pass per segment",
        });
    }
    if before.checkpoint_dtype != after.checkpoint_dtype && !is_locked("training.checkpoint_dtype")
    {
        provenance.derived(
            "training.checkpoint_dtype",
            "Pareto selection across checkpoint memory and update fidelity",
        );
        applied.push(AppliedLever {
            id: "checkpoint_dtype",
            from: format!("{:?}", before.checkpoint_dtype).to_ascii_lowercase(),
            to: format!("{:?}", after.checkpoint_dtype).to_ascii_lowercase(),
            note: "the retained activations use a narrower dtype",
            cost: "recompute bit-parity",
        });
    }
}

fn workload_of(input: &ResolveInput<'_>, config: &RunConfig) -> Workload {
    Workload {
        kind: workload_kind_of(config),
        examples: input.data.examples,
        co_resident_bytes: co_resident_bytes(config, input.teacher),
    }
}

/// Device bytes a second model holds beside the one being sized.
///
/// Non-zero for exactly one algorithm, and only when its geometry was supplied.
/// Zero on a `distill` document whose teacher was not inspected is a *known*
/// under-estimate, not a claim that there is no teacher: `distill_warnings`
/// says so on the same resolution, because a budget that is short by a whole
/// model and silent about it is worse than one that refuses.
fn co_resident_bytes(config: &RunConfig, teacher: Option<&ModelInfo>) -> u64 {
    match (&config.algorithm, teacher) {
        // The offline mode never opens the teacher - the sidecar is what it left
        // behind - so charging its weights to the device would refuse
        // configurations that run.
        (retrograd_config::Algorithm::Distill(distill), Some(teacher))
            if distill.mode.is_rollout() =>
        {
            crate::cost::co_resident_model_bytes(teacher, &config.training)
        }
        _ => 0,
    }
}

/// `(iterations, total_steps)`.
fn step_counts(input: &ResolveInput<'_>, config: &RunConfig) -> (u64, u64) {
    let steps_per_row = (config.training.n_ctx / config.training.n_batch.max(1)).max(1) as u64;
    match &config.algorithm {
        retrograd_config::Algorithm::Sft(_) => {
            let epochs = config.training.epochs as u64;
            (epochs, epochs * input.data.examples.max(1) * steps_per_row)
        }
        retrograd_config::Algorithm::Ppo(ppo) => {
            let rollouts = ppo.rollout_batch_size as u64;
            (
                ppo.updates as u64,
                ppo.updates as u64 * ppo.ppo_epochs as u64 * rollouts * steps_per_row,
            )
        }
        retrograd_config::Algorithm::Grpo(grpo) => {
            let rollouts = (grpo.prompts_per_update * grpo.group_size) as u64;
            (
                grpo.updates as u64,
                grpo.updates as u64 * grpo.grpo_epochs as u64 * rollouts * steps_per_row,
            )
        }
        // An offline run's iteration is an epoch over the corpus, which the
        // resolver cannot count without reading the corpus - the same position it
        // is in for SFT, and it answers the same way.
        retrograd_config::Algorithm::Distill(distill) if !distill.mode.is_rollout() => {
            let epochs = distill
                .mode
                .offline()
                .map(|offline| offline.epochs)
                .unwrap_or(1) as u64;
            (epochs, epochs * steps_per_row)
        }
        retrograd_config::Algorithm::Distill(distill) => {
            let rollouts = (distill.prompts_per_update * distill.samples_per_prompt) as u64;
            (
                distill.updates as u64,
                distill.updates as u64 * distill.distill_epochs as u64 * rollouts * steps_per_row,
            )
        }
        retrograd_config::Algorithm::AgentGrpo(agent) => {
            let config = &agent.config;
            let rollouts = (config.scenarios_per_update * config.group_size) as u64;
            (
                config.updates as u64,
                config.updates as u64 * config.epochs as u64 * rollouts * steps_per_row,
            )
        }
    }
}

/// Judge requests over the whole run - a figure the plan carries,
/// because a judge costs money per call.
fn judge_calls(input: &ResolveInput<'_>, config: &RunConfig) -> u64 {
    let Some(judge) = &input.recipe.judge else {
        return 0;
    };
    let retrograd_config::Algorithm::Grpo(grpo) = &config.algorithm else {
        return 0;
    };
    let per_group = match judge.mode.as_deref() {
        Some("pairwise") => judge
            .max_pairs
            .map(u64::from)
            // Without a cap, pairwise compares every pair of the group.
            .unwrap_or_else(|| (grpo.group_size * (grpo.group_size - 1) / 2) as u64),
        // One listwise request per group, plus its chunked fallback in the worst
        // case; `auto` and `chunked` both land here.
        _ => 2,
    };
    grpo.updates as u64 * grpo.prompts_per_update as u64 * per_group
}

fn collect_warnings(
    config: &RunConfig,
    teacher: Option<&ModelInfo>,
    backend: Backend,
    warnings: &mut Vec<PlanWarning>,
) {
    // Every backend this crate can name carries both fused cross-entropy nodes,
    // so the only run whose fused tail might not be on the device is one whose
    // accelerator nobody recognised. Whether it really does is a probe
    // (`cap_fused_sparse_ce`), and the estimate below assumes it does.
    if config.training.chunked_cross_entropy && backend == Backend::Unknown {
        warn(
            warnings,
            "chunked_ce_on_an_unrecognised_backend",
            Some("training.chunked_cross_entropy"),
            "this device's backend was not recognised, so whether it runs both fused \
             cross-entropy nodes is unknown; the estimate assumes it does, and \
             training.require_gpu_resident makes a CPU fallback fail the preflight instead \
             of running quietly",
        );
    }
    if config.training.kv_dtype == retrograd_core::KvDtype::F16 {
        warn(
            warnings,
            "kv_f16_may_fall_back",
            Some("training.kv_dtype"),
            "kv_dtype = f16 is the default and falls back to F32 on a device without \
             differentiable flash attention for this head geometry; the estimate then \
             understates the KV cache",
        );
    }
    if config.training.gradient_checkpointing
        && config.training.checkpoint_dtype != retrograd_core::CheckpointDtype::F32
    {
        // The one place a 16-bit checkpoint is visible before the run: it costs
        // ~1e-3 of the update, four orders of magnitude more than the F32
        // recompute's reassociation error, and nothing downstream distinguishes
        // a perturbed update from a clean one.
        warn(
            warnings,
            "checkpoint_dtype_perturbs_the_update",
            Some("training.checkpoint_dtype"),
            "a 16-bit activation checkpoint perturbs the LoRA update by roughly its own \
             relative precision (~1e-3); set training.checkpoint_dtype = 'f32' for a \
             bit-exact recompute",
        );
    }
    // Only the on-policy mode holds the teacher. An offline run reads a sidecar
    // and never opens the model, so warning that the budget is short of a teacher
    // would be false there.
    let resident_teacher = matches!(
        &config.algorithm,
        retrograd_config::Algorithm::Distill(distill) if distill.mode.is_rollout()
    );
    if resident_teacher && teacher.is_none() {
        // The one gap this resolver knows it has, said where it happens.
        // A distillation run holds the teacher's weights and KV beside the
        // student for its whole length; without the teacher's geometry the
        // budget below is short by exactly that, and a plan that fits by a
        // margin thinner than a model is not a plan. Not a refusal, because the
        // rest of the resolution - geometry, schedule, levers - is still right,
        // and the caller may have a reason to size the student alone.
        warn(
            warnings,
            "teacher_absent_from_the_memory_budget",
            Some("distill.teacher_path"),
            "the memory estimate does not include the distillation teacher, which stays \
             resident beside the student for the whole run: supply the teacher's geometry to \
             size it, or read the device figure as a lower bound",
        );
    }
    let generates = match &config.algorithm {
        retrograd_config::Algorithm::Ppo(_)
        | retrograd_config::Algorithm::Grpo(_)
        | retrograd_config::Algorithm::AgentGrpo(_) => true,
        // An offline distillation run has no generation context to make fast.
        retrograd_config::Algorithm::Distill(distill) => distill.mode.is_rollout(),
        retrograd_config::Algorithm::Sft(_) => false,
    };
    if config.training.fast_generation_context && generates {
        // The one warning invariant 4 now leans on: it is emitted from the final
        // configuration, so it fires whether phase 2bis turned the setting on or
        // the client did.
        warn(
            warnings,
            "sampling_distribution_approximated",
            Some("training.fast_sampling_context"),
            "fast_sampling_context makes the sampling distribution differ from the trained \
             policy by rounding; set training.fast_sampling_context = false for exact parity",
        );
    }
    if matches!(config.lora.config.targets, TargetSet::Auto) {
        warn(
            warnings,
            "lora_targets_resolved_by_runtime",
            Some("lora.targets"),
            "the LoRA target set is resolved per architecture by the runtime; the estimate \
             assumes every attention and feed-forward projection",
        );
    }
}

fn param_u32(params: &Value, section: &str, key: &str) -> Option<u32> {
    param_u64(params, section, key).and_then(|value| u32::try_from(value).ok())
}

fn param_u64(params: &Value, section: &str, key: &str) -> Option<u64> {
    params.get(section)?.get(key)?.as_u64()
}

fn param_bool(params: &Value, section: &str, key: &str) -> Option<bool> {
    params.get(section)?.get(key)?.as_bool()
}

fn param_str<'a>(params: &'a Value, section: &str, key: &str) -> Option<&'a str> {
    params.get(section)?.get(key)?.as_str()
}

/// The path a validation error refers to, when the resolver can name one.
pub fn path_of(error: &ResolveError) -> Option<&str> {
    match error {
        ResolveError::Invalid { path, .. } => path.as_deref(),
        ResolveError::OverrideConflict { path, .. } => Some(path),
        _ => None,
    }
}
