//! The seam between an HTTP request and `retrograd-plan`.
//!
//! Everything that needs the server - validating client paths, reading the model
//! geometry through the device semaphore, substituting catalogue ids for the
//! commands only the operator may set, and redacting them again on the way out,
//! happens here. The resolution itself is a pure call into the plan crate.

pub mod calibrate;
pub mod fork;

use std::path::PathBuf;

use retrograd_config::{ConfigDocument, RunConfig};
use retrograd_core::{CatalogContract, Device, ModelInfo};
use retrograd_dataset::DataFormat;
use retrograd_plan::recipe::{DataSpec, Recipe};
use retrograd_plan::resolver::{Resolution, ResolveError, ResolveInput};
use retrograd_plan::{DatasetStats, MemoryEstimate};
use serde_json::Value;

use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::lock::Recover as _;
use crate::state::AppState;

/// The generated RIR catalogue, parsed and validated once.
///
/// It is compiled in, so there is no I/O - but it is ~140 KB of JSON, and both
/// `GET /v1/capabilities` and every execution profile ask for it. Parsing it per
/// request would spend that on every call to answer with a constant.
pub(crate) fn kernel_catalog() -> ApiResult<&'static retrograd_core::KernelCatalog> {
    static CATALOG: std::sync::OnceLock<Result<retrograd_core::KernelCatalog, String>> =
        std::sync::OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let catalog: retrograd_core::KernelCatalog =
                serde_json::from_str(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../generated/rir/catalog/catalog.json"
                )))
                .map_err(|error| error.to_string())?;
            // The cell is read by reference forever after, so what it holds has
            // to be shareable: the typed `ContractError` is rendered here, at
            // the one boundary that cannot hand out an owned error.
            catalog.validate().map_err(|error| error.to_string())?;
            Ok(catalog)
        })
        .as_ref()
        .map_err(|error| ApiError::internal(format!("generated RIR catalogue is invalid: {error}")))
}

/// How many snapshots each planning cache keeps.
///
/// Both entries are large - a profile carries the whole kernel list - and both
/// keys grow without bound in normal use: one profile per (model, mtime, device),
/// one preflight per resolved shape, and the calibration loop resolves several
/// shapes per request. A server that plans all day would otherwise never give any
/// of it back.
const PLANNING_CACHE_ENTRIES: usize = 32;

/// Remembers `value`, bounded. The cache is emptied rather than evicted
/// one-by-one when it is full: every entry is recomputable from the engine, so
/// the cost of being wrong about which one to drop is one cold probe, and an
/// insertion-order queue for that is machinery this does not need.
///
/// Poisoning is recovered, like every other lock in this crate: the map holds
/// only recomputable entries, so a panicking writer leaves nothing half-applied
/// and refusing to plan afterwards would turn one failed request into an outage.
fn remember<T>(
    cache: &std::sync::RwLock<std::collections::BTreeMap<String, T>>,
    key: String,
    value: T,
) {
    let mut cache = cache.write().recover();
    if cache.len() >= PLANNING_CACHE_ENTRIES {
        cache.clear();
    }
    cache.insert(key, value);
}

/// What a redacted `reward_command` reads as in the effective configuration.
///
/// The client gets back something that is unmistakably not a command: rendering
/// the real one would defeat the contract, and rendering an empty list would look like a
/// resolver bug.
fn redacted(id: &str) -> Vec<String> {
    vec![format!("<reward:{id}>")]
}

/// How many times a measured resolution may go round.
///
/// One measurement, and - if it raised a correction factor - one re-resolution
/// with the real numbers, measured again. A third pass would be a search: the
/// factors are running maxima, so a second raise means the configuration is at
/// the edge of what the machine can do, and the honest answer there is
/// `insufficient_memory` rather than a slow convergence a caller is waiting on.
const MAX_CALIBRATION_PASSES: u32 = 2;

/// What a resolution produced: the answer to send, and the configuration a run
/// would execute.
pub struct Resolved {
    pub response: dto::PlanResponse,
    /// Never redacted, unlike `response.effective_config`: this one carries the
    /// operator's reward command and is what the engine receives.
    pub config: Box<RunConfig>,
    /// Epochs for SFT, updates for a rollout algorithm. The runtime reports
    /// progress against it before the first observer event arrives.
    pub iterations: u64,
    /// A recipe omitted `output`; run creation replaces the resolver's
    /// placeholder with `<state_dir>/<run-id>/adapter.gguf`.
    pub managed_adapter: bool,
}

/// Resolves a recipe into a full plan.
pub async fn plan_recipe(
    state: &AppState,
    recipe: &Recipe,
    params: &Value,
    name: Option<String>,
    calibrate: bool,
) -> ApiResult<Resolved> {
    let mut recipe = recipe.clone();
    // `[output].path` is where a run names its result; when neither the recipe
    // nor the params name one, the server substitutes a managed path.
    let managed_adapter =
        recipe.output.is_none() && params.pointer("/output/path").is_none();
    for pointer in [
        "/model/path",
        "/sft/data",
        "/ppo/prompts",
        "/grpo/prompts",
        "/agent/scenarios",
    ] {
        if params.pointer(pointer).is_some() {
            return Err(
                ApiError::invalid("recipe params cannot replace a recipe's input sources")
                    .with_field(
                        format!("/params{pointer}"),
                        ErrorCode::OverrideConflict,
                        "declare the source in recipe instead",
                    ),
            );
        }
    }
    if let Err(issues) = recipe.validate() {
        let mut problem = ApiError::invalid("the recipe has one or more invalid fields");
        for issue in issues {
            problem = problem.with_field(issue.pointer, ErrorCode::InvalidValue, issue.message);
        }
        return Err(problem);
    }

    // Paths first: everything after this reads files.
    let model_path = state.resolve_path(&recipe.model.to_string_lossy(), "/recipe/model")?;
    let data_source = resolve_data_source(state, &mut recipe.data, "/recipe/data")?;
    let eval_source = match recipe.eval.as_mut() {
        Some(eval) => Some(resolve_data_source(state, eval, "/recipe/eval")?),
        None => None,
    };
    let data_path = data_source.path.clone();
    let eval_path = eval_source.as_ref().map(|source| source.path.clone());

    // Catalogue: the client sent ids, the resolver needs what they stand for.
    let (reward_command, reward_protocol) = match (&recipe.reward, recipe.objective.is_rollout()) {
        (Some(reference), _) => {
            let entry = state.catalog.reward(&reference.id).map_err(|error| {
                ApiError::new(ProblemKind::UnknownCatalogId, error.to_string()).with_field(
                    "/recipe/reward/id",
                    ErrorCode::UnknownCatalogId,
                    "not declared by this server",
                )
            })?;
            (entry.command.clone(), Some(entry.protocol))
        }
        // A judged run still needs a reward command slot in the document; the
        // judge itself is wired at step 9, so a judge-only recipe is refused
        // rather than silently resolved into something that would not run.
        (None, true) => {
            return Err(ApiError::invalid(
                "a reinforcement objective needs a declared reward; a judge on its own \
                 is not wired yet",
            )
            .with_field("/recipe/reward", ErrorCode::MissingField, "required"));
        }
        (None, false) => (Vec::new(), None),
    };
    if let Some(judge) = &recipe.judge {
        state.catalog.judge(&judge.id).map_err(|error| {
            ApiError::new(ProblemKind::UnknownCatalogId, error.to_string()).with_field(
                "/recipe/judge/id",
                ErrorCode::UnknownCatalogId,
                "not declared by this server",
            )
        })?;
    }
    for (index, tool) in recipe.tools.iter().enumerate() {
        state.catalog.mcp_server(tool).map_err(|error| {
            ApiError::new(ProblemKind::UnknownCatalogId, error.to_string()).with_field(
                format!("/recipe/tools/{index}"),
                ErrorCode::UnknownCatalogId,
                "not declared by this server",
            )
        })?;
    }

    let format = data_format(recipe.data.format.as_deref(), &data_path).map_err(|error| {
        error.with_field(
            "/recipe/data/format",
            ErrorCode::UnsupportedFormat,
            "expected auto, text or jsonl",
        )
    })?;
    let eval_format = match (&recipe.eval, &eval_path) {
        (Some(spec), Some(path)) => {
            Some(data_format(spec.format.as_deref(), path).map_err(|error| {
                error.with_field(
                    "/recipe/eval/format",
                    ErrorCode::UnsupportedFormat,
                    "expected auto, text or jsonl",
                )
            })?)
        }
        _ => None,
    };

    // The device decides the calibration key as much as the model does, and a
    // parameter may pin it. Reading it here keeps the factors that are *used* and
    // the factors that are *written* under one key.
    //
    // Read before the dataset statistics, not after: the model's geometry is
    // what names the tokenizer, and therefore whether a measurement of these
    // lengths already exists to be used instead of the estimate.
    let device = requested_device(params);
    let model = geometry(state, &model_path, device).await?;
    // Recipes default to LoRA, but params can select a base-weight policy.
    let needs_inventory = params
        .pointer("/training/trainable")
        .and_then(Value::as_str)
        .and_then(|value| retrograd_core::TrainablePolicy::parse(value).ok())
        .is_some_and(retrograd_core::TrainablePolicy::trains_base_weights);
    let inventory = if needs_inventory {
        tensor_inventory(state, &model_path, device).await?
    } else {
        None
    };
    let execution_profile = execution_profile(state, &model_path, device).await?;
    let tokenizer_key = if data_source.stored.is_some()
        || eval_source
            .as_ref()
            .is_some_and(|source| source.stored.is_some())
    {
        let fingerprint = state.model_fingerprint(&model_path).await?;
        Some(crate::datasets::tokenizer_key(&model, &fingerprint))
    } else {
        None
    };

    let data = dataset_stats(&data_source, &data_path, format, tokenizer_key.as_deref()).map_err(
        |error| {
            error.with_field(
                "/recipe/data/path",
                ErrorCode::InvalidValue,
                "could not be read",
            )
        },
    )?;
    // The eval dataset gets the same treatment as the training one
    // read and validated now, not discovered malformed
    // hours into a run at the first evaluation.
    let eval_data = match (&eval_source, &eval_path, eval_format) {
        (Some(source), Some(path), Some(eval_format)) => Some(
            dataset_stats(source, path, eval_format, tokenizer_key.as_deref()).map_err(
                |error| {
                    error.with_field(
                        "/recipe/eval/path",
                        ErrorCode::InvalidValue,
                        "could not be read",
                    )
                },
            )?,
        ),
        _ => None,
    };
    let mut passes = 0;
    let mut packing_tuned = false;
    let mut packing_skipped = None;
    let mut packing_measurements = retrograd_plan::packing_tuning::PackingMeasurements::new();
    let mut preflight = None;
    let mut preflight_key = None;
    let (mut resolution, measurement) = loop {
        let calibrations = state.calibration.read().recover().clone();
        let resolution = retrograd_plan::resolve(&ResolveInput {
            recipe: &recipe,
            params,
            model: &model,
            // No `Objective` resolves to `distill`, so a resolution never
            // produces a document with a teacher to size. The `/assess` path
            // below is where a client-written one arrives, and where the
            // teacher's geometry is read.
            teacher: None,
            inventory: inventory.as_ref(),
            data: &data,
            eval: eval_data.as_ref(),
            data_format: format,
            baseline: state.baseline,
            hardware: state.hardware(device),
            execution_profile: Some(&execution_profile),
            preflight: preflight.as_ref(),
            server_budgets: state.server_budgets(),
            margin: state.config.margin,
            calibrations: Some(&calibrations),
            packing_measurements: Some(&packing_measurements),
            reward_command: reward_command.clone(),
            reward_protocol,
            root: PathBuf::from("."),
        })
        .map_err(problem)?;
        state.validate_run_paths(&resolution.config, managed_adapter)?;
        let key = retrograd_plan::calibration_key(&execution_profile, &model, &resolution.config);
        if preflight_key.as_deref() != Some(key.as_str()) {
            let report = preflight_candidate(
                state,
                &model_path,
                &resolution.config,
                execution_profile.fingerprint(),
                &key,
            )
            .await?;
            preflight = Some(report);
            preflight_key = Some(key.clone());
            continue;
        }
        if !calibrate {
            break (resolution, None);
        }
        if !packing_tuned {
            packing_tuned = true;
            let tuning = calibrate::tune_packing(
                state,
                &model_path,
                &resolution.config,
                &resolution.packing_probes,
            )
            .await?;
            packing_measurements = tuning.measurements;
            packing_skipped = tuning.skipped;
            if !packing_measurements.is_empty() {
                // The resolver applies the same locks, fidelity and memory
                // rules again, now with comparable complete-update timings.
                continue;
            }
        }
        let measurement =
            calibrate::measure(state, &model_path, &model, &resolution.config, &key).await?;
        passes += 1;
        if measurement.observation.raised && passes < MAX_CALIBRATION_PASSES {
            // Reapply memory levers using the measured costs. The levers are
            // re-derived from scratch rather than patched: an estimate that grew
            // may need a *cheaper* lever than the one already applied, and
            // stacking corrections onto a chosen configuration would never find
            // it.
            preflight = None;
            preflight_key = None;
            continue;
        }
        break (resolution, Some(measurement));
    };

    if let Some(error) = packing_skipped {
        resolution.plan.warnings.push(retrograd_plan::PlanWarning {
            code: "packing_geometry_unmeasured",
            field: None,
            message: format!(
                "kept the analytical packed geometry: the optimizer benchmark could not run \
                 ({error})"
            ),
        });
    }
    if let Some(measurement) = &measurement {
        if let Some(error) = calibrate::over_budget(measurement, &resolution.plan) {
            return Err(error);
        }
        calibrate::record(
            measurement,
            &mut resolution.plan,
            &mut resolution.provenance,
        );
    }

    let iterations = resolution.plan.iterations;
    let config = Box::new(resolution.config.clone());
    Ok(Resolved {
        response: render(
            resolution,
            recipe.reward.as_ref().map(|r| r.id.as_str()),
            name,
        ),
        config,
        iterations,
        managed_adapter,
    })
}

/// The device a client parameter pins, if any. Anything unparseable is left to
/// the configuration builder, which reports it with the right pointer.
fn requested_device(params: &Value) -> Device {
    params
        .get("model")
        .and_then(|model| model.get("device"))
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<Device>().ok())
        .unwrap_or(Device::Auto)
}

/// Estimates a complete configuration supplied by the caller.
///
/// The semantic phases are skipped, the budget check is not. `force` is the
/// caller taking responsibility for an overflow, which is a legitimate choice
/// and an explicit one.
pub async fn plan_config(
    state: &AppState,
    document: &ConfigDocument,
    params: &Value,
    name: Option<String>,
    force: bool,
    calibrate: bool,
    // `reward_id`: the catalogue id to redact the reward command back to. `Some`
    // only for a fork, which resolved a parent's `<reward:id>` into the real
    // command and has to put the marker back on the way out.
    reward_id: Option<&str>,
) -> ApiResult<Resolved> {
    let mut document = document.clone();
    retrograd_plan::merge::reject_nulls(params).map_err(|error| {
        ApiError::invalid(error.message).with_field(
            format!("/params{}", error.pointer),
            ErrorCode::InvalidValue,
            "not accepted",
        )
    })?;
    if !params.as_object().is_none_or(|map| map.is_empty()) {
        let mut tree = serde_json::to_value(&document)
            .map_err(|error| ApiError::internal(format!("config could not be re-read: {error}")))?;
        retrograd_plan::merge::deep_merge(&mut tree, params).map_err(|error| {
            ApiError::invalid(error.message).with_field(
                format!("/params{}", error.pointer),
                ErrorCode::InvalidValue,
                "not applicable",
            )
        })?;
        document = crate::extract::from_json_value(tree)?;
    }

    // The API has no `--model` to fall back on: a document reaching here names
    // its base model or it names nothing runnable.
    document.model.path.as_ref().ok_or_else(|| {
        ApiError::invalid("[model].path is missing").with_field(
            "/config/model/path",
            ErrorCode::InvalidValue,
            "required",
        )
    })?;
    let config =
        retrograd_config::build(document.clone(), std::path::Path::new(".")).map_err(|error| {
            ApiError::from(error).with_field(
                "/config",
                ErrorCode::InvalidValue,
                "invalid configuration",
            )
        })?;
    state.validate_run_paths(&config, false)?;
    let model_path = config.model.clone();
    let model = geometry(state, &model_path, config.training.device).await?;
    // The per-tensor table, read only for a policy priced from one; a base
    // document that cannot get one is refused by `resolve_trainable_set`.
    let inventory = if config.training.trainable.policy.trains_base_weights() {
        tensor_inventory(state, &model_path, config.training.device).await?
    } else {
        None
    };
    let base_trainable =
        retrograd_plan::resolve_trainable_set(&config.training.trainable, inventory.as_ref())
            .map_err(|error| {
                ApiError::from(error).with_field(
                    "/config/training/trainable",
                    ErrorCode::InvalidValue,
                    "not resolvable against this model",
                )
            })?;
    let execution_profile = execution_profile(state, &model_path, config.training.device).await?;
    let key = retrograd_plan::calibration_key(&execution_profile, &model, &config);
    let preflight = preflight_candidate(
        state,
        &model_path,
        &config,
        execution_profile.fingerprint(),
        &key,
    )
    .await?;
    let factors = state.calibration.read().recover().factors(&key);

    // Only the example count matters to the estimate here; the caller already
    // chose the context, so no percentile has to be read off the data.
    let data = DatasetStats::measured(Vec::new());
    // The second model of a distillation run, inspected so the budget can carry
    // it. A `/assess` on a `distill` document that skipped this would answer a
    // device figure short by a whole model - the planner warns when it happens,
    // and this is what stops it happening here.
    let teacher = match &config.algorithm {
        // Only the on-policy mode keeps the teacher resident. Inspecting it for
        // an offline document would load a model to size a term that path does
        // not pay.
        retrograd_config::Algorithm::Distill(distill) if distill.mode.is_rollout() => {
            Some(geometry(state, &distill.teacher_path, config.training.device).await?)
        }
        _ => None,
    };
    let (estimate, budgets, overflow) = retrograd_plan::assess(
        &config,
        &model,
        teacher.as_ref(),
        base_trainable.as_ref(),
        &data,
        state.baseline,
        state.server_budgets(),
        state.config.margin,
        factors,
    );
    if !overflow.fits() && !force {
        return Err(insufficient(
            &estimate,
            &budgets,
            overflow.total(),
            &[],
            &[],
        ));
    }

    let mut provenance = retrograd_plan::Provenance::new();
    for path in retrograd_plan::merge::overridden_paths(params) {
        provenance.overridden(path);
    }
    let mut plan = retrograd_plan::PlanSummary {
        schema_version: 2,
        total_steps: 0,
        iterations: iterations(&config),
        memory: retrograd_plan::MemoryPlan {
            resources: estimate.resources(),
            measured: None,
            budgets,
        },
        execution: retrograd_plan::execution::explain_execution(
            Some(&execution_profile),
            Some(&preflight),
            state.hardware(config.training.device).backend.id(),
        ),
        defaults_applied: Vec::new(),
        levers: Vec::new(),
        truncation_fraction: 0.0,
        // Form (b)/(c) skips the semantic phases entirely - there is no
        // `recipe.eval` to have read a dataset off in the first place.
        eval_truncation_fraction: None,
        judge_calls_expected: 0,
        // Phases 1 to 3 are skipped here - the caller chose - so there is nothing
        // to announce except the one thing the caller took on: an overflow.
        warnings: if overflow.fits() {
            Vec::new()
        } else {
            vec![retrograd_plan::PlanWarning {
                code: "over_budget_accepted_by_force",
                field: None,
                message: format!(
                    "the configuration is {} bytes over budget and was accepted because \
                     force=true was set",
                    overflow.total()
                ),
            }]
        },
    };

    // A configuration the caller wrote is never re-derived, so a measurement here
    // cannot change it - it can only confirm it or refuse it. That asymmetry with
    // the recipe form is the point of forms (b) and (c): the caller chose.
    if calibrate {
        let measurement = calibrate::measure(state, &model_path, &model, &config, &key).await?;
        if !force && let Some(error) = calibrate::over_budget(&measurement, &plan) {
            return Err(error);
        }
        calibrate::record(&measurement, &mut plan, &mut provenance);
    }

    let iterations = plan.iterations;
    Ok(Resolved {
        response: dto::PlanResponse {
            name,
            effective_config: redact(document, reward_id),
            provenance,
            plan,
        },
        config: Box::new(config),
        iterations,
        managed_adapter: false,
    })
}

fn iterations(config: &retrograd_config::RunConfig) -> u64 {
    match &config.algorithm {
        retrograd_config::Algorithm::Sft(_) => config.training.epochs as u64,
        retrograd_config::Algorithm::Ppo(ppo) => ppo.updates as u64,
        retrograd_config::Algorithm::Grpo(grpo) => grpo.updates as u64,
        // An offline run's iteration is an epoch over the corpus.
        retrograd_config::Algorithm::Distill(distill) => match distill.mode.offline() {
            Some(offline) => offline.epochs as u64,
            None => distill.updates as u64,
        },
        retrograd_config::Algorithm::AgentGrpo(agent) => agent.config.updates as u64,
    }
}

fn render(
    resolution: Resolution,
    reward_id: Option<&str>,
    name: Option<String>,
) -> dto::PlanResponse {
    dto::PlanResponse {
        name,
        effective_config: redact(resolution.document, reward_id),
        provenance: resolution.provenance,
        plan: resolution.plan,
    }
}

/// Replaces every server-declared value in the effective configuration with the
/// id it came from. The configuration the client reads back is therefore *not*
/// runnable as-is, which is the point: it shows what was decided without handing
/// out what the operator declared.
fn redact(mut document: ConfigDocument, reward_id: Option<&str>) -> ConfigDocument {
    let label = reward_id
        .map(redacted)
        .unwrap_or_else(|| vec!["<reward>".to_string()]);
    if let Some(grpo) = document.grpo.as_mut() {
        grpo.reward_command = label.clone();
    }
    if let Some(ppo) = document.ppo.as_mut() {
        ppo.reward_command = label;
    }
    document
}

/// Reads the model's geometry on a blocking thread.
///
/// Behind the *probe* pool, not the device semaphore: a geometry read is a
/// host-only mmap and a header parse, so serializing it with the runs - as a
/// literal reading of the device rule would - means no plan can be made while a run
/// trains. What the rule is really about is the calibration pass, which does load the model
/// onto the device, and that one takes the device permit.
/// The model's per-tensor table, on the probe queue like every other model
/// read. `None` when the probe cannot open a GGUF.
pub(crate) async fn tensor_inventory(
    state: &AppState,
    model: &std::path::Path,
    device: Device,
) -> ApiResult<Option<retrograd_core::TensorInventory>> {
    let permit = state
        .probes
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the probe queue is closed"))?;
    let probe = state.probe.clone();
    let model = model.to_path_buf();
    let inventory = tokio::task::spawn_blocking(move || {
        let outcome = probe.tensor_inventory(&model, device);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("model probe failed: {error}")))??;
    Ok(inventory)
}

pub(crate) async fn geometry(
    state: &AppState,
    model: &std::path::Path,
    device: Device,
) -> ApiResult<ModelInfo> {
    let permit = state
        .probes
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the probe queue is closed"))?;
    let probe = state.probe.clone();
    let model = model.to_path_buf();
    let info = tokio::task::spawn_blocking(move || {
        let outcome = probe.geometry(&model, device);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("model probe failed: {error}")))??;
    Ok(info)
}

/// Immutable engine/device/catalogue snapshot for one resolution. Capability
/// probing loads the model, so it is serialized with runs and measured
/// candidates by the device semaphore.
pub(crate) async fn execution_profile(
    state: &AppState,
    model: &std::path::Path,
    device: Device,
) -> ApiResult<retrograd_core::ExecutionProfile> {
    let metadata = std::fs::metadata(model)
        .map_err(|error| ApiError::invalid(format!("could not inspect model: {error}")))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |value| value.as_nanos());
    let cache_key = format!(
        "{}:{device:?}:{}:{modified}",
        model.display(),
        metadata.len()
    );
    if let Some(profile) = state
        .execution_profiles
        .read()
        .recover()
        .get(&cache_key)
        .cloned()
    {
        return Ok(profile);
    }
    let permit = state
        .device
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the device queue is closed"))?;
    let probe = state.probe.clone();
    let model = model.to_path_buf();
    let capabilities = tokio::task::spawn_blocking(move || {
        let outcome = probe.model_capabilities(&model, device);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("capability probe failed: {error}")))??;

    let catalog = kernel_catalog()?;
    let (rir_mode, latched) = retrograd_engine::rir_runtime_policy()
        .map_err(|error| ApiError::internal(format!("could not read RIR policy: {error}")))?;
    let hardware = state.hardware(device);
    let runtime_device = retrograd_engine::backend_list()
        .ok()
        .and_then(|listing| {
            listing.lines().find_map(|line| {
                let mut fields = line.split('\t');
                let kind = fields.next()?;
                let name = fields.next().unwrap_or_default();
                let description = fields.next().unwrap_or_default();
                let selected = if device == Device::Cpu {
                    kind == "cpu"
                } else {
                    kind == "gpu"
                };
                selected.then(|| format!("{name}:{description}"))
            })
        })
        .unwrap_or_else(|| hardware.backend.id().to_string());
    let profile = retrograd_core::ExecutionProfile {
        schema_version: retrograd_core::EXECUTION_PROFILE_VERSION,
        engine_fingerprint: format!("retrograd-server:{}", env!("CARGO_PKG_VERSION")),
        kernel_catalog_fingerprint: catalog.fingerprint.clone(),
        device: retrograd_core::DeviceProfile {
            stable_id: format!(
                "{}:{}:{}",
                hardware.backend.id(),
                runtime_device,
                hardware.device_total_bytes.unwrap_or(0)
            ),
            backend: hardware.backend.id().to_string(),
            unified_memory: hardware.unified_memory,
            device_total_bytes: hardware.device_total_bytes,
            host_total_bytes: state.baseline.host_total,
            features: Vec::new(),
            limits: retrograd_core::DeviceLimits::default(),
        },
        runtime_policy: retrograd_core::RuntimePolicy { rir_mode, latched },
        kernels: catalog.kernels.clone(),
        capabilities,
    };
    remember(&state.execution_profiles, cache_key, profile.clone());
    Ok(profile)
}

async fn preflight_candidate(
    state: &AppState,
    model: &std::path::Path,
    config: &RunConfig,
    profile_fingerprint: String,
    cache_key: &str,
) -> ApiResult<retrograd_core::PreflightReport> {
    if let Some(report) = state.preflights.read().recover().get(cache_key).cloned() {
        return Ok(report);
    }
    let permit = state
        .device
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::new(ProblemKind::DeviceBusy, "the device queue is closed"))?;
    let probe = state.probe.clone();
    let model = model.to_path_buf();
    let config = config.clone();
    let report = tokio::task::spawn_blocking(move || {
        let outcome = probe.preflight_config(&model, &config, profile_fingerprint);
        drop(permit);
        outcome
    })
    .await
    .map_err(|error| ApiError::internal(format!("preflight task failed: {error}")))??;
    report
        .validate()
        .map_err(|error| ApiError::internal(format!("invalid preflight report: {error}")))?;
    remember(&state.preflights, cache_key.to_string(), report.clone());
    Ok(report)
}

/// A `DataSpec` after resolution: the file to read, and the store's card for it
/// when the source was a dataset id. The card is what carries a tokenized
/// measurement, so keeping it is what lets [`dataset_stats`] skip the estimate.
struct DataSource {
    path: PathBuf,
    stored: Option<crate::datasets::DatasetMeta>,
}

/// Resolves a `DataSpec` to a real, on-disk path: a stored dataset's own file,
/// or the client's literal path when the operator allows one.
/// `Recipe::validate` already guaranteed exactly one of `path`/`dataset` is
/// set.
///
/// A dataset id has no path of its own to preserve, so `spec` is rewritten in
/// place to the file it actually names, with the format the store recorded -
/// the resolver, and the engine that eventually opens the file, never have to
/// know what a dataset id is. A literal `path` is left exactly as the client
/// sent it: the returned path is only `canonicalize()`d for the roots check
/// and for reading the file, and substituting that back would change the
/// effective configuration for a case that needed no change.
fn resolve_data_source(
    state: &AppState,
    spec: &mut DataSpec,
    pointer: &str,
) -> ApiResult<DataSource> {
    if let Some(id) = &spec.dataset {
        let meta = state.datasets.get(id).ok_or_else(|| {
            ApiError::not_found(format!("no dataset with id {id}"))
                .with_field(
                    format!("{pointer}/dataset"),
                    ErrorCode::NotFound,
                    "not found",
                )
                .with_hint("upload it first with POST /v1/datasets")
        })?;
        let path = state.datasets.data_path(&meta);
        spec.path = Some(path.clone());
        spec.dataset = None;
        spec.format = Some(meta.format.clone());
        return Ok(DataSource {
            path,
            stored: Some(meta),
        });
    }
    // `Recipe::validate` refuses a `DataSpec` with neither field set, so this
    // is reached only when `dataset` was absent and `path` therefore present.
    let path = spec
        .path
        .as_ref()
        .expect("Recipe::validate guarantees exactly one of path or dataset");
    if !state.config.allow_local_paths() {
        return Err(ApiError::new(
            ProblemKind::ForbiddenPath,
            "local dataset paths are disabled on this server; upload it with POST /v1/datasets",
        )
        .with_field(
            format!("{pointer}/path"),
            ErrorCode::ForbiddenPath,
            "local paths are disabled on this server",
        )
        .with_hint("upload it with POST /v1/datasets"));
    }
    let path = state.resolve_path(&path.to_string_lossy(), &format!("{pointer}/path"))?;
    Ok(DataSource { path, stored: None })
}

/// The length distribution the resolution runs on: the real one when this
/// dataset has already been tokenized against this model, the character
/// heuristic otherwise.
///
/// A warm cache costs nothing to read, so `/v1/plan` stays as fast and as free
/// of device side effects as it has to be while dropping the estimate's
/// deliberate over-count - which is what suppresses the "lengths are estimated"
/// warning and tightens `n_ctx`. Only a stored dataset can have a cache: a
/// literal path has no card to hang one on.
fn dataset_stats(
    source: &DataSource,
    path: &std::path::Path,
    format: DataFormat,
    tokenizer_key: Option<&str>,
) -> ApiResult<DatasetStats> {
    if let Some(measured) = source
        .stored
        .as_ref()
        .and_then(|meta| tokenizer_key.and_then(|key| meta.tokenized_for(key)))
        .and_then(|tokenized| tokenized.dataset_stats())
    {
        return Ok(measured);
    }
    DatasetStats::estimate_from_file(path, format).map_err(ApiError::from)
}

fn data_format(requested: Option<&str>, path: &std::path::Path) -> ApiResult<DataFormat> {
    match requested {
        None | Some("auto") => DataFormat::infer(path).map_err(ApiError::from),
        Some("text") | Some("txt") => Ok(DataFormat::Text),
        Some("jsonl") | Some("chat") | Some("chat-jsonl") => Ok(DataFormat::ChatJsonl),
        Some(other) => Err(ApiError::invalid(format!(
            "unknown data format '{other}'; use auto, text or jsonl"
        ))),
    }
}

/// Maps a resolver failure onto the problem type the contract assigns it.
fn problem(error: ResolveError) -> ApiError {
    match error {
        ResolveError::Invalid { message, path } => {
            let error = ApiError::invalid(message);
            match path {
                Some(path) => {
                    error.with_field(pointer(&path), ErrorCode::InvalidValue, "not accepted")
                }
                None => error,
            }
        }
        ResolveError::NeedsOptIn { allow, message } => ApiError::invalid(format!(
            "{message}; add \"{}\" to the recipe's `allow` list to accept it",
            allow.id()
        ))
        .with_field(
            "/recipe/allow",
            ErrorCode::NeedsOptIn,
            format!("missing '{}'", allow.id()),
        ),
        ResolveError::OverrideConflict { path, message } => {
            // 409, not 422: the request is well-formed, it is the combination
            // that cannot hold. Never a client parameter quietly moved.
            ApiError::new(ProblemKind::Conflict, message).with_field(
                format!("/params{}", pointer(&path)),
                ErrorCode::OverrideConflict,
                "cannot be honoured",
            )
        }
        ResolveError::InsufficientMemory(report) => {
            let mut problem = insufficient(
                &report.estimate,
                &report.budgets,
                report.overflow.total(),
                &report.applied,
                &report.unlocks,
            );
            problem.detail = report.detail();
            problem
        }
    }
}

/// Builds the `insufficient-memory` problem, its structured payload in `meta`
/// rather than smuggled through `errors[]`.
fn insufficient(
    estimate: &MemoryEstimate,
    budgets: &retrograd_plan::Budgets,
    overflow: u64,
    applied: &[retrograd_plan::AppliedLever],
    unlocks: &[retrograd_plan::Allow],
) -> ApiError {
    let dominant_posts: Vec<_> = estimate
        .dominant_posts()
        .into_iter()
        .take(3)
        .map(|(name, bytes)| serde_json::json!({"post": name, "bytes": bytes}))
        .collect();
    ApiError::new(
        ProblemKind::InsufficientMemory,
        format!("{overflow} bytes over the effective budget"),
    )
    .with_meta(serde_json::json!({
        "overflow_bytes": overflow,
        "vram_budget_bytes": budgets.vram.effective_bytes,
        "ram_budget_bytes": budgets.ram.effective_bytes,
        "dominant_posts": dominant_posts,
        "levers_applied": applied.iter().map(|lever| lever.id).collect::<Vec<_>>(),
        "unlocks": unlocks.iter().map(|allow| allow.id()).collect::<Vec<_>>(),
    }))
}

/// Dotted resolver path to the JSON Pointer the client's body uses.
fn pointer(path: &str) -> String {
    let mut out = String::new();
    for segment in path.split('.') {
        out.push('/');
        out.push_str(&segment.replace('~', "~0").replace('/', "~1"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dotted_path_becomes_a_json_pointer() {
        assert_eq!(pointer("training.ctx"), "/training/ctx");
        assert_eq!(pointer("lora.rank"), "/lora/rank");
        assert_eq!(pointer("a/b.c~d"), "/a~1b/c~0d");
    }

    #[test]
    fn a_redacted_config_never_carries_the_declared_command() {
        let document = retrograd_config::parse_toml(
            r#"
[run]
algorithm = "grpo"
[model]
path = "m.gguf"
[output]
path = "a.gguf"
[lora]
[grpo]
prompts = "p.jsonl"
reward_command = ["python", "secret.py"]
updates = 1
prompts_per_update = 1
group_size = 2
grpo_epochs = 1
clip_range_low = 0.2
clip_range_high = 0.28
kl_coefficient = 0.0
[grpo.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 4
seed = 1
"#,
            "test",
        )
        .expect("parses");
        let redacted = redact(document, Some("sql-exec"));
        let rendered = serde_json::to_string(&redacted).unwrap();
        assert!(!rendered.contains("secret.py"), "{rendered}");
        assert!(rendered.contains("<reward:sql-exec>"), "{rendered}");
    }

    #[test]
    fn the_data_format_is_inferred_or_named() {
        let path = std::path::Path::new("data.jsonl");
        assert_eq!(data_format(None, path).unwrap(), DataFormat::ChatJsonl);
        assert_eq!(
            data_format(Some("text"), path).unwrap(),
            DataFormat::Text,
            "an explicit format wins over the extension"
        );
        assert!(data_format(Some("parquet"), path).is_err());
    }
}
