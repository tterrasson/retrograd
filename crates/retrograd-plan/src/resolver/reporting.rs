//! Reporting

use super::*;

// The rollout count is a `usize` product of two numbers the config file
// supplies, carried as `u64` into the memory budget. Truncating it to `u32` is
// the cast shape the numeric-conversion convention calls out: `2^32 + 8`
// rollouts became `8`. Capping it at `u32::MAX` would still underprice every
// larger workload, so the conversion preserves the full count and saturates
// only at `u64::MAX`.
pub(super) fn workload_kind_of(config: &RunConfig) -> WorkloadKind {
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

pub(super) fn completion_bound(config: &RunConfig) -> u32 {
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

pub(super) fn fanout_state(value: SharedPrefixFanout) -> String {
    match value {
        SharedPrefixFanout::Auto => "auto".to_string(),
        SharedPrefixFanout::Off => "off".to_string(),
        SharedPrefixFanout::Max => "max".to_string(),
        SharedPrefixFanout::Exact(value) => value.to_string(),
    }
}

pub(super) fn record_candidate_transition(
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

pub(super) fn workload_of(input: &ResolveInput<'_>, config: &RunConfig) -> Workload {
    Workload {
        kind: workload_kind_of(config),
        examples: input.data.examples,
        co_resident_bytes: co_resident_bytes(config, input.co_resident()),
    }
}

/// The models a run holds beside the one being sized, as geometry the caller
/// could read. A declared model whose geometry is `None` is a cost missing
/// from the budget; `collect_warnings` says so.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoResident<'a> {
    /// Geometry of the distillation teacher, when the document names one.
    pub teacher: Option<&'a ModelInfo>,
    /// Geometry of the fixed-reference anchor, when the document names one.
    pub reference: Option<&'a ModelInfo>,
}

/// Device bytes those models hold beside the one being sized. A model that is
/// declared but arrives without geometry is a *known* under-estimate, flagged
/// by `collect_warnings`.
pub(super) fn co_resident_bytes(config: &RunConfig, models: CoResident<'_>) -> u64 {
    let teacher = match (&config.algorithm, models.teacher) {
        // The offline mode never opens the teacher - the sidecar is what it left
        // behind - so charging its weights to the device would refuse
        // configurations that run.
        (retrograd_config::Algorithm::Distill(distill), Some(teacher))
            if distill.mode.is_rollout() =>
        {
            crate::cost::co_resident_model_bytes(teacher, &config.training)
        }
        _ => 0,
    };
    // The anchor is resident for the whole run regardless of algorithm, and at
    // its own width when the document gave it one.
    let reference = match (&config.reference, models.reference) {
        (Some(declared), Some(geometry)) => crate::cost::co_resident_model_bytes_at(
            geometry,
            &config.training,
            declared.n_ctx.unwrap_or(config.training.n_ctx),
        ),
        _ => 0,
    };
    teacher.saturating_add(reference)
}

/// `(iterations, total_steps)`.
pub(super) fn step_counts(input: &ResolveInput<'_>, config: &RunConfig) -> (u64, u64) {
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
pub(super) fn judge_calls(input: &ResolveInput<'_>, config: &RunConfig) -> u64 {
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

pub(super) fn collect_warnings(
    config: &RunConfig,
    models: CoResident<'_>,
    backend: Backend,
    warnings: &mut Vec<PlanWarning>,
) {
    let teacher = models.teacher;
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
    // Same gap, other model: an anchor named without its geometry keeps the
    // estimate one model short.
    if config.reference.is_some() && models.reference.is_none() {
        warn(
            warnings,
            "reference_absent_from_the_memory_budget",
            Some("reference.model"),
            "the memory estimate does not include the fixed-reference model, which stays \
             resident beside the trained model for the whole run: supply the anchor's \
             geometry to size it, or read the device figure as a lower bound",
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
    if config
        .lora
        .as_ref()
        .is_some_and(|lora| matches!(lora.config.targets, TargetSet::Auto))
    {
        warn(
            warnings,
            "lora_targets_resolved_by_runtime",
            Some("lora.targets"),
            "the LoRA target set is resolved per architecture by the runtime; the estimate \
             assumes every attention and feed-forward projection",
        );
    }
}

pub(super) fn param_u32(params: &Value, section: &str, key: &str) -> Option<u32> {
    param_u64(params, section, key).and_then(|value| u32::try_from(value).ok())
}

pub(super) fn param_u64(params: &Value, section: &str, key: &str) -> Option<u64> {
    params.get(section)?.get(key)?.as_u64()
}

pub(super) fn param_bool(params: &Value, section: &str, key: &str) -> Option<bool> {
    params.get(section)?.get(key)?.as_bool()
}

pub(super) fn param_str<'a>(params: &'a Value, section: &str, key: &str) -> Option<&'a str> {
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
