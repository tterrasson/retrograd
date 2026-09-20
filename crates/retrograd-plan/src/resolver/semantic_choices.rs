//! Semantic choices

use super::*;

pub(super) fn choose_context(
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
pub(super) struct RolloutShape {
    /// Zero for anything but GRPO - the only algorithm whose configuration has
    /// a `training.generation_concurrency` to size.
    pub(super) group_size: u32,
}

pub(super) fn draft_document(
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
            output: Some(OutputToml {
                path: recipe
                    .output
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("adapter.gguf")),
                // The resolver only drafts LoRA runs, so the default kind is
                // the one it produces; naming it here would pin a choice the
                // recipe never made.
                kind: None,
            }),
            lora: Some(LoraToml {
                rank: Some(rank),
                alpha: Some(alpha),
                seed: Some(recipe.seed.unwrap_or(42)),
                targets,
                init_adapter: None,
                dtype: None,
            }),
            training,
            // The resolver only ever drafts LoRA runs: its cost model sizes an
            // adapter's parameters, gradients and optimizer state, and a base
            // selection would change all three, which the planner cannot yet
            // price from a resolved trainable set.
            trainable: None,
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
            // The drafts are LoRA runs, whose anchor is their own frozen base
            // weights; a separate one is declared in the document, if at all.
            reference: None,
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
pub(super) fn draft_lora(
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
pub(super) struct LoraDraft {
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
pub(super) fn draft_rollout_width(
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
pub(super) struct RolloutWidth {
    group_size: usize,
    prompts_per_update: usize,
    updates: u32,
    inner_epochs: u32,
}

pub(super) fn budgeted_epochs(
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

pub(super) fn budgeted_updates(
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
