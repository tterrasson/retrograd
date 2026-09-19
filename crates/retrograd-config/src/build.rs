//! Document-wide assembly: from a parsed [`ConfigDocument`] to a [`RunConfig`].
//!
//! Each section builds itself in its own module. What is here is what no single
//! section can decide: the shared `[model]`, `[lora]` and `[training]` blocks,
//! the rules that cross two sections, and the rollout geometry the algorithm
//! pins onto the optimizer window.

use std::fs;
use std::path::Path;

use retrograd_core::{
    CheckpointDtype, Device, Error, LayerRange, LoraConfig, LoraDtype, OptimizerKind, Result,
    SharedPrefixFanout, TensorDtype, TrainConfig, TrainablePolicy, TrainableRunConfig,
    TrainableSelector,
};

use crate::common::{
    parse_scheduler, parse_string_enum, require_non_negative_f32, require_non_negative_f64,
    require_nonzero, require_positive_f32, required, resolve,
};
use crate::distill::build_distill;
use crate::document::{
    CheckpointToml, ConfigDocument, EvaluationToml, LoraToml, ModelOverride, ObserveToml,
    OutputToml, SharedPrefixFanoutToml, TrainableToml, TrainingToml,
};
use crate::grpo::build_grpo;
use crate::ppo::build_ppo;
use crate::sft::build_sft;
use crate::{
    Algorithm, CheckpointConfig, CheckpointMode, DEFAULT_SEED, DEFAULT_TARGETS, EvaluationConfig,
    LoraRunConfig, MetricsConfig, ObserveConfig, OutputConfig, OutputKind, RunConfig, agent,
    parse_targets,
};

/// Reads, parses and builds the TOML file at `path` into a [`RunConfig`].
pub fn load(path: impl AsRef<Path>) -> Result<RunConfig> {
    load_with(path, ModelOverride::default())
}

/// [`load`] with the `[model]` fields a frontend supplies itself.
pub fn load_with(path: impl AsRef<Path>, overrides: ModelOverride) -> Result<RunConfig> {
    let path = path.as_ref();
    let span = tracing::info_span!(
        target: "retrograd::config",
        "config",
        path = %path.display()
    );
    let _entered = span.enter();
    let source = fs::read_to_string(path)?;
    let document = parse_toml(&source, &path.display().to_string())?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    build_with(document, root, overrides)
}

/// Parses a TOML document without reading a file, for the callers that already
/// hold the text (a request body, a test fixture). `origin` only names the
/// source in the error message.
pub fn parse_toml(source: &str, origin: &str) -> Result<ConfigDocument> {
    toml::from_str(source)
        .map_err(|error| Error::config(format!("{origin}: invalid TOML: {error}")))
}
/// Validates and resolves a parsed [`ConfigDocument`] into a [`RunConfig`],
/// relative paths joined against `root`. The only way from a document to a
/// `RunConfig`; [`load`] calls it after parsing the file.
pub fn build(file: ConfigDocument, root: &Path) -> Result<RunConfig> {
    build_with(file, root, ModelOverride::default())
}

/// [`build`] with the `[model]` fields a frontend supplies itself, taking
/// precedence over whatever the document wrote.
///
/// The order is the one the rules need: the shared blocks first, then the
/// algorithm, then the two things only the pair can decide - the optimizer
/// window and the rollout geometry - and last the sections that validate
/// against the algorithm they were chosen beside.
pub fn build_with(
    file: ConfigDocument,
    root: &Path,
    overrides: ModelOverride,
) -> Result<RunConfig> {
    let model = match (overrides.path, file.model.path.clone()) {
        (Some(path), _) => path,
        (None, Some(path)) => resolve(root, path),
        (None, None) => {
            return Err(Error::config(
                "[model].path is missing and no model was given on the command line",
            ));
        }
    };
    let mut training = build_training(
        &file.training,
        file.model.device.as_deref(),
        overrides.device,
        file.run.verbose,
    )?;
    training.trainable = build_trainable(&file.training, file.trainable.as_ref())?;
    let policy = training.trainable.policy;
    let lora = build_lora(policy, file.lora.as_ref())?;
    let output = build_output(policy, file.output.as_ref(), root)?;

    let algorithm_name = file.run.algorithm.to_ascii_lowercase();
    only_the_selected_section(&file, &algorithm_name)?;
    let ConfigDocument {
        sft,
        ppo,
        grpo,
        distill,
        agent: agent_section,
        evaluation,
        checkpoint,
        observe,
        lora: lora_toml,
        metrics,
        training: training_toml,
        ..
    } = file;
    // `trainable` was consumed above; the destructuring drops the DTO.

    let algorithm = match algorithm_name.as_str() {
        "sft" => {
            let value = required(sft, "[sft] is required when run.algorithm = 'sft'")?;
            Algorithm::Sft(build_sft(
                value,
                root,
                &mut training,
                lora.as_ref().map_or(DEFAULT_SEED, |config| config.seed),
            )?)
        }
        "ppo" => {
            let value = required(ppo, "[ppo] is required when run.algorithm = 'ppo'")?;
            Algorithm::Ppo(build_ppo(value, root)?)
        }
        "grpo" => {
            let value = required(grpo, "[grpo] is required when run.algorithm = 'grpo'")?;
            Algorithm::Grpo(build_grpo(value, root)?)
        }
        "distill" => {
            let value = required(
                distill,
                "[distill] is required when run.algorithm = 'distill'",
            )?;
            Algorithm::Distill(build_distill(value, root)?)
        }
        "agent_grpo" => {
            let value = required(
                agent_section,
                "[agent] is required when run.algorithm = 'agent_grpo'",
            )?;
            Algorithm::AgentGrpo(Box::new(agent::build_agent(value, root, resolve)?))
        }
        _ => {
            return Err(Error::config(
                "run.algorithm must be one of sft, ppo, grpo, distill, or agent_grpo",
            ));
        }
    };

    derive_optimizer_window(
        &mut training,
        training_toml.gradient_accumulation,
        &algorithm,
    )?;
    pin_rollout_geometry(
        &mut training,
        &algorithm,
        training_toml.generation_concurrency,
    )?;

    let evaluation = evaluation
        .map(|value| build_evaluation(value, root))
        .transpose()?;
    let checkpoint = checkpoint
        .map(|value| build_checkpoint(value, root))
        .transpose()?;
    let observe = observe
        .map(|value| build_observe(value, &algorithm, root))
        .transpose()?;
    check_across_sections(
        &algorithm,
        evaluation.as_ref(),
        checkpoint.as_ref(),
        lora_toml.as_ref(),
        lora.as_ref(),
        &training.trainable,
    )?;

    Ok(RunConfig {
        algorithm,
        model,
        lora: lora.map(|config| LoraRunConfig {
            config,
            init_adapter: lora_toml
                .as_ref()
                .and_then(|section| section.init_adapter.clone())
                .map(|path| resolve(root, path)),
        }),
        output,
        training,
        metrics: MetricsConfig {
            tensorboard_dir: metrics.tensorboard_dir.map(|path| resolve(root, path)),
            wandb_export_dir: metrics.wandb_export_dir.map(|path| resolve(root, path)),
        },
        evaluation,
        checkpoint,
        observe,
    })
}

/// The shared `[training]` block, plus the `[model]` device and the verbosity
/// `[run]` carries. `n_batch` stays at its default: it is derived from
/// `gradient_accumulation` and the algorithm, neither of which is known yet.
fn build_training(
    file: &TrainingToml,
    device: Option<&str>,
    override_device: Option<Device>,
    verbose: bool,
) -> Result<TrainConfig> {
    let mut training = TrainConfig::default();
    if let Some(value) = file.ctx {
        training.n_ctx = value;
    }
    if let Some(value) = file.micro_batch {
        training.n_ubatch = value;
    }
    if let Some(value) = &file.shared_prefix_fanout {
        training.shared_prefix_fanout = match value {
            SharedPrefixFanoutToml::Name(name) => parse_string_enum!(
                name,
                "training.shared_prefix_fanout must be 'auto', 'off', 'max', or an integer >= 2",
                "auto" => SharedPrefixFanout::Auto,
                "off" => SharedPrefixFanout::Off,
                "max" => SharedPrefixFanout::Max,
            )?,
            SharedPrefixFanoutToml::Exact(value) if *value >= 2 => {
                SharedPrefixFanout::Exact(*value)
            }
            SharedPrefixFanoutToml::Exact(_) => {
                return Err(Error::config(
                    "training.shared_prefix_fanout integer must be at least 2; use 'off' to disable it",
                ));
            }
        };
    }
    // `n_batch` - the token window of one optimizer step - is derived from
    // `micro_batch * gradient_accumulation`, and a rollout algorithm pins it to
    // the whole context. The algorithm is only known further down, so the
    // derivation happens there.
    if let Some(value) = file.threads {
        require_nonzero(value, "training.threads must be greater than zero")?;
        training.threads = value;
    }
    if let Some(value) = file.epochs {
        training.epochs = value;
    }
    if let Some(value) = file.lr {
        training.learning_rate = value;
    }
    if let Some(value) = file.weight_decay {
        training.weight_decay = value;
    }
    if let Some(value) = file.max_grad_norm {
        training.max_grad_norm = value;
    }
    if let Some(value) = file.warmup_steps {
        training.warmup_steps = value;
    }
    if let Some(value) = &file.lr_scheduler {
        training.lr_scheduler = parse_scheduler(value)?;
    }
    if let Some(value) = file.fast_sampling_context {
        training.fast_generation_context = value;
    }
    if let Some(value) = file.kv_dtype {
        training.kv_dtype = value;
    }
    if let Some(value) = file.chunked_cross_entropy {
        training.chunked_cross_entropy = value;
    }
    if let Some(value) = file.chunked_ce_tiles {
        require_nonzero(value, "training.chunked_ce_tiles must be greater than zero")?;
        training.chunked_ce_tiles = value;
    }
    if let Some(value) = file.chunked_ce_seq_chunk {
        training.chunked_ce_seq_chunk = value;
    }
    if let Some(value) = file.gradient_checkpointing {
        training.gradient_checkpointing = value;
    }
    if let Some(value) = file.checkpoint_every_n_layers {
        require_nonzero(
            value,
            "training.checkpoint_every_n_layers must be greater than zero",
        )?;
        training.checkpoint_every_n_layers = value;
    }
    if let Some(value) = file.checkpoint_dtype {
        // The runtime ignores this without checkpointing, and "ignored" is
        // indistinguishable from "applied" in every artifact the run produces,
        // a 16-bit checkpoint is a request to give up ~1e-3 of gradient
        // fidelity, so accepting it silently in a run that has no checkpoints
        // is the one answer that cannot be checked afterwards.
        if value != CheckpointDtype::F32 && !training.gradient_checkpointing {
            return Err(Error::config(
                "training.checkpoint_dtype only applies to retained activation checkpoints; \
                 set training.gradient_checkpointing = true or leave it at 'f32'",
            ));
        }
        training.checkpoint_dtype = value;
    }
    if let Some(value) = file.require_gpu_resident {
        training.require_gpu_resident = value;
    }
    if let Some(value) = file.max_gpu_duty_cycle {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(Error::config(
                "training.max_gpu_duty_cycle must be finite and in (0, 1]; \
                 use the run-control pause to stop a live run",
            ));
        }
        // An explicit `1.0` normalizes to the same `None` an omitted key does.
        // The two say the same thing - no limit - and collapsing them here is
        // what keeps a single disabled path in the runtime instead of one that
        // installs a limiter and then never sleeps.
        training.max_gpu_duty_cycle = (value < 1.0).then_some(value);
    }
    if let Some(value) = file.generation_batch {
        require_nonzero(value, "training.generation_batch must be greater than zero")?;
        training.generation_batch = value;
    }
    if let Some(value) = device {
        training.device = value.parse()?;
    }
    if let Some(device) = override_device {
        training.device = device;
    }
    training.verbose = verbose;
    require_nonzero(training.epochs, "training.epochs must be greater than zero")?;
    require_nonzero(training.n_ctx, "training.ctx must be greater than zero")?;
    require_nonzero(
        training.n_ubatch,
        "training.micro_batch must be greater than zero",
    )?;
    require_nonzero(
        file.gradient_accumulation.unwrap_or(1),
        "training.gradient_accumulation must be greater than zero",
    )?;
    require_positive_f32(training.learning_rate, "training.lr")?;
    require_non_negative_f32(training.weight_decay, "training.weight_decay")?;
    require_positive_f32(training.max_grad_norm, "training.max_grad_norm")?;
    Ok(training)
}

/// `[training].trainable`, `[training].optimizer` and `[trainable]`, resolved
/// together because only the pair decides whether a section is meaningful.
///
/// The rule this enforces first, and the reason the function exists: a selector
/// a policy ignores is a selector the user believes is in effect. `lora` with a
/// `[trainable]` section is refused rather than silently trained as LoRA.
///
/// Modes and optimizers this build cannot honour are refused here too.
/// Parsing a name and then running something else
/// would produce a trajectory nobody asked for and a checkpoint that records
/// the wrong optimizer.
fn build_trainable(
    training: &TrainingToml,
    selector: Option<&TrainableToml>,
) -> Result<TrainableRunConfig> {
    let policy = match &training.trainable {
        Some(value) => TrainablePolicy::parse(value)?,
        None => TrainablePolicy::default(),
    };
    let optimizer = match &training.optimizer {
        Some(value) => OptimizerKind::parse(value)?,
        None => OptimizerKind::default(),
    };
    if !optimizer.is_implemented() {
        return Err(Error::config(format!(
            "training.optimizer = '{optimizer}' is not available in this build: \
             only adamw and sgd have an update step here. \
             Accepting the name and running AdamW would write a checkpoint that \
             records an optimizer the run never used"
        )));
    }

    if policy == TrainablePolicy::Lora {
        if selector.is_some() {
            return Err(Error::config(
                "[trainable] selects base tensors, but training.trainable = 'lora' \
                 trains none: remove the section, or set training.trainable to \
                 'partial' or 'hybrid'",
            ));
        }
        return Ok(TrainableRunConfig {
            policy,
            selector: TrainableSelector::default(),
            optimizer,
        });
    }

    Ok(TrainableRunConfig {
        policy,
        selector: build_trainable_selector(policy, selector)?,
        optimizer,
    })
}

/// The `[trainable]` section's own consistency, given the policy that reads it.
fn build_trainable_selector(
    policy: TrainablePolicy,
    section: Option<&TrainableToml>,
) -> Result<TrainableSelector> {
    if policy == TrainablePolicy::Full {
        // `full` derives every eligible family from the model itself, so a
        // narrowing selector beside it is a contradiction, not a refinement.
        if section.is_some() {
            return Err(Error::config(
                "training.trainable = 'full' trains every supported eligible tensor \
                 and takes no [trainable] selectors: use 'partial' to narrow it",
            ));
        }
        return Ok(TrainableSelector::default());
    }

    let section = section.ok_or_else(|| {
        Error::config(format!(
            "training.trainable = '{policy}' requires a [trainable] section \
             naming what to train"
        ))
    })?;
    let selector = TrainableSelector {
        layers: match &section.layers {
            Some(value) => LayerRange::parse(value)?,
            None => LayerRange::All,
        },
        modules: section.modules.clone(),
        norms: section.norms.unwrap_or(false),
        biases: section.biases.unwrap_or(false),
        output_head: section.output_head.unwrap_or(false),
    };
    if selector.is_empty() {
        return Err(Error::config(format!(
            "[trainable] selects nothing for training.trainable = '{policy}': \
             set at least one of modules, norms, biases or output_head"
        )));
    }
    if policy == TrainablePolicy::Hybrid && (!selector.modules.is_empty() || selector.output_head) {
        // Hybrid's first delivery is the adapter plus base norms/biases. A
        // module or a head beside an adapter means two parameterizations of one
        // weight, whose composite export has no parity coverage yet.
        return Err(Error::config(
            "training.trainable = 'hybrid' initially permits only base norms and \
             biases beside the adapter: [trainable].modules and .output_head are \
             not supported with it",
        ));
    }
    Ok(selector)
}

/// The shared `[lora]` block, when the policy has an adapter at all.
///
/// Required by `lora` and `hybrid`, refused by `full` and `partial`: a section
/// describing an adapter the run never creates is a section whose rank, alpha
/// and targets have no effect, which is the same failure as a `[trainable]`
/// selector beside `lora`.
fn build_lora(policy: TrainablePolicy, file: Option<&LoraToml>) -> Result<Option<LoraConfig>> {
    let trains_adapter = policy != TrainablePolicy::Full && policy != TrainablePolicy::Partial;
    match (trains_adapter, file) {
        (true, Some(file)) => build_lora_config(file).map(Some),
        (true, None) => Err(Error::config(format!(
            "training.trainable = '{policy}' trains a LoRA adapter and requires a \
             [lora] section"
        ))),
        (false, Some(_)) => Err(Error::config(format!(
            "training.trainable = '{policy}' trains base tensors and no adapter: \
             remove [lora], or use 'hybrid' to train both"
        ))),
        (false, None) => Ok(None),
    }
}

/// Where the run's result goes, and what kind of result it is.
fn build_output(
    policy: TrainablePolicy,
    section: Option<&OutputToml>,
    root: &Path,
) -> Result<OutputConfig> {
    let Some(path) = section.map(|value| value.path.clone()) else {
        return Err(Error::config(
            "[output].path is missing: a run must say where to write its result",
        ));
    };

    // The default is what the policy produces, so an ordinary document never
    // has to name a kind - and a document that names the wrong one is told so
    // rather than silently given the default.
    let default = if policy.trains_base_weights() {
        OutputKind::Trainable
    } else {
        OutputKind::Adapter
    };
    let kind = match section.and_then(|value| value.kind.as_deref()) {
        Some(value) => OutputKind::parse(value)?,
        None => default,
    };
    match (kind, policy) {
        (OutputKind::Adapter, TrainablePolicy::Full | TrainablePolicy::Partial) => {
            return Err(Error::config(format!(
                "output.kind = 'adapter' with training.trainable = '{policy}': this run \
                 trains base tensors and creates no adapter, so an adapter export would \
                 be an empty file"
            )));
        }
        (OutputKind::Adapter, TrainablePolicy::Hybrid) => {
            // The adapter alone is half of what a hybrid run produced, and the
            // half a loader cannot tell is incomplete.
            return Err(Error::config(
                "output.kind = 'adapter' with training.trainable = 'hybrid' would drop the \
                 trained base tensors: use the composite 'trainable' bundle",
            ));
        }
        (OutputKind::Trainable, TrainablePolicy::Lora) => {
            return Err(Error::config(
                "output.kind = 'trainable' with training.trainable = 'lora': a LoRA run \
                 trains no base tensor, and its portable result is the adapter",
            ));
        }
        (OutputKind::Model, TrainablePolicy::Lora) => {
            return Err(Error::config(
                "output.kind = 'model' with training.trainable = 'lora': the run's result \
                 lives in the adapter, and folding it into the weights is a merge with no \
                 parity coverage here. Use 'adapter'",
            ));
        }
        (OutputKind::Model, TrainablePolicy::Hybrid) => {
            // Same merge problem, quieter: the base tensors would be written
            // and the adapter dropped.
            return Err(Error::config(
                "output.kind = 'model' with training.trainable = 'hybrid' would write the \
                 trained base tensors and drop the adapter: merging one into the weights \
                 has no parity coverage here. Use the composite 'trainable' bundle",
            ));
        }
        _ => {}
    }
    Ok(OutputConfig {
        path: resolve(root, path),
        kind,
    })
}

/// The `[lora]` block itself. `init_adapter` excludes the creation keys: an
/// adapter file already carries rank, alpha, seed, dtype and targets.
fn build_lora_config(file: &LoraToml) -> Result<LoraConfig> {
    if file.init_adapter.is_some()
        && (file.rank.is_some()
            || file.alpha.is_some()
            || file.seed.is_some()
            || file.dtype.is_some()
            || !file.targets.is_empty())
    {
        return Err(Error::config(
            "lora.init_adapter cannot be combined with lora.rank, lora.alpha, \
             lora.seed, lora.dtype, or lora.targets: they come from the adapter file",
        ));
    }
    let mut lora = LoraConfig::auto(file.rank.unwrap_or(8), file.alpha.unwrap_or(16.0));
    lora.seed = file.seed.unwrap_or(DEFAULT_SEED);
    lora.targets = if file.targets.is_empty() {
        parse_targets(&DEFAULT_TARGETS.map(String::from))?
    } else {
        parse_targets(&file.targets)?
    };
    lora.dtype = file.dtype.unwrap_or_default();
    require_nonzero(lora.rank, "lora.rank must be greater than zero")?;
    require_positive_f32(lora.alpha, "lora.alpha")?;
    Ok(lora)
}

/// One run trains one way, so a section other than the selected algorithm's is
/// refused rather than ignored.
fn only_the_selected_section(file: &ConfigDocument, algorithm_name: &str) -> Result<()> {
    // One run trains one way. Every algorithm section other than the selected
    // one is a leftover from an edit, and silently ignoring it is how a config
    // ends up describing a run nobody is having.
    for (name, present) in [
        ("sft", file.sft.is_some()),
        ("ppo", file.ppo.is_some()),
        ("grpo", file.grpo.is_some()),
        ("distill", file.distill.is_some()),
        ("agent", file.agent.is_some()),
    ] {
        let selected = match name {
            "agent" => algorithm_name == "agent_grpo",
            other => algorithm_name == other,
        };
        if present && !selected {
            return Err(Error::config(format!(
                "only the section for the selected algorithm may be present: \
                 run.algorithm = '{algorithm_name}' but [{name}] is set"
            )));
        }
    }
    Ok(())
}

/// `n_batch` is not spelled in the file: it is `micro_batch` times
/// `gradient_accumulation`, and a rollout algorithm pins it to the whole
/// context. Validates the geometry the result has to satisfy.
fn derive_optimizer_window(
    training: &mut TrainConfig,
    gradient_accumulation: Option<u32>,
    algorithm: &Algorithm,
) -> Result<()> {
    let rollout = match &algorithm {
        Algorithm::Ppo(_) | Algorithm::Grpo(_) | Algorithm::AgentGrpo(_) => true,
        Algorithm::Distill(distill) => distill.mode.is_rollout(),
        Algorithm::Sft(_) => false,
    };
    let accumulation = match (gradient_accumulation, rollout) {
        (Some(value), _) => value,
        (None, true) => training.n_ctx.div_ceil(training.n_ubatch),
        (None, false) => 1,
    };
    training.n_batch = training
        .n_ubatch
        .checked_mul(accumulation)
        .ok_or_else(|| Error::overflow("training.micro_batch * gradient_accumulation overflows"))?;
    // Both rules live on `TrainConfig` so every frontend that can build a
    // rollout run applies the same ones - this loader, the planner, and
    // `retrograd-agent`'s own TOML.
    if rollout {
        training.validate_rollout_geometry()?;
    } else {
        training.validate_geometry()?;
    }
    // Not `grpo_geometry` any more: three algorithms generate, and what this
    // pins - `n_seq_max` and `generation_concurrency` - is the geometry of a
    // rollout rather than of a group baseline. Distillation reads
    // `samples_per_prompt` where GRPO reads `group_size`; they are the same
    // number to the sampler, which branches a prompt that many ways.
    Ok(())
}

/// The sequence widths a generating algorithm pins onto the optimizer window.
fn pin_rollout_geometry(
    training: &mut TrainConfig,
    algorithm: &Algorithm,
    requested_generation_concurrency: Option<u32>,
) -> Result<()> {
    let rollout_geometry = match &algorithm {
        Algorithm::Grpo(grpo) => Some(("grpo", grpo.group_size, grpo.prompts_per_update)),
        Algorithm::Distill(distill) if distill.mode.is_rollout() => Some((
            "distill",
            distill.samples_per_prompt,
            distill.prompts_per_update,
        )),
        Algorithm::AgentGrpo(agent) => Some((
            "agent",
            agent.config.group_size,
            agent.config.scenarios_per_update,
        )),
        _ => None,
    };
    if let Some((section, group_size, prompts_per_update)) = rollout_geometry {
        if group_size > training.n_batch as usize {
            return Err(Error::config(format!(
                "{section}.group_size must not exceed the optimizer window \
                 (training.micro_batch * training.gradient_accumulation) for batched generation"
            )));
        }
        // Optimizer packing and rollout generation have independent sequence
        // widths. A logical GRPO group may be generated over several waves.
        //
        // The cast cannot lose bits: the guard above refuses
        // `group_size > training.n_batch as usize`, and `n_batch` is a `u32`
        // (the proof stays here rather than becoming a
        // `try_from`, whose error arm would be unreachable and whose placement
        // before the guard would refuse the value for the wrong reason).
        training.n_seq_max = group_size as u32;
        if let SharedPrefixFanout::Exact(fanout) = training.shared_prefix_fanout
            && fanout > training.n_seq_max
        {
            return Err(Error::config(format!(
                "training.shared_prefix_fanout ({fanout}) exceeds {section}.group_size ({group_size})"
            )));
        }
        let concurrent_sequences = prompts_per_update
            .checked_mul(group_size)
            .ok_or_else(|| Error::overflow("continuous generation sequence count overflows"))?;
        let default_concurrency =
            concurrent_sequences.min(training.n_batch as usize).min(256) as u32;
        training.generation_concurrency =
            requested_generation_concurrency.unwrap_or(default_concurrency);
        if training.generation_concurrency == 0 {
            return Err(Error::config(
                "training.generation_concurrency must be greater than zero",
            ));
        }
        if training.generation_concurrency > training.n_batch {
            return Err(Error::config(
                "training.generation_concurrency must not exceed the optimizer window \
                 (training.micro_batch * training.gradient_accumulation)",
            ));
        }
        if training.generation_concurrency > 256 {
            return Err(Error::config(
                "training.generation_concurrency must not exceed 256",
            ));
        }
        if training.generation_concurrency as usize > concurrent_sequences {
            return Err(Error::config(format!(
                "training.generation_concurrency must not exceed the {concurrent_sequences} \
                 {section} rollouts per update"
            )));
        }
    } else if requested_generation_concurrency.is_some() {
        return Err(Error::config(
            "training.generation_concurrency is only supported for a generating algorithm \
             (grpo, distill, agent_grpo)",
        ));
    }
    Ok(())
}

/// The rules no single section can check: they hold between the algorithm and
/// `[evaluation]`, `[checkpoint]` or `[lora]`.
fn check_across_sections(
    algorithm: &Algorithm,
    evaluation: Option<&EvaluationConfig>,
    checkpoint: Option<&CheckpointConfig>,
    lora: Option<&LoraToml>,
    lora_config: Option<&LoraConfig>,
    trainable: &TrainableRunConfig,
) -> Result<()> {
    // The fixed reference, and why base training cannot have one yet.
    //
    // Every consumer below scores against "the model with its adapter
    // disabled" and calls that the original policy. That identity holds
    // because a LoRA run leaves the base weights untouched - and a run that
    // updates them breaks it on the first step, silently: the KL is then taken
    // against a moving target, and its value keeps being a number.
    //
    // Checked on the configured coefficient and schedule rather than on the
    // current warmup value, because a run that starts at zero and warms up to a
    // penalty is a run with a reference.
    if trainable.policy.trains_base_weights()
        && let Some(consumer) = fixed_reference_consumer(algorithm)
    {
        return Err(Error::config(format!(
            "{consumer} with training.trainable = '{}': the penalty is taken against \
             the model with its adapter disabled, which is the original policy only \
             while the base weights are frozen. A run that trains them needs a \
             separate reference model, which this build does not have - set the \
             coefficient to zero, or train an adapter",
            trainable.policy
        )));
    }
    // put on it - a verify command, a test suite, a task's own grading. The
    // judge cannot stand in: every RULER strategy scores the members of a group
    // against each other, so its scores are renormalized at every update and a
    // mean over them is not comparable across the run. A configuration that
    // asks to be evaluated with nothing that grades is refused here rather than
    // at the first evaluation, an hour into the run.
    if let Algorithm::AgentGrpo(agent) = algorithm
        && evaluation.is_some()
        && agent.environment.is_none()
    {
        return Err(Error::config(
            "[evaluation] with run.algorithm = 'agent_grpo' needs [agent.environment]: an \
                 agentic evaluation measures the reward the environment puts on a trajectory, \
                 and a judge cannot stand in - its scores are relative inside a group and not \
                 comparable across updates",
        ));
    }
    if checkpoint
        .as_ref()
        .is_some_and(|value| value.mode.includes_best_eval())
        && evaluation.is_none()
    {
        return Err(Error::config(
            "checkpoint.mode includes best_eval but [evaluation] is missing",
        ));
    }
    // A resume owns the adapter it restores; combining it with a cold adapter
    // load would leave which weights actually train ambiguous.
    if checkpoint
        .as_ref()
        .is_some_and(|value| value.resume_from.is_some())
        && lora.is_some_and(|section| section.init_adapter.is_some())
    {
        return Err(Error::config(
            "checkpoint.resume_from and lora.init_adapter are mutually exclusive",
        ));
    }
    // The update step is a kernel with a dtype table, and SGD's carries F32
    // alone. F16 is the *default* adapter storage, so this pair is the ordinary
    // way to ask for it - and the runtime's own refusal would arrive after the
    // model is loaded and the training graph is built.
    // Asked through `supports_dtype` rather than by naming SGD and F16: the
    // table belongs to the optimizer.
    let adapter_dtype = lora_config.map(|config| match config.dtype {
        LoraDtype::F32 => TensorDtype::F32,
        LoraDtype::F16 => TensorDtype::F16,
    });
    if adapter_dtype
        .as_ref()
        .is_some_and(|dtype| !trainable.optimizer.supports_dtype(dtype))
        && !lora.is_some_and(|section| section.init_adapter.is_some())
        && !checkpoint.is_some_and(|value| value.resume_from.is_some())
    {
        return Err(Error::config(format!(
            "training.optimizer = '{}' cannot write a {} adapter: its update step does not \
             carry that precision. Set lora.dtype = 'f32', or keep the default optimizer",
            trainable.optimizer,
            adapter_dtype.as_ref().expect("checked just above"),
        )));
    }

    Ok(())
}

/// Which enabled consumer of a fixed reference this algorithm carries, if any.
///
/// A KL coefficient of zero is not a reference: the penalty term is skipped
/// entirely and nothing scores the frozen model. PPO's stored old-policy
/// logprobs are a different concept again - they come from the policy that
/// generated the rollout, not from an anchor - and are deliberately not listed.
fn fixed_reference_consumer(algorithm: &Algorithm) -> Option<&'static str> {
    match algorithm {
        Algorithm::Sft(_) => None,
        Algorithm::Ppo(ppo) => (ppo.kl_coefficient > 0.0).then_some("ppo.kl_coefficient"),
        Algorithm::Grpo(grpo) => {
            // The schedule as well as the value: a run configured to warm up to
            // a penalty has a reference from the first update, whatever the
            // coefficient reads at update one.
            (grpo.kl_coefficient > 0.0 || grpo.kl_schedule.is_some())
                .then_some("grpo.kl_coefficient")
        }
        Algorithm::Distill(distill) => {
            (distill.kl_coefficient > 0.0).then_some("distill.kl_coefficient")
        }
        Algorithm::AgentGrpo(agent) => {
            (agent.config.kl_coefficient > 0.0).then_some("agent.kl_coefficient")
        }
    }
}

fn build_observe(value: ObserveToml, algorithm: &Algorithm, root: &Path) -> Result<ObserveConfig> {
    // Only the rollout algorithms produce something to look at.
    match algorithm {
        Algorithm::Ppo(_) | Algorithm::Grpo(_) | Algorithm::AgentGrpo(_) => {}
        Algorithm::Sft(_) | Algorithm::Distill(_) => {
            return Err(Error::config(
                "[observe] exports rollouts and is only supported for ppo, grpo and agent_grpo",
            ));
        }
    }
    let every = value.every.unwrap_or(1);
    require_nonzero(every, "observe.every must be greater than zero")?;
    Ok(ObserveConfig {
        directory: resolve(root, value.directory),
        every,
        max_text_chars: value.max_text_chars.unwrap_or(0),
    })
}

fn build_evaluation(value: EvaluationToml, root: &Path) -> Result<EvaluationConfig> {
    let every_iterations = value.every_iterations.unwrap_or(1);
    require_nonzero(
        every_iterations,
        "evaluation.every_iterations must be greater than zero",
    )?;
    if let Some(patience) = value.patience {
        require_nonzero(
            patience,
            "evaluation.patience must be greater than zero when set",
        )?;
    }
    let min_delta = value.min_delta.unwrap_or(0.0);
    require_non_negative_f64(min_delta, "evaluation.min_delta")?;
    if let Some(max_examples) = value.max_examples {
        require_nonzero(
            max_examples,
            "evaluation.max_examples must be greater than zero when set",
        )?;
    }
    Ok(EvaluationConfig {
        data: resolve(root, value.data),
        every_iterations,
        patience: value.patience,
        min_delta,
        max_examples: value.max_examples,
    })
}

fn build_checkpoint(value: CheckpointToml, root: &Path) -> Result<CheckpointConfig> {
    let mode = parse_string_enum!(
        value.mode,
        "checkpoint.mode must be steps, best_eval, or steps_and_best_eval",
        "steps" => CheckpointMode::Steps,
        "best_eval" => CheckpointMode::BestEval,
        "steps_and_best_eval" => CheckpointMode::StepsAndBestEval,
    )?;
    if mode.includes_steps() && value.every_steps.is_none_or(|steps| steps == 0) {
        return Err(Error::config(
            "checkpoint.every_steps must be greater than zero when mode includes steps",
        ));
    }
    if !mode.includes_steps() && value.every_steps.is_some() {
        return Err(Error::config(
            "checkpoint.every_steps is only valid when mode includes steps",
        ));
    }
    Ok(CheckpointConfig {
        directory: resolve(root, value.directory),
        mode,
        every_steps: value.every_steps,
        resume_from: value.resume_from.map(|path| resolve(root, path)),
    })
}
