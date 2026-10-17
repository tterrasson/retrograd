//! Identity of a run: what a resume must find unchanged, and what a metrics
//! backend needs to label it.

use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use retrograd_checkpoint as checkpoint;
use retrograd_config::{Algorithm, PreferenceLoss, RunConfig};
use retrograd_core::{
    Error, GenerationKvDtype, ImportanceSamplingLevel, LrScheduler, Result, SamplingParams,
};
use retrograd_metrics::RunMetadata;
use retrograd_tools::ToolPlanResolve;

pub fn scheduler_name(scheduler: LrScheduler) -> &'static str {
    match scheduler {
        LrScheduler::Constant => "constant",
        LrScheduler::Linear => "linear",
        LrScheduler::Cosine => "cosine",
    }
}

pub fn model_bytes(path: &Path) -> u64 {
    fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}

pub fn checked_total_steps(label: &str, factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1_u64, |total, factor| {
        total
            .checked_mul(*factor)
            .ok_or_else(|| Error::overflow(format!("{label} optimizer step count overflows u64")))
    })
}

/// Accumulates the `|key=value` fields of a trajectory signature. The byte
/// format is frozen: field names, order and encodings are the checkpoint
/// contract, and existing signatures must not change.
struct Descriptor(String);

impl Descriptor {
    fn new(header: &str) -> Self {
        Self(header.to_owned())
    }

    /// A bare `|tag` with no `=`, e.g. the algorithm name.
    fn tag(&mut self, tag: &str) {
        self.0.push('|');
        self.0.push_str(tag);
    }

    fn field(&mut self, name: &str, value: impl fmt::Display) {
        self.0.push('|');
        self.0.push_str(name);
        self.0.push('=');
        write!(self.0, "{value}").expect("writing to a String never fails");
    }

    fn debug(&mut self, name: &str, value: impl fmt::Debug) {
        self.field(name, format!("{value:?}"));
    }

    /// f32 as its bits in eight hex digits, the encoding every float uses.
    fn float(&mut self, name: &str, value: f32) {
        self.field(name, format!("{:08x}", value.to_bits()));
    }

    /// The four rollout sampling parameters, shared verbatim by PPO, GRPO and
    /// distillation: `temperature,top_p,max_new_tokens,seed`.
    fn sampling(&mut self, sampling: &SamplingParams) {
        self.field(
            "sampling",
            format!(
                "{:08x},{:08x},{},{}",
                sampling.temperature.to_bits(),
                sampling.top_p.to_bits(),
                sampling.max_new_tokens,
                sampling.seed,
            ),
        );
    }

    /// The sequence-level ratio is a different objective, so a resume must not
    /// cross it; the per-token default writes nothing, which keeps every
    /// signature written before the level existed.
    fn importance_sampling(&mut self, level: ImportanceSamplingLevel) {
        if level != ImportanceSamplingLevel::Token {
            self.field("is", level.as_str());
        }
    }

    fn finish(self) -> String {
        self.0
    }
}

/// Stable fingerprint of every setting that can alter the continued training
/// trajectory. Output paths and logging-only options are deliberately absent.
pub fn trajectory_signature(config: &RunConfig) -> Result<String> {
    let training = &config.training;
    let mut descriptor = Descriptor::new("trajectory-v1");
    descriptor.field("n_ctx", training.n_ctx);
    descriptor.field("n_batch", training.n_batch);
    descriptor.field("n_ubatch", training.n_ubatch);
    descriptor.field("n_seq_max", training.n_seq_max);
    descriptor.field("generation_concurrency", training.generation_concurrency);
    descriptor.field("fast_generation", training.fast_generation_context);
    descriptor.debug("kv_dtype", training.kv_dtype);
    descriptor.field("gradient_checkpointing", training.gradient_checkpointing);
    descriptor.field(
        "checkpoint_every_n_layers",
        training.checkpoint_every_n_layers,
    );
    // Part of the trajectory identity, not a tuning knob: F16 checkpoints
    // change the recomputed activations, so a resume must not silently cross
    // that boundary.
    descriptor.debug("checkpoint_dtype", training.checkpoint_dtype);
    // The rounding path of each update, so a resume must not cross it.
    descriptor.field("master_weights", training.master_weights);
    descriptor.field("threads", training.threads);
    descriptor.field("epochs", training.epochs);
    descriptor.float("lr", training.learning_rate);
    descriptor.float("wd", training.weight_decay);
    descriptor.float("max_grad_norm", training.max_grad_norm);
    descriptor.field("scheduler", scheduler_name(training.lr_scheduler));
    descriptor.field("warmup", training.warmup_steps);
    descriptor.debug("device", training.device);
    // The distribution the rollouts are drawn from, like `fast_generation`.
    // Absent at the default, so existing signatures stay put.
    if training.generation_kv_dtype != GenerationKvDtype::F16 {
        descriptor.field("generation_kv_dtype", training.generation_kv_dtype.as_str());
    }
    if let Some(reference) = &config.reference {
        // The context can change RoPE scaling and therefore the anchor's scores.
        // Its file identity is checked separately by the checkpoint manifest.
        descriptor.field("reference_ctx", reference.n_ctx.unwrap_or(training.n_ctx));
    }
    match &config.algorithm {
        Algorithm::Sft(sft) => {
            descriptor.tag("sft");
            descriptor.debug("format", sft.data_format);
            // The prepared tokens are fingerprinted already, but a resume is
            // refused on the variables themselves, as an agentic run's is.
            // Absent when empty, so existing signatures stay put.
            if !sft.template_variables.is_empty() {
                descriptor.field(
                    "template_variables",
                    checkpoint::fingerprint(sft.template_variables_json().as_bytes()),
                );
            }
        }
        Algorithm::Ppo(ppo) => {
            descriptor.tag("ppo");
            descriptor.debug("reward", &ppo.reward_command);
            descriptor.field("updates", ppo.updates);
            descriptor.field("batch", ppo.rollout_batch_size);
            descriptor.field("epochs", ppo.ppo_epochs);
            descriptor.float("clip", ppo.clip_range);
            descriptor.float("kl", ppo.kl_coefficient);
            // One `critic=` field carrying five sub-values.
            descriptor.field(
                "critic",
                format!(
                    "{},{:08x},{:08x},{:08x},{}",
                    ppo.critic.enabled,
                    ppo.critic.gamma.to_bits(),
                    ppo.critic.gae_lambda.to_bits(),
                    ppo.critic.value_lr.to_bits(),
                    ppo.critic.value_epochs,
                ),
            );
            descriptor.sampling(&ppo.sampling);
        }
        Algorithm::Grpo(grpo) => {
            descriptor.tag("grpo");
            descriptor.debug("reward", &grpo.reward_command);
            descriptor.field("updates", grpo.updates);
            descriptor.field("prompts", grpo.prompts_per_update);
            descriptor.field("group", grpo.group_size);
            descriptor.field("epochs", grpo.grpo_epochs);
            descriptor.field(
                "clip",
                format!(
                    "{:08x},{:08x}",
                    grpo.clip_range_low.to_bits(),
                    grpo.clip_range_high.to_bits(),
                ),
            );
            descriptor.float("kl", grpo.kl_coefficient);
            descriptor.field("mask", grpo.mask_truncated);
            descriptor.debug("baseline", grpo.baseline);
            descriptor.debug("order", grpo.prompt_order);
            descriptor.debug("overlong", grpo.overlong_penalty);
            descriptor.debug("kl_schedule", grpo.kl_schedule);
            descriptor.debug("dynamic", grpo.dynamic_sampling);
            // The judge is part of the reward, so it is part of what a
            // resume must find unchanged: swapping the model behind it, or
            // the weight its verdict carries, is a different objective.
            descriptor.debug(
                "judge",
                grpo.judge.as_ref().map(|judge| {
                    (
                        &judge.config,
                        judge.weight.to_bits(),
                        judge.failure,
                        judge.max_dropped_fraction.to_bits(),
                    )
                }),
            );
            descriptor.sampling(&grpo.sampling);
            descriptor.importance_sampling(grpo.importance_sampling_level);
        }
        // The offline mode resumes against a sidecar, and the sidecar *is* the
        // objective the way the teacher is on the on-policy path. Its path plus
        // its header - k, the corpus hash, the tokenizer hash - is what a resume
        // must not change; the entries themselves are not read, for the same
        // reason a GGUF is not hashed.
        Algorithm::Distill(distill) if !distill.mode.is_rollout() => {
            let offline = distill
                .mode
                .offline()
                .expect("a non-rollout distill mode is the offline one");
            let header = retrograd_dataset::topk::TopKSidecar::read(&offline.sidecar)?.header();
            descriptor.tag("distill_offline");
            descriptor.field("teacher", distill.teacher_path.display());
            descriptor.field("data", offline.data.display());
            descriptor.field("sidecar", offline.sidecar.display());
            descriptor.field("epochs", offline.epochs);
            descriptor.field("k", header.k);
            descriptor.field("corpus", format!("{:016x}", header.source_hash));
            descriptor.field("tokenizer", format!("{:016x}", header.tokenizer_hash));
            descriptor.float("weight_clip", distill.weight_clip);
        }
        Algorithm::Distill(distill) => {
            descriptor.tag("distill");
            // The teacher *is* the objective here, the way the reward
            // command is GRPO's: resuming against another model would keep
            // the optimizer state and change what it is being pulled
            // toward. Its path, not its bytes - the same treatment
            // `config.model` gets, since a GGUF is not read to be hashed.
            descriptor.field("teacher", distill.teacher_path.display());
            descriptor.field("updates", distill.updates);
            descriptor.field("prompts", distill.prompts_per_update);
            descriptor.field("samples", distill.samples_per_prompt);
            descriptor.field("epochs", distill.distill_epochs);
            descriptor.field(
                "clip",
                format!(
                    "{:08x},{:08x}",
                    distill.clip_range_low.to_bits(),
                    distill.clip_range_high.to_bits(),
                ),
            );
            descriptor.float("weight_clip", distill.weight_clip);
            descriptor.float("kl", distill.kl_coefficient);
            descriptor.field("mask", distill.mask_truncated);
            descriptor.debug("order", distill.prompt_order);
            descriptor.sampling(&distill.sampling);
        }
        // The prepared pairs are fingerprinted with the dataset, together with
        // the reference's identity; what is here is the objective.
        Algorithm::Preference(preference) => {
            let (label_smoothing, gamma_beta_ratio) = match preference.loss {
                PreferenceLoss::Dpo {
                    label_smoothing, ..
                } => (label_smoothing, 0.0),
                PreferenceLoss::Simpo {
                    gamma_beta_ratio, ..
                } => (0.0, gamma_beta_ratio),
                PreferenceLoss::Ipo { .. } | PreferenceLoss::Orpo { .. } => (0.0, 0.0),
            };
            descriptor.tag("preference");
            descriptor.field("data", preference.data.display());
            descriptor.field("loss", preference.loss.name());
            descriptor.float("beta", preference.loss.beta());
            descriptor.field(
                "ref",
                preference
                    .reference
                    .map_or("none", |reference| reference.as_str()),
            );
            descriptor.float("ls", label_smoothing);
            descriptor.float("gamma", gamma_beta_ratio);
            descriptor.field("shuffle", preference.shuffle);
            descriptor.debug("pps", preference.pairs_per_step);
        }
        Algorithm::AgentGrpo(agent) => {
            let config = &agent.config;
            let corpus = fs::read(&agent.scenarios)?;
            // The merged view, not the declaration: a server whose command or
            // filters changed inside an `mcp.json` changes what the policy can
            // do, and a resume must not mistake it for the same run. Reading
            // those files is legitimate here - the signature is computed by the
            // machine that is about to execute the document.
            let (mcp_servers, _warnings) = agent.tool_plan.merged_servers().map_err(Error::from)?;
            // The session tools as resolved, not as declared: a definition file
            // or an exec script edited between a run and its resume changes what
            // the model is offered, and the catalogue hash is what sees it.
            let tools = agent
                .tool_plan
                .resolve_local(&retrograd_tools::ToolRegistry::builtin())
                .map_err(Error::from)?
                .catalog
                .sha256;
            descriptor.tag("agent_grpo");
            descriptor.field("scenarios", agent.scenarios.display());
            descriptor.field("corpus", checkpoint::fingerprint(&corpus));
            descriptor.field("updates", config.updates);
            descriptor.field("scenarios_per_update", config.scenarios_per_update);
            descriptor.field("group", config.group_size);
            descriptor.field("epochs", config.epochs);
            descriptor.field(
                "clip",
                format!(
                    "{:08x},{:08x}",
                    config.clip_range_low.to_bits(),
                    config.clip_range_high.to_bits(),
                ),
            );
            descriptor.float("kl", config.kl_coefficient);
            // One `limits=` field carrying the six rollout bounds.
            descriptor.field(
                "limits",
                format!(
                    "{},{},{},{},{},{}",
                    config.limits.max_turns,
                    config.limits.max_new_tokens_per_turn,
                    config.limits.max_trajectory_tokens,
                    config.limits.max_rollout_secs,
                    config.limits.end_on_no_tool_call,
                    config.limits.max_failed_turns,
                ),
            );
            descriptor.debug("judge_failure", config.judge_failure);
            descriptor.float("dropped", config.max_dropped_fraction);
            descriptor.field("degenerate", config.drop_degenerate_groups);
            descriptor.field("skip_empty", config.skip_empty_updates);
            descriptor.debug("truncation", config.truncation);
            descriptor.field("seed", config.seed);
            descriptor.debug("judge", &agent.judge);
            descriptor.debug("environment", &agent.environment);
            descriptor.debug("mcp", mcp_servers);
            descriptor.field("tools", tools);
            descriptor.importance_sampling(config.importance_sampling_level);
            // Preserve existing checkpoint signatures when both options use their defaults.
            if !agent.system_suffix.is_empty() {
                descriptor.field(
                    "system_suffix",
                    checkpoint::fingerprint(agent.system_suffix.as_bytes()),
                );
            }
            if !agent.template_variables.is_empty() {
                descriptor.field(
                    "template_variables",
                    checkpoint::fingerprint(agent.template_variables_json().as_bytes()),
                );
            }
            // It decides which tokens a trajectory holds and which it trains.
            if let Some(budget) = agent.reasoning_budget {
                descriptor.field("reasoning_budget", budget);
            }
        }
    }
    if let Some(evaluation) = &config.evaluation {
        let content = fs::read(&evaluation.data)?;
        descriptor.field(
            "evaluation",
            format!(
                "{},{:?},{:016x},{:?},{}",
                evaluation.every_iterations,
                evaluation.patience,
                evaluation.min_delta.to_bits(),
                evaluation.max_examples,
                checkpoint::fingerprint(&content),
            ),
        );
    } else {
        descriptor.field("evaluation", "none");
    }
    Ok(checkpoint::fingerprint(descriptor.finish().as_bytes()))
}

/// Fingerprints an SFT dataset over its prepared rows rather than the source
/// file: tokenization and the label mask are part of what a resume must find
/// unchanged, and both are already materialized here.
///
/// Hashed in place: the rows are the run's largest host buffer (and `labels`
/// holds `k` entries per position under offline distillation), so serializing
/// them first would double it at startup for a 32-byte answer.
pub fn dataset_fingerprint(tokens: &[i32], labels: &[i32]) -> String {
    let mut fingerprinter = checkpoint::Fingerprinter::new();
    for value in tokens.iter().chain(labels) {
        fingerprinter.update(&value.to_le_bytes());
    }
    fingerprinter.finish()
}

/// Dataset descriptor of a rollout run: the prompts file, fingerprinted by
/// content. Its tokenization is derived deterministically from the same model,
/// which the manifest already pins.
pub fn prompts_dataset(path: &Path, max_new_tokens: u32) -> Result<checkpoint::Dataset> {
    let content = fs::read(path)?;
    let examples = content
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .count() as u64;
    Ok(checkpoint::Dataset {
        version: checkpoint::FORMAT_VERSION,
        path: path.display().to_string(),
        fingerprint: checkpoint::fingerprint(&content),
        examples,
        row_width: max_new_tokens as u64,
        format: "prompts".into(),
        permutation: Vec::new(),
        cursor: 0,
    })
}

pub fn metadata(config: &RunConfig) -> RunMetadata {
    let (algorithm, train_data) = match &config.algorithm {
        Algorithm::Sft(value) => ("sft", value.data.display().to_string()),
        Algorithm::Ppo(value) => ("ppo", value.prompts.display().to_string()),
        Algorithm::Grpo(value) => ("grpo", value.prompts.display().to_string()),
        Algorithm::Distill(value) => (
            "distill",
            match value.mode.offline() {
                Some(offline) => offline.data.display().to_string(),
                None => value.prompts.display().to_string(),
            },
        ),
        Algorithm::AgentGrpo(value) => ("agent_grpo", value.scenarios.display().to_string()),
        Algorithm::Preference(value) => ("preference", value.data.display().to_string()),
    };
    RunMetadata {
        algorithm: algorithm.into(),
        model: config.model.display().to_string(),
        train_data,
        eval_data: config
            .evaluation
            .as_ref()
            .map(|evaluation| evaluation.data.display().to_string()),
        epochs: config.training.epochs,
        learning_rate: config.training.learning_rate,
        scheduler: config.training.lr_scheduler.name().into(),
        warmup_steps: config.training.warmup_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{temp_path, write_sft_config};
    use retrograd_config as config;

    #[test]
    fn trajectory_and_dataset_fingerprints_are_stable_and_sensitive_to_inputs() {
        let root = temp_path("fingerprint");
        let config_path = write_sft_config(&root);
        let config = config::load(&config_path).unwrap();
        let signature = trajectory_signature(&config).unwrap();
        assert_eq!(signature, trajectory_signature(&config).unwrap());

        let mut changed = config.clone();
        changed.training.epochs += 1;
        assert_ne!(signature, trajectory_signature(&changed).unwrap());
        assert_ne!(
            dataset_fingerprint(&[1, 2], &[-1, 2]),
            dataset_fingerprint(&[1, 2], &[-1, 3])
        );
        assert_ne!(
            dataset_fingerprint(&[1, 2], &[-1, 2]),
            dataset_fingerprint(&[2, 1], &[-1, 2])
        );
        assert_eq!(checked_total_steps("test", &[2, 3, 4]).unwrap(), 24);
        assert!(checked_total_steps("test", &[u64::MAX, 2]).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dataset_fingerprint_matches_the_serialized_rows_it_replaced() {
        // A checkpoint written before hashing moved in place must still resume.
        let tokens = [1, -2, 300_000, i32::MAX];
        let labels = [-100, 7, i32::MIN];
        let bytes = tokens
            .iter()
            .chain(&labels)
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            dataset_fingerprint(&tokens, &labels),
            checkpoint::fingerprint(&bytes)
        );
    }

    #[test]
    fn reference_context_changes_the_trajectory_but_its_path_does_not() {
        let root = temp_path("reference-signature");
        let mut config = config::load(write_sft_config(&root)).unwrap();
        config.reference = Some(retrograd_config::ReferenceConfig {
            model: root.join("anchor.gguf"),
            n_ctx: None,
        });
        let signature = trajectory_signature(&config).unwrap();
        config.reference.as_mut().unwrap().n_ctx = Some(config.training.n_ctx);
        assert_eq!(signature, trajectory_signature(&config).unwrap());
        config.reference.as_mut().unwrap().model = root.join("renamed.gguf");
        assert_eq!(signature, trajectory_signature(&config).unwrap());
        config.reference.as_mut().unwrap().n_ctx = Some(config.training.n_ctx * 2);
        assert_ne!(signature, trajectory_signature(&config).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sft_template_variables_change_the_trajectory_only_when_set() {
        let root = temp_path("sft-template-variables-signature");
        let config = config::load(write_sft_config(&root)).unwrap();
        let with = |variables: &[(&str, bool)]| {
            let mut config = config.clone();
            let config::Algorithm::Sft(sft) = &mut config.algorithm else {
                panic!("an SFT config");
            };
            sft.template_variables = variables
                .iter()
                .map(|(name, value)| ((*name).to_owned(), serde_json::Value::Bool(*value)))
                .collect();
            trajectory_signature(&config).unwrap()
        };
        // Empty is the default every existing checkpoint was written under.
        assert_eq!(trajectory_signature(&config).unwrap(), with(&[]));
        let off = with(&[("enable_thinking", false)]);
        assert_ne!(with(&[]), off);
        assert_ne!(off, with(&[("enable_thinking", true)]));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_gpu_duty_cycle_stays_out_of_the_trajectory_signature() {
        // Sleeping cannot alter the training problem or the serialized
        // optimizer state, so a checkpoint must resume under a different duty
        // cycle. Putting it in the signature would refuse that resume.
        let root = temp_path("duty-cycle-signature");
        let config_path = write_sft_config(&root);
        let config = config::load(&config_path).unwrap();
        let signature = trajectory_signature(&config).unwrap();

        let mut throttled = config.clone();
        throttled.training.max_gpu_duty_cycle = Some(0.25);
        assert_eq!(signature, trajectory_signature(&throttled).unwrap());

        let mut other = config;
        other.training.max_gpu_duty_cycle = Some(0.75);
        assert_eq!(signature, trajectory_signature(&other).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_importance_sampling_level_changes_the_trajectory_only_when_sequence() {
        let root = temp_path("importance-sampling-signature");
        fs::create_dir_all(&root).unwrap();
        let config_path = root.join("run.toml");
        fs::write(
            &config_path,
            "[run]\nalgorithm='grpo'\n[model]\npath='model.gguf'\n[output]\npath='out.gguf'\n\
             [lora]\n[grpo]\nprompts='p.jsonl'\nreward_command=['r']\nupdates=1\n\
             prompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\nclip_range_low=0.2\n\
             clip_range_high=0.28\nkl_coefficient=0.0\n[grpo.sampling]\ntemperature=1.0\n\
             top_p=1.0\nmax_new_tokens=8\nseed=1\n",
        )
        .unwrap();
        let config = config::load(&config_path).unwrap();
        fs::remove_dir_all(root).unwrap();
        let token = trajectory_signature(&config).unwrap();
        // Frozen: a checkpoint written by a per-token run before the level
        // existed must keep resuming.
        assert_eq!(token, "296194daf3e30267382c81e66820b053");
        let mut sequence = config;
        let config::Algorithm::Grpo(grpo) = &mut sequence.algorithm else {
            panic!("a GRPO config");
        };
        grpo.importance_sampling_level = ImportanceSamplingLevel::Sequence;
        assert_ne!(token, trajectory_signature(&sequence).unwrap());
        // A quantized sampling cache draws from another distribution: a resume
        // must not cross it either way.
        let mut quantized = sequence.clone();
        quantized.training.generation_kv_dtype = GenerationKvDtype::Q8_0;
        assert_ne!(
            trajectory_signature(&sequence).unwrap(),
            trajectory_signature(&quantized).unwrap()
        );
    }

    #[test]
    fn metadata_describes_the_run() {
        let root = temp_path("metadata");
        let config_path = write_sft_config(&root);
        let config = config::load(&config_path).unwrap();
        let metadata = metadata(&config);
        assert_eq!(metadata.algorithm, "sft");
        assert_eq!(
            metadata.train_data,
            root.join("data.txt").display().to_string()
        );
        assert_eq!(metadata.scheduler, "constant");
        fs::remove_dir_all(root).unwrap();
    }
}
