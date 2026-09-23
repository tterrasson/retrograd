//! Identity of a run: what a resume must find unchanged, and what a metrics
//! backend needs to label it.

use std::fs;
use std::path::Path;

use retrograd_checkpoint as checkpoint;
use retrograd_config::{Algorithm, RunConfig};
use retrograd_core::{Error, LrScheduler, Result};
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

/// Stable fingerprint of every setting that can alter the continued training
/// trajectory. Output paths and logging-only options are deliberately absent.
pub fn trajectory_signature(config: &RunConfig) -> Result<String> {
    use std::fmt::Write as _;

    let training = &config.training;
    let mut descriptor = format!(
        "trajectory-v1|n_ctx={}|n_batch={}|n_ubatch={}|n_seq_max={}|generation_concurrency={}|fast_generation={}|kv_dtype={:?}|gradient_checkpointing={}|checkpoint_every_n_layers={}|checkpoint_dtype={:?}|master_weights={}|threads={}|epochs={}|lr={:08x}|wd={:08x}|max_grad_norm={:08x}|scheduler={}|warmup={}|device={:?}",
        training.n_ctx,
        training.n_batch,
        training.n_ubatch,
        training.n_seq_max,
        training.generation_concurrency,
        training.fast_generation_context,
        training.kv_dtype,
        training.gradient_checkpointing,
        training.checkpoint_every_n_layers,
        // Part of the trajectory identity, not a tuning knob: F16 checkpoints
        // change the recomputed activations, so a resume must not silently cross
        // that boundary.
        training.checkpoint_dtype,
        // The rounding path of each update, so a resume must not cross it.
        training.master_weights,
        training.threads,
        training.epochs,
        training.learning_rate.to_bits(),
        training.weight_decay.to_bits(),
        training.max_grad_norm.to_bits(),
        scheduler_name(training.lr_scheduler),
        training.warmup_steps,
        training.device,
    );
    if let Some(reference) = &config.reference {
        // The context can change RoPE scaling and therefore the anchor's scores.
        // Its file identity is checked separately by the checkpoint manifest.
        write!(
            &mut descriptor,
            "|reference_ctx={}",
            reference.n_ctx.unwrap_or(training.n_ctx)
        )
        .expect("writing to a String never fails");
    }
    match &config.algorithm {
        Algorithm::Sft(sft) => {
            write!(&mut descriptor, "|sft|format={:?}", sft.data_format)
                .expect("writing to a String never fails");
        }
        Algorithm::Ppo(ppo) => {
            write!(
                &mut descriptor,
                "|ppo|reward={:?}|updates={}|batch={}|epochs={}|clip={:08x}|kl={:08x}|critic={},{:08x},{:08x},{:08x},{}|sampling={:08x},{:08x},{},{}",
                ppo.reward_command,
                ppo.updates,
                ppo.rollout_batch_size,
                ppo.ppo_epochs,
                ppo.clip_range.to_bits(),
                ppo.kl_coefficient.to_bits(),
                ppo.critic.enabled,
                ppo.critic.gamma.to_bits(),
                ppo.critic.gae_lambda.to_bits(),
                ppo.critic.value_lr.to_bits(),
                ppo.critic.value_epochs,
                ppo.sampling.temperature.to_bits(),
                ppo.sampling.top_p.to_bits(),
                ppo.sampling.max_new_tokens,
                ppo.sampling.seed,
            )
            .expect("writing to a String never fails");
        }
        Algorithm::Grpo(grpo) => {
            write!(
                &mut descriptor,
                "|grpo|reward={:?}|updates={}|prompts={}|group={}|epochs={}|clip={:08x},{:08x}|kl={:08x}|mask={}|baseline={:?}|order={:?}|overlong={:?}|kl_schedule={:?}|dynamic={:?}|judge={:?}|sampling={:08x},{:08x},{},{}",
                grpo.reward_command,
                grpo.updates,
                grpo.prompts_per_update,
                grpo.group_size,
                grpo.grpo_epochs,
                grpo.clip_range_low.to_bits(),
                grpo.clip_range_high.to_bits(),
                grpo.kl_coefficient.to_bits(),
                grpo.mask_truncated,
                grpo.baseline,
                grpo.prompt_order,
                grpo.overlong_penalty,
                grpo.kl_schedule,
                grpo.dynamic_sampling,
                // The judge is part of the reward, so it is part of what a
                // resume must find unchanged: swapping the model behind it, or
                // the weight its verdict carries, is a different objective.
                grpo.judge.as_ref().map(|judge| {
                    (
                        &judge.config,
                        judge.weight.to_bits(),
                        judge.failure,
                        judge.max_dropped_fraction.to_bits(),
                    )
                }),
                grpo.sampling.temperature.to_bits(),
                grpo.sampling.top_p.to_bits(),
                grpo.sampling.max_new_tokens,
                grpo.sampling.seed,
            )
            .expect("writing to a String never fails");
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
            write!(
                &mut descriptor,
                "|distill_offline|teacher={}|data={}|sidecar={}|epochs={}|k={}|corpus={:016x}|tokenizer={:016x}|weight_clip={:08x}",
                distill.teacher_path.display(),
                offline.data.display(),
                offline.sidecar.display(),
                offline.epochs,
                header.k,
                header.source_hash,
                header.tokenizer_hash,
                distill.weight_clip.to_bits(),
            )
            .expect("writing to a String never fails");
        }
        Algorithm::Distill(distill) => {
            write!(
                &mut descriptor,
                "|distill|teacher={}|updates={}|prompts={}|samples={}|epochs={}|clip={:08x},{:08x}|weight_clip={:08x}|kl={:08x}|mask={}|order={:?}|sampling={:08x},{:08x},{},{}",
                // The teacher *is* the objective here, the way the reward
                // command is GRPO's: resuming against another model would keep
                // the optimizer state and change what it is being pulled
                // toward. Its path, not its bytes - the same treatment
                // `config.model` gets, since a GGUF is not read to be hashed.
                distill.teacher_path.display(),
                distill.updates,
                distill.prompts_per_update,
                distill.samples_per_prompt,
                distill.distill_epochs,
                distill.clip_range_low.to_bits(),
                distill.clip_range_high.to_bits(),
                distill.weight_clip.to_bits(),
                distill.kl_coefficient.to_bits(),
                distill.mask_truncated,
                distill.prompt_order,
                distill.sampling.temperature.to_bits(),
                distill.sampling.top_p.to_bits(),
                distill.sampling.max_new_tokens,
                distill.sampling.seed,
            )
            .expect("writing to a String never fails");
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
            write!(
                &mut descriptor,
                "|agent_grpo|scenarios={}|corpus={}|updates={}|scenarios_per_update={}|group={}|epochs={}|clip={:08x},{:08x}|kl={:08x}|limits={},{},{},{},{},{}|judge_failure={:?}|dropped={:08x}|degenerate={}|skip_empty={}|truncation={:?}|seed={}|judge={:?}|environment={:?}|mcp={:?}",
                agent.scenarios.display(),
                checkpoint::fingerprint(&corpus),
                config.updates,
                config.scenarios_per_update,
                config.group_size,
                config.epochs,
                config.clip_range_low.to_bits(),
                config.clip_range_high.to_bits(),
                config.kl_coefficient.to_bits(),
                config.limits.max_turns,
                config.limits.max_new_tokens_per_turn,
                config.limits.max_trajectory_tokens,
                config.limits.max_rollout_secs,
                config.limits.end_on_no_tool_call,
                config.limits.max_failed_turns,
                config.judge_failure,
                config.max_dropped_fraction.to_bits(),
                config.drop_degenerate_groups,
                config.skip_empty_updates,
                config.truncation,
                config.seed,
                agent.judge,
                agent.environment,
                mcp_servers,
            )
            .expect("writing to a String never fails");
            // Preserve existing checkpoint signatures when both options use their defaults.
            if !agent.system_suffix.is_empty() {
                write!(
                    &mut descriptor,
                    "|system_suffix={}",
                    checkpoint::fingerprint(agent.system_suffix.as_bytes()),
                )
                .expect("writing to a String never fails");
            }
            if !agent.template_variables.is_empty() {
                write!(
                    &mut descriptor,
                    "|template_variables={}",
                    checkpoint::fingerprint(agent.template_variables_json().as_bytes()),
                )
                .expect("writing to a String never fails");
            }
        }
    }
    if let Some(evaluation) = &config.evaluation {
        let content = fs::read(&evaluation.data)?;
        write!(
            &mut descriptor,
            "|evaluation={},{:?},{:016x},{:?},{}",
            evaluation.every_iterations,
            evaluation.patience,
            evaluation.min_delta.to_bits(),
            evaluation.max_examples,
            checkpoint::fingerprint(&content),
        )
        .expect("writing to a String never fails");
    } else {
        descriptor.push_str("|evaluation=none");
    }
    Ok(checkpoint::fingerprint(descriptor.as_bytes()))
}

/// Fingerprints an SFT dataset over its prepared rows rather than the source
/// file: tokenization and the label mask are part of what a resume must find
/// unchanged, and both are already materialized here.
pub fn dataset_fingerprint(tokens: &[i32], labels: &[i32]) -> String {
    let mut bytes = Vec::with_capacity((tokens.len() + labels.len()) * 4);
    for value in tokens.iter().chain(labels) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    checkpoint::fingerprint(&bytes)
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
