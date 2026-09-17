//! Document round-trip

use super::*;

pub(super) fn apply_params(
    document: &mut ConfigDocument,
    params: &Value,
) -> Result<(), ResolveError> {
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
pub(super) fn write_back(
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
pub(super) fn write_cadences(
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
