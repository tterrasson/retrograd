//! Proactive defaults

use super::*;

/// Applies the active defaults, in order, on what phase 2 chose.
///
/// The estimate is recomputed before each row, so a default that lowers the
/// activation term is visible to the one that reads it. Nothing here is
/// conditional on the run overflowing: that is exactly the difference from phase
/// 3.
pub(super) fn apply_defaults(
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
        trainable,
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
            trainable,
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
