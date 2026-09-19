//! Memory recovery

use super::*;

/// What phase 3 did: the levers it pulled, and the ones it could not.
pub(super) struct LeverOutcome {
    pub(super) applied: Vec<AppliedLever>,
    /// Levers a lock forbade - reported so the refusal can say the run was
    /// over budget *and* which rescue the caller had ruled out.
    pub(super) blocked: Vec<&'static str>,
}

pub(super) fn apply_levers(
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
            Some(lora),
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
