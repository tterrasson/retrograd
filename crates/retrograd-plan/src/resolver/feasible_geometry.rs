//! Feasible geometry

use super::*;

/// Picks the geometry.
///
/// The objective is "the largest geometry whose estimate fits **with the safety
/// margin**", not "the largest that fits the raw budget".
/// That is what [`Budgets::overflow`] already compares against - `effective_bytes`
/// is the total less the baseline *and* less [`MarginPolicy`] - so the search
/// below stops at the first candidate inside the margin, which is where a run
/// that has to survive fragmentation wants to be.
pub(super) fn choose_geometry(
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
            Some(lora),
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
