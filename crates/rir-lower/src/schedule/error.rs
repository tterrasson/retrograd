//! Schedule errors, and the validation every consumer goes through before it
//! reads a field.

use rir_core::IrId;

use rir_core::{Kernel, Op, ReductionSemantics, ValueId};

use crate::schedule::*;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error(
        "reduction %{} ({:?}) incompatible with schedule strategy {}",
        .value.0, .semantics, .strategy.name()
    )]
    IncompatibleReduction {
        value: ValueId,
        semantics: ReductionSemantics,
        strategy: ReductionStrategy,
    },
    /// Two schedules of one (kernel, backend) pair claim the same identity, so
    /// their artifacts would overwrite each other - the exact failure the
    /// variant name exists to prevent.
    #[error(
        "{}: two schedules use variant '{variant}' - their artifacts would collide",
        .backend.name()
    )]
    DuplicateVariant { backend: Backend, variant: String },
    /// A (kernel, backend) pair with no rule-free schedule. Selection would be
    /// partial: some shape would match no variant, and the dispatcher would
    /// silently fall back to the native kernel instead of saying why.
    #[error(
        "{}: no schedule without a shape rule - an uncovered shape would select nothing",
        .backend.name()
    )]
    NoFallback { backend: Backend },
    /// More than one rule-free schedule: which one wins would be decided by
    /// table order, which is exactly what `priority` exists to replace.
    #[error(
        "{}: two schedules without a shape rule - the fallback must be unique",
        .backend.name()
    )]
    AmbiguousFallback { backend: Backend },
    /// A named variant with no shape rule (it would shadow the fallback
    /// everywhere) or ranked at or below it (it could never be selected).
    #[error("{}: variant '{variant}' {why}",.backend.name())]
    UnselectableVariant {
        backend: Backend,
        variant: String,
        why: &'static str,
    },
    /// A rule naming an axis the kernel does not declare.
    #[error(
        "{}: variant '{variant}' constrains axis '{axis}', which the kernel does not declare",
        .backend.name()
    )]
    UnknownRuleAxis {
        backend: Backend,
        variant: String,
        axis: String,
    },
    /// Two specializations at the same priority: any shape both claim would be
    /// decided by table order.
    #[error(
        "{}: '{}' and '{}' both have priority {priority} - table order would decide a shape claimed by both",
        .backend.name(), .variants.0, .variants.1
    )]
    AmbiguousPriority {
        backend: Backend,
        priority: u8,
        variants: (String, String),
    },
    /// A numeric argument a constructor was handed leaves the schedule
    /// meaningless: a zero block, a zero vector width, a staged tile of depth
    /// zero.
    ///
    /// Invalid numeric arguments must be rejected instead of being normalized
    /// with `max(1)`, which could make the shader's block differ from the
    /// dispatcher's divisor.
    #[error("{}: {why}",.backend.name())]
    Malformed { backend: Backend, why: &'static str },
}

/// Checks what makes a *set* of schedules a well-formed variant table for one
/// kernel: within each backend, identities are distinct, exactly one variant
/// accepts every shape, and every specialization can actually be selected.
///
/// These are the properties per-shape selection rests on. Together they make
/// the dispatcher's rule - "among the variants whose shape rules hold, the
/// highest priority wins" - both **total** (a fallback always matches) and
/// **deterministic** (no tie can be broken by table order). Checked here, at
/// generation, rather than discovered as a variant that silently never runs.
pub fn check_schedule_table(kernel: &Kernel, schedules: &[Schedule]) -> Result<(), ScheduleError> {
    for backend in [Backend::Cpu, Backend::Cuda, Backend::Metal, Backend::Vulkan] {
        let group: Vec<&Schedule> = schedules.iter().filter(|s| s.backend == backend).collect();
        if group.is_empty() {
            continue;
        }

        let mut seen: Vec<Option<&'static str>> = Vec::new();
        for s in &group {
            if seen.contains(&s.variant) {
                return Err(ScheduleError::DuplicateVariant {
                    backend,
                    variant: s.variant.unwrap_or("<fallback>").to_string(),
                });
            }
            seen.push(s.variant);
        }

        // The pair's fallback is the **unnamed** lowering: identity decides,
        // not what it claims. That separation is what lets the fallback carry a
        // layout claim (`vector_width`) while a shape rule on it stays refused
        // below - the two degrade differently. A shape a variant did not claim
        // is arbitrated by the table and must always find a taker; a layout it
        // did not claim is refused with a published `stride` restriction, which hands
        // the node to the native kernel. One is a silent wrong pick, the other
        // is a stated refusal.
        let fallbacks: Vec<&&Schedule> = group.iter().filter(|s| s.variant.is_none()).collect();
        match fallbacks.len() {
            0 => return Err(ScheduleError::NoFallback { backend }),
            1 => {}
            _ => return Err(ScheduleError::AmbiguousFallback { backend }),
        }
        let fallback = fallbacks[0];
        if !fallback.eligible_when.is_empty() {
            return Err(ScheduleError::UnselectableVariant {
                backend,
                variant: "<fallback>".to_string(),
                why: "has a shape rule: an unclaimed shape would select nothing",
            });
        }

        for (s, variant) in group.iter().filter_map(|s| s.variant.map(|v| (s, v))) {
            let name = variant.to_string();
            if !s.claims() {
                return Err(ScheduleError::UnselectableVariant {
                    backend,
                    variant: name,
                    why: "claims no subdomain and would therefore shadow the fallback everywhere",
                });
            }
            if s.priority <= fallback.priority {
                return Err(ScheduleError::UnselectableVariant {
                    backend,
                    variant: name,
                    why: "does not exceed fallback priority and would therefore never be selected",
                });
            }
            for rule in &s.eligible_when {
                // A rule over no axis constrains nothing: the variant would
                // claim every shape and shadow the fallback it outranks.
                if rule.axes.is_empty() {
                    return Err(ScheduleError::UnselectableVariant {
                        backend,
                        variant: name,
                        why: "has an axis-free rule and therefore constrains nothing",
                    });
                }
                // An empty interval can never hold, so the variant is dead code
                // that no shape reaches - the mirror of the case above.
                if rule.min > rule.max {
                    return Err(ScheduleError::UnselectableVariant {
                        backend,
                        variant: name,
                        why: "has an empty interval (min > max) and is therefore never satisfied",
                    });
                }
                for axis in rule.axes {
                    if !kernel.axes().iter().any(|a| a.name == *axis) {
                        return Err(ScheduleError::UnknownRuleAxis {
                            backend,
                            variant: name,
                            axis: (*axis).to_string(),
                        });
                    }
                }
            }
        }

        // Two specializations of equal priority would be separated by table
        // order for any shape both claim, which is exactly what `priority`
        // exists to replace. Refusing the tie outright is stronger than proving
        // their rules disjoint, and it costs nothing: giving one of them a
        // different number is always possible and always says which wins.
        let mut ranks: Vec<(u8, &'static str)> = group
            .iter()
            .filter_map(|s| s.variant.map(|v| (s.priority, v)))
            .collect();
        ranks.sort_unstable();
        for pair in ranks.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(ScheduleError::AmbiguousPriority {
                    backend,
                    priority: pair[0].0,
                    variants: (pair[0].1.to_string(), pair[1].1.to_string()),
                });
            }
        }
    }
    Ok(())
}

/// A schedule incompatible with `ReductionSemantics` is a `rir-gen`
/// **compilation error**, not a downstream test failure.
///
/// `lower` calls this before it reads a single schedule field, and
/// it starts with `check_shape`: the numeric arguments
/// come first because everything after them indexes or divides by one.
///
/// # Errors
///
/// `ScheduleError::Malformed` for a degenerate block, depth or width;
/// `ScheduleError::IncompatibleReduction` for a strategy the declared
/// semantics forbids.
pub fn check_schedule(kernel: &Kernel, schedule: &Schedule) -> Result<(), ScheduleError> {
    schedule.check_shape()?;
    for (i, op) in kernel.ops().iter().enumerate() {
        if let Op::Reduce { semantics, .. } = op {
            let ok = match semantics {
                ReductionSemantics::Associative => true,
                // Exhaustive since `MultiPass` was removed: every strategy the
                // table can name has a fixed,
                // run-to-run topology. Written out rather than `=> true` so a
                // strategy added later has to classify itself here.
                ReductionSemantics::Deterministic => matches!(
                    schedule.reduction,
                    ReductionStrategy::Serial
                        | ReductionStrategy::SharedTree
                        | ReductionStrategy::SubgroupTree
                        | ReductionStrategy::HierarchicalTree
                        | ReductionStrategy::TiledStage
                ),
                // `TiledStage` moves *where the operands are read from*, not
                // the order they are combined in: the reduction axis is still
                // walked ascending, term by term. It is therefore the one
                // collective-looking strategy an exact-order reduction may use.
                ReductionSemantics::ExactOrder => matches!(
                    schedule.reduction,
                    ReductionStrategy::Serial | ReductionStrategy::TiledStage
                ),
            };
            if !ok {
                return Err(ScheduleError::IncompatibleReduction {
                    value: ValueId::at(i),
                    semantics: *semantics,
                    strategy: schedule.reduction,
                });
            }
        }
    }
    Ok(())
}
