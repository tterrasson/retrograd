//! Bounded candidate search for the physical execution choices of a plan.
//!
//! Semantic configuration is already normalized and validated before entering
//! this module. Candidates never change group size, objective, dataset or a
//! locked field; they only choose physical execution geometry.

use std::collections::BTreeSet;

use retrograd_core::{
    CheckpointDtype, ExecutionProfile, LoraConfig, ModelInfo, SharedPrefixFanout, TrainConfig,
    checkpoint_stride_for,
};
use serde::Serialize;

use crate::budget::Budgets;
use crate::cost::{Calibration, MemoryEstimate, ResourceEstimate, Workload, estimate};

pub const MAX_CANDIDATES: usize = 4096;

/// Physical width the runtime pads a packed micro-batch up to. Anything the
/// packing arithmetic produces is rounded to it, so a candidate is costed on the
/// width the device would actually see, not on the token count on paper.
pub const PACKING_ALIGNMENT: u32 = 32;

#[derive(Clone, Debug)]
pub struct NormalizedIntent {
    pub training: TrainConfig,
    pub group_size: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub packing_alignment: u32,
    pub rollout: bool,
    pub locked: BTreeSet<String>,
    pub shared_prefix_capability: Option<bool>,
}

pub fn normalize(
    training: &TrainConfig,
    group_size: u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    rollout: bool,
    locked: impl IntoIterator<Item = String>,
    profile: Option<&ExecutionProfile>,
) -> NormalizedIntent {
    NormalizedIntent {
        training: training.clone(),
        group_size: group_size.max(1),
        prompt_tokens,
        completion_tokens,
        packing_alignment: PACKING_ALIGNMENT,
        rollout,
        locked: locked.into_iter().collect(),
        shared_prefix_capability: profile
            .map(|profile| profile.capabilities.shared_prefix_packed_training),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Candidate {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub shared_prefix_fanout: SharedPrefixFanout,
    pub gradient_checkpointing: bool,
    pub checkpoint_every_n_layers: u32,
    pub checkpoint_dtype: CheckpointDtype,
}

impl Candidate {
    pub fn applied_to(&self, base: &TrainConfig) -> TrainConfig {
        let mut training = base.clone();
        self.apply(&mut training);
        training
    }

    pub fn apply(&self, training: &mut TrainConfig) {
        training.n_ctx = self.n_ctx;
        training.n_batch = self.n_batch;
        training.n_ubatch = self.n_ubatch;
        training.shared_prefix_fanout = self.shared_prefix_fanout;
        training.gradient_checkpointing = self.gradient_checkpointing;
        training.checkpoint_every_n_layers = self.checkpoint_every_n_layers;
        training.checkpoint_dtype = self.checkpoint_dtype;
    }

    pub fn canonical_key(&self) -> String {
        format!(
            "ctx={}/batch={}/ubatch={}/fanout={:?}/checkpoint={}:{}:{:?}",
            self.n_ctx,
            self.n_batch,
            self.n_ubatch,
            self.shared_prefix_fanout,
            self.gradient_checkpointing,
            self.checkpoint_every_n_layers,
            self.checkpoint_dtype
        )
        .to_ascii_lowercase()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EnumerateError {
    #[error("candidate enumeration reached its hard cap of {limit}")]
    LimitReached { limit: usize },
    #[error("a locked shared_prefix_fanout has no candidate to enumerate")]
    LockedFanoutUnsupported,
}

/// Enumerates only divisors, fanout pivots and checkpoint boundaries. The hard
/// cap is an error, never silent truncation.
pub fn enumerate(
    intent: &NormalizedIntent,
    model: &ModelInfo,
) -> Result<Vec<Candidate>, EnumerateError> {
    let micro_locked = intent.locked.contains("training.micro_batch");
    let fanout_locked = intent.locked.contains("training.shared_prefix_fanout");
    let checkpointing_locked = intent.locked.contains("training.gradient_checkpointing");
    let stride_locked = intent.locked.contains("training.checkpoint_every_n_layers");
    let dtype_locked = intent.locked.contains("training.checkpoint_dtype");

    let ubatches = if micro_locked {
        vec![intent.training.n_ubatch]
    } else {
        divisors_desc(intent.training.n_batch)
    };
    let fanouts = fanout_values(intent, fanout_locked)?;
    // An inactive checkpoint stride has no execution meaning. Preserve the
    // caller's value so selecting the "off" candidate cannot manufacture
    // a no-op config/provenance transition (`off` -> `off`).
    let off = (false, intent.training.checkpoint_every_n_layers.max(1));
    let on = (
        true,
        // The two fields lock independently: pinning the stride must not also
        // pin whether checkpointing runs, and vice versa.
        if stride_locked {
            intent.training.checkpoint_every_n_layers.max(1)
        } else {
            checkpoint_stride_for(model.n_layer).max(1)
        },
    );
    let checkpoints = if checkpointing_locked {
        vec![if intent.training.gradient_checkpointing {
            on
        } else {
            off
        }]
    } else {
        vec![off, on]
    };
    let dtypes = if dtype_locked {
        vec![intent.training.checkpoint_dtype]
    } else {
        vec![CheckpointDtype::F32, CheckpointDtype::F16]
    };

    let size = ubatches
        .len()
        .saturating_mul(fanouts.len())
        .saturating_mul(checkpoints.len())
        .saturating_mul(dtypes.len());
    if size > MAX_CANDIDATES {
        return Err(EnumerateError::LimitReached {
            limit: MAX_CANDIDATES,
        });
    }

    let mut out = Vec::with_capacity(size);
    for n_ubatch in ubatches {
        for fanout in &fanouts {
            for &(gradient_checkpointing, checkpoint_every_n_layers) in &checkpoints {
                for &checkpoint_dtype in &dtypes {
                    if !gradient_checkpointing && checkpoint_dtype != CheckpointDtype::F32 {
                        continue;
                    }
                    out.push(Candidate {
                        n_ctx: intent.training.n_ctx,
                        n_batch: intent.training.n_batch,
                        n_ubatch,
                        shared_prefix_fanout: *fanout,
                        gradient_checkpointing,
                        checkpoint_every_n_layers,
                        checkpoint_dtype,
                    });
                }
            }
        }
    }
    out.sort_by_key(Candidate::canonical_key);
    out.dedup();
    Ok(out)
}

/// Largest physical width first, so the widest micro-batch is costed before the
/// ones that only add passes.
fn divisors_desc(value: u32) -> Vec<u32> {
    let mut values = crate::geometry::divisors(value);
    values.reverse();
    values
}

fn fanout_values(
    intent: &NormalizedIntent,
    locked: bool,
) -> Result<Vec<SharedPrefixFanout>, EnumerateError> {
    if !intent.rollout || intent.group_size < 2 {
        return Ok(vec![intent.training.shared_prefix_fanout]);
    }
    if locked {
        let requested = intent.training.shared_prefix_fanout;
        // Refused only against a published `false`. An absent profile is not
        // evidence: a pure analytical caller supplies none, and turning "nobody
        // asked the engine" into an override conflict would reject a
        // configuration the runtime accepts.
        if matches!(
            requested,
            SharedPrefixFanout::Max | SharedPrefixFanout::Exact(_)
        ) && intent.shared_prefix_capability == Some(false)
        {
            return Err(EnumerateError::LockedFanoutUnsupported);
        }
        return Ok(vec![requested]);
    }
    match intent.shared_prefix_capability {
        // The engine published that this graph cannot branch a shared prompt, so
        // packing it is not a choice the planner gets to make.
        Some(false) => return Ok(vec![SharedPrefixFanout::Off]),
        // Nobody asked the engine. Leaving the incoming value - `Auto` unless the
        // caller said otherwise - keeps the decision where it can still be made
        // correctly; writing `off` back would disable packing for a run the
        // runtime would have probed for itself.
        None => return Ok(vec![intent.training.shared_prefix_fanout]),
        Some(true) => {}
    }
    let mut values = vec![SharedPrefixFanout::Off];
    let mut fanout = 2u32;
    while fanout < intent.group_size {
        values.push(SharedPrefixFanout::Exact(fanout));
        let Some(next) = fanout.checked_mul(2) else {
            break;
        };
        fanout = next;
    }
    values.push(SharedPrefixFanout::Exact(intent.group_size));
    Ok(values)
}

#[derive(Clone, Debug)]
pub struct CandidateEvaluation {
    pub candidate: Candidate,
    pub estimate: MemoryEstimate,
    pub resources: ResourceEstimate,
    pub valid: bool,
    pub rejection: Option<&'static str>,
    pub physical_tokens: u64,
    pub physical_passes: u32,
    pub execution_cost: u64,
    pub fidelity: u8,
    pub device_margin_bytes: u64,
    /// Floor the physical micro-batch may not cross once this candidate is
    /// selected: the rescue levers run after the search and would otherwise undo
    /// the packing it just validated.
    pub min_ubatch: u32,
}

pub fn evaluate(
    candidate: Candidate,
    intent: &NormalizedIntent,
    model: &ModelInfo,
    lora: &LoraConfig,
    workload: &Workload,
    budgets: &Budgets,
    calibration: Calibration,
) -> CandidateEvaluation {
    let mut training = intent.training.clone();
    candidate.apply(&mut training);
    let estimate = estimate(model, &training, lora, workload, calibration);
    let resources = estimate.resources();
    let packing = packing_cost(&candidate, intent);
    let Packing {
        physical_tokens,
        passes: physical_passes,
        valid: packing_valid,
        min_ubatch,
    } = packing;
    let overflow = budgets.overflow(resources.device_peak_bytes, resources.host_peak_bytes);
    let valid = overflow.fits() && packing_valid;
    let rejection = if !packing_valid {
        Some("fanout_does_not_fit")
    } else if !overflow.fits() {
        Some("budget")
    } else {
        None
    };
    let recompute = if candidate.gradient_checkpointing {
        candidate.n_ubatch as u64
    } else {
        0
    };
    // Only work goes in here. The checkpoint dtype costs fidelity, not time, and
    // `fidelity` below is already the criterion that ranks it - charging it twice
    // would let a narrow dtype lose on a scale it does not belong to.
    let execution_cost = physical_tokens
        .saturating_mul(physical_passes as u64)
        .saturating_add(
            (candidate.n_batch / candidate.n_ubatch.max(1)) as u64 * candidate.n_ctx as u64,
        )
        .saturating_add(recompute);
    let fidelity = match (candidate.gradient_checkpointing, candidate.checkpoint_dtype) {
        (false, _) => 3,
        (true, CheckpointDtype::F32) => 2,
        (true, _) => 1,
    };
    let device_margin_bytes = budgets
        .vram
        .effective_bytes
        .saturating_sub(resources.device_peak_bytes);
    CandidateEvaluation {
        candidate,
        estimate,
        resources,
        valid,
        rejection,
        physical_tokens,
        physical_passes,
        execution_cost,
        fidelity,
        device_margin_bytes,
        min_ubatch,
    }
}

/// What one candidate's packing costs: the physical token width of a subgroup,
/// how many passes cover the logical group, whether that width fits the
/// micro-batch, and the floor the micro-batch may not go below afterwards.
struct Packing {
    physical_tokens: u64,
    passes: u32,
    valid: bool,
    min_ubatch: u32,
}

fn packing_cost(candidate: &Candidate, intent: &NormalizedIntent) -> Packing {
    if !intent.rollout {
        return Packing {
            physical_tokens: candidate.n_ubatch as u64,
            passes: 1,
            valid: true,
            min_ubatch: 1,
        };
    }
    let fanout = match candidate.shared_prefix_fanout {
        SharedPrefixFanout::Off => 1,
        // `auto` is a deferral, not a request: the runtime packs only if the
        // graph can, so costing it as a full group when nobody published that
        // capability would invalidate every candidate over a width the run is
        // not going to use.
        SharedPrefixFanout::Auto => {
            if intent.shared_prefix_capability == Some(true) {
                intent.group_size
            } else {
                1
            }
        }
        SharedPrefixFanout::Max => intent.group_size,
        SharedPrefixFanout::Exact(value) => value.min(intent.group_size),
    }
    .max(1);
    let raw = intent.prompt_tokens as u64 + fanout as u64 * (intent.completion_tokens as u64 + 1);
    let alignment = intent.packing_alignment.max(1) as u64;
    let physical = raw.div_ceil(alignment) * alignment;
    let passes = intent.group_size.div_ceil(fanout);
    // Only a real subgroup imposes a width. `off` packs nothing, so the
    // micro-batch stays a free memory lever below it.
    let packed = fanout >= 2;
    Packing {
        physical_tokens: physical,
        passes,
        valid: !packed || physical <= candidate.n_ubatch as u64,
        min_ubatch: if packed {
            physical.min(u32::MAX as u64) as u32
        } else {
            1
        },
    }
}

/// Removes candidates weakly worse in both resource peaks and execution cost.
pub fn eliminate_dominated(mut candidates: Vec<CandidateEvaluation>) -> Vec<CandidateEvaluation> {
    // Cached: the key is a formatted string, and a plain `sort_by_key` would
    // rebuild it on every comparison.
    candidates.sort_by_cached_key(|candidate| candidate.candidate.canonical_key());
    let keep = (0..candidates.len())
        .map(|i| {
            !(0..candidates.len()).any(|j| {
                if i == j || candidates[i].valid != candidates[j].valid {
                    return false;
                }
                let left = &candidates[j];
                let right = &candidates[i];
                left.resources.device_peak_bytes <= right.resources.device_peak_bytes
                    && left.resources.host_peak_bytes <= right.resources.host_peak_bytes
                    && left.execution_cost <= right.execution_cost
                    && left.fidelity >= right.fidelity
                    && (left.resources.device_peak_bytes < right.resources.device_peak_bytes
                        || left.resources.host_peak_bytes < right.resources.host_peak_bytes
                        || left.execution_cost < right.execution_cost
                        || left.fidelity > right.fidelity)
            })
        })
        .collect::<Vec<_>>();
    candidates
        .into_iter()
        .zip(keep)
        .filter_map(|(candidate, keep)| keep.then_some(candidate))
        .collect()
}

/// Lexicographic choice: valid, fidelity, predicted cost, memory margin, then a
/// canonical key. No opaque weighted sum can trade semantics against speed.
pub fn select(candidates: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    candidates
        .iter()
        .filter(|candidate| candidate.valid)
        .min_by(|a, b| {
            b.fidelity
                .cmp(&a.fidelity)
                .then(a.execution_cost.cmp(&b.execution_cost))
                .then(b.device_margin_bytes.cmp(&a.device_margin_bytes))
                // Last resort only, and lazily: the key allocates, and everything
                // above it already separates all but exact ties.
                .then_with(|| {
                    a.candidate
                        .canonical_key()
                        .cmp(&b.candidate.canonical_key())
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{execution_profile, sft_workload, tiny_model};

    #[test]
    fn a_locked_fanout_is_refused_against_a_published_no() {
        let training = TrainConfig {
            shared_prefix_fanout: SharedPrefixFanout::Exact(4),
            ..Default::default()
        };
        let locked = || ["training.shared_prefix_fanout".to_string()];
        let unsupported = execution_profile(false);
        let intent = normalize(&training, 8, 64, 64, true, locked(), Some(&unsupported));
        assert_eq!(
            enumerate(&intent, &tiny_model()),
            Err(EnumerateError::LockedFanoutUnsupported)
        );

        // No profile is not a published "no": an analytical caller supplies
        // none, and the runtime is still the one that decides.
        let intent = normalize(&training, 8, 64, 64, true, locked(), None);
        assert_eq!(
            enumerate(&intent, &tiny_model())
                .expect("an absent profile is not a refusal")
                .iter()
                .filter(|candidate| {
                    candidate.shared_prefix_fanout != SharedPrefixFanout::Exact(4)
                })
                .count(),
            0,
            "a locked fanout stays the only one enumerated"
        );
    }

    #[test]
    fn dominance_keeps_the_pareto_frontier() {
        let model = tiny_model();
        let training = TrainConfig {
            n_ctx: 128,
            n_batch: 128,
            n_ubatch: 128,
            ..Default::default()
        };
        let intent = normalize(&training, 1, 0, 0, false, [], None);
        let candidates = enumerate(&intent, &model).unwrap();
        let budgets = Budgets::resolve(
            crate::budget::MemoryBaseline {
                device_total: Some(u64::MAX),
                host_total: Some(u64::MAX),
                ..Default::default()
            },
            (
                crate::budget::BudgetRequest::All,
                crate::budget::BudgetRequest::All,
            ),
            (None, None),
            crate::budget::MarginPolicy {
                fraction: 0.0,
                floor_bytes: 0,
            },
        );
        let evaluated = candidates
            .into_iter()
            .map(|candidate| {
                evaluate(
                    candidate,
                    &intent,
                    &model,
                    &LoraConfig::auto(2, 4.0),
                    &sft_workload(),
                    &budgets,
                    Calibration::default(),
                )
            })
            .collect();
        let frontier = eliminate_dominated(evaluated);
        assert!(!frontier.is_empty());
        assert!(select(&frontier).is_some());
    }
}
