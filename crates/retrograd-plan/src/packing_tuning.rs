//! Request-local timing evidence for packed optimizer geometry. Never compare
//! wall seconds with analytical work units, or use a row update as a packed
//! update benchmark: they need not have the same optimizer-step boundaries.

use std::collections::BTreeMap;

use retrograd_core::SharedPrefixFanout;

use crate::Budgets;
use crate::candidate::{Candidate, CandidateEvaluation, NormalizedIntent, select};

pub const MAX_PACKING_PROBES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingShape {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub group_size: u32,
    pub n_seq_max: u32,
}

impl PackingShape {
    pub fn from_intent(intent: &NormalizedIntent) -> Self {
        Self {
            prompt_tokens: intent.prompt_tokens,
            completion_tokens: intent.completion_tokens,
            group_size: intent.group_size,
            n_seq_max: intent.training.n_seq_max,
        }
    }

    pub fn key(self, candidate: &Candidate) -> String {
        format!(
            "{}/prompt={}/completion={}/group={}/seq={}",
            candidate.canonical_key(),
            self.prompt_tokens,
            self.completion_tokens,
            self.group_size,
            self.n_seq_max
        )
    }

    fn usable(self) -> bool {
        self.prompt_tokens >= 2 && self.completion_tokens > 0 && self.group_size >= 2
    }
}

#[derive(Clone, Debug)]
pub struct PackingProbe {
    pub candidate: Candidate,
    pub shape: PackingShape,
}

impl PackingProbe {
    pub fn key(&self) -> String {
        self.shape.key(&self.candidate)
    }
}

/// One warm-up is excluded; at least three complete logical updates are timed.
/// Memory is attributable runtime allocation, not device-wide usage, which
/// would count other processes a second time against the available budget.
/// `failure` is a refusal of this geometry (allocation, non-finite loss, row
/// fallback); an environment failure is an error of the probe, not a sample.
#[derive(Clone, Debug)]
pub struct PackingMeasurement {
    pub seconds: Vec<f64>,
    pub device_bytes: u64,
    pub failure: Option<String>,
}

impl PackingMeasurement {
    pub fn interval(&self) -> Option<(f64, f64)> {
        if self.failure.is_some()
            || self.seconds.len() < 3
            || self.seconds.iter().any(|s| !s.is_finite() || *s <= 0.0)
        {
            return None;
        }
        Some(
            self.seconds
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &s| {
                    (lo.min(s), hi.max(s))
                }),
        )
    }

    fn clearly_faster_than(&self, other: &Self) -> bool {
        match (self.interval(), other.interval()) {
            // Even the slowest challenger must beat the fastest incumbent by
            // 5%. Overlapping timing noise never decides a promotion.
            (Some((_, hi)), Some((lo, _))) => hi < lo * 0.95,
            _ => false,
        }
    }
}

pub type PackingMeasurements = BTreeMap<String, PackingMeasurement>;

pub fn apply_measurements(
    candidates: &mut [CandidateEvaluation],
    shape: PackingShape,
    measurements: &PackingMeasurements,
    budgets: &Budgets,
) {
    for candidate in candidates {
        if let Some(sample) = measurements.get(&shape.key(&candidate.candidate)) {
            // The host side keeps its estimate: the probe measures the device.
            // `overflow` sums both on unified memory, as the final check does.
            let overflow =
                budgets.overflow(sample.device_bytes, candidate.resources.host_peak_bytes);
            let rejection = if sample.failure.is_some() {
                Some("packing_benchmark_failed")
            } else if !overflow.fits() {
                Some("measured_packing_budget")
            } else {
                None
            };
            if let Some(reason) = rejection {
                candidate.valid = false;
                candidate.rejection = Some(reason);
            }
        }
    }
}

fn packed(candidate: &Candidate) -> bool {
    matches!(
        candidate.shared_prefix_fanout,
        SharedPrefixFanout::Auto | SharedPrefixFanout::Max | SharedPrefixFanout::Exact(2..)
    )
}

/// Keep the analytical incumbent unless a comparable measured candidate wins
/// outside the observed noise. Memory failures are applied before this call.
pub fn select_measured<'a>(
    candidates: &'a [CandidateEvaluation],
    shape: PackingShape,
    measurements: &PackingMeasurements,
) -> Option<&'a CandidateEvaluation> {
    let incumbent = select(candidates)?;
    if !packed(&incumbent.candidate) {
        return Some(incumbent);
    }
    let Some(baseline) = measurements.get(&shape.key(&incumbent.candidate)) else {
        return Some(incumbent);
    };
    candidates
        .iter()
        .filter(|candidate| {
            candidate.valid
                && candidate.fidelity == incumbent.fidelity
                && packed(&candidate.candidate)
        })
        .filter_map(|candidate| {
            let sample = measurements.get(&shape.key(&candidate.candidate))?;
            if !sample.clearly_faster_than(baseline) {
                return None;
            }
            // `clearly_faster_than` only holds for a valid interval.
            let (_, upper) = sample.interval()?;
            Some((candidate, upper))
        })
        .min_by(|(a, a_upper), (b, b_upper)| {
            a_upper.total_cmp(b_upper).then_with(|| {
                a.candidate
                    .canonical_key()
                    .cmp(&b.candidate.canonical_key())
            })
        })
        .map(|(candidate, _)| candidate)
        .or(Some(incumbent))
}

/// Benchmark only fitting candidates at the incumbent's fidelity. Include
/// different physical widths before spending the remaining slots on fanouts.
/// Use the full candidate set: an analytical Pareto filter must not discard a
/// geometry before the measurements that could disprove that prediction.
pub fn shortlist(candidates: &[CandidateEvaluation], shape: PackingShape) -> Vec<PackingProbe> {
    let Some(best) = select(candidates) else {
        return Vec::new();
    };
    if !shape.usable() || !packed(&best.candidate) {
        return Vec::new();
    }
    let mut ranked: Vec<_> = candidates
        .iter()
        .filter(|c| {
            c.valid
                && c.fidelity == best.fidelity
                && packed(&c.candidate)
                && shape.prompt_tokens.saturating_add(shape.completion_tokens) <= c.candidate.n_ctx
        })
        .collect();
    ranked.sort_by_key(|c| (c.execution_cost, c.candidate.canonical_key()));
    let mut selected = vec![best.candidate.clone()];
    for different_width in [true, false] {
        for c in &ranked {
            if selected.len() == MAX_PACKING_PROBES {
                break;
            }
            if selected.contains(&c.candidate)
                || (different_width && selected.iter().any(|s| s.n_ubatch == c.candidate.n_ubatch))
            {
                continue;
            }
            selected.push(c.candidate.clone());
        }
    }
    if selected.len() < 2 {
        return Vec::new();
    }
    selected
        .into_iter()
        .map(|candidate| PackingProbe { candidate, shape })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{enumerate, evaluate, normalize};
    use crate::testing::{execution_profile, sft_workload, tiny_model};
    use crate::{BudgetRequest, Budgets, Calibration, MarginPolicy, MemoryBaseline};
    use retrograd_core::TrainConfig;

    fn budgets(device: u64, host: u64, unified: bool) -> Budgets {
        Budgets::resolve(
            MemoryBaseline {
                device_total: Some(device),
                host_total: Some(host),
                unified,
                ..Default::default()
            },
            (BudgetRequest::All, BudgetRequest::All),
            (None, None),
            MarginPolicy {
                fraction: 0.0,
                floor_bytes: 0,
            },
        )
    }

    fn fixture() -> (Vec<CandidateEvaluation>, PackingShape) {
        let model = tiny_model();
        let training = TrainConfig {
            n_ctx: 128,
            n_batch: 128,
            n_ubatch: 128,
            n_seq_max: 4,
            ..Default::default()
        };
        let intent = normalize(&training, 4, 8, 4, true, [], Some(&execution_profile(true)));
        let budgets = budgets(u64::MAX, u64::MAX, false);
        let values = enumerate(&intent, &model)
            .unwrap()
            .into_iter()
            .filter(|c| {
                !c.gradient_checkpointing
                    && c.shared_prefix_fanout == SharedPrefixFanout::Exact(4)
                    && c.n_ubatch >= 32
            })
            .map(|c| {
                evaluate(
                    c,
                    &intent,
                    &model,
                    crate::cost::Trainable::adapter(&retrograd_core::LoraConfig::auto(2, 4.0)),
                    &sft_workload(),
                    &budgets,
                    Calibration::default(),
                )
            })
            .collect();
        (values, PackingShape::from_intent(&intent))
    }
    fn record(
        map: &mut PackingMeasurements,
        c: &CandidateEvaluation,
        shape: PackingShape,
        times: &[f64],
        bytes: u64,
    ) {
        map.insert(
            shape.key(&c.candidate),
            PackingMeasurement {
                seconds: times.to_vec(),
                device_bytes: bytes,
                failure: None,
            },
        );
    }
    #[test]
    fn measured_speed_can_overturn_an_analytically_dominated_geometry() {
        let (candidates, shape) = fixture();
        let baseline = select(&candidates).unwrap();
        assert_eq!(baseline.candidate.n_ubatch, 128);
        let faster = candidates
            .iter()
            .find(|c| c.candidate.n_ubatch == 64)
            .unwrap();
        let mut evidence = PackingMeasurements::new();
        record(&mut evidence, baseline, shape, &[10.0, 10.1, 9.9], 100);
        record(&mut evidence, faster, shape, &[7.9, 8.0, 8.1], 100);
        assert_eq!(
            select_measured(&candidates, shape, &evidence)
                .unwrap()
                .candidate
                .n_ubatch,
            64
        );
        let probes = shortlist(&candidates, shape);
        assert!(probes.len() <= MAX_PACKING_PROBES);
        assert_eq!(probes[0].candidate, baseline.candidate);
        assert!(probes.iter().any(|p| p.candidate == faster.candidate));
    }
    #[test]
    fn noise_missing_baseline_and_changed_workload_never_promote() {
        let (candidates, shape) = fixture();
        let baseline = select(&candidates).unwrap();
        let challenger = candidates
            .iter()
            .find(|c| c.candidate.n_ubatch == 64)
            .unwrap();
        let mut evidence = PackingMeasurements::new();
        record(&mut evidence, challenger, shape, &[1.0; 3], 100);
        assert_eq!(
            select_measured(&candidates, shape, &evidence)
                .unwrap()
                .candidate,
            baseline.candidate
        );
        record(&mut evidence, baseline, shape, &[10.0; 3], 100);
        for samples in [
            &[9.6, 9.6, 9.6][..],
            &[8.0, 11.0, 8.0],
            &[f64::NAN; 3],
            &[0.0; 3],
            &[1.0, 1.0],
        ] {
            record(&mut evidence, challenger, shape, samples, 100);
            assert_eq!(
                select_measured(&candidates, shape, &evidence)
                    .unwrap()
                    .candidate,
                baseline.candidate
            );
        }
        record(&mut evidence, challenger, shape, &[1.0; 3], 100);
        assert_eq!(
            select_measured(
                &candidates,
                PackingShape {
                    completion_tokens: 5,
                    ..shape
                },
                &evidence
            )
            .unwrap()
            .candidate,
            baseline.candidate
        );
    }
    #[test]
    fn memory_failure_and_lower_fidelity_outrank_speed() {
        let (mut candidates, shape) = fixture();
        let baseline = select(&candidates).unwrap().candidate.clone();
        let challenger = candidates
            .iter()
            .find(|c| c.candidate.n_ubatch == 64)
            .unwrap();
        let mut evidence = PackingMeasurements::new();
        record(
            &mut evidence,
            candidates.iter().find(|c| c.candidate == baseline).unwrap(),
            shape,
            &[10.0; 3],
            100,
        );
        record(&mut evidence, challenger, shape, &[1.0; 3], 201);
        apply_measurements(
            &mut candidates,
            shape,
            &evidence,
            &budgets(200, u64::MAX, false),
        );
        assert_eq!(
            select_measured(&candidates, shape, &evidence)
                .unwrap()
                .candidate,
            baseline
        );
        let challenger = candidates
            .iter_mut()
            .find(|c| c.candidate.n_ubatch == 64)
            .unwrap();
        assert_eq!(challenger.rejection, Some("measured_packing_budget"));
        challenger.valid = true;
        challenger.fidelity = 1;
        assert_eq!(
            select_measured(&candidates, shape, &evidence)
                .unwrap()
                .candidate,
            baseline
        );
        evidence.get_mut(&shape.key(&baseline)).unwrap().failure = Some("non-finite loss".into());
        apply_measurements(
            &mut candidates,
            shape,
            &evidence,
            &budgets(u64::MAX, u64::MAX, false),
        );
        assert!(
            !candidates
                .iter()
                .find(|c| c.candidate == baseline)
                .unwrap()
                .valid
        );
    }
    #[test]
    fn unified_memory_charges_the_host_estimate_against_the_same_pool() {
        let (mut candidates, shape) = fixture();
        let challenger = candidates
            .iter()
            .find(|c| c.candidate.n_ubatch == 64)
            .unwrap();
        let host = challenger.resources.host_peak_bytes;
        assert!(host > 50);
        let total = host.max(100) + 50;
        let mut evidence = PackingMeasurements::new();
        record(&mut evidence, challenger, shape, &[1.0; 3], 100);
        // Split pools: each side fits `total` on its own.
        apply_measurements(
            &mut candidates,
            shape,
            &evidence,
            &budgets(total, total, false),
        );
        let find = |c: &[CandidateEvaluation]| {
            c.iter()
                .find(|c| c.candidate.n_ubatch == 64)
                .unwrap()
                .rejection
        };
        assert_eq!(find(&candidates), None);
        // One pool: `host + 100` does not.
        apply_measurements(
            &mut candidates,
            shape,
            &evidence,
            &budgets(total, total, true),
        );
        assert_eq!(find(&candidates), Some("measured_packing_budget"));
    }
    #[test]
    fn a_single_locked_geometry_is_not_benchmarked() {
        let (candidates, shape) = fixture();
        assert!(shortlist(&candidates[..1], shape).is_empty());
    }
}
