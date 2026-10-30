//! Guardrails on the judge's verdicts: their agreement with the verifiable
//! reward, and the alert when that agreement drifts.
//!
//! A run that blends a judge into the reward command's score has both halves
//! per completion: the original reward-command output and the verdict.
//! The Spearman rank correlation between the two, computed *within each
//! group*, says whether the judge still ranks what can be checked - the early
//! signal of a drift to analyse (not by itself a proof of reward hacking,
//! which is why the alert logs instead of stopping the run).
//!
//! A group whose verdicts are all equal, whose verifiable rewards are all
//! equal, or that has fewer than two judged members has an *undefined*
//! correlation. Those groups are counted separately - replacing them by zero
//! would drag the mean down and cry wolf on a task the reward cannot separate.

use std::collections::VecDeque;

use retrograd_spec::judge::JudgeGuardrails;

use crate::rollout::JudgeVerdict;

/// Per-update statistics over the batch's groups.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VerdictStats {
    /// Groups the batch holds (`verdicts.len() / group_size`).
    pub(crate) total_groups: usize,
    /// Groups with at least two judged members.
    pub(crate) judged_groups: usize,
    /// Groups whose judged verdicts are all equal (within `f32::EPSILON`):
    /// a zero-advantage group for the judge half, whatever the value.
    pub(crate) constant_verdict_groups: usize,
    /// Groups whose verdict/verifiable correlation is defined.
    pub(crate) defined_groups: usize,
    /// Groups whose correlation is undefined (constant on either side, fewer
    /// than two judged members, or no verdict at all).
    pub(crate) undefined_groups: usize,
    /// Sum of the defined per-group Spearman correlations; the alert window
    /// pools sums and counts, not means, so sliding it stays exact.
    pub(crate) spearman_sum: f64,
    /// Mean of the defined per-group correlations; 0.0 when none is defined
    /// (read together with `undefined_groups`, which says so).
    pub(crate) spearman_mean: f32,
    /// Mean within-group population variance of the verdicts, over the
    /// groups with at least two judged members.
    pub(crate) verdict_variance_mean: f32,
}

/// The statistics of one update's batch, from the aligned original
/// reward-command outputs and raw verdicts.
pub(crate) fn verdict_stats(
    verifiable_rewards: &[f32],
    verdicts: &[Option<JudgeVerdict>],
    group_size: usize,
) -> VerdictStats {
    debug_assert_eq!(verifiable_rewards.len(), verdicts.len());
    let mut stats = VerdictStats {
        total_groups: verdicts.len().checked_div(group_size).unwrap_or(0),
        ..VerdictStats::default()
    };
    if group_size == 0 || verdicts.is_empty() {
        return stats;
    }
    let mut variance_sum = 0.0_f64;
    for start in (0..verdicts.len()).step_by(group_size) {
        let end = (start + group_size).min(verdicts.len());
        let judged: Vec<usize> = (start..end)
            .filter(|&index| verdicts[index].is_some())
            .collect();
        if judged.len() < 2 {
            stats.undefined_groups += 1;
            continue;
        }
        stats.judged_groups += 1;
        let values: Vec<f32> = judged
            .iter()
            .map(|&index| verdicts[index].as_ref().expect("judged").value)
            .collect();
        let first = values[0];
        if values
            .iter()
            .all(|&value| (value - first).abs() <= f32::EPSILON)
        {
            stats.constant_verdict_groups += 1;
        }
        let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64;
        let sum_sq: f64 = values
            .iter()
            .map(|&value| {
                let delta = f64::from(value) - mean;
                delta * delta
            })
            .sum();
        // Population variance of this group's verdicts.
        variance_sum += sum_sq / values.len() as f64;
        // Original values: undoing the f32 blend can invent or erase ties.
        let verifiable: Vec<f32> = judged
            .iter()
            .map(|&index| verifiable_rewards[index])
            .collect();
        match spearman(&values, &verifiable) {
            Some(rho) => {
                stats.defined_groups += 1;
                stats.spearman_sum += rho;
            }
            None => stats.undefined_groups += 1,
        }
    }
    if stats.defined_groups > 0 {
        stats.spearman_mean = (stats.spearman_sum / stats.defined_groups as f64) as f32;
    }
    if stats.judged_groups > 0 {
        stats.verdict_variance_mean = (variance_sum / stats.judged_groups as f64) as f32;
    }
    stats
}

/// Spearman rank correlation (average ranks for ties, then Pearson on the
/// ranks). `None` when either side is constant or there are fewer than two
/// points: the correlation is undefined there, not zero.
fn spearman(xs: &[f32], ys: &[f32]) -> Option<f64> {
    if xs.len() < 2 || xs.len() != ys.len() {
        return None;
    }
    if xs.iter().all(|&value| value == xs[0]) || ys.iter().all(|&value| value == ys[0]) {
        return None;
    }
    pearson(&ranks(xs), &ranks(ys))
}

/// Average ranks (1-based) of `values`, ties sharing the mean of their span.
fn ranks(values: &[f32]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0_f64; values.len()];
    let mut start = 0_usize;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end) as f64 / 2.0 + 1.0;
        for &index in &order[start..=end] {
            ranks[index] = rank;
        }
        start = end + 1;
    }
    ranks
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    for index in 0..n {
        let dx = xs[index] - mean_x;
        let dy = ys[index] - mean_y;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// The sliding alert window over updates. Each update pushes its
/// `(defined groups, spearman sum)`; the window pools them so a group counts
/// once however many updates it survives in view.
pub(crate) struct GuardrailWindow {
    threshold: f32,
    window_updates: usize,
    min_defined_groups: usize,
    entries: VecDeque<(usize, f64)>,
    defined: usize,
    spearman_sum: f64,
}

impl GuardrailWindow {
    pub(crate) fn new(guardrails: JudgeGuardrails) -> Self {
        Self {
            threshold: guardrails.verdict_correlation_threshold,
            window_updates: guardrails.window_updates as usize,
            min_defined_groups: guardrails.min_defined_groups,
            entries: VecDeque::new(),
            defined: 0,
            spearman_sum: 0.0,
        }
    }

    /// Pushes one update's stats. Returns the window mean when the window
    /// holds enough defined groups and the mean is below the threshold - the
    /// alert condition - and `None` otherwise (including "not enough data",
    /// which is not an alert).
    pub(crate) fn push(&mut self, stats: &VerdictStats) -> Option<f64> {
        self.entries
            .push_back((stats.defined_groups, stats.spearman_sum));
        self.defined += stats.defined_groups;
        self.spearman_sum += stats.spearman_sum;
        while self.entries.len() > self.window_updates {
            let (defined, sum) = self.entries.pop_front().expect("non-empty");
            self.defined -= defined;
            self.spearman_sum -= sum;
        }
        self.alert()
    }

    /// The window's mean over its defined groups, alert or not.
    pub(crate) fn window_mean(&self) -> f32 {
        if self.defined == 0 {
            0.0
        } else {
            (self.spearman_sum / self.defined as f64) as f32
        }
    }

    /// The configured alert threshold.
    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }

    /// How many updates the window spans.
    pub(crate) fn window_updates(&self) -> usize {
        self.window_updates
    }

    /// How many defined per-group correlations the window currently holds.
    pub(crate) fn defined_groups(&self) -> usize {
        self.defined
    }

    fn alert(&self) -> Option<f64> {
        if self.defined < self.min_defined_groups {
            return None;
        }
        let mean = self.spearman_sum / self.defined as f64;
        if mean < f64::from(self.threshold) {
            Some(mean)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(value: f32) -> Option<JudgeVerdict> {
        Some(JudgeVerdict {
            value,
            explanation: None,
        })
    }

    #[test]
    fn spearman_is_one_for_a_monotone_agreement_and_undefined_on_a_constant() {
        // Perfect agreement ranks.
        assert!((spearman(&[0.1, 0.4, 0.9], &[0.0, 1.0, 2.0]).unwrap() - 1.0).abs() < 1e-12);
        // Perfect disagreement.
        assert!((spearman(&[0.1, 0.4, 0.9], &[2.0, 1.0, 0.0]).unwrap() + 1.0).abs() < 1e-12);
        // Ties get average ranks: ranks [1.5, 1.5, 3] vs [1, 2, 3] -> 1.5/sqrt(3).
        assert!((spearman(&[1.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap() - 0.8660).abs() < 1e-3);
        // A constant side is undefined, not zero.
        assert_eq!(spearman(&[0.5, 0.5, 0.5], &[0.0, 1.0, 2.0]), None);
        assert_eq!(spearman(&[0.1], &[0.2]), None);
    }

    #[test]
    fn stats_split_defined_and_undefined_groups() {
        // Two groups of three. Group 0: verdicts track the verifiable reward
        // perfectly. Group 1: constant verdicts, undefined correlation.
        let verifiable = [0.5, 0.3, 0.1, 0.5, 0.5, 0.5];
        let verdicts = [
            verdict(0.9),
            verdict(0.5),
            verdict(0.1),
            verdict(0.5),
            verdict(0.5),
            verdict(0.5),
        ];
        let stats = verdict_stats(&verifiable, &verdicts, 3);
        assert_eq!(stats.total_groups, 2);
        assert_eq!(stats.judged_groups, 2);
        assert_eq!(stats.defined_groups, 1);
        assert_eq!(stats.undefined_groups, 1);
        assert_eq!(stats.constant_verdict_groups, 1);
        assert!((stats.spearman_mean - 1.0).abs() < 1e-6);
        // Group 0's verdict variance: ((0.9-.5)^2 + 0 + (0.1-.5)^2)/3.
        let expected = (0.16 + 0.16) / 3.0 / 2.0 + 0.0 / 2.0;
        assert!((f64::from(stats.verdict_variance_mean) - expected).abs() < 1e-6);
    }

    #[test]
    fn a_group_without_verdicts_is_undefined_not_zero() {
        let stats = verdict_stats(&[1.0, 0.0], &[None, None], 2);
        assert_eq!(stats.defined_groups, 0);
        assert_eq!(stats.undefined_groups, 1);
        assert_eq!(stats.spearman_mean, 0.0);
        assert_eq!(stats.spearman_sum, 0.0);
    }

    #[test]
    fn constant_verifiable_rewards_stay_undefined_despite_blend_rounding() {
        let verifiable = [0.1_f32; 3];
        let values = [0.1, 0.5, 0.9];
        let reconstructed = values.map(|value| (0.1_f32 + value) - value);
        assert_ne!(reconstructed[0], reconstructed[1]);
        let stats = verdict_stats(&verifiable, &values.map(verdict), 3);
        assert_eq!(stats.defined_groups, 0);
        assert_eq!(stats.undefined_groups, 1);
        assert_eq!(stats.constant_verdict_groups, 0);
    }

    #[test]
    fn the_window_alerts_only_with_enough_defined_groups() {
        let guardrails = JudgeGuardrails {
            verdict_correlation_threshold: 0.0,
            window_updates: 2,
            min_defined_groups: 3,
        };
        let mut window = GuardrailWindow::new(guardrails);
        // One defined group per update, negative correlation: below the
        // threshold, but the window never reaches min_defined_groups.
        let negative = VerdictStats {
            defined_groups: 1,
            spearman_sum: -1.0,
            ..VerdictStats::default()
        };
        assert_eq!(window.push(&negative), None);
        assert_eq!(window.push(&negative), None);
        assert_eq!(window.defined, 2);
        // A third update (window slides to the last two) still under the
        // minimum... with window_updates = 2 the window holds 2 groups.
        let full = VerdictStats {
            defined_groups: 2,
            spearman_sum: -2.0,
            ..VerdictStats::default()
        };
        let alert = window.push(&full);
        assert_eq!(window.defined, 3);
        assert!(alert.is_some(), "3 defined groups below threshold alert");
        assert!((alert.unwrap() + 1.0).abs() < 1e-12);
        // A positive mean clears the alert. The window holds the last two
        // updates: (2, -2) and (3, +3) -> mean 0.2 over 5 defined groups.
        let good = VerdictStats {
            defined_groups: 3,
            spearman_sum: 3.0,
            ..VerdictStats::default()
        };
        assert_eq!(window.push(&good), None);
        assert!((window.window_mean() - 0.2).abs() < 1e-6);
    }
}
