//! The numbers that justify the pool, or contradict it.
//!
//! They are part of the deliverable rather than an afterthought: "reuse is
//! faster" and "backpressure holds" are claims, and `env/pool_reuse_fraction`
//! and `env/live_containers_max` are what turn them into facts.

use std::sync::atomic::{AtomicU64, Ordering};

use retrograd_metrics::MetricValue;

/// Atomic counters behind [`SandboxPool::metric_values`](crate::SandboxPool::metric_values);
/// see the module docs for why each one exists.
#[derive(Debug, Default)]
pub struct SandboxMetrics {
    create_count: AtomicU64,
    create_micros: AtomicU64,
    acquire_count: AtomicU64,
    acquire_micros: AtomicU64,
    reuse_count: AtomicU64,
    live_max: AtomicU64,
    exec_count: AtomicU64,
    timeout_count: AtomicU64,
    broken_count: AtomicU64,
}

impl SandboxMetrics {
    pub(crate) fn record_create(&self, micros: u64) {
        self.create_count.fetch_add(1, Ordering::Relaxed);
        self.create_micros.fetch_add(micros, Ordering::Relaxed);
    }

    pub(crate) fn record_acquire(&self, micros: u64, reused: bool) {
        self.acquire_count.fetch_add(1, Ordering::Relaxed);
        self.acquire_micros.fetch_add(micros, Ordering::Relaxed);
        if reused {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_live(&self, live: usize) {
        self.live_max.fetch_max(live as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_exec(&self) {
        self.exec_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_timeout(&self) {
        self.timeout_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broken(&self) {
        self.broken_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshots the counters as the metric lines a training run logs.
    pub fn metric_values(&self) -> Vec<MetricValue> {
        let mean = |total: &AtomicU64, count: &AtomicU64| {
            let count = count.load(Ordering::Relaxed);
            if count == 0 {
                0.0
            } else {
                total.load(Ordering::Relaxed) as f32 / count as f32 / 1000.0
            }
        };
        let fraction = |part: &AtomicU64, whole: &AtomicU64| {
            let whole = whole.load(Ordering::Relaxed);
            if whole == 0 {
                0.0
            } else {
                part.load(Ordering::Relaxed) as f32 / whole as f32
            }
        };
        vec![
            MetricValue {
                name: "env/container_create_ms".into(),
                value: mean(&self.create_micros, &self.create_count),
            },
            MetricValue {
                name: "env/acquire_ms_mean".into(),
                value: mean(&self.acquire_micros, &self.acquire_count),
            },
            MetricValue {
                name: "env/pool_reuse_fraction".into(),
                value: fraction(&self.reuse_count, &self.acquire_count),
            },
            MetricValue {
                name: "env/live_containers_max".into(),
                value: self.live_max.load(Ordering::Relaxed) as f32,
            },
            MetricValue {
                name: "env/exec_timeout_fraction".into(),
                value: fraction(&self.timeout_count, &self.exec_count),
            },
            MetricValue {
                name: "env/sandbox_broken_fraction".into(),
                value: fraction(&self.broken_count, &self.acquire_count),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(metrics: &SandboxMetrics, name: &str) -> f32 {
        metrics
            .metric_values()
            .into_iter()
            .find(|metric| metric.name == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .value
    }

    #[test]
    fn an_empty_run_reports_zeroes_rather_than_dividing_by_zero() {
        let metrics = SandboxMetrics::default();
        for name in [
            "env/container_create_ms",
            "env/acquire_ms_mean",
            "env/pool_reuse_fraction",
            "env/exec_timeout_fraction",
        ] {
            assert_eq!(value(&metrics, name), 0.0, "{name}");
        }
    }

    #[test]
    fn reuse_and_timing_are_reported_as_the_pool_sees_them() {
        let metrics = SandboxMetrics::default();
        metrics.record_acquire(4_000, false);
        metrics.record_acquire(1_000, true);
        metrics.record_acquire(1_000, true);
        metrics.record_create(20_000);
        metrics.record_live(3);
        metrics.record_live(2);

        assert!((value(&metrics, "env/pool_reuse_fraction") - 2.0 / 3.0).abs() < 1e-6);
        assert!((value(&metrics, "env/acquire_ms_mean") - 2.0).abs() < 1e-6);
        assert!((value(&metrics, "env/container_create_ms") - 20.0).abs() < 1e-6);
        // A maximum, not a last value: the point is the peak the machine had to
        // hold at once.
        assert_eq!(value(&metrics, "env/live_containers_max"), 3.0);
    }
}
