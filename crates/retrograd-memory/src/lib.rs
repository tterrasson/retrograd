//! Dependency-free process-memory accounting.
//!
//! The runtime does not expose per-buffer accounting, so attribution works by
//! deltas: a [`MemoryTracker`] snapshots the process footprint between named
//! phases (model load, adapter creation, dataset preparation,...) and charges
//! each phase with the growth it caused. On Apple Silicon the physical
//! footprint includes Metal allocations, so GPU training buffers are visible
//! here too.
//!
//! On a discrete GPU the host counters miss the part that actually decides
//! whether a run fits: `VmRSS` never contains VRAM. [`device_snapshot`] reads
//! the backend's own budget so the device side is tracked separately, and the
//! tracker keeps its peak next to the host peak.

use std::fmt::Write as _;

use retrograd_metrics::MetricValue;

/// One reading of the process memory counters, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Resident set size: physical pages currently mapped.
    pub resident: u64,
    /// Physical footprint: the OS's dirty + compressed + IOKit charge on
    /// macOS (the number `footprint(1)` and Jetsam use); equals `resident`
    /// on Linux.
    pub footprint: u64,
    /// Lifetime peak resident size.
    pub peak: u64,
}

/// Reads the current counters, or `None` on unsupported platforms.
pub fn snapshot() -> Option<MemorySnapshot> {
    imp::snapshot()
}

/// Total physical RAM installed on the machine, or `None` on an unsupported
/// platform.
///
/// The host counterpart of [`DeviceMemory::total`]: a RAM budget expressed
/// as a fraction, or capped against "all of the system", needs this denominator,
/// and the per-process counters in [`MemorySnapshot`] cannot supply it.
pub fn system_memory_bytes() -> Option<u64> {
    imp::system_memory_bytes()
}

/// One reading of the GPU device memory budget, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Free device memory reported by the backend.
    pub free: u64,
    /// Total device memory reported by the backend.
    pub total: u64,
}

impl DeviceMemory {
    /// Device memory in use. The backend reports a device-wide budget, so this
    /// includes other processes: compare deltas between phases rather than
    /// reading the absolute value as "this run's VRAM".
    pub fn used(self) -> u64 {
        self.total.saturating_sub(self.free)
    }
}

/// Reads the first GPU device's memory budget, or `None` when the build has no
/// GPU backend (CPU-only builds, or a GPU-less machine).
pub fn device_snapshot() -> Option<DeviceMemory> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: both out-pointers are valid for the duration of the call; the
    // runtime writes them only when it returns 0.
    let ok = unsafe { retrograd_ffi::retro_device_memory(&mut free, &mut total) } == 0;
    // A backend that cannot report its budget answers 0/0, which is not a
    // measurement; treat it as unavailable rather than as an empty device.
    (ok && total > 0).then_some(DeviceMemory {
        free: free as u64,
        total: total as u64,
    })
}

#[cfg(target_os = "macos")]
mod imp {
    use super::MemorySnapshot;

    /// `struct rusage_info_v0` from XNU's `<sys/resource.h>`.
    #[repr(C)]
    #[derive(Default)]
    struct RusageInfoV0 {
        ri_uuid: [u8; 16],
        ri_user_time: u64,
        ri_system_time: u64,
        ri_pkg_idle_wkups: u64,
        ri_interrupt_wkups: u64,
        ri_pageins: u64,
        ri_wired_size: u64,
        ri_resident_size: u64,
        ri_phys_footprint: u64,
        ri_proc_start_abstime: u64,
        ri_proc_exit_abstime: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
    }

    /// `struct rusage` from `<sys/resource.h>`; only `ru_maxrss` is read but
    /// the full layout must be declared so the kernel writes in bounds.
    #[repr(C)]
    #[derive(Default)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        ru_nswap: i64,
        ru_inblock: i64,
        ru_oublock: i64,
        ru_msgsnd: i64,
        ru_msgrcv: i64,
        ru_nsignals: i64,
        ru_nvcsw: i64,
        ru_nivcsw: i64,
    }

    const RUSAGE_INFO_V0: i32 = 0;
    const RUSAGE_SELF: i32 = 0;

    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV0) -> i32;
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        fn getpid() -> i32;
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    pub fn system_memory_bytes() -> Option<u64> {
        let mut value: u64 = 0;
        let mut length = std::mem::size_of::<u64>();
        // SAFETY: the name is a NUL-terminated literal, and `value`/`length`
        // describe a correctly sized destination the kernel writes at most once.
        let ok = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut value as *mut u64).cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            ) == 0
        };
        (ok && value > 0).then_some(value)
    }

    pub fn snapshot() -> Option<MemorySnapshot> {
        let mut info = RusageInfoV0::default();
        let mut usage = Rusage::default();
        // SAFETY: both structs match the C layouts above and outlive the calls.
        let (rusage_ok, getrusage_ok) = unsafe {
            (
                proc_pid_rusage(getpid(), RUSAGE_INFO_V0, &mut info) == 0,
                getrusage(RUSAGE_SELF, &mut usage) == 0,
            )
        };
        if !rusage_ok {
            return None;
        }
        Some(MemorySnapshot {
            resident: info.ri_resident_size,
            footprint: info.ri_phys_footprint,
            // macOS reports ru_maxrss in bytes.
            peak: if getrusage_ok {
                usage.ru_maxrss.max(0) as u64
            } else {
                0
            },
        })
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::MemorySnapshot;

    pub fn snapshot() -> Option<MemorySnapshot> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let field = |name: &str| -> Option<u64> {
            let line = status.lines().find(|line| line.starts_with(name))?;
            let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            Some(kib * 1024)
        };
        let resident = field("VmRSS:")?;
        Some(MemorySnapshot {
            resident,
            footprint: resident,
            peak: field("VmHWM:").unwrap_or(0),
        })
    }

    pub fn system_memory_bytes() -> Option<u64> {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = info.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        (kib > 0).then_some(kib * 1024)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn snapshot() -> Option<super::MemorySnapshot> {
        None
    }

    pub fn system_memory_bytes() -> Option<u64> {
        None
    }
}

/// Log a change only once it is both absolutely and relatively meaningful,
/// so steady-state training stays quiet.
const LOG_ABSOLUTE_THRESHOLD: u64 = 32 * 1024 * 1024;
const LOG_RELATIVE_THRESHOLD: f64 = 0.05;

/// One closed phase: how much the host footprint and the device budget moved
/// while it ran.
#[derive(Clone, Debug)]
struct Phase {
    name: String,
    host_delta: i64,
    /// `None` when the build has no GPU device to read.
    device_delta: Option<i64>,
}

/// Attributes footprint growth to named phases and reports significant moves.
#[derive(Debug, Default)]
pub struct MemoryTracker {
    baseline: Option<MemorySnapshot>,
    previous: Option<MemorySnapshot>,
    logged_footprint: u64,
    phases: Vec<Phase>,
    device_baseline: Option<DeviceMemory>,
    device_previous: Option<DeviceMemory>,
    device_peak_used: u64,
}

impl MemoryTracker {
    /// Starts tracking from the current process footprint.
    pub fn new() -> Self {
        let baseline = snapshot();
        let device_baseline = device_snapshot();
        Self {
            baseline,
            previous: baseline,
            logged_footprint: baseline.map_or(0, |snapshot| snapshot.footprint),
            phases: Vec::new(),
            device_baseline,
            device_previous: device_baseline,
            device_peak_used: device_baseline.map_or(0, DeviceMemory::used),
        }
    }

    /// Closes a phase: everything the footprint grew since the previous phase
    /// is charged to `name`. Returns a printable one-line report.
    pub fn phase(&mut self, name: &str) -> Option<String> {
        let device_delta = self.observe_device().and_then(|current| {
            let previous = self.device_previous.replace(current)?;
            Some(current.used() as i64 - previous.used() as i64)
        });
        let current = snapshot()?;
        let previous = self.previous.replace(current)?;
        let delta = current.footprint as i64 - previous.footprint as i64;
        self.phases.push(Phase {
            name: name.to_string(),
            host_delta: delta,
            device_delta,
        });
        self.logged_footprint = current.footprint;
        let device = device_delta
            .map(|delta| format!(", vram {}", format_signed_bytes(delta)))
            .unwrap_or_default();
        Some(format!(
            "memory {name}: {} (footprint {}{device})",
            format_signed_bytes(delta),
            format_bytes(current.footprint),
        ))
    }

    /// Samples the device budget and folds it into the running peak. Returns
    /// `None` when no GPU device is available.
    pub fn observe_device(&mut self) -> Option<DeviceMemory> {
        let current = device_snapshot()?;
        self.device_peak_used = self.device_peak_used.max(current.used());
        Some(current)
    }

    /// Highest device usage seen so far, or `None` without a GPU device.
    pub fn device_peak_used(&self) -> Option<u64> {
        self.device_baseline.map(|_| self.device_peak_used)
    }

    /// Multi-line component breakdown: baseline, one line per phase, totals.
    pub fn summary(&self) -> Option<String> {
        let baseline = self.baseline?;
        let current = snapshot()?;
        let mut text = String::new();
        let _ = writeln!(
            text,
            "baseline           {:>10}",
            format_bytes(baseline.footprint)
        );
        for phase in &self.phases {
            let device = phase
                .device_delta
                .map(|delta| format!("   vram {}", format_signed_bytes(delta)))
                .unwrap_or_default();
            let _ = writeln!(
                text,
                "{:<18} {:>10}{device}",
                phase.name,
                format_signed_bytes(phase.host_delta),
            );
        }
        let _ = writeln!(
            text,
            "footprint          {:>10}   resident {}   peak rss {}",
            format_bytes(current.footprint),
            format_bytes(current.resident),
            format_bytes(current.peak),
        );
        if let (Some(baseline), Some(peak)) = (self.device_baseline, self.device_peak_used()) {
            let _ = writeln!(
                text,
                "vram device        {:>10}   total {}   peak used {}",
                format_bytes(device_snapshot().unwrap_or(baseline).used()),
                format_bytes(baseline.total),
                format_bytes(peak),
            );
        }
        text.pop();
        Some(text)
    }

    /// Samples the counters mid-training. Returns a log line only when the
    /// footprint moved significantly since the last reported value, so the
    /// caller can print it without flooding the output.
    pub fn observe(&mut self) -> (Option<MemorySnapshot>, Option<String>) {
        let Some(current) = snapshot() else {
            return (None, None);
        };
        self.previous = Some(current);
        let delta = current.footprint as i64 - self.logged_footprint as i64;
        let relative = delta.unsigned_abs() as f64 / (self.logged_footprint.max(1) as f64);
        let message = (delta.unsigned_abs() >= LOG_ABSOLUTE_THRESHOLD
            && relative >= LOG_RELATIVE_THRESHOLD)
            .then(|| {
                self.logged_footprint = current.footprint;
                format!(
                    "memory: footprint {} ({}), resident {}, peak rss {}",
                    format_bytes(current.footprint),
                    format_signed_bytes(delta),
                    format_bytes(current.resident),
                    format_bytes(current.peak),
                )
            });
        (Some(current), message)
    }

    /// Device-side metric values for the sinks, distinct from the host RSS
    /// series. Empty when the build has no GPU device.
    pub fn device_metric_values(&mut self) -> Vec<MetricValue> {
        let Some(current) = self.observe_device() else {
            return Vec::new();
        };
        const MIB: f32 = 1024.0 * 1024.0;
        vec![
            MetricValue {
                name: "system/vram_used_mib".into(),
                value: current.used() as f32 / MIB,
            },
            MetricValue {
                name: "system/vram_free_mib".into(),
                value: current.free as f32 / MIB,
            },
            MetricValue {
                name: "system/vram_peak_used_mib".into(),
                value: self.device_peak_used as f32 / MIB,
            },
        ]
    }
}

/// Metric values for the sinks, in MiB so the charts stay readable.
pub fn metric_values(snapshot: MemorySnapshot) -> Vec<MetricValue> {
    const MIB: f32 = 1024.0 * 1024.0;
    vec![
        MetricValue {
            name: "system/memory_footprint_mib".into(),
            value: snapshot.footprint as f32 / MIB,
        },
        MetricValue {
            name: "system/memory_resident_mib".into(),
            value: snapshot.resident as f32 / MIB,
        },
        MetricValue {
            name: "system/memory_peak_rss_mib".into(),
            value: snapshot.peak as f32 / MIB,
        },
    ]
}

/// Renders a byte count with the coarsest unit (KiB/MiB/GiB) that keeps at
/// least one significant digit, e.g. `1.50 GiB`.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else {
        format!("{:.0} KiB", bytes / KIB)
    }
}

fn format_signed_bytes(delta: i64) -> String {
    let magnitude = format_bytes(delta.unsigned_abs());
    if delta < 0 {
        format!("-{magnitude}")
    } else {
        format!("+{magnitude}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RAM denominator behind a fractional host budget. Either the platform
    /// supports it and the figure is plausible, or it is absent - a zero or a
    /// laughably small total would silently shrink every budget derived from it.
    #[test]
    fn system_memory_is_plausible_or_absent() {
        match system_memory_bytes() {
            Some(total) => {
                assert!(
                    total >= 256 * 1024 * 1024,
                    "implausible system memory total: {total}"
                );
                let resident = snapshot().map(|s| s.resident).unwrap_or(0);
                assert!(
                    resident <= total,
                    "resident {resident} exceeds total {total}"
                );
            }
            // The two platforms this crate implements must always answer.
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            None => panic!("macOS and Linux must report a system memory total"),
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            None => {}
        }
    }

    #[test]
    fn snapshot_reports_nonzero_counters_on_supported_platforms() {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let snapshot = snapshot().expect("snapshot available");
            assert!(snapshot.resident > 0);
            assert!(snapshot.footprint > 0);
            assert!(snapshot.peak >= snapshot.resident / 2);
        }
    }

    #[test]
    fn tracker_attributes_growth_to_phases() {
        let mut tracker = MemoryTracker::new();
        // Allocate well above the reporting granularity and touch every page
        // so the kernel actually charges it to the footprint.
        let mut ballast = vec![0_u8; 96 * 1024 * 1024];
        for page in ballast.chunks_mut(4096) {
            page[0] = 1;
        }
        let report = tracker.phase("ballast").expect("phase report");
        assert!(report.contains("memory ballast: +"), "{report}");
        let phase = tracker.phases.last().expect("recorded phase");
        assert_eq!(phase.name, "ballast");
        assert!(
            phase.host_delta > 64 * 1024 * 1024,
            "expected the ballast to dominate the phase delta, got {}",
            phase.host_delta
        );
        drop(ballast);
        let summary = tracker.summary().expect("summary");
        assert!(summary.contains("ballast"));
        assert!(summary.contains("footprint"));
    }

    #[test]
    fn observe_logs_only_significant_moves() {
        let mut tracker = MemoryTracker::new();
        let baseline = tracker.logged_footprint;
        let (snapshot, message) = tracker.observe();
        let snapshot = snapshot.expect("snapshot available");
        // Sibling tests share this process and allocate, so the footprint can
        // move between the two readings: assert the reporting rule rather than
        // an absence of movement.
        let delta = snapshot.footprint as i64 - baseline as i64;
        let significant = delta.unsigned_abs() >= LOG_ABSOLUTE_THRESHOLD
            && delta.unsigned_abs() as f64 / (baseline.max(1) as f64) >= LOG_RELATIVE_THRESHOLD;
        assert_eq!(
            message.is_some(),
            significant,
            "delta {delta} reported as {message:?}"
        );
    }

    #[test]
    fn device_snapshot_is_consistent_or_absent() {
        // CPU-only builds legitimately have no device; when one exists its
        // budget must be self-consistent so the deltas mean something.
        let Some(device) = device_snapshot() else {
            return;
        };
        assert!(device.total > 0);
        assert!(device.free <= device.total);
        assert_eq!(device.used(), device.total - device.free);
    }

    #[test]
    fn device_metrics_are_reported_only_with_a_device() {
        let mut tracker = MemoryTracker::new();
        let values = tracker.device_metric_values();
        if device_snapshot().is_none() {
            assert!(values.is_empty());
            assert!(tracker.device_peak_used().is_none());
            return;
        }
        let names: Vec<_> = values.iter().map(|value| value.name.as_ref()).collect();
        assert_eq!(
            names,
            [
                "system/vram_used_mib",
                "system/vram_free_mib",
                "system/vram_peak_used_mib",
            ]
        );
        assert!(tracker.device_peak_used().is_some());
        // The peak is a running maximum, so it can never sit below the sample
        // that just produced it.
        assert!(values[2].value >= values[0].value);
    }

    #[test]
    fn byte_formatting_picks_a_readable_unit() {
        assert_eq!(format_bytes(512 * 1024), "512 KiB");
        assert_eq!(format_bytes(8 * 1024 * 1024), "8.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024 / 2), "1.50 GiB");
        assert_eq!(format_signed_bytes(-(8 * 1024 * 1024)), "-8.0 MiB");
        assert_eq!(format_signed_bytes(8 * 1024 * 1024), "+8.0 MiB");
    }

    #[test]
    fn metric_values_are_reported_in_mib() {
        let values = metric_values(MemorySnapshot {
            resident: 2 * 1024 * 1024,
            footprint: 3 * 1024 * 1024,
            peak: 4 * 1024 * 1024,
        });
        let names: Vec<_> = values.iter().map(|value| value.name.as_ref()).collect();
        assert_eq!(
            names,
            [
                "system/memory_footprint_mib",
                "system/memory_resident_mib",
                "system/memory_peak_rss_mib",
            ]
        );
        assert_eq!(values[0].value, 3.0);
        assert_eq!(values[1].value, 2.0);
        assert_eq!(values[2].value, 4.0);
    }
}
