//! Verifies that the build links Metal and registers a GPU device.
//!
//! This test does not touch a model - it only exercises backend registration,
//! so it is the cheapest independent check that the build.rs Metal wiring works.

mod common;

#[test]
fn backend_list_is_non_empty_and_has_cpu() {
    let list = retrograd::backend_list().expect("backend list should succeed");
    eprintln!("registered ggml devices:\n{list}");
    assert!(
        list.lines().any(|line| line.starts_with("cpu\t")),
        "expected a CPU device to always be present, got:\n{list}"
    );
}

#[cfg(retro_metal)]
#[test]
fn metal_build_registers_a_gpu_device() {
    let list = retrograd::backend_list().expect("backend list should succeed");
    eprintln!("registered ggml devices:\n{list}");
    // The Metal backend registers its device as "MTL<n>" (e.g. MTL0). On Apple
    // hardware that is the only GPU device, so a `gpu` line proves the Metal
    // archive was linked and self-registered.
    let gpu_line = list
        .lines()
        .find(|line| line.starts_with("gpu\t"))
        .unwrap_or_else(|| panic!("Metal build must register a GPU device, got:\n{list}"));
    assert!(
        gpu_line.contains("MTL"),
        "GPU device should be the Metal (MTL) device, got: {gpu_line:?}"
    );
}

#[cfg(retro_cuda)]
#[test]
fn cuda_build_registers_a_cuda_gpu_device() {
    let list = retrograd::backend_list().expect("backend list should succeed");
    eprintln!("registered ggml devices:\n{list}");
    // The CUDA backend registers each device as "CUDA<n>" (e.g. CUDA0). A `gpu`
    // line naming CUDA proves the ggml-cuda archive was linked and self-registered
    // through ggml-base's backend registry. Whether that device can actually
    // initialize a context/stream at runtime is a separate concern the CUDA probe
    // lane checks with gpu_runtime_available(); this level-0 test only proves the
    // build linked the backend, so it must not be masked by a missing GPU.
    let gpu_line = list
        .lines()
        .find(|line| line.starts_with("gpu\t") && line.contains("CUDA"))
        .unwrap_or_else(|| panic!("CUDA build must register a CUDA GPU device, got:\n{list}"));
    eprintln!("CUDA device: {gpu_line}");
}

#[cfg(all(not(retro_metal), not(retro_vulkan), not(retro_cuda)))]
#[test]
fn cpu_only_build_has_no_gpu_device() {
    let list = retrograd::backend_list().expect("backend list should succeed");
    assert!(
        !list.lines().any(|line| line.starts_with("gpu\t")),
        "CPU-only build should not expose a GPU device, got:\n{list}"
    );
}

/// The transfer probe must produce rates, and must say what it measured.
///
/// Activation offloading is not worth building when
/// the traffic exceeds 30 % of the compute it must hide behind, and that ratio is
/// only as trustworthy as this probe. What is asserted here is not a bandwidth,
/// that is the machine's, and a threshold on it would fail on the next one - but
/// that the measurement is self-describing: the size and iteration count come
/// back, the rates are positive and finite, and a backend with no pinned host
/// buffer type says so instead of passing its pageable rate off as a pinned one.
#[cfg(any(retro_metal, retro_vulkan, retro_cuda))]
#[test]
fn the_transfer_probe_reports_self_describing_rates() {
    if !retrograd::gpu_runtime_available() {
        eprintln!("skipping: a GPU backend is registered but its runtime queue is unavailable");
        return;
    }
    const BYTES: usize = 8 * 1024 * 1024;
    const ITERATIONS: u32 = 4;
    let rates = retrograd::transfer_probe(BYTES, ITERATIONS).expect("transfer probe");
    eprintln!("{rates:#?}");
    assert_eq!(rates.bytes_per_transfer, BYTES as u64);
    assert_eq!(rates.iterations, u64::from(ITERATIONS));
    for (label, rate) in [
        ("pinned h2d", rates.pinned_h2d_bytes_per_second),
        ("pinned d2h", rates.pinned_d2h_bytes_per_second),
        ("pageable h2d", rates.pageable_h2d_bytes_per_second),
        ("pageable d2h", rates.pageable_d2h_bytes_per_second),
    ] {
        assert!(rate.is_finite() && rate > 0.0, "{label} rate is {rate}");
    }
    if rates.pinned_is_pageable {
        // The fallback must be visible as a repeat, not as a comparison.
        assert_eq!(
            rates.pinned_h2d_bytes_per_second,
            rates.pageable_h2d_bytes_per_second
        );
    }
    // A round trip of a real size must be a real duration.
    let seconds = rates
        .round_trip_seconds(BYTES as u64)
        .expect("positive rates give a round-trip time");
    assert!(
        seconds > 0.0 && seconds.is_finite(),
        "round trip {seconds} s"
    );
}

/// Zero size and zero iterations are refused rather than divided by.
#[cfg(any(retro_metal, retro_vulkan, retro_cuda))]
#[test]
fn the_transfer_probe_refuses_a_degenerate_request() {
    if !retrograd::gpu_runtime_available() {
        eprintln!("skipping: a GPU backend is registered but its runtime queue is unavailable");
        return;
    }
    assert!(retrograd::transfer_probe(0, 4).is_err());
    assert!(retrograd::transfer_probe(1024, 0).is_err());
}
