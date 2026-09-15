//! Verifies that device selection offloads model tensors to Metal.
//!
//! Requires a local GGUF model (see tests/common). Skips gracefully otherwise.

mod common;

use retrograd::{Device, TrainConfig, Trainer};

fn config_with(device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 32,
        n_batch: 32,
        n_ubatch: 16,
        epochs: 1,
        device,
        ..TrainConfig::default()
    }
}

#[test]
fn cpu_device_keeps_model_on_cpu() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let trainer = Trainer::new(&model, config_with(Device::Cpu)).expect("load model on cpu");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- CPU report ---\n{report}");

    assert!(
        report.contains("gpu_active: false"),
        "CPU device must not activate GPU:\n{report}"
    );
    assert!(
        !common::section_has_metal(&report, "model_tensors_by_buffer"),
        "CPU device must not place model tensors on Metal:\n{report}"
    );
}

#[cfg(retro_metal)]
#[test]
fn gpu_device_offloads_model_to_metal() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let trainer = Trainer::new(&model, config_with(Device::Gpu)).expect("load model on gpu");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- GPU report ---\n{report}");

    assert!(
        report.contains("gpu_active: true"),
        "GPU device must activate GPU offload:\n{report}"
    );
    assert!(
        common::section_has_metal(&report, "model_tensors_by_buffer"),
        "GPU device must place model tensors on a Metal buffer:\n{report}"
    );
}

#[cfg(retro_metal)]
#[test]
fn auto_device_prefers_gpu_when_present() {
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device registered at runtime");
        return;
    }
    let Some(model) = common::model_path_if_available() else {
        eprintln!(
            "skipping: no local model at {}",
            common::model_path().display()
        );
        return;
    };

    let trainer = Trainer::new(&model, config_with(Device::Auto)).expect("load model auto");
    let report = trainer.backend_report().expect("backend report");
    eprintln!("--- AUTO report ---\n{report}");
    assert!(
        report.contains("gpu_active: true"),
        "auto device should pick the GPU when one is present:\n{report}"
    );
}
