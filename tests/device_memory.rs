//! The device-memory instrumentation of the optimizer path (VRAM plan, M1).
//!
//! Summing `ggml_backend_buffer` sizes - which is what `backend_report`'s byte
//! breakdown does - misses the term that decides whether a long-context step
//! fits: the backends' own scratch (CUDA pool, Vulkan `prealloc_*`) plus the graph
//! allocator's transient reserve. Sampling from the host between steps misses it
//! too, because that scratch is released when the step returns. So the runtime
//! samples inside the step, and what it captures is the high-water retained by the
//! pools it instruments; these tests pin the contract of what it reports.
//!
//! Two properties matter and they are asserted separately, because only one of
//! them is attributable:
//!
//! * `scratch_*` is this process's backends - noise-free, and the term the
//!   bounded-dequantization work acts on.
//! * `device_*` comes from the device-wide budget, so it includes the compositor
//!   and every other process. Only its self-consistency can be asserted here; a
//!   directional claim on it belongs to a controlled sweep, not a unit test.

mod common;

use retrograd::training::batch::train_grpo_batch;
use retrograd::{
    Device, GrpoBatchParams, LoraConfig, LoraDtype, TargetSet, TrainConfig, TrainSequence, Trainer,
};

fn config(device: Device) -> TrainConfig {
    TrainConfig {
        // llama.cpp rounds this model's context to 256; matching that width is
        // what selects the packed multi-sequence graph, which is the path the
        // runtime instruments.
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 256,
        n_seq_max: 2,
        epochs: 1,
        learning_rate: 1.0e-3,
        device,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(1, 2.0);
    cfg.seed = 7;
    cfg.dtype = LoraDtype::F32;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

fn sequences(trainer: &mut Trainer) -> Vec<TrainSequence> {
    let prefix = trainer
        .tokenize_text("A shared prompt asks for")
        .expect("tokenize shared prefix");
    ["the first answer.", "a different second answer."]
        .into_iter()
        .enumerate()
        .map(|(index, suffix)| {
            let suffix = trainer.tokenize_text(suffix).expect("tokenize suffix");
            assert!(suffix.len() >= 2);
            let mut tokens = prefix.clone();
            tokens.extend_from_slice(&suffix[suffix.len() - 2..]);
            let mut train_mask = vec![false; tokens.len()];
            let last = tokens.len() - 1;
            train_mask[last - 1] = true;
            train_mask[last] = true;
            let old_logprobs = trainer
                .score_masked_tokens(&tokens, &train_mask)
                .expect("score packed row");
            TrainSequence {
                tokens,
                old_logprobs,
                train_mask,
                reward: index as f32,
                group_id: 17,
                intermediate_returns: vec![1.0, 0.0],
            }
        })
        .collect()
}

/// Trains one packed update and returns the trainer, so the caller can read both
/// the measured memory and the report from the same live context.
fn trained(device: Device) -> Trainer {
    let cfg = config(device);
    let mut trainer = Trainer::new(common::model_path(), cfg.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let sequences = sequences(&mut trainer);
    let params = GrpoBatchParams {
        epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 4,
        seed: 42,
        scheduler_total_rollouts: None,
    };
    let metrics = train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {})
        .expect("train packed batch");
    assert!(metrics.train_loss.is_finite());
    trainer
}

fn report_u64(report: &str, key: &str) -> Option<u64> {
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{key}: ")))
        .and_then(|value| value.trim().parse().ok())
}

/// A CPU run must report *unavailable*, not zero.
///
/// Every byte field is zero on the CPU, so without `n_samples` a caller cannot
/// tell "this build has no device to measure" from "the device peaked at 0 bytes".
/// The second reading would silently turn every VRAM regression test into a
/// tautology, which is the exact failure this field exists to prevent.
#[test]
fn a_cpu_run_reports_no_measurement_rather_than_zero_bytes() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let trainer = trained(Device::Cpu);
    let memory = trainer.optimizer_memory().expect("optimizer memory");
    assert!(
        !memory.is_measured(),
        "a CPU-only optimizer context has no device budget to sample, got {memory:?}"
    );
    assert_eq!(memory.device_total_bytes, 0);
    assert_eq!(memory.device_peak_used_bytes, 0);
    assert_eq!(memory.scratch_peak_bytes, 0);

    // The report must be equally explicit: the sample count is always present so
    // the absence of the byte lines is a statement, not an omission.
    let report = trainer.backend_report().expect("backend report");
    assert_eq!(
        report_u64(&report, "device_memory_samples"),
        Some(0),
        "{report}"
    );
    assert!(
        report_u64(&report, "device_peak_used_bytes").is_none(),
        "an unmeasured run must not publish a peak: {report}"
    );
}

/// On a GPU the readings must be self-consistent and reachable from both surfaces.
///
/// Anything that reports a peak below the sample that produced it, or a scratch
/// figure above the whole device, is measuring the wrong thing - and a broken
/// gauge here would be read as a VRAM win by every test that depends on it.
#[test]
fn a_gpu_run_measures_a_self_consistent_device_peak() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();

    let trainer = trained(Device::Gpu);
    let memory = trainer.optimizer_memory().expect("optimizer memory");
    assert!(
        memory.is_measured(),
        "an active GPU optimizer context must sample its device budget"
    );
    assert!(memory.device_total_bytes > 0, "{memory:?}");
    assert!(
        memory.device_used_bytes <= memory.device_total_bytes,
        "{memory:?}"
    );
    assert!(
        memory.device_peak_used_bytes >= memory.device_used_bytes,
        "a running maximum cannot sit below the sample that produced it: {memory:?}"
    );
    assert!(
        memory.scratch_peak_bytes >= memory.scratch_bytes,
        "{memory:?}"
    );
    assert!(
        memory.scratch_peak_bytes <= memory.device_total_bytes,
        "backend scratch cannot exceed the device: {memory:?}"
    );

    // Both surfaces must agree; the report is what the CLI and the other memory
    // tests read, and it is served from a cache that has to notice a moved peak.
    let report = trainer.backend_report().expect("backend report");
    assert_eq!(
        report_u64(&report, "device_peak_used_bytes"),
        Some(memory.device_peak_used_bytes),
        "{report}"
    );
    assert_eq!(
        report_u64(&report, "backend_scratch_peak_bytes"),
        Some(memory.scratch_peak_bytes),
        "{report}"
    );
}

/// OUT_PROD decodes production quant types in place, so the compatibility-path
/// budget must not affect a Q8_0 training graph's scratch peak.
/// The pool still has other users, hence the non-zero accounting assertion.
#[cfg(retro_cuda)]
#[test]
fn native_quant_out_prod_scratch_is_independent_of_the_legacy_dequant_budget() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();

    let scratch_peak = |budget_mb: Option<&str>| -> u64 {
        let _env = budget_mb.map(|mb| common::EnvGuard::set("GGML_CUDA_DEQUANT_BUDGET_MB", mb));
        let trainer = trained(Device::Gpu);
        let memory = trainer.optimizer_memory().expect("optimizer memory");
        assert!(
            memory.is_measured(),
            "the budget cannot be characterized without a measurement"
        );
        memory.scratch_peak_bytes
    };

    let default_budget = scratch_peak(None);
    let tight_budget = scratch_peak(Some("1"));
    eprintln!("backend scratch peak: default {default_budget} B, 1 MiB budget {tight_budget} B");
    assert!(
        default_budget > 0,
        "a Q8_0 GPU training step must allocate backend scratch; measuring zero means \
         the accounting is not wired to the pool"
    );
    assert_eq!(
        tight_budget, default_budget,
        "the legacy dequantization budget changed a native-Q2 training graph's scratch"
    );
}

/// The report must not pin the peak of the first step for the rest of the run.
///
/// `backend_report` is cached and invalidated only by structural mutations. The
/// measured peak moves with every step and nothing else in the report moves with
/// it, so without an explicit staleness rule the cache would keep serving the
/// first step's figure - which reads as "the peak never grew".
#[test]
fn the_report_follows_the_peak_across_further_training() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();

    let cfg = config(Device::Gpu);
    let mut trainer = Trainer::new(common::model_path(), cfg.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let sequences = sequences(&mut trainer);
    let params = GrpoBatchParams {
        epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        loss_denominator: 4,
        seed: 42,
        scheduler_total_rollouts: None,
    };

    train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {}).expect("first update");
    let first = trainer.backend_report().expect("report after first update");
    let first_peak = report_u64(&first, "device_peak_used_bytes")
        .unwrap_or_else(|| panic!("no peak after a GPU update:\n{first}"));

    train_grpo_batch(&mut trainer, &sequences, &params, &cfg, &mut |_| {}).expect("second update");
    let second = trainer
        .backend_report()
        .expect("report after second update");
    let second_peak = report_u64(&second, "device_peak_used_bytes")
        .unwrap_or_else(|| panic!("no peak after the second GPU update:\n{second}"));

    // A running maximum: never decreasing is the property. Requiring growth would
    // be wrong - a steady-state second step legitimately peaks no higher.
    assert!(
        second_peak >= first_peak,
        "peak went backwards: {first_peak} then {second_peak}"
    );
    // Every measured field, not just the peak. They come from one sampled block and
    // the cache is keyed on its sample count, so a rule that tracked only the peak
    // would serve a stale `device_used_bytes` or scratch figure whenever the peak
    // happened to sit still - which is the ordinary case in steady state.
    let live = trainer.optimizer_memory().expect("optimizer memory");
    for (key, expected) in [
        ("device_memory_samples", live.n_samples),
        ("device_total_bytes", live.device_total_bytes),
        ("device_used_bytes", live.device_used_bytes),
        ("device_peak_used_bytes", live.device_peak_used_bytes),
        ("backend_scratch_bytes", live.scratch_bytes),
        ("backend_scratch_peak_bytes", live.scratch_peak_bytes),
    ] {
        assert_eq!(
            report_u64(&second, key),
            Some(expected),
            "the cached report drifted from the live {key}:\n{second}"
        );
    }
}
