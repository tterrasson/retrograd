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
    Device, GrpoBatchParams, LoraConfig, LoraDtype, OptimizerKind, TargetSet, TrainConfig,
    TrainSequence, TrainableEntry, TrainablePolicy, TrainableRunConfig, TrainableSelector, Trainer,
    resolve_base, tensor_inventory,
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

/// The dequantization budget bounds a quantized training step's scratch peak.
///
/// A quantized `OUT_PROD` has two CUDA paths and this graph uses both: the
/// scratch-free in-place decoder for small reductions and strided weights, and
/// bounded dequantize+SGEMM for the wide contiguous projections, where
/// serializing the whole reduction inside a tile costs milliseconds per node.
/// The second one sizes its F32 scratch from `GGML_CUDA_DEQUANT_BUDGET_MB`, so
/// the budget *is* expected to move the peak - downwards. What must hold is
/// that it only ever bounds it: a tighter budget never buys more scratch, and
/// the accounting still sees a non-zero pool, without which the comparison
/// would be two zeros. That the budget changes no *number* the step produces
/// is a separate claim, asserted on the op itself by
/// `common::assert_out_prod_quant_budget_independent` and by the type sweep in
/// `tests/out_prod_quant.rs`.
#[cfg(retro_cuda)]
#[test]
fn the_legacy_dequant_budget_only_bounds_a_quant_training_steps_scratch() {
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
        "a quantized GPU training step must allocate backend scratch; measuring zero \
         means the accounting is not wired to the pool"
    );
    assert!(
        tight_budget <= default_budget,
        "a tighter dequantization budget must not raise the scratch peak: \
         1 MiB {tight_budget} B against {default_budget} B at the default"
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

/// The same shape as [`config`], with a base policy. `partial` rather than
/// `full`: the download fixture is quantized, so its F32 norms are the only
/// eligible tensors.
fn base_config(device: Device) -> TrainConfig {
    TrainConfig {
        trainable: TrainableRunConfig {
            policy: TrainablePolicy::Partial,
            selector: TrainableSelector {
                norms: true,
                ..Default::default()
            },
            optimizer: OptimizerKind::AdamW,
        },
        ..config(device)
    }
}

/// Trains one base update the way a run does, set resolved against the model's
/// tensor table and declared before the graph exists, and returns the trainer.
fn trained_base(device: Device) -> Trainer {
    let cfg = base_config(device);
    let inventory = tensor_inventory(common::model_path(), device).expect("tensor inventory");
    let set = resolve_base(&inventory, cfg.trainable.policy, &cfg.trainable.selector)
        .expect("a quantized fixture still has F32 norms");
    assert!(!set.entries.is_empty());
    let names: Vec<String> = set
        .entries
        .iter()
        .map(|entry: &TrainableEntry| entry.name.clone())
        .collect();

    let mut trainer = Trainer::new(common::model_path(), cfg).expect("load trainer");
    trainer.set_trainable_base(&names).expect("declare the set");
    let tokens = trainer
        .tokenize_text(&"A shared prompt asks for the first answer. ".repeat(64))
        .expect("tokenize");
    let metrics = trainer
        .train_tokens(&tokens)
        .expect("train one base update");
    assert!(metrics.train_loss.is_finite());
    trainer
}

/// A base run's components and its measured peak, asserted apart.
///
/// The component totals are exact: the device/host split is a partition of
/// their sum. The measured peak is not exact: it is the device-wide budget,
/// so it carries other processes, and the gap is backend scratch plus the
/// allocator's transient reserve. The gap is recorded, not gated.
#[test]
fn a_gpu_base_run_accounts_for_its_components_and_measures_its_peak() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();

    let trainer = trained_base(Device::Gpu);
    let report = trainer.memory_report().expect("memory report");

    // --- The run trains what is claimed -------------------------------
    assert!(
        report.trainable_parameters_are_model_subset,
        "a base run trains tensors that are already in the loaded weights: {report:?}"
    );
    assert!(report.trainable_gradient_bytes > 0, "{report:?}");
    assert!(
        report.optimizer_state_bytes >= report.trainable_gradient_bytes * 2,
        "AdamW keeps two F32 moments per parameter: {report:?}"
    );
    assert!(report.model_weight_bytes > 0, "{report:?}");
    assert!(report.optimizer_compute_bytes > 0, "{report:?}");

    // --- Components: an exact partition ---------------------------------
    // No adapter, so `trainable_parameter_bytes` is a slice of the model
    // weights and is not added.
    let components = report.model_weight_bytes
        + report.optimizer_kv_bytes
        + report.optimizer_compute_bytes
        + report.generation_kv_bytes
        + report.generation_compute_bytes
        + report.trainable_gradient_bytes
        + report.optimizer_state_bytes;
    assert_eq!(
        report.device_bytes + report.host_bytes,
        components,
        "the device/host split is not a partition of the components: {report:?}"
    );
    assert!(
        report.device_bytes > 0,
        "an offloaded base run allocates on the device: {report:?}"
    );

    // --- Measurement: self-consistency only -----------------------------
    assert!(
        report.is_measured(),
        "an active GPU optimizer context must sample its device budget: {report:?}"
    );
    assert!(report.device_total_bytes > 0, "{report:?}");
    assert!(
        report.device_used_bytes <= report.device_total_bytes,
        "{report:?}"
    );
    assert!(
        report.device_peak_used_bytes >= report.device_used_bytes,
        "{report:?}"
    );
    assert!(
        report.backend_scratch_peak_bytes <= report.device_total_bytes,
        "{report:?}"
    );

    // --- Recorded, not gated ---------------------------------------------
    let unaccounted = report
        .unaccounted_device_bytes()
        .expect("a measured run answers the gap");
    eprintln!(
        "base gpu run: components {components} B (device {} B, host {} B), \
         device peak {} B over {} sample(s), backend scratch peak {} B, \
         unaccounted (driver + allocator reserve) {unaccounted} B, \
         base trainable on host: {}",
        report.device_bytes,
        report.host_bytes,
        report.device_peak_used_bytes,
        report.device_memory_samples,
        report.backend_scratch_peak_bytes,
        report.base_trainable_on_host,
    );
}

/// The CPU counterpart: every measured field is zero on the CPU, and the
/// component partition must still hold.
#[test]
fn a_cpu_base_run_accounts_for_its_components_and_reports_no_measurement() {
    let Some(_model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();

    let trainer = trained_base(Device::Cpu);
    let report = trainer.memory_report().expect("memory report");

    let components = report.model_weight_bytes
        + report.optimizer_kv_bytes
        + report.optimizer_compute_bytes
        + report.generation_kv_bytes
        + report.generation_compute_bytes
        + report.trainable_gradient_bytes
        + report.optimizer_state_bytes;
    assert_eq!(
        report.device_bytes + report.host_bytes,
        components,
        "{report:?}"
    );
    assert_eq!(
        report.device_bytes, 0,
        "a CPU run has no device budget to draw on: {report:?}"
    );
    assert!(
        report.base_trainable_on_host,
        "a CPU run's base tensors are host tensors: {report:?}"
    );

    assert!(!report.is_measured(), "{report:?}");
    assert_eq!(report.device_peak_used_bytes, 0);
    assert_eq!(report.backend_scratch_peak_bytes, 0);
    assert_eq!(report.unaccounted_device_bytes(), None);
}

/// A run's anchor is a second model, and on a device it is a second model *on
/// the device*.
///
/// The anchor's cost has always been an arithmetic term in the plan's estimate,
/// `co_resident_bytes`, and nothing had measured it where it is actually paid.
/// This is that measurement: the same base run, once without an anchor and once
/// with, on the device, with the runtime's own report on both sides.
///
/// What is asserted is the part that is attributable. The anchor's own report
/// is exact - it is a forward-only trainer, so its bytes are weights plus KV
/// and nothing else - and the *process's* backend scratch is noise-free. The
/// device-wide budget is not asserted directionally, for the reason the module
/// header gives: it carries every other process on the card.
#[test]
fn an_anchor_on_a_device_is_a_second_model_on_the_device() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    if !common::gpu_device_present() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let _guard = common::serialize_models();

    let mut trainer = trained_base(Device::Gpu);
    assert!(
        trainer
            .reference_memory_report()
            .expect("the anchor's report")
            .is_none(),
        "a run with no anchor has no anchor report"
    );
    let alone = trainer.memory_report().expect("memory report");

    // A copy of the file rather than the file: an anchor that shares the
    // loaded weights would measure nothing at all.
    let root = std::env::temp_dir().join(format!("retrograd-anchor-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create scratch directory");
    let copy = root.join("anchor.gguf");
    std::fs::copy(&model, &copy).expect("copy the model");

    trainer
        .attach_reference(&copy, &base_config(Device::Gpu), None)
        .expect("attach the anchor");

    let anchor = trainer
        .reference_memory_report()
        .expect("the anchor's report")
        .expect("an attached anchor reports its own memory");

    // --- What the anchor is, exactly ------------------------------------
    assert!(
        anchor.model_weight_bytes > 0,
        "the anchor holds the weights it scores with: {anchor:?}"
    );
    assert_eq!(
        anchor.trainable_gradient_bytes, 0,
        "a forward-only anchor has no gradients: {anchor:?}"
    );
    assert_eq!(
        anchor.optimizer_state_bytes, 0,
        "a forward-only anchor has no optimizer state: {anchor:?}"
    );
    assert!(
        anchor.device_bytes > 0,
        "an anchor attached to a device run is resident on the device, not on the \
         host: {anchor:?}"
    );
    // The same weights as the run it anchors: it is a copy of the same file.
    assert_eq!(
        anchor.model_weight_bytes, alone.model_weight_bytes,
        "a copy of the model weighs what the model weighs"
    );

    // --- That it is actually used --------------------------------------
    let tokens = trainer
        .tokenize_text("A shared prompt asks for the first answer.")
        .expect("tokenize");
    let scores = trainer
        .score_reference_tokens(&tokens)
        .expect("the anchor scores this run's tokens");
    assert!(
        !scores.is_empty()
            && scores
                .iter()
                .all(|score| score.is_finite() && *score <= 0.0),
        "the anchor's scores are finite log-probabilities"
    );

    // --- Recorded, not gated --------------------------------------------
    let with_anchor = trainer.memory_report().expect("memory report");
    eprintln!(
        "anchor on a device: run device {} B, anchor device {} B (weights {} B, kv {} B), \
         run peak {} B -> {} B, scratch peak {} B -> {} B",
        alone.device_bytes,
        anchor.device_bytes,
        anchor.model_weight_bytes,
        anchor.optimizer_kv_bytes,
        alone.device_peak_used_bytes,
        with_anchor.device_peak_used_bytes,
        alone.backend_scratch_peak_bytes,
        with_anchor.backend_scratch_peak_bytes,
    );

    // The run's own report describes the trained model alone and must not have
    // absorbed the anchor: whether the two should ever be summed is a question
    // about the estimate, and this is the measurement that keeps it answerable.
    assert_eq!(
        with_anchor.model_weight_bytes, alone.model_weight_bytes,
        "the run's report describes the model it trains, not the model it scores against"
    );

    let _ = std::fs::remove_dir_all(&root);
}
