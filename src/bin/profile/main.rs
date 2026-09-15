//! Standalone end-to-end training profiler.
//!
//! Runs a short GRPO or agentic-GRPO training and reports wall-clock phases,
//! host memory, device memory, and backend allocation details. Component and
//! per-buffer figures come from the post-preflight backend report.
//!
//! The supported algorithms run their training loops directly so raw timing and
//! scoring progress reaches the profiler instead of the condensed run observer.
//!
//! Run from the repository root so the reward command and dataset resolve:
//!   cargo run --bin profile -- [config.toml] [--device gpu] [--updates 2]...

#[cfg(feature = "agent")]
mod agent;
// The `retrograd` binary's flag reader, included by path: this binary uses less
// of it than that one does.
#[allow(dead_code)]
#[path = "../../args.rs"]
mod args;
mod report;

use std::time::Instant;

use args::Args;
use report::{
    Workload, print_buffer_memory_table, print_checkpoint_table, print_component_memory_table,
    print_duty_cycle_line, print_footer, print_header, print_init_table, print_memory_table,
    print_optimizer_timing_table, print_phase_table, print_updates_table, print_vram_table,
};
use retrograd::{
    Device, MemoryReport, Trainer,
    config::{self, Algorithm, RunConfig},
    memory,
    training::{self, Progress},
};

/// Workload overrides. `None` keeps the value from the config file.
struct Options {
    config_path: String,
    model: Option<std::path::PathBuf>,
    device: Option<Device>,
    updates: Option<u32>,
    group_size: Option<usize>,
    epochs: Option<u32>,
    prompts_per_update: Option<usize>,
    micro_batch: Option<u32>,
    max_new_tokens: Option<u32>,
    full: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            config_path: "examples/smoke_tiny_grpo.toml".to_string(),
            model: None,
            device: None,
            // Fast defaults so a run fits in a few seconds; --full opts out.
            updates: Some(1),
            group_size: Some(4),
            epochs: Some(1),
            // Several prompts so at least one group has reward variance;
            // zero-variance groups are dropped by GRPO and never reach the
            // optimizer, which would leave its phase reading 0 s.
            prompts_per_update: Some(3),
            micro_batch: None,
            // Keep the config's budget: shrinking it makes every completion hit
            // the truncation limit, and `mask_truncated` then drops them all, so
            // the optimizer never runs and its phase reads a misleading 0 s.
            max_new_tokens: None,
            full: false,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("\x1b[31merror\x1b[0m {error}");
        std::process::exit(1);
    }
}

fn run() -> retrograd::Result<()> {
    let options = parse_args()?;
    let wall = Instant::now();

    let run_config = config::load_with(
        &options.config_path,
        config::ModelOverride {
            path: options.model.clone(),
            device: options.device,
        },
    )?;

    match &run_config.algorithm {
        Algorithm::Grpo(_) => run_grpo(options, run_config, wall),
        #[cfg(feature = "agent")]
        Algorithm::AgentGrpo(_) => agent::run(options, run_config, wall),
        #[cfg(not(feature = "agent"))]
        Algorithm::AgentGrpo(_) => Err(retrograd::Error::invalid(
            "run.algorithm = 'agent_grpo' needs the agentic loop, which is not compiled into \
             this binary: rebuild with `--features agent`",
        )),
        _ => Err(retrograd::Error::invalid(
            "the profiler currently supports GRPO and agent_grpo configs only",
        )),
    }
}

fn run_grpo(options: Options, mut run_config: RunConfig, wall: Instant) -> retrograd::Result<()> {
    let Algorithm::Grpo(grpo) = &mut run_config.algorithm else {
        unreachable!("run_grpo is only called for an Algorithm::Grpo config");
    };

    if !options.full {
        if let Some(value) = options.updates {
            grpo.updates = value;
        }
        if let Some(value) = options.group_size {
            grpo.group_size = value;
        }
        if let Some(value) = options.epochs {
            grpo.grpo_epochs = value;
        }
        if let Some(value) = options.prompts_per_update {
            grpo.prompts_per_update = value;
        }
        if let Some(value) = options.max_new_tokens {
            grpo.sampling.max_new_tokens = value;
        }
    }
    let group_size = grpo.group_size;
    let prompts_per_update = grpo.prompts_per_update;
    apply_micro_batch_and_packing(&options, &mut run_config, group_size, prompts_per_update)?;

    let Algorithm::Grpo(grpo) = &run_config.algorithm else {
        unreachable!("run_grpo is only called for an Algorithm::Grpo config");
    };
    let grpo = grpo.clone();
    let training = run_config.training.clone();

    let mut vram = VramTrack::new();
    let mem_start = snapshot_bytes();
    let (mut trainer, init) = init_trainer(&run_config, &mut vram)?;

    let mut phases = PhaseTotals::default();
    let mut updates: Vec<UpdateRow> = Vec::new();
    let train_start = Instant::now();
    let mut on_progress = |trainer: &mut Trainer, progress: Progress| -> retrograd::Result<bool> {
        // Sample progress events because the optimizer's activation peak is
        // transient; fold in the runtime's in-step measurement for that peak.
        vram.sample();
        vram.fold_runtime(trainer.optimizer_memory()?);
        if let Some(row) = extract_update_row(&progress) {
            phases.add(&row);
            updates.push(row);
        }
        Ok(true)
    };
    let metrics =
        training::grpo::run_resumed(&mut trainer, &grpo, &training, None, &mut on_progress)?;
    let train_time = train_start.elapsed();
    let mem_peak = peak_bytes();
    vram.sample();
    vram.fold_runtime(trainer.optimizer_memory()?);

    let workload = Workload {
        updates: grpo.updates,
        per_update: grpo.prompts_per_update,
        per_update_label: "prompts/upd",
        group_size: grpo.group_size,
        epochs_per_update: grpo.grpo_epochs,
        max_new_tokens: grpo.sampling.max_new_tokens,
    };
    print_report(
        &options,
        &run_config,
        &workload,
        &init,
        &mut trainer,
        &phases,
        &updates,
        &metrics,
        mem_start,
        mem_peak,
        &vram,
        train_time,
        wall,
    )
}

/// Apply optimizer group packing and generation concurrency for either algorithm.
fn apply_micro_batch_and_packing(
    options: &Options,
    run_config: &mut RunConfig,
    group_size: usize,
    per_update: usize,
) -> retrograd::Result<()> {
    if let Some(value) = options.micro_batch {
        // Reject invalid geometry before loading the model. Rollout optimizer
        // windows are fixed at `ctx`, so the micro-batch must divide that value.
        if value == 0 || !run_config.training.n_batch.is_multiple_of(value) {
            return Err(retrograd::Error::invalid(format!(
                "--micro-batch must be a non-zero divisor of the optimizer window: \
                 {value} does not divide {}",
                run_config.training.n_batch,
            )));
        }
        run_config.training.n_ubatch = value;
    }
    run_config.training.n_seq_max = group_size as u32;
    run_config.training.generation_concurrency = per_update
        .saturating_mul(group_size)
        .min(run_config.training.n_batch as usize)
        .min(256) as u32;
    if run_config.training.n_batch < group_size as u32 {
        run_config.training.n_batch = group_size as u32;
    }
    Ok(())
}

/// Everything measured before a single optimizer update: model + context load,
/// LoRA adapter creation, and the preflight that builds the backward graph
/// once. Identical for every algorithm, since none of it depends on how the
/// updates themselves are driven.
pub(crate) struct Init {
    pub load_time: std::time::Duration,
    pub lora_time: std::time::Duration,
    pub preflight_time: std::time::Duration,
    pub mem_after_model: u64,
    pub mem_after_lora: u64,
    pub vram_after_model: Option<u64>,
    pub vram_after_lora: Option<u64>,
    pub vram_after_preflight: Option<u64>,
    pub backend: String,
    pub backend_after: String,
    pub memory_after: MemoryReport,
}

pub(crate) fn init_trainer(
    run_config: &RunConfig,
    vram: &mut VramTrack,
) -> retrograd::Result<(Trainer, Init)> {
    // --- Phase: model + context load. ---
    let t = Instant::now();
    let mut trainer = Trainer::new(&run_config.model, run_config.training.clone())?;
    let load_time = t.elapsed();
    let mem_after_model = snapshot_bytes();
    let vram_after_model = vram.sample();

    // --- Phase: LoRA adapter creation. ---
    let t = Instant::now();
    trainer.create_lora(&run_config.lora.config)?;
    let lora_time = t.elapsed();
    let mem_after_lora = snapshot_bytes();
    let vram_after_lora = vram.sample();

    // Read the backend report after adapter creation so its dtype fields are real.
    let backend = trainer.backend_report()?;

    // --- Phase: preflight (builds the backward graph once). ---
    let t = Instant::now();
    let _ = trainer.train_preflight()?;
    let preflight_time = t.elapsed();
    let vram_after_preflight = vram.sample();

    // Preflight reserves compute buffers; use the post-preflight report so their
    // `compute` bytes are included.
    let backend_after = trainer.backend_report()?;
    // The component table reads byte totals from this structured snapshot.
    let memory_after = trainer.memory_report()?;

    Ok((
        trainer,
        Init {
            load_time,
            lora_time,
            preflight_time,
            mem_after_model,
            mem_after_lora,
            vram_after_model,
            vram_after_lora,
            vram_after_preflight,
            backend,
            backend_after,
            memory_after,
        },
    ))
}

/// One `UpdateRow` out of a progress event's `timing/*` / `scoring/*` values,
/// or `None` when the event carries neither - a mid-update sample, not an
/// update boundary.
pub(crate) fn extract_update_row(progress: &Progress) -> Option<UpdateRow> {
    let timing: Vec<_> = progress
        .values
        .iter()
        .filter(|value| value.name.starts_with("timing/") || value.name.starts_with("scoring/"))
        .collect();
    if timing.is_empty() {
        return None;
    }
    let mut row = UpdateRow {
        step: progress.metrics.global_step,
        loss: progress.metrics.train_loss,
        tok_s: progress.metrics.tokens_per_second,
        ..UpdateRow::default()
    };
    for value in timing {
        match value.name.as_ref() {
            "timing/sampling_seconds" => row.sampling = value.value,
            "timing/generation_seconds" => row.generation = value.value,
            "timing/behavior_scoring_seconds" => row.behavior_scoring = value.value,
            "timing/reward_seconds" => row.reward = value.value,
            "timing/reference_seconds" => row.reference = value.value,
            "timing/optimizer_seconds" => row.optimizer = value.value,
            "timing/optimizer_graph_build_seconds" => row.opt_graph_build = value.value,
            "timing/optimizer_allocation_seconds" => row.opt_allocation = value.value,
            "timing/optimizer_execution_seconds" => row.opt_execution = value.value,
            // Agentic GRPO reports rollout, judge, and optimizer work directly;
            // its KL computation is part of the optimizer batch.
            "timing/rollout_seconds" => row.sampling = value.value,
            "timing/judge_seconds" => row.reward = value.value,
            "timing/optimizer_wall_seconds" => row.optimizer = value.value,
            "scoring/prefix_decodes_per_group" => {
                row.scoring_prefix_decodes_per_group = value.value
            }
            "scoring/device_logprob_fraction" => row.scoring_device_logprob_fraction = value.value,
            _ => {}
        }
    }
    Some(row)
}

/// The report section, shared by every algorithm: the checkpoint gates read
/// from the trainer after training, and the nine tables plus the footer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn print_report(
    options: &Options,
    run_config: &RunConfig,
    workload: &Workload,
    init: &Init,
    trainer: &mut Trainer,
    phases: &PhaseTotals,
    updates: &[UpdateRow],
    metrics: &retrograd::TrainMetrics,
    mem_start: u64,
    mem_peak: u64,
    vram: &VramTrack,
    train_time: std::time::Duration,
    wall: Instant,
) -> retrograd::Result<()> {
    // Read checkpoint and device-peak data after training; both depend on the
    // completed backward graph and observed optimizer allocations.
    let memory_final = trainer.memory_report()?;
    // Measure transfer rates at the long-lived checkpoint size used by the
    // offload decision. No device means no transfer measurement.
    let transfer_rates = (memory_final.checkpoint_long_lived_bytes > 0)
        .then(|| {
            retrograd::transfer_probe(
                memory_final.checkpoint_long_lived_bytes as usize,
                TRANSFER_PROBE_ITERATIONS,
            )
            .ok()
        })
        .flatten();
    // Compare one checkpoint round trip with one optimizer step; execution time
    // is cumulative over the run.
    let backward_seconds = if metrics.global_step > 0 {
        f64::from(phases.opt_execution) / metrics.global_step as f64
    } else {
        0.0
    };

    print_header(options, run_config, workload, &init.backend);
    print_init_table(init, mem_start, vram);
    print_phase_table(phases, train_time);
    // Read after training: the counters are lifetime totals and only the final
    // value describes the whole run.
    print_duty_cycle_line(&trainer.duty_cycle_stats()?);
    print_optimizer_timing_table(phases);
    print_updates_table(updates);
    print_component_memory_table(&init.memory_after, &init.backend_after);
    print_buffer_memory_table(&init.backend_after);
    print_checkpoint_table(&memory_final, transfer_rates.as_ref(), backward_seconds);
    print_memory_table(
        mem_start,
        init.mem_after_model,
        init.mem_after_lora,
        mem_peak,
    );
    print_vram_table(vram, run_config.training.n_ubatch);
    print_footer(metrics, train_time, wall.elapsed());
    if phases.optimizer < 1e-3 {
        println!(
            "\n  \x1b[33m⚠ the optimizer never ran: every group was signal-less\n  \
             (reward without variance) or truncated. Raise --prompts / --group-size\n  \
             to get variance, or point at a more discriminating prompt.\x1b[0m"
        );
    }
    Ok(())
}

/// Round trips timed per direction. Enough that the per-copy fixed cost is not
/// what is being measured, few enough that the probe stays under a second even on
/// a slow link at the sizes a checkpoint set reaches.
const TRANSFER_PROBE_ITERATIONS: u32 = 16;

#[derive(Default, Clone, Copy)]
pub(crate) struct UpdateRow {
    step: u64,
    loss: f32,
    tok_s: f32,
    sampling: f32,
    /// `sampling` split: incremental decode of the rollouts.
    generation: f32,
    /// `sampling` split: teacher-forced re-scoring of the behavior policy.
    behavior_scoring: f32,
    reward: f32,
    reference: f32,
    optimizer: f32,
    // Per-update split of `llama_opt_timing`; these durations sum to `optimizer`.
    opt_graph_build: f32,
    opt_allocation: f32,
    opt_execution: f32,
    /// Shared-prefix decodes per scored group: 1.0 when every branch reuses the
    /// one prefix decode, `group_size` when the scorer fell back to a full
    /// prefill per completion.
    scoring_prefix_decodes_per_group: f32,
    /// Share of scored positions whose target logprob was gathered on the
    /// device instead of reduced from a full vocabulary row on the host (H5).
    scoring_device_logprob_fraction: f32,
}

#[derive(Default)]
pub(crate) struct PhaseTotals {
    sampling: f32,
    generation: f32,
    behavior_scoring: f32,
    reward: f32,
    reference: f32,
    optimizer: f32,
    opt_graph_build: f32,
    opt_allocation: f32,
    opt_execution: f32,
    // Behavior-scoring shape, averaged over the updates that reported it.
    scoring_prefix_decodes_per_group: f32,
    scoring_device_logprob_fraction: f32,
    scoring_updates: f32,
}

impl PhaseTotals {
    fn add(&mut self, row: &UpdateRow) {
        self.sampling += row.sampling;
        self.generation += row.generation;
        self.behavior_scoring += row.behavior_scoring;
        self.reward += row.reward;
        self.reference += row.reference;
        self.optimizer += row.optimizer;
        // Sum the per-update split because skipped updates contribute zero and
        // the three components describe the same optimizer duration.
        self.opt_graph_build += row.opt_graph_build;
        self.opt_allocation += row.opt_allocation;
        self.opt_execution += row.opt_execution;
        self.scoring_prefix_decodes_per_group += row.scoring_prefix_decodes_per_group;
        self.scoring_device_logprob_fraction += row.scoring_device_logprob_fraction;
        self.scoring_updates += 1.0;
    }
    fn accounted(&self) -> f32 {
        self.sampling + self.reward + self.reference + self.optimizer
    }
}

fn parse_args() -> retrograd::Result<Options> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut options = Options::default();
    let mut args = Args::new("profile", &raw);
    let mut positional_config = None;
    while let Some(arg) = args.next_arg() {
        match arg {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--model" => options.model = Some(std::path::PathBuf::from(args.value(arg)?)),
            "--device" => options.device = Some(args.value(arg)?.parse()?),
            "--updates" => options.updates = Some(args.parse(arg, "an integer")?),
            "--group-size" => options.group_size = Some(args.parse(arg, "an integer")?),
            "--epochs" => options.epochs = Some(args.parse(arg, "an integer")?),
            "--prompts" => options.prompts_per_update = Some(args.parse(arg, "an integer")?),
            "--micro-batch" => options.micro_batch = Some(args.parse(arg, "an integer")?),
            "--max-new-tokens" => options.max_new_tokens = Some(args.parse(arg, "an integer")?),
            "--full" => options.full = true,
            other if other.starts_with('-') => return Err(args.unknown(other)),
            config => positional_config = Some(config.to_owned()),
        }
    }
    if let Some(path) = positional_config {
        options.config_path = path;
    }
    Ok(options)
}

fn print_help() {
    println!(
        "retrograd profile - end-to-end GRPO / agent_grpo training profiler

USAGE:
  cargo run --bin profile -- [CONFIG] [OPTIONS]

ARGS:
  CONFIG                 GRPO or agent_grpo config TOML (default: examples/smoke_tiny_grpo.toml)

OPTIONS:
  --model <path.gguf>       Override the model path
  --device <auto|cpu|gpu>   Override the training device
  --updates <N>             Number of updates        (fast default: 1)
  --group-size <N>          Group size               (fast default: 4)
  --epochs <N>              Epochs per update        (fast default: 1)
  --prompts <N>             Prompts/scenarios per update (fast default: 3)
  --micro-batch <N>         Override training.micro_batch (must divide the step window)
  --max-new-tokens <N>      Sampling budget          (default: from config)
  --full                    Use the config as-is, no fast overrides
  -h, --help                Show this help

Run from the repository root so the reward command and, for agent_grpo, the
environment/scenarios resolve."
    );
}

// --- Memory helpers (host RSS; on Linux this excludes VRAM). ---

/// Device-budget readings sampled at phase boundaries and progress events.
/// `used` is device-wide, so readings are stored relative to a pre-allocation
/// baseline.
pub(crate) struct VramTrack {
    baseline: Option<u64>,
    total: u64,
    peak: u64,
    /// Peak measured inside an optimizer step; boundary sampling cannot see the
    /// allocation and backward transient.
    runtime_peak: u64,
    /// Backend-owned scratch (CUDA pool, Vulkan `prealloc_*`) inside that peak.
    /// Unlike the device budget, it is attributable to this process.
    runtime_scratch_peak: u64,
}

impl VramTrack {
    fn new() -> Self {
        let start = memory::device_snapshot();
        Self {
            baseline: start.map(memory::DeviceMemory::used),
            total: start.map_or(0, |device| device.total),
            peak: start.map_or(0, memory::DeviceMemory::used),
            runtime_peak: 0,
            runtime_scratch_peak: 0,
        }
    }

    /// Folds in the runtime's own in-step measurement. Both fields are running
    /// maxima on the runtime side, so this is idempotent.
    fn fold_runtime(&mut self, memory: retrograd::OptimizerMemory) {
        if !memory.is_measured() {
            return;
        }
        self.runtime_peak = self.runtime_peak.max(memory.device_peak_used_bytes);
        self.runtime_scratch_peak = self.runtime_scratch_peak.max(memory.scratch_peak_bytes);
        self.peak = self.peak.max(memory.device_peak_used_bytes);
    }

    /// Read the device budget, fold it into the peak, and return usage relative
    /// to the baseline. `None` without a GPU device.
    fn sample(&mut self) -> Option<u64> {
        let used = memory::device_snapshot()?.used();
        self.peak = self.peak.max(used);
        Some(used.saturating_sub(self.baseline.unwrap_or(0)))
    }

    fn available(&self) -> bool {
        self.baseline.is_some()
    }

    /// Peak usage relative to the baseline, used by `n_ubatch` sweeps.
    fn peak_above_baseline(&self) -> u64 {
        self.peak.saturating_sub(self.baseline.unwrap_or(0))
    }
}

/// Current process footprint, zero when the platform has no reading - a
/// missing measurement must not stop a profiling run.
fn snapshot_bytes() -> u64 {
    memory::snapshot().map(|s| s.footprint).unwrap_or(0)
}

fn peak_bytes() -> u64 {
    memory::snapshot().map(|s| s.peak).unwrap_or(0)
}
