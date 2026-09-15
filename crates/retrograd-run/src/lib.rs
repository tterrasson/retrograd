//! Run orchestration, above `retrograd-training` and below every frontend.
//!
//! This crate owns what a run *is*: load the model, create or restore the
//! adapter, wire the metrics bus, drive the algorithm, evaluate and checkpoint
//! on the schedule, and save the adapter. It owns none of how a run is *shown*:
//! progress is exposed as events on [`RunObserver`], so the terminal and the
//! HTTP server observe the same run.
//!
//! The frontend keeps: argument or request parsing, rendering, and the final
//! summary line. Nothing else.

#[cfg(feature = "agent")]
mod agent;
#[cfg(feature = "agent")]
pub use agent::evaluate_judge;
mod algorithm;
mod control;
mod controller;
#[cfg(feature = "agent")]
mod interrupt;
mod observer;
mod resume;
mod signature;

use std::time::Instant;

use retrograd_config::{Algorithm, RunConfig};
use retrograd_core::{Error, MemoryReport, Result, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_memory::{self as memory, MemoryTracker};
use retrograd_metrics::{MetricEvent, MetricsBus, MetricsSink, TensorBoardSink, WandbExportSink};

pub use control::{
    AdHocEvaluation, ControlPoint, Flow, FreeRunning, GenerationOutput, GenerationRequest,
    RunControl, RunControls,
};
pub use controller::{EvalDirection, RunController};
pub use observer::{
    EvalOutcome, EvaluationReport, LoopPlan, RolloutEpoch, RunObserver, SftEpoch, SftStep,
    SilentObserver,
};
pub use resume::{apply_resume_override, latest_checkpoint};
pub use signature::{
    checked_total_steps, dataset_fingerprint, metadata, model_bytes, prompts_dataset,
    scheduler_name, trajectory_signature,
};

/// The trainer, moved out for the duration of an algorithm that owns it.
///
/// Every synchronous algorithm borrows it; the agentic loop hands it to a
/// policy actor and hands it back, which is the one case the `Option` exists
/// for. `as_mut` panics rather than returning an error because the borrow-only
/// path never empties the slot.
struct TrainerSlot(Option<Trainer>);

impl TrainerSlot {
    fn as_mut(&mut self) -> &mut Trainer {
        self.0
            .as_mut()
            .expect("the trainer is only moved out by the agentic algorithm, which puts it back")
    }

    #[cfg(feature = "agent")]
    fn take(&mut self) -> Trainer {
        self.0
            .take()
            .expect("the trainer is taken exactly once, by the algorithm that owns it")
    }

    #[cfg(feature = "agent")]
    fn restore(&mut self, trainer: Option<Trainer>) {
        self.0 = trainer;
    }
}

/// What a finished run leaves behind for the frontend to report.
#[derive(Clone, Copy, Debug)]
pub struct RunOutcome {
    pub metrics: TrainMetrics,
    /// The run stopped because evaluation patience ran out, not because it
    /// reached its last iteration.
    pub early_stopped: bool,
}

/// Runs `config` to completion, reporting progress to `observer`.
///
/// This is the body of the former `src/train.rs::train`, minus the terminal
/// chrome: the frontend prints its own header before the call and its own
/// summary after it. Nothing interrupts it and its metrics go where the
/// configuration says - which is exactly what the CLI wants.
pub fn execute(config: &RunConfig, observer: &mut dyn RunObserver) -> Result<RunOutcome> {
    execute_controlled(config, observer, &mut FreeRunning, Vec::new())
}

/// [`execute`], plus the two seams a control plane needs.
///
/// `control` is polled at every progress callback - the only safe interruption
/// point - and may pause, stop, or adjust the schedule. `sinks` are added to
/// the metrics bus alongside the ones the configuration declares, so a server can
/// watch the same values TensorBoard receives without the run knowing it exists.
pub fn execute_controlled(
    config: &RunConfig,
    observer: &mut dyn RunObserver,
    control: &mut dyn RunControl,
    sinks: Vec<Box<dyn MetricsSink>>,
) -> Result<RunOutcome> {
    let span = tracing::info_span!(target: "retrograd::run", "run");
    let _entered = span.enter();
    let verbose = config.training.verbose;
    let started = Instant::now();
    let mut tracker = MemoryTracker::new();

    observer.model_load_started();
    let mut trainer = Trainer::new(&config.model, config.training.clone()).inspect_err(|_| {
        observer.model_load_failed();
    })?;
    observer.model_load_finished(started.elapsed());
    if let Some(line) = tracker.phase("model + context") {
        observer.info(&line);
    }

    // A resume owns its adapter: the checkpoint restores it together with the
    // optimizer state, once the dataset it was taken on has been validated.
    let resume_from = config
        .checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.resume_from.as_ref());
    match (resume_from, &config.lora.init_adapter) {
        (Some(path), _) => observer.info(&format!("resuming checkpoint: {}", path.display())),
        (None, Some(path)) => {
            observer.info(&format!("resuming adapter: {}", path.display()));
            trainer.load_lora(path)?;
        }
        (None, None) => trainer.create_lora(&config.lora.config)?,
    }
    if let Some(line) = tracker.phase("lora adapter") {
        observer.info(&line);
    }
    // Report after the adapter exists: lora_dtype and optimizer_f16 describe
    // the effective training state, not their pre-LoRA defaults.
    observer.info(&memory_breakdown_line(&trainer.memory_report()?));
    if verbose {
        observer.diagnostic("backend", &trainer.backend_report()?);
    }
    if verbose && resume_from.is_none() {
        observer.diagnostic("before training", &trainer.describe_lora()?);
        observer.diagnostic("preflight", &trainer.train_preflight()?);
    }

    let metadata = signature::metadata(config);
    let mut bus = MetricsBus::new();
    if let Some(path) = &config.metrics.tensorboard_dir {
        bus.add(TensorBoardSink::new(path)?);
    }
    if let Some(path) = &config.metrics.wandb_export_dir {
        bus.add(WandbExportSink::new(path, &metadata)?);
    }
    for sink in sinks {
        bus.add_boxed(sink);
    }
    bus.emit(&MetricEvent::RunStarted { metadata })?;
    let mut controller = RunController::new(config)?;
    let mut trainer = TrainerSlot(Some(trainer));
    let result = {
        let mut ctx = algorithm::Context {
            config,
            bus: &mut bus,
            controller: &mut controller,
            memory: &mut tracker,
            observer,
            control,
        };
        match &config.algorithm {
            Algorithm::Sft(sft) => algorithm::run_sft(trainer.as_mut(), sft, &mut ctx),
            Algorithm::Ppo(ppo) => algorithm::run_ppo(trainer.as_mut(), ppo, &mut ctx),
            Algorithm::Grpo(grpo) => algorithm::run_grpo(trainer.as_mut(), grpo, &mut ctx),
            Algorithm::Distill(distill) => {
                algorithm::run_distill(trainer.as_mut(), distill, &mut ctx)
            }
            // The agentic loop owns the trainer while it runs - the policy actor
            // holds it on its own task - so it is handed over and handed back
            // rather than borrowed.
            #[cfg(feature = "agent")]
            Algorithm::AgentGrpo(config) => {
                let outcome = agent::run_agent_grpo(trainer.take(), config, &mut ctx);
                trainer.restore(outcome.trainer);
                outcome.result
            }
            // The section still parses without the feature, so the answer is
            // "this build cannot run it" rather than "unknown field".
            #[cfg(not(feature = "agent"))]
            Algorithm::AgentGrpo(_) => Err(Error::config(
                "run.algorithm = 'agent_grpo' needs the agentic loop, which is not compiled \
                 into this binary: rebuild with `--features agent`",
            )),
        }
    };
    let metrics = match result {
        Ok(metrics) => metrics,
        Err(error) => {
            let _ = bus.emit(&MetricEvent::RunFailed {
                message: error.to_string(),
            });
            return Err(error);
        }
    };
    // Only an agentic run can come back without its trainer, and only when the
    // policy actor died holding it. There is then no adapter to save, and
    // saying so beats panicking on the slot.
    let mut trainer = trainer.0.ok_or_else(|| {
        Error::runtime("the policy actor failed before returning the trainer: nothing to save")
    })?;
    bus.emit(&MetricEvent::RunFinished {
        global_step: metrics.global_step,
    })?;
    if controller.early_stopped() {
        observer.info("early stopping: patience exhausted");
    }
    if verbose {
        observer.diagnostic("after training", &trainer.describe_lora()?);
    }
    if let Some(summary) = tracker.summary() {
        observer.diagnostic("memory", &summary);
    }
    trainer.save_lora(&config.lora.output)?;
    Ok(RunOutcome {
        metrics,
        early_stopped: controller.early_stopped(),
    })
}

/// One-line memory breakdown: the static allocations known before training
/// (weights, KV caches, LoRA state) and the device/host totals. Always emitted,
/// not only on `--verbose` runs, so every run says where memory goes.
///
/// Compute buffers are omitted: they are transient and read zero before the
/// preflight graph is reserved, which is where this line is printed.
pub fn memory_breakdown_line(report: &MemoryReport) -> String {
    let bytes = memory::format_bytes;
    let mut parts = vec![
        format!("model {}", bytes(report.model_weight_bytes)),
        format!("optim KV {}", bytes(report.optimizer_kv_bytes)),
    ];
    if report.has_generation_context {
        parts.push(format!("gen KV {}", bytes(report.generation_kv_bytes)));
    }
    parts.push(format!("LoRA+optim {}", bytes(report.lora_total_bytes())));
    parts.push(format!(
        "\u{2192} device {} / host {}",
        bytes(report.device_bytes),
        bytes(report.host_bytes)
    ));
    format!("memory: {}", parts.join(", "))
}

#[cfg(test)]
mod tests_support {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub fn temp_path(label: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "retrograd-run-{label}-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    pub fn write_sft_config(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let config = root.join("run.toml");
        fs::write(
            &config,
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        )
        .unwrap();
        config
    }
}
