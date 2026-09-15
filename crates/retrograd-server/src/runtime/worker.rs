//! The worker: one thread per run, and the observer that turns what the run
//! says into what the journal holds.
//!
//! The supervising half is a tokio task and the working half is an OS thread.
//! That split is not incidental:
//!
//! - the **thread** exists because a `Trainer` never crosses a thread boundary,
//!   so the model, the optimizer and every FFI pointer stay on one thread
//!   from first to last;
//! - the **task** exists because waiting for the device is asynchronous. A run
//!   whose turn has not come must not occupy a thread, and the queue must be
//!   fair: `Semaphore::acquire_owned` hands permits out in request order, which
//!   is the required FIFO.

use std::collections::BTreeMap;
use std::sync::Arc;

use retrograd_config::RunConfig;
use retrograd_metrics::{MetricEvent, MetricsSink};
use retrograd_run::{EvaluationReport, LoopPlan, RolloutEpoch, RunObserver, SftEpoch, SftStep};
use tokio::sync::Semaphore;
use tokio::sync::mpsc;

use super::control::{ChannelControl, RunCommand};
use super::engine::RunEngine;
use super::registry::RunHandle;
use crate::dto;

/// Queues a run and drives it to a terminal state.
///
/// Returns immediately; everything after the queueing happens in the background.
/// The caller has already answered `201`, and a run is observed through its
/// state and its events, not through the request that created it.
pub fn spawn(
    engine: Arc<dyn RunEngine>,
    device: Arc<Semaphore>,
    handle: Arc<RunHandle>,
    config: Box<RunConfig>,
    commands: mpsc::UnboundedReceiver<RunCommand>,
) {
    tokio::spawn(async move {
        // `queued` until a permit is free. With `max_concurrent_runs = 1` - the
        // default - this is where a second run waits for the first, rather than
        // being refused: a queue is what a caller can plan around, a 503 is not.
        let permit = match device.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                handle.fail("the device queue was closed before the run started");
                return;
            }
        };
        if handle.status().is_terminal() {
            // Cancelled while queued (step 7). Nothing was loaded, so there is
            // nothing to unwind.
            return;
        }
        handle.transition(dto::RunStatus::Starting);

        let worker_handle = handle.clone();
        let thread = std::thread::Builder::new()
            .name(format!("retrograd-run-{}", handle.id))
            .spawn(move || {
                let mut observer = JournalObserver::new(worker_handle.clone());
                let mut control = ChannelControl::new(commands, worker_handle.clone());
                let sinks: Vec<Box<dyn MetricsSink>> =
                    vec![Box::new(EventSink::new(worker_handle))];
                let outcome = engine.execute(&config, &mut observer, &mut control, sinks);
                (outcome, control.was_cancelled())
            });
        let thread = match thread {
            Ok(thread) => thread,
            Err(error) => {
                handle.fail(format!("could not start the run's thread: {error}"));
                return;
            }
        };

        // Joining is blocking work, so it goes to the blocking pool: parking an
        // async worker on a training run would starve every other request.
        let outcome = tokio::task::spawn_blocking(move || thread.join()).await;
        drop(permit);

        match outcome {
            Ok(Ok((Ok(result), cancelled))) => {
                if result.early_stopped {
                    handle.emit(dto::RunEventPayload::Log {
                        message: "early stopping: patience exhausted".to_string(),
                    });
                }
                // Only the step count. The loss fields are left as the last
                // epoch reported them: `TrainMetrics` initializes `eval_loss` to
                // zero, so copying it here would report "0.0" as an evaluation
                // result on a run that never evaluated.
                handle.progress(|progress| {
                    progress.global_step = result.metrics.global_step;
                });
                // A cancelled run ends cleanly - the loop was asked to stop, so
                // `execute` returns `Ok` and the adapter is saved - but it did
                // not do what it was created to do, and reporting it `completed`
                // would tell a client its run finished.
                if cancelled {
                    handle.cancelled();
                } else {
                    handle.complete();
                }
            }
            // The run reported a failure: its own message is the useful one.
            Ok(Ok((Err(error), _))) => handle.fail(error.to_string()),
            // The thread panicked. Not expected, and precisely why the join is
            // inspected instead of assumed: a panicking worker must leave a
            // `failed` run, not one stuck in `running` forever.
            Ok(Err(panic)) => {
                handle.fail(format!("the run's thread panicked: {}", panic_text(&panic)))
            }
            Err(error) => handle.fail(format!("the run's thread could not be joined: {error}")),
        }
    });
}

fn panic_text(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Turns [`RunObserver`] callbacks into journal events and progress.
///
/// The mapping is deliberately lossy in one direction: the multi-line
/// diagnostics (`backend report`, `describe_lora`, the preflight) are *not*
/// forwarded. They are pages of text produced once per run for a human reading a
/// terminal; putting them in the event stream would drown the events a client
/// polls, and `POST /v1/preflight` already serves them to anyone who wants them.
struct JournalObserver {
    handle: Arc<RunHandle>,
}

impl JournalObserver {
    fn new(handle: Arc<RunHandle>) -> Self {
        Self { handle }
    }

    fn log(&self, message: impl Into<String>) {
        self.handle.emit(dto::RunEventPayload::Log {
            message: message.into(),
        });
    }
}

impl RunObserver for JournalObserver {
    fn info(&mut self, message: &str) {
        self.log(message);
    }

    fn model_load_started(&mut self) {
        self.log("loading the model");
    }

    fn model_load_failed(&mut self) {
        // The error itself comes back through `execute`; this only records that
        // the run died at load time rather than in the loop.
        self.log("the model could not be loaded");
    }

    fn model_load_finished(&mut self, elapsed: std::time::Duration) {
        self.log(format!("model loaded in {:.1}s", elapsed.as_secs_f64()));
    }

    fn loop_started(&mut self, plan: &LoopPlan) {
        // The loop starting is the moment the run stops setting up and starts
        // training, which is what `running` means to a client.
        self.handle.transition(dto::RunStatus::Running);
        let iterations = match plan {
            LoopPlan::Sft { epochs, .. } => *epochs as u64,
            LoopPlan::Rollout { updates, .. } => *updates as u64,
        };
        self.handle.progress(|progress| {
            progress.iterations = iterations;
        });
    }

    fn sft_epoch(&mut self, epoch: &SftEpoch) {
        self.handle.progress(|progress| {
            progress.iteration = epoch.epoch as u64;
            progress.iterations = epoch.total_epochs as u64;
            progress.global_step = epoch.global_step;
            progress.train_loss = dto::finite(epoch.train_loss);
            // `NaN` is the observer's "not evaluated this epoch"; keeping the
            // previous value would report a stale loss as a fresh one.
            progress.eval_loss = dto::finite(epoch.eval_loss);
            progress.learning_rate = dto::finite(epoch.learning_rate);
            progress.tokens_per_second = dto::finite(epoch.tokens_per_second);
        });
    }

    /// Mid-epoch progress, without an event: an SFT epoch is long enough that a
    /// polling client would otherwise see a frozen `global_step` for its whole
    /// duration. `iteration` stays on the completed epochs, and `eval_loss`
    /// keeps whatever the last evaluation put there - nothing measured it here.
    fn sft_step(&mut self, step: &SftStep) {
        self.handle.progress(|progress| {
            progress.iterations = step.total_epochs as u64;
            progress.global_step = step.global_step;
            progress.train_loss = dto::finite(step.train_loss);
            progress.learning_rate = dto::finite(step.learning_rate);
            progress.tokens_per_second = dto::finite(step.tokens_per_second);
        });
    }

    fn rollout_epoch(&mut self, epoch: &RolloutEpoch) {
        self.handle.progress(|progress| {
            progress.iteration = epoch.update;
            progress.iterations = epoch.updates as u64;
            progress.global_step = epoch.global_step;
            progress.train_loss = dto::finite(epoch.train_loss);
            progress.reward = dto::finite(epoch.reward);
            progress.learning_rate = dto::finite(epoch.learning_rate);
            progress.tokens_per_second = dto::finite(epoch.tokens_per_second);
        });
    }

    fn evaluation(&mut self, report: &EvaluationReport) {
        let payload = match report {
            EvaluationReport::Sft {
                epoch,
                loss,
                perplexity,
                outcome,
            } => dto::RunEventPayload::Evaluation {
                iteration: *epoch as u64,
                loss: Some(*loss),
                perplexity: Some(*perplexity),
                mean_reward: None,
                improved: outcome.improved,
                best: outcome.best,
                stale: outcome.stale,
                keep_training: outcome.keep_training,
            },
            EvaluationReport::Rollout {
                update,
                mean_reward,
                outcome,
                ..
            } => dto::RunEventPayload::Evaluation {
                iteration: *update,
                loss: None,
                perplexity: None,
                mean_reward: dto::finite(*mean_reward),
                improved: outcome.improved,
                best: outcome.best,
                stale: outcome.stale,
                keep_training: outcome.keep_training,
            },
        };
        self.handle.emit(payload);
    }

    fn checkpoint_written(&mut self, path: &std::path::Path) {
        self.handle.emit(dto::RunEventPayload::Checkpoint {
            path: path.display().to_string(),
        });
    }

    fn memory_note(&mut self, note: &str) {
        self.handle.emit(dto::RunEventPayload::Memory {
            note: note.to_string(),
        });
    }

    fn loop_finished(&mut self) {
        self.log("training loop finished");
    }
}

/// One more `MetricsSink` on the run's existing bus.
///
/// Deliberately *additional*: TensorBoard and the W&B export receive exactly what
/// they received before, and the HTTP stream is a third destination rather than a
/// reimplementation. Every metric a client can pull or stream is therefore the
/// same number a training curve was drawn from.
///
/// Only `Step` is forwarded. `RunStarted`, `RunFinished` and `RunFailed` describe
/// a lifecycle the run already reports through `status` and `terminal` events,
/// and sending them twice would make a client choose which copy to believe.
struct EventSink {
    handle: Arc<RunHandle>,
}

impl EventSink {
    fn new(handle: Arc<RunHandle>) -> Self {
        Self { handle }
    }
}

impl MetricsSink for EventSink {
    fn emit(&mut self, event: &MetricEvent) -> retrograd_core::Result<()> {
        let MetricEvent::Step {
            epoch,
            global_step,
            values,
        } = event
        else {
            return Ok(());
        };
        // A `BTreeMap` so the rendered bytes are stable, and non-finite values
        // dropped rather than written: `serde_json` renders a `NaN` as `null`,
        // which would be the one place in the API where absent and null disagree
        let values: BTreeMap<String, f32> = values
            .iter()
            .filter(|value| value.value.is_finite())
            .map(|value| (value.name.to_string(), value.value))
            .collect();
        if values.is_empty() {
            return Ok(());
        }
        self.handle.emit(dto::RunEventPayload::Metrics {
            iteration: *epoch as u64,
            global_step: *global_step,
            values,
        });
        Ok(())
    }
}
