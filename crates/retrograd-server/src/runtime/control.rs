//! The command channel between a handler and a live run.
//!
//! A handler runs on a tokio worker; a run owns an OS thread it never leaves.
//! Everything one wants to ask of the other crosses this channel, and the run
//! reads it **from inside its progress callback** - the only point at which
//! stopping is safe. Nothing here kills a thread, and nothing here touches
//! the `Trainer` from the outside.
//!
//! The channel is unbounded on purpose. It carries control messages, which are
//! human-paced and tiny; a bounded one would make a handler either block on a
//! run that is mid-generation or drop a cancellation, and both are worse than
//! holding a handful of enum values.

use std::sync::Arc;

use retrograd_config::CheckpointMode;
use retrograd_core::Result as CoreResult;
use retrograd_run::{
    AdHocEvaluation, ControlPoint, Flow, GenerationOutput, GenerationRequest, RunControl,
    RunControls,
};
use tokio::sync::{mpsc, oneshot};

use super::registry::RunHandle;
use crate::dto;

/// When a cancellation takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelAt {
    /// At the next resumable point. The current epoch or update finishes, so a
    /// checkpoint can be written and no work is lost.
    Boundary,
    /// At the very next progress callback. Faster, and the run stops between two
    /// boundaries, which is why it cannot checkpoint.
    Now,
}

/// The whitelist of the contract, as data.
///
/// Every field is a schedule. There is deliberately no way to express `n_ctx`, a
/// LoRA rank, an algorithm or a path here: the type is the whitelist, so a field
/// outside it cannot be smuggled in as a string.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adjustments {
    pub learning_rate: Option<f32>,
    pub evaluation_every_iterations: Option<u32>,
    pub evaluation_patience: Option<u32>,
    pub checkpoint_every_steps: Option<u64>,
    pub checkpoint_mode: Option<CheckpointMode>,
}

impl Adjustments {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The dotted document paths this set touches, in a fixed order, for the
    /// response and the run's log.
    pub fn paths(&self) -> Vec<&'static str> {
        let mut paths = Vec::new();
        if self.learning_rate.is_some() {
            paths.push("training.lr");
        }
        if self.evaluation_every_iterations.is_some() {
            paths.push("evaluation.every_iterations");
        }
        if self.evaluation_patience.is_some() {
            paths.push("evaluation.patience");
        }
        if self.checkpoint_every_steps.is_some() {
            paths.push("checkpoint.every_steps");
        }
        if self.checkpoint_mode.is_some() {
            paths.push("checkpoint.mode");
        }
        paths
    }
}

/// Where the run was when it answered, so a reply is locatable in the run's
/// history rather than being a number out of context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub iteration: u64,
    pub global_step: u64,
}

impl From<ControlPoint> for At {
    fn from(point: ControlPoint) -> Self {
        Self {
            iteration: point.iteration as u64,
            global_step: point.global_step,
        }
    }
}

/// The answering half of a request-shaped command.
///
/// `evaluate` and `generate` are the only two commands with a result, and the
/// only two a handler waits on. Everything else on this channel is fire-and-
/// forget, because everything else is observable in the run's state.
type Reply<T> = oneshot::Sender<CoreResult<(At, T)>>;

/// What a handler asks of a run.
pub enum RunCommand {
    Pause,
    Resume,
    Cancel {
        at: CancelAt,
        checkpoint: bool,
    },
    /// Write a checkpoint at the next boundary.
    Checkpoint,
    Adjust(Adjustments),
    /// Run the configured evaluation dataset now, out of schedule.
    Evaluate(Reply<AdHocEvaluation>),
    /// Sample once against the adapter as it is at the next callback.
    Generate(Box<GenerationRequest>, Reply<GenerationOutput>),
}

/// The sending half, held by the registry for the life of the run.
///
/// Cloneable and `Sync`, so several handlers may command the same run without a
/// lock; ordering between two concurrent commands is the channel's, which is the
/// order they were sent.
#[derive(Clone)]
pub struct ControlSender(mpsc::UnboundedSender<RunCommand>);

impl ControlSender {
    /// Queues a command. `false` means the run's thread is gone - it finished or
    /// never started - which a handler turns into a 409 rather than a silent
    /// success.
    pub fn send(&self, command: RunCommand) -> bool {
        self.0.send(command).is_ok()
    }
}

pub fn channel() -> (ControlSender, mpsc::UnboundedReceiver<RunCommand>) {
    let (tx, rx) = mpsc::unbounded_channel();
    (ControlSender(tx), rx)
}

/// The receiving half, living on the run's own thread.
///
/// It is the run's [`RunControl`]: polled once per progress callback, it drains
/// the channel, applies what it finds, blocks while paused, and reports whether
/// the loop should carry on.
pub struct ChannelControl {
    commands: mpsc::UnboundedReceiver<RunCommand>,
    handle: Arc<RunHandle>,
    paused: bool,
    cancel: Option<PendingCancel>,
}

#[derive(Clone, Copy)]
struct PendingCancel {
    at: CancelAt,
    checkpoint: bool,
}

impl ChannelControl {
    pub fn new(commands: mpsc::UnboundedReceiver<RunCommand>, handle: Arc<RunHandle>) -> Self {
        Self {
            commands,
            handle,
            paused: false,
            cancel: None,
        }
    }

    /// Whether this run stopped because it was cancelled rather than because it
    /// ran out of iterations. What tells the supervisor to report `cancelled`
    /// instead of `completed`.
    pub fn was_cancelled(&self) -> bool {
        self.cancel.is_some()
    }

    fn log(&self, message: impl Into<String>) {
        self.handle.emit(dto::RunEventPayload::Log {
            message: message.into(),
        });
    }

    fn apply(&mut self, command: RunCommand, controls: &mut dyn RunControls, at: ControlPoint) {
        match command {
            RunCommand::Pause => self.paused = true,
            RunCommand::Resume => self.paused = false,
            RunCommand::Cancel { at, checkpoint } => {
                // A cancellation is never downgraded: a second one that asked for
                // less than the first would let a retry weaken what was already
                // decided. `now` wins over `boundary`, and a requested checkpoint
                // stays requested.
                let previous = self.cancel.unwrap_or(PendingCancel { at, checkpoint });
                self.cancel = Some(PendingCancel {
                    at: if at == CancelAt::Now || previous.at == CancelAt::Now {
                        CancelAt::Now
                    } else {
                        CancelAt::Boundary
                    },
                    checkpoint: checkpoint || previous.checkpoint,
                });
                // A cancellation releases a pause: otherwise the callback would
                // block forever on a run that has been told to stop.
                self.paused = false;
            }
            RunCommand::Checkpoint => {
                controls.request_checkpoint();
                self.log("checkpoint requested; it lands at the next boundary");
            }
            RunCommand::Adjust(adjustments) => self.adjust(adjustments, controls),
            RunCommand::Evaluate(reply) => {
                // A closed channel means the handler gave up - a client
                // disconnect, or the timeout. Skipping the pass rather than
                // running it for nobody is the difference between a lost request
                // and a lost forward pass over the whole evaluation set.
                if reply.is_closed() {
                    return;
                }
                let outcome = controls.evaluate();
                if outcome.is_ok() {
                    self.log(format!("ad-hoc evaluation at step {}", at.global_step));
                }
                let _ = reply.send(outcome.map(|evaluation| (at.into(), evaluation)));
            }
            RunCommand::Generate(request, reply) => {
                if reply.is_closed() {
                    return;
                }
                let outcome = controls.generate(&request);
                let _ = reply.send(outcome.map(|output| (at.into(), output)));
            }
        }
    }

    /// Applies the whitelist, field by field.
    ///
    /// A field that the run refuses is logged and skipped rather than propagated:
    /// the request was already validated against what this run has, so a
    /// failure here is a race - an evaluation schedule that ended, say - and
    /// failing a training job over a control message would be the wrong trade.
    fn adjust(&mut self, adjustments: Adjustments, controls: &mut dyn RunControls) {
        let mut applied = Vec::new();
        let mut refused = Vec::new();
        let mut record = |path: &'static str, outcome: retrograd_core::Result<()>| match outcome {
            Ok(()) => applied.push(path),
            Err(error) => refused.push(format!("{path} ({error})")),
        };
        if let Some(rate) = adjustments.learning_rate {
            record("training.lr", controls.set_learning_rate(rate));
        }
        if let Some(every) = adjustments.evaluation_every_iterations {
            record(
                "evaluation.every_iterations",
                controls.set_evaluation_every(every),
            );
        }
        if let Some(patience) = adjustments.evaluation_patience {
            record("evaluation.patience", controls.set_patience(Some(patience)));
        }
        if let Some(every) = adjustments.checkpoint_every_steps {
            record(
                "checkpoint.every_steps",
                controls.set_checkpoint_every_steps(every),
            );
        }
        if let Some(mode) = adjustments.checkpoint_mode {
            record("checkpoint.mode", controls.set_checkpoint_mode(mode));
        }
        if !applied.is_empty() {
            self.log(format!("adjusted {}", applied.join(", ")));
        }
        if !refused.is_empty() {
            self.log(format!("could not adjust {}", refused.join(", ")));
        }
    }
}

impl RunControl for ChannelControl {
    fn poll(
        &mut self,
        controls: &mut dyn RunControls,
        at: ControlPoint,
    ) -> retrograd_core::Result<Flow> {
        while let Ok(command) = self.commands.try_recv() {
            self.apply(command, controls, at);
        }

        // A pause is a callback that does not return. The model, the KV caches
        // and the optimizer state stay allocated - the trade the API documents and
        // `holds_device` exposes - and the run resumes on the next command
        // without reloading anything.
        while self.paused {
            // Through `Pausing`, never straight to `Paused`: the transition table
            // is the single source of truth about what a run may do next, and a
            // run paused from `starting` - before any handler saw it running,
            // would otherwise ask for a step the table refuses and stay silently
            // unpaused.
            self.handle.transition(dto::RunStatus::Pausing);
            self.handle.transition(dto::RunStatus::Paused);
            // A paused run still answers `evaluate` and `generate`: the model is
            // loaded and idle, which is the best moment to ask it something.
            match self.commands.blocking_recv() {
                Some(command) => self.apply(command, controls, at),
                // Every sender is gone, so nobody can resume this run. Carrying
                // on is the only outcome that is not a permanent hang.
                None => {
                    self.paused = false;
                    self.log("the control channel closed while paused; resuming");
                }
            }
        }
        if self.handle.status() == dto::RunStatus::Paused {
            self.handle.transition(dto::RunStatus::Running);
        }

        let Some(cancel) = self.cancel else {
            return Ok(Flow::Continue);
        };
        if cancel.at == CancelAt::Boundary && !at.at_boundary {
            return Ok(Flow::Continue);
        }
        if cancel.checkpoint && at.at_boundary {
            controls.request_checkpoint();
        }
        Ok(Flow::Stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recorder standing in for the loop. What makes the state machine above
    /// testable without a GGUF - the point of `RunControls` being a trait.
    #[derive(Default)]
    struct Recorder {
        learning_rate: Option<f32>,
        every_iterations: Option<u32>,
        patience: Option<u32>,
        checkpoints_requested: usize,
        evaluations: usize,
        refuse_evaluation: bool,
    }

    impl RunControls for Recorder {
        fn set_learning_rate(&mut self, learning_rate: f32) -> retrograd_core::Result<()> {
            self.learning_rate = Some(learning_rate);
            Ok(())
        }
        fn set_evaluation_every(&mut self, every: u32) -> retrograd_core::Result<()> {
            if self.refuse_evaluation {
                return Err(retrograd_core::Error::invalid("no evaluation dataset"));
            }
            self.every_iterations = Some(every);
            Ok(())
        }
        fn set_patience(&mut self, patience: Option<u32>) -> retrograd_core::Result<()> {
            self.patience = patience;
            Ok(())
        }
        fn set_checkpoint_every_steps(&mut self, _every: u64) -> retrograd_core::Result<()> {
            Ok(())
        }
        fn set_checkpoint_mode(&mut self, _mode: CheckpointMode) -> retrograd_core::Result<()> {
            Ok(())
        }
        fn request_checkpoint(&mut self) {
            self.checkpoints_requested += 1;
        }
        fn evaluate(&mut self) -> retrograd_core::Result<AdHocEvaluation> {
            self.evaluations += 1;
            if self.refuse_evaluation {
                return Err(retrograd_core::Error::invalid("no evaluation dataset"));
            }
            Ok(AdHocEvaluation {
                loss: Some(0.5),
                perplexity: Some(1.5),
                examples: 8,
                ..Default::default()
            })
        }
        fn generate(
            &mut self,
            request: &GenerationRequest,
        ) -> retrograd_core::Result<GenerationOutput> {
            Ok(GenerationOutput {
                text: format!("answer to {}", request.prompt),
                prompt_tokens: 3,
                tokens: 4,
                base_text: None,
            })
        }
    }

    fn at(boundary: bool) -> ControlPoint {
        ControlPoint {
            iteration: 1,
            global_step: 10,
            at_boundary: boundary,
        }
    }

    fn wired() -> (ControlSender, ChannelControl) {
        let (sender, receiver) = channel();
        let handle = crate::runtime::registry::testing::detached_handle();
        (sender, ChannelControl::new(receiver, handle))
    }

    #[test]
    fn a_boundary_cancel_waits_for_a_boundary_and_a_now_cancel_does_not() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        sender.send(RunCommand::Cancel {
            at: CancelAt::Boundary,
            checkpoint: true,
        });
        assert_eq!(
            control.poll(&mut recorder, at(false)).unwrap(),
            Flow::Continue
        );
        assert_eq!(recorder.checkpoints_requested, 0);
        assert_eq!(control.poll(&mut recorder, at(true)).unwrap(), Flow::Stop);
        assert_eq!(recorder.checkpoints_requested, 1);
        assert!(control.was_cancelled());

        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        sender.send(RunCommand::Cancel {
            at: CancelAt::Now,
            checkpoint: false,
        });
        assert_eq!(control.poll(&mut recorder, at(false)).unwrap(), Flow::Stop);
        assert_eq!(
            recorder.checkpoints_requested, 0,
            "there is no resumable point between two boundaries"
        );
    }

    #[test]
    fn a_second_cancel_never_asks_for_less_than_the_first() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        sender.send(RunCommand::Cancel {
            at: CancelAt::Now,
            checkpoint: true,
        });
        // A retry that asks for the gentler form must not undo the urgent one.
        sender.send(RunCommand::Cancel {
            at: CancelAt::Boundary,
            checkpoint: false,
        });
        assert_eq!(control.poll(&mut recorder, at(false)).unwrap(), Flow::Stop);
    }

    #[test]
    fn the_whitelist_is_applied_field_by_field_and_a_refusal_is_not_fatal() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder {
            refuse_evaluation: true,
            ..Default::default()
        };
        sender.send(RunCommand::Adjust(Adjustments {
            learning_rate: Some(5.0e-5),
            evaluation_every_iterations: Some(2),
            evaluation_patience: Some(4),
            ..Default::default()
        }));
        assert_eq!(
            control.poll(&mut recorder, at(true)).unwrap(),
            Flow::Continue
        );
        assert_eq!(recorder.learning_rate, Some(5.0e-5));
        assert_eq!(recorder.patience, Some(4));
        assert_eq!(
            recorder.every_iterations, None,
            "one refused field does not take the others down with it"
        );
    }

    #[test]
    fn a_pause_blocks_the_callback_until_it_is_told_otherwise() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        sender.send(RunCommand::Pause);
        let resumer = sender.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            resumer.send(RunCommand::Resume);
        });
        let started = std::time::Instant::now();
        assert_eq!(
            control.poll(&mut recorder, at(true)).unwrap(),
            Flow::Continue
        );
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(20),
            "the callback returned before it was resumed"
        );

        // And a cancellation releases a pause, or a paused run could never be
        // stopped. The releasing thread starts *before* the blocking poll, or
        // the test is the deadlock it is meant to rule out.
        let canceller = sender.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            canceller.send(RunCommand::Cancel {
                at: CancelAt::Now,
                checkpoint: false,
            });
        });
        sender.send(RunCommand::Pause);
        assert_eq!(control.poll(&mut recorder, at(false)).unwrap(), Flow::Stop);
    }

    #[test]
    fn a_request_shaped_command_answers_with_where_the_run_was() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        let (reply, answer) = oneshot::channel();
        sender.send(RunCommand::Evaluate(reply));
        let (generate_reply, generated) = oneshot::channel();
        sender.send(RunCommand::Generate(
            Box::new(GenerationRequest {
                prompt: "hello".into(),
                chat: true,
                max_new_tokens: 4,
                temperature: 1.0,
                top_p: 1.0,
                seed: 1,
                include_base: false,
            }),
            generate_reply,
        ));
        assert_eq!(
            control.poll(&mut recorder, at(false)).unwrap(),
            Flow::Continue
        );

        let (at, evaluation) = answer.blocking_recv().expect("a reply").expect("evaluated");
        assert_eq!(at.global_step, 10);
        assert_eq!(evaluation.loss, Some(0.5));
        let (_, output) = generated
            .blocking_recv()
            .expect("a reply")
            .expect("sampled");
        assert_eq!(output.text, "answer to hello");
    }

    #[test]
    fn an_abandoned_request_does_not_cost_the_run_a_forward_pass() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder::default();
        let (reply, answer) = oneshot::channel();
        sender.send(RunCommand::Evaluate(reply));
        // The handler gave up - a timeout, or a disconnected client. Evaluating
        // for nobody would spend a full pass over the dataset.
        drop(answer);
        assert_eq!(
            control.poll(&mut recorder, at(true)).unwrap(),
            Flow::Continue
        );
        assert_eq!(recorder.evaluations, 0);
    }

    #[test]
    fn a_refused_evaluation_is_reported_to_the_caller_and_not_to_the_run() {
        let (sender, mut control) = wired();
        let mut recorder = Recorder {
            refuse_evaluation: true,
            ..Default::default()
        };
        let (reply, answer) = oneshot::channel();
        sender.send(RunCommand::Evaluate(reply));
        assert_eq!(
            control.poll(&mut recorder, at(true)).unwrap(),
            Flow::Continue,
            "a failed ad-hoc evaluation never stops the run"
        );
        assert!(answer.blocking_recv().expect("a reply").is_err());
    }

    #[test]
    fn the_paths_a_patch_touches_are_reported_in_a_fixed_order() {
        let adjustments = Adjustments {
            checkpoint_mode: Some(CheckpointMode::Steps),
            learning_rate: Some(1.0e-4),
            ..Default::default()
        };
        assert_eq!(adjustments.paths(), ["training.lr", "checkpoint.mode"]);
        assert!(Adjustments::default().is_empty());
        assert!(!adjustments.is_empty());
    }
}
