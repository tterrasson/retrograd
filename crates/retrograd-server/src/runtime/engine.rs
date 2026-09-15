//! What actually runs a configuration, behind a trait.
//!
//! The trait exists for one reason: the whole HTTP surface - creation, the state
//! machine, the journal, the listing, the control commands, the event stream,
//! has to be testable in the `fast-rust` lane, with no GGUF and no device. A fake
//! engine that reports a few epochs and returns covers all of that in
//! milliseconds, and the real one is a single delegation, so there is nowhere for
//! the two to diverge.
//!
//! The `control` and `sinks` arguments are the two seams the control plane needs.
//! They are on this trait rather than hidden inside the real implementation
//! precisely so a fake has to honour them: a fake that ignored `control` would
//! make every pause and cancel test pass against nothing.

use retrograd_config::RunConfig;
use retrograd_core::Result as CoreResult;
use retrograd_metrics::MetricsSink;
use retrograd_run::{RunControl, RunObserver, RunOutcome};

/// Runs one configuration to completion on the calling thread.
///
/// **Called on the run's own thread**, never on an async worker: the
/// implementation builds a `Trainer`, which must not move between threads.
pub trait RunEngine: Send + Sync + 'static {
    fn execute(
        &self,
        config: &RunConfig,
        observer: &mut dyn RunObserver,
        control: &mut dyn RunControl,
        sinks: Vec<Box<dyn MetricsSink>>,
    ) -> CoreResult<RunOutcome>;
}

/// The real engine: exactly what `retrograd train` runs, with an observer that
/// writes events instead of drawing a terminal, and a control channel the CLI
/// does not have.
pub struct TrainingEngine;

impl RunEngine for TrainingEngine {
    fn execute(
        &self,
        config: &RunConfig,
        observer: &mut dyn RunObserver,
        control: &mut dyn RunControl,
        sinks: Vec<Box<dyn MetricsSink>>,
    ) -> CoreResult<RunOutcome> {
        retrograd_run::execute_controlled(config, observer, control, sinks)
    }
}
