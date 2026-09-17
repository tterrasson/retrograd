//! `[observe]` for the rollout algorithms: the sink is opened and closed here,
//! and its warnings reach the run observer from here.

use retrograd_config::RunConfig;
use retrograd_core::Result;
use retrograd_metrics::MetricValue;
use retrograd_observe::{Algorithm, ObserveSink, RunInfo, SinkConfig};
use serde_json::{Map, Value};

use crate::observer::RunObserver;

/// Opens the sink when the configuration asks for one.
///
/// `resumed_from_update` is the number of updates the restored checkpoint
/// consumed; `params` are the hyperparameters the viewer shows.
pub(crate) fn open(
    config: &RunConfig,
    algorithm: Algorithm,
    resumed_from_update: Option<u64>,
    params: impl IntoIterator<Item = (&'static str, u64)>,
) -> Result<Option<ObserveSink>> {
    let Some(observe) = &config.observe else {
        return Ok(None);
    };
    let params = params
        .into_iter()
        .map(|(name, value)| (name.to_string(), Value::from(value)))
        .collect::<Map<_, _>>();
    let sink = ObserveSink::open(
        &SinkConfig {
            directory: observe.directory.clone(),
            every: observe.every,
            max_text_chars: observe.max_text_chars,
        },
        RunInfo {
            algorithm,
            model: config.model.display().to_string(),
            // A checkpoint past `u32::MAX` updates is refused by the loop right
            // after this; until then it only hides nothing.
            resumed_from_update: resumed_from_update
                .map(|updates| u32::try_from(updates).unwrap_or(u32::MAX)),
            params,
        },
    )?;
    Ok(Some(sink))
}

/// Reports the sink's new warnings and adds its drop counter to `values`.
pub(crate) fn report(
    sink: &ObserveSink,
    observer: &mut dyn RunObserver,
    values: &mut Vec<MetricValue>,
) {
    for warning in sink.take_warnings() {
        observer.info(&warning);
    }
    values.push(MetricValue {
        name: "observe/dropped_batches".into(),
        // A count the metric backends carry as f32: exact up to 2^24 drops.
        value: sink.dropped_batches() as f32,
    });
}

/// Closes the sink, bounded in time, and reports what it said last.
pub(crate) fn close(sink: Option<ObserveSink>, observer: &mut dyn RunObserver) {
    let Some(mut sink) = sink else {
        return;
    };
    sink.finish();
    for warning in sink.take_warnings() {
        observer.info(&warning);
    }
}
