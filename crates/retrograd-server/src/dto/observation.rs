//! Observation

use super::*;

schema! {
/// `GET /v1/runs/{id}/metrics`
///
/// The pull half of the event stream: the same events, filtered to `metrics` and
/// served from the ring or `events.jsonl`. `next_since` is what to pass back to
/// continue, so polling is a loop with no bookkeeping on the client.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricsPage {
    pub metrics: Vec<MetricsSample>,
    pub next_since: u64,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricsSample {
    pub seq: u64,
    /// Unix milliseconds.
    pub at: u64,
    pub iteration: u64,
    pub global_step: u64,
    pub values: BTreeMap<String, f32>,
}
}
