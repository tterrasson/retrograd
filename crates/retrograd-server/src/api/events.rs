//! Watching a run: SSE per run, SSE across runs, and the pull form.
//!
//! One rule shapes all three: **a slow client must never slow the training
//! loop**. So a run emits into a bounded broadcast that drops rather than blocks,
//! and a subscriber that falls behind is told it lagged instead of being quietly
//! given a hole.
//!
//! The other rule is that a reconnecting client neither loses nor duplicates an
//! event, which is what makes `Last-Event-ID` worth honouring at all. Three
//! things together give it:
//!
//! 1. `seq` is allocated in one place, per run, and is gapless;
//! 2. the live subscription is taken **before** the replay, so nothing emitted
//!    between the two falls through the crack;
//! 3. anything the replay already yielded is filtered out of the live stream by
//!    its `seq`, which is what makes (2) safe rather than duplicating.
//!
//! The replay itself comes from the in-memory ring when it reaches far enough
//! back and from `events.jsonl` otherwise, so `?since=0` replays a finished run
//! in full - including one this process only read back from disk.

use std::convert::Infallible;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_core::Stream;
use http::HeaderMap;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast::error::RecvError;

use crate::dto;
use crate::error::ApiResult;
use crate::state::AppState;

/// Largest page `GET /v1/runs/{id}/metrics` answers. A run emits a metric event
/// per optimizer callback, so an unbounded pull over a long run would render
/// megabytes into one response body.
const MAX_METRICS: usize = 1000;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamQuery {
    /// Resume after this `seq`. Ignored when `Last-Event-ID` is present, which
    /// is what a browser's `EventSource` sends by itself.
    #[serde(default)]
    pub since: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsQuery {
    #[serde(default)]
    pub since: Option<u64>,
    /// Comma-separated metric names. Absent means every name.
    #[serde(default)]
    pub names: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /v1/runs/{id}/events`
pub async fn stream(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<StreamQuery>,
    headers: HeaderMap,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let handle = super::runs::lookup(&state, &id)?;
    let since = resume_point(&headers, query.since);
    Ok(stream_from(handle, since))
}

fn stream_from(
    handle: std::sync::Arc<crate::runtime::RunHandle>,
    since: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Subscribe first, replay second. The other order loses every event emitted
    // between the two calls, which on a busy run is where the interesting ones
    // are.
    let mut live = handle.subscribe();
    let terminal_already = handle.events_finished();
    let replay = handle.replay(since);

    let stream = async_stream::stream! {
        let mut last = since;
        for event in replay {
            if event.seq <= last {
                continue;
            }
            last = event.seq;
            let done = matches!(event.payload, dto::RunEventPayload::Terminal { .. });
            if let Some(frame) = frame(&event) {
                yield Ok(frame);
            }
            if done {
                return;
            }
        }
        // A finished run has nothing more to say. Ending the stream is what lets
        // a client `await` it instead of polling for a state change it will never
        // be told about.
        if terminal_already {
            return;
        }
        loop {
            match live.recv().await {
                Ok(event) => {
                    // Already served by the replay: (2) above is only safe
                    // because of this line.
                    if event.seq <= last {
                        continue;
                    }
                    last = event.seq;
                    let done = matches!(event.payload, dto::RunEventPayload::Terminal { .. });
                    if let Some(frame) = frame(&event) {
                        yield Ok(frame);
                    }
                    if done {
                        return;
                    }
                }
                // The client fell behind the broadcast. It is told from where to
                // catch up rather than silently skipped, because a gap it cannot
                // see is worse than one it can.
                Err(RecvError::Lagged(_)) => {
                    yield Ok(Event::default()
                        .event("lagged")
                        .data(format!("{{\"since\":{last}}}")));
                }
                Err(RecvError::Closed) => return,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// `GET /v1/events` - every run's events, merged.
///
/// Live only, and deliberately so: `seq` is per run, so there is no single
/// sequence a `?since=` could resume from. A client that needs replay opens the
/// per-run stream, which has one.
pub async fn stream_all(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut live = state.registry.subscribe_all();
    let stream = async_stream::stream! {
        loop {
            match live.recv().await {
                Ok(tagged) => {
                    let body = Tagged {
                        run: &tagged.run,
                        event: &tagged.event,
                    };
                    if let Ok(frame) = Event::default()
                        .event(tagged.event.payload.kind())
                        .json_data(&body)
                    {
                        yield Ok(frame);
                    }
                }
                Err(RecvError::Lagged(_)) => {
                    yield Ok(Event::default().event("lagged").data("{}"));
                }
                Err(RecvError::Closed) => return,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// `GET /v1/runs/{id}/metrics`
pub async fn metrics(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<MetricsQuery>,
) -> ApiResult<Json<dto::MetricsPage>> {
    let handle = super::runs::lookup(&state, &id)?;
    let since = query.since.unwrap_or(0);
    let limit = query.limit.unwrap_or(MAX_METRICS).clamp(1, MAX_METRICS);
    let wanted: Option<Vec<&str>> = query.names.as_deref().map(|names| {
        names
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .collect()
    });

    let mut samples = Vec::new();
    // Starts at what the client already has, so an empty page moves nothing and
    // a truncated one resumes exactly where it stopped.
    let mut next_since = since;
    for event in handle.replay(since) {
        // Never advance past an event that was not part of this snapshot.
        // Reading last_seq after the replay can skip a concurrent metric.
        if samples.len() == limit {
            break;
        }
        next_since = event.seq;
        let dto::RunEventPayload::Metrics {
            iteration,
            global_step,
            values,
        } = &event.payload
        else {
            continue;
        };
        let values = match &wanted {
            Some(names) => values
                .iter()
                .filter(|(name, _)| names.iter().any(|wanted| wanted == name))
                .map(|(name, value)| (name.clone(), *value))
                .collect(),
            None => values.clone(),
        };
        samples.push(dto::MetricsSample {
            seq: event.seq,
            at: event.at,
            iteration: *iteration,
            global_step: *global_step,
            values,
        });
    }
    Ok(Json(dto::MetricsPage {
        metrics: samples,
        next_since,
    }))
}

/// One frame of the aggregate stream: the event, plus which run it came from.
#[derive(Serialize)]
struct Tagged<'a> {
    run: &'a str,
    #[serde(flatten)]
    event: &'a dto::RunEvent,
}

/// `Last-Event-ID` wins over `?since=`: a browser resends the header on its own
/// after a dropped connection, and the query string it was opened with is the
/// stale one.
fn resume_point(headers: &HeaderMap, since: Option<u64>) -> u64 {
    headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or(since)
        .unwrap_or(0)
}

/// `id:` carries the `seq` so `Last-Event-ID` works, and `event:` carries the
/// same discriminant the JSON body's `type` does.
fn frame(event: &dto::RunEvent) -> Option<Event> {
    Event::default()
        .id(event.seq.to_string())
        .event(event.payload.kind())
        .json_data(event)
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use std::time::Duration;

    #[tokio::test]
    async fn a_terminal_status_does_not_close_before_the_terminal_event() {
        let handle = crate::runtime::registry::testing::detached_handle();
        assert!(handle.transition(dto::RunStatus::Completed));
        let response = stream_from(handle.clone(), 0).into_response();
        let reader = tokio::spawn(async move {
            axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap()
        });
        // The status and the final event are separate publications. Hold the
        // latter back until the consumer has polled the replay.
        tokio::task::yield_now().await;
        assert!(!reader.is_finished(), "closed before the terminal event");
        handle.emit(dto::RunEventPayload::Terminal {
            status: dto::RunStatus::Completed,
            error: None,
        });
        let bytes = tokio::time::timeout(Duration::from_secs(1), reader)
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("event: terminal"));
    }

    #[tokio::test]
    async fn replaying_the_terminal_event_closes_even_without_a_terminal_status() {
        let handle = crate::runtime::registry::testing::detached_handle();
        handle.emit(dto::RunEventPayload::Terminal {
            status: dto::RunStatus::Completed,
            error: None,
        });
        let response = stream_from(handle, 0).into_response();
        let bytes = tokio::time::timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), 64 * 1024),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("event: terminal"));
    }
}
