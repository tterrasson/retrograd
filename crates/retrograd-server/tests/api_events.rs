//! Step 8: the event stream, its replay, and the pull form of the metrics.
//!
//! The required property is
//! that **a client that reconnects neither loses nor duplicates an event**. It is
//! checked here by streaming a run in two halves and asserting that the
//! concatenation is exactly what one uninterrupted stream would have carried:
//! gapless, in order, with the same seq range.

mod support;

use std::time::Duration;

use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use support::{
    FakeEngine, Fixture, get, post, post_empty, recipe, router_for, send_raw, wait_for_status,
    wait_for_terminal,
};
use tower::ServiceExt;

/// One SSE frame, as a client would parse it.
#[derive(Debug)]
struct Frame {
    id: Option<u64>,
    event: String,
    data: Value,
}

/// The minimal `text/event-stream` reader: enough to assert on ids, types and
/// bodies, and no more.
fn parse(body: &[u8]) -> Vec<Frame> {
    let text = String::from_utf8_lossy(body);
    let mut frames = Vec::new();
    for block in text.split("\n\n") {
        let (mut id, mut event, mut data) = (None, String::new(), String::new());
        for line in block.lines() {
            if let Some(value) = line.strip_prefix("id:") {
                id = value.trim().parse::<u64>().ok();
            } else if let Some(value) = line.strip_prefix("event:") {
                event = value.trim().to_string();
            } else if let Some(value) = line.strip_prefix("data:") {
                data = value.trim().to_string();
            }
        }
        if event.is_empty() {
            continue;
        }
        frames.push(Frame {
            id,
            event,
            data: serde_json::from_str(&data).unwrap_or(Value::Null),
        });
    }
    frames
}

/// Reads a stream to its end. Every run in this file terminates, and the handler
/// closes the stream when it does, so this returns rather than hanging - which is
/// itself part of what is being asserted.
async fn stream(router: &axum::Router, uri: &str, last_event_id: Option<u64>) -> Vec<Frame> {
    let mut request = Request::builder().method("GET").uri(uri);
    if let Some(id) = last_event_id {
        request = request.header("last-event-id", id.to_string());
    }
    let (status, content_type, body) = send_raw(
        router,
        request.body(Body::empty()).expect("build the request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("text/event-stream"),
        "{content_type}"
    );
    parse(&body)
}

#[tokio::test]
async fn a_finished_run_replays_its_whole_history_and_the_stream_ends() {
    let fixture = Fixture::new("events-replay");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "watched")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_terminal(&router, &id).await;

    let frames = stream(&router, &format!("/v1/runs/{id}/events"), None).await;
    let ids: Vec<u64> = frames.iter().filter_map(|frame| frame.id).collect();
    assert_eq!(
        ids,
        (1..=ids.len() as u64).collect::<Vec<_>>(),
        "seq is gapless and starts at 1, so ?since=0 means 'everything'"
    );
    // `event:` and the body's `type` are the same string, so a client may filter
    // on either.
    for frame in &frames {
        assert_eq!(frame.data["type"], frame.event.as_str(), "{frame:?}");
        assert_eq!(frame.data["seq"].as_u64(), frame.id, "{frame:?}");
    }
    let kinds: Vec<&str> = frames.iter().map(|frame| frame.event.as_str()).collect();
    assert!(kinds.contains(&"status"), "{kinds:?}");
    assert!(kinds.contains(&"progress"), "{kinds:?}");
    assert!(kinds.contains(&"metrics"), "{kinds:?}");
    assert_eq!(kinds.last(), Some(&"terminal"), "{kinds:?}");

    // And a server restarted on the same directory replays it from disk, with no
    // ring and no live run.
    let restarted =
        support::build_router(support::state_of(&fixture, FakeEngine::succeeding(), false));
    let again = stream(&restarted, &format!("/v1/runs/{id}/events"), None).await;
    let replayed: Vec<u64> = again.iter().filter_map(|frame| frame.id).collect();
    assert_eq!(
        replayed, ids,
        "the journal is the record, the ring is a cache"
    );
}

/// The reconnection property, in one test.
#[tokio::test]
async fn a_reconnecting_client_loses_nothing_and_repeats_nothing() {
    let fixture = Fixture::new("events-reconnect");
    let router = router_for(&fixture, FakeEngine::slow(6, Duration::from_millis(15)));
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "interrupted")).await;
    let id = created["id"].as_str().unwrap().to_string();

    // First connection: everything up to some point, cut short by the run still
    // being alive when we stop reading.
    wait_for_status(&router, &id, "running").await;
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (_, page) = get(&router, &format!("/v1/runs/{id}/metrics")).await;
    let seen = page["next_since"].as_u64().unwrap();
    assert!(seen > 0, "the run has emitted something: {page}");

    wait_for_terminal(&router, &id).await;

    // Second connection, resuming with `Last-Event-ID` - what a browser's
    // `EventSource` sends by itself.
    let rest = stream(&router, &format!("/v1/runs/{id}/events"), Some(seen)).await;
    let resumed: Vec<u64> = rest.iter().filter_map(|frame| frame.id).collect();
    assert_eq!(
        resumed.first(),
        Some(&(seen + 1)),
        "the replay starts at the first event the client had not seen"
    );
    assert_eq!(
        resumed,
        (seen + 1..=seen + resumed.len() as u64).collect::<Vec<_>>(),
        "no gap and no repeat"
    );

    // The whole stream, and the two halves, cover exactly the same range.
    let whole = stream(&router, &format!("/v1/runs/{id}/events"), None).await;
    let all: Vec<u64> = whole.iter().filter_map(|frame| frame.id).collect();
    assert_eq!(all.last(), resumed.last());
    assert_eq!(all.iter().filter(|seq| **seq <= seen).count() as u64, seen);

    // `?since=` is the same resume point without the header.
    let query = stream(&router, &format!("/v1/runs/{id}/events?since={seen}"), None).await;
    assert_eq!(
        query
            .iter()
            .filter_map(|frame| frame.id)
            .collect::<Vec<_>>(),
        resumed
    );
}

#[tokio::test]
async fn the_metrics_pull_pages_filters_and_moves_its_cursor() {
    let fixture = Fixture::new("events-metrics");
    let router = router_for(&fixture, FakeEngine::slow(4, Duration::ZERO));
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "measured")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_terminal(&router, &id).await;

    let (status, page) = get(&router, &format!("/v1/runs/{id}/metrics")).await;
    assert_eq!(status, StatusCode::OK, "{page}");
    let samples = page["metrics"].as_array().expect("metrics");
    assert_eq!(samples.len(), 4, "one emission per iteration: {page}");
    assert!(samples[0]["values"]["train/loss"].is_number());
    assert_eq!(samples[0]["global_step"], 10);

    // Names narrow the values without dropping the samples.
    let (_, filtered) = get(&router, &format!("/v1/runs/{id}/metrics?names=train/lr")).await;
    let first = &filtered["metrics"][0]["values"];
    assert!(first["train/lr"].is_number(), "{filtered}");
    assert!(first.get("train/loss").is_none(), "{filtered}");

    // A page, then the rest, then nothing - and the cursor never rewinds.
    let (_, page) = get(&router, &format!("/v1/runs/{id}/metrics?limit=2")).await;
    assert_eq!(page["metrics"].as_array().unwrap().len(), 2);
    let cursor = page["next_since"].as_u64().unwrap();
    let (_, rest) = get(
        &router,
        &format!("/v1/runs/{id}/metrics?limit=2&since={cursor}"),
    )
    .await;
    assert_eq!(rest["metrics"].as_array().unwrap().len(), 2);
    let cursor = rest["next_since"].as_u64().unwrap();
    let (_, empty) = get(&router, &format!("/v1/runs/{id}/metrics?since={cursor}")).await;
    assert!(empty["metrics"].as_array().unwrap().is_empty(), "{empty}");
    assert!(
        empty["next_since"].as_u64().unwrap() >= cursor,
        "an empty page never rewinds the cursor: {empty}"
    );
}

/// The aggregate stream is live-only - `seq` is per run, so there is no single
/// sequence to resume from - and every frame says which run it came from.
/// Reads a stream that never ends, frame by frame, until `budget` runs out.
///
/// Collecting the whole body would never return here - and a timeout around the
/// collection would drop everything read so far, which is exactly the mistake
/// this helper exists to avoid.
async fn stream_until(router: &axum::Router, uri: &str, budget: Duration) -> Vec<Frame> {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("build the request"),
        )
        .await
        .expect("route the request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    let deadline = tokio::time::Instant::now() + budget;
    while let Ok(Some(Ok(frame))) = tokio::time::timeout_at(deadline, body.frame()).await {
        if let Some(data) = frame.data_ref() {
            bytes.extend_from_slice(data);
        }
    }
    parse(&bytes)
}

#[tokio::test]
async fn the_aggregate_stream_tags_every_event_with_its_run() {
    let fixture = Fixture::new("events-all");
    let router = router_for(&fixture, FakeEngine::slow(4, Duration::from_millis(15)));

    let watcher = router.clone();
    let reader = tokio::spawn(async move {
        stream_until(&watcher, "/v1/events", Duration::from_millis(500)).await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "aggregated")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_terminal(&router, &id).await;

    let frames = reader.await.expect("the reader task");
    assert!(!frames.is_empty(), "the aggregate stream carried nothing");
    for frame in &frames {
        assert_eq!(frame.data["run"], id.as_str(), "{frame:?}");
        assert_eq!(frame.data["type"], frame.event.as_str(), "{frame:?}");
    }
}

#[tokio::test]
async fn watching_a_run_that_does_not_exist_is_not_found() {
    let fixture = Fixture::new("events-absent");
    let router = router_for(&fixture, FakeEngine::succeeding());
    for route in ["events", "metrics"] {
        let (status, problem) = get(
            &router,
            &format!("/v1/runs/0d1e2f30-0000-4000-8000-000000000000/{route}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{route}: {problem}");
    }
    // A run that is over answers a pause with a conflict, not a stream.
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "done")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_terminal(&router, &id).await;
    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::CONFLICT);
}
