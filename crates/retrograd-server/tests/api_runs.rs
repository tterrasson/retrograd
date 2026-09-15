//! `POST /v1/runs`, `GET /v1/runs`, `GET /v1/runs/{id}` - the runtime, driven
//! through the router in memory.
//!
//! Nothing here loads a model or touches a device: the engine is a fake that
//! reports a handful of observer events and returns. What that leaves under test
//! is everything the runtime actually owns - the state machine, the device
//! queue, the journal on disk, idempotency, the listing and its paging - which is
//! the part a GPU lane would only slow down without covering better.

mod support;

use std::sync::Arc;

use http::StatusCode;
use serde_json::Value;
use support::{
    Behaviour, FakeEngine, Fixture, Gate, build_router, get, post, post_with, recipe, router_for,
    state_of, wait_for_status, wait_for_terminal,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_created_run_is_answered_immediately_and_runs_to_completion() {
    let fixture = Fixture::new("create");
    let engine = FakeEngine::succeeding();
    let router = router_for(&fixture, engine.clone());

    let (status, created) = post(&router, "/v1/runs", recipe(&fixture, "sft-v1")).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("an id").to_string();
    assert_eq!(
        created["effective_config"]["lora"]["output"],
        fixture
            .state_dir()
            .join(&id)
            .join("adapter.gguf")
            .to_string_lossy()
            .as_ref()
    );
    assert_eq!(created["name"], "sft-v1");
    // The answer is the plan, plus an identity and a state. A client never has
    // to ask twice for what it just asked to create.
    assert!(
        created["effective_config"]["training"]["ctx"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        created["plan"]["memory"]["resources"]["device_peak_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(created["provenance"]["training.ctx"]["source"].is_string());
    assert_eq!(
        created["status"], "queued",
        "creation answers before the run starts: {created}"
    );

    let finished = wait_for_terminal(&router, &id).await;
    assert_eq!(finished["status"], "completed", "{finished}");
    assert_eq!(finished["progress"]["iteration"], 2);
    assert_eq!(finished["progress"]["iterations"], 2);
    assert_eq!(finished["progress"]["global_step"], 20);
    assert!(finished["progress"]["train_loss"].is_number());
    assert!(
        finished["progress"].get("eval_loss").is_none(),
        "a NaN measurement is absent, never null: {}",
        finished["progress"]
    );
    assert!(
        !finished["holds_device"].as_bool().unwrap(),
        "a finished run releases the device"
    );
    assert!(finished["started_at"].is_number() && finished["finished_at"].is_number());
    assert_eq!(engine.executions(), 1);
}

#[tokio::test]
async fn a_run_writes_its_journal_where_a_restart_can_read_it() {
    let fixture = Fixture::new("journal");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "journalled")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_terminal(&router, &id).await;

    let dir = fixture.state_dir().join(&id);
    let document: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).expect("run.json"))
            .expect("valid JSON");
    assert_eq!(document["status"], "completed");
    assert_eq!(document["name"], "journalled");
    // Every state it went through, in order - not just the last one.
    let transitions: Vec<&str> = document["transitions"]
        .as_array()
        .expect("transitions")
        .iter()
        .map(|entry| entry["status"].as_str().unwrap())
        .collect();
    assert_eq!(transitions, ["queued", "starting", "running", "completed"]);

    let events = std::fs::read_to_string(dir.join("events.jsonl")).expect("events.jsonl");
    let events: Vec<Value> = events
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
        .collect();
    let sequence: Vec<u64> = events
        .iter()
        .map(|event| event["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(
        sequence,
        (1..=events.len() as u64).collect::<Vec<_>>(),
        "seq is monotonic and gapless, which is what a replaying client needs"
    );
    let kinds: Vec<&str> = events
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"status"), "{kinds:?}");
    assert!(kinds.contains(&"progress"), "{kinds:?}");
    // The metrics bus writes to the journal too, so a run's numbers survive the
    // process that produced them and `GET …/metrics` can answer for a run this
    // server only read back from disk.
    assert!(kinds.contains(&"metrics"), "{kinds:?}");
    assert_eq!(kinds.last(), Some(&"terminal"), "{kinds:?}");

    // And a server started on the same state directory sees the history.
    let restarted = build_router(state_of(&fixture, FakeEngine::succeeding(), false));
    let (status, listing) = get(&restarted, "/v1/runs").await;
    assert_eq!(status, StatusCode::OK, "{listing}");
    assert_eq!(listing["runs"][0]["id"], id);
    assert_eq!(listing["runs"][0]["status"], "completed");
    let (status, restored) = get(&restarted, &format!("/v1/runs/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{restored}");
    assert_eq!(
        restored["effective_config"], created["effective_config"],
        "a restored run answers what it answered while it was alive"
    );
}

#[tokio::test]
async fn a_run_that_fails_says_why_and_one_that_panics_does_not_hang() {
    let fixture = Fixture::new("failure");
    let router = router_for(
        &fixture,
        FakeEngine::with(Behaviour::Fails("no such tensor")),
    );
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "doomed")).await;
    let id = created["id"].as_str().unwrap().to_string();
    let finished = wait_for_terminal(&router, &id).await;
    assert_eq!(finished["status"], "failed");
    assert!(
        finished["error"]
            .as_str()
            .unwrap()
            .contains("no such tensor"),
        "{finished}"
    );

    // A panicking worker must leave a `failed` run, not one stuck in `running`.
    let fixture = Fixture::new("panic");
    let router = router_for(&fixture, FakeEngine::with(Behaviour::Panics));
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "panics")).await;
    let id = created["id"].as_str().unwrap().to_string();
    let finished = wait_for_terminal(&router, &id).await;
    assert_eq!(finished["status"], "failed");
    assert!(
        finished["error"].as_str().unwrap().contains("panic"),
        "{finished}"
    );
}

/// One device, one run at a time: the second waits, it is not refused.
#[tokio::test]
async fn a_second_run_queues_behind_the_first() {
    let fixture = Fixture::new("queue");
    let gate = Arc::new(Gate::default());
    let engine = FakeEngine::gated(gate.clone());
    let router = router_for(&fixture, engine.clone());

    let (_, first) = post(&router, "/v1/runs", recipe(&fixture, "first")).await;
    let first_id = first["id"].as_str().unwrap().to_string();
    let running = wait_for_status(&router, &first_id, "running").await;
    assert!(running["holds_device"].as_bool().unwrap());

    let (status, second) = post(&router, "/v1/runs", recipe(&fixture, "second")).await;
    assert_eq!(status, StatusCode::CREATED, "{second}");
    let second_id = second["id"].as_str().unwrap().to_string();
    let (_, queued) = get(&router, &format!("/v1/runs/{second_id}")).await;
    assert_eq!(
        queued["status"], "queued",
        "a second run waits for the device rather than loading a second model"
    );
    assert!(!queued["holds_device"].as_bool().unwrap());
    assert_eq!(engine.executions(), 1, "the second run has not started");

    gate.release();
    assert_eq!(
        wait_for_terminal(&router, &first_id).await["status"],
        "completed"
    );
    assert_eq!(
        wait_for_terminal(&router, &second_id).await["status"],
        "completed"
    );
    assert_eq!(engine.executions(), 2);
}

#[tokio::test]
async fn a_repeated_request_with_the_same_key_does_not_start_a_second_run() {
    let fixture = Fixture::new("idempotency");
    let engine = FakeEngine::succeeding();
    let router = router_for(&fixture, engine.clone());
    let headers = [("idempotency-key", "abc-123")];

    let (status, first) = post_with(&router, "/v1/runs", recipe(&fixture, "once"), &headers).await;
    assert_eq!(status, StatusCode::CREATED, "{first}");
    let (status, again) = post_with(&router, "/v1/runs", recipe(&fixture, "once"), &headers).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a replay is not a creation: {again}"
    );
    assert_eq!(again["id"], first["id"]);

    // A different key is a different run.
    let (status, other) = post_with(
        &router,
        "/v1/runs",
        recipe(&fixture, "twice"),
        &[("idempotency-key", "def-456")],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{other}");
    assert_ne!(other["id"], first["id"]);

    wait_for_terminal(&router, first["id"].as_str().unwrap()).await;
    wait_for_terminal(&router, other["id"].as_str().unwrap()).await;
    assert_eq!(engine.executions(), 2);
}

#[tokio::test]
async fn the_listing_is_newest_first_filterable_and_paged() {
    let fixture = Fixture::new("listing");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let mut ids = Vec::new();
    for index in 0..3 {
        let (_, created) = post(
            &router,
            "/v1/runs",
            recipe(&fixture, &format!("run-{index}")),
        )
        .await;
        let id = created["id"].as_str().unwrap().to_string();
        wait_for_terminal(&router, &id).await;
        ids.push(id);
    }

    let (status, listing) = get(&router, "/v1/runs").await;
    assert_eq!(status, StatusCode::OK, "{listing}");
    let listed: Vec<&str> = listing["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        listed,
        vec![ids[2].as_str(), ids[1].as_str(), ids[0].as_str()]
    );
    assert!(listing.get("next_cursor").is_none());

    let (_, named) = get(&router, "/v1/runs?name=run-1").await;
    assert_eq!(named["runs"].as_array().unwrap().len(), 1);
    assert_eq!(named["runs"][0]["id"], ids[1]);

    let (_, none) = get(&router, "/v1/runs?status=running").await;
    assert!(none["runs"].as_array().unwrap().is_empty());

    let (_, page) = get(&router, "/v1/runs?limit=2").await;
    assert_eq!(page["runs"].as_array().unwrap().len(), 2);
    let cursor = page["next_cursor"].as_str().expect("a second page");
    let (_, rest) = get(&router, &format!("/v1/runs?limit=2&cursor={cursor}")).await;
    assert_eq!(rest["runs"].as_array().unwrap().len(), 1);
    assert_eq!(rest["runs"][0]["id"], ids[0]);
}

#[tokio::test]
async fn an_unknown_run_is_a_problem_document_and_a_dry_run_creates_nothing() {
    let fixture = Fixture::new("absent");
    let engine = FakeEngine::succeeding();
    let router = router_for(&fixture, engine.clone());

    let (status, problem) = get(&router, "/v1/runs/0d1e2f30-0000-4000-8000-000000000000").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{problem}");
    assert_eq!(problem["type"], "https://retrograd.dev/problems/not-found");
    // An id is opaque to a client, so a malformed one is the same answer.
    let (status, problem) = get(&router, "/v1/runs/not-a-uuid").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{problem}");

    let (status, dry) = post(&router, "/v1/runs?dry_run=true", recipe(&fixture, "ghost")).await;
    assert_eq!(status, StatusCode::OK, "{dry}");
    assert!(dry.get("id").is_none(), "nothing was created to identify");
    let (_, listing) = get(&router, "/v1/runs").await;
    assert!(listing["runs"].as_array().unwrap().is_empty());
    assert_eq!(engine.executions(), 0);
}

/// Creating a run measures by default: the model is loaded anyway.
#[tokio::test]
async fn creating_a_run_measures_the_candidate_unless_told_not_to() {
    let fixture = Fixture::new("calibrated-run");
    let state = state_of(&fixture, FakeEngine::succeeding(), true);
    let router = build_router(state);
    let (status, created) = post(&router, "/v1/runs", recipe(&fixture, "measured")).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert!(
        created["plan"]["memory"]["measured"].is_object(),
        "a created run carries what it measured: {}",
        created["plan"]["memory"]
    );
    assert_eq!(created["provenance"]["training.ctx"]["source"], "measured");
    wait_for_terminal(&router, created["id"].as_str().unwrap()).await;

    // `calibrate_runs = false` - the default in these tests - leaves the plan an
    // estimate, and says nothing about a measurement.
    let plain = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&plain, "/v1/runs", recipe(&fixture, "estimated")).await;
    assert!(created["plan"]["memory"].get("measured").is_none());
    wait_for_terminal(&plain, created["id"].as_str().unwrap()).await;
}
