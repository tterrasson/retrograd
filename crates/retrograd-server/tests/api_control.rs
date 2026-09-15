//! Step 7: pause, resume, cancel, checkpoint on demand and the `PATCH`
//! whitelist, driven through the router.
//!
//! The fake engine here is not a stub that ignores control - it polls it twice
//! per iteration, once between boundaries and once at one, exactly as the real
//! loops do. That is what makes these assertions about the control channel
//! rather than about the test.

mod support;

use std::time::Duration;

use http::StatusCode;
use serde_json::json;
use support::{
    FakeEngine, Fixture, get, patch, post, post_empty, recipe, recipe_with_schedules, router_for,
    wait_for_status, wait_for_terminal,
};

/// Long enough that a command reaches the run while it is alive, short enough
/// that a lane does not notice.
fn slow() -> std::sync::Arc<support::FakeEngine> {
    FakeEngine::slow(40, Duration::from_millis(15))
}

#[tokio::test]
async fn a_paused_run_holds_the_device_and_resumes_where_it_stopped() {
    let fixture = Fixture::new("pause");
    let router = router_for(&fixture, slow());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "pausable")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, accepted) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(accepted["id"], id);
    assert!(
        accepted["applies_at_iteration"].as_u64().unwrap() >= 1,
        "a pause lands at the next callback, not retroactively: {accepted}"
    );

    let paused = wait_for_status(&router, &id, "paused").await;
    assert!(
        paused["holds_device"].as_bool().unwrap(),
        "pause keeps the weights, the KV caches and the optimizer state (§9)"
    );
    let stopped_at = paused["progress"]["iteration"].as_u64().unwrap();

    // A repeated pause is not a conflict: a client that retried after a timeout
    // must not be told it did something wrong.
    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::OK);

    // And it really is stopped: nothing moves while it is paused.
    tokio::time::sleep(Duration::from_millis(80)).await;
    let (_, still) = get(&router, &format!("/v1/runs/{id}")).await;
    assert_eq!(
        still["progress"]["iteration"].as_u64().unwrap(),
        stopped_at,
        "a paused run does not advance"
    );

    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/resume")).await;
    assert_eq!(status, StatusCode::OK);
    let finished = wait_for_terminal(&router, &id).await;
    assert_eq!(finished["status"], "completed", "{finished}");
    assert!(finished["progress"]["iteration"].as_u64().unwrap() > stopped_at);
}

#[tokio::test]
async fn a_cancelled_run_ends_cancelled_and_not_completed() {
    let fixture = Fixture::new("cancel");
    let router = router_for(&fixture, slow());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "doomed")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, accepted) = post(
        &router,
        &format!("/v1/runs/{id}/cancel"),
        json!({"at": "now"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(accepted["status"], "cancelling");

    let finished = wait_for_terminal(&router, &id).await;
    assert_eq!(
        finished["status"], "cancelled",
        "a run that was stopped did not complete: {finished}"
    );
    assert!(
        finished["progress"]["iteration"].as_u64().unwrap() < 40,
        "it stopped early: {finished}"
    );
    assert!(!finished["holds_device"].as_bool().unwrap());

    // Cancelling again is idempotent, cancelling something over is a conflict
    // pointing at the way forward.
    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, problem) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{problem}");
    assert_eq!(problem["type"], "https://retrograd.dev/problems/conflict");
    assert!(
        problem["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("fork_from"),
        "{problem}"
    );
}

/// A queued run has nothing loaded, so cancelling it is immediate and the worker
/// that was waiting for the device never starts.
#[tokio::test]
async fn a_queued_run_is_cancelled_without_ever_starting() {
    let fixture = Fixture::new("cancel-queued");
    let engine = slow();
    let router = router_for(&fixture, engine.clone());
    let (_, first) = post(&router, "/v1/runs", recipe(&fixture, "first")).await;
    let first_id = first["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &first_id, "running").await;

    let (_, second) = post(&router, "/v1/runs", recipe(&fixture, "queued")).await;
    let second_id = second["id"].as_str().unwrap().to_string();
    let (status, accepted) = post_empty(&router, &format!("/v1/runs/{second_id}/cancel")).await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(accepted["status"], "cancelled");
    assert!(
        accepted.get("applies_at_iteration").is_none(),
        "nothing was loaded, so there is no iteration to wait for: {accepted}"
    );

    post_empty(&router, &format!("/v1/runs/{first_id}/cancel")).await;
    wait_for_terminal(&router, &first_id).await;
    // Give the queued worker every chance to wake up and start anyway.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        engine.executions(),
        1,
        "a run cancelled while queued never reaches the engine"
    );
}

#[tokio::test]
async fn a_boundary_cancel_can_keep_a_checkpoint_and_an_immediate_one_cannot() {
    let fixture = Fixture::new("cancel-checkpoint");
    let router = router_for(&fixture, slow());
    let (_, created) = post(
        &router,
        "/v1/runs",
        recipe_with_schedules(&fixture, "snapshot"),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    // The two together are refused rather than silently dropped: there is no
    // resumable point between two boundaries, and a client that believed it had
    // a checkpoint would not.
    let (status, problem) = post(
        &router,
        &format!("/v1/runs/{id}/cancel"),
        json!({"at": "now", "checkpoint": true}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/invalid-request"
    );

    let (status, _) = post(
        &router,
        &format!("/v1/runs/{id}/cancel"),
        json!({"at": "boundary", "checkpoint": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    wait_for_terminal(&router, &id).await;

    let events = std::fs::read_to_string(fixture.state_dir().join(&id).join("events.jsonl"))
        .expect("events.jsonl");
    assert!(
        events.contains(r#""type":"checkpoint""#),
        "a boundary cancellation wrote its checkpoint: {events}"
    );
}

#[tokio::test]
async fn a_checkpoint_can_be_asked_for_and_only_where_there_is_somewhere_to_put_it() {
    let fixture = Fixture::new("checkpoint");
    let router = router_for(&fixture, slow());
    let (_, created) = post(
        &router,
        "/v1/runs",
        recipe_with_schedules(&fixture, "on-demand"),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, accepted) = post_empty(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
    let events = std::fs::read_to_string(fixture.state_dir().join(&id).join("events.jsonl"))
        .expect("events.jsonl");
    assert!(events.contains(r#""type":"checkpoint""#), "{events}");

    // A run with no checkpoint directory has nowhere to write one, and is told
    // so before it queues a command nobody would apply.
    let fixture = Fixture::new("checkpoint-none");
    let router = router_for(&fixture, slow());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "no-directory")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;
    let (status, problem) = post_empty(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{problem}");
    post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn patch_applies_the_whitelist_and_refuses_everything_else() {
    let fixture = Fixture::new("patch");
    let engine = slow();
    let router = router_for(&fixture, engine.clone());
    let (_, created) = post(
        &router,
        "/v1/runs",
        recipe_with_schedules(&fixture, "adjustable"),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, accepted) = patch(
        &router,
        &format!("/v1/runs/{id}"),
        json!({
            "training": {"lr": 5.0e-5},
            "evaluation": {"every_iterations": 3, "patience": 2},
            "checkpoint": {"every_steps": 50, "mode": "steps_and_best_eval"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(
        accepted["applied"],
        json!([
            "training.lr",
            "evaluation.every_iterations",
            "evaluation.patience",
            "checkpoint.every_steps",
            "checkpoint.mode"
        ]),
        "the paths speak the document grammar, in a fixed order"
    );
    assert!(accepted["applies_at_iteration"].as_u64().unwrap() >= 1);

    // Anything outside the whitelist never reaches a handler: the request type
    // *is* the list.
    for body in [
        json!({"training": {"ctx": 4096}}),
        json!({"lora": {"rank": 32}}),
        json!({"model": "other.gguf"}),
        json!({"algorithm": "grpo"}),
    ] {
        let (status, problem) = patch(&router, &format!("/v1/runs/{id}"), body.clone()).await;
        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{body} was accepted: {problem}"
        );
    }
    // As does a request that adjusts nothing at all.
    let (status, _) = patch(&router, &format!("/v1/runs/{id}"), json!({})).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    // And a value the schedule cannot take.
    let (status, _) = patch(
        &router,
        &format!("/v1/runs/{id}"),
        json!({"training": {"lr": 0}}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;

    // The run really was driven: the knobs the loop offered were turned.
    let knobs = engine.knobs.lock().expect("knobs");
    assert_eq!(knobs.learning_rate, Some(5.0e-5));
    assert_eq!(knobs.every_iterations, Some(3));
    assert_eq!(knobs.patience, Some(2));
    assert_eq!(knobs.every_steps, Some(50));
}

/// A schedule this run does not have cannot be adjusted, and saying so is a
/// conflict rather than a silent no-op.
#[tokio::test]
async fn patch_refuses_a_schedule_the_run_does_not_have() {
    let fixture = Fixture::new("patch-absent");
    let router = router_for(&fixture, slow());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "bare")).await;
    let id = created["id"].as_str().unwrap().to_string();
    wait_for_status(&router, &id, "running").await;

    for body in [
        json!({"evaluation": {"every_iterations": 2}}),
        json!({"checkpoint": {"every_steps": 10}}),
    ] {
        let (status, problem) = patch(&router, &format!("/v1/runs/{id}"), body.clone()).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}: {problem}");
    }
    // The learning rate is always there, because every run has an optimizer.
    let (status, accepted) = patch(
        &router,
        &format!("/v1/runs/{id}"),
        json!({"training": {"lr": 1.0e-5}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{accepted}");

    post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;

    // Once it is over, no control field is adjustable.
    let (status, problem) = patch(
        &router,
        &format!("/v1/runs/{id}"),
        json!({"training": {"lr": 1.0e-5}}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{problem}");
}

#[tokio::test]
async fn commands_against_a_run_that_does_not_exist_are_not_found() {
    let fixture = Fixture::new("control-absent");
    let router = router_for(&fixture, FakeEngine::succeeding());
    for route in ["pause", "resume", "cancel", "checkpoints"] {
        let (status, problem) = post_empty(
            &router,
            &format!("/v1/runs/0d1e2f30-0000-4000-8000-000000000000/{route}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{route}: {problem}");
    }
    let (status, _) = patch(
        &router,
        "/v1/runs/0d1e2f30-0000-4000-8000-000000000000",
        json!({"training": {"lr": 1.0e-5}}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
