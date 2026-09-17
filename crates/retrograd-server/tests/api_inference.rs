//! Step 9: asking a live run a question - `evaluate` and `generate`.
//!
//! These are the only two routes that wait for the run, so what is under test is
//! the round trip: a command down the control channel, served inside the progress
//! callback, and a reply carrying the step it was taken at. The fake engine really
//! polls the control twice per iteration, so a reply here means the loop really
//! answered - not that a handler invented a number.

mod support;

use std::time::Duration;

use http::StatusCode;
use serde_json::json;
use support::*;

/// A run slow enough to be interrogated while it is alive, already `running`.
async fn live(fixture: &Fixture, recipe_body: serde_json::Value) -> (axum::Router, String) {
    let engine = FakeEngine::slow(40, Duration::from_millis(25));
    let router = router_for(fixture, engine);
    let (status, body) = post(&router, "/v1/runs", recipe_body).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().expect("an id").to_string();
    wait_for_status(&router, &id, "running").await;
    (router, id)
}

#[tokio::test]
async fn an_ad_hoc_evaluation_answers_with_the_step_it_was_taken_at() {
    let fixture = Fixture::new("inference-evaluate");
    let (router, id) = live(&fixture, recipe_with_schedules(&fixture, "eval")).await;

    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/evaluate")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], id);
    assert_eq!(body["loss"], 0.375);
    assert_eq!(body["examples"], 200);
    // The step is the loop's, not the handler's: the fake engine polls at
    // `epoch * 10` and `epoch * 10 - 5`, so anything else means the number was
    // made up somewhere between.
    let step = body["global_step"].as_u64().expect("a step");
    assert!(step > 0 && step % 5 == 0, "{body}");
    assert!(
        body["iteration"].as_u64().expect("an iteration") >= 1,
        "{body}"
    );
    // An SFT evaluation has no reward, and an absent measurement is absent rather
    // than null.
    assert!(body.get("mean_reward").is_none(), "{body}");

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn an_ad_hoc_evaluation_is_refused_on_a_run_that_has_no_dataset() {
    let fixture = Fixture::new("inference-no-eval");
    // The plain recipe has no `eval` section, so there is nothing to evaluate
    // against - and the refusal happens before a command is queued, so the run
    // never spends a pass on it.
    let (router, id) = live(&fixture, recipe(&fixture, "no-eval")).await;

    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/evaluate")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["errors"][0]["pointer"], "/evaluation");

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn an_evaluate_request_takes_no_options() {
    let fixture = Fixture::new("inference-evaluate-options");
    let (router, id) = live(&fixture, recipe_with_schedules(&fixture, "eval-options")).await;

    // A client that thought it could point the evaluation at another dataset is
    // told it cannot, rather than having the field silently ignored.
    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/evaluate"),
        json!({"dataset": "other.jsonl"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    // An empty object is fine: it is the request.
    let (status, _) = post(&router, &format!("/v1/runs/{id}/evaluate"), json!({})).await;
    assert_eq!(status, StatusCode::OK);

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn a_generation_reaches_the_model_with_the_settings_that_were_sent() {
    let fixture = Fixture::new("inference-generate");
    let (router, id) = live(&fixture, recipe(&fixture, "generate")).await;

    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/generate"),
        json!({"prompt": "how many moons", "max_new_tokens": 16, "include_base": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // `chat` defaults to on, and `max_new_tokens` came from the request: the fake
    // model echoes both, so this asserts the request was not replaced by defaults
    // on the way down.
    assert_eq!(body["text"], "[chat=true n=16] answer");
    assert_eq!(body["base_text"], "base answer");
    assert_eq!(body["prompt_tokens"], 3);
    assert_eq!(body["tokens"], 5);
    assert!(body["global_step"].as_u64().is_some(), "{body}");

    // And without `include_base`, no second pass and no field.
    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/generate"),
        json!({"prompt": "again", "chat": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["text"], "[chat=false n=128] answer");
    assert!(body.get("base_text").is_none(), "{body}");

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn a_generation_request_is_validated_before_it_reaches_the_loop() {
    let fixture = Fixture::new("inference-generate-invalid");
    let (router, id) = live(&fixture, recipe(&fixture, "generate-invalid")).await;
    let uri = format!("/v1/runs/{id}/generate");

    for (body, pointer) in [
        (json!({"prompt": "   "}), "/prompt"),
        (
            json!({"prompt": "hi", "max_new_tokens": 0}),
            "/max_new_tokens",
        ),
        (json!({"prompt": "hi", "temperature": -1.0}), "/temperature"),
        (json!({"prompt": "hi", "top_p": 1.5}), "/top_p"),
    ] {
        let (status, answer) = post(&router, &uri, body.clone()).await;
        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{body} -> {answer}"
        );
        assert_eq!(answer["errors"][0]["pointer"], pointer, "{answer}");
    }
    // And a field outside the request schema, which is refused by the
    // deserializer rather than by a handler.
    let (status, _) = post(&router, &uri, json!({"prompt": "hi", "top_k": 40})).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn a_paused_run_still_answers_a_question() {
    let fixture = Fixture::new("inference-paused");
    let (router, id) = live(&fixture, recipe_with_schedules(&fixture, "paused")).await;
    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::OK);
    wait_for_status(&router, &id, "paused").await;

    // The model is loaded and idle, which is the cheapest moment there is to ask
    // it something - and the command is served from inside the blocked callback.
    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/evaluate")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["loss"], 0.375);
    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/generate"),
        json!({"prompt": "while paused"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Still paused afterwards: answering a question is not resuming.
    let (_, view) = get(&router, &format!("/v1/runs/{id}")).await;
    assert_eq!(view["status"], "paused");

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn a_run_that_is_not_executing_its_loop_refuses_both() {
    let fixture = Fixture::new("inference-terminal");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "done")).await;
    let id = created["id"].as_str().expect("an id").to_string();
    wait_for_terminal(&router, &id).await;

    for (uri, body) in [
        (format!("/v1/runs/{id}/evaluate"), json!({})),
        (format!("/v1/runs/{id}/generate"), json!({"prompt": "hi"})),
    ] {
        let (status, answer) = post(&router, &uri, body).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {answer}");
        assert!(
            answer["detail"]
                .as_str()
                .unwrap_or_default()
                .contains("completed"),
            "{answer}"
        );
    }

    // And an unknown run is a 404 on both, like everywhere else.
    let (status, _) = post(
        &router,
        "/v1/runs/00000000-0000-4000-8000-000000000000/generate",
        json!({"prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_request_that_outlives_its_bound_is_a_504() {
    let fixture = Fixture::new("inference-timeout");
    // Two seconds before the first poll against a one-second bound: the run is
    // alive and healthy, it simply has not reached a callback yet. That is a
    // timeout, not a failure of the run, and the two must not read the same.
    // `stalled` rather than `slow`: with `slow` the first poll follows
    // `running` at once, and a runner that schedules the engine thread late
    // serves the request there, at step 5.
    let engine = FakeEngine::stalled(10, Duration::from_millis(2000));
    let mut state = state_of(&fixture, engine, false);
    state.config = std::sync::Arc::new(retrograd_server::ServerConfig {
        command_timeout_seconds: Some(1),
        ..(*state.config).clone()
    });
    let router = build_router(state);

    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "slow")).await;
    let id = created["id"].as_str().expect("an id").to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/generate"),
        json!({"prompt": "are you there"}),
    )
    .await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(
        body["type"]
            .as_str()
            .unwrap_or_default()
            .ends_with("/timeout"),
        "{body}"
    );

    let _ = post(
        &router,
        &format!("/v1/runs/{id}/cancel"),
        json!({"at": "now"}),
    )
    .await;
    wait_for_terminal(&router, &id).await;
}
