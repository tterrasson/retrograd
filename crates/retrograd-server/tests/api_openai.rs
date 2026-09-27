//! `/v1/models` and `/v1/chat/completions` over the runs of this server.
//!
//! Against the fake engine and a fake serving loader: what is under test is what
//! a model id means here - which weights a run, a checkpoint or a base names,
//! who holds the device, and that every failure reaches the client in the
//! envelope an OpenAI SDK reads.

mod support;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use retrograd_openai::testing::FakeLoader;
use retrograd_server::AppState;
use serde_json::{Value, json};
use support::*;

fn serving(fixture: &Fixture, engine: Arc<FakeEngine>) -> (Router, FakeLoader, AppState) {
    let loader = FakeLoader::new("from the run");
    let state = state_of(fixture, engine, false).with_serving_loader(Arc::new(loader.clone()));
    (build_router(state.clone()), loader, state)
}

fn chat(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hello"}]})
}

async fn post_chat(router: &Router, model: &str) -> (StatusCode, Value) {
    post(router, "/v1/chat/completions", chat(model)).await
}

/// A finished run whose adapter and two checkpoints are on disk.
async fn finished(fixture: &Fixture, router: &Router) -> String {
    let (status, body) = post(router, "/v1/runs", recipe_with_schedules(fixture, "served")).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().expect("an id").to_string();
    wait_for_terminal(router, &id).await;
    let (_, artifacts) = get(router, &format!("/v1/runs/{id}/artifacts")).await;
    let adapter = artifacts["artifacts"]
        .as_array()
        .expect("an inventory")
        .iter()
        .find(|entry| entry["name"] == "adapter")
        .and_then(|entry| entry["path"].as_str())
        .expect("a LoRA run writes an adapter")
        .to_string();
    std::fs::write(&adapter, b"not a real adapter").expect("write the adapter");
    let directory = fixture.dir.join("ckpt");
    std::fs::create_dir_all(&directory).expect("the checkpoint directory");
    for (id, step) in [("step-000000000010", 10), ("best", 5)] {
        write_checkpoint(
            &directory,
            id,
            step,
            "sft",
            "signature",
            &fixture.dir.join("model.gguf"),
        );
    }
    id
}

#[tokio::test]
async fn a_finished_run_lists_every_set_of_weights_it_left() {
    let fixture = Fixture::new("openai-list");
    let (router, _, _) = serving(&fixture, FakeEngine::succeeding());
    let id = finished(&fixture, &router).await;

    let (status, body) = get(&router, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "list");
    let ids = body["data"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|card| card["id"].as_str().expect("an id").to_string())
        .collect::<Vec<_>>();
    for expected in [
        id.clone(),
        format!("{id}@final"),
        format!("{id}@step-000000000010"),
        format!("{id}@best"),
        format!("{id}@base"),
    ] {
        assert!(ids.contains(&expected), "{expected} missing from {ids:?}");
    }
    let (status, card) = get(&router, &format!("/v1/models/{id}@best")).await;
    assert_eq!(status, StatusCode::OK, "{card}");
    assert_eq!(card["owned_by"], "retrograd");
}

#[tokio::test]
async fn each_id_loads_the_weights_it_names() {
    let fixture = Fixture::new("openai-resolve");
    let (router, loader, _) = serving(&fixture, FakeEngine::succeeding());
    let id = finished(&fixture, &router).await;
    let model = fixture.dir.join("model.gguf");
    let checkpoint = |name: &str| fixture.dir.join("ckpt").join(format!("{name}.gguf"));

    for (suffix, fingerprint, adapter) in [
        ("", "final", None),
        ("@final", "final", None),
        ("@best", "best", Some(checkpoint("best"))),
        (
            "@step-10",
            "step-000000000010",
            Some(checkpoint("step-000000000010")),
        ),
        (
            "@latest",
            "step-000000000010",
            Some(checkpoint("step-000000000010")),
        ),
        ("@base", "base", None),
    ] {
        let (status, body) = post_chat(&router, &format!("{id}{suffix}")).await;
        assert_eq!(status, StatusCode::OK, "{suffix}: {body}");
        assert_eq!(body["system_fingerprint"], format!("{id}@{fingerprint}"));
        assert_eq!(body["choices"][0]["message"]["content"], "from the run");
        let loaded = loader.specs().pop().expect("a load");
        assert_eq!(loaded.model, model, "{suffix}");
        match (suffix, adapter) {
            ("" | "@final", _) => assert!(
                loaded
                    .adapter
                    .as_ref()
                    .is_some_and(|adapter| adapter.ends_with("adapter.gguf")),
                "{suffix}: {loaded:?}"
            ),
            (_, expected) => assert_eq!(loaded.adapter, expected, "{suffix}"),
        }
    }

    let (status, body) = post_chat(&router, &format!("{id}@step-99")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    let message = body["error"]["message"].as_str().expect("a message");
    assert!(
        message.contains("@best") && message.contains("@final"),
        "the refusal lists what the run can serve: {message}"
    );

    for unknown in [uuid::Uuid::new_v4().to_string(), "base".to_string()] {
        let (status, body) = post_chat(&router, &unknown).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{unknown}: {body}");
        assert_eq!(body["error"]["param"], "model");
    }
}

#[tokio::test]
async fn a_held_device_refuses_a_load_and_names_why() {
    let fixture = Fixture::new("openai-busy");
    let (router, _, state) = serving(&fixture, FakeEngine::succeeding());
    let id = finished(&fixture, &router).await;

    // What a training run holds while it trains.
    let permit = state
        .device
        .clone()
        .try_acquire_owned()
        .expect("the device is free");
    let (status, headers, body) = send_raw(
        &router,
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(chat(&id).to_string()))
            .expect("request"),
    )
    .await;
    let body: Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(headers.starts_with("application/json"), "{headers}");
    assert_eq!(body["error"]["type"], "server_busy");
    assert_eq!(body["error"]["code"], "device_busy");
    drop(permit);

    let (status, body) = post_chat(&router, &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn a_run_that_needs_the_device_gets_it_back_from_the_session() {
    let fixture = Fixture::new("openai-yield");
    let (router, loader, state) = serving(&fixture, FakeEngine::succeeding());
    let id = finished(&fixture, &router).await;

    let (status, body) = post_chat(&router, &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(loader.loads(), 1);
    assert!(state.serving.loaded().is_some(), "the model stays loaded");
    assert!(
        state.device.clone().try_acquire_owned().is_err(),
        "and holds the device while it is"
    );

    // With the one permit lent to the session, this run would sit in `queued`
    // for the whole idle bound if nobody asked for it back.
    let (status, body) = post(&router, "/v1/runs", recipe(&fixture, "after")).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let next = body["id"].as_str().expect("an id").to_string();
    let view = wait_for_terminal(&router, &next).await;
    assert_eq!(view["status"], "completed", "{view}");
    assert!(state.serving.loaded().is_none());
}

#[tokio::test]
async fn a_live_run_answers_on_its_own_thread() {
    let fixture = Fixture::new("openai-live");
    let (router, loader, _) = serving(&fixture, FakeEngine::slow(40, Duration::from_millis(25)));
    let (status, body) = post(&router, "/v1/runs", recipe_with_schedules(&fixture, "live")).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().expect("an id").to_string();
    wait_for_status(&router, &id, "running").await;

    let (_, models) = get(&router, "/v1/models").await;
    let card = models["data"]
        .as_array()
        .expect("a list")
        .iter()
        .find(|card| card["id"] == id.as_str())
        .cloned()
        .expect("a live run is listed");
    assert_eq!(card["live"], true);

    // The fake loop has no trainer to lend, and says so from inside its
    // callback: the request went down the run's own channel, and nothing was
    // loaded beside it.
    let (status, body) = post_chat(&router, &id).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("does not lend its trainer"),
        "{body}"
    );
    assert_eq!(loader.loads(), 0);

    let (status, body) = post_chat(&router, &format!("{id}@best")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "device_busy");
    let (status, body) = post_chat(&router, &format!("{id}@final")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");

    let _ = post_empty(&router, &format!("/v1/runs/{id}/cancel")).await;
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn failures_on_these_routes_are_openai_envelopes() {
    let fixture = Fixture::new("openai-envelopes");
    let loader = FakeLoader::default();
    let mut state =
        state_of(&fixture, FakeEngine::succeeding(), false).with_serving_loader(Arc::new(loader));
    let mut config = (*state.config).clone();
    config.auth_token = Some("s3cret".into());
    config.serving.max_body_bytes = Some(64);
    state.config = Arc::new(config);
    let router = build_router(state);

    // The SDKs show an empty message for a problem document.
    let (status, body) = post_chat(&router, "whatever").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert_eq!(body["error"]["type"], "authentication_error");

    let authorized = |uri: &str, body: String| {
        Request::post(uri)
            .header("authorization", "Bearer s3cret")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .expect("request")
    };
    let long = json!({"model": "m", "messages": [{"role": "user", "content": "x".repeat(200)}]});
    let (status, body) = send(
        &router,
        authorized("/v1/chat/completions", long.to_string()),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert!(body["error"].is_object(), "{body}");

    let (status, body) = send(
        &router,
        authorized("/v1/chat/completions", "{not json".into()),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    // The rest of the API keeps its problem documents.
    let (status, body) = send(
        &router,
        Request::get("/v1/runs/nope")
            .header("authorization", "Bearer s3cret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["type"].is_string(), "{body}");
}

#[tokio::test]
async fn a_disabled_endpoint_answers_404_in_the_envelope() {
    let fixture = Fixture::new("openai-disabled");
    let mut state = state_of(&fixture, FakeEngine::succeeding(), false);
    let mut config = (*state.config).clone();
    config.serving.enabled = Some(false);
    state.config = Arc::new(config);
    let router = build_router(state);
    let (status, body) = get(&router, "/v1/models").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "not_found_error");
}

/// A `run.json` written before the serving metadata existed still names its
/// base and its context in the rendered configuration, which is enough.
#[tokio::test]
async fn a_run_older_than_the_serving_metadata_is_served_from_its_configuration() {
    let fixture = Fixture::new("openai-legacy");
    let (router, loader, _) = serving(&fixture, FakeEngine::succeeding());
    let id = finished(&fixture, &router).await;
    let run_json = fixture.state_dir().join(&id).join("run.json");
    let mut stored: Value =
        serde_json::from_slice(&std::fs::read(&run_json).expect("read run.json")).expect("JSON");
    assert!(
        stored["artifacts"]["serving"].is_object(),
        "a new run records it: {stored}"
    );
    stored["artifacts"]
        .as_object_mut()
        .expect("an object")
        .remove("serving");
    std::fs::write(&run_json, serde_json::to_vec(&stored).expect("JSON")).expect("write");
    drop(router);

    // A new process over the same state directory.
    let (router, _, _) = serving(&fixture, FakeEngine::succeeding());
    let (status, body) = post_chat(&router, &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(loader.loads(), 0, "the first process loaded nothing");
}
