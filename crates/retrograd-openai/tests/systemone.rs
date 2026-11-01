//! The `/v1/systemone` contract, without a model: the answer document of
//! llama-server (answers in the order of the questions, `usage` with no
//! output tokens), the `model` routing, and the statuses - 400 for a body the
//! decision logic refuses, 501 for a model that is not a decision model, for
//! images, and for a decision model asked to generate.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_core::Device;
use retrograd_decision::Json;
use retrograd_openai::testing::FakeLoader;
use retrograd_openai::{
    Endpoint, OpenAiEnvelope, Session, SessionOptions, SingleModel, Unshared, WeightsSpec,
};
use serde_json::{Value, json};
use tower::ServiceExt;

fn spec(adapter: Option<&str>) -> WeightsSpec {
    WeightsSpec {
        model: "decider.gguf".into(),
        adapter: adapter.map(Into::into),
        n_ctx: 4096,
        device: Device::Cpu,
        n_seq_max: None,
        chat_template_variables: None,
        decision: Default::default(),
    }
}

/// A server of one model, `decider`, plus its base under `base` when
/// `with_adapter`.
fn router_with(loader: FakeLoader, with_adapter: bool) -> Router {
    let session = Arc::new(
        Session::new(
            Arc::new(loader),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("session"),
    );
    let (served, base) = match with_adapter {
        true => (spec(Some("decider-lora.gguf")), Some(spec(None))),
        false => (spec(None), None),
    };
    Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(SingleModel {
                name: "decider".into(),
                spec: served,
                base,
                created: 1,
            }),
            session,
            timeout: Duration::from_secs(30),
            enabled: true,
            store: Arc::new(retrograd_openai::MemoryStore::default()),
        }),
    )
}

/// The status and the body text, the envelope marker checked on errors.
async fn post(router: &Router, path: &str, body: String) -> (StatusCode, String) {
    let response = router
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("a response");
    let status = response.status();
    assert!(
        status.is_success() || response.extensions().get::<OpenAiEnvelope>().is_some(),
        "an error left without its envelope marker"
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("a body")
        .to_bytes();
    (status, String::from_utf8(body.to_vec()).expect("UTF-8"))
}

/// Three questions whose ids are not in alphabetical order: the answers must
/// keep the order written.
const QUESTIONS: &str = r#"{
    "zeta": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "shipping": null, "technical": "broken"}},
    "alpha": {"type": "noul", "instructions": "Is it urgent?"},
    "mid": {"type": "score", "instructions": "How angry?",
            "criteria": ["calm", "annoyed", "angry"]}
}"#;

fn decision_body(model: Option<&str>) -> String {
    let model = model.map_or(String::new(), |model| format!(r#""model": "{model}", "#));
    format!(
        r#"{{{model}"state": {{"message": "charged twice", "plan": "pro"}}, "questions": {QUESTIONS}}}"#
    )
}

#[tokio::test]
async fn answers_come_back_in_the_order_of_the_questions_with_llama_servers_document() {
    let router = router_with(FakeLoader::default().decision(), false);
    let (status, text) = post(&router, "/v1/systemone", decision_body(Some("decider"))).await;
    assert_eq!(status, StatusCode::OK, "{text}");

    // Read with the key order kept: a `Value` would sort it.
    let ordered: Json = serde_json::from_str(&text).expect("JSON");
    let keys = |value: &Json| -> Vec<String> {
        value
            .as_object()
            .expect("an object")
            .keys()
            .cloned()
            .collect()
    };
    assert_eq!(keys(&ordered), ["model", "answers", "usage"]);
    assert_eq!(keys(&ordered["answers"]), ["zeta", "alpha", "mid"]);
    assert_eq!(
        keys(&ordered["answers"]["zeta"]["probabilities"]),
        ["billing", "shipping", "technical"]
    );

    let body: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(body["model"], "decider");
    // The fake scores the first option highest.
    assert_eq!(body["answers"]["zeta"]["type"], "choice");
    assert_eq!(body["answers"]["zeta"]["choice"], "billing");
    assert_eq!(body["answers"]["alpha"]["type"], "noul");
    assert!(body["answers"]["alpha"]["noul"].as_f64().expect("noul") < 0.5);
    assert_eq!(body["answers"]["mid"]["type"], "score");
    assert_eq!(body["answers"]["mid"]["legend"]["2"], "angry");
    // One token per option plus one, per question.
    assert_eq!(body["usage"]["input_tokens"], 4 + 3 + 4);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn a_body_without_model_is_served_when_one_model_is_and_refused_otherwise() {
    let router = router_with(FakeLoader::default().decision(), false);
    let (status, text) = post(&router, "/v1/systemone", decision_body(None)).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let body: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(body["model"], "decider");

    // `decider` and `base`: the id is required.
    let router = router_with(FakeLoader::default().decision(), true);
    let (status, text) = post(&router, "/v1/systemone", decision_body(None)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    let body: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(body["error"]["param"], "model");
    // Both ids answer, the adapter one and the base.
    for model in ["decider", "base"] {
        let (status, text) = post(&router, "/v1/systemone", decision_body(Some(model))).await;
        assert_eq!(status, StatusCode::OK, "{model}: {text}");
    }

    let (status, _) = post(&router, "/v1/systemone", decision_body(Some("nope"))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_body_the_decision_logic_refuses_is_a_bad_request_with_its_message() {
    let router = router_with(FakeLoader::default().decision(), false);
    for (body, message) in [
        ("{".to_owned(), "invalid JSON"),
        ("[1]".to_owned(), "the body must be a JSON object"),
        (
            format!(r#"{{"questions": {QUESTIONS}}}"#),
            "\"state\" must be provided",
        ),
        (
            r#"{"state": "s", "questions": {}}"#.to_owned(),
            "\"questions\" must be a non-empty object",
        ),
        (
            r#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}}"#
                .to_owned(),
            "questions.q: \"criteria\" must be an array of 2 to 10 levels",
        ),
    ] {
        let (status, text) = post(&router, "/v1/systemone", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
        let error: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|text| text.contains(message)),
            "{body}: {text}"
        );
    }
}

#[tokio::test]
async fn images_on_a_text_decision_model_are_not_implemented() {
    let router = router_with(FakeLoader::default().decision(), false);
    let body = format!(
        r#"{{"state": "s", "images": ["data:image/png;base64,iVBORw0KGgo="], "questions": {QUESTIONS}}}"#
    );
    let (status, text) = post(&router, "/v1/systemone", body).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{text}");
    let error: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(error["error"]["type"], "not_supported_error");
}

#[tokio::test]
async fn a_model_that_generates_does_not_decide_and_a_decision_model_does_not_generate() {
    let router = router_with(FakeLoader::default(), false);
    let (status, text) = post(&router, "/v1/systemone", decision_body(Some("decider"))).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{text}");
    let error: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(error["error"]["type"], "not_supported_error");
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|text| text.contains("not a decision model")),
        "{text}"
    );

    let router = router_with(FakeLoader::default().decision(), false);
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model": "decider", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "/v1/completions",
            json!({"model": "decider", "prompt": "hi"}),
        ),
        ("/v1/embeddings", json!({"model": "decider", "input": "hi"})),
    ] {
        let (status, text) = post(&router, path, body.to_string()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {text}");
        let error: Value = serde_json::from_str(&text).expect("JSON");
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|text| text.contains("/v1/systemone")),
            "{path}: {text}"
        );
    }
}

#[tokio::test]
async fn the_model_list_announces_the_route() {
    let router = router_with(FakeLoader::default().decision(), false);
    let response = router
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["capabilities"]["systemone"], true);
}
