//! The contract end to end, without a model: requests shaped the way the
//! clients that matter write them, through the router, against the fake
//! session.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_core::Device;
use retrograd_openai::testing::FakeLoader;
use retrograd_openai::{
    Endpoint, OpenAiEnvelope, Session, SessionOptions, SingleModel, Unshared, WeightsSpec,
};
use serde_json::{Value, json};
use tower::ServiceExt;

fn spec(adapter: Option<&str>) -> WeightsSpec {
    WeightsSpec {
        model: "base.gguf".into(),
        adapter: adapter.map(Into::into),
        n_ctx: 4096,
        device: Device::Cpu,
        chat_template_variables: None,
    }
}

fn router_with(reply: &str) -> (Router, FakeLoader) {
    let loader = FakeLoader::new(reply);
    let session = Session::new(
        Arc::new(loader.clone()),
        Arc::new(Unshared),
        SessionOptions::default(),
    )
    .expect("session");
    let source = SingleModel {
        name: "tuned".into(),
        spec: spec(Some("tuned.gguf")),
        base: Some(spec(None)),
        created: 1,
    };
    let router = Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(source),
            session: Arc::new(session),
            timeout: Duration::from_secs(30),
            enabled: true,
        }),
    );
    (router, loader)
}

async fn send(router: &Router, request: Request<Body>) -> (StatusCode, http::HeaderMap, Vec<u8>) {
    let response = router.clone().oneshot(request).await.expect("a response");
    let status = response.status();
    let headers = response.headers().clone();
    assert!(
        status.is_success() || response.extensions().get::<OpenAiEnvelope>().is_some(),
        "an error left without its envelope marker"
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("a body")
        .to_bytes()
        .to_vec();
    (status, headers, body)
}

async fn post(router: &Router, body: Value) -> (StatusCode, http::HeaderMap, Value) {
    let (status, headers, body) = send(
        router,
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request"),
    )
    .await;
    let body = serde_json::from_slice(&body).expect("a JSON body");
    (status, headers, body)
}

async fn get(router: &Router, path: &str) -> (StatusCode, Value) {
    let (status, _, body) = send(
        router,
        Request::get(path).body(Body::empty()).expect("request"),
    )
    .await;
    (status, serde_json::from_slice(&body).expect("a JSON body"))
}

/// What the Python SDK sends for `client.chat.completions.create(...)`.
fn python_sdk() -> Value {
    json!({
        "model": "tuned",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."}
        ],
        "temperature": 0.2,
        "max_tokens": 64,
        "stream": false
    })
}

/// What lm-eval's `local-chat-completions` sends: `seed`, `stop` as a list,
/// `max_tokens`, and a temperature of zero for greedy decoding.
fn lm_eval() -> Value {
    json!({
        "model": "tuned",
        "messages": [{"role": "user", "content": "Question: 2+2?\nAnswer:"}],
        "max_tokens": 256,
        "temperature": 0,
        "seed": 1234,
        "stop": ["Question:", "</s>", "<|im_end|>"]
    })
}

/// What Open WebUI sends: a stream, usage requested, and fields this server
/// has no use for.
fn open_webui() -> Value {
    json!({
        "model": "tuned",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "user": "someone",
        "metadata": {"chat_id": "abc"},
        "parallel_tool_calls": true
    })
}

#[tokio::test]
async fn models_are_listed_and_looked_up_in_the_openai_shape() {
    let (router, _) = router_with("hi");
    let (status, body) = get(&router, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let ids = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|card| card["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["tuned", "base"]);
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["owned_by"], "retrograd");

    let (status, body) = get(&router, "/v1/models/tuned").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], "tuned");

    let (status, body) = get(&router, "/v1/models/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn a_python_sdk_request_gets_a_chat_completion() {
    let (router, loader) = router_with("Hello!");
    let (status, _, body) = post(&router, python_sdk()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["model"], "tuned");
    assert_eq!(body["system_fingerprint"], "tuned");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    let usage = &body["usage"];
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
    assert_eq!(loader.specs(), [spec(Some("tuned.gguf"))]);
}

#[tokio::test]
async fn an_lm_eval_request_is_greedy_and_stops_where_it_was_told() {
    let (router, _) = router_with("4\nQuestion: next");
    let (status, _, body) = post(&router, lm_eval()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "4\n");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn the_base_is_served_under_its_own_id() {
    let (router, loader) = router_with("plain");
    let mut request = python_sdk();
    request["model"] = json!("base");
    let (status, _, body) = post(&router, request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["system_fingerprint"], "base");
    assert_eq!(loader.specs(), [spec(None)]);
}

#[tokio::test]
async fn an_open_webui_stream_is_role_content_finish_usage_done() {
    let (router, _) = router_with("streamed");
    let (status, headers, body) = send(
        &router,
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(open_webui().to_string()))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let text = String::from_utf8(body).unwrap();
    let events = text
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(events.last(), Some(&"[DONE]"), "{text}");
    let chunks = events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str::<Value>(event).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "streamed");
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert!(chunks[3]["usage"]["total_tokens"].as_u64().unwrap() > 0);
    let ids = chunks
        .iter()
        .map(|chunk| chunk["id"].as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), 1, "one id for every chunk of one answer");
}

#[tokio::test]
async fn a_failure_after_the_stream_opened_is_an_error_event() {
    let (router, _) = router_with("x");
    let mut request = open_webui();
    // Resolved before the stream opens, validated before the stream opens:
    // only something the model itself refuses is left for an in-stream error.
    request["max_tokens"] = json!(1_000_000);
    let (status, _, body) = send(
        &router,
        Request::post("/v1/chat/completions")
            .body(Body::from(request.to_string()))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    let events = text
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2, "{text}");
    let error: Value = serde_json::from_str(events[0]).unwrap();
    assert_eq!(error["error"]["code"], "context_length_exceeded");
    assert_eq!(events[1], "[DONE]");
}

#[tokio::test]
async fn refusals_are_openai_envelopes_with_the_field_named() {
    let (router, _) = router_with("x");
    let mut request = python_sdk();
    request["logprobs"] = json!(true);
    let (status, _, body) = post(&router, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "logprobs");
    assert_eq!(body["error"]["code"], "unsupported_parameter");

    let mut request = python_sdk();
    request["model"] = json!("unknown");
    let (status, _, body) = post(&router, request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["param"], "model");

    let (status, _, body) = post(&router, json!({"model": "tuned"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("messages")
    );
}

#[tokio::test]
async fn an_unreadable_call_is_reported_beside_a_successful_answer() {
    let (router, _) = router_with("<tool_call>{broken</tool_call>");
    let mut request = python_sdk();
    request["tools"] = json!([{"type": "function", "function": {"name": "move"}}]);
    let (status, headers, body) = post(&router, request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(headers[retrograd_openai::PARSE_ERRORS_HEADER], "1");
    assert!(body["choices"][0]["message"].get("tool_calls").is_none());
}

#[tokio::test]
async fn a_disabled_endpoint_is_a_404_in_the_envelope() {
    let session = Session::fake();
    let router = retrograd_openai::router(Endpoint {
        source: Arc::new(SingleModel {
            name: "m".into(),
            spec: spec(None),
            base: None,
            created: 0,
        }),
        session: Arc::new(session),
        timeout: Duration::from_secs(1),
        enabled: false,
    });
    let (status, body) = get(&router, "/models").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found_error");
}
