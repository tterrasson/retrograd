//! The legacy Completions API contract, without a model: a raw prompt (no
//! chat template), `n` choices, the legacy logprobs shape, `echo`, `stop`, and
//! streaming - through the router against the fake session.

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

fn spec() -> WeightsSpec {
    WeightsSpec {
        model: "base.gguf".into(),
        adapter: None,
        n_ctx: 4096,
        device: Device::Cpu,
        n_seq_max: None,
        chat_template_variables: None,
        decision: Default::default(),
    }
}

fn router_with(reply: &str) -> Router {
    let session = Arc::new(
        Session::new(
            Arc::new(FakeLoader::new(reply)),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("session"),
    );
    Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(SingleModel {
                name: "tuned".into(),
                spec: spec(),
                base: None,
                created: 1,
            }),
            session,
            timeout: Duration::from_secs(30),
            enabled: true,
            store: Arc::new(retrograd_openai::MemoryStore::default()),
        }),
    )
}

async fn post(router: &Router, body: Value) -> (StatusCode, Value) {
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
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
    (status, serde_json::from_slice(&body).expect("a JSON body"))
}

/// A string prompt is answered raw: the usage's prompt_tokens is the prompt's
/// own byte length - a chat template would have made it the rendered JSON's.
#[tokio::test]
async fn a_string_prompt_is_one_raw_completion() {
    let router = router_with("hello there");
    let (status, body) = post(&router, json!({"model": "tuned", "prompt": "hi"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "text_completion");
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[0]["text"], "hello there");
    assert_eq!(choices[0]["finish_reason"], "stop");
    // The raw prompt "hi" is two tokens; no template was rendered.
    assert_eq!(body["usage"]["prompt_tokens"], 2, "{body}");
    assert_eq!(
        body["usage"]["total_tokens"].as_u64().unwrap(),
        body["usage"]["prompt_tokens"].as_u64().unwrap()
            + body["usage"]["completion_tokens"].as_u64().unwrap()
    );
}

/// A list of prompts answers one choice per prompt, and `n` multiplies them;
/// `echo` prepends the prompt to the text.
#[tokio::test]
async fn prompt_lists_n_and_echo() {
    let router = router_with("X");
    let (status, body) = post(
        &router,
        json!({"model": "tuned", "prompt": ["a", "bb"], "n": 2, "echo": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 4, "{body}");
    let indexes: Vec<u64> = choices
        .iter()
        .map(|c| c["index"].as_u64().unwrap())
        .collect();
    assert_eq!(indexes, vec![0, 1, 2, 3]);
    assert_eq!(choices[0]["text"], "aX");
    assert_eq!(choices[2]["text"], "bbX");
    assert_eq!(body["usage"]["prompt_tokens"], 3, "{body}");
}

/// The legacy logprobs shape: parallel per-token arrays and a token→logprob
/// map for the top pairs.
#[tokio::test]
async fn legacy_logprobs_are_parallel_arrays() {
    let router = router_with("abc");
    let (status, body) = post(
        &router,
        json!({"model": "tuned", "prompt": "hi", "logprobs": 3}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let logprobs = &body["choices"][0]["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 3, "{body}"); // "abc", the EOG carries no piece
    assert_eq!(tokens[0], "a");
    assert_eq!(
        logprobs["token_logprobs"].as_array().unwrap().len(),
        tokens.len()
    );
    assert_eq!(
        logprobs["text_offset"].as_array().unwrap(),
        &vec![json!(0), json!(1), json!(2)]
    );
    let top = logprobs["top_logprobs"][0].as_object().expect("a map");
    assert!(top.values().all(|v| v.is_number()), "{body}");
}

/// `stop` truncates the text at its earliest match and reads as a stop.
#[tokio::test]
async fn a_stop_sequence_truncates() {
    let router = router_with("one. two! three");
    let (status, body) = post(
        &router,
        json!({"model": "tuned", "prompt": "x", "stop": ["!", "."], "max_tokens": 20}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["text"], "one");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

/// Refused with the field named: `suffix`, `best_of`, a token-array prompt.
#[tokio::test]
async fn unsupported_fields_are_refused_with_the_field_named() {
    let router = router_with("x");
    for (body, param) in [
        (
            json!({"model": "tuned", "prompt": "x", "suffix": "y"}),
            "suffix",
        ),
        (
            json!({"model": "tuned", "prompt": "x", "best_of": 3}),
            "best_of",
        ),
        (json!({"model": "tuned", "prompt": [[1, 2]]}), "prompt[0]"),
        (
            json!({"model": "tuned", "prompt": "x", "logprobs": 21}),
            "logprobs",
        ),
        (json!({"model": "tuned", "prompt": "x", "n": 0}), "n"),
    ] {
        let (status, body) = post(&router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{param}: {body}");
        assert_eq!(body["error"]["param"], param, "{param}: {body}");
    }
}

/// Streaming: one chunk per decoded piece in `choices[].text`, a finish chunk,
/// then `[DONE]`.
#[tokio::test]
async fn a_streamed_completion_ends_with_done() {
    let router = router_with("streamed answer");
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "tuned", "prompt": "hi", "stream": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("[DONE]"), "{text}");
    let chunks: Vec<Value> = text
        .split("\n\n")
        .filter_map(|event| {
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            if data == "[DONE]" {
                return None;
            }
            serde_json::from_str(data).ok()
        })
        .collect();
    assert!(!chunks.is_empty());
    assert!(
        chunks.iter().all(|c| c["object"] == "text_completion"),
        "{chunks:?}"
    );
    let mut pieces = String::new();
    let mut finish = None;
    for chunk in &chunks {
        for choice in chunk["choices"].as_array().unwrap() {
            pieces.push_str(choice["text"].as_str().unwrap());
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                finish = Some(reason.to_owned());
            }
        }
    }
    assert_eq!(pieces, "streamed answer");
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[tokio::test]
async fn zero_logprobs_still_reports_sampled_token_scores() {
    let (status, body) = post(
        &router_with("abc"),
        json!({"model":"tuned", "prompt":"hi", "logprobs":0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["logprobs"]["token_logprobs"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        body["choices"][0]["logprobs"]["top_logprobs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| entry.as_object().unwrap().is_empty())
    );
    let (status, body) = post(
        &router_with("abc"),
        json!({"model":"tuned", "prompt":"hi", "logprobs":0, "echo":true}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "echo");
}

#[tokio::test]
async fn streaming_echo_is_once_per_choice_and_prompt() {
    let router = router_with("X");
    let response = router
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"tuned", "prompt":["a", "bb"],
            "n":2, "echo":true, "stream":true})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let mut choices = std::collections::BTreeMap::<u64, String>::new();
    for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(line).unwrap();
        assert_eq!(chunk["object"], "text_completion", "{chunk}");
        for choice in chunk["choices"].as_array().unwrap() {
            choices
                .entry(choice["index"].as_u64().unwrap())
                .or_default()
                .push_str(choice["text"].as_str().unwrap());
        }
    }
    assert_eq!(
        choices.into_values().collect::<Vec<_>>(),
        ["aX", "aX", "bbX", "bbX"]
    );
}
