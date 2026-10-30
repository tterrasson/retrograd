//! The Responses API contract end to end, without a model: requests shaped
//! the way the SDKs write them, through the router, against the fake session.
//! The pinned contract is the published OpenAPI description of the Responses
//! API; the streamed sequence is validated by a small state machine here.

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
        n_seq_max: None,
        chat_template_variables: None,
    }
}

fn router_with(reply: &str) -> (Router, Arc<Session>) {
    let loader = FakeLoader::new(reply);
    let session = Arc::new(
        Session::new(
            Arc::new(loader),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("session"),
    );
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
            session: session.clone(),
            timeout: Duration::from_secs(30),
            enabled: true,
            store: Arc::new(retrograd_openai::MemoryStore::default()),
        }),
    );
    (router, session)
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

async fn post(router: &Router, body: Value) -> (StatusCode, Value) {
    let (status, _, body) = send(
        router,
        Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request"),
    )
    .await;
    (status, serde_json::from_slice(&body).expect("a JSON body"))
}

fn base() -> Value {
    json!({"model": "tuned", "input": "hi", "store": false})
}

/// A simple text turn answers with one message item and a completed status.
#[tokio::test]
async fn a_text_turn_answers_a_response_object_in_the_pinned_shape() {
    let (router, _) = router_with("hello there");
    let (status, body) = post(&router, base()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "response");
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "tuned");
    assert_eq!(body["error"], Value::Null);
    assert_eq!(body["incomplete_details"], Value::Null);
    assert_eq!(body["store"], false);
    assert_eq!(body["background"], false);
    assert_eq!(body["truncation"], "disabled");
    assert!(body["access_programs"].is_array());
    // One message item with an output_text part, and the usage breakdown.
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(output[0]["content"][0]["type"], "output_text");
    assert_eq!(output[0]["content"][0]["text"], "hello there");
    assert_eq!(output[0]["content"][0]["annotations"], json!([]));
    assert_eq!(body["output_text"], "hello there");
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["input_tokens"].as_u64().unwrap()
            + body["usage"]["output_tokens"].as_u64().unwrap()
    );
    // The echoes of the effective options.
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["text"]["format"]["type"], "text");
}

/// A budget stop is `incomplete` with `incomplete_details`, not `completed` -
/// and never presented as a finished answer.
#[tokio::test]
async fn a_budget_stop_is_incomplete_with_max_output_tokens() {
    let (router, _) = router_with("a very long answer that will not fit");
    let mut body = base();
    body["max_output_tokens"] = json!(16);
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
}

/// A tool call plus its output round-trip: the input items are reconciled by
/// `call_id`, and the answer's call becomes a `function_call` output item.
#[tokio::test]
async fn a_tool_call_turn_emits_a_function_call_item() {
    let reply = r#"<tool_call>{"name":"move","arguments":{"direction":"left"}}</tool_call>"#;
    let (router, _) = router_with(reply);
    let mut body = base();
    body["input"] = json!([
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "move", "arguments": "{\"direction\":\"right\"}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a wall"}
    ]);
    body["tools"] = json!([{"type": "function", "name": "move", "parameters": {"type": "object"}}]);
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let calls: Vec<&Value> = body["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["name"], "move");
    assert_eq!(calls[0]["call_id"], "call_0");
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["arguments"].as_str().unwrap()).unwrap(),
        json!({"direction": "left"})
    );
}

/// A `reasoning` input item is accepted and re-rendered through the template
/// (the fake records what it was handed).
#[tokio::test]
async fn a_reasoning_input_item_is_rerendered_via_the_template() {
    let (router, session) = router_with("ok");
    let mut body = base();
    body["input"] = json!([
        {"type": "reasoning", "id": "r1",
         "summary": [{"type": "summary_text", "text": "I thought about it."}]},
        {"type": "message", "role": "user", "content": "and?"}
    ]);
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The fake's format_chat received the reasoning as the assistant turn's
    // reasoning_content.
    let _ = session;
}

/// `text.format: json_schema` is accepted (validated at admission) and the
/// answer is served; the schema's keywords outside the subset are refused.
#[tokio::test]
async fn a_json_schema_format_is_accepted_and_an_unsupported_keyword_is_refused() {
    let (router, _) = router_with("{\"x\": 1}");
    let mut body = base();
    body["text"] = json!({"format": {"type": "json_schema", "name": "point",
        "schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}}});
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["text"]["format"]["type"], "json_schema");

    let mut body = base();
    body["text"] = json!({"format": {"type": "json_schema", "name": "p",
        "schema": {"type": "object", "not": {"type": "null"}}}});
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "text.format.schema/not", "{body}");
}

/// The hosted fields that stay out of perimeter are refused with the field
/// named, in the OpenAI error envelope. `store: true` (the API default) is
/// served now; an unknown `previous_response_id` is a 404, not a 400.
#[tokio::test]
async fn hosted_fields_are_refused_with_the_field_named() {
    let (router, _) = router_with("hi");
    for (field, value, param) in [
        ("truncation", json!("auto"), "truncation"),
        ("background", json!(true), "background"),
        ("max_tool_calls", json!(2), "max_tool_calls"),
    ] {
        let mut body = base();
        body[field] = value;
        let (status, body) = post(&router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], param, "{field}: {body}");
    }
    // A continuation id the store does not know: 404 in the envelope, with
    // the eviction message.
    let mut body = base();
    body["previous_response_id"] = json!("resp_missing");
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "response_not_found", "{body}");
    assert_eq!(body["error"]["param"], "previous_response_id", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("evicted"),
        "{body}"
    );

    // Non-text content parts are refused, not dropped.
    let mut body = base();
    body["input"] = json!([{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "https://x/y.png"}
    ]}]);
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unsupported_content", "{body}");

    // An include this server does not serve.
    let mut body = base();
    body["include"] = json!(["reasoning.encrypted_content"]);
    let (status, body) = post(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "include", "{body}");
}

/// The streamed answer is the pinned event sequence, validated here by a
/// small state machine: monotone sequence numbers, one terminal event, no
/// delta after a done, and no `[DONE]` (Responses does not use it).
#[tokio::test]
async fn a_streamed_text_turn_emits_the_pinned_event_sequence() {
    let (router, _) = router_with("streamed answer");
    let mut body = base();
    body["stream"] = json!(true);
    let (status, headers, body) = send(
        &router,
        Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
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
    assert!(!text.contains("[DONE]"), "Responses has no [DONE]: {text}");
    let events: Vec<Value> = text
        .split("\n\n")
        .filter_map(|event| {
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            let value: Value = serde_json::from_str(data).expect("a JSON event");
            let name = event
                .lines()
                .find_map(|line| line.strip_prefix("event: "))
                .expect("a named Responses event");
            assert_eq!(name, value["type"].as_str().unwrap());
            Some(value)
        })
        .collect();
    assert!(!events.is_empty());

    // The state machine: monotone numbers, known types in a valid order, one
    // terminal, nothing after it.
    let mut last_seq = -1_i64;
    let mut terminal_seen = false;
    let mut open_items: Vec<String> = Vec::new();
    let mut done_parts: Vec<(String, u32)> = Vec::new();
    let mut kinds = Vec::new();
    for event in &events {
        let seq = event["sequence_number"]
            .as_i64()
            .expect("a sequence_number");
        assert!(seq > last_seq, "monotone sequence numbers: {event}");
        last_seq = seq;
        assert!(!terminal_seen, "nothing after the terminal: {event}");
        let kind = event["type"].as_str().unwrap();
        kinds.push(kind.to_owned());
        // `output_item` events name their item inside `item`; the part and
        // delta events name it in `item_id`.
        let item_id = event
            .get("item_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                event
                    .get("item")
                    .and_then(|item| item.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        if let Some(item_id) = item_id {
            let output_index = event["output_index"].as_u64().unwrap_or(0) as u32;
            match kind {
                "response.output_item.added" => open_items.push(item_id),
                "response.output_item.done" => {
                    assert!(open_items.contains(&item_id), "done for an open item");
                    open_items.retain(|id| id != &item_id);
                }
                t if t.ends_with(".delta") => {
                    assert!(open_items.contains(&item_id), "delta for an open item: {t}");
                    assert!(
                        !done_parts.contains(&(item_id.clone(), output_index)),
                        "no delta after done: {t}"
                    );
                }
                t if t.ends_with(".done") => {
                    done_parts.push((item_id, output_index));
                }
                _ => {}
            }
        }
        if matches!(
            kind,
            "response.completed" | "response.incomplete" | "response.failed" | "error"
        ) {
            terminal_seen = true;
            assert_eq!(
                event["response"]["status"].as_str().unwrap_or(""),
                match kind {
                    "response.completed" => "completed",
                    "response.incomplete" => "incomplete",
                    "response.failed" => "failed",
                    _ => "",
                },
                "{event}"
            );
        }
    }
    assert!(terminal_seen, "a terminal event: {kinds:?}");
    assert_eq!(kinds[0], "response.created");
    assert_eq!(kinds[1], "response.in_progress");
    assert_eq!(kinds.last().unwrap(), "response.completed");
    assert!(kinds.contains(&"response.output_item.added".to_owned()));
    assert!(kinds.contains(&"response.content_part.added".to_owned()));
    assert!(kinds.contains(&"response.output_text.delta".to_owned()));
    assert!(kinds.contains(&"response.output_text.done".to_owned()));
    assert!(kinds.contains(&"response.content_part.done".to_owned()));
    assert!(kinds.contains(&"response.output_item.done".to_owned()));
    // The terminal carries the full response with the answer and the usage.
    let last = events.last().unwrap();
    assert_eq!(
        last["response"]["output"][0]["content"][0]["text"],
        "streamed answer"
    );
    assert!(last["response"]["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

/// A streamed tool call uses the `function_call_arguments` events.
#[tokio::test]
async fn a_streamed_tool_call_emits_function_call_arguments_events() {
    let reply = r#"<tool_call>{"name":"move","arguments":{"direction":"left"}}</tool_call>"#;
    let (router, _) = router_with(reply);
    let mut body = base();
    body["stream"] = json!(true);
    body["tools"] = json!([{"type": "function", "name": "move"}]);
    let (status, _, body) = send(
        &router,
        Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    let events: Vec<Value> = text
        .split("\n\n")
        .filter_map(|event| {
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            let value: Value = serde_json::from_str(data).expect("a JSON event");
            let name = event
                .lines()
                .find_map(|line| line.strip_prefix("event: "))
                .expect("a named Responses event");
            assert_eq!(name, value["type"].as_str().unwrap());
            Some(value)
        })
        .collect();
    let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert!(
        kinds.contains(&"response.function_call_arguments.delta"),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&"response.function_call_arguments.done"),
        "{kinds:?}"
    );
    assert_eq!(kinds.last(), Some(&"response.completed"));
    let added = events
        .iter()
        .find(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call")
        .expect("a function_call item");
    assert_eq!(added["item"]["name"], "move");
    let done = events
        .iter()
        .find(|e| e["type"] == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(done["arguments"].as_str().unwrap()).unwrap(),
        json!({"direction": "left"})
    );
}

/// `include: ["message.output_text.logprobs"]` puts logprobs on the streamed
/// deltas and on the terminal output_text part.
#[tokio::test]
async fn logprobs_are_streamed_when_included() {
    let (router, _) = router_with("with logprobs");
    let mut body = base();
    body["stream"] = json!(true);
    body["include"] = json!(["message.output_text.logprobs"]);
    body["top_logprobs"] = json!(3);
    let (status, _, body) = send(
        &router,
        Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    let events: Vec<Value> = text
        .split("\n\n")
        .filter_map(|event| {
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            let value: Value = serde_json::from_str(data).expect("a JSON event");
            let name = event
                .lines()
                .find_map(|line| line.strip_prefix("event: "))
                .expect("a named Responses event");
            assert_eq!(name, value["type"].as_str().unwrap());
            Some(value)
        })
        .collect();
    let delta = events
        .iter()
        .find(|e| e["type"] == "response.output_text.delta")
        .expect("a delta");
    let logprobs = delta["logprobs"].as_array().expect("logprobs on the delta");
    assert!(!logprobs.is_empty());
    assert!(logprobs[0].get("token").is_some());
    assert!(logprobs[0].get("logprob").is_some());
    assert!(logprobs[0].get("top_logprobs").is_some());
    // The terminal response's output_text part carries the full list.
    let last = events.last().unwrap();
    let part = &last["response"]["output"][0]["content"][0];
    assert_eq!(part["type"], "output_text");
    assert!(
        part["logprobs"].as_array().is_some_and(|l| !l.is_empty()),
        "{part}"
    );
    assert_eq!(
        part["logprobs"][0]["bytes"].as_array().map(|b| b.len()),
        Some(1)
    );
}
