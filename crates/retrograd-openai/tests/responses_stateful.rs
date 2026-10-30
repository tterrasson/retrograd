//! The stateful Responses API contract, without a model: `store: true` (the
//! default) writes to the endpoint's store, `previous_response_id` replays a
//! stored chain, and the read routes answer from it - all through the router
//! against the fake session.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_core::Device;
use retrograd_openai::testing::FakeLoader;
use retrograd_openai::{
    Endpoint, ModelCard, ModelSource, OpenAiEnvelope, OpenAiError, ResponseStore, Session,
    SessionOptions, SingleModel, Target, Unshared, WeightsSpec,
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

fn router_with(reply: &str) -> (Router, Arc<dyn ResponseStore>) {
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
    let store: Arc<dyn ResponseStore> = Arc::new(retrograd_openai::MemoryStore::default());
    let router = Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(source),
            session,
            timeout: Duration::from_secs(30),
            enabled: true,
            store: store.clone(),
        }),
    );
    (router, store)
}

async fn send(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = router.clone().oneshot(request).await.expect("a response");
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

async fn post(router: &Router, path: &str, body: Value) -> (StatusCode, Value) {
    send(
        router,
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request"),
    )
    .await
}

async fn get(router: &Router, path: &str) -> (StatusCode, Value) {
    send(
        router,
        Request::get(path).body(Body::empty()).expect("request"),
    )
    .await
}

async fn delete(router: &Router, path: &str) -> (StatusCode, Value) {
    send(
        router,
        Request::delete(path).body(Body::empty()).expect("request"),
    )
    .await
}

/// `store` defaults to `true`: the answer is stored, echoes `store: true`,
/// and `GET /v1/responses/{id}` returns it. `store: false` writes nothing.
#[tokio::test]
async fn a_stored_response_is_readable_and_an_unstored_one_is_not() {
    let (router, store) = router_with("hello there");
    let mut body = json!({"model": "tuned", "input": "hi"});
    let (status, response) = post(&router, "/v1/responses", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["store"], true);
    assert_eq!(response["previous_response_id"], Value::Null);
    let id = response["id"].as_str().unwrap().to_owned();
    assert_eq!(store.get(&id).expect("stored").model, "tuned");

    let (status, fetched) = get(&router, &format!("/v1/responses/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{fetched}");
    assert_eq!(fetched["id"], response["id"]);
    assert_eq!(fetched["output"][0]["content"][0]["text"], "hello there");

    // `store: false` writes nothing.
    body["store"] = json!(false);
    let (status, response) = post(&router, "/v1/responses", body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["store"], false);
    let id = response["id"].as_str().unwrap().to_owned();
    assert!(store.get(&id).is_none());
    let (status, _) = get(&router, &format!("/v1/responses/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// A stored turn keeps its KV in the prefix cache: the turn that continues it
/// reports the shared prefix as `cached_tokens`, part of its `input_tokens`.
#[tokio::test]
async fn a_continuation_reports_the_prefix_it_reused_as_cached_tokens() {
    let (router, _) = router_with("first answer");
    let (status, first) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(first["usage"]["input_tokens_details"]["cached_tokens"], 0);
    let (status, second) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "again", "previous_response_id": first["id"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let cached = second["usage"]["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    let input = second["usage"]["input_tokens"].as_u64().unwrap();
    assert!(
        cached > 0 && cached < input,
        "cached={cached} input={input}"
    );
}

/// A continuation replays the stored chain (input items plus the previous
/// output) in front of the new input, and echoes `previous_response_id`.
#[tokio::test]
async fn a_continuation_replays_the_chain_and_keeps_call_ids() {
    let (router, _) = router_with("first answer");
    let (status, first) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let first_id = first["id"].as_str().unwrap().to_owned();

    let (status, second) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "again", "previous_response_id": first_id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["previous_response_id"], json!(first_id));
    let second_id = second["id"].as_str().unwrap().to_owned();

    // The stored input items of the second response are the whole chain:
    // the first user message, the first answer as an assistant message, the
    // new user message.
    let (status, items) = get(&router, &format!("/v1/responses/{second_id}/input_items")).await;
    assert_eq!(status, StatusCode::OK, "{items}");
    assert_eq!(items["object"], "list");
    let data = items["data"].as_array().unwrap();
    assert_eq!(data.len(), 3, "{items}");
    assert_eq!(data[0]["role"], "user");
    assert_eq!(data[0]["content"], "hi");
    assert_eq!(data[1]["role"], "assistant");
    assert_eq!(data[1]["content"][0]["text"], "first answer");
    assert_eq!(data[2]["role"], "user");
    assert_eq!(data[2]["content"], "again");
    // Every item has an id, so pagination cursors can name it.
    assert!(data.iter().all(|item| item["id"].is_string()), "{items}");

    // The chain can branch: both continuations of the first response work.
    for input in ["one", "two"] {
        let (status, _) = post(
            &router,
            "/v1/responses",
            json!({"model": "tuned", "input": input, "previous_response_id": first_id}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "branching from {first_id}");
    }
}

/// Pagination of `input_items`: `limit`, `order`, `after`/`before` cursors.
#[tokio::test]
async fn input_items_paginate_with_limit_order_and_cursors() {
    let (router, _) = router_with("ok");
    let input: Vec<Value> = (0..5)
        .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
        .collect();
    let (status, response) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": input}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let id = response["id"].as_str().unwrap().to_owned();

    let (status, page) = get(&router, &format!("/v1/responses/{id}/input_items?limit=2")).await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page["data"].as_array().unwrap().len(), 2);
    assert_eq!(page["has_more"], true);
    let first_id = page["first_id"].as_str().unwrap().to_owned();
    let last_id = page["last_id"].as_str().unwrap().to_owned();
    assert_ne!(first_id, last_id);

    // `after` starts the next page where the cursor names.
    let (status, page2) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?limit=2&after={last_id}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page2}");
    assert_eq!(page2["data"].as_array().unwrap().len(), 2);
    assert_eq!(page2["data"][0]["content"], "m2");
    assert_eq!(page2["has_more"], true);

    // `desc` reverses, and `before` ends at the cursor.
    let (status, desc) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?limit=2&order=desc"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{desc}");
    assert_eq!(desc["data"][0]["content"], "m4");

    let (status, before) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?limit=10&before={first_id}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{before}");
    assert!(before["data"].as_array().unwrap().is_empty());
    assert_eq!(before["has_more"], false);

    let cursor = desc["last_id"].as_str().unwrap();
    let (status, tail) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?order=desc&after={cursor}&limit=10"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{tail}");
    assert_eq!(tail["data"][0]["content"], "m2");
    assert_eq!(tail["data"][2]["content"], "m0");
    assert_eq!(tail["has_more"], false);

    // A page that covers everything says there is no more.
    let (status, all) = get(&router, &format!("/v1/responses/{id}/input_items?limit=10")).await;
    assert_eq!(status, StatusCode::OK, "{all}");
    assert_eq!(all["data"].as_array().unwrap().len(), 5);
    assert_eq!(all["has_more"], false);

    let (status, body) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?limit=invalid"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"].is_object(), "{body}");

    // Bounds are refused with the field named.
    let (status, body) = get(&router, &format!("/v1/responses/{id}/input_items?limit=0")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "limit");
    let (status, body) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?order=sideways"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "order");
    let (status, body) = get(
        &router,
        &format!("/v1/responses/{id}/input_items?after=item_nope"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "after");
}

/// `DELETE` removes the response; afterwards it reads 404 and a continuation
/// from it gets the eviction 404.
#[tokio::test]
async fn a_delete_then_reads_and_continuations_are_404() {
    let (router, _) = router_with("bye");
    let (status, first) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let id = first["id"].as_str().unwrap().to_owned();

    let (status, deleted) = delete(&router, &format!("/v1/responses/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    assert_eq!(deleted["deleted"], true);
    assert_eq!(deleted["id"], json!(id));

    let (status, body) = get(&router, &format!("/v1/responses/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "response_not_found");

    let (status, body) = post(
        &router,
        "/v1/responses",
        json!({"model": "tuned", "input": "again", "previous_response_id": id}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["param"], "previous_response_id");

    // Deleting twice is a 404, and a malformed id never touches the disk.
    let (status, _) = delete(&router, &format!("/v1/responses/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = get(&router, "/v1/responses/..%2Fescape").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// A streamed answer with `store` at its default is stored once its terminal
/// event went out.
#[tokio::test]
async fn a_streamed_answer_is_stored_at_its_terminal_event() {
    let (router, store) = router_with("streamed");
    let request = Request::post("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "tuned", "input": "hi", "stream": true}).to_string(),
        ))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let terminal: Value = text
        .split("\n\n")
        .filter_map(|event| {
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str(data).ok()
        })
        .find(|event: &Value| event["type"] == "response.completed")
        .expect("a terminal event");
    let id = terminal["response"]["id"].as_str().unwrap().to_owned();
    assert!(
        store.get(&id).is_some(),
        "stored once the terminal went out"
    );
    assert_eq!(terminal["response"]["store"], true);
}

/// A continuation onto a different base model is refused clearly, instead of
/// replaying tokens computed under another model.
struct TwoBases;

#[async_trait]
impl ModelSource for TwoBases {
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError> {
        Ok(vec![ModelCard::new("a", 1), ModelCard::new("b", 1)])
    }

    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError> {
        let spec = match model {
            "a" => WeightsSpec {
                model: "a.gguf".into(),
                adapter: None,
                n_ctx: 4096,
                device: Device::Cpu,
                n_seq_max: None,
                chat_template_variables: None,
            },
            "b" => WeightsSpec {
                model: "b.gguf".into(),
                adapter: None,
                n_ctx: 4096,
                device: Device::Cpu,
                n_seq_max: None,
                chat_template_variables: None,
            },
            other => return Err(OpenAiError::model_not_found(other, "no such model")),
        };
        Ok(Target::Cold {
            spec,
            fingerprint: model.to_owned(),
        })
    }
}

#[tokio::test]
async fn a_continuation_with_another_model_is_refused_clearly() {
    let session = Arc::new(
        Session::new(
            Arc::new(FakeLoader::new("hi")),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("session"),
    );
    let router = Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(TwoBases),
            session,
            timeout: Duration::from_secs(30),
            enabled: true,
            store: Arc::new(retrograd_openai::MemoryStore::default()),
        }),
    );
    let (status, first) = post(
        &router,
        "/v1/responses",
        json!({"model": "a", "input": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let id = first["id"].as_str().unwrap().to_owned();

    // Another model id (here on another base file) is refused with the field
    // named, instead of replaying tokens computed under another model.
    let (status, body) = post(
        &router,
        "/v1/responses",
        json!({"model": "b", "input": "again", "previous_response_id": id}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "previous_response_id");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("created for model `a`"),
        "{body}"
    );

    // The same model id continues fine.
    let (status, _) = post(
        &router,
        "/v1/responses",
        json!({"model": "a", "input": "again", "previous_response_id": id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn stored_responses_are_isolated_between_owners_and_ownerless_requests() {
    use retrograd_openai::StoreOwner;
    let (router, _) = router_with("ok");
    for creator in [None, Some("alice")] {
        let mut request = Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"tuned", "input":"hi"}).to_string(),
            ))
            .unwrap();
        if let Some(owner) = creator {
            request.extensions_mut().insert(StoreOwner(owner.into()));
        }
        let (_, response) = send(&router, request).await;
        let id = response["id"].as_str().unwrap();
        for reader in [None, Some("alice"), Some("bob")] {
            for method in ["GET", "DELETE", "POST"] {
                // Only exercise forbidden deletes; retain the entry for the other checks.
                if method == "DELETE" && reader == creator {
                    continue;
                }
                let path = if method == "POST" {
                    "/v1/responses".to_owned()
                } else {
                    format!("/v1/responses/{id}")
                };
                let body = if method == "POST" {
                    Body::from(
                        json!({"model":"tuned", "input":"next",
                    "previous_response_id":id, "store":false})
                        .to_string(),
                    )
                } else {
                    Body::empty()
                };
                let mut request = Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(body)
                    .unwrap();
                if let Some(owner) = reader {
                    request.extensions_mut().insert(StoreOwner(owner.into()));
                }
                let (status, body) = send(&router, request).await;
                let expected = if reader == creator {
                    StatusCode::OK
                } else {
                    StatusCode::NOT_FOUND
                };
                assert_eq!(
                    status, expected,
                    "{creator:?} -> {reader:?} {method}: {body}"
                );
            }
        }
    }
}
