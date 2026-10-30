//! The Embeddings API contract, without a model: the four `input` shapes, the
//! two encodings, the refusals, and the determinism the contract promises -
//! identical inputs give identical vectors, and a batch gives the same vectors
//! as separate calls.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_core::Device;
use retrograd_openai::testing::{FAKE_EMBEDDING_SIZE, FakeLoader};
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
    }
}

fn router_with(loader: FakeLoader) -> Router {
    let session = Arc::new(
        Session::new(
            Arc::new(loader),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("session"),
    );
    Router::new().nest(
        "/v1",
        retrograd_openai::router(Endpoint {
            source: Arc::new(SingleModel {
                name: "embedder".into(),
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
            Request::post("/v1/embeddings")
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

#[tokio::test]
async fn the_four_input_shapes_answer_one_vector_each() {
    let router = router_with(FakeLoader::new("unused"));
    for (input, count, prompt_tokens) in [
        (json!("hi"), 1, 2),
        (json!(["a", "bc"]), 2, 3),
        (json!([1, 2, 3]), 1, 3),
        (json!([[1, 2], [3, 4, 5]]), 2, 5),
    ] {
        let (status, body) = post(&router, json!({"model": "embedder", "input": input})).await;
        assert_eq!(status, StatusCode::OK, "{input}: {body}");
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), count, "{input}: {body}");
        for (index, object) in data.iter().enumerate() {
            assert_eq!(object["object"], "embedding");
            assert_eq!(object["index"], index as u64);
            assert_eq!(
                object["embedding"].as_array().unwrap().len(),
                FAKE_EMBEDDING_SIZE,
                "{input}: {body}"
            );
        }
        assert_eq!(
            body["usage"]["prompt_tokens"], prompt_tokens,
            "{input}: {body}"
        );
        assert_eq!(
            body["usage"]["total_tokens"],
            body["usage"]["prompt_tokens"]
        );
    }
}

/// The determinism the contract states: the same input gives the same vector,
/// and a batch gives the same vectors as separate calls.
#[tokio::test]
async fn identical_inputs_give_identical_vectors_and_a_batch_matches_separate_calls() {
    let router = router_with(FakeLoader::new("unused"));
    let (status, single) = post(&router, json!({"model": "embedder", "input": "ab"})).await;
    assert_eq!(status, StatusCode::OK, "{single}");
    let (status, batch) = post(
        &router,
        json!({"model": "embedder", "input": ["ab", "ab", "c"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{batch}");
    assert_eq!(batch["data"][0]["embedding"], batch["data"][1]["embedding"]);
    assert_eq!(
        batch["data"][0]["embedding"],
        single["data"][0]["embedding"]
    );
    assert_ne!(batch["data"][0]["embedding"], batch["data"][2]["embedding"]);
}

/// `encoding_format: "base64"` is the standard base64 of the little-endian
/// float32 bytes of the vector.
#[tokio::test]
async fn base64_encoding_is_float32_little_endian() {
    let router = router_with(FakeLoader::new("unused"));
    let (status, body) = post(
        &router,
        json!({"model": "embedder", "input": "hi", "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let encoded = body["data"][0]["embedding"].as_str().expect("base64");
    let bytes = base64_decode(encoded);
    assert_eq!(bytes.len(), FAKE_EMBEDDING_SIZE * 4, "{encoded}");
    // And it decodes to the same floats as the `float` encoding.
    let (status, floats) = post(&router, json!({"model": "embedder", "input": "hi"})).await;
    assert_eq!(status, StatusCode::OK, "{floats}");
    let expected = floats["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let decoded: Vec<f32> = bytes
        .chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(decoded, expected);
}

/// `dimensions` and a malformed input are refused with the field named; a
/// model that declares no pooling gets the endpoint's refusal.
#[tokio::test]
async fn refusals_name_the_field_and_a_model_without_pooling_is_refused() {
    let router = router_with(FakeLoader::new("unused"));
    for (body, param) in [
        (
            json!({"model": "embedder", "input": "x", "dimensions": 8}),
            "dimensions",
        ),
        (
            json!({"model": "embedder", "input": "x", "encoding_format": "int8"}),
            "encoding_format",
        ),
        (json!({"model": "embedder", "input": ["a", 3]}), "input"),
        (json!({"model": "embedder", "input": []}), "input"),
    ] {
        let (status, body) = post(&router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{param}: {body}");
        assert_eq!(body["error"]["param"], param, "{param}: {body}");
    }

    let no_pooling = router_with(FakeLoader::new("unused").without_embeddings());
    let (status, body) = post(&no_pooling, json!({"model": "embedder", "input": "x"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "embeddings");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no pooling"),
        "{body}"
    );
}

/// Standard base64 with padding, for the test above.
fn base64_decode(text: &str) -> Vec<u8> {
    let mut table = [255u8; 256];
    for (index, byte) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .iter()
        .enumerate()
    {
        table[*byte as usize] = index as u8;
    }
    let mut out = Vec::new();
    let mut accumulator = 0u32;
    let mut bits = 0;
    for &byte in text.as_bytes() {
        if byte == b'=' {
            break;
        }
        let value = table[byte as usize];
        assert!(value != 255, "not base64: {text}");
        accumulator = (accumulator << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
        }
    }
    out
}

#[tokio::test]
async fn a_flat_token_array_is_one_input_and_malformed_ids_are_refused() {
    let router = router_with(FakeLoader::new("unused"));
    let (status, body) = post(&router, json!({"model":"embedder", "input":vec![1; 300]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["usage"]["prompt_tokens"], 300);
    for input in [json!([-1]), json!([[]])] {
        let (status, body) = post(&router, json!({"model":"embedder", "input":input})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["param"]
                .as_str()
                .unwrap()
                .starts_with("input")
        );
    }
}
