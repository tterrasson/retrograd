//! Datasets as a resource: upload, listing, the card, the preview, and delete,
//! over `build_router` with no model and no socket, exactly like every other
//! `api_*` binary.

mod support;

use axum::body::Body;
use http::{Request, StatusCode};
use serde_json::json;
use support::*;

const CHAT_JSONL: &[u8] = b"{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},{\"role\":\"assistant\",\"content\":\"A1\"}]}\n\
{\"messages\":[{\"role\":\"user\",\"content\":\"Q2\"},{\"role\":\"assistant\",\"content\":\"A2\"}]}\n";

async fn upload(router: &axum::Router, uri: &str, body: &[u8]) -> (StatusCode, serde_json::Value) {
    send(
        router,
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(http::header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from(body.to_vec()))
            .expect("build the request"),
    )
    .await
}

#[tokio::test]
async fn an_upload_is_content_addressed_and_the_same_bytes_come_back_idempotent() {
    let fixture = Fixture::new("datasets-idempotent");
    let router = router_for(&fixture, FakeEngine::succeeding());

    let (status, first) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::CREATED, "{first}");
    let id = first["id"].as_str().expect("an id").to_string();
    assert!(id.starts_with("ds_"), "{first}");
    assert_eq!(first["format"], "chat-jsonl");
    assert_eq!(first["examples"], 2);
    assert_eq!(first["bytes"], CHAT_JSONL.len());
    assert_eq!(first["stats"]["measured"], false);
    assert!(first["stats"]["p99"].as_u64().unwrap() > 0);

    // The same bytes again: same id, `200` rather than `201`, nothing new.
    let (status, second) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["id"], first["id"]);

    let (status, listing) = get(&router, "/v1/datasets").await;
    assert_eq!(status, StatusCode::OK);
    let datasets = listing["datasets"].as_array().expect("a list");
    assert_eq!(datasets.len(), 1, "{listing}");

    let (status, card) = get(&router, &format!("/v1/datasets/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(card, first);
}

#[tokio::test]
async fn a_name_and_an_explicit_format_are_recorded() {
    let fixture = Fixture::new("datasets-name-format");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = upload(
        &router,
        "/v1/datasets?name=sql-train&format=jsonl",
        CHAT_JSONL,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["name"], "sql-train");
    assert_eq!(body["format"], "chat-jsonl");
}

#[tokio::test]
async fn a_text_dataset_is_accepted_with_an_explicit_format() {
    let fixture = Fixture::new("datasets-text");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = upload(
        &router,
        "/v1/datasets?format=text",
        b"a small corpus of plain text, nothing structured about it at all\n",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["format"], "text");
}

#[tokio::test]
async fn an_unknown_format_query_param_is_a_422() {
    let fixture = Fixture::new("datasets-bad-format");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = upload(&router, "/v1/datasets?format=parquet", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["errors"][0]["code"], "unsupported_format");
}

#[tokio::test]
async fn a_malformed_chat_jsonl_reports_every_line_error_at_once() {
    let fixture = Fixture::new("datasets-invalid");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let broken = b"not json\n\
{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}\n\
{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]}\n";
    let (status, body) = upload(&router, "/v1/datasets?format=jsonl", broken).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let line_errors = body["meta"]["line_errors"]
        .as_array()
        .expect("collected errors");
    assert_eq!(line_errors.len(), 2, "{body}");
    assert_eq!(body["meta"]["errors_total"], 2);
    assert_eq!(line_errors[0]["line"], 1);
    assert_eq!(line_errors[0]["code"], "invalid_json");
    assert_eq!(line_errors[1]["line"], 2);
    assert_eq!(line_errors[1]["code"], "invalid_value");

    // Nothing was stored: a failed ingestion leaves no dataset behind.
    let (_, listing) = get(&router, "/v1/datasets").await;
    assert!(
        listing["datasets"].as_array().unwrap().is_empty(),
        "{listing}"
    );
}

#[tokio::test]
async fn the_preview_reads_exactly_what_was_uploaded() {
    let fixture = Fixture::new("datasets-preview");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().unwrap();

    let (status, preview) = get(&router, &format!("/v1/datasets/{id}/preview?limit=1")).await;
    assert_eq!(status, StatusCode::OK, "{preview}");
    let examples = preview["examples"].as_array().expect("examples");
    assert_eq!(examples.len(), 1);
    assert_eq!(
        examples[0]["messages"][0]["content"], "Q1",
        "the preview is the parsed record, not a re-formatted one: {preview}"
    );
}

#[tokio::test]
async fn deleting_a_dataset_removes_its_card_and_a_repeat_is_a_404() {
    let fixture = Fixture::new("datasets-delete");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, body) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    let id = body["id"].as_str().unwrap().to_string();
    let dataset_dir = fixture.state_dir().join("datasets").join(&id);
    assert!(dataset_dir.is_dir());

    let (status, _, response_body) = send_raw(
        &router,
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/datasets/{id}"))
            .body(Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(response_body.is_empty());
    assert!(!dataset_dir.exists(), "the dataset directory was removed");

    let (status, _) = get(&router, &format!("/v1/datasets/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _, _) = send_raw(
        &router,
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/datasets/{id}"))
            .body(Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn an_unknown_dataset_id_is_a_404_everywhere() {
    let fixture = Fixture::new("datasets-404");
    let router = router_for(&fixture, FakeEngine::succeeding());
    for uri in [
        "/v1/datasets/ds_does_not_exist",
        "/v1/datasets/ds_does_not_exist/preview",
    ] {
        let (status, body) = get(&router, uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{uri}: {body}");
    }
}

#[tokio::test]
async fn a_dataset_over_the_configured_limit_is_refused_before_it_is_all_written() {
    let fixture = Fixture::new("datasets-too-large");
    // A tiny per-dataset limit, so an ordinary fixture body already exceeds it.
    let toml = format!(
        "state_dir = \"{}\"\nmax_dataset_bytes = 8\n",
        fixture.path("state")
    );
    let config: retrograd_server::ServerConfig = toml::from_str(&toml).expect("parse");
    let catalog = retrograd_server::Catalog::declare(&[], &[], &[], &[]).expect("declare");
    let state =
        retrograd_server::AppState::new(config, catalog, std::sync::Arc::new(FakeProbe(model())))
            .with_engine(FakeEngine::succeeding());
    let router = build_router(state);

    let (status, body) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
}

#[tokio::test]
async fn tokenizing_measures_real_lengths_caches_them_and_tightens_the_next_plan() {
    let fixture = Fixture::new("datasets-tokenize");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, uploaded) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    let id = uploaded["id"].as_str().unwrap().to_string();
    // Before any measurement, the card carries the character heuristic.
    assert_eq!(uploaded["stats"]["measured"], false);
    let estimated_p99 = uploaded["stats"]["p99"].as_u64().unwrap();

    let (status, measured) = post(
        &router,
        &format!("/v1/datasets/{id}/tokenize"),
        json!({"model": fixture.path("model.gguf")}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{measured}");
    assert_eq!(measured["stats"]["measured"], true, "{measured}");
    assert!(!measured["tokenizer"].as_str().unwrap().is_empty());
    // The fake tokenizer is four characters to the token against the estimate's
    // deliberate three, so measuring can only tighten the figure.
    let measured_p99 = measured["stats"]["p99"].as_u64().unwrap();
    assert!(
        measured_p99 < estimated_p99,
        "measured {measured_p99} should be under the estimated {estimated_p99}"
    );
    // These examples are tiny, so nothing truncates at any reported context.
    for context in ["512", "1024", "2048", "4096"] {
        assert_eq!(measured["truncation"][context], 0.0, "{measured}");
    }

    // Asking again answers the cache: same numbers, no second measurement.
    let (status, again) = post(
        &router,
        &format!("/v1/datasets/{id}/tokenize"),
        json!({"model": fixture.path("model.gguf")}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(again, measured);

    // And a plan over this dataset now runs on the measurement rather than the
    // estimate - the whole point of the cache.
    let (status, plan) = post(
        &router,
        "/v1/plan",
        json!({
            "recipe": {
                "objective": "instruction-tuning",
                "model": fixture.path("model.gguf"),
                "data": {"dataset": id},
                "budget": {"epochs": 1},
                "seed": 7
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert!(
        !plan["plan"]["warnings"]
            .as_array()
            .expect("warnings")
            .iter()
            .any(|warning| warning.to_string().contains("estimated")),
        "a warm cache removes the estimated-lengths warning: {plan}"
    );
}

#[tokio::test]
async fn tokenizing_an_unknown_dataset_or_an_unreachable_model_is_refused() {
    let fixture = Fixture::new("datasets-tokenize-refused");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, uploaded) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    let id = uploaded["id"].as_str().unwrap().to_string();

    let (status, body) = post(
        &router,
        "/v1/datasets/ds_0000000000000000/tokenize",
        json!({"model": fixture.path("model.gguf")}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

    let (status, body) = post(
        &router,
        &format!("/v1/datasets/{id}/tokenize"),
        json!({"model": fixture.path("nowhere.gguf")}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["errors"][0]["code"], "path_not_found", "{body}");
}

#[tokio::test]
async fn a_recipe_can_reference_a_stored_dataset_by_id() {
    let fixture = Fixture::new("datasets-recipe");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, uploaded) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::CREATED, "{uploaded}");
    let id = uploaded["id"].as_str().unwrap().to_string();

    let body = json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"dataset": id},
            "budget": {"epochs": 1},
            "seed": 7
        }
    });
    let (status, plan) = post(&router, "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert_eq!(
        plan["effective_config"]["sft"]["data_format"], "chat-jsonl",
        "{plan}"
    );
    let data_path = plan["effective_config"]["sft"]["data"]
        .as_str()
        .expect("a data path");
    assert!(
        data_path.ends_with(&format!("datasets/{id}/data.jsonl")),
        "{data_path}"
    );
}

#[tokio::test]
async fn a_recipe_naming_both_a_path_and_a_dataset_is_refused() {
    let fixture = Fixture::new("datasets-both");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let mut body = recipe(&fixture, "both");
    body["recipe"]["data"]["dataset"] = json!("ds_0000000000000000");
    let (status, response) = post(&router, "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response}");
    assert!(
        response["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("not both"),
        "{response}"
    );
}

#[tokio::test]
async fn an_unknown_dataset_id_in_a_recipe_is_a_404_naming_the_upload_route() {
    let fixture = Fixture::new("datasets-recipe-404");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let body = json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"dataset": "ds_0000000000000000"},
            "budget": {"epochs": 1},
            "seed": 7
        }
    });
    let (status, response) = post(&router, "/v1/plan", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{response}");
    assert!(
        response["errors"][0]["hint"]
            .as_str()
            .unwrap()
            .contains("POST /v1/datasets"),
        "{response}"
    );
}

#[tokio::test]
async fn local_dataset_paths_are_refused_when_the_operator_disables_them() {
    let fixture = Fixture::new("datasets-local-disabled");
    let toml = format!(
        "state_dir = \"{}\"\nallow_local_paths = false\n",
        fixture.path("state")
    );
    let config: retrograd_server::ServerConfig = toml::from_str(&toml).expect("parse");
    let catalog = retrograd_server::Catalog::declare(&[], &[], &[], &[]).expect("declare");
    let mut state =
        retrograd_server::AppState::new(config, catalog, std::sync::Arc::new(FakeProbe(model())))
            .with_engine(FakeEngine::succeeding());
    // A generous fake baseline: this test is about the path gate, not about
    // whether the fixture model fits on whatever machine runs the suite.
    state.baseline = retrograd_plan::MemoryBaseline {
        device_total: Some(24 * GIB),
        device_used: 0,
        host_total: Some(64 * GIB),
        host_used: 0,
        unified: false,
    };
    let router = build_router(state);

    let (status, body) = post(&router, "/v1/plan", recipe(&fixture, "disabled")).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    assert_eq!(body["errors"][0]["code"], "forbidden_path");
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("POST /v1/datasets"),
        "{body}"
    );

    // A dataset id still works: the refusal is about literal paths only.
    let (status, uploaded) = upload(&router, "/v1/datasets", CHAT_JSONL).await;
    assert_eq!(status, StatusCode::CREATED, "{uploaded}");
    let id = uploaded["id"].as_str().unwrap().to_string();
    let body = json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"dataset": id},
            "budget": {"epochs": 1},
            "seed": 7
        }
    });
    let (status, plan) = post(&router, "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
}
