//! Step 10: the hardening, as behaviour rather than as configuration.
//!
//! Each test here is one hardening item reduced to the question that matters: can
//! an unauthenticated caller reach a handler, can an oversized body make the
//! process allocate, does a failure ever come back as something other than a
//! problem document, and does a message on the way out describe the server's
//! filesystem.

mod support;

use std::sync::Arc;

use axum::body::Body;
use http::{Request, StatusCode};
use retrograd_server::dto;
use serde_json::json;
use support::*;

/// The fixture's server with `config` fields replaced.
fn hardened(
    fixture: &Fixture,
    adjust: impl FnOnce(&mut retrograd_server::ServerConfig),
) -> axum::Router {
    let mut state = state_of(fixture, FakeEngine::succeeding(), false);
    let mut config = (*state.config).clone();
    adjust(&mut config);
    state.config = Arc::new(config);
    build_router(state)
}

fn with_header(uri: &str, name: &str, value: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header(name, value)
        .body(Body::empty())
        .expect("the request")
}

#[tokio::test]
async fn a_token_gates_every_route_but_the_liveness_probe() {
    let fixture = Fixture::new("hardening-token");
    let router = hardened(&fixture, |config| {
        config.auth_token = Some("s3cret".into());
    });

    // No token at all.
    let (status, body) = get(&router, "/v1/capabilities").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(
        body["type"]
            .as_str()
            .unwrap_or_default()
            .ends_with("/unauthorized"),
        "{body}"
    );

    // The wrong token answers exactly the same way: which of the two it was is not
    // information a caller without a token is entitled to.
    let (wrong, _) = send(
        &router,
        with_header("/v1/capabilities", "authorization", "Bearer nope"),
    )
    .await;
    assert_eq!(wrong, StatusCode::UNAUTHORIZED);
    let (missing, _) = send(
        &router,
        with_header("/v1/capabilities", "authorization", "Basic s3cret"),
    )
    .await;
    assert_eq!(missing, StatusCode::UNAUTHORIZED);

    // The right one gets through, scheme matched case-insensitively.
    let (status, _) = send(
        &router,
        with_header("/v1/capabilities", "authorization", "bearer s3cret"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // A liveness probe has no credentials by definition, and learns nothing a port
    // scan does not.
    let (status, body) = get(&router, "/v1/health").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "ok");

    // A 401 without a challenge is not a 401 (RFC 9110 section 11.6.1).
    let response = tower::ServiceExt::oneshot(
        router.clone(),
        Request::builder()
            .uri("/v1/runs")
            .body(Body::empty())
            .expect("the request"),
    )
    .await
    .expect("route");
    assert_eq!(
        response
            .headers()
            .get(http::header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer")
    );
}

#[tokio::test]
async fn an_unauthenticated_server_may_only_be_reachable_from_this_machine() {
    let mut config = retrograd_server::ServerConfig::default();
    // The default: loopback, no token, no roots. Allowed, because a caller who is
    // already on the machine can read the filesystem and start processes anyway.
    assert!(config.binds_loopback());
    config.validate().expect("the default is valid");

    // The moment the socket leaves the machine, both are the difference between a
    // training API and a remote file-and-process primitive.
    config.bind = Some("0.0.0.0:8471".into());
    let error = config.validate().expect_err("no token off loopback");
    assert!(error.to_string().contains("auth_token"), "{error}");

    config.auth_token = Some("token".into());
    let error = config.validate().expect_err("no roots off loopback");
    assert!(error.to_string().contains("path_roots"), "{error}");

    config.path_roots = vec![std::path::PathBuf::from("/srv/models")];
    config.validate().expect("a token and a root are enough");

    // IPv6 loopback is loopback, and a wildcard is not.
    for (address, loopback) in [
        ("127.0.0.1:8471", true),
        ("[::1]:8471", true),
        ("127.0.0.2:9000", true),
        ("0.0.0.0:8471", false),
        ("[::]:8471", false),
        ("192.168.1.10:8471", false),
        // A host name cannot be confirmed, so it is treated as the unsafe case.
        ("training.internal:8471", false),
    ] {
        let candidate = retrograd_server::ServerConfig {
            bind: Some(address.into()),
            ..Default::default()
        };
        assert_eq!(candidate.binds_loopback(), loopback, "{address}");
    }
}

#[tokio::test]
async fn a_body_over_the_limit_is_refused_as_a_problem_document() {
    let fixture = Fixture::new("hardening-body");
    let router = hardened(&fixture, |config| config.max_body_bytes = Some(256));

    let big = json!({"recipe": {"objective": "instruction-tuning", "model": "x".repeat(4096)}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/plan")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(big.to_string()))
        .expect("the request");
    // `send_raw` asserts the content type of every failure, so reaching the
    // assertion below at all is the property: the body limit's 413 is generated by
    // middleware and would otherwise be one of the few failures with no body.
    let (status, content_type, _) = send_raw(&router, request).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(content_type, "application/problem+json");
}

#[tokio::test]
async fn every_failure_middleware_invents_is_still_a_problem_document() {
    let fixture = Fixture::new("hardening-problems");
    let router = router_for(&fixture, FakeEngine::succeeding());

    // A method axum's routing does not allow on a path that exists: 405, invented
    // by the router and not by a handler.
    let request = Request::builder()
        .method("PUT")
        .uri("/v1/runs")
        .body(Body::empty())
        .expect("the request");
    let (status, content_type, body) = send_raw(&router, request).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(content_type, "application/problem+json");
    let document: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(document["status"], 405);
    assert!(document["type"].as_str().is_some(), "{document}");

    // An unknown path, which the fallback handles.
    let (status, body) = get(&router, "/v1/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("/v1/nope")
    );

    // A malformed JSON body, which the extractor rejects.
    let request = Request::builder()
        .method("POST")
        .uri("/v1/plan")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from("{not json"))
        .expect("the request");
    let (status, content_type, _) = send_raw(&router, request).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(content_type, "application/problem+json");
}

#[tokio::test]
async fn an_error_that_leaves_the_machine_does_not_describe_its_filesystem() {
    let fixture = Fixture::new("hardening-redaction");
    let router = hardened(&fixture, |config| {
        // What a non-loopback bind turns on by default; set directly so the test
        // does not have to open a socket to prove it.
        config.redact_error_paths = Some(true);
    });

    let missing = fixture.path("no-such-model.gguf");
    let (status, body) = post(
        &router,
        "/v1/plan",
        json!({"recipe": {
            "objective": "instruction-tuning",
            "model": missing,
            "data": {"path": fixture.path("data.jsonl")},
            "budget": {"epochs": 1}
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let rendered = body.to_string();
    // The file name survives, so the message still says which file; the directory
    // does not, so it says nothing about the server.
    assert!(rendered.contains("no-such-model.gguf"), "{rendered}");
    assert!(
        !rendered.contains(fixture.dir.to_string_lossy().as_ref()),
        "the server's own path leaked: {rendered}"
    );
    assert!(rendered.contains("<path>/"), "{rendered}");

    // And with redaction off - the loopback default - the full path is there,
    // because that is what makes an error useful to whoever can already read it.
    let plain = router_for(&fixture, FakeEngine::succeeding());
    let (_, body) = post(
        &plain,
        "/v1/plan",
        json!({"recipe": {
            "objective": "instruction-tuning",
            "model": missing,
            "data": {"path": fixture.path("data.jsonl")},
            "budget": {"epochs": 1}
        }}),
    )
    .await;
    assert!(
        body.to_string()
            .contains(fixture.dir.to_string_lossy().as_ref()),
        "{body}"
    );
}

#[tokio::test]
async fn a_client_path_outside_every_root_is_refused() {
    let fixture = Fixture::new("hardening-roots");
    // One root: the fixture's own directory. Everything the recipe names is under
    // it, and nothing else is reachable.
    let allowed = fixture.dir.clone();
    let router = hardened(&fixture, move |config| {
        config.path_roots = vec![allowed];
    });

    let (status, body) = post(&router, "/v1/plan", recipe(&fixture, "inside")).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // A real file just outside the root: the fixture directory sits directly under
    // the temp directory, so `..` from inside it lands on a path that exists and is
    // not allowed - which is the case worth testing. A `..` that resolves to
    // nothing would only prove that canonicalization fails on missing files.
    let outside = std::env::temp_dir().join("retrograd-outside-the-root.gguf");
    std::fs::write(&outside, b"not a real GGUF").expect("a file outside the root");

    for model in [
        outside.to_string_lossy().into_owned(),
        fixture.path("../retrograd-outside-the-root.gguf"),
    ] {
        let (status, answer) = post(
            &router,
            "/v1/plan",
            json!({"recipe": {
                "objective": "instruction-tuning",
                "model": model,
                "data": {"path": fixture.path("data.jsonl")},
                "budget": {"epochs": 1}
            }}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{model}: {answer}");
        // The message names neither the roots nor the resolved path: both would
        // describe the filesystem to a caller being told they may not read it.
        assert!(
            !answer.to_string().contains("retrograd-outside-the-root"),
            "the refusal echoed the path back: {answer}"
        );
    }
    let _ = std::fs::remove_file(&outside);
}

#[tokio::test]
async fn complete_configs_apply_roots_to_every_input_and_output() {
    let fixture = Fixture::new("hardening-complete-config");
    let allowed = fixture.dir.clone();
    let router = hardened(&fixture, move |config| config.path_roots = vec![allowed]);
    let outside = std::env::temp_dir().join("retrograd-config-outside.jsonl");
    std::fs::write(&outside, b"outside\n").unwrap();

    let config = |data: String, output: String| {
        json!({
            "config": {
                "run": {"algorithm": "sft"},
                "model": {"path": fixture.path("model.gguf")},
                "output": {"path": output},
                "lora": {"rank": 8, "alpha": 16.0},
                "training": {"ctx": 128, "micro_batch": 32, "gradient_accumulation": 4, "epochs": 1},
                "sft": {"data": data, "data_format": "jsonl"}
            }
        })
    };
    let (status, body) = post(
        &router,
        "/v1/plan",
        config(
            outside.to_string_lossy().into_owned(),
            fixture.path("adapter.gguf"),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");

    let (status, body) = post(
        &router,
        "/v1/plan",
        config(
            fixture.path("data.jsonl"),
            std::env::temp_dir()
                .join("retrograd-outside-adapter.gguf")
                .to_string_lossy()
                .into_owned(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");

    let _ = std::fs::remove_file(outside);
}

#[test]
fn the_observe_directory_is_held_to_the_path_roots() {
    let fixture = Fixture::new("hardening-observe");
    let mut state = state_of(&fixture, FakeEngine::succeeding(), false);
    let mut config = (*state.config).clone();
    config.path_roots = vec![fixture.dir.clone()];
    state.config = Arc::new(config);
    let document = |observe: &std::path::Path| {
        format!(
            "[run]\nalgorithm='ppo'\n[model]\npath='{}'\n[output]\npath='{}'\n[lora]\n\
             [ppo]\nprompts='{}'\nreward_command=['true']\nupdates=1\nrollout_batch_size=1\n\
             ppo_epochs=1\nclip_range=0.2\nkl_coefficient=0.0\n\
             [ppo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n\
             [observe]\ndirectory='{}'\n",
            fixture.path("model.gguf"),
            fixture.path("adapter.gguf"),
            fixture.path("data.jsonl"),
            observe.display()
        )
    };
    let build = |source: String| {
        retrograd_config::build(
            retrograd_config::parse_toml(&source, "run.toml").expect("parse"),
            std::path::Path::new("/"),
        )
        .expect("build")
    };
    state
        .validate_run_paths(&build(document(&fixture.dir.join("observe"))), false)
        .expect("inside the root");
    let error = state
        .validate_run_paths(
            &build(document(
                &std::env::temp_dir().join("retrograd-outside-observe"),
            )),
            false,
        )
        .expect_err("outside every root");
    let body = format!("{error:?}");
    assert!(body.contains("/config/observe/directory"), "{body}");
}

#[tokio::test]
async fn path_roots_use_component_boundaries() {
    let fixture = Fixture::new("hardening-component-boundary");
    let allowed = fixture.dir.join("models");
    let sibling = fixture.dir.join("models-evil");
    std::fs::create_dir_all(&allowed).unwrap();
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(allowed.join("data.jsonl"), b"row\n").unwrap();
    std::fs::write(sibling.join("model.gguf"), b"model\n").unwrap();
    let router = hardened(&fixture, move |config| config.path_roots = vec![allowed]);
    let (status, body) = post(
        &router,
        "/v1/plan",
        json!({"recipe": {
            "objective": "instruction-tuning",
            "model": sibling.join("model.gguf"),
            "data": {"path": fixture.dir.join("models/data.jsonl")},
            "budget": {"epochs": 1}
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
}

#[tokio::test]
async fn a_zero_bound_is_rejected_rather_than_taken_literally() {
    for adjust in [
        |config: &mut retrograd_server::ServerConfig| config.max_concurrent_requests = Some(0),
        |config: &mut retrograd_server::ServerConfig| config.max_body_bytes = Some(0),
        |config: &mut retrograd_server::ServerConfig| config.command_timeout_seconds = Some(0),
        |config: &mut retrograd_server::ServerConfig| config.request_timeout_seconds = Some(0),
        |config: &mut retrograd_server::ServerConfig| config.max_concurrent_runs = Some(0),
    ] {
        let mut config = retrograd_server::ServerConfig::default();
        adjust(&mut config);
        assert!(
            config.validate().is_err(),
            "a bound of zero would make the server refuse everything silently"
        );
    }
}

#[tokio::test]
async fn the_openapi_document_describes_the_routes_the_router_serves() {
    let fixture = Fixture::new("hardening-openapi");
    let router = router_for(&fixture, FakeEngine::succeeding());

    let (status, body) = get(&router, "/v1/openapi.json").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["openapi"], "3.1.0");
    assert_eq!(body["info"]["title"], "retrograd-server");

    // Every route the API contract lists has an operation. A route added to the
    // router and not to the table fails here, which is what keeps the document
    // from drifting.
    for path in [
        "/v1/health",
        "/v1/capabilities",
        "/v1/defaults",
        "/v1/rewards",
        "/v1/judges",
        "/v1/mcp-servers",
        "/v1/preflight",
        "/v1/plan",
        "/v1/runs",
        "/v1/runs/{id}",
        "/v1/runs/{id}/pause",
        "/v1/runs/{id}/resume",
        "/v1/runs/{id}/cancel",
        "/v1/runs/{id}/checkpoints",
        "/v1/runs/{id}/evaluate",
        "/v1/runs/{id}/generate",
        "/v1/runs/{id}/artifacts",
        "/v1/runs/{id}/artifacts/{name}",
        "/v1/runs/{id}/metrics",
        "/v1/runs/{id}/events",
        "/v1/events",
        "/v1/datasets",
        "/v1/datasets/{id}",
        "/v1/datasets/{id}/preview",
    ] {
        assert!(body["paths"][path].is_object(), "{path} is undocumented");
    }
    // The schemas are derived from the DTOs, not written twice.
    assert!(
        body["components"]["schemas"]["RunView"].is_object(),
        "the component section is empty; is the `openapi` feature on?"
    );
    assert!(body["components"]["schemas"]["GenerateRequest"].is_object());
    assert_eq!(
        body["components"]["securitySchemes"]["bearer"]["scheme"],
        "bearer"
    );
}

/// The OpenAPI document and the API must spell a run state the same way.
///
/// They are produced by two different derives from the same enum, and the two
/// do not read the same attributes: `serde` honours a per-variant
/// `#[serde(rename)]`, `utoipa` builds the schema's `enum` from the variant
/// identifier and the container's `rename_all` alone. A declaration that
/// satisfies one and not the other publishes `"Queued"` for a field the server
/// sends as `"queued"` - a broken contract that compiles, that every existing
/// test passes, and that a client only finds at runtime. Found while moving the
/// vocabulary onto `wire_enum!`.
#[tokio::test]
async fn run_status_is_published_as_it_is_serialized() {
    let fixture = Fixture::new("hardening-run-status-schema");
    let router = router_for(&fixture, FakeEngine::succeeding());

    let (status, body) = get(&router, "/v1/openapi.json").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let published = body["components"]["schemas"]["RunStatus"]["enum"]
        .as_array()
        .expect("RunStatus is a documented schema");
    let serialized: Vec<String> = dto::RunStatus::ALL
        .iter()
        .map(|state| state.as_str().to_string())
        .collect();
    assert_eq!(
        published
            .iter()
            .map(|value| value.as_str().expect("a string").to_string())
            .collect::<Vec<_>>(),
        serialized,
        "the OpenAPI enum does not match what the API serializes"
    );
    // And `as_str` is what serde writes, not a third spelling beside it.
    for state in dto::RunStatus::ALL {
        assert_eq!(
            serde_json::to_string(state).expect("a fieldless enum serializes"),
            format!("\"{}\"", state.as_str()),
        );
    }
}
