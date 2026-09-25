//! The whole discovery surface, driven through the router in memory.
//!
//! No socket, no GGUF, no GPU: `build_router` is exercised with
//! `tower::ServiceExt::oneshot` and a fake [`ModelProbe`], which is what lets
//! these cases run in the fast lane. A test that needed a model would only run
//! before a PR, and the HTTP contract is exactly the part that must not be
//! checked that rarely.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_core::{Device, ModelInfo, Result as CoreResult, TargetSet};
use retrograd_server::state::ModelProbe;
use retrograd_server::{AppState, Catalog, ServerConfig, build_router};
use serde_json::Value;
use tower::ServiceExt;

/// Records what the handler asked for, so the tests can assert the request was
/// translated (path resolved, device parsed, targets parsed) and not merely
/// accepted.
#[derive(Default)]
struct FakeProbe {
    calls: AtomicUsize,
    report: String,
}

impl FakeProbe {
    fn with_report(report: &str) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            report: report.to_string(),
        })
    }
}

impl ModelProbe for FakeProbe {
    fn preflight(
        &self,
        model: &Path,
        device: Device,
        targets: TargetSet,
        profile_fingerprint: String,
    ) -> CoreResult<retrograd_core::PreflightReport> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(retrograd_core::PreflightReport {
            schema_version: retrograd_core::PREFLIGHT_REPORT_VERSION,
            profile_fingerprint,
            graph_fingerprint: format!("{}:{device:?}:{targets:?}", model.display()),
            warnings: (!self.report.is_empty())
                .then(|| retrograd_core::PreflightWarning {
                    code: "fixture".to_string(),
                    message: self.report.clone(),
                })
                .into_iter()
                .collect(),
            ..Default::default()
        })
    }

    fn geometry(&self, _model: &Path, _device: Device) -> CoreResult<ModelInfo> {
        Ok(ModelInfo {
            n_layer: 28,
            architecture: "qwen3".into(),
            ..Default::default()
        })
    }

    fn measure(
        &self,
        _model: &Path,
        _config: &retrograd_config::RunConfig,
    ) -> CoreResult<retrograd_core::MemoryReport> {
        // No discovery route measures anything. A probe that answered here would
        // be scaffolding for a call that never happens, and a test that started
        // measuring by accident would pass quietly.
        Err(retrograd_core::Error::invalid(
            "the discovery fake does not measure",
        ))
    }

    fn tokenize_lengths(
        &self,
        _model: &Path,
        _path: &Path,
        _format: retrograd_dataset::DataFormat,
    ) -> CoreResult<Vec<u32>> {
        // Same reasoning as `measure`: no discovery route tokenizes anything.
        Err(retrograd_core::Error::invalid(
            "the discovery fake does not tokenize",
        ))
    }
}

fn router_from(config_toml: &str, probe: Arc<dyn ModelProbe>) -> Router {
    let config: ServerConfig = toml::from_str(config_toml).expect("parse server config");
    let catalog = Catalog::declare(
        &config.rewards,
        &config.judges,
        &config.mcp_servers,
        &config.environments,
    )
    .expect("declare catalog");
    build_router(AppState::new(config, catalog, probe))
}

fn router() -> Router {
    router_from("", FakeProbe::with_report("training graph preflight"))
}

async fn send(router: Router, request: Request<Body>) -> (StatusCode, String, Value) {
    let response = router.oneshot(request).await.expect("route the request");
    let status = response.status();
    let content_type = response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("read body")
        .to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "response body is not JSON ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, content_type, body)
}

async fn get(router: Router, path: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("build request");
    let (status, _, body) = send(router, request).await;
    (status, body)
}

async fn post_json(router: Router, path: &str, body: Value) -> (StatusCode, String, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("build request");
    send(router, request).await
}

#[test]
fn health_reports_the_version_and_the_compiled_backends() {
    let (status, body) = tokio_block(get(router(), "/v1/health"));
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    // Every build registers a CPU device, so an empty list means the runtime did
    // not answer at all.
    let backends = body["backends"].as_array().expect("backends is a list");
    assert!(
        backends.iter().any(|backend| backend == "cpu"),
        "cpu must always be listed: {backends:?}"
    );
}

#[test]
fn capabilities_reports_budgets_that_subtract_the_baseline_and_the_margin() {
    let (status, body) = tokio_block(get(router(), "/v1/capabilities"));
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["features"]["max_concurrent_runs"], 1);
    assert!(body["unified_memory"].is_boolean());

    for name in ["vram", "ram"] {
        let budget = &body["budgets"][name];
        assert_eq!(budget["requested"], "all", "{name}");
        // A budget with no reported total promises nothing; one with a total
        // must never promise more than the total minus what is already taken.
        match budget["total_bytes"].as_u64() {
            None => assert_eq!(budget["effective_bytes"], 0, "{name}"),
            Some(total) => {
                let effective = budget["effective_bytes"].as_u64().expect("effective");
                let baseline = budget["baseline_bytes"].as_u64().expect("baseline");
                let margin = budget["margin_bytes"].as_u64().expect("margin");
                assert_eq!(effective + baseline + margin, total, "{name}");
                assert!(margin > 0, "{name} must keep a safety margin");
            }
        }
    }
}

#[test]
fn a_configured_budget_is_reported_as_asked_and_never_exceeds_the_device() {
    let router = router_from(
        "vram_budget = '1GiB'\nram_budget = 0.25\nmax_concurrent_runs = 2\n",
        FakeProbe::with_report("report"),
    );
    let (status, body) = tokio_block(get(router, "/v1/capabilities"));
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["features"]["max_concurrent_runs"], 2);
    assert_eq!(body["budgets"]["vram"]["requested"], "1073741824");
    assert_eq!(body["budgets"]["ram"]["requested"], "0.25");
    if let Some(total) = body["budgets"]["ram"]["total_bytes"].as_u64() {
        let effective = body["budgets"]["ram"]["effective_bytes"]
            .as_u64()
            .expect("effective");
        assert!(
            effective < total / 2,
            "a quarter-of-RAM budget resolved to {effective} of {total}"
        );
    }
}

/// `GET /v1/defaults` replaced `GET /v1/presets`: the rules themselves, with
/// their thresholds, instead of three words a client had to trust.
#[test]
fn defaults_expose_the_rules_and_their_thresholds_rather_than_profiles() {
    let (status, body) = tokio_block(get(router(), "/v1/defaults"));
    assert_eq!(status, StatusCode::OK);

    let derived = body["derived"].as_array().expect("derived is a list");
    assert!(!derived.is_empty());
    for field in derived {
        let path = field["path"].as_str().expect("path");
        assert!(path.contains('.'), "'{path}' is not a dotted field path");
        assert!(!field["rule"].as_str().expect("rule").is_empty(), "{path}");
        assert!(!field["applies_to"].as_str().expect("applies_to").is_empty());
        assert_eq!(field["client_can_set"], true, "{path}");
    }
    // The rank rule is the one the plan spells out in full, thresholds included.
    let rank = derived
        .iter()
        .find(|field| field["path"] == "lora.rank")
        .expect("lora.rank is documented");
    assert_eq!(rank["thresholds"]["examples_rank_8"], 2000.0);
    assert_eq!(rank["thresholds"]["examples_rank_16"], 20000.0);

    // The proactive half: every one of these is on without being asked for, so
    // every one has to be nameable and switchable off.
    let active = body["active_defaults"]
        .as_array()
        .expect("active_defaults is a list");
    let ids: Vec<&str> = active
        .iter()
        .map(|entry| entry["id"].as_str().expect("id"))
        .collect();
    assert_eq!(
        ids,
        [
            "gradient_checkpointing",
            "chunked_cross_entropy",
            "fast_sampling_context",
            "generation_concurrency"
        ]
    );
    for entry in active {
        assert_eq!(entry["client_can_disable"], true);
        assert!(!entry["paths"].as_array().expect("paths").is_empty());
        assert!(!entry["condition"].as_str().expect("condition").is_empty());
    }
    // The one degradation a plan may apply without an opt-in publishes the code
    // a client detects it by.
    let fast = active
        .iter()
        .find(|entry| entry["id"] == "fast_sampling_context")
        .expect("the rollout default is listed");
    assert_eq!(fast["degrades"], true);
    assert_eq!(fast["warning"], "sampling_distribution_approximated");
    assert!(
        active
            .iter()
            .filter(|entry| entry["degrades"] == true)
            .count()
            == 1,
        "only one default may degrade anything"
    );

    // And the route it replaced is gone rather than silently kept alive.
    let (status, _) = tokio_block(get(router(), "/v1/presets"));
    assert_eq!(status, StatusCode::NOT_FOUND);
}

const CATALOG: &str = r#"
[[reward]]
id = "sql-exec"
description = "Runs the generated query"
command = ["python", "rewards/sql_exec.py"]
timeout_seconds = 45
mode = "oneshot"

[[judge]]
id = "ruler-mini"
description = "The default judge"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[[mcp_server]]
id = "calc"
description = "A calculator"
command = ["python", "rewards/calculator_server.py"]
denied_tools = ["write_*"]
required = false

[[environment]]
id = "py-sandbox"
description = "Python tasks"
type = "local"
allow_unsandboxed = true

[environment.tools]
default = "no-shell"

[environment.tools.toolset.no-shell]
include = ["python"]
deny = ["bash"]
"#;

#[test]
fn the_catalog_lists_ids_and_never_the_values_behind_them() {
    let probe = FakeProbe::with_report("report");
    for (path, key) in [
        ("/v1/rewards", "rewards"),
        ("/v1/judges", "judges"),
        ("/v1/mcp-servers", "mcp_servers"),
        ("/v1/environments", "environments"),
    ] {
        let (status, body) = tokio_block(get(router_from(CATALOG, probe.clone()), path));
        assert_eq!(status, StatusCode::OK, "{path}");
        let entries = body[key].as_array().expect("a list of entries");
        assert_eq!(entries.len(), 1, "{path}");
        let rendered = body.to_string();
        for secret in [
            "sql_exec.py",
            "calculator_server.py",
            "api.openai.com",
            "OPENAI_API_KEY",
            "gpt-5-mini",
        ] {
            assert!(
                !rendered.contains(secret),
                "{path} leaked '{secret}': {rendered}"
            );
        }
    }
}

#[test]
fn declared_entries_carry_what_a_client_actually_needs() {
    let probe = FakeProbe::with_report("report");
    let (_, rewards) = tokio_block(get(router_from(CATALOG, probe.clone()), "/v1/rewards"));
    assert_eq!(rewards["rewards"][0]["id"], "sql-exec");
    assert_eq!(rewards["rewards"][0]["timeout_seconds"], 45);
    // The transport is the operator's to declare and the client's to read: it
    // is what says whether this command holds one worker for the whole run.
    assert_eq!(rewards["rewards"][0]["mode"], "oneshot");

    let (_, judges) = tokio_block(get(router_from(CATALOG, probe.clone()), "/v1/judges"));
    assert_eq!(judges["judges"][0]["id"], "ruler-mini");
    assert_eq!(judges["judges"][0]["kind"], "ruler");
    let settings = judges["judges"][0]["client_settings"]
        .as_array()
        .expect("client settings");
    assert!(settings.iter().any(|setting| setting == "max_pairs"));

    let (_, environments) =
        tokio_block(get(router_from(CATALOG, probe.clone()), "/v1/environments"));
    let environment = &environments["environments"][0];
    assert_eq!(environment["id"], "py-sandbox");
    assert_eq!(environment["kind"], "local");
    // The tool names are what a client needs to write a scenario; the image,
    // the limits and the pool are the operator's and stay here.
    let tools = environment["tools"].as_array().expect("tools");
    assert!(tools.iter().any(|tool| tool == "submit"));
    assert!(!tools.iter().any(|tool| tool == "bash"));

    let (_, servers) = tokio_block(get(router_from(CATALOG, probe), "/v1/mcp-servers"));
    let server = &servers["mcp_servers"][0];
    assert_eq!(server["id"], "calc");
    assert_eq!(server["required"], false);
    // `Catalog::declare` does not connect, so nothing is verified yet and the
    // entry must say so rather than claim an empty tool set is the real one.
    assert_eq!(server["status"], "skipped");
    assert!(server["tools"].as_array().expect("tools").is_empty());
}

#[test]
fn an_empty_catalog_lists_nothing_rather_than_failing() {
    for (path, key) in [
        ("/v1/rewards", "rewards"),
        ("/v1/judges", "judges"),
        ("/v1/mcp-servers", "mcp_servers"),
        ("/v1/environments", "environments"),
    ] {
        let (status, body) = tokio_block(get(router(), path));
        assert_eq!(status, StatusCode::OK, "{path}");
        assert!(body[key].as_array().expect("a list").is_empty(), "{path}");
    }
}

#[test]
fn preflight_resolves_the_path_parses_the_device_and_returns_the_report() {
    // A real file is needed because the path is canonicalized before use; the
    // *model* is fake, the path is not.
    let model =
        std::env::temp_dir().join(format!("retrograd-server-preflight-{}", std::process::id()));
    std::fs::write(&model, b"not really a gguf").expect("write fixture");

    let (status, content_type, body) = tokio_block(post_json(
        router(),
        "/v1/preflight",
        serde_json::json!({
            "model": model.display().to_string(),
            "device": "cpu",
            "targets": ["q", "v"]
        }),
    ));
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(body["schema_version"], 2);
    let fingerprint = body["graph_fingerprint"].as_str().expect("fingerprint");
    assert!(fingerprint.contains("Cpu"), "{fingerprint}");
    // The shorthands must reach the probe expanded, the way `--targets q,v` does.
    assert!(fingerprint.contains("attn_q.weight"), "{fingerprint}");
    assert!(fingerprint.contains("attn_v.weight"), "{fingerprint}");
    let warnings = body["warnings"].as_array().expect("warnings");
    assert_eq!(warnings[0]["code"], "fixture");
    assert_eq!(warnings[0]["message"], "training graph preflight");

    std::fs::remove_file(&model).expect("clean up fixture");
}

#[test]
fn preflight_reaches_the_probe_exactly_once_per_request() {
    let model = std::env::temp_dir().join(format!(
        "retrograd-server-once-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::write(&model, b"x").expect("write fixture");
    let probe = FakeProbe::with_report("report");
    let router = build_router(AppState::new(
        ServerConfig::default(),
        Catalog::default(),
        probe.clone(),
    ));
    let body = serde_json::json!({"model": model.display().to_string()});
    let (status, _, _) = tokio_block(post_json(router.clone(), "/v1/preflight", body.clone()));
    assert_eq!(status, StatusCode::OK);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);

    // A rejected request must not have loaded anything: validation happens before
    // the device permit is taken.
    let (status, _, _) = tokio_block(post_json(
        router,
        "/v1/preflight",
        serde_json::json!({"model": "/definitely/not/here.gguf"}),
    ));
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);

    std::fs::remove_file(&model).expect("clean up fixture");
}

#[test]
fn preflight_surfaces_cpu_fallbacks_as_warnings() {
    let model =
        std::env::temp_dir().join(format!("retrograd-server-warnings-{}", std::process::id()));
    std::fs::write(&model, b"x").expect("write fixture");
    let router = router_from(
        "",
        FakeProbe::with_report("  chunked_cross_entropy_status: cpu_fallback (no kernel)"),
    );

    let (status, _, body) = tokio_block(post_json(
        router,
        "/v1/preflight",
        serde_json::json!({"model": model.display().to_string()}),
    ));
    assert_eq!(status, StatusCode::OK);
    let warnings = body["warnings"].as_array().expect("warnings");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0]["code"], "fixture");
    assert!(
        warnings[0]["message"]
            .as_str()
            .expect("warning")
            .contains("cpu_fallback")
    );

    std::fs::remove_file(&model).expect("clean up fixture");
}

#[test]
fn preflight_rejects_bad_input_as_a_problem_document() {
    let cases: Vec<(Value, StatusCode, &str, &str)> = vec![
        (
            serde_json::json!({"model": "/definitely/not/here.gguf"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid-request",
            "/model",
        ),
        (
            serde_json::json!({"model": "x", "device": "quantum"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid-request",
            "/model",
        ),
    ];
    for (request, expected_status, expected_type, expected_pointer) in cases {
        let (status, content_type, body) =
            tokio_block(post_json(router(), "/v1/preflight", request.clone()));
        assert_eq!(status, expected_status, "{request}");
        assert_eq!(content_type, "application/problem+json", "{request}");
        assert!(
            body["type"]
                .as_str()
                .expect("type")
                .ends_with(expected_type),
            "{body}"
        );
        assert_eq!(body["status"], expected_status.as_u16(), "{request}");
        assert_eq!(body["errors"][0]["pointer"], expected_pointer, "{body}");
    }
}

#[test]
fn an_unknown_field_in_a_request_is_refused_rather_than_ignored() {
    let (status, _, _) = tokio_block(post_json(
        router(),
        "/v1/preflight",
        serde_json::json!({"model": "x", "devise": "cpu"}),
    ));
    // A silently ignored typo would run the preflight on the wrong device.
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[test]
fn an_unknown_route_answers_with_a_problem_document() {
    let request = Request::builder()
        .uri("/v1/nope")
        .body(Body::empty())
        .expect("build request");
    let (status, content_type, body) = tokio_block(send(router(), request));
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(content_type, "application/problem+json");
    assert!(body["type"].as_str().expect("type").ends_with("not-found"));
    assert_eq!(body["status"], 404);
    // An unversioned path is not a route either: `/v1` is the whole surface.
    let (status, _) = tokio_block(get(router(), "/health"));
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[test]
fn responses_are_byte_stable_across_calls() {
    // Determinism is an invariant of the resolver  and starts here: every
    // map in the schema is a BTreeMap and every list has a defined order, so two
    // identical requests must produce identical bytes.
    for path in [
        "/v1/defaults",
        "/v1/rewards",
        "/v1/judges",
        "/v1/mcp-servers",
        "/v1/environments",
    ] {
        let probe = FakeProbe::with_report("report");
        let first = tokio_block(get(router_from(CATALOG, probe.clone()), path)).1;
        let second = tokio_block(get(router_from(CATALOG, probe), path)).1;
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap(),
            "{path}"
        );
    }
}

/// One current-thread runtime per case. The handlers are async but the suite is
/// not, and a shared runtime would hide the fact that each case is independent.
fn tokio_block<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime")
        .block_on(future)
}
