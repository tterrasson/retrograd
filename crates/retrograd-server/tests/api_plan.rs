//! `POST /v1/plan` and `POST /v1/runs?dry_run=true`, driven through the router
//! in memory.
//!
//! The resolver's own coverage lives in `retrograd-plan` (snapshots and
//! properties, no HTTP). What is checked here is the seam: the three request
//! forms, the contract guard on a body that finally has something to guard, the
//! catalogue substitution and its redaction, and the mapping from a resolver
//! failure to a problem document.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_config::RunConfig;
use retrograd_core::{Device, MemoryReport, ModelInfo, Result as CoreResult, TargetSet};
use retrograd_plan::MemoryBaseline;
use retrograd_plan::cost::{self, Calibration, Workload, WorkloadKind};
use retrograd_server::state::ModelProbe;
use retrograd_server::{AppState, Catalog, ServerConfig, build_router};
use serde_json::{Value, json};
use tower::ServiceExt;

const GIB: u64 = 1024 * 1024 * 1024;

fn resource_post_bytes(memory: &Value, phase: &str, name: &str) -> u64 {
    memory["resources"][phase]["posts"]
        .as_array()
        .and_then(|posts| posts.iter().find(|post| post["name"] == name))
        .and_then(|post| post["bytes"].as_u64())
        .unwrap_or_else(|| panic!("missing resource post {phase}.{name}: {memory}"))
}

/// Answers with a synthetic geometry, so no GGUF is ever opened.
///
/// Its *measurement* is the cost model times a fixed factor, which makes this
/// fake a machine that consistently needs a quarter more compute than analysis
/// predicts - exactly the systematic error the calibration pass exists for, and
/// deterministic enough to assert on.
struct FakeProbe {
    model: ModelInfo,
    compute_factor: f64,
    measured: AtomicUsize,
    packing: bool,
    packing_calls: std::sync::Mutex<Vec<u32>>,
}

impl FakeProbe {
    fn new(compute_factor: f64) -> Arc<Self> {
        Arc::new(Self {
            model: model(),
            compute_factor,
            measured: AtomicUsize::new(0),
            packing: false,
            packing_calls: Default::default(),
        })
    }

    fn with_packing() -> Arc<Self> {
        Arc::new(Self {
            model: model(),
            compute_factor: 1.0,
            measured: AtomicUsize::new(0),
            packing: true,
            packing_calls: Default::default(),
        })
    }

    fn measurements(&self) -> usize {
        self.measured.load(Ordering::SeqCst)
    }
}

impl ModelProbe for FakeProbe {
    fn model_capabilities(
        &self,
        _model: &Path,
        _device: Device,
    ) -> CoreResult<retrograd_core::ModelCapabilities> {
        Ok(retrograd_core::ModelCapabilities {
            shared_prefix_packed_training: self.packing,
            ..Default::default()
        })
    }

    fn benchmark_packing(
        &self,
        _model: &Path,
        config: &RunConfig,
        _shape: retrograd_plan::packing_tuning::PackingShape,
    ) -> CoreResult<Option<retrograd_plan::packing_tuning::PackingMeasurement>> {
        if !self.packing {
            return Ok(None);
        }
        let width = config.training.n_ubatch;
        self.packing_calls
            .lock()
            .expect("no probe call panics while holding the lock")
            .push(width);
        let seconds = match width {
            256 => 1.0,
            128 => 5.0,
            64 => 6.0,
            _ => 10.0,
        };
        Ok(Some(retrograd_plan::packing_tuning::PackingMeasurement {
            seconds: vec![seconds; 3],
            device_bytes: GIB,
            failure: None,
        }))
    }

    fn preflight(
        &self,
        _model: &Path,
        _device: Device,
        _targets: TargetSet,
        profile_fingerprint: String,
    ) -> CoreResult<retrograd_core::PreflightReport> {
        Ok(retrograd_core::PreflightReport {
            schema_version: retrograd_core::PREFLIGHT_REPORT_VERSION,
            profile_fingerprint,
            graph_fingerprint: "fixture".to_string(),
            ..Default::default()
        })
    }

    fn geometry(&self, _model: &Path, _device: Device) -> CoreResult<ModelInfo> {
        Ok(self.model.clone())
    }

    fn measure(&self, _model: &Path, config: &RunConfig) -> CoreResult<MemoryReport> {
        self.measured.fetch_add(1, Ordering::SeqCst);
        let estimate = cost::estimate(
            &self.model,
            &config.training,
            &config.lora.config,
            &Workload {
                kind: WorkloadKind::Sft,
                examples: 0,
                // A fake probe loads one model, so it measures one model.
                co_resident_bytes: 0,
            },
            Calibration::default(),
        );
        let compute = (estimate.optimizer_compute_bytes as f64 * self.compute_factor) as u64;
        Ok(MemoryReport {
            model_weight_bytes: estimate.model_weight_bytes,
            optimizer_kv_bytes: estimate.optimizer_kv_bytes,
            optimizer_compute_bytes: compute,
            lora_parameter_bytes: estimate.lora_parameter_bytes,
            lora_gradient_bytes: estimate.lora_gradient_bytes,
            adamw_momenta_bytes: estimate.adamw_momenta_bytes,
            device_bytes: estimate.device_bytes() - estimate.optimizer_compute_bytes + compute,
            backend_scratch_peak_bytes: estimate.dequant_scratch_bytes,
            device_memory_samples: 1,
            ..Default::default()
        })
    }

    fn tokenize_lengths(
        &self,
        _model: &Path,
        _path: &Path,
        _format: retrograd_dataset::DataFormat,
    ) -> CoreResult<Vec<u32>> {
        // This binary covers the plan seam, not tokenization - `api_datasets`
        // owns that, with a fake tokenizer behind it. Failing loudly beats
        // inventing lengths a test here might come to assert on.
        Err(retrograd_core::Error::invalid(
            "this fake probe does not tokenize",
        ))
    }
}

/// Files the handler canonicalizes before reading. Cleaned up on drop, so a
/// failing test does not leave a fixture behind for the next one to trip on.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("retrograd-plan-api-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        std::fs::write(dir.join("model.gguf"), b"not a real GGUF").expect("write the model stub");
        let mut data = String::new();
        for index in 0..200 {
            data.push_str(&format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"question {index} \
                 with a little padding\"}},{{\"role\":\"assistant\",\"content\":\
                 \"an answer of a similar length\"}}]}}\n"
            ));
        }
        std::fs::write(dir.join("data.jsonl"), data).expect("write the dataset");
        Self { dir }
    }

    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }

    /// A server configuration whose state directory is inside the fixture, so a
    /// test that writes `calibration.json` writes it where the fixture's `Drop`
    /// will remove it.
    fn state_dir_toml(&self) -> String {
        format!("state_dir = \"{}\"\n", self.path("state"))
    }

    /// A second, untouched state directory: the same server with no correction
    /// table, which is what a stored factor has to be compared against.
    fn fresh_state_dir_toml(&self) -> String {
        format!("state_dir = \"{}\"\n", self.path("state-fresh"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn model() -> ModelInfo {
    ModelInfo {
        n_layer: 24,
        n_embd: 1024,
        n_head: 16,
        n_head_kv: 8,
        n_embd_head_k: 64,
        n_embd_head_v: 64,
        n_embd_k_gqa: 512,
        n_embd_v_gqa: 512,
        n_vocab: 151_936,
        n_ctx_train: 32_768,
        n_params: 600_000_000,
        model_size_bytes: 400 * 1024 * 1024,
        file_size_bytes: 400 * 1024 * 1024,
        tied_embeddings: true,
        architecture: "qwen3".into(),
        dominant_weight_type: "Q4_K".into(),
        ..Default::default()
    }
}

/// A router over a machine with a fixed amount of memory, so a plan does not
/// depend on whatever card the test happens to run on.
fn router_with(config_toml: &str, device_bytes: u64) -> Router {
    router_from(config_toml, device_bytes, FakeProbe::new(1.0))
}

fn router_from(config_toml: &str, device_bytes: u64, probe: Arc<FakeProbe>) -> Router {
    let config: ServerConfig = toml::from_str(config_toml).expect("parse the server config");
    let catalog = Catalog::declare(
        &config.rewards,
        &config.judges,
        &config.mcp_servers,
        &config.environments,
    )
    .expect("declare the catalog");
    let mut state = AppState::new(config, catalog, probe);
    state.baseline = MemoryBaseline {
        device_total: Some(device_bytes),
        device_used: 0,
        host_total: Some(64 * GIB),
        host_used: 0,
        unified: false,
    };
    build_router(state)
}

fn router() -> Router {
    router_with("", 24 * GIB)
}

async fn post(router: Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(
        router,
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("build the request"),
    )
    .await
}

/// Same as [`post`], but hands back the response body verbatim.
///
/// Parsing to `Value` would hide the very thing some tests are about: how a
/// number is *written*.
async fn post_raw(router: Router, uri: &str, body: Value) -> (StatusCode, String) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("build the request"),
        )
        .await
        .expect("route the request");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("read the body")
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn send(router: Router, request: Request<Body>) -> (StatusCode, Value) {
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
        .expect("read the body")
        .to_bytes();
    if !status.is_success() {
        assert_eq!(
            content_type,
            "application/problem+json",
            "every failure is a problem document: {}",
            String::from_utf8_lossy(&bytes)
        );
    }
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "the body is not JSON ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, body)
}

fn recipe(fixture: &Fixture) -> Value {
    json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "seed": 7
        },
        "name": "qwen3-sft-v3"
    })
}

#[tokio::test]
async fn a_recipe_resolves_into_an_effective_config_and_a_plan() {
    let fixture = Fixture::new("recipe");
    let (status, body) = post(router(), "/v1/plan", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    assert_eq!(body["name"], "qwen3-sft-v3");
    let config = &body["effective_config"];
    assert_eq!(config["run"]["algorithm"], "sft");
    assert!(config["training"]["ctx"].as_u64().unwrap() > 0);
    assert_eq!(
        config["training"]["ctx"].as_u64().unwrap()
            % (config["training"]["micro_batch"].as_u64().unwrap()
                * config["training"]["gradient_accumulation"]
                    .as_u64()
                    .unwrap()),
        0,
        "the answer must satisfy the runtime divisibilities"
    );

    // Provenance is a flat, dotted map with a reason on every derivation.
    let provenance = body["provenance"].as_object().expect("a provenance map");
    let context = &provenance["training.ctx"];
    assert_eq!(context["source"], "derived");
    assert!(
        context["reason"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
    );

    // The plan carries the decomposition, the budgets and the levers.
    let plan = &body["plan"];
    assert!(plan["total_steps"].as_u64().unwrap() > 0);
    assert!(resource_post_bytes(&plan["memory"], "persistent_device", "model_weight_bytes") > 0);
    assert!(
        plan["memory"]["budgets"]["vram"]["effective_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        plan["memory"].get("measured").is_none(),
        "nothing was measured"
    );
    assert_eq!(plan["truncation_fraction"], 0.0);
}

/// The learning rate on the wire is the learning rate the engine gets.
///
/// `training.lr` is an `f32`. Serialized straight from the typed response it
/// reads `0.0001`; routed through a `serde_json::Value` first - which widens
/// every `f32` to `f64` - it reads `0.00009999999747378752`. Both parse back to
/// the same bits, but the second is unreadable. The response must render the
/// original `f32` without routing it through a widened value.
#[tokio::test]
async fn a_float_is_written_as_the_value_that_was_chosen() {
    let fixture = Fixture::new("float-rendering");
    let (status, body) = post_raw(router(), "/v1/plan", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.contains(r#""lr":0.0001"#),
        "the learning rate was widened somewhere on the way out: {body}"
    );
    assert!(
        !body.contains("0.00009999999"),
        "an f32 was rendered through an f64: {body}"
    );
}

#[tokio::test]
async fn the_same_request_answers_the_same_bytes() {
    let fixture = Fixture::new("determinism");
    let (_, first) = post(router(), "/v1/plan", recipe(&fixture)).await;
    let (_, second) = post(router(), "/v1/plan", recipe(&fixture)).await;
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap(),
        "invariant 3: same inputs, same byte of JSON"
    );
}

#[tokio::test]
async fn a_param_comes_back_untouched_and_marked_as_the_callers() {
    let fixture = Fixture::new("params");
    let mut body = recipe(&fixture);
    body["params"] = json!({"training": {"ctx": 512}, "lora": {"rank": 32}});
    let (status, body) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["effective_config"]["training"]["ctx"], 512);
    assert_eq!(body["effective_config"]["lora"]["rank"], 32);
    assert_eq!(body["provenance"]["training.ctx"]["source"], "override");
    assert!(
        body["provenance"]["training.ctx"].get("reason").is_none(),
        "a value the caller supplied has nothing to explain"
    );
}

#[tokio::test]
async fn an_explicit_null_in_the_params_is_refused() {
    let fixture = Fixture::new("null-param");
    let mut body = recipe(&fixture);
    body["params"] = json!({"training": {"ctx": null}});
    let (status, body) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        body["detail"].as_str().unwrap().contains("omit it"),
        "{body}"
    );
}

#[tokio::test]
async fn a_full_config_is_estimated_but_not_re_derived() {
    let fixture = Fixture::new("config-form");
    let body = json!({
        "config": {
            "run": {"algorithm": "sft"},
            "model": {"path": fixture.path("model.gguf")},
            "lora": {"output": fixture.path("adapter.gguf"), "rank": 8, "alpha": 16.0},
            "training": {"ctx": 512, "micro_batch": 64, "gradient_accumulation": 8, "epochs": 1},
            "sft": {"data": fixture.path("data.jsonl"), "data_format": "jsonl"}
        }
    });
    let (status, body) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["effective_config"]["training"]["ctx"], 512);
    assert_eq!(body["effective_config"]["training"]["micro_batch"], 64);
    assert!(
        body["plan"]["levers"].as_array().unwrap().is_empty(),
        "a supplied config is never re-derived"
    );
    assert!(
        body["plan"]["memory"]["resources"]["device_peak_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[tokio::test]
async fn a_toml_body_is_the_same_document_as_the_json_one() {
    let fixture = Fixture::new("toml-form");
    let toml = format!(
        r#"
[run]
algorithm = "sft"
[model]
path = "{model}"
[lora]
output = "{adapter}"
rank = 8
alpha = 16.0
[training]
ctx = 512
micro_batch = 64
gradient_accumulation = 8
epochs = 1
[sft]
data = "{data}"
data_format = "jsonl"
"#,
        model = fixture.path("model.gguf"),
        adapter = fixture.path("adapter.gguf"),
        data = fixture.path("data.jsonl"),
    );
    let (status, body) = send(
        router(),
        Request::builder()
            .method("POST")
            .uri("/v1/plan")
            .header(http::header::CONTENT_TYPE, "application/toml")
            .body(Body::from(toml))
            .expect("build the request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["effective_config"]["training"]["ctx"], 512);
    assert_eq!(body["effective_config"]["run"]["algorithm"], "sft");
}

#[tokio::test]
async fn a_configuration_over_budget_is_refused_with_its_decomposition() {
    let fixture = Fixture::new("over-budget");
    let body = json!({
        "config": {
            "run": {"algorithm": "sft"},
            "model": {"path": fixture.path("model.gguf")},
            "lora": {"output": fixture.path("adapter.gguf"), "rank": 8, "alpha": 16.0},
            "training": {"ctx": 32768, "micro_batch": 512, "gradient_accumulation": 64, "epochs": 1},
            "sft": {"data": fixture.path("data.jsonl"), "data_format": "jsonl"}
        }
    });
    // 1 GiB cannot hold a 400 MiB model plus a 32k KV cache.
    let (status, problem) = post(router_with("", GIB), "/v1/plan", body.clone()).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/insufficient-memory"
    );
    // The dominant posts live in `meta`, not `errors[]`: a pointer belongs to the request, and these describe the response.
    let posts: Vec<&str> = problem["meta"]["dominant_posts"]
        .as_array()
        .expect("meta.dominant_posts")
        .iter()
        .map(|entry| entry["post"].as_str().unwrap())
        .collect();
    assert!(
        !posts.is_empty(),
        "the answer must name which post overflows: {posts:?}"
    );
    assert!(problem["meta"]["overflow_bytes"].as_u64().is_some());

    // `force` is the caller taking responsibility, and it says so in a warning
    // that carries a code - every warning does, since one of them is how a client
    // detects an approximated sampling distribution.
    let (status, accepted) = post(router_with("", GIB), "/v1/plan?force=true", body).await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    let warnings = accepted["plan"]["warnings"].as_array().unwrap();
    assert!(
        warnings
            .iter()
            .any(|warning| warning["code"] == "over_budget_accepted_by_force"
                && warning["message"]
                    .as_str()
                    .is_some_and(|text| text.contains("force=true"))),
        "{warnings:?}"
    );
}

#[tokio::test]
async fn a_degradation_is_refused_until_the_recipe_opts_in() {
    let fixture = Fixture::new("opt-in");
    let mut body = recipe(&fixture);

    // A corpus with a long tail, so the resolved context is far above what fits
    // and the only remaining lever would throw data away. The resolver must
    // refuse the plan when no non-destructive lever remains. F16 is the default
    // cache, so `kv_f16` is not a lever; truncation is the only degradation
    // a supervised recipe can be pushed into. It needs a tail: the resolver will
    // not cut a context below the median of the data it was derived from.
    let mut tailed = String::new();
    for index in 0..400 {
        let answer = "an answer ".repeat(if index % 100 == 0 { 4_000 } else { 1 });
        tailed.push_str(&format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"question {index}\"}},\
             {{\"role\":\"assistant\",\"content\":\"{answer}\"}}]}}\n"
        ));
    }
    let path = fixture.path("tailed.jsonl");
    std::fs::write(&path, tailed).expect("write the tailed dataset");
    body["recipe"]["data"] = json!({"path": path, "format": "jsonl"});

    let (status, problem) = post(router_with("", GIB), "/v1/plan", body.clone()).await;
    assert!(
        status == StatusCode::UNPROCESSABLE_ENTITY || status == StatusCode::CONFLICT,
        "expected a refusal, got {status}: {problem}"
    );
    assert!(
        problem["detail"]
            .as_str()
            .is_some_and(|text| !text.is_empty()),
        "a refusal has to say what it would have done: {problem}"
    );

    // Granting the opt-in the answer named resolves the same request.
    if problem["type"] == "https://retrograd.dev/problems/invalid-request" {
        let detail = problem["detail"].as_str().unwrap().to_string();
        for opt_in in ["truncate_context", "exceed_train_context"] {
            if detail.contains(opt_in) {
                body["recipe"]["allow"] = json!([opt_in]);
                let (status, granted) = post(router_with("", GIB), "/v1/plan", body.clone()).await;
                assert_eq!(
                    status,
                    StatusCode::OK,
                    "the opt-in the error asked for must resolve it: {granted}"
                );
                return;
            }
        }
    }
}

#[tokio::test]
async fn a_server_declared_value_is_refused_wherever_it_appears() {
    let fixture = Fixture::new("guard");
    let mut body = recipe(&fixture);
    body["params"] = json!({"grpo": {"reward_command": ["sh", "-c", "curl evil"]}});
    let (status, problem) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/server-declared"
    );
    assert_eq!(
        problem["errors"][0]["pointer"],
        "/params/grpo/reward_command"
    );
}

#[tokio::test]
async fn an_unknown_catalog_id_lists_what_is_declared() {
    let fixture = Fixture::new("catalog");
    let body = json!({
        "recipe": {
            "objective": "reasoning-rl",
            "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "reward": {"id": "nope"}
        }
    });
    let (status, problem) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/unknown-catalog-id"
    );
    assert_eq!(problem["errors"][0]["pointer"], "/recipe/reward/id");
}

#[tokio::test]
async fn a_declared_reward_is_substituted_and_redacted_again() {
    let fixture = Fixture::new("reward");
    let declaration = r#"
[[reward]]
id = "sql-exec"
description = "runs the generated query"
command = ["python3", "rewards/sql_exec.py"]
"#;
    let body = json!({
        "recipe": {
            "objective": "reasoning-rl",
            "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "reward": {"id": "sql-exec"},
            "budget": {"updates": 20}
        }
    });
    let (status, body) = post(router_with(declaration, 24 * GIB), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["effective_config"]["run"]["algorithm"], "grpo");
    assert_eq!(body["effective_config"]["grpo"]["updates"], 20);

    let rendered = serde_json::to_string(&body).unwrap();
    assert!(
        !rendered.contains("sql_exec.py"),
        "the declared command must never leave the server: {rendered}"
    );
    assert_eq!(
        body["effective_config"]["grpo"]["reward_command"],
        json!(["<reward:sql-exec>"])
    );
}

#[tokio::test]
async fn a_recipe_and_a_config_together_are_refused() {
    let fixture = Fixture::new("both-forms");
    let mut body = recipe(&fixture);
    body["config"] = json!({
        "run": {"algorithm": "sft"},
        "model": {"path": fixture.path("model.gguf")},
        "lora": {"output": fixture.path("adapter.gguf")},
        "sft": {"data": fixture.path("data.jsonl")}
    });
    let (status, problem) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert!(problem["detail"].as_str().unwrap().contains("not both"));

    let (status, problem) = post(router(), "/v1/plan", json!({})).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert!(problem["detail"].as_str().unwrap().contains("required"));
}

#[tokio::test]
async fn a_path_outside_the_allowed_roots_is_forbidden() {
    let fixture = Fixture::new("roots");
    let declaration = format!("path_roots = [\"{}\"]\n", fixture.path("nowhere"));
    std::fs::create_dir_all(fixture.dir.join("nowhere")).unwrap();
    let (status, problem) = post(
        router_with(&declaration, 24 * GIB),
        "/v1/plan",
        recipe(&fixture),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/forbidden-path"
    );
}

/// Regression for the case that started: a dataset path that
/// does not exist must be unmistakable, not a `null` an unwary `jq` produces.
#[tokio::test]
async fn a_missing_dataset_path_is_a_regression_the_response_names_by_code_and_hint() {
    let fixture = Fixture::new("missing-dataset");
    let mut request = recipe(&fixture);
    request["recipe"]["data"]["path"] = json!(fixture.path("no-such-file.jsonl"));
    let (status, problem) = post(router(), "/v1/plan", request).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    let errors = problem["errors"].as_array().expect("field errors");
    let error = errors
        .iter()
        .find(|error| error["code"] == "path_not_found")
        .unwrap_or_else(|| panic!("expected a path_not_found error: {problem}"));
    assert_eq!(error["pointer"], "/recipe/data/path");
    let hint = error["hint"].as_str().expect("a hint");
    assert!(hint.contains("POST /v1/datasets"), "{hint}");
}

#[tokio::test]
async fn a_dry_run_answers_the_same_plan_without_creating_anything() {
    let fixture = Fixture::new("dry-run");
    let (status, dry) = post(router(), "/v1/runs?dry_run=true", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{dry}");
    let (_, planned) = post(router(), "/v1/plan", recipe(&fixture)).await;
    assert_eq!(
        serde_json::to_string(&dry).unwrap(),
        serde_json::to_string(&planned).unwrap(),
        "a dry run is a plan; the two must not drift"
    );
    assert!(dry.get("id").is_none(), "nothing was created to identify");
}

/// A plan is an estimate until someone asks for a measurement.
#[tokio::test]
async fn a_plan_measures_nothing_unless_it_is_asked_to() {
    let fixture = Fixture::new("no-calibration");
    let probe = FakeProbe::new(1.25);
    let router = router_from(&fixture.state_dir_toml(), 24 * GIB, probe.clone());
    let (status, body) = post(router, "/v1/plan", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        probe.measurements(),
        0,
        "a plan must stay free of device side effects (§5.6)"
    );
    assert!(body["plan"]["memory"].get("measured").is_none());
}

/// Measures, corrects, and re-resolves a plan end to end.
#[tokio::test]
async fn a_measured_plan_records_the_gap_and_corrects_the_next_one() {
    let fixture = Fixture::new("calibration");
    let probe = FakeProbe::new(1.25);
    let router = router_from(&fixture.state_dir_toml(), 24 * GIB, probe.clone());
    let (status, body) = post(router, "/v1/plan?calibrate=true", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The measurement is reported in the same shape as the estimate, so the two
    // subtract field by field.
    let memory = &body["plan"]["memory"];
    let measured = &memory["measured"];
    assert!(measured["optimizer_compute_bytes"].as_u64().unwrap() > 0);
    // A post the runtime does not separate stays zero rather than being invented.
    assert_eq!(measured["activation_bytes"], 0);

    // The first pass measured a quarter more compute than analysis predicted, so
    // phase 3 ran again with the corrected factor - and the second pass measured
    // exactly what the corrected estimate now says. Two measurements, one per
    // pass, and no third: the estimate caught up.
    assert_eq!(probe.measurements(), 2);
    assert_eq!(
        measured["optimizer_compute_bytes"].as_u64().unwrap(),
        resource_post_bytes(memory, "optimizer_transient", "optimizer_compute_bytes"),
        "the corrected estimate must agree with what the machine did: {memory}"
    );

    // What was derived is now measured, and says by what.
    let context = &body["provenance"]["training.ctx"];
    assert_eq!(context["source"], "measured", "{}", body["provenance"]);
    assert!(context["reason"].as_str().unwrap().contains("calibration"));

    // The correction outlives the request: a second, uncalibrated plan on the
    // same machine already carries it.
    let store = std::fs::read_to_string(fixture.path("state/calibration.json"))
        .expect("calibration.json was written");
    assert!(store.contains("compute_scale"), "{store}");
    assert!(store.contains("1.25"), "{store}");

    let router = router_from(&fixture.state_dir_toml(), 24 * GIB, FakeProbe::new(1.25));
    let (status, corrected) = post(router, "/v1/plan", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{corrected}");
    let (_, plain) = post(
        router_with(&fixture.fresh_state_dir_toml(), 24 * GIB),
        "/v1/plan",
        recipe(&fixture),
    )
    .await;
    assert!(
        resource_post_bytes(
            &corrected["plan"]["memory"],
            "optimizer_transient",
            "optimizer_compute_bytes"
        ) > resource_post_bytes(
            &plain["plan"]["memory"],
            "optimizer_transient",
            "optimizer_compute_bytes"
        ),
        "the stored factor must apply without measuring again"
    );
}

#[tokio::test]
async fn a_malformed_body_is_still_a_problem_document() {
    let (status, problem) = send(
        router(),
        Request::builder()
            .method("POST")
            .uri("/v1/plan")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{not json"))
            .expect("build the request"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert_eq!(
        problem["type"],
        "https://retrograd.dev/problems/invalid-request"
    );

    let (status, problem) = send(
        router(),
        Request::builder()
            .method("POST")
            .uri("/v1/plan")
            .header(http::header::CONTENT_TYPE, "application/toml")
            .body(Body::from("this is not toml ="))
            .expect("build the request"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
}

// ---------------------------------------------------------------------------
// `params` and the partial TOML body
// ---------------------------------------------------------------------------

/// The pre-V2 alias is not part of the new `/v1` contract.
#[tokio::test]
async fn overrides_is_an_unknown_field() {
    let fixture = Fixture::new("old-spelling");
    let mut body = recipe(&fixture);
    body["overrides"] = json!({"lora": {"rank": 32}});
    let (status, problem) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    assert!(
        problem["detail"]
            .as_str()
            .is_some_and(|text| text.contains("overrides")),
        "{problem}"
    );
}

/// A TOML body without `run.algorithm` and `model.path` is read as a `params`
/// tree rather than as a configuration missing half of itself - which is what
/// lets a client post the part of `examples/smoke_tiny_grpo.toml` it has an
/// opinion about.
///
/// Params alone are not a request, though: they say *how* to train and never
/// *what*. The refusal names that rather than listing the three forms, because a
/// client that posted a partial TOML on purpose needs to know what to add.
#[tokio::test]
async fn a_partial_toml_body_is_read_as_params_and_asks_for_something_to_apply_it_to() {
    let (status, problem) = send(
        router(),
        Request::builder()
            .method("POST")
            .uri("/v1/plan")
            .header(http::header::CONTENT_TYPE, "application/toml")
            .body(Body::from("[lora]\nrank = 8\n[training]\nctx = 1024\n"))
            .expect("build the request"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{problem}");
    let detail = problem["detail"].as_str().expect("detail");
    assert!(detail.contains("how to train"), "{detail}");
    // Not "invalid TOML" and not "missing model.path": it parsed, and it was
    // understood as parameters.
    assert!(!detail.contains("invalid TOML"), "{detail}");
}

/// The same tuning, sent as JSON `params` beside a recipe: locked, untouched, and
/// exactly what a client-side "profile" is.
#[tokio::test]
async fn a_client_side_profile_is_just_params_beside_a_recipe() {
    let fixture = Fixture::new("client-profile");
    let mut body = recipe(&fixture);
    body["params"] = json!({
        "lora": {"rank": 8, "alpha": 16.0},
        "training": {"ctx": 512, "lr": 1e-5, "lr_scheduler": "constant"}
    });
    let (status, body) = post(router(), "/v1/plan", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let config = &body["effective_config"];
    assert_eq!(config["lora"]["rank"], 8);
    assert_eq!(config["training"]["ctx"], 512);
    assert_eq!(config["training"]["lr_scheduler"], "constant");
    for path in ["lora.rank", "training.ctx", "training.lr_scheduler"] {
        assert_eq!(
            body["provenance"][path]["source"], "override",
            "{path} was re-derived"
        );
    }
}

/// A plan that needed no memory lever still reports which proactive defaults it
/// enabled, keeping defaults and recovery levers distinct on the wire.
#[tokio::test]
async fn a_plan_reports_the_defaults_it_applied_separately_from_its_levers() {
    let fixture = Fixture::new("defaults-applied");
    let (status, body) = post(router(), "/v1/plan", recipe(&fixture)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let plan = &body["plan"];
    let applied = plan["defaults_applied"]
        .as_array()
        .expect("defaults_applied is always present, even empty");
    let levers = plan["levers"].as_array().expect("levers");
    for entry in applied {
        let id = entry["id"].as_str().expect("id");
        assert!(!entry["note"].as_str().expect("note").is_empty(), "{id}");
        assert!(!entry["cost"].as_str().expect("cost").is_empty(), "{id}");
        assert_ne!(
            entry["from"], entry["to"],
            "{id} reported a move it did not make"
        );
        assert!(
            !levers.iter().any(|lever| lever["id"] == id),
            "'{id}' is both a default and a lever"
        );
    }
    // Every warning is an object with a code, not a bare string.
    for warning in plan["warnings"].as_array().expect("warnings") {
        assert!(
            warning["code"]
                .as_str()
                .is_some_and(|code| !code.is_empty())
        );
        assert!(
            warning["message"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        );
    }
}

#[tokio::test]
async fn measured_packing_changes_geometry_and_preserves_explicit_locks() {
    let fixture = Fixture::new("packing-measurements");
    let declaration = format!(
        r#"{}
[[reward]]
id = "packing-test"
command = ["true"]
"#,
        fixture.state_dir_toml()
    );
    let request = json!({
        "recipe": { "objective": "reasoning-rl", "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "reward": {"id": "packing-test"}, "budget": {"updates": 1} },
        "params": { "training": {"ctx": 512, "gradient_checkpointing": false},
            "grpo": {"group_size": 8, "sampling": {"max_new_tokens": 16}} }
    });
    let probe = FakeProbe::with_packing();
    let router = router_from(&declaration, 24 * GIB, probe.clone());
    let (status, analytical) = post(router.clone(), "/v1/plan", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{analytical}");
    assert!(probe.packing_calls.lock().unwrap().is_empty());
    assert_eq!(
        analytical["effective_config"]["training"]["micro_batch"],
        512
    );
    let (status, measured) = post(router.clone(), "/v1/plan?calibrate=true", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{measured}");
    assert_eq!(
        measured["effective_config"]["training"]["micro_batch"], 256,
        "{measured}"
    );
    assert!(
        measured["plan"]["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w["code"] == "packing_geometry_measured")
    );
    let calls = probe.packing_calls.lock().unwrap().clone();
    assert!(
        (2..=retrograd_plan::packing_tuning::MAX_PACKING_PROBES).contains(&calls.len()),
        "{calls:?}"
    );
    assert!(calls.contains(&512) && calls.contains(&256), "{calls:?}");
    probe.packing_calls.lock().unwrap().clear();
    let mut locked = request;
    locked["params"]["training"]["micro_batch"] = json!(512);
    locked["params"]["training"]["shared_prefix_fanout"] = json!(8);
    let (status, body) = post(router, "/v1/plan?calibrate=true", locked).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["effective_config"]["training"]["micro_batch"], 512);
    assert_eq!(
        body["effective_config"]["training"]["shared_prefix_fanout"],
        8
    );
    assert!(probe.packing_calls.lock().unwrap().is_empty());
}
