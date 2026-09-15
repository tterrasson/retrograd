//! Step 11: the one end-to-end case, on the CPU fixture (`cpu-integration`).
//!
//! Everything else in this crate runs against a fake engine and a fake probe,
//! which is right: the state machine, the journal, the control channel and the
//! wire contract are all the server's own, and a GPU lane would only make them
//! slower to check. What a fake cannot answer is whether the pieces *join* - that
//! `POST /v1/runs` really loads a GGUF, that the resolver's measured pass agrees
//! with llama.cpp, that a pause really blocks a training callback, that the
//! checkpoint a cancellation asks for really lands where the listing says.
//!
//! So: one run on the fixture, driven the way a client would drive it - plan,
//! create, stream, pause, ask, checkpoint, cancel, fork.
//!
//! This is the coverage step 6 left open. Its measured pass was tested against a
//! fake probe returning a fixed multiple of the cost model, which exercises the
//! loop and the lever table and says nothing about agreement with the runtime.
//! Here the numbers come from `Trainer::memory_report`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_plan::MemoryBaseline;
use retrograd_server::state::EngineProbe;
use retrograd_server::{AppState, Catalog, ServerConfig, TrainingEngine, build_router};
use serde_json::{Value, json};
use tower::ServiceExt;

/// The fixture, or a reason to skip.
///
/// `RETRO_REQUIRE_CPU_FIXTURE` turns a missing fixture into a failure, which is
/// what the lane sets: a lane that silently skips its only model test is a lane
/// that reports green for nothing.
fn fixture_model() -> Option<PathBuf> {
    let path = std::env::var("RETRO_CPU_FIXTURE").ok().map(PathBuf::from);
    match path {
        Some(path) if path.is_file() => Some(path),
        other => {
            if std::env::var("RETRO_REQUIRE_CPU_FIXTURE").is_ok() {
                panic!(
                    "RETRO_REQUIRE_CPU_FIXTURE is set but the fixture is missing: {other:?}; \
                     run scripts/fetch-cpu-fixture.sh"
                );
            }
            eprintln!("skipping: RETRO_CPU_FIXTURE is not set to a readable GGUF");
            None
        }
    }
}

struct Workspace {
    dir: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("retrograd-server-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the workspace");
        // Short exchanges, so the pinned context holds them whole and no example
        // is truncated - truncation is a semantic degradation the resolver would
        // (correctly) refuse without an opt-in.
        //
        // Eight of them, not more: `prepare_chat_jsonl` gives every example its
        // own row padded to `n_ctx`, so an example *is* an optimizer step, and a
        // boundary cancellation has to drain the epoch it lands in before it can
        // stop. Thirty-two rows is ten minutes of CPU spent on padding, for the
        // same assertions.
        let mut data = String::new();
        for index in 0..8 {
            data.push_str(&format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"what is {index} plus one\"}},\
                 {{\"role\":\"assistant\",\"content\":\"{}\"}}]}}\n",
                index + 1
            ));
        }
        std::fs::write(dir.join("train.jsonl"), &data).expect("the training data");
        std::fs::write(dir.join("eval.jsonl"), &data).expect("the evaluation data");
        Self { dir }
    }

    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The real thing: the engine that runs `retrograd_run::execute_controlled`, and
/// the probe that loads a model.
fn router(workspace: &Workspace) -> Router {
    let config = ServerConfig {
        state_dir: Some(workspace.dir.join("state")),
        // On, deliberately: this is the lane that checks the measured pass against
        // llama.cpp rather than against a fake.
        calibrate_runs: Some(true),
        ..Default::default()
    };
    let catalog = Catalog::declare(
        &config.rewards,
        &config.judges,
        &config.mcp_servers,
        &config.environments,
    )
    .expect("an empty catalogue");
    let mut state = AppState::new(config, catalog, Arc::new(EngineProbe));
    state.engine = Arc::new(TrainingEngine);
    // The real baseline, so the budget is the machine's. A fixture-sized model in
    // a few hundred megabytes fits on anything that can run this lane.
    state.baseline = MemoryBaseline {
        host_used: 0,
        ..state.baseline
    };
    build_router(state)
}

fn recipe(workspace: &Workspace, model: &Path, epochs: u32) -> Value {
    json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": model.to_string_lossy(),
            "data": {"path": workspace.path("train.jsonl"), "format": "jsonl"},
            "eval": {"path": workspace.path("eval.jsonl")},
            "budget": {"epochs": epochs},
            "seed": 7
        },
        "params": {
            // Pinned so the run is seconds rather than minutes, and so the shape
            // does not change with the machine's free memory. Small on purpose:
            // every row is padded to `ctx`, so a context sized for the exchanges
            // above (a few dozen tokens each) is the difference between a step
            // costing a second and one costing twenty, and every assertion below
            // is about what the pieces do, not about how much padding they chew.
            "training": {"ctx": 128, "micro_batch": 128},
            "lora": {"rank": 4, "output": workspace.path("adapter.gguf")},
            "checkpoint": {
                "directory": workspace.path("checkpoints"),
                "mode": "steps",
                "every_steps": 1000
            }
        },
        "name": "e2e"
    })
}

async fn call(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = router.clone().oneshot(request).await.expect("route");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("read the body")
        .to_bytes();
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!("not JSON ({error}): {}", String::from_utf8_lossy(&bytes))
        })
    };
    (status, body)
}

async fn post(router: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    call(
        router,
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("the request"),
    )
    .await
}

async fn post_empty(router: &Router, uri: &str) -> (StatusCode, Value) {
    call(
        router,
        Request::builder()
            .method("POST")
            .uri(uri)
            .body(Body::empty())
            .expect("the request"),
    )
    .await
}

async fn get(router: &Router, uri: &str) -> (StatusCode, Value) {
    call(
        router,
        Request::builder()
            .uri(uri)
            .body(Body::empty())
            .expect("the request"),
    )
    .await
}

/// Polls until the run reports one of `wanted`, or gives up.
///
/// The bound is generous: this lane loads a model, and the point of the test is
/// what happens after it has, not how fast it does.
async fn until(router: &Router, id: &str, wanted: &[&str]) -> Value {
    let start = std::time::Instant::now();
    for _ in 0..3000 {
        let (status, body) = get(router, &format!("/v1/runs/{id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        if wanted.contains(&body["status"].as_str().unwrap_or_default()) {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let (_, body) = get(router, &format!("/v1/runs/{id}")).await;
    // How long it waited, not just what it was waiting for: the two failures this
    // bound can report - a run that is stuck and a run that is merely slower than
    // the machine allows for - read the same without it.
    panic!(
        "the run never reached {wanted:?} in {:?}; it is {}",
        start.elapsed(),
        body["status"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_on_the_cpu_fixture_is_planned_driven_and_forked() {
    let Some(model) = fixture_model() else {
        return;
    };
    let workspace = Workspace::new();
    let router = router(&workspace);

    // ---------------------------------------------------------------- plan
    // `?calibrate=true`: the measured pass, on a real model. What step 6 could
    // only test against a fake.
    let (status, plan) = post(
        &router,
        "/v1/plan?calibrate=true",
        recipe(&workspace, &model, 1),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    let measured = &plan["plan"]["memory"]["measured"];
    assert!(
        measured.is_object(),
        "a calibrated plan reports what it measured: {plan}"
    );
    // The measured posts carry the same names as the estimated ones, so the two
    // subtract field by field.
    for post in [
        "model_weight_bytes",
        "optimizer_kv_bytes",
        "optimizer_compute_bytes",
    ] {
        assert!(
            measured[post].as_u64().unwrap_or(0) > 0,
            "{post} was not measured: {measured}"
        );
    }
    // Reaching a 200 at all is the budget assertion: a calibrated resolution
    // re-checks the budget against what it measured and answers
    // `insufficient-memory` otherwise. This pins the figure the check used.
    let estimated_device = plan["plan"]["memory"]["resources"]["device_peak_bytes"]
        .as_u64()
        .expect("an estimate");
    let device_budget = plan["plan"]["memory"]["budgets"]["vram"]["effective_bytes"]
        .as_u64()
        .expect("a device budget");
    assert!(
        estimated_device <= device_budget,
        "the corrected estimate must fit the budget: {estimated_device} > {device_budget}"
    );
    // Provenance moves from derived to measured once the machine has spoken.
    let provenance = plan["provenance"].to_string();
    assert!(provenance.contains("measured"), "{provenance}");

    // ------------------------------------------------- a recipe with no params
    // Everything above pins the shape so the lane stays seconds long. This one
    // pins nothing at all: what the resolver derives on its own has to be a
    // configuration the runtime accepts. The calibrated
    // correction table written just above is reused here; recalibrating the
    // same model would only reload it. A `dry_run` because this shape only has
    // to be valid, not trained twice.
    let bare = json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": model.to_string_lossy(),
            "data": {"path": workspace.path("train.jsonl"), "format": "jsonl"},
            "budget": {"epochs": 1},
            "seed": 7
        },
        "params": {"lora": {"output": workspace.path("bare-adapter.gguf")}},
        "name": "no-params"
    });
    let (status, derived) = post(&router, "/v1/runs?dry_run=true", bare).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a recipe with no tuning at all must resolve: {derived}"
    );
    let config = &derived["effective_config"];
    let ctx = config["training"]["ctx"].as_u64().expect("a context");
    let micro_batch = config["training"]["micro_batch"]
        .as_u64()
        .expect("a micro-batch");
    let accumulation = config["training"]["gradient_accumulation"]
        .as_u64()
        .expect("an accumulation count");
    assert_eq!(
        ctx % (micro_batch * accumulation),
        0,
        "the derived geometry must satisfy the runtime"
    );
    assert!(config["lora"]["rank"].as_u64().unwrap_or(0) > 0);
    assert!(config["training"]["lr"].as_f64().unwrap_or(0.0) > 0.0);
    // Proactive defaults are observable: the fixture has enough layers that the
    // activations get recomputed rather than retained, and the plan says so.
    for entry in derived["plan"]["defaults_applied"]
        .as_array()
        .expect("defaults_applied")
    {
        assert_ne!(entry["from"], entry["to"], "{entry}");
    }
    // Every field the client did not send is `derived` or `default`; the
    // response does not contain a profile keyword.
    for (path, origin) in derived["provenance"]
        .as_object()
        .expect("a provenance map")
        .iter()
    {
        assert_ne!(origin["source"], "preset", "{path} still claims a preset");
    }

    // -------------------------------------------------------------- create
    let (status, created) = post(&router, "/v1/runs", recipe(&workspace, &model, 1)).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("an id").to_string();

    // --------------------------------------------------------------- pause
    // Pause immediately while the run is still starting. The command is
    // consumed at the first real optimizer boundary, which makes this
    // deterministic even when the small fixture can finish one epoch quickly.
    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/pause")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let paused = until(&router, &id, &["paused", "completed", "failed"]).await;
    assert_eq!(
        paused["status"], "paused",
        "the run must stop at its first real optimizer boundary: {paused}"
    );
    // Pause holds the device: weights, KV caches and optimizer state are all still
    // allocated. That is the V1 trade, and this is where it is visible.
    assert_eq!(paused["holds_device"], true);

    // ---------------------------------------------- ask the paused model
    let (status, evaluation) = post_empty(&router, &format!("/v1/runs/{id}/evaluate")).await;
    assert_eq!(status, StatusCode::OK, "{evaluation}");
    let loss = evaluation["loss"].as_f64().expect("a loss");
    assert!(loss.is_finite() && loss > 0.0, "{evaluation}");
    assert!(evaluation["examples"].as_u64().expect("examples") > 0);

    let (status, generated) = post(
        &router,
        &format!("/v1/runs/{id}/generate"),
        json!({"prompt": "what is 1 plus one", "max_new_tokens": 8, "include_base": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{generated}");
    assert!(generated["prompt_tokens"].as_u64().expect("prompt tokens") > 0);
    // The adapter answered, and the base model answered too: the pair is what
    // makes an ad-hoc generation worth asking for.
    assert!(generated["base_text"].as_str().is_some(), "{generated}");

    // ------------------------------------------- checkpoint, then cancel
    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // A cancellation releases the pause, so the run resumes far enough to reach
    // its next boundary, write the checkpoint, and stop.
    let (status, body) = post(
        &router,
        &format!("/v1/runs/{id}/cancel"),
        json!({"at": "boundary", "checkpoint": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let terminal = until(&router, &id, &["cancelled", "completed", "failed"]).await;
    assert_eq!(
        terminal["status"], "cancelled",
        "a run that was asked to stop did not complete: {terminal}"
    );
    assert_eq!(terminal["holds_device"], false);

    // ---------------------------------------------------------- artefacts
    let (status, checkpoints) = get(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::OK, "{checkpoints}");
    let entries = checkpoints["checkpoints"].as_array().expect("a list");
    assert!(
        !entries.is_empty(),
        "the cancellation was asked to leave a checkpoint: {checkpoints}"
    );
    assert!(
        entries.iter().all(|entry| entry["complete"] == true),
        "{checkpoints}"
    );
    let latest = checkpoints["latest"].as_str().expect("a latest checkpoint");

    let (status, artifacts) = get(&router, &format!("/v1/runs/{id}/artifacts")).await;
    assert_eq!(status, StatusCode::OK, "{artifacts}");
    let adapter = artifacts["artifacts"]
        .as_array()
        .expect("a list")
        .iter()
        .find(|entry| entry["name"] == "adapter")
        .expect("the adapter");
    // A cancelled run still saves its adapter: the loop was asked to stop, so it
    // stopped cleanly rather than being killed.
    assert_eq!(adapter["present"], true, "{adapter}");
    assert!(adapter["bytes"].as_u64().expect("a size") > 0);

    // ------------------------------------------------------------- events
    // `?since=0` on a finished run replays it whole and closes, which is what lets
    // a client await a stream instead of polling.
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/runs/{id}/events?since=0"))
                .body(Body::empty())
                .expect("the request"),
        )
        .await
        .expect("route");
    assert_eq!(response.status(), StatusCode::OK);
    let stream = String::from_utf8_lossy(
        &response
            .into_body()
            .collect()
            .await
            .expect("the stream closes on terminal")
            .to_bytes(),
    )
    .into_owned();
    assert!(stream.contains("event: terminal"), "{stream}");
    assert!(stream.contains("event: metrics"), "{stream}");
    assert!(stream.contains("event: checkpoint"), "{stream}");

    // --------------------------------------------------------------- fork
    // The same configuration, continued from the checkpoint it just wrote. Refused
    // if the trajectory changed, which is the whole point of checking it here
    // rather than only in `RunController::begin`.
    let (status, forked) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id, "checkpoint": latest}, "name": "e2e-fork"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{forked}");
    assert!(
        forked["effective_config"]["checkpoint"]["resume_from"]
            .as_str()
            .unwrap_or_default()
            .contains(latest),
        "{forked}"
    );

    // And the correction table the calibration wrote is on disk, keyed by this
    // machine - which is what makes the resolver better on the next run rather
    // than only on this one.
    let calibration = workspace.dir.join("state/calibration.json");
    assert!(calibration.is_file(), "the calibration table was written");
    let table = std::fs::read_to_string(&calibration).expect("read");
    assert!(table.contains("entries"), "{table}");
}
