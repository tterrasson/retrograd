//! Shared harness for the runtime tests: a fixture on disk, a fake model probe,
//! a fake engine, and the request helpers.
//!
//! Nothing here loads a model or touches a device. What that leaves under test is
//! everything the server actually owns - the state machine, the device queue, the
//! control channel, the journal, the event ring and the HTTP surface over all of
//! them - which is the part a GPU lane would only slow down without covering
//! better.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use retrograd_config::{CheckpointMode, RunConfig};
use retrograd_core::{
    Device, MemoryReport, ModelInfo, Result as CoreResult, TargetSet, TrainMetrics,
};
use retrograd_dataset::DataFormat;
use retrograd_metrics::{MetricEvent, MetricValue, MetricsSink};
use retrograd_plan::MemoryBaseline;
use retrograd_plan::cost::{self, Calibration, Workload, WorkloadKind};
use retrograd_run::{
    AdHocEvaluation, ControlPoint, Flow, GenerationOutput, GenerationRequest, LoopPlan, RunControl,
    RunControls, RunObserver, RunOutcome, SftEpoch,
};
pub use retrograd_server::build_router;
use retrograd_server::runtime::RunEngine;
use retrograd_server::state::ModelProbe;
use retrograd_server::{AppState, Catalog, ServerConfig};
use serde_json::{Value, json};
use tower::ServiceExt;

pub const GIB: u64 = 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Fake probe
// ---------------------------------------------------------------------------

pub struct FakeProbe(pub ModelInfo);

impl ModelProbe for FakeProbe {
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
        Ok(self.0.clone())
    }

    fn measure(&self, _model: &Path, config: &RunConfig) -> CoreResult<MemoryReport> {
        let estimate = cost::estimate(
            &self.0,
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
        Ok(MemoryReport {
            model_weight_bytes: estimate.model_weight_bytes,
            optimizer_kv_bytes: estimate.optimizer_kv_bytes,
            optimizer_compute_bytes: estimate.optimizer_compute_bytes,
            device_bytes: estimate.device_bytes(),
            device_memory_samples: 1,
            ..Default::default()
        })
    }

    fn tokenize_lengths(
        &self,
        _model: &Path,
        path: &Path,
        format: DataFormat,
    ) -> CoreResult<Vec<u32>> {
        retrograd_dataset::measured_lengths(&FakeTokenizer, path, format)
    }
}

/// A stand-in tokenizer: four characters to the token, which is roughly where a
/// real BPE vocabulary sits for English. `retrograd_plan`'s character heuristic
/// deliberately assumes three, so a length measured through this fake is below
/// the estimate. This keeps that relationship testable without a GGUF.
struct FakeTokenizer;

impl retrograd_dataset::DatasetBackend for FakeTokenizer {
    fn tokenize_text(&self, text: &str) -> CoreResult<Vec<i32>> {
        Ok(vec![0; text.chars().count().div_ceil(4)])
    }

    fn eos_token(&self) -> CoreResult<i32> {
        Ok(0)
    }

    fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> CoreResult<String> {
        let mut rendered = String::new();
        for (role, content) in messages {
            rendered.push_str(&format!("<|{role}|>{content}<|end|>"));
        }
        if add_assistant {
            rendered.push_str("<|assistant|>");
        }
        Ok(rendered)
    }
}

pub fn model() -> ModelInfo {
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

// ---------------------------------------------------------------------------
// Fake engine
// ---------------------------------------------------------------------------

/// A gate the test opens when it wants the run to proceed. Without one the run
/// finishes immediately; with one it holds the device until released, which is
/// how the queue is observable.
#[derive(Default)]
pub struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    pub fn wait(&self) {
        let mut open = self.open.lock().expect("gate");
        while !*open {
            open = self.changed.wait(open).expect("gate");
        }
    }

    pub fn release(&self) {
        *self.open.lock().expect("gate") = true;
        self.changed.notify_all();
    }
}

pub enum Behaviour {
    Succeeds,
    Fails(&'static str),
    Panics,
}

/// What the fake loop offers the control plane, and what it remembers of being
/// driven. The real one is a `Trainer` plus a `RunController`; this one is three
/// fields, which is the whole point of `RunControls` being a trait.
#[derive(Default)]
pub struct Knobs {
    pub learning_rate: Option<f32>,
    pub every_iterations: Option<u32>,
    pub patience: Option<u32>,
    pub every_steps: Option<u64>,
    pub mode: Option<CheckpointMode>,
    checkpoint_pending: bool,
    /// How many ad-hoc evaluations the control plane asked for, and how many
    /// completions it sampled. The point of the counters is that a *request* the
    /// run never served must not look like one it did.
    pub evaluations: usize,
    pub generations: usize,
    pub last_prompt: Option<String>,
}

impl RunControls for Knobs {
    fn set_learning_rate(&mut self, learning_rate: f32) -> CoreResult<()> {
        self.learning_rate = Some(learning_rate);
        Ok(())
    }
    fn set_evaluation_every(&mut self, every_iterations: u32) -> CoreResult<()> {
        self.every_iterations = Some(every_iterations);
        Ok(())
    }
    fn set_patience(&mut self, patience: Option<u32>) -> CoreResult<()> {
        self.patience = patience;
        Ok(())
    }
    fn set_checkpoint_every_steps(&mut self, every_steps: u64) -> CoreResult<()> {
        self.every_steps = Some(every_steps);
        Ok(())
    }
    fn set_checkpoint_mode(&mut self, mode: CheckpointMode) -> CoreResult<()> {
        self.mode = Some(mode);
        Ok(())
    }
    fn request_checkpoint(&mut self) {
        self.checkpoint_pending = true;
    }
    fn evaluate(&mut self) -> CoreResult<AdHocEvaluation> {
        self.evaluations += 1;
        Ok(AdHocEvaluation {
            loss: Some(0.375),
            perplexity: Some(1.455),
            examples: 200,
            ..Default::default()
        })
    }
    fn generate(&mut self, request: &GenerationRequest) -> CoreResult<GenerationOutput> {
        self.generations += 1;
        self.last_prompt = Some(request.prompt.clone());
        Ok(GenerationOutput {
            // Echoes the two flags a caller can set, so a test can prove the
            // request reached the model rather than a default somewhere between.
            text: format!(
                "[chat={} n={}] answer",
                request.chat, request.max_new_tokens
            ),
            prompt_tokens: request.prompt.split_whitespace().count() as u64,
            tokens: 5,
            base_text: request.include_base.then(|| "base answer".to_string()),
        })
    }
}

pub struct FakeEngine {
    behaviour: Behaviour,
    epochs: u32,
    /// Wall time per epoch. Zero for the tests that only care about the outcome;
    /// a few milliseconds for the ones that have to reach a live run with a
    /// command before it finishes.
    tick: Duration,
    gate: Option<Arc<Gate>>,
    runs: AtomicUsize,
    /// What the control plane turned, readable from the test after the run.
    pub knobs: Arc<Mutex<Knobs>>,
}

impl FakeEngine {
    fn build(
        behaviour: Behaviour,
        epochs: u32,
        tick: Duration,
        gate: Option<Arc<Gate>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            behaviour,
            epochs,
            tick,
            gate,
            runs: AtomicUsize::new(0),
            knobs: Arc::new(Mutex::new(Knobs::default())),
        })
    }

    pub fn succeeding() -> Arc<Self> {
        Self::build(Behaviour::Succeeds, 2, Duration::ZERO, None)
    }

    pub fn with(behaviour: Behaviour) -> Arc<Self> {
        Self::build(behaviour, 1, Duration::ZERO, None)
    }

    pub fn gated(gate: Arc<Gate>) -> Arc<Self> {
        Self::build(Behaviour::Succeeds, 1, Duration::ZERO, Some(gate))
    }

    /// A run long enough to be commanded while it is alive: `epochs` iterations,
    /// each one `tick` long, polling control twice per iteration (once between
    /// boundaries, once at one).
    pub fn slow(epochs: u32, tick: Duration) -> Arc<Self> {
        Self::build(Behaviour::Succeeds, epochs, tick, None)
    }

    pub fn executions(&self) -> usize {
        self.runs.load(Ordering::SeqCst)
    }
}

impl RunEngine for FakeEngine {
    fn execute(
        &self,
        config: &RunConfig,
        observer: &mut dyn RunObserver,
        control: &mut dyn RunControl,
        mut sinks: Vec<Box<dyn MetricsSink>>,
    ) -> CoreResult<RunOutcome> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        observer.model_load_started();
        observer.model_load_finished(Duration::from_millis(1));
        observer.loop_started(&LoopPlan::Sft {
            epochs: self.epochs,
            has_eval: false,
            steps_per_epoch: 10,
        });
        if let Some(gate) = &self.gate {
            gate.wait();
        }
        if let Behaviour::Panics = self.behaviour {
            panic!("the fake engine was asked to panic");
        }

        let knobs = self.knobs.clone();
        let mut knobs = knobs.lock().expect("knobs");
        let mut last_step = 0;
        for epoch in 1..=self.epochs {
            let step = epoch as u64 * 10;
            // Mid-iteration: where a `cancel at=now` lands, and where a pause
            // that arrived between two boundaries takes hold.
            let flow = control.poll(
                &mut *knobs,
                ControlPoint {
                    iteration: epoch,
                    global_step: step - 5,
                    at_boundary: false,
                },
            )?;
            if flow == Flow::Stop {
                break;
            }
            std::thread::sleep(self.tick);
            last_step = step;
            observer.sft_epoch(&SftEpoch {
                epoch,
                total_epochs: self.epochs,
                global_step: step,
                train_loss: 1.0 / epoch as f32,
                // The observer's "not evaluated this epoch". It must not reach
                // the wire as `null`.
                eval_loss: f32::NAN,
                learning_rate: knobs.learning_rate.unwrap_or(config.training.learning_rate),
                tokens_per_second: 1234.0,
            });
            for sink in &mut sinks {
                sink.emit(&MetricEvent::Step {
                    epoch,
                    global_step: step,
                    values: vec![
                        MetricValue {
                            name: "train/loss".into(),
                            value: 1.0 / epoch as f32,
                        },
                        MetricValue {
                            name: "train/lr".into(),
                            value: knobs.learning_rate.unwrap_or(config.training.learning_rate),
                        },
                    ],
                })?;
            }
            // The boundary: the only point a checkpoint may be written, and
            // where a `cancel at=boundary` lands.
            let flow = control.poll(
                &mut *knobs,
                ControlPoint {
                    iteration: epoch,
                    global_step: step,
                    at_boundary: true,
                },
            )?;
            if std::mem::take(&mut knobs.checkpoint_pending) {
                observer
                    .checkpoint_written(&PathBuf::from(format!("checkpoints/step-{step}.state")));
            }
            if flow == Flow::Stop {
                break;
            }
        }
        observer.loop_finished();
        match self.behaviour {
            Behaviour::Fails(message) => Err(retrograd_core::Error::runtime(message)),
            _ => Ok(RunOutcome {
                metrics: TrainMetrics {
                    global_step: last_step,
                    train_loss: 0.25,
                    ..Default::default()
                },
                early_stopped: false,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture and requests
// ---------------------------------------------------------------------------

pub struct Fixture {
    pub dir: PathBuf,
}

impl Fixture {
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("retrograd-runs-api-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        std::fs::write(dir.join("model.gguf"), b"not a real GGUF").expect("write the model stub");
        let mut data = String::new();
        for index in 0..200 {
            data.push_str(&format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"question {index}\"}},\
                 {{\"role\":\"assistant\",\"content\":\"an answer\"}}]}}\n"
            ));
        }
        std::fs::write(dir.join("data.jsonl"), data).expect("write the dataset");
        Self { dir }
    }

    pub fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }

    pub fn state_dir(&self) -> PathBuf {
        self.dir.join("state")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A server over the fixture's state directory.
///
/// `calibrate_runs` defaults to false here and to *true* in production: these
/// tests are about the runtime, and measuring on every creation would put the
/// calibration pass in the way of every assertion.
pub fn state_of(fixture: &Fixture, engine: Arc<dyn RunEngine>, calibrate: bool) -> AppState {
    let toml = format!(
        "state_dir = \"{}\"\ncalibrate_runs = {calibrate}\n",
        fixture.path("state")
    );
    let config: ServerConfig = toml::from_str(&toml).expect("parse the server config");
    let catalog = Catalog::declare(
        &config.rewards,
        &config.judges,
        &config.mcp_servers,
        &config.environments,
    )
    .expect("declare the catalog");
    let mut state =
        AppState::new(config, catalog, Arc::new(FakeProbe(model()))).with_engine(engine);
    state.baseline = MemoryBaseline {
        device_total: Some(24 * GIB),
        device_used: 0,
        host_total: Some(64 * GIB),
        host_used: 0,
        unified: false,
    };
    state
}

pub fn router_for(fixture: &Fixture, engine: Arc<dyn RunEngine>) -> Router {
    build_router(state_of(fixture, engine, false))
}

pub fn recipe(fixture: &Fixture, name: &str) -> Value {
    json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "budget": {"epochs": 2},
            "seed": 7
        },
        "name": name
    })
}

/// The same recipe with an evaluation dataset and a checkpoint directory, so the
/// `PATCH` whitelist has something to adjust.
pub fn recipe_with_schedules(fixture: &Fixture, name: &str) -> Value {
    json!({
        "recipe": {
            "objective": "instruction-tuning",
            "model": fixture.path("model.gguf"),
            "data": {"path": fixture.path("data.jsonl"), "format": "jsonl"},
            "eval": {"path": fixture.path("data.jsonl")},
            "budget": {"epochs": 2},
            "seed": 7
        },
        "params": {
            "checkpoint": {"directory": fixture.path("ckpt"), "mode": "steps", "every_steps": 10}
        },
        "name": name
    })
}

// ---------------------------------------------------------------------------
// Checkpoints on disk
// ---------------------------------------------------------------------------

/// Writes a complete checkpoint the way `RunController` would.
///
/// Through `retrograd_checkpoint::Checkpoint::write` rather than by hand, so what
/// the listing and the fork read is the real format - manifest last, adapter
/// inside, sibling GGUF export - and not a fixture that agrees with the reader
/// only by coincidence.
pub fn write_checkpoint(
    directory: &Path,
    id: &str,
    global_step: u64,
    algorithm: &str,
    trajectory_signature: &str,
    model: &Path,
) -> PathBuf {
    use retrograd_checkpoint as ckpt;

    let state_dir = directory.join(format!("{id}.{}", ckpt::STATE_SUFFIX));
    let checkpoint = ckpt::Checkpoint {
        manifest: ckpt::Manifest {
            format_version: ckpt::FORMAT_VERSION,
            checkpoint_id: id.to_string(),
            global_step,
            adapter: ckpt::ADAPTER_FILE.to_string(),
            files: ckpt::REQUIRED_FILES.iter().map(|f| f.to_string()).collect(),
            app_version: "test".into(),
            llama_cpp_commit: "test".into(),
            model_signature: "test-signature".into(),
            model_bytes: std::fs::metadata(model).map(|meta| meta.len()).unwrap_or(0),
            model_fingerprint: retrograd_checkpoint::fingerprint_file(model)
                .expect("fingerprint the model stub"),
            algorithm: algorithm.to_string(),
            trajectory_signature: trajectory_signature.to_string(),
            resume_boundary: "epoch".into(),
            artifacts: Default::default(),
        },
        progress: ckpt::Progress {
            version: ckpt::FORMAT_VERSION,
            epoch: 1,
            global_step,
            cursor: 0,
            algorithm: algorithm.to_string(),
            phase: "train".into(),
            best_eval: None,
            stale_evaluations: 0,
            kl_multiplier: None,
        },
        scheduler: ckpt::Scheduler {
            version: ckpt::FORMAT_VERSION,
            ..Default::default()
        },
        optimizer: ckpt::Optimizer {
            version: ckpt::FORMAT_VERSION,
            ..Default::default()
        },
        rng: ckpt::Rng {
            version: ckpt::FORMAT_VERSION,
            ..Default::default()
        },
        dataset: ckpt::Dataset {
            version: ckpt::FORMAT_VERSION,
            ..Default::default()
        },
        artifacts: Default::default(),
    };
    checkpoint
        .write(&state_dir, |path| {
            std::fs::write(path, b"not a real adapter").map_err(Into::into)
        })
        .expect("write the checkpoint");
    state_dir
}

/// The trajectory fingerprint of a run the API already created, read back from its
/// own effective configuration.
///
/// A fork is refused when the fingerprints disagree, so a test that wants an
/// *accepted* fork has to write a checkpoint carrying the parent's - which means
/// computing it the way the server does, from the document the server rendered.
pub fn trajectory_of(effective_config: &Value) -> String {
    let document: retrograd_config::ConfigDocument =
        serde_json::from_value(effective_config.clone()).expect("the effective config re-reads");
    let config = retrograd_config::build(document, Path::new(".")).expect("the config builds");
    retrograd_run::trajectory_signature(&config).expect("a trajectory signature")
}

pub async fn send(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let (status, _, body) = send_raw(router, request).await;
    let body = serde_json::from_slice(&body).unwrap_or_else(|error| {
        panic!(
            "the body is not JSON ({error}): {}",
            String::from_utf8_lossy(&body)
        )
    });
    (status, body)
}

/// The bytes as they went on the wire, for the assertions that are about the
/// bytes rather than the values.
pub async fn send_raw(router: &Router, request: Request<Body>) -> (StatusCode, String, Vec<u8>) {
    let response = router
        .clone()
        .oneshot(request)
        .await
        .expect("route the request");
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
        .to_bytes()
        .to_vec();
    if !status.is_success() {
        assert_eq!(
            content_type,
            "application/problem+json",
            "every failure is a problem document: {}",
            String::from_utf8_lossy(&bytes)
        );
    }
    (status, content_type, bytes)
}

pub async fn post(router: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    post_with(router, uri, body, &[]).await
}

pub async fn post_with(
    router: &Router,
    uri: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, Value) {
    send(router, build(http::Method::POST, uri, Some(body), headers)).await
}

pub async fn patch(router: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(router, build(http::Method::PATCH, uri, Some(body), &[])).await
}

/// A `POST` with no body at all, which is what a client cancelling a run sends.
pub async fn post_empty(router: &Router, uri: &str) -> (StatusCode, Value) {
    send(
        router,
        Request::builder()
            .method("POST")
            .uri(uri)
            .body(Body::empty())
            .expect("build the request"),
    )
    .await
}

pub async fn get(router: &Router, uri: &str) -> (StatusCode, Value) {
    send(router, build(http::Method::GET, uri, None, &[])).await
}

fn build(
    method: http::Method,
    uri: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> Request<Body> {
    let mut request = Request::builder().method(method).uri(uri);
    if body.is_some() {
        request = request.header(http::header::CONTENT_TYPE, "application/json");
    }
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .expect("build the request")
}

/// Polls a run until it reaches a terminal state.
///
/// A run is asynchronous by design - the `201` comes back before the model is
/// loaded - so a test that asserted on the body of the creation would be
/// asserting on a race. Ten seconds is far beyond what a fake engine needs and
/// short enough to fail rather than hang a lane.
pub async fn wait_for_terminal(router: &Router, id: &str) -> Value {
    for _ in 0..1000 {
        let (status, body) = get(router, &format!("/v1/runs/{id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let state = body["status"].as_str().unwrap_or_default().to_string();
        if matches!(
            state.as_str(),
            "completed" | "failed" | "cancelled" | "interrupted"
        ) {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the run never reached a terminal state");
}

pub async fn wait_for_status(router: &Router, id: &str, expected: &str) -> Value {
    for _ in 0..1000 {
        let (_, body) = get(router, &format!("/v1/runs/{id}")).await;
        if body["status"] == expected {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (_, body) = get(router, &format!("/v1/runs/{id}")).await;
    panic!(
        "the run never reached '{expected}', it is '{}'",
        body["status"]
    );
}
