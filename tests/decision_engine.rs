//! Decision models on the engine, end to end, against the golden fixtures of
//! `tests/fixtures/decision`, recorded from llama-server's decision code on
//! the same model and backend.
//!
//! The engine renders the `systemone` template the GGUF carries, tokenizes
//! with the special token texts parsed and none added, lays the tokens out,
//! evaluates every question of a request in one call (one forward pass per
//! question, or several packed together on request) and turns the scores into
//! answers.
//! What has to hold: the tokens of every pass and where they are read are the
//! recorded ones exactly, the token count of the response is exact, a refused
//! request is refused with the recorded message, and the probabilities agree
//! with the recorded answers of the same backend.
//!
//! The same recordings are replayed through the serving session that answers
//! `/v1/systemone`, with the production loader: what a client of the server
//! gets must be what the engine computes.
//!
//! The models: the small `laya` test model (`TINYLAYA_FIXTURE.toml`; the test
//! skips when it is absent, unless `RETRO_REQUIRE_CPU_FIXTURE` is set), and one
//! tiny model per readout family, generated here by
//! `scripts/gen-tiny-decision-fixture.py --type NAME` and checked against the
//! digest its recording names. A CPU recording is valid for the architecture
//! it was recorded on (`golden-<model>.cpu-<arch>.json`); without one for this
//! host, the replay skips.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use retrograd::decision::readout::Head;
use retrograd::decision::request::SystemOneRequest;
use retrograd::decision::template::{self, PromptInput};
use retrograd::decision::{DecisionError, Json};
use retrograd::trainer::{DECISION_TEMPLATE, TemplateSource, decision_metadata};
use retrograd::{Device, Engine, EngineConfig, EngineError, SamplingParams};
use retrograd_openai::{
    DecisionOptions, EngineLoader, Session, SessionOptions, Unshared, WeightsSpec,
};
use serde_json::Value;

/// How far a probability may be from the recorded one on the same backend when
/// several questions share a forward pass: that changes the shapes of the
/// matrix products, hence the kernels that compute them (measured on this
/// model: 8.9e-5 on Metal, 7.7e-4 on the CPU). The tolerance the fixtures'
/// README states for a same-backend comparison.
const TOLERANCE: f64 = 1e-3;

/// One question per forward pass runs the recording's kernels on the
/// recording's shapes - the CPU recordings are built with this crate's ggml
/// options for that reason: the scores are the recorded ones, and the
/// probabilities differ by the printing of the recorded JSON alone.
const EXACT: f64 = 1e-12;

/// The generated tiny model of each readout family and its request file.
const FAMILIES: [(&str, &str); 11] = [
    ("openjev", "requests-labels.json"),
    ("lev", "requests-labels.json"),
    ("nimble", "requests-labels.json"),
    ("vllm-d3", "requests-labels.json"),
    ("lfm2-d1", "requests-labels.json"),
    ("lfm3-d1", "requests-labels.json"),
    ("h2o-lightning", "requests-h2o.json"),
    ("vllm-decision2", "requests-joint.json"),
    ("clef", "requests-joint.json"),
    ("kev", "requests-kev.json"),
    ("lfm2-d1-omni", "requests-d1omni.json"),
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decision")
}

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("RETRO_TINYLAYA_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixtures().join("tinylaya-for-testing-Q8_0.gguf"));
    if path.exists() {
        Some(path)
    } else if std::env::var_os("RETRO_REQUIRE_CPU_FIXTURE").is_some() {
        panic!(
            "decision fixture missing at {}; run scripts/fetch-cpu-fixture.sh decision/TINYLAYA_FIXTURE",
            path.display()
        );
    } else {
        eprintln!("skipping: decision fixture missing at {}", path.display());
        None
    }
}

/// The recording of `model` on `backend` (`cpu` is this host's architecture),
/// or `None` when there is none.
fn golden_path(model: &str, backend: &str) -> Option<PathBuf> {
    let backend = if backend == "cpu" {
        format!("cpu-{}", std::env::consts::ARCH)
    } else {
        backend.to_owned()
    };
    let path = fixtures().join(format!("golden-{model}.{backend}.json"));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping {model}: no recording {}", path.display());
        None
    }
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture is JSON")
}

/// The request file, keys in the order written: a `Value` sorts them, and the
/// order of the questions is the order of the answers.
fn ordered_requests(file: &str) -> Vec<Json> {
    let text = std::fs::read_to_string(fixtures().join(file)).expect("requests readable");
    let requests: Json = serde_json::from_str(&text).expect("requests are JSON");
    requests["requests"].as_array().expect("requests").to_vec()
}

fn sha256(path: &Path) -> String {
    let out = Command::new("python3")
        .args([
            "-c",
            "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",
        ])
        .arg(path)
        .output()
        .expect("python3 runs");
    assert!(out.status.success(), "sha256 of {}", path.display());
    String::from_utf8(out.stdout)
        .expect("hex")
        .trim()
        .to_owned()
}

/// The generated tiny model of `kind`, written into the test target directory
/// and checked against the digest `golden` recorded.
fn generated_model(kind: &str, golden: &Value) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("decision-fixtures");
    std::fs::create_dir_all(&dir).expect("fixture directory");
    let path = dir.join(format!("retrograd-tiny-decision-{kind}-f32.gguf"));
    let expected = golden["model"]["sha256"].as_str().expect("recorded digest");
    if !path.exists() || sha256(&path) != expected {
        let generator =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/gen-tiny-decision-fixture.py");
        let status = Command::new("python3")
            .arg(generator)
            .args(["--type", kind])
            .arg(&path)
            .status()
            .expect("python3 runs");
        assert!(status.success(), "generating the {kind} fixture");
        assert_eq!(
            sha256(&path),
            expected,
            "the {kind} generator no longer writes the recorded model"
        );
    }
    path
}

/// An engine whose micro-batch holds the longest recorded prompt: the generated
/// laya model spells its byte-level prompts in up to 700 tokens.
fn engine(path: &Path, device: Device) -> Engine {
    Engine::new(
        path,
        &EngineConfig {
            device,
            n_batch: 2048,
            n_ubatch: 2048,
            ..EngineConfig::default()
        },
    )
    .expect("engine loads the decision model")
}

fn int_vec(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("integer array")
        .iter()
        .map(|token| i32::try_from(token.as_i64().expect("integer")).expect("fits i32"))
        .collect()
}

/// Every probability (and `noul`, `score`) of `actual` against `expected`,
/// the keys in the same order, the other fields equal. Returns the largest
/// difference.
fn compare_answers(name: &str, expected: &Value, actual: &Value) -> f64 {
    let expected = expected.as_object().expect("answers object");
    let actual = actual.as_object().expect("answers object");
    assert_eq!(
        expected.keys().collect::<Vec<_>>(),
        actual.keys().collect::<Vec<_>>(),
        "{name}: question order"
    );
    let mut largest = 0.0_f64;
    for (id, want) in expected {
        let got = &actual[id];
        assert_eq!(want["type"], got["type"], "{name}/{id}: type");
        assert_eq!(want.get("legend"), got.get("legend"), "{name}/{id}: legend");
        for field in ["noul", "score", "confidence"] {
            if let Some(value) = want.get(field) {
                let diff =
                    (value.as_f64().expect("number") - got[field].as_f64().expect("number")).abs();
                largest = largest.max(diff);
            }
        }
        if let Some(probabilities) = want.get("probabilities").and_then(Value::as_object) {
            let got_probabilities = got["probabilities"].as_object().expect("probabilities");
            assert_eq!(
                probabilities.keys().collect::<Vec<_>>(),
                got_probabilities.keys().collect::<Vec<_>>(),
                "{name}/{id}: option order"
            );
            for (key, value) in probabilities {
                let diff = (value.as_f64().expect("number")
                    - got_probabilities[key].as_f64().expect("number"))
                .abs();
                largest = largest.max(diff);
            }
        }
    }
    largest
}

/// The questions whose choice is not the recorded one.
fn changed_choices(name: &str, expected: &Value, actual: &Value, changed: &mut Vec<String>) {
    for (id, want) in expected.as_object().expect("answers") {
        if let Some(choice) = want.get("choice")
            && choice != &actual[id]["choice"]
        {
            changed.push(format!("{name}/{id}"));
        }
    }
}

/// The message of a refused decision, as a client reads it.
fn refusal(error: &EngineError) -> Option<&str> {
    match error {
        EngineError::Decision(DecisionError::InvalidRequest(message)) => Some(message),
        _ => None,
    }
}

/// A model of the recordings: the file, its recording and its requests.
struct Fixture {
    name: String,
    path: PathBuf,
    golden: Value,
    requests: &'static str,
}

/// Replays every recorded request; `pack` is how many questions one forward
/// pass may evaluate. Returns the largest probability difference and the
/// questions whose choice is not the recorded one.
fn replay(fixture: &Fixture, device: Device, pack: usize) -> (f64, Vec<String>) {
    let golden = &fixture.golden;
    let mut engine = engine(&fixture.path, device);
    let model_info = engine
        .decision_info()
        .expect("decision head")
        .with_packing(pack);
    assert_eq!(
        model_info.decision_type,
        golden["model"]["decision_type"].as_str().expect("type")
    );
    if let Some(budget) = golden["model"]["max_head_tokens"].as_u64() {
        assert_eq!(model_info.model_max_head_tokens, Some(budget as usize));
    }

    let mut largest = 0.0_f64;
    let mut choices_changed = Vec::new();
    for (entry, recorded) in ordered_requests(fixture.requests)
        .iter()
        .zip(golden["requests"].as_array().expect("golden requests"))
    {
        let name = entry["name"].as_str().expect("name");
        assert_eq!(recorded["name"], name);
        let info = match entry["max_head_tokens"].as_u64() {
            Some(budget) => model_info
                .clone()
                .with_max_head_tokens(usize::try_from(budget).expect("budget fits usize"))
                .expect("budget override"),
            None => model_info.clone(),
        };
        let body = entry["body"].to_string().into_bytes();
        let parsed = SystemOneRequest::from_slice(&body, &info.readout);
        if let Some(message) = recorded["error"].as_str() {
            // Refused when it is parsed, or when it is laid out.
            match parsed {
                Err(DecisionError::InvalidRequest(got)) => assert_eq!(got, message, "{name}"),
                Err(other) => panic!("{name}: refused with {other}, recorded {message}"),
                Ok(request) => {
                    let error = engine.decide(&info, &request).expect_err(name);
                    assert_eq!(refusal(&error), Some(message), "{name}: {error}");
                }
            }
            continue;
        }
        let request = parsed.expect("valid request");

        // The tokens of every pass and where they are read are the recorded
        // ones.
        let passes = engine.decision_passes(&info, &request).expect("passes");
        let traces = recorded["questions"].as_array().expect("passes");
        assert_eq!(passes.len(), traces.len(), "{name}: passes");
        for (pass, trace) in passes.iter().zip(traces) {
            let at = format!("{name}/{}", trace["id"]);
            assert_eq!(pass.tokens, int_vec(&trace["tokens"]), "{at}: tokens");
            let reads = match &info.readout.head {
                Head::LabelLogits { .. } | Head::LabelGroups => int_vec(&trace["labels"]),
                Head::Joint => int_vec(&trace["order"]),
                Head::Pooled { .. } => Vec::new(),
                _ => int_vec(&trace["markers"]),
            };
            assert_eq!(pass.reads, reads, "{at}: reads");
            assert_eq!(
                pass.label_groups,
                int_vec(&trace["label_groups"]),
                "{at}: label groups"
            );
            assert_eq!(
                Some(i64::from(pass.pointer)),
                trace["pointer"].as_i64(),
                "{at}: pointer"
            );
            if matches!(
                info.readout.head,
                Head::MarkerColumn { .. } | Head::D1Omni { .. }
            ) {
                assert_eq!(
                    Some(i64::from(pass.column)),
                    trace["column"].as_i64(),
                    "{at}"
                );
            }
            assert_eq!(
                pass.n_scores,
                trace["scores"].as_array().expect("scores").len(),
                "{at}: scores"
            );
        }

        // A marker prompt renders through the engine's template as recorded.
        if matches!(info.readout.head, Head::MarkerColumn { .. }) {
            for (trace, (id, question)) in traces.iter().zip(&request.questions) {
                let input = PromptInput {
                    state: &request.state,
                    questions: Some(&request.questions),
                    question_id: id,
                    question,
                    labels: &[],
                    variant: 0,
                    n_images: 0,
                    media_marker: "",
                };
                let context = template::context(&info.readout, &input).expect("context");
                let prompt = engine
                    .render_named_template(DECISION_TEMPLATE, &context)
                    .expect("render");
                assert_eq!(
                    Some(prompt.as_str()),
                    trace["prompt"].as_str(),
                    "{name}/{id}"
                );
            }
        }

        let decision = engine.decide(&info, &request).expect("decide");
        assert_eq!(
            Some(decision.input_tokens as u64),
            recorded["response"]["usage"]["input_tokens"].as_u64(),
            "{name}: input tokens"
        );
        let answers = serde_json::to_value(&decision.answers).expect("answers serialize");
        let expected = &recorded["response"]["answers"];
        changed_choices(name, expected, &answers, &mut choices_changed);
        largest = largest.max(compare_answers(name, expected, &answers));
    }
    eprintln!(
        "{} ({}), {pack} question(s) per pass: largest probability difference {largest:e}, \
         choices changed: {choices_changed:?}",
        fixture.name, golden["backend"]
    );
    (largest, choices_changed)
}

/// Replays every recorded request through the serving session, one question
/// per pass (the serving default), the budget override of a request passed as
/// the decision option of its spec. Returns the largest probability
/// difference and the questions whose choice is not the recorded one; the
/// token counts and the refusals must be exact.
async fn serve_replay(fixture: &Fixture, device: Device) -> (f64, Vec<String>) {
    let session = Session::new(
        Arc::new(EngineLoader),
        Arc::new(Unshared),
        SessionOptions {
            idle: None,
            ..SessionOptions::default()
        },
    )
    .expect("the serving thread starts");
    let mut largest = 0.0_f64;
    let mut choices_changed = Vec::new();
    for (entry, recorded) in ordered_requests(fixture.requests).iter().zip(
        fixture.golden["requests"]
            .as_array()
            .expect("golden requests"),
    ) {
        let name = entry["name"].as_str().expect("name");
        let spec = WeightsSpec {
            model: fixture.path.clone(),
            adapter: None,
            n_ctx: EngineConfig::default().n_ctx,
            device,
            n_seq_max: None,
            chat_template_variables: None,
            decision: DecisionOptions {
                max_head_tokens: entry["max_head_tokens"]
                    .as_u64()
                    .map(|budget| usize::try_from(budget).expect("budget fits usize")),
                ..DecisionOptions::default()
            },
            experts: Default::default(),
            expert_history: None,
        };
        let result = session.decide(spec, entry["body"].clone()).await;
        if let Some(message) = recorded["error"].as_str() {
            let error = result.err().unwrap_or_else(|| panic!("{name}: answered"));
            assert_eq!(error.to_string(), message, "{name}");
            continue;
        }
        let decision = result.unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            Some(decision.input_tokens as u64),
            recorded["response"]["usage"]["input_tokens"].as_u64(),
            "{name}: input tokens"
        );
        let answers = serde_json::to_value(&decision.answers).expect("answers serialize");
        let expected = &recorded["response"]["answers"];
        changed_choices(name, expected, &answers, &mut choices_changed);
        largest = largest.max(compare_answers(name, expected, &answers));
    }
    eprintln!(
        "{} ({}), served: largest probability difference {largest:e}, \
         choices changed: {choices_changed:?}",
        fixture.name, fixture.golden["backend"]
    );
    (largest, choices_changed)
}

fn metal_available() -> bool {
    let available = cfg!(target_os = "macos") && retrograd::gpu_runtime_available();
    if !available {
        eprintln!("skipping: no Metal device");
    }
    available
}

/// A model of the `requests.json` recordings, on `backend`.
fn laya_fixture(name: &str, path: PathBuf, backend: &str) -> Option<Fixture> {
    Some(Fixture {
        name: name.to_owned(),
        path,
        golden: read_json(&golden_path(name, backend)?),
        requests: "requests.json",
    })
}

/// The generated tiny models with a recording for this host.
fn family_fixtures() -> Vec<Fixture> {
    FAMILIES
        .iter()
        .filter_map(|&(kind, requests)| {
            let name = format!("tiny-{kind}");
            let golden = read_json(&golden_path(&name, "cpu")?);
            Some(Fixture {
                path: generated_model(kind, &golden),
                name,
                golden,
                requests,
            })
        })
        .collect()
}

/// Asserts a replay is exact on the CPU, one question per pass.
fn assert_exact(fixture: &Fixture, (largest, changed): (f64, Vec<String>)) {
    assert!(
        changed.is_empty(),
        "{}: choices changed: {changed:?}",
        fixture.name
    );
    assert!(
        largest <= EXACT,
        "{}: one question per pass differs by {largest:e}",
        fixture.name
    );
}

#[test]
fn engine_reproduces_the_recorded_answers_on_cpu() {
    let Some(path) = model_path() else { return };
    let Some(fixture) = laya_fixture("tinylaya", path, "cpu") else {
        return;
    };
    let (largest, changed) = replay(&fixture, Device::Cpu, 8);
    assert!(changed.is_empty(), "cpu: choices changed: {changed:?}");
    assert!(
        largest <= TOLERANCE,
        "cpu: probabilities differ by {largest:e}"
    );
    assert_exact(&fixture, replay(&fixture, Device::Cpu, 1));
}

/// Every readout family on its generated tiny model, one question per pass.
#[test]
fn every_readout_family_reproduces_its_recorded_answers_on_cpu() {
    for fixture in family_fixtures() {
        assert_exact(&fixture, replay(&fixture, Device::Cpu, 1));
    }
    let tiny_laya = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/retrograd-tiny-decision-f32.gguf");
    let tiny_laya = std::env::var_os("RETRO_TINY_DECISION_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or(tiny_laya);
    if tiny_laya.exists()
        && let Some(fixture) = laya_fixture("tiny-laya", tiny_laya, "cpu")
    {
        assert_exact(&fixture, replay(&fixture, Device::Cpu, 1));
    }
}

#[test]
fn engine_reproduces_the_recorded_answers_on_metal() {
    let Some(path) = model_path() else { return };
    if !metal_available() {
        return;
    }
    let Some(fixture) = laya_fixture("tinylaya", path, "mtl") else {
        return;
    };
    let (largest, changed) = replay(&fixture, Device::Gpu, 8);
    assert!(changed.is_empty(), "mtl: choices changed: {changed:?}");
    assert!(
        largest <= TOLERANCE,
        "mtl: probabilities differ by {largest:e}"
    );
    let (largest, changed) = replay(&fixture, Device::Gpu, 1);
    assert!(changed.is_empty(), "mtl: choices changed: {changed:?}");
    assert!(
        largest <= EXACT,
        "mtl: one question per pass differs by {largest:e}"
    );
}

/// The real models, opt-in as for the recording (`RETRO_DECISION_JULIA`,
/// `RETRO_DECISION_LAYA`, `RETRO_DECISION_KEV`). One question per pass is
/// exact on either backend; packed passes are printed, not asserted: how far
/// packing moves the answers depends on the model (Julia-1 moves by 2.6e-1 on
/// the CPU).
#[test]
fn real_models_reproduce_the_recorded_answers_when_present() {
    for (model, variable, requests) in [
        ("julia-1", "RETRO_DECISION_JULIA", "requests.json"),
        ("laya", "RETRO_DECISION_LAYA", "requests.json"),
        ("kev", "RETRO_DECISION_KEV", "requests-kev.json"),
    ] {
        let Some(path) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("skipping {model}: {variable} is not set");
            continue;
        };
        let mut backends = vec![(Device::Cpu, "cpu")];
        if metal_available() {
            backends.push((Device::Gpu, "mtl"));
        }
        for (device, backend) in backends {
            let Some(golden) = golden_path(model, backend) else {
                continue;
            };
            let fixture = Fixture {
                name: model.to_owned(),
                path: path.clone(),
                golden: read_json(&golden),
                requests,
            };
            replay(&fixture, device, 8);
            assert_exact(&fixture, replay(&fixture, device, 1));
        }
    }
}

#[tokio::test]
async fn the_service_reproduces_the_recorded_answers() {
    let Some(path) = model_path() else { return };
    let mut backends = vec![(Device::Cpu, "cpu")];
    if metal_available() {
        backends.push((Device::Gpu, "mtl"));
    }
    for (device, backend) in backends {
        let Some(fixture) = laya_fixture("tinylaya", path.clone(), backend) else {
            continue;
        };
        assert_exact(&fixture, serve_replay(&fixture, device).await);
    }
    for fixture in family_fixtures() {
        assert_exact(&fixture, serve_replay(&fixture, Device::Cpu).await);
    }
}

/// The real models through the service, opt-in as for the engine replay.
#[tokio::test]
async fn the_service_reproduces_the_real_models_when_present() {
    for (model, variable, requests) in [
        ("julia-1", "RETRO_DECISION_JULIA", "requests.json"),
        ("laya", "RETRO_DECISION_LAYA", "requests.json"),
        ("kev", "RETRO_DECISION_KEV", "requests-kev.json"),
    ] {
        let Some(path) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("skipping {model}: {variable} is not set");
            continue;
        };
        let mut backends = vec![(Device::Cpu, "cpu")];
        if metal_available() {
            backends.push((Device::Gpu, "mtl"));
        }
        for (device, backend) in backends {
            let Some(golden) = golden_path(model, backend) else {
                continue;
            };
            let fixture = Fixture {
                name: model.to_owned(),
                path: path.clone(),
                golden: read_json(&golden),
                requests,
            };
            assert_exact(&fixture, serve_replay(&fixture, device).await);
        }
    }
}

#[test]
fn metadata_alone_names_the_head_and_its_template() {
    let Some(path) = model_path() else { return };
    let metadata = decision_metadata(&path).expect("metadata");
    assert_eq!(metadata.decision_type.as_deref(), Some("laya"));
    assert_eq!(metadata.architecture.as_deref(), Some("modern-bert"));
    assert!(
        metadata
            .templates
            .iter()
            .any(|name| name == DECISION_TEMPLATE)
    );
    assert!(metadata.mask_token.is_some() && metadata.sep_token.is_some());
    assert_eq!(
        metadata.metadata.get("type").map(String::as_str),
        Some("laya")
    );
    assert_eq!(metadata.n_embd_out, 3, "one column per question type");
}

#[test]
fn an_encoder_refuses_generation_with_a_clear_error() {
    let Some(path) = model_path() else { return };
    let mut engine = engine(&path, Device::Cpu);
    let error = engine
        .generate(
            &[1, 2, 3],
            &SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens: 4,
                seed: 0,
            },
        )
        .expect_err("an encoder does not generate");
    assert!(
        error.to_string().contains("does not generate text"),
        "unexpected error: {error}"
    );
}

#[test]
fn an_override_template_replaces_the_model_one() {
    let Some(path) = model_path() else { return };
    let mut engine = engine(&path, Device::Cpu);
    let context = Json::from(serde_json::json!({"id": "q"}));
    engine
        .set_decision_template(Some("override {{ id }}"))
        .expect("override parses");
    assert_eq!(
        engine.decision_info().expect("info").template,
        TemplateSource::Override
    );
    assert_eq!(
        engine
            .render_named_template(DECISION_TEMPLATE, &context)
            .expect("render"),
        "override q"
    );
    // A template that loses the markers of the options is a layout error, not
    // a wrong answer.
    let info = engine.decision_info().expect("info");
    let request = SystemOneRequest::from_slice(
        br#"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": "y"}}}}"#,
        &info.readout,
    )
    .expect("valid request");
    let error = engine.decide(&info, &request).expect_err("no markers");
    assert!(
        matches!(error, EngineError::Decision(_)),
        "unexpected error: {error}"
    );

    assert!(engine.set_decision_template(Some("{% if %}")).is_err());
    engine
        .set_decision_template(None)
        .expect("back to the model");
    assert_eq!(
        engine.decision_info().expect("info").template,
        TemplateSource::Model
    );
}

#[test]
fn a_prompt_longer_than_a_micro_batch_is_refused_as_invalid() {
    let Some(path) = model_path() else { return };
    let mut engine = Engine::new(
        &path,
        &EngineConfig {
            device: Device::Cpu,
            n_batch: 64,
            n_ubatch: 64,
            n_seq_max: 4,
            ..EngineConfig::default()
        },
    )
    .expect("engine");
    let info = engine.decision_info().expect("info");
    let request = SystemOneRequest::from_slice(
        br#"{"state": "a long state that the encoder cannot read in one micro-batch of sixty-four tokens, repeated: a long state that the encoder cannot read in one micro-batch of sixty-four tokens", "questions": {"q": {"type": "noul", "instructions": "Is it long?"}}}"#,
        &info.readout,
    )
    .expect("valid request");
    let error = engine.decide(&info, &request).expect_err("too long");
    assert!(
        matches!(error, EngineError::InvalidArgument(_)),
        "unexpected error: {error}"
    );
}
