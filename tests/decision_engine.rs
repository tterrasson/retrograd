//! Decision models on the engine, end to end, against the golden fixtures of
//! `tests/fixtures/decision`, recorded from llama-server's decision code on
//! the same model and backend.
//!
//! The engine renders the `systemone` template the GGUF carries, tokenizes
//! with the special token texts parsed and none added, lays the tokens out,
//! evaluates every question of a request in one call (one forward pass per
//! question, or several packed together on request) and turns the scores into
//! answers.
//! What has to hold: the prompt and its tokens are the recorded ones exactly,
//! the token count of the response is exact, and the probabilities agree with
//! the recorded answers of the same backend.
//!
//! The model is the small `laya` test model (`TINYLAYA_FIXTURE.toml`); the
//! test skips when it is absent, unless `RETRO_REQUIRE_CPU_FIXTURE` is set.

use std::path::{Path, PathBuf};

use retrograd::decision::Json;
use retrograd::decision::readout::Readout;
use retrograd::decision::request::SystemOneRequest;
use retrograd::decision::template::{self, PromptInput};
use retrograd::trainer::{DECISION_TEMPLATE, TemplateSource, decision_metadata};
use retrograd::{Device, Engine, EngineConfig, EngineError, SamplingParams};
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

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture is JSON")
}

fn engine(path: &Path, device: Device) -> Engine {
    Engine::new(
        path,
        &EngineConfig {
            device,
            ..EngineConfig::default()
        },
    )
    .expect("engine loads the decision model")
}

fn token_vec(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("token array")
        .iter()
        .map(|token| i32::try_from(token.as_i64().expect("token id")).expect("token fits i32"))
        .collect()
}

/// Every probability (and `noul`, `score`) of `actual` against `expected`,
/// the keys in the same order. Returns the largest difference.
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
        for field in ["noul", "score"] {
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

/// Replays every recorded request; `pack` is how many questions one forward
/// pass may evaluate. Returns the largest probability difference and the
/// questions whose choice is not the recorded one.
fn replay(
    model: &str,
    path: &Path,
    device: Device,
    backend: &str,
    pack: usize,
) -> (f64, Vec<String>) {
    let golden = read_json(&fixtures().join(format!("golden-{model}.{backend}.json")));
    let requests = read_json(&fixtures().join("requests.json"));
    // The bodies again, keys in the order written: a `Value` sorts them, and
    // the order of the questions is the order of the answers.
    let ordered: Json = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("requests.json")).expect("requests readable"),
    )
    .expect("requests are JSON");
    let mut engine = engine(path, device);
    let model_info = engine
        .decision_info()
        .expect("decision head")
        .with_packing(pack);
    assert!(matches!(model_info.readout, Readout::MarkerColumn { .. }));
    assert_eq!(
        model_info.model_max_head_tokens,
        golden["model"]["max_head_tokens"]
            .as_u64()
            .map(|n| n as usize)
    );

    let mut largest = 0.0_f64;
    let mut choices_changed = Vec::new();
    let bodies = ordered["requests"].as_array().expect("requests");
    for ((entry, recorded), ordered_entry) in requests["requests"]
        .as_array()
        .expect("requests")
        .iter()
        .zip(golden["requests"].as_array().expect("golden requests"))
        .zip(bodies)
    {
        let name = entry["name"].as_str().expect("name");
        assert_eq!(recorded["name"], entry["name"]);
        let info = match entry.get("max_head_tokens").and_then(Value::as_u64) {
            Some(budget) => model_info
                .clone()
                .with_max_head_tokens(usize::try_from(budget).expect("budget fits usize"))
                .expect("budget override"),
            None => model_info.clone(),
        };
        let body = ordered_entry["body"].to_string().into_bytes();
        let request = SystemOneRequest::from_slice(&body, &info.readout).expect("valid request");

        // The prompt and its tokens, through the engine's renderer and
        // tokenizer, are the recorded ones.
        for (trace, (id, question)) in recorded["questions"]
            .as_array()
            .expect("question traces")
            .iter()
            .zip(&request.questions)
        {
            assert_eq!(
                trace["id"].as_str(),
                Some(id.as_str()),
                "{name}: question order"
            );
            let input = PromptInput {
                state: &request.state,
                question_id: id,
                question,
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
                "{name}/{id}: prompt"
            );
            let tokens = engine
                .tokenize_with(&prompt, false, true)
                .expect("tokenize");
            assert_eq!(
                tokens,
                token_vec(&trace["rendered_tokens"]),
                "{name}/{id}: tokens"
            );
        }

        let decision = engine.decide(&info, &request).expect("decide");
        assert_eq!(
            Some(decision.input_tokens as u64),
            recorded["response"]["usage"]["input_tokens"].as_u64(),
            "{name}: input tokens"
        );
        let answers = serde_json::to_value(&decision.answers).expect("answers serialize");
        let expected = &recorded["response"]["answers"];
        for (id, want) in expected.as_object().expect("answers") {
            if let Some(choice) = want.get("choice")
                && choice != &answers[id]["choice"]
            {
                choices_changed.push(format!("{name}/{id}"));
            }
        }
        largest = largest.max(compare_answers(name, expected, &answers));
    }
    eprintln!(
        "{model} on {backend}, {pack} question(s) per pass: \
         largest probability difference {largest:e}, choices changed: {choices_changed:?}"
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

#[test]
fn engine_reproduces_the_recorded_answers_on_cpu() {
    let Some(path) = model_path() else { return };
    let (largest, changed) = replay("tinylaya", &path, Device::Cpu, "cpu", 8);
    assert!(changed.is_empty(), "cpu: choices changed: {changed:?}");
    assert!(
        largest <= TOLERANCE,
        "cpu: probabilities differ by {largest:e}"
    );
    let (largest, changed) = replay("tinylaya", &path, Device::Cpu, "cpu", 1);
    assert!(changed.is_empty(), "cpu: choices changed: {changed:?}");
    assert!(
        largest <= EXACT,
        "cpu: one question per pass differs by {largest:e}"
    );
}

#[test]
fn engine_reproduces_the_recorded_answers_on_metal() {
    let Some(path) = model_path() else { return };
    if !metal_available() {
        return;
    }
    let (largest, changed) = replay("tinylaya", &path, Device::Gpu, "mtl", 8);
    assert!(changed.is_empty(), "mtl: choices changed: {changed:?}");
    assert!(
        largest <= TOLERANCE,
        "mtl: probabilities differ by {largest:e}"
    );
    let (largest, changed) = replay("tinylaya", &path, Device::Gpu, "mtl", 1);
    assert!(changed.is_empty(), "mtl: choices changed: {changed:?}");
    assert!(
        largest <= EXACT,
        "mtl: one question per pass differs by {largest:e}"
    );
}

/// The real models, opt-in as for the recording (`RETRO_DECISION_JULIA`,
/// `RETRO_DECISION_LAYA`). One question per pass is exact on either backend;
/// packed passes are printed, not asserted: how far packing moves the answers
/// depends on the model (Julia-1 moves by 2.6e-1 on the CPU).
#[test]
fn real_models_reproduce_the_recorded_answers_when_present() {
    for (model, variable) in [
        ("julia-1", "RETRO_DECISION_JULIA"),
        ("laya", "RETRO_DECISION_LAYA"),
    ] {
        let Some(path) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("skipping {model}: {variable} is not set");
            continue;
        };
        replay(model, &path, Device::Cpu, "cpu", 8);
        let (largest, changed) = replay(model, &path, Device::Cpu, "cpu", 1);
        assert!(changed.is_empty(), "{model}: choices changed: {changed:?}");
        assert!(
            largest <= EXACT,
            "{model}: one question per pass differs by {largest:e}"
        );
        if metal_available() {
            replay(model, &path, Device::Gpu, "mtl", 8);
            let (largest, changed) = replay(model, &path, Device::Gpu, "mtl", 1);
            assert!(changed.is_empty(), "{model}: choices changed: {changed:?}");
            assert!(
                largest <= EXACT,
                "{model}: one question per pass differs by {largest:e}"
            );
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
