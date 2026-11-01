//! The decision crate against the golden fixtures of `tests/fixtures/decision`:
//! prompts, tokens after the layout, marker positions, and the answers computed
//! from the recorded raw scores. The files were recorded from llama-server, so
//! agreement here is agreement with it, for the three models and both
//! backends.

use std::path::{Path, PathBuf};

use retrograd_decision::answer::{Calibration, format_answer};
use retrograd_decision::json::Object;
use retrograd_decision::layout::{self, Budget};
use retrograd_decision::readout::Readout;
use retrograd_decision::request::SystemOneRequest;
use retrograd_decision::template::{self, PromptInput};
use retrograd_decision::{Json, Token};

/// How far a probability may be from the recorded one. The scores are the
/// recorded ones and the arithmetic the server's, so only the last digits of the
/// JSON printing can differ.
const TOLERANCE: f64 = 1e-12;

/// A model of the fixtures: what its vocabulary and its metadata give.
struct Model {
    name: &'static str,
    cls: Token,
    marker: Token,
    sep: Token,
    marker_text: &'static str,
    /// `<arch>.decision.temperature.*` as the text of the metadata.
    temperatures: &'static [(&'static str, &'static str)],
}

/// Laya-Q8_0 and the tiny test model share the temperatures and the tokenizer
/// of ModernBERT.
const LAYA_TEMPERATURES: &[(&str, &str)] = &[
    ("temperature.choice", "1.636903"),
    ("temperature.score", "1.251430"),
    ("temperature.noul", "1.983400"),
    ("temperature.choice.3_5", "1.760152"),
    ("temperature.choice.6_10", "1.000016"),
    ("temperature.score.3_5", "1.251430"),
    ("temperature.noul.2", "1.983400"),
    ("temperature.choice.11", "0.100583"),
    ("temperature.choice.2", "1.906356"),
];

const MODELS: [Model; 3] = [
    Model {
        name: "tinylaya",
        cls: 50281,
        marker: 50284,
        sep: 50282,
        marker_text: "[MASK]",
        temperatures: LAYA_TEMPERATURES,
    },
    Model {
        name: "laya",
        cls: 50281,
        marker: 50284,
        sep: 50282,
        marker_text: "[MASK]",
        temperatures: LAYA_TEMPERATURES,
    },
    Model {
        name: "julia-1",
        cls: 2,
        marker: 4,
        sep: 1,
        marker_text: "<mask>",
        temperatures: &[],
    },
];

const BACKENDS: [&str; 2] = ["cpu", "mtl"];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/decision")
}

fn load(file: &str) -> Json {
    let path = fixtures().join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn golden(model: &Model, backend: &str) -> Json {
    load(&format!("golden-{}.{backend}.json", model.name))
}

fn readout(model: &Model) -> Readout {
    Readout::MarkerColumn {
        marker: model.marker,
        sep: model.sep,
        marker_text: model.marker_text.to_owned(),
    }
}

/// The body of the request named `name` in the frozen set.
fn body(name: &str) -> Json {
    load("requests.json")["requests"]
        .as_array()
        .expect("requests")
        .iter()
        .find(|request| request["name"] == name)
        .unwrap_or_else(|| panic!("no request {name}"))["body"]
        .clone()
}

fn array(value: &Json) -> &[Json] {
    value.as_array().expect("array")
}

fn tokens(value: &Json) -> Vec<Token> {
    array(value)
        .iter()
        .map(|t| Token::try_from(t.as_i64().expect("token")).expect("token id"))
        .collect()
}

fn positions(value: &Json) -> Vec<usize> {
    array(value)
        .iter()
        .map(|p| usize::try_from(p.as_u64().expect("position")).expect("position"))
        .collect()
}

/// A recorded score. The files hold `f32` values printed with 15 significant
/// digits, which is enough to find the `f32` again: the narrowing is the
/// rounding to that value, and the assertion proves the print was of an `f32`.
fn score(value: &Json) -> f32 {
    let wide = value.as_f64().expect("score");
    let narrow = wide as f32;
    assert!(
        (f64::from(narrow) - wide).abs() <= 1e-14 * wide.abs(),
        "recorded score is not an f32: {wide}"
    );
    narrow
}

#[test]
fn layout_reproduces_the_golden_tokens_and_markers() {
    for model in &MODELS {
        let readout = readout(model);
        for backend in BACKENDS {
            let file = golden(model, backend);
            let mut cut = 0;
            for entry in array(&file["requests"]) {
                let name = entry["name"].as_str().expect("name");
                let budget = Budget::new(
                    usize::try_from(entry["max_head_tokens"].as_u64().expect("budget"))
                        .expect("budget"),
                )
                .expect("budget");
                let request = SystemOneRequest::parse(&body(name), &readout).expect("request");

                // The answers come in the order of the request.
                let golden_ids: Vec<_> = array(&entry["questions"])
                    .iter()
                    .map(|q| q["id"].as_str().expect("id"))
                    .collect();
                let request_ids: Vec<_> = request.questions.keys().map(String::as_str).collect();
                assert_eq!(request_ids, golden_ids, "{} {backend} {name}", model.name);

                let mut input_tokens = 0;
                for recorded in array(&entry["questions"]) {
                    let id = recorded["id"].as_str().expect("id");
                    let context = format!("{} {backend} {name} {id}", model.name);
                    let question = &request.questions[id];
                    let rendered = tokens(&recorded["rendered_tokens"]);
                    let laid_out =
                        layout::arrange(&readout, &rendered, question, &budget).expect(&context);

                    assert_eq!(
                        laid_out.tokens,
                        tokens(&recorded["tokens"]),
                        "{context}: tokens"
                    );
                    assert_eq!(
                        laid_out.markers,
                        positions(&recorded["markers"]),
                        "{context}: markers"
                    );
                    assert_eq!(laid_out.markers.len(), question.options.len(), "{context}");
                    assert_eq!(laid_out.pointer, None);
                    assert_eq!(
                        recorded["column"].as_u64().expect("column"),
                        u64::try_from(question.kind.column()).expect("column"),
                        "{context}: column"
                    );
                    // The recorded layout has the shape the model was trained on.
                    assert_eq!(laid_out.tokens[0], model.cls, "{context}");
                    assert_eq!(laid_out.tokens.last(), Some(&model.sep), "{context}");
                    assert!(
                        laid_out
                            .markers
                            .iter()
                            .all(|&m| laid_out.tokens[m] == model.marker),
                        "{context}"
                    );

                    if laid_out.tokens != rendered {
                        cut += 1;
                    }
                    input_tokens += laid_out.tokens.len();
                }
                // `usage.input_tokens` is the sum over the questions.
                assert_eq!(
                    entry["response"]["usage"]["input_tokens"]
                        .as_u64()
                        .map(usize::try_from),
                    Some(Ok(input_tokens)),
                    "{} {backend} {name}",
                    model.name
                );
            }
            // The files exercise the cuts: the 20 long options and the long
            // question with the budget of the model.
            assert_eq!(cut, 2, "{} {backend}: questions that were cut", model.name);
        }
    }
}

/// The same long question with the budget of the model and with 512 keeps
/// different tokens; the layout follows the budget it is given.
#[test]
fn the_head_budget_changes_the_layout_of_a_long_question() {
    for model in &MODELS {
        let readout = readout(model);
        let file = golden(model, "cpu");
        let find = |name: &str| {
            array(&file["requests"])
                .iter()
                .find(|entry| entry["name"] == name)
                .expect("entry")["questions"][0]
                .clone()
        };
        let at_model = find("long_question_model_budget");
        let at_512 = find("long_question_budget_512");
        // The two requests render the same prompt.
        assert_eq!(
            at_model["rendered_tokens"], at_512["rendered_tokens"],
            "{}",
            model.name
        );

        let request = SystemOneRequest::parse(&body("long_question_model_budget"), &readout)
            .expect("request");
        let question = &request.questions[0];
        let rendered = tokens(&at_model["rendered_tokens"]);
        let model_budget = Budget::new(
            usize::try_from(
                golden(model, "cpu")["model"]["max_head_tokens"]
                    .as_u64()
                    .expect("budget"),
            )
            .expect("budget"),
        )
        .expect("budget");
        let wide = Budget::new(512).expect("budget");

        let narrow = layout::arrange(&readout, &rendered, question, &model_budget).expect("layout");
        let wide = layout::arrange(&readout, &rendered, question, &wide).expect("layout");
        assert_eq!(narrow.tokens, tokens(&at_model["tokens"]), "{}", model.name);
        assert_eq!(wide.tokens, tokens(&at_512["tokens"]), "{}", model.name);
        assert!(narrow.tokens.len() < wide.tokens.len(), "{}", model.name);
        // The state follows the question in both, and none of it is cut.
        assert_eq!(wide.tokens, rendered, "{}", model.name);
    }
}

/// Compares an answer with a recorded one: the same keys in the same order, the
/// same strings, the numbers within [`TOLERANCE`].
fn assert_same(actual: &Json, recorded: &Json, context: &str) {
    match (actual, recorded) {
        (Json::Number(a), Json::Number(b)) => {
            let (a, b) = (a.as_f64().expect("f64"), b.as_f64().expect("f64"));
            assert!((a - b).abs() <= TOLERANCE, "{context}: {a} != {b}");
        }
        (Json::Object(a), Json::Object(b)) => {
            let keys = |o: &Object| o.keys().cloned().collect::<Vec<_>>();
            assert_eq!(keys(a), keys(b), "{context}: keys");
            for (key, value) in a {
                assert_same(value, &b[key], &format!("{context}.{key}"));
            }
        }
        (a, b) => assert_eq!(a, b, "{context}"),
    }
}

#[test]
fn answers_reproduce_the_golden_json_from_the_golden_scores() {
    for model in &MODELS {
        let readout = readout(model);
        let calibration =
            Calibration::from_metadata(model.temperatures.iter().copied()).expect("calibration");
        for backend in BACKENDS {
            let file = golden(model, backend);
            for entry in array(&file["requests"]) {
                let name = entry["name"].as_str().expect("name");
                let request = SystemOneRequest::parse(&body(name), &readout).expect("request");
                for recorded in array(&entry["questions"]) {
                    let id = recorded["id"].as_str().expect("id");
                    let context = format!("{} {backend} {name} {id}", model.name);
                    let scores = vec![
                        array(&recorded["scores"])
                            .iter()
                            .map(score)
                            .collect::<Vec<_>>(),
                    ];
                    let answer = format_answer(
                        &readout,
                        &calibration,
                        &request.questions[id],
                        &scores,
                        false,
                    )
                    .expect(&context);
                    // Through text: the key order of the answer is part of what is compared.
                    let actual: Json =
                        serde_json::from_str(&serde_json::to_string(&answer).expect("json"))
                            .expect("json");
                    assert_same(&actual, &entry["response"]["answers"][id], &context);
                }
            }
        }
    }
}

/// `json.dumps` as the Jinja `tojson` of the templates writes it: `", "` and
/// `": "` separators, keys in order.
fn tojson(value: &Json) -> String {
    match value {
        Json::Array(items) => format!(
            "[{}]",
            items.iter().map(tojson).collect::<Vec<_>>().join(", ")
        ),
        Json::Object(members) => format!(
            "{{{}}}",
            members
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("key"),
                    tojson(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// `x if x is string else x | tojson`
fn text(value: &Json) -> String {
    value.as_str().map_or_else(|| tojson(value), str::to_owned)
}

fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Number(n) => n.as_f64() != Some(0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

/// The `systemone` template of `Laya-Q8_0.gguf`, written out by hand.
fn render_laya(context: &Json) -> Result<String, String> {
    let kind = context["type"].as_str().expect("type");
    let mut out = format!(
        "[CLS]{kind} question: {}[SEP]",
        text(&context["instructions"])
    );
    for option in array(&context["options"]) {
        let key = option["key"].as_str().expect("key");
        let description = &option["description"];
        out.push_str("[MASK] ");
        match kind {
            "choice" => {
                out.push_str(key);
                if truthy(description) {
                    out.push_str(&format!(": {}", text(description)));
                }
            }
            "score" => out.push_str(&format!("level {key}: {}", text(description))),
            _ if truthy(description) => out.push_str(&format!("{key}: {}", text(description))),
            _ if key == "true" => out.push_str(&format!("{key}: yes, the statement holds")),
            _ => out.push_str(&format!("{key}: no, the statement does not hold")),
        }
    }
    out.push_str(&format!("[SEP]{}[SEP]", text(&context["state"])));
    Ok(out)
}

/// The `systemone` template of `Julia-1-Q8_0.gguf`, written out by hand.
fn render_julia(context: &Json) -> Result<String, String> {
    let kind = context["type"].as_str().expect("type");
    let mut out = format!(
        "<bos>{kind} question: {}<eos>",
        text(&context["instructions"])
    );
    for option in array(&context["options"]) {
        out.push_str("<mask> ");
        let description = &option["description"];
        out.push_str(&if truthy(description) {
            text(description)
        } else {
            option["key"].as_str().expect("key").to_owned()
        });
    }
    out.push_str(&format!("<eos>{}<eos>", text(&context["state"])));
    Ok(out)
}

/// The frozen context, sanitised and rendered by the templates of the models,
/// is the prompt the server rendered: the order of the keys of a state, the
/// order of the options, the replacement of the marker text.
#[test]
fn context_renders_the_golden_prompts() {
    for model in &MODELS {
        let readout = readout(model);
        let render: fn(&Json) -> Result<String, String> = if model.name == "julia-1" {
            render_julia
        } else {
            render_laya
        };
        let file = golden(model, "cpu");
        for entry in array(&file["requests"]) {
            let name = entry["name"].as_str().expect("name");
            let request = SystemOneRequest::parse(&body(name), &readout).expect("request");
            for recorded in array(&entry["questions"]) {
                let id = recorded["id"].as_str().expect("id");
                let input = PromptInput {
                    state: &request.state,
                    question_id: id,
                    question: &request.questions[id],
                    variant: 0,
                    n_images: 0,
                    media_marker: "<__media__>",
                };
                let prompt = template::render(&readout, &input, &render).expect("rendered");
                assert_eq!(
                    prompt,
                    recorded["prompt"].as_str().expect("prompt"),
                    "{} {name} {id}",
                    model.name
                );
            }
        }
    }
}
