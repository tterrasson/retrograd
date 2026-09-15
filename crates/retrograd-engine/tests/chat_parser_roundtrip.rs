//! What a chat template writes, the parser derived from that same template
//! reads back.
//!
//! The rollout tests fabricate their generations in `<tool_call>` literals,
//! so the pair (native rendering, `<tool_call>` parser) passed for every model,
//! including the ones whose template writes something else entirely. A
//! round-trip assumes no format at all - it renders a structured call through a
//! real template, hands the rendered text to the parser that template yields,
//! and demands the call back.
//!
//! Fixtures are the templates of the models the fast lane can afford to name but
//! not to load: `tokenizer.chat_template`, extracted verbatim from the GGUFs.
//! Neither half needs a model, so this stays in `scripts/test-fast-rust.sh`.

use retrograd_engine::{
    parse_assistant_output, render_chat_template_source, tool_call_parser_from_source,
};

const TOOLS: &str = r#"[{"type":"function","function":{"name":"move",
    "description":"move the agent",
    "parameters":{"type":"object","properties":{"direction":{"type":"string"}},
    "required":["direction"]}}}]"#;

/// Content of the plain assistant turn used to isolate the template's turn
/// closer. Templates must not trim, split, or re-encode it.
const PLAIN: &str = "zqxplain";

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/chat-templates/{name}.jinja",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

fn messages_with(assistant: &str) -> String {
    format!(r#"[{{"role":"user","content":"go"}},{{"role":"assistant",{assistant}}}]"#)
}

/// The text a model would have sampled for that assistant turn.
///
/// Everything the template writes after the generation prompt, minus the closer
/// it appends once the turn is over - which a generation stops before, since it
/// is what makes the runtime stop. The closer is not guessed: it is read off a
/// plain-content turn, as whatever the template put after the content it was
/// given.
fn sampled(template: &str, assistant: &str, tools: Option<&str>) -> String {
    let prefix =
        render_chat_template_source(template, r#"[{"role":"user","content":"go"}]"#, tools, true)
            .expect("render the prompt");

    let plain = render_chat_template_source(
        template,
        &messages_with(&format!(r#""content":"{PLAIN}""#)),
        tools,
        false,
    )
    .expect("render a plain assistant turn");
    let after_content = plain
        .rfind(PLAIN)
        .map(|at| &plain[at + PLAIN.len()..])
        .expect("the template renders the assistant content verbatim");

    let whole = render_chat_template_source(template, &messages_with(assistant), tools, false)
        .expect("render the assistant turn");
    let body = whole
        .strip_prefix(prefix.as_str())
        .unwrap_or_else(|| panic!("the assistant turn does not extend the prompt:\n{whole}"));
    body.strip_suffix(after_content).unwrap_or(body).to_owned()
}

fn parse(template: &str, tools: Option<&str>, text: &str) -> serde_json::Value {
    let parser = tool_call_parser_from_source(template, tools)
        .unwrap_or_else(|error| panic!("derive a parser: {error}"));
    let document = parse_assistant_output(&parser, text)
        .unwrap_or_else(|error| panic!("parse {text:?}: {error}"));
    serde_json::from_str(&document).expect("the runtime emits a JSON object")
}

/// The templates whose model declares tools. Each one writes the call in its own
/// format - `<|tool_call_start|>[move(direction='left')]<|tool_call_end|>` for
/// LFM2, `<tool_call>{…}</tool_call>` for Qwen3, nested `<function=>` tags for
/// Qwen3.5 - and the test names none of them.
const TOOL_TEMPLATES: [&str; 3] = ["lfm2.5-230m", "qwen3-0.6b", "qwen3.5-0.8b"];

#[test]
fn a_rendered_tool_call_comes_back_as_the_same_call() {
    for name in TOOL_TEMPLATES {
        let template = fixture(name);
        let text = sampled(
            &template,
            r#""content":"","tool_calls":[{"type":"function","function":{"name":"move","arguments":{"direction":"left"}}}]"#,
            Some(TOOLS),
        );
        let parsed = parse(&template, Some(TOOLS), &text);
        let calls = parsed["tool_calls"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no tool_calls array in {parsed}"));
        assert_eq!(calls.len(), 1, "{name}: parsed {parsed} from {text:?}");
        assert_eq!(calls[0]["name"], "move", "{name}: from {text:?}");
        let arguments: serde_json::Value =
            serde_json::from_str(calls[0]["arguments"].as_str().expect("arguments string"))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            arguments,
            serde_json::json!({"direction": "left"}),
            "{name}: from {text:?}"
        );
    }
}

#[test]
fn two_rendered_calls_come_back_as_two() {
    for name in TOOL_TEMPLATES {
        let template = fixture(name);
        let text = sampled(
            &template,
            r#""content":"","tool_calls":[
                {"type":"function","function":{"name":"move","arguments":{"direction":"left"}}},
                {"type":"function","function":{"name":"move","arguments":{"direction":"up"}}}]"#,
            Some(TOOLS),
        );
        let parsed = parse(&template, Some(TOOLS), &text);
        let calls = parsed["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 2, "{name}: parsed {parsed} from {text:?}");
    }
}

#[test]
fn an_answer_that_calls_nothing_stays_content() {
    // The other half of the invariant: a parser that read prose as a call would
    // be as wrong as one that reads a call as prose, and it would end every
    // trajectory on a tool that was never asked for.
    for name in TOOL_TEMPLATES {
        let template = fixture(name);
        let parsed = parse(&template, Some(TOOLS), "the exit is to the left");
        assert!(
            parsed["tool_calls"].as_array().expect("array").is_empty(),
            "{name}: {parsed}"
        );
        assert!(
            parsed["content"]
                .as_str()
                .expect("content string")
                .contains("the exit is to the left"),
            "{name}: {parsed}"
        );
    }
}

#[test]
fn a_template_blind_to_tools_still_yields_a_content_parser() {
    // Gemma's template ignores `tools` entirely. Nothing to read back, but the
    // derivation must not fail either: the rollout falls back on the
    // prompt-described convention, and it decides that from
    // `chat_template_supports_tools`, not from an error here.
    let template = fixture("gemma-3-270m");
    let parsed = parse(&template, None, "just an answer");
    assert!(parsed["tool_calls"].as_array().expect("array").is_empty());
    assert!(
        parsed["content"]
            .as_str()
            .expect("content string")
            .contains("just an answer"),
        "{parsed}"
    );
}

#[test]
fn a_missing_or_broken_parser_blob_is_an_error_and_not_a_silence() {
    assert!(parse_assistant_output("", "anything").is_err());
    assert!(parse_assistant_output("{\"parser\":42}", "anything").is_err());
    assert!(tool_call_parser_from_source("{% for %}", None).is_err());
    assert!(
        tool_call_parser_from_source(&fixture("lfm2.5-230m"), Some("not JSON")).is_err(),
        "an invalid caller-supplied catalog must not look like a parser fallback"
    );
}
