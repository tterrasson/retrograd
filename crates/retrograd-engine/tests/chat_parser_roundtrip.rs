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
///
/// `variables` are the template variables of both renders, the prompt's and the
/// turn's, as a rollout renders both under the same ones.
fn sampled(
    template: &str,
    assistant: &str,
    tools: Option<&str>,
    variables: Option<&str>,
) -> String {
    let prefix = render_chat_template_source(
        template,
        r#"[{"role":"user","content":"go"}]"#,
        tools,
        variables,
        true,
    )
    .expect("render the prompt");

    let plain = render_chat_template_source(
        template,
        &messages_with(&format!(r#""content":"{PLAIN}""#)),
        tools,
        variables,
        false,
    )
    .expect("render a plain assistant turn");
    let after_content = plain
        .rfind(PLAIN)
        .map(|at| &plain[at + PLAIN.len()..])
        .expect("the template renders the assistant content verbatim");

    let whole =
        render_chat_template_source(template, &messages_with(assistant), tools, variables, false)
            .expect("render the assistant turn");
    let body = whole
        .strip_prefix(prefix.as_str())
        .unwrap_or_else(|| panic!("the assistant turn does not extend the prompt:\n{whole}"));
    body.strip_suffix(after_content).unwrap_or(body).to_owned()
}

fn parse(
    template: &str,
    tools: Option<&str>,
    variables: Option<&str>,
    text: &str,
) -> serde_json::Value {
    let parser = tool_call_parser_from_source(template, tools, variables)
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
            None,
        );
        let parsed = parse(&template, Some(TOOLS), None, &text);
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
            None,
        );
        let parsed = parse(&template, Some(TOOLS), None, &text);
        let calls = parsed["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 2, "{name}: parsed {parsed} from {text:?}");
    }
}

#[test]
fn an_answer_that_calls_nothing_stays_content() {
    // The other half of the invariant: a parser that read prose as a call would
    // be as wrong as one that reads a call as prose, and it would end every
    // trajectory on a tool that was never asked for.
    // Sampled rather than written: the answer is whatever the template puts
    // after its generation prompt, and a template whose prompt opens a
    // reasoning block reads an answer that never closed it as reasoning.
    for name in TOOL_TEMPLATES {
        let template = fixture(name);
        let text = sampled(
            &template,
            r#""content":"the exit is to the left""#,
            Some(TOOLS),
            None,
        );
        let parsed = parse(&template, Some(TOOLS), None, &text);
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
    let parsed = parse(&template, None, None, "just an answer");
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
    assert!(tool_call_parser_from_source("{% for %}", None, None).is_err());
    assert!(
        tool_call_parser_from_source(&fixture("lfm2.5-230m"), Some("not JSON"), None).is_err(),
        "an invalid caller-supplied catalog must not look like a parser fallback"
    );
}

/// The templates whose model reasons in a block of its own: `<think>` opened by
/// the model for Qwen3, by the prompt itself for Qwen3.5 under
/// `enable_thinking`.
const THINKING_TEMPLATES: [&str; 2] = ["qwen3-0.6b", "qwen3.5-0.8b"];

const THINKING_ON: &str = r#"{"enable_thinking":true}"#;
const THINKING_OFF: &str = r#"{"enable_thinking":false}"#;

const CALL: &str = r#""tool_calls":[{"type":"function","function":{"name":"move","arguments":{"direction":"left"}}}]"#;

fn calls(parsed: &serde_json::Value) -> Vec<String> {
    parsed["tool_calls"]
        .as_array()
        .expect("tool_calls array")
        .iter()
        .map(|call| call["name"].as_str().expect("name").to_owned())
        .collect()
}

#[test]
fn reasoning_comes_back_apart_from_the_call_and_without_its_tags() {
    // Neither tag is named here: the template writes the block, the parser
    // derived from it takes it out, and what is left is the call.
    for name in THINKING_TEMPLATES {
        let template = fixture(name);
        let text = sampled(
            &template,
            &format!(r#""content":"","reasoning_content":"left is the exit",{CALL}"#),
            Some(TOOLS),
            Some(THINKING_ON),
        );
        let parsed = parse(&template, Some(TOOLS), Some(THINKING_ON), &text);
        assert_eq!(calls(&parsed), ["move"], "{name}: {parsed} from {text:?}");
        assert_eq!(
            parsed["reasoning_content"].as_str().map(str::trim),
            Some("left is the exit"),
            "{name}: {parsed} from {text:?}"
        );
        let content = parsed["content"].as_str().expect("content string");
        assert!(!content.contains("think>"), "{name}: {parsed}");
    }
}

#[test]
fn the_parser_starts_where_the_prompt_rendered_under_the_same_variables_ends() {
    // Qwen3.5 closes an empty reasoning block in the prompt when thinking is
    // off. A parser derived with thinking on assumed the block still open and
    // handed the reader a `<think>` the model never wrote.
    for name in THINKING_TEMPLATES {
        let template = fixture(name);
        let text = sampled(
            &template,
            &format!(r#""content":"",{CALL}"#),
            Some(TOOLS),
            Some(THINKING_OFF),
        );
        let parsed = parse(&template, Some(TOOLS), Some(THINKING_OFF), &text);
        assert_eq!(calls(&parsed), ["move"], "{name}: {parsed} from {text:?}");
        for field in ["content", "reasoning_content"] {
            let value = parsed[field].as_str().unwrap_or_default();
            assert!(!value.contains("think>"), "{name}: {field} of {parsed}");
        }
    }
}

#[test]
fn a_turn_cut_inside_its_reasoning_reads_as_reasoning_and_nothing_else() {
    // What a turn that ran out of tokens while thinking looks like: no call,
    // no answer, only reasoning - which is how a rollout tells "the budget was
    // too small" from "the call format is not the parser's".
    for name in THINKING_TEMPLATES {
        let template = fixture(name);
        let prompt = render_chat_template_source(
            &template,
            r#"[{"role":"user","content":"go"}]"#,
            Some(TOOLS),
            Some(THINKING_ON),
            true,
        )
        .expect("render the prompt");
        // The model opens the block itself unless the prompt already did.
        let text = match prompt.trim_end().ends_with("<think>") {
            true => "left, or maybe".to_owned(),
            false => "<think>\nleft, or maybe".to_owned(),
        };
        let parsed = parse(&template, Some(TOOLS), Some(THINKING_ON), &text);
        assert!(calls(&parsed).is_empty(), "{name}: {parsed}");
        assert_eq!(
            parsed["content"].as_str().map(str::trim),
            Some(""),
            "{name}: {parsed}"
        );
        assert!(
            parsed["reasoning_content"]
                .as_str()
                .is_some_and(|reasoning| reasoning.contains("left, or maybe")),
            "{name}: {parsed}"
        );
    }
}

#[test]
fn variables_are_checked_where_they_are_read() {
    let template = fixture("qwen3.5-0.8b");
    for variables in [
        r#"{"enable_thinking":"false"}"#,
        r#"{"messages":[]}"#,
        "[]",
        "not JSON",
    ] {
        assert!(
            tool_call_parser_from_source(&template, Some(TOOLS), Some(variables)).is_err(),
            "{variables}"
        );
        assert!(
            render_chat_template_source(&template, "[]", None, Some(variables), true).is_err(),
            "{variables}"
        );
    }
}

#[test]
fn an_unset_enable_thinking_is_the_template_s_own_default() {
    // Qwen3 thinks unless told not to, Qwen3.5 the reverse. Rendering an unset
    // variable keeps each one's default, and the parser - which llama.cpp
    // derives with the variable always set - is handed the value that renders
    // the same prompt, so it reads a turn of either without being told.
    for (name, default) in [("qwen3-0.6b", THINKING_ON), ("qwen3.5-0.8b", THINKING_OFF)] {
        let template = fixture(name);
        let render = |variables| {
            render_chat_template_source(
                &template,
                r#"[{"role":"user","content":"go"}]"#,
                None,
                variables,
                true,
            )
            .expect("render the prompt")
        };
        assert_eq!(render(None), render(Some(default)), "{name}");
        assert_eq!(render(Some("{}")), render(Some(default)), "{name}");

        let text = sampled(
            &template,
            &format!(r#""content":"",{CALL}"#),
            Some(TOOLS),
            None,
        );
        let parsed = parse(&template, Some(TOOLS), None, &text);
        assert_eq!(calls(&parsed), ["move"], "{name}: {parsed} from {text:?}");
        assert!(
            !parsed["content"]
                .as_str()
                .unwrap_or_default()
                .contains("think>"),
            "{name}: {parsed}"
        );
    }
}
