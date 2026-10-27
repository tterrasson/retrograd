//! A tool record prepares through the chat templates of the models the fast
//! lane names, and what it trains is what the rollout reads back.
//!
//! Same fixtures as `chat_parser_roundtrip.rs`, and the same reason to be in
//! the fast lane: the template engine and the parser derivation need no model.
//! The tokenizer is a stand-in - one token per special marker the templates
//! write, one per byte for the rest - which is all the framing arithmetic needs.

use retrograd_dataset::chat_template::{
    TemplateMessage, ToolRenderingKind, observation_text, split_turns, template_messages,
};
use retrograd_dataset::{ChatExample, DatasetBackend, ToolConversationRenderer, ToolStream};
use retrograd_engine::{
    parse_assistant_output, render_chat_template_source, tool_call_parser_from_source,
};
use serde_json::json;

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/chat-templates/{name}.jinja",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

/// The templates' control tokens, as the real vocabularies have them: a marker
/// is one token, never its bytes.
const MARKERS: [&str; 16] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|startoftext|>",
    "<|endoftext|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_response_start|>",
    "<|tool_response_end|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
];

/// The markers generation stops on.
const EOG: [&str; 3] = ["<|im_end|>", "<end_of_turn>", "<|endoftext|>"];

const BOS: i32 = -2;
const EOS: i32 = 20_000;

fn marker_id(index: usize) -> i32 {
    10_000 + i32::try_from(index).expect("few markers")
}

struct FixtureBackend {
    template: String,
    /// The template variables every render runs under, as a run's
    /// `template_variables`.
    variables: Option<&'static str>,
}

impl FixtureBackend {
    fn new(name: &str) -> Self {
        Self::with_variables(name, None)
    }

    fn with_variables(name: &str, variables: Option<&'static str>) -> Self {
        Self {
            template: fixture(name),
            variables,
        }
    }

    fn tokenize(text: &str) -> Vec<i32> {
        let mut tokens = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (index, marker) in MARKERS.iter().enumerate() {
                if let Some(after) = rest.strip_prefix(marker) {
                    tokens.push(marker_id(index));
                    rest = after;
                    continue 'outer;
                }
            }
            let byte = rest.as_bytes()[0];
            tokens.push(i32::from(byte));
            let width = rest
                .char_indices()
                .nth(1)
                .map_or(rest.len(), |(width, _)| width);
            // A multi-byte character is its bytes, one token each.
            tokens.extend(
                rest.as_bytes()[1..width]
                    .iter()
                    .map(|&byte| i32::from(byte)),
            );
            rest = &rest[width..];
        }
        tokens
    }
}

impl DatasetBackend for FixtureBackend {
    fn tokenize_text(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        let mut tokens = vec![BOS];
        tokens.extend(Self::tokenize(text));
        Ok(tokens)
    }

    fn eos_token(&self) -> retrograd_core::Result<i32> {
        Ok(EOS)
    }

    fn format_chat(
        &self,
        _messages: &[(&str, &str)],
        _add_assistant: bool,
    ) -> retrograd_core::Result<String> {
        unreachable!("tool records never take the plain path")
    }

    fn tokenize_fragment(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        Ok(Self::tokenize(text))
    }

    fn is_eog_token(&self, token: i32) -> retrograd_core::Result<bool> {
        let marker = MARKERS
            .iter()
            .enumerate()
            .find(|(index, _)| marker_id(*index) == token);
        Ok(token == EOS || marker.is_some_and(|(_, marker)| EOG.contains(marker)))
    }

    fn format_chat_messages(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        add_assistant: bool,
    ) -> retrograd_core::Result<String> {
        render_chat_template_source(
            &self.template,
            messages_json,
            tools_json,
            self.variables,
            add_assistant,
        )
    }

    /// Probed the way the runtime probes it: a catalog whose tool name must
    /// reach the rendered text.
    fn chat_template_supports_tools(&self) -> retrograd_core::Result<bool> {
        let probe = r#"[{"type":"function","function":{"name":"retro_probe_tool_7f3a",
            "description":"","parameters":{"type":"object","properties":{}}}}]"#;
        let rendered = render_chat_template_source(
            &self.template,
            r#"[{"role":"user","content":"x"}]"#,
            Some(probe),
            None,
            true,
        )?;
        Ok(rendered.contains("retro_probe_tool_7f3a"))
    }

    fn tool_call_parser(&self, tools_json: &str) -> retrograd_core::Result<Option<String>> {
        tool_call_parser_from_source(&self.template, Some(tools_json), self.variables).map(Some)
    }

    fn parse_assistant(&self, parser: &str, text: &str) -> retrograd_core::Result<String> {
        parse_assistant_output(parser, text)
    }
}

/// Two calls in one turn, their two results, and the answer.
fn record() -> ChatExample {
    serde_json::from_value(json!({
        "tools": [{"type": "function", "function": {
            "name": "move",
            "description": "move the agent",
            "parameters": {"type": "object",
                           "properties": {"direction": {"type": "string"}},
                           "required": ["direction"]},
        }}],
        "messages": [
            {"role": "system", "content": "You explore a maze."},
            {"role": "user", "content": "Find the exit."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "move", "arguments": {"direction": "left"}}},
                {"id": "call_1", "type": "function",
                 "function": {"name": "move", "arguments": {"direction": "up"}}},
            ]},
            {"role": "tool", "tool_call_id": "call_0", "content": "a wall"},
            {"role": "tool", "tool_call_id": "call_1", "content": "the exit"},
            {"role": "assistant", "content": "The exit is up."},
        ],
    }))
    .expect("a valid record")
}

/// The untrained runs of a stream, in order.
fn framing_runs(stream: &ToolStream) -> Vec<Vec<i32>> {
    let mut runs: Vec<Vec<i32>> = Vec::new();
    let mut previous = true;
    for (&token, &trained) in stream.tokens.iter().zip(&stream.train_mask) {
        if !trained {
            if previous {
                runs.push(Vec::new());
            }
            runs.last_mut().expect("opened above").push(token);
        }
        previous = trained;
    }
    runs
}

/// What an agentic rollout of `example` commits as framing, rebuilt from the
/// shared contract rather than read from the renderer under test: the prompt,
/// then, after each tool turn, the piece the re-render appends, with its
/// opening closer folded into the sampled end of the turn.
///
/// Native rendering only: the catalog goes to the template, not into a system
/// turn this rebuild would have to write too.
fn rollout_framing(backend: &FixtureBackend, example: &ChatExample) -> Vec<Vec<i32>> {
    let tools =
        retrograd_dataset::chat_template::template_tools(&example.tools).expect("a valid catalog");
    let messages = example
        .messages
        .iter()
        .map(|message| match message.role.as_str() {
            "tool" => {
                let id = message
                    .tool_call_id
                    .as_deref()
                    .expect("the record names them");
                TemplateMessage {
                    tool_call_id: Some(id),
                    ..TemplateMessage::text(
                        "tool",
                        observation_text(ToolRenderingKind::Native, id, &message.content, false),
                    )
                }
            }
            role => TemplateMessage::text(role, message.content.as_str()),
        })
        .collect::<Vec<_>>();
    let assistants = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == "assistant")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut runs = Vec::new();
    for (turn, &index) in assistants.iter().enumerate() {
        let (json, sentinels) = template_messages(&messages[..index]).expect("a valid history");
        let rendered = backend
            .format_chat_messages(&json, Some(&tools), true)
            .expect("the template renders");
        let piece = &split_turns(&rendered, &sentinels)
            .expect("sentinels in order")
            .tail;
        let tokens = match turn {
            0 => backend.tokenize_text(piece).expect("infallible stand-in"),
            _ => {
                let tokens = FixtureBackend::tokenize(piece);
                match tokens.first() {
                    Some(&closer) if backend.is_eog_token(closer).expect("infallible stand-in") => {
                        tokens[1..].to_vec()
                    }
                    _ => tokens,
                }
            }
        };
        runs.push(tokens);
    }
    runs
}

const TOOL_TEMPLATES: [&str; 3] = ["lfm2.5-230m", "qwen3-0.6b", "qwen3.5-0.8b"];

#[test]
fn a_structured_record_prepares_through_each_template_that_declares_tools() {
    for name in TOOL_TEMPLATES {
        let backend = FixtureBackend::new(name);
        assert!(backend.chat_template_supports_tools().unwrap(), "{name}");
        // Preparing at all is the read-back: every turn's text went through the
        // parser this template yields and came back as exactly its calls.
        let stream = ToolConversationRenderer::new(&backend)
            .stream(&record())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(stream.train_mask.iter().any(|&trained| trained), "{name}");
        assert!(
            backend
                .is_eog_token(*stream.tokens.last().unwrap())
                .unwrap(),
            "{name}: the stream ends on the closer the policy stops on"
        );
        assert_eq!(
            framing_runs(&stream),
            rollout_framing(&backend, &record()),
            "{name}: the framing is the one a rollout of this conversation commits"
        );
    }
}

#[test]
fn a_template_blind_to_tools_takes_the_prompt_path() {
    let backend = FixtureBackend::new("gemma-3-270m");
    assert!(!backend.chat_template_supports_tools().unwrap());
    // Gemma's template refuses any role but a strict user/model alternation, so
    // a record with a tool turn cannot go through it - nor can a rollout. A
    // record that declares the catalog and answers directly shows the path
    // taken, and its answer is still read back, by the Hermes reader.
    let example: ChatExample = serde_json::from_value(json!({
        "tools": [{"name": "move", "parameters": {"type": "object"}}],
        "messages": [
            {"role": "system", "content": "You explore a maze."},
            {"role": "user", "content": "Which way?"},
            {"role": "assistant", "content": "Up."},
        ],
    }))
    .unwrap();
    let stream = ToolConversationRenderer::new(&backend)
        .stream(&example)
        .unwrap();
    let context = String::from_utf8(
        framing_runs(&stream)[0]
            .iter()
            .filter(|&&token| (0..256).contains(&token))
            .map(|&token| u8::try_from(token).unwrap())
            .collect(),
    )
    .unwrap();
    assert!(
        context.contains("You explore a maze.\n\nAvailable tools (JSON): "),
        "the catalog goes into the system turn: {context}"
    );
}

/// The modes of the templates that reason in a block of their own, each with
/// whether its generation prompt leaves the model room to reason: Qwen3 opens
/// nothing and lets the model open the block, Qwen3.5 under `enable_thinking`
/// opens it itself, and either one closes an empty block in the prompt when
/// thinking is off.
const REASONING_MODES: [(&str, Option<&str>, bool); 9] = [
    ("qwen3-0.6b", None, true),
    ("qwen3-0.6b", THINKING_ON, true),
    ("qwen3-0.6b", THINKING_OFF, false),
    ("qwen3-1.7b", None, true),
    ("qwen3-1.7b", THINKING_ON, true),
    ("qwen3-1.7b", THINKING_OFF, false),
    ("qwen3.5-0.8b", None, false),
    ("qwen3.5-0.8b", THINKING_ON, true),
    ("qwen3.5-0.8b", THINKING_OFF, false),
];

const THINKING_ON: Option<&str> = Some(r#"{"enable_thinking":true}"#);
const THINKING_OFF: Option<&str> = Some(r#"{"enable_thinking":false}"#);

/// [`record`], each assistant turn reasoning.
fn reasoning_record() -> ChatExample {
    let mut example = record();
    example.messages[2].reasoning_content = Some("left looks open".into());
    example.messages[5].reasoning_content = Some("the exit was up".into());
    example.validate().expect("reasoning on assistant turns");
    example
}

fn trained(stream: &ToolStream) -> Vec<i32> {
    stream
        .tokens
        .iter()
        .zip(&stream.train_mask)
        .filter(|(_, trained)| **trained)
        .map(|(&token, _)| token)
        .collect()
}

fn contains(haystack: &[i32], needle: &[i32]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// A record prepares under every mode of a template that reasons, as long as
/// it is one a rollout under that mode could produce: reasoning where the
/// prompt leaves the model room for it, none where the prompt closed the
/// block. The framing is the one a rollout commits either way - including
/// where the template's history leaves out the empty block its prompt wrote.
#[test]
fn a_record_prepares_through_each_mode_of_a_template_that_reasons() {
    for (name, variables, reasons) in REASONING_MODES {
        let backend = FixtureBackend::with_variables(name, variables);
        let example = match reasons {
            true => reasoning_record(),
            false => record(),
        };
        let case = format!("{name} {variables:?}");
        let stream = ToolConversationRenderer::new(&backend)
            .stream(&example)
            .unwrap_or_else(|error| panic!("{case}: {error}"));
        assert_eq!(
            framing_runs(&stream),
            rollout_framing(&backend, &example),
            "{case}: the framing is the one a rollout of this conversation commits"
        );
        if reasons {
            let trained = trained(&stream);
            for reasoning in ["left looks open", "the exit was up"] {
                assert!(
                    contains(&trained, &FixtureBackend::tokenize(reasoning)),
                    "{case}: {reasoning:?} is not among the trained tokens"
                );
            }
            // Nothing the template writes for a turn without reasoning stands
            // before one that has some.
            let empty = FixtureBackend::tokenize("<think>\n\n</think>");
            assert!(!contains(&stream.tokens, &empty), "{case}");
        }
    }
}

/// The text a reasoning turn is trained as is the one a model reasoning under
/// that mode samples: from the tag the prompt left it to open - none when the
/// prompt opened the block - to the end of its content.
#[test]
fn a_reasoning_turn_is_trained_from_where_its_prompt_stops() {
    for (name, variables, expected) in [
        (
            "qwen3-0.6b",
            None,
            "<think>\nthe exit was up\n</think>\n\nThe exit is up.",
        ),
        (
            "qwen3.5-0.8b",
            THINKING_ON,
            "the exit was up\n</think>\n\nThe exit is up.",
        ),
    ] {
        let backend = FixtureBackend::with_variables(name, variables);
        let stream = ToolConversationRenderer::new(&backend)
            .stream(&reasoning_record())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut last_turn = FixtureBackend::tokenize(expected);
        last_turn.extend(FixtureBackend::tokenize("<|im_end|>"));
        assert!(trained(&stream).ends_with(&last_turn), "{name}");
    }
}

/// Reasoning under a mode whose prompt closed the block is a turn no rollout
/// could sample - the model would reason after its own empty block - and the
/// record says so instead of training it.
#[test]
fn reasoning_where_the_prompt_closed_the_block_is_refused() {
    for (name, variables, reasons) in REASONING_MODES {
        if reasons {
            continue;
        }
        let backend = FixtureBackend::with_variables(name, variables);
        let error = ToolConversationRenderer::new(&backend)
            .stream(&reasoning_record())
            .expect_err("a turn no rollout under this mode samples");
        assert!(
            error.to_string().contains("reasoning"),
            "{name} {variables:?}: {error}"
        );
    }
}

/// LFM2.5 reads a turn's reasoning from a field of its own and never from
/// `reasoning_content`: a record that reasons would be trained without its
/// reasoning, so it is refused.
#[test]
fn a_template_that_does_not_write_the_reasoning_refuses_it() {
    let backend = FixtureBackend::new("lfm2.5-230m");
    let error = ToolConversationRenderer::new(&backend)
        .stream(&reasoning_record())
        .expect_err("the reasoning would be lost");
    assert!(error.to_string().contains("reasoning"), "{error}");
}

#[test]
fn reasoning_belongs_to_a_structured_assistant_turn_only() {
    let mut on_user = record();
    on_user.messages[1].reasoning_content = Some("x".into());
    let error = on_user.validate().expect_err("reasoning on a user turn");
    assert!(error.to_string().contains("reasoning_content"), "{error}");

    let mut on_raw = record();
    on_raw.messages[5].raw = true;
    on_raw.messages[5].reasoning_content = Some("x".into());
    let error = on_raw.validate().expect_err("reasoning beside a raw turn");
    assert!(error.to_string().contains("reasoning_content"), "{error}");
}
