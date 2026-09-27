//! A tool record prepares through the chat templates of the models the fast
//! lane names, and what it trains is what the rollout reads back.
//!
//! Same fixtures as `chat_parser_roundtrip.rs`, and the same reason to be in
//! the fast lane: the template engine and the parser derivation need no model.
//! The tokenizer is a stand-in - one token per special marker the templates
//! write, one per byte for the rest - which is all the framing arithmetic needs.

use retrograd_dataset::chat_template::{
    TemplateMessage, ToolRenderingKind, observation_text, split_assistant_spans, template_messages,
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
}

impl FixtureBackend {
    fn new(name: &str) -> Self {
        Self {
            template: fixture(name),
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
        render_chat_template_source(&self.template, messages_json, tools_json, add_assistant)
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
            true,
        )?;
        Ok(rendered.contains("retro_probe_tool_7f3a"))
    }

    fn tool_call_parser(&self, tools_json: &str) -> retrograd_core::Result<Option<String>> {
        tool_call_parser_from_source(&self.template, Some(tools_json)).map(Some)
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
    let tools = retrograd_dataset::chat_template::template_tools(&example.tools).unwrap();
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
        let (json, sentinels) = template_messages(&messages[..index]).unwrap();
        let rendered = backend
            .format_chat_messages(&json, Some(&tools), true)
            .unwrap();
        let pieces = split_assistant_spans(&rendered, &sentinels).unwrap();
        let piece = pieces.last().expect("one piece per gap");
        let tokens = match turn {
            0 => backend.tokenize_text(piece).unwrap(),
            _ => {
                let tokens = FixtureBackend::tokenize(piece);
                match tokens.first() {
                    Some(&closer) if backend.is_eog_token(closer).unwrap() => tokens[1..].to_vec(),
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
