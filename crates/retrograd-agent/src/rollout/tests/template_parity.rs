//! A rollout through a real chat template, exported and prepared back for SFT,
//! is the stream it was collected as - for every mode of the templates that
//! reason in a block of their own.
//!
//! [`sft_parity`](super::sft_parity) pins the contract on a template small
//! enough to predict by hand; this one runs it on the fixtures the fast lane
//! names, where the template decides what a turn's reasoning block looks like
//! and where it goes. No generation is written by hand: each is the text the
//! template itself writes for the turn after its own generation prompt.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use retrograd_core::SamplingParams;
use retrograd_dataset::chat_template::split_turns;
use retrograd_dataset::{DatasetBackend, ToolConversationRenderer, ToolStream};
use retrograd_engine::{
    parse_assistant_output, render_chat_template_source, tool_call_parser_from_source,
};
use serde_json::{Value, json};

use super::*;
use crate::chat::{template_messages, template_tools};
use crate::export::{AssistantForm, to_chat_example};
use crate::policy::{Policy, PolicyGeneration, RenderedFraming};
use crate::template_parser::TemplateToolCallParser;
use crate::tools::{ToolCallParser, ToolSpec};
use crate::trajectory::{Message, Trajectory};

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/../retrograd-engine/tests/fixtures/chat-templates/{name}.jinja",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

/// The templates' control tokens, one token each as in the real vocabularies.
const MARKERS: [&str; 11] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|startoftext|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
];
const EOG: &str = "<|im_end|>";
const BOS: i32 = -2;

fn marker_id(index: usize) -> i32 {
    10_000 + i32::try_from(index).expect("few markers")
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
        let width = rest.chars().next().map_or(1, char::len_utf8);
        tokens.extend(rest.as_bytes()[..width].iter().map(|&byte| i32::from(byte)));
        rest = &rest[width..];
    }
    tokens
}

fn eog_id() -> i32 {
    marker_id(MARKERS.iter().position(|marker| *marker == EOG).unwrap())
}

/// What the policy says on one turn, as structured fields the template writes.
#[derive(Clone)]
struct Turn {
    reasoning: Option<&'static str>,
    content: &'static str,
    call: bool,
}

/// One model behind a real template: the policy that samples the turns, and
/// the dataset backend that prepares the export back.
struct FixtureModel {
    template: String,
    variables: Option<&'static str>,
    turns: Vec<Turn>,
    next: Mutex<usize>,
}

impl FixtureModel {
    fn new(name: &str, variables: Option<&'static str>, turns: Vec<Turn>) -> Self {
        Self {
            template: fixture(name),
            variables,
            turns,
            next: Mutex::new(0),
        }
    }

    fn render(&self, messages: &Value, tools: Option<&str>, add: bool) -> String {
        render_chat_template_source(
            &self.template,
            &messages.to_string(),
            tools,
            self.variables,
            add,
        )
        .expect("the template renders")
    }

    /// The text the template writes for `turn` right after its own generation
    /// prompt, without the closer the policy stops on.
    fn sampled(&self, turn: &Turn, tools: &str) -> String {
        let opening = json!([
            {"role": "system", "content": "be useful"},
            {"role": "user", "content": "hello"},
        ]);
        let prompt = self.render(&opening, Some(tools), true);
        let mut assistant = json!({"role": "assistant", "content": turn.content});
        if let Some(reasoning) = turn.reasoning {
            assistant["reasoning_content"] = reasoning.into();
        }
        if turn.call {
            assistant["tool_calls"] = json!([{"type": "function", "function": {
                "name": "test__echo", "arguments": {"text": "hi"},
            }}]);
        }
        let mut whole = opening.clone();
        whole.as_array_mut().unwrap().push(assistant);
        let rendered = self.render(&whole, Some(tools), false);
        let turn_text = rendered
            .strip_prefix(&prompt)
            .unwrap_or_else(|| panic!("the turn extends the prompt: {rendered:?}"));
        let at = turn_text
            .rfind(EOG)
            .unwrap_or_else(|| panic!("the turn is closed: {turn_text:?}"));
        turn_text[..at].to_owned()
    }

    fn generation(&self, tools: &str) -> PolicyGeneration {
        let mut next = self.next.lock().unwrap();
        let turn = self.turns[*next].clone();
        *next += 1;
        let text = self.sampled(&turn, tools);
        let mut tokens = tokenize(&text);
        tokens.push(eog_id());
        PolicyGeneration {
            tokens,
            text,
            stopped_at_eog: true,
            forced: Vec::new(),
        }
    }

    fn tools_json() -> String {
        template_tools(&[echo_spec()]).unwrap()
    }
}

impl DatasetBackend for FixtureModel {
    fn tokenize_text(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        let mut tokens = vec![BOS];
        tokens.extend(tokenize(text));
        Ok(tokens)
    }

    fn eos_token(&self) -> retrograd_core::Result<i32> {
        Ok(eog_id())
    }

    fn format_chat(&self, _: &[(&str, &str)], _: bool) -> retrograd_core::Result<String> {
        unreachable!("a tool record never takes the plain path")
    }

    fn tokenize_fragment(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        Ok(tokenize(text))
    }

    fn is_eog_token(&self, token: i32) -> retrograd_core::Result<bool> {
        Ok(token == eog_id())
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

    fn chat_template_supports_tools(&self) -> retrograd_core::Result<bool> {
        Ok(true)
    }

    fn tool_call_parser(&self, tools_json: &str) -> retrograd_core::Result<Option<String>> {
        tool_call_parser_from_source(&self.template, Some(tools_json), self.variables).map(Some)
    }

    fn parse_assistant(&self, parser: &str, text: &str) -> retrograd_core::Result<String> {
        parse_assistant_output(parser, text)
    }
}

#[async_trait]
impl Policy for FixtureModel {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add_assistant: bool,
    ) -> Result<RenderedFraming> {
        let (json, sentinels) = template_messages(messages)?;
        let tools = (!tools.is_empty())
            .then(|| template_tools(tools))
            .transpose()?;
        let rendered = self.format_chat_messages(&json, tools.as_deref(), add_assistant)?;
        let framing = split_turns(&rendered, &sentinels)?;
        let tail = match framing.turns.is_empty() {
            true => self.tokenize_text(&framing.tail)?,
            false => tokenize(&framing.tail),
        };
        Ok(RenderedFraming { framing, tail })
    }

    async fn supports_native_tools(&self) -> Result<bool> {
        Ok(true)
    }

    async fn tool_call_parser(
        &self,
        tools: &[ToolSpec],
    ) -> Result<Option<Arc<dyn ToolCallParser>>> {
        let parser = tool_call_parser_from_source(
            &self.template,
            Some(&template_tools(tools)?),
            self.variables,
        )?;
        Ok(Some(Arc::new(TemplateToolCallParser::new(parser))))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        let tools = Self::tools_json();
        Ok(sampling.iter().map(|_| self.generation(&tools)).collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        let tools = Self::tools_json();
        Ok(requests.iter().map(|_| self.generation(&tools)).collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Echo, with the schema a template-derived parser needs to read its argument.
fn echo_spec() -> ToolSpec {
    ToolSpec {
        name: "test__echo".into(),
        description: "echo".into(),
        input_schema: json!({"type": "object", "properties": {"text": {"type": "string"}}}),
    }
}

struct SchemaEchoTools;

#[async_trait]
impl ToolProvider for SchemaEchoTools {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(vec![echo_spec()])
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        EchoTools.call(call).await
    }
}

async fn collect(model: &Arc<FixtureModel>) -> Result<Trajectory> {
    let limits = RolloutLimits {
        max_turns: 4,
        max_new_tokens_per_turn: 512,
        max_trajectory_tokens: 8192,
        ..Default::default()
    };
    RolloutEngine::with_tools(
        model.clone(),
        Arc::new(SchemaEchoTools),
        Arc::new(HermesToolCallParser),
        limits,
    )?
    .rollout(&scenario(), 7)
    .await
}

fn prepare(
    model: &FixtureModel,
    trajectory: &Trajectory,
    form: AssistantForm,
) -> Result<ToolStream> {
    let example = to_chat_example(trajectory, &scenario(), form)?;
    Ok(ToolConversationRenderer::new(model).stream(&example)?)
}

const THINKING_ON: Option<&str> = Some(r#"{"enable_thinking":true}"#);
const THINKING_OFF: Option<&str> = Some(r#"{"enable_thinking":false}"#);

fn call(reasoning: Option<&'static str>) -> Turn {
    Turn {
        reasoning,
        content: "",
        call: true,
    }
}

fn answer(reasoning: Option<&'static str>) -> Turn {
    Turn {
        reasoning,
        content: "It said hi.",
        call: false,
    }
}

/// One mode of one template, and what a rollout under it does.
struct Case {
    template: &'static str,
    variables: Option<&'static str>,
    /// Whether the generation prompt leaves the model room to reason: every
    /// turn reasons when it does, none can when the prompt closed the block.
    reasons: bool,
    /// Turns the template re-renders differently from how they were sampled.
    normalized: usize,
}

const CASES: [Case; 10] = [
    Case {
        template: "lfm2.5-230m",
        variables: None,
        reasons: false,
        normalized: 0,
    },
    Case {
        template: "qwen3-0.6b",
        variables: None,
        reasons: true,
        normalized: 0,
    },
    Case {
        template: "qwen3-0.6b",
        variables: THINKING_ON,
        reasons: true,
        normalized: 0,
    },
    // The prompt closes an empty block the history of a turn leaves out.
    Case {
        template: "qwen3-0.6b",
        variables: THINKING_OFF,
        reasons: false,
        normalized: 1,
    },
    Case {
        template: "qwen3-1.7b",
        variables: THINKING_ON,
        reasons: true,
        normalized: 0,
    },
    Case {
        template: "qwen3-1.7b",
        variables: THINKING_OFF,
        reasons: false,
        normalized: 1,
    },
    Case {
        template: "qwen3.5-0.8b",
        variables: None,
        reasons: false,
        normalized: 0,
    },
    // The prompt opens the block and the model closes it.
    Case {
        template: "qwen3.5-0.8b",
        variables: THINKING_ON,
        reasons: true,
        normalized: 0,
    },
    Case {
        template: "qwen3.5-0.8b",
        variables: THINKING_OFF,
        reasons: false,
        normalized: 0,
    },
    Case {
        template: "qwen3-1.7b",
        variables: None,
        reasons: true,
        normalized: 0,
    },
];

#[tokio::test]
async fn a_rollout_through_each_mode_prepares_back_to_the_stream_it_was_collected_as() {
    for case in CASES {
        let reasoning = |text| case.reasons.then_some(text);
        let model = Arc::new(FixtureModel::new(
            case.template,
            case.variables,
            vec![call(reasoning("look left")), answer(reasoning("it echoed"))],
        ));
        let label = format!("{} {:?}", case.template, case.variables);
        let trajectory = collect(&model)
            .await
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        let roles = trajectory
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "tool", "assistant"],
            "{label}"
        );
        let provenance = trajectory.provenance.as_ref().unwrap();
        assert_eq!(provenance.checked_turns, 1, "{label}");
        assert_eq!(provenance.normalized_turns, case.normalized, "{label}");
        if case.reasons {
            assert_eq!(
                trajectory.messages[2].reasoning.as_deref(),
                Some("look left"),
                "{label}: the turn keeps the reasoning its parser read"
            );
        }
        for form in [AssistantForm::Structured, AssistantForm::Raw] {
            let stream = prepare(&model, &trajectory, form)
                .unwrap_or_else(|error| panic!("{label} {form:?}: {error}"));
            assert_eq!(stream.tokens, trajectory.tokens, "{label} {form:?}");
            assert_eq!(stream.train_mask, trajectory.train_mask, "{label} {form:?}");
        }
    }
}

/// A model that writes an empty reasoning block of its own - Qwen3 thinking
/// and finding nothing to say - samples turns whose structured form has no
/// reasoning to hand back: the record holds the answer, and the empty block is
/// the one thing it cannot say. The raw form keeps it.
#[tokio::test]
async fn an_empty_block_the_model_wrote_survives_in_the_raw_form_only() {
    let model = Arc::new(FixtureModel::new(
        "qwen3-0.6b",
        THINKING_ON,
        vec![call(None), answer(None)],
    ));
    let trajectory = collect(&model).await.unwrap();
    let raw = prepare(&model, &trajectory, AssistantForm::Raw).unwrap();
    assert_eq!(raw.tokens, trajectory.tokens);
    assert_eq!(raw.train_mask, trajectory.train_mask);

    let structured = prepare(&model, &trajectory, AssistantForm::Structured).unwrap();
    let empty = tokenize("<think>\n\n</think>\n\n");
    let mut collected = trajectory.tokens.clone();
    while let Some(at) = collected
        .windows(empty.len())
        .position(|window| window == empty)
    {
        collected.drain(at..at + empty.len());
    }
    assert_eq!(structured.tokens, collected);
}

/// LFM2.5 never writes `reasoning_content`, so a turn that reasoned - inline,
/// in the only place its template leaves for it - comes back from the history
/// without its reasoning: an inference server re-rendering the conversation
/// would show the model a history its next turns do not continue.
#[tokio::test]
async fn a_template_that_drops_the_reasoning_of_earlier_turns_fails_the_rollout() {
    let model = Arc::new(FixtureModel::new(
        "lfm2.5-230m",
        None,
        vec![
            Turn {
                reasoning: None,
                content: "<think>look left</think>",
                call: true,
            },
            answer(None),
        ],
    ));
    let error = collect(&model).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("drops the reasoning of assistant turn 0"),
        "{error}"
    );
}
