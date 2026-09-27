//! A rollout, exported and prepared back for SFT, is the stream it was
//! collected as: same tokens, same mask.
//!
//! The policy below is also the dataset backend - one template, one tokenizer -
//! so the only thing that can separate the two streams is the code under test.
//! Its generations are the canonical tokenization of the text a template writes
//! for the turn; sampling is free to pick another split of the same text, which
//! no record can reproduce, and the last case pins that this costs the
//! assistant span and nothing else.

use std::sync::Arc;

use async_trait::async_trait;
use retrograd_core::SamplingParams;
use retrograd_dataset::chat_template::{decode_parsed_assistant, split_assistant_spans};
use retrograd_dataset::{DatasetBackend, ToolConversationRenderer, ToolStream};
use serde_json::{Value, json};

use super::*;
use crate::chat::{template_messages, template_tools};
use crate::export::{AssistantForm, to_chat_example};
use crate::policy::{Policy, PolicyGeneration};
use crate::tools::{ParsedAssistant, ToolCallParser, ToolSpec};
use crate::trajectory::{Message, Trajectory};

const BOS: i32 = 999;
const EOG: i32 = 1000;
const CLOSER: &str = "</assistant>";
/// A token no canonical tokenization produces: `"hi"` in one piece.
const HI: i32 = 2000;

/// `<role>content</role>`, a `<tools>` header when handed a catalog, calls as
/// `<call>{json}</call>`, and an assistant closer that is the vocabulary's one
/// end-of-generation token.
struct TemplatePolicy {
    native: bool,
    /// Samples `"hi"` as one token instead of two.
    non_canonical: bool,
}

fn tokenize(text: &str) -> Vec<i32> {
    let mut tokens = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(CLOSER) {
        tokens.extend(rest[..at].bytes().map(i32::from));
        tokens.push(EOG);
        rest = &rest[at + CLOSER.len()..];
    }
    tokens.extend(rest.bytes().map(i32::from));
    tokens
}

impl DatasetBackend for TemplatePolicy {
    fn tokenize_text(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        let mut tokens = vec![BOS];
        tokens.extend(tokenize(text));
        Ok(tokens)
    }

    fn eos_token(&self) -> retrograd_core::Result<i32> {
        Ok(EOG)
    }

    fn format_chat(&self, _: &[(&str, &str)], _: bool) -> retrograd_core::Result<String> {
        unreachable!("a tool record never takes the plain path")
    }

    fn tokenize_fragment(&self, text: &str) -> retrograd_core::Result<Vec<i32>> {
        Ok(tokenize(text))
    }

    fn is_eog_token(&self, token: i32) -> retrograd_core::Result<bool> {
        Ok(token == EOG)
    }

    fn format_chat_messages(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        add_assistant: bool,
    ) -> retrograd_core::Result<String> {
        let messages: Vec<Value> = serde_json::from_str(messages_json).unwrap();
        let mut rendered = String::new();
        if let Some(tools) = tools_json {
            rendered.push_str(&format!("<tools>{tools}</tools>"));
        }
        for message in &messages {
            let role = message["role"].as_str().unwrap();
            rendered.push_str(&format!("<{role}>{}", message["content"].as_str().unwrap()));
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                let function = &call["function"];
                rendered.push_str(&format!(
                    "<call>{}</call>",
                    json!({"name": function["name"], "arguments": function["arguments"]})
                ));
            }
            match role {
                "assistant" => rendered.push_str(CLOSER),
                role => rendered.push_str(&format!("</{role}>")),
            }
        }
        if add_assistant {
            rendered.push_str("<assistant>");
        }
        Ok(rendered)
    }

    fn chat_template_supports_tools(&self) -> retrograd_core::Result<bool> {
        Ok(self.native)
    }

    fn tool_call_parser(&self, _tools_json: &str) -> retrograd_core::Result<Option<String>> {
        Ok(self.native.then(|| "calls".to_owned()))
    }

    fn parse_assistant(&self, _parser: &str, text: &str) -> retrograd_core::Result<String> {
        Ok(parse_calls(text))
    }
}

/// The parser this template yields, as the runtime's document.
fn parse_calls(text: &str) -> String {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("<call>") {
        content.push_str(&rest[..open]);
        let (body, after) = rest[open + "<call>".len()..]
            .split_once("</call>")
            .expect("well-formed calls");
        let body: Value = serde_json::from_str(body).unwrap();
        calls.push(json!({
            "id": "", "name": body["name"], "arguments": body["arguments"].to_string(),
        }));
        rest = after;
    }
    content.push_str(rest);
    json!({"content": content, "tool_calls": calls}).to_string()
}

struct CallParser;

impl ToolCallParser for CallParser {
    fn parse(&self, output: &str) -> ParsedAssistant {
        decode_parsed_assistant(&parse_calls(output))
            .expect("the fake parser writes valid documents")
            .into()
    }
}

impl TemplatePolicy {
    /// One call on the first turn, the answer on the second - each the text
    /// the template writes for that turn, then the closer.
    fn generation(&self, prompt: &[i32]) -> PolicyGeneration {
        let answered = prompt.contains(&EOG);
        let text = match (answered, self.native) {
            (true, _) => "It said hi.".to_owned(),
            (false, true) => {
                r#"<call>{"arguments":{"text":"hi"},"name":"test__echo"}</call>"#.to_owned()
            }
            (false, false) => {
                r#"<tool_call>{"name":"test__echo","arguments":{"text":"hi"}}</tool_call>"#
                    .to_owned()
            }
        };
        let mut tokens = tokenize(&text);
        if self.non_canonical
            && let Some(at) = tokens.windows(2).position(|pair| pair == [104, 105])
        {
            tokens.splice(at..at + 2, [HI]);
        }
        tokens.push(EOG);
        PolicyGeneration {
            tokens,
            text,
            stopped_at_eog: true,
        }
    }
}

#[async_trait]
impl Policy for TemplatePolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add_assistant: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let (json, sentinels) = template_messages(messages)?;
        let tools = match tools.is_empty() {
            true => None,
            false => Some(template_tools(tools)?),
        };
        let rendered = self.format_chat_messages(&json, tools.as_deref(), add_assistant)?;
        Ok(split_assistant_spans(&rendered, &sentinels)?
            .iter()
            .enumerate()
            .map(|(index, piece)| match index {
                0 => self.tokenize_text(piece).unwrap(),
                _ => tokenize(piece),
            })
            .collect())
    }

    async fn supports_native_tools(&self) -> Result<bool> {
        Ok(self.native)
    }

    async fn tool_call_parser(
        &self,
        _tools: &[ToolSpec],
    ) -> Result<Option<Arc<dyn ToolCallParser>>> {
        Ok(self
            .native
            .then(|| Arc::new(CallParser) as Arc<dyn ToolCallParser>))
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling.iter().map(|_| self.generation(&prompt)).collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(requests
            .iter()
            .map(|(prompt, _)| self.generation(prompt))
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

async fn collect(policy: &Arc<TemplatePolicy>) -> Trajectory {
    let limits = RolloutLimits {
        max_turns: 4,
        max_new_tokens_per_turn: 256,
        max_trajectory_tokens: 4096,
        ..Default::default()
    };
    let trajectory = engine_with(policy.clone(), limits)
        .rollout(&scenario(), 7)
        .await
        .unwrap();
    assert!(!trajectory.truncated);
    assert_eq!(trajectory.provenance.as_ref().unwrap().invalid_turns, 0);
    let roles = trajectory
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "tool", "assistant"],
        "a call, its result, the answer"
    );
    trajectory
}

fn prepare(policy: &TemplatePolicy, trajectory: &Trajectory, form: AssistantForm) -> ToolStream {
    let example = to_chat_example(trajectory, &scenario(), form).unwrap();
    ToolConversationRenderer::new(policy)
        .stream(&example)
        .unwrap_or_else(|error| panic!("{form:?}: {error}"))
}

#[tokio::test]
async fn an_exported_rollout_prepares_back_to_the_stream_it_was_collected_as() {
    for native in [true, false] {
        let policy = Arc::new(TemplatePolicy {
            native,
            non_canonical: false,
        });
        let trajectory = collect(&policy).await;
        for form in [AssistantForm::Structured, AssistantForm::Raw] {
            let stream = prepare(&policy, &trajectory, form);
            let case = format!("native={native} {form:?}");
            assert_eq!(stream.tokens, trajectory.tokens, "{case}");
            assert_eq!(stream.train_mask, trajectory.train_mask, "{case}");
            // A label exactly where the rollout trained, and it is the sampled
            // token.
            for (t, label) in stream.labels().iter().enumerate() {
                match trajectory.train_mask[t + 1] {
                    true => assert_eq!(*label, trajectory.tokens[t + 1], "{case}"),
                    false => assert_eq!(*label, retrograd_dataset::IGNORE_LABEL, "{case}"),
                }
            }
        }
    }
}

#[tokio::test]
async fn a_non_canonical_sample_differs_in_its_span_and_nowhere_else() {
    let policy = Arc::new(TemplatePolicy {
        native: true,
        non_canonical: true,
    });
    let trajectory = collect(&policy).await;
    let stream = prepare(&policy, &trajectory, AssistantForm::Raw);
    assert_ne!(stream.tokens, trajectory.tokens);

    // Framing is identical run for run; each trained run differs only by the
    // one split the sampler chose.
    let runs = |tokens: &[i32], mask: &[bool]| {
        let mut runs: Vec<(bool, Vec<i32>)> = Vec::new();
        for (&token, &trained) in tokens.iter().zip(mask) {
            match runs.last_mut() {
                Some((kind, run)) if *kind == trained => run.push(token),
                _ => runs.push((trained, vec![token])),
            }
        }
        runs
    };
    let collected = runs(&trajectory.tokens, &trajectory.train_mask);
    let prepared = runs(&stream.tokens, &stream.train_mask);
    assert_eq!(collected.len(), prepared.len());
    for ((trained, sampled), (_, rebuilt)) in collected.iter().zip(&prepared) {
        match trained {
            false => assert_eq!(sampled, rebuilt, "the framing never moves"),
            true => {
                let canonical = sampled
                    .iter()
                    .flat_map(|&token| match token {
                        HI => vec![104, 105],
                        token => vec![token],
                    })
                    .collect::<Vec<_>>();
                assert_eq!(&canonical, rebuilt, "same text, canonical split");
            }
        }
    }
}
