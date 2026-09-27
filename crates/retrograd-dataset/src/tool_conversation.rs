//! Tool records: the rules a conversation with calls has to follow, and the
//! token stream it is trained as.
//!
//! A tool record is not rendered by prefix differences like a plain one. It is
//! rendered the way an agentic rollout renders its conversation - the template
//! writes only the framing around the assistant turns, each assistant turn is
//! tokenized on its own, and the turn's end-of-generation token is where the
//! policy stops - so the stream a warm-start trains on is the stream a rollout
//! samples and parses.
//!
//! One limit is inherent and not a defect: sampling may choose a non-canonical
//! split of the same text (`"]]" + "])"` where the tokenizer writes `"]" +
//! "]])"`), which no text can reproduce. The framing and the mask match a
//! rollout exactly; an assistant span matches it up to that canonicalization.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use retrograd_core::{Error, Result};
use serde_json::{Value, json};

use crate::chat_template::{
    ChatTemplateError, ParsedCalls, TemplateCall, TemplateMessage, TemplateTool, ToolRenderingKind,
    assistant_span, decide_rendering, decode_parsed_assistant, hermes_call_text,
    inject_tool_instructions, observation_text, parse_hermes, prompt_tool_instructions,
    split_assistant_spans, template_messages, template_messages_revealing, template_tools,
};
use crate::{ChatExample, ChatExampleError, DatasetBackend, IGNORE_LABEL};

/// The catalog and the calls, checked without a model: what the server can
/// refuse at upload and what an export is checked against before it is
/// written.
///
/// A call is answered by the `tool` messages that directly follow its turn,
/// each at most once, by id or - without one - in call order. Every call is
/// answered before the next message that is not a `tool` one, because that is
/// the only order a rollout produces: the environment answers every call of a
/// turn before the policy speaks again.
pub(crate) fn validate_tools(example: &ChatExample) -> std::result::Result<(), ChatExampleError> {
    let mut declared = HashSet::with_capacity(example.tools.len());
    for tool in &example.tools {
        if tool.name.is_empty() {
            return Err(ChatExampleError::EmptyToolName);
        }
        if !declared.insert(tool.name.as_str()) {
            return Err(ChatExampleError::DuplicateToolName {
                name: tool.name.clone(),
            });
        }
    }

    // `(id, name)` of the calls of the last assistant turn still waiting for
    // their result, in call order.
    let mut pending: Vec<(String, &str)> = Vec::new();
    // Whether a `tool` message may come next: right after an assistant turn
    // with calls, or after another observation of that same turn.
    let mut answering = false;
    for message in &example.messages {
        match message.role.as_str() {
            "tool" => {
                if !answering {
                    return Err(ChatExampleError::ToolWithoutCall);
                }
                let position = match &message.tool_call_id {
                    Some(id) => pending
                        .iter()
                        .position(|(pending, _)| pending == id)
                        .ok_or_else(|| ChatExampleError::UnknownCallId { id: id.clone() })?,
                    None if pending.is_empty() => return Err(ChatExampleError::ToolWithoutCall),
                    None => 0,
                };
                let (id, call) = pending.remove(position);
                if let Some(name) = &message.name
                    && name != call
                {
                    return Err(ChatExampleError::ToolNameMismatch {
                        id,
                        name: name.clone(),
                        call: call.to_owned(),
                    });
                }
            }
            role => {
                if let Some((id, _)) = pending.first() {
                    return Err(ChatExampleError::UnansweredCall { id: id.clone() });
                }
                answering = role == "assistant" && !message.tool_calls.is_empty();
                for (index, call) in message.tool_calls.iter().enumerate() {
                    if call.name.is_empty() {
                        return Err(ChatExampleError::EmptyToolName);
                    }
                    if !call.arguments.is_object() {
                        return Err(ChatExampleError::ArgumentsNotAnObject {
                            name: call.name.clone(),
                        });
                    }
                    if example.tools.is_empty() {
                        return Err(ChatExampleError::CallsWithoutTools);
                    }
                    if !declared.contains(call.name.as_str()) {
                        return Err(ChatExampleError::UndeclaredTool {
                            name: call.name.clone(),
                        });
                    }
                    pending.push((call.resolved_id(index), call.name.as_str()));
                }
            }
        }
    }
    match pending.first() {
        Some((id, _)) => Err(ChatExampleError::UnansweredCall { id: id.clone() }),
        None => Ok(()),
    }
}

/// Why a valid tool record cannot be trained on as the model's own template
/// renders it.
#[derive(Debug, thiserror::Error)]
pub enum ToolRenderError {
    #[error(transparent)]
    Template(#[from] ChatTemplateError),
    /// Only the observations of the last turn's calls may follow it: an
    /// environment that ends the episode on a call leaves the trajectory there.
    #[error(
        "a tool record must end on an assistant turn, or on the observations that answer it: \
         nothing after the last one is learned"
    )]
    EndsWithoutAssistant,
    /// Rendering the turn for real changed text outside it, so there is no
    /// span that is "what the template writes for this turn".
    #[error("the chat template renders assistant turn {turn} non-locally; use raw assistant turns")]
    NonLocalTurn { turn: usize },
    #[error(
        "assistant turn {turn} does not read back as the calls it declares: expected \
         {expected}, parsed {parsed}"
    )]
    CallsDoNotRoundTrip {
        turn: usize,
        expected: String,
        parsed: String,
    },
    #[error("assistant turn {turn} gives its tool calls twice, in content and in tool_calls")]
    CallsGivenTwice { turn: usize },
    /// The rollout refuses the same template, on the same check.
    #[error(
        "the chat template rewrites the framing before assistant turn {turn} once the \
         conversation continues past it, so it cannot be used for multi-turn rollouts"
    )]
    RewrittenFraming { turn: usize },
}

impl From<ToolRenderError> for Error {
    fn from(error: ToolRenderError) -> Self {
        Error::invalid(error.to_string())
    }
}

/// A tool record as a token stream: every token, and which of them are the
/// model's own - the assistant turns and the end-of-generation token that
/// closes each, never an observation or the template's framing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolStream {
    pub tokens: Vec<i32>,
    pub train_mask: Vec<bool>,
}

impl ToolStream {
    /// One label per transition: `labels[t - 1]` is `tokens[t]` where that
    /// token is trained, [`IGNORE_LABEL`] elsewhere.
    pub fn labels(&self) -> Vec<i32> {
        (1..self.tokens.len())
            .map(|t| match self.train_mask[t] {
                true => self.tokens[t],
                false => IGNORE_LABEL,
            })
            .collect()
    }

    /// The training row and its labels, with the checks a plain record gets.
    pub(crate) fn into_row(self, n_ctx: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        if self.tokens.len() < 2 {
            return Err(Error::invalid(
                "formatted conversation has fewer than two tokens",
            ));
        }
        if self.tokens.len() > n_ctx + 1 {
            return Err(Error::invalid(format!(
                "formatted conversation has {} tokens, exceeding ctx={n_ctx}",
                self.tokens.len()
            )));
        }
        let labels = self.labels();
        if labels.iter().all(|&label| label == IGNORE_LABEL) {
            return Err(Error::tokenize(
                "assistant responses produced no trainable tokens",
            ));
        }
        let mut tokens = self.tokens;
        tokens.pop();
        Ok((tokens, labels))
    }
}

/// How one catalog is rendered, decided once for every record that offers it:
/// deriving a parser from the template is not free, and a dataset usually has a
/// single catalog.
struct Rendering {
    kind: ToolRenderingKind,
    /// The catalog handed to the template, under native rendering only.
    tools_json: Option<String>,
    parser: Option<String>,
    /// The catalog written into the system turn, under prompt rendering only.
    instructions: Option<String>,
}

/// Renders tool records into [`ToolStream`]s, caching the rendering decision
/// per catalog for as long as it lives - one dataset.
pub struct ToolConversationRenderer<'b, B: DatasetBackend + ?Sized> {
    backend: &'b B,
    renderings: HashMap<String, Rc<Rendering>>,
}

impl<'b, B: DatasetBackend + ?Sized> ToolConversationRenderer<'b, B> {
    pub fn new(backend: &'b B) -> Self {
        Self {
            backend,
            renderings: HashMap::new(),
        }
    }

    /// The same decision the rollout takes for the same catalog: native when the
    /// template renders tools and yields a parser for them, the prompt-described
    /// convention otherwise.
    fn rendering(&mut self, tools: &[TemplateTool]) -> Result<Rc<Rendering>> {
        if tools.is_empty() {
            return Ok(Rc::new(Rendering {
                kind: ToolRenderingKind::None,
                tools_json: None,
                parser: None,
                instructions: None,
            }));
        }
        let tools_json = template_tools(tools).map_err(ToolRenderError::from)?;
        if let Some(rendering) = self.renderings.get(&tools_json) {
            return Ok(rendering.clone());
        }
        let native = self.backend.chat_template_supports_tools()?;
        let parser = match native {
            true => self.backend.tool_call_parser(&tools_json)?,
            false => None,
        };
        let rendering = Rc::new(
            match decide_rendering(tools.len(), native, parser.is_some()) {
                ToolRenderingKind::Native => Rendering {
                    kind: ToolRenderingKind::Native,
                    tools_json: Some(tools_json.clone()),
                    parser,
                    instructions: None,
                },
                kind => Rendering {
                    kind,
                    tools_json: None,
                    parser: None,
                    instructions: Some(
                        prompt_tool_instructions(tools).map_err(ToolRenderError::from)?,
                    ),
                },
            },
        );
        self.renderings.insert(tools_json, rendering.clone());
        Ok(rendering)
    }

    /// The token stream a rollout of this conversation would have produced.
    pub fn stream(&mut self, example: &ChatExample) -> Result<ToolStream> {
        example.validate()?;
        let assistants = example
            .messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.role == "assistant")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if assistants.is_empty() {
            return Err(Error::invalid(
                "messages must contain an assistant response",
            ));
        }
        if example
            .messages
            .iter()
            .rev()
            .find(|message| message.role != "tool")
            .is_some_and(|message| message.role != "assistant")
        {
            return Err(ToolRenderError::EndsWithoutAssistant.into());
        }
        let rendering = self.rendering(&example.tools)?;
        let (calls, answers) = resolve_calls(example);

        let mut messages = example
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| match message.role.as_str() {
                "tool" => {
                    let id = answers[index]
                        .as_deref()
                        .expect("validated: every observation answers a call");
                    TemplateMessage {
                        role: "tool",
                        content: observation_text(
                            rendering.kind,
                            id,
                            &message.content,
                            message.is_error,
                        )
                        .into(),
                        tool_call_id: Some(id),
                        tool_calls: &[],
                    }
                }
                role => TemplateMessage {
                    tool_calls: &calls[index],
                    ..TemplateMessage::text(role, message.content.as_str())
                },
            })
            .collect::<Vec<_>>();
        if let Some(instructions) = &rendering.instructions {
            inject_tool_instructions(&mut messages, instructions);
        }

        let tools_json = rendering.tools_json.as_deref();
        // Positions among the messages the template is handed, which the
        // prompt catalog may have opened with a system turn of its own.
        let turn_positions = messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.role == "assistant")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let pieces = self.framing(&messages, &turn_positions, tools_json)?;
        // The whole conversation, every turn a sentinel: what a structured turn
        // is cut out of, and where the last turn's closer is read.
        let (json, sentinels) = template_messages(&messages).map_err(ToolRenderError::from)?;
        let rendered = self
            .backend
            .format_chat_messages(&json, tools_json, false)?;
        let closing = split_assistant_spans(&rendered, &sentinels)
            .map_err(ToolRenderError::from)?
            .pop()
            .expect("a split yields one piece more than there are turns");

        let mut turns = Vec::with_capacity(assistants.len());
        for (turn, &index) in assistants.iter().enumerate() {
            let message = &example.messages[index];
            let declared = &calls[index];
            let text = match (message.raw, declared.is_empty(), rendering.kind) {
                (true, _, _) | (false, true, _) => message.content.clone(),
                (false, false, ToolRenderingKind::Native) => {
                    let revealed = template_messages_revealing(&messages, turn)
                        .map_err(ToolRenderError::from)?;
                    let whole = self
                        .backend
                        .format_chat_messages(&revealed, tools_json, false)?;
                    local_turn(&rendered, &whole, &assistant_span(turn))
                        .ok_or(ToolRenderError::NonLocalTurn { turn })?
                        .to_owned()
                }
                (false, false, _) => {
                    let calls = hermes_call_text(declared).map_err(ToolRenderError::from)?;
                    match message.content.is_empty() {
                        true => calls,
                        false => format!("{}\n{calls}", message.content),
                    }
                }
            };
            // Prose that already holds calls would teach every call twice.
            if !message.raw && !declared.is_empty() && !message.content.is_empty() {
                let prose = self.read_back(&rendering, &message.content);
                if prose.is_ok_and(|parsed| !parsed.calls.is_empty() || !parsed.errors.is_empty()) {
                    return Err(ToolRenderError::CallsGivenTwice { turn }.into());
                }
            }
            self.check_read_back(&rendering, turn, &text, declared)?;
            turns.push(text);
        }

        let eos = self.backend.eos_token()?;
        let mut tokens = self.backend.tokenize_text(&pieces[0])?;
        let mut train_mask = vec![false; tokens.len()];
        for (turn, text) in turns.iter().enumerate() {
            let sampled = self.backend.tokenize_fragment(text)?;
            train_mask.resize(train_mask.len() + sampled.len(), true);
            tokens.extend(sampled);
            // The policy stops on an end-of-generation token. When the template
            // opens the next piece by closing the turn with one, that token is
            // the one the policy sampled and is trained; otherwise the policy
            // still had to stop, on the model's own end of sequence, and the
            // template's closer is framing.
            let next = self
                .backend
                .tokenize_fragment(pieces.get(turn + 1).unwrap_or(&closing))?;
            let framing = match next.first() {
                Some(&closer) if self.backend.is_eog_token(closer)? => {
                    tokens.push(closer);
                    &next[1..]
                }
                _ => {
                    tokens.push(eos);
                    &next[..]
                }
            };
            train_mask.push(true);
            // A rollout ends on the last sampled turn; what the template writes
            // after it is never part of the stream.
            if turn + 1 < turns.len() {
                tokens.extend_from_slice(framing);
                train_mask.resize(tokens.len(), false);
            }
        }
        Ok(ToolStream { tokens, train_mask })
    }

    /// The framing before each assistant turn, rendered the way a rollout
    /// renders it: the conversation up to that turn, with the generation prompt,
    /// every earlier turn a sentinel. Returns one piece per assistant turn.
    ///
    /// Rendering the finished conversation once would not do. A template is
    /// free to write a turn it is handed differently from the prompt it opens a
    /// generation with - Qwen3 puts an empty `<think>` block before the last
    /// assistant turn of a conversation, and no generation prompt writes one -
    /// and the rollout only ever sees the second. It also refuses, on each
    /// render, a template that rewrites a piece already committed, which a
    /// rollout could not use either.
    fn framing(
        &self,
        messages: &[TemplateMessage<'_>],
        assistants: &[usize],
        tools_json: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut committed: Vec<String> = Vec::with_capacity(assistants.len());
        for (turn, &index) in assistants.iter().enumerate() {
            let (json, sentinels) =
                template_messages(&messages[..index]).map_err(ToolRenderError::from)?;
            let rendered = self.backend.format_chat_messages(&json, tools_json, true)?;
            let pieces =
                split_assistant_spans(&rendered, &sentinels).map_err(ToolRenderError::from)?;
            if pieces[..turn] != committed[..] {
                let at = (0..turn).find(|&i| pieces[i] != committed[i]).unwrap_or(0);
                return Err(ToolRenderError::RewrittenFraming { turn: at }.into());
            }
            committed = pieces;
        }
        Ok(committed)
    }

    /// Reads an assistant turn back with the parser its rendering pairs with,
    /// or says why the reading itself failed.
    fn read_back(
        &self,
        rendering: &Rendering,
        text: &str,
    ) -> std::result::Result<ParsedCalls, String> {
        match (rendering.kind, &rendering.parser) {
            (ToolRenderingKind::Native, Some(parser)) => {
                let document = self
                    .backend
                    .parse_assistant(parser, text)
                    .map_err(|error| error.to_string())?;
                decode_parsed_assistant(&document).map_err(|error| error.to_string())
            }
            (ToolRenderingKind::Prompt, _) => Ok(parse_hermes(text)),
            _ => Ok(ParsedCalls {
                content: text.to_owned(),
                ..Default::default()
            }),
        }
    }

    /// What a turn teaches is what a rollout will read: the same calls, in the
    /// same order, with the same arguments, and nothing that fails to parse.
    fn check_read_back(
        &self,
        rendering: &Rendering,
        turn: usize,
        text: &str,
        declared: &[TemplateCall],
    ) -> Result<()> {
        let parsed = self.read_back(rendering, text);
        let matches = parsed.as_ref().is_ok_and(|parsed| {
            parsed.errors.is_empty()
                && parsed.calls.len() == declared.len()
                && parsed.calls.iter().zip(declared).all(|(parsed, declared)| {
                    parsed.name == declared.name && parsed.arguments == declared.arguments
                })
        });
        if matches {
            return Ok(());
        }
        let parsed = match parsed {
            Ok(parsed) => {
                let mut text = calls_text(&parsed.calls);
                for (index, error) in &parsed.errors {
                    text.push_str(&format!("; call {index}: {error}"));
                }
                text
            }
            Err(error) => error,
        };
        Err(ToolRenderError::CallsDoNotRoundTrip {
            turn,
            expected: calls_text(declared),
            parsed,
        }
        .into())
    }
}

fn calls_text(calls: &[TemplateCall]) -> String {
    Value::Array(
        calls
            .iter()
            .map(|call| json!({"name": call.name, "arguments": call.arguments}))
            .collect(),
    )
    .to_string()
}

/// What `whole` - `hidden` with one turn rendered for real - has in place of
/// that turn's sentinel, when the rest of the text is untouched.
fn local_turn<'w>(hidden: &str, whole: &'w str, sentinel: &str) -> Option<&'w str> {
    let at = hidden.find(sentinel)?;
    let (before, after) = (&hidden[..at], &hidden[at + sentinel.len()..]);
    (whole.len() >= before.len() + after.len()
        && whole.starts_with(before)
        && whole.ends_with(after))
    .then(|| &whole[before.len()..whole.len() - after.len()])
}

/// Every assistant turn's calls under the ids they are answered by, and every
/// observation's call id - its own, or the next call still waiting, the same
/// resolution [`validate_tools`] checks.
fn resolve_calls(example: &ChatExample) -> (Vec<Vec<TemplateCall>>, Vec<Option<String>>) {
    let mut calls = Vec::with_capacity(example.messages.len());
    let mut answers = Vec::with_capacity(example.messages.len());
    let mut pending: Vec<String> = Vec::new();
    for message in &example.messages {
        let turn = message
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| TemplateCall {
                id: call.resolved_id(index),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect::<Vec<_>>();
        let answer = match message.role.as_str() {
            "tool" => {
                let position = message
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| pending.iter().position(|pending| pending == id))
                    .unwrap_or(0);
                (position < pending.len()).then(|| pending.remove(position))
            }
            _ => {
                pending = turn.iter().map(|call| call.id.clone()).collect();
                None
            }
        };
        calls.push(turn);
        answers.push(answer);
    }
    (calls, answers)
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;
    use crate::chat_template::TemplateTool;
    use crate::{ChatMessage, ChatToolCall};

    fn tool(name: &str) -> TemplateTool {
        TemplateTool {
            name: name.into(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        }
    }

    fn call(id: Option<&str>, name: &str) -> ChatToolCall {
        ChatToolCall {
            id: id.map(str::to_owned),
            name: name.into(),
            arguments: json!({"cmd": "ls"}),
        }
    }

    fn calling(calls: Vec<ChatToolCall>) -> ChatMessage {
        ChatMessage {
            tool_calls: calls,
            ..ChatMessage::text("assistant", "")
        }
    }

    fn answer(id: Option<&str>, content: &str) -> ChatMessage {
        ChatMessage {
            tool_call_id: id.map(str::to_owned),
            ..ChatMessage::text("tool", content)
        }
    }

    fn record(tools: &[&str], messages: Vec<ChatMessage>) -> ChatExample {
        ChatExample {
            tools: tools.iter().map(|name| tool(name)).collect(),
            messages,
            rubric: None,
            metadata: Default::default(),
        }
    }

    fn user(content: &str) -> ChatMessage {
        ChatMessage::text("user", content)
    }

    fn assistant(content: &str) -> ChatMessage {
        ChatMessage::text("assistant", content)
    }

    #[test]
    fn a_call_answered_then_a_final_answer_is_a_valid_record() {
        let example = record(
            &["run"],
            vec![
                user("go"),
                calling(vec![call(Some("a"), "run"), call(None, "run")]),
                // Answered out of order by id, then positionally.
                answer(Some("a"), "one"),
                answer(None, ""),
                assistant("done"),
            ],
        );
        example.validate().unwrap();
        assert!(example.is_tool_record());
    }

    #[test]
    fn an_observation_needs_a_call_to_answer() {
        let error = record(
            &["run"],
            vec![user("go"), answer(None, "x"), assistant("a")],
        )
        .validate()
        .unwrap_err();
        assert_eq!(error, ChatExampleError::ToolWithoutCall);

        // Every call already has its answer.
        let error = record(
            &["run"],
            vec![
                user("go"),
                calling(vec![call(None, "run")]),
                answer(None, "x"),
                answer(None, "y"),
                assistant("a"),
            ],
        )
        .validate()
        .unwrap_err();
        assert_eq!(error, ChatExampleError::ToolWithoutCall);
    }

    #[test]
    fn a_call_is_answered_before_anyone_else_speaks() {
        let error = record(
            &["run"],
            vec![
                user("go"),
                calling(vec![call(None, "run"), call(None, "run")]),
                answer(None, "x"),
                assistant("a"),
            ],
        )
        .validate()
        .unwrap_err();
        assert_eq!(
            error,
            ChatExampleError::UnansweredCall {
                id: "call_1".into()
            }
        );
        assert_eq!(
            error.to_string(),
            "tool call 'call_1' is never answered by a 'tool' message"
        );

        let error = record(&["run"], vec![user("go"), calling(vec![call(None, "run")])])
            .validate()
            .unwrap_err();
        assert!(matches!(error, ChatExampleError::UnansweredCall { .. }));
    }

    #[test]
    fn an_observation_answers_a_call_of_its_own_turn_once() {
        let error = record(
            &["run"],
            vec![
                user("go"),
                calling(vec![call(Some("a"), "run")]),
                answer(Some("b"), "x"),
                assistant("a"),
            ],
        )
        .validate()
        .unwrap_err();
        assert_eq!(error, ChatExampleError::UnknownCallId { id: "b".into() });
    }

    #[test]
    fn an_observation_named_after_another_tool_is_refused() {
        let mut observation = answer(None, "x");
        observation.name = Some("other".into());
        let error = record(
            &["run", "other"],
            vec![
                user("go"),
                calling(vec![call(None, "run")]),
                observation,
                assistant("a"),
            ],
        )
        .validate()
        .unwrap_err();
        assert!(
            matches!(error, ChatExampleError::ToolNameMismatch { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_call_names_a_declared_tool() {
        let messages = || {
            vec![
                user("go"),
                calling(vec![call(None, "run")]),
                answer(None, "x"),
                assistant("a"),
            ]
        };
        assert_eq!(
            record(&["other"], messages()).validate().unwrap_err(),
            ChatExampleError::UndeclaredTool { name: "run".into() }
        );
        assert_eq!(
            record(&[], messages()).validate().unwrap_err(),
            ChatExampleError::CallsWithoutTools
        );
    }

    #[test]
    fn the_catalog_names_each_tool_once() {
        assert_eq!(
            record(&["run", "run"], vec![user("go"), assistant("a")])
                .validate()
                .unwrap_err(),
            ChatExampleError::DuplicateToolName { name: "run".into() }
        );
        assert_eq!(
            record(&[""], vec![user("go"), assistant("a")])
                .validate()
                .unwrap_err(),
            ChatExampleError::EmptyToolName
        );
        let mut nameless = call(None, "");
        nameless.name.clear();
        assert_eq!(
            record(&["run"], vec![user("go"), calling(vec![nameless])])
                .validate()
                .unwrap_err(),
            ChatExampleError::EmptyToolName
        );
    }

    #[test]
    fn arguments_are_an_object_or_a_string_holding_one() {
        let parsed: ChatToolCall = serde_json::from_value(json!({
            "id": "a", "type": "function",
            "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"},
        }))
        .unwrap();
        assert_eq!(parsed.arguments, json!({"cmd": "ls"}));
        // Written back as the object.
        assert_eq!(
            serde_json::to_value(&parsed).unwrap()["function"]["arguments"],
            json!({"cmd": "ls"})
        );

        for arguments in [json!("not json"), json!("[1]"), json!([1])] {
            let parsed: ChatToolCall = serde_json::from_value(json!({
                "function": {"name": "run", "arguments": arguments},
            }))
            .unwrap();
            let error = record(
                &["run"],
                vec![
                    user("go"),
                    calling(vec![parsed]),
                    answer(None, ""),
                    assistant("a"),
                ],
            )
            .validate()
            .unwrap_err();
            assert_eq!(
                error,
                ChatExampleError::ArgumentsNotAnObject { name: "run".into() },
                "{arguments}"
            );
        }
    }

    #[test]
    fn a_tool_field_stays_on_its_own_role() {
        let mut stray = user("go");
        stray.tool_call_id = Some("a".into());
        assert_eq!(
            record(&["run"], vec![stray, assistant("a")])
                .validate()
                .unwrap_err(),
            ChatExampleError::ToolFieldOnWrongRole {
                field: "tool_call_id",
                role: "user".into()
            }
        );
        let mut stray = answer(None, "x");
        stray.raw = true;
        let error = record(
            &["run"],
            vec![user("go"), calling(vec![call(None, "run")]), stray],
        )
        .validate()
        .unwrap_err();
        assert_eq!(
            error,
            ChatExampleError::ToolFieldOnWrongRole {
                field: "raw",
                role: "tool".into()
            }
        );
    }

    #[test]
    fn only_a_calling_turn_or_an_observation_may_be_empty() {
        record(
            &["run"],
            vec![
                user("go"),
                calling(vec![call(None, "run")]),
                answer(None, ""),
                assistant("a"),
            ],
        )
        .validate()
        .unwrap();
        assert_eq!(
            record(&["run"], vec![user("go"), assistant("")])
                .validate()
                .unwrap_err(),
            ChatExampleError::EmptyContent
        );
    }

    /// A chat template small enough to predict by hand: `<role>content</role>`,
    /// a `<tools>` header, calls written `<call>{json}</call>`, and an assistant
    /// closer that is the one special token of a byte-level vocabulary.
    pub(crate) struct TemplateBackend {
        native: bool,
        /// Whether the assistant closer is an end-of-generation token.
        eog_closer: bool,
        /// Whether rendering a turn with its calls leaves the rest of the text
        /// alone.
        local: bool,
        /// Written before the last message when it is an assistant turn, the
        /// way Qwen3 writes an empty `<think>` block there.
        last_turn_prefix: &'static str,
        /// Opens an assistant turn already in the conversation, where the
        /// generation prompt writes `<assistant>`.
        historic_opener: &'static str,
    }

    const BOS: i32 = 999;
    const EOG: i32 = 1000;
    const EOS: i32 = 998;
    const CLOSER: &str = "</assistant>";

    impl Default for TemplateBackend {
        fn default() -> Self {
            Self {
                native: true,
                eog_closer: true,
                local: true,
                last_turn_prefix: "",
                historic_opener: "<assistant>",
            }
        }
    }

    impl TemplateBackend {
        fn closer(&self) -> &'static str {
            match self.eog_closer {
                true => CLOSER,
                false => "<end/>",
            }
        }
    }

    fn bytes(text: &str) -> Vec<i32> {
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

    impl DatasetBackend for TemplateBackend {
        fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
            let mut tokens = vec![BOS];
            tokens.extend(bytes(text));
            Ok(tokens)
        }

        fn eos_token(&self) -> Result<i32> {
            Ok(EOS)
        }

        fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
            let mut rendered = String::new();
            for (role, content) in messages {
                rendered.push_str(&format!("<{role}>{content}</{role}>"));
            }
            if add_assistant {
                rendered.push_str("<assistant>");
            }
            Ok(rendered)
        }

        fn tokenize_fragment(&self, text: &str) -> Result<Vec<i32>> {
            Ok(bytes(text))
        }

        fn is_eog_token(&self, token: i32) -> Result<bool> {
            Ok(token == EOG || token == EOS)
        }

        fn format_chat_messages(
            &self,
            messages_json: &str,
            tools_json: Option<&str>,
            add_assistant: bool,
        ) -> Result<String> {
            let messages: Vec<Value> = serde_json::from_str(messages_json).unwrap();
            let mut rendered = String::new();
            if let Some(tools) = tools_json {
                rendered.push_str(&format!("<tools>{tools}</tools>"));
            }
            for (index, message) in messages.iter().enumerate() {
                let role = message["role"].as_str().unwrap();
                let calls = message["tool_calls"].as_array();
                if !self.local && calls.is_some() {
                    rendered.insert_str(0, "<calls/>");
                }
                let opener = match role {
                    "assistant" if index + 1 == messages.len() => {
                        format!("{}{}", self.historic_opener, self.last_turn_prefix)
                    }
                    "assistant" => self.historic_opener.to_owned(),
                    role => format!("<{role}>"),
                };
                rendered.push_str(&format!("{opener}{}", message["content"].as_str().unwrap()));
                for call in calls.into_iter().flatten() {
                    let function = &call["function"];
                    rendered.push_str(&format!(
                        "<call>{}</call>",
                        json!({"name": function["name"], "arguments": function["arguments"]})
                    ));
                }
                match role {
                    "assistant" => rendered.push_str(self.closer()),
                    role => rendered.push_str(&format!("</{role}>")),
                }
            }
            if add_assistant {
                rendered.push_str("<assistant>");
            }
            Ok(rendered)
        }

        fn chat_template_supports_tools(&self) -> Result<bool> {
            Ok(self.native)
        }

        fn tool_call_parser(&self, _tools_json: &str) -> Result<Option<String>> {
            Ok(self.native.then(|| "calls".to_owned()))
        }

        fn parse_assistant(&self, parser: &str, text: &str) -> Result<String> {
            assert_eq!(parser, "calls");
            let mut content = String::new();
            let mut calls = Vec::new();
            let mut rest = text;
            while let Some(open) = rest.find("<call>") {
                content.push_str(&rest[..open]);
                let (body, after) = rest[open + "<call>".len()..]
                    .split_once("</call>")
                    .ok_or_else(|| Error::runtime("unterminated call"))?;
                let body: Value = serde_json::from_str(body)
                    .map_err(|error| Error::runtime(error.to_string()))?;
                calls.push(json!({
                    "id": "",
                    "name": body["name"],
                    "arguments": body["arguments"].to_string(),
                }));
                rest = after;
            }
            content.push_str(rest);
            Ok(json!({"content": content, "tool_calls": calls}).to_string())
        }
    }

    fn run_record(messages: Vec<ChatMessage>) -> ChatExample {
        let mut example = record(&["run"], messages);
        example.tools[0].description = "d".into();
        example
    }

    /// user → a call → its result → the answer.
    fn one_call(content: &str) -> ChatExample {
        let mut calling = calling(vec![call(None, "run")]);
        calling.content = content.into();
        run_record(vec![
            user("go"),
            calling,
            answer(None, "a.txt"),
            assistant("done"),
        ])
    }

    fn stream(backend: &TemplateBackend, example: &ChatExample) -> Result<ToolStream> {
        ToolConversationRenderer::new(backend).stream(example)
    }

    fn trained(stream: &ToolStream) -> Vec<i32> {
        stream
            .tokens
            .iter()
            .zip(&stream.train_mask)
            .filter(|(_, trained)| **trained)
            .map(|(token, _)| *token)
            .collect()
    }

    #[test]
    fn only_the_assistant_turns_and_their_closers_are_trained() {
        let backend = TemplateBackend::default();
        let stream = stream(&backend, &one_call("")).unwrap();
        let tools = template_tools(&run_record(Vec::new()).tools).unwrap();
        let call_text = r#"<call>{"arguments":{"cmd":"ls"},"name":"run"}</call>"#;

        let mut tokens = vec![BOS];
        let mut mask = vec![false];
        let mut push = |text: &str, train: bool| {
            let piece = bytes(text);
            mask.resize(mask.len() + piece.len(), train);
            tokens.extend(piece);
        };
        push(
            &format!("<tools>{tools}</tools><user>go</user><assistant>"),
            false,
        );
        // What the template writes for the turn, then the closer the policy
        // stops on.
        push(call_text, true);
        push(CLOSER, true);
        push("<tool>a.txt</tool><assistant>", false);
        push("done", true);
        push(CLOSER, true);
        assert_eq!(stream.tokens, tokens);
        assert_eq!(stream.train_mask, mask);
        // Nothing the template writes after the last closer is in the stream.
        assert_eq!(stream.tokens.last(), Some(&EOG));

        let labels = stream.labels();
        assert_eq!(labels.len(), stream.tokens.len() - 1);
        for (t, label) in labels.iter().enumerate() {
            match stream.train_mask[t + 1] {
                true => assert_eq!(*label, stream.tokens[t + 1]),
                false => assert_eq!(*label, IGNORE_LABEL),
            }
        }
    }

    #[test]
    fn prose_before_a_call_is_part_of_the_turn() {
        let backend = TemplateBackend::default();
        let stream = stream(&backend, &one_call("Let me look.")).unwrap();
        let trained = trained(&stream);
        assert!(
            trained.starts_with(&bytes("Let me look.<call>")),
            "{:?}",
            String::from_utf8_lossy(&trained.iter().map(|&t| t as u8).collect::<Vec<_>>())
        );
    }

    #[test]
    fn a_blind_template_gets_the_prompt_catalog_and_the_hermes_calls() {
        let backend = TemplateBackend {
            native: false,
            ..Default::default()
        };
        let stream = stream(&backend, &one_call("Let me look.")).unwrap();
        let text = String::from_utf8_lossy(
            &stream
                .tokens
                .iter()
                .filter(|&&token| token < 256)
                .map(|&token| token as u8)
                .collect::<Vec<_>>(),
        )
        .into_owned();
        assert!(
            text.starts_with("<system>Available tools (JSON): "),
            "a system turn of its own carries the catalog: {text}"
        );
        assert!(
            !text.contains("<tools>"),
            "the template is handed no catalog"
        );
        assert!(text.contains("<tool>call_0: a.txt</tool>"), "{text}");
        let trained = trained(&stream);
        let expected = bytes(
            "Let me look.\n<tool_call>{\"name\":\"run\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>",
        );
        assert!(trained.starts_with(&expected), "{text}");
    }

    #[test]
    fn an_existing_system_turn_gets_the_catalog_after_a_blank_line() {
        let backend = TemplateBackend {
            native: false,
            ..Default::default()
        };
        let mut example = one_call("");
        example
            .messages
            .insert(0, ChatMessage::text("system", "You fix bugs."));
        let stream = stream(&backend, &example).unwrap();
        assert!(stream.tokens[1..].starts_with(&bytes("<system>You fix bugs.\n\nAvailable tools")),);
    }

    #[test]
    fn a_failed_call_is_marked_the_way_a_rollout_marks_it() {
        let mut example = one_call("");
        example.messages[2].is_error = true;
        let native = stream(&TemplateBackend::default(), &example).unwrap();
        assert!(
            native
                .tokens
                .windows(bytes("<tool>ERROR: a.txt</tool>").len())
                .any(|window| window == bytes("<tool>ERROR: a.txt</tool>"))
        );
    }

    #[test]
    fn a_raw_turn_is_tokenized_verbatim_and_must_read_back_as_its_calls() {
        let backend = TemplateBackend::default();
        let written = r#"<call>{"arguments":{"cmd":"ls"},"name":"run"}</call>"#;
        let mut example = one_call(written);
        example.messages[1].raw = true;
        let stream = stream(&backend, &example).unwrap();
        assert!(trained(&stream).starts_with(&bytes(written)));

        // The text calls something else than the record declares.
        let mut example = one_call(r#"<call>{"arguments":{"cmd":"rm"},"name":"run"}</call>"#);
        example.messages[1].raw = true;
        let error = self::stream(&backend, &example).unwrap_err();
        assert!(error.to_string().contains("does not read back"), "{error}");

        // A teacher of another family: nothing this template's parser reads.
        let mut example = one_call("<tool_call>{\"name\":\"run\"}</tool_call>");
        example.messages[1].raw = true;
        let error = self::stream(&backend, &example).unwrap_err();
        assert!(error.to_string().contains("does not read back"), "{error}");
    }

    #[test]
    fn a_turn_with_no_declared_call_must_not_read_as_one() {
        let mut example = one_call("");
        example.messages[3].content = r#"<call>{"arguments":{},"name":"run"}</call>"#.into();
        let error = stream(&TemplateBackend::default(), &example).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("assistant turn 1 does not read back"),
            "{error}"
        );
    }

    #[test]
    fn calls_written_in_the_prose_and_declared_are_refused() {
        let error = stream(
            &TemplateBackend::default(),
            &one_call(r#"<call>{"arguments":{"cmd":"ls"},"name":"run"}</call>"#),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("gives its tool calls twice"),
            "{error}"
        );

        let blind = TemplateBackend {
            native: false,
            ..Default::default()
        };
        let error = stream(
            &blind,
            &one_call("<tool_call>{\"name\":\"run\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>"),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("gives its tool calls twice"),
            "{error}"
        );
    }

    #[test]
    fn a_template_that_rewrites_other_text_for_a_turn_is_refused() {
        let backend = TemplateBackend {
            local: false,
            ..Default::default()
        };
        let error = stream(&backend, &one_call("")).unwrap_err();
        assert!(error.to_string().contains("non-locally"), "{error}");
    }

    #[test]
    fn a_closer_that_does_not_stop_generation_is_framing_and_eos_is_trained() {
        let backend = TemplateBackend {
            eog_closer: false,
            ..Default::default()
        };
        let stream = stream(&backend, &one_call("")).unwrap();
        let trained = trained(&stream);
        let call_text = bytes(r#"<call>{"arguments":{"cmd":"ls"},"name":"run"}</call>"#);
        let mut expected = call_text;
        expected.push(EOS);
        expected.extend(bytes("done"));
        expected.push(EOS);
        assert_eq!(trained, expected);
        // The template's closer is still in the stream, as context.
        assert!(
            stream
                .tokens
                .windows(bytes("<end/><tool>").len())
                .any(|window| window == bytes("<end/><tool>"))
        );
    }

    #[test]
    fn a_record_ending_on_the_observation_of_its_last_call_stops_at_that_call() {
        // The shape an environment that ends the episode on `submit` produces.
        let example = run_record(vec![
            user("go"),
            calling(vec![call(None, "run")]),
            answer(None, "a.txt"),
        ]);
        let stream = stream(&TemplateBackend::default(), &example).unwrap();
        assert_eq!(
            stream.tokens.last(),
            Some(&EOG),
            "the calling turn's closer"
        );
        assert!(stream.train_mask.last().copied().unwrap());
        let observation = bytes("a.txt");
        assert!(
            !stream
                .tokens
                .windows(observation.len())
                .any(|window| window == observation),
            "nothing follows the observation, so it is not in the stream"
        );

        // A user turn after the last answer is something else: nothing
        // answers it.
        let mut example = one_call("");
        example.messages.push(user("and then?"));
        let error = self::stream(&TemplateBackend::default(), &example).unwrap_err();
        assert!(
            error.to_string().contains("must end on an assistant turn"),
            "{error}"
        );
    }

    #[test]
    fn a_backend_that_cannot_render_tools_says_so() {
        struct Plain;
        impl DatasetBackend for Plain {
            fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
                Ok(bytes(text))
            }
            fn eos_token(&self) -> Result<i32> {
                Ok(0)
            }
            fn format_chat(&self, _: &[(&str, &str)], _: bool) -> Result<String> {
                Ok(String::new())
            }
        }
        let error = ToolConversationRenderer::new(&Plain)
            .stream(&one_call(""))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot render tool conversations"),
            "{error}"
        );
    }

    #[test]
    fn two_calls_of_one_turn_are_answered_in_order() {
        let calling = calling(vec![call(Some("a"), "run"), call(Some("b"), "run")]);
        let example = run_record(vec![
            user("go"),
            calling,
            answer(Some("b"), "second"),
            answer(Some("a"), "first"),
            assistant("done"),
        ]);
        let blind = TemplateBackend {
            native: false,
            ..Default::default()
        };
        let stream = stream(&blind, &example).unwrap();
        let context = bytes("<tool>b: second</tool><tool>a: first</tool>");
        assert!(
            stream
                .tokens
                .windows(context.len())
                .any(|window| window == context)
        );
    }

    #[test]
    fn plain_and_tool_records_share_a_file_and_each_keeps_its_own_path() {
        let backend = TemplateBackend::default();
        let tool_line = serde_json::to_string(&one_call("")).unwrap();
        let plain_line =
            r#"{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}"#;
        let path = std::env::temp_dir().join(format!(
            "retrograd-dataset-mixed-{}.jsonl",
            std::process::id()
        ));
        std::fs::write(&path, format!("{plain_line}\n{tool_line}\n")).unwrap();

        let n_ctx = 512;
        let prepared =
            crate::prepare(&backend, &path, crate::DataFormat::ChatJsonl, n_ctx).unwrap();
        assert_eq!(prepared.examples, 2);
        let plain: ChatExample = serde_json::from_str(plain_line).unwrap();
        let (tokens, labels) = crate::prepare_conversation(&backend, &plain, n_ctx).unwrap();
        assert_eq!(&prepared.tokens[..tokens.len()], tokens.as_slice());
        assert_eq!(&prepared.labels[..labels.len()], labels.as_slice());

        let stream = self::stream(&backend, &one_call("")).unwrap();
        let (tokens, labels) = stream.clone().into_row(n_ctx).unwrap();
        assert_eq!(
            &prepared.tokens[n_ctx..n_ctx + tokens.len()],
            tokens.as_slice()
        );
        assert_eq!(
            &prepared.labels[n_ctx..n_ctx + labels.len()],
            labels.as_slice()
        );

        // The planner measures what preparation builds.
        let lengths =
            crate::measured_lengths(&backend, &path, crate::DataFormat::ChatJsonl).unwrap();
        assert_eq!(lengths[1] as usize, stream.tokens.len());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn the_framing_is_the_generation_prompt_not_the_rendered_turn() {
        let backend = TemplateBackend {
            last_turn_prefix: "<think></think>",
            ..Default::default()
        };
        let stream = stream(&backend, &one_call("")).unwrap();
        let prefix = bytes("<think></think>");
        assert!(
            !stream
                .tokens
                .windows(prefix.len())
                .any(|window| window == prefix),
            "a rollout opens the last turn on the generation prompt, which writes no prefix"
        );
        assert_eq!(
            stream,
            self::stream(&TemplateBackend::default(), &one_call("")).unwrap()
        );
    }

    #[test]
    fn a_template_that_rewrites_committed_framing_is_refused_as_a_rollout_refuses_it() {
        let backend = TemplateBackend {
            historic_opener: "<model>",
            ..Default::default()
        };
        let error = stream(&backend, &one_call("")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("rewrites the framing before assistant turn 0"),
            "{error}"
        );
    }
}
