//! One chat request against one model: the prompt the rollouts would render,
//! the sample, and the answer read back the way the rollouts read it.
//!
//! Two halves, split by where they run. [`complete`] needs the model and runs
//! on whichever thread owns it - the serving session's, or a live run's inside
//! its progress callback. [`finish`] needs nothing but the text and runs on the
//! request's own task: stop sequences, the tool-call parser, the finish reason.
//!
//! The prompt is rendered from the conversation as the client sent it
//! ([`chat::template_messages_verbatim`]), which is where a server and a rollout
//! differ, and the only place: a rollout re-injects its sampled tokens, a server
//! has only the text the client sends back. The opening turn is the same bytes
//! either way.

use std::collections::HashMap;
use std::sync::Arc;

use retrograd_agent::TemplateToolCallParser;
use retrograd_agent::chat;
use retrograd_agent::rendering::{
    ToolRendering, inject_tool_instructions, observation_message, resolve_tool_rendering_with,
};
use retrograd_agent::tools::{HermesToolCallParser, ToolCallParser, ToolResult};
use retrograd_agent::trajectory::{Message, Role};
use retrograd_core::{Result as CoreResult, SamplingParams};
use retrograd_engine::Trainer;

use crate::convert::ChatPrompt;
use crate::error::OpenAiError;
use crate::wire::{FinishReason, FunctionOut, ToolCallOut, Usage};

/// The part of a model a chat request needs. [`Trainer`] is the one that
/// matters; the trait is what lets the whole request path run in the fast lane
/// against [`crate::testing::FakeModel`].
pub trait ChatModel {
    fn supports_native_tools(&self) -> CoreResult<bool>;
    /// The parser the model's template yields for this catalog, if any.
    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>>;
    /// The generation prompt for a conversation, through the model's own
    /// template.
    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String>;
    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>>;
    fn context_size(&self) -> CoreResult<usize>;
    /// Samples a completion, from the model the run started from when `base`
    /// is set: the adapter disabled, never the run's anchor.
    fn sample(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
        base: bool,
    ) -> CoreResult<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32], unparse_special: bool) -> CoreResult<String>;
    fn is_eog(&self, token: i32) -> CoreResult<bool>;
}

impl ChatModel for Trainer {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        self.chat_template_supports_tools()
    }

    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>> {
        Trainer::tool_call_parser(self, Some(tools_json))
    }

    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        self.format_chat_messages(messages_json, tools_json, true)
    }

    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        self.tokenize_text(text)
    }

    fn context_size(&self) -> CoreResult<usize> {
        Trainer::context_size(self)
    }

    fn sample(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
        base: bool,
    ) -> CoreResult<Vec<i32>> {
        let generation = match base {
            true => self.with_base_model(|trainer| trainer.generate(prompt, sampling))?,
            false => self.generate(prompt, sampling)?,
        };
        Ok(generation.tokens)
    }

    fn detokenize(&self, tokens: &[i32], unparse_special: bool) -> CoreResult<String> {
        Trainer::detokenize(self, tokens, unparse_special)
    }

    fn is_eog(&self, token: i32) -> CoreResult<bool> {
        self.is_eog_token(token)
    }
}

/// The parser a template yields, per tool catalog, for as long as one set of
/// weights stays loaded. Same key the rollouts cache their rendering by.
#[derive(Default)]
pub struct ParserCache(HashMap<String, Option<String>>);

impl ParserCache {
    fn parser(&mut self, model: &dyn ChatModel, tools_json: &str) -> CoreResult<Option<String>> {
        if let Some(parser) = self.0.get(tools_json) {
            return Ok(parser.clone());
        }
        let parser = model.tool_call_parser(tools_json)?;
        self.0.insert(tools_json.to_owned(), parser.clone());
        Ok(parser)
    }
}

/// What the model produced, before anything was read out of it.
pub struct Sampled {
    pub text: String,
    pub prompt_tokens: usize,
    /// End-of-generation token included, when one stopped it: it was sampled.
    pub completion_tokens: usize,
    pub stopped_at_eog: bool,
    /// The parser the rendering chose, or `None` when there were no tools or
    /// the client asked for none to be read back.
    pub reader: Option<Arc<dyn ToolCallParser>>,
}

/// Renders, samples and detokenizes one request. Runs on the thread that owns
/// the model.
pub fn complete(
    model: &mut dyn ChatModel,
    prompt: &ChatPrompt,
    base: bool,
    parsers: &mut ParserCache,
) -> Result<Sampled, OpenAiError> {
    let tools_json = match prompt.tools.is_empty() {
        true => None,
        false => Some(chat::template_tools(&prompt.tools)?),
    };
    let rendering = match &tools_json {
        None => ToolRendering::None,
        Some(tools_json) => {
            let native = model.supports_native_tools()?;
            let parser = match native {
                true => parsers.parser(model, tools_json)?,
                false => None,
            };
            resolve_tool_rendering_with(
                native,
                parser.map(|blob| {
                    Arc::new(TemplateToolCallParser::new(blob)) as Arc<dyn ToolCallParser>
                }),
                prompt.tools.clone(),
            )?
        }
    };

    let mut messages = prompt
        .messages
        .iter()
        .map(|message| match message.role {
            // The client sends the tool's output; how it is framed is the
            // rendering's decision, the same one a rollout's observation gets.
            Role::Tool => observation_message(
                &ToolResult {
                    call_id: message.tool_call_id.clone().unwrap_or_default(),
                    content: message.content.clone(),
                    is_error: false,
                },
                &rendering,
            ),
            _ => message.clone(),
        })
        .collect::<Vec<Message>>();
    if let ToolRendering::Prompt { instructions, .. } = &rendering {
        inject_tool_instructions(&mut messages, instructions);
    }
    let native_catalog = match rendering {
        ToolRendering::Native { .. } => tools_json.as_deref(),
        ToolRendering::Prompt { .. } | ToolRendering::None => None,
    };
    let text = model.format_chat(
        &chat::template_messages_verbatim(&messages)?,
        native_catalog,
    )?;
    let tokens = model.tokenize(&text)?;
    if tokens.is_empty() {
        return Err(OpenAiError::invalid_param(
            "messages",
            "the conversation tokenized to nothing",
        ));
    }
    let context = model.context_size()?;
    let requested = prompt.sampling.max_new_tokens;
    if tokens.len().saturating_add(requested as usize) > context {
        return Err(OpenAiError::context_length_exceeded(
            tokens.len(),
            requested,
            context,
        ));
    }

    let mut sampled = model.sample(&tokens, &prompt.sampling, base)?;
    let completion_tokens = sampled.len();
    let mut stopped_at_eog = false;
    while let Some(&last) = sampled.last()
        && model.is_eog(last)?
    {
        sampled.pop();
        stopped_at_eog = true;
    }
    let reader: Option<Arc<dyn ToolCallParser>> = match &rendering {
        ToolRendering::None => None,
        ToolRendering::Native { parser, .. } => Some(parser.clone()),
        ToolRendering::Prompt { .. } => Some(Arc::new(HermesToolCallParser)),
    };
    // A template-derived parser matches the model's own call delimiters, and
    // those are control tokens: they have to survive detokenization for it to
    // read anything. Every other reader wants the text a person reads.
    let unparse_special = matches!(rendering, ToolRendering::Native { .. }) && reader.is_some();
    Ok(Sampled {
        text: model.detokenize(&sampled, unparse_special)?,
        prompt_tokens: tokens.len(),
        completion_tokens,
        stopped_at_eog,
        reader,
    })
}

/// The assistant turn a client reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    /// `None` when the turn is nothing but calls.
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallOut>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Calls the parser could not read. The raw text stays the content, as a
    /// rollout keeps it; the count is reported beside the answer.
    pub parse_errors: usize,
}

/// Reads the answer out of what was sampled: stop sequences first, then the
/// calls. Runs off the model's thread.
pub fn finish(sampled: Sampled, stop: &[String]) -> Completion {
    let (text, truncated) = truncate_at_stop(&sampled.text, stop);
    let (content, calls, parse_errors) = match &sampled.reader {
        Some(reader) => {
            let parsed = reader.parse(text);
            (parsed.content, parsed.tool_calls, parsed.parse_errors.len())
        }
        None => (text.to_owned(), Vec::new(), 0),
    };
    let tool_calls = calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| ToolCallOut {
            id: match call.id.is_empty() {
                true => format!("call_{index}"),
                false => call.id,
            },
            kind: "function",
            function: FunctionOut {
                name: call.name,
                arguments: call.arguments.to_string(),
            },
        })
        .collect::<Vec<_>>();
    let finish_reason = match (tool_calls.is_empty(), sampled.stopped_at_eog || truncated) {
        (false, _) => FinishReason::ToolCalls,
        (true, true) => FinishReason::Stop,
        (true, false) => FinishReason::Length,
    };
    let content = match (tool_calls.is_empty(), content.trim().is_empty()) {
        (false, true) => None,
        _ => Some(content),
    };
    // Widening: token counts of one request.
    let prompt_tokens = sampled.prompt_tokens as u64;
    let completion_tokens = sampled.completion_tokens as u64;
    Completion {
        content,
        tool_calls,
        finish_reason,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        parse_errors,
    }
}

/// The text up to the earliest match of any stop sequence, and whether one
/// matched.
fn truncate_at_stop<'a>(text: &'a str, stop: &[String]) -> (&'a str, bool) {
    match stop
        .iter()
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
    {
        Some(at) => (&text[..at], true),
        None => (text, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::prepare;
    use crate::testing::FakeModel;
    use serde_json::json;

    fn prompt(body: serde_json::Value) -> ChatPrompt {
        prepare(serde_json::from_value(body).unwrap())
            .unwrap()
            .prompt
    }

    fn run(model: &mut FakeModel, body: serde_json::Value) -> Completion {
        let prompt = prompt(body);
        let sampled = complete(model, &prompt, false, &mut ParserCache::default()).unwrap();
        finish(sampled, &prompt.stop)
    }

    #[test]
    fn a_natural_stop_is_stop_and_a_budget_is_length() {
        let mut model = FakeModel::new("hello there");
        let done = run(
            &mut model,
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert_eq!(done.content.as_deref(), Some("hello there"));
        assert_eq!(done.finish_reason, FinishReason::Stop);
        // Eleven bytes and the end-of-generation token.
        assert_eq!(done.usage.completion_tokens, 12);
        assert_eq!(
            done.usage.total_tokens,
            done.usage.prompt_tokens + done.usage.completion_tokens
        );

        let cut = run(
            &mut model,
            json!({"model": "m", "max_tokens": 1,
                   "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert_eq!(cut.content.as_deref(), Some("h"));
        assert_eq!(cut.finish_reason, FinishReason::Length);
    }

    #[test]
    fn a_stop_sequence_truncates_at_its_earliest_match() {
        // The budget is spent on "one. two!": both stops are in the sample,
        // the one listed first matches later, and the cut still reads as a
        // stop rather than as the budget.
        let mut model = FakeModel::new("one. two! three");
        let done = run(
            &mut model,
            json!({"model": "m", "max_tokens": 9, "stop": ["!", "."],
                   "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert_eq!(done.content.as_deref(), Some("one"));
        assert_eq!(done.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn a_prompt_that_does_not_fit_is_refused_before_sampling() {
        let mut model = FakeModel::new("x").with_context(16);
        let prompt = prompt(json!({"model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "a long enough question"}]}));
        let error = complete(&mut model, &prompt, false, &mut ParserCache::default())
            .err()
            .expect("refused");
        assert_eq!(error.code, Some("context_length_exceeded"));
        assert_eq!(model.samples(), 0);
    }

    /// A template blind to tools gets the catalog in its system turn, the
    /// observations in the `id: content` shape, and its answer read in the
    /// `<tool_call>` convention - the prompt rendering a rollout would use.
    #[test]
    fn a_call_is_read_back_in_the_convention_the_prompt_taught() {
        let mut model = FakeModel::new(
            r#"<tool_call>{"name":"move","arguments":{"direction":"left"}}</tool_call>"#,
        );
        let done = run(
            &mut model,
            json!({"model": "m",
                "tools": [{"type": "function", "function": {"name": "move"}}],
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "tool_calls": [{"id": "c0", "type": "function",
                        "function": {"name": "move", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c0", "content": "a wall"}]}),
        );
        assert_eq!(done.finish_reason, FinishReason::ToolCalls);
        assert_eq!(done.content, None);
        assert_eq!(done.tool_calls[0].function.name, "move");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&done.tool_calls[0].function.arguments)
                .unwrap(),
            json!({"direction": "left"})
        );
        let rendered: serde_json::Value =
            serde_json::from_str(&model.last_messages().unwrap()).unwrap();
        assert_eq!(rendered[0]["role"], "system");
        assert!(
            rendered[0]["content"]
                .as_str()
                .unwrap()
                .contains("Available tools")
        );
        assert_eq!(rendered[3]["content"], "c0: a wall");
        assert_eq!(
            rendered[2]["tool_calls"][0]["function"]["arguments"],
            json!({})
        );
        assert!(model.last_tools().is_none(), "the catalog is in the prompt");
    }

    #[test]
    fn tool_choice_none_keeps_the_catalog_out_of_the_prompt() {
        let mut model = FakeModel::new("no call");
        let done = run(
            &mut model,
            json!({"model": "m", "tool_choice": "none",
                "tools": [{"type": "function", "function": {"name": "move"}}],
                "messages": [{"role": "user", "content": "go"}]}),
        );
        assert_eq!(done.finish_reason, FinishReason::Stop);
        assert!(done.tool_calls.is_empty());
        assert!(model.last_tools().is_none());
        assert!(!model.last_messages().unwrap().contains("Available tools"));
    }

    #[test]
    fn an_unreadable_call_leaves_the_text_and_is_counted() {
        let mut model = FakeModel::new("<tool_call>{not json}</tool_call>");
        let done = run(
            &mut model,
            json!({"model": "m",
                "tools": [{"type": "function", "function": {"name": "move"}}],
                "messages": [{"role": "user", "content": "go"}]}),
        );
        assert_eq!(done.parse_errors, 1);
        assert!(done.tool_calls.is_empty());
        assert_eq!(done.finish_reason, FinishReason::Stop);
    }
}
