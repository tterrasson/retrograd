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

use retrograd_agent_core::TemplateToolCallParser;
use retrograd_agent_core::chat;
use retrograd_agent_core::parser::{HermesToolCallParser, ToolCallParser};
use retrograd_agent_core::rendering::{
    ToolRendering, inject_tool_instructions, observation_message, resolve_tool_rendering_with,
};
use retrograd_agent_core::tools::ToolResult;
use retrograd_agent_core::trajectory::{Message, Role};
use retrograd_core::{Result as CoreResult, SamplingParams, SamplingParamsV2};
use retrograd_engine::Trainer;

use crate::chat::wire::{self, FinishReason, FunctionOut, ToolCallOut, Usage};
use crate::error::OpenAiError;
use crate::request::{GenerationRequest, ToolConstraint};

/// The read-only model queries prompt rendering needs: the same ones a serving
/// backend ([`crate::backend::ServeBackend`]) answers on the thread that owns
/// the weights. [`ChatModel`] adds sampling on top for the callers that sample
/// in place - a live run's trainer.
pub trait PromptModel {
    fn supports_native_tools(&self) -> CoreResult<bool>;
    /// The parser the model's template yields for this catalog, if any.
    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>>;
    /// The GBNF grammar the model's template yields for this catalog under
    /// the requested constraint, if the template can express it.
    fn tool_call_grammar(
        &self,
        tools_json: &str,
        constraint: &ToolConstraint,
    ) -> CoreResult<Option<String>>;
    /// The generation prompt for a conversation, through the model's own
    /// template.
    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String>;
    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>>;
    fn context_size(&self) -> CoreResult<usize>;
}

/// A boxed prompt model is a prompt model.
impl<T: PromptModel + ?Sized> PromptModel for Box<T> {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        (**self).supports_native_tools()
    }
    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>> {
        (**self).tool_call_parser(tools_json)
    }
    fn tool_call_grammar(
        &self,
        tools_json: &str,
        constraint: &ToolConstraint,
    ) -> CoreResult<Option<String>> {
        (**self).tool_call_grammar(tools_json, constraint)
    }
    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        (**self).format_chat(messages_json, tools_json)
    }
    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        (**self).tokenize(text)
    }
    fn context_size(&self) -> CoreResult<usize> {
        (**self).context_size()
    }
}

/// The part of a model a chat request needs. [`Trainer`] is the one that
/// matters; the trait is what lets the whole request path run in the fast lane
/// against [`crate::testing::FakeModel`].
pub trait ChatModel: PromptModel {
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

impl PromptModel for Trainer {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        self.chat_template_supports_tools()
    }

    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>> {
        Trainer::tool_call_parser(self, Some(tools_json))
    }

    fn tool_call_grammar(
        &self,
        tools_json: &str,
        constraint: &ToolConstraint,
    ) -> CoreResult<Option<String>> {
        let (mode, forced) = match constraint {
            ToolConstraint::Required => (0, None),
            ToolConstraint::Forced(name) => (1, Some(name.as_str())),
            ToolConstraint::AtMostOne => (2, None),
        };
        Trainer::tool_call_grammar(self, tools_json, mode, forced)
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
}

impl ChatModel for Trainer {
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
    fn parser(&mut self, model: &dyn PromptModel, tools_json: &str) -> CoreResult<Option<String>> {
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
    /// The per-token logprobs the client asked for, when it did: one entry
    /// per sampled token, in order.
    pub logprobs: Option<Vec<SampledLogprob>>,
    /// How many sampled tokens the parser attributed to the reasoning block.
    pub reasoning_tokens: usize,
}

/// One sampled token's logprob entry: the UTF-8 bytes of its piece, its
/// logprob under the post-transformation distribution, and the top pairs of
/// the same position.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledLogprob {
    pub piece: Vec<u8>,
    pub logprob: f32,
    pub top: Vec<(i32, f32, Vec<u8>)>,
}

/// What rendering a conversation produced: the prompt tokens, the tool-call
/// reader the rendering chose, whether special tokens must survive
/// detokenization for that reader to work, and the per-choice sampling
/// parameters (one prefill, one copy per choice; a tool constraint or a
/// `response_format` grammar attached where the rendering decided it).
pub struct Rendered {
    pub tokens: Vec<i32>,
    pub reader: Option<Arc<dyn ToolCallParser>>,
    pub unparse_special: bool,
    pub sampling: Vec<SamplingParamsV2>,
}

/// Renders one conversation into the prompt the model is asked: the tools
/// decision, the template, the tokens, and the context bound. Runs wherever the
/// model lives - on the serving session's thread for a
/// [`crate::backend::ServeBackend`], on a live run's thread for a
/// [`Trainer`].
pub fn render_prompt(
    model: &dyn PromptModel,
    request: &GenerationRequest,
    parsers: &mut ParserCache,
) -> Result<Rendered, OpenAiError> {
    let tools_json = match request.tools.is_empty() {
        true => None,
        false => Some(chat::template_tools(&request.tools)?),
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
                request.tools.clone(),
            )?
        }
    };

    let mut messages = request
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
    let requested = request.sampling.max_new_tokens;
    if tokens.len().saturating_add(requested as usize) > context {
        return Err(OpenAiError::context_length_exceeded(
            tokens.len(),
            requested,
            context,
        ));
    }
    let reader: Option<Arc<dyn ToolCallParser>> = if request.sampling.grammar_gbnf.is_some() {
        // Structured output constrains the emitted bytes directly; a template
        // parser may otherwise treat them as an implicitly opened thought.
        None
    } else {
        match &rendering {
            ToolRendering::None => parsers
                .parser(model, "[]")?
                .map(|blob| Arc::new(TemplateToolCallParser::new(blob)) as Arc<dyn ToolCallParser>),
            ToolRendering::Native { parser, .. } => Some(parser.clone()),
            ToolRendering::Prompt { .. } => Some(Arc::new(HermesToolCallParser)),
        }
    };
    // A template-derived parser matches the model's own call delimiters, and
    // those are control tokens: they have to survive detokenization for it to
    // read anything. Every other reader wants the text a person reads.
    let unparse_special = matches!(rendering, ToolRendering::Native { .. })
        || matches!(rendering, ToolRendering::None) && reader.is_some();
    // One sampling entry per choice: same parameters, a derived seed, so the
    // choices differ instead of repeating one sample `n` times.
    let mut sampling: Vec<SamplingParamsV2> = (0..request.n.max(1))
        .map(|index| request.sampling.for_choice(index))
        .collect();
    // A tool constraint becomes a grammar derived from the template's own
    // call format - the only way to force or limit calls without deleting
    // them after the fact. It is mutually exclusive with a `response_format`
    // grammar (refused at conversion), so it simply replaces it here.
    if let (Some(constraint), Some(tools_json)) = (&request.tool_constraint, &tools_json) {
        let grammar = model.tool_call_grammar(tools_json, constraint)?;
        let Some(gbnf) = grammar else {
            return Err(OpenAiError::unsupported(
                "tool_choice",
                "the model's chat template cannot express this tool constraint as a grammar; \
                 send \"auto\" or omit it",
            ));
        };
        let gbnf: Arc<str> = gbnf.into();
        for params in &mut sampling {
            params.grammar_gbnf = Some(gbnf.clone());
        }
    }
    Ok(Rendered {
        tokens,
        reader,
        unparse_special,
        sampling,
    })
}

/// Renders, samples and detokenizes one request. Runs on the thread that owns
/// the model.
///
/// The live-run path samples through the rollout generator, which is the v1
/// contract (temperature + top_p): a request asking for extended sampling or
/// for several choices is refused here rather than silently narrowed.
pub fn complete(
    model: &mut dyn ChatModel,
    request: &GenerationRequest,
    base: bool,
    parsers: &mut ParserCache,
) -> Result<Sampled, OpenAiError> {
    if request.n > 1 {
        return Err(OpenAiError::unsupported(
            "n",
            "a live run answers one choice per request; `n > 1` is served by the weights on disk",
        ));
    }
    if request.tool_constraint.is_some() {
        return Err(OpenAiError::unsupported(
            "tool_choice",
            "constrained tool calls are served by the weights on disk",
        ));
    }
    if request.reasoning_budget.is_some() {
        return Err(OpenAiError::unsupported(
            "reasoning.effort",
            "per-request reasoning budgets are served by the weights on disk",
        ));
    }
    if !request.sampling.is_v1() {
        return Err(OpenAiError::unsupported(
            "sampling",
            "a live run samples with its rollout parameters (temperature, top_p); extended \
             sampling is served by the weights on disk",
        ));
    }
    if request.logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "logprobs",
            "a live run answers from its rollout generator, which does not report token \
             logprobs; logprobs are served by the weights on disk",
        ));
    }
    let Rendered {
        tokens,
        reader,
        unparse_special,
        ..
    } = render_prompt(model, request, parsers)?;

    let rollout = SamplingParams {
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        max_new_tokens: request.sampling.max_new_tokens,
        seed: request.sampling.seed,
    };
    let mut sampled = model.sample(&tokens, &rollout, base)?;
    let completion_tokens = sampled.len();
    let mut stopped_at_eog = false;
    while let Some(&last) = sampled.last()
        && model.is_eog(last)?
    {
        sampled.pop();
        stopped_at_eog = true;
    }
    Ok(Sampled {
        text: model.detokenize(&sampled, unparse_special)?,
        prompt_tokens: tokens.len(),
        completion_tokens,
        stopped_at_eog,
        reader,
        logprobs: None,
        reasoning_tokens: 0,
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
    /// The logprobs of the sampled tokens, when the request asked for them.
    pub logprobs: Option<wire::Logprobs>,
    /// The reasoning block the parser split out of the turn, tags removed;
    /// empty when the turn did not reason or no parser was chosen.
    pub reasoning: String,
    /// How many sampled tokens the parser attributed to that block.
    pub reasoning_tokens: usize,
}

/// Reads the answer out of what was sampled: stop sequences first, then the
/// calls. Runs off the model's thread.
pub fn finish(sampled: Sampled, stop: &[String]) -> Completion {
    let (text, truncated) = truncate_at_stop(&sampled.text, stop);
    let (content, calls, parse_errors, reasoning) = match &sampled.reader {
        Some(reader) => {
            let parsed = reader.parse(text);
            (
                parsed.content,
                parsed.tool_calls,
                parsed.parse_errors.len(),
                parsed.reasoning,
            )
        }
        None => (text.to_owned(), Vec::new(), 0, String::new()),
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
    let finish_reason = match (sampled.stopped_at_eog || truncated, tool_calls.is_empty()) {
        (false, _) => FinishReason::Length,
        (true, false) => FinishReason::ToolCalls,
        (true, true) => FinishReason::Stop,
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
        reasoning,
        reasoning_tokens: sampled.reasoning_tokens,
        logprobs: sampled.logprobs.map(|entries| wire::Logprobs {
            content: entries
                .into_iter()
                .map(|entry| wire::TokenLogprobs {
                    token: String::from_utf8_lossy(&entry.piece).into_owned(),
                    logprob: entry.logprob,
                    bytes: entry.piece,
                    top_logprobs: entry
                        .top
                        .into_iter()
                        .map(|(_, logprob, piece)| wire::TopLogprob {
                            token: String::from_utf8_lossy(&piece).into_owned(),
                            logprob,
                            bytes: piece,
                        })
                        .collect(),
                })
                .collect(),
        }),
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

/// Zero-mass tokens have logprob -infinity. JSON has no infinity; the wire
/// contract uses the low-probability sentinel instead of serializing null.
pub(crate) fn serialize_logprob<S: serde::Serializer>(
    value: &f32,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_f32(if value.is_finite() { *value } else { -9999.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::convert::prepare;
    use crate::testing::FakeModel;
    use serde_json::json;

    fn request(body: serde_json::Value) -> GenerationRequest {
        prepare(serde_json::from_value(body).unwrap())
            .unwrap()
            .request
    }

    fn run(model: &mut FakeModel, body: serde_json::Value) -> Completion {
        let request = request(body);
        let sampled = complete(model, &request, false, &mut ParserCache::default()).unwrap();
        finish(sampled, &request.stop)
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
        let request = request(json!({"model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "a long enough question"}]}));
        let error = complete(&mut model, &request, false, &mut ParserCache::default())
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
