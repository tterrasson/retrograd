use retrograd_dataset::chat_template::{ParsedCalls, parse_hermes};

use crate::tools::{ToolCall, ToolResult};

pub use retrograd_dataset::chat_template::ToolCallParseError;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedAssistant {
    pub content: String,
    /// The turn's reasoning block, without its tags, as the template-derived
    /// parser split it out; empty when the turn did not reason.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    /// Invalid calls become observations so the policy can repair them on the
    /// next turn rather than aborting the whole trajectory.
    pub parse_errors: Vec<ToolResult>,
}

/// Each call that could not be read becomes the observation of its position,
/// `parse_error_{index}`, carrying the variant's text.
impl From<ParsedCalls> for ParsedAssistant {
    fn from(parsed: ParsedCalls) -> Self {
        Self {
            content: parsed.content,
            reasoning: parsed.reasoning,
            tool_calls: parsed.calls.into_iter().map(ToolCall::from).collect(),
            parse_errors: parsed
                .errors
                .into_iter()
                .map(|(index, error)| parse_error(index, error))
                .collect(),
        }
    }
}

/// Splits a model's raw generation into its prose content and the tool calls
/// it asked for, recovering as an observation ([`ParsedAssistant::parse_errors`])
/// whatever it could not read rather than failing the turn outright.
pub trait ToolCallParser: Send + Sync {
    fn parse(&self, output: &str) -> ParsedAssistant;

    /// What is recognized of a turn that is still arriving: the same split, but
    /// a call's arguments stay the raw string the parser produced, possibly
    /// incomplete - the streaming caller sends them as deltas and lets the
    /// final full [`Self::parse`] validate them.
    ///
    /// The default reads the whole text, which commits a trailing partial
    /// marker as content; a streaming caller holds its own buffer over such a
    /// prefix and never retracts what it already sent. Parsers that can do
    /// better override this.
    fn parse_partial(&self, output: &str) -> PartialAssistant {
        self.parse(output).into()
    }
}

/// What a parser recognized of an assistant turn still being generated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartialAssistant {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<PartialToolCall>,
}

/// A recognized call whose arguments may still be arriving.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartialToolCall {
    pub id: String,
    pub name: String,
    /// The raw arguments text, possibly incomplete.
    pub arguments: String,
}

impl From<ParsedAssistant> for PartialAssistant {
    fn from(parsed: ParsedAssistant) -> Self {
        Self {
            content: parsed.content,
            reasoning: parsed.reasoning,
            tool_calls: parsed
                .tool_calls
                .into_iter()
                .map(|call| PartialToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments.to_string(),
                })
                .collect(),
        }
    }
}

/// Parser for the `<tool_call>...</tool_call>` conventions.
///
/// Two bodies are accepted, because the family split one: the original
/// Hermes/Qwen2.5 form is a JSON object, while Qwen3's template instructs the
/// model to nest `<function=name>` and `<parameter=name>` tags instead. Which
/// one a model emits is decided by its own template, not by configuration, so
/// reading only one of them turns every tool call of the other family into a
/// parse error - and a rollout that cannot call a tool burns all its turns and
/// is thrown away as truncated.
///
/// This is the fallback, not the default. The default is
/// `TemplateToolCallParser`, which derives the parser from the model's own chat
/// template through the C ABI, so that rendering and parsing are one decision
/// rather than two that can disagree. Three cases keep this one alive: a
/// template blind to tools, whose catalog goes into the system prompt in the
/// `<tool_call>` convention above; a template no parser can be derived from; and
/// the tests, which fabricate their generations in this format.
///
/// The reading itself is [`parse_hermes`], which SFT on tool conversations
/// shares, so what a warm-start teaches is what a rollout reads back.
#[derive(Clone, Copy, Debug, Default)]
pub struct HermesToolCallParser;

impl ToolCallParser for HermesToolCallParser {
    fn parse(&self, output: &str) -> ParsedAssistant {
        parse_hermes(output).into()
    }

    fn parse_partial(&self, output: &str) -> PartialAssistant {
        parse_hermes_partial(output)
    }
}

const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";

/// Holds back an unfinished marker or call. The fallback has no incremental
/// argument grammar, so only closed calls are decoded, through the same reader
/// as a full turn. This preserves validated arguments, tag-form calls and ids.
fn parse_hermes_partial(output: &str) -> PartialAssistant {
    let hold = partial_marker_len(output);
    let text = &output[..output.len() - hold];
    let mut rest = text;
    let mut committed = text.len();
    while let Some(open) = rest.find(HERMES_OPEN) {
        let body_start = open + HERMES_OPEN.len();
        let Some(close) = rest[body_start..].find(HERMES_CLOSE) else {
            committed = text.len() - rest.len() + open;
            break;
        };
        rest = &rest[body_start + close + HERMES_CLOSE.len()..];
    }
    HermesToolCallParser.parse(&text[..committed]).into()
}

/// The length of the longest suffix of `text` that is a proper prefix of one
/// of the markers: the part that may still become a call and is therefore held
/// back rather than committed.
fn partial_marker_len(text: &str) -> usize {
    [HERMES_OPEN, HERMES_CLOSE]
        .iter()
        .filter_map(|marker| {
            (1..marker.len())
                .rev()
                .find(|&len| text.ends_with(&marker[..len]))
        })
        .max()
        .unwrap_or(0)
}

/// The observation a call that could not be read comes back as.
pub fn parse_error(index: usize, error: impl std::fmt::Display) -> ToolResult {
    ToolResult {
        call_id: format!("parse_error_{index}"),
        content: error.to_string(),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_read_holds_a_trailing_marker_prefix_back() {
        let parsed = HermesToolCallParser.parse_partial("answer <tool_ca");
        assert_eq!(parsed.content, "answer");
        assert!(parsed.tool_calls.is_empty());
        // An unfinished call stays buffered until its arguments can be decoded.
        let parsed = HermesToolCallParser.parse_partial("answer <tool_call>{\"name\":\"mo");
        assert_eq!(parsed.content, "answer");
        assert!(parsed.tool_calls.is_empty());
        // The envelope is never sent as if it were the function's arguments.
        let parsed =
            HermesToolCallParser.parse_partial("answer <tool_call>{\"name\":\"move\",\"argu");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn a_partial_read_of_a_complete_call_matches_the_full_read() {
        let full = "plan <tool_call>{\"name\":\"echo\",\"arguments\":{\"x\":1}}</tool_call> done";
        let partial = HermesToolCallParser.parse_partial(full);
        let parsed = HermesToolCallParser.parse(full);
        assert_eq!(partial.content, parsed.content);
        assert_eq!(partial.tool_calls.len(), parsed.tool_calls.len());
        assert_eq!(partial.tool_calls[0].name, parsed.tool_calls[0].name);
        assert_eq!(
            partial.tool_calls[0].arguments,
            parsed.tool_calls[0].arguments.to_string()
        );
        // Re-reading the whole prefix preserves positional ids, including an
        // invalid sibling, rather than decoding every block as call_0.
        let output = format!(
            "{full}<tool_call>{{broken}}</tool_call><tool_call>{{\"name\":\"next\"}}</tool_call>"
        );
        let partial = HermesToolCallParser.parse_partial(&output);
        let full = HermesToolCallParser.parse(&output);
        assert_eq!(partial, PartialAssistant::from(full));
        assert_eq!(partial.tool_calls[1].id, "call_2");
    }

    #[test]
    fn parses_multiple_calls_and_mixed_content() {
        let parsed = HermesToolCallParser.parse(
            "thinking <tool_call>{\"name\":\"echo\",\"arguments\":{\"x\":1}}</tool_call>\n\
             <tool_call>{\"id\":\"b\",\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>",
        );
        assert_eq!(parsed.content, "thinking");
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "call_0");
        assert_eq!(parsed.tool_calls[1].id, "b");
        assert!(parsed.parse_errors.is_empty());
    }

    #[test]
    fn malformed_json_becomes_a_tool_observation() {
        let parsed = HermesToolCallParser.parse("<tool_call>{bad}</tool_call>");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.parse_errors.len(), 1);
        assert!(parsed.parse_errors[0].is_error);
        assert!(
            parsed.parse_errors[0]
                .content
                .contains("invalid tool-call JSON")
        );
    }

    #[test]
    fn plain_prose_yields_no_calls_and_ends_the_trajectory() {
        let parsed = HermesToolCallParser.parse("  just an answer  ");
        assert_eq!(parsed.content, "just an answer");
        assert!(parsed.tool_calls.is_empty());
        assert!(parsed.parse_errors.is_empty());
    }

    #[test]
    fn an_unterminated_block_is_reported_and_keeps_its_text() {
        // Typically a turn cut off by the token budget: the opening tag must
        // stay in the content so the trajectory still detokenizes faithfully.
        let parsed = HermesToolCallParser.parse("before <tool_call>{\"name\":\"echo\"");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.parse_errors.len(), 1);
        assert!(parsed.parse_errors[0].content.contains("unterminated"));
        assert!(parsed.content.contains("<tool_call>"));
    }

    #[test]
    fn a_call_without_a_usable_name_becomes_an_observation() {
        for body in [
            "{\"arguments\":{}}",
            "{\"name\":\"\",\"arguments\":{}}",
            "{\"name\":\"echo\",\"arguments\":[]}",
            "[1,2]",
        ] {
            let parsed = HermesToolCallParser.parse(&format!("<tool_call>{body}</tool_call>"));
            assert!(parsed.tool_calls.is_empty(), "accepted {body}");
            assert_eq!(parsed.parse_errors.len(), 1, "no observation for {body}");
        }
    }

    #[test]
    fn the_tag_form_qwen3_templates_ask_for_is_read_as_a_call() {
        // Verbatim shape of what a Qwen3 template tells the model to emit, down
        // to the newlines around each value.
        let parsed = HermesToolCallParser.parse(
            "Let me look.<tool_call>\n\
             <function=bash>\n\
             <parameter=cmd>\n\
             ls -la /work\n\
             </parameter>\n\
             </function>\n\
             </tool_call>",
        );
        assert!(
            parsed.parse_errors.is_empty(),
            "{:?}",
            parsed.parse_errors[0].content
        );
        assert_eq!(parsed.content, "Let me look.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.tool_calls[0].id, "call_0");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({"cmd": "ls -la /work"})
        );
    }

    #[test]
    fn a_tag_form_value_keeps_its_own_newlines_and_its_own_type() {
        let parsed = HermesToolCallParser.parse(
            "<tool_call>\n<function=write_file>\n\
             <parameter=path>\nsolution.py\n</parameter>\n\
             <parameter=content>\ndef f():\n\n    return 1\n</parameter>\n\
             <parameter=lines>\n[1, 2]\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert!(parsed.parse_errors.is_empty());
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({
                "path": "solution.py",
                // Only the framing newlines are stripped; a blank line inside a
                // file the model is writing is part of the file.
                "content": "def f():\n\n    return 1",
                // An array was written as JSON by the template, so it comes back
                // as one. A scalar stays the string the template made of it.
                "lines": [1, 2],
            })
        );
    }

    #[test]
    fn a_tag_form_call_with_no_parameters_is_still_a_call() {
        let parsed =
            HermesToolCallParser.parse("<tool_call>\n<function=submit>\n</function>\n</tool_call>");
        assert!(parsed.parse_errors.is_empty());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "submit");
        assert_eq!(parsed.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn a_broken_tag_form_call_becomes_an_observation_too() {
        for body in [
            "<function=bash>\n<parameter=cmd>\nls\n</function>",
            "<function=>\n</function>",
            "<function=bash>\n<parameter=cmd>\nls\n</parameter>\n",
            "not a call at all",
        ] {
            let parsed = HermesToolCallParser.parse(&format!("<tool_call>{body}</tool_call>"));
            assert!(parsed.tool_calls.is_empty(), "accepted {body}");
            assert_eq!(parsed.parse_errors.len(), 1, "no observation for {body}");
        }
    }

    #[test]
    fn a_call_without_arguments_defaults_to_an_empty_object() {
        let parsed = HermesToolCallParser.parse("<tool_call>{\"name\":\"ping\"}</tool_call>");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments, serde_json::json!({}));
    }

    /// Each variant, and the exact observation the policy reads for it.
    #[test]
    fn every_parse_error_keeps_its_observation_text() {
        let json_error =
            || serde_json::from_str::<serde_json::Value>("{bad}").expect_err("invalid JSON");
        assert!(
            ToolCallParseError::InvalidJson(json_error())
                .to_string()
                .starts_with("invalid tool-call JSON: ")
        );
        assert!(
            ToolCallParseError::InvalidArguments(json_error())
                .to_string()
                .starts_with("invalid tool-call arguments: ")
        );
        for (error, text) in [
            (
                ToolCallParseError::UnterminatedToolCall,
                "unterminated <tool_call> block",
            ),
            (
                ToolCallParseError::NotAnObject,
                "tool call must be a JSON object",
            ),
            (
                ToolCallParseError::MissingName,
                "tool call requires a non-empty string name",
            ),
            (
                ToolCallParseError::ArgumentsNotAnObject,
                "tool call arguments must be a JSON object",
            ),
            (
                ToolCallParseError::UnknownForm,
                "tool call must be a JSON object or a <function=name> block",
            ),
            (
                ToolCallParseError::UnterminatedFunctionTag,
                "unterminated <function= tag",
            ),
            (
                ToolCallParseError::MissingFunctionName,
                "tool call requires a non-empty function name",
            ),
            (
                ToolCallParseError::UnterminatedFunctionBlock,
                "unterminated <function> block",
            ),
            (
                ToolCallParseError::UnterminatedParameterTag,
                "unterminated <parameter= tag",
            ),
            (
                ToolCallParseError::MissingParameterName,
                "tool call parameter requires a non-empty name",
            ),
            (
                ToolCallParseError::UnterminatedParameterBlock("cmd".to_owned()),
                "unterminated <parameter=cmd> block",
            ),
        ] {
            assert_eq!(error.to_string(), text);
        }
    }

    /// The observation is the variant's text, under the positional id.
    #[test]
    fn a_parse_error_becomes_the_observation_of_its_call() {
        let parsed = HermesToolCallParser.parse(
            "<tool_call>{\"name\":\"a\"}</tool_call><tool_call><function=></function></tool_call>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.parse_errors.len(), 1);
        assert_eq!(parsed.parse_errors[0].call_id, "parse_error_1");
        assert_eq!(
            parsed.parse_errors[0].content,
            ToolCallParseError::MissingFunctionName.to_string()
        );
    }
}
