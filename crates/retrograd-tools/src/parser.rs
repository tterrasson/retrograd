use retrograd_agent_core::tools::{ToolCall, ToolResult};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedAssistant {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Invalid calls become observations so the policy can repair them on the
    /// next turn rather than aborting the whole trajectory.
    pub parse_errors: Vec<ToolResult>,
}

/// Splits a model's raw generation into its prose content and the tool calls
/// it asked for, recovering as an observation ([`ParsedAssistant::parse_errors`])
/// whatever it could not read rather than failing the turn outright.
pub trait ToolCallParser: Send + Sync {
    fn parse(&self, output: &str) -> ParsedAssistant;
}

/// Why one tool call could not be read.
///
/// The message is what the policy reads back as an observation, so it is part
/// of the training data: a variant's wording does not change without a reason.
#[derive(Debug, thiserror::Error)]
pub enum ToolCallParseError {
    #[error("unterminated <tool_call> block")]
    UnterminatedToolCall,
    #[error("invalid tool-call JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("invalid tool-call arguments: {0}")]
    InvalidArguments(#[source] serde_json::Error),
    #[error("tool call must be a JSON object")]
    NotAnObject,
    #[error("tool call requires a non-empty string name")]
    MissingName,
    #[error("tool call arguments must be a JSON object")]
    ArgumentsNotAnObject,
    #[error("tool call must be a JSON object or a <function=name> block")]
    UnknownForm,
    #[error("unterminated <function= tag")]
    UnterminatedFunctionTag,
    #[error("tool call requires a non-empty function name")]
    MissingFunctionName,
    #[error("unterminated <function> block")]
    UnterminatedFunctionBlock,
    #[error("unterminated <parameter= tag")]
    UnterminatedParameterTag,
    #[error("tool call parameter requires a non-empty name")]
    MissingParameterName,
    #[error("unterminated <parameter={0}> block")]
    UnterminatedParameterBlock(String),
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
#[derive(Clone, Copy, Debug, Default)]
pub struct HermesToolCallParser;

impl ToolCallParser for HermesToolCallParser {
    fn parse(&self, output: &str) -> ParsedAssistant {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";

        let mut parsed = ParsedAssistant::default();
        let mut remainder = output;
        let mut call_index = 0_usize;
        while let Some(open) = remainder.find(OPEN) {
            parsed.content.push_str(&remainder[..open]);
            let body_start = open + OPEN.len();
            let Some(relative_close) = remainder[body_start..].find(CLOSE) else {
                parsed.parse_errors.push(parse_error(
                    call_index,
                    ToolCallParseError::UnterminatedToolCall,
                ));
                parsed.content.push_str(&remainder[open..]);
                remainder = "";
                break;
            };
            let close = body_start + relative_close;
            let body = remainder[body_start..close].trim();
            match parse_call(body, call_index) {
                Ok(call) => parsed.tool_calls.push(call),
                Err(error) => parsed.parse_errors.push(parse_error(call_index, error)),
            }
            call_index += 1;
            remainder = &remainder[close + CLOSE.len()..];
        }
        parsed.content.push_str(remainder);
        parsed.content = parsed.content.trim().to_owned();
        parsed
    }
}

fn parse_call(body: &str, index: usize) -> Result<ToolCall, ToolCallParseError> {
    match body.starts_with('{') {
        true => parse_json_call(body, index),
        false => parse_tagged_call(body, index),
    }
}

fn parse_json_call(body: &str, index: usize) -> Result<ToolCall, ToolCallParseError> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(ToolCallParseError::InvalidJson)?;
    let object = value.as_object().ok_or(ToolCallParseError::NotAnObject)?;
    let name = object
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or(ToolCallParseError::MissingName)?;
    let arguments = object
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    if !arguments.is_object() {
        return Err(ToolCallParseError::ArgumentsNotAnObject);
    }
    let id = object
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_{index}"));
    Ok(ToolCall {
        id,
        name: name.to_owned(),
        arguments,
    })
}

/// Reads the tag form: `<function=name>` wrapping one `<parameter=name>` block
/// per argument. The call carries no id of its own, so it gets the positional
/// one the JSON form falls back to.
fn parse_tagged_call(body: &str, index: usize) -> Result<ToolCall, ToolCallParseError> {
    let after = body
        .strip_prefix("<function=")
        .ok_or(ToolCallParseError::UnknownForm)?;
    let (name, body) = after
        .split_once('>')
        .ok_or(ToolCallParseError::UnterminatedFunctionTag)?;
    if name.is_empty() {
        return Err(ToolCallParseError::MissingFunctionName);
    }
    let body = body
        .trim_end()
        .strip_suffix("</function>")
        .ok_or(ToolCallParseError::UnterminatedFunctionBlock)?;

    let mut arguments = serde_json::Map::new();
    let mut remainder = body;
    while let Some(open) = remainder.find("<parameter=") {
        let after = &remainder[open + "<parameter=".len()..];
        let (argument, after) = after
            .split_once('>')
            .ok_or(ToolCallParseError::UnterminatedParameterTag)?;
        if argument.is_empty() {
            return Err(ToolCallParseError::MissingParameterName);
        }
        let (value, after) = after
            .split_once("</parameter>")
            .ok_or_else(|| ToolCallParseError::UnterminatedParameterBlock(argument.to_owned()))?;
        arguments.insert(argument.to_owned(), parameter_value(value));
        remainder = after;
    }
    Ok(ToolCall {
        id: format!("call_{index}"),
        name: name.to_owned(),
        arguments: serde_json::Value::Object(arguments),
    })
}

/// Recovers a parameter's type the way the template encoded it: an object or an
/// array was written as JSON, and everything else - including a number or a
/// boolean - was stringified. Inverting more than that would turn a path named
/// `123` into an integer.
fn parameter_value(raw: &str) -> serde_json::Value {
    // The template puts the value on its own lines, so exactly one newline on
    // each side belongs to the framing rather than to the value.
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    let trimmed = value.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(json) = serde_json::from_str(trimmed)
    {
        return json;
    }
    serde_json::Value::String(value.to_owned())
}

fn parse_error(index: usize, error: impl std::fmt::Display) -> ToolResult {
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
