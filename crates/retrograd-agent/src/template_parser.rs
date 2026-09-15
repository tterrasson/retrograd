//! Reading an assistant turn back in the format the model's own template
//! taught it.
//!
//! [`HermesToolCallParser`](crate::tools::HermesToolCallParser) reads one
//! format, `<tool_call>{…}</tool_call>`, and retrograd renders in whichever
//! format the model's template writes. The pair only agrees for the families
//! whose template happens to be the Hermes one; for LFM2,
//! `<|tool_call_start|>[move(direction='left')]<|tool_call_end|>` - it does not,
//! and the disagreement is silent: no call, no parse error, a trajectory that
//! never touches its environment.
//!
//! [`TemplateToolCallParser`] closes that by deriving the parser from the same
//! template the rendering comes from, so the two halves cannot be chosen
//! separately. The derivation itself lives in llama.cpp and is reached through
//! [`retrograd_engine::parse_assistant_output`]; what is here is the
//! translation into [`ParsedAssistant`].
//!
//! The type is deliberately not in `retrograd-tools`, which stays free of the
//! FFI bridge.

use retrograd_agent_core::tools::{ToolCall, ToolResult};

use crate::tools::{ParsedAssistant, ToolCallParseError, ToolCallParser};

/// A serialized PEG parser produced from one model's chat template and one tool
/// catalog, ready to run on any thread.
pub struct TemplateToolCallParser {
    parser: String,
}

impl TemplateToolCallParser {
    pub fn new(parser: String) -> Self {
        Self { parser }
    }
}

impl std::fmt::Debug for TemplateToolCallParser {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The blob is a few kilobytes of grammar; its size is the only part of
        // it worth reading in a log line.
        formatter
            .debug_struct("TemplateToolCallParser")
            .field("parser_bytes", &self.parser.len())
            .finish()
    }
}

impl ToolCallParser for TemplateToolCallParser {
    fn parse(&self, output: &str) -> ParsedAssistant {
        let document = match retrograd_engine::parse_assistant_output(&self.parser, output) {
            Ok(document) => document,
            // The generation did not match the template's own format. That is
            // an observation the policy can act on next turn, not a lost
            // trajectory - same treatment the Hermes parser gives a malformed
            // call. The raw text stays the content so the turn still
            // detokenizes faithfully.
            Err(error) => {
                return ParsedAssistant {
                    content: output.to_owned(),
                    tool_calls: Vec::new(),
                    parse_errors: vec![parse_error(0, error)],
                };
            }
        };
        decode(&document).unwrap_or_else(|error| ParsedAssistant {
            content: output.to_owned(),
            tool_calls: Vec::new(),
            parse_errors: vec![parse_error(0, error)],
        })
    }
}

/// Why the runtime's document could not be read at all - a runtime defect, not
/// a model output.
#[derive(Debug, thiserror::Error)]
enum DocumentError {
    #[error("invalid parsed-assistant document: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("parsed assistant must be a JSON object")]
    NotAnObject,
    #[error("parsed assistant must carry a tool_calls array")]
    NoToolCalls,
}

/// Turns the runtime's JSON into a [`ParsedAssistant`].
///
/// Only a malformed *document* is an `Err` here - that would be a runtime bug,
/// not a model output. A malformed `arguments` string is the model's doing and
/// becomes one observation, leaving the other calls of the same turn usable.
fn decode(document: &str) -> Result<ParsedAssistant, DocumentError> {
    let value: serde_json::Value =
        serde_json::from_str(document).map_err(DocumentError::InvalidJson)?;
    let object = value.as_object().ok_or(DocumentError::NotAnObject)?;
    let field = |name: &str| {
        object
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    };
    // Reasoning is left in the content by the parser this crate builds, but a
    // template whose parser splits it anyway must not lose the text.
    let mut content = field("reasoning_content").to_owned();
    content.push_str(field("content"));

    let mut parsed = ParsedAssistant {
        content: content.trim().to_owned(),
        tool_calls: Vec::new(),
        parse_errors: Vec::new(),
    };
    let calls = object
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .ok_or(DocumentError::NoToolCalls)?;
    for (index, call) in calls.iter().enumerate() {
        match decode_call(call, index) {
            Ok(call) => parsed.tool_calls.push(call),
            Err(error) => parsed.parse_errors.push(parse_error(index, error)),
        }
    }
    Ok(parsed)
}

fn decode_call(call: &serde_json::Value, index: usize) -> Result<ToolCall, ToolCallParseError> {
    let object = call.as_object().ok_or(ToolCallParseError::NotAnObject)?;
    let name = object
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or(ToolCallParseError::MissingName)?;
    let raw = object
        .get("arguments")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim();
    // An absent argument list is an empty one: a template may render a
    // no-argument call with nothing between its markers.
    let arguments = match raw.is_empty() {
        true => serde_json::Value::Object(Default::default()),
        false => serde_json::from_str(raw).map_err(ToolCallParseError::InvalidArguments)?,
    };
    if !arguments.is_object() {
        return Err(ToolCallParseError::ArgumentsNotAnObject);
    }
    // Templates that carry no call id give the same positional one the Hermes
    // parser falls back to, so a tool result can be matched to its call either
    // way.
    let id = object
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_{index}"));
    Ok(ToolCall {
        id,
        name: name.to_owned(),
        arguments,
    })
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

    // The FFI half needs a model; what is testable without one is the
    // translation, which is where a call can silently go missing.
    #[test]
    fn a_call_keeps_its_name_arguments_and_id() {
        let parsed = decode(
            r#"{"content":"looking","reasoning_content":"",
                "tool_calls":[{"id":"","name":"move","arguments":"{\"direction\":\"left\"}"}]}"#,
        )
        .expect("well-formed document");
        assert_eq!(parsed.content, "looking");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "move");
        assert_eq!(parsed.tool_calls[0].id, "call_0");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({"direction": "left"})
        );
        assert!(parsed.parse_errors.is_empty());
    }

    #[test]
    fn reasoning_stays_in_the_content_rather_than_being_dropped() {
        let parsed =
            decode(r#"{"content":"answer","reasoning_content":"thinking ","tool_calls":[]}"#)
                .expect("well-formed document");
        assert_eq!(parsed.content, "thinking answer");
    }

    #[test]
    fn a_call_with_no_arguments_is_still_a_call() {
        let parsed = decode(r#"{"content":"","tool_calls":[{"name":"submit","arguments":""}]}"#)
            .expect("well-formed document");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn one_malformed_call_becomes_an_observation_and_leaves_its_siblings() {
        let parsed = decode(
            r#"{"content":"","tool_calls":[
                {"name":"move","arguments":"{oops"},
                {"name":"look","arguments":"{}"}]}"#,
        )
        .expect("well-formed document");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "look");
        assert_eq!(parsed.parse_errors.len(), 1);
        assert!(parsed.parse_errors[0].is_error);
        assert!(
            parsed.parse_errors[0]
                .content
                .contains("invalid tool-call arguments")
        );
    }

    #[test]
    fn a_document_that_is_not_one_is_reported_rather_than_guessed_at() {
        for (document, text) in [
            ("not json", "invalid parsed-assistant document: "),
            ("[]", "parsed assistant must be a JSON object"),
            (
                r#"{"content":"x"}"#,
                "parsed assistant must carry a tool_calls array",
            ),
        ] {
            let error = decode(document).expect_err(document);
            assert!(error.to_string().starts_with(text), "{document}: {error}");
        }
    }

    /// The calls a template yields share the Hermes parser's vocabulary, so the
    /// policy reads the same observation whichever parser produced it.
    #[test]
    fn a_rejected_call_reads_as_the_shared_parse_error() {
        let parsed = decode(
            r#"{"content":"","tool_calls":[{"name":"","arguments":""},{"name":"a","arguments":"[]"}]}"#,
        )
        .expect("well-formed document");
        let observations: Vec<&str> = parsed
            .parse_errors
            .iter()
            .map(|error| error.content.as_str())
            .collect();
        assert_eq!(
            observations,
            [
                ToolCallParseError::MissingName.to_string(),
                ToolCallParseError::ArgumentsNotAnObject.to_string(),
            ]
        );
    }
}
