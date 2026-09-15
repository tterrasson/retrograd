//! The JSON an HF-style chat template reads.
//!
//! `(role, content)` pairs are all a template needs for plain chat, and all the
//! renderer receives. A model trained with tools declares them in its own
//! template - a tool catalog, `tool_calls` on an assistant turn, a `tool` role
//! for what comes back - and rendering none of that means training the model on
//! a format it never saw during pre-training. This module builds what the
//! template expects instead.

use serde_json::{Map, Value, json};

use crate::tools::ToolSpec;
use crate::trajectory::{Message, Role};
use crate::{Error, Result};

/// The placeholder an assistant turn's sampled text is replaced by before the
/// template sees it.
///
/// Deliberately plain: no whitespace at either edge and no markup, so a template
/// that trims content or splits it on `</think>` - Qwen's does both - passes it
/// through byte for byte and stays findable in the output. Suffixed rather than
/// prefixed with the index so no sentinel is a prefix of another.
fn assistant_span(index: usize) -> String {
    format!("retro_span_{index}_9d41c7")
}

/// Serializes a conversation for the model's chat template, replacing every
/// assistant turn's content with a sentinel. Returns the JSON and the sentinels
/// in turn order.
///
/// The template never sees a sampled turn, on purpose. It cannot be trusted with
/// one: it is free to trim the content, to lift a `<think>` block out of it and
/// re-emit it somewhere else, or to render its own serialization of the tool
/// calls - and even a template that does none of that only gets the *text* back,
/// while what is being trained on is the *tokens*, which sampling does not
/// produce in canonical form (a model that sampled `"]]` + `])` re-tokenizes as
/// `"]` + `]])`). Splitting the rendered text on the sentinels yields the framing
/// around the sampled turns, which is the only part of a multi-turn prompt the
/// template gets to decide.
///
/// An assistant turn's `tool_calls` are not handed over either, for the same
/// reason: the calls are already inside the sampled tokens.
pub fn template_messages(messages: &[Message]) -> Result<(String, Vec<String>)> {
    let mut sentinels = Vec::new();
    let rendered = messages
        .iter()
        .map(|message| {
            let content = if message.role == Role::Assistant {
                let sentinel = assistant_span(sentinels.len());
                sentinels.push(sentinel.clone());
                sentinel
            } else {
                message.content.clone()
            };
            let mut object = Map::new();
            object.insert("role".into(), Value::String(message.role.as_str().into()));
            object.insert("content".into(), Value::String(content));
            if let Some(call_id) = &message.tool_call_id {
                object.insert("tool_call_id".into(), Value::String(call_id.clone()));
            }
            Value::Object(object)
        })
        .collect::<Vec<_>>();
    let json = serde_json::to_string(&rendered)
        .map_err(|error| Error::invalid(format!("serialize chat messages: {error}")))?;
    Ok((json, sentinels))
}

/// Cuts a rendered conversation into the framing around the sampled turns.
///
/// Returns `sentinels.len() + 1` pieces: what the template put before the first
/// sampled turn, between consecutive ones, and after the last. The sentinels are
/// consumed in order, so a template that reorders or drops an assistant turn is
/// caught here rather than corrupting the token stream downstream.
pub fn split_assistant_spans(rendered: &str, sentinels: &[String]) -> Result<Vec<String>> {
    let mut segments = Vec::with_capacity(sentinels.len() + 1);
    let mut rest = rendered;
    for sentinel in sentinels {
        let Some((before, after)) = rest.split_once(sentinel.as_str()) else {
            return Err(Error::invalid(format!(
                "chat template did not render assistant turn {} of {}: a template that drops or \
                 reorders a turn cannot be used for multi-turn rollouts",
                segments.len(),
                sentinels.len()
            )));
        };
        segments.push(before.to_string());
        rest = after;
    }
    if let Some(sentinel) = sentinels
        .iter()
        .find(|sentinel| rest.contains(&***sentinel))
    {
        return Err(Error::invalid(format!(
            "chat template rendered the assistant span sentinel {sentinel} more than once"
        )));
    }
    segments.push(rest.to_string());
    Ok(segments)
}

/// Serializes the tool catalog in the OpenAI function shape, the one HF chat
/// templates iterate over.
pub fn template_tools(tools: &[ToolSpec]) -> Result<String> {
    let rendered = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&rendered)
        .map_err(|error| Error::invalid(format!("serialize tool definitions: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCall;
    use crate::trajectory::Role;

    #[test]
    fn an_observation_carries_its_call_id_and_a_sampled_action_never_reaches_the_template() {
        let messages = vec![
            Message::text(Role::User, "hello"),
            Message {
                role: Role::Assistant,
                content: "<tool_call>{\"name\":\"echo\"}</tool_call>".into(),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "echo".into(),
                    arguments: json!({"text": "hi"}),
                }],
                tool_call_id: None,
                is_error: false,
            },
            Message {
                role: Role::Tool,
                content: "hi".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("call-1".into()),
                is_error: false,
            },
        ];
        let (json, sentinels) = template_messages(&messages).unwrap();
        let rendered: Value = serde_json::from_str(&json).unwrap();
        let rendered = rendered.as_array().unwrap();
        assert_eq!(rendered[0]["role"], "user");
        assert_eq!(
            sentinels.len(),
            1,
            "one sentinel per assistant turn, in turn order"
        );
        assert_eq!(
            rendered[1]["content"], sentinels[0],
            "the sampled text must be withheld from the template: it can trim it, restructure it, \
             or hand back a re-tokenization of it that is not what was sampled"
        );
        assert!(
            rendered[1].get("tool_calls").is_none(),
            "re-rendering the sampled calls would replace or duplicate trained tokens"
        );
        assert_eq!(rendered[2]["role"], "tool");
        assert_eq!(rendered[2]["tool_call_id"], "call-1");
    }

    #[test]
    fn framing_is_the_rendered_text_minus_the_sampled_turns() {
        let messages = vec![
            Message::text(Role::User, "hello"),
            Message::text(Role::Assistant, "sampled one"),
            Message::text(Role::Tool, "observation"),
            Message::text(Role::Assistant, "sampled two"),
        ];
        let (_, sentinels) = template_messages(&messages).unwrap();
        let rendered = format!(
            "<user>hello</user><a>{}</a><tool>observation</tool><a>{}</a><gen>",
            sentinels[0], sentinels[1]
        );
        assert_eq!(
            split_assistant_spans(&rendered, &sentinels).unwrap(),
            vec![
                "<user>hello</user><a>",
                "</a><tool>observation</tool><a>",
                "</a><gen>",
            ]
        );
    }

    #[test]
    fn a_template_that_drops_an_assistant_turn_is_refused() {
        let sentinels = vec!["span-0".to_string(), "span-1".to_string()];
        let error = split_assistant_spans("head span-1 tail", &sentinels).unwrap_err();
        assert!(error.to_string().contains("did not render assistant turn"));
    }

    #[test]
    fn the_catalog_uses_the_shape_chat_templates_iterate_over() {
        let tools = vec![ToolSpec {
            name: "echo".into(),
            description: "echoes".into(),
            input_schema: json!({"type": "object", "properties": {}}),
        }];
        let rendered: Value = serde_json::from_str(&template_tools(&tools).unwrap()).unwrap();
        assert_eq!(rendered[0]["type"], "function");
        assert_eq!(rendered[0]["function"]["name"], "echo");
        assert_eq!(rendered[0]["function"]["description"], "echoes");
        assert_eq!(rendered[0]["function"]["parameters"]["type"], "object");
    }
}
