//! The JSON an HF-style chat template reads.
//!
//! `(role, content)` pairs are all a template needs for plain chat, and all the
//! renderer receives. A model trained with tools declares them in its own
//! template - a tool catalog, `tool_calls` on an assistant turn, a `tool` role
//! for what comes back - and rendering none of that means training the model on
//! a format it never saw during pre-training. This module builds what the
//! template expects instead.

use retrograd_dataset::chat_template::{self, TemplateMessage, TemplateTool};

pub use retrograd_dataset::chat_template::split_assistant_spans;

use crate::Result;
use crate::tools::ToolSpec;
use crate::trajectory::Message;

/// The template's view of a rollout message. Its calls are never handed over:
/// an assistant turn is always a sentinel, and the calls are already inside the
/// sampled tokens.
fn template_message(message: &Message) -> TemplateMessage<'_> {
    TemplateMessage {
        role: message.role.as_str(),
        content: message.content.as_str().into(),
        tool_call_id: message.tool_call_id.as_deref(),
        tool_calls: &[],
    }
}

/// Serializes a conversation for the model's chat template, every assistant
/// turn replaced by a sentinel; see [`chat_template::template_messages`].
pub fn template_messages(messages: &[Message]) -> Result<(String, Vec<String>)> {
    let messages = messages.iter().map(template_message).collect::<Vec<_>>();
    Ok(chat_template::template_messages(&messages)?)
}

/// Serializes the tool catalog in the OpenAI function shape, the one HF chat
/// templates iterate over.
pub fn template_tools(tools: &[ToolSpec]) -> Result<String> {
    let tools = tools.iter().map(TemplateTool::from).collect::<Vec<_>>();
    Ok(chat_template::template_tools(&tools)?)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

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
