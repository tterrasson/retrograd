//! The JSON an HF-style chat template reads.
//!
//! `(role, content)` pairs are all a template needs for plain chat, and all the
//! renderer receives. A model trained with tools declares them in its own
//! template - a tool catalog, `tool_calls` on an assistant turn, a `tool` role
//! for what comes back - and rendering none of that means training the model on
//! a format it never saw during pre-training. This module builds what the
//! template expects instead.

use retrograd_dataset::chat_template::{self, TemplateCall, TemplateMessage, TemplateTool};

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

/// Serializes a conversation as a client sent it: assistant content and
/// `tool_calls` handed to the template, arguments as an object - the shape HF
/// templates read (`| tojson` over a string would encode it twice).
///
/// Not what a rollout renders, and deliberately so. A rollout re-injects its own
/// sampled tokens and lets the template decide only the framing around them;
/// a server has no tokens, only the text the client sends back, so a multi-turn
/// history is re-rendered by the template the way every OpenAI-compatible
/// server does it. The opening turn - system and user, before anything was
/// sampled - is byte for byte what [`template_messages`] writes.
pub fn template_messages_verbatim(messages: &[Message]) -> Result<String> {
    let calls = messages
        .iter()
        .map(|message| {
            message
                .tool_calls
                .iter()
                .map(TemplateCall::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let messages = messages
        .iter()
        .zip(&calls)
        .map(|(message, calls)| TemplateMessage {
            tool_calls: calls,
            ..template_message(message)
        })
        .collect::<Vec<_>>();
    Ok(chat_template::template_messages_verbatim(&messages)?)
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

    fn fixture(name: &str) -> String {
        let path = format!(
            "{}/../retrograd-engine/tests/fixtures/chat-templates/{name}.jinja",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
    }

    fn move_tool() -> Vec<ToolSpec> {
        vec![ToolSpec {
            name: "move".into(),
            description: "move the agent".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"direction": {"type": "string"}},
                "required": ["direction"],
            }),
        }]
    }

    fn move_call() -> Message {
        Message {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_0".into(),
                name: "move".into(),
                arguments: json!({"direction": "left"}),
            }],
            tool_call_id: None,
            is_error: false,
        }
    }

    #[test]
    fn a_verbatim_turn_hands_its_calls_over_with_object_arguments() {
        let messages = vec![Message::text(Role::User, "go"), move_call()];
        let rendered: Value =
            serde_json::from_str(&template_messages_verbatim(&messages).unwrap()).unwrap();
        let call = &rendered[1]["tool_calls"][0];
        assert_eq!(call["id"], "call_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "move");
        assert_eq!(
            call["function"]["arguments"],
            json!({"direction": "left"}),
            "a string here would be encoded twice by the templates that `tojson` it"
        );
        assert_eq!(rendered[1]["content"], "");
    }

    #[test]
    fn an_opening_turn_renders_as_the_rollout_renders_it() {
        let messages = vec![
            Message::text(Role::System, "be brief"),
            Message::text(Role::User, "hello"),
        ];
        let (rollout, sentinels) = template_messages(&messages).unwrap();
        assert!(sentinels.is_empty());
        assert_eq!(template_messages_verbatim(&messages).unwrap(), rollout);
    }

    /// What a template writes for a verbatim assistant turn is what the parser
    /// derived from the same template reads back, for every family that
    /// declares tools. A multi-turn history re-rendered through the template
    /// must be one the model could have sampled.
    #[test]
    fn a_verbatim_call_reads_back_through_the_template_s_own_parser() {
        let tools = template_tools(&move_tool()).unwrap();
        let user = vec![Message::text(Role::User, "go")];
        let with_call = vec![Message::text(Role::User, "go"), move_call()];
        for name in ["lfm2.5-230m", "qwen3-0.6b", "qwen3.5-0.8b"] {
            let template = fixture(name);
            let prompt = retrograd_engine::render_chat_template_source(
                &template,
                &template_messages_verbatim(&user).unwrap(),
                Some(&tools),
                true,
            )
            .unwrap();
            let whole = retrograd_engine::render_chat_template_source(
                &template,
                &template_messages_verbatim(&with_call).unwrap(),
                Some(&tools),
                false,
            )
            .unwrap();
            // The closer the template appends once the turn is over is not
            // sampled - generation stops on it - so it is read off a plain turn
            // and cut, rather than guessed.
            let plain = retrograd_engine::render_chat_template_source(
                &template,
                &template_messages_verbatim(&[
                    Message::text(Role::User, "go"),
                    Message::text(Role::Assistant, "zqxplain"),
                ])
                .unwrap(),
                Some(&tools),
                false,
            )
            .unwrap();
            let closer = &plain[plain.rfind("zqxplain").unwrap() + "zqxplain".len()..];
            let body = whole
                .strip_prefix(prompt.as_str())
                .unwrap_or_else(|| panic!("{name}: the turn does not extend the prompt"));
            let body = body.strip_suffix(closer).unwrap_or(body);
            let parser =
                retrograd_engine::tool_call_parser_from_source(&template, Some(&tools)).unwrap();
            let parsed = crate::TemplateToolCallParser::new(parser);
            let parsed = crate::tools::ToolCallParser::parse(&parsed, body);
            assert_eq!(parsed.tool_calls.len(), 1, "{name}: {body:?} -> {parsed:?}");
            assert_eq!(parsed.tool_calls[0].name, "move", "{name}");
            assert_eq!(
                parsed.tool_calls[0].arguments,
                json!({"direction": "left"}),
                "{name}"
            );
        }
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
