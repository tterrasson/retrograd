//! Tool records: the rules a conversation with calls has to follow.

use std::collections::HashSet;

use crate::{ChatExample, ChatExampleError};

/// The catalog and the calls, checked without a model: what the server can
/// refuse at upload and what an export is checked against before it is
/// written.
///
/// A call is answered by the `tool` messages that directly follow its turn,
/// each at most once, by id or - without one - in call order. Every call is
/// answered before the next message that is not a `tool` one, because that is
/// the only order a rollout produces: the environment answers every call of a
/// turn before the policy speaks again.
pub(crate) fn validate_tools(example: &ChatExample) -> Result<(), ChatExampleError> {
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

#[cfg(test)]
mod tests {
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
}
