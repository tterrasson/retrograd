//! A collected trajectory, written back as a chat record SFT can train on.
//!
//! The record is the conversation as *data*: the scenario's own system turn
//! rather than the one the prompt-described catalog was pasted into, each
//! observation as the tool returned it rather than as it was framed, and the
//! catalog in a field of its own. Preparing it goes back through the same
//! rendering contract, so what comes out is the stream the rollout trained on
//! - the parity the tests hold it to.

use retrograd_dataset::chat_template::{
    TemplateTool, ToolRenderingKind, inject_tool_instructions, observation_payload,
    prompt_tool_instructions,
};
use retrograd_dataset::{ChatExample, ChatMessage, ChatToolCall};

use crate::trajectory::{Message, Role, Trajectory};
use crate::{Error, Result};
use retrograd_agent_core::scenario::Scenario;

/// How an assistant turn is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AssistantForm {
    /// The prose and the calls apart, for the student's own template to write.
    /// Portable: across model families, and to a record written by hand.
    #[default]
    Structured,
    /// The generated text verbatim, call markup included - token for token what
    /// the policy sampled, but only readable by a model of the same family.
    Raw,
}

impl AssistantForm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Structured => "structured",
            Self::Raw => "raw",
        }
    }
}

/// Writes `trajectory`, collected on `scenario`, as a chat record.
///
/// Refuses a trajectory that does not rebuild cleanly - a system turn that is
/// not the scenario's plus the catalog, an observation that is not one the
/// rollout wrote, or a record that fails its own validation. A trajectory that
/// passed the collection filters never should, so each of these is a defect to
/// report, not data to work around.
pub fn to_chat_example(
    trajectory: &Trajectory,
    scenario: &Scenario,
    form: AssistantForm,
) -> Result<ChatExample> {
    let provenance = trajectory.provenance.as_ref().ok_or_else(|| {
        Error::invalid("only a trajectory this process collected can be exported")
    })?;
    let tools = provenance
        .tools
        .iter()
        .map(TemplateTool::from)
        .collect::<Vec<_>>();

    // The rollout opened on a system turn exactly when the scenario has one or
    // the catalog was written into one.
    let opened_on_system =
        scenario.system.is_some() || provenance.rendering == ToolRenderingKind::Prompt;
    let messages = match (trajectory.messages.split_first(), opened_on_system) {
        (Some((first, rest)), true) if first.role == Role::System => {
            check_system_turn(first, scenario, &tools, provenance.rendering)?;
            rest
        }
        (_, false) => trajectory.messages.as_slice(),
        _ => {
            return Err(Error::invalid(
                "the trajectory does not open on the system turn its scenario and rendering \
                 call for",
            ));
        }
    };
    let mut records = Vec::with_capacity(messages.len() + 1);
    if let Some(system) = &scenario.system {
        records.push(ChatMessage::text("system", system.as_str()));
    }

    let mut assistant = 0;
    for message in messages {
        records.push(match message.role {
            Role::System | Role::User => {
                ChatMessage::text(message.role.as_str(), message.content.as_str())
            }
            Role::Assistant => {
                let prose = provenance.assistant_prose.get(assistant).ok_or_else(|| {
                    Error::invalid(format!(
                        "trajectory has no recorded prose for assistant turn {assistant}"
                    ))
                })?;
                assistant += 1;
                assistant_message(message, prose, form)
            }
            Role::Tool => tool_message(message, provenance.rendering)?,
        });
    }

    let example = ChatExample {
        tools,
        messages: records,
        ..ChatExample::default()
    };
    example.validate().map_err(|error| {
        Error::invalid(format!(
            "exported trajectory of '{}' is not a valid record: {error}",
            trajectory.scenario_id
        ))
    })?;
    Ok(example)
}

/// The rollout's first system turn must be the scenario's, with the catalog
/// written in only when the prompt path wrote it: dropping it for
/// `scenario.system` is then a rewrite, not a guess.
fn check_system_turn(
    turn: &Message,
    scenario: &Scenario,
    tools: &[TemplateTool],
    rendering: ToolRenderingKind,
) -> Result<()> {
    let mut expected = scenario
        .system
        .iter()
        .map(|system| Message::text(Role::System, system))
        .collect::<Vec<_>>();
    if rendering == ToolRenderingKind::Prompt {
        let instructions = prompt_tool_instructions(tools)?;
        inject_tool_instructions(&mut expected, &instructions);
    }
    match expected.first() {
        Some(expected) if expected.content == turn.content => Ok(()),
        _ => Err(Error::invalid(
            "the trajectory's system turn is not the scenario's, with the catalog as the \
             rollout writes it",
        )),
    }
}

fn assistant_message(message: &Message, prose: &str, form: AssistantForm) -> ChatMessage {
    let (content, raw) = match form {
        AssistantForm::Structured => (prose, false),
        AssistantForm::Raw => (message.content.as_str(), true),
    };
    ChatMessage {
        // Kept in the raw form too, so the record can be checked without a
        // model: the calls are what the observations that follow answer.
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| ChatToolCall {
                id: Some(call.id.clone()),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect(),
        raw,
        ..ChatMessage::text("assistant", content)
    }
}

fn tool_message(message: &Message, rendering: ToolRenderingKind) -> Result<ChatMessage> {
    let id = message
        .tool_call_id
        .as_deref()
        .ok_or_else(|| Error::invalid("a trajectory observation has no call id"))?;
    let content = observation_payload(rendering, id, &message.content, message.is_error)
        .ok_or_else(|| {
            Error::invalid(format!(
                "observation of call '{id}' is not one the rollout wrote for it"
            ))
        })?;
    Ok(ChatMessage {
        tool_call_id: Some(id.to_owned()),
        is_error: message.is_error,
        ..ChatMessage::text("tool", content)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::tools::{ToolCall, ToolSpec};
    use crate::trajectory::Provenance;

    fn spec() -> ToolSpec {
        ToolSpec {
            name: "run".into(),
            description: "run a command".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn scenario(system: Option<&str>) -> Scenario {
        Scenario {
            id: "fix".into(),
            system: system.map(str::to_owned),
            user: "fix it".into(),
            metadata: Default::default(),
        }
    }

    /// A trajectory the way the rollout writes one: the catalog in the system
    /// turn under the prompt path, `id: ` framing on the observations.
    fn trajectory(rendering: ToolRenderingKind, system: Option<&str>) -> Trajectory {
        let tools = [TemplateTool::from(&spec())];
        let mut messages = system
            .iter()
            .map(|system| Message::text(Role::System, *system))
            .collect::<Vec<_>>();
        if rendering == ToolRenderingKind::Prompt {
            inject_tool_instructions(&mut messages, &prompt_tool_instructions(&tools).unwrap());
        }
        let framed = |content: &str, is_error| {
            retrograd_dataset::chat_template::observation_text(
                rendering, "call_0", content, is_error,
            )
        };
        messages.extend([
            Message::text(Role::User, "fix it"),
            Message::text(Role::User, "The workspace is /work."),
            Message {
                tool_calls: vec![ToolCall {
                    id: "call_0".into(),
                    name: "run".into(),
                    arguments: json!({"cmd": "pytest"}),
                }],
                ..Message::text(
                    Role::Assistant,
                    "Looking.\n<tool_call>{\"name\":\"run\",\"arguments\":{\"cmd\":\"pytest\"}}</tool_call>",
                )
            },
            Message {
                tool_call_id: Some("call_0".into()),
                is_error: true,
                ..Message::text(Role::Tool, framed("1 failed", true))
            },
            Message::text(Role::Assistant, "Fixed."),
        ]);
        Trajectory {
            scenario_id: "fix".into(),
            messages,
            tokens: vec![1, 2],
            old_logprobs: vec![-0.1],
            train_mask: vec![false, true],
            steps: Vec::new(),
            reward: Some(1.0),
            truncated: false,
            metadata: Default::default(),
            provenance: Some(Provenance {
                tools: Arc::from([spec()]),
                rendering,
                assistant_prose: vec!["Looking.".into(), "Fixed.".into()],
                ..Default::default()
            }),
        }
    }

    #[test]
    fn a_structured_record_carries_the_prose_the_calls_and_the_raw_observation() {
        let example = to_chat_example(
            &trajectory(ToolRenderingKind::Prompt, Some("You fix bugs.")),
            &scenario(Some("You fix bugs.")),
            AssistantForm::Structured,
        )
        .unwrap();
        assert_eq!(example.tools, [TemplateTool::from(&spec())]);
        let roles = example
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            ["system", "user", "user", "assistant", "tool", "assistant"]
        );
        assert_eq!(
            example.messages[0].content, "You fix bugs.",
            "the catalog the prompt path pasted in is the tools field now"
        );
        assert_eq!(
            example.messages[2].content, "The workspace is /work.",
            "the environment's opening stays a user turn of its own"
        );
        let calling = &example.messages[3];
        assert_eq!(calling.content, "Looking.");
        assert!(!calling.raw);
        assert_eq!(calling.tool_calls[0].id.as_deref(), Some("call_0"));
        assert_eq!(calling.tool_calls[0].arguments, json!({"cmd": "pytest"}));
        let observation = &example.messages[4];
        assert_eq!(observation.content, "1 failed");
        assert!(observation.is_error);
        assert_eq!(observation.tool_call_id.as_deref(), Some("call_0"));
    }

    #[test]
    fn a_raw_record_keeps_the_generated_text_and_its_calls() {
        let example = to_chat_example(
            &trajectory(ToolRenderingKind::Prompt, None),
            &scenario(None),
            AssistantForm::Raw,
        )
        .unwrap();
        assert_eq!(
            example.messages[0].role, "user",
            "a system turn the catalog alone created goes with it"
        );
        let calling = &example.messages[2];
        assert!(calling.raw);
        assert!(calling.content.contains("<tool_call>"));
        assert_eq!(
            calling.tool_calls.len(),
            1,
            "kept for model-free validation"
        );
    }

    #[test]
    fn a_native_observation_is_its_output_as_is() {
        let example = to_chat_example(
            &trajectory(ToolRenderingKind::Native, Some("You fix bugs.")),
            &scenario(Some("You fix bugs.")),
            AssistantForm::Structured,
        )
        .unwrap();
        assert_eq!(example.messages[4].content, "1 failed");
    }

    #[test]
    fn a_system_turn_that_is_not_the_scenarios_is_a_defect() {
        let error = to_chat_example(
            &trajectory(ToolRenderingKind::Prompt, Some("You fix bugs.")),
            &scenario(Some("Something else.")),
            AssistantForm::Structured,
        )
        .unwrap_err();
        assert!(error.to_string().contains("system turn"), "{error}");
    }

    #[test]
    fn an_observation_the_rollout_did_not_write_is_a_defect() {
        let mut trajectory = trajectory(ToolRenderingKind::Prompt, None);
        // After the system turn the catalog opened, the two user turns and the
        // calling one.
        trajectory.messages[4].content = "call_9: 1 failed".into();
        let error =
            to_chat_example(&trajectory, &scenario(None), AssistantForm::Structured).unwrap_err();
        assert!(
            error.to_string().contains("not one the rollout wrote"),
            "{error}"
        );
    }

    #[test]
    fn a_record_that_fails_its_own_validation_is_not_exported() {
        let mut trajectory = trajectory(ToolRenderingKind::Prompt, None);
        // The observation answers a call nobody made.
        trajectory.messages[3].tool_calls.clear();
        let error =
            to_chat_example(&trajectory, &scenario(None), AssistantForm::Structured).unwrap_err();
        assert!(error.to_string().contains("not a valid record"), "{error}");
    }
}
