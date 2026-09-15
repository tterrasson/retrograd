use std::sync::Arc;

use crate::tools::{ToolCallParser, ToolResult, ToolSpec};
use crate::trajectory::{Message, Role};
use crate::{Error, Result};

/// How the model gets told what tools it has.
///
/// A model trained with tools declares them in its own chat template. When it
/// does, handing the catalog to the template renders it in the format the model
/// was pre-trained on; when it does not, the catalog has to be described in the
/// system prompt, and the response parsed back out of free text by convention.
/// The second path is the only one that works for every template, so it stays
/// the fallback rather than being retired.
///
/// The parser travels *with* the rendering rather than beside it, because the
/// two are one decision. Choosing them separately is what let a template render
/// its own tool catalog while a hard-wired `<tool_call>` parser read the answer:
/// the model called its tools in its family's format and every call was read as
/// prose. Held here, the pair cannot come apart.
#[derive(Clone)]
pub(super) enum ToolRendering {
    /// The template renders the catalog itself, frames tool observations in its
    /// own `tool` role, and `parser` reads the call format that same template
    /// teaches the model.
    Native {
        specs: Vec<ToolSpec>,
        parser: Arc<dyn ToolCallParser>,
    },
    /// The template is blind to tools, or yields no parser for its own format:
    /// they are described in the system prompt and read back by convention.
    Prompt { instructions: String, tools: usize },
    /// No tools at all - the policy answers in a single turn.
    None,
}

impl ToolRendering {
    /// How many tools the run declared, whichever way they are rendered. Zero
    /// is what makes "not one tool call was parsed" a normal outcome rather
    /// than the diagnostic in
    /// [`tool_call_parse_warning`](crate::grpo::selection::tool_call_parse_warning).
    pub(super) fn declared_tools(&self) -> usize {
        match self {
            Self::Native { specs, .. } => specs.len(),
            Self::Prompt { tools, .. } => *tools,
            Self::None => 0,
        }
    }
}

impl std::fmt::Debug for ToolRendering {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native { specs, .. } => formatter
                .debug_struct("Native")
                .field("specs", specs)
                .finish_non_exhaustive(),
            Self::Prompt {
                instructions,
                tools,
            } => formatter
                .debug_struct("Prompt")
                .field("instructions", instructions)
                .field("tools", tools)
                .finish(),
            Self::None => formatter.write_str("None"),
        }
    }
}

/// Turns one observation into the message the template will render.
///
/// Under native rendering the template has a `tool` role of its own and the call
/// id travels as `tool_call_id`, so the content is the tool's output and nothing
/// else. Without it, the same information has to survive inside free text, which
/// is what the `id: content` shape is for. The error marker stays in the content
/// either way: no chat template has a concept of a failed tool result, and the
/// policy needs to read the failure to react to it.
pub(super) fn observation_message(observation: &ToolResult, rendering: &ToolRendering) -> Message {
    let error_marker = if observation.is_error { "ERROR: " } else { "" };
    let content = match rendering {
        ToolRendering::Native { .. } => format!("{error_marker}{}", observation.content),
        ToolRendering::Prompt { .. } | ToolRendering::None => format!(
            "{}: {error_marker}{}",
            observation.call_id, observation.content
        ),
    };
    Message {
        role: Role::Tool,
        content,
        tool_calls: Vec::new(),
        tool_call_id: Some(observation.call_id.clone()),
        is_error: observation.is_error,
    }
}

/// The fallback rendering: the catalog written into the system turn, in the
/// `<tool_call>` convention [`HermesToolCallParser`](crate::tools::HermesToolCallParser)
/// reads back. Returns the whole [`ToolRendering`] rather than just the text so
/// that the count the diagnostic needs travels with it.
pub(super) fn prompt_tool_rendering(tools: &[ToolSpec]) -> Result<ToolRendering> {
    let mut canonical = tools.to_vec();
    canonical.sort_by(|a, b| a.name.cmp(&b.name));
    let definitions = serde_json::to_string(&canonical)
        .map_err(|error| Error::invalid(format!("serialize tool definitions: {error}")))?;
    Ok(ToolRendering::Prompt {
        instructions: format!(
            "Available tools (JSON): {definitions}\n\
             Call a tool with <tool_call>{{\"name\":\"tool_name\",\"arguments\":{{...}}}}</tool_call>."
        ),
        tools: tools.len(),
    })
}

pub(super) fn inject_tool_instructions(messages: &mut Vec<Message>, instructions: String) {
    if let Some(system) = messages
        .iter_mut()
        .find(|message| message.role == Role::System)
    {
        system.content.push_str("\n\n");
        system.content.push_str(&instructions);
    } else {
        messages.insert(0, Message::text(Role::System, instructions));
    }
}
