use std::sync::Arc;

use retrograd_dataset::chat_template::{
    self, TemplateTool, ToolRenderingKind, observation_text, prompt_tool_instructions,
};

use crate::Result;
use crate::tools::{ToolCallParser, ToolResult, ToolSpec};
use crate::trajectory::{Message, Role};

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
        specs: Arc<[ToolSpec]>,
        parser: Arc<dyn ToolCallParser>,
    },
    /// The template is blind to tools, or yields no parser for its own format:
    /// they are described in the system prompt and read back by convention.
    Prompt {
        instructions: String,
        specs: Arc<[ToolSpec]>,
    },
    /// No tools at all - the policy answers in a single turn.
    None,
}

impl ToolRendering {
    /// How many tools the run declared, whichever way they are rendered. Zero
    /// is what makes "not one tool call was parsed" a normal outcome rather
    /// than the diagnostic in
    /// [`tool_call_parse_warning`](crate::grpo::selection::tool_call_parse_warning).
    pub(super) fn declared_tools(&self) -> usize {
        self.specs().len()
    }

    /// The catalog, in the order the environment listed it.
    pub(super) fn specs(&self) -> Arc<[ToolSpec]> {
        match self {
            Self::Native { specs, .. } | Self::Prompt { specs, .. } => specs.clone(),
            Self::None => Arc::default(),
        }
    }

    /// The decision alone, without the payload each branch carries.
    pub(super) fn kind(&self) -> ToolRenderingKind {
        match self {
            Self::Native { .. } => ToolRenderingKind::Native,
            Self::Prompt { .. } => ToolRenderingKind::Prompt,
            Self::None => ToolRenderingKind::None,
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
                specs,
            } => formatter
                .debug_struct("Prompt")
                .field("instructions", instructions)
                .field("specs", specs)
                .finish(),
            Self::None => formatter.write_str("None"),
        }
    }
}

/// Turns one observation into the message the template will render; the text
/// is [`observation_text`], shared with SFT on tool conversations.
pub(super) fn observation_message(observation: &ToolResult, rendering: &ToolRendering) -> Message {
    Message {
        role: Role::Tool,
        content: observation_text(
            rendering.kind(),
            &observation.call_id,
            &observation.content,
            observation.is_error,
        ),
        tool_calls: Vec::new(),
        tool_call_id: Some(observation.call_id.clone()),
        is_error: observation.is_error,
    }
}

/// The fallback rendering: the catalog written into the system turn, in the
/// `<tool_call>` convention [`HermesToolCallParser`](crate::tools::HermesToolCallParser)
/// reads back. Returns the whole [`ToolRendering`] rather than just the text so
/// that the catalog the diagnostic and the export need travels with it.
pub(super) fn prompt_tool_rendering(tools: Vec<ToolSpec>) -> Result<ToolRendering> {
    let definitions = tools.iter().map(TemplateTool::from).collect::<Vec<_>>();
    Ok(ToolRendering::Prompt {
        instructions: prompt_tool_instructions(&definitions)?,
        specs: tools.into(),
    })
}

pub(super) fn inject_tool_instructions(messages: &mut Vec<Message>, instructions: &str) {
    chat_template::inject_tool_instructions(messages, instructions);
}
