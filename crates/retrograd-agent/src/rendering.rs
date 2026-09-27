//! How a conversation's tools reach the model, and how its answer is read back.
//!
//! Shared by every caller that renders a prompt with tools and parses what the
//! model writes: the agentic rollout, and a server answering chat requests. One
//! decision, here, is what keeps them apart from each other by nothing more than
//! who holds the trainer - what is evaluated is what is trained.

use std::sync::Arc;

use retrograd_dataset::chat_template::{
    self, TemplateTool, ToolRenderingKind, decide_rendering, observation_text,
    prompt_tool_instructions,
};

use crate::Result;
use crate::policy::Policy;
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
pub enum ToolRendering {
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
    /// than a diagnostic.
    pub fn declared_tools(&self) -> usize {
        self.specs().len()
    }

    /// The catalog, in the order the environment listed it.
    pub fn specs(&self) -> Arc<[ToolSpec]> {
        match self {
            Self::Native { specs, .. } | Self::Prompt { specs, .. } => specs.clone(),
            Self::None => Arc::default(),
        }
    }

    /// The decision alone, without the payload each branch carries.
    pub fn kind(&self) -> ToolRenderingKind {
        match self {
            Self::Native { .. } => ToolRenderingKind::Native,
            Self::Prompt { .. } => ToolRenderingKind::Prompt,
            Self::None => ToolRenderingKind::None,
        }
    }

    /// The catalog the chat template itself is handed: the whole of it under
    /// native rendering, nothing otherwise - under the prompt rendering it is in
    /// the system turn already, and handing it over a second time would render
    /// it twice.
    pub fn template_specs(&self) -> &[ToolSpec] {
        match self {
            Self::Native { specs, .. } => specs,
            Self::Prompt { .. } | Self::None => &[],
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

/// The rendering a catalog gets, from what the model's template answered: whether
/// it renders tools at all, and the parser derived from it for this catalog.
///
/// Synchronous, for a caller that holds the trainer itself;
/// [`resolve_tool_rendering`] asks a [`Policy`] the same two questions.
pub fn resolve_tool_rendering_with(
    supports_native: bool,
    parser: Option<Arc<dyn ToolCallParser>>,
    specs: Vec<ToolSpec>,
) -> Result<ToolRendering> {
    if specs.is_empty() {
        return Ok(ToolRendering::None);
    }
    match (
        decide_rendering(specs.len(), supports_native, parser.is_some()),
        parser,
    ) {
        (ToolRenderingKind::Native, Some(parser)) => Ok(ToolRendering::Native {
            specs: specs.into(),
            parser,
        }),
        _ => {
            // The template renders tools but nothing could be derived to read
            // them back. Rendering natively anyway would rebuild the exact
            // asymmetry this is here to prevent, so both halves fall back
            // together.
            if supports_native {
                tracing::warn!(
                    "the model's chat template renders tools but yields no parser for its \
                     own call format; falling back to the prompt-described convention"
                );
            }
            prompt_tool_rendering(specs)
        }
    }
}

/// [`resolve_tool_rendering_with`] over a policy.
pub async fn resolve_tool_rendering(
    policy: &dyn Policy,
    specs: Vec<ToolSpec>,
) -> Result<ToolRendering> {
    if specs.is_empty() {
        return Ok(ToolRendering::None);
    }
    let native = policy.supports_native_tools().await?;
    // Asked here and not once per policy, because the generated grammar may
    // name the functions: the parser is derived from the template *and* the
    // catalog.
    let parser = match native {
        true => policy.tool_call_parser(&specs).await?,
        false => None,
    };
    resolve_tool_rendering_with(native, parser, specs)
}

/// Turns one observation into the message the template will render; the text
/// is [`observation_text`], shared with SFT on tool conversations.
pub fn observation_message(observation: &ToolResult, rendering: &ToolRendering) -> Message {
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
pub fn prompt_tool_rendering(tools: Vec<ToolSpec>) -> Result<ToolRendering> {
    let definitions = tools.iter().map(TemplateTool::from).collect::<Vec<_>>();
    Ok(ToolRendering::Prompt {
        instructions: prompt_tool_instructions(&definitions)?,
        specs: tools.into(),
    })
}

/// Writes the prompt-described catalog into the first system turn, or into a
/// system turn of its own at the head of the conversation.
pub fn inject_tool_instructions(messages: &mut Vec<Message>, instructions: &str) {
    chat_template::inject_tool_instructions(messages, instructions);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::HermesToolCallParser;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn the_decision_falls_back_as_one_when_either_half_is_missing() {
        let parser: Arc<dyn ToolCallParser> = Arc::new(HermesToolCallParser);
        let native =
            resolve_tool_rendering_with(true, Some(parser.clone()), vec![spec("a")]).unwrap();
        assert_eq!(native.kind(), ToolRenderingKind::Native);
        assert_eq!(native.template_specs().len(), 1);

        let no_parser = resolve_tool_rendering_with(true, None, vec![spec("a")]).unwrap();
        assert_eq!(no_parser.kind(), ToolRenderingKind::Prompt);
        assert!(no_parser.template_specs().is_empty());

        let blind = resolve_tool_rendering_with(false, Some(parser), vec![spec("a")]).unwrap();
        assert_eq!(blind.kind(), ToolRenderingKind::Prompt);

        let none = resolve_tool_rendering_with(true, None, Vec::new()).unwrap();
        assert_eq!(none.kind(), ToolRenderingKind::None);
    }
}
