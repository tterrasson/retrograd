//! The one request every generation surface translates into.
//!
//! Chat Completions and the Responses API differ on the wire, not in what they
//! ask the model: [`GenerationRequest`] is that common middle - the
//! conversation as the surface converted it, the tool catalog, the sampling,
//! the constraints, and the output options. Rendering it through the model's
//! template happens later, on the thread that owns the weights
//! ([`crate::render::render_prompt`]), because only that thread may ask the
//! model anything.

use retrograd_agent_core::tools::ToolSpec;
use retrograd_agent_core::trajectory::Message;
use retrograd_core::SamplingParams;

/// What any generation surface asks the model, independent of which weights
/// answer and of which protocol asked.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    /// In the order the client sent them. A `tool` message carries the tool's
    /// own output; how it is framed depends on the rendering, decided later.
    pub messages: Vec<Message>,
    /// Empty for `tool_choice: "none"`: with no constrained decoding, a
    /// catalog in the prompt is an invitation to call, so none is rendered.
    pub tools: Vec<ToolSpec>,
    pub sampling: SamplingParams,
    /// Applied to the text after generation, the first match truncating it.
    pub stop: Vec<String>,
    /// How many completions of the same prompt the client asked for (`n` on
    /// the wire). One today: the surfaces refuse `n > 1` at conversion, and
    /// the field carries it so a later surface can widen it in one place.
    pub n: u32,
    /// How many top-logprob pairs each emitted token carries, when the client
    /// asked for logprobs. `None` today: refused at conversion, like `n > 1`.
    pub logprobs: Option<u32>,
}
