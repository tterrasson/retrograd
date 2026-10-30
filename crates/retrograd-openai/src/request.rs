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
use retrograd_core::SamplingParamsV2;

/// What any generation surface asks the model, independent of which weights
/// answer and of which protocol asked.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    /// In the order the client sent them. A `tool` message carries the tool's
    /// own output; how it is framed depends on the rendering, decided later.
    /// Empty when `raw_prompt` is set.
    pub messages: Vec<Message>,
    /// When set, the prompt is this text tokenized as-is: no chat template, no
    /// tools, no parser (the legacy completions surface answers the raw
    /// prompt, the way the model was pretrained on, not the way a chat is
    /// formatted).
    pub raw_prompt: Option<String>,
    /// Empty for `tool_choice: "none"`: with no constrained decoding, a
    /// catalog in the prompt is an invitation to call, so none is rendered.
    pub tools: Vec<ToolSpec>,
    pub sampling: SamplingParamsV2,
    /// Applied to the text after generation, the first match truncating it.
    pub stop: Vec<String>,
    /// How many completions of the same prompt the client asked for (`n` on
    /// the wire): one prefill, then one copy of the prompt per further choice.
    pub n: u32,
    /// How many top-logprob pairs each emitted token carries, when the client
    /// asked for logprobs. `None` asks for no logprobs at all.
    pub logprobs: Option<u32>,
    /// Constrained tool decoding (`tool_choice: "required"`, a forced
    /// function, `parallel_tool_calls: false`): rendered into a grammar
    /// derived from the model's own template on the thread that owns the
    /// weights, never by deleting calls after the fact.
    pub tool_constraint: Option<ToolConstraint>,
    /// A reasoning budget in tokens (`-1`: the model decides), from a
    /// Responses `reasoning.effort` through the documented local policy.
    /// Applied to the serving engine at admission; the runtime builds each
    /// request's budget machine from the engine's plan. The session drains
    /// requests before changing that plan.
    pub reasoning_budget: Option<i32>,
}

/// A tool-calling constraint the template's own format can express as a
/// grammar. Mutually exclusive with a `response_format` grammar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolConstraint {
    /// `tool_choice: "required"`: at least one call.
    Required,
    /// `tool_choice: {"type":"function","function":{"name":…}}`: exactly this
    /// call.
    Forced(String),
    /// `parallel_tool_calls: false`: at most one call.
    AtMostOne,
}
