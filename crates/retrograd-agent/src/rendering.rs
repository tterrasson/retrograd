//! How a conversation's tools reach the model, and how its answer is read back.
//!
//! The decision itself lives in [`retrograd_agent_core::rendering`] and is
//! shared by every caller that renders a prompt with tools and parses what the
//! model writes - the agentic rollout, and a server answering chat requests.
//! What stays here is the one entry point that asks a [`Policy`] the two
//! questions the decision needs, which only this crate can do: the policy
//! trait is defined below.

pub use retrograd_agent_core::rendering::*;

use crate::Result;
use crate::policy::Policy;
use crate::tools::ToolSpec;

/// [`resolve_tool_rendering_with`] over a policy: whether the model's template
/// renders the catalog natively, and the parser it yields for this catalog.
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
