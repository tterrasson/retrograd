//! `[reference]`: the frozen model a KL penalty is taken against.
//!
//! Algorithm-neutral: several objectives declare their KL term under their own
//! section, but they all score against the same anchor file.

use std::path::{Path, PathBuf};

use retrograd_core::{Error, Result};

use crate::common::{require_nonzero, resolve};
use crate::document::ReferenceToml;

/// The frozen anchor of a run, resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceConfig {
    /// The GGUF the anchor is loaded from.
    pub model: PathBuf,
    /// Context width of the anchor. `None` follows `[training].ctx`.
    pub n_ctx: Option<u32>,
}

/// Builds `[reference]`. There is no precision knob: the anchor's dtype is the
/// one in its file, so the scores depend on the model, not on a setting.
pub fn build_reference(
    value: ReferenceToml,
    root: &Path,
    training_ctx: u32,
) -> Result<ReferenceConfig> {
    if let Some(ctx) = value.ctx {
        require_nonzero(ctx, "reference.ctx must be greater than zero")?;
        // The anchor scores the sequences this run produces, so it must hold
        // them.
        if ctx < training_ctx {
            return Err(Error::config(format!(
                "reference.ctx = {ctx} is narrower than training.ctx = {training_ctx}: \
                 the anchor scores the sequences this run produces, so it cannot hold \
                 fewer tokens than they can carry"
            )));
        }
    }
    if value.model.as_os_str().is_empty() {
        return Err(Error::config("reference.model must name a file"));
    }
    let model = resolve(root, value.model);
    Ok(ReferenceConfig {
        model,
        n_ctx: value.ctx,
    })
}
