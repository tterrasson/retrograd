//! `[optimizer.<name>]`: the coefficients and structural values of the
//! optimizer `[training].optimizer` named. Each key lands on a row the
//! optimizer's own layout declares; a section for an optimizer the run did not
//! choose is refused.

use retrograd_core::{
    Error, GEFEN_CODEBOOK_LEVELS, GefenLayout, GefenVariant, HyperparameterValue,
    HyperparameterVector, OptimizerKind, Result,
};

use crate::document::{GefenToml, MuonToml, OptimizerToml};

/// The chosen optimizer with its layout resolved, and the vector its update
/// reads. The vector is built from the layout, so a key one variant does not
/// declare is refused by the vector itself.
pub struct ResolvedOptimizer {
    pub kind: OptimizerKind,
    pub hyperparameters: HyperparameterVector,
}

/// Resolves `[optimizer.*]` against the optimizer the document named. The
/// run's `learning_rate`, `weight_decay` and `max_grad_norm` are mirrored in,
/// so the vector is what the update actually reads.
pub fn build_optimizer(
    section: Option<&OptimizerToml>,
    chosen: OptimizerKind,
    learning_rate: f32,
    weight_decay: f32,
    max_grad_norm: f32,
) -> Result<ResolvedOptimizer> {
    let muon = section.and_then(|value| value.muon.as_ref());
    let gefen = section.and_then(|value| value.gefen.as_ref());

    if muon.is_some() && chosen != OptimizerKind::Muon {
        return Err(Error::config(format!(
            "[optimizer.muon] configures an optimizer this run does not use: \
             training.optimizer = '{chosen}'"
        )));
    }
    if gefen.is_some() && chosen.gefen_layout().is_none() {
        return Err(Error::config(format!(
            "[optimizer.gefen] configures an optimizer this run does not use: \
             training.optimizer = '{chosen}'"
        )));
    }

    // The variant selects the slot table, so it is resolved before the vector
    // exists: the declared rows depend on it.
    let kind = match (chosen.gefen_layout(), gefen) {
        (Some(declared), Some(value)) => OptimizerKind::Gefen(gefen_layout(declared, value)?),
        _ => chosen,
    };

    let mut hyperparameters = kind.declared_hyperparameters();
    hyperparameters.set_scalar("learning_rate", learning_rate)?;
    hyperparameters.set_scalar("weight_decay", weight_decay)?;
    hyperparameters.set_scalar("max_grad_norm", max_grad_norm)?;
    if let Some(value) = muon {
        apply_muon(&mut hyperparameters, value)?;
    }
    if let Some(value) = gefen {
        apply_gefen(&mut hyperparameters, value)?;
    }
    Ok(ResolvedOptimizer {
        kind,
        hyperparameters,
    })
}

/// Gefen's structural values: the ones that decide a slot's shape and which
/// parameters the optimizer owns.
fn gefen_layout(declared: GefenLayout, value: &GefenToml) -> Result<GefenLayout> {
    let mut layout = declared;
    if let Some(variant) = &value.variant {
        layout.variant = GefenVariant::parse(variant)?;
    }
    if let Some(block_size) = value.block_size {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(Error::config(format!(
                "optimizer.gefen.block_size must be a positive power of two; got {block_size}"
            )));
        }
        layout.block_size = block_size;
    }
    if let Some(min_numel) = value.min_numel {
        if min_numel == 0 {
            return Err(Error::config(
                "optimizer.gefen.min_numel must be greater than zero",
            ));
        }
        layout.min_numel = min_numel;
    }
    // Only a uniform codebook and fixed partition are implemented; anything
    // else is rejected rather than accepted and ignored.
    if let Some(codebook) = &value.codebook
        && !codebook.trim().eq_ignore_ascii_case("uniform")
    {
        return Err(Error::config(format!(
            "optimizer.gefen.codebook must be 'uniform'; got '{codebook}'"
        )));
    }
    if let Some(partition) = &value.partition
        && !partition.trim().eq_ignore_ascii_case("fixed")
    {
        return Err(Error::config(format!(
            "optimizer.gefen.partition must be 'fixed'; got '{partition}'"
        )));
    }
    Ok(layout)
}

fn apply_muon(vector: &mut HyperparameterVector, value: &MuonToml) -> Result<()> {
    if let Some(momentum) = value.momentum {
        vector.set_scalar("momentum", momentum)?;
    }
    if let Some(nesterov) = value.nesterov {
        vector.set("nesterov", HyperparameterValue::Toggle(nesterov))?;
    }
    if let Some(ns_steps) = value.ns_steps {
        vector.set("ns_steps", HyperparameterValue::Structural(ns_steps.into()))?;
    }
    if let Some(ns_epsilon) = value.ns_epsilon {
        vector.set_scalar("ns_epsilon", ns_epsilon)?;
    }
    if let Some(rate) = value.fallback_learning_rate {
        vector.set_scalar("fallback_learning_rate", rate)?;
    }
    Ok(())
}

fn apply_gefen(vector: &mut HyperparameterVector, value: &GefenToml) -> Result<()> {
    if let Some(beta1) = value.beta1 {
        vector.set_scalar("beta1", beta1)?;
    }
    if let Some(beta2) = value.beta2 {
        vector.set_scalar("beta2", beta2)?;
    }
    if let Some(eps) = value.eps {
        vector.set_scalar("eps", eps)?;
    }
    if let Some(block_size) = value.block_size {
        vector.set(
            "block_size",
            HyperparameterValue::Structural(i64::try_from(block_size).unwrap_or(i64::MAX)),
        )?;
    }
    if let Some(min_numel) = value.min_numel {
        vector.set(
            "min_numel",
            HyperparameterValue::Structural(i64::try_from(min_numel).unwrap_or(i64::MAX)),
        )?;
    }
    if let Some(levels) = value.codebook_levels {
        // The indices are unsigned bytes, so the codebook is exactly as wide as
        // a byte; a different width is a different encoding, not a setting.
        if levels != GEFEN_CODEBOOK_LEVELS {
            return Err(Error::config(format!(
                "optimizer.gefen.codebook_levels must be {GEFEN_CODEBOOK_LEVELS}: the first \
                 moment is stored as one unsigned byte per element, so the codebook is exactly \
                 as wide as a byte; got {levels}"
            )));
        }
        vector.set(
            "codebook_levels",
            HyperparameterValue::Structural(i64::try_from(levels).unwrap_or(i64::MAX)),
        )?;
    }
    Ok(())
}
