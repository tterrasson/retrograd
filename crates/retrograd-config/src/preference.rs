//! The `[preference]` section: offline preference optimization over pairs of
//! responses - DPO, IPO, SimPO and ORPO.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result};

use crate::common::{parse_string_enum, require_non_negative_f32, require_positive_f32, resolve};

/// Default `preference.logps_drop_warn`, in nats.
pub const DEFAULT_LOGPS_DROP_WARN: f32 = 2.0;

/// The objective a preference run minimizes, with the constants it reads.
///
/// One variant per loss rather than a flat set of optional knobs: a constant
/// only one loss reads is only representable beside that loss, so a document
/// that sets it for another is refused rather than silently ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreferenceLoss {
    /// `-log sigmoid(beta * margin)` over the reference-relative log-ratios,
    /// with conservative label smoothing when `label_smoothing > 0`.
    Dpo { beta: f32, label_smoothing: f32 },
    /// The squared distance of the per-token margin to `1 / (2 beta)`.
    Ipo { beta: f32 },
    /// Reference-free: the per-token mean log-probabilities with a target
    /// margin `gamma_beta_ratio`, in TRL's unit.
    Simpo { beta: f32, gamma_beta_ratio: f32 },
    /// Reference-free: the chosen response's NLL plus `beta` times the log-odds
    /// ratio term. `beta` is TRL's name for the weight usually written lambda.
    Orpo { beta: f32 },
}

impl PreferenceLoss {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dpo { .. } => "dpo",
            Self::Ipo { .. } => "ipo",
            Self::Simpo { .. } => "simpo",
            Self::Orpo { .. } => "orpo",
        }
    }

    pub fn beta(&self) -> f32 {
        match *self {
            Self::Dpo { beta, .. }
            | Self::Ipo { beta }
            | Self::Simpo { beta, .. }
            | Self::Orpo { beta } => beta,
        }
    }

    /// Whether the loss compares the policy with a reference.
    pub fn uses_reference(&self) -> bool {
        matches!(self, Self::Dpo { .. } | Self::Ipo { .. })
    }

    pub fn validate(&self) -> Result<()> {
        require_positive_f32(self.beta(), "preference.beta")?;
        match *self {
            Self::Dpo {
                label_smoothing, ..
            } => {
                if !(0.0..0.5).contains(&label_smoothing) {
                    return Err(Error::config(
                        "preference.label_smoothing must be in [0, 0.5): at 0.5 the pair \
                         prefers nothing",
                    ));
                }
            }
            Self::Simpo {
                gamma_beta_ratio, ..
            } => require_non_negative_f32(gamma_beta_ratio, "preference.gamma_beta_ratio")?,
            Self::Ipo { .. } | Self::Orpo { .. } => {}
        }
        Ok(())
    }
}

/// What a reference-relative loss compares the policy with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceSource {
    /// The policy as it is before the first step: the base plus the initial
    /// adapter, or the untouched base under a base-weight policy. Scored once,
    /// before anything trains, so it holds whatever the run trains.
    Initial,
    /// The model with its adapter disabled. Adapter runs only.
    Base,
    /// The frozen anchor `[reference]` declares.
    Model,
}

impl ReferenceSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Base => "base",
            Self::Model => "model",
        }
    }
}

/// `[preference]`, resolved.
#[derive(Clone, Debug)]
pub struct PreferenceConfig {
    /// A preference JSONL: one prompt and two responses per line.
    pub data: PathBuf,
    pub loss: PreferenceLoss,
    /// `None` for a loss with no reference.
    pub reference: Option<ReferenceSource>,
    /// Permute the pairs at the start of every epoch, seeded from `seed`.
    pub shuffle: bool,
    /// The permutation seed: `lora.seed`, the one seed that describes the run.
    pub seed: u32,
    /// Most pairs one optimizer step may hold. `None` fills the window.
    pub pairs_per_step: Option<u32>,
    /// Drop of the chosen responses' mean log-probability, in nats, at which
    /// the run warns that both sides are falling together.
    pub logps_drop_warn: f32,
}

impl PreferenceConfig {
    pub fn validate(&self) -> Result<()> {
        self.loss.validate()?;
        if self.pairs_per_step == Some(0) {
            return Err(Error::config(
                "preference.pairs_per_step must be greater than zero when set",
            ));
        }
        require_positive_f32(self.logps_drop_warn, "preference.logps_drop_warn")?;
        if self.loss.uses_reference() != self.reference.is_some() {
            return Err(Error::invalid(
                "a preference loss has a reference exactly when it compares against one",
            ));
        }
        Ok(())
    }
}

/// `[preference]`. Everything but `data` has a default; the constant a loss
/// does not read is refused beside it.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PreferenceToml {
    #[cfg_attr(feature = "openapi", schema(value_type = String, format = "path"))]
    pub data: PathBuf,
    /// `dpo` (the default), `ipo`, `simpo` or `orpo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<String>,
    /// Defaults per loss: 0.1 for dpo, ipo and orpo, 2.0 for simpo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f32>,
    /// dpo and ipo: `initial` (the default) or `base`. Absent when
    /// `[reference]` declares the anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// dpo only, in [0, 0.5). Defaults to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_smoothing: Option<f32>,
    /// simpo only, non-negative. Defaults to 0.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma_beta_ratio: Option<f32>,
    /// Defaults to `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairs_per_step: Option<u32>,
    /// In nats, greater than zero. Defaults to 2.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logps_drop_warn: Option<f32>,
}

/// A key a loss does not read, named with the loss that does.
fn only_read_by(key: &str, value_set: bool, reader: &str, loss: &str) -> Result<()> {
    if value_set && reader != loss {
        return Err(Error::config(format!(
            "preference.{key} is only read by loss = \"{reader}\", and this run's loss is \
             \"{loss}\""
        )));
    }
    Ok(())
}

/// Builds `[preference]`. `anchor_declared` says whether the document has a
/// `[reference]` section, which is one of the three sources a reference-relative
/// loss can read.
pub(crate) fn build_preference(
    value: PreferenceToml,
    root: &Path,
    anchor_declared: bool,
    seed: u32,
) -> Result<PreferenceConfig> {
    let loss_name = value
        .loss
        .as_deref()
        .map(str::trim)
        .unwrap_or("dpo")
        .to_ascii_lowercase();
    only_read_by(
        "label_smoothing",
        value.label_smoothing.is_some(),
        "dpo",
        &loss_name,
    )?;
    only_read_by(
        "gamma_beta_ratio",
        value.gamma_beta_ratio.is_some(),
        "simpo",
        &loss_name,
    )?;
    let loss = match loss_name.as_str() {
        "dpo" => PreferenceLoss::Dpo {
            beta: value.beta.unwrap_or(0.1),
            label_smoothing: value.label_smoothing.unwrap_or(0.0),
        },
        "ipo" => PreferenceLoss::Ipo {
            beta: value.beta.unwrap_or(0.1),
        },
        "simpo" => PreferenceLoss::Simpo {
            beta: value.beta.unwrap_or(2.0),
            gamma_beta_ratio: value.gamma_beta_ratio.unwrap_or(0.5),
        },
        "orpo" => PreferenceLoss::Orpo {
            beta: value.beta.unwrap_or(0.1),
        },
        other => {
            return Err(Error::config(format!(
                "preference.loss must be dpo, ipo, simpo or orpo; got '{other}'"
            )));
        }
    };
    let reference = match (
        loss.uses_reference(),
        value.reference.as_deref(),
        anchor_declared,
    ) {
        (false, Some(_), _) => {
            return Err(Error::config(format!(
                "preference.reference is only read by a loss that compares against a \
                 reference (dpo, ipo), and loss = \"{}\" is reference-free",
                loss.name()
            )));
        }
        // A reference-free loss beside `[reference]` is refused with the other
        // anchors nothing reads, in the cross-section rules.
        (false, None, _) => None,
        (true, Some(_), true) => {
            return Err(Error::config(
                "two references declared: [reference] names an anchor and \
                 preference.reference names another - remove one",
            ));
        }
        (true, None, true) => Some(ReferenceSource::Model),
        (true, None, false) => Some(ReferenceSource::Initial),
        (true, Some(name), false) => Some(parse_string_enum!(
            name,
            "preference.reference must be initial or base",
            "initial" => ReferenceSource::Initial,
            "base" => ReferenceSource::Base,
        )?),
    };
    let config = PreferenceConfig {
        data: resolve(root, value.data),
        loss,
        reference,
        shuffle: value.shuffle.unwrap_or(true),
        seed,
        pairs_per_step: value.pairs_per_step,
        logps_drop_warn: value.logps_drop_warn.unwrap_or(DEFAULT_LOGPS_DROP_WARN),
    };
    config.validate()?;
    Ok(config)
}
