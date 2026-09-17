//! The `[ppo]` section: PPO against an external reward command.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, FeatureDtype, Result, RewardMode, RewardProtocol, SamplingParams};

use crate::common::{
    require_non_negative_f32, require_nonzero, require_positive_f32, resolve, reward_protocol,
    sampling, validate_command,
};
use crate::document::SamplingToml;

#[derive(Clone, Debug)]
pub struct PpoConfig {
    pub prompts: PathBuf,
    pub reward_command: Vec<String>,
    /// How that command is spoken to: one worker for the loop, or one process
    /// per batch, and the deadline of a batch either way.
    pub reward_protocol: RewardProtocol,
    pub updates: u32,
    pub rollout_batch_size: usize,
    pub ppo_epochs: u32,
    pub clip_range: f32,
    pub kl_coefficient: f32,
    pub critic: CriticConfig,
    pub sampling: SamplingParams,
}

/// Linear-probe value head over the model's hidden states. When disabled the
/// advantage falls back to the whitened per-sequence reward.
#[derive(Clone, Debug)]
pub struct CriticConfig {
    pub enabled: bool,
    /// Per-token discount for GAE; 1.0 = undiscounted.
    pub gamma: f32,
    /// GAE bias/variance trade-off; 1.0 = Monte-Carlo returns.
    pub gae_lambda: f32,
    /// Adam learning rate of the value head regression.
    pub value_lr: f32,
    /// Full-batch Adam epochs per PPO update.
    pub value_epochs: u32,
    /// Storage precision of the host-side feature matrix
    /// (`total_completion_states * hidden_dim`, the run's largest host
    /// allocation). The fit stays in F32 whatever this says; a 16-bit setting
    /// halves the buffer and rounds the rows it regresses on.
    pub feature_dtype: FeatureDtype,
}

impl Default for CriticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gamma: 1.0,
            gae_lambda: 0.95,
            value_lr: 1.0e-2,
            value_epochs: 8,
            feature_dtype: FeatureDtype::default(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PpoToml {
    pub prompts: PathBuf,
    pub reward_command: Vec<String>,
    /// `"persistent"` (default) or `"oneshot"`. Persistent keeps one worker
    /// alive for the whole loop behind a version handshake; a command that
    /// reads its stdin to the end before answering needs `"oneshot"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_mode: Option<RewardMode>,
    /// Deadline of one reward batch, in seconds. Also covers the first batch's
    /// worker startup in the persistent mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_timeout_seconds: Option<u64>,
    pub updates: u32,
    pub rollout_batch_size: usize,
    pub ppo_epochs: u32,
    pub clip_range: f32,
    pub kl_coefficient: f32,
    #[serde(default, skip_serializing_if = "CriticToml::is_empty")]
    pub critic: CriticToml,
    pub sampling: SamplingToml,
}
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct CriticToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gae_lambda: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_lr: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_epochs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_dtype: Option<FeatureDtype>,
}

impl CriticToml {
    fn is_empty(&self) -> bool {
        self.enabled.is_none()
            && self.gamma.is_none()
            && self.gae_lambda.is_none()
            && self.value_lr.is_none()
            && self.value_epochs.is_none()
            && self.feature_dtype.is_none()
    }
}
pub(crate) fn build_ppo(value: PpoToml, root: &Path) -> Result<PpoConfig> {
    validate_command(&value.reward_command)?;
    let reward_protocol = reward_protocol("ppo", value.reward_mode, value.reward_timeout_seconds)?;
    if value.updates == 0 || value.rollout_batch_size == 0 || value.ppo_epochs == 0 {
        return Err(Error::config(
            "ppo updates, rollout_batch_size, and ppo_epochs must be greater than zero",
        ));
    }
    if !(value.clip_range > 0.0 && value.clip_range < 1.0) {
        return Err(Error::config("ppo.clip_range must be between 0 and 1"));
    }
    require_non_negative_f32(value.kl_coefficient, "ppo.kl_coefficient")?;
    Ok(PpoConfig {
        prompts: resolve(root, value.prompts),
        reward_command: value.reward_command,
        reward_protocol,
        updates: value.updates,
        rollout_batch_size: value.rollout_batch_size,
        ppo_epochs: value.ppo_epochs,
        clip_range: value.clip_range,
        kl_coefficient: value.kl_coefficient,
        critic: critic(value.critic)?,
        sampling: sampling(value.sampling)?,
    })
}
pub(crate) fn critic(value: CriticToml) -> Result<CriticConfig> {
    let defaults = CriticConfig::default();
    let config = CriticConfig {
        enabled: value.enabled.unwrap_or(defaults.enabled),
        gamma: value.gamma.unwrap_or(defaults.gamma),
        gae_lambda: value.gae_lambda.unwrap_or(defaults.gae_lambda),
        value_lr: value.value_lr.unwrap_or(defaults.value_lr),
        value_epochs: value.value_epochs.unwrap_or(defaults.value_epochs),
        feature_dtype: value.feature_dtype.unwrap_or(defaults.feature_dtype),
    };
    if !config.enabled && value.feature_dtype.is_some() {
        return Err(Error::config(
            "ppo.critic.feature_dtype only applies to the critic's feature matrix; \
             enable ppo.critic or drop the field",
        ));
    }
    if !(config.gamma > 0.0 && config.gamma <= 1.0) {
        return Err(Error::config("ppo.critic.gamma must be in (0, 1]"));
    }
    if !(config.gae_lambda >= 0.0 && config.gae_lambda <= 1.0) {
        return Err(Error::config("ppo.critic.gae_lambda must be in [0, 1]"));
    }
    require_positive_f32(config.value_lr, "ppo.critic.value_lr")?;
    require_nonzero(
        config.value_epochs,
        "ppo.critic.value_epochs must be greater than zero",
    )?;
    Ok(config)
}
