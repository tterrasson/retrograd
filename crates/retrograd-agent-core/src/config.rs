//! What an agentic run is configured with, above the rollout vocabulary.
//!
//! These types live here rather than next to the optimizer because the TOML
//! loader (`retrograd-config`) has to build them, and that crate deliberately
//! does not link the training engine. `retrograd-agent` re-exports every name
//! below, so `retrograd_agent::AgentGrpoConfig` keeps resolving.

use serde::{Deserialize, Serialize};

use crate::scenario::{RolloutLimits, TruncationPolicy};
use crate::{Error, Result};

/// What a judge failure does to the update it happened in.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeFailurePolicy {
    #[default]
    DropGroup,
    Fail,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    pub group_size: usize,
    pub seed: u64,
    pub limits: RolloutLimits,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            seed: 42,
            limits: RolloutLimits::default(),
        }
    }
}

impl AgentConfig {
    pub fn validate(&self) -> Result<()> {
        if self.group_size < 2 {
            return Err(Error::invalid("agent group_size must be at least 2"));
        }
        self.limits.validate()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentGrpoConfig {
    pub updates: u32,
    pub scenarios_per_update: usize,
    pub group_size: usize,
    pub epochs: u32,
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    pub kl_coefficient: f32,
    pub seed: u64,
    pub limits: RolloutLimits,
    pub judge_failure: JudgeFailurePolicy,
    pub max_dropped_fraction: f32,
    /// Discard groups whose members all received the same score. Off by default:
    /// such a group contributes no gradient, but dropping it changes the number
    /// of groups an update trains on, so it is a decision to state rather than
    /// one to inherit. Watch `judge/degenerate_group_fraction` before turning it
    /// on.
    pub drop_degenerate_groups: bool,
    /// Let an update that came back with fewer than two trainable trajectories
    /// pass without an optimizer step, instead of ending the run. Off by
    /// default: an update with nothing to train on normally means the
    /// environment or the judge stopped working, and silently continuing would
    /// turn that into a run that burns its budget learning nothing. Turn it on
    /// for tasks where an occasional whole-group failure is expected - every
    /// skip is logged with the same accounting the failure would have reported.
    pub skip_empty_updates: bool,
    pub truncation: TruncationPolicy,
}

impl Default for AgentGrpoConfig {
    fn default() -> Self {
        Self {
            updates: 1,
            scenarios_per_update: 1,
            group_size: 8,
            epochs: 4,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            seed: 42,
            limits: RolloutLimits::default(),
            judge_failure: JudgeFailurePolicy::DropGroup,
            max_dropped_fraction: 0.5,
            drop_degenerate_groups: false,
            skip_empty_updates: false,
            truncation: TruncationPolicy::Drop,
        }
    }
}

impl AgentGrpoConfig {
    pub fn validate(&self) -> Result<()> {
        if self.updates == 0 || self.scenarios_per_update == 0 || self.epochs == 0 {
            return Err(Error::invalid(
                "agent updates, scenarios_per_update, and epochs must be greater than zero",
            ));
        }
        if self.group_size < 2 {
            return Err(Error::invalid("agent group_size must be at least 2"));
        }
        if !(self.clip_range_low > 0.0
            && self.clip_range_low < 1.0
            && self.clip_range_high > 0.0
            && self.clip_range_high < 1.0)
        {
            return Err(Error::invalid("agent clip ranges must be in (0, 1)"));
        }
        // Same rule as `[grpo]` (`retrograd-config`): Clip-Higher widens the
        // band *upwards*, so a high bound below the low one is a config someone
        // mistyped - and the two loops must not disagree on what they accept.
        if self.clip_range_high < self.clip_range_low {
            return Err(Error::invalid(
                "agent clip_range_high must not be below clip_range_low (Clip-Higher)",
            ));
        }
        if !self.kl_coefficient.is_finite() || self.kl_coefficient < 0.0 {
            return Err(Error::invalid(
                "agent kl_coefficient must be finite and non-negative",
            ));
        }
        if !self.max_dropped_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_dropped_fraction)
        {
            return Err(Error::invalid(
                "agent max_dropped_fraction must be in [0, 1]",
            ));
        }
        self.limits.validate()
    }

    /// The constant the GRPO loss is divided by: **the trajectory's token
    /// budget, not the batch's own token count**.
    ///
    /// This is Dr-GRPO's normalization, and it is deliberate. Dividing by the
    /// realized length would make a token's weight depend on how long its own
    /// trajectory happened to be, which is a length bias in the gradient - long
    /// trajectories would be systematically demoted, short ones promoted, for
    /// reasons that have nothing to do with reward. The single-turn sampler
    /// normalizes by `sampling.max_new_tokens` for exactly the same reason
    /// (`retrograd-training/src/grpo.rs`); the multi-turn budget is the same
    /// quantity, over a whole trajectory instead of one completion.
    ///
    /// The consequence to hold on to: **the rollout limits set the scale of the
    /// gradient, so changing them rescales the effective learning rate.** A
    /// trajectory of a few hundred trained tokens under a 3072-token budget is
    /// divided by roughly ten; halving `max_turns` doubles the gradient without
    /// touching `lr`. Re-tune `lr` when the limits move - that coupling is the
    /// price of a length-unbiased normalizer, not an oversight.
    ///
    /// The budget is what the limits *actually* allow: `max_trajectory_tokens`
    /// bounds the whole sequence, so it also bounds the generated part, and a
    /// denominator above it would be a constant nobody can reach.
    pub fn loss_denominator(&self) -> Result<usize> {
        let per_turn_budget = self
            .limits
            .max_turns
            .checked_mul(self.limits.max_new_tokens_per_turn as usize)
            .ok_or_else(|| Error::invalid("agent policy token budget overflows usize"))?;
        Ok(per_turn_budget.min(self.limits.max_trajectory_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpo_config_covers_boundaries_and_overflow() {
        let mut config = AgentGrpoConfig::default();
        assert!(config.validate().is_ok());
        config.group_size = 1;
        assert!(config.validate().is_err());
        config.group_size = 2;
        config.max_dropped_fraction = f32::NAN;
        assert!(config.validate().is_err());
        config.max_dropped_fraction = 0.5;
        // Clip-Higher, and the same refusal `[grpo]` makes: the band widens
        // upwards or the configuration is a typo.
        config.clip_range_high = 0.1;
        assert!(config.validate().is_err());
        config.clip_range_high = config.clip_range_low;
        assert!(config.validate().is_ok());
        config.clip_range_high = 0.28;
        config.limits.max_turns = usize::MAX;
        config.limits.max_new_tokens_per_turn = u32::MAX;
        assert!(config.loss_denominator().is_err());
    }

    /// The loss denominator is a learning-rate scale in disguise (see its
    /// doc-comment), so the value a given config produces is pinned here: it
    /// must not move without someone deciding that it should.
    #[test]
    fn the_loss_denominator_is_the_reachable_token_budget() {
        let config = AgentGrpoConfig::default();
        assert_eq!(config.loss_denominator().unwrap(), 6 * 512);

        // A trajectory budget below the per-turn budget is the real bound: the
        // generated tokens cannot exceed the sequence they live in.
        let clamped = AgentGrpoConfig {
            limits: RolloutLimits {
                max_turns: 6,
                max_new_tokens_per_turn: 512,
                max_trajectory_tokens: 1024,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(clamped.loss_denominator().unwrap(), 1024);
    }
}
