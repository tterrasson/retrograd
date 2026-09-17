//! The `[grpo]` section: Dr. GRPO over a prompt set, with its DAPO extensions.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result, RewardMode, RewardProtocol, SamplingParams};

use crate::PromptOrder;
use crate::common::{
    parse_string_enum, require_non_negative_f32, require_nonzero, require_positive_f32, required,
    resolve, reward_protocol, sampling, validate_command,
};
use crate::document::SamplingToml;

/// Group baseline that centers rewards into advantages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvantageBaseline {
    /// Dr. GRPO default: subtract the group mean (over live members).
    Mean,
    /// RLOO: each member's baseline is the mean of the *other* live members,
    /// which rescales the advantage by `G/(G-1)` - the direction is identical
    /// but the corrected scale matters for small groups.
    LeaveOneOut,
}
/// DAPO soft overlong punishment: a progressive reward penalty over the last
/// `buffer_tokens` of the generation budget, applied before the group
/// baseline. Teaches the policy to conclude instead of merely ignoring
/// overruns (as `mask_truncated` does).
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverlongPenalty {
    pub buffer_tokens: usize,
    pub max_penalty: f32,
}

/// DAPO dynamic sampling: after the group baseline drops zero-signal groups,
/// keep drawing replacement prompts (continuing the round-robin) until the
/// update carries `prompts_per_update` informative groups, or the candidate
/// budget `prompts_per_update * max_resample_factor` is exhausted. Keeps the
/// effective batch full as the policy converges.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicSampling {
    /// Cap on candidate groups per update, as a multiple of
    /// `prompts_per_update`. Must be at least 2 (1 would allow no resampling).
    pub max_resample_factor: usize,
}

/// A judge next to the reward command, for the single-turn GRPO loop.
///
/// The judge scores a *group* - the `group_size` completions of one prompt,
/// compared against each other - which is exactly the unit GRPO centres its
/// advantage on. Its verdict never replaces the reward command: it is added to
/// it, weighted, so the verifiable part keeps deciding what it can decide and
/// the judge only separates the candidates it left tied.
#[derive(Clone, Debug)]
pub struct GrpoJudge {
    /// Backend and transport, shared verbatim with `[agent.judge]`.
    pub config: retrograd_spec::judge::JudgeConfig,
    /// Weight of the verdict in the final reward, for a completion whose reward
    /// line does not carry a `judge_weight` of its own. `reward + weight *
    /// verdict`, with `verdict` in [0, 1].
    pub weight: f32,
    pub failure: retrograd_agent_core::config::JudgeFailurePolicy,
    /// Share of an update's groups the judge may fail on before the run stops.
    pub max_dropped_fraction: f32,
}

impl GrpoJudge {
    fn validate(&self) -> Result<()> {
        require_non_negative_f32(self.weight, "grpo.judge_weight")?;
        if !self.max_dropped_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_dropped_fraction)
        {
            return Err(Error::config(
                "grpo.max_judge_dropped_fraction must be finite and in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Optional KL schedule/controller, only meaningful when `kl_coefficient > 0`.
#[derive(Clone, Copy, Debug)]
pub struct KlSchedule {
    /// Linear warmup: the effective coefficient ramps from 0 to
    /// `kl_coefficient` over the first `warmup_updates` updates.
    pub warmup_updates: u32,
    /// PPO-style adaptive target: after each update the effective coefficient
    /// is multiplied up/down to chase this measured KL. `None` keeps it fixed
    /// (at the warmed-up value).
    pub target: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct GrpoConfig {
    pub prompts: PathBuf,
    pub reward_command: Vec<String>,
    /// How that command is spoken to: one worker for the loop, or one process
    /// per batch, and the deadline of a batch either way.
    pub reward_protocol: RewardProtocol,
    pub updates: u32,
    pub prompts_per_update: usize,
    pub group_size: usize,
    pub grpo_epochs: u32,
    /// DAPO Clip-Higher: the surrogate clip is decoupled. `clip_range_low`
    /// bounds the ratio from below (`1 - low`), `clip_range_high` from above
    /// (`1 + high`). A larger upper range leaves room for low-probability
    /// tokens with positive advantage to grow, countering entropy collapse.
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    /// With rule-based / verifiable rewards, `0.0` (no KL anchor) is a good
    /// default; the anchor mostly slows learning there.
    pub kl_coefficient: f32,
    /// DAPO overlong filtering: when true, completions truncated at the
    /// generation budget are excluded from both the group baseline and the
    /// optimizer epochs, since their reward judges an incomplete response.
    pub mask_truncated: bool,
    /// Group baseline (mean vs. leave-one-out / RLOO). Defaults to `Mean`.
    pub baseline: AdvantageBaseline,
    /// Prompt draw order. Defaults to `Sequential` (unchanged behavior).
    pub prompt_order: PromptOrder,
    /// Optional DAPO soft overlong punishment.
    pub overlong_penalty: Option<OverlongPenalty>,
    /// Optional KL warmup / adaptive controller.
    pub kl_schedule: Option<KlSchedule>,
    /// Optional DAPO dynamic sampling (resample zero-signal groups).
    pub dynamic_sampling: Option<DynamicSampling>,
    /// Optional judge, blended into the reward command's score.
    pub judge: Option<GrpoJudge>,
    /// Consecutive fully zero-signal updates tolerated before the run stops.
    /// A stalled update is reported (a warning on every one of them, and
    /// `batch/trained_fraction` at zero) long before it is fatal: a model that
    /// cannot yet earn a reward at all is a normal early state, not a broken
    /// run. `0` disables the stop entirely and lets the loop run its
    /// `updates` out.
    pub max_stalled_updates: u32,
    pub sampling: SamplingParams,
}

/// Default for `grpo.max_stalled_updates`. Long enough that a hard prompt set
/// the policy has not cracked yet keeps its budget, short enough that a
/// collapsed or converged run does not silently burn a whole schedule.
pub const DEFAULT_MAX_STALLED_UPDATES: u32 = 25;

impl GrpoConfig {
    /// Validates both TOML-loaded and programmatically constructed GRPO
    /// configurations. Dr. GRPO is deliberately on-policy: sampling must use
    /// the unmodified model distribution that defines `pi_old`.
    pub fn validate(&self) -> Result<()> {
        validate_command(&self.reward_command)?;
        if self.updates == 0 || self.prompts_per_update == 0 || self.grpo_epochs == 0 {
            return Err(Error::config(
                "grpo updates, prompts_per_update, and grpo_epochs must be greater than zero",
            ));
        }
        if self.group_size < 2 {
            return Err(Error::config("grpo.group_size must be at least 2"));
        }
        if self.group_size > 256 {
            return Err(Error::config("grpo.group_size must not exceed 256"));
        }
        if !(self.clip_range_low > 0.0
            && self.clip_range_low < 1.0
            && self.clip_range_high > 0.0
            && self.clip_range_high < 1.0)
        {
            return Err(Error::config(
                "grpo.clip_range_low and grpo.clip_range_high must be between 0 and 1",
            ));
        }
        if self.clip_range_high < self.clip_range_low {
            return Err(Error::config(
                "grpo.clip_range_high must not be below grpo.clip_range_low (Clip-Higher)",
            ));
        }
        require_non_negative_f32(self.kl_coefficient, "grpo.kl_coefficient")?;
        if let Some(judge) = &self.judge {
            judge.validate()?;
        }
        require_nonzero(
            self.sampling.max_new_tokens,
            "grpo.sampling.max_new_tokens must be greater than zero",
        )?;
        if self.sampling.temperature != 1.0 || self.sampling.top_p != 1.0 {
            return Err(Error::config(
                "Dr. GRPO requires on-policy sampling with temperature = 1 and top_p = 1",
            ));
        }
        if let Some(penalty) = &self.overlong_penalty {
            if penalty.buffer_tokens == 0
                || penalty.buffer_tokens >= self.sampling.max_new_tokens as usize
            {
                return Err(Error::config(
                    "grpo.overlong_penalty.buffer_tokens must be in 1..max_new_tokens",
                ));
            }
            require_positive_f32(penalty.max_penalty, "grpo.overlong_penalty.max_penalty")?;
        }
        if let Some(schedule) = &self.kl_schedule {
            if self.kl_coefficient == 0.0 {
                return Err(Error::config(
                    "grpo.kl_schedule requires kl_coefficient > 0",
                ));
            }
            if let Some(target) = schedule.target {
                require_positive_f32(target, "grpo.kl_schedule.target")?;
            }
        }
        if let Some(dynamic) = &self.dynamic_sampling
            && dynamic.max_resample_factor < 2
        {
            return Err(Error::config(
                "grpo.dynamic_sampling.max_resample_factor must be at least 2",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GrpoToml {
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
    pub prompts_per_update: usize,
    pub group_size: usize,
    pub grpo_epochs: u32,
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    pub kl_coefficient: f32,
    #[serde(default)]
    pub mask_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_order: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlong_penalty: Option<OverlongPenalty>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl_schedule: Option<KlScheduleToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_sampling: Option<DynamicSampling>,
    /// `[grpo.judge]`, in the same spelling as `[agent.judge]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<retrograd_spec::judge::JudgeConfig>,
    /// Default weight of a verdict in the final reward. Required as soon as a
    /// judge is declared: a judge that contributes an unstated amount to the
    /// gradient is the one thing this section must not allow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_weight: Option<f32>,
    /// `"drop_group"` (default) or `"fail"`, as in `[agent].judge_failure`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_failure: Option<retrograd_agent_core::config::JudgeFailurePolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_judge_dropped_fraction: Option<f32>,
    /// Consecutive zero-signal updates tolerated before the run stops; `0`
    /// never stops. Defaults to `DEFAULT_MAX_STALLED_UPDATES`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_stalled_updates: Option<u32>,
    pub sampling: SamplingToml,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KlScheduleToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_updates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<f32>,
}
pub(crate) fn build_grpo(value: GrpoToml, root: &Path) -> Result<GrpoConfig> {
    let baseline = value
        .baseline
        .as_deref()
        .map(|name| {
            parse_string_enum!(
                name,
                "grpo.baseline must be mean or leave_one_out (rloo)",
                "mean" => AdvantageBaseline::Mean,
                "leave_one_out" => AdvantageBaseline::LeaveOneOut,
                "rloo" => AdvantageBaseline::LeaveOneOut,
            )
        })
        .transpose()?
        .unwrap_or(AdvantageBaseline::Mean);
    let prompt_order = value
        .prompt_order
        .as_deref()
        .map(|name| {
            parse_string_enum!(
                name,
                "grpo.prompt_order must be sequential or shuffled",
                "sequential" => PromptOrder::Sequential,
                "shuffled" => PromptOrder::Shuffled,
                "shuffle" => PromptOrder::Shuffled,
            )
        })
        .transpose()?
        .unwrap_or(PromptOrder::Sequential);
    let overlong_penalty = value.overlong_penalty;
    let kl_schedule = value.kl_schedule.map(|s| KlSchedule {
        warmup_updates: s.warmup_updates.unwrap_or(0),
        target: s.target,
    });
    let dynamic_sampling = value.dynamic_sampling;
    let judge = build_grpo_judge(
        value.judge,
        value.judge_weight,
        value.judge_failure,
        value.max_judge_dropped_fraction,
        root,
    )?;
    let config = GrpoConfig {
        prompts: resolve(root, value.prompts),
        reward_command: value.reward_command,
        reward_protocol: reward_protocol("grpo", value.reward_mode, value.reward_timeout_seconds)?,
        updates: value.updates,
        prompts_per_update: value.prompts_per_update,
        group_size: value.group_size,
        grpo_epochs: value.grpo_epochs,
        clip_range_low: value.clip_range_low,
        clip_range_high: value.clip_range_high,
        kl_coefficient: value.kl_coefficient,
        mask_truncated: value.mask_truncated,
        baseline,
        prompt_order,
        overlong_penalty,
        kl_schedule,
        dynamic_sampling,
        judge,
        max_stalled_updates: value
            .max_stalled_updates
            .unwrap_or(DEFAULT_MAX_STALLED_UPDATES),
        sampling: sampling(value.sampling)?,
    };
    config.validate()?;
    Ok(config)
}
/// `[grpo.judge]` and the three `[grpo]` keys that govern it.
///
/// The three are refused without the section rather than ignored: a
/// `judge_weight` written next to no judge is someone expecting a verdict in
/// their reward, and silence there is a training run that measures something
/// else than what its author read.
pub(crate) fn build_grpo_judge(
    judge: Option<retrograd_spec::judge::JudgeConfig>,
    weight: Option<f32>,
    failure: Option<retrograd_agent_core::config::JudgeFailurePolicy>,
    max_dropped_fraction: Option<f32>,
    root: &Path,
) -> Result<Option<GrpoJudge>> {
    let Some(mut config) = judge else {
        if weight.is_some() || failure.is_some() || max_dropped_fraction.is_some() {
            return Err(Error::config(
                "grpo.judge_weight, grpo.judge_failure and grpo.max_judge_dropped_fraction \
                 need a [grpo.judge] section",
            ));
        }
        return Ok(None);
    };
    let weight = required(
        weight,
        "grpo.judge_weight is required next to [grpo.judge]: it is how much of the \
         reward the verdict carries",
    )?;
    // The one path inside the judge that is written relative to the config file
    // - same treatment as `[agent.judge]`.
    if let retrograd_spec::judge::JudgeConfig::Ruler { config } = &mut config {
        config.cache_path = config
            .cache_path
            .take()
            .map(|path| resolve(root, path.to_path_buf()));
    }
    let judge = GrpoJudge {
        config,
        weight,
        failure: failure.unwrap_or_default(),
        max_dropped_fraction: max_dropped_fraction.unwrap_or(0.5),
    };
    judge.validate()?;
    Ok(Some(judge))
}
