//! Strict TOML configuration used by the training CLI.
//!
//! One document, one schema, whichever frontend reads it. `[model]`, `[lora]`,
//! `[training]`, `[metrics]`, `[evaluation]` and `[checkpoint]` are shared;
//! `[run].algorithm` selects which of `[sft]`, `[ppo]`, `[grpo]` or `[agent]`
//! is the run's own section, and exactly that one may be present.

mod agent;
#[cfg(test)]
mod round_trip;

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{
    CheckpointDtype, DEFAULT_REWARD_TIMEOUT_SECONDS, Device, Error, FeatureDtype, KvDtype,
    LoraConfig, LoraDtype, LrScheduler, Result, RewardMode, RewardProtocol, SamplingParams,
    SharedPrefixFanout, TargetSet, TrainConfig,
};
use retrograd_dataset::DataFormat;

pub use agent::{AgentRunConfig, AgentToml, ScenarioGenerationConfig};

macro_rules! parse_string_enum {
    ($value:expr_2021, $error:expr_2021, $($pattern:literal => $variant:expr_2021),+ $(,)?) => {{
        let normalized = $value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            $($pattern => Ok($variant),)+
            _ => Err(Error::config($error)),
        }
    }};
}

fn require_nonzero<T>(value: T, message: &str) -> Result<()>
where
    T: Default + PartialEq,
{
    if value == T::default() {
        return Err(Error::config(message));
    }
    Ok(())
}

fn require_positive_f32(value: f32, name: &str) -> Result<()> {
    if !(value > 0.0 && value.is_finite()) {
        return Err(Error::config(format!(
            "{name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

fn require_non_negative_f32(value: f32, name: &str) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(Error::config(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn require_non_negative_f64(value: f64, name: &str) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(Error::config(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

/// The engine-shaped configuration a run is built from. Produced from a
/// [`ConfigDocument`] by [`build`]/[`build_with`], which is the only path a
/// document may reach it by - see [`ConfigDocument`] for why.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub algorithm: Algorithm,
    pub model: PathBuf,
    pub lora: LoraRunConfig,
    pub training: TrainConfig,
    pub metrics: MetricsConfig,
    pub evaluation: Option<EvaluationConfig>,
    pub checkpoint: Option<CheckpointConfig>,
}

/// Which training objective a run carries, with that objective's own settings.
#[derive(Clone, Debug)]
pub enum Algorithm {
    Sft(SftConfig),
    Ppo(PpoConfig),
    Grpo(GrpoConfig),
    /// On-policy distillation against a frozen teacher. A rollout objective
    /// like `Grpo`, with the teacher's log-probability of the student's own
    /// token in place of a reward.
    Distill(DistillConfig),
    /// Multi-turn GRPO against a judge and an environment. A rollout objective
    /// like `Grpo`, with a trajectory in place of a completion.
    ///
    /// Boxed: `AgentRunConfig` carries the whole agentic declaration - scenario
    /// generation, environment, MCP servers - and inlining it would make every
    /// `Algorithm` the size of the largest variant, including the SFT run that
    /// holds three fields.
    AgentGrpo(Box<AgentRunConfig>),
}

#[derive(Clone, Debug)]
pub struct LoraRunConfig {
    pub config: LoraConfig,
    pub output: PathBuf,
    /// Existing adapter GGUF to resume training from, instead of creating a
    /// fresh adapter. Rank, alpha, and targets then come from the file.
    pub init_adapter: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct MetricsConfig {
    pub tensorboard_dir: Option<PathBuf>,
    pub wandb_export_dir: Option<PathBuf>,
}

/// Evaluation policy shared by SFT, PPO, and GRPO. An iteration is one SFT
/// epoch or one PPO/GRPO rollout update.
#[derive(Clone, Debug)]
pub struct EvaluationConfig {
    pub data: PathBuf,
    pub every_iterations: u32,
    pub patience: Option<u32>,
    pub min_delta: f64,
    /// Caps a rollout evaluation at this many prompts, taken evenly spaced
    /// across the dataset. Rollout evaluations generate one completion per
    /// prompt, so their cost scales with the dataset; SFT evaluation is a
    /// forward pass and ignores this.
    pub max_examples: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointMode {
    Steps,
    BestEval,
    StepsAndBestEval,
}

impl CheckpointMode {
    pub fn includes_steps(self) -> bool {
        matches!(self, Self::Steps | Self::StepsAndBestEval)
    }

    pub fn includes_best_eval(self) -> bool {
        matches!(self, Self::BestEval | Self::StepsAndBestEval)
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    pub directory: PathBuf,
    pub mode: CheckpointMode,
    pub every_steps: Option<u64>,
    /// Complete checkpoint to resume from: the `.state` directory, or the
    /// adapter GGUF whose sibling `.state` directory resolves unambiguously.
    /// A GGUF on its own is a cold adapter load (`lora.init_adapter`), never a
    /// resume, so the two are mutually exclusive.
    pub resume_from: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct SftConfig {
    pub data: PathBuf,
    pub data_format: DataFormat,
    /// Permute the training rows at the start of every epoch, seeded from
    /// `lora.seed`. On by default; see [`TrainConfig::shuffle_dataset`], which
    /// is where the runtime reads it from.
    pub shuffle: bool,
}

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

/// Order in which training prompts are drawn across updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptOrder {
    /// Strict round-robin over the prompt file (bit-exact reference behavior).
    Sequential,
    /// A seeded permutation re-drawn each full pass over the dataset, so
    /// correlated adjacent prompts do not always land in the same update.
    Shuffled,
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

/// Periodic JSONL dump of sampled completions for offline inspection of reward
/// hacking and collapse.
#[derive(Clone, Debug)]
pub struct CompletionLog {
    pub every: u32,
    pub path: PathBuf,
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
    /// Optional periodic completion journal.
    pub log_completions: Option<CompletionLog>,
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
        if let Some(log) = &self.log_completions {
            require_nonzero(
                log.every,
                "grpo.log_completions.every must be greater than zero",
            )?;
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

/// Which of the two distillation objectives a run carries.
///
/// They share a teacher and nothing else. On-policy generates, scores and
/// optimizes inside one update; offline top-k reads a distribution someone
/// already computed and never generates at all - so the two differ in their
/// memory budget, their resume unit and whether they need a sampler, and a
/// single flag deciding all of that is what keeps every `match` on `Algorithm`
/// honest about which one it is looking at.
#[derive(Clone, Debug, PartialEq)]
pub enum DistillMode {
    /// The student samples, the teacher notes the same tokens, the gap becomes
    /// a per-token advantage.
    OnPolicy,
    /// The teacher's truncated distribution over a fixed corpus, precomputed by
    /// `retrograd distill-teacher` and read from a sidecar.
    /// Nothing is generated: this is an SFT-shaped pass whose target is a sparse
    /// distribution instead of one token.
    TopkOffline(OfflineDistillConfig),
}

impl DistillMode {
    /// Whether the run generates. The offline path does not, which is what
    /// takes it out of every rollout-shaped budget and schedule.
    pub fn is_rollout(&self) -> bool {
        matches!(self, Self::OnPolicy)
    }

    pub fn offline(&self) -> Option<&OfflineDistillConfig> {
        match self {
            Self::TopkOffline(config) => Some(config),
            Self::OnPolicy => None,
        }
    }
}

/// The offline top-k path's own inputs.
///
/// `data` and `sidecar` are two halves of one artifact and are refused as a
/// pair, not separately: the sidecar's header carries the fingerprint of the
/// prepared corpus, and a mismatch there is the only thing standing between a
/// misaligned sidecar and a run that trains without complaint on the teacher's
/// distribution for *other tokens*.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineDistillConfig {
    /// Chat JSONL, the same shape an SFT run reads.
    pub data: PathBuf,
    /// The `.topk` sidecar `retrograd distill-teacher` wrote for `data`.
    pub sidecar: PathBuf,
    /// Passes over the corpus. The target does not move between them, so this
    /// is an SFT epoch count and not an optimizer-epoch count as in on-policy.
    pub epochs: u32,
}

/// Distillation against a frozen teacher, in one of two modes.
///
/// On-policy is GRPO's section minus everything that only a reward makes
/// meaningful: the student samples, the teacher scores the same tokens, and the
/// log-probability gap is the per-token advantage the existing weighted
/// objective consumes. Offline top-k replaces the sampler and the advantage by a
/// precomputed sparse distribution, and reads only the handful of fields
/// `DistillMode::TopkOffline` names.
#[derive(Clone, Debug)]
pub struct DistillConfig {
    /// Which objective this run carries.
    pub mode: DistillMode,
    /// The teacher GGUF. Held for inference only: it never gets an adapter, so
    /// it costs its weights plus its KV cache and nothing else.
    ///
    /// Read at run time on the on-policy path, and by `distill-teacher` on the
    /// offline one - where the training run itself never opens it, because the
    /// sidecar is what the teacher left behind.
    pub teacher_path: PathBuf,
    pub prompts: PathBuf,
    pub updates: u32,
    pub prompts_per_update: usize,
    /// Unlike `grpo.group_size`, one sample per prompt is admissible: the
    /// signal is dense and per-token, so a group of one still carries it. More
    /// than one buys prompt reuse across a shared prefix, not a baseline.
    pub samples_per_prompt: usize,
    /// `1` is strictly on-policy: the ratio `pi/pi_old` is exactly 1 and the
    /// token weight is the advantage itself. Beyond that the clipped surrogate
    /// applies, which is what `clip_range_*` below are for.
    pub distill_epochs: u32,
    /// Read only when `distill_epochs > 1`, for the same reason as in GRPO.
    pub clip_range_low: f32,
    pub clip_range_high: f32,
    /// Bound on `|A_t|`, in nats. Not cosmetic: on a token the student was
    /// confident about and the teacher was not, the gap runs to -20, and global
    /// gradient-norm clipping then rescales every other token to nearly
    /// nothing - one token owns the update.
    pub weight_clip: f32,
    /// The teacher already plays the anchor's role, so `0.0` is the default and
    /// the reference pass is skipped entirely. A value above zero adds the base
    /// model back on top of it.
    pub kl_coefficient: f32,
    /// Completions cut at the generation budget are excluded from the optimizer
    /// epochs: their tail was never sampled, so scoring it judges a response
    /// the student did not finish.
    pub mask_truncated: bool,
    /// Prompt draw order. Defaults to `Sequential`, as in GRPO.
    pub prompt_order: PromptOrder,
    pub sampling: SamplingParams,
}

impl DistillConfig {
    /// The teacher's existence is *not* checked here, deliberately: no path in
    /// this crate is stat'ed at build time, and the real gate is
    /// `Teacher::open` followed by `Teacher::compatibility`, which refuses an
    /// absent teacher and a disagreeing tokenizer with the same error type at
    /// the moment the run opens.
    pub fn validate(&self) -> Result<()> {
        if let Some(offline) = self.mode.offline() {
            if offline.epochs == 0 {
                return Err(Error::config(
                    "distill.offline_epochs must be greater than zero",
                ));
            }
            // The sampler, the update count and the clipping ranges describe a
            // generation loop this mode does not run. Rather than validate them
            // against a run that ignores them, the fields keep their defaults
            // and only what the offline path reads is checked.
            require_positive_f32(self.weight_clip, "distill.weight_clip")?;
            return Ok(());
        }
        if self.updates == 0 || self.prompts_per_update == 0 || self.distill_epochs == 0 {
            return Err(Error::config(
                "distill updates, prompts_per_update, and distill_epochs must be greater than zero",
            ));
        }
        if self.samples_per_prompt == 0 {
            return Err(Error::config(
                "distill.samples_per_prompt must be at least 1",
            ));
        }
        if self.samples_per_prompt > 256 {
            return Err(Error::config(
                "distill.samples_per_prompt must not exceed 256",
            ));
        }
        if !(self.clip_range_low > 0.0
            && self.clip_range_low < 1.0
            && self.clip_range_high > 0.0
            && self.clip_range_high < 1.0)
        {
            return Err(Error::config(
                "distill.clip_range_low and distill.clip_range_high must be between 0 and 1",
            ));
        }
        if self.clip_range_high < self.clip_range_low {
            return Err(Error::config(
                "distill.clip_range_high must not be below distill.clip_range_low (Clip-Higher)",
            ));
        }
        require_positive_f32(self.weight_clip, "distill.weight_clip")?;
        require_non_negative_f32(self.kl_coefficient, "distill.kl_coefficient")?;
        require_nonzero(
            self.sampling.max_new_tokens,
            "distill.sampling.max_new_tokens must be greater than zero",
        )?;
        // The behaviour log-probabilities the sampler records define `pi_old`,
        // and the advantage is a difference against *them*. A tempered or
        // truncated sampler would make them the log-probs of a distribution the
        // optimizer never updates.
        if self.sampling.temperature != 1.0 || self.sampling.top_p != 1.0 {
            return Err(Error::config(
                "distillation requires on-policy sampling with temperature = 1 and top_p = 1",
            ));
        }
        Ok(())
    }
}

/// The configuration *document*: the exact schema of the CLI's TOML file.
///
/// Public because it is the one schema three frontends share: the CLI reads it
/// from TOML, the server accepts it as JSON, and the resolver renders its own
/// output back through it. The resolver builds one of these instead of a
/// [`RunConfig`] directly, so that "the resolver cannot produce a config the
/// CLI would refuse" holds *by construction*: the only way from a document to
/// a `RunConfig` is [`build`], which is what `load` calls.
///
/// [`RunConfig`], by contrast, stays internal: it is the engine's shape and
/// changes at the engine's pace.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigDocument {
    pub run: RunToml,
    #[serde(default)]
    pub model: ModelToml,
    pub lora: LoraToml,
    #[serde(default)]
    pub training: TrainingToml,
    #[serde(default, skip_serializing_if = "MetricsToml::is_empty")]
    pub metrics: MetricsToml,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<CheckpointToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sft: Option<SftToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ppo: Option<PpoToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grpo: Option<GrpoToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill: Option<DistillToml>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentToml>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunToml {
    pub algorithm: String,
    #[serde(default)]
    pub verbose: bool,
}
/// `[model]` is optional as a *document* section because a frontend may supply
/// the base model itself - the CLI's `--model`/`--device`, a request field.
/// What is not optional is the resolved [`RunConfig::model`]: [`build_with`]
/// refuses a document whose path is neither written nor overridden.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraToml {
    pub output: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_adapter: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<LoraDtype>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct TrainingToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    /// Physical forward/backward width, in tokens - llama.cpp's `n_ubatch` and
    /// the equivalent of TRL's `per_device_train_batch_size`. This is the
    /// activation-memory lever.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub micro_batch: Option<u32>,
    /// Physical completion fanout for differentiable shared-prefix training:
    /// "auto", "off", "max", or an integer >= 2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_prefix_fanout: Option<SharedPrefixFanoutToml>,
    /// Micro-batches accumulated before one optimizer step - TRL's
    /// `gradient_accumulation_steps`. `micro_batch * gradient_accumulation` is
    /// the token window one AdamW step trains, and it must divide `ctx`.
    /// Optional on a rollout algorithm, where it is pinned to
    /// `ctx / micro_batch` (one step never spans less than a whole rollout).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_accumulation: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_scheduler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_steps: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast_sampling_context: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_dtype: Option<KvDtype>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_concurrency: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_batch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunked_cross_entropy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunked_ce_tiles: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunked_ce_seq_chunk: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_checkpointing: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_every_n_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_dtype: Option<CheckpointDtype>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_gpu_resident: Option<bool>,
    /// Upper target for the fraction of wall time this trainer spends waiting
    /// on GPU work it submitted, so a neighbouring workload gets regular
    /// compute windows. Finite, in `(0, 1]`; `1.0` is the unthrottled default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gpu_duty_cycle: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SharedPrefixFanoutToml {
    Name(String),
    Exact(u32),
}
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MetricsToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensorboard_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wandb_export_dir: Option<PathBuf>,
}

impl MetricsToml {
    fn is_empty(&self) -> bool {
        self.tensorboard_dir.is_none() && self.wandb_export_dir.is_none()
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationToml {
    pub data: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_iterations: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patience: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_examples: Option<usize>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointToml {
    pub directory: PathBuf,
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_steps: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<PathBuf>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SftToml {
    pub data: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_format: Option<String>,
    /// Shuffle the training rows between epochs, seeded from `lora.seed`.
    /// Defaults to `true`; set it to `false` to replay the file order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
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
    pub log_completions: Option<CompletionLogToml>,
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
/// `[distill]`. Every key that has a sensible value without being written has
/// one: a distillation run is described by a teacher, a prompt file and a
/// budget, and the rest is tuning.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistillToml {
    /// `on_policy` (the default) or `topk_offline`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Offline mode only: the chat JSONL the sidecar describes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<PathBuf>,
    /// Offline mode only: the `.topk` sidecar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecar: Option<PathBuf>,
    /// Offline mode only: passes over the corpus.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offline_epochs: Option<u32>,
    pub teacher_path: PathBuf,
    /// On-policy mode only. Optional in the schema and required by
    /// `build_distill` for that mode, rather than required here: an offline
    /// document has no prompt file, no update count and no sampler, and forcing
    /// it to write three ignored keys is how a reader ends up believing they
    /// mean something.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts_per_update: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_per_prompt: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill_epochs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_range_low: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_range_high: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_clip: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl_coefficient: Option<f32>,
    #[serde(default)]
    pub mask_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_order: Option<String>,
    /// On-policy mode only, same reasoning as `prompts`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingToml>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionLogToml {
    pub every: u32,
    pub path: PathBuf,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KlScheduleToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_updates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<f32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingToml {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: u32,
    pub seed: u32,
}

/// What a frontend may substitute for `[model]` without editing the document.
///
/// It is an *input* to [`build_with`] rather than a patch applied to the
/// resulting [`RunConfig`]: a document that names no `[model].path` must still
/// build when the caller supplies one, and that decision belongs to the same
/// function that would otherwise refuse it. A path here comes from the caller's
/// working directory, not the document's, so it is used as given.
#[derive(Clone, Debug, Default)]
pub struct ModelOverride {
    pub path: Option<PathBuf>,
    pub device: Option<Device>,
}

/// Reads, parses and builds the TOML file at `path` into a [`RunConfig`].
pub fn load(path: impl AsRef<Path>) -> Result<RunConfig> {
    load_with(path, ModelOverride::default())
}

/// [`load`] with the `[model]` fields a frontend supplies itself.
pub fn load_with(path: impl AsRef<Path>, overrides: ModelOverride) -> Result<RunConfig> {
    let path = path.as_ref();
    let span = tracing::info_span!(
        target: "retrograd::config",
        "config",
        path = %path.display()
    );
    let _entered = span.enter();
    let source = fs::read_to_string(path)?;
    let document = parse_toml(&source, &path.display().to_string())?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    build_with(document, root, overrides)
}

/// Parses a TOML document without reading a file, for the callers that already
/// hold the text (a request body, a test fixture). `origin` only names the
/// source in the error message.
pub fn parse_toml(source: &str, origin: &str) -> Result<ConfigDocument> {
    toml::from_str(source)
        .map_err(|error| Error::config(format!("{origin}: invalid TOML: {error}")))
}

fn resolve(root: &Path, value: PathBuf) -> PathBuf {
    if value.is_absolute() {
        value
    } else {
        root.join(value)
    }
}

/// Validates and resolves a parsed [`ConfigDocument`] into a [`RunConfig`],
/// relative paths joined against `root`. The only way from a document to a
/// `RunConfig`; [`load`] calls it after parsing the file.
pub fn build(file: ConfigDocument, root: &Path) -> Result<RunConfig> {
    build_with(file, root, ModelOverride::default())
}

/// [`build`] with the `[model]` fields a frontend supplies itself, taking
/// precedence over whatever the document wrote.
pub fn build_with(
    file: ConfigDocument,
    root: &Path,
    overrides: ModelOverride,
) -> Result<RunConfig> {
    let model = match (overrides.path, file.model.path) {
        (Some(path), _) => path,
        (None, Some(path)) => resolve(root, path),
        (None, None) => {
            return Err(Error::config(
                "[model].path is missing and no model was given on the command line",
            ));
        }
    };
    let mut training = TrainConfig::default();
    if let Some(value) = file.training.ctx {
        training.n_ctx = value;
    }
    if let Some(value) = file.training.micro_batch {
        training.n_ubatch = value;
    }
    if let Some(value) = &file.training.shared_prefix_fanout {
        training.shared_prefix_fanout = match value {
            SharedPrefixFanoutToml::Name(name) => parse_string_enum!(
                name,
                "training.shared_prefix_fanout must be 'auto', 'off', 'max', or an integer >= 2",
                "auto" => SharedPrefixFanout::Auto,
                "off" => SharedPrefixFanout::Off,
                "max" => SharedPrefixFanout::Max,
            )?,
            SharedPrefixFanoutToml::Exact(value) if *value >= 2 => {
                SharedPrefixFanout::Exact(*value)
            }
            SharedPrefixFanoutToml::Exact(_) => {
                return Err(Error::config(
                    "training.shared_prefix_fanout integer must be at least 2; use 'off' to disable it",
                ));
            }
        };
    }
    // `n_batch` - the token window of one optimizer step - is derived from
    // `micro_batch * gradient_accumulation`, and a rollout algorithm pins it to
    // the whole context. The algorithm is only known further down, so the
    // derivation happens there.
    if let Some(value) = file.training.threads {
        require_nonzero(value, "training.threads must be greater than zero")?;
        training.threads = value;
    }
    if let Some(value) = file.training.epochs {
        training.epochs = value;
    }
    if let Some(value) = file.training.lr {
        training.learning_rate = value;
    }
    if let Some(value) = file.training.weight_decay {
        training.weight_decay = value;
    }
    if let Some(value) = file.training.max_grad_norm {
        training.max_grad_norm = value;
    }
    if let Some(value) = file.training.warmup_steps {
        training.warmup_steps = value;
    }
    if let Some(value) = file.training.lr_scheduler {
        training.lr_scheduler = parse_scheduler(&value)?;
    }
    if let Some(value) = file.training.fast_sampling_context {
        training.fast_generation_context = value;
    }
    if let Some(value) = file.training.kv_dtype {
        training.kv_dtype = value;
    }
    if let Some(value) = file.training.chunked_cross_entropy {
        training.chunked_cross_entropy = value;
    }
    if let Some(value) = file.training.chunked_ce_tiles {
        require_nonzero(value, "training.chunked_ce_tiles must be greater than zero")?;
        training.chunked_ce_tiles = value;
    }
    if let Some(value) = file.training.chunked_ce_seq_chunk {
        training.chunked_ce_seq_chunk = value;
    }
    if let Some(value) = file.training.gradient_checkpointing {
        training.gradient_checkpointing = value;
    }
    if let Some(value) = file.training.checkpoint_every_n_layers {
        require_nonzero(
            value,
            "training.checkpoint_every_n_layers must be greater than zero",
        )?;
        training.checkpoint_every_n_layers = value;
    }
    if let Some(value) = file.training.checkpoint_dtype {
        // The runtime ignores this without checkpointing, and "ignored" is
        // indistinguishable from "applied" in every artifact the run produces,
        // a 16-bit checkpoint is a request to give up ~1e-3 of gradient
        // fidelity, so accepting it silently in a run that has no checkpoints
        // is the one answer that cannot be checked afterwards.
        if value != CheckpointDtype::F32 && !training.gradient_checkpointing {
            return Err(Error::config(
                "training.checkpoint_dtype only applies to retained activation checkpoints; \
                 set training.gradient_checkpointing = true or leave it at 'f32'",
            ));
        }
        training.checkpoint_dtype = value;
    }
    if let Some(value) = file.training.require_gpu_resident {
        training.require_gpu_resident = value;
    }
    if let Some(value) = file.training.max_gpu_duty_cycle {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(Error::config(
                "training.max_gpu_duty_cycle must be finite and in (0, 1]; \
                 use the run-control pause to stop a live run",
            ));
        }
        // An explicit `1.0` normalizes to the same `None` an omitted key does.
        // The two say the same thing - no limit - and collapsing them here is
        // what keeps a single disabled path in the runtime instead of one that
        // installs a limiter and then never sleeps.
        training.max_gpu_duty_cycle = (value < 1.0).then_some(value);
    }
    let requested_generation_concurrency = file.training.generation_concurrency;
    if let Some(value) = file.training.generation_batch {
        require_nonzero(value, "training.generation_batch must be greater than zero")?;
        training.generation_batch = value;
    }
    if let Some(value) = file.model.device {
        training.device = value.parse()?;
    }
    if let Some(device) = overrides.device {
        training.device = device;
    }
    training.verbose = file.run.verbose;
    require_nonzero(training.epochs, "training.epochs must be greater than zero")?;
    require_nonzero(training.n_ctx, "training.ctx must be greater than zero")?;
    require_nonzero(
        training.n_ubatch,
        "training.micro_batch must be greater than zero",
    )?;
    require_nonzero(
        file.training.gradient_accumulation.unwrap_or(1),
        "training.gradient_accumulation must be greater than zero",
    )?;
    require_positive_f32(training.learning_rate, "training.lr")?;
    require_non_negative_f32(training.weight_decay, "training.weight_decay")?;
    require_positive_f32(training.max_grad_norm, "training.max_grad_norm")?;

    if file.lora.init_adapter.is_some()
        && (file.lora.rank.is_some()
            || file.lora.alpha.is_some()
            || file.lora.seed.is_some()
            || file.lora.dtype.is_some()
            || !file.lora.targets.is_empty())
    {
        return Err(Error::config(
            "lora.init_adapter cannot be combined with lora.rank, lora.alpha, \
             lora.seed, lora.dtype, or lora.targets: they come from the adapter file",
        ));
    }
    let mut lora = LoraConfig::auto(file.lora.rank.unwrap_or(8), file.lora.alpha.unwrap_or(16.0));
    lora.seed = file.lora.seed.unwrap_or(42);
    lora.targets = if file.lora.targets.is_empty() {
        parse_targets(&DEFAULT_TARGETS.map(String::from))?
    } else {
        parse_targets(&file.lora.targets)?
    };
    lora.dtype = file.lora.dtype.unwrap_or_default();
    require_nonzero(lora.rank, "lora.rank must be greater than zero")?;
    require_positive_f32(lora.alpha, "lora.alpha")?;
    let algorithm_name = file.run.algorithm.to_ascii_lowercase();
    // One run trains one way. Every algorithm section other than the selected
    // one is a leftover from an edit, and silently ignoring it is how a config
    // ends up describing a run nobody is having.
    for (name, present) in [
        ("sft", file.sft.is_some()),
        ("ppo", file.ppo.is_some()),
        ("grpo", file.grpo.is_some()),
        ("distill", file.distill.is_some()),
        ("agent", file.agent.is_some()),
    ] {
        let selected = match name {
            "agent" => algorithm_name == "agent_grpo",
            other => algorithm_name == other,
        };
        if present && !selected {
            return Err(Error::config(format!(
                "only the section for the selected algorithm may be present: \
                 run.algorithm = '{algorithm_name}' but [{name}] is set"
            )));
        }
    }
    let algorithm = match algorithm_name.as_str() {
        "sft" => {
            let value = required(file.sft, "[sft] is required when run.algorithm = 'sft'")?;
            let data = resolve(root, value.data);
            let data_format = match value
                .data_format
                .as_deref()
                .map(parse_data_format)
                .transpose()?
            {
                Some(format) => format,
                // Resolved against `root` first: an unknown extension makes
                // `infer` open the file to sniff its content, and a relative
                // path is only valid once joined with `root`.
                None => DataFormat::infer(&data)?,
            };
            let shuffle = value.shuffle.unwrap_or(true);
            // The runtime owns the row cursor inside an epoch, so the shuffle
            // is applied there and reaches it through `TrainConfig`. One seed
            // describes the whole run: reusing `lora.seed` keeps the number of
            // knobs at one, and the permutation is a function of it and the
            // epoch index alone.
            training.shuffle_dataset = shuffle;
            training.shuffle_seed = lora.seed as u64;
            Algorithm::Sft(SftConfig {
                data,
                data_format,
                shuffle,
            })
        }
        "ppo" => {
            let value = required(file.ppo, "[ppo] is required when run.algorithm = 'ppo'")?;
            Algorithm::Ppo(build_ppo(value, root)?)
        }
        "grpo" => {
            let value = required(file.grpo, "[grpo] is required when run.algorithm = 'grpo'")?;
            Algorithm::Grpo(build_grpo(value, root)?)
        }
        "distill" => {
            let value = required(
                file.distill,
                "[distill] is required when run.algorithm = 'distill'",
            )?;
            Algorithm::Distill(build_distill(value, root)?)
        }
        "agent_grpo" => {
            let value = required(
                file.agent,
                "[agent] is required when run.algorithm = 'agent_grpo'",
            )?;
            Algorithm::AgentGrpo(Box::new(agent::build_agent(value, root, resolve)?))
        }
        _ => {
            return Err(Error::config(
                "run.algorithm must be one of sft, ppo, grpo, distill, or agent_grpo",
            ));
        }
    };
    // `n_batch` is not spelled in the file: it is `micro_batch` times
    // `gradient_accumulation`. A rollout algorithm has only one admissible
    // value - the whole context - so the field is optional there and the
    // default is the pinned one; a value that contradicts it is still reported
    // rather than silently overridden.
    // Offline top-k distillation is a `Distill` run that does not generate, so it
    // takes the SFT side of every rule below - the optimizer window, the geometry
    // validation, and the sequence widths a sampler would otherwise pin.
    let rollout = match &algorithm {
        Algorithm::Ppo(_) | Algorithm::Grpo(_) | Algorithm::AgentGrpo(_) => true,
        Algorithm::Distill(distill) => distill.mode.is_rollout(),
        Algorithm::Sft(_) => false,
    };
    let accumulation = match (file.training.gradient_accumulation, rollout) {
        (Some(value), _) => value,
        (None, true) => training.n_ctx.div_ceil(training.n_ubatch),
        (None, false) => 1,
    };
    training.n_batch = training
        .n_ubatch
        .checked_mul(accumulation)
        .ok_or_else(|| Error::overflow("training.micro_batch * gradient_accumulation overflows"))?;
    // Both rules live on `TrainConfig` so every frontend that can build a
    // rollout run applies the same ones - this loader, the planner, and
    // `retrograd-agent`'s own TOML.
    if rollout {
        training.validate_rollout_geometry()?;
    } else {
        training.validate_geometry()?;
    }
    // Not `grpo_geometry` any more: three algorithms generate, and what this
    // pins - `n_seq_max` and `generation_concurrency` - is the geometry of a
    // rollout rather than of a group baseline. Distillation reads
    // `samples_per_prompt` where GRPO reads `group_size`; they are the same
    // number to the sampler, which branches a prompt that many ways.
    let rollout_geometry = match &algorithm {
        Algorithm::Grpo(grpo) => Some(("grpo", grpo.group_size, grpo.prompts_per_update)),
        Algorithm::Distill(distill) if distill.mode.is_rollout() => Some((
            "distill",
            distill.samples_per_prompt,
            distill.prompts_per_update,
        )),
        Algorithm::AgentGrpo(agent) => Some((
            "agent",
            agent.config.group_size,
            agent.config.scenarios_per_update,
        )),
        _ => None,
    };
    if let Some((section, group_size, prompts_per_update)) = rollout_geometry {
        if group_size > training.n_batch as usize {
            return Err(Error::config(format!(
                "{section}.group_size must not exceed the optimizer window \
                 (training.micro_batch * training.gradient_accumulation) for batched generation"
            )));
        }
        // Optimizer packing and rollout generation have independent sequence
        // widths. A logical GRPO group may be generated over several waves.
        //
        // The cast cannot lose bits: the guard above refuses
        // `group_size > training.n_batch as usize`, and `n_batch` is a `u32`
        // (the proof stays here rather than becoming a
        // `try_from`, whose error arm would be unreachable and whose placement
        // before the guard would refuse the value for the wrong reason).
        training.n_seq_max = group_size as u32;
        if let SharedPrefixFanout::Exact(fanout) = training.shared_prefix_fanout
            && fanout > training.n_seq_max
        {
            return Err(Error::config(format!(
                "training.shared_prefix_fanout ({fanout}) exceeds {section}.group_size ({group_size})"
            )));
        }
        let concurrent_sequences = prompts_per_update
            .checked_mul(group_size)
            .ok_or_else(|| Error::overflow("continuous generation sequence count overflows"))?;
        let default_concurrency =
            concurrent_sequences.min(training.n_batch as usize).min(256) as u32;
        training.generation_concurrency =
            requested_generation_concurrency.unwrap_or(default_concurrency);
        if training.generation_concurrency == 0 {
            return Err(Error::config(
                "training.generation_concurrency must be greater than zero",
            ));
        }
        if training.generation_concurrency > training.n_batch {
            return Err(Error::config(
                "training.generation_concurrency must not exceed the optimizer window \
                 (training.micro_batch * training.gradient_accumulation)",
            ));
        }
        if training.generation_concurrency > 256 {
            return Err(Error::config(
                "training.generation_concurrency must not exceed 256",
            ));
        }
        if training.generation_concurrency as usize > concurrent_sequences {
            return Err(Error::config(format!(
                "training.generation_concurrency must not exceed the {concurrent_sequences} \
                 {section} rollouts per update"
            )));
        }
    } else if requested_generation_concurrency.is_some() {
        return Err(Error::config(
            "training.generation_concurrency is only supported for a generating algorithm \
             (grpo, distill, agent_grpo)",
        ));
    }
    let evaluation = file
        .evaluation
        .map(|value| build_evaluation(value, root))
        .transpose()?;
    let checkpoint = file
        .checkpoint
        .map(|value| build_checkpoint(value, root))
        .transpose()?;
    // An agentic evaluation grades a trajectory with the reward its environment
    // put on it - a verify command, a test suite, a task's own grading. The
    // judge cannot stand in: every RULER strategy scores the members of a group
    // against each other, so its scores are renormalized at every update and a
    // mean over them is not comparable across the run. A configuration that
    // asks to be evaluated with nothing that grades is refused here rather than
    // at the first evaluation, an hour into the run.
    if let Algorithm::AgentGrpo(agent) = &algorithm
        && evaluation.is_some()
        && agent.environment.is_none()
    {
        return Err(Error::config(
            "[evaluation] with run.algorithm = 'agent_grpo' needs [agent.environment]: an \
                 agentic evaluation measures the reward the environment puts on a trajectory, \
                 and a judge cannot stand in - its scores are relative inside a group and not \
                 comparable across updates",
        ));
    }
    if checkpoint
        .as_ref()
        .is_some_and(|value| value.mode.includes_best_eval())
        && evaluation.is_none()
    {
        return Err(Error::config(
            "checkpoint.mode includes best_eval but [evaluation] is missing",
        ));
    }
    // A resume owns the adapter it restores; combining it with a cold adapter
    // load would leave which weights actually train ambiguous.
    if checkpoint
        .as_ref()
        .is_some_and(|value| value.resume_from.is_some())
        && file.lora.init_adapter.is_some()
    {
        return Err(Error::config(
            "checkpoint.resume_from and lora.init_adapter are mutually exclusive",
        ));
    }

    Ok(RunConfig {
        algorithm,
        model,
        lora: LoraRunConfig {
            config: lora,
            output: resolve(root, file.lora.output),
            init_adapter: file.lora.init_adapter.map(|path| resolve(root, path)),
        },
        training,
        metrics: MetricsConfig {
            tensorboard_dir: file.metrics.tensorboard_dir.map(|path| resolve(root, path)),
            wandb_export_dir: file
                .metrics
                .wandb_export_dir
                .map(|path| resolve(root, path)),
        },
        evaluation,
        checkpoint,
    })
}

fn build_evaluation(value: EvaluationToml, root: &Path) -> Result<EvaluationConfig> {
    let every_iterations = value.every_iterations.unwrap_or(1);
    require_nonzero(
        every_iterations,
        "evaluation.every_iterations must be greater than zero",
    )?;
    if let Some(patience) = value.patience {
        require_nonzero(
            patience,
            "evaluation.patience must be greater than zero when set",
        )?;
    }
    let min_delta = value.min_delta.unwrap_or(0.0);
    require_non_negative_f64(min_delta, "evaluation.min_delta")?;
    if let Some(max_examples) = value.max_examples {
        require_nonzero(
            max_examples,
            "evaluation.max_examples must be greater than zero when set",
        )?;
    }
    Ok(EvaluationConfig {
        data: resolve(root, value.data),
        every_iterations,
        patience: value.patience,
        min_delta,
        max_examples: value.max_examples,
    })
}

fn build_checkpoint(value: CheckpointToml, root: &Path) -> Result<CheckpointConfig> {
    let mode = parse_string_enum!(
        value.mode,
        "checkpoint.mode must be steps, best_eval, or steps_and_best_eval",
        "steps" => CheckpointMode::Steps,
        "best_eval" => CheckpointMode::BestEval,
        "steps_and_best_eval" => CheckpointMode::StepsAndBestEval,
    )?;
    if mode.includes_steps() && value.every_steps.is_none_or(|steps| steps == 0) {
        return Err(Error::config(
            "checkpoint.every_steps must be greater than zero when mode includes steps",
        ));
    }
    if !mode.includes_steps() && value.every_steps.is_some() {
        return Err(Error::config(
            "checkpoint.every_steps is only valid when mode includes steps",
        ));
    }
    Ok(CheckpointConfig {
        directory: resolve(root, value.directory),
        mode,
        every_steps: value.every_steps,
        resume_from: value.resume_from.map(|path| resolve(root, path)),
    })
}

fn required<T>(value: Option<T>, message: &'static str) -> Result<T> {
    value.ok_or_else(|| Error::config(message))
}
fn build_ppo(value: PpoToml, root: &Path) -> Result<PpoConfig> {
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
fn build_grpo(value: GrpoToml, root: &Path) -> Result<GrpoConfig> {
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
    let log_completions = value.log_completions.map(|c| CompletionLog {
        every: c.every,
        path: resolve(root, c.path),
    });
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
        log_completions,
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

/// Defaults of `[distill]`, named because they are choices rather than zeros.
/// `weight_clip` is in nats and is picked in the same order of magnitude as
/// `REFERENCE_LOG_RATIO_LIMIT` in `rollout::weights`, for the same reason: past
/// it, one token owns the update.
pub const DEFAULT_DISTILL_WEIGHT_CLIP: f32 = 5.0;
/// One epoch per batch keeps the run strictly on-policy - the ratio is exactly
/// 1 and the token weight is the advantage itself.
pub const DEFAULT_DISTILL_EPOCHS: u32 = 1;
/// The clip range is only read above one epoch; the pair is GRPO's Clip-Higher
/// default, so a run that raises `distill_epochs` behaves like the objective it
/// borrows rather than like an unstated third thing.
pub const DEFAULT_DISTILL_CLIP_RANGE_LOW: f32 = 0.2;
pub const DEFAULT_DISTILL_CLIP_RANGE_HIGH: f32 = 0.28;

fn build_distill(value: DistillToml, root: &Path) -> Result<DistillConfig> {
    let prompt_order = value
        .prompt_order
        .as_deref()
        .map(|name| {
            parse_string_enum!(
                name,
                "distill.prompt_order must be sequential or shuffled",
                "sequential" => PromptOrder::Sequential,
                "shuffled" => PromptOrder::Shuffled,
                "shuffle" => PromptOrder::Shuffled,
            )
        })
        .transpose()?
        .unwrap_or(PromptOrder::Sequential);
    // `mode` selects the objective, and the offline one is refused without the
    // two files it reads - naming the mode and forgetting the sidecar is the one
    // mistake that would otherwise surface as an on-policy run nobody asked for.
    let mode = match value.mode.as_deref().unwrap_or("on_policy") {
        "on_policy" | "onpolicy" => {
            if value.data.is_some() || value.sidecar.is_some() || value.offline_epochs.is_some() {
                return Err(Error::config(
                    "distill.data, distill.sidecar and distill.offline_epochs belong to \
                     distill.mode = \"topk_offline\"",
                ));
            }
            DistillMode::OnPolicy
        }
        "topk_offline" => DistillMode::TopkOffline(OfflineDistillConfig {
            data: resolve(
                root,
                required(
                    value.data,
                    "distill.data is required with distill.mode = \"topk_offline\": \
                     it is the corpus the sidecar describes",
                )?,
            ),
            sidecar: resolve(
                root,
                required(
                    value.sidecar,
                    "distill.sidecar is required with distill.mode = \"topk_offline\": \
                     it is the teacher's precomputed distribution",
                )?,
            ),
            epochs: value.offline_epochs.unwrap_or(1),
        }),
        other => {
            return Err(Error::config(format!(
                "distill.mode must be on_policy or topk_offline, got {other:?}"
            )));
        }
    };
    let on_policy = mode.is_rollout();
    let require_on_policy = |missing: &str| -> Error {
        Error::config(format!(
            "{missing} is required with distill.mode = \"on_policy\""
        ))
    };
    let config = DistillConfig {
        teacher_path: resolve(root, value.teacher_path),
        prompts: match value.prompts {
            Some(path) => resolve(root, path),
            None if on_policy => return Err(require_on_policy("distill.prompts")),
            // Never read on the offline path; the corpus is `distill.data`.
            None => PathBuf::new(),
        },
        updates: match value.updates {
            Some(value) => value,
            None if on_policy => return Err(require_on_policy("distill.updates")),
            None => 0,
        },
        prompts_per_update: match value.prompts_per_update {
            Some(value) => value,
            None if on_policy => return Err(require_on_policy("distill.prompts_per_update")),
            None => 0,
        },
        mode,
        samples_per_prompt: value.samples_per_prompt.unwrap_or(1),
        distill_epochs: value.distill_epochs.unwrap_or(DEFAULT_DISTILL_EPOCHS),
        clip_range_low: value
            .clip_range_low
            .unwrap_or(DEFAULT_DISTILL_CLIP_RANGE_LOW),
        clip_range_high: value
            .clip_range_high
            .unwrap_or(DEFAULT_DISTILL_CLIP_RANGE_HIGH),
        weight_clip: value.weight_clip.unwrap_or(DEFAULT_DISTILL_WEIGHT_CLIP),
        kl_coefficient: value.kl_coefficient.unwrap_or(0.0),
        mask_truncated: value.mask_truncated,
        prompt_order,
        sampling: match value.sampling {
            Some(value) => sampling(value)?,
            None if on_policy => return Err(require_on_policy("distill.sampling")),
            None => SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens: 1,
                seed: 0,
            },
        },
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
fn build_grpo_judge(
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
fn critic(value: CriticToml) -> Result<CriticConfig> {
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
/// The reward transport of a `[ppo]` or `[grpo]` section, defaults included.
/// Written once for both because it is one contract: the same two keys, the
/// same refusal, and a reward process that cannot tell which section called it.
fn reward_protocol(
    section: &str,
    mode: Option<RewardMode>,
    timeout_seconds: Option<u64>,
) -> Result<RewardProtocol> {
    let seconds = timeout_seconds.unwrap_or(DEFAULT_REWARD_TIMEOUT_SECONDS);
    if seconds == 0 {
        return Err(Error::config(format!(
            "{section}.reward_timeout_seconds must be greater than zero"
        )));
    }
    Ok(RewardProtocol {
        mode: mode.unwrap_or_default(),
        timeout: std::time::Duration::from_secs(seconds),
    })
}

fn validate_command(command: &[String]) -> Result<()> {
    if command.is_empty() || command[0].trim().is_empty() {
        Err(Error::config("reward_command must contain an executable"))
    } else {
        Ok(())
    }
}
fn sampling(value: SamplingToml) -> Result<SamplingParams> {
    require_positive_f32(value.temperature, "sampling.temperature")?;
    if !(value.top_p > 0.0 && value.top_p <= 1.0 && value.top_p.is_finite()) {
        return Err(Error::config("sampling.top_p must be in (0, 1]"));
    }
    require_nonzero(
        value.max_new_tokens,
        "sampling.max_new_tokens must be greater than zero",
    )?;
    Ok(SamplingParams {
        temperature: value.temperature,
        top_p: value.top_p,
        max_new_tokens: value.max_new_tokens,
        seed: value.seed,
    })
}
fn parse_scheduler(value: &str) -> Result<LrScheduler> {
    parse_string_enum!(
        value,
        "training.lr_scheduler must be constant, linear, or cosine",
        "constant" => LrScheduler::Constant,
        "linear" => LrScheduler::Linear,
        "cosine" => LrScheduler::Cosine,
    )
}
fn parse_data_format(value: &str) -> Result<DataFormat> {
    parse_string_enum!(
        value,
        "sft.data_format must be text or jsonl",
        "text" => DataFormat::Text,
        "txt" => DataFormat::Text,
        "jsonl" => DataFormat::ChatJsonl,
        "chat" => DataFormat::ChatJsonl,
        "chat-jsonl" => DataFormat::ChatJsonl,
    )
}
/// Target set a config gets when `lora.targets` is absent: every attention and
/// feed-forward projection, the set that actually trains well. `targets =
/// ['auto']` is still how you defer to the runtime's per-architecture choice.
pub const DEFAULT_TARGETS: [&str; 7] = ["q", "k", "v", "o", "ffn_up", "ffn_down", "ffn_gate"];

/// Expands LoRA target aliases (`q`, `v`, `ffn_up`,...) into tensor patterns.
/// Shared by the TOML config and the `preflight` CLI command.
pub fn parse_targets(values: &[String]) -> Result<TargetSet> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("auto"))
    {
        if values.len() <= 1 {
            return Ok(TargetSet::Auto);
        }
        return Err(Error::config(
            "lora.targets = ['auto'] cannot be combined with other targets",
        ));
    }
    let mut patterns = Vec::new();
    for value in values {
        let pattern = match value.as_str() {
            "q" => "blk.*.attn_q.weight",
            "k" => "blk.*.attn_k.weight",
            "v" => "blk.*.attn_v.weight",
            "o" => "blk.*.attn_output.weight",
            "ffn_up" => "blk.*.ffn_up.weight",
            "ffn_down" => "blk.*.ffn_down.weight",
            "ffn_gate" => "blk.*.ffn_gate.weight",
            custom if custom.contains('*') || custom.contains(".weight") => custom,
            _ => return Err(Error::config(format!("unknown LoRA target '{value}'"))),
        };
        patterns.push(pattern.to_string());
    }
    Ok(TargetSet::Patterns(patterns))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_config(source: &str) -> PathBuf {
        // Tests run in parallel and a nanosecond timestamp is not unique on its
        // own: two of them can land on the same directory and delete each
        // other's config on teardown. The counter makes the name collision-free.
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "retrograd-config-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        let file = path.join("run.toml");
        fs::write(&file, source).unwrap();
        file
    }

    fn remove_config(file: &Path) {
        fs::remove_dir_all(file.parent().unwrap()).unwrap();
    }

    /// The shipped examples are the documentation a reader copies first, so a
    /// rename of the `[training]` surface has to reach them or they teach a
    /// spelling the loader rejects. `load` does not touch the filesystem beyond
    /// the document itself, so this is a validation pass, not a run.
    #[test]
    fn every_shipped_example_still_loads() {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .join("examples");
        let mut checked = 0;
        let mut pending = vec![examples];
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir).expect("read examples") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "toml") {
                    continue;
                }
                // A params-only document has no `[run]`: it is a partial tree for
                // the server's resolver, not a configuration.
                let source = fs::read_to_string(&path).expect("read example");
                if !source.lines().any(|line| line.trim_end() == "[run]") {
                    continue;
                }
                // The shipped examples point at the CPU fixture, which this lane
                // does not fetch, so it supplies a path: what is under test is
                // the rest of the document, not where the GGUF lives.
                let overrides = ModelOverride {
                    path: Some(PathBuf::from("model.gguf")),
                    device: None,
                };
                load_with(&path, overrides)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                checked += 1;
            }
        }
        // The three smoke configurations: SFT, PPO and GRPO.
        assert!(checked >= 3, "only {checked} examples were checked");
    }

    fn valid_sampling() -> SamplingToml {
        SamplingToml {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 7,
        }
    }

    fn valid_grpo() -> GrpoConfig {
        GrpoConfig {
            prompts: "prompts.jsonl".into(),
            reward_command: vec!["reward".into()],
            reward_protocol: RewardProtocol::default(),
            updates: 1,
            prompts_per_update: 2,
            group_size: 2,
            grpo_epochs: 1,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            mask_truncated: false,
            baseline: AdvantageBaseline::Mean,
            prompt_order: PromptOrder::Sequential,
            overlong_penalty: None,
            log_completions: None,
            kl_schedule: None,
            dynamic_sampling: None,
            judge: None,
            max_stalled_updates: DEFAULT_MAX_STALLED_UPDATES,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens: 8,
                seed: 7,
            },
        }
    }

    fn valid_ppo_toml() -> PpoToml {
        PpoToml {
            prompts: "prompts.jsonl".into(),
            reward_command: vec!["reward".into()],
            reward_mode: None,
            reward_timeout_seconds: None,
            updates: 1,
            rollout_batch_size: 2,
            ppo_epochs: 1,
            clip_range: 0.2,
            kl_coefficient: 0.1,
            critic: CriticToml::default(),
            sampling: valid_sampling(),
        }
    }

    #[test]
    fn targets_expand_aliases() {
        let TargetSet::Patterns(patterns) = parse_targets(&[
            "q".into(),
            "v".into(),
            "ffn_gate".into(),
            "blk.0.ffn_up.weight".into(),
        ])
        .unwrap() else {
            panic!("explicit targets should produce patterns")
        };
        assert_eq!(
            patterns,
            [
                "blk.*.attn_q.weight",
                "blk.*.attn_v.weight",
                "blk.*.ffn_gate.weight",
                "blk.0.ffn_up.weight",
            ]
        );
        assert!(matches!(parse_targets(&[]).unwrap(), TargetSet::Auto));
        assert!(parse_targets(&["auto".into(), "q".into()]).is_err());
        assert!(parse_targets(&["not-a-target".into()]).is_err());
    }

    /// An omitted `lora.targets` trains every projection; only an explicit
    /// `['auto']` hands the choice back to the runtime.
    #[test]
    fn omitted_targets_default_to_every_projection() {
        let base =
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n";
        let default_file = write_config(&format!("{base}[sft]\ndata='data.txt'\n"));
        assert_eq!(
            load(&default_file).unwrap().lora.config.targets,
            TargetSet::Patterns(vec![
                "blk.*.attn_q.weight".into(),
                "blk.*.attn_k.weight".into(),
                "blk.*.attn_v.weight".into(),
                "blk.*.attn_output.weight".into(),
                "blk.*.ffn_up.weight".into(),
                "blk.*.ffn_down.weight".into(),
                "blk.*.ffn_gate.weight".into(),
            ])
        );
        remove_config(&default_file);

        let auto_file = write_config(&format!("{base}targets=['auto']\n[sft]\ndata='data.txt'\n"));
        assert_eq!(
            load(&auto_file).unwrap().lora.config.targets,
            TargetSet::Auto
        );
        remove_config(&auto_file);
    }
    #[test]
    fn sampling_validates_every_numeric_boundary() {
        let parsed = sampling(valid_sampling()).unwrap();
        assert_eq!(parsed.temperature, 1.0);
        assert_eq!(parsed.top_p, 1.0);
        assert_eq!(parsed.max_new_tokens, 8);
        assert_eq!(parsed.seed, 7);

        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut value = valid_sampling();
            value.temperature = invalid;
            assert!(sampling(value).is_err(), "temperature={invalid}");
        }
        for invalid in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            let mut value = valid_sampling();
            value.top_p = invalid;
            assert!(sampling(value).is_err(), "top_p={invalid}");
        }
        let mut value = valid_sampling();
        value.max_new_tokens = 0;
        assert!(sampling(value).is_err());
    }
    #[test]
    fn resolves_paths_from_the_config_directory() {
        let file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        );
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.model, file.parent().unwrap().join("model.gguf"));
        assert_eq!(loaded.training.max_grad_norm, 1.0);
        assert!(
            matches!(loaded.algorithm, Algorithm::Sft(SftConfig { data, .. }) if data == file.parent().unwrap().join("data.txt"))
        );
        remove_config(&file);
    }

    /// A document may leave `[model]` out entirely when the frontend supplies
    /// it - that is what `train --model` is. The override is an input to the
    /// build, so the document that has no path still has to build; and the
    /// override's path is the caller's, not resolved against the document's
    /// directory the way a written one is.
    #[test]
    fn the_model_override_stands_in_for_a_missing_model_section() {
        let file = write_config(
            "[run]\nalgorithm='sft'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        );
        let error = load(&file).unwrap_err().to_string();
        assert!(error.contains("[model].path is missing"), "{error}");

        let loaded = load_with(
            &file,
            ModelOverride {
                path: Some(PathBuf::from("elsewhere/model.gguf")),
                device: Some(Device::Cpu),
            },
        )
        .unwrap();
        assert_eq!(loaded.model, PathBuf::from("elsewhere/model.gguf"));
        assert_eq!(loaded.training.device, Device::Cpu);
        remove_config(&file);
    }

    /// `--model` wins over a written `[model].path`, and `--device` over
    /// `[model].device`: the flag exists to swap a model without editing the
    /// TOML, which it would not do if the document had the last word.
    #[test]
    fn the_model_override_outranks_a_written_model_section() {
        let file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\ndevice='gpu'\n\
             [lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        );
        let loaded = load_with(
            &file,
            ModelOverride {
                path: Some(PathBuf::from("/other/model.gguf")),
                device: Some(Device::Cpu),
            },
        )
        .unwrap();
        assert_eq!(loaded.model, PathBuf::from("/other/model.gguf"));
        assert_eq!(loaded.training.device, Device::Cpu);
        remove_config(&file);
    }

    #[test]
    fn sft_shuffles_by_default_seeded_from_the_lora_seed() {
        let default_file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\nseed=7\n[sft]\ndata='data.txt'\n",
        );
        let loaded = load(&default_file).unwrap();
        assert!(matches!(loaded.algorithm, Algorithm::Sft(SftConfig { shuffle, .. }) if shuffle));
        assert!(loaded.training.shuffle_dataset);
        // The runtime reads the seed from `training`, so the two must agree: a
        // shuffle silently running on 42 while the config names 7 is exactly
        // the class of bug this pair exists to prevent.
        assert_eq!(loaded.training.shuffle_seed, 7);
        remove_config(&default_file);

        let off_file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\nshuffle=false\n",
        );
        let loaded = load(&off_file).unwrap();
        assert!(matches!(loaded.algorithm, Algorithm::Sft(SftConfig { shuffle, .. }) if !shuffle));
        assert!(!loaded.training.shuffle_dataset);
        assert_eq!(loaded.training.shuffle_seed, 42, "the lora.seed default");
        remove_config(&off_file);
    }

    #[test]
    fn lora_dtype_defaults_to_f16_and_accepts_only_f16_or_f32() {
        let default_file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        );
        assert_eq!(
            load(&default_file).unwrap().lora.config.dtype,
            LoraDtype::F16
        );
        remove_config(&default_file);

        let f32_file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ndtype='f32'\n[sft]\ndata='data.txt'\n",
        );
        assert_eq!(load(&f32_file).unwrap().lora.config.dtype, LoraDtype::F32);
        remove_config(&f32_file);

        let invalid_file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ndtype='bf16'\n[sft]\ndata='data.txt'\n",
        );
        assert!(
            load(&invalid_file)
                .unwrap_err()
                .to_string()
                .contains("dtype")
        );
        remove_config(&invalid_file);
    }

    #[test]
    fn evaluation_and_checkpoint_policy_are_shared_and_resolve_paths() {
        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[evaluation]\ndata='eval.txt'\nevery_iterations=2\npatience=3\nmin_delta=0.01\nmax_examples=16\n",
            "[checkpoint]\ndirectory='checkpoints'\nmode='steps_and_best_eval'\nevery_steps=10\n",
            "[sft]\ndata='train.txt'\n",
        ));
        let loaded = load(&file).unwrap();
        let evaluation = loaded.evaluation.unwrap();
        assert_eq!(evaluation.data, file.parent().unwrap().join("eval.txt"));
        assert_eq!(evaluation.every_iterations, 2);
        assert_eq!(evaluation.patience, Some(3));
        assert_eq!(evaluation.min_delta, 0.01);
        assert_eq!(evaluation.max_examples, Some(16));
        let checkpoint = loaded.checkpoint.unwrap();
        assert_eq!(
            checkpoint.directory,
            file.parent().unwrap().join("checkpoints")
        );
        assert_eq!(checkpoint.mode, CheckpointMode::StepsAndBestEval);
        assert_eq!(checkpoint.every_steps, Some(10));
        remove_config(&file);
    }

    #[test]
    fn resume_from_resolves_and_excludes_a_cold_adapter_load() {
        let base = "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[sft]\ndata='train.txt'\n";
        let checkpoint = "[checkpoint]\ndirectory='ckpt'\nmode='steps'\nevery_steps=2\nresume_from='ckpt/step-000000000010.state'\n";

        let file = write_config(&format!("{base}[lora]\noutput='out.gguf'\n{checkpoint}"));
        let loaded = load(&file).unwrap();
        assert_eq!(
            loaded.checkpoint.unwrap().resume_from,
            Some(file.parent().unwrap().join("ckpt/step-000000000010.state"))
        );
        remove_config(&file);

        // A resume restores its own adapter, so pairing it with a cold adapter
        // load would leave which weights actually train ambiguous.
        let file = write_config(&format!(
            "{base}[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n{checkpoint}"
        ));
        let error = load(&file).unwrap_err().to_string();
        assert!(error.contains("mutually exclusive"), "{error}");
        remove_config(&file);
    }

    #[test]
    fn evaluation_and_checkpoint_defaults_and_dependencies_are_validated() {
        let base = "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='train.txt'\n";

        let file = write_config(&format!("{base}[evaluation]\ndata='eval.txt'\n"));
        let evaluation = load(&file).unwrap().evaluation.unwrap();
        assert_eq!(evaluation.every_iterations, 1);
        assert_eq!(evaluation.patience, None);
        assert_eq!(evaluation.min_delta, 0.0);
        assert_eq!(evaluation.max_examples, None);
        remove_config(&file);

        for section in [
            "[evaluation]\ndata='eval.txt'\nevery_iterations=0\n",
            "[evaluation]\ndata='eval.txt'\npatience=0\n",
            "[evaluation]\ndata='eval.txt'\nmin_delta=-0.1\n",
            "[evaluation]\ndata='eval.txt'\nmax_examples=0\n",
            "[checkpoint]\ndirectory='ckpt'\nmode='steps'\n",
            "[checkpoint]\ndirectory='ckpt'\nmode='best_eval'\n",
            "[checkpoint]\ndirectory='ckpt'\nmode='best_eval'\nevery_steps=2\n[evaluation]\ndata='eval.txt'\n",
            "[checkpoint]\ndirectory='ckpt'\nmode='sometimes'\n",
        ] {
            let file = write_config(&format!("{base}{section}"));
            assert!(
                load(&file).is_err(),
                "section should be rejected: {section}"
            );
            remove_config(&file);
        }
    }

    #[test]
    fn algorithm_sections_no_longer_accept_evaluation_data() {
        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[sft]\ndata='train.txt'\neval_data='eval.txt'\n",
        ));
        assert!(
            load(&file)
                .unwrap_err()
                .to_string()
                .contains("invalid TOML")
        );
        remove_config(&file);
    }

    #[test]
    fn an_omitted_or_unit_duty_cycle_normalizes_to_the_unthrottled_path() {
        // The two say the same thing - no limit - and collapsing them here is
        // what keeps a single disabled path in the runtime rather than one that
        // installs a limiter and then never sleeps.
        for training in ["", "[training]\nmax_gpu_duty_cycle=1.0\n"] {
            let file = write_config(&format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n\
                 [lora]\noutput='out.gguf'\n{training}[sft]\ndata='data.txt'\n"
            ));
            let loaded = load(&file).unwrap();
            assert_eq!(loaded.training.max_gpu_duty_cycle, None, "{training:?}");
            remove_config(&file);
        }

        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nmax_gpu_duty_cycle=0.5\n",
            "[sft]\ndata='data.txt'\n",
        ));
        assert_eq!(load(&file).unwrap().training.max_gpu_duty_cycle, Some(0.5));
        remove_config(&file);
    }

    #[test]
    fn a_cpu_device_accepts_a_duty_cycle_it_cannot_honour() {
        // `auto` can resolve to CPU too, so rejecting the document at parse
        // time would be wrong. The runtime keeps the requested value and its
        // report says `active: false, reason: cpu_backend` instead.
        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\ndevice='cpu'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nmax_gpu_duty_cycle=0.25\n",
            "[sft]\ndata='data.txt'\n",
        ));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.max_gpu_duty_cycle, Some(0.25));
        assert_eq!(loaded.training.device, Device::Cpu);
        remove_config(&file);
    }

    #[test]
    fn training_gradient_clip_is_configurable() {
        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nmax_grad_norm=0.5\nthreads=6\n",
            "[sft]\ndata='data.txt'\n",
        ));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.max_grad_norm, 0.5);
        assert_eq!(loaded.training.threads, 6);
        remove_config(&file);
    }

    /// The fused cross-entropy is on with a bounded token chunk unless the file
    /// says otherwise, and both knobs have to survive the TOML round trip: a
    /// silently dropped key would look exactly like a feature that does not
    /// work.
    #[test]
    fn chunked_ce_is_on_by_default_and_round_trips() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
            )
        };

        let defaults = write_config(&source(""));
        let loaded = load(&defaults).unwrap();
        assert!(loaded.training.chunked_cross_entropy);
        assert_eq!(
            loaded.training.chunked_ce_seq_chunk,
            retrograd_core::DEFAULT_CE_SEQ_CHUNK
        );
        remove_config(&defaults);

        let tuned = write_config(&source(concat!(
            "chunked_cross_entropy=false\nchunked_ce_tiles=4\n",
            "chunked_ce_seq_chunk=256\n",
        )));
        let loaded = load(&tuned).unwrap();
        assert!(!loaded.training.chunked_cross_entropy);
        assert_eq!(loaded.training.chunked_ce_tiles, 4);
        assert_eq!(loaded.training.chunked_ce_seq_chunk, 256);
        remove_config(&tuned);
    }

    /// The in-place `grad_h` write follows `chunked_ce_seq_chunk` now, so the
    /// key is rejected rather than accepted and ignored.
    #[test]
    fn the_offload_logsoftmax_key_is_rejected() {
        let file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n\
             [training]\nchunked_ce_offload_logsoftmax=true\n[sft]\ndata='data.txt'\n",
        );
        let error = load(&file).unwrap_err().to_string();
        assert!(error.contains("chunked_ce_offload_logsoftmax"), "{error}");
        remove_config(&file);
    }

    /// A 16-bit checkpoint in a run that retains no checkpoints is a request for
    /// less precision that nothing would honour, and no artifact of the run
    /// would show it was dropped.
    #[test]
    fn a_16bit_checkpoint_dtype_needs_checkpointing() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
            )
        };

        let orphan = write_config(&source("checkpoint_dtype='f16'\n"));
        let error = load(&orphan).unwrap_err().to_string();
        assert!(error.contains("checkpoint_dtype"), "{error}");
        remove_config(&orphan);

        let paired = write_config(&source(
            "gradient_checkpointing=true\ncheckpoint_dtype='f16'\n",
        ));
        assert_eq!(
            load(&paired).unwrap().training.checkpoint_dtype,
            CheckpointDtype::F16
        );
        remove_config(&paired);

        // F32 is the default, so naming it explicitly is not a request for
        // anything and must not depend on checkpointing.
        let explicit = write_config(&source("checkpoint_dtype='f32'\n"));
        assert_eq!(
            load(&explicit).unwrap().training.checkpoint_dtype,
            CheckpointDtype::F32
        );
        remove_config(&explicit);
    }

    #[test]
    fn gradient_checkpointing_is_opt_in_and_validates_its_interval() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
            )
        };

        let defaults = write_config(&source(""));
        let loaded = load(&defaults).unwrap();
        assert!(!loaded.training.gradient_checkpointing);
        // Not 1: a checkpoint on every layer is the largest retained term
        // checkpointing can produce, for the same extra forward.
        assert_eq!(
            loaded.training.checkpoint_every_n_layers,
            retrograd_core::DEFAULT_CHECKPOINT_STRIDE
        );
        remove_config(&defaults);

        let enabled = write_config(&source(
            "gradient_checkpointing=true\ncheckpoint_every_n_layers=3\n",
        ));
        let loaded = load(&enabled).unwrap();
        assert!(loaded.training.gradient_checkpointing);
        assert_eq!(loaded.training.checkpoint_every_n_layers, 3);
        remove_config(&enabled);

        let invalid = write_config(&source("checkpoint_every_n_layers=0\n"));
        assert!(
            load(&invalid)
                .unwrap_err()
                .to_string()
                .contains("checkpoint_every_n_layers")
        );
        remove_config(&invalid);
    }

    #[test]
    fn fast_sampling_context_defaults_to_fast_and_can_be_disabled() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{training}[sft]\ndata='data.txt'\n"
            )
        };
        for (training, expected) in [
            ("", true),
            ("[training]\nfast_sampling_context=false\n", false),
            ("[training]\nfast_sampling_context=true\n", true),
        ] {
            let file = write_config(&source(training));
            let loaded = load(&file).unwrap();
            assert_eq!(
                loaded.training.fast_generation_context, expected,
                "{training:?}"
            );
            remove_config(&file);
        }
    }

    #[test]
    fn training_kv_dtype_is_f16_by_default_and_accepts_f32() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
            )
        };

        // The half-precision cache is the default: it halves the one term that
        // grows with the context, and the runtime probes the device before
        // applying it rather than assuming.
        let defaults = write_config(&source(""));
        assert_eq!(load(&defaults).unwrap().training.kv_dtype, KvDtype::F16);
        remove_config(&defaults);

        let f32 = write_config(&source("kv_dtype='f32'\n"));
        assert_eq!(load(&f32).unwrap().training.kv_dtype, KvDtype::F32);
        remove_config(&f32);

        let invalid = write_config(&source("kv_dtype='bf16'\n"));
        assert!(load(&invalid).unwrap_err().to_string().contains("kv_dtype"));
        remove_config(&invalid);
    }

    #[test]
    fn training_checkpoint_dtype_defaults_to_f32_and_rejects_unknown_precisions() {
        let source = |training: &str| {
            format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
            )
        };

        // The bit-exact recompute stays the default: narrowing costs real gradient
        // fidelity, so it must be asked for.
        let defaults = write_config(&source(""));
        assert_eq!(
            load(&defaults).unwrap().training.checkpoint_dtype,
            CheckpointDtype::F32
        );
        remove_config(&defaults);

        for (spelling, expected) in [
            ("f16", CheckpointDtype::F16),
            ("bf16", CheckpointDtype::Bf16),
        ] {
            let file = write_config(&source(&format!(
                "gradient_checkpointing=true\ncheckpoint_dtype='{spelling}'\n"
            )));
            assert_eq!(load(&file).unwrap().training.checkpoint_dtype, expected);
            remove_config(&file);
        }

        // A silently ignored precision would read as a memory win that never
        // happened, so an unknown spelling has to fail loudly.
        let invalid = write_config(&source(
            "gradient_checkpointing=true\ncheckpoint_dtype='q8_0'\n",
        ));
        assert!(
            load(&invalid)
                .unwrap_err()
                .to_string()
                .contains("checkpoint_dtype")
        );
        remove_config(&invalid);
    }

    #[test]
    fn scientific_notation_is_accepted_for_float_settings() {
        let file = write_config(concat!(
            "[run]\nalgorithm='sft'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\nalpha=1.6e1\n",
            "[training]\nlr=5e-6\nweight_decay=1E-2\nmax_grad_norm=1.5e0\n",
            "[sft]\ndata='data.txt'\n",
        ));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.learning_rate, 5e-6);
        assert_eq!(loaded.training.weight_decay, 1e-2);
        assert_eq!(loaded.training.max_grad_norm, 1.5);
        assert_eq!(loaded.lora.config.alpha, 16.0);
        remove_config(&file);
    }

    #[test]
    fn init_adapter_resolves_and_rejects_creation_keys() {
        let file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n[sft]\ndata='data.txt'\n",
        );
        let loaded = load(&file).unwrap();
        assert_eq!(
            loaded.lora.init_adapter,
            Some(file.parent().unwrap().join("adapter.gguf"))
        );
        remove_config(&file);

        for conflicting in [
            "rank=4",
            "alpha=8.0",
            "seed=1",
            "dtype='f16'",
            "targets=['q']",
        ] {
            let source = format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n{conflicting}\n[sft]\ndata='data.txt'\n"
            );
            let file = write_config(&source);
            let error = load(&file).unwrap_err();
            assert!(
                error.to_string().contains("init_adapter"),
                "{conflicting}: {error}"
            );
            remove_config(&file);
        }
    }

    #[test]
    fn rejects_unknown_keys_before_model_loading() {
        let file = write_config(
            "[run]\nalgorithm='sft'\nunknown=true\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
        );
        assert!(
            load(&file)
                .unwrap_err()
                .to_string()
                .contains("invalid TOML")
        );
        remove_config(&file);
    }

    #[test]
    fn training_and_lora_values_are_rejected_before_model_loading() {
        let cases = [
            ("ctx=0", "training.ctx must be greater than zero"),
            (
                "micro_batch=0",
                "training.micro_batch must be greater than zero",
            ),
            (
                "gradient_accumulation=0",
                "training.gradient_accumulation must be greater than zero",
            ),
            // 96 tokens per step does not tile a 128-token context.
            (
                "micro_batch=48\ngradient_accumulation=2",
                "must be a multiple of the optimizer window",
            ),
            ("lr=0.0", "lr must be finite"),
            ("lr=nan", "lr must be finite"),
            ("weight_decay=-0.1", "weight_decay"),
            ("weight_decay=nan", "weight_decay"),
            ("max_grad_norm=0.0", "max_grad_norm"),
            ("max_grad_norm=-1.0", "max_grad_norm"),
            ("max_grad_norm=nan", "max_grad_norm"),
            // Zero is not a spelling for pause: the run-control pause operation
            // is already the safe way to stop a live run.
            ("max_gpu_duty_cycle=0.0", "max_gpu_duty_cycle"),
            ("max_gpu_duty_cycle=-0.5", "max_gpu_duty_cycle"),
            ("max_gpu_duty_cycle=1.5", "max_gpu_duty_cycle"),
            ("max_gpu_duty_cycle=nan", "max_gpu_duty_cycle"),
            ("max_gpu_duty_cycle=inf", "max_gpu_duty_cycle"),
        ];
        for (training, expected) in cases {
            let source = format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}\n[sft]\ndata='data.txt'\n"
            );
            let file = write_config(&source);
            let error = load(&file).unwrap_err();
            assert!(error.to_string().contains(expected), "{training}: {error}");
            remove_config(&file);
        }

        for alpha in ["0.0", "-1.0", "nan"] {
            let source = format!(
                "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\nalpha={alpha}\n[sft]\ndata='data.txt'\n"
            );
            let file = write_config(&source);
            assert!(load(&file).unwrap_err().to_string().contains("lora.alpha"));
            remove_config(&file);
        }
    }

    #[test]
    fn critic_defaults_and_boundaries_are_validated() {
        let defaults = critic(CriticToml::default()).unwrap();
        assert!(defaults.enabled);
        assert_eq!(defaults.gamma, 1.0);
        assert_eq!(defaults.gae_lambda, 0.95);
        assert_eq!(defaults.value_lr, 1.0e-2);
        assert_eq!(defaults.value_epochs, 8);
        // O8 is opt-in: the default keeps the feature matrix in F32, so enabling
        // a critic never silently rounds what the value head regresses on.
        assert_eq!(defaults.feature_dtype, FeatureDtype::F32);
        for (spelling, expected) in [
            ("f32", FeatureDtype::F32),
            ("f16", FeatureDtype::F16),
            ("bf16", FeatureDtype::Bf16),
        ] {
            let parsed: FeatureDtype = toml::from_str(&format!("v='{spelling}'"))
                .map(|table: toml::Value| table["v"].clone())
                .and_then(FeatureDtype::deserialize)
                .expect("a documented spelling parses");
            assert_eq!(parsed, expected);
        }
        // A narrowing asked for on a run with no critic has no matrix to narrow;
        // silently ignoring it would leave the document uncontradicted.
        assert!(
            critic(CriticToml {
                enabled: Some(false),
                feature_dtype: Some(FeatureDtype::F16),
                ..CriticToml::default()
            })
            .is_err()
        );
        assert!(
            critic(CriticToml {
                enabled: Some(false),
                ..CriticToml::default()
            })
            .is_ok()
        );

        for invalid in [0.0, -0.1, 1.1, f32::NAN] {
            assert!(
                critic(CriticToml {
                    gamma: Some(invalid),
                    ..CriticToml::default()
                })
                .is_err()
            );
        }
        for invalid in [-0.1, 1.1, f32::NAN] {
            assert!(
                critic(CriticToml {
                    gae_lambda: Some(invalid),
                    ..CriticToml::default()
                })
                .is_err()
            );
        }
        assert!(
            critic(CriticToml {
                value_epochs: Some(0),
                ..CriticToml::default()
            })
            .is_err()
        );
        for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert!(
                critic(CriticToml {
                    value_lr: Some(invalid),
                    ..CriticToml::default()
                })
                .is_err()
            );
        }
    }

    #[test]
    fn ppo_validation_rejects_invalid_geometry_and_objective_values() {
        build_ppo(valid_ppo_toml(), Path::new("/config")).unwrap();

        let mut value = valid_ppo_toml();
        value.reward_command.clear();
        assert!(build_ppo(value, Path::new("/config")).is_err());

        let mut value = valid_ppo_toml();
        value.rollout_batch_size = 0;
        assert!(build_ppo(value, Path::new("/config")).is_err());

        for invalid in [0.0, -0.1, 1.0, f32::NAN, f32::INFINITY] {
            let mut value = valid_ppo_toml();
            value.clip_range = invalid;
            assert!(
                build_ppo(value, Path::new("/config")).is_err(),
                "clip={invalid}"
            );
        }
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            let mut value = valid_ppo_toml();
            value.kl_coefficient = invalid;
            assert!(
                build_ppo(value, Path::new("/config")).is_err(),
                "kl={invalid}"
            );
        }
    }

    /// The transport of the reward command: defaulted when the document says
    /// nothing, spelled the same way in both sections, and refused at zero -
    /// a deadline of zero would fail every batch on its first millisecond.
    #[test]
    fn the_reward_transport_defaults_to_one_persistent_worker() {
        let ppo = build_ppo(valid_ppo_toml(), Path::new("/config")).unwrap();
        assert_eq!(ppo.reward_protocol, RewardProtocol::default());
        assert_eq!(ppo.reward_protocol.mode, RewardMode::Persistent);

        let mut value = valid_ppo_toml();
        value.reward_mode = Some(RewardMode::OneShot);
        value.reward_timeout_seconds = Some(12);
        let ppo = build_ppo(value, Path::new("/config")).unwrap();
        assert_eq!(ppo.reward_protocol.mode, RewardMode::OneShot);
        assert_eq!(
            ppo.reward_protocol.timeout,
            std::time::Duration::from_secs(12)
        );

        let mut value = valid_ppo_toml();
        value.reward_timeout_seconds = Some(0);
        let error = build_ppo(value, Path::new("/config")).unwrap_err();
        assert!(
            error.to_string().contains("ppo.reward_timeout_seconds"),
            "{error}"
        );

        // The wire spelling is the one the documents use, and an unknown one is
        // refused by name rather than defaulted to.
        let document: GrpoToml = toml::from_str(
            "prompts = 'p.jsonl'\nreward_command = ['r']\nreward_mode = 'oneshot'\n\
             updates = 1\nprompts_per_update = 1\ngroup_size = 2\ngrpo_epochs = 1\n\
             clip_range_low = 0.2\nclip_range_high = 0.2\nkl_coefficient = 0.0\n\
             [sampling]\ntemperature = 1.0\ntop_p = 1.0\nmax_new_tokens = 8\nseed = 1\n",
        )
        .unwrap();
        assert_eq!(document.reward_mode, Some(RewardMode::OneShot));
        let mut value = document.clone();
        value.reward_timeout_seconds = Some(0);
        let error = build_grpo(value, Path::new("/config")).unwrap_err();
        assert!(
            error.to_string().contains("grpo.reward_timeout_seconds"),
            "{error}"
        );
        let grpo = build_grpo(document, Path::new("/config")).unwrap();
        assert_eq!(grpo.reward_protocol.mode, RewardMode::OneShot);
        assert_eq!(
            grpo.reward_protocol.timeout,
            std::time::Duration::from_secs(DEFAULT_REWARD_TIMEOUT_SECONDS)
        );
        assert!(
            toml::from_str::<GrpoToml>(
                "prompts = 'p.jsonl'\nreward_command = ['r']\nreward_mode = 'socket'\n\
                 updates = 1\nprompts_per_update = 1\ngroup_size = 2\ngrpo_epochs = 1\n\
                 clip_range_low = 0.2\nclip_range_high = 0.2\nkl_coefficient = 0.0\n\
                 [sampling]\ntemperature = 1.0\ntop_p = 1.0\nmax_new_tokens = 8\nseed = 1\n",
            )
            .is_err()
        );
    }

    #[test]
    fn parsers_accept_documented_aliases_and_reject_unknown_values() {
        assert_eq!(
            parse_scheduler(" CONSTANT ").unwrap(),
            LrScheduler::Constant
        );
        assert_eq!(parse_scheduler("linear").unwrap(), LrScheduler::Linear);
        assert_eq!(parse_scheduler("Cosine").unwrap(), LrScheduler::Cosine);
        assert!(parse_scheduler("cyclic").is_err());

        for alias in ["text", "txt"] {
            assert_eq!(parse_data_format(alias).unwrap(), DataFormat::Text);
        }
        for alias in ["jsonl", "chat", "chat-jsonl"] {
            assert_eq!(parse_data_format(alias).unwrap(), DataFormat::ChatJsonl);
        }
        assert!(parse_data_format("csv").is_err());
    }

    #[test]
    fn selected_algorithm_requires_exactly_its_own_section() {
        for (algorithm, section, expected) in [
            ("sft", "", "[sft] is required"),
            ("ppo", "", "[ppo] is required"),
            ("grpo", "", "[grpo] is required"),
            ("agent_grpo", "", "[agent] is required"),
            ("unknown", "", "must be one of"),
            (
                "sft",
                "[sft]\ndata='data.txt'\n[ppo]\nprompts='p.jsonl'\nreward_command=['r']\nupdates=1\nrollout_batch_size=1\nppo_epochs=1\nclip_range=0.2\nkl_coefficient=0.0\n[ppo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=1\nseed=1\n",
                "only the section",
            ),
        ] {
            let source = format!(
                "[run]\nalgorithm='{algorithm}'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{section}"
            );
            let file = write_config(&source);
            let error = load(&file).unwrap_err();
            assert!(error.to_string().contains(expected), "{algorithm}: {error}");
            remove_config(&file);
        }
    }

    /// The agentic loop shares one document with the other algorithms. Shared
    /// sections use the common schema, while sections it cannot honor are
    /// rejected explicitly.
    #[test]
    fn an_agentic_document_shares_every_section_it_can_and_refuses_the_rest() {
        let base = concat!(
            "[run]\nalgorithm='agent_grpo'\n",
            "[model]\npath='model.gguf'\ndevice='cpu'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nctx=2048\nmicro_batch=64\nlr=1e-5\nlr_scheduler='constant'\n",
            "[metrics]\ntensorboard_dir='tb'\n",
            "[agent]\nscenarios='s.jsonl'\nupdates=3\nscenarios_per_update=2\n",
            "group_size=4\nepochs_per_update=2\nmax_new_tokens_per_turn=128\n",
            "system_suffix='Reply with one tool call.'\n",
            "template_variables={ enable_thinking = false }\n",
            "[agent.judge]\ntype='command'\ncommand=['python','judge.py']\n",
        );
        let file = write_config(base);
        let config = load(&file).expect("the agentic document must load");
        remove_config(&file);
        let Algorithm::AgentGrpo(agent) = &config.algorithm else {
            panic!("expected an agentic algorithm");
        };
        assert_eq!(agent.config.updates, 3);
        assert_eq!(agent.config.epochs, 2);
        assert_eq!(agent.config.limits.max_new_tokens_per_turn, 128);
        assert_eq!(agent.system_suffix, "Reply with one tool call.");
        // A TOML table crosses into the JSON object the runtime hands the template; `false`
        // must stay a boolean, not become the string "false", because a template branching on
        // it would read any string as truthy.
        assert_eq!(
            agent.template_variables_json(),
            r#"{"enable_thinking":false}"#
        );
        assert!(agent.scenarios.ends_with("s.jsonl"));
        // The shared sections are read by the same code as every other
        // algorithm.
        assert_eq!(config.training.lr_scheduler, LrScheduler::Constant);
        assert_eq!(config.training.device, retrograd_core::Device::Cpu);
        assert!(config.metrics.tensorboard_dir.is_some());
        // A rollout objective: one optimizer step spans the whole context.
        assert_eq!(config.training.n_batch, config.training.n_ctx);
        // Agent GRPO uses the same batched generation geometry as single-turn
        // GRPO: both scenarios' four-member groups fit in one decode wave.
        assert_eq!(config.training.n_seq_max, 4);
        assert_eq!(config.training.generation_concurrency, 8);
        // Unset means "the whole context", decided against the loaded model.
        assert_eq!(agent.trajectory_limit(2048).unwrap(), 2048);

        for (extra, expected) in [
            // Nothing grades a trajectory here, so there is no evaluation to be
            // had - see the loader's comment on why the judge cannot stand in.
            (
                "[evaluation]\ndata='eval.jsonl'\n",
                "needs [agent.environment]",
            ),
            (
                "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\nupdates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\nclip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=1\nseed=1\n",
                "only the section",
            ),
            // An unknown key is rejected rather than treated as a second dialect.
            (
                "[agent.environment]\ntype='local'\nallow_unsandboxd=true\n",
                "invalid TOML",
            ),
        ] {
            let file = write_config(&format!("{base}{extra}"));
            let error = load(&file).unwrap_err().to_string();
            remove_config(&file);
            assert!(error.contains(expected), "expected {expected}, got {error}");
        }
    }

    /// Evaluation and checkpointing are shared machinery, and an agentic run
    /// uses the same sections for them as every other algorithm - as long as
    /// something grades its trajectories.
    #[test]
    fn an_agentic_run_evaluates_and_checkpoints_like_any_other() {
        let source = concat!(
            "[run]\nalgorithm='agent_grpo'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nctx=2048\nmicro_batch=64\n",
            "[agent]\nscenarios='s.jsonl'\ngroup_size=4\n",
            "[agent.judge]\ntype='command'\ncommand=['judge']\n",
            "[agent.environment]\ntype='local'\nallow_unsandboxed=true\n",
            "[evaluation]\ndata='eval.jsonl'\nevery_iterations=2\npatience=3\n",
            "[checkpoint]\ndirectory='ckpt'\nmode='steps_and_best_eval'\nevery_steps=10\n",
        );
        let file = write_config(source);
        let config = load(&file).expect("an environment-graded agentic run may be evaluated");
        remove_config(&file);
        let evaluation = config.evaluation.expect("[evaluation] is kept");
        assert_eq!(evaluation.every_iterations, 2);
        assert_eq!(evaluation.patience, Some(3));
        let checkpoint = config.checkpoint.expect("[checkpoint] is kept");
        assert!(checkpoint.mode.includes_steps() && checkpoint.mode.includes_best_eval());
    }

    /// A trajectory budget the model cannot hold is a configuration to fix, not
    /// a number to clamp: the loss denominator is derived from it.
    #[test]
    fn an_explicit_trajectory_budget_is_checked_against_the_model_context() {
        let source = concat!(
            "[run]\nalgorithm='agent_grpo'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nctx=2048\nmicro_batch=64\n",
            "[agent]\nscenarios='s.jsonl'\nmax_trajectory_tokens=4096\n",
            "[agent.judge]\ntype='command'\ncommand=['judge']\n",
        );
        let file = write_config(source);
        let config = load(&file).expect("the document itself is valid");
        remove_config(&file);
        let Algorithm::AgentGrpo(agent) = &config.algorithm else {
            panic!("expected an agentic algorithm");
        };
        assert_eq!(agent.trajectory_limit(8192).unwrap(), 4096);
        assert!(
            agent
                .trajectory_limit(2048)
                .unwrap_err()
                .to_string()
                .contains("exceeds the model context")
        );
    }

    /// Scenarios and *a source of reward* are the two things a rollout cannot
    /// invent, and `[agent]`'s defaults would otherwise stand in for both. The
    /// second is a pair: a judge, an environment that grades its own steps, or
    /// both - never neither.
    #[test]
    fn an_agentic_run_without_scenarios_or_any_reward_is_refused() {
        for (section, expected) in [
            ("[agent]\nupdates=1\n", "agent.scenarios is required"),
            (
                "[agent]\nscenarios='s.jsonl'\n",
                "neither a judge nor an environment",
            ),
            // Declared and empty is a mistake, not a way of asking for no
            // judge: the way of asking is not to declare the table.
            (
                "[agent]\nscenarios='s.jsonl'\n[agent.judge]\ntype='command'\ncommand=[]\n",
                "declares an empty command",
            ),
        ] {
            let source = format!(
                "[run]\nalgorithm='agent_grpo'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{section}"
            );
            let file = write_config(&source);
            let error = load(&file).unwrap_err().to_string();
            remove_config(&file);
            assert!(error.contains(expected), "expected {expected}, got {error}");
        }
    }

    /// The point of the change: a verifiable task declares an environment and
    /// no judge, and that is a complete document.
    #[test]
    fn an_environment_graded_agentic_run_needs_no_judge() {
        let source = concat!(
            "[run]\nalgorithm='agent_grpo'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
            "[training]\nctx=2048\nmicro_batch=64\n",
            "[agent]\nscenarios='s.jsonl'\ngroup_size=4\n",
            "[agent.environment]\ntype='http'\nbase_url='http://127.0.0.1:8099'\n",
        );
        let file = write_config(source);
        let config = load(&file).expect("an environment grades this run on its own");
        remove_config(&file);
        let Algorithm::AgentGrpo(agent) = &config.algorithm else {
            panic!("expected an agentic algorithm");
        };
        assert!(agent.judge.is_none(), "no [agent.judge] was declared");
    }

    #[test]
    fn grpo_optional_features_parse_and_validate() {
        let base = concat!(
            "[run]\nalgorithm='grpo'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
        );
        let grpo = concat!(
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=4\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=2\n",
            "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.02\n",
            "baseline='rloo'\nprompt_order='shuffled'\n",
            "overlong_penalty={buffer_tokens=4,max_penalty=1.0}\n",
            "log_completions={every=2,path='completions.jsonl'}\n",
            "kl_schedule={warmup_updates=2,target=0.05}\n",
            "dynamic_sampling={max_resample_factor=3}\n",
            "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=16\nseed=1\n",
        );
        let file = write_config(&format!("{base}{grpo}"));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.n_seq_max, 4);
        assert_eq!(loaded.training.generation_concurrency, 8);
        let Algorithm::Grpo(config) = loaded.algorithm else {
            panic!("expected a grpo config")
        };
        assert_eq!(config.baseline, AdvantageBaseline::LeaveOneOut);
        assert_eq!(config.prompt_order, PromptOrder::Shuffled);
        let penalty = config.overlong_penalty.unwrap();
        assert_eq!(penalty.buffer_tokens, 4);
        assert_eq!(penalty.max_penalty, 1.0);
        let log = config.log_completions.unwrap();
        assert_eq!(log.every, 2);
        assert_eq!(log.path, file.parent().unwrap().join("completions.jsonl"));
        let schedule = config.kl_schedule.unwrap();
        assert_eq!(schedule.warmup_updates, 2);
        assert_eq!(schedule.target, Some(0.05));
        assert_eq!(config.dynamic_sampling.unwrap().max_resample_factor, 3);
        remove_config(&file);
    }

    /// `[grpo.judge]` is `[agent.judge]`, in the section of the other loop: the
    /// same table, the same spelling, and the cache path rebased against the
    /// document like every other relative path it declares.
    #[test]
    fn grpo_judge_parses_next_to_the_reward_command() {
        let base = concat!(
            "[run]\nalgorithm='grpo'\n",
            "[model]\npath='model.gguf'\n",
            "[lora]\noutput='out.gguf'\n",
        );
        let grpo = concat!(
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
            "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
            "judge_weight=0.3\njudge_failure='fail'\nmax_judge_dropped_fraction=0.25\n",
            "[grpo.judge]\ntype='ruler'\nbase_url='http://x/v1'\nmodel='lite'\n",
            "cache_path='judge.cache'\n",
            "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
        );
        let file = write_config(&format!("{base}{grpo}"));
        let loaded = load(&file).unwrap();
        let Algorithm::Grpo(config) = loaded.algorithm else {
            panic!("expected a grpo config")
        };
        let judge = config.judge.expect("a judge was declared");
        assert_eq!(judge.weight, 0.3);
        assert_eq!(judge.max_dropped_fraction, 0.25);
        assert!(matches!(
            judge.failure,
            retrograd_agent_core::config::JudgeFailurePolicy::Fail
        ));
        let retrograd_spec::judge::JudgeConfig::Ruler { config } = judge.config else {
            panic!("expected a RULER judge")
        };
        assert_eq!(
            config.cache_path.as_deref(),
            Some(file.parent().unwrap().join("judge.cache").as_path())
        );
        remove_config(&file);

        // The weight is what the verdict is worth, and there is no honest
        // default for it; and none of the three keys means anything without the
        // section they govern.
        for (source, expected) in [
            (
                concat!(
                    "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
                    "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
                    "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
                    "[grpo.judge]\ntype='ruler'\nbase_url='http://x/v1'\nmodel='lite'\n",
                    "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
                ),
                "grpo.judge_weight is required",
            ),
            (
                concat!(
                    "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
                    "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
                    "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
                    "judge_weight=0.3\n",
                    "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
                ),
                "need a [grpo.judge] section",
            ),
        ] {
            let file = write_config(&format!("{base}{source}"));
            let error = load(&file).unwrap_err().to_string();
            remove_config(&file);
            assert!(error.contains(expected), "expected {expected}, got {error}");
        }
    }

    #[test]
    fn grpo_shared_prefix_fanout_parses_and_respects_group_size() {
        let grpo = concat!(
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=1\nprompts_per_update=1\ngroup_size=4\ngrpo_epochs=1\n",
            "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
            "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
        );
        let source = |fanout: &str| {
            format!(
                "[run]\nalgorithm='grpo'\n[model]\npath='model.gguf'\n\
             [lora]\noutput='out.gguf'\n[training]\nctx=256\nmicro_batch=32\n\
             shared_prefix_fanout={fanout}\n{grpo}"
            )
        };

        for (value, expected) in [
            ("'auto'", SharedPrefixFanout::Auto),
            ("'off'", SharedPrefixFanout::Off),
            ("'max'", SharedPrefixFanout::Max),
            ("3", SharedPrefixFanout::Exact(3)),
        ] {
            let file = write_config(&source(value));
            let loaded = load(&file).unwrap();
            assert_eq!(loaded.training.shared_prefix_fanout, expected);
            remove_config(&file);
        }

        for (value, expected) in [
            ("1", "at least 2"),
            ("5", "exceeds grpo.group_size"),
            ("'fast'", "must be 'auto', 'off', 'max'"),
        ] {
            let file = write_config(&source(value));
            let error = load(&file).unwrap_err().to_string();
            assert!(error.contains(expected), "expected {expected}, got {error}");
            remove_config(&file);
        }
    }

    #[test]
    fn grpo_generation_concurrency_is_independent_and_validated() {
        let grpo = concat!(
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=1\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=1\n",
            "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
            "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
        );
        let source = |concurrency: u32, batch: u32| {
            format!(
                "[run]\nalgorithm='grpo'\n[model]\npath='model.gguf'\n\
                 [lora]\noutput='out.gguf'\n[training]\nctx={batch}\n\
                 micro_batch=4\ngeneration_concurrency={concurrency}\n{grpo}"
            )
        };

        let file = write_config(&source(1, 256));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.n_seq_max, 4);
        assert_eq!(loaded.training.generation_concurrency, 1);
        remove_config(&file);

        for (concurrency, batch, expected) in [
            (0, 256, "must be greater than zero"),
            (9, 256, "must not exceed the 8 grpo rollouts"),
            (8, 4, "must not exceed the optimizer window"),
            (257, 512, "must not exceed 256"),
        ] {
            let file = write_config(&source(concurrency, batch));
            let error = load(&file).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            remove_config(&file);
        }
    }

    /// An optimizer window below `ctx` turns one rollout into several steps,
    /// most of them over prompt positions carrying no label, and the policy
    /// leaves its trust region inside the first update. So a rollout algorithm
    /// pins `gradient_accumulation` to `ctx / micro_batch`. SFT is unaffected:
    /// there a row is a document, and stepping through it several times is the
    /// point.
    #[test]
    fn a_rollout_algorithm_pins_the_optimizer_step_to_the_trained_window() {
        let grpo = concat!(
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=1\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=1\n",
            "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
            "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
        );
        let ppo = concat!(
            "[ppo]\nprompts='p.jsonl'\nreward_command=['r']\n",
            "updates=1\nrollout_batch_size=2\nppo_epochs=1\n",
            "clip_range=0.2\nkl_coefficient=0.0\n",
            "[ppo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
        );
        let source = |algorithm: &str, section: &str, accumulation: &str| {
            format!(
                "[run]\nalgorithm='{algorithm}'\n[model]\npath='model.gguf'\n\
                 [lora]\noutput='out.gguf'\n[training]\nctx=256\n\
                 micro_batch=4\n{accumulation}{section}"
            )
        };

        for (algorithm, section) in [("grpo", grpo), ("ppo", ppo)] {
            let file = write_config(&source(algorithm, section, "gradient_accumulation=16\n"));
            let error = load(&file).unwrap_err().to_string();
            assert!(error.contains("one optimizer step per rollout"), "{error}");
            assert!(
                error.contains("pinned to ctx / micro_batch = 64"),
                "{error}"
            );
            remove_config(&file);

            // Omitted, it resolves to the only admissible value.
            let file = write_config(&source(algorithm, section, ""));
            let loaded = load(&file).unwrap_or_else(|error| panic!("{algorithm}: {error}"));
            assert_eq!(loaded.training.n_batch, 256);
            assert_eq!(loaded.training.gradient_accumulation(), 64);
            remove_config(&file);
        }

        // SFT keeps its accumulation window as a free parameter.
        let file = write_config(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n\
             [training]\nctx=256\nmicro_batch=4\ngradient_accumulation=16\n\
             [sft]\ndata='data.txt'\n",
        );
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.n_batch, 64);
        remove_config(&file);
    }

    #[test]
    fn grpo_optional_feature_boundaries_are_rejected() {
        let mut config = valid_grpo();
        config.overlong_penalty = Some(OverlongPenalty {
            buffer_tokens: 8, // == max_new_tokens
            max_penalty: 1.0,
        });
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("buffer_tokens")
        );

        let mut config = valid_grpo();
        config.overlong_penalty = Some(OverlongPenalty {
            buffer_tokens: 2,
            max_penalty: 0.0,
        });
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_penalty")
        );

        // A KL schedule with a zero base coefficient is contradictory.
        let mut config = valid_grpo();
        config.kl_coefficient = 0.0;
        config.kl_schedule = Some(KlSchedule {
            warmup_updates: 1,
            target: None,
        });
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("kl_schedule requires")
        );

        let mut config = valid_grpo();
        config.log_completions = Some(CompletionLog {
            every: 0,
            path: "c.jsonl".into(),
        });
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("log_completions.every")
        );

        // A resample factor of 1 permits no resampling and is rejected.
        let mut config = valid_grpo();
        config.dynamic_sampling = Some(DynamicSampling {
            max_resample_factor: 1,
        });
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_resample_factor")
        );
    }

    #[test]
    fn grpo_validation_enforces_group_geometry_and_on_policy_sampling() {
        valid_grpo().validate().unwrap();

        let mut config = valid_grpo();
        config.group_size = 1;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("at least 2")
        );

        let mut config = valid_grpo();
        config.clip_range_high = 0.1;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("Clip-Higher")
        );

        let mut config = valid_grpo();
        config.sampling.temperature = 0.9;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("on-policy")
        );

        let mut config = valid_grpo();
        config.sampling.top_p = 0.9;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("on-policy")
        );

        let mut config = valid_grpo();
        config.reward_command.clear();
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("executable")
        );

        let mut config = valid_grpo();
        config.updates = 0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("greater than zero")
        );

        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            let mut config = valid_grpo();
            config.kl_coefficient = invalid;
            assert!(config.validate().is_err(), "kl={invalid}");
        }
    }
}
