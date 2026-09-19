//! Strict TOML configuration used by the training CLI.
//!
//! One document, one schema, whichever frontend reads it. `[model]`, `[lora]`,
//! `[training]`, `[metrics]`, `[evaluation]` and `[checkpoint]` are shared;
//! `[run].algorithm` selects which of `[sft]`, `[ppo]`, `[grpo]` or `[agent]`
//! is the run's own section, and exactly that one may be present.

//!
//! One module per section of the document, each holding that section's schema,
//! the engine-shaped type it builds into, and the function between them -
//! `[agent]` was already written that way. What stays here is what a run has
//! whichever algorithm it runs; `build` owns the document-wide assembly and the
//! rules that cross two sections.

mod agent;
mod build;
mod common;
mod distill;
mod document;
mod grpo;
mod ppo;
mod sft;

#[cfg(test)]
mod round_trip;
#[cfg(test)]
mod tests;

use std::path::PathBuf;

use retrograd_core::{Error, LoraConfig, Result, TargetSet, TrainConfig};

pub use agent::{AgentRunConfig, AgentToml, ScenarioGenerationConfig};
pub use build::{build, build_with, load, load_with, parse_toml};
pub use distill::{
    DEFAULT_DISTILL_CLIP_RANGE_HIGH, DEFAULT_DISTILL_CLIP_RANGE_LOW, DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_WEIGHT_CLIP, DistillConfig, DistillMode, DistillToml, OfflineDistillConfig,
};
pub use document::{
    CheckpointToml, ConfigDocument, EvaluationToml, LoraToml, MetricsToml, ModelOverride,
    ModelToml, ObserveToml, OutputToml, RunToml, SamplingToml, SharedPrefixFanoutToml,
    TrainableToml, TrainingToml,
};
pub use grpo::{
    AdvantageBaseline, DEFAULT_MAX_STALLED_UPDATES, DynamicSampling, GrpoConfig, GrpoJudge,
    GrpoToml, KlSchedule, KlScheduleToml, OverlongPenalty,
};
pub use ppo::{CriticConfig, CriticToml, PpoConfig, PpoToml};
pub use sft::{SftConfig, SftToml};
/// The engine-shaped configuration a run is built from. Produced from a
/// [`ConfigDocument`] by [`build`]/[`build_with`], which is the only path a
/// document may reach it by - see [`ConfigDocument`] for why.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub algorithm: Algorithm,
    pub model: PathBuf,
    /// The adapter this run trains, when it trains one. `None` for `full` and
    /// `partial`, which have no adapter at all - not an adapter of rank zero.
    pub lora: Option<LoraRunConfig>,
    /// What the run writes when it finishes, and what kind of thing that is.
    pub output: OutputConfig,
    pub training: TrainConfig,
    pub metrics: MetricsConfig,
    pub evaluation: Option<EvaluationConfig>,
    pub checkpoint: Option<CheckpointConfig>,
    pub observe: Option<ObserveConfig>,
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
    /// Existing adapter GGUF to resume training from, instead of creating a
    /// fresh adapter. Rank, alpha, and targets then come from the file.
    pub init_adapter: Option<PathBuf>,
}

/// What a finished run writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputConfig {
    pub path: PathBuf,
    pub kind: OutputKind,
}

/// Which kind of result an output path holds.
///
/// Not interchangeable, and not inferable from the extension: three different
/// things are GGUFs here, and only one of them loads with
/// `llama_adapter_lora_init`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    /// A portable LoRA adapter. The only kind a `lora` run produces.
    Adapter,
    /// A Retrograd bundle: the trained base tensors by absolute value, plus the
    /// adapter when the run has one. Requires the matching base model and this
    /// loader; it is not a standalone model.
    Trainable,
    /// A standalone model GGUF. Not implemented: the saver has a
    /// per-architecture support predicate, and merging an adapter into
    /// supported weights has no parity coverage, so accepting the name would
    /// promise a file the run cannot write.
    Model,
}

impl OutputKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Adapter => "adapter",
            Self::Trainable => "trainable",
            Self::Model => "model",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adapter" => Ok(Self::Adapter),
            "trainable" => Ok(Self::Trainable),
            "model" => Ok(Self::Model),
            other => Err(Error::config(format!(
                "output.kind must be adapter, trainable or model; got '{other}'"
            ))),
        }
    }
}

impl std::fmt::Display for OutputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct MetricsConfig {
    pub tensorboard_dir: Option<PathBuf>,
    pub wandb_export_dir: Option<PathBuf>,
}

/// `[observe]`: live export of the rollouts of PPO, GRPO and agentic GRPO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObserveConfig {
    pub directory: PathBuf,
    /// Exports the texts of one update in `every`, counted from one.
    pub every: u32,
    /// Cut applied to every exported text; `0` keeps them whole.
    pub max_text_chars: usize,
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
/// Order in which training prompts are drawn across updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptOrder {
    /// Strict round-robin over the prompt file (bit-exact reference behavior).
    Sequential,
    /// A seeded permutation re-drawn each full pass over the dataset, so
    /// correlated adjacent prompts do not always land in the same update.
    Shuffled,
}
/// Seed a run gets when no `[lora].seed` names one: the dataset permutation
/// and the adapter initialization share it, so a run that trains base tensors
/// and has no `[lora]` section still shuffles reproducibly.
pub const DEFAULT_SEED: u32 = 42;
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
