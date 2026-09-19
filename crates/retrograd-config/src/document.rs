//! The document schema: the exact shape of the TOML file, section by section.
//!
//! Every type here is a DTO - what serde reads and writes. The engine-shaped
//! types they build into live beside their `build_*` function, one module per
//! section.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use retrograd_core::{CheckpointDtype, Device, KvDtype, LoraDtype};

use crate::agent::AgentToml;
use crate::distill::DistillToml;
use crate::grpo::GrpoToml;
use crate::ppo::PpoToml;
use crate::sft::SftToml;

/// The configuration *document*: the exact schema of the CLI's TOML file.
///
/// Public because it is the one schema three frontends share: the CLI reads it
/// from TOML, the server accepts it as JSON, and the resolver renders its own
/// output back through it. The resolver builds one of these instead of a
/// [`RunConfig`](crate::RunConfig) directly, so that "the resolver cannot
/// produce a config the CLI would refuse" holds *by construction*: the only way
/// from a document to a `RunConfig` is [`build`](crate::build()), which is what
/// `load` calls.
///
/// [`RunConfig`](crate::RunConfig), by contrast, stays internal: it is the
/// engine's shape and changes at the engine's pace.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigDocument {
    pub run: RunToml,
    #[serde(default)]
    pub model: ModelToml,
    /// `[lora]`: required by `lora` and `hybrid`, refused by `full` and
    /// `partial`. Optional as a *document* section because a run that trains
    /// base tensors has no adapter at all, and demanding rank, alpha and
    /// targets from it would be demanding the shape of something it does not
    /// create.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraToml>,
    /// `[output]`: where the run's result goes, and which kind of result it is.
    /// The kind is defaulted from the policy when absent; the path is not, so
    /// a document that names no output is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputToml>,
    #[serde(default)]
    pub training: TrainingToml,
    /// `[trainable]`: which base tensors a `partial` or `hybrid` run selects.
    /// Absent for `lora`, and refused when present alongside it - a selector a
    /// policy ignores is a selector the user believes is in effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainable: Option<TrainableToml>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observe: Option<ObserveToml>,
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
/// What is not optional is the resolved
/// [`RunConfig::model`](crate::RunConfig::model): [`build_with`](crate::build_with)
/// refuses a document whose path is neither written nor overridden.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelToml {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}
/// `[output]`: the run's result.
///
/// `path` is where it is written; `kind` is what it is. The two are separate
/// because they answer different questions and only the second can be wrong:
/// a `model` export of a hybrid run would silently drop the adapter, and an
/// `adapter` export of a full run would be an empty file.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputToml {
    pub path: PathBuf,
    /// `adapter`, `trainable` or `model`. Defaults to `adapter` for a `lora`
    /// run and `trainable` for every policy that trains base tensors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraToml {
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
    /// Which family of parameters this run trains: `"lora"` (the default),
    /// `"full"`, `"partial"` or `"hybrid"`. See `retrograd_core::TrainablePolicy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainable: Option<String>,
    /// Which optimizer updates them: `"adamw"` (the default), `"sgd"`,
    /// `"muon"` or `"gefen"`. A name this build cannot honour is refused
    /// rather than accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<String>,
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

/// `[trainable]`: the base selection of a `partial` or `hybrid` run.
///
/// Every selector is off by default and an omitted `layers` means all blocks,
/// so an empty section selects nothing - which `build` refuses rather than
/// resolving to the empty set.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainableToml {
    /// `"all"`, `"last:<count>"` or an inclusive `"<first>..<last>"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<String>,
    /// Module aliases (`"attn"`, `"ffn"`), individual stems (`"attn_q"`) or
    /// explicit tensor patterns. Norms are not modules: they follow `norms`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modules: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norms: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub biases: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_head: Option<bool>,
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
pub struct ObserveToml {
    pub directory: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_text_chars: Option<usize>,
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
/// It is an *input* to [`build_with`](crate::build_with) rather than a patch
/// applied to the resulting [`RunConfig`](crate::RunConfig): a document that
/// names no `[model].path` must still build when the caller supplies one, and
/// that decision belongs to the same
/// function that would otherwise refuse it. A path here comes from the caller's
/// working directory, not the document's, so it is used as given.
#[derive(Clone, Debug, Default)]
pub struct ModelOverride {
    pub path: Option<PathBuf>,
    pub device: Option<Device>,
}
