//! The `[distill]` section: on-policy and offline top-k distillation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result, SamplingParams};

use crate::PromptOrder;
use crate::common::{
    parse_string_enum, require_non_negative_f32, require_nonzero, require_positive_f32, required,
    resolve, sampling,
};
use crate::document::SamplingToml;

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
pub(crate) fn build_distill(value: DistillToml, root: &Path) -> Result<DistillConfig> {
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
