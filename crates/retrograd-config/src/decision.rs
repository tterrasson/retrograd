//! The `[decision]` section: fine-tuning a decision model on typed questions.
//!
//! A decision model reads a state and a typed question (`choice`, `score`,
//! `noul`) and scores the options in one forward pass. The run trains it on
//! labelled questions through the grouped softmax cross-entropy, evaluates
//! it on `[evaluation].data`, fits its temperatures there when it finishes,
//! and writes them into what it exports.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result, TrainConfig};
use retrograd_decision::dataset::DataFormat;

use crate::common::{parse_string_enum, require_nonzero, resolve};

/// Default `decision.questions_per_batch`.
pub const DEFAULT_QUESTIONS_PER_BATCH: u32 = 8;

/// `[decision]`, resolved.
#[derive(Clone, Debug)]
pub struct DecisionConfig {
    /// One question per line, aligned on the service's request.
    pub data: PathBuf,
    pub data_format: DataFormat,
    /// A `systemone` template to render the questions with in place of the
    /// model's. Exported with the result.
    pub template: Option<PathBuf>,
    /// The head budget in place of the model's `max_head_tokens`. Exported
    /// with the result.
    pub max_head_tokens: Option<u32>,
    /// Permute the options of every `choice` question at every epoch, so the
    /// model cannot learn the position of the right answer.
    pub option_shuffle: bool,
    /// Permute the questions at every epoch.
    pub shuffle: bool,
    /// The seed of both permutations: `lora.seed`, the run's one seed.
    pub seed: u32,
    /// Fit the temperatures on `[evaluation].data` once trained, and export
    /// them.
    pub calibrate: bool,
    /// Most questions one micro-batch holds, each its own sequence. The
    /// micro-batch also holds at most `training.micro_batch` tokens.
    pub questions_per_batch: u32,
    /// Micro-batches per optimizer update.
    pub gradient_accumulation: u32,
    /// Where to write a copy of the base model carrying the fitted
    /// temperatures and the effective template and budget, tensors unchanged:
    /// what `llama-server --lora` needs to serve the adapter with its own
    /// temperatures.
    pub export_base: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct DecisionToml {
    #[cfg_attr(feature = "openapi", schema(value_type = String, format = "path"))]
    pub data: PathBuf,
    /// `native` (default): the service's question, one per line, with
    /// `answer` or `probabilities`. `julia`: `question`, `options`, `type`,
    /// `target` and optional `teacher_logits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>, format = "path"))]
    pub template: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_head_tokens: Option<u32>,
    /// Defaults to `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_shuffle: Option<bool>,
    /// Defaults to `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
    /// Defaults to whether `[evaluation]` is present, which `true` requires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub questions_per_batch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>, format = "path"))]
    pub export_base: Option<PathBuf>,
}

pub(crate) fn parse_decision_format(value: &str) -> Result<DataFormat> {
    parse_string_enum!(
        value,
        "decision.data_format must be native or julia",
        "native" => DataFormat::Native,
        "julia" => DataFormat::Julia,
    )
}

/// Builds `[decision]`. The optimizer window it implies is pinned onto
/// `training` here: one micro-batch is one runtime batch (`n_batch =
/// micro_batch`), and the accumulation of micro-batches into an update is the
/// grouped step's own period, so `gradient_accumulation` never multiplies a
/// context the encoder does not have.
pub(crate) fn build_decision(
    value: DecisionToml,
    root: &Path,
    training: &mut TrainConfig,
    gradient_accumulation: Option<u32>,
    seed: u32,
    has_evaluation: bool,
) -> Result<DecisionConfig> {
    let data_format = value
        .data_format
        .as_deref()
        .map(parse_decision_format)
        .transpose()?
        .unwrap_or(DataFormat::Native);
    if let Some(max_head_tokens) = value.max_head_tokens {
        require_nonzero(
            max_head_tokens,
            "decision.max_head_tokens must be greater than zero when set",
        )?;
    }
    let questions_per_batch = value
        .questions_per_batch
        .unwrap_or(DEFAULT_QUESTIONS_PER_BATCH);
    require_nonzero(
        questions_per_batch,
        "decision.questions_per_batch must be greater than zero",
    )?;
    let calibrate = value.calibrate.unwrap_or(has_evaluation);
    if calibrate && !has_evaluation {
        return Err(Error::config(
            "decision.calibrate fits the temperatures on held-out questions and needs \
             [evaluation].data: add the section, or set decision.calibrate = false",
        ));
    }
    let gradient_accumulation = gradient_accumulation.unwrap_or(1);
    training.n_batch = training.n_ubatch;
    training.n_seq_max = questions_per_batch;
    // Nothing is generated: no second, forward-only context.
    training.fast_generation_context = false;
    if !training.n_ctx.is_multiple_of(training.n_ubatch) {
        return Err(Error::config(format!(
            "training.ctx ({}) must be a multiple of training.micro_batch ({}): a decision \
             step is one micro-batch of questions",
            training.n_ctx, training.n_ubatch
        )));
    }
    Ok(DecisionConfig {
        data: resolve(root, value.data),
        data_format,
        template: value.template.map(|path| resolve(root, path)),
        max_head_tokens: value.max_head_tokens,
        option_shuffle: value.option_shuffle.unwrap_or(true),
        shuffle: value.shuffle.unwrap_or(true),
        seed,
        calibrate,
        questions_per_batch,
        gradient_accumulation,
        export_base: value.export_base.map(|path| resolve(root, path)),
    })
}
