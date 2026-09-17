//! The `[sft]` section: supervised fine-tuning on a dataset.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use retrograd_core::{Result, TrainConfig};
use retrograd_dataset::DataFormat;

use crate::common::{parse_data_format, resolve};

#[derive(Clone, Debug)]
pub struct SftConfig {
    pub data: PathBuf,
    pub data_format: DataFormat,
    /// Permute the training rows at the start of every epoch, seeded from
    /// `lora.seed`. On by default; see [`TrainConfig::shuffle_dataset`], which
    /// is where the runtime reads it from.
    pub shuffle: bool,
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

/// Builds the `[sft]` section. `shuffle` reaches the runtime through
/// [`TrainConfig`] rather than through [`SftConfig`]: the runtime owns the row
/// cursor inside an epoch. The seed is the LoRA one - one seed describes the
/// whole run, and the permutation is a function of it and the epoch index.
pub(crate) fn build_sft(
    value: SftToml,
    root: &Path,
    training: &mut TrainConfig,
    seed: u32,
) -> Result<SftConfig> {
    let data = resolve(root, value.data);
    let data_format = match value
        .data_format
        .as_deref()
        .map(parse_data_format)
        .transpose()?
    {
        Some(format) => format,
        // Resolved against `root` first: an unknown extension makes `infer`
        // open the file to sniff its content, and a relative path is only
        // valid once joined with `root`.
        None => DataFormat::infer(&data)?,
    };
    let shuffle = value.shuffle.unwrap_or(true);
    training.shuffle_dataset = shuffle;
    training.shuffle_seed = u64::from(seed);
    Ok(SftConfig {
        data,
        data_format,
        shuffle,
    })
}
