//! Validators and small parsers shared by every section.

use std::path::{Path, PathBuf};

use retrograd_core::{
    DEFAULT_REWARD_TIMEOUT_SECONDS, Error, LrScheduler, Result, RewardMode, RewardProtocol,
    SamplingParams,
};
use retrograd_dataset::DataFormat;

use crate::document::SamplingToml;

macro_rules! parse_string_enum {
    ($value:expr_2021, $error:expr_2021, $($pattern:literal => $variant:expr_2021),+ $(,)?) => {{
        let normalized = $value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            $($pattern => Ok($variant),)+
            _ => Err(Error::config($error)),
        }
    }};
}
pub(crate) fn require_nonzero<T>(value: T, message: &str) -> Result<()>
where
    T: Default + PartialEq,
{
    if value == T::default() {
        return Err(Error::config(message));
    }
    Ok(())
}

pub(crate) fn require_positive_f32(value: f32, name: &str) -> Result<()> {
    if !(value > 0.0 && value.is_finite()) {
        return Err(Error::config(format!(
            "{name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

pub(crate) fn require_non_negative_f32(value: f32, name: &str) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(Error::config(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

pub(crate) fn require_non_negative_f64(value: f64, name: &str) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(Error::config(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}
pub(crate) fn resolve(root: &Path, value: PathBuf) -> PathBuf {
    if value.is_absolute() {
        value
    } else {
        root.join(value)
    }
}
pub(crate) fn required<T>(value: Option<T>, message: &'static str) -> Result<T> {
    value.ok_or_else(|| Error::config(message))
}
/// The reward transport of a `[ppo]` or `[grpo]` section, defaults included.
/// Written once for both because it is one contract: the same two keys, the
/// same refusal, and a reward process that cannot tell which section called it.
pub(crate) fn reward_protocol(
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

pub(crate) fn validate_command(command: &[String]) -> Result<()> {
    if command.is_empty() || command[0].trim().is_empty() {
        Err(Error::config("reward_command must contain an executable"))
    } else {
        Ok(())
    }
}
pub(crate) fn sampling(value: SamplingToml) -> Result<SamplingParams> {
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
pub(crate) fn parse_scheduler(value: &str) -> Result<LrScheduler> {
    parse_string_enum!(
        value,
        "training.lr_scheduler must be constant, linear, or cosine",
        "constant" => LrScheduler::Constant,
        "linear" => LrScheduler::Linear,
        "cosine" => LrScheduler::Cosine,
    )
}
pub(crate) fn parse_data_format(value: &str) -> Result<DataFormat> {
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

pub(crate) use parse_string_enum;
