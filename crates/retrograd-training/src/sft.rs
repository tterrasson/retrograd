//! Supervised fine-tuning: one teacher-forced pass per epoch over a prepared
//! dataset, through the runtime's own epoch loop.

use std::path::Path;

use retrograd_config::SftConfig;
use retrograd_core::{Error, Result};
use retrograd_dataset::{self as dataset, DataFormat, PreparedDataset};
use retrograd_engine::Trainer;

use super::Progress;

pub fn prepare(trainer: &Trainer, config: &SftConfig) -> Result<PreparedDataset> {
    let n_ctx = trainer.context_size()?;
    let train = dataset::prepare(trainer, &config.data, config.data_format, n_ctx)?;
    validate_train(&train)?;
    Ok(train)
}

pub fn prepare_eval(trainer: &Trainer, path: &Path, format: DataFormat) -> Result<PreparedDataset> {
    let data = dataset::prepare(trainer, path, format, trainer.context_size()?)?;
    validate_eval(&data)?;
    Ok(data)
}

/// Runs SFT, restarting at the epoch boundary `resume` restores, if any. The
/// epoch is the resume unit for SFT: within an epoch the runtime owns the row
/// cursor, so restarting mid-epoch would replay or skip rows. The
/// learning-rate horizon stays the full configured run.
pub fn run_resumed(
    trainer: &mut Trainer,
    train: &PreparedDataset,
    resume: Option<super::Boundary>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<retrograd_core::TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::sft", "training");
    let _entered = span.enter();
    if let Some(boundary) = resume {
        let completed = u32::try_from(boundary.completed_iterations).map_err(|_| {
            Error::overflow("the checkpoint epoch count does not fit in the epoch counter")
        })?;
        trainer.set_resume_point(completed)?;
    }
    trainer.train_sft_controlled(train, None, |trainer, metrics| {
        on_progress(trainer, Progress::sft(metrics, false))
    })
}

fn validate_train(train: &PreparedDataset) -> Result<()> {
    if train.is_empty() || train.supervised_tokens == 0 {
        return Err(Error::invalid(
            "SFT needs at least one supervised training token",
        ));
    }
    Ok(())
}

fn validate_eval(eval: &PreparedDataset) -> Result<()> {
    if eval.is_empty() || eval.supervised_tokens == 0 {
        return Err(Error::invalid(
            "SFT evaluation data needs supervised tokens",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(n_ctx: usize, examples: usize, supervised_tokens: usize) -> PreparedDataset {
        PreparedDataset {
            n_ctx,
            tokens: vec![1; n_ctx * examples],
            labels: vec![2; n_ctx * examples],
            examples,
            supervised_tokens,
        }
    }

    #[test]
    fn validation_requires_supervision_in_train_and_eval() {
        let train = dataset(4, 1, 1);
        validate_train(&train).unwrap();
        validate_eval(&dataset(4, 1, 1)).unwrap();

        for invalid in [dataset(4, 0, 0), dataset(4, 1, 0)] {
            assert!(validate_train(&invalid).is_err());
        }
        for invalid_eval in [dataset(4, 0, 0), dataset(4, 1, 0)] {
            assert!(validate_eval(&invalid_eval).is_err());
        }
    }
}
