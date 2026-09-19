//! Resolution of `--resume`: which checkpoint a stopped run restarts from.
//!
//! Shared with the server because `fork_from` is the same operation, including
//! the "latest checkpoint" resolution.

use std::fs;
use std::path::{Path, PathBuf};

use retrograd_config::RunConfig;
use retrograd_core::{Error, Result};

/// Sets `checkpoint.resume_from`, defaulting to the most recent step checkpoint
/// under `checkpoint.directory` when no path is given.
pub fn apply_resume_override(config: &mut RunConfig, explicit: Option<PathBuf>) -> Result<()> {
    if config
        .lora
        .as_ref()
        .is_some_and(|lora| lora.init_adapter.is_some())
    {
        return Err(Error::invalid(
            "--resume and lora.init_adapter are mutually exclusive",
        ));
    }
    let checkpoint = config.checkpoint.as_mut().ok_or_else(|| {
        Error::invalid("--resume requires a [checkpoint] section with a directory")
    })?;
    if checkpoint.resume_from.is_some() {
        return Err(Error::invalid(
            "--resume conflicts with checkpoint.resume_from already set in the TOML; use only one",
        ));
    }
    checkpoint.resume_from = Some(match explicit {
        Some(path) => path,
        None => latest_checkpoint(&checkpoint.directory)?,
    });
    Ok(())
}

/// Picks the highest-numbered `step-*.state` directory, falling back to
/// `best.state` when no step checkpoint exists.
pub fn latest_checkpoint(directory: &Path) -> Result<PathBuf> {
    let entries = fs::read_dir(directory).map_err(|error| {
        Error::checkpoint(format!(
            "--resume could not read checkpoint directory {}: {error}",
            directory.display()
        ))
    })?;
    let mut best_step: Option<(u64, PathBuf)> = None;
    let mut best_eval: Option<PathBuf> = None;
    for entry in entries {
        let entry = entry.map_err(|error| Error::checkpoint(format!("--resume: {error}")))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".state") else {
            continue;
        };
        if stem == "best" {
            best_eval = Some(path);
            continue;
        }
        if let Some(step) = stem
            .strip_prefix("step-")
            .and_then(|digits| digits.parse::<u64>().ok())
            && best_step
                .as_ref()
                .is_none_or(|(current, _)| step > *current)
        {
            best_step = Some((step, path));
        }
    }
    best_step
        .map(|(_, path)| path)
        .or(best_eval)
        .ok_or_else(|| {
            Error::invalid(format!(
                "--resume found no checkpoints in {}",
                directory.display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{temp_path, write_sft_config};
    use retrograd_config::{self as config, CheckpointConfig};

    #[test]
    fn latest_checkpoint_chooses_highest_step_and_validates_empty_directories() {
        let root = temp_path("checkpoints");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir(root.join("step-000000000007.state")).unwrap();
        fs::create_dir(root.join("step-000000000042.state")).unwrap();
        fs::create_dir(root.join("step-invalid.state")).unwrap();
        fs::create_dir(root.join("best.state")).unwrap();
        fs::write(root.join("step-999999999999.state"), b"not a directory").unwrap();
        assert_eq!(
            latest_checkpoint(&root).unwrap(),
            root.join("step-000000000042.state")
        );

        let empty = temp_path("empty-checkpoints");
        fs::create_dir_all(&empty).unwrap();
        let error = latest_checkpoint(&empty).unwrap_err();
        assert!(error.to_string().contains("found no checkpoints"));
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(empty).unwrap();

        let eval_only = temp_path("eval-only-checkpoints");
        fs::create_dir_all(&eval_only).unwrap();
        fs::create_dir(eval_only.join("best.state")).unwrap();
        assert_eq!(
            latest_checkpoint(&eval_only).unwrap(),
            eval_only.join("best.state")
        );
        fs::remove_dir_all(eval_only).unwrap();
    }

    #[test]
    fn resume_override_uses_the_latest_checkpoint_and_rejects_conflicts() {
        let root = temp_path("resume");
        let config_path = write_sft_config(&root);
        let checkpoint_dir = root.join("checkpoints");
        fs::create_dir_all(checkpoint_dir.join("step-000000000003.state")).unwrap();

        let mut config = config::load(&config_path).unwrap();
        config.checkpoint = Some(CheckpointConfig {
            directory: checkpoint_dir.clone(),
            mode: config::CheckpointMode::Steps,
            every_steps: Some(1),
            resume_from: None,
        });
        apply_resume_override(&mut config, None).unwrap();
        assert_eq!(
            config.checkpoint.unwrap().resume_from,
            Some(checkpoint_dir.join("step-000000000003.state"))
        );

        let mut config = config::load(&config_path).unwrap();
        config.checkpoint = Some(CheckpointConfig {
            directory: root.join("other"),
            mode: config::CheckpointMode::Steps,
            every_steps: Some(1),
            resume_from: Some(root.join("already.state")),
        });
        let error =
            apply_resume_override(&mut config, Some(root.join("explicit.state"))).unwrap_err();
        assert!(error.to_string().contains("conflicts"));

        let mut config = config::load(&config_path).unwrap();
        config.lora.as_mut().expect("a lora fixture").init_adapter =
            Some(root.join("adapter.gguf"));
        let error = apply_resume_override(&mut config, None).unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"));

        let mut config = config::load(&config_path).unwrap();
        config.checkpoint = None;
        let error = apply_resume_override(&mut config, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires a [checkpoint] section")
        );

        let mut config = config::load(&config_path).unwrap();
        config.checkpoint = Some(CheckpointConfig {
            directory: root.join("missing-checkpoint-dir"),
            mode: config::CheckpointMode::Steps,
            every_steps: Some(1),
            resume_from: None,
        });
        let error = apply_resume_override(&mut config, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("could not read checkpoint directory")
        );

        fs::remove_dir_all(root).unwrap();
    }
}
