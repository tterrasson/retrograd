//! Checkpoint policy, evaluation cadence and early stopping.
//!
//! Extracted verbatim from `src/train.rs` so the CLI and the HTTP server share
//! one definition of when a checkpoint is written and when a run stops early.
//! Two implementations would diverge, and the divergence would only show up as
//! a resume that replays or skips work.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use retrograd_checkpoint as checkpoint;
use retrograd_config::{Algorithm, CheckpointConfig, CheckpointMode, EvaluationConfig, RunConfig};
use retrograd_core::{CheckpointMetadata, Error, Result};
use retrograd_engine::Trainer;
use retrograd_training as training;

use crate::observer::EvalOutcome;
use crate::signature::{model_bytes, scheduler_name, trajectory_signature};

/// Which way an evaluation metric improves. A loss goes down, a reward goes up,
/// and `min_delta` has to be applied on the right side of the comparison.
#[derive(Clone, Copy)]
pub enum EvalDirection {
    Lower,
    Higher,
}

pub struct RunController {
    evaluation: Option<EvaluationConfig>,
    checkpoint: Option<CheckpointConfig>,
    next_checkpoint_step: Option<u64>,
    /// The step schedule fired, but the run has not reached a boundary it can
    /// be resumed from yet. A checkpoint is written at the next one.
    checkpoint_due: bool,
    /// Someone asked for a checkpoint out of schedule. Kept apart from
    /// `checkpoint_due` because it must not disturb the step schedule: an
    /// on-demand snapshot is not the periodic one arriving early.
    checkpoint_requested: bool,
    /// Last step the loop reported, so a schedule changed mid-run is rearmed
    /// relative to where the run actually is rather than to zero.
    last_global_step: u64,
    best_eval: Option<f64>,
    stale_evaluations: u32,
    early_stopped: bool,
    context: CheckpointContext,
}

/// Run-level facts a checkpoint needs and the runtime cannot supply. The
/// optimizer, scheduler, and RNG state come from the runtime instead.
#[derive(Clone, Debug)]
struct CheckpointContext {
    algorithm: String,
    trajectory_signature: String,
    /// Coarsest unit a resume may restart at, for the manifest.
    resume_boundary: String,
    scheduler_kind: String,
    learning_rate: f32,
    /// The optimizer the run configures, as a checkpoint spells it. Compared
    /// with what the checkpoint recorded: AdamW moments mean nothing to an SGD
    /// step, and neither resumes onto the other's trajectory.
    optimizer_kind: String,
    weight_decay: f32,
    max_grad_norm: f32,
    warmup_steps: u64,
    model_path: PathBuf,
    /// Filled once the algorithm has prepared its data.
    dataset: checkpoint::Dataset,
    seeds: BTreeMap<String, u64>,
}

impl RunController {
    pub fn new(config: &RunConfig) -> Result<Self> {
        let next_checkpoint_step = config
            .checkpoint
            .as_ref()
            .filter(|checkpoint| checkpoint.mode.includes_steps())
            .and_then(|checkpoint| checkpoint.every_steps);
        let (algorithm, seeds) = match &config.algorithm {
            Algorithm::Sft(_) => ("sft", BTreeMap::new()),
            Algorithm::Ppo(ppo) => (
                "ppo",
                BTreeMap::from([("sampling".to_string(), ppo.sampling.seed as u64)]),
            ),
            Algorithm::Grpo(grpo) => (
                "grpo",
                BTreeMap::from([("sampling".to_string(), grpo.sampling.seed as u64)]),
            ),
            Algorithm::Distill(distill) => (
                "distill",
                BTreeMap::from([("sampling".to_string(), distill.sampling.seed as u64)]),
            ),
            Algorithm::AgentGrpo(agent) => (
                "agent_grpo",
                BTreeMap::from([("rollout".to_string(), agent.config.seed)]),
            ),
        };
        Ok(Self {
            evaluation: config.evaluation.clone(),
            checkpoint: config.checkpoint.clone(),
            next_checkpoint_step,
            checkpoint_due: false,
            checkpoint_requested: false,
            last_global_step: 0,
            best_eval: None,
            stale_evaluations: 0,
            early_stopped: false,
            context: CheckpointContext {
                algorithm: algorithm.into(),
                trajectory_signature: trajectory_signature(config)?,
                resume_boundary: if algorithm == "sft" {
                    "epoch"
                } else {
                    "update"
                }
                .into(),
                scheduler_kind: scheduler_name(config.training.lr_scheduler).into(),
                learning_rate: config.training.learning_rate,
                optimizer_kind: config.training.trainable.optimizer.to_string(),
                weight_decay: config.training.weight_decay,
                max_grad_norm: config.training.max_grad_norm,
                warmup_steps: config.training.warmup_steps,
                model_path: config.model.clone(),
                dataset: checkpoint::Dataset::default(),
                seeds: BTreeMap::from_iter(seeds),
            },
        })
    }

    /// Whether the run stopped because patience ran out, rather than because it
    /// reached its last iteration.
    pub fn early_stopped(&self) -> bool {
        self.early_stopped
    }

    pub fn should_evaluate(&self, iteration: u32, total_iterations: u32) -> bool {
        self.evaluation.as_ref().is_some_and(|evaluation| {
            iteration.is_multiple_of(evaluation.every_iterations) || iteration == total_iterations
        })
    }

    /// Registers the prepared dataset and, when `checkpoint.resume_from` is
    /// set, restores the checkpoint into the trainer. Returns the boundary the
    /// algorithm must restart at, or `None` for a fresh run.
    ///
    /// The restore happens here rather than at model load because the dataset
    /// fingerprint is part of what makes a resume valid, and it only exists
    /// once the algorithm has prepared its data.
    pub fn begin(
        &mut self,
        trainer: &mut Trainer,
        dataset: checkpoint::Dataset,
        total_steps: u64,
    ) -> Result<Option<training::Boundary>> {
        self.context.dataset = dataset;
        // Against the whole schedule, not one save: a run that can write its
        // first checkpoint and not its tenth has spent hours to find that out.
        if let Some(config) = &self.checkpoint {
            checkpoint::DiskBudget {
                footprint: trainer.checkpoint_footprint()?,
                retained: retained_checkpoints(config, total_steps),
            }
            .check(&config.directory, "this run cannot keep its checkpoints")?;
        }
        let Some(resume_from) = self
            .checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.resume_from.clone())
        else {
            return Ok(None);
        };
        let optimizer_hyperparameters = trainer.optimizer_hyperparameters()?;
        let expected = checkpoint::Compatibility {
            model_signature: trainer.model_signature()?,
            model_bytes: model_bytes(&self.context.model_path),
            model_fingerprint: checkpoint::fingerprint_file_cached(&self.context.model_path)?,
            // From the trainer, not the document: the resume is compared
            // against the anchor actually attached.
            reference_fingerprint: trainer.reference_fingerprint()?,
            algorithm: self.context.algorithm.clone(),
            trajectory_signature: self.context.trajectory_signature.clone(),
            dataset_fingerprint: self.context.dataset.fingerprint.clone(),
            scheduler_kind: self.context.scheduler_kind.clone(),
            learning_rate: self.context.learning_rate,
            warmup_steps: self.context.warmup_steps,
            total_steps: Some(total_steps),
            optimizer_kind: self.context.optimizer_kind.clone(),
            // Only the trainer knows the layout version; no document spells it.
            optimizer_layout_version: trainer.optimizer_layout_version()?,
            optimizer_hyperparameters: optimizer_hyperparameters.lines(),
            weight_decay: self.context.weight_decay,
            max_grad_norm: self.context.max_grad_norm,
            // Read from the trainer rather than from the document: the
            // signature is over the set the optimizer actually marked, which is
            // the only side a checkpoint can be compared against. For a LoRA
            // run both are the policy name and an empty string.
            trainable_policy: trainer.trainable_policy().as_str().to_string(),
            trainable_signature: trainer.trainable_signature()?,
        };
        let info = trainer.load_checkpoint(&resume_from, &expected)?;
        self.best_eval = info.progress.best_eval;
        self.stale_evaluations = info.progress.stale_evaluations;
        // Restart the step schedule from where the checkpoint left off, so the
        // next checkpoint is one interval away rather than immediate.
        if let Some(frequency) = self
            .checkpoint
            .as_ref()
            .filter(|checkpoint| checkpoint.mode.includes_steps())
            .and_then(|checkpoint| checkpoint.every_steps)
        {
            let step = info.global_step();
            self.next_checkpoint_step = Some((step / frequency + 1) * frequency);
        }
        Ok(Some(training::Boundary {
            completed_iterations: info.epoch(),
            cursor: info.progress.cursor,
            kl_multiplier: info.progress.kl_multiplier,
        }))
    }

    /// Arms the step schedule. A checkpoint is only *written* at the next safe
    /// boundary: a mid-epoch snapshot could not be resumed without replaying or
    /// skipping rows.
    pub fn note_step(&mut self, global_step: u64) {
        self.last_global_step = global_step;
        if self
            .next_checkpoint_step
            .is_some_and(|next| global_step >= next)
        {
            self.checkpoint_due = true;
        }
    }

    /// Writes the pending step checkpoint, if one is due, at a safe boundary.
    pub fn checkpoint_boundary(
        &mut self,
        trainer: &mut Trainer,
        boundary: training::Boundary,
        global_step: u64,
        notify: &mut dyn FnMut(&std::path::Path),
    ) -> Result<()> {
        if !self.checkpoint_due && !self.checkpoint_requested {
            return Ok(());
        }
        // An on-demand checkpoint on a run configured without a checkpoint
        // directory has nowhere to go. The request is dropped rather than
        // failing the run: the caller was told at request time (the API refuses
        // it), and a training job must not die because of a control message.
        if self.checkpoint.is_none() {
            self.checkpoint_requested = false;
            return Ok(());
        }
        let path = self.write_checkpoint(
            trainer,
            &format!("step-{global_step:012}"),
            boundary,
            global_step,
        )?;
        notify(&path);
        // Only a *scheduled* checkpoint moves the schedule on. An on-demand one
        // that rearmed it would let a client silently reset the cadence the
        // configuration asked for by requesting snapshots.
        if self.checkpoint_due
            && let Some(frequency) = self
                .checkpoint
                .as_ref()
                .and_then(|checkpoint| checkpoint.every_steps)
        {
            self.next_checkpoint_step = Some((global_step / frequency + 1) * frequency);
        }
        self.checkpoint_due = false;
        self.checkpoint_requested = false;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Hot adjustments. Everything here is a *schedule*: when to evaluate,
    // when to snapshot, how long to wait for an improvement. Nothing that fixes
    // the memory footprint is reachable, because the budget was validated for one
    // geometry and the optimizer graph is already allocated.
    // -----------------------------------------------------------------------

    /// Asks for a checkpoint at the next resumable boundary.
    ///
    /// Not "now": `write_checkpoint` needs a [`training::Boundary`], and a
    /// snapshot taken between two of them would replay or skip rows on resume.
    /// The caller learns it landed from the `checkpoint` event.
    pub fn request_checkpoint(&mut self) {
        self.checkpoint_requested = true;
    }

    pub fn has_evaluation(&self) -> bool {
        self.evaluation.is_some()
    }

    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// Changes the evaluation cadence. Takes effect at the next iteration the
    /// new cadence selects.
    pub fn set_evaluation_every(&mut self, every_iterations: u32) -> Result<()> {
        if every_iterations == 0 {
            return Err(Error::invalid(
                "evaluation.every_iterations must be greater than zero",
            ));
        }
        let evaluation = self.evaluation.as_mut().ok_or_else(|| {
            Error::invalid("this run has no evaluation dataset, so it has no cadence to change")
        })?;
        evaluation.every_iterations = every_iterations;
        Ok(())
    }

    /// Changes the early-stopping patience. The count of stale evaluations so
    /// far is kept: lowering the patience below it stops the run at the next
    /// evaluation, which is what asking for a shorter patience means.
    pub fn set_patience(&mut self, patience: Option<u32>) -> Result<()> {
        let evaluation = self.evaluation.as_mut().ok_or_else(|| {
            Error::invalid("this run has no evaluation dataset, so it has no patience to change")
        })?;
        evaluation.patience = patience;
        Ok(())
    }

    /// Changes the step cadence of checkpoints, rearmed from where the run is
    /// now - not from step zero, which would fire immediately on a long run.
    pub fn set_checkpoint_every_steps(&mut self, every_steps: u64) -> Result<()> {
        if every_steps == 0 {
            return Err(Error::invalid(
                "checkpoint.every_steps must be greater than zero",
            ));
        }
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| Error::invalid("this run writes no checkpoints"))?;
        checkpoint.every_steps = Some(every_steps);
        let mode = checkpoint.mode;
        self.next_checkpoint_step = mode
            .includes_steps()
            .then(|| (self.last_global_step / every_steps + 1) * every_steps);
        Ok(())
    }

    /// Changes what a checkpoint is written for: the step schedule, the best
    /// evaluation, or both.
    pub fn set_checkpoint_mode(&mut self, mode: CheckpointMode) -> Result<()> {
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| Error::invalid("this run writes no checkpoints"))?;
        if mode.includes_best_eval() && self.evaluation.is_none() {
            return Err(Error::invalid(
                "checkpoint.mode includes best_eval but this run has no evaluation dataset",
            ));
        }
        checkpoint.mode = mode;
        let every_steps = checkpoint.every_steps;
        self.next_checkpoint_step = match (mode.includes_steps(), every_steps) {
            (true, Some(frequency)) => Some((self.last_global_step / frequency + 1) * frequency),
            // Dropping the step schedule also disarms a checkpoint that was
            // already due: the client just said it does not want those.
            _ => {
                self.checkpoint_due = false;
                None
            }
        };
        Ok(())
    }

    /// Writes one complete checkpoint and returns its state directory. The
    /// adapter GGUF lands beside it as a pure LoRA export.
    fn write_checkpoint(
        &self,
        trainer: &mut Trainer,
        id: &str,
        boundary: training::Boundary,
        global_step: u64,
    ) -> Result<PathBuf> {
        let directory = &self
            .checkpoint
            .as_ref()
            .expect("writing a checkpoint requires a checkpoint configuration")
            .directory;
        fs::create_dir_all(directory)?;
        let state_dir = directory.join(format!("{id}.{}", checkpoint::STATE_SUFFIX));
        let mut dataset = self.context.dataset.clone();
        dataset.cursor = boundary.cursor;
        let metadata = CheckpointMetadata {
            checkpoint_id: id.to_string(),
            algorithm: self.context.algorithm.clone(),
            trajectory_signature: self.context.trajectory_signature.clone(),
            resume_boundary: self.context.resume_boundary.clone(),
            scheduler_kind: self.context.scheduler_kind.clone(),
            warmup_steps: self.context.warmup_steps,
            progress: checkpoint::Progress {
                version: checkpoint::FORMAT_VERSION,
                epoch: boundary.completed_iterations,
                global_step,
                cursor: boundary.cursor,
                algorithm: self.context.algorithm.clone(),
                phase: "train".into(),
                best_eval: self.best_eval,
                stale_evaluations: self.stale_evaluations,
                kl_multiplier: boundary.kl_multiplier,
            },
            dataset,
            seeds: self.context.seeds.clone(),
            artifacts: BTreeMap::new(),
            model_path: self.context.model_path.clone(),
        };
        trainer.save_checkpoint(&state_dir, &metadata)?;
        Ok(state_dir)
    }

    pub fn record_evaluation(
        &mut self,
        trainer: &mut Trainer,
        value: f64,
        direction: EvalDirection,
        can_stop: bool,
        boundary: Option<training::Boundary>,
        global_step: u64,
    ) -> Result<EvalOutcome> {
        if !value.is_finite() {
            return Err(Error::runtime("evaluation metric is not finite"));
        }
        let evaluation = self
            .evaluation
            .as_ref()
            .expect("record_evaluation requires an evaluation configuration");
        let improved = self.best_eval.is_none_or(|best| match direction {
            EvalDirection::Lower => value < best - evaluation.min_delta,
            EvalDirection::Higher => value > best + evaluation.min_delta,
        });
        let mut saved = None;
        if improved {
            self.best_eval = Some(value);
            self.stale_evaluations = 0;
            let keeps_best = self
                .checkpoint
                .as_ref()
                .is_some_and(|checkpoint| checkpoint.mode.includes_best_eval());
            // Evaluations run at boundaries, so `best` is always resumable.
            if let (true, Some(boundary)) = (keeps_best, boundary) {
                saved = Some(self.write_checkpoint(trainer, "best", boundary, global_step)?);
            }
        } else {
            self.stale_evaluations = self.stale_evaluations.saturating_add(1);
        }
        let patience = evaluation.patience;
        let exhausted =
            can_stop && patience.is_some_and(|patience| self.stale_evaluations >= patience);
        if exhausted {
            self.early_stopped = true;
        }
        Ok(EvalOutcome {
            improved,
            best: self.best_eval.expect("an evaluation has been recorded"),
            stale: self.stale_evaluations,
            patience,
            saved,
            keep_training: !exhausted,
        })
    }
}

/// Complete checkpoints the directory will hold once the run has finished.
///
/// Nothing deletes one. The best-evaluation checkpoint counts once however
/// often the metric improves: it is one directory rewritten in place.
///
/// Checkpoints already on disk are subtracted: their bytes are already out of
/// the free figure, and counting them twice would refuse a resume for the
/// space it stands on.
fn retained_checkpoints(config: &CheckpointConfig, total_steps: u64) -> u64 {
    let scheduled = config
        .mode
        .includes_steps()
        .then_some(config.every_steps)
        .flatten()
        .filter(|every| *every > 0)
        .map_or(0, |every| total_steps / every);
    let best = u64::from(config.mode.includes_best_eval());
    scheduled
        .saturating_add(best)
        .saturating_sub(checkpoints_on_disk(&config.directory))
}

/// Complete checkpoints already in `directory`, counted by the state
/// directories that carry a manifest. An unreadable or absent directory is
/// zero.
fn checkpoints_on_disk(directory: &std::path::Path) -> u64 {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            entry.path().extension().is_some_and(|extension| {
                extension == std::ffi::OsStr::new(checkpoint::STATE_SUFFIX)
            }) && entry.path().join(checkpoint::MANIFEST_FILE).is_file()
        })
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{temp_path, write_sft_config};

    /// An SFT run with the sections a test asks for and nothing else.
    fn controller(
        name: &str,
        evaluation: bool,
        checkpoint: Option<(CheckpointMode, Option<u64>)>,
    ) -> (RunController, PathBuf) {
        let root = temp_path(name);
        let mut config = retrograd_config::load(write_sft_config(&root)).expect("load");
        config.evaluation = evaluation.then(|| {
            let data = root.join("eval.txt");
            fs::write(&data, "evaluation").expect("write the evaluation data");
            EvaluationConfig {
                data,
                every_iterations: 2,
                patience: Some(3),
                min_delta: 0.01,
                max_examples: None,
            }
        });
        config.checkpoint = checkpoint.map(|(mode, every_steps)| CheckpointConfig {
            directory: root.join("checkpoints"),
            mode,
            every_steps,
            resume_from: None,
        });
        (RunController::new(&config).expect("controller"), root)
    }

    #[test]
    fn the_disk_budget_counts_every_checkpoint_the_schedule_writes() {
        let root = temp_path("retained");
        let directory = root.join("checkpoints");
        let config = CheckpointConfig {
            directory: directory.clone(),
            mode: CheckpointMode::Steps,
            every_steps: Some(10),
            resume_from: None,
        };
        // Nothing deletes a checkpoint, so 100 steps at 10 leave ten
        // directories.
        assert_eq!(retained_checkpoints(&config, 100), 10);
        // The best-evaluation checkpoint is one directory, however often the
        // metric improves.
        let both = CheckpointConfig {
            mode: CheckpointMode::StepsAndBestEval,
            ..config.clone()
        };
        assert_eq!(retained_checkpoints(&both, 100), 11);
        let best_only = CheckpointConfig {
            mode: CheckpointMode::BestEval,
            ..config.clone()
        };
        assert_eq!(retained_checkpoints(&best_only, 100), 1);
        // No step cadence: the schedule predicts nothing.
        let on_demand = CheckpointConfig {
            every_steps: None,
            ..config.clone()
        };
        assert_eq!(retained_checkpoints(&on_demand, 100), 0);
        assert!(
            !directory.exists(),
            "the schedule is read, never written to"
        );
    }

    #[test]
    fn checkpoints_already_written_are_not_budgeted_twice() {
        let root = temp_path("retained-resume");
        let directory = root.join("checkpoints");
        fs::create_dir_all(&directory).expect("create the checkpoint directory");
        for step in [10_u64, 20, 30] {
            let state = directory.join(format!("step-{step:012}.{}", checkpoint::STATE_SUFFIX));
            fs::create_dir_all(&state).expect("create a state directory");
            fs::write(state.join(checkpoint::MANIFEST_FILE), b"manifest").expect("write");
        }
        // A directory without the marker is an interrupted write, not a
        // checkpoint.
        let partial = directory.join(format!("step-000000000040.{}", checkpoint::STATE_SUFFIX));
        fs::create_dir_all(&partial).expect("create a partial state directory");
        let config = CheckpointConfig {
            directory,
            mode: CheckpointMode::Steps,
            every_steps: Some(10),
            resume_from: None,
        };
        // Ten scheduled, three already out of the free figure.
        assert_eq!(retained_checkpoints(&config, 100), 7);
        fs::remove_dir_all(root).expect("clean up");
    }

    #[test]
    fn a_new_step_cadence_is_rearmed_from_where_the_run_is() {
        let (mut controller, root) =
            controller("rearm", false, Some((CheckpointMode::Steps, Some(10))));
        controller.note_step(25);
        assert!(controller.checkpoint_due);
        controller
            .set_checkpoint_every_steps(8)
            .expect("a positive cadence");
        assert_eq!(controller.next_checkpoint_step, Some(32));
        let error = controller
            .set_checkpoint_every_steps(0)
            .expect_err("a zero cadence");
        assert!(error.to_string().contains("greater than zero"), "{error}");
        fs::remove_dir_all(root).expect("clean up");
    }

    #[test]
    fn dropping_the_step_schedule_disarms_a_due_checkpoint() {
        let (mut controller, root) = controller(
            "disarm",
            true,
            Some((CheckpointMode::StepsAndBestEval, Some(10))),
        );
        controller.note_step(10);
        assert!(controller.checkpoint_due);
        controller
            .set_checkpoint_mode(CheckpointMode::BestEval)
            .expect("the run has an evaluation");
        assert!(!controller.checkpoint_due);
        assert_eq!(controller.next_checkpoint_step, None);
        fs::remove_dir_all(root).expect("clean up");
    }

    #[test]
    fn a_best_eval_checkpoint_needs_an_evaluation() {
        let (mut controller, root) =
            controller("best-eval", false, Some((CheckpointMode::Steps, Some(10))));
        let error = controller
            .set_checkpoint_mode(CheckpointMode::BestEval)
            .expect_err("no evaluation dataset");
        assert!(
            error.to_string().contains("has no evaluation dataset"),
            "{error}"
        );
        fs::remove_dir_all(root).expect("clean up");
    }

    #[test]
    fn an_adjustment_without_its_section_is_refused() {
        let (mut controller, root) = controller("absent", false, None);
        assert!(
            controller
                .set_evaluation_every(0)
                .expect_err("zero")
                .to_string()
                .contains("greater than zero")
        );
        assert!(controller.set_evaluation_every(3).is_err());
        assert!(controller.set_patience(Some(1)).is_err());
        assert!(
            controller
                .set_checkpoint_every_steps(5)
                .expect_err("no checkpoints")
                .to_string()
                .contains("writes no checkpoints")
        );
        fs::remove_dir_all(root).expect("clean up");
    }

    #[test]
    fn a_changed_evaluation_cadence_takes_over_the_schedule() {
        let (mut controller, root) = controller("cadence", true, None);
        assert!(controller.should_evaluate(2, 10));
        controller
            .set_evaluation_every(3)
            .expect("a positive cadence");
        assert!(!controller.should_evaluate(2, 10));
        assert!(controller.should_evaluate(3, 10));
        // The last iteration is always evaluated, whatever the cadence.
        assert!(controller.should_evaluate(10, 10));
        fs::remove_dir_all(root).expect("clean up");
    }
}
