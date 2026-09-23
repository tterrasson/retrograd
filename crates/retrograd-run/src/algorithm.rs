//! Per-algorithm wiring: `prepare → begin → run_resumed → eval → checkpoint →
//! bus.emit`, with progress reported through [`RunObserver`].

use retrograd_checkpoint as checkpoint;
use retrograd_config::{self as config, EvaluationConfig, RunConfig};
use retrograd_core::{Error, Result, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_memory::{self as memory, MemoryTracker};
use retrograd_metrics::{MetricEvent, MetricValue, MetricsBus};
use retrograd_observe::{Algorithm as ObservedAlgorithm, ObserveSink, TrajectoryObserver};
use retrograd_training as training;

use crate::control::{
    AdHocEvaluation, ControlPoint, GenerationOutput, GenerationRequest, RunControl, RunControls,
};
use crate::controller::{EvalDirection, RunController};
use crate::observe;
use crate::observer::{EvaluationReport, LoopPlan, RolloutEpoch, RunObserver, SftEpoch, SftStep};
use crate::signature::{checked_total_steps, dataset_fingerprint, prompts_dataset};

pub(crate) struct Context<'a> {
    pub config: &'a RunConfig,
    pub bus: &'a mut MetricsBus,
    pub controller: &'a mut RunController,
    pub memory: &'a mut MemoryTracker,
    pub observer: &'a mut dyn RunObserver,
    pub control: &'a mut dyn RunControl,
}

impl Context<'_> {
    /// Reports what the prepared datasets added to memory, then publishes the
    /// dataset series, plus `extra`, at step zero.
    fn datasets_prepared(
        &mut self,
        examples: usize,
        supervised_tokens: usize,
        extra: Vec<MetricValue>,
    ) -> Result<()> {
        if let Some(line) = self.memory.phase("datasets") {
            self.observer.info(&line);
        }
        let mut values = vec![
            MetricValue {
                name: "data/train_examples".into(),
                value: examples as f32,
            },
            MetricValue {
                name: "data/train_supervised_tokens".into(),
                value: supervised_tokens as f32,
            },
        ];
        values.extend(extra);
        self.bus.emit(&MetricEvent::Step {
            epoch: 0,
            global_step: 0,
            values,
        })
    }
}

/// Optimizer steps per prepared row: the runtime advances `n_batch` tokens per
/// step over an `n_ctx`-token row.
fn steps_per_row(config: &RunConfig) -> u64 {
    (config.training.n_ctx / config.training.n_batch) as u64
}

/// Appends the host and device memory series to `values`, and passes a
/// significant move on to the observer.
fn observe_memory(
    memory: &mut MemoryTracker,
    observer: &mut dyn RunObserver,
    values: &mut Vec<MetricValue>,
) {
    let (snapshot, note) = memory.observe();
    if let Some(snapshot) = snapshot {
        values.extend(memory::metric_values(snapshot));
    }
    values.extend(memory.device_metric_values());
    if let Some(note) = note {
        observer.memory_note(&note);
    }
}

/// How an algorithm answers an out-of-schedule evaluation.
///
/// A closure rather than a method on [`RunController`] because the dataset an
/// evaluation needs is the algorithm's: SFT holds a `PreparedDataset` it
/// tokenized once, PPO and GRPO hold a generate-and-score pass over prompts.
/// Neither is reachable from the controller, and hoisting either into it would
/// move algorithm state into the policy that schedules it.
pub(crate) type Evaluator<'a> = dyn FnMut(&mut Trainer) -> Result<AdHocEvaluation> + 'a;

/// The live half of [`RunControls`]: the trainer for the learning rate and for
/// the two readers, the controller for every schedule. Built for the duration of
/// one poll, so it borrows rather than owns and nothing outlives the callback.
pub(crate) struct LiveControls<'a, 'e> {
    pub(crate) trainer: &'a mut Trainer,
    pub(crate) controller: &'a mut RunController,
    pub(crate) evaluate: &'a mut Evaluator<'e>,
}

impl RunControls for LiveControls<'_, '_> {
    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        self.trainer.set_learning_rate(learning_rate)
    }

    fn set_evaluation_every(&mut self, every_iterations: u32) -> Result<()> {
        self.controller.set_evaluation_every(every_iterations)
    }

    fn set_patience(&mut self, patience: Option<u32>) -> Result<()> {
        self.controller.set_patience(patience)
    }

    fn set_checkpoint_every_steps(&mut self, every_steps: u64) -> Result<()> {
        self.controller.set_checkpoint_every_steps(every_steps)
    }

    fn set_checkpoint_mode(&mut self, mode: config::CheckpointMode) -> Result<()> {
        self.controller.set_checkpoint_mode(mode)
    }

    fn request_checkpoint(&mut self) {
        self.controller.request_checkpoint();
    }

    fn evaluate(&mut self) -> Result<AdHocEvaluation> {
        (self.evaluate)(self.trainer)
    }

    /// One sequence, so the training context serves when a run has no separate
    /// generation context - an SFT run never has one, and refusing to sample
    /// there would make the route useless for exactly the runs it is most useful
    /// for.
    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        let text = if request.chat {
            self.trainer
                .format_chat(&[("user", request.prompt.as_str())], true)?
        } else {
            request.prompt.clone()
        };
        let prompt = self.trainer.tokenize_text(&text)?;
        if prompt.is_empty() {
            return Err(Error::tokenize("the prompt tokenized to nothing"));
        }
        let sampling = retrograd_core::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            max_new_tokens: request.max_new_tokens,
            seed: request.seed,
        };
        let sampled = self.trainer.generate(&prompt, &sampling)?;
        let completion = self.trainer.detokenize(&sampled.tokens, false)?;
        // The reference pass reuses the same seed on purpose: with sampling
        // held fixed, the difference between the two texts is training and
        // nothing else.
        let base_text = request
            .include_base
            .then(|| {
                let base = self.trainer.generate_base(&prompt, &sampling)?;
                self.trainer.detokenize(&base.tokens, false)
            })
            .transpose()?;
        Ok(GenerationOutput {
            text: completion,
            prompt_tokens: prompt.len() as u64,
            tokens: sampled.tokens.len() as u64,
            base_text,
        })
    }
}

pub(crate) fn run_sft(
    trainer: &mut Trainer,
    sft: &config::SftConfig,
    ctx: &mut Context<'_>,
) -> Result<TrainMetrics> {
    let config = ctx.config;
    let train = training::sft::prepare(trainer, sft)?;
    // The dataset identity is part of what makes a resume valid, so it is
    // registered before the checkpoint is restored.
    let resume = ctx.controller.begin(
        trainer,
        checkpoint::Dataset {
            version: checkpoint::FORMAT_VERSION,
            path: sft.data.display().to_string(),
            fingerprint: dataset_fingerprint(&train.tokens, &train.labels),
            examples: train.examples as u64,
            row_width: train.n_ctx as u64,
            format: format!("{:?}", sft.data_format).to_lowercase(),
            permutation: Vec::new(),
            cursor: 0,
        },
        checked_total_steps(
            "SFT",
            &[
                train.examples as u64,
                steps_per_row(config),
                config.training.epochs as u64,
            ],
        )?,
    )?;
    let eval = config
        .evaluation
        .as_ref()
        .map(|evaluation| training::sft::prepare_eval(trainer, &evaluation.data, sft.data_format))
        .transpose()?;
    ctx.datasets_prepared(train.examples, train.supervised_tokens, Vec::new())?;
    // The product `checked_total_steps` validated above, minus the epoch factor.
    let steps_per_epoch = train.examples as u64 * steps_per_row(config);
    let total_epochs = config.training.epochs;
    run_supervised_epochs(
        trainer,
        ctx,
        eval,
        total_epochs,
        steps_per_epoch,
        |trainer, on_progress| training::sft::run_resumed(trainer, &train, resume, on_progress),
    )
}

/// The epoch loop the two supervised objectives share: SFT, and offline top-k
/// distillation.
///
/// They differ in what a target *is* - one token or `k` of them - and in
/// nothing else this function can see: the same prepared corpus, the same epoch
/// resume unit, the same forward-pass evaluation, the same control plane.
fn run_supervised_epochs(
    trainer: &mut Trainer,
    ctx: &mut Context<'_>,
    eval: Option<retrograd_dataset::PreparedDataset>,
    total_epochs: u32,
    steps_per_epoch: u64,
    run: impl FnOnce(
        &mut Trainer,
        &mut dyn FnMut(&mut Trainer, training::Progress) -> Result<bool>,
    ) -> Result<TrainMetrics>,
) -> Result<TrainMetrics> {
    let has_eval = eval.is_some();
    ctx.observer.loop_started(&LoopPlan::Sft {
        // `total_epochs` and not `training.epochs`: an offline distillation run
        // counts its passes with `distill.offline_epochs`, and the plan the
        // observer prints is the one the loop will actually run.
        epochs: total_epochs,
        has_eval,
        steps_per_epoch,
    });
    let mut last_eval_loss = f32::NAN;
    let result = run(trainer, &mut |trainer, mut event| {
        let Context {
            bus,
            controller,
            memory,
            observer,
            control,
            ..
        } = &mut *ctx;
        controller.note_step(event.metrics.global_step);
        // An ad-hoc evaluation is the scheduled forward pass without the
        // bookkeeping: the same prepared dataset, the same `eval_sft`, and no
        // `record_evaluation`.
        let mut ad_hoc = |trainer: &mut Trainer| {
            let dataset = eval.as_ref().ok_or_else(|| {
                Error::invalid(
                    "this run has no evaluation dataset, so there is nothing to evaluate",
                )
            })?;
            let metrics = trainer.eval_sft(dataset)?;
            Ok(AdHocEvaluation {
                loss: Some(metrics.loss()),
                perplexity: Some(metrics.perplexity()),
                examples: dataset.rows() as u64,
                ..Default::default()
            })
        };
        // Control is consulted first and acted on last. First, because a pause
        // must not sit between an evaluation and the checkpoint that records it;
        // last, because a stop still lets this callback finish - evaluating,
        // checkpointing and emitting metrics - so nothing already computed is
        // thrown away.
        let flow = control.poll(
            &mut LiveControls {
                trainer,
                controller,
                evaluate: &mut ad_hoc,
            },
            ControlPoint {
                iteration: event.metrics.epoch,
                global_step: event.metrics.global_step,
                at_boundary: event.boundary.is_some(),
            },
        )?;
        let mut keep_training = true;
        if event.metrics.epoch_complete
            && controller.should_evaluate(event.metrics.epoch, total_epochs)
        {
            let eval = eval
                .as_ref()
                .expect("an evaluation schedule has a prepared SFT dataset");
            let eval_metrics = trainer.eval_sft(eval)?;
            let loss = eval_metrics.loss();
            event.metrics.eval_loss = loss as f32;
            last_eval_loss = event.metrics.eval_loss;
            event = training::Progress::sft(event.metrics, true);
            event.values.push(MetricValue {
                name: "eval/perplexity".into(),
                value: eval_metrics.perplexity() as f32,
            });
            let outcome = controller.record_evaluation(
                trainer,
                loss,
                EvalDirection::Lower,
                event.metrics.epoch < total_epochs,
                event.boundary,
                event.metrics.global_step,
            )?;
            keep_training = outcome.keep_training;
            observer.evaluation(&EvaluationReport::Sft {
                epoch: event.metrics.epoch,
                loss,
                perplexity: eval_metrics.perplexity(),
                outcome,
            });
        }
        // An epoch boundary is the only point an SFT run can resume from.
        if let Some(boundary) = event.boundary {
            controller.checkpoint_boundary(
                trainer,
                boundary,
                event.metrics.global_step,
                &mut |path| observer.checkpoint_written(path),
            )?;
        }
        if event.metrics.epoch_complete {
            observe_memory(memory, &mut **observer, &mut event.values);
        }
        bus.emit(&MetricEvent::Step {
            epoch: event.metrics.epoch,
            global_step: event.metrics.global_step,
            values: event.values,
        })?;
        if event.metrics.epoch_complete {
            observer.sft_epoch(&SftEpoch {
                epoch: event.metrics.epoch,
                total_epochs,
                global_step: event.metrics.global_step,
                train_loss: event.metrics.train_loss,
                eval_loss: event.metrics.eval_loss,
                learning_rate: event.metrics.learning_rate,
                tokens_per_second: event.metrics.tokens_per_second,
            });
        } else {
            observer.sft_step(&SftStep {
                epoch: event.metrics.epoch,
                total_epochs,
                global_step: event.metrics.global_step,
                // The global step counter survives a resume, so the position
                // inside the epoch is read off it rather than counted here.
                epoch_step: event
                    .metrics
                    .global_step
                    .saturating_sub(steps_per_epoch * (event.metrics.epoch as u64 - 1)),
                steps_per_epoch,
                train_loss: event.metrics.train_loss,
                learning_rate: event.metrics.learning_rate,
                tokens_per_second: event.metrics.tokens_per_second,
            });
        }
        Ok(keep_training && flow.is_continue())
    });
    ctx.observer.loop_finished();
    let mut metrics = result?;
    metrics.eval_loss = last_eval_loss;
    Ok(metrics)
}

pub(crate) fn run_ppo(
    trainer: &mut Trainer,
    ppo: &config::PpoConfig,
    ctx: &mut Context<'_>,
) -> Result<TrainMetrics> {
    let config = ctx.config;
    let resume = ctx.controller.begin(
        trainer,
        prompts_dataset(&ppo.prompts, ppo.sampling.max_new_tokens)?,
        checked_total_steps(
            "PPO",
            &[
                ppo.updates as u64,
                ppo.ppo_epochs as u64,
                ppo.rollout_batch_size as u64,
                steps_per_row(config),
            ],
        )?,
    )?;
    let sink = observe::open(
        config,
        ObservedAlgorithm::Ppo,
        resume.map(|boundary| boundary.completed_iterations),
        [
            ("rollout_batch_size", ppo.rollout_batch_size as u64),
            ("max_new_tokens", u64::from(ppo.sampling.max_new_tokens)),
            ("updates", u64::from(ppo.updates)),
            ("epochs", u64::from(ppo.ppo_epochs)),
        ],
    )?;
    run_rollout_updates(
        trainer,
        ctx,
        "ppo",
        "reward",
        ppo.updates,
        ppo.ppo_epochs,
        sink,
        |trainer, observer, on_progress| {
            training::ppo::run_resumed(
                trainer,
                ppo,
                &config.training,
                resume,
                observer,
                on_progress,
            )
        },
        |trainer, evaluation| {
            training::ppo::evaluate(
                trainer,
                ppo,
                &config.training,
                &evaluation.data,
                evaluation.max_examples,
            )
        },
    )
}

pub(crate) fn run_grpo(
    trainer: &mut Trainer,
    grpo: &config::GrpoConfig,
    ctx: &mut Context<'_>,
) -> Result<TrainMetrics> {
    let config = ctx.config;
    let resume = ctx.controller.begin(
        trainer,
        prompts_dataset(&grpo.prompts, grpo.sampling.max_new_tokens)?,
        checked_total_steps(
            "GRPO",
            &[
                grpo.updates as u64,
                grpo.grpo_epochs as u64,
                grpo.prompts_per_update as u64,
                grpo.group_size as u64,
                steps_per_row(config),
            ],
        )?,
    )?;
    let sink = observe::open(
        config,
        ObservedAlgorithm::Grpo,
        resume.map(|boundary| boundary.completed_iterations),
        [
            ("group_size", grpo.group_size as u64),
            ("prompts_per_update", grpo.prompts_per_update as u64),
            ("max_new_tokens", u64::from(grpo.sampling.max_new_tokens)),
            ("updates", u64::from(grpo.updates)),
            ("epochs", u64::from(grpo.grpo_epochs)),
        ],
    )?;
    run_rollout_updates(
        trainer,
        ctx,
        "grpo",
        "reward",
        grpo.updates,
        grpo.grpo_epochs,
        sink,
        |trainer, observer, on_progress| {
            training::grpo::run_resumed(
                trainer,
                grpo,
                &config.training,
                resume,
                observer,
                on_progress,
            )
        },
        |trainer, evaluation| {
            training::grpo::evaluate(
                trainer,
                grpo,
                &config.training,
                &evaluation.data,
                evaluation.max_examples,
            )
        },
    )
}

pub(crate) fn run_distill(
    trainer: &mut Trainer,
    distill: &config::DistillConfig,
    ctx: &mut Context<'_>,
) -> Result<TrainMetrics> {
    // The two modes share a section and nothing else. Offline top-k never
    // generates, holds no teacher and resumes by epoch, so it takes the
    // supervised driver rather than the rollout one.
    if let Some(offline) = distill.mode.offline() {
        return run_distill_offline(trainer, offline, ctx);
    }
    let config = ctx.config;
    let resume = ctx.controller.begin(
        trainer,
        prompts_dataset(&distill.prompts, distill.sampling.max_new_tokens)?,
        checked_total_steps(
            "distillation",
            &[
                distill.updates as u64,
                distill.distill_epochs as u64,
                distill.prompts_per_update as u64,
                distill.samples_per_prompt as u64,
                steps_per_row(config),
            ],
        )?,
    )?;
    // One teacher for the run, shared by the update loop and the scheduled
    // evaluation. Two handles, one model: an evaluation that opened its own
    // would double the single memory term this algorithm adds over GRPO, at the
    // update boundary where the student's AdamW state is resident too.
    //
    // Empty here and filled by whichever of the two reaches it first, which is
    // always the loop: `resume_state` opens it after the refusals that cost
    // nothing, and the evaluation only runs at the end of an update.
    let teacher = training::distill::SharedTeacher::new();
    run_rollout_updates(
        trainer,
        ctx,
        "distill",
        "neg_teacher_kl",
        distill.updates,
        distill.distill_epochs,
        // `[observe]` is refused for distillation by the loader.
        None,
        |trainer, _, on_progress| {
            training::distill::run_resumed(
                trainer,
                distill,
                &config.training,
                &teacher,
                resume,
                on_progress,
            )
        },
        |trainer, evaluation| {
            training::distill::evaluate(
                trainer,
                distill,
                &config.training,
                &teacher,
                &evaluation.data,
                evaluation.max_examples,
            )
        },
    )
}

/// Offline top-k distillation: the teacher's precomputed
/// distribution over a fixed corpus, trained on as an SFT epoch loop.
///
/// The dataset identity registered with the controller covers *both* files:
/// the sidecar's own hashes have already refused a mismatched pair in
/// `distill::offline::prepare`, and folding `k` into the fingerprint here is
/// what stops a resume from continuing a run under a sidecar that has been
/// regenerated with a different `k` since.
fn run_distill_offline(
    trainer: &mut Trainer,
    offline: &config::OfflineDistillConfig,
    ctx: &mut Context<'_>,
) -> Result<TrainMetrics> {
    let config = ctx.config;
    let prepared = training::distill::offline::prepare(trainer, offline)?;
    let resume = ctx.controller.begin(
        trainer,
        checkpoint::Dataset {
            version: checkpoint::FORMAT_VERSION,
            path: offline.data.display().to_string(),
            fingerprint: dataset_fingerprint(&prepared.batch.tokens, &prepared.batch.labels),
            examples: prepared.batch.n_rows as u64,
            row_width: prepared.batch.n_ctx as u64,
            format: format!("topk_offline_k{}", prepared.k),
            permutation: Vec::new(),
            cursor: 0,
        },
        checked_total_steps(
            "offline distillation",
            &[
                prepared.batch.n_rows as u64,
                steps_per_row(config),
                offline.epochs as u64,
            ],
        )?,
    )?;
    ctx.datasets_prepared(
        prepared.batch.n_rows,
        prepared.supervised_positions,
        vec![MetricValue {
            name: "distill/topk_entries".into(),
            value: prepared.k as f32,
        }],
    )?;
    let steps_per_epoch = prepared.batch.n_rows as u64 * steps_per_row(config);
    let epochs = offline.epochs;
    let training_config = config.training.clone();
    // No evaluation dataset: the forward-pass loss of an SFT evaluation is
    // against one-hot targets, which is not the quantity this run optimizes.
    // `bench` on the same document is what reads the divergence to the teacher,
    // and it needs the teacher a training run deliberately does not hold.
    run_supervised_epochs(
        trainer,
        ctx,
        None,
        epochs,
        steps_per_epoch,
        |trainer, on_progress| {
            training::distill::offline::run_resumed(
                trainer,
                &prepared,
                &training_config,
                epochs,
                resume,
                on_progress,
            )
        },
    )
}

/// Shared driver of the rollout-based algorithms: one observation per optimizer
/// epoch, metrics forwarded to the sinks, evaluation and checkpoint at update
/// boundaries. `sink` is the `[observe]` export, closed here whatever the
/// outcome.
#[expect(clippy::too_many_arguments)]
fn run_rollout_updates(
    trainer: &mut Trainer,
    ctx: &mut Context<'_>,
    algorithm: &'static str,
    // Stem of the `eval/*` series this loop publishes, i.e. what the evaluated
    // scalar actually is. `"reward"` for the three algorithms graded by a reward
    // process; a distillation run is graded on its divergence from the teacher,
    // and calling that a reward in the metric stream would mislabel the one
    // number a reader uses to decide the run is working.
    eval_series: &'static str,
    updates: u32,
    epochs_per_update: u32,
    sink: Option<ObserveSink>,
    run: impl FnOnce(
        &mut Trainer,
        Option<&dyn TrajectoryObserver>,
        &mut dyn FnMut(&mut Trainer, training::Progress) -> Result<bool>,
    ) -> Result<TrainMetrics>,
    mut evaluate: impl FnMut(&mut Trainer, &EvaluationConfig) -> Result<training::RewardEvalMetrics>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::rollout", "rollout", algorithm);
    let _entered = span.enter();
    let total_epochs = updates as u64 * epochs_per_update as u64;
    ctx.observer.loop_started(&LoopPlan::Rollout {
        algorithm,
        updates,
        epochs_per_update,
        total_epochs,
    });
    let mut rollout_position = RolloutPosition::default();
    let config = ctx.config;
    let handle = sink.as_ref().map(ObserveSink::observer);
    let result = run(trainer, handle.as_deref(), &mut |trainer, mut event| {
        let Context {
            bus,
            controller,
            memory,
            observer,
            control,
            ..
        } = &mut *ctx;
        controller.note_step(event.metrics.global_step);
        // The algorithm collects its free-form lines instead of printing them,
        // because on the CLI a raw write lands on the progress bar's row. They
        // go out before the epoch row so they keep their place in the run.
        for note in event.notes.drain(..) {
            observer.info(&note);
        }
        // The scheduled rollout evaluation, on demand. It borrows `evaluate`
        // mutably, which is why it is built here and dropped before the
        // scheduled call below reaches for the same closure.
        let mut ad_hoc = |trainer: &mut Trainer| {
            let evaluation = config.evaluation.as_ref().ok_or_else(|| {
                Error::invalid(
                    "this run has no evaluation dataset, so there is nothing to evaluate",
                )
            })?;
            let metrics = evaluate(trainer, evaluation)?;
            Ok(AdHocEvaluation {
                mean_reward: Some(metrics.mean_reward),
                reward_min: Some(metrics.reward_min),
                reward_max: Some(metrics.reward_max),
                examples: metrics.examples as u64,
                ..Default::default()
            })
        };
        // Before the early return: a cancel must not have to wait for the next
        // completed optimizer epoch, which on a rollout algorithm is a whole
        // generation pass away.
        let flow = control.poll(
            &mut LiveControls {
                trainer,
                controller,
                evaluate: &mut ad_hoc,
            },
            ControlPoint {
                iteration: event.metrics.epoch,
                global_step: event.metrics.global_step,
                at_boundary: event.boundary.is_some(),
            },
        )?;
        if !event.metrics.epoch_complete {
            return Ok(flow.is_continue());
        }
        // `metrics.epoch` is the absolute one-based update index supplied by
        // PPO/GRPO. Deriving it from callbacks would restart evaluation and UI
        // schedules at one after a checkpoint resume.
        let (update, policy_epoch, absolute_epoch) =
            rollout_position.observe(event.metrics.epoch, epochs_per_update);
        if policy_epoch == epochs_per_update as u64 {
            tracing::info!(
                target: "retrograd::training::update",
                algorithm,
                update,
                global_step = event.metrics.global_step,
                "optimizer update completed"
            );
        }
        let metric = |name: &str| {
            event
                .values
                .iter()
                .find(|value| value.name == name)
                .map(|value| value.value)
                .unwrap_or(f32::NAN)
        };
        observer.rollout_epoch(&RolloutEpoch {
            update,
            updates,
            policy_epoch,
            epochs_per_update,
            absolute_epoch,
            global_step: event.metrics.global_step,
            train_loss: event.metrics.train_loss,
            reward: metric("reward/mean"),
            kl: metric("policy/kl"),
            clip_fraction: metric("policy/clip_fraction"),
            learning_rate: event.metrics.learning_rate,
            tokens_per_second: event.metrics.tokens_per_second,
        });
        let mut keep_training = true;
        if policy_epoch == epochs_per_update as u64
            && controller.should_evaluate(update as u32, updates)
        {
            let evaluation = config
                .evaluation
                .as_ref()
                .expect("an evaluation schedule has an evaluation dataset");
            observer.evaluation_started(update, updates);
            let metrics = evaluate(trainer, evaluation)?;
            event.values.extend([
                MetricValue {
                    name: format!("eval/mean_{eval_series}").into(),
                    value: metrics.mean_reward,
                },
                MetricValue {
                    name: format!("eval/{eval_series}_min").into(),
                    value: metrics.reward_min,
                },
                MetricValue {
                    name: format!("eval/{eval_series}_max").into(),
                    value: metrics.reward_max,
                },
                MetricValue {
                    name: "eval/examples".into(),
                    value: metrics.examples as f32,
                },
                // Non-zero means some held-out prompts were generated under a
                // shorter budget than the rest, so the mean mixes regimes.
                MetricValue {
                    name: "eval/budget_clamped_fraction".into(),
                    value: metrics.budget_clamped_fraction,
                },
            ]);
            let outcome = controller.record_evaluation(
                trainer,
                metrics.mean_reward as f64,
                EvalDirection::Higher,
                update < updates as u64,
                event.boundary,
                event.metrics.global_step,
            )?;
            keep_training = outcome.keep_training;
            observer.evaluation(&EvaluationReport::Rollout {
                update,
                updates,
                mean_reward: metrics.mean_reward,
                reward_min: metrics.reward_min,
                reward_max: metrics.reward_max,
                examples: metrics.examples,
                outcome,
            });
        }
        // Save only after evaluation has updated best/stale/patience state.
        // The boundary itself already carries the algorithm state for the next
        // update (including GRPO's adaptive-KL multiplier).
        if let Some(boundary) = event.boundary {
            controller.checkpoint_boundary(
                trainer,
                boundary,
                event.metrics.global_step,
                &mut |path| observer.checkpoint_written(path),
            )?;
        }
        observe_memory(memory, &mut **observer, &mut event.values);
        if let Some(sink) = &sink {
            observe::report(sink, &mut **observer, &mut event.values);
            if policy_epoch == epochs_per_update as u64 {
                sink.update_summary(
                    event.metrics.epoch,
                    event
                        .values
                        .iter()
                        .map(|value| (value.name.as_ref(), value.value)),
                );
            }
        }
        bus.emit(&MetricEvent::Step {
            epoch: event.metrics.epoch,
            global_step: event.metrics.global_step,
            values: event.values,
        })?;
        Ok(keep_training && flow.is_continue())
    });
    drop(handle);
    observe::close(sink, ctx.observer);
    ctx.observer.loop_finished();
    result
}

#[derive(Default)]
struct RolloutPosition {
    current_update: Option<u64>,
    policy_epoch: u64,
}

impl RolloutPosition {
    fn observe(&mut self, absolute_update: u32, epochs_per_update: u32) -> (u64, u64, u64) {
        let update = absolute_update as u64;
        if self.current_update == Some(update) {
            self.policy_epoch += 1;
        } else {
            self.current_update = Some(update);
            self.policy_epoch = 1;
        }
        let absolute_epoch = update
            .saturating_sub(1)
            .saturating_mul(epochs_per_update as u64)
            .saturating_add(self.policy_epoch);
        (update, self.policy_epoch, absolute_epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_position_keeps_absolute_updates_after_resume() {
        let mut position = RolloutPosition::default();
        assert_eq!(position.observe(6, 2), (6, 1, 11));
        assert_eq!(position.observe(6, 2), (6, 2, 12));
        assert_eq!(position.observe(7, 2), (7, 1, 13));
    }
}
