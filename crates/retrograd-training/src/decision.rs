//! Fine-tuning a decision model on typed questions.
//!
//! Every question is rendered through the model's `systemone` template, laid
//! out by its readout and tokenized exactly as the server does it, so the
//! tokens trained on are the tokens served. Each question has its own
//! sequence. Marker heads train grouped softmax cross-entropy over their
//! option scores; causal heads train a masked letter completion against hard
//! or sparse soft targets. Several micro-batches accumulate into one update.
//!
//! The whole run is laid out before its first step - the order of the
//! questions, the order of the options of every `choice` question, the
//! micro-batches and the updates of every epoch - so the learning-rate
//! horizon is the exact number of updates and an epoch boundary is enough to
//! resume.
//!
//! Marker readouts train with grouped cross-entropy. Causal letter readouts
//! train the final prompt position with masked sparse next-token targets.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;

use retrograd_checkpoint as checkpoint;
use retrograd_config::DecisionConfig;
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_decision::answer::softmax;
use retrograd_decision::calibrate::{self, Sample};
use retrograd_decision::dataset::{self, Example};
use retrograd_decision::metrics::{Metrics, Outcome};
use retrograd_decision::readout::Readout;
use retrograd_decision::request::{Question, QuestionKind};
use retrograd_engine::{
    DecisionInfo, DecisionPass, GroupedBatch, GroupedQuestion, PackedSequenceBatch, Trainer,
};
use retrograd_metrics::MetricValue;

use crate::grpo::shuffled_indices;
use crate::rollout::evenly_spaced_subset;
use crate::{Boundary, Progress};

/// The scores are trained at temperature 1. The temperatures the model is
/// served with are fitted afterwards, on held-out questions.
const TRAINING_INV_TEMPERATURE: f32 = 1.0;

/// One question ready to train: its tokens, where its options are read, and
/// the target probability of each option in the order it was rendered.
#[derive(Clone, Debug)]
struct Rendered {
    pass: DecisionPass,
    targets: Vec<f32>,
}

/// One epoch: its questions in the order they train, and its optimizer
/// updates, each a list of micro-batches given as ranges of `questions`.
#[derive(Debug)]
struct EpochPlan {
    questions: Vec<Rendered>,
    updates: Vec<Vec<Range<usize>>>,
}

/// One held-out question.
#[derive(Clone, Debug)]
struct EvalItem {
    kind: QuestionKind,
    n_options: usize,
    pass: DecisionPass,
    target: Vec<f64>,
}

/// A decision dataset ready to train: every epoch laid out, and the held-out
/// questions rendered once.
#[derive(Debug)]
pub struct PreparedDecision {
    info: DecisionInfo,
    template: Option<String>,
    micro_batch: usize,
    epochs: Vec<EpochPlan>,
    eval: Vec<EvalItem>,
    total_steps: u64,
    examples: usize,
    tokens: usize,
    fingerprint: String,
}

impl PreparedDecision {
    /// The head the questions were rendered for, budget override included.
    pub fn info(&self) -> &DecisionInfo {
        &self.info
    }

    /// The effective replacement template, ready to embed in the exported GGUF.
    pub fn template(&self) -> Option<&str> {
        self.template.as_deref()
    }

    /// Training questions (one per example).
    pub fn examples(&self) -> usize {
        self.examples
    }

    /// Held-out questions.
    pub fn eval_examples(&self) -> usize {
        self.eval.len()
    }

    /// Tokens of one epoch's questions.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The exact number of optimizer updates the run takes.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Updates in one epoch, the first; every epoch has about as many.
    pub fn steps_per_epoch(&self) -> u64 {
        self.epochs
            .first()
            .map_or(0, |epoch| epoch.updates.len() as u64)
    }

    /// Fingerprint of every token, read and target of the run, held-out set
    /// included, in training order.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Refuses an example the readout cannot train on as the dataset wrote it:
/// fewer than two or too many options, or several renderings averaged.
fn check_example(readout: &Readout, example: &Example, index: usize) -> Result<()> {
    let n_options = example.question.options.len();
    if n_options < 2 {
        return Err(Error::invalid(format!(
            "decision example {} needs at least two training options",
            index + 1
        )));
    }
    if n_options > readout.n_options_max() {
        return Err(Error::invalid(format!(
            "decision example {} has {n_options} options; this model reads at most {}",
            index + 1,
            readout.n_options_max()
        )));
    }
    if readout.n_variants(&example.question) != 1 {
        return Err(Error::invalid(
            "a readout that averages several renderings of a question does not train here",
        ));
    }
    Ok(())
}

/// The options of a `choice` question in a seeded order, and its targets in
/// the same order. Score levels are ordinal and `noul` is fixed: only a
/// choice is shuffled.
fn shuffled_question(example: &Example, seed: u64) -> (Question, Vec<f64>) {
    let n = example.question.options.len();
    let targets = example.target.distribution(n);
    if example.question.kind != QuestionKind::Choice || n < 2 {
        return (example.question.clone(), targets);
    }
    let order = shuffled_indices(n, seed);
    let mut question = example.question.clone();
    question.options = order
        .iter()
        .map(|&i| example.question.options[i].clone())
        .collect();
    (question, order.iter().map(|&i| targets[i]).collect())
}

/// A seed per (run, epoch, example): a resumed epoch shuffles as it did.
fn option_seed(seed: u32, epoch: u32, example: usize) -> u64 {
    let mut state = (u64::from(seed) << 32) ^ u64::from(epoch).rotate_left(17) ^ example as u64;
    // splitmix64, so neighbouring examples get unrelated orders.
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn render(
    trainer: &Trainer,
    info: &DecisionInfo,
    example: &Example,
    question: &Question,
    targets: &[f64],
    micro_batch: usize,
    index: usize,
) -> Result<Rendered> {
    let pass = trainer.decision_pass(info, &example.state, &example.question_id, question, 0)?;
    if pass.tokens.len() > micro_batch {
        return Err(Error::config(format!(
            "decision example {} renders to {} tokens, more than one micro-batch of {}: raise \
             training.micro_batch (and training.ctx with it), or lower decision.max_head_tokens",
            index + 1,
            pass.tokens.len(),
            micro_batch
        )));
    }
    if pass.reads.len() != targets.len() {
        return Err(Error::runtime(format!(
            "decision example {} reads {} options for {} targets",
            index + 1,
            pass.reads.len(),
            targets.len()
        )));
    }
    Ok(Rendered {
        pass,
        // Narrowing: a probability in [0, 1], which f32 holds to 1e-7.
        targets: targets.iter().map(|&p| p as f32).collect(),
    })
}

/// Consecutive questions grouped into micro-batches of at most
/// `questions_per_batch` questions and `micro_batch` tokens, and the
/// micro-batches into updates of `accumulation`. The last update of an epoch
/// may be shorter: the period of an update is set at its first step.
fn plan_updates(
    questions: &[Rendered],
    micro_batch: usize,
    questions_per_batch: usize,
    accumulation: usize,
) -> Vec<Vec<Range<usize>>> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut tokens = 0;
    for (index, question) in questions.iter().enumerate() {
        let n = question.pass.tokens.len();
        if index > start && (tokens + n > micro_batch || index - start >= questions_per_batch) {
            batches.push(start..index);
            start = index;
            tokens = 0;
        }
        tokens += n;
    }
    if start < questions.len() {
        batches.push(start..questions.len());
    }
    batches
        .chunks(accumulation.max(1))
        .map(<[Range<usize>]>::to_vec)
        .collect()
}

/// Reads and renders the training questions of every epoch, and the
/// held-out questions when `eval` names a file (capped at `max_examples`
/// evenly spaced ones).
///
/// The template override of the run is installed on the trainer here, before
/// anything is rendered.
pub fn prepare(
    trainer: &mut Trainer,
    config: &DecisionConfig,
    training: &TrainConfig,
    epochs: u32,
    eval: Option<(&Path, Option<usize>)>,
) -> Result<PreparedDecision> {
    trainer.set_chat_template_variables(Some(
        &serde_json::Value::Object(config.template_variables.clone()).to_string(),
    ))?;
    let override_source = config
        .template
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|error| {
                Error::config(format!(
                    "decision.template {} cannot be read: {error}",
                    path.display()
                ))
            })
        })
        .transpose()?;
    let metadata = trainer.decision_metadata()?;
    let (mut info, mut template) = if metadata.decision_type.is_none() {
        let (info, source) = trainer.configure_causal_decision()?;
        (info, Some(source))
    } else {
        if let Some(source) = &override_source {
            trainer.set_decision_template(Some(source))?;
        }
        (trainer.decision_info()?, None)
    };
    if let Some(source) = override_source {
        trainer.set_decision_template(Some(&source))?;
        template = Some(source);
    }
    if let Some(max_head_tokens) = config.max_head_tokens {
        info = info.with_max_head_tokens(max_head_tokens as usize)?;
    }
    if !matches!(info.readout, Readout::MarkerColumn { .. }) && info.decision_type != "openjev" {
        return Err(Error::config(format!(
            "decision type '{}' does not support training; use a marker or openjev letter readout",
            info.decision_type
        )));
    }

    let micro_batch = training.n_ubatch as usize;
    // Hybrid/recurrent graphs require equal-sequence ubatches. A single
    // question uses that ordinary split; dense causal graphs can pack several.
    let questions_per_batch = if matches!(info.readout, Readout::LabelLogits { .. })
        && !trainer.supports_shared_prefix_packed_training()?
    {
        1
    } else {
        config.questions_per_batch as usize
    };
    let accumulation = config.gradient_accumulation as usize;
    let examples: Vec<_> = dataset::read_jsonl(&config.data, config.data_format)?
        .into_iter()
        .map(|example| example.for_readout(&info.readout))
        .collect();
    for (index, example) in examples.iter().enumerate() {
        check_example(&info.readout, example, index)?;
    }

    // Questions whose rendering does not depend on the epoch are rendered
    // once and reused.
    let fixed: Vec<Option<Rendered>> = examples
        .iter()
        .enumerate()
        .map(|(index, example)| {
            let shuffles = config.option_shuffle && example.question.kind == QuestionKind::Choice;
            if shuffles {
                return Ok(None);
            }
            let targets = example.target.distribution(example.question.options.len());
            render(
                trainer,
                &info,
                example,
                &example.question,
                &targets,
                micro_batch,
                index,
            )
            .map(Some)
        })
        .collect::<Result<_>>()?;

    let mut plans = Vec::with_capacity(epochs as usize);
    for epoch in 0..epochs {
        let order: Vec<usize> = if config.shuffle {
            let seed = (u64::from(config.seed) << 32) ^ u64::from(epoch).wrapping_add(1);
            shuffled_indices(examples.len(), seed)
        } else {
            (0..examples.len()).collect()
        };
        let questions = order
            .iter()
            .map(|&index| match &fixed[index] {
                Some(rendered) => Ok(rendered.clone()),
                None => {
                    let example = &examples[index];
                    let (question, targets) =
                        shuffled_question(example, option_seed(config.seed, epoch, index));
                    render(
                        trainer,
                        &info,
                        example,
                        &question,
                        &targets,
                        micro_batch,
                        index,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let updates = plan_updates(&questions, micro_batch, questions_per_batch, accumulation);
        plans.push(EpochPlan { questions, updates });
    }

    let eval = match eval {
        Some((path, max_examples)) => {
            let examples: Vec<_> = dataset::read_jsonl(path, config.data_format)?
                .into_iter()
                .map(|example| example.for_readout(&info.readout))
                .collect();
            let examples = match max_examples {
                Some(max) => evenly_spaced_subset(examples, max),
                None => examples,
            };
            examples
                .iter()
                .enumerate()
                .map(|(index, example)| {
                    check_example(&info.readout, example, index)?;
                    let pass = trainer.decision_pass(
                        &info,
                        &example.state,
                        &example.question_id,
                        &example.question,
                        0,
                    )?;
                    Ok(EvalItem {
                        kind: example.question.kind,
                        n_options: example.question.options.len(),
                        pass,
                        target: example.target.distribution(example.question.options.len()),
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => Vec::new(),
    };

    let mut fingerprinter = checkpoint::Fingerprinter::new();
    let mut feed = |label: &[u8], pass: &DecisionPass, targets: &mut dyn Iterator<Item = f64>| {
        fingerprinter.update(label);
        fingerprinter.update(&(pass.tokens.len() as u64).to_le_bytes());
        for token in &pass.tokens {
            fingerprinter.update(&token.to_le_bytes());
        }
        for read in &pass.reads {
            fingerprinter.update(&read.to_le_bytes());
        }
        fingerprinter.update(&pass.column.to_le_bytes());
        for target in targets {
            fingerprinter.update(&target.to_le_bytes());
        }
    };
    for plan in &plans {
        for question in &plan.questions {
            feed(
                b"train",
                &question.pass,
                &mut question.targets.iter().map(|&p| f64::from(p)),
            );
        }
    }
    for item in &eval {
        feed(b"eval", &item.pass, &mut item.target.iter().copied());
    }
    let total_steps = plans
        .iter()
        .map(|plan| plan.updates.len() as u64)
        .fold(0_u64, u64::saturating_add);
    let tokens = plans.first().map_or(0, |plan| {
        plan.questions
            .iter()
            .map(|question| question.pass.tokens.len())
            .sum()
    });
    Ok(PreparedDecision {
        info,
        template,
        micro_batch,
        epochs: plans,
        eval,
        total_steps,
        examples: examples.len(),
        tokens,
        fingerprint: fingerprinter.finish(),
    })
}

/// One masked letter completion per question. A hard target has K=1; a soft
/// target carries its nonzero option probabilities. Prompt and padding
/// positions have label -1 and zero weight on both dense and fused CE paths.
fn causal_batch(questions: &[Rendered], width: usize) -> Result<PackedSequenceBatch> {
    let k = questions
        .iter()
        .map(|q| q.targets.iter().filter(|&&p| p > 0.0).count())
        .max()
        .unwrap_or(0);
    if k == 0 || k > retrograd_core::FUSED_CE_K_MAX {
        return Err(Error::invalid(
            "causal targets exceed the sparse cross-entropy capacity",
        ));
    }
    let entries = width
        .checked_mul(k)
        .ok_or_else(|| Error::overflow("causal batch shape"))?;
    let mut batch = PackedSequenceBatch {
        tokens: Vec::with_capacity(width),
        positions: Vec::with_capacity(width),
        labels: vec![-1; entries],
        weights: vec![0.0; entries],
        seq_offsets: (0..=width).collect(),
        seq_ids: Vec::with_capacity(width),
        n_sequences: questions.len(),
        n_topk: k,
    };
    for (sequence, question) in questions.iter().enumerate() {
        if question.pass.tokens.is_empty()
            || question.pass.reads.len() != question.targets.len()
            || batch.tokens.len() + question.pass.tokens.len() > width
        {
            return Err(Error::invalid(
                "causal question does not fit its micro-batch",
            ));
        }
        let sequence =
            i32::try_from(sequence).map_err(|_| Error::overflow("causal sequence ID"))?;
        for (position, &token) in question.pass.tokens.iter().enumerate() {
            batch.tokens.push(token);
            batch
                .positions
                .push(i32::try_from(position).map_err(|_| Error::overflow("causal position"))?);
            batch.seq_ids.push(sequence);
        }
        let offset = (batch.tokens.len() - 1) * k;
        for (j, (&token, &probability)) in question
            .pass
            .reads
            .iter()
            .zip(&question.targets)
            .filter(|(_, p)| **p > 0.0)
            .enumerate()
        {
            batch.labels[offset + j] = token;
            batch.weights[offset + j] = probability;
        }
    }
    // Padding extends the last sequence after its answer position, so causal
    // masking prevents it from affecting any supervised logits.
    let last = batch
        .tokens
        .last()
        .copied()
        .ok_or_else(|| Error::invalid("empty causal batch"))?;
    let sequence = *batch.seq_ids.last().expect("a token has a sequence");
    let mut position = *batch.positions.last().expect("a token has a position");
    while batch.tokens.len() < width {
        position = position
            .checked_add(1)
            .ok_or_else(|| Error::overflow("causal padding position"))?;
        batch.tokens.push(last);
        batch.positions.push(position);
        batch.seq_ids.push(sequence);
    }
    Ok(batch)
}

/// What one optimizer update adds up to.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    questions: usize,
    loss: f64,
    tokens: usize,
    seconds: f64,
}

impl Totals {
    fn merge(&mut self, other: &Totals) {
        self.questions += other.questions;
        self.loss += other.loss;
        self.tokens += other.tokens;
        self.seconds += other.seconds;
    }

    fn loss(&self) -> f64 {
        self.loss / self.questions.max(1) as f64
    }

    fn tokens_per_second(&self) -> f32 {
        if self.seconds > 0.0 {
            (self.tokens as f64 / self.seconds) as f32
        } else {
            0.0
        }
    }
}

/// One optimizer update: its micro-batches, one grouped step each.
fn update(
    trainer: &mut Trainer,
    prepared: &PreparedDecision,
    questions: &[Rendered],
    batches: &[Range<usize>],
) -> Result<(TrainMetrics, Totals)> {
    let period = u32::try_from(batches.len())
        .map_err(|_| Error::overflow("micro-batches per update exceed u32"))?;
    let mut totals = Totals::default();
    let mut metrics = TrainMetrics::default();
    for batch in batches {
        let questions = &questions[batch.clone()];
        let started = std::time::Instant::now();
        if matches!(prepared.info.readout, Readout::LabelLogits { .. }) {
            let packed = causal_batch(questions, prepared.micro_batch)?;
            metrics = trainer
                .train_packed_sequences_controlled(
                    &packed,
                    prepared.total_steps,
                    period,
                    |_, _| Ok(true),
                )?
                .0;
        } else {
            let members: Vec<GroupedQuestion<'_>> = questions
                .iter()
                .map(|question| GroupedQuestion {
                    tokens: &question.pass.tokens,
                    reads: &question.pass.reads,
                    column: question.pass.column,
                    pointer: question.pass.pointer,
                    targets: &question.targets,
                })
                .collect();
            let grouped = GroupedBatch::pack(&members, TRAINING_INV_TEMPERATURE, period)?;
            metrics = trainer.train_grouped(&grouped, prepared.total_steps)?;
        }
        totals.seconds += started.elapsed().as_secs_f64();
        // The step's loss is the mean over its questions.
        totals.loss += f64::from(metrics.train_loss) * questions.len() as f64;
        totals.questions += questions.len();
        totals.tokens += questions.iter().map(|q| q.pass.tokens.len()).sum::<usize>();
    }
    Ok((metrics, totals))
}

/// Runs the epochs of `prepared`, from the epoch boundary `resume` restores,
/// if any. One progress event per optimizer update; the last update of an
/// epoch reports the epoch, with its boundary.
pub fn run_resumed(
    trainer: &mut Trainer,
    prepared: &PreparedDecision,
    resume: Option<Boundary>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(
        target: "retrograd::training::decision",
        "training",
        questions = prepared.examples
    );
    let _entered = span.enter();
    let start_epoch = match resume {
        Some(boundary) => u32::try_from(boundary.completed_iterations).map_err(|_| {
            Error::overflow("the checkpoint epoch count does not fit in the epoch counter")
        })?,
        None => 0,
    };
    let mut global_step = trainer.advance_scheduler_steps(0)?;
    let mut final_metrics = TrainMetrics::default();
    for (epoch, plan) in prepared.epochs.iter().enumerate() {
        // Bounded by `epochs`, a u32.
        let epoch = epoch as u32;
        if epoch < start_epoch {
            continue;
        }
        let mut epoch_totals = Totals::default();
        let last = plan.updates.len().saturating_sub(1);
        for (index, batches) in plan.updates.iter().enumerate() {
            let (mut metrics, totals) = update(trainer, prepared, &plan.questions, batches)?;
            let taken = metrics
                .global_step
                .checked_sub(global_step)
                .ok_or_else(|| Error::runtime("the optimizer step counter moved backwards"))?;
            if taken != 1 {
                return Err(Error::runtime(format!(
                    "a decision update took {taken} optimizer steps where its plan has one"
                )));
            }
            global_step = metrics.global_step;
            epoch_totals.merge(&totals);
            let at_epoch_end = index == last;
            metrics.epoch = epoch + 1;
            metrics.epoch_complete = at_epoch_end;
            metrics.eval_loss = f32::NAN;
            let shown = if at_epoch_end { &epoch_totals } else { &totals };
            metrics.train_loss = shown.loss() as f32;
            metrics.tokens_per_second = shown.tokens_per_second();
            let mut progress = Progress::sft(metrics, false);
            progress.values.push(MetricValue {
                name: "decision/questions_per_update".into(),
                value: totals.questions as f32,
            });
            final_metrics = metrics;
            if !on_progress(trainer, progress)? {
                return Ok(final_metrics);
            }
        }
    }
    Ok(final_metrics)
}

/// What a held-out pass measured.
#[derive(Clone, Debug)]
pub struct DecisionEval {
    /// The mean served-readout NLL at temperature 1, which early stopping
    /// compares. Causal training also penalizes vocabulary mass outside labels.
    pub loss: f64,
    pub metrics: Metrics,
    /// The raw scores and targets, for fitting the temperatures.
    pub samples: Vec<Sample>,
}

impl DecisionEval {
    /// The series a held-out pass publishes, overall and per question type.
    pub fn values(&self) -> Vec<MetricValue> {
        let mut values = Vec::new();
        let mut push = |name: String, value: f64| {
            values.push(MetricValue {
                name: name.into(),
                value: value as f32,
            });
        };
        let overall = &self.metrics.overall;
        push("eval/decision_accuracy".into(), overall.accuracy);
        push("eval/decision_brier".into(), overall.brier);
        push("eval/decision_ece".into(), overall.ece);
        for (kind, summary) in &self.metrics.per_type {
            push(format!("eval/decision_accuracy_{kind}"), summary.accuracy);
            push(format!("eval/decision_nll_{kind}"), summary.nll);
        }
        values
    }
}

/// Scores the held-out questions under the current weights and measures them
/// at temperature 1. One question per forward pass, as the server evaluates
/// it by default.
pub fn evaluate(trainer: &mut Trainer, prepared: &PreparedDecision) -> Result<DecisionEval> {
    if prepared.eval.is_empty() {
        return Err(Error::invalid(
            "this decision run has no held-out questions to evaluate",
        ));
    }
    let passes: Vec<DecisionPass> = prepared.eval.iter().map(|item| item.pass.clone()).collect();
    let scores = trainer.decide_passes(&prepared.info, &passes, 1)?;
    let mut outcomes = Vec::with_capacity(scores.len());
    let mut samples = Vec::with_capacity(scores.len());
    for (item, scores) in prepared.eval.iter().zip(scores) {
        let scores: Vec<f64> = scores.into_iter().map(f64::from).collect();
        outcomes.push(Outcome::new(
            item.kind,
            softmax(&scores, 1.0),
            item.target.clone(),
        )?);
        samples.push(Sample::new(
            &prepared.info.readout,
            item.kind,
            item.n_options,
            scores,
            item.target.clone(),
        )?);
    }
    let metrics = Metrics::of(&outcomes);
    Ok(DecisionEval {
        loss: metrics.overall.nll,
        metrics,
        samples,
    })
}

/// The temperatures fitted on a held-out pass: one per `<type>.<bucket>`
/// seen, and one per `<type>` over all its buckets, the fallback of a bucket
/// with no sample.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedTemperatures {
    /// By key (`choice`, `choice.3_5`, ...).
    pub temperatures: BTreeMap<String, f32>,
    /// Mean NLL of the held-out set at T = 1, and with the fitted temperatures.
    pub nll_before: f64,
    pub nll_after: f64,
}

pub fn fit_temperatures(eval: &DecisionEval) -> Result<FittedTemperatures> {
    let buckets = calibrate::fit_all(&eval.samples)?;
    let types = calibrate::fit_types(&eval.samples)?;
    let mut temperatures = BTreeMap::new();
    let mut weighted_before = 0.0;
    let mut weighted_after = 0.0;
    let mut n = 0_usize;
    for (key, fit) in &buckets {
        weighted_before += fit.nll_before * fit.n_samples as f64;
        weighted_after += fit.nll_after * fit.n_samples as f64;
        n += fit.n_samples;
        // Narrowing: a temperature in [0.01, 100].
        temperatures.insert(key.clone(), fit.temperature as f32);
    }
    for (key, fit) in types {
        temperatures.insert(key, fit.temperature as f32);
    }
    Ok(FittedTemperatures {
        temperatures,
        nll_before: weighted_before / n.max(1) as f64,
        nll_after: weighted_after / n.max(1) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(tokens: usize) -> Rendered {
        Rendered {
            pass: DecisionPass {
                tokens: vec![0; tokens],
                reads: vec![0, 1],
                column: 0,
                pointer: -1,
            },
            targets: vec![1.0, 0.0],
        }
    }

    #[test]
    fn causal_targets_are_only_on_the_last_prompt_position() {
        let mut hard = rendered(3);
        hard.pass.reads = vec![65, 66];
        let mut soft = rendered(2);
        soft.pass.reads = vec![65, 66];
        soft.targets = vec![0.25, 0.75];
        let batch = causal_batch(&[hard.clone(), soft], 8).unwrap();
        assert_eq!(batch.n_topk, 2);
        assert_eq!(batch.n_sequences, 2);
        assert_eq!(&batch.labels[4..6], &[65, -1]);
        assert_eq!(&batch.labels[8..10], &[65, 66]);
        assert_eq!(&batch.weights[8..10], &[0.25, 0.75]);
        for position in [0, 1, 3, 5, 6, 7] {
            assert_eq!(&batch.labels[position * 2..position * 2 + 2], &[-1, -1]);
        }
        assert_eq!(batch.positions, [0, 1, 2, 0, 1, 2, 3, 4]);
        assert_eq!(causal_batch(&[hard], 4).unwrap().n_topk, 1);
    }

    #[test]
    fn all_letter_probabilities_fit_one_sparse_target() {
        let mut question = rendered(2);
        question.pass.reads = (0..52).collect();
        question.targets = vec![1.0 / 52.0; 52];
        let batch = causal_batch(&[question], 4).unwrap();
        assert_eq!(batch.n_topk, 52);
        assert_eq!(&batch.labels[52..104], &(0..52).collect::<Vec<_>>());
    }

    #[test]
    fn updates_hold_whole_questions_within_both_bounds() {
        let questions: Vec<Rendered> = [100, 200, 300, 50, 50, 50, 500]
            .into_iter()
            .map(rendered)
            .collect();
        // 512 tokens or 3 questions per micro-batch, 2 micro-batches per update.
        let updates = plan_updates(&questions, 512, 3, 2);
        assert_eq!(updates, vec![vec![0..2, 2..5], vec![5..6, 6..7]]);
        let single = plan_updates(&questions, 512, 3, 3);
        assert_eq!(single.len(), 2);
        assert_eq!(single[1], vec![6..7]);
    }

    #[test]
    fn only_choices_are_shuffled_and_targets_follow_their_options() {
        let text = r#"{"state": "s", "question": {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y", "c": "z", "d": "w"}}, "answer": "c"}
{"state": "s", "question": {"type": "score", "instructions": "i", "criteria": ["low", "mid", "high"]}, "answer": 2}"#;
        let examples = dataset::parse_jsonl(text, "data.jsonl", dataset::DataFormat::Native)
            .expect("examples");
        let (question, targets) = shuffled_question(&examples[0], 7);
        let right = targets.iter().position(|&p| p == 1.0).expect("a target");
        assert_eq!(question.options[right].key, "c");
        let mut keys: Vec<&str> = question.options.iter().map(|o| o.key.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["a", "b", "c", "d"]);
        let (score, score_targets) = shuffled_question(&examples[1], 7);
        assert_eq!(score, examples[1].question);
        assert_eq!(score_targets, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn option_seeds_differ_across_epochs_and_examples() {
        let seeds = [
            option_seed(42, 0, 0),
            option_seed(42, 0, 1),
            option_seed(42, 1, 0),
            option_seed(43, 0, 0),
        ];
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
