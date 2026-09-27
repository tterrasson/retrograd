//! Offline preference optimization: DPO, IPO, SimPO and ORPO over pairs of
//! responses to one prompt.
//!
//! Nothing here generates and nothing needs a new kernel. Every loss depends on
//! the parameters only through each response's summed log-probability, so its
//! gradient is one detached coefficient per response times the gradient of
//! that response's tokens - exactly what the weighted objective PPO and GRPO
//! already train through ([`loss`] has the derivation). One step is therefore
//! a forward-only scoring pass under the current parameters, the coefficients
//! in `f64`, and one weighted step over the chunk's sequences.
//!
//! One responsibility per file: [`loss`] the objectives, [`reference`] the
//! frozen scores and their cache, `plan` which pairs share a step, and
//! [`evaluate`] the held-out pass.

mod evaluate;
pub(crate) mod loss;
mod plan;
pub mod reference;

use std::ops::Range;
use std::path::Path;

use retrograd_checkpoint as checkpoint;
use retrograd_config::{PreferenceConfig, PreferenceLoss, ReferenceSource};
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_dataset::{PreparedSequence, prepare_preference_jsonl};
use retrograd_engine::Trainer;
use retrograd_metrics::MetricValue;

use crate::rollout::step::{
    PackingSelection, WeightedMember, select_packing, train_weighted_members,
};
use crate::rollout::{Rollout, RowLayout, WeightedStepScratch, evenly_spaced_subset};
use crate::{Boundary, Progress};

pub use evaluate::{PreferenceEval, evaluate};
pub use reference::{
    CacheHeader, ReferenceCache, ReferenceCacheError, ReferenceTable, compute as compute_reference,
};

use loss::{PairTerms, SideScores, batch_weight, pair_terms};

/// One prepared pair, as the two rollouts the weighted step trains. Both start
/// with the same `prompt_len` tokens.
#[derive(Clone, Debug)]
pub(crate) struct Pair {
    pub(crate) prompt_len: usize,
    pub(crate) chosen: Rollout,
    pub(crate) rejected: Rollout,
}

impl Pair {
    fn new(prompt_len: usize, chosen: PreparedSequence, rejected: PreparedSequence) -> Self {
        let rollout = |sequence: PreparedSequence| Rollout {
            tokens: sequence.tokens,
            train_mask: sequence.train_mask,
            old_logprobs: Vec::new(),
        };
        Self {
            prompt_len,
            chosen: rollout(chosen),
            rejected: rollout(rejected),
        }
    }

    fn sides(&self) -> [&Rollout; 2] {
        [&self.chosen, &self.rejected]
    }
}

/// The two responses of a pair scored under the trainer's current policy, in
/// one forward pass over their shared prompt: the summed log-probability of
/// each response's trained tokens.
pub(crate) fn score_pair(trainer: &mut Trainer, pair: &Pair) -> Result<[f64; 2]> {
    let scores = trainer.score_token_suffix_batch(&[
        (&pair.chosen.tokens, pair.prompt_len),
        (&pair.rejected.tokens, pair.prompt_len),
    ])?;
    let mut sums = [0.0; 2];
    for ((sum, rollout), scores) in sums.iter_mut().zip(pair.sides()).zip(&scores) {
        if scores.len() != rollout.tokens.len() - pair.prompt_len {
            return Err(Error::runtime(
                "suffix scores do not match the preference response length",
            ));
        }
        *sum = rollout.train_mask[pair.prompt_len..]
            .iter()
            .zip(scores)
            .filter(|(train, _)| **train)
            .map(|(_, &score)| f64::from(score))
            .sum();
    }
    Ok(sums)
}

/// A preference dataset ready to train: the pairs, their costs, and the step
/// plan they imply.
#[derive(Debug)]
pub struct PreparedPreference {
    pub(crate) pairs: Vec<Pair>,
    pub(crate) eval: Vec<Pair>,
    /// Physical ubatches each pair costs inside one optimizer window.
    pub(crate) costs: Vec<u64>,
    pub(crate) layout: RowLayout,
    shuffle: bool,
    seed: u32,
    pairs_per_step: Option<usize>,
    epochs: u32,
    total_steps: u64,
    supervised_tokens: usize,
    fingerprint: String,
}

impl PreparedPreference {
    pub fn pairs(&self) -> usize {
        self.pairs.len()
    }

    pub fn eval_pairs(&self) -> usize {
        self.eval.len()
    }

    /// Trained tokens over both responses of every training pair.
    pub fn supervised_tokens(&self) -> usize {
        self.supervised_tokens
    }

    /// The exact number of optimizer steps the configured epochs take.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Width of one training row, the context the pairs were prepared for.
    pub fn row_width(&self) -> usize {
        self.layout.row_width
    }

    /// Fingerprint of the prepared training and evaluation pairs: their
    /// tokens, masks and prompt lengths, in file order.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// The optimizer chunks of one epoch, as ranges of `order`. Each takes
    /// exactly one optimizer step: a chunk of several pairs fits the period,
    /// and a pair over it was only admitted if it trains as one packed pass.
    fn plan_epoch(&self, epoch: u32) -> (Vec<usize>, Vec<Range<usize>>) {
        let order = plan::epoch_order(self.pairs.len(), self.shuffle, self.seed, epoch);
        let period = self.layout.accumulation_period();
        let chunks = plan::chunk_pairs(&order, &self.costs, period, self.pairs_per_step);
        (order, chunks)
    }
}

fn members<'a>(pairs: &[&'a Pair]) -> Vec<WeightedMember<'a>> {
    pairs
        .iter()
        .enumerate()
        .flat_map(|(group, pair)| {
            pair.sides().map(|rollout| WeightedMember {
                group_id: group as u64,
                rollout,
            })
        })
        .collect()
}

fn prepare_pairs(trainer: &Trainer, path: &Path, window: usize) -> Result<Vec<Pair>> {
    Ok(prepare_preference_jsonl(trainer, path, window)?
        .into_iter()
        .map(|pair| Pair::new(pair.prompt_len, pair.chosen, pair.rejected))
        .collect())
}

/// Reads and prepares the training pairs, and the evaluation pairs when
/// `eval` names a file, capped at `max_examples` evenly spaced ones; then lays
/// out every epoch's steps.
pub fn prepare(
    trainer: &mut Trainer,
    config: &PreferenceConfig,
    training: &TrainConfig,
    epochs: u32,
    eval: Option<(&Path, Option<usize>)>,
) -> Result<PreparedPreference> {
    config.validate()?;
    let layout = RowLayout::resolve(trainer, training)?;
    let pairs = prepare_pairs(trainer, &config.data, layout.window)?;
    let eval = match eval {
        Some((path, max_examples)) => {
            let pairs = prepare_pairs(trainer, path, layout.window)?;
            match max_examples {
                Some(max) => evenly_spaced_subset(pairs, max),
                None => pairs,
            }
        }
        None => Vec::new(),
    };
    let costs = pairs
        .iter()
        .map(|pair| Ok(layout.rollout_evals(&pair.chosen)? + layout.rollout_evals(&pair.rejected)?))
        .collect::<Result<Vec<_>>>()?;
    // A pair's coefficients are its loss's gradient only under the parameters
    // that scored it, so both responses must train in the same optimizer step.
    // The runtime closes a step every period of rows, so a pair over the
    // period is trainable only as one packed pass.
    let period = layout.accumulation_period();
    let capability = trainer.supports_shared_prefix_packed_training()?;
    let mut oversized = Vec::new();
    for (index, (pair, &cost)) in pairs.iter().zip(&costs).enumerate() {
        if cost > period
            && !matches!(
                select_packing(&members(&[pair]), &layout, capability)?,
                PackingSelection::Packed(_)
            )
        {
            oversized.push(index);
        }
    }
    if let Some(&first) = oversized.first() {
        return Err(Error::config(format!(
            "{} of {} preference pairs do not fit one optimizer step (the first is pair {}): \
             their two responses together exceed training.ctx = {} tokens and cannot share \
             one packed pass, and a pair split between two steps would not train its loss; \
             raise training.ctx, or allow training.shared_prefix_fanout on a model that \
             supports packed training",
            oversized.len(),
            pairs.len(),
            first + 1,
            layout.window
        )));
    }
    let supervised_tokens = pairs
        .iter()
        .flat_map(Pair::sides)
        .map(Rollout::completion_len)
        .sum();
    let mut fingerprinter = checkpoint::Fingerprinter::new();
    for (set, pairs) in [("train", &pairs), ("eval", &eval)] {
        fingerprinter.update(set.as_bytes());
        for pair in pairs.iter() {
            fingerprinter.update(&(pair.prompt_len as u64).to_le_bytes());
            for rollout in pair.sides() {
                fingerprinter.update(&(rollout.tokens.len() as u64).to_le_bytes());
                for (&token, &train) in rollout.tokens.iter().zip(&rollout.train_mask) {
                    fingerprinter.update(&token.to_le_bytes());
                    fingerprinter.update(&[u8::from(train)]);
                }
            }
        }
    }
    let mut prepared = PreparedPreference {
        pairs,
        eval,
        costs,
        layout,
        shuffle: config.shuffle,
        seed: config.seed,
        // A count of pairs, so a `u32` always fits `usize` on the targets this
        // runs on.
        pairs_per_step: config.pairs_per_step.map(|cap| cap as usize),
        epochs,
        total_steps: 0,
        supervised_tokens,
        fingerprint: fingerprinter.finish(),
    };
    if prepared.pairs.is_empty() {
        return Err(Error::invalid("a preference run needs at least one pair"));
    }
    prepared.total_steps = (0..epochs)
        .map(|epoch| prepared.plan_epoch(epoch).1.len() as u64)
        .fold(0_u64, u64::saturating_add);
    Ok(prepared)
}

/// What the pairs of one step, or of one epoch, add up to.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    pairs: usize,
    steps: u64,
    loss: f64,
    margin: f64,
    correct: usize,
    chosen_reward: f64,
    rejected_reward: f64,
    chosen_logps: f64,
    rejected_logps: f64,
    nll: f64,
    log_odds: f64,
    clamped: u64,
}

impl Totals {
    fn add_pair(&mut self, terms: &PairTerms, chosen_mean: f64, rejected_mean: f64) {
        self.pairs += 1;
        self.loss += terms.loss;
        let margin = terms.chosen_reward - terms.rejected_reward;
        self.margin += margin;
        self.correct += usize::from(margin > 0.0);
        self.chosen_reward += terms.chosen_reward;
        self.rejected_reward += terms.rejected_reward;
        self.chosen_logps += chosen_mean;
        self.rejected_logps += rejected_mean;
        if let Some(orpo) = terms.orpo {
            self.nll += orpo.nll;
            self.log_odds += orpo.log_odds;
            self.clamped += u64::from(orpo.clamped);
        }
    }

    fn merge(&mut self, other: &Totals) {
        self.pairs += other.pairs;
        self.steps += other.steps;
        self.loss += other.loss;
        self.margin += other.margin;
        self.correct += other.correct;
        self.chosen_reward += other.chosen_reward;
        self.rejected_reward += other.rejected_reward;
        self.chosen_logps += other.chosen_logps;
        self.rejected_logps += other.rejected_logps;
        self.nll += other.nll;
        self.log_odds += other.log_odds;
        self.clamped += other.clamped;
    }

    fn mean(&self, sum: f64) -> f64 {
        sum / self.pairs.max(1) as f64
    }

    fn loss(&self) -> f64 {
        self.mean(self.loss)
    }

    fn chosen_logps(&self) -> f64 {
        self.mean(self.chosen_logps)
    }

    fn rejected_logps(&self) -> f64 {
        self.mean(self.rejected_logps)
    }

    fn values(&self, loss: &PreferenceLoss) -> Vec<MetricValue> {
        let value = |name: &str, value: f64| MetricValue {
            name: format!("preference/{name}").into(),
            value: value as f32,
        };
        let mut values = vec![
            value("loss", self.loss()),
            value("margin", self.mean(self.margin)),
            value("accuracy", self.mean(self.correct as f64)),
            value("chosen_reward", self.mean(self.chosen_reward)),
            value("rejected_reward", self.mean(self.rejected_reward)),
            value("chosen_logps", self.chosen_logps()),
            value("rejected_logps", self.rejected_logps()),
            value(
                "pairs_per_step",
                self.pairs as f64 / self.steps.max(1) as f64,
            ),
        ];
        if matches!(loss, PreferenceLoss::Orpo { .. }) {
            values.push(value("nll", self.mean(self.nll)));
            values.push(value("log_odds", self.mean(self.log_odds)));
            values.push(value("orpo_clamped", self.clamped as f64));
        }
        values
    }
}

/// Scores every pair under the current policy and turns the pairs' terms into
/// weights: each response's per-token coefficient, divided by the number of
/// pairs, on every trained token.
fn chunk_terms(
    trainer: &mut Trainer,
    pairs: &[(&Pair, Option<[f64; 2]>)],
    loss: &PreferenceLoss,
    scratch: &mut WeightedStepScratch,
) -> Result<Totals> {
    scratch.begin_members(2 * pairs.len());
    let mut totals = Totals::default();
    // The step's loss is the mean over its pairs, so each pair's gradient is
    // scaled by its share. Exact: a chunk holds at most one context of pairs.
    let share = 1.0 / pairs.len() as f64;
    for (index, (pair, reference)) in pairs.iter().enumerate() {
        let current = score_pair(trainer, pair)?;
        let [chosen, rejected] = [0, 1].map(|side| SideScores {
            sum: current[side],
            tokens: pair.sides()[side].completion_len(),
            reference: reference.map(|reference| reference[side]),
        });
        let terms = pair_terms(loss, chosen, rejected)?;
        for (side, (coefficient, rollout)) in [terms.chosen_weight, terms.rejected_weight]
            .into_iter()
            .zip(pair.sides())
            .enumerate()
        {
            let weight = batch_weight(coefficient * share)?;
            let buffer = &mut scratch.member_weights[2 * index + side];
            buffer.clear();
            buffer.resize(rollout.completion_len(), weight);
        }
        totals.add_pair(
            &terms,
            chosen.sum / chosen.tokens as f64,
            rejected.sum / rejected.tokens as f64,
        );
    }
    Ok(totals)
}

/// One optimizer transaction over whole pairs: all of them scored under the
/// same parameters before the step, so the gradient applied is the gradient of
/// the chunk's mean loss.
fn step(
    trainer: &mut Trainer,
    pairs: &[(&Pair, Option<[f64; 2]>)],
    loss: &PreferenceLoss,
    prepared: &PreparedPreference,
    scratch: &mut WeightedStepScratch,
) -> Result<(TrainMetrics, Totals)> {
    let totals = chunk_terms(trainer, pairs, loss, scratch)?;
    let chunk = pairs.iter().map(|(pair, _)| *pair).collect::<Vec<_>>();
    let members = members(&chunk);
    let (metrics, _) = train_weighted_members(
        trainer,
        &members,
        &prepared.layout,
        // The coefficients are the exact gradient of the sum; nothing else
        // divides it.
        1,
        prepared.total_steps,
        scratch,
        // A stop is honoured between steps, where the run has a boundary.
        &mut |_, _| Ok(true),
    )?;
    Ok((metrics, totals))
}

/// The warning a run gives once, when the margin grows because both sides are
/// falling rather than because the chosen one rises.
struct DriftWatch {
    first: Option<(f64, f64)>,
    threshold: f64,
    warned: bool,
}

impl DriftWatch {
    fn observe(&mut self, chosen: f64, rejected: f64) -> Option<String> {
        let (first_chosen, first_rejected) = *self.first.get_or_insert((chosen, rejected));
        let drop = first_chosen - chosen;
        if self.warned || drop <= self.threshold || rejected >= first_rejected {
            return None;
        }
        self.warned = true;
        Some(format!(
            "preference: the chosen responses' mean log-probability fell {drop:.2} nats since \
             the first step, and the rejected ones' fell with it - the margin is growing \
             because both sides are dropping, the known drift of reference-relative losses \
             (warned at preference.logps_drop_warn = {:.2})",
            self.threshold
        ))
    }
}

/// Runs the preference epochs, restarting at the step boundary `resume`
/// restores, if any.
///
/// A boundary is emitted after every step: the order and the chunks of an
/// epoch are derived from the seed and the epoch index, so a cursor of pairs
/// consumed is enough to continue exactly where the run stopped.
pub fn run_resumed(
    trainer: &mut Trainer,
    prepared: &PreparedPreference,
    reference: Option<&ReferenceTable>,
    config: &PreferenceConfig,
    resume: Option<Boundary>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(
        target: "retrograd::training::preference",
        "training",
        loss = config.loss.name()
    );
    let _entered = span.enter();
    if config.loss.uses_reference() != reference.is_some() {
        return Err(Error::invalid(format!(
            "preference loss '{}' {} a reference table",
            config.loss.name(),
            match reference.is_some() {
                true => "reads no",
                false => "needs",
            }
        )));
    }
    if let Some(reference) = reference
        && (reference.train.len() != prepared.pairs.len()
            || reference.eval.len() != prepared.eval.len())
    {
        return Err(Error::invalid(
            "the reference table does not align with the prepared pairs",
        ));
    }
    let (start_epoch, cursor) = match resume {
        Some(boundary) => (
            u32::try_from(boundary.completed_iterations).map_err(|_| {
                Error::overflow("the checkpoint epoch count does not fit in the epoch counter")
            })?,
            usize::try_from(boundary.cursor)
                .map_err(|_| Error::overflow("the checkpoint pair cursor exceeds usize"))?,
        ),
        None => (0, 0),
    };
    let mut scratch = WeightedStepScratch::new(&prepared.layout);
    let mut global_step = trainer.advance_scheduler_steps(0)?;
    let mut drift = DriftWatch {
        first: None,
        threshold: f64::from(config.logps_drop_warn),
        warned: false,
    };
    let mut final_metrics = TrainMetrics::default();
    for epoch in 0..prepared.epochs {
        if epoch < start_epoch {
            continue;
        }
        let (order, chunks) = prepared.plan_epoch(epoch);
        let first = match epoch == start_epoch {
            true => plan::resume_chunk(&chunks, cursor)?,
            false => 0,
        };
        let mut epoch_totals = Totals::default();
        let last_chunk = chunks.len().saturating_sub(1);
        for (index, chunk) in chunks.iter().enumerate().skip(first) {
            let pairs = order[chunk.clone()]
                .iter()
                .map(|&pair| {
                    (
                        &prepared.pairs[pair],
                        reference.map(|table| table.train[pair]),
                    )
                })
                .collect::<Vec<_>>();
            let (mut metrics, mut totals) =
                step(trainer, &pairs, &config.loss, prepared, &mut scratch)?;
            // The plan owns the step count: one per chunk. A row whose weights
            // all rounded to zero is skipped by the runtime, and the schedule
            // still counts it, so the horizon stays the one the run was
            // started with.
            let taken = metrics
                .global_step
                .checked_sub(global_step)
                .ok_or_else(|| Error::runtime("the optimizer step counter moved backwards"))?;
            if taken > 1 {
                return Err(Error::runtime(format!(
                    "a preference step took {taken} optimizer steps where its plan has one"
                )));
            }
            global_step = trainer.advance_scheduler_steps(1 - taken)?;
            totals.steps = 1;
            metrics.global_step = global_step;
            metrics.epoch = epoch + 1;
            metrics.epoch_complete = false;
            metrics.train_loss = totals.loss() as f32;
            metrics.eval_loss = f32::NAN;
            epoch_totals.merge(&totals);

            let mut progress = Progress::sft(metrics, false);
            progress
                .values
                .extend(totals.values(&config.loss));
            progress.notes.append(&mut scratch.notes);
            if let Some(warning) = drift.observe(totals.chosen_logps(), totals.rejected_logps()) {
                progress.notes.push(warning);
            }
            let at_epoch_end = index == last_chunk;
            // The last step of an epoch is followed by the epoch's own boundary,
            // which is where it resumes from.
            progress.boundary = (!at_epoch_end).then_some(Boundary {
                completed_iterations: u64::from(epoch),
                cursor: chunk.end as u64,
                kl_multiplier: None,
            });
            let keep_going = on_progress(trainer, progress)?;
            final_metrics = metrics;
            if !at_epoch_end && !keep_going {
                return Ok(final_metrics);
            }
            if at_epoch_end {
                // A stop asked for on the last step still gets its epoch:
                // evaluated, checkpointed and reported.
                let mut metrics = metrics;
                metrics.epoch_complete = true;
                metrics.train_loss = epoch_totals.loss() as f32;
                final_metrics = metrics;
                let mut progress = Progress::sft(metrics, false);
                progress
                    .values
                    .extend(epoch_totals.values(&config.loss));
                if !on_progress(trainer, progress)? || !keep_going {
                    return Ok(final_metrics);
                }
            }
        }
    }
    Ok(final_metrics)
}

/// Scores the reference `config` names for `prepared`, when its loss has one.
/// Must run before the first step: the initial policy is the trainer as it is.
pub fn reference_table(
    trainer: &mut Trainer,
    prepared: &PreparedPreference,
    config: &PreferenceConfig,
) -> Result<Option<ReferenceTable>> {
    config
        .reference
        .map(|source| reference::compute(trainer, prepared, source))
        .transpose()
}

/// A whole run without a controller: prepare, score the reference, train, and
/// evaluate after every epoch when `eval` names a file. There is no resume on
/// this path, and so no cache.
pub fn run(
    trainer: &mut Trainer,
    config: &PreferenceConfig,
    training: &TrainConfig,
    epochs: u32,
    eval: Option<&Path>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    if config.reference == Some(ReferenceSource::Model)
        && trainer.reference_fingerprint()?.is_empty()
    {
        return Err(Error::invalid(
            "preference reference 'model' needs an anchor attached to the trainer",
        ));
    }
    let prepared = prepare(
        trainer,
        config,
        training,
        epochs,
        eval.map(|path| (path, None)),
    )?;
    let reference = reference_table(trainer, &prepared, config)?;
    let mut last_eval_loss = f32::NAN;
    let mut metrics = run_resumed(
        trainer,
        &prepared,
        reference.as_ref(),
        config,
        None,
        &mut |trainer, mut progress| {
            if progress.metrics.epoch_complete && !prepared.eval.is_empty() {
                let result = evaluate(trainer, &prepared, reference.as_ref(), &config.loss)?;
                last_eval_loss = result.loss as f32;
                progress.metrics.eval_loss = last_eval_loss;
                progress.values.extend(result.values());
            }
            on_progress(progress);
            Ok(true)
        },
    )?;
    metrics.eval_loss = last_eval_loss;
    Ok(metrics)
}
