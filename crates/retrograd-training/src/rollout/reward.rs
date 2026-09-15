//! External reward scoring: what a reward line means, and held-out evaluation
//! of a rollout policy through it.
//!
//! The transport - spawning the command, the handshake, the deadline, one
//! response line per request - is [`RewardProcess`], shared with the judge's
//! `command` backend. What is here is the schema those lines carry and the
//! refusals only this side can state: a reward that is not finite, a
//! `judge_weight` on a loop that has no judge to weigh.

use std::path::Path;

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result, RewardProtocol, SamplingParams, TrainConfig};
use retrograd_engine::Trainer;
use retrograd_judge::RewardProcess;

use super::prompts::{Prompt, read_eval_prompts, tokenize_prompt};
use super::sampling::{RowLayout, generation_room, generation_wave};
use super::weights::mean_std;
use crate::RewardEvalMetrics;

#[derive(Serialize)]
struct RewardRequest<'a> {
    prompt: &'a str,
    completion: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference: Option<&'a str>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RewardResponse {
    reward: f32,
    /// How much of a judge's verdict this completion's reward is to carry,
    /// `reward + judge_weight * verdict`. Absent means the `[grpo].judge_weight`
    /// of the document.
    ///
    /// Per line, and not per run, because the reward process is the only party
    /// that knows whether a verdict has anything left to decide: a completion it
    /// matched exactly, or one it rejected on format, is already ranked, and
    /// paying a judge to rank it again only adds noise to a settled answer.
    #[serde(default)]
    judge_weight: Option<f32>,
}

/// The reward process of a run: one command, and - in the persistent mode,
/// one worker kept alive across every batch it is asked for.
///
/// Built once per loop and once per evaluation pass rather than once per call:
/// that ownership *is* the amortization, since a GRPO update scores one batch
/// per sampling wave.
pub(crate) fn reward_process(
    command: &[String],
    protocol: RewardProtocol,
) -> Result<RewardProcess> {
    Ok(RewardProcess::new(command, protocol)?)
}

/// One scored rollout: what the reward process returned, before a judge.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RewardRow {
    pub(crate) reward: f32,
    pub(crate) judge_weight: Option<f32>,
}

/// Evaluates a rollout policy on prompts from a held-out JSONL dataset.
/// Sampling seeds are stable across evaluations so changes reflect the policy
/// rather than a changing seed schedule. `max_examples` caps the cost of one
/// evaluation: the prompts are then taken evenly spaced across the dataset
/// (the same subset every time) instead of just the leading ones, so the
/// sample keeps the dataset's spread.
pub(crate) fn evaluate_rewards(
    trainer: &mut Trainer,
    path: &Path,
    process: &mut RewardProcess,
    sampling: &SamplingParams,
    training: &TrainConfig,
    max_examples: Option<usize>,
) -> Result<RewardEvalMetrics> {
    let (rewards, budget_clamped_fraction) =
        evaluate_reward_values_spread(trainer, path, process, sampling, training, max_examples)?;
    let (mean, _, examples) = mean_std(rewards.iter().copied());
    Ok(RewardEvalMetrics {
        mean_reward: mean as f32,
        reward_min: rewards.iter().copied().fold(f32::INFINITY, f32::min),
        reward_max: rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        examples,
        budget_clamped_fraction,
    })
}

/// Generates and scores held-out prompts, retaining each reward for the
/// standalone bench report. Training evaluation uses the aggregate wrapper
/// above; both paths therefore exercise exactly the same policy objective.
/// `limit` keeps the leading prompts (the bench CLI contract); training
/// evaluation goes through [`evaluate_reward_values_spread`] instead.
pub(crate) fn evaluate_reward_values(
    trainer: &mut Trainer,
    path: &Path,
    process: &mut RewardProcess,
    sampling: &SamplingParams,
    training: &TrainConfig,
    limit: Option<usize>,
) -> Result<Vec<f32>> {
    let mut prompts = read_eval_prompts(path)?;
    if let Some(limit) = limit {
        prompts.truncate(limit);
    }
    Ok(evaluate_prompt_rewards(trainer, prompts, process, sampling, training)?.0)
}

fn evaluate_reward_values_spread(
    trainer: &mut Trainer,
    path: &Path,
    process: &mut RewardProcess,
    sampling: &SamplingParams,
    training: &TrainConfig,
    max_examples: Option<usize>,
) -> Result<(Vec<f32>, f32)> {
    let mut prompts = read_eval_prompts(path)?;
    if let Some(max) = max_examples {
        prompts = evenly_spaced_subset(prompts, max);
    }
    evaluate_prompt_rewards(trainer, prompts, process, sampling, training)
}

/// Deterministically keeps `max` elements evenly spaced across the input, so
/// a capped evaluation still covers the whole dataset rather than its head.
pub(crate) fn evenly_spaced_subset<T>(values: Vec<T>, max: usize) -> Vec<T> {
    if max == 0 || values.len() <= max {
        return values;
    }
    let len = values.len();
    let mut picked = vec![false; len];
    for rank in 0..max {
        // rank * (len - 1) stays well within u64 at dataset sizes; usize on
        // 32-bit targets could overflow, so route through u128 to be exact.
        let index = (rank as u128 * (len - 1) as u128 / (max - 1).max(1) as u128) as usize;
        picked[index] = true;
    }
    values
        .into_iter()
        .zip(picked)
        .filter_map(|(value, keep)| keep.then_some(value))
        .collect()
}

/// Generates one completion per held-out prompt, then scores them all in one
/// reward call.
///
/// The prompts are heterogeneous, so they go through the continuous decode path
/// in waves of `generation_concurrency`: one prompt at a time cost one full
/// prefill per prompt, and prefill - not decode - dominates rollout generation
/// (`docs/engineering/optims/SAMPLING.md`). Seeds still derive from each prompt's index
/// in the dataset, so an evaluation stays comparable across updates; the
/// sampled tokens themselves may differ from the unbatched path, exactly as
/// they do between two different `generation_concurrency` settings during
/// training.
fn evaluate_prompt_rewards(
    trainer: &mut Trainer,
    prompts: Vec<Prompt>,
    process: &mut RewardProcess,
    sampling: &SamplingParams,
    training: &TrainConfig,
) -> Result<(Vec<f32>, f32)> {
    let layout = RowLayout::resolve(trainer, training)?;
    let prompt_rows = prompts
        .iter()
        .map(|prompt| tokenize_prompt(trainer, prompt, &layout))
        .collect::<Result<Vec<_>>>()?;

    let wave = generation_wave(training, &layout);
    let mut completions = Vec::with_capacity(prompts.len());

    // Training rejects a prompt that cannot hold the whole budget, naming its
    // JSONL line; an evaluation dataset is held out and is not the run's to
    // reject, so a long prompt is generated under a shorter budget - and
    // counted, so `eval/mean_reward` never silently mixes two budgets.
    let mut budget_clamped = 0_usize;
    for (chunk_index, chunk) in prompt_rows.chunks(wave).enumerate() {
        let mut requests = Vec::with_capacity(chunk.len());
        for (member, tokens) in chunk.iter().enumerate() {
            let index = chunk_index * wave + member;
            let room = generation_room(&layout, tokens.len())?;
            if room < sampling.max_new_tokens {
                budget_clamped += 1;
            }
            requests.push((
                tokens.as_slice(),
                SamplingParams {
                    max_new_tokens: sampling.max_new_tokens.min(room),
                    seed: sampling.seed.wrapping_add(index as u32),
                    ..*sampling
                },
            ));
        }
        for completion in trainer.generate_tokens_continuous(&requests)? {
            completions.push(trainer.detokenize(&completion, false)?);
        }
    }

    // A `judge_weight` on an evaluation line is dropped, deliberately. A judge
    // ranks the members of a group against each other, and an evaluation
    // generates one completion per prompt: there is no group to rank, and a
    // verdict renormalized inside a group of one would make two passes
    // incomparable - which is the only thing `[evaluation]` is for.
    let rewards = rows_with_references(
        process,
        prompts
            .iter()
            .zip(&completions)
            .map(|(prompt, completion)| {
                (
                    prompt.reward_text(),
                    completion.as_str(),
                    prompt.reference(),
                )
            }),
    )?
    .into_iter()
    .map(|row| row.reward)
    .collect::<Vec<_>>();

    let clamped_fraction = if prompts.is_empty() {
        0.0
    } else {
        budget_clamped as f32 / prompts.len() as f32
    };

    Ok((rewards, clamped_fraction))
}

/// Enforces one finite reward response for every rollout, in input order.
///
/// The scalar form, for the algorithms that have no judge to blend a verdict
/// into: a `judge_weight` is refused here rather than ignored, since a reward
/// process that asks for one is written against a loop that would have used it.
pub(crate) fn score<'a>(
    process: &mut RewardProcess,
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<Vec<f32>> {
    score_rows(process, pairs)?
        .into_iter()
        .enumerate()
        .map(|(index, row)| match row.judge_weight {
            None => Ok(row.reward),
            Some(_) => Err(Error::invalid(format!(
                "reward response {} carries a judge_weight, which only [grpo.judge] honours",
                index + 1
            ))),
        })
        .collect()
}

/// The protocol as written, for the GRPO loop, which does have a judge.
pub(crate) fn score_rows<'a>(
    process: &mut RewardProcess,
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<Vec<RewardRow>> {
    rows_with_references(
        process,
        pairs
            .into_iter()
            .map(|(prompt, completion)| (prompt, completion, None)),
    )
}

pub(super) fn rows_with_references<'a>(
    process: &mut RewardProcess,
    pairs: impl IntoIterator<Item = (&'a str, &'a str, Option<&'a str>)>,
) -> Result<Vec<RewardRow>> {
    let responses: Vec<RewardResponse> = process.call(pairs.into_iter().map(
        |(prompt, completion, reference)| RewardRequest {
            prompt,
            completion,
            reference,
        },
    ))?;
    responses
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            if !row.reward.is_finite() {
                return Err(Error::invalid(format!(
                    "reward response {} is not finite",
                    index + 1
                )));
            }
            if row
                .judge_weight
                .is_some_and(|weight| !weight.is_finite() || weight < 0.0)
            {
                return Err(Error::invalid(format!(
                    "reward response {} has a judge_weight that is not finite and non-negative",
                    index + 1
                )));
            }
            Ok(RewardRow {
                reward: row.reward,
                judge_weight: row.judge_weight,
            })
        })
        .collect()
}
