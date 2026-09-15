//! Held-out measurement of a distillation run: how far the student's policy is
//! from the teacher's, on prompts the loop never trained on.
//!
//! What is measured, and why it is not the loss. The weighted loss of the update
//! loop has no interpretable unit - it is a surrogate whose scale depends on the
//! advantage clamp and on the generation budget. The quantity below is in nats
//! per token and it is the divergence itself:
//!
//! ```text
//! KL(pi_S || p_T) = E_{y ~ pi_S} [ log pi_S(y) - log p_T(y) ]
//! ```
//!
//! Sampling `y` from the student and averaging `log pi_S(y_t) - log p_T(y_t)`
//! is an unbiased Monte-Carlo estimate of that expectation. This is worth being
//! precise about, because the same difference is also described as a *biased*
//! estimator: the bias there is in the **gradient** the single sample carries,
//! not in the value it estimates. Measuring is exactly the case where the
//! estimator is sound, which is why the divergence can be reported long before
//! it can be optimized without bias.
//!
//! Two departures from the loop, both deliberate:
//!
//! - **No clamp.** `weight_clip` bounds what one token may contribute to an
//!   update; a measurement that inherited it would report the bound rather than
//!   the divergence, and would stop moving exactly when the run got worse.
//! - **No truncation masking.** A held-out set is not the run's to filter. A
//!   completion cut at the budget is still a sample of the policy, and dropping
//!   it would make two evaluations under different budgets incomparable - which
//!   is the one thing `[evaluation]` exists for.

use std::path::Path;

use retrograd_config::DistillConfig;
use retrograd_core::{Error, Result, SamplingParams, TrainConfig};
use retrograd_engine::Trainer;

use super::teacher::SharedTeacher;
use crate::RewardEvalMetrics;
use crate::rollout::{
    Prompt, RowLayout, evenly_spaced_subset, generation_room, generation_wave, read_eval_prompts,
    tokenize_prompt,
};

/// Generates one completion per held-out prompt and reports the student's
/// divergence from the teacher over the tokens it just produced.
///
/// The scalar the run controller compares is `-KL`, so that "larger is better"
/// stays true across every algorithm and `EvalDirection::Higher` keeps meaning
/// what it says. The series the run publishes are named for that sign; the
/// divergence itself is `-mean_reward` and never leaves this function positive
/// by accident.
///
/// `max_examples` caps the cost of one evaluation, and the retained prompts are
/// evenly spaced across the dataset - the same subset every time - so a capped
/// pass still covers the file's spread instead of its head.
pub fn evaluate(
    trainer: &mut Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    teacher: &SharedTeacher,
    data: &Path,
    max_examples: Option<usize>,
) -> Result<RewardEvalMetrics> {
    let prompts = held_out_prompts(data, |prompts| match max_examples {
        Some(max) => evenly_spaced_subset(prompts, max),
        None => prompts,
    })?;
    let pass = HeldOutPass::sample(trainer, config, training, &prompts)?;
    let mut teacher = teacher.get_or_open(&config.teacher_path, training)?;
    teacher.compatibility(trainer)?;

    let mut divergences = Vec::with_capacity(pass.rows.len());
    for row in &pass.rows {
        divergences.push(row.divergence(trainer, &mut teacher)?);
    }
    if divergences.is_empty() {
        return Err(empty_completions(data));
    }
    // Each prompt weighs the same, whatever its completion length: `examples`
    // counts prompts, and a token-weighted mean under a prompt-counted n would
    // be two different denominators in one report.
    let mean_kl =
        divergences.iter().map(|&kl| f64::from(kl)).sum::<f64>() / divergences.len() as f64;
    let worst = divergences
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let best = divergences.iter().copied().fold(f32::INFINITY, f32::min);
    Ok(RewardEvalMetrics {
        // Negated once, here. Everything downstream - the controller's
        // `EvalDirection::Higher`, the best-checkpoint rule, the early stop -
        // reads a scalar that is larger when the run is better, and a
        // divergence is not.
        mean_reward: -(mean_kl as f32),
        reward_min: -worst,
        reward_max: -best,
        examples: divergences.len(),
        budget_clamped_fraction: pass.budget_clamped_fraction,
    })
}

/// The three figures of a distillation bench, measured on one held-out set.
///
/// Kept apart from [`evaluate`] rather than folded into it because they answer
/// different questions and cost differently. A scheduled evaluation runs inside
/// a training loop and reports one comparable scalar; a bench runs once, is
/// allowed to be slower, and exists to be *read* - so it keeps the per-prompt
/// distribution and adds the two figures that need a second pair of forward
/// passes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DistillBench {
    /// Per-prompt `KL(pi_S || p_T)` in nats per token, one entry per prompt the
    /// student produced a completion for. Retained rather than aggregated: a
    /// mean hides which prompts moved, and comparing base against adapter is
    /// the whole point of the pass.
    pub teacher_kl: Vec<f32>,
    /// Positions where the two models' argmax agree, over positions compared.
    /// Exact - it reads both truncated distributions, not a sample - and it is
    /// the figure a divergence cannot supply: two models can disagree on every
    /// token they would emit while assigning near-identical probability to the
    /// ones that were drawn.
    pub top1_agreed: usize,
    pub top1_positions: usize,
    /// Teacher-forced negative log-likelihood of the reference answers, in nats,
    /// and the number of tokens it covers. `None` when no line of the dataset
    /// carried one - this is the only figure of the three that needs a written
    /// answer rather than the student's own completion.
    pub reference_nll: Option<(f64, usize)>,
    /// Held-out lines that carried a reference answer, out of `teacher_kl.len()`
    /// scored ones.
    pub reference_examples: usize,
    /// Share of held-out prompts generated under a shortened budget.
    pub budget_clamped_fraction: f32,
}

impl DistillBench {
    /// Share of compared positions where the student and the teacher would emit
    /// the same token. `None` when nothing was compared.
    pub fn top1_agreement(&self) -> Option<f32> {
        (self.top1_positions > 0).then(|| self.top1_agreed as f32 / self.top1_positions as f32)
    }

    /// Teacher-forced perplexity of the student on the reference answers.
    pub fn reference_perplexity(&self) -> Option<f64> {
        self.reference_nll
            .filter(|&(_, tokens)| tokens > 0)
            .map(|(nll, tokens)| (nll / tokens as f64).exp())
    }

    /// Mean per-token divergence over the scored prompts, in nats.
    pub fn teacher_kl_mean(&self) -> Option<f32> {
        if self.teacher_kl.is_empty() {
            return None;
        }
        let sum = self.teacher_kl.iter().map(|&kl| f64::from(kl)).sum::<f64>();
        Some((sum / self.teacher_kl.len() as f64) as f32)
    }
}

/// Runs a distillation bench over `data`, keeping the leading `limit` prompts -
/// the same contract `bench` gives every other algorithm.
pub fn benchmark(
    trainer: &mut Trainer,
    config: &DistillConfig,
    training: &TrainConfig,
    teacher: &SharedTeacher,
    data: &Path,
    limit: Option<usize>,
) -> Result<DistillBench> {
    let prompts = held_out_prompts(data, |mut prompts| {
        if let Some(limit) = limit {
            prompts.truncate(limit);
        }
        prompts
    })?;
    let references = prompts
        .iter()
        .map(|prompt| prompt.reference().map(str::to_owned))
        .collect::<Vec<_>>();
    let pass = HeldOutPass::sample(trainer, config, training, &prompts)?;
    let mut teacher = teacher.get_or_open(&config.teacher_path, training)?;
    teacher.compatibility(trainer)?;

    let mut bench = DistillBench {
        budget_clamped_fraction: pass.budget_clamped_fraction,
        ..DistillBench::default()
    };
    for row in &pass.rows {
        bench
            .teacher_kl
            .push(row.divergence(trainer, &mut teacher)?);
        let (agreed, positions) =
            top1_agreement(trainer, &mut teacher, &row.sequence, row.n_prompt)?;
        bench.top1_agreed += agreed;
        bench.top1_positions += positions;
    }
    if bench.teacher_kl.is_empty() {
        return Err(empty_completions(data));
    }

    // The third figure, and the only one that is not about the teacher: how well
    // the student predicts a written answer. It is what makes a distillation
    // bench comparable to an SFT one, which is the comparison the bench exists for.
    //
    // The reference is tokenized on its own and appended to the prompt row,
    // which is exactly the shape a sampled completion has - the row already ends
    // on the assistant header. Tokenizing the whole conversation instead would
    // let a leading-space merge across that boundary shift the ids, and the two
    // passes would then be measuring two different token sequences.
    let mut nll = 0.0_f64;
    let mut reference_tokens = 0_usize;
    for (row, reference) in pass.rows.iter().zip(&references) {
        let Some(reference) = reference else {
            continue;
        };
        let answer = trainer.tokenize_text(reference)?;
        if answer.is_empty() {
            continue;
        }
        let mut sequence = row.sequence[..row.n_prompt].to_vec();
        sequence.extend_from_slice(&answer);
        if sequence.len() > training.n_ctx as usize {
            continue;
        }
        for logprob in trainer.score_token_suffix(&sequence, row.n_prompt)? {
            if !logprob.is_finite() {
                return Err(Error::runtime(
                    "distillation bench scored a non-finite log-probability on a reference answer",
                ));
            }
            nll -= f64::from(logprob);
            reference_tokens += 1;
        }
        bench.reference_examples += 1;
    }
    if reference_tokens > 0 {
        bench.reference_nll = Some((nll, reference_tokens));
    }
    Ok(bench)
}

/// One held-out prompt the student answered: the prompt with its completion
/// appended, and where the completion starts.
struct HeldOutRow {
    sequence: Vec<i32>,
    n_prompt: usize,
}

impl HeldOutRow {
    /// Mean per-token `log pi_S - log p_T` over this completion.
    ///
    /// The same call on both sides. `llama_decode` is not invariant by batch, so
    /// scoring the student in a group and the teacher alone would fold the
    /// difference between two tile arrangements into the difference between two
    /// models.
    fn divergence(&self, trainer: &mut Trainer, teacher: &mut super::Teacher) -> Result<f32> {
        let student = trainer.score_token_suffix(&self.sequence, self.n_prompt)?;
        let teacher = teacher.logprobs_suffix(&self.sequence, self.n_prompt)?;
        prompt_divergence(&student, &teacher)
    }
}

/// One generation pass over a held-out set: the student's own completions, and
/// what the budget had to give up to fit them.
struct HeldOutPass {
    rows: Vec<HeldOutRow>,
    budget_clamped_fraction: f32,
}

impl HeldOutPass {
    fn sample(
        trainer: &mut Trainer,
        config: &DistillConfig,
        training: &TrainConfig,
        prompts: &[Prompt],
    ) -> Result<Self> {
        let layout = RowLayout::resolve(trainer, training)?;
        let prompt_rows = prompts
            .iter()
            .map(|prompt| tokenize_prompt(trainer, prompt, &layout))
            .collect::<Result<Vec<_>>>()?;
        let wave = generation_wave(training, &layout);
        let mut rows = Vec::with_capacity(prompt_rows.len());
        // Training refuses a prompt that cannot hold the whole budget, naming
        // its JSONL line; a held-out prompt is not the run's to reject, so it is
        // generated under a shorter budget - and counted, so a mean never
        // silently mixes two budgets.
        let mut budget_clamped = 0_usize;
        for (chunk_index, chunk) in prompt_rows.chunks(wave).enumerate() {
            let mut requests = Vec::with_capacity(chunk.len());
            for (member, tokens) in chunk.iter().enumerate() {
                let index = chunk_index * wave + member;
                let room = generation_room(&layout, tokens.len())?;
                if room < config.sampling.max_new_tokens {
                    budget_clamped += 1;
                }
                requests.push((
                    tokens.as_slice(),
                    SamplingParams {
                        // Seeds derive from the prompt's index in the dataset,
                        // not from the update, so two passes over a run compare
                        // two policies rather than two seed schedules.
                        seed: config.sampling.seed.wrapping_add(index as u32),
                        max_new_tokens: config.sampling.max_new_tokens.min(room),
                        ..config.sampling
                    },
                ));
            }
            for (tokens, completion) in chunk
                .iter()
                .zip(trainer.generate_tokens_continuous(&requests)?)
            {
                // A policy that emits EOS immediately produced no token to
                // score. Skipped rather than counted as zero divergence:
                // agreeing with the teacher about nothing is not agreement.
                if completion.is_empty() {
                    continue;
                }
                let n_prompt = tokens.len();
                let mut sequence = tokens.clone();
                sequence.extend_from_slice(&completion);
                rows.push(HeldOutRow { sequence, n_prompt });
            }
        }
        Ok(Self {
            rows,
            budget_clamped_fraction: if prompt_rows.is_empty() {
                0.0
            } else {
                budget_clamped as f32 / prompt_rows.len() as f32
            },
        })
    }
}

/// Reads a held-out prompt file and applies the caller's subsetting rule.
/// An empty file is refused here rather than producing a mean over nothing.
fn held_out_prompts(
    data: &Path,
    subset: impl FnOnce(Vec<Prompt>) -> Vec<Prompt>,
) -> Result<Vec<Prompt>> {
    let prompts = subset(read_eval_prompts(data)?);
    if prompts.is_empty() {
        return Err(Error::invalid(format!(
            "the distillation evaluation dataset {} holds no prompts",
            data.display()
        )));
    }
    Ok(prompts)
}

fn empty_completions(data: &Path) -> Error {
    Error::invalid(format!(
        "every completion generated for {} was empty, so there was nothing to compare against \
         the teacher",
        data.display()
    ))
}

/// Mean per-token `log pi_S - log p_T` over one completion, in nats.
fn prompt_divergence(student: &[f32], teacher: &[f32]) -> Result<f32> {
    if student.len() != teacher.len() || student.is_empty() {
        return Err(Error::runtime(format!(
            "the teacher scored {} tokens of a completion the student scored {} of",
            teacher.len(),
            student.len()
        )));
    }
    let mut total = 0.0_f64;
    for (index, (&student, &teacher)) in student.iter().zip(teacher).enumerate() {
        if !student.is_finite() || !teacher.is_finite() {
            return Err(Error::runtime(format!(
                "distillation evaluation scored a non-finite log-probability at token {index}"
            )));
        }
        total += f64::from(student) - f64::from(teacher);
    }
    Ok((total / student.len() as f64) as f32)
}

/// Top-1 agreement between the student and the teacher on the same
/// teacher-forced sequence: the share of completion positions where the two
/// models would greedily emit the same token.
///
/// Exact, and it is what `teacher_kl` is not - a divergence estimated from one
/// sample says how far apart the two distributions are on the tokens that were
/// drawn, while this says whether they would *act* the same. A run can improve
/// one without the other, and reading only the first is how a student that
/// matches the teacher's confidence while picking different tokens looks
/// aligned.
pub fn top1_agreement(
    trainer: &mut Trainer,
    teacher: &mut super::Teacher,
    tokens: &[i32],
    n_prompt: usize,
) -> Result<(usize, usize)> {
    let student = trainer.top_logprobs_suffix(tokens, n_prompt, 1)?;
    let teacher = teacher.top_logprobs_suffix(tokens, n_prompt, 1)?;
    if student.rows() != teacher.rows() {
        return Err(Error::runtime(
            "the two models returned different numbers of scored positions",
        ));
    }
    let agreed = student
        .argmax()
        .zip(teacher.argmax())
        .filter(|((student, _), (teacher, _))| student == teacher)
        .count();
    Ok((agreed, student.rows()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_teacher_that_agrees_everywhere_leaves_no_divergence() {
        let logprobs = [-0.5, -2.25, -7.0];
        assert_eq!(
            prompt_divergence(&logprobs, &logprobs).expect("divergence"),
            0.0
        );
    }

    #[test]
    fn the_divergence_is_signed_and_unclamped() {
        // A student far more confident than the teacher: positive, and past any
        // `weight_clip` the loop would have applied.
        let kl = prompt_divergence(&[-0.1, -0.1], &[-40.0, -20.0]).expect("divergence");
        assert!((kl - 29.9).abs() < 1e-3, "{kl}");
    }

    #[test]
    fn a_length_mismatch_is_a_runtime_failure_rather_than_a_truncation() {
        assert!(prompt_divergence(&[-1.0, -2.0], &[-1.0]).is_err());
        assert!(prompt_divergence(&[], &[]).is_err());
    }

    #[test]
    fn a_non_finite_score_is_refused_by_name() {
        let error =
            prompt_divergence(&[-1.0, f32::NEG_INFINITY], &[-1.0, -2.0]).expect_err("non-finite");
        assert!(error.to_string().contains("token 1"), "{error}");
    }
}
