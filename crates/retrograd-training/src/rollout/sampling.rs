//! Rollout geometry and sampling: the fixed row layout every algorithm packs
//! into, the token-level training mask, and the batched decode paths that turn
//! prompts into rollouts.

use retrograd_core::{Error, Result, SamplingParams, SharedPrefixFanout, TrainConfig};
use retrograd_engine::Trainer;

use super::weights::score_train_mask_group;

/// One sampled rollout: the full context sequence, an explicit token-level
/// training mask, and rollout-time policy logprobs aligned with the `true`
/// positions of that mask.
#[derive(Clone, Debug)]
pub(crate) struct Rollout {
    pub(crate) tokens: Vec<i32>,
    pub(crate) train_mask: Vec<bool>,
    pub(crate) old_logprobs: Vec<f32>,
}

impl crate::TokenSpan for Rollout {
    fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    fn train_mask(&self) -> &[bool] {
        &self.train_mask
    }
}

impl Rollout {
    pub(crate) fn completion_len(&self) -> usize {
        self.train_mask.iter().filter(|&&train| train).count()
    }

    pub(crate) fn first_train_index(&self) -> Result<usize> {
        first_train_index(&self.tokens, &self.train_mask)
    }

    pub(crate) fn last_train_index(&self) -> Result<usize> {
        self.first_train_index()?;
        Ok(self
            .train_mask
            .iter()
            .rposition(|&train| train)
            .expect("first_train_index verified a supervised target"))
    }

    /// Input positions needed to cover every supervised target, including
    /// untrained observation tokens between disjoint policy segments.
    pub(super) fn training_span_len(&self) -> Result<usize> {
        let first = self.first_train_index()?;
        let last = self
            .train_mask
            .iter()
            .rposition(|&train| train)
            .expect("first_train_index verified a supervised target");
        Ok(last - first + 1)
    }
}

/// Fixed geometry of every packed training row for one run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLayout {
    /// Effective llama.cpp context - the padded width of a packed row.
    pub(crate) row_width: usize,
    /// Trained window: prompt + completion must fit here.
    pub(crate) window: usize,
    pub(crate) pad_token: i32,
    pub(crate) batch: usize,
    pub(crate) ubatch: usize,
    pub(crate) n_seq_max: usize,
    pub(crate) shared_prefix_fanout: SharedPrefixFanout,
    /// Optimizer steps one packed row produces.
    pub(crate) steps_per_row: u64,
}

impl RowLayout {
    /// llama.cpp may round the requested context up (`row_width`), but the
    /// optimizer only trains the first `training.n_ctx` positions of each row,
    /// so rollouts must fit that window while rows are padded to full width.
    /// One optimizer step runs per `n_batch` chunk of the trained window;
    /// pinning the full horizon from `steps_per_row` keeps linear/cosine
    /// schedules meaningful across updates.
    pub(crate) fn resolve(trainer: &Trainer, training: &TrainConfig) -> Result<Self> {
        let row_width = trainer.context_size()?;
        let window = (training.n_ctx as usize).min(row_width);
        let steps_per_row = (window / training.n_batch as usize).max(1) as u64;
        Ok(Self {
            row_width,
            window,
            pad_token: trainer.eos_token()?,
            batch: training.n_batch as usize,
            ubatch: training.n_ubatch as usize,
            n_seq_max: training.n_seq_max as usize,
            shared_prefix_fanout: training.shared_prefix_fanout,
            steps_per_row,
        })
    }

    /// Physical ubatches gradient accumulation spans before one optimizer
    /// step: one optimizer window (`training.micro_batch *
    /// training.gradient_accumulation`) of tokens. The runtime carries the
    /// accumulation across row boundaries, so this is also the eval budget
    /// that groups several short rollouts under a single optimizer step.
    pub(crate) fn accumulation_period(&self) -> u64 {
        (self.batch / self.ubatch.max(1)).max(1) as u64
    }

    /// Physical ubatches the runtime actually evaluates for this rollout's
    /// row: everything up to its last supervised label, rounded up to
    /// `ubatch`. The row's remaining padding carries no labels and is skipped
    /// by the runtime's weighted epoch.
    pub(crate) fn rollout_evals(&self, rollout: &Rollout) -> Result<u64> {
        let last_train = rollout.last_train_index()?;
        // The label of target `last_train` sits at row position
        // `last_train - 1`.
        let evals = ((last_train - 1) / self.ubatch.max(1) + 1) as u64;
        let cap = (self.window / self.ubatch.max(1)).max(1) as u64;
        Ok(evals.min(cap))
    }
}

pub(super) fn first_train_index(tokens: &[i32], train_mask: &[bool]) -> Result<usize> {
    if tokens.len() != train_mask.len() {
        return Err(Error::invalid(
            "rollout tokens and train_mask lengths do not match",
        ));
    }
    let first = train_mask
        .iter()
        .position(|&train| train)
        .ok_or_else(|| Error::tokenize("rollout has no trainable token"))?;
    if first == 0 {
        return Err(Error::invalid(
            "the first token of a rollout cannot be trainable",
        ));
    }
    Ok(first)
}

fn make_train_mask(prompt_len: usize, total_len: usize) -> Vec<bool> {
    (0..total_len).map(|index| index >= prompt_len).collect()
}

/// Generation budget left next to a prompt inside the trained window. The
/// callers below validate their prompts up front, so this only fires on a
/// programmatic caller - but a wrapping subtraction there would hand the
/// runtime a nonsense token budget instead of an error.
pub(crate) fn generation_room(layout: &RowLayout, prompt_len: usize) -> Result<u32> {
    match layout.window.checked_sub(prompt_len) {
        Some(room) if room > 0 => Ok(room.min(u32::MAX as usize) as u32),
        _ => Err(Error::invalid(format!(
            "prompt of {prompt_len} tokens leaves no room to generate in the trained window {}",
            layout.window
        ))),
    }
}

/// How many rollout sequences may decode at once: the explicit
/// `generation_concurrency`, or the full optimizer sequence width when it is
/// left at zero (the "retain the maximum" contract).
pub(crate) fn generation_wave(training: &TrainConfig, layout: &RowLayout) -> usize {
    if training.generation_concurrency == 0 {
        layout.n_seq_max.max(1)
    } else {
        training.generation_concurrency as usize
    }
}

/// Samples one completion for the prompt, then re-scores the full sequence
/// through the same teacher-forced path the optimizer epochs use, so every
/// ratio is exactly 1 before the first optimizer step instead of carrying
/// incremental-vs-batched decode noise. Later size-one SGD steps legitimately
/// move the policy before subsequent rollouts are visited. Returns the rollout
/// and the detokenized completion text.
///
/// `score_policy` controls whether the behavior-policy log-probabilities are
/// scored immediately; PPO with a critic passes `false` because they arrive
/// together with the hidden states in its fused `batch_advantages` pass.
pub(crate) fn sample_rollout(
    trainer: &mut Trainer,
    prompt_tokens: &[i32],
    sampling: &SamplingParams,
    seed_offset: usize,
    layout: &RowLayout,
    score_policy: bool,
) -> Result<(Rollout, String)> {
    sample_rollouts(
        trainer,
        prompt_tokens,
        sampling,
        &[seed_offset],
        layout,
        score_policy,
    )?
    .pop()
    .ok_or_else(|| Error::runtime("generation batch returned no rollout"))
}

/// Samples a group from one prompt in a single multi-sequence decode. Seeds
/// are derived independently from the supplied offsets, so batching changes
/// scheduling only, not any member's random stream.
pub(crate) fn sample_rollouts(
    trainer: &mut Trainer,
    prompt_tokens: &[i32],
    sampling: &SamplingParams,
    seed_offsets: &[usize],
    layout: &RowLayout,
    score_policy: bool,
) -> Result<Vec<(Rollout, String)>> {
    if seed_offsets.is_empty() {
        return Err(Error::invalid("rollout batch requires seed offsets"));
    }
    let room = generation_room(layout, prompt_tokens.len())?;
    let params = seed_offsets
        .iter()
        .map(|&seed_offset| SamplingParams {
            max_new_tokens: sampling.max_new_tokens.min(room),
            seed: sampling.seed.wrapping_add(seed_offset as u32),
            ..*sampling
        })
        .collect::<Vec<_>>();
    let completions = trainer.generate_tokens_batch(prompt_tokens, &params)?;
    let n_prompt = prompt_tokens.len();
    let mut rollouts = Vec::with_capacity(completions.len());
    for completion in completions {
        let completion_text = trainer.detokenize(&completion, false)?;
        let mut tokens = Vec::with_capacity(n_prompt + completion.len());
        tokens.extend_from_slice(prompt_tokens);
        tokens.extend_from_slice(&completion);
        rollouts.push((
            Rollout {
                train_mask: make_train_mask(n_prompt, tokens.len()),
                tokens,
                old_logprobs: Vec::new(),
            },
            completion_text,
        ));
    }
    if score_policy {
        // Every member shares this prompt, so the behavior policy is scored
        // through one shared-prefix pass per sequence-width chunk instead of one
        // full prompt prefill per member.
        for chunk in rollouts.chunks_mut(layout.n_seq_max.max(1)) {
            let scores = {
                let members = chunk.iter().map(|(rollout, _)| rollout).collect::<Vec<_>>();
                score_train_mask_group(trainer, &members)?
            };
            for ((rollout, _), scores) in chunk.iter_mut().zip(scores) {
                rollout.old_logprobs = scores;
            }
        }
    }
    Ok(rollouts)
}

/// Samples heterogeneous prompt groups in continuous decode batches. A group
/// may span several physical waves when `sequence_capacity < group_size`; its
/// independently seeded members are reassembled in their original order.
///
/// Waves are filled group by group and a group is never split just because the
/// current wave is nearly full, so a capacity that is not a multiple of the
/// group size costs idle sequence slots rather than duplicate prompt prefills.
pub(crate) fn sample_rollout_groups_continuous(
    trainer: &mut Trainer,
    groups: &[(&[i32], Vec<usize>)],
    sampling: &SamplingParams,
    layout: &RowLayout,
    sequence_capacity: usize,
) -> Result<Vec<Vec<(Rollout, String)>>> {
    if groups.is_empty() || sequence_capacity == 0 {
        return Err(Error::invalid(
            "continuous rollout sampling requires capacity and groups",
        ));
    }
    if groups.iter().any(|(_, offsets)| offsets.is_empty()) {
        return Err(Error::invalid(
            "continuous rollout groups must not be empty",
        ));
    }
    let mut output = groups
        .iter()
        .map(|(_, offsets)| Vec::with_capacity(offsets.len()))
        .collect::<Vec<_>>();
    let group_sizes = groups
        .iter()
        .map(|(_, offsets)| offsets.len())
        .collect::<Vec<_>>();
    for wave in continuous_waves(&group_sizes, sequence_capacity) {
        let mut requests = Vec::with_capacity(wave.len());
        for &(group_index, member_index) in &wave {
            let (prompt, offsets) = &groups[group_index];
            let room = generation_room(layout, prompt.len())?;
            requests.push((
                *prompt,
                SamplingParams {
                    max_new_tokens: sampling.max_new_tokens.min(room),
                    seed: sampling.seed.wrapping_add(offsets[member_index] as u32),
                    ..*sampling
                },
            ));
        }
        let completions = trainer.generate_tokens_continuous(&requests)?;
        for (&(group_index, _), completion) in wave.iter().zip(completions) {
            let prompt = groups[group_index].0;
            let n_prompt = prompt.len();
            let completion_text = trainer.detokenize(&completion, false)?;
            let mut tokens = Vec::with_capacity(n_prompt + completion.len());
            tokens.extend_from_slice(prompt);
            tokens.extend_from_slice(&completion);
            output[group_index].push((
                Rollout {
                    train_mask: make_train_mask(n_prompt, tokens.len()),
                    tokens,
                    old_logprobs: Vec::new(),
                },
                completion_text,
            ));
        }
    }
    Ok(output)
}

/// Splits `group_sizes` into physical sampling waves of at most
/// `sequence_capacity` sequences, as `(group, member)` pairs in submission
/// order.
///
/// Groups are kept whole: a group that would straddle a wave boundary starts the
/// next one instead, because the runtime prefills a shared prompt once per call
/// and a straddling group pays for its prompt in both waves. A group wider than
/// the capacity is the one case that still splits - no wave could hold it - and
/// it then fills whole waves from the start.
pub(super) fn continuous_waves(
    group_sizes: &[usize],
    sequence_capacity: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut waves = Vec::new();
    let mut wave: Vec<(usize, usize)> = Vec::new();
    for (group, &size) in group_sizes.iter().enumerate() {
        let mut member = 0_usize;
        while member < size {
            let remaining = size - member;
            if !wave.is_empty()
                && member == 0
                && remaining <= sequence_capacity
                && wave.len() + remaining > sequence_capacity
            {
                waves.push(std::mem::take(&mut wave));
            }
            let take = (sequence_capacity - wave.len()).min(remaining);
            wave.extend((member..member + take).map(|member| (group, member)));
            member += take;
            if wave.len() == sequence_capacity {
                waves.push(std::mem::take(&mut wave));
            }
        }
    }
    if !wave.is_empty() {
        waves.push(wave);
    }
    waves
}
