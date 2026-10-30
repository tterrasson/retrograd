use super::*;
use crate::generation::{check_lengths, refuse_forced, split_rows};

/// One sampled row and, token for token, whether the reasoning budget forced
/// it (see [`Trainer::set_reasoning_budget`]). A forced token was the
/// runtime's choice, not the policy's, so a caller that trains on the row
/// masks it out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SampledRow {
    pub tokens: Vec<i32>,
    pub forced: Vec<bool>,
}

impl Trainer {
    /// Samples a completion for the prompt with the current model + LoRA
    /// weights, returning the sampled tokens and their temperature-1 policy
    /// logprobs. Generation stops at `max_new_tokens`, an end-of-generation
    /// token (kept in the output), or a full context.
    pub fn generate(&mut self, prompt: &[i32], sampling: &SamplingParams) -> Result<Generation> {
        self.generate_batch(prompt, std::slice::from_ref(sampling))?
            .pop()
            .ok_or_else(|| Error::runtime("generation batch returned no sequence"))
    }

    /// Samples independent completions for one shared prompt. The runtime
    /// decodes the prompt once and advances every live sequence together;
    /// each entry keeps its own parameters and RNG seed.
    pub fn generate_batch(
        &mut self,
        prompt: &[i32],
        sampling: &[SamplingParams],
    ) -> Result<Vec<Generation>> {
        let (tokens, logprobs, forced) = self.generate_batch_raw(prompt, sampling, true)?;
        refuse_forced(&forced)?;
        Ok(tokens
            .into_iter()
            .zip(logprobs.expect("logprobs requested for generation batch"))
            .map(|(tokens, logprobs)| Generation { tokens, logprobs })
            .collect())
    }

    /// Samples a completion under this run's reference policy: the attached
    /// anchor when there is one, otherwise the frozen base model with any
    /// loaded LoRA adapter temporarily disabled. A base-weight run with no
    /// anchor is refused: there is no frozen model left to sample from.
    pub fn generate_base(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
    ) -> Result<Generation> {
        self.with_reference_policy(|trainer| trainer.generate(prompt, sampling))
    }

    /// Logprob-free sibling of [`Trainer::generate_batch`] used by rollout
    /// collection before exact teacher-forced scoring.
    pub fn generate_tokens_batch(
        &mut self,
        prompt: &[i32],
        sampling: &[SamplingParams],
    ) -> Result<Vec<Vec<i32>>> {
        let (tokens, _, forced) = self.generate_batch_raw(prompt, sampling, false)?;
        refuse_forced(&forced)?;
        Ok(tokens)
    }

    /// Samples heterogeneous prompts together. The runtime keeps one KV
    /// sequence per request and advances all live rows in the same decode
    /// calls; caller-controlled chunking keeps the request within its
    /// configured sequence capacity.
    pub fn generate_tokens_continuous(
        &mut self,
        sequences: &[(&[i32], SamplingParams)],
    ) -> Result<Vec<Vec<i32>>> {
        let rows = self.generate_sampled_continuous(sequences)?;
        refuse_forced(rows.iter().map(|row| &row.forced))?;
        Ok(rows.into_iter().map(|row| row.tokens).collect())
    }

    /// [`Trainer::generate_sampled_continuous`], with extended sampling per
    /// row: [`SamplingParamsV2`] carries the GBNF grammar that constrains the
    /// output - the path a structured judge answer takes. Every row is sampled
    /// by the CPU oracle (the device sampler chain stays v1), so this is not a
    /// rollout loop. Grammar handles are staged per distinct GBNF text and
    /// released when the call returns, which the runtime contract allows.
    pub fn generate_sampled_continuous_v2(
        &mut self,
        sequences: &[(&[i32], SamplingParamsV2)],
    ) -> Result<Vec<SampledRow>> {
        if sequences.is_empty() {
            return Err(Error::invalid(
                "continuous v2 generation requires sequences",
            ));
        }
        if sequences
            .iter()
            .any(|(prompt, params)| prompt.is_empty() || params.max_new_tokens == 0)
        {
            return Err(Error::invalid(
                "continuous v2 generation requires non-empty prompts and positive token budgets",
            ));
        }
        for (_, params) in sequences {
            if params.logit_bias.len() > 4096 {
                return Err(Error::invalid("logit_bias takes at most 4096 entries"));
            }
        }
        let stride = sequences
            .iter()
            .map(|(_, params)| params.max_new_tokens as usize)
            .max()
            .expect("non-empty continuous v2 generation batch");
        let capacity = sequences.len().checked_mul(stride).ok_or_else(|| {
            Error::Overflow("continuous v2 generation shape overflows usize".into())
        })?;
        // Stage the logit-bias pairs once, then point each row at its slice:
        // the Vec must not grow between the pointers being taken and the call.
        let (bias_ids, bias_values): (Vec<i32>, Vec<f32>) = sequences
            .iter()
            .flat_map(|(_, params)| params.logit_bias.iter().copied())
            .unzip();
        let mut grammars: Vec<(std::sync::Arc<str>, ffi::Grammar)> = Vec::new();
        let mut ffi_sequences = Vec::with_capacity(sequences.len());
        let mut bias_offset = 0_usize;
        for (prompt, params) in sequences {
            let n_bias = params.logit_bias.len();
            let mut sampling = crate::sampling_params_v2_to_ffi(
                params,
                bias_ids[bias_offset..].as_ptr(),
                bias_values[bias_offset..].as_ptr(),
            );
            // Each row was checked above to contain at most 4096 entries.
            sampling.n_logit_bias = n_bias as u32;
            if let Some(gbnf) = &params.grammar_gbnf {
                let handle = match grammars.iter().find(|(text, _)| text == gbnf) {
                    Some((_, handle)) => handle,
                    None => {
                        let handle = ffi::Grammar::from_gbnf(gbnf).map_err(Error::invalid)?;
                        grammars.push((gbnf.clone(), handle));
                        &grammars.last().expect("just pushed").1
                    }
                };
                sampling.grammar = handle.as_ptr();
            }
            ffi_sequences.push(ffi::RetroGenerationSequenceV2 {
                prompt_tokens: prompt.as_ptr(),
                n_prompt: prompt.len(),
                sampling,
            });
            bias_offset += n_bias;
        }
        let mut tokens = vec![0_i32; capacity];
        let mut forced = vec![0_u8; capacity];
        let mut lengths = vec![0_usize; sequences.len()];
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments -
        // prompts, staged bias arrays and grammar handles - live through this
        // synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_generate_continuous_batch_v2(
                self.raw.as_ptr(),
                ffi_sequences.as_ptr(),
                ffi_sequences.len(),
                tokens.as_mut_ptr(),
                std::ptr::null_mut(),
                stride,
                lengths.as_mut_ptr(),
                forced.as_mut_ptr(),
            )
        })?;
        check_lengths(&lengths, stride)?;
        Ok(lengths
            .into_iter()
            .enumerate()
            .map(|(row, len)| {
                let span = row * stride..row * stride + len;
                SampledRow {
                    tokens: tokens[span.clone()].to_vec(),
                    forced: forced[span].iter().map(|&flag| flag != 0).collect(),
                }
            })
            .collect())
    }

    /// [`Trainer::generate_tokens_continuous`], with each row's forced-token
    /// mask: the generator for a caller that runs under a reasoning budget.
    pub fn generate_sampled_continuous(
        &mut self,
        sequences: &[(&[i32], SamplingParams)],
    ) -> Result<Vec<SampledRow>> {
        if sequences.is_empty() {
            return Err(Error::invalid("continuous generation requires sequences"));
        }
        if sequences
            .iter()
            .any(|(prompt, params)| prompt.is_empty() || params.max_new_tokens == 0)
        {
            return Err(Error::invalid(
                "continuous generation requires non-empty prompts and positive token budgets",
            ));
        }
        let stride = sequences
            .iter()
            .map(|(_, params)| params.max_new_tokens as usize)
            .max()
            .expect("non-empty continuous generation batch");
        let capacity = sequences
            .len()
            .checked_mul(stride)
            .ok_or_else(|| Error::Overflow("continuous generation shape overflows usize".into()))?;
        let mut ffi_sequences = sequences
            .iter()
            .map(|(prompt, params)| ffi::RetroGenerationSequence {
                prompt_tokens: prompt.as_ptr(),
                n_prompt: prompt.len(),
                sampling: sampling_params_to_ffi(params),
            })
            .collect::<Vec<_>>();
        let mut tokens = vec![0_i32; capacity];
        let mut forced = vec![0_u8; capacity];
        let mut lengths = vec![0_usize; sequences.len()];
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_generate_continuous_batch(
                self.raw.as_ptr(),
                ffi_sequences.as_mut_ptr(),
                ffi_sequences.len(),
                tokens.as_mut_ptr(),
                std::ptr::null_mut(),
                stride,
                lengths.as_mut_ptr(),
                forced.as_mut_ptr(),
            )
        })?;
        check_lengths(&lengths, stride)?;
        Ok(lengths
            .into_iter()
            .enumerate()
            .map(|(row, len)| {
                let span = row * stride..row * stride + len;
                SampledRow {
                    tokens: tokens[span.clone()].to_vec(),
                    forced: forced[span].iter().map(|&flag| flag != 0).collect(),
                }
            })
            .collect())
    }

    fn generate_batch_raw(
        &mut self,
        prompt: &[i32],
        sampling: &[SamplingParams],
        include_logprobs: bool,
    ) -> Result<GenerationBatch> {
        if prompt.is_empty() {
            return Err(Error::invalid("generation requires a non-empty prompt"));
        }
        if sampling.is_empty() {
            return Err(Error::invalid("generation batch requires sequences"));
        }
        if sampling.iter().any(|params| params.max_new_tokens == 0) {
            return Err(Error::invalid("max_new_tokens must be greater than zero"));
        }
        let stride = sampling
            .iter()
            .map(|params| params.max_new_tokens as usize)
            .max()
            .expect("non-empty sampling batch");
        let capacity = sampling
            .len()
            .checked_mul(stride)
            .ok_or_else(|| Error::overflow("generation batch shape overflows usize"))?;
        let ffi_sampling = sampling
            .iter()
            .map(sampling_params_to_ffi)
            .collect::<Vec<_>>();
        let mut flat_tokens = vec![0_i32; capacity];
        let mut flat_forced = vec![0_u8; capacity];
        let mut flat_logprobs = include_logprobs.then(|| vec![0.0_f32; capacity]);
        let mut lengths = vec![0_usize; sampling.len()];
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_generate_batch(
                self.raw.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
                ffi_sampling.as_ptr(),
                sampling.len(),
                flat_tokens.as_mut_ptr(),
                flat_logprobs
                    .as_mut()
                    .map_or(std::ptr::null_mut(), |values| values.as_mut_ptr()),
                stride,
                lengths.as_mut_ptr(),
                flat_forced.as_mut_ptr(),
            )
        })?;
        // An empty row is rejected here, like in `generate_tokens_continuous`:
        // it would otherwise become a rollout whose train mask is entirely
        // false, surfacing much later as "rollout has no trainable token".
        check_lengths(&lengths, stride)?;
        let rows = split_rows(&lengths, stride, &flat_tokens);
        let logprobs = flat_logprobs.map(|values| split_rows(&lengths, stride, &values));
        let forced = split_rows(&lengths, stride, &flat_forced)
            .into_iter()
            .map(|row| row.into_iter().map(|flag| flag != 0).collect())
            .collect();
        Ok((rows, logprobs, forced))
    }
}
