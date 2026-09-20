use super::*;

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
        let (tokens, logprobs) = self.generate_batch_raw(prompt, sampling, true)?;
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
        Ok(self.generate_batch_raw(prompt, sampling, false)?.0)
    }

    /// Samples heterogeneous prompts together. The runtime keeps one KV
    /// sequence per request and advances all live rows in the same decode
    /// calls; caller-controlled chunking keeps the request within its
    /// configured sequence capacity.
    pub fn generate_tokens_continuous(
        &mut self,
        sequences: &[(&[i32], SamplingParams)],
    ) -> Result<Vec<Vec<i32>>> {
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
        let mut ffi_sequences = sequences
            .iter()
            .map(|(prompt, params)| ffi::RetroGenerationSequence {
                prompt_tokens: prompt.as_ptr(),
                n_prompt: prompt.len(),
                sampling: sampling_params_to_ffi(params),
            })
            .collect::<Vec<_>>();
        let mut tokens = vec![0_i32; sequences.len() * stride];
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
            )
        })?;
        lengths
            .into_iter()
            .enumerate()
            .map(|(row, len)| {
                if len == 0 || len > stride {
                    return Err(Error::runtime(
                        "continuous generation runtime returned an invalid sequence length",
                    ));
                }
                Ok(tokens[row * stride..row * stride + len].to_vec())
            })
            .collect()
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
            )
        })?;
        // An empty row is rejected here, like in `generate_tokens_continuous`:
        // it would otherwise become a rollout whose train mask is entirely
        // false, surfacing much later as "rollout has no trainable token".
        if lengths.iter().any(|&len| len == 0 || len > stride) {
            return Err(Error::runtime(
                "generation runtime returned an invalid sequence length",
            ));
        }
        let rows = lengths
            .iter()
            .enumerate()
            .map(|(row, &len)| flat_tokens[row * stride..row * stride + len].to_vec())
            .collect();
        let logprobs = flat_logprobs.map(|values| {
            lengths
                .iter()
                .enumerate()
                .map(|(row, &len)| values[row * stride..row * stride + len].to_vec())
                .collect()
        });
        Ok((rows, logprobs))
    }
}
