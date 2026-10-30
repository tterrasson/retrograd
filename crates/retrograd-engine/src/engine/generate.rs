//! The engine's generation API: the same semantics as the trainer's
//! ([`crate::Trainer::generate_batch`] and friends), over the engine's own FFI
//! entry points. Both run the same runtime core, so at equal geometry, seeds
//! and weights the outputs are bit-identical.

use super::*;
use crate::generation::{check_lengths, refuse_forced, split_rows};
use crate::{ForcedRows, GenerationBatch, TokenRows};

// The runtime stops at a full context, so allocating past that point only
// wastes memory (a u32::MAX token budget must not request a multi-GB buffer).
fn output_limit(n_ctx: usize, n_prompt: usize, max_new_tokens: u32) -> Result<usize, EngineError> {
    if n_prompt >= n_ctx {
        return Err(EngineError::invalid(
            "prompt leaves no room to generate in the context",
        ));
    }
    Ok((max_new_tokens as usize).min(n_ctx - n_prompt))
}

impl Engine {
    fn check_sequence_count(&self, count: usize) -> Result<(), EngineError> {
        if count > self.n_seq_max {
            return Err(EngineError::invalid(
                "generation batch exceeds the configured sequence capacity",
            ));
        }
        Ok(())
    }

    /// Samples a completion for the prompt with the current model + adapter,
    /// returning the sampled tokens and their temperature-1 policy logprobs.
    /// Generation stops at `max_new_tokens`, an end-of-generation token (kept
    /// in the output), or a full context.
    pub fn generate(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
    ) -> std::result::Result<Generation, EngineError> {
        self.generate_batch(prompt, std::slice::from_ref(sampling))?
            .pop()
            .ok_or_else(|| EngineError::Runtime("generation batch returned no sequence".into()))
    }

    /// Samples independent completions for one shared prompt. Each entry
    /// keeps its own parameters and RNG seed.
    pub fn generate_batch(
        &mut self,
        prompt: &[i32],
        sampling: &[SamplingParams],
    ) -> std::result::Result<Vec<Generation>, EngineError> {
        let (tokens, logprobs, forced) = self.generate_batch_raw(prompt, sampling, true)?;
        refuse_forced(&forced)?;
        Ok(tokens
            .into_iter()
            .zip(logprobs.expect("logprobs requested for generation batch"))
            .map(|(tokens, logprobs)| Generation { tokens, logprobs })
            .collect())
    }

    /// Logprob-free sibling of [`Engine::generate_batch`].
    pub fn generate_tokens_batch(
        &mut self,
        prompt: &[i32],
        sampling: &[SamplingParams],
    ) -> std::result::Result<Vec<Vec<i32>>, EngineError> {
        let (tokens, _, forced) = self.generate_batch_raw(prompt, sampling, false)?;
        refuse_forced(&forced)?;
        Ok(tokens)
    }

    /// Samples heterogeneous prompts together, one KV sequence per request.
    pub fn generate_tokens_continuous(
        &mut self,
        sequences: &[(&[i32], SamplingParams)],
    ) -> std::result::Result<Vec<Vec<i32>>, EngineError> {
        let rows = self.generate_sampled_continuous(sequences)?;
        refuse_forced(rows.iter().map(|row| &row.forced))?;
        Ok(rows.into_iter().map(|row| row.tokens).collect())
    }

    /// [`Engine::generate_tokens_continuous`], with each row's forced-token
    /// mask: the generator for a caller that runs under a reasoning budget.
    pub fn generate_sampled_continuous(
        &mut self,
        sequences: &[(&[i32], SamplingParams)],
    ) -> std::result::Result<Vec<SampledRow>, EngineError> {
        if sequences.is_empty() {
            return Err(EngineError::invalid(
                "continuous generation requires sequences",
            ));
        }
        // Refuse before allocating row-major buffers for an unusable batch.
        self.check_sequence_count(sequences.len())?;
        if sequences
            .iter()
            .any(|(prompt, params)| prompt.is_empty() || params.max_new_tokens == 0)
        {
            return Err(EngineError::invalid(
                "continuous generation requires non-empty prompts and positive token budgets",
            ));
        }
        let n_ctx = self.context_size()? as usize;
        let stride = sequences.iter().try_fold(0, |stride, (prompt, params)| {
            output_limit(n_ctx, prompt.len(), params.max_new_tokens).map(|limit| stride.max(limit))
        })?;
        let capacity = sequences.len().checked_mul(stride).ok_or_else(|| {
            EngineError::Overflow("continuous generation shape overflows usize".into())
        })?;
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
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_generate_continuous_batch(
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
    ) -> std::result::Result<GenerationBatch, EngineError> {
        if prompt.is_empty() {
            return Err(EngineError::invalid(
                "generation requires a non-empty prompt",
            ));
        }
        if sampling.is_empty() {
            return Err(EngineError::invalid("generation batch requires sequences"));
        }
        self.check_sequence_count(sampling.len())?;
        if sampling.iter().any(|params| params.max_new_tokens == 0) {
            return Err(EngineError::invalid(
                "max_new_tokens must be greater than zero",
            ));
        }
        let stride = sampling
            .iter()
            .map(|params| params.max_new_tokens)
            .max()
            .expect("non-empty sampling batch");
        let stride = output_limit(self.context_size()? as usize, prompt.len(), stride)?;
        let capacity = sampling.len().checked_mul(stride).ok_or_else(|| {
            EngineError::Overflow("generation batch shape overflows usize".into())
        })?;
        let ffi_sampling = sampling
            .iter()
            .map(sampling_params_to_ffi)
            .collect::<Vec<_>>();
        let mut flat_tokens = vec![0_i32; capacity];
        let mut flat_forced = vec![0_u8; capacity];
        let mut flat_logprobs = include_logprobs.then(|| vec![0.0_f32; capacity]);
        let mut lengths = vec![0_usize; sampling.len()];
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_generate_batch(
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
        check_lengths(&lengths, stride)?;
        let rows: TokenRows = split_rows(&lengths, stride, &flat_tokens);
        let logprobs = flat_logprobs.map(|values| split_rows(&lengths, stride, &values));
        let forced: ForcedRows = split_rows(&lengths, stride, &flat_forced)
            .into_iter()
            .map(|row| row.into_iter().map(|flag| flag != 0).collect())
            .collect();
        Ok((rows, logprobs, forced))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_buffers_are_bounded_by_the_remaining_context() {
        assert_eq!(output_limit(256, 240, u32::MAX).unwrap(), 16);
        assert_eq!(output_limit(256, 240, 4).unwrap(), 4);
        assert!(matches!(
            output_limit(256, 256, 1),
            Err(EngineError::InvalidArgument(_))
        ));
        assert!(matches!(
            output_limit(256, 257, 1),
            Err(EngineError::InvalidArgument(_))
        ));
    }
}
