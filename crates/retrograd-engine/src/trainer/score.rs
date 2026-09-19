use super::*;

impl Trainer {
    /// Teacher-forced scoring: returns `log p(tokens[i+1] | tokens[..=i])` for
    /// every position, i.e. `tokens.len() - 1` values. Forward only.
    pub fn score_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.len() < 2 {
            return Err(Error::tokenize("scoring requires at least two tokens"));
        }
        let len = tokens.len() - 1;
        // SAFETY: the `Trainer` invariant holds, all borrowed arguments live
        // through this synchronous call, and the runtime writes every target
        // position on success.
        unsafe {
            ffi_out_vec(len, |logprobs_out| {
                self.check(ffi::retro_trainer_score_tokens(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    logprobs_out,
                ))?;
                Ok(len)
            })
        }
    }

    /// Teacher-forced scores selected by an explicit target-token mask.
    /// Returned values follow increasing token-index order. This is the public
    /// collection primitive used by multi-turn agents, where policy actions
    /// may be separated by untrained tool observations.
    pub fn score_masked_tokens(&mut self, tokens: &[i32], train_mask: &[bool]) -> Result<Vec<f32>> {
        let first = first_masked_target(tokens, train_mask)?;
        // Score from the first selected target rather than from token 1: the
        // logit head then runs only over the tail, and this is bit-for-bit the
        // same range the optimizer re-scores, so the first-step ratio stays 1.
        let suffix_scores = self.score_token_suffix(tokens, first)?;
        Ok((first..tokens.len())
            .zip(suffix_scores)
            .filter_map(|(target, score)| train_mask[target].then_some(score))
            .collect())
    }

    /// Teacher-forced scores for completion targets only. `n_prompt` is the
    /// first target-token index, so the output has `tokens.len() - n_prompt`
    /// values while the full prompt still conditions the forward pass.
    pub fn score_token_suffix(&mut self, tokens: &[i32], n_prompt: usize) -> Result<Vec<f32>> {
        let mut logprobs = Vec::new();
        self.score_token_suffix_into(tokens, n_prompt, &mut logprobs)?;
        Ok(logprobs)
    }

    /// Reusable-buffer variant for the current-policy rescoring performed once
    /// per PPO/GRPO optimizer step.
    pub fn score_token_suffix_into(
        &mut self,
        tokens: &[i32],
        n_prompt: usize,
        logprobs: &mut Vec<f32>,
    ) -> Result<()> {
        validate_suffix(tokens, n_prompt)?;
        let len = tokens.len() - n_prompt;
        // SAFETY: the `Trainer` invariant holds, all borrowed arguments live
        // through this synchronous call, and the runtime writes every suffix
        // position on success.
        unsafe {
            ffi_refill_vec(logprobs, len, |logprobs_out| {
                self.check(ffi::retro_trainer_score_token_suffix(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    n_prompt,
                    logprobs_out,
                ))
            })
        }
    }

    /// The truncated distribution this model puts on every completion target:
    /// the `k` most probable next tokens of each position and their
    /// temperature-1 log-probabilities, in decreasing probability order.
    ///
    /// This is the only scoring entry point that returns more than the value of
    /// a token the caller already had. It runs the teacher-forced forward pass
    /// [`Trainer::score_token_suffix`] runs, and reduces the whole logits row on
    /// the host instead of gathering one target in the decode graph - which is
    /// the point, and also why it is not on the per-optimizer-step path.
    ///
    /// For `k = 1` the id of a row is its argmax and the value beside it is what
    /// [`Trainer::score_token_suffix`] returns when handed that same id.
    pub fn top_logprobs_suffix(
        &mut self,
        tokens: &[i32],
        n_prompt: usize,
        k: usize,
    ) -> Result<TopLogprobs> {
        validate_suffix(tokens, n_prompt)?;
        if k == 0 {
            return Err(Error::invalid(
                "top-k suffix scoring requires at least one entry per position",
            ));
        }
        let entries = (tokens.len() - n_prompt)
            .checked_mul(k)
            .ok_or_else(|| Error::overflow("top-k suffix scoring output size overflows usize"))?;
        let mut ids = vec![0_i32; entries];
        let mut logprobs = vec![0.0_f32; entries];
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_top_logprobs_suffix(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                n_prompt,
                k,
                ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                entries,
            )
        })?;
        Ok(TopLogprobs::from_flat(k, ids, logprobs))
    }

    /// Scores several suffixes that share an identical prompt in one
    /// forward-only call.  The runtime decodes that prefix once, branches its
    /// KV state, and returns rows in the input order.
    pub fn score_token_suffix_batch(
        &mut self,
        sequences: &[(&[i32], usize)],
    ) -> Result<Vec<Vec<f32>>> {
        if sequences.is_empty() {
            return Err(Error::invalid("suffix scoring batch requires sequences"));
        }
        let stride = sequences
            .iter()
            .map(|(tokens, n_prompt)| {
                if *n_prompt == 0 || *n_prompt >= tokens.len() {
                    Err(Error::invalid(
                        "suffix scoring requires a prompt and at least one completion token",
                    ))
                } else {
                    Ok(tokens.len() - *n_prompt)
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .expect("non-empty suffix scoring batch");
        let ffi_sequences = sequences
            .iter()
            .map(|(tokens, n_prompt)| ffi::RetroTokenSuffixSequence {
                tokens: tokens.as_ptr(),
                n_tokens: tokens.len(),
                n_prompt: *n_prompt,
            })
            .collect::<Vec<_>>();
        let mut scores = vec![0.0_f32; sequences.len() * stride];
        let mut lengths = vec![0_usize; sequences.len()];
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_score_token_suffix_batch(
                self.raw.as_ptr(),
                ffi_sequences.as_ptr(),
                ffi_sequences.len(),
                scores.as_mut_ptr(),
                stride,
                lengths.as_mut_ptr(),
            )
        })?;
        sequences
            .iter()
            .enumerate()
            .map(|(row, (tokens, n_prompt))| {
                let expected = tokens.len() - *n_prompt;
                if lengths[row] != expected || lengths[row] > stride {
                    return Err(Error::runtime(
                        "suffix scoring runtime returned an invalid sequence length",
                    ));
                }
                Ok(scores[row * stride..row * stride + expected].to_vec())
            })
            .collect()
    }

    /// Teacher-forced scoring under the frozen base model, with the LoRA
    /// adapter temporarily disabled. This is GRPO's fixed reference policy.
    pub fn score_reference_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.with_lora_disabled(|trainer| trainer.score_tokens(tokens))
    }

    pub(super) fn set_lora_enabled(&mut self, enabled: bool) -> Result<()> {
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_set_lora_enabled(self.raw.as_ptr(), enabled) })
    }

    /// Runs several forward-only calls with LoRA disabled, restoring it even
    /// if the operation fails. GRPO uses this to amortize the graph rebuild
    /// over a whole fixed-reference scoring batch.
    /// Refuses base-weight training: disabling LoRA cannot recover a frozen
    /// reference once the base weights are trainable.
    pub fn with_lora_disabled<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.trains_base_weights {
            return Err(Error::invalid(
                "a fixed reference requires a separate frozen model when training base weights",
            ));
        }
        self.set_lora_enabled(false)?;
        let result = operation(self);
        let restore = self.set_lora_enabled(true);
        match (result, restore) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (_, Err(restore_error)) => Err(restore_error),
        }
    }

    /// The model's output hidden-state width: features per position returned
    /// by [`Trainer::hidden_states`].
    pub fn hidden_size(&self) -> Result<usize> {
        let mut n_embd = 0_u32;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live through this synchronous call.
        self.check(unsafe { ffi::retro_trainer_hidden_size(self.raw.as_ptr(), &mut n_embd) })?;
        Ok(n_embd as usize)
    }

    /// Teacher-forced feature extraction: the final-layer hidden state of
    /// every position, as `tokens.len()` rows of [`Trainer::hidden_size`]
    /// floats (row-major). Forward only; feeds the PPO value head.
    pub fn hidden_states(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::tokenize("hidden-state extraction requires tokens"));
        }
        let n_embd = self.hidden_size()?;
        let len = tokens
            .len()
            .checked_mul(n_embd)
            .ok_or_else(|| Error::overflow("hidden-state output size overflows usize"))?;
        // SAFETY: the `Trainer` invariant holds, all borrowed arguments live
        // through this synchronous call, and the runtime writes every feature
        // on success.
        unsafe {
            ffi_out_vec(len, |features_out| {
                self.check(ffi::retro_trainer_hidden_states(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    features_out,
                    len,
                ))?;
                Ok(len)
            })
        }
    }

    /// Scores completion targets and extracts the corresponding pre-token
    /// hidden states in one teacher-forced forward pass. `logprobs` is
    /// refilled with one value per completion token; the feature rows are
    /// appended to `features`, so a batch can accumulate rollouts in place.
    pub fn score_token_suffix_and_hidden_states_into(
        &mut self,
        tokens: &[i32],
        n_prompt: usize,
        n_embd: usize,
        logprobs: &mut Vec<f32>,
        features: &mut Vec<f32>,
    ) -> Result<()> {
        if n_prompt == 0 || n_prompt >= tokens.len() {
            return Err(Error::invalid(
                "combined scoring requires a prompt and completion tokens",
            ));
        }
        let rows = tokens.len() - n_prompt;
        let feature_len = rows
            .checked_mul(n_embd)
            .ok_or_else(|| Error::overflow("combined feature output size overflows usize"))?;
        // SAFETY: the `Trainer` invariant holds, all borrowed arguments live
        // through this synchronous call, and the runtime writes both output
        // buffers completely on success.
        unsafe {
            ffi_refill_vec(logprobs, rows, |logprobs_out| {
                ffi_extend_vec(features, feature_len, |features_out| {
                    self.check(ffi::retro_trainer_score_token_suffix_and_hidden_states(
                        self.raw.as_ptr(),
                        tokens.as_ptr(),
                        tokens.len(),
                        n_prompt,
                        logprobs_out,
                        features_out,
                        feature_len,
                    ))
                })
            })
        }
    }
}
