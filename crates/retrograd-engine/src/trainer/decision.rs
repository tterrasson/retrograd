//! Decision models on the trainer: the same template, tokenizer and layout
//! the engine serves with, the forward pass under the weights being trained,
//! and the grouped training step.
//!
//! A training run renders every question through the model's `systemone`
//! template (or the override the run names), exactly as
//! [`Engine::decide`](crate::Engine::decide) does, so what is trained is what
//! will be served.

use std::ffi::CString;

use retrograd_decision::request::Question;
use retrograd_decision::{Json, Token};

use super::*;
use crate::engine::{DecisionHost, decision_info_from, decision_pass, readout_code};
use crate::{
    DECISION_TEMPLATE, DecisionInfo, DecisionMetadata, DecisionPass, EngineError, GroupedBatch,
};

impl DecisionHost for Trainer {
    fn render_decision(&self, context: &Json) -> std::result::Result<String, EngineError> {
        let name = CString::new(DECISION_TEMPLATE).map_err(nul_error)?;
        let context = CString::new(context.to_string()).map_err(nul_error)?;
        // SAFETY: the `Trainer` invariant holds and both strings live through
        // this synchronous call, which follows the two-call string contract.
        Ok(unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_trainer_render_named_template(
                    self.raw.as_ptr(),
                    name.as_ptr(),
                    context.as_ptr(),
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    fn tokenize_decision(
        &self,
        text: &str,
        parse_special: bool,
    ) -> std::result::Result<Vec<Token>, EngineError> {
        Ok(self.tokenize_with(text, false, parse_special)?)
    }

    fn token_text(&self, token: Token) -> std::result::Result<String, EngineError> {
        Ok(self.detokenize(&[token], true)?)
    }
}

impl Trainer {
    /// Installs a generic letter-logit readout on a causal model. The returned
    /// standalone template is the exact source to embed in an exported GGUF.
    /// Chat-template variables must be set before calling this method.
    pub fn configure_causal_decision(&mut self) -> Result<(DecisionInfo, String)> {
        let mut metadata = self.decision_metadata()?;
        if metadata.decision_type.is_some() {
            return Err(Error::invalid(
                "the model already declares a decision readout",
            ));
        }
        let source = retrograd_decision::causal::standalone(|content| {
            self.format_chat(&[("user", content)], true)
        })?;
        self.set_decision_template(Some(&source))?;
        metadata.decision_type = Some("openjev".into());
        let info = decision_info_from(self, &metadata, true)?;
        Ok((info, source))
    }

    /// What the loaded model declares about its decision head.
    pub fn decision_metadata(&self) -> Result<DecisionMetadata> {
        // SAFETY: the `Trainer` invariant holds and the entry point follows
        // the two-call string contract `read_string` expects.
        let text = unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_trainer_decision_info(self.raw.as_ptr(), buffer, n_buffer, out)
            })?
        };
        serde_json::from_str(&text).map_err(|error| {
            Error::runtime(format!(
                "the runtime described the decision head badly: {error}"
            ))
        })
    }

    /// The decision head of the loaded model, as
    /// [`Engine::decision_info`](crate::Engine::decision_info) reads it.
    pub fn decision_info(&self) -> Result<DecisionInfo> {
        let metadata = self.decision_metadata()?;
        Ok(decision_info_from(
            self,
            &metadata,
            self.decision_template_overridden,
        )?)
    }

    /// Renders `source` in place of the model's `systemone` template from now
    /// on, or goes back to the model's with `None`. Parsed here: a template
    /// that does not parse is refused before anything is rendered.
    pub fn set_decision_template(&mut self, source: Option<&str>) -> Result<()> {
        let name = CString::new(DECISION_TEMPLATE).map_err(nul_error)?;
        let source = source
            .map(|source| CString::new(source).map_err(nul_error))
            .transpose()?;
        // SAFETY: the `Trainer` invariant holds and all borrowed arguments live
        // through this synchronous call.
        self.check(unsafe {
            ffi::retro_trainer_set_named_template(
                self.raw.as_ptr(),
                name.as_ptr(),
                source
                    .as_ref()
                    .map_or(std::ptr::null(), |source| source.as_ptr()),
            )
        })?;
        self.decision_template_overridden = source.is_some();
        Ok(())
    }

    /// Renders, tokenizes and lays out one variant of one question, as the
    /// engine does before evaluating it.
    pub fn decision_pass(
        &self,
        info: &DecisionInfo,
        state: &Json,
        question_id: &str,
        question: &Question,
        variant: usize,
    ) -> Result<DecisionPass> {
        Ok(decision_pass(
            self,
            info,
            state,
            question_id,
            question,
            variant,
        )?)
    }

    /// The raw scores of `passes` under the current weights, adapter included:
    /// one vector per pass, its reads in order. At most `pack` passes share a
    /// forward pass (bounded by the context's sequence slots).
    pub fn decide_passes(
        &mut self,
        info: &DecisionInfo,
        passes: &[DecisionPass],
        pack: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if passes.is_empty() {
            return Ok(Vec::new());
        }
        let prompts: Vec<ffi::RetroDecisionPrompt> =
            passes.iter().map(DecisionPass::as_ffi).collect();
        let n_scores = passes.iter().map(|pass| pass.reads.len()).sum::<usize>();
        let mut scores = vec![0.0_f32; n_scores];
        // SAFETY: the `Trainer` invariant holds; every prompt points into
        // `passes`, which outlives this synchronous call, and `scores` holds
        // the reads of every prompt.
        let code = unsafe {
            ffi::retro_trainer_decide(
                self.raw.as_ptr(),
                readout_code(&info.readout),
                prompts.as_ptr(),
                prompts.len(),
                pack,
                scores.as_mut_ptr(),
                scores.len(),
            )
        };
        if code == ffi::RETRO_DECIDE_INVALID_INPUT {
            return Err(Error::invalid(ffi::last_error()));
        }
        self.check(code)?;
        let mut out = Vec::with_capacity(passes.len());
        let mut offset = 0;
        for pass in passes {
            out.push(scores[offset..offset + pass.reads.len()].to_vec());
            offset += pass.reads.len();
        }
        Ok(out)
    }

    /// One physical step of the grouped softmax cross-entropy over `batch`.
    /// `scheduler_total_steps` pins the learning-rate horizon, in optimizer
    /// updates. The metrics' `global_step` counts the updates so far: it moves
    /// once every `batch.accumulation_steps` calls.
    pub fn train_grouped(
        &mut self,
        batch: &GroupedBatch,
        scheduler_total_steps: u64,
    ) -> Result<TrainMetrics> {
        let span = tracing::info_span!(
            target: "retrograd::engine::update",
            "update",
            algorithm = "grouped",
            questions = batch.n_groups
        );
        let _entered = span.enter();
        let data = ffi::RetroGroupedBatch {
            tokens: batch.tokens.as_ptr(),
            positions: batch.positions.as_ptr(),
            seq_ids: batch.seq_ids.as_ptr(),
            outputs: batch.outputs.as_ptr(),
            n_tokens: batch.tokens.len(),
            n_groups: batch.n_groups,
            n_opt_max: batch.n_opt_max,
            rows: batch.rows.as_ptr(),
            columns: batch.columns.as_ptr(),
            targets: batch.targets.as_ptr(),
            inv_temperature: batch.inv_temperature,
            accumulation_steps: batch.accumulation_steps,
        };
        if batch.positions.len() != batch.tokens.len()
            || batch.seq_ids.len() != batch.tokens.len()
            || batch.outputs.len() != batch.tokens.len()
            || batch.columns.len() != batch.n_groups as usize
            || batch.rows.len() != batch.targets.len()
            || batch.rows.len() != batch.n_groups as usize * batch.n_opt_max as usize
        {
            return Err(Error::invalid("inconsistent grouped batch shape"));
        }
        let mut metrics = ffi::RetroTrainMetrics::default();
        // SAFETY: the `Trainer` invariant holds; every array the batch points
        // into is owned by `batch`, which outlives this synchronous call, and
        // their lengths were checked above.
        let code = unsafe {
            ffi::retro_trainer_train_grouped(
                self.raw.as_ptr(),
                &data,
                scheduler_total_steps,
                &mut metrics,
            )
        };
        if code == ffi::RETRO_DECIDE_INVALID_INPUT {
            return Err(Error::invalid(ffi::last_error()));
        }
        self.check(code)?;
        Ok(train_metrics_from_ffi(metrics))
    }
}
