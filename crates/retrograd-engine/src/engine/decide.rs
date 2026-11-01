//! Decision models on the engine: a model whose `<arch>.decision.type` names a
//! readout answers typed questions (`choice`, `score`, `noul`) in one forward
//! pass per question and generates nothing.
//!
//! The logic around the pass is `retrograd-decision`'s: the request, the
//! context of the `systemone` template, the layout of the tokens, the answer.
//! The engine supplies what that crate leaves to its caller - the Jinja
//! engine, the tokenizer, the vocabulary - and the pass itself
//! (`retro_engine_decide`). The template is the one the GGUF carries under
//! `tokenizer.chat_template.systemone`, rendered as llama-server renders it, so
//! the same file gives the same prompts here and there.

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use indexmap::IndexMap;
use retrograd_decision::answer::{self, Answer, Calibration};
use retrograd_decision::layout::{self, Budget};
use retrograd_decision::readout::{Readout, Vocabulary};
use retrograd_decision::request::SystemOneRequest;
use retrograd_decision::template::{self, PromptInput};
use retrograd_decision::{DecisionError, Json, Token};
use serde::Deserialize;

use super::*;
use crate::{nul_error, read_string};

/// The name of the template a decision model is rendered with.
pub const DECISION_TEMPLATE: &str = "systemone";

/// What a GGUF declares about its decision head, read from its metadata.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct DecisionMetadata {
    /// `general.architecture`.
    pub architecture: Option<String>,
    /// `<arch>.decision.type`; `None` for a model that is not a decision model.
    pub decision_type: Option<String>,
    /// Every `<arch>.decision.*` entry, the prefix removed, as the text
    /// llama.cpp prints it - which is what its server parses the temperatures
    /// from.
    pub metadata: BTreeMap<String, String>,
    /// The names of the model's named templates (`tokenizer.chat_template.<name>`).
    pub templates: Vec<String>,
    pub has_default_template: bool,
    pub mask_token: Option<Token>,
    pub sep_token: Option<Token>,
    pub n_embd_out: i64,
}

impl DecisionMetadata {
    fn parse(text: &str) -> std::result::Result<Self, EngineError> {
        serde_json::from_str(text).map_err(|error| {
            EngineError::Runtime(format!(
                "the runtime described the decision head badly: {error}"
            ))
        })
    }

    /// `<arch>.decision.max_head_tokens`, read as llama-server reads it: an
    /// unsigned prefix of the text, zero when there is none.
    fn max_head_tokens(&self) -> usize {
        self.metadata.get("max_head_tokens").map_or(0, |text| {
            let digits: String = text
                .trim_start()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().unwrap_or(0)
        })
    }
}

/// Reads the decision metadata of a GGUF without loading its weights or
/// building a context.
pub fn decision_metadata(
    model_path: impl AsRef<std::path::Path>,
) -> retrograd_core::Result<DecisionMetadata> {
    let path = path_to_cstring(model_path.as_ref())?;
    // SAFETY: `path` lives through this synchronous call and the entry point
    // follows the two-call string contract `read_string` expects.
    let text = unsafe {
        read_string(|buffer, n_buffer, out| {
            ffi::retro_read_decision_info(path.as_ptr(), buffer, n_buffer, out)
        })?
    };
    Ok(DecisionMetadata::parse(&text)?)
}

/// Where the `systemone` template of a decision comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateSource {
    /// The model's `tokenizer.chat_template.systemone`.
    Model,
    /// A source set by [`Engine::set_decision_template`].
    Override,
}

/// Everything a decision needs to know about the loaded model.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionInfo {
    /// `<arch>.decision.type`.
    pub decision_type: String,
    pub readout: Readout,
    /// The temperatures and score biases of the model.
    pub calibration: Calibration,
    /// The `temperature.*` entries of the metadata, as written there.
    pub temperatures: BTreeMap<String, String>,
    /// The head budget of a marker readout: the model's, unless
    /// [`DecisionInfo::with_max_head_tokens`] replaced it.
    pub budget: Option<Budget>,
    /// `<arch>.decision.max_head_tokens` of the model, for a marker readout.
    pub model_max_head_tokens: Option<usize>,
    pub template: TemplateSource,
    /// The most options a question may have.
    pub n_options_max: usize,
    /// How many questions one forward pass may evaluate together; 1 unless
    /// [`DecisionInfo::with_packing`] raised it.
    pub pack: usize,
}

impl DecisionInfo {
    /// Replaces the head budget of the model, as the configuration may. Only
    /// a marker readout has one.
    pub fn with_max_head_tokens(
        mut self,
        max_head_tokens: usize,
    ) -> std::result::Result<Self, EngineError> {
        if !matches!(self.readout, Readout::MarkerColumn { .. }) {
            return Err(EngineError::invalid(format!(
                "max_head_tokens applies to a model that reads its options at markers, \
                 not to a `{}` decision model",
                self.decision_type
            )));
        }
        if max_head_tokens == 0 {
            return Err(EngineError::invalid("max_head_tokens must be positive"));
        }
        self.budget = Some(Budget::new(max_head_tokens)?);
        Ok(self)
    }

    /// Lets one forward pass evaluate up to `pack` questions (each its own
    /// sequence, within the engine's sequence slots), for throughput. The
    /// default, one, is what makes an answer independent of the other
    /// questions of its request and equal to llama-server's capture: several
    /// sequences in one pass change the shapes of the matrix products, hence
    /// the kernels' arithmetic, and a sensitive model's answers with them.
    pub fn with_packing(mut self, pack: usize) -> Self {
        self.pack = pack.max(1);
        self
    }
}

/// The answers of one request, in the order of its questions, and the tokens
/// evaluated for them (every variant of every question counted, as
/// llama-server counts `usage.input_tokens`).
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub answers: IndexMap<String, Answer>,
    pub input_tokens: usize,
}

/// The engine's vocabulary as the readouts query it. A failed tokenization
/// reads as no token at all, which every readout refuses as a label that is
/// not a single token.
struct EngineVocabulary<'a> {
    engine: &'a Engine,
    mask: Option<Token>,
    sep: Option<Token>,
}

impl Vocabulary for EngineVocabulary<'_> {
    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<Token> {
        self.engine
            .tokenize_with(text, false, parse_special)
            .unwrap_or_default()
    }

    fn mask_token(&self) -> Option<Token> {
        self.mask
    }

    fn sep_token(&self) -> Option<Token> {
        self.sep
    }

    fn token_text(&self, token: Token) -> String {
        self.engine.token_piece(token, true).unwrap_or_default()
    }
}

impl Engine {
    /// What the loaded model declares about its decision head.
    pub fn decision_metadata(&self) -> std::result::Result<DecisionMetadata, EngineError> {
        // SAFETY: the `Engine` invariant holds and the entry point follows the
        // two-call string contract `read_string` expects.
        let text = unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_engine_decision_info(self.raw.as_ptr(), buffer, n_buffer, out)
            })?
        };
        DecisionMetadata::parse(&text)
    }

    /// The decision head of the loaded model: its readout, temperatures, head
    /// budget and template. [`DecisionError::NotDecisionModel`] for a model
    /// without `<arch>.decision.type`, [`DecisionError::UnknownType`] for a
    /// type whose readout is not implemented, and
    /// [`DecisionError::InvalidModel`] for a model whose metadata a decision
    /// cannot use - the checks llama-server makes when it loads one.
    pub fn decision_info(&self) -> std::result::Result<DecisionInfo, EngineError> {
        let metadata = self.decision_metadata()?;
        let vocabulary = EngineVocabulary {
            engine: self,
            mask: metadata.mask_token,
            sep: metadata.sep_token,
        };
        let readout = Readout::from_decision_type(metadata.decision_type.as_deref(), &vocabulary)?;
        let decision_type = metadata.decision_type.clone().unwrap_or_default();
        let template = if self.decision_template_overridden {
            TemplateSource::Override
        } else if metadata
            .templates
            .iter()
            .any(|name| name == DECISION_TEMPLATE)
        {
            TemplateSource::Model
        } else {
            return Err(DecisionError::InvalidModel(format!(
                "decision model has no \"{DECISION_TEMPLATE}\" template"
            ))
            .into());
        };
        let calibration = Calibration::from_metadata(
            metadata
                .metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )?;
        let temperatures = metadata
            .metadata
            .iter()
            .filter(|(key, _)| key.starts_with("temperature."))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let (budget, model_max_head_tokens) = match readout {
            Readout::MarkerColumn { .. } => {
                let max_head_tokens = metadata.max_head_tokens();
                (Some(Budget::new(max_head_tokens)?), Some(max_head_tokens))
            }
            _ => (None, None),
        };
        let n_options_max = readout.n_options_max();
        Ok(DecisionInfo {
            decision_type,
            readout,
            calibration,
            temperatures,
            budget,
            model_max_head_tokens,
            template,
            n_options_max,
            pack: 1,
        })
    }

    /// Renders `source` in place of the model's `systemone` template from now
    /// on, or goes back to the model's with `None`. The source must take the
    /// same context as the model's; it is parsed here, so a template that does
    /// not parse is refused before any request.
    pub fn set_decision_template(
        &mut self,
        source: Option<&str>,
    ) -> std::result::Result<(), EngineError> {
        let name = CString::new(DECISION_TEMPLATE).map_err(nul_error)?;
        let source = source
            .map(|source| CString::new(source).map_err(nul_error))
            .transpose()?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live
        // through this synchronous call.
        self.check(unsafe {
            ffi::retro_engine_set_named_template(
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

    /// Renders the model's template `name` over `context`, which is the
    /// template's whole input.
    pub fn render_named_template(
        &self,
        name: &str,
        context: &Json,
    ) -> std::result::Result<String, EngineError> {
        let name = CString::new(name).map_err(nul_error)?;
        let context = CString::new(context.to_string()).map_err(nul_error)?;
        // SAFETY: the `Engine` invariant holds and all borrowed arguments live
        // through this synchronous call.
        Ok(unsafe {
            read_string(|buffer, n_buffer, out| {
                ffi::retro_engine_render_named_template(
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

    /// Tokenizes `text` with both flags explicit: `add_special` adds the
    /// model's BOS/EOS, `parse_special` reads the text of a special token as
    /// that token. An empty text tokenizes to nothing.
    pub fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> std::result::Result<Vec<Token>, EngineError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let text = CString::new(text).map_err(nul_error)?;
        let call = |tokens: *mut i32, n_tokens_max: usize, out: *mut usize| {
            // SAFETY: the `Engine` invariant holds and all borrowed arguments
            // live through this synchronous call.
            unsafe {
                ffi::retro_engine_tokenize(
                    self.raw.as_ptr(),
                    text.as_ptr(),
                    add_special,
                    parse_special,
                    tokens,
                    n_tokens_max,
                    out,
                )
            }
        };
        let mut needed = 0_usize;
        self.check(call(std::ptr::null_mut(), 0, &mut needed))?;
        if needed == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the entry point writes the `written` ids it reports.
        Ok(unsafe {
            ffi_out_vec(needed, |tokens| {
                let mut written = 0_usize;
                if call(tokens, needed, &mut written) != 0 {
                    return Err(runtime_error());
                }
                Ok(written)
            })?
        })
    }

    /// The text of one token; `special` renders a special token as its text.
    pub fn token_piece(
        &self,
        token: Token,
        special: bool,
    ) -> std::result::Result<String, EngineError> {
        // SAFETY: the `Engine` invariant holds and the entry point follows the
        // two-call string contract.
        Ok(unsafe {
            read_string_sized_lossy(32, |buffer: *mut c_char, n_buffer, out| {
                ffi::retro_engine_token_piece(
                    self.raw.as_ptr(),
                    token,
                    special,
                    buffer,
                    n_buffer,
                    out,
                )
            })?
        })
    }

    /// Answers the questions of `request` on the loaded model, with the head
    /// `info` describes ([`Engine::decision_info`], possibly with a budget
    /// override). Each question is rendered through the `systemone` template,
    /// tokenized without added special tokens, laid out by the readout and
    /// evaluated in one pass per variant; the passes of the whole request are
    /// handed to the runtime at once, which evaluates `info.pack` of them per
    /// forward pass.
    ///
    /// The request must have been parsed for `info.readout`
    /// ([`SystemOneRequest::parse`]). Images are refused with
    /// [`DecisionError::ImagesUnsupported`]: the engine has no multimodal
    /// projector.
    pub fn decide(
        &mut self,
        info: &DecisionInfo,
        request: &SystemOneRequest,
    ) -> std::result::Result<Decision, EngineError> {
        if !request.images.is_empty() {
            return Err(DecisionError::ImagesUnsupported.into());
        }
        let readout = &info.readout;
        // Read by a marker readout only, which always carries its own.
        let budget = match info.budget {
            Some(budget) => budget,
            None => Budget::new(usize::MAX)?,
        };
        let renderer = |context: &Json| self.render_named_template(DECISION_TEMPLATE, context);

        struct Pass {
            tokens: Vec<Token>,
            reads: Vec<i32>,
            column: i32,
            pointer: i32,
        }
        let mut passes = Vec::new();
        let mut input_tokens = 0_usize;
        for (id, question) in &request.questions {
            for variant in 0..readout.n_variants(question) {
                let input = PromptInput {
                    state: &request.state,
                    question_id: id,
                    question,
                    variant,
                    n_images: 0,
                    media_marker: "",
                };
                let prompt = template::render(readout, &input, &renderer)?;
                let rendered = self.tokenize_with(&prompt, false, true)?;
                let laid_out = layout::arrange(readout, &rendered, question, &budget)?;
                let reads = match readout {
                    Readout::LabelLogits { .. } => readout
                        .label_tokens(question)
                        .ok_or_else(|| {
                            DecisionError::Evaluation(
                                "the question has more outputs than the model has labels".into(),
                            )
                        })?
                        .to_vec(),
                    Readout::MarkerColumn { .. } | Readout::Pointer { .. } => laid_out
                        .markers
                        .iter()
                        .map(|&position| position_i32(position))
                        .collect::<std::result::Result<_, _>>()?,
                };
                let pointer = match laid_out.pointer {
                    Some(position) => position_i32(position)?,
                    None => -1,
                };
                // Narrowing: the column of a question type is 0, 1 or 2.
                let column = i32::try_from(question.kind.column())
                    .map_err(|_| EngineError::Overflow("decision column".into()))?;
                input_tokens = input_tokens
                    .checked_add(laid_out.tokens.len())
                    .ok_or_else(|| EngineError::Overflow("decision input tokens".into()))?;
                passes.push(Pass {
                    tokens: laid_out.tokens,
                    reads,
                    column,
                    pointer,
                });
            }
        }

        let prompts: Vec<ffi::RetroDecisionPrompt> = passes
            .iter()
            .map(|pass| ffi::RetroDecisionPrompt {
                tokens: pass.tokens.as_ptr(),
                n_tokens: pass.tokens.len(),
                reads: pass.reads.as_ptr(),
                n_reads: pass.reads.len(),
                column: pass.column,
                pointer: pass.pointer,
            })
            .collect();
        let n_scores = passes.iter().map(|pass| pass.reads.len()).sum::<usize>();
        let mut scores = vec![0.0_f32; n_scores];
        let readout_code = match readout {
            Readout::MarkerColumn { .. } => ffi::RETRO_DECIDE_MARKER_COLUMN,
            Readout::LabelLogits { .. } => ffi::RETRO_DECIDE_LABEL_LOGITS,
            Readout::Pointer { .. } => ffi::RETRO_DECIDE_POINTER,
        };
        // SAFETY: the `Engine` invariant holds; every prompt points into
        // `passes`, which outlives this synchronous call, and `scores` holds
        // the `n_reads` of every prompt.
        let code = unsafe {
            ffi::retro_engine_decide(
                self.raw.as_ptr(),
                readout_code,
                prompts.as_ptr(),
                prompts.len(),
                info.pack,
                scores.as_mut_ptr(),
                scores.len(),
            )
        };
        if code == ffi::RETRO_DECIDE_INVALID_INPUT {
            return Err(EngineError::InvalidArgument(ffi::last_error()));
        }
        self.check(code)?;

        let mut answers = IndexMap::with_capacity(request.questions.len());
        let mut offset = 0_usize;
        let mut pass = passes.iter();
        for (id, question) in &request.questions {
            let mut variants = Vec::new();
            for _ in 0..readout.n_variants(question) {
                let n = pass.next().map_or(0, |pass| pass.reads.len());
                variants.push(scores[offset..offset + n].to_vec());
                offset += n;
            }
            let answer =
                answer::format_answer(readout, &info.calibration, question, &variants, false)?;
            answers.insert(id.clone(), answer);
        }
        Ok(Decision {
            answers,
            input_tokens,
        })
    }
}

/// A token position as the runtime takes it.
fn position_i32(position: usize) -> std::result::Result<i32, EngineError> {
    i32::try_from(position)
        .map_err(|_| EngineError::Overflow(format!("decision position {position}")))
}
