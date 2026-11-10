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
use retrograd_decision::readout::{
    Head, HeadMetadata, PromptScope, Readout, SpecialToken, Vocabulary,
};
use retrograd_decision::request::{Question, SystemOneRequest};
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
    /// The number of classifier labels.
    pub n_cls_out: i64,
    /// `<arch>.decision.labels`, in order.
    pub decision_labels: Vec<String>,
    /// The control and user-defined tokens, in id order.
    pub special_tokens: Vec<SpecialToken>,
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

/// The key prefix of what an adapter carries about the decision head it was
/// trained for: `adapter.decision.temperature.<name>`, the temperatures fitted
/// on the adapted weights. llama.cpp does not read them.
pub const ADAPTER_DECISION_PREFIX: &str = "adapter.decision.";

/// The metadata entries of a GGUF whose key starts with `prefix`, as the text
/// llama.cpp prints for each value (an F32 with six decimals). Only the
/// header is read.
pub fn read_gguf_metadata(
    path: impl AsRef<std::path::Path>,
    prefix: &str,
) -> retrograd_core::Result<BTreeMap<String, String>> {
    let path = path_to_cstring(path.as_ref())?;
    let prefix = CString::new(prefix).map_err(nul_error)?;
    // SAFETY: both strings live through this synchronous call and the entry
    // point follows the two-call string contract `read_string` expects.
    let text = unsafe {
        read_string(|buffer, n_buffer, out| {
            ffi::retro_gguf_read_metadata(path.as_ptr(), prefix.as_ptr(), buffer, n_buffer, out)
        })?
    };
    serde_json::from_str(&text).map_err(|error| {
        retrograd_core::Error::runtime(format!("the runtime listed the metadata badly: {error}"))
    })
}

/// All GGUF paths in the order used by the model loader. The metadata, not
/// the filename alone, determines whether the model uses shards.
pub fn gguf_shard_paths(
    model: &std::path::Path,
) -> retrograd_core::Result<Vec<std::path::PathBuf>> {
    use retrograd_core::Error;
    let metadata = read_gguf_metadata(model, "split.")?;
    let count = metadata
        .get("split.count")
        .map_or(Ok(1), |count| count.parse::<u16>())
        .map_err(|_| Error::invalid("invalid GGUF split.count"))?;
    if count <= 1 {
        return Ok(vec![model.to_path_buf()]);
    }
    let suffix = format!("-00001-of-{count:05}.gguf");
    let prefix = model
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(&suffix))
        .filter(|prefix| !prefix.is_empty())
        .ok_or_else(|| Error::invalid(format!("expected the first GGUF shard ({suffix})")))?;
    if !metadata
        .get("split.no")
        .is_some_and(|index| index.parse::<u16>() == Ok(0))
    {
        return Err(Error::invalid(
            "the first GGUF shard must have split.no = 0",
        ));
    }
    Ok((1..=count)
        .map(|index| model.with_file_name(format!("{prefix}-{index:05}-of-{count:05}.gguf")))
        .collect())
}

/// The `adapter.decision.*` entries of an adapter GGUF, the prefix removed:
/// empty for an adapter that carries none.
pub fn adapter_decision_metadata(
    path: impl AsRef<std::path::Path>,
) -> retrograd_core::Result<BTreeMap<String, String>> {
    Ok(read_gguf_metadata(path, ADAPTER_DECISION_PREFIX)?
        .into_iter()
        .filter_map(|(key, value)| {
            key.strip_prefix(ADAPTER_DECISION_PREFIX)
                .map(|key| (key.to_owned(), value))
        })
        .collect())
}

/// Writes a copy of the GGUF at `source` to `destination` with `entries` set,
/// replaced or removed (`null`), its tensors copied byte for byte. A string
/// is written as a string, an array of strings as a string array, a boolean as
/// a boolean, a number with a fraction as F32 and an integer as the integer
/// type the key already has (UINT32 for a new key). The copy is written beside
/// `destination` and renamed into place, so `destination` may be `source`.
pub fn rewrite_gguf_metadata(
    source: impl AsRef<std::path::Path>,
    destination: impl AsRef<std::path::Path>,
    entries: &serde_json::Map<String, serde_json::Value>,
) -> retrograd_core::Result<()> {
    let source = path_to_cstring(source.as_ref())?;
    let destination = path_to_cstring(destination.as_ref())?;
    let entries =
        CString::new(serde_json::Value::Object(entries.clone()).to_string()).map_err(nul_error)?;
    // SAFETY: the three strings live through this synchronous call.
    let code = unsafe {
        ffi::retro_gguf_rewrite_metadata(source.as_ptr(), destination.as_ptr(), entries.as_ptr())
    };
    if code != 0 {
        return Err(retrograd_core::Error::checkpoint(ffi::last_error()));
    }
    Ok(())
}

/// Renders a standalone decision template over its complete context, without
/// loading a model. Uses the same Jinja engine as the model-backed renderer.
pub fn render_decision_template_source(
    source: &str,
    context: &Json,
) -> std::result::Result<String, EngineError> {
    let source = CString::new(source).map_err(nul_error)?;
    let context = CString::new(context.to_string()).map_err(nul_error)?;
    // SAFETY: both strings live through this synchronous two-call string read.
    Ok(unsafe {
        read_string(|buffer, n_buffer, out| {
            ffi::retro_template_render_context(
                source.as_ptr(),
                context.as_ptr(),
                buffer,
                n_buffer,
                out,
            )
        })?
    })
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
        if !matches!(self.readout.head, Head::MarkerColumn { .. }) {
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

    /// Takes the temperatures of an adapter's metadata
    /// ([`adapter_decision_metadata`]) in place of the model's: they were
    /// fitted on the adapted weights, and the model's on others. An adapter
    /// that carries none leaves the model's in place.
    pub fn with_adapter_temperatures(
        mut self,
        adapter: &BTreeMap<String, String>,
    ) -> std::result::Result<Self, EngineError> {
        let temperatures: BTreeMap<String, String> = adapter
            .iter()
            .filter(|(key, _)| key.starts_with("temperature."))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if temperatures.is_empty() {
            return Ok(self);
        }
        self.calibration = self.calibration.with_temperatures_of(
            temperatures
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )?;
        self.temperatures = temperatures;
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

/// What a decision needs from whatever holds the model: the engine that
/// serves it, or the trainer that trains it. Both render the same template,
/// tokenize with the same flags and read the same metadata, which is what
/// makes a trained question the question the server answers.
pub(crate) trait DecisionHost {
    /// The model's `systemone` template (or its override) over `context`.
    fn render_decision(&self, context: &Json) -> std::result::Result<String, EngineError>;
    /// `text` tokenized without added special tokens; `parse_special` reads
    /// the text of a special token as that token.
    fn tokenize_decision(
        &self,
        text: &str,
        parse_special: bool,
    ) -> std::result::Result<Vec<Token>, EngineError>;
    /// The text of one token, a special one rendered as its text.
    fn token_text(&self, token: Token) -> std::result::Result<String, EngineError>;
    /// The bytes of one token, a special one rendered as its text.
    fn token_bytes(&self, token: Token) -> std::result::Result<Vec<u8>, EngineError> {
        Ok(self.token_text(token)?.into_bytes())
    }
}

/// The host's vocabulary as the readouts query it. A failed tokenization
/// reads as no token at all, which every readout refuses as a label that is
/// not a single token.
struct HostVocabulary<'a, H: ?Sized> {
    host: &'a H,
    mask: Option<Token>,
    sep: Option<Token>,
    special_tokens: &'a [SpecialToken],
}

impl<'a, H: ?Sized> HostVocabulary<'a, H> {
    /// The vocabulary a prompt is built with: the tokenizer alone.
    fn tokenizer(host: &'a H) -> Self {
        Self {
            host,
            mask: None,
            sep: None,
            special_tokens: &[],
        }
    }
}

impl<H: DecisionHost + ?Sized> Vocabulary for HostVocabulary<'_, H> {
    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<Token> {
        self.host
            .tokenize_decision(text, parse_special)
            .unwrap_or_default()
    }

    fn mask_token(&self) -> Option<Token> {
        self.mask
    }

    fn sep_token(&self) -> Option<Token> {
        self.sep
    }

    fn token_text(&self, token: Token) -> String {
        self.host.token_text(token).unwrap_or_default()
    }

    fn token_bytes(&self, token: Token) -> Vec<u8> {
        self.host.token_bytes(token).unwrap_or_default()
    }

    fn special_tokens(&self) -> Vec<SpecialToken> {
        self.special_tokens.to_vec()
    }
}

/// The decision head `metadata` describes on `host`, with the checks
/// llama-server makes when it loads one: a readout for the type, a
/// `systemone` template (the model's, unless `overridden`), positive
/// temperatures and, for a marker readout, a positive head budget.
pub(crate) fn decision_info_from<H: DecisionHost + ?Sized>(
    host: &H,
    metadata: &DecisionMetadata,
    overridden: bool,
) -> std::result::Result<DecisionInfo, EngineError> {
    let vocabulary = HostVocabulary {
        host,
        mask: metadata.mask_token,
        sep: metadata.sep_token,
        special_tokens: &metadata.special_tokens,
    };
    let head = HeadMetadata {
        labels: metadata.decision_labels.clone(),
        // A negative count is no classifier at all.
        n_classes: usize::try_from(metadata.n_cls_out).unwrap_or(0),
        n_embd_out: metadata.n_embd_out,
    };
    let readout =
        Readout::from_decision_type(metadata.decision_type.as_deref(), &head, &vocabulary)?;
    let decision_type = metadata.decision_type.clone().unwrap_or_default();
    let template = if overridden {
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
    let (budget, model_max_head_tokens) = match readout.head {
        Head::MarkerColumn { .. } => {
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

/// One forward pass of a decision: the tokens of one variant of one question
/// (or of every question of a joint prompt) as the model reads them, and
/// where its scores are read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionPass {
    pub tokens: Vec<Token>,
    /// Positions of the option markers, the label token ids of a label
    /// readout, or the decision order of each token of a joint prompt.
    pub reads: Vec<i32>,
    /// The column of the question type in the embeddings output.
    pub column: i32,
    /// The position of the query of a pointer readout, -1 otherwise.
    pub pointer: i32,
    /// The consecutive labels of each score of a label group readout, whose
    /// score is their largest logit; empty when each label is a score.
    pub label_groups: Vec<i32>,
    /// The scores the pass returns.
    pub n_scores: usize,
}

impl DecisionPass {
    pub(crate) fn as_ffi(&self) -> ffi::RetroDecisionPrompt {
        ffi::RetroDecisionPrompt {
            tokens: self.tokens.as_ptr(),
            n_tokens: self.tokens.len(),
            reads: self.reads.as_ptr(),
            n_reads: self.reads.len(),
            column: self.column,
            pointer: self.pointer,
            label_groups: if self.label_groups.is_empty() {
                std::ptr::null()
            } else {
                self.label_groups.as_ptr()
            },
            n_label_groups: self.label_groups.len(),
            n_scores: self.n_scores,
        }
    }
}

/// Renders, tokenizes and lays out one variant of one question: what the
/// server evaluates for it, and what a training step trains on. The question
/// is asked alone, about the state of the request.
pub(crate) fn decision_pass<H: DecisionHost + ?Sized>(
    host: &H,
    info: &DecisionInfo,
    state: &Json,
    question_id: &str,
    question: &Question,
    variant: usize,
) -> std::result::Result<DecisionPass, EngineError> {
    let state = template::prepare_state(&info.readout, state, &HostVocabulary::tokenizer(host))?;
    if matches!(
        info.readout.kind.prompt,
        PromptScope::Joint | PromptScope::JointQuestion
    ) {
        return joint_pass(host, info, &state, &[(question_id, question)]);
    }
    question_pass(host, info, &state, None, question_id, question, variant)
}

/// One variant of one question of a request; `state` is the prepared state
/// ([`template::prepare_state`]), `questions` every question of the request.
fn question_pass<H: DecisionHost + ?Sized>(
    host: &H,
    info: &DecisionInfo,
    state: &Json,
    questions: Option<&IndexMap<String, Question>>,
    question_id: &str,
    question: &Question,
    variant: usize,
) -> std::result::Result<DecisionPass, EngineError> {
    let readout = &info.readout;
    // Read by a marker readout only, which always carries its own.
    let budget = match info.budget {
        Some(budget) => budget,
        None => Budget::new(usize::MAX)?,
    };
    let labels = readout.question_labels(question, &HostVocabulary::tokenizer(host))?;
    let renderer = |context: &Json| host.render_decision(context);
    let input = PromptInput {
        state,
        questions,
        question_id,
        question,
        labels: &labels.texts,
        variant,
        n_images: 0,
        media_marker: "",
    };
    let prompt = template::render(readout, &input, &renderer)?;
    let laid_out = layout::arrange_prompt(readout, &prompt, &[question], &budget, |text| {
        host.tokenize_decision(text, true)
    })?;
    let (reads, n_scores) = match &readout.head {
        Head::LabelLogits { .. } | Head::LabelGroups => {
            let n_scores = if labels.groups.is_empty() {
                labels.tokens.len()
            } else {
                labels.groups.len()
            };
            (labels.tokens, n_scores)
        }
        Head::MarkerColumn { .. } | Head::D1Omni { .. } | Head::Pointer { .. } => {
            let reads = laid_out
                .markers
                .iter()
                .map(|&position| position_i32(position))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let n_scores = reads.len();
            (reads, n_scores)
        }
        Head::Pooled { n_classes } => (Vec::new(), *n_classes),
        Head::Joint => {
            return Err(DecisionError::Evaluation(
                "a joint head reads the joint prompt of its questions".into(),
            )
            .into());
        }
    };
    let label_groups = labels
        .groups
        .iter()
        .map(|&n| {
            i32::try_from(n).map_err(|_| EngineError::Overflow("decision label group".into()))
        })
        .collect::<std::result::Result<_, _>>()?;
    let pointer = match laid_out.pointer {
        Some(position) => position_i32(position)?,
        None => -1,
    };
    Ok(DecisionPass {
        tokens: laid_out.tokens,
        reads,
        column: question_column(question)?,
        pointer,
        label_groups,
        n_scores,
    })
}

/// The joint prompt of `questions`, read by a head that scores every option
/// of every question in one pass.
fn joint_pass<H: DecisionHost + ?Sized>(
    host: &H,
    info: &DecisionInfo,
    state: &Json,
    questions: &[(&str, &Question)],
) -> std::result::Result<DecisionPass, EngineError> {
    let context = template::joint_context(state, questions, 0, "")?;
    let prompt = host.render_decision(&context)?;
    let listed: Vec<&Question> = questions.iter().map(|(_, question)| *question).collect();
    let budget = Budget::new(usize::MAX)?;
    let laid_out = layout::arrange_prompt(&info.readout, &prompt, &listed, &budget, |text| {
        host.tokenize_decision(text, true)
    })?;
    Ok(DecisionPass {
        tokens: laid_out.tokens,
        reads: laid_out.order,
        column: 0,
        pointer: -1,
        label_groups: Vec::new(),
        n_scores: listed.iter().map(|question| question.options.len()).sum(),
    })
}

/// The column of the question type, as the runtime takes it.
fn question_column(question: &Question) -> std::result::Result<i32, EngineError> {
    // Narrowing: the column of a question type is 0, 1 or 2.
    i32::try_from(question.kind.column())
        .map_err(|_| EngineError::Overflow("decision column".into()))
}

/// The runtime code of a readout.
pub(crate) fn readout_code(readout: &Readout) -> i32 {
    match readout.head {
        Head::MarkerColumn { .. } | Head::D1Omni { .. } => ffi::RETRO_DECIDE_MARKER_COLUMN,
        Head::LabelLogits { .. } | Head::LabelGroups => ffi::RETRO_DECIDE_LABEL_LOGITS,
        Head::Pointer { .. } => ffi::RETRO_DECIDE_POINTER,
        Head::Joint => ffi::RETRO_DECIDE_JOINT,
        Head::Pooled { .. } => ffi::RETRO_DECIDE_POOLED,
    }
}

impl DecisionHost for Engine {
    fn render_decision(&self, context: &Json) -> std::result::Result<String, EngineError> {
        self.render_named_template(DECISION_TEMPLATE, context)
    }

    fn tokenize_decision(
        &self,
        text: &str,
        parse_special: bool,
    ) -> std::result::Result<Vec<Token>, EngineError> {
        self.tokenize_with(text, false, parse_special)
    }

    fn token_text(&self, token: Token) -> std::result::Result<String, EngineError> {
        self.token_piece(token, true)
    }

    fn token_bytes(&self, token: Token) -> std::result::Result<Vec<u8>, EngineError> {
        self.token_piece_bytes(token, true)
    }
}

impl Engine {
    /// Installs a generic letter-logit readout on a causal model. The returned
    /// standalone template is the exact source to embed in an exported GGUF.
    /// Chat-template variables must be set before calling this method.
    pub fn configure_causal_decision(
        &mut self,
    ) -> std::result::Result<(DecisionInfo, String), EngineError> {
        let mut metadata = self.decision_metadata()?;
        if metadata.decision_type.is_some() {
            return Err(EngineError::invalid(
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
        decision_info_from(self, &metadata, self.decision_template_overridden)
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

    /// The bytes of one token, a piece of a multi-byte character kept as it
    /// is; `special` renders a special token as its text.
    pub fn token_piece_bytes(
        &self,
        token: Token,
        special: bool,
    ) -> std::result::Result<Vec<u8>, EngineError> {
        // SAFETY: the `Engine` invariant holds and the entry point follows the
        // two-call string contract.
        Ok(unsafe {
            read_bytes_sized(32, |buffer: *mut c_char, n_buffer, out| {
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

    /// The forward passes that answer `request`, in order: one per variant
    /// of each question, or one for every question of a joint prompt. What
    /// [`Engine::decide`] evaluates.
    pub fn decision_passes(
        &self,
        info: &DecisionInfo,
        request: &SystemOneRequest,
    ) -> std::result::Result<Vec<DecisionPass>, EngineError> {
        if !request.images.is_empty() {
            return Err(DecisionError::ImagesUnsupported.into());
        }
        let readout = &info.readout;
        let state =
            template::prepare_state(readout, &request.state, &HostVocabulary::tokenizer(self))?;
        let mut passes = Vec::new();
        if readout.kind.prompt == PromptScope::Joint {
            let questions: Vec<(&str, &Question)> = request
                .questions
                .iter()
                .map(|(id, question)| (id.as_str(), question))
                .collect();
            passes.push(joint_pass(self, info, &state, &questions)?);
        } else {
            for (id, question) in &request.questions {
                if readout.kind.prompt == PromptScope::JointQuestion {
                    passes.push(joint_pass(self, info, &state, &[(id.as_str(), question)])?);
                    continue;
                }
                for variant in 0..readout.n_variants(question) {
                    passes.push(question_pass(
                        self,
                        info,
                        &state,
                        Some(&request.questions),
                        id,
                        question,
                        variant,
                    )?);
                }
            }
        }
        Ok(passes)
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
        let readout = &info.readout;
        let joint = readout.kind.prompt == PromptScope::Joint;
        let passes = self.decision_passes(info, request)?;
        let input_tokens = passes.iter().try_fold(0_usize, |sum, pass| {
            sum.checked_add(pass.tokens.len())
                .ok_or_else(|| EngineError::Overflow("decision input tokens".into()))
        })?;

        let prompts: Vec<ffi::RetroDecisionPrompt> =
            passes.iter().map(DecisionPass::as_ffi).collect();
        let n_scores = passes.iter().map(|pass| pass.n_scores).sum::<usize>();
        let mut scores = vec![0.0_f32; n_scores];
        // A prompt that must be alone in its forward pass is never packed.
        let pack = if readout.kind.alone_in_pass {
            1
        } else {
            info.pack
        };
        // SAFETY: the `Engine` invariant holds; every prompt points into
        // `passes`, which outlives this synchronous call, and `scores` holds
        // the scores of every prompt.
        let code = unsafe {
            ffi::retro_engine_decide(
                self.raw.as_ptr(),
                readout_code(readout),
                prompts.as_ptr(),
                prompts.len(),
                pack,
                scores.as_mut_ptr(),
                scores.len(),
            )
        };
        if code == ffi::RETRO_DECIDE_INVALID_INPUT {
            return Err(EngineError::InvalidArgument(ffi::last_error()));
        }
        self.check(code)?;

        // The scores of each question: its variants, or its options among the
        // scores of a joint prompt, in order.
        let mut answers = IndexMap::with_capacity(request.questions.len());
        let mut offset = 0_usize;
        let mut pass = passes.iter();
        for (id, question) in &request.questions {
            let mut variants = Vec::new();
            if joint {
                let n = question.options.len();
                variants.push(scores[offset..offset + n].to_vec());
                offset += n;
            } else {
                for _ in 0..readout.n_variants(question) {
                    let n = pass.next().map_or(0, |pass| pass.n_scores);
                    variants.push(scores[offset..offset + n].to_vec());
                    offset += n;
                }
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
