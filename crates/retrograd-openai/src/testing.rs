//! A model that needs no GGUF, for the tests of every crate that serves this
//! contract.
//!
//! Byte-level: a token is a byte, the reply is fixed, and end of generation is
//! the token `-1`. That is enough to drive every branch between the request and
//! the response - rendering, context bound, stop, finish reason, parsing -
//! without a device. The fake serves through a
//! [`FakeBackend`] scripted with the reply, so the
//! whole session path (scheduler, event stream, collection) is exercised.
//!
//! [`FakeLoader::decision`] loads a fake decision model instead: a `laya`-like
//! head whose first option always scores highest, answered through the real
//! answer formatting, so the `/v1/systemone` contract is tested end to end.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use retrograd_core::{Result as CoreResult, SamplingParams};
use retrograd_decision::answer::{self, Calibration};
use retrograd_decision::layout::Budget;
use retrograd_decision::readout::{DecisionType, Head, Readout};
use retrograd_decision::request::SystemOneRequest;
use retrograd_engine::{Decision, DecisionInfo, EngineMemory, StepOutput, StepRow, TemplateSource};
use retrograd_infer::{
    AdmissionMode, DecodeBackend, FakeBackend, InferError, SchedulerOptions, SharedFake,
};

use crate::backend::{Loader, ServeBackend, ServingBackend};
use crate::error::OpenAiError;
use crate::render::{ChatModel, PromptModel};
use crate::session::WeightsSpec;

/// The end-of-generation token of [`FakeModel`].
pub const EOG: i32 = -1;

/// The fake's vocabulary: one token per byte.
const VOCAB: u32 = 256;

/// Answers every prompt with the same reply, byte for byte.
pub struct FakeModel {
    reply: String,
    context: usize,
    samples: usize,
    /// What the last `format_chat` was handed, so a test can read the prompt
    /// the model was actually given.
    last: Option<(String, Option<String>)>,
}

impl FakeModel {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            context: 1 << 16,
            samples: 0,
            last: None,
        }
    }

    pub fn with_context(mut self, context: usize) -> Self {
        self.context = context;
        self
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn last_messages(&self) -> Option<String> {
        self.last.as_ref().map(|(messages, _)| messages.clone())
    }

    pub fn last_tools(&self) -> Option<String> {
        self.last.as_ref().and_then(|(_, tools)| tools.clone())
    }
}

impl PromptModel for FakeModel {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        Ok(false)
    }

    fn tool_call_parser(&self, _tools_json: &str) -> CoreResult<Option<String>> {
        Ok(None)
    }

    fn tool_call_grammar(
        &self,
        _tools_json: &str,
        _constraint: &crate::request::ToolConstraint,
    ) -> CoreResult<Option<String>> {
        // The fake has no template: a constrained tool call is answered with
        // a permissive grammar, and the contract tests assert the constraint
        // was requested, not the grammar's content.
        Ok(Some("root ::= tool-call\ntool-call ::= \"x\"".to_owned()))
    }

    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        // `&self`, like the trainer's: the prompt comes back to `sample` as
        // tokens, and that is where it is recorded.
        Ok(format!(
            "{messages_json}\u{1}{}",
            tools_json.unwrap_or_default()
        ))
    }

    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        Ok(text.bytes().map(i32::from).collect())
    }

    fn context_size(&self) -> CoreResult<usize> {
        Ok(self.context)
    }
}

impl ChatModel for FakeModel {
    fn sample(
        &mut self,
        prompt: &[i32],
        sampling: &SamplingParams,
        base: bool,
    ) -> CoreResult<Vec<i32>> {
        self.samples += 1;
        let rendered = prompt
            .iter()
            .map(|&token| u8::try_from(token).expect("the fake tokenizer emits bytes"))
            .collect::<Vec<_>>();
        let rendered = String::from_utf8(rendered).expect("the fake tokenizer emits UTF-8");
        let (messages, tools) = rendered
            .split_once('\u{1}')
            .expect("the fake template separates messages and tools");
        self.last = Some((
            messages.to_owned(),
            (!tools.is_empty()).then(|| tools.to_owned()),
        ));
        let reply = match base {
            true => format!("base: {}", self.reply),
            false => self.reply.clone(),
        };
        let budget = sampling.max_new_tokens as usize;
        let mut tokens = reply
            .bytes()
            .map(i32::from)
            .take(budget)
            .collect::<Vec<_>>();
        if tokens.len() < budget {
            tokens.push(EOG);
        }
        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[i32], _unparse_special: bool) -> CoreResult<String> {
        let bytes = tokens
            .iter()
            .filter(|&&token| token != EOG)
            .map(|&token| u8::try_from(token).expect("the fake tokenizer emits bytes"))
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn is_eog(&self, token: i32) -> CoreResult<bool> {
        Ok(token == EOG)
    }
}

/// A [`ServeBackend`] over a scripted [`FakeBackend`]: the decode side answers
/// the scripted reply, the prompt-rendering side is a [`FakeModel`]. The
/// backend is shared, so a test can inspect its slot journal.
pub struct FakeServing {
    model: FakeModel,
    backend: SharedFake,
    budget_changes: Arc<Mutex<Vec<(i32, usize)>>>,
    /// Whether the fake answers embeddings; off, the endpoint's refusal path
    /// is what a model without a pooling mode gets.
    embeddings: bool,
}

/// The fake's embedding width.
pub const FAKE_EMBEDDING_SIZE: usize = 8;

impl FakeServing {
    fn build(reply: &str, with_eog: bool) -> (Self, SharedFake) {
        let mut script: Vec<i32> = reply.bytes().map(i32::from).collect();
        if with_eog {
            script.push(EOG);
        }
        let pieces: Vec<String> = (0..=u8::MAX)
            .map(|byte| char::from(byte).to_string())
            .collect();
        let n_ctx_seq = 1_u32 << 16;
        let backend = SharedFake::new(
            FakeBackend::new(4, n_ctx_seq, u64::from(n_ctx_seq) * 4, VOCAB, EOG, pieces)
                .with_script(script),
        );
        (
            Self {
                model: FakeModel::new(reply),
                backend: SharedFake(backend.0.clone()),
                budget_changes: Arc::default(),
                embeddings: true,
            },
            backend,
        )
    }

    fn new(reply: &str) -> Self {
        Self::build(reply, true).0
    }

    /// One whose script carries no end-of-generation token: the answer only
    /// stops by its budget, the context, or cancellation.
    fn endless(reply: &str) -> (Self, SharedFake) {
        Self::build(reply, false)
    }
}

impl DecodeBackend for FakeServing {
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
        self.backend.step(rows, out)
    }
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        self.backend.seq_rm(slot, p0, p1)
    }
    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
        self.backend.seq_cp(src, dst, p0, p1)
    }
    fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
        self.backend.seq_pos(slot)
    }
    fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
        self.backend.seq_save_len(slot, flags)
    }
    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> Result<usize, InferError> {
        self.backend.seq_save_into(slot, flags, buffer)
    }
    fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
        self.backend.seq_load(slot, flags, data)
    }
    fn kv_unified(&self) -> Result<bool, InferError> {
        self.backend.kv_unified()
    }
    fn weights_version(&self) -> Result<u64, InferError> {
        self.backend.weights_version()
    }
    fn model_fingerprint(&self) -> Result<u64, InferError> {
        self.backend.model_fingerprint()
    }
    fn is_hybrid(&self) -> Result<bool, InferError> {
        self.backend.is_hybrid()
    }
    fn memory(&self) -> Result<EngineMemory, InferError> {
        self.backend.memory()
    }
    fn request_free(&mut self, request: i32) -> Result<(), InferError> {
        self.backend.request_free(request)
    }
    fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
        self.backend.is_eog_token(token)
    }
    fn detokenize_bytes(
        &self,
        tokens: &[i32],
        unparse_special: bool,
    ) -> Result<Vec<u8>, InferError> {
        self.backend.detokenize_bytes(tokens, unparse_special)
    }
    fn context_size(&self) -> Result<u32, InferError> {
        self.backend.context_size()
    }
    fn vocab_size(&self) -> Result<u32, InferError> {
        self.backend.vocab_size()
    }
    fn load_adapter(&mut self, name: &str, path: &std::path::Path) -> Result<(), InferError> {
        self.backend.load_adapter(name, path)
    }
    fn activate_adapter(&mut self, name: Option<&str>) -> Result<(), InferError> {
        self.backend.activate_adapter(name)
    }
    fn active_adapter(&self) -> Result<Option<String>, InferError> {
        self.backend.active_adapter()
    }
}

impl PromptModel for FakeServing {
    fn supports_native_tools(&self) -> CoreResult<bool> {
        PromptModel::supports_native_tools(&self.model)
    }
    fn tool_call_parser(&self, tools_json: &str) -> CoreResult<Option<String>> {
        PromptModel::tool_call_parser(&self.model, tools_json)
    }
    fn tool_call_grammar(
        &self,
        tools_json: &str,
        constraint: &crate::request::ToolConstraint,
    ) -> CoreResult<Option<String>> {
        PromptModel::tool_call_grammar(&self.model, tools_json, constraint)
    }
    fn format_chat(&self, messages_json: &str, tools_json: Option<&str>) -> CoreResult<String> {
        PromptModel::format_chat(&self.model, messages_json, tools_json)
    }
    fn tokenize(&self, text: &str) -> CoreResult<Vec<i32>> {
        PromptModel::tokenize(&self.model, text)
    }
    fn context_size(&self) -> CoreResult<usize> {
        PromptModel::context_size(&self.model)
    }
}

impl ServeBackend for FakeServing {
    fn supports_embeddings(&self) -> CoreResult<bool> {
        Ok(self.embeddings)
    }

    /// A deterministic stand-in: the fake's tokens are bytes, and the vector
    /// is a fixed function of them - identical inputs give identical vectors,
    /// and a batch gives the same vectors as separate calls.
    fn embed(&mut self, tokens: &[i32]) -> CoreResult<Vec<f32>> {
        let mut out = vec![0.0_f32; FAKE_EMBEDDING_SIZE];
        if !tokens.is_empty() {
            for (index, token) in tokens.iter().enumerate() {
                out[index % FAKE_EMBEDDING_SIZE] +=
                    (u8::try_from(*token).unwrap_or(0) as f32) * (index as f32 + 1.0);
            }
            for value in &mut out {
                *value /= tokens.len() as f32;
            }
        }
        Ok(out)
    }

    /// Scores option `i` of every question `-i`: the first option wins, and
    /// the probabilities are the ones the real formatting gives those scores.
    /// A question costs one token per option plus one.
    fn decide(
        &mut self,
        info: &DecisionInfo,
        request: &SystemOneRequest,
    ) -> Result<Decision, OpenAiError> {
        let mut answers = IndexMap::with_capacity(request.questions.len());
        let mut input_tokens = 0;
        for (id, question) in &request.questions {
            let n = info.readout.n_outputs(question);
            // Exact: an option index is at most the readout's option bound.
            let scores: Vec<f32> = (0..n).map(|index| -(index as f32)).collect();
            input_tokens += n + 1;
            let answer = answer::format_answer(
                &info.readout,
                &info.calibration,
                question,
                &[scores],
                false,
            )?;
            answers.insert(id.clone(), answer);
        }
        Ok(Decision {
            answers,
            input_tokens,
        })
    }

    fn set_reasoning_budget(&mut self, budget: i32) -> CoreResult<()> {
        let sampled = self
            .backend
            .0
            .lock()
            .expect("fake backend lock")
            .steps
            .iter()
            .flatten()
            .filter(|(_, _, request)| request.is_some())
            .count();
        self.budget_changes
            .lock()
            .expect("fake budget lock")
            .push((budget, sampled));
        Ok(())
    }
}

/// Loads a [`FakeServing`] for any spec, and remembers what it was asked to
/// load.
pub struct FakeLoader {
    reply: String,
    loads: Arc<AtomicUsize>,
    specs: Arc<Mutex<Vec<WeightsSpec>>>,
    /// A backend built up front and reused by every load, so a test can inspect
    /// its slot journal across reloads.
    shared: Option<SharedFake>,
    budget_changes: Arc<Mutex<Vec<(i32, usize)>>>,
    /// Whether the fake backend answers embeddings.
    embeddings: bool,
    /// Whether the fake is loaded as a decision model.
    decision: bool,
}

/// The head of the fake decision model: a marker readout with a 256-token
/// budget, no temperatures, one question per pass.
pub fn fake_decision_info() -> DecisionInfo {
    let readout = Readout::new(
        DecisionType::named("laya").expect("laya is a decision type"),
        Head::MarkerColumn {
            marker: 0,
            sep: 1,
            marker_text: "<mask>".into(),
        },
    );
    DecisionInfo {
        decision_type: "laya".into(),
        n_options_max: readout.n_options_max(),
        readout,
        calibration: Calibration::default(),
        temperatures: Default::default(),
        budget: Some(Budget::new(256).expect("a positive budget")),
        model_max_head_tokens: Some(256),
        template: TemplateSource::Model,
        pack: 1,
    }
}

impl Clone for FakeLoader {
    fn clone(&self) -> Self {
        Self {
            reply: self.reply.clone(),
            loads: self.loads.clone(),
            specs: self.specs.clone(),
            budget_changes: self.budget_changes.clone(),
            shared: self
                .shared
                .as_ref()
                .map(|shared| SharedFake(shared.0.clone())),
            embeddings: self.embeddings,
            decision: self.decision,
        }
    }
}

impl FakeLoader {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
            loads: Arc::default(),
            specs: Arc::default(),
            budget_changes: Arc::default(),
            shared: None,
            embeddings: true,
            decision: false,
        }
    }

    /// A loader whose model is a decision model: it answers `/v1/systemone`
    /// and refuses generation and embeddings.
    pub fn decision(mut self) -> Self {
        self.decision = true;
        self
    }

    /// A loader whose backend answers no embeddings - the refusal path a
    /// model without a pooling mode gets.
    pub fn without_embeddings(mut self) -> Self {
        self.embeddings = false;
        self
    }

    /// A loader whose answer never ends on its own, returning the shared
    /// backend whose journal says which slots were returned.
    pub fn shared_endless(reply: impl Into<String>) -> (Self, SharedFake) {
        let reply = reply.into();
        let (serving, shared) = FakeServing::endless(&reply);
        let loader = Self {
            reply,
            loads: Arc::default(),
            specs: Arc::default(),
            budget_changes: Arc::default(),
            shared: Some(SharedFake(serving.backend.0.clone())),
            embeddings: true,
            decision: false,
        };
        (loader, shared)
    }

    pub fn budget_changes(&self) -> Vec<(i32, usize)> {
        self.budget_changes
            .lock()
            .expect("fake budget lock")
            .clone()
    }

    pub fn loads(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }

    pub fn specs(&self) -> Vec<WeightsSpec> {
        self.specs.lock().expect("fake loader lock").clone()
    }
}

impl Default for FakeLoader {
    fn default() -> Self {
        Self::new("hello from retrograd")
    }
}

impl Loader for FakeLoader {
    fn load(
        &self,
        spec: &WeightsSpec,
        queue: usize,
        admission: AdmissionMode,
    ) -> Result<ServingBackend, OpenAiError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.specs
            .lock()
            .expect("fake loader lock")
            .push(spec.clone());
        let backend: Box<dyn ServeBackend> = match &self.shared {
            Some(shared) => Box::new(FakeServing {
                model: FakeModel::new(&self.reply),
                backend: SharedFake(shared.0.clone()),
                budget_changes: self.budget_changes.clone(),
                embeddings: self.embeddings,
            }),
            None => {
                let mut backend = FakeServing::new(&self.reply);
                backend.budget_changes = self.budget_changes.clone();
                backend.embeddings = self.embeddings;
                Box::new(backend)
            }
        };
        Ok(ServingBackend {
            expert_history: None,
            backend,
            scheduler: SchedulerOptions {
                mode: admission,
                queue_capacity: queue.max(1),
                ..SchedulerOptions::default()
            },
            decision: self.decision.then(fake_decision_info),
        })
    }
}
