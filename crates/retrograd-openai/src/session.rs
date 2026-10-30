//! The serving session: one thread, one model, many requests at once.
//!
//! The session drives a [`Scheduler`] over a
//! [`ServeBackend`] - the inference `Engine` in
//! production - on a thread of its own, so the weights never leave it. Several
//! chat requests are batched in one decode (`Continuous` admission by default;
//! `Lockstep` on request), and each request's answer is read back from its
//! bounded event stream.
//!
//! Several LoRA checkpoints of the one base model can be resident: a request
//! names its adapter, the adapter is registered once (loading it again for
//! another request of the same file is free), and the scheduler activates it
//! per request - serialized per adapter, so a decode batch never mixes them,
//! and every switch invalidates the KV. Changing the base model is the only
//! full reload.
//!
//! The device is borrowed, not owned. The session holds a [`DeviceLease`] for
//! as long as a model is loaded, gives it back after [`SessionOptions::idle`]
//! without a request, and gives it back at once when a training run asks for
//! it ([`Session::yield_device`]): in-flight requests are finished properly,
//! the scheduler (and with it the engine's contexts and weights) is dropped on
//! the session thread, and only then is the lease released - clearing the KV
//! alone would not return the device memory to a run.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use retrograd_agent_core::parser::ToolCallParser;
use retrograd_core::Device;
use retrograd_engine::EngineMemory;
use retrograd_infer::{
    AdmissionMode, Event, EventStream, FinishReason, KvStats, Request, Scheduler,
};
use tokio::sync::oneshot;

use crate::backend::{Loader, ServeBackend, adapter_name};
use crate::error::OpenAiError;
use crate::render::{self, ParserCache, Sampled};
use crate::request::GenerationRequest;

/// Everything needed to rebuild the weights one model id names. Equality of
/// the *base* fields (model, context, device, chat-template variables) is
/// what decides whether the loaded backend can answer without a reload; the
/// adapter is switched without one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsSpec {
    /// The base GGUF, or a full-weight export.
    pub model: PathBuf,
    /// `None` serves the model alone.
    pub adapter: Option<PathBuf>,
    pub n_ctx: u32,
    pub device: Device,
    /// Sequence slots of the serving pool (the `n_seq_max` of the engine).
    /// `None` lets the loader's default geometry decide.
    pub n_seq_max: Option<u32>,
    /// The chat-template variables the run rendered its data with, as JSON.
    pub chat_template_variables: Option<String>,
}

impl WeightsSpec {
    /// Whether `other` runs on the same loaded base (adapter excluded).
    fn same_base(&self, other: &WeightsSpec) -> bool {
        self.model == other.model
            && self.n_ctx == other.n_ctx
            && self.device == other.device
            && self.n_seq_max == other.n_seq_max
            && self.chat_template_variables == other.chat_template_variables
    }
}

/// What holding the device looks like to whoever lends it. Dropped to give it
/// back.
pub type Lease = Box<dyn std::any::Any + Send>;

/// Whoever decides who may use the device. A server lends it from the permit
/// its training runs queue on; a process serving one model owns it outright.
pub trait DeviceLease: Send + Sync + 'static {
    /// Takes the device, or refuses with the reason a client can act on.
    fn acquire(&self) -> Result<Lease, OpenAiError>;
}

/// The lease of a process with nothing else to share the device with.
pub struct Unshared;

impl DeviceLease for Unshared {
    fn acquire(&self) -> Result<Lease, OpenAiError> {
        Ok(Box::new(()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SessionOptions {
    /// How long a loaded model may sit unused before it is dropped and the
    /// device given back. `None` keeps it for the life of the process.
    pub idle: Option<Duration>,
    /// Requests waiting for admission. One more is refused with a 503.
    pub queue: usize,
    /// How queued requests are admitted. Serving defaults to `Continuous` (a
    /// freed slot is refilled at the next step); `Lockstep` is for evaluation
    /// and tests that need the canonical wave order.
    pub admission: AdmissionMode,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            idle: Some(Duration::from_secs(300)),
            queue: 8,
            admission: AdmissionMode::Continuous,
        }
    }
}

/// A submitted chat request, rendered and queued: the caller reads the answer
/// off `stream` while the session thread decodes.
pub struct ServingJob {
    pub stream: EventStream,
    pub prompt_tokens: usize,
    /// The tool-call reader the rendering chose, if any.
    pub reader: Option<Arc<dyn ToolCallParser>>,
}

/// One input of an embeddings request: a text (tokenized on the session's
/// thread, with the model's own tokenizer) or a token list as the client sent
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedItem {
    Text(String),
    Tokens(Vec<i32>),
}

/// The answer to an embeddings request: one vector per input, and each
/// input's token count for the usage.
#[derive(Clone, Debug)]
pub struct EmbeddingOutcome {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: Vec<usize>,
}

/// A point-in-time report of the serving session, for a stats endpoint.
#[derive(Clone, Debug)]
pub struct EngineStats {
    /// The weights currently loaded.
    pub spec: WeightsSpec,
    /// The adapters registered on the backend, by name.
    pub adapters: Vec<String>,
    /// Prefix-cache counters.
    pub kv: KvStats,
    /// KV-pool occupancy.
    pub memory: EngineMemory,
    /// Requests waiting for admission.
    pub queued: usize,
    /// Live sequences.
    pub live: usize,
    /// The active adapter, `None` for the base model.
    pub adapter: Option<String>,
}

enum Envelope {
    Job(Job),
    /// Prefill-only embedding extraction on the loaded backend: no generation
    /// slot is taken, and live requests are drained first.
    Embed {
        spec: WeightsSpec,
        inputs: Vec<EmbedItem>,
        reply: oneshot::Sender<Result<EmbeddingOutcome, OpenAiError>>,
    },
    Stats(oneshot::Sender<Option<EngineStats>>),
    /// Nothing to do but look at the flags: sent so a thread asleep on an idle
    /// model notices a yield without waiting out its timeout.
    Wake,
}

struct Job {
    spec: WeightsSpec,
    /// `None` loads and answers nothing. Boxed: the envelope enum is sent
    /// through a bounded channel and the request is the large field.
    prompt: Option<Box<GenerationRequest>>,
    reply: oneshot::Sender<Result<Option<ServingJob>, OpenAiError>>,
}

/// The loaded backend with everything beside it.
struct Core {
    spec: WeightsSpec,
    scheduler: Scheduler<Box<dyn ServeBackend>>,
    parsers: ParserCache,
    reasoning_budget: i32,
    /// Adapter names registered on the backend.
    adapters: HashSet<String>,
    /// Held, never read: dropping it is what gives the device back - after the
    /// scheduler above, so the weights are freed first.
    _lease: Lease,
}

/// The handle every request goes through. Cheap to share; the thread ends when
/// the last handle is dropped.
pub struct Session {
    jobs: SyncSender<Envelope>,
    yield_requested: Arc<AtomicBool>,
    loaded: Arc<Mutex<Option<WeightsSpec>>>,
}

impl Core {
    /// Registers the adapter at `path` once per file - another request for the
    /// same checkpoint loads nothing - and returns its name.
    fn register(&mut self, path: &std::path::Path) -> Result<String, OpenAiError> {
        let name = adapter_name(path);
        if !self.adapters.contains(&name) {
            self.scheduler.backend_mut().load_adapter(&name, path)?;
            self.adapters.insert(name.clone());
        }
        Ok(name)
    }
}

impl Session {
    pub fn new(
        loader: Arc<dyn Loader>,
        lease: Arc<dyn DeviceLease>,
        options: SessionOptions,
    ) -> Result<Self, OpenAiError> {
        let (jobs, inbox) = mpsc::sync_channel(options.queue.max(1) + 1);
        let yield_requested = Arc::new(AtomicBool::new(false));
        let loaded = Arc::new(Mutex::new(None));
        let worker = Worker {
            loader,
            lease,
            idle: options.idle,
            queue: options.queue,
            admission: options.admission,
            yield_requested: yield_requested.clone(),
            published: loaded.clone(),
        };
        std::thread::Builder::new()
            .name("retrograd-serve".into())
            .spawn(move || worker.run(inbox))
            .map_err(|error| {
                OpenAiError::internal(format!("could not start the serving thread: {error}"))
            })?;
        Ok(Self {
            jobs,
            yield_requested,
            loaded,
        })
    }

    /// A session over [`crate::testing::FakeLoader`], for tests that need the
    /// contract and no model.
    pub fn fake() -> Self {
        Self::new(
            Arc::new(crate::testing::FakeLoader::default()),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .expect("the serving thread starts")
    }

    /// Loads `spec` now rather than at the first request, so a model that does
    /// not load fails at startup instead of on a client's first call.
    pub async fn preload(&self, spec: WeightsSpec) -> Result<(), OpenAiError> {
        self.submit(spec, None).await.map(|_| ())
    }

    /// Answers one request with the weights `spec` names: the request is
    /// submitted, its event stream drained whole, and the answers read back -
    /// one per choice when the request asked for `n > 1`.
    pub async fn complete(
        &self,
        spec: WeightsSpec,
        request: GenerationRequest,
    ) -> Result<Vec<Sampled>, OpenAiError> {
        let job = self.open(spec, request).await?;
        let prompt_tokens = job.prompt_tokens;
        let reader = job.reader;
        let collecting_reader = reader.clone();
        let collected = tokio::task::spawn_blocking(move || collect(job.stream, collecting_reader))
            .await
            .map_err(|_| OpenAiError::internal("the answer collection was cancelled"))??;
        Ok(collected
            .into_iter()
            .map(|choice| Sampled {
                text: choice.text,
                prompt_tokens,
                completion_tokens: choice.tokens,
                stopped_at_eog: choice.stopped,
                reader: reader.clone(),
                logprobs: choice.wants_logprobs.then_some(choice.logprobs),
                reasoning_tokens: choice.reasoning_tokens,
                cached_tokens: choice.cached_tokens,
            })
            .collect())
    }

    /// Submits one request with the weights `spec` names and hands back its
    /// live event stream, for a caller that streams the answer while the
    /// session thread decodes it. Dropping [`ServingJob::stream`] cancels the
    /// request: the scheduler removes it at its next step and returns its slot.
    pub async fn open(
        &self,
        spec: WeightsSpec,
        request: GenerationRequest,
    ) -> Result<ServingJob, OpenAiError> {
        self.submit(spec, Some(request))
            .await?
            .ok_or_else(|| OpenAiError::internal("the serving thread answered without a request"))
    }

    /// The weights currently loaded, if any.
    pub fn loaded(&self) -> Option<WeightsSpec> {
        self.loaded.lock().map(|spec| spec.clone()).unwrap_or(None)
    }

    /// Embeds each input with the weights `spec` names, on the session's
    /// thread: prefill-only, one vector per input, in order. The inputs are
    /// processed one at a time - no generation slot is taken, and the KV (and
    /// with it the prefix cache) is cleared around the extraction, so live
    /// requests are drained first.
    pub async fn embed(
        &self,
        spec: WeightsSpec,
        inputs: Vec<EmbedItem>,
    ) -> Result<EmbeddingOutcome, OpenAiError> {
        let (reply, answer) = oneshot::channel();
        self.jobs
            .try_send(Envelope::Embed {
                spec,
                inputs,
                reply,
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => OpenAiError::busy(
                    "queue_full",
                    "this server is already holding as many requests as it queues; retry \
                     shortly",
                    1,
                ),
                TrySendError::Disconnected(_) => {
                    OpenAiError::internal("the serving thread is no longer running")
                }
            })?;
        answer.await.map_err(|_| {
            OpenAiError::internal("the serving thread stopped before answering the request")
        })?
    }

    /// A point-in-time report of the serving thread, or `None` when no model is
    /// loaded - or when the thread does not answer within a few seconds (busy
    /// loading, or a full inbox).
    pub async fn stats(&self) -> Option<EngineStats> {
        let (reply, answer) = oneshot::channel();
        self.jobs.try_send(Envelope::Stats(reply)).ok()?;
        tokio::time::timeout(Duration::from_secs(5), answer)
            .await
            .ok()?
            .ok()
            .flatten()
    }

    /// Asks the thread to give the device back as soon as possible: in-flight
    /// requests are finished properly, queued ones are refused, then the
    /// scheduler - engine, contexts, weights - is freed on the session thread
    /// and the lease released. What a training run that is waiting for the
    /// device calls; the requests after it are refused by the lease, and the
    /// run gets the device.
    pub fn yield_device(&self) {
        self.yield_requested.store(true, Ordering::SeqCst);
        // Best effort: a full queue means the thread is busy and will look at
        // the flag after its current step anyway.
        let _ = self.jobs.try_send(Envelope::Wake);
    }

    async fn submit(
        &self,
        spec: WeightsSpec,
        prompt: Option<GenerationRequest>,
    ) -> Result<Option<ServingJob>, OpenAiError> {
        let (reply, answer) = oneshot::channel();
        self.jobs
            .try_send(Envelope::Job(Job {
                spec,
                prompt: prompt.map(Box::new),
                reply,
            }))
            .map_err(|error| match error {
                TrySendError::Full(_) => OpenAiError::busy(
                    "queue_full",
                    "this server is already holding as many chat requests as it queues; retry \
                     shortly",
                    1,
                ),
                TrySendError::Disconnected(_) => {
                    OpenAiError::internal("the serving thread is no longer running")
                }
            })?;
        answer.await.map_err(|_| {
            OpenAiError::internal("the serving thread stopped before answering the request")
        })?
    }
}

/// What reading one request's event stream produced, per choice: the stream
/// carries one completion per `index`, and every one of them must end.
struct Collected {
    text: String,
    /// Token events seen, EOG included when one ended the completion.
    tokens: usize,
    /// Whether the completion stopped on EOG or a stop string.
    stopped: bool,
    /// The per-token logprobs, when the request asked for them. Special
    /// tokens (EOG) carry no piece and are not listed.
    logprobs: Vec<render::SampledLogprob>,
    /// Whether any logprobs were requested at all.
    wants_logprobs: bool,
    /// Tokens attributed to the reasoning block: a token counts when the
    /// partial parse of the text so far grew the reasoning block.
    reasoning_tokens: usize,
    /// The reasoning length the partial parse last reported.
    reasoning_seen: usize,
    /// Prompt tokens copied from the prefix cache, from the end event.
    cached_tokens: usize,
}

/// Drains one request's stream on a blocking thread, gathering every
/// completion (`index`) the request asked for, in index order.
fn collect(
    stream: EventStream,
    reader: Option<Arc<dyn ToolCallParser>>,
) -> Result<Vec<Collected>, OpenAiError> {
    let mut choices: Vec<Option<Collected>> = Vec::new();
    let mut ended: Vec<Option<FinishReason>> = Vec::new();
    for event in stream {
        match event {
            Event::Token(token) => {
                while choices.len() <= token.index {
                    choices.push(Some(Collected {
                        text: String::new(),
                        tokens: 0,
                        stopped: false,
                        logprobs: Vec::new(),
                        wants_logprobs: token.sampled_logprob.is_some(),
                        reasoning_tokens: 0,
                        reasoning_seen: 0,
                        cached_tokens: 0,
                    }));
                    ended.push(None);
                }
                let choice = choices[token.index].as_mut().expect("resized just above");
                choice.tokens += 1;
                choice.text.push_str(&token.text);
                // Attribute this token to the reasoning block when the partial
                // parse of the text so far grew it.
                if let Some(reader) = &reader {
                    let reasoning_len = reader.parse_partial(&choice.text).reasoning.len();
                    if reasoning_len > choice.reasoning_seen {
                        choice.reasoning_tokens += 1;
                        choice.reasoning_seen = reasoning_len;
                    }
                }
                if token.sampled_logprob.is_some() {
                    choice.wants_logprobs = true;
                    // EOG and other special tokens have no piece: they carry
                    // no text, so they are not listed in the logprobs.
                    if !token.piece.is_empty() {
                        choice.logprobs.push(render::SampledLogprob {
                            piece: token.piece.clone(),
                            logprob: token.sampled_logprob.unwrap_or(0.0),
                            top: token
                                .top_logprobs
                                .iter()
                                .map(|top| (top.token, top.logprob, top.piece.clone()))
                                .collect(),
                        });
                    }
                }
            }
            Event::End {
                index,
                reason,
                text,
                cached_tokens,
            } => {
                while ended.len() <= index {
                    ended.push(None);
                    choices.push(None);
                }
                if let Some(choice) = choices[index].as_mut() {
                    choice.text.push_str(&text);
                    choice.cached_tokens = cached_tokens;
                }
                ended[index] = Some(reason);
            }
            Event::Error(error) => return Err(error.into()),
        }
    }
    let mut out = Vec::with_capacity(choices.len());
    for (index, choice) in choices.into_iter().enumerate() {
        let Some(mut choice) = choice else {
            return Err(OpenAiError::internal(
                "the serving thread stopped before the request ended",
            ));
        };
        let reason = ended.get(index).copied().flatten();
        choice.stopped = matches!(reason, Some(FinishReason::Eog | FinishReason::Stop));
        match reason {
            Some(FinishReason::Cancelled) => {
                return Err(OpenAiError::internal(
                    "the request was cancelled while it waited for the device",
                ));
            }
            Some(_) => out.push(choice),
            None => {
                return Err(OpenAiError::internal(
                    "the serving thread stopped before the request ended",
                ));
            }
        }
    }
    if out.is_empty() {
        return Err(OpenAiError::internal(
            "the serving thread stopped before the request ended",
        ));
    }
    Ok(out)
}

struct Worker {
    loader: Arc<dyn Loader>,
    lease: Arc<dyn DeviceLease>,
    idle: Option<Duration>,
    queue: usize,
    admission: AdmissionMode,
    yield_requested: Arc<AtomicBool>,
    published: Arc<Mutex<Option<WeightsSpec>>>,
}

impl Worker {
    fn run(self, inbox: mpsc::Receiver<Envelope>) {
        let mut core: Option<Core> = None;
        loop {
            // While requests are live, do not block on the inbox: pump one step
            // and look again, so new requests join the running batch.
            let live = core.as_ref().is_some_and(|core| !core.scheduler.is_idle());
            let envelope = match (live, &core, self.idle) {
                (true, _, _) => match inbox.try_recv() {
                    Ok(envelope) => Some(envelope),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                },
                (false, Some(_), Some(idle)) => match inbox.recv_timeout(idle) {
                    Ok(envelope) => Some(envelope),
                    Err(RecvTimeoutError::Timeout) => {
                        tracing::info!(
                            "no chat request within the idle bound; releasing the device"
                        );
                        self.unload(&mut core, false);
                        None
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                },
                (false, _, _) => match inbox.recv() {
                    Ok(envelope) => Some(envelope),
                    Err(_) => return,
                },
            };
            if let Some(envelope) = envelope {
                match envelope {
                    Envelope::Job(job) => {
                        // The request was abandoned - a client disconnect, or its
                        // timeout - while it waited: nobody reads the answer, so
                        // neither a load nor a sample is spent on it.
                        let Job {
                            spec,
                            prompt,
                            reply,
                        } = job;
                        if !reply.is_closed() {
                            let outcome = self.serve(&mut core, &spec, prompt.map(|boxed| *boxed));
                            let _ = reply.send(outcome);
                        }
                    }
                    Envelope::Embed {
                        spec,
                        inputs,
                        reply,
                    } => {
                        // The request was abandoned while it waited: no embed is
                        // spent on it.
                        if !reply.is_closed() {
                            let outcome = self.embed(&mut core, &spec, inputs);
                            let _ = reply.send(outcome);
                        }
                    }
                    Envelope::Stats(reply) => {
                        let _ = reply.send(self.stats(&core));
                    }
                    Envelope::Wake => {}
                }
            }
            if self.yield_requested.swap(false, Ordering::SeqCst) && core.is_some() {
                tracing::info!("a training run is waiting for the device; releasing it");
                self.unload(&mut core, true);
                continue;
            }
            if let Some(core) = &mut core
                && !core.scheduler.is_idle()
            {
                // A failed step has already failed every live request.
                let _ = core.scheduler.pump();
            }
        }
    }

    /// Drops the loaded model. `graceful` finishes the in-flight requests and
    /// refuses the queued ones first; either way the scheduler - and with it
    /// the backend's contexts and weights - is freed on this thread before the
    /// lease is released, so the device memory is actually back.
    fn unload(&self, core: &mut Option<Core>, graceful: bool) {
        let Some(mut core) = core.take() else {
            return;
        };
        if graceful {
            core.scheduler.fail_queued();
            let _ = core.scheduler.run_until_idle();
        }
        drop(core);
        self.publish(None);
    }

    fn stats(&self, core: &Option<Core>) -> Option<EngineStats> {
        let core = core.as_ref()?;
        let stats = core.scheduler.stats();
        Some(EngineStats {
            spec: core.spec.clone(),
            adapters: core.adapters.iter().cloned().collect(),
            kv: stats.kv,
            memory: stats.memory,
            queued: stats.queued,
            live: stats.live,
            adapter: stats.adapter,
        })
    }

    fn serve(
        &self,
        core: &mut Option<Core>,
        spec: &WeightsSpec,
        request: Option<GenerationRequest>,
    ) -> Result<Option<ServingJob>, OpenAiError> {
        let core = self.ensure(core, spec)?;
        let adapter = match &spec.adapter {
            Some(path) => Some(core.register(path)?),
            None => None,
        };
        let Some(request) = request else {
            return Ok(None);
        };
        let rendered =
            render::render_prompt(core.scheduler.backend(), &request, &mut core.parsers)?;
        let prompt_tokens = rendered.tokens.len();
        // The engine owns one budget plan. Drain the previous budget group
        // before replacing it, and restore the default for requests omitting it.
        let budget = request.reasoning_budget.unwrap_or(-1);
        if budget != core.reasoning_budget {
            core.scheduler.run_until_idle()?;
            if let Err(error) = core.scheduler.backend_mut().set_reasoning_budget(budget) {
                // A refused budget may already have replaced the engine plan.
                core.scheduler
                    .backend_mut()
                    .set_reasoning_budget(core.reasoning_budget)?;
                return Err(error.into());
            }
            core.reasoning_budget = budget;
        }
        // `n` completions of one prompt is the `generate_batch` shape: one
        // prefill, then one copy of the prompt per further choice. The
        // per-choice entries (derived seeds, the grammar the rendering
        // attached) come from the rendering.
        let request = Request {
            prompt: rendered.tokens,
            sampling: rendered.sampling,
            stop: request.stop,
            n_top_logprobs: request.logprobs.unwrap_or(0),
            logprobs: request.logprobs.is_some(),
            adapter,
            unparse_special: rendered.unparse_special,
            retention: request.retention,
        };
        let stream = core.scheduler.submit(request)?;
        Ok(Some(ServingJob {
            stream,
            prompt_tokens,
            reader: rendered.reader,
        }))
    }

    /// Prefill-only embedding extraction on the loaded backend. Live requests
    /// are drained first (the embedding graph differs from the decode graph,
    /// so the KV it leaves is not a generation's), the requested adapter is
    /// applied for the extraction and released after, and the inputs are
    /// embedded one at a time so no batch of decodes is queued behind them.
    fn embed(
        &self,
        core: &mut Option<Core>,
        spec: &WeightsSpec,
        inputs: Vec<EmbedItem>,
    ) -> Result<EmbeddingOutcome, OpenAiError> {
        if inputs.is_empty() {
            return Err(OpenAiError::invalid_param("input", "`input` is required"));
        }
        let core = self.ensure(core, spec)?;
        let adapter = match &spec.adapter {
            Some(path) => Some(core.register(path)?),
            None => None,
        };
        core.scheduler.run_until_idle()?;
        let previous_adapter = core.scheduler.backend().active_adapter()?;
        core.scheduler
            .backend_mut()
            .activate_adapter(adapter.as_deref())?;
        let result = (|| {
            if !core.scheduler.backend().supports_embeddings()? {
                return Err(OpenAiError::unsupported(
                    "embeddings",
                    "the model declares no pooling (a causal chat model): embeddings are not \
                     served for it",
                ));
            }
            let mut embeddings = Vec::with_capacity(inputs.len());
            let mut prompt_tokens = Vec::with_capacity(inputs.len());
            for item in inputs {
                let tokens = match item {
                    EmbedItem::Text(text) => core.scheduler.backend().tokenize(&text)?,
                    EmbedItem::Tokens(tokens) => tokens,
                };
                if tokens.is_empty() {
                    return Err(OpenAiError::invalid_param(
                        "input",
                        "an embedding input must tokenize to at least one token",
                    ));
                }
                let context = crate::render::PromptModel::context_size(core.scheduler.backend())?;
                if tokens.len() > context {
                    return Err(OpenAiError::context_length_exceeded(
                        tokens.len(),
                        0,
                        context,
                    ));
                }
                let embedding = core.scheduler.backend_mut().embed(&tokens)?;
                prompt_tokens.push(tokens.len());
                embeddings.push(embedding);
            }
            Ok(EmbeddingOutcome {
                embeddings,
                prompt_tokens,
            })
        })();
        // Restore the scheduler's adapter even after a failed extraction.
        let restored = core
            .scheduler
            .backend_mut()
            .activate_adapter(previous_adapter.as_deref());
        match result {
            Err(error) => Err(error),
            Ok(outcome) => {
                restored?;
                Ok(outcome)
            }
        }
    }

    fn ensure<'a>(
        &self,
        core: &'a mut Option<Core>,
        spec: &WeightsSpec,
    ) -> Result<&'a mut Core, OpenAiError> {
        if let Some(current) = core.as_mut().filter(|current| current.spec.same_base(spec)) {
            if current.spec != *spec {
                // An adapter switch on the same base: what is published follows.
                current.spec = spec.clone();
                self.publish(Some(spec));
            }
            return Ok(core.as_mut().expect("checked just above"));
        }
        // The previous model goes first, with its lease: two sets of weights on
        // one device is exactly what the lease exists to prevent.
        self.unload(core, false);
        let lease = self.lease.acquire()?;
        let started = Instant::now();
        let serving = self.loader.load(spec, self.queue, self.admission)?;
        let mut scheduler = Scheduler::new(serving.backend, serving.scheduler);
        scheduler.initialize()?;
        tracing::info!(
            model = %spec.model.display(),
            adapter = ?spec.adapter,
            elapsed_ms = started.elapsed().as_millis(),
            "loaded weights for serving"
        );
        self.publish(Some(spec));
        Ok(core.insert(Core {
            spec: spec.clone(),
            scheduler,
            parsers: ParserCache::default(),
            reasoning_budget: -1,
            adapters: HashSet::new(),
            _lease: lease,
        }))
    }

    fn publish(&self, spec: Option<&WeightsSpec>) {
        if let Ok(mut published) = self.published.lock() {
            *published = spec.cloned();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::testing::FakeLoader;

    fn spec(adapter: Option<&str>) -> WeightsSpec {
        WeightsSpec {
            model: "base.gguf".into(),
            adapter: adapter.map(PathBuf::from),
            n_ctx: 512,
            device: Device::Cpu,
            n_seq_max: None,
            chat_template_variables: None,
        }
    }

    fn request() -> GenerationRequest {
        crate::chat::convert::prepare(
            serde_json::from_value(serde_json::json!({
                "model": "m", "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap(),
        )
        .unwrap()
        .request
    }

    #[tokio::test]
    async fn embeddings_select_the_base_and_restore_the_schedulers_adapter() {
        let (loader, shared) = FakeLoader::shared_endless("abc");
        let session = Session::new(
            Arc::new(loader),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        let mut prompt = request();
        prompt.sampling.max_new_tokens = 2;
        session
            .complete(spec(Some("a.gguf")), prompt.clone())
            .await
            .unwrap();
        assert_eq!(shared.lock().active_adapter.as_deref(), Some("a.gguf"));
        shared.lock().adapter_ops.clear();
        session
            .embed(spec(None), vec![EmbedItem::Tokens(vec![1, 2])])
            .await
            .unwrap();
        let ops = shared.lock().adapter_ops.clone();
        assert_eq!(
            ops,
            vec![
                ("activate".into(), None),
                ("activate".into(), Some("a.gguf".into()))
            ]
        );
        assert_eq!(shared.lock().active_adapter.as_deref(), Some("a.gguf"));
        session
            .complete(spec(Some("a.gguf")), prompt)
            .await
            .unwrap();
        assert_eq!(shared.lock().active_adapter.as_deref(), Some("a.gguf"));
    }

    /// A lease that can be taken away, and counts how often it was lent.
    #[derive(Default)]
    struct Switch {
        refuse: AtomicBool,
        lent: AtomicUsize,
    }

    impl DeviceLease for Switch {
        fn acquire(&self) -> Result<Lease, OpenAiError> {
            if self.refuse.load(Ordering::SeqCst) {
                return Err(OpenAiError::busy("device_busy", "held by a run", 30));
            }
            self.lent.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(()))
        }
    }

    #[tokio::test]
    async fn the_same_weights_answer_without_a_reload_and_an_adapter_switch_does_not_reload_the_base()
     {
        let loader = FakeLoader::default();
        let session = Session::new(
            Arc::new(loader.clone()),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        session
            .complete(spec(Some("a.gguf")), request())
            .await
            .unwrap();
        session
            .complete(spec(Some("a.gguf")), request())
            .await
            .unwrap();
        assert_eq!(loader.loads(), 1, "the same checkpoint is not reloaded");
        // Another checkpoint of the same base: the adapter is registered, the
        // base is not rebuilt.
        session
            .complete(spec(Some("b.gguf")), request())
            .await
            .unwrap();
        assert_eq!(loader.loads(), 1, "an adapter change is not a base reload");
        session.complete(spec(None), request()).await.unwrap();
        assert_eq!(loader.loads(), 1, "the base model was already loaded");
        assert_eq!(
            session.loaded().map(|spec| spec.model),
            Some(PathBuf::from("base.gguf"))
        );
        let stats = session.stats().await.expect("a model is loaded");
        assert_eq!(stats.adapters.len(), 2, "both checkpoints stay registered");
    }

    #[tokio::test]
    async fn a_refused_lease_is_the_answer_and_a_yield_gives_the_device_back() {
        let loader = FakeLoader::default();
        let lease = Arc::new(Switch::default());
        let session = Session::new(
            Arc::new(loader.clone()),
            lease.clone(),
            SessionOptions::default(),
        )
        .unwrap();
        session.complete(spec(None), request()).await.unwrap();
        assert_eq!(lease.lent.load(Ordering::SeqCst), 1);

        session.yield_device();
        // The yield lands between two envelopes; a request queued behind it
        // finds the model gone and asks the lease again.
        lease.refuse.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(5);
        while session.loaded().is_some() {
            assert!(Instant::now() < deadline, "the session never yielded");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let error = session.complete(spec(None), request()).await.err().unwrap();
        assert_eq!(error.code, Some("device_busy"));
        assert_eq!(loader.loads(), 1);
    }

    #[tokio::test]
    async fn an_idle_model_is_released() {
        let session = Session::new(
            Arc::new(FakeLoader::default()),
            Arc::new(Unshared),
            SessionOptions {
                idle: Some(Duration::from_millis(20)),
                queue: 1,
                admission: AdmissionMode::Continuous,
            },
        )
        .unwrap();
        session.preload(spec(None)).await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while session.loaded().is_some() {
            assert!(Instant::now() < deadline, "the idle model was kept");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// The answers read back through the session's streams are the rendered
    /// reply, and stop at the budget.
    #[tokio::test]
    async fn answers_come_back_through_the_scheduler_streams() {
        let session = Session::new(
            Arc::new(FakeLoader::new("hello from retrograd")),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        let sampled = session
            .complete(spec(None), request())
            .await
            .unwrap()
            .pop()
            .expect("one choice");
        assert_eq!(sampled.text, "hello from retrograd");
        assert!(sampled.stopped_at_eog);
        // The reply plus the end-of-generation token.
        assert_eq!(sampled.completion_tokens, "hello from retrograd".len() + 1);

        let mut short = request();
        short.sampling.max_new_tokens = 5;
        let sampled = session
            .complete(spec(None), short)
            .await
            .unwrap()
            .pop()
            .expect("one choice");
        assert_eq!(sampled.text, "hello");
        assert!(!sampled.stopped_at_eog);
        assert_eq!(sampled.completion_tokens, 5);
    }

    /// Dropping an open stream - a client that went away - cancels the request:
    /// the scheduler removes it at its next step and the slot goes back, long
    /// before the endless answer would have finished.
    #[tokio::test]
    async fn dropping_an_open_stream_cancels_the_request_and_returns_its_slot() {
        let (loader, fake) = FakeLoader::shared_endless("x".repeat(64));
        let session = Session::new(
            Arc::new(loader),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        let mut endless = request();
        endless.sampling.max_new_tokens = 60_000;
        let job = session.open(spec(None), endless).await.unwrap();
        // Wait until the request is admitted and decoding.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !session.stats().await.is_some_and(|stats| stats.live == 1) {
            assert!(Instant::now() < deadline, "the request was never admitted");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        drop(job.stream);
        // The slot is returned at the next step.
        let deadline = Instant::now() + Duration::from_secs(5);
        while fake.lock().removed.is_empty() {
            assert!(
                Instant::now() < deadline,
                "the cancelled request kept its slot"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while session.stats().await.is_some_and(|stats| stats.live != 0) {
            assert!(
                Instant::now() < deadline,
                "the scheduler still counts the request"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    #[tokio::test]
    async fn changing_reasoning_budgets_drains_the_previous_request_and_restores_default() {
        let (loader, _) = FakeLoader::shared_endless("x");
        let session = Session::new(
            Arc::new(loader.clone()),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        let mut first = request();
        first.reasoning_budget = Some(0);
        first.sampling.max_new_tokens = 3;
        let job = session.open(spec(None), first).await.unwrap();
        let mut next = request();
        next.sampling.max_new_tokens = 1;
        session.complete(spec(None), next).await.unwrap();
        let changes = loader.budget_changes();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0], (0, 0));
        assert_eq!(
            changes[1],
            (-1, 3),
            "the first request finished before its budget was replaced"
        );
        let events: Vec<_> = job.stream.collect();
        assert!(events.iter().any(|event| matches!(
            event,
            Event::End {
                reason: FinishReason::MaxTokens,
                ..
            }
        )));
    }
}
