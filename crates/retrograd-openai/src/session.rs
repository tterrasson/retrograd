//! The serving session: one thread, one model, many requests at once.
//!
//! The session drives a [`Scheduler`](retrograd_infer::Scheduler) over a
//! [`ServeBackend`](crate::backend::ServeBackend) - the inference `Engine` in
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
    Stats(oneshot::Sender<Option<EngineStats>>),
    /// Nothing to do but look at the flags: sent so a thread asleep on an idle
    /// model notices a yield without waiting out its timeout.
    Wake,
}

struct Job {
    spec: WeightsSpec,
    /// `None` loads and answers nothing.
    prompt: Option<GenerationRequest>,
    reply: oneshot::Sender<Result<Option<ServingJob>, OpenAiError>>,
}

/// The loaded backend with everything beside it.
struct Core {
    spec: WeightsSpec,
    scheduler: Scheduler<Box<dyn ServeBackend>>,
    parsers: ParserCache,
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

    /// Answers one request with the weights `spec` names.
    pub async fn complete(
        &self,
        spec: WeightsSpec,
        request: GenerationRequest,
    ) -> Result<Sampled, OpenAiError> {
        let job = self.submit(spec, Some(request)).await?.ok_or_else(|| {
            OpenAiError::internal("the serving thread answered without a request")
        })?;
        let prompt_tokens = job.prompt_tokens;
        let reader = job.reader;
        let collected = tokio::task::spawn_blocking(move || collect(job.stream))
            .await
            .map_err(|_| OpenAiError::internal("the answer collection was cancelled"))??;
        Ok(Sampled {
            text: collected.text,
            prompt_tokens,
            completion_tokens: collected.tokens,
            stopped_at_eog: collected.stopped,
            reader,
        })
    }

    /// The weights currently loaded, if any.
    pub fn loaded(&self) -> Option<WeightsSpec> {
        self.loaded.lock().map(|spec| spec.clone()).unwrap_or(None)
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
                prompt,
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

/// What reading one request's event stream produced.
struct Collected {
    text: String,
    /// Token events seen, EOG included when one ended the completion.
    tokens: usize,
    /// Whether the completion stopped on EOG or a stop string.
    stopped: bool,
}

/// Drains one request's stream on a blocking thread. Chat requests carry a
/// single completion (`index == 0`).
fn collect(stream: EventStream) -> Result<Collected, OpenAiError> {
    let mut text = String::new();
    let mut tokens = 0;
    let mut reason = None;
    for event in stream {
        match event {
            Event::Token(token) => {
                if token.index == 0 {
                    tokens += 1;
                    text.push_str(&token.text);
                }
            }
            Event::End {
                index: 0,
                reason: end,
                text: tail,
            } => {
                text.push_str(&tail);
                reason = Some(end);
            }
            Event::End { .. } => {}
            Event::Error(error) => return Err(error.into()),
        }
    }
    let stopped = matches!(reason, Some(FinishReason::Eog | FinishReason::Stop));
    match reason {
        Some(FinishReason::Cancelled) => Err(OpenAiError::internal(
            "the request was cancelled while it waited for the device",
        )),
        Some(_) => Ok(Collected {
            text,
            tokens,
            stopped,
        }),
        None => Err(OpenAiError::internal(
            "the serving thread stopped before the request ended",
        )),
    }
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
                            let outcome = self.serve(&mut core, &spec, prompt);
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
        // `n` completions of one prompt is the `generate_batch` shape: one
        // prefill, then one copy of the prompt per further choice.
        let sampling = vec![request.sampling; request.n as usize];
        let request = Request {
            prompt: rendered.tokens,
            sampling,
            stop: request.stop,
            n_top_logprobs: request.logprobs.unwrap_or(0),
            adapter,
            unparse_special: rendered.unparse_special,
        };
        let stream = core.scheduler.submit(request)?;
        Ok(Some(ServingJob {
            stream,
            prompt_tokens,
            reader: rendered.reader,
        }))
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
        let sampled = session.complete(spec(None), request()).await.unwrap();
        assert_eq!(sampled.text, "hello from retrograd");
        assert!(sampled.stopped_at_eog);
        // The reply plus the end-of-generation token.
        assert_eq!(sampled.completion_tokens, "hello from retrograd".len() + 1);

        let mut short = request();
        short.sampling.max_new_tokens = 5;
        let sampled = session.complete(spec(None), short).await.unwrap();
        assert_eq!(sampled.text, "hello");
        assert!(!sampled.stopped_at_eog);
        assert_eq!(sampled.completion_tokens, 5);
    }
}
