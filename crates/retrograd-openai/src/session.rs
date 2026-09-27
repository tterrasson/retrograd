//! The serving session: one thread, one model, one request at a time.
//!
//! A `Trainer` never leaves the thread that created it, so the weights a chat
//! request is answered with live on a thread of their own, built on the same
//! actor shape as the rollout policy. There is no concurrency inside: one
//! device, one model, and the queue in front of the thread is the concurrency.
//!
//! The thread keeps the last weights it loaded. A request for the same
//! [`WeightsSpec`] is served without reloading; another one drops the model and
//! loads the next - there is no way to swap only an adapter yet, so a reload is
//! a full one, logged with its duration.
//!
//! The device is borrowed, not owned. The session holds a [`DeviceLease`] for as
//! long as a model is loaded, gives it back after [`SessionOptions::idle`]
//! without a request, and gives it back at once when a training run asks for it
//! ([`Session::yield_device`]): an inference nobody is using must not keep a run
//! queued.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use retrograd_core::Device;
use tokio::sync::oneshot;

use crate::convert::ChatPrompt;
use crate::error::OpenAiError;
use crate::render::{self, ChatModel, ParserCache, Sampled};

/// Everything needed to rebuild the weights one model id names. Equality is
/// what decides whether the loaded model can answer without a reload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsSpec {
    /// The base GGUF, or a full-weight export.
    pub model: PathBuf,
    /// `None` serves the model alone.
    pub adapter: Option<PathBuf>,
    pub n_ctx: u32,
    pub device: Device,
    /// The chat-template variables the run rendered its data with, as JSON.
    pub chat_template_variables: Option<String>,
}

/// Builds the model a spec names. Called on the session's thread, so what it
/// returns never has to be `Send`.
pub trait Loader: Send + Sync + 'static {
    fn load(&self, spec: &WeightsSpec) -> Result<Box<dyn ChatModel>, OpenAiError>;
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

/// Loads with a [`retrograd_engine::Trainer`], configured for inference the way
/// `retrograd chat` configures it.
pub struct TrainerLoader;

impl Loader for TrainerLoader {
    fn load(&self, spec: &WeightsSpec) -> Result<Box<dyn ChatModel>, OpenAiError> {
        let defaults = retrograd_core::TrainConfig::default();
        // Generation decodes the prompt in `n_batch` chunks inside one context,
        // so the batch only has to stay within the window.
        let n_batch = defaults.n_batch.min(spec.n_ctx);
        let config = retrograd_core::TrainConfig {
            n_ctx: spec.n_ctx,
            n_batch,
            n_ubatch: defaults.n_ubatch.min(n_batch),
            device: spec.device,
            ..defaults
        };
        let mut trainer = retrograd_engine::Trainer::new(&spec.model, config)?;
        if let Some(adapter) = &spec.adapter {
            trainer.load_lora(adapter)?;
        }
        if spec.chat_template_variables.is_some() {
            trainer.set_chat_template_variables(spec.chat_template_variables.as_deref())?;
        }
        Ok(Box::new(trainer))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SessionOptions {
    /// How long a loaded model may sit unused before it is dropped and the
    /// device given back. `None` keeps it for the life of the process.
    pub idle: Option<Duration>,
    /// Requests waiting for the thread. One more is refused with a 503.
    pub queue: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            idle: Some(Duration::from_secs(300)),
            queue: 8,
        }
    }
}

enum Envelope {
    Job(Job),
    /// Nothing to do but look at the flags: sent so a thread asleep on an idle
    /// model notices a yield without waiting out its timeout.
    Wake,
}

struct Job {
    spec: WeightsSpec,
    /// `None` loads and answers nothing.
    prompt: Option<ChatPrompt>,
    reply: oneshot::Sender<Result<Option<Sampled>, OpenAiError>>,
}

struct Loaded {
    spec: WeightsSpec,
    model: Box<dyn ChatModel>,
    parsers: ParserCache,
    /// Held, never read: dropping it is what gives the device back.
    _lease: Lease,
}

/// The handle every request goes through. Cheap to share; the thread ends when
/// the last handle is dropped.
pub struct Session {
    jobs: SyncSender<Envelope>,
    yield_requested: Arc<AtomicBool>,
    loaded: Arc<Mutex<Option<WeightsSpec>>>,
}

impl Session {
    pub fn new(
        loader: Arc<dyn Loader>,
        lease: Arc<dyn DeviceLease>,
        options: SessionOptions,
    ) -> Result<Self, OpenAiError> {
        let (jobs, inbox) = mpsc::sync_channel(options.queue.max(1));
        let yield_requested = Arc::new(AtomicBool::new(false));
        let loaded = Arc::new(Mutex::new(None));
        let worker = Worker {
            loader,
            lease,
            idle: options.idle,
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
        prompt: ChatPrompt,
    ) -> Result<Sampled, OpenAiError> {
        self.submit(spec, Some(prompt))
            .await?
            .ok_or_else(|| OpenAiError::internal("the serving thread answered without sampling"))
    }

    /// The weights currently loaded, if any.
    pub fn loaded(&self) -> Option<WeightsSpec> {
        self.loaded.lock().map(|spec| spec.clone()).unwrap_or(None)
    }

    /// Asks the thread to give the device back as soon as the request it is
    /// serving, if any, is answered. What a training run that is waiting for
    /// the device calls; the requests after it are refused by the lease, and
    /// the run gets the device.
    pub fn yield_device(&self) {
        self.yield_requested.store(true, Ordering::SeqCst);
        // Best effort: a full queue means the thread is busy and will look at
        // the flag after its current request anyway.
        let _ = self.jobs.try_send(Envelope::Wake);
    }

    async fn submit(
        &self,
        spec: WeightsSpec,
        prompt: Option<ChatPrompt>,
    ) -> Result<Option<Sampled>, OpenAiError> {
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

struct Worker {
    loader: Arc<dyn Loader>,
    lease: Arc<dyn DeviceLease>,
    idle: Option<Duration>,
    yield_requested: Arc<AtomicBool>,
    published: Arc<Mutex<Option<WeightsSpec>>>,
}

impl Worker {
    fn run(self, inbox: mpsc::Receiver<Envelope>) {
        let mut loaded: Option<Loaded> = None;
        loop {
            let envelope = match (&loaded, self.idle) {
                (Some(_), Some(idle)) => inbox.recv_timeout(idle),
                _ => inbox.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };
            match envelope {
                Ok(Envelope::Job(job)) => {
                    // The request was abandoned - a client disconnect, or its
                    // timeout - while it waited: nobody reads the answer, so
                    // neither a load nor a sample is spent on it.
                    if !job.reply.is_closed() {
                        let outcome = self.serve(&mut loaded, &job.spec, job.prompt.as_ref());
                        let _ = job.reply.send(outcome);
                    }
                }
                Ok(Envelope::Wake) => {}
                Err(RecvTimeoutError::Timeout) => {
                    tracing::info!("no chat request within the idle bound; releasing the device");
                    self.unload(&mut loaded);
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
            if self.yield_requested.swap(false, Ordering::SeqCst) && loaded.is_some() {
                tracing::info!("a training run is waiting for the device; releasing it");
                self.unload(&mut loaded);
            }
        }
    }

    fn unload(&self, loaded: &mut Option<Loaded>) {
        *loaded = None;
        if let Ok(mut published) = self.published.lock() {
            *published = None;
        }
    }

    fn serve(
        &self,
        loaded: &mut Option<Loaded>,
        spec: &WeightsSpec,
        prompt: Option<&ChatPrompt>,
    ) -> Result<Option<Sampled>, OpenAiError> {
        let current = self.ensure(loaded, spec)?;
        prompt
            .map(|prompt| {
                render::complete(current.model.as_mut(), prompt, false, &mut current.parsers)
            })
            .transpose()
    }

    fn ensure<'a>(
        &self,
        loaded: &'a mut Option<Loaded>,
        spec: &WeightsSpec,
    ) -> Result<&'a mut Loaded, OpenAiError> {
        if loaded.as_ref().is_some_and(|current| current.spec == *spec) {
            return Ok(loaded.as_mut().expect("checked just above"));
        }
        // The previous model goes first, with its lease: two sets of weights on
        // one device is exactly what the lease exists to prevent.
        self.unload(loaded);
        let lease = self.lease.acquire()?;
        let started = Instant::now();
        let model = self.loader.load(spec)?;
        tracing::info!(
            model = %spec.model.display(),
            adapter = ?spec.adapter,
            elapsed_ms = started.elapsed().as_millis(),
            "loaded weights for serving"
        );
        if let Ok(mut published) = self.published.lock() {
            *published = Some(spec.clone());
        }
        Ok(loaded.insert(Loaded {
            spec: spec.clone(),
            model,
            parsers: ParserCache::default(),
            _lease: lease,
        }))
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
            chat_template_variables: None,
        }
    }

    fn prompt() -> ChatPrompt {
        crate::convert::prepare(
            serde_json::from_value(serde_json::json!({
                "model": "m", "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap(),
        )
        .unwrap()
        .prompt
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
    async fn the_same_weights_answer_without_a_reload_and_others_replace_them() {
        let loader = FakeLoader::default();
        let session = Session::new(
            Arc::new(loader.clone()),
            Arc::new(Unshared),
            SessionOptions::default(),
        )
        .unwrap();
        session
            .complete(spec(Some("a.gguf")), prompt())
            .await
            .unwrap();
        session
            .complete(spec(Some("a.gguf")), prompt())
            .await
            .unwrap();
        assert_eq!(loader.loads(), 1);
        session.complete(spec(None), prompt()).await.unwrap();
        assert_eq!(loader.loads(), 2);
        assert_eq!(session.loaded(), Some(spec(None)));
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
        session.complete(spec(None), prompt()).await.unwrap();
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
        let error = session.complete(spec(None), prompt()).await.err().unwrap();
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
}
