//! The actor: one thread owns the (non-`Send`) backend, builds it there,
//! and serves requests arriving over a bounded channel. Each request gets a
//! bounded stream of events; a client that stops reading is cancelled after
//! a bounded backpressure wait, so it never blocks the decode thread
//! indefinitely.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread::JoinHandle;

use crate::backend::DecodeBackend;
use crate::error::InferError;
use crate::kv::{KvStats, PinId};
use crate::request::{EventStream, Request};
use crate::scheduler::{
    Envelope, QueuePermit, Queued, Scheduler, SchedulerOptions, SchedulerStats,
};

/// The actor's knobs beyond the scheduler's: how many requests the inbox
/// holds before `submit` is refused.
#[derive(Clone, Debug, Default)]
pub struct ActorOptions {
    pub scheduler: SchedulerOptions,
}

/// The handle requests go through. Cheap to share; the actor thread ends
/// when the last handle is dropped (after draining) or after
/// [`ActorHandle::shutdown`].
pub struct ActorHandle {
    inbox: SyncSender<Envelope>,
    shutdown: Arc<AtomicBool>,
    event_capacity: usize,
    queue_capacity: usize,
    queued: Arc<AtomicUsize>,
    join: Option<JoinHandle<Result<(), InferError>>>,
}

impl ActorHandle {
    /// Starts the actor thread and builds `backend` on it - the backend never
    /// crosses a thread boundary, so it does not have to be `Send`.
    pub fn spawn<B, F>(build: F, options: ActorOptions) -> Result<Self, InferError>
    where
        B: DecodeBackend + 'static,
        F: FnOnce() -> Result<B, InferError> + Send + 'static,
    {
        let queue_capacity = options.scheduler.queue_capacity.max(1);
        let event_capacity = options.scheduler.event_capacity.max(1);
        let (inbox, rx) = mpsc::sync_channel(queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let flag = shutdown.clone();
        let join = std::thread::Builder::new()
            .name("retrograd-infer".into())
            .spawn(move || {
                let backend = build()?;
                let mut scheduler = Scheduler::new(backend, options.scheduler);
                scheduler.run(rx, flag)
            })
            .map_err(|error| {
                InferError::Failed(format!("could not start the actor thread: {error}"))
            })?;
        Ok(Self {
            inbox,
            shutdown,
            event_capacity,
            queue_capacity,
            queued: Arc::new(AtomicUsize::new(0)),
            join: Some(join),
        })
    }

    /// Submits one request. Refused with [`InferError::QueueFull`] when the
    /// admission queue is full (a capacity condition, not a caller error),
    /// or with an invalid-argument error when the request itself is malformed.
    pub fn submit(&self, request: Request) -> Result<EventStream, InferError> {
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(InferError::Cancelled);
        }
        request.validate()?;
        self.queued
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                (count < self.queue_capacity).then(|| count + 1)
            })
            .map_err(|_| InferError::QueueFull {
                capacity: self.queue_capacity,
            })?;
        let permit = QueuePermit(self.queued.clone());
        let (events, rx) = mpsc::sync_channel(self.event_capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let queued = Queued {
            request,
            events,
            cancelled: cancelled.clone(),
            permit: Some(permit),
        };
        self.inbox
            .try_send(Envelope::Submit(queued))
            .map_err(|error| match error {
                TrySendError::Full(_) => InferError::QueueFull {
                    capacity: self.queue_capacity,
                },
                TrySendError::Disconnected(_) => {
                    InferError::Failed("the actor thread is no longer running".into())
                }
            })?;
        Ok(EventStream::new(rx, cancelled))
    }

    /// The prefix-cache counters of the actor's scheduler (hits, prefilled vs
    /// prompt cells, evictions, swaps, tree size).
    pub fn kv_stats(&self) -> Result<KvStats, InferError> {
        Ok(self.stats()?.kv)
    }

    /// A point-in-time report of the actor's scheduler and backend (prefix-cache
    /// counters, pool occupancy, queue/live lengths, active adapter).
    pub fn stats(&self) -> Result<SchedulerStats, InferError> {
        let (reply, rx) = mpsc::channel();
        self.inbox
            .try_send(Envelope::Stats(reply))
            .map_err(|_| InferError::Failed("the actor thread is not running".into()))?;
        rx.recv()
            .map_err(|_| InferError::Failed("the actor thread is gone".into()))
    }

    /// Registers an adapter (a LoRA checkpoint of the same base model) on the
    /// backend, without applying it: per row when the backend can, as a
    /// serialized named adapter otherwise (see [`Scheduler::load_adapter`]).
    /// The actor thread runs the load, so the backend never crosses a thread
    /// boundary.
    ///
    /// [`Scheduler::load_adapter`]: crate::Scheduler::load_adapter
    pub fn load_adapter(&self, name: &str, path: &Path) -> Result<(), InferError> {
        let (reply, rx) = mpsc::channel();
        self.inbox
            .try_send(Envelope::LoadAdapter {
                name: name.to_owned(),
                path: path.to_owned(),
                reply,
            })
            .map_err(|_| InferError::Failed("the actor thread is not running".into()))?;
        rx.recv()
            .map_err(|_| InferError::Failed("the actor thread is gone".into()))?
    }

    /// Unloads the per-row adapter `name` once its live requests end; see
    /// [`Scheduler::unload_adapter`](crate::Scheduler::unload_adapter).
    pub fn unload_adapter(&self, name: &str) -> Result<(), InferError> {
        let (reply, rx) = mpsc::channel();
        self.inbox
            .try_send(Envelope::UnloadAdapter {
                name: name.to_owned(),
                reply,
            })
            .map_err(|_| InferError::Failed("the actor thread is not running".into()))?;
        rx.recv()
            .map_err(|_| InferError::Failed("the actor thread is gone".into()))?
    }

    /// Pins the prefix-tree node at exactly `prefix` (which must be resident)
    /// so eviction never releases it.
    pub fn pin_prefix(&self, prefix: &[i32]) -> Result<PinId, InferError> {
        let (reply, rx) = mpsc::channel();
        self.inbox
            .try_send(Envelope::Pin {
                prefix: prefix.to_vec(),
                reply,
            })
            .map_err(|_| InferError::Failed("the actor thread is not running".into()))?;
        rx.recv()
            .map_err(|_| InferError::Failed("the actor thread is gone".into()))?
    }

    /// Releases one pin.
    pub fn unpin(&self, id: PinId) -> Result<bool, InferError> {
        let (reply, rx) = mpsc::channel();
        self.inbox
            .try_send(Envelope::Unpin { id, reply })
            .map_err(|_| InferError::Failed("the actor thread is not running".into()))?;
        rx.recv()
            .map_err(|_| InferError::Failed("the actor thread is gone".into()))
    }

    /// Graceful shutdown: no new admissions; live requests run to their end,
    /// queued ones are answered with an error. [`ActorHandle::join`] then
    /// returns.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Best effort: a full inbox means the thread is busy and will see
        // the flag at the top of its loop anyway.
        let _ = self.inbox.try_send(Envelope::Wake);
    }

    /// Starts graceful shutdown, waits for the actor thread to finish, and
    /// returns its outcome. A
    /// backend that failed to build is reported here.
    pub fn join(mut self) -> Result<(), InferError> {
        self.shutdown();
        match self.join.take() {
            Some(handle) => match handle.join() {
                Ok(result) => result,
                Err(_) => Err(InferError::Failed("the actor thread panicked".into())),
            },
            None => Ok(()),
        }
    }
}

impl Drop for ActorHandle {
    fn drop(&mut self) {
        // The last handle going away shuts the actor down: the inbox
        // disconnects and the thread drains and exits.
        self.shutdown();
    }
}
