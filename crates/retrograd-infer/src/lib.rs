//! The step-wise inference policy: a scheduler, an actor and incremental
//! detokenization over any [`DecodeBackend`].
//!
//! This crate owns the *policy* of step-wise generation - who is admitted
//! and when, how a step is composed within the batch budget, when a sequence
//! is removed, how text is streamed - and nothing about its *mechanism*: the
//! backend runs one decode per step ([`DecodeBackend::step`]) plus the raw
//! sequence operations. [`Engine`](retrograd_engine::Engine) implements the
//! trait directly, a `&mut Trainer` through [`TrainerBackend`], and
//! [`FakeBackend`] is a deterministic in-memory backend for tests that need
//! no model, so the whole policy is testable without a GGUF.
//!
//! The [`ActorHandle`] spawns one thread that builds and owns the backend
//! (which therefore never has to be `Send`): requests arrive over a bounded
//! channel and each request receives a bounded stream of [`Event`]s. A slow
//! client gets bounded backpressure, then cancellation; dropping the stream
//! removes the request at the next step and returns its slot.
//!
//! The HTTP binaries that would expose this over the network live in a
//! separate facade crate depending on this one, so this crate's dependency
//! tree stays free of the protocol and of every training crate (checked by
//! `tests/chain_boundary.rs`).

mod actor;
mod backend;
mod detok;
mod error;
mod fake;
mod kv;
mod request;
mod scheduler;
mod sync;

pub use actor::{ActorHandle, ActorOptions};
pub use backend::{DecodeBackend, TrainerBackend};
pub use detok::Detokenizer;
pub use error::InferError;
pub use fake::{FakeBackend, SharedFake, sampling as fake_sampling};
pub use kv::{KvCacheKey, KvStats, PinId};
pub use request::{Event, EventStream, FinishReason, Poll, Request, TokenEvent};
pub use scheduler::{AdmissionMode, Scheduler, SchedulerOptions, SchedulerStats};
pub use sync::{Completion, RequestOutcome, run_requests, run_requests_with};
