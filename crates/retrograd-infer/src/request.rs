//! The request and the event stream: what crosses the actor boundary.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use retrograd_core::SamplingParams;

use crate::InferError;

/// One generation request: a prompt and `n` sampling parameter sets, i.e.
/// `n` completions of the same prompt (the `generate_batch` shape - one
/// prefill, then `n` `seq_cp`), plus stop strings and an optional top-logprob
/// count per emitted token.
#[derive(Clone, Debug)]
pub struct Request {
    /// The prompt tokens (BOS included, as the tokenizer produced them).
    pub prompt: Vec<i32>,
    /// One entry per completion; at least one. Entry `i` seeds completion `i`.
    pub sampling: Vec<SamplingParams>,
    /// Stop strings, matched on the detokenized text (sliding suffix). The
    /// matched text is not emitted.
    pub stop: Vec<String>,
    /// How many top-logprob pairs each token event carries; 0 for none.
    pub n_top_logprobs: u32,
}

impl Request {
    /// A single-completion request.
    pub fn single(prompt: Vec<i32>, sampling: SamplingParams) -> Self {
        Self {
            prompt,
            sampling: vec![sampling],
            stop: Vec::new(),
            n_top_logprobs: 0,
        }
    }

    /// The number of completions this request asks for.
    pub fn n(&self) -> usize {
        self.sampling.len()
    }

    /// Cheap checks that need no backend: the actor-side checks that need the
    /// pool geometry (reservation vs capacity) happen at admission.
    pub fn validate(&self) -> Result<(), InferError> {
        if self.prompt.is_empty() {
            return Err(InferError::invalid("a request requires a non-empty prompt"));
        }
        if self.sampling.is_empty() {
            return Err(InferError::invalid(
                "a request requires at least one sampling entry",
            ));
        }
        if self
            .sampling
            .iter()
            .any(|params| params.max_new_tokens == 0)
        {
            return Err(InferError::invalid(
                "max_new_tokens must be greater than zero",
            ));
        }
        if self
            .sampling
            .iter()
            .any(|params| !params.temperature.is_finite() || params.temperature <= 0.0)
        {
            return Err(InferError::invalid(
                "temperature must be finite and greater than zero",
            ));
        }
        if self
            .sampling
            .iter()
            .any(|params| !(params.top_p > 0.0 && params.top_p <= 1.0))
        {
            return Err(InferError::invalid("top_p must be in (0, 1]"));
        }
        if self.stop.iter().any(|stop| stop.is_empty()) {
            return Err(InferError::invalid("stop strings must not be empty"));
        }
        Ok(())
    }
}

/// One sampled token and its incremental text.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenEvent {
    /// Which completion of the request this token belongs to.
    pub index: usize,
    pub token: i32,
    pub logprob: f32,
    /// Set when the reasoning budget forced this token, not the sampler.
    pub forced: bool,
    /// The text newly emittable after this token (may be empty: partial
    /// UTF-8 or a possible stop prefix is held back).
    pub text: String,
    /// The `n_top_logprobs` pairs of this position, empty when not requested.
    pub top_logprobs: Vec<(i32, f32)>,
}

/// Why a completion ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An end-of-generation token was sampled (kept in the token events).
    Eog,
    /// A stop string completed in the text.
    Stop,
    /// The request's `max_new_tokens` budget was reached.
    MaxTokens,
    /// The per-sequence context was exhausted before the budget.
    ContextFull,
    /// The stream was dropped (client disconnect) or the bounded
    /// backpressure timed out.
    Cancelled,
}

/// What one request's stream carries.
#[derive(Clone, Debug)]
pub enum Event {
    /// A sampled token.
    Token(TokenEvent),
    /// One completion ended; `text` is the held-back tail that turned out
    /// not to be a stop. The stream closes once every completion has ended.
    End {
        index: usize,
        reason: FinishReason,
        text: String,
    },
    /// The request failed (refused reservation, backend failure). The stream
    /// closes after an error.
    Error(InferError),
}

/// The result of a non-blocking receive: an event, an empty but open stream,
/// or a closed one (every sender dropped - the actor removed the request).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poll<T> {
    Ready(T),
    Empty,
    Closed,
}

/// The bounded receiving end of one request. Dropping it cancels the request:
/// the scheduler removes it at the next step and returns its slot.
pub struct EventStream {
    rx: mpsc::Receiver<Event>,
    cancelled: Arc<AtomicBool>,
}

impl EventStream {
    pub(crate) fn new(rx: mpsc::Receiver<Event>, cancelled: Arc<AtomicBool>) -> Self {
        Self { rx, cancelled }
    }

    /// Blocks until the next event or the end of the stream.
    pub fn recv(&self) -> Option<Event> {
        self.rx.recv().ok()
    }

    /// The next event if one is already buffered; `None` when the stream is
    /// empty or closed.
    pub fn try_recv(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    /// Non-blocking receive that tells an empty stream from a closed one.
    pub fn poll(&self) -> Poll<Event> {
        match self.rx.try_recv() {
            Ok(event) => Poll::Ready(event),
            Err(mpsc::TryRecvError::Empty) => Poll::Empty,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Closed,
        }
    }

    /// Whether the scheduler has seen this stream dropped (cancellation
    /// requested, removal at the next step).
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Iterator for EventStream {
    type Item = Event;

    fn next(&mut self) -> Option<Event> {
        self.rx.recv().ok()
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        // The scheduler checks this flag before composing each step: the
        // request is removed at the next step, its slot returned.
        self.cancelled.store(true, Ordering::SeqCst);
    }
}
