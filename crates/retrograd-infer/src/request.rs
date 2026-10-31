//! The request and the event stream: what crosses the actor boundary.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use retrograd_core::SamplingParamsV2;

use crate::InferError;

/// One generation request: a prompt and `n` sampling parameter sets, i.e.
/// `n` completions of the same prompt (the `generate_batch` shape - one
/// prefill, then `n` `seq_cp`), plus stop strings and optional logprob output
/// per emitted token.
#[derive(Clone, Debug)]
pub struct Request {
    /// The prompt tokens (BOS included, as the tokenizer produced them).
    pub prompt: Vec<i32>,
    /// One entry per completion; at least one. Entry `i` seeds completion `i`.
    pub sampling: Vec<SamplingParamsV2>,
    /// Stop strings, matched on the detokenized text (sliding suffix). The
    /// matched text is not emitted.
    pub stop: Vec<String>,
    /// How many top-logprob pairs each token event carries; 0 for none.
    pub n_top_logprobs: u32,
    /// Also report each sampled token's own logprob (under the
    /// post-transformation distribution) even when `n_top_logprobs` is 0.
    pub logprobs: bool,
    /// The named adapter (a LoRA checkpoint of the same base model) this
    /// request runs under, or `None` for the base model. An adapter the
    /// backend serves per row shares its steps with every other per-row
    /// adapter and the base model; any other adapter serializes: one step
    /// never mixes it with anything, and switching the active adapter
    /// invalidates the KV (weights version bump). Either way the adapter is
    /// resolved once, at admission, and stays the request's to its end.
    pub adapter: Option<String>,
    /// Render special tokens as text in the emitted pieces (a template-derived
    /// tool-call parser needs the call delimiters to read anything).
    pub unparse_special: bool,
    /// Keep the finished sequence's KV in the prefix tree under a caller's
    /// key, e.g. the tail of a conversation its next turn will extend. A
    /// hint, never a guarantee: see [`Retention`].
    pub retention: Option<Retention>,
}

/// What a request asks the prefix tree to keep once it ends: its first
/// completion's sequence is pinned under `key` (pinned nodes are never
/// evicted), and the retention named `release` - the turn this one continues -
/// is let go, since the conversation's tail moved on. Retentions are bounded
/// in count (each keeps a slot dormant) and in time, and the oldest is
/// released when an admission would otherwise not fit: a released prefix
/// costs a prefill, never a wrong answer. A cancelled or failed request
/// retains nothing and releases nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retention {
    pub key: String,
    pub release: Option<String>,
}

impl Request {
    /// A single-completion request.
    pub fn single(prompt: Vec<i32>, sampling: impl Into<SamplingParamsV2>) -> Self {
        Self {
            prompt,
            sampling: vec![sampling.into()],
            stop: Vec::new(),
            n_top_logprobs: 0,
            logprobs: false,
            adapter: None,
            unparse_special: false,
            retention: None,
        }
    }

    /// Runs this request under the named adapter.
    pub fn with_adapter(mut self, adapter: Option<String>) -> Self {
        self.adapter = adapter;
        self
    }

    /// Renders special tokens as text in the emitted pieces.
    pub fn with_unparse_special(mut self, unparse_special: bool) -> Self {
        self.unparse_special = unparse_special;
        self
    }

    /// Retains the finished sequence under `retention.key`.
    pub fn with_retention(mut self, retention: Option<Retention>) -> Self {
        self.retention = retention;
        self
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
        if self.sampling.iter().any(|params| {
            !params.min_p.is_finite()
                || params.min_p < 0.0
                || params.min_p > 1.0
                || !params.repeat_penalty.is_finite()
                || params.repeat_penalty <= 0.0
                || !params.frequency_penalty.is_finite()
                || !params.presence_penalty.is_finite()
                || params.logit_bias.iter().any(|&(_, bias)| !bias.is_finite())
        }) {
            return Err(InferError::invalid(
                "min_p must be in [0, 1], repeat_penalty finite and > 0, frequency/presence \
                 penalties and logit biases finite",
            ));
        }
        if self
            .sampling
            .iter()
            .any(|params| params.logit_bias.len() > 4096)
        {
            return Err(InferError::invalid("logit_bias takes at most 4096 entries"));
        }
        if self.stop.iter().any(|stop| stop.is_empty()) {
            return Err(InferError::invalid("stop strings must not be empty"));
        }
        if self.adapter.as_ref().is_some_and(|name| name.is_empty()) {
            return Err(InferError::invalid("an adapter name must not be empty"));
        }
        if self
            .retention
            .as_ref()
            .is_some_and(|retention| retention.key.is_empty())
        {
            return Err(InferError::invalid("a retention key must not be empty"));
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
    /// The temperature-1 policy logprob of `token` (the training semantic).
    pub logprob: f32,
    /// The logprob of `token` under the post-transformation sampling
    /// distribution (after logit_bias, penalties, grammar, top_k, min_p,
    /// temperature, top_p) - the serving logprobs semantic. `None` when the
    /// request did not ask for logprobs.
    pub sampled_logprob: Option<f32>,
    /// Set when the reasoning budget forced this token, not the sampler.
    pub forced: bool,
    /// The text newly emittable after this token (may be empty: partial
    /// UTF-8 or a possible stop prefix is held back).
    pub text: String,
    /// The UTF-8 bytes of this token's own piece (special tokens render as
    /// their text when `unparse_special` is set, empty for EOG).
    pub piece: Vec<u8>,
    /// The `n_top_logprobs` entries of this position (post-transformation
    /// distribution), empty when not requested.
    pub top_logprobs: Vec<TopLogprob>,
}

/// One top-logprob entry of a sampled position: the token id, its logprob
/// under the post-transformation distribution, and the UTF-8 bytes of its
/// piece (what a client reads as `token`/`bytes` in the logprobs output).
#[derive(Clone, Debug, PartialEq)]
pub struct TopLogprob {
    pub token: i32,
    pub logprob: f32,
    pub piece: Vec<u8>,
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
    /// not to be a stop. `cached_tokens` is how many prompt tokens were
    /// copied from the prefix tree instead of prefilled (the same for every
    /// completion of the request). The stream closes once every completion
    /// has ended.
    End {
        index: usize,
        reason: FinishReason,
        text: String,
        cached_tokens: usize,
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
