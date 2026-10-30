//! Driving the scheduler synchronously, on the thread that owns the backend.
//!
//! The actor ([`ActorHandle`](crate::ActorHandle)) serves callers whose
//! backend is `Send`; a `&mut Trainer` is not, so code running on the
//! trainer's own thread drives a [`Scheduler`] directly through
//! [`Scheduler::submit`] and [`Scheduler::pump`]. [`run_requests`] packages
//! the common shape: submit a batch of requests, pump until every one of
//! them has ended, and collect each request's events into a structured
//! outcome - draining the streams as it goes, so the bounded event channels
//! never fill mid-step.

use std::collections::VecDeque;

use crate::backend::DecodeBackend;
use crate::error::InferError;
use crate::request::{Event, EventStream, FinishReason, Poll, Request};
use crate::scheduler::{Scheduler, StallGuard};

/// One completion of a request, assembled from its token events.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Completion {
    /// Sampled tokens, in order; an EOG token that ended the completion is
    /// kept, matching the batched generators.
    pub tokens: Vec<i32>,
    /// The temperature-1 policy logprob of each sampled token.
    pub logprobs: Vec<f32>,
    /// Whether the reasoning budget forced each token rather than the
    /// sampler.
    pub forced: Vec<bool>,
    /// The emitted text (a stop match is held back by the detokenizer and
    /// only appended by the final event).
    pub text: String,
    /// Why the completion ended; `None` when the request failed instead.
    pub finish: Option<FinishReason>,
}

/// What one request produced: one completion per sampling entry, in the
/// request's order - or the error that failed the request.
#[derive(Clone, Debug)]
pub struct RequestOutcome {
    pub completions: Vec<Completion>,
    pub error: Option<InferError>,
}

impl RequestOutcome {
    fn new(n: usize) -> Self {
        Self {
            completions: vec![Completion::default(); n],
            error: None,
        }
    }

    fn absorb(&mut self, event: Event) {
        match event {
            Event::Token(token) => match self.completions.get_mut(token.index) {
                Some(member) => {
                    member.tokens.push(token.token);
                    member.logprobs.push(token.logprob);
                    member.forced.push(token.forced);
                    member.text.push_str(&token.text);
                }
                None => {
                    self.error = Some(InferError::Failed(format!(
                        "a token event named completion {}, but the request has {}",
                        token.index,
                        self.completions.len()
                    )));
                }
            },
            Event::End {
                index,
                reason,
                text,
                ..
            } => match self.completions.get_mut(index) {
                Some(member) => {
                    member.finish = Some(reason);
                    member.text.push_str(&text);
                }
                None => {
                    self.error = Some(InferError::Failed(format!(
                        "an end event named completion {}, but the request has {}",
                        index,
                        self.completions.len()
                    )));
                }
            },
            Event::Error(error) => self.error = Some(error),
        }
    }
}

/// Submits `requests` in order and pumps `scheduler` until every request has
/// ended, returning one outcome per request in submission order.
///
/// Requests beyond the queue capacity are submitted as earlier ones are
/// admitted, so the queue bound never truncates a batch. A request whose
/// reservation can never fit is not an error of the call: it comes back as
/// an [`Event::Error`] in its outcome. A step failure is fatal to the call
/// and is returned.
pub fn run_requests<B: DecodeBackend>(
    scheduler: &mut Scheduler<B>,
    requests: Vec<Request>,
) -> Result<Vec<RequestOutcome>, InferError> {
    run_requests_with(scheduler, requests, |_| {})
}

/// [`run_requests`], calling `on_step` after every pump - for a caller that
/// samples the scheduler (live sequences, occupancy) while the batch runs.
pub fn run_requests_with<B: DecodeBackend>(
    scheduler: &mut Scheduler<B>,
    requests: Vec<Request>,
    mut on_step: impl FnMut(&Scheduler<B>),
) -> Result<Vec<RequestOutcome>, InferError> {
    let total = requests.len();
    let mut pending: VecDeque<Request> = requests.into();
    let mut streams: Vec<EventStream> = Vec::with_capacity(total);
    let mut outcomes: Vec<RequestOutcome> = Vec::with_capacity(total);
    let mut stall = StallGuard::default();
    loop {
        while !pending.is_empty() && scheduler.queued_len() < scheduler.queue_capacity() {
            let request = pending.pop_front().expect("pending was checked");
            let n = request.n();
            streams.push(scheduler.submit(request)?);
            outcomes.push(RequestOutcome::new(n));
        }
        if pending.is_empty() && scheduler.is_idle() {
            break;
        }
        let progressed = scheduler.pump()?;
        on_step(scheduler);
        // Drain what the step produced, so the bounded event channels of a
        // synchronous caller never fill and trigger the backpressure timeout.
        for (stream, outcome) in streams.iter().zip(&mut outcomes) {
            while let Poll::Ready(event) = stream.poll() {
                outcome.absorb(event);
            }
        }
        stall.observe(progressed)?;
    }
    // The streams are closed by now (every sender dropped with its member);
    // collect whatever the channel still buffers.
    for (stream, outcome) in streams.iter().zip(&mut outcomes) {
        while let Some(event) = stream.recv() {
            outcome.absorb(event);
        }
    }
    Ok(outcomes)
}
