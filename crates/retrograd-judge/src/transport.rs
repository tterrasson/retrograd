//! The seam between the RULER protocol and whatever executes its requests.
//!
//! The protocol ([`crate::ruler`]) renders a group into [`JudgeRequest`]s and
//! parses the [`JudgeReply`]s; this module is the contract a transport
//! implements to execute them. It is deliberately *synchronous and mutable*:
//! an in-process transport wraps a `&mut` of whatever owns the weights (a
//! trainer on its own thread), which is neither `Send` nor shareable in an
//! `Arc<dyn …>`, so the transport is handed to the execution phase as `&mut`
//! instead of being captured by the judge.
//!
//! Nothing here depends on the `http` feature: the HTTP transport lives behind
//! it, and any other transport (in-process, a subprocess, a test fake) needs
//! neither `reqwest` nor `rustls`.

use retrograd_agent_core::Error;

/// One rendered judge request.
///
/// `prompt` is the raw prompt text, exactly as the protocol built it and as
/// the verdict cache keys it; the transport is free to add its own wire
/// framing around it.
#[derive(Clone, Debug)]
pub struct JudgeRequest {
    pub prompt: String,
    /// The JSON schema the answer must match, also stated in the prompt text
    /// (some endpoints render `response_format` into the prompt, so the shape
    /// has to survive in the text alone).
    pub schema: serde_json::Value,
    pub schema_name: String,
    /// Sampling temperature for this request; `None` leaves the endpoint's
    /// default, which is what the protocol asks for when the configuration
    /// states none.
    pub temperature: Option<f32>,
    /// How the transport should request structured output. A wire detail of
    /// the OpenAI chat API: transports that do not speak it may ignore the
    /// field, since the schema is in the prompt as well.
    pub response_format: Option<serde_json::Value>,
}

impl JudgeRequest {
    /// A request whose answer must match `schema`.
    pub fn structured(
        prompt: String,
        schema: serde_json::Value,
        schema_name: &str,
        temperature: Option<f32>,
    ) -> Self {
        Self {
            prompt,
            schema,
            schema_name: schema_name.to_owned(),
            temperature,
            response_format: None,
        }
    }
}

/// One raw reply: the text the judge model produced.
#[derive(Clone, Debug)]
pub struct JudgeReply {
    pub content: String,
}

/// Why a transport could not answer a batch.
#[derive(Debug, thiserror::Error)]
pub enum JudgeTransportError {
    #[error("judge transport failed: {0}")]
    Failed(String),
    #[error("judge transport returned {returned} replies for {requested} requests")]
    Arity { requested: usize, returned: usize },
}

/// The transport belongs to the judge's caller, so a transport failure reaches
/// the agent stack as a reward failure, like every other judge failure.
impl From<JudgeTransportError> for Error {
    fn from(error: JudgeTransportError) -> Self {
        Error::Reward(error.to_string())
    }
}

/// Executes batches of judge requests, one reply per request, in order.
///
/// The batch is the unit of concurrency: a transport is expected to run its
/// requests concurrently (bounded by its own configuration) and to return the
/// replies aligned with the requests. A transport that cannot answer one
/// request returns `Err` for that slot only, and the protocol decides - retry,
/// or fail the group - exactly as it does for an unparsable answer.
///
/// The trait has no `Send`/`Sync` bound on purpose: the in-process transport
/// borrows the model mutably, and the model lives on one thread.
pub trait JudgeTransport {
    fn complete(
        &mut self,
        requests: &[JudgeRequest],
    ) -> Vec<Result<JudgeReply, JudgeTransportError>>;
}
