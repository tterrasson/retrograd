//! The inference policy's error type. Converted into
//! [`retrograd_core::Error`] by the `From` impl below, which lives here - in
//! the crate that owns this error - rather than in `retrograd-core`.

use retrograd_engine::EngineError;

/// Everything the scheduler, the actor and the detokenizer can refuse.
#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// What the caller passed is wrong: an empty prompt, a zero token budget,
    /// a reservation that can never fit the pool.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A computed size, count or product left the range of its type.
    #[error("arithmetic overflow: {0}")]
    Overflow(String),
    /// The KV pool cannot hold what a step or an admission needs right now.
    ///
    /// This is a *capacity* condition, not a caller mistake: the HTTP layer
    /// translates this variant into 503 + `Retry-After`. It must stay a
    /// structured variant until that translation - collapsing it into a
    /// string inside a generic error would make the status code
    /// undecidable - so match on it, never on its message.
    #[error("kv pool capacity: {needed} cells needed, {available} available")]
    KvFull {
        /// Cells the caller was trying to reserve or decode.
        needed: u64,
        /// Cells actually free in the pool.
        available: u64,
    },
    /// The admission queue is full: the actor is already holding as many
    /// requests as it was configured to queue. Also a capacity condition
    /// (same 503 + `Retry-After` translation as [`InferError::KvFull`]).
    #[error("admission queue is full (capacity {capacity})")]
    QueueFull {
        /// The configured queue capacity.
        capacity: usize,
    },
    /// The request was cancelled: its event stream was dropped, or its
    /// client stopped reading and the bounded backpressure timed out.
    #[error("request cancelled")]
    Cancelled,
    /// The decode backend refused or failed while running.
    #[error("backend error: {0}")]
    Backend(#[from] EngineError),
    /// An error from the shared core facade, carried unchanged.
    #[error(transparent)]
    Core(#[from] retrograd_core::Error),
    /// A failure broadcast to several request streams at once, where the
    /// original typed error has already been returned to its first caller.
    /// The structured variants above are never flattened into this one on
    /// the path that decides a status code.
    #[error("inference failed: {0}")]
    Failed(String),
}

impl InferError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }
}

// `Clone` is manual because the two carried error types are not `Clone`; the
// broadcast path maps them to `Failed`, which is why the capacity variants -
// the ones a status code is decided on - are the ones that clone exactly.
impl Clone for InferError {
    fn clone(&self) -> Self {
        match self {
            Self::InvalidArgument(message) => Self::InvalidArgument(message.clone()),
            Self::Overflow(message) => Self::Overflow(message.clone()),
            Self::KvFull { needed, available } => Self::KvFull {
                needed: *needed,
                available: *available,
            },
            Self::QueueFull { capacity } => Self::QueueFull {
                capacity: *capacity,
            },
            Self::Cancelled => Self::Cancelled,
            Self::Backend(error) => Self::Failed(error.to_string()),
            Self::Core(error) => Self::Failed(error.to_string()),
            Self::Failed(message) => Self::Failed(message.clone()),
        }
    }
}

impl From<InferError> for retrograd_core::Error {
    fn from(error: InferError) -> Self {
        match error {
            InferError::InvalidArgument(message) => retrograd_core::Error::InvalidArgument(message),
            InferError::Overflow(message) => retrograd_core::Error::Overflow(message),
            // The facade has no capacity variant; the HTTP translation reads
            // `InferError::KvFull`/`QueueFull` *before* this conversion, so
            // what lands here is only what a non-HTTP caller sees.
            InferError::KvFull { needed, available } => retrograd_core::Error::runtime(format!(
                "kv pool saturated: {needed} cells needed, {available} available"
            )),
            InferError::QueueFull { capacity } => retrograd_core::Error::runtime(format!(
                "admission queue saturated: capacity {capacity}"
            )),
            InferError::Cancelled => retrograd_core::Error::runtime("request cancelled"),
            InferError::Backend(error) => error.into(),
            InferError::Core(error) => error,
            InferError::Failed(message) => retrograd_core::Error::runtime(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_full_stays_a_variant_and_the_facade_keeps_it_a_runtime() {
        let error = InferError::KvFull {
            needed: 10,
            available: 2,
        };
        // The variant is matched structurally, not by message.
        assert!(matches!(
            error.clone(),
            InferError::KvFull {
                needed: 10,
                available: 2
            }
        ));
        let facade: retrograd_core::Error = error.into();
        assert!(matches!(facade, retrograd_core::Error::Runtime(_)));
        assert!(!facade.is_user_error());
    }

    #[test]
    fn every_variant_maps_into_the_core_facade() {
        let cases = [
            InferError::invalid("bad"),
            InferError::Overflow("too big".into()),
            InferError::QueueFull { capacity: 3 },
            InferError::Cancelled,
            InferError::Backend(EngineError::invalid("nope")),
            InferError::Core(retrograd_core::Error::tokenize("empty")),
            InferError::Failed("boom".into()),
        ];
        let invalid: retrograd_core::Error = cases[0].clone().into();
        assert!(matches!(invalid, retrograd_core::Error::InvalidArgument(_)));
        assert!(invalid.is_user_error());
        let overflow: retrograd_core::Error = cases[1].clone().into();
        assert!(matches!(overflow, retrograd_core::Error::Overflow(_)));
        let backend: retrograd_core::Error =
            InferError::Backend(EngineError::invalid("nope")).into();
        assert!(matches!(backend, retrograd_core::Error::InvalidArgument(_)));
        let core: retrograd_core::Error =
            InferError::Core(retrograd_core::Error::tokenize("empty")).into();
        assert!(matches!(core, retrograd_core::Error::Tokenize(_)));
        // A broadcast clone never invents a structured variant.
        assert!(matches!(cases[4].clone(), InferError::Failed(_)));
    }
}
