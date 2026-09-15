//! The applicative facade: one error type returned by the crates above the
//! engine, forming the applicative half of the two-chain rule.
//!
//! What a variant is for: **the sentence the user reads must be true.** The
//! variants below distinguish configuration, input, arithmetic, tokenization,
//! dataset, checkpoint, and runtime failures so callers can branch on the
//! failure class.
//!
//! The `From` impls that feed it are **not** here, and cannot be: every crate
//! with an error type of its own sits above this one, so each writes its own
//! `impl From<TheirError> for retrograd_core::Error` and keeps the translation
//! next to the type it translates.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// What the caller passed is wrong: a CLI flag, an API field, a value out of
    /// range. The user fixes it by writing something else.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The configuration document is wrong - a field, a combination of fields,
    /// or a section this build cannot honour. Distinct from `InvalidArgument`
    /// because the fix is in a file, not on a command line.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// A computed size, count or product left the range of its type. Not the
    /// user's fault in the way an argument is: it is a shape the code cannot
    /// represent, and the message says which one.
    #[error("arithmetic overflow: {0}")]
    Overflow(String),
    /// The tokenizer refused, or produced something a training step cannot use,
    /// an empty prompt, a mask selecting nothing.
    #[error("tokenization: {0}")]
    Tokenize(String),
    /// A dataset record is malformed. The line is carried apart from the message
    /// because it is the one thing that lets the user find it.
    #[error("dataset {path}:{line}: {message}")]
    Dataset {
        path: String,
        line: usize,
        message: String,
    },
    /// A checkpoint cannot be read, written or resumed from: the file exists and
    /// says something the run cannot use.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// Everything that failed while running rather than while being read.
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// What a caller can branch on without matching a message.
///
/// The point of a facade is that the crates above it can *decide* - retry, fall
/// back, answer 4xx rather than 5xx - and a `String` inside one variant let them
/// decide nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    InvalidArgument,
    Config,
    Overflow,
    Tokenize,
    Dataset,
    Checkpoint,
    Runtime,
    Io,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    pub fn overflow(message: impl Into<String>) -> Self {
        Self::Overflow(message.into())
    }

    pub fn tokenize(message: impl Into<String>) -> Self {
        Self::Tokenize(message.into())
    }

    pub fn dataset(path: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        Self::Dataset {
            path: path.into(),
            line,
            message: message.into(),
        }
    }

    pub fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint(message.into())
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    /// The class of this error, for a caller that has to decide rather than
    /// print.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::InvalidArgument(_) => ErrorKind::InvalidArgument,
            Self::Config(_) => ErrorKind::Config,
            Self::Overflow(_) => ErrorKind::Overflow,
            Self::Tokenize(_) => ErrorKind::Tokenize,
            Self::Dataset { .. } => ErrorKind::Dataset,
            Self::Checkpoint(_) => ErrorKind::Checkpoint,
            Self::Runtime(_) => ErrorKind::Runtime,
            Self::Io(_) => ErrorKind::Io,
        }
    }

    /// Is this something the user fixes by writing something else - a flag, a
    /// field, a record - rather than by changing the machine or the code?
    ///
    /// The predicate a frontend needs to choose between "you asked for something
    /// impossible" and "it broke", and the reason the classes above are variants
    /// rather than prefixes inside one string.
    pub fn is_user_error(&self) -> bool {
        matches!(
            self.kind(),
            ErrorKind::InvalidArgument | ErrorKind::Config | ErrorKind::Dataset
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_select_the_matching_variant() {
        assert!(matches!(Error::invalid("x"), Error::InvalidArgument(_)));
        assert!(matches!(Error::runtime("x"), Error::Runtime(_)));
        assert!(matches!(Error::config("x"), Error::Config(_)));
        assert!(matches!(Error::overflow("x"), Error::Overflow(_)));
        assert!(matches!(Error::tokenize("x"), Error::Tokenize(_)));
        assert!(matches!(Error::checkpoint("x"), Error::Checkpoint(_)));
        assert!(matches!(
            Error::dataset("d.jsonl", 4, "x"),
            Error::Dataset { line: 4, .. }
        ));
    }

    #[test]
    fn display_prefixes_distinguish_the_variants() {
        // main.rs surfaces these strings to the user, so the prefixes are a
        // contract worth pinning.
        assert_eq!(
            Error::invalid("bad flag").to_string(),
            "invalid argument: bad flag"
        );
        assert_eq!(
            Error::runtime("kernel blew up").to_string(),
            "runtime error: kernel blew up"
        );
        assert_eq!(
            Error::config("lora.rank must be positive").to_string(),
            "invalid configuration: lora.rank must be positive"
        );
        assert_eq!(
            Error::overflow("critic feature size overflows usize").to_string(),
            "arithmetic overflow: critic feature size overflows usize"
        );
        assert_eq!(
            Error::tokenize("prompt tokenized to zero tokens").to_string(),
            "tokenization: prompt tokenized to zero tokens"
        );
        assert_eq!(
            Error::dataset("train.jsonl", 12, "invalid JSON").to_string(),
            "dataset train.jsonl:12: invalid JSON"
        );
    }

    /// The sentence a user reads names the thing that actually went wrong, and a
    /// frontend that routes on the class does not answer "your request was
    /// invalid" to an overflow.
    #[test]
    fn a_message_about_arithmetic_does_not_claim_a_bad_argument() {
        let error = Error::overflow("the checkpoint epoch count does not fit in the epoch counter");
        assert!(!error.to_string().contains("invalid argument"));
        assert_eq!(error.kind(), ErrorKind::Overflow);
        assert!(!error.is_user_error());
        assert!(Error::config("model.path is required").is_user_error());
    }

    #[test]
    fn io_errors_convert_and_display_with_io_prefix() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing.txt");
        let err: Error = io.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().starts_with("io error: "));
        assert!(err.to_string().contains("missing.txt"));
    }
}
