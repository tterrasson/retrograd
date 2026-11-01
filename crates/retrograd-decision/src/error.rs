use std::io;

use thiserror::Error;

/// What can go wrong around a decision model.
///
/// Callers that speak HTTP match on this type *before* converting it: the
/// status code of a variant is finer than the class of
/// [`retrograd_core::Error`] it lands in (see the `From` impl below).
#[derive(Debug, Error)]
pub enum DecisionError {
    /// The request body is wrong. The message is the one llama-server returns.
    #[error("{0}")]
    InvalidRequest(String),
    /// The model has no `<arch>.decision.type` metadata. A server answers 501.
    #[error("this model is not a decision model")]
    NotDecisionModel,
    /// Images were sent to a model whose prompt has no place for them. A server
    /// answers 501.
    #[error("this model does not support image input for decisions")]
    ImagesUnsupported,
    /// `<arch>.decision.type` names a readout this crate does not implement.
    #[error("unsupported decision model type: {name}")]
    UnknownType { name: String },
    /// The model file says something a decision cannot use: a label that is not
    /// a single token, no mask token, an invalid temperature.
    #[error("invalid decision model: {0}")]
    InvalidModel(String),
    /// The rendered prompt does not have the shape the layout expects, for
    /// example a template that dropped the marker of an option.
    #[error("unexpected layout of the decision prompt")]
    Layout,
    /// The caller's template renderer failed.
    #[error("template rendering failed: {0}")]
    Render(String),
    /// The raw model outputs cannot be turned into an answer, or a caller broke
    /// the contract of a function of this crate (a variant out of range, scores
    /// of the wrong length).
    #[error("the model could not evaluate the decision: {0}")]
    Evaluation(String),
    /// Calibration was asked for something it cannot fit.
    #[error("calibration: {0}")]
    Calibration(String),
    /// One line of a training file is wrong. `line` is 1-based; `0` means the
    /// file as a whole (no example at all).
    #[error("dataset {path}:{line}: {message}")]
    Dataset {
        path: String,
        line: usize,
        message: String,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// The class a decision failure lands in once it leaves this crate.
///
/// `InvalidArgument` is the sentence "what the caller passed is wrong", and for
/// a decision that covers the request body, the model handed to `--model` or
/// `"model"` when it is not a decision model (or one of an unported type), and
/// a validation set that cannot be fitted. Those are `is_user_error()`, so a
/// frontend answers 4xx or exits with the usage code. The 501 of a non-decision
/// model is not a class of the core error: the HTTP layer reads it off
/// [`DecisionError::NotDecisionModel`] / [`DecisionError::ImagesUnsupported`]
/// before converting.
///
/// A broken model file, a prompt that lost its markers, a renderer failure and
/// unusable scores are the machine's or the template's fault, not something the
/// caller writes differently: `Runtime`.
impl From<DecisionError> for retrograd_core::Error {
    fn from(error: DecisionError) -> Self {
        match error {
            DecisionError::InvalidRequest(_)
            | DecisionError::NotDecisionModel
            | DecisionError::ImagesUnsupported
            | DecisionError::UnknownType { .. }
            | DecisionError::Calibration(_) => Self::invalid(error.to_string()),
            DecisionError::InvalidModel(_)
            | DecisionError::Layout
            | DecisionError::Render(_)
            | DecisionError::Evaluation(_) => Self::runtime(error.to_string()),
            DecisionError::Dataset {
                path,
                line,
                message,
            } => Self::dataset(path, line, message),
            DecisionError::Io(error) => Self::Io(error),
        }
    }
}

impl DecisionError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }
}

#[cfg(test)]
mod tests {
    use retrograd_core::{Error, ErrorKind};

    use super::*;

    #[test]
    fn classes_follow_who_can_fix_it() {
        let request: Error = DecisionError::invalid("\"state\" must be provided").into();
        assert_eq!(request.kind(), ErrorKind::InvalidArgument);
        assert!(request.is_user_error());

        let not_decision: Error = DecisionError::NotDecisionModel.into();
        assert_eq!(not_decision.kind(), ErrorKind::InvalidArgument);

        let layout: Error = DecisionError::Layout.into();
        assert_eq!(layout.kind(), ErrorKind::Runtime);
        assert!(!layout.is_user_error());

        let dataset: Error = DecisionError::Dataset {
            path: "train.jsonl".into(),
            line: 7,
            message: "bad".into(),
        }
        .into();
        assert!(matches!(dataset, Error::Dataset { line: 7, .. }));
    }
}
