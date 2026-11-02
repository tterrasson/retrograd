//! Typed decision models: everything around the forward pass.
//!
//! A decision model does not generate. It reads a `state` and typed questions
//! (`choice`, `score`, `noul`) in one pass and returns a probability
//! distribution over the options the caller supplied. This crate holds the
//! pure half of that, with no FFI and no model:
//!
//! - [`json`]: a JSON value that keeps the order of the keys, which the state
//!   needs;
//! - [`request`]: the `/v1/systemone` body and its validation;
//! - [`readout`]: where a model reads the score of an option, from the
//!   `<arch>.decision.type` metadata and the vocabulary;
//! - [`template`]: the frozen context handed to the `systemone` Jinja template;
//! - [`layout`]: how the rendered tokens are cut and where the scores are read;
//! - [`answer`]: raw scores to the JSON answer, temperatures and confidence;
//! - [`dataset`]: the training examples;
//! - [`calibrate`]: fitting the temperatures on a validation set;
//! - [`metrics`]: accuracy, NLL, Brier, calibration error and coverage of a
//!   labelled set.
//!
//! The behaviour is the one of the decision server of llama.cpp
//! (`tools/server/server-decision.cpp`), so the same GGUF gives the same
//! numbers here and there. The three things the caller owns are the Jinja
//! engine ([`template::TemplateRenderer`]), the tokenizer and the vocabulary
//! ([`readout::Vocabulary`]), and the model itself.
//!
//! Every `decision.type` llama.cpp's decision server knows is a row of
//! [`readout::DECISION_TYPES`]; any other is [`DecisionError::UnknownType`].
//! Media (images, audio, video, spoken instructions) are recognised and
//! refused with [`DecisionError::ImagesUnsupported`]: the multimodal
//! projector is not this crate's caller's.

/// A `Json` literal in the tests. Its objects are alphabetical, as a
/// `serde_json::json!` value's are: a test about key order parses text.
#[cfg(test)]
macro_rules! json {
    ($($tokens:tt)*) => {
        $crate::json::Json::from(serde_json::json!($($tokens)*))
    };
}

pub mod answer;
pub mod calibrate;
pub mod causal;
pub mod dataset;
mod error;
pub mod json;
pub mod layout;
pub mod metrics;
mod python;
pub mod readout;
pub mod request;
pub mod template;

pub use error::DecisionError;
pub use json::Json;

/// `Json` from text, objects in the order written.
#[cfg(test)]
fn ordered(text: &str) -> Json {
    serde_json::from_str(text).expect("valid JSON")
}

/// A vocabulary id, the type of `llama_token`.
pub type Token = i32;
