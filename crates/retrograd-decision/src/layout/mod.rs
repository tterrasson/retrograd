//! How rendered tokens are cut and where the scores are read.
//!
//! The input is the token ids of the rendered prompt (tokenized without
//! special tokens added, special token texts parsed); the output is the tokens
//! the model evaluates and the positions of the outputs that hold the scores.

pub mod laya;
pub mod pointer;

pub use laya::Budget;

use crate::readout::Readout;
use crate::request::Question;
use crate::{DecisionError, Token};

/// The prompt a model evaluates and where it is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub tokens: Vec<Token>,
    /// Positions whose output holds the score of an option, in option order.
    /// Empty for a readout that reads the logits at the last token.
    pub markers: Vec<usize>,
    /// Position of the query of a pointer readout.
    pub pointer: Option<usize>,
}

/// Arranges the rendered tokens of `question` for `readout`. `budget` is read
/// by a [`Readout::MarkerColumn`] only.
pub fn arrange(
    readout: &Readout,
    rendered: &[Token],
    question: &Question,
    budget: &Budget,
) -> Result<Layout, DecisionError> {
    match readout {
        Readout::MarkerColumn { marker, sep, .. } => {
            laya::arrange(rendered, question.options.len(), *marker, *sep, budget)
        }
        Readout::Pointer { marker } => pointer::arrange(rendered, question.options.len(), *marker),
        Readout::LabelLogits { .. } => Ok(Layout {
            tokens: rendered.to_vec(),
            markers: Vec::new(),
            pointer: None,
        }),
    }
}
