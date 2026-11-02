//! How rendered tokens are cut and where the scores are read.
//!
//! The input is the token ids of the rendered prompt (tokenized without
//! special tokens added, special token texts parsed); the output is the tokens
//! the model evaluates and the positions of the outputs that hold the scores.
//! Two layouts tokenize the prompt piece by piece, as their models were
//! trained: [`d1omni`] and [`joint`], through [`arrange_prompt`].

pub mod d1omni;
pub mod joint;
pub mod laya;
pub mod pointer;

pub use laya::Budget;

use crate::readout::{Head, Readout};
use crate::request::Question;
use crate::{DecisionError, Token};

/// The prompt a model evaluates and where it is read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub tokens: Vec<Token>,
    /// Positions whose output holds the score of an option, in option order.
    /// Empty for a readout that reads the logits at the last token.
    pub markers: Vec<usize>,
    /// Position of the query of a pointer readout.
    pub pointer: Option<usize>,
    /// The decision order of each token, for a joint head
    /// ([`joint::DecisionOrder`]); empty otherwise.
    pub order: Vec<i32>,
}

/// Arranges the rendered tokens of `question` for `readout`, a readout whose
/// prompt is tokenized whole. `budget` is read by a marker readout only.
pub fn arrange(
    readout: &Readout,
    rendered: &[Token],
    question: &Question,
    budget: &Budget,
) -> Result<Layout, DecisionError> {
    match &readout.head {
        Head::MarkerColumn { marker, sep, .. } => {
            laya::arrange(rendered, question.options.len(), *marker, *sep, budget)
        }
        Head::Pointer { marker } => pointer::arrange(rendered, question.options.len(), *marker),
        Head::LabelLogits { .. } | Head::LabelGroups | Head::Pooled { .. } => Ok(Layout {
            tokens: rendered.to_vec(),
            ..Layout::default()
        }),
        Head::D1Omni { .. } | Head::Joint => Err(DecisionError::Evaluation(format!(
            "a `{}` prompt is tokenized piece by piece, from its text",
            readout.name()
        ))),
    }
}

/// Tokenizes and arranges the rendered `prompt` of `questions`: one question,
/// or every question of a joint prompt. `tokenize` reads the text of a
/// special token as that token and adds none.
pub fn arrange_prompt<E: From<DecisionError>>(
    readout: &Readout,
    prompt: &str,
    questions: &[&Question],
    budget: &Budget,
    mut tokenize: impl FnMut(&str) -> Result<Vec<Token>, E>,
) -> Result<Layout, E> {
    if let Head::Joint = readout.head {
        return joint::arrange(prompt, questions, tokenize);
    }
    let [question] = questions else {
        return Err(DecisionError::Evaluation(format!(
            "a `{}` prompt holds one question, not {}",
            readout.name(),
            questions.len()
        ))
        .into());
    };
    if let Head::D1Omni { marker } = readout.head {
        return d1omni::arrange(prompt, question.options.len(), marker, tokenize);
    }
    let rendered = tokenize(prompt)?;
    Ok(arrange(readout, &rendered, question, budget)?)
}
