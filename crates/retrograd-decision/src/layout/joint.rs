//! The layout of a joint head: the template separates the pieces of the
//! prompt with [`JOINT_SEP`] and marks the start of the span of a question or
//! of an option. Each piece is tokenized alone, as the model was trained, and
//! every token gets the decision order of its span, which the head reads from
//! the batch: the score of option `i` of the prompt is row `i` of its output.

use super::Layout;
use crate::request::{Question, QuestionKind};
use crate::{DecisionError, Token};

/// What a joint input must not contain: the start of the texts below.
pub const JOINT_MARKER: &str = "<<clef:";
/// Between the pieces of the prompt.
pub const JOINT_SEP: &str = "<<clef:sep>>";
/// At the start of the span of a question.
pub const JOINT_MARK_QUESTION: &str = "<<clef:question>>";
/// At the start of the span of an option.
pub const JOINT_MARK_OPTION: &str = "<<clef:option>>";

/// The decision order of a token, `llama_decision_order` of llama.cpp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionOrder {
    /// Not read by the head.
    None = 0,
    QuestionNoul = 1,
    QuestionChoice = 2,
    QuestionScore = 3,
    Option = 4,
}

impl DecisionOrder {
    fn of_question(kind: QuestionKind) -> Self {
        match kind {
            QuestionKind::Noul => Self::QuestionNoul,
            QuestionKind::Choice => Self::QuestionChoice,
            QuestionKind::Score => Self::QuestionScore,
        }
    }
}

/// Tokenizes the pieces of `prompt` and marks the spans of `questions`, which
/// must all be there, in order, with all their options.
pub fn arrange<E: From<DecisionError>>(
    prompt: &str,
    questions: &[&Question],
    mut tokenize: impl FnMut(&str) -> Result<Vec<Token>, E>,
) -> Result<Layout, E> {
    let mut layout = Layout::default();
    let mut i_question = 0;
    let mut n_scores = 0;
    for piece in prompt.split(JOINT_SEP) {
        let (text, order) = if let Some(text) = piece.strip_prefix(JOINT_MARK_QUESTION) {
            let Some(question) = questions.get(i_question) else {
                return Err(DecisionError::Layout.into());
            };
            i_question += 1;
            (text, DecisionOrder::of_question(question.kind))
        } else if let Some(text) = piece.strip_prefix(JOINT_MARK_OPTION) {
            n_scores += 1;
            (text, DecisionOrder::Option)
        } else {
            (piece, DecisionOrder::None)
        };
        let tokens = tokenize(text)?;
        if order != DecisionOrder::None && tokens.is_empty() {
            return Err(DecisionError::invalid(
                "the instructions and the options of a question must not be empty",
            )
            .into());
        }
        layout.tokens.extend(tokens);
        layout.order.resize(layout.tokens.len(), order as i32);
    }
    let n_options: usize = questions
        .iter()
        .map(|question| question.options.len())
        .sum();
    if i_question != questions.len() || n_scores != n_options {
        return Err(DecisionError::Layout.into());
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Json;
    use crate::request::QuestionOption;

    fn question(kind: QuestionKind, n: usize) -> Question {
        Question {
            kind,
            instructions: Json::from("i"),
            options: (0..n)
                .map(|i| QuestionOption {
                    key: i.to_string(),
                    description: Json::Null,
                })
                .collect(),
        }
    }

    fn tokenize(text: &str) -> Result<Vec<Token>, DecisionError> {
        Ok(text.bytes().map(Token::from).collect())
    }

    #[test]
    fn spans_get_the_order_of_their_question_or_option() {
        let noul = question(QuestionKind::Noul, 2);
        let choice = question(QuestionKind::Choice, 1);
        let prompt = format!(
            "s{JOINT_SEP}{JOINT_MARK_QUESTION}qq{JOINT_SEP}{JOINT_MARK_OPTION}a{JOINT_SEP}\
             {JOINT_MARK_OPTION}b{JOINT_SEP}.{JOINT_SEP}{JOINT_MARK_QUESTION}r{JOINT_SEP}\
             {JOINT_MARK_OPTION}cc{JOINT_SEP}"
        );
        let layout = arrange(&prompt, &[&noul, &choice], tokenize).expect("layout");
        assert_eq!(layout.tokens, tokenize("sqqab.rcc").expect("tokens"));
        assert_eq!(layout.order, [0, 1, 1, 4, 4, 0, 2, 4, 4]);
    }

    #[test]
    fn missing_spans_and_empty_ones_are_refused() {
        let noul = question(QuestionKind::Noul, 2);
        let one_option = format!("{JOINT_MARK_QUESTION}q{JOINT_SEP}{JOINT_MARK_OPTION}a");
        assert!(matches!(
            arrange(&one_option, &[&noul], tokenize),
            Err(DecisionError::Layout)
        ));
        let empty = format!("{JOINT_MARK_QUESTION}{JOINT_SEP}{JOINT_MARK_OPTION}a");
        assert!(matches!(
            arrange(&empty, &[&noul], tokenize),
            Err(DecisionError::InvalidRequest(message))
                if message == "the instructions and the options of a question must not be empty"
        ));
        let extra_question = format!("{JOINT_MARK_QUESTION}q");
        assert!(matches!(
            arrange(&extra_question, &[], tokenize),
            Err(DecisionError::Layout)
        ));
    }
}
