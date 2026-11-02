//! The layout of a d1-omni model: the template separates the pieces of the
//! prompt with [`D1OMNI_SEP`], each piece is tokenized alone and cut to its
//! budget as the model was trained (d1-omni `prompt.py`: `encode`), the state
//! gets the room that is left. The options are read at the mask token that
//! opens each of them, in the column of the question type.

use super::Layout;
use crate::{DecisionError, Token};

/// What a d1-omni input must not contain: the start of the texts below.
pub const D1OMNI_MARKER: &str = "<<d1omni:";
/// Between the pieces of the prompt.
pub const D1OMNI_SEP: &str = "<<d1omni:sep>>";
/// At the start of the piece that holds the state.
pub const MARK_STATE: &str = "<<d1omni:state>>";
/// At the start of the piece that holds the question, cut to its budget.
pub const MARK_QUESTION: &str = "<<d1omni:question>>";
/// At the start of the piece of an option, cut to its budget.
pub const MARK_OPTION: &str = "<<d1omni:option>>";

/// `max_length` of the model configuration.
const MAX_TOKENS: usize = 16384;

/// Cuts the pieces of `prompt` and finds the marker of each of `n_options`
/// options. The first piece is the media markers, none here.
pub fn arrange<E: From<DecisionError>>(
    prompt: &str,
    n_options: usize,
    marker: Token,
    mut tokenize: impl FnMut(&str) -> Result<Vec<Token>, E>,
) -> Result<Layout, E> {
    let pieces: Vec<&str> = prompt.split(D1OMNI_SEP).collect();
    if pieces.first() != Some(&"") {
        return Err(DecisionError::Layout.into());
    }

    // The options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly.
    let n_max = MAX_TOKENS;
    let n = n_options.max(1);
    let n_budget = (24 * n + 32).min(n_max / 2).max(96);
    let n_option_max = (n_budget.saturating_sub(3 * n) / n).max(2);
    let n_question_max = n_budget.max(16);

    let mut head = Vec::new();
    let mut body = None;
    let mut tail = Vec::new();
    for piece in &pieces[1..] {
        let (text, limit, is_state) = if let Some(text) = piece.strip_prefix(MARK_STATE) {
            (text, None, true)
        } else if let Some(text) = piece.strip_prefix(MARK_QUESTION) {
            (text, Some(n_question_max), false)
        } else if let Some(text) = piece.strip_prefix(MARK_OPTION) {
            (text, Some(n_option_max), false)
        } else {
            (*piece, None, false)
        };
        let mut tokens = tokenize(text)?;
        if let Some(limit) = limit {
            tokens.truncate(limit);
        }
        if is_state {
            if body.is_some() {
                return Err(DecisionError::Layout.into());
            }
            body = Some(tokens);
        } else if body.is_some() {
            tail.extend(tokens);
        } else {
            head.extend(tokens);
        }
    }
    let Some(mut body) = body else {
        return Err(DecisionError::Layout.into());
    };

    let n_room = n_max - n_max.min(head.len() + tail.len());
    body.truncate(n_room);
    let mut tokens = head;
    tokens.extend(body);
    tokens.extend(tail);
    tokens.truncate(n_max);

    let markers: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| (token == marker).then_some(position))
        .collect();
    if markers.len() != n_options {
        return Err(DecisionError::invalid("the options do not fit in the context").into());
    }
    Ok(Layout {
        tokens,
        markers,
        ..Layout::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASK: Token = 4;

    /// One token per byte, the mask written `@`.
    fn tokenize(text: &str) -> Result<Vec<Token>, DecisionError> {
        Ok(text
            .bytes()
            .map(|byte| {
                if byte == b'@' {
                    MASK
                } else {
                    Token::from(byte)
                }
            })
            .collect())
    }

    fn prompt(question: &str, options: &[&str], state: &str) -> String {
        let mut out = format!("{D1OMNI_SEP}<q>{D1OMNI_SEP}{MARK_QUESTION}{question}");
        for option in options {
            out.push_str(&format!("{D1OMNI_SEP}{MARK_OPTION}@{option}"));
        }
        out.push_str(&format!("{D1OMNI_SEP}{MARK_STATE}{state}{D1OMNI_SEP}</s>"));
        out
    }

    #[test]
    fn pieces_are_joined_and_the_options_read_at_their_mask() {
        let layout =
            arrange(&prompt("why", &["ab", "c"], "st"), 2, MASK, tokenize).expect("layout");
        assert_eq!(
            layout.tokens,
            tokenize("<q>why@ab@cst</s>").expect("tokens")
        );
        assert_eq!(layout.markers, [6, 9]);
    }

    #[test]
    fn options_and_question_are_cut_to_their_budgets() {
        // Two options: a budget of 96, (96 - 6) / 2 = 45 tokens per option.
        let long = "x".repeat(60);
        let layout = arrange(
            &prompt(&"q".repeat(200), &[&long, "y"], ""),
            2,
            MASK,
            tokenize,
        )
        .expect("layout");
        assert_eq!(layout.tokens.len(), 3 + 96 + 45 + 2 + 4);
    }

    #[test]
    fn a_prompt_without_its_pieces_is_a_layout_error() {
        assert!(matches!(
            arrange("no pieces", 1, MASK, tokenize),
            Err(DecisionError::Layout)
        ));
        let no_state = format!("{D1OMNI_SEP}{MARK_OPTION}@a");
        assert!(matches!(
            arrange(&no_state, 1, MASK, tokenize),
            Err(DecisionError::Layout)
        ));
        let two_states = format!("{D1OMNI_SEP}{MARK_STATE}a{D1OMNI_SEP}{MARK_STATE}b");
        assert!(matches!(
            arrange(&two_states, 0, MASK, tokenize),
            Err(DecisionError::Layout)
        ));
        assert!(matches!(
            arrange(&prompt("q", &["a"], "s"), 2, MASK, tokenize),
            Err(DecisionError::InvalidRequest(_))
        ));
    }
}
