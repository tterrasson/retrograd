//! The layout of a `laya` model: `[cls] question [sep] ([marker] option)* [sep]
//! state [sep]`.
//!
//! The question and the options are cut to fit `max_head_tokens`, the way the
//! model was trained (`julia/data.py::sequence`); the state is not cut here.

use super::Layout;
use crate::{DecisionError, Token};

/// Tokens an option may keep, its marker not counted.
pub const MAX_OPTION_TOKENS: usize = 48;

/// What the head keeps free for the question before the options are shrunk.
const QUESTION_RESERVE: usize = 16;

/// The least an option keeps when the options are shrunk evenly.
const MIN_OPTION_TOKENS: usize = 4;

/// The least the question keeps.
const MIN_QUESTION_TOKENS: usize = 8;

/// The token budget of the head of the prompt: the question and the options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    max_head_tokens: usize,
    max_option_tokens: usize,
}

impl Budget {
    /// `max_head_tokens` is the value of `<arch>.decision.max_head_tokens` of
    /// the model, unless the configuration overrides it.
    pub fn new(max_head_tokens: usize) -> Result<Self, DecisionError> {
        if max_head_tokens == 0 {
            return Err(DecisionError::InvalidModel(
                "decision model has no valid max_head_tokens".into(),
            ));
        }
        Ok(Self {
            max_head_tokens,
            max_option_tokens: MAX_OPTION_TOKENS,
        })
    }

    pub fn max_head_tokens(&self) -> usize {
        self.max_head_tokens
    }
}

/// Cuts the head of `rendered` and finds the marker of each of `n_options`
/// options. `rendered` must have the shape the template gives it: the cls
/// token, the question, `sep`, then each option behind its `marker`, `sep`, the
/// state and a last `sep`.
pub fn arrange(
    rendered: &[Token],
    n_options: usize,
    marker: Token,
    sep: Token,
    budget: &Budget,
) -> Result<Layout, DecisionError> {
    let found: Vec<usize> = rendered
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| (token == marker).then_some(position))
        .collect();
    let (Some(&first), Some(&last)) = (found.first(), found.last()) else {
        return Err(DecisionError::Layout);
    };
    if found.len() != n_options
        || first < 2
        || rendered[first - 1] != sep
        || rendered.last() != Some(&sep)
    {
        return Err(DecisionError::Layout);
    }
    let head_end = first - 1;
    let opts_end = rendered[last..]
        .iter()
        .position(|&token| token == sep)
        .map_or(rendered.len(), |offset| last + offset);
    if opts_end + 1 >= rendered.len() {
        return Err(DecisionError::Layout);
    }

    // Marker and text of each option.
    let mut options: Vec<&[Token]> = found
        .iter()
        .enumerate()
        .map(|(i, &start)| &rendered[start..found.get(i + 1).copied().unwrap_or(opts_end)])
        .collect();
    let mut cut_to = |max: usize| -> usize {
        options
            .iter_mut()
            .for_each(|option| *option = &option[..option.len().min(max)]);
        options.iter().map(|option| option.len()).sum()
    };

    let max_head = budget.max_head_tokens;
    let mut n_options_tokens = cut_to(budget.max_option_tokens.saturating_add(1));
    if n_options_tokens.saturating_add(QUESTION_RESERVE) > max_head {
        // Too many or too long options: shrink them evenly.
        let each = (max_head - max_head.min(QUESTION_RESERVE)) / n_options;
        n_options_tokens = cut_to(each.max(MIN_OPTION_TOKENS));
    }
    let n_question_max = MIN_QUESTION_TOKENS.max(max_head - max_head.min(n_options_tokens));

    let mut tokens = Vec::with_capacity(rendered.len());
    tokens.push(rendered[0]);
    tokens.extend_from_slice(&rendered[1..head_end.min(n_question_max.saturating_add(1))]);
    tokens.push(sep);
    let mut markers = Vec::with_capacity(n_options);
    for option in options {
        markers.push(tokens.len());
        tokens.extend_from_slice(option);
    }
    tokens.extend_from_slice(&rendered[opts_end..]);
    Ok(Layout {
        tokens,
        markers,
        pointer: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLS: Token = 1000;
    const MARKER: Token = 4;
    const SEP: Token = 1;

    /// Token `j` of option `i`.
    fn id(i: usize, j: usize) -> Token {
        Token::try_from(100 * (i + 1) + j).expect("small")
    }

    /// `[cls] q.. [sep] ([marker] option..)* [sep] state.. [sep]`. Question
    /// tokens are 10, 11, ...; the state tokens are 5000, 5001, ...
    fn rendered(n_question: usize, option_lens: &[usize], n_state: usize) -> Vec<Token> {
        let mut tokens = vec![CLS];
        tokens.extend((10..).take(n_question));
        tokens.push(SEP);
        for (i, &len) in option_lens.iter().enumerate() {
            tokens.push(MARKER);
            tokens.extend((0..len).map(|j| id(i, j)));
        }
        tokens.push(SEP);
        tokens.extend((5000..).take(n_state));
        tokens.push(SEP);
        tokens
    }

    /// Option `i` cut to `kept` tokens, its marker included.
    fn option(i: usize, kept: usize) -> Vec<Token> {
        let mut tokens = vec![MARKER];
        tokens.extend((0..kept - 1).map(|j| id(i, j)));
        tokens
    }

    fn arrange_with(rendered: &[Token], n_options: usize, max_head: usize) -> Layout {
        arrange(
            rendered,
            n_options,
            MARKER,
            SEP,
            &Budget::new(max_head).expect("budget"),
        )
        .expect("layout")
    }

    #[test]
    fn a_prompt_inside_the_budget_is_unchanged() {
        let input = rendered(5, &[3, 2, 4], 7);
        let layout = arrange_with(&input, 3, 256);
        assert_eq!(layout.tokens, input);
        // [cls] + 5 question + [sep] = 7, then options of 4, 3 and 5 tokens.
        assert_eq!(layout.markers, [7, 11, 14]);
        assert_eq!(layout.pointer, None);
    }

    #[test]
    fn an_option_keeps_its_marker_and_48_tokens() {
        let input = rendered(4, &[60, 3], 2);
        let layout = arrange_with(&input, 2, 256);
        let mut expected = vec![CLS, 10, 11, 12, 13, SEP];
        expected.extend(option(0, 49));
        expected.extend(option(1, 4));
        expected.extend([SEP, 5000, 5001, SEP]);
        assert_eq!(layout.tokens, expected);
        assert_eq!(layout.markers, [6, 6 + 49]);
    }

    #[test]
    fn options_shrink_evenly_and_the_question_takes_what_is_left() {
        // 4 options of 20 tokens, budget 40: (40 - 16) / 4 = 6 per option,
        // marker included, which leaves 40 - 24 = 16 for a question of 30.
        let input = rendered(30, &[20; 4], 3);
        let layout = arrange_with(&input, 4, 40);
        let mut expected = vec![CLS];
        expected.extend((10..).take(16));
        expected.push(SEP);
        let first = expected.len();
        for i in 0..4 {
            expected.extend(option(i, 6));
        }
        expected.extend([SEP, 5000, 5001, 5002, SEP]);
        assert_eq!(layout.tokens, expected);
        assert_eq!(layout.markers, [first, first + 6, first + 12, first + 18]);
    }

    #[test]
    fn options_never_shrink_below_four_and_the_question_keeps_eight() {
        // Budget 20, 10 options: (20 - 16) / 10 = 0, raised to 4 per option;
        // 40 option tokens leave nothing, the question keeps its minimum of 8.
        let input = rendered(30, &[9; 10], 1);
        let layout = arrange_with(&input, 10, 20);
        let mut expected = vec![CLS];
        expected.extend((10..).take(8));
        expected.push(SEP);
        for i in 0..10 {
            expected.extend(option(i, 4));
        }
        expected.extend([SEP, 5000, SEP]);
        assert_eq!(layout.tokens, expected);
        assert!(layout.markers.windows(2).all(|w| w[1] - w[0] == 4));
    }

    #[test]
    fn a_question_inside_the_budget_is_not_cut() {
        // 2 options of 4 tokens leave 40 - 8 = 32 for a question of 30.
        let input = rendered(30, &[3, 3], 0);
        let layout = arrange_with(&input, 2, 40);
        assert_eq!(layout.tokens, input);
        assert_eq!(layout.markers, [32, 36]);
    }

    #[test]
    fn the_state_is_never_cut() {
        let input = rendered(3, &[2, 2], 5000);
        let layout = arrange_with(&input, 2, 64);
        assert_eq!(layout.tokens, input);
    }

    #[test]
    fn malformed_prompts_are_layout_errors() {
        let ok = rendered(3, &[2, 2], 2);
        let budget = Budget::new(64).expect("budget");
        let run = |tokens: &[Token], n: usize| arrange(tokens, n, MARKER, SEP, &budget);

        assert!(run(&ok, 2).is_ok());
        // The template lost or invented a marker.
        assert!(matches!(run(&ok, 3), Err(DecisionError::Layout)));
        assert!(matches!(run(&ok, 1), Err(DecisionError::Layout)));
        assert!(matches!(run(&[], 0), Err(DecisionError::Layout)));
        // No separator before the first option.
        let mut broken = ok.clone();
        broken[4] = 77;
        assert!(matches!(run(&broken, 2), Err(DecisionError::Layout)));
        // No state after the options: the separator that closes them is last.
        let no_state = rendered(3, &[2, 2], 0);
        assert!(matches!(
            run(&no_state[..no_state.len() - 1], 2),
            Err(DecisionError::Layout)
        ));
        // Does not end with a separator.
        assert!(matches!(
            run(&ok[..ok.len() - 1], 2),
            Err(DecisionError::Layout)
        ));
        // The first marker right after the cls leaves no room for a question.
        assert!(matches!(
            run(&[CLS, MARKER, 5, SEP, 6, SEP], 1),
            Err(DecisionError::Layout)
        ));
    }

    #[test]
    fn a_zero_budget_is_a_model_error() {
        assert!(matches!(
            Budget::new(0),
            Err(DecisionError::InvalidModel(_))
        ));
    }
}
