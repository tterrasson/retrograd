//! The layout of a pointer readout: the prompt is left as rendered, an option is
//! read at each end token and the question at the last token.

use super::Layout;
use crate::{DecisionError, Token};

/// Finds the end token of each option in `rendered`. There must be exactly
/// `n_options` of them.
pub fn arrange(
    rendered: &[Token],
    n_options: usize,
    marker: Token,
) -> Result<Layout, DecisionError> {
    let markers: Vec<usize> = rendered
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| (token == marker).then_some(position))
        .collect();
    if markers.len() != n_options || rendered.is_empty() {
        return Err(DecisionError::Layout);
    }
    Ok(Layout {
        tokens: rendered.to_vec(),
        markers,
        pointer: Some(rendered.len() - 1),
        order: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_are_read_at_their_end_token_and_the_question_at_the_last() {
        let layout = arrange(&[5, 9, 6, 9, 7], 2, 9).expect("layout");
        assert_eq!(layout.markers, [1, 3]);
        assert_eq!(layout.pointer, Some(4));
        assert_eq!(layout.tokens, [5, 9, 6, 9, 7]);
    }

    #[test]
    fn a_missing_or_extra_end_token_is_a_layout_error() {
        assert!(matches!(
            arrange(&[5, 9, 7], 2, 9),
            Err(DecisionError::Layout)
        ));
        assert!(matches!(
            arrange(&[9, 9, 9], 2, 9),
            Err(DecisionError::Layout)
        ));
        assert!(matches!(arrange(&[], 0, 9), Err(DecisionError::Layout)));
    }
}
