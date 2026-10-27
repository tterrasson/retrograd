//! Whether the history a chat template renders around a sampled turn is the
//! text that turn was sampled as.
//!
//! A multi-turn stream is built once and never rendered again: the prompt, the
//! sampled tokens, then - after each observation - only the framing the
//! template writes after the turn. What the re-render still has to be checked
//! for is the framing *before* the turn, and the framing *inside* it: a
//! template that writes a reasoning block frames it with tags of its own,
//! which the generation prompt may already have opened, may have closed empty,
//! or may leave to the model.
//!
//! The rollout and the preparation of SFT records call the same function, so
//! a record is refused, accepted or flagged exactly when a rollout of it would
//! be. Pure: the parser comes in as a closure, so this crate stays free of the
//! FFI that derives it.

use crate::chat_template::{ParsedCalls, TurnPieces};

/// What a parser reads out of one assistant turn, as far as telling two texts
/// of the same turn apart goes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnReading {
    pub reasoning: String,
    pub content: String,
    /// The text reads as a call, or as an attempt at one.
    pub calls: bool,
}

impl From<ParsedCalls> for TurnReading {
    fn from(parsed: ParsedCalls) -> Self {
        Self {
            reasoning: parsed.reasoning.trim().to_owned(),
            content: parsed.content.trim().to_owned(),
            calls: !parsed.calls.is_empty() || !parsed.errors.is_empty(),
        }
    }
}

/// One sampled turn, as the policy produced it and as its parser read it.
#[derive(Clone, Copy, Debug)]
pub struct SampledTurn<'a> {
    /// The turn detokenized the way its parser is handed it: without the
    /// end-of-generation token the policy stopped on.
    pub text: &'a str,
    /// The reasoning the parser split out, tags removed; empty for a turn that
    /// did not reason.
    pub reasoning: &'a str,
    /// The prose the parser read, without its calls.
    pub content: &'a str,
}

/// Which part of a turn's framing a re-render does not reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnPart {
    /// What comes before the turn: the generation prompt it was sampled after.
    Before,
    /// The tags around its reasoning block.
    Reasoning,
    /// What it reads back as once framed.
    Content,
}

impl std::fmt::Display for TurnPart {
    /// Worded to precede "assistant turn N".
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Before => "the framing before",
            Self::Reasoning => "the framing around the reasoning of",
            Self::Content => "the content of",
        })
    }
}

/// Where the history and the sampled text of a turn part ways: each from a
/// little before that point, cut short.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextGap {
    pub history: String,
    pub sampled: String,
}

impl std::fmt::Display for TextGap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the history reads {:?} where the turn was sampled as {:?}",
            self.history, self.sampled
        )
    }
}

/// Context kept before the first difference, and the most of each text shown.
const GAP_CONTEXT: usize = 16;
const GAP_WIDTH: usize = 64;

impl TextGap {
    fn between(history: &str, sampled: &str) -> Self {
        let at = first_difference(history, sampled);
        let start = floor_boundary(history, at.saturating_sub(GAP_CONTEXT));
        Self {
            history: excerpt(history, start),
            sampled: excerpt(sampled, start),
        }
    }
}

/// The byte offset of the first character the two texts disagree on, or the
/// length of the shorter one.
fn first_difference(a: &str, b: &str) -> usize {
    a.char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map_or(a.len().min(b.len()), |((at, _), _)| at)
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn excerpt(text: &str, start: usize) -> String {
    let rest = &text[start..];
    let mut shown = String::new();
    if start > 0 {
        shown.push('…');
    }
    shown.extend(rest.chars().take(GAP_WIDTH));
    if rest.chars().nth(GAP_WIDTH).is_some() {
        shown.push('…');
    }
    shown
}

/// How the history a template renders for a turn compares with what the turn
/// was sampled after and as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnCheck {
    /// The history is the sampled text, byte for byte.
    Exact,
    /// The texts differ, but the template's parser reads the same turn out of
    /// both: an empty reasoning block the history leaves out, whitespace
    /// around a tag. The sampled tokens stay what is trained; an inference
    /// server that re-renders the history shows the model the template's
    /// version instead.
    Normalized(TextGap),
    /// The parser reads something else out of the history, or the template
    /// rewrote the framing before the turn.
    Rewritten { part: TurnPart, gap: TextGap },
}

/// Checks one turn the first time the conversation is rendered past it.
///
/// `committed` is the text of the framing piece the turn was sampled after -
/// the generation prompt - and `pieces` what the re-render writes around the
/// turn. The history renders the turn as `pre + reasoning + mid + content`,
/// the stream holds `committed + sampled`; the turn is exact when the second
/// starts with everything the first writes before the content, as long as
/// that covers the whole prompt.
///
/// `read` is the parser the turn was read with, `None` under a rendering whose
/// parser knows no reasoning block: there, only an exact history is accepted.
pub fn verify_turn(
    committed: &str,
    sampled: &SampledTurn<'_>,
    pieces: &TurnPieces,
    read: Option<&dyn Fn(&str) -> TurnReading>,
) -> TurnCheck {
    let mid = pieces.mid.as_deref();
    let reasoning = match mid {
        Some(_) => sampled.reasoning,
        None => "",
    };
    let head = format!("{}{reasoning}{}", pieces.pre, mid.unwrap_or_default());
    let actual = format!("{committed}{}", sampled.text);
    if head.len() >= committed.len() && actual.starts_with(&head) {
        return TurnCheck::Exact;
    }
    let gap = TextGap::between(&head, &actual);
    // The history may open the turn further than the generation prompt did -
    // a block the model then wrote itself - or stop short of it, leaving out
    // a block the prompt wrote. Anything else rewrote the prompt.
    let opening = match pieces.pre.strip_prefix(committed) {
        Some(opening) => opening,
        None if committed.starts_with(pieces.pre.as_str()) => "",
        None => {
            return TurnCheck::Rewritten {
                part: TurnPart::Before,
                gap: TextGap::between(&pieces.pre, committed),
            };
        }
    };
    let Some(read) = read else {
        let part = match first_difference(&head, &actual) >= pieces.pre.len() && mid.is_some() {
            true => TurnPart::Reasoning,
            false => TurnPart::Before,
        };
        return TurnCheck::Rewritten { part, gap };
    };
    // What the parser would have read had the model sampled the history's
    // version of the turn. The content goes in as the parser read it: it is
    // the sampled text either way, and what is in question is whether the
    // template's tags around it still read as the same split.
    let expected = read(&format!(
        "{opening}{reasoning}{}{}",
        mid.unwrap_or_default(),
        sampled.content
    ));
    if expected.calls || expected.reasoning != sampled.reasoning.trim() {
        return TurnCheck::Rewritten {
            part: TurnPart::Reasoning,
            gap,
        };
    }
    if expected.content != sampled.content.trim() {
        return TurnCheck::Rewritten {
            part: TurnPart::Content,
            gap,
        };
    }
    TurnCheck::Normalized(gap)
}

/// The first piece a later render moved, among turns already checked: every
/// render after the one that checked a turn has to write its framing again,
/// byte for byte.
pub fn moved_piece(
    checked: &[TurnPieces],
    rendered: &[TurnPieces],
) -> Option<(usize, TurnPart, TextGap)> {
    checked
        .iter()
        .zip(rendered)
        .enumerate()
        .find_map(|(turn, (checked, rendered))| {
            if rendered.pre != checked.pre {
                return Some((
                    turn,
                    TurnPart::Before,
                    TextGap::between(&rendered.pre, &checked.pre),
                ));
            }
            (rendered.mid != checked.mid).then(|| {
                (
                    turn,
                    TurnPart::Reasoning,
                    TextGap::between(
                        rendered.mid.as_deref().unwrap_or_default(),
                        checked.mid.as_deref().unwrap_or_default(),
                    ),
                )
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for a template-derived parser: reasoning is whatever comes
    /// before `</r>` - opened by `<r>` or by the prompt, as `open` says - and
    /// `<c>` marks a call.
    fn reader(open: bool) -> impl Fn(&str) -> TurnReading {
        move |text: &str| {
            let body = match open {
                true => Some(text),
                false => text.trim_start().strip_prefix("<r>"),
            };
            let (reasoning, content) = match body.and_then(|body| body.split_once("</r>")) {
                Some((reasoning, content)) => (reasoning, content),
                None => ("", text),
            };
            TurnReading {
                reasoning: reasoning.trim().to_owned(),
                content: content.replace("<c>", "").trim().to_owned(),
                calls: content.contains("<c>"),
            }
        }
    }

    fn turn<'a>(text: &'a str, reasoning: &'a str, content: &'a str) -> SampledTurn<'a> {
        SampledTurn {
            text,
            reasoning,
            content,
        }
    }

    fn pieces(pre: &str, mid: Option<&str>) -> TurnPieces {
        TurnPieces {
            pre: pre.into(),
            mid: mid.map(str::to_owned),
        }
    }

    fn check(
        committed: &str,
        sampled: SampledTurn<'_>,
        pieces: &TurnPieces,
        open: bool,
    ) -> TurnCheck {
        let read = reader(open);
        verify_turn(committed, &sampled, pieces, Some(&read))
    }

    /// The model opens the block itself, and the history writes it for a turn
    /// that reasoned.
    #[test]
    fn a_block_opened_by_the_model_is_exact() {
        let sampled = turn("<r>\nR\n</r>\n\nC", "R", "C");
        assert_eq!(
            check("A", sampled, &pieces("A<r>\n", Some("\n</r>\n\n")), false),
            TurnCheck::Exact
        );
        // A turn that did not reason gets no block, and nothing to compare.
        let sampled = turn("C", "", "C");
        assert_eq!(
            check("A", sampled, &pieces("A", None), false),
            TurnCheck::Exact
        );
    }

    /// The prompt opens the block, the model closes it.
    #[test]
    fn a_block_opened_by_the_prompt_is_exact() {
        let sampled = turn("R\n</r>\n\nC", "R", "C");
        assert_eq!(
            check(
                "A<r>\n",
                sampled,
                &pieces("A<r>\n", Some("\n</r>\n\n")),
                true
            ),
            TurnCheck::Exact
        );
    }

    /// The prompt closes an empty block the history of a turn without
    /// reasoning leaves out.
    #[test]
    fn an_empty_block_the_history_leaves_out_is_normalized() {
        let sampled = turn("C<c>", "", "C");
        let TurnCheck::Normalized(gap) =
            check("A<r>\n\n</r>\n\n", sampled, &pieces("A", None), false)
        else {
            panic!("expected a normalized turn");
        };
        assert_eq!(gap.history, "A");
        assert_eq!(gap.sampled, "A<r>\n\n</r>\n\nC<c>");
    }

    /// A history that writes an empty block the prompt left open.
    #[test]
    fn an_empty_block_the_history_closes_is_normalized() {
        let sampled = turn("</r>C", "", "C");
        assert!(matches!(
            check("A<r>\n", sampled, &pieces("A<r>\n\n</r>\n\n", None), true),
            TurnCheck::Normalized(_)
        ));
    }

    #[test]
    fn whitespace_around_the_tags_is_normalized() {
        let sampled = turn("<r>\n\nR\n\n</r>\nC", "R", "C");
        assert!(matches!(
            check("A", sampled, &pieces("A<r>\n", Some("\n</r>\n\n")), false),
            TurnCheck::Normalized(_)
        ));
    }

    /// The history writes the reasoning where the parser does not read it as
    /// reasoning.
    #[test]
    fn a_history_that_reads_as_another_turn_is_rewritten() {
        let sampled = turn("<r>\nR\n</r>\nC", "R", "C");
        let check = check("A", sampled, &pieces("A[", Some("]\n")), false);
        assert!(
            matches!(
                check,
                TurnCheck::Rewritten {
                    part: TurnPart::Reasoning,
                    ..
                }
            ),
            "{check:?}"
        );
    }

    #[test]
    fn a_rewritten_prompt_is_the_framing_before_the_turn() {
        let sampled = turn("C", "", "C");
        let check = check("<assistant>", sampled, &pieces("<model>", None), false);
        let TurnCheck::Rewritten {
            part: TurnPart::Before,
            gap,
        } = check
        else {
            panic!("{check:?}");
        };
        assert_eq!(gap.history, "<model>");
        assert_eq!(gap.sampled, "<assistant>");
    }

    #[test]
    fn without_a_parser_only_an_exact_history_is_accepted() {
        let sampled = turn("C", "", "C");
        assert_eq!(
            verify_turn("A", &sampled, &pieces("A", None), None),
            TurnCheck::Exact
        );
        assert!(matches!(
            verify_turn("A<r></r>", &sampled, &pieces("A", None), None),
            TurnCheck::Rewritten {
                part: TurnPart::Before,
                ..
            }
        ));
    }

    #[test]
    fn a_checked_piece_must_not_move() {
        let checked = [pieces("A", None), pieces("B<r>", Some("</r>"))];
        assert_eq!(moved_piece(&checked, &checked), None);
        let (turn, part, _) = moved_piece(
            &checked,
            &[pieces("A", None), pieces("B<r>", Some("</r>\n"))],
        )
        .unwrap();
        assert_eq!((turn, part), (1, TurnPart::Reasoning));
        let (turn, part, _) =
            moved_piece(&checked, &[pieces("a", None), pieces("B<r>", Some("</r>"))]).unwrap();
        assert_eq!((turn, part), (0, TurnPart::Before));
    }

    #[test]
    fn a_gap_shows_where_the_texts_part_ways() {
        let gap = TextGap::between(
            &format!("{}x", "a".repeat(40)),
            &format!("{}y", "a".repeat(40)),
        );
        assert_eq!(gap.history, format!("…{}x", "a".repeat(GAP_CONTEXT)));
        assert_eq!(gap.sampled, format!("…{}y", "a".repeat(GAP_CONTEXT)));
        let long = "é".repeat(200);
        let gap = TextGap::between(&long, "");
        assert!(gap.history.ends_with('…'));
        assert_eq!(gap.sampled, "");
    }
}
