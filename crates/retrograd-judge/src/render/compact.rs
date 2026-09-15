//! LLM compaction of the middle of a transcript.
//!
//! Per-message elision ([`super::elide`]) is lossless about *where* it cut but
//! blind about *what* it cut: by default it keeps the first 40% and the last
//! 60% of a tool observation whether or not that is where the information is.
//! Compaction asks a model to summarise the middle turns instead. It sits
//! between elision and chunk planning: elision bounds each message,
//! compaction bounds the whole transcript, chunking bounds the request.
//!
//! Three constraints are not tunables, because breaking any of them injects a
//! bias straight into the GRPO advantage:
//!
//! 1. **Deterministic.** Temperature 0, cached by content hash in the same JSONL
//!    file as the verdicts. The same trajectory must compact to the same text on
//!    every replay, or the response cache and the run's reproducibility both go.
//! 2. **The same budget for every member of a group.** A member summarised
//!    harder than its siblings is handicapped for a reason that is not its
//!    performance. The decision to compact is therefore taken once per group,
//!    see [`plan_group`] - and never per trajectory.
//! 3. **Never the opening, never the last turns.** The opening states the task
//!    and the closing turns carry the outcome; only the middle is summarisable.
//!
//! Compaction *during* a rollout is a different and much heavier problem and is
//! explicitly out of scope: rewriting the context breaks the "one trajectory =
//! one growing sequence" invariant the rollout engine checks on every tool turn.

use serde::Deserialize;

use retrograd_agent_core::{Error, Result};

use super::JudgeMessage;

pub use retrograd_spec::judge::CompactionConfig;

/// Whether a group is compacted at all, decided from its *longest* member.
///
/// One decision for the whole group is the point: taking it per trajectory would
/// summarise the verbose members and leave the terse ones intact, and the judge
/// would then be comparing a summary against a transcript.
pub fn plan_group(config: &CompactionConfig, member_chars: &[usize]) -> bool {
    member_chars
        .iter()
        .any(|&chars| chars > config.trigger_chars)
}

/// Prompt asking for a summary of `messages`, under a stated budget.
pub fn compaction_prompt(config: &CompactionConfig, messages: &[JudgeMessage]) -> Result<String> {
    Ok(format!(
        "Summarise the middle of an agent transcript for a grader who will read the opening and the final turns in full.\n\
         Report what the agent attempted, what the tools answered, and which attempts failed. Keep it factual: no praise, no criticism, no score.\n\
         Stay under {} characters.\n\n<transcript>\n{}\n</transcript>",
        config.target_chars,
        serde_json::to_string(messages)
            .map_err(|error| Error::Reward(format!("serialize transcript to compact: {error}")))?
    ))
}

pub fn compaction_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}}
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompactionResponse {
    summary: String,
}

pub fn parse_compaction(content: &str) -> Result<String> {
    let response: CompactionResponse = serde_json::from_str(super::prompt::json_body(content)?)
        .map_err(|error| Error::Reward(format!("invalid judge summary JSON: {error}")))?;
    if response.summary.trim().is_empty() {
        return Err(Error::Reward("judge returned an empty summary".into()));
    }
    Ok(response.summary)
}

/// Replaces `range` with one summary message. The marker names the count so the
/// judge can tell a compacted transcript from a short one - an unmarked summary
/// would read as the agent's own words.
pub fn splice(messages: &mut Vec<JudgeMessage>, range: std::ops::Range<usize>, summary: &str) {
    let dropped = range.len();
    if dropped == 0 {
        return;
    }
    let marker = JudgeMessage {
        role: "system",
        content: format!("[{dropped} messages summarised]\n{summary}"),
        is_error: false,
    };
    messages.splice(range, [marker]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages(count: usize) -> Vec<JudgeMessage> {
        (0..count)
            .map(|index| JudgeMessage {
                role: "assistant",
                content: format!("turn-{index}"),
                is_error: false,
            })
            .collect()
    }

    #[test]
    fn the_opening_and_the_closing_turns_are_never_compactable() {
        let config = CompactionConfig {
            keep_last: 2,
            ..Default::default()
        };
        assert_eq!(config.middle(10), 1..8);
        // Nothing left in between: no request, no summary.
        assert!(config.middle(3).is_empty());
        assert!(config.middle(1).is_empty());
        assert!(config.middle(0).is_empty());
    }

    #[test]
    fn the_decision_is_taken_once_for_the_whole_group() {
        let config = CompactionConfig {
            trigger_chars: 100,
            ..Default::default()
        };
        // One long member pulls its short siblings in with it: the judge must
        // compare summaries with summaries, never a summary with a transcript.
        assert!(plan_group(&config, &[10, 10, 500]));
        assert!(!plan_group(&config, &[10, 10, 100]));
    }

    #[test]
    fn splicing_marks_the_summary_as_one() {
        let mut rendered = messages(6);
        splice(&mut rendered, 1..4, "they tried things");
        assert_eq!(rendered.len(), 4);
        assert_eq!(rendered[0].content, "turn-0");
        assert!(rendered[1].content.starts_with("[3 messages summarised]"));
        assert!(rendered[1].content.contains("they tried things"));
        assert_eq!(rendered[2].content, "turn-4");

        // An empty range is a no-op rather than an empty marker.
        let mut untouched = messages(3);
        splice(&mut untouched, 1..1, "nothing");
        assert_eq!(untouched.len(), 3);
    }

    #[test]
    fn a_configuration_that_cannot_shorten_anything_is_refused() {
        assert!(CompactionConfig::default().validate().is_ok());
        for config in [
            CompactionConfig {
                trigger_chars: 0,
                ..Default::default()
            },
            CompactionConfig {
                keep_last: 0,
                ..Default::default()
            },
            CompactionConfig {
                target_chars: 9_000,
                trigger_chars: 6_000,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err(), "accepted {config:?}");
        }
    }

    #[test]
    fn a_summary_response_is_parsed_and_an_empty_one_refused() {
        assert_eq!(
            parse_compaction(r#"{"summary":"they ran the tests"}"#).unwrap(),
            "they ran the tests"
        );
        assert!(parse_compaction(r#"{"summary":"   "}"#).is_err());
        assert!(parse_compaction("no json here").is_err());
    }
}
