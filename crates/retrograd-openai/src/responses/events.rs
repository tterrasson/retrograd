//! The typed event sequence of a streamed Responses answer:
//! `response.created`, `response.in_progress`, `response.output_item.added`,
//! `response.content_part.added`, `response.output_text.delta`,
//! `response.output_text.done`, `response.content_part.done`,
//! `response.output_item.done`, the `function_call_arguments` and reasoning
//! events for those item kinds, then exactly one terminal
//! `response.completed | incomplete | failed` (a mid-stream failure is a
//! plain `error` event). Every event carries a monotone `sequence_number`;
//! each carries its `output_index`/`content_index`/`item_id` per the pinned
//! schema. Responses does not use the Chat `[DONE]` terminator.
//!
//! The deltas themselves come from the shared
//! [`StreamParser`](crate::stream::StreamParser), which holds back whatever
//! the partial parse could still revise; this module only shapes them into
//! the events the protocol names. [`Sequencer`] hands out the sequence
//! numbers and enforces the two structural rules - one terminal event, and
//! no delta after a `done` - so a caller cannot emit an invalid sequence.

use serde::Serialize;

use crate::responses::wire::{EventLogprob, OutputContent, OutputItem, Response, Status};

/// One streamed event, as the pinned schema spells it. The `type` strings are
/// the protocol's, dots included.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: Response,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: Response,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContent,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        logprobs: Vec<EventLogprob>,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        logprobs: Vec<EventLogprob>,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContent,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        arguments: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: Response,
    },
    #[serde(rename = "response.incomplete")]
    Incomplete {
        sequence_number: u64,
        response: Response,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: Response,
    },
    /// A mid-stream failure: after the stream opened, an error is an event,
    /// not a new HTTP status.
    #[serde(rename = "error")]
    Error {
        sequence_number: u64,
        code: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        param: Option<String>,
    },
}

impl ResponseEvent {
    /// Overwrites the sequence number - the [`Sequencer`] assigns it.
    pub fn set_sequence_number(&mut self, value: u64) {
        match self {
            Self::Created {
                sequence_number, ..
            }
            | Self::InProgress {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::OutputTextDelta {
                sequence_number, ..
            }
            | Self::OutputTextDone {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | Self::ReasoningTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningTextDone {
                sequence_number, ..
            }
            | Self::Completed {
                sequence_number, ..
            }
            | Self::Incomplete {
                sequence_number, ..
            }
            | Self::Failed {
                sequence_number, ..
            }
            | Self::Error {
                sequence_number, ..
            } => *sequence_number = value,
        }
    }

    /// The event's sequence number.
    pub fn sequence_number(&self) -> u64 {
        match self {
            Self::Created {
                sequence_number, ..
            }
            | Self::InProgress {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::OutputTextDelta {
                sequence_number, ..
            }
            | Self::OutputTextDone {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | Self::ReasoningTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningTextDone {
                sequence_number, ..
            }
            | Self::Completed {
                sequence_number, ..
            }
            | Self::Incomplete {
                sequence_number, ..
            }
            | Self::Failed {
                sequence_number, ..
            }
            | Self::Error {
                sequence_number, ..
            } => *sequence_number,
        }
    }

    /// The event's `type` string, as serialized.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Created { .. } => "response.created",
            Self::InProgress { .. } => "response.in_progress",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            Self::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            Self::ReasoningTextDone { .. } => "response.reasoning_text.done",
            Self::Completed { .. } => "response.completed",
            Self::Incomplete { .. } => "response.incomplete",
            Self::Failed { .. } => "response.failed",
            Self::Error { .. } => "error",
        }
    }

    /// Whether this event ends the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. }
                | Self::Incomplete { .. }
                | Self::Failed { .. }
                | Self::Error { .. }
        )
    }

    /// The `status` a terminal event's response carries.
    pub fn terminal_status(&self) -> Option<Status> {
        match self {
            Self::Completed { .. } => Some(Status::Completed),
            Self::Incomplete { .. } => Some(Status::Incomplete),
            Self::Failed { .. } => Some(Status::Failed),
            _ => None,
        }
    }
}

/// Hands out monotone sequence numbers and enforces the structural rules of
/// the sequence: at most one terminal event, and no event after it. A second
/// terminal (or anything after one) is dropped - the stream has already
/// ended - rather than emitted as an invalid sequence.
///
/// Events are built with a placeholder `sequence_number` (0); [`Sequencer::push`]
/// assigns the real one, so a caller cannot emit two events with the same
/// number by accident.
pub struct Sequencer {
    next: u64,
    terminated: bool,
}

impl Sequencer {
    pub fn new() -> Self {
        Self {
            next: 0,
            terminated: false,
        }
    }

    /// The next sequence number, without emitting.
    pub fn peek(&self) -> u64 {
        self.next
    }

    /// Sequences `event` (assigning its number) if the sequence is still open;
    /// `None` once the terminal went out.
    pub fn push(&mut self, mut event: ResponseEvent) -> Option<ResponseEvent> {
        if self.terminated {
            return None;
        }
        event.set_sequence_number(self.next);
        self.next += 1;
        if event.is_terminal() {
            self.terminated = true;
        }
        Some(event)
    }
}

impl Default for Sequencer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::wire::{InputTokensDetails, OutputTokensDetails};

    fn response(status: Status) -> Response {
        Response {
            id: "resp_1".into(),
            object: "response",
            created_at: 1,
            completed_at: None,
            status,
            model: "m".into(),
            output: Vec::new(),
            output_text: None,
            usage: crate::responses::wire::Usage {
                input_tokens: 1,
                input_tokens_details: InputTokensDetails {
                    cached_tokens: 0,
                    cache_write_tokens: 0,
                },
                output_tokens: 2,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 3,
            },
            error: None,
            incomplete_details: None,
            instructions: None,
            tools: Vec::new(),
            tool_choice: serde_json::json!("auto"),
            parallel_tool_calls: true,
            metadata: None,
            temperature: None,
            top_p: None,
            top_logprobs: None,
            max_output_tokens: None,
            text: crate::responses::wire::TextEcho {
                format: serde_json::json!({"type": "text"}),
            },
            reasoning: None,
            store: false,
            background: false,
            previous_response_id: None,
            truncation: "disabled",
            service_tier: "default",
            access_programs: Vec::new(),
        }
    }

    /// The state machine: monotone numbers, one terminal, nothing after it.
    #[test]
    fn sequence_numbers_are_monotone_and_the_terminal_is_single() {
        let mut seq = Sequencer::new();
        let created = seq.push(ResponseEvent::Created {
            sequence_number: 0,
            response: response(Status::InProgress),
        });
        let completed = seq.push(ResponseEvent::Completed {
            sequence_number: 0,
            response: response(Status::Completed),
        });
        let after = seq.push(ResponseEvent::OutputTextDelta {
            sequence_number: 0,
            item_id: "msg_1".into(),
            output_index: 0,
            content_index: 0,
            delta: "late".into(),
            logprobs: Vec::new(),
        });
        let created = created.expect("the first event goes out");
        let completed = completed.expect("the terminal goes out");
        assert_eq!(created.sequence_number(), 0);
        assert_eq!(completed.sequence_number(), 1);
        assert!(after.is_none(), "nothing is emitted after the terminal");
        // A second terminal is refused as well.
        let again = seq.push(ResponseEvent::Failed {
            sequence_number: 0,
            response: response(Status::Failed),
        });
        assert!(again.is_none());
    }

    /// The event type strings are exactly the protocol's.
    #[test]
    fn event_type_strings_match_the_protocol() {
        let event = ResponseEvent::OutputTextDelta {
            sequence_number: 3,
            item_id: "msg_1".into(),
            output_index: 0,
            content_index: 0,
            delta: "hi".into(),
            logprobs: Vec::new(),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "response.output_text.delta");
        assert_eq!(value["sequence_number"], 3);
        assert_eq!(event.kind(), "response.output_text.delta");
        assert!(!event.is_terminal());
        assert!(
            ResponseEvent::Completed {
                sequence_number: 9,
                response: response(Status::Completed),
            }
            .is_terminal()
        );
    }
}
