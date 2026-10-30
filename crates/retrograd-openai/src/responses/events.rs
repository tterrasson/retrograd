//! The typed event sequence of a streamed Responses answer:
//! `response.created`, `response.in_progress`, `output_item` and
//! `content_part` added/done, `output_text.delta`, the reasoning events,
//! `function_call_arguments.delta/done`, then one terminal
//! `response.completed | incomplete | failed` - each with a monotone
//! `sequence_number`, and no delta after a `done`.
//!
//! Empty until the surface is served. The deltas themselves come from the
//! shared [`StreamParser`](crate::stream::StreamParser), which holds back
//! whatever the partial parse could still revise; this module only shapes them
//! into the events the protocol names.
