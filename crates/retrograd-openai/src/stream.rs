//! Reading a generation as it is decoded, shared by every streaming surface.
//!
//! The serving session decodes an answer piece by piece and hands each request
//! a stream of token events; a streaming surface sends each piece as it arrives
//! instead of waiting for the whole answer. What may be sent is decided here:
//! [`StreamParser`] accumulates the text, re-reads it with the model's
//! tool-call parser in its partial mode, and emits only the deltas that cannot
//! be revised anymore - text already sent is never retracted.

use std::sync::Arc;

use retrograd_agent_core::parser::{PartialAssistant, ToolCallParser};

/// One stable piece of a streamed answer.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamDelta {
    /// Prose newly safe to send.
    Content(String),
    /// A call was recognized: its id and name, to send once as a header.
    Call {
        index: usize,
        id: String,
        name: String,
    },
    /// A recognized call's arguments grew by this raw piece (possibly
    /// incomplete JSON - the final full parse validates it).
    CallArguments { index: usize, delta: String },
}

/// Accumulates one answer and diffs successive partial parses of it.
pub struct StreamParser {
    reader: Arc<dyn ToolCallParser>,
    text: String,
    /// The content already sent, as the parser spelled it.
    sent: String,
    calls: Vec<CallSent>,
}

#[derive(Default)]
struct CallSent {
    header: bool,
    arguments: String,
}

impl StreamParser {
    pub fn new(reader: Arc<dyn ToolCallParser>) -> Self {
        Self {
            reader,
            text: String::new(),
            sent: String::new(),
            calls: Vec::new(),
        }
    }

    /// Feeds one decoded piece and returns what is now safe to send.
    pub fn push(&mut self, piece: &str) -> Vec<StreamDelta> {
        self.text.push_str(piece);
        let parsed = self.reader.parse_partial(&self.text);
        self.diff(&parsed, false)
    }

    /// The answer is complete: the final full parse is authoritative and
    /// flushes everything the partial reads held back.
    pub fn finish(&mut self) -> Vec<StreamDelta> {
        let parsed = PartialAssistant::from(self.reader.parse(&self.text));
        self.diff(&parsed, true)
    }

    fn diff(&mut self, parsed: &PartialAssistant, final_: bool) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        // While only prose has been recognized, a trailing open tag of the
        // content may still turn out to open a call or a reasoning block: hold
        // it back. At the end, everything is flushed.
        let held = match final_ {
            true => 0,
            false => open_tag_len(&parsed.content),
        };
        let safe = parsed.content.len().saturating_sub(held);
        if parsed.content.starts_with(self.sent.as_str()) && safe > self.sent.len() {
            out.push(StreamDelta::Content(
                parsed.content[self.sent.len()..safe].to_owned(),
            ));
            self.sent = parsed.content[..safe].to_owned();
        }
        // A partial parse that revises a prefix already sent sends nothing: the
        // buffer is kept and the final parse reconciles.
        for (index, call) in parsed.tool_calls.iter().enumerate() {
            while self.calls.len() <= index {
                self.calls.push(CallSent::default());
            }
            let sent = &mut self.calls[index];
            // The header goes out once, with a name that can no longer grow:
            // in the retained formats the name is spelled before the
            // arguments, so a non-empty arguments string means it is closed.
            // A client keeps the first name it reads, so a partial one would
            // stick.
            if !sent.header && !call.name.is_empty() && (final_ || !call.arguments.is_empty()) {
                out.push(StreamDelta::Call {
                    index,
                    id: call.id.clone(),
                    name: call.name.clone(),
                });
                sent.header = true;
            }
            // Argument deltas only after the header (or at the end, which
            // flushes everything): a client keys them by the call's index.
            if (sent.header || final_)
                && call.arguments.starts_with(&sent.arguments)
                && call.arguments.len() > sent.arguments.len()
            {
                out.push(StreamDelta::CallArguments {
                    index,
                    delta: call.arguments[sent.arguments.len()..].to_owned(),
                });
                sent.arguments = call.arguments.clone();
            }
        }
        out
    }
}

/// The length of the trailing open tag of `text`: the longest suffix that
/// starts with `<` and has no `>` yet, i.e. a prefix that may still become a
/// call or reasoning marker. Every marker of the retained templates is a
/// `<...>` special token, so this holds back exactly the ambiguous part.
fn open_tag_len(text: &str) -> usize {
    match text.rfind('<') {
        // No tag at all: nothing to hold back.
        None => 0,
        Some(at) => match text[at..].contains('>') {
            // A closed tag is resolved: only an unclosed one is ambiguous.
            true => 0,
            false => text.len() - at,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::parser::{HermesToolCallParser, ParsedAssistant};

    fn parser() -> StreamParser {
        StreamParser::new(Arc::new(HermesToolCallParser))
    }

    fn collect(chunks: &[&str]) -> Vec<StreamDelta> {
        let mut parser = parser();
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(parser.push(chunk));
        }
        out.extend(parser.finish());
        out
    }

    fn texts(deltas: &[StreamDelta]) -> String {
        deltas
            .iter()
            .map(|delta| match delta {
                StreamDelta::Content(text) => text.as_str(),
                _ => "",
            })
            .collect()
    }

    #[test]
    fn prose_is_sent_as_it_arrives() {
        let deltas = collect(&["Hello", ", ", "world"]);
        assert_eq!(texts(&deltas), "Hello, world");
    }

    /// A trailing partial marker is held back, and what was sent is never
    /// retracted when the marker turns out to be prose after all.
    #[test]
    fn an_ambiguous_prefix_is_held_back_and_never_retracted() {
        // A parser without partial support commits everything, the ambiguous
        // prefix included; the diff still holds back an open tag.
        struct CommitAll;
        impl ToolCallParser for CommitAll {
            fn parse(&self, output: &str) -> ParsedAssistant {
                ParsedAssistant {
                    content: output.trim().to_owned(),
                    ..Default::default()
                }
            }
        }
        let mut parser = StreamParser::new(Arc::new(CommitAll));
        let first = parser.push("a <");
        assert_eq!(texts(&first), "a ", "the open tag is held back, not sent");
        let mut all = first;
        all.extend(parser.push(" b"));
        all.extend(parser.finish());
        assert_eq!(texts(&all), "a < b");
    }

    #[test]
    fn a_call_is_recognized_and_its_arguments_stream_as_deltas() {
        let deltas = collect(&[
            "before ",
            "<tool_call>{\"name\":\"move\",\"argu",
            "ments\":{\"direction\":\"left\"}}</tool_call>",
        ]);
        assert_eq!(texts(&deltas), "before");
        let call = deltas.iter().find_map(|delta| match delta {
            StreamDelta::Call { index, id, name } => Some((*index, id.clone(), name.clone())),
            _ => None,
        });
        assert_eq!(call, Some((0, "call_0".to_owned(), "move".to_owned())));
        let arguments: String = deltas
            .iter()
            .map(|delta| match delta {
                StreamDelta::CallArguments { delta, .. } => delta.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(arguments, "{\"direction\":\"left\"}");
    }

    /// The stability the streaming relies on: for the retained format, the
    /// content a partial parse commits is a prefix of what the next one
    /// commits, so the deltas never revise what was sent.
    #[test]
    fn partial_deltas_are_stable_for_the_hermes_format() {
        let full = "plan <tool_call>{\"name\":\"echo\",\"arguments\":{\"x\":1}}</tool_call> done";
        let mut parser = parser();
        let mut sent = String::new();
        let mut fed = 0;
        for end in 1..=full.len() {
            if !full.is_char_boundary(end) {
                continue;
            }
            for delta in parser.push(&full[fed..end]) {
                if let StreamDelta::Content(text) = delta {
                    sent.push_str(&text);
                }
            }
            fed = end;
        }
        parser.finish();
        let parsed = HermesToolCallParser.parse(full);
        assert_eq!(sent, parsed.content);
    }
}
