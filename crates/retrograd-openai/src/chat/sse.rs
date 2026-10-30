//! A completion as the `chat.completion.chunk` events a streaming client reads.
//!
//! Two ways to cut one answer into the events the protocol expects. A buffered
//! answer - the whole text exists - is sliced by [`chunks`], in the order the
//! protocol expects: the role, the text, one event per call, the finish
//! reason, the usage when asked for. A live answer - decoded piece by piece on
//! the serving session's thread - is cut by [`Chunker`] as the pieces arrive:
//! one chunk per stable text delta, the arguments of a call as deltas once
//! the parser recognized it. Until the answer exists the connection carries
//! keep-alive comments, which is what keeps a proxy from cutting a request a
//! live run takes an iteration to answer.

use std::sync::Arc;

use retrograd_agent_core::parser::ToolCallParser;
use retrograd_infer::FinishReason;

use crate::chat::wire::{
    ChatCompletionChunk, ChunkChoice, Delta, FinishReason as WireFinishReason, FunctionOut,
    TokenLogprobs, ToolCallDelta, ToolCallOut, Usage,
};
use crate::render::Completion;
use crate::stream::{StreamDelta, StreamParser};

/// What every chunk of one answer repeats.
#[derive(Clone, Debug)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
}

/// The chunks of one answer, `[DONE]` excluded. `index` is which choice of
/// the request the answer is (0 unless `n > 1`).
pub fn chunks(
    meta: &Meta,
    index: u32,
    completion: &Completion,
    include_usage: bool,
) -> Vec<ChatCompletionChunk> {
    // With usage requested every chunk carries the key, `null` until the last.
    let pending_usage = include_usage.then_some(None);
    let chunk = |delta: Delta, finish_reason| ChatCompletionChunk {
        id: meta.id.clone(),
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model.clone(),
        system_fingerprint: meta.system_fingerprint.clone(),
        choices: vec![ChunkChoice {
            index,
            delta,
            finish_reason,
            logprobs: None,
        }],
        usage: pending_usage,
    };
    let mut chunks = vec![chunk(
        Delta {
            role: Some("assistant"),
            ..Delta::default()
        },
        None,
    )];
    if let Some(content) = completion.content.as_ref().filter(|text| !text.is_empty()) {
        chunks.push(chunk(
            Delta {
                content: Some(content.clone()),
                ..Delta::default()
            },
            None,
        ));
    }
    for (index, call) in completion.tool_calls.iter().enumerate() {
        chunks.push(chunk(
            Delta {
                tool_calls: vec![ToolCallDelta {
                    // Bounded by the calls of one turn.
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    call: call.clone(),
                }],
                ..Delta::default()
            },
            None,
        ));
    }
    chunks.push(chunk(Delta::default(), Some(completion.finish_reason)));
    if include_usage {
        chunks.push(ChatCompletionChunk {
            choices: Vec::new(),
            usage: Some(Some(completion.usage)),
            ..chunk(Delta::default(), None)
        });
    }
    chunks
}

/// Cuts a live generation into chunks as it is decoded. The role chunk goes out
/// as soon as the request is admitted - the first byte a client can measure -
/// and every later chunk carries one stable delta: a piece of prose, or a piece
/// of a recognized call's arguments. Nothing already sent is ever revised: the
/// [`StreamParser`] holds back whatever could still change.
pub struct Chunker {
    meta: Meta,
    /// Which choice of the request this chunker streams (`n > 1` streams one
    /// `choices` entry per index).
    index: u32,
    parser: Option<StreamParser>,
    include_usage: bool,
    prompt_tokens: u64,
    /// Token events seen, end-of-generation included when one ended it.
    tokens: u64,
    /// Whether any call was streamed, which decides the finish reason.
    saw_call: bool,
}

impl Chunker {
    /// `reader` is the tool-call parser the rendering chose, if any.
    pub fn new(
        meta: Meta,
        index: u32,
        reader: Option<Arc<dyn ToolCallParser>>,
        prompt_tokens: usize,
        include_usage: bool,
    ) -> Self {
        Self {
            meta,
            index,
            parser: reader.map(StreamParser::new),
            include_usage,
            // Widening: token counts of one request.
            prompt_tokens: prompt_tokens as u64,
            tokens: 0,
            saw_call: false,
        }
    }

    /// The chunk carrying the role, sent once before the first delta.
    pub fn role(&self) -> ChatCompletionChunk {
        self.chunk(
            Delta {
                role: Some("assistant"),
                ..Delta::default()
            },
            None,
            None,
        )
    }

    /// One decoded token: its text is fed to the parser and the stable deltas
    /// come out as chunks. `text` may be empty (a partial UTF-8 sequence or a
    /// possible stop prefix held back by the detokenizer). `logprobs`, when
    /// the client asked for them, rides on the first chunk this token
    /// produced.
    pub fn token(
        &mut self,
        text: &str,
        logprobs: Option<TokenLogprobs>,
    ) -> Vec<ChatCompletionChunk> {
        self.tokens += 1;
        let deltas = match self.parser.as_mut() {
            Some(parser) => parser.push(text),
            None => match text.is_empty() {
                true => Vec::new(),
                false => vec![StreamDelta::Content(text.to_owned())],
            },
        };
        let mut chunks = self.deltas(deltas);
        if let Some(logprobs) = logprobs.filter(|entry| !entry.bytes.is_empty()) {
            // A token may emit no text yet (UTF-8, stop or parser buffering).
            // Its probability still belongs to the stream exactly once.
            if chunks.is_empty() {
                chunks.push(self.chunk(Delta::default(), None, None));
            }
            chunks[0].choices[0].logprobs = Some(logprobs);
        }
        chunks
    }

    /// The completion ended: `tail` is the text the detokenizer held back (it
    /// turned out not to be a stop), the final full parse flushes whatever the
    /// partial reads held, and the finish reason and usage close the stream.
    pub fn end(&mut self, reason: FinishReason, tail: &str) -> Vec<ChatCompletionChunk> {
        let deltas = match self.parser.as_mut() {
            Some(parser) => {
                let mut deltas = parser.push(tail);
                deltas.extend(parser.finish());
                deltas
            }
            None => match tail.is_empty() {
                true => Vec::new(),
                false => vec![StreamDelta::Content(tail.to_owned())],
            },
        };
        let mut chunks = self.deltas(deltas);
        let finish_reason = match (self.saw_call, reason) {
            (true, FinishReason::Eog | FinishReason::Stop) => WireFinishReason::ToolCalls,
            (false, FinishReason::Eog | FinishReason::Stop) => WireFinishReason::Stop,
            (_, _) => WireFinishReason::Length,
        };
        chunks.push(self.chunk(Delta::default(), Some(finish_reason), None));
        if self.include_usage {
            chunks.push(ChatCompletionChunk {
                choices: Vec::new(),
                usage: Some(Some(Usage {
                    prompt_tokens: self.prompt_tokens,
                    completion_tokens: self.tokens,
                    total_tokens: self.prompt_tokens + self.tokens,
                })),
                ..self.chunk(Delta::default(), None, None)
            });
        }
        chunks
    }

    fn deltas(&mut self, deltas: Vec<StreamDelta>) -> Vec<ChatCompletionChunk> {
        deltas
            .into_iter()
            .filter_map(|delta| match delta {
                StreamDelta::Content(text) => Some(self.chunk(
                    Delta {
                        content: Some(text),
                        ..Delta::default()
                    },
                    None,
                    None,
                )),
                // The chat surface does not stream the reasoning block as
                // content; the Responses surface reads it as its own item.
                StreamDelta::Reasoning(_) => None,
                StreamDelta::Call { index, id, name } => {
                    self.saw_call = true;
                    Some(self.call_chunk(index, id, name, String::new()))
                }
                StreamDelta::CallArguments { index, delta } => {
                    self.saw_call = true;
                    Some(self.call_chunk(index, String::new(), String::new(), delta))
                }
            })
            .collect()
    }

    fn call_chunk(
        &self,
        index: usize,
        id: String,
        name: String,
        arguments: String,
    ) -> ChatCompletionChunk {
        // Bounded by the calls of one turn.
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        self.chunk(
            Delta {
                tool_calls: vec![ToolCallDelta {
                    index,
                    call: ToolCallOut {
                        id,
                        kind: "function",
                        function: FunctionOut { name, arguments },
                    },
                }],
                ..Delta::default()
            },
            None,
            None,
        )
    }

    fn chunk(
        &self,
        delta: Delta,
        finish_reason: Option<WireFinishReason>,
        logprobs: Option<TokenLogprobs>,
    ) -> ChatCompletionChunk {
        let pending_usage = self.include_usage.then_some(None);
        ChatCompletionChunk {
            id: self.meta.id.clone(),
            object: "chat.completion.chunk",
            created: self.meta.created,
            model: self.meta.model.clone(),
            system_fingerprint: self.meta.system_fingerprint.clone(),
            choices: vec![ChunkChoice {
                index: self.index,
                delta,
                finish_reason,
                logprobs,
            }],
            usage: pending_usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::chat::wire::{FinishReason, ToolCallOut};

    fn meta() -> Meta {
        Meta {
            id: "chatcmpl-1".into(),
            created: 7,
            model: "m".into(),
            system_fingerprint: "base".into(),
        }
    }

    fn values(chunks: &[ChatCompletionChunk]) -> Vec<Value> {
        chunks
            .iter()
            .map(|chunk| serde_json::to_value(chunk).unwrap())
            .collect()
    }

    #[test]
    fn a_text_answer_is_role_content_finish() {
        let completion = Completion {
            content: Some("hello".into()),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            parse_errors: 0,
            logprobs: None,
            reasoning: String::new(),
            reasoning_tokens: 0,
        };
        let chunks = values(&chunks(&meta(), 0, &completion, false));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");
        assert_eq!(
            chunks[0]["choices"][0]["delta"],
            json!({"role": "assistant"})
        );
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], Value::Null);
        assert_eq!(
            chunks[1]["choices"][0]["delta"],
            json!({"content": "hello"})
        );
        assert_eq!(chunks[2]["choices"][0]["delta"], json!({}));
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        assert!(
            chunks.iter().all(|chunk| chunk.get("usage").is_none()),
            "no usage key unless asked for"
        );
    }

    #[test]
    fn calls_arrive_whole_one_per_chunk_and_usage_last() {
        let call = |name: &str| ToolCallOut {
            id: format!("call_{name}"),
            kind: "function",
            function: FunctionOut {
                name: name.into(),
                arguments: "{}".into(),
            },
        };
        let completion = Completion {
            content: None,
            tool_calls: vec![call("a"), call("b")],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            },
            parse_errors: 0,
            logprobs: None,
            reasoning: String::new(),
            reasoning_tokens: 0,
        };
        let chunks = values(&chunks(&meta(), 0, &completion, true));
        // role, two calls, finish, usage.
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0],
            json!({"index": 0, "id": "call_a", "type": "function",
                   "function": {"name": "a", "arguments": "{}"}})
        );
        assert_eq!(
            chunks[2]["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[3]["usage"], Value::Null);
        assert_eq!(chunks[4]["choices"], json!([]));
        assert_eq!(chunks[4]["usage"]["total_tokens"], 7);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn buffered_text_does_not_drop_a_tokens_logprobs() {
        let meta = Meta {
            id: "c".into(),
            created: 0,
            model: "m".into(),
            system_fingerprint: String::new(),
        };
        let mut chunker = Chunker::new(meta, 0, None, 1, false);
        let chunks = chunker.token(
            "",
            Some(TokenLogprobs {
                token: "a".into(),
                bytes: vec![b'a'],
                logprob: -0.5,
                top_logprobs: Vec::new(),
            }),
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].choices[0].logprobs.is_some());
    }
}
