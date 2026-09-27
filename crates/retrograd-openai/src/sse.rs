//! A completion as the `chat.completion.chunk` events a streaming client reads.
//!
//! The runtime returns a generation whole, so a stream is the whole answer cut
//! into the events the protocol expects, in the order it expects them: the
//! role, the text, one event per call - a call may arrive in a single delta -,
//! the finish reason, the usage when asked for. Until the answer exists the
//! connection carries keep-alive comments, which is what keeps a proxy from
//! cutting a request a live run takes an iteration to answer.

use crate::render::Completion;
use crate::wire::{ChatCompletionChunk, ChunkChoice, Delta, ToolCallDelta};

/// What every chunk of one answer repeats.
#[derive(Clone, Debug)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
}

/// The chunks of one answer, `[DONE]` excluded.
pub fn chunks(
    meta: &Meta,
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
            index: 0,
            delta,
            finish_reason,
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

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::wire::{FinishReason, FunctionOut, ToolCallOut, Usage};

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
        };
        let chunks = values(&chunks(&meta(), &completion, false));
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
        };
        let chunks = values(&chunks(&meta(), &completion, true));
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
