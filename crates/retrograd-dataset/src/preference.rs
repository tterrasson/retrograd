//! Preference records: one prompt, two responses to it, and which of the two is
//! preferred.
//!
//! The two responses are trained as two sequences that share the prompt token
//! for token. That identity is not assumed: a pair is refused unless the chat
//! template renders both conversations over the same prefix, because the
//! objective compares the two log-probabilities of a response under one
//! context, and a scorer that decodes the prompt once relies on it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result};

use crate::chat_template::TemplateTool;
use crate::{
    ChatExample, ChatExampleError, ChatMessage, DatasetBackend, RecordError, RecordErrorKind,
    ToolConversationRenderer, Validation, assistant_label_spans, dataset_error,
    implausible_alternation, read_text, validate_lines,
};

/// One line of a preference file.
///
/// `prompt`, `chosen` and `rejected` are chat messages in the SFT schema, tool
/// calls and observations included; `tools` is the catalog offered to both
/// responses.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceExample {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<TemplateTool>,
    pub prompt: Vec<ChatMessage>,
    pub chosen: Vec<ChatMessage>,
    pub rejected: Vec<ChatMessage>,
    /// Free-form provenance, kept for reports; never read by training.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Which response of a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Chosen,
    Rejected,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chosen => "chosen",
            Self::Rejected => "rejected",
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a record that parsed is still not a preference pair.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PreferenceExampleError {
    #[error(
        "prompt must not be empty and must end on a user turn, or on the tool observations \
         that answer a call: the responses are what the model says next"
    )]
    PromptMustEndWithUser,
    #[error("{side} must not be empty and must start with an assistant turn")]
    ResponseMustStartWithAssistant { side: Side },
    #[error("a system message may only open the prompt")]
    MisplacedSystem,
    #[error("prompt + {side}: {source}")]
    Conversation {
        side: Side,
        #[source]
        source: ChatExampleError,
    },
    #[error("prompt + {side}: {message}")]
    TurnOrder { side: Side, message: String },
    #[error("chosen and rejected are the same response: the pair prefers nothing")]
    IdenticalResponses,
}

/// A record handed in directly is an argument; a record read from a file goes
/// through [`RecordError`], which has the line.
impl From<PreferenceExampleError> for Error {
    fn from(error: PreferenceExampleError) -> Self {
        Error::invalid(error.to_string())
    }
}

impl PreferenceExample {
    pub fn side(&self, side: Side) -> &[ChatMessage] {
        match side {
            Side::Chosen => &self.chosen,
            Side::Rejected => &self.rejected,
        }
    }

    /// The whole conversation of one side, as the chat record it renders as.
    pub fn conversation(&self, side: Side) -> ChatExample {
        ChatExample {
            tools: self.tools.clone(),
            messages: self.prompt.iter().chain(self.side(side)).cloned().collect(),
            ..ChatExample::default()
        }
    }

    /// Whether the pair is rendered through the tool path. Decided for the
    /// record rather than for each side, so the two sides always share a
    /// rendering and therefore a prompt prefix.
    pub fn is_tool_record(&self) -> bool {
        Side::BOTH
            .iter()
            .any(|&side| self.conversation(side).is_tool_record())
    }

    pub fn validate(&self) -> std::result::Result<(), PreferenceExampleError> {
        if !self
            .prompt
            .last()
            .is_some_and(|message| matches!(message.role.as_str(), "user" | "tool"))
        {
            return Err(PreferenceExampleError::PromptMustEndWithUser);
        }
        let leading_system = self
            .prompt
            .iter()
            .take_while(|message| message.role == "system")
            .count();
        if self.prompt[leading_system..]
            .iter()
            .chain(&self.chosen)
            .chain(&self.rejected)
            .any(|message| message.role == "system")
        {
            return Err(PreferenceExampleError::MisplacedSystem);
        }
        for side in Side::BOTH {
            if !self
                .side(side)
                .first()
                .is_some_and(|message| message.role == "assistant")
            {
                return Err(PreferenceExampleError::ResponseMustStartWithAssistant { side });
            }
            let conversation = self.conversation(side);
            conversation
                .validate()
                .map_err(|source| PreferenceExampleError::Conversation { side, source })?;
            if let Some(message) = implausible_alternation(&conversation) {
                return Err(PreferenceExampleError::TurnOrder { side, message });
            }
        }
        if self.chosen == self.rejected {
            return Err(PreferenceExampleError::IdenticalResponses);
        }
        Ok(())
    }
}

impl Side {
    pub const BOTH: [Side; 2] = [Side::Chosen, Side::Rejected];
}

/// One parsed line of a preference file, with its 1-based source line.
#[derive(Clone, Debug)]
pub struct PreferenceRecord {
    pub line: usize,
    pub example: PreferenceExample,
}

fn parse_line(
    line_number: usize,
    line: &str,
) -> std::result::Result<PreferenceExample, RecordError> {
    if line.trim().is_empty() {
        return Err(RecordError::new(
            line_number,
            RecordErrorKind::EmptyRecord,
            "empty JSONL record",
        ));
    }
    let example = serde_json::from_str::<PreferenceExample>(line).map_err(|error| {
        RecordError::new(
            line_number,
            RecordErrorKind::InvalidJson,
            format!("invalid preference record: {error}"),
        )
    })?;
    example.validate().map_err(|error| {
        RecordError::new(
            line_number,
            RecordErrorKind::InvalidValue,
            error.to_string(),
        )
    })?;
    Ok(example)
}

/// Validates every record of a preference file, collecting every problem
/// instead of stopping at the first, as [`crate::validate_chat_jsonl`] does.
pub fn validate_preference_jsonl(path: &Path) -> Result<Validation> {
    validate_lines(path, |line_number, line| {
        parse_line(line_number, line).err()
    })
}

/// Every record of a preference file, or the first problem.
pub fn read_preference_jsonl(path: &Path) -> Result<Vec<PreferenceRecord>> {
    let source = read_text(path)?;
    let mut records = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        let example = parse_line(line_number, line).map_err(|error| error.at(path))?;
        records.push(PreferenceRecord {
            line: line_number,
            example,
        });
    }
    if records.is_empty() {
        return Err(Error::invalid(format!(
            "{} contains no JSONL records",
            path.display()
        )));
    }
    Ok(records)
}

/// One response of a prepared pair: the whole sequence, prompt included, and
/// which of its tokens are trained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedSequence {
    pub tokens: Vec<i32>,
    /// `train_mask[i]`: token `i` is a trained target. False over the whole
    /// prompt, and over what the response carries that the model did not write
    /// (tool observations, framing).
    pub train_mask: Vec<bool>,
}

impl PreparedSequence {
    pub fn trained_tokens(&self) -> usize {
        self.train_mask.iter().filter(|&&train| train).count()
    }
}

/// A pair ready for the objective. Both sequences start with the same
/// `prompt_len` tokens, and neither trains any of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedPair {
    pub prompt_len: usize,
    pub chosen: PreparedSequence,
    pub rejected: PreparedSequence,
}

impl PreparedPair {
    pub fn side(&self, side: Side) -> &PreparedSequence {
        match side {
            Side::Chosen => &self.chosen,
            Side::Rejected => &self.rejected,
        }
    }
}

fn prefix_not_preserved() -> Error {
    Error::tokenize(
        "model chat template does not preserve the prompt prefix required for preference pairs",
    )
}

/// Renders preference records into [`PreparedPair`]s, holding the tool
/// renderer's per-catalog decisions for as long as it lives - one dataset.
pub struct PreferencePreparer<'b, B: DatasetBackend + ?Sized> {
    backend: &'b B,
    tools: ToolConversationRenderer<'b, B>,
}

impl<'b, B: DatasetBackend> PreferencePreparer<'b, B> {
    pub fn new(backend: &'b B) -> Self {
        Self {
            backend,
            tools: ToolConversationRenderer::new(backend),
        }
    }

    /// Tokenizes and masks both responses of `example`. A sequence longer than
    /// `n_ctx` is refused rather than truncated: cutting the end of a response
    /// changes what the pair prefers.
    pub fn prepare(&mut self, example: &PreferenceExample, n_ctx: usize) -> Result<PreparedPair> {
        example.validate()?;
        let (prompt_len, chosen, rejected) = match example.is_tool_record() {
            true => self.tool_sides(example)?,
            false => self.plain_sides(example)?,
        };
        for (side, sequence) in [(Side::Chosen, &chosen), (Side::Rejected, &rejected)] {
            if sequence.tokens.len() > n_ctx {
                return Err(Error::invalid(format!(
                    "formatted {side} conversation has {} tokens, exceeding ctx={n_ctx}",
                    sequence.tokens.len()
                )));
            }
            if prompt_len == 0 || sequence.trained_tokens() == 0 {
                return Err(Error::tokenize(format!(
                    "the {side} response produced no trainable tokens"
                )));
            }
        }
        Ok(PreparedPair {
            prompt_len,
            chosen,
            rejected,
        })
    }

    /// The plain path: the prompt rendered as a generation prompt is the shared
    /// prefix, and each response is masked to its assistant turns by prefix
    /// differences, as SFT masks a conversation.
    fn plain_sides(
        &self,
        example: &PreferenceExample,
    ) -> Result<(usize, PreparedSequence, PreparedSequence)> {
        let prompt = example
            .prompt
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect::<Vec<_>>();
        let prefix = self
            .backend
            .tokenize_text(&self.backend.format_chat(&prompt, true)?)?;
        let [chosen, rejected] = Side::BOTH.map(|side| -> Result<PreparedSequence> {
            let conversation = example.conversation(side);
            let messages = conversation.as_pairs();
            let full = self.backend.format_chat(&messages, false)?;
            let tokens = self.backend.tokenize_text(&full)?;
            if !tokens.starts_with(&prefix) {
                return Err(prefix_not_preserved());
            }
            let spans = assistant_label_spans(
                self.backend,
                &messages,
                &full,
                &tokens,
                example.prompt.len(),
            )?
            .ok_or_else(prefix_not_preserved)?;
            let mut train_mask = vec![false; tokens.len()];
            for target in spans.into_iter().flatten() {
                if target > 0 && target < train_mask.len() {
                    train_mask[target] = true;
                }
            }
            Ok(PreparedSequence { tokens, train_mask })
        });
        Ok((prefix.len(), chosen?, rejected?))
    }

    /// The tool path: each side is the stream a rollout of its conversation
    /// would have produced. The prompt's own assistant turns are context, so
    /// training starts at the first turn of the response, and the stream up to
    /// that turn is the shared prefix.
    fn tool_sides(
        &mut self,
        example: &PreferenceExample,
    ) -> Result<(usize, PreparedSequence, PreparedSequence)> {
        let prompt_turns = example
            .prompt
            .iter()
            .filter(|message| message.role == "assistant")
            .count();
        let mut sides = Vec::with_capacity(2);
        for side in Side::BOTH {
            let conversation = example.conversation(side);
            let turns = conversation
                .messages
                .iter()
                .filter(|message| message.role == "assistant")
                .count();
            let stream = self.tools.stream(&conversation)?;
            // Each assistant turn is one run of trained tokens: its text and the
            // end-of-generation token that closes it, with framing between.
            let starts = (1..stream.train_mask.len())
                .filter(|&index| stream.train_mask[index] && !stream.train_mask[index - 1])
                .collect::<Vec<_>>();
            if starts.len() != turns {
                return Err(prefix_not_preserved());
            }
            let start = starts[prompt_turns];
            let mut train_mask = stream.train_mask;
            train_mask[..start].fill(false);
            sides.push((
                start,
                PreparedSequence {
                    tokens: stream.tokens,
                    train_mask,
                },
            ));
        }
        let (rejected_start, rejected) = sides.pop().expect("two sides");
        let (chosen_start, chosen) = sides.pop().expect("two sides");
        if chosen_start != rejected_start
            || chosen.tokens[..chosen_start] != rejected.tokens[..rejected_start]
        {
            return Err(prefix_not_preserved());
        }
        Ok((chosen_start, chosen, rejected))
    }
}

/// [`PreferencePreparer::prepare`] for one record.
pub fn prepare_pair(
    backend: &impl DatasetBackend,
    example: &PreferenceExample,
    n_ctx: usize,
) -> Result<PreparedPair> {
    PreferencePreparer::new(backend).prepare(example, n_ctx)
}

/// Every pair of a preference file, prepared, or the first record that cannot
/// be - named by its line.
pub fn prepare_preference_jsonl(
    backend: &impl DatasetBackend,
    path: &Path,
    n_ctx: usize,
) -> Result<Vec<PreparedPair>> {
    let mut preparer = PreferencePreparer::new(backend);
    read_preference_jsonl(path)?
        .iter()
        .map(|record| {
            preparer
                .prepare(&record.example, n_ctx)
                .map_err(|error| dataset_error(path, record.line, error))
        })
        .collect()
}

/// The real length of every pair of a preference file: the tokens of both
/// sequences together, which is what a pair costs inside one optimizer window.
pub fn preference_measured_lengths(
    backend: &impl DatasetBackend,
    path: impl AsRef<Path>,
) -> Result<Vec<u32>> {
    prepare_preference_jsonl(backend, path.as_ref(), usize::MAX)?
        .iter()
        .map(|pair| {
            let length = pair
                .chosen
                .tokens
                .len()
                .saturating_add(pair.rejected.tokens.len());
            Ok(length.min(u32::MAX as usize) as u32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::ChatToolCall;

    struct FakeBackend;

    impl DatasetBackend for FakeBackend {
        fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
            Ok(text.bytes().map(i32::from).collect())
        }

        fn eos_token(&self) -> Result<i32> {
            Ok(0)
        }

        fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
            let mut formatted = String::new();
            for (role, content) in messages {
                formatted.push_str(&format!("<{role}>{content}"));
            }
            if add_assistant {
                formatted.push_str("<assistant>");
            }
            Ok(formatted)
        }
    }

    /// Opens a finished assistant turn with another tag than the generation
    /// prompt does, so the prompt is not a token prefix of the conversation.
    struct RewritingBackend;

    impl DatasetBackend for RewritingBackend {
        fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
            FakeBackend.tokenize_text(text)
        }

        fn eos_token(&self) -> Result<i32> {
            Ok(0)
        }

        fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
            let mut formatted = FakeBackend.format_chat(messages, add_assistant)?;
            if !add_assistant {
                formatted = formatted.replace("<assistant>", "<model>");
            }
            Ok(formatted)
        }
    }

    fn messages(turns: &[(&str, &str)]) -> Vec<ChatMessage> {
        turns
            .iter()
            .map(|(role, content)| ChatMessage::text(*role, *content))
            .collect()
    }

    fn pair(
        prompt: &[(&str, &str)],
        chosen: &[(&str, &str)],
        rejected: &[(&str, &str)],
    ) -> PreferenceExample {
        PreferenceExample {
            prompt: messages(prompt),
            chosen: messages(chosen),
            rejected: messages(rejected),
            ..PreferenceExample::default()
        }
    }

    fn simple() -> PreferenceExample {
        pair(
            &[("system", "S"), ("user", "Q")],
            &[("assistant", "yes")],
            &[("assistant", "no")],
        )
    }

    fn text(text: &str) -> Vec<i32> {
        text.bytes().map(i32::from).collect()
    }

    fn trained_text(sequence: &PreparedSequence) -> String {
        sequence
            .tokens
            .iter()
            .zip(&sequence.train_mask)
            .filter(|(_, train)| **train)
            .map(|(&token, _)| token as u8 as char)
            .collect()
    }

    #[test]
    fn a_record_reads_from_its_documented_shape() {
        let line = json!({
            "prompt": [{"role": "user", "content": "Q"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
            "metadata": {"source": "test"}
        })
        .to_string();
        let example = parse_line(1, &line).unwrap();
        assert_eq!(example.metadata["source"], "test");

        let unknown = json!({
            "prompt": [{"role": "user", "content": "Q"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
            "score": 1
        })
        .to_string();
        assert_eq!(
            parse_line(1, &unknown).unwrap_err().kind,
            RecordErrorKind::InvalidJson
        );
    }

    #[test]
    fn each_rule_is_its_own_refusal() {
        simple().validate().unwrap();

        let cases = [
            (
                pair(&[], &[("assistant", "a")], &[("assistant", "b")]),
                PreferenceExampleError::PromptMustEndWithUser,
            ),
            (
                pair(
                    &[("user", "Q"), ("assistant", "A")],
                    &[("assistant", "a")],
                    &[("assistant", "b")],
                ),
                PreferenceExampleError::PromptMustEndWithUser,
            ),
            (
                pair(&[("user", "Q")], &[], &[("assistant", "b")]),
                PreferenceExampleError::ResponseMustStartWithAssistant { side: Side::Chosen },
            ),
            (
                pair(&[("user", "Q")], &[("assistant", "a")], &[("user", "b")]),
                PreferenceExampleError::ResponseMustStartWithAssistant {
                    side: Side::Rejected,
                },
            ),
            (
                pair(
                    &[("user", "Q"), ("system", "late"), ("user", "R")],
                    &[("assistant", "a")],
                    &[("assistant", "b")],
                ),
                PreferenceExampleError::MisplacedSystem,
            ),
            (
                pair(
                    &[("user", "Q")],
                    &[("assistant", "a"), ("system", "S")],
                    &[("assistant", "b")],
                ),
                PreferenceExampleError::MisplacedSystem,
            ),
            (
                pair(
                    &[("user", "Q")],
                    &[("assistant", "")],
                    &[("assistant", "b")],
                ),
                PreferenceExampleError::Conversation {
                    side: Side::Chosen,
                    source: ChatExampleError::EmptyContent,
                },
            ),
            (
                pair(
                    &[("user", "Q")],
                    &[("assistant", "a")],
                    &[("assistant", "a")],
                ),
                PreferenceExampleError::IdenticalResponses,
            ),
        ];
        for (example, expected) in cases {
            assert_eq!(example.validate().unwrap_err(), expected, "{example:?}");
        }

        let error = pair(
            &[("user", "Q"), ("user", "Q again")],
            &[("assistant", "a")],
            &[("assistant", "b")],
        )
        .validate()
        .unwrap_err();
        assert!(
            matches!(
                error,
                PreferenceExampleError::TurnOrder {
                    side: Side::Chosen,
                    ..
                }
            ),
            "{error}"
        );
        assert!(
            error.to_string().starts_with("prompt + chosen: "),
            "{error}"
        );
    }

    #[test]
    fn a_multi_turn_response_and_prompt_are_accepted() {
        pair(
            &[("user", "Q"), ("assistant", "A"), ("user", "R")],
            &[("assistant", "a"), ("user", "more"), ("assistant", "c")],
            &[("assistant", "b")],
        )
        .validate()
        .unwrap();
    }

    #[test]
    fn the_prompt_is_shared_and_only_the_response_is_trained() {
        let prepared = prepare_pair(&FakeBackend, &simple(), 64).unwrap();
        let prefix = text("<system>S<user>Q<assistant>");
        assert_eq!(prepared.prompt_len, prefix.len());
        assert!(prepared.chosen.tokens.starts_with(&prefix));
        assert!(prepared.rejected.tokens.starts_with(&prefix));
        assert_eq!(trained_text(&prepared.chosen), "yes");
        assert_eq!(trained_text(&prepared.rejected), "no");
        assert!(!prepared.chosen.train_mask[..prepared.prompt_len].contains(&true));
    }

    #[test]
    fn assistant_turns_of_the_prompt_are_context() {
        let example = pair(
            &[("user", "Q"), ("assistant", "A"), ("user", "R")],
            &[("assistant", "a"), ("user", "S"), ("assistant", "c")],
            &[("assistant", "b")],
        );
        let prepared = prepare_pair(&FakeBackend, &example, 128).unwrap();
        assert_eq!(trained_text(&prepared.chosen), "ac");
        assert_eq!(trained_text(&prepared.rejected), "b");
    }

    #[test]
    fn a_template_that_rewrites_the_prefix_is_refused() {
        let error = prepare_pair(&RewritingBackend, &simple(), 64).unwrap_err();
        assert!(error.to_string().contains("prompt prefix"), "{error}");
    }

    #[test]
    fn a_sequence_longer_than_the_context_is_refused_not_truncated() {
        let full = text("<system>S<user>Q<assistant>yes").len();
        prepare_pair(&FakeBackend, &simple(), full).unwrap();
        let error = prepare_pair(&FakeBackend, &simple(), full - 1).unwrap_err();
        assert!(
            error.to_string().contains("chosen conversation has"),
            "{error}"
        );
    }

    #[test]
    fn a_file_is_validated_line_by_line_and_measured_per_pair() {
        let good = serde_json::to_string(&simple()).unwrap();
        let bad = serde_json::to_string(&pair(
            &[("user", "Q")],
            &[("assistant", "a")],
            &[("assistant", "a")],
        ))
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "retrograd-preference-{}-validate.jsonl",
            std::process::id()
        ));
        std::fs::write(&path, format!("{good}\n{bad}\n\n")).unwrap();
        let validation = validate_preference_jsonl(&path).unwrap();
        assert_eq!(validation.total, 2);
        assert_eq!(validation.errors[0].line, 2);
        assert_eq!(validation.errors[0].kind, RecordErrorKind::InvalidValue);
        assert_eq!(validation.errors[1].kind, RecordErrorKind::EmptyRecord);
        let error = read_preference_jsonl(&path).unwrap_err();
        assert!(matches!(error, Error::Dataset { line: 2, .. }), "{error:?}");

        std::fs::write(&path, format!("{good}\n")).unwrap();
        let lengths = preference_measured_lengths(&FakeBackend, &path).unwrap();
        let expected = text("<system>S<user>Q<assistant>yes").len()
            + text("<system>S<user>Q<assistant>no").len();
        assert_eq!(lengths, vec![expected as u32]);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_tool_pair_trains_the_policy_turns_of_the_response_only() {
        let backend = crate::tool_conversation::tests::TemplateBackend::default();
        let call = ChatToolCall {
            id: None,
            name: "run".into(),
            arguments: json!({"cmd": "ls"}),
        };
        let mut example = pair(&[("user", "go")], &[], &[("assistant", "I cannot")]);
        example.tools = vec![TemplateTool {
            name: "run".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        }];
        example.chosen = vec![
            ChatMessage {
                tool_calls: vec![call],
                ..ChatMessage::text("assistant", "")
            },
            ChatMessage::text("tool", "a.txt"),
            ChatMessage::text("assistant", "done"),
        ];
        let prepared = prepare_pair(&backend, &example, 4096).unwrap();
        assert_eq!(
            prepared.chosen.tokens[..prepared.prompt_len],
            prepared.rejected.tokens[..prepared.prompt_len]
        );
        assert!(!prepared.chosen.train_mask[..prepared.prompt_len].contains(&true));
        // The observation sits between two trained runs and is not one.
        let observation = text("a.txt");
        let at = prepared
            .chosen
            .tokens
            .windows(observation.len())
            .position(|window| window == observation)
            .unwrap();
        assert!(!prepared.chosen.train_mask[at..at + observation.len()].contains(&true));
        assert!(trained_text(&prepared.rejected).starts_with("I cannot"));
    }
}
