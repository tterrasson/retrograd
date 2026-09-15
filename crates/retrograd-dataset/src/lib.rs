//! Dataset parsing and SFT example preparation.
//!
//! The runtime consumes fixed-size rows. Keeping parsing and masking here
//! makes the dataset contract independent from the llama.cpp ABI and leaves a
//! clear seam for future RL algorithms.

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use retrograd_core::{Error, Result};

pub mod topk;

pub const IGNORE_LABEL: i32 = -1;

/// How much of a file [`DataFormat::infer`] looks at to recognize its shape.
/// A JSONL record's opening brace is the first byte of the first non-empty
/// line; no legitimate file needs more than this to be recognized.
const SNIFF_BYTES: usize = 64 * 1024;

/// Fills as much of `buffer` as the reader has, returning how many bytes that
/// was. `Read::read` may stop short of the buffer for reasons that have nothing
/// to do with reaching the end.
fn read_at_most(reader: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 => break,
            read => filled += read,
        }
    }
    Ok(filled)
}

/// Shape of a training file: raw text, packed into overlapping next-token
/// windows, or one chat conversation per JSONL line, masked down to its
/// assistant turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFormat {
    Text,
    ChatJsonl,
}

impl DataFormat {
    /// Format implied by a known extension. `None` - including for a missing
    /// extension - is not a guess, it is "ask [`Self::infer`] to look at the
    /// content".
    fn from_extension(path: &Path) -> Option<Self> {
        let extension = path.extension().and_then(|extension| extension.to_str())?;
        if extension.eq_ignore_ascii_case("jsonl") || extension.eq_ignore_ascii_case("json") {
            Some(Self::ChatJsonl)
        } else if extension.eq_ignore_ascii_case("txt") || extension.eq_ignore_ascii_case("md") {
            Some(Self::Text)
        } else {
            None
        }
    }

    /// Determines a dataset's format: the extension when it is one of the known
    /// ones, else the shape of the first non-empty line, else an error.
    ///
    /// Never falls back to [`Self::Text`] silently: a
    /// `.csv` whose first line is not a JSON object is refused rather than
    /// trained on as raw text without a word of warning.
    pub fn infer(path: &Path) -> Result<Self> {
        if let Some(format) = Self::from_extension(path) {
            return Ok(format);
        }
        // Only a bounded prefix: a sniff must not depend on the file's size, and
        // an upload is untrusted content that may well be one gigabyte with no
        // newline in it.
        let mut prefix = vec![0u8; SNIFF_BYTES];
        let read = {
            let mut file = fs::File::open(path)?;
            read_at_most(&mut file, &mut prefix)?
        };
        prefix.truncate(read);
        match String::from_utf8_lossy(&prefix)
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
        {
            Some(line) if line.starts_with('{') => Ok(Self::ChatJsonl),
            Some(_) => Err(Error::invalid(format!(
                "{}: could not determine the dataset format from its extension or its \
                 content; pass an explicit format (text or jsonl)",
                path.display()
            ))),
            None => Err(Error::invalid(format!(
                "{}: could not determine the dataset format; the file has no recognized \
                 extension (.jsonl, .json, .txt, .md) and no non-empty line to inspect",
                path.display()
            ))),
        }
    }
}

/// Fixed-size training rows ready for the runtime: `tokens` and `labels` are
/// both `examples * n_ctx` long, row-major, with `labels[i]` the target for
/// `tokens[i]` and [`IGNORE_LABEL`] marking positions that carry no loss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedDataset {
    pub n_ctx: usize,
    pub tokens: Vec<i32>,
    pub labels: Vec<i32>,
    pub examples: usize,
    /// Count of label positions that are not [`IGNORE_LABEL`], i.e. the
    /// denominator of the mean training loss.
    pub supervised_tokens: usize,
}

impl PreparedDataset {
    pub fn rows(&self) -> usize {
        self.examples
    }

    pub fn is_empty(&self) -> bool {
        self.examples == 0
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatExample {
    pub messages: Vec<ChatMessage>,
    /// Criteria this record is to be judged on, read only by the rollout prompt
    /// reader when `[grpo.judge]` is set - the one place a judge can learn what
    /// an expert would have answered without that answer becoming a training
    /// target. Silently unused by SFT, which has no judge to hand it to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// One parsed line of a chat-JSONL file, with its 1-based source line kept
/// alongside for error reporting.
#[derive(Clone, Debug)]
pub struct ChatRecord {
    pub line: usize,
    pub example: ChatExample,
}

/// Why a record that parsed is still not a conversation.
///
/// The three cases were a formatted `String`, which every caller then had to
/// re-read as text: `semantic_error` to build a `RecordError`, the SFT
/// preparation to build a facade error. Naming them keeps the sentence in one
/// place and gives a caller something to match on.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChatExampleError {
    #[error("messages must not be empty")]
    NoMessages,
    /// The vocabulary is closed: `system`, `user`, `assistant`. A tool or
    /// function role is a different schema, not a variant of this one.
    #[error("unknown role '{role}'")]
    UnknownRole { role: String },
    #[error("message content must not be empty")]
    EmptyContent,
}

/// Said as the facade says it, at the one place a record is validated without a
/// file to name: a caller that hands a `ChatExample` in directly passed it, so
/// it is an argument. Reading a *file* goes through [`RecordError`], which has
/// the line and answers as [`Error::Dataset`].
impl From<ChatExampleError> for Error {
    fn from(error: ChatExampleError) -> Self {
        Error::invalid(error.to_string())
    }
}

impl ChatExample {
    pub fn validate(&self) -> std::result::Result<(), ChatExampleError> {
        if self.messages.is_empty() {
            return Err(ChatExampleError::NoMessages);
        }
        for message in &self.messages {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                return Err(ChatExampleError::UnknownRole {
                    role: message.role.clone(),
                });
            }
            if message.content.is_empty() {
                return Err(ChatExampleError::EmptyContent);
            }
        }
        Ok(())
    }

    pub fn as_pairs(&self) -> Vec<(&str, &str)> {
        self.messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect()
    }
}

/// What this crate needs from a model to turn text into training rows: its
/// tokenizer, its end-of-sequence token, and its chat template. Implemented by
/// the runtime; kept as a trait so dataset preparation can be tested without a
/// loaded model.
pub trait DatasetBackend {
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>>;
    fn eos_token(&self) -> Result<i32>;
    /// Renders `messages` through the model's chat template. `add_assistant`
    /// appends the assistant turn's opening tag with no content, for tokenizing
    /// a prompt that is about to be completed rather than a finished exchange.
    fn format_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String>;
}

fn read_text(path: impl AsRef<Path>) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

retrograd_core::wire_enum! {
    /// The closed vocabulary of [`RecordError::kind`] - small and specific to a
    /// dataset's own line-by-line validation, distinct from the server's
    /// `ErrorCode` (which classifies a *request*, not a record inside one).
    ///
    /// Nothing serializes it: `as_str()` names the kind inside the message a
    /// `RecordError` turns into, so the label form of the macro is the one that
    /// applies.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum RecordErrorKind {
        /// A JSONL line with nothing but whitespace.
        EmptyRecord = "empty_record",
        /// The line does not parse as the `ChatExample` schema at all: malformed
        /// JSON, a missing field, or an unknown one (`deny_unknown_fields`).
        InvalidJson = "invalid_json",
        /// The line parses but fails a semantic rule: an empty or unknown role, no
        /// assistant turn, an implausible turn order.
        InvalidValue = "invalid_value",
    }
}

/// One problem found on one line of a chat-JSONL file, as
/// [`validate_chat_jsonl`] collects them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordError {
    /// 1-based, matching the line a text editor would show.
    pub line: usize,
    pub kind: RecordErrorKind,
    pub message: String,
}

impl RecordError {
    fn new(line: usize, kind: RecordErrorKind, message: impl Into<String>) -> Self {
        Self {
            line,
            kind,
            message: message.into(),
        }
    }
}

/// How many [`RecordError`]s [`validate_chat_jsonl`] keeps. Past this the count
/// still grows but nothing more is allocated: a wholly malformed upload has one
/// error per line, and a million of them is a memory cost, not a diagnosis.
/// Well above the twenty a caller displays, so the
/// window a client sees is never the one this cap decided.
pub const MAX_COLLECTED_RECORD_ERRORS: usize = 200;

/// What [`validate_chat_jsonl`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Validation {
    /// The first [`MAX_COLLECTED_RECORD_ERRORS`] problems, in line order.
    pub errors: Vec<RecordError>,
    /// Every problem found, including those past the collection cap - so a
    /// caller can say "20 of 4 812" rather than "20".
    pub total: usize,
}

impl Validation {
    pub fn is_valid(&self) -> bool {
        self.total == 0
    }
}

/// One line, structurally. `Err` covers only what stops a line from being a
/// `ChatExample` at all - an empty line or invalid JSON - so the semantic
/// checks always see a real example.
fn parse_line(line_number: usize, line: &str) -> std::result::Result<ChatExample, RecordError> {
    if line.trim().is_empty() {
        return Err(RecordError::new(
            line_number,
            RecordErrorKind::EmptyRecord,
            "empty JSONL record",
        ));
    }
    serde_json::from_str::<ChatExample>(line).map_err(|error| {
        RecordError::new(
            line_number,
            RecordErrorKind::InvalidJson,
            format!("invalid JSON: {error}"),
        )
    })
}

/// The semantic rules, on a line that already parsed.
///
/// Not checked here: whether a record has an assistant turn. `read_chat_jsonl`
/// also serves `retrograd-training`'s rollout prompt reader, whose whole point
/// is a *user* turn with no assistant response yet - the SFT-specific
/// requirement ("nothing to learn from a record with no answer") stays where it
/// always was, in `prepare_conversation`.
fn semantic_error(line_number: usize, example: &ChatExample) -> Option<RecordError> {
    let message = example
        .validate()
        .err()
        .map(|error| error.to_string())
        .or_else(|| implausible_alternation(example))?;
    Some(RecordError::new(
        line_number,
        RecordErrorKind::InvalidValue,
        message,
    ))
}

/// Turns implausible turn order into a message, or `None` when the sequence
/// looks like a real conversation.
///
/// `system` messages are excluded from the check - they may legitimately sit
/// anywhere - so what is left must alternate: two consecutive `user` or two
/// consecutive `assistant` turns is the shape of a malformed export, not a
/// real conversation.
fn implausible_alternation(example: &ChatExample) -> Option<String> {
    let mut previous: Option<&str> = None;
    for message in &example.messages {
        if message.role == "system" {
            continue;
        }
        if previous == Some(message.role.as_str()) {
            return Some(format!(
                "messages should alternate between user and assistant; found two \
                 consecutive '{}' turns",
                message.role
            ));
        }
        previous = Some(message.role.as_str());
    }
    None
}

/// Validates every record of a chat-JSONL file, collecting every problem
/// instead of stopping at the first: a file with ten
/// mistakes is worth ten lines, not ten round trips through ingestion.
///
/// Structural problems (invalid JSON, an unknown field) and semantic ones (a
/// bad role, empty content, no assistant turn, implausible turn order) are
/// both reported this way, each pinned to its line. An empty file is still a
/// hard error - there is no line to report a problem *on*.
pub fn validate_chat_jsonl(path: &Path) -> Result<Validation> {
    let source = read_text(path)?;
    let mut validation = Validation::default();
    let mut lines = 0usize;
    for (index, line) in source.lines().enumerate() {
        lines += 1;
        let problem = match parse_line(index + 1, line) {
            Err(error) => Some(error),
            Ok(example) => semantic_error(index + 1, &example),
        };
        if let Some(problem) = problem {
            validation.total += 1;
            if validation.errors.len() < MAX_COLLECTED_RECORD_ERRORS {
                validation.errors.push(problem);
            }
        }
    }
    if lines == 0 {
        return Err(Error::invalid(format!(
            "{} contains no JSONL records",
            path.display()
        )));
    }
    Ok(validation)
}

/// Every record of a chat-JSONL file, or the *first* problem - collecting them
/// all is [`validate_chat_jsonl`]'s job; this one has to hand back real records,
/// so it stops as soon as one line cannot become one.
pub fn read_chat_jsonl(path: &Path) -> Result<Vec<ChatRecord>> {
    let source = read_text(path)?;
    let mut records = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        let example = parse_line(line_number, line)
            .map_err(|error| dataset_error(path, line_number, error.message))?;
        if let Some(error) = semantic_error(line_number, &example) {
            return Err(dataset_error(path, line_number, error.message));
        }
        records.push(ChatRecord {
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

/// Real per-example token lengths, using the model's own tokenizer and chat
/// template - measuring, not training, so
/// there is no `n_ctx` to pack rows against and no masking to compute.
///
/// A chat-JSONL example's length is its full conversation, formatted exactly
/// as [`prepare`] would format it (`add_assistant: false`): the same text
/// `prepare_conversation` tokenizes as `full_tokens`, without building the
/// label array that goes with it. A text corpus has one length, same as
/// `retrograd_plan`'s character estimate treats it: the whole file.
pub fn measured_lengths(
    trainer: &impl DatasetBackend,
    path: impl AsRef<Path>,
    format: DataFormat,
) -> Result<Vec<u32>> {
    match format {
        DataFormat::Text => {
            let tokens = trainer.tokenize_text(&read_text(path)?)?;
            Ok(vec![tokens.len().min(u32::MAX as usize) as u32])
        }
        DataFormat::ChatJsonl => read_chat_jsonl(path.as_ref())?
            .iter()
            .map(|record| {
                let messages = record.example.as_pairs();
                let full = trainer.format_chat(&messages, false)?;
                let tokens = trainer.tokenize_text(&full)?;
                Ok(tokens.len().min(u32::MAX as usize) as u32)
            })
            .collect(),
    }
}

/// Reads and tokenizes `path` into fixed-`n_ctx` training rows, tokenizing and
/// masking as [`DataFormat`] requires: overlapping windows for text, one
/// assistant-masked row per conversation for chat JSONL.
pub fn prepare(
    trainer: &impl DatasetBackend,
    path: impl AsRef<Path>,
    format: DataFormat,
    n_ctx: usize,
) -> Result<PreparedDataset> {
    if n_ctx == 0 {
        return Err(Error::invalid("n_ctx must be greater than zero"));
    }

    match format {
        DataFormat::Text => prepare_text(trainer, &read_text(path)?, n_ctx),
        DataFormat::ChatJsonl => prepare_chat_jsonl(trainer, path.as_ref(), n_ctx),
    }
}

fn prepare_text(
    trainer: &impl DatasetBackend,
    text: &str,
    n_ctx: usize,
) -> Result<PreparedDataset> {
    let tokens = trainer.tokenize_text(text)?;
    if tokens.len() <= n_ctx {
        return Err(Error::invalid(format!(
            "text dataset requires more than {n_ctx} tokens; got {}",
            tokens.len()
        )));
    }

    let stride = (n_ctx / 2).max(1);
    let row_count = 1 + (tokens.len() - n_ctx - 1) / stride;
    let capacity = row_count
        .checked_mul(n_ctx)
        .ok_or_else(|| Error::overflow("prepared text dataset size overflows usize"))?;
    let mut rows = Vec::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut offset = 0;
    while offset + n_ctx < tokens.len() {
        rows.extend_from_slice(&tokens[offset..offset + n_ctx]);
        labels.extend_from_slice(&tokens[offset + 1..offset + n_ctx + 1]);
        offset += stride;
    }
    let examples = rows.len() / n_ctx;
    Ok(PreparedDataset {
        n_ctx,
        tokens: rows,
        labels,
        examples,
        supervised_tokens: examples * n_ctx,
    })
}

fn prepare_chat_jsonl(
    trainer: &impl DatasetBackend,
    path: &Path,
    n_ctx: usize,
) -> Result<PreparedDataset> {
    let records = read_chat_jsonl(path)?;
    let eos = trainer.eos_token()?;
    let mut tokens = Vec::new();
    let mut labels = Vec::new();
    let mut supervised_tokens = 0;
    let mut examples = 0;

    for record in records {
        let (example_tokens, example_labels) =
            prepare_conversation(trainer, &record.example, n_ctx)
                .map_err(|error| dataset_error(path, record.line, error.to_string()))?;
        supervised_tokens += example_labels
            .iter()
            .filter(|&&label| label != IGNORE_LABEL)
            .count();
        tokens.extend(
            example_tokens
                .into_iter()
                .chain(std::iter::repeat(eos))
                .take(n_ctx),
        );
        labels.extend(
            example_labels
                .into_iter()
                .chain(std::iter::repeat(IGNORE_LABEL))
                .take(n_ctx),
        );
        examples += 1;
    }

    if supervised_tokens == 0 {
        return Err(Error::invalid(format!(
            "{} contains no assistant tokens to train",
            path.display()
        )));
    }

    Ok(PreparedDataset {
        n_ctx,
        tokens,
        labels,
        examples,
        supervised_tokens,
    })
}

fn prepare_conversation(
    trainer: &impl DatasetBackend,
    example: &ChatExample,
    n_ctx: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    example.validate()?;
    if !example
        .messages
        .iter()
        .any(|message| message.role == "assistant")
    {
        return Err(Error::invalid(
            "messages must contain an assistant response",
        ));
    }

    let messages = example.as_pairs();
    let full = trainer.format_chat(&messages, false)?;
    let full_tokens = trainer.tokenize_text(&full)?;
    if full_tokens.len() < 2 {
        return Err(Error::invalid(
            "formatted conversation has fewer than two tokens",
        ));
    }
    if full_tokens.len() > n_ctx + 1 {
        return Err(Error::invalid(format!(
            "formatted conversation has {} tokens, exceeding ctx={n_ctx}",
            full_tokens.len()
        )));
    }

    // Formatting boundaries may repeat (the full conversation is the common
    // one-assistant case). Cache tokenized renderings while retaining the
    // existing starts_with validation as the correctness guard.
    let mut tokenized_prefixes = std::collections::HashMap::new();
    tokenized_prefixes.insert(full.clone(), full_tokens.clone());
    let mut labels = vec![IGNORE_LABEL; full_tokens.len() - 1];
    for (assistant_index, message) in example.messages.iter().enumerate() {
        if message.role != "assistant" {
            continue;
        }
        let prefix_messages = &messages[..assistant_index];
        let upto_messages = &messages[..=assistant_index];
        let prefix_text = trainer.format_chat(prefix_messages, true)?;
        let prefix = if let Some(tokens) = tokenized_prefixes.get(&prefix_text) {
            tokens.clone()
        } else {
            let tokens = trainer.tokenize_text(&prefix_text)?;
            tokenized_prefixes.insert(prefix_text, tokens.clone());
            tokens
        };
        let upto_text = trainer.format_chat(upto_messages, false)?;
        let upto = if let Some(tokens) = tokenized_prefixes.get(&upto_text) {
            tokens.clone()
        } else {
            let tokens = trainer.tokenize_text(&upto_text)?;
            tokenized_prefixes.insert(upto_text, tokens.clone());
            tokens
        };
        if !full_tokens.starts_with(&prefix) || !full_tokens.starts_with(&upto) {
            return Err(Error::invalid(
                "model chat template does not preserve token prefixes required for SFT masking",
            ));
        }
        for target_index in prefix.len()..upto.len() {
            if target_index > 0 && target_index <= labels.len() {
                labels[target_index - 1] = full_tokens[target_index];
            }
        }
    }
    if labels.iter().all(|&label| label == IGNORE_LABEL) {
        return Err(Error::tokenize(
            "assistant responses produced no trainable tokens",
        ));
    }
    Ok((full_tokens[..full_tokens.len() - 1].to_vec(), labels))
}

fn dataset_error(path: &Path, line: usize, message: impl std::fmt::Display) -> Error {
    Error::dataset(path.display().to_string(), line, message.to_string())
}

/// A collected record error, said as the facade says it.
///
/// The conversion is here rather than in `retrograd-core`, which knows nothing
/// of a dataset. The line survives as a number, so a frontend can still point
/// at it.
impl From<RecordError> for Error {
    fn from(error: RecordError) -> Self {
        Error::dataset(
            "",
            error.line,
            format!("{}: {}", error.kind.as_str(), error.message),
        )
    }
}

impl RecordError {
    /// The same conversion with the file named, for a caller that has the path.
    pub fn at(self, path: &Path) -> Error {
        Error::dataset(
            path.display().to_string(),
            self.line,
            format!("{}: {}", self.kind.as_str(), self.message),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                formatted.push('<');
                formatted.push_str(role);
                formatted.push('>');
                formatted.push_str(content);
            }
            if add_assistant {
                formatted.push_str("<assistant>");
            }
            Ok(formatted)
        }
    }

    fn example(messages: &[(&str, &str)]) -> ChatExample {
        ChatExample {
            messages: messages
                .iter()
                .map(|(role, content)| ChatMessage {
                    role: (*role).to_string(),
                    content: (*content).to_string(),
                })
                .collect(),
            rubric: None,
        }
    }

    fn temp_file(name: &str, source: &str) -> std::path::PathBuf {
        temp_file_ext(name, "jsonl", source)
    }

    fn temp_file_ext(name: &str, extension: &str, source: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "retrograd-dataset-{name}-{}-{}.{extension}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, source).unwrap();
        path
    }

    #[test]
    fn infers_jsonl_and_text_formats_from_a_known_extension() {
        // A known extension never touches the filesystem, so a nonexistent path
        // is fine here.
        assert_eq!(
            DataFormat::infer(Path::new("data.jsonl")).unwrap(),
            DataFormat::ChatJsonl
        );
        assert_eq!(
            DataFormat::infer(Path::new("data.JSON")).unwrap(),
            DataFormat::ChatJsonl
        );
        assert_eq!(
            DataFormat::infer(Path::new("data.txt")).unwrap(),
            DataFormat::Text
        );
        assert_eq!(
            DataFormat::infer(Path::new("data.md")).unwrap(),
            DataFormat::Text
        );
    }

    #[test]
    fn an_unknown_extension_falls_back_to_sniffing_the_first_line() {
        let jsonish = temp_file_ext("sniff-jsonish", "dat", "{\"messages\":[]}\n");
        assert_eq!(DataFormat::infer(&jsonish).unwrap(), DataFormat::ChatJsonl);
        std::fs::remove_file(&jsonish).unwrap();

        // Blank leading lines are skipped before the sniff.
        let blank_then_json = temp_file_ext("sniff-blank", "dat", "\n\n{\"messages\":[]}\n");
        assert_eq!(
            DataFormat::infer(&blank_then_json).unwrap(),
            DataFormat::ChatJsonl
        );
        std::fs::remove_file(&blank_then_json).unwrap();
    }

    #[test]
    fn an_unrecognized_extension_and_content_is_refused_rather_than_guessed_as_text() {
        // An unrecognized `.csv` must not be silently treated as raw text
        let csv = temp_file_ext("sniff-csv", "csv", "id,label\n1,a\n2,b\n");
        let error = DataFormat::infer(&csv).unwrap_err();
        assert!(error.to_string().contains("could not determine"), "{error}");
        std::fs::remove_file(&csv).unwrap();

        let empty = temp_file_ext("sniff-empty", "dat", "");
        let error = DataFormat::infer(&empty).unwrap_err();
        assert!(error.to_string().contains("could not determine"), "{error}");
        std::fs::remove_file(&empty).unwrap();
    }

    #[test]
    fn read_text_reports_missing_files_as_io_errors() {
        let path = std::env::temp_dir().join("retrograd-does-not-exist-xyz.txt");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            read_text(&path),
            Err(retrograd_core::Error::Io(_))
        ));
    }

    #[test]
    fn text_preparation_builds_overlapping_next_token_rows() {
        let prepared = prepare_text(&FakeBackend, "abcdefgh", 4).unwrap();
        assert_eq!(prepared.n_ctx, 4);
        assert_eq!(prepared.examples, 2);
        assert_eq!(prepared.supervised_tokens, 8);
        assert_eq!(
            prepared.tokens,
            b"abcdcdef"
                .iter()
                .map(|&b| i32::from(b))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            prepared.labels,
            b"bcdedefg"
                .iter()
                .map(|&b| i32::from(b))
                .collect::<Vec<_>>()
        );
        assert_eq!(prepared.rows(), 2);
        assert!(!prepared.is_empty());
    }

    #[test]
    fn text_preparation_rejects_a_dataset_without_one_full_target_row() {
        let error = prepare_text(&FakeBackend, "abcd", 4).unwrap_err();
        assert!(error.to_string().contains("requires more than 4 tokens"));

        let error = prepare_text(&FakeBackend, "", 0).unwrap_err();
        assert!(error.to_string().contains("requires more than 0 tokens"));
    }

    #[test]
    fn conversation_masks_everything_except_assistant_content() {
        let (tokens, labels) = prepare_conversation(
            &FakeBackend,
            &example(&[("system", "S"), ("user", "Q"), ("assistant", "A")]),
            64,
        )
        .unwrap();
        let supervised = labels
            .iter()
            .enumerate()
            .filter(|(_, label)| **label != IGNORE_LABEL)
            .collect::<Vec<_>>();
        assert_eq!(supervised.len(), 1);
        let (position, label) = supervised[0];
        assert_eq!(*label, i32::from(b'A'));
        assert_eq!(
            position,
            tokens.len() - 1,
            "the last input predicts the final A token"
        );
    }

    #[test]
    fn record_validation_is_matchable_not_just_printable() {
        assert_eq!(
            example(&[("tool", "x")]).validate().unwrap_err(),
            ChatExampleError::UnknownRole {
                role: "tool".to_string()
            }
        );
        assert_eq!(
            example(&[]).validate().unwrap_err(),
            ChatExampleError::NoMessages
        );
        // The facade keeps the sentence, and `?` at the SFT boundary propagates
        // it without rebuilding it by hand.
        assert_eq!(
            Error::from(ChatExampleError::EmptyContent).to_string(),
            "invalid argument: message content must not be empty"
        );
    }

    #[test]
    fn conversation_validation_rejects_invalid_records() {
        let cases = [
            (example(&[]), "messages must not be empty"),
            (
                example(&[("tool", "x"), ("assistant", "a")]),
                "unknown role",
            ),
            (
                example(&[("user", ""), ("assistant", "a")]),
                "content must not be empty",
            ),
            (example(&[("user", "question")]), "assistant response"),
        ];
        for (example, expected) in cases {
            let error = prepare_conversation(&FakeBackend, &example, 128).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }

        let error = prepare_conversation(
            &FakeBackend,
            &example(&[("user", "question"), ("assistant", "answer")]),
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeding ctx=4"), "{error}");
    }

    #[test]
    fn measured_lengths_counts_the_tokens_of_each_full_conversation() {
        let path = temp_file(
            "measured",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}\n\
             {\"messages\":[{\"role\":\"user\",\"content\":\"longer question\"},{\"role\":\"assistant\",\"content\":\"a longer answer\"}]}\n",
        );
        let lengths = measured_lengths(&FakeBackend, &path, DataFormat::ChatJsonl).unwrap();
        assert_eq!(lengths.len(), 2);
        assert!(lengths[1] > lengths[0], "{lengths:?}");
        // Matches what `prepare_conversation` would tokenize as `full_tokens`.
        let full = FakeBackend
            .format_chat(&[("user", "Q"), ("assistant", "A")], false)
            .unwrap();
        assert_eq!(
            lengths[0] as usize,
            FakeBackend.tokenize_text(&full).unwrap().len()
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn measured_lengths_of_a_text_corpus_is_the_whole_file() {
        let path = temp_file_ext("measured-text", "txt", "some plain text corpus");
        let lengths = measured_lengths(&FakeBackend, &path, DataFormat::Text).unwrap();
        assert_eq!(lengths.len(), 1);
        assert_eq!(
            lengths[0] as usize,
            FakeBackend
                .tokenize_text("some plain text corpus")
                .unwrap()
                .len()
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn jsonl_preparation_pads_rows_and_reports_line_numbers() {
        let path = temp_file(
            "valid",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}\n",
        );
        let prepared = prepare_chat_jsonl(&FakeBackend, &path, 32).unwrap();
        assert_eq!(prepared.examples, 1);
        assert_eq!(prepared.tokens.len(), 32);
        assert_eq!(prepared.labels.len(), 32);
        assert_eq!(prepared.supervised_tokens, 1);
        assert_eq!(prepared.tokens.last(), Some(&0));
        assert_eq!(prepared.labels.last(), Some(&IGNORE_LABEL));
        std::fs::remove_file(&path).unwrap();

        let path = temp_file("invalid", "{}\n");
        let error = prepare_chat_jsonl(&FakeBackend, &path, 32).unwrap_err();
        assert!(error.to_string().contains(":1:"), "{error}");
        std::fs::remove_file(path).unwrap();

        let path = temp_file(
            "blank",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}\n\n",
        );
        let error = prepare_chat_jsonl(&FakeBackend, &path, 32).unwrap_err();
        assert!(
            error.to_string().contains(":2: empty JSONL record"),
            "{error}"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn jsonl_reader_rejects_empty_files_and_preserves_record_line_numbers() {
        let empty = temp_file("empty", "");
        let error = read_chat_jsonl(&empty).unwrap_err();
        assert!(error.to_string().contains("contains no JSONL records"));
        std::fs::remove_file(&empty).unwrap();

        let path = temp_file(
            "lines",
            "\n{\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}\n",
        );
        let error = read_chat_jsonl(&path).unwrap_err();
        assert!(error.to_string().contains(":1: empty JSONL record"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn chat_jsonl_supports_multiple_conversations_and_unicode_content() {
        let path = temp_file(
            "unicode",
            "{\"messages\":[{\"role\":\"system\",\"content\":\"Answer in English\"},{\"role\":\"user\",\"content\":\"🌍\"},{\"role\":\"assistant\",\"content\":\"Hello\"}]}\n{\"messages\":[{\"role\":\"user\",\"content\":\"Q2\"},{\"role\":\"assistant\",\"content\":\"A2\"}]}\n",
        );
        let prepared = prepare_chat_jsonl(&FakeBackend, &path, 128).unwrap();
        assert_eq!(prepared.examples, 2);
        assert_eq!(prepared.tokens.len(), 256);
        assert_eq!(prepared.labels.len(), 256);
        assert_eq!(prepared.supervised_tokens, 7);
        assert!(
            prepared.labels[..128]
                .iter()
                .any(|&label| label != IGNORE_LABEL)
        );
        assert!(
            prepared.labels[128..]
                .iter()
                .any(|&label| label != IGNORE_LABEL)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn chat_jsonl_rejects_unknown_fields_and_empty_text() {
        let path = temp_file("unknown-field", "{\"messages\":[],\"extra\":true}\n");
        let error = read_chat_jsonl(&path).unwrap_err();
        assert!(error.to_string().contains(":1: invalid JSON:"));
        assert!(error.to_string().contains("extra"));
        std::fs::remove_file(&path).unwrap();

        let path = temp_file(
            "empty-content",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\"},{\"role\":\"assistant\",\"content\":\"a\"}]}\n",
        );
        let error = prepare_chat_jsonl(&FakeBackend, &path, 32).unwrap_err();
        assert!(error.to_string().contains("content must not be empty"));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn validate_chat_jsonl_collects_every_problem_instead_of_stopping_at_the_first() {
        let mut source = String::new();
        for _ in 0..3 {
            source.push_str("not json\n");
        }
        source.push_str("{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}\n");
        source.push_str(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]}\n",
        );
        let path = temp_file("collect", &source);

        let errors = validate_chat_jsonl(&path).unwrap().errors;
        assert_eq!(
            errors.len(),
            4,
            "3 malformed lines + 1 unknown role: {errors:?}"
        );
        assert_eq!(errors[0].line, 1);
        assert_eq!(errors[0].kind, RecordErrorKind::InvalidJson);
        assert_eq!(errors[3].line, 4);
        assert_eq!(errors[3].kind, RecordErrorKind::InvalidValue);
        assert!(errors[3].message.contains("unknown role"));
        // The one valid record on line 5 raises nothing.
        assert!(errors.iter().all(|error| error.line != 5));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn validate_chat_jsonl_does_not_require_an_assistant_turn() {
        // `read_chat_jsonl` also serves the rollout prompt reader
        // (`retrograd-training`), whose records are exactly a user turn with no
        // assistant response yet.
        let path = temp_file(
            "prompt-only",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a prompt with no answer\"}]}\n",
        );
        assert!(validate_chat_jsonl(&path).unwrap().is_valid());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn validate_chat_jsonl_flags_implausible_turn_order() {
        let path = temp_file(
            "alternation",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"user\",\"content\":\"b\"},{\"role\":\"assistant\",\"content\":\"c\"}]}\n",
        );
        let errors = validate_chat_jsonl(&path).unwrap().errors;
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, RecordErrorKind::InvalidValue);
        assert!(errors[0].message.contains("alternate"), "{:?}", errors[0]);
        std::fs::remove_file(&path).unwrap();

        // A system message anywhere does not count against alternation.
        let path = temp_file(
            "alternation-system",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"assistant\",\"content\":\"c\"}]}\n",
        );
        assert!(validate_chat_jsonl(&path).unwrap().is_valid());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn validate_chat_jsonl_refuses_an_empty_file() {
        let path = temp_file("validate-empty", "");
        let error = validate_chat_jsonl(&path).unwrap_err();
        assert!(error.to_string().contains("contains no JSONL records"));
        std::fs::remove_file(&path).unwrap();
    }
}
