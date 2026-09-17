//! Prompt datasets: JSONL loading, per-record validation, and tokenization
//! against the trained window.

use std::path::Path;

use retrograd_core::{Error, Result};
use retrograd_dataset::{ChatExample, read_chat_jsonl};
use retrograd_engine::Trainer;

use super::sampling::RowLayout;

#[derive(Clone, Debug)]
pub(crate) struct Prompt {
    pub(super) conversation: ChatExample,
    reward_text: String,
    reference: Option<String>,
    /// `rubric` as written on the JSONL line: criteria for this prompt alone,
    /// handed to `[grpo.judge]` and to nothing else. Unlike `reference` it is
    /// allowed on a *training* line - it never becomes a target, only something
    /// a judge reads.
    rubric: Option<String>,
}

pub(crate) fn read_prompts(path: &std::path::Path) -> Result<Vec<Prompt>> {
    read_prompt_dataset(path, false)
}

pub(crate) fn read_eval_prompts(path: &std::path::Path) -> Result<Vec<Prompt>> {
    read_prompt_dataset(path, true)
}

fn read_prompt_dataset(path: &std::path::Path, allow_reference: bool) -> Result<Vec<Prompt>> {
    let mut prompts = Vec::new();
    for mut record in read_chat_jsonl(path)? {
        record.example.validate().map_err(|message| {
            Error::invalid(format!("{}:{}: {message}", path.display(), record.line))
        })?;
        let reference = if record
            .example
            .messages
            .last()
            .is_some_and(|message| message.role == "assistant")
        {
            if !allow_reference {
                return Err(prompt_user_end_error(path, record.line));
            }
            Some(
                record
                    .example
                    .messages
                    .pop()
                    .expect("the assistant reference exists")
                    .content,
            )
        } else {
            None
        };
        let rubric = record.example.rubric.take().filter(|rubric| {
            // An empty rubric is a line that meant to declare one and did not;
            // handing "" to the judge would silently replace the run-wide
            // rubric with nothing at all.
            !rubric.trim().is_empty()
        });
        let last = record
            .example
            .messages
            .last()
            .expect("validated as non-empty");
        if last.role != "user" {
            return Err(prompt_user_end_error(path, record.line));
        }
        prompts.push(Prompt {
            reward_text: last.content.clone(),
            conversation: record.example,
            reference,
            rubric,
        });
    }
    if prompts.is_empty() {
        return Err(Error::invalid(format!(
            "{}: prompt dataset must not be empty",
            path.display()
        )));
    }
    Ok(prompts)
}

fn prompt_user_end_error(path: &Path, line: usize) -> Error {
    Error::invalid(format!(
        "{}:{line}: prompt messages must end with a user message",
        path.display()
    ))
}

impl Prompt {
    pub(crate) fn reward_text(&self) -> &str {
        &self.reward_text
    }

    pub(crate) fn observed(&self, key: String) -> retrograd_observe::ObservedPrompt {
        retrograd_observe::ObservedPrompt {
            key,
            messages: self
                .conversation
                .messages
                .iter()
                .map(|message| {
                    retrograd_observe::ObservedMessage::text(&message.role, &message.content)
                })
                .collect(),
            reward_text: Some(self.reward_text.clone()),
            metadata: None,
        }
    }

    pub(crate) fn reference(&self) -> Option<&str> {
        self.reference.as_deref()
    }

    pub(super) fn rubric(&self) -> Option<&str> {
        self.rubric.as_deref()
    }
}

/// Tokenizes every training prompt once, up front. An oversized prompt then
/// aborts before any rollout compute - naming its JSONL line - instead of
/// failing mid-run once the cycling reaches it, and the update loop reuses
/// the tokenized rows across updates. `generation_budget` is the room that
/// must remain in the trained window next to the prompt: the full fixed
/// Dr. GRPO budget, or 1 for PPO's truncation-tolerant sampling.
pub(crate) fn tokenize_prompts(
    trainer: &Trainer,
    prompts: &[Prompt],
    layout: &RowLayout,
    generation_budget: usize,
    path: &std::path::Path,
) -> Result<Vec<Vec<i32>>> {
    prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let locate = |message: String| {
                format!("{}: prompt {}: {message}", path.display(), index + 1)
            };
            let tokens = tokenize_prompt(trainer, prompt, layout).map_err(|error| match error {
                Error::InvalidArgument(message) => Error::InvalidArgument(locate(message)),
                Error::Runtime(message) => Error::Runtime(locate(message)),
                other => other,
            })?;
            if tokens
                .len()
                .checked_add(generation_budget)
                .is_none_or(|required| required > layout.window)
            {
                return Err(Error::invalid(locate(format!(
                    "{} prompt tokens plus the generation budget {} exceed the trained window {}; raise training.ctx or shorten the prompt",
                    tokens.len(),
                    generation_budget,
                    layout.window
                ))));
            }
            Ok(tokens)
        })
        .collect()
}

/// Tokenizes a prompt and checks it leaves room to generate in the trained
/// window.
pub(crate) fn tokenize_prompt(
    trainer: &Trainer,
    prompt: &Prompt,
    layout: &RowLayout,
) -> Result<Vec<i32>> {
    let messages = prompt.conversation.as_pairs();
    let formatted = trainer.format_chat(&messages, true)?;
    let tokens = trainer.tokenize_text(&formatted)?;
    if tokens.is_empty() {
        return Err(Error::tokenize("prompt tokenized to zero tokens"));
    }
    if tokens.len() + 1 > layout.window {
        return Err(Error::invalid(format!(
            "prompt of {} tokens leaves no room to generate in the trained window {}",
            tokens.len(),
            layout.window
        )));
    }
    Ok(tokens)
}
