//! Reading `observe.jsonl` back: the one reader of the format.
//!
//! [`ObserveIndex`] follows the file as it grows. It keeps **offsets and
//! summaries** in memory - where each record starts, and the few numbers a
//! listing shows - and reads a text only when it is asked for one, so an index
//! over a long agentic run stays small.
//!
//! The rules it applies are the format's own:
//!
//! - a batch counts only once all of its `batch_len` lines are there; an
//!   unfinished one at the end is simply not read yet, and one broken off by a
//!   newer batch is dropped;
//! - a `run` record without `resumed_from_update` starts over; one with
//!   `resumed_from_update = K` supersedes, in the earlier segments, every update
//!   above `K` - the resumed run is about to produce them again;
//! - a prompt is named by its key within its segment;
//! - a `selection` or an `outcome` record overrides what the rollout record said
//!   about eligibility and training, because it is written later, by the actor
//!   that decided.
//!
//! A line of another schema version is an error, not something to guess at.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::IgnoredAny;
use serde_json::{Map, Value};

use crate::SCHEMA_VERSION;
use crate::writer::LOG_FILE;

/// What reading an export can fail with.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("{}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error(
        "{} has a record of schema version {version} at byte {offset}; this build reads version {}",
        path.display(),
        SCHEMA_VERSION
    )]
    Version {
        path: PathBuf,
        offset: u64,
        version: u32,
    },
    #[error("{} changed under the reader at byte {offset}", path.display())]
    Changed { path: PathBuf, offset: u64 },
}

/// Where a record is in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LineRef {
    offset: u64,
    len: usize,
}

/// A `run` record: one per segment.
#[derive(Clone, Debug, PartialEq)]
pub struct RunHeader {
    pub segment: u32,
    /// `ppo` | `grpo` | `agent_grpo`.
    pub algorithm: String,
    pub model: String,
    pub resumed_from_update: Option<u32>,
    /// One update in `every` exports its texts. Absent from an export written
    /// before the run record carried it.
    pub every: Option<u32>,
    pub params: Map<String, Value>,
    pub time: Option<String>,
}

/// One update, as a listing shows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UpdateEntry {
    pub segment: u32,
    pub update: u32,
    /// `completed` | `skipped`, once the `update` record is written.
    pub status: Option<String>,
    pub time: Option<String>,
    /// The scalars the update published. Non-finite values are absent.
    pub metrics: BTreeMap<String, f32>,
    pub groups: usize,
    pub rollouts: usize,
    /// Members confirmed trained by an `outcome` record.
    pub trained: usize,
    pub reward_mean: Option<f32>,
    pub reward_min: Option<f32>,
    pub reward_max: Option<f32>,
}

impl UpdateEntry {
    /// Whether this update exported its texts.
    pub fn texts(&self) -> bool {
        self.rollouts > 0
    }
}

/// One rollout without its text, its effective state already merged from the
/// `selection` and `outcome` records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemberEntry {
    /// `None` for PPO, which has no groups.
    pub group: Option<usize>,
    pub member: usize,
    /// Key of the prompt record.
    pub prompt: String,
    pub seed: u64,
    pub tokens: usize,
    pub truncated: bool,
    pub reward: Option<f32>,
    pub reward_raw: Option<f32>,
    pub judge_term: Option<f32>,
    pub advantage: Option<f32>,
    pub advantage_min: Option<f32>,
    pub advantage_max: Option<f32>,
    pub eligible: Option<bool>,
    /// `Some(true)` only once an `outcome` confirms it.
    pub trained: Option<bool>,
    pub skip_reason: Option<String>,
    /// Assistant messages of a conversation; zero for a completion.
    pub turns: usize,
    pub tool_calls: usize,
    /// Tool results marked as errors.
    pub tool_errors: usize,
}

/// The members of one group, sorted by member index.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupEntry {
    pub group: Option<usize>,
    pub prompt: String,
    pub members: Vec<MemberEntry>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

/// A `prompt` record.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PromptRecord {
    pub key: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub reward_text: Option<String>,
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StepReward {
    pub step_index: usize,
    pub kind: String,
    pub reward: f32,
    #[serde(default)]
    pub message_indices: Vec<usize>,
}

/// What a rollout said.
#[derive(Clone, Debug, PartialEq)]
pub enum RolloutText {
    Completion(String),
    Conversation {
        /// After the scenario prefix when `prefix` is true.
        messages: Vec<Message>,
        prefix: bool,
        step_rewards: Vec<StepReward>,
        terminal_reward_raw: Option<f32>,
        judge_explanation: Option<String>,
        /// The entries that differ from the scenario's.
        metadata: Map<String, Value>,
    },
}

/// A rollout with its text.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutRecord {
    pub entry: MemberEntry,
    pub text: RolloutText,
}

/// Everything a record may carry that the index keeps, and nothing it does
/// not: texts are skipped by the parser, not stored.
#[derive(Debug, Deserialize)]
struct IndexLine {
    v: u32,
    #[serde(rename = "type")]
    kind: String,
    segment: u32,
    batch_id: u64,
    batch_index: usize,
    batch_len: usize,
    #[serde(default)]
    time: Option<String>,
    // run
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    resumed_from_update: Option<u32>,
    #[serde(default)]
    every: Option<u32>,
    #[serde(default)]
    params: Option<Map<String, Value>>,
    // prompt
    #[serde(default)]
    key: Option<String>,
    // rollout, selection, outcome, update
    #[serde(default)]
    update: Option<u32>,
    #[serde(default)]
    group: Option<usize>,
    #[serde(default)]
    member: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    tokens: Option<usize>,
    #[serde(default)]
    truncated: Option<bool>,
    #[serde(default)]
    reward: Option<f32>,
    #[serde(default)]
    reward_raw: Option<f32>,
    #[serde(default)]
    judge_term: Option<f32>,
    #[serde(default)]
    advantage: Option<f32>,
    #[serde(default)]
    advantage_min: Option<f32>,
    #[serde(default)]
    advantage_max: Option<f32>,
    #[serde(default)]
    eligible: Option<bool>,
    #[serde(default)]
    trained: Option<bool>,
    #[serde(default)]
    skip_reason: Option<String>,
    #[serde(default)]
    messages: Option<Vec<MessageShape>>,
    #[serde(default)]
    entries: Option<Vec<EntryLine>>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    metrics: Option<BTreeMap<String, Option<f32>>>,
}

/// The shape of a message, for counting; its content is parsed past.
#[derive(Debug, Deserialize)]
struct MessageShape {
    role: String,
    #[serde(default)]
    tool_calls: Vec<IgnoredAny>,
    #[serde(default)]
    is_error: bool,
}

/// An entry of a `selection` (advantage, eligibility) or `outcome` (trained)
/// record.
#[derive(Clone, Debug, Deserialize)]
struct EntryLine {
    #[serde(default)]
    group: Option<usize>,
    member: usize,
    #[serde(default)]
    advantage: Option<f32>,
    #[serde(default)]
    eligible: Option<bool>,
    #[serde(default)]
    skip_reason: Option<String>,
    #[serde(default)]
    trained: Option<bool>,
}

/// Just enough of a line to read its version.
#[derive(Deserialize)]
struct Versioned {
    v: u32,
}

/// The full record of a rollout, read on demand.
#[derive(Deserialize)]
struct RolloutLine {
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    messages: Option<Vec<Message>>,
    #[serde(default)]
    prefix: bool,
    #[serde(default)]
    step_rewards: Vec<StepReward>,
    #[serde(default)]
    terminal_reward_raw: Option<f32>,
    #[serde(default)]
    judge_explanation: Option<String>,
    #[serde(default)]
    metadata: Option<Map<String, Value>>,
}

type MemberKey = (Option<usize>, usize);

#[derive(Debug, Default)]
struct UpdateIndex {
    rollouts: BTreeMap<MemberKey, (MemberEntry, LineRef)>,
    selection: HashMap<MemberKey, EntryLine>,
    outcome: HashMap<MemberKey, bool>,
    status: Option<String>,
    time: Option<String>,
    metrics: BTreeMap<String, f32>,
}

/// A segment still in view, and the last update of it that is.
#[derive(Clone, Copy, Debug)]
struct Active {
    segment: u32,
    upto: u32,
}

/// An index over one `observe.jsonl`, refreshed as the file grows.
#[derive(Debug)]
pub struct ObserveIndex {
    path: PathBuf,
    /// End of the last batch taken in; everything before it is indexed.
    committed: u64,
    runs: Vec<RunHeader>,
    active: Vec<Active>,
    updates: BTreeMap<(u32, u32), UpdateIndex>,
    prompts: HashMap<(u32, String), LineRef>,
    /// Batches dropped because they were broken off, and lines that were not
    /// records at all.
    skipped: u64,
}

impl ObserveIndex {
    /// An index over `<directory>/observe.jsonl`, read up to its current end.
    /// A directory without the file is an empty export, not an error: the run
    /// may not have written its first batch yet.
    pub fn open(directory: &Path) -> Result<Self, ReadError> {
        let mut index = Self {
            path: directory.join(LOG_FILE),
            committed: 0,
            runs: Vec::new(),
            active: Vec::new(),
            updates: BTreeMap::new(),
            prompts: HashMap::new(),
            skipped: 0,
        };
        index.refresh()?;
        Ok(index)
    }

    /// Reads what was appended since the last call. A file shorter than what
    /// was already indexed was rewritten - a resume drops the unfinished tail
    /// of the previous run - and is indexed again from the start.
    pub fn refresh(&mut self) -> Result<(), ReadError> {
        let length = match std::fs::metadata(&self.path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(source) => return Err(self.io(source)),
        };
        if length < self.committed {
            *self = Self {
                path: std::mem::take(&mut self.path),
                committed: 0,
                runs: Vec::new(),
                active: Vec::new(),
                updates: BTreeMap::new(),
                prompts: HashMap::new(),
                skipped: 0,
            };
        }
        if length == self.committed {
            return Ok(());
        }
        let file = File::open(&self.path).map_err(|source| self.io(source))?;
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(self.committed))
            .map_err(|source| self.io(source))?;
        let mut position = self.committed;
        let mut open: Vec<(IndexLine, LineRef)> = Vec::new();
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            let read = reader
                .read_until(b'\n', &mut buffer)
                .map_err(|source| self.io(source))?;
            if read == 0 || buffer.last() != Some(&b'\n') {
                // End of file, or a line still being written.
                break;
            }
            let line = LineRef {
                offset: position,
                len: read - 1,
            };
            position += read as u64;
            let text = &buffer[..read - 1];
            if text.iter().all(u8::is_ascii_whitespace) {
                if open.is_empty() {
                    self.committed = position;
                }
                continue;
            }
            let record = match serde_json::from_slice::<IndexLine>(text) {
                Ok(record) => record,
                Err(_) => {
                    if let Ok(Versioned { v }) = serde_json::from_slice(text)
                        && v != SCHEMA_VERSION
                    {
                        return Err(ReadError::Version {
                            path: self.path.clone(),
                            offset: line.offset,
                            version: v,
                        });
                    }
                    self.skipped += 1;
                    if open.is_empty() {
                        self.committed = position;
                    }
                    continue;
                }
            };
            if record.v != SCHEMA_VERSION {
                return Err(ReadError::Version {
                    path: self.path.clone(),
                    offset: line.offset,
                    version: record.v,
                });
            }
            let continues = open.first().is_some_and(|(first, _)| {
                first.segment == record.segment
                    && first.batch_id == record.batch_id
                    && first.batch_len == record.batch_len
                    && record.batch_index == open.len()
            });
            if !continues {
                if !open.is_empty() {
                    // Broken off by a newer batch: never finished, never read.
                    self.skipped += 1;
                    open.clear();
                }
                if record.batch_index != 0 || record.batch_len == 0 {
                    self.skipped += 1;
                    self.committed = position;
                    continue;
                }
            }
            let complete = record.batch_len == open.len() + 1;
            open.push((record, line));
            if complete {
                for (record, line) in open.drain(..) {
                    self.ingest(record, line);
                }
                self.committed = position;
            }
        }
        Ok(())
    }

    fn io(&self, source: io::Error) -> ReadError {
        ReadError::Io {
            path: self.path.clone(),
            source,
        }
    }

    fn ingest(&mut self, record: IndexLine, line: LineRef) {
        let segment = record.segment;
        match record.kind.as_str() {
            "run" => {
                let resumed_from_update = record.resumed_from_update;
                match resumed_from_update {
                    None => self.active.clear(),
                    Some(resumed) => {
                        for active in &mut self.active {
                            active.upto = active.upto.min(resumed);
                        }
                    }
                }
                self.active.push(Active {
                    segment,
                    upto: u32::MAX,
                });
                self.runs.push(RunHeader {
                    segment,
                    algorithm: record.algorithm.unwrap_or_default(),
                    model: record.model.unwrap_or_default(),
                    resumed_from_update,
                    every: record.every,
                    params: record.params.unwrap_or_default(),
                    time: record.time,
                });
            }
            "prompt" => {
                if let Some(key) = record.key {
                    self.prompts.insert((segment, key), line);
                }
            }
            "rollout" => {
                let (Some(update), Some(member)) = (record.update, record.member) else {
                    self.skipped += 1;
                    return;
                };
                let messages = record.messages.as_deref().unwrap_or_default();
                let entry = MemberEntry {
                    group: record.group,
                    member,
                    prompt: record.prompt.unwrap_or_default(),
                    seed: record.seed.unwrap_or_default(),
                    tokens: record.tokens.unwrap_or_default(),
                    truncated: record.truncated.unwrap_or_default(),
                    reward: record.reward,
                    reward_raw: record.reward_raw,
                    judge_term: record.judge_term,
                    advantage: record.advantage,
                    advantage_min: record.advantage_min,
                    advantage_max: record.advantage_max,
                    eligible: record.eligible,
                    trained: record.trained,
                    skip_reason: record.skip_reason,
                    turns: messages
                        .iter()
                        .filter(|message| message.role == "assistant")
                        .count(),
                    tool_calls: messages
                        .iter()
                        .map(|message| message.tool_calls.len())
                        .sum(),
                    tool_errors: messages
                        .iter()
                        .filter(|message| message.role == "tool" && message.is_error)
                        .count(),
                };
                self.updates
                    .entry((segment, update))
                    .or_default()
                    .rollouts
                    .insert((record.group, member), (entry, line));
            }
            "selection" | "outcome" => {
                let Some(update) = record.update else {
                    self.skipped += 1;
                    return;
                };
                let index = self.updates.entry((segment, update)).or_default();
                for entry in record.entries.unwrap_or_default() {
                    let key = (entry.group, entry.member);
                    if record.kind == "selection" {
                        index.selection.insert(key, entry);
                    } else if let Some(trained) = entry.trained {
                        index.outcome.insert(key, trained);
                    }
                }
            }
            "update" => {
                let Some(update) = record.update else {
                    self.skipped += 1;
                    return;
                };
                let index = self.updates.entry((segment, update)).or_default();
                index.status = record.status;
                index.time = record.time;
                index.metrics = record
                    .metrics
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|(name, value)| {
                        value.filter(|value| value.is_finite()).map(|v| (name, v))
                    })
                    .collect();
            }
            // A record type this build does not know is a newer writer's
            // addition within the same version: nothing here depends on it.
            _ => {}
        }
    }

    /// The `run` record of the latest segment.
    pub fn run(&self) -> Option<&RunHeader> {
        self.runs.last()
    }

    /// Every `run` record, oldest first.
    pub fn runs(&self) -> &[RunHeader] {
        &self.runs
    }

    /// Lines and batches that could not be taken in.
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    fn visible(&self, segment: u32, update: u32) -> bool {
        self.active
            .iter()
            .any(|active| active.segment == segment && update <= active.upto)
    }

    /// Every update still in view, by update number.
    pub fn updates(&self) -> Vec<UpdateEntry> {
        let mut entries: Vec<UpdateEntry> = self
            .updates
            .iter()
            .filter(|((segment, update), _)| self.visible(*segment, *update))
            .map(|(&(segment, update), index)| summarize(segment, update, index))
            .collect();
        entries.sort_by_key(|entry| (entry.update, entry.segment));
        entries
    }

    /// The segment whose `update` is in view; the latest one, should two be.
    fn locate(&self, update: u32) -> Option<(u32, &UpdateIndex)> {
        self.updates
            .iter()
            .filter(|((segment, number), _)| *number == update && self.visible(*segment, update))
            .map(|(&(segment, _), index)| (segment, index))
            .next_back()
    }

    /// One update in view and its groups, members without their texts.
    pub fn update(&self, update: u32) -> Option<(UpdateEntry, Vec<GroupEntry>)> {
        let (segment, index) = self.locate(update)?;
        let mut groups: BTreeMap<Option<usize>, GroupEntry> = BTreeMap::new();
        for (key, (entry, _)) in &index.rollouts {
            let group = groups.entry(key.0).or_insert_with(|| GroupEntry {
                group: key.0,
                prompt: entry.prompt.clone(),
                members: Vec::new(),
            });
            group.members.push(effective(index, entry));
        }
        Some((
            summarize(segment, update, index),
            groups.into_values().collect(),
        ))
    }

    /// The segment an update in view was written by.
    pub fn segment_of(&self, update: u32) -> Option<u32> {
        self.locate(update).map(|(segment, _)| segment)
    }

    /// A prompt, read from the file.
    pub fn prompt(&self, segment: u32, key: &str) -> Result<Option<PromptRecord>, ReadError> {
        let Some(line) = self.prompts.get(&(segment, key.to_string())) else {
            return Ok(None);
        };
        let mut file = File::open(&self.path).map_err(|source| self.io(source))?;
        self.read(&mut file, *line).map(Some)
    }

    /// The rollouts of one group of an update in view, texts included. `member`
    /// narrows to one: a group of long trajectories can weigh megabytes.
    pub fn rollouts(
        &self,
        update: u32,
        group: Option<usize>,
        member: Option<usize>,
    ) -> Result<Vec<RolloutRecord>, ReadError> {
        let Some((_, index)) = self.locate(update) else {
            return Ok(Vec::new());
        };
        let mut file = File::open(&self.path).map_err(|source| self.io(source))?;
        let mut records = Vec::new();
        for (key, (entry, line)) in &index.rollouts {
            if key.0 != group || member.is_some_and(|member| member != key.1) {
                continue;
            }
            let full: RolloutLine = self.read(&mut file, *line)?;
            let text = match (full.completion, full.messages) {
                (Some(completion), _) => RolloutText::Completion(completion),
                (None, messages) => RolloutText::Conversation {
                    messages: messages.unwrap_or_default(),
                    prefix: full.prefix,
                    step_rewards: full.step_rewards,
                    terminal_reward_raw: full.terminal_reward_raw,
                    judge_explanation: full.judge_explanation,
                    metadata: full.metadata.unwrap_or_default(),
                },
            };
            records.push(RolloutRecord {
                entry: effective(index, entry),
                text,
            });
        }
        Ok(records)
    }

    fn read<T: serde::de::DeserializeOwned>(
        &self,
        file: &mut File,
        line: LineRef,
    ) -> Result<T, ReadError> {
        file.seek(SeekFrom::Start(line.offset))
            .map_err(|source| self.io(source))?;
        let mut bytes = vec![0; line.len];
        file.read_exact(&mut bytes)
            .map_err(|source| self.io(source))?;
        serde_json::from_slice(&bytes).map_err(|_| ReadError::Changed {
            path: self.path.clone(),
            offset: line.offset,
        })
    }
}

/// A rollout as the optimizer finally saw it.
fn effective(index: &UpdateIndex, entry: &MemberEntry) -> MemberEntry {
    let key = (entry.group, entry.member);
    let selection = index.selection.get(&key);
    let eligible = selection
        .and_then(|selection| selection.eligible)
        .or(entry.eligible);
    let trained = match index.outcome.get(&key) {
        Some(trained) => Some(*trained),
        None if entry.trained == Some(false) || eligible == Some(false) => Some(false),
        None => None,
    };
    MemberEntry {
        eligible,
        trained,
        advantage: entry
            .advantage
            .or_else(|| selection.and_then(|selection| selection.advantage)),
        skip_reason: entry
            .skip_reason
            .clone()
            .or_else(|| selection.and_then(|selection| selection.skip_reason.clone())),
        ..entry.clone()
    }
}

fn summarize(segment: u32, update: u32, index: &UpdateIndex) -> UpdateEntry {
    let members: Vec<MemberEntry> = index
        .rollouts
        .values()
        .map(|(entry, _)| effective(index, entry))
        .collect();
    let rewards: Vec<f32> = members
        .iter()
        .filter_map(|member| member.reward)
        .filter(|reward| reward.is_finite())
        .collect();
    let groups = {
        let mut groups: Vec<Option<usize>> = members.iter().map(|member| member.group).collect();
        groups.dedup();
        groups.len()
    };
    UpdateEntry {
        segment,
        update,
        status: index.status.clone(),
        time: index.time.clone(),
        metrics: index.metrics.clone(),
        groups,
        rollouts: members.len(),
        trained: members
            .iter()
            .filter(|member| member.trained == Some(true))
            .count(),
        // A mean of at most a few thousand rewards: the count is exact in f32.
        reward_mean: (!rewards.is_empty())
            .then(|| rewards.iter().sum::<f32>() / rewards.len() as f32),
        reward_min: rewards.iter().copied().reduce(f32::min),
        reward_max: rewards.iter().copied().reduce(f32::max),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::record;

    struct Log {
        dir: PathBuf,
        lines: Vec<String>,
        next_batch: u64,
    }

    impl Log {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "retrograd-observe-reader-{label}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("temp dir");
            Self {
                dir,
                lines: Vec::new(),
                next_batch: 0,
            }
        }

        fn batch(&mut self, segment: u32, records: Vec<(&str, Value)>) -> &mut Self {
            let len = records.len();
            let id = self.next_batch;
            self.next_batch += 1;
            for (index, (kind, body)) in records.into_iter().enumerate() {
                self.lines.push(record::line(
                    kind,
                    body,
                    segment,
                    "2026-01-01T00:00:00.000Z",
                    (id, index, len),
                    0,
                ));
            }
            self
        }

        fn write(&self) {
            let mut text = self.lines.join("\n");
            text.push('\n');
            std::fs::write(self.dir.join(LOG_FILE), text).expect("write");
        }
    }

    impl Drop for Log {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn run(resumed: Option<u32>) -> (&'static str, Value) {
        (
            "run",
            json!({"algorithm": "grpo", "model": "/m/model.gguf",
                   "resumed_from_update": resumed, "every": 1, "params": {"group_size": 2}}),
        )
    }

    fn rollout(update: u32, group: usize, member: usize, reward: f32) -> (&'static str, Value) {
        (
            "rollout",
            json!({"update": update, "group": group, "member": member, "prompt": "p:0",
                   "seed": 7, "tokens": 12, "truncated": false, "reward": reward,
                   "reward_raw": reward, "judge_term": null, "advantage": null,
                   "eligible": null, "trained": null, "skip_reason": null,
                   "completion": format!("answer {member}")}),
        )
    }

    fn update(number: u32) -> (&'static str, Value) {
        (
            "update",
            json!({"update": number, "status": "completed",
                   "metrics": {"reward/mean": 0.5, "policy/kl": null}}),
        )
    }

    #[test]
    fn a_run_is_read_back_with_its_groups_and_texts() {
        let mut log = Log::new("basic");
        log.batch(0, vec![run(None)])
            .batch(
                0,
                vec![
                    (
                        "prompt",
                        json!({"key": "p:0", "messages": [{"role": "user", "content": "q"}]}),
                    ),
                    rollout(1, 0, 0, 1.0),
                    rollout(1, 0, 1, 0.0),
                ],
            )
            .batch(
                0,
                vec![(
                    "outcome",
                    json!({"update": 1, "entries": [{"group": 0, "member": 0, "trained": true}]}),
                )],
            )
            .batch(0, vec![update(1)]);
        log.write();

        let index = ObserveIndex::open(&log.dir).expect("open");
        let run = index.run().expect("run");
        assert_eq!(run.algorithm, "grpo");
        assert_eq!(run.every, Some(1));
        let updates = index.updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].rollouts, 2);
        assert_eq!(updates[0].groups, 1);
        assert_eq!(updates[0].trained, 1);
        assert_eq!(updates[0].reward_mean, Some(0.5));
        assert_eq!(updates[0].metrics.get("reward/mean"), Some(&0.5));
        assert!(!updates[0].metrics.contains_key("policy/kl"));

        let (_, groups) = index.update(1).expect("update");
        assert_eq!(groups[0].members[0].trained, Some(true));
        assert_eq!(groups[0].members[1].trained, None);
        let prompt = index.prompt(0, "p:0").expect("read").expect("prompt");
        assert_eq!(prompt.messages[0].content, "q");
        let rollouts = index.rollouts(1, Some(0), Some(1)).expect("read");
        assert_eq!(rollouts.len(), 1);
        assert_eq!(rollouts[0].text, RolloutText::Completion("answer 1".into()));
    }

    #[test]
    fn a_resume_supersedes_the_updates_it_replays() {
        let mut log = Log::new("resume");
        log.batch(0, vec![run(None)])
            .batch(0, vec![update(1)])
            .batch(0, vec![update(2)])
            .batch(0, vec![update(3)])
            .batch(1, vec![run(Some(1))])
            .batch(1, vec![update(2)]);
        log.write();
        let index = ObserveIndex::open(&log.dir).expect("open");
        let seen: Vec<(u32, u32)> = index
            .updates()
            .iter()
            .map(|entry| (entry.update, entry.segment))
            .collect();
        assert_eq!(seen, vec![(1, 0), (2, 1)]);
        assert_eq!(index.segment_of(2), Some(1));
    }

    #[test]
    fn a_run_without_a_checkpoint_starts_over() {
        let mut log = Log::new("fresh");
        log.batch(0, vec![run(None)])
            .batch(0, vec![update(1)])
            .batch(1, vec![run(None)]);
        log.write();
        let index = ObserveIndex::open(&log.dir).expect("open");
        assert!(index.updates().is_empty());
    }

    #[test]
    fn an_unfinished_batch_is_read_once_it_is_finished() {
        let mut log = Log::new("incremental");
        log.batch(0, vec![run(None)])
            .batch(0, vec![rollout(1, 0, 0, 1.0), rollout(1, 0, 1, 0.0)]);
        let complete = log.lines.clone();
        // The second rollout of the batch is not written yet.
        log.lines.truncate(complete.len() - 1);
        log.write();
        let mut index = ObserveIndex::open(&log.dir).expect("open");
        assert!(index.updates().is_empty());

        log.lines = complete;
        log.write();
        index.refresh().expect("refresh");
        assert_eq!(index.updates()[0].rollouts, 2);
    }

    #[test]
    fn a_batch_broken_off_by_a_newer_one_is_dropped() {
        let mut log = Log::new("broken");
        log.batch(0, vec![run(None)])
            .batch(0, vec![rollout(1, 0, 0, 1.0), rollout(1, 0, 1, 0.0)]);
        log.lines.pop();
        log.batch(0, vec![update(1)]);
        log.write();
        let index = ObserveIndex::open(&log.dir).expect("open");
        let updates = index.updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].rollouts, 0);
        assert_eq!(index.skipped(), 1);
    }

    #[test]
    fn a_file_that_shrank_is_indexed_again() {
        let mut log = Log::new("shrunk");
        log.batch(0, vec![run(None)]).batch(0, vec![update(1)]);
        log.write();
        let mut index = ObserveIndex::open(&log.dir).expect("open");
        assert_eq!(index.updates().len(), 1);

        log.lines.truncate(1);
        log.write();
        index.refresh().expect("refresh");
        assert!(index.updates().is_empty());

        log.batch(0, vec![update(1)]).batch(0, vec![update(2)]);
        log.write();
        index.refresh().expect("refresh");
        assert_eq!(index.updates().len(), 2);
    }

    #[test]
    fn another_schema_version_is_an_error() {
        let mut log = Log::new("version");
        log.batch(0, vec![run(None)]);
        log.lines[0] = log.lines[0].replacen("\"v\":1", "\"v\":9", 1);
        log.write();
        assert!(matches!(
            ObserveIndex::open(&log.dir),
            Err(ReadError::Version { version: 9, .. })
        ));
    }

    #[test]
    fn a_directory_without_a_log_is_an_empty_export() {
        let log = Log::new("empty");
        let index = ObserveIndex::open(&log.dir).expect("open");
        assert!(index.run().is_none());
        assert!(index.updates().is_empty());
    }
}
