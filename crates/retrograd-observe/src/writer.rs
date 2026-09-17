//! The writer thread: recovery, `observe.jsonl`, and the feed the viewer
//! polls. All serialization and I/O of the export happens here.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use serde_json::{Value, json};

use crate::SCHEMA_VERSION;
use crate::batch::{ObserveBatch, RunInfo};
use crate::record::{self, Header};
use crate::sink::Shared;

pub(crate) const LOG_FILE: &str = "observe.jsonl";
pub(crate) const FEED_DIR: &str = "feed";
pub(crate) const MANIFEST_FILE: &str = "manifest.js";

const INDEX_HTML: &str = include_str!("../assets/index.html");
const VIEWER_CSS: &str = include_str!("../assets/viewer.css");
const VIEWER_JS: &str = include_str!("../assets/viewer.js");

pub(crate) fn run(
    directory: &Path,
    max_text_chars: usize,
    shared: &Shared,
    run: RunInfo,
    receiver: Receiver<ObserveBatch>,
) {
    let mut active = match Active::start(directory, max_text_chars, shared, &run) {
        Ok(active) => active,
        Err(error) => {
            shared.fail(format!(
                "observe: cannot export to {}: {error}; this run exports nothing",
                directory.display()
            ));
            None
        }
    };
    for batch in receiver {
        #[cfg(test)]
        let _gate = shared.gate.lock();
        let Some(writer) = active.as_mut() else {
            continue;
        };
        if let Err(error) = writer.write(batch, shared) {
            shared.fail(format!(
                "observe: writing to {} failed: {error}; the export stops here, training goes on",
                directory.display()
            ));
            active = None;
        }
    }
    if let Some(mut writer) = active {
        let _ = writer.log.flush();
    }
}

struct Active {
    directory: PathBuf,
    max_text_chars: usize,
    log: BufWriter<File>,
    generation: String,
    chunks: u64,
    segment: u32,
    next_batch_id: u64,
    /// Prompt keys already written in this segment.
    prompts: HashSet<String>,
}

impl Active {
    /// Recovers the log and rebuilds the feed. `None` when the log cannot be
    /// trusted and is left untouched.
    fn start(
        directory: &Path,
        max_text_chars: usize,
        shared: &Shared,
        run: &RunInfo,
    ) -> io::Result<Option<Self>> {
        write_atomic(
            &directory.join("index.html"),
            INDEX_HTML
                .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
                .as_bytes(),
        )?;
        write_atomic(&directory.join("viewer.css"), VIEWER_CSS.as_bytes())?;
        write_atomic(&directory.join("viewer.js"), VIEWER_JS.as_bytes())?;

        let path = directory.join(LOG_FILE);
        let recovered = match fs::read(&path) {
            Ok(bytes) => match recover(&bytes) {
                Ok(recovered) => recovered,
                Err(corruption) => {
                    shared.fail(format!(
                        "observe: {} cannot be continued ({corruption}); it is left untouched \
                         and this run exports nothing",
                        path.display()
                    ));
                    return Ok(None);
                }
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => Recovered::default(),
            Err(error) => return Err(error),
        };
        let log = OpenOptions::new().create(true).append(true).open(&path)?;
        if recovered.valid_len < recovered.file_len {
            shared.warn(format!(
                "observe: dropped the incomplete tail of {} ({} bytes)",
                path.display(),
                recovered.file_len - recovered.valid_len
            ));
            log.set_len(recovered.valid_len)?;
        }

        let feed = directory.join(FEED_DIR);
        fs::create_dir_all(&feed)?;
        let generation = next_generation(&feed)?;
        fs::create_dir(feed.join(&generation))?;
        let mut active = Self {
            directory: directory.to_path_buf(),
            max_text_chars,
            log: BufWriter::new(log),
            generation,
            chunks: 0,
            segment: recovered.max_segment.map_or(0, |segment| segment + 1),
            next_batch_id: recovered.max_batch_id.map_or(0, |id| id + 1),
            prompts: HashSet::new(),
        };
        for lines in &recovered.batches {
            active.publish_chunk(lines)?;
        }
        active.publish_manifest(shared)?;
        active.append(vec![("run", body(run)?)], shared)?;
        Ok(Some(active))
    }

    fn write(&mut self, batch: ObserveBatch, shared: &Shared) -> io::Result<()> {
        let mut records = Vec::new();
        let mut new_prompts = Vec::new();
        match batch {
            ObserveBatch::Rollouts(batch) => {
                for prompt in &batch.prompts {
                    if !self.prompts.contains(&prompt.key) && !new_prompts.contains(&prompt.key) {
                        new_prompts.push(prompt.key.clone());
                        records.push(("prompt", body(prompt)?));
                    }
                }
                for rollout in &batch.rollouts {
                    records.push(("rollout", body(rollout)?));
                }
            }
            ObserveBatch::Selection { update, entries } => {
                records.push(("selection", json!({"update": update, "entries": entries})));
            }
            ObserveBatch::Outcome { update, entries } => {
                records.push(("outcome", json!({"update": update, "entries": entries})));
            }
            ObserveBatch::Update(summary) => records.push(("update", body(&summary)?)),
        }
        if records.is_empty() {
            return Ok(());
        }
        self.append(records, shared)?;
        self.prompts.extend(new_prompts);
        Ok(())
    }

    /// Writes one batch: the log first, then its chunk, then the manifest that
    /// names the chunk.
    fn append(&mut self, records: Vec<(&str, Value)>, shared: &Shared) -> io::Result<()> {
        let time = record::utc_now();
        let batch_id = self.next_batch_id;
        let batch_len = records.len();
        let lines = records
            .into_iter()
            .enumerate()
            .map(|(index, (kind, body))| {
                record::line(
                    kind,
                    body,
                    self.segment,
                    &time,
                    (batch_id, index, batch_len),
                    self.max_text_chars,
                )
            })
            .collect::<Vec<_>>();
        for line in &lines {
            self.log.write_all(line.as_bytes())?;
            self.log.write_all(b"\n")?;
        }
        self.log.flush()?;
        self.next_batch_id += 1;
        self.publish_chunk(&lines)?;
        self.publish_manifest(shared)
    }

    fn publish_chunk(&mut self, lines: &[String]) -> io::Result<()> {
        let path = self
            .directory
            .join(FEED_DIR)
            .join(&self.generation)
            .join(format!("{:06}.js", self.chunks));
        let script = format!(
            "RG_FEED.chunk({}, [\n{}\n]);\n",
            self.chunks,
            lines.join(",\n")
        );
        write_atomic(&path, script.as_bytes())?;
        self.chunks += 1;
        Ok(())
    }

    fn publish_manifest(&self, shared: &Shared) -> io::Result<()> {
        let manifest = json!({
            "generation": self.generation,
            "chunks": self.chunks,
            "segment": self.segment,
            "dropped_batches": shared.dropped(),
            "updated_at": record::utc_now(),
        });
        write_atomic(
            &self.directory.join(FEED_DIR).join(MANIFEST_FILE),
            format!("RG_FEED.manifest({manifest});\n").as_bytes(),
        )
    }
}

fn body(value: &impl serde::Serialize) -> io::Result<Value> {
    serde_json::to_value(value).map_err(io::Error::other)
}

/// Replaces `path` in one step, so a reader sees the old file or the new one.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)
}

/// One past the highest numbered generation under `feed`.
fn next_generation(feed: &Path) -> io::Result<String> {
    let mut highest = 0_u32;
    for entry in fs::read_dir(feed)? {
        let entry = entry?;
        if let Some(number) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        {
            highest = highest.max(number);
        }
    }
    Ok(format!("{:06}", highest + 1))
}

#[derive(Debug, Default)]
struct Recovered {
    /// The complete batches, each as its lines.
    batches: Vec<Vec<String>>,
    max_segment: Option<u32>,
    max_batch_id: Option<u64>,
    /// Bytes up to the end of the last complete batch.
    valid_len: u64,
    file_len: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
enum Corruption {
    #[error("line {0} is not a record")]
    Unreadable(usize),
    #[error(
        "line {line} has schema version {version}, this build writes {}",
        SCHEMA_VERSION
    )]
    Version { line: usize, version: u32 },
    #[error("line {0} breaks the batch sequence")]
    Sequence(usize),
}

/// Splits the log into complete batches. Only an incomplete tail - a partial
/// line, or the lines of a batch that was not finished - may be dropped;
/// anything wrong before it refuses the whole file.
fn recover(bytes: &[u8]) -> Result<Recovered, Corruption> {
    let mut recovered = Recovered {
        file_len: bytes.len() as u64,
        ..Recovered::default()
    };
    let mut open: Option<(Header, Vec<String>)> = None;
    let mut offset = 0;
    let mut line_number = 0;
    while let Some(end) = bytes[offset..].iter().position(|&byte| byte == b'\n') {
        line_number += 1;
        let text = std::str::from_utf8(&bytes[offset..offset + end])
            .map_err(|_| Corruption::Unreadable(line_number))?;
        offset += end + 1;
        let header: Header =
            serde_json::from_str(text).map_err(|_| Corruption::Unreadable(line_number))?;
        if header.v != SCHEMA_VERSION {
            return Err(Corruption::Version {
                line: line_number,
                version: header.v,
            });
        }
        let lines = match &mut open {
            None if header.batch_index == 0
                && header.batch_len > 0
                && recovered.max_batch_id.is_none_or(|id| header.batch_id > id) =>
            {
                &mut open.insert((header, Vec::new())).1
            }
            Some((first, lines))
                if header.batch_id == first.batch_id
                    && header.batch_len == first.batch_len
                    && header.segment == first.segment
                    && header.batch_index == lines.len() =>
            {
                lines
            }
            _ => return Err(Corruption::Sequence(line_number)),
        };
        lines.push(text.to_string());
        if lines.len() == header.batch_len {
            let (first, lines) = open.take().expect("the batch was just filled");
            recovered.batches.push(lines);
            recovered.max_segment = recovered.max_segment.max(Some(first.segment));
            recovered.max_batch_id = recovered.max_batch_id.max(Some(first.batch_id));
            recovered.valid_len = offset as u64;
        }
    }
    Ok(recovered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(segment: u32, batch: (u64, usize, usize)) -> String {
        record::line("update", json!({"update": 1}), segment, "t", batch, 0)
    }

    fn log(lines: &[String]) -> Vec<u8> {
        lines
            .iter()
            .flat_map(|line| format!("{line}\n").into_bytes())
            .collect()
    }

    #[test]
    fn complete_batches_are_kept_in_order() {
        let bytes = log(&[line(0, (0, 0, 1)), line(0, (1, 0, 2)), line(0, (1, 1, 2))]);
        let recovered = recover(&bytes).unwrap();
        assert_eq!(recovered.batches.len(), 2);
        assert_eq!(recovered.batches[1].len(), 2);
        assert_eq!(recovered.valid_len, bytes.len() as u64);
        assert_eq!(recovered.max_batch_id, Some(1));
        assert_eq!(recovered.max_segment, Some(0));
    }

    #[test]
    fn a_partial_line_and_an_unfinished_batch_are_the_tail() {
        let complete = log(&[line(0, (0, 0, 1))]);
        let mut bytes = complete.clone();
        bytes.extend(log(&[line(0, (1, 0, 2))]));
        bytes.extend_from_slice(b"{\"v\":1,\"trunc");
        let recovered = recover(&bytes).unwrap();
        assert_eq!(recovered.batches.len(), 1);
        assert_eq!(recovered.valid_len, complete.len() as u64);
        assert_eq!(recovered.max_batch_id, Some(0));
    }

    #[test]
    fn repeated_or_reversed_batch_ids_refuse_the_file() {
        for next in [0, 1] {
            let bytes = log(&[line(0, (1, 0, 1)), line(0, (next, 0, 1))]);
            assert_eq!(recover(&bytes).unwrap_err(), Corruption::Sequence(2));
        }
    }

    #[test]
    fn damage_before_the_tail_refuses_the_file() {
        let mut bytes = log(&[line(0, (0, 0, 1))]);
        bytes.extend_from_slice(b"garbage\n");
        bytes.extend(log(&[line(0, (1, 0, 1))]));
        assert_eq!(recover(&bytes).unwrap_err(), Corruption::Unreadable(2));

        let bytes = log(&[line(0, (0, 0, 2)), line(0, (1, 0, 1))]);
        assert_eq!(recover(&bytes).unwrap_err(), Corruption::Sequence(2));

        let newer = line(0, (0, 0, 1)).replacen("\"v\":1", "\"v\":2", 1);
        assert_eq!(
            recover(&log(&[newer])).unwrap_err(),
            Corruption::Version {
                line: 1,
                version: 2
            }
        );
    }
}
