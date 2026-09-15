//! The reward process, and the JSONL protocol spoken to it.
//!
//! One request per line in, one response per line out, in order, one for one.
//! Persistent requests and responses carry reserved batch/index fields, and an
//! echoed marker closes each batch; a late line therefore cannot answer the
//! following call.
//!
//! Two lifetimes for the process that speaks it, chosen by
//! [`RewardMode`](retrograd_core::RewardMode):
//! [`Persistent`](retrograd_core::RewardMode::Persistent) keeps a single worker
//! alive for the whole loop after a version handshake,
//! [`OneShot`](retrograd_core::RewardMode::OneShot) spawns one per batch and
//! closes its stdin.
//!
//! It lives here because this crate already owns a reward command
//! ([`crate::CommandReward`]) and the training loop's `reward_command` is the
//! same transport with a richer response schema: two callers, one protocol, and
//! nothing above `retrograd-core` in either direction. What is *not* here is
//! what a response means - a reward's finiteness, a `judge_weight`, a
//! [`crate::Score`] - which stays with the caller that reads it. This module
//! decides only that a line arrived, in time, and parsed.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

use retrograd_core::{RewardMode, RewardProtocol};

/// The token both ends write to say they speak this protocol. Versioned: a
/// second revision changes the string, and a worker written against one and a
/// trainer against the other refuse each other at startup instead of
/// disagreeing about a field halfway through a run.
pub const REWARD_PROTOCOL_VERSION: &str = "retrograd-reward/1";

/// Cap on what one one-shot batch may print. A reward command that streams
/// megabytes is not answering the protocol, and the parent must not grow with
/// it.
const MAX_COMMAND_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// Reserved fields carried by persistent requests and echoed by their
/// responses. They make a late line attributable to the batch that produced
/// it instead of letting it become the first response of the next batch.
const BATCH_FIELD: &str = "_retrograd_batch";
const INDEX_FIELD: &str = "_retrograd_index";
const BATCH_END_FIELD: &str = "_retrograd_batch_end";

/// How much of a persistent worker's stderr is kept for the message of the call
/// that finds it dead. Its stderr has no end to wait for, so it is drained
/// continuously - the pipe must never fill - and only the tail is retained.
const STDERR_TAIL_BYTES: usize = 8 * 1024;

/// How long a persistent worker is given to exit on its own after its stdin is
/// closed, before its process group is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(500);

#[derive(Debug, Error)]
pub enum RewardProcessError {
    #[error("reward command must contain an executable")]
    Empty,
    #[error("reward command timeout must be greater than zero")]
    ZeroTimeout,
    #[error("start reward command '{program}': {source}")]
    Spawn {
        program: String,
        #[source]
        source: std::io::Error,
    },
    #[error("reward command stdio is unavailable")]
    Stdio,
    #[error("serialize reward request: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("write to reward command: {0}")]
    Write(#[source] std::io::Error),
    #[error("read reward command stdout: {0}")]
    Read(#[source] std::io::Error),
    #[error("reward command output is not UTF-8: {0}")]
    NotUtf8(#[source] std::string::FromUtf8Error),
    #[error("reward command timed out after {} ms while {phase}", timeout.as_millis())]
    Timeout {
        timeout: Duration,
        phase: &'static str,
    },
    #[error("reward command {phase} worker panicked")]
    DrainPanicked { phase: &'static str },
    #[error("reward command exited with {status}: {stderr}")]
    Exited { status: String, stderr: String },
    #[error("reward command closed its output after {received} of {expected} responses: {stderr}")]
    Closed {
        received: usize,
        expected: usize,
        stderr: String,
    },
    /// The persistent handshake, in every way it can fail. One variant because
    /// they all have the same fix, and the fix is the message: a command that
    /// cannot answer is one line of TOML away from not being asked to.
    #[error(
        "reward command did not complete the {REWARD_PROTOCOL_VERSION} handshake ({detail}).\n\
         A persistent reward command must answer the line \
         {{\"protocol\":\"{REWARD_PROTOCOL_VERSION}\"}} with the same line, flushed, then one \
         flushed, correlated JSON line per request and echo the batch-end marker. Set \
         reward_mode = \"oneshot\" for a command that reads \
         its stdin to the end and exits."
    )]
    Handshake { detail: String },
    #[error("invalid reward response {index}: {source}")]
    Decode {
        index: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("reward command returned {received} responses for {expected} rollouts")]
    Count { received: usize, expected: usize },
    #[error("reward command returned more responses than the {expected} rollouts it was sent")]
    Extra { expected: usize },
    #[error("persistent reward requests must serialize to JSON objects")]
    RequestShape,
    #[error(
        "reward response {index} belongs to batch {received_batch:?}, item {received_index:?}; expected batch {expected_batch}, item {expected_index}"
    )]
    Correlation {
        index: usize,
        expected_batch: u64,
        expected_index: usize,
        received_batch: Option<u64>,
        received_index: Option<u64>,
    },
    #[error("reward command did not close batch {batch} after {expected} responses")]
    BatchEnd { batch: u64, expected: usize },
    #[error("persistent reward batch sequence is exhausted")]
    SequenceExhausted,
    #[error("reward command output exceeded {max_bytes} bytes")]
    OutputLimit { max_bytes: usize },
}

impl RewardProcessError {
    /// Whether the fault is in what the operator wired up - a command that is
    /// not there, a process answering something else than the protocol - rather
    /// than in the machine running it. It is what decides `422` from `500` at
    /// the API boundary, so it belongs to the error and not to a call site.
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::Empty
                | Self::ZeroTimeout
                | Self::Handshake { .. }
                | Self::Decode { .. }
                | Self::Count { .. }
                | Self::Extra { .. }
                | Self::RequestShape
                | Self::Correlation { .. }
                | Self::BatchEnd { .. }
                | Self::OutputLimit { .. }
                | Self::NotUtf8(_)
        )
    }
}

/// Back to the agent facade, which spells every reward failure the same way.
impl From<RewardProcessError> for retrograd_agent_core::Error {
    fn from(error: RewardProcessError) -> Self {
        Self::Reward(error.to_string())
    }
}

/// Back to the applicative facade, with the class preserved: a reward command
/// that answers nonsense is something a caller passed (`InvalidArgument`, and
/// `is_user_error()`), a reward command that died is something that ran
/// (`Runtime`). Collapsing the two is how a 422 becomes a 500.
impl From<RewardProcessError> for retrograd_core::Error {
    fn from(error: RewardProcessError) -> Self {
        if error.is_user_error() {
            retrograd_core::Error::invalid(error.to_string())
        } else {
            retrograd_core::Error::runtime(error.to_string())
        }
    }
}

type Result<T> = std::result::Result<T, RewardProcessError>;

/// A reward command, and - in [`RewardMode::Persistent`] - the process
/// currently speaking for it.
///
/// Owning one is what amortizes the command's startup: the worker is spawned on
/// the first call and reused by every later one. A failed call kills it, so the
/// next call starts a fresh process rather than resuming a stream nobody can
/// resynchronize; the failed call itself is *not* retried, because a reward
/// that answered wrongly once has no reason to answer rightly twice and a
/// silent retry would hide it.
pub struct RewardProcess {
    command: Vec<String>,
    protocol: RewardProtocol,
    session: Option<Session>,
    next_batch: u64,
}

impl std::fmt::Debug for RewardProcess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RewardProcess")
            .field("command", &self.command)
            .field("protocol", &self.protocol)
            .field("live", &self.session.is_some())
            .finish()
    }
}

impl RewardProcess {
    pub fn new(command: &[String], protocol: RewardProtocol) -> Result<Self> {
        if command
            .first()
            .is_none_or(|program| program.trim().is_empty())
        {
            return Err(RewardProcessError::Empty);
        }
        if protocol.timeout.is_zero() {
            return Err(RewardProcessError::ZeroTimeout);
        }
        Ok(Self {
            command: command.to_vec(),
            protocol,
            session: None,
            next_batch: 0,
        })
    }

    /// Sends one request per item and returns one response per request, in
    /// order. The count is part of the protocol: a command that answers a
    /// different number of lines has not answered.
    pub fn call<Q, R>(&mut self, requests: impl IntoIterator<Item = Q>) -> Result<Vec<R>>
    where
        Q: Serialize,
        R: DeserializeOwned,
    {
        match self.protocol.mode {
            RewardMode::OneShot => {
                let (payload, expected) = encode(requests)?;
                call_once(&self.command, payload, expected, self.protocol.timeout)
            }
            RewardMode::Persistent => {
                let batch = self.next_batch;
                self.next_batch = self
                    .next_batch
                    .checked_add(1)
                    .ok_or(RewardProcessError::SequenceExhausted)?;
                let (payload, expected) = encode_persistent(requests, batch)?;
                let outcome = self.call_persistent(payload, expected, batch);
                if outcome.is_err() {
                    // The stream cannot be resynchronized: whatever the worker
                    // was about to write belongs to a batch that no longer has
                    // a reader. Only a fresh process can answer the next call.
                    self.session = None;
                }
                outcome
            }
        }
    }

    fn call_persistent<R: DeserializeOwned>(
        &mut self,
        payload: Vec<u8>,
        expected: usize,
        batch: u64,
    ) -> Result<Vec<R>> {
        let timeout = self.protocol.timeout;
        let deadline = Instant::now() + timeout;
        let session = match &mut self.session {
            Some(session) => session,
            // The first call pays the spawn and the handshake, and pays them
            // inside its own deadline: a worker that loads a model at startup
            // has not timed out until the batch has.
            none => none.insert(Session::start(&self.command, deadline, timeout)?),
        };
        session.exchange(payload, expected, batch, deadline, timeout)
    }
}

/// One JSON line per request, and how many there were.
fn encode<Q: Serialize>(requests: impl IntoIterator<Item = Q>) -> Result<(Vec<u8>, usize)> {
    let mut payload = Vec::new();
    let mut count = 0_usize;
    for request in requests {
        serde_json::to_writer(&mut payload, &request).map_err(RewardProcessError::Serialize)?;
        payload.push(b'\n');
        count += 1;
    }
    Ok((payload, count))
}

/// Persistent requests carry their batch and position, and end with a marker
/// the worker must echo after every response. Correlation prevents a late line
/// from being accepted by the following call; the marker makes an extra line
/// fail the call that produced it without relying on a scheduler race.
fn encode_persistent<Q: Serialize>(
    requests: impl IntoIterator<Item = Q>,
    batch: u64,
) -> Result<(Vec<u8>, usize)> {
    let mut payload = Vec::new();
    let mut count = 0_usize;
    for request in requests {
        let mut value = serde_json::to_value(request).map_err(RewardProcessError::Serialize)?;
        let object = value
            .as_object_mut()
            .ok_or(RewardProcessError::RequestShape)?;
        object.insert(BATCH_FIELD.into(), serde_json::Value::from(batch));
        let index = u64::try_from(count).map_err(|_| RewardProcessError::SequenceExhausted)?;
        object.insert(INDEX_FIELD.into(), serde_json::Value::from(index));
        serde_json::to_writer(&mut payload, &value).map_err(RewardProcessError::Serialize)?;
        payload.push(b'\n');
        count += 1;
    }
    let mut end = serde_json::Map::new();
    end.insert(BATCH_END_FIELD.into(), serde_json::Value::from(batch));
    serde_json::to_writer(&mut payload, &end).map_err(RewardProcessError::Serialize)?;
    payload.push(b'\n');
    Ok((payload, count))
}

fn decode<R: DeserializeOwned>(index: usize, line: &str) -> Result<R> {
    serde_json::from_str(line).map_err(|source| RewardProcessError::Decode {
        index: index + 1,
        source,
    })
}

fn decode_persistent<R: DeserializeOwned>(index: usize, line: &str, batch: u64) -> Result<R> {
    let mut value: serde_json::Value =
        serde_json::from_str(line).map_err(|source| RewardProcessError::Decode {
            index: index + 1,
            source,
        })?;
    let object = value.as_object_mut();
    let received_batch = object
        .as_ref()
        .and_then(|object| object.get(BATCH_FIELD))
        .and_then(serde_json::Value::as_u64);
    let received_index = object
        .as_ref()
        .and_then(|object| object.get(INDEX_FIELD))
        .and_then(serde_json::Value::as_u64);
    if received_batch != Some(batch) || received_index != u64::try_from(index).ok() {
        return Err(RewardProcessError::Correlation {
            index: index + 1,
            expected_batch: batch,
            expected_index: index,
            received_batch,
            received_index,
        });
    }
    if let Some(object) = object {
        object.remove(BATCH_FIELD);
        object.remove(INDEX_FIELD);
    }
    serde_json::from_value(value).map_err(|source| RewardProcessError::Decode {
        index: index + 1,
        source,
    })
}

fn is_batch_end(line: &str, batch: u64) -> bool {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .is_some_and(|object| {
            object.len() == 1
                && object
                    .get(BATCH_END_FIELD)
                    .and_then(serde_json::Value::as_u64)
                    == Some(batch)
        })
}

fn spawn(command: &[String]) -> Result<Child> {
    let (program, args) = command.split_first().ok_or(RewardProcessError::Empty)?;
    let mut process = Command::new(program);
    process
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Its own group, so a command that leaves helpers behind can be closed
        // whole - and so the pipe readers cannot wait on a grandchild that
        // inherited them.
        process.process_group(0);
    }
    process.spawn().map_err(|source| RewardProcessError::Spawn {
        program: program.clone(),
        source,
    })
}

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    if let Ok(pid) = i32::try_from(pid) {
        // SAFETY: the child is its own process-group leader; negative pid
        // addresses the complete group. ESRCH only means it already exited.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

fn read_limited(mut reader: impl Read, max_bytes: usize) -> std::io::Result<(Vec<u8>, bool)> {
    let mut kept = Vec::with_capacity(max_bytes.min(64 * 1024));
    let mut chunk = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        let copy = read.min(max_bytes.saturating_sub(kept.len()));
        kept.extend_from_slice(&chunk[..copy]);
        truncated |= copy < read;
    }
    Ok((kept, truncated))
}

/// Drains persistent stdout without allowing either a single unterminated line
/// or queued complete lines to grow without bound. A capacity-one channel is
/// enough to keep the pipe moving while the exchange consumes responses, and
/// applies backpressure as soon as it stops.
fn read_bounded_lines(reader: impl Read, sender: SyncSender<Result<String>>, max_bytes: usize) {
    let mut reader = BufReader::new(reader);
    loop {
        let mut bytes = Vec::new();
        let read = match reader
            .by_ref()
            .take(
                u64::try_from(max_bytes)
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            )
            .read_until(b'\n', &mut bytes)
        {
            Ok(read) => read,
            Err(error) => {
                let _ = sender.send(Err(RewardProcessError::Read(error)));
                return;
            }
        };
        if read == 0 {
            return;
        }
        if bytes.len() > max_bytes {
            let _ = sender.send(Err(RewardProcessError::OutputLimit { max_bytes }));
            return;
        }
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        }
        let line = String::from_utf8(bytes).map_err(RewardProcessError::NotUtf8);
        if sender.send(line).is_err() {
            return;
        }
    }
}

fn receive_before<T>(
    receiver: &Receiver<T>,
    deadline: Instant,
    timeout: Duration,
    phase: &'static str,
) -> Result<T> {
    receiver
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|error| match error {
            RecvTimeoutError::Timeout => RewardProcessError::Timeout { timeout, phase },
            RecvTimeoutError::Disconnected => RewardProcessError::DrainPanicked { phase },
        })
}

/// One batch, one process: write the requests, close stdin, read what the
/// command printed before it exited.
///
/// stdout and stderr are drained by their own threads and stdin is written by a
/// third, because any two of the three can block on each other: writing
/// everything first deadlocks as soon as an output pipe fills, and waiting for
/// the exit first deadlocks as soon as the command blocks writing to a pipe
/// nobody reads.
fn call_once<R: DeserializeOwned>(
    command: &[String],
    payload: Vec<u8>,
    expected: usize,
    timeout: Duration,
) -> Result<Vec<R>> {
    let mut child = spawn(command)?;
    let pid = child.id();
    let mut stdin = child.stdin.take().ok_or(RewardProcessError::Stdio)?;
    let stdout = child.stdout.take().ok_or(RewardProcessError::Stdio)?;
    let stderr = child.stderr.take().ok_or(RewardProcessError::Stdio)?;

    let (writer_tx, writer_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = writer_tx.send(stdin.write_all(&payload).and_then(|()| stdin.flush()));
    });
    let (stdout_tx, stdout_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = stdout_tx.send(read_limited(stdout, MAX_COMMAND_OUTPUT_BYTES));
    });
    let (stderr_tx, stderr_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = stderr_tx.send(read_limited(stderr, MAX_COMMAND_OUTPUT_BYTES));
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            Ok(None) => {
                kill_process_group(pid);
                let _ = child.kill();
                let _ = child.wait();
                return Err(RewardProcessError::Timeout {
                    timeout,
                    phase: "answering",
                });
            }
            Err(error) => {
                kill_process_group(pid);
                let _ = child.kill();
                let _ = child.wait();
                return Err(RewardProcessError::Read(error));
            }
        }
    };
    // A reward command is not allowed to leave helpers behind. Closing the
    // whole group also guarantees the pipe readers below cannot wait forever.
    kill_process_group(pid);
    receive_before(&writer_rx, deadline, timeout, "stdin")?.map_err(RewardProcessError::Write)?;
    let (stdout, stdout_truncated) = receive_before(&stdout_rx, deadline, timeout, "stdout")?
        .map_err(RewardProcessError::Read)?;
    let (stderr, stderr_truncated) = receive_before(&stderr_rx, deadline, timeout, "stderr")?
        .map_err(RewardProcessError::Read)?;
    if stdout_truncated || stderr_truncated {
        return Err(RewardProcessError::OutputLimit {
            max_bytes: MAX_COMMAND_OUTPUT_BYTES,
        });
    }
    if !status.success() {
        return Err(RewardProcessError::Exited {
            status: status.to_string(),
            stderr: String::from_utf8_lossy(&stderr).trim().to_string(),
        });
    }
    let body = String::from_utf8(stdout).map_err(RewardProcessError::NotUtf8)?;
    let responses = body
        .lines()
        .enumerate()
        .map(|(index, line)| decode(index, line))
        .collect::<Result<Vec<R>>>()?;
    if responses.len() != expected {
        return Err(RewardProcessError::Count {
            received: responses.len(),
            expected,
        });
    }
    Ok(responses)
}

/// The last bytes a persistent worker wrote to stderr, so the call that finds
/// it dead can say what it complained about. Bounded: nothing ever waits for
/// this stream to end, so it is read forever and kept in part.
#[derive(Default)]
struct StderrTail(Vec<u8>);

impl StderrTail {
    fn push(&mut self, chunk: &[u8]) {
        self.0.extend_from_slice(chunk);
        if self.0.len() > STDERR_TAIL_BYTES {
            self.0.drain(..self.0.len() - STDERR_TAIL_BYTES);
        }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0).trim().to_string()
    }
}

/// A live persistent worker: the process, its stdin, the lines its stdout
/// produced, and the tail of its stderr.
struct Session {
    child: Child,
    pid: u32,
    /// `None` only while the writer thread of a call holds it.
    stdin: Option<ChildStdin>,
    lines: Receiver<Result<String>>,
    stderr: Arc<Mutex<StderrTail>>,
}

impl Session {
    fn start(command: &[String], deadline: Instant, timeout: Duration) -> Result<Self> {
        let mut child = spawn(command)?;
        let pid = child.id();
        let stdin = child.stdin.take().ok_or(RewardProcessError::Stdio)?;
        let stdout = child.stdout.take().ok_or(RewardProcessError::Stdio)?;
        let stderr = child.stderr.take().ok_or(RewardProcessError::Stdio)?;

        // Capacity one bounds queued output while still draining concurrently
        // with the exchange. The reader also refuses an oversized or
        // unterminated line instead of accumulating it for the whole timeout.
        let (lines_tx, lines) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            read_bounded_lines(stdout, lines_tx, MAX_COMMAND_OUTPUT_BYTES);
        });
        let tail = Arc::new(Mutex::new(StderrTail::default()));
        let drained = Arc::clone(&tail);
        std::thread::spawn(move || {
            let mut stderr = stderr;
            let mut chunk = [0_u8; 4096];
            while let Ok(read) = stderr.read(&mut chunk) {
                if read == 0 {
                    break;
                }
                if let Ok(mut tail) = drained.lock() {
                    tail.push(&chunk[..read]);
                }
            }
        });

        let mut session = Self {
            child,
            pid,
            stdin: Some(stdin),
            lines,
            stderr: tail,
        };
        session.handshake(deadline, timeout)?;
        tracing::debug!(
            target: "retrograd::reward",
            pid,
            command = ?command,
            "persistent reward worker started"
        );
        Ok(session)
    }

    fn handshake(&mut self, deadline: Instant, timeout: Duration) -> Result<()> {
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Hello<'a> {
            protocol: &'a str,
        }

        let hello = serde_json::to_vec(&Hello {
            protocol: REWARD_PROTOCOL_VERSION,
        })
        .map_err(RewardProcessError::Serialize)?;
        let stdin = self.stdin.as_mut().ok_or(RewardProcessError::Stdio)?;
        // One short line, so it cannot fill a pipe and cannot block: no writer
        // thread here, unlike a batch.
        stdin
            .write_all(&hello)
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|error| RewardProcessError::Handshake {
                detail: format!("writing to it failed: {error}"),
            })?;

        let line = self
            .next_line(deadline, timeout, "handshaking", 0, 1)
            .map_err(|error| RewardProcessError::Handshake {
                detail: match error {
                    RewardProcessError::Timeout { .. } => {
                        format!("no answer within {} ms", timeout.as_millis())
                    }
                    other => other.to_string(),
                },
            })?;
        let answer: Hello =
            serde_json::from_str(&line).map_err(|error| RewardProcessError::Handshake {
                detail: format!("answered {line:?}: {error}"),
            })?;
        if answer.protocol != REWARD_PROTOCOL_VERSION {
            return Err(RewardProcessError::Handshake {
                detail: format!("answered protocol {:?}", answer.protocol),
            });
        }
        Ok(())
    }

    fn exchange<R: DeserializeOwned>(
        &mut self,
        payload: Vec<u8>,
        expected: usize,
        batch: u64,
        deadline: Instant,
        timeout: Duration,
    ) -> Result<Vec<R>> {
        // The write goes to its own thread for the same reason the one-shot
        // path does it: a worker that stops reading must cost a timeout, not a
        // parent blocked in `write_all` with no deadline.
        let mut stdin = self.stdin.take().ok_or(RewardProcessError::Stdio)?;
        let (writer_tx, writer_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let outcome = stdin.write_all(&payload).and_then(|()| stdin.flush());
            let _ = writer_tx.send((stdin, outcome));
        });

        let mut responses = Vec::with_capacity(expected);
        for index in 0..expected {
            let line = self.next_line(deadline, timeout, "reading responses", index, expected)?;
            responses.push(decode_persistent(index, &line, batch)?);
        }
        // The worker closes the batch explicitly after its last response. An
        // extra response therefore fails this call deterministically, rather
        // than being noticed only if the stdout reader happened to enqueue it
        // before a try_recv(). Correlation fields also keep a line emitted
        // after this marker from being accepted by the next batch.
        let end = self.next_line(deadline, timeout, "closing the batch", expected, expected)?;
        if !is_batch_end(&end, batch) {
            return Err(RewardProcessError::BatchEnd { batch, expected });
        }

        let (stdin, written) = receive_before(&writer_rx, deadline, timeout, "stdin")?;
        self.stdin = Some(stdin);
        written.map_err(RewardProcessError::Write)?;
        Ok(responses)
    }

    fn next_line(
        &mut self,
        deadline: Instant,
        timeout: Duration,
        phase: &'static str,
        received: usize,
        expected: usize,
    ) -> Result<String> {
        match self
            .lines
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => Err(RewardProcessError::Timeout { timeout, phase }),
            // The reader thread only disconnects on EOF: the worker closed its
            // stdout, which for a process means it is gone or going.
            Err(RecvTimeoutError::Disconnected) => {
                let stderr = self.stderr_text();
                match self.child.try_wait() {
                    Ok(Some(status)) if !status.success() => Err(RewardProcessError::Exited {
                        status: status.to_string(),
                        stderr,
                    }),
                    _ => Err(RewardProcessError::Closed {
                        received,
                        expected,
                        stderr,
                    }),
                }
            }
        }
    }

    fn stderr_text(&self) -> String {
        self.stderr
            .lock()
            .map(|tail| tail.text())
            .unwrap_or_default()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Closing stdin is how the protocol says "no more batches"; a worker
        // that flushes a report or closes a file on EOF gets to. What it does
        // not get is the run's exit: after the grace period the group goes.
        self.stdin = None;
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                _ => break,
            }
        }
        kill_process_group(self.pid);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize)]
    struct Request<'a> {
        prompt: &'a str,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Response {
        reward: f32,
    }

    fn shell(script: &str) -> Vec<String> {
        vec!["/bin/sh".into(), "-c".into(), script.into()]
    }

    fn protocol(mode: RewardMode, millis: u64) -> RewardProtocol {
        RewardProtocol {
            mode,
            timeout: Duration::from_millis(millis),
        }
    }

    /// A worker in the shape the protocol asks for: answer the handshake, echo
    /// each request's correlation fields, and close every batch explicitly.
    /// `body` handles one ordinary request and can call `reply '{...}'`.
    fn worker_with_setup(setup: &str, body: &str) -> Vec<String> {
        shell(&format!(
            r#"IFS= read -r hello || exit 1
printf '{{"protocol":"{REWARD_PROTOCOL_VERSION}"}}\n'
reply() {{
  value=$(printf '%s\n' "$1" | sed 's/}}$//')
  printf '%s,"_retrograd_batch":%s,"_retrograd_index":%s}}\n' "$value" "$batch" "$index"
}}
{setup}
while IFS= read -r line; do
  case "$line" in
    *'"_retrograd_batch_end"'*) printf '%s\n' "$line";;
    *)
      batch=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_batch":\([0-9][0-9]*\).*/\1/p')
      index=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_index":\([0-9][0-9]*\).*/\1/p')
      {body}
      ;;
  esac
done"#
        ))
    }

    fn worker(body: &str) -> Vec<String> {
        worker_with_setup("", body)
    }

    fn call(command: &[String], mode: RewardMode, prompts: &[&str]) -> Vec<Response> {
        let mut process = RewardProcess::new(command, protocol(mode, 5_000)).unwrap();
        process
            .call(prompts.iter().map(|prompt| Request { prompt }))
            .unwrap()
    }

    /// A scratch path unique to this test binary and this test name, passed to
    /// the worker by interpolation rather than by environment: a test that sets
    /// a variable sets it for every other test running beside it.
    fn scratch(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "retrograd-reward-{name}-{}.txt",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn a_persistent_worker_answers_several_batches_from_one_process() {
        let marks = scratch("marks");
        let command = worker_with_setup(
            &format!("printf 'started\\n' >> '{}'", marks.display()),
            r#"reply '{"reward":1.5}'"#,
        );

        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 5_000)).unwrap();
        for _ in 0..3 {
            let responses: Vec<Response> = process
                .call([Request { prompt: "p" }, Request { prompt: "q" }])
                .unwrap();
            assert_eq!(responses.len(), 2);
            assert_eq!(responses[0].reward, 1.5);
        }
        drop(process);
        let started = std::fs::read_to_string(&marks).unwrap();
        assert_eq!(started.lines().count(), 1, "three batches, one process");
        let _ = std::fs::remove_file(&marks);
    }

    #[test]
    fn one_shot_spawns_per_batch_and_needs_no_handshake() {
        let command = shell(r#"while IFS= read -r line; do printf '{"reward":0.25}\n'; done"#);
        let responses = call(&command, RewardMode::OneShot, &["a", "b"]);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[1].reward, 0.25);
    }

    #[test]
    fn a_one_shot_command_asked_to_be_persistent_names_its_fix() {
        // A one-shot command that waits for EOF cannot answer the persistent
        // handshake because its stdin remains open.
        let command = shell(
            r#"body=$(cat); printf '%s\n' "$body" | while IFS= read -r l; do printf '{"reward":1}\n'; done"#,
        );
        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 300)).unwrap();
        let error = process
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(error.is_user_error(), "{error}");
        let message = error.to_string();
        assert!(message.contains("handshake"), "{message}");
        assert!(message.contains("reward_mode = \"oneshot\""), "{message}");
    }

    #[test]
    fn a_worker_that_dies_reports_its_stderr_and_the_next_call_restarts_it() {
        let once = scratch("once");
        let command = worker(&format!(
            "if [ -f '{path}' ]; then reply '{{\"reward\":2}}'; \
             else : > '{path}'; printf 'reward exploded\\n' >&2; exit 3; fi",
            path = once.display()
        ));

        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 5_000)).unwrap();
        let error = process
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(error.to_string().contains("reward exploded"), "{error}");
        assert!(!error.is_user_error(), "{error}");

        // The dead session is not reused: this call spawns a second process,
        // which takes the branch the marker file now selects.
        let responses: Vec<Response> = process.call([Request { prompt: "p" }]).unwrap();
        assert_eq!(responses[0].reward, 2.0);
        let _ = std::fs::remove_file(&once);
    }

    #[test]
    fn a_desynchronized_worker_is_caught_instead_of_shifting_the_next_batch() {
        // Two lines for one request: without the check, the extra line would be
        // read as the first answer of the following call.
        let command = worker(r#"reply '{"reward":1}'; reply '{"reward":2}'"#);
        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 5_000)).unwrap();
        let error = process
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(error.to_string().contains("did not close batch"), "{error}");
        assert!(error.is_user_error(), "{error}");
    }

    #[test]
    fn a_late_response_is_correlated_to_the_batch_that_produced_it() {
        // The extra line is deliberately emitted after the worker has closed
        // batch zero. It must not be attributed to batch one after the queue
        // has been observed empty.
        let command = worker_with_setup(
            "seen=0",
            r#"if [ "$seen" -eq 0 ]; then
                 reply '{"reward":1}'
                 (sleep 0.05; reply '{"reward":9}') &
                 seen=1
               else
                 reply '{"reward":2}'
               fi"#,
        );
        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 5_000)).unwrap();
        let first: Vec<Response> = process.call([Request { prompt: "first" }]).unwrap();
        assert_eq!(first[0].reward, 1.0);
        std::thread::sleep(Duration::from_millis(100));

        let error = process
            .call::<_, Response>([Request { prompt: "second" }])
            .unwrap_err();
        assert!(error.to_string().contains("belongs to batch"), "{error}");
        assert!(error.is_user_error(), "{error}");
    }

    #[test]
    fn a_short_or_unparsable_answer_is_a_user_error_in_both_modes() {
        for mode in [RewardMode::Persistent, RewardMode::OneShot] {
            let short = match mode {
                RewardMode::Persistent => worker_with_setup(
                    "seen=0",
                    r#"if [ "$seen" -eq 0 ]; then reply '{"reward":1}'; seen=1; else while :; do sleep 1; done; fi"#,
                ),
                RewardMode::OneShot => shell(r#"cat > /dev/null; printf '{"reward":1}\n'"#),
            };
            let mut process = RewardProcess::new(&short, protocol(mode, 400)).unwrap();
            let error = process
                .call::<_, Response>([Request { prompt: "a" }, Request { prompt: "b" }])
                .unwrap_err();
            match mode {
                // Persistent cannot know a batch is over, so a missing line is
                // a timeout; one-shot sees the exit and can count.
                RewardMode::Persistent => {
                    assert!(error.to_string().contains("timed out"), "{error}")
                }
                RewardMode::OneShot => assert!(
                    error.to_string().contains("1 responses for 2 rollouts"),
                    "{error}"
                ),
            }

            let garbage = match mode {
                RewardMode::Persistent => worker("printf 'not json\\n'"),
                RewardMode::OneShot => {
                    shell(r#"while IFS= read -r line; do printf 'not json\n'; done"#)
                }
            };
            let mut process = RewardProcess::new(&garbage, protocol(mode, 2_000)).unwrap();
            let error = process
                .call::<_, Response>([Request { prompt: "a" }])
                .unwrap_err();
            assert!(
                error.to_string().contains("invalid reward response 1"),
                "{error}"
            );
            assert!(error.is_user_error(), "{error}");
        }
    }

    #[test]
    fn a_hung_worker_times_out_and_a_missing_one_is_named() {
        let mut hung = RewardProcess::new(
            &worker("while :; do sleep 1; done"),
            protocol(RewardMode::Persistent, 200),
        )
        .unwrap();
        let error = hung
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");

        let mut missing = RewardProcess::new(
            &["/definitely/not/a/reward-command".to_string()],
            protocol(RewardMode::Persistent, 200),
        )
        .unwrap();
        let error = missing
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("start reward command '/definitely/not/a/reward-command'"),
            "{error}"
        );
    }

    /// A worker that writes a lot to stderr must not wedge: nothing waits for
    /// that stream to end, so it is drained for as long as the worker lives.
    #[test]
    fn stderr_is_drained_continuously_and_kept_in_part() {
        let command = worker("head -c 200000 /dev/zero >&2; reply '{\"reward\":0.5}'");
        let responses = call(&command, RewardMode::Persistent, &["p", "q"]);
        assert_eq!(responses.len(), 2);
    }

    #[test]
    fn persistent_stdout_refuses_an_oversized_unterminated_line() {
        let command = worker(&format!(
            "head -c {} /dev/zero",
            MAX_COMMAND_OUTPUT_BYTES + 1
        ));
        let mut process =
            RewardProcess::new(&command, protocol(RewardMode::Persistent, 5_000)).unwrap();
        let error = process
            .call::<_, Response>([Request { prompt: "p" }])
            .unwrap_err();
        assert!(error.to_string().contains("output exceeded"), "{error}");
        assert!(error.is_user_error(), "{error}");
    }

    #[test]
    fn an_empty_command_is_refused_before_anything_is_spawned() {
        assert!(RewardProcess::new(&[], RewardProtocol::default()).is_err());
        assert!(RewardProcess::new(&["  ".to_string()], RewardProtocol::default()).is_err());
    }
}
