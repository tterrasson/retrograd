use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use crate::batch::*;
use crate::reader::{ObserveIndex, RolloutText};
use crate::sink::{ObserveSink, SinkConfig};
use crate::writer::LOG_FILE;

fn temp_dir(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "retrograd-observe-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    path
}

fn config(directory: &Path) -> SinkConfig {
    SinkConfig {
        directory: directory.to_path_buf(),
        every: 2,
        max_text_chars: 0,
    }
}

fn run_info(resumed_from_update: Option<u32>) -> RunInfo {
    RunInfo {
        algorithm: Algorithm::Grpo,
        model: "model.gguf".into(),
        resumed_from_update,
        params: Map::new(),
    }
}

fn prompt(key: &str) -> ObservedPrompt {
    ObservedPrompt {
        key: key.into(),
        messages: vec![ObservedMessage::text("user", "question")],
        reward_text: Some("question".into()),
        metadata: None,
    }
}

fn rollout(update: u32, member: usize, prompt: &str) -> ObservedRollout {
    ObservedRollout {
        update,
        group: Some(0),
        member,
        prompt: prompt.into(),
        seed: 7,
        tokens: 3,
        truncated: false,
        reward: Some(1.0),
        reward_raw: Some(1.0),
        judge_term: Some(0.0),
        advantage: Some(0.5),
        advantage_min: None,
        advantage_max: None,
        eligible: Some(true),
        trained: None,
        skip_reason: None,
        content: RolloutContent::Completion {
            completion: "answer".into(),
        },
    }
}

fn rollouts(update: u32, key: &str) -> ObserveBatch {
    ObserveBatch::Rollouts(RolloutBatch {
        prompts: vec![prompt(key)],
        rollouts: vec![rollout(update, 0, key), rollout(update, 1, key)],
    })
}

fn summary(update: u32) -> ObserveBatch {
    ObserveBatch::Update(UpdateSummary::new(
        update,
        UpdateStatus::Completed,
        [("reward/mean", 1.0)],
    ))
}

fn records(directory: &Path) -> Vec<Value> {
    fs::read_to_string(directory.join(LOG_FILE))
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).expect("a complete JSON line"))
        .collect()
}

fn of_type<'a>(records: &'a [Value], kind: &str) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|record| record["type"] == kind)
        .collect()
}

/// How many whole batches the log holds right now. Read while the writer
/// writes, so a line still being written is skipped rather than parsed.
fn batches(directory: &Path) -> usize {
    fs::read_to_string(directory.join(LOG_FILE))
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| {
            record["batch_index"].as_u64().map(|index| index + 1) == record["batch_len"].as_u64()
        })
        .count()
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn records_are_complete_ordered_and_read_back() {
    let directory = temp_dir("ordered");
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    let observer = sink.observer();
    observer.observe(rollouts(2, "p:0"));
    observer.observe(ObserveBatch::Outcome {
        update: 2,
        entries: vec![OutcomeEntry {
            group: Some(0),
            member: 0,
            trained: true,
        }],
    });
    sink.update_summary(2, [("reward/mean", 0.5)]);
    sink.finish();
    assert!(sink.take_warnings().is_empty());

    let records = records(&directory);
    let kinds = records
        .iter()
        .map(|record| record["type"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        ["run", "prompt", "rollout", "rollout", "outcome", "update"]
    );
    let batches = records
        .iter()
        .map(|record| {
            (
                record["batch_id"].as_u64().unwrap(),
                record["batch_index"].as_u64().unwrap(),
                record["batch_len"].as_u64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        batches,
        [
            (0, 0, 1),
            (1, 0, 3),
            (1, 1, 3),
            (1, 2, 3),
            (2, 0, 1),
            (3, 0, 1)
        ]
    );
    assert!(
        records
            .iter()
            .all(|record| record["v"] == 1 && record["segment"] == 0)
    );
    assert_eq!(records[2]["completion"], "answer");
    assert_eq!(records[5]["metrics"]["reward/mean"], 0.5);

    assert_eq!(records[0]["every"], 2, "the run record carries the cadence");

    let index = ObserveIndex::open(&directory).unwrap();
    assert_eq!(index.run().unwrap().every, Some(2));
    let updates = index.updates();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].update, 2);
    assert_eq!(updates[0].rollouts, 2);
    assert_eq!(updates[0].trained, 1);
    assert_eq!(updates[0].metrics["reward/mean"], 0.5);
    let (_, groups) = index.update(2).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].prompt, "p:0");
    let trained: Vec<Option<bool>> = groups[0].members.iter().map(|m| m.trained).collect();
    assert_eq!(trained, [Some(true), None]);
    let texts = index.rollouts(2, Some(0), None).unwrap();
    assert_eq!(texts[0].text, RolloutText::Completion("answer".into()));
    assert_eq!(
        index.prompt(0, "p:0").unwrap().unwrap().messages[0].content,
        "question"
    );

    let mut files: Vec<String> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(
        files,
        [".observe.lock", LOG_FILE],
        "the log is the whole export"
    );
    let _ = fs::remove_dir_all(directory);
}

/// Fills the channel behind a stalled writer until a batch is dropped, and
/// returns how many batches it queued.
fn saturate(sink: &ObserveSink) -> u64 {
    let observer = sink.observer();
    let mut sent = 0;
    while sink.dropped_batches() == 0 {
        observer.observe(summary(1000 + sent as u32));
        sent += 1;
    }
    sent - 1
}

#[test]
fn a_full_channel_drops_the_batch_and_warns_once() {
    let directory = temp_dir("full");
    let mut sink = ObserveSink::open_with_capacity(&config(&directory), run_info(None), 1).unwrap();
    let shared = sink.shared();
    {
        let _stalled = shared.gate.lock().unwrap();
        saturate(&sink);
        sink.observer().observe(summary(1));
        assert!(sink.dropped_batches() >= 2);
    }
    let warnings = sink.take_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("dropped"), "{warnings:?}");
    wait_until(|| batches(&directory) >= 2);
    sink.update_summary(2, []);
    sink.finish();
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_disk_error_warns_once_and_turns_the_export_off() {
    let directory = temp_dir("io");
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    wait_until(|| batches(&directory) >= 1);
    sink.shared().fail_next_write();
    let observer = sink.observer();
    assert!(observer.wants(2));
    observer.observe(summary(2));
    wait_until(|| !observer.wants(2));
    observer.observe(summary(4));
    sink.finish();
    let warnings = sink.take_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("export stops here"), "{warnings:?}");
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn wants_follows_every_and_the_sink_state() {
    let directory = temp_dir("every");
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    let observer = sink.observer();
    assert!(!observer.wants(1));
    assert!(observer.wants(2));
    assert!(!observer.wants(3));
    assert!(observer.wants(4));
    sink.finish();
    assert!(!observer.wants(2), "a closed sink wants nothing");
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_resume_opens_a_new_segment_that_supersedes_the_replayed_updates() {
    let directory = temp_dir("resume");
    let mut first = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    first.observer().observe(rollouts(2, "p:0"));
    first.update_summary(2, []);
    first.finish();

    let mut second = ObserveSink::open(&config(&directory), run_info(Some(1))).unwrap();
    second.observer().observe(rollouts(2, "p:0"));
    second.finish();

    let records = records(&directory);
    let runs = of_type(&records, "run");
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[1]["segment"], 1);
    assert_eq!(runs[1]["resumed_from_update"], 1);
    // Prompts belong to their segment: the second one writes its own.
    assert_eq!(of_type(&records, "prompt").len(), 2);
    assert_eq!(records.last().unwrap()["batch_id"], 4);

    // The first segment's update 2 is the one the resume replayed: only the
    // second segment's is in view, without the summary the first one wrote.
    let index = ObserveIndex::open(&directory).unwrap();
    let updates = index.updates();
    assert_eq!(updates.len(), 1);
    assert_eq!((updates[0].update, updates[0].segment), (2, 1));
    assert_eq!(updates[0].status, None);
    let (_, groups) = index.update(2).unwrap();
    assert_eq!(groups[0].members.len(), 2);
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn an_incomplete_tail_is_dropped_and_internal_damage_is_left_alone() {
    let directory = temp_dir("tail");
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    sink.observer().observe(rollouts(2, "p:0"));
    sink.finish();
    let log = directory.join(LOG_FILE);
    let complete = fs::read(&log).unwrap();
    let mut damaged = complete.clone();
    // The first line of an unfinished batch, then half a line.
    damaged.extend_from_slice(
        b"{\"v\":1,\"type\":\"update\",\"segment\":0,\"time\":\"t\",\"batch_id\":2,\
          \"batch_index\":0,\"batch_len\":2,\"update\":2,\"status\":\"completed\",\"metrics\":{}}\n\
          {\"v\":1,\"type\":\"upd",
    );
    fs::write(&log, &damaged).unwrap();

    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    sink.finish();
    let warnings = sink.take_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("incomplete tail")),
        "{warnings:?}"
    );
    let records = records(&directory);
    assert_eq!(records.len(), 5, "two runs, one prompt, two rollouts");
    let index = ObserveIndex::open(&directory).unwrap();
    assert_eq!(index.runs().len(), 2);
    assert!(index.updates().is_empty(), "a fresh run starts over");
    assert_eq!(records[4]["segment"], 1);
    assert_eq!(records[4]["batch_id"], 2, "the dropped batch id is reused");

    let mut corrupt = fs::read(&log).unwrap();
    corrupt.splice(0..0, b"not json\n".iter().copied());
    fs::write(&log, &corrupt).unwrap();
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    let observer = sink.observer();
    wait_until(|| !observer.wants(2));
    observer.observe(summary(2));
    sink.finish();
    let warnings = sink.take_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("left untouched"), "{warnings:?}");
    assert_eq!(fs::read(&log).unwrap(), corrupt);
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_lock_that_cannot_be_opened_disables_export_without_failing_training() {
    let directory = temp_dir("unavailable-lock");
    fs::create_dir_all(directory.join(".observe.lock")).unwrap();
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    assert!(!sink.observer().wants(2));
    let warnings = sink.take_warnings();
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("cannot be locked"));
    sink.finish();
    assert!(!directory.join(LOG_FILE).exists());
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_second_writer_on_the_same_directory_is_disabled() {
    let directory = temp_dir("exclusive");
    let mut first = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    let mut second = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    assert!(!second.observer().wants(2));
    let warnings = second.take_warnings();
    assert!(warnings[0].contains("another run"), "{warnings:?}");
    second.observer().observe(summary(2));
    second.finish();
    first.finish();
    assert_eq!(of_type(&records(&directory), "run").len(), 1);

    let mut third = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    assert!(third.observer().wants(2), "the lock goes with the writer");
    third.finish();
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_prompt_lost_with_its_batch_comes_back_with_the_next_rollout() {
    let directory = temp_dir("prompt");
    let mut sink = ObserveSink::open_with_capacity(&config(&directory), run_info(None), 1).unwrap();
    let shared = sink.shared();
    let queued = {
        let _stalled = shared.gate.lock().unwrap();
        let queued = saturate(&sink);
        let dropped = sink.dropped_batches();
        sink.observer().observe(rollouts(2, "p:0"));
        assert_eq!(sink.dropped_batches(), dropped + 1);
        queued
    };
    // The run record and every queued filler come first.
    for (chunks, update) in (1 + queued as usize..).zip([4, 6]) {
        wait_until(|| batches(&directory) == chunks);
        sink.observer().observe(rollouts(update, "p:0"));
    }
    sink.finish();
    let records = records(&directory);
    let prompts = of_type(&records, "prompt");
    assert_eq!(
        prompts.len(),
        1,
        "written once, with the first batch that got through"
    );
    let rollouts = of_type(&records, "rollout");
    assert_eq!(rollouts.len(), 4);
    assert_eq!(rollouts[0]["update"], 4);
    assert_eq!(prompts[0]["batch_id"], rollouts[0]["batch_id"]);
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn a_stalled_writer_bounds_the_stop_and_its_warning_is_still_drained() {
    let directory = temp_dir("stop");
    let mut sink = ObserveSink::open(&config(&directory), run_info(None)).unwrap();
    let observer = sink.observer();
    let shared = sink.shared();
    let stalled = shared.gate.lock().unwrap();
    observer.observe(summary(2));
    observer.observe(summary(4));
    let started = Instant::now();
    sink.finish();
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    let warnings = sink.take_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("did not finish")),
        "{warnings:?}"
    );
    assert!(!observer.wants(2));
    drop(stalled);
    let _ = fs::remove_dir_all(directory);
}
