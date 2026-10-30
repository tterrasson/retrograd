//! The level-1 KV manager over `FakeBackend`: prefix-tree hits at admission
//! (whole-slot copies at exact boundaries, partial-range copies mid-sequence
//! on a unified pool), the fully-cached-prompt replay, LRU eviction with pin
//! protection, swap to host snapshots and restore, invalidation on a weights
//! version change, and the reservation accounting that charges only the
//! non-shared suffix. No model needed; the bit-parity of a real reuse path is
//! covered by the CPU fixture lane.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use retrograd_infer::{
    ActorHandle, ActorOptions, AdmissionMode, Event, FakeBackend, FinishReason, Poll, Request,
    SchedulerOptions, SharedFake, fake_sampling,
};

fn pieces(vocab: usize) -> Vec<String> {
    (0..vocab)
        .map(|i| {
            if i == 99 {
                String::new()
            } else {
                char::from_u32('a' as u32 + (i as u32 % 26))
                    .expect("a letter")
                    .to_string()
            }
        })
        .collect()
}

/// The shared prompt (8 tokens) every request in this file starts with, and
/// the scripted generation (every request samples 91, 92, ... in order).
const PROMPT: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const SCRIPT: [i32; 2] = [91, 92];

fn fake() -> Arc<Mutex<FakeBackend>> {
    Arc::new(Mutex::new(
        FakeBackend::new(4, 64, 256, 100, 99, pieces(100)).with_script(SCRIPT.to_vec()),
    ))
}

fn options() -> ActorOptions {
    ActorOptions {
        scheduler: SchedulerOptions {
            mode: AdmissionMode::Continuous,
            n_batch: 64,
            queue_capacity: 16,
            event_capacity: 256,
            slow_client_timeout: Duration::from_millis(200),
            prefix_cache: true,
            min_prefix_len: 1,
            ..SchedulerOptions::default()
        },
    }
}

fn spawn_on(fake: &Arc<Mutex<FakeBackend>>, options: ActorOptions) -> ActorHandle {
    let shared = SharedFake(fake.clone());
    ActorHandle::spawn(move || Ok(shared), options).expect("the actor thread starts")
}

fn recv_deadline(stream: &retrograd_infer::EventStream, timeout: Duration) -> Option<Event> {
    let deadline = Instant::now() + timeout;
    loop {
        match stream.poll() {
            Poll::Ready(event) => return Some(event),
            Poll::Closed => return None,
            Poll::Empty => {
                if Instant::now() >= deadline {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

fn collect(stream: retrograd_infer::EventStream) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = recv_deadline(&stream, Duration::from_secs(10)) {
        events.push(event);
    }
    events
}

fn tokens_of(events: &[Event]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Token(token) => Some(token.token),
            _ => None,
        })
        .collect()
}

fn end_reason(events: &[Event]) -> Option<FinishReason> {
    events.iter().find_map(|event| match event {
        Event::End { reason, .. } => Some(*reason),
        _ => None,
    })
}

/// Runs one request to completion and returns its events.
fn run_request(handle: &ActorHandle, prompt: Vec<i32>, seed: u32, max_new: u32) -> Vec<Event> {
    let stream = handle
        .submit(Request::single(prompt, fake_sampling(seed, max_new)))
        .expect("submit");
    collect(stream)
}

/// The step rows recorded after `mark`, as `(slot, n_tokens)` pairs.
fn steps_after(fake: &Arc<Mutex<FakeBackend>>, mark: usize) -> Vec<(i32, u32)> {
    fake.lock()
        .expect("lock")
        .steps
        .iter()
        .skip(mark)
        .flatten()
        .map(|(slot, n, _)| (*slot, *n))
        .collect()
}

#[test]
fn shared_prefix_is_copied_not_prefilled_and_tokens_are_unchanged() {
    let fake = fake();
    let handle = spawn_on(&fake, options());

    // First request: cold, prefills the whole prompt, then generates 91, 92.
    let first = run_request(&handle, PROMPT.to_vec(), 1, 2);
    assert_eq!(tokens_of(&first), SCRIPT.to_vec());
    assert_eq!(end_reason(&first), Some(FinishReason::MaxTokens));
    // Its finished slot became a prefix-tree node at depth 10 (prompt + 2).
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.nodes, 1);
    assert_eq!(stats.resident_slots, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 0);

    // Second request: prompt = shared prefix + 2 fresh tokens. The node at
    // depth 10 is a mid-sequence hit at 8 on a unified dense fake.
    let mark = fake.lock().expect("lock").steps.len();
    let mut prompt_b = PROMPT.to_vec();
    prompt_b.extend_from_slice(&[9, 10]);
    let second = run_request(&handle, prompt_b.clone(), 2, 2);
    if tokens_of(&second) != SCRIPT.to_vec() {
        let guard = fake.lock().expect("lock");
        panic!(
            "cache reuse keeps tokens identical: events={second:?} copies={:?} steps={:?} removed={:?}",
            guard.copies, guard.steps, guard.removed
        );
    }

    let fake_guard = fake.lock().expect("lock");
    // A partial-range copy of positions [0, 7] went from the node's slot.
    assert!(
        fake_guard
            .copies
            .iter()
            .any(|&(src, dst, p0, p1)| p0 == 0 && p1 == 8 && src != dst),
        "expected a partial copy [0, 8) (p1 exclusive), got {:?}",
        fake_guard.copies
    );
    drop(fake_guard);
    // The prefill rows of the second request carry only the 2-token suffix,
    // never the 8-token shared prefix.
    let rows = steps_after(&fake, mark);
    let prefilled: u32 = rows.iter().map(|(_, n)| *n).sum();
    // 2-token suffix prefill (which also samples the first token) + 1 decode.
    assert_eq!(prefilled, 3, "suffix prefill + decodes only: {rows:?}");

    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.prompt_cells, 8 + 10);
    assert_eq!(stats.prefilled_cells, 8 + 2);

    // Pin the shared-prefix node so slot pressure from the later requests
    // cannot evict it mid-test (that path is covered by its own test).
    let mut shared = PROMPT.to_vec();
    shared.push(SCRIPT[0]);
    let _pin = handle.pin_prefix(&shared).expect("the shared node exists");

    // Warm-up excluded: three more requests sharing the 8-token prefix with a
    // 1-token suffix each - the prefilled/prompt ratio over those hits is
    // 3 / 27 < 0.2, the sortie criterion for a >80 % shared prefix.
    for (i, seed) in (3..=5).enumerate() {
        let mut prompt = PROMPT.to_vec();
        prompt.push(9 + i as i32);
        let events = run_request(&handle, prompt, seed, 2);
        assert_eq!(tokens_of(&events), SCRIPT.to_vec());
    }
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.hits, 4, "{stats:?}");
    let warm_prefilled = stats.prefilled_cells - 8;
    let warm_prompt = stats.prompt_cells - 8;
    assert!(
        warm_prefilled * 5 < warm_prompt,
        "warm-up prefilled/prompt < 0.2: prefilled={warm_prefilled} prompt={warm_prompt}"
    );
    handle.join().expect("actor joins");
}

#[test]
fn fully_cached_prompt_prefills_only_the_last_token() {
    let fake = fake();
    let handle = spawn_on(&fake, options());
    run_request(&handle, PROMPT.to_vec(), 1, 2);

    // Same prompt exactly: the whole prompt is cached, but the applied hit is
    // clamped to a strict prefix (the backend refuses to resubmit a resident
    // token), so only the last prompt token is prefilled, then plain decodes.
    let mark = fake.lock().expect("lock").steps.len();
    let events = run_request(&handle, PROMPT.to_vec(), 2, 2);
    assert_eq!(tokens_of(&events), SCRIPT.to_vec());
    let rows = steps_after(&fake, mark);
    assert!(
        rows.iter().all(|(_, n)| *n == 1),
        "a fully cached prompt only prefills its last token then decodes: {rows:?}"
    );
    assert_eq!(rows.len(), 2, "last-token prefill + 1 decode: {rows:?}");
    let guard = fake.lock().expect("lock");
    assert!(
        guard
            .copies
            .iter()
            .any(|&(_, _, p0, p1)| p0 == 0 && p1 == 7),
        "a partial copy [0, 7) (hit clamped to len-1, p1 exclusive): {:?}",
        guard.copies
    );
    drop(guard);
    handle.join().expect("actor joins");
}

#[test]
fn exact_boundary_hit_uses_a_whole_slot_copy_even_on_a_partitioned_pool() {
    // Node path = prompt + script = [1..8, 91, 92] (depth 10).
    let mut path = PROMPT.to_vec();
    path.extend_from_slice(&SCRIPT);

    for unified in [true, false] {
        let fake = Arc::new(Mutex::new(
            FakeBackend::new(4, 64, 256, 100, 99, pieces(100))
                .with_script(SCRIPT.to_vec())
                .with_kv_unified(unified),
        ));
        let handle = spawn_on(&fake, options());
        run_request(&handle, PROMPT.to_vec(), 1, 2);

        // A prompt equal to the node's full path: exact boundary at depth 10,
        // served by a whole-slot copy - which a partitioned pool supports.
        let mark = fake.lock().expect("lock").steps.len();
        let events = run_request(&handle, path.clone(), 2, 2);
        assert_eq!(tokens_of(&events), SCRIPT.to_vec());
        let guard = fake.lock().expect("lock");
        assert!(
            guard
                .copies
                .iter()
                .any(|&(_, _, p0, p1)| p0 == -1 && p1 == -1),
            "unified={unified}: expected a whole-slot copy, got {:?}",
            guard.copies
        );
        drop(guard);
        let rows = steps_after(&fake, mark);
        assert!(
            rows.iter().all(|(_, n)| *n == 1),
            "unified={unified}: full-prompt hit replays one token: {rows:?}"
        );
        let stats = handle.kv_stats().expect("stats");
        assert_eq!(stats.hits, 1, "unified={unified}");
        handle.join().expect("actor joins");
    }
}

#[test]
fn partitioned_pool_refuses_mid_sequence_hits_and_recomputes() {
    let fake = Arc::new(Mutex::new(
        FakeBackend::new(4, 64, 256, 100, 99, pieces(100))
            .with_script(SCRIPT.to_vec())
            .with_kv_unified(false),
    ));
    let handle = spawn_on(&fake, options());
    run_request(&handle, PROMPT.to_vec(), 1, 2);

    // Prompt shorter than the node's path, ending mid-edge: needs a partial
    // copy, unsupported on a partitioned pool -> cold prefill.
    let mut prompt = PROMPT.to_vec();
    prompt.truncate(6);
    let mark = fake.lock().expect("lock").steps.len();
    let events = run_request(&handle, prompt, 2, 1);
    assert_eq!(tokens_of(&events), vec![91]);
    let rows = steps_after(&fake, mark);
    assert!(
        rows.first().is_some_and(|(_, n)| *n == 6),
        "cold prefill of the whole 6-token prompt: {rows:?}"
    );
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 2);
    handle.join().expect("actor joins");
}

#[test]
fn recurrent_models_only_reuse_strict_prefix_boundaries() {
    let fake = Arc::new(Mutex::new(
        FakeBackend::new(4, 64, 256, 100, 99, pieces(100))
            .with_script(SCRIPT.to_vec())
            .with_recurrent(true),
    ));
    let handle = spawn_on(&fake, options());
    run_request(&handle, PROMPT.to_vec(), 1, 2);

    // Full-prompt hit on a hybrid would replay the last token and
    // double-advance the recurrent state: refused, cold prefill instead.
    let mark = fake.lock().expect("lock").steps.len();
    let events = run_request(&handle, PROMPT.to_vec(), 2, 1);
    assert_eq!(tokens_of(&events), vec![91]);
    let rows = steps_after(&fake, mark);
    assert!(
        rows.first().is_some_and(|(_, n)| *n == 8),
        "hybrid full-prompt hit refused, cold prefill: {rows:?}"
    );

    // A strict-prefix hit (node boundary strictly inside the prompt) is a
    // whole-slot copy and is allowed.
    let mut path = PROMPT.to_vec();
    path.extend_from_slice(&SCRIPT);
    path.extend_from_slice(&[7, 8]);
    let mark = fake.lock().expect("lock").steps.len();
    let events = run_request(&handle, path, 3, 1);
    assert_eq!(tokens_of(&events), vec![91]);
    let guard = fake.lock().expect("lock");
    assert!(
        guard
            .copies
            .iter()
            .any(|&(_, _, p0, p1)| p0 == -1 && p1 == -1),
        "expected a whole-slot copy at the exact boundary: {:?}",
        guard.copies
    );
    drop(guard);
    let rows = steps_after(&fake, mark);
    assert!(
        rows.first().is_some_and(|(_, n)| *n == 3),
        "only the 3-token suffix is prefilled: {rows:?}"
    );
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.hits, 1);
    handle.join().expect("actor joins");
}

#[test]
fn lru_eviction_frees_slots_and_pin_protects_a_node() {
    let fake = fake();
    let mut opts = options();
    opts.scheduler.swap_min_tokens = usize::MAX; // drop, do not swap
    let handle = spawn_on(&fake, opts);
    run_request(&handle, PROMPT.to_vec(), 1, 1);

    // Pin the node at the resident prefix (the prompt itself: the single
    // generated token is sampled, never fed back, so the node covers the 8
    // prompt tokens). The node's slot is 0.
    let pin = handle.pin_prefix(&PROMPT).expect("the node exists");
    let base = fake.lock().expect("lock").removed.len();

    // Four more requests need the 4 slots: the pinned node must survive, the
    // other nodes are evicted LRU-first.
    for seed in 2..=5 {
        run_request(&handle, vec![50 + seed as i32, 60], seed, 1);
    }
    let stats = handle.kv_stats().expect("stats");
    assert!(stats.evictions >= 1, "{stats:?}");
    let guard = fake.lock().expect("lock");
    assert!(
        !guard.removed[base..].contains(&0),
        "the pinned node's slot must not be removed: {:?}",
        guard.removed
    );
    drop(guard);
    assert!(handle.unpin(pin).expect("unpin"));
    // After unpin, the node is evictable again.
    run_request(&handle, vec![70, 71], 9, 1);
    let guard = fake.lock().expect("lock");
    assert!(
        guard.removed[base..].contains(&0),
        "after unpin the node is evicted: {:?}",
        guard.removed
    );
    drop(guard);
    handle.join().expect("actor joins");
}

#[test]
fn long_nodes_swap_to_a_host_snapshot_and_restore_on_a_later_hit() {
    let fake = fake();
    let mut opts = options();
    opts.scheduler.swap_min_tokens = 1;
    opts.scheduler.max_snapshot_bytes = 1 << 20;
    let handle = spawn_on(&fake, opts);
    run_request(&handle, PROMPT.to_vec(), 1, 2);

    // Saturate the 4 slots so admission evicts the node: it is swapped (a full
    // snapshot is saved to host) instead of dropped.
    for seed in 2..=5 {
        run_request(&handle, vec![50 + seed as i32, 60], seed, 1);
    }
    {
        let guard = fake.lock().expect("lock");
        assert!(
            guard
                .saves
                .iter()
                .any(|&(_, flags, len)| flags == 0 && len > 0),
            "expected a full swap snapshot: {:?}",
            guard.saves
        );
    }
    let stats = handle.kv_stats().expect("stats");
    assert!(stats.swaps >= 1, "{stats:?}");
    assert!(stats.snapshot_bytes > 0, "{stats:?}");

    // A later request whose prompt extends the swapped node's exact path hits
    // it: the snapshot is loaded into its slot and only the suffix prefilled.
    let mut prompt = PROMPT.to_vec();
    prompt.extend_from_slice(&[SCRIPT[0], 9]);
    let mark = fake.lock().expect("lock").steps.len();
    let events = run_request(&handle, prompt, 6, 2);
    assert_eq!(tokens_of(&events), SCRIPT.to_vec());
    {
        let guard = fake.lock().expect("lock");
        assert!(
            guard
                .loads
                .iter()
                .any(|&(_, flags, len)| flags == 0 && len > 0),
            "expected a snapshot restore: {:?}",
            guard.loads
        );
    }
    let rows = steps_after(&fake, mark);
    let prefilled: u32 = rows.iter().map(|(_, n)| *n).sum();
    assert_eq!(prefilled, 2, "only the 1-token suffix + 1 decode: {rows:?}");
    handle.join().expect("actor joins");
}

#[test]
fn a_weights_version_change_drops_the_tree_and_the_next_request_is_cold() {
    let fake = fake();
    let handle = spawn_on(&fake, options());
    run_request(&handle, PROMPT.to_vec(), 1, 2);
    assert_eq!(handle.kv_stats().expect("stats").nodes, 1);

    // An optimizer step / adapter change on the backend side.
    fake.lock().expect("lock").bump_weights_version();

    let mark = fake.lock().expect("lock").steps.len();
    let mut prompt = PROMPT.to_vec();
    prompt.push(9);
    let events = run_request(&handle, prompt, 2, 1);
    assert_eq!(tokens_of(&events), vec![91]);
    let rows = steps_after(&fake, mark);
    assert!(
        rows.first().is_some_and(|(_, n)| *n == 9),
        "after invalidation the whole prompt is prefilled again: {rows:?}"
    );
    let guard = fake.lock().expect("lock");
    assert!(!guard.removed.is_empty(), "the tree's slots were released");
    drop(guard);
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.hits, 0);
    handle.join().expect("actor joins");
}

#[test]
fn a_hit_charges_only_the_suffix_against_the_cell_budget() {
    // Pool of 19 cells: the first request leaves 9 cells used (its finished
    // slot is a tree node spanning positions 0..8). A second request with a
    // 9-token prompt and budget 2 needs 11 cells cold - more than the 10 left -
    // but only 1 + 2 = 3 with the 8-token shared prefix.
    let fake = Arc::new(Mutex::new(
        FakeBackend::new(4, 64, 19, 100, 99, pieces(100)).with_script(SCRIPT.to_vec()),
    ));
    let handle = spawn_on(&fake, options());
    run_request(&handle, PROMPT.to_vec(), 1, 2);

    let mut prompt = PROMPT.to_vec();
    prompt.push(9);
    let events = run_request(&handle, prompt, 2, 2);
    assert_eq!(
        end_reason(&events),
        Some(FinishReason::MaxTokens),
        "the suffixed request is admitted thanks to the shared prefix: {events:?}"
    );
    assert_eq!(tokens_of(&events), SCRIPT.to_vec());
    let stats = handle.kv_stats().expect("stats");
    assert_eq!(stats.hits, 1, "{stats:?}");
    handle.join().expect("actor joins");
}
