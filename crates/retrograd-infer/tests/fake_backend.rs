//! The scheduler's properties, exercised on `FakeBackend` - no model needed:
//! no starving request, the per-step budget respected, a cancellation freeing
//! its slot at the next step, `n > 1` as one prefill plus `seq_cp`, the two
//! admission modes, queue-full and slow-client handling, shutdown, and the
//! streaming detokenization end to end.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use retrograd_infer::{
    ActorHandle, ActorOptions, AdmissionMode, DecodeBackend, Event, FakeBackend, FinishReason,
    InferError, Poll, Request, SchedulerOptions, SharedFake, fake_sampling, fake_sampling_v2,
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

fn fake() -> Arc<Mutex<FakeBackend>> {
    Arc::new(Mutex::new(FakeBackend::new(
        2,
        32,
        64,
        100,
        99,
        pieces(100),
    )))
}

fn options(mode: AdmissionMode, n_batch: usize) -> ActorOptions {
    ActorOptions {
        scheduler: SchedulerOptions {
            mode,
            n_batch,
            queue_capacity: 16,
            event_capacity: 256,
            slow_client_timeout: Duration::from_millis(150),
            ..SchedulerOptions::default()
        },
    }
}

fn spawn_on(fake: &Arc<Mutex<FakeBackend>>, options: ActorOptions) -> ActorHandle {
    let shared = SharedFake(fake.clone());
    ActorHandle::spawn(move || Ok(shared), options).expect("the actor thread starts")
}

// Hold backend construction until every request in a test wave is queued.
// Otherwise thread scheduling can split the wave before the second submit.
fn spawn_paused(
    fake: &Arc<Mutex<FakeBackend>>,
    options: ActorOptions,
) -> (ActorHandle, std::sync::mpsc::Sender<()>) {
    let shared = SharedFake(fake.clone());
    let (start, ready) = std::sync::mpsc::channel();
    let handle = ActorHandle::spawn(
        move || {
            ready.recv().expect("release the actor");
            Ok(shared)
        },
        options,
    )
    .expect("actor starts");
    (handle, start)
}

/// Receives with a deadline, so a broken scheduler fails the test instead of
/// hanging it. A closed stream ends the wait immediately.
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

fn tokens_of(events: &[Event], index: usize) -> Vec<i32> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Token(token) if token.index == index => Some(token.token),
            _ => None,
        })
        .collect()
}

fn end_of(events: &[Event], index: usize) -> Option<FinishReason> {
    events.iter().find_map(|event| match event {
        Event::End {
            index: i, reason, ..
        } if *i == index => Some(*reason),
        _ => None,
    })
}

#[test]
fn no_request_starves_and_the_step_budget_is_respected() {
    let fake = fake();
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 4));
    // Six requests, two slots: three rounds of admission at most.
    let mut streams = Vec::new();
    for seed in 1..=6 {
        streams.push(
            handle
                .submit(Request::single(vec![10, 11, 12], fake_sampling(seed, 3)))
                .expect("submit"),
        );
    }
    for stream in streams {
        let events = collect(stream);
        assert_eq!(tokens_of(&events, 0).len(), 3, "{events:?}");
        assert_eq!(
            end_of(&events, 0),
            Some(FinishReason::MaxTokens),
            "{events:?}"
        );
    }
    handle.shutdown();
    handle.join().expect("the actor drains cleanly");
    let fake = fake.lock().expect("lock");
    // The budget of one step is n_batch tokens, whatever the mode packs in it.
    for step in &fake.steps {
        let total: u64 = step.iter().map(|(_, n, _)| *n as u64).sum();
        assert!(total <= 4, "step {step:?} exceeds the n_batch budget");
    }
    assert!(!fake.steps.is_empty());
    // Every slot was returned (the initial clear and the per-admission clear
    // also land in the log).
    assert_eq!(fake.memory().unwrap().cells_used, 0);
    assert!(fake.removed.len() >= 6, "one seq_rm per finished request");
}

#[test]
fn lockstep_runs_waves_and_continuous_refills_freed_slots() {
    // Three requests, two slots. In Lockstep the third waits for the whole
    // first wave; in Continuous it takes the slot the first finisher frees.
    for (mode, expect_mixed) in [
        (AdmissionMode::Lockstep, false),
        (AdmissionMode::Continuous, true),
    ] {
        let fake = fake();
        let (handle, start) = spawn_paused(&fake, options(mode, 8));
        let prompts = [vec![1, 1, 1], vec![2, 2, 2], vec![3, 3, 3]];
        let budgets = [2, 5, 1];
        let mut streams = Vec::new();
        for (prompt, budget) in prompts.iter().zip(budgets) {
            streams.push(
                handle
                    .submit(Request::single(prompt.clone(), fake_sampling(7, budget)))
                    .expect("submit"),
            );
        }
        start.send(()).unwrap();
        for stream in streams {
            let events = collect(stream);
            assert!(end_of(&events, 0).is_some(), "{mode:?}: {events:?}");
        }
        handle.shutdown();
        handle.join().expect("clean drain");
        let fake = fake.lock().expect("lock");
        let order = &fake.request_order;
        assert_eq!(order.len(), 3, "{mode:?}: {order:?}");
        let (wave_one, late) = (&order[..2], order[2]);
        let mixed = fake.steps.iter().any(|step| {
            let requests: Vec<i32> = step.iter().filter_map(|(_, _, r)| *r).collect();
            requests.contains(&late) && requests.iter().any(|r| wave_one.contains(r))
        });
        assert_eq!(mixed, expect_mixed, "{mode:?}: steps {:?}", fake.steps);
    }
}

#[test]
fn lockstep_merges_identical_prompts_into_one_prefill_and_copies() {
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        4,
        32,
        128,
        100,
        99,
        pieces(100),
    )));
    let (handle, start) = spawn_paused(&fake, options(AdmissionMode::Lockstep, 8));
    // Two requests with the same prompt: one prefill, one seq_cp. Collect
    // before shutdown so the wave is admitted first.
    let first = handle
        .submit(Request::single(vec![5, 6, 7, 8], fake_sampling(1, 2)))
        .expect("submit");
    let second = handle
        .submit(Request::single(vec![5, 6, 7, 8], fake_sampling(2, 2)))
        .expect("submit");
    start.send(()).unwrap();
    let events_a = collect(first);
    let events_b = collect(second);
    for events in [&events_a, &events_b] {
        assert_eq!(tokens_of(events, 0).len(), 2, "{events:?}");
        assert_eq!(end_of(events, 0), Some(FinishReason::MaxTokens));
    }
    handle.shutdown();
    handle.join().expect("clean drain");
    let fake = fake.lock().expect("lock");
    assert_eq!(
        fake.copies.len(),
        1,
        "one copy for the duplicate: {:?}",
        fake.copies
    );
    // The shared prefix is decoded once: exactly one non-sampling row carries
    // the 3-token prefix (prompt.len() - 1).
    let prefix_rows = fake
        .steps
        .iter()
        .flatten()
        .filter(|(slot, n, request)| *n == 3 && request.is_none() && *slot == 0)
        .count();
    assert_eq!(
        prefix_rows, 1,
        "the prefix is prefilled once: {:?}",
        fake.steps
    );
}

#[test]
fn n_greater_than_one_prefills_once_then_copies_to_each_member() {
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        4,
        32,
        128,
        100,
        99,
        pieces(100),
    )));
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));
    let mut request = Request::single(vec![5, 6, 7, 8, 9], fake_sampling(1, 2));
    request.sampling = vec![
        fake_sampling_v2(1, 2),
        fake_sampling_v2(2, 2),
        fake_sampling_v2(3, 2),
    ];
    let stream = handle.submit(request).expect("submit");
    let events = collect(stream);
    handle.shutdown();
    handle.join().expect("clean drain");
    for index in 0..3 {
        assert_eq!(
            tokens_of(&events, index).len(),
            2,
            "member {index}: {events:?}"
        );
        assert_eq!(end_of(&events, index), Some(FinishReason::MaxTokens));
    }
    let fake = fake.lock().expect("lock");
    assert_eq!(
        fake.copies.len(),
        2,
        "leader copied to both members: {:?}",
        fake.copies
    );
    assert_eq!(fake.copies[0].0, fake.copies[1].0, "same leader slot");
    // One prefix prefill of prompt.len() - 1 tokens, then one terminal row per
    // member in a single step.
    let prefix = fake
        .steps
        .iter()
        .flatten()
        .filter(|(_, n, request)| *n == 4 && request.is_none())
        .count();
    assert_eq!(prefix, 1, "the prefix is prefilled once: {:?}", fake.steps);
    let terminal_step = fake
        .steps
        .iter()
        .find(|step| step.iter().filter(|(_, _, r)| r.is_some()).count() == 3);
    assert!(
        terminal_step.is_some(),
        "one step samples all members: {:?}",
        fake.steps
    );
}

#[test]
fn a_cancelled_request_frees_its_slot_at_the_next_step() {
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        1,
        64,
        64,
        100,
        99,
        pieces(100),
    )));
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));
    let long = handle
        .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 50)))
        .expect("submit");
    // Wait until the request is live, then abandon the stream.
    assert!(matches!(
        recv_deadline(&long, Duration::from_secs(5)),
        Some(Event::Token(_))
    ));
    drop(long);
    // The slot must come back: this request can only run if the cancelled
    // one's slot was freed at the following step.
    let second = handle
        .submit(Request::single(vec![4, 5, 6], fake_sampling(2, 2)))
        .expect("submit");
    let events = collect(second);
    assert_eq!(tokens_of(&events, 0).len(), 2, "{events:?}");
    assert_eq!(end_of(&events, 0), Some(FinishReason::MaxTokens));
    handle.shutdown();
    handle.join().expect("clean drain");
    let fake = fake.lock().expect("lock");
    assert!(
        fake.removed.contains(&0),
        "the cancelled slot was returned: {:?}",
        fake.removed
    );
    assert_eq!(fake.memory().unwrap().cells_used, 0);
}

#[test]
fn a_full_queue_is_refused_with_a_capacity_error() {
    let fake = fake();
    let mut opts = options(AdmissionMode::Continuous, 8);
    opts.scheduler.queue_capacity = 1;
    let (handle, start) = spawn_paused(&fake, opts);
    let first = handle
        .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 3)))
        .unwrap();
    assert!(matches!(
        handle.submit(Request::single(vec![7, 7, 7], fake_sampling(2, 2))),
        Err(InferError::QueueFull { capacity: 1 })
    ));
    start.send(()).unwrap();
    let events = collect(first);
    assert_eq!(
        end_of(&events, 0),
        Some(FinishReason::MaxTokens),
        "{events:?}"
    );
    handle.join().expect("join initiates graceful shutdown");
}

#[test]
fn a_slow_client_is_cancelled_after_the_bounded_backpressure() {
    let fake = fake();
    let mut opts = options(AdmissionMode::Continuous, 8);
    opts.scheduler.event_capacity = 2;
    opts.scheduler.slow_client_timeout = Duration::from_millis(100);
    let handle = spawn_on(&fake, opts);
    let slow = handle
        .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 30)))
        .expect("submit");
    // Read one event, then stop reading: the stream fills, the bounded wait
    // expires, the request is cancelled.
    assert!(matches!(
        recv_deadline(&slow, Duration::from_secs(5)),
        Some(Event::Token(_))
    ));
    // The other request keeps flowing: the decode thread is not blocked.
    let quick = handle
        .submit(Request::single(vec![4, 5, 6], fake_sampling(2, 2)))
        .expect("submit");
    let events = collect(quick);
    assert_eq!(
        end_of(&events, 0),
        Some(FinishReason::MaxTokens),
        "{events:?}"
    );
    // The slow request's stream closes (cancellation) rather than hanging;
    // the final End may have been dropped by the full buffer, which is why
    // only token events are guaranteed to be readable here.
    let tail = collect(slow);
    assert!(
        tail.iter().all(|event| matches!(event, Event::Token(_)))
            || matches!(
                tail.last(),
                Some(Event::End {
                    reason: FinishReason::Cancelled,
                    ..
                })
            ),
        "the slow request ended by cancellation: {tail:?}"
    );
    handle.shutdown();
    handle.join().expect("clean drain");
    let fake = fake.lock().expect("lock");
    assert_eq!(fake.memory().unwrap().cells_used, 0, "every slot came back");
}

#[test]
fn shutdown_lets_live_requests_finish_and_refuses_queued_ones() {
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        1,
        64,
        64,
        100,
        99,
        pieces(100),
    )));
    let mut opts = options(AdmissionMode::Continuous, 8);
    opts.scheduler.event_capacity = 1;
    opts.scheduler.slow_client_timeout = Duration::from_secs(5);
    let handle = spawn_on(&fake, opts);
    let live = handle
        .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 3)))
        .expect("submit");
    assert!(matches!(
        recv_deadline(&live, Duration::from_secs(5)),
        Some(Event::Token(_))
    ));
    // Queued behind the single slot: never admitted.
    let queued = handle
        .submit(Request::single(vec![4, 5, 6], fake_sampling(2, 2)))
        .expect("submit");
    handle.shutdown();
    let live_events = collect(live);
    assert_eq!(
        end_of(&live_events, 0),
        Some(FinishReason::MaxTokens),
        "{live_events:?}"
    );
    let queued_events = collect(queued);
    assert!(
        matches!(
            queued_events.last(),
            Some(Event::Error(InferError::Cancelled))
        ),
        "the queued request is refused at shutdown: {queued_events:?}"
    );
    handle.join().expect("the actor exits after draining");
}

#[test]
fn an_impossible_reservation_is_refused_instead_of_waiting() {
    // A pool of 10 cells cannot hold prompt(5) + max_new(10).
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        2,
        32,
        10,
        100,
        99,
        pieces(100),
    )));
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));
    let too_big = handle
        .submit(Request::single(vec![1, 2, 3, 4, 5], fake_sampling(1, 10)))
        .expect("submit");
    let events = collect(too_big);
    assert!(
        matches!(
            events.last(),
            Some(Event::Error(InferError::InvalidArgument(_)))
        ),
        "an impossible reservation is refused: {events:?}"
    );
    // A prompt that leaves no room in the per-sequence context is refused too.
    let no_room = handle
        .submit(Request::single(vec![1; 32], fake_sampling(2, 1)))
        .expect("submit");
    let events = collect(no_room);
    assert!(
        matches!(
            events.last(),
            Some(Event::Error(InferError::InvalidArgument(_)))
        ),
        "{events:?}"
    );
    handle.shutdown();
    handle.join().expect("clean drain");
}

#[test]
fn stop_strings_eog_and_budgets_end_the_stream_with_the_right_reason() {
    // A scripted model: token 10 renders "hello", 11 renders "</end>",
    // 12 renders "!".
    let mut table = pieces(100);
    table[10] = "hello".to_string();
    table[11] = "</end>".to_string();
    table[12] = "!".to_string();
    let fake = Arc::new(Mutex::new(
        FakeBackend::new(2, 32, 64, 100, 99, table).with_script(vec![10, 11, 12]),
    ));
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));

    // A stop string ends the stream and is not emitted.
    let mut request = Request::single(vec![1, 2, 3], fake_sampling(1, 10));
    request.stop = vec!["</end>".to_string()];
    let events = collect(handle.submit(request).expect("submit"));
    let texts: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            Event::Token(token) => Some(token.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        ["hello", ""],
        "the stop itself is not emitted: {events:?}"
    );
    assert_eq!(end_of(&events, 0), Some(FinishReason::Stop), "{events:?}");

    // Without the stop, the whole script is emitted and the budget ends it.
    let events = collect(
        handle
            .submit(Request::single(vec![1, 2, 3], fake_sampling(2, 2)))
            .expect("submit"),
    );
    let texts: String = events
        .iter()
        .filter_map(|event| match event {
            Event::Token(token) => Some(token.text.as_str()),
            Event::End { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts, "hello</end>",
        "no stop: everything is emitted: {events:?}"
    );
    assert_eq!(
        end_of(&events, 0),
        Some(FinishReason::MaxTokens),
        "{events:?}"
    );

    handle.shutdown();
    handle.join().expect("clean drain");
}

#[test]
fn eog_ends_the_stream_and_keeps_the_token() {
    let fake = Arc::new(Mutex::new(
        FakeBackend::new(2, 32, 64, 100, 99, pieces(100)).with_eog_after(2),
    ));
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));
    let events = collect(
        handle
            .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 10)))
            .expect("submit"),
    );
    let tokens = tokens_of(&events, 0);
    assert_eq!(
        tokens.len(),
        3,
        "two sampled tokens plus the EOG: {events:?}"
    );
    assert_eq!(tokens[2], 99, "the EOG token is kept in the stream");
    assert_eq!(end_of(&events, 0), Some(FinishReason::Eog), "{events:?}");
    handle.shutdown();
    handle.join().expect("clean drain");
}

#[test]
fn top_logprobs_ride_along_when_requested() {
    let fake = fake();
    let handle = spawn_on(&fake, options(AdmissionMode::Continuous, 8));
    let mut request = Request::single(vec![1, 2, 3], fake_sampling(1, 2));
    request.n_top_logprobs = 3;
    let events = collect(handle.submit(request).expect("submit"));
    handle.shutdown();
    handle.join().expect("clean drain");
    let token_events: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::Token(token) => Some(token),
            _ => None,
        })
        .collect();
    assert_eq!(token_events.len(), 2);
    for token in token_events {
        assert_eq!(token.top_logprobs.len(), 3, "{token:?}");
        assert_eq!(token.top_logprobs[0].token, 0);
    }
}

#[test]
fn queued_one_token_completions_run_without_another_inbox_message() {
    let fake = Arc::new(Mutex::new(FakeBackend::new(
        1,
        32,
        32,
        100,
        99,
        pieces(100),
    )));
    let (handle, start) = spawn_paused(&fake, options(AdmissionMode::Continuous, 8));
    let streams: Vec<_> = (1..=3)
        .map(|seed| {
            handle
                .submit(Request::single(vec![1], fake_sampling(seed, 1)))
                .unwrap()
        })
        .collect();
    start.send(()).unwrap();
    for stream in streams {
        let events = collect(stream);
        assert_eq!(
            end_of(&events, 0),
            Some(FinishReason::MaxTokens),
            "{events:?}"
        );
    }
    handle.join().unwrap();
}

#[test]
fn shutdown_refuses_every_request_still_waiting_in_both_queues() {
    let fake = fake();
    let (handle, start) = spawn_paused(&fake, options(AdmissionMode::Continuous, 8));
    let streams: Vec<_> = (1..=3)
        .map(|seed| {
            handle
                .submit(Request::single(vec![1], fake_sampling(seed, 1)))
                .unwrap()
        })
        .collect();
    handle.shutdown();
    assert!(matches!(
        handle.submit(Request::single(vec![1], fake_sampling(4, 1))),
        Err(InferError::Cancelled)
    ));
    start.send(()).unwrap();
    handle.join().unwrap();
    for stream in streams {
        let events = collect(stream);
        assert!(
            matches!(events.as_slice(), [Event::Error(InferError::Cancelled)]),
            "{events:?}"
        );
    }
    assert!(fake.lock().unwrap().steps.is_empty());
}

#[test]
fn invalid_logit_bias_is_refused_without_failing_other_requests() {
    let fake = fake();
    let (handle, start) = spawn_paused(&fake, options(AdmissionMode::Lockstep, 8));
    let mut invalid = Request::single(vec![1, 2], fake_sampling(1, 3));
    invalid.sampling[0].logit_bias = vec![(100, 1.0)];
    let refused = handle.submit(invalid).unwrap();
    let valid = handle
        .submit(Request::single(vec![3, 4], fake_sampling(2, 3)))
        .unwrap();
    start.send(()).unwrap();
    assert!(
        collect(refused)
            .iter()
            .any(|event| matches!(event, Event::Error(InferError::InvalidArgument(_))))
    );
    let events = collect(valid);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token(_)))
            .count(),
        3
    );
    assert!(events.iter().all(|event| !matches!(event, Event::Error(_))));
    handle.shutdown();
    handle.join().unwrap();
}
