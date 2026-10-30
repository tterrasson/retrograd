//! Named adapters on the scheduler, exercised on `FakeBackend` - no model
//! needed: a composed step never mixes adapters, the active adapter switches
//! only when nothing is live, and every switch invalidates the prefix tree
//! (the fake bumps its weights version on activation, like the runtime does).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use retrograd_infer::{
    ActorHandle, ActorOptions, AdmissionMode, DecodeBackend, Event, EventStream, FakeBackend, Poll,
    Request, Scheduler, SchedulerOptions, SharedFake, fake_sampling, run_requests,
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

fn scheduler(shared: &Arc<Mutex<FakeBackend>>, mode: AdmissionMode) -> Scheduler<SharedFake> {
    let mut scheduler = Scheduler::new(
        SharedFake(shared.clone()),
        SchedulerOptions {
            mode,
            n_batch: 16,
            queue_capacity: 16,
            event_capacity: 64,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        },
    );
    scheduler
        .initialize()
        .expect("the fake backend initializes");
    scheduler
}

fn request(adapter: Option<&str>, seed: u32) -> Request {
    // Eight prompt tokens so a finished sequence is long enough to stay
    // dormant in the prefix tree (min_prefix_len is 8).
    Request::single(vec![1, 2, 3, 4, 5, 6, 7, 8], fake_sampling(seed, 3))
        .with_adapter(adapter.map(str::to_owned))
}

fn load(scheduler: &mut Scheduler<SharedFake>, name: &str) {
    scheduler
        .backend_mut()
        .load_adapter(name, &PathBuf::from(format!("{name}.gguf")))
        .expect("the fake registers the adapter");
}

/// Drains one stream to its end and returns the events.
fn drain(stream: EventStream) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = stream.recv() {
        events.push(event);
    }
    events
}

#[test]
fn same_adapter_requests_batch_and_never_reactivate() {
    let shared = fake();
    let mut scheduler = scheduler(&shared, AdmissionMode::Continuous);
    load(&mut scheduler, "a");
    let outcomes = run_requests(
        &mut scheduler,
        vec![request(Some("a"), 1), request(Some("a"), 2)],
    )
    .unwrap();
    assert!(outcomes.iter().all(|outcome| outcome.error.is_none()));
    let ops = &shared.lock().unwrap().adapter_ops;
    let activations = ops.iter().filter(|(op, _)| op == "activate").count();
    assert_eq!(
        activations, 1,
        "one activation for the whole batch: {ops:?}"
    );
    // Both finished, nothing left dormant-and-swapped.
    assert!(scheduler.is_idle());
}

#[test]
fn adapters_serialize_one_at_a_time_and_each_switch_invalidates_the_kv() {
    let shared = fake();
    let mut scheduler = scheduler(&shared, AdmissionMode::Continuous);
    load(&mut scheduler, "a");
    load(&mut scheduler, "b");
    // First request under "a": finishes and leaves a dormant prefix node.
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 1)]).unwrap();
    assert!(outcomes[0].error.is_none());
    assert!(
        scheduler.kv_stats().resident_slots > 0,
        "the finished sequence stays dormant under its adapter"
    );
    let version_before = shared.lock().unwrap().weights_version;

    // A request under "b": the adapter switches (nothing was live), the tree
    // computed under "a" is dropped, and the new request is a cold miss.
    let outcomes = run_requests(&mut scheduler, vec![request(Some("b"), 2)]).unwrap();
    assert!(outcomes[0].error.is_none());
    let fake = shared.lock().unwrap();
    assert_eq!(fake.active_adapter.as_deref(), Some("b"));
    assert!(
        fake.weights_version > version_before,
        "activation bumped the weights version"
    );
    assert!(
        fake.adapter_ops
            .iter()
            .any(|(op, name)| op == "activate" && name.as_deref() == Some("b")),
        "the switch was journaled: {:?}",
        fake.adapter_ops
    );
    drop(fake);
    // Exactly one node: the one the "b" request left. Had the switch not
    // invalidated the tree, the node the "a" request left would still be there
    // (two nodes).
    assert_eq!(
        scheduler.kv_stats().nodes,
        1,
        "the prefix tree was invalidated by the adapter switch"
    );
    assert_eq!(scheduler.kv_stats().misses, 2, "both requests were cold");
}

#[test]
fn a_step_never_mixes_adapters_even_with_free_slots() {
    let shared = fake();
    // Two adapters queued: the second waits for the first to finish - the
    // batch stays homogeneous.
    let mut single = scheduler(&shared, AdmissionMode::Continuous);
    load(&mut single, "a");
    load(&mut single, "b");
    let first = single.submit(request(Some("a"), 1)).unwrap();
    let second = single.submit(request(Some("b"), 2)).unwrap();
    // Pump until both are done, recording the active adapter of each step.
    let mut seen = Vec::new();
    while !single.is_idle() {
        single.pump().unwrap();
        seen.push(shared.lock().unwrap().active_adapter.clone());
    }
    let first = drain(first);
    let second = drain(second);
    assert!(first.iter().any(|e| matches!(e, Event::End { .. })));
    assert!(second.iter().any(|e| matches!(e, Event::End { .. })));
    // Every step ran under exactly one adapter, and the switch happened only
    // at a boundary (between the two requests, not during one).
    let switches = seen.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        switches <= 1,
        "one switch at most with two requests: {seen:?}"
    );
    assert!(seen.contains(&Some("a".into())) && seen.contains(&Some("b".into())));
}

#[test]
fn a_request_for_an_unregistered_adapter_is_refused_not_hung() {
    let shared = fake();
    let mut scheduler = scheduler(&shared, AdmissionMode::Continuous);
    // No adapter registered: activation refuses, the request gets an error
    // event instead of waiting forever.
    let stream = scheduler.submit(request(Some("ghost"), 1)).unwrap();
    let outcome = loop {
        if scheduler.is_idle() && scheduler.queued_len() == 0 {
            break None;
        }
        scheduler.pump().ok();
        match stream.poll() {
            Poll::Ready(Event::Error(error)) => break Some(error),
            Poll::Ready(_) => {}
            Poll::Empty => {}
            Poll::Closed => break None,
        }
    };
    let error = outcome.expect("the request was refused with an error event");
    assert!(
        error.to_string().contains("no adapter registered"),
        "{error}"
    );
}

#[test]
fn the_actor_loads_adapters_and_reports_them_in_stats() {
    let shared = fake();
    let options = ActorOptions {
        scheduler: SchedulerOptions {
            mode: AdmissionMode::Continuous,
            n_batch: 16,
            queue_capacity: 16,
            event_capacity: 64,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        },
    };
    let actor: ActorHandle = ActorHandle::spawn(
        {
            let shared = shared.clone();
            move || Ok(SharedFake(shared))
        },
        options,
    )
    .expect("the actor thread starts");
    actor
        .load_adapter("best", &PathBuf::from("best.gguf"))
        .unwrap();
    let stats = actor.stats().unwrap();
    assert_eq!(stats.adapter, None, "loading does not activate");
    // A request under the adapter activates it at admission.
    let events = drain(actor.submit(request(Some("best"), 7)).expect("submitted"));
    assert!(events.iter().any(|e| matches!(e, Event::End { .. })));
    let stats = actor.stats().unwrap();
    assert_eq!(stats.adapter.as_deref(), Some("best"));
    assert!(stats.kv.hits + stats.kv.misses >= 1);
    actor.join().unwrap();
    let fake = shared.lock().unwrap();
    assert!(
        fake.adapter_ops
            .iter()
            .any(|(op, name)| op == "load" && name.as_deref() == Some("best"))
    );
}
