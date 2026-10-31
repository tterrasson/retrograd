//! Per-row adapters on the scheduler, exercised on `FakeBackend` - no model
//! needed: requests for different per-row adapters and the base model share
//! steps, never share a prefix, and an adapter replaced or unloaded while a
//! request decodes under it is freed only once that request ends.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use retrograd_engine::{AdapterId, RowAdapter};
use retrograd_infer::{
    AdmissionMode, Event, EventStream, FakeBackend, Request, Scheduler, SchedulerOptions,
    SharedFake, fake_sampling, run_requests,
};

fn fake() -> Arc<Mutex<FakeBackend>> {
    let pieces = (0..100)
        .map(|i| {
            if i == 99 {
                String::new()
            } else {
                char::from_u32('a' as u32 + (i as u32 % 26))
                    .expect("a letter")
                    .to_string()
            }
        })
        .collect();
    Arc::new(Mutex::new(
        FakeBackend::new(4, 32, 128, 100, 99, pieces).with_row_adapters(),
    ))
}

fn scheduler(shared: &Arc<Mutex<FakeBackend>>) -> Scheduler<SharedFake> {
    let mut scheduler = Scheduler::new(
        SharedFake(shared.clone()),
        SchedulerOptions {
            mode: AdmissionMode::Continuous,
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

fn request(adapter: Option<&str>, seed: u32, max_new_tokens: u32) -> Request {
    // Eight prompt tokens so a finished sequence is long enough to stay
    // dormant in the prefix tree (min_prefix_len is 8).
    Request::single(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        fake_sampling(seed, max_new_tokens),
    )
    .with_adapter(adapter.map(str::to_owned))
}

fn load(scheduler: &mut Scheduler<SharedFake>, name: &str) {
    scheduler
        .load_adapter(name, &PathBuf::from(format!("{name}.gguf")))
        .expect("the fake loads the adapter");
}

fn id(raw: i32) -> AdapterId {
    AdapterId::new(raw).expect("a positive id")
}

fn drain(stream: EventStream) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = stream.recv() {
        events.push(event);
    }
    events
}

#[test]
fn per_row_adapters_and_the_base_model_share_steps() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    load(&mut scheduler, "b");
    let outcomes = run_requests(
        &mut scheduler,
        vec![
            request(Some("a"), 1, 4),
            request(Some("b"), 2, 4),
            request(None, 3, 4),
        ],
    )
    .unwrap();
    assert!(outcomes.iter().all(|outcome| outcome.error.is_none()));
    let fake = shared.lock().unwrap();
    assert!(
        !fake.adapter_ops.iter().any(|(op, _)| op == "activate"),
        "per-row adapters never switch the context's adapter: {:?}",
        fake.adapter_ops
    );
    let mixed = fake
        .step_adapters
        .iter()
        .find(|rows| {
            rows.contains(&RowAdapter::Loaded(id(1)))
                && rows.contains(&RowAdapter::Loaded(id(2)))
                && rows.contains(&RowAdapter::Context)
        })
        .is_some();
    assert!(
        mixed,
        "one step carried both adapters and the base model: {:?}",
        fake.step_adapters
    );
    assert_eq!(fake.weights_version, 0, "no adapter load moved the weights");
}

#[test]
fn a_prefix_is_reused_under_its_own_adapter_only() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    load(&mut scheduler, "b");
    for adapter in ["a", "b"] {
        let outcomes = run_requests(&mut scheduler, vec![request(Some(adapter), 1, 2)]).unwrap();
        assert!(outcomes[0].error.is_none());
    }
    // Same prompt under "b" after "a": a cold miss, the KV of "a" is not "b"'s.
    assert_eq!(scheduler.kv_stats().hits, 0);
    assert_eq!(scheduler.kv_stats().misses, 2);
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 3, 2)]).unwrap();
    assert!(outcomes[0].error.is_none());
    assert_eq!(scheduler.kv_stats().hits, 1, "a's own prefix is reused");
}

#[test]
fn a_reloaded_adapter_keeps_live_requests_on_the_old_generation() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    let live = scheduler.submit(request(Some("a"), 1, 6)).unwrap();
    scheduler.pump().unwrap();
    assert_eq!(scheduler.live_len(), 1);

    // The reload gets a fresh id; the old one stays loaded while it decodes.
    load(&mut scheduler, "a");
    assert_eq!(shared.lock().unwrap().row_adapters.len(), 2);
    let next = scheduler.submit(request(Some("a"), 2, 2)).unwrap();
    scheduler.run_until_idle().unwrap();
    assert!(drain(live).iter().any(|e| matches!(e, Event::End { .. })));
    assert!(drain(next).iter().any(|e| matches!(e, Event::End { .. })));

    let fake = shared.lock().unwrap();
    assert!(
        fake.row_adapters.keys().eq([id(2)].iter()),
        "the old generation was freed once its request ended: {:?}",
        fake.row_adapters
    );
    assert!(
        fake.adapter_ops
            .iter()
            .any(|(op, name)| op == "unload_row" && name.as_deref() == Some("1"))
    );
    // Every row ever decoded named the generation its request was admitted
    // under: the live request's rows id 1, the later one's id 2.
    let ids: Vec<&RowAdapter> = fake.step_adapters.iter().flatten().collect();
    assert!(ids.contains(&&RowAdapter::Loaded(id(1))));
    assert!(ids.contains(&&RowAdapter::Loaded(id(2))));
}

#[test]
fn an_unloaded_adapter_refuses_new_requests_and_frees_its_prefixes() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 1, 2)]).unwrap();
    assert!(outcomes[0].error.is_none());
    assert!(scheduler.kv_stats().resident_slots > 0);

    scheduler.unload_adapter("a").unwrap();
    assert_eq!(
        scheduler.kv_stats().resident_slots,
        0,
        "the dormant prefix of the unloaded adapter was released"
    );
    assert!(shared.lock().unwrap().row_adapters.is_empty());
    assert!(scheduler.stats().row_adapters.is_empty());

    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 2, 2)]).unwrap();
    assert!(
        outcomes[0].error.is_some(),
        "a request naming an unloaded adapter is refused"
    );
    assert!(scheduler.unload_adapter("a").is_err());
}

#[test]
fn a_backend_without_per_row_support_keeps_serializing() {
    let shared = Arc::new(Mutex::new(FakeBackend::new(
        2,
        32,
        64,
        100,
        99,
        vec!["x".into(); 100],
    )));
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    assert!(scheduler.stats().row_adapters.is_empty());
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 1, 2)]).unwrap();
    assert!(outcomes[0].error.is_none());
    let fake = shared.lock().unwrap();
    assert!(
        fake.adapter_ops
            .iter()
            .any(|(op, name)| op == "activate" && name.as_deref() == Some("a"))
    );
    assert!(
        fake.step_adapters
            .iter()
            .flatten()
            .all(|adapter| *adapter == RowAdapter::Context)
    );
}

#[test]
fn a_serialized_replacement_retires_the_previous_row_adapter() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    load(&mut scheduler, "a");
    let live = scheduler.submit(request(Some("a"), 1, 6)).unwrap();
    scheduler.pump().unwrap();

    // Model a replacement that can only use the serialized path.
    shared.lock().unwrap().row_adapters_enabled = false;
    load(&mut scheduler, "a");
    assert!(scheduler.stats().row_adapters.is_empty());
    assert_eq!(shared.lock().unwrap().row_adapters.len(), 1);
    let next = scheduler.submit(request(Some("a"), 2, 2)).unwrap();
    scheduler.run_until_idle().unwrap();
    for stream in [live, next] {
        let events = drain(stream);
        assert!(!events.iter().any(|event| matches!(event, Event::Error(_))));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Event::End { .. }))
        );
    }
    let fake = shared.lock().unwrap();
    assert!(fake.row_adapters.is_empty());
    assert_eq!(fake.active_adapter.as_deref(), Some("a"));
}

#[test]
fn unloading_a_row_adapter_never_reactivates_an_old_named_adapter() {
    let shared = fake();
    let mut scheduler = scheduler(&shared);
    shared.lock().unwrap().row_adapters_enabled = false;
    load(&mut scheduler, "a");
    shared.lock().unwrap().row_adapters_enabled = true;
    load(&mut scheduler, "a");
    scheduler.unload_adapter("a").unwrap();
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 1, 2)]).unwrap();
    assert!(outcomes[0].error.is_some());
    // The name can be explicitly registered again after unloading.
    load(&mut scheduler, "a");
    let outcomes = run_requests(&mut scheduler, vec![request(Some("a"), 1, 2)]).unwrap();
    assert!(outcomes[0].error.is_none());
}
