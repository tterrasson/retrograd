//! The synchronous driver: a `Scheduler` pumped on the backend's own thread
//! through `submit` / `pump` / `run_until_idle`, and `run_requests`, which
//! submits a batch and collects one structured outcome per request. Exercised
//! on `FakeBackend` - no model needed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use retrograd_engine::{StepOutput, StepRow};
use retrograd_infer::{
    AdmissionMode, DecodeBackend, FakeBackend, FinishReason, InferError, Request, Scheduler,
    SchedulerOptions, SharedFake, fake_sampling, fake_sampling_v2, run_requests,
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

fn scheduler(
    fake: FakeBackend,
    mode: AdmissionMode,
    queue_capacity: usize,
) -> Scheduler<FakeBackend> {
    Scheduler::new(
        fake,
        SchedulerOptions {
            mode,
            n_batch: 16,
            queue_capacity,
            event_capacity: 8,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        },
    )
}

fn fake() -> FakeBackend {
    FakeBackend::new(4, 32, 128, 100, 99, pieces(100))
}

#[test]
fn run_requests_collects_every_completion_in_submission_order() {
    let mut scheduler = scheduler(fake(), AdmissionMode::Lockstep, 16);
    scheduler.initialize().unwrap();
    let requests = vec![
        Request::single(vec![1, 2, 3], fake_sampling(7, 5)),
        Request {
            prompt: vec![4, 5],
            sampling: vec![
                fake_sampling_v2(1, 4),
                fake_sampling_v2(2, 4),
                fake_sampling_v2(3, 4),
            ],
            stop: Vec::new(),
            n_top_logprobs: 0,
            logprobs: false,
            adapter: None,
            unparse_special: false,
        },
        Request::single(vec![9, 8, 7, 6], fake_sampling(8, 6)),
    ];
    let outcomes = run_requests(&mut scheduler, requests).expect("the batch runs");
    assert_eq!(outcomes.len(), 3);
    assert_eq!(outcomes[0].completions.len(), 1);
    assert_eq!(outcomes[1].completions.len(), 3);
    assert_eq!(outcomes[2].completions.len(), 1);
    for outcome in &outcomes {
        assert!(
            outcome.error.is_none(),
            "unexpected error: {:?}",
            outcome.error
        );
        for member in &outcome.completions {
            assert_eq!(member.finish, Some(FinishReason::MaxTokens));
            assert_eq!(member.tokens.len(), member.logprobs.len());
            assert_eq!(member.tokens.len(), member.forced.len());
            assert!(member.forced.iter().all(|&forced| !forced));
        }
    }
    assert_eq!(outcomes[0].completions[0].tokens.len(), 5);
    assert_eq!(outcomes[1].completions[0].tokens.len(), 4);
    assert_eq!(outcomes[2].completions[0].tokens.len(), 6);
    // Distinct seeds give distinct streams inside one request.
    let streams: Vec<&[i32]> = outcomes[1]
        .completions
        .iter()
        .map(|member| member.tokens.as_slice())
        .collect();
    assert!(streams[0] != streams[1] && streams[1] != streams[2]);
    assert!(scheduler.is_idle());
}

#[test]
fn run_requests_is_deterministic_per_seed() {
    let run = || {
        let mut scheduler = scheduler(fake(), AdmissionMode::Lockstep, 16);
        scheduler.initialize().unwrap();
        let outcomes = run_requests(
            &mut scheduler,
            vec![Request::single(vec![1, 2, 3], fake_sampling(42, 8))],
        )
        .unwrap();
        outcomes[0].completions[0].tokens.clone()
    };
    assert_eq!(run(), run());
}

#[test]
fn run_requests_submits_beyond_the_queue_capacity_as_slots_free() {
    let mut scheduler = scheduler(fake(), AdmissionMode::Continuous, 1);
    scheduler.initialize().unwrap();
    let requests: Vec<Request> = (0..3)
        .map(|seed| Request::single(vec![1, 2], fake_sampling(seed, 4)))
        .collect();
    let outcomes = run_requests(&mut scheduler, requests).expect("the batch runs");
    assert_eq!(outcomes.len(), 3);
    for outcome in &outcomes {
        assert!(outcome.error.is_none());
        assert_eq!(outcome.completions[0].tokens.len(), 4);
    }
}

#[test]
fn an_impossible_reservation_comes_back_as_an_outcome_error() {
    let mut scheduler = scheduler(fake(), AdmissionMode::Lockstep, 16);
    scheduler.initialize().unwrap();
    // The per-sequence context is 32 positions: a 32-token prompt leaves no
    // room to generate, so admission refuses it with an error event.
    let requests = vec![Request::single(vec![7; 32], fake_sampling(1, 4))];
    let outcomes = run_requests(&mut scheduler, requests).expect("the call itself succeeds");
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].error.is_some(), "the refusal is reported");
    assert!(outcomes[0].completions[0].finish.is_none());
}

#[test]
fn an_eog_token_is_kept_and_ends_the_completion() {
    let mut scheduler = scheduler(fake().with_eog_after(3), AdmissionMode::Lockstep, 16);
    scheduler.initialize().unwrap();
    let outcomes = run_requests(
        &mut scheduler,
        vec![Request::single(vec![1, 2], fake_sampling(5, 10))],
    )
    .unwrap();
    let member = &outcomes[0].completions[0];
    assert_eq!(member.finish, Some(FinishReason::Eog));
    assert_eq!(member.tokens.len(), 4, "three sampled tokens plus the EOG");
    assert_eq!(member.tokens[3], 99);
}

#[test]
fn a_dropped_stream_is_removed_at_the_next_step() {
    let shared = Arc::new(Mutex::new(fake()));
    let mut scheduler = Scheduler::new(
        SharedFake(shared.clone()),
        SchedulerOptions {
            mode: AdmissionMode::Continuous,
            n_batch: 16,
            slow_client_timeout: Duration::ZERO,
            prefix_cache: false,
            ..SchedulerOptions::default()
        },
    );
    scheduler.initialize().unwrap();
    let stream = scheduler
        .submit(Request::single(vec![1, 2, 3], fake_sampling(1, 8)))
        .unwrap();
    scheduler.pump().unwrap();
    assert_eq!(scheduler.live_len(), 1);
    drop(stream);
    scheduler.run_until_idle().unwrap();
    assert!(scheduler.is_idle());
    let backend = shared.lock().unwrap();
    assert!(
        !backend.removed.is_empty(),
        "the cancelled sequence's slot was released: {:?}",
        backend.removed
    );
}

#[test]
fn initialize_clears_residue_left_by_earlier_calls() {
    let shared = Arc::new(Mutex::new(fake()));
    {
        let mut scheduler = Scheduler::new(
            SharedFake(shared.clone()),
            SchedulerOptions {
                mode: AdmissionMode::Lockstep,
                prefix_cache: false,
                ..SchedulerOptions::default()
            },
        );
        scheduler.initialize().unwrap();
        run_requests(
            &mut scheduler,
            vec![Request::single(vec![1, 2], fake_sampling(1, 4))],
        )
        .unwrap();
    }
    // A fresh scheduler on the same backend starts from empty slots, like the
    // batched generators that clear the generation context on entry.
    let mut scheduler = Scheduler::new(
        SharedFake(shared.clone()),
        SchedulerOptions {
            mode: AdmissionMode::Lockstep,
            prefix_cache: false,
            ..SchedulerOptions::default()
        },
    );
    scheduler.initialize().unwrap();
    assert_eq!(scheduler.live_len(), 0);
    let memory = shared.lock().unwrap().memory().unwrap();
    assert_eq!(memory.cells_used, 0);
}

#[test]
fn a_step_failure_is_returned_and_fails_the_live_requests() {
    // A backend whose step always fails, wrapped around the fake.
    struct AlwaysFail(FakeBackend);
    impl DecodeBackend for AlwaysFail {
        fn step(&mut self, _rows: &[StepRow<'_>], _out: &mut StepOutput) -> Result<(), InferError> {
            Err(InferError::Failed("boom".into()))
        }
        fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
            self.0.seq_rm(slot, p0, p1)
        }
        fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
            self.0.seq_cp(src, dst, p0, p1)
        }
        fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
            self.0.seq_pos(slot)
        }
        fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
            self.0.seq_save_len(slot, flags)
        }
        fn seq_save_into(
            &mut self,
            slot: i32,
            flags: u32,
            buffer: &mut [u8],
        ) -> Result<usize, InferError> {
            self.0.seq_save_into(slot, flags, buffer)
        }
        fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
            self.0.seq_load(slot, flags, data)
        }
        fn kv_unified(&self) -> Result<bool, InferError> {
            self.0.kv_unified()
        }
        fn weights_version(&self) -> Result<u64, InferError> {
            self.0.weights_version()
        }
        fn model_fingerprint(&self) -> Result<u64, InferError> {
            self.0.model_fingerprint()
        }
        fn is_hybrid(&self) -> Result<bool, InferError> {
            self.0.is_hybrid()
        }
        fn memory(&self) -> Result<retrograd_engine::EngineMemory, InferError> {
            self.0.memory()
        }
        fn request_free(&mut self, request: i32) -> Result<(), InferError> {
            self.0.request_free(request)
        }
        fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
            self.0.is_eog_token(token)
        }
        fn detokenize_bytes(
            &self,
            tokens: &[i32],
            unparse_special: bool,
        ) -> Result<Vec<u8>, InferError> {
            self.0.detokenize_bytes(tokens, unparse_special)
        }
        fn context_size(&self) -> Result<u32, InferError> {
            self.0.context_size()
        }
        fn vocab_size(&self) -> Result<u32, InferError> {
            self.0.vocab_size()
        }
    }
    let mut scheduler = Scheduler::new(
        AlwaysFail(fake()),
        SchedulerOptions {
            mode: AdmissionMode::Lockstep,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        },
    );
    scheduler.initialize().unwrap();
    let stream = scheduler
        .submit(Request::single(vec![1, 2], fake_sampling(1, 4)))
        .unwrap();
    let error = scheduler.pump().expect_err("the step fails");
    assert!(matches!(error, InferError::Failed(_)), "{error:?}");
    assert!(scheduler.is_idle(), "the failed request was released");
    // The error reached the request's stream before it closed.
    let mut saw_error = false;
    while let Some(event) = stream.recv() {
        saw_error |= matches!(event, retrograd_infer::Event::Error(_));
    }
    assert!(saw_error);
}

#[test]
fn a_lent_slot_is_empty_and_leaves_the_prefix_tree_resident() {
    let shared = Arc::new(Mutex::new(fake()));
    let mut scheduler = Scheduler::new(
        SharedFake(shared.clone()),
        SchedulerOptions {
            n_batch: 16,
            queue_capacity: 16,
            event_capacity: 8,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        },
    );
    scheduler.initialize().unwrap();
    // Eight prompt tokens: a finished sequence is then long enough to stay
    // dormant (min_prefix_len is 8) - the resident span is the prompt plus the
    // fed-back tokens, so the prompt must clear the threshold on its own.
    let requests = vec![
        Request::single(vec![1, 2, 3, 4, 5, 6, 7, 8], fake_sampling(3, 4)),
        Request::single(vec![9, 10, 11, 12, 13, 14, 15, 16], fake_sampling(4, 4)),
    ];
    run_requests(&mut scheduler, requests).unwrap();
    let resident = scheduler.kv_stats().resident_slots;
    assert!(resident > 0, "finished sequences stay dormant");

    // Room beside the dormant prefixes: no eviction, an empty slot.
    let slot = scheduler.lend_slot(8).unwrap().expect("a free slot");
    assert_eq!(shared.lock().unwrap().seq_pos(slot).unwrap(), (-1, -1));
    let stats = scheduler.kv_stats();
    assert_eq!((stats.resident_slots, stats.evictions), (resident, 0));

    // The whole pool: every dormant prefix has to make room.
    let slot = scheduler
        .lend_slot(128)
        .unwrap()
        .expect("evicted into room");
    assert_eq!(shared.lock().unwrap().seq_pos(slot).unwrap(), (-1, -1));
    let stats = scheduler.kv_stats();
    assert_eq!(stats.resident_slots, 0);
    assert_eq!(stats.evictions, resident as u64);

    // More cells than the pool has: nothing left to evict.
    assert_eq!(scheduler.lend_slot(129).unwrap(), None);
}

#[test]
fn a_slot_is_not_lent_while_a_sequence_is_live() {
    let mut scheduler = scheduler(fake(), AdmissionMode::Continuous, 16);
    scheduler.initialize().unwrap();
    let _stream = scheduler
        .submit(Request::single(vec![1, 2, 3], fake_sampling(5, 8)))
        .unwrap();
    scheduler.pump().unwrap();
    assert!(scheduler.live_len() > 0);
    assert!(scheduler.lend_slot(1).is_err());
}
