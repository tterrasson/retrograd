//! Driving generation waves through a persistent step-wise scheduler.
//!
//! The policy actor does not call the batched `generate_sampled_continuous`
//! per turn any more: that runtime clears the generation context on entry and
//! on exit, so every turn of every trajectory re-prefilled its whole history.
//! Here a [`Scheduler`] owns the model's generation pool for the whole
//! rollout batch: each turn submits one request per live trajectory, and the
//! scheduler's prefix tree keeps finished sequences' KV as dormant slots, so
//! a trajectory's next turn hits its own prefix and `seq_cp`s it instead of
//! re-prefilling. Prefix reuse is compute-path-only, so outputs stay
//! bit-identical; in [`AdmissionMode::Lockstep`] a wave is admitted exactly the
//! way the batched continuous generator composes it, which is what keeps that
//! parity (see `tests/infer_runtime.rs`).
//!
//! The tree drops by itself when the weights version moves - an optimizer
//! step, an adapter load, any runtime-side KV invalidation bumps it - so no
//! prefix computed under superseded weights is ever reused.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

use retrograd_core::{SamplingParams, TrainConfig};
use retrograd_engine::{Engine, SampledRow, StepOutput, StepRow, Trainer, generation_batch};
use retrograd_infer::{
    Completion, DecodeBackend, InferError, KvStats, Request, Scheduler, SchedulerOptions,
    run_requests_with,
};

use crate::{Error, Result};

/// Maps an inference error into the agent facade through the core facade, so
/// the variant (`InvalidArgument`, capacity, ...) survives the crossing.
pub(crate) fn infer_error(error: InferError) -> Error {
    Error::Core(error.into())
}

/// A `Trainer` shared between the actor's direct calls and the scheduler's
/// backend, on the one thread that owns the trainer. The cell is `None` only
/// while the trainer is lent out at an update boundary, and no scheduler call
/// happens then: the actor awaits the borrower before serving anything else.
pub(crate) struct SharedTrainer(pub(crate) Rc<RefCell<Option<Trainer>>>);

impl SharedTrainer {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut Trainer) -> retrograd_core::Result<T>,
    ) -> std::result::Result<T, InferError> {
        let mut cell = self.0.borrow_mut();
        let trainer = cell.as_mut().ok_or_else(|| {
            InferError::Failed("the trainer is lent out at an update boundary".into())
        })?;
        Ok(f(trainer)?)
    }

    fn with_ref<T>(
        &self,
        f: impl FnOnce(&Trainer) -> retrograd_core::Result<T>,
    ) -> std::result::Result<T, InferError> {
        let cell = self.0.borrow();
        let trainer = cell.as_ref().ok_or_else(|| {
            InferError::Failed("the trainer is lent out at an update boundary".into())
        })?;
        Ok(f(trainer)?)
    }
}

impl DecodeBackend for SharedTrainer {
    fn step(
        &mut self,
        rows: &[StepRow<'_>],
        out: &mut StepOutput,
    ) -> std::result::Result<(), InferError> {
        self.with(|t| Trainer::step(t, rows, out))
    }
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> std::result::Result<(), InferError> {
        self.with(|t| Trainer::seq_rm(t, slot, p0, p1))
    }
    fn seq_cp(
        &mut self,
        src: i32,
        dst: i32,
        p0: i32,
        p1: i32,
    ) -> std::result::Result<(), InferError> {
        self.with(|t| Trainer::seq_cp(t, src, dst, p0, p1))
    }
    fn seq_pos(&self, slot: i32) -> std::result::Result<(i32, i32), InferError> {
        self.with_ref(|t| Trainer::seq_pos(t, slot))
    }
    fn seq_save_len(&self, slot: i32, flags: u32) -> std::result::Result<usize, InferError> {
        self.with_ref(|t| Trainer::seq_save_len(t, slot, flags))
    }
    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> std::result::Result<usize, InferError> {
        self.with(|t| Trainer::seq_save_into(t, slot, flags, buffer))
    }
    fn seq_load(
        &mut self,
        slot: i32,
        flags: u32,
        data: &[u8],
    ) -> std::result::Result<usize, InferError> {
        self.with(|t| Trainer::seq_load(t, slot, flags, data))
    }
    fn kv_unified(&self) -> std::result::Result<bool, InferError> {
        self.with_ref(Trainer::generation_kv_unified)
    }
    fn weights_version(&self) -> std::result::Result<u64, InferError> {
        self.with_ref(Trainer::generation_weights_version)
    }
    fn model_fingerprint(&self) -> std::result::Result<u64, InferError> {
        self.with_ref(Trainer::model_fingerprint)
    }
    fn is_hybrid(&self) -> std::result::Result<bool, InferError> {
        self.with_ref(Trainer::is_hybrid)
    }
    fn memory(&self) -> std::result::Result<retrograd_engine::EngineMemory, InferError> {
        self.with_ref(Trainer::memory)
    }
    fn request_free(&mut self, request: i32) -> std::result::Result<(), InferError> {
        self.with(|t| Trainer::request_free(t, request))
    }
    fn is_eog_token(&self, token: i32) -> std::result::Result<bool, InferError> {
        self.with_ref(|t| Trainer::is_eog_token(t, token))
    }
    fn detokenize_bytes(&self, tokens: &[i32]) -> std::result::Result<Vec<u8>, InferError> {
        self.with_ref(|t| Trainer::detokenize_bytes(t, tokens, false))
    }
    fn context_size(&self) -> std::result::Result<u32, InferError> {
        // Both native trainer contexts use a unified pool, so the per-sequence
        // limit equals the full pool size (the training-context getter would
        // report the other context's window).
        let memory = self.with_ref(Trainer::memory)?;
        u32::try_from(memory.cells_total).map_err(|_| {
            InferError::Overflow("trainer generation context size does not fit u32".into())
        })
    }
    fn vocab_size(&self) -> std::result::Result<u32, InferError> {
        let size = self.with_ref(Trainer::vocab_size)?;
        u32::try_from(size)
            .map_err(|_| InferError::Overflow("trainer vocab size does not fit u32".into()))
    }
}

/// An `Engine` shared the same way, for the engine-backed policy. Unlike the
/// trainer, nothing in the engine runtime bumps its weights version when host
/// code reclaims a slot directly (the step-wise scorer does exactly that), so
/// the wrapper carries its own invalidation counter: the actor bumps it after
/// every scoring pass, and the scheduler drops its prefix tree on the next
/// admission, exactly as it does for a runtime-side version bump.
pub(crate) struct SharedEngine {
    pub(crate) engine: Rc<RefCell<Option<Engine>>>,
    pub(crate) invalidations: Rc<Cell<u64>>,
}

impl SharedEngine {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut Engine) -> std::result::Result<T, retrograd_engine::EngineError>,
    ) -> std::result::Result<T, InferError> {
        let mut cell = self.engine.borrow_mut();
        let engine = cell
            .as_mut()
            .ok_or_else(|| InferError::Failed("the engine is not owned by the actor".into()))?;
        Ok(f(engine)?)
    }

    fn with_ref<T>(
        &self,
        f: impl FnOnce(&Engine) -> std::result::Result<T, retrograd_engine::EngineError>,
    ) -> std::result::Result<T, InferError> {
        let cell = self.engine.borrow();
        let engine = cell
            .as_ref()
            .ok_or_else(|| InferError::Failed("the engine is not owned by the actor".into()))?;
        Ok(f(engine)?)
    }
}

impl DecodeBackend for SharedEngine {
    fn step(
        &mut self,
        rows: &[StepRow<'_>],
        out: &mut StepOutput,
    ) -> std::result::Result<(), InferError> {
        self.with(|e| Engine::step(e, rows, out))
    }
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> std::result::Result<(), InferError> {
        self.with(|e| Engine::seq_rm(e, slot, p0, p1))
    }
    fn seq_cp(
        &mut self,
        src: i32,
        dst: i32,
        p0: i32,
        p1: i32,
    ) -> std::result::Result<(), InferError> {
        self.with(|e| Engine::seq_cp(e, src, dst, p0, p1))
    }
    fn seq_pos(&self, slot: i32) -> std::result::Result<(i32, i32), InferError> {
        self.with_ref(|e| Engine::seq_pos(e, slot))
    }
    fn seq_save_len(&self, slot: i32, flags: u32) -> std::result::Result<usize, InferError> {
        self.with_ref(|e| Engine::seq_save_len(e, slot, flags))
    }
    fn seq_save_into(
        &mut self,
        slot: i32,
        flags: u32,
        buffer: &mut [u8],
    ) -> std::result::Result<usize, InferError> {
        self.with(|e| Engine::seq_save_into(e, slot, flags, buffer))
    }
    fn seq_load(
        &mut self,
        slot: i32,
        flags: u32,
        data: &[u8],
    ) -> std::result::Result<usize, InferError> {
        self.with(|e| Engine::seq_load(e, slot, flags, data))
    }
    fn kv_unified(&self) -> std::result::Result<bool, InferError> {
        self.with_ref(Engine::kv_unified)
    }
    fn weights_version(&self) -> std::result::Result<u64, InferError> {
        let version = self.with_ref(Engine::weights_version)?;
        Ok(version.wrapping_add(self.invalidations.get()))
    }
    fn model_fingerprint(&self) -> std::result::Result<u64, InferError> {
        self.with_ref(Engine::model_fingerprint)
    }
    fn is_hybrid(&self) -> std::result::Result<bool, InferError> {
        self.with_ref(Engine::is_hybrid)
    }
    fn memory(&self) -> std::result::Result<retrograd_engine::EngineMemory, InferError> {
        self.with_ref(Engine::memory)
    }
    fn request_free(&mut self, request: i32) -> std::result::Result<(), InferError> {
        self.with(|e| Engine::request_free(e, request))
    }
    fn is_eog_token(&self, token: i32) -> std::result::Result<bool, InferError> {
        self.with_ref(|e| Engine::is_eog_token(e, token))
    }
    fn detokenize_bytes(&self, tokens: &[i32]) -> std::result::Result<Vec<u8>, InferError> {
        self.with_ref(|e| Engine::detokenize_bytes(e, tokens, false))
    }
    fn context_size(&self) -> std::result::Result<u32, InferError> {
        self.with_ref(Engine::context_size)
    }
    fn vocab_size(&self) -> std::result::Result<u32, InferError> {
        self.with_ref(Engine::vocab_size)
    }
}

/// The scheduler's knobs for a rollout run, derived from the run's
/// [`TrainConfig`]: the admission mode is `training.rollout_admission`, and
/// the step token budget is the generation context's own `n_batch`
/// ([`generation_batch`]).
pub(crate) fn scheduler_options(training: &TrainConfig, vocab: u32) -> SchedulerOptions {
    // u32 -> usize widenings.
    let sequence_capacity = training.effective_generation_concurrency().max(1) as usize;
    SchedulerOptions {
        mode: training.rollout_admission.into(),
        n_batch: generation_batch(training, vocab) as usize,
        // A coalesced wave never exceeds the sequence capacity, so the queue
        // always has room for a whole wave and submission never truncates it.
        queue_capacity: sequence_capacity.max(64),
        ..SchedulerOptions::default()
    }
}

/// What one drain of the driver's rollout accumulators reports. `kv` is the
/// scheduler's own cumulative prefix-cache counters; the rest are drained
/// (reset) so a caller differences two snapshots around one update.
#[derive(Clone, Debug, Default)]
pub struct RolloutStatsSnapshot {
    pub steps: u64,
    /// Sum of `live_len` over the drained steps.
    pub occupancy_sum: u64,
    pub n_slots: usize,
    pub tool_wait_seconds: f64,
    pub tool_wait_covered_seconds: f64,
    pub kv: KvStats,
}

#[derive(Default)]
struct RolloutAcc {
    steps: u64,
    occupancy_sum: u64,
    /// Trajectories currently waiting for a tool/environment.
    tool_waiting: u64,
    tool_wait_since: Option<Instant>,
    tool_wait_seconds: f64,
    tool_wait_covered_seconds: f64,
}

impl RolloutAcc {
    /// Closes the open tool-wait interval at `now`, counting it as covered
    /// when `live` sequences were decoding, and reopens one if anything is
    /// still waiting.
    fn accrue(&mut self, now: Instant, live: usize) {
        if let Some(since) = self.tool_wait_since.take()
            && self.tool_waiting > 0
        {
            let elapsed = now.saturating_duration_since(since).as_secs_f64();
            self.tool_wait_seconds += elapsed;
            if live > 0 {
                self.tool_wait_covered_seconds += elapsed;
            }
        }
        self.tool_wait_since = (self.tool_waiting > 0).then_some(now);
    }
}

/// A persistent scheduler over one decode backend, plus the rollout
/// accumulators the update metrics read.
pub(crate) struct SchedulerDriver<B: DecodeBackend> {
    scheduler: Scheduler<B>,
    n_slots: usize,
    acc: RolloutAcc,
}

impl<B: DecodeBackend> SchedulerDriver<B> {
    /// Builds the scheduler and clears the pool once, so residue from earlier
    /// batched calls cannot eat the budget.
    pub(crate) fn new(
        backend: B,
        options: SchedulerOptions,
    ) -> std::result::Result<Self, InferError> {
        let n_slots = backend.memory()?.n_slots as usize;
        let mut scheduler = Scheduler::new(backend, options);
        scheduler.initialize()?;
        Ok(Self {
            scheduler,
            n_slots: n_slots.max(1),
            acc: RolloutAcc::default(),
        })
    }

    /// Records that one more (or one fewer) trajectory is waiting on a
    /// tool/environment. Wall-clock accrual happens at pump and drain
    /// granularity, so the fractions this feeds are approximations of the
    /// true overlap, not exact timelines.
    pub(crate) fn note_tool_wait(&mut self, waiting: bool) {
        // Close the interval under the old count first, then re-open it if
        // anything is still waiting.
        let live = self.scheduler.live_len();
        self.acc.accrue(Instant::now(), live);
        if waiting {
            self.acc.tool_waiting += 1;
        } else {
            self.acc.tool_waiting = self.acc.tool_waiting.saturating_sub(1);
        }
        self.acc.accrue(Instant::now(), live);
    }

    /// Submits one request per sequence, in order, and pumps until every one
    /// has ended, returning one sampled row per sequence in submission order.
    /// Requests beyond the queue capacity are submitted as earlier ones are
    /// admitted, so the queue bound never truncates a wave.
    pub(crate) fn run_wave(
        &mut self,
        sequences: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<SampledRow>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let requests = sequences
            .into_iter()
            .map(|(prompt, sampling)| Request::single(prompt, sampling))
            .collect();
        let acc = &mut self.acc;
        let outcomes = run_requests_with(&mut self.scheduler, requests, |scheduler| {
            let live = scheduler.live_len();
            acc.steps += 1;
            // usize -> u64 widening.
            acc.occupancy_sum += live as u64;
            acc.accrue(Instant::now(), live);
        })
        .map_err(infer_error)?;
        outcomes
            .into_iter()
            .map(|outcome| {
                if let Some(error) = outcome.error {
                    return Err(infer_error(error));
                }
                // The end reason needs no mapping here: an EOG token is kept
                // in `tokens` and detected by `finish_generation` exactly as
                // the batched generators report it, and a budget/context stop
                // leaves `tokens.len()` at the turn budget, which is what the
                // rollout's truncation check reads.
                match <[Completion; 1]>::try_from(outcome.completions) {
                    Ok([completion]) => Ok(SampledRow {
                        tokens: completion.tokens,
                        forced: completion.forced,
                    }),
                    Err(_) => Err(Error::PolicyGeneration(
                        "a single-completion request returned no single completion".into(),
                    )),
                }
            })
            .collect()
    }

    /// Drains the per-update accumulators and reads the cumulative prefix-cache
    /// counters.
    pub(crate) fn drain_stats(&mut self) -> RolloutStatsSnapshot {
        self.acc.accrue(Instant::now(), self.scheduler.live_len());
        let acc = std::mem::take(&mut self.acc);
        RolloutStatsSnapshot {
            steps: acc.steps,
            occupancy_sum: acc.occupancy_sum,
            n_slots: self.n_slots,
            tool_wait_seconds: acc.tool_wait_seconds,
            tool_wait_covered_seconds: acc.tool_wait_covered_seconds,
            kv: self.scheduler.kv_stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_core::{RolloutAdmission, TrainConfig};
    use retrograd_infer::{AdmissionMode, FakeBackend, SharedFake, fake_sampling};

    fn fake() -> FakeBackend {
        FakeBackend::new(
            4,
            64,
            4 * 64,
            32,
            31,
            (0..32).map(|i| format!("t{i}")).collect(),
        )
    }

    fn driver(backend: FakeBackend) -> SchedulerDriver<FakeBackend> {
        SchedulerDriver::new(backend, SchedulerOptions::default()).unwrap()
    }

    #[test]
    fn a_wave_returns_one_row_per_sequence_in_submission_order() {
        let mut driver = driver(fake());
        let rows = driver
            .run_wave(vec![
                (vec![1, 2, 3], fake_sampling(7, 4)),
                (vec![1, 2, 4], fake_sampling(8, 4)),
            ])
            .unwrap();
        assert_eq!(rows.len(), 2);
        // The fake's sampler is a pure function of the request id and the
        // draw count, so each row carries its own request's tokens.
        assert!(!rows[0].tokens.is_empty());
        assert_eq!(rows[0].tokens.len(), rows[1].tokens.len());
    }

    #[test]
    fn a_wave_beyond_the_slot_count_still_completes() {
        // Six sequences over four slots: the queue refills as slots free.
        let mut driver = driver(fake());
        let sequences = (0..6_i32)
            .map(|i| (vec![i + 1, 2], fake_sampling(i as u32 + 1, 2)))
            .collect::<Vec<_>>();
        let rows = driver.run_wave(sequences).unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn a_failed_request_comes_back_as_an_error_not_a_short_wave() {
        let mut driver = driver(fake());
        // An empty prompt is refused at submit.
        let error = driver
            .run_wave(vec![(vec![], fake_sampling(1, 2))])
            .unwrap_err();
        assert!(error.to_string().contains("prompt"), "{error}");
    }

    #[test]
    fn tool_wait_accrues_only_while_waiting_and_counts_coverage_by_liveness() {
        let mut driver = driver(fake());
        driver.note_tool_wait(true);
        std::thread::sleep(std::time::Duration::from_millis(5));
        // No wave is running, so nothing is decoding: wait accrues, coverage
        // does not.
        driver.note_tool_wait(false);
        let stats = driver.drain_stats();
        assert!(stats.tool_wait_seconds > 0.0, "{stats:?}");
        assert_eq!(stats.tool_wait_covered_seconds, 0.0, "{stats:?}");
        // A second drain of an idle driver reports nothing new.
        let again = driver.drain_stats();
        assert_eq!(again.tool_wait_seconds, 0.0);
        assert_eq!(again.steps, 0);
    }

    #[test]
    fn occupancy_and_steps_are_drained_per_wave() {
        let mut driver = driver(fake());
        driver
            .run_wave(vec![(vec![1, 2], fake_sampling(3, 3))])
            .unwrap();
        let stats = driver.drain_stats();
        assert!(stats.steps > 0);
        assert!(stats.occupancy_sum > 0);
        assert!(stats.n_slots == 4);
        let again = driver.drain_stats();
        assert_eq!(again.steps, 0);
        assert_eq!(again.occupancy_sum, 0);
        // kv counters are cumulative, not drained.
        assert_eq!(again.kv.prompt_cells, stats.kv.prompt_cells);
    }

    #[test]
    fn scheduler_options_follow_the_admission_mode_and_the_generation_batch() {
        let base = TrainConfig {
            n_ctx: 1024,
            generation_batch: 0,
            rollout_admission: RolloutAdmission::Continuous,
            ..TrainConfig::default()
        };
        let options = scheduler_options(&base, 1000);
        assert_eq!(options.mode, AdmissionMode::Continuous);
        // Default batch: min(n_ctx, 512, affordable) then at least the
        // concurrency (the default resolves to 4). With n_ctx = 1024 and a
        // 1000-token vocabulary, the 512-token launch is the binding bound.
        let affordable = ((64_u64 << 20) / (1000 * 4)) as u32;
        assert!(affordable > 512, "the test needs the 512 cap to bind");
        assert_eq!(options.n_batch, 512_u32.min(affordable).max(4) as usize);
        assert!(options.queue_capacity >= 64);

        let explicit = TrainConfig {
            generation_batch: 77,
            rollout_admission: RolloutAdmission::Lockstep,
            ..base
        };
        let options = scheduler_options(&explicit, 1000);
        assert_eq!(options.mode, AdmissionMode::Lockstep);
        assert_eq!(options.n_batch, 77);
    }

    #[test]
    fn a_shared_fake_backend_drives_a_wave_through_the_driver() {
        // The same shape the actors use: the backend behind a shared handle,
        // so the actor can keep using the model between waves.
        let arc = std::sync::Arc::new(std::sync::Mutex::new(fake()));
        let mut driver =
            SchedulerDriver::new(SharedFake(arc.clone()), SchedulerOptions::default()).unwrap();
        let rows = driver
            .run_wave(vec![(vec![5, 6], fake_sampling(9, 2))])
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!arc.lock().expect("not poisoned").steps.is_empty());
    }
}
