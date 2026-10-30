//! Driving rollout generation through the step-wise scheduler
//! ([`retrograd_infer::Scheduler`]) instead of the batched FFI generators.
//!
//! The driver is generic over any [`DecodeBackend`] - a `&mut Trainer` through
//! [`TrainerBackend`] in production, a
//! [`FakeBackend`](retrograd_infer::FakeBackend) in tests - and works on the
//! backend's own thread: the scheduler is created, `initialize`d, pumped and
//! dropped inside one sampling call, so the trainer's mutable borrow ends
//! before the caller detokenizes or scores again.
//!
//! Two admission shapes, both fed from the same `(prompt, Vec<SamplingParams>)`
//! groups:
//!
//! - `Lockstep` keeps the wave splitting of the batched continuous generator
//!   ([`continuous_waves`]): one physical wave is submitted, run to idle, and
//!   only then the next wave. Each wave member is its own single-completion
//!   request, so the scheduler's canonical sort + identical-prompt merge
//!   reproduces the admitted set of one `generate_tokens_continuous` call bit
//!   for bit.
//! - `Continuous` submits every group of the update at once - chunking only
//!   groups wider than the slot count, the way [`continuous_waves`] does - and
//!   lets the scheduler refill freed slots.
//!
//! In both modes `max_new_tokens` is clamped to the generation room of the
//! trained window exactly as the batched callers clamped it, and a forced token
//! anywhere in the batch fails the call with the same message `refuse_forced`
//! produced.

use std::time::Duration;

use retrograd_core::{
    Error, Result, RolloutAdmission, SamplingParams, SamplingParamsV2, TrainConfig,
};
use retrograd_engine::Trainer;
use retrograd_infer::{
    DecodeBackend, KvStats, Request, RequestOutcome, Scheduler, SchedulerOptions, TrainerBackend,
    run_requests_with,
};

use super::sampling::{RowLayout, continuous_waves, generation_room};

/// The refusal the batched generators raise when a reasoning budget forced
/// tokens into a generation whose caller cannot mask them. Same text here, so
/// a run under a reasoning budget fails identically whichever path sampled it.
/// Single-turn rollout sampling never sets a budget, so this never fires in
/// practice - it keeps the old semantics.
const FORCED_REFUSAL: &str = "the reasoning budget forced tokens into a generation whose \
     caller cannot mask them; use generate_sampled_continuous";

/// One sampled group member: the completion tokens (an EOG token that ended
/// the completion is kept) and the per-token forced mask.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RolloutCompletion {
    pub tokens: Vec<i32>,
    pub forced: Vec<bool>,
}

/// What the driver observed while one scheduler ran: per-step live-sequence
/// samples and the prefix-cache cell counters, accumulated across every
/// sampling call of an update.
#[derive(Clone, Copy, Debug, Default)]
pub struct RolloutStats {
    /// Scheduler steps pumped.
    pub steps: u64,
    /// Sum of `live_len()` sampled after each pump.
    pub live_sum: u64,
    /// Prompt cells the admitted groups asked for (full prompt lengths).
    pub prompt_cells: u64,
    /// Prompt cells actually prefilled (the cache served the rest).
    pub prefilled_cells: u64,
}

impl RolloutStats {
    fn record_step(&mut self, live: usize) {
        self.steps += 1;
        // A budget-style saturating add: live_len is bounded by the slot
        // count, but a corrupt report must not wrap the accumulator.
        self.live_sum = self.live_sum.saturating_add(live as u64);
    }

    fn absorb_kv_delta(&mut self, before: &KvStats, after: &KvStats) {
        self.prompt_cells = self
            .prompt_cells
            .saturating_add(after.prompt_cells.saturating_sub(before.prompt_cells));
        self.prefilled_cells = self
            .prefilled_cells
            .saturating_add(after.prefilled_cells.saturating_sub(before.prefilled_cells));
    }

    /// Mean live sequences per scheduler step, as a share of the pool's slot
    /// count. Both operands are counts, so the f64 intermediate keeps the
    /// ratio exact enough for an f32 series; the quotient is in [0, 1] by
    /// construction (live_len never exceeds the slot count).
    pub fn slot_occupancy_mean(&self, slots: usize) -> f32 {
        if self.steps == 0 || slots == 0 {
            return 0.0;
        }
        let mean_live = self.live_sum as f64 / self.steps as f64;
        (mean_live / slots as f64) as f32
    }

    /// `1 - prefilled/prompt` over the update's admitted groups: the share of
    /// prompt cells the prefix cache served instead of re-prefilling. Zero
    /// when no prompt cell was submitted, which is the honest reading of
    /// "nothing was sampled" on a series that is otherwise a share.
    pub fn prefill_reuse_fraction(&self) -> f32 {
        if self.prompt_cells == 0 {
            return 0.0;
        }
        // prefilled <= prompt cell-count-wise, so the ratio is in [0, 1] and
        // the f64 -> f32 narrowing cannot overflow.
        (1.0 - self.prefilled_cells as f64 / self.prompt_cells as f64) as f32
    }
}

/// The scheduler options a rollout driver runs with: the admission mode and
/// the generation-context batch budget from the config, a queue large enough
/// for one call's groups (never below one wave), the prefix cache on, and a
/// short backpressure timeout - the driver drains every stream after each
/// pump, so a full stream means a bug, not a slow client.
pub(crate) fn scheduler_options(
    admission: RolloutAdmission,
    n_batch: usize,
    slots: usize,
    queue_capacity: usize,
) -> SchedulerOptions {
    SchedulerOptions {
        mode: admission.into(),
        n_batch: n_batch.max(1),
        queue_capacity: queue_capacity.max(slots).max(1),
        event_capacity: 64,
        slow_client_timeout: Duration::from_millis(100),
        prefix_cache: true,
        ..SchedulerOptions::default()
    }
}

/// Builds and initializes the rollout scheduler over the trainer's generation
/// context. `initialize` clears every pool slot once, matching the batched
/// generators that clear the generation context on entry; it also makes the
/// prefix tree drop on the next admission if the weights moved since.
pub(crate) fn new_scheduler<'a>(
    trainer: &'a mut Trainer,
    layout: &RowLayout,
    queue_capacity: usize,
) -> Result<Scheduler<TrainerBackend<'a>>> {
    let options = scheduler_options(
        layout.rollout_admission,
        layout.generation_batch,
        layout.generation_slots,
        queue_capacity,
    );
    let mut scheduler = Scheduler::new(TrainerBackend(trainer), options);
    scheduler.initialize()?;
    Ok(scheduler)
}

/// Runs one batch of `(prompt, params)` groups through `scheduler` and returns
/// the per-group completions in the original group/member order.
///
/// `scheduler` must have been initialized (see [`new_scheduler`]). Budgets are
/// clamped to the trained window's generation room per prompt, exactly as the
/// batched callers did. A request the pool can never admit comes back as an
/// error of the call, not a hang; a forced token anywhere in the batch fails
/// the call with the batched generators' `refuse_forced` message.
pub(crate) fn run_rollout_groups<B: DecodeBackend>(
    scheduler: &mut Scheduler<B>,
    admission: RolloutAdmission,
    layout: &RowLayout,
    sequence_capacity: usize,
    groups: &[(&[i32], Vec<SamplingParams>)],
    stats: &mut RolloutStats,
) -> Result<Vec<Vec<RolloutCompletion>>> {
    if groups.is_empty() || sequence_capacity == 0 {
        return Err(Error::invalid(
            "continuous rollout sampling requires capacity and groups",
        ));
    }
    if groups.iter().any(|(_, params)| params.is_empty()) {
        return Err(Error::invalid(
            "continuous rollout groups must not be empty",
        ));
    }
    // Clamp every member's budget to the room left beside its prompt in the
    // trained window - the same clamp the batched callers applied before
    // calling the generators, so budgets are unchanged.
    let mut clamped: Vec<(Vec<i32>, Vec<SamplingParams>)> = Vec::with_capacity(groups.len());
    for (prompt, params) in groups {
        let room = generation_room(layout, prompt.len())?;
        clamped.push((
            (*prompt).to_vec(),
            params
                .iter()
                .map(|params| SamplingParams {
                    max_new_tokens: params.max_new_tokens.min(room),
                    ..*params
                })
                .collect(),
        ));
    }
    let group_sizes: Vec<usize> = clamped.iter().map(|(_, params)| params.len()).collect();
    let mut output: Vec<Vec<RolloutCompletion>> = clamped
        .iter()
        .map(|(_, params)| vec![RolloutCompletion::default(); params.len()])
        .collect();
    let kv_before = scheduler.kv_stats();
    match admission {
        RolloutAdmission::Lockstep => {
            // One physical wave at a time, submitted whole and run to idle:
            // the admitted set is then exactly the one per-wave
            // generate_tokens_continuous call produced.
            for wave in continuous_waves(&group_sizes, sequence_capacity) {
                if wave.len() > scheduler.queue_capacity() {
                    return Err(Error::invalid(format!(
                        "a lockstep wave of {} requests exceeds the scheduler queue capacity {}",
                        wave.len(),
                        scheduler.queue_capacity()
                    )));
                }
                let requests: Vec<Request> = wave
                    .iter()
                    .map(|&(group, member)| {
                        Request::single(clamped[group].0.clone(), clamped[group].1[member])
                    })
                    .collect();
                let outcomes = run_wave(scheduler, requests, stats)?;
                refuse_forced(&outcomes)?;
                for (&(group, member), outcome) in wave.iter().zip(&outcomes) {
                    output[group][member] = take_single(outcome)?;
                }
            }
        }
        RolloutAdmission::Continuous => {
            // All groups at once; only a group wider than the slot count is
            // chunked (at the slot count, the way continuous_waves splits
            // oversized groups). Freed slots are refilled by the scheduler.
            let mut plan: Vec<(usize, usize, Request)> = Vec::new();
            for (group, (prompt, params)) in clamped.iter().enumerate() {
                let mut base = 0;
                while base < params.len() {
                    let take = (params.len() - base).min(sequence_capacity);
                    plan.push((
                        group,
                        base,
                        Request {
                            prompt: prompt.clone(),
                            sampling: params[base..base + take]
                                .iter()
                                .map(|params| SamplingParamsV2::from(*params))
                                .collect(),
                            stop: Vec::new(),
                            n_top_logprobs: 0,
                            logprobs: false,
                            adapter: None,
                            unparse_special: false,
                        },
                    ));
                    base += take;
                }
            }
            let outcomes = run_wave(
                scheduler,
                plan.iter().map(|(_, _, request)| request.clone()).collect(),
                stats,
            )?;
            refuse_forced(&outcomes)?;
            for ((group, base, _), outcome) in plan.iter().zip(&outcomes) {
                take_members(outcome, *group, *base, &mut output)?;
            }
        }
    }
    let kv_after = scheduler.kv_stats();
    stats.absorb_kv_delta(&kv_before, &kv_after);
    Ok(output)
}

/// Runs `requests` to completion through [`run_requests_with`], sampling the
/// live sequence count after each pump for [`RolloutStats`].
fn run_wave<B: DecodeBackend>(
    scheduler: &mut Scheduler<B>,
    requests: Vec<Request>,
    stats: &mut RolloutStats,
) -> Result<Vec<RequestOutcome>> {
    Ok(run_requests_with(scheduler, requests, |scheduler| {
        stats.record_step(scheduler.live_len());
    })?)
}

/// The batched generators' forced-token refusal, applied to a batch of
/// outcomes: any forced token fails the whole call.
fn refuse_forced(outcomes: &[RequestOutcome]) -> Result<()> {
    if outcomes
        .iter()
        .flat_map(|outcome| &outcome.completions)
        .any(|completion| completion.forced.iter().any(|&forced| forced))
    {
        return Err(Error::invalid(FORCED_REFUSAL));
    }
    Ok(())
}

/// Extracts the one completion of a single-member request.
fn take_single(outcome: &RequestOutcome) -> Result<RolloutCompletion> {
    if let Some(error) = &outcome.error {
        return Err(error.clone().into());
    }
    match outcome.completions.as_slice() {
        [single] => Ok(RolloutCompletion {
            tokens: single.tokens.clone(),
            forced: single.forced.clone(),
        }),
        _ => Err(Error::runtime(
            "a single-completion request returned no completion",
        )),
    }
}

/// Places a multi-member request's completions at `output[group][base..]`.
fn take_members(
    outcome: &RequestOutcome,
    group: usize,
    base: usize,
    output: &mut [Vec<RolloutCompletion>],
) -> Result<()> {
    if let Some(error) = &outcome.error {
        return Err(error.clone().into());
    }
    if base + outcome.completions.len() > output[group].len() {
        return Err(Error::runtime(
            "the scheduler returned more completions than the group asked for",
        ));
    }
    for (offset, completion) in outcome.completions.iter().enumerate() {
        output[group][base + offset] = RolloutCompletion {
            tokens: completion.tokens.clone(),
            forced: completion.forced.clone(),
        };
    }
    Ok(())
}

/// Public entry for integration tests: resolves the row layout, builds and
/// initializes the scheduler over `trainer`, and runs `groups` through it in
/// the config's admission mode. Returns the per-group completions and the
/// stats the driver sampled. Production code goes through
/// [`super::sampling::sample_rollout_groups_continuous`], which wraps this with
/// the detokenization and scoring that need the trainer back.
pub fn run_scheduled_rollout_groups(
    trainer: &mut Trainer,
    training: &TrainConfig,
    groups: &[(&[i32], Vec<SamplingParams>)],
    sequence_capacity: usize,
) -> Result<(Vec<Vec<RolloutCompletion>>, RolloutStats)> {
    let layout = RowLayout::resolve(trainer, training)?;
    let queue_bound = groups
        .iter()
        .map(|(_, params)| params.len())
        .sum::<usize>()
        .max(1);
    let mut stats = RolloutStats::default();
    let completions = {
        let mut scheduler = new_scheduler(trainer, &layout, queue_bound)?;
        run_rollout_groups(
            &mut scheduler,
            training.rollout_admission,
            &layout,
            sequence_capacity,
            groups,
            &mut stats,
        )?
    };
    Ok((completions, stats))
}
