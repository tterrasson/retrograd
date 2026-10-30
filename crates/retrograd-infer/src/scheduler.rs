//! The scheduler: one step at a time - admit, compose, execute, emit, remove.
//!
//! The plan of every step comes from here (the policy); its execution belongs
//! to the [`DecodeBackend`] - a single decode per step. Composition respects
//! the `n_batch` token budget: every live decoding sequence contributes one
//! token, and the remaining budget is filled with prefill chunks, so a long
//! prompt never freezes the other sequences' decode steps.
//!
//! Two admission modes ([`AdmissionMode`]): `Lockstep` admits a wave only
//! when nothing is running, in the canonical order (prompt length, prompt
//! tokens, then sampling parameters) and with identical prompts merged into
//! one prefill + `seq_cp` group - composing its steps the way the batched
//! continuous generator does. `Continuous` refills a freed slot at the next
//! step. In both modes a request whose reservation can never fit the pool is
//! refused with an error instead of waiting forever; admissible requests are
//! never starved: admission preserves FIFO order before ordering a wave.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::{Duration, Instant};

use retrograd_core::{RolloutAdmission, SamplingParamsV2};
use retrograd_engine::{EngineMemory, StepOutRow, StepOutput, StepRow};

use crate::backend::DecodeBackend;
use crate::detok::Detokenizer;
use crate::error::InferError;
use crate::kv::{KvCacheKey, KvStats, PinId, PrefixTree};
use crate::request::{Event, EventStream, FinishReason, Request, TokenEvent, TopLogprob};

/// Consecutive pumps without progress after which a synchronous driver gives
/// up: every admissible request is admitted, refused or retired within a few
/// iterations, so a long run of empty ones is a stuck queue, not a slow one.
pub(crate) const MAX_STALLED_PUMPS: usize = 1024;

/// Counts consecutive pumps that made no progress (see [`Scheduler::pump`]).
#[derive(Default)]
pub(crate) struct StallGuard(usize);

impl StallGuard {
    pub(crate) fn observe(&mut self, progressed: bool) -> Result<(), InferError> {
        if progressed {
            self.0 = 0;
            return Ok(());
        }
        self.0 += 1;
        if self.0 > MAX_STALLED_PUMPS {
            return Err(InferError::Failed(
                "the scheduler made no progress with requests still queued".into(),
            ));
        }
        Ok(())
    }
}

/// How queued requests are admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionMode {
    /// A wave is admitted only when no sequence is live, in canonical order,
    /// identical prompts merged into one prefill + `seq_cp` group. Composing
    /// steps in this mode reproduces the batched continuous generator.
    Lockstep,
    /// A slot freed at step N is refilled at step N+1 from the queue (FIFO).
    Continuous,
}

impl From<RolloutAdmission> for AdmissionMode {
    fn from(admission: RolloutAdmission) -> Self {
        match admission {
            RolloutAdmission::Lockstep => Self::Lockstep,
            RolloutAdmission::Continuous => Self::Continuous,
        }
    }
}

/// The scheduler's knobs.
#[derive(Clone, Debug)]
pub struct SchedulerOptions {
    pub mode: AdmissionMode,
    /// The token budget of one step (the backend's `n_batch`).
    pub n_batch: usize,
    /// How many requests may wait for admission; further `submit`s are
    /// refused with [`InferError::QueueFull`].
    pub queue_capacity: usize,
    /// The bounded event buffer of one request's stream.
    pub event_capacity: usize,
    /// How long a send to a full event stream may block before the request
    /// is cancelled (bounded backpressure, then cancellation).
    pub slow_client_timeout: Duration,
    /// Keep a radix tree of finished sequences' KV (dormant slots) and reuse
    /// the longest shared prefix at admission via `seq_cp`. Outputs stay bit
    /// identical; only the compute path changes.
    pub prefix_cache: bool,
    /// Sequences shorter than this are not inserted into the prefix tree.
    pub min_prefix_len: usize,
    /// Host byte budget for swapped-out prefix snapshots (seq_save to host).
    pub max_snapshot_bytes: u64,
    /// Nodes at least this long are swapped (snapshot kept) rather than
    /// dropped when evicted under pressure; shorter ones are recomputed on
    /// the next hit.
    pub swap_min_tokens: usize,
}

impl Default for SchedulerOptions {
    fn default() -> Self {
        Self {
            mode: AdmissionMode::Lockstep,
            n_batch: 512,
            queue_capacity: 64,
            event_capacity: 64,
            slow_client_timeout: Duration::from_secs(5),
            prefix_cache: true,
            min_prefix_len: 8,
            max_snapshot_bytes: 256 << 20,
            swap_min_tokens: 256,
        }
    }
}

/// A request waiting for admission.
pub(crate) struct Queued {
    pub request: Request,
    pub events: SyncSender<Event>,
    pub cancelled: Arc<AtomicBool>,
    pub permit: Option<QueuePermit>,
}

/// One shared queue slot, held while an envelope waits in either queue.
/// Dropping a rejected, cancelled or admitted envelope returns the slot.
pub(crate) struct QueuePermit(pub Arc<AtomicUsize>);

impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, AtomicOrdering::SeqCst);
    }
}

enum GroupPhase {
    /// The leader is prefilling `[0, prefix_end)` in chunks.
    Prefill { pos: usize },
    /// The prefix is copied; each member still needs its terminal prompt
    /// token sampled (one row per member).
    Terminal { next: usize },
    /// Every member decodes one token per step.
    Decode,
}

struct LiveMember {
    slot: i32,
    request_id: i32,
    /// The completion index within the member's own request (0..n), used in
    /// events - not the member's position in a merged group.
    index: usize,
    sampling: SamplingParamsV2,
    /// Whether token events also carry the sampled token's own logprob.
    want_logprob: bool,
    /// `min(max_new_tokens, n_ctx_seq - prompt.len())`: the stop budget.
    limit: usize,
    /// This member's share of the group's cell reservation (prompt minus the
    /// cache-shared prefix, plus the budget).
    reserved_share: u64,
    generated: usize,
    next_token: i32,
    /// The tokens this member generated, in order - the tail of the prefix
    /// path inserted into the tree when the member finishes.
    history: Vec<i32>,
    live: bool,
    n_top_logprobs: u32,
    /// Render special tokens as text (from the request).
    unparse_special: bool,
    detok: Detokenizer,
    events: Option<SyncSender<Event>>,
    cancelled: Arc<AtomicBool>,
}

struct LiveGroup {
    prompt: Vec<i32>,
    /// `prompt.len() - 1` for a shared group (the terminal token is decoded
    /// per member after the copies), `prompt.len()` for a single member.
    prefix_end: usize,
    /// The member holding the shared prefix while it is being prefilled.
    leader: usize,
    /// Cells reserved for the whole group (prompt + budgets, minus the
    /// cache-shared prefix).
    reserved: u64,
    members: Vec<LiveMember>,
    phase: GroupPhase,
}

/// Which live row a composed step row belongs to.
enum RowKind {
    Decode {
        group: usize,
        member: usize,
    },
    Prefill {
        group: usize,
        end: usize,
        sampled_member: Option<usize>,
    },
    Terminal {
        group: usize,
        member: usize,
    },
}

/// The cells one member reserves: the prompt plus its generation budget,
/// clamped to the per-sequence context. `None` when the prompt leaves no
/// room to generate at all.
fn member_need(prompt_len: usize, sampling: &SamplingParamsV2, n_ctx_seq: usize) -> Option<u64> {
    if prompt_len >= n_ctx_seq {
        return None;
    }
    let budget = (sampling.max_new_tokens as usize).min(n_ctx_seq - prompt_len);
    Some((prompt_len + budget) as u64)
}

fn reservation(request: &Request, n_ctx_seq: usize) -> u64 {
    request
        .sampling
        .iter()
        .map(|sampling| member_need(request.prompt.len(), sampling, n_ctx_seq).unwrap_or(0))
        // Reservation is a budget: saturate rather than wrapping an
        // impossible request back into the admissible range.
        .fold(0, u64::saturating_add)
}

/// The canonical order of the lockstep wave: prompt length, prompt tokens,
/// then the sampling parameters - the order the batched continuous
/// generator sorts its sequences in.
fn canonical_cmp(a: &Request, b: &Request) -> Ordering {
    let (pa, sa) = (&a.prompt, &a.sampling[0]);
    let (pb, sb) = (&b.prompt, &b.sampling[0]);
    pa.len()
        .cmp(&pb.len())
        .then_with(|| pa.cmp(pb))
        .then_with(|| sampling_cmp(sa, sb))
}

fn sampling_cmp(a: &SamplingParamsV2, b: &SamplingParamsV2) -> Ordering {
    a.seed
        .cmp(&b.seed)
        .then_with(|| a.max_new_tokens.cmp(&b.max_new_tokens))
        .then_with(|| a.temperature.total_cmp(&b.temperature))
        .then_with(|| a.top_p.total_cmp(&b.top_p))
        .then_with(|| a.top_k.cmp(&b.top_k))
        .then_with(|| a.min_p.total_cmp(&b.min_p))
        .then_with(|| a.repeat_penalty.total_cmp(&b.repeat_penalty))
        .then_with(|| a.frequency_penalty.total_cmp(&b.frequency_penalty))
        .then_with(|| a.presence_penalty.total_cmp(&b.presence_penalty))
        .then_with(|| a.penalty_window.cmp(&b.penalty_window))
        .then_with(|| {
            a.logit_bias
                .iter()
                .map(|&(id, bias)| (id, bias.to_bits()))
                .cmp(b.logit_bias.iter().map(|&(id, bias)| (id, bias.to_bits())))
        })
        .then_with(|| a.grammar_gbnf.cmp(&b.grammar_gbnf))
}

/// Sends one event to a request's bounded stream: `try_send` in a loop
/// until the deadline, so a full stream costs at most `timeout` of bounded
/// backpressure before the caller cancels the request. (`SyncSender::
/// send_timeout` is still unstable in std, hence the poll.)
fn send_bounded(
    events: &SyncSender<Event>,
    mut event: Event,
    timeout: Duration,
) -> Result<(), Event> {
    let started = Instant::now();
    loop {
        match events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Disconnected(event)) => return Err(event),
            Err(TrySendError::Full(returned)) => {
                if started.elapsed() >= timeout {
                    return Err(returned);
                }
                event = returned;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// The step-wise scheduler over a decode backend. Owns the queue, the live
/// groups and the reused step buffers; the actor drives it from its thread.
pub struct Scheduler<B: DecodeBackend> {
    backend: B,
    options: SchedulerOptions,
    queue: VecDeque<Queued>,
    live: Vec<LiveGroup>,
    n_slots: usize,
    next_request_id: i32,
    out: StepOutput,
    /// Staging for the one-token rows (decode and terminal), so composing a
    /// step allocates nothing in steady state.
    scratch: Vec<i32>,
    decode_cursor: usize,
    prefill_turn: bool,
    /// The prefix tree (dormant slots + snapshots), valid under `kv_key`.
    kv: PrefixTree,
    kv_key: Option<KvCacheKey>,
    kv_stats: KvStats,
    /// The adapter currently active on the backend. A composed step never
    /// mixes adapters: admission only lets requests for this adapter join the
    /// live set, and switches it when nothing is live.
    active_adapter: Option<String>,
}

/// A point-in-time report of the scheduler and its backend, for a stats
/// endpoint: prefix-cache counters, pool occupancy, queue/live lengths and
/// the active adapter.
#[derive(Clone, Debug)]
pub struct SchedulerStats {
    pub kv: KvStats,
    pub memory: EngineMemory,
    pub queued: usize,
    pub live: usize,
    pub adapter: Option<String>,
}

impl<B: DecodeBackend> Scheduler<B> {
    pub fn new(backend: B, mut options: SchedulerOptions) -> Self {
        options.queue_capacity = options.queue_capacity.max(1);
        let n_batch = options.n_batch.max(1);
        Self {
            backend,
            options,
            queue: VecDeque::new(),
            live: Vec::new(),
            n_slots: 0,
            next_request_id: 1,
            out: StepOutput::with_capacity(n_batch, 0),
            scratch: Vec::with_capacity(n_batch),
            decode_cursor: 0,
            prefill_turn: false,
            kv: PrefixTree::new(),
            kv_key: None,
            kv_stats: KvStats::default(),
            active_adapter: None,
        }
    }

    /// The prefix-cache counters (hits, prefilled vs prompt cells, evictions,
    /// swaps, live tree size).
    pub fn kv_stats(&self) -> KvStats {
        KvStats {
            nodes: self.kv.node_count(),
            resident_slots: self.kv.resident_slot_count(),
            snapshot_bytes: self.kv.snapshot_bytes(),
            ..self.kv_stats
        }
    }

    /// A point-in-time report of the scheduler and its backend.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            kv: self.kv_stats(),
            memory: self.backend.memory().unwrap_or_default(),
            queued: self.queue.len(),
            live: self.live_len(),
            adapter: self.active_adapter.clone(),
        }
    }

    /// Makes `wanted` the backend's active adapter when it is not already:
    /// activation invalidates the resident KV on the backend side (weights
    /// version bump), and the next [`Scheduler::refresh_key`] drops the prefix
    /// tree, so no prefix computed under another adapter is ever reused.
    fn ensure_adapter(&mut self, wanted: Option<&str>) -> Result<(), InferError> {
        let current = self.backend.active_adapter()?;
        if current.as_deref() != wanted {
            self.backend.activate_adapter(wanted)?;
            self.active_adapter = wanted.map(str::to_owned);
            self.refresh_key()?;
        }
        Ok(())
    }

    /// Pins the tree node at exactly `prefix` (which must be resident) so LRU
    /// eviction never releases it - the anchor a multi-turn conversation
    /// keeps its KV through.
    pub fn pin_prefix(&mut self, prefix: &[i32]) -> Result<PinId, InferError> {
        self.refresh_key()?;
        self.kv.pin_at(prefix).ok_or_else(|| {
            InferError::invalid("no resident prefix node at exactly that token path")
        })
    }

    /// Releases one pin.
    pub fn unpin(&mut self, id: PinId) -> bool {
        self.kv.unpin(id)
    }

    /// Drops the whole prefix tree when the backend's cache key changed (the
    /// weights/adapter generation moved): no prefix computed under other
    /// weights is ever reused.
    fn refresh_key(&mut self) -> Result<(), InferError> {
        let key = self.backend.cache_key()?;
        if self.kv_key != Some(key) {
            let slots = self.kv.clear();
            for slot in slots {
                self.backend.seq_rm(slot, -1, -1)?;
            }
            self.kv_key = Some(key);
        }
        Ok(())
    }

    pub(crate) fn enqueue(&mut self, queued: Queued) {
        self.queue.push_back(queued);
    }

    /// The backend, for the caller that owns the scheduler's thread (prompt
    /// rendering, adapter registration).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutable access to the backend, same thread only.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Refuses every queued (never admitted) request with a cancellation
    /// error, e.g. when the backend is about to be unloaded.
    pub fn fail_queued(&mut self) {
        while let Some(queued) = self.queue.pop_front() {
            let _ = queued.events.try_send(Event::Error(InferError::Cancelled));
        }
    }

    /// Answers a control envelope (stats / pin / unpin / adapter load) without
    /// touching the admission queue.
    fn handle_control(&mut self, envelope: Envelope) {
        match envelope {
            Envelope::Stats(reply) => {
                let _ = reply.send(self.stats());
            }
            Envelope::Pin { prefix, reply } => {
                let _ = reply.send(self.pin_prefix(&prefix));
            }
            Envelope::Unpin { id, reply } => {
                let _ = reply.send(self.unpin(id));
            }
            Envelope::LoadAdapter { name, path, reply } => {
                let _ = reply.send(self.backend.load_adapter(&name, &path));
            }
            Envelope::Submit(_) | Envelope::Wake => {}
        }
    }

    /// One-time pool setup: reads the pool geometry and clears every slot,
    /// so residue left by earlier batched generation calls cannot eat the
    /// budget. Both the actor loop and the synchronous driver (see
    /// [`Scheduler::submit`]) call this once before their first step.
    pub fn initialize(&mut self) -> Result<(), InferError> {
        // The scheduler owns placement: start from empty slots.
        let memory = self.backend.memory()?;
        if memory.n_slots > i32::MAX as u32 {
            return Err(InferError::Overflow(
                "backend slot count exceeds signed native IDs".into(),
            ));
        }
        self.n_slots = memory.n_slots as usize;
        for slot in 0..self.n_slots {
            self.backend.seq_rm(slot as i32, -1, -1)?;
        }
        self.active_adapter = self.backend.active_adapter()?;
        Ok(())
    }

    /// Submits one request without the actor, on the thread that owns the
    /// backend: the returned stream behaves exactly like the actor's (drop =
    /// cancellation at the next step). Refused with [`InferError::QueueFull`]
    /// when the admission queue is full - a capacity condition, not a caller
    /// error. See [`crate::run_requests`] for the common batch shape.
    pub fn submit(&mut self, request: Request) -> Result<EventStream, InferError> {
        request.validate()?;
        if self.queue.len() >= self.options.queue_capacity {
            return Err(InferError::QueueFull {
                capacity: self.options.queue_capacity,
            });
        }
        // A local caller drains the stream while pumping; the bound only has
        // to cover what one step can produce for this request.
        let capacity = self.options.event_capacity.max(request.n() + 2);
        let (events, rx) = mpsc::sync_channel(capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.enqueue(Queued {
            request,
            events,
            cancelled: cancelled.clone(),
            permit: None,
        });
        Ok(EventStream::new(rx, cancelled))
    }

    /// Requests waiting for admission.
    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    /// The admission queue capacity [`Scheduler::submit`] enforces.
    pub fn queue_capacity(&self) -> usize {
        self.options.queue_capacity
    }

    /// Live (admitted, not finished) sequences.
    pub fn live_len(&self) -> usize {
        self.live
            .iter()
            .flat_map(|group| &group.members)
            .filter(|member| member.live)
            .count()
    }

    /// Whether no request is queued and none is live.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.live.is_empty()
    }

    /// Runs one scheduler iteration: remove cancelled sequences, admit,
    /// compose and execute one step, emit, remove finished sequences. Returns
    /// whether the iteration made progress - executed a step, admitted,
    /// refused or retired a request. A failed step fails every live request
    /// and is returned.
    pub fn pump(&mut self) -> Result<bool, InferError> {
        let before = (self.queue.len(), self.live_len());
        match self.step_once() {
            Ok(stepped) => Ok(stepped || (self.queue.len(), self.live_len()) != before),
            Err(error) => {
                self.fail_all(&error);
                Err(error)
            }
        }
    }

    /// Lends the caller one pool slot outside the scheduler's bookkeeping,
    /// for host work between waves (a teacher-forced scorer): a slot no live
    /// sequence and no dormant prefix holds, with `cells` of the pool left
    /// beside the dormant prefixes. LRU prefix nodes are evicted (swapped to
    /// a host snapshot when they qualify) until both fit, so the rest of the
    /// prefix tree survives the loan. `None` when nothing more can be evicted.
    ///
    /// Only while no sequence is live. The caller must leave the slot empty
    /// (`seq_rm`) before the next pump.
    pub fn lend_slot(&mut self, cells: u64) -> Result<Option<i32>, InferError> {
        if !self.live.is_empty() {
            return Err(InferError::Failed(
                "a slot can only be lent while no sequence is live".into(),
            ));
        }
        loop {
            let (slots_left, cells_left) = self.capacity()?;
            if slots_left > 0 && cells_left >= cells {
                let mut held = vec![false; self.n_slots];
                for slot in self.kv.resident_slots() {
                    if let Some(held) = held.get_mut(slot as usize) {
                        *held = true;
                    }
                }
                // The backend's slot count fits i32, checked at startup.
                return Ok(held.iter().position(|held| !*held).map(|slot| slot as i32));
            }
            if !self.evict_one_lru()? {
                return Ok(None);
            }
        }
    }

    /// Pumps until no request is queued or live. Guarded against a stuck
    /// queue: after 1024 consecutive iterations without progress the
    /// scheduler reports a failure instead of spinning.
    pub fn run_until_idle(&mut self) -> Result<(), InferError> {
        let mut stall = StallGuard::default();
        while !self.is_idle() {
            stall.observe(self.pump()?)?;
        }
        Ok(())
    }

    /// The actor's main loop, on the thread that owns the backend.
    pub(crate) fn run(
        &mut self,
        inbox: Receiver<Envelope>,
        shutdown: Arc<AtomicBool>,
    ) -> Result<(), InferError> {
        self.initialize()?;
        let mut draining = false;
        loop {
            // Pull pending envelopes while the admission queue has room: a
            // full internal queue leaves the bounded inbox full, which is
            // what refuses further submits with QueueFull.
            while self.queue.len() < self.options.queue_capacity {
                match inbox.try_recv() {
                    Ok(Envelope::Submit(queued)) => self.enqueue(queued),
                    Ok(Envelope::Wake) => {}
                    Ok(control) => self.handle_control(control),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        draining = true;
                        break;
                    }
                }
            }
            let shutting_down = draining || shutdown.load(AtomicOrdering::SeqCst);
            if shutting_down {
                // Graceful: queued (never admitted) requests are refused,
                // live ones run to completion.
                // Include requests still buffered in the inbox.
                while let Ok(envelope) = inbox.try_recv() {
                    match envelope {
                        Envelope::Submit(queued) => self.enqueue(queued),
                        Envelope::Wake => {}
                        control => self.handle_control(control),
                    }
                }
                while let Some(queued) = self.queue.pop_front() {
                    let _ = queued.events.try_send(Event::Error(InferError::Cancelled));
                }
            }
            if self.live.is_empty() && self.queue.is_empty() {
                if shutting_down {
                    return Ok(());
                }
                match inbox.recv() {
                    Ok(Envelope::Submit(queued)) => self.enqueue(queued),
                    Ok(Envelope::Wake) => {}
                    Ok(control) => self.handle_control(control),
                    Err(_) => return Ok(()),
                }
                continue;
            }
            // A failure has already been reported to every live request.
            let _ = self.pump();
        }
    }

    /// One scheduler iteration: remove cancelled sequences, admit, compose
    /// and execute one step, emit, remove finished sequences. Returns whether
    /// a step was executed.
    fn step_once(&mut self) -> Result<bool, InferError> {
        self.queue
            .retain(|queued| !queued.cancelled.load(AtomicOrdering::SeqCst));
        self.remove_cancelled()?;
        self.admit()?;
        self.advance_phases()?;
        if self.live.is_empty() {
            return Ok(false);
        }
        let (rows, kinds, total_tokens) = Self::compose(
            &self.live,
            &mut self.scratch,
            self.options.n_batch,
            self.options.mode,
            &mut self.decode_cursor,
            &mut self.prefill_turn,
        )?;
        if rows.is_empty() {
            return Ok(false);
        }
        if let Err(error) = self.backend.step(&rows, &mut self.out) {
            // A failed step may have made a prefix of the submitted tokens
            // resident: clear every touched slot before reporting, so no
            // token is ever replayed twice.
            let error = self.classify_step_error(error, total_tokens);
            self.rollback(&kinds);
            return Err(error);
        }
        for (index, kind) in kinds.into_iter().enumerate() {
            match kind {
                RowKind::Decode { group, member } => {
                    let produced = self.out.row(index);
                    self.emit_sampled(group, member, produced, index)?;
                }
                RowKind::Terminal { group, member } => {
                    let produced = self.out.row(index);
                    self.emit_sampled(group, member, produced, index)?;
                    let g = &mut self.live[group];
                    if let GroupPhase::Terminal { next } = &mut g.phase {
                        *next = (*next).max(member + 1);
                        if *next >= g.members.len() {
                            g.phase = GroupPhase::Decode;
                        }
                    }
                }
                RowKind::Prefill {
                    group,
                    end,
                    sampled_member,
                } => {
                    {
                        let g = &mut self.live[group];
                        if let GroupPhase::Prefill { pos } = &mut g.phase {
                            *pos = end;
                        }
                    }
                    if let Some(member) = sampled_member {
                        let produced = self.out.row(index);
                        self.emit_sampled(group, member, produced, index)?;
                    }
                }
            }
        }
        self.live
            .retain(|group| group.members.iter().any(|member| member.live));
        Ok(true)
    }

    /// Composes the next step's plan: decode rows first (one token per live
    /// decoding member, in live order), then prefill/terminal rows with the
    /// remaining `n_batch` budget. In `Lockstep` mode a step never mixes
    /// decodes with prefills: a wave prefills entirely before decoding, the
    /// way the batched continuous generator does.
    ///
    /// An associated function over disjoint fields so the returned rows can
    /// borrow `live` and `scratch` while the caller steps the backend and
    /// its output buffers.
    fn compose<'a>(
        live: &'a [LiveGroup],
        scratch: &'a mut Vec<i32>,
        n_batch: usize,
        mode: AdmissionMode,
        decode_cursor: &mut usize,
        prefill_turn: &mut bool,
    ) -> Result<(Vec<StepRow<'a>>, Vec<RowKind>, u64), InferError> {
        let n_batch = n_batch.max(1);
        let mut used = 0_usize;
        let mut rows: Vec<StepRow<'_>> = Vec::new();
        let mut kinds: Vec<RowKind> = Vec::new();
        scratch.clear();

        let prefill_active = live
            .iter()
            .any(|group| !matches!(group.phase, GroupPhase::Decode));
        let allow_decode = mode == AdmissionMode::Continuous || !prefill_active;

        // Pass 1: pick the decode rows and stage their tokens.
        let mut decode_list: Vec<(usize, usize)> = Vec::new();
        if allow_decode {
            for (gi, group) in live.iter().enumerate() {
                if !matches!(group.phase, GroupPhase::Decode) {
                    continue;
                }
                for (mi, member) in group.members.iter().enumerate() {
                    if member.live {
                        decode_list.push((gi, mi));
                    }
                }
            }
        }
        // Reserve progress for prefill even when decodes fill the batch.
        // Rotate decoding members when there are more than fit in one step.
        let decode_budget = if mode == AdmissionMode::Continuous && prefill_active {
            if n_batch == 1 {
                *prefill_turn = !*prefill_turn;
                usize::from(!*prefill_turn)
            } else {
                n_batch - 1
            }
        } else {
            n_batch
        };
        if !decode_list.is_empty() {
            let start = *decode_cursor % decode_list.len();
            decode_list.rotate_left(start);
            let count = decode_budget.min(decode_list.len());
            *decode_cursor = (start + count) % decode_list.len();
            decode_list.truncate(count);
        }
        for &(gi, mi) in &decode_list {
            scratch.push(live[gi].members[mi].next_token);
        }
        used += decode_list.len();
        let position = |pos: usize| -> Result<i32, InferError> {
            i32::try_from(pos)
                .map_err(|_| InferError::Overflow("a step position does not fit i32".into()))
        };
        // Pass 2: build the rows (borrowing the live prompts and the scratch).
        let scratch_ref: &[i32] = scratch;
        for (offset, &(gi, mi)) in decode_list.iter().enumerate() {
            let group = &live[gi];
            let member = &group.members[mi];
            let pos = group.prompt.len() + member.generated - 1;
            rows.push(StepRow {
                request: member.request_id,
                slot: member.slot,
                tokens: Cow::Borrowed(&scratch_ref[offset..offset + 1]),
                pos0: position(pos)?,
                sample: true,
                sampling: member.sampling.clone(),
                n_top_logprobs: member.n_top_logprobs,
                want_logprob: member.want_logprob,
            });
            kinds.push(RowKind::Decode {
                group: gi,
                member: mi,
            });
        }
        for (gi, group) in live.iter().enumerate() {
            if used >= n_batch {
                break;
            }
            match &group.phase {
                GroupPhase::Prefill { pos } => {
                    let remaining = n_batch - used;
                    let chunk = remaining.min(group.prefix_end - *pos);
                    if chunk == 0 {
                        continue;
                    }
                    let end = pos + chunk;
                    let single = group.members.len() == 1;
                    let sampled_member = if single && end == group.prompt.len() {
                        Some(0)
                    } else {
                        None
                    };
                    let leader = &group.members[group.leader];
                    rows.push(StepRow {
                        request: leader.request_id,
                        slot: leader.slot,
                        tokens: Cow::Borrowed(&group.prompt[*pos..end]),
                        pos0: position(*pos)?,
                        sample: sampled_member.is_some(),
                        sampling: leader.sampling.clone(),
                        n_top_logprobs: if sampled_member.is_some() {
                            leader.n_top_logprobs
                        } else {
                            0
                        },
                        want_logprob: sampled_member.is_some() && leader.want_logprob,
                    });
                    kinds.push(RowKind::Prefill {
                        group: gi,
                        end,
                        sampled_member,
                    });
                    used += chunk;
                }
                GroupPhase::Terminal { next } => {
                    let last = group.prompt.len() - 1;
                    let mut mi = *next;
                    while mi < group.members.len() && used < n_batch {
                        let member = &group.members[mi];
                        if !member.live {
                            mi += 1;
                            continue;
                        }
                        rows.push(StepRow {
                            request: member.request_id,
                            slot: member.slot,
                            tokens: Cow::Borrowed(&group.prompt[last..]),
                            pos0: position(last)?,
                            sample: true,
                            sampling: member.sampling.clone(),
                            n_top_logprobs: member.n_top_logprobs,
                            want_logprob: member.want_logprob,
                        });
                        kinds.push(RowKind::Terminal {
                            group: gi,
                            member: mi,
                        });
                        used += 1;
                        mi += 1;
                    }
                }
                GroupPhase::Decode => {}
            }
        }
        let total_tokens = used as u64;
        Ok((rows, kinds, total_tokens))
    }

    /// Phase transitions that need no step row: a completed prefix is copied
    /// to the group's members (between steps), and a completed terminal wave
    /// becomes a decode wave.
    fn advance_phases(&mut self) -> Result<(), InferError> {
        for gi in 0..self.live.len() {
            let done = matches!(&self.live[gi].phase, GroupPhase::Prefill { pos } if *pos == self.live[gi].prefix_end);
            if done {
                if self.live[gi].members.len() > 1 {
                    let leader_slot = self.live[gi].members[self.live[gi].leader].slot;
                    let slots: Vec<i32> = self.live[gi]
                        .members
                        .iter()
                        .filter(|member| member.live && member.slot != leader_slot)
                        .map(|member| member.slot)
                        .collect();
                    if self.live[gi].prefix_end > 0 {
                        for slot in slots {
                            self.backend.seq_cp(leader_slot, slot, -1, -1)?;
                        }
                    }
                    self.live[gi].phase = GroupPhase::Terminal { next: 0 };
                } else {
                    self.live[gi].phase = GroupPhase::Decode;
                }
            } else if matches!(&self.live[gi].phase, GroupPhase::Terminal { next } if self.live[gi].members[*next..].iter().all(|member| !member.live))
            {
                self.live[gi].phase = GroupPhase::Decode;
            }
        }
        Ok(())
    }

    /// Admits queued requests according to the mode.
    fn admit(&mut self) -> Result<(), InferError> {
        if self.queue.is_empty() {
            return Ok(());
        }
        // The prefix tree is only valid under the current weights generation:
        // a key change drops every node and releases its slot.
        self.refresh_key()?;
        // Host byte budget for swapped snapshots: drop LRU swapped nodes when
        // over budget (they hold no pool cells, only host bytes).
        while self.kv.snapshot_bytes() > self.options.max_snapshot_bytes {
            let Some(node) = self.kv.snapshot_eviction_candidate() else {
                break;
            };
            self.kv.remove(node);
            self.kv_stats.evictions += 1;
        }
        let n_ctx_seq = self.backend.context_size()? as usize;
        let vocab = self.backend.vocab_size()? as usize;
        let memory = self.backend.memory()?;
        self.n_slots = memory.n_slots as usize;
        let cells_total = memory.cells_total;

        // Refuse what can never fit: an impossible reservation is an error,
        // not a wait.
        let mut index = 0;
        while index < self.queue.len() {
            let queued = &self.queue[index];
            let prompt_too_long = queued.request.prompt.len() >= n_ctx_seq;
            let too_many_members = queued.request.n() > self.n_slots;
            let top_too_wide = queued.request.n_top_logprobs as usize > vocab;
            let invalid_token = queued
                .request
                .prompt
                .iter()
                .any(|&token| token < 0 || token as usize >= vocab);
            let invalid_bias = queued.request.sampling.iter().any(|params| {
                params
                    .logit_bias
                    .iter()
                    .any(|&(token, _)| token < 0 || token as usize >= vocab)
            });
            let need = reservation(&queued.request, n_ctx_seq);
            if prompt_too_long
                || too_many_members
                || top_too_wide
                || invalid_token
                || invalid_bias
                || need > cells_total
            {
                let queued = self.queue.remove(index).expect("index in range");
                let message = if prompt_too_long {
                    format!(
                        "the prompt ({} tokens) leaves no room to generate in the per-sequence context ({n_ctx_seq})",
                        queued.request.prompt.len()
                    )
                } else if too_many_members {
                    format!(
                        "the request asks for {} completions but the pool has {} slots",
                        queued.request.n(),
                        self.n_slots
                    )
                } else if top_too_wide {
                    format!("n_top_logprobs exceeds the vocabulary ({vocab})")
                } else if invalid_bias {
                    format!("logit_bias token ids must be in 0..{vocab}")
                } else if invalid_token {
                    format!("prompt tokens must be in 0..{vocab}")
                } else {
                    format!(
                        "the reservation (prompt + max_new_tokens = {need} cells) exceeds the KV pool of {cells_total} cells"
                    )
                };
                let _ = queued
                    .events
                    .try_send(Event::Error(InferError::invalid(message)));
                continue;
            }
            index += 1;
        }
        if self.queue.is_empty() {
            return Ok(());
        }

        // Adapters serialize the batch: a composed step never mixes adapters.
        // When nothing is live, the front of the queue chooses the adapter to
        // activate; while sequences are live, only requests for the active
        // adapter may join them. Requests for other adapters wait - FIFO
        // across adapter groups, since the group at the front chose the target.
        let target = match self.live.is_empty() {
            true => self
                .queue
                .front()
                .and_then(|queued| queued.request.adapter.clone()),
            false => self.active_adapter.clone(),
        };
        // Partition the queue for this pass: only candidates (same adapter as
        // the target) are admitted now; the rest stay queued, in order, behind
        // the leftover candidates.
        let mut candidates = VecDeque::new();
        let mut rest = VecDeque::new();
        for queued in self.queue.drain(..) {
            if queued.request.adapter == target {
                candidates.push_back(queued);
            } else {
                rest.push_back(queued);
            }
        }
        // An adapter that cannot be activated (not registered, or the backend
        // has no named adapters) refuses its requests with an error event -
        // a pump failure is not the way, it would fail unrelated live work.
        // While sequences are live the target is the active adapter already.
        let switched = match self.live.is_empty() {
            true => self.ensure_adapter(target.as_deref()),
            false => Ok(()),
        };
        if let Err(error) = switched {
            while let Some(queued) = candidates.pop_front() {
                let _ = queued.events.try_send(Event::Error(error.clone()));
            }
            self.queue = rest;
            return Ok(());
        }
        self.queue = candidates;
        let admitted = self.admit_candidates(n_ctx_seq);
        // Put the other adapters' requests back, ahead of the candidates this
        // pass did not admit (on error too): the next pass re-reads the front
        // to choose the next adapter.
        let leftover = std::mem::take(&mut self.queue);
        self.queue = rest;
        self.queue.extend(leftover);
        admitted
    }

    /// Admits from the queue, which holds only requests for the active
    /// adapter at this point.
    fn admit_candidates(&mut self, n_ctx_seq: usize) -> Result<(), InferError> {
        // Bounded preemption: when the front of the queue does not fit, evict
        // LRU unpinned prefix nodes (swap to a host snapshot when long enough
        // and the byte budget allows, drop otherwise) until it fits or no
        // node is evictable. Never more than one pass over the slots per
        // admission call.
        let mut evictions_left = self.n_slots + 1;
        match self.options.mode {
            AdmissionMode::Lockstep => {
                // A wave is admitted only when nothing is live.
                if !self.live.is_empty() {
                    return Ok(());
                }
                // Select the oldest fitting wave before sorting it. Sorting
                // the whole queue first would let a stream of short prompts
                // overtake an older request forever. The node each selected
                // request hits is pinned until the wave is admitted, so a
                // later eviction cannot take back the cells it was charged
                // without.
                let mut pending = Vec::new();
                let mut wave_pins = Vec::new();
                let (mut held_slots, mut held_cells) = (0, 0);
                while let Some((members, need, node)) =
                    self.front_fits(n_ctx_seq, held_slots, held_cells, &mut evictions_left)?
                {
                    held_slots += members;
                    held_cells += need;
                    wave_pins.extend(node.map(|node| self.kv.pin_node(node)));
                    pending.push(self.queue.pop_front().expect("queue has a front"));
                }
                pending.sort_by(|a, b| canonical_cmp(&a.request, &b.request));
                let mut admitted = Ok(());
                while !pending.is_empty() && admitted.is_ok() {
                    let mut run_end = 1;
                    while run_end < pending.len()
                        && pending[run_end].request.prompt == pending[0].request.prompt
                    {
                        run_end += 1;
                    }
                    let run = pending.drain(..run_end).collect();
                    admitted = self.admit_group(run, n_ctx_seq);
                }
                for pin in wave_pins {
                    self.kv.unpin(pin);
                }
                admitted?;
            }
            AdmissionMode::Continuous => {
                // Do not let smaller arrivals indefinitely overtake a request
                // waiting for several slots or a larger cell reservation.
                while self
                    .front_fits(n_ctx_seq, 0, 0, &mut evictions_left)?
                    .is_some()
                {
                    let queued = self.queue.pop_front().expect("queue has a front");
                    self.admit_group(vec![queued], n_ctx_seq)?;
                }
            }
        }
        Ok(())
    }

    /// Free slots and cells beside the live groups and the dormant prefix
    /// slots. A live member never writes more than its reservation and a
    /// dormant slot holds at most its depth, so the commitment is the larger
    /// of what the pool reports and that bound.
    fn capacity(&self) -> Result<(usize, u64), InferError> {
        let memory = self.backend.memory()?;
        let used_slots = self
            .live
            .iter()
            .flat_map(|group| &group.members)
            .filter(|member| member.live)
            .count()
            + self.kv.resident_slot_count();
        let reserved: u64 = self.live.iter().map(|group| group.reserved).sum();
        let committed = memory
            .cells_used
            .max(reserved.saturating_add(self.kv.resident_cells()));
        Ok((
            self.n_slots.saturating_sub(used_slots),
            memory.cells_total.saturating_sub(committed),
        ))
    }

    /// Whether the front of the queue fits beside `held_slots`/`held_cells`
    /// already promised to a pending wave, evicting LRU prefix nodes (at most
    /// `evictions_left`) until it does. On a fit, returns its member count,
    /// its cell reservation net of its prefix-cache hit, and the hit node.
    fn front_fits(
        &mut self,
        n_ctx_seq: usize,
        held_slots: usize,
        held_cells: u64,
        evictions_left: &mut usize,
    ) -> Result<Option<(usize, u64, Option<usize>)>, InferError> {
        loop {
            let Some(queued) = self.queue.front() else {
                return Ok(None);
            };
            let members = queued.request.n();
            // Recomputed after every eviction: the evicted node may be the
            // one the hit came from.
            let hit = self.hit_candidate(&queued.request.prompt)?;
            let shared = hit.map_or(0, |(_, hit)| hit as u64 * members as u64);
            let need = reservation(&queued.request, n_ctx_seq).saturating_sub(shared);
            let (slots_left, cells_left) = self.capacity()?;
            if held_slots + members <= slots_left && held_cells + need <= cells_left {
                return Ok(Some((members, need, hit.map(|(node, _)| node))));
            }
            if *evictions_left == 0 || !self.evict_one_lru()? {
                return Ok(None);
            }
            *evictions_left -= 1;
        }
    }

    /// Evicts the LRU unpinned prefix node: swaps it to a host snapshot when it
    /// is long enough and the byte budget allows, drops it (recompute on the
    /// next hit) otherwise. Releases its slot with `seq_rm`. Returns false
    /// when no node is evictable.
    fn evict_one_lru(&mut self) -> Result<bool, InferError> {
        let Some(node) = self.kv.eviction_candidate() else {
            return Ok(false);
        };
        let slot = self.kv.slot(node).expect("candidate owns a slot");
        if self.kv.depth(node) >= self.options.swap_min_tokens {
            // A failed save only costs the swap: the node is dropped instead.
            let save_len = self.backend.seq_save_len(slot, 0).unwrap_or(0);
            if save_len > 0
                && self.kv.snapshot_bytes() + save_len as u64 <= self.options.max_snapshot_bytes
            {
                let mut buffer = vec![0_u8; save_len];
                if let Ok(written) = self.backend.seq_save_into(slot, 0, &mut buffer) {
                    buffer.truncate(written);
                    self.kv.attach_snapshot(node, Arc::from(buffer));
                    self.kv_stats.swaps += 1;
                }
            }
        }
        self.backend.seq_rm(slot, -1, -1)?;
        self.kv.detach_slot(node);
        if self.kv.snapshot(node).is_none() {
            // No snapshot: the node is gone entirely (recompute later).
            self.kv.remove(node);
        }
        self.kv_stats.evictions += 1;
        Ok(true)
    }

    /// Searches the longest usable prefix-cache hit for `prompt`, without
    /// mutating anything: `(node, applied_hit)` with
    /// `0 < applied_hit < prompt.len()`. The applied hit is always a *strict*
    /// prefix of the prompt: the last prompt token is never copied, it is
    /// prefilled - the backend refuses to resubmit a token whose position is
    /// already resident, and re-feeding it would double-advance a hybrid's
    /// recurrent state.
    fn hit_candidate(&self, prompt: &[i32]) -> Result<Option<(usize, usize)>, InferError> {
        if !self.options.prefix_cache || prompt.len() < 2 {
            return Ok(None);
        }
        // A mid-sequence hit (applied < node depth) needs a partial-range copy
        // of [0, applied): only on a unified pool (a partitioned pool asserts
        // on cross-stream partial copies) and only for a dense model (a
        // hybrid's recurrent state cannot be truncated mid-sequence).
        let partial_ok = self.backend.kv_unified()? && !self.backend.is_hybrid()?;
        let last = prompt.len() - 1;
        let (kv, backend) = (&self.kv, &self.backend);
        let mut usable = |node: usize, hit: usize| -> bool {
            // The applied hit is the raw hit clamped to a strict prefix.
            let applied = hit.min(last);
            let depth = kv.depth(node);
            if applied == 0 || (applied < depth && !partial_ok) {
                return false;
            }
            match kv.slot(node) {
                // The resident span must be frozen at the node's boundary,
                // which covers [0, applied) since applied <= depth.
                Some(slot) => backend
                    .seq_pos(slot)
                    .is_ok_and(|span| span == (0, depth as i32 - 1)),
                // Swapped node: restorable from its host snapshot, but only at
                // its exact boundary (a load restores the whole span, and the
                // suffix prefill must not overlap it).
                None => kv.snapshot(node).is_some() && applied == depth,
            }
        };
        Ok(kv
            .longest_prefix(prompt, partial_ok, &mut usable)
            .map(|(node, hit)| (node, hit.min(last))))
    }

    /// Applies the longest usable prefix-cache hit for `prompt` into the
    /// (empty) `leader_slot`. Returns the applied hit length (0 = cold); the
    /// applied hit is always `< prompt.len()`, so the last prompt token is
    /// prefilled by the normal prefill phase.
    fn apply_cache_hit(&mut self, prompt: &[i32], leader_slot: i32) -> Result<usize, InferError> {
        let Some((node, hit)) = self.hit_candidate(prompt)? else {
            return Ok(0);
        };
        debug_assert!(hit < prompt.len());
        match self.kv.slot(node) {
            Some(src) if hit == self.kv.depth(node) => {
                // Exact boundary: whole-slot copy (safe on any pool mode).
                self.backend.seq_cp(src, leader_slot, -1, -1)?;
            }
            Some(src) => {
                // Mid-sequence: partial-range copy of positions [0, hit)
                // (`p1` is exclusive in the backend's seq_cp).
                self.backend.seq_cp(src, leader_slot, 0, hit as i32)?;
            }
            None => {
                // Swapped node at its exact boundary: restore the full state
                // into the slot.
                let snapshot = self
                    .kv
                    .snapshot(node)
                    .cloned()
                    .expect("a usable swapped node carries a snapshot");
                self.backend.seq_load(leader_slot, 0, &snapshot)?;
            }
        }
        self.kv.touch(node);
        Ok(hit)
    }

    /// Places one group (one request, or a merged run of identical prompts)
    /// into free slots and reserves its cells.
    fn admit_group(&mut self, group: Vec<Queued>, n_ctx_seq: usize) -> Result<(), InferError> {
        let prompt = group[0].request.prompt.clone();
        let n_members: usize = group.iter().map(|queued| queued.request.n()).sum();
        let prefix_end = if n_members > 1 {
            prompt.len() - 1
        } else {
            prompt.len()
        };
        let mut members = Vec::with_capacity(n_members);
        // Slots already taken, including the ones this group is taking: the
        // group is pushed to `live` only after every member is placed.
        let mut used = vec![false; self.n_slots];
        for group in &self.live {
            for member in &group.members {
                if member.live && (member.slot as usize) < self.n_slots {
                    used[member.slot as usize] = true;
                }
            }
        }
        // Dormant prefix-tree slots are taken as well.
        for slot in self.kv.resident_slots() {
            if (slot as usize) < self.n_slots {
                used[slot as usize] = true;
            }
        }
        for queued in group {
            let Queued {
                request,
                events,
                cancelled,
                permit: _permit,
            } = queued;
            let unparse_special = request.unparse_special;
            for (index, sampling) in request.sampling.into_iter().enumerate() {
                let slot = used
                    .iter()
                    .position(|free| !*free)
                    // The backend's slot count fits i32, checked at startup.
                    .map(|slot| slot as i32)
                    .expect("admission checked the available slot count");
                used[slot as usize] = true;
                // Startup and completion cleanup leave every free slot empty.
                let request_id = self.next_request_id;
                // IDs are reused only after wraparound, skipping all live IDs.
                self.next_request_id = self.next_request_id.checked_add(1).unwrap_or(1);
                while self
                    .live
                    .iter()
                    .flat_map(|group| &group.members)
                    .chain(members.iter())
                    .any(|member: &LiveMember| {
                        member.live && member.request_id == self.next_request_id
                    })
                    || self.next_request_id == request_id
                {
                    self.next_request_id = self.next_request_id.checked_add(1).unwrap_or(1);
                }
                let limit = (sampling.max_new_tokens as usize).min(n_ctx_seq - prompt.len());
                members.push(LiveMember {
                    slot,
                    request_id,
                    index,
                    sampling,
                    want_logprob: request.logprobs || request.n_top_logprobs > 0,
                    limit,
                    // Set once the cache hit is known (the shared prefix is
                    // not re-allocated).
                    reserved_share: 0,
                    generated: 0,
                    next_token: 0,
                    history: Vec::new(),
                    live: true,
                    n_top_logprobs: request.n_top_logprobs,
                    unparse_special,
                    detok: Detokenizer::new(&request.stop),
                    events: Some(events.clone()),
                    cancelled: cancelled.clone(),
                });
            }
        }
        if self.options.mode == AdmissionMode::Lockstep {
            let mut slots: Vec<i32> = members.iter().map(|member| member.slot).collect();
            slots.sort_unstable();
            members.sort_by(|a, b| sampling_cmp(&a.sampling, &b.sampling));
            for (member, slot) in members.iter_mut().zip(slots) {
                member.slot = slot;
            }
        }
        // Prefix-cache hit into the leader's slot: the shared prefix is copied
        // (or restored from a snapshot) instead of prefilled.
        let leader_slot = members[0].slot;
        let cache_hit = self.apply_cache_hit(&prompt, leader_slot)?;
        let prefill_from = cache_hit.min(prefix_end);
        // The shared prefix [0, cache_hit) is already resident in the tree's
        // node: only the suffix and the budgets are new cells.
        for member in &mut members {
            member.reserved_share = ((prompt.len() - cache_hit) + member.limit) as u64;
        }
        let reserved = members.iter().map(|member| member.reserved_share).sum();
        if cache_hit > 0 {
            self.kv_stats.hits += 1;
        } else {
            self.kv_stats.misses += 1;
        }
        self.kv_stats.prompt_cells += prompt.len() as u64;
        self.kv_stats.prefilled_cells += (prefix_end - prefill_from) as u64;
        // The applied hit is always a strict prefix of the prompt, so the
        // prefill phase always has at least the last prompt token to decode
        // (and sample) - no replay of a resident position is ever needed.
        let phase = GroupPhase::Prefill { pos: prefill_from };
        self.live.push(LiveGroup {
            prompt,
            prefix_end,
            leader: 0,
            reserved,
            members,
            phase,
        });
        Ok(())
    }

    /// Emits one sampled token to its member's stream and applies the stop
    /// criteria (stop string, EOG, budget, context).
    fn emit_sampled(
        &mut self,
        group: usize,
        member: usize,
        produced: StepOutRow,
        row_index: usize,
    ) -> Result<(), InferError> {
        if self.live[group].members[member]
            .cancelled
            .load(AtomicOrdering::SeqCst)
        {
            return self.finish_member(group, member, FinishReason::Cancelled);
        }
        if !produced.sampled {
            return Err(InferError::Failed(
                "a sampling step row produced no token".into(),
            ));
        }
        let (stopped, send_failed) = {
            let group_ref = &mut self.live[group];
            let member_ref = &mut group_ref.members[member];
            member_ref.generated += 1;
            member_ref.next_token = produced.token;
            member_ref.history.push(produced.token);
            // The EOG token is carried in the token events but contributes no
            // text - with special tokens unparsed its piece would leak into
            // the answer (without, the backend renders it empty already).
            let piece =
                if member_ref.unparse_special && self.backend.is_eog_token(produced.token)? {
                    Vec::new()
                } else {
                    self.backend.detokenize_bytes(
                        std::slice::from_ref(&produced.token),
                        member_ref.unparse_special,
                    )?
                };
            let (text, stop_hit) = member_ref.detok.push_bytes(&piece);
            let top_logprobs = if member_ref.n_top_logprobs > 0 {
                self.out
                    .top(row_index)
                    .map(|(token, logprob)| {
                        // The piece of each top entry, read back the same way
                        // as the sampled token's own piece.
                        let piece = self
                            .backend
                            .detokenize_bytes(
                                std::slice::from_ref(&token),
                                member_ref.unparse_special,
                            )
                            .unwrap_or_default();
                        TopLogprob {
                            token,
                            logprob,
                            piece,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let event = Event::Token(TokenEvent {
                index: member_ref.index,
                token: produced.token,
                logprob: produced.logprob,
                sampled_logprob: member_ref.want_logprob.then_some(produced.sampled_logprob),
                forced: produced.forced,
                text,
                piece,
                top_logprobs,
            });
            // Bounded backpressure: a full stream is waited on for at most
            // `slow_client_timeout`, then the request is cancelled.
            let send_failed = send_bounded(
                member_ref
                    .events
                    .as_ref()
                    .expect("live member has a sender"),
                event,
                self.options.slow_client_timeout,
            )
            .is_err();
            (stop_hit, send_failed)
        };
        let reason = if send_failed {
            Some(FinishReason::Cancelled)
        } else if stopped {
            Some(FinishReason::Stop)
        } else if self.backend.is_eog_token(produced.token)? {
            Some(FinishReason::Eog)
        } else {
            let member_ref = &self.live[group].members[member];
            if member_ref.generated >= member_ref.limit {
                if member_ref.limit >= member_ref.sampling.max_new_tokens as usize {
                    Some(FinishReason::MaxTokens)
                } else {
                    Some(FinishReason::ContextFull)
                }
            } else {
                None
            }
        };
        if let Some(reason) = reason {
            self.finish_member(group, member, reason)?;
        }
        Ok(())
    }

    /// Ends one member: final event, slot returned to the pool (or handed to
    /// the prefix tree as a dormant node), per-request backend state released.
    fn finish_member(
        &mut self,
        group: usize,
        member: usize,
        reason: FinishReason,
    ) -> Result<(), InferError> {
        let (slot, request_id, index, text, events, path) = {
            let group_ref = &mut self.live[group];
            let member_ref = &mut group_ref.members[member];
            if !member_ref.live {
                return Ok(());
            }
            group_ref.reserved = group_ref.reserved.saturating_sub(member_ref.reserved_share);
            member_ref.live = false;
            let text = member_ref.detok.finish();
            let mut path = group_ref.prompt.clone();
            path.extend_from_slice(&member_ref.history);
            (
                member_ref.slot,
                member_ref.request_id,
                member_ref.index,
                text,
                member_ref.events.take().expect("live member has a sender"),
                path,
            )
        };
        // A timed-out stream is cancelled for every completion of its request.
        if reason == FinishReason::Cancelled {
            self.live[group].members[member]
                .cancelled
                .store(true, AtomicOrdering::SeqCst);
        }
        // Best effort: on cancellation the receiver may already be gone.
        if reason != FinishReason::Cancelled
            && send_bounded(
                &events,
                Event::End {
                    index,
                    reason,
                    text,
                },
                self.options.slow_client_timeout,
            )
            .is_err()
        {
            self.live[group].members[member]
                .cancelled
                .store(true, AtomicOrdering::SeqCst);
        }
        // Hand the finished sequence's slot to the prefix tree as a dormant
        // node, instead of releasing it - a later request sharing this prefix
        // will seq_cp from it. The node's depth is the *resident* span: the
        // last sampled token is never fed back, so it is not in the KV and
        // the node covers the first `resident` tokens only. A slot whose span
        // does not start at 0 (SWA evicted the low positions) cannot be
        // shared from zero and is released instead.
        let (pos_min, pos_max) = self.backend.seq_pos(slot)?;
        let resident = if pos_min == 0 {
            (pos_max + 1) as usize
        } else {
            0
        };
        if self.options.prefix_cache
            && resident >= self.options.min_prefix_len
            && resident <= path.len()
        {
            let (_, attached) = self.kv.insert(&path[..resident], slot);
            if !attached {
                // The exact path already has a node: keep the existing one.
                self.backend.seq_rm(slot, -1, -1)?;
            }
        } else {
            self.backend.seq_rm(slot, -1, -1)?;
        }
        self.backend.request_free(request_id)?;
        Ok(())
    }

    /// Cancellation: a dropped stream removes the request at the next step.
    fn remove_cancelled(&mut self) -> Result<(), InferError> {
        for group in 0..self.live.len() {
            // Preserve a partially decoded shared prefix before returning its
            // cancelled leader's slot. The replacement is still empty here.
            let g = &mut self.live[group];
            if let GroupPhase::Prefill { pos } = g.phase
                && g.members[g.leader].cancelled.load(AtomicOrdering::SeqCst)
                && let Some(replacement) = g.members.iter().position(|member| {
                    member.live && !member.cancelled.load(AtomicOrdering::SeqCst)
                })
            {
                if pos > 0 {
                    self.backend.seq_cp(
                        g.members[g.leader].slot,
                        g.members[replacement].slot,
                        -1,
                        -1,
                    )?;
                }
                g.leader = replacement;
            }
            let cancelled: Vec<usize> = self.live[group]
                .members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.live && member.cancelled.load(AtomicOrdering::SeqCst))
                .map(|(mi, _)| mi)
                .collect();
            for member in cancelled {
                self.finish_member(group, member, FinishReason::Cancelled)?;
            }
        }
        self.live
            .retain(|group| group.members.iter().any(|member| member.live));
        Ok(())
    }

    /// After a failed step: clear every touched slot (a prefix of the
    /// submitted tokens may be resident) and release the request states.
    fn rollback(&mut self, kinds: &[RowKind]) {
        for kind in kinds {
            let (group, member) = match kind {
                RowKind::Decode { group, member } | RowKind::Terminal { group, member } => {
                    (*group, *member)
                }
                RowKind::Prefill { group, .. } => (*group, self.live[*group].leader),
            };
            let (slot, request_id) = {
                let member_ref = &self.live[group].members[member];
                (member_ref.slot, member_ref.request_id)
            };
            let _ = self.backend.seq_rm(slot, -1, -1);
            let _ = self.backend.request_free(request_id);
        }
    }

    /// A step that failed while the pool is saturated is a capacity
    /// condition, not a caller mistake: keep it a structured variant.
    fn classify_step_error(&self, error: InferError, needed: u64) -> InferError {
        if matches!(error, InferError::Backend(_))
            && let Ok(memory) = self.backend.memory()
        {
            let available = memory.cells_total.saturating_sub(memory.cells_used);
            if needed > available {
                return InferError::KvFull { needed, available };
            }
        }
        error
    }

    /// A fatal error for every live request: broadcast, then release. Slots
    /// of already-finished members belong to the prefix tree now and are left
    /// alone.
    fn fail_all(&mut self, error: &InferError) {
        let mut to_release: Vec<(i32, i32)> = Vec::new();
        for group in 0..self.live.len() {
            for member in 0..self.live[group].members.len() {
                let member_ref = &mut self.live[group].members[member];
                if member_ref.live {
                    if let Some(events) = member_ref.events.take() {
                        let _ = send_bounded(
                            &events,
                            Event::Error(error.clone()),
                            self.options.slow_client_timeout,
                        );
                    }
                    member_ref.live = false;
                    to_release.push((member_ref.slot, member_ref.request_id));
                }
            }
        }
        for (slot, request_id) in to_release {
            let _ = self.backend.seq_rm(slot, -1, -1);
            let _ = self.backend.request_free(request_id);
        }
        self.live.clear();
    }
}

/// What crosses the actor's inbox.
pub(crate) enum Envelope {
    Submit(Queued),
    /// Wakes a thread blocked on an empty queue (shutdown).
    Wake,
    /// Scheduler + backend stats query.
    Stats(mpsc::Sender<SchedulerStats>),
    /// Pin the node at exactly `prefix`.
    Pin {
        prefix: Vec<i32>,
        reply: mpsc::Sender<Result<PinId, InferError>>,
    },
    /// Release one pin.
    Unpin {
        id: PinId,
        reply: mpsc::Sender<bool>,
    },
    /// Register a named adapter on the backend (without applying it).
    LoadAdapter {
        name: String,
        path: PathBuf,
        reply: mpsc::Sender<Result<(), InferError>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventStream, FakeBackend, Poll, fake_sampling, fake_sampling_v2};
    use std::sync::mpsc;

    fn scheduler(n_batch: usize) -> Scheduler<FakeBackend> {
        let backend = FakeBackend::new(4, 32, 128, 100, 99, vec!["x".into(); 100]);
        let mut scheduler = Scheduler::new(
            backend,
            SchedulerOptions {
                mode: AdmissionMode::Continuous,
                n_batch,
                slow_client_timeout: Duration::ZERO,
                // These tests exercise scheduling, not the prefix cache: keep
                // every finished slot released.
                prefix_cache: false,
                ..SchedulerOptions::default()
            },
        );
        scheduler.n_slots = 4;
        scheduler
    }

    fn enqueue<B: DecodeBackend>(scheduler: &mut Scheduler<B>, request: Request) -> EventStream {
        let (events, rx) = mpsc::sync_channel(256);
        let cancelled = Arc::new(AtomicBool::new(false));
        scheduler.enqueue(Queued {
            request,
            events,
            cancelled: cancelled.clone(),
            permit: None,
        });
        EventStream::new(rx, cancelled)
    }

    fn drain(scheduler: &mut Scheduler<FakeBackend>) {
        for _ in 0..200 {
            if scheduler.live.is_empty() && scheduler.queue.is_empty() {
                assert_eq!(scheduler.backend.memory().unwrap().cells_used, 0);
                return;
            }
            scheduler.step_once().unwrap();
        }
        panic!("scheduler did not finish");
    }

    fn finished(stream: &EventStream, tokens: usize) {
        let mut count = 0;
        let mut ended = false;
        loop {
            match stream.poll() {
                Poll::Ready(Event::Token(_)) => count += 1,
                Poll::Ready(Event::End {
                    reason: FinishReason::MaxTokens,
                    ..
                }) => ended = true,
                Poll::Closed => break,
                other => panic!("unexpected stream state: {other:?}"),
            }
        }
        assert_eq!(count, tokens);
        assert!(ended);
    }

    struct FailAfterDecode {
        inner: FakeBackend,
        fail_next: bool,
    }

    impl DecodeBackend for FailAfterDecode {
        fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), InferError> {
            self.inner.step(rows, out)?;
            if self.fail_next {
                self.fail_next = false;
                return Err(retrograd_engine::EngineError::Runtime(
                    "partial decode failure".into(),
                )
                .into());
            }
            Ok(())
        }
        fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), InferError> {
            self.inner.seq_rm(slot, p0, p1)
        }
        fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), InferError> {
            self.inner.seq_cp(src, dst, p0, p1)
        }
        fn seq_pos(&self, slot: i32) -> Result<(i32, i32), InferError> {
            self.inner.seq_pos(slot)
        }
        fn seq_save_len(&self, slot: i32, flags: u32) -> Result<usize, InferError> {
            self.inner.seq_save_len(slot, flags)
        }
        fn seq_save_into(
            &mut self,
            slot: i32,
            flags: u32,
            buffer: &mut [u8],
        ) -> Result<usize, InferError> {
            self.inner.seq_save_into(slot, flags, buffer)
        }
        fn seq_load(&mut self, slot: i32, flags: u32, data: &[u8]) -> Result<usize, InferError> {
            self.inner.seq_load(slot, flags, data)
        }
        fn kv_unified(&self) -> Result<bool, InferError> {
            self.inner.kv_unified()
        }
        fn weights_version(&self) -> Result<u64, InferError> {
            self.inner.weights_version()
        }
        fn model_fingerprint(&self) -> Result<u64, InferError> {
            self.inner.model_fingerprint()
        }
        fn is_hybrid(&self) -> Result<bool, InferError> {
            self.inner.is_hybrid()
        }
        fn memory(&self) -> Result<retrograd_engine::EngineMemory, InferError> {
            self.inner.memory()
        }
        fn request_free(&mut self, request: i32) -> Result<(), InferError> {
            self.inner.request_free(request)
        }
        fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
            self.inner.is_eog_token(token)
        }
        fn detokenize_bytes(
            &self,
            tokens: &[i32],
            unparse_special: bool,
        ) -> Result<Vec<u8>, InferError> {
            self.inner.detokenize_bytes(tokens, unparse_special)
        }
        fn context_size(&self) -> Result<u32, InferError> {
            self.inner.context_size()
        }
        fn vocab_size(&self) -> Result<u32, InferError> {
            self.inner.vocab_size()
        }
    }

    #[test]
    fn partial_step_failure_keeps_capacity_typed_and_clears_kv_before_reuse() {
        let backend = FailAfterDecode {
            inner: FakeBackend::new(1, 32, 3, 100, 99, vec!["x".into(); 100]),
            fail_next: true,
        };
        let mut scheduler = Scheduler::new(backend, SchedulerOptions::default());
        let stream = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(1, 1)),
        );
        let error = scheduler.step_once().unwrap_err();
        assert!(matches!(
            error,
            InferError::KvFull {
                needed: 2,
                available: 1
            }
        ));
        scheduler.fail_all(&error);
        assert_eq!(scheduler.backend.memory().unwrap().cells_used, 0);
        assert!(matches!(
            stream.collect::<Vec<_>>().as_slice(),
            [Event::Error(InferError::KvFull { .. })]
        ));
        let stream = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(2, 1)),
        );
        scheduler.step_once().unwrap();
        finished(&stream, 1);
        assert_eq!(scheduler.backend.memory().unwrap().cells_used, 0);
    }

    #[test]
    fn a_finished_member_closes_its_stream_and_returns_its_slot_before_its_group_ends() {
        let mut scheduler = scheduler(8);
        scheduler.options.mode = AdmissionMode::Lockstep;
        let short = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(1, 1)),
        );
        let long = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(2, 6)),
        );
        scheduler.step_once().unwrap(); // Shared prefix.
        scheduler.step_once().unwrap(); // Both terminal tokens.
        finished(&short, 1);
        scheduler.options.mode = AdmissionMode::Continuous;
        let mut request = Request::single(vec![3], fake_sampling(3, 1));
        request.sampling = vec![fake_sampling_v2(3, 1); 3];
        let next = enqueue(&mut scheduler, request);
        scheduler.step_once().unwrap();
        assert!(
            scheduler.queue.is_empty(),
            "all three returned/free slots can be reused"
        );
        drain(&mut scheduler);
        finished(&long, 6);
        assert_eq!(
            next.filter(|event| matches!(event, Event::End { .. }))
                .count(),
            3
        );
    }

    #[test]
    fn cancelling_the_prefill_leader_preserves_the_shared_prefix() {
        let mut scheduler = scheduler(1);
        scheduler.options.mode = AdmissionMode::Lockstep;
        let first = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2, 3, 4], fake_sampling(1, 3)),
        );
        let second = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2, 3, 4], fake_sampling(2, 3)),
        );
        scheduler.step_once().unwrap();
        drop(first);
        drain(&mut scheduler);
        finished(&second, 3);
    }

    #[test]
    fn cancelling_an_unsampled_terminal_member_never_decodes_its_returned_slot() {
        let mut scheduler = scheduler(1);
        scheduler.options.mode = AdmissionMode::Lockstep;
        let first = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(1, 3)),
        );
        let second = enqueue(
            &mut scheduler,
            Request::single(vec![1, 2], fake_sampling(2, 3)),
        );
        scheduler.step_once().unwrap();
        scheduler.step_once().unwrap();
        drop(second);
        drain(&mut scheduler);
        finished(&first, 3);
        assert_eq!(scheduler.backend.request_order.len(), 1);
    }

    #[test]
    fn single_token_prompts_and_more_members_than_batch_tokens_make_progress() {
        let mut scheduler = scheduler(1);
        let mut request = Request::single(vec![1], fake_sampling(1, 3));
        request.sampling = vec![fake_sampling_v2(1, 3); 3];
        let stream = enqueue(&mut scheduler, request);
        drain(&mut scheduler);
        let events: Vec<_> = stream.collect();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Token(_)))
                .count(),
            9
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::End { .. }))
                .count(),
            3
        );
        assert!(
            scheduler.backend.copies.is_empty(),
            "there is no shared prefix to copy"
        );
    }

    #[test]
    fn prefills_progress_while_a_decode_uses_the_entire_batch() {
        let mut scheduler = scheduler(1);
        let first = enqueue(
            &mut scheduler,
            Request::single(vec![1], fake_sampling(1, 20)),
        );
        scheduler.step_once().unwrap();
        let second = enqueue(
            &mut scheduler,
            Request::single(vec![2, 3], fake_sampling(2, 2)),
        );
        for _ in 0..8 {
            scheduler.step_once().unwrap();
        }
        finished(&second, 2);
        assert!(
            scheduler
                .live
                .iter()
                .flat_map(|group| &group.members)
                .any(|member| member.live)
        );
        drain(&mut scheduler);
        finished(&first, 20);
    }

    #[test]
    fn smaller_arrivals_cannot_overtake_a_request_waiting_for_multiple_slots() {
        let mut scheduler = scheduler(4);
        let first = enqueue(
            &mut scheduler,
            Request::single(vec![1], fake_sampling(1, 8)),
        );
        scheduler.step_once().unwrap();
        let mut request = Request::single(vec![2], fake_sampling(2, 2));
        request.sampling = vec![fake_sampling_v2(2, 2); 4];
        let large = enqueue(&mut scheduler, request);
        let small = enqueue(
            &mut scheduler,
            Request::single(vec![3], fake_sampling(3, 1)),
        );
        scheduler.step_once().unwrap();
        assert_eq!(
            scheduler.queue.len(),
            2,
            "the smaller arrival waits its turn"
        );
        drain(&mut scheduler);
        finished(&first, 8);
        assert_eq!(
            large
                .filter(|event| matches!(event, Event::End { .. }))
                .count(),
            4
        );
        finished(&small, 1);
    }

    #[test]
    fn a_cancelled_queued_request_never_reaches_the_backend() {
        let mut scheduler = scheduler(4);
        let stream = enqueue(
            &mut scheduler,
            Request::single(vec![1], fake_sampling(1, 3)),
        );
        drop(stream);
        scheduler.step_once().unwrap();
        assert!(scheduler.backend.steps.is_empty());
    }
}
