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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::{Duration, Instant};

use retrograd_core::SamplingParams;
use retrograd_engine::{StepOutRow, StepOutput, StepRow};

use crate::backend::DecodeBackend;
use crate::detok::Detokenizer;
use crate::error::InferError;
use crate::request::{Event, FinishReason, Request, TokenEvent};

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
}

impl Default for SchedulerOptions {
    fn default() -> Self {
        Self {
            mode: AdmissionMode::Lockstep,
            n_batch: 512,
            queue_capacity: 64,
            event_capacity: 64,
            slow_client_timeout: Duration::from_secs(5),
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
    sampling: SamplingParams,
    /// `min(max_new_tokens, n_ctx_seq - prompt.len())`: the stop budget.
    limit: usize,
    generated: usize,
    next_token: i32,
    live: bool,
    n_top_logprobs: u32,
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
    /// Cells reserved for the whole group (prompt + budgets).
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
fn member_need(prompt_len: usize, sampling: &SamplingParams, n_ctx_seq: usize) -> Option<u64> {
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

fn sampling_cmp(a: &SamplingParams, b: &SamplingParams) -> Ordering {
    a.seed
        .cmp(&b.seed)
        .then_with(|| a.max_new_tokens.cmp(&b.max_new_tokens))
        .then_with(|| a.temperature.total_cmp(&b.temperature))
        .then_with(|| a.top_p.total_cmp(&b.top_p))
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
        }
    }

    pub(crate) fn enqueue(&mut self, queued: Queued) {
        self.queue.push_back(queued);
    }

    /// The actor's main loop, on the thread that owns the backend.
    pub(crate) fn run(
        &mut self,
        inbox: Receiver<Envelope>,
        shutdown: Arc<AtomicBool>,
    ) -> Result<(), InferError> {
        // The scheduler owns placement: start from empty slots, so residue
        // left by earlier batched generation calls cannot eat the budget.
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
        let mut draining = false;
        loop {
            // Pull pending envelopes while the admission queue has room: a
            // full internal queue leaves the bounded inbox full, which is
            // what refuses further submits with QueueFull.
            while self.queue.len() < self.options.queue_capacity {
                match inbox.try_recv() {
                    Ok(Envelope::Submit(queued)) => self.enqueue(queued),
                    Ok(Envelope::Wake) => {}
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
                    if let Envelope::Submit(queued) = envelope {
                        self.enqueue(queued);
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
                    Err(_) => return Ok(()),
                }
                continue;
            }
            if let Err(error) = self.step_once() {
                self.fail_all(&error);
            }
        }
    }

    /// One scheduler iteration: remove cancelled sequences, admit, compose
    /// and execute one step, emit, remove finished sequences.
    fn step_once(&mut self) -> Result<(), InferError> {
        self.queue
            .retain(|queued| !queued.cancelled.load(AtomicOrdering::SeqCst));
        self.remove_cancelled()?;
        self.admit()?;
        self.advance_phases()?;
        if self.live.is_empty() {
            return Ok(());
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
            return Ok(());
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
        Ok(())
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
                sampling: member.sampling,
                n_top_logprobs: member.n_top_logprobs,
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
                        sampling: leader.sampling,
                        n_top_logprobs: if sampled_member.is_some() {
                            leader.n_top_logprobs
                        } else {
                            0
                        },
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
                            sampling: member.sampling,
                            n_top_logprobs: member.n_top_logprobs,
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
            let need = reservation(&queued.request, n_ctx_seq);
            if prompt_too_long
                || too_many_members
                || top_too_wide
                || invalid_token
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

        let used_slots: usize = self
            .live
            .iter()
            .flat_map(|group| &group.members)
            .filter(|member| member.live)
            .count();
        let reserved: u64 = self.live.iter().map(|group| group.reserved).sum();
        let committed = memory.cells_used.max(reserved);
        let mut slots_left = self.n_slots.saturating_sub(used_slots);
        let mut cells_left = cells_total.saturating_sub(committed);

        match self.options.mode {
            AdmissionMode::Lockstep => {
                // A wave is admitted only when nothing is live.
                if !self.live.is_empty() {
                    return Ok(());
                }
                // Select the oldest fitting wave before sorting it. Sorting
                // the whole queue first would let a stream of short prompts
                // overtake an older request forever.
                let mut pending = Vec::new();
                let (mut wave_slots, mut wave_cells) = (slots_left, cells_left);
                while let Some(queued) = self.queue.front() {
                    let members = queued.request.n();
                    let need = reservation(&queued.request, n_ctx_seq);
                    if members > wave_slots || need > wave_cells {
                        break;
                    }
                    wave_slots -= members;
                    wave_cells -= need;
                    pending.push(self.queue.pop_front().expect("queue has a front"));
                }
                pending.sort_by(|a, b| canonical_cmp(&a.request, &b.request));
                while !pending.is_empty() {
                    let mut run_end = 1;
                    while run_end < pending.len()
                        && pending[run_end].request.prompt == pending[0].request.prompt
                    {
                        run_end += 1;
                    }
                    let run = pending.drain(..run_end).collect();
                    self.admit_group(run, n_ctx_seq, &mut slots_left, &mut cells_left)?;
                }
            }
            AdmissionMode::Continuous => {
                // Do not let smaller arrivals indefinitely overtake a request
                // waiting for several slots or a larger cell reservation.
                while let Some(queued) = self.queue.front() {
                    if queued.request.n() > slots_left
                        || reservation(&queued.request, n_ctx_seq) > cells_left
                    {
                        break;
                    }
                    let queued = self.queue.pop_front().expect("queue has a front");
                    self.admit_group(vec![queued], n_ctx_seq, &mut slots_left, &mut cells_left)?;
                }
            }
        }
        Ok(())
    }

    /// Places one group (one request, or a merged run of identical prompts)
    /// into free slots and reserves its cells.
    fn admit_group(
        &mut self,
        group: Vec<Queued>,
        n_ctx_seq: usize,
        slots_left: &mut usize,
        cells_left: &mut u64,
    ) -> Result<(), InferError> {
        let prompt = group[0].request.prompt.clone();
        let n_members: usize = group.iter().map(|queued| queued.request.n()).sum();
        let prefix_end = if n_members > 1 {
            prompt.len() - 1
        } else {
            prompt.len()
        };
        let reserved: u64 = group
            .iter()
            .map(|queued| reservation(&queued.request, n_ctx_seq))
            .sum();
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
        for queued in group {
            let Queued {
                request,
                events,
                cancelled,
                permit: _permit,
            } = queued;
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
                    limit,
                    generated: 0,
                    next_token: 0,
                    live: true,
                    n_top_logprobs: request.n_top_logprobs,
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
        *slots_left -= n_members;
        *cells_left = cells_left.saturating_sub(reserved);
        self.live.push(LiveGroup {
            prompt,
            prefix_end,
            leader: 0,
            reserved,
            members,
            phase: GroupPhase::Prefill { pos: 0 },
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
            let piece = self
                .backend
                .detokenize_bytes(std::slice::from_ref(&produced.token))?;
            let (text, stop_hit) = member_ref.detok.push_bytes(&piece);
            let top_logprobs = if member_ref.n_top_logprobs > 0 {
                self.out.top(row_index).collect()
            } else {
                Vec::new()
            };
            let event = Event::Token(TokenEvent {
                index: member_ref.index,
                token: produced.token,
                logprob: produced.logprob,
                forced: produced.forced,
                text,
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

    /// Ends one member: final event, slot returned to the pool, per-request
    /// backend state released.
    fn finish_member(
        &mut self,
        group: usize,
        member: usize,
        reason: FinishReason,
    ) -> Result<(), InferError> {
        let (slot, request_id, index, text, events) = {
            let group_ref = &mut self.live[group];
            let member_ref = &mut group_ref.members[member];
            if !member_ref.live {
                return Ok(());
            }
            group_ref.reserved -= (group_ref.prompt.len() + member_ref.limit) as u64;
            member_ref.live = false;
            let text = member_ref.detok.finish();
            (
                member_ref.slot,
                member_ref.request_id,
                member_ref.index,
                text,
                member_ref.events.take().expect("live member has a sender"),
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
        self.backend.seq_rm(slot, -1, -1)?;
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

    /// A fatal error for every live request: broadcast, then release.
    fn fail_all(&mut self, error: &InferError) {
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
                }
            }
        }
        for group in self.live.drain(..) {
            for member in group.members {
                let _ = self.backend.seq_rm(member.slot, -1, -1);
                let _ = self.backend.request_free(member.request_id);
            }
        }
    }
}

/// What crosses the actor's inbox.
pub(crate) enum Envelope {
    Submit(Queued),
    /// Wakes a thread blocked on an empty queue (shutdown).
    Wake,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventStream, FakeBackend, Poll, fake_sampling};
    use std::sync::mpsc;

    fn scheduler(n_batch: usize) -> Scheduler<FakeBackend> {
        let backend = FakeBackend::new(4, 32, 128, 100, 99, vec!["x".into(); 100]);
        let mut scheduler = Scheduler::new(
            backend,
            SchedulerOptions {
                mode: AdmissionMode::Continuous,
                n_batch,
                slow_client_timeout: Duration::ZERO,
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
        fn memory(&self) -> Result<retrograd_engine::EngineMemory, InferError> {
            self.inner.memory()
        }
        fn request_free(&mut self, request: i32) -> Result<(), InferError> {
            self.inner.request_free(request)
        }
        fn is_eog_token(&self, token: i32) -> Result<bool, InferError> {
            self.inner.is_eog_token(token)
        }
        fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>, InferError> {
            self.inner.detokenize_bytes(tokens)
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
        request.sampling = vec![fake_sampling(3, 1); 3];
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
        request.sampling = vec![fake_sampling(1, 3); 3];
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
        request.sampling = vec![fake_sampling(2, 2); 4];
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
