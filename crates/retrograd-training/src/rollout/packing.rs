//! Turning weighted rollouts into physical optimizer batches: fixed-width rows
//! and the shared-prefix multi-sequence packing, both reusing scratch buffers
//! across steps.

use retrograd_core::{Error, Result, WeightedBatch};
use retrograd_engine::PackedSequenceBatch;

use super::sampling::{Rollout, RowLayout};
use super::step::ChunkMember;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackRefusal {
    FanoutDoesNotFit { required: usize, available: usize },
    DivergentPrefix,
    InvalidLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackOutcome {
    Packed {
        useful_tokens: usize,
        shared_tokens: usize,
    },
    Refused(PackRefusal),
}

/// The same compatible runs drive the size check and the physical packer.
fn compatible_run_end(chunk: &[ChunkMember<'_>], start: usize) -> Result<usize> {
    let first = &chunk[start];
    let first_train = first.rollout.first_train_index()?;
    let shared = &first.rollout.tokens[..first_train - 1];
    let mut end = start + 1;
    while end < chunk.len() {
        let candidate = &chunk[end];
        if candidate.group_id != first.group_id
            || candidate.rollout.first_train_index()? != first_train
            || candidate.rollout.tokens.get(..shared.len()) != Some(shared)
        {
            break;
        }
        end += 1;
    }
    Ok(end)
}

/// Exact width, including isolated tokens for unused sequence slots. No
/// context-sized buffers or weights are built while choosing a geometry.
pub(super) fn packed_width(
    chunk: &[ChunkMember<'_>],
    layout: &RowLayout,
) -> Result<std::result::Result<usize, PackRefusal>> {
    if chunk.is_empty() || chunk.len() > layout.n_seq_max {
        return Ok(Err(PackRefusal::InvalidLayout));
    }
    let mut required = layout.n_seq_max - chunk.len();
    let mut start = 0;
    while start < chunk.len() {
        let first_train = chunk[start].rollout.first_train_index()?;
        if first_train < 2 {
            return Ok(Err(PackRefusal::DivergentPrefix));
        }
        let end = compatible_run_end(chunk, start)?;
        required = required
            .checked_add(first_train - 1)
            .ok_or_else(|| Error::overflow("packed prefix width overflows usize"))?;
        for member in &chunk[start..end] {
            required = required
                .checked_add(member.rollout.training_span_len()?)
                .ok_or_else(|| Error::overflow("packed training width overflows usize"))?;
        }
        start = end;
    }
    Ok(if required > layout.ubatch {
        Err(PackRefusal::FanoutDoesNotFit {
            required,
            available: layout.ubatch,
        })
    } else {
        Ok(required)
    })
}

/// Packs one rollout into a fixed-width training row. Position `i` predicts
/// `tokens[i+1]`; every target token whose mask is true is supervised at its
/// predecessor position, `target - 1`.
#[cfg(test)]
pub(crate) fn pack_row(
    tokens: &[i32],
    train_mask: &[bool],
    token_weights: &[f32],
    n_ctx: usize,
    pad_token: i32,
) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    if tokens.len() != train_mask.len()
        || train_mask.first().copied().unwrap_or(true)
        || train_mask.iter().filter(|&&train| train).count() != token_weights.len()
    {
        return Err(Error::tokenize("rollout does not match its token weights"));
    }
    if tokens.len() > n_ctx {
        return Err(Error::invalid(format!(
            "rollout of {} tokens exceeds the training context {n_ctx}",
            tokens.len()
        )));
    }
    let mut row_tokens = vec![pad_token; n_ctx];
    row_tokens[..tokens.len()].copy_from_slice(tokens);
    let mut labels = vec![-1_i32; n_ctx];
    let mut weights = vec![0.0_f32; n_ctx];
    for ((target, _), &weight) in train_mask
        .iter()
        .enumerate()
        .filter(|(_, train)| **train)
        .zip(token_weights)
    {
        labels[target - 1] = tokens[target];
        weights[target - 1] = weight;
    }
    Ok((row_tokens, labels, weights))
}

/// Reusable buffers for the weighted optimizer steps in PPO/Dr. GRPO.
/// Rows are fixed-width, so re-packing only rewrites each slot's previously
/// touched prefix instead of reallocating context-sized vectors per step.
pub(crate) struct WeightedStepScratch {
    pub(super) token_weights: Vec<f32>,
    pub(super) current_logprobs: Vec<f32>,
    pub(super) suffix_logprobs: Vec<f32>,
    pub(super) batch: WeightedBatch,
    // Leading row positions written by the previous pack of each slot; the
    // label and weight live range always sits inside it, so one length per
    // slot bounds all three buffers.
    pub(super) packed_len: Vec<usize>,
    pub(super) row_width: usize,
    pub(super) pad_token: i32,
    pub(super) member_weights: Vec<Vec<f32>>,
    pub(super) packed_sequence_batch: PackedSequenceBatch,
    pub(crate) packing_fanout: usize,
    pub(crate) packing_passes: usize,
    pub(crate) packing_shared_tokens: usize,
    pub(crate) packing_useful_tokens: usize,
    pub(crate) packing_physical_tokens: usize,
    pub(super) packing_status: Option<String>,
    /// Packing state changes, waiting for the next progress event to carry
    /// them out rather than being printed over the caller's progress bar.
    /// See `Progress::notes`.
    pub(crate) notes: Vec<String>,
}

impl WeightedStepScratch {
    pub(super) fn note_packing(&mut self, status: String, message: String) {
        if self.packing_status.as_ref() != Some(&status) {
            self.notes.push(message);
            self.packing_status = Some(status);
        }
    }

    pub(crate) fn new(layout: &RowLayout) -> Self {
        Self {
            token_weights: Vec::new(),
            current_logprobs: Vec::new(),
            suffix_logprobs: Vec::new(),
            batch: WeightedBatch {
                tokens: Vec::new(),
                labels: Vec::new(),
                weights: Vec::new(),
                n_rows: 0,
                n_ctx: layout.row_width,
                n_topk: 1,
            },
            packed_len: Vec::new(),
            row_width: layout.row_width,
            pad_token: layout.pad_token,
            member_weights: Vec::new(),
            packed_sequence_batch: PackedSequenceBatch {
                tokens: Vec::new(),
                labels: Vec::new(),
                weights: Vec::new(),
                positions: Vec::new(),
                seq_offsets: Vec::new(),
                seq_ids: Vec::new(),
                n_sequences: 0,
                n_topk: 1,
            },
            packing_fanout: 1,
            packing_passes: 0,
            packing_shared_tokens: 0,
            packing_useful_tokens: 0,
            packing_physical_tokens: 0,
            packing_status: None,
            notes: Vec::new(),
        }
    }

    /// Sizes the packed batch for `n_rows` rows. Kept slots retain their
    /// buffers; slots appended after a shrink come back pad-filled, so their
    /// packed prefix restarts at zero.
    pub(super) fn begin(&mut self, n_rows: usize) {
        let len = n_rows * self.row_width;
        if self.batch.tokens.len() < len {
            self.batch.tokens.resize(len, self.pad_token);
            self.batch.labels.resize(len, -1);
            self.batch.weights.resize(len, 0.0);
        } else {
            self.batch.tokens.truncate(len);
            self.batch.labels.truncate(len);
            self.batch.weights.truncate(len);
        }
        self.packed_len.truncate(n_rows);
        self.packed_len.resize(n_rows, 0);
        self.batch.n_rows = n_rows;
    }

    /// Sizes the per-member weight buffers for one optimizer chunk. The
    /// vectors are rewritten in place by `grpo_token_weights_into`, so a chunk
    /// of stable width - the common case - allocates nothing after the first
    /// step.
    pub(super) fn begin_members(&mut self, n_members: usize) {
        self.member_weights.truncate(n_members);
        self.member_weights.resize_with(n_members, Vec::new);
    }

    pub(super) fn pack_row(
        &mut self,
        slot: usize,
        rollout: &Rollout,
        layout: &RowLayout,
        loss_denominator: usize,
    ) -> Result<()> {
        if slot >= self.batch.n_rows {
            return Err(Error::invalid("row slot outside the packed batch"));
        }
        if rollout.tokens.len() != rollout.train_mask.len()
            || rollout.train_mask.first().copied().unwrap_or(true)
            || rollout.completion_len() != self.token_weights.len()
        {
            return Err(Error::tokenize("rollout does not match its token weights"));
        }
        if rollout.tokens.len() > layout.row_width {
            return Err(Error::invalid(format!(
                "rollout of {} tokens exceeds the training context {}",
                rollout.tokens.len(),
                layout.row_width
            )));
        }

        let offset = slot * self.row_width;
        let packed_len = self.packed_len[slot];
        let tokens = &mut self.batch.tokens[offset..offset + self.row_width];
        let labels = &mut self.batch.labels[offset..offset + self.row_width];
        let weights = &mut self.batch.weights[offset..offset + self.row_width];
        if rollout.tokens.len() < packed_len {
            tokens[rollout.tokens.len()..packed_len].fill(layout.pad_token);
        }
        labels[..packed_len].fill(-1);
        weights[..packed_len].fill(0.0);
        tokens[..rollout.tokens.len()].copy_from_slice(&rollout.tokens);
        for ((target, _), &weight) in rollout
            .train_mask
            .iter()
            .enumerate()
            .filter(|(_, train)| **train)
            .zip(&self.token_weights)
        {
            labels[target - 1] = rollout.tokens[target];
            weights[target - 1] = weight;
        }
        self.packed_len[slot] = rollout.tokens.len();
        normalize_runtime_weights(
            &self.batch.labels[offset..offset + self.row_width],
            &mut self.batch.weights[offset..offset + self.row_width],
            loss_denominator,
            layout.batch,
            layout.ubatch,
        )
    }

    /// Packs one or more prompt groups into one physical multi-sequence
    /// ubatch. Within each contiguous group, every prompt token except the
    /// last is one differentiable node shared by all completion sequences.
    /// Different groups use disjoint sequence ids and therefore cannot attend
    /// to one another. Keeping all configured sequence ids present through
    /// isolated padding tokens gives the runtime one stable graph topology.
    pub(super) fn pack_sequences(
        &mut self,
        chunk: &[ChunkMember<'_>],
        member_weight_offset: usize,
        layout: &RowLayout,
        loss_denominator: usize,
    ) -> Result<PackOutcome> {
        if member_weight_offset > self.member_weights.len()
            || chunk.len() > self.member_weights.len() - member_weight_offset
        {
            return Ok(PackOutcome::Refused(PackRefusal::InvalidLayout));
        }
        let expected_width = match packed_width(chunk, layout)? {
            Ok(width) => width,
            Err(refusal) => return Ok(PackOutcome::Refused(refusal)),
        };
        for (member, weights) in chunk
            .iter()
            .zip(&self.member_weights[member_weight_offset..])
        {
            if member.rollout.completion_len() != weights.len() {
                return Err(Error::tokenize("rollout does not match its token weights"));
            }
        }

        let batch = &mut self.packed_sequence_batch;
        batch.tokens.clear();
        batch.labels.clear();
        batch.weights.clear();
        batch.positions.clear();
        batch.seq_offsets.clear();
        batch.seq_ids.clear();
        batch.tokens.resize(layout.ubatch, layout.pad_token);
        batch.labels.resize(layout.ubatch, -1);
        batch.weights.resize(layout.ubatch, 0.0);
        batch.positions.resize(layout.ubatch, 0);
        batch.seq_offsets.push(0);
        // Use the configured width, rather than the number of live members,
        // so n_seqs_unq and the resulting graph shape stay constant.
        batch.n_sequences = layout.n_seq_max;

        let mut cursor = 0_usize;
        let mut shared_count = 0_usize;
        let mut group_start = 0_usize;
        while group_start < chunk.len() {
            let first_train = chunk[group_start].rollout.first_train_index()?;
            let shared_len = first_train - 1;
            let shared_tokens = &chunk[group_start].rollout.tokens[..shared_len];
            // A reward group may contain externally supplied trajectories
            // with different prefixes. Share only the longest contiguous run
            // that is actually compatible; the remaining members become
            // independent physical prompt groups in the same packed ubatch.
            let group_end = compatible_run_end(chunk, group_start)?;
            let members = &chunk[group_start..group_end];
            if members.len() > 1 {
                shared_count += shared_len;
            }

            let sequence_ids = group_start..group_end;
            batch.tokens[cursor..cursor + shared_len].copy_from_slice(shared_tokens);
            for offset in 0..shared_len {
                batch.positions[cursor] = offset as i32;
                batch
                    .seq_ids
                    .extend(sequence_ids.clone().map(|id| id as i32));
                batch.seq_offsets.push(batch.seq_ids.len());
                cursor += 1;
            }

            for (member_index, member) in (group_start..group_end).zip(members) {
                let rollout = member.rollout;
                let training_span = rollout.training_span_len()?;
                let input = &rollout.tokens[first_train - 1..first_train - 1 + training_span];
                batch.tokens[cursor..cursor + training_span].copy_from_slice(input);
                let mut weight_index = 0_usize;
                for offset in 0..training_span {
                    let target = first_train + offset;
                    if rollout.train_mask[target] {
                        batch.labels[cursor] = rollout.tokens[target];
                        batch.weights[cursor] =
                            self.member_weights[member_weight_offset + member_index][weight_index];
                        weight_index += 1;
                    }
                    batch.positions[cursor] = (first_train - 1 + offset) as i32;
                    batch.seq_ids.push(member_index as i32);
                    batch.seq_offsets.push(batch.seq_ids.len());
                    cursor += 1;
                }
                debug_assert_eq!(
                    weight_index,
                    self.member_weights[member_weight_offset + member_index].len()
                );
            }
            group_start = group_end;
        }

        // Give every unused slot an isolated position-zero token. This keeps
        // n_seqs_unq invariant without coupling unrelated active sequences or
        // violating llama.cpp's per-sequence position continuity check.
        for sequence in chunk.len()..layout.n_seq_max {
            batch.positions[cursor] = 0;
            batch.seq_ids.push(sequence as i32);
            batch.seq_offsets.push(batch.seq_ids.len());
            cursor += 1;
        }
        debug_assert_eq!(cursor, expected_width);
        // Tail padding continues sequence 0 from the last position actually
        // written for it, which is the input position of its last supervised
        // target (`last_train - 1`) - not `tokens.len() - 2`. The two coincide
        // only when the final token is trainable; a multi-turn trajectory
        // ending on an untrained tool observation would otherwise leave a hole
        // and be rejected by llama.cpp's per-sequence continuity check.
        let sequence_zero_end = chunk[0]
            .rollout
            .last_train_index()?
            .checked_sub(1)
            .ok_or_else(|| Error::invalid("rollout is too short for teacher forcing"))?;
        for packed in cursor..layout.ubatch {
            batch.positions[packed] = (sequence_zero_end + 1 + packed - cursor) as i32;
            batch.seq_ids.push(0);
            batch.seq_offsets.push(batch.seq_ids.len());
        }

        let active = batch
            .labels
            .iter()
            .zip(&batch.weights)
            .filter(|(label, weight)| **label >= 0 && **weight != 0.0)
            .count();
        if active > 0 {
            let scale = active as f32 / loss_denominator as f32;
            for (&label, weight) in batch.labels.iter().zip(&mut batch.weights) {
                if label >= 0 && *weight != 0.0 {
                    *weight *= scale;
                }
            }
        }
        Ok(PackOutcome::Packed {
            useful_tokens: cursor - (layout.n_seq_max - chunk.len()),
            shared_tokens: shared_count,
        })
    }
}

/// llama.cpp averages each physical ubatch over its non-zero weighted labels,
/// then averages the ubatches in one logical batch. The RL objectives instead
/// require a `1 / loss_denominator` reduction, including clipped
/// (zero-gradient) tokens: PPO passes the completion length (`1/|o_i|`),
/// Dr. GRPO the constant generation budget. Rescaling the non-zero
/// coefficients per ubatch makes those two reductions algebraically identical
/// without changing the llama.cpp fork ABI.
pub(super) fn normalize_runtime_weights(
    labels: &[i32],
    weights: &mut [f32],
    loss_denominator: usize,
    batch: usize,
    ubatch: usize,
) -> Result<()> {
    if labels.len() != weights.len()
        || loss_denominator == 0
        || ubatch == 0
        || batch == 0
        || !batch.is_multiple_of(ubatch)
    {
        return Err(Error::invalid(
            "invalid weighted-loss normalization geometry",
        ));
    }
    let accumulation_period = (batch / ubatch) as f32;
    for (label_chunk, weight_chunk) in labels.chunks(ubatch).zip(weights.chunks_mut(ubatch)) {
        let nonzero = label_chunk
            .iter()
            .zip(weight_chunk.iter())
            .filter(|(label, weight)| **label >= 0 && **weight != 0.0)
            .count();
        if nonzero == 0 {
            continue;
        }
        let scale = accumulation_period * nonzero as f32 / loss_denominator as f32;
        for (&label, weight) in label_chunk.iter().zip(weight_chunk) {
            if label >= 0 && *weight != 0.0 {
                *weight *= scale;
            }
        }
    }
    Ok(())
}
