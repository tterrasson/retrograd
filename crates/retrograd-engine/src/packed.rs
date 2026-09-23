use std::collections::BTreeSet;

use retrograd_core::{Error, Result};

/// One physical multi-sequence optimizer batch. Prefix tokens are visible to
/// every sequence and are therefore represented once in the differentiable
/// graph; branch tokens carry an explicit sequence id and causal position.
#[derive(Clone, Debug)]
pub struct PackedSequenceBatch {
    pub tokens: Vec<i32>,
    pub labels: Vec<i32>,
    pub weights: Vec<f32>,
    pub positions: Vec<i32>,
    /// CSR offsets into `seq_ids`, one entry per token plus a sentinel.
    pub seq_offsets: Vec<usize>,
    pub seq_ids: Vec<i32>,
    pub n_sequences: usize,
    /// Sparse targets per position, with the same layout and meaning as
    /// [`retrograd_core::WeightedBatch::n_topk`].
    pub n_topk: usize,
}

impl PackedSequenceBatch {
    pub(crate) fn validate(&self) -> Result<()> {
        let len = self.tokens.len();
        if self.n_topk == 0 || self.n_topk > retrograd_core::FUSED_CE_K_MAX {
            return Err(Error::invalid(format!(
                "packed-sequence batch n_topk must be in 1..={}, got {}",
                retrograd_core::FUSED_CE_K_MAX,
                self.n_topk
            )));
        }
        let entries = len
            .checked_mul(self.n_topk)
            .ok_or_else(|| Error::overflow("packed-sequence batch shape overflows usize"))?;
        if len == 0
            || self.labels.len() != entries
            || self.weights.len() != entries
            || self.positions.len() != len
            || self.seq_offsets.len() != len + 1
            || self.seq_offsets.first() != Some(&0)
            || self.seq_offsets.last() != Some(&self.seq_ids.len())
            || self.seq_offsets.windows(2).any(|pair| pair[0] >= pair[1])
            || self.n_sequences == 0
            || self.n_sequences > u32::MAX as usize
        {
            return Err(Error::invalid("invalid packed-sequence batch shape"));
        }
        if let Some(position) = self.positions.iter().position(|&value| value < 0) {
            return Err(Error::invalid(format!(
                "packed-sequence batch token {position} has a negative position"
            )));
        }
        if let Some(index) = self
            .seq_ids
            .iter()
            .position(|&id| id < 0 || id as usize >= self.n_sequences)
        {
            return Err(Error::invalid(format!(
                "packed-sequence batch membership {index} names sequence {} outside 0..{}",
                self.seq_ids[index], self.n_sequences
            )));
        }
        // llama.cpp rejects a batch whose per-sequence positions leave a hole
        // (`llama_batch_allocr::init`). Catch it here, where the offending
        // sequence and the missing position can still be named, instead of
        // losing the whole update to a message from the batch allocator.
        let mut positions_by_sequence = vec![BTreeSet::<i32>::new(); self.n_sequences];
        for (token, range) in self.seq_offsets.windows(2).enumerate() {
            for &id in &self.seq_ids[range[0]..range[1]] {
                positions_by_sequence[id as usize].insert(self.positions[token]);
            }
        }
        for (sequence, positions) in positions_by_sequence.iter().enumerate() {
            let (Some(&first), Some(&last)) = (positions.first(), positions.last()) else {
                continue;
            };
            if (last - first + 1) as usize != positions.len() {
                let missing = (first..=last).find(|position| !positions.contains(position));
                return Err(Error::invalid(format!(
                    "packed-sequence batch leaves sequence {sequence} non-continuous over \
                     positions {first}..={last}: {} distinct positions, first hole at {}",
                    positions.len(),
                    missing.unwrap_or(last),
                )));
            }
        }
        Ok(())
    }
}
