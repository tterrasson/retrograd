//! Which pairs share an optimizer step, and how many steps a run takes.
//!
//! Both are functions of the epoch's order and the pairs' costs alone, so the
//! whole run is laid out before its first step: the learning-rate horizon is
//! the exact step count rather than a bound, and a resume re-derives the
//! chunks of its epoch instead of storing them.

use std::ops::Range;

use retrograd_core::{Error, Result};

use crate::grpo::shuffled_indices;

/// The order an epoch visits the pairs in: a permutation seeded by the run's
/// seed and the epoch index, or the file order.
pub(crate) fn epoch_order(len: usize, shuffle: bool, seed: u32, epoch: u32) -> Vec<usize> {
    if !shuffle {
        return (0..len).collect();
    }
    let seed = (u64::from(seed) << 32) ^ u64::from(epoch).wrapping_add(1);
    shuffled_indices(len, seed)
}

/// Consecutive whole pairs of `order`, grouped while their summed cost - in
/// physical ubatches - fits one accumulation `period`, and while the group
/// holds fewer than `cap` pairs. A pair is never split between two groups; a
/// pair that alone exceeds the period forms its own. Ranges index `order`.
pub(crate) fn chunk_pairs(
    order: &[usize],
    costs: &[u64],
    period: u64,
    cap: Option<usize>,
) -> Vec<Range<usize>> {
    let cap = cap.unwrap_or(usize::MAX).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut budget = 0_u64;
    for (position, &pair) in order.iter().enumerate() {
        let cost = costs[pair];
        if position > start && (budget.saturating_add(cost) > period || position - start >= cap) {
            chunks.push(start..position);
            start = position;
            budget = 0;
        }
        budget = budget.saturating_add(cost);
    }
    if start < order.len() {
        chunks.push(start..order.len());
    }
    chunks
}

/// Optimizer steps one chunk takes: one when it is packed or fits the period,
/// otherwise one per started period of its rows - the runtime closes a step
/// every `period` ubatches and pads the last one.
pub(crate) fn chunk_steps(cost: u64, period: u64, packed: bool) -> u64 {
    match packed {
        true => 1,
        false => cost.div_ceil(period.max(1)).max(1),
    }
}

/// Where a resumed epoch starts: the index of the chunk that begins after
/// `cursor` pairs. A cursor inside a chunk cannot come from this run's own
/// boundaries, which only ever fall between chunks.
pub(crate) fn resume_chunk(chunks: &[Range<usize>], cursor: usize) -> Result<usize> {
    if cursor == 0 {
        return Ok(0);
    }
    chunks
        .iter()
        .position(|chunk| chunk.start == cursor)
        .or_else(|| (chunks.last().map(|chunk| chunk.end) == Some(cursor)).then_some(chunks.len()))
        .ok_or_else(|| {
            Error::checkpoint(format!(
                "the checkpoint resumes {cursor} pairs into an epoch, which is not the start of \
                 an optimizer step of this run: the pairs, their order or the step geometry \
                 changed since it was written"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_never_splits_a_pair_and_respects_its_cap() {
        let order = [0, 1, 2, 3, 4];
        let costs = [2, 1, 1, 3, 1];
        assert_eq!(chunk_pairs(&order, &costs, 4, None), vec![0..3, 3..5]);
        assert_eq!(
            chunk_pairs(&order, &costs, 4, Some(2)),
            vec![0..2, 2..4, 4..5]
        );
        // A pair larger than the period is a chunk of its own.
        let costs = [2, 9, 1, 1, 1];
        assert_eq!(chunk_pairs(&order, &costs, 4, None), vec![0..1, 1..2, 2..5]);
        assert_eq!(chunk_steps(9, 4, false), 3);
        assert_eq!(chunk_steps(9, 4, true), 1);
        assert_eq!(chunk_steps(3, 4, false), 1);
    }

    #[test]
    fn the_order_is_reproducible_by_seed_and_epoch() {
        assert_eq!(epoch_order(5, false, 1, 3), vec![0, 1, 2, 3, 4]);
        let first = epoch_order(32, true, 7, 0);
        assert_eq!(first, epoch_order(32, true, 7, 0));
        assert_ne!(first, epoch_order(32, true, 7, 1));
        assert_ne!(first, epoch_order(32, true, 8, 0));
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn a_resume_lands_on_a_chunk_boundary_or_is_refused() {
        let chunks = [0..3, 3..5];
        assert_eq!(resume_chunk(&chunks, 0).unwrap(), 0);
        assert_eq!(resume_chunk(&chunks, 3).unwrap(), 1);
        assert_eq!(resume_chunk(&chunks, 5).unwrap(), 2);
        let error = resume_chunk(&chunks, 4).unwrap_err();
        assert!(matches!(error, Error::Checkpoint(_)), "{error}");
    }
}
