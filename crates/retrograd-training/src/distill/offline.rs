//! Offline top-k distillation: the teacher's truncated distribution, computed
//! once by `retrograd distill-teacher` and read from a sidecar.
//!
//! Nothing here samples and nothing here holds a teacher. The run is an SFT
//! epoch loop whose target is a sparse *distribution* instead of one token:
//!
//! ```text
//! loss = - sum_j p_j(t) * log softmax(z_t)[v_j(t)]
//! ```
//!
//! which is the same weighted cross-entropy the policy-gradient path already
//! drives, with `k` entries per position instead of one. That is the whole of
//! the objective: `WeightedBatch::n_topk` carries `k` down to the runtime, the
//! dense label row becomes the distribution it always was able to hold, and the
//! fused operator accumulates over the entries. No estimator, no bias - the
//! gradient is exact against the teacher's truncated distribution.
//!
//! What it does *not* remove is the truncation itself: the mass the teacher put
//! outside its top `k` is renormalized away by
//! [`TopKSidecar::weighted_targets`], so the student is fitted to the
//! conditional distribution "given the teacher's top `k`". That is a modelling
//! choice the sidecar's `k` makes, not an approximation this module introduces.

use retrograd_config::OfflineDistillConfig;
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics, WeightedBatch};
use retrograd_dataset::topk::{TopKSidecar, corpus_fingerprint, tokenizer_fingerprint};
use retrograd_dataset::{self as dataset, DataFormat, IGNORE_LABEL, PreparedDataset};
use retrograd_engine::Trainer;

use super::teacher::WITNESS_SENTENCES;
use crate::{Boundary, Progress};

/// The corpus and the teacher's distribution over it, checked against each
/// other and against the student.
///
/// The prepared corpus is not kept: its tokens move into `batch`, and its
/// labels are spent once the sidecar is turned into weighted targets. Rows and
/// width are `batch.n_rows` and `batch.n_ctx`.
#[derive(Debug)]
pub struct OfflineBatch {
    pub batch: WeightedBatch,
    /// Entries per position, as the sidecar declared them.
    pub k: usize,
    /// Positions the corpus supervises. The number of terms the loss has, and
    /// the divisor the runtime normalizes by - not `positions * k`.
    pub supervised_positions: usize,
}

/// Reads the corpus, reads the sidecar, and refuses the pair unless they
/// describe the same tokens under the same vocabulary.
///
/// The three refusals are the whole safety of this path. A sidecar is a list of
/// numbers indexed by position; nothing in the numbers themselves says which
/// corpus they came from, and a run given the wrong one trains perfectly well
/// on the teacher's opinion about *other tokens*. That failure has no symptom
/// but a distillation that does not work, which is why the check is at load and
/// not in a test.
pub fn prepare(trainer: &Trainer, config: &OfflineDistillConfig) -> Result<OfflineBatch> {
    let n_ctx = trainer.context_size()?;
    let format = DataFormat::infer(&config.data)?;
    let prepared = dataset::prepare(trainer, &config.data, format, n_ctx)?;
    if prepared.is_empty() || prepared.supervised_tokens == 0 {
        return Err(Error::invalid(
            "offline distillation needs at least one supervised training token",
        ));
    }
    let sidecar = TopKSidecar::read(&config.sidecar)?;
    let vocab_size = u32::try_from(trainer.vocab_size()?)
        .map_err(|_| Error::overflow("the student's vocabulary size does not fit in u32"))?;
    let witnesses = witness_ids(trainer)?;
    sidecar.check_against(
        &prepared,
        vocab_size,
        tokenizer_fingerprint(vocab_size, &witnesses),
        corpus_fingerprint(&prepared),
    )?;

    let (labels, weights) = sidecar.weighted_targets(&prepared)?;
    let k = sidecar.k();
    drop(sidecar);
    let batch = WeightedBatch {
        tokens: prepared.tokens,
        labels,
        weights,
        n_rows: prepared.examples,
        n_ctx: prepared.n_ctx,
        n_topk: k,
    };
    batch.validate()?;
    Ok(OfflineBatch {
        k,
        supervised_positions: prepared.supervised_tokens,
        batch,
    })
}

/// The witness sentences as this model tokenizes them - the input of the
/// fingerprint the sidecar carries.
///
/// The same sentences `Teacher::compatibility` compares id for id on the
/// on-policy path. Reusing them is what makes the two gates one decision: a
/// teacher/student pair this refuses is exactly a pair that would refuse there.
pub fn witness_ids(trainer: &Trainer) -> Result<Vec<Vec<i32>>> {
    WITNESS_SENTENCES
        .iter()
        .map(|sentence| trainer.tokenize_text(sentence))
        .collect()
}

/// One pass over the corpus per epoch, restarting at an epoch boundary restored
/// from a checkpoint.
///
/// The epoch is the resume unit, as in SFT and for the same reason: inside one,
/// the runtime owns the row cursor. Unlike on-policy distillation there is no
/// second unit to choose from - the target never moves, so an "update" would be
/// an epoch under another name.
pub fn run_resumed(
    trainer: &mut Trainer,
    prepared: &OfflineBatch,
    training: &TrainConfig,
    epochs: u32,
    resume: Option<Boundary>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::distill_offline", "training");
    let _entered = span.enter();
    let start_epoch = match resume {
        Some(boundary) => u32::try_from(boundary.completed_iterations).map_err(|_| {
            Error::overflow("the checkpoint epoch count does not fit in the epoch counter")
        })?,
        None => 0,
    };
    // Every epoch is the same number of optimizer steps over the same rows, so
    // the horizon is known up front and pinned once. Left to accumulate per
    // call, the schedule would decay over the first epoch and flatten after it.
    let steps_per_epoch = prepared.batch.n_rows as u64
        * (prepared.batch.n_ctx as u64).div_euclid(batch_window(training)?);
    let total_steps = steps_per_epoch.saturating_mul(u64::from(epochs)).max(1);

    let mut final_metrics = TrainMetrics::default();
    for epoch in start_epoch..epochs {
        let (mut metrics, keep_going) = trainer.train_weighted_controlled(
            &prepared.batch,
            total_steps,
            |trainer, metrics| on_progress(trainer, Progress::sft(metrics, false)),
        )?;
        // `train_weighted` reports "epoch 1" for every call it is given, since
        // one call *is* one weighted epoch. The run's epoch counter is this
        // loop's, and the two have to agree before anything downstream reads a
        // boundary off the metrics.
        metrics.epoch = epoch + 1;
        metrics.epoch_complete = true;
        final_metrics = metrics;
        let mut progress = Progress::sft(metrics, false);
        progress.values.push(retrograd_metrics::MetricValue {
            name: "distill/topk_entries".into(),
            value: prepared.k as f32,
        });
        progress.boundary = Some(Boundary {
            completed_iterations: u64::from(epoch + 1),
            cursor: 0,
            kl_multiplier: None,
        });
        if !on_progress(trainer, progress)? || !keep_going {
            return Ok(final_metrics);
        }
    }
    Ok(final_metrics)
}

/// One teacher-forced pass per row, turned into one block per position.
///
/// A row of the prepared dataset is `[t_0 .. t_{n-1}]` with `labels[p]` the
/// token that follows position `p`. `top_logprobs_suffix(tokens, n_prompt)`
/// returns the distribution over what follows each position from `n_prompt - 1`
/// on, so a whole row is scored with `n_prompt = 1` and the result read off by
/// position. Rows shorter than two tokens carry no target at all and are
/// skipped; their blocks stay absent, which the trainer reads as masked.
pub fn score_corpus(
    teacher: &mut Trainer,
    prepared: &PreparedDataset,
    k: usize,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let entries = prepared
        .tokens
        .len()
        .checked_mul(k)
        .ok_or_else(|| Error::overflow("top-k sidecar shape overflows usize"))?;
    let mut ids = vec![IGNORE_LABEL; entries];
    let mut logprobs = vec![f32::NEG_INFINITY; entries];
    for row in 0..prepared.examples {
        let start = row * prepared.n_ctx;
        let row_tokens = &prepared.tokens[start..start + prepared.n_ctx];
        let row_labels = &prepared.labels[start..start + prepared.n_ctx];
        // Trailing padding carries no label, and scoring it would spend a
        // forward pass on positions the loss masks anyway. The last supervised
        // position of the row is where the pass stops.
        let Some(last) = row_labels.iter().rposition(|&label| label >= 0) else {
            continue;
        };
        // A teacher-forced pass over `tokens[..width]` yields the distribution
        // at positions `0 ..= width - 2`: the target of position `width - 1` is
        // the token that follows the window, and the pass does not contain it.
        // A chat row pads its tail, so its last supervised position is well
        // inside the row; a full-width *text* row's last position has a label
        // the row does not hold, and the block for it stays absent - which the
        // trainer reads as masked. Dropping one position of such a row is the
        // honest answer: the alternative is to score it against a token the
        // corpus never handed the teacher.
        let width = (last + 2).min(row_tokens.len());
        if width < 2 {
            continue;
        }
        let scored = teacher.top_logprobs_suffix(&row_tokens[..width], 1, k)?;
        let scored_positions = last.min(width - 2) + 1;
        for (position, &label) in row_labels.iter().enumerate().take(scored_positions) {
            if label < 0 {
                continue;
            }
            // `top_logprobs_suffix` skips the prompt: its row 0 is the
            // distribution over what follows position 0, so the two indices
            // coincide here because the prompt is one token wide.
            let Some((row_ids, row_logprobs)) = scored.row(position) else {
                continue;
            };
            let offset = (start + position) * k;
            ids[offset..offset + k].copy_from_slice(row_ids);
            logprobs[offset..offset + k].copy_from_slice(row_logprobs);
        }
    }
    Ok((ids, logprobs))
}

/// Tokens the runtime advances per optimizer step. One row is
/// `n_ctx / n_batch` steps, the same product the run driver validates.
fn batch_window(training: &TrainConfig) -> Result<u64> {
    if training.n_batch == 0 {
        return Err(Error::invalid(
            "training.micro_batch * gradient_accumulation is zero",
        ));
    }
    Ok(training.n_batch as u64)
}

#[cfg(test)]
mod tests {
    use retrograd_dataset::IGNORE_LABEL;
    use retrograd_dataset::topk::{TopKHeader, TopKSidecar, VERSION};

    use super::*;

    fn prepared(labels: Vec<i32>) -> PreparedDataset {
        PreparedDataset {
            n_ctx: labels.len(),
            tokens: vec![1; labels.len()],
            supervised_tokens: labels.iter().filter(|&&label| label >= 0).count(),
            labels,
            examples: 1,
        }
    }

    /// The batch handed to the runtime is `[K, n_positions]` and its masked
    /// positions are inert. This is the shape contract between this module and
    /// `WeightedBatch::validate`, and getting it wrong is a panic three crates
    /// away rather than an error here.
    #[test]
    fn the_batch_built_from_a_sidecar_has_k_entries_per_position() {
        let data = prepared(vec![IGNORE_LABEL, 4, 5]);
        let sidecar = TopKSidecar::new(
            TopKHeader {
                version: VERSION,
                k: 2,
                n_rows: 3,
                vocab_size: 16,
                tokenizer_hash: 1,
                source_hash: 2,
            },
            vec![1, 2, 4, 6, 5, 7],
            vec![-0.1, -1.0, -0.2, -1.5, -0.3, -2.0],
        )
        .expect("shape");
        let (labels, weights) = sidecar.weighted_targets(&data).expect("targets");
        let batch = WeightedBatch {
            tokens: data.tokens.clone(),
            labels,
            weights,
            n_rows: data.examples,
            n_ctx: data.n_ctx,
            n_topk: 2,
        };
        batch.validate().expect("valid");
        assert_eq!(batch.labels.len(), 6);
        assert_eq!(&batch.labels[0..2], &[IGNORE_LABEL, IGNORE_LABEL]);
        assert_eq!(&batch.weights[0..2], &[0.0, 0.0]);
        let mass: f32 = batch.weights[2..4].iter().sum();
        assert!((mass - 1.0).abs() < 1e-6, "{mass}");
    }
}
