//! The dataset's length distribution - phase 0's second input.
//!
//! `n_ctx` is chosen from a percentile of it, and the host budget from its size,
//! so the resolver needs it before anything is loaded. Two ways to get it:
//!
//! - **measured** - real token lengths, which needs the model's tokenizer and
//!   therefore a `Trainer`. The caller passes them in; this crate never loads a
//!   model.
//! - **estimated** - character lengths divided by a deliberately low
//!   characters-per-token ratio, so the token count comes out *high*. `/v1/plan`
//!   uses this: a plan must stay a fast, side-effect-free operation, and
//!   over-estimating lengths only ever produces a roomier `n_ctx`.

use std::path::Path;

use serde::Serialize;

use retrograd_core::{Error, Result};
use retrograd_dataset::{DataFormat, read_chat_jsonl};

/// Characters per token assumed when no tokenizer has run.
///
/// Real ratios sit around 3.5–4.5 for English text with a modern BPE vocabulary.
/// Three is below that on purpose: fewer characters per token means more tokens,
/// which is the direction a memory budget must err in.
const CONSERVATIVE_CHARS_PER_TOKEN: f64 = 3.0;

/// Length statistics of a prepared dataset.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct DatasetStats {
    pub examples: u64,
    pub total_tokens: u64,
    /// Per-example token lengths, ascending. Kept whole rather than reduced to a
    /// few quantiles because the resolver reports the *exact* truncation rate at
    /// whatever `n_ctx` it lands on, and that cannot be recovered from summaries.
    #[serde(skip)]
    lengths: Vec<u32>,
    /// False when the lengths come from the character heuristic.
    pub measured: bool,
}

impl DatasetStats {
    /// From real tokenized lengths.
    pub fn measured(lengths: Vec<u32>) -> Self {
        Self::build(lengths, true)
    }

    /// From lengths obtained any other way.
    pub fn estimated(lengths: Vec<u32>) -> Self {
        Self::build(lengths, false)
    }

    fn build(mut lengths: Vec<u32>, measured: bool) -> Self {
        lengths.sort_unstable();
        Self {
            examples: lengths.len() as u64,
            total_tokens: lengths.iter().map(|length| *length as u64).sum(),
            lengths,
            measured,
        }
    }

    /// Reads a dataset and estimates its per-example token lengths from
    /// character counts, without a tokenizer.
    ///
    /// A text corpus has no examples to speak of - it is windowed into rows of
    /// `n_ctx` by `dataset::prepare` - so its "length distribution" is a single
    /// figure: the whole file. The percentile of that is the file, which would
    /// pin `n_ctx` to the corpus size; the caller handles `Text` by asking for a
    /// context instead of deriving one, and this function reports one row so the
    /// host-side arithmetic still has a size to work with.
    pub fn estimate_from_file(path: &Path, format: DataFormat) -> Result<Self> {
        match format {
            DataFormat::Text => {
                let text = std::fs::read_to_string(path)?;
                Ok(Self::estimated(vec![tokens_from_chars(
                    text.chars().count(),
                )]))
            }
            DataFormat::ChatJsonl => {
                let records = read_chat_jsonl(path)?;
                let lengths = records
                    .iter()
                    .map(|record| {
                        let characters: usize = record
                            .example
                            .messages
                            .iter()
                            .map(|message| {
                                // Every message also carries a role header and
                                // its turn delimiters in the chat template; a
                                // dozen tokens per message covers the templates
                                // in use and keeps the estimate high.
                                message.content.chars().count() + message.role.chars().count() + 36
                            })
                            .sum();
                        tokens_from_chars(characters)
                    })
                    .collect::<Vec<_>>();
                if lengths.is_empty() {
                    return Err(Error::invalid(format!(
                        "{} contains no examples",
                        path.display()
                    )));
                }
                Ok(Self::estimated(lengths))
            }
        }
    }

    /// The token length at `quantile ∈ [0, 1]`, rounded up to the next example.
    pub fn percentile(&self, quantile: f64) -> u32 {
        if self.lengths.is_empty() {
            return 0;
        }
        let quantile = quantile.clamp(0.0, 1.0);
        let index = ((self.lengths.len() as f64 - 1.0) * quantile).ceil() as usize;
        self.lengths[index.min(self.lengths.len() - 1)]
    }

    pub fn max_length(&self) -> u32 {
        self.lengths.last().copied().unwrap_or(0)
    }

    /// Every per-example length, ascending - what a caller needs to persist a
    /// measurement and rebuild the identical statistics later, since no set of
    /// quantiles recovers an exact truncation rate.
    pub fn lengths(&self) -> &[u32] {
        &self.lengths
    }

    /// The share of examples that would be truncated at `n_ctx`. This is the
    /// number invariant 4 makes the caller opt into, so it is exact, not
    /// interpolated.
    pub fn truncation_fraction(&self, n_ctx: u32) -> f64 {
        if self.lengths.is_empty() {
            return 0.0;
        }
        let truncated = self
            .lengths
            .iter()
            .filter(|length| **length > n_ctx)
            .count();
        truncated as f64 / self.lengths.len() as f64
    }

    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }
}

fn tokens_from_chars(characters: usize) -> u32 {
    ((characters as f64 / CONSERVATIVE_CHARS_PER_TOKEN).ceil() as u64).clamp(1, u32::MAX as u64)
        as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_read_off_the_sorted_lengths() {
        let stats = DatasetStats::measured(vec![10, 100, 20, 200, 30]);
        assert_eq!(stats.examples, 5);
        assert_eq!(stats.total_tokens, 360);
        assert_eq!(stats.percentile(0.0), 10);
        assert_eq!(stats.percentile(1.0), 200);
        assert_eq!(stats.max_length(), 200);
        // P99 of five examples is the largest: rounding up never drops the tail.
        assert_eq!(stats.percentile(0.99), 200);
    }

    #[test]
    fn the_truncation_rate_is_exact() {
        let stats = DatasetStats::measured(vec![10, 20, 30, 40]);
        assert_eq!(stats.truncation_fraction(40), 0.0);
        assert_eq!(stats.truncation_fraction(25), 0.5);
        assert_eq!(stats.truncation_fraction(0), 1.0);
    }

    #[test]
    fn the_character_heuristic_over_estimates_rather_than_under() {
        // 300 characters is 100 tokens at the assumed ratio, well above the ~75
        // a real BPE vocabulary would produce.
        assert_eq!(tokens_from_chars(300), 100);
        assert_eq!(tokens_from_chars(0), 1);
    }

    #[test]
    fn a_chat_jsonl_file_yields_one_length_per_record() {
        let dir = std::env::temp_dir().join("retrograd-plan-dataset-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chat.jsonl");
        std::fs::write(
            &path,
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},\
             {\"role\":\"assistant\",\"content\":\"hello\"}]}\n\
             {\"messages\":[{\"role\":\"user\",\"content\":\"a longer question here\"},\
             {\"role\":\"assistant\",\"content\":\"and a longer answer to go with it\"}]}\n",
        )
        .unwrap();
        let stats = DatasetStats::estimate_from_file(&path, DataFormat::ChatJsonl).unwrap();
        assert_eq!(stats.examples, 2);
        assert!(!stats.measured);
        assert!(stats.percentile(1.0) > stats.percentile(0.0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_empty_distribution_answers_zero_rather_than_panicking() {
        let stats = DatasetStats::measured(Vec::new());
        assert!(stats.is_empty());
        assert_eq!(stats.percentile(0.99), 0);
        assert_eq!(stats.truncation_fraction(128), 0.0);
    }
}
