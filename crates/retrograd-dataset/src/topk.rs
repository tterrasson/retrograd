//! The offline top-k sidecar: a teacher's truncated distribution over every
//! position of a prepared dataset.
//!
//! **Binary, and beside the JSONL rather than inside it.** A position costs
//! `k * 8` bytes here; the same numbers as JSON cost forty times that and have
//! to be parsed before the first optimizer step. At `k = 16` a ten-million-token
//! corpus is 1.3 GiB - large, but streamed and read once per epoch, which is not
//! true of anything written as text.
//!
//! **One block per position of the prepared dataset, in dataset order.** The
//! sidecar is not indexed by example, by line or by character offset: row `i` of
//! the file describes token `i` of [`PreparedDataset::tokens`]. That is what
//! makes a one-token misalignment impossible rather than merely unlikely - the
//! producer, `retrograd distill-teacher`, walks the very rows the trainer will
//! walk, so there is no second tokenization to agree with.
//!
//! The two hashes in the header are the guard for everything the layout cannot
//! express: a sidecar produced from another corpus, or by a model with another
//! vocabulary, is refused at load. It is the same gate as `Teacher::compatibility`
//! on the on-policy path, reached by another road.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use retrograd_core::{Error, Result};

use crate::{IGNORE_LABEL, PreparedDataset};

/// First eight bytes of a sidecar. Version travels in its own field; this only
/// says "not some other file".
const MAGIC: [u8; 8] = *b"RGTOPK\0\0";

/// The only format this crate reads or writes.
pub const VERSION: u32 = 1;

/// Bytes of header before the first block.
const HEADER_BYTES: usize = 64;

/// Bytes one entry costs: a `u32` vocabulary id and an `f32` log-probability.
const ENTRY_BYTES: usize = 8;

/// Entries per position, chosen once here rather than per run.
///
/// Sixteen, from a band the plan left open at 8 to 20. Below eight, the
/// truncated distribution of an instruct model at temperature 1 loses mass that
/// is not tail; above twenty the file grows for probabilities the renormalization
/// then divides away to nothing. Sixteen sits inside that band and makes a
/// position exactly 128 bytes, so a block never straddles a cache line - which
/// costs nothing to have and is not free to add later.
///
/// A sidecar records its own `k`; this is the default the producer writes, not a
/// constraint on what the reader accepts.
pub const DEFAULT_K: usize = 16;

/// Ceiling on `k`, mirroring [`retrograd_core::FUSED_CE_K_MAX`]: past it the
/// runtime's own operator refuses the batch, so a sidecar that large could be
/// written and never trained on.
pub const MAX_K: usize = retrograd_core::FUSED_CE_K_MAX;

/// An absent entry - a position the teacher did not score, or padding inside a
/// block whose `k` the teacher could not fill. Written as `u32::MAX` because no
/// vocabulary reaches it, and read back as [`IGNORE_LABEL`].
const ABSENT_ID: u32 = u32::MAX;

/// What a sidecar says about itself.
///
/// `tokenizer_hash` and `source_hash` are the two fields that make the file
/// refusable. They are compared, never repaired: a mismatch is a different
/// corpus or a different vocabulary, and both produce a run that trains happily
/// on nonsense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKHeader {
    pub version: u32,
    /// Entries per position, `1..=MAX_K`.
    pub k: u32,
    /// Positions covered - the token count of the prepared dataset.
    pub n_rows: u64,
    pub vocab_size: u32,
    pub tokenizer_hash: u64,
    pub source_hash: u64,
}

impl TopKHeader {
    fn write(&self, out: &mut impl Write) -> Result<()> {
        let mut bytes = [0_u8; HEADER_BYTES];
        bytes[0..8].copy_from_slice(&MAGIC);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.k.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.n_rows.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.vocab_size.to_le_bytes());
        // Entry dtype. One value is defined - 0, the (u32, f32) pair the rest of
        // this module describes - and the field exists so a future f16 variant
        // is a version the reader can name rather than a file it silently
        // misreads. The log-probabilities stay F32 on purpose: halving them
        // would put quantization noise on the exact quantity this whole phase
        // exists to stop estimating.
        bytes[28..32].copy_from_slice(&0_u32.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.tokenizer_hash.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.source_hash.to_le_bytes());
        out.write_all(&bytes)?;
        Ok(())
    }

    fn read(input: &mut impl Read, path: &Path) -> Result<Self> {
        let mut bytes = [0_u8; HEADER_BYTES];
        input.read_exact(&mut bytes).map_err(|error| {
            Error::invalid(format!(
                "{}: not a top-k sidecar, it is shorter than a {HEADER_BYTES}-byte header ({error})",
                path.display()
            ))
        })?;
        if bytes[0..8] != MAGIC {
            return Err(Error::invalid(format!(
                "{}: not a top-k sidecar (bad magic)",
                path.display()
            )));
        }
        let u32_at = |offset: usize| {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        let u64_at = |offset: usize| {
            let mut value = [0_u8; 8];
            value.copy_from_slice(&bytes[offset..offset + 8]);
            u64::from_le_bytes(value)
        };
        let version = u32_at(8);
        if version != VERSION {
            return Err(Error::invalid(format!(
                "{}: top-k sidecar version {version}, this build reads {VERSION}",
                path.display()
            )));
        }
        let dtype = u32_at(28);
        if dtype != 0 {
            return Err(Error::invalid(format!(
                "{}: top-k sidecar entry dtype {dtype}, this build reads 0 (u32 id, f32 logprob)",
                path.display()
            )));
        }
        let k = u32_at(12);
        if k == 0 || k as usize > MAX_K {
            return Err(Error::invalid(format!(
                "{}: top-k sidecar declares k = {k}, which is outside 1..={MAX_K}",
                path.display()
            )));
        }
        Ok(Self {
            version,
            k,
            n_rows: u64_at(16),
            vocab_size: u32_at(24),
            tokenizer_hash: u64_at(32),
            source_hash: u64_at(40),
        })
    }
}

/// A teacher's truncated distribution, one block per position of a prepared
/// dataset.
///
/// `ids` and `logprobs` hold `n_rows * k` values in the layout the runtime's
/// operator reads directly: entry `j` of position `p` at `p * k + j`. An absent
/// entry is [`IGNORE_LABEL`] with a log-probability of negative infinity.
#[derive(Clone, Debug)]
pub struct TopKSidecar {
    header: TopKHeader,
    ids: Vec<i32>,
    logprobs: Vec<f32>,
}

impl TopKSidecar {
    /// Builds a sidecar from the flat arrays a producer accumulated. The two
    /// must hold `n_rows * k` values; anything else is a producer bug and is
    /// refused here rather than written to disk.
    pub fn new(header: TopKHeader, ids: Vec<i32>, logprobs: Vec<f32>) -> Result<Self> {
        let k = header.k as usize;
        let expected = (header.n_rows as usize)
            .checked_mul(k)
            .ok_or_else(|| Error::overflow("top-k sidecar shape overflows usize"))?;
        if ids.len() != expected || logprobs.len() != expected {
            return Err(Error::invalid(format!(
                "top-k sidecar holds {} ids and {} log-probabilities, expected {expected}",
                ids.len(),
                logprobs.len()
            )));
        }
        Ok(Self {
            header,
            ids,
            logprobs,
        })
    }

    pub fn header(&self) -> TopKHeader {
        self.header
    }

    pub fn k(&self) -> usize {
        self.header.k as usize
    }

    pub fn rows(&self) -> usize {
        self.header.n_rows as usize
    }

    /// One position's entries, ids and log-probabilities in the same order.
    pub fn row(&self, index: usize) -> Option<(&[i32], &[f32])> {
        let k = self.k();
        let start = index.checked_mul(k)?;
        let end = start.checked_add(k)?;
        if end > self.ids.len() {
            return None;
        }
        Some((&self.ids[start..end], &self.logprobs[start..end]))
    }

    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    pub fn logprobs(&self) -> &[f32] {
        &self.logprobs
    }

    /// Writes the sidecar, header first, blocks in position order.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        self.header.write(&mut out)?;
        let mut block = [0_u8; ENTRY_BYTES];
        for (&id, &logprob) in self.ids.iter().zip(&self.logprobs) {
            let stored = if id < 0 { ABSENT_ID } else { id as u32 };
            block[0..4].copy_from_slice(&stored.to_le_bytes());
            block[4..8].copy_from_slice(&logprob.to_le_bytes());
            out.write_all(&block)?;
        }
        out.flush()?;
        Ok(())
    }

    /// Reads a sidecar whole. The file is bounded by its own header, so a
    /// truncated one is caught by the length check rather than by a short read
    /// halfway through training.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| {
            Error::invalid(format!(
                "{}: cannot open the top-k sidecar ({error})",
                path.display()
            ))
        })?;
        let mut input = BufReader::new(file);
        let header = TopKHeader::read(&mut input, path)?;
        let k = header.k as usize;
        let entries = (header.n_rows as usize)
            .checked_mul(k)
            .ok_or_else(|| Error::overflow("top-k sidecar shape overflows usize"))?;
        let mut ids = Vec::with_capacity(entries);
        let mut logprobs = Vec::with_capacity(entries);
        let mut block = [0_u8; ENTRY_BYTES];
        for entry in 0..entries {
            input.read_exact(&mut block).map_err(|error| {
                Error::invalid(format!(
                    "{}: top-k sidecar declares {} positions of k = {k} but ends at entry {entry} \
                     ({error})",
                    path.display(),
                    header.n_rows
                ))
            })?;
            let id = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let logprob = f32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            if id == ABSENT_ID {
                ids.push(IGNORE_LABEL);
                logprobs.push(f32::NEG_INFINITY);
            } else {
                if id >= header.vocab_size {
                    return Err(Error::invalid(format!(
                        "{}: top-k sidecar entry {entry} names token {id}, outside the vocabulary \
                         of {} it declares",
                        path.display(),
                        header.vocab_size
                    )));
                }
                ids.push(id as i32);
                logprobs.push(logprob);
            }
        }
        let mut trailing = [0_u8; 1];
        if input.read(&mut trailing)? != 0 {
            return Err(Error::invalid(format!(
                "{}: top-k sidecar is longer than the {} positions of k = {k} its header declares",
                path.display(),
                header.n_rows
            )));
        }
        Self::new(header, ids, logprobs)
    }

    /// Refuses a sidecar that does not belong to this dataset, this vocabulary
    /// or this corpus.
    ///
    /// The three checks fail differently on purpose: a shape mismatch is a
    /// sidecar for another *preparation* (another `n_ctx`, another template), a
    /// tokenizer mismatch is another *model*, and a source mismatch is another
    /// *corpus*. Reporting them as one error would leave the reader guessing
    /// which of the three files to rebuild.
    pub fn check_against(
        &self,
        dataset: &PreparedDataset,
        vocab_size: u32,
        tokenizer_hash: u64,
        source_hash: u64,
    ) -> Result<()> {
        if self.rows() != dataset.tokens.len() {
            return Err(Error::invalid(format!(
                "top-k sidecar covers {} positions, the prepared dataset has {}: the sidecar was \
                 produced from another preparation of the corpus",
                self.rows(),
                dataset.tokens.len()
            )));
        }
        if self.header.vocab_size != vocab_size {
            return Err(Error::invalid(format!(
                "top-k sidecar was produced by a model with a vocabulary of {}, the student has \
                 {vocab_size}",
                self.header.vocab_size
            )));
        }
        if self.header.tokenizer_hash != tokenizer_hash {
            return Err(Error::invalid(
                "top-k sidecar was produced by a model whose tokenizer disagrees with the \
                 student's: its entries name other tokens",
            ));
        }
        if self.header.source_hash != source_hash {
            return Err(Error::invalid(
                "top-k sidecar was produced from another corpus than the one configured",
            ));
        }
        Ok(())
    }

    /// The `[K, n_positions]` label and weight arrays the runtime's weighted
    /// batch expects, for the positions `dataset` supervises.
    ///
    /// The teacher's log-probabilities become **renormalized probabilities**:
    /// `p_j <- exp(logprob_j) / sum_j exp(logprob_j)`, so a position's entries
    /// sum to one whatever mass the truncation dropped. That is the default the
    /// plan settled on, and the reason it needs no alternative: the residual
    /// mass has no vocabulary row to be charged to, and putting it in a sink
    /// token would train the student to emit that token.
    ///
    /// A position the dataset masks (`labels[p] < 0`) comes back with all its
    /// entries absent and all its weights zero, which is exactly how the
    /// operator recognizes an inactive position.
    pub fn weighted_targets(&self, dataset: &PreparedDataset) -> Result<(Vec<i32>, Vec<f32>)> {
        if self.rows() != dataset.tokens.len() {
            return Err(Error::invalid(format!(
                "top-k sidecar covers {} positions, the prepared dataset has {}",
                self.rows(),
                dataset.tokens.len()
            )));
        }
        let k = self.k();
        let mut labels = vec![IGNORE_LABEL; self.ids.len()];
        let mut weights = vec![0.0_f32; self.ids.len()];
        for position in 0..self.rows() {
            if dataset.labels[position] < 0 {
                continue;
            }
            let (ids, logprobs) = self
                .row(position)
                .expect("row index below the sidecar's own length");
            // Log-sum-exp over the surviving entries, then exp of the shifted
            // values. Shifting by the maximum is not an optimization here: an
            // instruct model's top-1 log-probability is routinely above -1e-3
            // and the tail below -30, and exponentiating those directly loses
            // the tail entirely.
            let max = logprobs
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                continue;
            }
            let total: f32 = ids
                .iter()
                .zip(logprobs)
                .filter(|(id, logprob)| **id >= 0 && logprob.is_finite())
                .map(|(_, logprob)| (logprob - max).exp())
                .sum();
            if !total.is_finite() || total <= 0.0 {
                continue;
            }
            for entry in 0..k {
                let id = ids[entry];
                let logprob = logprobs[entry];
                if id < 0 || !logprob.is_finite() {
                    continue;
                }
                labels[position * k + entry] = id;
                weights[position * k + entry] = (logprob - max).exp() / total;
            }
        }
        Ok((labels, weights))
    }
}

/// Fingerprint of a vocabulary, as the ids it gives a fixed set of sentences.
///
/// Not a hash of the GGUF: two checkpoints of one family share a tokenizer while
/// differing everywhere else, and refusing that pair would rule out the very
/// case distillation is for. What has to match is what the ids *mean*, and the
/// cheapest honest test of that is to tokenize and compare - the same test
/// `Teacher::compatibility` runs, folded into eight bytes so it can live in a
/// file header.
pub fn tokenizer_fingerprint(vocab_size: u32, witnesses: &[Vec<i32>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(u64::from(vocab_size));
    for witness in witnesses {
        mix(witness.len() as u64);
        for &id in witness {
            mix(id as u32 as u64);
        }
    }
    hash
}

/// Fingerprint of the corpus a sidecar was produced from: the bytes of the
/// prepared token stream, not of the source file.
///
/// The prepared stream is what both sides actually share. Hashing the JSONL
/// would tie the sidecar to a file that may legitimately be reformatted, and
/// would *not* catch the failure that matters - the same lines prepared under a
/// different `n_ctx` or a different chat template, which shifts every position
/// by an unpredictable amount.
pub fn corpus_fingerprint(dataset: &PreparedDataset) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(dataset.n_ctx as u64);
    mix(dataset.examples as u64);
    for &token in &dataset.tokens {
        mix(token as u32 as u64);
    }
    for &label in &dataset.labels {
        mix(label as u32 as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(labels: Vec<i32>) -> PreparedDataset {
        PreparedDataset {
            n_ctx: labels.len(),
            tokens: vec![1; labels.len()],
            supervised_tokens: labels.iter().filter(|&&label| label >= 0).count(),
            labels,
            examples: 1,
        }
    }

    fn header(k: u32, n_rows: u64) -> TopKHeader {
        TopKHeader {
            version: VERSION,
            k,
            n_rows,
            vocab_size: 32,
            tokenizer_hash: 7,
            source_hash: 9,
        }
    }

    #[test]
    fn a_sidecar_round_trips_through_the_file_it_writes() {
        let sidecar = TopKSidecar::new(
            header(2, 3),
            vec![5, 6, 3, IGNORE_LABEL, 1, 2],
            vec![-0.1, -2.0, -0.5, f32::NEG_INFINITY, -0.2, -1.5],
        )
        .expect("shape");
        let path = std::env::temp_dir().join("retrograd-topk-round-trip.bin");
        sidecar.write(&path).expect("write");
        let read = TopKSidecar::read(&path).expect("read");
        assert_eq!(read.header(), sidecar.header());
        assert_eq!(read.ids(), sidecar.ids());
        assert_eq!(read.logprobs(), sidecar.logprobs());
        std::fs::remove_file(&path).ok();
    }

    /// The length check is the whole point of carrying `n_rows` in the header:
    /// a sidecar cut short by a full disk must not train on whatever it got.
    #[test]
    fn a_truncated_sidecar_is_refused_rather_than_read_short() {
        let sidecar =
            TopKSidecar::new(header(2, 2), vec![1, 2, 3, 4], vec![-0.1; 4]).expect("shape");
        let path = std::env::temp_dir().join("retrograd-topk-truncated.bin");
        sidecar.write(&path).expect("write");
        let bytes = std::fs::read(&path).expect("read back");
        std::fs::write(&path, &bytes[..bytes.len() - ENTRY_BYTES]).expect("truncate");
        let error = TopKSidecar::read(&path).unwrap_err().to_string();
        assert!(error.contains("ends at entry"), "{error}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_longer_sidecar_is_refused_too() {
        let sidecar = TopKSidecar::new(header(1, 1), vec![1], vec![-0.1]).expect("shape");
        let path = std::env::temp_dir().join("retrograd-topk-long.bin");
        sidecar.write(&path).expect("write");
        let mut bytes = std::fs::read(&path).expect("read back");
        bytes.extend_from_slice(&[0_u8; ENTRY_BYTES]);
        std::fs::write(&path, &bytes).expect("extend");
        let error = TopKSidecar::read(&path).unwrap_err().to_string();
        assert!(error.contains("longer than"), "{error}");
        std::fs::remove_file(&path).ok();
    }

    /// The renormalization is what makes the entries a distribution: the mass
    /// the truncation dropped is divided away, not left as a deficit the loss
    /// would silently scale down.
    #[test]
    fn the_entries_of_a_supervised_position_sum_to_one() {
        let sidecar = TopKSidecar::new(
            header(3, 2),
            vec![1, 2, 3, 4, 5, 6],
            vec![-0.5, -1.0, -3.0, -0.1, -2.0, -4.0],
        )
        .expect("shape");
        let (labels, weights) = sidecar
            .weighted_targets(&dataset(vec![7, 8]))
            .expect("targets");
        assert_eq!(labels, vec![1, 2, 3, 4, 5, 6]);
        for position in 0..2 {
            let mass: f32 = weights[position * 3..position * 3 + 3].iter().sum();
            assert!((mass - 1.0).abs() < 1e-6, "position {position}: {mass}");
        }
        // Order is preserved and the argmax keeps the largest share.
        assert!(weights[0] > weights[1] && weights[1] > weights[2]);
    }

    /// A masked position must come back inert. The runtime reads "no entry with
    /// a real id and a non-zero weight" as an inactive position, and that is the
    /// only thing keeping a prompt token out of the loss.
    #[test]
    fn a_masked_position_carries_no_entry_and_no_weight() {
        let sidecar =
            TopKSidecar::new(header(2, 2), vec![1, 2, 3, 4], vec![-0.5, -1.0, -0.5, -1.0])
                .expect("shape");
        let (labels, weights) = sidecar
            .weighted_targets(&dataset(vec![IGNORE_LABEL, 4]))
            .expect("targets");
        assert_eq!(&labels[0..2], &[IGNORE_LABEL, IGNORE_LABEL]);
        assert_eq!(&weights[0..2], &[0.0, 0.0]);
        assert_eq!(&labels[2..4], &[3, 4]);
        assert!(weights[2] > 0.0 && weights[3] > 0.0);
    }

    /// Padding entries survive the round trip as absences, not as token 4294967295.
    #[test]
    fn an_absent_entry_reads_back_as_an_ignored_label() {
        let sidecar = TopKSidecar::new(
            header(2, 1),
            vec![3, IGNORE_LABEL],
            vec![-0.2, f32::NEG_INFINITY],
        )
        .expect("shape");
        let path = std::env::temp_dir().join("retrograd-topk-absent.bin");
        sidecar.write(&path).expect("write");
        let read = TopKSidecar::read(&path).expect("read");
        assert_eq!(read.ids(), &[3, IGNORE_LABEL]);
        assert!(read.logprobs()[1].is_infinite());
        let (labels, weights) = read.weighted_targets(&dataset(vec![3])).expect("targets");
        assert_eq!(labels, vec![3, IGNORE_LABEL]);
        assert_eq!(weights, vec![1.0, 0.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_sidecar_for_another_preparation_is_refused_by_shape() {
        let sidecar = TopKSidecar::new(header(1, 3), vec![1, 2, 3], vec![-0.1; 3]).expect("shape");
        let error = sidecar
            .check_against(&dataset(vec![1, 2]), 32, 7, 9)
            .unwrap_err()
            .to_string();
        assert!(error.contains("another preparation"), "{error}");
    }

    #[test]
    fn a_sidecar_from_another_vocabulary_or_corpus_is_refused_by_its_hashes() {
        let sidecar = TopKSidecar::new(header(1, 2), vec![1, 2], vec![-0.1; 2]).expect("shape");
        let data = dataset(vec![1, 2]);
        let vocabulary = sidecar
            .check_against(&data, 64, 7, 9)
            .unwrap_err()
            .to_string();
        assert!(vocabulary.contains("vocabulary of 32"), "{vocabulary}");
        let tokenizer = sidecar
            .check_against(&data, 32, 8, 9)
            .unwrap_err()
            .to_string();
        assert!(tokenizer.contains("tokenizer disagrees"), "{tokenizer}");
        let corpus = sidecar
            .check_against(&data, 32, 7, 10)
            .unwrap_err()
            .to_string();
        assert!(corpus.contains("another corpus"), "{corpus}");
        sidecar.check_against(&data, 32, 7, 9).expect("matching");
    }

    /// Two preparations that differ only by their masking are different corpora
    /// for this purpose: the sidecar's blocks are indexed by position, and a
    /// mask change moves what a position means.
    #[test]
    fn the_corpus_fingerprint_separates_two_preparations_of_one_file() {
        let a = corpus_fingerprint(&dataset(vec![1, 2, 3]));
        let b = corpus_fingerprint(&dataset(vec![1, IGNORE_LABEL, 3]));
        assert_ne!(a, b);
        assert_eq!(a, corpus_fingerprint(&dataset(vec![1, 2, 3])));
    }

    #[test]
    fn the_tokenizer_fingerprint_follows_the_ids_and_not_the_sentences() {
        let a = tokenizer_fingerprint(100, &[vec![1, 2, 3]]);
        assert_eq!(a, tokenizer_fingerprint(100, &[vec![1, 2, 3]]));
        assert_ne!(a, tokenizer_fingerprint(100, &[vec![1, 2, 4]]));
        assert_ne!(a, tokenizer_fingerprint(101, &[vec![1, 2, 3]]));
    }
}
