use super::*;

/// Sentences two vocabularies must agree on, id for id.
///
/// Covers what actually differs between two vocabularies of the same family:
/// ASCII words, digit grouping, a leading-space merge, non-Latin scripts and
/// an astral-plane codepoint. No chat control marker on purpose: instruct and
/// base checkpoints may differ in template while their vocabulary is the same.
pub const VOCABULARY_WITNESSES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Distillation: 3.14159 nats, 1_000_000 tokens, 42% done.",
    "Écrire 你好世界 puis 🙂 sans rien perdre.",
];

/// How two models' vocabularies differ. Returned rather than raised so each
/// caller can name both sides in its own words (teacher/student, anchor/policy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabularyMismatch {
    Size {
        left: usize,
        right: usize,
    },
    Tokens {
        sentence: &'static str,
        left: Vec<i32>,
        right: Vec<i32>,
    },
}

impl Trainer {
    /// Whether this model and `other` tokenize identically, over the vocabulary
    /// size and [`VOCABULARY_WITNESSES`]. `None` is agreement.
    ///
    /// Scoring one model's ids under the other's vocabulary returns finite
    /// log-probabilities of the wrong tokens, so this check is cheap insurance
    /// against silent nonsense.
    pub fn vocabulary_mismatch(&self, other: &Trainer) -> Result<Option<VocabularyMismatch>> {
        let left = self.vocab_size()?;
        let right = other.vocab_size()?;
        if left != right {
            return Ok(Some(VocabularyMismatch::Size { left, right }));
        }
        for sentence in VOCABULARY_WITNESSES {
            let left = self.tokenize_text(sentence)?;
            let right = other.tokenize_text(sentence)?;
            if left != right {
                return Ok(Some(VocabularyMismatch::Tokens {
                    sentence,
                    left,
                    right,
                }));
            }
        }
        Ok(None)
    }

    /// Loads the frozen model this run's reference term scores against, and
    /// checks it can stand in for the anchor before use.
    ///
    /// Forward-only: no adapter, no backward graph, no optimizer state, so the
    /// whole cost is weights plus KV.
    ///
    /// `n_ctx` overrides the training width for the anchor alone; the rest of
    /// the geometry travels from `training`, because the anchor scores the same
    /// batches the policy does.
    pub fn attach_reference(
        &mut self,
        path: impl AsRef<Path>,
        training: &TrainConfig,
        n_ctx: Option<u32>,
    ) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(Error::config(format!(
                "reference model not found: {}",
                path.display()
            )));
        }
        let started = Instant::now();
        let reference = Trainer::new(
            &path,
            TrainConfig {
                device: training.device,
                n_ctx: n_ctx.unwrap_or(training.n_ctx),
                n_batch: training.n_batch,
                n_ubatch: training.n_ubatch,
                n_seq_max: training.n_seq_max,
                threads: training.threads,
                kv_dtype: training.kv_dtype,
                ..TrainConfig::default()
            },
        )?;
        test_timing("reference_load", started);
        // Identity before use: the ids must mean the same thing in both models,
        // and the scores must be log-probabilities. Whether the two models were
        // trained on the same objective cannot be checked from a file.
        if let Some(mismatch) = reference.vocabulary_mismatch(self)? {
            return Err(Error::config(match mismatch {
                VocabularyMismatch::Size { left, right } => format!(
                    "reference model {} has vocabulary size {left}, this run's model has \
                     {right}: an anchor can only score this policy's tokens if both share \
                     one tokenizer",
                    path.display()
                ),
                VocabularyMismatch::Tokens {
                    sentence,
                    left,
                    right,
                } => format!(
                    "reference model {} tokenizes {sentence:?} as {left:?}, this run's model \
                     as {right:?}: the two models do not share one tokenizer",
                    path.display()
                ),
            }));
        }
        let mut reference = reference;
        check_scoring_convention(&mut reference, &path)?;
        self.reference = Some(Box::new(reference));
        self.reference_path = Some(path);
        Ok(())
    }

    /// The model this run's anchor was loaded from, when it has one.
    pub fn reference_path(&self) -> Option<&Path> {
        self.reference_path.as_deref()
    }

    /// Content fingerprint of the anchor file, empty when there is no anchor.
    /// Recorded in checkpoints so a resume cannot quietly measure its penalty
    /// against a different anchor.
    pub fn reference_fingerprint(&self) -> Result<String> {
        match &self.reference_path {
            Some(path) => checkpoint::fingerprint_file_cached(path),
            None => Ok(String::new()),
        }
    }

    /// What the anchor costs: its weights and its KV, no gradients and no
    /// optimizer state.
    pub fn reference_memory_report(&self) -> Result<Option<MemoryReport>> {
        self.reference
            .as_ref()
            .map(|reference| reference.memory_report())
            .transpose()
    }
}

/// Refuses an anchor whose scores are not log-probabilities: one
/// teacher-forced pass, checking that every score is finite and at most zero.
fn check_scoring_convention(reference: &mut Trainer, path: &Path) -> Result<()> {
    let witness = VOCABULARY_WITNESSES[0];
    let mut tokens = reference.tokenize_text(witness)?;
    tokens.truncate(reference.context_size()?);
    if tokens.len() < 2 {
        return Err(Error::config(format!(
            "reference model {} tokenizes {witness:?} into fewer than two tokens, so its \
             scoring convention cannot be checked",
            path.display()
        )));
    }
    let scores = reference.score_tokens(&tokens)?;
    if let Some(bad) = scores
        .iter()
        .copied()
        .find(|score| !score.is_finite() || *score > 0.0)
    {
        return Err(Error::config(format!(
            "reference model {} scored a token at {bad}, which is not a log-probability: \
             an anchor has to answer in the units the penalty subtracts",
            path.display()
        )));
    }
    Ok(())
}
