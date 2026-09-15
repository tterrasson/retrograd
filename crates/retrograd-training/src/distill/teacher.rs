//! The teacher of an on-policy distillation run: a second model, held for
//! inference only, that scores the tokens the student sampled.
//!
//! Everything expensive about a `Trainer` is created by `create_lora`: the
//! backward graph is defined by the trainable parameters, so an adapter is what
//! brings gradients, AdamW moments and the differentiable context into
//! existence. A `Trainer` that never gets one is an inference object, and that
//! is the whole reason a second model is affordable here - weights plus KV,
//! nothing else.

use std::cell::{RefCell, RefMut};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use retrograd_core::{Error, MemoryReport, Result, TrainConfig};
use retrograd_engine::{TopLogprobs, Trainer};

use crate::TokenSpan;
use crate::rollout::score_train_mask_group;

/// Sentences the two tokenizers must agree on, id for id.
/// Chosen to cover what actually differs between two vocabularies of the same
/// family: ASCII words, digit grouping, a leading-space merge, non-Latin
/// scripts and an astral-plane codepoint. No chat control marker appears here
/// on purpose - a teacher and a student may legitimately be an instruct and a
/// base checkpoint, whose templates differ while their vocabulary does not, and
/// scoring the student's tokens only depends on the vocabulary.
pub const WITNESS_SENTENCES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Distillation: 3.14159 nats, 1_000_000 tokens, 42% done.",
    "Écrire 你好世界 puis 🙂 sans rien perdre.",
];

/// A frozen model that notes the student's tokens.
pub struct Teacher {
    trainer: Trainer,
    path: PathBuf,
}

impl std::fmt::Debug for Teacher {
    /// The model it holds, not the runtime handle it holds it by.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Teacher")
            .field("path", &self.path)
            .finish()
    }
}

impl Teacher {
    /// Loads the teacher for inference. The training configuration is read for
    /// the terms that describe *where and how wide* the forward pass runs, and
    /// for nothing else: no epochs, no learning rate, no optimizer.
    ///
    /// The geometry travels with it because the teacher runs the very same
    /// shared-prefix scoring batch as the student - a group of `n` completions
    /// is `n` branched sequences, and a context built with the default
    /// `n_seq_max = 1` would refuse them.
    pub fn open(path: impl AsRef<Path>, training: &TrainConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(Error::config(format!(
                "distillation teacher model not found: {}",
                path.display()
            )));
        }
        let trainer = Trainer::new(
            &path,
            TrainConfig {
                device: training.device,
                n_ctx: training.n_ctx,
                n_batch: training.n_batch,
                n_ubatch: training.n_ubatch,
                n_seq_max: training.n_seq_max,
                threads: training.threads,
                kv_dtype: training.kv_dtype,
                ..TrainConfig::default()
            },
        )?;
        Ok(Self { trainer, path })
    }

    /// Log-probabilities the teacher assigns to the student's tokens, one row
    /// per span and one value per supervised position, in increasing token
    /// order. One shared-prefix forward pass per group, exactly like the
    /// student's own scoring: the prompt is decoded once and branched.
    ///
    /// Spans are borrowed individually rather than as a slice because a caller
    /// may score a *subset* of a group - truncation masking drops members, and
    /// the survivors are not contiguous.
    pub fn logprobs_group<S: TokenSpan + ?Sized>(&mut self, spans: &[&S]) -> Result<Vec<Vec<f32>>> {
        score_train_mask_group(&mut self.trainer, spans)
    }

    /// Log-probabilities of one sequence's completion targets, through the
    /// same single-sequence call the student's evaluation uses.
    ///
    /// The pairing matters more than the call does: `llama_decode` is not
    /// invariant by batch, so a divergence measured with the teacher batched and
    /// the student not would carry the difference between two tile arrangements
    /// on top of the difference between two models.
    pub fn logprobs_suffix(&mut self, tokens: &[i32], n_prompt: usize) -> Result<Vec<f32>> {
        self.trainer.score_token_suffix(tokens, n_prompt)
    }

    /// The teacher's truncated distribution over the same targets: the
    /// `top_logprobs_suffix` of the model this holds. Column zero is the
    /// teacher's argmax, which is the half of a top-1 agreement the student
    /// cannot supply.
    pub fn top_logprobs_suffix(
        &mut self,
        tokens: &[i32],
        n_prompt: usize,
        k: usize,
    ) -> Result<TopLogprobs> {
        self.trainer.top_logprobs_suffix(tokens, n_prompt, k)
    }

    /// Refuses a pair whose tokenizers disagree.
    ///
    /// This is a gate, not a warning. Scoring the student's ids under a
    /// different vocabulary returns log-probabilities of *other tokens*: the
    /// numbers are finite, the run trains, and the objective is nonsense. It is
    /// the worst failure mode this algorithm has, and the only one that costs a
    /// single forward pass to rule out.
    pub fn compatibility(&self, student: &Trainer) -> Result<()> {
        let teacher_vocab = self.trainer.vocab_size()?;
        let student_vocab = student.vocab_size()?;
        if teacher_vocab != student_vocab {
            return Err(Error::config(format!(
                "distillation teacher {} has vocabulary size {teacher_vocab}, the student has \
                 {student_vocab}: the teacher can only score the student's tokens if both models \
                 share one tokenizer",
                self.path.display()
            )));
        }
        for sentence in WITNESS_SENTENCES {
            let teacher = self.trainer.tokenize_text(sentence)?;
            let student = student.tokenize_text(sentence)?;
            if teacher != student {
                return Err(Error::config(format!(
                    "distillation teacher {} tokenizes {sentence:?} as {teacher:?}, the student as \
                     {student:?}: the two models do not share one tokenizer",
                    self.path.display()
                )));
            }
        }
        Ok(())
    }

    /// What this teacher costs, for the invariant it is loaded under: its
    /// weights and its KV, no gradients and no optimizer state. Published
    /// because "a forward-only trainer allocates nothing else" is a claim worth
    /// measuring rather than repeating.
    pub fn memory_report(&self) -> Result<MemoryReport> {
        self.trainer.memory_report()
    }

    /// The model this teacher was loaded from, for diagnostics and for the
    /// resume signature.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The teacher of one run, opened at most once and shared by the update loop
/// and the scheduled evaluation.
///
/// One model, not two. The single risk that can make this algorithm unusable on
/// a given machine is holding a second model beside the student and its AdamW
/// state; an evaluation that opened its own teacher would double exactly that
/// term, and would do it at the update boundary where the optimizer state is
/// resident. `Rc<RefCell<_>>` and not a lock because a run is one thread: the
/// loop and its progress callback never hold the teacher at the same moment,
/// and a borrow that overlapped would be a bug this makes loud instead of
/// silently serializing.
#[derive(Clone, Default)]
pub struct SharedTeacher(Rc<RefCell<Option<Teacher>>>);

impl std::fmt::Debug for SharedTeacher {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.try_borrow() {
            Ok(slot) => formatter
                .debug_tuple("SharedTeacher")
                .field(&*slot)
                .finish(),
            Err(_) => formatter.write_str("SharedTeacher(<in use>)"),
        }
    }
}

impl SharedTeacher {
    pub fn new() -> Self {
        Self::default()
    }

    /// The teacher, loading it on the first call. Later callers get the model
    /// already resident, which is the whole point of the type.
    pub fn get_or_open(
        &self,
        path: impl AsRef<Path>,
        training: &TrainConfig,
    ) -> Result<RefMut<'_, Teacher>> {
        let mut slot = self.0.try_borrow_mut().map_err(|_| {
            Error::runtime("the distillation teacher is already in use on this thread")
        })?;
        if slot.is_none() {
            *slot = Some(Teacher::open(path, training)?);
        }
        Ok(RefMut::map(slot, |slot| {
            slot.as_mut().expect("the teacher was just opened")
        }))
    }

    /// Whether the teacher has been loaded yet. The scheduled evaluation is the
    /// only caller that can reach a run's teacher before its first update.
    pub fn is_open(&self) -> bool {
        self.0.try_borrow().is_ok_and(|slot| slot.is_some())
    }
}
