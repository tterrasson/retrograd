//! The safe trainer API over the C++ runtime.
//!
//! [`Trainer`] owns one loaded model with its LoRA adapter and optimizer, and
//! turns each call into the `retrograd-ffi` entry point that performs it, with
//! typed errors instead of status codes. The training loops of
//! `retrograd-training` drive it; nothing here knows what an algorithm is.

#![warn(clippy::undocumented_unsafe_blocks)]

use std::ffi::{CString, NulError, c_char, c_void};
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;
use std::time::Instant;

use retrograd_checkpoint as checkpoint;
use retrograd_core::{
    CheckpointMetadata, Device, EvalMetrics, FusedCeProbe, FusedCeWeightType, Generation,
    KernelImpl, KernelReject, KernelRunInfo, LoraConfig, MemoryReport, ModelInfo, OpPlacement,
    OptimizerKind, PreflightReport, PreflightWarning, ProbeOp, ResumeInfo, RirCounters, RirMode,
    SamplingParams, TensorDesc, TensorDtype, TensorInventory, TrainConfig, TrainMetrics,
    WeightedBatch,
};
use retrograd_core::{Error, Result};
use retrograd_dataset::PreparedDataset;
use retrograd_ffi as ffi;

#[cfg(test)]
use retrograd_core::{CheckpointDtype, KvDtype, LrScheduler};

type TokenRows = Vec<Vec<i32>>;
type LogprobRows = Vec<Vec<f32>>;
type GenerationBatch = (TokenRows, Option<LogprobRows>);

/// Phase timings for a human reading a test lane (`RETRO_TEST_TIMING`): tests
/// install no `tracing` subscriber, so this writes to stderr directly.
fn test_timing(label: &str, start: Instant) {
    if std::env::var_os("RETRO_TEST_TIMING").is_some() {
        eprintln!(
            "test_phase={label} duration_ms={}",
            start.elapsed().as_millis()
        );
    }
}

fn validate_train_tokens(tokens: &[i32]) -> Result<()> {
    if tokens.len() < 2 {
        return Err(Error::tokenize("training requires at least two tokens"));
    }
    Ok(())
}

fn first_masked_target(tokens: &[i32], train_mask: &[bool]) -> Result<usize> {
    if tokens.len() != train_mask.len() {
        return Err(Error::invalid(
            "tokens and train_mask must have the same length",
        ));
    }
    let first = train_mask
        .iter()
        .position(|&train| train)
        .ok_or_else(|| Error::tokenize("train_mask selects no tokens"))?;
    if first == 0 {
        return Err(Error::invalid(
            "the first token cannot be selected for teacher-forced scoring",
        ));
    }
    Ok(first)
}

fn validate_suffix(tokens: &[i32], n_prompt: usize) -> Result<()> {
    if n_prompt == 0 || n_prompt >= tokens.len() {
        return Err(Error::invalid(
            "suffix scoring requires a prompt and at least one completion token",
        ));
    }
    Ok(())
}

fn train_config_to_ffi(config: &TrainConfig) -> Result<ffi::RetroTrainConfig> {
    let gefen_layout = config.trainable.optimizer.gefen_layout();
    // AdamW declares `beta1`, `beta2` and `eps` too, and they are not Gefen's:
    // the wire fields are read only under Gefen, so they are sent only then.
    let gefen_beta = |name: &str| {
        if gefen_layout.is_some() {
            config.optimizer_scalar(name)
        } else {
            0.0
        }
    };
    Ok(ffi::RetroTrainConfig {
        n_ctx: config.n_ctx,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_seq_max: config.n_seq_max,
        generation_concurrency: config.effective_generation_concurrency(),
        fast_generation_context: config.fast_generation_context,
        kv_dtype: config.kv_dtype.as_ffi(),
        threads: config.threads,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        weight_decay: config.weight_decay,
        max_grad_norm: config.max_grad_norm,
        lr_scheduler: config.lr_scheduler.as_ffi(),
        warmup_steps: config.warmup_steps,
        verbose: config.verbose,
        device: config.device.as_ffi(),
        chunked_cross_entropy: config.chunked_cross_entropy,
        chunked_ce_tiles: config.chunked_ce_tiles.max(1),
        chunked_ce_seq_chunk: config.chunked_ce_seq_chunk,
        // Derived, not configured. The in-place `grad_h` write is numerically
        // inert and only needs a token chunk to bound its staging buffer, so
        // the one condition under which it is legal is also the only condition
        // under which anyone would decline it. The runtime keeps the field
        // because that is where the graph reads it.
        chunked_ce_offload_logsoftmax: config.chunked_ce_seq_chunk > 0,
        gradient_checkpointing: config.gradient_checkpointing,
        checkpoint_every_n_layers: config.checkpoint_every_n_layers.max(1),
        checkpoint_dtype: config.checkpoint_dtype.as_ffi(),
        master_weights: config.master_weights.as_ffi(),
        require_gpu_resident: config.require_gpu_resident,
        generation_batch: config.generation_batch,
        shuffle_dataset: config.shuffle_dataset,
        optimizer: config.trainable.optimizer.as_ffi(),
        shuffle_seed: config.shuffle_seed,
        trainable: config.trainable.policy.as_ffi(),
        // One declared row per wire field. The runtime treats zero as "the
        // frozen default", and a row the chosen optimizer does not declare
        // reads as zero here, so a Gefen run sends no Muon coefficients and
        // an AdamW run sends neither optimizer's.
        muon_momentum: config.optimizer_scalar("momentum"),
        muon_ns_epsilon: config.optimizer_scalar("ns_epsilon"),
        muon_fallback_learning_rate: config.optimizer_scalar("fallback_learning_rate"),
        muon_ns_steps: config.optimizer_structural("ns_steps"),
        muon_nesterov: config.optimizer_toggle("nesterov", true),
        // Structural, so it comes from the layout the optimizer value carries
        // rather than from the coefficient vector.
        gefen_variant: gefen_layout
            .map(|layout| layout.variant.as_ffi())
            .unwrap_or(0),
        gefen_block_size: gefen_layout
            .map(|layout| u32::try_from(layout.block_size).unwrap_or(u32::MAX))
            .unwrap_or(0),
        gefen_beta1: gefen_beta("beta1"),
        gefen_beta2: gefen_beta("beta2"),
        gefen_eps: gefen_beta("eps"),
    })
}

fn train_metrics_from_ffi(metrics: ffi::RetroTrainMetrics) -> TrainMetrics {
    TrainMetrics {
        epoch: metrics.epoch,
        epoch_complete: metrics.epoch_complete,
        global_step: metrics.global_step,
        train_loss: metrics.train_loss,
        eval_loss: metrics.eval_loss,
        tokens_per_second: metrics.tokens_per_second,
        learning_rate: metrics.learning_rate,
    }
}

fn sampling_params_to_ffi(params: &SamplingParams) -> ffi::RetroSamplingParams {
    ffi::RetroSamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        max_new_tokens: params.max_new_tokens,
        seed: params.seed,
    }
}

struct FfiCallbackState<F> {
    trainer: *mut Trainer,
    callback: F,
    error: Option<Error>,
    keep_training: bool,
}

unsafe extern "C" fn ffi_callback_trampoline<F>(
    _epoch: u32,
    metrics: *const ffi::RetroTrainMetrics,
    user_data: *mut c_void,
) -> bool
where
    F: FnMut(&mut Trainer, TrainMetrics) -> Result<bool>,
{
    if metrics.is_null() || user_data.is_null() {
        return false;
    }
    // The state is created immediately before the synchronous FFI call and
    // remains valid until it returns.
    // SAFETY: `user_data` points to the callback state kept alive by the synchronous FFI call.
    let state = unsafe { &mut *(user_data as *mut FfiCallbackState<F>) };
    if state.error.is_some() {
        return false;
    }
    // SAFETY: the callback state carries the live, exclusively borrowed trainer for this call.
    let trainer = unsafe { &mut *state.trainer };
    // SAFETY: the null check above guarantees a live metrics value supplied by the runtime.
    let metrics = train_metrics_from_ffi(unsafe { *metrics });
    match (state.callback)(trainer, metrics) {
        Ok(keep_training) => {
            state.keep_training = keep_training;
            keep_training
        }
        Err(error) => {
            state.error = Some(error);
            false
        }
    }
}

macro_rules! read_string_method {
    ($(#[$meta:meta])* $name:ident, $ffi_name:ident) => {
        $(#[$meta])*
        pub fn $name(&self) -> Result<String> {
            // SAFETY: the helper owns a live allocation for every raw pointer
            // passed to the runtime, and `retro_*` string getters write the
            // byte count they report.
            unsafe {
                read_string(|buffer, n_buffer, out| {
                    ffi::$ffi_name(self.raw.as_ptr(), buffer, n_buffer, out)
                })
            }
        }
    };
    ($(#[$meta:meta])* $name:ident, $ffi_name:ident, mut) => {
        $(#[$meta])*
        pub fn $name(&mut self) -> Result<String> {
            // SAFETY: the helper owns a live allocation for every raw pointer
            // passed to the runtime, and `retro_*` string getters write the
            // byte count they report.
            unsafe {
                read_string(|buffer, n_buffer, out| {
                    ffi::$ffi_name(self.raw.as_ptr(), buffer, n_buffer, out)
                })
            }
        }
    };
    (private $(#[$meta:meta])* $name:ident, $ffi_name:ident) => {
        $(#[$meta])*
        fn $name(&mut self) -> Result<String> {
            // SAFETY: the helper owns a live allocation for every raw pointer
            // passed to the runtime, and `retro_*` string getters write the
            // byte count they report.
            unsafe {
                read_string(|buffer, n_buffer, out| {
                    ffi::$ffi_name(self.raw.as_ptr(), buffer, n_buffer, out)
                })
            }
        }
    };
}

/// A model's truncated next-token distribution over the completion targets of
/// one sequence: `k` (id, log-probability) pairs per target position, in
/// decreasing probability order.
///
/// Stored flat, one allocation per call rather than one per position: the two
/// consumers are a metric that folds every row into a scalar and a sidecar
/// writer that appends rows back to back, and neither wants `Vec<Vec<_>>`.
#[derive(Clone, Debug, PartialEq)]
pub struct TopLogprobs {
    k: usize,
    ids: Vec<i32>,
    logprobs: Vec<f32>,
}

impl TopLogprobs {
    /// Wraps two arrays the runtime just filled. Crate-private: the invariant
    /// that both hold `rows * k` entries is the runtime's, established by the
    /// capacity it was handed and checked there.
    pub(crate) fn from_flat(k: usize, ids: Vec<i32>, logprobs: Vec<f32>) -> Self {
        debug_assert_eq!(ids.len(), logprobs.len());
        Self { k, ids, logprobs }
    }

    /// Entries per position. Always at least one, and never above the
    /// vocabulary size - the runtime refuses both.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Target positions covered, i.e. `tokens.len() - n_prompt`.
    pub fn rows(&self) -> usize {
        self.ids.len() / self.k
    }

    /// One position's entries, ids and log-probabilities in the same order.
    /// `None` past the last row rather than a panic: a caller indexing by a
    /// token offset it computed elsewhere is the expected way in.
    pub fn row(&self, index: usize) -> Option<(&[i32], &[f32])> {
        let start = index.checked_mul(self.k)?;
        let end = start.checked_add(self.k)?;
        if end > self.ids.len() {
            return None;
        }
        Some((&self.ids[start..end], &self.logprobs[start..end]))
    }

    /// Every position in increasing token order.
    pub fn iter(&self) -> impl Iterator<Item = (&[i32], &[f32])> {
        self.ids.chunks(self.k).zip(self.logprobs.chunks(self.k))
    }

    /// The most probable token of each position and its log-probability - the
    /// `k = 1` view of a row scored with a wider `k`.
    pub fn argmax(&self) -> impl Iterator<Item = (i32, f32)> + '_ {
        self.iter().map(|(ids, logprobs)| (ids[0], logprobs[0]))
    }
}

pub struct Trainer {
    // Invariant for every FFI call made by the trainer modules: `raw` is the
    // unique non-null handle returned by `retro_trainer_new`, remains live for
    // the whole Rust value, and is freed exactly once by `Drop`. All slice,
    // string and out-parameter pointers passed beside it are borrowed only for
    // the duration of the synchronous call. The progress callback is the sole
    // exception; `run_with_callback` pins its state until that call returns.
    raw: NonNull<ffi::RetroTrainer>,
    /// What this run trains, kept beside the handle because the runtime takes
    /// it as a scalar at creation and never hands it back, and because three
    /// different answers depend on it: whether a checkpoint carries a bundle,
    /// whether a score may reuse an adapter-free context, and which policy a
    /// resume is compared against.
    trainable_policy: retrograd_core::TrainablePolicy,
    /// The resolved set this run declared, when it declared one.
    ///
    /// Kept because a checkpoint's trainable signature has to be available
    /// *before* the optimizer graph exists - a resume compares it while
    /// deciding whether to restore at all - while the marked set only exists
    /// after. Rule 6 is what makes the two interchangeable once both do.
    declared_trainable: Option<retrograd_core::TrainableSet>,
    /// Which optimizer owns each marked parameter, when an assignment was
    /// declared; empty for a single-optimizer run.
    declared_assignment: Vec<(String, retrograd_core::OptimizerKind)>,
    /// The optimizer this run's [`TrainConfig`] named. Kept because an
    /// optimizer with an eligibility rule owns only part of a set, and the
    /// runtime puts every marked parameter on the run's optimizer unless an
    /// assignment says otherwise - so the plan and the live table would
    /// describe two different runs with nobody asking for it.
    chosen_optimizer: retrograd_core::OptimizerKind,
    /// Whether this run keeps an F32 master copy of its half-precision
    /// parameters. Kept beside the handle for the same reason the optimizer
    /// is: it decides which slots the runtime allocates, so a plan built
    /// without it would describe a different run.
    master_weights: retrograd_core::MasterWeights,
    /// The frozen model this run's reference term scores against, when one was
    /// attached. `generate_base` and `score_reference_tokens` fall back to
    /// this model with its adapter disabled when it is absent. Boxed to keep
    /// `Trainer` a bounded type.
    reference: Option<Box<Trainer>>,
    /// The file that anchor was loaded from, kept for the checkpoint's
    /// fingerprint and diagnostics.
    reference_path: Option<std::path::PathBuf>,
}

impl Trainer {
    /// What this run trains.
    pub fn trainable_policy(&self) -> retrograd_core::TrainablePolicy {
        self.trainable_policy
    }

    /// The resolved set this run declared, if any.
    pub fn declared_trainable_set(&self) -> Option<&retrograd_core::TrainableSet> {
        self.declared_trainable.as_ref()
    }

    /// The declared per-parameter assignment; empty for a single-optimizer run.
    pub fn declared_optimizer_assignment(&self) -> &[(String, retrograd_core::OptimizerKind)] {
        &self.declared_assignment
    }

    /// The optimizer this run's configuration named.
    pub fn chosen_optimizer(&self) -> retrograd_core::OptimizerKind {
        self.chosen_optimizer
    }

    /// Whether this run keeps an F32 master copy, for the set it is about to
    /// plan. `auto` asks the base half of the set, which is where a storage
    /// grid can be coarser than the step; naming it asks for it everywhere.
    /// The same resolution the runtime performs, and it has to be, or the plan
    /// and the live table would count different slots.
    pub(crate) fn keeps_master_copy(&self, set: &retrograd_core::TrainableSet) -> bool {
        match self.master_weights {
            retrograd_core::MasterWeights::Off => false,
            retrograd_core::MasterWeights::F32 => true,
            retrograd_core::MasterWeights::Auto => set.base_entries().any(|entry| {
                matches!(
                    entry.dtype,
                    retrograd_core::TensorDtype::F16 | retrograd_core::TensorDtype::BF16
                )
            }),
        }
    }

    /// The slot-layout version this run allocates under: the optimizer's own,
    /// moved by one when the run keeps an F32 master copy.
    ///
    /// Read off the run rather than from the optimizer's name, because a
    /// master copy is the run's decision and adds a slot to whatever the
    /// optimizer declared. It is what a resume compares, so a payload written
    /// with masters is refused by name before its row count is counted.
    pub fn optimizer_layout_version(&mut self) -> Result<u32> {
        let kind = self.chosen_optimizer;
        let set = match &self.declared_trainable {
            Some(set) => set.clone(),
            None => self.marked_trainable_set().unwrap_or_default(),
        };
        Ok(kind.layout_version_with_master(self.keeps_master_copy(&set)))
    }

    /// The declared state table for this run: the optimizer's own policy
    /// unless an assignment was declared, in which case it is that assignment.
    pub(crate) fn optimizer_plan(
        &self,
        kind: retrograd_core::OptimizerKind,
        set: &retrograd_core::TrainableSet,
    ) -> retrograd_core::OptimizerPlan {
        let master = self.keeps_master_copy(set);
        if self.declared_assignment.is_empty() {
            return kind.plan_with_master(set, master, |entry| kind.assign(entry));
        }
        kind.plan_with_master(set, master, |entry| {
            self.declared_assignment
                .iter()
                .find(|(name, _)| *name == entry.name)
                .map(|(_, optimizer)| *optimizer)
                .or_else(|| kind.assign(entry))
        })
    }

    fn trains_base_weights(&self) -> bool {
        self.trainable_policy.trains_base_weights()
    }
}

mod packed;
mod probe;
mod quant;
mod rir;
mod stats;
mod trainer;

pub use packed::PackedSequenceBatch;
pub use probe::*;
pub use quant::*;
pub use rir::*;
pub use stats::{DutyCycleStats, GenerationStats, OptimizerMemory, OptimizerTiming, ScoringStats};
pub use trainer::{
    VOCABULARY_WITNESSES, VocabularyMismatch, parse_assistant_output, render_chat_template_source,
    tool_call_parser_from_source,
};

// The "uninitialized buffer filled through FFI" helpers below each mark a
// buffer initialized on the strength of what their closure reports. Nothing
// they can check proves the write happened, so the obligation is the caller's
// and the helpers are `unsafe fn`: calling one asserts that the runtime entry
// point inside the closure initializes the entries it reports before returning
// success. A failed call leaves the buffer logically empty.

/// Allocates `capacity` uninitialized slots, lets the runtime fill them, and
/// keeps the number of entries `fill` reports written.
///
/// # Safety
///
/// `fill` must initialize `n` entries at the pointer it is given whenever it
/// returns `Ok(n)`.
unsafe fn ffi_out_vec<T>(
    capacity: usize,
    fill: impl FnOnce(*mut T) -> Result<usize>,
) -> Result<Vec<T>> {
    let mut out = Vec::<T>::with_capacity(capacity);
    let written = fill(out.as_mut_ptr())?;
    if written > capacity {
        return Err(Error::runtime(format!(
            "runtime reported {written} output entries for a buffer of {capacity}"
        )));
    }
    // SAFETY: the caller's obligation, and `written` fits the allocation.
    unsafe { out.set_len(written) };
    Ok(out)
}

/// [`ffi_out_vec`] into a reused buffer: clears it and has the runtime write
/// exactly `len` entries.
///
/// # Safety
///
/// `fill` must initialize `len` entries at the pointer it is given whenever it
/// returns `Ok(())`.
unsafe fn ffi_refill_vec<T>(
    out: &mut Vec<T>,
    len: usize,
    fill: impl FnOnce(*mut T) -> Result<()>,
) -> Result<()> {
    out.clear();
    // SAFETY: forwarded to the caller, whose `fill` this is.
    unsafe { ffi_extend_vec(out, len, fill) }
}

/// [`ffi_out_vec`] appending to a growing buffer: the runtime writes
/// `additional` entries after the current end.
///
/// # Safety
///
/// `fill` must initialize `additional` entries at the pointer it is given
/// whenever it returns `Ok(())`.
unsafe fn ffi_extend_vec<T>(
    out: &mut Vec<T>,
    additional: usize,
    fill: impl FnOnce(*mut T) -> Result<()>,
) -> Result<()> {
    out.reserve(additional);
    let start = out.len();
    // SAFETY: `reserve` covers `additional` entries past `start`.
    fill(unsafe { out.as_mut_ptr().add(start) })?;
    // SAFETY: the caller's obligation.
    unsafe { out.set_len(start + additional) };
    Ok(())
}

/// Runs the two-call (size probe, then fill) pattern shared by the runtime's
/// string-returning entry points.
///
/// # Safety
///
/// `call` must write `*out` content bytes plus a trailing NUL into the buffer
/// it is given whenever it returns 0.
unsafe fn read_string<F>(mut call: F) -> Result<String>
where
    F: FnMut(*mut c_char, usize, *mut usize) -> i32,
{
    let mut needed = 0_usize;
    let code = call(std::ptr::null_mut(), 0, &mut needed);
    if code != 0 {
        return Err(runtime_error());
    }
    // SAFETY: forwarded to the caller, whose `call` this is.
    string_from_runtime(unsafe { read_bytes_exact(needed, &mut call) }?)
}

/// Like [`read_string`], but skips the size probe by optimistically calling
/// with `size_hint` bytes of capacity, retrying once with the exact size the
/// runtime reports when the hint was too small (-2). Worth it on endpoints
/// whose probe re-renders the full text, like detokenization.
///
/// Decodes leniently: unlike the runtime's own diagnostics, rendered model
/// text can legitimately end on a partial multi-byte character.
///
/// # Safety
///
/// Same obligation as [`read_string`].
unsafe fn read_string_sized_lossy<F>(size_hint: usize, mut call: F) -> Result<String>
where
    F: FnMut(*mut c_char, usize, *mut usize) -> i32,
{
    let capacity = size_hint.max(1);
    let mut needed = 0_usize;
    let mut buffer = Vec::<u8>::with_capacity(capacity);
    let code = call(buffer.as_mut_ptr().cast(), capacity, &mut needed);
    if code == 0 {
        if needed > capacity {
            return Err(Error::runtime(format!(
                "runtime reported a {needed}-byte string for a {capacity}-byte buffer"
            )));
        }
        // SAFETY: the caller's obligation, and `needed` fits the allocation.
        unsafe { buffer.set_len(needed) };
        return Ok(string_from_runtime_lossy(buffer));
    }
    if code != -2 {
        return Err(runtime_error());
    }
    // -2 reports the required size in `needed`.
    // SAFETY: forwarded to the caller, whose `call` this is.
    Ok(string_from_runtime_lossy(unsafe {
        read_bytes_exact(needed, &mut call)
    }?))
}

/// # Safety
///
/// Same obligation as [`read_string`].
unsafe fn read_bytes_exact<F>(needed: usize, call: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(*mut c_char, usize, *mut usize) -> i32,
{
    let capacity = needed
        .checked_add(1)
        .ok_or_else(|| Error::runtime("runtime string size overflows usize"))?;
    // SAFETY: forwarded to the caller: `call` writes `written` content bytes
    // plus a trailing NUL when it returns 0.
    let buffer = unsafe {
        ffi_out_vec::<u8>(capacity, |buffer| {
            let mut written = 0_usize;
            if call(buffer.cast(), capacity, &mut written) != 0 {
                return Err(runtime_error());
            }
            Ok(written)
        })
    }?;
    Ok(buffer)
}

fn string_from_runtime(buffer: Vec<u8>) -> Result<String> {
    String::from_utf8(buffer)
        .map_err(|err| Error::runtime(format!("runtime returned non-UTF-8 diagnostics: {err}")))
}

/// Replaces malformed sequences with U+FFFD instead of failing. Only for
/// model-generated text, never for runtime diagnostics: there, invalid UTF-8
/// means a real contract violation worth surfacing.
fn string_from_runtime_lossy(buffer: Vec<u8>) -> String {
    match String::from_utf8(buffer) {
        Ok(text) => text,
        Err(err) => String::from_utf8_lossy(err.as_bytes()).into_owned(),
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let value = path
        .to_str()
        .ok_or_else(|| Error::invalid(format!("path is not valid UTF-8: {}", path.display())))?;
    CString::new(value).map_err(nul_error)
}

fn nul_error(err: NulError) -> Error {
    Error::invalid(format!("string contains an interior NUL byte: {err}"))
}

fn runtime_error() -> Error {
    Error::runtime(ffi::last_error())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossy_decoding_survives_a_completion_cut_mid_emoji() {
        // A rollout stopped at max_tokens can leave the leading bytes of a
        // multi-byte character behind; that must not fail the update.
        let mut bytes = "ok ".as_bytes().to_vec();
        bytes.extend_from_slice(&"🎉".as_bytes()[..2]);
        let text = string_from_runtime_lossy(bytes);
        assert!(text.starts_with("ok "));
        assert!(text.contains('\u{fffd}'));
    }

    #[test]
    fn strict_decoding_still_rejects_non_utf8_diagnostics() {
        assert!(string_from_runtime(vec![0xff]).is_err());
    }

    #[test]
    fn interior_nul_paths_are_rejected_with_a_public_argument_error() {
        let error = nul_error(std::ffi::CString::new("adapter\0gguf").unwrap_err());
        assert!(matches!(error, Error::InvalidArgument(_)));
        assert!(error.to_string().contains("interior NUL"));
    }

    #[test]
    fn token_and_mask_guards_fail_before_the_runtime_is_needed() {
        let tokens = [1, 2, 3, 4];

        let mismatch = first_masked_target(&tokens, &[true, false, true]).unwrap_err();
        assert!(mismatch.to_string().contains("same length"));

        let empty = first_masked_target(&tokens, &[false; 4]).unwrap_err();
        assert!(empty.to_string().contains("selects no tokens"));

        let leading = first_masked_target(&tokens, &[true, false, false, false]).unwrap_err();
        assert!(leading.to_string().contains("cannot be selected"));

        let empty_prompt = validate_suffix(&tokens, 0).unwrap_err();
        assert!(empty_prompt.to_string().contains("prompt"));
        let no_completion = validate_suffix(&tokens, tokens.len()).unwrap_err();
        assert!(no_completion.to_string().contains("completion token"));

        let short_training = validate_train_tokens(&[1]).unwrap_err();
        assert!(short_training.to_string().contains("at least two tokens"));
    }

    #[test]
    fn train_config_as_ffi_maps_every_field() {
        let config = TrainConfig {
            n_ctx: 256,
            n_batch: 64,
            n_ubatch: 8,
            n_seq_max: 4,
            shared_prefix_fanout: retrograd_core::SharedPrefixFanout::Exact(3),
            generation_concurrency: 3,
            fast_generation_context: true,
            kv_dtype: KvDtype::F16,
            threads: 6,
            epochs: 3,
            learning_rate: 2.5e-4,
            weight_decay: 0.01,
            max_grad_norm: 0.75,
            lr_scheduler: LrScheduler::Cosine,
            warmup_steps: 12,
            verbose: true,
            device: Device::Gpu,
            chunked_cross_entropy: true,
            chunked_ce_tiles: 4,
            chunked_ce_seq_chunk: 512,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 2,
            checkpoint_dtype: CheckpointDtype::F16,
            master_weights: retrograd_core::MasterWeights::F32,
            require_gpu_resident: true,
            generation_batch: 256,
            shuffle_dataset: false,
            shuffle_seed: 1234,
            // Not part of the published layout, so it cannot appear in `ffi`
            // below; the setter carries it instead.
            max_gpu_duty_cycle: Some(0.5),
            // Only the two scalars cross: the selector is resolved against the
            // model's inventory and handed over by name, not by layout.
            trainable: retrograd_core::TrainableRunConfig {
                policy: retrograd_core::TrainablePolicy::Partial,
                selector: retrograd_core::TrainableSelector {
                    norms: true,
                    ..Default::default()
                },
                optimizer: OptimizerKind::Sgd,
            },
            optimizer_hyperparameters: OptimizerKind::Sgd.declared_hyperparameters(),
        };
        let ffi = train_config_to_ffi(&config).unwrap();
        assert_eq!(ffi.n_ctx, 256);
        assert_eq!(ffi.n_batch, 64);
        assert_eq!(ffi.n_ubatch, 8);
        assert_eq!(ffi.n_seq_max, 4);
        assert_eq!(ffi.generation_concurrency, 3);
        assert!(ffi.fast_generation_context);
        assert_eq!(ffi.kv_dtype, 1);
        assert_eq!(ffi.threads, 6);
        assert_eq!(ffi.epochs, 3);
        assert_eq!(ffi.learning_rate, 2.5e-4);
        assert_eq!(ffi.weight_decay, 0.01);
        assert_eq!(ffi.max_grad_norm, 0.75);
        assert_eq!(ffi.lr_scheduler, 2);
        assert_eq!(ffi.warmup_steps, 12);
        assert!(ffi.verbose);
        assert_eq!(ffi.device, 2);
        assert!(ffi.chunked_cross_entropy);
        assert_eq!(ffi.chunked_ce_tiles, 4);
        assert_eq!(ffi.chunked_ce_seq_chunk, 512);
        assert!(
            ffi.chunked_ce_offload_logsoftmax,
            "a token chunk is the only precondition, so it follows seq_chunk"
        );
        assert!(ffi.gradient_checkpointing);
        assert_eq!(ffi.checkpoint_every_n_layers, 2);
        assert_eq!(ffi.checkpoint_dtype, 1);
        assert_eq!(ffi.master_weights, 1);
        assert!(ffi.require_gpu_resident);
        assert_eq!(ffi.generation_batch, 256);
        assert!(!ffi.shuffle_dataset);
        assert_eq!(ffi.shuffle_seed, 1234);
        assert_eq!(ffi.optimizer, 1);
        assert_eq!(ffi.trainable, 2);
        // SGD declares none of the optimizer-specific rows, and a row nobody
        // declares crosses as zero rather than as another optimizer's default.
        assert_eq!(ffi.muon_momentum, 0.0);
        assert_eq!(ffi.muon_ns_steps, 0);
        assert_eq!(ffi.gefen_block_size, 0);
        assert_eq!(ffi.gefen_beta1, 0.0);

        // Muon's own rows do cross, and its fallback rate is its own value.
        let muon = TrainConfig {
            trainable: retrograd_core::TrainableRunConfig {
                optimizer: OptimizerKind::Muon,
                ..Default::default()
            },
            optimizer_hyperparameters: OptimizerKind::Muon.declared_hyperparameters(),
            ..TrainConfig::default()
        };
        let ffi = train_config_to_ffi(&muon).unwrap();
        assert_eq!(ffi.optimizer, 2);
        assert_eq!(ffi.muon_momentum, 0.95);
        assert_eq!(ffi.muon_ns_steps, 5);
        assert!(ffi.muon_nesterov);
        assert_eq!(ffi.muon_fallback_learning_rate, 1.0e-3);
        // AdamW declares `beta1` as well, and it is not Gefen's: the Gefen
        // fields stay zero unless the run is one.
        assert_eq!(ffi.gefen_beta1, 0.0);

        let gefen_kind = OptimizerKind::Gefen(retrograd_core::GefenLayout {
            variant: retrograd_core::GefenVariant::QuantizedM,
            ..Default::default()
        });
        let gefen = TrainConfig {
            trainable: retrograd_core::TrainableRunConfig {
                optimizer: gefen_kind,
                ..Default::default()
            },
            optimizer_hyperparameters: gefen_kind.declared_hyperparameters(),
            ..TrainConfig::default()
        };
        let ffi = train_config_to_ffi(&gefen).unwrap();
        assert_eq!(ffi.optimizer, 3);
        assert_eq!(ffi.gefen_variant, 1);
        assert_eq!(ffi.gefen_block_size, 1024);
        assert_eq!(ffi.gefen_beta1, 0.9);
        assert_eq!(ffi.gefen_beta2, 0.999);
        assert_eq!(ffi.muon_momentum, 0.0);
        assert!(
            TrainConfig::default().shuffle_dataset,
            "the shuffle is on unless a configuration turns it off"
        );

        let inherited = train_config_to_ffi(&TrainConfig {
            n_seq_max: 4,
            ..TrainConfig::default()
        })
        .unwrap();
        assert_eq!(inherited.generation_concurrency, 4);

        // The whole-token path is the one case where the in-place write has no
        // buffer to stay inside, so the derivation has to turn it back off.
        let unchunked = train_config_to_ffi(&TrainConfig {
            chunked_ce_seq_chunk: 0,
            ..TrainConfig::default()
        })
        .unwrap();
        assert!(!unchunked.chunked_ce_offload_logsoftmax);
    }

    #[test]
    fn train_metrics_from_ffi_copies_every_field() {
        let raw = ffi::RetroTrainMetrics {
            epoch: 4,
            epoch_complete: true,
            global_step: 9,
            train_loss: 1.5,
            eval_loss: 2.0,
            tokens_per_second: 123.0,
            learning_rate: 0.0001,
        };
        let metrics = train_metrics_from_ffi(raw);
        assert_eq!(metrics.epoch, 4);
        assert!(metrics.epoch_complete);
        assert_eq!(metrics.global_step, 9);
        assert_eq!(metrics.train_loss, 1.5);
        assert_eq!(metrics.eval_loss, 2.0);
        assert_eq!(metrics.tokens_per_second, 123.0);
        assert_eq!(metrics.learning_rate, 0.0001);
    }

    #[test]
    fn sampling_params_as_ffi_maps_every_field() {
        let ffi = sampling_params_to_ffi(&SamplingParams {
            temperature: 0.75,
            top_p: 0.9,
            max_new_tokens: 123,
            seed: 456,
        });
        assert_eq!(ffi.temperature, 0.75);
        assert_eq!(ffi.top_p, 0.9);
        assert_eq!(ffi.max_new_tokens, 123);
        assert_eq!(ffi.seed, 456);
    }

    #[test]
    fn vectorized_token_logprob_matches_scalar_reference() {
        for &n_vocab in &[1_usize, 31, 32, 257, 4096, 32768] {
            let logits = (0..n_vocab)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.017).sin() * 11.0 + (x * 0.0031).cos() * 3.0
                })
                .collect::<Vec<_>>();
            for token in [0, n_vocab / 2, n_vocab - 1] {
                let mut scalar = 0.0_f32;
                let mut vectorized = 0.0_f32;
                // SAFETY: the local helper owns a live allocation for every raw pointer passed to the runtime.
                let scalar_code = unsafe {
                    ffi::retro_probe_token_logprob(
                        logits.as_ptr(),
                        logits.len(),
                        token as i32,
                        false,
                        &mut scalar,
                    )
                };
                // SAFETY: the local helper owns a live allocation for every raw pointer passed to the runtime.
                let vectorized_code = unsafe {
                    ffi::retro_probe_token_logprob(
                        logits.as_ptr(),
                        logits.len(),
                        token as i32,
                        true,
                        &mut vectorized,
                    )
                };
                assert_eq!(scalar_code, 0, "{}", ffi::last_error());
                assert_eq!(vectorized_code, 0, "{}", ffi::last_error());
                assert!(
                    (vectorized - scalar).abs() <= 2.0e-6,
                    "n_vocab={n_vocab} token={token}: vectorized={vectorized} scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn weighted_batch_validation_rejects_empty_mismatched_and_overflowing_shapes() {
        let valid = WeightedBatch {
            tokens: vec![1, 2, 3, 4],
            labels: vec![-1, 2, -1, 4],
            weights: vec![0.0, 1.0, 0.0, -0.5],
            n_rows: 2,
            n_ctx: 2,
            n_topk: 1,
        };
        valid.validate().unwrap();

        let mut invalid = valid.clone();
        invalid.weights.pop();
        assert!(invalid.validate().is_err());

        let mut empty = valid.clone();
        empty.n_rows = 0;
        assert!(empty.validate().is_err());

        let overflowing = WeightedBatch {
            tokens: Vec::new(),
            labels: Vec::new(),
            weights: Vec::new(),
            n_rows: usize::MAX,
            n_ctx: 2,
            n_topk: 1,
        };
        assert!(
            overflowing
                .validate()
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );
    }

    fn packed_batch() -> PackedSequenceBatch {
        // Two sequences: 0 owns positions 0..=2, 1 owns positions 0..=1.
        PackedSequenceBatch {
            tokens: vec![1, 2, 3, 4],
            labels: vec![-1, 2, -1, 4],
            weights: vec![0.0, 1.0, 0.0, 1.0],
            positions: vec![0, 1, 2, 1],
            seq_offsets: vec![0, 2, 3, 4, 5],
            seq_ids: vec![0, 1, 0, 0, 1],
            n_sequences: 2,
            n_topk: 1,
        }
    }

    #[test]
    fn packed_batch_validation_accepts_shared_prefixes_and_rejects_bad_memberships() {
        packed_batch().validate().unwrap();

        let mut out_of_domain = packed_batch();
        out_of_domain.seq_ids[1] = 2;
        assert!(
            out_of_domain
                .validate()
                .unwrap_err()
                .to_string()
                .contains("outside 0..2")
        );

        let mut negative = packed_batch();
        negative.positions[3] = -1;
        assert!(
            negative
                .validate()
                .unwrap_err()
                .to_string()
                .contains("negative position")
        );
    }

    #[test]
    fn packed_batch_validation_names_a_hole_in_a_sequence() {
        // Sequence 0 keeps positions 0, 2 and 3: position 1 is missing, which
        // is exactly what llama.cpp's continuity check refuses.
        let mut holed = packed_batch();
        holed.positions = vec![0, 2, 3, 1];
        let message = holed.validate().unwrap_err().to_string();
        assert!(message.contains("sequence 0"), "{message}");
        assert!(message.contains("first hole at 1"), "{message}");
    }

    #[test]
    fn ffi_output_helpers_reject_lengths_larger_than_their_buffers() {
        // SAFETY: this is the over-report path, which both helpers reject
        // before touching the buffer, so neither closure has to write.
        let error = unsafe { ffi_out_vec::<u8>(1, |_| Ok(2)) }.unwrap_err();
        assert!(error.to_string().contains("2 output entries"), "{error}");

        // SAFETY: as above; `needed` is the helper's valid out-parameter.
        let error = unsafe {
            read_string_sized_lossy(1, |_buffer, _capacity, needed| {
                *needed = 2;
                0
            })
        }
        .unwrap_err();
        assert!(error.to_string().contains("2-byte string"), "{error}");
    }
}
