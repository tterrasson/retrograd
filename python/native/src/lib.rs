//! `retrograd._native`, the extension module behind the Python package.
//!
//! A thin PyO3 layer over the `retrograd` library and the agentic runner:
//! configuration crosses as keyword arguments and plain tuples, and every call
//! lands on the Rust `Trainer`, a training loop or `AgenticRun`. An error for
//! which `is_user_error()` holds becomes `ValueError`; anything else becomes
//! `RetrogradNativeError`.

use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use retrograd::config::{
    AdvantageBaseline, CriticConfig, DEFAULT_MAX_STALLED_UPDATES, DistillConfig, DistillMode,
    GefenToml, GrpoConfig, MuonToml, OptimizerToml, PpoConfig, PromptOrder, build_optimizer,
};
use retrograd::dataset::{self, DataFormat, PreparedDataset};
use retrograd::training::{self, Progress};
use retrograd::{
    CheckpointDtype, CheckpointMetadata, CheckpointProgress, DEFAULT_CE_SEQ_CHUNK,
    DEFAULT_CHECKPOINT_STRIDE, DatasetRecord, Device, Error, FeatureDtype, GrpoBatchParams,
    KvDtype, LayerRange, LoraConfig, LrScheduler, MasterWeights, OptimizerKind, RewardMode,
    RewardProtocol, SamplingParams, SharedPrefixFanout, TargetSet, TrainConfig, TrainMetrics,
    TrainSequence, TrainablePolicy, TrainableRunConfig, TrainableSelector, TrainableSet, Trainer,
    WeightedBatch, backend_list, checkpoint,
};
use retrograd_agent::tools::{McpServerConfig, McpToolProvider, ToolProvider};
use retrograd_agent::{
    AgentGrpoConfig, AgenticRun, EnvironmentConfig, Environments, Error as AgentError,
    JudgeBackend, JudgeConfig, JudgeFailurePolicy, RolloutLimits, Scenario, SharedToolsFactory,
    TruncationPolicy,
};

create_exception!(_native, RetrogradNativeError, PyRuntimeError);

type MetricsTuple = (u32, bool, u64, f32, f32, f32, f32);
type ProgressTuple = (MetricsTuple, Vec<(String, f32)>);
/// name, role, dtype, ggml dimensions, elements, stored bytes.
type TrainableEntryTuple = (String, String, String, Vec<i64>, u64, u64);
/// policy, resolved entries, eligible tensors left out with the reason.
type TrainableSetTuple = (String, Vec<TrainableEntryTuple>, Vec<(String, String)>);
/// global_step, epoch, cursor, seeds, adapter, trainable bundle, optimizer
/// graph present at save time, slots restored.
type ResumeTuple = (
    u64,
    u64,
    u64,
    Vec<(String, u64)>,
    Option<String>,
    Option<String>,
    bool,
    usize,
);

fn metrics_tuple(metrics: TrainMetrics) -> MetricsTuple {
    (
        metrics.epoch,
        metrics.epoch_complete,
        metrics.global_step,
        metrics.train_loss,
        metrics.eval_loss,
        metrics.tokens_per_second,
        metrics.learning_rate,
    )
}

fn progress_tuple(progress: Progress) -> ProgressTuple {
    (
        metrics_tuple(progress.metrics),
        progress
            .values
            .into_iter()
            .map(|value| (value.name.into_owned(), value.value))
            .collect(),
    )
}

/// Map runtime errors to Python's exception hierarchy: user errors become
/// `ValueError`, I/O errors become `OSError`, and other failures keep the
/// module-specific exception.
fn python_error(error: Error) -> PyErr {
    match error {
        Error::Io(error) => PyOSError::new_err(error.to_string()),
        error if error.is_user_error() => PyValueError::new_err(error.to_string()),
        error => RetrogradNativeError::new_err(error.to_string()),
    }
}

fn agent_python_error(error: AgentError) -> PyErr {
    match error {
        AgentError::Core(error) => python_error(error),
        AgentError::Invalid(message) => PyValueError::new_err(message),
        other => RetrogradNativeError::new_err(other.to_string()),
    }
}

fn parse_device(value: &str) -> PyResult<Device> {
    value.parse().map_err(python_error)
}

fn parse_kv_dtype(value: &str) -> PyResult<KvDtype> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(KvDtype::F32),
        "f16" => Ok(KvDtype::F16),
        _ => Err(PyValueError::new_err(format!(
            "unknown kv_dtype '{value}'; use f32 or f16"
        ))),
    }
}

/// Precision the gradient-checkpoint activations are held in. `f32` inserts no
/// casts and keeps the recompute bit-exact; the 16-bit options trade real gradient
/// fidelity for memory (see `CheckpointDtype`).
fn parse_checkpoint_dtype(value: &str) -> PyResult<CheckpointDtype> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(CheckpointDtype::F32),
        "f16" => Ok(CheckpointDtype::F16),
        "bf16" => Ok(CheckpointDtype::Bf16),
        _ => Err(PyValueError::new_err(format!(
            "unknown checkpoint_dtype '{value}'; use f32, f16, or bf16"
        ))),
    }
}

/// Whether half-precision base weights are trained through an F32 master copy.
/// `auto` keeps one exactly when a marked base tensor is half precision, which
/// is the only case it changes anything (see `MasterWeights`).
fn parse_master_weights(value: &str) -> PyResult<MasterWeights> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(MasterWeights::Auto),
        "f32" => Ok(MasterWeights::F32),
        "off" => Ok(MasterWeights::Off),
        _ => Err(PyValueError::new_err(format!(
            "unknown master_weights '{value}'; use auto, f32, or off"
        ))),
    }
}

/// Precision the PPO critic's host-side feature matrix is stored in. `f32` keeps
/// the hidden states as the runtime produced them; the 16-bit options halve the
/// run's largest host allocation and round the rows the value head regresses on
/// (see `FeatureDtype`).
fn parse_feature_dtype(value: &str) -> PyResult<FeatureDtype> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(FeatureDtype::F32),
        "f16" => Ok(FeatureDtype::F16),
        "bf16" => Ok(FeatureDtype::Bf16),
        _ => Err(PyValueError::new_err(format!(
            "unknown feature_dtype '{value}'; use f32, f16, or bf16"
        ))),
    }
}

/// How the reward command is spoken to. Same two spellings as the TOML
/// `reward_mode`, and the same default: one persistent worker for the loop.
fn parse_reward_protocol(mode: &str, timeout_seconds: u64) -> PyResult<RewardProtocol> {
    let mode = match mode.trim().to_ascii_lowercase().as_str() {
        "persistent" => RewardMode::Persistent,
        "oneshot" | "one_shot" => RewardMode::OneShot,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown reward_mode '{other}'; use persistent or oneshot"
            )));
        }
    };
    if timeout_seconds == 0 {
        return Err(PyValueError::new_err(
            "reward_timeout_seconds must be greater than zero",
        ));
    }
    Ok(RewardProtocol {
        mode,
        timeout: std::time::Duration::from_secs(timeout_seconds),
    })
}

fn parse_scheduler(value: &str) -> PyResult<LrScheduler> {
    match value.trim().to_ascii_lowercase().as_str() {
        "constant" => Ok(LrScheduler::Constant),
        "linear" => Ok(LrScheduler::Linear),
        "cosine" => Ok(LrScheduler::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "unknown scheduler '{value}'; use constant, linear, or cosine"
        ))),
    }
}

/// The optimizer the run's update step builds. Unimplemented names are refused
/// by their own name rather than substituted: a run that asked for one
/// optimizer and got another would publish a trajectory nobody configured, and
/// a checkpoint recording an optimizer it never used.
fn parse_optimizer(value: &str) -> PyResult<OptimizerKind> {
    let kind = OptimizerKind::parse(value).map_err(python_error)?;
    if !kind.is_implemented() {
        return Err(PyValueError::new_err(format!(
            "optimizer '{kind}' is not available in this build"
        )));
    }
    Ok(kind)
}

/// One `[optimizer.<name>]` section, as a dict of TOML keys. Unknown keys
/// are refused rather than ignored.
struct OptimizerSection<'py> {
    name: &'static str,
    dict: Bound<'py, pyo3::types::PyDict>,
}

impl<'py> OptimizerSection<'py> {
    fn new(
        name: &'static str,
        dict: Bound<'py, pyo3::types::PyDict>,
        known: &[&str],
    ) -> PyResult<Self> {
        for key in dict.keys() {
            let key: String = key.extract()?;
            if !known.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown {name} option '{key}'; known options are {}",
                    known.join(", ")
                )));
            }
        }
        Ok(Self { name, dict })
    }

    /// `None` and a missing key are the same: the row keeps the declared
    /// default.
    fn item(&self, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.dict.get_item(key)?.filter(|value| !value.is_none()))
    }

    fn wrong_type(&self, key: &str, expected: &str) -> PyErr {
        PyValueError::new_err(format!("{}.{key} must be {expected}", self.name))
    }

    fn get_f32(&self, key: &str) -> PyResult<Option<f32>> {
        match self.item(key)? {
            None => Ok(None),
            Some(value) => value
                .extract::<f32>()
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a number")),
        }
    }

    fn get_bool(&self, key: &str) -> PyResult<Option<bool>> {
        match self.item(key)? {
            None => Ok(None),
            Some(value) => value
                .extract::<bool>()
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a bool")),
        }
    }

    fn get_u32(&self, key: &str) -> PyResult<Option<u32>> {
        match self.item(key)? {
            None => Ok(None),
            Some(value) => value
                .extract::<u32>()
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a non-negative integer")),
        }
    }

    fn get_u64(&self, key: &str) -> PyResult<Option<u64>> {
        match self.item(key)? {
            None => Ok(None),
            Some(value) => value
                .extract::<u64>()
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a non-negative integer")),
        }
    }

    fn get_string(&self, key: &str) -> PyResult<Option<String>> {
        match self.item(key)? {
            None => Ok(None),
            Some(value) => value
                .extract::<String>()
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a string")),
        }
    }
}

const MUON_OPTIONS: [&str; 5] = [
    "momentum",
    "nesterov",
    "ns_steps",
    "ns_epsilon",
    "fallback_learning_rate",
];

const GEFEN_OPTIONS: [&str; 9] = [
    "variant",
    "block_size",
    "min_numel",
    "codebook",
    "codebook_levels",
    "partition",
    "beta1",
    "beta2",
    "eps",
];

/// The chosen optimizer with its layout resolved and the vector its update
/// reads, built by the same function the TOML frontend calls so both sides
/// share the same refusals.
fn resolve_optimizer(
    optimizer: &str,
    muon: Option<&Bound<'_, pyo3::types::PyDict>>,
    gefen: Option<&Bound<'_, pyo3::types::PyDict>>,
    learning_rate: f32,
    weight_decay: f32,
    max_grad_norm: f32,
) -> PyResult<(OptimizerKind, retrograd::HyperparameterVector)> {
    let chosen = parse_optimizer(optimizer)?;
    let muon = match muon {
        None => None,
        Some(dict) => {
            let section = OptimizerSection::new("muon", dict.clone(), &MUON_OPTIONS)?;
            Some(MuonToml {
                momentum: section.get_f32("momentum")?,
                nesterov: section.get_bool("nesterov")?,
                ns_steps: section.get_u32("ns_steps")?,
                ns_epsilon: section.get_f32("ns_epsilon")?,
                fallback_learning_rate: section.get_f32("fallback_learning_rate")?,
            })
        }
    };
    let gefen = match gefen {
        None => None,
        Some(dict) => {
            let section = OptimizerSection::new("gefen", dict.clone(), &GEFEN_OPTIONS)?;
            Some(GefenToml {
                variant: section.get_string("variant")?,
                block_size: section.get_u64("block_size")?,
                min_numel: section.get_u64("min_numel")?,
                codebook: section.get_string("codebook")?,
                codebook_levels: section.get_u64("codebook_levels")?,
                partition: section.get_string("partition")?,
                beta1: section.get_f32("beta1")?,
                beta2: section.get_f32("beta2")?,
                eps: section.get_f32("eps")?,
            })
        }
    };
    let section = OptimizerToml { muon, gefen };
    let resolved = build_optimizer(
        Some(&section),
        chosen,
        learning_rate,
        weight_decay,
        max_grad_norm,
    )
    .map_err(python_error)?;
    Ok((resolved.kind, resolved.hyperparameters))
}

/// Fingerprint of everything in a Python run that can change the continued
/// trajectory; a resume compares it. It is its own namespace (not the
/// document frontend's `trajectory-v1`, which pins the algorithm section that
/// has no Python equivalent), and the caller's own algorithm settings are
/// appended as an opaque string.
fn python_trajectory_signature(config: &TrainConfig, algorithm: &str, extra: &str) -> String {
    let descriptor = format!(
        "trajectory-py-v1|n_ctx={}|n_batch={}|n_ubatch={}|n_seq_max={}|\
         generation_concurrency={}|fast_generation={}|kv_dtype={:?}|\
         gradient_checkpointing={}|checkpoint_every_n_layers={}|checkpoint_dtype={:?}|\
         master_weights={}|threads={}|epochs={}|lr={:08x}|wd={:08x}|max_grad_norm={:08x}|scheduler={}|\
         warmup={}|device={:?}|algorithm={algorithm}|extra={extra}",
        config.n_ctx,
        config.n_batch,
        config.n_ubatch,
        config.n_seq_max,
        config.generation_concurrency,
        config.fast_generation_context,
        config.kv_dtype,
        config.gradient_checkpointing,
        config.checkpoint_every_n_layers,
        config.checkpoint_dtype,
        config.master_weights,
        config.threads,
        config.epochs,
        config.learning_rate.to_bits(),
        config.weight_decay.to_bits(),
        config.max_grad_norm.to_bits(),
        match config.lr_scheduler {
            LrScheduler::Constant => "constant",
            LrScheduler::Linear => "linear",
            LrScheduler::Cosine => "cosine",
        },
        config.warmup_steps,
        config.device,
    );
    checkpoint::fingerprint(descriptor.as_bytes())
}

/// The dataset identity a checkpoint records: over the prepared rows,
/// so a different tokenization or label mask is a different dataset.
fn dataset_record(dataset: &PreparedDataset, path: &str) -> DatasetRecord {
    DatasetRecord {
        version: checkpoint::FORMAT_VERSION,
        path: path.to_string(),
        fingerprint: retrograd::run::dataset_fingerprint(&dataset.tokens, &dataset.labels),
        examples: dataset.examples as u64,
        row_width: dataset.n_ctx as u64,
        format: "prepared".into(),
        permutation: Vec::new(),
        cursor: 0,
    }
}

/// Parses the base-weight policy and its selection, refusing a selection the
/// policy would ignore rather than silently dropping it.
fn parse_trainable(
    policy: &str,
    layers: &str,
    modules: Vec<String>,
    norms: bool,
    biases: bool,
    output_head: bool,
) -> PyResult<(TrainablePolicy, TrainableSelector)> {
    let policy = TrainablePolicy::parse(policy).map_err(python_error)?;
    let selector = TrainableSelector {
        layers: LayerRange::parse(layers).map_err(python_error)?,
        modules,
        norms,
        biases,
        output_head,
    };
    let selects = !selector.is_empty() || selector.layers != LayerRange::All;
    match policy {
        TrainablePolicy::Lora if selects => Err(PyValueError::new_err(
            "trainable_policy = 'lora' trains no base tensor: drop the trainable_* \
             selectors, or ask for 'partial' or 'hybrid'",
        )),
        TrainablePolicy::Full if selects => Err(PyValueError::new_err(
            "trainable_policy = 'full' trains every supported eligible tensor and takes \
             no selectors: use 'partial' to narrow it",
        )),
        TrainablePolicy::Lora | TrainablePolicy::Full => Ok((policy, TrainableSelector::default())),
        TrainablePolicy::Partial | TrainablePolicy::Hybrid if selector.is_empty() => {
            Err(PyValueError::new_err(format!(
                "trainable_policy = '{policy}' selects nothing: set at least one of \
                 trainable_modules, trainable_norms, trainable_biases or \
                 trainable_output_head"
            )))
        }
        TrainablePolicy::Hybrid if !selector.modules.is_empty() || selector.output_head => {
            Err(PyValueError::new_err(
                "trainable_policy = 'hybrid' permits only base norms and biases beside the \
                 adapter: trainable_modules and trainable_output_head are not supported \
                 with it",
            ))
        }
        TrainablePolicy::Partial | TrainablePolicy::Hybrid => Ok((policy, selector)),
    }
}

/// Whether a standalone model GGUF may be written for `architecture`.
///
/// `None` is the LoRA case: no inventory was read and the runtime refuses the
/// export for its own reason.
fn check_model_export(architecture: Option<&str>) -> std::result::Result<(), String> {
    let Some(architecture) = architecture else {
        return Ok(());
    };
    if retrograd::architecture_exports_model(architecture) {
        return Ok(());
    }
    let known: Vec<&str> = retrograd::model_export_architectures().collect();
    Err(format!(
        "a standalone model export is not available for architecture '{architecture}': \
         it is covered for [{}] today. Save the trainable bundle instead",
        known.join(", ")
    ))
}

fn parse_format(value: &str) -> PyResult<DataFormat> {
    match value {
        "text" => Ok(DataFormat::Text),
        "chat_jsonl" => Ok(DataFormat::ChatJsonl),
        _ => Err(PyValueError::new_err(format!(
            "unknown dataset format '{value}'; use text or chat_jsonl"
        ))),
    }
}

#[pyclass(name = "_PreparedDataset", frozen)]
struct PyPreparedDataset {
    inner: PreparedDataset,
}

#[pymethods]
impl PyPreparedDataset {
    #[getter]
    fn context_size(&self) -> usize {
        self.inner.n_ctx
    }

    #[getter]
    fn examples(&self) -> usize {
        self.inner.examples
    }

    #[getter]
    fn supervised_tokens(&self) -> usize {
        self.inner.supervised_tokens
    }

    fn __len__(&self) -> usize {
        self.inner.examples
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedDataset(examples={}, context_size={}, supervised_tokens={})",
            self.inner.examples, self.inner.n_ctx, self.inner.supervised_tokens
        )
    }
}

#[pyclass(name = "_Trainer", unsendable)]
struct PyTrainer {
    inner: Option<Trainer>,
    /// The file the trainer was opened on; kept because a resume compares the
    /// model's fingerprint and size.
    model_path: std::path::PathBuf,
    training: TrainConfig,
    /// Resolved base set, kept for the `trainable_set` getter. `None` for a LoRA run.
    trainable: Option<TrainableSet>,
    /// Model architecture, read from the same inventory. Used to check standalone export coverage.
    architecture: Option<String>,
}

impl PyTrainer {
    fn trainer(&self) -> PyResult<&Trainer> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("trainer is closed"))
    }

    fn trainer_mut(&mut self) -> PyResult<&mut Trainer> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("trainer is closed"))
    }
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (
        model_path,
        *,
        n_ctx=128,
        n_batch=128,
        n_ubatch=32,
        n_seq_max=1,
        generation_concurrency=0,
        generation_batch=0,
        fast_generation_context=false,
        kv_dtype="f16",
        threads=0,
        epochs=1,
        learning_rate=1.0e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        scheduler="constant",
        warmup_steps=0,
        optimizer="adamw",
        muon=None,
        gefen=None,
        trainable_policy="lora",
        trainable_layers="all",
        trainable_modules=None,
        trainable_norms=false,
        trainable_biases=false,
        trainable_output_head=false,
        chunked_cross_entropy=true,
        chunked_ce_tiles=8,
        chunked_ce_seq_chunk=DEFAULT_CE_SEQ_CHUNK,
        gradient_checkpointing=false,
        checkpoint_every_n_layers=DEFAULT_CHECKPOINT_STRIDE,
        checkpoint_dtype="f32",
        master_weights="auto",
        require_gpu_resident=false,
        shuffle=true,
        shuffle_seed=42,
        max_gpu_duty_cycle=1.0,
        verbose=false,
        device="auto"
    ))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        model_path: String,
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
        n_seq_max: u32,
        generation_concurrency: u32,
        generation_batch: u32,
        fast_generation_context: bool,
        kv_dtype: &str,
        threads: u32,
        epochs: u32,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        scheduler: &str,
        warmup_steps: u64,
        optimizer: &str,
        muon: Option<&Bound<'_, pyo3::types::PyDict>>,
        gefen: Option<&Bound<'_, pyo3::types::PyDict>>,
        trainable_policy: &str,
        trainable_layers: &str,
        trainable_modules: Option<Vec<String>>,
        trainable_norms: bool,
        trainable_biases: bool,
        trainable_output_head: bool,
        chunked_cross_entropy: bool,
        chunked_ce_tiles: u32,
        chunked_ce_seq_chunk: u32,
        gradient_checkpointing: bool,
        checkpoint_every_n_layers: u32,
        checkpoint_dtype: &str,
        master_weights: &str,
        require_gpu_resident: bool,
        shuffle: bool,
        shuffle_seed: u64,
        max_gpu_duty_cycle: f32,
        verbose: bool,
        device: &str,
    ) -> PyResult<Self> {
        if chunked_ce_tiles == 0 {
            return Err(PyValueError::new_err(
                "chunked_ce_tiles must be greater than zero",
            ));
        }
        if checkpoint_every_n_layers == 0 {
            return Err(PyValueError::new_err(
                "checkpoint_every_n_layers must be greater than zero",
            ));
        }
        // A narrower checkpoint type is invalid when no checkpoints are retained.
        if checkpoint_dtype != "f32" && !gradient_checkpointing {
            return Err(PyValueError::new_err(
                "checkpoint_dtype only applies to retained activation checkpoints; \
                 pass gradient_checkpointing=True or leave it at 'f32'",
            ));
        }
        if !max_gpu_duty_cycle.is_finite() || max_gpu_duty_cycle <= 0.0 || max_gpu_duty_cycle > 1.0
        {
            return Err(PyValueError::new_err(
                "max_gpu_duty_cycle must be finite and in (0, 1]",
            ));
        }
        let (policy, selector) = parse_trainable(
            trainable_policy,
            trainable_layers,
            trainable_modules.unwrap_or_default(),
            trainable_norms,
            trainable_biases,
            trainable_output_head,
        )?;
        let (chosen_optimizer, optimizer_hyperparameters) = resolve_optimizer(
            optimizer,
            muon,
            gefen,
            learning_rate,
            weight_decay,
            max_grad_norm,
        )?;
        let config = TrainConfig {
            n_ctx,
            n_batch,
            n_ubatch,
            n_seq_max,
            shared_prefix_fanout: SharedPrefixFanout::Auto,
            generation_concurrency,
            generation_batch,
            fast_generation_context,
            kv_dtype: parse_kv_dtype(kv_dtype)?,
            threads,
            epochs,
            learning_rate,
            weight_decay,
            max_grad_norm,
            lr_scheduler: parse_scheduler(scheduler)?,
            warmup_steps,
            trainable: TrainableRunConfig {
                policy,
                selector,
                optimizer: chosen_optimizer,
            },
            // Built by `resolve_optimizer`: the declared rows, with the
            // `muon=`/`gefen=` overrides.
            optimizer_hyperparameters,
            chunked_cross_entropy,
            chunked_ce_tiles,
            chunked_ce_seq_chunk,
            gradient_checkpointing,
            checkpoint_every_n_layers,
            checkpoint_dtype: parse_checkpoint_dtype(checkpoint_dtype)?,
            master_weights: parse_master_weights(master_weights)?,
            require_gpu_resident,
            shuffle_dataset: shuffle,
            shuffle_seed,
            // `1.0` is how C spells "no limit"; the Rust side keeps the
            // disabled state explicit so it never crosses the FFI boundary.
            max_gpu_duty_cycle: (max_gpu_duty_cycle < 1.0).then_some(max_gpu_duty_cycle),
            verbose,
            device: parse_device(device)?,
        };
        // Validate the same batch and context geometry enforced by the TOML API.
        config.validate_geometry().map_err(python_error)?;
        // Resolve which base tensors this run trains against the model's tensor
        // table and declare the set before any graph is built.
        let resolved = if policy.trains_base_weights() {
            let inventory =
                retrograd::tensor_inventory(&model_path, config.device).map_err(python_error)?;
            let set = retrograd::resolve_base(&inventory, policy, &config.trainable.selector)
                .map_err(python_error)?;
            Some((inventory.architecture, set))
        } else {
            None
        };
        let mut inner = Trainer::new(&model_path, config.clone()).map_err(python_error)?;
        let (architecture, trainable) = match resolved {
            Some((architecture, set)) => {
                inner.declare_trainable_set(&set).map_err(python_error)?;
                (Some(architecture), Some(set))
            }
            None => (None, None),
        };
        Ok(Self {
            inner: Some(inner),
            model_path: std::path::PathBuf::from(model_path),
            training: config,
            trainable,
            architecture,
        })
    }

    #[pyo3(signature = (*, rank=8, alpha=16.0, dropout=0.0, seed=42, targets=None, dtype="f32"))]
    fn create_lora(
        &mut self,
        rank: u32,
        alpha: f32,
        dropout: f32,
        seed: u32,
        targets: Option<Vec<String>>,
        dtype: &str,
    ) -> PyResult<()> {
        let targets = match targets {
            None => TargetSet::Auto,
            Some(patterns) => TargetSet::Patterns(patterns),
        };
        self.trainer_mut()?
            .create_lora(&LoraConfig {
                rank,
                alpha,
                dropout,
                seed,
                targets,
                dtype: match dtype.to_ascii_lowercase().as_str() {
                    "f32" => retrograd::LoraDtype::F32,
                    "f16" => retrograd::LoraDtype::F16,
                    _ => return Err(PyValueError::new_err("dtype must be f32 or f16")),
                },
            })
            .map_err(python_error)
    }

    fn load_lora(&mut self, adapter_path: String) -> PyResult<()> {
        self.trainer_mut()?
            .load_lora(adapter_path)
            .map_err(python_error)
    }

    fn save_lora(&mut self, adapter_path: String) -> PyResult<()> {
        self.trainer_mut()?
            .save_lora(adapter_path)
            .map_err(python_error)
    }

    /// What the base-weight policy resolved to, or `None` for a LoRA run.
    #[getter]
    fn trainable_set(&self) -> Option<TrainableSetTuple> {
        let set = self.trainable.as_ref()?;
        let entries = set
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.name.clone(),
                    entry.role.as_str().to_string(),
                    entry.dtype.name().to_string(),
                    entry.ne.to_vec(),
                    entry.n_elements,
                    entry.n_bytes,
                )
            })
            .collect();
        let exclusions = set
            .exclusions
            .iter()
            .map(|exclusion| (exclusion.name.clone(), exclusion.reason.to_string()))
            .collect();
        Some((set.policy.to_string(), entries, exclusions))
    }

    /// Writes the trained base tensors, by absolute value, as a GGUF bundle.
    fn save_trainable(&mut self, bundle_path: String) -> PyResult<()> {
        self.trainer_mut()?
            .save_trainable(bundle_path)
            .map_err(python_error)
    }

    /// Restores base tensor values from such a bundle onto the live model.
    fn load_trainable(&mut self, bundle_path: String) -> PyResult<()> {
        self.trainer_mut()?
            .load_trainable(bundle_path)
            .map_err(python_error)
    }

    /// Writes the whole model out as a standalone GGUF, trained weights included.
    fn save_model(&mut self, model_path: String) -> PyResult<()> {
        // Check export coverage before the file is opened rather than half written.
        check_model_export(self.architecture.as_deref()).map_err(PyValueError::new_err)?;
        self.trainer_mut()?
            .save_model(model_path)
            .map_err(python_error)
    }

    /// Loads the frozen model this run's reference term scores against.
    /// `n_ctx` overrides the training width for the anchor alone.
    #[pyo3(signature = (reference_path, *, n_ctx=None))]
    fn attach_reference(&mut self, reference_path: String, n_ctx: Option<u32>) -> PyResult<()> {
        let training = self.training.clone();
        self.trainer_mut()?
            .attach_reference(reference_path, &training, n_ctx)
            .map_err(python_error)
    }

    #[getter]
    fn reference_path(&self) -> PyResult<Option<String>> {
        Ok(self
            .trainer()?
            .reference_path()
            .map(|path| path.display().to_string()))
    }

    /// Adapter, trainable bundle and optimizer state size of one checkpoint.
    fn checkpoint_footprint(&mut self) -> PyResult<(u64, u64, u64)> {
        let footprint = self
            .trainer_mut()?
            .checkpoint_footprint()
            .map_err(python_error)?;
        Ok((
            footprint.adapter_bytes,
            footprint.trainable_bytes,
            footprint.optimizer_state_bytes,
        ))
    }

    /// Writes a complete training checkpoint into `directory`.
    ///
    /// The caller supplies the dataset, the stopping point and the algorithm;
    /// everything else is read from the live runtime.
    #[pyo3(signature = (
        directory,
        dataset,
        *,
        checkpoint_id,
        algorithm,
        global_step,
        epoch,
        cursor=0,
        dataset_path="",
        trajectory_extra="",
        phase="train",
        resume_boundary="epoch",
        seeds=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn save_checkpoint(
        &mut self,
        directory: String,
        dataset: &PyPreparedDataset,
        checkpoint_id: &str,
        algorithm: &str,
        global_step: u64,
        epoch: u64,
        cursor: u64,
        dataset_path: &str,
        trajectory_extra: &str,
        phase: &str,
        resume_boundary: &str,
        seeds: Option<std::collections::BTreeMap<String, u64>>,
    ) -> PyResult<()> {
        let metadata = CheckpointMetadata {
            checkpoint_id: checkpoint_id.to_string(),
            algorithm: algorithm.to_string(),
            trajectory_signature: python_trajectory_signature(
                &self.training,
                algorithm,
                trajectory_extra,
            ),
            resume_boundary: resume_boundary.to_string(),
            scheduler_kind: retrograd::run::scheduler_name(self.training.lr_scheduler).to_string(),
            warmup_steps: self.training.warmup_steps,
            progress: CheckpointProgress {
                version: checkpoint::FORMAT_VERSION,
                epoch,
                global_step,
                cursor,
                algorithm: algorithm.to_string(),
                phase: phase.to_string(),
                best_eval: None,
                stale_evaluations: 0,
                kl_multiplier: None,
            },
            dataset: dataset_record(&dataset.inner, dataset_path),
            seeds: seeds.unwrap_or_default(),
            // The Python driver writes its own files beside the checkpoint.
            artifacts: std::collections::BTreeMap::new(),
            model_path: self.model_path.clone(),
        };
        self.trainer_mut()?
            .save_checkpoint(&directory, &metadata)
            .map_err(python_error)
    }

    /// Restores a checkpoint from `directory`, refusing one that does not
    /// match this run.
    ///
    /// Returns `(global_step, epoch, cursor, seeds, adapter, trainable,
    /// had_optimizer_graph, restored_optimizer_slots)`.
    #[pyo3(signature = (
        directory,
        dataset,
        *,
        algorithm,
        dataset_path="",
        trajectory_extra="",
        total_steps=None
    ))]
    fn load_checkpoint(
        &mut self,
        directory: String,
        dataset: &PyPreparedDataset,
        algorithm: &str,
        dataset_path: &str,
        trajectory_extra: &str,
        total_steps: Option<u64>,
    ) -> PyResult<ResumeTuple> {
        let training = self.training.clone();
        let model_path = self.model_path.clone();
        let record = dataset_record(&dataset.inner, dataset_path);
        let trajectory_signature =
            python_trajectory_signature(&training, algorithm, trajectory_extra);
        let scheduler_kind = retrograd::run::scheduler_name(training.lr_scheduler).to_string();
        let model_fingerprint =
            checkpoint::fingerprint_file_cached(&model_path).map_err(python_error)?;
        let trainer = self.trainer_mut()?;
        let hyperparameters = trainer.optimizer_hyperparameters().map_err(python_error)?;
        // Every field is read from the live runtime, not from caller
        // arguments.
        let expected = checkpoint::Compatibility {
            model_signature: trainer.model_signature().map_err(python_error)?,
            model_bytes: retrograd::run::model_bytes(&model_path),
            model_fingerprint,
            reference_fingerprint: trainer.reference_fingerprint().map_err(python_error)?,
            algorithm: algorithm.to_string(),
            trajectory_signature,
            dataset_fingerprint: record.fingerprint.clone(),
            scheduler_kind,
            learning_rate: training.learning_rate,
            warmup_steps: training.warmup_steps,
            total_steps,
            optimizer_kind: hyperparameters.optimizer().to_string(),
            optimizer_layout_version: hyperparameters.optimizer().layout_version(),
            optimizer_hyperparameters: hyperparameters.lines(),
            weight_decay: training.weight_decay,
            max_grad_norm: training.max_grad_norm,
            trainable_policy: trainer.trainable_policy().as_str().to_string(),
            trainable_signature: trainer.trainable_signature().map_err(python_error)?,
        };
        let info = trainer
            .load_checkpoint(&directory, &expected)
            .map_err(python_error)?;
        Ok((
            info.progress.global_step,
            info.progress.epoch,
            info.progress.cursor,
            info.seeds.into_iter().collect(),
            info.adapter.map(|path| path.display().to_string()),
            info.trainable.map(|path| path.display().to_string()),
            info.had_optimizer_graph,
            info.restored_optimizer_slots,
        ))
    }

    fn prepare_dataset(
        &self,
        path: String,
        format: &str,
        context_size: usize,
    ) -> PyResult<PyPreparedDataset> {
        let inner = dataset::prepare(self.trainer()?, path, parse_format(format)?, context_size)
            .map_err(python_error)?;
        Ok(PyPreparedDataset { inner })
    }

    #[pyo3(signature = (train, eval=None, callback=None))]
    fn fit_sft(
        &mut self,
        train: &PyPreparedDataset,
        eval: Option<&PyPreparedDataset>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        let mut callback_error = None;
        let result = self.trainer_mut()?.train_sft_with_progress(
            &train.inner,
            eval.map(|value| &value.inner),
            |metrics| {
                if callback_error.is_some() {
                    return;
                }
                if let Some(callback) = callback.as_ref() {
                    Python::attach(|py| {
                        if let Err(error) = callback.call1(py, (metrics_tuple(metrics),)) {
                            callback_error = Some(error);
                        }
                    });
                }
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map(metrics_tuple).map_err(python_error)
    }

    fn train_tokens(&mut self, tokens: Vec<i32>) -> PyResult<MetricsTuple> {
        self.trainer_mut()?
            .train_tokens(&tokens)
            .map(metrics_tuple)
            .map_err(python_error)
    }

    fn train_weighted(
        &mut self,
        tokens: Vec<i32>,
        labels: Vec<i32>,
        weights: Vec<f32>,
        n_rows: usize,
        n_ctx: usize,
        scheduler_total_steps: u64,
    ) -> PyResult<MetricsTuple> {
        self.trainer_mut()?
            .train_weighted(
                &WeightedBatch {
                    tokens,
                    labels,
                    weights,
                    n_rows,
                    n_ctx,
                    // One target per position: the Python weighted step is the
                    // policy-gradient objective, not offline top-k KD. That one
                    // has no binding of its own - it reads a corpus and a
                    // sidecar, so it arrives through a `[distill]
                    // mode = "topk_offline"` document.
                    n_topk: 1,
                },
                scheduler_total_steps,
            )
            .map(metrics_tuple)
            .map_err(python_error)
    }

    #[pyo3(signature = (
        token_ids,
        old_logprobs,
        train_masks,
        rewards,
        group_ids,
        *,
        intermediate_returns=None,
        epochs=4,
        clip_range_low=0.2,
        clip_range_high=0.28,
        kl_coefficient=0.0,
        loss_denominator,
        seed=42,
        scheduler_total_rollouts=None,
        callback=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn train_grpo_batch(
        &mut self,
        token_ids: Vec<Vec<i32>>,
        old_logprobs: Vec<Vec<f32>>,
        train_masks: Vec<Vec<bool>>,
        rewards: Vec<f32>,
        group_ids: Vec<u64>,
        intermediate_returns: Option<Vec<Vec<f32>>>,
        epochs: u32,
        clip_range_low: f32,
        clip_range_high: f32,
        kl_coefficient: f32,
        loss_denominator: usize,
        seed: u64,
        scheduler_total_rollouts: Option<u64>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        // Rollout training uses the full context for one optimizer step; SFT and
        // generation-only trainers do not have this restriction.
        self.training
            .validate_rollout_geometry()
            .map_err(python_error)?;
        let rows = token_ids.len();
        let intermediate_returns = intermediate_returns.unwrap_or_else(|| vec![Vec::new(); rows]);
        if rows == 0
            || old_logprobs.len() != rows
            || train_masks.len() != rows
            || rewards.len() != rows
            || group_ids.len() != rows
            || intermediate_returns.len() != rows
        {
            return Err(PyValueError::new_err(
                "token_ids, old_logprobs, train_masks, rewards, and group_ids must have the same non-zero length",
            ));
        }
        let sequences = token_ids
            .into_iter()
            .zip(old_logprobs)
            .zip(train_masks)
            .zip(rewards)
            .zip(group_ids)
            .zip(intermediate_returns)
            .map(
                |(
                    ((((tokens, old_logprobs), train_mask), reward), group_id),
                    intermediate_returns,
                )| TrainSequence {
                    tokens,
                    old_logprobs,
                    train_mask,
                    reward,
                    group_id,
                    intermediate_returns,
                },
            )
            .collect::<Vec<_>>();
        let params = GrpoBatchParams {
            epochs,
            clip_range_low,
            clip_range_high,
            kl_coefficient,
            loss_denominator,
            seed,
            scheduler_total_rollouts,
        };
        let training = self.training.clone();
        let mut callback_error = None;
        let result = training::batch::train_grpo_batch(
            self.trainer_mut()?,
            &sequences,
            &params,
            &training,
            &mut |progress| {
                if callback_error.is_some() {
                    return;
                }
                if let Some(callback) = callback.as_ref() {
                    Python::attach(|py| {
                        if let Err(error) = callback.call1(py, (progress_tuple(progress),)) {
                            callback_error = Some(error);
                        }
                    });
                }
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map(metrics_tuple).map_err(python_error)
    }

    #[pyo3(signature = (
        prompts,
        reward_command,
        *,
        reward_mode="persistent",
        reward_timeout_seconds=300,
        updates=1,
        rollout_batch_size=4,
        ppo_epochs=4,
        clip_range=0.2,
        kl_coefficient=0.01,
        critic_enabled=true,
        gamma=1.0,
        gae_lambda=0.95,
        value_learning_rate=0.01,
        value_epochs=8,
        feature_dtype="f32",
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=128,
        seed=42,
        callback=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn fit_ppo(
        &mut self,
        prompts: String,
        reward_command: Vec<String>,
        reward_mode: &str,
        reward_timeout_seconds: u64,
        updates: u32,
        rollout_batch_size: usize,
        ppo_epochs: u32,
        clip_range: f32,
        kl_coefficient: f32,
        critic_enabled: bool,
        gamma: f32,
        gae_lambda: f32,
        value_learning_rate: f32,
        value_epochs: u32,
        feature_dtype: &str,
        temperature: f32,
        top_p: f32,
        max_new_tokens: u32,
        seed: u32,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        if !feature_dtype.trim().eq_ignore_ascii_case("f32") && !critic_enabled {
            return Err(PyValueError::new_err(
                "feature_dtype only applies to the critic's feature matrix; \
                 enable the critic or drop the argument",
            ));
        }
        let config = PpoConfig {
            prompts: prompts.into(),
            reward_command,
            reward_protocol: parse_reward_protocol(reward_mode, reward_timeout_seconds)?,
            updates,
            rollout_batch_size,
            ppo_epochs,
            clip_range,
            kl_coefficient,
            critic: CriticConfig {
                enabled: critic_enabled,
                gamma,
                gae_lambda,
                value_lr: value_learning_rate,
                value_epochs,
                feature_dtype: parse_feature_dtype(feature_dtype)?,
            },
            sampling: SamplingParams {
                temperature,
                top_p,
                max_new_tokens,
                seed,
            },
        };
        let training = self.training.clone();
        let mut callback_error = None;
        let result = training::ppo::run(self.trainer_mut()?, &config, &training, &mut |progress| {
            if callback_error.is_some() {
                return;
            }
            if let Some(callback) = callback.as_ref() {
                Python::attach(|py| {
                    if let Err(error) = callback.call1(py, (progress_tuple(progress),)) {
                        callback_error = Some(error);
                    }
                });
            }
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map(metrics_tuple).map_err(python_error)
    }

    #[pyo3(signature = (
        prompts,
        reward_command,
        *,
        reward_mode="persistent",
        reward_timeout_seconds=300,
        updates=1,
        prompts_per_update=1,
        group_size=4,
        grpo_epochs=4,
        clip_range_low=0.2,
        clip_range_high=0.28,
        kl_coefficient=0.0,
        mask_truncated=false,
        max_new_tokens=128,
        seed=42,
        callback=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn fit_grpo(
        &mut self,
        prompts: String,
        reward_command: Vec<String>,
        reward_mode: &str,
        reward_timeout_seconds: u64,
        updates: u32,
        prompts_per_update: usize,
        group_size: usize,
        grpo_epochs: u32,
        clip_range_low: f32,
        clip_range_high: f32,
        kl_coefficient: f32,
        mask_truncated: bool,
        max_new_tokens: u32,
        seed: u32,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        let config = GrpoConfig {
            prompts: prompts.into(),
            reward_command,
            reward_protocol: parse_reward_protocol(reward_mode, reward_timeout_seconds)?,
            updates,
            prompts_per_update,
            group_size,
            grpo_epochs,
            clip_range_low,
            clip_range_high,
            kl_coefficient,
            mask_truncated,
            baseline: AdvantageBaseline::Mean,
            prompt_order: PromptOrder::Sequential,
            overlong_penalty: None,
            kl_schedule: None,
            dynamic_sampling: None,
            // This binding accepts a reward command; judge configuration belongs
            // to the document-based APIs.
            judge: None,
            max_stalled_updates: DEFAULT_MAX_STALLED_UPDATES,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens,
                seed,
            },
        };
        let training = self.training.clone();
        let mut callback_error = None;
        let result =
            training::grpo::run(self.trainer_mut()?, &config, &training, &mut |progress| {
                if callback_error.is_some() {
                    return;
                }
                if let Some(callback) = callback.as_ref() {
                    Python::attach(|py| {
                        if let Err(error) = callback.call1(py, (progress_tuple(progress),)) {
                            callback_error = Some(error);
                        }
                    });
                }
            });
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map(metrics_tuple).map_err(python_error)
    }

    #[pyo3(signature = (
        teacher_path,
        prompts,
        *,
        updates=1,
        prompts_per_update=1,
        samples_per_prompt=1,
        distill_epochs=1,
        clip_range_low=0.2,
        clip_range_high=0.28,
        weight_clip=5.0,
        kl_coefficient=0.0,
        mask_truncated=false,
        max_new_tokens=128,
        seed=42,
        callback=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn fit_distill(
        &mut self,
        teacher_path: String,
        prompts: String,
        updates: u32,
        prompts_per_update: usize,
        samples_per_prompt: usize,
        distill_epochs: u32,
        clip_range_low: f32,
        clip_range_high: f32,
        weight_clip: f32,
        kl_coefficient: f32,
        mask_truncated: bool,
        max_new_tokens: u32,
        seed: u32,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        let config = DistillConfig {
            // `fit_distill` is the on-policy binding. Offline top-k reads two
            // files and runs no sampler, so it belongs to `distill-teacher` and
            // a `[distill] mode = "topk_offline"` document, not to a call that
            // hands over a prompt list and a sampling seed.
            mode: DistillMode::OnPolicy,
            teacher_path: teacher_path.into(),
            prompts: prompts.into(),
            updates,
            prompts_per_update,
            samples_per_prompt,
            distill_epochs,
            clip_range_low,
            clip_range_high,
            weight_clip,
            kl_coefficient,
            mask_truncated,
            prompt_order: PromptOrder::Sequential,
            // Strictly on-policy, and not a parameter: the advantage subtracts
            // the behaviour log-probability the sampler recorded, so a modified
            // sampling distribution would make the two terms describe two
            // different policies.
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens,
                seed,
            },
        };
        let training = self.training.clone();
        let mut callback_error = None;
        let result =
            training::distill::run(self.trainer_mut()?, &config, &training, &mut |progress| {
                if callback_error.is_some() {
                    return;
                }
                if let Some(callback) = callback.as_ref() {
                    Python::attach(|py| {
                        if let Err(error) = callback.call1(py, (progress_tuple(progress),)) {
                            callback_error = Some(error);
                        }
                    });
                }
            });
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map(metrics_tuple).map_err(python_error)
    }

    #[pyo3(signature = (
        scenarios_json,
        judge_json,
        mcp_servers_json,
        *,
        environment_json=None,
        updates=1,
        scenarios_per_update=1,
        group_size=8,
        epochs=4,
        max_turns=6,
        max_new_tokens=512,
        max_trajectory_tokens,
        max_rollout_secs=300,
        end_on_no_tool_call=true,
        max_failed_turns=0,
        clip_range_low=0.2,
        clip_range_high=0.28,
        kl_coefficient=0.0,
        judge_failure="drop_group",
        max_dropped_fraction=0.5,
        drop_degenerate_groups=false,
        skip_empty_updates=false,
        truncation="drop",
        seed=42,
        callback=None
    ))]
    #[expect(clippy::too_many_arguments)]
    fn fit_agentic_grpo(
        &mut self,
        scenarios_json: &str,
        judge_json: Option<&str>,
        mcp_servers_json: &str,
        environment_json: Option<&str>,
        updates: u32,
        scenarios_per_update: usize,
        group_size: usize,
        epochs: u32,
        max_turns: usize,
        max_new_tokens: u32,
        max_trajectory_tokens: usize,
        max_rollout_secs: u64,
        end_on_no_tool_call: bool,
        max_failed_turns: usize,
        clip_range_low: f32,
        clip_range_high: f32,
        kl_coefficient: f32,
        judge_failure: &str,
        max_dropped_fraction: f32,
        drop_degenerate_groups: bool,
        skip_empty_updates: bool,
        truncation: &str,
        seed: u64,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<MetricsTuple> {
        let scenarios: Vec<Scenario> = serde_json::from_str(scenarios_json)
            .map_err(|error| PyValueError::new_err(format!("invalid scenarios JSON: {error}")))?;
        let mcp_configs: Vec<McpServerConfig> = serde_json::from_str(mcp_servers_json)
            .map_err(|error| PyValueError::new_err(format!("invalid MCP servers JSON: {error}")))?;
        // `None` is a run with no judge: only what the environment scored is
        // trained on. The pair is checked on the Python side, where the error
        // can name the two fields.
        let judge: Option<JudgeConfig> = judge_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|error| PyValueError::new_err(format!("invalid judge JSON: {error}")))?;
        // Same serialized shape as the TOML frontend and the server catalogue,
        // so `type = "container"` means one thing across the three of them.
        let environment: Option<EnvironmentConfig> = environment_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|error| PyValueError::new_err(format!("invalid environment JSON: {error}")))?;
        // Shared servers must be stateless when an environment owns trajectory
        // session state.
        if environment.is_some()
            && let Some(server) = mcp_configs.iter().find(|server| !server.stateless)
        {
            return Err(PyValueError::new_err(format!(
                "MCP server '{}' must set stateless = true before it can be shared with an environment; otherwise group members may contaminate each other",
                server.name
            )));
        }
        let failure = match judge_failure {
            "drop_group" => JudgeFailurePolicy::DropGroup,
            "fail" => JudgeFailurePolicy::Fail,
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown judge_failure '{value}'"
                )));
            }
        };
        let truncation = match truncation {
            "drop" => TruncationPolicy::Drop,
            "min_reward" => TruncationPolicy::MinReward,
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown truncation '{value}'"
                )));
            }
        };
        let config = AgentGrpoConfig {
            updates,
            scenarios_per_update,
            group_size,
            epochs,
            clip_range_low,
            clip_range_high,
            kl_coefficient,
            seed,
            limits: RolloutLimits {
                max_turns,
                max_new_tokens_per_turn: max_new_tokens,
                max_trajectory_tokens,
                max_rollout_secs,
                end_on_no_tool_call,
                max_failed_turns,
            },
            judge_failure: failure,
            max_dropped_fraction,
            drop_degenerate_groups,
            skip_empty_updates,
            truncation,
        };
        config.validate().map_err(agent_python_error)?;
        // Python resolves paths before handing them over, so the judge's cache
        // path is already absolute.
        let reward = judge
            .map(|judge| judge.build(std::path::Path::to_path_buf))
            .transpose()
            .map_err(agent_python_error)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|error| PyRuntimeError::new_err(format!("build agent runtime: {error}")))?;
        let local = tokio::task::LocalSet::new();
        // Connecting to a daemon, pulling an image and validating every task
        // declaration happens before a single token is generated.
        let mut environments = match &environment {
            Some(config) => Some(
                local
                    .block_on(&runtime, config.build())
                    .map_err(agent_python_error)?,
            ),
            None => None,
        };
        let provider = if mcp_configs.is_empty() {
            None
        } else {
            Some(
                local
                    .block_on(&runtime, McpToolProvider::connect(mcp_configs))
                    .map_err(agent_python_error)?,
            )
        };
        // McpToolProvider is intentionally not Clone; wrap the concrete value
        // once and retain a second Arc for bounded shutdown.
        let provider = provider.map(std::sync::Arc::new);
        let tools: Option<std::sync::Arc<dyn ToolProvider>> = provider
            .as_ref()
            .map(|provider| provider.clone() as std::sync::Arc<dyn ToolProvider>);
        // Merge shared servers into the environment's catalogue so the model sees
        // only tools the environment can route.
        if let Some(shared) = tools.clone()
            && let Some(world) = environments.take()
        {
            environments = Some(std::sync::Arc::new(
                local
                    .block_on(&runtime, SharedToolsFactory::new(world, shared))
                    .map_err(agent_python_error)?,
            ));
        }
        let trainer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("trainer is closed"))?;
        let training = self.training.clone();
        let mut callback_error = None;
        let mut on_progress = |progress: Progress| {
            if callback_error.is_some() {
                return;
            }
            if let Some(callback) = callback.as_ref() {
                Python::attach(|py| {
                    if let Err(error) = callback.call1(py, (progress_tuple(progress),)) {
                        callback_error = Some(error);
                    }
                });
            }
        };
        let mut run =
            AgenticRun::new(trainer, scenarios, config, training).on_progress(&mut on_progress);
        if let Some(reward) = reward {
            run = run.with_judge(reward);
        }
        if environments.is_none()
            && let Some(tools) = tools
        {
            run = run.with_tools(tools);
        }
        if let Some(environments) = environments {
            run = run.with_environments(environments);
        }
        let outcome = local.block_on(&runtime, run.run());
        if let Some(provider) = provider {
            local.block_on(&runtime, provider.shutdown());
        }
        self.inner = outcome.trainer;
        if let Some(error) = callback_error {
            return Err(error);
        }
        outcome
            .result
            .map(metrics_tuple)
            .map_err(agent_python_error)
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<i32>> {
        self.trainer()?.tokenize_text(text).map_err(python_error)
    }

    fn detokenize(&self, tokens: Vec<i32>) -> PyResult<String> {
        self.trainer()?
            .detokenize(&tokens, false)
            .map_err(python_error)
    }

    #[pyo3(signature = (messages, add_assistant=false))]
    fn format_chat(
        &self,
        messages: Vec<(String, String)>,
        add_assistant: bool,
    ) -> PyResult<String> {
        let borrowed = messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect::<Vec<_>>();
        self.trainer()?
            .format_chat(&borrowed, add_assistant)
            .map_err(python_error)
    }

    #[pyo3(signature = (prompt, *, temperature=1.0, top_p=1.0, max_new_tokens=128, seed=42))]
    fn generate(
        &mut self,
        prompt: Vec<i32>,
        temperature: f32,
        top_p: f32,
        max_new_tokens: u32,
        seed: u32,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        let generation = self
            .trainer_mut()?
            .generate(
                &prompt,
                &SamplingParams {
                    temperature,
                    top_p,
                    max_new_tokens,
                    seed,
                },
            )
            .map_err(python_error)?;
        Ok((generation.tokens, generation.logprobs))
    }

    fn score(&mut self, tokens: Vec<i32>) -> PyResult<Vec<f32>> {
        self.trainer_mut()?
            .score_tokens(&tokens)
            .map_err(python_error)
    }

    fn score_reference(&mut self, tokens: Vec<i32>) -> PyResult<Vec<f32>> {
        self.trainer_mut()?
            .score_reference_tokens(&tokens)
            .map_err(python_error)
    }

    fn hidden_states(&mut self, tokens: Vec<i32>) -> PyResult<Vec<f32>> {
        self.trainer_mut()?
            .hidden_states(&tokens)
            .map_err(python_error)
    }

    #[getter]
    fn context_size(&self) -> PyResult<usize> {
        self.trainer()?.context_size().map_err(python_error)
    }

    #[getter]
    fn eos_token(&self) -> PyResult<i32> {
        self.trainer()?.eos_token().map_err(python_error)
    }

    #[getter]
    fn hidden_size(&self) -> PyResult<usize> {
        self.trainer()?.hidden_size().map_err(python_error)
    }

    fn describe_lora(&self) -> PyResult<String> {
        self.trainer()?.describe_lora().map_err(python_error)
    }

    fn backend_report(&self) -> PyResult<String> {
        self.trainer()?.backend_report().map_err(python_error)
    }

    fn capability_report(&self) -> PyResult<String> {
        self.trainer()?.capability_report().map_err(python_error)
    }

    fn preflight(&mut self) -> PyResult<String> {
        self.trainer_mut()?.train_preflight().map_err(python_error)
    }

    fn close(&mut self) {
        self.inner = None;
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }
}

#[pyfunction]
fn list_backends() -> PyResult<String> {
    backend_list().map_err(python_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPreparedDataset>()?;
    module.add_class::<PyTrainer>()?;
    module.add_function(wrap_pyfunction!(list_backends, module)?)?;
    module.add(
        "RetrogradNativeError",
        module.py().get_type::<RetrogradNativeError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // No fixture reaches this branch, so it is exercised here.
    #[test]
    fn a_model_export_is_refused_for_an_architecture_no_row_covers() {
        let message = check_model_export(Some("gpt2")).expect_err("gpt2 has no row");
        assert!(message.contains("'gpt2'"), "{message}");
        assert!(
            message.contains("qwen2"),
            "the refusal names what is covered"
        );
        // Being listed is not enough: the row must grant the export.
        assert!(check_model_export(Some("llama")).is_err());
        assert!(check_model_export(Some("qwen2")).is_ok());
        assert!(check_model_export(None).is_ok());
    }

    #[test]
    fn a_selection_the_policy_would_ignore_is_refused_on_both_sides() {
        assert!(parse_trainable("lora", "all", vec![], false, false, false).is_ok());
        assert!(parse_trainable("lora", "all", vec![], true, false, false).is_err());
        assert!(parse_trainable("lora", "last:2", vec![], false, false, false).is_err());
        assert!(parse_trainable("full", "all", vec![], false, false, false).is_ok());
        assert!(parse_trainable("full", "all", vec![], false, true, false).is_err());
        assert!(parse_trainable("partial", "all", vec![], false, false, false).is_err());
        assert!(
            parse_trainable("partial", "all", vec!["attn".into()], false, false, false).is_ok()
        );
        // Hybrid takes norms and biases beside the adapter, and nothing else.
        assert!(parse_trainable("hybrid", "all", vec![], true, true, false).is_ok());
        assert!(parse_trainable("hybrid", "all", vec!["attn".into()], true, false, false).is_err());
        assert!(parse_trainable("hybrid", "all", vec![], true, false, true).is_err());
    }

    // The frontend must accept exactly the strings the runtime's parser accepts.
    #[test]
    fn a_layer_range_accepts_here_what_the_runtime_accepts() {
        let range = |layers: &str| {
            parse_trainable("partial", layers, vec!["attn".into()], false, false, false)
        };
        assert!(range("1..abc").is_err());
        assert!(range("1..4294967296").is_err());
        assert!(range("last:99999999999999999999").is_err());
        assert!(range("00..02").is_ok());
        assert!(range("last:+2").is_ok());
        assert!(range("0..4294967295").is_ok());
    }
}
