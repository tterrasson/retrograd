//! Types shared by every applicative crate, and the error façade they report
//! through.
//!
//! This crate sits at the bottom of the applicative chain: the configuration,
//! the engine, the training loops and the frontends read these types, and it
//! depends on none of them. [`Error`] is the one error type a user reads; the
//! crate that owns a source error writes its `From` into it. The kernel
//! catalogue schema is declared in `rir-core` and only re-exported here, since
//! the RIR chain never depends on this crate.

mod capability;
mod dims;
mod error;
mod execution;
mod hex;
mod json_pointer;
mod optimizer;
mod trainable;
mod types;
#[macro_use]
mod wire_enum;

pub use capability::{
    ArchitectureCapability, CAPABILITY_TABLE, architecture_capability, architecture_exports_model,
    model_export_architectures, tensor_family,
};
pub use dims::{saturating_dim, saturating_dim_product};
pub use error::{Error, ErrorKind, Result};
pub use execution::{
    CatalogContract, Contract, ContractError, DeviceLimits, DeviceProfile,
    EXECUTION_PROFILE_VERSION, ExecutionProfile, KERNEL_CATALOG_VERSION, KernelSelectionSummary,
    ModelCapabilities, OpPlacement, PREFLIGHT_REPORT_VERSION, PreflightReport, PreflightWarning,
    RuntimePolicy,
};
pub use hex::hex_lower;
pub use json_pointer::{PointerPath, escape_json_pointer};
pub use optimizer::{
    GEFEN_CODEBOOK_LEVELS, GEFEN_DEFAULT_BLOCK_SIZE, GEFEN_DEFAULT_MIN_NUMEL,
    GEFEN_ZERO_BLOCK_INDEX, HyperparameterBound, HyperparameterDefinition, HyperparameterValue,
    HyperparameterVector, OptimizerDescriptor, OptimizerKind, OptimizerPlan, PlannedParameter,
    PlannedSlot, SharedSlot, SlotDefinition, SlotDtype, SlotInit, SlotShape,
};
/// The catalogue schema, declared in `rir-core` and re-exported for the
/// applicative side that reads it.
pub use rir_core::catalog::{
    BackendPolicy, GgmlBackend, KernelCatalog, KernelDomainAssumption, KernelPolicy,
    KernelShapeRule, KernelVariantPolicy,
};
pub use trainable::{
    ALWAYS_FROZEN, ExclusionReason, LayerRange, OUTPUT_HEAD, OUTPUT_HEAD_BIAS, OUTPUT_NORM,
    ROPE_FREQS_SUFFIX, TENSOR_INVENTORY_VERSION, TensorDesc, TensorDtype, TensorInventory,
    TensorRole, TrainableEntry, TrainableExclusion, TrainablePolicy, TrainableRunConfig,
    TrainableSelector, TrainableSet, resolve_base,
};
pub use types::{
    ArtifactPolicy, CheckpointDtype, CheckpointMetadata, DEFAULT_CE_SEQ_CHUNK,
    DEFAULT_CHECKPOINT_STRIDE, DEFAULT_REWARD_TIMEOUT_SECONDS, Dataset, Device, EvalMetrics,
    FUSED_CE_K_MAX, FeatureDtype, FusedCeProbe, FusedCeWeightType, Generation, KernelImpl,
    KernelReject, KernelRunInfo, KvDtype, LoraConfig, LoraDtype, LrScheduler, MemoryReport,
    ModelInfo, ProbeOp, Progress, ResumeInfo, RewardMode, RewardProtocol, RirCounters, RirMode,
    SamplingParams, SharedPrefixFanout, TargetSet, TrainConfig, TrainMetrics, WeightedBatch,
    checkpoint_stride_for,
};
