pub use retrograd_checkpoint as checkpoint;
pub use retrograd_config as config;
pub use retrograd_dataset as dataset;
pub use retrograd_engine as trainer;
pub use retrograd_memory as memory;
pub use retrograd_metrics as metrics;
pub use retrograd_run as run;
pub use retrograd_training as training;

// `Progress` and `Dataset` are re-exported under longer names: the facade
// already has a `training::Progress` and a `dataset` module, and neither is
// what a checkpoint records.
pub use retrograd_core::{
    ALWAYS_FROZEN, ArchitectureCapability, ArtifactPolicy, BASE_DTYPE_TABLE, BaseDtypeCapability,
    CAPABILITY_TABLE, CheckpointDtype, CheckpointMetadata, DEFAULT_CE_SEQ_CHUNK,
    DEFAULT_CHECKPOINT_STRIDE, Device, Error, EvalMetrics, FeatureDtype, FusedCeProbe,
    FusedCeWeightType, GEFEN_CODEBOOK_LEVELS, GEFEN_DEFAULT_BLOCK_SIZE, GEFEN_DEFAULT_MIN_NUMEL,
    GEFEN_ZERO_BLOCK_INDEX, GefenLayout, GefenVariant, Generation, HyperparameterValue,
    HyperparameterVector, KernelImpl, KernelReject, KernelRunInfo, KvDtype, LayerRange, LoraConfig,
    LoraDtype, LrScheduler, MemoryReport, ModelInfo, OUTPUT_HEAD, OUTPUT_HEAD_BIAS, OptimizerKind,
    OptimizerPlan, ProbeOp, Result, ResumeInfo, RewardMode, RewardProtocol, RirCounters, RirMode,
    SamplingParams, SharedPrefixFanout, SlotDefinition, SlotDtype, SlotInit, SlotShape,
    TENSOR_INVENTORY_VERSION, TargetSet, TensorDesc, TensorDtype, TensorInventory, TensorRole,
    TrainConfig, TrainMetrics, TrainableEntry, TrainablePolicy, TrainableRunConfig,
    TrainableSelector, TrainableSet, WeightedBatch, architecture_capability,
    architecture_exports_model, base_dtype_admits, base_dtype_backends, base_dtype_capability,
    base_dtype_is_tabled, checkpoint_stride_for, model_export_architectures, resolve_base,
    tensor_family,
};
pub use retrograd_core::{Dataset as DatasetRecord, Progress as CheckpointProgress};
pub use retrograd_engine::{
    DutyCycleStats, FusedCeProbeInputs, FusedCeProbeShape, OptimizerMemory, ProbeInputs,
    ProbeOperand, Trainer, TransferRates, backend_list, dequant_types, fused_sparse_ce_probe,
    fused_sparse_ce_probe_offloaded, gpu_runtime_available, model_info, parse_assistant_output,
    probe_op, probe_op_ex, render_chat_template_source, rir_census_report, rir_counters,
    rir_runtime_policy, rir_selection_selftest, rir_variant_report, set_rir_runtime_policy,
    slot_initial_bytes, tensor_inventory, tool_call_parser_from_source, transfer_probe,
};
pub use training::batch::{GrpoBatchParams, TrainSequence};
