"""Pythonic training and inference API backed by Retrograd's Rust/C++ runtime."""

from __future__ import annotations

from ._binding import load_native
from .agent import (
    AgenticGRPOConfig,
    CommandJudge,
    ContainerEnvironment,
    HttpEnvironment,
    JudgeContext,
    LocalEnvironment,
    McpServer,
    RulerJudge,
    SandboxLimits,
    SandboxPool,
    Scenario,
)
from .config import (
    CriticConfig,
    DatasetFormat,
    Device,
    DistillConfig,
    GefenOptions,
    GefenVariant,
    GRPOConfig,
    KvDtype,
    LoraConfig,
    LoraDtype,
    MasterWeights,
    MuonOptions,
    Optimizer,
    PPOConfig,
    SamplingConfig,
    Scheduler,
    TrainableConfig,
    TrainablePolicy,
    TrainingConfig,
)
from .models import (
    Backend,
    CheckpointFootprint,
    Dataset,
    Generation,
    ResumeState,
    TokenScores,
    TrainableExclusion,
    TrainableSelection,
    TrainableTensor,
    TrainingMetrics,
    TrainingProgress,
    TrainSequence,
    WeightedBatch,
)
from .trainer import RetrogradError, Trainer


def list_backends() -> tuple[Backend, ...]:
    """Return the CPU/GPU devices compiled into this native build."""

    native = load_native()
    try:
        report = native.list_backends()
    except native.RetrogradNativeError as error:
        raise RetrogradError(str(error)) from error
    backends = []
    for line in report.splitlines():
        if not line.strip():
            continue
        kind, name, description = line.split("\t", maxsplit=2)
        backends.append(Backend(kind, name, description))
    return tuple(backends)


__all__ = [
    "AgenticGRPOConfig",
    "Backend",
    "CheckpointFootprint",
    "CommandJudge",
    "ContainerEnvironment",
    "CriticConfig",
    "Dataset",
    "DatasetFormat",
    "Device",
    "DistillConfig",
    "GRPOConfig",
    "GefenOptions",
    "GefenVariant",
    "Generation",
    "HttpEnvironment",
    "JudgeContext",
    "KvDtype",
    "LocalEnvironment",
    "LoraConfig",
    "LoraDtype",
    "MasterWeights",
    "McpServer",
    "MuonOptions",
    "Optimizer",
    "PPOConfig",
    "ResumeState",
    "RetrogradError",
    "RulerJudge",
    "SamplingConfig",
    "SandboxLimits",
    "SandboxPool",
    "Scenario",
    "Scheduler",
    "TokenScores",
    "TrainSequence",
    "TrainableConfig",
    "TrainableExclusion",
    "TrainablePolicy",
    "TrainableSelection",
    "TrainableTensor",
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingProgress",
    "WeightedBatch",
    "list_backends",
]
