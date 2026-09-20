from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias, TypeVar

from ._binding import load_native
from .agent import AgenticGRPOConfig
from .config import (
    DatasetFormat,
    DistillConfig,
    GRPOConfig,
    LoraConfig,
    PPOConfig,
    SamplingConfig,
    TrainingConfig,
)
from .models import (
    Dataset,
    Generation,
    TokenScores,
    TrainableSelection,
    TrainingMetrics,
    TrainingProgress,
    TrainSequence,
    WeightedBatch,
)

PathLike: TypeAlias = str | os.PathLike[str]
TokenInput: TypeAlias = str | Sequence[int]
Message: TypeAlias = tuple[str, str] | Mapping[str, str]
ProgressCallback: TypeAlias = Callable[[TrainingMetrics], None]
RolloutProgressCallback: TypeAlias = Callable[[TrainingProgress], None]

_T = TypeVar("_T")
_V = TypeVar("_V")


def _wrap_callback(
    callback: Callable[[_T], None] | None, transform: Callable[[_V], _T]
) -> Callable[[_V], None] | None:
    if callback is None:
        return None

    def _native_callback(value: _V) -> None:
        callback(transform(value))

    return _native_callback


class RetrogradError(RuntimeError):
    """Error reported by the native Rust/llama.cpp runtime."""


class Trainer:
    """High-level owner of a model, its optional LoRA adapter, and training state."""

    def __init__(
        self,
        model: PathLike,
        *,
        training: TrainingConfig | None = None,
        lora: LoraConfig | None = None,
        adapter: PathLike | None = None,
        _native_module: ModuleType | Any | None = None,
    ) -> None:
        if lora is not None and adapter is not None:
            raise ValueError("lora and adapter are mutually exclusive")
        config = training or TrainingConfig()
        policy = config.trainable.policy if config.trainable is not None else "lora"
        has_adapter = lora is not None or adapter is not None
        if policy in ("full", "partial") and has_adapter:
            raise ValueError(
                f"TrainingConfig.trainable policy '{policy}' trains base tensors and no "
                "adapter: drop lora/adapter, or use 'hybrid' to train both"
            )
        if policy == "hybrid" and not has_adapter:
            raise ValueError(
                "TrainingConfig.trainable policy 'hybrid' trains an adapter beside the "
                "base tensors: pass lora or adapter"
            )
        self._native_module = _native_module or load_native()
        native_kwargs = config.native_kwargs()
        # Resolve an unset shuffle seed before constructing the native trainer;
        # it inherits the LoRA seed when an adapter is being trained.
        if config.shuffle_seed is None and lora is not None:
            native_kwargs["shuffle_seed"] = lora.seed
        self._native = self._translate(
            self._native_module._Trainer,
            os.fspath(model),
            **native_kwargs,
        )
        self._training = config
        self._dataset_owner = object()
        if lora is not None:
            self.create_lora(lora)
        elif adapter is not None:
            self.load_adapter(adapter)

    def _translate(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except self._native_module.RetrogradNativeError as error:
            raise RetrogradError(str(error)) from error

    @property
    def closed(self) -> bool:
        return self._native.closed

    @property
    def context_size(self) -> int:
        return self._translate(lambda: self._native.context_size)

    @property
    def eos_token(self) -> int:
        return self._translate(lambda: self._native.eos_token)

    @property
    def hidden_size(self) -> int:
        return self._translate(lambda: self._native.hidden_size)

    def create_lora(self, config: LoraConfig | None = None) -> Trainer:
        config = config or LoraConfig()
        self._translate(
            self._native.create_lora,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            seed=config.seed,
            targets=config.native_targets(),
            dtype=config.dtype,
        )
        return self

    def load_adapter(self, path: PathLike) -> Trainer:
        self._translate(self._native.load_lora, os.fspath(path))
        return self

    def save_adapter(self, path: PathLike) -> Path:
        destination = Path(path)
        self._translate(self._native.save_lora, os.fspath(destination))
        return destination

    @property
    def trainable_set(self) -> TrainableSelection | None:
        """The base tensors this run resolved, or ``None`` for a LoRA run."""

        value = self._translate(lambda: self._native.trainable_set)
        return None if value is None else TrainableSelection.from_native(value)

    def save_trainable(self, path: PathLike) -> Path:
        """Write the trained base tensors, by absolute value, as a GGUF bundle.

        A bundle is not an adapter: a hybrid run's adapter is saved beside it
        with :meth:`save_adapter`.
        """

        destination = Path(path)
        self._translate(self._native.save_trainable, os.fspath(destination))
        return destination

    def load_trainable(self, path: PathLike) -> Trainer:
        """Restore base tensor values from such a bundle onto the live model."""

        self._translate(self._native.load_trainable, os.fspath(path))
        return self

    def save_model(self, path: PathLike) -> Path:
        """Write the whole model out as a standalone GGUF, trained weights included.

        The result needs neither the source model nor a bundle. Refused when
        no base tensor changes, when an adapter is carried, or when the
        architecture's export is not covered.
        """

        destination = Path(path)
        self._translate(self._native.save_model, os.fspath(destination))
        return destination

    def attach_reference(self, path: PathLike, *, context_size: int | None = None) -> Trainer:
        """Load the frozen model this run's reference term scores against.

        Held forward-only, so it costs its weights plus its KV cache. Its
        tokenizer must agree with this model's and its scores must be
        log-probabilities; both are checked here, not at the first token.

        Required by a base-weight run: without an anchor the reference falls
        back to this model with its adapter disabled, which is only equivalent
        while the base weights are frozen.
        """

        self._translate(self._native.attach_reference, os.fspath(path), n_ctx=context_size)
        return self

    @property
    def reference_path(self) -> str | None:
        return self._translate(lambda: self._native.reference_path)

    def prepare_dataset(
        self,
        path: PathLike,
        *,
        format: DatasetFormat = "auto",
        context_size: int | None = None,
    ) -> Dataset:
        source = Path(path)
        if format == "auto":
            format = "chat_jsonl" if source.suffix.lower() in {".json", ".jsonl"} else "text"
        if format not in ("text", "chat_jsonl"):
            raise ValueError("format must be auto, text, or chat_jsonl")
        size = context_size or self.context_size
        if size <= 0:
            raise ValueError("context_size must be greater than zero")
        handle = self._translate(self._native.prepare_dataset, os.fspath(source), format, size)
        return Dataset(handle, self._dataset_owner)

    def fit(
        self,
        train: Dataset | PathLike,
        *,
        eval: Dataset | PathLike | None = None,
        format: DatasetFormat = "auto",
        callback: ProgressCallback | None = None,
    ) -> TrainingMetrics:
        train_data = self._coerce_dataset(train, format)
        eval_data = self._coerce_dataset(eval, format) if eval is not None else None

        native_callback = _wrap_callback(callback, TrainingMetrics.from_native)

        values = self._translate(
            self._native.fit_sft,
            train_data._handle,
            eval_data._handle if eval_data is not None else None,
            native_callback,
        )
        return TrainingMetrics.from_native(values)

    fit_sft = fit

    def fit_ppo(
        self,
        config: PPOConfig,
        *,
        callback: RolloutProgressCallback | None = None,
    ) -> TrainingMetrics:
        native_callback = _wrap_callback(callback, TrainingProgress.from_native)

        values = self._translate(
            self._native.fit_ppo,
            os.fspath(config.prompts),
            list(config.reward_command),
            **config.native_kwargs(),
            callback=native_callback,
        )
        return TrainingMetrics.from_native(values)

    def fit_grpo(
        self,
        config: GRPOConfig,
        *,
        callback: RolloutProgressCallback | None = None,
    ) -> TrainingMetrics:
        if config.group_size > self._training.max_sequences:
            raise ValueError(
                "GRPO group_size exceeds TrainingConfig.max_sequences; "
                "configure both to the same value when creating the trainer"
            )
        native_callback = _wrap_callback(callback, TrainingProgress.from_native)

        values = self._translate(
            self._native.fit_grpo,
            os.fspath(config.prompts),
            list(config.reward_command),
            **config.native_kwargs(),
            callback=native_callback,
        )
        return TrainingMetrics.from_native(values)

    def fit_distill(
        self,
        config: DistillConfig,
        *,
        callback: RolloutProgressCallback | None = None,
    ) -> TrainingMetrics:
        """Distill a frozen teacher into this trainer's adapter, on-policy.

        The teacher is loaded here and held beside the student for the whole
        run - weights and KV cache, no adapter and therefore no optimizer state.
        Size the trainer with that second model in mind: nothing in this call
        can make room for it.
        """

        if config.samples_per_prompt > self._training.max_sequences:
            raise ValueError(
                "distillation samples_per_prompt exceeds TrainingConfig.max_sequences; "
                "the teacher scores a group as that many branched sequences"
            )
        native_callback = _wrap_callback(callback, TrainingProgress.from_native)

        values = self._translate(
            self._native.fit_distill,
            os.fspath(config.teacher_path),
            os.fspath(config.prompts),
            **config.native_kwargs(),
            callback=native_callback,
        )
        return TrainingMetrics.from_native(values)

    def train_grpo_batch(
        self,
        sequences: Sequence[TrainSequence],
        *,
        loss_denominator: int,
        epochs: int = 4,
        clip_range_low: float = 0.2,
        clip_range_high: float = 0.28,
        kl_coefficient: float = 0.0,
        seed: int = 42,
        scheduler_total_rollouts: int | None = None,
        callback: RolloutProgressCallback | None = None,
    ) -> TrainingMetrics:
        """Train a pre-generated, immutable-policy GRPO batch.

        ``scheduler_total_rollouts`` is the number of sequences the *whole*
        training run will optimize, not this batch. The native scheduler step
        accumulates across calls, so a decaying learning rate needs the global
        horizon; without it the rate would reach zero on the second call, and a
        non-constant scheduler is rejected outright.
        """
        rows = tuple(sequences)
        if not rows:
            raise ValueError("sequences must not be empty")
        if loss_denominator <= 0 or epochs <= 0:
            raise ValueError("loss_denominator and epochs must be greater than zero")
        if not 0 < clip_range_low < 1 or not 0 < clip_range_high < 1:
            raise ValueError("clip ranges must be in (0, 1)")
        if kl_coefficient < 0:
            raise ValueError("kl_coefficient must not be negative")
        if scheduler_total_rollouts is not None and scheduler_total_rollouts <= 0:
            raise ValueError("scheduler_total_rollouts must be greater than zero")

        native_callback = _wrap_callback(callback, TrainingProgress.from_native)
        values = self._translate(
            self._native.train_grpo_batch,
            [list(row.tokens) for row in rows],
            [list(row.old_logprobs) for row in rows],
            [list(row.train_mask) for row in rows],
            [row.reward for row in rows],
            [row.group_id for row in rows],
            intermediate_returns=[list(row.intermediate_returns) for row in rows],
            epochs=epochs,
            clip_range_low=clip_range_low,
            clip_range_high=clip_range_high,
            kl_coefficient=kl_coefficient,
            loss_denominator=loss_denominator,
            seed=seed,
            scheduler_total_rollouts=scheduler_total_rollouts,
            callback=native_callback,
        )
        return TrainingMetrics.from_native(values)

    def fit_agentic_grpo(
        self,
        config: AgenticGRPOConfig,
        *,
        callback: RolloutProgressCallback | None = None,
    ) -> TrainingMetrics:
        """Collect, judge, and train multi-turn on-policy GRPO trajectories."""
        trajectory_limit = config.max_trajectory_tokens or self.context_size
        if trajectory_limit > self.context_size:
            raise ValueError("max_trajectory_tokens must not exceed the model context size")
        native_callback = _wrap_callback(callback, TrainingProgress.from_native)
        values = self._translate(
            self._native.fit_agentic_grpo,
            config.scenarios_json(),
            config.judge_json(),
            config.mcp_servers_json(),
            **config.native_kwargs(max_trajectory_tokens=trajectory_limit),
            callback=native_callback,
        )
        return TrainingMetrics.from_native(values)

    def _coerce_dataset(self, value: Dataset | PathLike, format: DatasetFormat) -> Dataset:
        if isinstance(value, Dataset):
            if value._owner is not self._dataset_owner:
                raise ValueError("dataset was prepared by another Trainer")
            return value
        return self.prepare_dataset(value, format=format)

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self._translate(self._native.tokenize, text))

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self._translate(self._native.detokenize, list(tokens))

    def format_chat(self, messages: Sequence[Message], *, add_assistant: bool = False) -> str:
        normalized = [_normalize_message(message) for message in messages]
        return self._translate(self._native.format_chat, normalized, add_assistant)

    def generate(
        self,
        prompt: TokenInput,
        *,
        sampling: SamplingConfig | None = None,
    ) -> Generation:
        prompt_tokens = self._tokens(prompt)
        config = sampling or SamplingConfig()
        tokens, logprobs = self._translate(
            self._native.generate, list(prompt_tokens), **config.native_kwargs()
        )
        return Generation(
            text=self.detokenize(tokens),
            tokens=tuple(tokens),
            logprobs=tuple(logprobs),
            prompt_tokens=prompt_tokens,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        sampling: SamplingConfig | None = None,
    ) -> Generation:
        prompt = self.format_chat(messages, add_assistant=True)
        return self.generate(prompt, sampling=sampling)

    def score(self, value: TokenInput, *, reference: bool = False) -> TokenScores:
        tokens = self._tokens(value)
        operation = self._native.score_reference if reference else self._native.score
        logprobs = self._translate(operation, list(tokens))
        return TokenScores(tokens=tokens, logprobs=tuple(logprobs))

    def hidden_states(self, value: TokenInput) -> tuple[tuple[float, ...], ...]:
        tokens = self._tokens(value)
        width = self.hidden_size
        flat = self._translate(self._native.hidden_states, list(tokens))
        return tuple(tuple(flat[offset : offset + width]) for offset in range(0, len(flat), width))

    def train_tokens(self, tokens: Sequence[int]) -> TrainingMetrics:
        values = self._translate(self._native.train_tokens, list(tokens))
        return TrainingMetrics.from_native(values)

    def train_weighted(
        self, batch: WeightedBatch, *, scheduler_total_steps: int = 0
    ) -> TrainingMetrics:
        values = self._translate(
            self._native.train_weighted,
            list(batch.tokens),
            list(batch.labels),
            list(batch.weights),
            batch.rows,
            batch.context_size,
            scheduler_total_steps,
        )
        return TrainingMetrics.from_native(values)

    def _tokens(self, value: TokenInput) -> tuple[int, ...]:
        return self.tokenize(value) if isinstance(value, str) else tuple(value)

    def describe_adapter(self) -> str:
        return self._translate(self._native.describe_lora)

    def backend_report(self) -> str:
        return self._translate(self._native.backend_report)

    def capability_report(self) -> str:
        return self._translate(self._native.capability_report)

    def preflight(self) -> str:
        return self._translate(self._native.preflight)

    def close(self) -> None:
        self._native.close()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _normalize_message(message: Message) -> tuple[str, str]:
    if isinstance(message, Mapping):
        try:
            return message["role"], message["content"]
        except KeyError as error:
            raise ValueError("message mappings require role and content") from error
    if len(message) != 2:
        raise ValueError("messages must be (role, content) pairs")
    return message
