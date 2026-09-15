from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    epoch: int
    epoch_complete: bool
    global_step: int
    train_loss: float
    eval_loss: float
    tokens_per_second: float
    learning_rate: float

    @classmethod
    def from_native(cls, values: tuple[Any, ...]) -> TrainingMetrics:
        return cls(*values)


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    metrics: TrainingMetrics
    values: Mapping[str, float]

    @classmethod
    def from_native(cls, value: tuple[Any, Any]) -> TrainingProgress:
        metrics, values = value
        return cls(TrainingMetrics.from_native(metrics), dict(values))


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    prompt_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TokenScores:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]

    @property
    def total_logprob(self) -> float:
        return fsum(self.logprobs)

    @property
    def mean_logprob(self) -> float:
        return self.total_logprob / len(self.logprobs) if self.logprobs else float("nan")


@dataclass(frozen=True, slots=True)
class Backend:
    kind: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class WeightedBatch:
    """Advanced weighted objective used to implement PPO-like algorithms."""

    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    weights: tuple[float, ...]
    rows: int
    context_size: int

    def __post_init__(self) -> None:
        expected = self.rows * self.context_size
        if self.rows <= 0 or self.context_size <= 0:
            raise ValueError("rows and context_size must be greater than zero")
        if any(len(values) != expected for values in (self.tokens, self.labels, self.weights)):
            raise ValueError(f"tokens, labels, and weights must each contain {expected} values")


@dataclass(frozen=True, slots=True)
class TrainSequence:
    """One pre-generated on-policy trajectory for a GRPO batch update."""

    tokens: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    train_mask: tuple[bool, ...]
    reward: float
    group_id: int
    intermediate_returns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "old_logprobs", tuple(self.old_logprobs))
        object.__setattr__(self, "train_mask", tuple(self.train_mask))
        object.__setattr__(self, "intermediate_returns", tuple(self.intermediate_returns))
        if len(self.tokens) != len(self.train_mask):
            raise ValueError("tokens and train_mask must have the same length")
        if len(self.tokens) < 2:
            raise ValueError("a training sequence requires at least two tokens")
        if self.train_mask[0]:
            raise ValueError("the first token cannot be trainable")
        trained_tokens = sum(self.train_mask)
        if trained_tokens == 0:
            raise ValueError("a training sequence requires at least one trainable token")
        if len(self.old_logprobs) != trained_tokens:
            raise ValueError("old_logprobs must align with true train_mask positions")
        if not all(isfinite(value) for value in self.old_logprobs):
            raise ValueError("old_logprobs must all be finite")
        if not isfinite(self.reward):
            raise ValueError("reward must be finite")
        if self.group_id < 0 or self.group_id >= 2**64:
            raise ValueError("group_id must fit in an unsigned 64-bit integer")
        if self.intermediate_returns and len(self.intermediate_returns) != trained_tokens:
            raise ValueError("intermediate_returns must align with true train_mask positions")
        if not all(isfinite(value) for value in self.intermediate_returns):
            raise ValueError("intermediate_returns must all be finite")


class Dataset:
    """Prepared fixed-shape dataset whose storage remains owned by Rust."""

    __slots__ = ("_handle", "_owner")

    def __init__(self, handle: Any, owner: object) -> None:
        self._handle = handle
        self._owner = owner

    @property
    def examples(self) -> int:
        return self._handle.examples

    @property
    def context_size(self) -> int:
        return self._handle.context_size

    @property
    def supervised_tokens(self) -> int:
        return self._handle.supervised_tokens

    def __len__(self) -> int:
        return self.examples

    def __repr__(self) -> str:
        return (
            f"Dataset(examples={self.examples}, context_size={self.context_size}, "
            f"supervised_tokens={self.supervised_tokens})"
        )
