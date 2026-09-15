"""Check that the Python configuration matches the compiled PyO3 signatures.

The test compares keyword names from the extension's own signatures. Deliberate
omissions are listed in ``NOT_EXPOSED`` with their reasons.
"""

from __future__ import annotations

import inspect

import pytest

from retrograd import (
    AgenticGRPOConfig,
    CommandJudge,
    DistillConfig,
    GRPOConfig,
    LoraConfig,
    PPOConfig,
    SamplingConfig,
    TrainingConfig,
)
from retrograd._binding import load_native

#: Native keywords the façade knowingly does not surface.
#: These fields are intentionally absent from the Python façade. Keep the list
#: explicit so a new native keyword cannot become unreachable silently.
NOT_EXPOSED = {
    # The dtype retained activations are stored in. `TrainingConfig` exposes
    # `gradient_checkpointing` and `checkpoint_every_n_layers` but not this.
    "checkpoint_dtype",
    # A runtime guard that refuses to start when the weights did not land on the
    # device. Reachable from the TOML config and from the server, not from here.
    "require_gpu_resident",
}


def _keywords(callable_object: object, *, drop: set[str] = frozenset()) -> set[str]:
    """Keyword parameter names of a PyO3 callable, positional ones removed."""

    parameters = inspect.signature(callable_object).parameters
    return {
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    } - set(drop)


@pytest.fixture(scope="module")
def native():
    return load_native()


def test_the_training_keywords_are_the_ones_the_binding_accepts(native) -> None:
    accepted = _keywords(native._Trainer)
    sent = set(TrainingConfig().native_kwargs())

    unreachable = accepted - sent - NOT_EXPOSED
    assert not unreachable, (
        "the native trainer accepts keywords the Python façade never sends: "
        f"{sorted(unreachable)}. Add them to TrainingConfig.native_kwargs, or to "
        "NOT_EXPOSED with the reason."
    )

    rejected = sent - accepted
    assert not rejected, (
        "the Python façade sends keywords the native trainer does not accept: "
        f"{sorted(rejected)}. Every call would raise TypeError."
    )

    stale = NOT_EXPOSED - accepted
    assert not stale, (
        f"NOT_EXPOSED names keywords the binding no longer has: {sorted(stale)}. "
        "An exemption that outlives its field hides the next drift."
    )


def test_the_sampling_keywords_are_the_ones_generate_accepts(native) -> None:
    accepted = _keywords(native._Trainer.generate)
    sent = set(SamplingConfig().native_kwargs())
    assert accepted == sent, (
        "SamplingConfig and _Trainer.generate disagree: "
        f"only native {sorted(accepted - sent)}, only Python {sorted(sent - accepted)}"
    )


def test_the_lora_fields_are_the_ones_create_lora_accepts(native) -> None:
    accepted = _keywords(native._Trainer.create_lora)
    # `LoraConfig.targets` is translated by `native_targets()` - aliases expanded,
    # `"auto"` becoming `None` - so the field name is the contract, not its value.
    declared = set(LoraConfig.__dataclass_fields__)
    assert accepted == declared, (
        "LoraConfig and _Trainer.create_lora disagree: "
        f"only native {sorted(accepted - declared)}, "
        f"only Python {sorted(declared - accepted)}"
    )


def test_the_ppo_keywords_are_the_ones_fit_ppo_accepts(native) -> None:
    accepted = _keywords(native._Trainer.fit_ppo)
    config = PPOConfig(prompts="prompts.txt", reward_command=("true",))
    sent = set(config.native_kwargs()) | {"callback"}
    assert accepted == sent, (
        "PPOConfig and _Trainer.fit_ppo disagree: "
        f"only native {sorted(accepted - sent)}, only Python {sorted(sent - accepted)}"
    )


def test_the_grpo_keywords_are_the_ones_fit_grpo_accepts(native) -> None:
    accepted = _keywords(native._Trainer.fit_grpo)
    config = GRPOConfig(prompts="prompts.txt", reward_command=("true",))
    sent = set(config.native_kwargs()) | {"callback"}
    assert accepted == sent, (
        "GRPOConfig and _Trainer.fit_grpo disagree: "
        f"only native {sorted(accepted - sent)}, only Python {sorted(sent - accepted)}"
    )


def test_the_distill_keywords_are_the_ones_fit_distill_accepts(native) -> None:
    accepted = _keywords(native._Trainer.fit_distill)
    config = DistillConfig(teacher_path="teacher.gguf", prompts="prompts.jsonl")
    sent = set(config.native_kwargs()) | {"callback"}
    assert accepted == sent, (
        "DistillConfig and _Trainer.fit_distill disagree: "
        f"only native {sorted(accepted - sent)}, only Python {sorted(sent - accepted)}"
    )


def test_the_agentic_keywords_are_the_ones_fit_agentic_grpo_accepts(native) -> None:
    accepted = _keywords(native._Trainer.fit_agentic_grpo)
    config = AgenticGRPOConfig(
        scenarios="scenarios.jsonl",
        judge=CommandJudge(command=("true",)),
    )
    # `max_trajectory_tokens` is the one keyword the trainer computes rather
    # than reads: it defaults to the model's context size, which a configuration
    # does not know.
    sent = set(config.native_kwargs(max_trajectory_tokens=1024)) | {"callback"}
    assert accepted == sent, (
        "AgenticGRPOConfig and _Trainer.fit_agentic_grpo disagree: "
        f"only native {sorted(accepted - sent)}, only Python {sorted(sent - accepted)}"
    )
