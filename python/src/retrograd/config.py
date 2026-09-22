from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

Device: TypeAlias = Literal["auto", "cpu", "gpu"]
Scheduler: TypeAlias = Literal["constant", "linear", "cosine"]
DatasetFormat: TypeAlias = Literal["auto", "text", "chat_jsonl"]
LoraDtype: TypeAlias = Literal["f32", "f16"]
KvDtype: TypeAlias = Literal["f32", "f16"]
MasterWeights: TypeAlias = Literal["auto", "f32", "off"]
#: Optimizer kind. ``muon`` and ``gefen`` take their own knobs from
#: :attr:`TrainingConfig.optimizer_options`.
Optimizer: TypeAlias = Literal["adamw", "sgd", "muon", "gefen"]
#: Gefen slot layout; a checkpoint only restores into a matching variant.
GefenVariant: TypeAlias = Literal["shared_v", "quantized_m"]
#: Which parameters carry a gradient. ``lora`` is the absence of :class:`TrainableConfig`,
#: so it is not selectable here.
TrainablePolicy: TypeAlias = Literal["full", "partial", "hybrid"]


#: Largest block index the runtime's ``u32`` parser can hold.
_MAX_BLOCK_INDEX = 2**32 - 1


def _parse_block_index(text: str, what: str) -> int:
    """Read a block index the way the runtime's ``u32`` parser does:
    leading zeros and ``+`` are accepted, anything else is refused here
    rather than surfacing as a native error.
    """

    digits = text[1:] if text.startswith("+") else text
    if not digits or not all("0" <= character <= "9" for character in digits):
        raise ValueError(f"layers {what} is not a number: {text!r}")
    value = int(digits)
    if value > _MAX_BLOCK_INDEX:
        raise ValueError(f"layers {what} is larger than a block index can be: {text!r}")
    return value


def _selects_every_layer(value: str) -> bool:
    """Whether this range is the whole model. ``"all"`` is the only spelling the
    runtime reads case-insensitively, so it is the only one read that way here."""

    return value.strip().lower() == "all"


def _validate_layer_range(value: str) -> None:
    """Check ``"all"``, ``"last:<count>"`` or ``"<first>..<last>"``."""

    text = value.strip()
    if _selects_every_layer(text):
        return
    if text.startswith("last:"):
        count = _parse_block_index(text[len("last:") :].strip(), "count")
        if count == 0:
            raise ValueError("layers 'last:<count>' requires a count greater than zero")
        return
    first, separator, last = text.partition("..")
    if not separator:
        raise ValueError("layers must be 'all', 'last:<count>', or '<first>..<last>'")
    if _parse_block_index(first.strip(), "lower bound") > _parse_block_index(
        last.strip(), "upper bound"
    ):
        raise ValueError(
            "layers '<first>..<last>' is inclusive on both ends, so the lower bound "
            "must not exceed the upper one"
        )


@dataclass(frozen=True, slots=True)
class TrainableConfig:
    """Which base tensors a run trains, and how it selects them.

    Leaving :attr:`TrainingConfig.trainable` unset is the LoRA policy, which is
    why ``"lora"`` is not one of these policies. ``full`` takes no selectors;
    ``partial`` and ``hybrid`` require at least one, and a selector the policy
    would ignore is refused rather than dropped.
    """

    policy: TrainablePolicy = "full"
    #: ``"all"``, ``"last:<count>"``, or ``"<first>..<last>"`` - inclusive on
    #: both ends, as it reads.
    layers: str = "all"
    #: Module aliases (``attn``, ``ffn``), stems (``attn_q``, ``ffn_up``), or
    #: explicit tensor patterns. Never norms: those follow :attr:`norms`.
    modules: tuple[str, ...] = ()
    #: Every normalization in the selected blocks, plus the model-wide ones,
    #: which sit in no block and so do not follow the layer range.
    norms: bool = False
    #: Every ``.bias`` inside the selected blocks, as its own family.
    biases: bool = False
    #: The vocabulary projection, independent of the layer range. Refused when
    #: the model ties it to the input embedding.
    output_head: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modules", tuple(self.modules))
        if self.policy == "lora":
            raise ValueError(
                "policy 'lora' is the absence of TrainableConfig: leave "
                "TrainingConfig.trainable unset to train an adapter alone"
            )
        if self.policy not in ("full", "partial", "hybrid"):
            raise ValueError("policy must be full, partial, or hybrid")
        _validate_layer_range(self.layers)
        selects = bool(self.modules) or self.norms or self.biases or self.output_head
        if self.policy == "full":
            if selects or not _selects_every_layer(self.layers):
                raise ValueError(
                    "policy 'full' trains every supported eligible tensor and takes no "
                    "selectors: use 'partial' to narrow it"
                )
        elif not selects:
            raise ValueError(
                f"policy '{self.policy}' selects nothing: set at least one of modules, "
                "norms, biases, or output_head"
            )
        if self.policy == "hybrid" and (self.modules or self.output_head):
            raise ValueError(
                "policy 'hybrid' permits only base norms and biases beside the adapter: "
                "modules and output_head are not supported with it"
            )

    def native_kwargs(self) -> dict[str, object]:
        return {
            "trainable_policy": self.policy,
            "trainable_layers": self.layers,
            "trainable_modules": list(self.modules),
            "trainable_norms": self.norms,
            "trainable_biases": self.biases,
            "trainable_output_head": self.output_head,
        }


def _lora_trainable_kwargs() -> dict[str, object]:
    # Built per call: the value carries a list that must not be shared.
    return {
        "trainable_policy": "lora",
        "trainable_layers": "all",
        "trainable_modules": [],
        "trainable_norms": False,
        "trainable_biases": False,
        "trainable_output_head": False,
    }


TARGET_ALIASES = {
    "q": "blk.*.attn_q.weight",
    "k": "blk.*.attn_k.weight",
    "v": "blk.*.attn_v.weight",
    "o": "blk.*.attn_output.weight",
    "ffn_up": "blk.*.ffn_up.weight",
    "ffn_down": "blk.*.ffn_down.weight",
    "ffn_gate": "blk.*.ffn_gate.weight",
}

#: Default LoRA target aliases. ``targets="auto"`` defers to the runtime's
#: per-architecture target set instead.
DEFAULT_TARGETS = tuple(TARGET_ALIASES)


@dataclass(frozen=True, slots=True)
class MuonOptions:
    """``[optimizer.muon]``: Muon's own coefficients.

    ``None`` fields keep the declared defaults. Only valid with
    :attr:`TrainingConfig.optimizer` set to ``muon``.
    """

    #: EMA coefficient of the momentum the update orthogonalizes.
    momentum: float | None = None
    nesterov: bool | None = None
    #: Newton-Schulz iterations; each adds to the update graph's size.
    ns_steps: int | None = None
    #: Added to the Frobenius norm before normalizing.
    ns_epsilon: float | None = None
    #: Rate for the parameters Muon declines (embeddings, head, norms, biases,
    #: LoRA factors), which AdamW updates instead. An absolute rate, not a
    #: ratio of :attr:`TrainingConfig.learning_rate`.
    fallback_learning_rate: float | None = None

    def native_kwargs(self) -> dict[str, object]:
        return {"muon": {key: value for key, value in _fields(self).items()}}


@dataclass(frozen=True, slots=True)
class GefenOptions:
    """``[optimizer.gefen]``: fixed-block second moments, optionally with a
    quantized first moment.

    Experimental. Only valid with :attr:`TrainingConfig.optimizer` set to
    ``gefen``.
    """

    #: ``shared_v``: F32 moments per block (``4N + 4K``). ``quantized_m``: the
    #: first moment is one byte per element over a shared 256-entry codebook,
    #: plus an F32 scale and second moment per block (``N + 8K``).
    variant: GefenVariant | None = None
    #: Elements per block; a positive power of two.
    block_size: int | None = None
    #: Below this element count a selected parameter falls back to AdamW.
    min_numel: int | None = None
    #: Only ``uniform`` exists.
    codebook: Literal["uniform"] | None = None
    #: Must be 256; the codebook index is one unsigned byte.
    codebook_levels: int | None = None
    #: Only ``fixed`` exists.
    partition: Literal["fixed"] | None = None
    beta1: float | None = None
    #: Per-block second-moment coefficient.
    beta2: float | None = None
    eps: float | None = None

    def native_kwargs(self) -> dict[str, object]:
        return {"gefen": {key: value for key, value in _fields(self).items()}}


def _fields(options: MuonOptions | GefenOptions) -> dict[str, object]:
    """Keys the caller set; ``None`` means the declared default, so drop it."""

    return {
        name: getattr(options, name)
        for name in options.__slots__
        if getattr(options, name) is not None
    }


#: The optimizer each options class configures.
_OPTIONS_OPTIMIZER: dict[type, str] = {MuonOptions: "muon", GefenOptions: "gefen"}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Runtime and optimizer settings used by a trainer."""

    context_size: int = 128
    #: Tokens processed by one physical forward/backward micro-batch.
    micro_batch: int = 32
    #: Micro-batches accumulated per optimizer step. Their product must divide
    #: ``context_size``; rollout objectives require it to equal that context.
    gradient_accumulation: int = 4
    max_sequences: int = 1
    generation_concurrency: int = 0
    generation_batch: int = 0
    fast_generation_context: bool = False
    #: KV-cache precision for the differentiable optimizer context. The runtime
    #: falls back to ``f32`` when the device lacks compatible flash attention.
    kv_dtype: KvDtype = "f16"
    threads: int = 0
    epochs: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    scheduler: Scheduler = "constant"
    warmup_steps: int = 0
    #: Which update step the optimizer graph builds. ``sgd`` keeps no
    #: per-parameter state, and its kernel is F32-only: pair it with
    #: ``LoraConfig(dtype="f32")`` rather than the F16 default.
    optimizer: Optimizer = "adamw"
    #: The optimizer's own knobs; must match :attr:`optimizer`.
    optimizer_options: MuonOptions | GefenOptions | None = None
    #: Which base tensors carry a gradient. ``None`` trains a LoRA adapter and
    #: leaves every base weight frozen.
    trainable: TrainableConfig | None = None
    #: Stream vocabulary logits in tiles instead of materializing
    #: ``[n_vocab, n_tokens]``.
    chunked_cross_entropy: bool = True
    chunked_ce_tiles: int = 8
    #: Maximum token count per fused cross-entropy chunk. ``0`` processes the
    #: whole step at once.
    chunked_ce_seq_chunk: int = 512
    gradient_checkpointing: bool = False
    #: Number of layers between retained activation checkpoints.
    checkpoint_every_n_layers: int = 4
    #: Whether half-precision base weights are trained through an F32 master
    #: copy: ``"auto"``, ``"f32"`` or ``"off"``. Without one the update is
    #: rounded back into the store it read, so a step under one unit in the
    #: last place of that store is a whole unit taken at random and the run is
    #: refused. ``auto`` keeps a copy exactly when a marked base tensor is half
    #: precision, at four bytes per trained element.
    master_weights: MasterWeights = "auto"
    #: Shuffle SFT training rows at the start of every epoch. Evaluation rows
    #: retain their order; this option is ignored outside SFT.
    shuffle: bool = True
    #: Seed for per-epoch SFT shuffles. ``None`` inherits the LoRA seed.
    shuffle_seed: int | None = None
    #: Upper bound on the fraction of wall time the trainer spends waiting on
    #: GPU work it submitted, so another workload gets regular compute windows.
    #: ``None`` - the default, and what ``1.0`` means - leaves the trainer
    #: unthrottled. Releases compute, not device memory. Accepted but inactive
    #: on a CPU device; enabling it costs decode pipelining once, so it is worth
    #: its overhead at ``0.75`` and below.
    max_gpu_duty_cycle: float | None = None
    device: Device = "auto"
    verbose: bool = False

    @property
    def step_tokens(self) -> int:
        """Tokens one optimizer step trains: llama.cpp's ``n_batch``."""

        return self.micro_batch * self.gradient_accumulation

    def __post_init__(self) -> None:
        positive = {
            "context_size": self.context_size,
            "micro_batch": self.micro_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "max_sequences": self.max_sequences,
            "epochs": self.epochs,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.threads < 0:
            raise ValueError("threads must not be negative")
        if self.context_size % self.step_tokens:
            raise ValueError(
                "context_size must be divisible by micro_batch * gradient_accumulation"
            )
        if self.max_sequences > 256:
            raise ValueError("max_sequences must not exceed 256")
        if self.max_sequences > self.step_tokens:
            raise ValueError("max_sequences must not exceed micro_batch * gradient_accumulation")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must not be negative")
        if self.generation_concurrency < 0:
            raise ValueError("generation_concurrency must not be negative")
        if self.generation_concurrency > self.step_tokens:
            raise ValueError(
                "generation_concurrency must not exceed micro_batch * gradient_accumulation"
            )
        if self.generation_concurrency > 256:
            raise ValueError("generation_concurrency must not exceed 256")
        if self.generation_batch < 0:
            raise ValueError("generation_batch must not be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero")
        if self.chunked_ce_tiles <= 0:
            raise ValueError("chunked_ce_tiles must be greater than zero")
        if self.chunked_ce_seq_chunk < 0:
            raise ValueError("chunked_ce_seq_chunk must not be negative")
        if self.checkpoint_every_n_layers <= 0:
            raise ValueError("checkpoint_every_n_layers must be greater than zero")
        if self.shuffle_seed is not None and self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must not be negative")
        if self.max_gpu_duty_cycle is not None and not (0.0 < self.max_gpu_duty_cycle <= 1.0):
            raise ValueError("max_gpu_duty_cycle must be in (0, 1]")
        if self.scheduler not in ("constant", "linear", "cosine"):
            raise ValueError("scheduler must be constant, linear, or cosine")
        if self.device not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be auto, cpu, or gpu")
        if self.kv_dtype not in ("f32", "f16"):
            raise ValueError("kv_dtype must be f32 or f16")
        if self.master_weights not in ("auto", "f32", "off"):
            raise ValueError("master_weights must be auto, f32, or off")
        if self.optimizer not in ("adamw", "sgd", "muon", "gefen"):
            raise ValueError("optimizer must be adamw, sgd, muon or gefen")
        if self.optimizer_options is not None:
            expected = _OPTIONS_OPTIMIZER.get(type(self.optimizer_options))
            if expected is None:
                raise ValueError("optimizer_options must be a MuonOptions or a GefenOptions")
            if expected != self.optimizer:
                raise ValueError(
                    f"{type(self.optimizer_options).__name__} configures an optimizer "
                    f"this run does not use: optimizer = {self.optimizer!r}"
                )

    def native_kwargs(self) -> dict[str, object]:
        return {
            "n_ctx": self.context_size,
            "n_batch": self.step_tokens,
            "n_ubatch": self.micro_batch,
            "n_seq_max": self.max_sequences,
            "generation_concurrency": self.generation_concurrency,
            "generation_batch": self.generation_batch,
            "fast_generation_context": self.fast_generation_context,
            "kv_dtype": self.kv_dtype,
            "threads": self.threads,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "optimizer": self.optimizer,
            **({} if self.optimizer_options is None else self.optimizer_options.native_kwargs()),
            **(
                _lora_trainable_kwargs()
                if self.trainable is None
                else self.trainable.native_kwargs()
            ),
            "chunked_cross_entropy": self.chunked_cross_entropy,
            "chunked_ce_tiles": self.chunked_ce_tiles,
            "chunked_ce_seq_chunk": self.chunked_ce_seq_chunk,
            "gradient_checkpointing": self.gradient_checkpointing,
            "checkpoint_every_n_layers": self.checkpoint_every_n_layers,
            "master_weights": self.master_weights,
            "shuffle": self.shuffle,
            # `Trainer` replaces this fallback with the LoRA seed when needed.
            "shuffle_seed": 42 if self.shuffle_seed is None else self.shuffle_seed,
            # The C ABI has no need for the distinction the dataclass keeps
            # between omitted and explicit 1.0: both mean no limit.
            "max_gpu_duty_cycle": (
                1.0 if self.max_gpu_duty_cycle is None else self.max_gpu_duty_cycle
            ),
            "device": self.device,
            "verbose": self.verbose,
        }


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """LoRA adapter settings. Targets accept aliases such as ``q`` and ``v``."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    seed: int = 42
    targets: Literal["auto", "qv"] | tuple[str, ...] = DEFAULT_TARGETS
    dtype: LoraDtype = "f16"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero")
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.targets, str):
            object.__setattr__(self, "targets", tuple(self.targets))
            if not self.targets:
                raise ValueError("targets must not be empty; use 'auto' for automatic targets")
        elif self.targets not in ("auto", "qv"):
            raise ValueError("string targets must be 'auto' or 'qv'")
        if self.dtype not in ("f32", "f16"):
            raise ValueError("dtype must be f32 or f16")

    def native_targets(self) -> list[str] | None:
        if self.targets == "auto":
            return None
        targets = ("q", "v") if self.targets == "qv" else self.targets
        return [TARGET_ALIASES.get(target, target) for target in targets]


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

    def native_kwargs(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class CriticConfig:
    enabled: bool = True
    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 1.0e-2
    epochs: int = 8
    #: Storage precision of the critic's feature matrix. Narrower types reduce
    #: memory at the cost of a lossy feature round trip.
    feature_dtype: Literal["f32", "f16", "bf16"] = "f32"

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("critic learning_rate and epochs must be greater than zero")
        if self.feature_dtype not in ("f32", "f16", "bf16"):
            raise ValueError("feature_dtype must be f32, f16, or bf16")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    prompts: str | Path
    reward_command: tuple[str, ...]
    #: Reward command protocol: ``"persistent"`` reuses one worker after the
    #: ``retrograd-reward/1`` handshake; ``"oneshot"`` starts one per batch.
    reward_mode: str = "persistent"
    #: Deadline of one reward batch, in seconds. In the persistent mode the
    #: first batch also pays whatever the worker loads at startup.
    reward_timeout_seconds: int = 300
    updates: int = 1
    rollout_batch_size: int = 4
    epochs: int = 4
    clip_range: float = 0.2
    kl_coefficient: float = 0.01
    critic: CriticConfig = CriticConfig()
    sampling: SamplingConfig = SamplingConfig()

    def native_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by ``_Trainer.fit_ppo``."""

        return {
            "reward_mode": self.reward_mode,
            "reward_timeout_seconds": self.reward_timeout_seconds,
            "updates": self.updates,
            "rollout_batch_size": self.rollout_batch_size,
            "ppo_epochs": self.epochs,
            "clip_range": self.clip_range,
            "kl_coefficient": self.kl_coefficient,
            "critic_enabled": self.critic.enabled,
            "gamma": self.critic.gamma,
            "gae_lambda": self.critic.gae_lambda,
            "value_learning_rate": self.critic.learning_rate,
            "value_epochs": self.critic.epochs,
            "feature_dtype": self.critic.feature_dtype,
            **self.sampling.native_kwargs(),
        }

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_command", tuple(self.reward_command))
        if not self.reward_command:
            raise ValueError("reward_command must not be empty")
        if self.reward_mode not in ("persistent", "oneshot"):
            raise ValueError("reward_mode must be 'persistent' or 'oneshot'")
        if self.reward_timeout_seconds <= 0:
            raise ValueError("reward_timeout_seconds must be greater than zero")
        if min(self.updates, self.rollout_batch_size, self.epochs) <= 0:
            raise ValueError("updates, rollout_batch_size, and epochs must be greater than zero")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be in (0, 1)")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must not be negative")


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    prompts: str | Path
    reward_command: tuple[str, ...]
    #: Reward command protocol: ``"persistent"`` reuses one worker after the
    #: ``retrograd-reward/1`` handshake; ``"oneshot"`` starts one per batch.
    reward_mode: str = "persistent"
    #: Deadline of one reward batch, in seconds. In the persistent mode the
    #: first batch also pays whatever the worker loads at startup.
    reward_timeout_seconds: int = 300
    updates: int = 1
    prompts_per_update: int = 1
    group_size: int = 4
    epochs: int = 4
    clip_range_low: float = 0.2
    clip_range_high: float = 0.28
    kl_coefficient: float = 0.0
    mask_truncated: bool = False
    max_new_tokens: int = 128
    seed: int = 42

    def native_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by ``_Trainer.fit_grpo``."""

        return {
            "reward_mode": self.reward_mode,
            "reward_timeout_seconds": self.reward_timeout_seconds,
            "updates": self.updates,
            "prompts_per_update": self.prompts_per_update,
            "group_size": self.group_size,
            "grpo_epochs": self.epochs,
            "clip_range_low": self.clip_range_low,
            "clip_range_high": self.clip_range_high,
            "kl_coefficient": self.kl_coefficient,
            "mask_truncated": self.mask_truncated,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_command", tuple(self.reward_command))
        if not self.reward_command:
            raise ValueError("reward_command must not be empty")
        if self.reward_mode not in ("persistent", "oneshot"):
            raise ValueError("reward_mode must be 'persistent' or 'oneshot'")
        if self.reward_timeout_seconds <= 0:
            raise ValueError("reward_timeout_seconds must be greater than zero")
        if min(self.updates, self.prompts_per_update, self.epochs) <= 0:
            raise ValueError("updates, prompts_per_update, and epochs must be greater than zero")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.group_size > 256:
            raise ValueError("group_size must not exceed 256")
        if not 0 < self.clip_range_low <= self.clip_range_high < 1:
            raise ValueError("clip ranges must satisfy 0 < low <= high < 1")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must not be negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")


@dataclass(frozen=True, slots=True)
class DistillConfig:
    """On-policy distillation against a frozen teacher.

    The student samples, the teacher scores the very same tokens, and the
    log-probability gap becomes a dense per-token advantage. There is no reward
    command and no judge: the teacher *is* the objective, which is why this
    config carries a second model path where :class:`GRPOConfig` carries a
    command.
    """

    #: The teacher GGUF. Held for inference only - it never gets an adapter, so
    #: it costs its weights plus its KV cache and nothing else. Its tokenizer
    #: must match the student's; the run refuses the pair otherwise, before it
    #: scores a single token.
    teacher_path: str | Path
    prompts: str | Path
    updates: int = 1
    prompts_per_update: int = 1
    #: One sample per prompt is admissible, unlike ``GRPOConfig.group_size``:
    #: the signal is dense and per-token, so a group of one still carries it.
    samples_per_prompt: int = 1
    #: ``1`` is strictly on-policy - the policy ratio is exactly 1 and the token
    #: weight is the advantage itself.
    epochs: int = 1
    clip_range_low: float = 0.2
    clip_range_high: float = 0.28
    #: Bound on ``|A_t|``, in nats. Not cosmetic: one token whose gap runs to
    #: -20 would own the update once gradient-norm clipping rescales the rest.
    weight_clip: float = 5.0
    #: The teacher already anchors the policy, so the base-model reference pass
    #: is skipped at zero. Above zero it is added back on top.
    kl_coefficient: float = 0.0
    mask_truncated: bool = True
    max_new_tokens: int = 128
    seed: int = 42

    def native_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by ``_Trainer.fit_distill``."""

        return {
            "updates": self.updates,
            "prompts_per_update": self.prompts_per_update,
            "samples_per_prompt": self.samples_per_prompt,
            "distill_epochs": self.epochs,
            "clip_range_low": self.clip_range_low,
            "clip_range_high": self.clip_range_high,
            "weight_clip": self.weight_clip,
            "kl_coefficient": self.kl_coefficient,
            "mask_truncated": self.mask_truncated,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }

    def __post_init__(self) -> None:
        if min(self.updates, self.prompts_per_update, self.epochs) <= 0:
            raise ValueError("updates, prompts_per_update, and epochs must be greater than zero")
        if not 1 <= self.samples_per_prompt <= 256:
            raise ValueError("samples_per_prompt must be between 1 and 256")
        if not 0 < self.clip_range_low <= self.clip_range_high < 1:
            raise ValueError("clip ranges must satisfy 0 < low <= high < 1")
        if not self.weight_clip > 0 or self.weight_clip == float("inf"):
            raise ValueError("weight_clip must be finite and greater than zero")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must not be negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
