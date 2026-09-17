# Retrograd Python

This directory is an optional, high-level Python API for Retrograd. It does not
replace the Rust CLI and it does not add Python to the Rust/C++ training path.

The dependency direction stays one-way:

```text
Python API -> small PyO3 module -> public Retrograd Rust API -> C++/llama.cpp
```

Datasets and model state remain native. Python receives small immutable value
objects (`TrainingMetrics`, `Generation`, `TokenScores`) rather than C structs
or raw pointers.

## Install for development

Initialize the llama.cpp submodule from the repository root first. Then let
`uv` create the environment and build the native extension:

```sh
scripts/setup-llama-cpp.sh
cd python
uv sync
```

The native build honors the same environment variables as Cargo, notably
`LLAMA_CPP_DIR` and `RETRO_STRICT_LLAMA`. Wheels enable Metal on macOS by
default and CPU elsewhere. For an editable CPU-only build from `python/`:

```sh
uv run --config-settings-package 'retrograd:build-args=--no-default-features' --reinstall-package retrograd python -c 'import retrograd; print(retrograd.list_backends())'
```

GPU features are `platform-gpu`, `metal`, `vulkan`, and `cuda`. For example,
set the per-package value to
`retrograd:build-args=--features vulkan` for Vulkan instead of the platform
default (a named backend replaces `platform-gpu`).
Force reinstallation whenever changing the selection. To restore the platform
default, run `uv run --reinstall-package retrograd python -c
'import retrograd; print(retrograd.list_backends())'` without the CPU setting.
The extension stays statically linked in `python/target` (unless
`CARGO_TARGET_DIR` overrides it).

Run the checks with:

```sh
uv run pytest
uv run ruff check .
cargo check --manifest-path native/Cargo.toml
```

## SFT

```python
from retrograd import LoraConfig, Trainer, TrainingConfig

with Trainer(
    "model.gguf",
    training=TrainingConfig(
        epochs=3,
        learning_rate=1e-4,
        device="auto",
    ),
    lora=LoraConfig(rank=8, alpha=16, targets=("q", "v")),
) as trainer:
    train = trainer.prepare_dataset("train.jsonl")
    eval_data = trainer.prepare_dataset("eval.jsonl")
    metrics = trainer.fit(train, eval=eval_data, callback=print)
    trainer.save_adapter("adapter.gguf")
```

`.json` and `.jsonl` files use Retrograd's chat-template-aware assistant
masking. Other extensions use the overlapping next-token text preparation.
The optional `format=` argument can override detection.

## Sharing a GPU

`TrainingConfig(max_gpu_duty_cycle=0.5)` bounds the fraction of wall time the
trainer spends waiting on GPU work it submitted, so another workload gets
regular compute windows in between. It covers generation, scoring and the
optimizer, and needs no NVML, MPS or elevated permissions.

It releases **compute time, not device memory**: weights, KV caches and
optimizer state stay allocated while the trainer sleeps, so a neighbour that
needs VRAM still needs a smaller `micro_batch` or a lower
`generation_concurrency`.

Enabling it costs the overlap between host-side sampling and the previous decode,
once, before any sleep - so `0.99` is not approximately `1.0`. It is worth its
overhead at `0.75` and below. The value is accepted but inactive when the device
resolves to CPU, where `threads` remains the control.

## Generation and chat

```python
from retrograd import SamplingConfig, Trainer

with Trainer("model.gguf", adapter="adapter.gguf") as trainer:
    answer = trainer.chat(
        [{"role": "user", "content": "Explain LoRA in one sentence."}],
        sampling=SamplingConfig(temperature=0.7, top_p=0.9, max_new_tokens=80),
    )
    print(answer.text)
    print(answer.tokens, answer.logprobs)
```

The same object exposes tokenization, chat formatting, teacher-forced scoring,
reference-model scoring, hidden states, backend/capability reports, and
preflight checks.

## PPO and GRPO

The complete Rust PPO and GRPO loops are also available. They retain the CLI's
safe external reward protocol: `reward_command` is an argv tuple, never a shell
string. As on the CLI, it is started once and kept alive for the run
(`reward_mode="persistent"`, the default): the command answers a one-line
`{"protocol": "retrograd-reward/1"}` handshake and then flushes one response
line per request. `reward_mode="oneshot"` spawns one process per batch
instead. Progress includes both common `TrainingMetrics` and the algorithm's
named metrics.

```python
from retrograd import GRPOConfig, LoraConfig, Trainer, TrainingConfig

# The multi-sequence context is sized when the model is loaded.
trainer = Trainer(
    "model.gguf",
    training=TrainingConfig(max_sequences=8),
    lora=LoraConfig(),
)

metrics = trainer.fit_grpo(
    GRPOConfig(
        prompts="prompts.jsonl",
        reward_command=("python", "reward.py"),
        updates=10,
        prompts_per_update=4,
        group_size=8,
    ),
    callback=lambda progress: print(progress.values),
)
```

`WeightedBatch` and `train_weighted()` additionally expose the differentiable
weighted objective for custom reinforcement-learning orchestration, behind a
validated Python value object.

## Agentic GRPO and MCP

`Trainer.fit_agentic_grpo()` keeps multi-turn collection, MCP calls, RULER
scoring and the GRPO update in Rust. `Trainer.train_grpo_batch()` is the
lower-level convergence point for externally collected `TrainSequence`
objects. See [`../docs/engineering/AGENTIC_GRPO.md`](../docs/engineering/AGENTIC_GRPO.md) for the
configuration, safety invariants and complete examples.

See [`examples/sft.py`](examples/sft.py) for a complete training lifecycle.

## Offline sequence-KD

`scripts/sequence_kd.py` samples a teacher over a prompt-only chat JSONL, filters
the candidates, and writes an ordinary chat corpus — so training on a teacher's
answers is a plain `sft` run and needs nothing new anywhere:

```sh
uv run python scripts/sequence_kd.py --model teacher.gguf \
  --prompts prompts.jsonl --out corpus.jsonl --samples 4 --keep 2 \
  --verifier ./check.sh --report corpus.report.json
```

The verifier is any command speaking one JSON line each way (`prompt`,
`completion`, `reference`, `rubric`, `index` in; `{"keep": bool}` or
`{"score": float}` with `--threshold` out), which is how both an exact checker
and a judge are wired. It is started once for the whole run. Prompt lines may
end on an `assistant` turn: that answer is the reference handed to the verifier,
never something the teacher is asked to reproduce.
