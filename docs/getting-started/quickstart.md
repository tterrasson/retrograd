# Quickstart

This page trains a small LoRA adapter with supervised fine-tuning (SFT). You
need Rust with Cargo, a local GGUF model, and the repository cloned with its
submodule. Retrograd does not download or convert models.

## 1. Build

```bash
git clone --recurse-submodules https://github.com/tterrasson/retrograd.git
cd retrograd
cargo build --release
```

The binary is `target/release/retrograd`. On macOS the default build includes
Metal; elsewhere it is CPU-only. See [Build variants](../reference/builds) for
CUDA and Vulkan.

Check that the model loads:

```bash
./target/release/retrograd inspect --model /path/to/base.gguf
```

::: tip Smoke test
`examples/smoke_tiny_sft.toml` trains a rank-1 adapter in seconds. Run it with
`--model /path/to/base.gguf`, or run `scripts/fetch-cpu-fixture.sh` first to
use the bundled tiny test model.

```bash
./target/release/retrograd train examples/smoke_tiny_sft.toml --model /path/to/base.gguf
```
:::

## 2. Write a dataset

Save a few examples as `data/train.jsonl`, one conversation per line:

```jsonl
{"messages":[{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}]}
{"messages":[{"role":"user","content":"red"},{"role":"assistant","content":"blue"}]}
```

Only assistant messages are trained. See [Datasets](./datasets) for plain text
and validation rules.

## 3. Write a configuration

Save this as `configs/quickstart.toml`:

```toml
[run]
algorithm = "sft"

[model]
path = "/path/to/base.gguf"

[output]
path = "../artifacts/adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 256
epochs = 2
lr = 0.0001

[sft]
data = "../data/train.jsonl"
```

Relative paths resolve against the directory of the TOML file. Unknown keys
are errors, so a typo never goes unnoticed.

## 4. Train

```bash
./target/release/retrograd train configs/quickstart.toml
```

The adapter is written to `[output].path` at the end of the run. Add a
[`[checkpoint]`](../operations/checkpoints) section to make a run resumable.

## 5. Try the adapter

Chat with the base model and the adapter side by side:

```bash
./target/release/retrograd chat configs/quickstart.toml \
  --adapter artifacts/adapter.gguf --compare
```

The adapter is a standard LoRA GGUF, so llama.cpp loads it as well:

```bash
llama-cli -m /path/to/base.gguf --lora artifacts/adapter.gguf -p "ping"
```

## Next steps

Pick the algorithm that matches the signal you have:

| You have | Use |
| --- | --- |
| Example answers | [SFT](../training/sft) |
| A program that scores one answer | [PPO](../training/ppo) |
| A program that scores and ranks several answers to the same prompt | [GRPO](../training/grpo) |
| Tasks that need tool calls over several turns | [Agentic GRPO](../training/agent) |
| A larger model that already behaves well | [Distillation](../training/distill) |
