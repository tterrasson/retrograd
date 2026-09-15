# Quickstart

This guide runs a small supervised fine-tuning (SFT) job. You need a local GGUF
model, Rust with Cargo, and a writable directory for the adapter and checkpoints.
Run the commands from the repository root. Retrograd does not download or
convert models for you.

## 1. Build the CLI

```bash
cargo build --release
```

The binary is written to `target/release/retrograd`. Check that it can load the
model before training:

```bash
./target/release/retrograd inspect --model /path/to/base.gguf --device auto
```

Use `--device cpu` for a CPU-only run, or `--device gpu` to require an available
compiled GPU backend. `auto` uses an available compiled GPU backend and otherwise
falls back to CPU. See [Build variants](../reference/builds) to enable a backend.

## 2. Create a dataset

For a first run, use a plain text file such as `data/train.txt`:

```text
Question: ping
Answer: pong

Question: red
Answer: blue

Question: small
Answer: tiny
```

See [Datasets and paths](./datasets) for chat JSONL and validation rules.

## 3. Write a configuration

Save this as `configs/quickstart.toml`:

```toml
[run]
algorithm = "sft"
verbose = true

[model]
path = "/path/to/base.gguf"
device = "auto"

[lora]
output = "../artifacts/quickstart-adapter.gguf"
rank = 8
alpha = 16.0
seed = 42
dtype = "f16"

[training]
ctx = 256
micro_batch = 32
gradient_accumulation = 1
epochs = 2
lr = 0.0001
lr_scheduler = "constant"
warmup_steps = 0
weight_decay = 0.0
max_grad_norm = 1.0

[sft]
data = "../data/train.txt"
data_format = "text"
shuffle = true

[checkpoint]
directory = "../artifacts/quickstart-checkpoints"
mode = "steps"
every_steps = 50
```

Relative paths in the TOML file resolve against the directory containing that
file. `lora.output` and `[checkpoint].directory` must point to writable
locations.

The configuration is strict. Unknown keys are rejected, and only the section
matching `[run].algorithm` may be present. For example, an SFT file must contain
`[sft]` and must not also contain `[ppo]` or `[grpo]`.

## 4. Run training

```bash
./target/release/retrograd train configs/quickstart.toml
```

The CLI reports the selected model, output adapter, progress, loss, and
throughput. At the end, the adapter is available at the configured
`lora.output` path. Interrupting a run does not create a resumable checkpoint
unless `[checkpoint]` is configured.

## 5. Test the adapter

Run a benchmark against a compatible evaluation file:

```bash
./target/release/retrograd bench configs/quickstart.toml \
  --data data/eval.txt \
  --adapter artifacts/quickstart-adapter.gguf
```

Or start an interactive session:

```bash
./target/release/retrograd chat configs/quickstart.toml \
  --adapter artifacts/quickstart-adapter.gguf \
  --compare
```

The `chat` command uses the model's GGUF chat template. A text SFT dataset is
not automatically a chat template; use chat JSONL when the model expects
role-tagged conversations.

The adapter is a standalone LoRA GGUF. It works with the base model it was
trained on - same architecture and tensor names - for example through
`llama-cli`:

```bash
llama-cli -m /path/to/base.gguf --lora artifacts/quickstart-adapter.gguf \
  -p "Question: ping\nAnswer:" -n 64
llama-cli -m /path/to/base.gguf --lora-scaled artifacts/quickstart-adapter.gguf:0.5 \
  -p "Question: ping\nAnswer:" -n 64
```

## Choosing an algorithm

Use SFT when each training example includes the response that should be
learned. Use PPO when an external program can return one scalar reward for each
sampled response. Use GRPO when each prompt can be sampled several times and
the reward is most useful as a ranking within that group.

- [SFT training](../training/sft)
- [PPO training](../training/ppo)
- [GRPO training](../training/grpo)
