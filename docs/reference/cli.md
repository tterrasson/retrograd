# CLI reference

`cargo build --release` writes the `retrograd` binary to `target/release/`.
`retrograd --help` lists every command and flag.

## Common flags

| Flag | Meaning |
| --- | --- |
| `--model PATH` | Use this GGUF instead of `[model].path`. |
| `--device auto` | Use a GPU if one is available, else the CPU. |
| `--device cpu` | Force the CPU. |
| `--device gpu` | Require a GPU; fail otherwise. |

Paths on the command line resolve against the current directory; paths in the
TOML file resolve against the file's directory.

## `train`

```text
retrograd train CONFIG.toml [--resume [CHECKPOINT.state]] [--model M] [--device D]
```

Runs the algorithm in `[run].algorithm`. `--resume` continues from the given
checkpoint, or from the latest one in `[checkpoint].directory`. See
[Checkpoints](../operations/checkpoints#resume).

## `bench`

```text
retrograd bench CONFIG.toml [--data FILE] [--adapter ADAPTER.gguf] [--limit N]
                [--format auto|text|jsonl] [--ctx N] [--model M] [--device D]
```

Evaluates without training. Without `--adapter` it measures the base model;
with it, the base model and the adapter on the same examples. `--data`
defaults to `[evaluation].data`.

| Algorithm | Reports |
| --- | --- |
| `sft` | Loss and perplexity on assistant tokens. |
| `ppo`, `grpo` | Reward distribution of generated answers. |
| `distill` | KL to the teacher, top-1 agreement, perplexity. Loads the teacher. |
| `agent_grpo` | Not supported. |

## `chat`

```text
retrograd chat CONFIG.toml [--adapter ADAPTER.gguf] [--compare] [--base-only]
               [--system PROMPT] [--temp T] [--top-p P]
               [--max-new-tokens N] [--seed N] [--ctx N] [--model M] [--device D]
```

Interactive chat using the model's chat template. `--compare` shows the base
model's and the adapter's answer to each turn; `--base-only` disables the
adapter. Defaults: temperature `0.7`, top-p `0.95`, `512` new tokens. Type
`/reset` to clear the history and `/exit` to quit.

## `inspect`

```text
retrograd inspect --model MODEL.gguf [--device D]
```

Prints the model's architecture, tensor types, suggested LoRA targets and the
device that will be used.

## `preflight`

```text
retrograd preflight --model MODEL.gguf [--targets q,k,v] [--strict] [--device D]
```

Builds the training graph without running it and lists the operations the
device cannot run (they fall back to the CPU and slow every step).
`--strict` exits with an error if there are any. The same check runs
automatically at the start of every training run.

## `distill-teacher`

```text
retrograd distill-teacher CONFIG.toml [--data FILE] [--out FILE.topk] [--k N]
                          [--ctx N] [--model M] [--device D]
```

Writes the teacher's top-k predictions for
[offline distillation](../training/distill#offline-top-k).

## Agentic commands

```text
retrograd tools list CONFIG.toml [--json] [--no-connect]
retrograd scenarios generate CONFIG.toml [--dry-run] [--force]
retrograd judge eval CONFIG.toml --fixtures FILE.jsonl
```

See [Agentic GRPO](../training/agent#tooling).

## `profile`

A separate binary that runs a short GRPO or agentic GRPO workload and reports
time and memory per phase. See [Performance and memory](../operations/performance#profiling).

## Environment variables

| Variable | Effect |
| --- | --- |
| `RETRO_THREADS` | Overrides `training.threads`. |
| `RETRO_REQUIRE_GPU_RESIDENT=1` | Same as `training.require_gpu_resident = true`. |
| `RETRO_DEVICE_SAMPLING=0` | Sample on the host instead of the GPU. |
| `RETRO_DEVICE_LOGPROBS=0` | Compute sampling log-probabilities on the host. |
| `RETRO_FAST_TOP_P=1` | Faster top-p selection; ties may resolve differently. |
| `RUST_LOG` | Log level (default `warn`). |

Build-time variables are listed in [Build variants](./builds).
