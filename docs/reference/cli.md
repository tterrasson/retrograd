# CLI reference

The binary is built as `target/release/retrograd` from the repository root.

## Commands

### `train`

```text
retrograd train CONFIG.toml [--resume [checkpoint.state]]
    [--model MODEL.gguf] [--device auto|cpu|gpu]
```

Runs the algorithm selected by `[run].algorithm`. `--model` and `--device`
override the matching `[model]` values.

`--resume` restarts a stopped run from a checkpoint: the `.state` directory
given, or, with no path, the highest-numbered `step-*.state` under
`[checkpoint].directory`, falling back to `best.state`. It needs a
`[checkpoint]` section, and it is refused alongside `lora.init_adapter` or a
`checkpoint.resume_from` already set in the document. See
[Checkpoints and monitoring](../operations/checkpoints#resume).

### `bench`

```text
retrograd bench CONFIG.toml [--data DATA] [--model MODEL.gguf]
    [--adapter ADAPTER.gguf] [--device auto|cpu|gpu]
    [--format auto|text|jsonl] [--ctx N] [--limit N]
```

Runs evaluation without updating the adapter. `--data`, `--adapter`, and
`--model` override the corresponding configured paths; the dataset defaults to
`[evaluation].data`, and `--limit N` caps the examples. `--format` is useful for
files whose extension is not `.txt`, `.md`, `.json`, or `.jsonl`.

Without `--adapter` it scores the base model only; with it, the base model and
the adapter on the same examples and seeds.

What is measured follows `[run].algorithm`: the teacher-forced loss and
perplexity on the assistant targets for `sft`; for `ppo` and `grpo`, completions
generated with the configured sampling and the distribution of their rewards
(mean, low and high, p10/p50/p90, standard deviation); and for `distill` the three figures of a
distillation report - the KL to the teacher, top-1 agreement with it, and the
teacher-forced perplexity of the reference answers when the held-out lines carry
any. `agent_grpo` is refused: an agentic score needs a live environment, which
is a run and not a measurement pass. A `distill` bench loads the teacher, so it
needs room for both models.

### `chat`

```text
retrograd chat CONFIG.toml [--compare] [--base-only]
    [--adapter ADAPTER.gguf] [--model MODEL.gguf]
    [--device auto|cpu|gpu] [--ctx N] [--system PROMPT]
    [--temperature T] [--top-p P] [--max-new-tokens N] [--seed N]
```

Starts an interactive chat session. It reads one turn per line from stdin,
threads the conversation through the model's chat template, and prints the
reply. `--base-only` disables the adapter. `--compare` answers each turn from
both the base model and the adapter with the same sampling seed, and carries the
adapter's answer forward as the shared history. `--system PROMPT` seeds a system
message.

Sampling defaults to temperature `0.7`, top-p `0.95`, `512` new tokens and seed
`0`. `/reset` clears the history; `/exit`, `/quit` or Ctrl-D quits. The
spellings `--temp`, `--top_p`, `--base_only`, `--max_new_tokens`, `--max_tokens`
and `--max-tokens` are accepted as well.

### `inspect`

```text
retrograd inspect --model MODEL.gguf [--device auto|cpu|gpu]
```

Loads a model and prints its capability report without training: architecture,
tensor types, the automatic LoRA target profile, candidate target patterns and
the effective device.

### `distill-teacher`

```text
retrograd distill-teacher CONFIG.toml [--data DATA] [--out SIDECAR]
                          [--k N] [--model MODEL.gguf]
                          [--device auto|cpu|gpu] [--ctx N]
```

Writes the top-k sidecar an offline distillation run reads: for every position
of the corpus, the teacher's `k` most probable tokens and their
log-probabilities. Defaults come from `[distill]` - `teacher_path`, `data` and
`sidecar` - and the flags override them. `--k` defaults to 16.

The corpus is tokenized through the same path the trainer will use, so a
sidecar cannot be misaligned with the batch it will be trained against. The
command loads one model at a time (the student to prepare the corpus, then the
teacher to score it), so it runs where the pair would not fit.

Producing a sidecar is a one-off: the target does not move between epochs, and
a run that resumes checks the sidecar's header rather than recomputing it.

### `profile`

`profile` is a separate binary, not a subcommand of `retrograd`:

```text
cargo run --release --bin profile -- [CONFIG] [OPTIONS]
```

It runs a short end-to-end GRPO or `agent_grpo` workload and reports phase
timings, host memory, device memory, and backend allocations. It does not
currently profile SFT or PPO.

The default configuration is `examples/smoke_tiny_grpo.toml`. Without `--full`,
the profiler applies a short workload so that a measurement can complete
quickly:

| Option | Fast default | Description |
| --- | ---: | --- |
| `--model PATH` | from config | Override the GGUF model path. |
| `--device auto\|cpu\|gpu` | from config | Override the training device. |
| `--updates N` | `1` | Number of updates. |
| `--group-size N` | `4` | GRPO group size. |
| `--epochs N` | `1` | GRPO epochs per update. |
| `--prompts N` | `3` | Prompts or scenarios per update. |
| `--micro-batch N` | from config | Override `training.micro_batch`; it must divide the optimizer window. |
| `--max-new-tokens N` | from config | Override the sampling budget. |
| `--full` | off | Use the configuration values without fast workload overrides. |

Examples:

```bash
# Short GRPO profile using a different model and GPU.
cargo run --release --bin profile -- examples/smoke_tiny_grpo.toml \
  --model /models/base.gguf --device gpu

# Measure the configured workload as written.
cargo run --release --bin profile -- examples/smoke_tiny_grpo.toml --full
```

Run from the repository root so relative reward commands, datasets, and
agentic scenario files resolve correctly.

### `preflight`

```text
retrograd preflight --model MODEL.gguf [--device auto|cpu|gpu]
    [--targets q,k,v] [--strict]
```

Builds the exact training graph - forward, backward and optimizer step - without
running it, and reports, per backend device, every op the device cannot execute
and every op with no gradient rule. The report ends with
`active_device_fallback_nodes`: each fallback node costs a device and host round
trip on every step, and `--strict` turns a non-zero count into a non-zero exit.
When the cause is a quantization type the device cannot decode, an
`undecodable_types` line names the types and what to requantize to.

The same preflight runs automatically before the first optimizer step of every
run.

When compiled with the agent feature, the binary also provides `judge eval`,
`tools list`, and `scenarios generate`. These commands are for agentic GRPO and
are not required by SFT, PPO, or single-turn GRPO.

## Model and device overrides

The model and device flags have the same meaning across the binaries that
accept them:

| Flag | Meaning |
| --- | --- |
| `--model PATH` | Use `PATH` as the base GGUF model for this invocation, overriding `[model].path` where a configuration is used. |
| `--device auto` | Use a compiled GPU backend when one is available; otherwise use the CPU. |
| `--device cpu` | Force CPU execution. |
| `--device gpu` | Require a compiled and available GPU backend; fail instead of falling back to CPU. |

`--model` paths supplied on the command line resolve against the shell's working
directory. Paths written in TOML resolve against the TOML file's directory.
Device values select `auto`, `cpu`, or `gpu`; use `RETRO_BACKENDS` at build time
to choose which GPU backend is compiled into the binary.

## Runtime environment variables

The most relevant runtime controls are:

| Variable | Effect |
| --- | --- |
| `RETRO_THREADS` | Overrides `training.threads`. |
| `RETRO_FAST_TOP_P=1` | Uses heap selection for the top-p tail; seeded ties may differ. |
| `RETRO_DEVICE_SAMPLING=0` | Disables device-side sampling and uses the host path. |
| `RETRO_DEVICE_LOGPROBS=0` | Disables device-side behavior-logprob gathering. |
| `RETRO_REQUIRE_GPU_RESIDENT=1` | Requires training graph work to stay on the selected GPU backend. |
| `RETRO_UBATCH_FINITE_CHECK=0` | Skips the load-time check that the requested micro-batch produces finite logits on the selected GPU, and with it the escalation to the full logical batch. |
| `RETRO_PACKED_SEQ_PROBE=0` | Trusts the model's packed multi-sequence declaration instead of verifying it on this device. |
| `RETRO_RECURRENT_ROLLBACK=auto` | Applies the derived recurrent-state rollback depth instead of leaving it at zero. Off by default; the derivation is reported either way. |
| `RETRO_RECURRENT_ROLLBACK_BUDGET_MB` | Byte budget bounding that depth (default 64). |

`RETRO_BACKENDS` selects compiled backends when Cargo builds the binary; it has
no effect when set only while launching an existing binary. `RETRO_NATIVE=1`
is also a build input and enables host-specific CPU kernels. See [Build
variants](./builds) for backend build commands. A compiled GPU backend must
still be available on the machine at runtime.
