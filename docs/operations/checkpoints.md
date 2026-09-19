# Checkpoints and monitoring

Checkpoints are optional. Enable them when a run should be resumable or when
the best evaluation result should be preserved.

## Step checkpoints

```toml
[checkpoint]
directory = "artifacts/checkpoints"
mode = "steps"
every_steps = 100
```

Each checkpoint is a state directory containing whatever the run produced, the
optimizer state, progress, dataset identity, and the run metadata needed for
compatibility checks. The loader refuses a checkpoint when the model, dataset,
schedule, optimizer settings, trainable policy, resolved trainable set, or
algorithm no longer match the saved run.

What the directory holds follows what the run trains. A `lora` run writes
`adapter.gguf`; a `full` or `partial` run writes `trainable.gguf`, the trained
base tensors by absolute value rather than as a delta; a `hybrid` run writes
both. Optimizer state is kept out of those files entirely: it lives in
`optimizer-state.bin` as the concatenated payloads of the slots the manifest
lists, so saving and restoring a multi-gigabyte state streams through a bounded
buffer instead of a host copy of it.

`steps` writes `step-XXXXXXXXXXXX.state` (the zero-padded optimizer step),
`best_eval` maintains `best.state`, and `steps_and_best_eval` does both. The
`.state` directory is the authoritative result plus resume state, written as one
atomic unit; the sibling `.gguf` is a cold-load export of the same adapter, and
is published only for a run that has one - the name every helper reads as "the
adapter" must not resolve to a file no adapter loader accepts. `[output].path`
always receives the final result, including after early stopping.

## Best evaluation checkpoints

```toml
[evaluation]
data = "eval.jsonl"
every_iterations = 1

[checkpoint]
directory = "artifacts/checkpoints"
mode = "steps_and_best_eval"
every_steps = 100
```

`best_eval` and `steps_and_best_eval` require `[evaluation]`. For SFT, an
iteration is one epoch. For PPO and GRPO, it is one rollout update.

SFT evaluates token-level loss and perplexity, where lower is better. PPO and
GRPO generate one completion per held-out prompt, score it with the reward
command, and maximize the mean reward; their evaluation seeds are fixed across
evaluations. Evaluation also runs at the final iteration when it does not land
on the interval.

A rollout evaluation costs one full generation per prompt. `max_examples` caps
it, taking prompts evenly spaced across the file; size
`every_iterations × max_examples` so evaluation stays a fraction of the
completions each update samples anyway. `patience` counts consecutive
evaluations that fail to improve by `min_delta`; reaching it stops the loop
after that epoch or update.

## Resume

Resume from the most recent checkpoint of the configured directory:

```bash
retrograd train run.toml --resume
```

Or provide an explicit state directory:

```bash
retrograd train run.toml --resume artifacts/checkpoints/step-000000000100.state
```

Without a path, `--resume` picks the highest-numbered `step-*.state` under
`[checkpoint].directory`, falling back to `best.state`. It needs a
`[checkpoint]` section, and it is refused alongside a `checkpoint.resume_from`
already written in the document, which is the equivalent without the flag
(resolved relative to the TOML's directory).

Resuming restores the full trainer state, not just the weights: the adapter,
AdamW moments, the scheduler step, both RNGs (the runtime sampler and ggml's
own), the shuffle cursor, `global_step`, the learning-rate schedule, the
checkpoint cadence, and the evaluation counters. A checkpoint written before the
first optimizer step has no moments to restore, and the optimizer starts cold.

A plain adapter GGUF loaded with `lora.init_adapter` is a cold adapter start: it
does not contain optimizer state and is not a full resume. These two mechanisms
are mutually exclusive.

## Metrics

```toml
[metrics]
tensorboard_dir = "artifacts/tensorboard"
wandb_export_dir = "artifacts/wandb"
```

SFT reports training and evaluation loss. PPO reports reward, surrogate loss,
KL, clipping, learning rate, and critic metrics when enabled. GRPO reports
reward, group statistics, clipping, KL, packing, and generation metrics.

Metric names are grouped by type: `reward/`, `policy/`, `completions/`,
`batch/`, `eval/`, `optimizer/`, `system/` and `data/`. The trainer also tracks
the process memory footprint per startup phase and during training, emits
`system/memory_*` metrics and prints a per-component summary at the end; on
Apple Silicon the footprint includes Metal allocations. On GPU backends the
backend report also carries `device_peak_used_bytes` and
`backend_scratch_peak_bytes`, sampled inside the training step.

`tensorboard_dir` is a TensorBoard log root: each run gets a timestamped
subdirectory. `wandb_export_dir` is offline - it writes `run.json` and
`metrics.jsonl` with no Python, network or credentials during training. Import
it afterwards with the official SDK:

```bash
WANDB_API_KEY=... python3 scripts/import_wandb.py artifacts/wandb --project my-project [--entity my-team]
```

For a stalled GRPO run, inspect reward variation first. A group in which every
completion receives the same reward has no group-relative learning signal.
`grpo.max_stalled_updates` controls how many consecutive zero-signal updates are
tolerated; set it to `0` to disable the stop condition.

## Memory controls

Use these controls in this order when a run does not fit:

1. Lower `training.micro_batch`.
2. Lower `training.generation_concurrency` for GRPO generation memory.
3. Keep `training.fast_sampling_context = true` unless exact sampling context
   behavior is required.
4. Enable `training.gradient_checkpointing` for packed rollout backward passes.

`ctx`, `group_size`, and `prompts_per_update` change the training problem, so
do not change them only to solve a memory limit without reviewing the resulting
optimization geometry.
