# Checkpoints and metrics

## Checkpoints

Checkpoints are optional. Add them to make a run resumable or to keep its best
evaluation.

```toml
[checkpoint]
directory = "artifacts/checkpoints"
mode = "steps"
every_steps = 100
```

Each checkpoint is a `.state` directory holding the trained weights (the
adapter, the base tensors, or both), the optimizer state and the run's
progress. `steps` writes `step-<N>.state` every `every_steps` optimizer steps.
A LoRA run also writes a `.gguf` copy of the adapter next to each checkpoint.

The final result always goes to `[output].path`, including when the run stops
early.

### Keep the best evaluation

```toml
[evaluation]
data = "eval.jsonl"

[checkpoint]
directory = "artifacts/checkpoints"
mode = "steps_and_best_eval"
every_steps = 100
```

`best_eval` keeps `best.state`, updated each time the evaluation improves;
`steps_and_best_eval` does both. SFT keeps the lowest loss, PPO and GRPO the
highest mean reward. Evaluation also runs at the last iteration.

### Resume

```bash
retrograd train run.toml --resume
retrograd train run.toml --resume artifacts/checkpoints/step-000000000100.state
```

Without a path, `--resume` picks the latest `step-*.state`, or `best.state`.
The run continues exactly where it stopped: weights, optimizer state, learning
rate schedule, random generators and data position are restored.

A checkpoint is refused if the model, data, algorithm or training settings no
longer match. To start from an existing adapter without its optimizer state,
use `lora.init_adapter` instead.

## Metrics

Progress, loss and throughput are printed to the terminal. To keep them:

```toml
[metrics]
tensorboard_dir = "artifacts/tensorboard"
wandb_export_dir = "artifacts/wandb"
```

`tensorboard_dir` gets one subdirectory per run. `wandb_export_dir` writes an
offline export (no network or account during training); upload it afterwards
with:

```bash
WANDB_API_KEY=... python3 scripts/import_wandb.py artifacts/wandb --project my-project
```

Metric names are grouped by prefix: `train/`, `eval/`, `reward/`, `policy/`,
`completions/`, `batch/`, `optimizer/`, `system/` (including memory use) and,
per algorithm, `distill/`, `agent/`, `judge/`, `env/`.
