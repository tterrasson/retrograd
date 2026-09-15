# SFT training

Supervised fine-tuning trains the LoRA adapter against known assistant
responses. It is the direct choice when the dataset contains the behavior the
model should reproduce.

## Minimal configuration

```toml
[run]
algorithm = "sft"

[model]
path = "base.gguf"
device = "auto"

[lora]
output = "sft-adapter.gguf"
rank = 8
alpha = 16.0
seed = 42

[training]
ctx = 512
micro_batch = 32
gradient_accumulation = 1
epochs = 3
lr = 0.0001
lr_scheduler = "cosine"
warmup_steps = 10
weight_decay = 0.0
max_grad_norm = 1.0

[sft]
data = "train.jsonl"
data_format = "jsonl"
shuffle = true
```

Run it with:

```bash
retrograd train sft.toml
```

`training.epochs` is the number of full passes over the SFT dataset. The
default is `1`. Rows are shuffled at the beginning of each epoch by default;
the permutation is derived from `lora.seed`.

## What is trained

For chat JSONL, Retrograd formats each conversation using the GGUF chat
template and masks every token except assistant responses. A record can have
multiple assistant turns; each assistant span is a target. For plain text, the
file is treated as a continuous token stream and all next-token labels are
active.

The base model weights remain frozen. The adapter is initialized from the LoRA
settings and written as a standalone GGUF file after training.

## Context and batch geometry

`training.ctx` is the maximum token window. `micro_batch` is the physical number
of tokens processed by one forward/backward pass and is the primary activation
memory control. `gradient_accumulation` combines micro-batches before one
optimizer step.

The product must divide `ctx`:

```text
micro_batch × gradient_accumulation divides ctx
```

If a run does not fit in memory, lower `micro_batch` first. Keep the effective
optimizer window stable by increasing `gradient_accumulation` when appropriate.

## Sharing the GPU

`training.max_gpu_duty_cycle` bounds the fraction of wall time the trainer
spends waiting on GPU work it submitted, leaving the rest to another workload:

```toml
[training]
max_gpu_duty_cycle = 0.5
```

Both loops of an epoch are covered - the training passes and the evaluation
split - so a large held-out set does not escape the limit at the epoch
boundary.

It frees **compute time, not device memory**: everything the run has allocated
stays allocated while it sleeps. Enabling it also costs the decode pipelining
once, before any sleep, so it is worth its overhead at `0.75` and below. See
[Configuration reference](../reference/configuration) for the full contract.

## Evaluation and output

Add a held-out set to measure loss during training:

```toml
[evaluation]
data = "eval.jsonl"
every_iterations = 1
patience = 3
min_delta = 0.0
```

For SFT, one evaluation iteration is one epoch. `patience` counts consecutive
evaluations without an improvement of at least `min_delta`.

Use `bench` after training to compare an adapter on a fixed dataset, or `chat`
to inspect responses interactively. See [Checkpoints and monitoring](../operations/checkpoints)
for saving the best evaluation result and resuming a run.

## SFT parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `sft.data` | required | Text or chat JSONL training file. |
| `sft.data_format` | inferred | `text`/`txt` or `jsonl`/`chat`/`chat-jsonl`. |
| `sft.shuffle` | `true` | Shuffle training rows between epochs using `lora.seed`. |
| `training.epochs` | `1` | Complete passes over the dataset. |
| `training.ctx` | `128` | Training context length in tokens. |
| `training.micro_batch` | `32` | Physical forward/backward width. |
| `training.gradient_accumulation` | `1` | Micro-batches per optimizer step. |
| `training.lr` | `0.0001` | AdamW learning rate. |
| `training.lr_scheduler` | `constant` | `constant`, `linear`, or `cosine`. |
| `training.warmup_steps` | `0` | Scheduler warmup steps. |
| `training.weight_decay` | `0.0` | AdamW weight decay. |
| `training.max_grad_norm` | `1.0` | Global gradient norm limit. |
