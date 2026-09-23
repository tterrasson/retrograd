# SFT

Supervised fine-tuning trains the model to reproduce known answers. Use it when
your dataset already contains the responses you want.

## Configuration

```toml
[run]
algorithm = "sft"

[model]
path = "base.gguf"

[output]
path = "sft-adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 512
epochs = 3
lr = 0.0001
lr_scheduler = "cosine"
warmup_steps = 10

[sft]
data = "train.jsonl"
```

```bash
retrograd train sft.toml
```

With chat JSONL, only assistant turns are trained. With plain text, every token
is. Rows are shuffled at each epoch (`sft.shuffle = false` keeps file order).
See [Datasets](../getting-started/datasets).

## Batch size and memory

| Key | Default | Meaning |
| --- | ---: | --- |
| `training.ctx` | `128` | Longest sequence, in tokens. |
| `training.micro_batch` | `32` | Tokens per forward/backward pass. The main memory knob. |
| `training.gradient_accumulation` | `1` | Passes accumulated per optimizer step. |

`micro_batch × gradient_accumulation` must divide `ctx`. If the run runs out
of memory, lower `micro_batch` and raise `gradient_accumulation` to keep the
same step size. More options are in [Performance and memory](../operations/performance).

## Evaluation

```toml
[evaluation]
data = "eval.jsonl"
patience = 3
```

Evaluation runs after every epoch and reports held-out loss and perplexity.
`patience` stops the run after that many evaluations without improvement.
After training, compare the base model and the adapter on the same data:

```bash
retrograd bench sft.toml --data eval.jsonl --adapter sft-adapter.gguf
```

## Training base weights

By default only a LoRA adapter is trained. `training.trainable` trains the
model's own weights instead, and works the same way for every algorithm:

| `trainable` | Trains | Needs |
| --- | --- | --- |
| `lora` (default) | A LoRA adapter | `[lora]` |
| `full` | Every trainable base tensor | nothing else |
| `partial` | The base tensors selected in `[trainable]` | `[trainable]` |
| `hybrid` | An adapter plus norms and biases | `[lora]` and `[trainable]` |

```toml
[training]
trainable = "partial"

[trainable]
layers = "last:4"
modules = ["attn", "ffn"]
norms = true

[output]
path = "tuned.gguf"
kind = "model"   # a standalone GGUF, usable without Retrograd
```

Quantized tensors are never trained, so the selected tensors must be stored in
F32, F16 or BF16; which of these each backend accepts is in the
[support matrix](../engineering/SUPPORT#base-weight-storage-precision). See
[`[trainable]`](../reference/configuration#trainable) and
[`[output]`](../reference/configuration#output) for every option.
