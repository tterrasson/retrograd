# Decision models

A decision model reads a state and a typed question and scores the options in
one forward pass: a `choice` among named options, a `score` on a scale of 2 to
10 levels, a yes/no `noul`. It is served on [`POST /v1/systemone`](../running/api#decisions).
`run.algorithm = "decision"` fine-tunes one on labelled questions, fits its
temperatures on held-out questions, and writes them into what it exports.

Models whose `<arch>.decision.type` is `laya` train here (Julia, Laya, and
other ModernBERT decision heads): they read the score of an option at a
marker token of the embeddings output. The others are served, not trained.

## Configuration

```toml
[run]
algorithm = "decision"

[model]
path = "Julia-1-Q8_0.gguf"

[output]
path = "routing-adapter.gguf"

[lora]
rank = 16
alpha = 32
targets = ["qkv", "o", "ffn_up", "ffn_down"]
dtype = "f32"

[training]
ctx = 512
micro_batch = 512
gradient_accumulation = 4
epochs = 2
lr = 5e-4
warmup_steps = 10
lr_scheduler = "cosine"

[decision]
data = "train.jsonl"
questions_per_batch = 8
export_base = "routing-base.gguf"   # optional, see Exports

[evaluation]
data = "eval.jsonl"
```

```bash
retrograd train decision.toml
```

A ModernBERT block has a fused attention projection: target it as `qkv`
(`q`, `k` and `v` match nothing there). The scorer can be targeted by name,
`cls.weight`.

The training runs on the CPU: the GPU backend computes wrong gradients for an
encoder's training graph today. `device = "auto"` trains on the CPU and says
so; `device = "gpu"` is refused. Serving and `bench` run on the GPU as usual.

## Data

One question per line, written as the service receives it, with its answer:

```jsonl
{"state": {"message": "I was billed twice"}, "question": {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "Invoices", "shipping": null}}, "answer": "billing"}
{"state": "The parcel never arrived.", "question": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}, "answer": 2}
{"state": "Please refund me.", "question": {"type": "noul", "instructions": "Does the customer ask for a refund?"}, "answer": true}
```

- `answer` is the key of the option (`choice`), the index of the level
  (`score`) or a boolean (`noul`).
- `probabilities` replaces `answer` with a target distribution - distillation
  from a teacher, or label smoothing over the levels of a score - keyed by
  option key (`"0"`, `"1"`, ... for levels, `"false"` and `"true"` for a noul).
- `"questions": {id: {...}}` asks several questions of one state; each carries
  its own `answer` or `probabilities` and becomes one example.
- `data_format = "julia"` reads the format of the original Julia training code:
  `state`, `question`, `options`, `type`, `target`, optionally
  `teacher_logits`.

Every question is rendered through the model's `systemone` template and laid
out by its head budget exactly as the server does, so what is trained is what
is served. A question longer than `training.micro_batch` tokens is refused.

## How a step works

A micro-batch holds up to `questions_per_batch` questions and
`training.micro_batch` tokens, each question its own sequence. The loss is a
softmax cross-entropy over the scores the head reads at the option markers of
each question, at temperature 1. `training.gradient_accumulation`
micro-batches make one optimizer update.

The options of every `choice` question are shuffled at each epoch
(`option_shuffle`), so the model cannot learn where the right one usually is;
score levels and noul are ordered and stay as they are.

## Evaluation and temperatures

`[evaluation].data` is evaluated after each epoch at temperature 1: NLL (what
early stopping compares), accuracy, Brier score and expected calibration
error, overall and per type.

At the end, the run fits one temperature per type and option-count bucket
(`choice.3_5`, ...) and one per type, minimizing the held-out NLL. A
temperature never changes the winning option; it sets the spread of the
probabilities and with it the `confidence` a caller thresholds on.

## Exports

| `output.kind` | Temperatures |
| --- | --- |
| `adapter` | `adapter.decision.temperature.*` in the adapter. Retrograd serves the adapter with them; llama.cpp ignores them and keeps the base's. |
| `model` (`training.trainable = "full"` or `"partial"`) | `<arch>.decision.temperature.*` in the model, which `llama-server` serves as is. |
| `trainable` | None: the bundle carries no metadata. |

`decision.export_base` writes a copy of the base model carrying the fitted
temperatures - and the template and head budget of the run when they were
overridden - with its tensors untouched. That is what lets
`llama-server -m routing-base.gguf --lora routing-adapter.gguf` serve the
adapter with its own temperatures. The run says so when it exports an adapter
whose temperatures differ from the base's and no such copy was asked for.

## Measuring

`retrograd bench decision.toml [--adapter routing-adapter.gguf]` asks every
question of `[evaluation].data` through the engine, as the server would, and
reports accuracy, NLL, Brier and calibration error overall and per type, and
the share of questions kept at each confidence threshold with their accuracy.
The adapter is measured with its own temperatures.
