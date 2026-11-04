# Decision models

A decision model scores options for a typed question about a state. It supports
a `choice` among named options, a `score` on a scale of 2 to 10 levels, and a
yes/no `noul`. Native decision models are served on
[`POST /v1/systemone`](../running/api#decisions). The `decision` algorithm
fine-tunes supported models on labelled questions and can fit and export
temperatures using held-out questions.

Models whose `<arch>.decision.type` is `laya` train here (Julia, Laya, and
other ModernBERT decision heads): they read the score of an option at a
marker token of the embeddings output. Causal LLMs and `openjev` models train
on letter logits, `lev` models on their code labels, `kev` models on the
pointer product of their hidden states. See the
[API reference](../running/api#decisions) for the other readouts and their
serving constraints. Training rejects unsupported readouts before loading
the dataset.

## Smoke test

From the repository root, generate the tiny `laya` fixture and submit the
example request on CPU.

```bash
scripts/fetch-cpu-fixture.sh TINY_DECISION_FIXTURE
cargo run --release -q -- decide \
  --model tests/fixtures/retrograd-tiny-decision-f32.gguf \
  --request examples/smoke_decision_request.json --device cpu
```

The JSON response contains a `choice`, a `noul` and a `score`. Train a rank-1
LoRA adapter on the same fixture and then try the request with it.

```bash
cargo run --release -q -- train examples/smoke_tiny_decision.toml
cargo run --release -q -- decide \
  --model tests/fixtures/retrograd-tiny-decision-f32.gguf \
  --adapter /tmp/retrograd-smoke-decision-adapter.gguf \
  --request examples/smoke_decision_request.json --device cpu
```

The training configuration reads `examples/smoke_decision.jsonl` and fits
temperatures on `examples/smoke_decision_eval.jsonl`. The generated weights
and small datasets are intended for checking the workflow.

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

Training uses the configured device, including Metal. `auto` follows the
ordinary backend selection.

## Causal LLMs

Use the same `decision` algorithm with a causal GGUF that has no decision
metadata. Retrograd builds a generic question with labels `A..Z`, then `a..z`,
wraps it in the model's chat template as a user message with a generation
prompt, and trains the next letter. Every label must be one token; at most
52 options are accepted. `decision.max_head_tokens` applies only to marker
heads; a causal question must fit the micro-batch without truncation.

```toml
[model]
path = "Qwen3.5-0.8B-BF16.gguf"

[decision]
data = "train.jsonl"
template_variables = { enable_thinking = false }
export_base = "decision-base.gguf"
```

Use the causal model's usual LoRA targets (`q`, `k`, `v`, `o`, `ffn_up`,
`ffn_down` as available). Hard answers supervise one letter token at the last
prompt position. Soft answers use sparse cross-entropy over the full
vocabulary, with the teacher's probabilities on the option letters; prompt
and padding positions are masked. This loss also penalizes probability mass
outside the labels. Evaluation and calibration use the served softmax over
labels. A loss restricted to labels remains a future option.

Dense causal models pack questions. Recurrent/hybrid models use one question
per micro-batch, preserving their equal-sequence graph, and accumulate the
configured number of micro-batches into an update.

The exported base/model declares `<arch>.decision.type = "openjev"` and carries
one standalone `systemone` template, with the chat wrapper and its variables
frozen into it. No nested chat rendering is required by llama-server.
Templates that alter or repeat user content cannot be flattened and are
refused. An adapter export from a plain causal base requires `export_base`;
serve that copy with the adapter in either Retrograd or llama.cpp.

## Pointer and code-label heads (`kev`, `lev`)

A `kev` model reads its answer from a `[q|k]` projection of its hidden
states (`cls.output`): the score of an option is the product of the first
half of the output at the last prompt token with the second half of the
output at the token that closes the option, over `sqrt(n_embd_out / 2)`. A
run trains the softmax cross-entropy of those products, the grouped loss of
the marker heads with a query token per question. The model keeps its
language-model head; the run trains through the projection, never the
logits. A recurrent or hybrid trunk takes one question per micro-batch.

```toml
[model]
path = "Kev-4B-Q4_K_M.gguf"

[lora]
rank = 8
alpha = 16
targets = ["ffn_up", "ffn_down"]

[decision]
data = "train.jsonl"
questions_per_batch = 1
```

A `lev` model trains on its code labels like a causal letter readout, with
two differences that follow what it serves. A `choice` is rendered twice,
the second time with the options reversed, and both renderings train, each
with its targets in its own order. A `noul` is read on a scale of 9 ratings
whose expectation is the answer: it trains the two ends of the scale, the
probability of `false` on the lowest rating and of `true` on the highest.
Evaluation and calibration treat each rendering as one question; the served
answer averages the two orders of a choice.

## Teacher labels

```bash
retrograd decide label --teacher Laya-Q8_0.gguf --data questions.jsonl --out labelled.jsonl
retrograd decide label --teacher decision-base.gguf --adapter teacher.gguf \
  --data questions.jsonl --out labelled.jsonl
retrograd decide label --teacher http://localhost:8000/v1 --teacher-model judge \
  --api-key-env TEACHER_API_KEY --data questions.jsonl --out labelled.jsonl
```

Inputs are native single questions or service requests (`questions`), without
needing answers. Local teachers provide their served probabilities; a plain
causal teacher uses the generic letter prompt. `--template-variables JSON`
controls its chat wrapper (default `{"enable_thinking":false}`). HTTP teachers
are asked for a probability object and their reply must cover every option
with finite nonnegative mass summing to one. Requests have a timeout and a
response-size bound through the shared LLM client. All replies are validated
before the output is created; an existing output file is refused.

The output is native decision JSONL with `probabilities`, suitable for a
marker-readout student or a causal student. Calibration still needs a separate
held-out set. Remote probability reports are the judge's estimates, rather
than access to its token logits.

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

Every question uses the effective `systemone` template and the layout for its
readout. The marker heads apply their head budget before the micro-batch token
limit is checked. A rendered and laid-out question that still exceeds
`training.micro_batch` is refused.

## How a step works

A micro-batch holds up to `questions_per_batch` questions and
`training.micro_batch` tokens, with one sequence per question. Recurrent and
hybrid trunks process one question per micro-batch. Marker and pointer heads
use softmax cross-entropy over option scores at temperature 1. Letter and code
readouts train on label tokens. `training.gradient_accumulation`
micro-batches make one optimizer update.

The options of every `choice` question are shuffled at each epoch
(`option_shuffle`), so the model cannot learn where the right one usually is;
score levels and noul are ordered and stay as they are.

## Evaluation and temperatures

`[evaluation].data` is evaluated at temperature 1 on the epoch schedule set
by `evaluation.every_iterations` (every epoch by default) and at the final
epoch. The report includes NLL, accuracy, Brier score and expected
calibration error, overall and per type. Early stopping compares NLL.

When `decision.calibrate` is enabled, as it is by default with
`[evaluation]`, the run fits one temperature per type and option-count
bucket (`choice.3_5`, for example) and one per type to minimize held-out NLL.
Temperature scaling changes the spread of each distribution and the
resulting `confidence`. Aggregate answers such as expected scores can also
change.

## Exports

| `output.kind` | Temperatures |
| --- | --- |
| `adapter` | `adapter.decision.temperature.*` in the adapter. Retrograd serves the adapter with them; llama.cpp ignores them and keeps the base's. |
| `model` (`training.trainable = "full"` or `"partial"`) | `<arch>.decision.temperature.*` in the model, which `llama-server` serves as is. |
| `trainable` | The bundle carries no temperatures. When the run also writes a sibling LoRA adapter, the adapter carries them. |

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
For coverage, the benchmark computes confidence from the returned
probabilities using the standard choice and score formulas. A head with its
own confidence formula can report a different value on the API. The adapter
is measured with its own temperatures.
