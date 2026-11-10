# Running models

Retrograd serves GGUF models through an
[OpenAI-compatible API](./api). The model is loaded as is. A LoRA adapter trained
by Retrograd, or by another tool that writes the llama.cpp format, is served
next to its base model.

## Which binary

| You want to | Use | Default address |
| --- | --- | --- |
| Serve a model and its adapters | `retrograd-infer serve.toml` | `127.0.0.1:8000` |
| Serve what a training configuration produced | `retrograd serve run.toml` | `127.0.0.1:8000` |
| Train over HTTP, and serve the runs | [`retrograd-server`](./server) | `127.0.0.1:8471` |

All three answer the same OpenAI routes with the same code. `retrograd-infer`
is the smallest: its dependency tree carries no training, agent or
run-configuration crate.

## Build

Build the binaries you need from the repository root:

```bash
cargo build --release -p retrograd -p retrograd-server -p retrograd-serve
```

`retrograd-serve` is the package; its binary is
`target/release/retrograd-infer`. On its own, `-p retrograd-infer` builds a
library and no binary.

::: warning GPU backends
Only the root package enables a GPU backend. `cargo build -p retrograd-serve` or
`cargo build -p retrograd-server` without `-p retrograd` gives a CPU-only
binary. See [Build variants](../reference/builds#servers).
:::

## Serve a model and its adapters

Write a configuration. Only `model` is required:

```toml
# serve.toml
model = "/models/base.gguf"
device = "auto"
n_ctx = 8192
n_seq_max = 8

[[adapters]]
name = "tuned"
path = "/models/tuned.gguf"

[[adapters]]
name = "tuned-v2"
path = "/models/tuned-v2.gguf"
```

Start the server:

```bash
./target/release/retrograd-infer serve.toml
```

It loads the model, and the first adapter, before it accepts connections. A
model that does not load stops the process. `Ctrl-C` stops the server.

The model is served as `base`, and each adapter under its `name`:

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "tuned", "messages": [{"role": "user", "content": "Hello"}]}'
```

Any OpenAI client works by pointing it at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
answer = client.chat.completions.create(
    model="tuned",
    messages=[{"role": "user", "content": "Hello"}],
)
print(answer.choices[0].message.content)
```

### Configuration of `retrograd-infer`

Unknown keys are errors. Unlike a run configuration, relative paths in this
file resolve against the working directory.

| Key | Default | Description |
| --- | ---: | --- |
| `model` | required | Base GGUF. |
| `[[adapters]]` | none | LoRA adapters of `model`: `name` and `path`. A name must be unique and must not be `base`. |
| `device` | `auto` | `auto`, `cpu` or `gpu`. |
| `n_ctx` | `8192` | Context per sequence, in tokens. |
| `n_seq_max` | `8` | Sequence slots: how many answers are decoded at once. |
| `host` | `127.0.0.1` | IP address, or `localhost`. |
| `port` | `8000` | TCP port. |
| `api_key` | none | Bearer token. Required when `host` is not a loopback address. |
| `queue` | `8` | Requests waiting for a slot. One more is refused with a 503. |
| `admission` | `continuous` | `continuous` refills a freed slot at the next step. `lockstep` admits requests in waves, for evaluation and tests that need a fixed order. |
| `idle_seconds` | `300` | Seconds without a request before the model is unloaded. The next request loads it again. |
| `responses_store_dir` | none | Directory of the [on-disk Responses store](./api#stored-responses). Without it, stored responses stay in memory. |
| `[decision]` | none | Replaces the head settings of a [decision model](./api#decisions). Refused at load for a model that is not one. |
| `decision.template` | the model's | A replacement `systemone` Jinja template. It receives the same context and must preserve the readout's markers or labels. |
| `decision.max_head_tokens` | the model's | Tokens of the question and its options kept by the truncation, for a model that reads its options at markers. |
| `decision.pack` | `1` | Prompts evaluated together in one forward pass for compatible heads, up to `n_seq_max`. Batching can improve throughput and change scores relative to separate passes. |

`--host`, `--port` and `--api-key` override the file:

```bash
./target/release/retrograd-infer serve.toml --host 0.0.0.0 --port 9000 --api-key "$API_KEY"
```

### Expert routing history

Set `[experts].history_dir` to collect routing counts on a validated MoE
(currently OLMoE). In a run configuration, use `[serving.experts]` instead.
The state directory must be separate from the model directory.

```toml
[experts]
history_dir = "state/expert-routing"
route_sample_every = 64  # Default; set to 1 to count every micro-batch.
```

Counts record the experts actually used by each bank. A token skipped by a
layer contributes no selection to that layer. Sampled counts are stored as
observed. History is keyed by the contents of all GGUF shards and merged in
the background. It does not change expert placement or routing.

## Serve from a training configuration

`retrograd serve` reads a run's TOML and serves what the run wrote. Use it to
evaluate a result right after training:

```bash
retrograd serve run.toml --model-name tuned
```

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=unused python my_eval.py
```

`--adapter`, `--checkpoint` or `--base-only` picks the weights; without them,
the server uses `[output].path`. The base model is served beside an adapter as
`base`. An adapter is served with the model's chat
template and the configuration's `template_variables`, so it is queried in the
format it was trained on. A `trainable` output is not a standalone model and is
refused: pass `--adapter`, `--checkpoint` or `--base-only` instead.

The context comes from `training.ctx`, which is often smaller than a chat needs.
Raise it with `--ctx`:

```bash
retrograd serve run.toml --ctx 8192
```

The model stays loaded for the life of the process, and stored responses stay
in memory. `--slots` sets the sequence slots (default 8). All flags are in the [CLI reference](../reference/cli#serve).

## Serve the runs of a server

`retrograd-server` serves every run it knows, including a run still in progress.
See [Serving runs](./server#serving-runs).

## How requests are served

Requests go through a scheduler that advances all active sequences one decode
at a time. The rollouts of a training run use the same scheduler.

- **Batching.** Sequences from different requests share a decode, up to
  `n_seq_max` slots. A request that finishes frees its slot at the next step.
- **Streaming.** `stream: true` sends one chunk per stable text delta. Text that
  could still turn into a tool call or a reasoning block is held back until it
  is settled, so nothing already sent is taken back.
- **Cancellation.** A client that disconnects frees its slot at the next step.
- **Prefix cache.** A prompt that starts like an earlier one reuses its KV
  cells instead of decoding them again.
  `prompt_tokens_details.cached_tokens` reports how many tokens were reused.
- **Adapters.** The adapters of one base stay loaded beside it. Requests for
  different adapters share a decode, each sequence selecting its own, and the
  prefix cache is partitioned per adapter. An adapter the per-sequence path
  refuses (activated LoRA, embedding or expert targets) is activated alone
  instead, and its requests do not share a decode with the others.

A model id with other base weights, such as a standalone exported model, needs
its own load: switching to it unloads the previous model.

### Sizing

`n_seq_max` is what allows concurrent requests, and it also caps `n` in a
request. Memory for the KV cache grows with `n_ctx × n_seq_max`, so a context of 8192
on 8 slots holds 65536 cells. Lower either to fit a device, and watch `memory` in the statistics below. A prompt plus its
answer must fit in `n_ctx`.

`queue` bounds how many requests wait. When it is full, the server answers 503
with a `Retry-After` header rather than holding the connection.

### Statistics

`GET /v1/engine/stats` reports the scheduler. It is outside the OpenAI contract,
and answers 404 until a model is loaded.

```bash
curl http://127.0.0.1:8000/v1/engine/stats
```

| Field | Meaning |
| --- | --- |
| `model` | The loaded base model. |
| `adapter` | The active adapter, empty for the base model. |
| `adapters` | The registered adapters. |
| `queued`, `live` | Requests waiting, and requests being decoded. |
| `kv.hits`, `kv.misses` | Prompts that did or did not reuse the prefix cache. |
| `kv.prompt_cells`, `kv.prefilled_cells` | Prompt tokens requested, and tokens actually decoded. |
| `kv.evictions`, `kv.swaps` | Cache entries dropped, and moved to host memory. |
| `kv.nodes`, `kv.resident_slots`, `kv.snapshot_bytes` | Size of the prefix tree, slots holding a cached prefix, and bytes of snapshots in host memory. |
| `memory.n_slots` | Sequence slots. |
| `memory.cells_total`, `memory.cells_used` | KV cells. |
| `memory.kv_bytes`, `memory.recurrent_bytes` | KV memory, and recurrent state for recurrent architectures. |

## Security

Without a key, the server listens on loopback only. A non-loopback address is
refused at startup unless `api_key` (or `--api-key`) is set. With a key, every
route requires `Authorization: Bearer KEY`.

`retrograd-infer` has one key and no per-client permissions. Put it behind a
reverse proxy that terminates TLS if it is reachable from a network.
