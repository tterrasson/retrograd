# Retrograd server

`retrograd-server` is a long-running process that trains and serves. Clients
create and control runs over HTTP, and every run's weights are available through
the [OpenAI-compatible API](./api). The [web interface](../operations/web-ui) is
a client of the same API.

## Start it

Build it with the root package, so that it gets the GPU backend:

```bash
cargo build --release -p retrograd -p retrograd-server
```

Without a configuration file, it listens on `127.0.0.1:8471`, keeps its state in
`./runs`, and has no authentication:

```bash
./target/release/retrograd-server
```

```bash
curl http://127.0.0.1:8471/v1/health
```

```json
{"status": "ok", "version": "0.1.0", "backends": ["cpu", "metal"]}
```

With a configuration file:

```bash
./target/release/retrograd-server server.toml --bind 127.0.0.1:9000 --state-dir /var/lib/retrograd
```

| Flag | Overrides |
| --- | --- |
| `--bind ADDR` | `bind` |
| `--state-dir DIR` | `state_dir` |
| `--vram-budget VALUE` | `vram_budget` |
| `--ram-budget VALUE` | `ram_budget` |

Three subcommands do not start the server:

```text
retrograd-server openapi         # write the OpenAPI document to stdout
retrograd-server config-schema   # write the run configuration's JSON Schema
retrograd-server view DIR        # serve one observe directory, read-only (needs `ui`)
```

`RUST_LOG` sets the log level (default `retrograd_server=info,tower_http=info`).

## Expose it to a network

On loopback, the defaults are open: a caller on the machine can already read its
files. On any other address the server refuses to start unless both of these
are set, because every model, dataset and output path comes from the client:

```toml
bind = "0.0.0.0:8471"
auth_token = "a-long-random-token"
path_roots = ["/models", "/srv/retrograd"]
```

Prefer the environment for the token. `RETROGRAD_SERVER_TOKEN` is read when the
file does not set `auth_token`.

- Clients send `Authorization: Bearer TOKEN`. There is one token, and it gives
  full control of the server.
- `GET /v1/health` is public. So are the web interface's static files.
- `path_roots` restricts every client-supplied path to those directories.
- Off loopback, `allow_local_paths` defaults to off: a recipe must use uploaded
  datasets, and `redact_error_paths` removes absolute paths from error responses.
  The `trace_id` of a response finds them again in the server's logs.

Commands and credentials never come from a client. Rewards, judges, MCP servers
and environments are declared in this file, and a client names them by `id`. See
[Catalog](#catalog).

The server does not terminate TLS. Put it behind a reverse proxy, with buffering
off on `/v1/` so that event streams arrive as they are written:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8471;
    proxy_buffering off;
}
```

## Configuration

Unknown keys are errors. The file is read once at startup.

### General

| Key | Default | Description |
| --- | ---: | --- |
| `bind` | `127.0.0.1:8471` | Listen address. |
| `state_dir` | `runs` | Runs, datasets and calibration data. |
| `auth_token` | none | Shared bearer token. Required off loopback. |
| `path_roots` | none | Directories client paths must resolve inside. Required off loopback. |
| `allow_local_paths` | on for loopback | Accept a literal path in `recipe.data.path` and `recipe.eval.path`. |
| `redact_error_paths` | off for loopback | Remove absolute paths from error responses. |
| `ui` | `true` | Serve the web interface, in a build that has it. |
| `observe_target_updates` | `100` | Updates of a rollout run that keep their texts, when the client does not choose. |

### Memory and runs

| Key | Default | Description |
| --- | ---: | --- |
| `vram_budget`, `ram_budget` | `"all"` | `"all"`, a share such as `0.8`, or a size such as `"12GiB"`. |
| `margin.fraction` | `0.05` | Share of each budget kept free, in `[0, 1)`. |
| `margin.floor_bytes` | 256 MiB | Smallest margin. |
| `max_concurrent_runs` | `1` | Runs that hold a device at once. Others queue. |
| `calibrate_runs` | `true` | Measure a configuration before creating the run. `false` answers faster, against the estimate only. |

### Limits

| Key | Default | Description |
| --- | ---: | --- |
| `command_timeout_seconds` | `300` | Wait for a run to answer `evaluate`, `generate` or a request for its live weights. |
| `request_timeout_seconds` | `60` | Any other request. Event streams and dataset uploads are exempt. |
| `upload_idle_timeout_seconds` | `60` | Longest pause between two chunks of an upload. |
| `max_body_bytes` | 1 MiB | Largest request body. |
| `max_concurrent_requests` | `256` | Requests served at once. |
| `max_dataset_bytes` | 1 GiB | Largest uploaded dataset. |
| `max_datasets_bytes` | 20 GiB | Total of the stored datasets. |

### `[serving]`

| Key | Default | Description |
| --- | ---: | --- |
| `enabled` | `true` | Serve the OpenAI routes. |
| `idle_seconds` | `300` | Seconds a loaded model may sit unused before the device is given back. |
| `device_wait_seconds` | `0` | How long a load waits for a device a run holds. `0` refuses at once with a 503 naming the run. |
| `queue` | `8` | Requests waiting for admission. |
| `n_seq_max` | `8` | Sequence slots. KV memory grows with the run's context times this value. |
| `admission` | `continuous` | `continuous` or `lockstep`, as in [`retrograd-infer`](./#configuration-of-retrograd-infer). |
| `max_body_bytes` | 4 MiB | Largest request on the OpenAI routes. |
| `responses_store_dir` | none | Directory of the on-disk Responses store. |
| `responses_store_ttl_seconds` | `86400` | Lifetime of a stored response. The ancestors of a living chain are kept. |

### Catalog

Each entry has an `id` and an optional `description`. A client sees the id, never
the rest.

```toml
[[reward]]
id = "exact"
description = "Exact match against the reference answer"
command = ["python3", "/srv/retrograd/reward.py"]
cwd = "/srv/retrograd"     # optional
timeout_seconds = 30       # default
mode = "persistent"        # or "oneshot", see the reward program protocol

[[judge]]
id = "rubric"
type = "ruler"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[[mcp_server]]
id = "calc"
command = ["python3", "/srv/retrograd/calculator_server.py"]
stateless = true

[[environment]]
id = "python-sandbox"
type = "container"
profile = "python"

[environment.tools]
default = "python"
```

The fields are those of the [judge](../reference/configuration#judge), of
[`[[agent.mcp_servers]]`](../reference/configuration#agent-mcp-servers) and of
[`[agent.environment]`](../reference/configuration#agent-environment), and the
reward program follows the [PPO protocol](../training/ppo#reward-program).
`[[environment]]` of type `container` needs a server built with
`--features retrograd-server/container`.

The server connects to every MCP server at startup, so that the published tool
lists are verified. A required server that does not answer stops the startup.

`GET /v1/rewards`, `/v1/judges`, `/v1/mcp-servers` and `/v1/environments` list
what is declared.

## Train over HTTP

A client can describe a run in three ways:

- a **recipe**, an intention that the server resolves into a configuration;
- a **`config`**, a complete configuration in JSON;
- a raw **TOML** body (`content-type: application/toml`), the same file
  `retrograd train` reads.

With a recipe, the server fills in everything left out: context, adapter rank,
memory settings. `GET /v1/defaults` publishes the rules, and `params` sets any
value by hand, using the names of the
[configuration reference](../reference/configuration). A value in `params` is
never changed by the resolver.

### Upload a dataset

```bash
curl 'http://127.0.0.1:8471/v1/datasets?name=demo&format=chat-jsonl' \
  -H 'content-type: application/x-ndjson' \
  --data-binary @train.jsonl
```

The answer carries the dataset's `id` (`ds_…`). The id is derived from the
content, so uploading the same file twice gives the same id. `format` is
`chat-jsonl`, `preference-jsonl` or `text`. A file with invalid lines is refused,
and the error lists them.

### Plan and create a run

`POST /v1/plan` resolves a request and creates nothing. It returns the effective
configuration, where each value came from, and a memory estimate:

```bash
curl http://127.0.0.1:8471/v1/plan \
  -H 'content-type: application/json' \
  -d '{
    "recipe": {
      "model": "/models/base.gguf",
      "data": {"dataset": "ds_4a75c8…"},
      "budget": {"epochs": 2}
    }
  }'
```

`POST /v1/runs` takes the same body and answers 201 with the run. Send an
`Idempotency-Key` header to make a retry safe.

| Recipe field | Description |
| --- | --- |
| `objective` | `instruction-tuning` (default, SFT), `reasoning-rl` (GRPO), `preference-rl` (PPO), `agentic`, `preference-tuning` (DPO). |
| `model` | Path of the base GGUF. |
| `data`, `eval` | `{"dataset": "ds_…"}`, or `{"path": "…"}` where local paths are allowed. |
| `budget` | One of `epochs`, `updates` or `minutes`. |
| `limits` | `vram` and `ram`, for this run. |
| `allow` | Accepted degradations: `truncate_context`, `exceed_train_context`. |
| `reward`, `judge` | `{"id": "…"}`, from the catalog. |
| `tools` | Tool ids. |
| `output`, `checkpoint_dir` | Where to write. By default, under the run's directory. |
| `seed` | Seed. One is recorded either way. |

A request that cannot fit the memory budget is refused with
`insufficient-memory`, and the answer says what dominates the estimate and which
`allow` entries would help.

`fork_from` continues from a checkpoint: `{"fork_from": {"run": "RUN_ID"}}` is a
complete body. `POST /v1/preflight` builds the training graph for a model and
reports the operations the device cannot run.

### Follow and control a run

| Route | Effect |
| --- | --- |
| `GET /v1/runs`, `GET /v1/runs/{id}` | List runs, or read one: state, progress, effective configuration, plan. |
| `GET /v1/runs/{id}/events`, `GET /v1/events` | Server-sent events, replayed and then live. Reconnect with `Last-Event-ID`. |
| `GET /v1/runs/{id}/metrics` | Metric samples. |
| `POST /v1/runs/{id}/pause`, `/resume` | Pause at the next progress callback, and resume. |
| `POST /v1/runs/{id}/cancel` | `{"at": "boundary", "checkpoint": true}` stops at the end of the epoch or update. `{"at": "now"}` stops at the next callback, and cannot checkpoint. |
| `PATCH /v1/runs/{id}` | Change `training.lr`, the evaluation or the checkpoint schedule. Anything else is refused with a 409: fork instead. |
| `POST /v1/runs/{id}/checkpoints` | Ask for a checkpoint at the next boundary. |
| `POST /v1/runs/{id}/evaluate` | Run the configured evaluation now. The result does not count toward `best` or `patience`. |
| `POST /v1/runs/{id}/generate` | Sample from the adapter as it is now. |
| `GET /v1/runs/{id}/artifacts`, `/artifacts/{name}` | List what the run wrote, and download it. |
| `GET /v1/runs/{id}/trajectories` | The run's [`[observe]`](../training/observe) export. |
| `DELETE /v1/runs/{id}` | Forget a finished run. |

The complete contract is `GET /v1/openapi.json`. Errors are
[problem documents](../engineering/server/ERRORS).

## Serving runs

Each run is a set of model ids on the OpenAI routes:

| Model id | Weights |
| --- | --- |
| `RUN_ID` | The live weights while the run trains, its final result afterwards. |
| `RUN_ID@final` | What the run wrote when it finished. |
| `RUN_ID@best` | The best checkpoint. |
| `RUN_ID@step-N` | A checkpoint, by step. |
| `RUN_ID@latest` | The checkpoint `--resume` would pick. |
| `RUN_ID@base` | The base model, without the adapter. |

`GET /v1/models` lists the ids that can be served now. A trainable bundle has no
standalone form and is not listed.

```bash
curl http://127.0.0.1:8471/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "b30df52e-0636-48d5-b2a2-3a12ac11651d@best", "messages": [{"role": "user", "content": "ping"}]}'
```

Served weights use the run's context length and chat template variables, on
`serving.n_seq_max` slots. The LoRA checkpoints of one run and its `@base` share
decodes.

**A run in progress.** A request for `RUN_ID` is answered on the run's own
thread, at its next progress callback, within `command_timeout_seconds`. These
requests are limited to chat and Responses, with a buffered stream, one choice
and no logprobs. Embeddings are refused.

**Sharing the device.** Serving weights from disk borrows the device a run would
use. If a run holds it, the request gets a 503 naming the run, or waits
`device_wait_seconds` if that is set. A run that needs the device gets it back:
the server finishes the streams in progress, unloads the model and starts the
run. An idle model is unloaded after `idle_seconds`.

## State and restarts

```text
<state_dir>/
  <run-id>/          run.json, events.jsonl, the adapter and the checkpoints
  datasets/          uploaded datasets
  calibration.json   measured memory per configuration
```

The server reads this directory at startup. A run left `running` by a previous
process is marked `interrupted`. Fork it to continue from its last checkpoint.

`Ctrl-C` (`SIGINT`) stops accepting requests and asks each live run to checkpoint
and stop at its next boundary, waiting up to ten minutes. A second `Ctrl-C` exits
at once, and the runs are read as `interrupted` at the next start. A service
manager should stop the server with `SIGINT`: other signals skip this drain.
