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
| `preference` | Not supported. A run with [`[evaluation]`](../training/preference#evaluation) reports its preference accuracy at each evaluation. |

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

## `serve`

```text
retrograd serve CONFIG.toml [--adapter ADAPTER.gguf | --checkpoint DIR/best.state | --base-only]
                [--model M] [--device D] [--ctx N] [--host 127.0.0.1] [--port 8000]
                [--api-key KEY] [--model-name NAME]
```

Serves the model through the OpenAI API, so any OpenAI client - the `openai`
SDK, lm-eval's `local-chat-completions` and `local-completions`, Open WebUI,
inspect - can query it:

| Route | What it serves |
|---|---|
| `GET /v1/models` | the served ids, plus a `capabilities` extension on the list object |
| `POST /v1/chat/completions` | chat, tools, streaming |
| `POST /v1/responses` | the Responses API, stored or not (`store: false`) |
| `GET`/`DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` | the stored responses |
| `POST /v1/completions` | legacy raw-prompt completions, no chat template |
| `POST /v1/embeddings` | pooled embeddings, `float` or `base64` (little-endian `float32`) |

```bash
retrograd serve run.toml --model-name tuned
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=unused python my_eval.py
```

The weights are chosen like `chat`: the adapter the configuration trains
unless `--adapter`, `--checkpoint` (the adapter exported beside a `.state`
directory) or `--base-only` says otherwise. The model is loaded before the
server answers. Its id is `--model-name`, by default the adapter's file name;
the base model is served beside it as `base`, which reloads the weights on each
switch between the two.

The prompt is rendered and parsed with the code the agentic rollouts use:
`tools` reach the model through its own chat template when it has a tool
format (otherwise through the system prompt), and calls come back as
`tool_calls`. `stream: true` streams one chunk per stable delta; a client that
disconnects frees its slot at the next step. `temperature: 0` is greedy.
`tool_choice: "none"` leaves the catalog out of the prompt.

Sampling accepts `top_p`, `top_k`, `min_p`, the repeat/frequency/presence
penalties, `logit_bias`, `seed`, `stop` and `n` (at most 128). `logprobs` and
`top_logprobs` (at most 20) describe the distribution the token was drawn
from, after every transformation. `response_format` (`json_object`, or
`json_schema` within the supported subset - an unsupported keyword is refused
with its path), `tool_choice: "required"`, a forced function and
`parallel_tool_calls: false` constrain decoding with a grammar; no call is
ever removed after the fact. What stays unsupported (audio, non-text
modalities, Responses `background`, `truncation`, hosted tools, encrypted
reasoning) is refused with a 400 naming the field.

Responses are stored by default in a bounded in-memory store (1024 responses,
64 MiB, chains up to 64 deep), so `previous_response_id` continues a
conversation; an evicted or unknown id answers 404. A stored turn also keeps
its KV pinned in the prefix cache until the turn that continues it, so that
turn prefills only what is new - reported as
`usage.input_tokens_details.cached_tokens` (`prompt_tokens_details` in chat).
This is a hint, not a guarantee: pins are bounded (a quarter of the slots,
five minutes) and released first when an admission needs the room.
`reasoning.effort` maps to a local reasoning budget (`none` 0, `minimal` 256,
`low` 1024, `medium` 4096, `high` 16384, `xhigh` 65536, `max` unlimited),
not to a hosted model's effort.

Binding anything but a loopback address requires `--api-key`, which clients
send as `Authorization: Bearer KEY`.

`retrograd-server` serves the same routes for its runs: `RUN_ID` (the live
weights while the run trains, its final adapter afterwards), `RUN_ID@final`,
`RUN_ID@best`, `RUN_ID@step-N`, `RUN_ID@latest` and `RUN_ID@base`. A live run
answers at its next progress callback. Its `[serving]` section sets `enabled`,
`idle_seconds` (default 300), `device_wait_seconds` (default 0: a load refused
while a run holds the device), `queue` (8), `max_body_bytes` (4 MiB),
`responses_store_dir` (a file store instead of the in-memory one) and
`responses_store_ttl_seconds` (default 86400). The LoRA checkpoints of one run
(`RUN_ID@step-N`, `@best`, …) and `@base` share decodes: each request's rows
select their adapter, the prefix cache is partitioned per adapter, and an
adapter the per-row path refuses (activated LoRA, embedding or expert
targets) falls back to being activated alone, one adapter at a time. A live run
serves chat and Responses with a buffered stream, one choice and no logprobs.

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

## `collect`

```text
retrograd collect CONFIG.toml --out FILE.jsonl [--k N] [--keep N] [--min-reward R]
                  [--require-verified|--allow-unverified] [--raw] [--limit N]
                  [--seed N] [--report FILE.json] [--force] [--model M | --api]
                  [--device D] [--pairs [--min-gap G]]
```

Rolls out the scenarios of an `agent_grpo` configuration and writes the
successful traces as [tool records](../getting-started/datasets#tool-conversations)
for SFT. It reads the model and `init_adapter`, `[agent]` and its judge; it
trains nothing.

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--k` | `group_size` | Attempts per scenario. With `1`, the environment must grade every scenario. |
| `--keep` | `1` | Traces kept per scenario: the highest reward, the shortest at equal reward, no exact duplicate. |
| `--min-reward` | none | Drop traces below this total reward. |
| `--require-verified` | on when every scenario has a `verify` | Keep only traces the environment verified. `--allow-unverified` turns it off. |
| `--raw` | off | Write assistant turns verbatim instead of structured. Only a student of the generator's family can train on them. |
| `--limit` | all | Collect from the first N scenarios. |
| `--seed` | `[agent].seed` | Scenario `i` is rolled out from `seed + i × k`. |
| `--report` | none | Per-scenario attempts, passes, kept traces and pass rate, as JSON. |
| `--force` | off | Overwrite `--out`. |
| `--api` | off | Generate with the [`[agent.collect_api]`](./configuration#agent-collect-api) endpoint instead of a local model. No model is loaded; only the environment grades, and turns are structured. |
| `--pairs` | off | Write one [preference pair](../training/preference) per scenario instead: the best trace that passes the filters as `chosen`, the worst complete one that can be written as `rejected`. Takes no `--keep`. |
| `--min-gap` | `0.5` | With `--pairs`: skip a scenario whose two rewards differ by less. |

Traces that were truncated, contain a malformed call, or fail a filter are
counted by cause and printed at the end, and with `--pairs` so are the
scenarios that gave no pair. The file is written as `FILE.jsonl.tmp`
and renamed once complete; an interrupted collection leaves the `.tmp` file
with what it had kept.

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
