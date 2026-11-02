# OpenAI-compatible API

`retrograd-infer`, `retrograd serve` and `retrograd-server` serve the same
routes under `/v1`. The scope is the text surface listed here, not the whole
OpenAI API. A field that is not served is refused with a 400 naming it, never
ignored.

| Route | What it serves |
| --- | --- |
| `GET /v1/models` | The served ids, with a `capabilities` extension on the list object. |
| `GET /v1/models/{model}` | One id. |
| `POST /v1/chat/completions` | Chat, tools, streaming. |
| `POST /v1/responses` | The Responses API, stored or not. |
| `GET`, `DELETE /v1/responses/{id}` | One stored response. |
| `GET /v1/responses/{id}/input_items` | Its input items, paginated with `limit`, `order`, `before` and `after`. |
| `POST /v1/completions` | Raw-prompt completions, without a chat template. |
| `POST /v1/embeddings` | Pooled embeddings. |
| `POST /v1/systemone` | [Typed decisions](#decisions) of a decision model, with llama-server's contract. |
| `GET /v1/engine/stats` | The [scheduler's statistics](./#statistics). Not part of the OpenAI contract. |

## Chat Completions

The prompt is rendered and parsed with the code the agentic rollouts use, so a
model is evaluated on the format it was trained on. `tools` reach the model
through its own chat template when it has a tool format, and through the system
prompt otherwise. Calls come back as `tool_calls`. `tool_choice: "none"` leaves
the catalog out of the prompt.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "base",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

`temperature: 0` is greedy. `stream: true` streams one chunk per stable delta.
Add `stream_options: {"include_usage": true}` to receive the usage in a last
chunk.

## Sampling

| Field | Accepted values |
| --- | --- |
| `temperature`, `top_p` | As in the OpenAI API. |
| `top_k` | `-1`, or greater than 0. An extension. |
| `min_p` | `[0, 1]`. An extension. |
| `frequency_penalty`, `presence_penalty` | `[-2, 2]`. |
| `repeat_penalty` | Greater than 0. An extension. |
| `penalty_window` | Tokens the penalties look back over. An extension. |
| `logit_bias` | Token id to a bias in `[-100, 100]`, at most 4096 entries. |
| `seed` | Fixed seed. Each choice derives its own from it. |
| `stop` | Stop strings. |
| `n` | Choices per request, at most 128 and at most the server's sequence slots. They share one prefill. |

`logprobs` and `top_logprobs` (at most 20) describe the distribution the token
was drawn from, after every transformation above. They differ from the
log-probabilities used for training, which are taken at temperature 1. In the
Responses API, ask for them with `include: ["message.output_text.logprobs"]`
and `top_logprobs`.

## Structured outputs

`response_format` (`text.format` in the Responses API) accepts `json_object` and
`json_schema`. The constraint is a grammar applied while decoding, so nothing is
removed after the fact.

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"]
      }
    }
  }
}
```

Supported JSON Schema keywords: `type`, `properties`, `required`,
`additionalProperties`, `items`, `prefixItems`, `minItems`, `maxItems`,
`pattern`, `minLength`, `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`,
`exclusiveMaximum`, `minProperties`, `maxProperties`, `enum`, `const`, `anyOf`,
`oneOf`, `allOf`, local `$ref` (`#/definitions/…`, `#/$defs/…`), `definitions`,
`$defs`, and the annotations `title`, `description`, `default`, `$schema` and
`examples`.

Any other keyword (`format`, `not`, `if`, `contains`, `patternProperties`,
`multipleOf`, a non-local `$ref`) is refused with its path in the schema.

An answer cut by its token budget may not be valid. It is reported as
`finish_reason: "length"` in chat and `status: "incomplete"` in Responses.

`tool_choice: "required"`, a forced function and `parallel_tool_calls: false`
use a grammar too. Two combinations are refused: `response_format` with tools,
and `parallel_tool_calls: false` with a constraining `tool_choice`.

## Responses

`POST /v1/responses` takes `input` as a string or a list of items.
`max_output_tokens` must be at least 16.

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'content-type: application/json' \
  -d '{"model": "base", "input": "Say hello.", "max_output_tokens": 64}'
```

### Stored responses

Responses are stored by default, so `previous_response_id` continues a
conversation. Send `store: false` to keep nothing.

| Store | Limits |
| --- | --- |
| In memory (default) | 1024 responses, 64 MiB, chains up to 64 deep. |
| On disk | One JSON file per response, written atomically. Set `responses_store_dir`. On `retrograd-server`, `responses_store_ttl_seconds` sets the purge delay (default a day). |

An evicted or unknown id answers 404. The in-memory store is lost when the
process stops.

A stored turn keeps its KV pinned in the prefix cache until the turn that
continues it, so that turn prefills only what is new. It is reported as
`usage.input_tokens_details.cached_tokens` (`prompt_tokens_details` in chat).
This is a hint, not a guarantee: pins are limited to a quarter of the slots (so
none below four slots) and to five minutes, and are released first when a new
request needs the room.

### Reasoning

`reasoning.effort` maps to a local budget of reasoning tokens, not to a hosted
model's effort:

| `effort` | `none` | `minimal` | `low` | `medium` | `high` | `xhigh` | `max` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tokens | 0 | 256 | 1024 | 4096 | 16384 | 65536 | unlimited |

## Completions

`POST /v1/completions` decodes `prompt` without a chat template. It is what
lm-eval's `local-completions` calls.

## Embeddings

`POST /v1/embeddings` returns pooled embeddings as `float`, or as `base64`
(little-endian `float32`). `input` is a string, a list of strings, or token
lists. Embeddings are computed from the weights on disk, so a model id that
names a run in progress is refused unless it has a suffix such as `@final`. A
causal chat model that declares no pooling is refused.

## Decisions

`POST /v1/systemone` serves a decision model: a GGUF whose metadata declares
`<arch>.decision.type`. It answers typed questions in one forward pass each
and generates nothing. The body and the answer are llama-server's, so a client
of its decision server works unchanged:

```json
{
  "model": "base",
  "state": {"message": "I was charged twice", "plan": "pro"},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "wants money back", "other": "anything else"}},
    "urgent": {"type": "noul", "instructions": "Does this need a human within the hour?"},
    "frustration": {"type": "score", "instructions": "How frustrated?",
                    "criteria": ["calm", "annoyed", "angry"]}
  }
}
```

```json
{
  "model": "base",
  "answers": {
    "intent": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.93, "other": 0.07}, "confidence": 0.62},
    "urgent": {"type": "noul", "noul": 0.21},
    "frustration": {"type": "score", "score": 0.8, "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
                    "probabilities": {"0": 0.38, "1": 0.44, "2": 0.18}, "confidence": 0.05}
  },
  "usage": {"input_tokens": 241, "output_tokens": 0}
}
```

- The answers keep the order of the questions, and the `state` is read with
  the order of its keys: it is part of the prompt.
- `model` picks the base or one of its adapters, as on the other routes. It may
  be omitted when the server serves one id, as llama-server reads no `model`.
  An adapter [trained by retrograd](../training/decision) is served with the
  temperatures fitted on it (`adapter.decision.temperature.*`) in place of the
  model's.
- The prompt is the model's `systemone` template. The model's
  `max_head_tokens` and temperatures apply. `retrograd-infer` can replace the
  template and the budget in its [`[decision]` section](./#configuration-of-retrograd-infer).
- One question per forward pass by default: the answers then match
  llama-server's on the same backend and build. Packing several questions in a
  pass (`pack`) is faster, and makes each answer depend on the other questions
  of its request.
- Every `<arch>.decision.type` of llama.cpp's decision server is recognised,
  with its own checks of the body and its own messages:

  | Type | Where the scores are read | What it changes |
  | --- | --- | --- |
  | `laya` | the embeddings output at the mask token of each option, in the column of the question type | the question and the options are cut to `max_head_tokens` |
  | `lfm2-d1-omni` | the same column, at the mask token | the prompt is cut piece by piece to its budgets; a null `state`; `noul` descriptions also read under `no`/`yes` |
  | `openjev` | the logits of one letter per option (`A`..`Z`, `a`..`z`), at the last token | `noul` lists `true` first |
  | `lev` | the logits of a code per option (`A`..`Z`, `AA`..`ZZ`) | keys sorted; a `choice` is read in two orders; `noul` is the expected rating of a 9-label scale |
  | `nimble` | same codes | the prompt lists every question of the request |
  | `pplx-decider`, `granite-switch` | same codes | `granite-switch` lists `true` first |
  | `vllm-d3` | same codes | a null `state` or `instructions`; non-text legend entries as canonical JSON; entropy confidence |
  | `lfm2-d1`, `lfm3-d1`, `lfm3-d1-omni` | the largest logit of a group of label tokens per option, the labels chosen per question | a null `state`; `lfm3` breaks special token texts in the input, its confidence is the answer's probability and its score the median level |
  | `h2o-lightning` | the logits of the labels of `<arch>.decision.labels` | texts rendered as its Python reference does; the `type` defaults to `choice`, the `instructions` to empty |
  | `kev` | `q·k` of the hidden states of the last token and of each option's end token | inputs flattened to text |
  | `clef` | a joint head, one score per option, every question in one prompt | `choice` options sorted by key; `noul` lists `true` first |
  | `vllm-decision2` | the same head, one prompt per question | text, an object or an array only; at least two options; entropy confidence |
  | `openbmb-minicpm-ocr-jev` | one score per class, the pooled output | a fixed question about two images: always answered 501 here |

- Images, audio, video and spoken instructions are refused with a 501: the
  engine has no multimodal projector. A request that a model refuses before it
  looks at its media (a missing state, the second image of
  `openbmb-minicpm-ocr-jev`) is answered with llama-server's 400.
- A decision model refuses `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses` and `/v1/embeddings` with a 501. A model that is not a
  decision model refuses `/v1/systemone` the same way.

`retrograd decide --model decision.gguf --request request.json` answers one
body from the command line, without a server.

## Clients

Anything that speaks this API can connect: the `openai` SDKs, lm-eval's
`local-chat-completions` and `local-completions`, inspect and Open WebUI. Set
the base URL to `http://HOST:PORT/v1`. On a server without a key, the API key
can be any string.

`GET /v1/models` adds what the server supports:

```json
{
  "object": "list",
  "data": [{"id": "base", "object": "model", "created": 1793421020, "owned_by": "retrograd"}],
  "capabilities": {"completions": true, "embeddings": true, "responses_store": true, "systemone": true}
}
```

## Not served

Audio, non-text modalities, and in the Responses API `background`, `truncation`,
hosted tools and encrypted reasoning. Each is refused with a 400 naming the
field.

## Errors

These routes answer with the OpenAI error envelope, which the SDKs read:

```json
{"error": {"type": "invalid_request_error", "code": null, "param": "max_output_tokens", "message": "`max_output_tokens` must be at least 16"}}
```

| Status | Meaning |
| --- | --- |
| 400 | A field is invalid or not served. `param` names it. |
| 401 | The bearer token is missing or wrong. |
| 404 | Unknown model, or unknown or evicted response. |
| 501 | `not_supported_error`: the model cannot answer this route at all - a decision asked of a model that is not a decision model, images sent to a decision model, a decision model asked to generate. |
| 503 | The queue is full or the device is held by a run. `Retry-After` says when to try again. |

The other routes of `retrograd-server` use
[problem documents](../engineering/server/ERRORS) instead.
