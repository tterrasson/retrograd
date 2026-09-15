# Errors

Every failure from the V1 API is an `application/problem+json` document (RFC
9457): every `type` the server can answer with, every `code` that can appear
on a `errors[]` entry, what triggers it, and what a client should do about it.

`crates/retrograd-server/tests/error_catalog.rs` checks this file against the
code, not the other way around: it fails if `ProblemKind::ALL` or
`ErrorCode::ALL` (`crates/retrograd-server/src/error.rs`) gains a variant with
no row here. A catalogue that can drift silently from what the server
actually emits would be worse than no catalogue.

## Document shape

```jsonc
{
  "type": "https://retrograd.dev/problems/invalid-request",
  "title": "invalid request",
  "status": 422,
  "detail": "the dataset could not be read",
  "trace_id": "01JQ8...",                       // correlates with the server's logs
  "errors": [
    {"pointer": "/recipe/data/path",
     "code": "path_not_found",                  // closed vocabulary, see below
     "message": "no such file or directory",
     "hint": "upload it with POST /v1/datasets instead"}   // only when there is a short answer
  ],
  "meta": {}                                     // typed payload specific to `type`, see below
}
```

`type` and `code` are the stable, testable parts of the contract. `title`,
`detail`, `message` and `hint` are prose and may be reworded without notice.

## `type` (`ProblemKind`)

| `type` | Status | When | What to do |
|---|---|---|---|
| `invalid-request` | 422 | Malformed body, a recipe or configuration that fails validation, an unsupported data format, a bad TOML document. | Fix the field(s) named in `errors[]`. |
| `server-declared` | 422 | The client supplied a reward command, a judge endpoint/key, or an MCP server command directly instead of referencing a catalogue id. Executable commands and credentials are declared by the operator, never sent by a client. | Declare the value server-side and send its `id` instead. |
| `unknown-catalog-id` | 422 | A `reward`, `judge`, MCP server, or `fork_from.run`'s reward id is not declared by this server. | `GET /v1/rewards`, `/v1/judges`, `/v1/mcp-servers` to see what is declared. |
| `insufficient-memory` | 422 | No arrangement of the memory levers makes the estimate (or, after calibration, the measurement) fit the effective budget. | Read `meta.dominant_posts` for what to shrink, `meta.unlocks` for opt-ins that would help, or lower `training.ctx`/`limits`. |
| `unauthorized` | 401 | The `Authorization` header is missing or its bearer token is wrong. Answers with `WWW-Authenticate: Bearer`. | Send a valid token. |
| `forbidden-path` | 403 | A client-supplied path resolves outside every root this server was configured with (`path_roots`). | Upload the file instead of naming a local path, or ask the operator to widen `path_roots`. |
| `not-found` | 404 | Unknown run, route, checkpoint, or artifact. | Check the id/URL; list the resource's collection endpoint. |
| `conflict` | 409 | A legal request against a resource whose current state forbids it: a control command on a run that cannot accept it, a fork that would change the training trajectory, a checkpoint request with nowhere to write one. | Read `detail`; wait for a different state, or start a new run instead of forking. |
| `payload-too-large` | 413 | The request body is larger than this server accepts. | Send a smaller body, or use `POST /v1/datasets` for large data instead of a bare `params`/`config` payload. |
| `unsupported-media-type` | 415 | The body did not arrive labelled as JSON. `POST /v1/plan` and `POST /v1/runs` also accept `application/toml`; `POST /v1/datasets` reads `application/x-ndjson`. | Send `content-type: application/json` (or the content type that route documents). |
| `method-not-allowed` | 405 | The path exists but not for this HTTP method. | Check the method against `GET /v1/openapi.json`. |
| `device-busy` | 503 | Every run slot is taken, or the probe/device queue is closed (server shutting down). | Retry later, or increase `max_concurrent_runs`. |
| `timeout` | 504 | `evaluate` or `generate` did not get an answer from the run's loop before `command_timeout_seconds`. | Retry; check the run is actually progressing. |
| `not-implemented` | 501 | A route that exists in the V1 contract but is not wired in this build. | Not yet available; nothing a client can do. |
| `internal` | 500 | A failure that is this server's fault, not the caller's. | Retry; if it persists, report `trace_id` and the server's logs to the operator. |

## `code` (`ErrorCode`, on `errors[]`)

| `code` | Meaning | Typical `type` |
|---|---|---|
| `missing_field` | A required field was absent. | `invalid-request` |
| `unknown_field` | `deny_unknown_fields` rejected a field the schema does not have. `hint` names the closest known field, when one is within edit distance 2. | `invalid-request` |
| `invalid_value` | The field is present but its value does not fit: wrong shape, wrong enum member, or a combination with a sibling field that is not allowed. | `invalid-request` |
| `out_of_range` | A numeric field is outside the range that field accepts (zero where positive is required, a temperature outside `[0, ∞)`, …). | `invalid-request` |
| `unsupported_format` | A data or media-type field named something this server does not parse. | `invalid-request` |
| `path_not_found` | A path field names something that does not exist on this server's filesystem. | `invalid-request` (the status stays 422: the body is syntactically valid, only its content resolves nowhere) |
| `forbidden_path` | A path field names something outside every root this server serves. | `forbidden-path` |
| `unknown_catalog_id` | A catalogue reference (`reward`, `judge`, an MCP server, a forked run's reward) the operator has not declared. | `unknown-catalog-id` |
| `needs_opt_in` | The request needs a degradation (`Allow`) and did not opt into it. `detail` says what the degradation would have done. | `invalid-request` |
| `override_conflict` | A client-supplied `params` value conflicts with the rest of the request. A client-supplied parameter is never silently dropped or moved. | `conflict` |
| `conflict` | A legal request against a resource whose current state forbids it. | `conflict` |
| `unauthorized` | The caller's credentials are absent or wrong. | `unauthorized` |
| `not_found` | The referenced resource does not exist. | `not-found` |
| `server_declared` | The field is reserved for the operator; a client may not set it. | `server-declared` |

Saturation and shutdown have no `code`: `device-busy`, `timeout` and `internal`
are about the server, not about a field of the request, so they answer with a
`detail` and an empty `errors[]`. `tests/error_catalog.rs` enforces the
converse of the table - a `code` this server never emits does not get to sit
here looking actionable.

## `meta`

Typed, problem-specific payload. Only two shapes exist today:

```jsonc
// insufficient-memory
"meta": {
  "overflow_bytes": 402653184,
  "vram_budget_bytes": 6442450944,
  "ram_budget_bytes": 68719476736,
  "dominant_posts": [{"post": "logits_bytes", "bytes": 1866989568}, "…"],
  "levers_applied": ["gradient_checkpointing"],
  "unlocks": ["truncate_context"]
}
```

`levers_applied` and `unlocks` are empty when the caller supplied a whole
configuration (`config`/TOML forms) rather than a `recipe`: those forms skip
phase 3, so there is nothing to report beyond the overflow itself.

A future `dataset-invalid` problem (per-line dataset validation) will carry
`meta.line_errors`; it is not implemented yet.

## `trace_id`

Every response - success or failure - is handled inside one request span
tagged with a random `trace_id`. A problem document echoes it; every log line
written while handling that request carries the same one
(`tracing::info_span!("request", trace_id = …)`). When `redact_error_paths` is
on, `trace_id` is the only way to get from a truncated `<path>/model.gguf` in
a response back to the full path in the server's own logs - that is its only
purpose.

## Warnings

`plan.warnings[]` (on a successful `/v1/plan` or `/v1/runs` response) is a
separate, smaller vocabulary - it never fails a request, it tells a client
what the resolver decided on its behalf. Each entry is `{code, field, message}`
with `field` optional (absent when the warning is about the request as a
whole rather than one dotted path). It is not yet unified with the `code`
vocabulary above; the missing piece is mainly an `impact` field on each
warning, still deferred.
