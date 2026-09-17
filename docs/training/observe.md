# Observing rollouts

`[observe]` exports what a PPO, GRPO or agentic GRPO run produces, update by
update: the prompts or scenarios, the completions or conversations with their
tool calls, the rewards, advantages and training status. Agentic runs also
include the judge's explanation when available. A static viewer is written next to the data.

```toml
[observe]
directory = "runs/42/observe"   # required; created if missing
every = 1                      # export rollouts every N updates
max_text_chars = 0              # 0 keeps texts whole
```

`every` must be positive. Updates are numbered from one: `every = 10`
exports rollouts for updates 10, 20, 30, and so on. Summaries are queued at
every update, regardless of this setting. `max_text_chars` truncates each
exported text and marks the cut as `…[+N chars]`; identifiers, tool names and the JSON structure of
tool arguments are kept. It does not bound the size of a batch.

`[observe]` is refused for `sft` and `distill`.

The training loop does not wait for disk writes or for space in the export
queue. Preparing a batch still costs time and memory proportional to its
contents. A writer thread handles serialization and disk writes.

A full queue drops the batch and reports a warning once. A disk error disables
further export with a warning; training continues. `observe/dropped_batches`
counts queue delivery failures, not all records lost after a disk error.
Failure to create the directory or start the writer thread prevents the run
from starting. Closing the export waits at most two seconds for pending writes.

## Viewing a run

Open `<directory>/index.html` in a browser. No server is needed: the page loads
`feed/` through `<script>` tags, so it works from `file://`, and it polls every
two seconds. The dot in the header is green while the feed changed in the last
30 seconds.

The same directory can be served statically (`python3 -m http.server`),
uploaded to a bucket, or read from a remote GPU host through `sshfs`.
To copy a live export with `rsync`, copy the whole directory repeatedly. Publish
chunks before `feed/manifest.js` so the manifest never advertises files that
have not arrived yet. The writer replaces its local manifest atomically;
a remote copy must preserve that ordering.

**Open .jsonl** (or dropping a file on the page) loads an `observe.jsonl` that
was copied elsewhere; incomplete or invalid batches are ignored with a warning.

The viewer shows:

- curves of `reward/mean` (with its standard deviation), `batch/trained_fraction`,
  `completions/length_mean` and `policy/kl`, plus turns, tool calls and failure
  rate on an agentic run. Clicking a point opens its update;
- the list of updates. `j`/`k` and the arrow keys move through it; *follow
  latest* is turned off as soon as you navigate. An update without its summary
  is shown as incomplete: the update did not finish, or its record was dropped;
- for an update, one card per group (a flat list for PPO), the prompt once, and
  the members sorted by reward, with their advantage, token count and badges:
  `truncated`, `skipped: <reason>`, `trained`, `unknown execution` and
  `duplicate`;
- for an agentic run, the conversation: tool calls with their arguments, tool
  results linked to their call and marked when they failed, step rewards next
  to the messages they were given for, and the judge's explanation. The compact
  mode shows one line per trajectory;
- filters on text, reward range, trained members, truncated or skipped members
  and tool name, and a side-by-side comparison of two members.

Model and tool outputs are only ever rendered as text.

Past about 200 000 records, only the last N updates keep their rollouts in
memory (`keep` in the side bar); the curves keep every update.

## Directory layout

```
<directory>/
  index.html, viewer.css, viewer.js   rewritten when the run starts
  observe.jsonl                       the source of truth, append-only
  feed/
    manifest.js                       replaced after every batch
    <generation>/000000.js            one immutable chunk per batch
```

`observe.jsonl` is the file to read with `jq` or pandas; the feed is a
projection of it that is rebuilt whenever a run opens the directory. Neither is
guaranteed to survive a power failure: writes do not use `fsync`.

Only one run may export to a directory at a time. If the directory is already
locked, or the filesystem cannot provide a lock, export is disabled with a
warning.

## Record schema

Every line is a JSON object with `v` (schema version, `1`), `type`, `segment`,
`time` (UTC, RFC 3339), and `batch_id`, `batch_index` (from zero) and
`batch_len`, which identify the batch that wrote it. Updates are numbered from
one.

| `type` | Written | Fields |
| --- | --- | --- |
| `run` | when the run opens the directory | `algorithm` (`ppo`, `grpo`, `agent_grpo`), `model`, `resumed_from_update` (or `null`), `params` |
| `prompt` | with the first rollouts that reference it in the segment | `key` (`p:<index>` or `s:<scenario_id>`), `messages`, `reward_text` (PPO/GRPO), `metadata` (agent) |
| `rollout` | once per completion or trajectory, before the optimizer | see below |
| `selection` | agent only, before the epochs | `update`, `entries: [{group, member, advantage, eligible, skip_reason}]` |
| `outcome` | after all epochs of the update succeed | `update`, `entries: [{group, member, trained}]` |
| `update` | at the end of every update, trained or skipped | `update`, `status` (`completed` or `skipped`), `metrics` (the scalars the update published) |

A `rollout` record carries `update`, `group` (`null` for PPO), `member`,
`prompt` (the key of its `prompt` record), `seed`, `tokens`, `truncated`,
`reward` (the reward used to compute advantages), `reward_raw`, `judge_term`,
`advantage`, `eligible`, `trained` and `skip_reason`, then:

- **PPO and GRPO**: `completion`. GRPO's `reward_raw` is the reward before
  `overlong_penalty`, and `judge_term` the judge's share of it. PPO's
  `advantage` is the mean of the whitened per-token advantages, with
  `advantage_min` and `advantage_max` when the critic is enabled.
- **Agentic GRPO**: `messages` (after the scenario's system and user messages
  when `prefix` is `true`, whole otherwise), `step_rewards: [{step_index, kind,
  reward, message_indices}]`, `terminal_reward_raw`, `judge_explanation` and
  `metadata` (only the entries that differ from the scenario's). `tokens` counts
  trainable tokens. `reward_raw` is the terminal reward plus the step rewards
  before any truncation policy, `reward` the total handed to training; both are
  `null` when there is none. `advantage` and `eligible` arrive in the
  `selection` record. A step reward whose messages are unknown has an empty
  `message_indices`. Rollouts that failed produce no trajectory and no record;
  they are counted in the `agent/*` metrics.

`skip_reason` is one of `truncated`, `zero_signal`, `judge_dropped` (GRPO),
`unscored` and `update_skipped` (agent). A member truncated but kept by
`truncation = "min_reward"` stays `truncated: true` without being skipped.

`trained` is `false` for a confirmed exclusion and `true` only once an
`outcome` record confirms it. When a run stops, fails or loses that record, the
selected members keep `trained: null` (execution unknown or possibly partial).
An advantage alone does not confirm training.

## Resuming

A run that resumes from a checkpoint in the same directory recovers the log
first: an incomplete tail (a partial line, or a batch that was not finished) is
removed with a warning; anything damaged before it disables the export and
leaves the file untouched. The feed is then rebuilt in a new generation, and
the run starts a new `segment`.

A `run` record with `resumed_from_update = K` hides, in the viewer, the updates
above K that earlier segments wrote. A run started without a checkpoint opens an
independent segment and resets the view; the history stays in `observe.jsonl`.
Records are never matched across segments.

## Server

A run created through `retrograd-server` lists two artifacts when it declares
`[observe]`: `observe_log`, the `observe.jsonl` file, and `observe`, the
directory. The directory must be under an allowed path root.
