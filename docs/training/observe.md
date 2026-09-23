# Observing rollouts

`[observe]` records what a PPO, GRPO or agentic GRPO run generates, update by
update (prompts, answers or conversations, tool calls, rewards, advantages)
and writes a viewer next to it. It is not available for SFT and distillation.

```toml
[observe]
directory = "runs/42/observe"   # created if missing
every = 10                      # rollouts for updates 10, 20, …; summaries every update
max_text_chars = 0              # truncate long texts; 0 keeps them whole
```

Exporting never slows training down: if the disk cannot keep up, a batch is
dropped with a warning (`observe/dropped_batches`), and a disk error disables
the export without stopping the run.

## Viewing a run

Open `<directory>/index.html` in a browser. No server is needed, and the page
refreshes as the run progresses. The directory can also be served statically,
synced from a remote machine, or opened later with **Open .jsonl**.

The viewer shows:

- curves of reward, trained fraction, answer length and KL (plus turns and tool
  calls for agentic runs); click a point to open its update;
- each group with its prompt and its answers sorted by reward, with their
  advantage and badges such as `truncated` or `skipped`;
- for agentic runs, the full conversation: tool calls, tool results, step
  rewards and the judge's explanation;
- filters and a side-by-side comparison of two answers.

Use `j`/`k` or the arrow keys to move between updates.

## Files

```text
<directory>/
  index.html, viewer.css, viewer.js
  observe.jsonl     all records, append-only: the file to read with jq or pandas
  feed/             what the viewer loads, rebuilt from observe.jsonl
```

Only one run can write to a directory at a time. When a run resumes from a
checkpoint in the same directory, it continues the same log; the viewer hides
updates that the resumed run will replay.

## Record schema

Each line of `observe.jsonl` is a JSON object with a `type`:

| `type` | Written | Main fields |
| --- | --- | --- |
| `run` | when the run starts | `algorithm`, `model`, `resumed_from_update`, `params` |
| `prompt` | once per prompt | `key`, `messages` |
| `rollout` | once per answer or trajectory | `update`, `group`, `member`, `prompt`, `completion` or `messages`, `tokens`, `truncated`, `reward`, `advantage`, `trained`, `skip_reason` |
| `selection` | agentic runs, before training | `update`, `entries`: advantage and eligibility per member |
| `outcome` | after the update trained | `update`, `entries`: which members were trained |
| `update` | at the end of every update | `update`, `status` (`completed` or `skipped`), `metrics` |

Every record also carries `v` (schema version), `time`, `segment` and batch
identifiers. Updates are numbered from 1.

A few fields worth knowing:

- `reward` is the value used for training. For GRPO, `reward_raw` is the value
  before `overlong_penalty` and `judge_term` the judge's share.
- `skip_reason` is one of `truncated`, `zero_signal`, `judge_dropped`,
  `unscored` or `update_skipped`.
- `trained` is `true` only once an `outcome` record confirms it, `false` for a
  member that was excluded, and `null` if the run stopped before confirming.
- Agentic rollouts also carry `step_rewards`, `terminal_reward_raw` and
  `judge_explanation`. Trajectories that crashed are not recorded; they are
  counted in the `agent/*` metrics.
