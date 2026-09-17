# GRPO training

GRPO samples several completions for each prompt and centers their rewards
within the group. It is useful when the reward can rank alternative answers but
does not provide a stable absolute value across prompts.

Retrograd implements Dr. GRPO behavior with DAPO-style corrections: no reward
standard-deviation normalization, a constant generation-budget loss divisor,
optional truncation masking, and separate lower and upper clip ranges.

## Configuration

```toml
[run]
algorithm = "grpo"

[model]
path = "base.gguf"
device = "auto"

[lora]
output = "grpo-adapter.gguf"
rank = 8
alpha = 16.0
seed = 42

[training]
ctx = 512
micro_batch = 32
shared_prefix_fanout = "auto"
generation_concurrency = 8
lr = 0.00001
max_grad_norm = 1.0

[grpo]
prompts = "prompts.jsonl"
reward_command = ["python3", "reward.py"]
reward_mode = "persistent"
updates = 50
prompts_per_update = 4
group_size = 8
grpo_epochs = 2
clip_range_low = 0.2
clip_range_high = 0.28
kl_coefficient = 0.0
mask_truncated = true
baseline = "mean"
prompt_order = "sequential"
max_stalled_updates = 25

[grpo.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 128
seed = 42
```

Run it with:

```bash
retrograd train grpo.toml
```

GRPO prompts use the same chat JSONL shape as PPO and must end with a user
message. Each prompt is sampled `group_size` times. A group with rewards
`[r1, r2, ...]` uses mean-centered advantages:

```text
advantage_i = reward_i - mean(group rewards)
```

The group size must be at least `2` and fit the optimizer window.

## On-policy sampling

Dr. GRPO is strictly on-policy. Retrograd therefore requires:

```toml
[grpo.sampling]
temperature = 1.0
top_p = 1.0
```

`top_p` and `temperature` values other than `1.0` are rejected for GRPO. Use
`seed` to make sampling reproducible where the backend and model execution are
otherwise deterministic.

## Update sequence

For each update, Retrograd:

1. Draws `prompts_per_update` prompts.
2. Samples `group_size` completions per prompt, possibly over several generation
   waves controlled by `training.generation_concurrency`.
3. Scores all completions with the reward command.
4. Drops groups with no reward variation. If `mask_truncated = true`, budget-
   truncated completions are excluded before the group baseline is computed.
5. Computes group-relative advantages and makes `grpo_epochs` optimizer passes.

`training.generation_concurrency` changes the number of live decode sequences,
not the statistical group. Lower it to reduce generation KV-cache memory; raise
it to improve throughput when memory allows. `shared_prefix_fanout` changes
physical packing of completions that share a prompt and does not change the
group baseline.

### Shared-prefix packing

Completions of the same prompt can be packed into one pass so the prompt is
processed once. This works on attention-only models and on LFM2/LFM2MoE (after
a device check). Other models fall back to one row per completion.

| `shared_prefix_fanout` | Behavior |
| --- | --- |
| `"auto"` | Packs as many completions per pass as fit; falls back to rows otherwise. |
| `"off"` | Always uses rows. |
| `"max"` | Requires the largest possible packing; errors if it does not fit. |
| integer | Requires that many completions per pass; errors if it does not fit. |

A pass needs this many tokens, and must fit in `micro_batch`:

```text
(prompt_tokens - 1) + sum(completion_tokens) + (n_seq_max - completions)
```

For example, a 250-token prompt (chat template included) with eight 15-token
completions needs `249 + 8 * 15 = 369` tokens: `micro_batch = 512` fits,
`256` does not. Raising `ctx` does not help; raise `micro_batch`.

Training logs explain when and why packing falls back. Packing usually saves
time, but not always: check optimizer timings.

### Measuring packing geometry

`POST /v1/plan?calibrate=true` benchmarks up to four `micro_batch`/fanout
combinations for GRPO and agent GRPO, and keeps the fastest one if it is clearly
(more than 5%) faster than the default choice. The benchmark times full
optimizer updates on synthetic data, with a throwaway adapter, so it ignores
generation and rewards. The result appears in the `packing_geometry_measured`
plan warning (or `packing_geometry_unmeasured` if it could not run). Regular
planning and `retrograd train` never run it.

## Optional group judge

`[grpo.judge]` adds a second, *relative* score to the reward-command result.
The judge sees all completions for one prompt together and returns one verdict
per completion in the range `[0, 1]`; it is therefore useful when correctness
is qualitative (helpfulness, style, completeness, or a rubric) rather than
fully captured by a scalar programmatic reward. The final reward for a
completion is:

```text
reward_command_score + judge_weight * judge_verdict
```

`judge_weight` is required whenever a judge is configured. Keep it small when
the command already establishes correctness, and raise it only when the judge
is intended to decide most of the ranking. A reward-command response can
override the configured weight for that one completion with
`{"reward": 0.8, "judge_weight": 0.0}`; this is useful for an answer that a
deterministic checker has already settled. The judge is used during training
rollouts, not held-out `[evaluation]` scoring.

### RULER judge

RULER is an OpenAI-compatible LLM judge. Keep the API key out of TOML and name
the environment variable that supplies it:

```toml
[grpo]
judge_weight = 0.35
judge_failure = "drop_group"
max_judge_dropped_fraction = 0.25

[grpo.judge]
type = "ruler"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"
rubric = "Prefer correct, directly responsive answers. Penalize unsupported claims."
temperature = 0.0
max_concurrency = 4
timeout_secs = 120
max_retries = 2
cache_path = "artifacts/grpo-judge-cache.jsonl"
```

`base_url`, `model`, and `api_key_env` are required (the default environment
variable is `OPENAI_API_KEY`). `temperature` is optional and must be
non-negative; use `0.0` for the most repeatable verdicts when the endpoint
supports it. `max_concurrency`, `timeout_secs`, and `max_retries` bound request
pressure, latency, and transient-error retries. `cache_path` is relative to
the training configuration file and caches verdicts, which is especially useful
while iterating on the same prompt set. `rubric` provides run-wide criteria; a
prompt JSONL record may supply its own `rubric`, which takes precedence.

RULER's `strategy` selects the comparison shape:

| Mode | Requests per group | Use it when |
| --- | ---: | --- |
| `auto` (default) | One, or one per chunk | General default: listwise when the group fits the context budget, otherwise anchored chunks. |
| `listwise` | 1 | Groups are short and cost matters. It fails rather than truncating when the group cannot fit. |
| `chunked` | One per chunk | Groups are too long for a single request but should still receive listwise scores. |
| `pairwise` | Up to `max_pairs` (twice that with `both_orders`) | Highest-quality relative signal or very long completions; each request contains only two candidates. |

For example, pairwise judging with an explicit comparison rubric is configured
as follows:

```toml
[grpo.judge]
pairwise_rubric = "Choose the answer that is more correct, complete, and concise. Return a tie when neither is better."

[grpo.judge.strategy]
mode = "pairwise"
max_pairs = 12
both_orders = true
aggregation = "bradley_terry"
```

Pairwise scores are win rates by default (a tie counts as half a win).
`aggregation = "bradley_terry"` is most useful when `max_pairs` truncates the
comparison schedule: it accounts for which opponents a completion beat.
`both_orders = true` is the default and evaluates each pair in both presentation
orders; it doubles judge requests but detects position bias, treating a
disagreement as a tie. `pairwise_rubric` is separate from `rubric` because a
pairwise request asks for a winner, not an absolute score.

For forced chunking, retain the shared anchor so separate requests remain on
one scale:

```toml
[grpo.judge.strategy]
mode = "chunked"
anchor = true
```

### Context, failures, and a command judge

RULER context limits are character budgets, which makes them independent of
the provider's tokenizer. Long messages are elided from the middle while
preserving both ends:

```toml
[grpo.judge.context]
max_request_chars = 60000
max_trajectory_chars = 8000
max_message_chars = 2000
head_ratio = 0.4
```

`auto` changes to chunked judging when `max_request_chars` would be exceeded.
The per-message and trajectory limits must not exceed their enclosing budgets;
`head_ratio` is in `[0, 1]`. Optional `[grpo.judge.compaction]` can have an LLM
summarize the middle of especially long transcripts (`trigger_chars`,
`target_chars`, and `keep_last`), but it adds extra model calls and is normally
unnecessary for single-turn GRPO.

When a judge request fails, `judge_failure = "drop_group"` (the default) drops
the entire affected completion group so scores from different scales are never
mixed. `judge_failure = "fail"` stops immediately. In either case,
`max_judge_dropped_fraction` (default `0.5`) limits the fraction of groups an
update may lose under `drop_group`; lower it when the judge is expected to be
reliable. Monitor `judge/dropped_group_fraction`, `judge/score_mean`,
`judge/score_std`, and, for pairwise mode, position-disagreement metrics.

If an external program should judge instead, use the same command backend as
agentic runs:

```toml
[grpo.judge]
type = "command"
command = ["python3", "judge.py"]
timeout_secs = 30
```

The command still judges groups through the shared judge protocol; retain
`judge_weight` and failure controls in `[grpo]`. Use it for a local model or a
domain-specific judge when an OpenAI-compatible RULER endpoint is not the right
integration.

## Optional GRPO controls

```toml
[grpo.overlong_penalty]
buffer_tokens = 16
max_penalty = 1.0

[grpo.kl_schedule]
warmup_updates = 10
target = 0.05

[grpo.dynamic_sampling]
max_resample_factor = 3
```

`overlong_penalty` softly penalizes the final part of a completion budget.
`mask_truncated` instead removes budget-truncated completions from a group;
choose one based on whether the reward can judge incomplete answers.

`kl_schedule` requires `kl_coefficient > 0` and supports warmup plus an optional
adaptive target. `dynamic_sampling` draws replacement prompts after zero-signal
groups are removed, up to `prompts_per_update × max_resample_factor` candidates.

## Observing rollouts

`[observe]` exports selected updates' prompts, completions, rewards before and
after `overlong_penalty`, advantages and exclusion causes, and writes a viewer
next to them. An agentic run exports its conversations, tool calls, step
rewards and judge explanations the same way. See
[Observing rollouts](./observe).

```toml
[observe]
directory = "artifacts/observe"
every = 10  # rollouts for updates 10, 20, …; summaries for every update
```

## Sharing the GPU

`training.max_gpu_duty_cycle` bounds the fraction of wall time the trainer
spends waiting on GPU work it submitted, leaving the rest to another workload:

```toml
[training]
max_gpu_duty_cycle = 0.5
```

Generation, reference and behaviour scoring, the per-epoch re-scoring and the
optimizer are all covered, so an update spends its idle time where it spends
its compute rather than only at the optimizer step.

It frees **compute time, not device memory**: everything the run has allocated
stays allocated while it sleeps. Enabling it also costs the decode pipelining
once, before any sleep, so it is worth its overhead at `0.75` and below. See
[Configuration reference](../reference/configuration) for the full contract.

## Container-backed agentic GRPO

Single-turn GRPO uses `[grpo]` and an external reward command. It does not need
the `container` Cargo feature. Multi-turn agentic GRPO uses
`run.algorithm = "agent_grpo"` and can execute tools inside a container for
each trajectory; that configuration does require the feature.

Build and run the container-enabled binary:

```bash
cargo build --release --features container
cargo run --release --features container -- train agent.toml
```

The container daemon must be running. Docker Desktop, Colima, and Podman in
Docker-compatible mode are supported; set `DOCKER_HOST` when the daemon uses a
non-default endpoint.

A minimal container environment declaration is:

```toml
[run]
algorithm = "agent_grpo"

[agent]
scenarios = "scenarios.jsonl"

[agent.environment]
type = "container"
profile = "python"
image = "python:slim"
allow_network = false

[agent.environment.limits]
cpus = 1.0
memory_mb = 1024
pids = 256
exec_timeout_secs = 30
max_output_bytes = 65536

[agent.environment.pool]
max_live = 8
min_idle = 0
reuse = "never"
max_leases_per_container = 32
```

`profile = "python"` selects the built-in Python tool set. Use `image` to
replace the profile image, preferably with a digest-pinned image for a
reproducible run. Network access is disabled by default. The container image
must provide the commands required by the selected tools and by the timeout
wrapper.

The container pool limits how many environments are live at once. A GRPO group
may therefore wait for a lease when `max_live` is lower than the number of
trajectories requested by an update. `reuse = "workspace"` recycles a cleaned
container; `reuse = "never"` destroys it after each episode.

## GRPO parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `grpo.prompts` | required | Chat JSONL prompt file. |
| `grpo.reward_command` | required | Executable argv array returning scalar rewards. |
| `grpo.reward_mode` | `persistent` | `persistent` or `oneshot`. |
| `grpo.reward_timeout_seconds` | `300` | Batch deadline in seconds. |
| `grpo.updates` | required | Number of grouped rollout updates. |
| `grpo.prompts_per_update` | required | Distinct prompts in one update. |
| `grpo.group_size` | required | Completions per prompt; `2..256`. |
| `grpo.grpo_epochs` | required | Optimizer passes over one grouped batch. |
| `grpo.clip_range_low` | required | Lower ratio clip, `(0, 1)`. |
| `grpo.clip_range_high` | required | Upper ratio clip, `(0, 1)`, at least the lower range. |
| `grpo.kl_coefficient` | required | Non-negative fixed-base KL coefficient. |
| `grpo.mask_truncated` | `false` | Exclude budget-truncated members before baselining. |
| `grpo.baseline` | `mean` | `mean` or `leave_one_out`/`rloo`. |
| `grpo.prompt_order` | `sequential` | `sequential` or `shuffled`. |
| `grpo.max_stalled_updates` | `25` | Stop after this many zero-signal updates; `0` disables the stop. |
| `grpo.sampling.temperature` | required | Must be `1.0`. |
| `grpo.sampling.top_p` | required | Must be `1.0`. |
| `grpo.sampling.max_new_tokens` | required | Completion budget and loss denominator. |
| `grpo.sampling.seed` | required | Sampling seed. |
