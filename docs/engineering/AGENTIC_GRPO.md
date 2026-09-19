# Agentic GRPO

Retrograd can collect multi-turn trajectories in Rust, execute MCP tools,
score groups with a relative RULER judge, and train the resulting token masks
without exporting policy generation or logprobs to another runtime.

## Guarantees

- Only policy-action tokens are trained. Initial context, user turns and MCP
  results remain in the conditioning sequence with zero loss weight.
- `old_logprobs` are re-scored by teacher forcing after the complete
  trajectory, using the same GGUF tokenizer/template and policy adapter.
- The policy actor and optimizer share exclusive ownership of one `Trainer`;
  no update can overlap collection from the same batch.
- Judge failures never become neutral rewards. A group is dropped or the run
  fails according to `judge_failure`. A run with no judge at all reports a group
  its environment left unscored the same way, through the same policy.
- A group is collected in lockstep - one decode batch per turn, shared prompt
  at turn 0 and continuous batching afterwards - and produces token for token
  what the same rollouts run one at a time produce. Batching is a throughput
  decision, never a statistical one.
- Each trajectory owns its environment. One instance is created per rollout,
  `reset` before the first turn and closed at the end - on the failure and
  deadline paths too. Members of a group can therefore not contaminate each
  other's state, which is what the relative baseline assumes.
- A rollout that fails or overruns its deadline costs its own trajectory, not
  the update. Truncated, failed and unscored members are counted together
  against `max_dropped_fraction`, once, over the number of rollouts the update
  asked for; past it the update stops. A group that falls under two surviving
  members is dropped whole, since it no longer has a relative baseline. An
  update left with no baseline at all normally ends the run; under
  `skip_empty_updates` it is logged and skipped instead (see below).
- **Sandbox isolation is a property of the runtime, not of a cleanup routine.**
  The default reuse policy destroys a container at the end of its episode. Under
  `reuse = "workspace"` the container is recycled, but the workspace is erased
  and the episode's leftover processes are killed before the next lease - and a
  cleanup that fails destroys the container instead of returning it to the pool.
  In doubt, destroy: the cost is one container creation, and the alternative is a
  group baseline that silently measures contamination.
- **A verifiable reward beats an opinion.** When a scenario declares a `verify`
  command, the environment grades the episode and the group never reaches the
  judge - every member carries a reward, including `0.0` for the ones where
  nothing happened, so which mechanism scored the update does not depend on how
  its members behaved.
- **An observation's text is a deterministic function of the action.** Those
  bytes become training tokens, so a sandbox never lets a container id, a
  hostname, a pid, a host path, a duration, a timestamp or an unsorted listing
  reach the transcript: every episode of every group sees the same `/work`, and
  listings are sorted inside the sandbox. Without this, two members of a group
  receive observations that differ by noise, and the relative baseline measures
  the noise. This is checked by a test, not left as a style rule.

## How tools are rendered

A model trained with tools declares them in its own chat template: a tool
catalog, a `tool` role for what comes back. When the template does, Retrograd
hands it the catalog and lets it frame the observations - the format the model
actually saw during pre-training. When it does not, the catalog is described in
the system prompt and the calls are read back out of the response by the
Hermes/Qwen `<tool_call>{...}</tool_call>` convention. Support is *probed*, once
per run, rather than pattern-matched: a template that mentions `tools` and then
drops it counts as blind. The probe renders the whole shape a rollout will later
ask for - a sentinel catalog, an assistant turn carrying only its sampled
`content`, and an observation under the `tool` role - and requires both sentinels
to come out the other side. Accepting the catalog is not enough: some templates
(`openai-gpt-oss` among them) name a tool result from the *structured*
`tool_calls` of the preceding assistant turn and raise when there are none, which
the renderer deliberately never sends (see below). A catalog-only probe would
declare those compatible and then fail on the second turn of every rollout; they
belong on the prompt-written fallback.

Two things stay identical on both paths:

- **The assistant turn is the policy's own bytes.** Its `tool_calls` are never
  handed back to the template, even though the trajectory carries them: those
  calls are already inside the sampled content, and they are the tokens being
  trained on. A template re-rendering its own version of them would append a
  second copy, or replace the sampled text with a re-serialization that no longer
  matches it. Native rendering covers what the model does *not* write.
- **Parsing stays convention-based.** Reading tool calls back out of a response
  is a property of the model's output, not of its template, so the parser is
  unchanged by native rendering.

Before a second policy turn, the engine verifies that the chat template renders
the prior tokens as an exact prefix. A non-prefix-stable template fails
explicitly, because continuing would invalidate the collected logprobs.

## Python

```python
from retrograd import (
    AgenticGRPOConfig,
    LoraConfig,
    McpServer,
    RulerJudge,
    Scenario,
    Trainer,
)

config = AgenticGRPOConfig(
    scenarios=(Scenario("math-1", "Use the calculator: 37 + 5"),),
    judge=RulerJudge(
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_key_env="OPENAI_API_KEY",
        # Optional: "pairwise" compares two trajectories per request instead of
        # ranking the whole group. See "Judging" below.
        mode="auto",
    ),
    mcp_servers=(McpServer(name="calc", command=("python", "calculator_server.py")),),
    updates=10,
    scenarios_per_update=2,
    group_size=8,
    max_turns=6,
    max_new_tokens=256,
)

with Trainer("model.gguf", lora=LoraConfig()) as trainer:
    metrics = trainer.fit_agentic_grpo(config, callback=print)
    trainer.save_adapter("agent.gguf")
```

Scenario JSONL is strict and uses one record per line:

```json
{"id":"math-1","system":"Use tools when useful.","user":"What is 37 + 5?","metadata":{}}
```

The low-level `Trainer.train_grpo_batch()` accepts `TrainSequence` objects for
collectors outside Retrograd. A sequence contains full tokens, exact
behavior-policy logprobs, a token-level policy mask, reward, group id and
optional intermediate returns.

Calling it repeatedly - one call per update, which is the point - needs
`scheduler_total_rollouts`: the number of sequences the *whole* run will train,
not this batch. The native scheduler step accumulates across calls, so a
per-batch horizon would put the second update at the end of the decay and drive
the learning rate to zero. It is optional only under `scheduler="constant"`,
where the horizon is unused; any other scheduler without it is rejected.

## Native TOML runner

An agentic run is one `run.algorithm` of the single configuration document, not
a second schema: `[model]`, `[lora]`, `[training]` and `[metrics]` mean here
exactly what they mean for SFT, PPO and GRPO, and `[agent]` is this algorithm's
own section. It is the same binary, too - the container pool and the MCP
transports are compile-time features of it, not separate programs.

```sh
cargo run -- train agent.toml                       # MCP tools, HTTP environments
cargo run --features container -- train agent.toml  # plus the container pool
```

```toml
[run]
algorithm = "agent_grpo"

[model]
path = "model.gguf"
device = "auto"

[output]
path = "agent.gguf"

[lora]
rank = 8
alpha = 16.0
targets = ["blk.*.attn_q.weight", "blk.*.attn_v.weight"]

[training]
ctx = 2048
micro_batch = 64   # forward/backward width: the activation-memory lever
# `gradient_accumulation` is pinned to ctx / micro_batch = 32 for a rollout
# algorithm - one optimizer step per rollout - so it is left out.
lr = 1e-5
max_grad_norm = 1.0

[agent]
scenarios = "scenarios.jsonl"
updates = 10
scenarios_per_update = 2
group_size = 8
# Optimizer passes over one update's rollouts. Spelled in full because
# `training.epochs` is a different quantity - passes over an SFT dataset.
epochs_per_update = 4
max_turns = 6
# Per assistant turn. `[grpo.sampling].max_new_tokens` is the single-turn
# equivalent; a turn is not a completion, hence the two names.
max_new_tokens_per_turn = 256
max_rollout_secs = 300
# `false` when the world, not the policy, decides the episode is over; see
# "A turn that calls no tool".
end_on_no_tool_call = true
# Consecutive turns without a valid call before the trajectory is cut as a
# truncation; 0 never cuts. Only matters under `end_on_no_tool_call = false`.
max_failed_turns = 0
clip_range_low = 0.2
clip_range_high = 0.28
judge_failure = "drop_group"
max_dropped_fraction = 0.5
skip_empty_updates = false # see "An update with nothing to train on"
truncation = "drop"        # or "min_reward"; see "What truncation costs"
# See "Prompt and template settings"; applies to training and evaluation.
system_suffix = ""
seed = 42

# Optional, and an addition rather than a default: a task whose environment
# grades its own steps needs no judge, and stacking an LLM opinion on top of a
# verifiable reward would only add noise. What a run cannot have is neither -
# a document with no judge and no [agent.environment] is refused at load.
[agent.judge]
type = "ruler"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[[agent.mcp_servers]]
name = "calc"
command = ["python", "calculator_server.py"]
tool_timeout_secs = 30

[[agent.mcp_servers]]
name = "search"
url = "https://example.test/mcp"
required = false
denied_tools = ["write_*"]
```

HTTP MCP servers use `url` and optional `headers` instead of `command`. Secrets
should remain in environment variables or server-side authentication; the
RULER API key is never accepted directly in TOML.

The common exchange format can be used directly with `mcp_config = "mcp.json"`
(or a list). `${VAR}` references in MCP environment variables and headers are
expanded once, when the runner merges the files; missing variables fail closed
and secret values are never serialized into a catalogue or manifest. Merging is
deliberately not part of parsing: a planner or a server reads a document written
for a machine that is not its own, where those files and those variables need
not exist. A matching `[[agent.mcp_servers]]` entry may contain policy only
(`stateless`, filters, timeouts): its transport comes from `mcp.json`, and so
does every field the entry leaves at its default - `cwd` and `timeout` included.

Inspect the exact canonical catalogue without loading a model, then optionally
generate an immutable scenario corpus from it:

```sh
retrograd tools list agent.toml --json
retrograd tools list agent.toml --no-connect
retrograd scenarios generate agent.toml --dry-run
retrograd scenarios generate agent.toml
```

Scenario generation is configured under `[agent.scenario_generation]` with
`model`, `base_url` and `api_key_env`. It writes `agent.scenarios` plus a
`.manifest.json` sidecar containing prompt, catalogue and corpus hashes.
Training verifies that sidecar when present; generation is never triggered
implicitly by `train`.

The judge and MCP tables deserialize straight into the crate's `JudgeConfig` and
`McpServerConfig`, the same types the Python binding and any embedder use - the
TOML runner declares no schema of its own, so an option added to the crate is
available in every frontend at once.

### Evaluating and checkpointing

`[evaluation]` and `[checkpoint]` are the shared sections, and they mean here
what they mean everywhere else. Both act at an **update boundary** - the one
quiet moment in an agentic run, when the update's rollouts, judging and
optimizer step have completed and the next update has not started. `--resume`
restarts at the update the checkpoint completed; the scenario selection and
every seed are derived from the update index alone, so a resumed run replays
exactly the schedule the interrupted one would have had.

```toml
[evaluation]
data = "held-out.jsonl"     # scenarios, same format as agent.scenarios
every_iterations = 5        # in updates
patience = 3                # stop after 3 evaluations with no improvement
max_examples = 4            # cap: one full trajectory per scenario

[checkpoint]
directory = "checkpoints"
mode = "steps_and_best_eval"
every_steps = 200
```

**What an agentic evaluation measures - and why it is not the judge.** Every
RULER strategy (listwise, chunked, pairwise) scores the members of a group
*against each other*. Those scores are renormalized inside every group, so their
mean is not comparable from one update to the next: a curve built on them would
move with the normalization, and `patience` would stop the run on noise. What is
comparable is the reward the **environment** puts on a trajectory - a `verify`
command, a test suite, a task's own grading - which measures the same quantity
at every update. It is the same total the optimizer trains on, so `eval/*` and
the training reward are in one unit.

A held-out set whose scenarios declare no `metadata.env.verify` is therefore
refused **at startup**, not at the first evaluation an hour of rollouts later -
for a sandbox environment, which is what `verify` grades. An HTTP environment
returns its reward from `/step` and declares no verify command, so nothing in
the document says at startup whether it grades; there the refusal comes from the
first pass instead.

Partial grading is refused too, at every pass: a mean over a moving subset of
the held-out set is not a measurement. The scenarios that drop out of it are the
ones the policy failed or left ungraded, so averaging over the survivors would
report a *better* number for having failed them - and that number is what
`best_eval` and `patience` act on. A pass covers the whole set or it is an
error saying how much of it was lost.

Cost is the reason `max_examples` exists: an evaluation is one complete
trajectory per scenario - containers, turns, tool calls - so an unbounded
held-out set evaluated every other update can cost more than the training it
measures. The cap keeps the scenarios evenly spaced across the file rather than
taking the first N, which on a set ordered by topic would not be a sample of it.

## Rollout health

Every progress event carries the collection's own diagnostics next to the
optimizer's. The failure fractions are over the rollouts the update asked for,
including the ones that never produced a trajectory; the others are over what
was actually collected, since a rollout that died was never truncated and never
called a tool:

| Metric                       | What it says                                       |
| ---------------------------- | -------------------------------------------------- |
| `agent/truncated_fraction`   | of the collected trajectories: hit a token budget, a turn limit, or the deadline |
| `agent/failed_fraction`      | died mid-rollout, all causes                        |
| `agent/failed_tool`          | …of which: the tool provider was unreachable        |
| `agent/failed_policy`        | …of which: the policy or the runtime failed         |
| `agent/lost_fraction`        | collected, then lost with a group under two members |
| `agent/tool_error_fraction`  | tool calls that answered with an error              |
| `agent/turns_per_traj_mean`  | policy turns per trajectory                         |
| `agent/tokens_per_traj_mean` | trajectory length                                   |

A run with a sandbox pool adds the counters that decide whether the pool is
worth its complexity. They are part of the delivery, not an afterthought: they
justify the design of the pool or contradict it.

| Metric                          | What it says                                    |
| ------------------------------- | ----------------------------------------------- |
| `env/container_create_ms`       | mean cost of creating one container              |
| `env/acquire_ms_mean`           | what an episode actually waits for its sandbox   |
| `env/pool_reuse_fraction`       | leases served from an idle container             |
| `env/live_containers_max`       | the high-water mark `max_live` is supposed to cap |
| `env/exec_timeout_fraction`     | commands killed by `exec_timeout`                |
| `env/sandbox_broken_fraction`   | leases whose sandbox died - trajectories lost, not observations |

`max_rollout_secs` (default 300, `0` disables) bounds one rollout end to end,
over and above each tool's own timeout. It is not sampled between turns but
*applied to every stage* - creating the environments, the tool listing, `reset`,
each render, each decode, each environment step, the final rescore - so a single
call that never returns cannot outlive it. Overrunning truncates the trajectory
rather than failing it, so it flows into `agent/truncated_fraction`, with two
exceptions where there is nothing to keep: an overrun before any member has
sampled a token (during the creation, the listing, `reset` or the opening
render) is a group-wide setup failure, and one during the final rescore fails
the members it hits, since a trajectory without its old logprobs is not a
trajectory.

### What a turn costs, and why it used to grow with the square of the turn count

A rollout hands the runtime the whole trajectory on every turn: the prompt, the
tokens sampled so far, the observations the environment answered with. Until
this was fixed the runtime cleared the generation context's KV cache at the
entry and the exit of every call, so turn *t* re-decoded everything turns
`0..t-1` had already decoded. A trajectory of `T` turns of `L` tokens paid
`Σ (P + t·L)` positions of prefill - quadratic in `T`, against the linear decode
everyone budgets for.

At one or two turns that is invisible, which is why it stood: a
question-answering run's turn 0 is a *shared* prefill, one prefix for a whole
group. An agentic run is the other regime. A Sokoban trajectory of 28 moves at
~200 tokens a turn pays about 43 000 positions of prefill per member instead of
6 000, and a group of 32 turns that into 1.4 million.

The generation context now keeps each sequence's cells between calls and decodes
only what a prompt added. Three rules make that safe rather than merely fast,
and each one is a place where a cache like this goes wrong quietly:

- **A hit is an exact extension, never a partial rewind.** When a prompt
  diverges from what a slot holds, the whole slot is dropped and re-decoded.
  Keeping the matching head would mean removing a range from the middle of a
  sequence, and that is not available for the recurrent half of a hybrid
  architecture - where dropping a whole sequence always is.
- **Every weight change forgets everything.** Cells decoded under the adapter of
  update *N* do not describe update *N+1*. The invalidation sits on the four
  training entry points and on the adapter's create/load/enable, so a rollout
  after a step starts cold.
- **The last sampled token of a turn is not resident.** A token is decoded to
  produce the next one, so the token a turn stopped on was emitted and never fed
  back. Recording it would start the next prefill one position past what the
  cache holds, which llama.cpp refuses outright.

What the cache does not change is the decode step itself, and on a long
lockstep rollout that is now the bill. The generation context is one unified
KV cache, so every attention row of a decode step spans the cells of *every*
resident trajectory, finished or not, and the step's cost grows with that span:
measured on a 230M hybrid model, a step over 8 sequences holding 16 000
positions took ~5 ms, and a step over 30 sequences holding 60 000 took ~18 ms.
A wave also lasts as long as its longest member's turn, so a single member that
rambles to `max_new_tokens_per_turn` makes every other member wait for it. The
two levers are therefore the turn budget and the per-turn token budget, not the
cache; a turn cut by the latter is a truncation, and `truncation` prices it.

Three series report it, and the pair to read is the first two:
`generation/prefill_reuse_fraction` - resident tokens over requested ones, which
a single-turn run keeps near zero and a healthy multi-turn one drives toward
one - against `generation/prefilled_tokens`, what was actually decoded.
`generation/kv_eviction_fraction` separates the one failure the first two cannot
name: a cache that works and does not fit, because the update has more live
trajectories than `generation_concurrency` gives the context sequences. That is
the only lever for it.

`RETRO_GENERATION_PREFIX_CACHE=0` restores the previous behaviour exactly - one
clear and one full prefill per call - which is what makes it the reference the
reuse path is compared against on a given model. Expect the sampled tokens to
differ between the two: splitting a prefill across launches changes the order of
its floating-point reductions, the same class of difference
`fast_generation_context` already documents. It reaches the optimizer as a
different sample, not as a wrong gradient - GRPO's importance ratio comes from
the exact teacher-forced rescore, not from what generation reported.

### A turn that calls no tool

A generation the parser reads as content, with no call and no malformed call in
it, is by default the policy's answer: the trajectory is complete and the turn
loop stops there. That is the right reading for a run whose scenarios are
questions, and it is the only one available to a run with no tool catalog at
all.

It is the wrong reading for a world with a terminal state of its own, and wrong
in a way that shows up as a sign rather than as a loss. A trajectory that ends
on its first unparsed turn has banked no step rewards; every sibling that
engaged the environment has banked whatever the environment charged it for
moving. Against a group centered on its own mean, the member that refused to act
is the one with the positive advantage, so GRPO trains the policy out of calling
tools - precisely on the runs where a small model already struggles to hold its
family's call format.

`end_on_no_tool_call = false` reads the same turn as a wasted one instead. It
comes back as an error observation, in the same shape a malformed call takes -
one `ToolResult` with `is_error`, no environment step, no reward - and the
episode continues until the environment says `done` or the turn budget runs out.
The flag has no effect on a run that declared no tools: there, telling a policy
to call something would spend every turn of every trajectory on a tool it was
never given.

It is not free. A policy that calls nothing now decodes `max_turns` turns
instead of one, so `timing/rollout_seconds` grows with the fraction of turns
that parse nothing. Watch `agent/tool_calls_per_turn`, not the reward: it is the
quantity the flag exists to keep from collapsing, and it moves first.

`max_failed_turns` bounds that price. After N consecutive turns with no valid
call - prose, or a call the parser refused - the trajectory is cut, and cut the
way a turn-budget overrun cuts it: `truncated`, so `truncation` prices it, and
under `min_reward` it lands at the bottom of its group. That is where a member
that never engaged the environment was going to end anyway; the flag only gets
it there after N turns instead of after `max_turns` or the context limit, which
on a long lockstep rollout is most of the decode. The count is consecutive, not
total: one valid call resets it, because a policy that is calling tools is
playing, however badly. It changes no sign - a group whose every member is cut
still has no variance and is still dropped - so it makes a run that is not
learning cheaper to diagnose, not more likely to learn.

### An update with nothing to train on

Two thresholds guard an update, and they answer different questions.
`max_dropped_fraction` asks whether too much of the update disappeared - a judge
that refuses every group, an environment that fails half the rollouts. The
second is structural: fewer than two trainable trajectories left is not a small
update, it is no baseline at all, so there is nothing to center and nothing to
step on, whatever the threshold says.

By default that ends the run, because an update with nothing in it usually means
something upstream stopped working and a run that continues quietly just burns
its budget. `skip_empty_updates = true` (default `false`) makes that case skip
the optimizer step instead: the update is logged at `warn` with the same
accounting the failure would have reported, its scenarios count as consumed, and
the run continues to the next update - checkpointing and stopping still happen at
the boundary, so a run whose updates all come back empty stays interruptible.

The option covers *only* the no-baseline case. An update that kept two or more
trainable trajectories and still blew past `max_dropped_fraction` remains a
failure: that is the "the judge or the environment is broken" signal, and a skip
would hide it.

### Prompt and template settings

```toml
[agent]
system_suffix = "Answer with one tool call and nothing else."
template_variables = { enable_thinking = false }
```

`system_suffix` appends text to every training and held-out system message,
creating one if absent. `template_variables` supplies values to the model's
chat template; available names and defaults depend on that template.
Both default to empty and are included in the checkpoint resume signature.

### What truncation costs

`truncation` decides what happens to a member that ran out of budget
mid-response:

| Value | Effect |
| --- | --- |
| `drop` (default) | it never reaches the optimizer, and counts against `max_dropped_fraction` |
| `min_reward` | it trains with the smallest **total** reward of its group |

Neither branch is free, which is why this is a setting and not a fix. A partial
response cannot be judged as a completed trajectory, so `drop` is the honest
reading of the reward - but dropping is a *selection on length*: the model never
gets a negative signal for overrunning, and a task that systematically overruns
leaves the gradient entirely instead of being learned out of. `min_reward` keeps
the member at the bottom of its own group, so overrunning costs something, at the
price of a reward nobody measured. The minimum is over totals (final reward plus
step rewards), which is the quantity GRPO centers the group on; a truncated
member that had already banked step rewards therefore gets a negative final
reward so that its *total* lands on the group minimum.

Read `agent/truncated_fraction` before choosing. Past roughly 20% the training
distribution is no longer the one the scenarios describe, and the choice above
stops being academic.

### Loss normalization is a learning-rate scale

The GRPO loss is divided by a **constant**: the trajectory's token budget,
`min(max_turns × max_new_tokens, max_trajectory_tokens)` - 3072 with the
defaults. This is Dr-GRPO's normalization and it is deliberate. Dividing by the
realized length instead would make a token's weight depend on how long its own
trajectory happened to be, which is a length bias in the gradient: long
trajectories systematically demoted, short ones promoted, for reasons unrelated
to reward. The single-turn sampler normalizes by `sampling.max_new_tokens` for
exactly the same reason.

The consequence to keep in mind: **the rollout limits set the scale of the
gradient**. A trajectory of a few hundred trained tokens under a 3072-token
budget is divided by roughly ten, and halving `max_turns` doubles the gradient
without `lr` being touched. Re-tune `lr` when the limits move - that coupling is
the price of a length-unbiased normalizer, not an accident.

### Reading the trajectories

The metrics count failed rollouts; `[observe]` shows the trajectories that were
collected, including those excluded from training. Failed rollouts have no
trajectory to export. Each collected trajectory is exported with its conversation, tool calls, step
rewards attached to the messages the environment returned them with, and the
judge's explanation. The engine records a member's position, seed and
step-to-message map on the trajectory while collecting it
(`Trajectory::provenance`), so the export never has to re-derive them after
truncated members are withheld, groups are dropped or `MinReward` puts members
back. The policy actor publishes the advantages and the effective mask before
the epochs, and the outcome after them. See
[Observing rollouts](/training/observe).

## Judging: comparison shape and context budget

A group is scored by one of four strategies, selected with `strategy.mode`
(TOML/JSON) or `RulerJudge(mode=…)` (Python):

| Mode | Requests per group | When it is the right one |
| --- | --- | --- |
| `auto` (default) | 1, or one per chunk | General use: one listwise request, falling back to anchored chunks when the group exceeds the context budget |
| `listwise` | 1 | Short trajectories, cheapest judging; **fails** if the group overruns the judge's window |
| `chunked` | one per chunk | Long trajectories, when listwise scoring is still wanted |
| `pairwise` | up to `max_pairs` | Highest-quality signal: a request never holds more than two trajectories, so trajectory length stops being a constraint |

Pairwise verdicts become scores by win rate (ties count as half a win), which
keeps them in `[0,1]` and invariant to how many comparisons a trajectory got.
The schedule always covers every trajectory at least once - a ring
`(0,1), (1,2), … (n-1,0)` first, then remaining pairs in order - so raising
`max_pairs` only adds comparisons and never invalidates cached ones. A failed
comparison costs its two participants one match, not the group.

Set `aggregation = "bradley_terry"` to fit strengths instead, projected into
`[0,1]` within the group. It is worth its cost exactly when the schedule is
truncated: win rate cannot tell beating the group's best from beating its worst,
and a truncated schedule does not give everyone the same opponents. The default
stays `win_rate` until the comparison has been measured on a real judge.

### Position bias

The first bias of an LLM-as-a-judge is position: the same trajectory scores
differently depending on where it appears. Two mechanisms address it, and
neither is optional in listwise/chunked mode:

- Listwise and chunked requests present trajectories in a permutation derived
  from `(group_id, request index)` - deterministic, so the response cache and
  run reproducibility are untouched - and scores are put back in trajectory
  order. Without it a systematic offset on member 0 survives the GRPO group
  baseline, because rollout order is arbitrary but *stable* across updates.
- Pairwise `both_orders` (default `true`) judges each pair in both directions
  and keeps only the agreements; a contradiction counts as a tie and feeds
  `judge/position_disagreement`. **This doubles the request count** - budget it
  next to `max_pairs`, since pairwise judging is already the run's main API
  expense. What it buys is the one number that says whether the judge is reading
  the trajectories or their positions; near 0.5 it is answering at random.

### Rubric, environment state, degenerate groups

A scenario that declares `metadata.rubric` is judged with it, falling back to the
run-wide `rubric` otherwise: task-specific criteria are worth far more than a
generic wording, and the scenario is the only place that knows them. In pairwise
mode the criteria are *appended* to the pairwise rubric rather than replacing it,
which would ask for a per-trajectory score in a two-way comparison.

When an environment reports a terminal state (`EnvState::summary` - typically
`git diff`), it is shown to the judge alongside each transcript, controlled by
`context.include_env_state` (default `true`). For a code task this is the main
lever on reward noise: the judge stops grading the dialogue and grades the
result.

A group whose members all receive the same score has a zero advantage
everywhere: it consumed a rollout, a judge request and an optimizer slot and
contributed to no gradient. It is reported as `judge/degenerate_group_fraction`.
`drop_degenerate_groups` (default `false`) discards them - off by default
because dropping them changes how many groups an update trains on, and because
such groups then count against the cap on unscored trajectories.

Chunked scoring repeats the group's first trajectory in every chunk (`anchor`,
on by default) and shifts each chunk so that trajectory lands on the same value.
Without it, independent requests are not on a common scale and GRPO would read
a judge's per-request generosity as signal.

Prompt size is bounded by `context`, in characters (the judge's tokenizer is
unknown to us, and a character budget bounds tokens for any tokenizer):

```toml
[agent.judge.context]
max_request_chars = 60000     # a whole request; over it, `auto` chunks
max_trajectory_chars = 8000   # one trajectory; over it, middle messages are dropped
max_message_chars = 2000      # one message; over it, both ends are kept
head_ratio = 0.4              # share of an elided message kept from the head
```

Elision keeps both ends of a message and marks the cut, because tool output
carries its conclusion at the end and reasoning its intent at the start. The
shared opening of a group is sent once, not per trajectory. Watch
`judge/prompt_chars_max` and `judge/elided_fraction`: a judge reading mostly
elided trajectories is judging summaries, not behaviour.

An optional `compaction` table goes one step further and has an LLM summarise the
*middle* of each transcript:

```toml
[agent.judge.compaction]
trigger_chars = 6000   # the group is compacted when its longest member exceeds this
target_chars = 1000    # budget for the summary, enforced by elision
keep_last = 4          # closing turns kept verbatim
```

Three properties are structural rather than tunable. Compaction is deterministic
(temperature 0, cached by content hash in the verdict cache), the decision is
taken **once per group** from its longest member - a member summarised harder
than its siblings would be handicapped for a reason that is not its performance,
straight into the GRPO advantage - and the opening turn and the last `keep_last`
turns are never summarised. Watch `judge/compacted_fraction` like
`judge/elided_fraction`.

Compaction *during* a rollout is out of scope: rewriting the context breaks the
"one trajectory = one growing sequence" invariant checked on every tool turn.

### Measuring the judge

A judge nobody measured is a hypothesis. `retrograd judge eval` scores a
file of reference fixtures with the judge from a training config - no model, no
GPU - and reports rank agreement with the labels plus its self-consistency over
two passes:

```sh
retrograd judge eval config.toml --fixtures fixtures.jsonl
```

One JSON object per line, a shared opening plus the candidates that answered it:

```json
{"scenario_id": "fix-parser", "rubric": "prefer fewer edits",
 "context": [{"role": "user", "content": "test_parse_empty fails."}],
 "candidates": [{"messages": [{"role": "assistant", "content": "…"}], "label": 1.0, "env_state": "diff …"},
                {"messages": [{"role": "assistant", "content": "…"}], "label": 0.0}]}
```

Only the *ordering* of the labels is used - GRPO only ever reads a score relative
to its group, so a judge that is uniformly generous is not wrong for our purpose.
The report gives Kendall's τ and Spearman's ρ against the labels, and the same
two against a second pass of the judge on the same input. Self-consistency bounds
agreement: a judge that disagrees with itself cannot agree with anything else,
and the gap between the two separates "the rubric is wrong" from "the judge is
noisy". The RULER cache is ignored during `judge eval`, or the second pass would
report perfect consistency for free.

## Tools

Stateless tools, shared by every trajectory. For anything with state, see
[Environments](#environments) below. Three provider kinds compose through
`CompositeToolProvider`, which merges tool lists once and refuses ambiguous names
at construction:

- `McpToolProvider` - MCP servers over stdio or streamable HTTP.
- `LocalToolProvider` - in-process tools implementing `LocalTool` (two methods).
  A deterministic calculator or a dataset lookup does not need a subprocess.
- any other `ToolProvider` implementation.

MCP servers accept `allowed_tools`/`denied_tools` patterns (`*` wildcard, deny
wins) and `required = false` to be skipped when unreachable instead of failing
the run. Skipped servers and filters that expose nothing are reported by
`McpToolProvider::warnings()`. A run where *every* server failed is an error,
not a silently toolless rollout.

```rust
let mcp = McpToolProvider::connect(vec![
    McpServerConfig::stdio("calc", ["python", "calculator_server.py"]),
    McpServerConfig::http("search", "https://example.test/mcp")
        .optional()
        .deny(["write_*"]),
])
.await?;
let tools = CompositeToolProvider::connect(vec![
    Arc::new(mcp),
    Arc::new(LocalToolProvider::new(vec![Arc::new(MyTool)])?),
])
.await?;

let outcome = AgenticRun::new(trainer, scenarios, config, training)
    .with_judge(judge) // optional: without it, only the environment's reward counts
    .with_tools(Arc::new(tools))
    .on_progress(&mut |progress| eprintln!("{}", progress.metrics.train_loss))
    .run()
    .await;
```

### Session tools - the extension point

A `ToolProvider` is stateless by construction, so it has nowhere to put "the
filesystem of *this* trajectory". `SessionTool` is the other half: the tool
receives the sandbox of the trajectory it is being called in.

```rust
#[async_trait]
pub trait SessionTool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, sandbox: &dyn Sandbox, arguments: Value) -> Result<ToolOutcome>;
}

pub struct ToolOutcome {
    pub content: String,
    pub is_error: bool,
    /// A step reward - how a `run_tests` tool grades the task without a judge.
    pub reward: Option<f32>,
    /// The tool declares the episode over (`submit`).
    pub done: bool,
}
```

`ToolOutcome` projects exactly onto the existing `StepOutcome`, so the rollout
engine knows nothing about any of this. Registration validates at build time -
empty names, duplicates and malformed input schemas are refused there, because a
name conflict discovered on the 300th rollout is a lost run:

```rust
let tools = ToolSet::builder()
    .with_profile(Profile::Python)          // the built-ins
    .with(Arc::new(MyOwnTool::new(config))) // yours
    .deny(["bash"])                         // same wildcard syntax as MCP filters
    .build()?;
```

One design constraint follows directly from the determinism guarantee: **a
tool's error text is part of the training distribution.** It is fixed, formatted
identically on every call, and tested for that. An error message that varies
teaches the model to react to noise.

## Environments

A `ToolProvider` is stateless and shared: one instance answers every member of
every group. That is right for read-only tools and wrong the moment the task has
state - a filesystem, a shell, a database. Two members of a group writing to the
same state contaminate each other, and a group-relative baseline over
contaminated members measures nothing.

An `Environment` is therefore per trajectory:

```rust
#[async_trait]
pub trait Environment: Send {
    async fn reset(&mut self, scenario: &Scenario, seed: u64) -> Result<Option<String>>;
    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome>;
    async fn tools(&self) -> Result<Vec<ToolSpec>>;
    /// Serializable snapshot of the terminal state. Read once, before `close`.
    async fn state(&mut self) -> Result<EnvState> { Ok(EnvState::default()) }
    async fn close(&mut self) -> Result<()>;
}
```

`state()` has exactly one consumer, the judge: for a code task what has to be
graded is the diff the trajectory produced, not the dialogue that produced it.
The rollout reads it before `close` - which is what releases the sandbox the
state is read from - and stores it under the member's `metadata.env_state`. A
failure there is a warning and nothing else, since the judge falls back to the
transcript; trading a complete rollout for a `git diff` would be the worse deal.

`AgenticRun::with_environments(factory)` takes an `EnvironmentFactory`;
`with_tools(provider)` is the same call over a `ToolProviderFactory`, so every
configuration written before environments existed behaves exactly as it did.

`StepOutcome` carries two things a tool result cannot:

- `done` - the environment declares the task finished. A third stop condition
  next to "the policy asked for no tool" and "the budget is spent", and unlike
  those two it is **not** a truncation: the trajectory is complete and trainable.
- `reward` - a step reward. It lands on the `ToolResult` step as an intermediate
  return, and it takes the whole group out of the judge's hands: a verifiable
  task should not be scored by an LLM. A group is scored by its environment or
  by the judge, never half by each - otherwise a judge reward would stack on top
  of step rewards.

`Environment::step` draws the line between a failed action and a broken world.
`Err` means the environment itself is unusable - a lost session, a server that
went away - and it costs the trajectory: the tokens already collected are
conditioned on a world that no longer answers, so scoring and training them would
train a rollout that never happened. An action that merely failed is `Ok` with
`is_error` set; that one enters the prompt as an observation the policy reads and
reacts to, and is counted by `agent/tool_error_fraction`. A stateless
`ToolProvider` has no world to lose, so `ToolProviderEnvironment` reports its
errors on the observation side - which is what the MCP provider already did with
its own timeouts and unknown tools.

`reset` may return an opening observation, appended to the scenario as a user
turn. Returning `None` - the usual case - is what keeps a group's turn 0 on a
single shared decode: distinct openings make the members diverge before their
first sampled token, so each one is rendered and decoded on its own line of the
continuous batch. Everything else is unchanged; the tool list is fetched once
per run, since it describes the environment kind rather than an instance.

### Sandboxed environments

`SandboxEnvironment` is the built-in one: the task is declared in
`scenario.metadata.env`, materialized in a sandbox at `reset`, acted on by
*session tools* that receive that sandbox, summarized by `state()` and released
at `close`.

```json
{
  "id": "fix-parser-3",
  "user": "The test test_parse_empty fails. Fix it.",
  "metadata": {
    "env": {
      "files": {"src/parser.py": "…", "tests/test_parser.py": "…"},
      "setup": ["pip install -e ."],
      "verify": {"command": ["pytest", "-q"], "reward_on_success": 1.0},
      "summary": ["git", "diff"]
    }
  }
}
```

The whole declaration is typed and validated in `prepare()` - once, at startup,
for *every* scenario - together with the image pull and the pool warm-up. A
malformed task or a missing image discovered on the 700th rollout is a lost run,
and it is the operator's own configuration either way. `setup` failing is an
`Err`, not an observation: the episode would start from a state the scenario does
not describe.

`verify` is what makes the reward verifiable rather than an opinion, and
`summary` is what the judge reads instead of the transcript. A scenario that
declares neither still works - the judge grades the dialogue, as before.

Two backends run that task:

| `type` | Where it runs | When |
|---|---|---|
| `container` | a pool of containers on the local Docker/Podman daemon | model-generated code |
| `local` | a `TempDir` on the host, no isolation | a task you trust, or testing the tools without a daemon |

`local` is refused unless `allow_unsandboxed = true` is typed by someone: it runs
model-generated code with the training process's privileges and network. It also
cannot hold the determinism rule above all the way - a `pwd`, a traceback or a
compiler diagnostic leaks the temporary directory's real name, which differs per
episode. The container is what actually solves that, by giving every episode the
same `/work`.

Tools come from a *profile* rather than a per-language code path: `python`
(`python:3.12-slim`, tools `bash`/`python`/`read_file`/`write_file`/`edit_file`/
`list_dir`/`grep`/`run_tests`/`submit`, `/opt/cache/pip` mounted), `typescript`
(`node:22-slim`, `node` instead of `python`, `npm test`, `/opt/cache/npm`), or
`custom` with an explicit image. A profile only announces the tools its image
actually has - advertising `python` on a Node image would teach the model to fail
on a call that could not have worked.

```toml
[agent.environment]
type = "container"
profile = "python"
image = "ghcr.io/me/py-tasks@sha256:…"   # optional; prefer a digest to a tag
allow_network = false

[agent.environment.limits]
cpus = 1.0
memory_mb = 1024
exec_timeout_secs = 30
max_output_bytes = 65536

[agent.environment.pool]
max_live = 8                  # a semaphore: a larger group serializes
min_idle = 2
reuse = "never"               # or "workspace"
max_leases_per_container = 32
```

Defaults are restrictive and all overridable: no network, read-only rootfs,
tmpfs `/work`, non-root, `cap_drop=ALL`, `no-new-privileges`, `AutoRemove`,
`memory == memory_swap`. Mounting the Docker socket and running as root are
refused whatever the configuration says. Two things worth writing down rather
than implying: **Docker is not a security boundary** against hostile code - for
untrusted code at scale the answer is an isolating runtime (gVisor, Kata), which
is what the `runtime` field of `ContainerSpec` exists for - and containers are
labelled with the run id so a reaper can clean up what a `SIGKILL` left behind.

The pool answers three separate problems: `max_live` bounds how many containers
exist at once, `min_idle` plus a warm-up **during the optimizer step** (the one
window where the environment side is idle) turns `env/acquire_ms_mean` from a
container creation into a lookup, and `reuse` decides whether isolation comes
from the runtime or from a cleanup routine. Watch `env/pool_reuse_fraction`,
`env/acquire_ms_mean`, `env/container_create_ms`, `env/live_containers_max`,
`env/exec_timeout_fraction` and `env/sandbox_broken_fraction` - they justify this
design or contradict it.

`type = "container"` is behind a compile-time feature. A binary built without it
says so at configuration load, with the sentence that names the fix
(`--features container`) - not a panic, and never a silent fallback to something
weaker.

### HTTP environments

`HttpEnvironment` reaches an environment server - typically the TypeScript/bun
container infrastructure, or an existing OpenEnv server, whose wire contract is
already the same thing under another name - over five JSON routes:

| Route               | Request                       | Response                                                  |
| ------------------- | ----------------------------- | --------------------------------------------------------- |
| `GET /tools`        | -                             | `{"tools": [ToolSpec]}`                                     |
| `POST /reset`       | `{"scenario": …, "seed": n}`  | `{"env_id": "…", "observation": null｜"…"}`                 |
| `POST /step`        | `{"env_id": …, "call": …}`    | `{"result": ToolResult, "reward": null｜n, "done": false}`  |
| `GET /state/{id}`   | -                             | `EnvState`; **`404` means "this server reports no state"**  |
| `POST /close`       | `{"env_id": …}`               | any 2xx, body never read - `204 No Content` is fine         |

`/state` is the one route where a `404` is not a lost session: it degrades to
`EnvState::default()` instead of costing the trajectory. That is safe precisely
because the state is read after the rollout, by the judge, and nothing downstream
is conditioned on it.

It is an *environment*, never an inference provider: tokenization, batching and
the prefix invariant all stay in Rust. A `404`/`410` on a route carrying an
`env_id` means the server restarted or reaped the session; that fails the
trajectory rather than silently starting a fresh episode against a prefix
conditioned on a world that no longer exists.

```toml
[agent.environment]
type = "http"
base_url = "http://127.0.0.1:8099"
request_timeout_secs = 60   # one call; max_rollout_secs still bounds the trajectory
pool_size = 16              # below group_size serializes the turn batching parallelized
max_result_bytes = 65536
```

An environment reward is for a task the environment can verify itself: a
puzzle whose `/step` grades the move needs no judge.

An environment may be composed with MCP servers only when every shared server
sets `stateless = true`. This is an operator assertion: Retrograd cannot prove a
remote server has no observable state, but it refuses the unsafe default and
warns when a declared-stateless server exposes write-like verbs.

## Intermediate rewards

Step rewards use undiscounted return (`gamma = 1`). The trajectory's scalar
group reward is the final judge reward plus all step rewards. For each
trainable token, the intermediate return is the sum of rewards on steps whose
range starts at or after that token. These returns are mean-centered inside
the trajectory and added to its group-relative advantage. Centering preserves
the trajectory-level GRPO scale while moving credit toward actions that precede
useful intermediate outcomes. Context and tool-result tokens stay masked.
