# GRPO example: the register machine

A small model learns to write a short program that moves a tiny machine from
one state to another. It answers in one turn, with no tool calls: the whole
program comes at once, and a Python reward server runs it and scores it.

```text
Start: A=3 B=7 C=1
Goal: A=7 B=3 C=5
Allowed commands: add, sub, copy, swap
Use at most 3 commands.
```

```text
<program>
swap A B
add C 4
</program>
```

The task is small enough for a sub-billion-parameter model. Scoring is exact
and cheap, and the reward gives partial credit, so GRPO has a signal to work
with before the model solves anything.

## Files

| File | Role |
| --- | --- |
| `machine.py` | The world: commands, task format, execution, shortest-path search. The generator and the reward both use it. |
| `reward_server.py` | The reward command, using Retrograd's persistent protocol (`retrograd-reward/1`). |
| `generate.py` | Writes `prompts.jsonl` and `eval.jsonl` (seeded, reproducible). |
| `prompts.jsonl` | 600 training tasks, in easy-first order. |
| `eval.jsonl` | 78 held-out tasks, each ending with one shortest program as the assistant reference. |
| `grpo.toml` | The training configuration. |
| `test_register_machine.py` | Standard-library tests: the machine, the reward, the generator, the shipped data and the protocol. |

Everything in Python uses only the standard library, so the reward command
needs nothing but `python3` (3.11 or newer).

## The machine

There are two to four registers, `A` to `D`. Each one holds a digit from `0` to
`9`.

| Command | Effect |
| --- | --- |
| `add X n` | `X = X + n`, with `n` from 1 to 9 |
| `sub X n` | `X = X - n`, with `n` from 1 to 9 |
| `copy X Y` | `Y = X` (`X` keeps its value) |
| `swap X Y` | exchanges `X` and `Y` |

The program stops at the first line that cannot run:

- a line that is not a command;
- a command the task does not allow;
- a register the task does not have;
- a value that would leave `0..9`;
- a command beyond the task's limit.

The rules are in the system message of every prompt. The user message holds
only the task, in a fixed format that `Task.parse` reads back: Retrograd sends
the reward command only the **last user message**, and a prompt line may not
carry extra fields.

## Difficulty

`generate.py` sets four knobs:

- **Registers:** 2, 3 or 4.
- **Depth:** the length of the *shortest* program, found by breadth-first
  search, not the length of the walk that produced the task.
- **Slack:** how many commands the task allows beyond that shortest length. The
  limit is `depth + slack`.
- **Allowed commands:**
  - With both `add` and `sub`, each register is one command from its goal, so
    the depth never exceeds the register count.
  - Leaving out one direction (for example only `add` and `copy`) forces
    detours through `copy` and `swap`, and that is what makes the long
    programs.

A **shortcut** task is one whose shortest program is shorter than the number of
registers to change. A `swap` has to fix two registers at once, so counting the differences is not enough to solve it.

| Tier | Share | Registers | Depth | Slack | Commands | Shortcut tasks |
| --- | ---: | --- | --- | --- | --- | ---: |
| easy | 20% | 2 | 1-2 | 1 | both directions | 20% |
| medium | 35% | 3 | 2-3 | 1 | both directions, or one direction + `copy` or `swap` | 40% |
| hard | 45% | 3-4 | 3-5 | 0-1 | one direction only, or all four | 20% |

`--train-hard-extra` and `--eval-hard-extra` then draw more hard tasks on top of
those shares, which is what the committed files hold: 314 hard of 600 training
tasks (52%) and 42 of 78 held-out ones (54%). The partial-progress reward is
what a weak policy earns on the easy tasks early; the hard ones are what still
separates a group's rewards once it stops failing them.

The training file is a soft curriculum. Each task is sorted by its tier rank
plus a random jitter wider than one tier, so easy tasks come first but the
tiers overlap. `grpo.prompt_order = "sequential"` keeps that order.
Evaluation tasks alternate between the tiers to spread difficulty across the
file. The configured 30-example subset covers all three tiers; arbitrary subset
sizes need not do so. No task appears twice,
and no evaluation task is in the training file.

Regenerate the files, or make other sizes, with:

```bash
python3 examples/register_machine/generate.py --train 520 --eval 66 \
  --train-hard-extra 80 --eval-hard-extra 12 --seed 7
```

To change the difficulty, edit the `TIERS` table in `generate.py`.

## Reward

The reward is at most `1.0` and is the sum of five parts:

| Part | Weight | Earned when |
| --- | ---: | --- |
| syntax | 0.05 | Share of the lines inside the block that parse as a command. The only part that is neither gated nor all-or-nothing. |
| format | 0.05 | The reply is exactly one `<program>…</program>` block with nothing around it, every line is a command, **and the program made some progress**. |
| progress | 0.5 | Share of the distance to the goal covered by the commands that ran. Halved if the program stopped on an error. |
| solved | 0.3 | The program ran to its end, within the limit, and left the machine in the goal state. |
| efficiency | 0.1 | Only for a solved program: `shortest length / commands used`. |

**Without a `<program>` block, the reward is 0.** A reply cut off by
`max_new_tokens` has no closing tag, so it also scores 0.

The reward is built this way for six reasons:

- **Progress uses the real distance, not digit differences.** The distance of a
  state is the exact number of commands still needed. It comes from one
  breadth-first search backwards from the goal, cached per task. Counting
  differing digits would be wrong both ways: a `swap` can fix two registers at
  once, and a `copy` can overwrite the only copy of a digit the goal needs. Such
  a dead end has no distance and earns no progress.
- **Credit is given per command, and it adds up.** Each command that runs is
  credited with how much it shortened the distance, so the credits sum to
  `(d_start - d_end) / d_start`. Padding cannot inflate the reward: a detour
  that is later undone (`add A 1`, `sub A 1`) nets zero, and a step backwards
  cancels an earlier gain. The total never goes below zero.
- **A program that breaks is worth less than one that stops cleanly.**
  Commands after the first failing line earn nothing, and the progress already
  made is halved. Otherwise "half the work, then garbage" would tie with "half
  the work".
- **The limit is part of the task.** A command beyond it is a failing line, so
  reaching the goal one command too late is not a solve.
- **The syntax credit is partial, and it is the way in.** A policy that has not
  yet learned the command grammar writes nothing a gated reward can grade: it
  echoes the task back (`A=9 B=0 C=3`), every part scores 0, every group is
  uniform and GRPO has no gradient at all. Crediting the *share* of lines that
  parse makes one valid command out of two beat none, which is a direction to
  walk. It is the smallest part on purpose, and it saturates as soon as the
  grammar is learned - identical across a group, so it cancels in the baseline.
- **The format point is conditioned on progress.** Unconditional, it is the one
  answer that scores on *every* task: the shortest well-formed program that goes
  nowhere collects it whatever was asked. That is a local optimum with global
  support, and a policy that cannot yet solve anything collapses onto it - every
  group then samples the same completion, the group standard deviation is zero,
  and GRPO has no gradient left to escape with.

Examples for the task at the top of this page (shortest program: 2 commands,
limit: 3):

| Program | Syntax | Format | Progress | Solved | Efficiency | Reward | Stopped by |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `swap A B` / `add C 4` | 0.05 | 0.05 | 0.5 | 0.3 | 0.1 | **1.0** | |
| `add A 4` / `sub B 4` / `add C 4` | 0.05 | 0.05 | 0.5 | 0.3 | 0.067 | **0.967** | |
| the optimal program, with a sentence before it | 0.05 | 0 | 0.5 | 0.3 | 0.1 | **0.95** | |
| `swap A B` | 0.05 | 0.05 | 0.25 | 0 | 0 | **0.35** | |
| `add A 1` / `sub A 1` / `swap A B` | 0.05 | 0.05 | 0.25 | 0 | 0 | **0.35** | |
| `swap A B` / `add C 9` | 0.05 | 0.05 | 0.125 | 0 | 0 | **0.225** | `C` would be 10 |
| `add A 1` / `sub A 1` / `swap A B` / `add C 4` | 0.05 | 0.05 | 0.125 | 0 | 0 | **0.225** | more than 3 commands |
| `swap A B` / `add C to 5` | 0.025 | 0 | 0.125 | 0 | 0 | **0.15** | not a command |
| `sub C 1` | 0.05 | 0 | 0 | 0 | 0 | **0.05** | no progress, so no format point |
| `A=7 B=9 C=0` in a block | 0 | 0 | 0 | 0 | 0 | **0** | no line is a command |
| the optimal program without tags | 0 | 0 | 0 | 0 | 0 | **0** | no block |

GRPO only uses differences within a group, so the scale matters less than the
order. The format point is small on purpose: once every member of a group has
the format right, it cancels out and the ranking comes from progress and
solving. It is also the only part that is gated rather than added, so the
cheapest reply that satisfies every task is worth the syntax credit alone.

To see how a completion was scored, pass request lines to the server with
`--explain`:

```bash
printf '%s\n' '{"prompt":"Start: A=3 B=7 C=1\nGoal: A=7 B=3 C=5\nAllowed commands: add, sub, copy, swap\nUse at most 3 commands.","completion":"<program>\nswap A B\nadd C 9\n</program>"}' | python3 examples/register_machine/reward_server.py --explain
```

## Running it

From the repository root:

```bash
scripts/fetch-cpu-fixture.sh
```

```bash
cargo run --release -- train examples/register_machine/grpo.toml
```

```bash
cargo run --release -- bench examples/register_machine/grpo.toml
```

`grpo.toml` points at the 230M CPU fixture so that the example runs anywhere.
The 230M model can already improve on this task. You can also set `model.path`
to a larger instruct model, subject to available memory. Model and dataset paths
are relative to `grpo.toml`; the reward command is relative to the working
directory, so run these commands from the repository root.

Every 10 updates, a sample of completions is written to
`/tmp/retrograd-register-machine-completions.jsonl`. These records contain a
`prompt_index`, not the prompt text. Supply the training file to inspect them:

```bash
python3 examples/register_machine/reward_server.py --explain \
  --prompts examples/register_machine/prompts.jsonl \
  < /tmp/retrograd-register-machine-completions.jsonl
```

Use the exact training file from that run: regenerating it changes the indices.
The log appends across runs, so save or remove an old log before starting with
new data.

The bench mean includes format and partial-progress rewards: an improvement is
not necessarily an increase in fully solved tasks. Inspect completions alongside
the mean; `--explain` reports `solved: 0.3` for a solved task. Compare runs on the
same held-out file and sampling settings.

Things to watch:

- **Groups with no signal.** A group that scores the same everywhere (all
  zeros before the model uses the tags, or all ones on easy tasks later) is
  dropped. `[grpo.dynamic_sampling]` draws replacement prompts. If the run
  stops on `max_stalled_updates` right at the start, the model is not
  producing `<program>` blocks at all. Use a stronger instruct model, or run a
  short SFT warm-up first. `generate.reference(task)` writes the shortest
  program of a task in the expected format.
- **Reasoning models.** A model that thinks before answering will run out of
  its 96-token budget. Disable thinking in the chat template, or raise
  `max_new_tokens`, keeping the prompt plus the budget within `ctx`.

## Tests

```bash
python3 -m unittest discover -s examples/register_machine -v
```

The tests need neither the Rust build nor a model. They check that:

- the reverse search agrees with a plain forward search;
- every shipped prompt parses and is solvable within its limit;
- every evaluation reference scores `1.0`;
- the generator is deterministic;
- the reward ranks the ladder above in that order;
- the server answers each protocol line before the next one arrives, as it
  must, because the trainer keeps its stdin open.
