#!/usr/bin/env python3
"""Writes the prompt files of the GRPO register-machine example.

    python3 generate.py [--train 520] [--eval 66] [--seed 7] [--out-dir .]

Difficulty is set by four knobs, grouped into three tiers:

- registers: 2, 3 or 4;
- depth: the length of the *shortest* program, found by search, not the length
  of whatever walk produced the task;
- slack: how many commands above that shortest length the task allows;
- allowed commands: with both `add` and `sub`, any register is one command
  away from its goal, so the depth never exceeds the register count. Leaving
  out one direction forces detours through `copy` and `swap`, which is what
  makes the long tasks.

`shortcut` is the share of tasks drawn so that their shortest program is
shorter than the number of registers to change: a `swap` has to fix two
registers at once, so counting the differences is not enough.

The training file is ordered as a soft curriculum (easy first, tiers
overlapping) for `grpo.prompt_order = "sequential"`. Each evaluation line ends
with one shortest program as the assistant reference, for reading only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from machine import (
    MAX_VALUE,
    MIN_VALUE,
    OPERATIONS,
    SYSTEM_PROMPT,
    State,
    Task,
    distances,
    solve,
)

BOTH_WAYS = (
    OPERATIONS,
    ("add", "sub"),
    ("add", "sub", "swap"),
    ("add", "sub", "copy"),
)
ONE_WAY = (
    ("add", "copy"),
    ("sub", "copy"),
    ("add", "swap"),
    ("sub", "swap"),
    ("add", "copy", "swap"),
    ("sub", "copy", "swap"),
)


@dataclass(frozen=True)
class Tier:
    name: str
    share: float
    registers: tuple[int, ...]
    depths: tuple[int, ...]
    slacks: tuple[int, ...]
    operation_sets: tuple[tuple[str, ...], ...]
    shortcut: float


TIERS = (
    Tier("easy", 0.3, (2,), (1, 2), (1,), BOTH_WAYS, shortcut=0.2),
    Tier("medium", 0.4, (3,), (2, 3), (1,), BOTH_WAYS + ONE_WAY[:4], shortcut=0.4),
    Tier("hard", 0.3, (3, 4), (3, 4, 5), (0, 1), (*ONE_WAY, OPERATIONS), shortcut=0.2),
)


def is_shortcut(start: State, goal: State, depth: int) -> bool:
    """More registers to change than commands in the shortest program."""
    return sum(a != b for a, b in zip(start, goal, strict=True)) > depth


def sample_task(rng: random.Random, tier: Tier) -> Task:
    """Draws a goal, then a start at exactly the drawn depth from it. Whether
    the task is a shortcut is drawn once: the retries only redraw what can make
    it possible (a shortcut needs a `swap` or a `copy`, and enough registers)."""
    shortcut = rng.random() < tier.shortcut
    while True:
        size = rng.choice(tier.registers)
        operations = rng.choice(tier.operation_sets)
        depth = rng.choice(tier.depths)
        goal = tuple(rng.randint(MIN_VALUE, MAX_VALUE) for _ in range(size))
        distance = distances(Task(goal, goal, operations, limit=1))
        candidates = sorted(
            state
            for state, steps in distance.items()
            if steps == depth and is_shortcut(state, goal, depth) == shortcut
        )
        if candidates:
            start = rng.choice(candidates)
            return Task(start, goal, operations, limit=depth + rng.choice(tier.slacks))


def reference(task: Task) -> str:
    program = solve(task)
    assert program is not None, "a sampled task is reachable by construction"
    return "<program>\n" + "\n".join(map(str, program)) + "\n</program>"


def record(task: Task, with_reference: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.render()},
    ]
    if with_reference:
        messages.append({"role": "assistant", "content": reference(task)})
    return json.dumps({"messages": messages}, ensure_ascii=False)


def sample_tier(
    rng: random.Random, tier: Tier, count: int, seen: set[Task]
) -> list[tuple[Tier, Task]]:
    tasks = []
    while len(tasks) < count:
        task = sample_task(rng, tier)
        # Two tasks differing only by their limit are the same puzzle.
        key = Task(task.start, task.goal, task.operations, limit=1)
        if key not in seen:
            seen.add(key)
            tasks.append((tier, task))
    return tasks


def counts(total: int) -> list[int]:
    sizes = [int(total * tier.share) for tier in TIERS]
    sizes[-1] += total - sum(sizes)
    return sizes


def describe(label: str, tasks: list[tuple[Tier, Task]]) -> None:
    by_tier = Counter(tier.name for tier, _ in tasks)
    by_depth = Counter(distances(task)[task.start] for _, task in tasks)
    print(
        f"{label}: {len(tasks)} tasks, tiers {dict(sorted(by_tier.items()))}, "
        f"shortest program lengths {dict(sorted(by_depth.items()))}"
    )


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=520, help="training prompts")
    parser.add_argument("--eval", type=int, default=66, help="held-out prompts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()

    if arguments.train < 0 or arguments.eval < 0:
        parser.error("--train and --eval must be non-negative")

    rng = random.Random(arguments.seed)
    seen: set[Task] = set()

    # The held-out set is drawn first, so its tasks never reach the training file.
    eval_tiers = [
        sample_tier(rng, tier, size, seen)
        for tier, size in zip(TIERS, counts(arguments.eval), strict=True)
    ]
    # Round-robin over the tiers: `evaluation.max_examples` takes evenly
    # spaced lines. Interleaving spreads difficulty across the file, though
    # arbitrary subset sizes need not retain every tier.
    evaluation = [
        tier_tasks[index]
        for index in range(max(map(len, eval_tiers)))
        for tier_tasks in eval_tiers
        if index < len(tier_tasks)
    ]

    training = [
        entry
        for tier, size in zip(TIERS, counts(arguments.train), strict=True)
        for entry in sample_tier(rng, tier, size, seen)
    ]
    # A soft curriculum: the tier rank plus a jitter wider than one tier, so
    # the tiers overlap and no update sees only one of them for long.
    rank = {tier.name: index for index, tier in enumerate(TIERS)}
    keyed = sorted(
        (rank[tier.name] + rng.uniform(0.0, 1.6), index, (tier, task))
        for index, (tier, task) in enumerate(training)
    )
    training = [entry for _, _, entry in keyed]

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    for name, tasks, with_reference in (
        ("prompts.jsonl", training, False),
        ("eval.jsonl", evaluation, True),
    ):
        lines = (record(task, with_reference) + "\n" for _, task in tasks)
        (arguments.out_dir / name).write_text("".join(lines), encoding="utf-8")
        describe(name, tasks)


if __name__ == "__main__":
    main()
