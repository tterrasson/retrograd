#!/usr/bin/env python3
"""Reward server of the GRPO register-machine example.

Speaks the persistent reward protocol (`reward_mode = "persistent"`): answer
the handshake once, then one *flushed* JSON line per
{"prompt": ..., "completion": ...} request, for the whole run. `prompt` is the
last user message - the task as `machine.Task.render` wrote it.

The reward, at most 1.0, is the sum of five parts:

    syntax     0.05  share of the lines of the block that parse as a command,
                     scaled by the length the task needs and forfeited by a
                     program that ends farther from the goal than it started
    format     0.05  one <program> block, nothing around it, every line a command
                     - and only for a program that made some progress
    progress    0.5  share of the way to the goal covered by the commands that ran,
                     halved when the program stopped on an error
    solved      0.3  the program ran to its end, within the limit, onto the goal
    efficiency  0.1  shortest / used, for a solved program only

Progress is measured in exact remaining commands (a breadth-first search back
from the goal), not in digit differences: one `swap` can fix two registers,
and a `copy` can make the goal unreachable. Each command that runs is credited
with how much it shortened that distance, so the credits add up to
`(d_start - d_end) / d_start` - a detour that is later undone earns nothing,
and the commands after the first one that cannot run earn nothing either. The
halving keeps a program that breaks below the same progress made cleanly: the
task asks for a program that runs, not for a good first half.

Without a <program> block the reward is 0 and nothing else is looked at. The
format point is conditioned on progress for the same reason: a well-formed
program that goes nowhere is the one answer that scores on *every* task, and an
unconditional bonus makes it a local optimum a weak policy collapses onto.

`syntax` is the one part that is neither gated nor all-or-nothing, because a
policy that has not yet learned the command grammar writes nothing a gated
reward can grade: it echoes the task back ("A=9 B=0 C=3"), every part scores 0,
every group is uniform and there is no gradient to escape with. Crediting the
*share* of lines that parse makes one valid command out of two beat none, which
is a direction. It is the smallest part on purpose, and it saturates as soon as
the grammar is learned - at which point it is identical across a group and
cancels in the baseline.

Being ungated, it is also the part a collapsing policy hunts for - a single
valid command parses on every task, so the credit was the same 0.05 whatever
was asked, and a weak policy settled on one line of `swap A B` for every
prompt. Two factors now stop it from being earned the same way twice:

  * it is scaled by `min(1, lines / d_start)`, because a program with fewer
    commands than the shortest solution cannot be right whatever it holds. One
    command where three are needed is worth a third of the credit, and the way
    out of that third is to write a program of the length the task asks for.
  * it is forfeited when the program ends farther from the goal than it
    started, dead ends included. A reply that ignores the task moves away on
    part of the prompts and scores 0 on those, while a program that stops on
    its first line, or takes a detour and undoes it, keeps the credit: the way
    in stays open for a policy that has not learned the grammar and closes on
    one that uses the grammar to go nowhere.

Both are per-task, so what was one flat value across the whole dataset is now
a spread the group baseline can work against: on the committed prompts, that
one line of `swap A B` scores between 0 and 0.025 wherever it makes no
progress, and a well-formed program of the right length beats it there.

    python3 reward_server.py --explain < requests.jsonl

prints the breakdown of each request instead of speaking the protocol.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from machine import Task, distances, parse_command, run

PROTOCOL = "retrograd-reward/1"

SYNTAX_WEIGHT = 0.05
FORMAT_WEIGHT = 0.05
PROGRESS_WEIGHT = 0.5
SOLVED_WEIGHT = 0.3
EFFICIENCY_WEIGHT = 0.1
# The share of its progress a program keeps when it stops on an error.
STOPPED_PROGRESS = 0.5

PROGRAM = re.compile(r"<program>(.*?)</program>", re.DOTALL)


@dataclass(frozen=True)
class Score:
    reward: float
    # Each part already weighted: `reward` is their sum.
    syntax: float
    format: float
    progress: float
    solved: float
    efficiency: float
    # Commands that ran, and why the program stopped early (`None`: it did not).
    executed: int
    error: str | None
    # Fewest commands to the goal, before the program and where it stopped;
    # `None` when the stopping state can no longer reach the goal.
    distance_start: int
    distance_end: int | None


def score(prompt: str, completion: str) -> Score:
    task = Task.parse(prompt)
    distance = distances(task)
    start = distance.get(task.start)
    if not start:
        raise ValueError(f"the goal must be reachable from a different start: {prompt!r}")

    blocks = PROGRAM.findall(completion)
    if not blocks:
        return Score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "no <program> block", start, start)

    lines = [line for line in blocks[0].splitlines() if line.strip()]
    parses = [parse_command(line) is not None for line in lines]
    well_formed = (
        len(blocks) == 1 and not PROGRAM.sub("", completion).strip() and bool(lines) and all(parses)
    )

    result = run(task, lines)
    end = distance.get(result.state)
    progress = 0.0 if end is None else max(0, start - end) / start
    if result.error is not None:
        progress *= STOPPED_PROGRESS
    solved = result.error is None and result.state == task.goal and result.executed > 0
    efficiency = start / result.executed if solved else 0.0

    # The grammar credit: the share of the lines that parse, scaled by the
    # length the task needs, and forfeited by a program that ends farther from
    # the goal than it started (a dead end included). Both factors are what
    # keeps it from being earned the same way on every task - see the module
    # docstring.
    syntax = 0.0
    if parses and (end is not None and end <= start):
        syntax = sum(parses) / len(parses) * min(1.0, len(lines) / start)

    parts = (
        SYNTAX_WEIGHT * syntax,
        FORMAT_WEIGHT * (well_formed and progress > 0),
        PROGRESS_WEIGHT * progress,
        SOLVED_WEIGHT * solved,
        EFFICIENCY_WEIGHT * efficiency,
    )
    return Score(
        round(sum(parts), 6),
        *(round(part, 6) for part in parts),
        executed=result.executed,
        error=result.error if lines else "empty program",
        distance_start=start,
        distance_end=end,
    )


def serve(stdin, stdout) -> None:
    handshake = json.loads(stdin.readline())
    if handshake.get("protocol") != PROTOCOL:
        raise SystemExit(f"unsupported reward protocol: {handshake}")
    print(json.dumps({"protocol": PROTOCOL}), file=stdout, flush=True)

    for line in stdin:
        request = json.loads(line)
        if "_retrograd_batch_end" in request:
            print(json.dumps(request, separators=(",", ":")), file=stdout, flush=True)
            continue
        response = {
            "reward": score(request["prompt"], request.get("completion", "")).reward,
            "_retrograd_batch": request["_retrograd_batch"],
            "_retrograd_index": request["_retrograd_index"],
        }
        print(json.dumps(response, separators=(",", ":")), file=stdout, flush=True)


def explain(stdin, stdout, prompts: list[str] | None = None) -> None:
    for line in stdin:
        if line.strip():
            request = json.loads(line)
            if "prompt" in request:
                prompt = request["prompt"]
            elif prompts is not None:
                index = request["prompt_index"]
                if type(index) is not int or not 0 <= index < len(prompts):
                    raise ValueError(f"prompt_index out of range: {index!r}")
                prompt = prompts[index]
            else:
                raise ValueError("completion logs require --prompts with the training JSONL")
            details = score(prompt, request.get("completion", ""))
            print(json.dumps(asdict(details)), file=stdout, flush=True)


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the score breakdown of each request line instead of serving",
    )
    parser.add_argument(
        "--prompts", type=Path, help="training JSONL used by a completion log (with --explain)"
    )
    arguments = parser.parse_args()
    if arguments.prompts is not None and not arguments.explain:
        parser.error("--prompts requires --explain")
    if arguments.explain:
        prompts = None
        if arguments.prompts is not None:
            records = arguments.prompts.read_text(encoding="utf-8").splitlines()
            prompts = [
                next(
                    message["content"]
                    for message in reversed(json.loads(line)["messages"])
                    if message["role"] == "user"
                )
                for line in records
            ]
        explain(sys.stdin, sys.stdout, prompts)
    else:
        serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
