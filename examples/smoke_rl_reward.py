#!/usr/bin/env python3
"""Tiny deterministic reward for the PPO/GRPO arithmetic smoke tests.

Speaks the persistent reward protocol (`reward_mode = "persistent"`, the
default): answer the handshake once, then one *flushed* JSON line per
{"prompt": ..., "completion": ...} request, for as many batches as the loop
sends. Nothing closes this process's stdin between batches, so a line left in
a buffer is a line the trainer never receives.

The score is smooth enough to give a group a useful ranking before the model
learns to answer exactly, while an exact answer receives the maximum reward.
"""

import json
import re
import sys

# The smoke prompts write multiplication and division with their Unicode signs,
# hence the `noqa`s.
EXPRESSION = re.compile(
    r"(?<!\w)(-?\d+)\s*([+\-*/×÷])\s*(-?\d+)"  # noqa: RUF001
    r"(?:\s*([+\-*/×÷])\s*(-?\d+))?(?!\w)"  # noqa: RUF001
)
INTEGER = re.compile(r"[-+]?\d+")


def evaluate_expression(match: re.Match[str]) -> int:
    result = int(match.group(1))
    operands = ((match.group(2), match.group(3)), (match.group(4), match.group(5)))
    for operator, operand in operands:
        if operator is None:
            break
        value = int(operand)
        if operator == "+":
            result += value
        elif operator == "-":
            result -= value
        elif operator in ("*", "×"):  # noqa: RUF001
            result *= value
        elif operator in ("/", "÷"):
            if value == 0 or result % value:
                raise ValueError("division must have a non-zero exact divisor")
            result //= value
    return result


def reward(prompt: str, completion: str) -> float:
    expression = EXPRESSION.search(prompt)
    if expression is None:
        raise ValueError(f"unsupported prompt: {prompt!r}")

    answer = INTEGER.search(completion)
    if answer is None:
        return 0.0

    expected = evaluate_expression(expression)
    distance = abs(int(answer.group(0)) - expected)
    return 1.0 / (1.0 + distance)


PROTOCOL = "retrograd-reward/1"

handshake = json.loads(sys.stdin.readline())
if handshake.get("protocol") != PROTOCOL:
    raise SystemExit(f"unsupported reward protocol: {handshake}")
print(json.dumps({"protocol": PROTOCOL}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if "_retrograd_batch_end" in request:
        print(json.dumps(request, separators=(",", ":")), flush=True)
        continue
    batch = request.pop("_retrograd_batch")
    index = request.pop("_retrograd_index")
    print(
        json.dumps(
            {
                "reward": reward(request["prompt"], request.get("completion", "")),
                "_retrograd_batch": batch,
                "_retrograd_index": index,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
