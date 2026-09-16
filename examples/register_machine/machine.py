"""The register machine of the GRPO register-machine example.

A task gives a few registers (A to D) holding digits, a goal state, the
commands the program may use and a command budget. A program is a list of
commands, one per line. This module is the single definition of that world:
`generate.py` renders tasks with it and `reward_server.py` parses them back and
runs programs through it, so the two cannot disagree on a rule.

Standard library only, so the reward command needs nothing but `python3`.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache

REGISTERS = "ABCD"
MIN_VALUE = 0
MAX_VALUE = 9
MAX_AMOUNT = 9
# Canonical order: a task lists its allowed commands in this order.
OPERATIONS = ("add", "sub", "copy", "swap")

State = tuple[int, ...]

SYSTEM_PROMPT = """\
You write programs for a tiny machine. Its registers are named A, B, C and D,
and each register holds a whole number from 0 to 9.

Commands:
add X n   adds n to register X (n is 1 to 9)
sub X n   subtracts n from register X (n is 1 to 9)
copy X Y  copies the value of X into Y (X keeps its value)
swap X Y  exchanges the values of X and Y

Rules:
- Use only the allowed commands and the registers listed in the task.
- A command that would take a register below 0 or above 9 is an error and stops the program.
- Never use more commands than the task allows.

Reply with the program only: one command per line, between <program> and </program>.
Example reply:
<program>
swap A B
add C 2
</program>"""


class IllegalCommand(ValueError):
    """A well-formed command the task does not permit in this state."""


@dataclass(frozen=True)
class Command:
    op: str
    register: str
    # The amount for `add`/`sub`, the second register for `copy`/`swap`.
    argument: int | str

    def __str__(self) -> str:
        return f"{self.op} {self.register} {self.argument}"


_ARITHMETIC = re.compile(r"(add|sub)\s+([a-z])\s+([1-9])", re.IGNORECASE)
_TRANSFER = re.compile(r"(copy|swap)\s+([a-z])\s+([a-z])", re.IGNORECASE)


def parse_command(line: str) -> Command | None:
    """Reads one command, or `None` when the line is not in the grammar.

    Grammar only: whether the command is allowed, names a register of the task
    and keeps every register in range is for `step` to decide.
    """
    text = line.strip()
    if match := _ARITHMETIC.fullmatch(text):
        return Command(match[1].lower(), match[2].upper(), int(match[3]))
    if match := _TRANSFER.fullmatch(text):
        first, second = match[2].upper(), match[3].upper()
        if first != second:
            return Command(match[1].lower(), first, second)
    return None


def _state_text(state: State) -> str:
    return " ".join(f"{name}={value}" for name, value in zip(REGISTERS, state, strict=False))


_STATE = re.compile(r"([A-D])=(\d)")
_TASK = re.compile(
    r"^Start: (?P<start>.+)\n"
    r"Goal: (?P<goal>.+)\n"
    r"Allowed commands: (?P<operations>.+)\n"
    r"Use at most (?P<limit>\d+) commands?\.$",
    re.MULTILINE,
)


def _parse_state(text: str) -> State:
    pairs = _STATE.findall(text)
    if " ".join(f"{name}={value}" for name, value in pairs) != text.strip():
        raise ValueError(f"malformed register list: {text!r}")
    if "".join(name for name, _ in pairs) != REGISTERS[: len(pairs)]:
        raise ValueError(f"registers must be named in order from A: {text!r}")
    return tuple(int(value) for _, value in pairs)


@dataclass(frozen=True)
class Task:
    start: State
    goal: State
    operations: tuple[str, ...]
    limit: int

    def __post_init__(self) -> None:
        if not 2 <= len(self.start) <= len(REGISTERS) or len(self.goal) != len(self.start):
            raise ValueError("a task has 2 to 4 registers, the same in start and goal")
        if any(not MIN_VALUE <= value <= MAX_VALUE for value in self.start + self.goal):
            raise ValueError("register values must be digits")
        if not self.operations or self.operations != tuple(
            op for op in OPERATIONS if op in self.operations
        ):
            raise ValueError(f"operations must be a non-empty subset of {OPERATIONS}, in order")
        if self.limit < 1:
            raise ValueError("the command limit must be positive")

    @property
    def registers(self) -> str:
        return REGISTERS[: len(self.start)]

    def render(self) -> str:
        """The user message of the task, which `parse` reads back."""
        plural = "" if self.limit == 1 else "s"
        return (
            f"Start: {_state_text(self.start)}\n"
            f"Goal: {_state_text(self.goal)}\n"
            f"Allowed commands: {', '.join(self.operations)}\n"
            f"Use at most {self.limit} command{plural}."
        )

    @classmethod
    def parse(cls, text: str) -> Task:
        match = _TASK.search(text)
        if match is None:
            raise ValueError(f"not a register-machine task: {text!r}")
        operations = tuple(op.strip() for op in match["operations"].split(","))
        if any(op not in OPERATIONS for op in operations):
            raise ValueError(f"unknown command in {match['operations']!r}")
        return cls(
            start=_parse_state(match["start"]),
            goal=_parse_state(match["goal"]),
            operations=operations,
            limit=int(match["limit"]),
        )


def _index(task: Task, register: str) -> int:
    index = task.registers.find(register)
    if index < 0:
        raise IllegalCommand(f"register {register} is not part of this task")
    return index


def step(task: Task, state: State, command: Command) -> State:
    """The state after `command`, or `IllegalCommand` when it may not run."""
    if command.op not in task.operations:
        raise IllegalCommand(f"{command.op} is not an allowed command")
    index = _index(task, command.register)
    values = list(state)
    if command.op in ("add", "sub"):
        amount = int(command.argument)
        value = values[index] + (amount if command.op == "add" else -amount)
        if not MIN_VALUE <= value <= MAX_VALUE:
            raise IllegalCommand(f"{command} would set {command.register} to {value}")
        values[index] = value
    else:
        other = _index(task, str(command.argument))
        if command.op == "copy":
            values[other] = values[index]
        else:
            values[index], values[other] = values[other], values[index]
    return tuple(values)


def commands(task: Task) -> Iterator[Command]:
    """Every well-formed command over the task's registers and operations."""
    for op in task.operations:
        for register in task.registers:
            if op in ("add", "sub"):
                for amount in range(1, MAX_AMOUNT + 1):
                    yield Command(op, register, amount)
            else:
                for other in task.registers:
                    # `swap A B` and `swap B A` are one edge.
                    if other != register and (op == "copy" or other > register):
                        yield Command(op, register, other)


def moves(task: Task, state: State) -> Iterator[tuple[Command, State]]:
    """The legal commands from `state` and where each one leads."""
    for command in commands(task):
        try:
            yield command, step(task, state, command)
        except IllegalCommand:
            continue


def predecessors(task: Task, state: State) -> Iterator[State]:
    """Every state that one legal command turns into `state` (with repeats).

    The reverse edges of `moves`, written out rather than searched for, so that
    `distances` costs one walk back from the goal instead of a search per state.
    """
    size = len(state)
    for op in task.operations:
        for index in range(size):
            if op in ("add", "sub"):
                sign = 1 if op == "add" else -1
                for amount in range(1, MAX_AMOUNT + 1):
                    before = state[index] - sign * amount
                    if MIN_VALUE <= before <= MAX_VALUE:
                        yield (*state[:index], before, *state[index + 1 :])
                continue
            for other in range(size):
                if other == index:
                    continue
                if op == "swap":
                    values = list(state)
                    values[index], values[other] = values[other], values[index]
                    yield tuple(values)
                elif state[other] == state[index]:
                    # `copy index other` left `other` equal to `index`; what
                    # `other` held before is lost, so any digit leads here.
                    for before in range(MIN_VALUE, MAX_VALUE + 1):
                        yield (*state[:other], before, *state[other + 1 :])


@lru_cache(maxsize=512)
def _distances(goal: State, operations: tuple[str, ...]) -> dict[State, int]:
    task = Task(goal, goal, operations, limit=1)
    distance = {goal: 0}
    frontier = deque([goal])
    while frontier:
        state = frontier.popleft()
        for before in predecessors(task, state):
            if before not in distance:
                distance[before] = distance[state] + 1
                frontier.append(before)
    return distance


def distances(task: Task) -> dict[State, int]:
    """Fewest commands from each state to the goal. A state missing from the
    map cannot reach the goal at all (a `copy` can destroy the only copy of a
    digit the goal needs). Cached: the reward asks for the same task once per
    completion of a group."""
    return _distances(task.goal, task.operations)


def solve(task: Task) -> list[Command] | None:
    """One shortest program from start to goal, ignoring the limit: from each
    state, the first command that brings the goal one command closer."""
    distance = distances(task)
    if task.start not in distance:
        return None
    program: list[Command] = []
    state = task.start
    while state != task.goal:
        command, state = next(
            (command, after)
            for command, after in moves(task, state)
            if distance.get(after) == distance[state] - 1
        )
        program.append(command)
    return program


@dataclass(frozen=True)
class Run:
    # The state after the last command that ran.
    state: State
    # How many commands ran.
    executed: int
    # Why the program stopped before its end, or `None` if it ran to its end.
    error: str | None


def run(task: Task, lines: Sequence[str]) -> Run:
    """Runs a program line by line, stopping at the first line that cannot run:
    past the limit, outside the grammar, or illegal in the current state."""
    state = task.start
    for count, line in enumerate(lines):
        if count == task.limit:
            return Run(state, count, f"more than {task.limit} commands")
        command = parse_command(line)
        if command is None:
            return Run(state, count, f"not a command: {line.strip()!r}")
        try:
            state = step(task, state, command)
        except IllegalCommand as error:
            return Run(state, count, str(error))

    return Run(state, len(lines), None)
