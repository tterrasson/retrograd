"""Tests of the register-machine example, standard library only:

    python3 -m unittest discover -s examples/register_machine -v

They cover the machine, the reward and the generator, check the committed
prompt files against the rules, and speak the persistent reward protocol to the
server as the trainer would - nothing here needs the Rust build.
"""

from __future__ import annotations

import io
import itertools
import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import generate
from machine import (
    MAX_VALUE,
    OPERATIONS,
    SYSTEM_PROMPT,
    Command,
    IllegalCommand,
    Task,
    commands,
    distances,
    moves,
    parse_command,
    predecessors,
    run,
    solve,
    step,
)
from reward_server import PROTOCOL, explain, score, serve

HERE = Path(__file__).resolve().parent

TASK = Task(start=(3, 7, 1), goal=(7, 3, 5), operations=OPERATIONS, limit=3)
PROMPT = TASK.render()


def program(*lines: str) -> str:
    return "<program>\n" + "\n".join(lines) + "\n</program>"


def forward_distance(task: Task) -> int | None:
    """Plain breadth-first search from the start: the oracle `distances`,
    which walks the written-out reverse edges, has to agree with."""
    seen = {task.start: 0}
    frontier = deque([task.start])
    while frontier:
        state = frontier.popleft()
        if state == task.goal:
            return seen[state]
        for _, after in moves(task, state):
            if after not in seen:
                seen[after] = seen[state] + 1
                frontier.append(after)
    return None


class CommandTest(unittest.TestCase):
    def test_grammar(self) -> None:
        self.assertEqual(parse_command("add A 2"), Command("add", "A", 2))
        self.assertEqual(parse_command("  SUB b 9 "), Command("sub", "B", 9))
        self.assertEqual(parse_command("copy A C"), Command("copy", "A", "C"))
        self.assertEqual(parse_command("swap\tD  B"), Command("swap", "D", "B"))
        for line in (
            "add A 0",
            "add A 10",
            "add A -1",
            "add A",
            "add AB 1",
            "copy A A",
            "swap A 2",
            "mul A 2",
            "1. add A 2",
            "add A 2;",
            "",
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_command(line))

    def test_str_round_trips(self) -> None:
        for command in commands(TASK):
            self.assertEqual(parse_command(str(command)), command)
        for line in ("add C 4", "copy B A", "swap A D"):
            self.assertEqual(str(parse_command(line)), line)


class StepTest(unittest.TestCase):
    def test_semantics(self) -> None:
        self.assertEqual(step(TASK, (3, 7, 1), Command("add", "A", 2)), (5, 7, 1))
        self.assertEqual(step(TASK, (3, 7, 1), Command("sub", "B", 7)), (3, 0, 1))
        # `copy X Y` writes X into Y.
        self.assertEqual(step(TASK, (3, 7, 1), Command("copy", "A", "C")), (3, 7, 3))
        self.assertEqual(step(TASK, (3, 7, 1), Command("swap", "A", "B")), (7, 3, 1))

    def test_illegal(self) -> None:
        arithmetic = Task((3, 7), (4, 7), ("add", "sub"), limit=2)
        cases = (
            (TASK, Command("add", "B", 3), "would set B to 10"),
            (TASK, Command("sub", "C", 2), "would set C to -1"),
            (TASK, Command("swap", "A", "D"), "register D"),
            (arithmetic, Command("add", "C", 1), "register C"),
            (arithmetic, Command("swap", "A", "B"), "swap is not an allowed"),
        )
        for task, command, message in cases:
            with (
                self.subTest(command=str(command)),
                self.assertRaisesRegex(IllegalCommand, message),
            ):
                step(task, task.start, command)

    def test_run_stops_at_the_first_line_that_cannot_run(self) -> None:
        self.assertEqual(run(TASK, ["swap A B", "add C 4"]).error, None)
        stopped = run(TASK, ["swap A B", "add C 9", "add C 4"])
        self.assertEqual((stopped.state, stopped.executed), ((7, 3, 1), 1))
        self.assertIn("would set C to 10", stopped.error or "")
        over = run(TASK, ["add A 1", "sub A 1", "swap A B", "add C 4"])
        self.assertEqual((over.state, over.executed), ((7, 3, 1), 3))
        self.assertEqual(over.error, "more than 3 commands")
        junk = run(TASK, ["Here is the program:"])
        self.assertEqual((junk.state, junk.executed), (TASK.start, 0))


class TaskTest(unittest.TestCase):
    def test_render_parse_round_trip(self) -> None:
        tasks = (
            TASK,
            Task((0, 9), (9, 0), ("swap",), limit=1),
            Task((1, 2, 3, 4), (4, 3, 2, 1), ("sub", "copy", "swap"), limit=5),
        )
        for task in tasks:
            with self.subTest(task=task):
                self.assertEqual(Task.parse(task.render()), task)
        self.assertIn("Use at most 1 command.", tasks[1].render())

    def test_parse_rejects_what_it_did_not_write(self) -> None:
        for text in (
            "What is 2 + 2?",
            PROMPT.replace("A=3 B=7 C=1", "A=3 C=7 B=1"),
            PROMPT.replace("A=3 B=7 C=1", "A=3, B=7, C=1"),
            PROMPT.replace("A=3 B=7 C=1", "A=3 B=7"),
            PROMPT.replace("A=3 B=7 C=1", "A=13 B=7 C=1"),
            PROMPT.replace("add, sub", "add, mul"),
            PROMPT.replace("add, sub, copy", "copy, add, sub"),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Task.parse(text)


class SearchTest(unittest.TestCase):
    OPERATION_SETS = (*generate.BOTH_WAYS, *generate.ONE_WAY, ("swap",), ("copy",))

    def test_predecessors_are_the_reverse_of_moves(self) -> None:
        for operations in self.OPERATION_SETS:
            task = Task((0, 0), (0, 0), operations, limit=1)
            states = list(itertools.product(range(MAX_VALUE + 1), repeat=2))
            forward = {(before, after) for before in states for _, after in moves(task, before)}
            backward = {
                (before, after)
                for after in states
                for before in predecessors(task, after)
                if before != after
            }
            forward = {edge for edge in forward if edge[0] != edge[1]}
            with self.subTest(operations=operations):
                self.assertEqual(forward, backward)

    def test_distances_agree_with_a_forward_search(self) -> None:
        rng = random.Random(0)
        for _ in range(150):
            size = rng.choice((2, 3))
            operations = rng.choice(self.OPERATION_SETS)
            task = Task(
                tuple(rng.randint(0, MAX_VALUE) for _ in range(size)),
                tuple(rng.randint(0, MAX_VALUE) for _ in range(size)),
                operations,
                limit=1,
            )
            with self.subTest(task=task):
                self.assertEqual(distances(task).get(task.start), forward_distance(task))

    def test_solve_is_shortest_and_reaches_the_goal(self) -> None:
        rng = random.Random(1)
        for tier in generate.TIERS:
            for _ in range(10):
                task = generate.sample_task(rng, tier)
                solution = solve(task)
                assert solution is not None
                self.assertEqual(len(solution), forward_distance(task))
                result = run(task, [str(command) for command in solution])
                self.assertEqual((result.state, result.error), (task.goal, None))

    def test_unreachable(self) -> None:
        # Only `copy`: two equal registers can never differ again.
        task = Task((4, 4), (4, 5), ("copy",), limit=3)
        self.assertNotIn(task.start, distances(task))
        self.assertIsNone(solve(task))


class RewardTest(unittest.TestCase):
    def reward(self, completion: str, prompt: str = PROMPT) -> float:
        return score(prompt, completion).reward

    def test_parts(self) -> None:
        optimal = score(PROMPT, program("swap A B", "add C 4"))
        self.assertEqual(
            (optimal.format, optimal.progress, optimal.solved, optimal.efficiency),
            (0.1, 0.5, 0.3, 0.1),
        )
        self.assertEqual(optimal.reward, 1.0)
        self.assertEqual((optimal.distance_start, optimal.distance_end), (2, 0))

        # Three commands where two were enough: 0.1 * 2 / 3.
        self.assertAlmostEqual(
            self.reward(program("add A 4", "sub B 4", "add C 4")), 0.9 + 0.2 / 3, places=6
        )
        # Half the way, cleanly: format and half the progress.
        self.assertAlmostEqual(self.reward(program("swap A B")), 0.1 + 0.25)
        # The same half, then a command that breaks: the progress is halved.
        self.assertAlmostEqual(self.reward(program("swap A B", "add C 9")), 0.1 + 0.125)
        # ... and a line outside the grammar also costs the format point.
        self.assertAlmostEqual(self.reward(program("swap A B", "add C to 5")), 0.125)

    def test_format(self) -> None:
        solved = 0.9
        self.assertAlmostEqual(self.reward("Sure!\n" + program("swap A B", "add C 4")), solved)
        self.assertAlmostEqual(self.reward(program("swap A B", "add C 4") + "\nDone."), solved)
        self.assertAlmostEqual(
            self.reward(program("swap A B", "add C 4") + program("add A 1")), solved
        )
        self.assertEqual(self.reward("<program>swap A B\nadd C 4</program>"), 1.0)
        self.assertEqual(self.reward("\n\n" + program("", "swap A B", "  ", "add C 4") + "\n"), 1.0)

    def test_nothing_to_run_scores_zero(self) -> None:
        for completion in (
            "",
            "swap A B\nadd C 4",
            "<program>\n</program>",
            # Cut off by the token budget before the closing tag.
            "<program>\nswap A B\nadd C",
            "<PROGRAM>\nswap A B\nadd C 4\n</PROGRAM>",
        ):
            with self.subTest(completion=completion):
                self.assertEqual(self.reward(completion), 0.0)

    def test_padding_and_detours_earn_nothing(self) -> None:
        half = self.reward(program("swap A B"))
        self.assertEqual(self.reward(program("add A 1", "sub A 1", "swap A B")), half)
        self.assertEqual(self.reward(program("swap A C", "swap A C")), 0.1)
        # Moving away is not negative progress below the start.
        self.assertEqual(self.reward(program("sub C 1")), 0.1)

    def test_limit(self) -> None:
        # The goal is reached, but only at the fourth command of three.
        over = score(PROMPT, program("add A 1", "sub A 1", "swap A B", "add C 4"))
        self.assertEqual((over.solved, over.executed), (0.0, 3))
        self.assertAlmostEqual(over.reward, 0.1 + 0.125)

    def test_goal_reached_then_broken(self) -> None:
        broken = score(PROMPT, program("swap A B", "add C 4", "add C 5"))
        self.assertEqual(broken.solved, 0.0)
        self.assertAlmostEqual(broken.reward, 0.1 + 0.25)

    def test_dead_end_is_no_progress(self) -> None:
        task = Task((4, 7), (7, 5), ("sub", "copy"), limit=4)
        prompt = task.render()
        # Copying 4 over the only 7 leaves no way to write 7 into A.
        dead = score(prompt, program("copy A B"))
        self.assertIsNone(dead.distance_end)
        self.assertEqual(dead.reward, 0.1)

    def test_ranking(self) -> None:
        ladder = [
            program("swap A B", "add C 4"),
            program("add A 4", "sub B 4", "add C 4"),
            "Here it is:\n" + program("swap A B", "add C 4"),
            program("swap A B"),
            program("swap A B", "add C 9"),
            program("sub C 1"),
            "swap A B\nadd C 4",
        ]
        rewards = [self.reward(completion) for completion in ladder]
        self.assertEqual(rewards, sorted(rewards, reverse=True))
        self.assertEqual(len(set(rewards)), len(rewards))

    def test_bounds_on_random_programs(self) -> None:
        rng = random.Random(2)
        lines = ["add A 3", "sub B 2", "copy A C", "swap B C", "add D 1", "nope", "sub C 9"]
        for _ in range(300):
            completion = program(*rng.choices(lines, k=rng.randint(0, 5)))
            value = self.reward(completion)
            self.assertTrue(0.0 <= value <= 1.0, (completion, value))

    def test_unreachable_prompt_is_a_dataset_error(self) -> None:
        with self.assertRaises(ValueError):
            score(Task((4, 4), (4, 5), ("copy",), 3).render(), program("copy A B"))
        with self.assertRaises(ValueError):
            score(Task((4, 5), (4, 5), OPERATIONS, 3).render(), program("add A 1"))


class GeneratorTest(unittest.TestCase):
    def test_tiers(self) -> None:
        rng = random.Random(3)
        for tier in generate.TIERS:
            for _ in range(25):
                task = generate.sample_task(rng, tier)
                depth = distances(task)[task.start]
                with self.subTest(tier=tier.name, task=task):
                    self.assertIn(len(task.start), tier.registers)
                    self.assertIn(depth, tier.depths)
                    self.assertIn(task.limit - depth, tier.slacks)
                    self.assertIn(task.operations, tier.operation_sets)

    def test_negative_sizes_do_not_write_files(self) -> None:
        for argument in ("--train=-1", "--eval=-1"):
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(HERE / "generate.py"), argument, f"--out-dir={directory}"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be non-negative", result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_output_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "generate.py"),
                        "--train=40",
                        "--eval=9",
                        "--seed=11",
                        f"--out-dir={directory}",
                    ],
                    check=True,
                    capture_output=True,
                )
            for name in ("prompts.jsonl", "eval.jsonl"):
                self.assertEqual(
                    (Path(first) / name).read_text(), (Path(second) / name).read_text()
                )
            train = read_tasks(Path(first) / "prompts.jsonl")
            held_out = read_tasks(Path(first) / "eval.jsonl")
            self.assertEqual((len(train), len(held_out)), (40, 9))
            keys = [(task.start, task.goal, task.operations) for task in train + held_out]
            self.assertEqual(len(set(keys)), len(keys))


def read_tasks(path: Path) -> list[Task]:
    lines = path.read_text().splitlines()
    return [Task.parse(json.loads(line)["messages"][1]["content"]) for line in lines]


class CommittedDataTest(unittest.TestCase):
    def check(self, name: str, with_reference: bool) -> None:
        records = [json.loads(line) for line in (HERE / name).read_text().splitlines()]
        self.assertTrue(records)
        for number, record in enumerate(records, start=1):
            messages = record["messages"]
            with self.subTest(file=name, line=number):
                self.assertEqual(set(record), {"messages"})
                self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
                self.assertEqual(messages[1]["role"], "user")
                task = Task.parse(messages[1]["content"])
                self.assertEqual(task.render(), messages[1]["content"])
                depth = distances(task).get(task.start)
                self.assertIsNotNone(depth)
                self.assertGreaterEqual(task.limit, depth)
                if with_reference:
                    self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
                    reference = score(messages[1]["content"], messages[2]["content"])
                    self.assertEqual(reference.reward, 1.0)
                else:
                    self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_no_duplicates_between_or_within_splits(self) -> None:
        tasks = read_tasks(HERE / "prompts.jsonl") + read_tasks(HERE / "eval.jsonl")
        keys = [(task.start, task.goal, task.operations) for task in tasks]
        self.assertEqual(len(keys), len(set(keys)))

    def test_prompts(self) -> None:
        self.check("prompts.jsonl", with_reference=False)

    def test_eval(self) -> None:
        self.check("eval.jsonl", with_reference=True)


class ProtocolTest(unittest.TestCase):
    def requests(self) -> list[str]:
        lines = [json.dumps({"protocol": PROTOCOL})]
        completions = (program("swap A B", "add C 4"), "no idea", program("swap A B"))
        for index, completion in enumerate(completions):
            lines.append(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "completion": completion,
                        "_retrograd_batch": 7,
                        "_retrograd_index": index,
                    }
                )
            )
        lines.append(json.dumps({"_retrograd_batch_end": 7}))
        return lines

    def check(self, output: str) -> None:
        lines = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(lines[0], {"protocol": PROTOCOL})
        self.assertEqual(
            lines[1:4],
            [
                {"reward": 1.0, "_retrograd_batch": 7, "_retrograd_index": 0},
                {"reward": 0.0, "_retrograd_batch": 7, "_retrograd_index": 1},
                {"reward": 0.35, "_retrograd_batch": 7, "_retrograd_index": 2},
            ],
        )
        self.assertEqual(lines[4], {"_retrograd_batch_end": 7})
        self.assertEqual(len(lines), 5)

    def test_in_process(self) -> None:
        output = io.StringIO()
        serve(io.StringIO("\n".join(self.requests()) + "\n"), output)
        self.check(output.getvalue())

    def test_subprocess_answers_each_line_before_the_next(self) -> None:
        """The trainer keeps stdin open between batches: a response left in a
        buffer would never reach it, so each line is read back before the next
        one is written."""
        server = subprocess.Popen(
            [sys.executable, str(HERE / "reward_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert server.stdin is not None and server.stdout is not None
        try:
            output = []
            for line in self.requests():
                server.stdin.write(line + "\n")
                server.stdin.flush()
                output.append(server.stdout.readline())
            self.check("".join(output))
        finally:
            server.stdin.close()
            self.assertEqual(server.wait(timeout=10), 0)
            server.stdout.close()

    def test_wrong_handshake(self) -> None:
        with self.assertRaises(SystemExit):
            serve(io.StringIO('{"protocol":"other/1"}\n'), io.StringIO())

    def test_explain_completion_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            prompts.write_text(generate.record(TASK, False) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "reward_server.py"),
                    "--explain",
                    "--prompts",
                    str(prompts),
                ],
                input=json.dumps({"prompt_index": 0, "completion": program("swap A B", "add C 4")}),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["reward"], 1.0)
        for index in (-1, 1, True):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "prompt_index"):
                explain(io.StringIO(json.dumps({"prompt_index": index})), io.StringIO(), [PROMPT])

    def test_explain(self) -> None:
        request = json.dumps({"prompt": PROMPT, "completion": program("swap A B", "add C 9")})
        result = subprocess.run(
            [sys.executable, str(HERE / "reward_server.py"), "--explain"],
            input=request + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        details = json.loads(result.stdout)
        self.assertEqual(details["executed"], 1)
        self.assertEqual(details["error"], "add C 9 would set C to 10")
        self.assertAlmostEqual(details["reward"], 0.225)


if __name__ == "__main__":
    unittest.main()
