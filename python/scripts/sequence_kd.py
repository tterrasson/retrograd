"""Offline sequence-KD: sample the teacher, keep what survives, write a corpus.

The cheapest distillation there is, and the baseline every on-policy run has to
beat: the teacher answers the prompts, the answers that survive filtering become
an ordinary SFT corpus, and training is a plain ``sft`` run with no new code
anywhere. It is deliberately Python and deliberately outside the crates - it
produces *text*, so a token-level misalignment cannot happen here (unlike the
top-k sidecar of D6.2, which is why that one is a subcommand instead).

Input is the prompt-only chat JSONL the rollout algorithms read: one JSON object
per line, ``{"messages": [...]}``, the conversation ending on a ``user`` turn. A
trailing ``assistant`` turn is accepted and *not* sampled over - it is the
reference answer, handed to the verifier and dropped from the output.

Output is chat JSONL as ``retrograd-dataset`` consumes it. That reader rejects
unknown fields, so nothing but ``messages`` and ``rubric`` is written; run
statistics go to ``--report`` instead of onto the training lines.

Filtering, in the order the plan asks for it:

* an exact verifier when the domain has one (``--verifier CMD``): the command
  receives one JSON object per candidate on stdin - ``prompt``, ``completion``,
  ``reference``, ``rubric``, ``index`` - and answers one JSON object per line,
  ``{"keep": true}`` or ``{"score": 0.87}`` with ``--threshold``. A judge is
  wired in the same way: point ``--verifier`` at a command that calls it.
* deduplication, on the whitespace-normalized completion. Four samples of a
  short factual answer are usually one answer four times, and training on it
  four times is only a reweighting of that prompt.
* a cap of ``--keep`` survivors per prompt, longest first, and a floor of one -
  a prompt whose candidates were all rejected leaves no line at all.

Usage::

    uv run python scripts/sequence_kd.py --model teacher.gguf \\
        --prompts prompts.jsonl --out corpus.jsonl --samples 4 --keep 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

#: Upper bound on ``--keep``. Past a handful of survivors a prompt stops being
#: one example and becomes a weighting of the corpus towards itself.
MAX_SURVIVORS = 4


class SamplingError(RuntimeError):
    """The pipeline could not produce a corpus."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt line: the conversation to continue, and what judges it."""

    line: int
    messages: tuple[dict[str, str], ...]
    reference: str | None = None
    rubric: str | None = None

    @property
    def text(self) -> str:
        """The final user turn - what a verifier is asked about."""

        return self.messages[-1]["content"]


class Sampler(Protocol):
    """What the pipeline needs from a model: k completions for one prompt."""

    def sample(self, messages: Sequence[dict[str, str]], seed: int) -> str: ...


@dataclass
class Report:
    """What the run did, per stage. Written next to the corpus, never into it."""

    prompts: int = 0
    candidates: int = 0
    empty: int = 0
    rejected: int = 0
    duplicates: int = 0
    truncated_to_cap: int = 0
    kept: int = 0
    prompts_without_survivor: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompts": self.prompts,
            "candidates": self.candidates,
            "empty": self.empty,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "truncated_to_cap": self.truncated_to_cap,
            "kept": self.kept,
            "prompts_without_survivor": self.prompts_without_survivor,
            "survival_rate": self.kept / self.candidates if self.candidates else 0.0,
        }


def read_prompts(path: Path) -> list[Prompt]:
    """Reads prompt-only chat JSONL, with the same rules as the Rust reader."""

    prompts: list[Prompt] = []
    with path.open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise SamplingError(f"{path}:{line}: {error}") from error
            if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
                raise SamplingError(f"{path}:{line}: a record needs a 'messages' list")
            messages = [_message(path, line, value) for value in record["messages"]]
            if not messages:
                raise SamplingError(f"{path}:{line}: a record needs at least one message")
            reference = None
            if messages[-1]["role"] == "assistant":
                reference = messages.pop()["content"]
            if not messages or messages[-1]["role"] != "user":
                raise SamplingError(f"{path}:{line}: messages must end with a user message")
            rubric = record.get("rubric")
            prompts.append(
                Prompt(
                    line=line,
                    messages=tuple(messages),
                    reference=reference,
                    rubric=rubric if isinstance(rubric, str) and rubric.strip() else None,
                )
            )
    if not prompts:
        raise SamplingError(f"{path}: prompt dataset must not be empty")
    return prompts


def _message(path: Path, line: int, value: object) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("role"), str)
        or not isinstance(value.get("content"), str)
    ):
        raise SamplingError(f"{path}:{line}: a message needs string 'role' and 'content'")
    return {"role": value["role"], "content": value["content"]}


def normalize(completion: str) -> str:
    """The form two completions are considered the same in."""

    return " ".join(completion.split())


def deduplicate(completions: Iterable[str]) -> list[str]:
    """Keeps the first spelling of each distinct answer, in sampling order."""

    seen: set[str] = set()
    unique: list[str] = []
    for completion in completions:
        key = normalize(completion)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(completion.strip())
    return unique


def select(completions: Sequence[str], keep: int) -> list[str]:
    """Caps the survivors of one prompt, longest first.

    Length is a weak proxy and it is the honest one available without a reward:
    among answers a verifier already accepted, the longer one carries more of
    the teacher's reasoning. Ties keep sampling order.
    """

    if keep < 1 or keep > MAX_SURVIVORS:
        raise SamplingError(f"--keep must be between 1 and {MAX_SURVIVORS}")
    ranked = sorted(enumerate(completions), key=lambda pair: (-len(pair[1]), pair[0]))
    return [completion for _, completion in ranked[:keep]]


class CommandVerifier:
    """An external judgement, one JSON line in and one JSON line out.

    Started once for the whole run rather than per candidate: a verifier is
    usually a model or a sandbox, and paying its startup on every completion is
    what makes people skip the filtering step altogether.
    """

    def __init__(self, command: str, threshold: float | None = None) -> None:
        self._command = command
        self._threshold = threshold
        # `shell=True`: the command is the operator's own line, written next to
        # the model path in their own run script, and it is expected to be a
        # pipeline.
        self._process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def __enter__(self) -> CommandVerifier:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def keep(self, prompt: Prompt, completion: str, index: int) -> bool:
        assert self._process.stdin is not None and self._process.stdout is not None
        request = {
            "prompt": prompt.text,
            "completion": completion,
            "reference": prompt.reference,
            "rubric": prompt.rubric,
            "index": index,
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        answer = self._process.stdout.readline()
        if not answer:
            raise SamplingError(f"verifier {self._command!r} closed its output")
        try:
            verdict = json.loads(answer)
        except json.JSONDecodeError as error:
            raise SamplingError(f"verifier answered {answer!r}: {error}") from error
        if "keep" in verdict:
            return bool(verdict["keep"])
        if "score" in verdict:
            if self._threshold is None:
                raise SamplingError("a verifier answering 'score' needs --threshold")
            return float(verdict["score"]) >= self._threshold
        raise SamplingError(f"verifier answered {verdict!r} without 'keep' or 'score'")

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait()


def distill_prompt(
    prompt: Prompt,
    sampler: Sampler,
    *,
    samples: int,
    keep: int,
    seed: int,
    verifier: CommandVerifier | None,
    report: Report,
) -> list[str]:
    """Samples one prompt and returns the completions that survive."""

    candidates: list[str] = []
    for index in range(samples):
        completion = sampler.sample(prompt.messages, seed + index)
        report.candidates += 1
        if not completion.strip():
            report.empty += 1
            continue
        if verifier is not None and not verifier.keep(prompt, completion, index):
            report.rejected += 1
            continue
        candidates.append(completion)
    unique = deduplicate(candidates)
    report.duplicates += len(candidates) - len(unique)
    survivors = select(unique, keep)
    report.truncated_to_cap += len(unique) - len(survivors)
    report.kept += len(survivors)
    if not survivors:
        report.prompts_without_survivor.append(prompt.line)
    return survivors


def training_line(prompt: Prompt, completion: str) -> dict[str, Any]:
    """One SFT record. Only the fields ``retrograd-dataset`` accepts."""

    record: dict[str, Any] = {
        "messages": [*(dict(message) for message in prompt.messages)],
    }
    record["messages"].append({"role": "assistant", "content": completion})
    if prompt.rubric is not None:
        record["rubric"] = prompt.rubric
    return record


def run(
    prompts: Sequence[Prompt],
    sampler: Sampler,
    out: Path,
    *,
    samples: int,
    keep: int,
    seed: int,
    verifier: CommandVerifier | None = None,
) -> Report:
    """Writes the corpus and returns what the run did."""

    if samples < 1:
        raise SamplingError("--samples must be at least one")
    report = Report(prompts=len(prompts))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for offset, prompt in enumerate(prompts):
            survivors = distill_prompt(
                prompt,
                sampler,
                samples=samples,
                keep=keep,
                seed=seed + offset * samples,
                verifier=verifier,
                report=report,
            )
            for completion in survivors:
                line = training_line(prompt, completion)
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return report


class TeacherSampler:
    """The default sampler: the teacher, through the Python binding.

    Imported lazily so every pure stage above stays testable without a model.
    """

    def __init__(
        self,
        model: Path,
        *,
        context_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        device: str,
    ) -> None:
        from retrograd import SamplingConfig, Trainer, TrainingConfig

        self._trainer = Trainer(
            model,
            training=TrainingConfig(context_size=context_size, device=device),
        )
        self._sampling = SamplingConfig
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

    def sample(self, messages: Sequence[dict[str, str]], seed: int) -> str:
        generation = self._trainer.chat(
            list(messages),
            sampling=self._sampling(
                temperature=self._temperature,
                top_p=self._top_p,
                max_new_tokens=self._max_new_tokens,
                seed=seed,
            ),
        )
        return generation.text

    def close(self) -> None:
        self._trainer.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="teacher GGUF")
    parser.add_argument("--prompts", type=Path, required=True, help="prompt-only chat JSONL")
    parser.add_argument("--out", type=Path, required=True, help="chat JSONL to write")
    parser.add_argument("--report", type=Path, default=None, help="run statistics, as JSON")
    parser.add_argument("--samples", type=int, default=4, help="completions sampled per prompt")
    parser.add_argument("--keep", type=int, default=2, help=f"survivors kept (1-{MAX_SURVIVORS})")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--ctx", type=int, default=2048, help="teacher context size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--verifier", default=None, help="filter command, one JSON line each way")
    parser.add_argument("--threshold", type=float, default=None, help="keep 'score' >= this")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prompts = read_prompts(args.prompts)
        sampler = TeacherSampler(
            args.model,
            context_size=args.ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        verifier = CommandVerifier(args.verifier, args.threshold) if args.verifier else None
        try:
            report = run(
                prompts,
                sampler,
                args.out,
                samples=args.samples,
                keep=args.keep,
                seed=args.seed,
                verifier=verifier,
            )
        finally:
            sampler.close()
            if verifier is not None:
                verifier.close()
    except SamplingError as error:
        print(f"error {error}", file=sys.stderr)
        return 1
    payload = json.dumps(report.as_dict(), indent=2)
    if args.report is not None:
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
