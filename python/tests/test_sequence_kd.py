"""The offline sequence-KD pipeline, driven end to end by a fake teacher.

Every stage but the sampling itself is pure, and the sampler is a protocol, so
the whole pipeline runs here without a GGUF: the point of the test is the
filtering and the shape of what gets written, which is what a model would not
help decide.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sequence_kd.py"
_SPEC = importlib.util.spec_from_file_location("sequence_kd", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sequence_kd = importlib.util.module_from_spec(_SPEC)
sys.modules["sequence_kd"] = sequence_kd
_SPEC.loader.exec_module(sequence_kd)


class ListTeacher:
    """Answers in order, one answer per call."""

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.seeds: list[int] = []

    def sample(self, messages: Sequence[dict[str, str]], seed: int) -> str:
        self.seeds.append(seed)
        return self.answers.pop(0)


def write_prompts(path: Path, records: Sequence[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_prompt_only_records_round_trip_with_their_reference_and_rubric(tmp_path: Path) -> None:
    source = write_prompts(
        tmp_path / "prompts.jsonl",
        [
            {"messages": [{"role": "user", "content": "2+2?"}]},
            {
                "messages": [
                    {"role": "user", "content": "3+3?"},
                    {"role": "assistant", "content": "6"},
                ],
                "rubric": "exact arithmetic",
            },
        ],
    )
    prompts = sequence_kd.read_prompts(source)
    assert [prompt.text for prompt in prompts] == ["2+2?", "3+3?"]
    assert prompts[0].reference is None and prompts[0].rubric is None
    # The reference answer is what a verifier compares against, and it never
    # reaches the sampler: it would otherwise be a target the teacher copies.
    assert prompts[1].reference == "6"
    assert prompts[1].rubric == "exact arithmetic"
    assert [message["role"] for message in prompts[1].messages] == ["user"]


@pytest.mark.parametrize(
    "record",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "answer first"}]},
        {"prompt": "not a conversation"},
    ],
)
def test_an_unusable_prompt_line_is_refused(tmp_path: Path, record: dict[str, object]) -> None:
    source = write_prompts(tmp_path / "prompts.jsonl", [record])
    with pytest.raises(sequence_kd.SamplingError):
        sequence_kd.read_prompts(source)


def test_duplicate_answers_collapse_to_one_spelling() -> None:
    unique = sequence_kd.deduplicate(["  4  ", "4", "four", "", "four\n"])
    assert unique == ["4", "four"]


def test_survivors_are_capped_longest_first() -> None:
    assert sequence_kd.select(["a", "abc", "ab"], 2) == ["abc", "ab"]
    assert sequence_kd.select(["a", "abc", "ab"], 4) == ["abc", "ab", "a"]
    with pytest.raises(sequence_kd.SamplingError):
        sequence_kd.select(["a"], 0)
    with pytest.raises(sequence_kd.SamplingError):
        sequence_kd.select(["a"], sequence_kd.MAX_SURVIVORS + 1)


def test_the_corpus_is_chat_jsonl_with_no_field_the_dataset_reader_refuses(
    tmp_path: Path,
) -> None:
    source = write_prompts(
        tmp_path / "prompts.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "rubric": "exact arithmetic",
            }
        ],
    )
    prompts = sequence_kd.read_prompts(source)
    teacher = ListTeacher(["4", "  4 ", "", "The answer is four."])
    out = tmp_path / "corpus.jsonl"
    report = sequence_kd.run(prompts, teacher, out, samples=4, keep=2, seed=7)

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    for line in lines:
        # `retrograd-dataset` denies unknown fields: provenance would make the
        # corpus unreadable rather than merely noisy.
        assert set(line) <= {"messages", "rubric"}
        assert [message["role"] for message in line["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
    assert lines[0]["messages"][-1]["content"] == "The answer is four."
    assert lines[1]["messages"][-1]["content"] == "4"
    assert lines[0]["rubric"] == "exact arithmetic"

    assert report.candidates == 4
    assert report.empty == 1
    assert report.duplicates == 1
    assert report.kept == 2
    assert report.prompts_without_survivor == []
    # One seed per sample, distinct, so two prompts never share a completion.
    assert teacher.seeds == [7, 8, 9, 10]


def test_seeds_do_not_repeat_across_prompts(tmp_path: Path) -> None:
    source = write_prompts(
        tmp_path / "prompts.jsonl",
        [{"messages": [{"role": "user", "content": f"{index}?"}]} for index in range(3)],
    )
    prompts = sequence_kd.read_prompts(source)
    teacher = ListTeacher([f"answer {index}" for index in range(6)])
    report = sequence_kd.run(prompts, teacher, tmp_path / "corpus.jsonl", samples=2, keep=1, seed=0)
    assert teacher.seeds == [0, 1, 2, 3, 4, 5]
    assert report.kept == 3


def test_a_prompt_whose_candidates_are_all_empty_writes_no_line(tmp_path: Path) -> None:
    source = write_prompts(
        tmp_path / "prompts.jsonl", [{"messages": [{"role": "user", "content": "?"}]}]
    )
    prompts = sequence_kd.read_prompts(source)
    out = tmp_path / "corpus.jsonl"
    report = sequence_kd.run(prompts, ListTeacher(["", "   "]), out, samples=2, keep=1, seed=0)
    assert out.read_text(encoding="utf-8") == ""
    assert report.kept == 0
    assert report.prompts_without_survivor == [1]


def test_the_verifier_sees_the_reference_and_its_verdict_decides(tmp_path: Path) -> None:
    source = write_prompts(
        tmp_path / "prompts.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ],
    )
    prompts = sequence_kd.read_prompts(source)
    # An exact verifier: keep the completion when it contains the reference.
    script = tmp_path / "verifier.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        '    keep = request["reference"] in request["completion"]\n'
        '    sys.stdout.write(json.dumps({"keep": keep}) + "\\n")\n'
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    verifier = sequence_kd.CommandVerifier(f"{sys.executable} {script}")
    out = tmp_path / "corpus.jsonl"
    try:
        report = sequence_kd.run(
            prompts,
            ListTeacher(["the answer is 4", "five", "4!"]),
            out,
            samples=3,
            keep=4,
            seed=0,
            verifier=verifier,
        )
    finally:
        verifier.close()
    completions = [
        json.loads(line)["messages"][-1]["content"]
        for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert completions == ["the answer is 4", "4!"]
    assert report.rejected == 1
