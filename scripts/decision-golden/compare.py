#!/usr/bin/env python3
"""Compare decision answers against a golden fixture.

  compare.py golden GOLDEN.json OTHER.json [--tolerance T]
      tokens and markers must be equal, probabilities within T

  compare.py server GOLDEN.json REQUESTS.json URL [--tolerance T]
      replays every request the server can take (no head-budget override)
      against a running llama-server and compares its answers and usage; a
      request the capture refused must be refused with the same message

Exit status 1 on the first family of differences, with each one printed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request

# What a pass records besides its scores; a field absent from both is skipped.
PASS_FIELDS = (
    "id",
    "variant",
    "prompt",
    "pieces",
    "rendered_tokens",
    "tokens",
    "markers",
    "column",
    "pointer",
    "labels",
    "label_groups",
    "order",
    "n_scores",
    "pooled",
)


def answer_values(answers: dict) -> dict:
    """Flattens the answers of one request to {(question, key): probability}."""
    out = {}
    for qid, answer in answers.items():
        if "probabilities" in answer:
            for key, p in answer["probabilities"].items():
                out[(qid, key)] = p
        else:
            out[(qid, "noul")] = answer["noul"]
    return out


def compare_answers(
    name: str, expected: dict, actual: dict, tolerance: float, errors: list
) -> float:
    exp, act = answer_values(expected), answer_values(actual)
    if list(exp) != list(act):
        errors.append(f"{name}: questions or options differ: {list(exp)} != {list(act)}")
        return 0.0
    worst = 0.0
    for key, p in exp.items():
        if not math.isfinite(p) or not math.isfinite(act[key]):
            errors.append(f"{name}: {key[0]}.{key[1]}: non-finite probability")
            continue
        diff = abs(p - act[key])
        worst = max(worst, diff)
        if diff > tolerance:
            errors.append(
                f"{name}: {key[0]}.{key[1]}: {act[key]} vs golden {p} (|diff| {diff:.2e})"
            )
    for qid, answer in expected.items():
        if answer.get("choice") != actual[qid].get("choice"):
            got, want = actual[qid].get("choice"), answer.get("choice")
            errors.append(f"{name}: {qid}: choice {got} vs golden {want}")
    return worst


def golden(args: argparse.Namespace) -> list:
    with open(args.golden) as f:
        gold = json.load(f)
    with open(args.other) as f:
        other = json.load(f)
    errors: list = []
    worst = 0.0
    for field in ("model", "backend"):
        if gold[field] != other[field]:
            errors.append(f"{field} differs")
    for g, o in zip(gold["requests"], other["requests"], strict=True):
        for field in ("name", "max_head_tokens", "error"):
            if g.get(field) != o.get(field):
                errors.append(f"{g['name']}: {field} differs")
        if "error" in g:
            continue
        for qg, qo in zip(g["questions"], o["questions"], strict=True):
            for field in PASS_FIELDS:
                if qg.get(field) != qo.get(field):
                    errors.append(f"{g['name']}: {qg.get('id')}: {field} differs")
        if g["response"]["usage"] != o["response"]["usage"]:
            errors.append(
                f"{g['name']}: usage {o['response']['usage']} vs golden {g['response']['usage']}"
            )
        worst = max(
            worst,
            compare_answers(
                g["name"],
                g["response"]["answers"],
                o["response"]["answers"],
                args.tolerance,
                errors,
            ),
        )
    print(f"{args.other}: worst probability difference {worst:.2e}")
    return errors


def post(url: str, path: str, body) -> tuple[int, dict]:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.load(res)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


# The types whose prompt of a question shows the other questions of the
# request: a question cannot be asked alone without changing its prompt.
LISTS_ALL_QUESTIONS = ("nimble", "clef")


def server(args: argparse.Namespace) -> list:
    """Replays the requests on llama-server, its slot erased before each.

    llama-server keeps the prompt of its slot and evaluates only what a new
    pass does not share with it. On a model with a KV cache that changes the
    shapes of the matrix products, hence the kernels' arithmetic, while the
    capture evaluates every pass whole. So a pass is compared at the strict
    tolerance when it is the first of its request after the erase - the first
    question of the full request, and each question asked alone - and at
    `--reuse-tolerance` otherwise (the questions after the first, the second
    order of a question read twice).
    """
    with open(args.golden) as f:
        gold = json.load(f)
    with open(args.requests) as f:
        requests = json.load(f)["requests"]
    splittable = gold["model"]["decision_type"] not in LISTS_ALL_QUESTIONS
    errors: list = []
    worst = 0.0
    worst_reused = 0.0

    def ask(name: str, body, record, strict_ids) -> None:
        nonlocal worst, worst_reused
        post(args.url, "/slots/0?action=erase", {})
        status, answer = post(args.url, "/v1/systemone", body)
        if status != 200:
            message = answer.get("error", {}).get("message")
            if record.get("error") != message:
                errors.append(f"{name}: refused with {message!r}, golden {record.get('error')!r}")
            return
        if "error" in record:
            errors.append(f"{name}: answered, golden refused with {record['error']!r}")
            return
        expected = {qid: record["response"]["answers"][qid] for qid in answer["answers"]}
        if list(answer["answers"]) != list(expected):
            errors.append(f"{name}: question order differs")
            return
        for qid in expected:
            strict = qid in strict_ids
            diff = compare_answers(
                f"{name}/{qid}",
                {qid: expected[qid]},
                {qid: answer["answers"][qid]},
                args.tolerance if strict else args.reuse_tolerance,
                errors,
            )
            if strict:
                worst = max(worst, diff)
            else:
                worst_reused = max(worst_reused, diff)
        if (
            strict_ids
            and len(expected) == len(record["response"]["answers"])
            and answer["usage"] != record["response"]["usage"]
        ):
            errors.append(
                f"{name}: usage {answer['usage']} vs golden {record['response']['usage']}"
            )

    for entry, record in zip(requests, gold["requests"], strict=True):
        if "max_head_tokens" in entry:
            continue  # the server reads the budget from the model only
        name = entry["name"]
        if "error" in record:
            ask(name, entry["body"], record, set())
            continue
        passes: dict = {}
        for trace in record["questions"]:
            passes[trace["id"]] = passes.get(trace["id"], 0) + 1
        ids = list(record["response"]["answers"])
        first = {ids[0]} if passes.get(ids[0], 1) == 1 or None in passes else set()
        if None in passes:
            first = set(ids)  # a joint prompt is one pass for every question
        ask(name, entry["body"], record, first)
        if splittable and None not in passes and len(ids) > 1:
            for qid in ids:
                body = dict(entry["body"])
                body["questions"] = {qid: entry["body"]["questions"][qid]}
                strict = {qid} if passes.get(qid, 1) == 1 else set()
                ask(f"{name} ({qid} alone)", body, record, strict)
    print(
        f"{args.golden} against {args.url}: worst probability difference {worst:.2e} "
        f"for a pass evaluated whole, {worst_reused:.2e} for a pass after a shared prefix"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("golden")
    g.add_argument("golden")
    g.add_argument("other")
    g.add_argument("--tolerance", type=float, default=1e-3)
    s = sub.add_parser("server")
    s.add_argument("golden")
    s.add_argument("requests")
    s.add_argument("url")
    s.add_argument("--tolerance", type=float, default=1e-6)
    s.add_argument("--reuse-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and non-negative")
    errors = golden(args) if args.mode == "golden" else server(args)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
