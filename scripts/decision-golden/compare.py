#!/usr/bin/env python3
"""Compare decision answers against a golden fixture.

  compare.py golden GOLDEN.json OTHER.json [--tolerance T]
      tokens and markers must be equal, probabilities within T

  compare.py server GOLDEN.json REQUESTS.json URL [--tolerance T]
      replays every request the server can take (no head-budget override)
      against a running llama-server and compares its answers and usage

Exit status 1 on the first family of differences, with each one printed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request


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
        for field in ("name", "max_head_tokens"):
            if g[field] != o[field]:
                errors.append(f"{g['name']}: {field} differs")
        for qg, qo in zip(g["questions"], o["questions"], strict=True):
            for field in ("id", "prompt", "rendered_tokens", "tokens", "markers", "column"):
                if qg[field] != qo[field]:
                    errors.append(f"{g['name']}: {qg['id']}: {field} differs")
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


def server(args: argparse.Namespace) -> list:
    with open(args.golden) as f:
        gold = json.load(f)
    with open(args.requests) as f:
        requests = json.load(f)["requests"]
    errors: list = []
    worst = 0.0
    for entry, record in zip(requests, gold["requests"], strict=True):
        if "max_head_tokens" in entry:
            continue  # the server reads the budget from the model only
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone",
            data=json.dumps(entry["body"]).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            body = json.load(res)
        if body["usage"] != record["response"]["usage"]:
            errors.append(
                f"{entry['name']}: usage {body['usage']} vs golden {record['response']['usage']}"
            )
        if list(body["answers"]) != list(record["response"]["answers"]):
            errors.append(f"{entry['name']}: question order differs")
            continue
        worst = max(
            worst,
            compare_answers(
                entry["name"],
                record["response"]["answers"],
                body["answers"],
                args.tolerance,
                errors,
            ),
        )
    print(f"{args.golden} against {args.url}: worst probability difference {worst:.2e}")
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
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and non-negative")
    errors = golden(args) if args.mode == "golden" else server(args)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
