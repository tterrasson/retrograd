#!/usr/bin/env bash
# HTTP control-plane lane: the whole of retrograd-server against a fake model
# probe and a fake run engine. No GGUF, no device, no socket -- `build_router`
# is driven in memory with `tower::ServiceExt::oneshot`.
#
# This is the single place the server's test binary is named.
# `scripts/test-fast-rust.sh` calls this script rather than repeating the list,
# so the every-commit lane and this one cannot cover different sets.
#
# What is *not* here: `tests/e2e_cpu.rs`, which needs the CPU fixture and runs
# in `scripts/test-cpu-integration.sh`. It is the one case a fake cannot answer
# -- whether the pieces join on a real model.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

# One binary: every `tests/api/*.rs` is a module of `tests/api/main.rs`, so a
# new test file is declared there and needs nothing here.
binaries=(--test api)

timed_step compile:lib cargo test -p retrograd-server --lib --bins --no-run
timed_step compile:api cargo test -p retrograd-server "${binaries[@]}" --no-run

timed_step run:lib cargo test -p retrograd-server --lib --bins
timed_step run:api cargo test -p retrograd-server "${binaries[@]}"
