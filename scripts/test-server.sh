#!/usr/bin/env bash
# HTTP control-plane lane: the whole of retrograd-server against a fake model
# probe and a fake run engine. No GGUF, no device, no socket -- `build_router`
# is driven in memory with `tower::ServiceExt::oneshot`.
#
# This is the single place the server's test binaries are named.
# `scripts/test-fast-rust.sh` calls this script rather than repeating the list,
# so the every-commit lane and this one cannot cover different sets.
#
# What is *not* here: `tests/e2e_cpu.rs`, which needs the CPU fixture and runs
# in `scripts/test-cpu-integration.sh`. It is the one case a fake cannot answer
# -- whether the pieces join on a real model.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

# Kept in one array so a new test binary is added once.
binaries=(
  --test api_discovery
  --test api_plan
  --test api_runs
  --test api_control
  --test api_events
  --test api_inference
  --test api_artifacts
  --test api_fork
  --test api_hardening
  --test api_datasets
  --test error_catalog
)

timed_step compile:lib cargo test -p retrograd-server --lib --bins --no-run
timed_step compile:api cargo test -p retrograd-server "${binaries[@]}" --no-run

timed_step run:lib cargo test -p retrograd-server --lib --bins
timed_step run:api cargo test -p retrograd-server "${binaries[@]}"
