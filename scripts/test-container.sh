#!/usr/bin/env bash
# Container lane: the parts of retrograd-container that only a real daemon can
# check -- limits actually applied by the runtime, isolation between two
# episodes on a recycled container, byte-identical observations across
# containers, and the reaper.
#
# This is the first lane in the repository that depends on a service rather than
# on the machine, so it does something no other lane does: when no daemon
# answers, it announces itself SKIPPED and exits non-zero unless
# RETRO_CONTAINER_OPTIONAL=1. A lane that silently passes because its
# prerequisite was missing is worse than no lane at all -- the properties it
# guards are exactly the ones whose absence is invisible.
#
# Run it before a PR that touches retrograd-container. The pool's policy, the
# spec's refusals and the reaper's rule are unit tests and stay in fast-rust.
#
# The daemon may be Docker Desktop, colima, or Podman in Docker-compatible mode;
# DOCKER_HOST covers all three. The image needs GNU find and coreutils timeout
# (any Debian-based one does); override with RETRO_CONTAINER_TEST_IMAGE.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

image="${RETRO_CONTAINER_TEST_IMAGE:-debian:bookworm-slim}"

probe() {
  if command -v docker >/dev/null 2>&1; then
    docker info >/dev/null 2>&1 && return 0
  fi
  if command -v podman >/dev/null 2>&1; then
    podman info >/dev/null 2>&1 && return 0
  fi
  return 1
}

if ! probe; then
  echo "SKIPPED: no container daemon answers (DOCKER_HOST=${DOCKER_HOST:-unset})." >&2
  echo "         Start Docker Desktop, colima or 'podman machine start', then re-run." >&2
  if [[ "${RETRO_CONTAINER_OPTIONAL:-0}" == "1" ]]; then
    exit 0
  fi
  exit 1
fi

# Pulled here rather than inside the first test: a multi-gigabyte download on a
# cold cache would otherwise look like a hung test.
if command -v docker >/dev/null 2>&1; then
  timed_step pull docker pull "$image"
elif command -v podman >/dev/null 2>&1; then
  timed_step pull podman pull "$image"
fi

timed_step compile cargo test -p retrograd-container --tests --no-run
# Single-threaded: the tests share one daemon and assert on how many containers
# are labelled as theirs, which only holds if they do not overlap.
timed_step run cargo test -p retrograd-container --tests -- --test-threads=1

# Nothing may survive the suite, including after a failing test: leftover
# containers are what a training machine dies of overnight.
leftovers=0
if command -v docker >/dev/null 2>&1; then
  leftovers=$(docker ps -a --filter "label=retrograd.pool" --format '{{.ID}}' | wc -l | tr -d ' ')
elif command -v podman >/dev/null 2>&1; then
  leftovers=$(podman ps -a --filter "label=retrograd.pool" --format '{{.ID}}' | wc -l | tr -d ' ')
fi
if [[ "$leftovers" != "0" ]]; then
  echo "FAIL: $leftovers sandbox container(s) survived the suite" >&2
  exit 1
fi
echo "no sandbox containers left behind"
