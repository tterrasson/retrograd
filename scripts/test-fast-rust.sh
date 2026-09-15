#!/usr/bin/env bash
# Fast Rust lane: unit tests and binaries, no GPU, no fixture, no integration
# binaries under the root package's tests/. Safe to run on every commit.
#
# Two exceptions, for the same reason: retrograd-server's HTTP tests drive
# `build_router` in memory with a fake model probe and a fake run engine, and
# retrograd-plan's resolver tests run on synthetic model geometries. Neither needs
# a model or a device, and the wire contract, the run state machine and the cost
# model are the last things that should only be checked before a PR -- a
# regression in any of them is silent.
#
# The server half is delegated to `scripts/test-server.sh`, which names its test
# binaries. Calling it instead of repeating the list is what keeps the two lanes
# from covering different sets; run it on its own while iterating on a handler.
#
# `rir-runtime`'s `device_parity.rs` and `family_parity.rs` are compiled here
# but run by `scripts/test-rir-parity.sh`: on a machine with a Vulkan stack they
# run the whole registry through an interpreted oracle, which a debug build
# cannot afford. Compiling them here makes a parity test that stops building
# fail on the commit that broke it, not on the day someone runs the GPU lane.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

export RETRO_BACKENDS=cpu
# A build tree of this lane's own, for the reason `.gitignore` already gives the
# per-backend ones: `RETRO_BACKENDS` is part of the build fingerprint, and this
# lane pins `cpu` while a plain `cargo build` on macOS resolves to `cpu,metal`
# and `scripts/test-abi.sh` asks for it explicitly. Sharing one tree with them
# means reconfiguring llama.cpp and relinking the workspace on every alternation
# -- 25 to 60 s per flip, measured, for nothing this lane uses.
# It lives *under* `target/` rather than beside it: separate fingerprints, one
# directory to size, ignore and clean. Set RETRO_FAST_TARGET_DIR=target to share
# the default tree back (one build tree instead of two, at that price per switch).
export CARGO_TARGET_DIR="${RETRO_FAST_TARGET_DIR:-$repo_root/target/lanes/fast}"

timed_step compile cargo test --workspace --lib --bins --no-run
timed_step compile:plan cargo test -p retrograd-plan --tests --no-run
timed_step compile:mcp cargo test -p retrograd-tools --test mcp_stdio --no-run
# The other half of the judge's `http` feature: the graph check
# below proves `rustls` is absent, but only a build proves the crate still
# compiles without it. One ungated `use` of the transport is enough to break it,
# and nothing else in the lane builds this feature combination.
timed_step compile:judge-no-http cargo check -p retrograd-judge --no-default-features
# Compiled, not run: the lane that runs them is `scripts/test-rir-parity.sh`.
timed_step compile:rir cargo test -p rir-runtime --test device_parity --test family_parity --no-run
# rir-gen's tests live in tests/, so `--lib` does not reach them: they have to
# be named. Same for the F16 oracle comparison, which lives in rir-kernels/tests/.
timed_step compile:rir-gen cargo test -p rir-gen --tests --no-run
timed_step compile:rir-f16 cargo test -p rir-kernels --test f16_oracles --no-run
# Name public-contract integration tests explicitly because `--lib` never
# reaches a `tests/` directory, and a test binary no lane names is dead
# coverage. None of the four needs a model or a device;
# `chat_parser_roundtrip` runs on chat-template fixtures, which is exactly why
# it belongs here and not in a model lane.
timed_step compile:contracts cargo test \
  -p retrograd-agent --test train_sequences \
  -p retrograd-engine --test chat_parser_roundtrip \
  -p retrograd-judge --test reward_batch \
  -p retrograd-training --test value_head \
  --no-run

timed_step run cargo test --workspace --lib --bins
timed_step run:plan cargo test -p retrograd-plan --tests
# The stdio MCP transport end to end: the test re-executes this same binary as
# the server, so it needs no daemon and no network.
timed_step run:mcp cargo test -p retrograd-tools --test mcp_stdio
# The generation lane: regeneration produces no diff, the tables validate, the
# fork copies match. Two of its cases spawn one compiler process per generated
# source -- glslc over the `.comp`, `xcrun metal` over the MSL, nvcc over the
# `.cu` where a toolkit exists. They stay in this lane because they need no
# device, and they run across the machine's cores (~2.4 s).
timed_step run:rir-gen cargo test -p rir-gen --tests
# The three F16 conversions, compared over all 65 536 halves. It is the
# test that found both copies decoding subnormals one binade too small.
timed_step run:rir-f16 cargo test -p rir-kernels --test f16_oracles
timed_step run:contracts cargo test \
  -p retrograd-agent --test train_sequences \
  -p retrograd-engine --test chat_parser_roundtrip \
  -p retrograd-judge --test reward_batch \
  -p retrograd-training --test value_head
timed_step run:server "$repo_root/scripts/test-server.sh"

# The optionality property of the MCP-only build, checked on the graph
# rather than trusted: "MCP tools only" must not compile the container backend.
# It is one distracted `[dependencies]` line away from being lost, and nothing
# else in the lane would notice.
timed_step run:graph bash -c '
  set -euo pipefail
  for spec in "--no-default-features --features mcp" ""; do
    # shellcheck disable=SC2086
    if cargo tree -e no-dev -p retrograd-agent $spec | grep -q bollard; then
      echo "bollard reached retrograd-agent with features: ${spec:-default}" >&2
      exit 1
    fi
  done
  # Same property one level up: the CLI runs agentic configs by default, and
  # that must not drag the container backend into every build of it.
  for spec in "--no-default-features" "" "--features mcp"; do
    # shellcheck disable=SC2086
    if cargo tree -e no-dev -p retrograd $spec | grep -q bollard; then
      echo "bollard reached the retrograd binary with features: ${spec:-default}" >&2
      exit 1
    fi
  done
  # And the switch still works: asking for it compiles it.
  if ! cargo tree -e no-dev -p retrograd --features container | grep -q bollard; then
    echo "--features container did not reach bollard" >&2
    exit 1
  fi
  # A judge build whose rewards are all `command` links no TLS stack. The
  # transport lives behind `retrograd-llm-client`, so the optional dependency
  # this asserts on is that crate rather than `reqwest`.
  if cargo tree -e no-dev -p retrograd-judge --no-default-features | grep -q rustls; then
    echo "rustls reached retrograd-judge without the http feature" >&2
    exit 1
  fi
  scenario_tree="$(cargo tree -e no-dev -p retrograd-scenario-gen)"
  for forbidden in retrograd-engine retrograd-ffi rmcp; do
    if grep -q "$forbidden" <<<"$scenario_tree"; then
      echo "$forbidden reached retrograd-scenario-gen" >&2
      exit 1
    fi
  done'
