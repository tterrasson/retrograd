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

# Keep package selections identical for graph checks, compilation and execution.
# Disabling defaults workspace-wide would also remove the judge's HTTP coverage.
workspace_cpu=(--workspace --exclude retrograd --exclude retrograd-python)
# `cli` is named explicitly because the two binaries carry
# `required-features = ["cli"]`: without it `--bins` below would select
# nothing and the lane would stop compiling them without saying so.
root_cpu=(-p retrograd --no-default-features --features agent,cli)
python_cpu=(-p retrograd-python --no-default-features)
assert_cpu_graph() {
  local tree line found=0
  tree="$(cargo tree "$@" -e normal,build,dev --prefix none --format '{p} features=[{f}]')" || return
  while IFS= read -r line; do
    if [[ "$line" == retrograd-ffi\ * ]]; then
      found=1
      if [[ "$line" =~ features=\[[^]]*(platform-gpu|metal|vulkan|cuda) ]]; then
        echo "GPU feature reached the CPU lane: $line" >&2
        return 1
      fi
    fi
  done <<<"$tree"
  if [[ "$found" != 1 ]]; then
    echo "retrograd-ffi missing from the CPU lane graph: $*" >&2
    return 1
  fi
}
assert_cpu_graph "${workspace_cpu[@]}"
assert_cpu_graph "${root_cpu[@]}"
assert_cpu_graph "${python_cpu[@]}"

timed_step compile cargo test "${workspace_cpu[@]}" --lib --bins --no-run
timed_step compile:root cargo test "${root_cpu[@]}" --lib --bins --no-run
timed_step compile:python cargo test "${python_cpu[@]}" --lib --bins --no-run
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

timed_step run cargo test "${workspace_cpu[@]}" --lib --bins
timed_step run:root cargo test "${root_cpu[@]}" --lib --bins
timed_step run:python cargo test "${python_cpu[@]}" --lib --bins
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
#
# Every tree is captured into a variable and matched with a here-string rather
# than piped into `grep -q`: `grep -q` exits at the first match, the writer
# upstream dies of SIGPIPE, and `pipefail` turns that into a failed pipeline --
# a coin flip on the size of the tree, read as "the dependency is absent".
timed_step run:graph bash -c '
  set -euo pipefail
  for spec in "--no-default-features --features mcp" ""; do
    # shellcheck disable=SC2086
    agent_tree="$(cargo tree -e no-dev -p retrograd-agent $spec)"
    if grep -q bollard <<<"$agent_tree"; then
      echo "bollard reached retrograd-agent with features: ${spec:-default}" >&2
      exit 1
    fi
  done
  # Same property one level up: the CLI runs agentic configs by default, and
  # that must not drag the container backend into every build of it.
  for spec in "--no-default-features" "" "--features mcp"; do
    # shellcheck disable=SC2086
    cli_tree="$(cargo tree -e no-dev -p retrograd $spec)"
    if grep -q bollard <<<"$cli_tree"; then
      echo "bollard reached the retrograd binary with features: ${spec:-default}" >&2
      exit 1
    fi
  done
  # And the switch still works: asking for it compiles it.
  container_tree="$(cargo tree -e no-dev -p retrograd --features container)"
  if ! grep -q bollard <<<"$container_tree"; then
    echo "--features container did not reach bollard" >&2
    exit 1
  fi
  # A judge build whose rewards are all `command` links no TLS stack. The
  # transport lives behind `retrograd-llm-client`, so the optional dependency
  # this asserts on is that crate rather than `reqwest`.
  judge_tree="$(cargo tree -e no-dev -p retrograd-judge --no-default-features)"
  if grep -q rustls <<<"$judge_tree"; then
    echo "rustls reached retrograd-judge without the http feature" >&2
    exit 1
  fi
  scenario_tree="$(cargo tree -e no-dev -p retrograd-scenario-gen)"
  for forbidden in retrograd-engine retrograd-ffi rmcp; do
    if grep -q "$forbidden" <<<"$scenario_tree"; then
      echo "$forbidden reached retrograd-scenario-gen" >&2
      exit 1
    fi
  done
  # The HTTP control plane is a leaf crate with its own binary, and nothing in
  # the workspace depends on it. That is what keeps axum, matchit, utoipa and
  # the OpenAPI derive out of every build of the CLI -- a stronger guarantee
  # than a feature would give, since `--all-features` cannot turn it back on.
  # But `retrograd-server` already has a `[workspace.dependencies]` entry that
  # no manifest consumes, so one absent-minded `retrograd-server.workspace =
  # true` in the root manifest would pull the whole stack back in silently.
  # Matched on names unique to that stack: `tower-http` and `uuid` are *not*
  # among them -- they reach the CLI legitimately through reqwest and rmcp.
  for spec in "--no-default-features" "" "--features mcp" "--features container" "--all-features"; do
    # shellcheck disable=SC2086
    server_tree="$(cargo tree -e no-dev -p retrograd $spec --prefix none | awk "{print \$1}")"
    for forbidden in retrograd-server axum axum-core matchit utoipa utoipa-gen async-stream serde_path_to_error; do
      if grep -qx "$forbidden" <<<"$server_tree"; then
        echo "$forbidden reached the retrograd binary with features: ${spec:-default}" >&2
        exit 1
      fi
    done
  done
  # And the crate that is supposed to carry it still does: an assertion that
  # only ever says "absent" would also pass if the server stopped using axum.
  carrier_tree="$(cargo tree -e no-dev -p retrograd-server --prefix none | awk "{print \$1}")"
  if ! grep -qx axum <<<"$carrier_tree"; then
    echo "axum is no longer in retrograd-server: the assertion above proves nothing" >&2
    exit 1
  fi
  # The Python extension links the library and never a binary, so the terminal
  # presentation layer behind `cli` must not reach it. The regex engine comes
  # from `tracing-subscriber/env-filter` and the terminal stack from
  # `indicatif`; neither has anything to draw on inside a Python process.
  wheel_tree="$(cargo tree -e no-dev -p retrograd-python --prefix none | awk "{print \$1}")"
  for forbidden in comfy-table indicatif retrograd-cli-ui tracing-subscriber crossterm console unicode-width regex-automata; do
    if grep -qx "$forbidden" <<<"$wheel_tree"; then
      echo "$forbidden reached the Python extension: the cli feature leaked" >&2
      exit 1
    fi
  done
  # And the CLI still gets them, so the assertion above is about placement
  # rather than about nobody depending on them any more.
  cli_ui_tree="$(cargo tree -e no-dev -p retrograd --prefix none | awk "{print \$1}")"
  if ! grep -qx comfy-table <<<"$cli_ui_tree"; then
    echo "comfy-table left the CLI: the cli feature no longer carries anything" >&2
    exit 1
  fi'
