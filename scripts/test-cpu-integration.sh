#!/usr/bin/env bash
# CPU-only model lane: a missing fixture is an error, never a green skip.
#
#   scripts/test-cpu-integration.sh                              # the whole lane
#   scripts/test-cpu-integration.sh grpo_runtime [filter ...]    # one binary, while iterating
#
# RETRO_TEST_CPU_THREADS sets the llama.cpp CPU workers (default 4).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"
"$repo_root/scripts/fetch-cpu-fixture.sh"

export RETRO_CPU_FIXTURE="${RETRO_CPU_FIXTURE:-$repo_root/tests/fixtures/LFM2.5-230M-Q4_K_M.gguf}"
export RETRO_REQUIRE_CPU_FIXTURE=1
export RETRO_RUNTIME_LOCK_PATH="${RETRO_RUNTIME_LOCK_PATH:-${TMPDIR:-/tmp}/retrograd-runtime.lock}"
export RETRO_THREADS="${RETRO_TEST_CPU_THREADS:-4}"
export RETRO_TEST_TIMING="${RETRO_TEST_TIMING:-1}"

# One expensive binary on its own, with the fixture, backend, lock and thread
# guarantees of the full lane.
if [[ $# -gt 0 ]]; then
  test_binary="$1"
  shift
  exec cargo test --no-default-features --features agent --test "$test_binary" -- --test-threads=1 "$@"
fi

# Compilation profiling is optional: a normal `cargo test` already builds
# missing artifacts, so unconditional `--no-run` passes only add Cargo startup
# and dependency-scanning overhead to warm lane runs.
if [[ "${RETRO_PROFILE_TESTS:-0}" == "1" ]]; then
  timed_step compile cargo test --no-default-features --features agent --no-run \
    --test capabilities \
    --test trainable_inventory \
    --test base_training \
    --test ppo_runtime \
    --test grpo_runtime \
    --test distill_runtime \
    --test distill_topk \
    --test checkpoint_resume \
    --test cli \
    --test engine_contracts \
    --test kv_projection_gradients \
    --test recurrent_families \
    --test lora_resume \
    --test lora_f16 \
    --test train_parity \
    --test gradient_checkpointing \
    --test fused_ce_parity \
    --test ubatch_parity
  timed_step compile:server cargo test -p retrograd-server --test e2e_cpu --no-run
fi

# These binaries contain the focused capability, LoRA/generation/SFT/PPO,
# GRPO, distillation-teacher, offline top-k distillation, checkpoint and CLI
# smoke tests. Cargo may start them concurrently; `serialize_models` supplies
# the cross-process runtime lock.
timed_step run:capabilities-suite cargo test --no-default-features --features agent \
  --test capabilities \
  --test ppo_runtime \
  --test grpo_runtime \
  --test distill_runtime \
  --test distill_topk \
  --test checkpoint_resume \
  --test cli \
  --test engine_contracts \
  --test kv_projection_gradients \
  --test recurrent_families \
  --test trainable_inventory \
  --test base_training \
  -- --test-threads=1

# CPU-only slices of mixed CPU/GPU binaries, merged into a single cargo start.
# The libtest harness accepts multiple FILTERS (OR-matched by substring, not
# just one), so one invocation can select the CPU-only test from each binary
# without pulling in its Metal/CUDA siblings. Each filter below is verified
# unique to its own test within this binary set (no cross-binary collision):
# `recompute_matches_the_` matches only the two intended gradient_checkpointing
# cases, not `recompute_matches_differentiable_vulkan_attention` (Vulkan-only,
# in the same file). `fused_ce_parity` and `ubatch_parity` have no non-CPU
# sibling tests, so their test names are listed as filters too rather than
# left unfiltered, since any filter argument restricts every selected binary.
timed_step run:cpu-slices cargo test --no-default-features --features agent \
  --test lora_resume \
  --test lora_f16 \
  --test train_parity \
  --test gradient_checkpointing \
  --test fused_ce_parity \
  --test ubatch_parity \
  -- --test-threads=1 \
  cpu_resume_trains_a_loaded_adapter \
  cpu_f16_lora_trains_and_round_trips_for_edge_ranks \
  cpu_end_to_end \
  recompute_matches_the_ \
  offloaded_log_softmax_matches_the_fused_path_end_to_end \
  every_micro_batch_is_finite_and_preserves_the_characterized_difference

# The HTTP control plane, end to end on the fixture: plan with the measured pass,
# create, pause, evaluate and generate against the paused model, checkpoint,
# cancel, list, fork. Every other server test runs against a fake engine in
# `scripts/test-server.sh`; this is the one case a fake cannot answer -- whether
# the pieces join on a real model. Separate cargo start because it is a different
# package, and `--test-threads=1` for the same reason as everything above: it
# loads a model.
timed_step run:server-e2e cargo test -p retrograd-server --test e2e_cpu -- --test-threads=1
