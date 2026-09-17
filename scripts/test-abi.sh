#!/usr/bin/env bash
# ABI/contract lane: runtime build, device registration and hand-written
# kernels checked against the CPU reference. No model is loaded, so this lane
# never needs the CPU fixture.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

# Unset means platform default; explicitly empty means CPU only.
backends="${RETRO_ABI_FEATURES-platform-gpu}"
# CPU only needs the root's defaults off. A named backend keeps them, as a user
# build would: the explicit backend replaces `platform-gpu`, and FFI gets the
# same feature set as under the root, so both groups share one native variant.
root_args=(--no-default-features --features agent)

ffi_args=()
if [[ -n "$backends" ]]; then
  if [[ "$backends" == ,* || "$backends" == *, || "$backends" == *,,* || "$backends" == *[[:space:]]* ]]; then
    echo "RETRO_ABI_FEATURES must be a comma-separated list of backend features" >&2
    exit 2
  fi
  IFS=, read -r -a names <<<"$backends"
  for name in "${names[@]}"; do
    case "$name" in
      platform-gpu|metal|vulkan|cuda) ;;
      *) echo "invalid RETRO_ABI_FEATURES backend: $name (CPU is implicit)" >&2; exit 2 ;;
    esac
  done
  root_args=(--features "$backends")
  ffi_args=(--features "platform-gpu,$backends")
fi

# Two precompile calls, matching the two ways the run step below selects
# targets: retrograd-ffi's own lib tests need -p, the four root-package
# `tests/*.rs` binaries need --test. Cargo rejects mixing -p with --test for
# a different package in one invocation.
timed_step compile:ffi-lib cargo test ${ffi_args[@]+"${ffi_args[@]}"} --no-run -p retrograd-ffi --lib
timed_step compile:root-tests cargo test "${root_args[@]}" --no-run \
  --test backend_devices --test metal_ops --test weighted_ce \
  --test gated_delta_net_chunked --test flash_attn_back --test out_prod_quant \
  --test engine_contracts --test rir_probe --test rir_quant_oracle

timed_step run:ffi-lib cargo test ${ffi_args[@]+"${ffi_args[@]}"} --lib -p retrograd-ffi
timed_step run:backend_devices cargo test "${root_args[@]}" --test backend_devices
timed_step run:metal_ops cargo test "${root_args[@]}" --test metal_ops
timed_step run:weighted_ce cargo test "${root_args[@]}" --test weighted_ce
# CPU only, no model: the two implementations of GATED_DELTA_NET_BACK against
# each other. Lives here rather than in the cuda lane because it is what tells a
# wrong derivation apart from a wrong kernel.
timed_step run:gated_delta_net_chunked cargo test "${root_args[@]}" --test gated_delta_net_chunked
# Model-free native CPU backward against the independent analytic streaming
# oracle, including the fixed-order dK/dV determinism contract.
timed_step run:flash_attn_back cargo test "${root_args[@]}" --test flash_attn_back
# CPU `out_prod` decoding in place across the complete shared quantization
# table, without a model.
# `--test-threads=1`: the budget case forces `GGML_CUDA_DEQUANT_BUDGET_MB`
# through `common::EnvGuard`, and mutating the process environment is only
# sound while no sibling test is reading it.
timed_step run:out_prod_quant cargo test "${root_args[@]}" --test out_prod_quant -- --test-threads=1
timed_step run:engine_contracts cargo test "${root_args[@]}" --test engine_contracts \
  trainer_new_reports_a_clean_error_for_a_missing_model_path
# RIR contract at the FFI boundary. Run twice on purpose:
# the RIR policy is fixed before the backend context exists, so "RIR refused
# cleanly when off" and "RIR really dispatched under prefer" cannot be asserted
# in the same process.
# The quantized-format table and the RIR dequantization oracle against ggml's
# own decoder, on identical bytes. Model-free.
timed_step run:rir_quant_oracle cargo test "${root_args[@]}" --test rir_quant_oracle
# `off` is now *explicit*, and it has to be: an
# absent RETRO_RIR_MODE means `prefer`, so this pass would otherwise have become
# a duplicate of the next one - and the two assertions it carries ("a RIR
# request off the mode is a clean error", "the native path is what runs") would
# have gone quietly unexercised.
RETRO_RIR_MODE=off \
  timed_step run:rir_probe_off cargo test "${root_args[@]}" --test rir_probe
RETRO_RIR_MODE=prefer \
  timed_step run:rir_probe_prefer cargo test "${root_args[@]}" --test rir_probe
# `require` needs two more processes of its own: one
# where every targeted node is eligible, which must dispatch RIR with no
# fallback, and one where the test-only contract injection makes the op
# ineligible, which must fail the graph before anything is encoded. The second
# is filtered to the require tests: the injection deliberately breaks the op, so
# the rest of the binary has nothing to assert in that process.
RETRO_RIR_MODE=require \
  timed_step run:rir_probe_require cargo test "${root_args[@]}" --test rir_probe
RETRO_RIR_MODE=require RETRO_RIR_TEST_REJECT=L2_NORM_BACK \
  timed_step run:rir_probe_require_reject cargo test "${root_args[@]}" --test rir_probe require_
# The second integrated op. The registry pins CUMSUM to
# `observe_generated`, so the pass above already asserts that it is measured and
# never encoded; this one promotes it for the process so the differential matrix
# - RIR against the native scan and against the f64 reference - can run at all.
# That measurement is the input to the promotion decision, not the promotion.
RETRO_RIR_MODE=prefer RETRO_RIR_TEST_PREFER=CUMSUM \
  timed_step run:rir_probe_cumsum cargo test "${root_args[@]}" --test rir_probe cumsum
unset RETRO_RIR_MODE
