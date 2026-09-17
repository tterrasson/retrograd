#!/usr/bin/env bash
# RIR device-parity lane: the generated shaders run on a real GPU and are
# compared to the Loop IR oracle. Model-free, fixture-free, minutes rather than
# seconds -- run it before a PR touching a kernel, an emitter or a schedule.
#
# It runs three device-parity binaries:
#
#   device_parity.rs  one hand-written case per kernel, on the strides a derived
#                     harness cannot produce: views with a gap, tiled schedules,
#                     extents that are no multiple of a workgroup.
#   family_parity.rs  the same property derived from `rir_kernels::registry()`,
#                     so a kernel is covered the day it is registered.
#   dispatch_plan.rs  the multi-dispatch capability:
#                     three dispatches and a scratch return the scan one
#                     dispatch returns. Same skip rules, same oracle discipline;
#                     its timing half stays behind RIR_TIME=1.
#
# They skip themselves, with the reason, when there is no `libvulkan`, no GPU
# and no GLSL compiler. On a machine that has the stack (a macOS laptop with
# MoltenVK and glslc is enough) they run the whole registry through an
# interpreted oracle, which is only affordable in **release** - a debug build
# takes over fifteen minutes. It builds in the shared `target/`: release
# artifacts live in `target/release`, apart from the debug ones, and
# `rir-runtime` does not depend on `retrograd-ffi`, so the `RETRO_BACKENDS` this
# lane pins reconfigures nothing another build uses.
#
#   scripts/test-rir-parity.sh              # all three binaries
#   scripts/test-rir-parity.sh out_prod     # a filter, while iterating
#
# Environment:
#   RIR_PARITY_THREADS    workers in family_parity (default: the machine's cores)
#   RIR_PARITY_TARGET_DIR where to build (default: the workspace's target/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

# No GPU is requested from the *ggml* runtime here: nothing in this lane loads a
# model. The device these tests open is opened by `rir-runtime` directly.
export RETRO_BACKENDS=cpu
export CARGO_TARGET_DIR="${RIR_PARITY_TARGET_DIR:-$repo_root/target}"

timed_step compile cargo test --release -p rir-runtime \
  --test device_parity --test family_parity --test dispatch_plan --no-run

timed_step run:device_parity cargo test --release -p rir-runtime \
  --test device_parity "$@"
timed_step run:family_parity cargo test --release -p rir-runtime \
  --test family_parity "$@"
timed_step run:dispatch_plan cargo test --release -p rir-runtime \
  --test dispatch_plan "$@"
