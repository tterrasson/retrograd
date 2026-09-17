#!/usr/bin/env bash
# RIR coverage lane on a **real** training graph (ADR-5 section 5).
#
# `scripts/test-rir.sh` judges a kernel in isolation, on the shapes a
# conformance bench enumerates. This lane checks the condition for removing a
# native kernel: on a graph a model actually
# trains, did every node that went native do so for a reason the registry
# *declares* - and does every pair whose native is retired still serve all of
# its nodes, since there is no longer anything else to serve them.
#
#   scripts/test-rir-graph.sh                  # every GPU backend present
#   scripts/test-rir-graph.sh --backend metal  # one backend
#   scripts/test-rir-graph.sh --backend cuda   # the third one
#
# It loads a model, which is what separates it from `test-rir.sh` and what makes
# it slow. The model is the versioned CPU fixture by default, so the numbers the
# plan publishes are reproducible rather than dependent on whatever GGUF the
# runner happens to have; `RETRO_RIR_TEST_MODEL` overrides it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

backends_arg="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backends_arg="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done

# The model. Fetched and checksum-verified like the CPU lane's, so "the graph
# the plan measured" is a file this repository pins and not a local artefact.
if [[ -z "${RETRO_RIR_TEST_MODEL:-}" ]]; then
    "$repo_root/scripts/fetch-cpu-fixture.sh"
    export RETRO_RIR_TEST_MODEL="$repo_root/tests/fixtures/LFM2.5-230M-Q4_K_M.gguf"
fi
[[ -f "$RETRO_RIR_TEST_MODEL" ]] || {
    echo "model not found: $RETRO_RIR_TEST_MODEL" >&2; exit 1
}

# What this default model cannot show, said here rather than discovered as a
# green run. The graph a model produces decides which
# pairs this lane can judge at all, and `L2_NORM_BACK` - the one pair whose
# native kernel is **gone** on CUDA - is a `GATED_DELTA_NET` op: the census
# needed a second, hybrid model to see a single one of its nodes. On a
# graph without them the pair simply has no site, every assertion about it holds
# vacuously, and the lane says OK about a kernel it never reached.
#
# So the fixture stays the default - it is what makes the published numbers
# reproducible - and the *absence* is printed, with the override that closes it.
echo "model : $RETRO_RIR_TEST_MODEL"
echo "         a graph with no GATED_DELTA_NET produces no L2_NORM_BACK, hence"
echo "         no site for the pair whose native is retired on CUDA."
echo "         RETRO_RIR_TEST_MODEL=<GDN gguf> to cover it."

# A lane that skips is a lane that lies. Every reason `rir_graph_coverage` has
# to step aside - no GPU, no model, a graph with no registered op - becomes a
# failure here, which is the whole difference between a test someone runs by
# hand and a lane.
export RETRO_REQUIRE_RIR_GRAPH=1
# Set `prefer` explicitly so this lane remains meaningful if the build default changes.
export RETRO_RIR_MODE="${RETRO_RIR_MODE:-prefer}"
export RETRO_RUNTIME_LOCK_PATH="${RETRO_RUNTIME_LOCK_PATH:-${TMPDIR:-/tmp}/retrograd-runtime.lock}"

# --- MoltenVK: the detected default rather than two variables to remember.
# Same block as scripts/test-rir.sh: `<prefix>/etc/vulkan/icd.d/MoltenVK_icd.json`
# goes with `<prefix>/lib`, and neither is on the default path.
if [[ "$(uname -s)" == "Darwin" && -z "${VK_ICD_FILENAMES:-}" ]]; then
    for prefix in /opt/homebrew /usr/local; do
        icd="$prefix/etc/vulkan/icd.d/MoltenVK_icd.json"
        if [[ -f "$icd" ]]; then
            export VK_ICD_FILENAMES="$icd"
            export DYLD_LIBRARY_PATH="$prefix/lib:${DYLD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

case "$backends_arg" in
    all)
        wanted=()
        [[ "$(uname -s)" == "Darwin" ]] && wanted+=(metal)
        [[ -n "${VK_ICD_FILENAMES:-}" || "$(uname -s)" != "Darwin" ]] && wanted+=(vulkan)
        # CUDA on the same principle `scripts/test-rir.sh` applies to it and to
        # Metal: only where a toolkit exists. Without one
        # the build selects no CUDA backend and the run below would be a Vulkan
        # run under a CUDA heading.
        command -v "${CUDACXX:-nvcc}" >/dev/null 2>&1 && wanted+=(cuda)
        ;;
    metal|vulkan|cuda) wanted=("$backends_arg") ;;
    *) echo "unknown backend: $backends_arg" >&2; exit 2;;
esac
[[ ${#wanted[@]} -gt 0 ]] || { echo "no plausible GPU backend on this machine" >&2; exit 1; }

# One process per backend, and that is not a convenience: the RIR mode and the
# backend list are both read once, before the first context exists, so two
# backends in one process would measure whichever one the scheduler picked.
#
# Feature variants keep their native outputs separate in the shared Cargo tree.
# CUDA still sets native inputs; keep those and caller RUSTFLAGS constant when timing.
for be in "${wanted[@]}"; do
    echo "== real graph on $be"
    native_env=(env)
    if [[ "$be" == "cuda" ]]; then
        # One architecture and not the whole list, for the reason `test-rir.sh`
        # gives: a fat binary costs minutes of `nvcc` per translation unit and
        # this lane measures the GPU that is here.
        native_env+=(RETRO_CUDA_ARCHITECTURES="${RETRO_CUDA_ARCHITECTURES:-native}")
    fi
    timed_step "run:$be" "${native_env[@]}" cargo test --release --features "$be" \
        --test rir_graph_coverage -- --nocapture --test-threads=1 \
        the_real_graph_never_falls_back_silently_under_prefer
done

echo "test-rir-graph: OK"
