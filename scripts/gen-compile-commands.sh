#!/usr/bin/env bash
# Regenerate the C/C++ compilation database used by clangd (IDE only - the
# shipping build goes through crates/retrograd-ffi/build.rs).
#
# Run this after adding/removing a runtime source or changing include paths;
# the resulting compile_commands.json is what.clangd points at.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$repo_root/crates/retrograd-ffi/runtime"

if [[ ! -f "$runtime_dir/vendor/llama.cpp/include/llama.h" ]]; then
    echo "llama.cpp submodule missing; run scripts/setup-llama-cpp.sh first" >&2
    exit 1
fi

cmake -S "$runtime_dir" -B "$runtime_dir/build" -DCMAKE_BUILD_TYPE=Debug >/dev/null
echo "wrote $runtime_dir/build/compile_commands.json"
