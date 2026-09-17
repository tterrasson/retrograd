#!/usr/bin/env bash
# Fast Python lane: builds the native extension CPU-only, then runs the full
# Python suite (fake-native unit tests plus the PyO3 boundary smoke). CPU-only
# is required here: building with the macOS default (cpu,metal) lets the real
# fixture smoke test reach a Metal buffer allocation that segfaults when the
# Metal queue cannot be created outside a GPU-capable runner.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"
cd "$repo_root/python"

uv_cpu=(uv run --config-settings-package 'retrograd:build-args=--no-default-features')

# Every tracked Python file, not only the package: `ruff.toml` sits at the root.
timed_step lint "${uv_cpu[@]}" ruff check . ../examples ../scripts
timed_step format "${uv_cpu[@]}" ruff format --check . ../examples ../scripts
timed_step compile "${uv_cpu[@]}" --reinstall-package retrograd python -c "import retrograd; b = retrograd.list_backends(); assert any(x.kind == 'cpu' for x in b) and not any(x.kind == 'gpu' for x in b), b"
timed_step run "${uv_cpu[@]}" pytest -q "$@"
