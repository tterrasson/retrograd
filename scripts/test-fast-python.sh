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

export RETRO_BACKENDS=cpu

# Every tracked Python file, not only the package: `ruff.toml` sits at the root.
timed_step lint uv run ruff check . ../examples ../scripts
timed_step format uv run ruff format --check . ../examples ../scripts
timed_step compile uv run --reinstall-package retrograd python -c "import retrograd"
timed_step run uv run pytest -q "$@"
