#!/usr/bin/env bash
# Web lane: the interface under web/, and the server route that serves it.
#
# In order: the locked install, the contract (web/openapi.json and the types
# generated from it are what this build serves), lint and types, the unit and
# component tests, the licences of what ships, the production build and its
# size budget - and then, unless `--no-cargo`, the server's `ui` tests against
# the `web/dist` just built, which is the only place the real embedded assets
# are served.
#
# `--no-cargo` is for a machine with Bun and no Rust toolchain (the CI web
# job); the Rust half runs where the dist is handed over.
#
# Playwright is not here: it needs browsers and is its own step,
# `bun run test:e2e`.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/lib-step-timing.sh"

cargo_half=1
for argument in "$@"; do
  case "$argument" in
    --no-cargo) cargo_half=0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

cd "$repo_root/web"
timed_step install bun install --frozen-lockfile
timed_step check:api bun run check:api
timed_step lint bun run lint
timed_step test bun run test
timed_step check:licenses bun run check:licenses
# `build` type-checks (vue-tsc) before bundling.
timed_step build bun run build
timed_step check:bundle bun run check:bundle

if [[ "$cargo_half" == 1 ]]; then
  cd "$repo_root"
  timed_step run:ui cargo test -p retrograd-server --features ui --test api ui::
fi
