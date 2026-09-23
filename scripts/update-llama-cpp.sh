#!/usr/bin/env bash
set -euo pipefail

# Manual, reviewable upstream update helper. It deliberately does not push or
# commit the submodule pointer: review the rebase and tests, then publish with
# scripts/push-llama-cpp-fork.sh --force-with-lease.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${LLAMA_CPP_DIR:-${ROOT_DIR}/crates/retrograd-ffi/runtime/vendor/llama.cpp}"
BRANCH="${RETROGRAD_FORK_BRANCH:-retrograd/main}"
UPSTREAM_REF="${1:-upstream/master}"

git -C "${SOURCE_DIR}" rev-parse HEAD >/dev/null 2>&1 || {
  echo "llama.cpp submodule not initialized at ${SOURCE_DIR}; run scripts/setup-llama-cpp.sh" >&2
  exit 1
}

# A rebase rewrites whatever HEAD is. Detached, the rewritten series would be
# reachable from no ref at all; on another branch it would rewrite the wrong
# history. Both are recoverable only through the reflog, so refuse instead.
current="$(git -C "${SOURCE_DIR}" branch --show-current)"
if [[ "${current}" != "${BRANCH}" ]]; then
  echo "refusing to rebase: ${SOURCE_DIR} is on '${current:-a detached HEAD}', not ${BRANCH}" >&2
  echo "  git -C ${SOURCE_DIR#"${ROOT_DIR}/"} checkout ${BRANCH}" >&2
  exit 1
fi

if [[ -e "$(git -C "${SOURCE_DIR}" rev-parse --git-path rebase-merge)" ||
      -e "$(git -C "${SOURCE_DIR}" rev-parse --git-path rebase-apply)" ]]; then
  echo "refusing to rebase: a rebase is already in progress in ${SOURCE_DIR}" >&2
  exit 1
fi

git -C "${SOURCE_DIR}" diff --quiet || {
  echo "refusing to rebase: ${SOURCE_DIR} has unstaged changes" >&2
  exit 1
}
git -C "${SOURCE_DIR}" diff --cached --quiet || {
  echo "refusing to rebase: ${SOURCE_DIR} has staged changes" >&2
  exit 1
}

git -C "${SOURCE_DIR}" fetch upstream
git -C "${SOURCE_DIR}" rev-parse --verify --quiet "${UPSTREAM_REF}^{commit}" >/dev/null || {
  echo "unknown upstream ref: ${UPSTREAM_REF}" >&2
  exit 1
}

echo "rebasing ${BRANCH} ($(git -C "${SOURCE_DIR}" rev-parse --short HEAD)) onto ${UPSTREAM_REF}..."
# --autosquash folds the `fixup! retro(<family>)` commits made since the last
# sync into their family, so the series is one commit per family again.
git -C "${SOURCE_DIR}" rebase --autosquash "${UPSTREAM_REF}"

# Reuse before writing. A rebase is the only moment
# where a local patch can have become redundant - upstream gains ops over time,
# and a family that duplicates upstream is patch to delete, not to merge. The
# probes are advisory here on purpose: the lockfile still points at the *old*
# upstream commit at this stage, so a hit means "upstream had it already", and a
# miss means nothing until step 2 below moves the base.
echo
echo "Checking patch families against upstream (advisory until upstream_commit is updated):"
"${SCRIPT_DIR}/check-llama-cpp-integration.sh" --upstream-status || true

cat <<EOF

Rebase complete. Before publishing:
  1. Test with the lane scripts, not a raw cargo test (docs/engineering/tests/notice.md):
       scripts/test-fast-rust.sh && scripts/test-fast-python.sh &&
         scripts/test-abi.sh && scripts/test-cpu-integration.sh
     Add the rir and rir-graph lanes when the rebase touched a RIR kernel,
     schedule, emitter or registry.
  2. Set upstream_commit in crates/retrograd-ffi/runtime/llama.cpp.lock to: $(git -C "${SOURCE_DIR}" rev-parse "${UPSTREAM_REF}")
     then re-run: scripts/check-llama-cpp-integration.sh --upstream-status
     Any probe it reports as present upstream is a family to **delete**, not to
     rebase. Record PR links and revised dispositions in [upstream_status].
  3. Review the rebased patch series, then publish fork + submodule bump with:
       scripts/push-llama-cpp-fork.sh --force-with-lease
EOF
