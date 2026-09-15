#!/usr/bin/env bash
set -euo pipefail

# Materialize the llama.cpp fork submodule on the first run. On subsequent runs
# it only *reports* a checkout that has moved away from the pinned commit,
# moving it back is --pin, and never implicit, because the checkout is where
# fork work happens. The checkout is prepared for that work: retrograd/main
# branch, ggml-org upstream remote.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_PATH="crates/retrograd-ffi/runtime/vendor/llama.cpp"
SOURCE_DIR="${ROOT_DIR}/${SUBMODULE_PATH}"
UPSTREAM_URL="https://github.com/ggml-org/llama.cpp.git"
BRANCH="${RETROGRAD_FORK_BRANCH:-retrograd/main}"

usage() {
  cat <<EOF
Usage: scripts/setup-llama-cpp.sh [--pin]

Fetches and initializes ${SUBMODULE_PATH} on the first run. It also configures
the checkout for development: local ${BRANCH} branch tracking origin, plus an
'upstream' remote pointing at ${UPSTREAM_URL}.

An already-initialized checkout is left where it is: this script reports a
divergence from the pinned commit but never resolves it, so fork commits that
are not yet pinned survive a rerun.

  --pin  Reset an already-initialized checkout back to the pinned commit
         (refuses if the checkout has local changes, or if it would drop
         commits that are on no other branch).
EOF
}

PIN=0
for arg in "$@"; do
  case "${arg}" in
    --pin) PIN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: ${arg}" >&2; usage >&2; exit 2;;
  esac
done

pinned_gitlink() {
  git -C "${ROOT_DIR}" ls-tree HEAD -- "${SUBMODULE_PATH}" |
    awk '$1 == "160000" && $2 == "commit" { print $3 }'
}

if ! git -C "${SOURCE_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "fetching ${SUBMODULE_PATH}..."
  git -C "${ROOT_DIR}" submodule update --init --checkout "${SUBMODULE_PATH}"
else
  pinned="$(pinned_gitlink)"
  head="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${PIN}" == "1" ]]; then
    [[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=no)" ]] || {
      echo "refusing --pin: ${SUBMODULE_PATH} has local changes" >&2
      exit 1
    }
    # Commits reachable only from HEAD would become unreferenced by the reset.
    unreferenced="$(git -C "${SOURCE_DIR}" rev-list --count \
      "${head}" --not "${pinned}" --branches --remotes)"
    if [[ "${unreferenced}" != "0" ]]; then
      echo "refusing --pin: ${unreferenced} commit(s) at ${head} are on no branch" >&2
      echo "  name them first: git -C ${SUBMODULE_PATH} branch <name> ${head}" >&2
      exit 1
    fi
    echo "resetting ${SUBMODULE_PATH} to the pinned commit ${pinned}..."
    git -C "${ROOT_DIR}" submodule update --checkout "${SUBMODULE_PATH}"
  elif [[ "${head}" != "${pinned}" ]]; then
    echo "note: ${SUBMODULE_PATH} is at ${head}, not the pinned ${pinned}" >&2
    echo "      leaving it alone; rerun with --pin to reset it" >&2
  fi
fi

# Development conveniences: work on a branch instead of the detached gitlink,
# and keep the upstream remote available for rebases.
head="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ -z "$(git -C "${SOURCE_DIR}" branch --show-current)" ]]; then
  if git -C "${SOURCE_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    if [[ "$(git -C "${SOURCE_DIR}" rev-parse "refs/heads/${BRANCH}")" == "${head}" ]]; then
      git -C "${SOURCE_DIR}" checkout --quiet "${BRANCH}"
    else
      echo "note: staying detached at ${head}; local ${BRANCH} points elsewhere" >&2
    fi
  else
    git -C "${SOURCE_DIR}" checkout --quiet -b "${BRANCH}" --track "origin/${BRANCH}" 2>/dev/null ||
      git -C "${SOURCE_DIR}" checkout --quiet -b "${BRANCH}"
  fi
fi

if ! git -C "${SOURCE_DIR}" remote get-url upstream >/dev/null 2>&1; then
  git -C "${SOURCE_DIR}" remote add upstream "${UPSTREAM_URL}"
fi

current="$(git -C "${SOURCE_DIR}" branch --show-current)"
echo "llama.cpp fork submodule ready:"
echo "  path:   ${SOURCE_DIR}"
echo "  commit: $(git -C "${SOURCE_DIR}" rev-parse HEAD)"
echo "  branch: ${current:-detached}"
