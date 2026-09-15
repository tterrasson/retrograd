#!/usr/bin/env bash
set -euo pipefail

# Publish the llama.cpp fork submodule: push its HEAD to the fork branch on
# origin, then commit and push the submodule pointer bump in retrograd.
# Pushes are fast-forward by default; pass --force-with-lease after a rebase.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_PATH="crates/retrograd-ffi/runtime/vendor/llama.cpp"
SOURCE_DIR="${ROOT_DIR}/${SUBMODULE_PATH}"
LOCKFILE="${LLAMA_CPP_LOCKFILE:-${ROOT_DIR}/crates/retrograd-ffi/runtime/llama.cpp.lock}"
BRANCH="${RETROGRAD_FORK_BRANCH:-retrograd/main}"
ASSUME_YES=false
DRY_RUN=false
FORCE_WITH_LEASE=false

lock_value() {
  sed -nE "s/^$1[[:space:]]*=[[:space:]]*\"?([^\"#[:space:]]+).*/\1/p" "${LOCKFILE}" | head -n 1
}

usage() {
  cat <<EOF
Usage: scripts/push-llama-cpp-fork.sh [--yes] [--dry-run] [--force-with-lease]

Pushes the submodule HEAD to origin/${BRANCH}, then commits the submodule
pointer bump (and any staged crates/retrograd-ffi/runtime/llama.cpp.lock update) in retrograd and
pushes it.

  --yes               Push without the interactive confirmation.
  --dry-run           Validate everything, but do not push or commit.
  --force-with-lease  Allow a non-fast-forward fork push (after a rebase).
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --yes) ASSUME_YES=true ;;
    --dry-run) DRY_RUN=true ;;
    --force-with-lease) FORCE_WITH_LEASE=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2;;
  esac
done

git -C "${SOURCE_DIR}" rev-parse HEAD >/dev/null 2>&1 || {
  echo "llama.cpp submodule not initialized; run scripts/setup-llama-cpp.sh" >&2
  exit 1
}
git check-ref-format --branch "${BRANCH}" >/dev/null

[[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=no)" ]] || {
  echo "refusing to push a non-clean llama.cpp checkout" >&2
  exit 1
}

upstream_commit="$(lock_value upstream_commit)"
head_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
git -C "${SOURCE_DIR}" merge-base --is-ancestor "${upstream_commit}" "${head_commit}" || {
  echo "fork HEAD ${head_commit} does not descend from lockfile upstream_commit ${upstream_commit}" >&2
  echo "update upstream_commit in ${LOCKFILE#"${ROOT_DIR}/"} after a rebase" >&2
  exit 1
}

git -C "${SOURCE_DIR}" fetch origin "refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}" 2>/dev/null || true

fork_push_needed=true
push_args=()
remote_ref="refs/remotes/origin/${BRANCH}"
if git -C "${SOURCE_DIR}" rev-parse --verify --quiet "${remote_ref}" >/dev/null; then
  remote_commit="$(git -C "${SOURCE_DIR}" rev-parse "${remote_ref}")"
  if [[ "${remote_commit}" == "${head_commit}" ]]; then
    fork_push_needed=false
  elif ! git -C "${SOURCE_DIR}" merge-base --is-ancestor "${remote_ref}" "${head_commit}"; then
    if "${FORCE_WITH_LEASE}"; then
      push_args+=("--force-with-lease=refs/heads/${BRANCH}:${remote_commit}")
    else
      echo "origin/${BRANCH} (${remote_commit}) is not an ancestor of ${head_commit}" >&2
      echo "this is a history rewrite; rerun with --force-with-lease after review" >&2
      exit 1
    fi
  fi
fi

pinned="$(git -C "${ROOT_DIR}" ls-tree HEAD -- "${SUBMODULE_PATH}" | awk '$1 == "160000" && $2 == "commit" { print $3 }')"
bump_needed=true
[[ "${pinned}" == "${head_commit}" ]] && bump_needed=false

if "${DRY_RUN}"; then
  if "${fork_push_needed}"; then
    echo "would push ${head_commit} to origin/${BRANCH}${push_args:+ (with lease)}"
  else
    echo "origin/${BRANCH} already at ${head_commit}; no fork push needed"
  fi
  if "${bump_needed}"; then
    echo "would commit the ${SUBMODULE_PATH} pointer bump ${pinned:-none} -> ${head_commit} and push retrograd"
  else
    echo "retrograd already pins ${head_commit}; nothing to commit"
  fi
  exit 0
fi

if ! "${ASSUME_YES}"; then
  echo "about to push ${head_commit} to origin/${BRANCH} and pin it in retrograd"
  read -r -p "continue? [y/N] " answer
  [[ "${answer}" == "y" || "${answer}" == "Y" ]] || { echo "aborted"; exit 1; }
fi

if "${fork_push_needed}"; then
  git -C "${SOURCE_DIR}" push origin "${push_args[@]+"${push_args[@]}"}" "HEAD:refs/heads/${BRANCH}"
else
  echo "origin/${BRANCH} already at ${head_commit}; skipping fork push"
fi

if "${bump_needed}"; then
  git -C "${ROOT_DIR}" add -- "${SUBMODULE_PATH}" "${LOCKFILE}"
  git -C "${ROOT_DIR}" commit -m "chore: pin llama.cpp fork to ${head_commit}" -- "${SUBMODULE_PATH}" "${LOCKFILE}"
  git -C "${ROOT_DIR}" push
  echo "submodule pinned to ${head_commit} and pushed"
else
  echo "retrograd already pins ${head_commit}; nothing to commit"
fi
