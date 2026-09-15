#!/usr/bin/env bash
set -euo pipefail

# Verifies the llama.cpp fork submodule: initialized, clean, at the commit
# pinned by this repository, on the expected fork remote, and descending from
# the upstream base declared in crates/retrograd-ffi/runtime/llama.cpp.lock.
#
# It also verifies that every patch family in [upstream_status] declares where
# it is going: the goal is not "no patch left", which would aim wrong, but no
# family without a written disposition.
#
# With --upstream-status it additionally runs each family's `probe` against the
# upstream base: a symbol the fork introduces and upstream now has too is a
# family that has become redundant, and a patch that duplicates upstream is
# free to delete. That check belongs to a rebase, so scripts/update-llama-cpp.sh
# runs it there; it is off by default because it greps the whole tree twice per
# family.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_NAME="runtime/vendor/llama.cpp"
SUBMODULE_PATH="crates/retrograd-ffi/runtime/vendor/llama.cpp"
SOURCE_DIR="${LLAMA_CPP_DIR:-${ROOT_DIR}/${SUBMODULE_PATH}}"
LOCKFILE="${LLAMA_CPP_LOCKFILE:-${ROOT_DIR}/crates/retrograd-ffi/runtime/llama.cpp.lock}"

RUN_PROBES=0
for arg in "$@"; do
  case "${arg}" in
    --upstream-status) RUN_PROBES=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2;;
  esac
done

lock_value() {
  local key="$1"
  sed -nE "s/^${key}[[:space:]]*=[[:space:]]*\"?([^\"#[:space:]]+).*/\1/p" "${LOCKFILE}" | head -n 1
}

# The [upstream_status] section: one inline table per family, on one line.
status_entries() {
  sed -nE '/^\[upstream_status\]/,/^\[/p' "${LOCKFILE}" | grep -E '^[a-z0-9_]+ = \{'
}

# Reads one key out of an inline table. Returns empty when the key is absent,
# which is what the format check below distinguishes from an empty value.
entry_field() {
  local line="$1" key="$2"
  printf '%s' "${line}" | sed -nE "s/.*[{,][[:space:]]*${key} = \"([^\"]*)\".*/\1/p"
}

entry_has() {
  local line="$1" key="$2"
  printf '%s' "${line}" | grep -qE "[{,][[:space:]]*${key} = \""
}

check_upstream_status() {
  local failed=0 n=0
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    n=$((n + 1))
    local family status
    family="${line%% =*}"
    status="$(entry_field "${line}" status)"
    case "${status}" in
      upstream)
        entry_has "${line}" pr || {
          echo "upstream_status: ${family} is upstream but names no PR" >&2
          failed=1
        }
        ;;
      proposable)
        case "$(entry_field "${line}" odds)" in
          high|medium|low) ;;
          *)
            echo "upstream_status: ${family} is proposable without odds = high|medium|low" >&2
            failed=1
            ;;
        esac
        ;;
      retrograd) ;;
      *)
        echo "upstream_status: ${family} has no status = upstream|proposable|retrograd" >&2
        failed=1
        ;;
    esac
    # `probe` may be empty - a family that fixes behaviour in existing code
    # introduces no symbol - but it may not be missing: absent and empty are
    # different claims, and only one of them has been thought about.
    entry_has "${line}" probe || {
      echo "upstream_status: ${family} declares no probe (use probe = \"\" when it has no symbol)" >&2
      failed=1
    }
    [[ -n "$(entry_field "${line}" why)" ]] || {
      echo "upstream_status: ${family} has no reason written" >&2
      failed=1
    }

    if [[ "${RUN_PROBES}" == "1" ]]; then
      local probe
      probe="$(entry_field "${line}" probe)"
      [[ -n "${probe}" ]] || continue
      git -C "${SOURCE_DIR}" grep -q "${probe}" HEAD -- ggml src include || {
        echo "upstream_status: ${family} probe '${probe}' is not in the fork any more" >&2
        failed=1
        continue
      }
      if git -C "${SOURCE_DIR}" grep -q "${probe}" "${upstream_commit}" -- ggml src include; then
        echo "upstream_status: ${family} probe '${probe}' now exists upstream - the family may be redundant, delete the patch rather than rebasing it" >&2
        failed=1
      fi
    fi
  done < <(status_entries)

  [[ "${n}" -gt 0 ]] || {
    echo "invalid lockfile ${LOCKFILE}: no [upstream_status] entry" >&2
    return 1
  }
  return "${failed}"
}

git -C "${SOURCE_DIR}" rev-parse HEAD >/dev/null 2>&1 || {
  echo "llama.cpp submodule not initialized at ${SOURCE_DIR}; run scripts/setup-llama-cpp.sh" >&2
  exit 1
}

upstream_commit="$(lock_value upstream_commit)"
[[ -n "${upstream_commit}" ]] || {
  echo "invalid lockfile ${LOCKFILE}: missing upstream_commit" >&2
  exit 1
}

fork_url="$(git -C "${ROOT_DIR}" config -f .gitmodules "submodule.${SUBMODULE_NAME}.url")"
actual="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
origin="$(git -C "${SOURCE_DIR}" remote get-url origin)"
pinned="$(git -C "${ROOT_DIR}" ls-files -s -- "${SUBMODULE_PATH}" | awk '$1 == "160000" { print $2 }')"

[[ -n "${pinned}" ]] || {
  echo "no submodule gitlink for ${SUBMODULE_PATH} in the index" >&2
  exit 1
}
[[ "${actual}" == "${pinned}" ]] || {
  echo "llama.cpp is at ${actual}, expected pinned submodule commit ${pinned}" >&2
  echo "commit the submodule bump in retrograd, or run scripts/setup-llama-cpp.sh --pin" >&2
  exit 1
}
[[ "${origin}" == "${fork_url}" ]] || {
  echo "llama.cpp origin is ${origin}, expected ${fork_url}" >&2
  exit 1
}
[[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=no)" ]] || {
  echo "llama.cpp submodule has local modifications" >&2
  exit 1
}
git -C "${SOURCE_DIR}" merge-base --is-ancestor "${upstream_commit}" HEAD || {
  echo "fork commit does not descend from declared upstream commit ${upstream_commit}" >&2
  exit 1
}

check_upstream_status || {
  echo "every patch family must declare where it is going" >&2
  exit 1
}

echo "llama.cpp fork submodule is clean and pinned: ${actual} (upstream ${upstream_commit})"
