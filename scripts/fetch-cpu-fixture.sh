#!/usr/bin/env bash
# Materialize the declared CPU fixtures and verify their digests. Keep this
# script dependency-free so local runs and CI verify exactly the same bytes.
#
#   scripts/fetch-cpu-fixture.sh                  # every manifest
#   scripts/fetch-cpu-fixture.sh TINY_FIXTURE     # one of them
#
# A manifest declares either a `url` (fetched with curl) or a `generator`
# (a repository-relative script, run with the destination as its only
# argument). Both are verified by digest.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

field() {
  sed -n "s/^$2 = \"\(.*\)\"\$/\1/p" "$1"
}

number() {
  sed -n "s/^$2 = \([0-9]*\)\$/\1/p" "$1"
}

fetch_one() (
  local manifest="$1"
  if [[ ! -f "$manifest" ]]; then
    echo "no such fixture manifest: $manifest" >&2
    exit 2
  fi

  local filename url generator override_env destination expected_size expected_sha stamp
  filename="$(field "$manifest" filename)"
  url="$(field "$manifest" url)"
  generator="$(field "$manifest" generator)"
  override_env="$(field "$manifest" override_env)"
  expected_size="$(number "$manifest" size_bytes)"
  expected_sha="$(field "$manifest" sha256)"

  if [[ -z "$filename" || -z "$expected_size" || -z "$expected_sha" ]]; then
    echo "invalid fixture manifest: $manifest" >&2
    exit 2
  fi
  if [[ -z "$url" && -z "$generator" ]]; then
    echo "fixture manifest declares neither url nor generator: $manifest" >&2
    exit 2
  fi

  destination="$repo_root/tests/fixtures/$filename"
  if [[ -n "$override_env" && -n "${!override_env:-}" ]]; then
    destination="${!override_env}"
  fi
  stamp="${destination}.verified"

  local metadata signature
  if [[ -f "$destination" ]]; then
    metadata="$(stat -c '%s %Y' "$destination" 2>/dev/null || stat -f '%z %m' "$destination")"
    signature="$expected_sha $metadata"
    if [[ "${metadata%% *}" == "$expected_size" ]]; then
      if [[ -f "$stamp" && "$(cat "$stamp")" == "$signature" ]]; then
        echo "fixture already verified: $destination"
        return 0
      fi
      if [[ "$(shasum -a 256 "$destination" | awk '{print $1}')" == "$expected_sha" ]]; then
        printf '%s\n' "$signature" > "$stamp"
        echo "fixture already verified: $destination"
        return 0
      fi
    fi
  fi

  mkdir -p "$(dirname "$destination")"
  local temporary="${destination}.partial.$$"
  trap 'rm -f "$temporary"' EXIT
  if [[ -n "$generator" ]]; then
    "$repo_root/$generator" "$temporary"
  else
    curl --fail --location --retry 3 --output "$temporary" "$url"
  fi

  local actual_sha actual_size
  actual_sha="$(shasum -a 256 "$temporary" | awk '{print $1}')"
  actual_size="$(stat -c '%s' "$temporary" 2>/dev/null || stat -f '%z' "$temporary")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "fixture size mismatch for $filename: expected $expected_size, got $actual_size" >&2
    exit 1
  fi
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "fixture checksum mismatch for $filename: expected $expected_sha, got $actual_sha" >&2
    exit 1
  fi
  mv "$temporary" "$destination"
  signature="$expected_sha $(stat -c '%s %Y' "$destination" 2>/dev/null || stat -f '%z %m' "$destination")"
  printf '%s\n' "$signature" > "$stamp"
  echo "fixture ready and verified: $destination"
)

if [[ $# -gt 0 ]]; then
  for name in "$@"; do
    fetch_one "$repo_root/tests/fixtures/${name%.toml}.toml"
  done
else
  for manifest in "$repo_root"/tests/fixtures/*_FIXTURE.toml; do
    fetch_one "$manifest"
  done
fi
