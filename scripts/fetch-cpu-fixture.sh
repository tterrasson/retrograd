#!/usr/bin/env bash
# Fetch the one model used by CPU integration tests. Keep this script
# dependency-free so local runs and CI verify exactly the same bytes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/tests/fixtures/CPU_FIXTURE.toml"
filename="$(sed -n 's/^filename = "\(.*\)"$/\1/p' "$manifest")"
destination="${RETRO_CPU_FIXTURE:-$repo_root/tests/fixtures/$filename}"
url="$(sed -n 's/^url = "\(.*\)"$/\1/p' "$manifest")"
expected_size="$(sed -n 's/^size_bytes = \([0-9]*\)$/\1/p' "$manifest")"
expected_sha="$(sed -n 's/^sha256 = "\([0-9a-f]*\)"$/\1/p' "$manifest")"
stamp="${destination}.verified"

if [[ -z "$filename" || -z "$url" || -z "$expected_size" || -z "$expected_sha" ]]; then
  echo "invalid fixture manifest: $manifest" >&2
  exit 2
fi

verify() {
  [[ -f "$destination" ]] || return 1
  local metadata signature
  metadata="$(stat -c '%s %Y' "$destination" 2>/dev/null || stat -f '%z %m' "$destination")"
  [[ "${metadata%% *}" == "$expected_size" ]] || return 1
  signature="$expected_sha $metadata"
  if [[ -f "$stamp" && "$(cat "$stamp")" == "$signature" ]]; then
    return 0
  fi
  [[ "$(shasum -a 256 "$destination" | awk '{print $1}')" == "$expected_sha" ]] || return 1
  printf '%s\n' "$signature" > "$stamp"
}

if verify; then
  echo "CPU fixture already verified: $destination"
  exit 0
fi

mkdir -p "$(dirname "$destination")"
temporary="${destination}.partial.$$"
trap 'rm -f "$temporary"' EXIT
curl --fail --location --retry 3 --output "$temporary" "$url"
actual_sha="$(shasum -a 256 "$temporary" | awk '{print $1}')"
actual_size="$(stat -c '%s' "$temporary" 2>/dev/null || stat -f '%z' "$temporary")"
if [[ "$actual_size" != "$expected_size" ]]; then
  echo "fixture size mismatch: expected $expected_size, got $actual_size" >&2
  exit 1
fi
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "fixture checksum mismatch: expected $expected_sha, got $actual_sha" >&2
  exit 1
fi
mv "$temporary" "$destination"
signature="$expected_sha $(stat -c '%s %Y' "$destination" 2>/dev/null || stat -f '%z %m' "$destination")"
printf '%s\n' "$signature" > "$stamp"
trap - EXIT
echo "CPU fixture downloaded and verified: $destination"
