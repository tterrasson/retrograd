#!/usr/bin/env bash
# Record, or check, the golden decision fixtures under tests/fixtures/decision/.
#
#   scripts/decision-golden.sh            # record golden-<model>.<backend>.json
#   scripts/decision-golden.sh --check    # recompute and compare, write nothing
#
# The reference is the decision server of the vendored fork: the workspace
# build turns the fork's `tools` off, so this script builds `llama-server` and
# the capture tool (scripts/decision-golden/) out of the workspace tree, in
# target/decision-golden (RETRO_DECISION_GOLDEN_BUILD_DIR to move it). The
# capture tool drives the server's own `server_decision_context` - parsing,
# rendering, layout, answer - and records what the server does not expose: the
# prompt of every pass, its tokens, where the scores are read and the raw
# scores. Each CPU capture is then replayed against `llama-server` on the CPU,
# which must give the same answers to 1e-6 (and the same message for a refused
# request): that is what makes the fixture the server's numbers and not the
# tool's.
#
# Models. Always: the small Apache-2.0 laya test model (fetched by
# scripts/fetch-cpu-fixture.sh decision/TINYLAYA_FIXTURE), the generated tiny
# laya model (TINY_DECISION_FIXTURE) and one generated tiny model per readout
# family (scripts/gen-tiny-decision-fixture.py --type NAME), each with the
# request file of its family. The real ones are opt-in, their files are never
# committed:
#
#   RETRO_DECISION_JULIA=/path/Julia-1-Q8_0.gguf
#   RETRO_DECISION_LAYA=/path/Laya-Q8_0.gguf
#   RETRO_DECISION_KEV=/path/Kev-4B-Q4_K_M.gguf
#
# Backends. A fixture is valid for the backend it was recorded on: the same
# weights give answers that differ by up to 1e-1 between the CPU and Metal,
# and the CPU kernels differ between architectures (see
# tests/fixtures/decision/README.md). A CPU fixture is named after the host
# architecture (`cpu-x86_64`, `cpu-aarch64`); the script records on the CPU,
# and on the GPU when there is one.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${RETRO_DECISION_GOLDEN_BUILD_DIR:-$repo_root/target/decision-golden}"
fixtures="$repo_root/tests/fixtures/decision"
compare="$repo_root/scripts/decision-golden/compare.py"
generator="$repo_root/scripts/gen-tiny-decision-fixture.py"

check=0
case "${1:-}" in
  --check) check=1 ;;
  "") ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

case "$(uname -m)" in
  arm64|aarch64) cpu_arch=aarch64 ;;
  *) cpu_arch="$(uname -m)" ;;
esac

cmake -S "$repo_root/scripts/decision-golden" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=OFF >/dev/null 2>&1
cmake --build "$build_dir" --target decision-golden llama-server -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >/dev/null
tool="$build_dir/bin/decision-golden"
server_bin="$build_dir/bin/llama-server"

"$repo_root/scripts/fetch-cpu-fixture.sh" decision/TINYLAYA_FIXTURE >/dev/null
"$repo_root/scripts/fetch-cpu-fixture.sh" TINY_DECISION_FIXTURE >/dev/null

# name=model=requests
models=(
  "tinylaya=${RETRO_TINYLAYA_FIXTURE:-$fixtures/tinylaya-for-testing-Q8_0.gguf}=requests.json"
  "tiny-laya=${RETRO_TINY_DECISION_FIXTURE:-$repo_root/tests/fixtures/retrograd-tiny-decision-f32.gguf}=requests.json"
)
# The generated fixture of each readout family and the requests of that family.
generated=(
  "openjev:requests-labels.json"
  "lev:requests-labels.json"
  "nimble:requests-labels.json"
  "vllm-d3:requests-labels.json"
  "lfm2-d1:requests-labels.json"
  "lfm3-d1:requests-labels.json"
  "h2o-lightning:requests-h2o.json"
  "vllm-decision2:requests-joint.json"
  "clef:requests-joint.json"
  "kev:requests-kev.json"
  "lfm2-d1-omni:requests-d1omni.json"
)
mkdir -p "$build_dir/fixtures"
for entry in "${generated[@]}"; do
  type="${entry%%:*}"
  path="$build_dir/fixtures/retrograd-tiny-decision-$type-f32.gguf"
  "$generator" --type "$type" "$path"
  models+=("tiny-$type=$path=${entry#*:}")
done
[[ -n "${RETRO_DECISION_JULIA:-}" ]] && models+=("julia-1=$RETRO_DECISION_JULIA=requests.json")
[[ -n "${RETRO_DECISION_LAYA:-}" ]] && models+=("laya=$RETRO_DECISION_LAYA=requests.json")
[[ -n "${RETRO_DECISION_KEV:-}" ]] && models+=("kev=$RETRO_DECISION_KEV=requests-kev.json")

scratch="$(mktemp -d)"
server_pid=""
cleanup() {
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true
  rm -rf "$scratch"
}
trap cleanup EXIT

# Starts llama-server on the CPU for one model and waits for it.
start_server() {
  local model="$1" port=18931
  mkdir -p "$scratch/slots"
  "$server_bin" -m "$model" --host 127.0.0.1 --port "$port" -c 8192 -b 8192 -ub 8192 -np 1 \
    -dev none -ngl 0 --no-webui --slot-save-path "$scratch/slots" >"$scratch/server.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      server_url="http://127.0.0.1:$port"
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$scratch/server.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "llama-server did not come up" >&2
  return 1
}

stop_server() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}

status=0
for entry in "${models[@]}"; do
  name="${entry%%=*}"
  rest="${entry#*=}"
  model="${rest%=*}"
  requests="$fixtures/${rest##*=}"
  if [[ ! -f "$model" ]]; then
    echo "no such model for $name: $model" >&2
    exit 2
  fi
  sha="$(shasum -a 256 "$model" | awk '{print $1}')"

  for device in cpu gpu; do
    out="$scratch/$name.$device.json"
    if ! "$tool" --model "$model" --model-sha256 "$sha" --requests "$requests" --out "$out" --device "$device" 2>"$scratch/tool.log"; then
      if [[ "$device" == gpu ]] && grep -q "no GPU device" "$scratch/tool.log"; then
        continue
      fi
      cat "$scratch/tool.log" >&2
      exit 1
    fi
    backend="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["backend"])' "$out")"
    if [[ "$backend" == cpu ]]; then
      backend="cpu-$cpu_arch"
    fi
    golden="$fixtures/golden-$name.$backend.json"

    if [[ "$device" == cpu ]]; then
      start_server "$model"
      if ! python3 "$compare" server "$out" "$requests" "$server_url"; then
        stop_server
        exit 1
      fi
      stop_server
    fi

    if [[ "$check" == 1 ]]; then
      if [[ ! -f "$golden" ]]; then
        echo "missing $golden" >&2
        status=1
        continue
      fi
      python3 "$compare" golden "$golden" "$out" || status=1
    else
      cp "$out" "$golden"
      echo "recorded $golden"
    fi
  done
done
exit "$status"
