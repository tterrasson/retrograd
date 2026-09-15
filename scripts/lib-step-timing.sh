# Shared helper: run a command, print how long it took as its own line so
# compilation and execution show up separately in lane logs instead of one
# opaque total. Source this file; do not execute it directly.
#
# Usage: timed_step <label> <command> [args...]
timed_step() {
  local label="$1"
  shift
  local start end
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  echo "step=${label} duration=$((end - start))s"
}
