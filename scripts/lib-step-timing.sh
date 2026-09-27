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

# Every lane that runs cargo sources this file, so the lanes' build settings
# live here too. Incremental compilation is off: a lane builds a crate under
# several feature sets, and each keeps an incremental directory of its own that
# nothing ever removes - most of what `target/debug` grows by. It only pays on
# a small edit rebuilt in the same configuration, which is the editor's loop and
# not a lane's. `CARGO_INCREMENTAL=1 scripts/test-….sh` restores it.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
