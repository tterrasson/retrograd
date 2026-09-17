#!/usr/bin/env bash
# RIR promotion lane (ADR-5 section 2): for every (op, backend) pair the
# generated registry declares, run the ggml op matrix twice on the same shapes,
# once native, once RIR - assert the counters, then time both and publish the
# ratio.
#
# This is the lane that decides whether a generated kernel may replace a
# hand-written one. It loads no model and trains nothing: the whole point is
# that a kernel is judged in minutes, in isolation.
#
#   scripts/test-rir.sh                       # every pair, every available GPU
#   scripts/test-rir.sh --backend metal       # one backend
#   scripts/test-rir.sh --op CUMSUM           # one op, while iterating
#   scripts/test-rir.sh --no-perf             # correctness only
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fork="$repo_root/crates/retrograd-ffi/runtime/vendor/llama.cpp"
build="$fork/build-rir"
registry="$repo_root/generated/rir/registry/rir_registry.cpp"
# Claimed-node counts per pair on the matrix, recorded so a *narrowing* is
# visible. See the check in `assert_counters` for why the declared domain alone
# cannot do this (ADR-4 section 6).
baseline="$repo_root/scripts/rir-domain-baseline.tsv"

backends_arg="all"
ops_filter=""
tolerance="1.05"
run_perf=1
force_build=0
run_timeout=300
repeat=3
# Extra passes a *refusal* must survive before the lane calls it a regression
# (ADR-5 section 6). Zero disables the escalation and restores the
# behaviour that produced a refusal from three draws.
escalate=6
update_baseline=0

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --backend metal|vulkan|cuda|all
                               backends to exercise (default: all available)
  --op OP[,OP...]              restrict to these ggml ops (short name)
  --tolerance R                max accepted t_rir/t_native (default 1.05)
  --repeat N                   timing passes per pair; the verdict is the median
                               of the N ratios (default 3)
  --escalate N                 extra passes a refusal must survive before it is
                               reported as a regression (default 6, 0 to disable)
  --update-domain-baseline     record the claimed-node counts instead of
                               checking them (after a deliberate domain change)
  --no-perf                    skip the timing pass
  --timeout S                  wall-clock budget per matrix run (default 300 s)
  --build                      reconfigure/rebuild the fork before running
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backends_arg="$2"; shift 2 ;;
        --op) ops_filter="$2"; shift 2 ;;
        --tolerance) tolerance="$2"; shift 2 ;;
        --repeat) repeat="$2"; shift 2 ;;
        --escalate) escalate="$2"; shift 2 ;;
        --update-domain-baseline) update_baseline=1; shift ;;
        --no-perf) run_perf=0; shift ;;
        --timeout) run_timeout="$2"; shift 2 ;;
        --build) force_build=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

[[ "$repeat" =~ ^[0-9]+$ && "$repeat" -ge 1 ]] || {
    echo "--repeat expects an integer >= 1 (got: $repeat)" >&2; exit 2
}
[[ "$escalate" =~ ^[0-9]+$ ]] || {
    echo "--escalate expects an integer >= 0 (got: $escalate)" >&2; exit 2
}

[[ -f "$registry" ]] || { echo "registry not found: $registry" >&2; exit 1; }
[[ -d "$fork/ggml" ]] || {
    echo "llama.cpp fork missing; run scripts/setup-llama-cpp.sh" >&2
    exit 1
}

# --- MoltenVK: the detected default rather than four variables to remember ----
# `<prefix>/etc/vulkan/icd.d/MoltenVK_icd.json` goes with `<prefix>/lib`, which
# is where the loader lives; both are needed and neither is on the default path.
if [[ "$(uname -s)" == "Darwin" && -z "${VK_ICD_FILENAMES:-}" ]]; then
    for prefix in /opt/homebrew /usr/local; do
        icd="$prefix/etc/vulkan/icd.d/MoltenVK_icd.json"
        if [[ -f "$icd" ]]; then
            export VK_ICD_FILENAMES="$icd"
            export DYLD_LIBRARY_PATH="$prefix/lib:${DYLD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

# --- build --------------------------------------------------------------------
bin="$build/bin/test-backend-ops"
if [[ $force_build -eq 1 || ! -x "$bin" ]]; then
    echo "== build : $build"
    # Metal only where Metal exists: `find_library(Foundation)` is a configure
    # **error** elsewhere, so `-DGGML_METAL=ON` on Linux does not degrade to a
    # skipped backend, it stops the lane before it builds anything.
    metal=OFF
    [[ "$(uname -s)" == "Darwin" ]] && metal=ON
    # CUDA on the same principle as Metal: only where a toolkit exists, because
    # `-DGGML_CUDA=ON` without one is a configure error and not a skipped
    # backend.
    cuda=OFF
    cuda_cmake=()
    cuda_compiler="${CUDACXX:-nvcc}"
    if command -v "$cuda_compiler" >/dev/null 2>&1; then
        cuda=ON
        # One architecture and not every one nvcc knows: a fat binary over the
        # default list costs minutes of `nvcc` per translation unit, and the
        # lane measures the GPU that is here. `RETRO_CUDA_ARCHITECTURES` is the
        # same knob the FFI build reads.
        cuda_cmake=(
            -DCMAKE_CUDA_COMPILER="$cuda_compiler"
            -DCMAKE_CUDA_ARCHITECTURES="${RETRO_CUDA_ARCHITECTURES:-native}"
        )
        # The host compiler `nvcc` accepts is **probed**, never assumed: a
        # toolkit refusing the system `gcc` is a property of this machine, and
        # writing `g++-15` in here would make the lane build on one box and fail
        # to configure everywhere else. Same env override as the Rust side.
        echo 'int main(){return 0;}' > "$build.probe.cu"
        for cc in "${RIR_NVCC_CCBIN:-}" "" g++-15 g++-14 g++-13; do
            args=(-c "$build.probe.cu" -o "$build.probe.o")
            if [[ -n "$cc" ]]; then
                command -v "$cc" >/dev/null 2>&1 || continue
                args=(-ccbin "$cc" "${args[@]}")
            fi
            if "$cuda_compiler" "${args[@]}" >/dev/null 2>&1; then
                if [[ -n "$cc" ]]; then
                    cuda_cmake+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v "$cc")")
                fi
                break
            fi
        done
        rm -f "$build.probe.cu" "$build.probe.o"
    fi
    cmake -B "$build" -S "$fork" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=$metal -DGGML_VULKAN=ON -DGGML_CUDA=$cuda \
        ${cuda_cmake[@]+"${cuda_cmake[@]}"} \
        -DLLAMA_BUILD_TESTS=ON >/dev/null
    cmake --build "$build" --target test-backend-ops -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
fi
[[ -x "$bin" ]] || { echo "test-backend-ops not found after build" >&2; exit 1; }

strip_ansi() { sed -E $'s/\x1b\\[[0-9;]*m//g'; }

# --- pairs, straight from the generated registry ------------------------------
# A pair is a line of `rir_op_policies` on Metal, Vulkan or CUDA whose policy is
# not NATIVE_ONLY. Adding an op to the registry adds it to this lane; there is no
# list of ops in this script.
pairs=()
while IFS= read -r line; do
    pairs+=("$line")
done < <(
    awk -F'[{},]' '
        /RIR_POLICY_/ && /RIR_BACKEND_(METAL|VULKAN|CUDA)/ {
            op=$2; gsub(/[" ]/, "", op); sub(/^GGML_OP_/, "", op)
            be=$3; gsub(/[ ]/, "", be); sub(/^RIR_BACKEND_/, "", be)
            po=$4; gsub(/[ ]/, "", po); sub(/^RIR_POLICY_/, "", po)
            # 5th field of the row: rir_op_policy.native_retired
            re=$6; gsub(/[ }]/, "", re)
            if (po != "NATIVE_ONLY") { print op "\t" tolower(be) "\t" tolower(po) "\t" (re == "1" ? "retired" : "kept") }
        }' "$registry"
)
[[ ${#pairs[@]} -gt 0 ]] || { echo "no dispatchable (op, backend) pair in the registry" >&2; exit 1; }

ggml_backend_name() { case "$1" in metal) echo "MTL0" ;; vulkan) echo "Vulkan0" ;; cuda) echo "CUDA0" ;; esac; }

# The `variant_id`s the registry declares for one (op, backend). A pair may
# publish several lowerings arbitrated per shape; the lane has to know which,
# because "some RIR ran" no longer proves the arbitration works - a broken
# backend hookup that always picks the fallback still yields rir>0, native=0
# and a green self-test.
#
# Read from the `rir_variants` rows: `{"kernel", "GGML_OP_X", "variant_id",
# "entrypoint", variant, RIR_BACKEND_Y, priority, …`.
registry_variants() {
    local op="$1" be="$2"
    # The third field is `ggml_op_variant`, and it is
    # `nullptr` for every op that is not a family - so a `"`-delimited field
    # index would name two different columns depending on the row. It is dropped
    # first, and then `variant_id` is back where it always was.
    sed -E 's/^ *\{("[^"]*", "[^"]*", )(nullptr|"[^"]*"), /\1/' "$registry" \
    | awk -F'"' -v op="GGML_OP_$op" -v be="RIR_BACKEND_$(tr '[:lower:]' '[:upper:]' <<<"$2")" '
        $2 != "" && $4 == op && index($0, be) > 0 { print $6 }' | sort -u
}

# --- which GPUs are actually here --------------------------------------------
probe="$("$bin" -o NOPE_NOT_AN_OP 2>&1 | strip_ansi || true)"
is_available() {
    grep -qE "^Backend [0-9]+/[0-9]+: $(ggml_backend_name "$1")\$" <<<"$probe"
}

case "$backends_arg" in
    all) wanted=(metal vulkan cuda) ;;
    metal|vulkan|cuda) wanted=("$backends_arg") ;;
    *) echo "unknown backend: $backends_arg" >&2; exit 2;;
esac

selected=()
for be in "${wanted[@]}"; do
    if is_available "$be"; then
        selected+=("$be")
    elif [[ "$backends_arg" != "all" ]]; then
        # An explicitly requested device that is not here is a lane failure, not
        # a skip: a green lane must never mean "no GPU".
        echo "backend $be requested but absent from this build/machine" >&2
        exit 1
    else
        echo "-- $be absent, skipped"
    fi
done
[[ ${#selected[@]} -gt 0 ]] || { echo "no GPU backend available" >&2; exit 1; }

# --- the session stamp ---------------------------------------------------------
# A ratio compares two runs of *this* pass, and nothing else: 28% of machine
# drift has been measured between two days on the same native kernel and the
# same shape. The lane therefore names its session, and the
# rule that follows is written in ADR-5: a ratio is cited with its
# stamp, or not at all.
#
# A before/after between two RIR versions is done by running the lane **twice
# in the same session**, and checking that the native column is stable across
# the two passes: that stability is what makes the two RIR columns
# comparable. It's free - the lane already produces both columns - and it is
# the chosen candidate against anchoring each session on a witness shape.
session_host="$(uname -n)"
session_id="${session_host%%.*}-$(date -u +%Y%m%dT%H%M%SZ)"
git_stamp() {
    local dir="$1" rev dirty=""
    rev="$(git -C "$dir" rev-parse --short HEAD 2>/dev/null)" || { echo "not in git"; return; }
    [[ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]] && dirty=" +dirty"
    echo "$rev$dirty"
}
# The *measured* devices, hence `selected` and not every line of the probe:
# on a Metal + Vulkan machine, `--backend vulkan` only measures one of the two,
# and a stamp that cited the other would describe a scope that was not
# exercised. The filters that produced this list are printed alongside it, for
# the same reason: the stamp must be enough to reconstruct the pass.
session_devices=""
for be in "${selected[@]}"; do
    dev="$(grep -E "^Backend [0-9]+/[0-9]+: $(ggml_backend_name "$be")\$" <<<"$probe" \
        | sed -E 's/^Backend [0-9]+\/[0-9]+: //')"
    session_devices+="${session_devices:+, }$be ($dev)"
done
echo "== session $session_id"
echo "    machine  : $(uname -srm)"
echo "    devices  : ${session_devices:-(none)}"
echo "    repo     : $(git_stamp "$repo_root")"
echo "    fork     : $(git_stamp "$fork")"
echo "    filters  : --backend $backends_arg --op ${ops_filter:-(all)}"
echo "    measure  : --repeat $repeat --escalate $escalate --tolerance $tolerance"
echo "    a ratio is only cited with this identifier (ADR-5 §7): native and RIR"
echo "    are measured here, yesterday's measurement does not compare to this one."

# --- one run of the matrix ----------------------------------------------------
# mode=native: RIR off, the hand-written kernel and nothing else
# mode=rir: RIR on; an `observe` pair is promoted for this process only,
#                otherwise it would never encode and nothing could measure it.
# What `-o` must be given to select this pair's cases.
#
# It is `ggml_op_desc()` and not `ggml_op_name()`, and the two differ for
# exactly one op: `GGML_OP_UNARY` describes itself by its *member*
# (`ggml_unary_op_name`), because it is a family of twenty-two functions behind
# one op. Passing "UNARY" selects zero cases and
# `test-backend-ops` reports a green `0/0` - the one failure this lane already
# refuses by name.
#
# The list is the fourteen members RIR writes. The eight it does not are
# deliberately absent: their nodes are refused with `op_variant`, which the
# registry declares, and running them here would only measure the native kernel
# against itself.
matrix_op_filter() {
    case "$1" in
        UNARY) echo "ABS,SGN,NEG,STEP,TANH,ELU,RELU,SIGMOID,GELU_QUICK,SILU,HARDSWISH,HARDSIGMOID,EXP,EXPM1" ;;
        *)     echo "$1";;
    esac
}

run_matrix() {
    local mode="$1" op="$2" be="$3" policy="$4" what="$5" out="$6"
    local -a envs=(RETRO_RIR_STATS=1)
    if [[ "$mode" == "native" ]]; then
        envs+=(RETRO_RIR_MODE=off)
    else
        envs+=(RETRO_RIR_MODE=prefer)
        # The promotion list is keyed by `ggml_op_name()`, i.e. the short name.
        [[ "$policy" == "observe_generated" ]] && envs+=("RETRO_RIR_TEST_PREFER=$op")
    fi
    local -a cmd=("$bin")
    [[ "$what" == "perf" ]] && cmd+=(perf)
    cmd+=(-o "$(matrix_op_filter "$op")" -b "$(ggml_backend_name "$be")")

    # A lane must be bounded, and the thing being measured is precisely a kernel
    # that may be far slower than the one it wants to replace: a sequential scan
    # over the largest perf shape can run for hours. Past the budget the run is
    # killed and reported as such - "too slow to measure" is a verdict, not a
    # reason to hang.
    env "${envs[@]}" "${cmd[@]}" >"$out" 2>&1 &
    local pid=$!
    ( sleep "$run_timeout"; kill -9 "$pid" 2>/dev/null ) >/dev/null 2>&1 &
    local watchdog=$!
    local rc=0
    wait "$pid" || rc=$?
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
    # 137 = SIGKILL, i.e. the watchdog fired.
    [[ $rc -eq 137 ]] && return 2
    [[ $rc -eq 0 ]] || return 1
    return 0
}

# `ggml-rir: site GGML_OP_X/metal seen=.. eligible=.. rir=N native=M [...]`
assert_counters() {
    local op="$1" be="$2" log="$3"
    local line
    line="$(grep -E "^ggml-rir: site GGML_OP_$op/$be " "$log" | tail -1 || true)"
    if [[ -z "$line" ]]; then
        echo "    !! no site line for GGML_OP_$op/$be: RIR saw nothing"
        return 1
    fi
    local seen rir native variants
    seen="$(sed -E 's/.* seen=([0-9]+).*/\1/' <<<"$line")"
    rir="$(sed -E 's/.* rir=([0-9]+).*/\1/' <<<"$line")"
    native="$(sed -E 's/.* native=([0-9]+).*/\1/' <<<"$line")"
    # Which variant(s) actually ran. With one lowering per pair this restated
    # `rir=`; since a pair may publish several arbitrated per shape, it is the
    # only thing that says *what* the matrix below measured - a green ratio
    # attributed to the wrong lowering would be the new faux positif.
    # `[A-Za-z0-9_]`, not `[a-z…]`: a variant id carries the dtype spelling and
    # ggml writes the K quants with a capital (`q4_K`). A lowercase class read
    # the line as three variants where four had run - the assertions below use
    # the declared names and were right, only this display was wrong.
    variants="$(grep -oE '\[[A-Za-z0-9_]+\]=[0-9]+' <<<"$line" | tr '\n' ' ')"
    echo "    counters: seen=$seen rir=$rir native=$native ${variants:-(no named variant)}"
    if grep -q '\[overflow\]=' <<<"$line"; then
        echo "    !! more variants dispatched than the site line can carry"
        return 1
    fi
    if [[ "$rir" -eq 0 ]]; then
        echo "    !! a green matrix with no RIR dispatch is a false positive"
        return 1
    fi

    # Every node that went native must match a declared domain restriction.
    # This replaces a blunter rule - `native = 0` - that was only ever right by
    # accident. It held while the two integrated ops covered their ggml op
    # entirely; `OUT_PROD` is the first whose ggml domain is strictly larger
    # than what a RIR variant claims (a quantized `src0`, or a `src0` broadcast
    # over ne2/ne3, are both legitimate ggml and both outside the DSL). Demanding
    # zero there would forbid integrating any partial-domain op, which from here
    # on is most of them.
    #
    # So the lane asks the sharper question instead: *why* did each fallback
    # happen. A portable-contract reason is the variant declining a shape it
    # never claimed - visible, published in the registry, and answered by the
    # native kernel. A device reason (`missing_feature`, `pipeline`,
    # `device_grid`, `device_alignment`) means the variant we published cannot
    # run on this machine, and `wrong_op` /
    # `policy_native` at a pair the lane just promoted means the selection chain
    # is not wired to this site at all. Those are defects, not domains.
    #
    # And a contract reason is only legitimate if the registry **declared** it.
    # `domain=` on the site line is `rir_op_policy.assumed_domain`, i.e. the
    # parts of the ggml op the kernel published as out of scope, with the reason
    # for each next to the row that carries it. A rejection for a reason outside
    # that declaration is a node the kernel said it would serve and did not:
    # otherwise indistinguishable from an op whose domain had
    # always been partial (ADR-4 section 6).
    local domain
    domain="$(sed -nE 's/.* domain=([a-z|_]+).*/\1/p' <<<"$line")"
    if [[ -z "$domain" ]]; then
        # Not a tolerated absence: without it every contract reject would be
        # excused, which is precisely the state this check replaces.
        echo "    !! the site line does not publish a declared domain - fork too old?"
        return 1
    fi

    local reason n contract=0 broken=0 broken_why="" undeclared=""
    for reason in dtype rank shape stride quant_block integer_range op_variant; do
        n="$(sed -nE "s/.* $reason=([0-9]+).*/\1/p" <<<"$line")"
        contract=$((contract + ${n:-0}))
        if [[ "${n:-0}" -ne 0 && "|$domain|" != *"|$reason|"* ]]; then
            undeclared="$undeclared $reason=$n"
        fi
    done
    for reason in missing_feature pipeline device_grid device_alignment wrong_op policy_native; do
        n="$(sed -nE "s/.* $reason=([0-9]+).*/\1/p" <<<"$line")"
        if [[ "${n:-0}" -ne 0 ]]; then
            broken=$((broken + n))
            broken_why="$broken_why $reason=$n"
        fi
    done
    # Reported before the domain check below, and the order is deliberate: a
    # variant that cannot run here explains everything else on the line, whereas
    # an undeclared contract reject read on top of it would be a second symptom
    # of the same cause.
    if [[ "$broken" -ne 0 ]]; then
        echo "    !! non-contractual fallback:$broken_why - the published variant does not run here"
        return 1
    fi
    if [[ -n "$undeclared" ]]; then
        echo "    !! fallback outside declared domain ($domain):$undeclared"
        echo "       the kernel claims these nodes and does not serve them; declare the"
        echo "       restriction in rir_kernels::integrations, or lift it"
        return 1
    fi
    if [[ "$native" -ne "$contract" ]]; then
        echo "    !! $native native fallbacks for $contract contract rejects: an unexplained fallback"
        return 1
    fi
    # Publish coverage for every pair, not only those that fall back. A pair at
    # 100 % is the interesting case as much as one at 25 %: it is the one whose
    # native kernel can be removed.
    local rejects pct
    rejects="$(grep -oE ' (dtype|rank|shape|stride|quant_block|integer_range|op_variant)=[0-9]+' <<<"$line" \
        | tr -d '\n' | sed 's/^ //')"
    # `s > 0` inside a printf argument would be parsed as a redirection, so the
    # guard is a statement.
    pct="$(awk -v r="$rir" -v s="$seen" 'BEGIN { if (s > 0) printf "%.0f", 100 * r / s; else printf "0" }')"
    if [[ "$native" -ne 0 ]]; then
        echo "    coverage: $rir/$seen nodes claimed (${pct} %), $native out of contract ($rejects)"
    else
        echo "    coverage: $rir/$seen nodes claimed (${pct} %) - whole ggml domain"
    fi

    # --- the count, checked against a recorded baseline -----------------------
    # The declared domain above is a **category**: `dtype` says "some dtypes are
    # out of scope", not *which*. So declaring it for F16 also excuses a kernel
    # that starts refusing an F32 node it used to serve, and declaring `shape`
    # for broadcast excuses any new shape rejection. The mask cannot see a
    # narrowing *inside* a category it already names, and pretending otherwise
    # would be the same over-claim this whole section replaced.
    #
    # What does see it is the count. The matrix is a fixed list of cases, so for
    # a given fork `rir` is a constant per pair; a domain that narrows for any
    # reason, declared or not, makes it drop. That is the check the mask cannot
    # perform, and the two are complementary: the mask names *why* a fallback is
    # legitimate, the baseline notices *how many* stopped being served.
    local ref_line ref_rir ref_seen
    ref_line="$(awk -F'\t' -v p="$op/$be" '$1 == p { print }' "$baseline" 2>/dev/null || true)"
    if [[ $update_baseline -eq 1 ]]; then
        : # rewriting below; comparing against the file being replaced is noise
    elif [[ -z "$ref_line" ]]; then
        # Never a failure: a newly integrated op must be able to enter the lane
        # before anyone has recorded what it covers.
        echo "    -- no domain reference for $op/$be (--update-domain-baseline to record it)"
    else
        IFS=$'\t' read -r _ ref_rir ref_seen <<<"$ref_line"
        if [[ "$seen" -ne "$ref_seen" ]]; then
            # The matrix itself changed - almost always a llama.cpp update
            # adding or removing cases. Comparing counts across two different
            # case lists would fail for a reason that has nothing to do with the
            # kernel, so it is reported and not judged.
            echo "    -- matrix changed for $op/$be ($ref_seen → $seen cases): stale reference, not compared"
        elif [[ "$rir" -lt "$ref_rir" ]]; then
            echo "    !! domain narrowing: $ref_rir → $rir nodes claimed out of $seen"
            echo "       the published reason ($domain) has not changed, coverage has -"
            echo "       this is exactly what a category mask cannot see"
            return 1
        elif [[ "$rir" -gt "$ref_rir" ]]; then
            echo "    -- domain widened: $ref_rir → $rir nodes (--update-domain-baseline to record it)"
        fi
    fi
    new_baseline+=("$op/$be	$rir	$seen")
    cov_rows+=("$op/$be	$rir/$seen	${pct} %	${rejects:--}	$domain")

    # Every variant the registry declares for this pair must have run at least
    # once, and the per-variant counts must add up to `rir`. Printing the
    # distribution is not enough: a backend that resolves the pipeline by kernel
    # name instead of by variant would send every node to the fallback and still
    # pass every check above. This is the assertion that makes the arbitration
    # itself covered on a device, which the synthetic self-test cannot do.
    local total=0 missing=""
    local -a declared=()
    while IFS= read -r v; do [[ -n "$v" ]] && declared+=("$v"); done < <(registry_variants "$op" "$be")
    for v in ${declared[@]+"${declared[@]}"}; do
        local n
        n="$(sed -nE "s/.*\[$v\]=([0-9]+).*/\1/p" <<<"$line")"
        n="${n:-0}"
        total=$((total + n))
        [[ "$n" -eq 0 ]] && missing="$missing $v"
    done
    if [[ -n "$missing" ]]; then
        # Not a "shape not exercised" excuse: the matrix covers both regimes for
        # every pair the registry arbitrates. A variant at zero means the
        # dispatcher never chose it.
        echo "    !! declared variant(s) never dispatched:$missing"
        return 1
    fi
    if [[ "$total" -ne "$rir" ]]; then
        echo "    !! sum of variants ($total) != rir ($rir): a variant outside the registry"
        return 1
    fi
    # The selection rule itself, checked on synthetic tables inside the process
    # under test: priority, policy veto, tie order, and - since a pair may
    # publish several lowerings - that a shape rule removes a variant from the
    # running instead of merely ranking it lower.
    local selftest
    # Unanchored: the matrix writes its progress to stdout without a trailing
    # newline, so this line often begins mid-line after a shape label.
    selftest="$(grep -oE 'ggml-rir: selftest selection=0x[0-9a-f]+' "$log" | tail -1 || true)"
    if [[ -z "$selftest" ]]; then
        echo "    !! no selection self-test in the output"
        return 1
    fi
    if [[ "$selftest" != *"selection=0x0" ]]; then
        echo "    !! $selftest - the selection rule does not hold"
        return 1
    fi
    # The process-wide reject line, for the same reason and with the same
    # distinction: a contract reject is a domain the registry publishes, a
    # feature or pipeline reject is a variant that cannot run here. This catches
    # what the per-site loop above cannot - a reject attributed to no site.
    local aggregate
    aggregate="$(grep -E '^ggml-rir: rejects' "$log" | tail -1 || true)"
    if grep -qE '(missing_feature|pipeline|device_grid|device_alignment|wrong_op|policy_native)=' <<<"$aggregate"; then
        echo "    !! $aggregate"
        return 1
    fi
    return 0
}

# `  OP(args): N runs - X us/run -...`, the label and the time possibly split
# across lines by a backend's own logging.
extract_times() {
    strip_ansi <"$1" | awk '
        /^[[:space:]]*[A-Z0-9_]+\(/ { lab=$0; sub(/\):.*/, ")", lab); gsub(/^[[:space:]]+/, "", lab); pending=lab }
        /us\/run/ {
            if (match($0, /[0-9]+\.[0-9]+ us\/run/)) {
                t=substr($0, RSTART, RLENGTH); sub(/ us\/run/, "", t)
                print pending "\t" t
            }
        }'
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
failures=0
declare -a perf_rows=()
declare -a cov_rows=()
declare -a new_baseline=()

# Median, min, max and count of column `$1` of the samples recorded for shape
# `$2` in file `$3`. One helper for the ratio and for the two columns that
# produced it, so the published µs are always the same draw as the verdict.
col_stats() {
    awk -F'\t' -v l="$2" -v c="$1" '$1 == l { print $c }' "$3" | sort -g | awk '
        { v[NR] = $1 }
        END {
            if (NR == 0) { exit 1 }
            m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
            printf "%s\t%s\t%s\t%d", m, v[1], v[NR], NR
        }'
}

# `$1` timing passes for the pair the loop below is on, appended to its samples
# file. `$2` is only the denominator the progress lines print, so an escalation
# can say "passe 4/9" instead of restarting the count at one.
#
# The pair is read from the loop's variables rather than passed: the two `perf`
# runs, the two extracted time files and the samples file are all named after
# the same `$tag`, and threading five arguments through would not make any of
# them independent.
timing_passes() {
    local n_passes="$1" of="$2" i rc
    for ((i = of - n_passes + 1; i <= of; i++)); do
        if ! run_matrix native "$op" "$be" "$policy" perf "$tmp/$tag.native.perf"; then
            echo "    !! native timing interrupted or failed (pass $i/$of)"
            failures=$((failures + 1))
        fi
        rc=0
        run_matrix rir "$op" "$be" "$policy" perf "$tmp/$tag.rir.perf" || rc=$?
        if [[ $rc -ne 0 ]]; then
            if [[ $rc -eq 2 ]]; then
                # Over budget is a measurement in itself: the generated kernel is
                # at least an order of magnitude off, which is a `chantier` for an
                # observed pair and a regression for a promoted one.
                echo "    !! RIR timing interrupted after ${run_timeout}s (pass $i/$of)"
                [[ "$policy" == "prefer_generated" ]] && failures=$((failures + 1))
            else
                echo "    !! RIR timing failed (pass $i/$of)"
                failures=$((failures + 1))
            fi
        fi
        extract_times "$tmp/$tag.native.perf" >"$tmp/$tag.native.times"
        extract_times "$tmp/$tag.rir.perf" >"$tmp/$tag.rir.times"
        # `label \t native \t rir \t ratio`, one line per shape and per pass.
        awk -F'\t' -v natfile="$tmp/$tag.native.times" '
            BEGIN { while ((getline l < natfile) > 0) { split(l, f, "\t"); nat[f[1]] = f[2] } }
            ($1 in nat) && nat[$1] > 0 { printf "%s\t%s\t%s\t%.4f\n", $1, nat[$1], $2, $2 / nat[$1] }
        ' "$tmp/$tag.rir.times" >>"$tmp/$tag.samples"
    done
}

# Whether any shape of this pair is a *refusal the passes disagree about*: the
# median is over the tolerance while at least one pass came in under it. That is
# the signature of the 5-40 µs class, not of a slow kernel. It is the only state
# where more passes can change the answer. A median over the tolerance whose
# *best* pass is still over it is a regression on every draw; escalating it
# would only cost minutes (ADR-5 section 6).
has_undecided_refusal() {
    local samples="$1" label stats ratio rmin
    while IFS= read -r label; do
        stats="$(col_stats 4 "$label" "$samples")" || continue
        IFS=$'\t' read -r ratio rmin _ _ <<<"$stats"
        awk -v r="$ratio" -v t="$tolerance" 'BEGIN { exit !(r > t) }' || continue
        awk -v r="$rmin" -v t="$tolerance" 'BEGIN { exit !(r <= t) }' && return 0
    done < <(awk -F'\t' '!seen[$1]++ { print $1 }' "$samples")
    return 1
}

for pair in ${pairs[@]+"${pairs[@]}"}; do
    IFS=$'\t' read -r op be policy native <<<"$pair"
    [[ " ${selected[*]} " == *" $be "* ]] || continue
    if [[ -n "$ops_filter" && ",$ops_filter," != *",$op,"* ]]; then continue; fi

    if [[ "$native" == "retired" ]]; then
        echo "== $op / $be (policy: $policy, native retired)"
    else
        echo "== $op / $be (policy: $policy)"
    fi
    tag="$op-$be"

    # The native half of the lane, where there still is one.
    #
    # For a pair whose native kernel has been removed (ADR-5 section 5)
    # this run cannot be made to mean anything, and the failure mode is worse
    # than useless: under `RETRO_RIR_MODE=off`, `ggml_rir_supports_op` answers
    # false, the backend declines every case, and `test-backend-ops` prints
    # `0/0 tests passed` followed by a green `OK`. The lane would publish
    # "native matrix: OK" for a matrix that ran nothing. So the run is skipped
    # and *named*, rather than kept for the shape of the output.
    #
    # What is lost with it is the differential - and that is the honest price of
    # a removal, not a gap to paper over. Correctness is unaffected: the RIR
    # matrix below still compares the generated kernel against the CPU
    # reference, case by case, which is what `test-backend-ops` does.
    if [[ "$native" == "retired" ]]; then
        echo "    native matrix: - (native retired, no witness left)"
    elif run_matrix native "$op" "$be" "$policy" eval "$tmp/$tag.native.log"; then
        if grep -qE '^ *0/0 tests passed' "$tmp/$tag.native.log"; then
            # A matrix that ran zero cases is not a green matrix. It is what a
            # backend prints when it declines every shape, and reading it as a
            # pass is the one way this lane could certify a kernel nothing
            # exercised.
            echo "    !! green native matrix on 0 cases - the backend refuses everything"
            failures=$((failures + 1)); continue
        fi
        echo "    native matrix: OK"
    else
        echo "    !! native matrix failed - see $tmp/$tag.native.log"
        tail -20 "$tmp/$tag.native.log"; failures=$((failures + 1)); continue
    fi

    if run_matrix rir "$op" "$be" "$policy" eval "$tmp/$tag.rir.log"; then
        if grep -qE '^ *0/0 tests passed' "$tmp/$tag.rir.log"; then
            echo "    !! green RIR matrix on 0 cases - the backend refuses everything"
            failures=$((failures + 1)); continue
        fi
        echo "    RIR matrix: OK"
    else
        echo "    !! RIR matrix failed - see $tmp/$tag.rir.log"
        tail -20 "$tmp/$tag.rir.log"; failures=$((failures + 1)); continue
    fi

    assert_counters "$op" "$be" "$tmp/$tag.rir.log" || failures=$((failures + 1))

    [[ $run_perf -eq 1 ]] || continue
    if [[ "$native" == "retired" ]]; then
        # No second column to divide by. The ratios this pair was promoted on
        # were logged when the native kernel was retired, and reproducing them
        # means checking out the commit that removed it - which is what the
        # fork point exists for.
        echo "    -- no differential timing: native is retired"
        continue
    fi

    # --- timing, N times -------------------------------------------------------
    # One run is a draw, not a measurement. Between 5 and 40 µs per dispatch the
    # spread between two runs of the same binary reaches 12 % - more than twice
    # the 5 % tolerance - so a pair at parity would be promoted or refused at
    # random, and that is exactly the class the interesting kernels are in.
    # Each pass re-runs both paths back to back, so a
    # machine-wide slowdown moves the two columns together and cancels in the
    # ratio; the verdict is then the *median* of the N ratios, and min/max are
    # published next to it so a wide spread is visible rather than averaged away.
    : >"$tmp/$tag.samples"
    timing_passes "$repeat" "$repeat"

    if [[ ! -s "$tmp/$tag.samples" ]]; then
        echo "    -- no perf shape for this op (nothing to time)"
        continue
    fi

    # A refusal the passes disagree about is not a verdict yet, so the lane pays
    # for more passes rather than publishing a coin flip. It escalates the whole
    # pair, not the one shape: the two paths are timed back to back in a single
    # matrix run, and re-running one shape would lose exactly the cancellation
    # that makes the ratio robust.
    n_passes="$repeat"
    if [[ $escalate -gt 0 && "$policy" == "prefer_generated" ]] \
        && has_undecided_refusal "$tmp/$tag.samples"; then
        n_passes=$((repeat + escalate))
        echo "    -- undecided refusal over $repeat pass(es): escalating to $n_passes"
        timing_passes "$escalate" "$n_passes"
    fi

    # The ratio is a verdict for a `prefer` pair - it claims to replace the
    # native kernel - and an observation for an `observe` pair, which is by
    # definition an open performance item.
    while IFS= read -r label; do
        # median of the ratios, and the medians of the two columns that produced
        # them - the displayed µs are then the same draw the verdict is made on.
        stats="$(col_stats 4 "$label" "$tmp/$tag.samples")" || continue
        IFS=$'\t' read -r ratio rmin rmax n <<<"$stats"
        ratio="$(printf '%.2f' "$ratio")"
        rmin="$(printf '%.2f' "$rmin")"
        rmax="$(printf '%.2f' "$rmax")"
        nat_us="$(printf '%.2f' "$(col_stats 2 "$label" "$tmp/$tag.samples" | cut -f1)")"
        rir_us="$(printf '%.2f' "$(col_stats 3 "$label" "$tmp/$tag.samples" | cut -f1)")"
        verdict="ok"
        if awk -v r="$ratio" -v t="$tolerance" 'BEGIN { exit !(r > t) }'; then
            if [[ "$policy" == "prefer_generated" ]]; then
                verdict="REGRESSION"; failures=$((failures + 1))
            else
                verdict="work item"
            fi
        elif [[ "$n" -gt 1 ]] && awk -v lo="$rmin" -v hi="$rmax" -v t="$tolerance" \
            'BEGIN { exit !(hi > t || (lo <= 1.0 && hi >= 1.0)) }'; then
            # Two states are undecidable. Either one pass crossed the tolerance while
            # the median held,
            # or the passes bracket 1,00 - the shape's own spread covers the
            # whole distance to the native kernel, so the lane measured its
            # floor and not a difference. Neither is a win to claim nor a
            # regression to fail on; both have to be *named*, because an `ok`
            # printed on a ratio the passes do not support is the faux positif
            # this lane exists to prevent.
            verdict="parity"
        fi
        # Braces around the two bounds: the separator is an en dash, and bash 3.2
        # reads its bytes as part of the name of the variable before it.
        perf_rows+=("$op/$be	$label	$nat_us	$rir_us	$ratio	${rmin}–${rmax}	$n	$verdict")
    done < <(awk -F'\t' '!seen[$1]++ { print $1 }' "$tmp/$tag.samples")
done

if [[ ${#cov_rows[@]} -gt 0 ]]; then
    echo
    # The domain axis, next to the performance one. A pair can be green on every
    # shape and still be unfit for the removal of its native kernel, because the
    # native remains the only path for what the kernel never claimed. The two
    # tables answer different questions and neither substitutes for the other.
    echo "== claimed domain (matrix nodes; \u201cdeclared\u201d = restrictions published by the registry)"
    { printf 'pair\tclaimed\trate\tout of contract\tdeclared\n'
      printf '%s\n' "${cov_rows[@]}"; } | column -t -s$'\t'
fi

# Rewriting the baseline only ever *adds to* or *replaces* the pairs this run
# actually measured: a `--op` or `--backend` filter must not silently delete the
# reference of everything it did not look at.
if [[ $update_baseline -eq 1 && ${#new_baseline[@]} -gt 0 ]]; then
    merged="$tmp/baseline.tsv"
    : >"$merged"
    if [[ -f "$baseline" ]]; then
        while IFS= read -r old; do
            [[ -z "$old" || "$old" == \#* ]] && continue
            keep=1
            for row in "${new_baseline[@]}"; do
                [[ "${old%%$'\t'*}" == "${row%%$'\t'*}" ]] && { keep=0; break; }
            done
            [[ $keep -eq 1 ]] && printf '%s\n' "$old" >>"$merged"
        done <"$baseline"
    fi
    printf '%s\n' "${new_baseline[@]}" >>"$merged"
    { printf '# test-backend-ops matrix nodes claimed per pair, recorded\n'
      printf '# to make visible a domain *narrowing* that the registry category\n'
      printf '# mask cannot see.\n'
      printf '# Regenerate with: scripts/test-rir.sh --no-perf --update-domain-baseline\n'
      printf '# pair\tclaimed\tseen\n'
      sort "$merged"; } >"$baseline"
    echo
    echo "domain reference written: ${baseline#"$repo_root/"} (${#new_baseline[@]} pair(s))"
fi

if [[ ${#perf_rows[@]} -gt 0 ]]; then
    echo
    echo "== performance - session $session_id (median µs/run, tolerance" \
         "$tolerance ; \u201cparity\u201d = the gap fits within the passes' spread)"
    echo "   both columns come from this session; a ratio cited without it"
    echo "   means nothing (ADR-5 §7)"
    { printf 'pair\tshape\tnative\tRIR\tratio\tmin–max\tpasses\tverdict\n'
      printf '%s\n' "${perf_rows[@]}"; } | column -t -s$'\t'
fi

echo
if [[ $failures -eq 0 ]]; then
    echo "test-rir: OK"
else
    echo "test-rir: $failures failure(s)"
fi
exit $((failures == 0 ? 0 : 1))
